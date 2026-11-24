//! The forward pass: one `step` runs a batch of token rows through all layers on every rank.
//!
//! A row is one token position of one sequence. Rows of the same sequence are consecutive and
//! form a group. A step either commits its rows (prefill) or leaves them pending (decode /
//! speculative verify): pending rows only become part of the recurrent state when the caller
//! `accept`s a prefix of them, and the next step replays that prefix into the state before
//! doing its own work. Rejected rows cost nothing to roll back.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::time::Instant;

use crate::hip::{self, BMAX, DevBuf, Groups, Stream};
use crate::model::{Dims, HParams, LayerW, Mat, Mixer, Shard, Weights};

/// Per GDN layer, per rank: conv history / raw rows, conv output rows, recurrent state.
struct GdnBufs {
    mix: DevBuf<f32>,
    qkv: DevBuf<f32>,
    state: DevBuf<f32>,
}

struct KvBufs {
    k: DevBuf<u16>,
    v: DevBuf<u16>,
}

pub struct Rank {
    pub dev: usize,
    pub s: Stream,
    /// Side stream for bulk card-to-card copies, and events marking, per half of a split step,
    /// "stage computed" on the main stream and "partial delivered" on the side stream.
    sx: Stream,
    ev_done: [hip::Event; 2],
    ev_copied: [hip::Event; 2],
    pub dims: Dims,
    pub w: Weights,
    tokens: DevBuf<i32>,
    /// Residual stream of the block being run (trunk or MTP).
    h: DevBuf<f32>,
    xq: DevBuf<i8>,
    xd: DevBuf<f32>,
    /// Final-norm hidden rows of the last head pass (trunk: h_nextn; MTP: its own h).
    xf: DevBuf<f32>,
    y: DevBuf<f32>,
    /// Partial sublayer outputs: own (pa, pb) and the peer's copies.
    pa: DevBuf<f32>,
    pb: DevBuf<f32>,
    peer_a: DevBuf<f32>,
    peer_b: DevBuf<f32>,
    logits: DevBuf<f32>,
    best: DevBuf<i32>,
    best_peer: DevBuf<i32>,
    /// Draft tokens of the current chain, [step][row].
    drafted: DevBuf<i32>,
    /// Count of exchange timeouts; non-zero means this rank's results are garbage.
    err: DevBuf<i32>,
    qr: DevBuf<u16>,
    /// Attention segment partials, [row tile][head][segment].
    part: DevBuf<f32>,
    /// GDN scratch for large committed batches (see kernels.cpp).
    mix_s: DevBuf<f32>,
    qkv_s: DevBuf<f32>,
    gdn: Vec<Option<GdnBufs>>,
    kv: Vec<Option<KvBufs>>,
    /// MTP: its KV cache, token embeddings, per-row h input, and each slot's last accepted h.
    mtp_kv: Option<KvBufs>,
    emb: DevBuf<f32>,
    hin: DevBuf<f32>,
    hprev: DevBuf<f32>,
}

#[derive(Clone, Copy, Default)]
struct Seq {
    /// Tokens accepted so far = position of the next row.
    len: usize,
    cur: i32,
    last_rows: usize,
    last_row0: usize,
    last_acc: usize,
    last_committed: bool,
    awaiting_accept: bool,
}

#[derive(Clone, Copy)]
pub struct RowOut {
    pub token: u32,
    pub logit: f32,
}

pub struct Engine {
    pub hp: HParams,
    pub ranks: Vec<Rank>,
    seqs: Vec<Seq>,
    cap: usize,
    /// Largest batch a step may carry.
    rows: usize,
    /// When set, every layer's output for the last row is printed, for comparison against a reference implementation.
    pub dump: bool,
    /// When set, every launch is followed by a sync and its time is charged to a bucket.
    /// Serialises the ranks, so totals overstate a real step; use it for proportions.
    pub prof: Option<RefCell<Prof>>,
    xchg: Option<Xchg>,
}

/// Shared host memory for the cross-card exchange: [rank][parity] data areas, then per-rank flags.
struct Xchg {
    mem: hip::Pinned,
    seq: std::cell::Cell<i64>,
}

/// Flag words per rank (one per exchange workgroup, padded).
const XCHG_FLAGS: usize = 32;

/// Largest batch whose partials are traded by the on-card exchange kernel. Every step pays one
/// full pass over the weights, so a 40-token prompt should be one step, not three. Above this
/// size the exchange (PCIe-bound, ~2 µs per row) is worth hiding behind compute, and the batch
/// runs as two interleaved halves instead. Measured: a 262-token prompt takes 201 ms split
/// above 256 rows, 185 ms above 128, 219 ms above 64.
const XCHG_ROWS: usize = 128;

pub struct Prof {
    last: Instant,
    pub buckets: BTreeMap<&'static str, (f64, u64)>,
}

impl Prof {
    pub fn new() -> Self {
        Self { last: Instant::now(), buckets: BTreeMap::new() }
    }

    pub fn report(&self) {
        let total: f64 = self.buckets.values().map(|v| v.0).sum();
        for (k, (t, n)) in &self.buckets {
            eprintln!("  {k:<12} {:8.2} ms total  {:5.1}%  {n:6} launches  {:7.1} us each", t * 1e3, 100.0 * t / total, t * 1e6 / *n as f64);
        }
        eprintln!("  {:<12} {:8.2} ms", "total", total * 1e3);
    }
}

const R: usize = BMAX;

impl Rank {
    /// `rows` is the largest batch a step may carry (the prefill chunk).
    pub fn new(dev: usize, s: Stream, hp: &HParams, sh: Shard, w: Weights, n_slots: usize, cap: usize, rows: usize) -> Self {
        let dm = sh.dims(hp);
        let xmax = (2 * dm.d).max(dm.ff).max(dm.gv * hp.gs).max(dm.hq * hp.hd);
        let ymax = (2 * dm.ff).max(dm.gdn_nin).max(dm.attn_nin);
        // Zeroed: attention reads whole 16-slot tiles and masks the slots a row may not see, so
        // unwritten slots must hold finite values.
        let new_kv = || KvBufs { k: DevBuf::zeroed(cap * dm.hkv * hp.hd, s), v: DevBuf::zeroed(cap * dm.hkv * hp.hd, s) };
        let (mut gdn, mut kv) = (Vec::new(), Vec::new());
        for l in 0..hp.n_layer {
            if hp.is_attn(l) {
                gdn.push(None);
                kv.push(Some(new_kv()));
            } else {
                kv.push(None);
                gdn.push(Some(GdnBufs {
                    mix: DevBuf::zeroed(n_slots * 2 * (3 + R) * dm.ch, s),
                    qkv: DevBuf::zeroed(n_slots * 2 * R * (dm.ch + 2 * dm.gv), s),
                    state: DevBuf::zeroed(n_slots * dm.gv * hp.gs * hp.gs, s),
                }));
            }
        }
        // The batched kernels read activations in whole tiles of up to 128 rows.
        let padded = rows.next_multiple_of(128);
        let f = |n: usize| DevBuf::<f32>::new(rows * n);
        Self {
            dev,
            s,
            sx: hip::stream_create(),
            ev_done: [hip::event_create(), hip::event_create()],
            ev_copied: [hip::event_create(), hip::event_create()],
            dims: dm,
            mtp_kv: w.mtp.as_ref().map(|_| new_kv()),
            w,
            tokens: DevBuf::new(rows),
            h: f(dm.d),
            // +16 rows: the last-row head of a full batch reads one tile past the end.
            xq: DevBuf::zeroed((padded + 16) * xmax, s),
            xd: DevBuf::zeroed((padded + 16) * xmax / 32, s),
            xf: f(dm.d),
            y: f(ymax),
            pa: f(dm.d),
            pb: f(dm.d),
            peer_a: f(dm.d),
            peer_b: f(dm.d),
            logits: DevBuf::new(BMAX * dm.vocab),
            best: DevBuf::new(BMAX * 2),
            best_peer: DevBuf::new(BMAX * 2),
            drafted: DevBuf::new(BMAX * BMAX),
            err: DevBuf::zeroed(1, s),
            qr: DevBuf::zeroed(padded * dm.hq * hp.hd, s),
            // Decode: a few row tiles, each cut into segments; prefill: many row tiles, one segment each.
            part: DevBuf::new((n_slots * cap.div_ceil(hip::ATTN_SEG_SLOTS)).max(attn_tiles(rows)) * dm.hq * hip::ATTN_PART),
            mix_s: DevBuf::new((3 * hip::MAX_GROUPS + rows) * dm.ch),
            qkv_s: f(dm.ch + 2 * dm.gv),
            gdn,
            kv,
            emb: f(dm.d),
            hin: f(dm.d),
            hprev: DevBuf::zeroed(n_slots * dm.d, s),
        }
    }
}

impl Rank {
    /// The batch's first row in the block-major activation buffers.
    fn act(&self, bt: &Batch) -> (*mut i8, *mut f32) {
        unsafe { (self.xq.ptr().add(bt.off * 32), self.xd.ptr().add(bt.off)) }
    }
}

/// Row stride of the block-major activation layout for a batch of `b` rows: one tile of 16
/// for decode-sized batches, whole 128-row tiles above that.
fn xs(b: usize) -> i32 {
    (if b <= BMAX { BMAX } else { b.next_multiple_of(128) }) as i32
}

/// Row tiles the attention scratch must hold for a batch of up to `rows` rows.
fn attn_tiles(rows: usize) -> usize {
    rows.div_ceil(16) + hip::MAX_GROUPS
}

/// Which partial (own + peer) the next norm must fold into the residual stream.
#[derive(Clone, Copy, PartialEq)]
enum Pend {
    None,
    A,
    B,
}

#[derive(Clone, Copy)]
enum Sel {
    Trunk(usize),
    Mtp,
}

struct Batch {
    g: Groups,
    b: usize,
    /// First row of the batch in the row buffers (non-zero for the second half of a split step),
    /// and where its share of the GDN and attention scratch starts.
    off: usize,
    mix_row: usize,
    tile_off: usize,
    /// Attention geometry: row tiles in the batch, and the most cache segments any tile spans.
    tiles: usize,
    segs: usize,
    seg_tiles: i32,
    /// Large batches: first and last cache stretch to sweep.
    sweep: (i32, i32),
    /// Above 16 rows: committed, GDN on scratch, bulk exchange, head on the last row only.
    big: bool,
    /// Row stride of the activation layout.
    xs: i32,
}

impl Engine {
    pub fn new(hp: HParams, ranks: Vec<Rank>, n_slots: usize, cap: usize, rows: usize) -> Self {
        assert!(n_slots <= 2, "the KV pool is laid out for at most two sequences");
        let seqs = vec![Seq { last_committed: true, ..Default::default() }; n_slots];
        assert!(ranks.len() <= 2, "the exchange is pairwise: one or two ranks");
        let xchg = (ranks.len() == 2).then(|| Xchg { mem: hip::Pinned::shared(4 * XCHG_ROWS * hp.d * 4 + 2 * XCHG_FLAGS * 8), seq: std::cell::Cell::new(0) });
        Self { hp, ranks, seqs, cap, rows, dump: false, prof: None, xchg }
    }

    /// Forget a sequence: zero its recurrent state and conv history.
    pub fn reset(&mut self, slot: usize) {
        for r in &self.ranks {
            hip::set_device(r.dev);
            for g in r.gdn.iter().flatten() {
                // Buffers are per-slot contiguous; zero this slot's share.
                let zero = |b: &DevBuf<f32>| {
                    let per = b.len() / self.seqs.len();
                    unsafe { hip::zero_range(b.ptr().add(slot * per) as *mut c_void, per * 4, r.s) }
                };
                zero(&g.mix);
                zero(&g.qkv);
                zero(&g.state);
            }
        }
        let d = self.hp.d;
        for r in &self.ranks {
            hip::set_device(r.dev);
            unsafe { hip::zero_range(r.hprev.ptr().add(slot * d) as *mut c_void, d * 4, r.s) };
        }
        self.seqs[slot] = Seq { last_committed: true, ..Default::default() };
    }

    /// Accept the first `n` rows of the slot's last (pending) step.
    pub fn accept(&mut self, slot: usize, n: usize) {
        let s = &mut self.seqs[slot];
        assert!(s.awaiting_accept && n >= 1 && n <= s.last_rows);
        s.last_acc = n;
        s.len += n;
        s.awaiting_accept = false;
        let row = s.last_row0 + n - 1;
        self.save_hprev(slot, row);
    }

    /// Remember the trunk hidden of `row` as the slot's h_prev for the MTP head.
    fn save_hprev(&self, slot: usize, row: usize) {
        let d = self.hp.d;
        for r in self.ranks.iter().filter(|r| r.w.mtp.is_some()) {
            hip::set_device(r.dev);
            unsafe { hip::copy_dev(r.hprev.ptr().add(slot * d), r.xf.ptr().add(row * d), d, r.s) };
        }
    }

    /// Profiling hook: charge everything queued on `r` since the last call to `bucket`.
    fn t(&self, r: &Rank, bucket: &'static str) {
        if let Some(p) = &self.prof {
            hip::sync(r.s);
            let mut p = p.borrow_mut();
            let dt = p.last.elapsed().as_secs_f64();
            let e = p.buckets.entry(bucket).or_insert((0.0, 0));
            e.0 += dt;
            e.1 += 1;
            p.last = Instant::now();
        }
    }

    fn norm(&self, r: &Rank, pend: Pend, w: &DevBuf<f32>, bt: &Batch, xf: bool) {
        let null = std::ptr::null();
        let two = self.ranks.len() > 1;
        let o = bt.off * self.hp.d;
        let at = |buf: &DevBuf<f32>| unsafe { buf.ptr().add(o) } as *const f32;
        let (p0, p1) = match pend {
            Pend::None => (null, null),
            Pend::A => (at(&r.pa), if two { at(&r.peer_a) } else { null }),
            Pend::B => (at(&r.pb), if two { at(&r.peer_b) } else { null }),
        };
        let xf = if xf { at(&r.xf) as *mut f32 } else { std::ptr::null_mut() };
        let (xq, xd) = r.act(bt);
        unsafe { hip::pk_norm_quant(r.s, at(&r.h) as *mut f32, p0, p1, w.ptr(), self.hp.eps, self.hp.d as i32, bt.b as i32, xq, xd, bt.xs, 0, xf) };
        self.t(r, "norm");
    }

    /// y[row][m.rows] = m x activations, for the batch's rows. `y` is the batch's first row.
    fn gemv(&self, r: &Rank, m: &Mat, y: *mut f32, bt: &Batch) {
        let (xq, xd) = r.act(bt);
        unsafe { hip::pk_gemm(r.s, m.qs.ptr(), m.d.ptr(), xq, xd, bt.xs, 0, y, m.rows as i64, m.cols as i32, bt.b as i32, m.rows as i32) }
        self.t(r, match m.rows { 0..=6000 => "gemv-out", 6001..=40000 => "gemv-in", _ => "gemv-head" });
    }

    /// Trade a sublayer's partial outputs between the two ranks.
    ///
    /// Batches up to XCHG_ROWS rows do it entirely on the cards: each rank publishes its partial to
    /// shared host memory and pulls the peer's. Nothing blocks the host; the kernels order
    /// themselves with per-exchange sequence numbers, and two alternating buffers keep exchange
    /// n's data alive while n+1 is being published.
    fn exchange(&self, a: bool, bt: &Batch) {
        let Some(x) = &self.xchg else { return };
        if bt.b > XCHG_ROWS {
            // Many megabytes per rank: a bulk card-to-card copy beats per-element atomics.
            self.peer_copy(a, bt, false);
            self.sync_all();
            self.t(&self.ranks[0], "exchange");
            return;
        }
        let seq = x.seq.get() + 1;
        x.seq.set(seq);
        let n = bt.b * self.hp.d;
        let area = XCHG_ROWS * self.hp.d;
        let data = x.mem.as_ptr() as *mut f32;
        let flags = unsafe { data.add(4 * area) } as *mut i64;
        for (i, r) in self.ranks.iter().enumerate() {
            hip::set_device(r.dev);
            let (own, peer) = if a { (&r.pa, &r.peer_a) } else { (&r.pb, &r.peer_b) };
            let half = (seq & 1) as usize;
            unsafe {
                hip::pk_xchg(r.s, own.ptr(), data.add((i * 2 + half) * area), flags.add(i * XCHG_FLAGS), data.add(((1 - i) * 2 + half) * area), flags.add((1 - i) * XCHG_FLAGS), peer.ptr(), n as i32, seq, r.err.ptr());
            }
        }
        if self.prof.is_some() {
            self.sync_all();
            self.t(&self.ranks[0], "exchange");
        }
    }

    /// Queue the bulk copy of each rank's partial (the batch's rows) into its peer. On the copy
    /// stream it runs beside whatever the main streams are computing.
    fn peer_copy(&self, a: bool, bt: &Batch, side_stream: bool) {
        let o = bt.off * self.hp.d;
        for (i, r) in self.ranks.iter().enumerate() {
            let peer = &self.ranks[1 - i];
            hip::set_device(r.dev);
            let (src, dst) = if a { (&r.pa, &peer.peer_a) } else { (&r.pb, &peer.peer_b) };
            unsafe { hip::copy_peer(dst.ptr().add(o), peer.dev, src.ptr().add(o) as *const f32, r.dev, bt.b * self.hp.d, if side_stream { r.sx } else { r.s }) };
        }
    }

    fn sync_all(&self) {
        for r in &self.ranks {
            hip::set_device(r.dev);
            hip::sync(r.s);
        }
    }

    fn layer_of<'a>(&self, r: &'a Rank, sel: Sel) -> (&'a LayerW, Option<&'a KvBufs>) {
        match sel {
            Sel::Trunk(l) => (&r.w.layers[l], r.kv[l].as_ref()),
            Sel::Mtp => (&r.w.mtp.as_ref().unwrap().layer, r.mtp_kv.as_ref()),
        }
    }

    /// First half of a decoder block on every rank: fold in the pending partial, run the mixer
    /// (GDN or attention), leave its output partial in `pa`.
    fn mixer_stage(&self, sel: Sel, bt: &Batch, pend: Pend) {
        let hp = &self.hp;
        let (bi, g) = (bt.b as i32, &bt.g);
        for r in &self.ranks {
            hip::set_device(r.dev);
            let (lw, kv) = self.layer_of(r, sel);
            let dm = &r.dims;
            let (xq, xd) = r.act(bt);
            let pa = unsafe { r.pa.ptr().add(bt.off * hp.d) };
            self.norm(r, pend, &lw.attn_norm, bt, false);
            match &lw.mixer {
                Mixer::Gdn { win, wout, conv_w, dt_bias, ssm_a, norm_w } => {
                    let Sel::Trunk(l) = sel else { unreachable!() };
                    let gb = r.gdn[l].as_ref().unwrap();
                    let y = unsafe { r.y.ptr().add(bt.off * dm.gdn_nin) };
                    self.gemv(r, win, y, bt);
                    let null = std::ptr::null_mut();
                    let (mix_s, qkv_s) = if bt.big { unsafe { (r.mix_s.ptr().add(bt.mix_row * dm.ch), r.qkv_s.ptr().add(bt.off * (dm.ch + 2 * dm.gv))) } } else { (null, null) };
                    unsafe {
                        hip::pk_gdn_conv(r.s, y, dm.gdn_nin as i32, gb.mix.ptr(), gb.qkv.ptr(), R as i32, conv_w.ptr(), dt_bias.ptr(), ssm_a.ptr(), dm.gk as i32, dm.gv as i32, hp.eps, mix_s, qkv_s, bi, g);
                        hip::pk_gdn_step(r.s, gb.state.ptr(), gb.qkv.ptr(), R as i32, y, dm.gdn_nin as i32, norm_w.ptr(), dm.gk as i32, dm.gv as i32, hp.eps, xq, xd, bt.xs, qkv_s, g);
                        if bt.big {
                            // Keep the last 3 raw rows as the next step's conv history.
                            for i in 0..g.n as usize {
                                let (start, len, slot, cur) = (g.start[i] as usize, g.len[i] as usize, g.slot[i] as usize, g.cur[i] as usize);
                                hip::copy_dev(gb.mix.ptr().add((slot * 2 + cur) * (3 + R) * dm.ch), mix_s.add((start + 3 * i + len) * dm.ch), 3 * dm.ch, r.s);
                            }
                        }
                    }
                    self.t(r, "gdn");
                    self.gemv(r, wout, pa, bt);
                }
                Mixer::Attn { win, wout, q_norm, k_norm } => {
                    let kv = kv.unwrap();
                    let y = unsafe { r.y.ptr().add(bt.off * dm.attn_nin) };
                    self.gemv(r, win, y, bt);
                    unsafe {
                        let qr = r.qr.ptr().add(bt.off * dm.hq * hp.hd);
                        let part = r.part.ptr().add(bt.tile_off * dm.hq * bt.segs * hip::ATTN_PART);
                        hip::pk_attn_prep(r.s, y, dm.attn_nin as i32, dm.hq as i32, dm.hkv as i32, 0, (dm.hq * 2 * hp.hd) as i32, q_norm.ptr(), k_norm.ptr(), hp.eps, hp.theta_scale, hp.n_rot as i32, qr, kv.k.ptr(), kv.v.ptr(), bi, g);
                        if bt.big {
                            hip::pk_attn_swept(r.s, qr, y, dm.attn_nin as i32, dm.hq as i32, dm.hkv as i32, kv.k.ptr(), kv.v.ptr(), bt.seg_tiles, bt.sweep.0, bt.sweep.1, part, bt.tiles as i32, xq, xd, bt.xs, bi, g);
                        } else {
                            hip::pk_attn(r.s, qr, y, dm.attn_nin as i32, dm.hq as i32, dm.hkv as i32, kv.k.ptr(), kv.v.ptr(), bt.seg_tiles, part, bt.tiles as i32, bt.segs as i32, xq, xd, bt.xs, bi, g);
                        }
                    }
                    self.t(r, "attn");
                    self.gemv(r, wout, pa, bt);
                }
            }
        }
    }

    /// Second half of a decoder block on every rank: fold in the mixer partial, run the FFN,
    /// leave its output partial in `pb`.
    fn ffn_stage(&self, sel: Sel, bt: &Batch) {
        for r in &self.ranks {
            hip::set_device(r.dev);
            let (lw, _) = self.layer_of(r, sel);
            let (xq, xd) = r.act(bt);
            let y = unsafe { r.y.ptr().add(bt.off * 2 * r.dims.ff) };
            self.norm(r, Pend::A, &lw.post_norm, bt, false);
            self.gemv(r, &lw.upgate, y, bt);
            unsafe { hip::pk_ffn_act(r.s, y, r.dims.ff as i32, bt.b as i32, xq, xd, bt.xs) };
            self.t(r, "ffn-act");
            self.gemv(r, &lw.down, unsafe { r.pb.ptr().add(bt.off * self.hp.d) }, bt);
        }
    }

    /// One decoder block (mixer + FFN) on every rank. Leaves its FFN partial pending (Pend::B).
    fn block(&self, sel: Sel, bt: &Batch, pend: Pend) {
        self.mixer_stage(sel, bt, pend);
        self.exchange(true, bt);
        if let (true, Sel::Trunk(l)) = (self.dump, sel) {
            if l > 0 {
                self.dump_h(&format!("l_out-{}", l - 1), bt.b);
            }
        }
        self.ffn_stage(sel, bt);
        self.exchange(false, bt);
    }

    /// The whole trunk for two halves of a large committed batch, interleaved stage by stage:
    /// while one half's partials cross between the cards, the other half computes. The halves
    /// are consecutive steps of one sequence, so the second sees the first's state and cache.
    fn trunk_interleaved(&self, halves: [&Batch; 2]) {
        let stages = 2 * self.hp.n_layer;
        let launch = |k: usize, h: usize| {
            let (sel, bt) = (Sel::Trunk(k / 2), halves[h]);
            if k % 2 == 0 {
                self.mixer_stage(sel, bt, if k == 0 { Pend::None } else { Pend::B });
            } else {
                self.ffn_stage(sel, bt);
            }
            for r in &self.ranks {
                hip::set_device(r.dev);
                hip::event_record(r.ev_done[h], r.s);
            }
        };
        launch(0, 0);
        launch(0, 1);
        for k in 0..stages {
            for h in 0..2 {
                // Stage k of this half is computed: ship its partial, and as soon as it lands
                // queue the half's next stage behind whatever the other half is running.
                for r in &self.ranks {
                    hip::event_wait(r.ev_done[h]);
                }
                self.peer_copy(k % 2 == 0, halves[h], true);
                for r in &self.ranks {
                    hip::set_device(r.dev);
                    hip::event_record(r.ev_copied[h], r.sx);
                }
                for r in &self.ranks {
                    hip::event_wait(r.ev_copied[h]);
                }
                if k + 1 < stages {
                    launch(k + 1, h);
                }
            }
        }
    }

    /// Queue the final norm (normed hidden rows stay in xf), the LM head and the per-rank argmax.
    /// The MTP head scores only its pruned vocabulary. A large batch scores only its last row.
    /// Returns the number of rows scored.
    fn head_launch(&self, mtp: bool, bt: &Batch) -> usize {
        let (first, n) = if bt.big { (bt.b - 1, 1) } else { (0, bt.b) };
        for r in &self.ranks {
            hip::set_device(r.dev);
            let (w, head) = match (mtp, &r.w.mtp) {
                (true, Some(m)) => (&m.head_norm, &m.draft_head),
                _ => (&r.w.out_norm, &r.w.head),
            };
            self.norm(r, Pend::B, w, bt, true);
            let (xq, xd) = r.act(bt);
            unsafe {
                hip::pk_gemm(r.s, head.qs.ptr(), head.d.ptr(), xq, xd, bt.xs, first as i32, r.logits.ptr(), head.rows as i64, head.cols as i32, n as i32, head.rows as i32);
                self.t(r, "gemv-head");
                hip::pk_argmax(r.s, r.logits.ptr(), head.rows as i64, head.rows as i64, n as i32, r.best.ptr());
            }
        }
        n
    }

    /// Wait for everything queued and fail loudly if the ranks lost step with each other.
    fn finish(&self) {
        let mut errs = vec![[0i32]; self.ranks.len()];
        for (r, e) in self.ranks.iter().zip(errs.iter_mut()) {
            hip::set_device(r.dev);
            r.err.download(e, r.s);
        }
        self.sync_all();
        assert!(errs.iter().all(|e| e[0] == 0), "cross-card exchange timed out ({errs:?}): the ranks lost step with each other");
    }

    /// Trunk head: greedy pick per scored row, merged across ranks on the host. Synchronises.
    fn head(&self, bt: &Batch) -> Vec<RowOut> {
        let b = self.head_launch(false, bt);
        let mut best = vec![vec![0i32; b * 2]; self.ranks.len()];
        for (r, out) in self.ranks.iter().zip(best.iter_mut()) {
            hip::set_device(r.dev);
            r.best.download(out, r.s);
        }
        self.finish();
        (0..b)
            .map(|i| {
                // Highest logit across ranks; the lower vocab range wins ties.
                let mut out = RowOut { token: 0, logit: f32::NEG_INFINITY };
                for (r, bst) in self.ranks.iter().zip(&best) {
                    let logit = f32::from_bits(bst[i * 2] as u32);
                    if logit > out.logit {
                        out = RowOut { token: (r.dims.vocab0 + bst[i * 2 + 1] as usize) as u32, logit };
                    }
                }
                out
            })
            .collect()
    }

    /// MTP head: pick each row's draft token on the cards and feed it straight back as the
    /// next draft step's input. Nothing waits on the host.
    fn head_draft(&self, bt: &Batch, step: usize) {
        let b = bt.b;
        self.head_launch(true, bt);
        if let Some(x) = &self.xchg {
            let seq = x.seq.get() + 1;
            x.seq.set(seq);
            let area = XCHG_ROWS * self.hp.d;
            let data = x.mem.as_ptr() as *mut f32;
            let flags = unsafe { data.add(4 * area) } as *mut i64;
            let half = (seq & 1) as usize;
            for (i, r) in self.ranks.iter().enumerate() {
                hip::set_device(r.dev);
                unsafe {
                    hip::pk_xchg(r.s, r.best.ptr() as *const f32, data.add((i * 2 + half) * area), flags.add(i * XCHG_FLAGS), data.add(((1 - i) * 2 + half) * area), flags.add((1 - i) * XCHG_FLAGS), r.best_peer.ptr() as *mut f32, (b * 2) as i32, seq, r.err.ptr());
                }
            }
        }
        for (i, r) in self.ranks.iter().enumerate() {
            hip::set_device(r.dev);
            let v0 = |r: &Rank| r.w.mtp.as_ref().unwrap().draft_v0 as i32;
            // With one rank the "peer" is the rank itself: same pair, same answer.
            let (peer_best, peer_v0) = if self.ranks.len() == 2 { (r.best_peer.ptr(), v0(&self.ranks[1 - i])) } else { (r.best.ptr(), v0(r)) };
            unsafe { hip::pk_pick(r.s, r.best.ptr(), peer_best, v0(r), peer_v0, r.tokens.ptr(), r.drafted.ptr().add(step * BMAX), b as i32) };
        }
    }

    /// Describe a batch of `b` rows that starts at row `off` of the row buffers. `total` is the
    /// row count of the whole step (it fixes the activation layout's row stride).
    fn batch(&self, g: Groups, b: usize, off: usize, total: usize) -> Batch {
        let (mut tiles, mut segs) = (0, 1);
        let mut sweep = (i32::MAX, 0);
        for i in 0..g.n as usize {
            let (len, pos0) = (g.len[i] as usize, g.pos0[i] as usize);
            tiles += len.div_ceil(16);
            // The last tile reaches furthest; its slot range is [0, p] or [base - p, base].
            let p = pos0 + len - 1;
            let (lo, hi) = if g.kv_dir[i] > 0 { (0, p) } else { (g.kv_base[i] as usize - p, g.kv_base[i] as usize) };
            if b <= R {
                segs = segs.max(hi / hip::ATTN_SEG_SLOTS - lo / hip::ATTN_SEG_SLOTS + 1);
            }
            sweep = (sweep.0.min((lo / hip::ATTN_SWEEP_SLOTS) as i32), sweep.1.max((hi / hip::ATTN_SWEEP_SLOTS) as i32));
        }
        // A large batch keeps one partial per row tile and sweeps the cache in long stretches.
        let seg_tiles = (if b > R { hip::ATTN_SWEEP_SLOTS } else { hip::ATTN_SEG_SLOTS } / 16) as i32;
        // A second half sits after the first in every scratch area: 3 history rows + rows in
        // the GDN scratch, whole row tiles in the attention scratch.
        let (mix_row, tile_off) = if off == 0 { (0, 0) } else { (3 + off, off.div_ceil(16)) };
        assert!(tile_off + tiles <= attn_tiles(self.rows));
        Batch { g, b, off, mix_row, tile_off, tiles, segs: if b > R { 1 } else { segs }, seg_tiles, sweep, big: b > R, xs: xs(total) }
    }

    fn kv_groups(&self, i: usize, slot: usize, g: &mut Groups) {
        g.slot[i] = slot as i32;
        g.kv_dir[i] = if slot == 0 { 1 } else { -1 };
        g.kv_base[i] = if slot == 0 { 0 } else { self.cap as i64 - 1 };
    }

    /// MTP input for the batch's rows of `tokens`/`hin`: h = eh_proj([enorm(embed(tok)) | hnorm(hin)]).
    fn mtp_input(&self, r: &Rank, bt: &Batch) {
        let m = r.w.mtp.as_ref().unwrap();
        let (d, bi) = (self.hp.d as i32, bt.b as i32);
        let o = bt.off * self.hp.d;
        let null = std::ptr::null();
        let (xq, xd) = r.act(bt);
        unsafe {
            hip::pk_embed(r.s, r.w.embd.qs.ptr(), r.w.embd.d.ptr(), r.tokens.ptr().add(bt.off), r.emb.ptr().add(o), d, bi);
            hip::pk_norm_quant(r.s, r.emb.ptr().add(o), null, null, m.enorm.ptr(), self.hp.eps, d, bi, xq, xd, bt.xs, 0, std::ptr::null_mut());
            hip::pk_norm_quant(r.s, r.hin.ptr().add(o), null, null, m.hnorm.ptr(), self.hp.eps, d, bi, xq, xd, bt.xs, d, std::ptr::null_mut());
            self.gemv(r, &m.eh_proj, r.h.ptr().add(o), bt);
        }
    }

    /// Refill the MTP head's KV for the rows the trunk just ran: position p pairs token x_p with
    /// the trunk hidden of p-1. Only K and V are needed, so this skips attention and the FFN.
    fn mtp_catchup(&self, bt: &Batch) {
        let hp = &self.hp;
        let d = hp.d;
        for r in self.ranks.iter().filter(|r| r.w.mtp.is_some()) {
            hip::set_device(r.dev);
            let m = r.w.mtp.as_ref().unwrap();
            let Mixer::Attn { win, q_norm, k_norm, .. } = &m.layer.mixer else { unreachable!() };
            let kv = r.mtp_kv.as_ref().unwrap();
            let dm = &r.dims;
            for i in 0..bt.g.n as usize {
                let (start, len, slot) = (bt.off + bt.g.start[i] as usize, bt.g.len[i] as usize, bt.g.slot[i] as usize);
                unsafe {
                    hip::copy_dev(r.hin.ptr().add(start * d), r.hprev.ptr().add(slot * d), d, r.s);
                    if len > 1 {
                        hip::copy_dev(r.hin.ptr().add((start + 1) * d), r.xf.ptr().add(start * d), (len - 1) * d, r.s);
                    }
                }
            }
            self.mtp_input(r, bt);
            self.norm(r, Pend::None, &m.layer.attn_norm, bt, false);
            // K and V rows sit after the q rows of the fused "in" matrix.
            let q_rows = dm.hq * 2 * hp.hd;
            let kv_rows = 2 * dm.hkv * hp.hd;
            let (xq, xd) = r.act(bt);
            unsafe {
                let y = r.y.ptr().add(bt.off * kv_rows);
                hip::pk_gemm(r.s, win.qs.ptr().add(q_rows * win.cols), win.d.ptr().add(q_rows * win.cols / 32), xq, xd, bt.xs, 0, y, kv_rows as i64, win.cols as i32, bt.b as i32, kv_rows as i32);
                hip::pk_attn_prep(r.s, y, kv_rows as i32, dm.hq as i32, dm.hkv as i32, dm.hq as i32, 0, q_norm.ptr(), k_norm.ptr(), hp.eps, hp.theta_scale, hp.n_rot as i32, r.qr.ptr(), kv.k.ptr(), kv.v.ptr(), bt.b as i32, &bt.g);
            }
        }
    }

    /// Draft up to `n_max` tokens per sequence with the MTP head. `last` gives each slot's most
    /// recent token (sampled but not yet run through the trunk). Nothing here touches trunk state.
    pub fn draft(&mut self, last: &[(usize, u32)], n_max: usize) -> Vec<Vec<u32>> {
        let d = self.hp.d;
        let b = last.len();
        assert!(n_max <= BMAX && b <= BMAX);
        let first: Vec<i32> = last.iter().map(|&(_, t)| t as i32).collect();
        let t_draft = Instant::now();
        for step in 0..n_max {
            let mut g = Groups { n: b as i32, ..Default::default() };
            for (i, &(slot, _)) in last.iter().enumerate() {
                assert!(!self.seqs[slot].awaiting_accept);
                g.start[i] = i as i32;
                g.len[i] = 1;
                g.pos0[i] = (self.seqs[slot].len + step) as i32;
                self.kv_groups(i, slot, &mut g);
            }
            let bt = self.batch(g, b, 0, b);
            for r in &self.ranks {
                hip::set_device(r.dev);
                if step == 0 {
                    r.tokens.upload(&first, r.s);
                }
                // h input: the trunk's last accepted hidden first, then the head's own output.
                for (i, &(slot, _)) in last.iter().enumerate() {
                    let src = if step == 0 { unsafe { r.hprev.ptr().add(slot * d) } } else { unsafe { r.xf.ptr().add(i * d) } };
                    unsafe { hip::copy_dev(r.hin.ptr().add(i * d), src, d, r.s) };
                }
                self.mtp_input(r, &bt);
            }
            self.block(Sel::Mtp, &bt, Pend::None);
            self.head_draft(&bt, step);
        }
        let mut hist = vec![0i32; n_max * BMAX];
        let r0 = &self.ranks[0];
        hip::set_device(r0.dev);
        r0.drafted.download(&mut hist, r0.s);
        self.finish();
        if std::env::var_os("FCL_TRACE").is_some() {
            eprintln!("draft x{n_max}: {:.2} ms", t_draft.elapsed().as_secs_f64() * 1e3);
        }
        (0..b).map(|i| (0..n_max).map(|step| hist[step * BMAX + i] as u32).collect()).collect()
    }

    /// Open a group of `n` rows for `slot` starting at batch row `start`: fill its entry of `g`
    /// from the sequence's state and advance that state past this step.
    fn open_group(&mut self, g: &mut Groups, i: usize, slot: usize, start: usize, n: usize, commit: bool) {
        let s = self.seqs[slot];
        assert!(!s.awaiting_accept, "slot {slot}: accept() the previous step first");
        assert!(n > 0 && (n <= R || commit), "only committed batches may exceed {R} rows");
        g.start[i] = start as i32;
        g.len[i] = n as i32;
        g.cur[i] = 1 - s.cur;
        g.hist_off[i] = s.last_acc as i32;
        g.nacc[i] = if s.last_committed { 0 } else { s.last_acc as i32 };
        g.commit[i] = commit as i32;
        g.pos0[i] = s.len as i32;
        self.kv_groups(i, slot, g);
        self.seqs[slot] = Seq {
            len: s.len + if commit { n } else { 0 },
            cur: 1 - s.cur,
            last_rows: n,
            last_row0: start,
            // A large batch leaves its history at the head of the buffer, a small one in place.
            last_acc: if commit && n <= R { n } else { 0 },
            last_committed: commit,
            awaiting_accept: !commit,
        };
    }

    fn embed(&self, bt: &Batch) {
        for r in &self.ranks {
            hip::set_device(r.dev);
            unsafe { hip::pk_embed(r.s, r.w.embd.qs.ptr(), r.w.embd.d.ptr(), r.tokens.ptr().add(bt.off), r.h.ptr().add(bt.off * self.hp.d), self.hp.d as i32, bt.b as i32) };
        }
    }

    /// Run one batch through the trunk. `rows` lists (slot, tokens, commit) groups; returns the
    /// greedy next token per row (for a batch above 16 rows: for its last row only). Each group
    /// either commits (prefill) or stays pending; pending groups must be `accept`ed before any
    /// other call.
    pub fn step(&mut self, rows: &[(usize, &[u32], bool)]) -> Vec<RowOut> {
        let tokens: Vec<i32> = rows.iter().flat_map(|r| r.1.iter().map(|&t| t as i32)).collect();
        let b = tokens.len();
        assert!(b <= self.rows && (b <= BMAX || rows.len() == 1), "a large batch carries one sequence");
        assert!(self.seqs.iter().map(|s| s.len).sum::<usize>() + b <= self.cap, "KV pool full");
        for r in &self.ranks {
            hip::set_device(r.dev);
            r.tokens.upload(&tokens, r.s);
        }
        let t_step = Instant::now();

        // Two cards and a batch too big for the on-card exchange: run it as two interleaved
        // halves so the bulk card-to-card transfers overlap compute.
        let paired = self.ranks.len() == 2 && b > XCHG_ROWS && self.prof.is_none() && !self.dump;
        let out = if paired {
            let (slot, _, commit) = rows[0];
            let first = (b / 2).next_multiple_of(16);
            let mut halves = Vec::new();
            for (off, n) in [(0, first), (first, b - first)] {
                let mut g = Groups { n: 1, ..Default::default() };
                self.open_group(&mut g, 0, slot, 0, n, commit);
                halves.push(self.batch(g, n, off, b));
            }
            let (ha, hb) = (&halves[0], &halves[1]);
            self.embed(ha);
            self.embed(hb);
            self.trunk_interleaved([ha, hb]);
            for r in &self.ranks {
                hip::set_device(r.dev);
                self.norm(r, Pend::B, &r.w.out_norm, ha, true);
            }
            let out = self.head(hb);
            self.mtp_catchup(ha);
            self.save_hprev(slot, first - 1);
            self.mtp_catchup(hb);
            self.save_hprev(slot, b - 1);
            out
        } else {
            let mut g = Groups { n: rows.len() as i32, ..Default::default() };
            let mut start = 0;
            for (i, &(slot, toks, commit)) in rows.iter().enumerate() {
                self.open_group(&mut g, i, slot, start, toks.len(), commit);
                start += toks.len();
            }
            let bt = self.batch(g, b, 0, b);
            self.embed(&bt);
            let mut pend = Pend::None;
            for l in 0..self.hp.n_layer {
                self.block(Sel::Trunk(l), &bt, pend);
                pend = Pend::B;
            }
            let t_enq = Instant::now();
            let out = self.head(&bt);
            if std::env::var_os("FCL_TRACE").is_some() {
                eprintln!("step b={b}: enqueue {:.2} ms, then wait {:.2} ms", (t_enq - t_step).as_secs_f64() * 1e3, t_enq.elapsed().as_secs_f64() * 1e3);
            }
            if self.dump {
                self.dump_h(&format!("l_out-{}", self.hp.n_layer - 1), b);
            }
            self.mtp_catchup(&bt);
            for (i, &(slot, toks, commit)) in rows.iter().enumerate() {
                if commit {
                    self.save_hprev(slot, bt.g.start[i] as usize + toks.len() - 1);
                }
            }
            out
        };
        out
    }

    /// Copy the last row's full logits (all ranks, vocab order) to the host.
    pub fn logits(&self, row: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.hp.vocab];
        for r in &self.ranks {
            hip::set_device(r.dev);
            let dst = &mut out[r.dims.vocab0..r.dims.vocab0 + r.dims.vocab];
            unsafe { hip::download_raw(dst.as_mut_ptr() as *mut u8, r.logits.ptr().add(row * r.dims.vocab) as *const u8, dst.len() * 4, r.s) };
            hip::sync(r.s);
        }
        out
    }

    fn dump_h(&self, name: &str, b: usize) {
        let r = &self.ranks[0];
        hip::set_device(r.dev);
        hip::sync(r.s);
        let mut h = vec![0f32; b * self.hp.d];
        r.h.download(&mut h, r.s);
        hip::sync(r.s);
        let row = &h[(b - 1) * self.hp.d..];
        let sum: f64 = h.iter().map(|&v| v as f64).sum();
        println!("{name}: sum {sum:.6} first [{:.4}, {:.4}, {:.4}] last [{:.4}, {:.4}, {:.4}]", row[0], row[1], row[2], row[self.hp.d - 3], row[self.hp.d - 2], row[self.hp.d - 1]);
    }
}
