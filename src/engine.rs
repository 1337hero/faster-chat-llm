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
use crate::model::{Dims, HParams, Mat, Mixer, Shard, Weights};

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
    qr: DevBuf<f32>,
    scores: DevBuf<f32>,
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
    pub fn new(dev: usize, s: Stream, hp: &HParams, sh: Shard, w: Weights, n_slots: usize, cap: usize) -> Self {
        let dm = sh.dims(hp);
        let xmax = (2 * dm.d).max(dm.ff).max(dm.gv * hp.gs).max(dm.hq * hp.hd);
        let ymax = (2 * dm.ff).max(dm.gdn_nin).max(dm.attn_nin);
        let new_kv = || KvBufs { k: DevBuf::new(cap * dm.hkv * hp.hd), v: DevBuf::new(cap * dm.hkv * hp.hd) };
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
        let f = |n: usize| DevBuf::<f32>::new(BMAX * n);
        Self {
            dev,
            s,
            dims: dm,
            mtp_kv: w.mtp.as_ref().map(|_| new_kv()),
            w,
            tokens: DevBuf::new(BMAX),
            h: f(dm.d),
            xq: DevBuf::new(BMAX * xmax),
            xd: f(xmax / 32),
            xf: f(dm.d),
            y: f(ymax),
            pa: f(dm.d),
            pb: f(dm.d),
            peer_a: f(dm.d),
            peer_b: f(dm.d),
            logits: f(dm.vocab),
            best: DevBuf::new(BMAX * 2),
            best_peer: DevBuf::new(BMAX * 2),
            drafted: DevBuf::new(BMAX * BMAX),
            err: DevBuf::zeroed(1, s),
            qr: f(dm.hq * hp.hd),
            scores: f(dm.hq * cap),
            gdn,
            kv,
            emb: f(dm.d),
            hin: f(dm.d),
            hprev: DevBuf::zeroed(n_slots * dm.d, s),
        }
    }
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
}

impl Engine {
    pub fn new(hp: HParams, ranks: Vec<Rank>, n_slots: usize, cap: usize) -> Self {
        assert!(n_slots <= 2, "the KV pool is laid out for at most two sequences");
        let seqs = vec![Seq { last_committed: true, ..Default::default() }; n_slots];
        assert!(ranks.len() <= 2, "the exchange is pairwise: one or two ranks");
        let xchg = (ranks.len() == 2).then(|| Xchg { mem: hip::Pinned::shared(4 * BMAX * hp.d * 4 + 2 * XCHG_FLAGS * 8), seq: std::cell::Cell::new(0) });
        Self { hp, ranks, seqs, cap, dump: false, prof: None, xchg }
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

    fn norm(&self, r: &Rank, pend: Pend, w: &DevBuf<f32>, b: usize, xf: bool) {
        let null = std::ptr::null();
        let two = self.ranks.len() > 1;
        let (p0, p1) = match pend {
            Pend::None => (null, null),
            Pend::A => (r.pa.ptr() as *const f32, if two { r.peer_a.ptr() as *const f32 } else { null }),
            Pend::B => (r.pb.ptr() as *const f32, if two { r.peer_b.ptr() as *const f32 } else { null }),
        };
        let d = self.hp.d as i32;
        let xf = if xf { r.xf.ptr() } else { std::ptr::null_mut() };
        unsafe { hip::pk_norm_quant(r.s, r.h.ptr(), p0, p1, w.ptr(), self.hp.eps, d, b as i32, r.xq.ptr(), r.xd.ptr(), d, xf) };
        self.t(r, "norm");
    }

    fn gemv(&self, r: &Rank, m: &Mat, y: *mut f32, b: usize) {
        unsafe { hip::pk_gemm(r.s, m.qs.ptr(), m.d.ptr(), r.xq.ptr(), r.xd.ptr(), y, m.rows as i64, m.cols as i32, b as i32, m.rows as i32) }
        self.t(r, match m.rows { 0..=6000 => "gemv-out", 6001..=40000 => "gemv-in", _ => "gemv-head" });
    }

    /// Trade a sublayer's partial outputs between the two ranks, entirely on the cards: each
    /// rank publishes its partial to shared host memory and pulls the peer's. Nothing here
    /// blocks the host; the kernels order themselves with per-exchange sequence numbers.
    /// Two alternating buffers keep exchange n's data alive while n+1 is being published.
    fn exchange(&self, a: bool, b: usize) {
        let Some(x) = &self.xchg else { return };
        let seq = x.seq.get() + 1;
        x.seq.set(seq);
        let n = b * self.hp.d;
        let area = BMAX * self.hp.d;
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

    fn sync_all(&self) {
        for r in &self.ranks {
            hip::set_device(r.dev);
            hip::sync(r.s);
        }
    }

    /// One decoder block (mixer + FFN) on every rank. Leaves its FFN partial pending (Pend::B).
    fn block(&self, sel: Sel, bt: &Batch, pend: Pend) {
        let hp = &self.hp;
        let (b, bi, g) = (bt.b, bt.b as i32, &bt.g);
        for r in &self.ranks {
            hip::set_device(r.dev);
            let (lw, kv) = match sel {
                Sel::Trunk(l) => (&r.w.layers[l], r.kv[l].as_ref()),
                Sel::Mtp => (&r.w.mtp.as_ref().unwrap().layer, r.mtp_kv.as_ref()),
            };
            let dm = &r.dims;
            self.norm(r, pend, &lw.attn_norm, b, false);
            match &lw.mixer {
                Mixer::Gdn { win, wout, conv_w, dt_bias, ssm_a, norm_w } => {
                    let Sel::Trunk(l) = sel else { unreachable!() };
                    let gb = r.gdn[l].as_ref().unwrap();
                    self.gemv(r, win, r.y.ptr(), b);
                    unsafe {
                        hip::pk_gdn_conv(r.s, r.y.ptr(), dm.gdn_nin as i32, gb.mix.ptr(), gb.qkv.ptr(), R as i32, conv_w.ptr(), dt_bias.ptr(), ssm_a.ptr(), dm.gk as i32, dm.gv as i32, hp.eps, bi, g);
                        hip::pk_gdn_step(r.s, gb.state.ptr(), gb.qkv.ptr(), R as i32, r.y.ptr(), dm.gdn_nin as i32, norm_w.ptr(), dm.gk as i32, dm.gv as i32, hp.eps, r.xq.ptr(), r.xd.ptr(), g);
                    }
                    self.t(r, "gdn");
                    self.gemv(r, wout, r.pa.ptr(), b);
                }
                Mixer::Attn { win, wout, q_norm, k_norm } => {
                    let kv = kv.unwrap();
                    self.gemv(r, win, r.y.ptr(), b);
                    unsafe {
                        hip::pk_attn_prep(r.s, r.y.ptr(), dm.attn_nin as i32, dm.hq as i32, dm.hkv as i32, 0, (dm.hq * 2 * hp.hd) as i32, q_norm.ptr(), k_norm.ptr(), hp.eps, hp.theta_scale, hp.n_rot as i32, r.qr.ptr(), kv.k.ptr(), kv.v.ptr(), bi, g);
                        hip::pk_attn(r.s, r.qr.ptr(), r.y.ptr(), dm.attn_nin as i32, dm.hq as i32, dm.hkv as i32, kv.k.ptr(), kv.v.ptr(), r.scores.ptr(), self.cap as i64, r.xq.ptr(), r.xd.ptr(), bi, g);
                    }
                    self.t(r, "attn");
                    self.gemv(r, wout, r.pa.ptr(), b);
                }
            }
        }
        self.exchange(true, b);
        if let (true, Sel::Trunk(l)) = (self.dump, sel) {
            if l > 0 {
                self.dump_h(&format!("l_out-{}", l - 1), b);
            }
        }
        for r in &self.ranks {
            hip::set_device(r.dev);
            let lw = match sel {
                Sel::Trunk(l) => &r.w.layers[l],
                Sel::Mtp => &r.w.mtp.as_ref().unwrap().layer,
            };
            self.norm(r, Pend::A, &lw.post_norm, b, false);
            self.gemv(r, &lw.upgate, r.y.ptr(), b);
            unsafe { hip::pk_ffn_act(r.s, r.y.ptr(), r.dims.ff as i32, bi, r.xq.ptr(), r.xd.ptr()) };
            self.t(r, "ffn-act");
            self.gemv(r, &lw.down, r.pb.ptr(), b);
        }
        self.exchange(false, b);
    }

    /// Queue the final norm (normed hidden rows stay in xf), the LM head and the per-rank argmax.
    /// The MTP head scores only its pruned vocabulary.
    fn head_launch(&self, mtp: bool, b: usize) {
        for r in &self.ranks {
            hip::set_device(r.dev);
            let (w, head) = match (mtp, &r.w.mtp) {
                (true, Some(m)) => (&m.head_norm, &m.draft_head),
                _ => (&r.w.out_norm, &r.w.head),
            };
            self.norm(r, Pend::B, w, b, true);
            self.gemv(r, head, r.logits.ptr(), b);
            unsafe { hip::pk_argmax(r.s, r.logits.ptr(), head.rows as i64, head.rows as i64, b as i32, r.best.ptr()) };
        }
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

    /// Trunk head: greedy pick per row, merged across ranks on the host. Synchronises.
    fn head(&self, b: usize) -> Vec<RowOut> {
        self.head_launch(false, b);
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
    fn head_draft(&self, b: usize, step: usize) {
        self.head_launch(true, b);
        if let Some(x) = &self.xchg {
            let seq = x.seq.get() + 1;
            x.seq.set(seq);
            let area = BMAX * self.hp.d;
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

    fn kv_groups(&self, i: usize, slot: usize, g: &mut Groups) {
        g.slot[i] = slot as i32;
        g.kv_dir[i] = if slot == 0 { 1 } else { -1 };
        g.kv_base[i] = if slot == 0 { 0 } else { self.cap as i64 - 1 };
    }

    /// MTP input for the rows in `tokens`/`hin`: h = eh_proj([enorm(embed(tok)) | hnorm(hin)]).
    fn mtp_input(&self, r: &Rank, b: usize) {
        let m = r.w.mtp.as_ref().unwrap();
        let (d, bi) = (self.hp.d as i32, b as i32);
        let null = std::ptr::null();
        unsafe {
            hip::pk_embed(r.s, r.w.embd.qs.ptr(), r.w.embd.d.ptr(), r.tokens.ptr(), r.emb.ptr(), d, bi);
            hip::pk_norm_quant(r.s, r.emb.ptr(), null, null, m.enorm.ptr(), self.hp.eps, d, bi, r.xq.ptr(), r.xd.ptr(), 2 * d, std::ptr::null_mut());
            hip::pk_norm_quant(r.s, r.hin.ptr(), null, null, m.hnorm.ptr(), self.hp.eps, d, bi, r.xq.ptr().add(d as usize), r.xd.ptr().add(d as usize / 32), 2 * d, std::ptr::null_mut());
        }
        self.gemv(r, &m.eh_proj, r.h.ptr(), b);
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
                let (start, len, slot) = (bt.g.start[i] as usize, bt.g.len[i] as usize, bt.g.slot[i] as usize);
                unsafe {
                    hip::copy_dev(r.hin.ptr().add(start * d), r.hprev.ptr().add(slot * d), d, r.s);
                    if len > 1 {
                        hip::copy_dev(r.hin.ptr().add((start + 1) * d), r.xf.ptr().add(start * d), (len - 1) * d, r.s);
                    }
                }
            }
            self.mtp_input(r, bt.b);
            self.norm(r, Pend::None, &m.layer.attn_norm, bt.b, false);
            // K and V rows sit after the q rows of the fused "in" matrix.
            let q_rows = dm.hq * 2 * hp.hd;
            let kv_rows = 2 * dm.hkv * hp.hd;
            unsafe {
                hip::pk_gemm(r.s, win.qs.ptr().add(q_rows * win.cols), win.d.ptr().add(q_rows * win.cols / 32), r.xq.ptr(), r.xd.ptr(), r.y.ptr(), kv_rows as i64, win.cols as i32, bt.b as i32, kv_rows as i32);
                hip::pk_attn_prep(r.s, r.y.ptr(), kv_rows as i32, dm.hq as i32, dm.hkv as i32, dm.hq as i32, 0, q_norm.ptr(), k_norm.ptr(), hp.eps, hp.theta_scale, hp.n_rot as i32, r.qr.ptr(), kv.k.ptr(), kv.v.ptr(), bt.b as i32, &bt.g);
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
            let bt = Batch { g, b };
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
                self.mtp_input(r, b);
            }
            self.block(Sel::Mtp, &bt, Pend::None);
            self.head_draft(b, step);
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

    /// Run one batch through the trunk. `rows` lists (slot, tokens) groups; returns the greedy
    /// next token per row. Each group either commits (prefill) or stays pending; pending groups
    /// must be `accept`ed before any other call.
    pub fn step(&mut self, rows: &[(usize, &[u32], bool)]) -> Vec<RowOut> {
        let mut g = Groups { n: rows.len() as i32, ..Default::default() };
        let mut tokens: Vec<i32> = Vec::new();
        for (i, &(slot, toks, commit)) in rows.iter().enumerate() {
            let s = &self.seqs[slot];
            assert!(!s.awaiting_accept, "slot {slot}: accept() the previous step first");
            assert!(!toks.is_empty() && toks.len() <= R);
            g.start[i] = tokens.len() as i32;
            g.len[i] = toks.len() as i32;
            g.cur[i] = 1 - s.cur;
            g.hist_off[i] = s.last_acc as i32;
            g.nacc[i] = if s.last_committed { 0 } else { s.last_acc as i32 };
            g.commit[i] = commit as i32;
            g.pos0[i] = s.len as i32;
            self.kv_groups(i, slot, &mut g);
            tokens.extend(toks.iter().map(|&t| t as i32));
        }
        let b = tokens.len();
        assert!(b <= BMAX);
        assert!(self.seqs.iter().map(|s| s.len).sum::<usize>() + b <= self.cap, "KV pool full");
        let bt = Batch { g, b };
        let t_step = Instant::now();

        for r in &self.ranks {
            hip::set_device(r.dev);
            r.tokens.upload(&tokens, r.s);
            unsafe { hip::pk_embed(r.s, r.w.embd.qs.ptr(), r.w.embd.d.ptr(), r.tokens.ptr(), r.h.ptr(), self.hp.d as i32, b as i32) };
        }
        let mut pend = Pend::None;
        for l in 0..self.hp.n_layer {
            self.block(Sel::Trunk(l), &bt, pend);
            pend = Pend::B;
        }
        let t_enq = Instant::now();
        let out = self.head(b);
        if std::env::var_os("FCL_TRACE").is_some() {
            eprintln!("step b={b}: enqueue {:.2} ms, then wait {:.2} ms", (t_enq - t_step).as_secs_f64() * 1e3, t_enq.elapsed().as_secs_f64() * 1e3);
        }
        if self.dump {
            self.dump_h(&format!("l_out-{}", self.hp.n_layer - 1), b);
        }
        self.mtp_catchup(&bt);

        for (i, &(slot, toks, commit)) in rows.iter().enumerate() {
            let s = &mut self.seqs[slot];
            s.cur = 1 - s.cur;
            s.last_rows = toks.len();
            s.last_row0 = bt.g.start[i] as usize;
            s.last_committed = commit;
            s.last_acc = if commit { toks.len() } else { 0 };
            s.awaiting_accept = !commit;
            if commit {
                s.len += toks.len();
                let row = s.last_row0 + toks.len() - 1;
                self.save_hprev(slot, row);
            }
        }
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
