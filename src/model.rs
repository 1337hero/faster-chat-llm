//! Model geometry, the tensor-parallel shard plan, and streaming weight upload.
//!
//! Q8_0 tensors go to the card as stored: the loader preads raw `block_q8_0` rows into pinned
//! memory, copies them to a device staging buffer, and a kernel de-interleaves scales from
//! quants. The host never holds more than one staging chunk.
//!
//! Shard plan at TP-n (rank r), chosen so every rank-local op is exact and each sublayer ends in
//! one sum across ranks:
//!   - "in" matrices (hidden → heads) are split by output rows, whole heads per rank;
//!   - "out" matrices (heads → hidden) are split by input columns (always 32-aligned);
//!   - norms and the embedding table are replicated; the LM head is split by vocab rows.

use std::time::{Duration, Instant};

use crate::gguf::Gguf;
use crate::gguf::dequant::dequantise;
use crate::gguf::types::GgmlType;
use crate::hip::{self, ColRanges, DevBuf, Pinned, Stream};

#[derive(Clone, Debug)]
pub struct HParams {
    pub n_layer: usize,
    pub d: usize,
    pub ff: usize,
    pub vocab: usize,
    /// Full attention: query heads, KV heads, head dim.
    pub hq: usize,
    pub hkv: usize,
    pub hd: usize,
    /// Gated DeltaNet: key heads, value heads, head dim.
    pub gk: usize,
    pub gv: usize,
    pub gs: usize,
    pub eps: f32,
    pub n_rot: usize,
    pub theta_scale: f32,
    pub full_interval: usize,
    /// MTP (nextn) blocks stored after the trunk; 0 or 1.
    pub n_mtp: usize,
}

impl HParams {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let arch = g.str_of("general.architecture").unwrap_or("");
        if arch != "qwen35" {
            return Err(format!("unsupported architecture {arch:?} (need qwen35)"));
        }
        let u = |k: &str| g.u64_of(&format!("qwen35.{k}")).map(|v| v as usize).ok_or(format!("missing qwen35.{k}"));
        let f = |k: &str| g.f32_of(&format!("qwen35.{k}")).ok_or(format!("missing qwen35.{k}"));
        let n_rot = u("rope.dimension_count")?;
        let hp = Self {
            n_layer: u("block_count")? - u("nextn_predict_layers").unwrap_or(0),
            d: u("embedding_length")?,
            ff: u("feed_forward_length")?,
            vocab: g.tensor("token_embd.weight").map_err(|e| e.to_string())?.dims[1] as usize,
            hq: u("attention.head_count")?,
            hkv: u("attention.head_count_kv")?,
            hd: u("attention.key_length")?,
            gk: u("ssm.group_count")?,
            gv: u("ssm.time_step_rank")?,
            gs: u("ssm.state_size")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_rot,
            theta_scale: f("rope.freq_base")?.powf(-2.0 / n_rot as f32),
            full_interval: u("full_attention_interval")?,
            n_mtp: u("nextn_predict_layers").unwrap_or(0),
        };
        // The kernels hard-code these head sizes.
        if hp.hd != 256 || hp.gs != 128 || hp.d % 256 != 0 || hp.ff % 256 != 0 || hp.n_mtp > 1 {
            return Err(format!("geometry not supported by the kernels: {hp:?}"));
        }
        Ok(hp)
    }

    pub fn is_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.full_interval == 0
    }
}

/// One rank's slice of the model.
#[derive(Clone, Copy, Debug)]
pub struct Shard {
    pub rank: usize,
    pub n: usize,
}

/// Rank-local sizes.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub d: usize,
    pub ff: usize,
    pub vocab: usize,
    pub vocab0: usize,
    pub hq: usize,
    pub hkv: usize,
    pub gk: usize,
    pub gv: usize,
    /// GDN conv channels and "in" row count.
    pub ch: usize,
    pub gdn_nin: usize,
    pub attn_nin: usize,
}

type Rows = Vec<(usize, usize)>;

impl Shard {
    pub fn dims(&self, hp: &HParams) -> Dims {
        let n = self.n;
        let (gk, gv) = (hp.gk / n, hp.gv / n);
        let ch = (2 * gk + gv) * hp.gs;
        Dims {
            d: hp.d,
            ff: hp.ff / n,
            vocab: hp.vocab / n,
            vocab0: self.rank * (hp.vocab / n),
            hq: hp.hq / n,
            hkv: hp.hkv / n,
            gk,
            gv,
            ch,
            gdn_nin: ch + gv * hp.gs + 2 * gv,
            attn_nin: (hp.hq / n) * 2 * hp.hd + 2 * (hp.hkv / n) * hp.hd,
        }
    }

    fn split(&self, total: usize) -> (usize, usize) {
        assert!(total % self.n == 0);
        (self.rank * (total / self.n), total / self.n)
    }

    /// Global GDN value heads owned by this rank, in local order. The GGUF stores value heads
    /// tiled over key heads (v head h pairs with k head h % gk), so a rank that owns a
    /// contiguous run of key heads owns gv/gk runs of value heads.
    fn gdn_v_runs(&self, hp: &HParams) -> Rows {
        let (k0, kn) = self.split(hp.gk);
        (0..hp.gv / hp.gk).map(|m| (m * hp.gk + k0, kn)).collect()
    }

    /// Rows of `attn_qkv` (and channels of the conv) owned by this rank: q heads, k heads, v heads.
    fn gdn_qkv_rows(&self, hp: &HParams) -> Rows {
        let s = hp.gs;
        let (k0, kn) = self.split(hp.gk);
        let mut rows = vec![(k0 * s, kn * s), ((hp.gk + k0) * s, kn * s)];
        rows.extend(self.gdn_v_runs(hp).iter().map(|&(h, n)| ((2 * hp.gk + h) * s, n * s)));
        rows
    }
}

/// A Q8_0 matrix on a card: `qs[rows][cols]` int8 and f16 scale bits `d[rows/16][cols/32][16]`.
pub struct Mat {
    pub qs: DevBuf<i8>,
    pub d: DevBuf<u16>,
    pub rows: usize,
    pub cols: usize,
}

pub enum Mixer {
    Gdn { win: Mat, wout: Mat, conv_w: DevBuf<f32>, dt_bias: DevBuf<f32>, ssm_a: DevBuf<f32>, norm_w: DevBuf<f32> },
    Attn { win: Mat, wout: Mat, q_norm: DevBuf<f32>, k_norm: DevBuf<f32> },
}

pub struct LayerW {
    pub attn_norm: DevBuf<f32>,
    pub post_norm: DevBuf<f32>,
    pub mixer: Mixer,
    /// Rows [0, ff) = ffn_up, [ff, 2ff) = ffn_gate.
    pub upgate: Mat,
    pub down: Mat,
}

/// The GGUF's multi-token-prediction head: one extra attention block fed by
/// `eh_proj([enorm(embed(token)) | hnorm(h_prev)])`, sharing the trunk's embedding and LM head.
pub struct MtpW {
    /// LM head rows for the first `draft_vocab` token ids only (this rank's share). Drafting
    /// over a subset is free of risk: every draft is checked against the full head.
    pub draft_head: Mat,
    pub draft_v0: usize,
    pub eh_proj: Mat,
    pub enorm: DevBuf<f32>,
    pub hnorm: DevBuf<f32>,
    pub head_norm: DevBuf<f32>,
    pub layer: LayerW,
}

pub struct Weights {
    pub embd: Mat,
    pub head: Mat,
    pub out_norm: DevBuf<f32>,
    pub layers: Vec<LayerW>,
    pub mtp: Option<MtpW>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct LoadStats {
    pub bytes: u64,
    pub read: Duration,
    pub upload: Duration,
}

/// Streams tensors from the GGUF to the current device.
pub struct Loader<'a> {
    g: &'a Gguf,
    s: Stream,
    pinned: Pinned,
    stage: DevBuf<u8>,
    pub stats: LoadStats,
}

const STAGE_BYTES: usize = 256 << 20;

impl<'a> Loader<'a> {
    pub fn new(g: &'a Gguf, s: Stream) -> Self {
        Self { g, s, pinned: Pinned::new(STAGE_BYTES), stage: DevBuf::new(STAGE_BYTES), stats: LoadStats::default() }
    }

    /// Build one device matrix from row ranges of one or more Q8_0 tensors (stacked in order),
    /// keeping only the given column block ranges (empty = all columns).
    fn mat(&mut self, parts: &[(String, Rows)], col_blocks: &[(usize, usize)]) -> Result<Mat, String> {
        let first = self.g.tensor(&parts[0].0).map_err(|e| e.to_string())?;
        let nb_src = first.dims[0] as usize / 32;
        let all = [(0, nb_src)];
        let col_blocks = if col_blocks.is_empty() { &all[..] } else { col_blocks };
        let nb_dst: usize = col_blocks.iter().map(|c| c.1).sum();
        let rows: usize = parts.iter().flat_map(|p| p.1.iter()).map(|r| r.1).sum();
        let mut cr = ColRanges { n: col_blocks.len() as i32, ..Default::default() };
        for (i, &(s, l)) in col_blocks.iter().enumerate() {
            cr.start[i] = s as i32;
            cr.len[i] = l as i32;
        }
        assert!(rows % 16 == 0, "{}: {rows} rows is not a whole number of 16-row tiles", parts[0].0);
        let mat = Mat { qs: DevBuf::new(rows * nb_dst * 32), d: DevBuf::new(rows * nb_dst), rows, cols: nb_dst * 32 };

        let row_bytes = nb_src * 34;
        let rows_per_chunk = STAGE_BYTES / row_bytes;
        let mut dst_row = 0usize;
        for (name, ranges) in parts {
            let t = self.g.tensor(name).map_err(|e| e.to_string())?;
            if t.ty != GgmlType::Q8_0 {
                return Err(format!("{name}: {} not supported for matrices (only Q8_0)", t.ty));
            }
            if t.dims[0] as usize != nb_src * 32 {
                return Err(format!("{name}: column count differs from {}", parts[0].0));
            }
            let base = self.g.data_offset + t.offset;
            for &(r0, rn) in ranges {
                assert!(r0 + rn <= t.dims[1] as usize, "{name}: rows {r0}+{rn} out of range");
                let mut done = 0;
                while done < rn {
                    let n = (rn - done).min(rows_per_chunk);
                    let bytes = n * row_bytes;
                    let t0 = Instant::now();
                    self.g
                        .pread(base + ((r0 + done) * row_bytes) as u64, &mut self.pinned.as_mut_slice()[..bytes])
                        .map_err(|e| format!("{name}: {e}"))?;
                    let t1 = Instant::now();
                    hip::upload_raw(self.stage.ptr(), self.pinned.as_ptr(), bytes, self.s);
                    unsafe {
                        hip::pk_unpack_q8(self.s, self.stage.ptr(), mat.qs.ptr(), mat.d.ptr(), n as i64, nb_src as i32, nb_dst as i32, dst_row as i64, &cr);
                    }
                    hip::sync(self.s);
                    self.stats.read += t1 - t0;
                    self.stats.upload += t1.elapsed();
                    self.stats.bytes += bytes as u64;
                    done += n;
                    dst_row += n;
                }
            }
        }
        Ok(mat)
    }

    fn f32s(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.g.tensor(name).map_err(|e| e.to_string())?;
        let raw = self.g.read_raw(name).map_err(|e| e.to_string())?;
        let mut out = vec![0f32; t.n_elements as usize];
        dequantise(t.ty, &raw, &mut out);
        self.stats.bytes += raw.len() as u64;
        Ok(out)
    }

    fn vec(&mut self, name: &str) -> Result<DevBuf<f32>, String> {
        let v = self.f32s(name)?;
        Ok(DevBuf::from_slice(&v, self.s))
    }

    /// Upload elements of an F32 tensor selected by ranges of `width`-element groups.
    fn gather(&mut self, name: &str, ranges: &Rows, width: usize) -> Result<DevBuf<f32>, String> {
        let v = self.f32s(name)?;
        let mut out = Vec::new();
        for &(s, n) in ranges {
            out.extend_from_slice(&v[s * width..(s + n) * width]);
        }
        Ok(DevBuf::from_slice(&out, self.s))
    }

    fn layer(&mut self, hp: &HParams, sh: Shard, l: usize, is_attn: bool) -> Result<LayerW, String> {
        let blk = |r: (usize, usize)| (r.0 / 32, r.1 / 32);
        let n = |s: &str| format!("blk.{l}.{s}");
        let mixer = if is_attn {
            let hd = hp.hd;
            let (q0, qn) = sh.split(hp.hq);
            let (k0, kn) = sh.split(hp.hkv);
            Mixer::Attn {
                win: self.mat(
                    &[
                        (n("attn_q.weight"), vec![(q0 * 2 * hd, qn * 2 * hd)]),
                        (n("attn_k.weight"), vec![(k0 * hd, kn * hd)]),
                        (n("attn_v.weight"), vec![(k0 * hd, kn * hd)]),
                    ],
                    &[],
                )?,
                wout: self.mat(&[(n("attn_output.weight"), vec![(0, hp.d)])], &[blk((q0 * hd, qn * hd))])?,
                q_norm: self.vec(&n("attn_q_norm.weight"))?,
                k_norm: self.vec(&n("attn_k_norm.weight"))?,
            }
        } else {
            let s = hp.gs;
            let v_runs = sh.gdn_v_runs(hp);
            let v_rows: Rows = v_runs.iter().map(|&(h, c)| (h * s, c * s)).collect();
            let qkv_rows = sh.gdn_qkv_rows(hp);
            let v_cols: Vec<_> = v_rows.iter().map(|&r| blk(r)).collect();
            Mixer::Gdn {
                win: self.mat(
                    &[
                        (n("attn_qkv.weight"), qkv_rows.clone()),
                        (n("attn_gate.weight"), v_rows.clone()),
                        (n("ssm_beta.weight"), v_runs.clone()),
                        (n("ssm_alpha.weight"), v_runs.clone()),
                    ],
                    &[],
                )?,
                wout: self.mat(&[(n("ssm_out.weight"), vec![(0, hp.d)])], &v_cols)?,
                conv_w: self.gather(&n("ssm_conv1d.weight"), &qkv_rows, 4)?,
                dt_bias: self.gather(&n("ssm_dt.bias"), &v_runs, 1)?,
                ssm_a: self.gather(&n("ssm_a"), &v_runs, 1)?,
                norm_w: self.vec(&n("ssm_norm.weight"))?,
            }
        };
        let f = sh.split(hp.ff);
        Ok(LayerW {
            attn_norm: self.vec(&n("attn_norm.weight"))?,
            post_norm: self.vec(&n("post_attention_norm.weight"))?,
            mixer,
            upgate: self.mat(&[(n("ffn_up.weight"), vec![f]), (n("ffn_gate.weight"), vec![f])], &[])?,
            down: self.mat(&[(n("ffn_down.weight"), vec![(0, hp.d)])], &[blk(f)])?,
        })
    }

    pub fn load(&mut self, hp: &HParams, sh: Shard, draft_vocab: usize, mut progress: impl FnMut(usize)) -> Result<Weights, String> {
        let embd = self.mat(&[("token_embd.weight".into(), vec![(0, hp.vocab)])], &[])?;
        let head = self.mat(&[("output.weight".into(), vec![sh.split(hp.vocab)])], &[])?;
        let out_norm = self.vec("output_norm.weight")?;
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            layers.push(self.layer(hp, sh, l, hp.is_attn(l))?);
            progress(l);
        }
        let mtp = if hp.n_mtp > 0 {
            let l = hp.n_layer;
            let n = |s: &str| format!("blk.{l}.nextn.{s}.weight");
            let per_rank = draft_vocab.min(hp.vocab) / sh.n / 16 * 16;
            let draft_v0 = sh.rank * per_rank;
            Some(MtpW {
                draft_head: self.mat(&[("output.weight".into(), vec![(draft_v0, per_rank)])], &[])?,
                draft_v0,
                eh_proj: self.mat(&[(n("eh_proj"), vec![(0, hp.d)])], &[])?,
                enorm: self.vec(&n("enorm"))?,
                hnorm: self.vec(&n("hnorm"))?,
                head_norm: self.vec(&n("shared_head_norm"))?,
                layer: self.layer(hp, sh, l, true)?,
            })
        } else {
            None
        };
        Ok(Weights { embd, head, out_norm, layers, mtp })
    }
}
