//! Block dequantisation. Q8_0 and F32/F16/BF16 only, matching what the loader ships.
//!
//! On the GPU path these blocks are never expanded: the Q8_0 dot product consumes
//! `{d: f16, qs: [i8; 32]}` directly. This module exists for the CPU reference path and
//! for verifying the loader against the checkpoint.

pub const QK8_0: usize = 32;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    /// Per-block scale.
    pub d: f32,
    pub qs: [i8; QK8_0],
}

impl BlockQ8_0 {
    pub fn from_bytes(b: &[u8; 34]) -> Self {
        let d = f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let mut qs = [0i8; QK8_0];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = b[2 + i] as i8;
        }
        Self { d, qs }
    }

    pub fn dot(&self, x: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), QK8_0);
        let mut sum = 0.0f32;
        for i in 0..QK8_0 {
            sum += self.qs[i] as f32 * x[i];
        }
        sum * self.d
    }
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        // subnormal (or zero)
        if frac == 0 {
            0.0
        } else {
            let m = frac as f32 * 2f32.powi(-24);
            m
        }
    } else if exp == 0x1f {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        let m = 1.0 + frac as f32 / 1024.0;
        m * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -out } else { out }
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Dequantise any supported type into `out` (row-major, `n_elements` long).
pub fn dequantise(ty: crate::gguf::types::GgmlType, src: &[u8], out: &mut [f32]) {
    use crate::gguf::types::GgmlType;
    match ty {
        GgmlType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f16_bits_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
            }
        }
        GgmlType::BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16_bits_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
            }
        }
        GgmlType::Q8_0 => {
            let nblocks = out.len() / QK8_0;
            for b in 0..nblocks {
                let raw: &[u8; 34] = src[b * 34..b * 34 + 34].try_into().unwrap();
                let blk = BlockQ8_0::from_bytes(raw);
                for i in 0..QK8_0 {
                    out[b * QK8_0 + i] = blk.qs[i] as f32 * blk.d;
                }
            }
        }
        other => panic!("dequantise: unsupported type {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_simple() {
        for (bits, want) in [
            (0x0000u16, 0.0f32),
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x3800, 0.5),
            (0x7bff, 65504.0),
        ] {
            assert_eq!(f16_bits_to_f32(bits), want, "bits {bits:#06x}");
        }
    }

    #[test]
    fn q8_0_block_dot_matches_expanded() {
        let mut raw = [0u8; 34];
        raw[0] = 0x00;
        raw[1] = 0x3c; // d = 1.0
        for i in 0..32 {
            raw[2 + i] = (i as i32 - 16) as u8;
        }
        let blk = BlockQ8_0::from_bytes(&raw);
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25).collect();
        let dot = blk.dot(&x);
        let mut deq = vec![0.0f32; 32];
        dequantise(crate::gguf::types::GgmlType::Q8_0, &raw, &mut deq);
        let manual: f32 = (0..32).map(|i| deq[i] * x[i]).sum();
        assert!((dot - manual).abs() < 1e-4, "{dot} vs {manual}");
    }
}
