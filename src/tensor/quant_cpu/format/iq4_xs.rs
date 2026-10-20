//! iq4_xs - 256 values in 136 bytes, four-bit codes into the non-linear table of iq4_nl.
//!
//! The block carries one f16 scale and a six-bit scale per thirty-two values, split between a
//! low nibble array and a shared 16-bit word of high pairs.

use super::iq4_nl::LEVELS;
use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIq4Xs {
    pub(crate) d: f16,
    pub(crate) scales_h: u16,
    pub(crate) scales_l: [u8; QK_K / 64],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIq4Xs>() == 136);

impl BlockFormat for BlockIq4Xs {
    const DTYPE: GgmlDType = GgmlDType::Iq4Xs;
    const BLOCK_LEN: usize = QK_K;
    type ActivationBlock = BlockQ8K32;

    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            scales_h: 0,
            scales_l: [0; QK_K / 64],
            qs: [0x88; QK_K / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        for (x, out) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
            let d = x.d.to_f32();
            for (ib, sub) in out.chunks_exact_mut(32).enumerate() {
                let low = (x.scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
                let high = ((x.scales_h >> (2 * ib)) & 3) as u8;
                let dl = d * ((low | (high << 4)) as i32 - 32) as f32;
                let q = &x.qs[16 * ib..16 * ib + 16];
                for j in 0..16 {
                    sub[j] = dl * LEVELS[(q[j] & 0x0F) as usize] as f32;
                    sub[j + 16] = dl * LEVELS[(q[j] >> 4) as usize] as f32;
                }
            }
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        encode(xs, ys, None, 0);
    }

    fn quantize_guided(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        encode(xs, ys, Some(imatrix_weights), n_per_row);
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `dot_avx2` enables.
            return unsafe { dot_avx2(xs, ys) };
        }
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let mut block = 0f32;
                for ib in 0..QK_K / 32 {
                    let packed: &[u8; 16] = x.qs[16 * ib..16 * ib + 16].try_into().unwrap();
                    let dot =
                        super::iq_read::dot_iq4_run_scalar(packed, &y.qs[32 * ib..32 * ib + 32]);
                    block += super::iq_read::Runs::run_scale(x, ib) * dot as f32 * y.d[ib].to_f32();
                }
                block
            })
            .sum()
    }

    /// [`Self::dot`] over several activations, each run's weights read from the table once.
    fn dot_many(xs: &[Self], ys: &[&[Self::ActivationBlock]], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `dot_many_avx2` enables.
            return unsafe { dot_many_avx2(xs, ys, out) };
        }
        for (o, y) in out.iter_mut().zip(ys) {
            *o = Self::dot(xs, y);
        }
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        super::iq_read::dot_by_decoding(xs, ys)
    }
}

impl super::iq_read::Runs for BlockIq4Xs {
    fn run_scale(&self, ib: usize) -> f32 {
        let low = (self.scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
        let high = ((self.scales_h >> (2 * ib)) & 3) as u8;
        self.d.to_f32() * ((low | (high << 4)) as i32 - 32) as f32
    }

    fn run_values(&self, ib: usize) -> [i8; 32] {
        let q = &self.qs[16 * ib..16 * ib + 16];
        let mut w = [0i8; 32];
        for j in 0..16 {
            w[j] = LEVELS[(q[j] & 0x0F) as usize];
            w[j + 16] = LEVELS[(q[j] >> 4) as usize];
        }
        w
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn dot_avx2(xs: &[BlockIq4Xs], ys: &[<BlockIq4Xs as BlockFormat>::ActivationBlock]) -> f32 {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            let mut block = 0f32;
            for ib in 0..QK_K / 32 {
                let packed: &[u8; 16] = x.qs[16 * ib..16 * ib + 16].try_into().unwrap();
                let dot = unsafe {
                    super::iq_read::dot_iq4_run_avx2(packed, &y.qs[32 * ib..32 * ib + 32])
                };
                block += super::iq_read::Runs::run_scale(x, ib) * dot as f32 * y.d[ib].to_f32();
            }
            block
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn dot_many_avx2(
    xs: &[BlockIq4Xs],
    ys: &[&[<BlockIq4Xs as BlockFormat>::ActivationBlock]],
    out: &mut [f32],
) {
    unsafe {
        for (ys, out) in ys
            .chunks(super::iq_read::MANY)
            .zip(out.chunks_mut(super::iq_read::MANY))
        {
            let mut totals = [0f32; super::iq_read::MANY];
            for (bi, x) in xs.iter().enumerate() {
                let mut blocks = [0f32; super::iq_read::MANY];
                for ib in 0..QK_K / 32 {
                    let packed: &[u8; 16] = x.qs[16 * ib..16 * ib + 16].try_into().unwrap();
                    let (w, magnitude) = super::iq_read::iq4_run_weights(packed);
                    let scale = super::iq_read::Runs::run_scale(x, ib);
                    for (block, y) in blocks.iter_mut().zip(ys) {
                        let y = &y[bi];
                        let dot =
                            super::iq_read::iq4_run_dot(w, magnitude, &y.qs[32 * ib..32 * ib + 32]);
                        *block += scale * dot as f32 * y.d[ib].to_f32();
                    }
                }
                for (total, block) in totals.iter_mut().zip(blocks) {
                    *total += block;
                }
            }
            out.copy_from_slice(&totals[..out.len()]);
        }
    }
}

/// Quantise whole blocks, optionally against an importance matrix: one weight per column of a row
/// `n_per_row` long.
///
/// ggml's `quantize_iq4_xs` (MIT): each run of thirty-two is fitted alone, as an iq4_nl block is;
/// the eight scales are then stored as six-bit steps of the block scale, which puts the largest
/// of them on the step -32, and each run's values are levelled against its scale as stored.
fn encode(xs: &[f32], ys: &mut [BlockIq4Xs], imatrix: Option<&[f32]>, n_per_row: usize) {
    use super::iq4_nl::{level_index, run_scale, run_weights, spread};
    use rayon::prelude::*;
    const RUNS: usize = QK_K / 32;
    ys.par_iter_mut()
        .zip(xs.par_chunks_exact(QK_K))
        .enumerate()
        .for_each(|(idx, (block, x))| {
            let row = imatrix.map(|m| {
                let base = (idx % (n_per_row / QK_K)) * QK_K;
                &m[base..base + QK_K]
            });
            let sigma2 = spread(x);
            let mut scales = [0f32; RUNS];
            let (mut widest, mut max_scale) = (0f32, 0f32);
            for (ib, scale) in scales.iter_mut().enumerate() {
                let xb = &x[32 * ib..32 * ib + 32];
                let w = run_weights(xb, row.map(|r| &r[32 * ib..32 * ib + 32]), sigma2);
                *scale = run_scale(xb, &w);
                if scale.abs() > widest {
                    (widest, max_scale) = (scale.abs(), *scale);
                }
            }
            *block = BlockIq4Xs::zeros();
            // An all-zero block stores +0, whichever sign the division would give it.
            block.d = f16::from_f32(if max_scale == 0.0 {
                0.0
            } else {
                -max_scale / 32.0
            });
            let d = block.d.to_f32();
            let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
            for (ib, &scale) in scales.iter().enumerate() {
                let l = nearest_int(inverse * scale).clamp(-32, 31);
                let dl = d * l as f32;
                let step = if dl != 0.0 { 1.0 / dl } else { 0.0 };
                let xb = &x[32 * ib..32 * ib + 32];
                for j in 0..16 {
                    block.qs[16 * ib + j] =
                        level_index(step * xb[j]) | level_index(step * xb[j + 16]) << 4;
                }
                let l = (l + 32) as u8;
                block.scales_l[ib / 2] |= (l & 0x0F) << (4 * (ib % 2));
                block.scales_h |= ((l >> 4) as u16) << (2 * ib);
            }
        });
}
