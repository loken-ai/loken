//! iq4_xs - 256 values in 136 bytes, four-bit codes into the non-linear table of iq4_nl.
//!
//! The block carries one f16 scale and a six-bit scale per thirty-two values, split between a
//! low nibble array and a shared 16-bit word of high pairs. Read-only: loken reads these files
//! and does not produce them.

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
    type ActivationBlock = BlockQ8K;

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

    fn quantize(_: &[f32], _: &mut [Self]) {
        panic!("iq4_xs is read, not written");
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
                    block += super::iq_read::Runs::run_scale(x, ib) * dot as f32;
                }
                block * y.d
            })
            .sum()
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        super::iq_read::dot_by_decoding(xs, ys)
    }
}

impl BlockIq4Xs {
    /// The signed six-bit scale of run `ib`, before the block's `d`.
    fn run_step(&self, ib: usize) -> i32 {
        let low = (self.scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
        let high = ((self.scales_h >> (2 * ib)) & 3) as u8;
        (low | (high << 4)) as i32 - 32
    }
}

impl super::iq_read::Runs for BlockIq4Xs {
    fn run_scale(&self, ib: usize) -> f32 {
        self.d.to_f32() * self.run_step(ib) as f32
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
            // One conversion per block, not one per run.
            let d = crate::tensor::quant_cpu::avx::f16c_to_f32(x.d);
            let mut block = 0f32;
            for ib in 0..QK_K / 32 {
                let packed: &[u8; 16] = x.qs[16 * ib..16 * ib + 16].try_into().unwrap();
                let dot = super::iq_read::dot_iq4_run_avx2(packed, &y.qs[32 * ib..32 * ib + 32]);
                block += d * x.run_step(ib) as f32 * dot as f32;
            }
            block * y.d
        })
        .sum()
}
