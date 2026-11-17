//! q5_0 - thirty-two values, one scale, five bits each.
//!
//! `y = (q - 16) . d`, the same shape as q4_0 with one more bit of code. The fifth bit does
//! not fit in the nibble array, so it lives in a separate 32-bit word where bit `v` belongs to
//! value `v`: the code for value `v` is `qs[v mod 16]`'s nibble with that bit stuck on top.
//!
//! The nibble packing is q4_0's: byte `j` holds value `j` low and value `j + 16` high, so the
//! two halves of a block are interleaved rather than consecutive.

use super::*;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 2 + 4 + QK5_0 / 2);

/// Code `c` means `c - Q5_0_BIAS`; five bits reach 0..31, so -16..15.
const Q5_0_BIAS: i32 = 16;

impl BlockFormat for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLOCK_LEN: usize = QK5_0;
    type ActivationBlock = BlockQ8_0;

    /// The all-zero block: every level at its zero code, scale zero. Dequantises to
    /// zeros, which is what the callers that pre-allocate a quantised buffer need.
    fn zeros() -> Self {
        Self {
            d: f16::ZERO,
            qh: [0; 4],
            qs: [0; QK5_0 / 2],
        }
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) {
        const HALF: usize = QK5_0 / 2;
        for (block, y) in blocks_with_output(xs, ys) {
            let d = block.d.to_f32();
            let qh = u32::from_le_bytes(block.qh);
            let top = |v: usize| ((qh >> v) & 1) << 4;
            let (low, high) = y.split_at_mut(HALF);
            for (j, ((lo, hi), &packed)) in low.iter_mut().zip(high).zip(&block.qs).enumerate() {
                let a = ((packed & 0x0F) as u32 | top(j)) as i32 - Q5_0_BIAS;
                let b = ((packed >> 4) as u32 | top(j + HALF)) as i32 - Q5_0_BIAS;
                *lo = a as f32 * d;
                *hi = b as f32 * d;
            }
        }
    }

    /// ggml's importance-weighted fit: see `legacy_guided`.
    fn quantize_guided(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        const HALF: usize = QK5_0 / 2;
        let fitted = super::legacy_guided::signed_blocks(xs, imatrix_weights, n_per_row, Q5_0_BIAS);
        for (block, (d, c)) in ys.iter_mut().zip(fitted) {
            block.d = d;
            let mut qh = 0u32;
            for j in 0..HALF {
                let (a, b) = (c[j] as u32, c[j + HALF] as u32);
                block.qs[j] = (a & 0x0F) as u8 | ((b & 0x0F) as u8) << 4;
                qh |= (a >> 4) << j | (b >> 4) << (j + HALF);
            }
            block.qh = qh.to_le_bytes();
        }
    }

    fn quantize(xs: &[f32], ys: &mut [Self]) {
        const HALF: usize = QK5_0 / 2;
        for (block, x) in blocks_with_input(xs, ys) {
            // As in q4_0: the extreme keeps its sign because only -16 is representable.
            let extreme = x
                .iter()
                .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            let d = extreme / -(Q5_0_BIAS as f32);
            block.d = f16::from_f32(d);

            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            let code = |v: f32| ((v * inv + Q5_0_BIAS as f32).round() as i32).clamp(0, 31) as u32;
            let mut qh = 0u32;
            let (low, high) = x.split_at(HALF);
            for (j, ((&lo, &hi), packed)) in
                low.iter().zip(high).zip(block.qs.iter_mut()).enumerate()
            {
                let (a, b) = (code(lo), code(hi));
                *packed = (a & 0x0F) as u8 | (((b & 0x0F) as u8) << 4);
                qh |= (a >> 4) << j;
                qh |= (b >> 4) << (j + HALF);
            }
            block.qh = qh.to_le_bytes();
        }
    }

    fn dot(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx512() {
            return avx512::block32::vec_dot_q5_0_q8_0(xs, ys);
        }
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            return avx::vec_dot_q5_0_q8_0(xs, ys);
        }

        #[allow(unreachable_code)]
        Self::dot_scalar(xs, ys)
    }

    fn dot_scalar(xs: &[Self], ys: &[Self::ActivationBlock]) -> f32 {
        const HALF: usize = QK5_0 / 2;
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let qh = u32::from_le_bytes(x.qh);
                let top = |v: usize| ((qh >> v) & 1) << 4;
                let (low, high) = y.qs.split_at(HALF);
                let dot: i32 =
                    x.qs.iter()
                        .zip(low)
                        .zip(high)
                        .enumerate()
                        .map(|(j, ((&packed, &a), &b))| {
                            let p = ((packed & 0x0F) as u32 | top(j)) as i32 - Q5_0_BIAS;
                            let q = ((packed >> 4) as u32 | top(j + HALF)) as i32 - Q5_0_BIAS;
                            p * a as i32 + q * b as i32
                        })
                        .sum();
                dot as f32 * x.d.to_f32() * y.d.to_f32()
            })
            .sum()
    }
}

impl BlockQ5_0 {
    /// The `Q8_0` block holding the same 32 values: the same scale, each code less its bias.
    pub(crate) fn widened(&self) -> BlockQ8_0 {
        const HALF: usize = QK5_0 / 2;
        let qh = u32::from_le_bytes(self.qh);
        let mut qs = [0i8; QK8_0];
        for (j, &packed) in self.qs.iter().enumerate() {
            let top = |v: usize| (((qh >> v) & 1) << 4) as i32;
            qs[j] = ((packed & 0x0F) as i32 + top(j) - Q5_0_BIAS) as i8;
            qs[j + HALF] = ((packed >> 4) as i32 + top(j + HALF) - Q5_0_BIAS) as i8;
        }
        BlockQ8_0 { d: self.d, qs }
    }
}

impl ExactDot for BlockQ5_0 {
    type Act = BlockQ8K32;
    fn dot_exact(xs: &[Self], ys: &[BlockQ8K32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() {
            // SAFETY: dispatch_avx2 implies the CPU has every feature `dot_q5_0` enables.
            return unsafe { super::exact_runs::dot_q5_0(xs, ys) };
        }
        dot_exact_scalar(xs, ys)
    }
}

/// [`ExactDot::dot_exact`] one value at a time: the steps the vector kernel is held to.
/// Run `r` of an activation block is weight block `r` of its 8, fewer in a row's last block.
pub(super) fn dot_exact_scalar(xs: &[BlockQ5_0], ys: &[BlockQ8K32]) -> f32 {
    let mut total = -0.0f32;
    for (x8, y) in xs.chunks(QK_K / QK5_0).zip(ys) {
        total += exact_block_share(y, x8.len(), |r| {
            (
                x8[r].d.to_f32(),
                exact_run_sum(&x8[r], &y.qs[QK5_0 * r..QK5_0 * (r + 1)]),
            )
        });
    }
    total
}

/// A block's integer sum against its 32 activation codes: the 5-bit codes less their bias
/// against the codes.
fn exact_run_sum(x: &BlockQ5_0, acts: &[i8]) -> i32 {
    let qh = u32::from_le_bytes(x.qh);
    acts.iter()
        .enumerate()
        .map(|(v, &a)| {
            let nibble = if v < QK5_0 / 2 {
                x.qs[v] & 0x0F
            } else {
                x.qs[v - QK5_0 / 2] >> 4
            };
            let q = (nibble as u32 | (((qh >> v) & 1) << 4)) as i32 - Q5_0_BIAS;
            q * a as i32
        })
        .sum()
}
