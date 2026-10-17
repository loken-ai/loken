//! The dot of a grid format (magnitudes from a codebook, a sign per value, an odd integer scale
//! per run) against one to `MANY` activations at once: each run of a block is decoded once into
//! registers and dotted against every activation, whose accumulators stay in registers across
//! the whole row.
//!
//! Each activation's result is the one-activation dot to the bit: per run, the integer pairs
//! under the run's odd scale, converted to floats and fused into the accumulator under the
//! block's scale times the activation run's scale, then one horizontal sum at the end.

use super::*;

/// A format read as runs of thirty-two grid magnitudes under sign bits and odd integer scales.
#[cfg(target_arch = "x86_64")]
pub(super) trait GridRun: BlockFormat<ActivationBlock = BlockQ8K32> {
    /// The block's float scale, which each run's integer sum takes with the activation's.
    fn run_unit(&self) -> f32;
    /// What a block's runs share, decoded once for all of them.
    type Block;
    /// Decodes what the block's runs share.
    ///
    /// # Safety
    /// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
    unsafe fn block(&self) -> Self::Block;
    /// Run `IB`: its magnitudes as unsigned bytes, its signs as bytes of -1 (negated) and +1,
    /// and its odd scales as sixteen i16 lanes, one for each pair of adjacent products.
    ///
    /// # Safety
    /// The caller runs with AVX2, FMA, F16C and BMI2 enabled, `block` decoded from `self`.
    unsafe fn grid_run<const IB: usize>(
        &self,
        block: &Self::Block,
    ) -> (
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m256i,
    );
    /// The block's eight runs of sign bits, run `ib` at word `ib`, bit `i` set where value `i`
    /// is negated: what [`Self::grid_run`] spreads into bytes.
    fn sign_words(&self, block: &Self::Block) -> *const u32;
}

/// The odd scales of one run out of a block's sixteen i16 lanes: lane `ib` of each 128-bit
/// half spread over that half.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn run_scale_lanes(
    scales: std::arch::x86_64::__m256i,
    ib: usize,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let pick = (2 * ib) as i16 | ((2 * ib + 1) as i16) << 8;
    _mm256_shuffle_epi8(scales, _mm256_set1_epi16(pick))
}

/// Eight entries of a table of 32-bit values, entry `codes[8 IB + l]` in lane `l`.
///
/// The codes are read four at a time and split in registers, which leaves the load ports to the
/// entries; each entry is read by a broadcast load and blended into place: no shuffle port, and
/// no gather, slow on a processor whose microcode blocks the leak of its data. Written out so
/// that the compiler does not fold the loads back into lane inserts or a gather.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn table_u32_by_bytes<const IB: usize>(
    table: &[u32; 256],
    codes: *const u8,
) -> std::arch::x86_64::__m256i {
    let out: std::arch::x86_64::__m256i;
    unsafe {
        std::arch::asm!(
            "mov eax, dword ptr [{c} + {o}]",
            "movzx ecx, al",
            "movzx edx, ah",
            "vpbroadcastd {lo}, dword ptr [{t} + rcx*4]",
            "vpbroadcastd {x}, dword ptr [{t} + rdx*4]",
            "vpblendd {lo}, {lo}, {x}, 0x02",
            "shr eax, 16",
            "movzx ecx, al",
            "movzx edx, ah",
            "vpbroadcastd {x}, dword ptr [{t} + rcx*4]",
            "vpblendd {lo}, {lo}, {x}, 0x04",
            "vpbroadcastd {x}, dword ptr [{t} + rdx*4]",
            "vpblendd {lo}, {lo}, {x}, 0x08",
            "mov eax, dword ptr [{c} + {o} + 4]",
            "movzx ecx, al",
            "movzx edx, ah",
            "vpbroadcastd {hi}, dword ptr [{t} + rcx*4]",
            "vpbroadcastd {x}, dword ptr [{t} + rdx*4]",
            "vpblendd {hi}, {hi}, {x}, 0x20",
            "shr eax, 16",
            "movzx ecx, al",
            "movzx edx, ah",
            "vpbroadcastd {x}, dword ptr [{t} + rcx*4]",
            "vpblendd {hi}, {hi}, {x}, 0x40",
            "vpbroadcastd {x}, dword ptr [{t} + rdx*4]",
            "vpblendd {hi}, {hi}, {x}, 0x80",
            "vpblendd {lo}, {lo}, {hi}, 0xF0",
            c = in(reg) codes,
            o = const 8 * IB,
            t = in(reg) table.as_ptr(),
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            lo = out(ymm_reg) out,
            hi = out(ymm_reg) _,
            x = out(ymm_reg) _,
            options(pure, readonly, nostack),
        );
    }
    out
}

/// Four entries of a table of 64-bit values, entry `codes[4 IB + l]` in quarter `l`, read as
/// [`table_u32_by_bytes`] reads its entries.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn table_u64_by_words<const IB: usize>(
    table: &[u64; 1024],
    codes: *const u16,
) -> std::arch::x86_64::__m256i {
    let out: std::arch::x86_64::__m256i;
    unsafe {
        std::arch::asm!(
            "movzx {i:e}, word ptr [{c} + {o}]",
            "vpbroadcastq {lo}, qword ptr [{t} + {i}*8]",
            "movzx {j:e}, word ptr [{c} + {o} + 4]",
            "vpbroadcastq {hi}, qword ptr [{t} + {j}*8]",
            "movzx {i:e}, word ptr [{c} + {o} + 2]",
            "vpbroadcastq {x}, qword ptr [{t} + {i}*8]",
            "vpblendd {lo}, {lo}, {x}, 0x0C",
            "movzx {j:e}, word ptr [{c} + {o} + 6]",
            "vpbroadcastq {x}, qword ptr [{t} + {j}*8]",
            "vpblendd {hi}, {hi}, {x}, 0xC0",
            "vpblendd {lo}, {lo}, {hi}, 0xF0",
            c = in(reg) codes,
            o = const 8 * IB,
            t = in(reg) table.as_ptr(),
            i = out(reg) _,
            j = out(reg) _,
            lo = out(ymm_reg) out,
            hi = out(ymm_reg) _,
            x = out(ymm_reg) _,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    out
}

/// The 32-bit value `p[IB]` in every lane, read by a broadcast load.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn broadcast_u32<const IB: usize>(p: *const u32) -> std::arch::x86_64::__m256i {
    let out: std::arch::x86_64::__m256i;
    unsafe {
        std::arch::asm!(
            "vpbroadcastd {r}, dword ptr [{p} + {o}]",
            p = in(reg) p,
            o = const 4 * IB,
            r = out(ymm_reg) out,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    out
}

/// The float `p[IB]` in every lane, read by a broadcast load.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn broadcast_f32<const IB: usize>(p: *const f32) -> std::arch::x86_64::__m256 {
    let out: std::arch::x86_64::__m256;
    unsafe {
        std::arch::asm!(
            "vbroadcastss {r}, dword ptr [{p} + {o}]",
            p = in(reg) p,
            o = const 4 * IB,
            r = out(ymm_reg) out,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    out
}

/// A run's 32 sign bits, broadcast to every 32-bit lane, as bytes of -1 (bit set) and +1:
/// byte `i` takes bit `i % 8` of byte `i / 8`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn run_negate_spread(
    signs: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let spread = _mm256_shuffle_epi8(
        signs,
        _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        ),
    );
    let bit = _mm256_set1_epi64x(i64::from_le_bytes([1, 2, 4, 8, 16, 32, 64, 128]));
    // 0xFF where the bit is set, which with a one ORed in is -1, and +1 elsewhere.
    _mm256_or_si256(
        _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit), bit),
        _mm256_set1_epi8(1),
    )
}

/// Run `IB` of block `bi` against `N` activations, fused into their accumulators.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn run_rows<B: GridRun, const N: usize, const IB: usize>(
    x: &B,
    block: &B::Block,
    ys: &[&[BlockQ8K32]; N],
    bi: usize,
    scales: &[[f32; QK_K / 32]; N],
    totals: &mut [std::arch::x86_64::__m256; N],
) {
    unsafe {
        use std::arch::x86_64::*;
        let (mags, negate, scale) = x.grid_run::<IB>(block);
        for t in 0..N {
            let a = _mm256_loadu_si256(ys[t][bi].qs.as_ptr().add(32 * IB) as *const __m256i);
            let pairs = _mm256_maddubs_epi16(mags, _mm256_sign_epi8(a, negate));
            totals[t] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, scale)),
                broadcast_f32::<IB>(scales[t].as_ptr()),
                totals[t],
            );
        }
    }
}

/// `out[t] = dot(xs, ys[t])` for exactly `N` activations.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_rows<B: GridRun, const N: usize>(xs: &[B], ys: &[&[BlockQ8K32]], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        let blocks = ys.iter().fold(xs.len(), |n, y| n.min(y.len()));
        let ys: [&[BlockQ8K32]; N] = std::array::from_fn(|t| &ys[t][..blocks]);
        let mut totals = [_mm256_setzero_ps(); N];
        for (bi, x) in xs[..blocks].iter().enumerate() {
            let unit = _mm256_set1_ps(x.run_unit());
            let block = x.block();
            // Each activation's eight run scales times the block's, read back one at a time.
            let mut scales = [[0f32; QK_K / 32]; N];
            for t in 0..N {
                let d = _mm_loadu_si128(ys[t][bi].d.as_ptr() as *const __m128i);
                _mm256_storeu_ps(
                    scales[t].as_mut_ptr(),
                    _mm256_mul_ps(unit, _mm256_cvtph_ps(d)),
                );
            }
            // Each run spelled out, so that what selects its lanes is a constant.
            const _: () = assert!(QK_K / 32 == 8);
            run_rows::<B, N, 0>(x, &block, &ys, bi, &scales, &mut totals);
            run_rows::<B, N, 1>(x, &block, &ys, bi, &scales, &mut totals);
            run_rows::<B, N, 2>(x, &block, &ys, bi, &scales, &mut totals);
            run_rows::<B, N, 3>(x, &block, &ys, bi, &scales, &mut totals);
            run_rows::<B, N, 4>(x, &block, &ys, bi, &scales, &mut totals);
            run_rows::<B, N, 5>(x, &block, &ys, bi, &scales, &mut totals);
            run_rows::<B, N, 6>(x, &block, &ys, bi, &scales, &mut totals);
            run_rows::<B, N, 7>(x, &block, &ys, bi, &scales, &mut totals);
        }
        for t in 0..N {
            out[t] = super::iq_read::sum_ps(totals[t]);
        }
    }
}

/// `out[t] = dot(xs, ys[t])` for every activation, `MANY` at a time.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn dot_many_grid<B: GridRun>(xs: &[B], ys: &[&[BlockQ8K32]], out: &mut [f32]) {
    unsafe {
        let many = super::iq_read::MANY;
        for (ys, out) in ys.chunks(many).zip(out.chunks_mut(many)) {
            match ys.len() {
                1 => dot_rows::<B, 1>(xs, ys, out),
                2 => dot_rows::<B, 2>(xs, ys, out),
                3 => dot_rows::<B, 3>(xs, ys, out),
                4 => dot_rows::<B, 4>(xs, ys, out),
                5 => dot_rows::<B, 5>(xs, ys, out),
                6 => dot_rows::<B, 6>(xs, ys, out),
                7 => dot_rows::<B, 7>(xs, ys, out),
                _ => dot_rows::<B, 8>(xs, ys, out),
            }
        }
    }
}

/// `dot(xs, y)` for one activation.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C and BMI2 enabled.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn dot_grid<B: GridRun>(xs: &[B], y: &[BlockQ8K32]) -> f32 {
    let mut out = [0f32];
    unsafe { dot_rows::<B, 1>(xs, &[y], &mut out) };
    out[0]
}

/// Runs `IB` and `IC = IB + 1` of block `bi` against `N` activations, both runs' products in
/// one 512-bit pass and each run fused into the accumulators in turn, as [`run_rows`] fuses it.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn run_pair_rows_avx512<B: GridRun, const N: usize, const IB: usize, const IC: usize>(
    x: &B,
    block: &B::Block,
    ys: &[&[BlockQ8K32]; N],
    bi: usize,
    scales: &[[f32; QK_K / 32]; N],
    totals: &mut [std::arch::x86_64::__m256; N],
) {
    unsafe {
        use std::arch::x86_64::*;
        const { assert!(IC == IB + 1) };
        let (mags_b, _, scale_b) = x.grid_run::<IB>(block);
        let (mags_c, _, scale_c) = x.grid_run::<IC>(block);
        let mags = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(mags_b), mags_c);
        let scale = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(scale_b), scale_c);
        let negate: __mmask64 = (x.sign_words(block).add(IB) as *const u64).read_unaligned();
        for t in 0..N {
            let a = _mm512_loadu_si512(ys[t][bi].qs.as_ptr().add(32 * IB) as *const _);
            // 0 - a where the sign bit is set: the byte negation of `_mm256_sign_epi8`.
            let signed = _mm512_mask_sub_epi8(a, negate, _mm512_setzero_si512(), a);
            let pairs = _mm512_maddubs_epi16(mags, signed);
            let sums = _mm512_cvtepi32_ps(_mm512_madd_epi16(pairs, scale));
            totals[t] = _mm256_fmadd_ps(
                _mm512_castps512_ps256(sums),
                broadcast_f32::<IB>(scales[t].as_ptr()),
                totals[t],
            );
            totals[t] = _mm256_fmadd_ps(
                _mm512_extractf32x8_ps::<1>(sums),
                broadcast_f32::<IC>(scales[t].as_ptr()),
                totals[t],
            );
        }
    }
}

/// [`dot_rows`] with the runs taken two at a time in 512-bit registers: the same products and
/// the same accumulation order, so the same bits.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_rows_avx512<B: GridRun, const N: usize>(
    xs: &[B],
    ys: &[&[BlockQ8K32]],
    out: &mut [f32],
) {
    unsafe {
        use std::arch::x86_64::*;
        let blocks = ys.iter().fold(xs.len(), |n, y| n.min(y.len()));
        let ys: [&[BlockQ8K32]; N] = std::array::from_fn(|t| &ys[t][..blocks]);
        let mut totals = [_mm256_setzero_ps(); N];
        for (bi, x) in xs[..blocks].iter().enumerate() {
            let unit = _mm256_set1_ps(x.run_unit());
            let block = x.block();
            let mut scales = [[0f32; QK_K / 32]; N];
            for t in 0..N {
                let d = _mm_loadu_si128(ys[t][bi].d.as_ptr() as *const __m128i);
                _mm256_storeu_ps(
                    scales[t].as_mut_ptr(),
                    _mm256_mul_ps(unit, _mm256_cvtph_ps(d)),
                );
            }
            const _: () = assert!(QK_K / 32 == 8);
            run_pair_rows_avx512::<B, N, 0, 1>(x, &block, &ys, bi, &scales, &mut totals);
            run_pair_rows_avx512::<B, N, 2, 3>(x, &block, &ys, bi, &scales, &mut totals);
            run_pair_rows_avx512::<B, N, 4, 5>(x, &block, &ys, bi, &scales, &mut totals);
            run_pair_rows_avx512::<B, N, 6, 7>(x, &block, &ys, bi, &scales, &mut totals);
        }
        for t in 0..N {
            out[t] = super::iq_read::sum_ps(totals[t]);
        }
    }
}

/// [`dot_many_grid`] in 512-bit registers, to the bit.
///
/// # Safety
/// The caller runs with AVX2, FMA, F16C, BMI2, AVX512F, AVX512BW, AVX512VL and AVX512DQ enabled.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn dot_many_grid_avx512<B: GridRun>(
    xs: &[B],
    ys: &[&[BlockQ8K32]],
    out: &mut [f32],
) {
    unsafe {
        let many = super::iq_read::MANY;
        for (ys, out) in ys.chunks(many).zip(out.chunks_mut(many)) {
            match ys.len() {
                1 => dot_rows_avx512::<B, 1>(xs, ys, out),
                2 => dot_rows_avx512::<B, 2>(xs, ys, out),
                3 => dot_rows_avx512::<B, 3>(xs, ys, out),
                4 => dot_rows_avx512::<B, 4>(xs, ys, out),
                5 => dot_rows_avx512::<B, 5>(xs, ys, out),
                6 => dot_rows_avx512::<B, 6>(xs, ys, out),
                7 => dot_rows_avx512::<B, 7>(xs, ys, out),
                _ => dot_rows_avx512::<B, 8>(xs, ys, out),
            }
        }
    }
}

/// [`dot_grid`] in 512-bit registers, to the bit.
///
/// # Safety
/// As [`dot_many_grid_avx512`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn dot_grid_avx512<B: GridRun>(xs: &[B], y: &[BlockQ8K32]) -> f32 {
    let mut out = [0f32];
    unsafe { dot_rows_avx512::<B, 1>(xs, &[y], &mut out) };
    out[0]
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(super) mod tests {
    //! The grid kernels against the run-by-run kernels they replace, kept as the reference, and
    //! the 512-bit kernels against the 256-bit ones.

    use super::super::iq2_s::{self, BlockIq2S};
    use super::super::iq3_xxs::{self, BlockIq3Xxs};
    use super::*;

    /// A deterministic stream of pseudo-random bytes.
    pub(in super::super) struct Bytes(pub(in super::super) u64);
    impl Bytes {
        pub(in super::super) fn next(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
    }

    /// `count` blocks of random codes, signs and scales under a small random block scale, the
    /// block's leading f16: every byte pattern of these formats is a valid block.
    pub(in super::super) fn weights<B: BlockFormat>(count: usize, rng: &mut Bytes) -> Vec<B> {
        let size = std::mem::size_of::<B>();
        let mut bytes: Vec<u8> = (0..count * size).map(|_| rng.next()).collect();
        for chunk in bytes.chunks_exact_mut(size) {
            let d = 1e-3 * (1.0 + rng.next() as f32 / 64.0);
            chunk[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        }
        let mut out = vec![B::zeros(); count];
        // SAFETY: the blocks are plain bytes of the same total size.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len())
        };
        out
    }

    fn activations(count: usize, rng: &mut Bytes) -> Vec<BlockQ8K32> {
        (0..count)
            .map(|_| {
                let mut b = BlockQ8K32::zeros();
                for d in b.d.iter_mut() {
                    *d = half::f16::from_f32(0.01 + rng.next() as f32 / 1024.0);
                }
                for q in b.qs.iter_mut() {
                    *q = (rng.next() as i8).max(-127);
                }
                b
            })
            .collect()
    }

    /// A kernel over one activation.
    pub(in super::super) type One<B> = fn(&[B], &[<B as BlockFormat>::ActivationBlock]) -> f32;
    /// A kernel over every activation of a pass.
    pub(in super::super) type Many<B> =
        fn(&[B], &[&[<B as BlockFormat>::ActivationBlock]], &mut [f32]);
    /// What makes `count` activation blocks.
    pub(in super::super) type Activations<B> =
        fn(usize, &mut Bytes) -> Vec<<B as BlockFormat>::ActivationBlock>;

    /// `got` against `want` to the bit, for 1 to `2 MANY + 1` activations over `blocks` blocks,
    /// and the one-activation kernels against the many.
    pub(in super::super) fn same_bits<B: BlockFormat>(
        name: &str,
        blocks: usize,
        activations: Activations<B>,
        want: (One<B>, Many<B>),
        got: (One<B>, Many<B>),
    ) {
        let mut rng = Bytes(0x5EED);
        for rows in 1..=2 * super::super::iq_read::MANY + 1 {
            for _ in 0..8 {
                let xs = weights::<B>(blocks, &mut rng);
                let acts: Vec<Vec<B::ActivationBlock>> =
                    (0..rows).map(|_| activations(blocks, &mut rng)).collect();
                let ys: Vec<&[B::ActivationBlock]> = acts.iter().map(|a| a.as_slice()).collect();
                let mut expected = vec![0f32; rows];
                let mut actual = vec![0f32; rows];
                want.1(&xs, &ys, &mut expected);
                got.1(&xs, &ys, &mut actual);
                for t in 0..rows {
                    assert_eq!(
                        actual[t].to_bits(),
                        expected[t].to_bits(),
                        "{name} {rows} rows, row {t}"
                    );
                    let one = got.0(&xs, ys[t]);
                    assert_eq!(one.to_bits(), want.0(&xs, ys[t]).to_bits(), "{name} dot");
                    assert_eq!(
                        one.to_bits(),
                        expected[t].to_bits(),
                        "{name} dot against many"
                    );
                }
            }
        }
    }

    /// Blocks a row of the tested matrices.
    const TESTED_BLOCKS: usize = 10;

    #[test]
    fn grid_kernels_equal_the_run_kernels_to_the_bit() {
        if !crate::cpu::dispatch_avx2() {
            return;
        }
        same_bits::<BlockIq3Xxs>(
            "iq3_xxs",
            TESTED_BLOCKS,
            activations,
            (
                |x, y| unsafe { iq3_xxs::reference_dot_avx2(x, y) },
                |x, y, o| unsafe { iq3_xxs::reference_dot_many_avx2(x, y, o) },
            ),
            (BlockIq3Xxs::dot, BlockIq3Xxs::dot_many),
        );
        same_bits::<BlockIq2S>(
            "iq2_s",
            TESTED_BLOCKS,
            activations,
            (
                |x, y| unsafe { iq2_s::reference_dot_avx2(x, y) },
                |x, y, o| unsafe { iq2_s::reference_dot_many_avx2(x, y, o) },
            ),
            (BlockIq2S::dot, BlockIq2S::dot_many),
        );
    }

    /// The 512-bit grid kernels against the 256-bit ones, on a CPU that has AVX-512.
    #[test]
    fn grid_kernels_in_512_bits_equal_the_256_bit_ones_to_the_bit() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        same_bits::<BlockIq3Xxs>(
            "iq3_xxs",
            TESTED_BLOCKS,
            activations,
            (
                |x, y| {
                    let mut o = [0f32];
                    unsafe { iq3_xxs::dot_many_avx2(x, &[y], &mut o) };
                    o[0]
                },
                |x, y, o| unsafe { iq3_xxs::dot_many_avx2(x, y, o) },
            ),
            (
                |x, y| {
                    let mut o = [0f32];
                    unsafe { iq3_xxs::dot_many_avx512(x, &[y], &mut o) };
                    o[0]
                },
                |x, y, o| unsafe { iq3_xxs::dot_many_avx512(x, y, o) },
            ),
        );
        same_bits::<BlockIq2S>(
            "iq2_s",
            TESTED_BLOCKS,
            activations,
            (
                |x, y| {
                    let mut o = [0f32];
                    unsafe { iq2_s::dot_many_avx2(x, &[y], &mut o) };
                    o[0]
                },
                |x, y, o| unsafe { iq2_s::dot_many_avx2(x, y, o) },
            ),
            (
                |x, y| {
                    let mut o = [0f32];
                    unsafe { iq2_s::dot_many_avx512(x, &[y], &mut o) };
                    o[0]
                },
                |x, y, o| unsafe { iq2_s::dot_many_avx512(x, y, o) },
            ),
        );
    }

    /// The matrices a pass computes.
    pub(in super::super) struct Shape {
        /// Output rows of a matrix.
        pub(in super::super) rows: usize,
        /// Blocks a row.
        pub(in super::super) blocks: usize,
        /// Experts a pass computes.
        pub(in super::super) experts: usize,
        /// Matrices an expert computes.
        pub(in super::super) matrices: usize,
    }

    /// Weight bytes the passes cycle through, past the last-level cache so that each pass reads
    /// its weights from memory as the host does.
    const COLD_BYTES: usize = 256 << 20;

    /// What a pass of `shape` costs under `before` and `after`, in bytes of weight per second,
    /// for 1, 2 and 4 activations, on one thread and through the pool.
    pub(in super::super) fn measure<B: BlockFormat + Sync>(
        name: &str,
        shape: &Shape,
        activations: Activations<B>,
        before: (&str, Many<B>),
        after: (&str, Many<B>),
    ) where
        B::ActivationBlock: Sync,
    {
        let mut rng = Bytes(0xC01D);
        let pass_bytes =
            shape.experts * shape.matrices * shape.rows * shape.blocks * std::mem::size_of::<B>();
        // Per pass, per expert, its matrices.
        let passes: Vec<Vec<Vec<Vec<B>>>> = (0..COLD_BYTES.div_ceil(pass_bytes))
            .map(|_| {
                (0..shape.experts)
                    .map(|_| {
                        (0..shape.matrices)
                            .map(|_| weights::<B>(shape.rows * shape.blocks, &mut rng))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        for rows in [1, 2, 4] {
            let acts: Vec<Vec<B::ActivationBlock>> = (0..rows)
                .map(|_| activations(shape.blocks, &mut rng))
                .collect();
            let ys: Vec<&[B::ActivationBlock]> = acts.iter().map(|a| a.as_slice()).collect();
            let unit = |pass: &Vec<Vec<Vec<B>>>, r: usize, kernel: Many<B>| {
                let mut out = [0f32; super::super::iq_read::MANY];
                let mut sink = 0f32;
                for matrix in pass.iter().flatten() {
                    let row = r * shape.blocks..(r + 1) * shape.blocks;
                    kernel(&matrix[row], &ys, &mut out[..rows]);
                    sink += out[0];
                }
                sink
            };
            let rate = |kernel: Many<B>, pooled: bool| {
                let rounds = 4;
                let start = std::time::Instant::now();
                let mut sink = 0f32;
                for _ in 0..rounds {
                    for pass in &passes {
                        if pooled {
                            let mut units = vec![0f32; shape.rows];
                            crate::tensor::quant_cpu::pool_for_each_mut(&mut units, 8, &|r, o| {
                                *o = unit(pass, r, kernel)
                            });
                            sink += units[0];
                        } else {
                            sink += (0..shape.rows).map(|r| unit(pass, r, kernel)).sum::<f32>();
                        }
                    }
                }
                std::hint::black_box(sink);
                (rounds * passes.len() * pass_bytes) as f64 / start.elapsed().as_secs_f64() / 1e9
            };
            for pooled in [false, true] {
                // The best of alternating trials, so that a pause of the machine weighs on neither.
                let (mut slow, mut fast) = (0f64, 0f64);
                for _ in 0..3 {
                    slow = slow.max(rate(before.1, pooled));
                    fast = fast.max(rate(after.1, pooled));
                }
                eprintln!(
                    "{name:8} {rows} rows {:6}: {} {slow:6.2} -> {} {fast:6.2} GB/s of weights",
                    if pooled { "pool" } else { "1 thr" },
                    before.0,
                    after.0,
                );
            }
        }
    }

    /// The gate and up projections of an expert whose hidden width is ten blocks, three experts
    /// a pass.
    const GATE_UP: Shape = Shape {
        rows: 640,
        blocks: 10,
        experts: 3,
        matrices: 2,
    };

    /// What the gate and up projections of a pass cost, before and after, in bytes of weight per
    /// second. Not a gate.
    #[test]
    #[ignore = "a measurement: cargo test --release --lib grid_rows_throughput -- --ignored --nocapture"]
    fn grid_rows_throughput() {
        if !crate::cpu::dispatch_avx2() {
            return;
        }
        measure::<BlockIq3Xxs>(
            "iq3_xxs",
            &GATE_UP,
            activations,
            ("runs", |x, y, o| unsafe {
                if y.len() == 1 {
                    o[0] = iq3_xxs::reference_dot_avx2(x, y[0])
                } else {
                    iq3_xxs::reference_dot_many_avx2(x, y, o)
                }
            }),
            ("grid", |x, y, o| unsafe { iq3_xxs::dot_many_avx2(x, y, o) }),
        );
        measure::<BlockIq2S>(
            "iq2_s",
            &GATE_UP,
            activations,
            ("runs", |x, y, o| unsafe {
                if y.len() == 1 {
                    o[0] = iq2_s::reference_dot_avx2(x, y[0])
                } else {
                    iq2_s::reference_dot_many_avx2(x, y, o)
                }
            }),
            ("grid", |x, y, o| unsafe { iq2_s::dot_many_avx2(x, y, o) }),
        );
    }

    /// The grid kernels in 256 and in 512 bits, on a CPU that has AVX-512. Not a gate.
    #[test]
    #[ignore = "a measurement: cargo test --release --lib grid_rows_avx512_throughput -- --ignored --nocapture"]
    fn grid_rows_avx512_throughput() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        measure::<BlockIq3Xxs>(
            "iq3_xxs",
            &GATE_UP,
            activations,
            ("avx2", |x, y, o| unsafe { iq3_xxs::dot_many_avx2(x, y, o) }),
            ("avx512", |x, y, o| unsafe {
                iq3_xxs::dot_many_avx512(x, y, o)
            }),
        );
        measure::<BlockIq2S>(
            "iq2_s",
            &GATE_UP,
            activations,
            ("avx2", |x, y, o| unsafe { iq2_s::dot_many_avx2(x, y, o) }),
            ("avx512", |x, y, o| unsafe {
                iq2_s::dot_many_avx512(x, y, o)
            }),
        );
    }
}
