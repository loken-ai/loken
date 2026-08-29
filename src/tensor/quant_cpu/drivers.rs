//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

// Transcribed kernels: the index arithmetic IS the layout, the argument lists are the
// reference's, and the `unsafe fn`s wrap intrinsics whose contract is the intrinsic's.
// Named rather than `clippy::all` so anything else here still gets reported.
//
// `modulo_one` is the odd one out: `with_blocks!` instantiates the scalar types too,
// whose BLOCK_LEN is 1, so every block-size guard reads as `len % 1` in those
// expansions - vacuous there, and exactly the check the block formats need.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    clippy::type_complexity,
    clippy::redundant_closure,
    clippy::modulo_one
)]

use super::*;

/// The half of [`matmul_q4k_plain`] that runs once the activation is already
/// quantised. Split out so several weights can be driven from ONE quantisation
/// of the same activation - at decode `q`, `k` and `v` all read the attention
/// norm, and `gate` and `up` both read the FFN norm, so quantising inside the
/// GEMV repeats that work once per projection. Same blocks, same routine, so a
/// shared quantisation is bit-identical to the per-call one.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
pub fn matmul_q4k_plain_pre(
    (k, n): (usize, usize),
    aq: &[BlockQ8K],
    rhs_x8: &[repack_q4k::BlockQ4Kx8],
    dst: &mut [f32],
) -> Result<()> {
    {
        let nb = k / QK_K;
        let groups = n / 8;
        debug_assert_eq!(rhs_x8.len(), groups * nb);
        debug_assert_eq!(aq.len(), nb);
        let pool = gemv_pool::pool();
        let nth = pool.threads.max(1);
        let chunk = groups.div_ceil(nth).max(1);
        let nchunks = groups.div_ceil(chunk);
        let dst_ptr = SendMutPtr(dst.as_mut_ptr());
        let aq_ref: &[BlockQ8K] = aq;
        pool.run(nchunks, &|c| {
            let dst_ptr = &dst_ptr;
            let g0 = c * chunk;
            let g1 = (g0 + chunk).min(groups);
            for g in g0..g1 {
                let bgrp = &rhs_x8[g * nb..(g + 1) * nb];
                let mut out8 = [0f32; 8];
                unsafe { repack_q4k::gemv_group_avx2_plain(bgrp, aq_ref, nb, &mut out8) };
                for (i, v) in out8.iter().enumerate() {
                    unsafe {
                        *dst_ptr.0.add(g * 8 + i) = *v;
                    }
                }
            }
        });
        Ok(())
    }
}

/// Decode GEMV over repacked Q4_K weights (`repack_q4k::BlockQ4Kx8`). `lhs` is
/// `[m, k]` f32; the activation is quantised once per row to Q8_K and each
/// 8-column group runs one wide dot. `n` must be a multiple of 8. Rayon-style
/// parallel over column groups via the GEMV pool, like `matmul`.
pub fn matmul_q4k_repacked(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_x8: &[repack_q4k::BlockQ4Kx8],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    let nb = k / QK_K;
    let groups = n / 8;
    debug_assert_eq!(rhs_x8.len(), groups * nb);
    let pool = gemv_pool::pool();

    // Activation -> Q8_K once per row (parallel for prefill, serial for decode).
    let mut lhs_b = vec![BlockQ8K::zeros(); m * nb];
    if m >= 4 {
        let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
        pool.run(m, &|row_idx| {
            let lhs_b_ptr = &lhs_b_ptr;
            let lhs_b_mut =
                unsafe { std::slice::from_raw_parts_mut(lhs_b_ptr.0.add(row_idx * nb), nb) };
            BlockQ8K::quantize(&lhs[row_idx * k..(row_idx + 1) * k], lhs_b_mut);
        });
    } else {
        for row_idx in 0..m {
            BlockQ8K::quantize(
                &lhs[row_idx * k..(row_idx + 1) * k],
                &mut lhs_b[row_idx * nb..(row_idx + 1) * nb],
            );
        }
    }
    let lhs_b = lhs_b.as_slice();

    // One chunk per (row, column-group); the pool absorbs scheduling jitter.
    let n_chunks = m * groups;
    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(n_chunks, &|chunk| {
        let dst_ptr = &dst_ptr;
        let row_idx = chunk / groups;
        let g = chunk % groups;
        let act = &lhs_b[row_idx * nb..(row_idx + 1) * nb];
        let bgrp = &rhs_x8[g * nb..(g + 1) * nb];
        let mut local = [0f32; 8];
        #[cfg(target_feature = "avx2")]
        unsafe {
            repack_q4k::gemv_group_avx2(bgrp, act, nb, &mut local);
        }
        #[cfg(not(target_feature = "avx2"))]
        repack_q4k::gemv_group_scalar(bgrp, act, nb, &mut local);
        // SAFETY: each chunk writes a disjoint [row, g*8..g*8+8] slice of dst.
        let out = unsafe { std::slice::from_raw_parts_mut(dst_ptr.0.add(row_idx * n + g * 8), 8) };
        out.copy_from_slice(&local);
    });
    Ok(())
}

thread_local! {
    /// Set by the model forward at entry when a SMALL model is prefilling. The
    /// activation quantize is bandwidth-bound and normally wants the physical
    /// pool - but a small model's activation is tiny, so the bandwidth cost is
    /// negligible while keeping the quantize on the SAME (logical) prefill pool
    /// as the matmul avoids waking the physical pool and paying its cross-pool
    /// idle-spin. Large models keep the quantize on the physical pool, where the
    /// bandwidth win outweighs the spin. Read only by [`quantize_act_q8k`].
    static SMALL_MODEL_PREFILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Flag the current thread's forward as a small-model prefill (see
/// [`SMALL_MODEL_PREFILL`]). Called once at the model forward entry; self-resets
/// because decode and large-model forwards pass `false`.
pub fn set_small_model_prefill(on: bool) {
    SMALL_MODEL_PREFILL.with(|c| c.set(on));
}

/// Quantize an `[m, k]` f32 activation to row-major Q8_K blocks (`m * k/256`),
/// row-parallel on the gemv pool. Projections that consume the same activation
/// (qkv from the attention input, gate+up from the FFN input) can quantize it
/// once here and feed every projection's `_prequant` GEMM, instead of each GEMM
/// re-quantizing the identical rows.
pub fn quantize_act_q8k(lhs: &[f32], m: usize, k: usize) -> Vec<BlockQ8K> {
    let nb = k / QK_K;
    let mut lhs_b = vec![BlockQ8K::zeros(); m * nb];
    let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
    // Bandwidth-bound (streams the activation), so it normally stays on the
    // physical-core pool - SMT siblings only contend on memory bandwidth. The
    // exception is a small model's prefill: its activation is tiny, so keeping
    // the quantize on the logical prefill pool (matching the matmul that
    // follows) to avoid waking the physical pool wins over the marginal
    // bandwidth. Large models keep the physical pool (routing them to logical
    // regressed prefill).
    let pool = if m >= 32 && SMALL_MODEL_PREFILL.with(|c| c.get()) {
        gemv_pool::prefill_pool()
    } else {
        gemv_pool::pool()
    };
    pool.run(m, &|row_idx| {
        let lhs_b_ptr = &lhs_b_ptr;
        let lhs_b_mut =
            unsafe { std::slice::from_raw_parts_mut(lhs_b_ptr.0.add(row_idx * nb), nb) };
        BlockQ8K::quantize(&lhs[row_idx * k..(row_idx + 1) * k], lhs_b_mut);
    });
    lhs_b
}

/// Pool for a tiled K-quant prefill matmul. The fat K-quant weight matrices
/// (large `n*k`) are compute-bound - the 6-bit scale/nibble decode keeps the ALU
/// busy - so their column-group GEMM hides memory latency behind SMT siblings and
/// gains from the logical-core [`prefill_pool`]. Narrow matrices, tiny row tiles
/// (`m < 32`, e.g. attention projections or short prompts) and the bandwidth-bound
/// 32-block quants stay on the physical-core [`pool`], where SMT only burns power.
/// Threshold measured: granite (Q4_K, `n*k`≈1.7e7) prefill +18% on logical, while
/// smaller K-quant weights (`n*k`≲4e6) see no gain.
pub(super) fn kquant_prefill_pool(m: usize, n: usize, k: usize) -> &'static gemv_pool::Pool {
    // Route ALL prefill (m>=32) K-quant GEMM to the logical-core prefill pool;
    // decode (m<32) stays on the physical pool. Combined with the small-head
    // attention GEMM also on this pool, tiny models run the whole forward on ONE
    // pool (no rayon idle-spin). Large-head models keep their attention on rayon,
    // so only their projection GEMM shares the prefill pool.
    let _ = (n, k);
    if m >= 32 {
        gemv_pool::prefill_pool()
    } else {
        gemv_pool::pool()
    }
}

pub fn matmul_q4k_repacked_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_x8: &[repack_q4k::BlockQ4Kx8],
    dst: &mut [f32],
) -> Result<()> {
    let lhs_b = quantize_act_q8k(lhs, m, k);
    matmul_q4k_repacked_tiled_prequant((m, k, n), &lhs_b, rhs_x8, dst)
}

/// Q4_K tiled prefill GEMM against a pre-quantized Q8_K activation (see
/// [`quantize_act_q8k`]). Identical to [`matmul_q4k_repacked_tiled`] minus the
/// internal quantize, so a caller sharing one activation across several
/// projections pays the quantize once.
pub fn matmul_q4k_repacked_tiled_prequant(
    (m, k, n): (usize, usize, usize),
    lhs_b: &[BlockQ8K],
    rhs_x8: &[repack_q4k::BlockQ4Kx8],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    let nb = k / QK_K;
    let groups = n / 8;
    debug_assert_eq!(rhs_x8.len(), groups * nb);
    debug_assert_eq!(lhs_b.len(), m * nb);
    let pool = kquant_prefill_pool(m, n, k);

    // One chunk per column group; each computes all M rows of its 8 columns,
    // tiling 8 rows per weight pass so a loaded weight nibble feeds 8 rows.
    //
    // Two costs shape the inner kernel, both measured on the isolated kernel:
    //  - The per-column scale/min vectors are widened out of the packed 6-bit
    //    scales by scalar inserts. They depend only on the weight group, so
    //    they are built once per group here and reused by every row tile  - 
    //    worthwhile as soon as a group has several tiles to amortize them over.
    //  - Inside a tile the integer partials are accumulated one row at a time.
    //    Holding all rows' partials at once exceeds the architectural vector
    //    register file and turns every k-step into spill traffic; re-decoding
    //    the weight nibbles per row is much cheaper than that.
    // Both keep the same dots, so results are unchanged.
    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    // Enough row tiles per group to amortize the one-off scale widening.
    #[cfg(target_feature = "avx2")]
    let hoist_scales = m >= 32;
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_x8[g * nb..(g + 1) * nb];
        #[cfg(target_feature = "avx2")]
        let gs = if hoist_scales {
            unsafe { repack_q4k::prep_group_scales(bgrp, nb) }
        } else {
            Vec::new()
        };
        let mut r = 0;
        while r < m {
            let mt = (m - r).min(8);
            let mut local = [0f32; 64];
            #[cfg(target_feature = "avx2")]
            {
                let mut act_refs: [&[BlockQ8K]; 8] = [&lhs_b[0..0]; 8];
                for (t, slot) in act_refs.iter_mut().enumerate().take(mt) {
                    *slot = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                }
                unsafe {
                    if hoist_scales {
                        repack_q4k::gemm_group_avx2_pre_rt::<1>(
                            bgrp,
                            &act_refs[..mt],
                            nb,
                            &gs,
                            &mut local,
                        );
                    } else {
                        repack_q4k::gemm_group_avx2_rt::<1>(bgrp, &act_refs[..mt], nb, &mut local);
                    }
                }
            }
            #[cfg(not(target_feature = "avx2"))]
            for t in 0..mt {
                let act = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                repack_q4k::gemv_group_scalar(bgrp, act, nb, &mut local[t * 8..t * 8 + 8]);
            }
            for t in 0..mt {
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.0.add((r + t) * n + g * 8), 8)
                };
                out.copy_from_slice(&local[t * 8..t * 8 + 8]);
            }
            r += mt;
        }
    });
    Ok(())
}

/// Prefill GEMM for Q6_K weights repacked into `[n/8][k/256]` `BlockQ6Kx8`.
/// Mirrors `matmul_q4k_repacked_tiled`: quantize the activation to Q8_K once,
/// then one pool chunk per 8-column group, tiling the rows so each column's
/// packed weight decodes once per tile of rows instead of once per row. `dst`
/// is f32 row-major `[m, n]`.
pub fn matmul_q6k_repacked_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_x8: &[repack_q6k::BlockQ6Kx8],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    let nb = k / QK_K;
    let groups = n / 8;
    debug_assert_eq!(rhs_x8.len(), groups * nb);
    let lhs_b = quantize_act_q8k(lhs, m, k);
    let lhs_b = lhs_b.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    kquant_prefill_pool(m, n, k).run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_x8[g * nb..(g + 1) * nb];
        let mut r = 0;
        while r < m {
            let mt = (m - r).min(8);
            let mut local = [0f32; 64];
            #[cfg(target_feature = "avx2")]
            {
                let mut act_refs: [&[BlockQ8K]; 8] = [&lhs_b[0..0]; 8];
                for (t, slot) in act_refs.iter_mut().enumerate().take(mt) {
                    *slot = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                }
                unsafe {
                    repack_q6k::gemm_group_avx2(bgrp, &act_refs[..mt], nb, &mut local);
                }
            }
            #[cfg(not(target_feature = "avx2"))]
            for t in 0..mt {
                let act = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                repack_q6k::gemv_group_scalar(bgrp, act, nb, &mut local[t * 8..t * 8 + 8]);
            }
            for t in 0..mt {
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.0.add((r + t) * n + g * 8), 8)
                };
                out.copy_from_slice(&local[t * 8..t * 8 + 8]);
            }
            r += mt;
        }
    });
    Ok(())
}

/// Prefill GEMM for Q5_K weights repacked into `[n/8][k/256]` `BlockQ5Kx8`.
/// Mirrors the other K-quant tiled drivers: quantize the activation to Q8_K
/// once, then one pool chunk per 8-column group, tiling the rows so each
/// column's weight decodes once per tile instead of once per row.
pub fn matmul_q5k_repacked_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_x8: &[repack_q5k::BlockQ5Kx8],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    let nb = k / QK_K;
    let groups = n / 8;
    debug_assert_eq!(rhs_x8.len(), groups * nb);
    let lhs_b = quantize_act_q8k(lhs, m, k);
    let lhs_b = lhs_b.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    kquant_prefill_pool(m, n, k).run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_x8[g * nb..(g + 1) * nb];
        let mut r = 0;
        while r < m {
            let mt = (m - r).min(8);
            let mut local = [0f32; 64];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            {
                let mut act_refs: [&[BlockQ8K]; 8] = [&lhs_b[0..0]; 8];
                for (t, slot) in act_refs.iter_mut().enumerate().take(mt) {
                    *slot = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                }
                unsafe {
                    repack_q5k::gemm_group_avx2(bgrp, &act_refs[..mt], nb, &mut local);
                }
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            for t in 0..mt {
                let act = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                repack_q5k::gemv_group_scalar(bgrp, act, nb, &mut local[t * 8..t * 8 + 8]);
            }
            for t in 0..mt {
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.0.add((r + t) * n + g * 8), 8)
                };
                out.copy_from_slice(&local[t * 8..t * 8 + 8]);
            }
            r += mt;
        }
    });
    Ok(())
}

/// Q4_0 decode GEMV over the `repack_q4_0` row-grouped layout: activation
/// rows -> Q8_0 once (decode m is 1-3, done serially), then one pool chunk per
/// 8-column group. Each group's blocks stream contiguously and the 8
/// independent accumulator chains amortise the activation load - the
/// per-column `vec_dot_q4_0_q8_0` re-streams the activation and serialises on
/// one fmadd chain, which is the measured mistral-nemo CPU decode gap (the
/// fleet's only dense Q4_0 model; Q4_K has `matmul_q4k_repacked_tiled`).
pub fn matmul_q4_0_repacked_gemv(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_g8: &[BlockQ4_0],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(k % QK4_0, 0);
    let nb = k / QK4_0;
    let groups = n / 8;
    debug_assert_eq!(rhs_g8.len(), groups * nb * 8);
    let pool = gemv_pool::pool();

    // Activation -> Q8_0 (m rows; decode m<=3 so serial quantization is cheap).
    // Per-thread scratch, NOT a fresh vec: this runs ~7xlayers times per token
    // and per-call allocation churn evicts the mmap'd weight pages (see
    // QSCRATCH's doc above).
    let mut lhs_vec = qscratch_take::<BlockQ8_0>(m * nb);
    for row_idx in 0..m {
        BlockQ8_0::quantize(
            &lhs[row_idx * k..(row_idx + 1) * k],
            &mut lhs_vec[row_idx * nb..(row_idx + 1) * nb],
        );
    }
    let lhs_b: &[BlockQ8_0] = &lhs_vec[..m * nb];

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_g8[g * nb * 8..(g + 1) * nb * 8];
        for r in 0..m {
            let act = &lhs_b[r * nb..(r + 1) * nb];
            let mut local = [0f32; 8];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            unsafe {
                repack_q4_0::gemv_group_avx2(bgrp, act, nb, &mut local);
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            repack_q4_0::gemv_group_scalar(bgrp, act, nb, &mut local);
            // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
            let out = unsafe { std::slice::from_raw_parts_mut(dst_ptr.0.add(r * n + g * 8), 8) };
            out.copy_from_slice(&local);
        }
    });
    qscratch_return(lhs_vec);
    Ok(())
}

/// Prefill GEMM for Q4_0 weights repacked into `[n/8][nb][8]` block-major
/// `BlockQ4_0` (the same layout the decode GEMV uses). Quantize the activation
/// to Q8_0 once, then one pool chunk per 8-column group, tiling the rows so each
/// column's weight block decodes once per tile instead of once per row. `dst`
/// is f32 row-major `[m, n]`.
pub fn matmul_q4_0_repacked_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_g8: &[BlockQ4_0],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(k % QK4_0, 0);
    let nb = k / QK4_0;
    let groups = n / 8;
    debug_assert_eq!(rhs_g8.len(), groups * nb * 8);
    let pool = gemv_pool::pool();

    let mut lhs_b = vec![BlockQ8_0::zeros(); m * nb];
    {
        let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
        pool.run(m, &|row_idx| {
            let lhs_b_ptr = &lhs_b_ptr;
            let lhs_b_mut =
                unsafe { std::slice::from_raw_parts_mut(lhs_b_ptr.0.add(row_idx * nb), nb) };
            BlockQ8_0::quantize(&lhs[row_idx * k..(row_idx + 1) * k], lhs_b_mut);
        });
    }
    let lhs_b = lhs_b.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_g8[g * nb * 8..(g + 1) * nb * 8];
        let mut r = 0;
        while r < m {
            let mt = (m - r).min(8);
            let mut local = [0f32; 64];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            {
                let mut act_refs: [&[BlockQ8_0]; 8] = [&lhs_b[0..0]; 8];
                for (t, slot) in act_refs.iter_mut().enumerate().take(mt) {
                    *slot = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                }
                unsafe {
                    repack_q4_0::gemm_group_avx2(bgrp, &act_refs[..mt], nb, &mut local);
                }
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            for t in 0..mt {
                let act = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                repack_q4_0::gemv_group_scalar(bgrp, act, nb, &mut local[t * 8..t * 8 + 8]);
            }
            for t in 0..mt {
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.0.add((r + t) * n + g * 8), 8)
                };
                out.copy_from_slice(&local[t * 8..t * 8 + 8]);
            }
            r += mt;
        }
    });
    Ok(())
}

/// Prefill GEMM for MXFP4 weights repacked into `[n/8][nb][8]` block-major
/// `BlockMxFp4`. Same driver as the Q4_0 tiled path: quantize the activation to
/// Q8_0 once, then one pool chunk per 8-column group, tiling the rows so each
/// column's weight block decodes once per tile instead of once per row. `dst`
/// is f32 row-major `[m, n]`.
pub fn matmul_mxfp4_repacked_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_g8: &[BlockMxFp4],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(k % QK_MXFP4, 0);
    let nb = k / QK_MXFP4;
    let groups = n / 8;
    debug_assert_eq!(rhs_g8.len(), groups * nb * 8);
    let pool = gemv_pool::pool();

    let mut lhs_b = vec![BlockQ8_0::zeros(); m * nb];
    {
        let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
        pool.run(m, &|row_idx| {
            let lhs_b_ptr = &lhs_b_ptr;
            let lhs_b_mut =
                unsafe { std::slice::from_raw_parts_mut(lhs_b_ptr.0.add(row_idx * nb), nb) };
            BlockQ8_0::quantize(&lhs[row_idx * k..(row_idx + 1) * k], lhs_b_mut);
        });
    }
    let lhs_b = lhs_b.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_g8[g * nb * 8..(g + 1) * nb * 8];
        let mut r = 0;
        while r < m {
            let mt = (m - r).min(8);
            let mut local = [0f32; 64];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            {
                let mut act_refs: [&[BlockQ8_0]; 8] = [&lhs_b[0..0]; 8];
                for (t, slot) in act_refs.iter_mut().enumerate().take(mt) {
                    *slot = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                }
                unsafe {
                    repack_mxfp4::gemm_group_avx2(bgrp, &act_refs[..mt], nb, &mut local);
                }
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            for t in 0..mt {
                let act = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                repack_mxfp4::gemv_group_scalar(bgrp, act, nb, &mut local[t * 8..t * 8 + 8]);
            }
            for t in 0..mt {
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.0.add((r + t) * n + g * 8), 8)
                };
                out.copy_from_slice(&local[t * 8..t * 8 + 8]);
            }
            r += mt;
        }
    });
    Ok(())
}

/// Column-interleaved prefill GEMM for MXFP4 (`block_mxfp4x8` weights, 8 output
/// columns held in the 8 SIMD lanes). Quantizes the activation to `block_q8_0x4`
/// (4 rows interleaved) once, then one pool chunk per 8-column group, each looping
/// the 4-row tiles. `rhs_x8` is the `[n/8][nb]` repacked weight; `dst` is f32
/// row-major `[m, n]`. Not bit-identical to `matmul_mxfp4_repacked_tiled` (a
/// different, reference-matching reduction order) but ~2x denser (no per-column
/// horizontal sum).
pub fn matmul_mxfp4_x8_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_x8: &[repack_mxfp4_x8::BlockMxFp4x8],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(k % QK_MXFP4, 0);
    let nb = k / QK_MXFP4;
    let groups = n / 8;
    let mgroups = m.div_ceil(4);
    debug_assert_eq!(rhs_x8.len(), groups * nb);
    let pool = gemv_pool::pool();

    let ax4 = repack_mxfp4_x8::quantize_act(lhs, m, k);
    let ax4 = ax4.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_x8[g * nb..(g + 1) * nb];
        let write_tile = |mg: usize, local: &[f32]| {
            for mm in 0..4 {
                let row = mg * 4 + mm;
                if row >= m {
                    continue;
                }
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out =
                    unsafe { std::slice::from_raw_parts_mut(dst_ptr.0.add(row * n + g * 8), 8) };
                out.copy_from_slice(&local[mm * 8..mm * 8 + 8]);
            }
        };
        let mut mg = 0;
        // 16-row chunks (4 activation groups) amortize the weight decode 4x.
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        while mg + 4 <= mgroups {
            let a = |i: usize| &ax4[(mg + i) * nb..(mg + i + 1) * nb];
            let mut local = [0f32; 128];
            unsafe {
                repack_mxfp4_x8::gemm_group16_avx2(a(0), a(1), a(2), a(3), bgrp, nb, &mut local);
            }
            for rp in 0..4 {
                write_tile(mg + rp, &local[rp * 32..rp * 32 + 32]);
            }
            mg += 4;
        }
        while mg < mgroups {
            let agrp = &ax4[mg * nb..(mg + 1) * nb];
            let mut local = [0f32; 32];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            unsafe {
                repack_mxfp4_x8::gemm_group_avx2(agrp, bgrp, nb, &mut local);
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            repack_mxfp4_x8::gemm_scalar(agrp, bgrp, nb, &mut local);
            write_tile(mg, &local);
            mg += 1;
        }
    });
    Ok(())
}

/// Column-interleaved prefill GEMM for Q8_0 (`block_q8_0x8` weights, 8 output
/// columns in the SIMD lanes). Same driver shape as `matmul_mxfp4_x8_tiled`;
/// `rhs_x8` is the `[n/8][nb]` repacked weight, `dst` is f32 row-major `[m, n]`.
pub fn matmul_q8_0_x8_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_x8: &[repack_q8_0_x8::BlockQ8_0x8],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(k % 32, 0);
    let nb = k / 32;
    let groups = n / 8;
    let mgroups = m.div_ceil(4);
    debug_assert_eq!(rhs_x8.len(), groups * nb);
    let pool = gemv_pool::pool();

    let ax4 = repack_q8_0_x8::quantize_act(lhs, m, k);
    let ax4 = ax4.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_x8[g * nb..(g + 1) * nb];
        let write_tile = |mg: usize, local: &[f32]| {
            for mm in 0..4 {
                let row = mg * 4 + mm;
                if row >= m {
                    continue;
                }
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out =
                    unsafe { std::slice::from_raw_parts_mut(dst_ptr.0.add(row * n + g * 8), 8) };
                out.copy_from_slice(&local[mm * 8..mm * 8 + 8]);
            }
        };
        let mut mg = 0;
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        while mg + 4 <= mgroups {
            let a = |i: usize| &ax4[(mg + i) * nb..(mg + i + 1) * nb];
            let mut local = [0f32; 128];
            unsafe {
                repack_q8_0_x8::gemm_group16_avx2(a(0), a(1), a(2), a(3), bgrp, nb, &mut local);
            }
            for rp in 0..4 {
                write_tile(mg + rp, &local[rp * 32..rp * 32 + 32]);
            }
            mg += 4;
        }
        while mg < mgroups {
            let agrp = &ax4[mg * nb..(mg + 1) * nb];
            let mut local = [0f32; 32];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            unsafe {
                repack_q8_0_x8::gemm_group_avx2(agrp, bgrp, nb, &mut local);
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            repack_q8_0_x8::gemm_scalar(agrp, bgrp, nb, &mut local);
            write_tile(mg, &local);
            mg += 1;
        }
    });
    Ok(())
}

/// Prefill GEMM for Q5_0 weights repacked into `[n/8][nb][8]` block-major
/// `BlockQ5_0`. Mirrors the K-quant tiled drivers: quantize the activation to
/// Q8_0 once, then one pool chunk per 8-column group, tiling the rows so each
/// column's weight block decodes once per tile instead of once per row. `dst`
/// is f32 row-major `[m, n]`.
pub fn matmul_q5_0_repacked_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_g8: &[BlockQ5_0],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(k % QK5_0, 0);
    let nb = k / QK5_0;
    let groups = n / 8;
    debug_assert_eq!(rhs_g8.len(), groups * nb * 8);
    let pool = gemv_pool::pool();

    let mut lhs_b = vec![BlockQ8_0::zeros(); m * nb];
    {
        let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
        pool.run(m, &|row_idx| {
            let lhs_b_ptr = &lhs_b_ptr;
            let lhs_b_mut =
                unsafe { std::slice::from_raw_parts_mut(lhs_b_ptr.0.add(row_idx * nb), nb) };
            BlockQ8_0::quantize(&lhs[row_idx * k..(row_idx + 1) * k], lhs_b_mut);
        });
    }
    let lhs_b = lhs_b.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_g8[g * nb * 8..(g + 1) * nb * 8];
        let mut r = 0;
        while r < m {
            let mt = (m - r).min(8);
            let mut local = [0f32; 64];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            {
                let mut act_refs: [&[BlockQ8_0]; 8] = [&lhs_b[0..0]; 8];
                for (t, slot) in act_refs.iter_mut().enumerate().take(mt) {
                    *slot = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                }
                unsafe {
                    repack_q5_0::gemm_group_avx2(bgrp, &act_refs[..mt], nb, &mut local);
                }
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            for t in 0..mt {
                let act = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                repack_q5_0::gemv_group_scalar(bgrp, act, nb, &mut local[t * 8..t * 8 + 8]);
            }
            for t in 0..mt {
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.0.add((r + t) * n + g * 8), 8)
                };
                out.copy_from_slice(&local[t * 8..t * 8 + 8]);
            }
            r += mt;
        }
    });
    Ok(())
}

pub fn matmul_q8_0_repacked_tiled(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_g8: &[BlockQ8_0],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(k % QK8_0, 0);
    let nb = k / QK8_0;
    let groups = n / 8;
    debug_assert_eq!(rhs_g8.len(), groups * nb * 8);
    let pool = gemv_pool::pool();

    let mut lhs_b = vec![BlockQ8_0::zeros(); m * nb];
    {
        let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
        pool.run(m, &|row_idx| {
            let lhs_b_ptr = &lhs_b_ptr;
            let lhs_b_mut =
                unsafe { std::slice::from_raw_parts_mut(lhs_b_ptr.0.add(row_idx * nb), nb) };
            BlockQ8_0::quantize(&lhs[row_idx * k..(row_idx + 1) * k], lhs_b_mut);
        });
    }
    let lhs_b = lhs_b.as_slice();

    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(groups, &|g| {
        let dst_ptr = &dst_ptr;
        let bgrp = &rhs_g8[g * nb * 8..(g + 1) * nb * 8];
        let mut r = 0;
        while r < m {
            let mt = (m - r).min(8);
            let mut local = [0f32; 64];
            #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
            {
                let mut act_refs: [&[BlockQ8_0]; 8] = [&lhs_b[0..0]; 8];
                for (t, slot) in act_refs.iter_mut().enumerate().take(mt) {
                    *slot = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                }
                unsafe {
                    repack_q8_0::gemm_group_avx2(bgrp, &act_refs[..mt], nb, &mut local);
                }
            }
            #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
            for t in 0..mt {
                let act = &lhs_b[(r + t) * nb..(r + t + 1) * nb];
                repack_q8_0::gemv_group_scalar(bgrp, act, nb, &mut local[t * 8..t * 8 + 8]);
            }
            for t in 0..mt {
                // SAFETY: disjoint [row, g*8..g*8+8] slices of dst.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.0.add((r + t) * n + g * 8), 8)
                };
                out.copy_from_slice(&local[t * 8..t * 8 + 8]);
            }
            r += mt;
        }
    });
    Ok(())
}

/// Send+Sync raw mut pointer for the disjoint-shard writes above.
pub(super) struct SendMutPtr<T>(pub(super) *mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

thread_local! {
    /// Per-thread activation-quantization scratch, keyed by quant type. Matmul
    /// runs hundreds of times per decode token; allocating a fresh
    /// `vec![ActivationBlock::zeros(); ...]` each time churns the allocator, and that
    /// churn pressures the page cache enough to evict the mmap'd GGUF weight
    /// pages - which the worker threads then re-fault every token (profiled:
    /// kernel `queued_spin_lock`/`filemap_add_folio`). Reusing one growable
    /// buffer per thread keeps RSS flat so weights stay resident - the
    /// allocation discipline ollama gets from its persistent graph work buffer,
    /// with no privileged `mlock`.
    static QSCRATCH: std::cell::RefCell<
        std::collections::HashMap<std::any::TypeId, Box<dyn std::any::Any>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Take the per-thread reusable buffer for `T` (resized to >= `len`, zero-filled
/// on growth), leaving an empty placeholder. Pair with `qscratch_return` to put
/// it back so its allocation is reused next call. While taken, a re-entrant
/// `qscratch_take::<T>` on the same thread just allocates a fresh Vec (correct,
/// only a missed reuse) - matmul is not re-entrant so the hot path always reuses.
pub(super) fn qscratch_take<T: BlockFormat + 'static>(len: usize) -> Vec<T> {
    QSCRATCH.with(|m| {
        let mut m = m.borrow_mut();
        let b = m
            .entry(std::any::TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<T>::new()));
        let v = b.downcast_mut::<Vec<T>>().unwrap();
        let mut taken = std::mem::take(v);
        if taken.len() < len {
            taken.resize(len, T::zeros());
        }
        taken
    })
}

/// Return a buffer taken by `qscratch_take`, keeping whichever has the larger
/// capacity so the scratch grows monotonically and stays resident.
pub(super) fn qscratch_return<T: BlockFormat + 'static>(buf: Vec<T>) {
    QSCRATCH.with(|m| {
        let mut m = m.borrow_mut();
        let b = m
            .entry(std::any::TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<T>::new()));
        let v = b.downcast_mut::<Vec<T>>().unwrap();
        if buf.capacity() >= v.capacity() {
            *v = buf;
        }
    })
}

/// Parallel `for_each` over a mutable slice via the persistent spin-pool: each
/// worker grabs a contiguous index range with one atomic `fetch_add`. Use for
/// the many tiny per-token MoE reduce regions instead of `rayon par_iter_mut`,
/// whose recursive fork/join split dominates the microscopic per-region work
/// (profiled: hundreds of MoE/SSM parallel regions per token spent ~45% of CPU
/// decode in `bridge_producer_consumer`). `f(i, &mut out[i])`; the closure must
/// write only its own index (ranges are disjoint).
pub fn pool_for_each_mut<T: Send + Sync>(
    out: &mut [T],
    min_len: usize,
    f: &(dyn Fn(usize, &mut T) + Sync),
) {
    let n = out.len();
    if n == 0 {
        return;
    }
    let pool = gemv_pool::pool();
    let threads = pool.threads.max(1);
    let chunk = n.div_ceil(threads).max(min_len.max(1));
    let n_chunks = n.div_ceil(chunk);
    let ptr = SendMutPtr(out.as_mut_ptr());
    pool.run(n_chunks, &|c| {
        let ptr = &ptr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(n);
        for i in lo..hi {
            // SAFETY: chunk `c` owns the disjoint index range [lo, hi).
            let o = unsafe { &mut *ptr.0.add(i) };
            f(i, o);
        }
    });
}

/// Parallel `par_chunks_mut(chunk)` over a mutable slice via the persistent
/// spin-pool: worker `c` gets the disjoint slice `out[c*chunk .. (c+1)*chunk]`
/// (last chunk short) with one atomic `fetch_add`. Use for the per-token
/// attention head-parallelism (one chunk = one query head) instead of rayon's
/// `par_chunks_mut`, so decode attention shares the SAME already-spinning
/// workers as the FFN GEMVs - one spin-pool for the whole layer, no rayon pool
/// wake/steal per attention call and no core contention between two live pools
/// (ggml's single-thread-pool model). Bit-identical to the serial/rayon loop:
/// each chunk is an independent computation; only the chunk->worker assignment
/// changes.
pub fn pool_par_chunks_mut(out: &mut [f32], chunk: usize, f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
    let n = out.len();
    if n == 0 || chunk == 0 {
        return;
    }
    let n_chunks = n.div_ceil(chunk);
    let ptr = SendMutPtr(out.as_mut_ptr());
    gemv_pool::pool().run(n_chunks, &|c| {
        let ptr = &ptr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(n);
        // SAFETY: chunk `c` owns the disjoint index range [lo, hi); `run`
        // keeps this frame (and `f`) alive until every grabbed chunk finishes.
        let s = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(lo), hi - lo) };
        f(c, s);
    });
}

/// Like [`pool_par_chunks_mut`] but on the logical-core [`gemv_pool::prefill_pool`],
/// for the F16 attention GEMM at prefill - so it shares the ONE prefill pool with
/// the quantized prefill GEMM instead of running on rayon (whose idle workers
/// otherwise steal SMT cycles from the quantized GEMM regions).
pub fn pool_par_chunks_mut_prefill(
    out: &mut [f32],
    chunk: usize,
    f: &(dyn Fn(usize, &mut [f32]) + Sync),
) {
    let n = out.len();
    if n == 0 || chunk == 0 {
        return;
    }
    let n_chunks = n.div_ceil(chunk);
    let ptr = SendMutPtr(out.as_mut_ptr());
    gemv_pool::prefill_pool().run(n_chunks, &|c| {
        let ptr = &ptr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(n);
        // SAFETY: chunk `c` owns the disjoint index range [lo, hi).
        let s = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(lo), hi - lo) };
        f(c, s);
    });
}

/// Generic-element version of [`pool_par_chunks_mut`] - same spin-pool chunking
/// for any `Send + Sync` element type (e.g. f16/bf16 casts), so those stay off
/// rayon and on the one active pool. `f(chunk_index, chunk_slice)`.
pub fn pool_par_chunks_mut_t<T: Send + Sync>(
    out: &mut [T],
    chunk: usize,
    f: &(dyn Fn(usize, &mut [T]) + Sync),
) {
    let n = out.len();
    if n == 0 || chunk == 0 {
        return;
    }
    let n_chunks = n.div_ceil(chunk);
    let ptr = SendMutPtr(out.as_mut_ptr());
    gemv_pool::pool().run(n_chunks, &|c| {
        let ptr = &ptr;
        let lo = c * chunk;
        let hi = (lo + chunk).min(n);
        // SAFETY: chunk `c` owns the disjoint index range [lo, hi).
        let s = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(lo), hi - lo) };
        f(c, s);
    });
}

/// Quantize the F16 activation rows into `lhs_b` (`ActivationBlock` blocks), shared
/// by the f16 GEMV entries below.
///
/// Decode (m<4) stays SINGLE-THREADED on purpose. It is tempting to split the
/// row's K-blocks across the pool the way the f32 `matmul` does: the pass runs
/// on the publisher while every other worker sits in the spin loop, and Amdahl
/// multiplies that serial time by the worker count (`perf`: ~6% of all cycles
/// land in that spin, the same order as the whole remaining decode gap). But it
/// MEASURES NEUTRAL - cool-gated median-15: 25.17 split vs 25.21 serial, i.e.
/// -0.2%, inside noise. The split needs its own `pool.run`, and one extra
/// fork-join region costs about what the serial quantize saves. That is the
/// general shape of this pool: adding a region never pays for itself, so the
/// only way to reclaim the spin is to have FEWER regions per token, not more.
pub(super) fn quantize_lhs_f16<T: BlockFormat>(
    (m, k): (usize, usize),
    lhs: &[f16],
    lhs_b: &mut [T::ActivationBlock],
) {
    let k_in_lhs_blocks = k.div_ceil(T::BLOCK_LEN);
    let pool = gemv_pool::pool();
    let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
    let quant_row = |row_idx: usize| {
        let lhs_b_ptr = &lhs_b_ptr;
        // SAFETY: rows are disjoint `k_in_lhs_blocks` slices of `lhs_b`.
        let lhs_b = unsafe {
            std::slice::from_raw_parts_mut(
                lhs_b_ptr.0.add(row_idx * k_in_lhs_blocks),
                k_in_lhs_blocks,
            )
        };
        let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
        let lhs_f32: Vec<_> = lhs.iter().map(|&x| x.to_f32()).collect();
        T::ActivationBlock::quantize(&lhs_f32, lhs_b);
    };
    if m >= 4 {
        // Prefill: row-parallel (each row is a whole quantize).
        pool.run(m, &quant_row);
        return;
    }
    for row_idx in 0..m {
        quant_row(row_idx);
    }
}

pub fn matmul_f16<T: BlockFormat>(
    mkn: (usize, usize, usize),
    lhs: &[f16],
    rhs_t: &[T],
    dst: &mut [f16],
) -> Result<()> {
    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        return Err(Error(format!(
            "unexpected lhs length {} {mkn:?}",
            lhs.len()
        )));
    }

    let k_in_lhs_blocks = k.div_ceil(T::BLOCK_LEN);
    let k_in_rhs_blocks = k.div_ceil(T::ActivationBlock::BLOCK_LEN);
    let pool = gemv_pool::pool();
    let mut lhs_b = vec![T::ActivationBlock::zeros(); m * k_in_lhs_blocks];
    quantize_lhs_f16::<T>((m, k), lhs, &mut lhs_b);
    let lhs_b = lhs_b.as_slice();

    // Column-parallel like the f32 `matmul`. Without this the f16 path ran
    // single-threaded - f16-activation models decoded their attention GEMVs
    // on one core.
    let (chunk_cols, n_chunks) = gemv_grid(m, n, pool.threads);
    let chunks_per_row = n.div_ceil(chunk_cols);
    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(n_chunks, &|chunk| {
        let dst_ptr = &dst_ptr;
        let row_idx = chunk / chunks_per_row;
        let col0 = (chunk % chunks_per_row) * chunk_cols;
        let col1 = (col0 + chunk_cols).min(n);
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        // SAFETY: chunks address disjoint [row, col0..col1] ranges of dst.
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(dst_ptr.0.add(row_idx * n + col0), col1 - col0)
        };
        for (i, d) in dst_row.iter_mut().enumerate() {
            let col_idx = col0 + i;
            let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
            let value = T::dot(rhs_col, lhs_row);
            *d = f16::from_f32(value);
        }
    });
    Ok(())
}

/// Fused dense SwiGLU gate+up projection for F16-activation decode:
/// `dst[j] = silu(gate[j].x) * (up[j].x)` in one pass. Mirrors `matmul_f16`
/// (same shared single activation-quantize, same dynamic `gemv_grid` column
/// partition) but reads BOTH weight matrices per column and folds silu.mul
/// inline - so the SwiGLU that was `gate.forward -> f32 cast -> silu`,
/// `up.forward -> f32 cast`, `mul -> f16 cast` (two separate matmul_f16 calls,
/// each re-quantizing the SAME activation into a fresh buffer, plus 4 whole-row
/// intermediate passes) collapses to: quantize x ONCE, one dst buffer, no
/// intermediate tensors. `gate_t`/`up_t` share the `[n, k]` block layout.
pub fn matmul_f16_gate_up_silu<T: BlockFormat>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f16],
    gate_t: &[T],
    up_t: &[T],
    dst: &mut [f16],
) -> Result<()> {
    if m * k != lhs.len() {
        return Err(Error(format!(
            "gate_up_silu: lhs len {} != {m}x{k}",
            lhs.len()
        )));
    }
    let k_in_lhs_blocks = k.div_ceil(T::BLOCK_LEN);
    let k_in_rhs_blocks = k.div_ceil(T::ActivationBlock::BLOCK_LEN);
    let pool = gemv_pool::pool();
    // Quantize the activation ONCE (shared by gate and up), exactly as matmul_f16.
    let mut lhs_b = vec![T::ActivationBlock::zeros(); m * k_in_lhs_blocks];
    quantize_lhs_f16::<T>((m, k), lhs, &mut lhs_b);
    let lhs_b = lhs_b.as_slice();

    let (chunk_cols, n_chunks) = gemv_grid(m, n, pool.threads);
    let chunks_per_row = n.div_ceil(chunk_cols);
    let dst_ptr = SendMutPtr(dst.as_mut_ptr());
    pool.run(n_chunks, &|chunk| {
        let dst_ptr = &dst_ptr;
        let row_idx = chunk / chunks_per_row;
        let col0 = (chunk % chunks_per_row) * chunk_cols;
        let col1 = (col0 + chunk_cols).min(n);
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        // SAFETY: chunks address disjoint [row, col0..col1] ranges of dst.
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(dst_ptr.0.add(row_idx * n + col0), col1 - col0)
        };
        for (i, d) in dst_row.iter_mut().enumerate() {
            let col_idx = col0 + i;
            let gcol = &gate_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
            let ucol = &up_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
            let g = T::dot(gcol, lhs_row);
            let u = T::dot(ucol, lhs_row);
            let sg = g / (1.0 + (-g).exp()); // silu(g)
            *d = f16::from_f32(sg * u);
        }
    });
    Ok(())
}

macro_rules! verify_block_size {
    ( $block_type:ident ) => {
        const _: () = assert!(
            $block_type::BLOCK_LEN == <$block_type as BlockFormat>::ActivationBlock::BLOCK_LEN
        );
    };
}

macro_rules! verify_block_sizes {
    ( $( $block_type:ident ),* ) => {
        $(
            verify_block_size!($block_type);
        )*
    };
}

verify_block_sizes!(
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, BlockQ2K, BlockQ3K, BlockQ4K,
    BlockQ5K, BlockQ6K, BlockQ8K, BlockMxFp4, f32, f16, bf16
);

#[cfg(test)]
mod mxfp4_avx_test {
    use super::*;

    // The native AVX2 MXFP4xQ8_0 dot must match the scalar dequant-then-dot.
    #[cfg(target_feature = "avx2")]
    #[test]
    pub(super) fn vec_dot_mxfp4_q8_0_avx_matches_scalar() {
        let nb = 16usize;
        let n = nb * QK_MXFP4;
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let wf: Vec<f32> = (0..n).map(|_| next() * 5.0).collect();
        let xf: Vec<f32> = (0..n).map(|_| next() * 3.0).collect();
        let mut wq = vec![BlockMxFp4::zeros(); nb];
        BlockMxFp4::quantize(&wf, &mut wq);
        let mut xq = vec![BlockQ8_0::zeros(); nb];
        BlockQ8_0::quantize(&xf, &mut xq);
        let scalar = BlockMxFp4::dot_scalar(&wq, &xq);
        let avx = super::avx::vec_dot_mxfp4_q8_0(&wq, &xq);
        let rel = (scalar - avx).abs() / scalar.abs().max(1e-6);
        assert!(rel < 1e-4, "avx {avx} scalar {scalar} rel {rel}");
    }

    // Q6_K AVX2 dot (hoisted-bsums -32 offset, ggml-style) must match the
    // scalar reference bit-for-bit-close - guards the offset-fold port.
    #[cfg(target_feature = "avx2")]
    #[test]
    pub(super) fn vec_dot_q6k_q8k_avx_matches_scalar() {
        let nb = 12usize;
        let n = nb * QK_K;
        let mut s = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let wf: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();
        let xf: Vec<f32> = (0..n).map(|_| next() * 3.0).collect();
        let mut wq = vec![BlockQ6K::zeros(); nb];
        BlockQ6K::quantize(&wf, &mut wq);
        let mut xq = vec![BlockQ8K::zeros(); nb];
        BlockQ8K::quantize(&xf, &mut xq);
        let scalar = BlockQ6K::dot_scalar(&wq, &xq);
        let avx = super::avx::vec_dot_q6k_q8k(&wq, &xq);
        let rel = (scalar - avx).abs() / scalar.abs().max(1e-6);
        assert!(rel < 1e-4, "q6k avx {avx} scalar {scalar} rel {rel}");
    }
}

#[cfg(test)]
mod repack_q4k_plain_bench {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// Single-thread cost of the plain-scales GEMV vs the packed v2 (run with
    /// --release --include-ignored): quantifies the scalar scale-extraction
    /// overhead that the unified in-place layout would pay at M=1.
    #[test]
    #[ignore]
    pub(super) fn bench_plain_vs_v2() {
        let (k, n) = (1024usize, 1024usize);
        let nb = k / QK_K;
        let mut sd = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            sd ^= sd << 13;
            sd ^= sd >> 7;
            sd ^= sd << 17;
            ((sd >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let wf: Vec<f32> = (0..n * k).map(|_| next() * 0.2).collect();
        let mut wq = vec![BlockQ4K::zeros(); n * nb];
        for row in 0..n {
            BlockQ4K::quantize(
                &wf[row * k..(row + 1) * k],
                &mut wq[row * nb..(row + 1) * nb],
            );
        }
        let af: Vec<f32> = (0..k).map(|_| next()).collect();
        let mut aq = vec![BlockQ8K::zeros(); nb];
        BlockQ8K::quantize(&af, &mut aq);
        let x8 = repack_q4k::repack(&wq, n, nb);
        let x8v2 = repack_q4k::repack_v2(&wq, n, nb);
        let groups = n / 8;
        let iters = 2000;
        let mut out = [0f32; 8];
        let mut sink = 0f32;
        #[cfg(target_feature = "avx2")]
        {
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                for g in 0..groups {
                    unsafe {
                        repack_q4k::gemv_group_avx2_v2(
                            &x8v2[g * nb..(g + 1) * nb],
                            &aq,
                            nb,
                            &mut out,
                        );
                    }
                    sink += out[0];
                }
            }
            let v2 = t0.elapsed().as_secs_f64() / iters as f64;
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                for g in 0..groups {
                    unsafe {
                        repack_q4k::gemv_group_avx2_plain(
                            &x8[g * nb..(g + 1) * nb],
                            &aq,
                            nb,
                            &mut out,
                        );
                    }
                    sink += out[0];
                }
            }
            let plain = t1.elapsed().as_secs_f64() / iters as f64;
            eprintln!(
                "gemv k={k} n={n}: v2 {:.1}us | plain {:.1}us | plain/v2 {:.3}x (sink {sink})",
                v2 * 1e6,
                plain * 1e6,
                plain / v2
            );
        }
    }
}

// ===========================================================================
// Float dot kernels (upstream src/cpu/{mod.rs,avx.rs} with the
// AVX2 `CurrentCpu*` specialization inlined: STEP=32, EPR=8, 4-register FMA
// accumulate + the same tree reduce - bit-exact with the fork)
// ===========================================================================

// ===========================================================================
// AVX2 dot kernels (upstream src/quantized/avx.rs) - verbatim
// ===========================================================================

// ===========================================================================
// Byte-slice dispatch layer (loken-new): `quant.rs` stores raw GGML block
// bytes (owned aligned buffers or zero-copy views); these helpers cast them
// to typed block slices and route to the engine above per dtype.
// ===========================================================================

/// Cast raw block bytes to a typed block slice. Errors (rather than UB) on
/// length or alignment mismatch - owned `QTensor` storage is 8-byte aligned
/// and GGUF mmap offsets are 32-aligned, so this only trips on foreign
/// pointers.
pub(crate) fn cast_blocks<T>(bytes: &[u8]) -> Result<&[T]> {
    let ts = std::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(ts) {
        return Err(Error(format!(
            "quant_cpu: byte length {} is not a multiple of the block size {ts}",
            bytes.len()
        )));
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(Error(format!(
            "quant_cpu: block pointer misaligned (need {})",
            std::mem::align_of::<T>()
        )));
    }
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / ts) })
}

macro_rules! with_blocks {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            GgmlDType::F32 => {
                type $T = f32;
                $body
            }
            GgmlDType::F16 => {
                type $T = f16;
                $body
            }
            GgmlDType::BF16 => {
                type $T = bf16;
                $body
            }
            GgmlDType::Q4_0 => {
                type $T = BlockQ4_0;
                $body
            }
            GgmlDType::Q4_1 => {
                type $T = BlockQ4_1;
                $body
            }
            GgmlDType::Q5_0 => {
                type $T = BlockQ5_0;
                $body
            }
            GgmlDType::Q5_1 => {
                type $T = BlockQ5_1;
                $body
            }
            GgmlDType::Q8_0 => {
                type $T = BlockQ8_0;
                $body
            }
            GgmlDType::Q2K => {
                type $T = BlockQ2K;
                $body
            }
            GgmlDType::Q3K => {
                type $T = BlockQ3K;
                $body
            }
            GgmlDType::Q4K => {
                type $T = BlockQ4K;
                $body
            }
            GgmlDType::Q5K => {
                type $T = BlockQ5K;
                $body
            }
            GgmlDType::Q6K => {
                type $T = BlockQ6K;
                $body
            }
            GgmlDType::Q8K => {
                type $T = BlockQ8K;
                $body
            }
            GgmlDType::MxFp4 => {
                type $T = BlockMxFp4;
                $body
            }
            // Q8_1 is an activation-side format (no `dequantize`, never a GGUF
            // weight dtype) - not routable as a weight.
            GgmlDType::Q8_1 => Err(Error("quant_cpu: Q8_1 has no weight-side path".into())),
        }
    };
}

/// Dtypes the engine can serve as a WEIGHT (dot matmul + dequantize).
pub fn supports(dtype: GgmlDType) -> bool {
    !matches!(dtype, GgmlDType::Q8_1)
}

/// Dequantize raw block bytes into `ys` (must be exactly elem_count long).
/// Large tensors shard across rayon (mirrors the fork's `QuantizedType::
/// dequantize` parallel path - per-block independent, so still bit-exact).
pub fn to_float_bytes(dtype: GgmlDType, bytes: &[u8], ys: &mut [f32]) -> Result<()> {
    with_blocks!(dtype, T => {
        let blocks: &[T] = cast_blocks(bytes)?;
        if ys.len() != blocks.len() * T::BLOCK_LEN {
            return Err(Error(format!(
                "quant_cpu dequantize: {} blocks of {:?} != {} output elems",
                blocks.len(), dtype, ys.len()
            )));
        }
        const PAR_ELEM_THRESHOLD: usize = 1 << 20;
        if ys.len() >= PAR_ELEM_THRESHOLD && T::BLOCK_LEN > 0 {
            let blck = T::BLOCK_LEN;
            let blocks_per_chunk = 4096usize;
            ys.par_chunks_mut(blocks_per_chunk * blck)
                .zip(blocks.par_chunks(blocks_per_chunk))
                .for_each(|(y_chunk, b_chunk)| {
                    T::dequantize(b_chunk, y_chunk);
                });
        } else {
            T::dequantize(blocks, ys);
        }
        Ok(())
    })
}

/// Cooperative dense-FFN DECODE region (M=1): rms-norm, gate/up GEMV, activation-mul,
/// down GEMV and the residual add all execute inside ONE pool region as work-stealing
/// phases (claim + done counters, blocked claims). The sequential path pays SIX pool
/// launches for the same work (a quantize + a GEMV region per projection) with serial
/// publisher glue between them; here the workers stay resident across the whole block.
///
/// Bit-identical to the sequential path: same `quantize` per activation block (block-
/// aligned chunking), same per-column `dot`, the caller's own norm / activation-mul
/// closures on identical inputs, and the same multiply-then-add residual sequence.
/// Returns Ok(false) when shapes/dtypes fall outside the supported fast path.
#[allow(clippy::too_many_arguments)]
pub fn ffn_swiglu_coop(
    x: &mut [f32],
    gate: Option<(GgmlDType, usize, usize, &[u8])>,
    up: (GgmlDType, usize, usize, &[u8]),
    down: (GgmlDType, usize, usize, &[u8]),
    residual_scale: Option<f32>,
    norm_fn: &(dyn Fn(&[f32], &mut [f32]) + Sync),
    act_mul_fn: &(dyn Fn(&[f32], &[f32], &mut [f32]) + Sync),
    norm_buf: &mut [f32],
    gu_buf: &mut [f32],
    act_buf: &mut [f32],
) -> Result<bool> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // Separate gate/up ([N,K] each) or a FUSED [2N,K] stack in `up` (rows 0..N =
    // gate, N..2N = up - the layout `forward_slice_cpu(&norm, &mut gateup)` fills).
    let (udt, uk, un, ubytes) = up;
    let (gdt, gk, gn, gbytes) = gate.unwrap_or(up);
    let (ddt, dk, dn, dbytes) = down;
    let hidden = x.len();
    let inter = if gate.is_some() { gn } else { un / 2 };
    if gdt != udt
        || gk != hidden
        || uk != hidden
        || (gate.is_some() && (gn != inter || un != inter))
        || (gate.is_none() && un != 2 * inter)
        || inter == 0
        || dk != inter
        || dn != hidden
        || norm_buf.len() != hidden
        || gu_buf.len() != 2 * inter
        || act_buf.len() != inter
        || !supports(gdt)
        || !supports(ddt)
    {
        return Ok(false);
    }
    with_blocks!(gdt, TG => {
        with_blocks!(ddt, TD => {
            if gk % TG::BLOCK_LEN != 0
                || TG::BLOCK_LEN != <<TG as BlockFormat>::ActivationBlock as BlockFormat>::BLOCK_LEN
                || dk % TD::BLOCK_LEN != 0
                || TD::BLOCK_LEN != <<TD as BlockFormat>::ActivationBlock as BlockFormat>::BLOCK_LEN
                || TG::COPIES_VERBATIM
                || TD::COPIES_VERBATIM
            {
                return Ok(false);
            }
            let grhs: &[TG] = cast_blocks(gbytes)?;
            let urhs: &[TG] = cast_blocks(ubytes)?;
            let drhs: &[TD] = cast_blocks(dbytes)?;
            let kb_g = hidden / TG::BLOCK_LEN;
            let kb_d = inter / TD::BLOCK_LEN;
            let fused = gate.is_none();
            let gu_rows_ok = if fused {
                urhs.len() == 2 * inter * kb_g
            } else {
                grhs.len() == inter * kb_g && urhs.len() == inter * kb_g
            };
            if !gu_rows_ok || drhs.len() != hidden * kb_d {
                return Ok(None).map(|_: Option<()>| false);
            }
            let mut aq_g = qscratch_take::<<TG as BlockFormat>::ActivationBlock>(kb_g);
            let mut aq_d = vec![<<TD as BlockFormat>::ActivationBlock as BlockFormat>::zeros(); kb_d];

            let pool = gemv_pool::pool();
            let nth = pool.threads.max(1);
            // Phase item counts. Quantize chunks mirror the sequential path's even
            // k-block split; GEMV column chunks mirror its chunk_cols formula.
            let qg_items = nth.min(kb_g);
            let qd_items = nth.min(kb_d);
            let col_chunk_gu = ((2 * inter) / (nth * 8)).clamp(8, 256).min((2 * inter).max(1));
            let gu_items = (2 * inter).div_ceil(col_chunk_gu);
            let act_chunk = inter.div_ceil(nth).max(1);
            let act_items = inter.div_ceil(act_chunk);
            let col_chunk_d = (hidden / (nth * 8)).clamp(8, 256).min(hidden.max(1));
            let d_items = hidden.div_ceil(col_chunk_d);

            let (c_norm, d_norm) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let (c_qg, d_qg) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let (c_gu, d_gu) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let (c_act, d_act) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let (c_qd, d_qd) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let c_down = AtomicUsize::new(0);

            let xptr = SendMutPtr(x.as_mut_ptr());
            let nptr = SendMutPtr(norm_buf.as_mut_ptr());
            let gptr = SendMutPtr(gu_buf.as_mut_ptr());
            let aptr = SendMutPtr(act_buf.as_mut_ptr());
            let aqg_ptr = SendMutPtr(aq_g.as_mut_ptr());
            let aqd_ptr = SendMutPtr(aq_d.as_mut_ptr());
            let xs_ref: &[f32] = unsafe { std::slice::from_raw_parts(xptr.0, hidden) };

            pool.run(nth, &|_j| {
                // Capture the SendMutPtr wrappers whole (2021 disjoint-field capture
                // would otherwise capture the raw `.0` pointers, which are not Sync).
                let (xptr, nptr, gptr, aptr, aqg_ptr, aqd_ptr) =
                    (&xptr, &nptr, &gptr, &aptr, &aqg_ptr, &aqd_ptr);
                // P0: rms norm (single item; ~2us - others arrive at the wait).
                if c_norm.fetch_add(1, Ordering::Relaxed) == 0 {
                    let nb = unsafe { std::slice::from_raw_parts_mut(nptr.0, hidden) };
                    norm_fn(xs_ref, nb);
                    d_norm.store(1, Ordering::Release);
                }
                while d_norm.load(Ordering::Acquire) == 0 { std::hint::spin_loop(); }
                let normed: &[f32] = unsafe { std::slice::from_raw_parts(nptr.0, hidden) };
                // P1: quantize normed -> aq_g (block-aligned even chunks).
                loop {
                    let t = c_qg.fetch_add(1, Ordering::Relaxed);
                    if t >= qg_items { break; }
                    let b0 = t * kb_g / qg_items;
                    let b1 = (t + 1) * kb_g / qg_items;
                    if b1 > b0 {
                        let xs = &normed[b0 * TG::BLOCK_LEN..b1 * TG::BLOCK_LEN];
                        let ys = unsafe {
                            std::slice::from_raw_parts_mut(aqg_ptr.0.add(b0), b1 - b0)
                        };
                        <<TG as BlockFormat>::ActivationBlock as BlockFormat>::quantize(xs, ys);
                    }
                    d_qg.fetch_add(1, Ordering::Release);
                }
                while d_qg.load(Ordering::Acquire) < qg_items { std::hint::spin_loop(); }
                let aqg: &[<TG as BlockFormat>::ActivationBlock] =
                    unsafe { std::slice::from_raw_parts(aqg_ptr.0, kb_g) };
                // P2: gate/up GEMV over 2*inter columns (cols < inter -> gate).
                loop {
                    let cchunk = c_gu.fetch_add(1, Ordering::Relaxed);
                    if cchunk >= gu_items { break; }
                    let c0 = cchunk * col_chunk_gu;
                    let c1 = (c0 + col_chunk_gu).min(2 * inter);
                    for c in c0..c1 {
                        let (w, col) = if fused {
                            (urhs, c)
                        } else if c < inter {
                            (grhs, c)
                        } else {
                            (urhs, c - inter)
                        };
                        let rhs_col = &w[col * kb_g..(col + 1) * kb_g];
                        let v = TG::dot(rhs_col, aqg);
                        unsafe { *gptr.0.add(c) = v; }
                    }
                    d_gu.fetch_add(1, Ordering::Release);
                }
                while d_gu.load(Ordering::Acquire) < gu_items { std::hint::spin_loop(); }
                // P3: activation-mul in chunks (elementwise: chunking is exact).
                loop {
                    let t = c_act.fetch_add(1, Ordering::Relaxed);
                    if t >= act_items { break; }
                    let a0 = t * act_chunk;
                    let a1 = (a0 + act_chunk).min(inter);
                    let g = unsafe { std::slice::from_raw_parts(gptr.0.add(a0), a1 - a0) };
                    let u = unsafe { std::slice::from_raw_parts(gptr.0.add(inter + a0), a1 - a0) };
                    let o = unsafe { std::slice::from_raw_parts_mut(aptr.0.add(a0), a1 - a0) };
                    act_mul_fn(g, u, o);
                    d_act.fetch_add(1, Ordering::Release);
                }
                while d_act.load(Ordering::Acquire) < act_items { std::hint::spin_loop(); }
                let act: &[f32] = unsafe { std::slice::from_raw_parts(aptr.0, inter) };
                // P4: quantize act -> aq_d.
                loop {
                    let t = c_qd.fetch_add(1, Ordering::Relaxed);
                    if t >= qd_items { break; }
                    let b0 = t * kb_d / qd_items;
                    let b1 = (t + 1) * kb_d / qd_items;
                    if b1 > b0 {
                        let xs = &act[b0 * TD::BLOCK_LEN..b1 * TD::BLOCK_LEN];
                        let ys = unsafe {
                            std::slice::from_raw_parts_mut(aqd_ptr.0.add(b0), b1 - b0)
                        };
                        <<TD as BlockFormat>::ActivationBlock as BlockFormat>::quantize(xs, ys);
                    }
                    d_qd.fetch_add(1, Ordering::Release);
                }
                while d_qd.load(Ordering::Acquire) < qd_items { std::hint::spin_loop(); }
                let aqd: &[<TD as BlockFormat>::ActivationBlock] =
                    unsafe { std::slice::from_raw_parts(aqd_ptr.0, kb_d) };
                // P5: down GEMV + fused residual (multiply-then-add, same order as
                // the sequential proj-scale-add sequence).
                loop {
                    let cchunk = c_down.fetch_add(1, Ordering::Relaxed);
                    if cchunk >= d_items { break; }
                    let c0 = cchunk * col_chunk_d;
                    let c1 = (c0 + col_chunk_d).min(hidden);
                    for c in c0..c1 {
                        let rhs_col = &drhs[c * kb_d..(c + 1) * kb_d];
                        let mut v = TD::dot(rhs_col, aqd);
                        if let Some(s) = residual_scale { v *= s; }
                        unsafe { *xptr.0.add(c) += v; }
                    }
                }
                // pool.run completion covers down stragglers.
            });
            qscratch_return(aq_g);
            let _ = aq_d;
            Ok(true)
        })
    })
}

/// `lhs [m,k] f32 x rhs_t [n,k] (raw blocks of `dtype`) -> dst [m,n]` through
/// the dot engine (activation quantized to the dtype's ActivationBlock once per
/// row, output columns rayon-parallel) - the production CPU decode path.
/// Multiply-accumulate submitted to the CPU matmuls, and the number of calls.
/// A kernel measured competitive against the reference while the engine spends
/// three times its CPU seconds means the excess is WORK, not efficiency - and
/// nothing counted the work. Read and reset per generation by the engine.
pub static CPU_MACS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static CPU_GEMV_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Quantized weight bytes streamed into those matmuls. Decode reads every
/// active weight once per token, so if the cost is traffic rather than
/// arithmetic this is the number that tracks the wall clock - and the MAC
/// count will not.
pub static CPU_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn take_cpu_mac_counters() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CPU_MACS.swap(0, Relaxed), CPU_GEMV_CALLS.swap(0, Relaxed))
}

pub fn take_cpu_bytes() -> u64 {
    CPU_BYTES.swap(0, std::sync::atomic::Ordering::Relaxed)
}

pub fn matmul_bytes(
    dtype: GgmlDType,
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_bytes: &[u8],
    dst: &mut [f32],
) -> Result<()> {
    {
        use std::sync::atomic::Ordering::Relaxed;
        CPU_MACS.fetch_add((m as u64) * (k as u64) * (n as u64), Relaxed);
        CPU_GEMV_CALLS.fetch_add(1, Relaxed);
        CPU_BYTES.fetch_add(rhs_bytes.len() as u64, Relaxed);
    }
    with_blocks!(dtype, T => {
        if !k.is_multiple_of(T::BLOCK_LEN) {
            return Err(Error(format!(
                "quant_cpu matmul: k {k} not a multiple of {:?} block size {}",
                dtype, T::BLOCK_LEN
            )));
        }
        let rhs: &[T] = cast_blocks(rhs_bytes)?;
        if rhs.len() != n * (k / T::BLOCK_LEN) {
            return Err(Error(format!(
                "quant_cpu matmul: rhs has {} blocks, want {} ({n} rows x {} / row)",
                rhs.len(), n * (k / T::BLOCK_LEN), k / T::BLOCK_LEN
            )));
        }
        if dst.len() != m * n {
            return Err(Error(format!(
                "quant_cpu matmul: dst len {} != {m}x{n}", dst.len()
            )));
        }
        matmul((m, k, n), lhs, rhs, dst)
    })
}

/// Flat multi-expert decode GEMV (MoE `mul_mat_id`, ggml-style). All routed
/// experts share `dtype`, `k`, `n`. Each expert's f32 activation `lhs[e]` (`[k]`)
/// is quantized once, then a SINGLE thread-pool region distributes
/// (expert x output-column-chunk) work - one parallelism level. This replaces the
/// per-expert path's nested `rayon(experts) x gemv_pool(columns)` (thread
/// oversubscription) plus its per-expert Tensor allocation. `dot` is identical
/// to `matmul_bytes`, so each expert's output is bit-for-bit equal to a separate
/// `matmul_bytes((1,k,n), lhs[e], rhs_bytes[e], outs[e])` call. Mirrors ggml's
/// `ggml_compute_forward_mul_mat_id` single-region scheduling.
pub fn matmul_bytes_multi(
    dtype: GgmlDType,
    k: usize,
    n: usize,
    rhs_bytes: &[&[u8]],
    lhs: &[&[f32]],
    outs: &mut [&mut [f32]],
) -> Result<()> {
    if lhs.len() != rhs_bytes.len() || outs.len() != rhs_bytes.len() {
        return Err(Error(format!(
            "matmul_bytes_multi: ne mismatch rhs {} lhs {} outs {}",
            rhs_bytes.len(),
            lhs.len(),
            outs.len()
        )));
    }
    if rhs_bytes.is_empty() {
        return Ok(());
    }
    with_blocks!(dtype, T => {
        matmul_bytes_multi_impl::<T>(k, n, rhs_bytes, lhs, outs)
    })
}

pub(super) fn matmul_bytes_multi_impl<T: BlockFormat>(
    k: usize,
    n: usize,
    rhs_bytes: &[&[u8]],
    lhs: &[&[f32]],
    outs: &mut [&mut [f32]],
) -> Result<()>
where
    T::ActivationBlock: 'static,
{
    let ne = rhs_bytes.len();
    if !k.is_multiple_of(T::BLOCK_LEN) {
        return Err(Error(format!(
            "matmul_bytes_multi: k {k} not a multiple of block size {}",
            T::BLOCK_LEN
        )));
    }
    let k_in_blocks = k / T::BLOCK_LEN;
    let mut rhs: Vec<&[T]> = Vec::with_capacity(ne);
    for e in 0..ne {
        if lhs[e].len() != k {
            return Err(Error(format!(
                "matmul_bytes_multi: lhs[{e}] len {} != k {k}",
                lhs[e].len()
            )));
        }
        if outs[e].len() != n {
            return Err(Error(format!(
                "matmul_bytes_multi: outs[{e}] len {} != n {n}",
                outs[e].len()
            )));
        }
        let blocks: &[T] = cast_blocks(rhs_bytes[e])?;
        if blocks.len() != n * k_in_blocks {
            return Err(Error(format!(
                "matmul_bytes_multi: rhs[{e}] has {} blocks, want {}",
                blocks.len(),
                n * k_in_blocks
            )));
        }
        rhs.push(blocks);
    }
    // Quantize each activation once (serial; at decode te=1 -> a single [k] row).
    let mut lhsq: Vec<Vec<T::ActivationBlock>> = Vec::with_capacity(ne);
    for e in 0..ne {
        let mut b = vec![T::ActivationBlock::zeros(); k_in_blocks];
        T::ActivationBlock::quantize(lhs[e], &mut b);
        lhsq.push(b);
    }
    let pool = gemv_pool::pool();
    let (chunk_cols, chunks_per_expert) = gemv_grid(1, n, pool.threads);
    let total = ne * chunks_per_expert;
    let out_ptrs: Vec<SendMutPtr<f32>> = outs
        .iter_mut()
        .map(|o| SendMutPtr(o.as_mut_ptr()))
        .collect();
    pool.run(total, &|chunk| {
        let e = chunk / chunks_per_expert;
        let ci = chunk % chunks_per_expert;
        let col0 = ci * chunk_cols;
        let col1 = (col0 + chunk_cols).min(n);
        if col1 <= col0 {
            return;
        }
        let xq = lhsq[e].as_slice();
        let re = rhs[e];
        let op = &out_ptrs[e];
        for col in col0..col1 {
            let rhs_col = &re[col * k_in_blocks..(col + 1) * k_in_blocks];
            let v = T::dot(rhs_col, xq);
            // SAFETY: each (expert e, column col) is written exactly once.
            unsafe { *op.0.add(col) = v };
        }
    });
    Ok(())
}

/// f16-carrier variant (gpt-oss attention path: parallel `matmul_f16`).
pub fn matmul_f16_bytes(
    dtype: GgmlDType,
    (m, k, n): (usize, usize, usize),
    lhs: &[f16],
    rhs_bytes: &[u8],
    dst: &mut [f16],
) -> Result<()> {
    with_blocks!(dtype, T => {
        if !k.is_multiple_of(T::BLOCK_LEN) {
            return Err(Error(format!(
                "quant_cpu matmul_f16: k {k} not a multiple of {:?} block size {}",
                dtype, T::BLOCK_LEN
            )));
        }
        let rhs: &[T] = cast_blocks(rhs_bytes)?;
        if rhs.len() != n * (k / T::BLOCK_LEN) {
            return Err(Error(format!(
                "quant_cpu matmul_f16: rhs has {} blocks, want {}",
                rhs.len(), n * (k / T::BLOCK_LEN)
            )));
        }
        if dst.len() != m * n {
            return Err(Error(format!(
                "quant_cpu matmul_f16: dst len {} != {m}x{n}", dst.len()
            )));
        }
        matmul_f16((m, k, n), lhs, rhs, dst)
    })
}

/// Dtype-dispatched byte entry for [`matmul_f16_gate_up_silu`]. `gate_bytes`
/// and `up_bytes` are the raw `[n, k]` block-quant weights (same dtype/shape).
pub fn matmul_f16_gate_up_silu_bytes(
    dtype: GgmlDType,
    (m, k, n): (usize, usize, usize),
    lhs: &[f16],
    gate_bytes: &[u8],
    up_bytes: &[u8],
    dst: &mut [f16],
) -> Result<()> {
    with_blocks!(dtype, T => {
        if !k.is_multiple_of(T::BLOCK_LEN) {
            return Err(Error(format!(
                "gate_up_silu: k {k} not a multiple of {:?} block size {}", dtype, T::BLOCK_LEN)));
        }
        let want = n * (k / T::BLOCK_LEN);
        let gate: &[T] = cast_blocks(gate_bytes)?;
        let up: &[T] = cast_blocks(up_bytes)?;
        if gate.len() != want || up.len() != want {
            return Err(Error(format!(
                "gate_up_silu: gate/up have {}/{} blocks, want {want}", gate.len(), up.len())));
        }
        if dst.len() != m * n {
            return Err(Error(format!("gate_up_silu: dst len {} != {m}x{n}", dst.len())));
        }
        matmul_f16_gate_up_silu((m, k, n), lhs, gate, up, dst)
    })
}

/// Quantize `xs` into raw block bytes of `dtype` (tests + KV/expert tooling).
pub fn from_float_bytes(dtype: GgmlDType, xs: &[f32]) -> Result<Vec<u8>> {
    with_blocks!(dtype, T => {
        if !xs.len().is_multiple_of(T::BLOCK_LEN) {
            return Err(Error(format!(
                "quant_cpu quantize: {} elems not a multiple of {:?} block size {}",
                xs.len(), dtype, T::BLOCK_LEN
            )));
        }
        let mut blocks = vec![T::zeros(); xs.len() / T::BLOCK_LEN];
        T::quantize(xs, &mut blocks);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<T>(),
            )
        };
        Ok(bytes.to_vec())
    })
}

#[cfg(test)]
mod q8_1_dequantize_test {
    use super::*;

    /// Q8_1 stores a scale and a block sum; only the scale reconstructs the values, and
    /// `s` must stay out of it. A block quantised to 8 bits cannot come back further than
    /// half a step from where it started, so that bound is the check - it fails both if
    /// the scale is dropped and if `s` is mistakenly folded in.
    #[test]
    pub(super) fn round_trip_stays_within_half_a_quantisation_step() {
        // A block whose largest value decides the scale, plus a block of one sign, so the
        // sum term is far from zero and would show up if it leaked into the output.
        let mut x = vec![0f32; 4 * QK8_1];
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f32;
            *v = match i / QK8_1 {
                0 => (t * 0.31).sin() * 3.0,
                1 => 12.0 + (t * 0.11).cos(), // all positive: a large block sum
                2 => 0.0,                     // all zero: the scale is zero, not a divide
                _ => (t * 0.07).sin() * 217.0, // granite's amplitude
            };
        }
        let mut q = vec![BlockQ8_1::zeros(); x.len() / QK8_1];
        BlockQ8_1::quantize(&x, &mut q);
        let mut back = vec![0f32; x.len()];
        BlockQ8_1::dequantize(&q, &mut back);

        for (b, blk) in q.iter().enumerate() {
            let step = f16::to_f32(blk.d);
            for j in 0..QK8_1 {
                let i = b * QK8_1 + j;
                let err = (x[i] - back[i]).abs();
                assert!(
                    err <= step * 0.5 + 1e-4,
                    "block {b} lane {j}: {} came back as {} (step {step})",
                    x[i],
                    back[i]
                );
            }
        }
        // The all-zero block must come back exactly zero rather than NaN from a 1/0.
        for v in &back[2 * QK8_1..3 * QK8_1] {
            assert_eq!(*v, 0.0);
        }
    }
}

#[cfg(test)]
mod matmul_bytes_multi_test {
    use super::*;

    // The flat multi-expert region must be BIT-IDENTICAL to calling the
    // production `matmul_bytes((1,k,n), ...)` once per expert - same dot,
    // same Q8 activation quantize, only the work scheduling differs.
    #[test]
    pub(super) fn multi_matches_per_expert() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for &dtype in &[GgmlDType::Q4K, GgmlDType::Q6K] {
            let (k, n, ne) = (512usize, 96usize, 5usize);
            let mut rhs_bytes: Vec<Vec<u8>> = Vec::new();
            let mut lhs: Vec<Vec<f32>> = Vec::new();
            for _ in 0..ne {
                let wf: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
                rhs_bytes.push(from_float_bytes(dtype, &wf).unwrap());
                lhs.push((0..k).map(|_| next()).collect());
            }
            // Reference: per-expert matmul_bytes.
            let mut want: Vec<Vec<f32>> = (0..ne).map(|_| vec![0f32; n]).collect();
            for e in 0..ne {
                matmul_bytes(dtype, (1, k, n), &lhs[e], &rhs_bytes[e], &mut want[e]).unwrap();
            }
            // Flat region.
            let mut got: Vec<Vec<f32>> = (0..ne).map(|_| vec![0f32; n]).collect();
            {
                let rb: Vec<&[u8]> = rhs_bytes.iter().map(|v| v.as_slice()).collect();
                let lv: Vec<&[f32]> = lhs.iter().map(|v| v.as_slice()).collect();
                let mut ov: Vec<&mut [f32]> = got.iter_mut().map(|v| v.as_mut_slice()).collect();
                matmul_bytes_multi(dtype, k, n, &rb, &lv, &mut ov).unwrap();
            }
            for e in 0..ne {
                for c in 0..n {
                    assert_eq!(
                        got[e][c].to_bits(),
                        want[e][c].to_bits(),
                        "{dtype:?} expert {e} col {c}: {} vs {}",
                        got[e][c],
                        want[e][c]
                    );
                }
            }
        }
    }
}

#[cfg(all(test, target_feature = "avx2"))]
mod q5_0_avx_tests {
    use super::*;

    #[test]
    pub(super) fn q5_0_avx_dot_matches_unopt() {
        // Cover several k (all %32==0): expert width, d_inner, in_proj width,
        // and a couple more - this kernel now runs on EVERY Q5_0 tensor, so
        // the qh-expand identity must hold across widths, not just k=2688.
        let mut state = 0x12345678u64;
        let mut rnd = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for &k in &[32usize, 64, 2688, 4096, 6144, 8192] {
            let xf: Vec<f32> = (0..k).map(|_| rnd()).collect();
            let yf: Vec<f32> = (0..k).map(|_| rnd()).collect();
            let mut xq = vec![BlockQ5_0::zeros(); k / QK5_0];
            let mut yq = vec![BlockQ8_0::zeros(); k / QK8_0];
            BlockQ5_0::quantize(&xf, &mut xq);
            BlockQ8_0::quantize(&yf, &mut yq);
            let scalar = BlockQ5_0::dot_scalar(&xq, &yq);
            let simd = avx::vec_dot_q5_0_q8_0(&xq, &yq);
            assert!(
                (scalar - simd).abs() <= scalar.abs().max(1.0) * 1e-5,
                "q5_0 avx k={k}: {simd} != scalar {scalar}"
            );
        }
    }
}

#[cfg(test)]
mod gate_up_silu_tests {
    use super::*;

    /// The fused dense SwiGLU (`matmul_f16_gate_up_silu`) must equal the unfused
    /// `gate matmul_f16 -> silu`, `up matmul_f16`, `mul` it replaces. Both share
    /// the same `dot`; the only numeric difference is the fused path keeps
    /// g/u in f32 through silu.mul while the reference rounds them to f16 first,
    /// so compare within an f16-ULP tolerance rather than bit-exact.
    #[test]
    pub(super) fn fused_gate_up_silu_matches_reference() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for &dtype in &[GgmlDType::Q4K, GgmlDType::Q6K] {
            let (k, n) = (512usize, 128usize);
            let gate: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
            let up: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
            let gb = from_float_bytes(dtype, &gate).unwrap();
            let ub = from_float_bytes(dtype, &up).unwrap();
            let xf: Vec<f32> = (0..k).map(|_| next()).collect();
            let xh: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
            // fused
            let mut got = vec![f16::ZERO; n];
            matmul_f16_gate_up_silu_bytes(dtype, (1, k, n), &xh, &gb, &ub, &mut got).unwrap();
            // reference: separate gate/up projections + silu.mul
            let mut gd = vec![f16::ZERO; n];
            let mut ud = vec![f16::ZERO; n];
            matmul_f16_bytes(dtype, (1, k, n), &xh, &gb, &mut gd).unwrap();
            matmul_f16_bytes(dtype, (1, k, n), &xh, &ub, &mut ud).unwrap();
            for j in 0..n {
                let g = gd[j].to_f32();
                let u = ud[j].to_f32();
                let want = f16::from_f32((g / (1.0 + (-g).exp())) * u).to_f32();
                let have = got[j].to_f32();
                assert!(
                    (have - want).abs() <= want.abs().max(1.0) * 2e-2,
                    "{dtype:?} col {j}: fused {have} vs ref {want}"
                );
            }
        }
    }
}
