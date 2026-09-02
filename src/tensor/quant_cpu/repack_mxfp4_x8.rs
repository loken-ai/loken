// Transcribed kernels: the index arithmetic IS the layout, the argument lists are the
// reference's, and the `unsafe fn`s wrap intrinsics whose contract is the intrinsic's.
// Named rather than `clippy::all` so anything else here still gets reported.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    clippy::type_complexity,
    clippy::redundant_closure
)]

use super::{BlockMxFp4, KVALUES_MXFP4};
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use half::f16;

/// 8 columns interleaved. `e[c]` = column c's E8M0 scale; `qs` holds, in 16
/// chunks of 8 bytes, `[col0.lo8, col1.lo8, .., col7.lo8, col0.hi8, .., col7.hi8]`
/// where each raw column block is 16 bytes (lo8 = bytes 0..8, hi8 = bytes 8..16).
#[repr(C)]
#[derive(Clone)]
pub struct BlockMxFp4x8 {
    pub e: [u8; 8],
    pub qs: [u8; 128],
}

/// 4 activation rows interleaved. `d[m]` = row m's Q8_0 block scale; element e
/// of row m lives at `qs[(e/8)*32 + m*8 + (e%8)]` (the layout the 8x8 gemm reads
/// as `qs[k*32 + m*8 + i]` low / `+64` high).
#[repr(C)]
#[derive(Clone)]
pub struct BlockQ8_0x4 {
    pub d: [f16; 4],
    pub qs: [i8; 128],
}

/// Repack raw row-major `[n][nb]` `BlockMxFp4` into `[n/8][nb]` `BlockMxFp4x8`
/// (mirror of llama.cpp `make_block_mxfp4x8`, `blck_size_interleave = 8`).
pub fn repack(raw: &[BlockMxFp4], n: usize, nb: usize) -> Vec<BlockMxFp4x8> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(raw.len(), n * nb);
    let groups = n / 8;
    let mut out = vec![
        BlockMxFp4x8 {
            e: [0u8; 8],
            qs: [0u8; 128]
        };
        groups * nb
    ];
    for g in 0..groups {
        for l in 0..nb {
            let ob = &mut out[g * nb + l];
            for c in 0..8 {
                ob.e[c] = raw[(g * 8 + c) * nb + l].e;
            }
            for i in 0..16 {
                let col = i % 8;
                let half = i / 8;
                let rb = &raw[(g * 8 + col) * nb + l];
                ob.qs[i * 8..i * 8 + 8].copy_from_slice(&rb.qs[half * 8..half * 8 + 8]);
            }
        }
    }
    out
}

/// Quantize a row-major `[m, k]` f32 activation into `[ceil(m/4)][nb]`
/// `BlockQ8_0x4` (Q8_0 per 32-block per row, rows interleaved). Rows past `m`
/// are left zero (scale 0), so padding contributes nothing.
pub fn quantize_act(lhs: &[f32], m: usize, k: usize) -> Vec<BlockQ8_0x4> {
    debug_assert_eq!(k % 32, 0);
    let nb = k / 32;
    let mgroups = m.div_ceil(4);
    let mut out = vec![
        BlockQ8_0x4 {
            d: [f16::ZERO; 4],
            qs: [0i8; 128]
        };
        mgroups * nb
    ];
    // Quantize on the shared spin-pool (one row-group per chunk) instead of
    // serially on the caller thread. This ran serial inside every Q8_0 matmul
    // while the pool's workers busy-waited at the previous matmul's barrier  -
    // the Q4_K/Q8K activation quantize is already pooled; this aligns Q8_0.
    let out_ptr = super::SendMutPtr(out.as_mut_ptr());
    super::gemv_pool::pool().run(mgroups, &|mg| {
        let out_ptr = &out_ptr;
        for l in 0..nb {
            // SAFETY: chunk `mg` owns the disjoint `out[mg*nb .. (mg+1)*nb]` range.
            let ob = unsafe { &mut *out_ptr.0.add(mg * nb + l) };
            for mm in 0..4 {
                let row = mg * 4 + mm;
                if row >= m {
                    continue;
                }
                let blk = &lhs[row * k + l * 32..row * k + l * 32 + 32];
                let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                ob.d[mm] = f16::from_f32(d);
                for (e, &x) in blk.iter().enumerate() {
                    let q = (x * id).round().clamp(-127.0, 127.0) as i8;
                    ob.qs[(e / 8) * 32 + mm * 8 + (e % 8)] = q;
                }
            }
        }
    });
    out
}

/// Scalar oracle: one 4-row x 8-column tile over `nb` blocks. `out[m*8 + j]`.
/// Mirrors `ggml_gemm_mxfp4_8x8_q8_0_generic` exactly (the AVX2 kernel must
/// match this bit-for-bit).
pub fn gemm_scalar(a: &[BlockQ8_0x4], b: &[BlockMxFp4x8], nb: usize, out: &mut [f32; 32]) {
    let mut sumf = [[0f32; 8]; 4];
    for l in 0..nb {
        for kk in 0..2 {
            for m in 0..4 {
                for j in 0..8 {
                    let mut sumi = 0i32;
                    for i in 0..8 {
                        let byte = b[l].qs[kk * 64 + j * 8 + i];
                        let v0 = KVALUES_MXFP4[(byte & 0x0F) as usize] as i32;
                        let v1 = KVALUES_MXFP4[(byte >> 4) as usize] as i32;
                        let a_lo = a[l].qs[kk * 32 + m * 8 + i] as i32;
                        let a_hi = a[l].qs[kk * 32 + m * 8 + i + 64] as i32;
                        sumi += v0 * a_lo + v1 * a_hi;
                    }
                    sumf[m][j] +=
                        sumi as f32 * super::e8m0_to_fp32_half(b[l].e[j]) * a[l].d[m].to_f32();
                }
            }
        }
    }
    for m in 0..4 {
        for j in 0..8 {
            out[m * 8 + j] = sumf[m][j];
        }
    }
}

/// Full matmul via the scalar oracle: `[m,k] x [n,k]ᵀ -> [m,n]` f32 row-major.
/// `raw` is the row-major `[n][nb]` MXFP4 weight. Used to validate the layout
/// and (later) the AVX2 kernel.
pub fn matmul_scalar(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    raw: &[BlockMxFp4],
    dst: &mut [f32],
) {
    let nb = k / 32;
    let wx8 = repack(raw, n, nb);
    let ax4 = quantize_act(lhs, m, k);
    let mgroups = m.div_ceil(4);
    let groups = n / 8;
    for mg in 0..mgroups {
        for g in 0..groups {
            let mut local = [0f32; 32];
            gemm_scalar(
                &ax4[mg * nb..(mg + 1) * nb],
                &wx8[g * nb..(g + 1) * nb],
                nb,
                &mut local,
            );
            for mm in 0..4 {
                let row = mg * 4 + mm;
                if row >= m {
                    continue;
                }
                for j in 0..8 {
                    dst[row * n + g * 8 + j] = local[mm * 8 + j];
                }
            }
        }
    }
}

/// `acc + madd(ones, maddubs(|x|, sign(y,x)))` - the int32-accumulating i8
/// pairwise dot (llama.cpp `mul_sum_i8_pairs_acc_int32x8`, non-VNNI path).
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn madd_acc(acc: __m256i, x: __m256i, y: __m256i) -> __m256i {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let dot = _mm256_maddubs_epi16(ax, sy);
    _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi16(1), dot))
}

/// AVX2 column-interleaved GEMM: one 4-row x 8-column tile over `nb` blocks,
/// with the 8 output columns held in the 8 SIMD lanes (no per-column hsum).
/// Faithful port of llama.cpp's `gemm_q4_b32_8x8_q8_0_lut_avx` (AVX2 path,
/// mxfp4 branch). Writes `out[m*8 + j]` for m in 0..4, j in 0..8. Bit-identical
/// to `gemm_scalar`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2(
    a: &[BlockQ8_0x4],
    b: &[BlockMxFp4x8],
    nb: usize,
    out: &mut [f32; 32],
) {
    let m4b = _mm256_set1_epi8(0x0F);
    let lut =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i));
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    let load_mask = _mm_blend_epi32::<3>(_mm_setzero_si128(), _mm_set1_epi32(-1));
    let mut acc_rows = [_mm256_setzero_ps(); 4];

    for blk in 0..nb {
        let bq = b[blk].qs.as_ptr();
        let rr0 = _mm256_loadu_si256(bq as *const __m256i);
        let rr1 = _mm256_loadu_si256(bq.add(32) as *const __m256i);
        let rr2 = _mm256_loadu_si256(bq.add(64) as *const __m256i);
        let rr3 = _mm256_loadu_si256(bq.add(96) as *const __m256i);

        let rm_0145_0 =
            _mm256_blend_epi32::<240>(rr0, _mm256_permutevar8x32_epi32(rr1, required_order));
        let rm_2367_0 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(rr0, required_order), rr1);
        let rm_0145_1 =
            _mm256_blend_epi32::<240>(rr2, _mm256_permutevar8x32_epi32(rr3, required_order));
        let rm_2367_1 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(rr2, required_order), rr3);

        let w0145_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(rm_0145_0, m4b));
        let w2367_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(rm_2367_0, m4b));
        let w0145_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(rm_0145_1, m4b));
        let w2367_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(rm_2367_1, m4b));
        let w0145_2 = _mm256_shuffle_epi8(
            lut,
            _mm256_and_si256(_mm256_srli_epi16::<4>(rm_0145_0), m4b),
        );
        let w2367_2 = _mm256_shuffle_epi8(
            lut,
            _mm256_and_si256(_mm256_srli_epi16::<4>(rm_2367_0), m4b),
        );
        let w0145_3 = _mm256_shuffle_epi8(
            lut,
            _mm256_and_si256(_mm256_srli_epi16::<4>(rm_0145_1), m4b),
        );
        let w2367_3 = _mm256_shuffle_epi8(
            lut,
            _mm256_and_si256(_mm256_srli_epi16::<4>(rm_2367_1), m4b),
        );

        // sp1 = shuffle 136 (0-3 lanes), sp2 = shuffle 221 (4-7 lanes).
        let w0145_0_1 = _mm256_shuffle_epi32::<136>(w0145_0);
        let w2367_0_1 = _mm256_shuffle_epi32::<136>(w2367_0);
        let w0145_1_1 = _mm256_shuffle_epi32::<136>(w0145_1);
        let w2367_1_1 = _mm256_shuffle_epi32::<136>(w2367_1);
        let w0145_2_1 = _mm256_shuffle_epi32::<136>(w0145_2);
        let w2367_2_1 = _mm256_shuffle_epi32::<136>(w2367_2);
        let w0145_3_1 = _mm256_shuffle_epi32::<136>(w0145_3);
        let w2367_3_1 = _mm256_shuffle_epi32::<136>(w2367_3);
        let w0145_0_2 = _mm256_shuffle_epi32::<221>(w0145_0);
        let w2367_0_2 = _mm256_shuffle_epi32::<221>(w2367_0);
        let w0145_1_2 = _mm256_shuffle_epi32::<221>(w0145_1);
        let w2367_1_2 = _mm256_shuffle_epi32::<221>(w2367_1);
        let w0145_2_2 = _mm256_shuffle_epi32::<221>(w0145_2);
        let w2367_2_2 = _mm256_shuffle_epi32::<221>(w2367_2);
        let w0145_3_2 = _mm256_shuffle_epi32::<221>(w0145_3);
        let w2367_3_2 = _mm256_shuffle_epi32::<221>(w2367_3);

        let e = &b[blk].e;
        let col_scale = _mm256_set_ps(
            super::e8m0_to_fp32_half(e[7]),
            super::e8m0_to_fp32_half(e[6]),
            super::e8m0_to_fp32_half(e[5]),
            super::e8m0_to_fp32_half(e[4]),
            super::e8m0_to_fp32_half(e[3]),
            super::e8m0_to_fp32_half(e[2]),
            super::e8m0_to_fp32_half(e[1]),
            super::e8m0_to_fp32_half(e[0]),
        );

        let aq = a[blk].qs.as_ptr();
        let l0 = _mm256_loadu_si256(aq as *const __m256i);
        let l01_0 = _mm256_permute2f128_si256::<0>(l0, l0);
        let l23_0 = _mm256_permute2f128_si256::<17>(l0, l0);
        let l1 = _mm256_loadu_si256(aq.add(32) as *const __m256i);
        let l01_1 = _mm256_permute2f128_si256::<0>(l1, l1);
        let l23_1 = _mm256_permute2f128_si256::<17>(l1, l1);
        let l2 = _mm256_loadu_si256(aq.add(64) as *const __m256i);
        let l01_2 = _mm256_permute2f128_si256::<0>(l2, l2);
        let l23_2 = _mm256_permute2f128_si256::<17>(l2, l2);
        let l3 = _mm256_loadu_si256(aq.add(96) as *const __m256i);
        let l01_3 = _mm256_permute2f128_si256::<0>(l3, l3);
        let l23_3 = _mm256_permute2f128_si256::<17>(l3, l3);

        // sp1 = shuffle 160, sp2 = shuffle 245.
        let a01_0_1 = _mm256_shuffle_epi32::<160>(l01_0);
        let a23_0_1 = _mm256_shuffle_epi32::<160>(l23_0);
        let a01_1_1 = _mm256_shuffle_epi32::<160>(l01_1);
        let a23_1_1 = _mm256_shuffle_epi32::<160>(l23_1);
        let a01_2_1 = _mm256_shuffle_epi32::<160>(l01_2);
        let a23_2_1 = _mm256_shuffle_epi32::<160>(l23_2);
        let a01_3_1 = _mm256_shuffle_epi32::<160>(l01_3);
        let a23_3_1 = _mm256_shuffle_epi32::<160>(l23_3);
        let a01_0_2 = _mm256_shuffle_epi32::<245>(l01_0);
        let a23_0_2 = _mm256_shuffle_epi32::<245>(l23_0);
        let a01_1_2 = _mm256_shuffle_epi32::<245>(l01_1);
        let a23_1_2 = _mm256_shuffle_epi32::<245>(l23_1);
        let a01_2_2 = _mm256_shuffle_epi32::<245>(l01_2);
        let a23_2_2 = _mm256_shuffle_epi32::<245>(l23_2);
        let a01_3_2 = _mm256_shuffle_epi32::<245>(l01_3);
        let a23_3_2 = _mm256_shuffle_epi32::<245>(l23_3);

        let z = _mm256_setzero_si256();
        let iacc_00_1 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a01_3_1, w0145_3_1), a01_2_1, w0145_2_1),
                a01_1_1,
                w0145_1_1,
            ),
            a01_0_1,
            w0145_0_1,
        );
        let iacc_01_1 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a01_3_1, w2367_3_1), a01_2_1, w2367_2_1),
                a01_1_1,
                w2367_1_1,
            ),
            a01_0_1,
            w2367_0_1,
        );
        let iacc_10_1 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a23_3_1, w0145_3_1), a23_2_1, w0145_2_1),
                a23_1_1,
                w0145_1_1,
            ),
            a23_0_1,
            w0145_0_1,
        );
        let iacc_11_1 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a23_3_1, w2367_3_1), a23_2_1, w2367_2_1),
                a23_1_1,
                w2367_1_1,
            ),
            a23_0_1,
            w2367_0_1,
        );
        let iacc_00_2 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a01_3_2, w0145_3_2), a01_2_2, w0145_2_2),
                a01_1_2,
                w0145_1_2,
            ),
            a01_0_2,
            w0145_0_2,
        );
        let iacc_01_2 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a01_3_2, w2367_3_2), a01_2_2, w2367_2_2),
                a01_1_2,
                w2367_1_2,
            ),
            a01_0_2,
            w2367_0_2,
        );
        let iacc_10_2 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a23_3_2, w0145_3_2), a23_2_2, w0145_2_2),
                a23_1_2,
                w0145_1_2,
            ),
            a23_0_2,
            w0145_0_2,
        );
        let iacc_11_2 = madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, a23_3_2, w2367_3_2), a23_2_2, w2367_2_2),
                a23_1_2,
                w2367_1_2,
            ),
            a23_0_2,
            w2367_0_2,
        );

        let iacc_00 = _mm256_add_epi32(iacc_00_1, iacc_00_2);
        let iacc_01 = _mm256_add_epi32(iacc_01_1, iacc_01_2);
        let iacc_10 = _mm256_add_epi32(iacc_10_1, iacc_10_2);
        let iacc_11 = _mm256_add_epi32(iacc_11_1, iacc_11_2);

        let row0 = _mm256_blend_epi32::<204>(iacc_00, _mm256_shuffle_epi32::<78>(iacc_01));
        let row1 = _mm256_blend_epi32::<204>(_mm256_shuffle_epi32::<78>(iacc_00), iacc_01);
        let row2 = _mm256_blend_epi32::<204>(iacc_10, _mm256_shuffle_epi32::<78>(iacc_11));
        let row3 = _mm256_blend_epi32::<204>(_mm256_shuffle_epi32::<78>(iacc_10), iacc_11);

        let raw = _mm_maskload_epi32(a[blk].d.as_ptr() as *const i32, load_mask);
        let rs = _mm256_cvtph_ps(_mm_shuffle_epi32::<68>(raw));
        acc_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row0),
            _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<0>(rs, rs)),
            acc_rows[0],
        );
        acc_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row1),
            _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<85>(rs, rs)),
            acc_rows[1],
        );
        acc_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row2),
            _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<170>(rs, rs)),
            acc_rows[2],
        );
        acc_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row3),
            _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<255>(rs, rs)),
            acc_rows[3],
        );
    }
    for (m, acc) in acc_rows.iter().enumerate() {
        _mm256_storeu_ps(out.as_mut_ptr().add(m * 8), *acc);
    }
}

/// Accumulate one 4-row activation block (`a`) against the pre-decoded weight
/// shuffle vectors of one weight block into `acc[0..4]`. Split out so the
/// 16-row kernel decodes each weight block ONCE and applies it to 4 row-groups
/// (the weight decode + `col_scale` are the per-block work the 4-row tile
/// otherwise repeats). `acc` must have length 4.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn accum_4rows(
    w0145_sp1: &[__m256i; 4],
    w2367_sp1: &[__m256i; 4],
    w0145_sp2: &[__m256i; 4],
    w2367_sp2: &[__m256i; 4],
    col_scale: __m256,
    a: &BlockQ8_0x4,
    load_mask: __m128i,
    acc: &mut [__m256],
) {
    let aq = a.qs.as_ptr();
    let l0 = _mm256_loadu_si256(aq as *const __m256i);
    let l1 = _mm256_loadu_si256(aq.add(32) as *const __m256i);
    let l2 = _mm256_loadu_si256(aq.add(64) as *const __m256i);
    let l3 = _mm256_loadu_si256(aq.add(96) as *const __m256i);
    let l01 = [
        _mm256_permute2f128_si256::<0>(l0, l0),
        _mm256_permute2f128_si256::<0>(l1, l1),
        _mm256_permute2f128_si256::<0>(l2, l2),
        _mm256_permute2f128_si256::<0>(l3, l3),
    ];
    let l23 = [
        _mm256_permute2f128_si256::<17>(l0, l0),
        _mm256_permute2f128_si256::<17>(l1, l1),
        _mm256_permute2f128_si256::<17>(l2, l2),
        _mm256_permute2f128_si256::<17>(l3, l3),
    ];
    let a01_1 = [
        _mm256_shuffle_epi32::<160>(l01[0]),
        _mm256_shuffle_epi32::<160>(l01[1]),
        _mm256_shuffle_epi32::<160>(l01[2]),
        _mm256_shuffle_epi32::<160>(l01[3]),
    ];
    let a23_1 = [
        _mm256_shuffle_epi32::<160>(l23[0]),
        _mm256_shuffle_epi32::<160>(l23[1]),
        _mm256_shuffle_epi32::<160>(l23[2]),
        _mm256_shuffle_epi32::<160>(l23[3]),
    ];
    let a01_2 = [
        _mm256_shuffle_epi32::<245>(l01[0]),
        _mm256_shuffle_epi32::<245>(l01[1]),
        _mm256_shuffle_epi32::<245>(l01[2]),
        _mm256_shuffle_epi32::<245>(l01[3]),
    ];
    let a23_2 = [
        _mm256_shuffle_epi32::<245>(l23[0]),
        _mm256_shuffle_epi32::<245>(l23[1]),
        _mm256_shuffle_epi32::<245>(l23[2]),
        _mm256_shuffle_epi32::<245>(l23[3]),
    ];
    let z = _mm256_setzero_si256();
    let acc4 = |aa: &[__m256i; 4], ww: &[__m256i; 4]| -> __m256i {
        madd_acc(
            madd_acc(
                madd_acc(madd_acc(z, aa[3], ww[3]), aa[2], ww[2]),
                aa[1],
                ww[1],
            ),
            aa[0],
            ww[0],
        )
    };
    let iacc_00 = _mm256_add_epi32(acc4(&a01_1, w0145_sp1), acc4(&a01_2, w0145_sp2));
    let iacc_01 = _mm256_add_epi32(acc4(&a01_1, w2367_sp1), acc4(&a01_2, w2367_sp2));
    let iacc_10 = _mm256_add_epi32(acc4(&a23_1, w0145_sp1), acc4(&a23_2, w0145_sp2));
    let iacc_11 = _mm256_add_epi32(acc4(&a23_1, w2367_sp1), acc4(&a23_2, w2367_sp2));
    let row0 = _mm256_blend_epi32::<204>(iacc_00, _mm256_shuffle_epi32::<78>(iacc_01));
    let row1 = _mm256_blend_epi32::<204>(_mm256_shuffle_epi32::<78>(iacc_00), iacc_01);
    let row2 = _mm256_blend_epi32::<204>(iacc_10, _mm256_shuffle_epi32::<78>(iacc_11));
    let row3 = _mm256_blend_epi32::<204>(_mm256_shuffle_epi32::<78>(iacc_10), iacc_11);
    let raw = _mm_maskload_epi32(a.d.as_ptr() as *const i32, load_mask);
    let rs = _mm256_cvtph_ps(_mm_shuffle_epi32::<68>(raw));
    acc[0] = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(row0),
        _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<0>(rs, rs)),
        acc[0],
    );
    acc[1] = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(row1),
        _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<85>(rs, rs)),
        acc[1],
    );
    acc[2] = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(row2),
        _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<170>(rs, rs)),
        acc[2],
    );
    acc[3] = _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(row3),
        _mm256_mul_ps(col_scale, _mm256_shuffle_ps::<255>(rs, rs)),
        acc[3],
    );
}

/// Decode one weight block into the 16 shuffle-pattern vectors + `col_scale`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn decode_weights(
    bb: &BlockMxFp4x8,
    lut: __m256i,
    m4b: __m256i,
    required_order: __m256i,
) -> (
    [__m256i; 4],
    [__m256i; 4],
    [__m256i; 4],
    [__m256i; 4],
    __m256,
) {
    let bq = bb.qs.as_ptr();
    let rr0 = _mm256_loadu_si256(bq as *const __m256i);
    let rr1 = _mm256_loadu_si256(bq.add(32) as *const __m256i);
    let rr2 = _mm256_loadu_si256(bq.add(64) as *const __m256i);
    let rr3 = _mm256_loadu_si256(bq.add(96) as *const __m256i);
    let rm_0145_0 =
        _mm256_blend_epi32::<240>(rr0, _mm256_permutevar8x32_epi32(rr1, required_order));
    let rm_2367_0 =
        _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(rr0, required_order), rr1);
    let rm_0145_1 =
        _mm256_blend_epi32::<240>(rr2, _mm256_permutevar8x32_epi32(rr3, required_order));
    let rm_2367_1 =
        _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(rr2, required_order), rr3);
    let dec = |v: __m256i| _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4b));
    let dech =
        |v: __m256i| _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4b));
    let w0145 = [
        dec(rm_0145_0),
        dec(rm_0145_1),
        dech(rm_0145_0),
        dech(rm_0145_1),
    ];
    let w2367 = [
        dec(rm_2367_0),
        dec(rm_2367_1),
        dech(rm_2367_0),
        dech(rm_2367_1),
    ];
    let sp1 = |w: &[__m256i; 4]| {
        [
            _mm256_shuffle_epi32::<136>(w[0]),
            _mm256_shuffle_epi32::<136>(w[1]),
            _mm256_shuffle_epi32::<136>(w[2]),
            _mm256_shuffle_epi32::<136>(w[3]),
        ]
    };
    let sp2 = |w: &[__m256i; 4]| {
        [
            _mm256_shuffle_epi32::<221>(w[0]),
            _mm256_shuffle_epi32::<221>(w[1]),
            _mm256_shuffle_epi32::<221>(w[2]),
            _mm256_shuffle_epi32::<221>(w[3]),
        ]
    };
    let e = &bb.e;
    let col_scale = _mm256_set_ps(
        super::e8m0_to_fp32_half(e[7]),
        super::e8m0_to_fp32_half(e[6]),
        super::e8m0_to_fp32_half(e[5]),
        super::e8m0_to_fp32_half(e[4]),
        super::e8m0_to_fp32_half(e[3]),
        super::e8m0_to_fp32_half(e[2]),
        super::e8m0_to_fp32_half(e[1]),
        super::e8m0_to_fp32_half(e[0]),
    );
    (
        sp1(&w0145),
        sp1(&w2367),
        sp2(&w0145),
        sp2(&w2367),
        col_scale,
    )
}

/// 16-row x 8-column tile: decodes each weight block ONCE and applies it to
/// the 4 activation groups `a0..a3` (mirror of llama.cpp's main 16-row loop),
/// amortizing the weight decode + `col_scale` 4x vs the 4-row tile. Writes
/// `out[rp*32 + m*8 + j]` for rp,m in 0..4, j in 0..8 (16 rows x 8 cols).
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group16_avx2(
    a0: &[BlockQ8_0x4],
    a1: &[BlockQ8_0x4],
    a2: &[BlockQ8_0x4],
    a3: &[BlockQ8_0x4],
    b: &[BlockMxFp4x8],
    nb: usize,
    out: &mut [f32; 128],
) {
    let m4b = _mm256_set1_epi8(0x0F);
    let lut =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i));
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    let load_mask = _mm_blend_epi32::<3>(_mm_setzero_si128(), _mm_set1_epi32(-1));
    let ag = [a0, a1, a2, a3];
    let mut acc = [_mm256_setzero_ps(); 16];
    for blk in 0..nb {
        let (w0145_sp1, w2367_sp1, w0145_sp2, w2367_sp2, col_scale) =
            decode_weights(&b[blk], lut, m4b, required_order);
        for rp in 0..4 {
            accum_4rows(
                &w0145_sp1,
                &w2367_sp1,
                &w0145_sp2,
                &w2367_sp2,
                col_scale,
                &ag[rp][blk],
                load_mask,
                &mut acc[rp * 4..rp * 4 + 4],
            );
        }
    }
    for (i, a) in acc.iter().enumerate() {
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), *a);
    }
}
