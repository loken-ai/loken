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

pub use super::repack_mxfp4_x8::{quantize_act, BlockQ8_0x4};
use super::BlockQ8_0;
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use half::f16;

#[repr(C)]
#[derive(Clone)]
pub struct BlockQ8_0x8 {
    pub d: [f16; 8],
    pub qs: [i8; 256],
}

/// Repack raw row-major `[n][nb]` `BlockQ8_0` into `[n/8][nb]` `BlockQ8_0x8`.
pub fn repack(raw: &[BlockQ8_0], n: usize, nb: usize) -> Vec<BlockQ8_0x8> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(raw.len(), n * nb);
    let groups = n / 8;
    let mut out = vec![
        BlockQ8_0x8 {
            d: [f16::ZERO; 8],
            qs: [0i8; 256]
        };
        groups * nb
    ];
    for g in 0..groups {
        for l in 0..nb {
            let ob = &mut out[g * nb + l];
            for c in 0..8 {
                ob.d[c] = raw[(g * 8 + c) * nb + l].d;
            }
            // element-group eg (8 elements) x [cols 0-3][cols 4-7].
            for eg in 0..4 {
                for cc in 0..4 {
                    let dl = eg * 64 + cc * 8;
                    let dh = eg * 64 + 32 + cc * 8;
                    ob.qs[dl..dl + 8]
                        .copy_from_slice(&raw[(g * 8 + cc) * nb + l].qs[eg * 8..eg * 8 + 8]);
                    ob.qs[dh..dh + 8]
                        .copy_from_slice(&raw[(g * 8 + 4 + cc) * nb + l].qs[eg * 8..eg * 8 + 8]);
                }
            }
        }
    }
    out
}

/// Scalar oracle: one 4-row x 8-column tile over `nb` blocks. `out[m*8 + j]`.
/// Integer accumulation is order-independent (i8xi8->i32 exact), so the AVX2
/// kernel must match this to rel<1e-4 (float-scale rounding only).
pub fn gemm_scalar(a: &[BlockQ8_0x4], b: &[BlockQ8_0x8], nb: usize, out: &mut [f32; 32]) {
    let mut sumf = [[0f32; 8]; 4];
    for l in 0..nb {
        for eg in 0..4 {
            for m in 0..4 {
                for j in 0..8 {
                    let half = if j < 4 { 0 } else { 32 };
                    let base = eg * 64 + half + (j % 4) * 8;
                    let mut sumi = 0i32;
                    for i in 0..8 {
                        let w = b[l].qs[base + i] as i32;
                        let av = a[l].qs[eg * 32 + m * 8 + i] as i32;
                        sumi += w * av;
                    }
                    sumf[m][j] += sumi as f32 * b[l].d[j].to_f32() * a[l].d[m].to_f32();
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

/// Full matmul via the scalar oracle (layout validation). `raw` is row-major
/// `[n][nb]` Q8_0 weight; `dst` is f32 row-major `[m, n]`.
pub fn matmul_scalar(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    raw: &[BlockQ8_0],
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

/// Decode one Q8_0 weight block: load the 4 element-groups (no nibble decode  -
/// weights are already i8), blend into the `{0,1,4,5}`/`{2,3,6,7}` arrangement,
/// apply the sp1/sp2 shuffles, load the 8 f16 column scales. Returns the same
/// shape `accum_4rows` consumes.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn decode_weights_q8(
    bb: &BlockQ8_0x8,
    required_order: __m256i,
) -> (
    [__m256i; 4],
    [__m256i; 4],
    [__m256i; 4],
    [__m256i; 4],
    __m256,
) {
    let bq = bb.qs.as_ptr();
    let mut w0145 = [_mm256_setzero_si256(); 4];
    let mut w2367 = [_mm256_setzero_si256(); 4];
    for eg in 0..4 {
        let rr_lo = _mm256_loadu_si256(bq.add(eg * 64) as *const __m256i);
        let rr_hi = _mm256_loadu_si256(bq.add(eg * 64 + 32) as *const __m256i);
        w0145[eg] =
            _mm256_blend_epi32::<240>(rr_lo, _mm256_permutevar8x32_epi32(rr_hi, required_order));
        w2367[eg] =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(rr_lo, required_order), rr_hi);
    }
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
    let col_scale = _mm256_cvtph_ps(_mm_loadu_si128(bb.d.as_ptr() as *const __m128i));
    (
        sp1(&w0145),
        sp1(&w2367),
        sp2(&w0145),
        sp2(&w2367),
        col_scale,
    )
}

/// AVX2 column-interleaved Q8_0 GEMM: one 4-row x 8-column tile. Reuses the
/// quant-agnostic `accum_4rows` from `repack_mxfp4_x8`. `out[m*8 + j]`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2(
    a: &[BlockQ8_0x4],
    b: &[BlockQ8_0x8],
    nb: usize,
    out: &mut [f32; 32],
) {
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    let load_mask = _mm_blend_epi32::<3>(_mm_setzero_si128(), _mm_set1_epi32(-1));
    let mut acc = [_mm256_setzero_ps(); 4];
    for blk in 0..nb {
        let (s10, s12, s20, s22, cs) = decode_weights_q8(&b[blk], required_order);
        super::repack_mxfp4_x8::accum_4rows(
            &s10, &s12, &s20, &s22, cs, &a[blk], load_mask, &mut acc,
        );
    }
    for (m, acc) in acc.iter().enumerate() {
        _mm256_storeu_ps(out.as_mut_ptr().add(m * 8), *acc);
    }
}

/// 16-row x 8-column tile - decode each weight block ONCE, apply to 4
/// activation groups. `out[rp*32 + m*8 + j]`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group16_avx2(
    a0: &[BlockQ8_0x4],
    a1: &[BlockQ8_0x4],
    a2: &[BlockQ8_0x4],
    a3: &[BlockQ8_0x4],
    b: &[BlockQ8_0x8],
    nb: usize,
    out: &mut [f32; 128],
) {
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    let load_mask = _mm_blend_epi32::<3>(_mm_setzero_si128(), _mm_set1_epi32(-1));
    let ag = [a0, a1, a2, a3];
    let mut acc = [_mm256_setzero_ps(); 16];
    for blk in 0..nb {
        let (s10, s12, s20, s22, cs) = decode_weights_q8(&b[blk], required_order);
        for rp in 0..4 {
            super::repack_mxfp4_x8::accum_4rows(
                &s10,
                &s12,
                &s20,
                &s22,
                cs,
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
