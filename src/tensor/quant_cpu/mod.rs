//! The CPU quantised compute engine: the GGUF block formats, their `dot` kernels
//! (AVX2 with a scalar fallback) and the column-parallel matmuls the CPU and hybrid decode
//! cells run on.
//!
//! ATTRIBUTION (see NOTICE.md): this engine began as candle's
//! `quantized/{k_quants.rs,utils.rs,avx.rs}` and the float kernels of `cpu/{mod.rs,avx.rs}` -
//! themselves adapted from llama.cpp's `k_quants.c`, both MIT / Apache-2.0. It is being
//! rewritten from the formats, and each file is measured: no file in this directory is now
//! above 21.7% of candle by exact body line, and the median is 4%. That figure is a FLOOR -
//! an exact-line meter sees neither a reformatted line nor a restructured one.
//!
//! Layout of this DIRECTORY - ours, not the one the port arrived in:
//!
//!   format/     one file per block format: layout, dequantiser, encoder, its kernel dispatch
//!   blocks.rs   what every format shares - the `BlockFormat` trait, element counts, LUTs
//!   scale.rs    the scale searches the encoders fit with
//!   avx/ cpu.rs float.rs      the dot kernels: vectorised, then the float carriers
//!   repack_*.rs drivers.rs    the interleaved layouts and the tiled matmul over them
//!   gemv_pool.rs              the persistent executor decode issues its regions to
//!
//! Three judges, because each answers a question the others cannot:
//! `oracle_parity` pins dequantisation bit-for-bit to ggml and records what the encoders
//! reconstruct; `avx/parity` holds every vectorised kernel to its scalar twin; `cpu_parity`
//! judges the float carriers against the values they hold, which is the only way to catch a
//! carrier whose vector path and scalar path are the same function.

use super::quantized::GgmlDType;
use super::Error;
use super::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::{bf16, f16, slice::HalfFloatSliceExt};
use rayon::prelude::*;

// ===========================================================================
mod scale;
pub use scale::*;
mod blocks;
pub mod format;
// The block types keep the paths their callers already use: `quant_cpu::BlockQ4K`
// names the same struct whether it lives here or in `format/q4_k.rs`.
pub use blocks::*;
pub use format::*;
// ===========================================================================
// Persistent GEMV executor.
//
// Decode issues hundreds of short (~ms) parallel matmul regions per token.
// A work-stealing pool that parks its workers between regions pays a futex
// wake cascade on every region: by the time the last workers are awake the
// region is nearly over, so average worker utilization stays low (sampled
// ~50% on dense decode). This executor keeps a fixed set of workers spinning
// briefly after each job so back-to-back matmuls find them already hot, and
// shards the output columns through an atomic chunk counter so load balance
// is exact and the per-chunk dispatch cost is one fetch_add.
//
// Results are bit-identical to the serial loop: each output column is an
// independent dot; only the assignment of columns to threads changes.
// ===========================================================================
pub(crate) mod gemv_pool;

/// Online block-interleaved Q4_K repack + 8-column-wide GEMV (llama.cpp's
/// `block_q4_Kx8` / `ggml_gemv_q4_K_8x8_q8_K` family, adapted to AVX2-only).
///
/// The per-column `dot` path re-unpacks the 6-bit scales/mins and reloads
/// the activation once per output column; on the wide FFN matmuls (large `n`)
/// that setup overhead drops the kernel well under the RAM streaming ceiling.
/// The 8x8 layout interleaves 8 weight rows so one activation superblock feeds
/// 8 dot products and the scale unpack is amortised 8-way, recovering ALU/IPC
/// headroom on the bandwidth-bound decode GEMV. Storage is CPU-side only; the
/// CUDA weight bytes are never touched.
pub mod repack_q4k;

/// Prefill-tiled Q6_K GEMM support: an 8-column weight group kept in the native
/// packed 6-bit planes, so a loaded weight decodes once and feeds several
/// activation rows. The per-column path re-reads and re-decodes every weight
/// column once per prompt row; grouping the columns and reusing each decode
/// across a tile of rows is the weight reuse that makes prefill compute-bound.
pub mod repack_q6k;

/// Q4_0 decode GEMV, row-grouped for contiguous 8-row streaming + activation
/// reuse. Q4_0 is symmetric (value = (nibble-8).d, no mins, one f16 scale per
/// 32): the repack just transposes the row-major blocks into
/// `[n/8 groups][nb blocks][8 rows]` so one activation Q8_0 load feeds 8 rows
/// back-to-back (8 INDEPENDENT fmadd accumulator chains hide the fmadd latency
/// = ILP the single-row `dot` can't get) and the 8 rows' block-`l` weights
/// stream contiguously. A/B NOTE: an interleaved `block_q4_0x8`+`madd_cols`
/// column-lane variant (the Q4_K shape) measured SLOWER (3.01 vs 3.33 tok/s
/// cool, mistral-nemo) - its per-call lane-gather (hadd/extract/insert)
/// amortises over Q4_K's 256-elem superblocks but costs 4x too much on Q4_0's
/// 32-elem blocks. Keep this simple form. Bit-identical to the per-row
/// `vec_dot_q4_0_q8_0` - verified by `repacked_q4_0_gemv_matches_vec_dot`.
/// This is Q4_0's decode path; mistral-nemo is the dense model that reaches it.
/// Software-prefetch `LINES` 64-byte cache lines at `p` into L1 (T0). The single
/// prefetch primitive shared by the repacked group-GEMV kernels below, so the
/// distance/line tuning that saturates RAM BW vs llama.cpp lives in ONE place
/// (measured: +2.0-2.4% decode on Q4_0 and Q4_K). Add a new quant's fast GEMV ->
/// it reuses this, no per-kernel prefetch divergence.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gemv_prefetch_t0<const LINES: usize>(p: *const i8) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    let mut i = 0;
    while i < LINES {
        _mm_prefetch::<{ _MM_HINT_T0 }>(p.add(i * 64));
        i += 1;
    }
}

/// Prefill-tiled Q5_K GEMM support. Q5_K is a 256-weight K-quant block laid out
/// like Q4_K - one super scale, one super min, packed 6-bit sub-scales/mins,
/// 4-bit base nibbles - plus a 1-bit high plane, so the weight is 5 bits. The
/// per-column path re-decodes and re-reads every super-block once per prompt
/// row; grouping 8 columns and reusing each decode across a tile of rows is the
/// weight reuse that makes prefill compute-bound. The planes are kept packed.
pub mod repack_q5k;

pub mod repack_q4_0;

/// Prefill-tiled MXFP4 GEMM support. MXFP4 is a 32-weight block (4-bit E2M1
/// `qs` codes + one E8M0 scale byte `e`). Structurally identical to Q4_0 - an
/// integer code fed to a pairwise int8 dot against the Q8_0 activation - except
/// the code maps through the `KVALUES_MXFP4` LUT (via an in-lane byte shuffle)
/// instead of a fixed `nibble - 8`, and the block scale is `e8m0_to_fp32_half(e)`
/// rather than an f16. The per-column path re-decodes and re-streams every weight
/// block per prompt row; grouping 8 columns and reusing each decoded block across
/// a row tile is the weight reuse that makes prefill compute-bound.
pub mod repack_mxfp4;

/// Column-interleaved MXFP4 prefill GEMM (8 output columns in the 8 SIMD lanes,
/// no per-column horizontal sum). The current `repack_mxfp4` kernel processes the
/// 8 columns of a group SERIALLY in SIMD (one `__m256` = 8 partial sums of ONE
/// column, `hsum`'d at the end), so it is FMA/maddubs-port-bound - ~1.4-2x slower
/// than llama.cpp's column-interleaved kernel, which puts 8 columns in the 8 lanes
/// (one dot op covers 8 columns). This module mirrors llama.cpp's `block_mxfp4x8`
/// (8 columns interleaved at 8-byte granularity) + `block_q8_0x4` (4 activation
/// rows interleaved) layouts and the `ggml_gemm_mxfp4_8x8_q8_0` reduction, so the
/// AVX2 kernel (added next) can be ported faithfully and validated against the
/// scalar oracle here. NOT bit-identical to `repack_mxfp4` (different float
/// reduction order) - a valid, reference-matching quantized GEMM.
pub mod repack_mxfp4_x8;

/// Column-interleaved Q8_0 prefill GEMM - the same technique as `repack_mxfp4_x8`
/// generalized to Q8_0 weights (8 output columns in the SIMD lanes, no per-column
/// hsum). Q8_0 dominates several fleet models' prefill (measured 72% of
/// llama3.2:1b) and the gpt-oss attention projections (BF16 loaded as Q8_0). The
/// kernel is SIMPLER than MXFP4 - the weights are already i8 (no nibble decode /
/// LUT shuffle) - so it reuses the `block_q8_0x4` activation and the same shuffle
/// dance, just loading the weight bytes directly. Layout `BlockQ8_0x8`: for each of
/// the 4 element-groups (8 elements) the 8 columns' 8 bytes are stored as
/// `[cols 0-3][cols 4-7]` (2x32 bytes), matching the kernel's blend into
/// `{0,1,4,5}`/`{2,3,6,7}`.
pub mod repack_q8_0_x8;

mod repack_inline;
pub use repack_inline::*;
mod tests_repack;
pub use tests_repack::*;
mod drivers;
pub use drivers::*;

/// The vectorised dot products. Every format has one - a format left on the
/// scalar path is a cliff nothing in the logs would explain.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
pub(crate) mod avx;
mod cpu;
mod cpu_parity;
mod dot_throughput;
mod float;
/// Marlin-tiled Q4_K repack, beside the x8-interleaved one it is an alternative to.
/// Foundation for the INT4 tensor-core GEMM; no call site yet.
pub mod repack_q4k_marlin;
#[cfg(test)]
mod repack_q4k_test;
#[cfg(test)]
mod repack_q6k_test;
