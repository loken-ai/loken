// Mat-vec against quantised weights: one warp per output row, the activation quantised to
// q8_1 so each product is an integer dot with two scales applied after it.
//
// The block layouts, the format constants and the per-format dot-product arithmetic are not
// here. `cuda/moe/gguf.cuh` states them and the loader concatenates it ahead of this file, so
// they are already in scope; restating any of them would put two versions of the same maths in
// one translation unit with nothing to keep them agreeing.
//
// What is here is what a mat-vec needs on top: wrappers that reach a block by INDEX into the
// weight array - the header's take a block that has already been resolved, and applying the
// index is exactly what a walk over a row of blocks adds - the reduction that brings the
// warps' partials together, and the entry points the launcher fetches by name.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
// stdint provided by prepended nvrtc_compat.h

// The substrate pads each weight row to this many elements, so a q8_1 activation block is
// addressed against the padded stride and not the logical one.
#define MATRIX_ROW_PADDING 512

/// A dot product that reaches its block by index: `(weights, activation, block, lane)`.
/// The header's `mmvq_dot_t` is the same idea with the block already resolved.
typedef float (*mmvq_dot_t)(const void *__restrict__ vx,
                            const block_q8_1 *__restrict__ y,
                            const int &block, const int &lane);

/// Reach a block by index, then let the header's dot product do the work.
///
/// Each format's dot product is stated once, taking a block pointer; a mat-vec walks an array
/// of blocks, so the index has to be applied first. Applying it is the whole of what these ten
/// add, and none of them restates any of the arithmetic they call into.
#define MMVQ_INDEXED(name, block, dot)                                            \
    static __device__ __forceinline__ float name(                                 \
        const void *__restrict__ vx, const block_q8_1 *__restrict__ y,            \
        const int &kbx, const int &iqs) {                                         \
        return dot((const block *) vx + kbx, y, iqs);                             \
    }

MMVQ_INDEXED(mmvq_dot_q4_0, block_q4_0, vec_dot_q4_0_q8_1)
MMVQ_INDEXED(mmvq_dot_q4_1, block_q4_1, vec_dot_q4_1_q8_1)
MMVQ_INDEXED(mmvq_dot_q5_0, block_q5_0, vec_dot_q5_0_q8_1)
MMVQ_INDEXED(mmvq_dot_q5_1, block_q5_1, vec_dot_q5_1_q8_1)
MMVQ_INDEXED(mmvq_dot_q8_0, block_q8_0, vec_dot_q8_0_q8_1)
MMVQ_INDEXED(mmvq_dot_q2_K, block_q2_K, vec_dot_q2_K_q8_1)
MMVQ_INDEXED(mmvq_dot_q3_K, block_q3_K, vec_dot_q3_K_q8_1)
MMVQ_INDEXED(mmvq_dot_q4_K, block_q4_K, vec_dot_q4_K_q8_1)
MMVQ_INDEXED(mmvq_dot_q5_K, block_q5_K, vec_dot_q5_K_q8_1)
MMVQ_INDEXED(mmvq_dot_q6_K, block_q6_K, vec_dot_q6_K_q8_1)

#undef MMVQ_INDEXED

static constexpr __device__ int mmvq_nwarps_for(int ncols_dst) {
  return (ncols_dst <= 4) ? 4 : 2;
}

static constexpr __device__ int mmvq_rows_per_cuda_block_for(int ncols_dst) {
  return (ncols_dst == 1) ? 1 : 2;
}

// small_k variant. When ncols_x is small enough that
// threads would otherwise be idle in the kbx loop, increase the number of
// output rows per block from 1 to nwarps (=4 for ncols_dst <= 4). Each warp
// contributes to a different row, keeping all threads busy.
//
// Caller MUST ensure nrows_x is divisible by `mmvq_rows_per_cuda_block_for_smallk`
// to avoid OOB weight reads - the kernel does not bounds-check in the inner loop, which is
// what buys the extra rows their throughput.
static constexpr __device__ int mmvq_rows_per_cuda_block_for_smallk(int ncols_dst) {
  return (ncols_dst == 1) ? mmvq_nwarps_for(ncols_dst) : 2;
}

/// One mat-vec: `dst[j, row] = Σ_k weight[row, k] . activation[j, k]`, over quantised blocks.
///
/// The work is split three ways and each split has a reason:
///
/// - A CUDA block owns `rows_per_cuda_block` output rows and all `ncols_dst` columns. The
///   weight is what streams from memory, so several columns share one pass over it - that is
///   the whole reason a mat-VEC kernel takes a batch dimension at all.
/// - Inside it, `nwarps` warps walk the row's blocks in strides of `blocks_per_iter`, so
///   consecutive lanes read consecutive blocks and the loads coalesce.
/// - A lane covers `vdr` of a block's `qi` ints, which is why the block index advances by
///   `tid / (qi / vdr)` and the lane's offset within it is `vdr * (tid % (qi / vdr))`.
///
/// Then the partial sums have to meet. Warps beyond the first write theirs to shared memory
/// and leave; warp zero adds them in and finishes with a butterfly across its own lanes. The
/// alternative - atomics into `dst` - would serialise on the same address `nwarps` times.
/// Which activation the gate goes through before it scales the up projection.
enum class GateAct { Silu, Gelu };

/// One pass over the activation, accumulating `nmat` weight matrices against it.
///
/// A plain mat-vec runs one matrix. A SwiGLU feed-forward runs two of the same shape and
/// combines them, and running them together is the whole point: both dot products read the
/// SAME q8_1 block, so the input stays in L1 and only the weights move. Everything between
/// the dot product and the store - the partials warps 1.. hand to warp 0, the butterfly that
/// finishes the sum, the guarded write - does not care how many matrices there were, so it
/// is written once and `combine` decides what reaches memory.
///
/// `small_k` packs four rows per block instead of one or two; the caller guarantees that
/// `nrows_x` divides by that, so the inner loop needs no bound check.
template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          mmvq_dot_t dot_block, int ncols_dst, int nmat, bool small_k, typename combine_t>
static __device__ void mmvq_core_impl(
    const void *const __restrict__ vx[nmat],
    const block_q8_1 *__restrict__ y,
    dst_t *__restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int stride_col_y, const int stride_col_dst,
    combine_t combine) {

  constexpr int nwarps = mmvq_nwarps_for(ncols_dst);
  constexpr int rows_per_cuda_block = small_k
    ? mmvq_rows_per_cuda_block_for_smallk(ncols_dst)
    : mmvq_rows_per_cuda_block_for(ncols_dst);

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float acc[nmat][ncols_dst][rows_per_cuda_block] = {};

  for (int block = tid / (qi / vdr); block < blocks_per_row; block += blocks_per_iter) {
    // The activation is q8_1 whatever the weight is, so its block index is the weight's
    // scaled by how many q8_1 blocks cover one weight block.
    const int act_block = block * (qk / QK8_1);
    const int lane_in_block = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const block_q8_1 *y_block = &y[j * stride_col_y + act_block];
        const int weight_block = (row0 + i) * blocks_per_row + block;
#pragma unroll
        for (int m = 0; m < nmat; ++m) {
          acc[m][j][i] += dot_block(vx[m], y_block, weight_block, lane_in_block);
        }
      }
    }
  }

  // Warps 1.. hand their partials to warp 0 and stop.
  __shared__ float partial[nmat][nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                          [rows_per_cuda_block][WARP_SIZE];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int m = 0; m < nmat; ++m) {
#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
          partial[m][threadIdx.y - 1][j][i][threadIdx.x] = acc[m][j][i];
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int m = 0; m < nmat; ++m) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
        for (int w = 0; w < nwarps - 1; ++w) {
          acc[m][j][i] += partial[m][w][j][i][threadIdx.x];
        }
        acc[m][j][i] = warp_sum(acc[m][j][i]);
      }
    }
    // After the butterfly every lane holds the total for row `i`, so lane `i` is the one
    // that writes it - which is also why only the first `rows_per_cuda_block` lanes store.
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      float mats[nmat];
#pragma unroll
      for (int m = 0; m < nmat; ++m) {
        mats[m] = acc[m][j][threadIdx.x];
      }
      dst[j * stride_col_dst + row0 + threadIdx.x] = (dst_t)combine(mats);
    }
  }
}

/// The plain mat-vec: one matrix, and the sum IS the result.
template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          mmvq_dot_t dot_block, int ncols_dst, bool small_k = false>
static __device__ void mmvq_core_plain(
    const void *__restrict__ vx,
    const block_q8_1 *__restrict__ y,
    dst_t *__restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int stride_col_y, const int stride_col_dst) {
  const void *const mats[1] = {vx};
  mmvq_core_impl<dst_t, qk, qi, block_q_t, vdr, dot_block, ncols_dst, 1, small_k>(
      mats, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst,
      [] (const float * a) { return a[0]; });
}

/// Gate and up in one pass, the gate through its activation.
template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          mmvq_dot_t dot_block, int ncols_dst, GateAct act>
static __device__ void mmvq_core_fused_impl(
    const void *__restrict__ vx,       // up_proj weights
    const void *__restrict__ vgate,    // gate_proj weights, same shape
    const block_q8_1 *__restrict__ y,
    dst_t *__restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int stride_col_y, const int stride_col_dst) {
  const void *const mats[2] = {vx, vgate};
  mmvq_core_impl<dst_t, qk, qi, block_q_t, vdr, dot_block, ncols_dst, 2, false>(
      mats, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst,
      [] (const float * a) {
        const float up = a[0], g = a[1];
        if constexpr (act == GateAct::Silu) {
          return up * (g / (1.0f + __expf(-g)));
        } else {
          // The tanh approximation, which is what these checkpoints were trained against  - 
          // the error-function form differs by a few thousandths and the model can tell.
          constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
          constexpr float kBeta = 0.044715f;
          return up * (0.5f * g * (1.0f + tanhf(kAlpha * (g + kBeta * g * g * g))));
        }
      });
}

// ---------------------------------------------------------------------------
// Extern-C kernel entry points
//
// Macro expands `MMVQ_PLAIN_ENTRY(tag, block_q_t, qk, qi, vdr, vec_dot,
// dst_tag, dst_c_type, ncols)` into one `__global__` function for batch
// size `ncols`. The Rust launcher switches on batch size 1..=8.
// ---------------------------------------------------------------------------

#define MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,     \
                          dst_tag, dst_c_type, ncols)                          \
  extern "C" __global__ void                                                   \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda##ncols(                         \
          const void *__restrict__ vx, const void *__restrict__ vy,            \
          dst_c_type *__restrict__ dst, const int ncols_x, const int nrows_x,  \
          const int stride_col_y, const int stride_col_dst) {                  \
    mmvq_core_plain<dst_c_type, qk_val, qi_val, block_q_t, vdr_val, vec_dot,   \
                    ncols>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,  \
                            stride_col_y, stride_col_dst);                     \
  }

/// Every weight format the mat-vec serves: its block, the geometry of that block, and its dot
/// product. Written once and handed to whichever family of entry points wants it, so a format
/// joins every family at once or none of them.
#define MMVQ_FORMATS(X)                                                        \
  X(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, mmvq_dot_q4_0)         \
  X(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, mmvq_dot_q4_1)         \
  X(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, mmvq_dot_q5_0)         \
  X(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, mmvq_dot_q5_1)         \
  X(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, mmvq_dot_q8_0)         \
  X(q2_k, block_q2_K, QK_K,  QI2_K, VDR_Q2_K_Q8_1_MMVQ, mmvq_dot_q2_K)         \
  X(q3_k, block_q3_K, QK_K,  QI3_K, VDR_Q3_K_Q8_1_MMVQ, mmvq_dot_q3_K)         \
  X(q4_k, block_q4_K, QK_K,  QI4_K, VDR_Q4_K_Q8_1_MMVQ, mmvq_dot_q4_K)         \
  X(q5_k, block_q5_K, QK_K,  QI5_K, VDR_Q5_K_Q8_1_MMVQ, mmvq_dot_q5_K)         \
  X(q6_k, block_q6_K, QK_K,  QI6_K, VDR_Q6_K_Q8_1_MMVQ, mmvq_dot_q6_K)
/// The eight batch sizes. The row count is a template parameter, so each one is its own
/// kernel; the Rust launcher switches on it.
#define MMVQ_PLAIN_BATCHES(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           dst_tag, dst_c_type)                                \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 1)                                              \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 2)                                              \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 3)                                              \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 4)                                              \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 5)                                              \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 6)                                              \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 7)                                              \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag,  \
                   dst_c_type, 8)

/// The three carriers the activation can arrive in, each over every batch size.
#define MMVQ_PLAIN_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot) \
  MMVQ_PLAIN_BATCHES(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,   \
                     __nv_bfloat16)                                            \
  MMVQ_PLAIN_BATCHES(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,    \
                     half)                                                     \
  MMVQ_PLAIN_BATCHES(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,    \
                     float)

MMVQ_FORMATS(MMVQ_PLAIN_BATCH_SET)

#define MMVQ_PLAIN_SMALLK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val,        \
                                  vec_dot, dst_tag, dst_c_type)                 \
  extern "C" __global__ void                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_smallk_cuda1(                         \
          const void *__restrict__ vx, const void *__restrict__ vy,             \
          dst_c_type *__restrict__ dst, const int ncols_x, const int nrows_x,   \
          const int stride_col_y, const int stride_col_dst) {                   \
    mmvq_core_plain<dst_c_type, qk_val, qi_val, block_q_t, vdr_val, vec_dot,    \
                    1, /*small_k=*/true>(                                       \
        vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x, stride_col_y,        \
        stride_col_dst);                                                        \
  }

#define MMVQ_PLAIN_SMALLK_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val,    \
                                      vec_dot)                                  \
  MMVQ_PLAIN_SMALLK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,     \
                            bf16, __nv_bfloat16)                                \
  MMVQ_PLAIN_SMALLK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,     \
                            f16, half)                                          \
  MMVQ_PLAIN_SMALLK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,     \
                            f32, float)

MMVQ_FORMATS(MMVQ_PLAIN_SMALLK_BATCH_SET)

// ---------------------------------------------------------------------------
// Fused MMVQ entries (Q4_K, F32 output, ncols_dst=1).
// Engine routes here when (ffn_up + ffn_gate + silu_mul) pattern detected.
// ---------------------------------------------------------------------------

extern "C" __global__ void
mmvq_gguf_q4_k_f32_fused_silu_cuda1(
    const void *__restrict__ vx,
    const void *__restrict__ vgate,
    const void *__restrict__ vy,
    float *__restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int stride_col_y, const int stride_col_dst) {
  mmvq_core_fused_impl<float, QK_K, QI4_K, block_q4_K,
                            VDR_Q4_K_Q8_1_MMVQ, mmvq_dot_q4_K, 1, GateAct::Silu>(
      vx, vgate, (const block_q8_1 *)vy, dst,
      ncols_x, nrows_x, stride_col_y, stride_col_dst);
}



extern "C" __global__ void
mmvq_gguf_q4_k_f32_fused_gelu_cuda1(
    const void *__restrict__ vx,
    const void *__restrict__ vgate,
    const void *__restrict__ vy,
    float *__restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int stride_col_y, const int stride_col_dst) {
  mmvq_core_fused_impl<float, QK_K, QI4_K, block_q4_K,
                            VDR_Q4_K_Q8_1_MMVQ, mmvq_dot_q4_K, 1, GateAct::Gelu>(
      vx, vgate, (const block_q8_1 *)vy, dst,
      ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

// BF16-output variant: same fused (up + gate + silu) compute as the F32
// kernel above, but writes BF16 directly to global memory. The template
// `dst_t` parameter handles the on-store cast inside the kernel, so the
// downstream residual broadcast_add doesn't pay a separate F32->BF16
// launch (which would otherwise eat the launch saving). Coupled with
// the BF16-INPUT quantize variant
// (launch_mmvq_gguf_quantize_q8_1_bf16) the full BF16->BF16 fused
// (norm-ish + gate + up + silu) chain stays in BF16 end-to-end.
extern "C" __global__ void
mmvq_gguf_q4_k_bf16_fused_silu_cuda1(
    const void *__restrict__ vx,
    const void *__restrict__ vgate,
    const void *__restrict__ vy,
    __nv_bfloat16 *__restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int stride_col_y, const int stride_col_dst) {
  mmvq_core_fused_impl<__nv_bfloat16, QK_K, QI4_K, block_q4_K,
                            VDR_Q4_K_Q8_1_MMVQ, mmvq_dot_q4_K, 1, GateAct::Silu>(
      vx, vgate, (const block_q8_1 *)vy, dst,
      ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

// Padding-aware BF16/F16/F32 -> Q8_1 quantisation.
//
// How a value becomes a q8_1 code, and what the block's shared pair holds, is the activation
// format's own statement and lives with it in the header. What the three entry points below
// add is the carrier each was handed.

#define MMVQ_QUANTIZE_ENTRY(suffix, src_t)                                     \
  extern "C" __global__ void mmvq_gguf_quantize_q8_1_##suffix(                 \
      const src_t *__restrict__ x, void *__restrict__ vy, const int kx,        \
      const int kx_padded) {                                                   \
    quantize_row_to_q8_1<src_t>(x, vy, kx, kx_padded);                         \
  }

MMVQ_QUANTIZE_ENTRY(bf16, __nv_bfloat16)
MMVQ_QUANTIZE_ENTRY(f16, half)
MMVQ_QUANTIZE_ENTRY(f32, float)

// Host-side launchers, called through the kernel FFI boundary.


// Host launcher functions (extern "C" void launch_mmvq_*) were stripped:
// NVRTC cannot compile host code (<<<>>> launches). The Rust launchers in
// quantized_cuda.rs drive the __global__ kernels above by name instead.
