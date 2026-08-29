/**
 * @brief Fused RMS norm + Q4_K matmul (single-token decode).
 *
 * Replaces three-launch sequence
 *   x_normed = rms_norm(x, w_norm)        // 1 launch
 *   x_q8_1   = quantize_q8_1(x_normed)    // 1 launch
 *   y        = dequantize_mul_mat(W_q4k, x_q8_1)  // 1 launch
 * with a single CUDA launch for the typical attn_norm + wqkv (or
 * ffn_norm + wgate-router) flow at decode time.
 *
 * Per-block layout:
 *   blockIdx.x = output row index (one block per row of W^T).
 *   blockDim.x = WARP_SIZE (32) per warp, blockDim.y = nwarps (default 4).
 *
 * Each block recomputes x_normed and its q8_1 representation in shared
 * memory, then runs the standard Q4_K_Q8_1 dot product to produce the
 * row's output element. The norm cost is amortized across the matmul's
 * existing global-memory traffic - net win is the elimination of the
 * two upstream launches and the round-trip through global F32
 * x_normed / q8_1 buffers.
 *
 * The dequantize input path (F32 -> Q8_1 in shared) follows
 * `quantize_q8_1` from the upstream quantized.cu but operates on the
 * post-norm data we just produced in shared, so we don't need a
 * separate quantize kernel.
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace loken_ll_rmm {

// Block-wide reduction across `nthreads` threads.
__device__ __forceinline__ float block_reduce_sum(float v, int nthreads, float* tmp) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
    }
    const int lane = threadIdx.x & 31;
    const int wid  = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    if (lane == 0) tmp[wid] = v;
    __syncthreads();
    if (wid == 0) {
        const int n_warps = (nthreads + WARP_SIZE - 1) / WARP_SIZE;
        v = (lane < n_warps) ? tmp[lane] : 0.f;
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
        }
        if (lane == 0) tmp[0] = v;
    }
    __syncthreads();
    return tmp[0];
}

// Quantize one F32 vector of length `hidden` (must be divisible by QK8_1=32)
// to block_q8_1 in shared memory. Each warp handles one or more blocks.
// `dst` length = hidden / QK8_1 blocks.
__device__ __forceinline__ void quantize_to_q8_1_shared(
    const float* __restrict__ src,
    block_q8_1*  __restrict__ dst,
    int hidden,
    int tid,                     // global thread index in block
    int nthreads
) {
    const int n_blocks = hidden / QK8_1;
    for (int b = tid / WARP_SIZE; b < n_blocks; b += nthreads / WARP_SIZE) {
        const int lane = tid & 31;
        // Each warp does one block of 32 elements
        float v = src[b * QK8_1 + lane];
        // Find absmax + sum across the warp. (`sum` here is the sum of
        // ORIGINAL float values - that's what the vec_dot_q*_K_q8_1
        // expects in `ds.y`. Earlier mistake: storing `sum_int8 * d`
        // produced gibberish output.)
        float amax = fabsf(v);
        float sum = v;
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, mask, WARP_SIZE));
            sum  = sum + __shfl_xor_sync(0xFFFFFFFFu, sum,  mask, WARP_SIZE);
        }
        const float d = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)__float2int_rn(v / d);
        if (lane == 0) {
            reinterpret_cast<__half&>(dst[b].ds.x) = __float2half(d);
            reinterpret_cast<__half&>(dst[b].ds.y) = __float2half(sum);
        }
        dst[b].qs[lane] = q;
    }
}

} // namespace loken_ll_rmm

// --- Fused RMS norm + Q8_1 quantize (NO matmul) -------------------------
// Replaces 2-launch sequence:
//   x_normed = rms_norm(x, w_norm)        // 1 launch
//   x_q8_1   = quantize_q8_1(x_normed)    // 1 launch
// with ONE launch that produces the q8_1 buffer directly. Caller then
// dispatches a standard mvq kernel against the q8_1 output. Avoids the
// per-block redundancy of the all-fused rms_qmatmul (which recomputes
// norm+quantize on every output row's block).
//
// Single-block kernel: hidden up to 16384 fits in shared mem.
namespace loken_ll_rmm {

template <int block_size>
__global__ void rms_quantize_q8_1_kernel(
    const float* __restrict__ x,         // [hidden] F32
    const float* __restrict__ w_norm,    // [hidden] F32
    block_q8_1* __restrict__ y,          // [hidden / QK8_1] q8_1
    int hidden,
    float rms_eps
) {
    extern __shared__ unsigned char s_raw[];
    float* x_normed = reinterpret_cast<float*>(s_raw);
    float* warp_tmp = reinterpret_cast<float*>(x_normed + hidden);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthr = blockDim.x * blockDim.y;

    // Pass 1: load x, compute sum-of-squares.
    float local_sumsq = 0.0f;
    for (int i = tid; i < hidden; i += nthr) {
        const float v = x[i];
        x_normed[i] = v;
        local_sumsq += v * v;
    }
    __syncthreads();
    const float sumsq = block_reduce_sum(local_sumsq, nthr, warp_tmp);
    const float inv_rms = rsqrtf(sumsq / (float)hidden + rms_eps);

    // Pass 2: scale by inv_rms * w_norm.
    for (int i = tid; i < hidden; i += nthr) {
        x_normed[i] = x_normed[i] * inv_rms * w_norm[i];
    }
    __syncthreads();

    // Pass 3: quantize to q8_1 (one warp per block, write directly to global y).
    quantize_to_q8_1_shared(x_normed, y, hidden, tid, nthr);
}

} // namespace loken_ll_rmm

extern "C" void loken_rms_quantize_q8_1(
    const float* x,
    const float* w_norm,
    void* y_q8_1,
    int hidden,
    float rms_eps,
    cudaStream_t stream
) {
    constexpr int block_size = 256;
    const size_t smem_x = (size_t)hidden * sizeof(float);
    const size_t smem_t = (size_t)(block_size / WARP_SIZE) * sizeof(float);
    const size_t smem = smem_x + smem_t + 64;
    dim3 grid(1, 1, 1);
    dim3 block(WARP_SIZE, block_size / WARP_SIZE, 1);
    loken_ll_rmm::rms_quantize_q8_1_kernel<block_size><<<grid, block, smem, stream>>>(
        x, w_norm, (block_q8_1*)y_q8_1, hidden, rms_eps
    );
}

// LayerNorm + bias + quantize_q8_1 in one launch.
// phi2 uses LayerNorm (mean + variance), not RMSNorm. This kernel fuses
// the layer norm, bias add, and Q8_1 quantization so the downstream
// MMVQ can consume a pre-quantized input via mvq_via_pre_quantized_q8_1.
// Saves: 1 layernorm launch + 1 bias add launch + 1 quantize launch -> 1 fused launch.
namespace loken_ll_rmm {

} // namespace loken_ll_rmm

// BF16-input variant. Same algorithm - only the x-load is BF16->F32.
// w_norm stays F32 (norm scales are tiny; quantizing them would not
// save meaningful bandwidth). Output stays Q8_1. Used by callers
// whose hidden state is BF16 (most LLM models) so the fusion saves
// the explicit pre-cast launch.
namespace loken_ll_rmm {
template <int block_size>
__global__ void rms_quantize_q8_1_kernel_bf16(
    const __nv_bfloat16* __restrict__ x, // [hidden] BF16
    const float* __restrict__ w_norm,    // [hidden] F32
    block_q8_1* __restrict__ y,          // [hidden / QK8_1] q8_1
    int hidden,
    float rms_eps
) {
    extern __shared__ unsigned char s_raw[];
    float* x_normed = reinterpret_cast<float*>(s_raw);
    float* warp_tmp = reinterpret_cast<float*>(x_normed + hidden);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthr = blockDim.x * blockDim.y;

    // Pass 1: load x (BF16->F32 on load), compute sum-of-squares in F32.
    float local_sumsq = 0.0f;
    for (int i = tid; i < hidden; i += nthr) {
        const float v = __bfloat162float(x[i]);
        x_normed[i] = v;
        local_sumsq += v * v;
    }
    __syncthreads();
    const float sumsq = block_reduce_sum(local_sumsq, nthr, warp_tmp);
    const float inv_rms = rsqrtf(sumsq / (float)hidden + rms_eps);

    for (int i = tid; i < hidden; i += nthr) {
        x_normed[i] = x_normed[i] * inv_rms * w_norm[i];
    }
    __syncthreads();

    quantize_to_q8_1_shared(x_normed, y, hidden, tid, nthr);
}
} // namespace loken_ll_rmm

extern "C" void loken_rms_quantize_q8_1_bf16(
    const void* x,           // const __nv_bfloat16*
    const float* w_norm,
    void* y_q8_1,
    int hidden,
    float rms_eps,
    cudaStream_t stream
) {
    constexpr int block_size = 256;
    const size_t smem_x = (size_t)hidden * sizeof(float);
    const size_t smem_t = (size_t)(block_size / WARP_SIZE) * sizeof(float);
    const size_t smem = smem_x + smem_t + 64;
    dim3 grid(1, 1, 1);
    dim3 block(WARP_SIZE, block_size / WARP_SIZE, 1);
    loken_ll_rmm::rms_quantize_q8_1_kernel_bf16<block_size><<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)x, w_norm, (block_q8_1*)y_q8_1, hidden, rms_eps
    );
}

