/**
 * @brief Token-sampling kernels for the continuous-batching decode loop.
 *
 * Greedy argmax, top-k denominator, and the repeat penalty - each over the whole batch in one
 * launch, so the logits are read once at full bandwidth instead of once per row.
 *
 * These lived beside the MoE routing kernel because both did a top-k, and inherited that
 * file's "adapted from llama.cpp" header along with it. They are neither: routing picks
 * experts for a token, sampling picks a token. Ours, and now filed as such.
 */
#include "cuda_fp16.h"
#include <cstdint>
#include <cmath>

#define WARP_SIZE 32

// -- Batched argmax over [n_rows, vocab] -> n_rows ids (one block per row) --
// A kernel of our own rather than a tensor-level argmax: the continuous-batching decode sampler picks the
// greedy token for all B rows in ONE launch, reading logits once at full BW. Replaces the
// an argmax plus a per-row host scan (profiled ~2ms/step at B=8). Ties break to the LOWEST
// index for bit-parity with the host/serial greedy sampler.
template<typename T>
__global__ void loken_batched_argmax_kernel(const T* __restrict__ logits,
                                                unsigned int* __restrict__ out, int vocab) {
    int row = blockIdx.x;
    const T* r = logits + (size_t)row * vocab;
    float lmax = -1e30f; int lidx = 0;
    for (int k = threadIdx.x; k < vocab; k += blockDim.x) {
        float v = (float)r[k];
        if (v > lmax) { lmax = v; lidx = k; }
    }
    __shared__ float smax[256];
    __shared__ int   sidx[256];
    smax[threadIdx.x] = lmax; sidx[threadIdx.x] = lidx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = smax[threadIdx.x + s]; int oi = sidx[threadIdx.x + s];
            if (ov > smax[threadIdx.x] || (ov == smax[threadIdx.x] && oi < sidx[threadIdx.x])) {
                smax[threadIdx.x] = ov; sidx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = (unsigned int)sidx[0];
}
extern "C" void loken_batched_argmax_f16(const void* logits, unsigned int* out,
                                             int n_rows, int vocab, cudaStream_t stream) {
    loken_batched_argmax_kernel<__half><<<n_rows, 256, 0, stream>>>((const __half*)logits, out, vocab);
}
extern "C" void loken_batched_argmax_f32(const void* logits, unsigned int* out,
                                             int n_rows, int vocab, cudaStream_t stream) {
    loken_batched_argmax_kernel<float><<<n_rows, 256, 0, stream>>>((const float*)logits, out, vocab);
}

// -- Batched top-k + softmax-denominator over [n_rows, vocab] (one block per row) --
// A fused sampler front-end of our own, rather than a full-vocab softmax plus a host select: per row
// compute rowmax + denom = Σ exp((logit-rowmax)/T), then extract the top-k logits+indices
// by iterated block-argmax with in-place masking (safe: the logits buffer is overwritten by
// the next graph replay). The host then finishes top-p + multinomial over just k entries with
// FULL-VOCAB-normalized probs exp((l-rowmax)/T)/denom - bit-exact with the serial sampler,
// but transfers only [n_rows,k]+[n_rows,2] instead of the whole [n_rows,vocab] logits.
// out_logit/out_idx are returned in DESCENDING order (iteration 0 = argmax).
template<typename T>
__global__ void loken_batched_topk_denom_kernel(
    T* __restrict__ logits, float* __restrict__ out_logit, unsigned int* __restrict__ out_idx,
    float* __restrict__ out_stats, int vocab, int k, float inv_temp) {
    int row = blockIdx.x;
    T* r = logits + (size_t)row * vocab;
    int t = threadIdx.x;
    __shared__ float sred[256];
    __shared__ int   sidx[256];
    // 1. row max (softmax stability)
    float lmax = -1e30f;
    for (int i = t; i < vocab; i += blockDim.x) lmax = fmaxf(lmax, (float)r[i]);
    sred[t] = lmax; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (t < s) sred[t] = fmaxf(sred[t], sred[t + s]); __syncthreads(); }
    float rowmax = sred[0]; __syncthreads();
    // 2. denom = Σ exp((logit - rowmax) * inv_temp)
    float lsum = 0.f;
    for (int i = t; i < vocab; i += blockDim.x) lsum += __expf(((float)r[i] - rowmax) * inv_temp);
    sred[t] = lsum; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (t < s) sred[t] += sred[t + s]; __syncthreads(); }
    if (t == 0) { out_stats[row * 2] = rowmax; out_stats[row * 2 + 1] = sred[0]; }
    __syncthreads();
    // 3. top-k by iterated argmax + in-place mask (ties -> lowest index)
    for (int it = 0; it < k; it++) {
        float lm = -1e30f; int li = 0;
        for (int i = t; i < vocab; i += blockDim.x) { float v = (float)r[i]; if (v > lm) { lm = v; li = i; } }
        sred[t] = lm; sidx[t] = li; __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (t < s) { float ov = sred[t + s]; int oi = sidx[t + s];
                if (ov > sred[t] || (ov == sred[t] && oi < sidx[t])) { sred[t] = ov; sidx[t] = oi; } }
            __syncthreads();
        }
        if (t == 0) { out_logit[row * k + it] = sred[0]; out_idx[row * k + it] = (unsigned)sidx[0]; r[sidx[0]] = (T)(-1e30f); }
        __syncthreads();
    }
}
extern "C" void loken_batched_topk_denom_f16(void* logits, float* out_logit, unsigned int* out_idx,
        float* out_stats, int n_rows, int vocab, int k, float inv_temp, cudaStream_t stream) {
    loken_batched_topk_denom_kernel<__half><<<n_rows, 256, 0, stream>>>((__half*)logits, out_logit, out_idx, out_stats, vocab, k, inv_temp);
}
extern "C" void loken_batched_topk_denom_f32(void* logits, float* out_logit, unsigned int* out_idx,
        float* out_stats, int n_rows, int vocab, int k, float inv_temp, cudaStream_t stream) {
    loken_batched_topk_denom_kernel<float><<<n_rows, 256, 0, stream>>>((float*)logits, out_logit, out_idx, out_stats, vocab, k, inv_temp);
}

// -- Repeat-penalty scatter: divide/multiply logits at each row's recent tokens --
// Applied in-place BEFORE the top-k kernel so the GPU sampler picks over penalized logits
// (parity with the serial sampler, which penalizes raw logits before softmax/top-k).
template<typename T>
__global__ void loken_repeat_penalty_kernel(T* __restrict__ logits, const int* __restrict__ rows,
        const unsigned int* __restrict__ toks, float rp, int n, int vocab) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t off = (size_t)rows[i] * vocab + toks[i];
    float v = (float)logits[off];
    logits[off] = (T)(v > 0.f ? v / rp : v * rp);
}
extern "C" void loken_repeat_penalty_f16(void* logits, const int* rows, const unsigned int* toks,
        float rp, int n, int vocab, cudaStream_t stream) {
    if (n > 0) loken_repeat_penalty_kernel<__half><<<(n + 255) / 256, 256, 0, stream>>>((__half*)logits, rows, toks, rp, n, vocab);
}
extern "C" void loken_repeat_penalty_f32(void* logits, const int* rows, const unsigned int* toks,
        float rp, int n, int vocab, cudaStream_t stream) {
    if (n > 0) loken_repeat_penalty_kernel<float><<<(n + 255) / 256, 256, 0, stream>>>((float*)logits, rows, toks, rp, n, vocab);
}
