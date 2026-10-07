// The wide residual's elementwise work, fused. A wide residual holds S copies of the hidden state
// of width D per token, row-major [rows, S, D]; every stage below is one launch over that layout.
//
//   norm        xn[g, :] = r[g, :] * rsqrt(mean(r[g, :]^2) + eps) * gamma[g % S, :]
//               one block per (row, stream) group g
//   low         low[i] = silu(down[i] / S)
//   mean_gate   mixed[row, d] = (1/S) * sum_s xn[row, s, d] * sigmoid(up[row, s, d])
//   combine     r'[row, s, :] = r[row, s, :] + out[row, :] * 2 * sigmoid(inject[row, s] / S)
//               and, when `gamma` is given, the next mixer's norm of r'
//
// `out` and `mixed` are F32 or F16, as the sublayer reads and writes them.

#include <cuda_fp16.h>
#include <stddef.h>

namespace {

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float partial[32];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x + 31) / 32) ? partial[lane] : 0.f;
    if (warp == 0) {
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    }
    __shared__ float total;
    if (threadIdx.x == 0) total = v;
    __syncthreads();
    return total;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float load_as_f32(const void * p, size_t i, int f16) {
    return f16 ? __half2float(((const __half *) p)[i]) : ((const float *) p)[i];
}

__device__ __forceinline__ void store_from_f32(void * p, size_t i, float v, int f16) {
    if (f16) ((__half *) p)[i] = __float2half(v); else ((float *) p)[i] = v;
}

} // namespace

extern "C" __global__ void loken_hc_norm_kernel(
        const float * __restrict__ r, const float * __restrict__ gamma, float * __restrict__ xn,
        int S, int D, float eps) {
    const int g = blockIdx.x;
    const float * rr = r + (size_t) g * D;
    const float * gm = gamma + (size_t) (g % S) * D;
    float * o = xn + (size_t) g * D;
    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { const float v = rr[d]; ss += v * v; }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) D + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = rr[d] * inv * gm[d];
}

extern "C" void loken_hc_norm(
        const float * r, const float * gamma, float * xn, int groups, int S, int D, float eps,
        cudaStream_t stream) {
    loken_hc_norm_kernel<<<groups, 256, 0, stream>>>(r, gamma, xn, S, D, eps);
}

extern "C" __global__ void loken_hc_low_kernel(
        const float * __restrict__ down, float * __restrict__ low, int n, float inv_streams) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = down[i] * inv_streams;
    low[i] = x / (1.f + expf(-x));
}

extern "C" void loken_hc_low(
        const float * down, float * low, int n, int S, cudaStream_t stream) {
    loken_hc_low_kernel<<<(n + 255) / 256, 256, 0, stream>>>(down, low, n, 1.f / (float) S);
}

extern "C" __global__ void loken_hc_mean_gate_kernel(
        const float * __restrict__ xn, const float * __restrict__ up, void * __restrict__ mixed,
        int S, int D, int rows, int out_f16) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) rows * D) return;
    const size_t row = idx / D, d = idx % D;
    float acc = 0.f;
    for (int s = 0; s < S; ++s) {
        const size_t i = (row * S + s) * D + d;
        acc += xn[i] * sigmoidf_(up[i]);
    }
    store_from_f32(mixed, idx, acc / (float) S, out_f16);
}

extern "C" void loken_hc_mean_gate(
        const float * xn, const float * up, void * mixed, int rows, int S, int D, int out_f16,
        cudaStream_t stream) {
    const size_t n = (size_t) rows * D;
    loken_hc_mean_gate_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(
        xn, up, mixed, S, D, rows, out_f16);
}

extern "C" __global__ void loken_hc_combine_kernel(
        const float * __restrict__ r, const void * __restrict__ out, const float * __restrict__ inject,
        float * __restrict__ r_new, float * __restrict__ xn, const float * __restrict__ gamma,
        int S, int D, float eps, int out_f16) {
    const int g = blockIdx.x;
    const size_t row = g / S;
    const int s = g % S;
    const float w = 2.f * sigmoidf_(inject[g] / (float) S);
    const float * rr = r + (size_t) g * D;
    float * rn = r_new + (size_t) g * D;
    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = rr[d] + load_as_f32(out, row * D + d, out_f16) * w;
        rn[d] = v;
        ss += v * v;
    }
    if (gamma == nullptr) return;
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) D + eps);
    const float * gm = gamma + (size_t) s * D;
    float * o = xn + (size_t) g * D;
    // This block wrote every value it reads back, so its own stores are what it sees.
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] = rn[d] * inv * gm[d];
}

extern "C" void loken_hc_combine(
        const float * r, const void * out, const float * inject, float * r_new, float * xn,
        const float * gamma, int groups, int S, int D, float eps, int out_f16,
        cudaStream_t stream) {
    loken_hc_combine_kernel<<<groups, 256, 0, stream>>>(
        r, out, inject, r_new, xn, gamma, S, D, eps, out_f16);
}

// The mixer's two projections for a decode step, one launch each.
//
//   down_inject  low[row, i]  = silu((down[i, :] . xn[row, :]) / S)        i < R
//                inj[row, j]  = inject[j, :] . xn[row, :]                  j < I
//                one block per output row of either weight, one grid row per token
//   up_mean      mixed[row, d] = (1/S) * sum_s xn[row, s, d] * sigmoid(up[s * D + d, :] . low[row, :])
//                one block per tile of UP_TILE columns d, all S streams of the tile
//
// The weights are dense F32, row-major `[n, k]`; the activations F32.

namespace {

constexpr int UP_TILE = 8;

__device__ __forceinline__ float dot_row(const float * __restrict__ w, const float * __restrict__ x,
        int k, int lane0, int stride) {
    float acc = 0.f;
    if ((k & 3) == 0) {
        const float4 * w4 = reinterpret_cast<const float4 *>(w);
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        for (int i = lane0; i < (k >> 2); i += stride) {
            const float4 a = w4[i], b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = lane0; i < k; i += stride) acc += w[i] * x[i];
    }
    return acc;
}

} // namespace

extern "C" __global__ void loken_hc_down_inject_kernel(
        const float * __restrict__ xn, const float * __restrict__ down,
        const float * __restrict__ inject, float * __restrict__ low, float * __restrict__ inj,
        int K, int R, int I, float inv_streams) {
    const int o = blockIdx.x;
    const size_t row = blockIdx.y;
    const float * w = (o < R) ? down + (size_t) o * K : inject + (size_t) (o - R) * K;
    float acc = dot_row(w, xn + row * K, K, threadIdx.x, blockDim.x);
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        if (o < R) {
            const float x = acc * inv_streams;
            low[row * R + o] = x / (1.f + expf(-x));
        } else {
            inj[row * I + (o - R)] = acc;
        }
    }
}

extern "C" void loken_hc_down_inject(
        const float * xn, const float * down, const float * inject, float * low, float * inj,
        int rows, int K, int R, int I, int S, cudaStream_t stream) {
    dim3 grid((unsigned) (R + I), (unsigned) rows);
    loken_hc_down_inject_kernel<<<grid, 256, 0, stream>>>(
        xn, down, inject, low, inj, K, R, I, 1.f / (float) S);
}

extern "C" __global__ void loken_hc_up_mean_kernel(
        const float * __restrict__ xn, const float * __restrict__ up,
        const float * __restrict__ low, void * __restrict__ mixed,
        int S, int D, int R, int out_f16) {
    extern __shared__ float shared[];
    float * low_s = shared;                    // R
    float * gate = shared + R;                 // S * UP_TILE
    const size_t row = blockIdx.y;
    const int d0 = blockIdx.x * UP_TILE;
    const int tile = min(UP_TILE, D - d0);
    for (int i = threadIdx.x; i < R; i += blockDim.x) low_s[i] = low[row * R + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    for (int j = warp; j < S * tile; j += warps) {
        const int s = j / tile, dd = j % tile;
        const float * w = up + ((size_t) s * D + d0 + dd) * R;
        float acc = dot_row(w, low_s, R, lane, 32);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) gate[s * UP_TILE + dd] = sigmoidf_(acc);
    }
    __syncthreads();
    if (threadIdx.x < tile) {
        const int d = d0 + threadIdx.x;
        float acc = 0.f;
        for (int s = 0; s < S; ++s) acc += xn[(row * S + s) * D + d] * gate[s * UP_TILE + threadIdx.x];
        store_from_f32(mixed, row * D + d, acc / (float) S, out_f16);
    }
}

extern "C" void loken_hc_up_mean(
        const float * xn, const float * up, const float * low, void * mixed, int rows, int S,
        int D, int R, int out_f16, cudaStream_t stream) {
    dim3 grid((unsigned) ((D + UP_TILE - 1) / UP_TILE), (unsigned) rows);
    const size_t shared = ((size_t) R + (size_t) S * UP_TILE) * sizeof(float);
    loken_hc_up_mean_kernel<<<grid, 256, shared, stream>>>(xn, up, low, mixed, S, D, R, out_f16);
}
