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
// The weights are dense F32 or F16, or int8 with a scale for each group of 64 values of a row,
// row-major `[n, k]` (the scales `[n, k / 64]`); the activations F32. `half_w` is 0 for F32,
// 1 for F16 and 2 for int8.

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

// The same dot product over a row of F16 weights, eight at a time.
__device__ __forceinline__ float dot_row_h(const __half * __restrict__ w, const float * __restrict__ x,
        int k, int lane0, int stride) {
    float acc = 0.f;
    if ((k & 7) == 0) {
        const uint4 * w8 = reinterpret_cast<const uint4 *>(w);
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        for (int i = lane0; i < (k >> 3); i += stride) {
            const uint4 u = w8[i];
            const __half2 * h = reinterpret_cast<const __half2 *>(&u);
            const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
            const float2 c = __half22float2(h[2]), d = __half22float2(h[3]);
            const float4 x0 = x4[2 * i], x1 = x4[2 * i + 1];
            acc += a.x * x0.x + a.y * x0.y + b.x * x0.z + b.y * x0.w;
            acc += c.x * x1.x + c.y * x1.y + d.x * x1.z + d.y * x1.w;
        }
    } else {
        for (int i = lane0; i < k; i += stride) acc += __half2float(w[i]) * x[i];
    }
    return acc;
}

// The same over a row of int8 weights, sixteen at a time, each group of 64 scaled by its entry of
// the row's `scale`.
__device__ __forceinline__ float dot_row_q8(const signed char * __restrict__ w,
        const float * __restrict__ scale, const float * __restrict__ x, int k, int lane0,
        int stride) {
    float acc = 0.f;
    if ((k & 15) == 0) {
        const int4 * w16 = reinterpret_cast<const int4 *>(w);
        const float4 * x4 = reinterpret_cast<const float4 *>(x);
        for (int i = lane0; i < (k >> 4); i += stride) {
            const int4 u = w16[i];
            const signed char * b = reinterpret_cast<const signed char *>(&u);
            const float4 x0 = x4[4 * i], x1 = x4[4 * i + 1], x2 = x4[4 * i + 2], x3 = x4[4 * i + 3];
            float part = (float) b[0] * x0.x + (float) b[1] * x0.y + (float) b[2] * x0.z + (float) b[3] * x0.w;
            part += (float) b[4] * x1.x + (float) b[5] * x1.y + (float) b[6] * x1.z + (float) b[7] * x1.w;
            part += (float) b[8] * x2.x + (float) b[9] * x2.y + (float) b[10] * x2.z + (float) b[11] * x2.w;
            part += (float) b[12] * x3.x + (float) b[13] * x3.y + (float) b[14] * x3.z + (float) b[15] * x3.w;
            acc += part * scale[i >> 2];
        }
    } else {
        for (int i = lane0; i < k; i += stride) acc += (float) w[i] * x[i] * scale[i >> 6];
    }
    return acc;
}

} // namespace

extern "C" __global__ void loken_hc_down_inject_kernel(
        const float * __restrict__ xn, const void * __restrict__ down,
        const void * __restrict__ inject, float * __restrict__ low, float * __restrict__ inj,
        int K, int R, int I, float inv_streams, int half_w,
        const float * __restrict__ down_scale, const float * __restrict__ inject_scale) {
    const int o = blockIdx.x;
    const size_t row = blockIdx.y;
    float acc;
    if (half_w == 2) {
        const signed char * w = (o < R) ? (const signed char *) down + (size_t) o * K
                                        : (const signed char *) inject + (size_t) (o - R) * K;
        const float * scale = (o < R) ? down_scale + (size_t) o * (K >> 6)
                                      : inject_scale + (size_t) (o - R) * (K >> 6);
        acc = dot_row_q8(w, scale, xn + row * K, K, threadIdx.x, blockDim.x);
    } else if (half_w) {
        const __half * w = (o < R) ? (const __half *) down + (size_t) o * K
                                   : (const __half *) inject + (size_t) (o - R) * K;
        acc = dot_row_h(w, xn + row * K, K, threadIdx.x, blockDim.x);
    } else {
        const float * w = (o < R) ? (const float *) down + (size_t) o * K
                                  : (const float *) inject + (size_t) (o - R) * K;
        acc = dot_row(w, xn + row * K, K, threadIdx.x, blockDim.x);
    }
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
        const float * xn, const void * down, const void * inject, float * low, float * inj,
        int rows, int K, int R, int I, int S, int half_w, const float * down_scale,
        const float * inject_scale, cudaStream_t stream) {
    dim3 grid((unsigned) (R + I), (unsigned) rows);
    loken_hc_down_inject_kernel<<<grid, 256, 0, stream>>>(
        xn, down, inject, low, inj, K, R, I, 1.f / (float) S, half_w, down_scale, inject_scale);
}

extern "C" __global__ void loken_hc_up_mean_kernel(
        const float * __restrict__ xn, const void * __restrict__ up,
        const float * __restrict__ low, void * __restrict__ mixed,
        int S, int D, int R, int out_f16, int half_w, const float * __restrict__ up_scale) {
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
        const size_t at = ((size_t) s * D + d0 + dd) * R;
        float acc = half_w == 2 ? dot_row_q8((const signed char *) up + at, up_scale + ((size_t) s * D + d0 + dd) * (R >> 6), low_s, R, lane, 32)
                  : half_w ? dot_row_h((const __half *) up + at, low_s, R, lane, 32)
                           : dot_row((const float *) up + at, low_s, R, lane, 32);
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
        const float * xn, const void * up, const float * low, void * mixed, int rows, int S,
        int D, int R, int out_f16, int half_w, const float * up_scale, cudaStream_t stream) {
    dim3 grid((unsigned) ((D + UP_TILE - 1) / UP_TILE), (unsigned) rows);
    const size_t shared = ((size_t) R + (size_t) S * UP_TILE) * sizeof(float);
    loken_hc_up_mean_kernel<<<grid, 256, shared, stream>>>(xn, up, low, mixed, S, D, R, out_f16, half_w, up_scale);
}

// Int8 weights with a scale for each group of 64 values, widened to F16 or F32 for the stages that
// multiply them as a matrix.
extern "C" __global__ void loken_hc_dequant_kernel(
        const signed char * __restrict__ q, const float * __restrict__ scale,
        void * __restrict__ out, int n, int k, int out_f32) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * k) return;
    const size_t row = i / k, col = i % k;
    const float v = (float) q[i] * scale[row * (k >> 6) + (col >> 6)];
    if (out_f32) ((float *) out)[i] = v; else ((__half *) out)[i] = __float2half(v);
}

extern "C" void loken_hc_dequant(
        const signed char * q, const float * scale, void * out, int n, int k, int out_f32,
        cudaStream_t stream) {
    const size_t total = (size_t) n * k;
    loken_hc_dequant_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, stream>>>(
        q, scale, out, n, k, out_f32);
}
