// The input stage of an attention layer's decoded token as one launch.
//
// For the projected rows `q` (n_head * hd), `k` and `v` (n_kv * hd each), F16 or F32:
//
//   q[h, :]  = rope(rms_norm(q[h, :]) * qw)     for each query head
//   k[h, :]  = rope(rms_norm(k[h, :]) * kw)     for each key head
//   v[h, :]  = v[h, :]
//
// all written as F32. A head without norm weights is rotated as it is. The rotation takes the
// first `rope_dim` lanes of a head, pairing lane `i` with `i + rope_dim / 2` (halves) or lane
// `2 i` with `2 i + 1` (interleaved), by the `cos` and `sin` of the token's position
// (`rope_dim / 2` values each); the lanes after `rope_dim` pass through. One block per head.

#include <cuda_fp16.h>
#include <stddef.h>

namespace {

__device__ __forceinline__ float load_in(const void * p, size_t i, int f16) {
    return f16 ? __half2float(((const __half *) p)[i]) : ((const float *) p)[i];
}

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

} // namespace

extern "C" __global__ void loken_qk_prep_kernel(
        const void * __restrict__ q, const void * __restrict__ k, const void * __restrict__ v,
        const float * __restrict__ qw, const float * __restrict__ kw,
        const float * __restrict__ cs, const float * __restrict__ sn,
        float * __restrict__ qo, float * __restrict__ ko, float * __restrict__ vo,
        int n_head, int n_kv, int hd, int rope_dim, int interleaved, float eps, int in_f16) {
    extern __shared__ float row[];
    const int b = blockIdx.x;
    const void * src;
    const float * w = nullptr;
    float * dst;
    bool rotate = true;
    if (b < n_head) {
        src = q;
        w = qw;
        dst = qo + (size_t) b * hd;
    } else if (b < n_head + n_kv) {
        src = k;
        w = kw;
        dst = ko + (size_t) (b - n_head) * hd;
    } else {
        src = v;
        dst = vo + (size_t) (b - n_head - n_kv) * hd;
        rotate = false;
    }
    const size_t base = b < n_head ? (size_t) b * hd
                                   : b < n_head + n_kv ? (size_t) (b - n_head) * hd
                                                       : (size_t) (b - n_head - n_kv) * hd;
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        const float x = load_in(src, base + d, in_f16);
        row[d] = x;
        ss += x * x;
    }
    float inv = 1.f;
    if (w != nullptr) inv = rsqrtf(block_sum(ss) / (float) hd + eps);
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        if (w != nullptr) row[d] = row[d] * inv * w[d];
    }
    __syncthreads();
    const int half = rope_dim / 2;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float out = row[d];
        if (rotate && d < rope_dim) {
            if (interleaved) {
                const int i = d >> 1;
                const float a = row[2 * i], c = row[2 * i + 1];
                out = (d & 1) ? c * cs[i] + a * sn[i] : a * cs[i] - c * sn[i];
            } else if (d < half) {
                out = row[d] * cs[d] - row[d + half] * sn[d];
            } else {
                const int i = d - half;
                out = row[d] * cs[i] + row[i] * sn[i];
            }
        }
        dst[d] = out;
    }
}

extern "C" void loken_qk_prep(
        const void * q, const void * k, const void * v, const float * qw, const float * kw,
        const float * cs, const float * sn, float * qo, float * ko, float * vo, int n_head,
        int n_kv, int hd, int rope_dim, int interleaved, float eps, int in_f16,
        cudaStream_t stream) {
    const int threads = hd < 256 ? ((hd + 31) / 32) * 32 : 256;
    loken_qk_prep_kernel<<<n_head + 2 * n_kv, threads, hd * sizeof(float), stream>>>(
        q, k, v, qw, kw, cs, sn, qo, ko, vo, n_head, n_kv, hd, rope_dim, interleaved, eps, in_f16);
}
