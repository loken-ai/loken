// The gated delta-net's input stage for the decoded tokens of one launch: one row, or the rows
// of a verify one after the other, each reading the window the row before it left.
//
// For the projected row `qkv` (conv_dim = 2 * kg * kd + vg * vd channels: q, then k, then v), the
// per-channel causal convolution with SiLU, the new convolution window, the L2 norm of every
// q and k head, their copy to the value heads each feeds, and the decay and write gates:
//
//   c[ch]      = silu(sum_j w[ch, j] * padded[ch, j]),  padded = [window[ch, 0..K-2], qkv[ch]]
//   window'    = padded[1..K-1]
//   q[hv, :]   = c_q[kh, :] * rsqrt(sum(c_q[kh, :]^2) + eps)   for each value head hv of head kh
//   k[hv, :]   likewise from c_k
//   v[vh, :]   = c_v[vh, :]
//   g[vh]      = a[vh] * log(exp(alpha[vh] + dt_bias[vh]) + 1),  beta[vh] = sigmoid(beta_in[vh])
//
// `tiled` picks the head order of the copy: value head j * kg + kh (tiled) or kh * rep + j.
// One block per q/k head and per value head; every channel is read and written by one thread, so
// the window may be advanced in place.

#include <cuda_fp16.h>
#include <stddef.h>

namespace {

__device__ __forceinline__ float load_in(const void * p, int i, int f16) {
    return f16 ? __half2float(((const __half *) p)[i]) : ((const float *) p)[i];
}

__device__ __forceinline__ float conv_silu(
        int ch, const void * qkv, float * window, const float * w, float * new_window, int K,
        int f16) {
    const int Km1 = K - 1;
    float cs[8], wl[8];
    for (int j = 0; j < Km1; ++j) cs[j] = window[ch * Km1 + j];
    for (int j = 0; j < K; ++j) wl[j] = w[ch * K + j];
    const float in = load_in(qkv, ch, f16);
    float sum = 0.f;
    for (int j = 0; j < K; ++j) sum += wl[j] * ((j < Km1) ? cs[j] : in);
    for (int j = 0; j < Km1; ++j) new_window[ch * Km1 + j] = (1 + j < Km1) ? cs[1 + j] : in;
    return sum / (1.f + expf(-sum));
}

__device__ __forceinline__ float block_total(float v) {
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

extern "C" __global__ void loken_gdn_prep_kernel(
        const void * __restrict__ qkv, float * window, const float * __restrict__ w,
        const void * __restrict__ alpha, const void * __restrict__ beta_in,
        const float * __restrict__ a_log, const float * __restrict__ dt_bias,
        float * __restrict__ q_out, float * __restrict__ k_out, float * __restrict__ v_out,
        float * __restrict__ g_out, float * __restrict__ beta_out, float * new_window,
        int rows, int kg, int kd, int vg, int vd, int K, int tiled, float eps, int in_f16) {
    extern __shared__ float shared[];
    const int qk = kg * kd;
    const int rep = vg / kg;
    const int b = blockIdx.x;
    const int conv_dim = 2 * qk + vg * vd;
    const int elem = in_f16 ? 2 : 4;
    // Row r reads the window row r - 1 left in `new_window`, the first the one given; every
    // channel is one thread's, so a row's window is complete before the thread reads it again.
    for (int r = 0; r < rows; ++r) {
    const void * qkv_r = (const char *) qkv + (size_t) r * conv_dim * elem;
    const void * alpha_r = (const char *) alpha + (size_t) r * vg * elem;
    const void * beta_r = (const char *) beta_in + (size_t) r * vg * elem;
    float * window_r = r == 0 ? window : new_window;
    float * q_r = q_out + (size_t) r * vg * kd;
    float * k_r = k_out + (size_t) r * vg * kd;
    float * v_r = v_out + (size_t) r * vg * vd;
    float * g_r = g_out + (size_t) r * vg;
    float * beta_r_out = beta_out + (size_t) r * vg;
    if (b < kg) {
        float * sq = shared;
        float * sk = shared + kd;
        float ssq = 0.f, ssk = 0.f;
        for (int d = threadIdx.x; d < kd; d += blockDim.x) {
            const float cq = conv_silu(b * kd + d, qkv_r, window_r, w, new_window, K, in_f16);
            const float ck = conv_silu(qk + b * kd + d, qkv_r, window_r, w, new_window, K, in_f16);
            sq[d] = cq;
            sk[d] = ck;
            ssq += cq * cq;
            ssk += ck * ck;
        }
        const float invq = rsqrtf(block_total(ssq) + eps);
        const float invk = rsqrtf(block_total(ssk) + eps);
        for (int j = 0; j < rep; ++j) {
            const int hv = tiled ? j * kg + b : b * rep + j;
            for (int d = threadIdx.x; d < kd; d += blockDim.x) {
                q_r[(size_t) hv * kd + d] = sq[d] * invq;
                k_r[(size_t) hv * kd + d] = sk[d] * invk;
            }
        }
        // The next row's conv reads the shared rows only after every thread wrote this row's.
        __syncthreads();
    } else {
        const int vh = b - kg;
        for (int d = threadIdx.x; d < vd; d += blockDim.x) {
            v_r[(size_t) vh * vd + d] =
                conv_silu(2 * qk + vh * vd + d, qkv_r, window_r, w, new_window, K, in_f16);
        }
        if (threadIdx.x == 0) {
            const float x = load_in(alpha_r, vh, in_f16) + dt_bias[vh];
            g_r[vh] = a_log[vh] * logf(expf(x) + 1.f);
            beta_r_out[vh] = 1.f / (1.f + expf(-load_in(beta_r, vh, in_f16)));
        }
    }
    }
}

extern "C" void loken_gdn_prep(
        const void * qkv, float * window, const float * w, const void * alpha,
        const void * beta_in, const float * a_log, const float * dt_bias, float * q_out,
        float * k_out, float * v_out, float * g_out, float * beta_out, float * new_window,
        int rows, int kg, int kd, int vg, int vd, int K, int tiled, float eps, int in_f16,
        cudaStream_t stream) {
    const size_t shared = (size_t) 2 * kd * sizeof(float);
    loken_gdn_prep_kernel<<<kg + vg, 128, shared, stream>>>(
        qkv, window, w, alpha, beta_in, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        new_window, rows, kg, kd, vg, vd, K, tiled, eps, in_f16);
}
