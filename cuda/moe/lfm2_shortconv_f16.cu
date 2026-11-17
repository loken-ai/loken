// F16-I/O variant of the LFM2 gated short-conv inner (cf. the F32 NVRTC kernel
// fused_lfm2_shortconv_f32). Reads bcx and writes y in F16 (the residual-stream
// dtype) so the surrounding in_proj->F32 and y->F16 casts vanish - lfm2 decode is
// launch-bound, and the shortconv is its biggest cast source (~2/layer x ~34
// layers). The conv state stays F32 and all math runs in F32, so the result is
// BIT-IDENTICAL to the F32 path (the old code did the same F16->F32 and F32->F16
// conversions, just as separate cast kernels).
//   bg/cg/xg = bcx[B|C|X][d]   (F16 -> F32);  bx = bg*xg
//   acc      = Σ_{k<L-1} state_in[b,d,k]*conv_w[d,k] + bx*conv_w[d,L-1]
//   y[b,d]   = (F32->F16)(cg * acc)
//   state_out= shift_left(state_in) with bx appended  (length L-1, F32)
#include <cuda_fp16.h>

extern "C" __global__ void loken_lfm2_shortconv_f16io_kernel(
    const __half * __restrict__ bcx,        // [b, 3*D] F16
    const float  * state_in,                // [b, D, L-1] F32; may be `state_out`, read ahead of each write
    const float  * __restrict__ conv_w,     // [D, L] F32
    __half * __restrict__ y_out,            // [b, D] F16
    float  * state_out,                     // [b, D, L-1] F32
    const int b, const int D, const int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * D) return;
    const int bi = idx / D;
    const int d  = idx % D;
    const float bg = __half2float(bcx[bi * 3 * D + d]);
    const float cg = __half2float(bcx[bi * 3 * D + D + d]);
    const float xg = __half2float(bcx[bi * 3 * D + 2 * D + d]);
    const float bx = bg * xg;
    const int Lm1 = L - 1;
    const float* st = state_in + (size_t)(bi * D + d) * Lm1;
    const float* cw = conv_w + (size_t)d * L;
    float acc = 0.0f;
    #pragma unroll 1
    for (int k = 0; k < Lm1; ++k) acc += st[k] * cw[k];
    acc += bx * cw[Lm1];
    y_out[bi * D + d] = __float2half(cg * acc);
    float* so = state_out + (size_t)(bi * D + d) * Lm1;
    #pragma unroll 1
    for (int k = 0; k < Lm1 - 1; ++k) so[k] = st[k + 1];
    so[Lm1 - 1] = bx;
}

extern "C" void loken_lfm2_shortconv_f16io(
    const void * bcx, const float * state_in, const float * conv_w,
    void * y_out, float * state_out, int b, int D, int L, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (b * D + threads - 1) / threads;
    loken_lfm2_shortconv_f16io_kernel<<<blocks, threads, 0, stream>>>(
        (const __half *)bcx, state_in, conv_w, (__half *)y_out, state_out, b, D, L);
}

// The same convolution over the rows of a prompt chunk, one launch for every row.
//   bx[t]    = B[t] * X[t]
//   padded   = [ state_in[b,d,0..L-2], bx[0], .., bx[seq-1] ]
//   acc[t]   = sum_{k<L-1} padded[t+k]*conv_w[d,k] + bx[t]*conv_w[d,L-1]
//   y[b,t,d] = C[t] * acc[t]
//   state_out[b,d,k] = padded[seq+k]
// Each row sums its taps in the order of the token kernel above, so the rows and the state are
// those of `seq` token steps.
// Layouts: bcx [b, seq, 3*D], y [b, seq, D], state [b, D, L-1].
// One thread per channel d and run of `SHORTCONV_ROWS` rows, batch row b on the grid's z.
// Only the first run reads the old window and writes the new one, so the state may be advanced
// in place: the later runs start past the window and never read it.

// Rows one thread convolves; at least the longest window, so only the first run reads it.
constexpr int SHORTCONV_ROWS = 32;

__device__ __forceinline__ float shortconv_in(const __half v) { return __half2float(v); }
__device__ __forceinline__ float shortconv_in(const float v) { return v; }
__device__ __forceinline__ void shortconv_out(__half * p, const float v) { *p = __float2half(v); }
__device__ __forceinline__ void shortconv_out(float * p, const float v) { *p = v; }

template <typename T>
__global__ void loken_lfm2_shortconv_rows_kernel(
    const T * __restrict__ bcx,             // [b, seq, 3*D]
    const float * state_in,                 // [b, D, L-1]; may be `state_out`
    const float * __restrict__ conv_w,      // [D, L] F32
    T * __restrict__ y_out,                 // [b, seq, D]
    float * state_out,                      // [b, D, L-1]
    const int seq, const int D, const int L
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const int bi = blockIdx.z;
    const int Lm1 = L - 1;
    const int t0 = blockIdx.y * SHORTCONV_ROWS;
    const int t1 = min(seq, t0 + SHORTCONV_ROWS);
    const T * rows = bcx + (size_t)bi * seq * 3 * D;
    const float * st = state_in + (size_t)(bi * D + d) * Lm1;
    const float * cw = conv_w + (size_t)d * L;
    // The value at `m` of the padded sequence: the old window, then the products.
    auto padded = [&](const int m) -> float {
        if (m < Lm1) return st[m];
        const T * row = rows + (size_t)(m - Lm1) * 3 * D;
        return shortconv_in(row[d]) * shortconv_in(row[2 * D + d]);
    };
    for (int t = t0; t < t1; ++t) {
        const T * row = rows + (size_t)t * 3 * D;
        const float bx = shortconv_in(row[d]) * shortconv_in(row[2 * D + d]);
        const float cg = shortconv_in(row[D + d]);
        float acc = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < Lm1; ++k) acc += padded(t + k) * cw[k];
        acc += bx * cw[Lm1];
        shortconv_out(y_out + ((size_t)bi * seq + t) * D + d, cg * acc);
    }
    if (blockIdx.y != 0) return;
    // Every value of the new window is read before the first is written.
    float window[SHORTCONV_ROWS];
    #pragma unroll 1
    for (int k = 0; k < Lm1; ++k) window[k] = padded(seq + k);
    float * so = state_out + (size_t)(bi * D + d) * Lm1;
    #pragma unroll 1
    for (int k = 0; k < Lm1; ++k) so[k] = window[k];
}

extern "C" int loken_lfm2_shortconv_rows_window() {
    return SHORTCONV_ROWS;
}

extern "C" void loken_lfm2_shortconv_rows(
    const void * bcx, const float * state_in, const float * conv_w,
    void * y_out, float * state_out, int b, int seq, int D, int L, int in_f16,
    cudaStream_t stream
) {
    const int threads = 256;
    const dim3 blocks((D + threads - 1) / threads, max(1, (seq + SHORTCONV_ROWS - 1) / SHORTCONV_ROWS), b);
    if (in_f16) {
        loken_lfm2_shortconv_rows_kernel<__half><<<blocks, threads, 0, stream>>>(
            (const __half *)bcx, state_in, conv_w, (__half *)y_out, state_out, seq, D, L);
    } else {
        loken_lfm2_shortconv_rows_kernel<float><<<blocks, threads, 0, stream>>>(
            (const float *)bcx, state_in, conv_w, (float *)y_out, state_out, seq, D, L);
    }
}
