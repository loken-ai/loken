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
    const float  * __restrict__ state_in,   // [b, D, L-1] F32
    const float  * __restrict__ conv_w,     // [D, L] F32
    __half * __restrict__ y_out,            // [b, D] F16
    float  * __restrict__ state_out,        // [b, D, L-1] F32
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
