// Fused causal depthwise conv1d + SiLU for the qwen3.5 gated-DeltaNet input
// projection. Replaces ~16 small tensor ops per layer (transpose, cat, Kxmul+add,
// silu, transpose, conv_state narrow) with ONE launch - the dominant remaining
// launch-overhead in the DeltaNet decode path.
//
//   out[t,c] = silu( sum_{j=0..K-1} w[c,j] * padded[c, t+j] )
//   padded[c,.] = [ conv_state[c, 0..K-2], qkv[0..seq-1, c] ]   (causal left-pad)
//   new_conv_state[c, j] = padded[c, seq + j]                   (last K-1 raw inputs)
//
// Row-major f32 layouts: qkv [seq, C], w [C, K], conv_state/new [C, K-1],
// out [seq, C]. One thread per channel c and run of `CONV_ROWS` rows; the thread of the first
// run also reads the old window and writes the new one, so the window may be advanced in place:
// the later runs start past the window and never read it.

// Rows one thread convolves; at least the longest window, so only the first run reads it.
constexpr int CONV_ROWS = 32;

extern "C" __global__ void loken_fused_conv_silu_kernel(
        const float * __restrict__ qkv,
        const float * conv_state,   // read into registers first: may be `new_conv_state`
        const float * __restrict__ w,
        float * __restrict__ out,
        float * new_conv_state,
        int C, int seq, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int Km1 = K - 1;
    const int t0 = blockIdx.y * CONV_ROWS;
    const int t1 = min(seq, t0 + CONV_ROWS);
    const bool first = blockIdx.y == 0;
    float wl[8];
    float cs[8];
    for (int j = 0; j < K; j++)   wl[j] = w[c * K + j];
    if (first) {
        for (int j = 0; j < Km1; j++) cs[j] = conv_state[c * Km1 + j];
    }

    for (int t = t0; t < t1; t++) {
        float sum = 0.f;
        for (int j = 0; j < K; j++) {
            const int m = t + j;
            const float val = (m < Km1) ? cs[m] : qkv[(m - Km1) * C + c];
            sum += wl[j] * val;
        }
        out[t * C + c] = sum / (1.f + expf(-sum));   // SiLU
    }
    if (!first) return;
    // new conv_state = last K-1 of padded
    for (int j = 0; j < Km1; j++) {
        const int m = seq + j;
        new_conv_state[c * Km1 + j] = (m < Km1) ? cs[m] : qkv[(m - Km1) * C + c];
    }
}

extern "C" void loken_fused_conv_silu(
        const float * qkv, const float * conv_state, const float * w,
        float * out, float * new_conv_state,
        int C, int seq, int K, cudaStream_t stream) {
    const int threads = 256;
    const dim3 blocks((C + threads - 1) / threads, max(1, (seq + CONV_ROWS - 1) / CONV_ROWS));
    loken_fused_conv_silu_kernel<<<blocks, threads, 0, stream>>>(
        qkv, conv_state, w, out, new_conv_state, C, seq, K);
}

// F16-INPUT variant: qkv read in F16 (the in_qkv projection output) so the
// in_qkv->F32 cast vanishes. Output + state stay F32 (downstream l2norm/recurrence
// need F32). __half2float is exact, so BIT-IDENTICAL to the F32 path.
#include <cuda_fp16.h>
extern "C" __global__ void loken_fused_conv_silu_f16in_kernel(
        const __half * __restrict__ qkv,
        const float * conv_state,   // read into registers first: may be `new_conv_state`
        const float * __restrict__ w,
        float * __restrict__ out,
        float * new_conv_state,
        int C, int seq, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int Km1 = K - 1;
    const int t0 = blockIdx.y * CONV_ROWS;
    const int t1 = min(seq, t0 + CONV_ROWS);
    const bool first = blockIdx.y == 0;
    float wl[8];
    float cs[8];
    for (int j = 0; j < K; j++)   wl[j] = w[c * K + j];
    if (first) {
        for (int j = 0; j < Km1; j++) cs[j] = conv_state[c * Km1 + j];
    }
    for (int t = t0; t < t1; t++) {
        float sum = 0.f;
        for (int j = 0; j < K; j++) {
            const int m = t + j;
            const float val = (m < Km1) ? cs[m] : __half2float(qkv[(m - Km1) * C + c]);
            sum += wl[j] * val;
        }
        out[t * C + c] = sum / (1.f + expf(-sum));   // SiLU
    }
    if (!first) return;
    for (int j = 0; j < Km1; j++) {
        const int m = seq + j;
        new_conv_state[c * Km1 + j] = (m < Km1) ? cs[m] : __half2float(qkv[(m - Km1) * C + c]);
    }
}

extern "C" void loken_fused_conv_silu_f16in(
        const void * qkv, const float * conv_state, const float * w,
        float * out, float * new_conv_state,
        int C, int seq, int K, cudaStream_t stream) {
    const int threads = 256;
    const dim3 blocks((C + threads - 1) / threads, max(1, (seq + CONV_ROWS - 1) / CONV_ROWS));
    loken_fused_conv_silu_f16in_kernel<<<blocks, threads, 0, stream>>>(
        (const __half *)qkv, conv_state, w, out, new_conv_state, C, seq, K);
}
