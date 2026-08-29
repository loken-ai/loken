// Fused DeltaNet gating for qwen3.5. Replaces ~6 separate tensor ops per layer:
//   g_pre[t,h] = a_log[h] * softplus(alpha[t,h] + dt_bias[h])   softplus = log(exp(x)+1)
//   beta[t,h]  = sigmoid(beta_in[t,h])
// `g_pre` is the PRE-exp decay the recurrence kernel later expf()'s. softplus
// uses the same naive form as the tensor-level path so greedy output is unchanged.
// Row-major f32: alpha, beta_in [N, H], a_log/dt_bias [H], g_pre/beta out [N, H].
// One thread per element (N*H is tiny: seq * n_v_heads).

extern "C" __global__ void loken_deltanet_gate_kernel(
        const float * __restrict__ alpha,
        const float * __restrict__ beta_in,
        const float * __restrict__ a_log,
        const float * __restrict__ dt_bias,
        float * __restrict__ g_pre,
        float * __restrict__ beta,
        int N, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * H) return;
    const int h = i % H;
    const float x = alpha[i] + dt_bias[h];
    g_pre[i] = a_log[h] * logf(expf(x) + 1.f);   // a_log . softplus(x)
    beta[i]  = 1.f / (1.f + expf(-beta_in[i]));   // sigmoid
}

extern "C" void loken_deltanet_gate(
        const float * alpha, const float * beta_in, const float * a_log, const float * dt_bias,
        float * g_pre, float * beta, int N, int H, cudaStream_t stream) {
    const int total = N * H;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    loken_deltanet_gate_kernel<<<blocks, threads, 0, stream>>>(
        alpha, beta_in, a_log, dt_bias, g_pre, beta, N, H);
}

// F16-INPUT variant: alpha/beta_in read in F16 (the a_proj/b_proj outputs) so
// their ->F32 casts vanish. a_log/dt_bias + outputs stay F32. __half2float is
// exact -> BIT-IDENTICAL to the F32 path.
#include <cuda_fp16.h>
extern "C" __global__ void loken_deltanet_gate_f16in_kernel(
        const __half * __restrict__ alpha,
        const __half * __restrict__ beta_in,
        const float * __restrict__ a_log,
        const float * __restrict__ dt_bias,
        float * __restrict__ g_pre,
        float * __restrict__ beta,
        int N, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * H) return;
    const int h = i % H;
    const float x = __half2float(alpha[i]) + dt_bias[h];
    g_pre[i] = a_log[h] * logf(expf(x) + 1.f);
    beta[i]  = 1.f / (1.f + expf(-__half2float(beta_in[i])));
}

extern "C" void loken_deltanet_gate_f16in(
        const void * alpha, const void * beta_in, const float * a_log, const float * dt_bias,
        float * g_pre, float * beta, int N, int H, cudaStream_t stream) {
    const int total = N * H;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    loken_deltanet_gate_f16in_kernel<<<blocks, threads, 0, stream>>>(
        (const __half *)alpha, (const __half *)beta_in, a_log, dt_bias, g_pre, beta, N, H);
}
