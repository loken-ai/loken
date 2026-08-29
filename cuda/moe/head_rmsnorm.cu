// Fused per-head RMSNorm for qwen3.5 attention Q/K (and any [..., D] RMSNorm).
// Replaces ~6 separate tensor ops per call (sqr, mean, +eps, sqrt, div, .weight):
//
//   y[r,d] = x[r,d] * rsqrt(mean_d(x[r,:]^2) + eps) * w[d]
//
// Row-major f32: x,y [N, D], w [D]. One warp per row reduces over D (multiple
// of 32, e.g. head_dim=256).

extern "C" __global__ void loken_head_rmsnorm_kernel(
        const float * __restrict__ x,
        const float * __restrict__ w,
        float * __restrict__ y,
        int N, int D, float eps) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= N) return;
    const float * xr = x + (long) row * D;
    float * yr = y + (long) row * D;

    float ss = 0.f;
    for (int d = lane; d < D; d += 32) ss += xr[d] * xr[d];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    ss = __shfl_sync(0xffffffff, ss, 0);
    const float rms = rsqrtf(ss / (float) D + eps);

    for (int d = lane; d < D; d += 32) yr[d] = xr[d] * rms * w[d];
}

extern "C" void loken_head_rmsnorm(
        const float * x, const float * w, float * y, int N, int D, float eps,
        cudaStream_t stream) {
    loken_head_rmsnorm_kernel<<<N, 32, 0, stream>>>(x, w, y, N, D, eps);
}
