// Fused L2-norm + GQA tile for the qwen3.5 gated-DeltaNet q/k. Replaces, per
// q and per k, the l2norm (sqr/sum/rsqrt/div ≈3 ops) AND the GQA broadcast
// (unsqueeze/broadcast_as/reshape ≈2 ops) with ONE kernel:
//
//   inv[t,kh] = rsqrt( sum_d x[t,kh,d]^2 + eps )
//   y[t, hv, d] = x[t, hv % kg, d] * inv[t, hv % kg]      (TILED: hv = j*kg + kh)
//
// Row-major f32: x [seq, kg, kd], y [seq, vg, kd] with vg = rep*kg. One warp per
// (t, kg-head) row reduces sum-of-squares over kd, then writes the normalized
// value to all `rep` v-heads it feeds.

extern "C" __global__ void loken_l2norm_gqa_kernel(
        const float * __restrict__ x,
        float * __restrict__ y,
        int seq, int kg, int kd, int rep, float eps) {
    const int row  = blockIdx.x;          // 0 .. seq*kg-1  (= t*kg + kh)
    const int lane = threadIdx.x;
    if (row >= seq * kg) return;
    const int t  = row / kg;
    const int kh = row % kg;
    const int vg = rep * kg;
    const float * xr = x + (long) row * kd;

    float ss = 0.f;
    for (int d = lane; d < kd; d += 32) ss += xr[d] * xr[d];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    ss = __shfl_sync(0xffffffff, ss, 0);
    const float inv = rsqrtf(ss + eps);

    for (int j = 0; j < rep; j++) {
        const int hv = j * kg + kh;
        float * yr = y + ((long) t * vg + hv) * kd;
        for (int d = lane; d < kd; d += 32) yr[d] = xr[d] * inv;
    }
}

extern "C" void loken_l2norm_gqa(
        const float * x, float * y, int seq, int kg, int kd, int rep, float eps,
        cudaStream_t stream) {
    loken_l2norm_gqa_kernel<<<seq * kg, 32, 0, stream>>>(x, y, seq, kg, kd, rep, eps);
}
