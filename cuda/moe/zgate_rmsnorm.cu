// Fused z-gated RMSNorm for the qwen3.5 gated-DeltaNet output. Replaces ~9
// separate tensor ops per layer (sqr, mean, +eps, sqrt, div, .norm_w, silu(z), ., reshape):
//
//   y[r,d] = (o[r,d] * rsqrt(mean_d(o[r,:]^2) + eps)) * norm_w[d] * silu(z[r,d])
//
// Row-major f32. r in [0, N) where N = seq*n_v_heads, d in [0, D) where D =
// head_v_dim. o,z [N,D], norm_w [D], y [N,D]. One warp per row; lanes reduce
// the sum-of-squares over D (D is a multiple of 32, e.g. 128).

extern "C" __global__ void loken_zgate_rmsnorm_kernel(
        const float * __restrict__ o,
        const float * __restrict__ z,
        const float * __restrict__ norm_w,
        float * __restrict__ y,
        int N, int D, float eps) {
    const int row  = blockIdx.x;          // one block (one warp) per row
    const int lane = threadIdx.x;         // 0..31
    if (row >= N) return;
    const float * orow = o + (long) row * D;
    const float * zrow = z + (long) row * D;
    float * yrow = y + (long) row * D;

    float ss = 0.f;
    for (int d = lane; d < D; d += 32) ss += orow[d] * orow[d];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    ss = __shfl_sync(0xffffffff, ss, 0);  // broadcast full sum to all lanes
    const float rms = rsqrtf(ss / (float) D + eps);

    for (int d = lane; d < D; d += 32) {
        const float zd = zrow[d];
        const float silu = zd / (1.f + expf(-zd));
        yrow[d] = orow[d] * rms * norm_w[d] * silu;
    }
}

extern "C" void loken_zgate_rmsnorm(
        const float * o, const float * z, const float * norm_w, float * y,
        int N, int D, float eps, cudaStream_t stream) {
    loken_zgate_rmsnorm_kernel<<<N, 32, 0, stream>>>(o, z, norm_w, y, N, D, eps);
}

// F16-I/O variant: z read in F16 (z_proj output) + y written in F16 (out_proj
// input) so those two casts vanish. o/norm_w stay F32; __half2float is exact and
// y's F32->F16 round is the same as before -> BIT-IDENTICAL.
#include <cuda_fp16.h>
extern "C" __global__ void loken_zgate_rmsnorm_f16io_kernel(
        const float * __restrict__ o,
        const __half * __restrict__ z,
        const float * __restrict__ norm_w,
        __half * __restrict__ y,
        int N, int D, float eps) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= N) return;
    const float * orow = o + (long) row * D;
    const __half * zrow = z + (long) row * D;
    __half * yrow = y + (long) row * D;
    float ss = 0.f;
    for (int d = lane; d < D; d += 32) ss += orow[d] * orow[d];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    ss = __shfl_sync(0xffffffff, ss, 0);
    const float rms = rsqrtf(ss / (float) D + eps);
    for (int d = lane; d < D; d += 32) {
        const float zd = __half2float(zrow[d]);
        const float silu = zd / (1.f + expf(-zd));
        yrow[d] = __float2half(orow[d] * rms * norm_w[d] * silu);
    }
}

extern "C" void loken_zgate_rmsnorm_f16io(
        const float * o, const void * z, const float * norm_w, void * y,
        int N, int D, float eps, cudaStream_t stream) {
    loken_zgate_rmsnorm_f16io_kernel<<<N, 32, 0, stream>>>(
        o, (const __half *)z, norm_w, (__half *)y, N, D, eps);
}
