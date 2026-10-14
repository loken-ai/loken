// A BF16 weight times a few F32 rows: y[r] = w x[r], accumulated in F32.
//
//   x: [rows, k]  F32
//   w: [n, k]     BF16, row-major
//   y: [rows, n]  F32
//
// One block per output row. Each thread reads eight weights at a time and applies them to every
// activation row, so the weight row is read once for all the rows. A row's sum takes the same
// order whatever the number of rows beside it: its result from a launch of several rows equals
// the result from a launch of its own.
//
// `k` must be a multiple of 8, `rows` at most 8 and the pointers 16-byte aligned.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace ll_bf16_matvec {

constexpr int WARP = 32;
constexpr int MAX_THREADS = 256;
constexpr int PER_LOAD = 8;

__device__ __forceinline__ float from_bf16(uint32_t bits16) {
    return __uint_as_float(bits16 << 16);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int ROWS>
__global__ void bf16_matvec_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                   float* __restrict__ y, int k, int n, int rows) {
    const int out = blockIdx.x;
    const int t = threadIdx.x;
    const uint4* wrow = reinterpret_cast<const uint4*>(w + (size_t)out * k);
    const int loads = k / PER_LOAD;

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (int c = t; c < loads; c += blockDim.x) {
        const uint4 p = wrow[c];
        const uint32_t words[4] = {p.x, p.y, p.z, p.w};
        float wv[PER_LOAD];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wv[2 * i] = from_bf16(words[i] & 0xffffu);
            wv[2 * i + 1] = from_bf16(words[i] >> 16);
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < rows) {
                const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * k) + 2 * c;
                const float4 a = xr[0];
                const float4 b = xr[1];
                acc[r] = __fmaf_rn(wv[0], a.x, acc[r]);
                acc[r] = __fmaf_rn(wv[1], a.y, acc[r]);
                acc[r] = __fmaf_rn(wv[2], a.z, acc[r]);
                acc[r] = __fmaf_rn(wv[3], a.w, acc[r]);
                acc[r] = __fmaf_rn(wv[4], b.x, acc[r]);
                acc[r] = __fmaf_rn(wv[5], b.y, acc[r]);
                acc[r] = __fmaf_rn(wv[6], b.z, acc[r]);
                acc[r] = __fmaf_rn(wv[7], b.w, acc[r]);
            }
        }
    }

    __shared__ float partial[ROWS][MAX_THREADS / WARP];
    const int warps = blockDim.x / WARP;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float s = warp_sum(acc[r]);
        if ((t & (WARP - 1)) == 0) partial[r][t / WARP] = s;
    }
    __syncthreads();
    if (t < WARP) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float s = warp_sum(t < warps ? partial[r][t] : 0.0f);
            if (t == 0 && r < rows) y[(size_t)r * n + out] = s;
        }
    }
}

// Threads a block takes for a row of `k`: one per load up to the cap, in whole warps. Fixed by
// `k` alone, so the order of a row's sum is too.
int threads_for(int k) {
    const int loads = k / PER_LOAD;
    const int warps = (loads + WARP - 1) / WARP;
    const int t = warps * WARP;
    return t < MAX_THREADS ? t : MAX_THREADS;
}

template <int ROWS>
void launch(const float* x, const uint16_t* w, float* y, int k, int n, int rows, cudaStream_t s) {
    bf16_matvec_kernel<ROWS><<<n, threads_for(k), 0, s>>>(x, w, y, k, n, rows);
}

}  // namespace ll_bf16_matvec

// The caller checks the shape: `k` a multiple of 8, `rows` from 1 to 8.
extern "C" void loken_bf16_matvec(const void* x, const void* w, void* y, int k, int n, int rows,
                                  cudaStream_t stream) {
    using namespace ll_bf16_matvec;
    const float* xf = (const float*)x;
    const uint16_t* wb = (const uint16_t*)w;
    float* yf = (float*)y;
    if (rows == 1) {
        launch<1>(xf, wb, yf, k, n, rows, stream);
    } else if (rows <= 2) {
        launch<2>(xf, wb, yf, k, n, rows, stream);
    } else if (rows <= 4) {
        launch<4>(xf, wb, yf, k, n, rows, stream);
    } else {
        launch<8>(xf, wb, yf, k, n, rows, stream);
    }
}
