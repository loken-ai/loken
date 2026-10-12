// The elementwise stages around the shared expert of a mixture, for one decoded token.
//
//   swiglu   out[i] = silu(gate[i]) * up[i]
//   tail     out[i] = routed[i] + sigmoid(dot(w, x)) * down[i]
//
// `gate`, `up` and `down` are F32 or F16, as the projections that made them wrote them; every
// other tensor is F32. The tail is one block: it reduces the scalar gate and applies it.

#include <cuda_fp16.h>
#include <stddef.h>

namespace {

__device__ __forceinline__ float load_as_f32(const void * p, size_t i, int f16) {
    return f16 ? __half2float(((const __half *) p)[i]) : ((const float *) p)[i];
}

} // namespace

extern "C" __global__ void loken_shexp_swiglu_kernel(
        const void * __restrict__ gate, const void * __restrict__ up, float * __restrict__ out,
        int n, int gate_f16, int up_f16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = load_as_f32(gate, i, gate_f16);
    out[i] = g / (1.f + expf(-g)) * load_as_f32(up, i, up_f16);
}

extern "C" void loken_shexp_swiglu(
        const void * gate, const void * up, float * out, int n, int gate_f16, int up_f16,
        cudaStream_t stream) {
    loken_shexp_swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        gate, up, out, n, gate_f16, up_f16);
}

// One block a row: `n` values a row, `w` the one gate row every row reads.
extern "C" __global__ void loken_shexp_tail_kernel(
        const float * __restrict__ routed, const void * __restrict__ down,
        const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ out,
        int n, int down_f16) {
    __shared__ float partial[32];
    const size_t row0 = (size_t) blockIdx.x * n;
    routed += row0;
    x += row0;
    out += row0;
    down = (const char *) down + row0 * (down_f16 ? 2 : 4);
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += w[i] * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = acc;
    __syncthreads();
    float total = (threadIdx.x < (blockDim.x + 31) / 32) ? partial[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) {
        for (int off = 16; off > 0; off >>= 1) total += __shfl_down_sync(0xffffffff, total, off);
        if (threadIdx.x == 0) partial[0] = total;
    }
    __syncthreads();
    const float gate = 1.f / (1.f + expf(-partial[0]));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = routed[i] + gate * load_as_f32(down, i, down_f16);
    }
}

extern "C" void loken_shexp_tail(
        const float * routed, const void * down, const float * x, const float * w, float * out,
        int n, int down_f16, int rows, cudaStream_t stream) {
    loken_shexp_tail_kernel<<<rows, 1024, 0, stream>>>(routed, down, x, w, out, n, down_f16);
}
