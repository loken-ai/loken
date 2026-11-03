// The rows of a float embedding table gathered on the card.
//
// Row `ids[r]` of a table of `dim` values a row, stored as f16, bf16 or f32, is widened to f32
// in `out[r]`. One block of threads a row, a thread a run of values; no host lookup, so the
// launch can sit inside a graph capture.
// An id at or past `vocab` writes zeros for its row.
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace loken_moe {

__device__ __forceinline__ float widen(const __half v) { return __half2float(v); }
// A bf16 value is the top half of the f32 it widens to.
__device__ __forceinline__ float widen_bf16(const uint16_t v) { return __uint_as_float((uint32_t) v << 16); }
__device__ __forceinline__ float widen(const float v) { return v; }

__global__ void loken_embed_rows_f16_kernel(
    const __half * __restrict__ table, const uint32_t * __restrict__ ids, int n, int dim, int vocab,
    float * __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= n) return;
    const uint32_t id = ids[r];
    const bool ok = id < (uint32_t) vocab;
    const __half * row = table + (size_t) (ok ? id : 0) * dim;
    float * o = out + (size_t) r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = ok ? widen(row[i]) : 0.0f;
}

__global__ void loken_embed_rows_bf16_kernel(
    const uint16_t * __restrict__ table, const uint32_t * __restrict__ ids, int n, int dim, int vocab,
    float * __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= n) return;
    const uint32_t id = ids[r];
    const bool ok = id < (uint32_t) vocab;
    const uint16_t * row = table + (size_t) (ok ? id : 0) * dim;
    float * o = out + (size_t) r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = ok ? widen_bf16(row[i]) : 0.0f;
}

__global__ void loken_embed_rows_f32_kernel(
    const float * __restrict__ table, const uint32_t * __restrict__ ids, int n, int dim, int vocab,
    float * __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= n) return;
    const uint32_t id = ids[r];
    const bool ok = id < (uint32_t) vocab;
    const float * row = table + (size_t) (ok ? id : 0) * dim;
    float * o = out + (size_t) r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = ok ? widen(row[i]) : 0.0f;
}

} // namespace loken_moe

extern "C" void loken_embed_rows_f16(
    const void * table, const uint32_t * ids, int n, int dim, int vocab, float * out, cudaStream_t stream) {
    if (n <= 0) return;
    loken_moe::loken_embed_rows_f16_kernel<<<n, 256, 0, stream>>>((const __half *) table, ids, n, dim, vocab, out);
}

extern "C" void loken_embed_rows_bf16(
    const void * table, const uint32_t * ids, int n, int dim, int vocab, float * out, cudaStream_t stream) {
    if (n <= 0) return;
    loken_moe::loken_embed_rows_bf16_kernel<<<n, 256, 0, stream>>>((const uint16_t *) table, ids, n, dim, vocab, out);
}

extern "C" void loken_embed_rows_f32(
    const void * table, const uint32_t * ids, int n, int dim, int vocab, float * out, cudaStream_t stream) {
    if (n <= 0) return;
    loken_moe::loken_embed_rows_f32_kernel<<<n, 256, 0, stream>>>((const float *) table, ids, n, dim, vocab, out);
}
