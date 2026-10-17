// The rows of a q8_0 embedding table gathered on the card.
//
// Row `ids[r]` of a table of `dim` values a row (dim a multiple of 32) is decoded to `out[r]`:
// each value is the block's f16 scale times its signed byte. One block of threads a row, a
// thread a run of values; no host lookup, so the launch can sit inside a graph capture.
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace loken_moe {

__global__ void loken_embed_rows_q8_0_kernel(
    const void * __restrict__ table, const uint32_t * __restrict__ ids, int n, int dim,
    float * __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= n) return;
    const int blocks = dim / QK8_0;
    const block_q8_0 * row = (const block_q8_0 *) table + (size_t) ids[r] * blocks;
    float * o = out + (size_t) r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const block_q8_0 & b = row[i / QK8_0];
        o[i] = __half2float(b.d) * (float) b.qs[i % QK8_0];
    }
}

} // namespace loken_moe

extern "C" void loken_embed_rows_q8_0(
    const void * table, const uint32_t * ids, int n, int dim, float * out, cudaStream_t stream) {
    if (n <= 0) return;
    loken_moe::loken_embed_rows_q8_0_kernel<<<n, 128, 0, stream>>>(table, ids, n, dim, out);
}
