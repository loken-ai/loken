// The MoE router gate, as an F32 GEMV split across many blocks.
//
// The gate matrix is small - a few thousand experts against a few hundred hidden units - so a
// library GEMV spends more time being launched than computing. Splitting the experts across
// blocks instead fills the device with one launch: each block is one warp, and it computes
// ROWS_PER_BLOCK output rows by walking `hidden` in warp-wide chunks.
//
//   xs:     [n_rows, hidden]        F32
//   gate_w: [n_experts, hidden]     F32, row-major
//   logits: [n_rows, n_experts]     F32
//
// One lane per F32 element of the input vector is what makes the walk exact, so `hidden` must
// be a multiple of the warp; the caller keeps a library GEMV for the shapes that are not.
#include <cuda.h>
#include <cuda_runtime.h>

// `WARP_SIZE` and the butterfly reduction come from the shared statement of the formats: this
// kernel reduces across a warp exactly as every other one here does.
#include "gguf.cuh"

namespace ll_gate_gemv {

template <int ROWS_PER_BLOCK>
__launch_bounds__(WARP_SIZE, 4)
__global__ void gate_gemv_f32_kernel(
    const float * __restrict__ xs,        // [n_rows, hidden]
    const float * __restrict__ gate_w,    // [n_experts, hidden]
    float       * __restrict__ logits,    // [n_rows, n_experts]
    int hidden, int n_rows, int n_experts) {
    const int row    = blockIdx.y;
    const int row0   = blockIdx.x * ROWS_PER_BLOCK;
    if (row >= n_rows) return;
    if (row0 >= n_experts) return;

    const int lane = threadIdx.x;

    const float * x_row = xs     + (size_t)row * hidden;
    float       * o_row = logits + (size_t)row * n_experts;

    // Per-thread accumulator for each of the ROWS_PER_BLOCK weight rows
    // this block is responsible for.
    float acc[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) acc[r] = 0.f;

    // Walk hidden in WARP_SIZE-wide strides; each lane handles one column.
    for (int k = lane; k < hidden; k += WARP_SIZE) {
        const float xv = x_row[k];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            const int erow = row0 + r;
            if (erow < n_experts) {
                const float wv = gate_w[(size_t)erow * hidden + k];
                acc[r] += wv * xv;
            }
        }
    }

    // Warp reduction across lanes, write per-row result.
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        const float sum = warp_sum(acc[r]);
        const int erow = row0 + r;
        if (lane == 0 && erow < n_experts) {
            o_row[erow] = sum;
        }
    }
}

} // namespace ll_gate_gemv

extern "C" void loken_gate_gemv_f32(
    const void * xs,           // [n_rows, hidden] F32 device pointer
    const void * gate_w,       // [n_experts, hidden] F32 device pointer
    void       * logits,       // [n_rows, n_experts] F32 device pointer, the output
    int hidden, int n_rows, int n_experts, cudaStream_t stream) {

    if (hidden % WARP_SIZE != 0) {
        return;  // a shape the lane-per-element walk cannot cover; the caller has a fallback
    }

    constexpr int ROWS_PER_BLOCK = 4;
    dim3 grid(ceil_div(n_experts, ROWS_PER_BLOCK), n_rows, 1);
    dim3 block(WARP_SIZE, 1, 1);
    ll_gate_gemv::gate_gemv_f32_kernel<ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(
        (const float *)xs, (const float *)gate_w, (float *)logits, hidden, n_rows, n_experts);
}
