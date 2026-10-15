// The routed pairs of a mixture layer as one card of a split layer reads them: the card holds the
// experts `first .. first + count` of the router's, and each of its pairs names one of them. A
// pair of an expert the card holds names it at its weight; a pair of another card's expert names
// a held expert its row does not route to, at weight zero, since the kernels that group a row's
// pairs by expert take each expert of a row once. The card's sums over its pairs, added across the
// cards, are the layer's.

#include <cuda_runtime.h>
#include <cstdint>

// Most experts a row routes to.
#define TP_LOCAL_MAX_K 32

extern "C" __global__ void loken_tp_local_pairs_kernel(
        const unsigned int * __restrict__ ids,     // [rows * k] the router's experts
        const float * __restrict__ weights,        // [rows * k] their weights
        unsigned int * __restrict__ local_ids,     // [rows * k] the card's expert each pair reads
        float * __restrict__ local_weights,        // [rows * k] its weight, zero for another card's
        int rows, int k, int first, int count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || k > TP_LOCAL_MAX_K) return;
    const unsigned int * r_ids = ids + (size_t) row * k;
    const float * r_w = weights + (size_t) row * k;
    unsigned int * o_ids = local_ids + (size_t) row * k;
    float * o_w = local_weights + (size_t) row * k;
    int held[TP_LOCAL_MAX_K];
    for (int j = 0; j < k; ++j) {
        const int l = (int) r_ids[j] - first;
        held[j] = (l >= 0 && l < count) ? l : -1;
    }
    // The spare experts, in order, skip those the row's own pairs name.
    int spare = 0;
    for (int j = 0; j < k; ++j) {
        if (held[j] >= 0) {
            o_ids[j] = (unsigned int) held[j];
            o_w[j] = r_w[j];
            continue;
        }
        for (;;) {
            bool taken = false;
            for (int i = 0; i < k; ++i) taken |= held[i] == spare;
            if (!taken) break;
            ++spare;
        }
        o_ids[j] = (unsigned int) (spare < count ? spare : 0);
        o_w[j] = 0.0f;
        ++spare;
    }
}

extern "C" void loken_tp_local_pairs(
        const unsigned int * ids, const float * weights, unsigned int * local_ids,
        float * local_weights, int rows, int k, int first, int count, cudaStream_t stream) {
    const int threads = 128;
    loken_tp_local_pairs_kernel<<<(rows + threads - 1) / threads, threads, 0, stream>>>(
        ids, weights, local_ids, local_weights, rows, k, first, count);
}
