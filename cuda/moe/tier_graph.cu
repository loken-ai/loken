// The card's side of a cached layer whose missing experts the host computes, for a decoded token
// or the rows of a draft verify recorded in a graph: nothing here returns to the host between two
// kernels. The layer hands the host a mailbox in page-locked memory and takes its answer from it.
//
//   plan   reads the router's choice of every row, looks each expert up in the cache's table (an
//          expert's slot, or a negative number), gives the card's experts their slots and weights,
//          and posts the choice, the slots, the rows and the layer's input in the mailbox before
//          raising the request word.
//   wait   stops the stream until the host has raised the reply word to the request's number.
//   merge  adds the host's product to what the card computed.
//
// Every request of a layer carries the next number of a counter on the card, so a replayed graph
// asks anew each time, and a reply is the answer to one request only.

#include <cuda_runtime.h>
#include <cstdint>

// Most routed pairs one launch of the plan looks up: the rows of a verify by the experts each
// routes to.
#define TIER_PLAN_PAIRS 256

extern "C" __global__ void loken_tier_plan_kernel(
        const unsigned int * __restrict__ ids,      // [rows * k] the router's experts
        const float * __restrict__ weights,         // [rows * k] their weights
        const float * __restrict__ x,               // [rows * hidden] the layer's input
        const int * table,                          // [experts] slot, or negative: not cached
        unsigned int * __restrict__ counter,        // [0] the layer's request counter, [1] whether a reply is due
        unsigned int * __restrict__ slot_out,       // [rows * k] the slot each pair reads
        float * __restrict__ weight_out,            // [rows * k] its weight, zero for a missing expert
        unsigned int * mail_ids,
        float * mail_weights,
        unsigned int * mail_slots,                  // [rows * k] the slot, or all ones for a missing expert
        float * mail_x,
        unsigned int * mail_rows,
        volatile unsigned int * mail_request,
        volatile unsigned int * mail_live,
        int rows, int k, int hidden) {
    __shared__ unsigned int seq;
    __shared__ unsigned int filler;
    __shared__ int slot_of[TIER_PLAN_PAIRS];
    const int t = threadIdx.x;
    const int pairs = rows * k;
    // The cache's table lives in page-locked memory, a read of it crosses the bus: the experts of
    // the choice are looked up together, one thread each.
    if (t < pairs) slot_of[t] = ((const volatile int *) table)[ids[t]];
    __syncthreads();
    if (t == 0) {
        seq = *counter + 1;
        *counter = seq;
        // A missing expert's pair reads a slot the card holds an expert in, at weight zero: the
        // first of the choice's slots, or the first slot when it has none. The host's reply is
        // waited for only when some expert of the choice is missing.
        unsigned int f = 0;
        bool found = false;
        unsigned int missing = 0u;
        for (int i = 0; i < pairs; ++i) {
            if (slot_of[i] >= 0) {
                if (!found) { f = (unsigned int) slot_of[i]; found = true; }
            } else {
                missing = 1u;
            }
        }
        filler = f;
        counter[1] = missing;
        *mail_rows = (unsigned int) rows;
        // A request every expert of which is cached is not waited for: the late pairs the
        // host handed for the request before must not be read again, so their count is
        // cleared here, before the host gets to it.
        if (missing == 0u) *mail_live = 0u;
    }
    __syncthreads();
    if (t < pairs) {
        const int s = slot_of[t];
        slot_out[t] = s >= 0 ? (unsigned int) s : filler;
        weight_out[t] = s >= 0 ? weights[t] : 0.0f;
        mail_ids[t] = ids[t];
        mail_weights[t] = weights[t];
        mail_slots[t] = s >= 0 ? (unsigned int) s : 0xFFFFFFFFu;
    }
    // The rows cross the bus to the host's memory: four floats a store, a whole line a warp.
    const int quads = (rows * hidden) / 4;
    const float4 * x4 = (const float4 *) x;
    float4 * mail4 = (float4 *) mail_x;
    for (int i = t; i < quads; i += blockDim.x) mail4[i] = x4[i];
    for (int i = 4 * quads + t; i < rows * hidden; i += blockDim.x) mail_x[i] = x[i];
    __syncthreads();
    __threadfence_system();
    if (t == 0) *mail_request = seq;
}

extern "C" __global__ void loken_tier_wait_kernel(
        const unsigned int * __restrict__ counter,
        volatile unsigned int * reply,
        volatile unsigned int * abort_word) {
    if (threadIdx.x == 0) {
        const unsigned int seq = counter[0];
        while (counter[1] != 0u && (int) (*reply - seq) < 0 && *abort_word == 0u) {
            __nanosleep(100);
        }
    }
    __syncthreads();
    __threadfence_system();
}

// `card` and `late` hold a weighted row per routed pair, `k` pairs a row of `hidden`: a row's
// value is the sum of its pairs' rows on the card, those run late from the transit room, and
// the host's product when one is due.
extern "C" __global__ void loken_tier_merge_kernel(
        const float * __restrict__ card,
        const float * __restrict__ late,
        const volatile float * product,
        const unsigned int * __restrict__ counter,
        float * __restrict__ out,
        int n, int k, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int row = i / hidden, col = i % hidden;
        float sum = 0.f;
        for (int j = 0; j < k; ++j) sum += card[(size_t) (row * k + j) * hidden + col];
        if (late != nullptr) {
            float extra = 0.f;
            for (int j = 0; j < k; ++j) extra += late[(size_t) (row * k + j) * hidden + col];
            sum += extra;
        }
        out[i] = counter[1] != 0u ? sum + product[i] : sum;
    }
}

extern "C" void loken_tier_plan(
        const unsigned int * ids, const float * weights, const float * x, const int * table,
        unsigned int * counter, unsigned int * slot_out, float * weight_out,
        unsigned int * mail_ids, float * mail_weights, unsigned int * mail_slots, float * mail_x,
        unsigned int * mail_rows, unsigned int * mail_request, unsigned int * mail_live,
        int rows, int k, int hidden, cudaStream_t stream) {
    loken_tier_plan_kernel<<<1, TIER_PLAN_PAIRS, 0, stream>>>(
        ids, weights, x, table, counter, slot_out, weight_out, mail_ids, mail_weights,
        mail_slots, mail_x, mail_rows, mail_request, mail_live, rows, k, hidden);
}

extern "C" void loken_tier_wait(
        const unsigned int * counter, unsigned int * reply, unsigned int * abort_word,
        cudaStream_t stream) {
    loken_tier_wait_kernel<<<1, 32, 0, stream>>>(counter, reply, abort_word);
}

extern "C" void loken_tier_merge(
        const float * card, const float * late, const float * product, const unsigned int * counter,
        float * out, int n, int k, int hidden, cudaStream_t stream) {
    const int threads = 256;
    loken_tier_merge_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        card, late, product, counter, out, n, k, hidden);
}

/// One more answer told to the host: the counter advanced on the card and its value written
/// where the host reads it, a page-locked mapping, after the copies queued before it landed.
extern "C" __global__ void loken_tier_tell_kernel(unsigned int * counter, volatile unsigned int * told) {
    if (threadIdx.x == 0) {
        const unsigned int next = counter[0] + 1u;
        counter[0] = next;
        __threadfence_system();
        *told = next;
    }
}

extern "C" void loken_tier_tell(unsigned int * counter, unsigned int * told, cudaStream_t stream) {
    loken_tier_tell_kernel<<<1, 32, 0, stream>>>(counter, told);
}
