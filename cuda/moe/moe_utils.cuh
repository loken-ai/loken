// Where each expert's tokens begin.
//
// The routing arrives sorted by expert, so one expert's tokens are one contiguous run and the
// GEMM only needs to know where each run starts. That is a histogram followed by a running
// total, which is what this header is.

#undef __CUDA_FP8_TYPES_EXIST__
#include <cuda.h>
#include <cuda_runtime.h>

/// How many tokens each expert was given.
///
/// An atomic increment per token: nothing downstream depends on the order the increments
/// arrive in, only on the totals.
static __global__ void count_tokens_per_expert_kernel(const int32_t* __restrict__ expert_ids,
                                                      int32_t* __restrict__ counts,
                                                      const int size_m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_m) {
        // The ids come from a sorted routing, so every one of them names an expert that exists.
        atomicAdd(&counts[expert_ids[i]], 1);
    }
}

/// The counts as offsets: `offsets[e]` is where expert `e`'s run begins, and
/// `offsets[num_experts]` is the total number of routed tokens.
///
/// One block does the whole scan. Each thread first sums a consecutive run of experts on its
/// own, so what the parallel scan carries is one value per thread whatever the expert count
/// is; the scan then doubles its reach each round and finishes in log2(threads) of them. A
/// second launch to do this would cost more than the scan itself.
static __global__ void expert_offsets_kernel(const int32_t* __restrict__ counts,
                                             int32_t* __restrict__ offsets,
                                             const int num_experts) {
    extern __shared__ int32_t running[];
    const int tid = threadIdx.x;
    const int per_thread = (num_experts + blockDim.x - 1) / blockDim.x;
    const int first = tid * per_thread;
    const int last = min(first + per_thread, num_experts);

    int32_t mine = 0;
    for (int e = first; e < last; ++e) {
        mine += counts[e];
    }
    running[tid] = mine;
    __syncthreads();

    for (int reach = 1; reach < blockDim.x; reach <<= 1) {
        const int32_t behind = tid >= reach ? running[tid - reach] : 0;
        __syncthreads();
        running[tid] += behind;
        __syncthreads();
    }

    // `running[tid]` now counts every expert up to and including this thread's run. An offset
    // is what came BEFORE, so the thread's own share comes back off and the rest is walked
    // forward - which puts the first expert at zero without a special case.
    int32_t before = running[tid] - mine;
    for (int e = first; e < last; ++e) {
        offsets[e] = before;
        before += counts[e];
    }
    if (tid == blockDim.x - 1) {
        offsets[num_experts] = running[tid];
    }
}

/// Fill `offsets` (`num_experts + 1` ints) from a sorted `expert_ids`. `counts` is scratch of
/// `num_experts` ints and does not need to arrive zeroed.
static void calculate_expert_offsets(const int32_t* d_expert_ids, int size_m,
                                     int32_t* d_expert_counts, int32_t* d_expert_offsets,
                                     int num_experts, cudaStream_t stream) {
    cudaMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int32_t), stream);

    constexpr int count_threads = 256;
    const int count_blocks = (size_m + count_threads - 1) / count_threads;
    count_tokens_per_expert_kernel<<<count_blocks, count_threads, 0, stream>>>(
        d_expert_ids, d_expert_counts, size_m);

    // A whole warp at least, one block at most; past that a thread takes several experts.
    const int scan_threads = min(1024, max(32, num_experts));
    expert_offsets_kernel<<<1, scan_threads, scan_threads * sizeof(int32_t), stream>>>(
        d_expert_counts, d_expert_offsets, num_experts);
}
