// Prefill GEMM for a Mixture-of-Experts layer whose weights stay in their quantised blocks.
//
// The routing arrives sorted by expert, so the pairs of one expert are one contiguous run of
// rows and `expert_offsets` says where that run begins. A block of this launch therefore owns
// one run and one strip of output columns.
//
// The matrix instruction wants 16-bit floats and the weights are four to six bits, so k is
// walked one quantisation block at a time: the block is copied into shared memory as the file
// stores it, expanded there, and the expanded tile is what the instruction reads.

#include "gguf.cuh"
#include <mma.h>
#include "moe_utils.cuh"

namespace wmma = nvcuda::wmma;

// ---------------------------------------------------------------------------------------
// The shape a block works in, and the fragments that shape implies.

/// The tile one warp multiplies. In half precision the instruction has a single shape and it is
/// not ours to pick; how many of them a block covers is.
struct mma_tile {
    static constexpr int m = 16, n = 16, k = 16;
};

/// Two warps along each axis: a block spans twice the instruction's tile in both directions, so
/// the weight tile one warp brings in serves the row of warps beside it.
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;

constexpr int M_BLK = WARPS_M * mma_tile::m;
constexpr int N_BLK = WARPS_N * mma_tile::n;

/// Each warp brings in this many of the strip's columns.
constexpr int COLS_PER_WARP = N_BLK / WARPS_PER_BLOCK;

template <typename act_t>
using a_fragment =
    wmma::fragment<wmma::matrix_a, mma_tile::m, mma_tile::n, mma_tile::k, act_t, wmma::row_major>;
template <typename act_t>
using b_fragment =
    wmma::fragment<wmma::matrix_b, mma_tile::m, mma_tile::n, mma_tile::k, act_t, wmma::col_major>;
using acc_fragment =
    wmma::fragment<wmma::accumulator, mma_tile::m, mma_tile::n, mma_tile::k, float>;

/// The activation moves sixteen bytes at a time - the widest load the hardware has, and eight
/// values of either carrier compiled here, which is what makes the copy one instruction a step.
using copy_vec = float4;
constexpr int VALUES_PER_COPY = sizeof(copy_vec) / sizeof(half);

// ---------------------------------------------------------------------------------------
// The formats this path compiles for.
//
// `gguf.cuh` states the list once, with the id each caller sends and the name every block type
// and expander is built from, so the switch below is generated rather than kept in step by
// hand. A format the list leaves out gets no case, no instantiation and no launch: that is how
// MXFP4 stays out, having a dot product but no block expander this path could call.

/// How many threads one block expander divides its block among.
///
/// It follows from how that function splits its codes, not from the block's width - a 32-value
/// block goes one value to a lane, while a superblock is split by how many of its codes one
/// lane can reach through a single byte, which differs between the six-bit layouts and the
/// nibble ones. So it is stated per block type.
template <typename block_t>
struct expander_threads;
template <>
struct expander_threads<block_q8_0> { static constexpr int value = 32; };
template <>
struct expander_threads<block_q5_0> { static constexpr int value = 32; };
template <>
struct expander_threads<block_q4_K> { static constexpr int value = 32; };
template <>
struct expander_threads<block_q2_K> { static constexpr int value = 64; };
template <>
struct expander_threads<block_q3_K> { static constexpr int value = 64; };
template <>
struct expander_threads<block_q5_K> { static constexpr int value = 64; };
template <>
struct expander_threads<block_q6_K> { static constexpr int value = 64; };

/// All the kernel asks about a format, keyed by the id the caller sends. Left undeclared for an
/// id with no expander, so such an id cannot be named here even by mistake.
template <int gguf_id>
struct prefill_format;

#define MOE_PREFILL_FORMAT(id, name, qk_values, qi, vdr)                             \
    template <>                                                                      \
    struct prefill_format<id> {                                                      \
        using block_t = block_##name;                                                \
        static constexpr int qk = qk_values;                                         \
        static constexpr int threads = expander_threads<block_##name>::value;        \
        template <typename act_t>                                                    \
        static __device__ __forceinline__ void expand(const uint8_t* packed,         \
                                                      act_t* out) {                  \
            dequantize_block_##name<act_t>(packed, out);                             \
        }                                                                            \
    };
MOE_GGUF_DEQUANTISABLE(MOE_PREFILL_FORMAT)
#undef MOE_PREFILL_FORMAT

// ---------------------------------------------------------------------------------------

/// Where a block's four working tiles sit inside its shared allocation.
///
/// Byte offsets from the base, in the order they are carved out: the activation in the carrier
/// the instruction reads, the weights in that same carrier once expanded, the weights as the
/// file stores them, then the float accumulator. The accumulator is pushed up to the next float
/// boundary, because a stored block is not a whole number of floats wide for every format.
///
/// The host sizes the allocation and the kernel carves it, both from here, so the layout and
/// the number of bytes it needs can no longer drift apart.
struct prefill_tiles {
    size_t activation, expanded, packed, accumulator, total;

    static __host__ __device__ constexpr prefill_tiles of(size_t qk, size_t block_bytes) {
        // Both carriers compiled here are sixteen bits wide, so one width covers them.
        constexpr size_t carrier = sizeof(half);
        const size_t activation = 0;
        const size_t expanded = activation + (size_t)M_BLK * qk * carrier;
        const size_t packed = expanded + (size_t)N_BLK * qk * carrier;
        const size_t packed_end = packed + (size_t)N_BLK * block_bytes;
        const size_t accumulator =
            packed_end + (alignof(float) - packed_end % alignof(float)) % alignof(float);
        return prefill_tiles{activation, expanded, packed, accumulator,
                             accumulator + (size_t)M_BLK * N_BLK * sizeof(float)};
    }
};

/// The run of routed rows one expert owns.
struct expert_run {
    int start;
    int rows;
};

__device__ __forceinline__ expert_run run_of(const int32_t* __restrict__ expert_offsets,
                                             int expert_id) {
    const int start = expert_offsets[expert_id];
    return expert_run{start, expert_offsets[expert_id + 1] - start};
}

/// The activation row a routed pair reads.
///
/// `scaled_here` tells the two callers apart: when this launch applies the routing weight, one
/// row of the activation is one routed pair; when it does not, `topk` consecutive pairs share
/// the row they were routed from and the caller combines them afterwards.
__device__ __forceinline__ int activation_row(int pair, int topk, bool scaled_here) {
    return pair / (scaled_here ? 1 : topk);
}

/// The activation rows this run names, gathered where the fragment expects to find them.
///
/// A row past the end of the run is filled with zeros rather than branched around, which keeps
/// the fragment load uniform across the block. So is a tail of k: the last quantisation block
/// is always walked, and whatever of it lies past `size_k` reads as zero.
template <typename act_t, int qk, int block_threads>
__device__ __forceinline__ void gather_activation(act_t* __restrict__ tile,
                                                  const act_t* __restrict__ input,
                                                  const int32_t* __restrict__ sorted_token_ids,
                                                  const expert_run run, int m_base, int k_base,
                                                  int size_k, int topk, bool scaled_here,
                                                  int thread_id) {
    constexpr size_t vectors = (size_t)M_BLK * qk / VALUES_PER_COPY;
    copy_vec zeros;
    zeros.x = zeros.y = zeros.z = zeros.w = 0.0f;

#pragma unroll
    for (size_t v = thread_id; v < vectors; v += block_threads) {
        const size_t idx = v * VALUES_PER_COPY;
        const size_t m_local = idx / qk, k_local = idx % qk;
        const int m_seg = m_base + (int)m_local;
        const int k_global = k_base + (int)k_local;

        copy_vec* dst = reinterpret_cast<copy_vec*>(&tile[m_local * qk + k_local]);
        if (m_seg < run.rows && k_global < size_k) {
            const int pair = sorted_token_ids[run.start + m_seg];
            const int row = activation_row(pair, topk, scaled_here);
            *dst = *reinterpret_cast<const copy_vec*>(&input[(size_t)row * size_k + k_global]);
        } else {
            *dst = zeros;
        }
    }
}

/// One stored block per column of the strip, copied whole, each warp taking `COLS_PER_WARP`.
template <typename fmt>
__device__ __forceinline__ void stage_packed(uint8_t* __restrict__ tile,
                                             const uint8_t* __restrict__ expert_w,
                                             size_t row_bytes, int kb, int n_base, int size_n,
                                             int warp_id) {
    using block_t = typename fmt::block_t;
    constexpr size_t block_bytes = sizeof(block_t);
    const size_t k_offset_bytes = (size_t)kb * block_bytes;

#pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) {
        const int n_local = warp_id * COLS_PER_WARP + c;
        const int n_global = n_base + n_local;
        if (n_local < N_BLK && n_global < size_n) {
            *reinterpret_cast<block_t*>(tile + n_local * block_bytes) =
                *reinterpret_cast<const block_t*>(expert_w + (size_t)n_global * row_bytes
                                                  + k_offset_bytes);
        }
    }
}

/// The same columns, expanded out of their blocks into the carrier the instruction reads.
template <typename fmt, typename act_t>
__device__ __forceinline__ void expand_packed(act_t* __restrict__ tile,
                                              const uint8_t* __restrict__ packed, int n_base,
                                              int size_n, int warp_id) {
    constexpr size_t block_bytes = sizeof(typename fmt::block_t);

#pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) {
        const int n_local = warp_id * COLS_PER_WARP + c;
        if (n_local < N_BLK && n_base + n_local < size_n) {
            fmt::template expand<act_t>(packed + n_local * block_bytes, tile + n_local * fmt::qk);
        }
    }
}

/// The finished strip, written out to the rows the routing named.
template <int block_threads>
__device__ __forceinline__ void scatter_strip(float* __restrict__ output,
                                              const float* __restrict__ tile,
                                              const int32_t* __restrict__ sorted_token_ids,
                                              const float* __restrict__ topk_weights,
                                              const expert_run run, int m_base, int n_base,
                                              int size_m, int size_n, int thread_id) {
#pragma unroll
    for (int i = thread_id; i < M_BLK * N_BLK; i += block_threads) {
        const int m_local = i / N_BLK, n_local = i % N_BLK;
        const int m_seg = m_base + m_local;
        const int n_global = n_base + n_local;
        if (m_seg >= run.rows || n_global >= size_n) continue;
        if (run.start + m_seg >= size_m) continue;

        const int pair = sorted_token_ids[run.start + m_seg];
        float val = tile[m_local * N_BLK + n_local];
        if (topk_weights) {
            val *= topk_weights[pair];
        }
        output[(size_t)pair * size_n + n_global] = val;
    }
}

// ---------------------------------------------------------------------------------------

/// One expert's rows against the activation, on the tensor cores.
///
/// The format arrives as a policy rather than a runtime id, so the expander is chosen once at
/// compile time and the inner loop has no switch left to walk.
template <typename act_t, typename fmt>
__global__ void loken_moe_gemm_gguf_prefill_kernel(const act_t* __restrict__ input,
                                                   const uint8_t* __restrict__ weights,
                                                   const int32_t* __restrict__ sorted_token_ids,
                                                   const int32_t* __restrict__ expert_offsets,
                                                   const float* __restrict__ topk_weights,
                                                   float* __restrict__ output,
                                                   const int num_experts, const int topk,
                                                   const int32_t size_m, const int32_t size_n,
                                                   const int32_t size_k) {
    const int expert_id = blockIdx.x;
    if (expert_id < 0 || expert_id >= num_experts) return;

    const expert_run run = run_of(expert_offsets, expert_id);
    if (run.rows == 0) return;

    const int n_base = blockIdx.y * N_BLK;
    if (n_base >= size_n) return;

    constexpr int qk = fmt::qk;
    constexpr size_t block_bytes = sizeof(typename fmt::block_t);
    constexpr int block_threads = WARPS_PER_BLOCK * fmt::threads;
    constexpr prefill_tiles tiles = prefill_tiles::of(qk, block_bytes);
    // One stored block of k is this many of the instruction's own k-tiles.
    constexpr int k_tiles = qk / mma_tile::k;

    extern __shared__ uint8_t smem[];
    act_t* activation = reinterpret_cast<act_t*>(smem + tiles.activation);
    act_t* expanded = reinterpret_cast<act_t*>(smem + tiles.expanded);
    uint8_t* packed = smem + tiles.packed;
    float* accumulator = reinterpret_cast<float*>(smem + tiles.accumulator);

    const int warp_id = threadIdx.y;
    const int thread_id = warp_id * fmt::threads + threadIdx.x;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const size_t row_bytes = (size_k / qk) * block_bytes;
    const uint8_t* expert_w = weights + (size_t)expert_id * size_n * row_bytes;

    const bool scaled_here = topk_weights != nullptr;
    const int k_blocks = ceil_div(size_k, qk);

    for (int m_base = 0; m_base < run.rows; m_base += M_BLK) {
        acc_fragment c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kb = 0; kb < k_blocks; ++kb) {
            gather_activation<act_t, qk, block_threads>(activation, input, sorted_token_ids, run,
                                                        m_base, kb * qk, size_k, topk,
                                                        scaled_here, thread_id);
            stage_packed<fmt>(packed, expert_w, row_bytes, kb, n_base, size_n, warp_id);
            __syncthreads();

            expand_packed<fmt, act_t>(expanded, packed, n_base, size_n, warp_id);
            __syncthreads();

#pragma unroll
            for (int t = 0; t < k_tiles; ++t) {
                const int k_tile = t * mma_tile::k;
                a_fragment<act_t> a_frag;
                b_fragment<act_t> b_frag;
                wmma::load_matrix_sync(a_frag, activation + warp_m * mma_tile::m * qk + k_tile, qk);
                wmma::load_matrix_sync(b_frag, expanded + warp_n * mma_tile::n * qk + k_tile, qk);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // The next step refills the activation tile before either barrier above, so every
            // warp has to be finished reading this one first. The weight tiles need no barrier
            // of their own: they are written after the first sync, which a warp only reaches
            // once it has left this loop.
            //
            // Without it, the two warps sharing `warp_m` read the activation at different
            // moments and one of them can be handed the next step's - half the columns of a row
            // come out wrong, and only once the warps drift far enough apart for it, which is to
            // say under load and soonest for the formats whose expander has the most to do.
            __syncthreads();
        }

        // Through shared memory rather than straight out: the fragment holds its elements in the
        // instruction's order, and the scatter below wants rows.
        wmma::store_matrix_sync(accumulator + warp_m * mma_tile::m * N_BLK + warp_n * mma_tile::n,
                                c_frag, N_BLK, wmma::mem_row_major);
        __syncthreads();

        scatter_strip<block_threads>(output, accumulator, sorted_token_ids, topk_weights, run,
                                     m_base, n_base, size_m, size_n, thread_id);
    }
}

// ---------------------------------------------------------------------------------------

/// Where each expert's run of routed tokens begins, with the histogram it is built from.
///
/// The counts are working memory that no caller reads, so allocating and freeing them belongs
/// with the call rather than around it - and both callers here were writing that scaffolding
/// out themselves. `offsets` holds `num_experts + 1` ints and is the caller's to free.
static void expert_offsets_on_stream(const int32_t* expert_ids, int size_m, int32_t* offsets,
                                     int num_experts, cudaStream_t stream) {
    int32_t* counts = nullptr;
    cudaMallocAsync(&counts, num_experts * sizeof(int32_t), stream);
    calculate_expert_offsets(expert_ids, size_m, counts, offsets, num_experts, stream);
    cudaFreeAsync(counts, stream);
}

/// One prefill launch.
///
/// The carrier is a template parameter - there are two, and each is its own kernel - while the
/// format id only exists at run time, so it is a switch. Every case draws its geometry, its
/// shared size and its block shape from that format's own policy, so the list is walked once
/// here instead of once to size the launch and again to make it.
template <typename act_t>
static void launch_prefill(dim3 grid, cudaStream_t stream, const void* input,
                           const uint8_t* weights, const int32_t* sorted_token_ids,
                           const int32_t* expert_offsets, const float* topk_weights,
                           float* output, int num_experts, int topk, int size_m, int size_n,
                           int size_k, int gguf_type) {
#define MOE_PREFILL_LAUNCH(id, name, qk_values, qi, vdr)                                   \
    case id: {                                                                             \
        using fmt = prefill_format<id>;                                                    \
        constexpr size_t smem_bytes =                                                      \
            prefill_tiles::of(fmt::qk, sizeof(typename fmt::block_t)).total;               \
        const dim3 block(fmt::threads, WARPS_PER_BLOCK, 1);                                \
        loken_moe_gemm_gguf_prefill_kernel<act_t, fmt>                                     \
            <<<grid, block, smem_bytes, stream>>>(reinterpret_cast<const act_t*>(input),   \
                                                  weights, sorted_token_ids, expert_offsets, \
                                                  topk_weights, output, num_experts, topk, \
                                                  size_m, size_n, size_k);                 \
        break;                                                                             \
    }
    switch (gguf_type) {
        MOE_GGUF_DEQUANTISABLE(MOE_PREFILL_LAUNCH)
        default:
            break;
    }
#undef MOE_PREFILL_LAUNCH
}

extern "C" void loken_moe_gemm_gguf_prefill(
    const void* input, const uint8_t* weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const float* topk_weights, float* output,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int input_dtype,   // the activation carrier: 0 half, 1 bfloat16
    int gguf_type,     // the weight format, by the id `gguf.cuh` gives it
    cudaStream_t stream
) {
    int32_t* expert_offsets = nullptr;
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
    expert_offsets_on_stream(expert_ids, size_m, expert_offsets, num_experts, stream);

    // One block per expert per strip of output columns.
    const dim3 grid(num_experts, ceil_div(size_n, N_BLK), 1);

    if (input_dtype == 0) {
        launch_prefill<half>(grid, stream, input, weights, sorted_token_ids, expert_offsets,
                             topk_weights, output, num_experts, topk, size_m, size_n, size_k,
                             gguf_type);
    } else {
#ifndef NO_BF16_KERNEL
        launch_prefill<nv_bfloat16>(grid, stream, input, weights, sorted_token_ids,
                                    expert_offsets, topk_weights, output, num_experts, topk,
                                    size_m, size_n, size_k, gguf_type);
#endif
    }
    cudaFreeAsync(expert_offsets, stream);
}

/// The offset builder on its own, for a judge that can compare it with a host prefix sum.
///
/// The expert offsets decide which tokens each expert's GEMM reads. Nothing downstream can
/// tell a wrong offset from a wrong weight - both come out as a plausible number in the wrong
/// place - so the builder needs a test that reaches it directly.
extern "C" void loken_moe_expert_offsets(const int32_t* expert_ids, int size_m,
                                         int32_t* expert_offsets, int num_experts,
                                         cudaStream_t stream) {
    expert_offsets_on_stream(expert_ids, size_m, expert_offsets, num_experts, stream);
}
