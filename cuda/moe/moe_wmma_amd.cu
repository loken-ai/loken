// The MoE prefill GEMM over quantised expert weights for AMD cards: the entry points of
// `moe_wmma_gguf.cu`, with the same arguments and the same results, on AMD's 16-bit matrix
// instructions (`cuda/hip/amd_mma16.cuh`).
//
//   input    f16 or bf16 [rows, K], one row per token or per routed pair
//   weights  [experts, N, K] in one GGUF block format
//   output   f32 [pairs, N], one row per routed pair at its slot
//
// The routing arrives sorted by expert, so one expert's pairs are one run of rows and
// `expert_offsets` says where it begins. A block owns one run and one strip of 32 output
// columns. k is walked one quantisation block at a time: the block is staged in shared memory
// as the file stores it, expanded there into the carrier, and the matrix instruction reads the
// expanded tile. The expanders split a block among the `threads` lanes of `threadIdx.x`, four
// rows of them along `threadIdx.y`, as on NVIDIA.
//
// The 32x32 strip is four 16x16 tiles. With four waves or more, wave `w` takes tile `w % 4`
// and the k tiles `t` of a block with `t % parts == w / 4`; the parts land in shared memory side
// by side and are summed in part order. With fewer waves, each takes several tiles.

#include "gguf.cuh"
#include "gguf_iq.cuh"
#include "moe_utils.cuh"
#include "amd_mma16.cuh"

namespace loken_moe_wmma_amd {

using loken_amd_mma16::Core;

/// Expander lanes along y, and the strip a block covers.
constexpr int ROWS_Y = 4;
constexpr int M_BLK = 32;
constexpr int N_BLK = 32;
constexpr int TILES = (M_BLK / 16) * (N_BLK / 16);
/// Strip columns each row of expander lanes stages.
constexpr int COLS_PER_ROW = N_BLK / ROWS_Y;

/// How many threads one block expander divides its block among.
template <typename block_t>
struct expander_threads;
template <>
struct expander_threads<block_q8_0> { static constexpr int value = 32; };
template <>
struct expander_threads<block_q5_0> { static constexpr int value = 32; };
template <>
struct expander_threads<block_q5_1> { static constexpr int value = 32; };
template <>
struct expander_threads<block_iq4_nl> { static constexpr int value = 32; };
template <>
struct expander_threads<block_iq4_xs> { static constexpr int value = 32; };
template <>
struct expander_threads<block_iq3_xxs> { static constexpr int value = 32; };
template <>
struct expander_threads<block_iq3_s> { static constexpr int value = 32; };
template <>
struct expander_threads<block_iq2_s> { static constexpr int value = 32; };
template <>
struct expander_threads<block_q2_0> { static constexpr int value = 32; };
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

template <int gguf_id>
struct prefill_format;

#define MOE_PREFILL_FORMAT(id, name, qk_values, qi, vdr)                                    \
    template <>                                                                             \
    struct prefill_format<id> {                                                             \
        using block_t = block_##name;                                                       \
        static constexpr int qk = qk_values;                                                \
        static constexpr int threads = expander_threads<block_##name>::value;               \
        template <typename act_t>                                                           \
        static __device__ __forceinline__ void expand(const uint8_t* packed, act_t* out) { \
            dequantize_block_##name<act_t>(packed, out);                                    \
        }                                                                                   \
    };
MOE_GGUF_DEQUANTISABLE(MOE_PREFILL_FORMAT)
#undef MOE_PREFILL_FORMAT

/// Waves of a block of `threads x ROWS_Y` lanes, the k parts they split a block into, and the
/// tiles each wave takes.
template <int threads>
struct Split {
    static constexpr int waves = threads * ROWS_Y / Core::wave;
    static constexpr int parts = waves >= TILES ? waves / TILES : 1;
    static constexpr int tiles_per_wave = waves >= TILES ? 1 : TILES / waves;
};

/// Byte offsets of the block's working tiles in its shared allocation: the activation and the
/// expanded weights in the carrier, the stored blocks, then one f32 accumulator per k part.
struct prefill_tiles {
    size_t activation, expanded, packed, accumulator, total;

    static __host__ __device__ constexpr prefill_tiles of(size_t qk, size_t block_bytes,
                                                          size_t parts) {
        constexpr size_t carrier = 2;
        const size_t activation = 0;
        const size_t expanded = activation + (size_t) M_BLK * qk * carrier;
        const size_t packed = expanded + (size_t) N_BLK * qk * carrier;
        const size_t packed_end = packed + (size_t) N_BLK * block_bytes;
        const size_t accumulator = (packed_end + 15) / 16 * 16;
        return prefill_tiles{activation, expanded, packed, accumulator,
                             accumulator + parts * M_BLK * N_BLK * sizeof(float)};
    }
};

struct expert_run {
    int start;
    int rows;
};

/// The activation rows of the run, zero past its end and past `size_k`.
template <typename act_t, int qk, int block_threads>
__device__ __forceinline__ void gather_activation(act_t* __restrict__ tile,
                                                  const act_t* __restrict__ input,
                                                  const int32_t* __restrict__ sorted_token_ids,
                                                  const expert_run run, int m_base, int k_base,
                                                  int size_k, int topk, bool scaled_here,
                                                  int thread_id) {
    constexpr int per_copy = 8;
    constexpr int vectors = M_BLK * qk / per_copy;
    for (int v = thread_id; v < vectors; v += block_threads) {
        const int idx = v * per_copy;
        const int m_local = idx / qk, k_local = idx % qk;
        const int m_seg = m_base + m_local;
        const int k_global = k_base + k_local;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (m_seg < run.rows && k_global < size_k) {
            const int pair = sorted_token_ids[run.start + m_seg];
            const int row = pair / (scaled_here ? 1 : topk);
            val = *reinterpret_cast<const uint4*>(&input[(size_t) row * size_k + k_global]);
        }
        *reinterpret_cast<uint4*>(&tile[m_local * qk + k_local]) = val;
    }
}

/// One stored block per strip column, each row of expander lanes taking `COLS_PER_ROW`.
template <typename fmt>
__device__ __forceinline__ void stage_packed(uint8_t* __restrict__ tile,
                                             const uint8_t* __restrict__ expert_w,
                                             size_t row_bytes, int kb, int n_base, int size_n,
                                             int row_y) {
    using block_t = typename fmt::block_t;
    constexpr size_t block_bytes = sizeof(block_t);
#pragma unroll
    for (int c = 0; c < COLS_PER_ROW; ++c) {
        const int n_local = row_y * COLS_PER_ROW + c;
        const int n_global = n_base + n_local;
        if (n_global < size_n) {
            *reinterpret_cast<block_t*>(tile + n_local * block_bytes) =
                *reinterpret_cast<const block_t*>(expert_w + (size_t) n_global * row_bytes +
                                                  (size_t) kb * block_bytes);
        }
    }
}

/// The same columns expanded into the carrier; a column past `size_n` is zero.
template <typename fmt, typename act_t>
__device__ __forceinline__ void expand_packed(act_t* __restrict__ tile,
                                              const uint8_t* __restrict__ packed, int n_base,
                                              int size_n, int row_y) {
    constexpr size_t block_bytes = sizeof(typename fmt::block_t);
#pragma unroll
    for (int c = 0; c < COLS_PER_ROW; ++c) {
        const int n_local = row_y * COLS_PER_ROW + c;
        act_t* out = tile + n_local * fmt::qk;
        if (n_base + n_local < size_n) {
            fmt::template expand<act_t>(packed + n_local * block_bytes, out);
        } else {
            for (int i = threadIdx.x; i < fmt::qk; i += fmt::threads)
                reinterpret_cast<short*>(out)[i] = 0;
        }
    }
}

/// One expert's rows against the activation.
template <typename act_t, typename fmt>
__global__ void __launch_bounds__(expander_threads<typename fmt::block_t>::value* ROWS_Y)
    moe_prefill_kernel(const act_t* __restrict__ input, const uint8_t* __restrict__ weights,
                       const int32_t* __restrict__ sorted_token_ids,
                       const int32_t* __restrict__ expert_offsets,
                       const float* __restrict__ topk_weights, float* __restrict__ output,
                       const int num_experts, const int topk, const int size_m, const int size_n,
                       const int size_k) {
    using S = Split<fmt::threads>;
    constexpr int qk = fmt::qk;
    constexpr int block_threads = fmt::threads * ROWS_Y;
    constexpr size_t block_bytes = sizeof(typename fmt::block_t);
    constexpr prefill_tiles tiles = prefill_tiles::of(qk, block_bytes, S::parts);
    constexpr int k_tiles = qk / 16;

    const int expert_id = blockIdx.x;
    if (expert_id >= num_experts) return;
    const expert_run run{expert_offsets[expert_id],
                         expert_offsets[expert_id + 1] - expert_offsets[expert_id]};
    if (run.rows == 0) return;
    const int n_base = blockIdx.y * N_BLK;
    if (n_base >= size_n) return;

    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    act_t* activation = reinterpret_cast<act_t*>(smem + tiles.activation);
    act_t* expanded = reinterpret_cast<act_t*>(smem + tiles.expanded);
    uint8_t* packed = smem + tiles.packed;
    float* accumulator = reinterpret_cast<float*>(smem + tiles.accumulator);

    const int row_y = threadIdx.y;
    const int thread_id = row_y * fmt::threads + threadIdx.x;
    const int wave = thread_id / Core::wave;
    const int lane = thread_id % Core::wave;
    const int part = wave / (S::waves / S::parts);

    const size_t row_bytes = (size_t) (size_k / qk) * block_bytes;
    const uint8_t* expert_w = weights + (size_t) expert_id * size_n * row_bytes;
    const bool scaled_here = topk_weights != nullptr;
    const int k_blocks = ceil_div(size_k, qk);

    for (int m_base = 0; m_base < run.rows; m_base += M_BLK) {
        Core::accum acc[S::tiles_per_wave] = {};

        for (int kb = 0; kb < k_blocks; ++kb) {
            gather_activation<act_t, qk, block_threads>(activation, input, sorted_token_ids, run,
                                                        m_base, kb * qk, size_k, topk,
                                                        scaled_here, thread_id);
            stage_packed<fmt>(packed, expert_w, row_bytes, kb, n_base, size_n, row_y);
            __syncthreads();
            expand_packed<fmt, act_t>(expanded, packed, n_base, size_n, row_y);
            __syncthreads();

#pragma unroll
            for (int i = 0; i < S::tiles_per_wave; ++i) {
                const int t = (wave % (S::waves / S::parts)) + i * S::waves;
                const int tm = t / (N_BLK / 16), tn = t % (N_BLK / 16);
                const act_t* a_row = activation + (tm * 16 + (lane & 15)) * qk;
                const act_t* b_row = expanded + (tn * 16 + (lane & 15)) * qk;
                for (int kt = part; kt < k_tiles; kt += S::parts) {
                    const Core::frag a = loken_amd_mma16::load_row(a_row + kt * 16, lane);
                    const Core::frag b = loken_amd_mma16::load_row(b_row + kt * 16, lane);
                    acc[i] = loken_amd_mma16::mma<act_t>(a, b, acc[i]);
                }
            }
            // The next block refills the tiles every wave reads here.
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < S::tiles_per_wave; ++i) {
            const int t = (wave % (S::waves / S::parts)) + i * S::waves;
            const int tm = t / (N_BLK / 16), tn = t % (N_BLK / 16);
            float* tile = accumulator + part * M_BLK * N_BLK;
#pragma unroll
            for (int r = 0; r < Core::acc; ++r)
                tile[(tm * 16 + Core::row(lane, r)) * N_BLK + tn * 16 + (lane & 15)] = acc[i][r];
        }
        __syncthreads();

        for (int i = thread_id; i < M_BLK * N_BLK; i += block_threads) {
            const int m_local = i / N_BLK, n_local = i % N_BLK;
            const int m_seg = m_base + m_local;
            const int n_global = n_base + n_local;
            if (m_seg >= run.rows || n_global >= size_n || run.start + m_seg >= size_m) continue;
            const int pair = sorted_token_ids[run.start + m_seg];
            float val = accumulator[i];
#pragma unroll
            for (int p = 1; p < S::parts; ++p) val += accumulator[p * M_BLK * N_BLK + i];
            if (topk_weights) val *= topk_weights[pair];
            output[(size_t) pair * size_n + n_global] = val;
        }
        // The next run of rows rewrites the accumulator the scatter reads.
        __syncthreads();
    }
}

static void expert_offsets_on_stream(const int32_t* expert_ids, int size_m, int32_t* offsets,
                                     int num_experts, hipStream_t stream) {
    int32_t* counts = nullptr;
    (void) hipMallocAsync(reinterpret_cast<void**>(&counts), num_experts * sizeof(int32_t),
                          stream);
    calculate_expert_offsets(expert_ids, size_m, counts, offsets, num_experts, stream);
    (void) hipFreeAsync(counts, stream);
}

template <typename act_t>
static void launch_prefill(dim3 grid, hipStream_t stream, const void* input,
                           const uint8_t* weights, const int32_t* sorted_token_ids,
                           const int32_t* expert_offsets, const float* topk_weights,
                           float* output, int num_experts, int topk, int size_m, int size_n,
                           int size_k, int gguf_type) {
#define MOE_PREFILL_LAUNCH(id, name, qk_values, qi, vdr)                                     \
    case id: {                                                                               \
        using fmt = prefill_format<id>;                                                      \
        constexpr size_t smem_bytes =                                                        \
            prefill_tiles::of(fmt::qk, sizeof(typename fmt::block_t),                        \
                              Split<fmt::threads>::parts)                                    \
                .total;                                                                      \
        const dim3 block(fmt::threads, ROWS_Y, 1);                                           \
        moe_prefill_kernel<act_t, fmt><<<grid, block, smem_bytes, stream>>>(                 \
            reinterpret_cast<const act_t*>(input), weights, sorted_token_ids, expert_offsets, \
            topk_weights, output, num_experts, topk, size_m, size_n, size_k);                \
        break;                                                                               \
    }
    switch (gguf_type) {
        MOE_GGUF_DEQUANTISABLE(MOE_PREFILL_LAUNCH)
        default:
            break;
    }
#undef MOE_PREFILL_LAUNCH
}

}  // namespace loken_moe_wmma_amd

extern "C" void loken_moe_gemm_gguf_prefill(const void* input, const uint8_t* weights,
                                            const int32_t* sorted_token_ids,
                                            const int32_t* expert_ids, const float* topk_weights,
                                            float* output, int num_experts, int topk, int size_m,
                                            int size_n, int size_k, int input_dtype,
                                            int gguf_type, hipStream_t stream) {
    using namespace loken_moe_wmma_amd;
    int32_t* expert_offsets = nullptr;
    if (hipMallocAsync(reinterpret_cast<void**>(&expert_offsets),
                       (num_experts + 1) * sizeof(int32_t), stream) != hipSuccess)
        return;
    expert_offsets_on_stream(expert_ids, size_m, expert_offsets, num_experts, stream);
    const dim3 grid(num_experts, ceil_div(size_n, N_BLK), 1);
    if (input_dtype == 0) {
        launch_prefill<__half>(grid, stream, input, weights, sorted_token_ids, expert_offsets,
                               topk_weights, output, num_experts, topk, size_m, size_n, size_k,
                               gguf_type);
    } else {
        launch_prefill<__hip_bfloat16>(grid, stream, input, weights, sorted_token_ids,
                                       expert_offsets, topk_weights, output, num_experts, topk,
                                       size_m, size_n, size_k, gguf_type);
    }
    (void) hipFreeAsync(expert_offsets, stream);
}

extern "C" void loken_moe_expert_offsets(const int32_t* expert_ids, int size_m,
                                         int32_t* expert_offsets, int num_experts,
                                         hipStream_t stream) {
    loken_moe_wmma_amd::expert_offsets_on_stream(expert_ids, size_m, expert_offsets, num_experts,
                                                 stream);
}
