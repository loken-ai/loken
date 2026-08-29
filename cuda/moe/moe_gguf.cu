/**
 * @brief Expert GEMMs over GGUF-quantised weights, for Mixture-of-Experts decode.
 *
 * A routed pair - one token and one expert - is a block. The block's warps each take a strip of
 * that expert's output rows and walk the k axis, multiplying the expert's quantised weight blocks
 * against the token's activation, which the launcher has quantised to q8_1 first. The kernels
 * here differ only in what they do with a row's total: write it, scale it and accumulate it into
 * a real token's row, or pair it with a second GEMM's total through a gated activation.
 */
#include "gguf.cuh"
#include "mxfp4_moe.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <cassert>
#include <mutex>

// A row is padded to this many values before it is quantised, so that every format's block size
// divides it and a row is a whole number of blocks whatever k the model has.
constexpr int MATRIX_ROW_PADDING = 512;

// Where a launch stages the activation it quantised to q8_1.
//
// The buffer is kept and grown rather than allocated per call, because a captured
// `cudaMallocAsync` hands back a different pointer on replay and the captured kernels would
// then read memory that has moved. A stable pointer is what CUDA-graph capture needs.
//
// One buffer per STREAM, not per device. The kernel that reads the buffer runs on the stream
// that wrote it, and that is the only ordering there is: a host lock does not order the card.
// Two streams on one card - an LLM and an image engine, two requests in flight, or simply two
// callers - quantise into the same bytes and read each other's if they share one.
struct y_q8_1_slot {
    int device;
    cudaStream_t stream;
    void* ptr;
    size_t bytes;
};
static y_q8_1_slot g_y_q8_1[64] = {};
static int g_y_q8_1_used = 0;
static std::mutex g_y_q8_1_lock;

// The buffer, held for as long as the caller needs it.
//
// One stream is not one caller: several host threads submit to the same stream, and they all
// stage into these same bytes. So the quantise and the matmul that reads what it wrote have to
// reach the stream as a pair - if another thread's pair lands between them, the matmul reads
// the other thread's activation. Handing the pointer out under a lock is not enough, because
// nothing has been enqueued yet when the lock is dropped; the caller holds the reservation
// until both launches are on the stream.
//
// It also keeps the buffer from moving. A caller that needs more frees this one and allocates a
// larger one, and `cudaFreeAsync` only orders against work already enqueued.
struct y_q8_1_reservation {
    std::unique_lock<std::mutex> held;
    void* ptr;
    operator void*() const { return ptr; }
};

static y_q8_1_reservation get_y_q8_1_scratch(size_t needed, cudaStream_t stream) {
    int dev = 0;
    cudaGetDevice(&dev);

    std::unique_lock<std::mutex> held(g_y_q8_1_lock);
    y_q8_1_slot* slot = nullptr;
    for (int i = 0; i < g_y_q8_1_used; ++i) {
        if (g_y_q8_1[i].device == dev && g_y_q8_1[i].stream == stream) {
            slot = &g_y_q8_1[i];
            break;
        }
    }
    if (!slot) {
        // Past the table the last slot is shared, which is the old behaviour rather than a
        // failure; sixty-four is well above the streams a machine puts on one card.
        const int at = g_y_q8_1_used < 64 ? g_y_q8_1_used++ : 63;
        slot = &g_y_q8_1[at];
        slot->device = dev;
        slot->stream = stream;
        slot->ptr = nullptr;
        slot->bytes = 0;
    }
    if (needed > slot->bytes) {
        if (slot->ptr) cudaFreeAsync(slot->ptr, stream);
        cudaMallocAsync(&slot->ptr, needed, stream);
        slot->bytes = needed;
    }
    return {std::move(held), slot->ptr};
}

/// The activation, quantised to q8_1 on `stream`, with its buffer still reserved.
///
/// Every launcher below starts this way, and the reservation travels back with the pointer
/// because the two launches have to reach the stream as a pair: the buffer belongs to the
/// stream, not to the caller, and another thread's quantise landing in between is what the
/// GEMM would then read.
static y_q8_1_reservation stage_activation(const float* inputs, int rows, int size_k,
                                           int k_padded, cudaStream_t stream) {
    const dim3 grid(ceil_div(k_padded, CUDA_QUANTIZE_BLOCK_SIZE), rows, 1);
    const dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const size_t bytes = (size_t)rows * (k_padded / QK8_1) * sizeof(block_q8_1);
    y_q8_1_reservation staged = get_y_q8_1_scratch(bytes, stream);
    quantize_q8_1<<<grid, block, 0, stream>>>(inputs, staged, size_k, k_padded);
    return staged;
}

/// The shape one routed GEMM works in.
///
/// These six numbers travelled together through every kernel signature and every launch macro
/// here, always in the same order, and a launcher that swapped two of them would still compile.
/// Named once, they cannot be swapped, and a kernel has one place to read its shape from.
struct moe_dims {
    int experts;       // experts the weight tensor holds
    int topk;          // routed pairs per real token
    int pairs;         // entries of the sorted routing this launch covers
    int cols;          // output rows one expert has
    int depth;         // k, as the weight blocks store it
    int depth_padded;  // k, as the activation was padded to before quantisation
};

/// How a mat-vec launch is spread over the card: a warp per strip of output rows, a row of
/// blocks per routed pair.
struct moe_launch {
    dim3 grid;
    dim3 block;
};

static moe_launch mat_vec_launch(int cols, int pairs, int warps, int rows_per_warp) {
    moe_launch shape;
    shape.grid  = dim3(ceil_div(cols, warps * rows_per_warp), pairs, 1);
    shape.block = dim3(WARP_SIZE, warps, 1);
    return shape;
}

namespace loken_moe {

/// Where a thread sits and which entry of the routing its block serves.
///
/// A block owns one routed pair and a strip of `warps * ROWS_PER_WARP` output rows; a warp owns
/// `ROWS_PER_WARP` contiguous rows inside that strip.
struct warp_slot {
    int lane;   // thread within the warp
    int warp;   // warp within the block
    int warps;  // warps the block was launched with
    int row0;   // first output row this warp writes
    int pair;   // entry of the sorted routing this block serves
};

template <int ROWS_PER_WARP>
static __device__ __forceinline__ warp_slot claim_slot() {
    warp_slot slot;
    slot.lane  = threadIdx.x;
    slot.warp  = threadIdx.y;
    slot.warps = blockDim.y;
    slot.row0  = blockIdx.x * slot.warps * ROWS_PER_WARP + slot.warp * ROWS_PER_WARP;
    slot.pair  = blockIdx.y;
    return slot;
}

/// One entry of the routing: the pair id the sort produced, and the expert it names.
struct routed_pair {
    int token;
    int expert;
};

static __device__ __forceinline__ routed_pair read_routing(
    const int32_t * __restrict__ sorted_token_ids, const int32_t * __restrict__ expert_ids,
    int pair) {
    routed_pair routed;
    routed.token  = sorted_token_ids[pair];
    routed.expert = expert_ids[pair];
    return routed;
}

/// One expert's slab of weights, out of a tensor holding `rows` rows per expert.
///
/// A row is `depth / qk` blocks and a block is one struct, so an expert is that many structs
/// times its row count - the same arithmetic whether the caller's rows are one projection or
/// two of them concatenated.
template <int qk, typename block_t>
static __device__ __forceinline__ const block_t * expert_slab(
    const void * __restrict__ base, int expert, int rows, int depth) {
    const size_t slab_bytes = (size_t)(rows * depth) / qk * sizeof(block_t);
    return (const block_t *)((const char *)base + (size_t)expert * slab_bytes);
}

/// One row of the quantised activation. Rows are `depth_padded` wide, not `depth`: padding is
/// what makes a row a whole number of q8_1 blocks.
static __device__ __forceinline__ const block_q8_1 * activation_row(
    const void * __restrict__ base, int row, int depth_padded) {
    const size_t row_bytes = (size_t)depth_padded / QK8_1 * sizeof(block_q8_1);
    return (const block_q8_1 *)((const char *)base + (size_t)row * row_bytes);
}

/// The output row a token writes into.
static __device__ __forceinline__ float * output_row(
    float * __restrict__ base, int token, int cols) {
    return base + (size_t)token * (size_t)cols;
}

/// How one lane walks the k axis.
///
/// A weight block takes `qi / vdr` lanes, each responsible for `vdr` ints of codes inside it.
/// Those lanes start at consecutive blocks, so a warp covers `vdr * WARP_SIZE / qi` blocks in
/// one pass and then advances by that much.
struct k_walk {
    int first;    // first weight block this lane reads
    int step;     // blocks a warp covers in one pass
    int codes;    // where this lane's codes start inside a block
    int per_row;  // weight blocks in one row
};

template <int qk, int qi, int vdr>
static __device__ __forceinline__ k_walk plan_k_walk(int lane, int depth) {
    k_walk walk;
    walk.per_row = depth / qk;
    walk.step    = vdr * WARP_SIZE / qi;
    walk.first   = lane / (qi / vdr);
    walk.codes   = vdr * (lane % (qi / vdr));
    return walk;
}

/// The activation block that lines up with weight block `kbx`.
///
/// The weights' block is `qk` values wide and the activation's is QK8_1; where the format's is
/// wider, one weight block spans several activation blocks and the dot product is handed the
/// first of them.
template <int qk>
static __device__ __forceinline__ const block_q8_1 * paired_activation(
    const block_q8_1 * __restrict__ y, int kbx) {
    return &y[kbx * (qk / QK8_1)];
}

/// The warp's output rows against one activation row.
///
/// `w_rows` points at the warp's first row; rows are `walk.per_row` blocks apart. k is walked
/// once for all of them, so the activation block is read at the top of the loop and every row's
/// dot product uses that one read - which is the whole reason a warp takes more than one row.
template <int qk, int qi, typename block_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda,
          int ROWS_PER_WARP>
static __device__ __forceinline__ void accumulate_rows(
    const block_t * __restrict__ w_rows, const block_q8_1 * __restrict__ y,
    const k_walk & walk, int row0, int cols, float (&acc)[ROWS_PER_WARP]) {
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) acc[r] = 0.0f;

    #pragma unroll
    for (int kbx = walk.first; kbx < walk.per_row; kbx += walk.step) {
        const block_q8_1 * y_blk = paired_activation<qk>(y, kbx);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            if (row0 + r < cols) {
                acc[r] += vec_dot_q_cuda(&w_rows[r * walk.per_row + kbx], y_blk, walk.codes);
            }
        }
    }
}

/// Both halves of a gated FFN against one activation row.
///
/// The gate and the up projection have the same shape and the same input, so one trip along k
/// feeds both: the activation block is read once and used twice. That is what makes the two
/// GEMMs worth fusing into one kernel.
template <int qk, int qi, typename block_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda,
          int ROWS_PER_WARP>
static __device__ __forceinline__ void accumulate_gated_rows(
    const block_t * __restrict__ gate_rows, const block_t * __restrict__ up_rows,
    const block_q8_1 * __restrict__ y, const k_walk & walk, int row0, int cols,
    float (&acc_g)[ROWS_PER_WARP], float (&acc_u)[ROWS_PER_WARP]) {
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) { acc_g[r] = 0.0f; acc_u[r] = 0.0f; }

    #pragma unroll
    for (int kbx = walk.first; kbx < walk.per_row; kbx += walk.step) {
        const block_q8_1 * y_blk = paired_activation<qk>(y, kbx);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            if (row0 + r < cols) {
                acc_g[r] += vec_dot_q_cuda(&gate_rows[r * walk.per_row + kbx], y_blk, walk.codes);
                acc_u[r] += vec_dot_q_cuda(&up_rows[r * walk.per_row + kbx],   y_blk, walk.codes);
            }
        }
    }
}

/// Each row's partial sums brought together across the warp and handed to `emit` on lane 0.
///
/// The row test is uniform across the warp - a row depends on the warp, not the lane - so every
/// lane reaches the reduction together, which is what the shuffle needs.
template <int ROWS_PER_WARP, typename Emit>
static __device__ __forceinline__ void emit_rows(
    int lane, int row0, int cols, float (&acc)[ROWS_PER_WARP], Emit emit) {
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (row < cols) {
            const float total = warp_sum(acc[r]);
            if (lane == 0) emit(row, total);
        }
    }
}

/// The same for a gated pair, both halves of a row reduced before either is used.
template <int ROWS_PER_WARP, typename Emit>
static __device__ __forceinline__ void emit_gated_rows(
    int lane, int row0, int cols, float (&acc_g)[ROWS_PER_WARP],
    float (&acc_u)[ROWS_PER_WARP], Emit emit) {
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (row < cols) {
            const float g = warp_sum(acc_g[r]);
            const float u = warp_sum(acc_u[r]);
            if (lane == 0) emit(row, g, u);
        }
    }
}

/*
 * Template Parameters:
 * @tparam qk                Values one weight block holds
 * @tparam qi                Ints of codes one weight block holds
 * @tparam block_t           The weight block struct
 * @tparam vdr               Ints of codes one lane takes from a block
 * @tparam vec_dot_q_cuda    That format's dot product against a q8_1 block
 * @tparam ROWS_PER_WARP     Output rows one warp computes
 */
// One expert GEMM, plain: each routed pair's row of the output is the expert's rows against the
// token's activation, scaled by the routing weight when there is one.
//
// The warp's weight rows are staged in shared memory, so the lanes read them once as a coalesced
// sweep instead of once per pass of the k loop.
template <int qk, int qi, typename block_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void loken_moe_gemm_gguf_kernel(
    const void * __restrict__ all_weights,          // [experts, cols, depth], quantised
    const void * __restrict__ all_inputs,           // one q8_1 row per activation row
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    const float * __restrict__ topk_weights,        // one per routed pair, or null
    float * __restrict__ all_outputs,               // one float row per routed pair
    moe_dims dims
) {
    const warp_slot slot = claim_slot<ROWS_PER_WARP>();
    if (slot.row0 >= dims.cols || slot.pair >= dims.pairs) {
        return;
    }

    const routed_pair routed = read_routing(sorted_token_ids, expert_ids, slot.pair);
    if (routed.expert < 0 || routed.expert >= dims.experts) return;

    const float scale = (topk_weights) ? topk_weights[routed.token] : 1.0f;

    const block_t * __restrict__ w_expert =
        expert_slab<qk, block_t>(all_weights, routed.expert, dims.cols, dims.depth);

    // With routing weights, one entry of the routing is one activation row. Without them the
    // entries of a token are consecutive and share the row that token quantised into.
    const int input_row = topk_weights ? routed.token : (routed.token / dims.topk);
    const block_q8_1 * __restrict__ y_ptr =
        activation_row(all_inputs, input_row, dims.depth_padded);

    const k_walk walk = plan_k_walk<qk, qi, vdr>(slot.lane, dims.depth);

    // The tile holds `warps * ROWS_PER_WARP` rows; each warp fills its own share of it, and
    // every lane copies a stride of each row so the reads coalesce.
    extern __shared__ int8_t shared_bytes[];
    block_t * tile = reinterpret_cast<block_t *>(shared_bytes)
                   + (size_t)slot.warp * ROWS_PER_WARP * walk.per_row;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        if (slot.row0 + r < dims.cols) {
            const block_t * __restrict__ src = &w_expert[(size_t)(slot.row0 + r) * walk.per_row];
            for (int i = slot.lane; i < walk.per_row; i += WARP_SIZE) {
                tile[r * walk.per_row + i] = src[i];
            }
        }
    }
    __syncthreads();

    float acc[ROWS_PER_WARP];
    accumulate_rows<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP>(
        tile, y_ptr, walk, slot.row0, dims.cols, acc);

    float * __restrict__ out_ptr = output_row(all_outputs, routed.token, dims.cols);
    emit_rows(slot.lane, slot.row0, dims.cols, acc,
              [&](int row, float total) { out_ptr[row] = total * scale; });
}

} // namespace loken_moe

// One arm of a launcher's dispatch: the block type and its dot product are types, not values,
// so a row of `MOE_GGUF_FORMATS` becomes a set of template arguments and this is a macro rather
// than a function. Every format in the table gets an arm, so no weight a caller is allowed to
// send can fall through to `default` and leave the output buffer as it was found.
#define LAUNCH_MOE_GGUF(qk, qi, block_t, vdr, vec_dot_q_cuda) \
    /* The tile: one weight row per warp per row-slot, plus a margin for its alignment. */ \
    const int shared_bytes = dims.depth / qk * sizeof(block_t) * nWraps * ROWS_PER_WARP + 1024; \
    loken_moe::loken_moe_gemm_gguf_kernel<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<shape.grid, shape.block, shared_bytes, stream>>>( \
            weights, y_q8_1, sorted_token_ids, expert_ids, topk_weights, outputs, dims);

extern "C" void loken_moe_gemm_gguf(
    const float* inputs, const void* weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids, const float* topk_weights,
    float* outputs,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int quant_type,        // the weight format, numbered as `MOE_GGUF_FORMATS` numbers it
    cudaStream_t stream
) {
    const int kx_padded = pad_to(size_k, MATRIX_ROW_PADDING);
    // With routing weights every entry of the routing has its own activation row; without them
    // `topk` entries share one, so there are that many fewer rows to quantise.
    const int act_rows = topk_weights ? size_m : size_m / topk;
    // Held until the GEMM below is on the stream - see `y_q8_1_reservation`.
    const y_q8_1_reservation y_q8_1 =
        stage_activation(inputs, act_rows, size_k, kx_padded, stream);

    // ROWS_PER_WARP > 1 doubles the per-block shared-memory cost (each warp now caches that
    // many weight rows). On Blackwell the shared-mem-per-block ceiling clamps blocks-resident-
    // per-SM to 1 for ROWS_PER_WARP=2 with K=2048 K-quants, which underutilizes SMs when
    // grid.x is small (e.g. moe_intermediate=768 -> grid.x=96). Keep =1; experiment with larger
    // only when the matmul shape favours it.
    const int nWraps = 4;
    constexpr int ROWS_PER_WARP = 1;
    const moe_launch shape = mat_vec_launch(size_n, size_m, nWraps, ROWS_PER_WARP);
    const moe_dims dims = { num_experts, topk, size_m, size_n, size_k, kx_padded };

    switch (quant_type) {
#define MOE_CASE(id, name, qk, qi, vdr)                            \
        case id: { LAUNCH_MOE_GGUF(qk, qi, block_##name, vdr, vec_dot_##name##_q8_1); break; }
        MOE_GGUF_FORMATS(MOE_CASE)
#undef MOE_CASE
        default: break;
    }
}

// ------------------------------------------------------------
// MoE down-projection GEMM with topk reduction fused inline.
//
// Replaces the sequence
//   ys = loken_moe_gemm_gguf(down_inputs, down_w, topk_weights)  // [M*topk, hidden]
//   ys = ys.reshape((M, topk, hidden))?.sum(D::Minus2)?    // [M, hidden]
// with a single CUDA launch that writes scaled partial results directly
// to a pre-zeroed [M, hidden] output via atomicAdd. Saves the explicit
// sum() launch and the [M*topk, hidden] intermediate.
//
// For decode (M=1 real token, topk=8): each (token, expert) pair adds
// its weighted contribution to the same output row. atomicAdd on F32
// has minimal contention at this scale (8 contributors per output
// position scattered across hidden=2048 lanes).
// ------------------------------------------------------------
namespace loken_moe {

template <int qk, int qi, typename block_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void loken_moe_gemm_gguf_down_reduce_kernel(
    const void * __restrict__ all_weights,          // [experts, cols = hidden, depth = inter]
    const void * __restrict__ all_inputs,           // one q8_1 row per routed pair
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    const float * __restrict__ topk_weights,        // one per routed pair, required here
    float * __restrict__ all_outputs,               // one row per real token, pre-filled by the
                                                    // caller with zeros or with the residual;
                                                    // this kernel only adds to it
    moe_dims dims,
    const float * __restrict__ down_bias            // [experts, cols] F32, or null
) {
    const warp_slot slot = claim_slot<1>();
    if (slot.row0 >= dims.cols || slot.pair >= dims.pairs) return;

    const routed_pair routed = read_routing(sorted_token_ids, expert_ids, slot.pair);
    if (routed.expert < 0 || routed.expert >= dims.experts) return;

    const float scale = topk_weights[routed.token];

    const block_t * __restrict__ w_expert =
        expert_slab<qk, block_t>(all_weights, routed.expert, dims.cols, dims.depth);

    // Down's input is per routed pair, so the pair's own index IS its activation row.
    const block_q8_1 * __restrict__ y_ptr =
        activation_row(all_inputs, routed.token, dims.depth_padded);

    // No shared-memory tile: a warp takes a single row, reads it straight from global, and L1
    // carries whatever reuse the block's other warps give it.
    const k_walk walk = plan_k_walk<qk, qi, vdr>(slot.lane, dims.depth);
    float acc[1];
    accumulate_rows<qk, qi, block_t, vdr, vec_dot_q_cuda, 1>(
        &w_expert[(size_t)slot.row0 * walk.per_row], y_ptr, walk, slot.row0, dims.cols, acc);

    emit_rows(slot.lane, slot.row0, dims.cols, acc, [&](int row, float total) {
        float v = total * scale;
        // Fold the per-expert down bias: out[t] += Σ_slot w[t,s].(down.h + db[e]).
        // The slot's contribution down.h is already xscale; add xscale.db[e][row]
        // so the atomicAdd accumulates the bias term per slot exactly. (Replaces
        // the index_select + broadcast_mul + sum(1) + add chain - which
        // also unblocks CUDA-graph capture: that chain's temporaries relocated.)
        if (down_bias != nullptr) {
            v += scale * down_bias[(size_t)routed.expert * dims.cols + row];
        }
        float * out_ptr = output_row(all_outputs, routed.token / dims.topk, dims.cols);
        atomicAdd(&out_ptr[row], v);
    });
}

} // namespace loken_moe

#define LAUNCH_MOE_GGUF_DOWN_REDUCE(qk, qi, block_t, vdr, vec_dot_q_cuda) \
    loken_moe::loken_moe_gemm_gguf_down_reduce_kernel<qk, qi, block_t, vdr, vec_dot_q_cuda> \
        <<<shape.grid, shape.block, 0, stream>>>( \
            weights, y_q8_1, sorted_token_ids, expert_ids, topk_weights, outputs, \
            dims, down_bias);

// Initialize an F32 output buffer from an F16 residual. Used to fuse
// the post-MLP residual add into the loken_moe_gemm_gguf_down_reduce kernel:
// instead of alloc_zeros + atomicAdd contributions + later add_residual,
// we init the buffer to residual values then atomicAdd contributions  - 
// the residual add is "free" (one initial cast + write per element).
__global__ void loken_init_f32_from_f16(float* __restrict__ dst, const __half* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
__global__ void loken_init_f32_from_bf16(float* __restrict__ dst, const __nv_bfloat16* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

extern "C" void loken_cast_init_f32_from_dtype(
    float* dst,
    const void* src,
    int n,
    int dtype,                      // 0=f16, 1=bf16
    cudaStream_t stream
) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    if (dtype == 0) {
        loken_init_f32_from_f16<<<grid, block, 0, stream>>>(dst, (const __half*)src, n);
    } else {
#ifndef NO_BF16_KERNEL
        loken_init_f32_from_bf16<<<grid, block, 0, stream>>>(dst, (const __nv_bfloat16*)src, n);
#endif
    }
}

extern "C" void loken_moe_gemm_gguf_down_reduce(
    const float* inputs,            // [M*topk, K] f32
    const void*  weights,           // [num_experts, N, K] quantized
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const float* topk_weights,      // [M*topk] f32 - required (not optional)
    float* outputs,                 // [num_real_tokens, N] f32 - caller MUST pre-fill
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int quant_type,                 // the weight format, numbered as `MOE_GGUF_FORMATS` does
    const float* down_bias,         // [num_experts, size_n] F32, or nullptr
    cudaStream_t stream
) {
    const int kx_padded = pad_to(size_k, MATRIX_ROW_PADDING);
    // Down quantises one activation row per routed pair - no sharing, so one row per entry.
    // Held until the GEMM below is on the stream - see `y_q8_1_reservation`.
    const y_q8_1_reservation y_q8_1 =
        stage_activation(inputs, size_m, size_k, kx_padded, stream);

    // Smaller blocks -> more parallelism across SMs (no shared-mem
    // barrier means small blocks are fine). Down has size_n=hidden,
    // typically much larger than gate/up's size_n=moe_inter, so the
    // grid is already big at nWraps=4; leaving =2 for symmetry and
    // marginally better SM occupancy on small batches.
    const int nWraps = 2;
    const moe_launch shape = mat_vec_launch(size_n, size_m, nWraps, 1);
    const moe_dims dims = { num_experts, topk, size_m, size_n, size_k, kx_padded };

    switch (quant_type) {
#define MOE_CASE(id, name, qk, qi, vdr)                            \
        case id: { LAUNCH_MOE_GGUF_DOWN_REDUCE(qk, qi, block_##name, vdr, vec_dot_##name##_q8_1); break; }
        MOE_GGUF_FORMATS(MOE_CASE)
#undef MOE_CASE
        default: break;
    }
}

// ------------------------------------------------------------
// Fused gate+up MoE GEMM with SiLU activation and elementwise multiply.
//
// Computes:    output[m, n] = silu(dot(gate_w[expert][n], input[m])) *
//                                    dot(up_w[expert][n],   input[m])
//
// Compared to running loken_moe_gemm_gguf twice (once for gate, once for up) and
// then a separate silu+mul elementwise kernel, this fused kernel:
//   • shares the quantize_q8_1 of the input (one launch instead of two  - 
//     the input was already shared, but the host-side kernel invocations
//     each did a separate quantize alloc),
//   • shares the *load* of `y_ptr` block bytes from L1/L2 across the gate
//     and up partial-sum loops (one trip through global memory for the
//     input vector, used twice),
//   • writes a single [M, N] output instead of two [M, N] intermediates +
//     one [M, N] final, halving global-memory write bandwidth and
//     eliminating the activation-and-multiply launches.
// ------------------------------------------------------------
namespace loken_moe {

// No shared-memory tile here. Reading the weights straight from global hits L1/L2 - for typical
// hidden sizes the working set fits - and the shared memory that saves lets more blocks reside
// per SM.
template <int qk, int qi, typename block_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void loken_moe_gemm_gguf_gate_up_silu_mul_kernel(
    const void * __restrict__ gate_weights,         // [experts, cols, depth], quantised
    const void * __restrict__ up_weights,           // [experts, cols, depth], quantised
    const void * __restrict__ all_inputs,           // one q8_1 row per token
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    float * __restrict__ all_outputs,               // silu(gate) * up, one row per routed pair
    moe_dims dims
) {
    const warp_slot slot = claim_slot<ROWS_PER_WARP>();
    if (slot.row0 >= dims.cols || slot.pair >= dims.pairs) {
        return;
    }

    const routed_pair routed = read_routing(sorted_token_ids, expert_ids, slot.pair);
    if (routed.expert < 0 || routed.expert >= dims.experts) return;

    const block_t * __restrict__ wg_expert =
        expert_slab<qk, block_t>(gate_weights, routed.expert, dims.cols, dims.depth);
    const block_t * __restrict__ wu_expert =
        expert_slab<qk, block_t>(up_weights, routed.expert, dims.cols, dims.depth);

    // Gate and up never carry the routing weight - only down does - so the `topk` entries of a
    // token share the row that token quantised into.
    const block_q8_1 * __restrict__ y_ptr =
        activation_row(all_inputs, routed.token / dims.topk, dims.depth_padded);

    const k_walk walk = plan_k_walk<qk, qi, vdr>(slot.lane, dims.depth);
    float acc_g[ROWS_PER_WARP];
    float acc_u[ROWS_PER_WARP];
    accumulate_gated_rows<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP>(
        &wg_expert[(size_t)slot.row0 * walk.per_row],
        &wu_expert[(size_t)slot.row0 * walk.per_row],
        y_ptr, walk, slot.row0, dims.cols, acc_g, acc_u);

    float * __restrict__ out_ptr = output_row(all_outputs, routed.token, dims.cols);
    emit_gated_rows(slot.lane, slot.row0, dims.cols, acc_g, acc_u,
                    [&](int row, float g, float u) {
                        const float silu_g = g / (1.0f + __expf(-g));
                        out_ptr[row] = silu_g * u;
                    });
}

// As gate_up_silu_mul but with the gpt-oss OAI clamped-SwiGLU epilogue and
// optional per-expert gate/up biases - fuses two expert GEMMs + bias-add +
// swiglu into one kernel, never materializing the [M,N] gate/up intermediates.
//   x   = min(gate + gbias, limit)
//   gc  = clamp(up + ubias, -limit, limit)
//   out = (x * sigmoid(alpha * x)) * (1 + gc)
template <int qk, int qi, typename block_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void loken_moe_gemm_gguf_gate_up_swiglu_oai_kernel(
    const void * __restrict__ gate_weights,
    const void * __restrict__ up_weights,
    const void * __restrict__ all_inputs,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    const float * __restrict__ gate_bias,           // [experts, cols] or null
    const float * __restrict__ up_bias,             // [experts, cols] or null
    float * __restrict__ all_outputs,
    moe_dims dims,
    float alpha, float limit
) {
    const warp_slot slot = claim_slot<ROWS_PER_WARP>();
    if (slot.row0 >= dims.cols || slot.pair >= dims.pairs) return;

    const routed_pair routed = read_routing(sorted_token_ids, expert_ids, slot.pair);
    if (routed.expert < 0 || routed.expert >= dims.experts) return;

    const block_t * __restrict__ wg_expert =
        expert_slab<qk, block_t>(gate_weights, routed.expert, dims.cols, dims.depth);
    const block_t * __restrict__ wu_expert =
        expert_slab<qk, block_t>(up_weights, routed.expert, dims.cols, dims.depth);

    const block_q8_1 * __restrict__ y_ptr =
        activation_row(all_inputs, routed.token / dims.topk, dims.depth_padded);

    const k_walk walk = plan_k_walk<qk, qi, vdr>(slot.lane, dims.depth);
    float acc_g[ROWS_PER_WARP];
    float acc_u[ROWS_PER_WARP];
    accumulate_gated_rows<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP>(
        &wg_expert[(size_t)slot.row0 * walk.per_row],
        &wu_expert[(size_t)slot.row0 * walk.per_row],
        y_ptr, walk, slot.row0, dims.cols, acc_g, acc_u);

    float * __restrict__ out_ptr = output_row(all_outputs, routed.token, dims.cols);
    emit_gated_rows(slot.lane, slot.row0, dims.cols, acc_g, acc_u,
                    [&](int row, float g, float u) {
                        if (gate_bias) g += gate_bias[(size_t)routed.expert * dims.cols + row];
                        if (up_bias)   u += up_bias[(size_t)routed.expert * dims.cols + row];
                        const float x   = fminf(g, limit);
                        const float gc  = fminf(fmaxf(u, -limit), limit);
                        const float sig = 1.0f / (1.0f + __expf(-alpha * x));
                        out_ptr[row] = (x * sig) * (1.0f + gc);
                    });
}

} // namespace loken_moe

#define LAUNCH_MOE_GGUF_GATE_UP(qk, qi, block_t, vdr, vec_dot_q_cuda) \
    loken_moe::loken_moe_gemm_gguf_gate_up_silu_mul_kernel<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<shape.grid, shape.block, 0, stream>>>( \
            gate_weights, up_weights, y_q8_1, sorted_token_ids, expert_ids, outputs, dims);

extern "C" void loken_moe_gemm_gguf_gate_up_silu_mul(
    const float* inputs,
    const void* gate_weights, const void* up_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    float* outputs,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int quant_type,        // the weight format, numbered as `MOE_GGUF_FORMATS` numbers it
    cudaStream_t stream
) {
    const int kx_padded = pad_to(size_k, MATRIX_ROW_PADDING);
    // The `topk` entries of a token share one activation row, so there are that many fewer.
    // Held until the GEMM below is on the stream - see `y_q8_1_reservation`.
    const y_q8_1_reservation y_q8_1 =
        stage_activation(inputs, size_m / topk, size_k, kx_padded, stream);

    // Without shared-mem barriers we can use smaller blocks to spread
    // work across more SMs. For typical MoE intermediate sizes (768) at
    // nWraps=4 we'd only generate 192 blocks (1.5/SM on a 128-SM card).
    // nWraps=2 doubles block count to 384, ~3 blocks/SM - better.
    const int nWraps = 2;
    constexpr int ROWS_PER_WARP = 1;
    const moe_launch shape = mat_vec_launch(size_n, size_m, nWraps, ROWS_PER_WARP);
    const moe_dims dims = { num_experts, topk, size_m, size_n, size_k, kx_padded };

    switch (quant_type) {
#define MOE_CASE(id, name, qk, qi, vdr)                            \
        case id: { LAUNCH_MOE_GGUF_GATE_UP(qk, qi, block_##name, vdr, vec_dot_##name##_q8_1); break; }
        MOE_GGUF_FORMATS(MOE_CASE)
#undef MOE_CASE
        default: break;
    }
}

#define LAUNCH_MOE_GGUF_GATE_UP_SWIGLU(qk, qi, block_t, vdr, vec_dot_q_cuda) \
    loken_moe::loken_moe_gemm_gguf_gate_up_swiglu_oai_kernel<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<shape.grid, shape.block, 0, stream>>>( \
            gate_weights, up_weights, y_q8_1, sorted_token_ids, expert_ids, \
            gate_bias, up_bias, outputs, dims, alpha, limit);

// Fused gate+up GEMM with the gpt-oss OAI clamped-SwiGLU epilogue + optional
// per-expert biases.
extern "C" void loken_moe_gemm_gguf_gate_up_swiglu_oai(
    const float* inputs,
    const void* gate_weights, const void* up_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const float* gate_bias, const float* up_bias,
    float* outputs,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int quant_type,        // the weight format, numbered as `MOE_GGUF_FORMATS` numbers it
    float alpha, float limit,
    cudaStream_t stream
) {
    const int kx_padded = pad_to(size_k, MATRIX_ROW_PADDING);
    // Held until the GEMM below is on the stream - see `y_q8_1_reservation`.
    const y_q8_1_reservation y_q8_1 =
        stage_activation(inputs, size_m / topk, size_k, kx_padded, stream);

    const int nWraps = 2;
    constexpr int ROWS_PER_WARP = 1;
    const moe_launch shape = mat_vec_launch(size_n, size_m, nWraps, ROWS_PER_WARP);
    const moe_dims dims = { num_experts, topk, size_m, size_n, size_k, kx_padded };

    switch (quant_type) {
#define MOE_CASE(id, name, qk, qi, vdr)                            \
        case id: { LAUNCH_MOE_GGUF_GATE_UP_SWIGLU(qk, qi, block_##name, vdr, vec_dot_##name##_q8_1); break; }
        MOE_GGUF_FORMATS(MOE_CASE)
#undef MOE_CASE
        default: break;
    }
}

// ------------------------------------------------------------
// MoE gate||up GEMM with GELU(tanh) activation and elementwise multiply,
// for the gemma4-MoE concat weight layout where gate and up share one
// [num_experts, 2*N, K] tensor (gate = rows 0..N, up = rows N..2N).
//
// Replaces the gemma4 MoE FFN sequence:
//   gu = loken_moe_gemm_gguf(input, gate_up_exps)         // [M*topk, 2N]
//   gate_act = gelu_tanh(gu[:, :N]); down_in = gate_act * gu[:, N:]
// with a single fused matmul that emits the activated [M*topk, N] output
// directly, saving the [2N] intermediate write + the activation+mul
// elementwise launches.
// ------------------------------------------------------------
namespace loken_moe {

template <int qk, int qi, typename block_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void loken_moe_gemm_gguf_gate_up_gelu_mul_concat_kernel(
    const void * __restrict__ gate_up_weights,      // [experts, 2*cols, depth], gate then up
    const void * __restrict__ all_inputs,           // one q8_1 row per token
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    float * __restrict__ all_outputs,               // gelu(gate) * up, one row per routed pair
    moe_dims dims
) {
    const warp_slot slot = claim_slot<ROWS_PER_WARP>();
    if (slot.row0 >= dims.cols || slot.pair >= dims.pairs) {
        return;
    }

    const routed_pair routed = read_routing(sorted_token_ids, expert_ids, slot.pair);
    if (routed.expert < 0 || routed.expert >= dims.experts) return;

    // An expert's slab is twice as tall in this layout: the gate's rows, then the up's.
    const block_t * __restrict__ w_expert =
        expert_slab<qk, block_t>(gate_up_weights, routed.expert, dims.cols << 1, dims.depth);

    const k_walk walk = plan_k_walk<qk, qi, vdr>(slot.lane, dims.depth);
    // The up half starts `cols` rows in.
    const block_t * __restrict__ wu_expert = w_expert + (size_t)dims.cols * walk.per_row;

    const block_q8_1 * __restrict__ y_ptr =
        activation_row(all_inputs, routed.token / dims.topk, dims.depth_padded);

    float acc_g[ROWS_PER_WARP];
    float acc_u[ROWS_PER_WARP];
    accumulate_gated_rows<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP>(
        &w_expert[(size_t)slot.row0 * walk.per_row],
        &wu_expert[(size_t)slot.row0 * walk.per_row],
        y_ptr, walk, slot.row0, dims.cols, acc_g, acc_u);

    float * __restrict__ out_ptr = output_row(all_outputs, routed.token, dims.cols);
    emit_gated_rows(slot.lane, slot.row0, dims.cols, acc_g, acc_u,
                    [&](int row, float g, float u) {
                        // GELU, tanh approximation:
                        //   0.5*g*(1 + tanh(sqrt(2/pi)*(g + 0.044715*g^3)))
                        const float g2 = g * g;
                        const float g3 = g2 * g;
                        const float k0 = 0.7978845608028654f;
                        const float k1 = 0.044715f;
                        const float gelu_g = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g3)));
                        out_ptr[row] = gelu_g * u;
                    });
}

} // namespace loken_moe

#define LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(qk, qi, block_t, vdr, vec_dot_q_cuda) \
    loken_moe::loken_moe_gemm_gguf_gate_up_gelu_mul_concat_kernel<qk, qi, block_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<shape.grid, shape.block, 0, stream>>>( \
            gate_up_weights, y_q8_1, sorted_token_ids, expert_ids, outputs, dims);

extern "C" void loken_moe_gemm_gguf_gate_up_gelu_mul_concat(
    const float* inputs,
    const void* gate_up_weights,                 // [num_experts, 2*size_n, size_k]
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    float* outputs,                              // [size_m, size_n] = gelu(gate) * up
    int num_experts, int topk,
    int size_m,                                  // M*topk (sorted pairs count)
    int size_n,                                  // expert ffn dim (= half of the stored 2N)
    int size_k,
    int quant_type,        // the weight format, numbered as `MOE_GGUF_FORMATS` numbers it
    cudaStream_t stream
) {
    const int kx_padded = pad_to(size_k, MATRIX_ROW_PADDING);
    // Held until the GEMM below is on the stream - see `y_q8_1_reservation`.
    const y_q8_1_reservation y_q8_1 =
        stage_activation(inputs, size_m / topk, size_k, kx_padded, stream);

    const int nWraps = 2;
    constexpr int ROWS_PER_WARP = 1;
    const moe_launch shape = mat_vec_launch(size_n, size_m, nWraps, ROWS_PER_WARP);
    const moe_dims dims = { num_experts, topk, size_m, size_n, size_k, kx_padded };

    switch (quant_type) {
#define MOE_CASE(id, name, qk, qi, vdr)                            \
        case id: { LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(qk, qi, block_##name, vdr, vec_dot_##name##_q8_1); break; }
        MOE_GGUF_FORMATS(MOE_CASE)
#undef MOE_CASE
        default: break;
    }
}
