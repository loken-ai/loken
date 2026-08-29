/**
 * @brief Fused MoE gate matmul + softmax + top-k.
 *
 * Replaces two launches per MoE layer:
 *   1. logits = gate.forward(x)   (F32 SGEMV via cublas)
 *   2. (weights, ids) = topk_softmax(logits)
 * with a single CUDA launch.
 *
 * Per token (one CUDA block, 32 threads / one warp):
 *   1. Load x[hidden] into shared memory (cooperatively).
 *   2. Each thread computes ceil(n_experts/32) logits, each = dot(x, W_row).
 *   3. Run the same warp-local softmax + iterative argmax + optional renorm
 *      as `topk_softmax_kernel` (kept inline to avoid template gymnastics).
 *
 * Layout:
 *   x:        [hidden]                    F32
 *   gate_w:   [n_experts, hidden]         F32 (row-major)
 *   weights:  [n_expert_used]             F32   (output)
 *   ids:      [n_expert_used]             u32   (output)
 *
 * Constraints:
 *   - hidden must be a multiple of 4 (we use float4 loads).
 *   - n_experts ∈ {32, 64, 128, 256}.
 *   - Single token per block (n_rows is a sweep dim, but typical use is
 *     per-token inside the MoE forward).
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
// `WARP_SIZE` and the two warp butterflies this file reduces with are stated once, with the
// rest of the shared CUDA vocabulary.
#include "gguf.cuh"

#define MAX_HIDDEN_FOR_FUSED_GATE 8192

namespace ll_gate_topk {

// The softmax + iterative argmax + optional renorm, over logits ALREADY held one-per-lane in
// `wt`. Factored out so the fused gate above and the logits-in entry point below run the
// same code rather than two copies that have to be kept agreeing - the tie-break in
// particular (equal weight -> lowest expert id) is what makes routing reproducible, and a
// second copy is a second place for it to drift.
template <int n_experts, int experts_per_thread, bool with_norm>
__device__ __forceinline__ void select_topk_from_warp_logits(
    float (&wt)[experts_per_thread],
    int lane,
    int n_expert_used,
    float * __restrict__ weights_row,
    uint32_t * __restrict__ ids_row
) {
    // -- Softmax (in-place over wt[]) --------------------------------
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * WARP_SIZE;
        const bool active = (n_experts % WARP_SIZE == 0) || (idx < n_experts);
        if (active) max_val = fmaxf(max_val, wt[i]);
    }
    max_val = warp_max(max_val);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * WARP_SIZE;
        const bool active = (n_experts % WARP_SIZE == 0) || (idx < n_experts);
        if (active) {
            const float v = __expf(wt[i] - max_val);
            wt[i] = v;
            sum += v;
        } else {
            wt[i] = 0.f;
        }
    }
    sum = warp_sum(sum);
    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] *= inv_sum;
    }

    // -- Iterative argmax to extract top-k ---------------------------
    float    output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) output_weights[i] = 0.f;


    float wt_sum = 0.f;
    for (int k = 0; k < n_expert_used; k++) {
        float max_v   = wt[0];
        int   max_exp = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_v) {
                max_v = wt[i];
                max_exp = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float v_o = __shfl_xor_sync(0xFFFFFFFFu, max_v,   mask, WARP_SIZE);
            const int   e_o = __shfl_xor_sync(0xFFFFFFFFu, max_exp, mask, WARP_SIZE);
            if (v_o > max_v || (v_o == max_v && e_o < max_exp)) {
                max_v = v_o;
                max_exp = e_o;
            }
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_v;
        }
        if ((max_exp & (WARP_SIZE - 1)) == lane) {
            wt[max_exp / WARP_SIZE] = -INFINITY;
            ids_row[k] = (uint32_t)max_exp;
            if constexpr (with_norm) {
                wt_sum += max_v;
            }
        }
    }

    if constexpr (with_norm) {
        wt_sum = warp_sum(wt_sum);
        const float inv_wt_sum = 1.0f / wt_sum;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_wt_sum;
        }
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            weights_row[idx] = output_weights[i];
        }
    }
}

// A sigmoid router differs from a softmax one in more than the activation. Each expert's
// probability stands on its own, so nothing normalises across them; and the top-k is chosen on
// the probability PLUS a per-expert bias that only steers the choice, while the weight handed
// back is the unbiased probability. Selection value and output value therefore travel as a
// pair through the argmax, which is what makes this a different reduction from the softmax
// one above rather than the same one with another activation in front of it.
//
// The tie-break is the same rule: equal selection values go to the lowest expert id, so
// routing does not depend on which lane happened to hold which expert.
template <int n_experts, int experts_per_thread, bool with_norm>
__device__ __forceinline__ void select_topk_from_warp_probs(
    float (&prob)[experts_per_thread],
    float (&sel)[experts_per_thread],
    int lane,
    int n_expert_used,
    float scale,
    float * __restrict__ weights_row,
    uint32_t * __restrict__ ids_row
) {
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) output_weights[i] = 0.f;

    float wt_sum = 0.f;
    for (int k = 0; k < n_expert_used; k++) {
        // Best of this lane's own experts, carrying the unbiased prob and the expert id.
        float max_s = sel[0]; float max_p = prob[0]; int max_e = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && sel[i] > max_s) {
                max_s = sel[i]; max_p = prob[i]; max_e = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float s_o = __shfl_xor_sync(0xFFFFFFFFu, max_s, mask, WARP_SIZE);
            const float p_o = __shfl_xor_sync(0xFFFFFFFFu, max_p, mask, WARP_SIZE);
            const int   e_o = __shfl_xor_sync(0xFFFFFFFFu, max_e, mask, WARP_SIZE);
            if (s_o > max_s || (s_o == max_s && e_o < max_e)) { max_s = s_o; max_p = p_o; max_e = e_o; }
        }
        if ((k & (WARP_SIZE - 1)) == lane) output_weights[k / WARP_SIZE] = max_p;
        // The lane that owns the winner takes it out of the running for the next round.
        if ((max_e & (WARP_SIZE - 1)) == lane) {
            sel[max_e / WARP_SIZE] = -INFINITY;
            ids_row[k] = (uint32_t)max_e;
            if constexpr (with_norm) wt_sum += max_p;
        }
    }

    if constexpr (with_norm) {
        wt_sum = warp_sum(wt_sum);
        // The floor is the smallest normal half: k probabilities can sum to less than that,
        // and dividing by it rather than by the sum keeps the result finite.
        const float inv = 1.0f / fmaxf(wt_sum, 6.103515625e-5f);
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) output_weights[i] *= inv;
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) output_weights[i] *= scale;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) weights_row[idx] = output_weights[i];
    }
}

/// Bring one row of the input into shared memory, the warp sharing the work.
///
/// Every expert's dot product reads the whole row, so it is read from global memory once. Four
/// floats at a time when the width allows it, since that is one instruction instead of four.
__device__ __forceinline__ void load_row_to_shared(
    const float * __restrict__ x_row, float * __restrict__ x_shared, int hidden, int lane
) {
    if ((hidden & 3) == 0) {
        const float4 * x4 = reinterpret_cast<const float4 *>(x_row);
        float4       * s4 = reinterpret_cast<float4 *>(x_shared);
        const int hidden4 = hidden >> 2;
        for (int i = lane; i < hidden4; i += WARP_SIZE) s4[i] = x4[i];
    } else {
        for (int i = lane; i < hidden; i += WARP_SIZE) x_shared[i] = x_row[i];
    }
}

/// One expert's logit: that expert's weight row against the shared input row.
///
/// A single lane runs the whole dot, because the warp is spending its width on experts rather
/// than on one expert's length. Four-wide where the row allows it, and the four products are
/// summed into the accumulator in a fixed order so every expert is scored the same way.
__device__ __forceinline__ float dot_row_shared(
    const float * __restrict__ w_row, const float * __restrict__ x_shared, int hidden
) {
    float acc = 0.f;
    if ((hidden & 3) == 0) {
        const float4 * w4 = reinterpret_cast<const float4 *>(w_row);
        const float4 * s4 = reinterpret_cast<const float4 *>(x_shared);
        const int hidden4 = hidden >> 2;
        #pragma unroll 4
        for (int k = 0; k < hidden4; k++) {
            float4 a = w4[k];
            float4 b = s4[k];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int k = 0; k < hidden; k++) {
            acc += w_row[k] * x_shared[k];
        }
    }
    return acc;
}

template <int n_experts, bool with_norm>
__launch_bounds__(WARP_SIZE, 4)
__global__ void gate_topk_softmax_kernel(
    const float * __restrict__ x_in,        // [n_rows, hidden]
    const float * __restrict__ gate_w,      // [n_experts, hidden]
    float       * __restrict__ weights_out, // [n_rows, n_expert_used]
    uint32_t    * __restrict__ ids_out,     // [n_rows, n_expert_used]
    int hidden,
    int n_rows,
    int n_expert_used
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float x_shared[];

    const int lane = threadIdx.x;

    load_row_to_shared(x_in + (size_t)row * hidden, x_shared, hidden, lane);
    __syncwarp();

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    // Compute per-thread logits: each lane handles experts_per_thread experts.
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = lane + i * WARP_SIZE;
        if (n_experts % WARP_SIZE == 0 || expert < n_experts) {
            wt[i] = dot_row_shared(gate_w + (size_t)expert * hidden, x_shared, hidden);
        } else {
            wt[i] = -INFINITY;
        }
    }

    select_topk_from_warp_logits<n_experts, experts_per_thread, with_norm>(
        wt, lane, n_expert_used,
        weights_out + (size_t)row * n_expert_used,
        ids_out + (size_t)row * n_expert_used);
}

// Same selection, over logits a caller computed elsewhere. The fused kernel above wins when
// the router GEMV fits its shared-memory budget; this is the other half of the path, taken
// when it does not.
template <int n_experts, bool with_norm>
__launch_bounds__(WARP_SIZE, 4)
__global__ void topk_softmax_from_logits_kernel(
    const float * __restrict__ logits,      // [n_rows, n_experts]
    float       * __restrict__ weights_out, // [n_rows, n_expert_used]
    uint32_t    * __restrict__ ids_out,     // [n_rows, n_expert_used]
    int n_rows,
    int n_expert_used
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    const float * logits_row = logits + (size_t)row * n_experts;
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = lane + i * WARP_SIZE;
        wt[i] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits_row[expert] : -INFINITY;
    }
    select_topk_from_warp_logits<n_experts, experts_per_thread, with_norm>(
        wt, lane, n_expert_used,
        weights_out + (size_t)row * n_expert_used,
        ids_out + (size_t)row * n_expert_used);
}

// SIGMOID-routed variant (lfm2 / deepseek-style): per-expert sigmoid (no
// cross-expert normalization), top-k SELECTED by (sigmoid + bias) but the
// OUTPUT weight is the UNBIASED sigmoid prob, optional top-k renorm, then
// xscale. One launch replaces gate-gemv + sigmoid + bias-add + sort + narrow
// + gather + sum + div + mul (~8-9 separate tensor ops per MoE layer).
template <int n_experts, bool with_norm>
__launch_bounds__(WARP_SIZE, 4)
__global__ void gate_topk_sigmoid_kernel(
    const float * __restrict__ x_in,        // [n_rows, hidden]
    const float * __restrict__ gate_w,      // [n_experts, hidden]
    const float * __restrict__ bias,        // [n_experts] or nullptr (selection bias)
    float       * __restrict__ weights_out, // [n_rows, n_expert_used]
    uint32_t    * __restrict__ ids_out,     // [n_rows, n_expert_used]
    int hidden,
    int n_rows,
    int n_expert_used,
    float scale
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    extern __shared__ float x_shared[];
    const int lane = threadIdx.x;

    load_row_to_shared(x_in + (size_t)row * hidden, x_shared, hidden, lane);
    __syncwarp();

    constexpr int ept = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float prob[ept], sel[ept];
#pragma unroll
    for (int i = 0; i < ept; i++) {
        const int expert = lane + i * WARP_SIZE;
        if (n_experts % WARP_SIZE == 0 || expert < n_experts) {
            const float acc = dot_row_shared(gate_w + (size_t)expert * hidden, x_shared, hidden);
            const float p = 1.0f / (1.0f + __expf(-acc));
            prob[i] = p;
            sel[i]  = p + (bias ? bias[expert] : 0.0f);
        } else {
            prob[i] = 0.0f;
            sel[i]  = -INFINITY;
        }
    }

    select_topk_from_warp_probs<n_experts, ept, with_norm>(
        prob, sel, lane, n_expert_used, scale,
        weights_out + (size_t)row * n_expert_used,
        ids_out + (size_t)row * n_expert_used);
}

// POST-matmul sigmoid router: takes PRE-COMPUTED logits (from the fast cuBLAS
// gemv) and fuses sigmoid + bias-select top-k + gather(unbiased prob) + renorm
// + scale into one launch. Unlike gate_topk_sigmoid this does NO matmul, so it
// keeps the optimized gemv and just erases the ~7 tiny post-ops (incl. the
// asort) that dominate launch count on launch-bound MoE decode.
template <int n_experts, bool with_norm>
__launch_bounds__(WARP_SIZE, 8)
__global__ void topk_sigmoid_post_kernel(
    const float * __restrict__ logits_in,   // [n_rows, n_experts]
    const float * __restrict__ bias,        // [n_experts] or nullptr
    float       * __restrict__ weights_out, // [n_rows, n_expert_used]
    uint32_t    * __restrict__ ids_out,     // [n_rows, n_expert_used]
    int n_rows,
    int n_expert_used,
    float scale
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int lane = threadIdx.x;
    const float * lrow = logits_in + (size_t)row * n_experts;

    constexpr int ept = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float prob[ept], sel[ept];
#pragma unroll
    for (int i = 0; i < ept; i++) {
        const int expert = lane + i * WARP_SIZE;
        if (n_experts % WARP_SIZE == 0 || expert < n_experts) {
            const float p = 1.0f / (1.0f + __expf(-lrow[expert]));
            prob[i] = p;
            sel[i]  = p + (bias ? bias[expert] : 0.0f);
        } else {
            prob[i] = 0.0f;
            sel[i]  = -INFINITY;
        }
    }

    select_topk_from_warp_probs<n_experts, ept, with_norm>(
        prob, sel, lane, n_expert_used, scale,
        weights_out + (size_t)row * n_expert_used,
        ids_out + (size_t)row * n_expert_used);
}

} // namespace ll_gate_topk

extern "C" void loken_gate_topk_softmax(
    const float * x_in,         // [n_rows, hidden] device ptr
    const float * gate_w,       // [n_experts, hidden] device ptr
    float       * weights_out,  // [n_rows, n_expert_used]
    uint32_t    * ids_out,      // [n_rows, n_expert_used]
    int hidden,
    int n_rows,
    int n_experts,
    int n_expert_used,
    int with_norm,
    cudaStream_t stream
) {
    if (hidden > MAX_HIDDEN_FOR_FUSED_GATE) {
        return; // Caller falls back to unfused path.
    }
    dim3 grid(n_rows, 1, 1);
    dim3 block(WARP_SIZE, 1, 1);
    int shared_bytes = hidden * (int)sizeof(float);

#define DISPATCH(NE) \
    do { \
        if (with_norm) ll_gate_topk::gate_topk_softmax_kernel<NE, true>  <<<grid, block, shared_bytes, stream>>>(x_in, gate_w, weights_out, ids_out, hidden, n_rows, n_expert_used); \
        else           ll_gate_topk::gate_topk_softmax_kernel<NE, false> <<<grid, block, shared_bytes, stream>>>(x_in, gate_w, weights_out, ids_out, hidden, n_rows, n_expert_used); \
    } while (0)

    switch (n_experts) {
        case 32:  DISPATCH(32);  break;
        case 64:  DISPATCH(64);  break;
        case 128: DISPATCH(128); break;
        case 256: DISPATCH(256); break;
        default:
            return; // Unsupported - caller falls back.
    }
#undef DISPATCH
}

// Route from logits a caller already has. Same signature as the kernel this replaces, so the
// Rust side is unchanged; 512 experts are dispatched because Qwen3-Next-80B needs them, and
// dropping that case silently would have cost that model its routing.
extern "C" void loken_topk_softmax(
    const float * logits,       // [n_rows, n_experts]
    float       * weights,      // [n_rows, n_expert_used]
    uint32_t    * ids,          // [n_rows, n_expert_used]
    int n_rows,
    int n_experts,
    int n_expert_used,
    int with_norm,
    cudaStream_t stream
) {
    dim3 grid(n_rows, 1, 1);
    dim3 block(WARP_SIZE, 1, 1);

#define DISPATCH_LOGITS(NE) \
    do { \
        if (with_norm) ll_gate_topk::topk_softmax_from_logits_kernel<NE, true>  <<<grid, block, 0, stream>>>(logits, weights, ids, n_rows, n_expert_used); \
        else           ll_gate_topk::topk_softmax_from_logits_kernel<NE, false> <<<grid, block, 0, stream>>>(logits, weights, ids, n_rows, n_expert_used); \
    } while (0)

    switch (n_experts) {
        case 32:  DISPATCH_LOGITS(32);  break;
        case 64:  DISPATCH_LOGITS(64);  break;
        case 128: DISPATCH_LOGITS(128); break;
        case 256: DISPATCH_LOGITS(256); break;
        case 512: DISPATCH_LOGITS(512); break;
        default:
            return; // Unsupported - the caller falls back to the unfused tensor ops.
    }
#undef DISPATCH_LOGITS
}

extern "C" void loken_gate_topk_sigmoid(
    const float * x_in,         // [n_rows, hidden]
    const float * gate_w,       // [n_experts, hidden]
    const float * bias,         // [n_experts] or nullptr
    float       * weights_out,  // [n_rows, n_expert_used]
    uint32_t    * ids_out,      // [n_rows, n_expert_used]
    int hidden,
    int n_rows,
    int n_experts,
    int n_expert_used,
    int with_norm,
    float scale,
    cudaStream_t stream
) {
    if (hidden > MAX_HIDDEN_FOR_FUSED_GATE) return;
    dim3 grid(n_rows, 1, 1);
    dim3 block(WARP_SIZE, 1, 1);
    int shared_bytes = hidden * (int)sizeof(float);

#define DISPATCH_SIG(NE) \
    do { \
        if (with_norm) ll_gate_topk::gate_topk_sigmoid_kernel<NE, true>  <<<grid, block, shared_bytes, stream>>>(x_in, gate_w, bias, weights_out, ids_out, hidden, n_rows, n_expert_used, scale); \
        else           ll_gate_topk::gate_topk_sigmoid_kernel<NE, false> <<<grid, block, shared_bytes, stream>>>(x_in, gate_w, bias, weights_out, ids_out, hidden, n_rows, n_expert_used, scale); \
    } while (0)

    switch (n_experts) {
        case 32:  DISPATCH_SIG(32);  break;
        case 64:  DISPATCH_SIG(64);  break;
        case 128: DISPATCH_SIG(128); break;
        case 256: DISPATCH_SIG(256); break;
        default:
            return;
    }
#undef DISPATCH_SIG
}

// Decode-path argsort for MoE expert grouping. Replaces the tensor-level sort_last_dim(true)
// on the flattened top-k expert ids (m = n_tok*topk <= 32 at decode -> one warp).
// Stable ascending rank-sort: produces ascending-sorted ids + the stable argsort
// permutation, EXACTLY matching sort_last_dim(true), so downstream is unchanged.
// A tensor-level sort is a large hidden cost on launch-bound MoE decode (pipeline stall).
extern "C" __global__ void loken_argsort_small_u32_kernel(
    const uint32_t * __restrict__ in,
    uint32_t       * __restrict__ sorted_out,
    uint32_t       * __restrict__ index_out,
    int m
) {
    const int lane = threadIdx.x;
    if (lane >= m) return;
    const uint32_t v = in[lane];
    int rank = 0;
    for (int j = 0; j < m; j++) {
        const uint32_t vj = in[j];
        // stable: ties broken by original position (matches a stable sort)
        if (vj < v || (vj == v && j < lane)) rank++;
    }
    sorted_out[rank] = v;
    index_out[rank]  = (uint32_t)lane;
}

extern "C" void loken_argsort_small_u32(
    const uint32_t * in, uint32_t * sorted_out, uint32_t * index_out, int m, cudaStream_t stream
) {
    if (m <= 0 || m > 32) return; // caller falls back
    loken_argsort_small_u32_kernel<<<1, 32, 0, stream>>>(in, sorted_out, index_out, m);
}

extern "C" void loken_topk_sigmoid_post(
    const float * logits_in,    // [n_rows, n_experts]
    const float * bias,         // [n_experts] or nullptr
    float       * weights_out,  // [n_rows, n_expert_used]
    uint32_t    * ids_out,      // [n_rows, n_expert_used]
    int n_rows,
    int n_experts,
    int n_expert_used,
    int with_norm,
    float scale,
    cudaStream_t stream
) {
    dim3 grid(n_rows, 1, 1);
    dim3 block(WARP_SIZE, 1, 1);

#define DISPATCH_POST(NE) \
    do { \
        if (with_norm) ll_gate_topk::topk_sigmoid_post_kernel<NE, true>  <<<grid, block, 0, stream>>>(logits_in, bias, weights_out, ids_out, n_rows, n_expert_used, scale); \
        else           ll_gate_topk::topk_sigmoid_post_kernel<NE, false> <<<grid, block, 0, stream>>>(logits_in, bias, weights_out, ids_out, n_rows, n_expert_used, scale); \
    } while (0)

    switch (n_experts) {
        case 32:  DISPATCH_POST(32);  break;
        case 64:  DISPATCH_POST(64);  break;
        case 128: DISPATCH_POST(128); break;
        case 256: DISPATCH_POST(256); break;
        default:
            return;
    }
#undef DISPATCH_POST
}
