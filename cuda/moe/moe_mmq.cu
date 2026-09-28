/**
 * @brief The small steps around the expert-batched tiled matmul of a MoE prefill.
 *
 * The tiled matmul (`launch_mmq_moe_gguf_*`) multiplies every expert against its own run of
 * activation columns, the columns sorted by expert. What it needs from the routing, and what
 * turns its outputs into the next matmul's input and then into one row per token, lives here:
 *
 *   plan     where each expert's run starts, and which token every sorted row comes from,
 *            from a sorted routing or grouped here from the router's choices;
 *   glu      silu(gate) * up, or gpt-oss's clamped SwiGLU with per-expert biases, over the
 *            rows the gate and up projections wrote side by side;
 *   fold     each token's routed rows, plus their expert's down bias when there is one, scaled
 *            by their router weight and summed in slot order, so the sum is the same on every
 *            run.
 */
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int THREADS = 256;

int blocks_for(int64_t n) {
    return (int) ((n + THREADS - 1) / THREADS);
}

/// `bounds[e]` is the first sorted row whose expert is `e` or later, for `e` in
/// `[0, n_experts]`; `src_rows[i]` is the token sorted row `i` was built from.
__global__ void moe_mmq_plan_kernel(const int32_t *__restrict__ experts,
                                    const int32_t *__restrict__ slots, int n, int topk,
                                    int n_experts, int32_t *__restrict__ bounds,
                                    int32_t *__restrict__ slots_out,
                                    int32_t *__restrict__ src_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        slots_out[i] = slots[i];
        src_rows[i] = slots[i] / topk;
    }
    if (i <= n_experts) {
        int lo = 0;
        int hi = n;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (experts[mid] < i) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        bounds[i] = lo;
    }
}

/// How many of the `n` routed pairs chose each expert.
__global__ void moe_mmq_count_kernel(const int32_t *__restrict__ choices, int n, int n_experts,
                                     int32_t *__restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int e = choices[i];
        if (e >= 0 && e < n_experts) {
            atomicAdd(&counts[e], 1);
        }
    }
}

/// `bounds` as the running sum of `counts`, and `counts` reset to each run's start so the
/// placement below can claim positions from it. One thread: the expert count is small.
__global__ void moe_mmq_scan_kernel(int32_t *__restrict__ counts, int n_experts,
                                    int32_t *__restrict__ bounds) {
    int at = 0;
    for (int e = 0; e < n_experts; ++e) {
        const int c = counts[e];
        bounds[e] = at;
        counts[e] = at;
        at += c;
    }
    bounds[n_experts] = at;
}

/// Every pair claims the next free row of its expert's run.
__global__ void moe_mmq_place_kernel(const int32_t *__restrict__ choices, int n, int topk,
                                     int n_experts, int32_t *__restrict__ cursor,
                                     int32_t *__restrict__ slots,
                                     int32_t *__restrict__ src_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int e = choices[i];
        if (e >= 0 && e < n_experts) {
            const int row = atomicAdd(&cursor[e], 1);
            slots[row] = i;
            src_rows[row] = i / topk;
        }
    }
}

/// `out[r, c] = silu(gu[r, c]) * gu[r, width + c]` for a `[rows, 2 * width]` input.
__global__ void moe_mmq_silu_mul_kernel(const float *__restrict__ gu, float *__restrict__ out,
                                        int64_t rows, int width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) {
        return;
    }
    const int64_t r = i / width;
    const int64_t c = i % width;
    const float g = gu[r * 2 * width + c];
    const float u = gu[r * 2 * width + width + c];
    out[i] = g / (1.0f + __expf(-g)) * u;
}

/// The expert sorted row `row` belongs to: the last run starting at or before it.
__device__ __forceinline__ int expert_of_row(const int32_t *__restrict__ bounds, int n_experts,
                                             int row) {
    int lo = 0;
    int hi = n_experts;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (bounds[mid] <= row) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// gpt-oss's gate: `x = min(g + gb, limit)`, `u' = clamp(u + ub, -limit, limit)`,
/// `out = x * sigmoid(alpha * x) * (1 + u')`, the biases those of the row's expert.
__global__ void moe_mmq_swiglu_oai_kernel(const float *__restrict__ gu, float *__restrict__ out,
                                          int64_t rows, int width,
                                          const int32_t *__restrict__ bounds, int n_experts,
                                          const float *__restrict__ gate_bias,
                                          const float *__restrict__ up_bias, float alpha,
                                          float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) {
        return;
    }
    const int64_t r = i / width;
    const int64_t c = i % width;
    float g = gu[r * 2 * width + c];
    float u = gu[r * 2 * width + width + c];
    if (gate_bias != nullptr || up_bias != nullptr) {
        const int64_t e = expert_of_row(bounds, n_experts, (int) r);
        if (gate_bias != nullptr) g += gate_bias[e * width + c];
        if (up_bias != nullptr) u += up_bias[e * width + c];
    }
    const float x = fminf(g, limit);
    const float uc = fminf(fmaxf(u, -limit), limit);
    const float sig = 1.0f / (1.0f + __expf(-alpha * x));
    out[i] = (x * sig) * (1.0f + uc);
}

/// `out[t, c] += Σ_k weights[p] * (pairs[p, c] + down_bias[choices[p], c])` with
/// `p = t * topk + k`, k ascending; the bias term only when there is a bias.
__global__ void moe_mmq_fold_kernel(const float *__restrict__ pairs,
                                    const float *__restrict__ weights,
                                    const int32_t *__restrict__ choices,
                                    const float *__restrict__ down_bias,
                                    float *__restrict__ out, int64_t tokens, int topk,
                                    int cols) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * cols) {
        return;
    }
    const int64_t t = i / cols;
    const int64_t c = i % cols;
    float acc = out[i];
    for (int k = 0; k < topk; ++k) {
        const int64_t p = t * topk + k;
        float v = pairs[p * cols + c];
        if (down_bias != nullptr) {
            v += down_bias[(int64_t) choices[p] * cols + c];
        }
        acc += weights[p] * v;
    }
    out[i] = acc;
}

} // namespace

extern "C" void loken_moe_mmq_plan(const int32_t *experts, const int32_t *slots, int n, int topk,
                                   int n_experts, int32_t *bounds, int32_t *slots_out,
                                   int32_t *src_rows, cudaStream_t stream) {
    const int span = n > n_experts + 1 ? n : n_experts + 1;
    moe_mmq_plan_kernel<<<blocks_for(span), THREADS, 0, stream>>>(
        experts, slots, n, topk, n_experts, bounds, slots_out, src_rows);
}

/// Group `n` routed pairs by expert: `counts` is `n_experts` zeroed ints of scratch.
extern "C" void loken_moe_mmq_route(const int32_t *choices, int n, int topk, int n_experts,
                                    int32_t *counts, int32_t *bounds, int32_t *slots,
                                    int32_t *src_rows, cudaStream_t stream) {
    moe_mmq_count_kernel<<<blocks_for(n), THREADS, 0, stream>>>(choices, n, n_experts, counts);
    moe_mmq_scan_kernel<<<1, 1, 0, stream>>>(counts, n_experts, bounds);
    moe_mmq_place_kernel<<<blocks_for(n), THREADS, 0, stream>>>(choices, n, topk, n_experts,
                                                                counts, slots, src_rows);
}

extern "C" void loken_moe_mmq_silu_mul(const float *gu, float *out, int64_t rows, int width,
                                       cudaStream_t stream) {
    moe_mmq_silu_mul_kernel<<<blocks_for(rows * width), THREADS, 0, stream>>>(gu, out, rows,
                                                                              width);
}

extern "C" void loken_moe_mmq_swiglu_oai(const float *gu, float *out, int64_t rows, int width,
                                         const int32_t *bounds, int n_experts,
                                         const float *gate_bias, const float *up_bias,
                                         float alpha, float limit, cudaStream_t stream) {
    moe_mmq_swiglu_oai_kernel<<<blocks_for(rows * width), THREADS, 0, stream>>>(
        gu, out, rows, width, bounds, n_experts, gate_bias, up_bias, alpha, limit);
}

extern "C" void loken_moe_mmq_fold(const float *pairs, const float *weights,
                                   const int32_t *choices, const float *down_bias, float *out,
                                   int64_t tokens, int topk, int cols, cudaStream_t stream) {
    moe_mmq_fold_kernel<<<blocks_for(tokens * cols), THREADS, 0, stream>>>(
        pairs, weights, choices, down_bias, out, tokens, topk, cols);
}
