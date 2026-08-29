/**
 * @file dense_q4k_imma_m8.cuh
 *
 * The dense Q4_K x Q8_1 tensor-core GEMM, said once.
 *
 * A block produces 16 weight rows x 8 consecutive tokens through
 * `mma.sync.m16n8k32.s8`, reading Q4_K weights and per-token Q8_1 activations.
 * Three dense entry points want that pass and differ only in what follows it:
 * gemma's gate+up with `gelu_pytorch_tanh`, mistral's gate+up with SiLU, and a
 * single-weight projection with no activation at all. So the pass takes the
 * epilogue as a parameter - the activation as a functor, and whether there is a
 * second weight to multiply by as a compile-time flag, which lets the up half
 * fold away entirely when there is none.
 *
 * Grid:
 *   gridDim.x = ceil(N / 16)
 *   gridDim.y = ceil(size_m / 8)
 * blockDim = 32
 */

// The block layouts come from `gguf.cuh`: this kernel reads the same bytes every
// other one does, and a layout restated here could stop saying so.
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define DENSE_Q4K_K_SUPER 256
#define DENSE_Q4K_M        16
#define DENSE_Q4K_N         8

/// No activation: the projection writes the accumulator as it stands.
struct dense_identity {
    static __device__ __forceinline__ float of(float g) { return g; }
};

/// SiLU(g) = g / (1 + exp(-g)) - the mistral-family SwiGLU gate.
struct dense_silu {
    static __device__ __forceinline__ float of(float g) { return g / (1.f + __expf(-g)); }
};

/// gemma's `gelu_pytorch_tanh`: PyTorch GELU(approximate="tanh"), llama.cpp's
/// `gelu_new`, and the HF transformers gemma activation are the same function.
///   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
struct dense_gelu_tanh {
    static __device__ __forceinline__ float of(float g) {
        const float k0 = 0.7978845608028654f; // sqrt(2/pi)
        const float k1 = 0.044715f;
        return 0.5f * g * (1.f + tanhf(k0 * (g + k1 * g * g * g)));
    }
};

/// `dst = Act(gate_w x x) * (up_w x x)`, or `dst = Act(gate_w x x)` when `WithUp` is false.
///
/// `inputs_q81` is `[size_m, K/32]` Q8_1, `gate_w` and `up_w` are `[N, K]` Q4_K, and `dst`
/// is `[size_m, N]` F32. `up_w` is read only when `WithUp` says there is one.
template <bool WithUp, typename Act>
static __device__ __forceinline__ void dense_q4k_imma_m8(
    const void * __restrict__ gate_w,
    const void * __restrict__ up_w,
    const void * __restrict__ inputs_q81,
    float * __restrict__ dst,
    int size_m,
    int N,
    int K
) {
    const int n_tile = blockIdx.y;
    const int tok_base = n_tile * DENSE_Q4K_N;
    if (tok_base >= size_m) return;

    const int m_tile = blockIdx.x;
    const int m_base = m_tile * DENSE_Q4K_M;
    if (m_base >= N) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / DENSE_Q4K_K_SUPER;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < N;
    const bool vb = row_b < N;

    const int my_tok = tok_base + g;
    const bool v_tok = my_tok < size_m;

    const block_q4_K * gate_base = (const block_q4_K *) gate_w;
    const block_q4_K * up_base   = WithUp ? (const block_q4_K *) up_w : nullptr;
    const block_q8_1 * y_base    = v_tok
        ? (const block_q8_1 *) inputs_q81 + (size_t)my_tok * num_super * 8
        : nullptr;

    float gate_0 = 0.f, gate_1 = 0.f, gate_2 = 0.f, gate_3 = 0.f;
    float up_0   = 0.f, up_1   = 0.f, up_2   = 0.f, up_3   = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q4_K * gwa = va ? gate_base + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K * gwb = vb ? gate_base + (size_t)row_b * num_super + isb : nullptr;
        const block_q4_K * uwa = (WithUp && va) ? up_base + (size_t)row_a * num_super + isb : nullptr;
        const block_q4_K * uwb = (WithUp && vb) ? up_base + (size_t)row_b * num_super + isb : nullptr;

        const float g_dall_a = gwa ? __low2float (gwa->dm) : 0.f;
        const float g_dmin_a = gwa ? __high2float(gwa->dm) : 0.f;
        const float g_dall_b = gwb ? __low2float (gwb->dm) : 0.f;
        const float g_dmin_b = gwb ? __high2float(gwb->dm) : 0.f;
        const float u_dall_a = uwa ? __low2float (uwa->dm) : 0.f;
        const float u_dmin_a = uwa ? __high2float(uwa->dm) : 0.f;
        const float u_dall_b = uwb ? __low2float (uwb->dm) : 0.f;
        const float u_dmin_b = uwb ? __high2float(uwb->dm) : 0.f;

        float g_da_a[8], g_dm_a[8], g_da_b[8], g_dm_b[8];
        float u_da_a[8], u_dm_a[8], u_da_b[8], u_dm_b[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc, m;
            if (gwa) { get_scale_min_k4(s, gwa->scales, sc, m); g_da_a[s] = g_dall_a * (float)sc; g_dm_a[s] = g_dmin_a * (float)m; }
            else     { g_da_a[s] = 0.f; g_dm_a[s] = 0.f; }
            if (gwb) { get_scale_min_k4(s, gwb->scales, sc, m); g_da_b[s] = g_dall_b * (float)sc; g_dm_b[s] = g_dmin_b * (float)m; }
            else     { g_da_b[s] = 0.f; g_dm_b[s] = 0.f; }
            if (uwa) { get_scale_min_k4(s, uwa->scales, sc, m); u_da_a[s] = u_dall_a * (float)sc; u_dm_a[s] = u_dmin_a * (float)m; }
            else     { u_da_a[s] = 0.f; u_dm_a[s] = 0.f; }
            if (uwb) { get_scale_min_k4(s, uwb->scales, sc, m); u_da_b[s] = u_dall_b * (float)sc; u_dm_b[s] = u_dmin_b * (float)m; }
            else     { u_da_b[s] = 0.f; u_dm_b[s] = 0.f; }
        }

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int qs_off = 32 * il + 8 * tj;

            uint2 gqa_pair = gwa ? __ldg((const uint2 *)(gwa->qs + qs_off)) : make_uint2(0, 0);
            uint2 gqb_pair = gwb ? __ldg((const uint2 *)(gwb->qs + qs_off)) : make_uint2(0, 0);
            int GA0, GA1, GA2, GA3;
            if (ip == 0) {
                GA0 = (int)(gqa_pair.x & 0x0F0F0F0F);
                GA2 = (int)(gqa_pair.y & 0x0F0F0F0F);
                GA1 = (int)(gqb_pair.x & 0x0F0F0F0F);
                GA3 = (int)(gqb_pair.y & 0x0F0F0F0F);
            } else {
                GA0 = (int)((gqa_pair.x >> 4) & 0x0F0F0F0F);
                GA2 = (int)((gqa_pair.y >> 4) & 0x0F0F0F0F);
                GA1 = (int)((gqb_pair.x >> 4) & 0x0F0F0F0F);
                GA3 = (int)((gqb_pair.y >> 4) & 0x0F0F0F0F);
            }

            const block_q8_1 * yb_my = y_base ? y_base + isb * 8 + s : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my    = yb_my ? __low2float (yb_my->ds) : 0.f;
            const float sumxd_my = yb_my ? __high2float(yb_my->ds) : 0.f;

            int GD0=0, GD1=0, GD2=0, GD3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(GD0), "+r"(GD1), "+r"(GD2), "+r"(GD3)
                : "r"(GA0), "r"(GA1), "r"(GA2), "r"(GA3), "r"(B0), "r"(B1));

            int UD0=0, UD1=0, UD2=0, UD3=0;
            if (WithUp) {
                uint2 uqa_pair = uwa ? __ldg((const uint2 *)(uwa->qs + qs_off)) : make_uint2(0, 0);
                uint2 uqb_pair = uwb ? __ldg((const uint2 *)(uwb->qs + qs_off)) : make_uint2(0, 0);
                int UA0, UA1, UA2, UA3;
                if (ip == 0) {
                    UA0 = (int)(uqa_pair.x & 0x0F0F0F0F);
                    UA2 = (int)(uqa_pair.y & 0x0F0F0F0F);
                    UA1 = (int)(uqb_pair.x & 0x0F0F0F0F);
                    UA3 = (int)(uqb_pair.y & 0x0F0F0F0F);
                } else {
                    UA0 = (int)((uqa_pair.x >> 4) & 0x0F0F0F0F);
                    UA2 = (int)((uqa_pair.y >> 4) & 0x0F0F0F0F);
                    UA1 = (int)((uqb_pair.x >> 4) & 0x0F0F0F0F);
                    UA3 = (int)((uqb_pair.y >> 4) & 0x0F0F0F0F);
                }
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+r"(UD0), "+r"(UD1), "+r"(UD2), "+r"(UD3)
                    : "r"(UA0), "r"(UA1), "r"(UA2), "r"(UA3), "r"(B0), "r"(B1));
            }

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_a    = __shfl_sync(0xffffffff, d8_my,    src_lane_a);
            const float d8_b    = __shfl_sync(0xffffffff, d8_my,    src_lane_b);
            const float sumxd_a = __shfl_sync(0xffffffff, sumxd_my, src_lane_a);
            const float sumxd_b = __shfl_sync(0xffffffff, sumxd_my, src_lane_b);

            const float gda_a = g_da_a[s], gdm_a = g_dm_a[s];
            const float gda_b = g_da_b[s], gdm_b = g_dm_b[s];

            gate_0 += gda_a * d8_a * (float)GD0 - gdm_a * sumxd_a;
            gate_1 += gda_a * d8_b * (float)GD1 - gdm_a * sumxd_b;
            gate_2 += gda_b * d8_a * (float)GD2 - gdm_b * sumxd_a;
            gate_3 += gda_b * d8_b * (float)GD3 - gdm_b * sumxd_b;

            if (WithUp) {
                const float uda_a = u_da_a[s], udm_a = u_dm_a[s];
                const float uda_b = u_da_b[s], udm_b = u_dm_b[s];
                up_0 += uda_a * d8_a * (float)UD0 - udm_a * sumxd_a;
                up_1 += uda_a * d8_b * (float)UD1 - udm_a * sumxd_b;
                up_2 += uda_b * d8_a * (float)UD2 - udm_b * sumxd_a;
                up_3 += uda_b * d8_b * (float)UD3 - udm_b * sumxd_b;
            }
        }
    }

    const int tok_a = tok_base + 2 * tj + 0;
    const int tok_b = tok_base + 2 * tj + 1;

    auto write = [&](int tok, int weight_row, float gv, float uv) {
        if (tok >= size_m || weight_row >= N) return;
        // Without a second weight there is nothing to multiply by, and `1.f` says so exactly.
        dst[(size_t)tok * N + weight_row] = Act::of(gv) * (WithUp ? uv : 1.f);
    };

    if (va) {
        write(tok_a, row_a, gate_0, up_0);
        write(tok_b, row_a, gate_1, up_1);
    }
    if (vb) {
        write(tok_a, row_b, gate_2, up_2);
        write(tok_b, row_b, gate_3, up_3);
    }
}

// -- The three entry points -------------------------------------------
//
// gemma3/4-family gate+up.gelu, the qwen2/qwen3/llama/mistral SwiGLU gate+up.silu,
// and the single-weight projection used where a dense QKV/O or FFN-down would
// otherwise fall back to dp4a MMVQ at prefill.

extern "C" __global__ void loken_dense_q4k_imma_m8_gelu_kernel(
    const void * __restrict__ gate_w,
    const void * __restrict__ up_w,
    const void * __restrict__ inputs_q81,
    float * __restrict__ dst,
    int size_m, int N, int K
) {
    dense_q4k_imma_m8<true, dense_gelu_tanh>(gate_w, up_w, inputs_q81, dst, size_m, N, K);
}

extern "C" __global__ void loken_dense_q4k_imma_m8_silu_kernel(
    const void * __restrict__ gate_w,
    const void * __restrict__ up_w,
    const void * __restrict__ inputs_q81,
    float * __restrict__ dst,
    int size_m, int N, int K
) {
    dense_q4k_imma_m8<true, dense_silu>(gate_w, up_w, inputs_q81, dst, size_m, N, K);
}

extern "C" __global__ void loken_dense_q4k_imma_m8_plain_kernel(
    const void * __restrict__ w,
    const void * __restrict__ inputs_q81,
    float * __restrict__ dst,
    int size_m, int N, int K
) {
    dense_q4k_imma_m8<false, dense_identity>(w, nullptr, inputs_q81, dst, size_m, N, K);
}

/// The grid every one of them runs on: a block per 16 weight rows x 8 tokens.
static dim3 dense_q4k_grid(int size_m, int N) {
    return dim3((N      + DENSE_Q4K_M - 1) / DENSE_Q4K_M,
                (size_m + DENSE_Q4K_N - 1) / DENSE_Q4K_N,
                1);
}

/// The gate+up pass, told which activation to close with - the q5_K sibling's convention, so
/// that a caller picks a format and not a spelling.
extern "C" void loken_dense_q4k_imma_m8_actmul(
    const void * gate_w, const void * up_w, const void * inputs_q81,
    float * dst_f32, int size_m, int N, int K, int use_gelu, cudaStream_t stream
) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    const dim3 grid = dense_q4k_grid(size_m, N);
    const dim3 blk(32, 1, 1);
    if (use_gelu) {
        loken_dense_q4k_imma_m8_gelu_kernel<<<grid, blk, 0, stream>>>(
            gate_w, up_w, inputs_q81, dst_f32, size_m, N, K);
    } else {
        loken_dense_q4k_imma_m8_silu_kernel<<<grid, blk, 0, stream>>>(
            gate_w, up_w, inputs_q81, dst_f32, size_m, N, K);
    }
}

extern "C" void loken_dense_q4k_imma_m8_plain(
    const void * w, const void * inputs_q81,
    float * dst_f32, int size_m, int N, int K, cudaStream_t stream
) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    loken_dense_q4k_imma_m8_plain_kernel<<<dense_q4k_grid(size_m, N), dim3(32, 1, 1), 0, stream>>>(
        w, inputs_q81, dst_f32, size_m, N, K);
}
