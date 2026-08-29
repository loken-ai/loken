/**
 * @file dense_q5k_imma_m8.cu
 *
 * IMMA M=8 INT8-tensor-core GEMM for DENSE Q5_K weights - the Q5_K sibling of
 * dense_q4k_imma_m8_{silu,gelu} + the plain single-weight GEMM. The fleet
 * (qwen3 / deepcoder / gemma4 / lfm2) is Q5_K_M, so the Q4_K prefill-IMMA path
 * fires on nothing; this makes it fire.
 *
 * Q5_K block = Q4_K block + a `qh[QK_K/8=32]` 5th-bit plane inserted between the
 * 12-byte packed scales and the 128-byte qs nibbles. The dequant value is the
 * 4-bit qs nibble OR the matching qh 5th bit (shifted to bit 4) -> 0..31, which
 * still fits the s8 MMA operand. Bit mapping (verified against llama.cpp
 * `dequantize_row_q5_K` / `BlockQ5K::to_float`):
 *   for qs-chunk c = il = s>>1 (0..3), nibble-half ip = s&1 (0=low,1=high):
 *       5th bit = qh[l] bit (2*c + ip),   l = within-chunk byte index.
 * The DSL kernel's GA0/GA2 (block a) and GA1/GA3 (block b) come from qs bytes
 * `32*il + 8*tj + {0..3}` (lo) and `{4..7}` (hi); their within-chunk index is
 * `l = 8*tj + {0..3}` / `{4..7}` -> qh bytes at `qh + 8*tj` (lo half in .x,
 * hi half in .y). Same layout the qs read uses, so the OR is lane-aligned.
 *
 * Everything else (super-block scale/min via get_scale_min_k4, the -dmin*sumx
 * epilogue, the m16n8k32 s8 MMA, the Q8_1 activation) is byte-identical to the
 * Q4_K kernels.
 *
 * Grid: gridDim.x = ceil(N/16), gridDim.y = ceil(size_m/8), blockDim = 32.
 */
#include "gguf.cuh"
// The block layouts come from `gguf.cuh`, included above: this kernel reads the same bytes
// every other one does, and a layout restated here could stop saying so.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace loken_dense_q5k_imma_m8_ns {

#define DSL_K_SUPER 256
#define DSL_M       16
#define DSL_N        8

// Load the 4+4 s8 weight operands (5-bit, qs nibble | qh 5th bit) for one
// (s = 2*il+ip) step of one Q5_K super-block. `w == nullptr` -> zeros.
//   out_lo = 4 weights from qs bytes qs_off+{0..3}  (qh bytes 8*tj+{0..3})
//   out_hi = 4 weights from qs bytes qs_off+{4..7}  (qh bytes 8*tj+{4..7})
__device__ __forceinline__ void dsl_q5k_load(
    const block_q5_K * w, int qs_off, int tj, int ip, int bit,
    int & out_lo, int & out_hi
) {
    if (!w) { out_lo = 0; out_hi = 0; return; }
    uint2 qpair = __ldg((const uint2 *)(w->qs + qs_off));
    uint2 hpair = __ldg((const uint2 *)(w->qh + 8 * tj));
    uint32_t q_lo = qpair.x, q_hi = qpair.y;
    uint32_t h_lo = hpair.x, h_hi = hpair.y;
    int lo, hi;
    if (ip == 0) {
        lo = (int)(q_lo & 0x0F0F0F0F);
        hi = (int)(q_hi & 0x0F0F0F0F);
    } else {
        lo = (int)((q_lo >> 4) & 0x0F0F0F0F);
        hi = (int)((q_hi >> 4) & 0x0F0F0F0F);
    }
    // OR the 5th bit (bit `bit` of each qh byte) into bit 4 of each lane.
    lo |= (int)(((h_lo >> bit) << 4) & 0x10101010u);
    hi |= (int)(((h_hi >> bit) << 4) & 0x10101010u);
    out_lo = lo;
    out_hi = hi;
}

} // namespace

// -- PLAIN single-weight GEMM (out = x . Wᵀ, identity epilogue) ----------------
extern "C" __global__ void loken_dense_q5k_imma_m8_plain_kernel(
    const void * __restrict__ w,                 // [N, K] Q5_K
    const void * __restrict__ inputs_q81,        // [size_m, K/32] Q8_1
    float * __restrict__ dst,                    // [size_m, N] F32
    int size_m, int N, int K
) {
    using namespace loken_dense_q5k_imma_m8_ns;
    const int n_tile = blockIdx.y;
    const int tok_base = n_tile * DSL_N;
    if (tok_base >= size_m) return;
    const int m_tile = blockIdx.x;
    const int m_base = m_tile * DSL_M;
    if (m_base >= N) return;
    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / DSL_K_SUPER;
    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < N;
    const bool vb = row_b < N;
    const int my_tok = tok_base + g;
    const bool v_tok = my_tok < size_m;
    const block_q5_K * w_base = (const block_q5_K *) w;
    const block_q8_1 * y_base = v_tok
        ? (const block_q8_1 *) inputs_q81 + (size_t)my_tok * num_super * 8
        : nullptr;
    float acc_0 = 0.f, acc_1 = 0.f, acc_2 = 0.f, acc_3 = 0.f;
    for (int isb = 0; isb < num_super; ++isb) {
        const block_q5_K * gwa = va ? w_base + (size_t)row_a * num_super + isb : nullptr;
        const block_q5_K * gwb = vb ? w_base + (size_t)row_b * num_super + isb : nullptr;
        const float g_dall_a = gwa ? __low2float (gwa->dm) : 0.f;
        const float g_dmin_a = gwa ? __high2float(gwa->dm) : 0.f;
        const float g_dall_b = gwb ? __low2float (gwb->dm) : 0.f;
        const float g_dmin_b = gwb ? __high2float(gwb->dm) : 0.f;
        float g_da_a[8], g_dm_a[8], g_da_b[8], g_dm_b[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc, m;
            if (gwa) { get_scale_min_k4(s, gwa->scales, sc, m); g_da_a[s] = g_dall_a * (float)sc; g_dm_a[s] = g_dmin_a * (float)m; }
            else     { g_da_a[s] = 0.f; g_dm_a[s] = 0.f; }
            if (gwb) { get_scale_min_k4(s, gwb->scales, sc, m); g_da_b[s] = g_dall_b * (float)sc; g_dm_b[s] = g_dmin_b * (float)m; }
            else     { g_da_b[s] = 0.f; g_dm_b[s] = 0.f; }
        }
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int il = s >> 1;
            const int ip = s & 1;
            const int bit = 2 * il + ip;
            const int qs_off = 32 * il + 8 * tj;
            int GA0, GA2, GA1, GA3;
            dsl_q5k_load(gwa, qs_off, tj, ip, bit, GA0, GA2);
            dsl_q5k_load(gwb, qs_off, tj, ip, bit, GA1, GA3);
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
            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_a    = __shfl_sync(0xffffffff, d8_my,    src_lane_a);
            const float d8_b    = __shfl_sync(0xffffffff, d8_my,    src_lane_b);
            const float sumxd_a = __shfl_sync(0xffffffff, sumxd_my, src_lane_a);
            const float sumxd_b = __shfl_sync(0xffffffff, sumxd_my, src_lane_b);
            const float gda_a = g_da_a[s], gdm_a = g_dm_a[s];
            const float gda_b = g_da_b[s], gdm_b = g_dm_b[s];
            acc_0 += gda_a * d8_a * (float)GD0 - gdm_a * sumxd_a;
            acc_1 += gda_a * d8_b * (float)GD1 - gdm_a * sumxd_b;
            acc_2 += gda_b * d8_a * (float)GD2 - gdm_b * sumxd_a;
            acc_3 += gda_b * d8_b * (float)GD3 - gdm_b * sumxd_b;
        }
    }
    const int tok_a = tok_base + 2 * tj + 0;
    const int tok_b = tok_base + 2 * tj + 1;
    auto plain_write = [&](int tok, int weight_row, float v) {
        if (tok >= size_m || weight_row >= N) return;
        dst[(size_t)tok * N + weight_row] = v;
    };
    if (va) { plain_write(tok_a, row_a, acc_0); plain_write(tok_b, row_a, acc_1); }
    if (vb) { plain_write(tok_a, row_b, acc_2); plain_write(tok_b, row_b, acc_3); }
}

// -- FUSED gate+up with silu(gate)*up or gelu(gate)*up -------------------------
// `use_gelu`: 0 -> SiLU, 1 -> gemma gelu_pytorch_tanh.
extern "C" __global__ void loken_dense_q5k_imma_m8_actmul_kernel(
    const void * __restrict__ gate_w,            // [N, K] Q5_K
    const void * __restrict__ up_w,              // [N, K] Q5_K
    const void * __restrict__ inputs_q81,        // [size_m, K/32] Q8_1
    float * __restrict__ dst,                    // [size_m, N] F32
    int size_m, int N, int K, int use_gelu
) {
    using namespace loken_dense_q5k_imma_m8_ns;
    const int n_tile = blockIdx.y;
    const int tok_base = n_tile * DSL_N;
    if (tok_base >= size_m) return;
    const int m_tile = blockIdx.x;
    const int m_base = m_tile * DSL_M;
    if (m_base >= N) return;
    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int tj   = lane & 3;
    const int num_super = K / DSL_K_SUPER;
    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool va = row_a < N;
    const bool vb = row_b < N;
    const int my_tok = tok_base + g;
    const bool v_tok = my_tok < size_m;
    const block_q5_K * gate_base = (const block_q5_K *) gate_w;
    const block_q5_K * up_base   = (const block_q5_K *) up_w;
    const block_q8_1 * y_base    = v_tok
        ? (const block_q8_1 *) inputs_q81 + (size_t)my_tok * num_super * 8
        : nullptr;

    float gate_0 = 0.f, gate_1 = 0.f, gate_2 = 0.f, gate_3 = 0.f;
    float up_0   = 0.f, up_1   = 0.f, up_2   = 0.f, up_3   = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        const block_q5_K * gwa = va ? gate_base + (size_t)row_a * num_super + isb : nullptr;
        const block_q5_K * gwb = vb ? gate_base + (size_t)row_b * num_super + isb : nullptr;
        const block_q5_K * uwa = va ? up_base   + (size_t)row_a * num_super + isb : nullptr;
        const block_q5_K * uwb = vb ? up_base   + (size_t)row_b * num_super + isb : nullptr;

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
            const int bit = 2 * il + ip;
            const int qs_off = 32 * il + 8 * tj;

            int GA0, GA2, GA1, GA3;
            dsl_q5k_load(gwa, qs_off, tj, ip, bit, GA0, GA2);
            dsl_q5k_load(gwb, qs_off, tj, ip, bit, GA1, GA3);
            int UA0, UA2, UA1, UA3;
            dsl_q5k_load(uwa, qs_off, tj, ip, bit, UA0, UA2);
            dsl_q5k_load(uwb, qs_off, tj, ip, bit, UA1, UA3);

            const block_q8_1 * yb_my = y_base ? y_base + isb * 8 + s : nullptr;
            int B0 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 0] : 0;
            int B1 = yb_my ? ((const int *)yb_my->qs)[2 * tj + 1] : 0;
            const float d8_my    = yb_my ? __low2float (yb_my->ds) : 0.f;
            const float sumxd_my = yb_my ? __high2float(yb_my->ds) : 0.f;

            int GD0=0, GD1=0, GD2=0, GD3=0, UD0=0, UD1=0, UD2=0, UD3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(GD0), "+r"(GD1), "+r"(GD2), "+r"(GD3)
                : "r"(GA0), "r"(GA1), "r"(GA2), "r"(GA3), "r"(B0), "r"(B1));
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(UD0), "+r"(UD1), "+r"(UD2), "+r"(UD3)
                : "r"(UA0), "r"(UA1), "r"(UA2), "r"(UA3), "r"(B0), "r"(B1));

            const int src_lane_a = (2 * tj + 0) * 4;
            const int src_lane_b = (2 * tj + 1) * 4;
            const float d8_a    = __shfl_sync(0xffffffff, d8_my,    src_lane_a);
            const float d8_b    = __shfl_sync(0xffffffff, d8_my,    src_lane_b);
            const float sumxd_a = __shfl_sync(0xffffffff, sumxd_my, src_lane_a);
            const float sumxd_b = __shfl_sync(0xffffffff, sumxd_my, src_lane_b);

            const float gda_a = g_da_a[s], gdm_a = g_dm_a[s];
            const float gda_b = g_da_b[s], gdm_b = g_dm_b[s];
            const float uda_a = u_da_a[s], udm_a = u_dm_a[s];
            const float uda_b = u_da_b[s], udm_b = u_dm_b[s];

            gate_0 += gda_a * d8_a * (float)GD0 - gdm_a * sumxd_a;
            gate_1 += gda_a * d8_b * (float)GD1 - gdm_a * sumxd_b;
            gate_2 += gda_b * d8_a * (float)GD2 - gdm_b * sumxd_a;
            gate_3 += gda_b * d8_b * (float)GD3 - gdm_b * sumxd_b;
            up_0   += uda_a * d8_a * (float)UD0 - udm_a * sumxd_a;
            up_1   += uda_a * d8_b * (float)UD1 - udm_a * sumxd_b;
            up_2   += uda_b * d8_a * (float)UD2 - udm_b * sumxd_a;
            up_3   += uda_b * d8_b * (float)UD3 - udm_b * sumxd_b;
        }
    }

    const int tok_a = tok_base + 2 * tj + 0;
    const int tok_b = tok_base + 2 * tj + 1;

    auto act_mul_write = [&](int tok, int weight_row, float gv, float uv) {
        if (tok >= size_m || weight_row >= N) return;
        float act;
        if (use_gelu) {
            // gemma gelu_pytorch_tanh (matches dense_q4k gelu kernel byte-for-byte)
            const float k0 = 0.7978845608028654f; // sqrt(2/pi)
            const float k1 = 0.044715f;
            const float inner = k0 * (gv + k1 * gv * gv * gv);
            act = 0.5f * gv * (1.f + tanhf(inner));
        } else {
            act = gv / (1.f + __expf(-gv)); // SiLU
        }
        dst[(size_t)tok * N + weight_row] = act * uv;
    };

    if (va) {
        act_mul_write(tok_a, row_a, gate_0, up_0);
        act_mul_write(tok_b, row_a, gate_1, up_1);
    }
    if (vb) {
        act_mul_write(tok_a, row_b, gate_2, up_2);
        act_mul_write(tok_b, row_b, gate_3, up_3);
    }
}

extern "C" void loken_dense_q5k_imma_m8_plain(
    const void * w, const void * inputs_q81, float * dst_f32,
    int size_m, int N, int K, cudaStream_t stream
) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    using namespace loken_dense_q5k_imma_m8_ns;
    dim3 grid((N + DSL_M - 1) / DSL_M, (size_m + DSL_N - 1) / DSL_N, 1);
    dim3 blk(32, 1, 1);
    loken_dense_q5k_imma_m8_plain_kernel<<<grid, blk, 0, stream>>>(
        w, inputs_q81, dst_f32, size_m, N, K);
}

extern "C" void loken_dense_q5k_imma_m8_actmul(
    const void * gate_w, const void * up_w, const void * inputs_q81, float * dst_f32,
    int size_m, int N, int K, int use_gelu, cudaStream_t stream
) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    using namespace loken_dense_q5k_imma_m8_ns;
    dim3 grid((N + DSL_M - 1) / DSL_M, (size_m + DSL_N - 1) / DSL_N, 1);
    dim3 blk(32, 1, 1);
    loken_dense_q5k_imma_m8_actmul_kernel<<<grid, blk, 0, stream>>>(
        gate_w, up_w, inputs_q81, dst_f32, size_m, N, K, use_gelu);
}
