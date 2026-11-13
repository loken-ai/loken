// F32 dense matmul kernels for safetensors image models (Z-Image, etc.)
//
// Two modes:
//   GEMV (seq_len=1): Original reduction-based kernel - one WG per output element
//   GEMM (seq_len>1): Tiled kernel - 64x64 output block per WG, slides over K dim
//
// Computes: C[m][n] = sum_k( A[m][k] * B[n][k] )
//   where A = input [seq_len, in_dim]  (MxK)
//         B = weight [out_dim, in_dim]  (NxK, row-major)
//         C = output [seq_len, out_dim] (MxN)
//
// Note: B is stored [N, K] but we need B^T for standard GEMM. The tiled kernel
// reads B row-by-row (each row = one output channel), effectively transposing.

// ============================================================================
// GEMV kernels (seq_len=1 or small) - original design, good for single-token
// ============================================================================

#define F32_WG 64

__kernel void f32_matmul(
    __global const float* weight,   // [out_dim, in_dim]
    __global const float* input,    // [seq_len, in_dim]
    __global float* output,         // [seq_len, out_dim]
    const uint seq_len,
    const uint in_dim,
    const uint out_dim)
{
    uint wg_id = get_group_id(0);
    uint lid   = get_local_id(0);

    if (wg_id >= seq_len * out_dim) return;
    uint seq = wg_id / out_dim;
    uint row = wg_id % out_dim;

    __global const float* inp = input + seq * in_dim;
    __global const float* w_row = weight + (ulong)row * in_dim;

    uint vec_dim = in_dim >> 2;
    float partial = 0.0f;

    for (uint i = lid; i < vec_dim; i += F32_WG) {
        float4 w = vload4(i, w_row);
        float4 x = vload4(i, inp);
        partial += dot(w, x);
    }
    uint tail_start = vec_dim << 2;
    for (uint i = tail_start + lid; i < in_dim; i += F32_WG) {
        partial += w_row[i] * inp[i];
    }

    __local float scratch[F32_WG];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = F32_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[wg_id] = scratch[0];
    }
}

__kernel void f32_matmul_bias(
    __global const float* weight,
    __global const float* input,
    __global const float* bias,
    __global float* output,
    const uint seq_len,
    const uint in_dim,
    const uint out_dim)
{
    uint wg_id = get_group_id(0);
    uint lid   = get_local_id(0);

    if (wg_id >= seq_len * out_dim) return;
    uint seq = wg_id / out_dim;
    uint row = wg_id % out_dim;

    __global const float* inp = input + seq * in_dim;
    __global const float* w_row = weight + (ulong)row * in_dim;

    uint vec_dim = in_dim >> 2;
    float partial = 0.0f;

    for (uint i = lid; i < vec_dim; i += F32_WG) {
        float4 w = vload4(i, w_row);
        float4 x = vload4(i, inp);
        partial += dot(w, x);
    }
    uint tail_start = vec_dim << 2;
    for (uint i = tail_start + lid; i < in_dim; i += F32_WG) {
        partial += w_row[i] * inp[i];
    }

    __local float scratch[F32_WG];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = F32_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[wg_id] = scratch[0] + bias[row];
    }
}

// ============================================================================
// Tiled GEMM kernels (seq_len > 1)
// ============================================================================
//
// C[M,N] = A[M,K] x B^T[N,K]
//
// A work-group of GEMM_TS x GEMM_TS work-items computes a block of GEMM_BT x GEMM_BT outputs.
// Each work-item holds GEMM_RB x GEMM_RB outputs in registers.
// Work-item (lc, lr) owns rows lr + GEMM_TS.i and columns lc + GEMM_TS.j.
// The K dimension is walked in slices of GEMM_TS.
// Each slice is staged in local memory with k as the outer index.
// Every output sums its products in increasing k, one fused multiply-add after another.
//
// Dispatch: local = (GEMM_TS, GEMM_TS).
// Dispatch: global = (ceil(N / GEMM_BT).GEMM_TS, ceil(M / GEMM_BT).GEMM_TS).
// Dimension 0 runs along the columns of C, dimension 1 along its rows.

#define GEMM_TS 16
#define GEMM_RB 4
#define GEMM_BT (GEMM_TS * GEMM_RB)
#define GEMM_BT_PAD (GEMM_BT + 1)

// Stages rows [base, base + GEMM_BT) by k [k0, k0 + GEMM_TS) of a row-major [rows, K]
// matrix into tile[k][row].
// Entries past the edges are zero.
#define GEMM_STAGE_TILE(tile, src, base, rows, k0, K, lc, lr)                    \
    for (uint s = 0; s < GEMM_RB; s++) {                                         \
        uint r = (lr) + GEMM_TS * s;                                             \
        uint gr = (base) + r;                                                    \
        uint gk = (k0) + (lc);                                                   \
        tile[(lc)][r] = (gr < (rows) && gk < (K)) ? (src)[(ulong)gr * (K) + gk] : 0.0f; \
    }

__kernel void f32_tiled_matmul(
    __global const float* A,        // [M, K] = input [seq_len, in_dim]
    __global const float* B,        // [N, K] = weight [out_dim, in_dim] (row-major)
    __global float* C,              // [M, N] = output [seq_len, out_dim]
    const uint M,                   // seq_len
    const uint N,                   // out_dim
    const uint K)                   // in_dim
{
    uint lc = get_local_id(0);
    uint lr = get_local_id(1);
    uint col0 = get_group_id(0) * GEMM_BT;
    uint row0 = get_group_id(1) * GEMM_BT;

    __local float tileA[GEMM_TS][GEMM_BT_PAD];
    __local float tileB[GEMM_TS][GEMM_BT_PAD];

    float acc[GEMM_RB][GEMM_RB];
    for (uint i = 0; i < GEMM_RB; i++)
        for (uint j = 0; j < GEMM_RB; j++)
            acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += GEMM_TS) {
        GEMM_STAGE_TILE(tileA, A, row0, M, k0, K, lc, lr)
        GEMM_STAGE_TILE(tileB, B, col0, N, k0, K, lc, lr)
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < GEMM_TS; k++) {
            float a[GEMM_RB];
            float b[GEMM_RB];
            for (uint i = 0; i < GEMM_RB; i++) a[i] = tileA[k][lr + GEMM_TS * i];
            for (uint j = 0; j < GEMM_RB; j++) b[j] = tileB[k][lc + GEMM_TS * j];
            for (uint i = 0; i < GEMM_RB; i++)
                for (uint j = 0; j < GEMM_RB; j++)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = 0; i < GEMM_RB; i++) {
        uint gr = row0 + lr + GEMM_TS * i;
        if (gr >= M) continue;
        for (uint j = 0; j < GEMM_RB; j++) {
            uint gc = col0 + lc + GEMM_TS * j;
            if (gc < N) C[(ulong)gr * N + gc] = acc[i][j];
        }
    }
}

// Tiled fused gate_up_silu: C = silu(A x gate^T) * (A x up^T)
// Two products sharing the same input, fused with the activation.
// Same tiling and dispatch as f32_tiled_matmul.
__kernel void f32_tiled_gate_up_silu(
    __global const float* gate_weight,  // [N, K]
    __global const float* up_weight,    // [N, K]
    __global const float* A,            // [M, K] = input
    __global float* act,                // [M, N] = silu(gate) * up
    const uint M,
    const uint N,
    const uint K)
{
    uint lc = get_local_id(0);
    uint lr = get_local_id(1);
    uint col0 = get_group_id(0) * GEMM_BT;
    uint row0 = get_group_id(1) * GEMM_BT;

    __local float tileA[GEMM_TS][GEMM_BT_PAD];
    __local float tileG[GEMM_TS][GEMM_BT_PAD];
    __local float tileU[GEMM_TS][GEMM_BT_PAD];

    float acc_g[GEMM_RB][GEMM_RB];
    float acc_u[GEMM_RB][GEMM_RB];
    for (uint i = 0; i < GEMM_RB; i++)
        for (uint j = 0; j < GEMM_RB; j++) {
            acc_g[i][j] = 0.0f;
            acc_u[i][j] = 0.0f;
        }

    for (uint k0 = 0; k0 < K; k0 += GEMM_TS) {
        GEMM_STAGE_TILE(tileA, A, row0, M, k0, K, lc, lr)
        GEMM_STAGE_TILE(tileG, gate_weight, col0, N, k0, K, lc, lr)
        GEMM_STAGE_TILE(tileU, up_weight, col0, N, k0, K, lc, lr)
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < GEMM_TS; k++) {
            float a[GEMM_RB];
            float g[GEMM_RB];
            float u[GEMM_RB];
            for (uint i = 0; i < GEMM_RB; i++) a[i] = tileA[k][lr + GEMM_TS * i];
            for (uint j = 0; j < GEMM_RB; j++) {
                g[j] = tileG[k][lc + GEMM_TS * j];
                u[j] = tileU[k][lc + GEMM_TS * j];
            }
            for (uint i = 0; i < GEMM_RB; i++)
                for (uint j = 0; j < GEMM_RB; j++) {
                    acc_g[i][j] = fma(a[i], g[j], acc_g[i][j]);
                    acc_u[i][j] = fma(a[i], u[j], acc_u[i][j]);
                }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = 0; i < GEMM_RB; i++) {
        uint gr = row0 + lr + GEMM_TS * i;
        if (gr >= M) continue;
        for (uint j = 0; j < GEMM_RB; j++) {
            uint gc = col0 + lc + GEMM_TS * j;
            if (gc < N) {
                float gate = acc_g[i][j];
                act[(ulong)gr * N + gc] = gate / (1.0f + exp(-gate)) * acc_u[i][j];
            }
        }
    }
}

// ============================================================================
// Original fused kernels kept for GEMV path (seq_len=1)
// ============================================================================

__kernel void f32_gate_up_silu(
    __global const float* gate_weight,
    __global const float* up_weight,
    __global const float* input,
    __global float* act,
    const uint seq_len,
    const uint in_dim,
    const uint out_dim)
{
    uint wg_id = get_group_id(0);
    uint lid   = get_local_id(0);

    if (wg_id >= seq_len * out_dim) return;
    uint seq = wg_id / out_dim;
    uint row = wg_id % out_dim;

    __global const float* inp = input + seq * in_dim;
    __global const float* g_row = gate_weight + (ulong)row * in_dim;
    __global const float* u_row = up_weight   + (ulong)row * in_dim;

    uint vec_dim = in_dim >> 2;
    float partial_gate = 0.0f;
    float partial_up   = 0.0f;

    for (uint i = lid; i < vec_dim; i += F32_WG) {
        float4 x = vload4(i, inp);
        float4 gw = vload4(i, g_row);
        float4 uw = vload4(i, u_row);
        partial_gate += dot(gw, x);
        partial_up   += dot(uw, x);
    }
    uint tail_start = vec_dim << 2;
    for (uint i = tail_start + lid; i < in_dim; i += F32_WG) {
        float x = inp[i];
        partial_gate += g_row[i] * x;
        partial_up   += u_row[i] * x;
    }

    __local float2 scratch2[F32_WG];
    scratch2[lid] = (float2)(partial_gate, partial_up);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = F32_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch2[lid] += scratch2[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float gate_val = scratch2[0].x;
        float up_val   = scratch2[0].y;
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        act[wg_id] = silu_gate * up_val;
    }
}

__kernel void f32_matmul_add(
    __global const float* weight,
    __global const float* input,
    __global const float* residual,
    __global float* output,
    const uint seq_len,
    const uint in_dim,
    const uint out_dim)
{
    uint wg_id = get_group_id(0);
    uint lid   = get_local_id(0);

    if (wg_id >= seq_len * out_dim) return;
    uint seq = wg_id / out_dim;
    uint row = wg_id % out_dim;

    __global const float* inp = input + seq * in_dim;
    __global const float* w_row = weight + (ulong)row * in_dim;

    uint vec_dim = in_dim >> 2;
    float partial = 0.0f;

    for (uint i = lid; i < vec_dim; i += F32_WG) {
        float4 w = vload4(i, w_row);
        float4 x = vload4(i, inp);
        partial += dot(w, x);
    }
    uint tail_start = vec_dim << 2;
    for (uint i = tail_start + lid; i < in_dim; i += F32_WG) {
        partial += w_row[i] * inp[i];
    }

    __local float scratch[F32_WG];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = F32_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        uint out_idx = seq * out_dim + row;
        output[out_idx] = scratch[0] + residual[out_idx];
    }
}
