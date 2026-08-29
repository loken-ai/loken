// F32 dense matmul kernels for safetensors image models (Z-Image, etc.)
//
// Two modes:
//   GEMV (seq_len=1): Original reduction-based kernel - one WG per output element
//   GEMM (seq_len>1): Tiled kernel - 16x16 output tile per WG, slides over K dim
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
// Tiled GEMM kernels (seq_len > 1) - 5-10x faster for image generation
// ============================================================================
//
// C[M,N] = A[M,K] x B^T[N,K]
// Tile size: TSxTS output elements per work-group
// Work-group: TSxTS threads, each computes one output element
// K dimension split into tiles of width TS, loaded into shared memory
//
// For Z-Image: M=1024 (seq), N=3840 (out_dim), K=3840 (in_dim)
//   -> 64x240 work-groups instead of 3.9M with GEMV
//
// Dispatch: global_work_size = (ceil(M/TS)*TS, ceil(N/TS)*TS)
//           local_work_size  = (TS, TS)

#define TS 16   // Tile size: 16x16 = 256 threads per WG, fits in 16KB local mem

__kernel void f32_tiled_matmul(
    __global const float* A,        // [M, K] = input [seq_len, in_dim]
    __global const float* B,        // [N, K] = weight [out_dim, in_dim] (row-major)
    __global float* C,              // [M, N] = output [seq_len, out_dim]
    const uint M,                   // seq_len
    const uint N,                   // out_dim
    const uint K)                   // in_dim
{
    uint row = get_local_id(0);     // thread row within tile [0..TS)
    uint col = get_local_id(1);     // thread col within tile [0..TS)
    uint gRow = get_group_id(0) * TS + row;  // global output row
    uint gCol = get_group_id(1) * TS + col;  // global output col

    __local float tileA[TS][TS];    // shared A tile
    __local float tileB[TS][TS];    // shared B^T tile

    float acc = 0.0f;

    uint numTiles = (K + TS - 1) / TS;

    for (uint t = 0; t < numTiles; t++) {
        // Load A tile: A[gRow, t*TS + col]
        uint aCol = t * TS + col;
        if (gRow < M && aCol < K) {
            tileA[row][col] = A[(ulong)gRow * K + aCol];
        } else {
            tileA[row][col] = 0.0f;
        }

        // Load B^T tile: B[gCol, t*TS + row] -> tileB[row][col] = B[gCol][t*TS+row]
        // Note: B is [N,K] row-major. We want B^T[K,N], i.e. B^T[k][n] = B[n][k]
        // tileB stores a TSxTS block of B^T: tileB[row][col] = B^T[t*TS+row, gCol]
        //                                                     = B[gCol, t*TS+row]
        uint bRow = t * TS + row;
        if (gCol < N && bRow < K) {
            tileB[row][col] = B[(ulong)gCol * K + bRow];
        } else {
            tileB[row][col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate: acc += tileA[row][k] * tileB[k][col]
        for (uint k = 0; k < TS; k++) {
            acc += tileA[row][k] * tileB[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gRow < M && gCol < N) {
        C[(ulong)gRow * N + gCol] = acc;
    }
}

// Tiled GEMM with bias
__kernel void f32_tiled_matmul_bias(
    __global const float* A,        // [M, K]
    __global const float* B,        // [N, K]
    __global const float* bias,     // [N]
    __global float* C,              // [M, N]
    const uint M,
    const uint N,
    const uint K)
{
    uint row = get_local_id(0);
    uint col = get_local_id(1);
    uint gRow = get_group_id(0) * TS + row;
    uint gCol = get_group_id(1) * TS + col;

    __local float tileA[TS][TS];
    __local float tileB[TS][TS];

    float acc = 0.0f;
    uint numTiles = (K + TS - 1) / TS;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TS + col;
        tileA[row][col] = (gRow < M && aCol < K) ? A[(ulong)gRow * K + aCol] : 0.0f;

        uint bRow = t * TS + row;
        tileB[row][col] = (gCol < N && bRow < K) ? B[(ulong)gCol * K + bRow] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < TS; k++) {
            acc += tileA[row][k] * tileB[k][col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gRow < M && gCol < N) {
        C[(ulong)gRow * N + gCol] = acc + bias[gCol];
    }
}

// Tiled GEMM + residual add: C = A x B^T + residual
__kernel void f32_tiled_matmul_add(
    __global const float* A,
    __global const float* B,
    __global const float* residual,
    __global float* C,
    const uint M,
    const uint N,
    const uint K)
{
    uint row = get_local_id(0);
    uint col = get_local_id(1);
    uint gRow = get_group_id(0) * TS + row;
    uint gCol = get_group_id(1) * TS + col;

    __local float tileA[TS][TS];
    __local float tileB[TS][TS];

    float acc = 0.0f;
    uint numTiles = (K + TS - 1) / TS;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TS + col;
        tileA[row][col] = (gRow < M && aCol < K) ? A[(ulong)gRow * K + aCol] : 0.0f;

        uint bRow = t * TS + row;
        tileB[row][col] = (gCol < N && bRow < K) ? B[(ulong)gCol * K + bRow] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < TS; k++) {
            acc += tileA[row][k] * tileB[k][col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gRow < M && gCol < N) {
        ulong idx = (ulong)gRow * N + gCol;
        C[idx] = acc + residual[idx];
    }
}

// Tiled fused gate_up_silu: C = silu(A x gate^T) * (A x up^T)
// Two matmuls sharing the same input, fused with activation.
// Each thread computes one (m,n) position for both gate and up projections.
//
// Dispatch: same as tiled_matmul - global = (ceil(M/TS)*TS, ceil(N/TS)*TS)
__kernel void f32_tiled_gate_up_silu(
    __global const float* gate_weight,  // [N, K]
    __global const float* up_weight,    // [N, K]
    __global const float* A,            // [M, K] = input
    __global float* act,                // [M, N] = silu(gate) * up
    const uint M,
    const uint N,
    const uint K)
{
    uint row = get_local_id(0);
    uint col = get_local_id(1);
    uint gRow = get_group_id(0) * TS + row;
    uint gCol = get_group_id(1) * TS + col;

    __local float tileA[TS][TS];
    __local float tileG[TS][TS];  // gate weight tile
    __local float tileU[TS][TS];  // up weight tile

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    uint numTiles = (K + TS - 1) / TS;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TS + col;
        tileA[row][col] = (gRow < M && aCol < K) ? A[(ulong)gRow * K + aCol] : 0.0f;

        uint bRow = t * TS + row;
        float gv = (gCol < N && bRow < K) ? gate_weight[(ulong)gCol * K + bRow] : 0.0f;
        float uv = (gCol < N && bRow < K) ? up_weight[(ulong)gCol * K + bRow] : 0.0f;
        tileG[row][col] = gv;
        tileU[row][col] = uv;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < TS; k++) {
            float a = tileA[row][k];
            acc_gate += a * tileG[k][col];
            acc_up   += a * tileU[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gRow < M && gCol < N) {
        float silu_gate = acc_gate / (1.0f + exp(-acc_gate));
        act[(ulong)gRow * N + gCol] = silu_gate * acc_up;
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
