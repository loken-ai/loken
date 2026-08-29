// Q4_K dequantize + GEMV kernel - sub-block level parallelism
//
// Q4_K block format (144 bytes per 256 elements):
//   2 bytes: f16 d (global scale)
//   2 bytes: f16 dmin (global min scale)
//   12 bytes: packed 6-bit scales and mins for 8 sub-blocks
//   128 bytes: packed 4-bit quants (256 values, 2 per byte)
//
// Each 256-element block has 8 sub-blocks of 32 elements:
//   Sub 0: lower nibbles of qs[0..31],  scale idx 0
//   Sub 1: upper nibbles of qs[0..31],  scale idx 1
//   Sub 2: lower nibbles of qs[32..63], scale idx 2
//   Sub 3: upper nibbles of qs[32..63], scale idx 3
//   Sub 4: lower nibbles of qs[64..95], scale idx 4
//   Sub 5: upper nibbles of qs[64..95], scale idx 5
//   Sub 6: lower nibbles of qs[96..127], scale idx 6
//   Sub 7: upper nibbles of qs[96..127], scale idx 7
//
// Dequantization: value = d * scale_i * nibble - dmin * min_i
//
// Dispatch: global_work_size = seq_len * out_dim * WG_SIZE
//           local_work_size  = WG_SIZE

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64
#define Q4K_BLOCK_BYTES 144

// Extract 6-bit scale and min from packed scales array
// j: sub-block index (0..7)
// sc: pointer to 12-byte scales array
inline float2 get_scale_min(uint j, __global const uchar* sc) {
    float scale, mn;
    if (j < 4) {
        scale = (float)(sc[j] & 63);
        mn = (float)(sc[j + 4] & 63);
    } else {
        scale = (float)((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
        mn = (float)((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
    return (float2)(scale, mn);
}

__kernel void q4k_matmul(
    __global const uchar* q4k_bytes,
    __global const float* input,
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

    // Number of 32-element sub-blocks across input dimension
    uint n_sub_blocks = in_dim >> 5;
    // Number of Q4K blocks per row
    uint n_q4k_blocks = in_dim >> 8;

    __global const float* inp = input + seq * in_dim;
    __global const uchar* w_row = q4k_bytes + (ulong)row * n_q4k_blocks * Q4K_BLOCK_BYTES;

    float partial = 0.0f;

    for (uint sb = lid; sb < n_sub_blocks; sb += WG_SIZE) {
        uint blk_idx   = sb >> 3;       // Q4K block index
        uint local_sub = sb & 7;        // sub-block within Q4K block (0..7)
        uint group     = local_sub >> 1; // group within block (0..3)
        uint is_upper  = local_sub & 1;  // 0=lower nibble, 1=upper nibble

        __global const uchar* blk = w_row + blk_idx * Q4K_BLOCK_BYTES;

        // Read global scales (f16)
        float d    = vload_half(0, (__global const half*)blk);
        float dmin = vload_half(1, (__global const half*)blk);

        // Extract per-sub-block scale and min (6-bit each)
        float2 sm = get_scale_min(local_sub, blk + 4);
        float d_eff = d * sm.x;      // effective scale
        float m_eff = dmin * sm.y;    // effective min

        // Quant bytes for this group (32 bytes covering 64 elements)
        __global const uchar* qs = blk + 16 + group * 32;

        // Input offset for this sub-block's 32 elements
        uint in_base = (blk_idx << 8) + (local_sub << 5);

        // Load 32 quant bytes as 2x vload16
        uchar16 q0 = vload16(0, qs);
        uchar16 q1 = vload16(1, qs);

        // Compute sum(nibble * input) and sum(input) for 32 elements
        // Branchless: use shift to select upper (>>4) or lower (&0xF) nibbles
        // value = d_eff * nibble - m_eff, so:
        //   partial += d_eff * sum(nibble * input) - m_eff * sum(input)
        uint shift = is_upper << 2;  // 0 for lower, 4 for upper
        float sum_nq = 0.0f;
        float sum_inp = 0.0f;

        #define NIB(byte) ((float)(((byte) >> shift) & 0xF))

        float4 i0 = vload4(0, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.s0), NIB(q0.s1), NIB(q0.s2), NIB(q0.s3)), i0);
        sum_inp += dot((float4)(1.0f), i0);

        float4 i1 = vload4(1, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.s4), NIB(q0.s5), NIB(q0.s6), NIB(q0.s7)), i1);
        sum_inp += dot((float4)(1.0f), i1);

        float4 i2 = vload4(2, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.s8), NIB(q0.s9), NIB(q0.sa), NIB(q0.sb)), i2);
        sum_inp += dot((float4)(1.0f), i2);

        float4 i3 = vload4(3, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.sc), NIB(q0.sd), NIB(q0.se), NIB(q0.sf)), i3);
        sum_inp += dot((float4)(1.0f), i3);

        float4 i4 = vload4(4, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.s0), NIB(q1.s1), NIB(q1.s2), NIB(q1.s3)), i4);
        sum_inp += dot((float4)(1.0f), i4);

        float4 i5 = vload4(5, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.s4), NIB(q1.s5), NIB(q1.s6), NIB(q1.s7)), i5);
        sum_inp += dot((float4)(1.0f), i5);

        float4 i6 = vload4(6, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.s8), NIB(q1.s9), NIB(q1.sa), NIB(q1.sb)), i6);
        sum_inp += dot((float4)(1.0f), i6);

        float4 i7 = vload4(7, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.sc), NIB(q1.sd), NIB(q1.se), NIB(q1.sf)), i7);
        sum_inp += dot((float4)(1.0f), i7);

        #undef NIB

        partial += d_eff * sum_nq - m_eff * sum_inp;
    }

    // Tree reduction in local memory
    __local float scratch[WG_SIZE];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[wg_id] = scratch[0];
    }
}

// Fused gate_up_silu kernel: reads input once, computes both gate and up projections,
// applies SiLU(gate) * up inline. Saves ~50% input bandwidth + eliminates 2 kernel launches.
//
// Dispatch: global_work_size = seq_len * out_dim * WG_SIZE
//           local_work_size  = WG_SIZE
// Output: act[seq * out_dim + row] = silu(gate_dot) * up_dot
__kernel void q4k_gate_up_silu(
    __global const uchar* gate_bytes,   // Q4_K gate weights
    __global const uchar* up_bytes,     // Q4_K up weights
    __global const float* input,
    __global float* act,                // fused output: silu(gate) * up
    const uint seq_len,
    const uint in_dim,
    const uint out_dim)
{
    uint wg_id = get_group_id(0);
    uint lid   = get_local_id(0);

    if (wg_id >= seq_len * out_dim) return;
    uint seq = wg_id / out_dim;
    uint row = wg_id % out_dim;

    uint n_sub_blocks = in_dim >> 5;
    uint n_q4k_blocks = in_dim >> 8;

    __global const float* inp = input + seq * in_dim;
    __global const uchar* gate_row = gate_bytes + (ulong)row * n_q4k_blocks * Q4K_BLOCK_BYTES;
    __global const uchar* up_row   = up_bytes   + (ulong)row * n_q4k_blocks * Q4K_BLOCK_BYTES;

    float partial_gate = 0.0f;
    float partial_up   = 0.0f;

    for (uint sb = lid; sb < n_sub_blocks; sb += WG_SIZE) {
        uint blk_idx   = sb >> 3;
        uint local_sub = sb & 7;
        uint group     = local_sub >> 1;
        uint is_upper  = local_sub & 1;

        // Gate block
        __global const uchar* g_blk = gate_row + blk_idx * Q4K_BLOCK_BYTES;
        float g_d    = vload_half(0, (__global const half*)g_blk);
        float g_dmin = vload_half(1, (__global const half*)g_blk);
        float2 g_sm  = get_scale_min(local_sub, g_blk + 4);
        float g_d_eff = g_d * g_sm.x;
        float g_m_eff = g_dmin * g_sm.y;
        __global const uchar* g_qs = g_blk + 16 + group * 32;
        uchar16 gq0 = vload16(0, g_qs);
        uchar16 gq1 = vload16(1, g_qs);

        // Up block
        __global const uchar* u_blk = up_row + blk_idx * Q4K_BLOCK_BYTES;
        float u_d    = vload_half(0, (__global const half*)u_blk);
        float u_dmin = vload_half(1, (__global const half*)u_blk);
        float2 u_sm  = get_scale_min(local_sub, u_blk + 4);
        float u_d_eff = u_d * u_sm.x;
        float u_m_eff = u_dmin * u_sm.y;
        __global const uchar* u_qs = u_blk + 16 + group * 32;
        uchar16 uq0 = vload16(0, u_qs);
        uchar16 uq1 = vload16(1, u_qs);

        uint in_base = (blk_idx << 8) + (local_sub << 5);

        // Branchless nibble extraction for both gate and up
        uint shift = is_upper << 2;  // 0 for lower, 4 for upper
        float g_sum_nq = 0.0f;
        float u_sum_nq = 0.0f;
        float sum_inp  = 0.0f;

        #define GNIB(byte) ((float)(((byte) >> shift) & 0xF))
        #define UNIB(byte) ((float)(((byte) >> shift) & 0xF))

        float4 i0 = vload4(0, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq0.s0),GNIB(gq0.s1),GNIB(gq0.s2),GNIB(gq0.s3)), i0);
        u_sum_nq += dot((float4)(UNIB(uq0.s0),UNIB(uq0.s1),UNIB(uq0.s2),UNIB(uq0.s3)), i0);
        sum_inp += dot((float4)(1.0f), i0);

        float4 i1 = vload4(1, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq0.s4),GNIB(gq0.s5),GNIB(gq0.s6),GNIB(gq0.s7)), i1);
        u_sum_nq += dot((float4)(UNIB(uq0.s4),UNIB(uq0.s5),UNIB(uq0.s6),UNIB(uq0.s7)), i1);
        sum_inp += dot((float4)(1.0f), i1);

        float4 i2 = vload4(2, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq0.s8),GNIB(gq0.s9),GNIB(gq0.sa),GNIB(gq0.sb)), i2);
        u_sum_nq += dot((float4)(UNIB(uq0.s8),UNIB(uq0.s9),UNIB(uq0.sa),UNIB(uq0.sb)), i2);
        sum_inp += dot((float4)(1.0f), i2);

        float4 i3 = vload4(3, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq0.sc),GNIB(gq0.sd),GNIB(gq0.se),GNIB(gq0.sf)), i3);
        u_sum_nq += dot((float4)(UNIB(uq0.sc),UNIB(uq0.sd),UNIB(uq0.se),UNIB(uq0.sf)), i3);
        sum_inp += dot((float4)(1.0f), i3);

        float4 i4 = vload4(4, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq1.s0),GNIB(gq1.s1),GNIB(gq1.s2),GNIB(gq1.s3)), i4);
        u_sum_nq += dot((float4)(UNIB(uq1.s0),UNIB(uq1.s1),UNIB(uq1.s2),UNIB(uq1.s3)), i4);
        sum_inp += dot((float4)(1.0f), i4);

        float4 i5 = vload4(5, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq1.s4),GNIB(gq1.s5),GNIB(gq1.s6),GNIB(gq1.s7)), i5);
        u_sum_nq += dot((float4)(UNIB(uq1.s4),UNIB(uq1.s5),UNIB(uq1.s6),UNIB(uq1.s7)), i5);
        sum_inp += dot((float4)(1.0f), i5);

        float4 i6 = vload4(6, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq1.s8),GNIB(gq1.s9),GNIB(gq1.sa),GNIB(gq1.sb)), i6);
        u_sum_nq += dot((float4)(UNIB(uq1.s8),UNIB(uq1.s9),UNIB(uq1.sa),UNIB(uq1.sb)), i6);
        sum_inp += dot((float4)(1.0f), i6);

        float4 i7 = vload4(7, inp + in_base);
        g_sum_nq += dot((float4)(GNIB(gq1.sc),GNIB(gq1.sd),GNIB(gq1.se),GNIB(gq1.sf)), i7);
        u_sum_nq += dot((float4)(UNIB(uq1.sc),UNIB(uq1.sd),UNIB(uq1.se),UNIB(uq1.sf)), i7);
        sum_inp += dot((float4)(1.0f), i7);

        #undef GNIB
        #undef UNIB

        partial_gate += g_d_eff * g_sum_nq - g_m_eff * sum_inp;
        partial_up   += u_d_eff * u_sum_nq - u_m_eff * sum_inp;
    }

    // Tree reduction for both gate and up (use float2 to reduce both simultaneously)
    __local float2 scratch2[WG_SIZE];
    scratch2[lid] = (float2)(partial_gate, partial_up);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch2[lid] += scratch2[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float gate_val = scratch2[0].x;
        float up_val   = scratch2[0].y;
        // SiLU(gate) * up = gate * sigmoid(gate) * up
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        act[wg_id] = silu_gate * up_val;
    }
}

// Fused matmul + residual add: output[i] = q4k_matmul(input, weights)[i] + residual[i]
// Eliminates separate add kernel and one full read/write of hidden state.
//
// Dispatch: global_work_size = seq_len * out_dim * WG_SIZE
//           local_work_size  = WG_SIZE
__kernel void q4k_matmul_add(
    __global const uchar* q4k_bytes,
    __global const float* input,
    __global const float* residual,  // added to matmul output
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

    uint n_sub_blocks = in_dim >> 5;
    uint n_q4k_blocks = in_dim >> 8;

    __global const float* inp = input + seq * in_dim;
    __global const uchar* w_row = q4k_bytes + (ulong)row * n_q4k_blocks * Q4K_BLOCK_BYTES;

    float partial = 0.0f;

    for (uint sb = lid; sb < n_sub_blocks; sb += WG_SIZE) {
        uint blk_idx   = sb >> 3;
        uint local_sub = sb & 7;
        uint group     = local_sub >> 1;
        uint is_upper  = local_sub & 1;

        __global const uchar* blk = w_row + blk_idx * Q4K_BLOCK_BYTES;

        float d    = vload_half(0, (__global const half*)blk);
        float dmin = vload_half(1, (__global const half*)blk);

        float2 sm = get_scale_min(local_sub, blk + 4);
        float d_eff = d * sm.x;
        float m_eff = dmin * sm.y;

        __global const uchar* qs = blk + 16 + group * 32;
        uint in_base = (blk_idx << 8) + (local_sub << 5);

        uchar16 q0 = vload16(0, qs);
        uchar16 q1 = vload16(1, qs);

        uint shift = is_upper << 2;
        float sum_nq = 0.0f;
        float sum_inp = 0.0f;

        #define NIB(byte) ((float)(((byte) >> shift) & 0xF))

        float4 i0 = vload4(0, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.s0), NIB(q0.s1), NIB(q0.s2), NIB(q0.s3)), i0);
        sum_inp += dot((float4)(1.0f), i0);

        float4 i1 = vload4(1, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.s4), NIB(q0.s5), NIB(q0.s6), NIB(q0.s7)), i1);
        sum_inp += dot((float4)(1.0f), i1);

        float4 i2 = vload4(2, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.s8), NIB(q0.s9), NIB(q0.sa), NIB(q0.sb)), i2);
        sum_inp += dot((float4)(1.0f), i2);

        float4 i3 = vload4(3, inp + in_base);
        sum_nq += dot((float4)(NIB(q0.sc), NIB(q0.sd), NIB(q0.se), NIB(q0.sf)), i3);
        sum_inp += dot((float4)(1.0f), i3);

        float4 i4 = vload4(4, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.s0), NIB(q1.s1), NIB(q1.s2), NIB(q1.s3)), i4);
        sum_inp += dot((float4)(1.0f), i4);

        float4 i5 = vload4(5, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.s4), NIB(q1.s5), NIB(q1.s6), NIB(q1.s7)), i5);
        sum_inp += dot((float4)(1.0f), i5);

        float4 i6 = vload4(6, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.s8), NIB(q1.s9), NIB(q1.sa), NIB(q1.sb)), i6);
        sum_inp += dot((float4)(1.0f), i6);

        float4 i7 = vload4(7, inp + in_base);
        sum_nq += dot((float4)(NIB(q1.sc), NIB(q1.sd), NIB(q1.se), NIB(q1.sf)), i7);
        sum_inp += dot((float4)(1.0f), i7);

        #undef NIB

        partial += d_eff * sum_nq - m_eff * sum_inp;
    }

    __local float scratch[WG_SIZE];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        // Fused: matmul result + residual
        uint out_idx = seq * out_dim + row;
        output[out_idx] = scratch[0] + residual[out_idx];
    }
}
