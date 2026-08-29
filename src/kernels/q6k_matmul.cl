// Q6_K dequantize + GEMV kernel
//
// Q6_K block format (210 bytes per 256 elements):
//   Offset 0:   ql[128] - lower 4 bits of 6-bit quants (2 nibbles per byte)
//   Offset 128: qh[64]  - upper 2 bits of 6-bit quants (4 x 2-bit per byte)
//   Offset 192: scales[16] - signed int8, one per 16-element sub-block
//   Offset 208: d (f16)  - global scale
//
// 6-bit reconstruction (per 128-element chunk, 2 chunks per block):
//   For l in 0..31, within chunk c (ql=ql[c*64..], qh=qh[c*32..]):
//     elem[l]      = (ql[l]    & 0xF) | ((qh[l]     & 0x03) << 4) - 32
//     elem[l+32]   = (ql[l+32] & 0xF) | (((qh[l]>>2) & 0x03) << 4) - 32
//     elem[l+64]   = (ql[l]    >> 4)  | (((qh[l]>>4) & 0x03) << 4) - 32
//     elem[l+96]   = (ql[l+32] >> 4)  | (((qh[l]>>6) & 0x03) << 4) - 32
//
// Dequantization: value = d * scale[sub_block] * q6_value
//
// Dispatch: global_work_size = seq_len * out_dim * WG_SIZE
//           local_work_size  = WG_SIZE

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define Q6K_WG 64
#define Q6K_BLOCK_BYTES 210

// Extract 16 dequantized Q6_K values, dot with 16 input floats.
// Returns d * scale * sum(q6_val * input).
inline float q6k_sub_dot(
    __global const uchar* blk,    // pointer to start of Q6K block
    __global const float* inp,    // pointer to 16 input floats
    uint local_sb)                // sub-block index within block (0..15)
{
    uint chunk = local_sb >> 3;          // 0 or 1
    uint within = local_sb & 7;
    uint group = within >> 1;            // 0..3
    uint hf = within & 1;               // 0 or 1 (first/second 16 elements)

    // ql and qh pointers within this chunk
    __global const uchar* ql = blk + chunk * 64 + (group & 1) * 32 + hf * 16;
    __global const uchar* qh = blk + 128 + chunk * 32 + hf * 16;

    uint use_upper = (group >= 2) ? 1 : 0;
    uint qh_shift = group * 2;

    // Global scale (f16)
    float d = vload_half(0, (__global const half*)(blk + 208));

    // Per-sub-block scale (signed int8)
    float scale = (float)((char)(blk[192 + local_sb]));

    // Extract 16 6-bit values and dot with input
    // Process in groups of 4 for vectorization
    float dot_sum = 0.0f;

    #define Q6_EXTRACT(idx) \
        ((float)((int)(((use_upper ? (ql[(idx)] >> 4) : (ql[(idx)] & 0xF)) | \
                 (((qh[(idx)] >> qh_shift) & 3) << 4))) - 32))

    float4 i0 = vload4(0, inp);
    float4 w0 = (float4)(Q6_EXTRACT(0), Q6_EXTRACT(1), Q6_EXTRACT(2), Q6_EXTRACT(3));
    dot_sum += dot(w0, i0);

    float4 i1 = vload4(1, inp);
    float4 w1 = (float4)(Q6_EXTRACT(4), Q6_EXTRACT(5), Q6_EXTRACT(6), Q6_EXTRACT(7));
    dot_sum += dot(w1, i1);

    float4 i2 = vload4(2, inp);
    float4 w2 = (float4)(Q6_EXTRACT(8), Q6_EXTRACT(9), Q6_EXTRACT(10), Q6_EXTRACT(11));
    dot_sum += dot(w2, i2);

    float4 i3 = vload4(3, inp);
    float4 w3 = (float4)(Q6_EXTRACT(12), Q6_EXTRACT(13), Q6_EXTRACT(14), Q6_EXTRACT(15));
    dot_sum += dot(w3, i3);

    #undef Q6_EXTRACT

    return d * scale * dot_sum;
}

__kernel void q6k_matmul(
    __global const uchar* q6k_bytes,
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

    // 16 sub-blocks per Q6K block, 16 elements per sub-block
    uint n_sub_blocks = in_dim >> 4;      // in_dim / 16
    uint n_q6k_blocks = in_dim >> 8;      // in_dim / 256

    __global const float* inp = input + seq * in_dim;
    __global const uchar* w_row = q6k_bytes + (ulong)row * n_q6k_blocks * Q6K_BLOCK_BYTES;

    float partial = 0.0f;

    for (uint sb = lid; sb < n_sub_blocks; sb += Q6K_WG) {
        uint blk_idx   = sb >> 4;      // which 256-element block
        uint local_sb  = sb & 15;      // sub-block within block (0..15)

        __global const uchar* blk = w_row + blk_idx * Q6K_BLOCK_BYTES;
        __global const float* sub_inp = inp + sb * 16;

        partial += q6k_sub_dot(blk, sub_inp, local_sb);
    }

    // Tree reduction in local memory
    __local float scratch[Q6K_WG];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = Q6K_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[wg_id] = scratch[0];
    }
}
