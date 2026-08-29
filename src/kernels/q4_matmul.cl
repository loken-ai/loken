// Q4_0 dequantize + GEMV kernel - row-per-workgroup reduction (optimized v2)
//
// Q4_0 block format (18 bytes per 32 elements):
//   2 bytes: f16 scale
//   16 bytes: packed nibbles (32 4-bit values, 2 per byte)
//
// Nibble layout: qs[n] for n in 0..16
//   lower nibble -> element n (0..15)
//   upper nibble -> element n+16 (16..31)
//
// Dequantization: value = (nibble - 8) * scale
//
// Dispatch: global_work_size = seq_len * out_dim * WG_SIZE
//           local_work_size  = WG_SIZE
//           One workgroup per output element.
//
// v2 optimizations:
//   - Vectorized vload16 for 16 quant bytes (1 wide load vs 16 byte loads)
//   - Vectorized vload4 + dot() for input (4 floats per transaction)
//   - Scale factored out: 1 multiply per block instead of 32
//   - WG_SIZE=64: fewer idle threads (n_blocks=160 -> 2-3 blocks/thread vs <1)
//   - Shallower reduction tree (6 levels vs 8)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel void q4_matmul(
    __global const uchar* q4_bytes,
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

    uint n_blocks = in_dim >> 5;  // in_dim / 32

    __global const float* inp = input + seq * in_dim;
    __global const uchar* w_row = q4_bytes + (ulong)row * n_blocks * 18;

    float partial = 0.0f;

    for (uint b = lid; b < n_blocks; b += WG_SIZE) {
        __global const uchar* blk = w_row + b * 18;

        // f16 scale (hardware-accelerated on Intel Arc)
        float scale = vload_half(0, (__global const half*)blk);

        // Load all 16 quant bytes in one vectorized read
        uchar16 qs = vload16(0, blk + 2);

        uint in_base = b << 5;  // b * 32

        // Process all 32 elements: vectorized input loads + dot products
        // Scale factored out - multiply once at end instead of 32 times
        float block_sum = 0.0f;

        // Lower nibbles -> elements 0..15 (4 dot products of float4)
        float4 i0 = vload4(0, inp + in_base);
        float4 w0 = (float4)((float)(qs.s0 & 0xF) - 8.0f, (float)(qs.s1 & 0xF) - 8.0f,
                              (float)(qs.s2 & 0xF) - 8.0f, (float)(qs.s3 & 0xF) - 8.0f);
        block_sum += dot(w0, i0);

        float4 i1 = vload4(1, inp + in_base);
        float4 w1 = (float4)((float)(qs.s4 & 0xF) - 8.0f, (float)(qs.s5 & 0xF) - 8.0f,
                              (float)(qs.s6 & 0xF) - 8.0f, (float)(qs.s7 & 0xF) - 8.0f);
        block_sum += dot(w1, i1);

        float4 i2 = vload4(2, inp + in_base);
        float4 w2 = (float4)((float)(qs.s8 & 0xF) - 8.0f, (float)(qs.s9 & 0xF) - 8.0f,
                              (float)(qs.sa & 0xF) - 8.0f, (float)(qs.sb & 0xF) - 8.0f);
        block_sum += dot(w2, i2);

        float4 i3 = vload4(3, inp + in_base);
        float4 w3 = (float4)((float)(qs.sc & 0xF) - 8.0f, (float)(qs.sd & 0xF) - 8.0f,
                              (float)(qs.se & 0xF) - 8.0f, (float)(qs.sf & 0xF) - 8.0f);
        block_sum += dot(w3, i3);

        // Upper nibbles -> elements 16..31 (4 dot products of float4)
        float4 i4 = vload4(4, inp + in_base);
        float4 w4 = (float4)((float)(qs.s0 >> 4) - 8.0f, (float)(qs.s1 >> 4) - 8.0f,
                              (float)(qs.s2 >> 4) - 8.0f, (float)(qs.s3 >> 4) - 8.0f);
        block_sum += dot(w4, i4);

        float4 i5 = vload4(5, inp + in_base);
        float4 w5 = (float4)((float)(qs.s4 >> 4) - 8.0f, (float)(qs.s5 >> 4) - 8.0f,
                              (float)(qs.s6 >> 4) - 8.0f, (float)(qs.s7 >> 4) - 8.0f);
        block_sum += dot(w5, i5);

        float4 i6 = vload4(6, inp + in_base);
        float4 w6 = (float4)((float)(qs.s8 >> 4) - 8.0f, (float)(qs.s9 >> 4) - 8.0f,
                              (float)(qs.sa >> 4) - 8.0f, (float)(qs.sb >> 4) - 8.0f);
        block_sum += dot(w6, i6);

        float4 i7 = vload4(7, inp + in_base);
        float4 w7 = (float4)((float)(qs.sc >> 4) - 8.0f, (float)(qs.sd >> 4) - 8.0f,
                              (float)(qs.se >> 4) - 8.0f, (float)(qs.sf >> 4) - 8.0f);
        block_sum += dot(w7, i7);

        partial += block_sum * scale;
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
