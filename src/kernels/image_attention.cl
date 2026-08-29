// Bidirectional scaled dot-product attention for image generation models
// NO causal mask, NO KV cache - all tokens attend to all tokens.
// Supports optional padding mask and Grouped Query Attention (GQA).
//
// Uses online softmax for numerical stability without materializing full score matrix.
//
// Input layout (flattened batch into dispatch):
//   Q: [total_heads, seq_len, head_dim]  where total_heads = batch * n_head
//   K: [total_kv_heads, seq_len, head_dim]
//   V: [total_kv_heads, seq_len, head_dim]
//
// mask: optional [batch, seq_len] - 1 for valid, 0 for padding
//
// Dispatch: global_work_size = batch * n_head * seq_len
//           One work item per (batch, head, query_position)
//
// Optimization: Q vector cached in private registers, float4 vectorized dot products.

__kernel void image_attention(
    __global const float* q,        // [batch * n_head, seq_len, head_dim]
    __global const float* k,        // [batch * n_kv_head, seq_len, head_dim]
    __global const float* v,        // [batch * n_kv_head, seq_len, head_dim]
    __global const float* mask,     // [batch, seq_len] or NULL (use mask_present flag)
    __global float* output,         // [batch * n_head, seq_len, head_dim]
    const uint batch,
    const uint n_head,
    const uint n_kv_head,
    const uint seq_len,
    const uint head_dim,
    const uint mask_present)        // 1 if mask is valid, 0 otherwise
{
    uint gid = get_global_id(0);
    if (gid >= batch * n_head * seq_len) return;

    uint q_pos = gid % seq_len;
    uint tmp = gid / seq_len;
    uint h = tmp % n_head;
    uint b = tmp / n_head;

    // GQA: map query head to KV head
    uint kv_h = (h * n_kv_head) / n_head;

    float scale = 1.0f / sqrt((float)head_dim);

    // Pointers into Q, K, V for this (batch, head)
    __global const float* q_ptr = q + ((ulong)(b * n_head + h) * seq_len + q_pos) * head_dim;
    __global const float* k_base = k + (ulong)(b * n_kv_head + kv_h) * seq_len * head_dim;
    __global const float* v_base = v + (ulong)(b * n_kv_head + kv_h) * seq_len * head_dim;

    // Mask for this batch (if present)
    __global const float* mask_ptr = mask_present ? (mask + b * seq_len) : 0;

    // Cache Q vector in private registers (head_dim=128 = 32 float4s)
    // This avoids re-reading Q from global memory for every key position
    float4 q_cache[32]; // supports up to head_dim=128
    uint vec_dim = head_dim >> 2;
    for (uint d = 0; d < vec_dim && d < 32; d++) {
        q_cache[d] = vload4(d, q_ptr);
    }

    // Online softmax: single pass over K/V positions
    float max_score = -3.402823466e+38f;
    float sum_exp = 0.0f;

    // V accumulator - head_dim=128 = 32 float4s
    float4 v_acc[32];
    for (uint d = 0; d < vec_dim && d < 32; d++) v_acc[d] = (float4)(0.0f);

    for (uint kp = 0; kp < seq_len; kp++) {
        // Check mask
        if (mask_present && mask_ptr[kp] == 0.0f) continue;

        __global const float* k_ptr = k_base + kp * head_dim;

        // Vectorized Q.K dot product using cached Q
        float score = 0.0f;
        for (uint d = 0; d < vec_dim && d < 32; d++) {
            float4 kv = vload4(d, k_ptr);
            score += dot(q_cache[d], kv);
        }
        score *= scale;

        // Online softmax update
        if (score > max_score) {
            float correction = exp(max_score - score);
            for (uint d = 0; d < vec_dim && d < 32; d++) {
                v_acc[d] *= correction;
            }
            sum_exp *= correction;
            max_score = score;
        }

        float w = exp(score - max_score);
        sum_exp += w;

        __global const float* v_ptr = v_base + kp * head_dim;
        for (uint d = 0; d < vec_dim && d < 32; d++) {
            v_acc[d] += w * vload4(d, v_ptr);
        }
    }

    // Normalize and write output
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    __global float* out_ptr = output + ((ulong)(b * n_head + h) * seq_len + q_pos) * head_dim;
    for (uint d = 0; d < vec_dim && d < 32; d++) {
        vstore4(v_acc[d] * inv_sum, d, out_ptr);
    }
}

// 3D RoPE application for Z-Image
// cos/sin are precomputed: [seq_len, head_dim/2] (interleaved pairs)
//
// For each pair (x0, x1) at positions 2i, 2i+1:
//   out[2i]   = x0 * cos[i] - x1 * sin[i]
//   out[2i+1] = x0 * sin[i] + x1 * cos[i]
//
// Dispatch: global_work_size = total_elements / 2
//   where total_elements = batch * n_head * seq_len * head_dim
//   (applied to Q and K separately)

__kernel void apply_rotary_emb(
    __global const float* input,     // [batch * n_head, seq_len, head_dim]
    __global const float* cos_emb,   // [seq_len, head_dim/2]
    __global const float* sin_emb,   // [seq_len, head_dim/2]
    __global float* output,          // [batch * n_head, seq_len, head_dim]
    const uint n_head_total,         // batch * n_head
    const uint seq_len,
    const uint head_dim)
{
    uint gid = get_global_id(0);
    uint total_pairs = n_head_total * seq_len * (head_dim >> 1);
    if (gid >= total_pairs) return;

    // Decode pair index -> (head_seq_idx, pair_in_head)
    uint half_dim = head_dim >> 1;
    uint pair_in_head = gid % half_dim;
    uint head_seq_idx = gid / half_dim;
    uint seq_idx = head_seq_idx % seq_len;

    // RoPE cos/sin are per (seq_pos, pair)
    uint rope_idx = seq_idx * half_dim + pair_in_head;
    float c = cos_emb[rope_idx];
    float s = sin_emb[rope_idx];

    // Input pair at interleaved positions
    uint base = head_seq_idx * head_dim + pair_in_head * 2;
    float x0 = input[base];
    float x1 = input[base + 1];

    output[base]     = x0 * c - x1 * s;
    output[base + 1] = x0 * s + x1 * c;
}

// ==================== Flux-specific kernels ====================

// QKV Split + Transpose: deinterleave [seq, 3*dim] -> Q, K, V each [n_heads, seq, head_dim]
//
// Input layout (per row of seq): [q_h0_d0..q_h0_dH, q_h1.., ..., q_hN.., k_h0.., ..., k_hN.., v_h0.., ..., v_hN..]
// Output: Q[h * seq * hd + s * hd + d] = in[s * 3*dim + h * hd + d]
//         K[h * seq * hd + s * hd + d] = in[s * 3*dim + dim + h * hd + d]
//         V[h * seq * hd + s * hd + d] = in[s * 3*dim + 2*dim + h * hd + d]
//
// Dispatch: global_work_size = n_heads * seq_len * head_dim
__kernel void qkv_split(
    __global const float* input,     // [seq_len, 3 * dim]
    __global float* q_out,           // [n_heads, seq_len, head_dim]
    __global float* k_out,           // [n_heads, seq_len, head_dim]
    __global float* v_out,           // [n_heads, seq_len, head_dim]
    const uint n_heads,
    const uint seq_len,
    const uint head_dim)
{
    uint gid = get_global_id(0);
    uint total = n_heads * seq_len * head_dim;
    if (gid >= total) return;

    uint d = gid % head_dim;
    uint tmp = gid / head_dim;
    uint s = tmp % seq_len;
    uint h = tmp / seq_len;

    uint dim = n_heads * head_dim;
    uint in_base = s * 3 * dim;
    uint head_off = h * head_dim + d;

    uint out_idx = h * seq_len * head_dim + s * head_dim + d;

    q_out[out_idx] = input[in_base + head_off];
    k_out[out_idx] = input[in_base + dim + head_off];
    v_out[out_idx] = input[in_base + 2 * dim + head_off];
}

// Flux RoPE: Apply 2x2 rotation matrix from precomputed position embeddings.
//
// pe layout: [1, 1, seq_len, head_dim/2, 2, 2]
//   For each position s, pair i: pe[s, i, :, :] = [[cos, -sin], [sin, cos]]
//   Stored as: pe[s * half_dim * 4 + i * 4 + 0] = cos
//              pe[s * half_dim * 4 + i * 4 + 1] = -sin
//              pe[s * half_dim * 4 + i * 4 + 2] = sin
//              pe[s * half_dim * 4 + i * 4 + 3] = cos
//
// x layout: [n_heads, seq_len, head_dim] (from qkv_split)
// For each pair (x0, x1) at positions 2i, 2i+1:
//   out[2i]   = pe[0,0]*x0 + pe[0,1]*x1  = cos*x0 - sin*x1
//   out[2i+1] = pe[1,0]*x0 + pe[1,1]*x1  = sin*x0 + cos*x1
//
// Dispatch: global_work_size = n_heads * seq_len * (head_dim/2)
__kernel void flux_rope(
    __global const float* input,     // [n_heads, seq_len, head_dim]
    __global const float* pe,        // [seq_len, head_dim/2, 2, 2]
    __global float* output,          // [n_heads, seq_len, head_dim]
    const uint n_heads,
    const uint seq_len,
    const uint head_dim)
{
    uint gid = get_global_id(0);
    uint half_dim = head_dim >> 1;
    uint total_pairs = n_heads * seq_len * half_dim;
    if (gid >= total_pairs) return;

    uint pair = gid % half_dim;
    uint tmp = gid / half_dim;
    uint s = tmp % seq_len;
    uint h = tmp / seq_len;

    // PE is shared across heads, indexed by (seq_pos, pair)
    uint pe_base = s * half_dim * 4 + pair * 4;
    float pe00 = pe[pe_base + 0]; // cos
    float pe01 = pe[pe_base + 1]; // -sin
    float pe10 = pe[pe_base + 2]; // sin
    float pe11 = pe[pe_base + 3]; // cos

    // Input pair at interleaved positions
    uint base = (h * seq_len + s) * head_dim + pair * 2;
    float x0 = input[base];
    float x1 = input[base + 1];

    output[base]     = pe00 * x0 + pe01 * x1;
    output[base + 1] = pe10 * x0 + pe11 * x1;
}

// Concatenate two sequences along the seq dimension:
// a: [n_heads, seq_a, head_dim], b: [n_heads, seq_b, head_dim]
// -> output: [n_heads, seq_a + seq_b, head_dim]
//
// Dispatch: global_work_size = n_heads * (seq_a + seq_b) * head_dim
__kernel void concat_seq(
    __global const float* a,         // [n_heads, seq_a, head_dim]
    __global const float* b,         // [n_heads, seq_b, head_dim]
    __global float* output,          // [n_heads, seq_a + seq_b, head_dim]
    const uint n_heads,
    const uint seq_a,
    const uint seq_b,
    const uint head_dim)
{
    uint gid = get_global_id(0);
    uint total_seq = seq_a + seq_b;
    uint total = n_heads * total_seq * head_dim;
    if (gid >= total) return;

    uint d = gid % head_dim;
    uint tmp = gid / head_dim;
    uint s = tmp % total_seq;
    uint h = tmp / total_seq;

    float val;
    if (s < seq_a) {
        val = a[h * seq_a * head_dim + s * head_dim + d];
    } else {
        val = b[h * seq_b * head_dim + (s - seq_a) * head_dim + d];
    }
    output[gid] = val;
}

// Split concatenated attention output back into two portions:
// input: [n_heads, seq_a + seq_b, head_dim] -> transpose+flatten -> [seq, dim]
// We split and transpose in one kernel to avoid intermediate buffers.
// output_a: [seq_a, dim], output_b: [seq_b, dim]  where dim = n_heads * head_dim
//
// Dispatch: global_work_size = (seq_a + seq_b) * dim
__kernel void split_heads(
    __global const float* input,     // [n_heads, total_seq, head_dim]
    __global float* output_a,        // [seq_a, n_heads * head_dim]
    __global float* output_b,        // [seq_b, n_heads * head_dim]
    const uint n_heads,
    const uint seq_a,
    const uint seq_b,
    const uint head_dim)
{
    uint gid = get_global_id(0);
    uint dim = n_heads * head_dim;
    uint total_seq = seq_a + seq_b;
    uint total = total_seq * dim;
    if (gid >= total) return;

    uint hd = gid % dim;           // position within dim = h * head_dim + d
    uint s = gid / dim;            // seq position in concatenated seq
    uint h = hd / head_dim;
    uint d = hd % head_dim;

    // Read from [n_heads, total_seq, head_dim] layout
    float val = input[h * total_seq * head_dim + s * head_dim + d];

    if (s < seq_a) {
        output_a[s * dim + hd] = val;
    } else {
        output_b[(s - seq_a) * dim + hd] = val;
    }
}

// ==================== Strided copy kernels ====================

// QKV split from strided rows: extract first 3*dim from each row of stride row_stride,
// then split+transpose into Q, K, V [n_heads, seq_len, head_dim].
// Replaces: per-row copy_region loop + qkv_split.
//
// Input: [seq_len, row_stride] where row_stride >= 3 * dim
// Dispatch: global_work_size = n_heads * seq_len * head_dim
__kernel void qkv_split_strided(
    __global const float* input,     // [seq_len, row_stride]
    __global float* q_out,           // [n_heads, seq_len, head_dim]
    __global float* k_out,
    __global float* v_out,
    const uint n_heads,
    const uint seq_len,
    const uint head_dim,
    const uint row_stride)           // actual stride per row (in floats)
{
    uint gid = get_global_id(0);
    uint total = n_heads * seq_len * head_dim;
    if (gid >= total) return;

    uint d = gid % head_dim;
    uint tmp = gid / head_dim;
    uint s = tmp % seq_len;
    uint h = tmp / seq_len;

    uint dim = n_heads * head_dim;
    uint in_base = s * row_stride;  // row_stride instead of 3*dim
    uint head_off = h * head_dim + d;

    uint out_idx = h * seq_len * head_dim + s * head_dim + d;

    q_out[out_idx] = input[in_base + head_off];
    k_out[out_idx] = input[in_base + dim + head_off];
    v_out[out_idx] = input[in_base + 2 * dim + head_off];
}

// Extract a slice from strided rows: copy columns [col_offset..col_offset+width] from each row.
// Input:  [seq_len, row_stride]
// Output: [seq_len, width]  (contiguous)
//
// Dispatch: global_work_size = seq_len * width
__kernel void strided_slice(
    __global const float* input,     // [seq_len, row_stride]
    __global float* output,          // [seq_len, width]
    const uint row_stride,           // stride per row (in floats)
    const uint col_offset,           // start column (in floats)
    const uint width,                // number of columns to extract
    const uint total_elements)       // seq_len * width
{
    uint gid = get_global_id(0);
    if (gid >= total_elements) return;

    uint col = gid % width;
    uint row = gid / width;

    output[gid] = input[row * row_stride + col_offset + col];
}

// Concatenate two column blocks into strided rows:
// a: [seq_len, width_a], b: [seq_len, width_b]
// -> output: [seq_len, width_a + width_b]  (contiguous)
//
// Dispatch: global_work_size = seq_len * (width_a + width_b)
__kernel void concat_cols(
    __global const float* a,         // [seq_len, width_a]
    __global const float* b,         // [seq_len, width_b]
    __global float* output,          // [seq_len, width_a + width_b]
    const uint width_a,
    const uint width_b,
    const uint total_elements)       // seq_len * (width_a + width_b)
{
    uint gid = get_global_id(0);
    if (gid >= total_elements) return;

    uint out_width = width_a + width_b;
    uint col = gid % out_width;
    uint row = gid / out_width;

    if (col < width_a) {
        output[gid] = a[row * width_a + col];
    } else {
        output[gid] = b[row * width_b + (col - width_a)];
    }
}
