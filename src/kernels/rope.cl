// RoPE (Rotary Position Embeddings) compute kernel - interleaved variant
//
// Pairs (x[2i], x[2i+1]) rotated by (cos[i], sin[i]) for each position
// Supports multiple heads: processes all heads, for both Q and K matrices

__kernel void rope(
    __global float* q,
    __global const float* cos,
    __global const float* sin,
    const uint seq_len,
    const uint n_heads,     // total heads in this buffer (n_head for Q, n_kv_head for K)
    const uint head_dim,
    const uint index_pos)
{
    // One work-item per (position, head, dimension_pair)
    int gid = get_global_id(0);
    uint half_dim = head_dim / 2;
    if (gid >= seq_len * n_heads * half_dim) return;

    // Unpack gid into (pos, head, pair)
    uint pos  = gid / (n_heads * half_dim);
    uint rem  = gid % (n_heads * half_dim);
    uint head = rem / half_dim;
    uint pair = rem % half_dim;

    // Global absolute position in tensor: [seq_len, n_heads, head_dim]
    uint base_idx = (pos * n_heads + head) * head_dim + pair * 2;

    // Load interleaved pair
    float x0 = q[base_idx];
    float x1 = q[base_idx + 1];

    // RoPE lookup: (absolute_pos, freq_pair)
    uint cos_idx = (index_pos + pos) * half_dim + pair;
    float c = cos[cos_idx];
    float s = sin[cos_idx];

    // Apply rotation: [cos -sin] [x0]
    //                 [sin  cos] [x1]
    q[base_idx]     = x0 * c - x1 * s;
    q[base_idx + 1] = x0 * s + x1 * c;
}

// Fused dual RoPE: apply RoPE to both Q and K in a single kernel launch.
// Dispatch: global_work_size = seq_len * (n_head + n_kv_head) * half_dim
// First n_head heads worth of work -> Q buffer, remaining n_kv_head -> K buffer.
__kernel void dual_rope(
    __global float* q_buf,
    __global float* k_buf,
    __global const float* cos,
    __global const float* sin,
    const uint seq_len,
    const uint n_head,
    const uint n_kv_head,
    const uint head_dim,
    const uint index_pos)
{
    int gid = get_global_id(0);
    uint half_dim = head_dim / 2;
    uint total_heads = n_head + n_kv_head;
    if (gid >= seq_len * total_heads * half_dim) return;

    uint pos  = gid / (total_heads * half_dim);
    uint rem  = gid % (total_heads * half_dim);
    uint head = rem / half_dim;
    uint pair = rem % half_dim;

    // Determine which buffer and local head index
    __global float* buf;
    uint local_head;
    uint local_n_heads;
    if (head < n_head) {
        buf = q_buf;
        local_head = head;
        local_n_heads = n_head;
    } else {
        buf = k_buf;
        local_head = head - n_head;
        local_n_heads = n_kv_head;
    }

    uint base_idx = (pos * local_n_heads + local_head) * head_dim + pair * 2;
    float x0 = buf[base_idx];
    float x1 = buf[base_idx + 1];

    uint cos_idx = (index_pos + pos) * half_dim + pair;
    float c = cos[cos_idx];
    float s = sin[cos_idx];

    buf[base_idx]     = x0 * c - x1 * s;
    buf[base_idx + 1] = x0 * s + x1 * c;
}
