// Append new K,V vectors to the KV cache at the specified offset
// Copies [seq_len, n_kv_head, head_dim] buffer into [cache_len, n_kv_head, head_dim] cache
// at row offset cache_offset (after cache_offset rows)

__kernel void append_kv(
    __global const float* src,      // [seq_len, n_kv_head, head_dim] - new K or V
    __global float* dst,            // [max_seq, n_kv_head, head_dim] - KV cache
    const uint n_kv_head,
    const uint head_dim,
    const uint cache_offset,        // where to write in dst (row number)
    const uint seq_len)
{
    int gid = get_global_id(0);
    if (gid >= seq_len * n_kv_head * head_dim) return;

    // Unpack linear gid into (seq, head, dim)
    int seq_pos = gid / (n_kv_head * head_dim);
    int remainder = gid % (n_kv_head * head_dim);

    // Copy from src[seq_pos, ...] to dst[cache_offset + seq_pos, ...]
    int dst_row = cache_offset + seq_pos;
    dst[dst_row * n_kv_head * head_dim + remainder] = src[gid];
}

// Fused dual append: copy both K and V to their caches in a single dispatch.
// Each thread copies one element to K cache AND one to V cache (same index).
// Dispatch: global_work_size = seq_len * n_kv_head * head_dim (same as single append_kv)
__kernel void dual_append_kv(
    __global const float* k_src,
    __global const float* v_src,
    __global float* k_dst,
    __global float* v_dst,
    const uint n_kv_head,
    const uint head_dim,
    const uint cache_offset,
    const uint seq_len)
{
    int gid = get_global_id(0);
    if (gid >= seq_len * n_kv_head * head_dim) return;

    int seq_pos = gid / (n_kv_head * head_dim);
    int remainder = gid % (n_kv_head * head_dim);

    int dst_row = cache_offset + seq_pos;
    int dst_idx = dst_row * n_kv_head * head_dim + remainder;
    k_dst[dst_idx] = k_src[gid];
    v_dst[dst_idx] = v_src[gid];
}
