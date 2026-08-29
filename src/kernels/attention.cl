// Scaled dot-product attention with causal + sliding-window masking
// Supports Grouped Query Attention (GQA) where n_head > n_kv_head
// Uses online softmax (single-pass accumulation, no temp buffer)
//
// Dispatch: global_work_size = seq_len * n_head
//           One work item per (query_position, head) pair.

__kernel void attention(
    __global const float* q,        // [seq_len, n_head, head_dim] f32
    __global const float* k_cache,  // [cache_len, n_kv_head, head_dim] f32
    __global const float* v_cache,  // [cache_len, n_kv_head, head_dim] f32
    __global float* output,         // [seq_len, n_head, head_dim] f32
    const uint n_head,
    const uint n_kv_head,
    const uint seq_len,
    const uint cache_len,           // = previous_cache_len + seq_len
    const uint head_dim,            // = 128
    const uint sliding_window)      // = 0 for unlimited
{
    // One work item per (query_position, head) pair
    int gid = get_global_id(0);
    if (gid >= seq_len * n_head) return;

    int q_pos = gid / n_head;
    int h     = gid % n_head;
    int kv_h  = (h * n_kv_head) / n_head;  // GQA: map query head to KV head
    int abs_q = (int)cache_len - (int)seq_len + q_pos;  // absolute position in sequence

    float scale = 1.0f / sqrt((float)head_dim);
    __global const float* q_ptr = q + (q_pos * n_head + h) * head_dim;

    // Private accumulator for V weighted sum (128 floats, compiler may use registers)
    float v_acc[128];
    for (uint d = 0; d < head_dim; d++) v_acc[d] = 0.0f;

    float max_score = -3.402823466e+38f;  // -inf
    float sum_exp = 0.0f;

    // Online softmax loop: compute scores, track max, accumulate weighted V
    for (int kp = 0; kp <= abs_q; kp++) {
        // Sliding window mask: skip positions too far back
        if (sliding_window > 0 && (abs_q - kp) >= (int)sliding_window) continue;

        __global const float* k_ptr = k_cache + (kp * n_kv_head + kv_h) * head_dim;

        // Compute Q.K score
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        // Update max and correction for numerical stability
        if (score > max_score) {
            float correction = exp(max_score - score);
            for (uint d = 0; d < head_dim; d++) {
                v_acc[d] *= correction;
            }
            sum_exp *= correction;
            max_score = score;
        }

        // Add this key's contribution to weighted V sum
        float w = exp(score - max_score);
        sum_exp += w;
        __global const float* v_ptr = v_cache + (kp * n_kv_head + kv_h) * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            v_acc[d] += w * v_ptr[d];
        }
    }

    // Normalize by sum of attention weights
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    __global float* out_ptr = output + (q_pos * n_head + h) * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        out_ptr[d] = v_acc[d] * inv_sum;
    }
}
