// RMSNorm compute kernel: y = x / sqrt(mean(x^2) + eps) * scale
//
// Uses workgroup-parallel reduction for sum-of-squares.
// Dispatch: global_work_size = seq_len * 256, local_work_size = 256
// Each workgroup (256 threads) processes one token row.
//
// v2 optimizations:
//   - vload4/vstore4 vectorization for 4x memory bandwidth utilization
//   - dot(val, val) for efficient sum-of-squares

#define RMS_WG 256

__kernel void rms_norm(
    __global const float* x,
    __global float* out,
    __global const float* scale,
    const uint hidden_dim,
    const float eps)
{
    uint row = get_group_id(0);    // one workgroup per token
    uint lid = get_local_id(0);

    __local float scratch[RMS_WG];
    uint vec_dim = hidden_dim >> 2;  // hidden_dim / 4

    // Phase 1: each thread computes partial sum of squares (float4 vectorized)
    float partial = 0.0f;
    for (uint i = lid; i < vec_dim; i += RMS_WG) {
        float4 val = vload4(i, x + row * hidden_dim);
        partial += dot(val, val);
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: tree reduction in local memory
    for (uint s = RMS_WG / 2; s > 0; s >>= 1) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Phase 3: compute norm factor (broadcast via local mem read)
    float norm = rsqrt(scratch[0] / (float)hidden_dim + eps);

    // Phase 4: apply norm + scale (float4 vectorized)
    for (uint i = lid; i < vec_dim; i += RMS_WG) {
        float4 val = vload4(i, x + row * hidden_dim);
        float4 sc = vload4(i, scale);
        vstore4(val * norm * sc, i, out + row * hidden_dim);
    }
}
