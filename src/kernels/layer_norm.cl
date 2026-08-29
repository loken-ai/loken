// LayerNorm kernel for Flux image generation model
//
// Computes: output[i] = (x[i] - mean) / sqrt(var + eps) * scale[i] + bias[i]
// where mean and var are computed per-row (each row = one token of dim elements)
//
// Dispatch: global_work_size = seq_len * WG_SIZE
//           local_work_size  = WG_SIZE
// One workgroup per row (token).

#define LN_WG 256

// LayerNorm with scale and bias (Flux uses this)
__kernel void layer_norm(
    __global const float* input,      // [seq_len, dim]
    __global float* output,           // [seq_len, dim]
    __global const float* scale,      // [dim]
    __global const float* bias,       // [dim]
    const uint dim,
    const float eps)
{
    uint row = get_group_id(0);
    uint lid = get_local_id(0);

    __global const float* x = input + row * dim;
    __global float* y = output + row * dim;

    // Pass 1: compute mean using vectorized loads
    float partial_sum = 0.0f;
    uint vec_dim = dim >> 2;
    for (uint i = lid; i < vec_dim; i += LN_WG) {
        float4 v = vload4(i, x);
        partial_sum += v.x + v.y + v.z + v.w;
    }
    uint tail_start = vec_dim << 2;
    for (uint i = tail_start + lid; i < dim; i += LN_WG) {
        partial_sum += x[i];
    }

    __local float scratch[LN_WG];
    scratch[lid] = partial_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = LN_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = scratch[0] / (float)dim;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2: compute variance
    float partial_var = 0.0f;
    for (uint i = lid; i < vec_dim; i += LN_WG) {
        float4 v = vload4(i, x);
        float4 d = v - (float4)(mean);
        partial_var += dot(d, d);
    }
    for (uint i = tail_start + lid; i < dim; i += LN_WG) {
        float d = x[i] - mean;
        partial_var += d * d;
    }

    scratch[lid] = partial_var;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = LN_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_std = rsqrt(scratch[0] / (float)dim + eps);

    // Pass 3: normalize + scale + bias
    for (uint i = lid; i < vec_dim; i += LN_WG) {
        float4 v = vload4(i, x);
        float4 s = vload4(i, scale);
        float4 b = vload4(i, bias);
        float4 normed = (v - (float4)(mean)) * inv_std;
        vstore4(normed * s + b, i, y);
    }
    for (uint i = tail_start + lid; i < dim; i += LN_WG) {
        float normed = (x[i] - mean) * inv_std;
        y[i] = normed * scale[i] + bias[i];
    }
}

// LayerNorm without bias (just scale) - some Flux layers use this
__kernel void layer_norm_no_bias(
    __global const float* input,
    __global float* output,
    __global const float* scale,
    const uint dim,
    const float eps)
{
    uint row = get_group_id(0);
    uint lid = get_local_id(0);

    __global const float* x = input + row * dim;
    __global float* y = output + row * dim;

    float partial_sum = 0.0f;
    uint vec_dim = dim >> 2;
    for (uint i = lid; i < vec_dim; i += LN_WG) {
        float4 v = vload4(i, x);
        partial_sum += v.x + v.y + v.z + v.w;
    }
    uint tail_start = vec_dim << 2;
    for (uint i = tail_start + lid; i < dim; i += LN_WG) {
        partial_sum += x[i];
    }

    __local float scratch[LN_WG];
    scratch[lid] = partial_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = LN_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = scratch[0] / (float)dim;
    barrier(CLK_LOCAL_MEM_FENCE);

    float partial_var = 0.0f;
    for (uint i = lid; i < vec_dim; i += LN_WG) {
        float4 v = vload4(i, x);
        float4 d = v - (float4)(mean);
        partial_var += dot(d, d);
    }
    for (uint i = tail_start + lid; i < dim; i += LN_WG) {
        float d = x[i] - mean;
        partial_var += d * d;
    }

    scratch[lid] = partial_var;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = LN_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_std = rsqrt(scratch[0] / (float)dim + eps);

    for (uint i = lid; i < vec_dim; i += LN_WG) {
        float4 v = vload4(i, x);
        float4 s = vload4(i, scale);
        vstore4((v - (float4)(mean)) * inv_std * s, i, y);
    }
    for (uint i = tail_start + lid; i < dim; i += LN_WG) {
        y[i] = (x[i] - mean) * inv_std * scale[i];
    }
}
