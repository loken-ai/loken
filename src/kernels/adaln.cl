// AdaLN (Adaptive Layer Normalization) kernels for image generation models
//
// Z-Image uses: output = (1 + scale) * rms_norm(x), then gating: x + gate * output
// Flux uses: output = (1 + scale) * layernorm(x) + shift
//
// These kernels fuse the scale/shift/gate operations with norm output
// to reduce memory bandwidth.

// Scale-shift after norm: output = (1 + scale) * normed_input
// scale is per-element (broadcast over seq dimension)
// Dispatch: global_work_size = total_elements / 4, one work item per 4 elements
__kernel void scale_after_norm(
    __global const float* normed,     // [seq_len, dim]
    __global const float* scale,      // [1, dim] or [dim]
    __global float* output,           // [seq_len, dim]
    const uint dim,
    const uint total_elements)
{
    uint idx = get_global_id(0);
    uint idx4 = idx * 4;
    if (idx4 + 3 < total_elements) {
        float4 n = vload4(idx, normed);
        uint dim_offset = idx4 % dim;
        float4 s = vload4(dim_offset >> 2, scale);
        vstore4(n * (1.0f + s), idx, output);
    } else {
        for (uint i = idx4; i < total_elements && i < idx4 + 4; i++) {
            uint d = i % dim;
            output[i] = normed[i] * (1.0f + scale[d]);
        }
    }
}

// Gated residual: output = x + tanh(gate) * y
// gate is [1, dim], x and y are [seq_len, dim]
// Dispatch: global_work_size = total_elements / 4
__kernel void gated_residual(
    __global const float* x,          // [seq_len, dim] residual
    __global const float* y,          // [seq_len, dim] block output
    __global const float* gate,       // [1, dim]
    __global float* output,           // [seq_len, dim]
    const uint dim,
    const uint total_elements)
{
    uint idx = get_global_id(0);
    uint idx4 = idx * 4;
    if (idx4 + 3 < total_elements) {
        float4 xv = vload4(idx, x);
        float4 yv = vload4(idx, y);
        uint dim_offset = idx4 % dim;
        float4 gv = vload4(dim_offset >> 2, gate);
        float4 gated = tanh(gv);
        vstore4(xv + gated * yv, idx, output);
    } else {
        for (uint i = idx4; i < total_elements && i < idx4 + 4; i++) {
            uint d = i % dim;
            float g = tanh(gate[d]);
            output[i] = x[i] + g * y[i];
        }
    }
}

// Note: residual_add (output = x + y) lives in add.cl - use that kernel instead.

// Elementwise multiply: output = a * b (broadcast b over seq dim)
// Useful for attention output scaling etc.
// Dispatch: global_work_size = total_elements / 4
__kernel void broadcast_mul(
    __global const float* input,      // [seq_len, dim]
    __global const float* scale,      // [1, dim]
    __global float* output,           // [seq_len, dim]
    const uint dim,
    const uint total_elements)
{
    uint idx = get_global_id(0);
    uint idx4 = idx * 4;
    if (idx4 + 3 < total_elements) {
        float4 a = vload4(idx, input);
        uint dim_offset = idx4 % dim;
        float4 s = vload4(dim_offset >> 2, scale);
        vstore4(a * s, idx, output);
    } else {
        for (uint i = idx4; i < total_elements && i < idx4 + 4; i++) {
            uint d = i % dim;
            output[i] = input[i] * scale[d];
        }
    }
}

// Broadcast add: output = input + bias (broadcast bias [dim] over seq dimension)
// input: [seq_len, dim], bias: [dim], output: [seq_len, dim]
// Dispatch: global_work_size = total_elements / 4
__kernel void broadcast_add(
    __global const float* input,      // [seq_len, dim]
    __global const float* bias,       // [1, dim]
    __global float* output,           // [seq_len, dim]
    const uint dim,
    const uint total_elements)
{
    uint idx = get_global_id(0);
    uint idx4 = idx * 4;
    if (idx4 + 3 < total_elements) {
        float4 a = vload4(idx, input);
        uint dim_offset = idx4 % dim;
        float4 b = vload4(dim_offset >> 2, bias);
        vstore4(a + b, idx, output);
    } else {
        for (uint i = idx4; i < total_elements && i < idx4 + 4; i++) {
            uint d = i % dim;
            output[i] = input[i] + bias[d];
        }
    }
}
