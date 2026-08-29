// GELU activation kernel for Flux image generation model
//
// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// Fast approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// Dispatch: global_work_size = total_elements / 4 (vectorized)

#define GELU_SQRT_2_OVER_PI 0.7978845608f  // sqrt(2/π)
#define GELU_COEFF          0.044715f

// Vectorized GELU activation (in-place capable: input == output is safe)
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const uint total_elements)
{
    uint idx = get_global_id(0);
    uint idx4 = idx * 4;
    if (idx4 + 3 < total_elements) {
        float4 x = vload4(idx, input);
        float4 x3 = x * x * x;
        float4 inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        float4 result = 0.5f * x * (1.0f + tanh(inner));
        vstore4(result, idx, output);
    } else {
        for (uint i = idx4; i < total_elements && i < idx4 + 4; i++) {
            float x = input[i];
            float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            output[i] = 0.5f * x * (1.0f + tanh(inner));
        }
    }
}

// Fused GELU + linear: output = gelu(input) applied to concatenated data
// For Flux SingleBlock: linear1 produces [qkv, mlp] where mlp needs GELU
// This kernel applies GELU only to the mlp portion (offset..offset+count)
__kernel void gelu_slice(
    __global float* data,             // [seq_len, full_dim] - modified in-place
    const uint full_dim,              // total last dimension
    const uint offset,                // start of GELU region within each row
    const uint count,                 // number of elements to GELU per row
    const uint seq_len)
{
    uint idx = get_global_id(0);
    uint total = seq_len * count;
    uint idx4 = idx * 4;

    if (idx4 + 3 < total) {
        // Map flat index to (row, col_within_slice)
        uint base_row = (idx4 / count);
        uint base_col = (idx4 % count);
        // Only use vectorized path if all 4 elements are in the same row
        if (base_col + 3 < count) {
            uint global_offset = base_row * full_dim + offset + base_col;
            float4 x = vload4(0, data + global_offset);
            float4 x3 = x * x * x;
            float4 inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
            vstore4(0.5f * x * (1.0f + tanh(inner)), 0, data + global_offset);
        } else {
            // Fall back to scalar for row boundaries
            for (uint i = idx4; i < total && i < idx4 + 4; i++) {
                uint row = i / count;
                uint col = i % count;
                uint gi = row * full_dim + offset + col;
                float x = data[gi];
                float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
                data[gi] = 0.5f * x * (1.0f + tanh(inner));
            }
        }
    } else {
        for (uint i = idx4; i < total && i < idx4 + 4; i++) {
            uint row = i / count;
            uint col = i % count;
            uint gi = row * full_dim + offset + col;
            float x = data[gi];
            float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            data[gi] = 0.5f * x * (1.0f + tanh(inner));
        }
    }
}
