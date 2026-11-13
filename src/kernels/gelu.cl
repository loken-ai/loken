// GELU activation kernel for Flux image generation model
//
// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// Fast approximation: GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Each work-item handles 4 consecutive elements.
// Dispatch: global_work_size = ceil(total_elements / 4)
//
// tanh(u) is written as 1 - 2 / (exp(2u) + 1).
// The form holds at both ends: an infinite exp gives 1 and a zero exp gives -1.

#define GELU_SQRT_2_OVER_PI 0.7978845608f  // sqrt(2/pi)
#define GELU_COEFF          0.044715f

float gelu_tanh_approx(float x) {
    float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    float t = 1.0f - 2.0f / (exp(2.0f * inner) + 1.0f);
    return 0.5f * x * (1.0f + t);
}

// GELU activation (in-place capable: input == output is safe)
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const uint total_elements)
{
    uint idx4 = get_global_id(0) * 4;
    for (uint i = idx4; i < total_elements && i < idx4 + 4; i++) {
        output[i] = gelu_tanh_approx(input[i]);
    }
}

// GELU over a slice of each row, in place.
// For Flux SingleBlock: linear1 produces [qkv, mlp] where mlp needs GELU.
// This kernel applies GELU only to the mlp portion (offset..offset+count) of each row.
__kernel void gelu_slice(
    __global float* data,             // [seq_len, full_dim] - modified in-place
    const uint full_dim,              // total last dimension
    const uint offset,                // start of GELU region within each row
    const uint count,                 // number of elements to GELU per row
    const uint seq_len)
{
    uint total = seq_len * count;
    uint idx4 = get_global_id(0) * 4;
    for (uint i = idx4; i < total && i < idx4 + 4; i++) {
        uint gi = (i / count) * full_dim + offset + (i % count);
        data[gi] = gelu_tanh_approx(data[gi]);
    }
}
