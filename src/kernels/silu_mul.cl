// SwiGLU activation: out = SiLU(gate) * up
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Uses float4 vectorization for 4x throughput

__kernel void silu_mul(
    __global const float* gate,
    __global const float* up,
    __global float* out,
    const uint num_elements)
{
    int idx = get_global_id(0);
    int idx4 = idx * 4;
    if (idx4 + 3 < num_elements) {
        float4 g = vload4(idx, gate);
        float4 u = vload4(idx, up);
        float4 silu = g / (1.0f + exp(-g));
        vstore4(silu * u, idx, out);
    } else {
        // Scalar tail
        for (int i = idx4; i < num_elements && i < idx4 + 4; i++) {
            float g = gate[i];
            float s = g / (1.0f + exp(-g));
            out[i] = s * up[i];
        }
    }
}
