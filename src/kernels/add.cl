// Element-wise addition: out = a + b (for residual connections)
// Uses float4 vectorization for 4x throughput on bandwidth-bound ops

__kernel void add(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const uint num_elements)
{
    int idx = get_global_id(0);
    // float4 path: each work item handles 4 elements
    int idx4 = idx * 4;
    if (idx4 + 3 < num_elements) {
        float4 va = vload4(idx, a);
        float4 vb = vload4(idx, b);
        vstore4(va + vb, idx, out);
    } else {
        // Scalar tail for non-multiple-of-4 sizes
        for (int i = idx4; i < num_elements && i < idx4 + 4; i++) {
            out[i] = a[i] + b[i];
        }
    }
}
