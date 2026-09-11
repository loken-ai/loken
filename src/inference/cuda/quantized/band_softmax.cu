// The softmax of one band of attention scores, in one pass.
//
// Between the two matmuls of an attention band, the scores are widened to full precision,
// masked, reduced to a row maximum, shifted, exponentiated, summed and narrowed again.
// Written as separate operations that is six passes over a tensor as large as the band,
// where the matmuls around it make two - so the shape of the attention costs more to
// describe than to compute. Here it is one pass that reads the scores and writes the
// weights, and the causal mask is an index comparison rather than a tensor: a square of
// zeros and infinities the size of the conversation, assembled on the host and copied to
// the card, says nothing the row's own position does not.
//
// One block per row. `run_max` carries each row's largest score from the bands before this
// one, so the weights come out already on the scale the caller will accumulate them at.

#define BAND_SOFTMAX_THREADS 256

extern "C" __global__ void __launch_bounds__(BAND_SOFTMAX_THREADS) band_softmax_f16(
    const __half * __restrict__ scores,
    const float  * __restrict__ run_max,
    __half       * __restrict__ weights,
    float        * __restrict__ new_max,
    float        * __restrict__ band_sum,
    int rows, int cols, int seq, int past, int c0
) {
    __shared__ float part[BAND_SOFTMAX_THREADS / 32];
    __shared__ float shared_top;

    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = BAND_SOFTMAX_THREADS / 32;
    const long base = (long)r * (long)cols;
    // The rows of a chunk are consecutive positions, so this one may see keys up to here.
    const int last = past + (r % seq);

    float m = -INFINITY;
    for (int j = tid; j < cols; j += BAND_SOFTMAX_THREADS) {
        if (c0 + j <= last) {
            const float v = __half2float(scores[base + j]);
            if (v > m) m = v;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) part[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float t = part[0];
        for (int w = 1; w < warps; ++w) t = fmaxf(t, part[w]);
        shared_top = fmaxf(t, run_max[r]);
    }
    __syncthreads();
    const float top = shared_top;

    float s = 0.0f;
    for (int j = tid; j < cols; j += BAND_SOFTMAX_THREADS) {
        float w = 0.0f;
        if (c0 + j <= last) w = __expf(__half2float(scores[base + j]) - top);
        weights[base + j] = __float2half(w);
        s += w;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) part[warp] = s;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < warps; ++w) t += part[w];
        new_max[r] = top;
        band_sum[r] = t;
    }
}
