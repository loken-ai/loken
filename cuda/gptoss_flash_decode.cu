// gpt-oss flash-attention DECODE (seq_q = 1) with per-head attention sinks + GQA,
// compiled by nvcc (F16 I/O; the NVRTC path is F32-only). Replaces the cuBLAS
// scores matmul + sink-softmax + V matmul chain with ONE launch - no cuBLAS (so
// the forward becomes fully CUDA-graph-capturable) and no [n_head, kv_len] scores
// materialized in HBM.
//
// One warp (32 lanes) per (query head, batch); each lane owns 2 of the hd=64
// dims (d = lane, lane+32). Online (FlashAttention-style) softmax in F32. The
// sink is a virtual key with logit = sinks[head] and value 0: seed the running
// max with it (m0 = sink, l0 = 1, acc0 = 0) so it lands in the denominator only.
// expf (not __expf) matches the reference precision so greedy argmax is unchanged.
//
// K/V come from the KV-cache narrow [b, n_kv, kv_len, hd] of a larger buffer, so
// they are non-contiguous: pass per-dim strides (in elements).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GPTOSS_HD 64

extern "C" __global__ void gptoss_flash_decode_f16_kernel(
    const __half* __restrict__ Q,      // [b, n_head, hd] (contiguous, seq=1)
    const __half* __restrict__ K,      // [b, n_kv, kv_len, hd] (strided)
    const __half* __restrict__ V,      // [b, n_kv, kv_len, hd] (strided)
    const float*  __restrict__ mask,   // [kv_len] additive, or nullptr
    const float*  __restrict__ sinks,  // [n_head]
    __half* __restrict__ out,          // [b, n_head, hd] (contiguous)
    int n_head, int n_kv, int kv_len, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride
) {
    const int head = blockIdx.x;
    const int bi   = blockIdx.y;
    const int lane = threadIdx.x;            // 0..31
    if (lane >= 32) return;
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;

    const __half* q = Q + ((long)bi * n_head + head) * GPTOSS_HD;
    const float q0 = __half2float(q[lane]);
    const float q1 = __half2float(q[lane + 32]);

    const __half* kh = K + (long)bi * k_batch_stride + (long)kvh * k_head_stride;
    const __half* vh = V + (long)bi * v_batch_stride + (long)kvh * v_head_stride;

    // Seed the online softmax with the sink (virtual key: logit=sink, value=0).
    // sinks==nullptr (e.g. lfm2, which has no attention sinks) => no virtual key:
    // start the running max at -inf and l=0 so only real keys populate softmax.
    float m   = sinks ? sinks[head] : -INFINITY;
    float l   = sinks ? 1.0f : 0.0f; // exp(sink - m) with m == sink, else empty
    float acc0 = 0.0f, acc1 = 0.0f;

    for (int j = 0; j < kv_len; ++j) {
        const __half* kj = kh + (long)j * k_pos_stride;
        float p = q0 * __half2float(kj[lane]) + q1 * __half2float(kj[lane + 32]);
        // warp reduce-sum the dot product
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
        p = __shfl_sync(0xffffffff, p, 0);   // broadcast lane 0's full sum
        float s = scale * p + (mask ? mask[j] : 0.0f);

        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);       // rescale prior accumulators
        float pj    = expf(s - m_new);
        l = l * alpha + pj;
        const __half* vj = vh + (long)j * v_pos_stride;
        acc0 = acc0 * alpha + pj * __half2float(vj[lane]);
        acc1 = acc1 * alpha + pj * __half2float(vj[lane + 32]);
        m = m_new;
    }

    const float inv = 1.0f / l;
    __half* o = out + ((long)bi * n_head + head) * GPTOSS_HD;
    o[lane]      = __float2half(acc0 * inv);
    o[lane + 32] = __float2half(acc1 * inv);
}

extern "C" void gptoss_flash_decode_f16(
    const void* Q, const void* K, const void* V, const float* mask, const float* sinks,
    void* out, int batch, int n_head, int n_kv, int kv_len, int head_dim, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride,
    cudaStream_t stream
) {
    if (head_dim != GPTOSS_HD) return;   // caller falls back for other head dims
    dim3 grid(n_head, batch, 1);
    dim3 block(32, 1, 1);
    gptoss_flash_decode_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, mask, sinks, (__half*)out,
        n_head, n_kv, kv_len, scale,
        k_batch_stride, k_head_stride, k_pos_stride,
        v_batch_stride, v_head_stride, v_pos_stride);
}

// -- Split-K Flash-Decoding ---------------------------------------------------
// The single-warp kernel above scans the WHOLE kv_len serially with one warp per
// head (64 warps for gpt-oss) - fine at short context, but at kv_len≈6 K it leaves
// the GPU almost entirely idle and per-token cost grows with the cache rather than
// staying flat. Split-K partitions kv_len into `nsplit` chunks: PASS 1 computes a
// partial online-softmax (m,l,acc) per (head, batch, chunk) - grid is n_headxbatchx
// nsplit blocks, so the KV scan is parallel across the sequence. PASS 2 reduces the
// nsplit partials per head with the standard log-sum-exp merge. Mathematically the
// same result as the serial kernel (FP reassociated; validated coherent).
//
// The attention SINK (gpt-oss) is a virtual key seeded into split 0 ONLY, so it
// lands in the global denominator exactly once after the merge.

extern "C" __global__ void gptoss_flash_decode_split_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    const float* __restrict__ mask, const float* __restrict__ sinks,
    float* __restrict__ part_m,    // [b, n_head, nsplit]
    float* __restrict__ part_l,    // [b, n_head, nsplit]
    float* __restrict__ part_acc,  // [b, n_head, nsplit, hd]
    int n_head, int n_kv, int kv_len, int nsplit, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride
) {
    const int head  = blockIdx.x;
    const int bi    = blockIdx.y;
    const int split = blockIdx.z;
    const int lane  = threadIdx.x;
    if (lane >= 32) return;
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;

    const int split_len = (kv_len + nsplit - 1) / nsplit;
    const int j0 = split * split_len;
    const int j1 = min(j0 + split_len, kv_len);

    const long pidx = ((long)bi * n_head + head) * nsplit + split;
    // sink seeds split 0 only; other splits start empty (m=-inf, l=0).
    float m = (split == 0 && sinks) ? sinks[head] : -INFINITY;
    float l = (split == 0 && sinks) ? 1.0f : 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    if (j0 < j1) {
        const __half* q = Q + ((long)bi * n_head + head) * GPTOSS_HD;
        const float q0 = __half2float(q[lane]);
        const float q1 = __half2float(q[lane + 32]);
        const __half* kh = K + (long)bi * k_batch_stride + (long)kvh * k_head_stride;
        const __half* vh = V + (long)bi * v_batch_stride + (long)kvh * v_head_stride;
        for (int j = j0; j < j1; ++j) {
            const __half* kj = kh + (long)j * k_pos_stride;
            float p = q0 * __half2float(kj[lane]) + q1 * __half2float(kj[lane + 32]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
            p = __shfl_sync(0xffffffff, p, 0);
            float s = scale * p + (mask ? mask[j] : 0.0f);
            // Hard-masked key (sliding window / causal): -inf. Skip it - otherwise a
            // split entirely outside the window seeds m=-inf and expf(-inf-(-inf))=NaN.
            // All 32 lanes share j and mask[j], so this branch is warp-uniform.
            if (s == -INFINITY) continue;
            float m_new = fmaxf(m, s);
            float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
            float pj    = expf(s - m_new);
            l = l * alpha + pj;
            const __half* vj = vh + (long)j * v_pos_stride;
            acc0 = acc0 * alpha + pj * __half2float(vj[lane]);
            acc1 = acc1 * alpha + pj * __half2float(vj[lane + 32]);
            m = m_new;
        }
    }
    part_m[pidx] = m;
    part_l[pidx] = l;
    part_acc[pidx * GPTOSS_HD + lane]      = acc0;
    part_acc[pidx * GPTOSS_HD + lane + 32] = acc1;
}

extern "C" __global__ void gptoss_flash_decode_combine_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, __half* __restrict__ out,
    int n_head, int nsplit
) {
    const int head = blockIdx.x;
    const int bi   = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const long base = ((long)bi * n_head + head) * nsplit;

    // pass 1: global max across splits
    float gm = -INFINITY;
    for (int s = 0; s < nsplit; ++s) gm = fmaxf(gm, part_m[base + s]);
    // pass 2: rescale-and-sum
    float l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        float ms = part_m[base + s];
        float alpha = (ms == -INFINITY) ? 0.0f : expf(ms - gm);
        l    += alpha * part_l[base + s];
        const float* pa = part_acc + (base + s) * GPTOSS_HD;
        acc0 += alpha * pa[lane];
        acc1 += alpha * pa[lane + 32];
    }
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    __half* o = out + ((long)bi * n_head + head) * GPTOSS_HD;
    o[lane]      = __float2half(acc0 * inv);
    o[lane + 32] = __float2half(acc1 * inv);
}

extern "C" void gptoss_flash_decode_split_f16(
    const void* Q, const void* K, const void* V, const float* mask, const float* sinks,
    void* out, float* part_m, float* part_l, float* part_acc,
    int batch, int n_head, int n_kv, int kv_len, int nsplit, int head_dim, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride,
    cudaStream_t stream
) {
    if (head_dim != GPTOSS_HD) return;
    dim3 grid(n_head, batch, nsplit);
    dim3 block(32, 1, 1);
    gptoss_flash_decode_split_kernel<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, mask, sinks,
        part_m, part_l, part_acc, n_head, n_kv, kv_len, nsplit, scale,
        k_batch_stride, k_head_stride, k_pos_stride,
        v_batch_stride, v_head_stride, v_pos_stride);
    dim3 cgrid(n_head, batch, 1);
    gptoss_flash_decode_combine_kernel<<<cgrid, block, 0, stream>>>(
        part_m, part_l, part_acc, (__half*)out, n_head, nsplit);
}
