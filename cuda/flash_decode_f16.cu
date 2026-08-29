// Head-dim-generalized flash-attention DECODE (seq_q = 1) with GQA, compiled by
// nvcc (F16 I/O). One fused launch replaces the cuBLAS scores matmul + softmax +
// V matmul chain - no [n_head, kv_len] scores in HBM. Whether this WINS depends
// on the arch's regime: it helps LAUNCH-bound decode (the saved launches/HBM
// round-trips beat the slower one-warp-per-head math) but loses on GPU-bound
// wide-head decode (proven: nemotron hd=128, -11%, where cuBLAS batched gemv is
// faster). qwen3.5 is launch-bound (~20-29% util) so it is worth testing there.
//
// One warp (32 lanes) per (query head, batch). Each lane owns DPL = head_dim/32
// dims: d = lane, lane+32, ..., lane+32*(DPL-1). Online (FlashAttention-style)
// softmax in F32; expf to match the reference so greedy argmax is unchanged.
// Optional per-head sinks; sinks==nullptr -> softmax seeded empty (m=-inf, l=0).
// Optional additive [kv_len] mask. K/V are the (strided) KV-cache narrow - pass
// per-dim element strides; innermost (hd) dim must be contiguous.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define LL_MAX_DPL 8   // supports head_dim up to 256 (8 dims/lane * 32 lanes)

extern "C" __global__ void loken_flash_decode_f16_kernel(
    const __half* __restrict__ Q,      // [b, n_head, hd] (contiguous, seq=1)
    const __half* __restrict__ K,      // [b, n_kv, kv_len, hd] (strided)
    const __half* __restrict__ V,      // [b, n_kv, kv_len, hd] (strided)
    const float*  __restrict__ mask,   // [kv_len] additive, or nullptr
    const float*  __restrict__ sinks,  // [n_head], or nullptr
    __half* __restrict__ out,          // [b, n_head, hd] (contiguous)
    int n_head, int n_kv, int kv_len, int head_dim, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride
) {
    const int head = blockIdx.x;
    const int bi   = blockIdx.y;
    const int lane = threadIdx.x;            // 0..31
    if (lane >= 32) return;
    const int dpl    = head_dim >> 5;        // dims per lane = head_dim / 32
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;

    const __half* q = Q + ((long)bi * n_head + head) * head_dim;
    float qd[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) qd[t] = (t < dpl) ? __half2float(q[lane + (t << 5)]) : 0.0f;

    const __half* kh = K + (long)bi * k_batch_stride + (long)kvh * k_head_stride;
    const __half* vh = V + (long)bi * v_batch_stride + (long)kvh * v_head_stride;

    // Seed online softmax: with sink, the virtual key (logit=sink, value=0); else empty.
    float m   = sinks ? sinks[head] : -INFINITY;
    float l   = sinks ? 1.0f : 0.0f;
    float acc[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) acc[t] = 0.0f;

    for (int j = 0; j < kv_len; ++j) {
        const __half* kj = kh + (long)j * k_pos_stride;
        float p = 0.0f;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t)
            if (t < dpl) p += qd[t] * __half2float(kj[lane + (t << 5)]);
        // warp reduce-sum the dot product over the 32 lanes
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
        p = __shfl_sync(0xffffffff, p, 0);   // broadcast lane 0's full sum
        float s = scale * p + (mask ? mask[j] : 0.0f);

        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);       // rescale prior accumulators
        float pj    = expf(s - m_new);
        l = l * alpha + pj;
        const __half* vj = vh + (long)j * v_pos_stride;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t)
            if (t < dpl) acc[t] = acc[t] * alpha + pj * __half2float(vj[lane + (t << 5)]);
        m = m_new;
    }

    const float inv = 1.0f / l;
    __half* o = out + ((long)bi * n_head + head) * head_dim;
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t)
        if (t < dpl) o[lane + (t << 5)] = __float2half(acc[t] * inv);
}

// Split-K flash-decode for hd up to 256 (mirror of loken_flash_decode_f16_kernel
// but partitioned over kv): the single-warp kernel above scans the WHOLE kv_len
// serially (one warp/head -> GPU idle at long ctx - qwen3.5 is launch-bound, ~20-29%
// util). Split-K partitions kv_len into `nsplit` chunks: PASS 1 computes a partial
// online-softmax (m,l,acc) per (head,batch,chunk), grid n_headxbatchxnsplit so the KV
// scan is parallel across the sequence; PASS 2 reduces the partials per head with the
// log-sum-exp merge. Mathematically identical to the serial kernel (modulo f32 add
// order). Same GQA + optional [kv_len] mask + optional per-head sink (seeded ONCE, on
// chunk 0, so the combine counts it exactly once).
extern "C" __global__ void loken_flash_decode_split_f16_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    const float* __restrict__ mask, const float* __restrict__ sinks,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    int n_head, int n_kv, int kv_len, int nsplit, int head_dim, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride
) {
    const int head  = blockIdx.x;
    const int bi    = blockIdx.y;
    const int split = blockIdx.z;
    const int lane  = threadIdx.x;
    if (lane >= 32) return;
    const int dpl    = head_dim >> 5;
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;
    const int split_len = (kv_len + nsplit - 1) / nsplit;
    const int j0 = split * split_len;
    const int j1 = min(j0 + split_len, kv_len);
    const long pidx = ((long)bi * n_head + head) * nsplit + split;
    if (j0 >= kv_len) {                          // empty chunk: neutral partial
        if (lane == 0) { part_m[pidx] = -INFINITY; part_l[pidx] = 0.0f; }
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) part_acc[pidx * head_dim + lane + (t << 5)] = 0.0f;
        return;
    }
    const __half* q = Q + ((long)bi * n_head + head) * head_dim;
    float qd[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) qd[t] = (t < dpl) ? __half2float(q[lane + (t << 5)]) : 0.0f;
    const __half* kh = K + (long)bi * k_batch_stride + (long)kvh * k_head_stride;
    const __half* vh = V + (long)bi * v_batch_stride + (long)kvh * v_head_stride;
    float m = (sinks && split == 0) ? sinks[head] : -INFINITY;
    float l = (sinks && split == 0) ? 1.0f : 0.0f;
    float acc[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) acc[t] = 0.0f;
    for (int j = j0; j < j1; ++j) {
        const __half* kj = kh + (long)j * k_pos_stride;
        float p = 0.0f;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) p += qd[t] * __half2float(kj[lane + (t << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
        p = __shfl_sync(0xffffffff, p, 0);
        float s = scale * p + (mask ? mask[j] : 0.0f);
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float pj = expf(s - m_new);
        l = l * alpha + pj;
        const __half* vj = vh + (long)j * v_pos_stride;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) acc[t] = acc[t] * alpha + pj * __half2float(vj[lane + (t << 5)]);
        m = m_new;
    }
    if (lane == 0) { part_m[pidx] = m; part_l[pidx] = l; }
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) part_acc[pidx * head_dim + lane + (t << 5)] = acc[t];
}

extern "C" __global__ void loken_flash_decode_combine_f16_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, __half* __restrict__ out,
    int n_head, int nsplit, int head_dim
) {
    const int head = blockIdx.x;
    const int bi   = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int dpl  = head_dim >> 5;
    const long base = ((long)bi * n_head + head) * nsplit;
    float gm = -INFINITY;
    for (int s = 0; s < nsplit; ++s) gm = fmaxf(gm, part_m[base + s]);
    float l = 0.0f;
    float acc[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) acc[t] = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float pm = part_m[base + s];
        if (pm == -INFINITY) continue;           // empty chunk
        const float w = expf(pm - gm);
        l += part_l[base + s] * w;
        const float* pa = part_acc + (base + s) * head_dim;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) acc[t] += pa[lane + (t << 5)] * w;
    }
    const float inv = 1.0f / l;
    __half* o = out + ((long)bi * n_head + head) * head_dim;
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) o[lane + (t << 5)] = __float2half(acc[t] * inv);
}

extern "C" void loken_flash_decode_split_f16(
    const void* Q, const void* K, const void* V, const float* mask, const float* sinks,
    void* out, float* part_m, float* part_l, float* part_acc,
    int batch, int n_head, int n_kv, int kv_len, int nsplit, int head_dim, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride, cudaStream_t stream
) {
    if (head_dim < 32 || (head_dim & 31) != 0 || (head_dim >> 5) > LL_MAX_DPL) return;
    dim3 grid(n_head, batch, nsplit);
    dim3 block(32, 1, 1);
    loken_flash_decode_split_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, mask, sinks,
        part_m, part_l, part_acc, n_head, n_kv, kv_len, nsplit, head_dim, scale,
        k_batch_stride, k_head_stride, k_pos_stride, v_batch_stride, v_head_stride, v_pos_stride);
    dim3 cgrid(n_head, batch, 1);
    loken_flash_decode_combine_f16_kernel<<<cgrid, block, 0, stream>>>(
        part_m, part_l, part_acc, (__half*)out, n_head, nsplit, head_dim);
}

// Capture-safe embedding gather: out[i] = table[(*tok)*d_model + i], reading the
// token id from a DEVICE buffer at kernel-execution time. A tensor-level index_select
// BAKES the gather index at CUDA-graph capture (host-reads the index), so the
// captured embed ignores input_tok on replay - this reads *tok on replay so the
// input token actually advances. table [vocab, d_model] F16; tok u32 [1]; out
// [d_model] F16.
extern "C" __global__ void loken_embed_gather_f16_kernel(
    const __half* __restrict__ table, const unsigned int* __restrict__ tok,
    __half* __restrict__ out, int d_model
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_model) return;
    out[i] = table[(long)(*tok) * d_model + i];
}
extern "C" void loken_embed_gather_f16(
    const void* table, const unsigned int* tok, void* out, int d_model, cudaStream_t stream
) {
    const int block = 256;
    const int grid = (d_model + block - 1) / block;
    loken_embed_gather_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)table, tok, (__half*)out, d_model);
}

// In-place device scalar writes (1 thread). Run OUTSIDE the captured graph region
// to advance the input token + position between replays (the value is a kernel arg,
// so it is NOT baked into a captured graph - these are launched before graph.launch).
// Device-to-device copy of one u32 (the new token id) into the fixed input_tok
// buffer - avoids a D2H read+sync per decode token in the graph path.
extern "C" __global__ void loken_copy_u32_kernel(const unsigned int* src, unsigned int* dst) { *dst = *src; }
extern "C" void loken_copy_u32(const unsigned int* src, unsigned int* dst, cudaStream_t s) {
    loken_copy_u32_kernel<<<1, 1, 0, s>>>(src, dst);
}
extern "C" __global__ void loken_set_i32_kernel(int* p, int v) { *p = v; }
extern "C" void loken_set_i32(int* p, int v, cudaStream_t s) {
    loken_set_i32_kernel<<<1, 1, 0, s>>>(p, v);
}
// Device-to-device copy of N floats into a fixed buffer. Used as the FINAL op of
// the captured gpt-oss decode forward: the lm_head writes its logits into a fresh
// (arena) tensor, which a captured graph cannot be relied on to keep stable across
// replays; copying into a persistent buffer INSIDE the captured region means each
// replay deposits the logits at a known address the host reads.
extern "C" __global__ void loken_copy_f32_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
extern "C" void loken_copy_f32(const float* src, float* dst, int n, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    loken_copy_f32_kernel<<<blocks, threads, 0, s>>>(src, dst, n);
}

// Device-position KV-cache write (for CUDA-graph decode): scatter the new token's
// k_new/v_new [b, n_kv, hd] into the fixed-max ring buffers k_buf/v_buf
// [b, n_kv, kv_max, hd] at slot `*pos_dev`. The position lives on-device so a
// captured graph advances it on replay (bump *pos_dev between replays) instead of
// a host-position slice_set, which freezes in a graph. One thread/element.
// k_new/v_new are [b, n_kv, seq, hd]; write each of the `seq` positions to slot
// (*pos_dev + s) - so this serves BOTH prefill (seq>1, populates the prompt KV)
// and decode (seq==1). Handles the base position on-device for graph replay.
extern "C" __global__ void loken_kv_write_at_pos_f16_kernel(
    const __half* __restrict__ k_new, const __half* __restrict__ v_new,
    __half* __restrict__ k_buf, __half* __restrict__ v_buf,
    const int* __restrict__ pos_dev, int b, int n_kv, int seq, int hd, int kv_max
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)b * n_kv * seq * hd;
    if (idx >= total) return;
    const int pos = *pos_dev;
    const int d = (int)(idx % hd);
    const long tmp = idx / hd;                       // flattened (b, n_kv, seq)
    const int s = (int)(tmp % seq);
    const long bn = tmp / seq;                        // flattened (b, n_kv)
    const long src = (bn * seq + s) * hd + d;         // k_new [b,n_kv,seq,hd]
    const long dst = bn * (long)kv_max * hd + (long)(pos + s) * hd + d; // k_buf [b,n_kv,kv_max,hd]
    k_buf[dst] = k_new[src];
    v_buf[dst] = v_new[src];
}

extern "C" void loken_kv_write_at_pos_f16(
    const void* k_new, const void* v_new, void* k_buf, void* v_buf,
    const int* pos_dev, int b, int n_kv, int seq, int hd, int kv_max, cudaStream_t stream
) {
    const int block = 256;
    const long total = (long)b * n_kv * seq * hd;
    const long grid = (total + block - 1) / block;
    loken_kv_write_at_pos_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)k_new, (const __half*)v_new, (__half*)k_buf, (__half*)v_buf,
        pos_dev, b, n_kv, seq, hd, kv_max);
}

// Device-kv_len variant for CUDA-graph decode: kv_len = *pos_dev + 1, read on
// device, so a captured graph attends the correct (growing) number of KV slots on
// replay instead of a host-frozen count. K/V are the FULL ring buffer
// [b, n_kv, kv_max, hd] (pos_stride=hd, head_stride=kv_max*hd). No mask, no sinks
// (lfm2). Otherwise identical to loken_flash_decode_f16 (bit-exact).
extern "C" __global__ void loken_flash_decode_devkvlen_f16_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    const int* __restrict__ pos_dev, __half* __restrict__ out,
    const float* __restrict__ sinks,   // [n_head] per-head attention sink logit, or nullptr
    int n_head, int n_kv, int head_dim, float scale, int window,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride
) {
    const int head = blockIdx.x;
    const int bi   = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int kv_len = *pos_dev + 1;            // device-resident count
    // Sliding-window attention (gpt-oss windowed layers): the query at the last
    // position sees only the most-recent `window` keys. window<=0 -> full attention.
    const int j_start = (window > 0 && kv_len > window) ? (kv_len - window) : 0;
    const int dpl    = head_dim >> 5;
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;
    const __half* q = Q + ((long)bi * n_head + head) * head_dim;
    float qd[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) qd[t] = (t < dpl) ? __half2float(q[lane + (t << 5)]) : 0.0f;
    const __half* kh = K + (long)bi * k_batch_stride + (long)kvh * k_head_stride;
    const __half* vh = V + (long)bi * v_batch_stride + (long)kvh * v_head_stride;
    // Seed the online softmax with the per-head sink as a virtual key (logit=sink,
    // value=0): m=sink, l=exp(sink-sink)=1, acc=0. Matches the reference
    // loken_flash_decode_f16_kernel. sinks==nullptr -> empty seed (lfm2).
    float m = sinks ? sinks[head] : -INFINITY;
    float l = sinks ? 1.0f : 0.0f;
    float acc[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) acc[t] = 0.0f;
    for (int j = j_start; j < kv_len; ++j) {
        const __half* kj = kh + (long)j * k_pos_stride;
        float p = 0.0f;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) p += qd[t] * __half2float(kj[lane + (t << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
        p = __shfl_sync(0xffffffff, p, 0);
        float s = scale * p;
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float pj = expf(s - m_new);
        l = l * alpha + pj;
        const __half* vj = vh + (long)j * v_pos_stride;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) acc[t] = acc[t] * alpha + pj * __half2float(vj[lane + (t << 5)]);
        m = m_new;
    }
    const float inv = 1.0f / l;
    __half* o = out + ((long)bi * n_head + head) * head_dim;
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) o[lane + (t << 5)] = __float2half(acc[t] * inv);
}

extern "C" void loken_flash_decode_devkvlen_f16(
    const void* Q, const void* K, const void* V, const int* pos_dev, void* out,
    const float* sinks,
    int batch, int n_head, int n_kv, int head_dim, float scale, int window,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride, cudaStream_t stream
) {
    if (head_dim < 32 || (head_dim & 31) != 0 || (head_dim >> 5) > LL_MAX_DPL) return;
    dim3 grid(n_head, batch, 1);
    dim3 block(32, 1, 1);
    loken_flash_decode_devkvlen_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, pos_dev, (__half*)out,
        sinks,
        n_head, n_kv, head_dim, scale, window,
        k_batch_stride, k_head_stride, k_pos_stride, v_batch_stride, v_head_stride, v_pos_stride);
}

extern "C" void loken_flash_decode_f16(
    const void* Q, const void* K, const void* V, const float* mask, const float* sinks,
    void* out, int batch, int n_head, int n_kv, int kv_len, int head_dim, float scale,
    long k_batch_stride, long k_head_stride, long k_pos_stride,
    long v_batch_stride, long v_head_stride, long v_pos_stride,
    cudaStream_t stream
) {
    // head_dim must be a multiple of 32 and <= 32*LL_MAX_DPL (256); caller falls back otherwise.
    if (head_dim < 32 || (head_dim & 31) != 0 || (head_dim >> 5) > LL_MAX_DPL) return;
    dim3 grid(n_head, batch, 1);
    dim3 block(32, 1, 1);
    loken_flash_decode_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, mask, sinks, (__half*)out,
        n_head, n_kv, kv_len, head_dim, scale,
        k_batch_stride, k_head_stride, k_pos_stride,
        v_batch_stride, v_head_stride, v_pos_stride);
}

// -- PAGED flash-decode: reads K/V from a flat paged store [num_slots, feat]
// (feat = n_kv*head_dim) via a per-seq block_table + seq_lens, ALL read from
// DEVICE buffers at kernel-exec time -> CUDA-graph-replay-safe (no host-baked
// index, unlike a tensor-level index_select). One warp per (head, batch). The paged
// gather and the attention are fused - no separate gather tensor. ------------
extern "C" __global__ void loken_paged_flash_decode_f16_kernel(
    const __half* __restrict__ Q,          // [batch, n_head, head_dim]
    const __half* __restrict__ Kp,         // [num_slots, feat]
    const __half* __restrict__ Vp,         // [num_slots, feat]
    const int* __restrict__ block_table,   // [batch, max_blocks]
    const int* __restrict__ seq_lens,      // [batch]
    __half* __restrict__ out,              // [batch, n_head, head_dim]
    int n_head, int n_kv, int head_dim, float scale,
    int block_size, int max_blocks, int feat
) {
    const int head = blockIdx.x;
    const int bi   = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int kv_len = seq_lens[bi];
    const int dpl    = head_dim >> 5;
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;
    const int hoff   = kvh * head_dim;              // kv-head offset within a slot
    const int* bt    = block_table + (long)bi * max_blocks;
    const __half* q  = Q + ((long)bi * n_head + head) * head_dim;
    float qd[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) qd[t] = (t < dpl) ? __half2float(q[lane + (t << 5)]) : 0.0f;
    float m = -INFINITY, l = 0.0f;
    float acc[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) acc[t] = 0.0f;
    for (int j = 0; j < kv_len; ++j) {
        const int slot = bt[j / block_size] * block_size + (j % block_size);
        const __half* kj = Kp + (long)slot * feat + hoff;
        float p = 0.0f;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) p += qd[t] * __half2float(kj[lane + (t << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
        p = __shfl_sync(0xffffffff, p, 0);
        float s = scale * p;
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float pj = expf(s - m_new);
        l = l * alpha + pj;
        const __half* vj = Vp + (long)slot * feat + hoff;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) acc[t] = acc[t] * alpha + pj * __half2float(vj[lane + (t << 5)]);
        m = m_new;
    }
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    __half* o = out + ((long)bi * n_head + head) * head_dim;
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) o[lane + (t << 5)] = __float2half(acc[t] * inv);
}

extern "C" void loken_paged_flash_decode_f16(
    const void* Q, const void* Kp, const void* Vp, const int* block_table, const int* seq_lens,
    void* out, int batch, int n_head, int n_kv, int head_dim, float scale,
    int block_size, int max_blocks, int feat, cudaStream_t stream
) {
    if (head_dim < 32 || (head_dim & 31) != 0 || (head_dim >> 5) > LL_MAX_DPL) return;
    dim3 grid(n_head, batch, 1);
    dim3 block(32, 1, 1);
    loken_paged_flash_decode_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)Kp, (const __half*)Vp, block_table, seq_lens, (__half*)out,
        n_head, n_kv, head_dim, scale, block_size, max_blocks, feat);
}

// -- PAGED SPLIT flash-decode: same paged gather as above, but the KV scan is SPLIT
// across `nsplit` blocks in z (grid n_headxbatchxnsplit) so a single low-batch decode
// FILLS the GPU instead of running just n_head warps serially over the whole sequence
// (the batch=1 long-context collapse). Online-softmax partials per (head,batch,split)
// -> reduced by the shared `loken_flash_decode_combine_f16_kernel` (identical
// partial layout). `nsplit` is constant per CUDA-graph capture; the per-split KV range
// is derived from the on-device `seq_lens` at exec time, so it stays correct as the
// sequence grows across graph replays. ----------------------------------------------
extern "C" __global__ void loken_paged_flash_decode_split_f16_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ Kp,
    const __half* __restrict__ Vp,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    int n_head, int n_kv, int nsplit, int head_dim, float scale,
    int block_size, int max_blocks, int feat
) {
    const int head  = blockIdx.x;
    const int bi    = blockIdx.y;
    const int split = blockIdx.z;
    const int lane  = threadIdx.x;
    if (lane >= 32) return;
    const int kv_len = seq_lens[bi];
    const int dpl    = head_dim >> 5;
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;
    const int hoff   = kvh * head_dim;
    const int split_len = (kv_len + nsplit - 1) / nsplit;
    const int j0 = split * split_len;
    const int j1 = min(j0 + split_len, kv_len);
    const long pidx = ((long)bi * n_head + head) * nsplit + split;
    if (j0 >= kv_len) {                          // empty chunk: neutral partial
        if (lane == 0) { part_m[pidx] = -INFINITY; part_l[pidx] = 0.0f; }
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) part_acc[pidx * head_dim + lane + (t << 5)] = 0.0f;
        return;
    }
    const int* bt = block_table + (long)bi * max_blocks;
    const __half* q = Q + ((long)bi * n_head + head) * head_dim;
    float qd[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) qd[t] = (t < dpl) ? __half2float(q[lane + (t << 5)]) : 0.0f;
    float m = -INFINITY, l = 0.0f;
    float acc[LL_MAX_DPL];
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) acc[t] = 0.0f;
    for (int j = j0; j < j1; ++j) {
        const int slot = bt[j / block_size] * block_size + (j % block_size);
        const __half* kj = Kp + (long)slot * feat + hoff;
        float p = 0.0f;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) p += qd[t] * __half2float(kj[lane + (t << 5)]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffff, p, o);
        p = __shfl_sync(0xffffffff, p, 0);
        float s = scale * p;
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float pj = expf(s - m_new);
        l = l * alpha + pj;
        const __half* vj = Vp + (long)slot * feat + hoff;
        #pragma unroll
        for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) acc[t] = acc[t] * alpha + pj * __half2float(vj[lane + (t << 5)]);
        m = m_new;
    }
    if (lane == 0) { part_m[pidx] = m; part_l[pidx] = l; }
    #pragma unroll
    for (int t = 0; t < LL_MAX_DPL; ++t) if (t < dpl) part_acc[pidx * head_dim + lane + (t << 5)] = acc[t];
}

extern "C" void loken_paged_flash_decode_split_f16(
    const void* Q, const void* Kp, const void* Vp, const int* block_table, const int* seq_lens,
    void* out, float* part_m, float* part_l, float* part_acc,
    int batch, int n_head, int n_kv, int nsplit, int head_dim, float scale,
    int block_size, int max_blocks, int feat, cudaStream_t stream
) {
    if (head_dim < 32 || (head_dim & 31) != 0 || (head_dim >> 5) > LL_MAX_DPL) return;
    dim3 grid(n_head, batch, nsplit);
    dim3 block(32, 1, 1);
    loken_paged_flash_decode_split_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)Kp, (const __half*)Vp, block_table, seq_lens,
        part_m, part_l, part_acc, n_head, n_kv, nsplit, head_dim, scale,
        block_size, max_blocks, feat);
    dim3 cgrid(n_head, batch, 1);
    loken_flash_decode_combine_f16_kernel<<<cgrid, block, 0, stream>>>(
        part_m, part_l, part_acc, (__half*)out, n_head, nsplit, head_dim);
}

// -- PAGED KV write: scatter B new tokens' K/V (each [feat]) into the flat paged
// store at slot_dev[bi] (slot read from a DEVICE buffer -> capture-safe scatter,
// unlike a tensor-level scatter_set, which host-bakes the index). grid(batch), one block. -
extern "C" __global__ void loken_paged_kv_write_f16_kernel(
    const __half* __restrict__ k_new,  // [batch, feat]
    const __half* __restrict__ v_new,  // [batch, feat]
    __half* __restrict__ Kp,           // [num_slots, feat]
    __half* __restrict__ Vp,           // [num_slots, feat]
    const int* __restrict__ slot_dev,  // [batch]
    int feat
) {
    const int bi = blockIdx.x;
    const int slot = slot_dev[bi];
    const __half* ks = k_new + (long)bi * feat;
    const __half* vs = v_new + (long)bi * feat;
    __half* kd = Kp + (long)slot * feat;
    __half* vd = Vp + (long)slot * feat;
    for (int i = threadIdx.x; i < feat; i += blockDim.x) { kd[i] = ks[i]; vd[i] = vs[i]; }
}

extern "C" void loken_paged_kv_write_f16(
    const void* k_new, const void* v_new, void* Kp, void* Vp, const int* slot_dev,
    int batch, int feat, cudaStream_t stream
) {
    dim3 grid(batch, 1, 1);
    dim3 block(256, 1, 1);
    loken_paged_kv_write_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)k_new, (const __half*)v_new, (__half*)Kp, (__half*)Vp, slot_dev, feat);
}
