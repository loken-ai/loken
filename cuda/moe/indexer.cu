// Block-sparse attention on the card for one decoded token: the indexer's cache, its selection of
// blocks, and the attention over the selected tokens.
//
// The indexer keeps, for each complete block of `block` tokens, one pooled key: the mean of the
// block's rows, RMS-normalised, scaled by gamma and rotated at the block's first position. A query
// scores every complete block with the rectified dot products of its heads and keeps the best
// `budget` blocks, the lowest index first among equal scores, plus the tokens of the incomplete
// block. The arithmetic below follows the host's, operation by operation and in the same order,
// each one rounded on its own, so the card keeps the blocks the host keeps.
//
//   append   a token's key joins the tail rows of its block; a completed block is pooled
//   select   the heads' queries are normalised and rotated, every complete block scored, the
//            best kept, and the kept tokens written in ascending order with their count
//   attend   one block per query head reads the kept tokens' keys and values from the cache
//
// The position is read from the card (`pos_dev`) when there is a pointer, so a recorded step replays
// at every position, and is `pos_host` otherwise.

#include <cuda_fp16.h>
#include <stddef.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub(float a, float b) { return __fsub_rn(a, b); }

// The order of floats the host's total_cmp gives, as unsigned integers.
__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Normalise `row` (length hd) by its RMS, scale by gamma, in the host's order.
__device__ __forceinline__ void rms_norm_row(float * row, const float * gamma, int hd, float eps) {
    float ss = 0.f;
    for (int d = 0; d < hd; ++d) ss = add(ss, mul(row[d], row[d]));
    const float ms = __fdiv_rn(ss, (float) hd);
    const float inv = __fdiv_rn(1.f, __fsqrt_rn(add(ms, eps)));
    for (int d = 0; d < hd; ++d) row[d] = mul(mul(row[d], inv), gamma[d]);
}

// Rotate the first 2 * half lanes of `row` by the tables cos[half], sin[half].
__device__ __forceinline__ void rotate_row(float * row, const float * rot, int half) {
    for (int i = 0; i < half; ++i) {
        const float c = rot[i], s = rot[half + i];
        const float a = row[i], b = row[i + half];
        row[i] = sub(mul(a, c), mul(b, s));
        row[i + half] = add(mul(b, c), mul(a, s));
    }
}

} // namespace

// One block. `key` is the token's raw key row (hd floats), `tail` the raw rows of the block being
// filled (block * hd), `pooled` the pooled keys (blocks * hd). `rot` holds the cos and sin tables
// at the first position of the block this token completes; it is read only then.
extern "C" __global__ void loken_indexer_append_kernel(
        const float * __restrict__ key, float * __restrict__ tail, float * __restrict__ pooled,
        const float * __restrict__ gamma, const float * __restrict__ rot,
        const int * __restrict__ pos_dev, int pos_host, int hd, int block, int half, float eps) {
    extern __shared__ float row[];
    const int pos = pos_dev ? *pos_dev : pos_host;
    const int slot = pos % block;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) tail[(size_t) slot * hd + d] = key[d];
    __syncthreads();
    if (slot != block - 1) return;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float p = 0.f;
        for (int t = 0; t < block; ++t) p = add(p, tail[(size_t) t * hd + d]);
        row[d] = __fdiv_rn(p, (float) block);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        rms_norm_row(row, gamma, hd, eps);
        rotate_row(row, rot, half);
    }
    __syncthreads();
    const size_t at = (size_t) (pos / block) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) pooled[at + d] = row[d];
}

extern "C" void loken_indexer_append(
        const float * key, float * tail, float * pooled, const float * gamma, const float * rot,
        const int * pos_dev, int pos_host, int hd, int block, int half, float eps,
        cudaStream_t stream) {
    loken_indexer_append_kernel<<<1, 128, hd * sizeof(float), stream>>>(
        key, tail, pooled, gamma, rot, pos_dev, pos_host, hd, block, half, eps);
}

// One block of 1024 threads. `q` is the query's raw heads (heads * hd), `rot` the cos and sin
// tables at the query's position, `pooled` the pooled keys, `scores` scratch for one score per
// block. Writes the kept tokens, ascending, to `sel` and their number to `count`.
extern "C" __global__ void loken_indexer_select_kernel(
        const float * __restrict__ q, const float * __restrict__ qgamma,
        const float * __restrict__ rot, const float * __restrict__ pooled,
        float * __restrict__ scores, uint32_t * __restrict__ sel, uint32_t * __restrict__ count,
        const int * __restrict__ pos_dev, int pos_host, int heads, int hd, int block, int budget,
        int half, float eps, float scale) {
    extern __shared__ float qn[];                 // heads * hd
    __shared__ uint32_t hist[256];
    __shared__ int scan_a[1024], scan_b[1024];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_remaining;
    const int nt = blockDim.x;
    const int t = threadIdx.x;
    const int pos = pos_dev ? *pos_dev : pos_host;
    const int visible = pos + 1;
    const int complete = visible / block;
    const int take = complete < budget ? complete : budget;

    // the query heads, normalised and rotated
    for (int h = t; h < heads; h += nt) {
        float * row = qn + (size_t) h * hd;
        for (int d = 0; d < hd; ++d) row[d] = q[(size_t) h * hd + d];
        rms_norm_row(row, qgamma, hd, eps);
        rotate_row(row, rot, half);
    }
    __syncthreads();

    // the score of every complete block
    for (int b = t; b < complete; b += nt) {
        const float * k = pooled + (size_t) b * hd;
        float total = 0.f;
        for (int h = 0; h < heads; ++h) {
            const float * qh = qn + (size_t) h * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d) dot = add(dot, mul(qh[d], k[d]));
            total = add(total, dot > 0.f ? dot : 0.f);
        }
        scores[b] = mul(total, scale);
    }
    __syncthreads();

    // the take-th largest score by radix selection, and how many blocks of that score are kept
    uint32_t threshold = 0;
    int need_eq = 0;
    const bool partial = take < complete;
    if (partial) {
        if (t == 0) { s_prefix = 0; s_mask = 0; s_remaining = take; }
        __syncthreads();
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = t; i < 256; i += nt) hist[i] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix, mask = s_mask;
            for (int b = t; b < complete; b += nt) {
                const uint32_t k = order_key(scores[b]);
                if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (t == 0) {
                int cum = 0, rem = s_remaining;
                int chosen = 0;
                for (int bin = 255; bin >= 0; --bin) {
                    const int c = (int) hist[bin];
                    if (cum + c >= rem) { chosen = bin; break; }
                    cum += c;
                }
                s_remaining = rem - cum;
                s_prefix = prefix | ((uint32_t) chosen << shift);
                s_mask = mask | (0xFFu << shift);
            }
            __syncthreads();
        }
        threshold = s_prefix;
        need_eq = s_remaining;
    }

    // each thread owns a run of blocks: count what it keeps, scan the counts, write the tokens
    const int chunk = (complete + nt - 1) / nt;
    const int lo = t * chunk;
    int hi = lo + chunk;
    if (hi > complete) hi = complete;
    int eq_here = 0;
    if (partial) {
        for (int b = lo; b < hi; ++b) if (order_key(scores[b]) == threshold) ++eq_here;
    }
    scan_a[t] = eq_here;
    __syncthreads();
    for (int off = 1; off < nt; off <<= 1) {
        const int v = t >= off ? scan_a[t - off] : 0;
        __syncthreads();
        scan_a[t] += v;
        __syncthreads();
    }
    int eq_rank = scan_a[t] - eq_here;
    int kept_here = 0;
    for (int b = lo; b < hi; ++b) {
        bool keep = true;
        if (partial) {
            const uint32_t k = order_key(scores[b]);
            keep = k > threshold;
            if (k == threshold) { keep = eq_rank < need_eq; ++eq_rank; }
        }
        if (keep) ++kept_here;
    }
    scan_b[t] = kept_here;
    __syncthreads();
    for (int off = 1; off < nt; off <<= 1) {
        const int v = t >= off ? scan_b[t - off] : 0;
        __syncthreads();
        scan_b[t] += v;
        __syncthreads();
    }
    int out = scan_b[t] - kept_here;
    eq_rank = scan_a[t] - eq_here;
    for (int b = lo; b < hi; ++b) {
        bool keep = true;
        if (partial) {
            const uint32_t k = order_key(scores[b]);
            keep = k > threshold;
            if (k == threshold) { keep = eq_rank < need_eq; ++eq_rank; }
        }
        if (keep) {
            for (int j = 0; j < block; ++j) sel[(size_t) out * block + j] = (uint32_t) (b * block + j);
            ++out;
        }
    }
    __syncthreads();
    // the tokens of the incomplete block follow
    const int tail = visible - complete * block;
    for (int j = t; j < tail; j += nt) sel[(size_t) take * block + j] = (uint32_t) (complete * block + j);
    if (t == 0) *count = (uint32_t) (take * block + tail);
}

extern "C" void loken_indexer_select(
        const float * q, const float * qgamma, const float * rot, const float * pooled,
        float * scores, uint32_t * sel, uint32_t * count, const int * pos_dev, int pos_host,
        int heads, int hd, int block, int budget, int half, float eps, float scale,
        cudaStream_t stream) {
    loken_indexer_select_kernel<<<1, 1024, heads * hd * sizeof(float), stream>>>(
        q, qgamma, rot, pooled, scores, sel, count, pos_dev, pos_host, heads, hd, block, budget,
        half, eps, scale);
}

// The token's key and value rows into the cache at the position the card holds. `kbuf` and `vbuf`
// are `[n_kv, cap, hd]` F16.
extern "C" __global__ void loken_kv_append_pos_kernel(
        const float * __restrict__ k_new, const float * __restrict__ v_new,
        __half * __restrict__ kbuf, __half * __restrict__ vbuf, const int * __restrict__ pos_dev,
        int pos_host, int n_kv, int cap, int hd) {
    const int pos = pos_dev ? *pos_dev : pos_host;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_kv * hd; i += gridDim.x * blockDim.x) {
        const int h = i / hd, d = i % hd;
        const size_t at = ((size_t) h * cap + pos) * hd + d;
        kbuf[at] = __float2half(k_new[i]);
        vbuf[at] = __float2half(v_new[i]);
    }
}

extern "C" void loken_kv_append_pos(
        const float * k_new, const float * v_new, void * kbuf, void * vbuf, const int * pos_dev,
        int pos_host, int n_kv, int cap, int hd, cudaStream_t stream) {
    const int n = n_kv * hd;
    loken_kv_append_pos_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        k_new, v_new, (__half *) kbuf, (__half *) vbuf, pos_dev, pos_host, n_kv, cap, hd);
}

// Attention of one decoded token over the kept tokens: one block per query head, a warp per token.
// `q` is `[n_head, hd]` F32, already rotated; `kbuf` and `vbuf` are `[n_kv, cap, hd]` F16; the
// kept tokens are `sel[0 .. *count]`. The dynamic shared memory holds the scores (`ncap`), the
// query and one partial output per warp.
extern "C" __global__ void loken_selected_attn_kernel(
        const float * __restrict__ q, const __half * __restrict__ kbuf,
        const __half * __restrict__ vbuf, const uint32_t * __restrict__ sel,
        const uint32_t * __restrict__ count, float * __restrict__ out, int n_head, int n_kv,
        int cap, int hd, int ncap, float scale) {
    extern __shared__ float smem[];
    float * scores = smem;
    float * qs = smem + ncap;
    float * part = qs + hd;
    const int h = blockIdx.x;
    const int kvh = h / (n_head / n_kv);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const int n = (int) *count;
    const int nvec = hd / 8;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = q[(size_t) h * hd + d];
    __syncthreads();
    const __half * kh = kbuf + (size_t) kvh * cap * hd;
    const __half * vh = vbuf + (size_t) kvh * cap * hd;
    for (int j = warp; j < n; j += nwarps) {
        const uint4 * row = (const uint4 *) (kh + (size_t) sel[j] * hd);
        float dot = 0.f;
        for (int v = lane; v < nvec; v += 32) {
            const uint4 w = row[v];
            const __half2 * hp = (const __half2 *) &w;
            const float * qv = qs + v * 8;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(hp[i]);
                dot += f.x * qv[2 * i] + f.y * qv[2 * i + 1];
            }
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        if (lane == 0) scores[j] = dot * scale;
    }
    __syncthreads();
    // softmax over the kept scores
    __shared__ float red[32];
    float m = __int_as_float(0xff800000);
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, scores[j]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < nwarps; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float s = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = expf(scores[j] - m);
        scores[j] = e;
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    s = 0.f;
    for (int i = 0; i < nwarps; ++i) s += red[i];
    const float inv = 1.f / s;
    // the weighted values: each warp sums its tokens, the warps' partial sums are added in order
    float acc[4][8];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[a][i] = 0.f;
    for (int j = warp; j < n; j += nwarps) {
        const float p = scores[j] * inv;
        const uint4 * row = (const uint4 *) (vh + (size_t) sel[j] * hd);
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int v = lane + 32 * a;
            if (v < nvec) {
                const uint4 w = row[v];
                const __half2 * hp = (const __half2 *) &w;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(hp[i]);
                    acc[a][2 * i] += p * f.x;
                    acc[a][2 * i + 1] += p * f.y;
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int v = lane + 32 * a;
        if (v < nvec) {
#pragma unroll
            for (int i = 0; i < 8; ++i) part[warp * hd + v * 8 + i] = acc[a][i];
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.f;
        for (int i = 0; i < nwarps; ++i) o += part[i * hd + d];
        out[(size_t) h * hd + d] = o;
    }
}

extern "C" void loken_selected_attn(
        const float * q, const void * kbuf, const void * vbuf, const uint32_t * sel,
        const uint32_t * count, float * out, int n_head, int n_kv, int cap, int hd, int ncap,
        float scale, cudaStream_t stream) {
    const int threads = 256;
    const size_t smem = (size_t) (ncap + hd + (threads / 32) * hd) * sizeof(float);
    loken_selected_attn_kernel<<<n_head, threads, smem, stream>>>(
        q, (const __half *) kbuf, (const __half *) vbuf, sel, count, out, n_head, n_kv, cap, hd,
        ncap, scale);
}
