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
// The pooled keys are held in groups of 32 blocks, each group `[dim][lane]`: the 32 threads of a warp
// scoring 32 consecutive blocks read consecutive floats.
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
    const int blk = pos / block;
    const size_t at = (size_t) (blk / 32) * hd * 32 + (blk % 32);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) pooled[at + (size_t) d * 32] = row[d];
}

// The keys of `rows` consecutive positions joined in turn, in one launch: row `r` is at
// `pos_dev[r]`, its rotation at `rot + r * rot_stride`.
extern "C" __global__ void loken_indexer_append_rows_kernel(
        const float * __restrict__ key, float * __restrict__ tail, float * __restrict__ pooled,
        const float * __restrict__ gamma, const float * __restrict__ rot,
        const int * __restrict__ pos_dev, int rows, int hd, int block, int half, float eps,
        int rot_stride) {
    extern __shared__ float row[];
    for (int r = 0; r < rows; ++r) {
        const int pos = pos_dev[r];
        const int slot = pos % block;
        const float * k = key + (size_t) r * hd;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) tail[(size_t) slot * hd + d] = k[d];
        __syncthreads();
        if (slot == block - 1) {
            for (int d = threadIdx.x; d < hd; d += blockDim.x) {
                float p = 0.f;
                for (int t = 0; t < block; ++t) p = add(p, tail[(size_t) t * hd + d]);
                row[d] = __fdiv_rn(p, (float) block);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                rms_norm_row(row, gamma, hd, eps);
                rotate_row(row, rot + (size_t) r * rot_stride, half);
            }
            __syncthreads();
            const int blk = pos / block;
            const size_t at = (size_t) (blk / 32) * hd * 32 + (blk % 32);
            for (int d = threadIdx.x; d < hd; d += blockDim.x) pooled[at + (size_t) d * 32] = row[d];
            __syncthreads();
        }
    }
}

extern "C" void loken_indexer_append_rows(
        const float * key, float * tail, float * pooled, const float * gamma, const float * rot,
        const int * pos_dev, int rows, int hd, int block, int half, float eps, int rot_stride,
        cudaStream_t stream) {
    loken_indexer_append_rows_kernel<<<1, 128, hd * sizeof(float), stream>>>(
        key, tail, pooled, gamma, rot, pos_dev, rows, hd, block, half, eps, rot_stride);
}

extern "C" void loken_indexer_append(
        const float * key, float * tail, float * pooled, const float * gamma, const float * rot,
        const int * pos_dev, int pos_host, int hd, int block, int half, float eps,
        cudaStream_t stream) {
    loken_indexer_append_kernel<<<1, 128, hd * sizeof(float), stream>>>(
        key, tail, pooled, gamma, rot, pos_dev, pos_host, hd, block, half, eps);
}

// The score of every complete block, in two launches: the first scores them, a thread to a block,
// across as many blocks of threads as the cache has room for; the second picks the best in one
// block of 1024 threads and writes the kept tokens, ascending, to `sel` and their number to
// `count`. `q` is the query's raw heads (heads * hd), `rot` the cos and sin tables at the query's
// position, `pooled` the pooled keys, `scores` one score per block.
//
// A thread takes the key of its block one value per step with its warp's reads side by side, and
// sums every head's dot product over the dimensions in order.
extern "C" __global__ void __launch_bounds__(128) loken_indexer_score_kernel(
        const float * __restrict__ q, const float * __restrict__ qgamma,
        const float * __restrict__ rot, const float * __restrict__ pooled,
        float * __restrict__ scores, const int * __restrict__ pos_dev, int pos_host, int heads,
        int hd, int block, int half, float eps, float scale, int rot_stride, int scores_stride) {
    extern __shared__ float qn[];                 // heads * hd
    const int t = threadIdx.x;
    // A launch over the rows of a prompt chunk has one row of blocks of threads per row: the row's
    // query, rotation tables and scores follow the previous row's, and its position is one more.
    const int row = blockIdx.y;
    q += (size_t) row * heads * hd;
    rot += (size_t) row * rot_stride;
    scores += (size_t) row * scores_stride;
    const int pos = (pos_dev ? *pos_dev : pos_host) + row;
    const int complete = (pos + 1) / block;
    if ((int) (blockIdx.x * blockDim.x) >= complete) return;
    // the query heads, normalised and rotated
    for (int h = t; h < heads; h += blockDim.x) {
        float * row = qn + (size_t) h * hd;
        for (int d = 0; d < hd; ++d) row[d] = q[(size_t) h * hd + d];
        rms_norm_row(row, qgamma, hd, eps);
        rotate_row(row, rot, half);
    }
    __syncthreads();
    const int b = blockIdx.x * blockDim.x + t;
    if (b >= complete) return;
    const float * k = pooled + (size_t) (b / 32) * hd * 32 + (b % 32);
    float dot[8];
#pragma unroll
    for (int h = 0; h < 8; ++h) dot[h] = 0.f;
    for (int d = 0; d < hd; ++d) {
        const float kd = k[(size_t) d * 32];
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            if (h < heads) dot[h] = add(dot[h], mul(qn[(size_t) h * hd + d], kd));
        }
    }
    float total = 0.f;
#pragma unroll
    for (int h = 0; h < 8; ++h) {
        if (h < heads) total = add(total, dot[h] > 0.f ? dot[h] : 0.f);
    }
    scores[b] = mul(total, scale);
}

extern "C" __global__ void __launch_bounds__(1024) loken_indexer_pick_kernel(
        const float * __restrict__ scores_in, uint32_t * __restrict__ sel_in,
        uint32_t * __restrict__ count_in, const int * __restrict__ pos_dev, int pos_host, int block,
        int budget, int scores_stride, int sel_stride) {
    const int row = blockIdx.x;
    const float * __restrict__ scores = scores_in + (size_t) row * scores_stride;
    uint32_t * __restrict__ sel = sel_in + (size_t) row * sel_stride;
    uint32_t * __restrict__ count = count_in + row;
    __shared__ uint32_t hist[256];
    __shared__ int scan_a[1024], scan_b[1024];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_remaining;
    const int nt = blockDim.x;
    const int t = threadIdx.x;
    const int pos = (pos_dev ? *pos_dev : pos_host) + row;
    const int visible = pos + 1;
    const int complete = visible / block;
    const int take = complete < budget ? complete : budget;

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
        int max_blocks, int rows, int rot_stride, int sel_stride, cudaStream_t stream) {
    loken_indexer_score_kernel<<<dim3((max_blocks + 127) / 128, rows), 128,
                                 heads * hd * sizeof(float), stream>>>(
        q, qgamma, rot, pooled, scores, pos_dev, pos_host, heads, hd, block, half, eps, scale,
        rot_stride, max_blocks);
    loken_indexer_pick_kernel<<<rows, 1024, 0, stream>>>(
        scores, sel, count, pos_dev, pos_host, block, budget, max_blocks, sel_stride);
}

// Rows `[n, hd]` row by row into the pooled keys' groups from block `first`, and back.
extern "C" __global__ void loken_indexer_rows_in_kernel(
        const float * __restrict__ rows, float * __restrict__ pooled, int first, int n, int hd) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * hd; i += gridDim.x * blockDim.x) {
        const int r = i / hd, d = i % hd;
        const int blk = first + r;
        pooled[(size_t) (blk / 32) * hd * 32 + (size_t) d * 32 + (blk % 32)] = rows[i];
    }
}

extern "C" __global__ void loken_indexer_rows_out_kernel(
        const float * __restrict__ pooled, float * __restrict__ rows, int first, int n, int hd) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * hd; i += gridDim.x * blockDim.x) {
        const int r = i / hd, d = i % hd;
        const int blk = first + r;
        rows[i] = pooled[(size_t) (blk / 32) * hd * 32 + (size_t) d * 32 + (blk % 32)];
    }
}

extern "C" void loken_indexer_rows_in(
        const float * rows, float * pooled, int first, int n, int hd, cudaStream_t stream) {
    const int total = n * hd;
    loken_indexer_rows_in_kernel<<<(total + 255) / 256, 256, 0, stream>>>(rows, pooled, first, n, hd);
}

extern "C" void loken_indexer_rows_out(
        const float * pooled, float * rows, int first, int n, int hd, cudaStream_t stream) {
    const int total = n * hd;
    loken_indexer_rows_out_kernel<<<(total + 255) / 256, 256, 0, stream>>>(pooled, rows, first, n, hd);
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

// `rows` rows of keys and values into the cache, row `r` at `pos_dev[r]`.
extern "C" __global__ void loken_kv_append_pos_rows_kernel(
        const float * __restrict__ k_new, const float * __restrict__ v_new,
        __half * __restrict__ kbuf, __half * __restrict__ vbuf, const int * __restrict__ pos_dev,
        int rows, int n_kv, int cap, int hd) {
    const int per = n_kv * hd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * per; i += gridDim.x * blockDim.x) {
        const int r = i / per, j = i % per;
        const int h = j / hd, d = j % hd;
        const int pos = pos_dev[r];
        const size_t at = ((size_t) h * cap + pos) * hd + d;
        kbuf[at] = __float2half(k_new[i]);
        vbuf[at] = __float2half(v_new[i]);
    }
}

extern "C" void loken_kv_append_pos_rows(
        const float * k_new, const float * v_new, void * kbuf, void * vbuf, const int * pos_dev,
        int rows, int n_kv, int cap, int hd, cudaStream_t stream) {
    const int n = rows * n_kv * hd;
    loken_kv_append_pos_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        k_new, v_new, (__half *) kbuf, (__half *) vbuf, pos_dev, rows, n_kv, cap, hd);
}

extern "C" void loken_kv_append_pos(
        const float * k_new, const float * v_new, void * kbuf, void * vbuf, const int * pos_dev,
        int pos_host, int n_kv, int cap, int hd, cudaStream_t stream) {
    const int n = n_kv * hd;
    loken_kv_append_pos_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        k_new, v_new, (__half *) kbuf, (__half *) vbuf, pos_dev, pos_host, n_kv, cap, hd);
}

// Attention of one decoded token over the kept tokens, in two launches. The kept tokens are cut
// into `splits` runs, and a block per (query head, run) scores its run, takes the run's softmax
// statistics and weighs its values, a warp at a time over four tokens at once so that the loads
// of one are in flight while the others are summed. The second launch joins the runs' answers.
// `q` is `[n_head, hd]` F32, already rotated; `kbuf` and `vbuf` are `[n_kv, cap, hd]` F16; the
// kept tokens are `sel[0 .. *count]`. The dynamic shared memory holds the run's scores, the query
// and one partial output per warp.
extern "C" __global__ void loken_selected_attn_split_kernel(
        const float * __restrict__ q, const __half * __restrict__ kbuf,
        const __half * __restrict__ vbuf, const uint32_t * __restrict__ sel,
        const uint32_t * __restrict__ count, float * __restrict__ part_o,
        float * __restrict__ part_ml, int n_head, int n_kv, int cap, int hd, int chunk_max,
        int splits, float scale, int sel_stride) {
    extern __shared__ float smem[];
    // Row `blockIdx.z` of a launch over several rows: its query, its kept tokens and its runs.
    {
        const size_t row = blockIdx.z;
        q += row * n_head * hd;
        sel += row * sel_stride;
        count += row;
        part_o += row * n_head * splits * hd;
        part_ml += row * n_head * splits * 2;
    }
    float * scores = smem;
    float * qs = smem + chunk_max;
    float * part = qs + hd;
    const int h = blockIdx.x, s = blockIdx.y;
    const int kvh = h / (n_head / n_kv);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const int n = (int) *count;
    const int chunk = (n + splits - 1) / splits;
    const int lo = s * chunk;
    const int cnt = n > lo ? (n - lo < chunk ? n - lo : chunk) : 0;
    const int nvec = hd / 8;
    const float neg_inf = __int_as_float(0xff800000);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = q[(size_t) h * hd + d];
    __syncthreads();
    const __half * kh = kbuf + (size_t) kvh * cap * hd;
    const __half * vh = vbuf + (size_t) kvh * cap * hd;
    // the scores of the run
    for (int base = warp * 4; base < cnt; base += nwarps * 4) {
        uint32_t ids[4];
        float dot[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            ids[u] = base + u < cnt ? sel[lo + base + u] : 0u;
            dot[u] = 0.f;
        }
        for (int v = lane; v < nvec; v += 32) {
            uint4 w[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) w[u] = ((const uint4 *) (kh + (size_t) ids[u] * hd))[v];
            const float * qv = qs + v * 8;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const __half2 * hp = (const __half2 *) &w[u];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(hp[i]);
                    dot[u] += f.x * qv[2 * i] + f.y * qv[2 * i + 1];
                }
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
            for (int u = 0; u < 4; ++u) dot[u] += __shfl_xor_sync(0xffffffffu, dot[u], off);
        }
        if (lane == 0) {
#pragma unroll
            for (int u = 0; u < 4; ++u) if (base + u < cnt) scores[base + u] = dot[u] * scale;
        }
    }
    __syncthreads();
    // the run's softmax statistics
    __shared__ float red[32];
    float m = neg_inf;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) m = fmaxf(m, scores[j]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < nwarps; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float l = 0.f;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
        const float e = expf(scores[j] - m);
        scores[j] = e;
        l += e;
    }
    for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if (lane == 0) red[warp] = l;
    __syncthreads();
    l = 0.f;
    for (int i = 0; i < nwarps; ++i) l += red[i];
    // the weighted values: each warp sums its tokens, the warps' partial sums are added in order
    float acc[4][8];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[a][i] = 0.f;
    for (int base = warp * 4; base < cnt; base += nwarps * 4) {
        uint32_t ids[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) ids[u] = base + u < cnt ? sel[lo + base + u] : 0u;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int v = lane + 32 * a;
            if (v < nvec) {
                uint4 w[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) w[u] = ((const uint4 *) (vh + (size_t) ids[u] * hd))[v];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float p = base + u < cnt ? scores[base + u] : 0.f;
                    const __half2 * hp = (const __half2 *) &w[u];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 f = __half22float2(hp[i]);
                        acc[a][2 * i] += p * f.x;
                        acc[a][2 * i + 1] += p * f.y;
                    }
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
    const size_t run = (size_t) h * splits + s;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.f;
        for (int i = 0; i < nwarps; ++i) o += part[i * hd + d];
        part_o[run * hd + d] = o;
    }
    if (threadIdx.x == 0) {
        part_ml[run * 2] = cnt > 0 ? m : neg_inf;
        part_ml[run * 2 + 1] = cnt > 0 ? l : 0.f;
    }
}

// The runs' answers joined: each run's sums are scaled to the largest maximum among them.
extern "C" __global__ void loken_selected_attn_join_kernel(
        const float * __restrict__ part_o, const float * __restrict__ part_ml,
        float * __restrict__ out, int hd, int splits, int n_head) {
    const int h = blockIdx.x;
    {
        const size_t row = blockIdx.y;
        part_o += row * n_head * splits * hd;
        part_ml += row * n_head * splits * 2;
        out += row * n_head * hd;
    }
    float m = __int_as_float(0xff800000);
    for (int s = 0; s < splits; ++s) m = fmaxf(m, part_ml[((size_t) h * splits + s) * 2]);
    float total = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t run = (size_t) h * splits + s;
        if (part_ml[run * 2 + 1] > 0.f) total += part_ml[run * 2 + 1] * expf(part_ml[run * 2] - m);
    }
    const float inv = 1.f / total;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.f;
        for (int s = 0; s < splits; ++s) {
            const size_t run = (size_t) h * splits + s;
            if (part_ml[run * 2 + 1] > 0.f) o += part_o[run * hd + d] * expf(part_ml[run * 2] - m);
        }
        out[(size_t) h * hd + d] = o * inv;
    }
}

extern "C" void loken_selected_attn(
        const float * q, const void * kbuf, const void * vbuf, const uint32_t * sel,
        const uint32_t * count, float * out, float * part_o, float * part_ml, int n_head,
        int n_kv, int cap, int hd, int ncap, int splits, float scale, cudaStream_t stream) {
    const int threads = 256;
    const int chunk_max = (ncap + splits - 1) / splits;
    const size_t smem = (size_t) (chunk_max + hd + (threads / 32) * hd) * sizeof(float);
    loken_selected_attn_split_kernel<<<dim3(n_head, splits), threads, smem, stream>>>(
        q, (const __half *) kbuf, (const __half *) vbuf, sel, count, part_o, part_ml, n_head, n_kv,
        cap, hd, chunk_max, splits, scale, 0);
    loken_selected_attn_join_kernel<<<n_head, hd < 1024 ? hd : 1024, 0, stream>>>(
        part_o, part_ml, out, hd, splits, n_head);
}

// `loken_selected_attn` over `rows` decode steps at once: row `r` reads its query at
// `q + r * n_head * hd`, its kept tokens at `sel + r * sel_stride` and `count[r]`, and writes
// `out + r * n_head * hd`, each row with the splits and sums of a step alone.
extern "C" void loken_selected_attn_steps(
        const float * q, const void * kbuf, const void * vbuf, const uint32_t * sel,
        const uint32_t * count, float * out, float * part_o, float * part_ml, int n_head,
        int n_kv, int cap, int hd, int ncap, int splits, float scale, int rows, int sel_stride,
        cudaStream_t stream) {
    const int threads = 256;
    const int chunk_max = (ncap + splits - 1) / splits;
    const size_t smem = (size_t) (chunk_max + hd + (threads / 32) * hd) * sizeof(float);
    loken_selected_attn_split_kernel<<<dim3(n_head, splits, rows), threads, smem, stream>>>(
        q, (const __half *) kbuf, (const __half *) vbuf, sel, count, part_o, part_ml, n_head, n_kv,
        cap, hd, chunk_max, splits, scale, sel_stride);
    loken_selected_attn_join_kernel<<<dim3(n_head, rows), hd < 1024 ? hd : 1024, 0, stream>>>(
        part_o, part_ml, out, hd, splits, n_head);
}

// The attention of `loken_selected_attn_steps` with a block for each key-value head, run and row:
// the group of query heads that share the head reads each kept key and value once. The run's
// scores are each head's, its statistics and weighted values too, written as the per-head runs
// `loken_selected_attn_join_kernel` joins. `hd` is at most the block's threads, the group at most
// GROUP_MAX.
#define GROUP_MAX 16

extern "C" __global__ void loken_selected_attn_group_kernel(
        const float * __restrict__ q, const __half * __restrict__ kbuf,
        const __half * __restrict__ vbuf, const uint32_t * __restrict__ sel,
        const uint32_t * __restrict__ count, float * __restrict__ part_o,
        float * __restrict__ part_ml, int n_head, int n_kv, int cap, int hd, int chunk_max,
        int splits, float scale, int sel_stride, const int * __restrict__ pos) {
    extern __shared__ float smem[];
    // Without a selection (`sel` null) a row keeps every token up to its position `pos`.
    {
        const size_t row = blockIdx.z;
        q += row * n_head * hd;
        if (sel != nullptr) {
            sel += row * sel_stride;
            count += row;
        } else {
            pos += row;
        }
        part_o += row * n_head * splits * hd;
        part_ml += row * n_head * splits * 2;
    }
    const int g = n_head / n_kv;
    float * scores = smem;                       // [g][chunk_max]
    float * qs = scores + (size_t) g * chunk_max; // [g][hd]
    __shared__ float red[32];
    __shared__ float stat_m[GROUP_MAX], stat_l[GROUP_MAX];
    const int kvh = blockIdx.x, s = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const int n = sel != nullptr ? (int) *count : *pos + 1;
    const int chunk = (n + splits - 1) / splits;
    const int lo = s * chunk;
    const int cnt = n > lo ? (n - lo < chunk ? n - lo : chunk) : 0;
    const int nvec = hd / 8;
    const float neg_inf = __int_as_float(0xff800000);
    for (int i = threadIdx.x; i < g * hd; i += blockDim.x) qs[i] = q[(size_t) (kvh * g) * hd + i];
    __syncthreads();
    const __half * kh = kbuf + (size_t) kvh * cap * hd;
    const __half * vh = vbuf + (size_t) kvh * cap * hd;
    // the scores: a warp takes four tokens at a time, its lanes the keys' pieces, each head's dot
    // summed over them
    for (int base = warp * 4; base < cnt; base += nwarps * 4) {
        uint32_t ids[4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
            ids[u] = base + u < cnt ? (sel != nullptr ? sel[lo + base + u] : (uint32_t) (lo + base + u)) : 0u;
        float dot[4][GROUP_MAX];
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int h = 0; h < GROUP_MAX; ++h) dot[u][h] = 0.f;
        for (int v = lane; v < nvec; v += 32) {
            uint4 w[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) w[u] = ((const uint4 *) (kh + (size_t) ids[u] * hd))[v];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const __half2 * hp = (const __half2 *) &w[u];
                float k8[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(hp[i]);
                    k8[2 * i] = f.x;
                    k8[2 * i + 1] = f.y;
                }
#pragma unroll
                for (int h = 0; h < GROUP_MAX; ++h) {
                    if (h < g) {
                        const float * qv = qs + h * hd + v * 8;
#pragma unroll
                        for (int i = 0; i < 8; ++i) dot[u][h] += k8[i] * qv[i];
                    }
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int h = 0; h < GROUP_MAX; ++h) {
                if (h < g) {
                    float d = dot[u][h];
                    for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                    if (lane == 0 && base + u < cnt) scores[h * chunk_max + base + u] = d * scale;
                }
            }
        }
    }
    __syncthreads();
    // each head's statistics over the run
    for (int h = 0; h < g; ++h) {
        float * sc = scores + h * chunk_max;
        float m = neg_inf;
        for (int j = threadIdx.x; j < cnt; j += blockDim.x) m = fmaxf(m, sc[j]);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (lane == 0) red[warp] = m;
        __syncthreads();
        m = red[0];
        for (int i = 1; i < nwarps; ++i) m = fmaxf(m, red[i]);
        __syncthreads();
        float l = 0.f;
        for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
            const float e = expf(sc[j] - m);
            sc[j] = e;
            l += e;
        }
        for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
        if (lane == 0) red[warp] = l;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.f;
            for (int i = 0; i < nwarps; ++i) t += red[i];
            stat_m[h] = m;
            stat_l[h] = t;
        }
        __syncthreads();
    }
    // the weighted values: a thread takes a dimension, each value read once for the group, eight
    // tokens' values loaded at a time
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc[GROUP_MAX];
#pragma unroll
        for (int h = 0; h < GROUP_MAX; ++h) acc[h] = 0.f;
        for (int j0 = 0; j0 < cnt; j0 += 8) {
            float vd[8];
#pragma unroll
            for (int u = 0; u < 8; ++u)
                vd[u] = j0 + u < cnt
                            ? __half2float(vh[(size_t) (sel != nullptr ? sel[lo + j0 + u] : (uint32_t) (lo + j0 + u)) * hd + d])
                            : 0.f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                if (j0 + u < cnt) {
#pragma unroll
                    for (int h = 0; h < GROUP_MAX; ++h)
                        if (h < g) acc[h] += scores[h * chunk_max + j0 + u] * vd[u];
                }
            }
        }
        for (int h = 0; h < g; ++h) {
            const size_t run = (size_t) (kvh * g + h) * splits + s;
            part_o[run * hd + d] = acc[h];
        }
    }
    if (threadIdx.x < g) {
        const size_t run = (size_t) (kvh * g + threadIdx.x) * splits + s;
        part_ml[run * 2] = cnt > 0 ? stat_m[threadIdx.x] : neg_inf;
        part_ml[run * 2 + 1] = cnt > 0 ? stat_l[threadIdx.x] : 0.f;
    }
}

extern "C" void loken_selected_attn_group(
        const float * q, const void * kbuf, const void * vbuf, const uint32_t * sel,
        const uint32_t * count, float * out, float * part_o, float * part_ml, int n_head,
        int n_kv, int cap, int hd, int ncap, int splits, float scale, int rows, int sel_stride,
        const int * pos, cudaStream_t stream) {
    const int threads = 256;
    const int g = n_head / n_kv;
    const int chunk_max = (ncap + splits - 1) / splits;
    const size_t smem = (size_t) (g * chunk_max + g * hd) * sizeof(float);
    loken_selected_attn_group_kernel<<<dim3(n_kv, splits, rows), threads, smem, stream>>>(
        q, (const __half *) kbuf, (const __half *) vbuf, sel, count, part_o, part_ml, n_head, n_kv,
        cap, hd, chunk_max, splits, scale, sel_stride, pos);
    loken_selected_attn_join_kernel<<<dim3(n_head, rows), hd < 1024 ? hd : 1024, 0, stream>>>(
        part_o, part_ml, out, hd, splits, n_head);
}

// The attention of `loken_selected_attn_rows_kernel` for cards without asynchronous copies: one
// row and one key-value head a block, thirty-two tokens at a time gathered into shared memory,
// scored and weighed through `wmma` with the same high and low halves, each head's softmax
// folded by one thread.
#include <mma.h>

#define SELROWS_W_HD 256
#define SELROWS_W_CELLS 32
#define SELROWS_W_LD (SELROWS_W_HD + 8)        // halves a shared row of keys, values or queries
#define SELROWS_W_PLD 36                     // floats a row of partial scores
#define SELROWS_W_PHLD (SELROWS_W_CELLS + 8)   // halves a row of probabilities
#define SELROWS_W_TLD (SELROWS_W_HD + 4)       // floats a row of the values' weighted sum
#define SELROWS_W_GMAX 16

// Bytes of shared memory a block needs, within the 64 KB a block has on every card: the partial
// scores and the weighted sums share the keys' tile, which is dead once the scores are taken.
#define SELROWS_W_SMEM (2 * 16 * SELROWS_W_LD * 2 + 2 * SELROWS_W_CELLS * SELROWS_W_LD * 2 \
                      + 2 * 16 * SELROWS_W_PHLD * 2 + 3 * 16 * 4)
static_assert(16 * SELROWS_W_TLD * 4 <= SELROWS_W_CELLS * SELROWS_W_LD * 2
                  && 4 * 16 * SELROWS_W_PLD * 4 <= SELROWS_W_CELLS * SELROWS_W_LD * 2,
              "the scores and the sums fit the keys' tile");
static_assert(SELROWS_W_SMEM <= 64 * 1024, "a block's shared memory fits every card");

extern "C" __global__ void __launch_bounds__(256) loken_selected_attn_rows_wmma_kernel(
        const float * __restrict__ q, const __half * __restrict__ kbuf,
        const __half * __restrict__ vbuf, const uint32_t * __restrict__ sel,
        const uint32_t * __restrict__ count, float * __restrict__ out, int n_head, int n_kv,
        int cap, int ncap, float scale) {
    using namespace nvcuda;
    extern __shared__ __align__(128) unsigned char raw[];
    const int g = n_head / n_kv;
    __half * qh = (__half *) raw;                                   // [16][LD] query, high half
    __half * ql = qh + 16 * SELROWS_W_LD;                             // [16][LD] query, low half
    __half * ks = ql + 16 * SELROWS_W_LD;                             // [CELLS][LD]
    __half * vs = ks + SELROWS_W_CELLS * SELROWS_W_LD;                  // [CELLS][LD]
    float * work = (float *) ks;                                     // partial scores, then the sums
    __half * ph = vs + SELROWS_W_CELLS * SELROWS_W_LD;                 // [16][PHLD] probabilities
    __half * pl = ph + 16 * SELROWS_W_PHLD;
    float * stats = (float *) (pl + 16 * SELROWS_W_PHLD);             // m[16], l[16], alpha[16]
    const int row = blockIdx.x, kvh = blockIdx.y, t = threadIdx.x;
    const int warp = t >> 5;
    const int n = (int) count[row];
    const float * qrow = q + ((size_t) row * n_head + (size_t) kvh * g) * SELROWS_W_HD;
    for (int i = t; i < 16 * SELROWS_W_HD; i += 256) {
        const int h = i / SELROWS_W_HD, d = i % SELROWS_W_HD;
        float v = h < g ? qrow[(size_t) h * SELROWS_W_HD + d] * scale : 0.f;
        const __half hi = __float2half(v);
        qh[h * SELROWS_W_LD + d] = hi;
        ql[h * SELROWS_W_LD + d] = __float2half(v - __half2float(hi));
    }
    for (int i = t; i < 16 * SELROWS_W_PHLD; i += 256) {
        ph[i] = __float2half(0.f);
        pl[i] = __float2half(0.f);
    }
    if (t < 16) {
        stats[t] = __int_as_float(0xff800000);
        stats[16 + t] = 0.f;
        stats[32 + t] = 0.f;
    }
    float acc[SELROWS_W_GMAX];
#pragma unroll
    for (int h = 0; h < SELROWS_W_GMAX; ++h) acc[h] = 0.f;
    const __half * kh = kbuf + (size_t) kvh * cap * SELROWS_W_HD;
    const __half * vh = vbuf + (size_t) kvh * cap * SELROWS_W_HD;
    const uint32_t * mine = sel + (size_t) row * ncap;
    for (int base = 0; base < n; base += SELROWS_W_CELLS) {
        const int cnt = n - base < SELROWS_W_CELLS ? n - base : SELROWS_W_CELLS;
        __syncthreads();
        for (int i = t; i < SELROWS_W_CELLS * 32; i += 256) {
            const int c = i >> 5, v4 = i & 31;
            if (c < cnt) {
                const size_t at = (size_t) mine[base + c] * SELROWS_W_HD;
                ((uint4 *) (ks + c * SELROWS_W_LD))[v4] = ((const uint4 *) (kh + at))[v4];
                ((uint4 *) (vs + c * SELROWS_W_LD))[v4] = ((const uint4 *) (vh + at))[v4];
            } else {
                // a cell the tile lacks weighs nothing; its values must still be numbers
                ((uint4 *) (ks + c * SELROWS_W_LD))[v4] = make_uint4(0, 0, 0, 0);
                ((uint4 *) (vs + c * SELROWS_W_LD))[v4] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        // the scores: warp w takes the cells [16 (w & 1), +16) and the dimensions [64 (w >> 1), +64)
        {
            const int nt = warp & 1, kg = warp >> 1;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.f);
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int d0 = (kg * 4 + step) * 16;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, ks + (nt * 16) * SELROWS_W_LD + d0, SELROWS_W_LD);
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, qh + d0, SELROWS_W_LD);
                wmma::mma_sync(fc, fa, fb, fc);
                wmma::load_matrix_sync(fa, ql + d0, SELROWS_W_LD);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            // every warp is done reading the keys the scores overwrite
            __syncthreads();
            wmma::store_matrix_sync(work + kg * 16 * SELROWS_W_PLD + nt * 16, fc, SELROWS_W_PLD,
                                    wmma::mem_row_major);
        }
        __syncthreads();
        if (t < g) {
            float s[SELROWS_W_CELLS];
            float m = stats[t];
            for (int c = 0; c < cnt; ++c) {
                s[c] = 0.f;
                for (int kg = 0; kg < 4; ++kg) s[c] += work[kg * 16 * SELROWS_W_PLD + t * SELROWS_W_PLD + c];
                m = fmaxf(m, s[c]);
            }
            const float alpha = expf(stats[t] - m);
            float l = stats[16 + t] * alpha;
            for (int c = 0; c < SELROWS_W_CELLS; ++c) {
                const float p = c < cnt ? expf(s[c] - m) : 0.f;
                const __half hi = __float2half(p);
                ph[t * SELROWS_W_PHLD + c] = hi;
                pl[t * SELROWS_W_PHLD + c] = __float2half(p - __half2float(hi));
                l += p;
            }
            stats[t] = m;
            stats[16 + t] = l;
            stats[32 + t] = alpha;
        }
        __syncthreads();
        // the weighted values of the tile: warp w takes the dimensions [32 w, +32)
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> f0, f1;
            wmma::fill_fragment(f0, 0.f);
            wmma::fill_fragment(f1, 0.f);
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fh, fl;
                wmma::load_matrix_sync(fh, ph + step * 16, SELROWS_W_PHLD);
                wmma::load_matrix_sync(fl, pl + step * 16, SELROWS_W_PHLD);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, vs + (step * 16) * SELROWS_W_LD + warp * 32, SELROWS_W_LD);
                wmma::mma_sync(f0, fh, fb, f0);
                wmma::mma_sync(f0, fl, fb, f0);
                wmma::load_matrix_sync(fb, vs + (step * 16) * SELROWS_W_LD + warp * 32 + 16, SELROWS_W_LD);
                wmma::mma_sync(f1, fh, fb, f1);
                wmma::mma_sync(f1, fl, fb, f1);
            }
            wmma::store_matrix_sync(work + warp * 32, f0, SELROWS_W_TLD, wmma::mem_row_major);
            wmma::store_matrix_sync(work + warp * 32 + 16, f1, SELROWS_W_TLD, wmma::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < SELROWS_W_GMAX; ++h)
            if (h < g) acc[h] = acc[h] * stats[32 + h] + work[h * SELROWS_W_TLD + t];
    }
    __syncthreads();
    float * orow = out + ((size_t) row * n_head + (size_t) kvh * g) * SELROWS_W_HD;
#pragma unroll
    for (int h = 0; h < SELROWS_W_GMAX; ++h) {
        if (h < g) {
            const float l = stats[16 + h];
            orow[h * SELROWS_W_HD + t] = l > 0.f ? acc[h] / l : 0.f;
        }
    }
}

// Attention of the rows of a prompt chunk, each over its own kept tokens. A block takes one row and
// one key-value head and serves the whole group of query heads that share it, so the keys and
// values of the row's kept tokens are read once for the group. The kept tokens go by tiles of
// SELROWS_CELLS: a tile's keys are copied to shared memory while the previous tile's values are
// weighed, and its values while its scores are taken, both on the tensor cores; each head's running
// softmax folds the tile in. A query and a probability are each split into a high and a low half,
// so that the products keep the precision of F32 while the keys and values, F16 already, are exact;
// the low half's products, small beside the high half's, are summed in F16 over one tile before
// they join the F32 sums. `q` is `[rows, n_head, 256]` F32, already rotated; `kbuf` and `vbuf` are
// `[n_kv, cap, 256]` F16; row `r` keeps `sel[r * ncap .. + count[r]]`; the answer is
// `[rows, n_head, 256]` F32.
#define SELROWS_HD 256
#define SELROWS_CELLS 32                   // kept tokens of a tile
#define SELROWS_M 16                       // query heads a block holds, the group padded
#define SELROWS_THREADS 256
#define SELROWS_LD (SELROWS_HD + 8)        // halves a shared row of keys, values or queries
#define SELROWS_WLD (SELROWS_CELLS + 4)    // floats a row of partial scores
#define SELROWS_PLD (SELROWS_CELLS + 8)    // halves a row of probabilities

// Bytes of shared memory a block needs: keys and values of a tile, the two halves of the partial
// scores, the probabilities' two halves, each head's rescale and total.
#define SELROWS_SMEM (2 * SELROWS_CELLS * SELROWS_LD * 2 + 2 * SELROWS_M * SELROWS_WLD * 4 \
                      + 2 * SELROWS_M * SELROWS_PLD * 2 + 2 * SELROWS_M * 4)

// The queries' two halves are staged where a tile's values go.
static_assert(2 * SELROWS_M <= SELROWS_CELLS, "the queries fit the values' tile");
// Warp w scores the cells [8 (w % 4), +8) over half the dimensions, and weighs the dimensions
// [32 w, +32): eight warps, four tiles of eight cells, eight slices of thirty-two dimensions.
static_assert(SELROWS_THREADS == 256 && SELROWS_CELLS == 32 && SELROWS_HD == 256,
              "the warps' share of a tile");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t selrows_shared(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void selrows_ldsm(uint32_t (&r)[4], const __half * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(selrows_shared(p)));
}

__device__ __forceinline__ void selrows_ldsm_t(uint32_t (&r)[4], const __half * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(selrows_shared(p)));
}

// c += a b over a 16x8x16 tile, F32 sums.
__device__ __forceinline__ void selrows_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// c += a b over a 16x8x16 tile, F16 sums.
__device__ __forceinline__ void selrows_mma_h(uint32_t (&c)[2], const uint32_t (&a)[4],
                                              uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, "
                 "{%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A thread's share of a tile: the cells `t / 32 + 8 j` and the pieces `t % 32` of their rows.
#define SELROWS_SHARE (SELROWS_CELLS * (SELROWS_HD / 8) / SELROWS_THREADS)
static_assert(SELROWS_SHARE * SELROWS_THREADS == SELROWS_CELLS * (SELROWS_HD / 8),
              "the threads share a tile evenly");

// The kept tokens of the cells of a tile at `base` a thread copies, those past `n` read as 0.
__device__ __forceinline__ void selrows_ids(uint32_t (&ids)[SELROWS_SHARE], const uint32_t * mine,
                                            int base, int n, int t) {
#pragma unroll
    for (int j = 0; j < SELROWS_SHARE; ++j) {
        const int c = (t + j * SELROWS_THREADS) / (SELROWS_HD / 8);
        ids[j] = base + c < n ? mine[base + c] : 0u;
    }
}

// The rows of the kept tokens `ids` of the tile at `base` copied from `src` to `dst`, the tile's
// rows past `n` zeroed, as one group of asynchronous copies.
__device__ __forceinline__ void selrows_gather(__half * dst, const __half * src,
                                               const uint32_t (&ids)[SELROWS_SHARE], int base,
                                               int n, int t) {
    constexpr int pieces = SELROWS_HD / 8;
#pragma unroll
    for (int j = 0; j < SELROWS_SHARE; ++j) {
        const int i = t + j * SELROWS_THREADS;
        const int c = i / pieces, v = i % pieces;
        const __half * from = src + (size_t) ids[j] * SELROWS_HD + v * 8;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                         selrows_shared(dst + c * SELROWS_LD + v * 8)),
                     "l"(from), "r"(base + c < n ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void selrows_commit_none() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Waits for every group of copies but the newest.
__device__ __forceinline__ void selrows_wait_older() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}
#endif

extern "C" __global__ void __launch_bounds__(SELROWS_THREADS, 2) loken_selected_attn_rows_kernel(
        const float * __restrict__ q, const __half * __restrict__ kbuf,
        const __half * __restrict__ vbuf, const uint32_t * __restrict__ sel,
        const uint32_t * __restrict__ count, float * __restrict__ out, int n_head, int n_kv,
        int cap, int ncap, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(128) unsigned char raw[];
    __half * ks = (__half *) raw;                                   // [CELLS][LD]
    __half * vs = ks + SELROWS_CELLS * SELROWS_LD;                  // [CELLS][LD]
    float * work = (float *) (vs + SELROWS_CELLS * SELROWS_LD);     // [2][M][WLD]
    __half * ph = (__half *) (work + 2 * SELROWS_M * SELROWS_WLD);  // [M][PLD]
    __half * pl = ph + SELROWS_M * SELROWS_PLD;                     // [M][PLD]
    float * alpha = (float *) (pl + SELROWS_M * SELROWS_PLD);       // [M]
    float * total = alpha + SELROWS_M;                              // [M]
    const int g = n_head / n_kv;
    const int row = blockIdx.x, kvh = blockIdx.y, t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    const int n = (int) count[row];
    const __half * kh = kbuf + (size_t) kvh * cap * SELROWS_HD;
    const __half * vh = vbuf + (size_t) kvh * cap * SELROWS_HD;
    const uint32_t * mine = sel + (size_t) row * ncap;
    uint32_t ids[SELROWS_SHARE];
    selrows_ids(ids, mine, 0, n, t);
    if (n > 0) selrows_gather(ks, kh, ids, 0, n, t);

    // the scaled queries' two halves, staged in the values' tile, then held by the warps
    {
        __half * qh = vs;
        __half * ql = vs + SELROWS_M * SELROWS_LD;
        const float * qrow = q + ((size_t) row * n_head + (size_t) kvh * g) * SELROWS_HD;
        for (int i = t; i < SELROWS_M * SELROWS_HD; i += SELROWS_THREADS) {
            const int h = i / SELROWS_HD, d = i % SELROWS_HD;
            const float v = h < g ? qrow[(size_t) h * SELROWS_HD + d] * scale : 0.f;
            const __half hi = __float2half(v);
            qh[h * SELROWS_LD + d] = hi;
            ql[h * SELROWS_LD + d] = __float2half(v - __half2float(hi));
        }
    }
    __syncthreads();
    const int nt = warp & 3, kq = warp >> 2;
    uint32_t qa[8][4], qb[8][4];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int at = (lane & 15) * SELROWS_LD + kq * 128 + s * 16 + (lane >> 4) * 8;
        selrows_ldsm(qa[s], vs + at);
        selrows_ldsm(qb[s], vs + SELROWS_M * SELROWS_LD + at);
    }
    __syncthreads();
    if (n > 0) selrows_gather(vs, vh, ids, 0, n, t);

    float o[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
    // the softmax of head `sr` over the cells `sc`, `sc + 1` of a tile, its state kept by each of
    // the sixteen threads of the head
    const int sr = t >> 4, sc = (t & 15) * 2;
    const float neg_inf = __int_as_float(0xff800000);
    float m_run = neg_inf, l_run = 0.f;
    for (int base = 0; base < n; base += SELROWS_CELLS) {
        const int cnt = n - base < SELROWS_CELLS ? n - base : SELROWS_CELLS;
        const bool more = base + SELROWS_CELLS < n;
        selrows_wait_older();
        __syncthreads();
        // the next tile's kept tokens, read while this one is weighed
        if (more) selrows_ids(ids, mine, base + SELROWS_CELLS, n, t);
        // the scores: warp w takes the cells [8 (w & 3), +8) and the dimensions [128 (w >> 2), +128)
        {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t d[2] = {0u, 0u};
#pragma unroll
            for (int s2 = 0; s2 < 4; ++s2) {
                uint32_t b[4];
                selrows_ldsm(b, ks + (nt * 8 + (lane & 7)) * SELROWS_LD + kq * 128 + s2 * 32
                                    + (lane >> 3) * 8);
                selrows_mma(c, qa[2 * s2], b[0], b[1]);
                selrows_mma_h(d, qb[2 * s2], b[0], b[1]);
                selrows_mma(c, qa[2 * s2 + 1], b[2], b[3]);
                selrows_mma_h(d, qb[2 * s2 + 1], b[2], b[3]);
            }
            const float2 d0 = __half22float2(*(const __half2 *) &d[0]);
            const float2 d1 = __half22float2(*(const __half2 *) &d[1]);
            float * w = work + kq * SELROWS_M * SELROWS_WLD + nt * 8 + (lane & 3) * 2;
            const int r0 = lane >> 2;
            *(float2 *) (w + r0 * SELROWS_WLD) = make_float2(c[0] + d0.x, c[1] + d0.y);
            *(float2 *) (w + (r0 + 8) * SELROWS_WLD) = make_float2(c[2] + d1.x, c[3] + d1.y);
        }
        __syncthreads();
        if (more) selrows_gather(ks, kh, ids, base + SELROWS_CELLS, n, t);
        else selrows_commit_none();
        // each head's running softmax takes in the tile
        {
            const float2 a = *(const float2 *) (work + sr * SELROWS_WLD + sc);
            const float2 b = *(const float2 *) (work + (SELROWS_M + sr) * SELROWS_WLD + sc);
            const bool in0 = sc < cnt, in1 = sc + 1 < cnt;
            const float s0 = a.x + b.x, s1 = a.y + b.y;
            float mx = fmaxf(in0 ? s0 : neg_inf, in1 ? s1 : neg_inf);
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_new = fmaxf(m_run, mx);
            const float al = expf(m_run - m_new);
            const float p0 = in0 ? expf(s0 - m_new) : 0.f;
            const float p1 = in1 ? expf(s1 - m_new) : 0.f;
            float ps = p0 + p1;
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
            l_run = l_run * al + ps;
            m_run = m_new;
            const __half h0 = __float2half(p0), h1 = __float2half(p1);
            *(__half2 *) (ph + sr * SELROWS_PLD + sc) = __halves2half2(h0, h1);
            *(__half2 *) (pl + sr * SELROWS_PLD + sc) = __halves2half2(
                __float2half(p0 - __half2float(h0)), __float2half(p1 - __half2float(h1)));
            if ((t & 15) == 0) alpha[sr] = al;
        }
        selrows_wait_older();
        __syncthreads();
        // the weighted values of the tile: warp w takes the dimensions [32 w, +32)
        {
            const float a0 = alpha[lane >> 2], a1 = alpha[(lane >> 2) + 8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                o[i][0] *= a0;
                o[i][1] *= a0;
                o[i][2] *= a1;
                o[i][3] *= a1;
            }
            uint32_t e[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) e[i][0] = e[i][1] = 0u;
#pragma unroll
            for (int k2 = 0; k2 < 2; ++k2) {
                uint32_t pa[4], pb[4];
                const int at = (lane & 15) * SELROWS_PLD + k2 * 16 + (lane >> 4) * 8;
                selrows_ldsm(pa, ph + at);
                selrows_ldsm(pb, pl + at);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint32_t b[4];
                    selrows_ldsm_t(b, vs + (k2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * SELROWS_LD
                                          + warp * 32 + j * 16 + (lane >> 4) * 8);
                    selrows_mma(o[2 * j], pa, b[0], b[1]);
                    selrows_mma_h(e[2 * j], pb, b[0], b[1]);
                    selrows_mma(o[2 * j + 1], pa, b[2], b[3]);
                    selrows_mma_h(e[2 * j + 1], pb, b[2], b[3]);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 lo0 = __half22float2(*(const __half2 *) &e[i][0]);
                const float2 lo1 = __half22float2(*(const __half2 *) &e[i][1]);
                o[i][0] += lo0.x;
                o[i][1] += lo0.y;
                o[i][2] += lo1.x;
                o[i][3] += lo1.y;
            }
        }
        __syncthreads();
        if (more) selrows_gather(vs, vh, ids, base + SELROWS_CELLS, n, t);
        else selrows_commit_none();
    }
    if ((t & 15) == 0) total[sr] = l_run;
    __syncthreads();
    float * orow = out + ((size_t) row * n_head + (size_t) kvh * g) * SELROWS_HD;
    const int r0 = lane >> 2;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int h = r0 + part * 8;
        if (h < g) {
            const float l = total[h];
            const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = warp * 32 + i * 8 + (lane & 3) * 2;
                *(float2 *) (orow + (size_t) h * SELROWS_HD + d) =
                    make_float2(o[i][2 * part] * inv, o[i][2 * part + 1] * inv);
            }
        }
    }
#endif
}

// The kernel the card runs: the asynchronous one when its code was built for compute 8.0 or later.
extern "C" void loken_selected_attn_rows(
        const float * q, const void * kbuf, const void * vbuf, const uint32_t * sel,
        const uint32_t * count, float * out, int rows, int n_head, int n_kv, int cap, int ncap,
        float scale, cudaStream_t stream) {
    cudaFuncAttributes built;
    if (cudaFuncGetAttributes(&built, loken_selected_attn_rows_kernel) == cudaSuccess
        && built.ptxVersion >= 80) {
        cudaFuncSetAttribute(loken_selected_attn_rows_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SELROWS_SMEM);
        loken_selected_attn_rows_kernel<<<dim3(rows, n_kv), SELROWS_THREADS, SELROWS_SMEM, stream>>>(
            q, (const __half *) kbuf, (const __half *) vbuf, sel, count, out, n_head, n_kv, cap,
            ncap, scale);
        return;
    }
    cudaFuncSetAttribute(loken_selected_attn_rows_wmma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SELROWS_W_SMEM);
    loken_selected_attn_rows_wmma_kernel<<<dim3(rows, n_kv), 256, SELROWS_W_SMEM, stream>>>(
        q, (const __half *) kbuf, (const __half *) vbuf, sel, count, out, n_head, n_kv, cap, ncap,
        scale);
}
