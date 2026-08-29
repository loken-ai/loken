// Query-head-packed tensor-core flash-DECODE for GQA over a Q8_0 KV cache.
//
// Lever #3 (docs/RESEARCH_LEVERS_2026_06.md): pack a GQA group's query heads
// (n_q_per_kv) AND the optional speculative verify tokens (qlen) into the MMA
// M-dimension, streaming each KV tile ONCE through the tensor cores. The cuda-
// core split-K path does the QK dot product with one warp-shuffle reduction PER
// query row PER token; here a single m16n8k16 HMMA produces the [M,16]-token
// score tile for all M packed rows at once, and a second HMMA does the P.V
// contraction. The 16-token KV tile is dequantized from Q8_0 into shared memory
// once and reused by every packed row.
//
// sm_120 (consumer Blackwell): classic register-MMA only - NO tcgen05/wgmma.
// nvcuda::wmma 16x16x16 f16 lowers to HMMA on sm_80+/sm_120; cp.async-class loads
// are implicit in the dequant staging. compute >=90a gencode (build.rs detects).
//
// Layout (matches Q8KvCache exactly): K_blocks / V_blocks are
//   block_q8_0[ t, n_kv, HDB ]  with HDB = head_dim/32 ; block={half d; int8 qs[32]}
// For (token t, kv-head h) the head_dim spans HDB consecutive blocks. seq_kv is
// read from a device int (graph-replay-safe): seq_kv = *seq_kv_dev + 1.
//
// Split-K over KV: grid = (nsplit, n_kv_heads), block = 1 warp. Each block emits
// its [Mrows, head_dim] partial + (m,l) per packed row into `partials`; the
// combine kernel flash-merges the splits per row (same log-sum-exp merge as the
// cuda-core split-K combine).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

#define FDTC_QK8 32
typedef struct { half d; int8_t qs[FDTC_QK8]; } fdtc_block_q8_0;

// One warp; HD=head_dim (templated, 128 here). M packed rows = n_q_per_kv*qlen,
// padded to 16 for the MMA. Online-softmax accumulator lives in shared memory as
// a flat [16 x HD] f32 tile (Oacc) so the WMMA P.V store accumulates trivially.
template <int HD>
static __device__ __forceinline__ void flashdecode_tc_partial_inner(
    const void  * __restrict__ K_blocks,
    const void  * __restrict__ V_blocks,
    const half  * __restrict__ Q,        // [16, HD] f16 row-major (Mrows valid, rest 0)
    float       * __restrict__ partials, // [Mrows, nsplit, HD+2]
    const int32_t * __restrict__ seq_kv_dev,
    int n_kv, int n_q_per_kv, int qlen, int nsplit, float scale,
    int h_kv, int split, int lane)
{
    constexpr int HDB    = HD / 32;       // q8_0 blocks per head row (4)
    constexpr int KSTEPS = HD / 16;       // QK^T contraction chunks (8)
    constexpr int HCHUNK = HD / 16;       // P.V output head-dim chunks (8)
    const int Mrows  = n_q_per_kv * qlen; // <=16
    // Global packed-row base for THIS kv-head: rows [grow, grow+Mrows) in the
    // flat [n_q_heads*qlen, HD] Q and [n_q_heads*qlen, nsplit, HD+2] partials.
    // grow = h_kv*Mrows because g = (h_kv*n_q_per_kv + qh)*qlen + tk = h_kv*Mrows + r.
    const int grow   = h_kv * Mrows;
    const int seq_kv = seq_kv_dev[0] + 1;
    const int chunk  = (seq_kv + nsplit - 1) / nsplit;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, seq_kv);

    __shared__ half  Qsh[16 * HD];        // staged Q
    __shared__ half  Kt [16 * HD];        // dequantized K tile
    __shared__ half  Vt [16 * HD];        // dequantized V tile
    __shared__ half  Psh[16 * 16];        // softmax probabilities tile
    __shared__ float Ssh[16 * 16];        // scores tile (fp32)
    __shared__ float Osh[16 * 16];        // PV result chunk (fp32)
    __shared__ float Oacc[16 * HD];       // online V accumulator (fp32)
    // Per-row online-softmax state MUST live in shared, not per-lane registers:
    // the softmax loop updates each row on the lane that owns it (lane==r), but
    // the partials are written by lane 0 - register copies would be stale.
    __shared__ float m_row[16];
    __shared__ float l_row[16];

    // Empty split -> neutral partials.
    if (t0 >= t1) {
        for (int r = 0; r < Mrows; ++r) {
            float * p = partials + ((size_t)(grow + r) * nsplit + split) * (HD + 2);
            for (int d = lane; d < HD; d += 32) p[d] = 0.0f;
            if (lane == 0) { p[HD] = -INFINITY; p[HD + 1] = 0.0f; }
        }
        return;
    }

    // Stage this kv-head's packed Q rows (Mrows x HD) into shared; zero the pad.
    for (int i = lane; i < 16 * HD; i += 32) {
        const int r = i / HD, d = i % HD;
        Qsh[i] = (r < Mrows) ? Q[(size_t)(grow + r) * HD + d] : __float2half(0.0f);
    }
    // Init accumulator + per-row softmax state.
    for (int i = lane; i < 16 * HD; i += 32) Oacc[i] = 0.0f;
    if (lane < 16) { m_row[lane] = -INFINITY; l_row[lane] = 0.0f; }
    __syncwarp();

    for (int tb = t0; tb < t1; tb += 16) {
        const int ntok = min(16, t1 - tb);
        // Dequantize the [16 x HD] KV tile from Q8_0 into shared.
        for (int idx = lane; idx < ntok * HDB; idx += 32) {
            const int tt = idx / HDB;
            const int b  = idx % HDB;
            const size_t blk = ((size_t)(tb + tt) * n_kv + h_kv) * HDB + b;
            const fdtc_block_q8_0 * kb = reinterpret_cast<const fdtc_block_q8_0 *>(K_blocks) + blk;
            const fdtc_block_q8_0 * vb = reinterpret_cast<const fdtc_block_q8_0 *>(V_blocks) + blk;
            const float kd = __half2float(kb->d);
            const float vd = __half2float(vb->d);
            half * kdst = Kt + tt * HD + b * 32;
            half * vdst = Vt + tt * HD + b * 32;
            #pragma unroll
            for (int e = 0; e < 32; ++e) {
                kdst[e] = __float2half((float)kb->qs[e] * kd);
                vdst[e] = __float2half((float)vb->qs[e] * vd);
            }
        }
        // Zero the padding tokens so their score is 0 and PV contributes 0.
        for (int idx = ntok * HD + lane; idx < 16 * HD; idx += 32) {
            Kt[idx] = __float2half(0.0f); Vt[idx] = __float2half(0.0f);
        }
        __syncwarp();

        // QK^T -> S[16(M) x 16(token)] (fp32). Contract over HD in 16-wide chunks.
        // A = Q[M x hd_chunk] row-major (ld=HD); B = K^T: Kt is [token x HD]
        // row-major, loaded col_major with ld=HD gives B[hd_chunk(K), token(N)].
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cS;
        wmma::fill_fragment(cS, 0.0f);
        #pragma unroll
        for (int kk = 0; kk < KSTEPS; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bF;
            wmma::load_matrix_sync(aF, Qsh + kk * 16, HD);
            wmma::load_matrix_sync(bF, Kt  + kk * 16, HD);
            wmma::mma_sync(cS, aF, bF, cS);
        }
        wmma::store_matrix_sync(Ssh, cS, 16, wmma::mem_row_major);
        __syncwarp();

        // Online softmax, per packed row (lanes 0..15 own rows; write Psh + the
        // per-row rescale factor `alpha` into shared so the Oacc rescale below
        // parallelizes across all 32 lanes instead of one-lane-per-row-serial).
        __shared__ float alpha_sh[16];
        for (int r = lane; r < 16; r += 32) {
            float smax = -INFINITY;
            #pragma unroll
            for (int n = 0; n < 16; ++n) {
                float s = (n < ntok && r < Mrows) ? Ssh[r * 16 + n] * scale : -INFINITY;
                Ssh[r * 16 + n] = s;
                smax = fmaxf(smax, s);
            }
            const float m_new = fmaxf(m_row[r], smax);
            const float alpha = (m_row[r] == -INFINITY) ? 0.0f : __expf(m_row[r] - m_new);
            float l_add = 0.0f;
            #pragma unroll
            for (int n = 0; n < 16; ++n) {
                float p = (Ssh[r * 16 + n] == -INFINITY) ? 0.0f : __expf(Ssh[r * 16 + n] - m_new);
                Psh[r * 16 + n] = __float2half(p);
                l_add += p;
            }
            l_row[r] = l_row[r] * alpha + l_add;
            m_row[r] = m_new;
            alpha_sh[r] = alpha;
        }
        __syncwarp();
        // Warp-parallel prior-accumulator rescale over the full [16 x HD] tile.
        for (int idx = lane; idx < 16 * HD; idx += 32) Oacc[idx] *= alpha_sh[idx / HD];
        __syncwarp();

        // P.V -> O[16(M) x HD], accumulate into Oacc. One HMMA per 16-wide hd chunk.
        // A = P[M x token] row-major (ld=16); B = V[token x hd_chunk] row-major (ld=HD).
        #pragma unroll
        for (int hc = 0; hc < HCHUNK; ++hc) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cO;
            wmma::fill_fragment(cO, 0.0f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vF;
            wmma::load_matrix_sync(pF, Psh, 16);
            wmma::load_matrix_sync(vF, Vt + hc * 16, HD);
            wmma::mma_sync(cO, pF, vF, cO);
            wmma::store_matrix_sync(Osh, cO, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                const int r = idx >> 4;          // 0..15
                const int c = idx & 15;          // 0..15 within this hd chunk
                Oacc[r * HD + hc * 16 + c] += Osh[idx];
            }
            __syncwarp();
        }
    }

    // Emit partials at global rows [grow, grow+Mrows).
    for (int r = 0; r < Mrows; ++r) {
        float * p = partials + ((size_t)(grow + r) * nsplit + split) * (HD + 2);
        for (int d = lane; d < HD; d += 32) p[d] = Oacc[r * HD + d];
        if (lane == 0) { p[HD] = m_row[r]; p[HD + 1] = l_row[r]; }
    }
}

#define FDTC_PARTIAL_KERNEL(NAME, HD)                                          \
extern "C" __global__ void NAME(                                              \
    const void * __restrict__ K_blocks, const void * __restrict__ V_blocks,   \
    const half * __restrict__ Q, float * __restrict__ partials,               \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int n_kv, int n_q_per_kv, int qlen, int nsplit, float scale)              \
{                                                                             \
    flashdecode_tc_partial_inner<HD>(                                         \
        K_blocks, V_blocks, Q, partials, seq_kv_dev,                          \
        n_kv, n_q_per_kv, qlen, nsplit, scale,                                \
        blockIdx.y, blockIdx.x, threadIdx.x);                                 \
}

FDTC_PARTIAL_KERNEL(flashdecode_tc_partial_hd128, 128)

// -- Combine: per packed row, flash-merge its nsplit partials -> out[row, HD].
// out is f16 [Mrows, HD]. One warp per row (grid.x = Mrows). Serial nsplit merge
// (nsplit is modest here, ~32-96); lane d owns dims d, d+32, ...
template <int HD>
static __device__ __forceinline__ void flashdecode_tc_combine_inner(
    const float * __restrict__ partials, half * __restrict__ out,
    int nsplit, int row, int lane)
{
    constexpr int DPL = HD / 32;
    const float * base = partials + (size_t)row * nsplit * (HD + 2);
    float gm = -INFINITY, gl = 0.0f;
    float acc[DPL];
    #pragma unroll
    for (int b = 0; b < DPL; ++b) acc[b] = 0.0f;
    for (int sp = 0; sp < nsplit; ++sp) {
        const float * p = base + (size_t)sp * (HD + 2);
        const float li = p[HD + 1];
        if (li <= 0.0f) continue;
        const float mi = p[HD];
        const float m_new = fmaxf(gm, mi);
        const float so = (gm == -INFINITY) ? 0.0f : __expf(gm - m_new);
        const float sn = __expf(mi - m_new);
        gl = gl * so + sn * li;
        #pragma unroll
        for (int b = 0; b < DPL; ++b) acc[b] = acc[b] * so + sn * p[b * 32 + lane];
        gm = m_new;
    }
    const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    #pragma unroll
    for (int b = 0; b < DPL; ++b) out[(size_t)row * HD + b * 32 + lane] = __float2half(acc[b] * inv);
}

#define FDTC_COMBINE_KERNEL(NAME, HD)                                          \
extern "C" __global__ void NAME(                                              \
    const float * __restrict__ partials, half * __restrict__ out, int nsplit) \
{                                                                             \
    flashdecode_tc_combine_inner<HD>(partials, out, nsplit, blockIdx.x, threadIdx.x); \
}

FDTC_COMBINE_KERNEL(flashdecode_tc_combine_hd128, 128)

// -- Host launchers (extern "C", driven from inference::flash_decode_tc) ------
// Q is f16 [16, HD] (caller pads Mrows->16, zeroing the unused rows). partials is
// f32 [Mrows*nsplit*(HD+2)] scratch. out is f16 [Mrows, HD]. seq_kv_dev is the
// device int (count = *seq_kv_dev + 1). All pointers device.
extern "C" void flashdecode_tc_hd128(
    const void * K_blocks, const void * V_blocks, const void * Q,
    float * partials, const int32_t * seq_kv_dev, void * out,
    int n_kv, int n_q_per_kv, int qlen, int nsplit, float scale,
    long long stream_i64)
{
    cudaStream_t stream = (cudaStream_t)stream_i64;
    const int total_rows = n_kv * n_q_per_kv * qlen;  // n_q_heads * qlen
    dim3 g1(nsplit, n_kv, 1), b1(32, 1, 1);
    flashdecode_tc_partial_hd128<<<g1, b1, 0, stream>>>(
        K_blocks, V_blocks, (const half *)Q, partials, seq_kv_dev,
        n_kv, n_q_per_kv, qlen, nsplit, scale);
    dim3 g2(total_rows, 1, 1), b2(32, 1, 1);
    flashdecode_tc_combine_hd128<<<g2, b2, 0, stream>>>(partials, (half *)out, nsplit);
}
