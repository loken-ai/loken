// FlashAttention-2 style NON-CAUSAL attention for diffusion transformers, BF16 on tensor
// cores, with the score matrix kept entirely in shared memory and registers.
//
// Why a second flash kernel. `flash_prefill_f16.cu` is shaped for LLM prefill: ONE warp
// owns 16 query rows and walks the whole K/V stream itself, which is right when the
// sequence is short and the grid is wide. A diffusion transformer hands attention twenty
// thousand tokens at once, so that kernel re-reads K and V about 1300 times and measured
// 6.4x SLOWER than the tiled cuBLAS path it was meant to beat. It is also causal, and a
// DiT attends both ways.
//
// What this one does differently: a THREAD BLOCK of four warps shares one K/V tile staged
// in shared memory, so the stream is read once per 64 query rows rather than once per 16,
// and a head's K/V (5.5 MB at 21504 tokens) stays resident in L2 across the blocks that
// re-read it. The [64 x seq] score slab that the tiled path writes to HBM, reads back for
// the softmax, writes again and reads a third time never leaves the chip.
//
// Measured motivation: attention is 82% of a 14B video block and runs at 31 TFLOP/s where
// the same card's BF16 GEMMs reach 86 - so it is the traffic, not the arithmetic.
//
// Layouts (all contiguous, row-major), no GQA, no mask:
//   Q [b, n_head, seq_q, HD]   K/V [b, n_head, seq_kv, HD]   O [b, n_head, seq_q, HD]
// The online softmax and the accumulator run in F32; only the two matrix products are BF16,
// which is the standard flash-attention precision and matches the tiled path this replaces.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WARPS 8
// Query rows a warp owns. TWO 16-row fragments rather than one: every K and V fragment a
// warp loads is then consumed by twice the matrix products. That ratio is what this kernel
// is bound by - its tensor-core instructions were under a tenth of its time and everything
// else was fetching operands for them.
#define MFRAG 2
#define WARP_M (MFRAG * 16)
#define BLOCK_M (WARPS * WARP_M)   // query rows per block
// K/V tokens staged per iteration, and warps per block. The binding constraint is not the
// tile count but OCCUPANCY: every byte of shared memory a block holds is a block the SM
// cannot also run, and this kernel's latency is hidden by having warps to switch to. At
// 44 KB a block the card ran two blocks - eight warps of a possible forty-eight - and the
// measured rate was 12 TFLOP/s against cuBLAS's 33. So the shared budget is spent on WARPS
// rather than on a wider tile, and the rescale scratch is aliased onto the score slab,
// which is dead by the time it is needed.
#define BLOCK_N 32

// Does this (query block, key tile) pair survive the RADIAL mask?
//
// Attention in a video transformer decays with distance in space AND in time, and Radial
// Attention (arXiv 2506.19852) turns that into a static pattern: between frames i and j the
// compute density is (1/2)^floor(log2|i-j|), realised as a diagonal band whose spatial width
// is `tokens_per_frame >> floor(log2|i-j|)`. Neighbouring frames attend fully; distant ones
// keep only a narrow band around the same spatial position. The count is O(n log n) rather
// than O(n^2), and the saving GROWS with the clip - which is the opposite of the problem.
//
// Decided per TILE, and kept if ANY pair inside it is kept: a tile is the unit this kernel
// can skip, and being generous at the edges costs a tile, where being wrong costs the
// picture. `frame_tokens == 0` disables the mask entirely (a non-video caller).
// Tokens of ONE frame the model saw at its trained resolution: 832x480 over a 16-pixel
// patch is 52x30. It is the bound on how much of its own frame a patch attends to, and it
// is a property of the CHECKPOINT rather than of the request - which is the point. Below it
// nothing is pruned and a 384 or 512-square render is untouched; above it the neighbourhood
// stops growing, so a 1024-square frame costs what the model was trained to handle rather
// than sixteen times it.
#define NATIVE_FRAME_PATCHES 1560

// One tile in this many is kept whatever the distance, so every patch keeps a coarse view of
// the whole frame. A neighbourhood alone loses the composition - the thing that tells a
// corner what the rest of the picture is doing - and that failure looks like a sharp image
// of the wrong scene, which no timing catches.
#define GLOBAL_TILE_STRIDE 8

// Does this (query block, key tile) pair survive the RADIAL mask?
//
// Attention in a video volume decays with distance in space AND in time, and Radial
// Attention (arXiv 2506.19852) turns that into a static pattern: between frames i and j the
// density is (1/2)^floor(log2|i-j|), realised as a window whose width shrinks by the same
// factor. Neighbouring frames attend fully; distant ones keep a narrow window around the
// same place.
//
// The distance is measured in TWO DIMENSIONS. Reading it off the raster index - which is
// what the first version did - makes "near" mean a full-width strip, because two patches one
// row apart are `grid_w` apart in raster order. At 384 square that is a rounding error; at
// 1024 square the frame is 64x64 patches and the strip keeps four thousand tokens where the
// neighbourhood holds a few hundred.
//
// Decided per TILE, and kept if ANY pair inside it is kept: a tile is the unit this kernel
// can skip, and being generous at the edges costs a tile where being wrong costs the
// picture. `frame_tokens == 0` disables the mask entirely (a non-video caller).
__device__ __forceinline__ bool radial_tile_kept(
    int q_lo, int q_hi, int k_lo, int k_hi, int frame_tokens, int grid_w, int tile_index)
{
    if (frame_tokens <= 0 || grid_w <= 0) return true;
    // A coarse view of the whole frame, whatever the distance.
    if ((tile_index % GLOBAL_TILE_STRIDE) == 0) return true;

    const int fq_lo = q_lo / frame_tokens, fq_hi = q_hi / frame_tokens;
    const int fk_lo = k_lo / frame_tokens, fk_hi = k_hi / frame_tokens;
    int d = 0;
    if (fk_lo > fq_hi)      d = fk_lo - fq_hi;
    else if (fq_lo > fk_hi) d = fq_lo - fk_hi;
    int band = 0;
    for (int t = d; t > 1; t >>= 1) ++band;        // floor(log2(d)), 0 for d <= 1

    // Radius in PATCHES, halving with the temporal band exactly as the density does, and
    // bounded at band 0 by the frame the model was trained on.
    const int grid_h = (frame_tokens + grid_w - 1) / grid_w;
    int r = (grid_w >> band) / 2;
    if (band == 0) {
        // Largest odd box holding at most NATIVE_FRAME_PATCHES tokens.
        int side = 1;
        while ((side + 2) * (side + 2) <= NATIVE_FRAME_PATCHES) side += 2;
        const int r_native = side / 2;
        if (r > r_native) r = r_native;
    }
    if (r < 1) r = 1;
    if (r >= grid_w && r >= grid_h) return true;   // the window covers the frame

    // A span that crosses a frame boundary touches every position in one of them.
    if (fq_hi > fq_lo || fk_hi > fk_lo) return true;
    const int sq_lo = q_lo % frame_tokens, sq_hi = q_hi % frame_tokens;
    const int sk_lo = k_lo % frame_tokens, sk_hi = k_hi % frame_tokens;
    const int qr0 = sq_lo / grid_w, qr1 = sq_hi / grid_w;
    const int kr0 = sk_lo / grid_w, kr1 = sk_hi / grid_w;
    int dr = 0;
    if (kr0 > qr1)      dr = kr0 - qr1;
    else if (qr0 > kr1) dr = qr0 - kr1;
    if (dr > r) return false;
    // A span of more than one row covers every column, so only a single-row span can be
    // declined on its columns.
    if (qr1 > qr0 || kr1 > kr0) return true;
    const int qc0 = sq_lo % grid_w, qc1 = sq_hi % grid_w;
    const int kc0 = sk_lo % grid_w, kc1 = sk_hi % grid_w;
    int dc = 0;
    if (kc0 > qc1)      dc = kc0 - qc1;
    else if (qc0 > kc1) dc = qc0 - kc1;
    return dc <= r;
}

template <int HD>
__global__ __launch_bounds__(WARPS * 32) void flash_dit_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
    int n_head, int seq_q, int seq_kv, float scale, int frame_tokens, int grid_w)
{
    constexpr int KSTEPS = HD / 16;   // contraction chunks of Q.K^T
    constexpr int NFRAG  = HD / 16;   // output head-dim chunks of P.V
    constexpr int SFRAG  = BLOCK_N / 16;

    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int qblk  = blockIdx.x;
    const int head  = blockIdx.y;
    const int bi    = blockIdx.z;

    const int q_base = qblk * BLOCK_M + warp * WARP_M;
    const int Mrows  = min(WARP_M, max(0, seq_q - q_base));

    const __nv_bfloat16* Qh = Q + ((size_t)(bi * n_head + head) * seq_q + q_base) * HD;
    const __nv_bfloat16* Kh = K + (size_t)(bi * n_head + head) * seq_kv * HD;
    const __nv_bfloat16* Vh = V + (size_t)(bi * n_head + head) * seq_kv * HD;
    __nv_bfloat16*       Oh = O + ((size_t)(bi * n_head + head) * seq_q + q_base) * HD;

    // One K/V tile for the whole block; the per-warp slabs are small because BLOCK_N is.
    __shared__ __nv_bfloat16 Ksh[BLOCK_N * HD];
    __shared__ __nv_bfloat16 Vsh[BLOCK_N * HD];
    // P shares the score slab. The scores are dead the moment each lane has read its own
    // columns into registers, and the probabilities are half as wide, so a separate buffer
    // was eight kilobytes a block bought nothing - and shared memory is what bounds how many
    // blocks an SM can run at once, which is what bounds the latency hiding.
    __shared__ float Ssh[WARPS][WARP_M * BLOCK_N];
    __nv_bfloat16* const Psh_w = reinterpret_cast<__nv_bfloat16*>(&Ssh[warp][0]);
    // The scores are dead once P has been written, so the rescale borrows their slab
    // instead of holding its own - 16x16 floats fit inside 16 x BLOCK_N for BLOCK_N >= 16.
    float* const scratch = &Ssh[warp][0];
    __shared__ float m_row[WARPS][WARP_M];
    __shared__ float l_row[WARPS][WARP_M];
    __shared__ float alpha_sh[WARPS][WARP_M];

    // Q stays in registers for the whole scan: it is read once, which is the point.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qfrag[MFRAG][KSTEPS];
    #pragma unroll
    for (int m = 0; m < MFRAG; ++m) {
        // Row groups past the end read whatever is there; their outputs are never stored.
        const bool live = (q_base + m * 16) < seq_q;
        #pragma unroll
        for (int s = 0; s < KSTEPS; ++s) {
            const __nv_bfloat16* src = live ? (Qh + (size_t)m * 16 * HD + s * 16) : Qh;
            wmma::load_matrix_sync(qfrag[m][s], src, HD);
        }
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> ofrag[MFRAG][NFRAG];
    #pragma unroll
    for (int m = 0; m < MFRAG; ++m) {
        #pragma unroll
        for (int n = 0; n < NFRAG; ++n) wmma::fill_fragment(ofrag[m][n], 0.0f);
    }
    for (int r = lane; r < WARP_M; r += 32) { m_row[warp][r] = -INFINITY; l_row[warp][r] = 0.0f; }
    __syncthreads();

    // This block's query span, for the mask below. Every warp in the block shares it, so
    // the decision is uniform and no warp diverges from the __syncthreads() that follow.
    const int blk_q_lo = qblk * BLOCK_M;
    const int blk_q_hi = min(blk_q_lo + BLOCK_M, seq_q) - 1;

    // This WARP's own query span. The mask is decided twice: once for the block, which is
    // what lets the staging be skipped, and once per warp, which is what makes it tight. A
    // block spans many query rows and has to keep any tile that touches any of them; a warp
    // spans few and can decline far more. Warps still reach every barrier - only the work
    // between them is skipped, never the synchronisation.
    const int w_q_lo = q_base;
    const int w_q_hi = min(q_base + WARP_M, seq_q) - 1;

    for (int t0 = 0; t0 < seq_kv; t0 += BLOCK_N) {
        const int ntok = min(BLOCK_N, seq_kv - t0);
        const int tile_index = t0 / BLOCK_N;
        if (!radial_tile_kept(blk_q_lo, blk_q_hi, t0, t0 + ntok - 1,
                              frame_tokens, grid_w, tile_index)) {
            continue;
        }
        const bool warp_kept = radial_tile_kept(w_q_lo, w_q_hi, t0, t0 + ntok - 1,
                                                frame_tokens, grid_w, tile_index);
        // Stage K and V cooperatively: every warp in the block reads this tile once.
        // Staged EIGHT elements at a time. One bf16 per instruction meant thirty-two loads
        // and thirty-two shared stores per thread per tile, and there are hundreds of tiles:
        // the staging was issuing more instructions than the matrix products it feeds. A
        // 128-bit access moves eight, and both rows are 256-byte aligned by construction
        // (HD is 64 or 128, and every offset into K and V is a whole number of rows).
        //
        // V is staged rather than read from global by the product below: on the last,
        // partial tile a global read would run past the end, and a garbage value multiplied
        // by a zero probability is still a NaN. The zero padding here is what makes the tail
        // exact - it is what the parity test at seq=130 and seq=333 catches.
        {
            constexpr int VEC = 8;                      // bf16 per 128-bit access
            constexpr int NVEC = BLOCK_N * HD / VEC;
            const int4* Kv = reinterpret_cast<const int4*>(Kh + (size_t)t0 * HD);
            const int4* Vv = reinterpret_cast<const int4*>(Vh + (size_t)t0 * HD);
            int4* Kd = reinterpret_cast<int4*>(Ksh);
            int4* Vd = reinterpret_cast<int4*>(Vsh);
            const int live_vecs = ntok * (HD / VEC);
            const int4 zero = make_int4(0, 0, 0, 0);
            for (int i = tid; i < NVEC; i += WARPS * 32) {
                const bool live = i < live_vecs;
                Kd[i] = live ? Kv[i] : zero;
                Vd[i] = live ? Vv[i] : zero;
            }
        }
        __syncthreads();
        if (!warp_kept) { __syncthreads(); continue; }

        // S = Q . K^T for this warp's 16 rows against the tile's BLOCK_N keys. K is stored
        // [tok, HD] row-major, so reading it column-major with ldm=HD yields K^T directly.
        #pragma unroll
        for (int nf = 0; nf < SFRAG; ++nf) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc[MFRAG];
            #pragma unroll
            for (int m = 0; m < MFRAG; ++m) wmma::fill_fragment(sacc[m], 0.0f);
            #pragma unroll
            for (int s = 0; s < KSTEPS; ++s) {
                // One K fragment, multiplied into BOTH row groups.
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kfrag;
                wmma::load_matrix_sync(kfrag, Ksh + nf * 16 * HD + s * 16, HD);
                #pragma unroll
                for (int m = 0; m < MFRAG; ++m)
                    wmma::mma_sync(sacc[m], qfrag[m][s], kfrag, sacc[m]);
            }
            #pragma unroll
            for (int m = 0; m < MFRAG; ++m)
                wmma::store_matrix_sync(&Ssh[warp][m * 16 * BLOCK_N + nf * 16],
                                        sacc[m], BLOCK_N, wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax over this tile, in ONE pass. With 32 query rows to a warp there
        // is exactly one row per lane, so a row's max and sum need no shuffle at all - the
        // half-row split this replaces existed only because 16 rows left half the warp idle.
        {
            const int r = lane;                  // WARP_M == 32
            float sv[BLOCK_N];
            float smax = -INFINITY;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; ++j) {
                const float s = (j < ntok && r < Mrows)
                    ? Ssh[warp][r * BLOCK_N + j] * scale : -INFINITY;
                sv[j] = s;
                smax = fmaxf(smax, s);
            }
            const float m_old = m_row[warp][r];
            const float m_new = fmaxf(m_old, smax);
            // Every lane holds its row now; only then may the slab be written over.
            __syncwarp();
            float lsum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; ++j) {
                const float p = (sv[j] == -INFINITY) ? 0.0f : __expf(sv[j] - m_new);
                Psh_w[r * BLOCK_N + j] = __float2bfloat16(p);
                lsum += p;
            }
            const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            m_row[warp][r] = m_new;
            l_row[warp][r] = l_row[warp][r] * alpha + lsum;
            alpha_sh[warp][r] = alpha;
        }
        __syncwarp();
        const bool rescale = __any_sync(0xffffffff, alpha_sh[warp][lane] != 1.0f);

        // O = O * alpha + P . V. The accumulator's element-to-row mapping is not part of
        // the WMMA contract, so the rescale goes through shared: store one 16x16 chunk,
        // scale it by its row's alpha, load it back. One chunk of scratch per warp, reused.
        // P depends only on the key chunk, not on the output chunk, so it is read ONCE per
        // tile rather than once per (output chunk, key chunk) pair - the loop below used to
        // reload it eight times over, which is eight times the shared traffic for the same
        // fragment.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pfrag[MFRAG][SFRAG];
        #pragma unroll
        for (int m = 0; m < MFRAG; ++m)
            #pragma unroll
            for (int s = 0; s < SFRAG; ++s)
                wmma::load_matrix_sync(pfrag[m][s], &Psh_w[m * 16 * BLOCK_N + s * 16], BLOCK_N);
        #pragma unroll
        for (int n = 0; n < NFRAG; ++n) {
            if (rescale) {
                #pragma unroll
                for (int m = 0; m < MFRAG; ++m) {
                    wmma::store_matrix_sync(scratch, ofrag[m][n], 16, wmma::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < 16 * 16; i += 32)
                        scratch[i] *= alpha_sh[warp][m * 16 + i / 16];
                    __syncwarp();
                    wmma::load_matrix_sync(ofrag[m][n], scratch, 16, wmma::mem_row_major);
                }
            }
            // One V fragment, multiplied into BOTH row groups.
            #pragma unroll
            for (int s = 0; s < SFRAG; ++s) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vfrag;
                wmma::load_matrix_sync(vfrag, Vsh + s * 16 * HD + n * 16, HD);
                #pragma unroll
                for (int m = 0; m < MFRAG; ++m)
                    wmma::mma_sync(ofrag[m][n], pfrag[m][s], vfrag, ofrag[m][n]);
            }
        }
        __syncthreads();
    }

    // Normalise by the running sum and write out.
    #pragma unroll
    for (int m = 0; m < MFRAG; ++m) {
        #pragma unroll
        for (int n = 0; n < NFRAG; ++n) {
            wmma::store_matrix_sync(scratch, ofrag[m][n], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 16 * 16; i += 32) {
                const int r = m * 16 + i / 16, d = i % 16;
                if (r < Mrows) {
                    const float l = l_row[warp][r];
                    Oh[(size_t)r * HD + n * 16 + d] =
                        __float2bfloat16(l > 0.0f ? scratch[i] / l : 0.0f);
                }
            }
            __syncwarp();
        }
    }
}

#define LAUNCH(HD)                                                                     \
    extern "C" void flash_dit_bf16_hd##HD(                                             \
        const void* Q, const void* K, const void* V, void* O,                          \
        int b, int n_head, int seq_q, int seq_kv, float scale, int frame_tokens,        \
        int grid_w, long long stream_i64)                                                          \
    {                                                                                  \
        dim3 grid((seq_q + BLOCK_M - 1) / BLOCK_M, n_head, b);                         \
        flash_dit_bf16_kernel<HD><<<grid, WARPS * 32, 0, (cudaStream_t)stream_i64>>>(  \
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K,                          \
            (const __nv_bfloat16*)V, (__nv_bfloat16*)O,                                \
            n_head, seq_q, seq_kv, scale, frame_tokens, grid_w);                       \
    }

LAUNCH(64)
LAUNCH(128)
