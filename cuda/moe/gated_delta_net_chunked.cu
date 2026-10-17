// The gated DeltaNet recurrence over many rows in the chunked form: the rows are cut into chunks
// of `GDN_CHUNK` rows, everything inside a chunk is matrix products computed for all chunks at
// once, and only the state passes from one chunk to the next.
//
// Notation for one head (M = the state as [kd(i), vd(col)], so M[i][col] is S[col][i] of the
// row-by-row kernel), in a chunk of rows t = 0..C-1 with decay logs g and write gates beta:
//   G[t]    = g[0] + ... + g[t]                    (the decay log since the chunk start)
//   L[t][s] = beta[t] * exp(G[t] - G[s]) * k[t].k[s]               for s < t, else 0
//   W       = (I + L)^-1 * diag(beta * exp(G)) * K,  U = (I + L)^-1 * diag(beta) * V
//   P[t][s] = scale * exp(G[t] - G[s]) * q[t].k[s]                 for s <= t, else 0
// With M0 the state at the chunk start, the per-row deltas, outputs and next state are
//   D  = U - W * M0
//   O  = scale * diag(exp(G)) * Q * M0 + P * D
//   M' = exp(G[C-1]) * M0 + K^T * diag(exp(G[C-1] - G)) * D
// which is the row-by-row recurrence of `gated_delta_net.cu` regrouped exactly.
//
// Two launches: `prep` (one block per chunk and head) writes W, U, P and G to the scratch;
// `scan` (one block per head and slice of state columns) walks the chunks in order with its
// slice of M in shared memory. Rows past the end of the last chunk read as zero with a zero
// gate, so they add nothing and leave the state as the last real row left it.

#include "gguf.cuh"  // ceil_div

// Rows of one chunk, threads of one block, and the width of a shared-memory tile.
constexpr int GDN_CHUNK   = 64;
constexpr int GDN_THREADS = 256;
constexpr int GDN_TILE    = 32;

// The scratch of one chunk and head, in floats: W and U ([C, S] each), P ([C, C]) and G ([C]).
__host__ __device__ constexpr long gdn_scratch_per_chunk(int s_v) {
    return 2L * GDN_CHUNK * s_v + (long) GDN_CHUNK * GDN_CHUNK + GDN_CHUNK;
}

// Four adjacent floats of shared memory, read as one.
static __device__ __forceinline__ float4 gdn_ld4(const float * p) {
    return *reinterpret_cast<const float4 *>(p);
}

template <int S>
__global__ void __launch_bounds__(GDN_THREADS) loken_gdn_chunk_prep_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        float * __restrict__ scratch,
        int H, int n_tokens,
        long sq1, long sq2, long sv1, long sv2, long sg1, long sg2,
        float scale) {
    constexpr int C = GDN_CHUNK;
    const int chunk = blockIdx.x;
    const int h     = blockIdx.y;
    const int tid   = threadIdx.x;
    const int row0  = chunk * C;
    const int valid = min(C, n_tokens - row0);

    // Q and K tiles, all rows of the chunk and GDN_TILE wide, rows padded to stay aligned.
    constexpr int VEC = 4;
    constexpr int LD  = GDN_TILE + VEC;
    constexpr int PER_TILE = C * GDN_TILE / GDN_THREADS;
    __shared__ __align__(16) float a_tile[C * LD];
    __shared__ __align__(16) float b_tile[C * LD];
    // L, its rows aligned so the solve reads four of a row at once.
    __shared__ __align__(16) float lt[C][C + 4];
    __shared__ float gs[C];
    __shared__ float bs[C];

    float * w_out = scratch + ((long) chunk * H + h) * gdn_scratch_per_chunk(S);
    float * u_out = w_out + (long) C * S;
    float * p_out = u_out + (long) C * S;
    float * g_out = p_out + (long) C * C;

    if (tid < C) {
        const long gbo = (long) (row0 + tid) * sg2 + (long) h * sg1;
        gs[tid] = tid < valid ? g[gbo] : 0.0f;
        bs[tid] = tid < valid ? beta[gbo] : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < C; t++) {
            acc += gs[t];
            gs[t] = acc;
        }
    }

    // K K^T and Q K^T: each thread owns a 4x4 block of rows ty + 16r and columns tx + 16j.
    constexpr int QK_SPAN = 16;
    constexpr int QK_REP  = C / QK_SPAN;
    const int ty = tid / QK_SPAN;
    const int tx = tid % QK_SPAN;
    float kk[QK_REP][QK_REP] = {};
    float qk[QK_REP][QK_REP] = {};
    // Each tile is fetched into registers while the one before it computes.
    float rq[PER_TILE], rk[PER_TILE];
    auto fetch = [&](int i0) {
#pragma unroll
        for (int j = 0; j < PER_TILE; j++) {
            const int e = tid + j * GDN_THREADS;
            const int t = e / GDN_TILE;
            const long off = (long) (row0 + t) * sq2 + (long) h * sq1 + i0 + e % GDN_TILE;
            rq[j] = t < valid ? q[off] : 0.0f;
            rk[j] = t < valid ? k[off] : 0.0f;
        }
    };
    fetch(0);
    for (int i0 = 0; i0 < S; i0 += GDN_TILE) {
#pragma unroll
        for (int j = 0; j < PER_TILE; j++) {
            const int e = tid + j * GDN_THREADS;
            a_tile[(e / GDN_TILE) * LD + e % GDN_TILE] = rq[j];
            b_tile[(e / GDN_TILE) * LD + e % GDN_TILE] = rk[j];
        }
        __syncthreads();
        if (i0 + GDN_TILE < S) fetch(i0 + GDN_TILE);
#pragma unroll 2
        for (int ii = 0; ii < GDN_TILE; ii += VEC) {
            float4 qa[QK_REP], ka[QK_REP], kb[QK_REP];
#pragma unroll
            for (int r = 0; r < QK_REP; r++) {
                qa[r] = gdn_ld4(&a_tile[(ty + QK_SPAN * r) * LD + ii]);
                ka[r] = gdn_ld4(&b_tile[(ty + QK_SPAN * r) * LD + ii]);
                kb[r] = gdn_ld4(&b_tile[(tx + QK_SPAN * r) * LD + ii]);
            }
#pragma unroll
            for (int r = 0; r < QK_REP; r++) {
#pragma unroll
                for (int j = 0; j < QK_REP; j++) {
                    kk[r][j] += ka[r].x * kb[j].x + ka[r].y * kb[j].y + ka[r].z * kb[j].z
                              + ka[r].w * kb[j].w;
                    qk[r][j] += qa[r].x * kb[j].x + qa[r].y * kb[j].y + qa[r].z * kb[j].z
                              + qa[r].w * kb[j].w;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < QK_REP; r++) {
        const int t = ty + QK_SPAN * r;
#pragma unroll
        for (int j = 0; j < QK_REP; j++) {
            const int s = tx + QK_SPAN * j;
            const float decay = expf(gs[t] - gs[s]);
            lt[t][s] = s < t ? bs[t] * decay * kk[r][j] : 0.0f;
            p_out[t * C + s] = s <= t ? scale * decay * qk[r][j] : 0.0f;
        }
    }
    if (tid < C) g_out[tid] = gs[tid];
    __syncthreads();

    // W and U solve (I + L) X = B by forward substitution, every column on its own: a thread
    // holds one column of B in registers and turns it into the same column of X, reading the
    // rows of L that all threads read at once.
    for (int col = tid; col < 2 * S; col += GDN_THREADS) {
        const bool is_w = col < S;
        const int  c    = is_w ? col : col - S;
        float x[C];
#pragma unroll
        for (int t = 0; t < C; t++) {
            if (t < valid) {
                const long row = row0 + t;
                x[t] = is_w ? k[row * sq2 + (long) h * sq1 + c] * bs[t] * expf(gs[t])
                            : v[row * sv2 + (long) h * sv1 + c] * bs[t];
            } else {
                x[t] = 0.0f;
            }
        }
#pragma unroll
        for (int t = 1; t < C; t++) {
            float acc = x[t];
#pragma unroll
            for (int s = 0; s < t; s++) acc -= lt[t][s] * x[s];
            x[t] = acc;
        }
        float * out = is_w ? w_out : u_out;
#pragma unroll
        for (int t = 0; t < C; t++) out[t * S + c] = x[t];
    }
}

// The state columns one scan block carries: a slice of at most 16 KiB of shared memory.
template <int S>
__host__ __device__ constexpr int gdn_scan_cols() {
    return S <= 128 ? 32 : 16;
}

// The scan walks every chunk as a fixed list of stages, each reading one tile: the tiles of W
// and Q over the key dim (D and the state's share of O), the tiles of P (the rows' share of
// O), then the tiles of K over the rows (the state update). No tile depends on the state, so
// each stage's tile is fetched into registers while the stage before it computes.
template <int S>
__global__ void __launch_bounds__(GDN_THREADS) loken_gdn_chunk_scan_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ scratch,
        // Read into shared memory first and written back last: may be the same buffer.
        const float * state_in,
        float * __restrict__ o_out,
        float * state_out,
        int H, int n_tokens,
        long sq1, long sq2,
        float scale) {
    constexpr int C    = GDN_CHUNK;
    constexpr int CB   = gdn_scan_cols<S>();
    constexpr int VEC  = 4;                       // adjacent floats read as one
    constexpr int TX   = CB / VEC;                // threads across the columns
    constexpr int SPAN = GDN_THREADS / TX;        // rows apart of one thread's outputs
    constexpr int REP  = C / SPAN;
    // A W, Q or P tile: all rows of the chunk, GDN_TILE wide, rows padded to stay aligned.
    constexpr int LD = GDN_TILE + VEC;
    constexpr int TILE_FLOATS = C * LD;
    // A K tile: KS rows of the chunk over the whole key dim, as many as fit the same space.
    constexpr int K_LD = S + VEC;
    constexpr int KS   = (TILE_FLOATS / K_LD) >= 32 ? 32 : (TILE_FLOATS / K_LD) >= 16 ? 16 : 8;
    // The state update: each thread owns IR adjacent key rows and VEC adjacent columns.
    constexpr int IG = GDN_THREADS / TX;
    constexpr int IR = S / IG;
    static_assert(KS * K_LD <= TILE_FLOATS && C % KS == 0, "a K tile fits a W tile");
    static_assert(C * GDN_TILE == KS * S, "every stage fetches as many floats");
    static_assert(IR == 2 || IR == 4, "the state rows of a thread are one vector");
    // The stages of one chunk, and the tile floats each thread fetches.
    constexpr int TILES_I  = S / GDN_TILE;
    constexpr int TILES_P  = C / GDN_TILE;
    constexpr int TILES_K  = C / KS;
    constexpr int STAGES   = TILES_I + TILES_P + TILES_K;
    constexpr int PER_TILE = C * GDN_TILE / GDN_THREADS;

    const int h   = blockIdx.x;
    const int c0  = blockIdx.y * CB;
    const int tid = threadIdx.x;
    const int ty  = tid / TX;
    const int tx  = (tid % TX) * VEC;
    const int ui  = (tid / TX) * IR;              // the state update's first key row
    const int n_stages = ceil_div(n_tokens, C) * STAGES;

    __shared__ __align__(16) float ms[S][CB];
    __shared__ __align__(16) float a_tile[TILE_FLOATS];
    __shared__ __align__(16) float b_tile[TILE_FLOATS];
    __shared__ __align__(16) float ds[C][CB];
    __shared__ float gs[C];
    __shared__ float fs[C];   // exp(G_last - G[t]): what row t's delta keeps at the chunk end

    const long state_base = (long) h * S * S + (long) c0 * S;
    for (int e = tid; e < S * CB; e += GDN_THREADS) {
        const int i = e % S;
        const int c = e / S;
        ms[i][c] = state_in[state_base + (long) c * S + i];
    }

    float ra[PER_TILE], rb[PER_TILE], rg = 0.0f;
    // Fetch the tile of stage `st` into registers; rows past the sequence read as zero.
    auto fetch = [&](int st) {
        const int chunk = st / STAGES;
        const int part  = st % STAGES;
        const int row0  = chunk * C;
        const int valid = min(C, n_tokens - row0);
        const float * w_in = scratch + ((long) chunk * H + h) * gdn_scratch_per_chunk(S);
        const float * p_in = w_in + 2L * C * S;
        if (part == 0 && tid < C) rg = p_in[(long) C * C + tid];
#pragma unroll
        for (int j = 0; j < PER_TILE; j++) {
            const int e = tid + j * GDN_THREADS;
            if (part < TILES_I) {
                const int t = e / GDN_TILE;
                const int i = part * GDN_TILE + e % GDN_TILE;
                ra[j] = w_in[t * S + i];
                rb[j] = t < valid ? q[(long) (row0 + t) * sq2 + (long) h * sq1 + i] : 0.0f;
            } else if (part < TILES_I + TILES_P) {
                const int t = e / GDN_TILE;
                ra[j] = p_in[t * C + (part - TILES_I) * GDN_TILE + e % GDN_TILE];
            } else {
                const int t = (part - TILES_I - TILES_P) * KS + e / S;
                ra[j] = t < valid ? k[(long) (row0 + t) * sq2 + (long) h * sq1 + e % S] : 0.0f;
            }
        }
    };

    float dacc[REP][VEC];
    float oacc[REP][VEC];
    fetch(0);
    for (int st = 0; st < n_stages; st++) {
        const int chunk = st / STAGES;
        const int part  = st % STAGES;
        const int row0  = chunk * C;
        const int valid = min(C, n_tokens - row0);
        const bool k_part = part >= TILES_I + TILES_P;
        const int s0 = (part - TILES_I - TILES_P) * KS;   // a K tile's first row
        if (part == 0 && tid < C) gs[tid] = rg;
#pragma unroll
        for (int j = 0; j < PER_TILE; j++) {
            const int e = tid + j * GDN_THREADS;
            if (k_part) {
                const int s = e / S;
                a_tile[s * K_LD + e % S] = ra[j] * fs[s0 + s];
            } else {
                a_tile[(e / GDN_TILE) * LD + e % GDN_TILE] = ra[j];
                if (part < TILES_I) b_tile[(e / GDN_TILE) * LD + e % GDN_TILE] = rb[j];
            }
        }
        __syncthreads();
        if (st + 1 < n_stages) fetch(st + 1);

        if (part < TILES_I) {
            // W * M0 and Q * M0 over this tile of the key dim.
            if (part == 0) {
#pragma unroll
                for (int r = 0; r < REP; r++) {
#pragma unroll
                    for (int c = 0; c < VEC; c++) dacc[r][c] = oacc[r][c] = 0.0f;
                }
            }
            const int i0 = part * GDN_TILE;
#pragma unroll 2
            for (int ii = 0; ii < GDN_TILE; ii += VEC) {
                float4 m[VEC];
#pragma unroll
                for (int x = 0; x < VEC; x++) m[x] = gdn_ld4(&ms[i0 + ii + x][tx]);
#pragma unroll
                for (int r = 0; r < REP; r++) {
                    const float4 w4 = gdn_ld4(&a_tile[(ty + SPAN * r) * LD + ii]);
                    const float4 q4 = gdn_ld4(&b_tile[(ty + SPAN * r) * LD + ii]);
                    const float wa[VEC] = {w4.x, w4.y, w4.z, w4.w};
                    const float qa[VEC] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                    for (int x = 0; x < VEC; x++) {
                        const float mv[VEC] = {m[x].x, m[x].y, m[x].z, m[x].w};
#pragma unroll
                        for (int c = 0; c < VEC; c++) {
                            dacc[r][c] += wa[x] * mv[c];
                            oacc[r][c] += qa[x] * mv[c];
                        }
                    }
                }
            }
            if (part == TILES_I - 1) {
                // D = U - W * M0, and the state's share of the output.
                const float * u_in = scratch
                    + ((long) chunk * H + h) * gdn_scratch_per_chunk(S) + (long) C * S;
#pragma unroll
                for (int r = 0; r < REP; r++) {
                    const int t = ty + SPAN * r;
                    const float og = scale * expf(gs[t]);
#pragma unroll
                    for (int c = 0; c < VEC; c++) {
                        ds[t][tx + c] = u_in[t * S + c0 + tx + c] - dacc[r][c];
                        oacc[r][c] *= og;
                    }
                }
                if (tid < C) fs[tid] = expf(gs[C - 1] - gs[tid]);
            }
        } else if (!k_part) {
            // O += P * D over this tile of P's columns.
            const int p0 = (part - TILES_I) * GDN_TILE;
#pragma unroll 2
            for (int ss = 0; ss < GDN_TILE; ss += VEC) {
                float4 d[VEC];
#pragma unroll
                for (int x = 0; x < VEC; x++) d[x] = gdn_ld4(&ds[p0 + ss + x][tx]);
#pragma unroll
                for (int r = 0; r < REP; r++) {
                    const float4 p4 = gdn_ld4(&a_tile[(ty + SPAN * r) * LD + ss]);
                    const float pa[VEC] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                    for (int x = 0; x < VEC; x++) {
                        const float dv[VEC] = {d[x].x, d[x].y, d[x].z, d[x].w};
#pragma unroll
                        for (int c = 0; c < VEC; c++) oacc[r][c] += pa[x] * dv[c];
                    }
                }
            }
            if (part == TILES_I + TILES_P - 1) {
#pragma unroll
                for (int r = 0; r < REP; r++) {
                    const int t = ty + SPAN * r;
                    if (t < valid) {
#pragma unroll
                        for (int c = 0; c < VEC; c++) {
                            o_out[(long) (row0 + t) * H * S + (long) h * S + c0 + tx + c] =
                                oacc[r][c];
                        }
                    }
                }
            }
        } else {
            // M' = exp(G_last) * M0 + (exp(G_last - G) * K)^T * D, this tile's rows of K.
            float acc[IR][VEC];
#pragma unroll
            for (int x = 0; x < IR; x++) {
                const float decay = part == TILES_I + TILES_P ? expf(gs[C - 1]) : 1.0f;
#pragma unroll
                for (int c = 0; c < VEC; c++) acc[x][c] = decay * ms[ui + x][tx + c];
            }
#pragma unroll 4
            for (int s = 0; s < KS; s++) {
                const float4 d4 = gdn_ld4(&ds[s0 + s][tx]);
                const float dv[VEC] = {d4.x, d4.y, d4.z, d4.w};
                float ka[IR];
                if constexpr (IR == 4) {
                    const float4 k4 = gdn_ld4(&a_tile[s * K_LD + ui]);
                    ka[0] = k4.x; ka[1] = k4.y; ka[2] = k4.z; ka[3] = k4.w;
                } else {
                    const float2 k2 = *reinterpret_cast<const float2 *>(&a_tile[s * K_LD + ui]);
                    ka[0] = k2.x; ka[1] = k2.y;
                }
#pragma unroll
                for (int x = 0; x < IR; x++) {
#pragma unroll
                    for (int c = 0; c < VEC; c++) acc[x][c] += ka[x] * dv[c];
                }
            }
#pragma unroll
            for (int x = 0; x < IR; x++) {
#pragma unroll
                for (int c = 0; c < VEC; c++) ms[ui + x][tx + c] = acc[x][c];
            }
        }
        __syncthreads();
    }

    for (int e = tid; e < S * CB; e += GDN_THREADS) {
        const int i = e % S;
        const int c = e / S;
        state_out[state_base + (long) c * S + i] = ms[i][c];
    }
}

// The rows below which the chunked form is not used: one chunk.
extern "C" int loken_gated_delta_net_chunk_rows() {
    return GDN_CHUNK;
}

// The scratch `loken_gated_delta_net_chunked` needs, in floats.
extern "C" long loken_gated_delta_net_chunked_scratch(int H, int n_tokens, int s_v) {
    return (long) ceil_div(n_tokens, GDN_CHUNK) * H * gdn_scratch_per_chunk(s_v);
}

// Same inputs, outputs and layouts as `loken_gated_delta_net` without snapshots, plus the
// scratch. An unsupported head dim launches nothing.
extern "C" void loken_gated_delta_net_chunked(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state_in,
        float * o_out, float * state_out,
        int H, int n_tokens, int s_v,
        long sq1, long sq2, long sv1, long sv2, long sg1, long sg2,
        float scale, float * scratch, cudaStream_t stream) {
    const dim3 prep_grid(ceil_div(n_tokens, GDN_CHUNK), H, 1);
#define LOKEN_GDN_CHUNKED_LAUNCH(s)                                                         \
    case s:                                                                                 \
        loken_gdn_chunk_prep_kernel<s><<<prep_grid, GDN_THREADS, 0, stream>>>(              \
            q, k, v, g, beta, scratch, H, n_tokens, sq1, sq2, sv1, sv2, sg1, sg2, scale);  \
        loken_gdn_chunk_scan_kernel<s>                                                      \
            <<<dim3(H, s / gdn_scan_cols<s>(), 1), GDN_THREADS, 0, stream>>>(               \
                q, k, scratch, state_in, o_out, state_out, H, n_tokens, sq1, sq2, scale);   \
        break;
    switch (s_v) {
        LOKEN_GDN_CHUNKED_LAUNCH(64)
        LOKEN_GDN_CHUNKED_LAUNCH(128)
        LOKEN_GDN_CHUNKED_LAUNCH(256)
        default:
            break;
    }
#undef LOKEN_GDN_CHUNKED_LAUNCH
}
