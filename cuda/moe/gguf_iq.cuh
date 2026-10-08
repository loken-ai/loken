// The formats whose values come out of a codebook: IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_S and Q2_0.
//
// Each format is one function that gives the value of a weight (or of a run of them) from its
// block, and both the decode dot and the prefill dequantiser are written over it. A dot reads a
// run of thirty-two weights against the activation block that covers them; the dequantiser
// writes the block's values. The layouts and the arithmetic are the host decoders' own
// (`quant_cpu/format/iq*.rs`), which a test holds to blocks read from real files.
//
// A lane of the decode kernels takes `vdr` ints of a block starting at `iqs`. These formats
// number their ints as the units each lane reads: one lane a run of thirty-two weights for the
// 256-value blocks (`QI` = 8 runs, `VDR` = 1), a half block of sixteen for q2_0 and a quarter of
// the block for iq4_nl.

#include "iq_tables.cuh"

// ------------------------------------------------------------------------------------------
// Layouts

#define QK4_NL 32
#define QI4_NL 4
#define VDR_IQ4_NL_Q8_1_MMVQ 2
typedef struct {
    half    d;
    uint8_t qs[QK4_NL / 2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(f16_bits) + QK4_NL / 2, "wrong iq4_nl block size/padding");

#define QI_IQ_RUNS 8
#define VDR_IQ_RUN_Q8_1_MMVQ 1

#define QK_IQ4_XS QK_K
#define QI_IQ4_XS QI_IQ_RUNS
#define VDR_IQ4_XS_Q8_1_MMVQ VDR_IQ_RUN_Q8_1_MMVQ
typedef struct {
    half     d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == 136, "wrong iq4_xs block size/padding");

#define QK_IQ3_XXS QK_K
#define QI_IQ3_XXS QI_IQ_RUNS
#define VDR_IQ3_XXS_Q8_1_MMVQ VDR_IQ_RUN_Q8_1_MMVQ
typedef struct {
    half    d;
    uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == 98, "wrong iq3_xxs block size/padding");

#define QK_IQ3_S QK_K
#define QI_IQ3_S QI_IQ_RUNS
#define VDR_IQ3_S_Q8_1_MMVQ VDR_IQ_RUN_Q8_1_MMVQ
typedef struct {
    half    d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[QK_K / 64];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == 110, "wrong iq3_s block size/padding");

#define QK_IQ2_S QK_K
#define QI_IQ2_S QI_IQ_RUNS
#define VDR_IQ2_S_Q8_1_MMVQ VDR_IQ_RUN_Q8_1_MMVQ
typedef struct {
    half    d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == 82, "wrong iq2_s block size/padding");

#define QK_Q2_0 64
#define QI_Q2_0 4
#define VDR_Q2_0_Q8_1_MMVQ 1
typedef struct {
    half    d;
    uint8_t qs[QK_Q2_0 / 4];
} block_q2_0;
static_assert(sizeof(block_q2_0) == 18, "wrong q2_0 block size/padding");

// ------------------------------------------------------------------------------------------
// The values of a run

/// Sign of bit `j` of `signs`: a set bit negates.
static __device__ __forceinline__ float sign_of(uint8_t signs, int j) {
    return (signs >> j) & 1 ? -1.0f : 1.0f;
}

/// Thirty-two values of an iq4_nl block, `y[j] = d . level[q]`, the low nibble of byte `j` for
/// value `j` and the high nibble for value `j + 16`.
static __device__ __forceinline__ void iq4_nl_run(const block_iq4_nl * b, float (&y)[32]) {
    const float d = __half2float(b->d);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        y[j]      = d * KVALUES_IQ4NL[b->qs[j] & 0x0F];
        y[j + 16] = d * KVALUES_IQ4NL[b->qs[j] >> 4];
    }
}

/// Run `ib` (of thirty-two values) of an iq4_xs block: a six-bit scale per run, low nibbles in
/// an array and the high pairs in one word.
static __device__ __forceinline__ void iq4_xs_run(const block_iq4_xs * b, int ib, float (&y)[32]) {
    const int low  = (b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
    const int high = (b->scales_h >> (2 * ib)) & 3;
    const float dl = __half2float(b->d) * (float) ((low | (high << 4)) - 32);
    const uint8_t * q = b->qs + 16 * ib;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        y[j]      = dl * KVALUES_IQ4NL[q[j] & 0x0F];
        y[j + 16] = dl * KVALUES_IQ4NL[q[j] >> 4];
    }
}

/// Run `ib` of an iq3_xxs block: eight codes into the four-magnitude grid, and a word that holds
/// four seven-bit sign patterns and a four-bit scale.
static __device__ __forceinline__ void iq3_xxs_run(const block_iq3_xxs * b, int ib, float (&y)[32]) {
    const uint8_t * codes = b->qs + 8 * ib;
    const uint8_t * tail  = b->qs + QK_K / 4 + 4 * ib;
    const uint32_t word = (uint32_t) tail[0] | ((uint32_t) tail[1] << 8) | ((uint32_t) tail[2] << 16) |
                          ((uint32_t) tail[3] << 24);
    const float db = __half2float(b->d) * (0.5f + (float) (word >> 28)) * 0.5f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint8_t signs = KSIGNS_IQ2XS[(word >> (7 * l)) & 127];
        const uint32_t g1 = IQ3XXS_GRID[codes[2 * l]];
        const uint32_t g2 = IQ3XXS_GRID[codes[2 * l + 1]];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[8 * l + j]     = db * (float) ((g1 >> (8 * j)) & 0xFF) * sign_of(signs, j);
            y[8 * l + j + 4] = db * (float) ((g2 >> (8 * j)) & 0xFF) * sign_of(signs, j + 4);
        }
    }
}

/// Run `ib` of an iq3_s block: nine-bit codes, a sign bit per value, one scale nibble per run.
static __device__ __forceinline__ void iq3_s_run(const block_iq3_s * b, int ib, float (&y)[32]) {
    const int pair = ib / 2;
    const int scale = (ib % 2 == 0) ? (b->scales[pair] & 0x0F) : (b->scales[pair] >> 4);
    const float db = __half2float(b->d) * (float) (1 + 2 * scale);
    const uint32_t qh = b->qh[ib];
    const uint8_t * q = b->qs + 8 * ib;
    const uint8_t * signs = b->signs + 4 * ib;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int i1 = q[2 * l]     | (int) ((qh << (8 - 2 * l)) & 256);
        const int i2 = q[2 * l + 1] | (int) ((qh << (7 - 2 * l)) & 256);
        const uint32_t g1 = IQ3S_GRID[i1];
        const uint32_t g2 = IQ3S_GRID[i2];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[8 * l + j]     = db * (float) ((g1 >> (8 * j)) & 0xFF) * sign_of(signs[l], j);
            y[8 * l + j + 4] = db * (float) ((g2 >> (8 * j)) & 0xFF) * sign_of(signs[l], j + 4);
        }
    }
}

/// Run `ib` of an iq2_s block: ten-bit codes into the eight-magnitude grid, a sign bit per value,
/// two scale nibbles per run, one for each half of it.
static __device__ __forceinline__ void iq2_s_run(const block_iq2_s * b, int ib, float (&y)[32]) {
    const uint8_t * signs = b->qs + QK_K / 8;
    const float d = __half2float(b->d);
    const float db0 = d * (0.5f + (float) (b->scales[ib] & 0x0F)) * 0.25f;
    const float db1 = d * (0.5f + (float) (b->scales[ib] >> 4)) * 0.25f;
    const uint32_t qh = b->qh[ib];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int code = b->qs[4 * ib + l] | (int) ((qh << (8 - 2 * l)) & 0x300);
        const uint64_t grid = IQ2S_GRID[code];
        const uint8_t sign = signs[4 * ib + l];
        const float db = l < 2 ? db0 : db1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            y[8 * l + j] = db * (float) ((grid >> (8 * j)) & 0xFF) * sign_of(sign, j);
        }
    }
}

// ------------------------------------------------------------------------------------------
// The dots: a run of weights against the q8_1 block that covers it

static __device__ __forceinline__ float dot_run_q8_1(const float (&y)[32], const block_q8_1 * q8) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        sum += y[j] * (float) q8->qs[j];
    }
    return sum * __low2float(q8->ds);
}

/// The iq4_nl levels of the low nibbles of the four bytes of `bytes`, as four signed bytes: the
/// nibble picks a byte of the sixteen levels, held as four words.
static __device__ __forceinline__ int levels_iq4_nl(uint32_t nibbles) {
    constexpr uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;
    const uint32_t t = (nibbles | (nibbles >> 4)) & 0x00FF00FFu;
    const uint32_t select = ((t | (t >> 8)) & 0xFFFFu) & 0x7777u;
    const uint32_t upper = ((nibbles >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t low  = __byte_perm(T0, T1, select);
    const uint32_t high = __byte_perm(T2, T3, select);
    return (int) ((low & ~upper) | (high & upper));
}

/// iq4_nl: a lane takes `VDR` ints of the sixteen code bytes, from `iqs`; four bytes at a time,
/// their low nibbles against the first half of the block's activations and the high nibbles
/// against the second.
static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq4_nl * b = (const block_iq4_nl *) vbq;
    const int * act = (const int *) bq8_1->qs;
    const uint16_t * halves = (const uint16_t *) b->qs;
    int sum = 0;
#pragma unroll
    for (int i = iqs; i < iqs + VDR_IQ4_NL_Q8_1_MMVQ; ++i) {
        const uint32_t word = (uint32_t) halves[2 * i] | ((uint32_t) halves[2 * i + 1] << 16);
        sum = __dp4a(levels_iq4_nl(word & 0x0F0F0F0Fu), act[i], sum);
        sum = __dp4a(levels_iq4_nl((word >> 4) & 0x0F0F0F0Fu), act[i + 4], sum);
    }
    return __half2float(b->d) * __low2float(bq8_1->ds) * (float) sum;
}

#define VEC_DOT_RUN(name, block_t, run)                                                    \
    static __device__ __forceinline__ float vec_dot_##name##_q8_1(                         \
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,              \
        const int & iqs) {                                                                 \
        float y[32];                                                                       \
        run((const block_t *) vbq, iqs, y);                                                \
        return dot_run_q8_1(y, bq8_1 + iqs);                                               \
    }
#undef VEC_DOT_RUN

/// iq4_xs, run `iqs` against its q8_1 block: sixteen code bytes, their low nibbles against the
/// first half of the activations and the high nibbles against the second.
static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq4_xs * b = (const block_iq4_xs *) vbq;
    const block_q8_1 * q8 = bq8_1 + iqs;
    const int * act = (const int *) q8->qs;
    const uint32_t * words = (const uint32_t *) (b->qs + 16 * iqs);
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum = __dp4a(levels_iq4_nl(words[i] & 0x0F0F0F0Fu), act[i], sum);
        sum = __dp4a(levels_iq4_nl((words[i] >> 4) & 0x0F0F0F0Fu), act[i + 4], sum);
    }
    const int low  = (b->scales_l[iqs / 2] >> (4 * (iqs % 2))) & 0x0F;
    const int high = (b->scales_h >> (2 * iqs)) & 3;
    const float dl = __half2float(b->d) * (float) ((low | (high << 4)) - 32);
    return dl * (float) sum * __low2float(q8->ds);
}

/// A byte mask per value for the low four bits of `bits`: 0xFF where a bit is set.
static __device__ __forceinline__ uint32_t sign_masks4(uint32_t bits) {
    return (((bits & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
}

/// The four signed bytes of `grid` negated where `masks` is 0xFF.
static __device__ __forceinline__ int negate_where(uint32_t grid, uint32_t masks) {
    return (int) __vsub4(grid ^ masks, masks);
}

/// Four magnitudes of `grid` against the next four activation values, signed by bits `signs`.
static __device__ __forceinline__ int dot4_signed(
    uint32_t grid, uint32_t signs, int activation, int acc) {
    return __dp4a(negate_where(grid, sign_masks4(signs)), activation, acc);
}

/// iq3_xxs, run `iqs` against its q8_1 block: the magnitudes and signs dotted in integers and
/// scaled once.
static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq3_xxs * b = (const block_iq3_xxs *) vbq;
    const block_q8_1 * q8 = bq8_1 + iqs;
    const int * act = (const int *) q8->qs;
    const uint8_t * codes = b->qs + 8 * iqs;
    const uint8_t * tail  = b->qs + QK_K / 4 + 4 * iqs;
    const uint32_t word = (uint32_t) tail[0] | ((uint32_t) tail[1] << 8) | ((uint32_t) tail[2] << 16) |
                          ((uint32_t) tail[3] << 24);
    int sum = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t signs = KSIGNS_IQ2XS[(word >> (7 * l)) & 127];
        sum = dot4_signed(IQ3XXS_GRID[codes[2 * l]],     signs,      act[2 * l],     sum);
        sum = dot4_signed(IQ3XXS_GRID[codes[2 * l + 1]], signs >> 4, act[2 * l + 1], sum);
    }
    const float db = __half2float(b->d) * (0.5f + (float) (word >> 28)) * 0.5f;
    return db * (float) sum * __low2float(q8->ds);
}

/// iq3_s, run `iqs` against its q8_1 block, in integers like iq3_xxs.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq3_s * b = (const block_iq3_s *) vbq;
    const block_q8_1 * q8 = bq8_1 + iqs;
    const int * act = (const int *) q8->qs;
    const int pair = iqs / 2;
    const int scale = (iqs % 2 == 0) ? (b->scales[pair] & 0x0F) : (b->scales[pair] >> 4);
    const uint32_t qh = b->qh[iqs];
    const uint8_t * q = b->qs + 8 * iqs;
    const uint8_t * signs = b->signs + 4 * iqs;
    int sum = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int i1 = q[2 * l]     | (int) ((qh << (8 - 2 * l)) & 256);
        const int i2 = q[2 * l + 1] | (int) ((qh << (7 - 2 * l)) & 256);
        const uint32_t sg = signs[l];
        sum = dot4_signed(IQ3S_GRID[i1], sg,      act[2 * l],     sum);
        sum = dot4_signed(IQ3S_GRID[i2], sg >> 4, act[2 * l + 1], sum);
    }
    return __half2float(b->d) * (float) (1 + 2 * scale) * (float) sum * __low2float(q8->ds);
}

/// iq2_s, run `iqs` against its q8_1 block: eight magnitudes per code, the two halves of the run
/// scaled apart.
static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq2_s * b = (const block_iq2_s *) vbq;
    const block_q8_1 * q8 = bq8_1 + iqs;
    const int * act = (const int *) q8->qs;
    const uint8_t * signs = b->qs + QK_K / 8;
    const uint32_t qh = b->qh[iqs];
    int low = 0, high = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int code = b->qs[4 * iqs + l] | (int) ((qh << (8 - 2 * l)) & 0x300);
        const uint64_t grid = IQ2S_GRID[code];
        const uint32_t sg = signs[4 * iqs + l];
        int t = dot4_signed((uint32_t) grid, sg, act[2 * l], 0);
        t = dot4_signed((uint32_t) (grid >> 32), sg >> 4, act[2 * l + 1], t);
        if (l < 2) low += t; else high += t;
    }
    const float d = __half2float(b->d);
    const float db0 = d * (0.5f + (float) (b->scales[iqs] & 0x0F)) * 0.25f;
    const float db1 = d * (0.5f + (float) (b->scales[iqs] >> 4)) * 0.25f;
    return (db0 * (float) low + db1 * (float) high) * __low2float(q8->ds);
}

/// q2_0: sixty-four weights, `y = (q - 1) . d`, four codes to a byte lowest first. A lane takes
/// the four bytes at `iqs` (sixteen weights), which lie in one of the two activation blocks.
static __device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_q2_0 * b = (const block_q2_0 *) vbq;
    const block_q8_1 * q8 = bq8_1 + iqs / 2;
    const int first = (iqs % 2) * 16;
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int w = 4 * iqs + j / 4;
        const int code = (b->qs[w] >> (2 * (j % 4))) & 3;
        sum += (code - 1) * (int) q8->qs[first + j];
    }
    return __half2float(b->d) * __low2float(q8->ds) * (float) sum;
}

// ------------------------------------------------------------------------------------------
// The dequantisers: one block into `yy`, by the threads of a block of the prefill kernel

#define DEQUANTISE_RUNS(name, block_t, run, values)                                        \
    template<typename dst_t>                                                               \
    inline __device__ void dequantize_block_##name(const void * __restrict__ vx,           \
                                                   dst_t * __restrict__ yy) {              \
        const block_t * b = (const block_t *) vx;                                          \
        for (int ib = threadIdx.x; ib < (values) / 32; ib += blockDim.x) {                 \
            float y[32];                                                                   \
            run(b, ib, y);                                                                 \
            for (int j = 0; j < 32; ++j) yy[32 * ib + j] = dst_t(y[j]);                    \
        }                                                                                  \
    }
DEQUANTISE_RUNS(iq4_xs,  block_iq4_xs,  iq4_xs_run,  QK_K)
DEQUANTISE_RUNS(iq3_xxs, block_iq3_xxs, iq3_xxs_run, QK_K)
DEQUANTISE_RUNS(iq3_s,   block_iq3_s,   iq3_s_run,   QK_K)
DEQUANTISE_RUNS(iq2_s,   block_iq2_s,   iq2_s_run,   QK_K)
#undef DEQUANTISE_RUNS

template<typename dst_t>
inline __device__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    if (threadIdx.x == 0) {
        float y[32];
        iq4_nl_run((const block_iq4_nl *) vx, y);
        for (int j = 0; j < 32; ++j) yy[j] = dst_t(y[j]);
    }
}

template<typename dst_t>
inline __device__ void dequantize_block_q2_0(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q2_0 * b = (const block_q2_0 *) vx;
    const float d = __half2float(b->d);
    for (int j = threadIdx.x; j < QK_Q2_0; j += blockDim.x) {
        yy[j] = dst_t((float) (((b->qs[j / 4] >> (2 * (j % 4))) & 3) - 1) * d);
    }
}
