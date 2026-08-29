// The GGUF block layouts and the dot products over them, for the CUDA kernels.
//
// A block format is a fixed number of values, their codes packed into bytes, and one or two
// scales that reconstruct them. Everything below follows from that: the structs mirror the
// byte layout a GGUF file stores, and each `vec_dot_*` walks a block against an int8-quantised
// activation, four codes at a time, through `dot4_i8`.
//
// The activation is always q8_1 - int8 codes plus a scale and the sum of the codes - because
// the offset formats need `Σ activations` to apply their minimum without a second pass.

#include "cuda_fp16.h"
#include "cuda_bf16.h"
// NVRTC compiles from a string with no system headers; the loader prepends `nvrtc_compat.h`,
// which defines the same types. Including <stdint.h> there is a hard error.
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif

#define UNUSED(x) (void)(x)

// 256 values per super-block, twelve bytes of packed six-bit scales. ggml can be built with
// 64-value super-blocks; nothing here reads such a file, so that shape is not carried.
#define QK_K 256
#define K_SCALE_SIZE 12

#define WARP_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define GGML_CUDA_DMMV_X 32
#define K_QUANTS_PER_ITERATION 2

typedef uint16_t f16_bits;
typedef float dfloat;
typedef float2 dfloat2;
typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);

/// Sum across the warp: each lane exchanges with the lane `mask` away and adds, halving the
/// distance each round, so after five rounds every lane holds the total.
static __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE);
    }
    return x;
}

/// The formats a MoE launch can be handed, by the id its caller sends.
///
/// The id is a wire value: it crosses from Rust as an int and every kernel that switches on it
/// must agree about what each number means. Written twice, the two agreed until one of them
/// gained a format - so it is written once, and a kernel builds its own switch from it.
///
/// Each row gives the id, the format's name, and the three numbers its blocks are read with.
/// The name is what `block_`, `vec_dot_` and `dequantize_block_` are built from.
#define MOE_GGUF_FORMATS(X)                          \
    X(0, q8_0,  QK8_0,    QI8_0,    VDR_Q8_0_Q8_1_MMVQ)   \
    X(1, q4_K,  QK_K,     QI4_K,    VDR_Q4_K_Q8_1_MMVQ)   \
    X(2, q2_K,  QK_K,     QI2_K,    VDR_Q2_K_Q8_1_MMVQ)   \
    X(3, q3_K,  QK_K,     QI3_K,    VDR_Q3_K_Q8_1_MMVQ)   \
    X(4, q5_K,  QK_K,     QI5_K,    VDR_Q5_K_Q8_1_MMVQ)   \
    X(5, q6_K,  QK_K,     QI6_K,    VDR_Q6_K_Q8_1_MMVQ)   \
    X(6, q5_0,  QK5_0,    QI5_0,    VDR_Q5_0_Q8_1_MMVQ)   \
    X(7, mxfp4, QK_MXFP4, QI_MXFP4, VDR_MXFP4_Q8_1_MMVQ)

/// The subset of the table above that a block dequantiser exists for.
///
/// MXFP4 is served by a dot product and never by a dequantise-then-multiply, so the tensor-core
/// path has no arm for it. Splitting the list says which kernels can take which format instead
/// of leaving one of them with a switch that silently falls through.
#define MOE_GGUF_DEQUANTISABLE(X)                    \
    X(0, q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ)     \
    X(1, q4_K, QK_K,  QI4_K, VDR_Q4_K_Q8_1_MMVQ)     \
    X(2, q2_K, QK_K,  QI2_K, VDR_Q2_K_Q8_1_MMVQ)     \
    X(3, q3_K, QK_K,  QI3_K, VDR_Q3_K_Q8_1_MMVQ)     \
    X(4, q5_K, QK_K,  QI5_K, VDR_Q5_K_Q8_1_MMVQ)     \
    X(5, q6_K, QK_K,  QI6_K, VDR_Q6_K_Q8_1_MMVQ)     \
    X(6, q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ)

/// `a` divided by `b`, rounded up - how many whole `b`s it takes to cover `a`.
///
/// Every launch that turns a problem size into a grid asks this, and three files spelled it
/// out under three names. Integers only: a kernel counts blocks, never fractions of one.
constexpr __host__ __device__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

/// `size` raised to the next multiple of `padding`, or left alone when there is no padding to
/// apply. What a row's length becomes when a format needs it aligned to its block.
constexpr __host__ __device__ int pad_to(int size, int padding) {
    return padding == 0 ? size : ceil_div(size, padding) * padding;
}

/// The same butterfly, keeping the larger of the two.
static __device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE));
    }
    return x;
}

/// Four bytes at `i32` as one int, where the pointer is only 2-byte aligned.
///
/// A block's code array starts after an f16 scale, so it can land on an odd 4-byte boundary;
/// reading it as an `int*` would be undefined. Two 16-bit reads are defined and cost nothing
/// the compiler cannot fold back together. Signed and unsigned differ only in the pointer
/// type - the bytes are the same, so one body serves both.
template <typename T>
static __device__ __forceinline__ int four_bytes_unaligned(const T * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32);
    return (int) x16[0] | ((int) x16[1] << 16);
}

/// Four bytes at `i32`, where the pointer IS 4-byte aligned.
template <typename T>
static __device__ __forceinline__ int four_bytes(const T * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32));
}

/// Four int8 products accumulated into `c`.
///
/// Every architecture this ships for is sm_80 or later, so `__dp4a` is always available and
/// the byte-by-byte fallback ggml carries for older cards is not.
static __device__ __forceinline__ int dot4_i8(const int a, const int b, int c) {
    return __dp4a(a, b, c);
}

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
// Thirty-two weights, one shared scale, four bits each. A code is read centred: the stored
// nibble less eight, times the scale. Low nibbles hold the first half of the block, high
// nibbles the second - not alternate weights.
typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(f16_bits) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
// As q4_0 with an offset instead of a centre: a code is `nibble * dm.x + dm.y`, which lets a
// block whose weights all sit on one side of zero use its whole range.
typedef struct {
    half2   dm;
    uint8_t qs[QK4_1 / 2];
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(f16_bits) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
// q4_0 with one more bit per weight. The extra bit cannot share a nibble, so all thirty-two of
// them are gathered into a separate word, one bit per weight in block order.
typedef struct {
    half    d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(f16_bits) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
// q4_1's scale-and-offset pair with q5_0's fifth-bit plane.
typedef struct {
    half2   dm;
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(f16_bits) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
// Thirty-two signed bytes and the scale they share - no packing, and nothing to unpack.
typedef struct {
    half   d;
    int8_t qs[QK8_0];
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(f16_bits) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
// The activation format. `ds.y` is the sum of the block's codes already multiplied by the
// scale: a weight format carrying an offset needs that sum to correct its dot product, and
// computing it here once saves every kernel from walking the codes a second time.
typedef struct {
    half2  ds;
    int8_t qs[QK8_0];
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(f16_bits) + QK8_0, "wrong q8_1 block size/padding");

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
// A superblock of 256 weights split into sixteen sub-blocks of sixteen. Each sub-block gets a
// four-bit scale and a four-bit offset sharing one byte, and `dm` is what those two are read
// against: a weight is `code * (dm.x * scale) - dm.y * offset`.
typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    half2   dm;
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(f16_bits) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
// Three bits per weight, and three does not divide a byte: the low two are packed four to a
// byte and the third lives in its own bit plane. Sixteen sub-blocks each carry a six-bit signed
// scale, packed twelve bytes for sixteen scales.
typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[K_SCALE_SIZE];
    half    d;
} block_q3_K;
//static_assert(sizeof(block_q3_K) == sizeof(f16_bits) + QK_K / 4 + QK_K / 8 + K_SCALE_SIZE, "wrong q3_K block size/padding");

/// One of the sixteen six-bit signed scales a q3_K superblock packs into twelve bytes.
///
/// Four low bits, eight to a byte, in the first `QK_K/32` bytes; two high bits, sixteen to a
/// byte, in the four that follow; and the assembled number offset by 32 to make it signed.
/// Both the dot product and the dequantiser want it, and a wrong unpack does not fail - it
/// returns a plausible scale and the weights come out quietly wrong.
static __device__ __forceinline__ int q3_K_scale(const uint8_t * __restrict__ scales, const int is) {
    const int low  = (scales[is % (QK_K/32)] >> (4 * (is / (QK_K/32)))) & 0xF;
    const int high = (scales[(QK_K/32) + is % (QK_K/64)] >> (2 * (is / (QK_K/64)))) & 3;
    return (low | (high << 4)) - 32;
}

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
// Eight sub-blocks of thirty-two, each with a six-bit scale and a six-bit offset. Twelve bytes
// hold all sixteen of those six-bit numbers; `get_scale_min_k4` below is the one statement of
// how they are laid out, and `dm` is what they are multiplied by.
typedef struct {
    half2   dm;
    uint8_t scales[3*QK_K/64];
    uint8_t qs[QK_K/2];
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(f16_bits) + 3*QK_K/64 + QK_K/2, "wrong q4_K block size/padding");

/// The scale and the minimum of one sub-block, out of the twelve bytes q4_K and q5_K pack eight
/// pairs into.
///
/// Six bits each, and eight of each is ninety-six bits: the first four pairs are stored whole
/// in the first eight bytes, and the last four take their low four bits from the last four
/// bytes and their top two from the spare top bits of the first four's. Which is why the
/// second half of the function reaches backwards.
///
/// A wrong unpack does not fail - it returns a plausible scale, and the weights come out
/// quietly wrong.
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    // The width of both numbers, and the reason the second half reassembles four bits plus two.
    constexpr uint8_t field = 0x3F;
    if (j < 4) {
        // Whole, a byte apiece: the scale in the first four bytes, its minimum four bytes on.
        d = q[j] & field;
        m = q[j + 4] & field;
        return;
    }
    // Past that, ONE byte of the last four carries the low nibble of both this pair's numbers,
    // and each top-up comes from the spare high bits of a byte the first four already used.
    const uint8_t nibbles = q[j + 4];
    d = (nibbles & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (nibbles >>  4) | ((q[j]     >> 6) << 4);
}

/// The (scale, minimum) pairs of two consecutive sub-blocks, laid out as `{sc0, sc1, m0, m1}`  - 
/// which is what a dot product wants, one array per role.
static __device__ __forceinline__ void unpack_quarter_scales_q45_K(
    const uint8_t * __restrict__ packed, const int first, uint8_t * __restrict__ out) {
    get_scale_min_k4(first + 0, packed, out[0], out[2]);
    get_scale_min_k4(first + 1, packed, out[1], out[3]);
}

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
// q4_K's scales and offsets with a fifth-bit plane over the whole superblock - one bit per
// weight, so 256 of them in 32 bytes.
typedef struct {
    half2   dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(f16_bits) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
// Six bits per weight, split four and two across two planes rather than packed across byte
// boundaries. Sixteen sub-blocks of sixteen, each with a full signed byte of scale - no six-bit
// packing here - read against the superblock's `d`.
typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(f16_bits) + 13*QK_K/16, "wrong q6_K block size/padding");

// The superblock activation format: 256 signed bytes, one float scale, and the sums of each
// run of sixteen codes. Those sums are what a weight format with a per-sub-block offset needs,
// and they are kept unscaled because each sub-block applies its own.
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");


// How many 32-bit words of a block one thread takes in a single dot, per format and per kernel
// shape. It is a property of the format's packing, not a tuning knob: a format whose quants
// need two words to make one useful group cannot be walked one word at a time.

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

/// Put the fifth bit of each of four codes where a nibble expects it.
///
/// `qh` holds one bit per value, packed for the whole block; a nibble sits every eight bits
/// of the assembled int, so bit `n` has to land at position `8n + 4`. Written as four shifted
/// masks because the distance differs per code and a loop would not fold.
#define FIFTH_BIT_LOW(qh)  ((((qh) <<  4) & 0x00000010) | (((qh) << 11) & 0x00001000) \
                          | (((qh) << 18) & 0x00100000) | (((qh) << 25) & 0x10000000))
#define FIFTH_BIT_HIGH(qh) ((((qh) >> 12) & 0x00000010) | (((qh) >>  5) & 0x00001000) \
                          | (((qh) <<  2) & 0x00100000) | (((qh) <<  9) & 0x10000000))

/// The nibble product, which four of the five 32-value formats share.
///
/// One int holds eight codes: the low four meet one activation int and the high four the next.
/// `five_bit` merges the bit a five-bit format keeps apart in `qh`; a four-bit one passes no
/// `vh` at all.
template <int vdr, bool five_bit>
static __device__ __forceinline__ int dot_nibbles_q8_1(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u) {
    int dot = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int low  = (vl[i] >> 0) & 0x0F0F0F0F;
        int high = (vl[i] >> 4) & 0x0F0F0F0F;
        if constexpr (five_bit) {
            low  |= FIFTH_BIT_LOW(vh[i]);
            high |= FIFTH_BIT_HIGH(vh[i]);
        }
        dot = dot4_i8(low,  u[2*i + 0], dot);
        dot = dot4_i8(high, u[2*i + 1], dot);
    }
    return dot;
}

/// A symmetric format's total. The codes carry a constant bias, so rather than subtract it from
/// every code the whole of it leaves the loop as `-bias.Σa` - and `da.Σa` is what q8_1 already
/// carries in the second half of its scale pair.
static __device__ __forceinline__ float total_biased(
    const int dot, const float & d, const float bias, const half2 & ds8) {
    const float2 act = __half22float2(ds8);   // .x = activation scale, .y = scale x Σcodes
    return d * (dot * act.x - bias * act.y);
}

/// An asymmetric format's total. `q.d + m` over a block is `d.da.Σ(q.a) + m.(da.Σa)`, and the
/// right-hand term is one multiply. `share` is how many threads cover the block, so that each
/// applies only its own part of the minimum.
static __device__ __forceinline__ float total_with_minimum(
    const int dot, const half2 & dm, const half2 & ds8, const int share) {
    const float2 weight = __half22float2(dm);  // .x = scale, .y = minimum
    const float2 act = __half22float2(ds8);
    return dot * weight.x * act.x + (weight.y * act.y) / share;
}

/// q4_0: four-bit codes biased by eight, `(code - 8).d`. `dp4a` wants an unsigned left
/// operand, which is what the raw nibbles are.
template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {
    return total_biased(dot_nibbles_q8_1<vdr, false>(v, nullptr, u), d4, 8 * vdr / QI4_0, ds8);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

/// q4_1: unsigned four-bit codes with a per-block minimum, `q.d + m`.
template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {
    return total_with_minimum(dot_nibbles_q8_1<vdr, false>(v, nullptr, u), dm4, ds8,
                              QI8_1 / (vdr * QR4_1));
}

/// The nibble formats' dot under one name, told apart by what the weight's scale carries: a
/// bare float is a scale, a half2 is a scale with the minimum that goes with it. The packing
/// they read is the same, so a caller that has the scale has already said which it wants.
template <int vdr> static __device__ __forceinline__ float vec_dot_nibbles_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {
    return vec_dot_q4_0_q8_1_impl<vdr>(v, u, d4, ds8);
}
template <int vdr> static __device__ __forceinline__ float vec_dot_nibbles_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {
    return vec_dot_q4_1_q8_1_impl<vdr>(v, u, dm4, ds8);
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

/// q5_0: five-bit codes biased by sixteen, the fifth bit held apart in `qh`.
template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {
    return total_biased(dot_nibbles_q8_1<vdr, true>(vl, vh, u), d5, 16 * vdr / QI5_0, ds8);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

/// q5_1: q5_0's fifth bit with q4_1's minimum, and no bias to fold - the codes are unsigned as
/// they stand.
template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {
    return total_with_minimum(dot_nibbles_q8_1<vdr, true>(vl, vh, u), dm5, ds8,
                              QI8_1 / (vdr * QR5_1));
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

/// q8_0 against a q8_1 activation: both sides are already signed bytes, so the block IS the
/// operand and there is nothing to unpack or bias.
template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {

    int dot = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        dot = dot4_i8(v[i], u[i], dot);
    }
    return d8_0 * d8_1 * dot;
}

/// q8_1 against a q8_1 activation. Both carry a minimum, but only the WEIGHT's is applied  - 
/// the activation's second scale is the sum it contributes, not an offset of its own.
template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int dot = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        dot = dot4_i8(v[i], u[i], dot);
    }

    const float2 weight = __half22float2(dm8);
    const float2 act = __half22float2(ds8);
    return dot * weight.x * act.x + (weight.y * act.y) / (QI8_1 / vdr);
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
/// q2_K against a q8_1 activation, one thread's share of a super-block.
///
/// Two bits per value, sixteen to a sub-block, with a four-bit scale and a four-bit minimum
/// sharing one byte. Both are applied here rather than to the codes: the scale multiplies
/// this quarter's dot product, and the minimum - constant over the sub-block - multiplies
/// the sum of the activations, which is obtained by dotting the activation against a word
/// filled with that same minimum. That is the whole reason `m` is broadcast into four bytes.
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float scaled = 0.0f;   // Σ scale . (codes . activations)
    float offset = 0.0f;   // Σ minimum . Σ activations

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int packed = scales[2*i];             // scale in the low nibble, minimum in the high
        const int codes = (v >> (2*i)) & 0x03030303; // this quarter's two-bit codes

        scaled += d8[i] * (dot4_i8(codes, u[i], 0) * (packed & 0xF));

        int minimum = packed >> 4;
        minimum |= minimum << 8;
        minimum |= minimum << 16;                    // the same value in all four bytes
        offset += d8[i] * dot4_i8(minimum, u[i], 0);
    }

    const float2 block = __half22float2(dm2);        // .x = scale, .y = minimum
    return block.x * scaled - block.y * offset;
}

// contiguous u/y values
/// q2_K, tiled. Scale and minimum still share a byte, and the minimum still reaches the
/// activation sum through a word filled with itself - but the sum accumulates across the
/// whole call rather than per run, because the minimum is the same for every value the
/// scale governs.
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {

    int scaled = 0;
    int offset = 0;

#pragma unroll
    for (int run = 0; run < QI8_1; run += QI8_1/2) {
        const int packed = scales[run / (QI8_1/2)];

        int minimum = packed >> 4;
        minimum |= minimum << 8;
        minimum |= minimum << 16;

        int in_run = 0;
#pragma unroll
        for (int i = run; i < run + QI8_1/2; ++i) {
            in_run = dot4_i8(v[i], u[i], in_run);
            offset = dot4_i8(minimum, u[i], offset);
        }
        scaled += in_run * (packed & 0xF);
    }

    const float2 block = __half22float2(dm2);
    return d8 * (block.x * scaled - block.y * offset);
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
// The superblock formats whose sub-blocks carry a scale and NO minimum, one thread's share of
// a super-block at a time.
//
// A lane's span is `qr` runs of four codes; each run meets one activation int, and the run's
// own scale multiplies that product before the activation's scale weighs it. A `plane` states
// the pair the format decides: how a run's codes come out of the two weight planes, and which
// of the super-block's scales governs it.

/// q3_K: three bits per value - two in `qs`, the third in a bit-plane - and no minimum. The
/// third bit is worth -4 when CLEAR, which is why it is SUBTRACTED rather than added. Sixteen
/// sub-blocks over the super-block, so consecutive runs are two scales apart.
struct vec_dot_plane_q3_K {
    using scale_t = uint8_t;
    static __device__ __forceinline__ int codes(const int & vl, const int & vh, const int i) {
        const int two_bits = (vl >> (2*i)) & 0x03030303;
        // Bit i of the plane, moved to position two - where a third bit is worth four.
        const int third = ((vh >> i) << 2) & 0x04040404;
        return __vsubss4(two_bits, third);
    }
    static __device__ __forceinline__ int scale(const uint8_t * s, const int first, const int i) {
        return q3_K_scale(s, first + 2*i);
    }
};

template <int qr, typename plane>
static __device__ __forceinline__ float vec_dot_scaled_runs(
    const int & vl, const int & vh, const int * __restrict__ u,
    const typename plane::scale_t * __restrict__ scales, const int & first_scale,
    const float & d, const float * __restrict__ d8) {

    float total = 0.0f;

#pragma unroll
    for (int i = 0; i < qr; ++i) {
        total += d8[i] * (dot4_i8(plane::codes(vl, vh, i), u[i], 0)
                          * plane::scale(scales, first_scale, i));
    }

    return d * total;
}

// contiguous u/y values
// -- the tiled-matmul side --
//
// The `_mmq` twin of each dot product above. Same arithmetic, different operands: the tiled
// matmul has already unpacked the codes into contiguous ints and unpacked the scales, so
// these take plain arrays and a single activation scale instead of walking a block. What
// they still have to do is apply a scale per SUB-BLOCK, which is why each one accumulates in
// runs of `QI8_1/2` before multiplying.

/// q3_K, tiled: the third bit and the bias are already folded into `v`, so only the
/// per-sub-block scale is left.
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int total = 0;
#pragma unroll
    for (int run = 0; run < QR3_K*VDR_Q3_K_Q8_1_MMQ; run += QI8_1/2) {
        int in_run = 0;
        for (int i = run; i < run + QI8_1/2; ++i) {
            in_run = dot4_i8(v[i], u[i], in_run);
        }
        total += in_run * scales[run / (QI8_1/2)];
    }
    return d3 * d8 * total;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
/// q4_K against a q8_1 activation, one thread's share of a super-block.
///
/// Four bits per value, thirty-two to a sub-block, with a six-bit scale and a six-bit
/// minimum already unpacked by the caller. The minimum needs `Σ activations` over the
/// sub-block, and there is no `bsums` to read on this side - so it is obtained by dotting the
/// activation against a word of ones, which `dp4a` does in the same instruction as the real
/// product.
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float scaled = 0.0f;   // Σ scale . (codes . activations)
    float offset = 0.0f;   // Σ minimum . Σ activations

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int lo = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int hi = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int product = dot4_i8(hi, u[2*i + 1], dot4_i8(lo, u[2*i + 0], 0));
        const int act_sum = dot4_i8(0x01010101, u[2*i + 1], dot4_i8(0x01010101, u[2*i + 0], 0));

        scaled += d8[i] * (product * sc[i]);
        offset += d8[i] * (act_sum * m[i]);
    }

    const float2 block = __half22float2(dm4);   // .x = scale, .y = minimum
    return block.x * scaled - block.y * offset;
}

// contiguous u/y values
/// q4_K, tiled. The activation arrives as one q8_1 block per sub-block, so its `s` - the
/// scale times the sum of its codes - is already the `Σ activations` the minimum needs. No
/// word of ones here: the number is read, not computed.
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float scaled = 0.0f;
    float offset = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int in_sub = 0;
#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            in_sub = dot4_i8((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], in_sub);
        }

        const float2 act = __half22float2(ds8[i]);  // .x = scale, .y = scale x Σcodes
        scaled += act.x * (sc[i] * in_sub);
        offset += act.y * m[i];
    }

    const float2 block = __half22float2(dm4);
    return block.x * scaled - block.y * offset;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
/// q5_K against a q8_1 activation: q4_K with a fifth bit taken from a plane.
///
/// The bit is worth sixteen and the codes are unsigned, so it is OR-ed into position four  - 
/// no bias to undo, unlike q5_0.
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u,
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m, const half2 & dm5,
    const float * __restrict__ d8) {

    float scaled = 0.0f;
    float offset = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int lo = ((vl[0] >> (4*i)) & 0x0F0F0F0F) | (((vh[0] >> i) << 4) & 0x10101010);
        const int hi = ((vl[1] >> (4*i)) & 0x0F0F0F0F) | (((vh[1] >> i) << 4) & 0x10101010);

        const int product = dot4_i8(lo, u[2*i + 0], dot4_i8(hi, u[2*i + 1], 0));
        const int act_sum = dot4_i8(0x01010101, u[2*i + 0], dot4_i8(0x01010101, u[2*i + 1], 0));

        scaled += d8[i] * (product * sc[i]);
        offset += d8[i] * (act_sum * m[i]);
    }

    const float2 block = __half22float2(dm5);
    return block.x * scaled - block.y * offset;
}

// contiguous u/y values
/// q5_K, tiled. Identical in shape to q4_K's twin - the fifth bit was folded into `v` before
/// the call, so nothing here distinguishes the two formats.
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const half2 * __restrict__ ds8) {

    float scaled = 0.0f;
    float offset = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int in_sub = 0;
#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            in_sub = dot4_i8(v[i*QI8_1 + j], u[i*QI8_1 + j], in_sub);
        }

        const float2 act = __half22float2(ds8[i]);
        scaled += act.x * (sc[i] * in_sub);
        offset += act.y * m[i];
    }

    const float2 block = __half22float2(dm5);
    return block.x * scaled - block.y * offset;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
/// q6_K: six bits per value - four in `ql`, two in `qh` - sixteen to a sub-block, with a full
/// signed byte of scale and a fixed zero point of 32. No minimum to apply: the 32 is constant,
/// so it comes off the assembled code with one saturating vector subtract. A run of four codes
/// is a quarter of what one scale governs, which is the step through the scale array.
struct vec_dot_plane_q6_K {
    using scale_t = int8_t;
    static __device__ __forceinline__ int codes(const int & vl, const int & vh, const int i) {
        const int low  = (vl >> (4*i)) & 0x0F0F0F0F;
        const int high = ((vh >> (4*i)) << 4) & 0x30303030;
        return __vsubss4(low | high, 0x20202020);
    }
    static __device__ __forceinline__ int scale(const int8_t * s, const int first, const int i) {
        return s[first + 4*i];
    }
};

// contiguous u/y values
/// q6_K, tiled. Two q6_K scales fall inside one q8_1 activation block - the sub-block is
/// sixteen values and the activation block thirty-two - so each round accumulates two
/// independent sums and applies a different scale to each before they meet.
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float total = 0.0f;

#pragma unroll
    for (int run = 0; run < VDR_Q6_K_Q8_1_MMQ; run += 4) {
        int2 pair = {0, 0};
#pragma unroll
        for (int i = run; i < run + 2; ++i) {
            pair.x = dot4_i8(v[2*i + 1], u[2*i + 1], dot4_i8(v[2*i + 0], u[2*i + 0], pair.x));
            pair.y = dot4_i8(v[2*i + 5], u[2*i + 5], dot4_i8(v[2*i + 4], u[2*i + 4], pair.y));
        }
        total += d8[run/4] * (sc[run/2 + 0] * pair.x + sc[run/2 + 1] * pair.y);
    }

    return d6 * total;
}

// The 32-value formats, against a q8_1 activation.
//
// A lane reads `vdr` ints of the weight block and the activation ints that cover the same
// weights. A four-bit format packs two weights to a byte, so one weight int meets TWO activation
// ints, `qi` apart; an eight-bit one meets exactly one.
//
// Whether the low bits are read four-byte aligned is the block's header talking: a two-byte
// scale leaves the payload on a two-byte boundary, a four-byte scale-and-minimum leaves it on a
// four-byte one. Reading the aligned case as unaligned would work and cost an instruction;
// reading the unaligned case as aligned would not work at all.

/// A four-bit format whose weights are all in the low nibbles.
#define VEC_DOT_NIBBLES(name, block, scale, read, qi, vdr)                                     \
    static __device__ __forceinline__ float name(                                             \
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) { \
        const block * bq = (const block *) vbq;                                               \
        int v[vdr];                                                                           \
        int u[2*vdr];                                                                         \
        _Pragma("unroll")                                                                     \
        for (int i = 0; i < vdr; ++i) {                                                       \
            v[i]       = read(bq->qs, iqs + i);                                               \
            u[2*i + 0] = four_bytes(bq8_1->qs, iqs + i);                                      \
            u[2*i + 1] = four_bytes(bq8_1->qs, iqs + i + qi);                                 \
        }                                                                                     \
        return name##_impl<vdr>(v, u, bq->scale, bq8_1->ds);                                  \
    }

/// A five-bit format: four bits with the others in a plane of their own. The high-bit word
/// covers the whole block, so every lane reads the same word and shifts its own values out.
#define VEC_DOT_NIBBLES_PLUS_ONE(name, block, scale, read, qi, vdr)                            \
    static __device__ __forceinline__ float name(                                             \
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) { \
        const block * bq = (const block *) vbq;                                               \
        int vl[vdr];                                                                          \
        int vh[vdr];                                                                          \
        int  u[2*vdr];                                                                        \
        _Pragma("unroll")                                                                     \
        for (int i = 0; i < vdr; ++i) {                                                       \
            vl[i]      = read(bq->qs, iqs + i);                                               \
            vh[i]      = read(bq->qh, 0) >> (4 * (iqs + i));                                  \
            u[2*i + 0] = four_bytes(bq8_1->qs, iqs + i);                                      \
            u[2*i + 1] = four_bytes(bq8_1->qs, iqs + i + qi);                                 \
        }                                                                                     \
        return name##_impl<vdr>(vl, vh, u, bq->scale, bq8_1->ds);                             \
    }

//              name                block       scale  low-bit reader          qi      vdr
VEC_DOT_NIBBLES(vec_dot_q4_0_q8_1, block_q4_0, d,  four_bytes_unaligned, QI4_0, VDR_Q4_0_Q8_1_MMVQ)
VEC_DOT_NIBBLES(vec_dot_q4_1_q8_1, block_q4_1, dm, four_bytes,           QI4_1, VDR_Q4_1_Q8_1_MMVQ)
VEC_DOT_NIBBLES_PLUS_ONE(
    vec_dot_q5_0_q8_1, block_q5_0, d,  four_bytes_unaligned, QI5_0, VDR_Q5_0_Q8_1_MMVQ)
VEC_DOT_NIBBLES_PLUS_ONE(
    vec_dot_q5_1_q8_1, block_q5_1, dm, four_bytes,           QI5_1, VDR_Q5_1_Q8_1_MMVQ)

#undef VEC_DOT_NIBBLES
#undef VEC_DOT_NIBBLES_PLUS_ONE

/// Eight bits per weight: one weight int meets exactly one activation int, and the activation's
/// per-block sum is not needed because there is no offset to correct for.
static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;
    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = four_bytes_unaligned(bq8_0->qs, iqs + i);
        u[i] = four_bytes(bq8_1->qs, iqs + i);
    }
    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

// ------------------------------------------------------------
// The superblock mat-vec dots.
//
// A lane takes four ints of one weight superblock and meets the activation blocks that cover
// the same weights. How many, and how far apart they sit, is what the format decides: Q2_K and
// Q3_K each cover their span with consecutive activation blocks, Q6_K with every other one,
// because six bits per weight means one weight block spans twice the activation.

/// The activation blocks a lane's span meets, and their scales.
///
/// `qr` blocks, `stride` apart, starting at `offset`. From each the lane takes `words` groups
/// of four codes beginning at `word`; a second group sits half a block further on, which is
/// where a format that reads two weight ints per activation block finds its other half.
template <int qr, int stride, int words>
static __device__ __forceinline__ void gather_q8_1(
        const block_q8_1 * __restrict__ bq8_1, const int offset, const int word,
        int * __restrict__ u, float * __restrict__ d8) {
#pragma unroll
    for (int i = 0; i < qr; ++i) {
        const block_q8_1 * b = bq8_1 + offset + stride*i;
#pragma unroll
        for (int w = 0; w < words; ++w) {
            u[words*i + w] = four_bytes(b->qs, word + w*(QI8_1/2));
        }
        d8[i] = __low2float(b->ds);
    }
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = four_bytes(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

    gather_q8_1<QR2_K, 1, 1>(bq8_1, bq8_offset, iqs % QI8_1, u, d8);

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = four_bytes_unaligned(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~four_bytes_unaligned(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

    gather_q8_1<QR3_K, 1, 1>(bq8_1, bq8_offset, iqs % QI8_1, u, d8);

    return vec_dot_scaled_runs<QR3_K, vec_dot_plane_q3_K>(
        vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

// Q4_K and Q5_K, a quarter of a superblock at a time - the superblock formats whose sub-blocks
// carry a scale AND a minimum, against a q8_1 activation.
//
// A lane takes one quarter of the superblock. `quarter` names which - it advances in steps of
// `qr` so that consecutive lanes land on consecutive activation blocks - and inside that
// quarter the lane's two weight ints sit four int units apart, which is sixteen bytes, one
// sub-block. The scale and the minimum of each come out of the twelve packed bytes, and the
// activation blocks covering the same weights are gathered by the same offset. What differs is
// the weight plane: Q4_K's is nibbles alone, Q5_K adds a fifth bit from a plane covering the
// whole superblock, and which bit of it a lane wants is the quarter it is reading.

/// Four bits per weight.
#define VEC_DOT_QUARTER(name, block, qr)                                                      \
    static __device__ __forceinline__ float name(                                             \
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) { \
        const block * bq = (const block *) vbq;                                               \
        const int quarter = qr * ((iqs/2) / (QI8_1/2));                                       \
        const int * q = (const int *)(bq->qs + 16 * quarter + 4 * ((iqs/2) % 4));             \
        int v[2] = { q[0], q[4] };                                                            \
        uint8_t sm[4];  /* {sc0, sc1, m0, m1} */                                              \
        unpack_quarter_scales_q45_K(bq->scales, quarter, sm);                                 \
        int   u[2*qr];                                                                        \
        float d8[qr];                                                                         \
        gather_q8_1<qr, 1, 2>(bq8_1, quarter, (iqs/2) % (QI8_1/2), u, d8);                    \
        return name##_impl_vmmq(v, u, sm, sm + 2, bq->dm, d8);                                 \
    }

/// Five bits: the fifth in a plane of its own, shifted down by the quarter this lane holds.
#define VEC_DOT_QUARTER_PLUS_ONE(name, block, qr)                                             \
    static __device__ __forceinline__ float name(                                             \
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) { \
        const block * bq = (const block *) vbq;                                               \
        const int quarter = qr * ((iqs/2) / (QI8_1/2));                                       \
        const int * ql = (const int *)(bq->qs + 16 * quarter + 4 * ((iqs/2) % 4));            \
        const int * qh = (const int *)(bq->qh + 4 * ((iqs/2) % 4));                           \
        int vl[2] = { ql[0], ql[4] };                                                         \
        int vh[2] = { qh[0] >> quarter, qh[4] >> quarter };                                   \
        uint8_t sm[4];  /* {sc0, sc1, m0, m1} */                                              \
        unpack_quarter_scales_q45_K(bq->scales, quarter, sm);                                 \
        int   u[2*qr];                                                                        \
        float d8[qr];                                                                         \
        gather_q8_1<qr, 1, 2>(bq8_1, quarter, (iqs/2) % (QI8_1/2), u, d8);                    \
        return name##_impl_vmmq(vl, vh, u, sm, sm + 2, bq->dm, d8);                            \
    }

VEC_DOT_QUARTER(vec_dot_q4_K_q8_1, block_q4_K, QR4_K)
VEC_DOT_QUARTER_PLUS_ONE(vec_dot_q5_K_q8_1, block_q5_K, QR5_K)

#undef VEC_DOT_QUARTER
#undef VEC_DOT_QUARTER_PLUS_ONE


static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = four_bytes_unaligned(bq6_K->ql, iqs);
    const int vh = four_bytes_unaligned(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    int    u[QR6_K];
    float d8[QR6_K];

    gather_q8_1<QR6_K, 2, 1>(bq8_1, bq8_offset, iqs % QI8_1, u, d8);

    return vec_dot_scaled_runs<QR6_K, vec_dot_plane_q6_K>(
        vl, vh, u, bq6_K->scales, scale_offset, bq6_K->d, d8);
}

/// The carriers an activation can arrive in, read as float. Which one a caller holds is its
/// own business; the quantiser below wants a number.
static __device__ __forceinline__ float as_float(const nv_bfloat16 v) { return __bfloat162float(v); }
static __device__ __forceinline__ float as_float(const half v) { return __half2float(v); }
static __device__ __forceinline__ float as_float(const float v) { return v; }

/// A row of activations into q8_1 blocks: one thread per value, one warp per 32-value block.
///
/// The warp finds the block's largest magnitude and its sum in two reductions, and one lane
/// writes the pair the block shares - the scale that puts the products back on their footing,
/// and the sum the weight formats with a per-sub-block minimum need. Padding past `kx`
/// quantises as zero, so a row that does not fill its last block still states one.
///
/// The source carrier is a parameter because it is the only thing that varies: every entry
/// point that quantises an activation, whatever it was handed, reaches this.
template<typename src_t>
static __device__ __forceinline__ void quantize_row_to_q8_1(
    const src_t * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) {
        return;
    }
    const int iy = blockDim.y*blockIdx.y + threadIdx.y;
    const int i_padded = iy*kx_padded + ix;
    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? as_float(x[iy*kx + ix]) : 0.0f;
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);

    const float d = amax / 127;
    y[ib].qs[iqs] = amax == 0.0f ? 0 : roundf(xi / d);

    // One lane of the block writes the pair the block shares. What that pair is and where it
    // sits is the activation format's business, so it goes out in the one move the field takes.
    if (iqs == 0) {
        y[ib].ds = __floats2half2_rn(d, sum);
    }
}

// The NVRTC units that prepend this header fetch this kernel BY NAME, so there it needs C
// linkage. The nvcc side includes the same header in a dozen translation units and links
// them together, where one exported definition each is a duplicate symbol - internal
// linkage is what that side needs. The two compilation models differ, and this is where.
#ifdef __CUDACC_RTC__
extern "C"
#else
static
#endif
__global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    quantize_row_to_q8_1<float>(x, vy, kx, kx_padded);
}

/// A dequantised weight put away in whatever carrier the caller keeps its output in.
///
/// The destination is the argument, so the carrier is deduced at the store instead of being
/// named again at every call site. `half` and `float` need no help - the half converts itself.
/// There is no direct half-to-bfloat16 route, so that one widens through float, which is exact.
static __device__ __forceinline__ void store_half(half & out, half val) { out = val; }
static __device__ __forceinline__ void store_half(float & out, half val) { out = val; }
static __device__ __forceinline__ void store_half(nv_bfloat16 & out, half val) {
    out = __float2bfloat16(__half2float(val));
}

// ------------------------------------------------------------
// The two-bit superblocks, dequantised.
//
// Q2_K and Q3_K both pack the low two bits of every weight four to a byte, and both divide
// their 256 weights into sixteen sub-blocks of sixteen. What those two bits mean differs, and
// so does where the sub-block's footing comes from:
//
//   Q2_K  an unsigned magnitude, against a 4-bit scale and a 4-bit minimum, both of which the
//         superblock's own (d, m) pair multiplies out
//   Q3_K  a third bit borrowed from a mask covering the superblock, signed by an offset of
//         four, against a 6-bit scale split across two planes and offset by 32
//
// A lane owns one byte of the two-bit plane, which is four weights 32 apart in the output.

/// Q2_K: two bits, unsigned, `d*sc*q - m*mn` per sub-block.
template<typename dst_t>
inline __device__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q2_K * x = (const block_q2_K *) vx;

    // Two warps: the first takes the low half of the superblock, the second the high half.
    const int hi = threadIdx.x / 32;
    const int l  = threadIdx.x % 32;
    const int is = 8*hi + l/16;

    const uint8_t q = x->qs[32*hi + l];
    dst_t * y = yy + 128*hi;

    const half dall = __low2half(x->dm);
    const half dmin = __high2half(x->dm);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint8_t sc = x->scales[is + 2*s];
        const half scaled = __hmul(dall, __int2half_rn((sc & 0x0F) * ((q >> (2*s)) & 3)));
        const half offset = __hmul(dmin, __int2half_rn(sc >> 4));
        store_half(y[l + 32*s], __hsub(scaled, offset));
    }
}

/// Q3_K: the same two bits plus a third from the superblock's mask, against a 6-bit scale.
///
/// The mask bit is set for a weight that is NOT offset, which is why the four comes off when
/// the bit is clear.
template<typename dst_t>
inline __device__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q3_K * x = (const block_q3_K *) vx;

    // Four lanes share a sub-block and take four weights each.
    const auto r = threadIdx.x / 4;
    const int tid = r / 2;
    const int is0 = r % 2;
    const int l0 = 16*is0 + 4*(threadIdx.x % 4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    const uint8_t mask_bit = 1 << (4*n + j);
    const int is = 8*n + 2*j + is0;
    const int shift = 2*j;

    const half dl = __hmul(x->d, __int2half_rn(q3_K_scale(x->scales, is)));

    dst_t * y = yy + 128*n + 32*j;
    const uint8_t * q = x->qs + 32*n;
    const uint8_t * hm = x->hmask;

    for (int l = l0; l < l0 + 4; ++l) {
        const int w = (int8_t) ((q[l] >> shift) & 3) - ((hm[l] & mask_bit) ? 0 : 4);
        store_half(y[l], __hmul(dl, __int2half_rn(w)));
    }
}

// ------------------------------------------------------------
// Q4_K and Q5_K, dequantised.
//
// Both hold 256 weights in eight sub-blocks with a six-bit scale and minimum apiece, packed so
// the last four borrow their top two bits from the first four's - which is `get_scale_min_k4`
// above. A lane takes two of those sub-blocks, 32 apart in the output, and puts each weight on
// its footing as `d*q - m`. What differs is the weight: Q4_K's is a nibble, Q5_K's adds a
// fifth bit from a plane covering the whole superblock, and how many weights a lane owns
// follows from how wide the block is.

/// Q4_K: a nibble. The byte's two nibbles are the two sub-blocks the lane writes, `s` telling
/// them apart.
struct dequant_plane_q4_K {
    using block = block_q4_K;
    static constexpr int per_lane = 4, lanes_per_quarter = 8;
    static __device__ __forceinline__ int code(const block * x, int il, int ir, int l, int s) {
        return (x->qs[32*il + per_lane*ir + l] >> (4*s)) & 0x0F;
    }
};

/// Q5_K: the same nibble plus a fifth bit, whose position in the superblock-wide plane is the
/// sub-block the nibble belongs to.
struct dequant_plane_q5_K {
    using block = block_q5_K;
    static constexpr int per_lane = 2, lanes_per_quarter = 16;
    static __device__ __forceinline__ int code(const block * x, int il, int ir, int l, int s) {
        return ((x->qs[32*il + per_lane*ir + l] >> (4*s)) & 0x0F)
             + ((x->qh[per_lane*ir + l] & (1 << (2*il + s))) ? 16 : 0);
    }
};

template<typename plane, typename dst_t>
inline __device__ void dequantize_block_nibble_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    // One superblock per call, so the block index drops out.
    const typename plane::block * x = (const typename plane::block *) vx;

    const int il = threadIdx.x / plane::lanes_per_quarter;   // which quarter of the superblock
    const int ir = threadIdx.x % plane::lanes_per_quarter;   // which slice of that quarter

    dst_t * y = yy + 64*il + plane::per_lane*ir;

    const half dall = __low2half(x->dm);
    const half dmin = __high2half(x->dm);

    uint8_t sm[4];   // {sc0, sc1, m0, m1} of the lane's two sub-blocks
    unpack_quarter_scales_q45_K(x->scales, 2*il, sm);

#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const half d = __hmul(dall, __int2half_rn(sm[s]));
        const half m = __hmul(dmin, __int2half_rn(sm[2 + s]));
#pragma unroll
        for (int l = 0; l < plane::per_lane; ++l) {
            store_half(y[l + 32*s],
                __hsub(__hmul(d, __int2half_rn(plane::code(x, il, ir, l, s))), m));
        }
    }
}

// ------------------------------------------------------------
// The two 32-value formats, dequantised the same way the superblocks above are: one warp per
// block, one lane per weight, the scale broadcast from lane zero. They belong here with the
// rest of the formats rather than inside whichever kernel happens to want them.

/// The scale the whole block shares: lane zero reads the two bytes it occupies and hands it to
/// the rest of the warp.
static __device__ __forceinline__ float block_scale_broadcast(const uint8_t * b) {
    const half d = (threadIdx.x == 0) ? *(const half *) b : (half) 0.0f;
    return __half2float(__shfl_sync(0xFFFFFFFF, d, 0));
}

/// Q8_0: a signed byte and one scale for the block.
template<typename dst_t>
inline __device__ void dequantize_block_q8_0(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const auto lane = threadIdx.x;
    const uint8_t * b = (const uint8_t *) vx;
    const float d = block_scale_broadcast(b);

    if (lane < QK8_0) {
        yy[lane] = dst_t((float) ((const int8_t *) (b + 2))[lane] * d);
    }
}

/// Q5_0: a nibble, a fifth bit from a plane covering the whole block, and an offset of sixteen
/// that makes the five bits signed. The two halves of a byte are sixteen weights apart, and so
/// are the bits of the plane that complete them.
template<typename dst_t>
inline __device__ void dequantize_block_q5_0(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const auto lane = threadIdx.x;
    const uint8_t * b = (const uint8_t *) vx;
    const float d = block_scale_broadcast(b);

    // The block is 22 bytes, so the fifth-bit plane is not four-byte aligned.
    uint32_t qh;
    memcpy(&qh, b + 2, 4);
    const uint8_t * qs = b + 6;

    if (lane < QK5_0) {
        const int j = lane & 15;
        const int q = lane < 16 ? ((qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10))
                                : ((qs[j] >> 4)   | ((qh >> (j + 12)) & 0x10));
        yy[lane] = dst_t((float) (q - 16) * d);
    }
}

template<typename dst_t>
inline __device__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_block_nibble_K<dequant_plane_q4_K, dst_t>(vx, yy);
}

template<typename dst_t>
inline __device__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    dequantize_block_nibble_K<dequant_plane_q5_K, dst_t>(vx, yy);
}

/// Q6_K: six bits a weight, split four and two across two planes.
///
/// The superblock is 256 weights in two halves of 128, and a lane owns four of them 32 apart,
/// so 64 lanes cover it. The four go together in one order: the low nibble of the lane's first
/// byte, the low nibble of the byte 32 further on, then the high nibbles of those same two  - 
/// and the two-bit fields of one `qh` byte follow that order, two bits at a time. Sixteen
/// consecutive weights share a signed scale, which is why the lane's four take every second
/// one, and a code is read centred at thirty-two.
template<typename dst_t>
inline __device__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K & block = *(const block_q6_K *) vx;
    const int half_of = threadIdx.x / 32; // which 128-weight half
    const int lane = threadIdx.x % 32;    // which of that half's 32 lanes

    const uint8_t * ql = block.ql + 64 * half_of + lane;
    const uint8_t qh = block.qh[32 * half_of + lane];
    const int8_t * sc = block.scales + 8 * half_of + lane / 16;
    const half d = block.d;
    dst_t * y = yy + 128 * half_of + lane;

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t byte = ql[32 * (k & 1)];
        const int low = (k < 2) ? (byte & 0x0F) : (byte >> 4);
        const int code = low | (((qh >> (2 * k)) & 3) << 4);
        store_half(y[32 * k], __hmul(d, __int2half_rn(sc[2 * k] * (code - 32))));
    }
}