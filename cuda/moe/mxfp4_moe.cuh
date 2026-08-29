// MXFP4 (OCP micro-scaling FP4, E2M1 with a per-32-block E8M0 shared exponent)
// decode primitives for the MoE-GEMM kernels. gpt-oss stores its experts in
// MXFP4; keeping them native (rather than requantizing to Q8_0) halves expert
// VRAM and read bandwidth at decode. The block layout, lookup table, and
// dot-product mirror the general quantized-matmul path so numerics match.
//
// Depends on gguf.cuh for: block_q8_1, dot4_i8, QK8_1.
#ifndef LOKEN_MXFP4_MOE_CUH
#define LOKEN_MXFP4_MOE_CUH

#include <cstdint>
#include <cstring>

#define QK_MXFP4 32
#define QR_MXFP4 2
#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))   // = 4
#define VDR_MXFP4_Q8_1_MMVQ 2

// 17-byte block: 1-byte E8M0 shared exponent + 16 bytes of packed FP4 nibbles.
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;
static_assert(sizeof(block_mxfp4) == 17, "wrong mxfp4 block size");

// E2M1 fp4 nibble -> int8 lookup. The 8 magnitudes {0,.5,1,1.5,2,3,4,6} are
// stored pre-doubled to stay integral ({0,1,2,3,4,6,8,12}); bit 3 of the nibble
// is the sign, and the negated magnitudes fill the table's upper half so a
// nibble is a direct index. The 0.5f in the per-block scale undoes the doubling.
static const __device__ int8_t kvalues_mxfp4[16] =
    {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/// The block's shared exponent, as a float.
///
/// E8M0 is a float with no sign and no mantissa: the byte IS the exponent, biased by 127, so
/// the value is two to the power of `x - 127`. Dropping the byte into a float's exponent field
/// says exactly that - and the one code it cannot say that way is `x = 0`, whose value 2^-127
/// is below the smallest normal float and has to be written as the subnormal that equals it.
/// That is not a sentinel; it is the same number spelled the only way it can be.
static __device__ __forceinline__ float e8m0_to_float(uint8_t x) {
    const uint32_t bits = x == 0 ? 0x00400000u : (uint32_t) x << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/// Four bytes at `i32`, from a pointer with no alignment to promise.
///
/// This format's nibbles start one byte into the block, after the shared exponent, so they
/// land on odd addresses and even a two-byte read would be undefined. Four byte reads are
/// defined, and the compiler folds them back into whatever the hardware allows.
static __device__ __forceinline__ int four_bytes_bytewise(const void * x, const int & i32) {
    const uint8_t * at = (const uint8_t *) x + sizeof(int) * i32;
    int packed = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        packed |= (int) at[b] << (8 * b);
    }
    return packed;
}

/// The eight nibbles packed in `q4`, looked up in a 16-entry byte table.
///
/// Read as four ints, the table lets a byte permute gather four entries at once. Bits 0-2 of a
/// nibble index one of the eight magnitudes and bit 3 is its sign, so each 16-bit half of `q4`
/// is gathered twice - once from each half of the table - and the sign bits then choose between
/// the two gathers, byte by byte. The eight results come back split even/odd, which is how they
/// pair with the two halves of the activation.
static __device__ __forceinline__ int2 mxfp4_get_int_from_table_16(const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    // The identity byte selection, with each nibble's sign bit folded in as its table half.
    const uint32_t sign_select = (0x32103210 | ((q4 & 0x88888888) >> 1));

    const uint32_t low  = __byte_perm(__byte_perm(table32[0], table32[1], q4),
                                      __byte_perm(table32[2], table32[3], q4),
                                      sign_select);
    const uint32_t high = __byte_perm(__byte_perm(table32[0], table32[1], q4 >> 16),
                                      __byte_perm(table32[2], table32[3], q4 >> 16),
                                      sign_select >> 16);

    return make_int2(__byte_perm(low, high, 0x6420),
                     __byte_perm(low, high, 0x7531));
}

// The dot product against a q8_1 activation, in the three-argument form every format here uses:
// `vbq` already points at this thread's weight block - the kernel folded the block index into
// the pointer - and `iqs` is the int offset of this thread's codes inside it. `four_bytes` comes
// from gguf.cuh, included before this header.
static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq;

    // This thread's share of the block is VDR_MXFP4_Q8_1_MMVQ groups of four packed bytes, each
    // group eight nibbles. The table returns a group split even/odd, and the odd codes are the
    // block's second half - QI_MXFP4 ints further along the activation.
    int dot = 0;
#pragma unroll
    for (int group = 0; group < VDR_MXFP4_Q8_1_MMVQ; ++group) {
        const int2 codes = mxfp4_get_int_from_table_16(
            four_bytes_bytewise(bq4->qs, iqs + group), kvalues_mxfp4);
        dot = dot4_i8(codes.x, four_bytes(bq8_1->qs, iqs + group), dot);
        dot = dot4_i8(codes.y, four_bytes(bq8_1->qs, iqs + group + QI_MXFP4), dot);
    }

    // The table's magnitudes are doubled to stay integral, so the block's scale carries the
    // halving that undoes it; the activation's own scale is the low half of its `ds` pair.
    const float d = e8m0_to_float(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * dot;
}

#endif // LOKEN_MXFP4_MOE_CUH
