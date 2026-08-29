// Turning stored blocks back into floats, and the small things the kernels around them share.
//
// This unit is compiled at runtime by NVRTC from a concatenated string, so it has no include
// path of its own: `<stdint.h>` and the `<math.h>` macros arrive through the shim the loader
// prepends.
//
// The block layouts are NOT here. `gguf.cuh` is concatenated ahead of this file and states
// them once - the structs, their element counts (`QK*`, `QR*`, `QI*`, `QK_K`, `K_SCALE_SIZE`),
// the size assertions over them, `get_scale_min_k4`, and the warp reductions. Everything below
// reads that statement; a format's geometry is never repeated here.
#include "cuda_fp16.h"
#include "cuda_bf16.h"

/// What a 32-value block multiplies its stored values by, and what it adds.
///
/// A symmetric block states only the scale, and its values are centred on the middle of the
/// range they can hold, so the offset is that midpoint scaled. An asymmetric one states both.
struct block_affine {
    float scale, offset;
};

static __device__ __forceinline__ block_affine affine_of(const block_q4_0 * b) {
    const float d = __half2float(b->d);
    return {d, -8.0f * d};
}
static __device__ __forceinline__ block_affine affine_of(const block_q4_1 * b) {
    const float2 dm = __half22float2(b->dm);
    return {dm.x, dm.y};
}
static __device__ __forceinline__ block_affine affine_of(const block_q5_0 * b) {
    const float d = __half2float(b->d);
    return {d, -16.0f * d};
}
static __device__ __forceinline__ block_affine affine_of(const block_q5_1 * b) {
    const float2 dm = __half22float2(b->dm);
    return {dm.x, dm.y};
}
static __device__ __forceinline__ block_affine affine_of(const block_q8_0 * b) {
    return {__half2float(b->d), 0.0f};
}

/// Whole 32-value blocks whose values carry a fifth bit in a word of their own.
///
/// The same eight-blocks-per-CUDA-block shape as the four-bit case; the fifth bit of byte `j`'s
/// low nibble sits at bit `j` of that word and its high nibble's twelve further on, sixteen
/// being how far apart the two halves of a block are.
template <typename block, typename dst_t>
static __device__ void dequantize_block_fifth_bit(const void * __restrict__ vx,
                                                  dst_t * __restrict__ yy, int nb32) {
    const int64_t i = blockIdx.x;
    const int quarter = threadIdx.x / 8;
    const int which   = threadIdx.x % 8;
    const int64_t ib  = 8*i + which;
    if (ib >= nb32) {
        return;
    }

    const block * x = (const block *) vx + ib;
    const block_affine a = affine_of(x);

    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));

    const uint8_t * q = x->qs + 4*quarter;
    dst_t * y = yy + 256*i + 32*which + 4*quarter;

    for (int l = 0; l < 4; ++l) {
        const int j = 4*quarter + l;
        const int hi_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const int hi_1 = ((qh >> (j + 12))     ) & 0x10;
        y[l +  0] = a.scale * ((q[l] & 0xF) | hi_0) + a.offset;
        y[l + 16] = a.scale * ((q[l] >>  4) | hi_1) + a.offset;
    }
}

/// Whole 32-value blocks of nibbles, eight blocks to a CUDA block.
///
/// Thirty-two threads: the low three bits of the lane pick which of the eight blocks it takes
/// and the high two which quarter of that block, so a warp's writes cover 256 consecutive
/// outputs. The two nibbles of a byte are sixteen values apart in the block, not adjacent.
template <typename block, typename dst_t>
static __device__ void dequantize_block_nibbles(const void * __restrict__ vx,
                                                dst_t * __restrict__ yy, int nb32) {
    const int64_t i = blockIdx.x;
    const int quarter = threadIdx.x / 8;
    const int which   = threadIdx.x % 8;
    const int64_t ib  = 8*i + which;
    if (ib >= nb32) {
        return;
    }

    const block * x = (const block *) vx + ib;
    const block_affine a = affine_of(x);
    const uint8_t * q = x->qs + 4*quarter;
    dst_t * y = yy + 256*i + 32*which + 4*quarter;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = a.scale * (q[l] & 0xF) + a.offset;
        y[l+16] = a.scale * (q[l] >>  4) + a.offset;
    }
}

//================================== k-quants

/// The same eight blocks per CUDA block, for a format that stores whole bytes: a lane takes a
/// quarter of a block, which is eight consecutive values rather than four split pairs.
template<typename dst_t>
static __device__ void dequantize_block_q8_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i = blockIdx.x;
    const int quarter = threadIdx.x / 8;
    const int which   = threadIdx.x % 8;
    const int ib = 8*i + which;
    if (ib >= nb32) {
        return;
    }

    const block_q8_0 * x = (const block_q8_0 *) vx + ib;
    const block_affine a = affine_of(x);
    const int8_t * q = x->qs + 8*quarter;
    dst_t * y = yy + 256*i + 32*which + 8*quarter;

    for (int l = 0; l < 8; ++l) {
        y[l] = a.scale * q[l] + a.offset;
    }
}

/// q8_K has nothing packed: 256 signed codes and the one float they are all read against.
///
/// One superblock per call, the caller having already resolved which, and 32 threads to cover
/// it: each takes eight codes in a row, so no thread has to work out where another one is. The
/// `bsums` a block also carries are a dot product's shortcut, not part of a weight's value,
/// and are ignored here.
template<typename dst_t>
static __device__ void dequantize_block_q8_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q8_K & block = *(const block_q8_K *) vx;
    const int first = 8 * threadIdx.x;

    const int8_t * q = block.qs + first;
    dst_t * y = yy + first;
    for (int l = 0; l < 8; ++l) {
        y[l] = q[l] * block.d;
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q5_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block_fifth_bit<block_q5_0>(vx, yy, nb32);
}

template<typename dst_t>
static __device__ void dequantize_block_q5_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block_fifth_bit<block_q5_1>(vx, yy, nb32);
}

template<typename dst_t>
static __device__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block_nibbles<block_q4_0>(vx, yy, nb32);
}

template<typename dst_t>
static __device__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block_nibbles<block_q4_1>(vx, yy, nb32);
}

// A superblock dequantiser takes ONE superblock and says so - it does not read `blockIdx`,
// because the other caller (the MoE tiled matmul) hands it a block it has already located.
// A launch covers a whole tensor with one CUDA block per superblock, so the entry point is the
// place where that block index becomes an offset, on both sides at once. Without it every
// block dequantised the first superblock and the rest of the tensor was never written.
#define SUPERBLOCK_ENTRY(FMT, TAG, DST)                                        \
    extern "C" __global__ void dequantize_block_##FMT##_##TAG(                 \
        const void * __restrict__ vx, DST * __restrict__ y                     \
    ) {                                                                        \
        const size_t sb = blockIdx.x;                                          \
        dequantize_block_##FMT((const block_##FMT *) vx + sb, y + sb * QK_K);  \
    }

// The 32-value formats bound themselves instead. One CUDA block covers eight of them, so the
// last one launched overruns the tensor and every thread weighs its own block against `nb32`
// before writing; locating the block is part of that, so nothing is resolved on the way in.
#define BLOCK32_ENTRY(FMT, TAG, DST)                                           \
    extern "C" __global__ void dequantize_block_##FMT##_##TAG(                 \
        const void * __restrict__ vx, DST * __restrict__ y, const int nb32     \
    ) {                                                                        \
        dequantize_block_##FMT(vx, y, nb32);                                   \
    }

// Both destinations, from one generator. The suffix and the type travel together because a
// kernel is resolved by name at runtime, so its name has to spell what it writes.
#define SUPERBLOCK_PAIR(FMT)                 \
    SUPERBLOCK_ENTRY(FMT, f32, float)        \
    SUPERBLOCK_ENTRY(FMT, f16, half)

#define BLOCK32_PAIR(FMT)                    \
    BLOCK32_ENTRY(FMT, f32, float)           \
    BLOCK32_ENTRY(FMT, f16, half)

// Each family's formats, stated once. A generator is pasted over the list rather than repeated
// under it, so a format added here reaches every entry point that has to exist for it.
#define SUPERBLOCK_FORMATS(EMIT) EMIT(q2_K) EMIT(q3_K) EMIT(q4_K) EMIT(q5_K) EMIT(q6_K) EMIT(q8_K)
#define BLOCK32_FORMATS(EMIT)    EMIT(q4_0) EMIT(q4_1) EMIT(q5_0) EMIT(q5_1) EMIT(q8_0)

SUPERBLOCK_FORMATS(SUPERBLOCK_PAIR)
BLOCK32_FORMATS(BLOCK32_PAIR)


// GPU-native Q8_0 quantizer, writing the format's own block: 32 signed codes and the one half
// scale they are read against, the scale being the block's largest magnitude over 127. Used
// for persistent KV-cache storage, where the cheap Q8_0 x Q8_1 gemv kernels can then run
// against the cache without a CPU roundtrip on every write.
extern "C" __global__ void quantize_q8_0(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y*blockIdx.y + threadIdx.y;

    const int i_padded = iy*kx_padded + ix;

    block_q8_0 * y = (block_q8_0 *) vy;

    const int ib = i_padded / QK8_0; // block index
    const int iqs = i_padded % QK8_0; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);

    amax = warp_max(amax);

    const float d = amax / 127;
    const float id = amax == 0.0f ? 0.0f : 1.0f / d;
    const int8_t q = roundf(xi * id);

    y[ib].qs[iqs] = q;

    if (iqs == 0) {
        y[ib].d = d;
    }
}

// Device-position variant of quantize_q8_0. Reads the destination token
// slot index from `slot_dev[0]` and writes the quantized blocks at
// `dst[slot_dev[0] * num_blocks_per_token * sizeof(block_q8_0)]`.
//
// Required for CUDA graph capture of the Q8 KV append step. The
// non-dev_pos kernel writes at a host-supplied dst pointer that's
// baked into the captured kernel parameters at capture time - every
// replay overwrites the same slot, so the captured graph is unusable
// past the first replay.
//
// Designed for the single-token-per-launch decode pattern (kx is one
// token's K or V row, ky=1 implied). For the engine's KV append
// the row width n_kv_heads x head_dim must be a multiple of
// MATRIX_ROW_PADDING (512) - same guard as quantize_q8_0_f32_into_offset
// in cuda.rs. Caller falls back to the staged path for non-aligned
// shapes.
extern "C" __global__ void quantize_q8_0_dev_slot(
    const float * __restrict__ x,
    void * __restrict__ vy,
    const int32_t * __restrict__ slot_dev,
    const int kx,
    const int kx_padded,
    const int num_blocks_per_token
) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) {
        return;
    }
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;

    // Resolve the per-token destination offset at REPLAY time.
    const int slot = slot_dev[0];
    block_q8_0 * y = (block_q8_0 *) vy + (size_t)slot * (size_t)num_blocks_per_token;

    const int ib  = i_padded / QK8_0;
    const int iqs = i_padded % QK8_0;

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);

    amax = warp_max(amax);

    const float d  = amax / 127;
    const float id = amax == 0.0f ? 0.0f : 1.0f / d;
    const int8_t q = roundf(xi * id);

    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].d = d;
    }
}