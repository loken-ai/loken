// Quantising the activation into the tile the matmul reads.
//
// The tiled matmul consumes the activation as `block_q8_1_mmq`: 128 values as int8, and the
// scales needed to put them back on their original footing. What "needed" means depends on
// the WEIGHT format, which is why one block layout has three arrangements of its metadata:
//
//   D4    four scales, one per 32 values. The weight is symmetric - q4_0, q5_0, q8_0 - so the
//         dot product is scale.scale.Σqq and nothing else is required.
//   DS4   four (scale, sum) pairs. The weight states a minimum as well as a scale - q4_1,
//         q5_1, q4_K, q5_K - and that minimum multiplies the activation's own sum, so the sum
//         has to be carried alongside.
//   D2S6  two scales over 64 values each, and six sums over 16 each. Q2_K's minima are finer
//         than a 32-value group, so its sums are too, and the scales are correspondingly
//         coarser to keep the block the same size.
//
// One thread owns four consecutive values, which is one 32-bit load and one 32-bit store. A
// scale therefore belongs to eight or sixteen threads and a sum to four or eight, and both are
// reduced by butterfly exchange within the warp rather than through shared memory.

#include "mmq_common.cuh"
#include "mmq_gguf.cuh"

// Four values per thread, so one block covers 512 activation values.
#define MMQ_QUANTIZE_THREADS 128

static_assert(MATRIX_ROW_PADDING % (4 * MMQ_QUANTIZE_THREADS) == 0,
              "A row must end on a block boundary or the last block reads past it.");

/// Reduce `v` across the `count/4` threads that share one group, leaving every one of them
/// holding the result.
///
/// The stride starts at half the group's thread count - `count/8` - because the group spans
/// `count/4` lanes. Butterfly rather than a shift-down reduction: every lane needs the answer,
/// since any of them may be the one that writes the scale.
template <int count, typename op>
static __device__ __forceinline__ float reduce_over_group(float v, op combine) {
#pragma unroll
    for (int stride = count / 8; stride > 0; stride >>= 1) {
        v = combine(v, __shfl_xor_sync(0xFFFFFFFF, v, stride, WARP_SIZE));
    }
    return v;
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(const float *__restrict__ x,
                                         const int32_t *__restrict__ ids, void *__restrict__ vy,
                                         const int64_t ne00, const int64_t s01, const int64_t s02,
                                         const int64_t s03, const int64_t ne0, const int ne1,
                                         const int ne2) {
    constexpr bool coarse_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6;
    constexpr int vals_per_scale = coarse_scale ? 64 : 32;
    constexpr int vals_per_sum = coarse_scale ? 16 : 32;

    // Where this thread's four values sit in the row.
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) {
        return;
    }

    const int64_t row = blockIdx.x;
    const int64_t channel = blockIdx.z % ne2;
    const int64_t sample = blockIdx.z / ne2;
    // `ids` reorders rows for the expert-batched form; without it a row is itself.
    const int64_t src_row = ids ? ids[row] : row;

    // The tile is laid out block-major within a channel: all rows of block 0, then block 1.
    const int64_t blocks_before_channel =
        blockIdx.z * ((int64_t) gridDim.x * gridDim.y * blockDim.x / QK8_1);
    const int64_t ib = blocks_before_channel + (i0 / (4 * QK8_1)) * ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4 * QK8_1);  // this thread's offset inside the block

    // Past the end of the real row is padding the matmul still reads, so it must be a
    // deterministic zero rather than whatever the allocation held.
    const float4 *x4 = (const float4 *) x;
    const float4 xi = i0 < ne00
                          ? x4[(sample * s03 + channel * s02 + src_row * s01 + i0) / 4]
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float amax = reduce_over_group<vals_per_scale>(
        fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w))),
        [](float a, float b) { return fmaxf(a, b); });

    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = reduce_over_group<vals_per_sum>(xi.x + xi.y + xi.z + xi.w,
                                              [](float a, float b) { return a + b; });
    }

    // A group of pure zeros has no scale to speak of. Deriving one anyway gives 127/0, and
    // every quant in the group becomes 0.inf - a NaN that only survives because the float to
    // int8 conversion happens to flush it. Say zero instead.
    const float d = amax / 127.0f;
    const float to_int8 = amax > 0.0f ? 127.0f / amax : 0.0f;

    // One 32-bit store rather than four byte stores: the row is walked once and this is the
    // whole of the bandwidth.
    char4 q;
    q.x = roundf(xi.x * to_int8);
    q.y = roundf(xi.y * to_int8);
    q.z = roundf(xi.z * to_int8);
    q.w = roundf(xi.w * to_int8);
    block_q8_1_mmq *y = (block_q8_1_mmq *) vy;
    ((char4 *) y[ib].qs)[iqs / 4] = q;

    // The metadata is written by the first thread of each group, and only by it.
    if (coarse_scale) {
        if (iqs % vals_per_sum == 0 && iqs < 6 * vals_per_sum) {
            y[ib].d2s6[2 + iqs / vals_per_sum] = sum;
        }
        if (iqs % vals_per_scale == 0) {
            y[ib].d2s6[iqs / vals_per_scale] = d;
        }
        return;
    }

    if (iqs % vals_per_scale != 0) {
        return;
    }
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs / vals_per_scale] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs / vals_per_scale] = d;
    }
}

/// The C entry point for one metadata arrangement. The three differ in that alone, so the
/// grid - one block per row, enough of them to cover the padded row length - is stated once.
///
/// `type_x` is accepted because the Rust declaration passes it; the arrangement is already
/// fixed by which of these three it called.
#define MMQ_QUANTIZE_ENTRY_POINT(layout, suffix)                                              \
    extern "C" void launch_mmq_quantize_q8_1_##suffix(                                        \
        const void *x_f32, const int32_t *ids, void *vy, int type_x, int64_t ne00,            \
        int64_t s01, int64_t s02, int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2,         \
        int64_t ne3, void *stream) {                                                          \
        (void) type_x;                                                                        \
        constexpr int per_block = 4 * MMQ_QUANTIZE_THREADS;                                   \
        const dim3 grid(ne1, (ne0 + per_block - 1) / per_block, ne2 *ne3);                    \
        const dim3 threads(MMQ_QUANTIZE_THREADS, 1, 1);                                       \
        quantize_mmq_q8_1<layout><<<grid, threads, 0, (cudaStream_t) stream>>>(               \
            (const float *) x_f32, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);              \
    }

MMQ_QUANTIZE_ENTRY_POINT(MMQ_Q8_1_DS_LAYOUT_D4, D4)
MMQ_QUANTIZE_ENTRY_POINT(MMQ_Q8_1_DS_LAYOUT_DS4, DS4)
MMQ_QUANTIZE_ENTRY_POINT(MMQ_Q8_1_DS_LAYOUT_D2S6, D2S6)
