#pragma once

#include "mmq_common.cuh"
#include "mmq_vecdotq.cuh"
#include "mmq_mma.cuh"

#include <climits>
#include <cstdint>

using namespace loken_mma;

#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K 256

// ---------------------------------------------------------------------------------------
// The activation, as the tiled matmul reads it.
//
// A scale is what a quantised value is multiplied by to recover the value it came from. A sum
// is the total of a span of values BEFORE quantisation: a weight format that states a minimum
// spends one sum per span in place of one product per value, which is the only reason to carry
// sums at all. A format that states no minimum leaves them unread.

/// Which of those two the activation carries, and over what span.
enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,   ///< a scale per 32 values: d0 d1 d2 d3
    MMQ_Q8_1_DS_LAYOUT_DS4,  ///< a scale and a sum per 32 values: d0 s0 d1 s1 d2 s2 d3 s3
    MMQ_Q8_1_DS_LAYOUT_D2S6, ///< a scale per 64 values, then a sum per 16: d0 d1 s0 s1 .. s5
};

/// Four 32-value groups of the activation, quantised to 8 bit, sharing one footing.
///
/// The activation is regrouped into runs of 128 values and transposed a run at a time, so that
/// a run reaches shared memory as one contiguous copy. Consecutive runs are held apart by a
/// fixed pad, which is what keeps them off the same shared-memory banks; the footing lives in
/// that pad and so costs nothing further. The three layouts above are three readings of that
/// one pad, and the weight format decides which reading its dot product can spend - hence the
/// union rather than three members.
struct block_q8_1_mmq {
    union {
        float d4[4];
        half2 ds4[4];
        half  d2s6[8];
    };
    int8_t qs[4*QK8_1];
};

// What the three readings and the pad have to agree on, stated from the members themselves.

// One pad, so no reading of it may outgrow the others.
static_assert(sizeof(block_q8_1_mmq::d4) == sizeof(block_q8_1_mmq::ds4) &&
              sizeof(block_q8_1_mmq::d4) == sizeof(block_q8_1_mmq::d2s6),
              "the three footings of an activation run must be the same width");
// A run is its values and its footing and nothing beyond: no member may be padded apart.
static_assert(sizeof(block_q8_1_mmq) == sizeof(block_q8_1_mmq::qs) + sizeof(block_q8_1_mmq::d4),
              "an activation run must weigh its values plus its footing");
// And it weighs exactly what the separate blocks it stands in for weigh, or the run is not the
// drop-in the quantiser and the tile loader both take it for.
static_assert(sizeof(block_q8_1_mmq) == (sizeof(block_q8_1_mmq::qs)/QK8_1) * sizeof(block_q8_1),
              "an activation run must weigh the blocks it replaces");

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

// ---------------------------------------------------------------------------------------
// The card, and the tile geometry that follows from it.
//
// How wide a batch a tile serves, how tall a column of weights it holds, how many output rows a
// warp takes, how many warps a block spends, which of the two tile layouts sits in shared memory
// and which multiply reads it: six questions with one subject. Each of them used to open its own
// chain over the same arms, so the card was described six times over. It is described once here,
// as facts, and every question after this one is an expression over those facts.
//
// With a tile-multiplying instruction the widest batch there is, because the tile is read once
// for all of it. Without one the dp4a path pays for its own activation quantisation and stops
// gaining past `MMQ_DP4A_MAX_BATCH_SIZE`. A card before Volta halves both tile dimensions: it
// has half the shared memory per multiprocessor to hold the tile in.

struct mmq_card {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    static constexpr bool tile_multiply     = true;   ///< multiplies whole tiles at once
    static constexpr bool wavefront_operand = true;   ///< and its operand is a whole wave wide
    static constexpr bool wide_dp4a_batch   = true;
    static constexpr bool tall_tile         = true;
#elif defined(TURING_MMA_AVAILABLE)
    static constexpr bool tile_multiply     = true;
    static constexpr bool wavefront_operand = false;
    static constexpr bool wide_dp4a_batch   = true;
    static constexpr bool tall_tile         = true;
#elif defined(GGML_USE_HIP) && defined(RDNA1)
    static constexpr bool tile_multiply     = false;
    static constexpr bool wavefront_operand = false;
    static constexpr bool wide_dp4a_batch   = false;
    static constexpr bool tall_tile         = false;
#elif defined(GGML_USE_HIP)
    static constexpr bool tile_multiply     = false;
    static constexpr bool wavefront_operand = false;
    static constexpr bool wide_dp4a_batch   = false;
    static constexpr bool tall_tile         = true;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    static constexpr bool tile_multiply     = false;
    static constexpr bool wavefront_operand = false;
    static constexpr bool wide_dp4a_batch   = true;
    static constexpr bool tall_tile         = true;
#else
    static constexpr bool tile_multiply     = false;
    static constexpr bool wavefront_operand = false;
    static constexpr bool wide_dp4a_batch   = false;
    static constexpr bool tall_tile         = false;
#endif
};

static constexpr __device__ int get_mmq_x_max_device() {
    return mmq_card::tile_multiply ? 128
                                   : (mmq_card::wide_dp4a_batch ? MMQ_DP4A_MAX_BATCH_SIZE : 64);
}

static constexpr __device__ int get_mmq_y_device() {
    return mmq_card::tall_tile ? 128 : 64;
}

// Every supported format iterates k in MMQ_ITER_K steps. Only the FP4 tile needs a different
// stride, and no FP4 format is instantiated here.
static constexpr __device__ int get_iter_k() {
    return MMQ_ITER_K;
}

/// How many output rows one warp takes.
///
/// A wider matrix instruction covers more rows at once, and above a certain tile width a warp
/// takes two of them rather than one.
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    if (mmq_card::wavefront_operand) {
        return mmq_x >= 128 ? 32 : 16;
    }
    if (mmq_card::tile_multiply) {
        return mmq_x >= 48 ? 16 : 8;
    }
    return 8;
}

/// How many warps a block spends on one tile.
///
/// The block is 256 threads wide, so the warp count is not chosen: it is what 256 threads come
/// to once the warp size is known. The matrix cores are the exception. Their operand is a whole
/// wavefront wide, so a block there is counted in wavefronts rather than in threads, and eight
/// is what keeps one tile fed without spilling the accumulators.
///
/// This one is asked from the host as well, which is told the compute capability rather than
/// compiled for it - hence the rule apart from the two ways of establishing its one fact.
static constexpr __host__ __device__ int mmq_nwarps(const bool matrix_cores, const int warp_size) {
    return matrix_cores ? 8 : 256/warp_size;
}

#if defined(GGML_USE_HIP)
static int mmq_get_nwarps_host(const int cc, const int warp_size) {
    return mmq_nwarps(amd_mfma_available(cc), warp_size);
}
#else
static int mmq_get_nwarps_host(const int /*cc*/, const int warp_size) {
    return mmq_nwarps(false, warp_size);
}
#endif // (GGML_USE_HIP)

static constexpr __device__ int mmq_get_nwarps_device() {
    return mmq_nwarps(mmq_card::wavefront_operand, ggml_cuda_get_physical_warp_size());
}

// The unit a tile row is counted in: 32 ints of quantised weights, scales not included. A tile
// row is one of these for the activation and, for most weight formats, two. It is deliberately
// not the warp size - the two are equal on one vendor and not on the other, and a tile whose
// shape moved with the warp would be a different tile per vendor.
#define MMQ_TILE_NE_K 32

// One activation tile row: its ints of weights, then the footing of each 32-value block.
#define MMQ_TILE_Y_K  (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)

#define MMQ_MMA_TILE_X_K_Q8_0  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_Q8_1  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_Q2_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K                           + 4)
#define MMQ_MMA_TILE_X_K_Q3_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2                         + 4)
#define MMQ_MMA_TILE_X_K_Q6_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7)

/// One invariant over the whole table of row widths above: a tensor-core row must end four
/// ints short of the eight `ldmatrix` reads at a time, so that row `i+1` starts half a read
/// past where row `i` did and consecutive rows never share a shared-memory bank. A width that
/// is a clean multiple of eight lines every row up on the same banks, which is the one thing
/// the padding exists to prevent.
template <typename... widths>
static constexpr bool mmq_rows_are_staggered(const widths... tile_k) {
    return ((tile_k % 8 == 4) && ...);
}

static_assert(mmq_rows_are_staggered(MMQ_MMA_TILE_X_K_Q8_0, MMQ_MMA_TILE_X_K_Q8_1,
                                     MMQ_MMA_TILE_X_K_Q2_K, MMQ_MMA_TILE_X_K_Q3_K,
                                     MMQ_MMA_TILE_X_K_Q6_K),
              "a tensor-core tile row must be staggered against the shared-memory banks");

// ---------------------------------------------------------------------------------------
// What each format asks of a tile.
//
// Three questions come up per format: how the activation states its scales, how wide one row
// of a tensor-core tile is, and how a dp4a tile divides its three planes. They are one row of
// one table, so a format is added - or dropped - in one place rather than in three switches
// over the same ten cases.
//
// A tile row is padded so that consecutive rows land in different shared-memory banks: one
// spare int per row for dp4a, four for the tensor cores, whose `ldmatrix` reads eight at a
// time. The dp4a plane sizes therefore depend on how many rows a tile holds, which is why
// they are stated as entries-per-row and a divisor rather than as constants.

/// One plane of a dp4a tile: `per_row` entries for each row, plus one spare per `pad` rows.
struct mmq_plane {
    int per_row;
    int pad;
};

struct mmq_format {
    mmq_q8_1_ds_layout ds_layout;   // the arrangement the activation's metadata takes
    int mma_tile_k;                 // ints per row of a tensor-core tile
    mmq_plane qs, dm, sc;           // the three planes of a dp4a tile
};

static constexpr __host__ __device__ mmq_format mmq_format_of(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return {MMQ_Q8_1_DS_LAYOUT_DS4,  MMQ_MMA_TILE_X_K_Q8_0,
                                     {MMQ_TILE_NE_K,   1}, {MMQ_TILE_NE_K/QI4_0,   QI4_0},     {0, 0}};
        case GGML_TYPE_Q4_1: return {MMQ_Q8_1_DS_LAYOUT_DS4,  MMQ_MMA_TILE_X_K_Q8_1,
                                     {MMQ_TILE_NE_K,   1}, {MMQ_TILE_NE_K/QI4_1,   QI4_1},     {0, 0}};
        case GGML_TYPE_Q5_0: return {MMQ_Q8_1_DS_LAYOUT_D4,   MMQ_MMA_TILE_X_K_Q8_0,
                                     {MMQ_TILE_NE_K*2, 1}, {MMQ_TILE_NE_K*2/QI8_0, QI8_0/2},   {0, 0}};
        case GGML_TYPE_Q5_1: return {MMQ_Q8_1_DS_LAYOUT_DS4,  MMQ_MMA_TILE_X_K_Q8_1,
                                     {MMQ_TILE_NE_K*2, 1}, {MMQ_TILE_NE_K*2/QI8_1, QI8_1/2},   {0, 0}};
        case GGML_TYPE_Q8_0: return {MMQ_Q8_1_DS_LAYOUT_D4,   MMQ_MMA_TILE_X_K_Q8_0,
                                     {MMQ_TILE_NE_K*2, 1}, {MMQ_TILE_NE_K*2/QI8_0, QI8_0/2},   {0, 0}};
        case GGML_TYPE_Q2_K: return {MMQ_Q8_1_DS_LAYOUT_D2S6, MMQ_MMA_TILE_X_K_Q2_K,
                                     {MMQ_TILE_NE_K*2, 1}, {MMQ_TILE_NE_K,         1},         {0, 0}};
        case GGML_TYPE_Q3_K: return {MMQ_Q8_1_DS_LAYOUT_D4,   MMQ_MMA_TILE_X_K_Q3_K,
                                     {MMQ_TILE_NE_K*2, 1}, {0,                     1},         {MMQ_TILE_NE_K/8, 8}};
        case GGML_TYPE_Q4_K: return {MMQ_Q8_1_DS_LAYOUT_DS4,  MMQ_MMA_TILE_X_K_Q8_1,
                                     {MMQ_TILE_NE_K,   1}, {MMQ_TILE_NE_K/QI4_K,   0},         {MMQ_TILE_NE_K/8, 8}};
        case GGML_TYPE_Q5_K: return {MMQ_Q8_1_DS_LAYOUT_DS4,  MMQ_MMA_TILE_X_K_Q8_1,
                                     {MMQ_TILE_NE_K*2, 1}, {MMQ_TILE_NE_K/QI5_K,   QI5_K},     {MMQ_TILE_NE_K/8, 8}};
        case GGML_TYPE_Q6_K: return {MMQ_Q8_1_DS_LAYOUT_D4,   MMQ_MMA_TILE_X_K_Q6_K,
                                     {MMQ_TILE_NE_K*2, 1}, {MMQ_TILE_NE_K/QI6_K,   QI6_K},     {MMQ_TILE_NE_K/8, 8}};
        default:             return {MMQ_Q8_1_DS_LAYOUT_D4,   0, {0, 0}, {0, 0}, {0, 0}};
    }
}

/// A plane the format does not have is `{0, 0}`, and so is nothing. Q3_K's scale plane is the
/// other end of the same rule: one entry per row and no per-row entries at all.
static constexpr __host__ __device__ int mmq_plane_size(const mmq_plane p, const int mmq_y) {
    return mmq_y*p.per_row + (p.pad ? mmq_y/p.pad : 0);
}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int mmq_y) {
    const mmq_format f = mmq_format_of(type);
    if (f.mma_tile_k == 0) {
        return tile_x_sizes{0, 0, 0};
    }
    return tile_x_sizes{mmq_plane_size(f.qs, mmq_y), mmq_plane_size(f.dm, mmq_y),
                        mmq_plane_size(f.sc, mmq_y)};
}

static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
    return mmq_format_of(type).mma_tile_k;
}

/// Where a tile keeps its planes, and where row `i` of each one starts.
///
/// The two multiplies want different layouts and every loader below said so twice. On the
/// tensor-core path `ldmatrix` reads a fixed shape, so every plane shares one padded row
/// stride and the scales always begin two weight-planes in. On the dp4a path each plane is
/// packed as tightly as the format allows, with one spare int every `pad` rows so consecutive
/// rows land in different shared-memory banks - which is the same statement the format table
/// already makes about the plane's size.
template <ggml_type type, int mmq_y>
struct mmq_tile {
    static constexpr mmq_format fmt = mmq_format_of(type);
    static constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);

    static constexpr int scales_at    = mmq_card::tile_multiply ? 2*MMQ_TILE_NE_K : txs.qs;
    static constexpr int subscales_at = mmq_card::tile_multiply ? scales_at + fmt.dm.per_row
                                                                : txs.qs + txs.dm;

    /// Where row `i` of one plane begins. A tensor-core tile gives every plane the same padded
    /// stride, that being the one shape `ldmatrix` reads; a dp4a tile packs each plane as
    /// tightly as the format allows and spaces its rows by the plane's own spare int.
    static __device__ __forceinline__ int plane_row(const mmq_plane p, const int i) {
        if constexpr (mmq_card::tile_multiply) {
            return i * fmt.mma_tile_k;
        } else {
            return i*p.per_row + (p.pad ? i/p.pad : 0);
        }
    }

    static __device__ __forceinline__ int qs_row(const int i) { return plane_row(fmt.qs, i); }
    static __device__ __forceinline__ int df_row(const int i) { return plane_row(fmt.dm, i); }
    static __device__ __forceinline__ int sc_row(const int i) { return plane_row(fmt.sc, i); }

    static __device__ __forceinline__ int * qs(int * tile) { return tile; }
    static __device__ __forceinline__ float * df(int * tile) {
        return (float *) (tile + scales_at);
    }
    static __device__ __forceinline__ half2 * dm(int * tile) {
        return (half2 *) (tile + scales_at);
    }
    static __device__ __forceinline__ int * sc(int * tile) { return tile + subscales_at; }
};

/// Which tile row a lane fills, when `threads_per_row` lanes share one.
///
/// The wrap is for the passes that cover more rows than the tile holds - where the loop does
/// not overshoot it costs nothing, `mmq_y` being a power of two. `need_check` then pins a row
/// past the end of the matrix to the last real one, so the thread re-reads a row that exists
/// rather than reading past the tensor.
template <int threads_per_row, int mmq_y>
static __device__ __forceinline__ int mmq_tile_row(const int i0, const int i_max,
                                                   const bool need_check) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int rows_at_once = warp_size / threads_per_row;
    const int i = (i0 + (rows_at_once == 1 ? threadIdx.y
                                           : threadIdx.y*rows_at_once + threadIdx.x/threads_per_row))
                  % mmq_y;
    return need_check ? min(i, i_max) : i;
}

// ---------------------------------------------------------------------------------------
// Emptying the accumulator into the result.
//
// This sits here, beside the tile geometry, because it belongs to the geometry and not to any
// format: whichever weights went in, the outputs come out the same way. Both spellings say the
// same sentence - each lane writes the outputs it holds, skipping any that fall past the edge
// of the matrix - and they stay two because the accumulators are not indexed alike. A dp4a lane
// owns a (row, column) of the tile and walks it in two nested steps; a matrix-core lane owns
// whatever entries the instruction's fragment hands it, and only `get_i`/`get_j` can say which
// output an entry is. Writing one loop over both would mean choosing an order, and a store
// order that is not the accumulator's is a scatter.

template<int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(
        const float * __restrict__ sum, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[j*stride + i] = sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size];
        }
    }
}

template<int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
        const float * __restrict__ sum, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int tileC_IJ = mmq_get_granularity_device(0);
    typedef tile<tileC_IJ, tileC_IJ, int, DATA_LAYOUT_J_MAJOR> tile_C;
    constexpr int rows_per_warp = granularity;
#else
    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;
#endif // defined(AMD_MFMA_AVAILABLE)
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    static_assert(nwarps*tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");
#else
    UNUSED(nwarps);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[j*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
}

/// The one the multiply this card has filled.
template<int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back(
        const float * __restrict__ sum, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
    if constexpr (mmq_card::tile_multiply) {
        mmq_write_back_mma<mmq_x, mmq_y, need_check>(sum, dst, stride, i_max, j_max);
    } else {
        mmq_write_back_dp4a<mmq_x, mmq_y, need_check>(sum, dst, stride, i_max, j_max);
    }
}

// ---------------------------------------------------------------------------------------
// Loading a weight tile.
//
// A tile row holds one weight row's contribution to a stretch of k: ints of four signed int8
// weights, plus the scale of each 32-weight block they came from. Four of the block formats
// pack two weights per byte and therefore fill two positions of the row from one read; what
// distinguishes them is only how those bytes become signed quads, whether the block carries a
// scale alone or a (scale, min) pair, and where the tile wants the result.
//
// The last of those depends on the card, not the format: a tensor-core matmul reads a padded
// row-major tile, the dp4a matmul reads a narrower one. That fork is written once here rather
// than eight times across four loaders.

/// One pass over the rows of a tile.
///
/// Every plane of every format is filled the same way: `threads_per_row` lanes share a weight
/// row, each writes its share of it, and the pass repeats until the tile is covered. What a
/// lane writes is the format's business. Which row it lands on, and what it does with a row
/// past the end of the matrix, is the same question for all of them and is asked here.
template <typename block, int threads_per_row, int mmq_y, bool need_check, typename fill_t>
static __device__ __forceinline__ void mmq_fill_rows(
        const char * __restrict__ x, const int kbx0, const int stride, const int i_max,
        fill_t fill) {
    constexpr int nwarps       = mmq_get_nwarps_device();
    constexpr int rows_at_once = ggml_cuda_get_physical_warp_size() / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += rows_at_once*nwarps) {
        const int i = mmq_tile_row<threads_per_row, mmq_y>(i0, i_max, need_check);
        fill(i, (const block *) x + kbx0 + i*stride);
    }
}

/// Whether the tile this card reads holds the two halves of a packed byte apart.
///
/// The tensor-core tile always does. A dp4a tile only does where the halves cannot be
/// recovered later: a format whose dot product splits nibbles itself keeps the byte pair
/// whole, which is one int written per lane instead of two.
template <typename plane>
static constexpr __device__ bool mmq_tile_splits_bytes() {
    return mmq_card::tile_multiply || plane::dp4a_holds_halves;
}

// Q4_0: 32 nibbles and one scale. The nibble is an offset from 8, so the weight is signed by
// subtracting 8 from all four bytes at once.
struct plane_q4_0 {
    using block = block_q4_0;
    using scale_t = float;
    static constexpr ggml_type type = GGML_TYPE_Q4_0;
    static constexpr int qi = QI4_0, qr = QR4_0;
    static constexpr bool dp4a_holds_halves = false;
    static __device__ __forceinline__ void unpack(const block * b, const int j, int & lo,
                                                  int & hi, int & raw) {
        raw = four_bytes_unaligned(b->qs, j);
        lo = __vsubss4((raw >> 0) & 0x0F0F0F0F, 0x08080808);
        hi = __vsubss4((raw >> 4) & 0x0F0F0F0F, 0x08080808);
    }
    static __device__ __forceinline__ float scale(const block * b) { return b->d; }
};

// Q4_1: the same nibbles, but the block states a scale AND a minimum, so the nibble is an
// unsigned magnitude and nothing is subtracted here - the minimum multiplies the activation's
// own sum, once per block, where the dot product applies it.
struct plane_q4_1 {
    using block = block_q4_1;
    using scale_t = half2;
    static constexpr ggml_type type = GGML_TYPE_Q4_1;
    static constexpr int qi = QI4_1, qr = QR4_1;
    static constexpr bool dp4a_holds_halves = false;
    static __device__ __forceinline__ void unpack(const block * b, const int j, int & lo,
                                                  int & hi, int & raw) {
        raw = four_bytes(b->qs, j);
        lo = (raw >> 0) & 0x0F0F0F0F;
        hi = (raw >> 4) & 0x0F0F0F0F;
    }
    static __device__ __forceinline__ half2 scale(const block * b) { return b->dm; }
};

// The fifth bit of a Q5 weight lives in a separate 32-bit plane, one bit per weight. Weight j
// of the low half is bit j and weight j of the high half is bit j+16; each has to arrive at
// bit 4 of its own byte, which is a different distance for each of the four bytes.
static __device__ __forceinline__ int q5_fifth_bits(const int qh) {
    return ((qh <<  4) & 0x00000010) | ((qh << 11) & 0x00001000)
         | ((qh << 18) & 0x00100000) | ((qh << 25) & 0x10000000);
}

// Q5_0: nibble plus fifth bit, offset from 16. Five bits do not fit a nibble, so BOTH tiles
// hold the weights already split - the dp4a one only lays the row out more tightly.
struct plane_q5_0 {
    using block = block_q5_0;
    using scale_t = float;
    static constexpr ggml_type type = GGML_TYPE_Q5_0;
    static constexpr int qi = QI5_0, qr = QR5_0;
    static constexpr bool dp4a_holds_halves = true;
    static __device__ __forceinline__ void unpack(const block * b, const int j, int & lo,
                                                  int & hi, int & raw) {
        raw = four_bytes_unaligned(b->qs, j);
        const int qh = four_bytes_unaligned(b->qh, 0) >> (4 * j);
        lo = __vsubss4(((raw >> 0) & 0x0F0F0F0F) | q5_fifth_bits(qh),       0x10101010);
        hi = __vsubss4(((raw >> 4) & 0x0F0F0F0F) | q5_fifth_bits(qh >> 16), 0x10101010);
    }
    static __device__ __forceinline__ float scale(const block * b) { return b->d; }
};

// Q5_1: the same five bits against a (scale, min) pair, so again unsigned.
struct plane_q5_1 {
    using block = block_q5_1;
    using scale_t = half2;
    static constexpr ggml_type type = GGML_TYPE_Q5_1;
    static constexpr int qi = QI5_1, qr = QR5_1;
    static constexpr bool dp4a_holds_halves = true;
    static __device__ __forceinline__ void unpack(const block * b, const int j, int & lo,
                                                  int & hi, int & raw) {
        raw = four_bytes(b->qs, j);
        const int qh = four_bytes(b->qh, 0) >> (4 * j);
        lo = ((raw >> 0) & 0x0F0F0F0F) | q5_fifth_bits(qh);
        hi = ((raw >> 4) & 0x0F0F0F0F) | q5_fifth_bits(qh >> 16);
    }
    static __device__ __forceinline__ half2 scale(const block * b) { return b->dm; }
};

template <typename plane, int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_paired(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    using block = typename plane::block;
    using scale_t = typename plane::scale_t;
    using tile = mmq_tile<plane::type, mmq_y>;
    constexpr int qi = plane::qi;

    int * x_qs = tile::qs(x_tile);
    scale_t * x_sc = (scale_t *) (x_tile + tile::scales_at);

    // One int of the tile covers four k-positions and one read covers two of them per byte,
    // so a row of MMQ_ITER_K positions is served by this many lanes.
    constexpr int lanes = MMQ_ITER_K / (4 * plane::qr);
    const int lane = threadIdx.x % lanes;
    const int kb = lane / qi;   // which block along the row
    const int ki = lane % qi;   // which int inside that block

    mmq_fill_rows<block, lanes, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block * row_blocks) {
            const block * bxi = row_blocks + kb;
            int lo, hi, raw;
            plane::unpack(bxi, ki, lo, hi, raw);

            int * row = x_qs + tile::qs_row(i);
            if constexpr (mmq_tile_splits_bytes<plane>()) {
                // The two halves of a byte are half a block apart in k, which is `qi` ints.
                row[kb*(2*qi) + ki + 0 ] = lo;
                row[kb*(2*qi) + ki + qi] = hi;
            } else {
                // Whole, one lane one int: `raw` is what the dot product will split.
                row[lane] = raw;
            }
        });

    // There is one scale per block rather than one per int, so the warp spreads differently.
    constexpr int blocks_per_row = MMQ_TILE_NE_K / qi;
    const int kbd = threadIdx.x % blocks_per_row;

    mmq_fill_rows<block, blocks_per_row, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block * row_blocks) {
            x_sc[tile::df_row(i) + kbd] = plane::scale(row_blocks + kbd);
        });
}

// Q8_0: the weights are already signed bytes, so the tile takes them whole and the only work
// is placing them. A lane owns one 32-byte block and its scale; two blocks of the row are
// filled per pass because the tile is twice a block wide.
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max,
    const int stride) {
    using tile = mmq_tile<GGML_TYPE_Q8_0, mmq_y>;

    int * x_qs = tile::qs(x_tile);
    float * x_df = tile::df(x_tile);

    // A tile row is two spans of MMQ_TILE_NE_K ints, and a lane owns one int of each.
    constexpr int ints_per_span = MMQ_TILE_NE_K;
    const int txi = threadIdx.x % ints_per_span;
    const int kqsx = txi % QI8_0;   // which int inside the block the lane's span puts it in

    mmq_fill_rows<block_q8_0, ints_per_span, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block_q8_0 * row_blocks) {
            const block_q8_0 * bxi = row_blocks + txi/QI8_0;
            x_qs[tile::qs_row(i) + 0             + txi] = four_bytes_unaligned(bxi[0].qs, kqsx);
            x_qs[tile::qs_row(i) + MMQ_TILE_NE_K + txi] =
                four_bytes_unaligned(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
        });

    // The scales: one per block, so fewer lanes are needed and each covers more rows.
    constexpr int blocks_per_tile_row = 2*MMQ_TILE_NE_K / QI8_0;
    const int kbxd = threadIdx.x % blocks_per_tile_row;

    mmq_fill_rows<block_q8_0, blocks_per_tile_row, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block_q8_0 * row_blocks) {
            x_df[tile::df_row(i) + kbxd] = row_blocks[kbxd].d;
        });
}

// ---------------------------------------------------------------------------------------
// The tile pass the dp4a path runs.
//
// Without tensor cores a lane owns one row of the weight tile and walks the column tile,
// stepping k by however much the format's dot product consumes at once. That walk is the same
// for every format; what differs is one step of the dot product, which is what each states
// below. `sum` is indexed by where the lane's (row, column) falls in the tile, not by k: the
// steps of a row accumulate into the same place.

template <int mmq_x, int mmq_y, int k_step, int k_first = 0, int k_last = MMQ_TILE_NE_K,
          typename step_t>
static __device__ __forceinline__ void vec_dot_over_tile_dp4a(float * __restrict__ sum, step_t step) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    for (int k01 = k_first; k01 < k_last; k01 += k_step) {
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*(mmq_y/warp_size) + i0/warp_size] += step(i, j, k01);
            }
        }
    }
}

// The tiled dot over a nibble format.
//
// Q4_0 and Q4_1 pack their quants identically and stride their tile identically; the only
// thing that differs is what the weight's scale carries, which is the carrier's type. Both
// read a Q8_1 activation whose halves sit `QI` apart from the nibbles they pair with, and
// that offset is the format's, not the kernel's.
template <int mmq_x, int mmq_y, ggml_type type, typename Scale, int qi, int vdr>
static __device__ __forceinline__ void vec_dot_nibbles_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int   * x_qs = (const int   *) x;
    const Scale * x_d  = (const Scale *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    // Two nibbles to a byte, so a nibble format's quant ratio is two.
    constexpr int qr = 2;
    vec_dot_over_tile_dp4a<mmq_x, mmq_y, qr*vdr>(sum,
        [&] (const int i, const int j, const int k01) {
            const int k0 = k00 + k01;
            const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);
            int u[2*vdr];
#pragma unroll
            for (int l = 0; l < vdr; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + kyqs +  l];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + kyqs + (l + qi)];
            }
            return vec_dot_nibbles_q8_1_impl<vdr>
                (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/qr], u,
                 x_d[i*(MMQ_TILE_NE_K/qi) + i/qi + k0/(qr*qi)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
        });
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    vec_dot_over_tile_dp4a<mmq_x, mmq_y, VDR_Q8_0_Q8_1_MMQ>(sum,
        [&] (const int i, const int j, const int k01) {
            const int k0 = k00 + k01;
            return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
                 x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
        });
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    vec_dot_over_tile_dp4a<mmq_x, mmq_y, VDR_Q8_0_Q8_1_MMQ>(sum,
        [&] (const int i, const int j, const int k01) {
            const int k0 = k00 + k01;
            return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
        });
}

// ---------------------------------------------------------------------------------------
// The walk a warp makes over one tile row of k.
//
// Every tensor-core dot product below traces the same path: take the stretch of the activation
// tile this warp answers for, fill A from the weights and B from the activation, multiply them
// into an int32 accumulator, then scatter the accumulator back into `sum` - one entry at a
// time, because `get_i`/`get_j` are what say which output an entry belongs to.
//
// Two things fork that path, and only two.
//
// The first is the width of the accumulator. A wide one is filled by a single instruction and
// spent immediately, so A is read again for every group of output columns; a narrow one takes
// twice as many minitiles to cover the same rows, which makes A worth reading once for the
// whole k row and holding in registers across the column loop. That is a different nesting of
// the same two loops, not a different body, so it is two walks rather than one.
//
// The second is how often the footing changes. A format whose footing holds for a whole
// 32-value activation block spends one operand per block; a format that re-states it halfway
// spends two, of half the reach. That is `sub` below, and it does not change the nesting.
//
// What each format actually does with an accumulator entry - which scales it meets, whether a
// minimum is owed against the activation's sum - is none of the walk's business and arrives as
// callbacks. `footing` is whatever a column group has to work out once before the entries of
// that group can be settled.

/// The stretch of the activation tile this warp's accumulator answers for. Warps sharing a
/// group of output columns differ only in which minitile of that group they hold.
template <int ntx, int tile_j>
static __device__ __forceinline__ const int * mma_lane_columns(const int * y) {
    return y + (threadIdx.y % ntx) * (tile_j*MMQ_TILE_Y_K);
}

/// A weight operand of all ones.
///
/// Multiplied against the activation it puts, in each accumulator entry, the SUM of the
/// activation over the span the instruction reaches - which is what a format owes its minimum
/// against for the span whose sum the activation's metadata does not state. Both instruction
/// widths below want one, and neither wants a different one.
template <typename tile_A>
static __device__ __forceinline__ tile_A mma_all_ones() {
    tile_A ones;
#pragma unroll
    for (int l = 0; l < tile_A::ne; ++l) {
        ones.x[l] = 0x01010101;
    }
    return ones;
}

/// The walk of a wide accumulator: A is read per k step and spent on every column group.
///
/// `fill_a` puts the weights of minitile `mt` at absolute k into an operand and `fill_b` does
/// the same for the activation, which is where instructions of differing operand shape part
/// company. `open` is handed the loaded B and works out the column group's footing. `place`
/// settles one entry: its value `c`, the weight row `i` it came from, the slot in `sum` it
/// belongs to, its index `e` in the accumulator, and the k this step covers.
template <int mmq_x, int ntx, int k_step, typename tile_A, typename tile_B, typename tile_C,
          typename footing, typename fill_a_t, typename fill_b_t, typename open_t, typename place_t>
static __device__ __forceinline__ void mma_walk_spending_a(
        const int i0, const int k00,
        fill_a_t fill_a, fill_b_t fill_b, open_t open, place_t place) {
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += k_step) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int mt = 0; mt < ntx; ++mt) {
            fill_a(A, mt, k0);
        }

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            fill_b(B, j0, k01);

            footing f;
            open(f, B, j0, k01);

#pragma unroll
            for (int mt = 0; mt < ntx; ++mt) {
                tile_C C;
                mma(C, A[mt], B);

#pragma unroll
                for (int e = 0; e < tile_C::ne; ++e) {
                    place(C.x[e], f, i0 + mt*tile_C::I + tile_C::get_i(e),
                          (j0/tile_C::J + mt)*tile_C::ne + e, e, k0, k01);
                }
            }
        }
    }
}

/// The walk of a narrow accumulator: A is already in registers, one operand per `sub`-th of
/// each activation block, and only the activation is read inside the loops.
///
/// `place` is handed the whole `sub`-tuple of accumulators at once, because a format that
/// splits a block that way owes one product per part and the sum of those parts is a single
/// term. `close` runs once per column group, for whatever a format still owes that no
/// accumulator carries.
template <int mmq_x, int ntx, int sub, typename tile_A, typename tile_B, typename tile_C,
          typename footing, typename fill_b_t, typename open_t, typename place_t, typename close_t>
static __device__ __forceinline__ void mma_walk_holding_a(
        const tile_A (&A)[ntx][MMQ_TILE_NE_K/tile_B::J],
        fill_b_t fill_b, open_t open, place_t place, close_t close) {
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B B[sub];
#pragma unroll
            for (int s = 0; s < sub; ++s) {
                fill_b(B[s], j0, k01 + s*tile_B::J);
            }

            footing f;
            open(f, B, j0, k01);

#pragma unroll
            for (int mt = 0; mt < ntx; ++mt) {
                tile_C C[sub];
#pragma unroll
                for (int s = 0; s < sub; ++s) {
                    mma(C[s], A[mt][k01/tile_B::J + s], B[s]);
                }

#pragma unroll
                for (int e = 0; e < tile_C::ne; ++e) {
                    place(C, f, mt, (j0/tile_C::J + mt)*tile_C::ne + e, e, k01);
                }
            }
        }

        close(j0);
    }
}

// ---------------------------------------------------------------------------------------
// The tiled dot product over 32-value blocks.
//
// These formats state their footing once per 32-value block, so one operand reaches a whole
// block and the walk above runs with `sub` of one. What distinguishes them is only what "put
// the accumulator back on its footing" means, and there are two answers:
//
//   a weight that states a scale alone   ->  d_A * d_B * C
//   a weight that also states a minimum  ->  d_A * d_B * C  +  m_A * s_B
//
// where s_B is the activation block's own SUM, which its quantiser carried alongside the
// scale for exactly this. The minimum does not multiply C: it is the same offset for every
// weight of the block, so it meets the activation's sum once rather than once per product.
//
// That is the policy below, and it is all these formats contribute: the tile shapes and the
// choice of walk follow from the accumulator the card offers, not from the format.

/// The weight states a scale and nothing else.
struct mmq_scale_only {
    static constexpr int tile_k = MMQ_MMA_TILE_X_K_Q8_0;
    using weight_scale = float;
    using activation_scale = float;

    static __device__ __forceinline__ float weight(const int * x_qs, const int idx) {
        return ((const float *) (x_qs + 2*MMQ_TILE_NE_K))[idx];
    }
    /// D4 stores four bare scales; the other arrangements store (scale, sum) pairs, and the
    /// sum goes unread here because this weight has no minimum to spend it on.
    template <mmq_q8_1_ds_layout ds_layout>
    static __device__ __forceinline__ float activation(const int * y, const int idx) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
            return ((const float *) y)[idx];
        }
        return __low2float(((const half2 *) y)[idx]);
    }
    static __device__ __forceinline__ float term(const int c, const float dA, const float dB) {
        return c * dA * dB;
    }
};

/// The weight states a scale and a minimum; the activation carries its sum to meet it.
struct mmq_scale_and_min {
    static constexpr int tile_k = MMQ_MMA_TILE_X_K_Q8_1;
    using weight_scale = float2;
    using activation_scale = float2;

    static __device__ __forceinline__ float2 weight(const int * x_qs, const int idx) {
        return __half22float2(((const half2 *) (x_qs + 2*MMQ_TILE_NE_K))[idx]);
    }
    template <mmq_q8_1_ds_layout>
    static __device__ __forceinline__ float2 activation(const int * y, const int idx) {
        return __half22float2(((const half2 *) y)[idx]);
    }
    static __device__ __forceinline__ float term(const int c, const float2 dmA, const float2 dsB) {
        return dmA.x * dsB.x * c + dmA.y * dsB.y;
    }
};

template <typename policy, int mmq_x, int mmq_y, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_blocked_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int tile_k = policy::tile_k;
    using WS = typename policy::weight_scale;
    using AS = typename policy::activation_scale;

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int rows_per_warp = mmq_get_granularity_device(mmq_x);
    constexpr int ntx = rows_per_warp/tile_C::I;   // x minitiles per warp

    y = mma_lane_columns<ntx, tile_C::J>(y);

    const int * x_qs = (const int *) x;
    const int * y_qs = (const int *) y + 4;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    /// This accumulator is column-major, so every entry of a column group shares the one
    /// column `get_j(0)` names, and with it one activation scale.
    struct footing { AS sB; };

    mma_walk_spending_a<mmq_x, ntx, QI8_0, tile_A, tile_B, tile_C, footing>(i0, k00,
        [&] (tile_A * A, const int mt, const int k0) {
            load_generic(A[mt], x_qs + (i0 + mt*tile_A::I)*tile_k + k0, tile_k);
        },
        [&] (tile_B & B, const int j0, const int k01) {
            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
        },
        [&] (footing & f, const tile_B &, const int j0, const int k01) {
            const int j = j0 + tile_C::get_j(0);
            f.sB = policy::template activation<ds_layout>(y, j*MMQ_TILE_Y_K + k01/QI8_1);
        },
        [&] (const int c, const footing & f, const int i, const int slot,
             const int, const int k0, const int) {
            const WS sA = policy::weight(x_qs, i*tile_k + k0/QI8_0);
            sum[slot] += policy::term(c, sA, f.sB);
        });
#else
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    // Two minitiles per granule here: this accumulator is half the width of AMD's, so a warp
    // covers the same rows with twice as many of them.
    constexpr int rows_per_warp = 2 * mmq_get_granularity_device(mmq_x);
    constexpr int ntx = rows_per_warp/tile_C::I;

    y = mma_lane_columns<ntx, tile_C::J>(y);

    const int * x_qs = (const int *) x;
    const int * y_qs = (const int *) y + 4;

    // A and its scales do not depend on the output column, so they are read once for the
    // whole k row and held in registers across the column loop below.
    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    WS sA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*tile_k + k00 + k01, tile_k);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                sA[n][l][k01/QI8_0] = policy::weight(x_qs, i*tile_k + (k00 + k01)/QI8_0);
            }
        }
    }

    /// This accumulator is row-major and half as wide, so a column group spans several
    /// columns and each entry meets the scale of the one it lands in.
    struct footing { AS sB[tile_C::ne/2]; };

    mma_walk_holding_a<mmq_x, ntx, 1, tile_A, tile_B, tile_C, footing>(A,
        [&] (tile_B & B, const int j0, const int k) {
            // Generic beats ldmatrix for B: the activation tile is already laid out the way
            // the instruction wants it, so there is no permutation to pay for.
            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k, MMQ_TILE_Y_K);
        },
        [&] (footing & f, const tile_B (&)[1], const int j0, const int k01) {
#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                f.sB[l] = policy::template activation<ds_layout>(y, j*MMQ_TILE_Y_K + k01/QI8_1);
            }
        },
        [&] (const tile_C (&C)[1], const footing & f, const int mt, const int slot,
             const int e, const int k01) {
            sum[slot] += policy::term(C[0].x[e], sA[mt][e/2][k01/QI8_0], f.sB[e%2]);
        },
        [&] (const int) {});
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// ---------------------------------------------------------------------------------------
// The formats whose footing changes every sixteen weights.
//
// Q3_K, Q6_K and Q2_K all re-state their footing four times as often as a 32-value block
// does, so the tile carries it per four ints rather than per eight and the loop walks k in
// fours. What that footing IS differs, and that is the whole of the difference:
//
//   Q3_K   the finished per-sub-block scale, already a float in the tile
//   Q6_K   a signed 6-bit sub-scale that still has to meet the superblock's own float
//   Q2_K   a (scale, minimum) pair, both already multiplied out against the superblock's
//
// A minimum is the same offset for every weight of its sub-block, so where the other two
// contribute `dA * C`, Q2_K contributes `dA * C - mA * S` - with S the activation's own SUM
// over that span, which is not a product but a sum of sixteen values. The activation's
// metadata carries three such sums per row of four spans, so three of the four are read and
// the fourth is obtained the only other way there is: a dot product against all ones.
//
// That is the policy. The loop below is written once for all three, twice over, because the
// instruction differs enough between vendors that the shape of the loop does too.

/// The tile already holds the finished per-sub-block scale.
struct mmq_subblock_float {
    static constexpr int tile_k = MMQ_MMA_TILE_X_K_Q3_K;
    static constexpr bool states_minimum = false;
    static __device__ __forceinline__ float weight(const int * x, const int i, const int k) {
        const float * x_df = (const float *) x + MMQ_TILE_NE_K*2;
        return x_df[i*tile_k + k/4];
    }
    static __device__ __forceinline__ float minimum(const int *, const int, const int) { return 0.0f; }
    /// One scale per 32-value activation block, which is two of this format's spans.
    static __device__ __forceinline__ float activation(const int * y, const int j, const int k01) {
        return ((const float *) y)[j*MMQ_TILE_Y_K + k01/QI8_1];
    }
    static __device__ __forceinline__ bool sum_is_stated(const int) { return false; }
    static __device__ __forceinline__ float2 stated_sum(const int *, const int, const int) {
        return make_float2(0.0f, 0.0f);
    }
};

/// A signed 6-bit sub-scale in its own plane, against one float for the whole row.
struct mmq_subblock_scaled {
    static constexpr int tile_k = MMQ_MMA_TILE_X_K_Q6_K;
    static constexpr bool states_minimum = false;
    static __device__ __forceinline__ float weight(const int * x, const int i, const int k) {
        const float * x_df = (const float *) x + MMQ_TILE_NE_K*2;
        const int * x_sc = (const int *) x_df + MMQ_TILE_NE_K/QI6_K;
        const int8_t * sc = (const int8_t *) (x_sc + i*tile_k + (k/16)*16/16);
        return sc[(k/4) % 4] * x_df[i*tile_k];
    }
    static __device__ __forceinline__ float minimum(const int *, const int, const int) { return 0.0f; }
    static __device__ __forceinline__ float activation(const int * y, const int j, const int k01) {
        return ((const float *) y)[j*MMQ_TILE_Y_K + k01/QI8_1];
    }
    static __device__ __forceinline__ bool sum_is_stated(const int) { return false; }
    static __device__ __forceinline__ float2 stated_sum(const int *, const int, const int) {
        return make_float2(0.0f, 0.0f);
    }
};

/// Q2_K: a (scale, minimum) pair per sub-block, each already multiplied out.
struct mmq_subblock_pair {
    static constexpr int tile_k = MMQ_MMA_TILE_X_K_Q2_K;
    static constexpr bool states_minimum = true;
    static __device__ __forceinline__ float2 pair(const int * x, const int i, const int k) {
        const half2 * x_dm = (const half2 *) ((const int *) x + MMQ_TILE_NE_K*2);
        return __half22float2(x_dm[i*tile_k + k/4]);
    }
    static __device__ __forceinline__ float weight(const int * x, const int i, const int k) {
        return pair(x, i, k).x;
    }
    static __device__ __forceinline__ float minimum(const int * x, const int i, const int k) {
        return pair(x, i, k).y;
    }
    /// This format's activation states one scale for each HALF of the tile row rather than one
    /// per 32-value block, so slot zero holds both and k picks between them.
    static __device__ __forceinline__ float activation(const int * y, const int j, const int k01) {
        const float2 d = __half22float2(((const half2 *) y)[j*MMQ_TILE_Y_K]);
        return k01 < MMQ_TILE_NE_K/2 ? d.x : d.y;
    }
    /// Three sums for four spans: the last quarter of the row has none to read.
    static __device__ __forceinline__ bool sum_is_stated(const int k01) {
        return k01 < MMQ_TILE_NE_K*3/4;
    }
    /// The two halves of the pair are the two sixteen-value spans of one activation block.
    static __device__ __forceinline__ float2 stated_sum(const int * y, const int j, const int k01) {
        return __half22float2(((const half2 *) y)[j*MMQ_TILE_Y_K + 1 + k01/QI8_1]);
    }
};

template <typename policy, int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_subblock_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int tile_k = policy::tile_k;
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
#if defined(AMD_MFMA_AVAILABLE)
    // MFMA's operand is 16x8 and has to be filled through a 64x2 view: the instruction reads a
    // whole wavefront's worth of rows, which is not the shape four ints of a span arrive in.
    // Feeding it twice the k it is stepped by is what the halved activation scale pays for.
    typedef tile<16, 8, int, input_layout> tile_A;
    typedef tile<16, 8, int, input_layout> tile_B;
    typedef tile<64, 2, int, input_layout> tile_load;
    constexpr float half_step = 0.5f;
#else
    // WMMA takes a 16x4 operand directly, which is the shape a four-int span already has.
    typedef tile<16, 4, int, input_layout> tile_A;
    typedef tile<16, 4, int, input_layout> tile_B;
    typedef tile_A tile_load;
    constexpr float half_step = 1.0f;
#endif
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int rows_per_warp = mmq_get_granularity_device(mmq_x);
    constexpr int ntx = rows_per_warp/tile_C::I;   // x minitiles per warp

    y = mma_lane_columns<ntx, tile_C::J>(y);

    const int * x_qs = (const int *) x;
    const int * y_qs = (const int *) y + 4;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    /// A column group's scale, and the sum the minimum is owed against - either read from the
    /// activation's metadata, or, for the span whose sum nothing states, accumulated against
    /// all ones so that every entry can take its own.
    struct footing { float dB; float sB; tile_C Cm; };

    mma_walk_spending_a<mmq_x, ntx, 4, tile_A, tile_B, tile_C, footing>(i0, k00,
        [&] (tile_A * A, const int mt, const int k0) {
            load_generic(((tile_load *) A)[mt], x_qs + (i0 + mt*tile_A::I)*tile_k + k0, tile_k);
        },
        [&] (tile_B & B, const int j0, const int k01) {
            load_generic(*((tile_load *) &B), y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
        },
        [&] (footing & f, const tile_B & B, const int j0, const int k01) {
            const int j = j0 + tile_C::get_j(0);
            f.dB = policy::activation(y, j, k01) * half_step;

            f.sB = 0.0f;
            if (policy::states_minimum) {
                if (policy::sum_is_stated(k01)) {
                    const float2 s = policy::stated_sum(y, j, k01);
                    f.sB = (k01/4) % 2 ? s.y : s.x;
                } else {
                    mma(f.Cm, mma_all_ones<tile_A>(), B);
                }
            }
        },
        [&] (const int c, const footing & f, const int i, const int slot,
             const int e, const int k0, const int k01) {
            float tmp = c * policy::weight(x, i, k0);
            if (policy::states_minimum) {
                const float mA = policy::minimum(x, i, k0);
                if (!policy::sum_is_stated(k01)) {
                    tmp -= f.Cm.x[e]*mA;
                }
                sum[slot] += tmp*f.dB;
                sum[slot] -= mA*f.sB;
            } else {
                sum[slot] += tmp*f.dB;
            }
        });
#elif defined(TURING_MMA_AVAILABLE)
    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    // Two minitiles per granule: this accumulator is half the width of AMD's.
    constexpr int rows_per_warp = 2 * mmq_get_granularity_device(mmq_x);
    constexpr int ntx = rows_per_warp/tile_C::I;

    y = mma_lane_columns<ntx, tile_C::J>(y);

    const int * x_qs = (const int *) x;
    const int * y_qs = (const int *) y + 4;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    // A and its footing do not depend on the output column: read once, held across the column
    // loop. The pair of int operands the 16x8 load fills is two four-int spans, hence eight.
    tile_A A[ntx][8];
    float sA[ntx][tile_C::ne/2][8];
    float mA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            load_ldmatrix(((tile_A_8 *) A[n])[k01/QI8_1],
                          x_qs + (i0 + n*tile_A::I)*tile_k + k00 + k01, tile_k);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                sA[n][l][k01/4] = policy::weight(x, i, k00 + k01);
                if (policy::states_minimum) {
                    mA[n][l][k01/4] = policy::minimum(x, i, k00 + k01);
                }
            }
        }
    }

    /// A column group's two scales - one per column the accumulator reaches - and the pair of
    /// all-ones accumulators that stand in for the sum of the span nothing states.
    struct footing { float dB[tile_C::ne/2]; tile_C Cm[2]; };

    mma_walk_holding_a<mmq_x, ntx, 2, tile_A, tile_B, tile_C, footing>(A,
        [&] (tile_B & B, const int j0, const int k) {
            // Generic beats ldmatrix for B: the activation tile is already laid out the way
            // the instruction wants it, so there is no permutation to pay for.
            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k, MMQ_TILE_Y_K);
        },
        [&] (footing & f, const tile_B (&B)[2], const int j0, const int k01) {
#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                f.dB[l] = policy::activation(y, j0 + tile_C::get_j(l), k01);
            }

            if (policy::states_minimum && !policy::sum_is_stated(k01)) {
                const tile_A ones = mma_all_ones<tile_A>();
                mma(f.Cm[0], ones, B[0]);
                mma(f.Cm[1], ones, B[1]);
            }
        },
        [&] (const tile_C (&C)[2], const footing & f, const int mt, const int slot,
             const int e, const int k01) {
            float tmp = C[0].x[e]*sA[mt][e/2][k01/4 + 0] + C[1].x[e]*sA[mt][e/2][k01/4 + 1];
            if (policy::states_minimum && !policy::sum_is_stated(k01)) {
                tmp -= f.Cm[0].x[e]*mA[mt][e/2][k01/4 + 0] + f.Cm[1].x[e]*mA[mt][e/2][k01/4 + 1];
            }
            sum[slot] += tmp*f.dB[e%2];
        },
        // The minimum against the sums the activation does state, in a pass of its own: those
        // spans have no accumulator to fold it into.
        [&] (const int j0) {
            if (policy::states_minimum) {
#pragma unroll
                for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                    if (!policy::sum_is_stated(k01)) {
                        continue;
                    }
                    float2 sB[tile_C::ne/2];
#pragma unroll
                    for (int l = 0; l < tile_C::ne/2; ++l) {
                        sB[l] = policy::stated_sum(y, j0 + tile_C::get_j(l), k01);
                    }

#pragma unroll
                    for (int mt = 0; mt < ntx; ++mt) {
#pragma unroll
                        for (int e = 0; e < tile_C::ne; ++e) {
                            const int slot = (j0/tile_C::J + mt)*tile_C::ne + e;
                            sum[slot] -= mA[mt][e/2][k01/4 + 0]*sB[e%2].x;
                            sum[slot] -= mA[mt][e/2][k01/4 + 1]*sB[e%2].y;
                        }
                    }
                }
            }
        });
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

/// The four sub-block footings packed into one word, written out one tile slot each.
///
/// A superblock states four sub-scales to a 32-bit word and one carrier for the whole block,
/// and a tensor-core tile wants them already met with that carrier, a slot per sub-block. What
/// one byte of the word becomes is the format's; that there are four of them is not.
template <typename footing_t, typename meet_t>
static __device__ __forceinline__ void mmq_spread_word(footing_t * row, const int word,
                                                       meet_t meet) {
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        row[int(sizeof(int))*word + l] = meet(l);
    }
}

// ---------------------------------------------------------------------------------------
// The superblock formats that hold two bits per weight.
//
// Q2_K and Q3_K both pack the low two bits of every weight into one plane, four weights to a
// byte, and both spread a superblock's sub-blocks eight ints apart along the tile row - a
// lane owns one byte-quad and writes it to `QR` positions. The difference is what those two
// bits mean: Q2_K's are an unsigned magnitude, scaled and offset by the (scale, min) pair its
// sub-block states; Q3_K borrows a third bit from a separate mask and is signed by an offset
// of four, against a scale alone.
//
// Their scales are stated differently enough that each keeps its own pass - Q2_K's fit in the
// same loop as the weights, Q3_K's are six bits split across two planes and shared by four
// lanes - so what is written once here is the weight pass.

/// Q2_K: two bits, unsigned, against a (scale, min) pair per sub-block.
struct plane_q2_K {
    using block = block_q2_K;
    static constexpr ggml_type type = GGML_TYPE_Q2_K;
    static constexpr int qr = QR2_K;
    static __device__ __forceinline__ int high_bits(const block *, const int) { return 0; }
    static __device__ __forceinline__ int weight(const int lo, const int, const int l) {
        return (lo >> (2 * l)) & 0x03030303;
    }
    /// One lane owns one sub-block, so it can multiply the 4-bit scale and minimum out
    /// against the superblock's pair where it stands.
    static __device__ __forceinline__ void store_scale(const block * bxi, const int kqsx,
                                                       half2 * row) {
        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        row[kqsx] = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 f = __half22float2(bxi->dm);
        row[kqsx] = make_half2(f.x * (sc_m & 0x0F), f.y * (sc_m >> 4));
#endif
    }
};

/// Q3_K: the same two bits plus a third from a whole-superblock mask, offset from four.
struct plane_q3_K {
    using block = block_q3_K;
    static constexpr ggml_type type = GGML_TYPE_Q3_K;
    static constexpr int qr = QR3_K;
    /// The mask covers the superblock, so this lane's four weights take the half of it their
    /// sub-block falls in.
    static __device__ __forceinline__ int high_bits(const block * b, const int kqsx) {
        return four_bytes_unaligned(b->hmask, kqsx % (QI3_K / 2)) >> (4 * (kqsx / (QI3_K / 2)));
    }
    static __device__ __forceinline__ int weight(const int lo, const int hi, const int l) {
        const int low_two = (lo >> (2 * l)) & 0x03030303;
        const int third = ((hi >> l) << 2) & 0x04040404;
        return __vsubss4(low_two | third, 0x04040404);
    }
    /// Six bits split across two planes, so gathered in a pass of its own below.
    static __device__ __forceinline__ void store_scale(const block *, const int, half2 *) {}
};

/// Fill the weight plane of a two-bit superblock tile.
template <typename plane, int mmq_y, bool need_check>
static __device__ __forceinline__ void load_weights_two_bit_K(const char * __restrict__ x,
                                                              int * __restrict__ x_tile,
                                                              const int kbx0, const int i_max,
                                                              const int stride) {
    using block = typename plane::block;
    using tile = mmq_tile<plane::type, mmq_y>;
    constexpr int lanes = MMQ_ITER_K / (4 * plane::qr);
    const int kqsx = threadIdx.x % lanes;

    int * x_qs = tile::qs(x_tile);
    half2 * x_dm = tile::dm(x_tile);

    mmq_fill_rows<block, lanes, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block * bxi) {
            const int lo = four_bytes_unaligned(bxi->qs, kqsx);
            const int hi = plane::high_bits(bxi, kqsx);

            int * row = x_qs + tile::qs_row(i);
#pragma unroll
            for (int l = 0; l < plane::qr; ++l) {
                // Sub-block `l` of this lane's quad sits eight ints further along the row; the
                // first term is which half of the superblock the lane is in.
                row[(kqsx / 8) * 32 + l * 8 + kqsx % 8] = plane::weight(lo, hi, l);
            }

            plane::store_scale(bxi, kqsx, x_dm + tile::df_row(i));
        });
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q2_K(const char * __restrict__ x,
                                                       int * __restrict__ x_tile, const int kbx0,
                                                       const int i_max, const int stride) {
    load_weights_two_bit_K<plane_q2_K, mmq_y, need_check>(x, x_tile, kbx0, i_max, stride);
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    using tile = mmq_tile<GGML_TYPE_Q2_K, mmq_y>;
    constexpr int nwarps = mmq_get_nwarps_device();

    const int   * x_qs = tile::qs(const_cast<int *>(x));
    const half2 * x_dm = tile::dm(const_cast<int *>(x));
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    // This format's activation states one scale for each HALF of the tile row rather than one
    // per block, so both are read up front and k picks between them.
    float2 y_df[mmq_x/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        y_df[j0/nwarps] = __half22float2(y_ds[(j0 + threadIdx.y)*MMQ_TILE_Y_K]);
    }

    // `ns` is how many of the activation's stated sums a step may read: two in the first half
    // of the row, one in the second, where the metadata runs out. The two halves are two
    // calls rather than one loop with a conditional because a conditional there stops the
    // compiler unrolling the k loop.
    const auto step = [&] (const int i, const int j, const int k01, const int ns_two) {
        const int k0 = k00 + k01;
        const float dB = k01 < MMQ_TILE_NE_K/2 ? y_df[(j - (int) threadIdx.y)/nwarps].x
                                               : y_df[(j - (int) threadIdx.y)/nwarps].y;
        return ns_two
            ? vec_dot_q2_K_q8_1_impl_mmq<2>(&x_qs[tile::qs_row(i) + k0],
                                            &y_qs[j*MMQ_TILE_Y_K + k01],
                                            &x_dm[tile::df_row(i) + k0/4], dB,
                                            &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)])
            : vec_dot_q2_K_q8_1_impl_mmq<1>(&x_qs[tile::qs_row(i) + k0],
                                            &y_qs[j*MMQ_TILE_Y_K + k01],
                                            &x_dm[tile::df_row(i) + k0/4], dB,
                                            &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
    };

    constexpr int k_step = QR2_K*VDR_Q2_K_Q8_1_MMQ;
    vec_dot_over_tile_dp4a<mmq_x, mmq_y, k_step, 0, MMQ_TILE_NE_K/2>(
        sum, [&] (const int i, const int j, const int k01) { return step(i, j, k01, 1); });
    vec_dot_over_tile_dp4a<mmq_x, mmq_y, k_step, MMQ_TILE_NE_K/2, MMQ_TILE_NE_K>(
        sum, [&] (const int i, const int j, const int k01) { return step(i, j, k01, 0); });
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q3_K(const char * __restrict__ x,
                                                       int * __restrict__ x_tile, const int kbx0,
                                                       const int i_max, const int stride) {
    load_weights_two_bit_K<plane_q3_K, mmq_y, need_check>(x, x_tile, kbx0, i_max, stride);

    using tile = mmq_tile<GGML_TYPE_Q3_K, mmq_y>;
    float * x_df = tile::df(x_tile);

    // Six bits of scale per sub-block, four low in one plane and two high in another, offset
    // from 32. One lane takes a word of four, so a weight row is shared by this many.
    constexpr int scale_words = MMQ_TILE_NE_K/8;
    const int ksc = threadIdx.x % scale_words;

    mmq_fill_rows<block_q3_K, scale_words, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block_q3_K * bxi) {
            const int ksc_low = ksc % (QI3_K/8);
            const int shift_low = 4 * (ksc / (QI3_K/8));
            const int sc_low = (four_bytes_unaligned(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

            const int ksc_high = QI3_K/8;
            const int shift_high = 2 * ksc;
            const int sc_high = ((four_bytes_unaligned(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

            const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

            if constexpr (mmq_card::tile_multiply) {
                // The tensor-core tile takes the finished scale, so the superblock's own float
                // is met with it here rather than left for the dot product.
                const int8_t * sc8 = (const int8_t *) &sc;
                const float d = bxi->d;
                mmq_spread_word(x_df + tile::df_row(i), ksc,
                                [&] (const int l) { return d*sc8[l]; });
            } else {
                tile::sc(x_tile)[tile::sc_row(i) + ksc] = sc;
            }
        });

    if constexpr (!mmq_card::tile_multiply) {
        // The dp4a tile keeps the two apart, so the superblock's float goes in on its own.
        mmq_fill_rows<block_q3_K, 1, mmq_y, need_check>(x, kbx0, stride, i_max,
            [&] (const int i, const block_q3_K * bxi) {
                x_df[tile::df_row(i)] = bxi->d;
            });
    }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    vec_dot_over_tile_dp4a<mmq_x, mmq_y, QR3_K*VDR_Q3_K_Q8_1_MMQ>(sum,
        [&] (const int i, const int j, const int k01) {
            const int k0 = k00 + k01;
            const int8_t * scales = ((const int8_t *) (x_sc + i*(MMQ_TILE_NE_K/8) + i/8)) + k0/4;
            return vec_dot_q3_K_q8_1_impl_mmq(
                &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
                x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
        });
}

static __device__ __forceinline__ int unpack_scale_word_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

// ---------------------------------------------------------------------------------------
// The 4- and 5-bit superblock weight planes.
//
// Q4_K and Q5_K share everything the tile cares about: 256 weights, a nibble each, eight
// sub-blocks with a (scale, min) pair apiece - which is why `load_scales_q45_K` already
// serves both. What differs is the weight plane. Q4_K's is nibbles alone, and the tile takes
// the two halves of a byte sixteen ints apart. Q5_K adds a fifth bit from a plane covering
// the whole superblock, and its halves land a quarter of a sub-block apart instead, because
// that plane is indexed by sub-block rather than by byte.
//
// The dp4a tile keeps the byte pair whole for Q4_K - its dot product splits the nibbles
// itself - while Q5_K must split them here, since a fifth bit does not fit in a nibble.

/// Q4_K: nibbles, nothing else.
struct plane_q4_K {
    using block = block_q4_K;
    static constexpr ggml_type type = GGML_TYPE_Q4_K;
    static constexpr int qr = QR4_K;
    static constexpr bool dp4a_holds_halves = false;
    static __device__ __forceinline__ void unpack(const block * b, const int txi, int & lo,
                                                  int & hi, int & k_lo, int & k_hi, int & raw) {
        raw = four_bytes(b->qs, txi);
        lo = (raw >> 0) & 0x0F0F0F0F;
        hi = (raw >> 4) & 0x0F0F0F0F;
        // The two halves of a byte are sixteen ints apart along the tile row.
        k_lo = 16 * (txi / 8) + txi % 8 + 0;
        k_hi = 16 * (txi / 8) + txi % 8 + 8;
    }
};

/// Q5_K: the same nibbles with a fifth bit from a superblock-wide plane.
struct plane_q5_K {
    using block = block_q5_K;
    static constexpr ggml_type type = GGML_TYPE_Q5_K;
    static constexpr int qr = QR5_K;
    /// A fifth bit does not fit a nibble, so both tiles hold the halves already apart.
    static constexpr bool dp4a_holds_halves = true;
    static __device__ __forceinline__ void unpack(const block * b, const int txi, int & lo,
                                                  int & hi, int & k_lo, int & k_hi, int & raw) {
        raw = four_bytes(b->qs, txi);
        // The fifth-bit plane is indexed by sub-block, and which of its two bits this lane
        // wants depends on which quarter of the sub-block it is in.
        const int qh = four_bytes(b->qh, txi % (QI5_K / 4));
        const int shift = 2 * (txi / (QI5_K / 4));
        lo = ((raw >> 0) & 0x0F0F0F0F) | (((qh >> (shift + 0)) << 4) & 0x10101010);
        hi = ((raw >> 4) & 0x0F0F0F0F) | (((qh >> (shift + 1)) << 4) & 0x10101010);

        // Here the halves land a quarter of a sub-block apart, that plane being indexed by
        // sub-block rather than by byte.
        const int ky = QR5_K * txi;
        k_lo = ky - ky % (QI5_K / 2) + txi % (QI5_K / 4) + 0;
        k_hi = ky - ky % (QI5_K / 2) + txi % (QI5_K / 4) + QI5_K / 4;
    }
};

/// Fill the weight plane of a 4- or 5-bit superblock tile.
template <typename plane, int mmq_y, bool need_check>
static __device__ __forceinline__ void load_weights_nibble_K(const char * __restrict__ x,
                                                             int * __restrict__ x_tile,
                                                             const int kbx0, const int i_max,
                                                             const int stride) {
    using block = typename plane::block;
    using tile = mmq_tile<plane::type, mmq_y>;
    constexpr int lanes = MMQ_ITER_K / (4 * plane::qr);
    const int txi = threadIdx.x % lanes;

    int * x_qs = tile::qs(x_tile);

    mmq_fill_rows<block, lanes, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block * bxi) {
            int lo, hi, k_lo, k_hi, raw;
            plane::unpack(bxi, txi, lo, hi, k_lo, k_hi, raw);

            int * row = x_qs + tile::qs_row(i);
            if constexpr (mmq_tile_splits_bytes<plane>()) {
                row[k_lo] = lo;
                row[k_hi] = hi;
            } else {
                // Whole, one lane one int: `raw` is what the dot product will split.
                row[txi] = raw;
            }
        });
}

/// The eight (scale, min) pairs a Q4_K or Q5_K superblock states.
///
/// Both spell them the same way - six bits each, packed into twelve bytes - so this is
/// written once and templated on the block. The minimum carries a negative sign because the
/// dot product ADDS its term rather than subtracting it.
template <typename block, int mmq_y, bool need_check>
static __device__ __forceinline__ void load_scales_q45_K(const char * __restrict__ x,
                                                         half2 * __restrict__ x_dm,
                                                         int * __restrict__ x_sc, const int kbx0,
                                                         const int i_max, const int stride) {
    if constexpr (mmq_card::tile_multiply) {
        // Two lanes to a row, each taking a word of four pairs already met with the
        // superblock's own. This pass does not go through the shared row walk, because it does
        // not answer the "which row" question the same way.
        constexpr int nwarps = mmq_get_nwarps_device();
        constexpr int rows_per_warp = ggml_cuda_get_physical_warp_size() / 2;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
            int row = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
            if constexpr (mmq_card::wavefront_operand) {
                // A sixty-four lane wave reaches past the tile in one step, so a row beyond it
                // is dropped rather than wrapped: wrapping would have two lanes load the same
                // row and write it twice.
                if (row >= mmq_y) {
                    continue;
                }
            } else {
                row %= mmq_y;
            }
            const int i = need_check ? min(row, i_max) : row;

            const block * bxi = (const block *) x + kbx0 + i*stride;
            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scale_word_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scale_word_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

            mmq_spread_word(x_dm + i*MMQ_MMA_TILE_X_K_Q8_1, ksc,
                            [&] (const int l) { return dm*make_half2(sc8[l], m8[l]); });
        }
    } else {
        // The dp4a tile keeps the superblock's own pair and the eight it multiplies apart, so
        // they are two passes: one lane to a row for the pair, one word of four to a lane for
        // the rest.
        mmq_fill_rows<block, 1, mmq_y, need_check>(x, kbx0, stride, i_max,
            [&] (const int i, const block * bxi) {
                x_dm[i] = bxi->dm;
            });

        constexpr int scale_words = MMQ_TILE_NE_K/8;
        const int ksc = threadIdx.x % scale_words;

        mmq_fill_rows<block, scale_words, mmq_y, need_check>(x, kbx0, stride, i_max,
            [&] (const int i, const block * row_blocks) {
                const block * bxi = row_blocks + ksc / (QI4_K/8);
                x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] =
                    unpack_scale_word_q45_K((const int *) bxi->scales, ksc);
            });
    }
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q4_K(const char * __restrict__ x,
                                              int * __restrict__ x_tile, const int kbx0,
                                              const int i_max, const int stride) {
    using tile = mmq_tile<GGML_TYPE_Q4_K, mmq_y>;
    load_weights_nibble_K<plane_q4_K, mmq_y, need_check>(x, x_tile, kbx0, i_max, stride);
    load_scales_q45_K<block_q4_K, mmq_y, need_check>(
        x, tile::dm(x_tile), tile::sc(x_tile), kbx0, i_max, stride);
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    vec_dot_over_tile_dp4a<mmq_x, mmq_y, QR4_K*VDR_Q4_K_Q8_1_MMQ>(sum,
        [&] (const int i, const int j, const int k01) {
            const int k0 = k00 + k01;
            const uint8_t * sc = (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/32] + 2*(k01/16);
            return vec_dot_q4_K_q8_1_impl_mmq(
                &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
        });
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q5_K(const char * __restrict__ x,
                                              int * __restrict__ x_tile, const int kbx0,
                                              const int i_max, const int stride) {
    using tile = mmq_tile<GGML_TYPE_Q5_K, mmq_y>;
    load_weights_nibble_K<plane_q5_K, mmq_y, need_check>(x, x_tile, kbx0, i_max, stride);
    load_scales_q45_K<block_q5_K, mmq_y, need_check>(
        x, tile::dm(x_tile), tile::sc(x_tile), kbx0, i_max, stride);
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    vec_dot_over_tile_dp4a<mmq_x, mmq_y, QR5_K*VDR_Q5_K_Q8_1_MMQ>(sum,
        [&] (const int i, const int j, const int k01) {
            const int k0 = k00 + k01;
            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k00/32]) + 2*(k01/16);
            return vec_dot_q5_K_q8_1_impl_mmq(
                &x_qs[i*(QR5_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
        });
}

// Q6_K: four low bits from one plane, two high bits from another, signed by an offset of 32.
//
// A lane owns four weights of the low plane, which is four ints of the output; the two bits
// that complete them sit in a plane indexed by sub-block, so which pair of bits this lane
// wants depends on which half of the sub-block it is in.
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_q6_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max,
    const int stride) {
    using tile = mmq_tile<GGML_TYPE_Q6_K, mmq_y>;

    int * x_qs = tile::qs(x_tile);
    float * x_df = tile::df(x_tile);
    int * x_sc = tile::sc(x_tile);

    constexpr int lanes = MMQ_ITER_K / (4 * QR6_K);
    const int txi = threadIdx.x % lanes;

    mmq_fill_rows<block_q6_K, lanes, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block_q6_K * bxi) {
            const int ql = four_bytes_unaligned(bxi->ql, txi);
            const int ql0 = (ql >> 0) & 0x0F0F0F0F;
            const int ql1 = (ql >> 4) & 0x0F0F0F0F;

            const int qh = four_bytes_unaligned(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
            const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
            const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

            // The two halves of a byte are half a sub-block apart along the tile row.
            int * row = x_qs + tile::qs_row(i);
            row[2*txi - txi % (QI6_K/2) + 0]        = __vsubss4(ql0 | qh0, 0x20202020);
            row[2*txi - txi % (QI6_K/2) + QI6_K/2]  = __vsubss4(ql1 | qh1, 0x20202020);
        });

    // One float for the whole superblock.
    mmq_fill_rows<block_q6_K, 1, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block_q6_K * bxi) {
            x_df[tile::df_row(i)] = bxi->d;
        });

    // The signed 6-bit sub-scales. The two multiplies read them at different granularities  - 
    // the tensor cores take four per lane, dp4a one - so the column differs, not just the row.
    constexpr int scale_words = MMQ_TILE_NE_K/8;

    mmq_fill_rows<block_q6_K, scale_words, mmq_y, need_check>(x, kbx0, stride, i_max,
        [&] (const int i, const block_q6_K * row_blocks) {
            const block_q6_K * bxi = row_blocks + (threadIdx.x % scale_words) / 4;
            if constexpr (mmq_card::tile_multiply) {
                x_sc[tile::sc_row(i) + threadIdx.x%4] =
                    four_bytes_unaligned(bxi->scales, threadIdx.x % scale_words);
            } else {
                x_sc[tile::sc_row(i) + threadIdx.x%scale_words] =
                    four_bytes_unaligned(bxi->scales, threadIdx.x%(QI6_K/8));
            }
        });
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    vec_dot_over_tile_dp4a<mmq_x, mmq_y, QR6_K*VDR_Q6_K_Q8_1_MMQ>(sum,
        [&] (const int i, const int j, const int k01) {
            const int k0 = k00 + k01;
            const int8_t * sc = ((const int8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/16]);
            return vec_dot_q6_K_q8_1_impl_mmq(
                &x_qs[i*(QR6_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
                x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
        });
}

// -------------------------------------------------------------------------------------------------------------------------------------

/// What one weight format contributes: a pass that fills a tile from its bytes, and the two
/// dot products over that tile - one per multiply a card might have.
///
/// The three are member function TEMPLATES rather than a table of function pointers, so the
/// tile width, the tile height and the edge check reach them from the call site instead of
/// being baked into a pointer at the point of declaration. Only the arm the card can run is
/// ever named, so only that arm is instantiated.
///
/// The parentheses each row wraps its three entries in are what let a template argument list
/// carry a comma through the expansion; what they wrap is called, not stored.
template <ggml_type type> struct mmq_ops;

#define MMQ_FORMAT(fmt, LOAD, DOT_MMA, DOT_DP4A)                             \
    template <> struct mmq_ops<fmt> {                                        \
        template <int mmq_y, bool need_check>                                \
        static __device__ __forceinline__ void load_tiles(                   \
                const char * __restrict__ x, int * __restrict__ x_tile,      \
                const int kbx0, const int i_max, const int stride) {         \
            LOAD(x, x_tile, kbx0, i_max, stride);                            \
        }                                                                    \
        template <int mmq_x, int mmq_y>                                      \
        static __device__ __forceinline__ void dot_mma(                      \
                const int * __restrict__ x, const int * __restrict__ y,      \
                float * __restrict__ sum, const int k00) {                   \
            DOT_MMA(x, y, sum, k00);                                         \
        }                                                                    \
        template <int mmq_x, int mmq_y>                                      \
        static __device__ __forceinline__ void dot_dp4a(                     \
                const int * __restrict__ x, const int * __restrict__ y,      \
                float * __restrict__ sum, const int k00) {                   \
            DOT_DP4A(x, y, sum, k00);                                        \
        }                                                                    \
    };

// A block of pairs - a nibble and its partner - reads through one loader, and the tensor-core
// dot then only cares whether the block carries a scale or a scale and an offset. Those two
// dots are the general `vec_dot_blocked_mma` under its two policies, named here rather than
// wrapped: a wrapper per format that only reorders its own template arguments is a row of this
// table written somewhere else.
MMQ_FORMAT(GGML_TYPE_Q4_0,
           (load_tiles_paired<plane_q4_0, mmq_y, need_check>),
           (vec_dot_blocked_mma<mmq_scale_only, mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>),
           (vec_dot_nibbles_q8_1_dp4a<mmq_x, mmq_y, GGML_TYPE_Q4_0, float, QI4_0,
                                      VDR_Q4_0_Q8_1_MMQ>))
MMQ_FORMAT(GGML_TYPE_Q4_1,
           (load_tiles_paired<plane_q4_1, mmq_y, need_check>),
           (vec_dot_blocked_mma<mmq_scale_and_min, mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>),
           (vec_dot_nibbles_q8_1_dp4a<mmq_x, mmq_y, GGML_TYPE_Q4_1, half2, QI4_1,
                                      VDR_Q4_1_Q8_1_MMQ>))
MMQ_FORMAT(GGML_TYPE_Q5_0,
           (load_tiles_paired<plane_q5_0, mmq_y, need_check>),
           (vec_dot_blocked_mma<mmq_scale_only, mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>),
           (vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>))
MMQ_FORMAT(GGML_TYPE_Q5_1,
           (load_tiles_paired<plane_q5_1, mmq_y, need_check>),
           (vec_dot_blocked_mma<mmq_scale_and_min, mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>),
           (vec_dot_q8_1_q8_1_dp4a<mmq_x, mmq_y>))
MMQ_FORMAT(GGML_TYPE_Q8_0,
           (load_tiles_q8_0<mmq_y, need_check>),
           (vec_dot_blocked_mma<mmq_scale_only, mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>),
           (vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>))

// The superblocks. Each has its own loader, because each packs its scales differently, and its
// tensor-core dot is the sub-block walk under whichever policy states its footing - Q3_K's is
// already a float in the tile, Q6_K's a sub-scale still to meet the superblock's own, Q2_K's a
// (scale, minimum) pair.
MMQ_FORMAT(GGML_TYPE_Q2_K,
           (load_tiles_q2_K<mmq_y, need_check>),
           (vec_dot_subblock_mma<mmq_subblock_pair, mmq_x, mmq_y>),
           (vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y>))
MMQ_FORMAT(GGML_TYPE_Q3_K,
           (load_tiles_q3_K<mmq_y, need_check>),
           (vec_dot_subblock_mma<mmq_subblock_float, mmq_x, mmq_y>),
           (vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y>))
MMQ_FORMAT(GGML_TYPE_Q4_K,
           (load_tiles_q4_K<mmq_y, need_check>),
           (vec_dot_blocked_mma<mmq_scale_and_min, mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>),
           (vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y>))
MMQ_FORMAT(GGML_TYPE_Q5_K,
           (load_tiles_q5_K<mmq_y, need_check>),
           (vec_dot_blocked_mma<mmq_scale_and_min, mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>),
           (vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y>))
MMQ_FORMAT(GGML_TYPE_Q6_K,
           (load_tiles_q6_K<mmq_y, need_check>),
           (vec_dot_subblock_mma<mmq_subblock_scaled, mmq_x, mmq_y>),
           (vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y>))

#undef MMQ_FORMAT

/// The dot product this card has, over one tile of one format.
template <ggml_type type, int mmq_x, int mmq_y>
static __device__ __forceinline__ void mmq_vec_dot(
        const int * __restrict__ x, const int * __restrict__ y,
        float * __restrict__ sum, const int k00) {
    if constexpr (mmq_card::tile_multiply) {
        mmq_ops<type>::template dot_mma<mmq_x, mmq_y>(x, y, sum, k00);
    } else {
        mmq_ops<type>::template dot_dp4a<mmq_x, mmq_y>(x, y, sum, k00);
    }
}

/// Copy one run of the activation into the staging buffer.
///
/// A block-wide copy: every thread moves the same number of ints and the passes together cover
/// the run exactly, which is why the buffer is reserved padded out to whole ones.
template <int mmq_x>
static __device__ __forceinline__ void mmq_stage_activation_run(
        int * __restrict__ tile_y, const int * __restrict__ by0) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = mmq_get_nwarps_device();

#pragma unroll
    for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
        const int l = l0 + threadIdx.y*warp_size + threadIdx.x;
        tile_y[l] = by0[l];
    }
}

template <ggml_type type, int mmq_x, bool need_check, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();

    // The block's shared memory, carved into the three parts `mmq_get_nbytes_shared` reserved:
    // the column indices, the activation tile padded out to whole block-wide copies, and the
    // weight tile, which takes whatever is left.
    extern __shared__ int mmq_shared[];
    int * tile_y = mmq_shared + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

    // How many weight blocks one pass covers, and what one run of the activation holds - the
    // values it stands for, and what it weighs in the ints the staging copy moves.
    constexpr int blocks_per_iter = get_iter_k() / qk;
    constexpr int run_values      = 4 * QK8_1;
    constexpr int run_ints        = sizeof(block_q8_1_mmq) / sizeof(int);

    // One entry per output this thread owns. How many that is follows from the tile and the
    // block; WHICH outputs they are is the multiply's business, and the write-back below asks
    // the same card the dot product did so that the two cannot disagree about it.
    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        mmq_ops<type>::template load_tiles<mmq_y, need_check>(
            x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

        // A pass covers two runs of the activation, and they share one staging buffer: each is
        // copied in, multiplied, and only then overwritten - which is what the pair of barriers
        // around the multiply is for.
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            mmq_stage_activation_run<mmq_x>(
                tile_y, y + ncols_y * (kb0*qk/run_values + half) * run_ints);

            __syncthreads();
            mmq_vec_dot<type, mmq_x, mmq_y>(tile_x, tile_y, sum, half * MMQ_TILE_NE_K);
            __syncthreads();
        }
    }

    if (fixup) {
        mmq_write_back<mmq_x, mmq_y, need_check>(
            sum, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        mmq_write_back<mmq_x, mmq_y, need_check>(
            sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

// ---------------------------------------------------------------------------------------
// Where an output tile is, and how a block reaches it.
//
// The driver walks ONE continuous index space - sample, then channel, then column tile, then
// row tile, then k-block - and hands each CUDA block an equal slice of it. A block therefore
// starts and stops wherever that division falls rather than on a tile boundary, so the same
// question comes up three times in a launch: given a position in that space, which tile is
// it, and what offsets reach that tile's weights, activations and destination? Below it is
// answered once.

/// A position in the driver's index space, unpacked.
struct mmq_tile_at {
    int it;   // row tile
    int jt;   // column tile
    int zt;   // channel
    int wt;   // sample
};

/// The strides a launch reaches its three operands by. Gathered once so that locating a tile
/// is one call rather than fourteen arguments repeated per call site.
struct mmq_layout {
    uint3 sample_ratio, channel_ratio;
    int stride_row_x, stride_col_dst;
    int stride_sample_x, stride_sample_y, stride_sample_dst;
    int stride_channel_x, stride_channel_y, stride_channel_dst;
};

/// Everything a tile pass needs beyond the tile's own coordinates.
struct mmq_tile_span {
    int offset_x, offset_y, offset_dst;   // where this tile's operands and result live
    int max_i, max_j;                     // the last row and column that is really there
};

static __device__ __forceinline__ mmq_tile_at mmq_tile_from_index(
        int flat, const uint3 ntx, const uint3 nchannels_y, const uint3 nsamples_y) {
    uint2 s = fast_div_modulo(flat, ntx);
    const int jt = s.y;
    s = fast_div_modulo(s.x, nchannels_y);
    const int zt = s.y;
    s = fast_div_modulo(s.x, nsamples_y);
    return {int(s.x), jt, zt, int(s.y)};
}

/// Where a tile's result begins, and the last row and column of it that is really there.
///
/// Asked twice per launch: once by the block that computes the tile, once by the block that
/// adds in what a split tile left behind. The second knows nothing of the operands, so this is
/// the part of locating a tile that does not mention them.
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ mmq_tile_span mmq_locate_dst(
        const mmq_tile_at t, const int stride_col_dst, const int stride_channel_dst,
        const int stride_sample_dst, const int ncols_dst, const int nrows_x) {
    mmq_tile_span s = {};
    s.offset_dst = t.wt*stride_sample_dst + t.zt*stride_channel_dst
                 + t.jt*mmq_x*stride_col_dst + t.it*mmq_y;
    s.max_i = nrows_x   - t.it*mmq_y - 1;
    s.max_j = ncols_dst - t.jt*mmq_x - 1;
    return s;
}

/// Resolve one tile: where its weights, activations and result live, and how much of it is
/// really there rather than padding.
///
/// The weights are shared across channels and samples where the activation has more of either,
/// which is what the two ratios divide out; the activation is addressed in whole runs.
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ mmq_tile_span mmq_locate_tile(
        const mmq_tile_at t, const mmq_layout & l, const int ncols_dst, const int nrows_x) {
    mmq_tile_span s = mmq_locate_dst<mmq_x, mmq_y>(
        t, l.stride_col_dst, l.stride_channel_dst, l.stride_sample_dst, ncols_dst, nrows_x);
    s.offset_y = t.wt*l.stride_sample_y + t.zt*l.stride_channel_y
               + t.jt*mmq_x*(sizeof(block_q8_1_mmq)/sizeof(int));
    s.offset_x = fastdiv(t.wt, l.sample_ratio)*l.stride_sample_x
               + fastdiv(t.zt, l.channel_ratio)*l.stride_channel_x
               + t.it*mmq_y*l.stride_row_x;
    return s;
}

/// How long the launch's single run of k-blocks is: every tile of every channel and every
/// sample, each one k row long. Both kernels cut the same run, so both count it the same way.
static __device__ __forceinline__ int64_t mmq_total_blocks(
        const int nrows_x, const int mmq_y, const uint3 ntx, const uint3 nchannels_y,
        const uint3 nsamples_y, const uint3 blocks_per_ne00) {
    const uint32_t nty = (nrows_x + mmq_y - 1) / mmq_y;   // row tiles
    return nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z;
}

/// Where CUDA block `bidx` starts in the launch's single run of k-blocks.
///
/// The run is cut into `gridDim.x` equal pieces and each cut is then pulled back to an
/// iteration boundary, because that is the stride the tile loop advances by: a block always
/// begins on a multiple of `blocks_per_iter` within its tile's k row. Asked for `bidx + 1` it
/// gives where the block stops.
static __device__ __forceinline__ int mmq_slice_start(
        const int bidx, const int64_t total_blocks, const uint3 blocks_per_ne00,
        const int blocks_per_iter) {
    int kbc = int64_t(bidx) * total_blocks / gridDim.x;
    kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;
    return kbc;
}

/// One tile, from its position in the index space.
///
/// `fixup` says whether this block reaches the end of the tile's k row - in which case the
/// result is written where it belongs - or holds only a part of it, which goes to the buffer
/// the second kernel adds in.
template <ggml_type type, int mmq_x, int mmq_y, bool need_check, bool fixup>
static __device__ __forceinline__ void mmq_run_tile(
        const mmq_tile_at t, const mmq_layout & l, const char * __restrict__ x,
        const int * __restrict__ y, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int ncols_dst, const int nrows_x, const int ncols_y,
        const int kb0_start, const int kb0_stop) {
    const mmq_tile_span s = mmq_locate_tile<mmq_x, mmq_y>(t, l, ncols_dst, nrows_x);
    mul_mat_q_process_tile<type, mmq_x, need_check, fixup>(
        x, s.offset_x, y + s.offset_y, dst + s.offset_dst, tmp_fixup, l.stride_row_x, ncols_y,
        l.stride_col_dst, s.max_i, s.max_j, kb0_start, kb0_stop);
}

/// The whole launch, as the two ways of dividing it below want to read it.
///
/// Gathered into one argument because there are two such ways, they take the same twelve
/// things, and a kernel whose body is a twelve-argument call twice over hides which of the two
/// it is running.
struct mmq_work {
    const char * __restrict__ x;
    const int  * __restrict__ y;
    float * __restrict__ dst;
    float * __restrict__ tmp_fixup;
    mmq_layout layout;
    uint3 blocks_per_ne00, nchannels_y, nsamples_y, ntx;
    int nrows_x, ncols_dst, ncols_y;
};

/// Conventional tiling: one block takes one tile and the whole of its k row.
template <ggml_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_block_takes_a_tile(const mmq_work & w) {
    const uint2 wz = fast_div_modulo(blockIdx.z, w.nchannels_y);
    const mmq_tile_at t = {int(blockIdx.x), int(blockIdx.y), int(wz.y), int(wz.x)};

    mmq_run_tile<type, mmq_x, mmq_y, need_check, false>(
        t, w.layout, w.x, w.y, w.dst, w.tmp_fixup, w.ncols_dst, w.nrows_x, w.ncols_y,
        0, w.blocks_per_ne00.z);
}

/// Stream-k: the whole launch is one run of k-blocks and a block takes an equal slice of it,
/// cut back to an iteration boundary. The slice starts and ends mid-tile, so the block finishes
/// every tile but its last and writes that one to the fixup buffer instead.
///
/// Described in https://arxiv.org/abs/2301.03598.
template <ggml_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_block_takes_a_slice_of_k(const mmq_work & w) {
    constexpr int blocks_per_iter = get_iter_k() / ggml_cuda_type_traits<type>::qk;

    const int64_t total_blocks = mmq_total_blocks(w.nrows_x, mmq_y, w.ntx, w.nchannels_y,
                                                  w.nsamples_y, w.blocks_per_ne00);
    int kbc            = mmq_slice_start(blockIdx.x,     total_blocks, w.blocks_per_ne00, blocks_per_iter);
    const int kbc_stop = mmq_slice_start(blockIdx.x + 1, total_blocks, w.blocks_per_ne00, blocks_per_iter);

    // Where in the current tile's k row this block starts, and where it stops: the end of the
    // row if the slice reaches it, otherwise wherever the slice runs out.
    int kb0_start = fastmodulo(kbc, w.blocks_per_ne00);
    int kb0_stop  = min(w.blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));

    // Every tile the block finishes. Reaching the end of a k row is what says it did.
    while (kbc < kbc_stop && kb0_stop == int(w.blocks_per_ne00.z)) {
        const mmq_tile_at t = mmq_tile_from_index(fastdiv(kbc, w.blocks_per_ne00),
                                                  w.ntx, w.nchannels_y, w.nsamples_y);
        mmq_run_tile<type, mmq_x, mmq_y, need_check, false>(
            t, w.layout, w.x, w.y, w.dst, w.tmp_fixup, w.ncols_dst, w.nrows_x, w.ncols_y,
            kb0_start, kb0_stop);

        kbc += w.blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, w.blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(w.blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // The partial tail. Another block holds the rest of this tile's k row, so the result goes
    // to the fixup buffer for the second kernel to add in.
    const mmq_tile_at t = mmq_tile_from_index(fastdiv(kbc, w.blocks_per_ne00),
                                              w.ntx, w.nchannels_y, w.nsamples_y);
    mmq_run_tile<type, mmq_x, mmq_y, need_check, true>(
        t, w.layout, w.x, w.y, w.dst, w.tmp_fixup, w.ncols_dst, w.nrows_x, w.ncols_y,
        kb0_start, kb0_stop);
}

template <ggml_type type, int mmq_x, bool need_check>
#if defined(GGML_USE_HIP)
#if defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 2)
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(CDNA) || defined(GCN)
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 1)
#else
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 2)
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP)
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y,
        float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx) {

    // Instantiations the launcher will never pick compile to nothing rather than to code.
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int mmq_y = get_mmq_y_device();

    const mmq_work w = {x, y, dst, tmp_fixup,
                        {sample_ratio, channel_ratio, stride_row_x, stride_col_dst,
                         stride_sample_x, stride_sample_y, stride_sample_dst,
                         stride_channel_x, stride_channel_y, stride_channel_dst},
                        blocks_per_ne00, nchannels_y, nsamples_y, ntx,
                        nrows_x, ncols_dst, ncols_y};

    // Which of the two divisions this card gets. Stream-k measured slower on non-CDNA AMD and
    // on anything before Volta, so those take a tile apiece instead.
#if (defined(GGML_USE_HIP) && !defined(CDNA)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    mmq_block_takes_a_tile<type, mmq_x, mmq_y, need_check>(w);
#else
    mmq_block_takes_a_slice_of_k<type, mmq_x, mmq_y, need_check>(w);
#endif // (defined(GGML_USE_HIP) && !defined(CDNA)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
}

/// Whether this block left part of a tile for the pass below to add in.
///
/// It did not if its slice held nothing, or if it began a tile - whoever begins a tile also
/// reaches the end of that k row and writes the tile whole - or if it both began and ended
/// inside one tile, in which case what it left is somebody else's to collect.
static __device__ __forceinline__ bool mmq_owes_a_fixup(
        const int kbc0, const int kbc0_stop, const uint3 blocks_per_ne00) {
    const bool held_nothing        = kbc0 == kbc0_stop;
    const bool began_a_tile        = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool finished_no_k_row   = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00)
                                  && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    return !(held_nothing || began_a_tile || finished_no_k_row);
}

/// Add up what the blocks that ended inside this tile left behind.
///
/// The walk runs backwards from this block. A predecessor whose slice held nothing contributes
/// nothing and is stepped over; the walk stops at the one that began this tile or began an
/// earlier one. `row` is the output row this lane owns, which is how the leftovers were
/// written and so how they are read.
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ bool mmq_add_up_leftovers(
        float (&sum)[mmq_x/nwarps], const float * __restrict__ tmp_last_tile, const int row,
        const int bidx0, const int kbc0, const int64_t total_blocks,
        const uint3 blocks_per_ne00, const int blocks_per_iter) {
    bool any_fixup = false;

    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = mmq_slice_start(bidx, total_blocks, blocks_per_ne00, blocks_per_iter);

        if (kbc == kbc_stop) { // this one had no data of its own
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + row];
        }

        // A predecessor that started in an earlier tile holds no more of this one.
        if (fastmodulo(kbc, blocks_per_ne00) == 0
                || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    return any_fixup;
}

template <ggml_type type, int mmq_x, bool need_check>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device()/2, 1)
static __global__ void mul_mat_q_stream_k_fixup(
        float * __restrict__ dst, float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
        const int stride_sample_dst, const uint3 ntx) {
    constexpr int mmq_y           = get_mmq_y_device();
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_iter = get_iter_k() / qk;

    // Half the warps of the kernel that produced the leftovers: this pass only adds them up.
    constexpr int nwarps = mmq_get_nwarps_device()/2;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    // One output row per lane, one column group per warp - the shape the leftovers were
    // written in, which is `mmq_write_back_dp4a`'s and not the accumulator's.
    float sum[mmq_x / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int bidx0 = blockIdx.x;

    // The same cut of the same run of k-blocks the first kernel made, recomputed rather than
    // carried: a block reads its predecessors' leftovers, so it has to know where they began.
    const int64_t total_blocks =
        mmq_total_blocks(nrows_x, mmq_y, ntx, nchannels_y, nsamples_y, blocks_per_ne00);
    const int kbc0      = mmq_slice_start(blockIdx.x,     total_blocks, blocks_per_ne00, blocks_per_iter);
    const int kbc0_stop = mmq_slice_start(blockIdx.x + 1, total_blocks, blocks_per_ne00, blocks_per_iter);

    if (!mmq_owes_a_fixup(kbc0, kbc0_stop, blocks_per_ne00)) {
        return;
    }

    const bool any_fixup = mmq_add_up_leftovers<mmq_x, mmq_y, nwarps>(
        sum, tmp_last_tile, i, bidx0, kbc0, total_blocks, blocks_per_ne00, blocks_per_iter);

    if (!any_fixup) {
        return;
    }

    // The tile those leftovers belong to, and where its result sits - the same question the
    // first kernel asked, so the same answer is used.
    const mmq_tile_at t =
        mmq_tile_from_index(fastdiv(kbc0, blocks_per_ne00), ntx, nchannels_y, nsamples_y);
    const mmq_tile_span s = mmq_locate_dst<mmq_x, mmq_y>(
        t, stride_col_dst, stride_channel_dst, stride_sample_dst, ncols_dst, nrows_x);
    dst += s.offset_dst;

    if (need_check && i > s.max_i) {
        return;
    }

    // Added in rather than stored: the block that finished the tile has already written it.
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > s.max_j) {
            return;
        }

        dst[j*stride_col_dst + i] += sum[j0/nwarps];
    }
}

struct mmq_args {
    const char * x; ggml_type type_x; const int * y; float * dst;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    bool use_stream_k; int64_t ncols_max;
};

/// What one block's tile pass needs of shared memory: the column indices, the weight tile and
/// the activation tile, which `mul_mat_q_process_tile` carves out of exactly this much.
///
/// The weight tile is the part that depends on the card: with matrix cores it is `mmq_y` padded
/// rows of one width, without them the three separately packed planes the format states. The
/// activation tile is padded out to whole block-wide copies, because that is how it is staged.
template<ggml_type type>
static size_t mmq_get_nbytes_shared(const int mmq_x, const int mmq_y, const int cc, const int warp_size, const int nwarps) {
    const bool matrix_cores = turing_mma_available(cc) || amd_mfma_available(cc)
                           || amd_wmma_available(cc);
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const size_t nbytes_x = matrix_cores
        ? size_t(mmq_y) * mmq_get_mma_tile_x_k(type) * sizeof(int)
        : txs.qs*sizeof(int) + txs.dm*sizeof(half2) + txs.sc*sizeof(int);

    return mmq_x*sizeof(int) + nbytes_x
         + GGML_PAD(mmq_x * sizeof(block_q8_1_mmq), nwarps*warp_size*sizeof(int));
}
