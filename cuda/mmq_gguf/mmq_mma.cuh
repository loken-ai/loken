#pragma once

// The tensor-core fragments the tiled quantised matmul computes in.
//
// A tensor-core instruction multiplies a row-major A (M x K) by a column-major B (K x N) into a
// column-major C (M x N). All three are held the same way here: an `I x J` tile of 32-bit
// registers spread across the warp, where **J counts registers, not logical elements** - an int
// register is four int8 weights, so a 16x8 int tile covers 16 x 32 of them.
//
// What an instruction fixes, and what this file therefore states, is WHICH element of the
// matrix a given lane's register `l` holds. Get it wrong and nothing fails to compile; the
// numbers are simply somebody else's.
//
// The mapping is said in three steps, none of which is a matrix multiply:
//
//   a CUT       - how a fragment is spread over the lanes. There are two of them, and every
//                 instruction here uses one or the other, differing only in a few numbers.
//   a FRAGMENT  - per instruction family, which `I x J` shapes exist, how many registers of one
//     SET         a lane owns, and what numbers its cut takes. One set per family; the build
//                 picks its own once, and no other declaration below names an architecture.
//   a TILE      - the registers themselves, plus the shape and the direction they are read in.
//                 A j-major tile is the same fragment with the two axes swapped; a mirrored one
//                 replaces the family's mapping with the one every thread holds whole.
//
// The loads and the accumulate calls sit on top of a tile and ask it where its registers go.
//
// Pointers handed to `load_ldmatrix` must be in shared memory and 16-byte aligned; the
// instruction has no unaligned form. `load_generic` assumes the same alignment.

#include "mmq_common.cuh"

namespace loken_mma {

/// The answer for a fragment shape this architecture has no instruction for.
///
/// Every mapping below ends in it, because a shape is either one the hardware knows or one
/// nothing can be said about - and saying nothing quietly would hand back an index. It traps
/// on the device and returns a value that cannot be mistaken for one.
static __device__ __forceinline__ int no_tile_for_this_shape() {
    NO_DEVICE_CODE;
    return -1;
}

// ============================================================
// The two cuts
// ============================================================

/// One lane, one row.
///
/// The lane index picks the row, wrapped over `rows` of them. The lanes past that wrap hold
/// further columns: the warp is cut into `runs` runs of `step` columns, a lane's own run is
/// chosen by which wrap it falls in, and its registers walk that run in order - one row, one
/// starting column, and nothing else to compute.
template <int rows, int runs, int step>
struct lane_holds_a_row {
    static __device__ __forceinline__ int row(const int /*l*/) {
        return threadIdx.x % rows;
    }
    static __device__ __forceinline__ int col(const int l) {
        return step * ((threadIdx.x / rows) % runs) + l;
    }
};

/// Four lanes, one row.
///
/// The warp's 32 lanes split into eight groups of four. The group picks a row - of eight for an
/// operand fragment, and for a taller accumulator the same group also holds the row eight, and
/// possibly sixteen and twenty-four, further down. The position within the group picks the
/// columns, and the register index says which of the group's rows and which of its columns.
///
/// The two shapes of that answer are the ISA's operand and accumulator tables, read in register
/// units rather than in int8 elements:
///   https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-multiply-accumulate-operation-using-mma-instruction
template <int I, int J>
struct four_lanes_hold_a_row {
    /// Which of the eight lane groups this lane belongs to.
    static __device__ __forceinline__ int group() {
        return threadIdx.x >> 2;
    }
    /// Where this lane sits inside its group of four.
    static __device__ __forceinline__ int in_group() {
        return threadIdx.x & 3;
    }

    /// Registers come in pairs on one row, and each further pair drops eight rows - as many
    /// times as the fragment has eights of rows before it repeats.
    static __device__ __forceinline__ int row(const int l) {
        return group() + 8 * ((l / 2) % (I / 8));
    }

    static __device__ __forceinline__ int col(const int l) {
        if constexpr (I == 8) {
            // An operand: one register per lane per quarter of k, the group's own quarter, and
            // a further register sixteen int8 - four registers - along.
            return in_group() + 4 * l;
        } else {
            // An accumulator: the group owns two adjacent columns, the low bit of `l` choosing
            // between them, and a second accumulator laid beside it sits eight columns along.
            return 2 * in_group() + (l % 2) + 8 * ((l / 4) % (J / 8));
        }
    }
};

// ============================================================
// The fragment sets
// ============================================================

/// The shapes a one-lane-one-row mapping states: sixteen rows, and a k covered by one, two or
/// four registers. Both families that cut fragments that way accept exactly these.
static constexpr bool sixteen_rows_shape(const int I, const int J) {
    return I == 16 && (J == 16 || J == 8 || J == 4);
}

/// `mma.sync`: 32 lanes, four to a row.
template <int I, int J, typename T>
struct mma_sync_fragments {
    static constexpr int regs = I * J / WARP_SIZE;
    static constexpr bool known = (I == 8 && (J == 4 || J == 8)) ||
                                  (I == 16 && (J == 8 || J == 16)) || (I == 32 && J == 8);
    /// Neither table hands a lane a run: an operand's registers are four columns apart, an
    /// accumulator's second pair is eight rows away.
    static constexpr bool contiguous = false;

    /// The 16-row mappings are the ACCUMULATOR's - the s32 C/D fragment shared by m16n8k16 and
    /// m16n8k32. The k=32 A operand happens to have the same `I x J` shape in registers but a
    /// different mapping; it is filled by `load_ldmatrix` and read by `mma`, never indexed.
    using cut = four_lanes_hold_a_row<I, J>;

    static __device__ __forceinline__ int row(const int l) { return cut::row(l); }
    static __device__ __forceinline__ int col(const int l) { return cut::col(l); }
};

/// `mfma`: a 64-lane wavefront, one lane to a row.
template <int I, int J, typename T>
struct mfma_fragments {
    static constexpr int regs = I * J / 64;
    /// 64x2 is how a 16x4 fragment is loaded when the instruction wants 16x8.
    static constexpr bool known = (I == 64 && J == 2) || (I == 16 && J == 8) ||
                                  (I == 32 && J == 4) || (I == 16 && J == 16) ||
                                  (I == 32 && J == 32);
    /// The 16x4-as-16x8 view is the one shape whose registers are not a single run.
    static constexpr bool contiguous = !(I == 64 && J == 2);

    /// How many rows the lane index wraps over: 32 for the wide fragments, 16 for the rest,
    /// including the 64x2 view whose 64 is a window and not a row count.
    static constexpr int rows = (I == 32) ? 32 : 16;
    /// The wave holds as many runs as it has wraps of rows - except in that same view, where
    /// the upper half repeats the lower.
    static constexpr int runs = (I == 64 && J == 2) ? 2 : 64 / rows;
    /// Columns from one run to the next. The 32-row wide fragment steps by half a lane's
    /// registers, the 32x4 one by all of them.
    ///
    /// The sixteen-row shapes step by twice a lane's registers, which walks `col` past `J`:
    /// 16x8 reaches column 13 and 16x16 reaches 27. Nothing here reconciles that with a
    /// 64-lane fragment; it is stated as measured and left alone.
    static constexpr int step = (I == 32 && J == 32) ? 8 : ((I == 16) ? 2 * regs : regs);

    using cut = lane_holds_a_row<rows, runs, step>;

    static __device__ __forceinline__ int row(const int l) { return cut::row(l); }
    static __device__ __forceinline__ int col(const int l) { return cut::col(l); }
};

/// `wmma`: a 32-lane wave, one lane to a row, the two halves of the wave holding the two runs.
template <int I, int J, typename T>
struct wmma_fragments {
    static constexpr int regs = I * J / 32;
    static constexpr bool known = sixteen_rows_shape(I, J);
    static constexpr bool contiguous = true;

    using cut = lane_holds_a_row<16, 2, regs>;

    static __device__ __forceinline__ int row(const int l) { return cut::row(l); }
    static __device__ __forceinline__ int col(const int l) {
#if defined(RDNA3)
        // One accumulator here is not two runs: the halves of the wave interleave column by
        // column when it accumulates in 32-bit, and share every column when it does not.
        if constexpr (J == 16 && (std::is_same_v<T, float> || std::is_same_v<T, int>)) {
            return 2 * l + (threadIdx.x / 16);
        } else if constexpr (J == 16) {
            return l;
        } else
#endif
        {
            return cut::col(l);
        }
    }
};

/// The mapping of a fragment that is not divided at all: every thread of a subgroup holds the
/// whole row, so `l` is the column outright and the tile carries twice the registers.
template <int I, int J, typename T>
struct mirrored_fragments {
    static constexpr int regs = I * J / 32 * 2;
    static constexpr bool known = sixteen_rows_shape(I, J);
    static constexpr bool contiguous = true;

    /// One run, which is what "holds it whole" means: no wrap of lanes moves the column.
    using cut = lane_holds_a_row<16, 1, regs>;

    static __device__ __forceinline__ int row(const int l) { return cut::row(l); }
    static __device__ __forceinline__ int col(const int l) { return cut::col(l); }
};

// ============================================================
// Which set this build's instructions state
//
// The only place below that names an architecture. `fills_by_copy` says whether a fragment
// whose registers are a run is filled by copying the run - where the answer is no, every
// fragment is filled element by element whatever its mapping allows.
// ============================================================

#if defined(AMD_MFMA_AVAILABLE)
template <int I, int J, typename T> using native_fragments = mfma_fragments<I, J, T>;
static constexpr bool fills_by_copy() { return true; }
#elif defined(AMD_WMMA_AVAILABLE)
template <int I, int J, typename T> using native_fragments = wmma_fragments<I, J, T>;
static constexpr bool fills_by_copy() { return true; }
#else
template <int I, int J, typename T> using native_fragments = mma_sync_fragments<I, J, T>;
static constexpr bool fills_by_copy() { return false; }
#endif

// ============================================================
// Tiles
// ============================================================

/// How a fragment's data is spread across a warp.
///
/// Some architectures run several matrix multiplies per warp at once - the warp splits into
/// subgroups, each doing one - and then there is a choice about which way the data is cut.
/// MIRRORED means every subgroup holds the whole value: each thread of it has its own copy.
enum data_layout {
    /// I is the major direction. For A and C that is row-major, for B column-major. Turing
    /// through consumer Blackwell always; A and B on RDNA4 and CDNA.
    DATA_LAYOUT_I_MAJOR = 0,
    /// Transposed: C on CDNA and RDNA4, and int/float C on RDNA3.
    DATA_LAYOUT_J_MAJOR = 10,
    /// Volta, and A and B on RDNA3.
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
    DATA_LAYOUT_J_MAJOR_MIRRORED = 30,
};

// The accumulate calls below implement (A, B) -> D for:
//   (I_MAJOR, I_MAJOR)          -> I_MAJOR
//   (I_MAJOR, I_MAJOR_MIRRORED) -> I_MAJOR
//   (I_MAJOR, J_MAJOR_MIRRORED) -> I_MAJOR

static constexpr bool is_i_major(const data_layout dl) {
    return dl == DATA_LAYOUT_I_MAJOR || dl == DATA_LAYOUT_I_MAJOR_MIRRORED;
}

static constexpr bool is_mirrored(const data_layout dl) {
    return dl == DATA_LAYOUT_I_MAJOR_MIRRORED || dl == DATA_LAYOUT_J_MAJOR_MIRRORED;
}

/// Which layout this architecture's A and B fragments arrive in.
static constexpr __device__ data_layout get_input_data_layout() {
#if defined(RDNA3) || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    return DATA_LAYOUT_I_MAJOR_MIRRORED;
#else
    return DATA_LAYOUT_I_MAJOR;
#endif
}

/// A mirrored layout answers with the shared mapping whatever the instruction's own says; every
/// other layout takes the one the build's instructions state. One question with two answers, so
/// one conditional and not a table.
template <int I, int J, typename T, data_layout dl>
using fragments_of = std::conditional_t<is_mirrored(dl), mirrored_fragments<I, J, T>,
                                        native_fragments<I, J, T>>;

/// A fragment: the shape it covers, the direction it is read in, and the registers themselves.
template <int I_, int J_, typename T, data_layout dl_ = DATA_LAYOUT_I_MAJOR>
struct tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr data_layout layout = dl_;

    using fragment = fragments_of<I_, J_, T, dl_>;

    static constexpr bool swapped = !is_i_major(layout);

    /// How many registers a lane owns depends on how many lanes share the fragment, which is
    /// the mapping's answer and not the shape's.
    static constexpr int ne = fragment::regs;

    /// Whether one wide copy can stand in for the element-wise fill: the architecture has to
    /// fill that way, the mapping has to lay a lane's registers out in memory order, and the
    /// tile has to be read the way the mapping states it.
    static constexpr bool runs_are_contiguous =
        fills_by_copy() && fragment::contiguous && !swapped;

    T x[ne] = {0};

    static constexpr __device__ bool supported() {
        return fragment::known;
    }

    /// Where this lane's register `l` sits, along whichever of the two axes is asked for.
    ///
    /// The mapping states a row and a column. Reading j-major is asking it for the other one of
    /// the two, so the two directions differ by which answer is taken and by nothing else - one
    /// question, not two mappings.
    ///
    /// A shape the mapping does not state is refused here, once, so that every set above can be
    /// the arithmetic alone.
    template <bool want_row>
    static __device__ __forceinline__ int along(const int l) {
        if constexpr (!supported()) {
            return no_tile_for_this_shape();
        } else if constexpr (want_row != swapped) {
            return fragment::row(l);
        } else {
            return fragment::col(l);
        }
    }

    static __device__ __forceinline__ int get_i(const int l) { return along<true>(l); }
    static __device__ __forceinline__ int get_j(const int l) { return along<false>(l); }
};

// ============================================================
// Filling a fragment
//
// Two fills, and the tile itself says which of them it admits.
// ============================================================

/// Ask the mapping for every register in turn.
///
/// Correct for every shape and every architecture whatever its mapping says, and the fallback
/// wherever the hardware has no matrix load.
template <typename Tile, typename T>
static __device__ __forceinline__ void fill_from_the_mapping(Tile &t, const T *__restrict__ xs0,
                                                             const int stride) {
#pragma unroll
    for (int l = 0; l < Tile::ne; ++l) {
        t.x[l] = xs0[Tile::get_i(l) * stride + Tile::get_j(l)];
    }
}

/// Move a lane's registers whole, where the tile says they are a run in memory.
///
/// The run goes in as many pieces as the widest single copy allows, each piece starting on the
/// register the wide copy would have reached anyway, so that every one of them is as aligned as
/// the first.
template <typename Tile, typename T>
static __device__ __forceinline__ void fill_as_one_run(Tile &t, const T *__restrict__ xs0,
                                                       const int stride) {
    constexpr int aligned_copy_bytes = ggml_cuda_get_max_cpy_bytes();
    constexpr int fragment_bytes = Tile::ne * sizeof(T);
    constexpr int pieces =
        fragment_bytes > aligned_copy_bytes ? fragment_bytes / aligned_copy_bytes : 1;
    static_assert(fragment_bytes % pieces == 0, "bad type size");
    constexpr int regs_per_piece = Tile::ne / pieces;
#pragma unroll
    for (int p = 0; p < pieces; ++p) {
        copy_bytes<fragment_bytes / pieces>(
            t.x + regs_per_piece * p,
            xs0 + Tile::get_i(0) * stride + Tile::get_j(regs_per_piece * p));
    }
}

/// Fill a fragment with ordinary loads.
template <int I, int J, typename T, data_layout dl>
static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> &t,
                                                    const T *__restrict__ xs0, const int stride) {
    if constexpr (tile<I, J, T, dl>::runs_are_contiguous) {
        fill_as_one_run(t, xs0, stride);
    } else {
        fill_from_the_mapping(t, xs0, stride);
    }
}

// ============================================================
// Filling a fragment with one instruction
// ============================================================

#ifdef TURING_MMA_AVAILABLE

/// `ldmatrix` moves 8x8 blocks of 16-bit values out of shared memory. Each lane supplies the
/// address of one row of one block - lanes 0..7 the rows of the first block, 8..15 of the
/// second, and so on for as many blocks as the `.xN` suffix names - and the hardware hands
/// back, per block, the pair of 16-bit values at row `l/4`, columns `2*(l%4)` and one past it.
/// One 32-bit register, therefore, per block per lane, at row `l/4` and int-column `l%4`.
///
/// A tile is `I/8` blocks of rows by `nj` blocks of columns, and the whole of the choice is
/// which region each lane points at, so that the blocks come back in the order the accumulate
/// instruction expects its operand: the lane's row wraps over the tile's rows, and every wrap
/// of lanes past that steps one block along in k. Where the warp holds more wraps than there
/// are blocks to step through - a short tile - the surplus wraps address the same blocks again.
/// The addresses are int addresses because these tiles hold packed int8, four to the register.
template <int nj, int I, int J, typename T>
static __device__ __forceinline__ const int *ldmatrix_source(const T *__restrict__ xs0,
                                                             const int stride) {
    if constexpr (WARP_SIZE / I > nj) {
        return (const int *) xs0 + (threadIdx.x % I) * stride + ((threadIdx.x / I) % nj) * (J / nj);
    } else {
        return (const int *) xs0 + (threadIdx.x % I) * stride + (threadIdx.x / I) * (J / nj);
    }
}

static __device__ __forceinline__ void ldmatrix_x2(int *xi, const int *xs) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(xi[0]), "=r"(xi[1]) : "l"(xs));
}

static __device__ __forceinline__ void ldmatrix_x4(int *xi, const int *xs) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                 : "l"(xs));
}

#endif // TURING_MMA_AVAILABLE

/// Fill a fragment with one instruction instead of `ne` loads.
///
/// Three tile shapes are stated, and what the shape decides is how many 8x8 blocks a lane
/// addresses and in what order they must come back:
///
///   8x8   two blocks over the same eight rows, the second half the tile's registers along in
///         k. Only the first sixteen lanes are read, so the wrap keeps the upper half of the
///         warp addressing those same two blocks.
///   16x4  two blocks that are the top and bottom halves of the tile, which is what the row
///         wrap over sixteen lanes already gives, and no lane steps along k.
///   16x8  four blocks: top half then bottom half of the first half of k, then the same of the
///         second - the order the k=32 A operand is stated in.
///
/// Turing introduced the matrix load. Volta and older fall back to the ordinary fill, except
/// for the two shapes whose Volta layout that fill cannot express.
template <int I, int J, typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix(tile<I, J, T, dl> &t,
                                                     const T *__restrict__ xs0, const int stride) {
    static_assert((I == 8 && J == 8) || (I == 16 && J == 4) || (I == 16 && J == 8),
                  "no matrix load states this tile shape");
#if defined(TURING_MMA_AVAILABLE)
    // Two blocks along k wherever J covers twice what one block does, and `I/8` blocks of rows.
    constexpr int nj = (J == 4) ? 1 : 2;
    constexpr int nblocks = nj * (I / 8);
    if constexpr (nblocks == 4) {
        ldmatrix_x4((int *) t.x, ldmatrix_source<nj, I, J>(xs0, stride));
    } else {
        ldmatrix_x2((int *) t.x, ldmatrix_source<nj, I, J>(xs0, stride));
    }
#elif __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    if constexpr (I == 16 && J == 8) {
        // Volta's own permutation makes each lane's eight registers two contiguous runs of four.
        static_assert(sizeof(T) == 4, "bad type size");
        copy_bytes<4 * sizeof(T)>(t.x + 0, xs0 + t.get_i(0) * stride + 0);
        copy_bytes<4 * sizeof(T)>(t.x + 4, xs0 + t.get_i(4) * stride + 4);
    } else if constexpr (I == 16) {
        // Volta holds a 16x4 mirrored across the subgroup, which the ordinary fill does not
        // describe; no caller reaches it there.
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
    } else {
        load_generic(t, xs0, stride);
    }
#else
    load_generic(t, xs0, stride);
#endif
}

// ============================================================
// Accumulating
// ============================================================

#ifdef TURING_MMA_AVAILABLE

/// The 8-row product Turing states, which is what it has instead of the 16-row ones.
static __device__ __forceinline__ void mma_m8n8k16(int &d0, int &d1, const int a, const int b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(d0), "+r"(d1)
        : "r"(a), "r"(b));
}

/// One k=16 step of a 16-row product, as the two 8-row halves it is issued in: a register of A
/// is the top or the bottom eight rows, and the accumulator splits the same way, so each half
/// is an 8-row product against the whole of B.
static __device__ __forceinline__ void mma_k16_by_halves(int *d, const int a_top,
                                                         const int a_bottom, const int b) {
    mma_m8n8k16(d[0], d[1], a_top, b);
    mma_m8n8k16(d[2], d[3], a_bottom, b);
}

#endif // TURING_MMA_AVAILABLE

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
// The matrix intrinsics take a fragment as a vector of registers rather than as an operand
// list, so a tile's array is handed over reinterpreted. These are the widths that takes.
using regs2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using regs4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using regs8 = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using regs16 = __attribute__((__vector_size__(16 * sizeof(int)))) int;
#endif

/// The products stated below, in the three numbers a product is.
///
/// A 16-row output comes at eight or sixteen columns over a k of one or two registers' worth. A
/// 32-row one is either the square CDNA's own instruction produces or any of those stacked.
static constexpr __device__ bool product_is_stated(const int I, const int J, const int K) {
    const bool over_sixteen_rows = (J == 8 || J == 16) && (K == 4 || K == 8);
    return (I == 16 && over_sixteen_rows) ||
           (I == 32 && (over_sixteen_rows || (J == 32 && K == 4)));
}

/// `D += A * B` over int8 operands accumulating in int32.
///
/// Every product these kernels take, at one entry point, because a product is three numbers and
/// not a family of calls: an output of `I` rows by `J` columns over a k of `K` registers - which
/// is `4*K` int8 values, the operands' J counting registers like every other J here. m16n8k16 is
/// therefore (16, 8, 4) and m16n8k32 is (16, 8, 8).
///
/// No two architectures state the same shapes, and the numbers are what pick between them.
/// NVIDIA has the 16x8 outputs, and below Ampere issues each as its two 8-row halves. A 16x16
/// output is what one AMD instruction produces, CDNA has a 32x32 of its own, and a 32-row output
/// nothing states outright is two 16-row products stacked.
template <int I, int J, int K, data_layout dl_d, data_layout dl_ab>
static __device__ __forceinline__ void mma(tile<I, J, int, dl_d> &D,
                                           const tile<I, K, int, dl_ab> &A,
                                           const tile<J, K, int, dl_ab> &B) {
    static_assert(product_is_stated(I, J, K), "no instruction states this product");

    if constexpr (I == 32 && J != 32) {
        // Two stacked 16-row tiles with the same layout, so the fragment splits by
        // reinterpretation and nothing moves.
        tile<16, J, int, dl_d> *D16 = reinterpret_cast<tile<16, J, int, dl_d> *>(&D);
        const tile<16, K, int, dl_ab> *A16 = reinterpret_cast<const tile<16, K, int, dl_ab> *>(&A);
        mma(D16[0], A16[0], B);
        mma(D16[1], A16[1], B);

    } else if constexpr (J == 8) {
        // NVIDIA's 16-row output.
#ifdef TURING_MMA_AVAILABLE
        if constexpr (K == 4) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
                : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
            mma_k16_by_halves(D.x, A.x[0], A.x[1], B.x[0]);
#endif
        } else {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
                "%9}, {%0, %1, %2, %3};"
                : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
                : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
            // A's registers run top-half then bottom-half within each half of k, and B's run one
            // per half of k, so pairing them is register order on both sides.
            mma_k16_by_halves(D.x, A.x[0], A.x[1], B.x[0]);
            mma_k16_by_halves(D.x, A.x[2], A.x[3], B.x[1]);
#endif
        }
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif

    } else if constexpr (J == 16 && K == 8) {
        // AMD's 16x16 accumulator over the whole of k. CDNA3 takes both halves of the operands
        // at once; CDNA2 and RDNA issue two k=16 steps.
#if defined(AMD_MFMA_AVAILABLE)
        regs4 *acc = (regs4 *) D.x;
#if defined(CDNA3)
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x32_i8(((int64_t *) A.x)[0], ((int64_t *) B.x)[0], acc[0], 0, 0, 0);
#elif defined(CDNA2) || defined(CDNA)
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x16i8(A.x[0], B.x[0], acc[0], 0, 0, 0);
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x16i8(A.x[1], B.x[1], acc[0], 0, 0, 0);
#endif
#elif defined(AMD_WMMA_AVAILABLE)
        regs8 *acc = (regs8 *) D.x;
#if defined(RDNA4)
        regs2 *a_vec = (regs2 *) A.x;
        regs2 *b_vec = (regs2 *) B.x;
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_vec[0], true, b_vec[0], acc[0], true);
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_vec[1], true, b_vec[1], acc[0], true);
#elif defined(RDNA3)
        regs4 *a_vec = (regs4 *) A.x;
        regs4 *b_vec = (regs4 *) B.x;
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
#endif
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif

    } else if constexpr (J == 16) {
        // The same accumulator over half the k: one k=16 step, no accumulation of a second.
#if defined(AMD_WMMA_AVAILABLE)
        regs8 *acc = (regs8 *) D.x;
#if defined(RDNA4)
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, ((regs2 *) A.x)[0], true, ((regs2 *) B.x)[0], acc[0], false);
#elif defined(RDNA3)
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, ((regs4 *) A.x)[0], true, ((regs4 *) B.x)[0], acc[0], false);
#endif
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif

    } else {
        // CDNA's 32-row instruction, whose accumulator is a whole 32x32 tile.
#if defined(AMD_MFMA_AVAILABLE)
        regs16 *acc = (regs16 *) D.x;
#if defined(CDNA3)
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x16_i8(((int64_t *) A.x)[0], ((int64_t *) B.x)[0], acc[0], 0, 0, 0);
#elif defined(CDNA2) || defined(CDNA)
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x8i8(A.x[0], B.x[0], acc[0], 0, 0, 0);
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x8i8(A.x[1], B.x[1], acc[0], 0, 0, 0);
#endif
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif
    }
}

}  // namespace loken_mma
