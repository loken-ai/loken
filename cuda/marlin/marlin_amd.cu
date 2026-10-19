// The W4A16 GEMM for AMD cards: the entry point of `marlin_launcher.cu`, with the same
// arguments, the same repacked operands and the same results, on AMD's matrix instructions.
//
//   A      = f16 [m, k] row-major (lda = k)
//   B      = repacked u4 [k/16, n*2] int32, integer zero point (src/tensor/marlin.rs)
//   scales = f16 [k/128, n], lane-permuted
//   zp     = u4  [k/128, n/8] int32, permuted and interleaved
//   C      = f16 [m, n]
//
// The repack places the weights for NVIDIA's `m16n8k16` fragments. One u32 of `b_q` holds, for
// one 16x16 tile, the two output columns `n0` and `n0 + 8` at the four reduction indices
// `2q, 2q+1, 2q+8, 2q+9`, where the word's slot inside its 64-column group is `4 (4 n0 + q) + j`
// for tile `j`. A 16x16x16 AMD matrix product wants each lane to hold one output column and a
// contiguous share of k instead. So a lane reads the words of its column for the quads `q` it
// covers, keeps the nibble half of its column, and gives the activation fragment the same k
// order: `2q, 2q+1` for every covered quad, then `2q+8, 2q+9` for every covered quad. A k order
// shared by both operands leaves the dot product unchanged.
//
// The three matrix-instruction families differ only in how many quads a lane covers and where
// a lane's accumulator rows sit:
//
//   family  wave  quads per lane  accumulator row of element r
//   RDNA4    32   2 (lane/16)     8 (lane/16) + r
//   RDNA3    32   4 (all)         2 r + lane/16
//   CDNA     64   1 (lane/16)     4 (lane/16) + r
//
// Work split: a block owns one tile of `16 MT` rows by 64 columns and one slice of k, cut on
// scale-group boundaries; its waves split the slice's k again and fold their accumulators
// through shared memory. A problem with fewer tiles than the card has multiprocessors cuts
// every tile into several slices: each slice leaves its f32 partial in `c_tmp`, and the last
// slice to finish, counted on the tile's lock, adds the partials in slice order, writes C and
// zeroes the lock. A problem with at least one tile per multiprocessor never touches `c_tmp`.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace loken_w4a16_amd {

typedef _Float16 h2 __attribute__((ext_vector_type(2)));

/// The matrix instruction of the target being compiled, and its fragment geometry.
struct Core {
#if defined(RDNA4)
    static constexpr int wave = 32, quads = 2, acc = 8;
    static __device__ __forceinline__ int row(int lane, int r) { return 8 * (lane >> 4) + r; }
#elif defined(RDNA3)
    static constexpr int wave = 32, quads = 4, acc = 8;
    static __device__ __forceinline__ int row(int lane, int r) { return 2 * r + (lane >> 4); }
#elif defined(CDNA)
    static constexpr int wave = 64, quads = 1, acc = 4;
    static __device__ __forceinline__ int row(int lane, int r) { return 4 * (lane >> 4) + r; }
#else
    // No matrix instruction: the kernel body stops the wave, and the host side never selects
    // this kernel for such a card.
    static constexpr int wave = 32, quads = 2, acc = 8;
    static __device__ __forceinline__ int row(int lane, int r) { return 8 * (lane >> 4) + r; }
#endif
    /// The lanes that share one output column, each holding a different part of k.
    static constexpr int lanes_per_column = 4 / quads;
    typedef _Float16 frag __attribute__((ext_vector_type(4 * quads)));
    typedef _Float16 half_frag __attribute__((ext_vector_type(2 * quads)));
    typedef float accum __attribute__((ext_vector_type(acc)));

    static __device__ __forceinline__ accum mma(const frag& a, const frag& b, const accum& c) {
#if defined(RDNA4)
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#elif defined(RDNA3)
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif defined(CDNA)
        return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
#else
        (void) a;
        (void) b;
        __builtin_trap();
        return c;
#endif
    }
};

/// Scale-group width in k, and the 16-wide k tiles one group spans.
constexpr int group_size = 128;
constexpr int tiles_per_group = group_size / 16;
/// Output columns per block: one 64-column group of the repack, four 16x16 tiles.
constexpr int block_n = 64;
/// k tiles a wave loads together before it multiplies them.
constexpr int unit_tiles = 4;

/// Threads per block for a tile of `mt` 16-row blocks.
constexpr int block_threads(int mt) { return mt == 1 ? 256 : 128; }

/// What a wave loads for `unit_tiles` k tiles: its weight words, its activation fragments, and
/// the raw scale and zero-point words of the group the unit falls in.
template <int MT>
struct Unit {
    uint4 w[unit_tiles][Core::quads];
    typename Core::half_frag a_lo[unit_tiles][MT], a_hi[unit_tiles][MT];
    uint4 s;
    uint32_t z;
};

template <int MT>
__device__ __forceinline__ void load_unit(Unit<MT>& u, int unit, const _Float16* __restrict__ A,
                                          const uint32_t* __restrict__ B,
                                          const _Float16* __restrict__ S,
                                          const uint32_t* __restrict__ Z, int m, int n, int k,
                                          int g, int row0, int c, int q0) {
    const int kt0 = unit * unit_tiles;
    const int grp = kt0 / tiles_per_group;
    const int c8 = c & 7;
#pragma unroll
    for (int t = 0; t < unit_tiles; ++t) {
        const uint4* w = reinterpret_cast<const uint4*>(B + (size_t) (kt0 + t) * (2 * n) +
                                                        g * 128 + 16 * c8 + 4 * q0);
#pragma unroll
        for (int q = 0; q < Core::quads; ++q) u.w[t][q] = w[q];
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int r = row0 + mt * 16 + c;
        const bool live = r < m;
        const _Float16* a = A + (size_t) (live ? r : 0) * k + kt0 * 16 + 2 * q0;
#pragma unroll
        for (int t = 0; t < unit_tiles; ++t) {
            typename Core::half_frag lo = {}, hi = {};
            if (live) {
                lo = *reinterpret_cast<const typename Core::half_frag*>(a + t * 16);
                hi = *reinterpret_cast<const typename Core::half_frag*>(a + t * 16 + 8);
            }
            u.a_lo[t][mt] = lo;
            u.a_hi[t][mt] = hi;
        }
    }
    u.s = *reinterpret_cast<const uint4*>(S + (size_t) grp * n + g * block_n + c8 * 8);
    u.z = Z[(size_t) grp * (n / 8) + g * 8 + c8];
}

template <int MT>
__device__ __forceinline__ void multiply_unit(const Unit<MT>& u, typename Core::accum (&acc)[MT][4],
                                              int c) {
    const int half = c >> 3;
    // Column `16 j + c` of the group: its scale sits at `8 (c % 8) + 2 j + c / 8` of the
    // group's permuted 64, its zero point at nibble `4 (c / 8) + j` of word `c % 8`.
    h2 zero[4], scale[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint16_t bits = (uint16_t) ((&u.s.x)[j] >> (16 * half));
        const _Float16 s = __builtin_bit_cast(_Float16, bits);
        const _Float16 z = (_Float16) (1024 + ((u.z >> (4 * (4 * half + j))) & 0xf));
        scale[j] = h2{s, s};
        zero[j] = h2{z, z};
    }
#pragma unroll
    for (int t = 0; t < unit_tiles; ++t) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            typename Core::frag b;
#pragma unroll
            for (int q = 0; q < Core::quads; ++q) {
                const uint32_t word = (&u.w[t][q].x)[j] >> (8 * half);
                // A nibble under the exponent of 1024 is the f16 `1024 + q`: subtracting
                // `1024 + z` leaves `q - z` exactly, and one multiply forms the weight.
                h2 lo = __builtin_bit_cast(h2, (word & 0x000f000fu) | 0x64006400u);
                h2 hi = __builtin_bit_cast(h2, ((word >> 4) & 0x000f000fu) | 0x64006400u);
                lo = (lo - zero[j]) * scale[j];
                hi = (hi - zero[j]) * scale[j];
                b[2 * q] = lo.x;
                b[2 * q + 1] = lo.y;
                b[2 * Core::quads + 2 * q] = hi.x;
                b[2 * Core::quads + 2 * q + 1] = hi.y;
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                typename Core::frag a;
#pragma unroll
                for (int i = 0; i < 2 * Core::quads; ++i) {
                    a[i] = u.a_lo[t][mt][i];
                    a[2 * Core::quads + i] = u.a_hi[t][mt][i];
                }
                acc[mt][j] = Core::mma(a, b, acc[mt][j]);
            }
        }
    }
}

/// One block: one `16 MT x 64` output tile over one slice of k.
template <int MT>
__global__ void __launch_bounds__(block_threads(MT))
    w4a16_gemm(const _Float16* __restrict__ A, const uint32_t* __restrict__ B,
               const _Float16* __restrict__ S, const uint32_t* __restrict__ Z,
               _Float16* __restrict__ C, float* __restrict__ c_tmp, int* __restrict__ locks, int m,
               int n, int k, int slices) {
    constexpr int threads = block_threads(MT);
    constexpr int waves = threads / Core::wave;
    constexpr int rows = 16 * MT;
    constexpr int tile = rows * block_n;
    __shared__ float fold[waves][tile];
    __shared__ int last_slice;

    const int mchunks = (m + rows - 1) / rows;
    const int slice = blockIdx.x % slices;
    const int t = blockIdx.x / slices;
    const int row0 = (t % mchunks) * rows;
    const int g = t / mchunks;

    const int wave = threadIdx.x / Core::wave;
    const int lane = threadIdx.x % Core::wave;
    const int c = lane & 15;
    const int q0 = ((lane >> 4) % Core::lanes_per_column) * Core::quads;

    // The slice, in scale groups, then the wave's share of it, in units.
    const int groups = k / group_size;
    const int units_per_group = tiles_per_group / unit_tiles;
    const int s0 = slice * groups / slices * units_per_group;
    const int s1 = (slice + 1) * groups / slices * units_per_group;
    const int u0 = s0 + wave * (s1 - s0) / waves;
    const int u1 = s0 + (wave + 1) * (s1 - s0) / waves;

    typename Core::accum acc[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[mt][j] = typename Core::accum{};

    if (u0 < u1) {
        Unit<MT> cur, next;
        load_unit(cur, u0, A, B, S, Z, m, n, k, g, row0, c, q0);
        for (int u = u0; u < u1; ++u) {
            if (u + 1 < u1) load_unit(next, u + 1, A, B, S, Z, m, n, k, g, row0, c, q0);
            multiply_unit(cur, acc, c);
            cur = next;
        }
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < Core::acc; ++r)
                fold[wave][(mt * 16 + Core::row(lane, r)) * block_n + 16 * j + c] = acc[mt][j][r];
    __syncthreads();

    // The waves' accumulators, added in wave order.
    float out[tile / threads];
#pragma unroll
    for (int i = 0; i < tile / threads; ++i) {
        const int e = i * threads + threadIdx.x;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < waves; ++w) v += fold[w][e];
        out[i] = v;
    }

    if (slices > 1) {
        float* part = c_tmp + (size_t) t * slices * tile;
#pragma unroll
        for (int i = 0; i < tile / threads; ++i)
            part[(size_t) slice * tile + i * threads + threadIdx.x] = out[i];
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last_slice = atomicAdd(&locks[t], 1) == slices - 1;
        __syncthreads();
        if (!last_slice) return;
        __threadfence();
#pragma unroll
        for (int i = 0; i < tile / threads; ++i) {
            const int e = i * threads + threadIdx.x;
            float v = 0.f;
            for (int s = 0; s < slices; ++s)
                v += part[(size_t) s * tile + e];
            out[i] = v;
        }
        if (threadIdx.x == 0) locks[t] = 0;
    }

#pragma unroll
    for (int i = 0; i < tile / threads; ++i) {
        const int e = i * threads + threadIdx.x;
        const int r = row0 + e / block_n;
        if (r < m) C[(size_t) r * n + g * block_n + e % block_n] = (_Float16) out[i];
    }
}

}  // namespace loken_w4a16_amd

// GEMM C[m,n] = A[m,k] . dequant(B) for the repacked AWQ layout.
//   c_tmp: f32 scratch for the partial-tile reduction (>= sms*64*256 floats).
//   locks: int workspace (>= sms entries), ZEROED at allocation; the kernel restores the zeros
//          so it can be reused without clearing it again.
//   sms / max_shared_mem: the device's multiprocessor count and its per-block opt-in shared
//          memory, queried once on the Rust side; -1 asks this function to query them.
// Returns 0 on success, negative on error:
//   -1 bad problem shape, -2 unsupported group size, -5 HIP error.
extern "C" int loken_marlin_awq_f16_gemm(const void* A, const void* B, const void* scales,
                                         const void* zp, void* C, float* c_tmp, int* locks,
                                         int prob_m, int prob_n, int prob_k, int num_groups,
                                         int sms, int max_shared_mem, void* stream) {
    using namespace loken_w4a16_amd;
    (void) max_shared_mem;

    if (prob_m <= 0 || prob_n <= 0 || prob_k <= 0) return -1;
    if (prob_k % 16 != 0 || prob_n % block_n != 0) return -1;
    if (num_groups <= 0 || prob_k % num_groups != 0) return -2;
    if (prob_k / num_groups != group_size) return -2;

    if (sms <= 0) {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess ||
            hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, dev) !=
                hipSuccess) {
            return -5;
        }
    }

    const int mt = prob_m <= 16 ? 1 : 2;
    const int rows = 16 * mt;
    const int tiles = (prob_m + rows - 1) / rows * (prob_n / block_n);

    // Fewer tiles than multiprocessors: cut k so the grid reaches a few blocks per
    // multiprocessor. The partials of `tiles * slices` blocks fit the scratch, whose size is
    // `sms * 64 * 256` floats, as long as that product stays within `sms * 256 / rows`.
    int slices = 1;
    if (tiles < sms) {
        slices = (4 * sms + tiles - 1) / tiles;
        if (slices > num_groups) slices = num_groups;
        const int fit = sms * 256 / rows / tiles;
        if (slices > fit) slices = fit;
        if (slices < 1) slices = 1;
    }

    hipStream_t s = static_cast<hipStream_t>(stream);
    const dim3 grid(tiles * slices);
    const _Float16* a = static_cast<const _Float16*>(A);
    const uint32_t* b = static_cast<const uint32_t*>(B);
    const _Float16* sc = static_cast<const _Float16*>(scales);
    const uint32_t* z = static_cast<const uint32_t*>(zp);
    _Float16* c = static_cast<_Float16*>(C);
    if (mt == 1) {
        w4a16_gemm<1><<<grid, block_threads(1), 0, s>>>(a, b, sc, z, c, c_tmp, locks, prob_m,
                                                         prob_n, prob_k, slices);
    } else {
        w4a16_gemm<2><<<grid, block_threads(2), 0, s>>>(a, b, sc, z, c, c_tmp, locks, prob_m,
                                                         prob_n, prob_k, slices);
    }
    return hipGetLastError() == hipSuccess ? 0 : -5;
}
