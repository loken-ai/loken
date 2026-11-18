// A float weight times a few F32 rows: y[r] = w x[r], summed in F32.
//
//   x: [rows, k]  F32
//   w: [n, k]     BF16, F16 or F32, row-major
//   y: [rows, n]  F32
//
// A group of 32 threads computes `OUTS` consecutive outputs.
// An output's row is walked a tile of `TILE` loads at a time, from a tile set by the output's
// span of `SPAN` outputs, wrapping around to the tiles before it.
// So the groups of a launch read different parts of their rows at a time.
// Within a tile, thread `l` of a group reads the 8 weights of loads l, l + 32 ... of each row.
// It applies them to every activation row, so a weight row is read once for all the rows.
// The group then adds its 32 partial sums.
//
// A row's sum takes the same order whatever the rows, the outputs per group and the groups per
// block of its launch.
// So its result from a launch of several rows equals the result from a launch of its own.
//
// Half weights on a CUDA card with the tensor-core path take it instead: see `MMA_CC`.
//
// `k` must be a multiple of 8, `rows` at most 16 and the pointers 16-byte aligned.
#include <cuda.h>
#include <cuda_fp16.h>
#if !defined(GGML_USE_HIP)
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>
#include <stdint.h>

namespace ll_float_matvec {

constexpr int GROUP = 32;
constexpr int PER_LOAD = 8;
// Loads of a row one tile holds: a multiple of `GROUP`.
constexpr int TILE = 64;
// Consecutive outputs whose walks start at the same tile.
constexpr int SPAN = 64;
// Groups of a block for one row.
constexpr int ONE_ROW_GROUPS = 8;
// The most groups of a block for several rows.
constexpr int MAX_GROUPS = 8;
// The lowest compute capability, as major * 100 + minor * 10, that copies to shared memory
// asynchronously.
#define LL_FLOAT_MATVEC_ASYNC_CC 800
constexpr int ASYNC_COPY_CC = LL_FLOAT_MATVEC_ASYNC_CC;
// The most shared memory a block takes without opting in to more.
constexpr int BLOCK_SHARED_BYTES = 48 * 1024;

__device__ __forceinline__ float from_bf16(uint32_t bits16) {
    return __uint_as_float(bits16 << 16);
}

__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
    for (int o = GROUP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The 8 weights of load `c` of the row at `row`, widened to F32.
struct Bf16 {
    static constexpr int BYTES = 2;
    __device__ __forceinline__ static void load(const void* row, int c, float* wv) {
        const uint4 p = reinterpret_cast<const uint4*>(row)[c];
        const uint32_t words[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wv[2 * i] = from_bf16(words[i] & 0xffffu);
            wv[2 * i + 1] = from_bf16(words[i] >> 16);
        }
    }
};

struct F16 {
    static constexpr int BYTES = 2;
    __device__ __forceinline__ static void load(const void* row, int c, float* wv) {
        const uint4 p = reinterpret_cast<const uint4*>(row)[c];
        const __half2* h = reinterpret_cast<const __half2*>(&p);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h[i]);
            wv[2 * i] = f.x;
            wv[2 * i + 1] = f.y;
        }
    }
};

struct F32 {
    static constexpr int BYTES = 4;
    __device__ __forceinline__ static void load(const void* row, int c, float* wv) {
        const float4* p = reinterpret_cast<const float4*>(row) + 2 * c;
        const float4 a = p[0];
        const float4 b = p[1];
        wv[0] = a.x;
        wv[1] = a.y;
        wv[2] = a.z;
        wv[3] = a.w;
        wv[4] = b.x;
        wv[5] = b.y;
        wv[6] = b.z;
        wv[7] = b.w;
    }
};

// Load `c` of the weight rows of the outputs from `first` applied to every row.
// The 8 values of load `c` of activation row `r` sit at `xr + r * stride`.
// An output's sum for a row takes its fused multiply-adds in the order of the 8 weights.
template <typename W, int ROWS, int OUTS>
__device__ __forceinline__ void apply_load(const char* base, size_t row_bytes, int first, int n,
                                           int c, const float* xr, size_t stride, int rows,
                                           float (&acc)[OUTS][ROWS]) {
    float wv[OUTS][PER_LOAD];
#pragma unroll
    for (int o = 0; o < OUTS; ++o) {
        if (first + o < n) W::load(base + (size_t)(first + o) * row_bytes, c, wv[o]);
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (r < rows) {
            const float4* p = reinterpret_cast<const float4*>(xr + (size_t)r * stride);
            const float4 a = p[0];
            const float4 b = p[1];
            const float xv[PER_LOAD] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int o = 0; o < OUTS; ++o) {
#pragma unroll
                for (int i = 0; i < PER_LOAD; ++i)
                    acc[o][r] = __fmaf_rn(wv[o][i], xv[i], acc[o][r]);
            }
        }
    }
}

// The group's 32 partial sums of each output and row added, and written by its first thread.
template <int ROWS, int OUTS>
__device__ __forceinline__ void write_sums(float (&acc)[OUTS][ROWS], float* y, int lane,
                                           int first, int n, int rows) {
#pragma unroll
    for (int o = 0; o < OUTS; ++o) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float s = group_sum(acc[o][r]);
            if (lane == 0 && r < rows && first + o < n) y[(size_t)r * n + first + o] = s;
        }
    }
}

// The tile the walk of the output `output` starts at, among `tiles`.
__device__ __forceinline__ int first_tile(int output, int tiles) {
    return (output / SPAN) % tiles;
}

// One row read from memory by each thread.
template <typename W, int OUTS>
__global__ void float_matvec_kernel(const float* __restrict__ x, const void* __restrict__ w,
                                    float* __restrict__ y, int k, int n) {
    const int lane = threadIdx.x & (GROUP - 1);
    const int first = (blockIdx.x * (blockDim.x / GROUP) + threadIdx.x / GROUP) * OUTS;
    if (first >= n) return;
    const int loads = k / PER_LOAD;
    const int tiles = (loads + TILE - 1) / TILE;
    const int start = first_tile(first, tiles);
    const char* base = reinterpret_cast<const char*>(w);
    const size_t row_bytes = (size_t)k * W::BYTES;
    float acc[OUTS][1];
#pragma unroll
    for (int o = 0; o < OUTS; ++o) acc[o][0] = 0.0f;
    for (int t = 0; t < tiles; ++t) {
        const int c0 = ((t + start) % tiles) * TILE;
        const int end = loads - c0 < TILE ? loads : c0 + TILE;
        for (int c = c0 + lane; c < end; c += GROUP)
            apply_load<W, 1, OUTS>(base, row_bytes, first, n, c, x + (size_t)c * PER_LOAD, k, 1,
                                   acc);
    }
    write_sums<1, OUTS>(acc, y, lane, first, n, 1);
}

// Several rows staged in shared memory a tile at a time, read there by every group of the block.
// The outputs of a block lie in one span, so the block walks the tiles in their order.
template <typename W, int ROWS, int OUTS>
__global__ void float_matvec_staged_kernel(const float* __restrict__ x,
                                           const void* __restrict__ w, float* __restrict__ y,
                                           int k, int n) {
    __shared__ __align__(16) float xs[ROWS * TILE * PER_LOAD];
    const int lane = threadIdx.x & (GROUP - 1);
    const int block_first = blockIdx.x * (blockDim.x / GROUP) * OUTS;
    const int first = block_first + (threadIdx.x / GROUP) * OUTS;
    const bool active = first < n;
    const int loads = k / PER_LOAD;
    const int tiles = (loads + TILE - 1) / TILE;
    const int start = first_tile(block_first, tiles);
    const char* base = reinterpret_cast<const char*>(w);
    const size_t row_bytes = (size_t)k * W::BYTES;
    float acc[OUTS][ROWS];
#pragma unroll
    for (int o = 0; o < OUTS; ++o)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[o][r] = 0.0f;
    for (int t = 0; t < tiles; ++t) {
        const int c0 = ((t + start) % tiles) * TILE;
        const int len = loads - c0 < TILE ? loads - c0 : TILE;
        // 2 float4 per load.
        for (int i = threadIdx.x; i < ROWS * len * 2; i += blockDim.x) {
            const int r = i / (len * 2);
            const int j = i - r * len * 2;
            reinterpret_cast<float4*>(xs + r * TILE * PER_LOAD)[j] =
                reinterpret_cast<const float4*>(x + (size_t)r * k + (size_t)c0 * PER_LOAD)[j];
        }
        __syncthreads();
        if (active) {
            for (int c = c0 + lane; c < c0 + len; c += GROUP)
                apply_load<W, ROWS, OUTS>(base, row_bytes, first, n, c,
                                          xs + (c - c0) * PER_LOAD, TILE * PER_LOAD, ROWS, acc);
        }
        __syncthreads();
    }
    if (active) write_sums<ROWS, OUTS>(acc, y, lane, first, n, ROWS);
}

// 16 bytes from global to shared memory, asynchronously where the card copies so.
__device__ __forceinline__ void copy16_async(float* dst, const float* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_ASYNC_CC
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
#else
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
#endif
}

// Close the copies issued so far into one group.
__device__ __forceinline__ void commit_copies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_ASYNC_CC
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until at most `PENDING` groups of copies remain in flight.
template <int PENDING>
__device__ __forceinline__ void wait_copies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_ASYNC_CC
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
#endif
}

// The staged kernel with 2 buffers: the next tile of the rows is copied while the current one
// is read.
// Each output takes its loads in the same order as in the staged kernel.
template <typename W, int ROWS, int OUTS>
__global__ void float_matvec_async_kernel(const float* __restrict__ x,
                                          const void* __restrict__ w, float* __restrict__ y,
                                          int k, int n) {
    extern __shared__ __align__(16) float buffers[];
    constexpr int BUF = ROWS * TILE * PER_LOAD;
    const int lane = threadIdx.x & (GROUP - 1);
    const int block_first = blockIdx.x * (blockDim.x / GROUP) * OUTS;
    const int first = block_first + (threadIdx.x / GROUP) * OUTS;
    const bool active = first < n;
    const int loads = k / PER_LOAD;
    const int tiles = (loads + TILE - 1) / TILE;
    const int start = first_tile(block_first, tiles);
    const char* base = reinterpret_cast<const char*>(w);
    const size_t row_bytes = (size_t)k * W::BYTES;
    float acc[OUTS][ROWS];
#pragma unroll
    for (int o = 0; o < OUTS; ++o)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[o][r] = 0.0f;
    // The rows of tile `t` into `buf`, 2 float4 per load.
    auto stage = [&](int t, float* buf) {
        const int c0 = ((t + start) % tiles) * TILE;
        const int len = loads - c0 < TILE ? loads - c0 : TILE;
        for (int i = threadIdx.x; i < ROWS * len * 2; i += blockDim.x) {
            const int r = i / (len * 2);
            const int j = i - r * len * 2;
            copy16_async(buf + r * TILE * PER_LOAD + 4 * j,
                         x + (size_t)r * k + (size_t)c0 * PER_LOAD + 4 * j);
        }
        commit_copies();
    };
    stage(0, buffers);
    for (int t = 0; t < tiles; ++t) {
        float* cur = buffers + (t & 1) * BUF;
        const int c0 = ((t + start) % tiles) * TILE;
        const int len = loads - c0 < TILE ? loads - c0 : TILE;
        if (t + 1 < tiles) {
            stage(t + 1, buffers + ((t + 1) & 1) * BUF);
            wait_copies<1>();
        } else {
            wait_copies<0>();
        }
        __syncthreads();
        if (active) {
            for (int c = c0 + lane; c < c0 + len; c += GROUP)
                apply_load<W, ROWS, OUTS>(base, row_bytes, first, n, c,
                                          cur + (c - c0) * PER_LOAD, TILE * PER_LOAD, ROWS, acc);
        }
        __syncthreads();
    }
    if (active) write_sums<ROWS, OUTS>(acc, y, lane, first, n, ROWS);
}

// `ROWS` rows: 2 outputs a group up to 4 rows, 4 above.
// The groups of a block halve from `MAX_GROUPS` while the blocks number fewer than the card's
// multiprocessors, down to 2.
// A CUDA card of compute capability `cc` from `ASYNC_COPY_CC` takes the kernel with 2 buffers
// where they fit the shared memory of a block.
template <typename W, int ROWS>
void launch_rows(const float* x, const void* w, float* y, int k, int n, int sms, int cc,
                 cudaStream_t s) {
    constexpr int OUTS = ROWS <= 4 ? 2 : 4;
    static_assert(SPAN % (MAX_GROUPS * OUTS) == 0, "a block lies in one span");
    int groups = MAX_GROUPS;
    while (groups > 2 && (n + groups * OUTS - 1) / (groups * OUTS) < sms) groups /= 2;
    const int blocks = (n + groups * OUTS - 1) / (groups * OUTS);
    constexpr int buffers_bytes = 2 * ROWS * TILE * PER_LOAD * (int)sizeof(float);
#if defined(GGML_USE_HIP)
    const bool async = false;
#else
    const bool async = cc >= ASYNC_COPY_CC && buffers_bytes <= BLOCK_SHARED_BYTES;
#endif
    if (async) {
        float_matvec_async_kernel<W, ROWS, OUTS>
            <<<blocks, groups * GROUP, buffers_bytes, s>>>(x, w, y, k, n);
    } else {
        float_matvec_staged_kernel<W, ROWS, OUTS><<<blocks, groups * GROUP, 0, s>>>(x, w, y, k, n);
    }
}

// The tensor-core path for half weights, on CUDA cards of compute capability `MMA_CC` and up.
//
// The weights of 8 rows are the B operand of an m16n8k16 MMA.
// Each activation row is split in 2 parts of the weights' half type: its nearest value, and
// what that misses.
// The high parts of activation rows 0 to 7 are rows 0 to 7 of the A operand of an MMA, and
// their low parts are its rows 8 to 15.
// Activation rows 8 to 15 take a second MMA laid out the same way.
// An activation row's result is its high part's F32 sum plus its low part's.
// A block of `MMA_SEGMENTS` warps owns 1, 2 or 4 groups of 8 weight rows, see `MMA_VARIANTS`.
// Warp `s` sums segment `s` of the row length, a step of 32 weights at a time.
// The warps of odd runs of `MMA_MOST_TILES` 8-row groups walk their steps from the middle of
// the segment, wrapping around.
// Lane (g, t) loads the 8 weights at 8t of the step of weight row g of each group.
// It reads the activations of rows g and g + 8 at the same 8 positions.
// The first 4 of them feed one MMA and the last 4 a second one.
// The block then adds the partial sums of its warps in segment order.
//
// An MMA output depends only on its row of A and its weight column.
// An activation row takes MMA rows g and g + 8 of its own MMA, whatever the rows beside it.
// The walk depends on the weight's shape alone.
// So a row's result from a launch of several rows equals the result from a launch of its own.
#define LL_FLOAT_MATVEC_MMA_CC 800
constexpr int MMA_CC = LL_FLOAT_MATVEC_MMA_CC;
// Weight rows of a block: the columns of one MMA.
constexpr int MMA_COLS = 8;
// Warps of a block, one segment of the row length each.
constexpr int MMA_SEGMENTS = 8;
// Weights of a step: 2 MMAs of 16.
constexpr int MMA_STEP = 32;
// Steps a warp loads before it multiplies them.
constexpr int MMA_AHEAD = 4;
// The most groups of 8 weight rows a warp owns.
constexpr int MMA_MOST_TILES = 4;
// The most activation rows: the rows of one MMA.
constexpr int MMA_MAX_ROWS = 16;

#if !defined(GGML_USE_HIP)
struct MmaF16 {
    // `a` and `b` as F16: their nearest values in `hi`, what those miss in `lo`.
    __device__ __forceinline__ static void split(float a, float b, uint32_t& hi, uint32_t& lo) {
        const __half2 h = __floats2half2_rn(a, b);
        const float2 back = __half22float2(h);
        const __half2 l = __floats2half2_rn(a - back.x, b - back.y);
        hi = *reinterpret_cast<const uint32_t*>(&h);
        lo = *reinterpret_cast<const uint32_t*>(&l);
    }
    __device__ __forceinline__ static void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                               uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_MMA_CC
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
    }
};

struct MmaBf16 {
    // `a` and `b` as BF16: their nearest values in `hi`, what those miss in `lo`.
    __device__ __forceinline__ static void split(float a, float b, uint32_t& hi, uint32_t& lo) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
        const float2 back = __bfloat1622float2(h);
        const __nv_bfloat162 l = __floats2bfloat162_rn(a - back.x, b - back.y);
        hi = *reinterpret_cast<const uint32_t*>(&h);
        lo = *reinterpret_cast<const uint32_t*>(&l);
    }
    __device__ __forceinline__ static void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                               uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_MMA_CC
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
    }
};

// A cache policy that evicts the lines it tags first, or last with `last`.
__device__ __forceinline__ uint64_t cache_policy(bool last) {
    uint64_t p = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_MMA_CC
    if (last)
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
    else
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
#endif
    return p;
}

// 8 weights read once, tagged with `policy`.
__device__ __forceinline__ uint4 load_streamed(const uint16_t* p, uint64_t policy) {
    uint4 r = make_uint4(0, 0, 0, 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_MMA_CC
    asm volatile("ld.global.nc.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p), "l"(policy));
#endif
    return r;
}

// 2 floats written with `policy`.
__device__ __forceinline__ void store_kept(float* p, float a, float b, uint64_t policy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= LL_FLOAT_MATVEC_MMA_CC
    asm volatile("st.global.L2::cache_hint.v2.f32 [%0], {%1,%2}, %3;\n" ::"l"(p), "f"(a), "f"(b),
                 "l"(policy));
#endif
}

// `HIGH`: activation rows 8 to 15 take part.
// `TILES`: groups of 8 weight rows each warp owns, all read against the same activation reads.
// The weights stream through the second-level cache ahead of everything else.
// The outputs stay there after them.
template <typename T, bool HIGH, int TILES>
__device__ __forceinline__ void mma_matvec(const float* __restrict__ x,
                                           const uint16_t* __restrict__ w, float* __restrict__ y,
                                           int k, int n, int rows) {
    __shared__ float part[MMA_SEGMENTS][GROUP][4 * TILES];
    const int lane = threadIdx.x & (GROUP - 1);
    const int s = threadIdx.x / GROUP;
    const int g = lane >> 2, t = lane & 3;
    const int steps = (k + MMA_STEP - 1) / MMA_STEP;
    const int first = (int)((long long)s * steps / MMA_SEGMENTS);
    const int len = (int)((long long)(s + 1) * steps / MMA_SEGMENTS) - first;
    // Odd runs of `MMA_MOST_TILES` 8-row groups walk their segments from the middle, whatever
    // the groups a block owns, so every variant takes a row's steps in the same order.
    const int start = ((blockIdx.x * TILES / MMA_MOST_TILES) & 1) * (len / 2);
    const bool low = g < rows;
    const bool high = HIGH && g + 8 < rows;
    const uint16_t* wr[TILES];
    bool row_in[TILES];
#pragma unroll
    for (int b = 0; b < TILES; ++b) {
        const int row = (blockIdx.x * TILES + b) * MMA_COLS + g;
        row_in[b] = row < n;
        wr[b] = w + (size_t)(row_in[b] ? row : 0) * k;
    }
    const float* x_low = x + (size_t)g * k;
    const float* x_high = x + (size_t)(g + 8) * k;
    const uint64_t streamed = cache_policy(false);
    // Sums of activation rows 0 to 7 in `c`, of rows 8 to 15 in `d`, for each group of rows.
    // In each, MMA rows 0 to 7 hold the high parts' sums and MMA rows 8 to 15 the low parts'.
    float c[TILES][4], d[TILES][4];
#pragma unroll
    for (int b = 0; b < TILES; ++b)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[b][i] = d[b][i] = 0.0f;
    for (int i0 = 0; i0 < len; i0 += MMA_AHEAD) {
        uint4 wv[MMA_AHEAD][TILES];
        float4 xl[MMA_AHEAD][2], xh[MMA_AHEAD][2];
#pragma unroll
        for (int u = 0; u < MMA_AHEAD; ++u) {
            int i = i0 + u + start;
            if (i >= len) i -= len;
            const int at = (first + i) * MMA_STEP + t * PER_LOAD;
            // A lane past the row length holds zeros, as do the lanes of absent rows.
            const bool in = i0 + u < len && at < k;
            const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int b = 0; b < TILES; ++b)
                wv[u][b] = in && row_in[b] ? load_streamed(wr[b] + at, streamed)
                                           : make_uint4(0, 0, 0, 0);
            xl[u][0] = in && low ? reinterpret_cast<const float4*>(x_low + at)[0] : z;
            xl[u][1] = in && low ? reinterpret_cast<const float4*>(x_low + at)[1] : z;
            if (HIGH) {
                xh[u][0] = in && high ? reinterpret_cast<const float4*>(x_high + at)[0] : z;
                xh[u][1] = in && high ? reinterpret_cast<const float4*>(x_high + at)[1] : z;
            }
        }
#pragma unroll
        for (int u = 0; u < MMA_AHEAD; ++u) {
            if (i0 + u < len) {
                // A registers of an MMA: the high part of row g at the 2 positions of a half,
                // and its low part as MMA row g + 8.
                uint32_t a0[4], a1[4];
                T::split(xl[u][0].x, xl[u][0].y, a0[0], a0[1]);
                T::split(xl[u][0].z, xl[u][0].w, a0[2], a0[3]);
                T::split(xl[u][1].x, xl[u][1].y, a1[0], a1[1]);
                T::split(xl[u][1].z, xl[u][1].w, a1[2], a1[3]);
#pragma unroll
                for (int b = 0; b < TILES; ++b) {
                    T::mma(c[b], a0, wv[u][b].x, wv[u][b].y);
                    T::mma(c[b], a1, wv[u][b].z, wv[u][b].w);
                }
                if (HIGH) {
                    T::split(xh[u][0].x, xh[u][0].y, a0[0], a0[1]);
                    T::split(xh[u][0].z, xh[u][0].w, a0[2], a0[3]);
                    T::split(xh[u][1].x, xh[u][1].y, a1[0], a1[1]);
                    T::split(xh[u][1].z, xh[u][1].w, a1[2], a1[3]);
#pragma unroll
                    for (int b = 0; b < TILES; ++b) {
                        T::mma(d[b], a0, wv[u][b].x, wv[u][b].y);
                        T::mma(d[b], a1, wv[u][b].z, wv[u][b].w);
                    }
                }
            }
        }
    }
    // Lane (g, t) holds weight rows 2t and 2t + 1 of activation rows g and g + 8: the high
    // part's sum plus the low part's.
#pragma unroll
    for (int b = 0; b < TILES; ++b) {
        part[s][lane][4 * b + 0] = c[b][0] + c[b][2];
        part[s][lane][4 * b + 1] = c[b][1] + c[b][3];
        part[s][lane][4 * b + 2] = d[b][0] + d[b][2];
        part[s][lane][4 * b + 3] = d[b][1] + d[b][3];
    }
    __syncthreads();
    if (s != 0) return;
#pragma unroll
    for (int b = 0; b < TILES; ++b) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = part[0][lane][4 * b + i];
#pragma unroll
            for (int j = 1; j < MMA_SEGMENTS; ++j) v[i] += part[j][lane][4 * b + i];
        }
        const int o = (blockIdx.x * TILES + b) * MMA_COLS + 2 * t;
        if ((n & 1) == 0) {
            const uint64_t kept = cache_policy(true);
            if (low && o < n) store_kept(y + (size_t)g * n + o, v[0], v[1], kept);
            if (high && o < n) store_kept(y + (size_t)(g + 8) * n + o, v[2], v[3], kept);
            continue;
        }
        if (o >= n) continue;
        if (low) y[(size_t)g * n + o] = v[0];
        if (low && o + 1 < n) y[(size_t)g * n + o + 1] = v[1];
        if (high) y[(size_t)(g + 8) * n + o] = v[2];
        if (high && o + 1 < n) y[(size_t)(g + 8) * n + o + 1] = v[3];
    }
}

// The product compiled for the default register budget of a block.
template <typename T, bool HIGH, int TILES>
__global__ void __launch_bounds__(MMA_SEGMENTS * GROUP)
    float_matvec_mma_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                            float* __restrict__ y, int k, int n, int rows) {
    mma_matvec<T, HIGH, TILES>(x, w, y, k, n, rows);
}

// The same product compiled for the register budget of one block a multiprocessor.
template <typename T, bool HIGH, int TILES>
__global__ void __launch_bounds__(MMA_SEGMENTS * GROUP, 1)
    float_matvec_mma_one_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                float* __restrict__ y, int k, int n, int rows) {
    mma_matvec<T, HIGH, TILES>(x, w, y, k, n, rows);
}
#endif

// Whether a card of compute capability `cc` takes half weights through the tensor cores.
inline bool takes_mma(int cc) {
#if defined(GGML_USE_HIP)
    (void)cc;
    return false;
#else
    return cc >= MMA_CC;
#endif
}

// The variants of the tensor-core path.
// Variant 0 takes 1 group of 8 weight rows a warp and 1 group of 8 activation rows, under the
// default register budget, up to 8 rows.
// Variants 1, 2 and 3 take 1, 2 and 4 groups of 8 weight rows a warp and 2 groups of 8
// activation rows, under the budget of one block a multiprocessor.
// Any launch of more than 8 rows takes the second form.
// A group without rows adds nothing, and every variant takes a row's steps, segments and parts
// in the same order, so all give the same bits.
constexpr int MMA_VARIANTS = 4;

// The groups of 8 weight rows a warp of variant `v` takes.
inline int variant_tiles(int v) {
    return v <= 0 ? 1 : 1 << (v - 1);
}

template <typename T, int TILES>
void launch_tiles(const float* x, const uint16_t* w, float* y, int k, int n, int rows,
                  int build, cudaStream_t s) {
#if !defined(GGML_USE_HIP)
    const int blocks = (n + TILES * MMA_COLS - 1) / (TILES * MMA_COLS);
    const int threads = MMA_SEGMENTS * GROUP;
    if (rows > 8 || build == 1)
        float_matvec_mma_one_kernel<T, true, TILES><<<blocks, threads, 0, s>>>(x, w, y, k, n,
                                                                               rows);
    else
        float_matvec_mma_kernel<T, false, TILES><<<blocks, threads, 0, s>>>(x, w, y, k, n,
                                                                            rows);
#else
    (void)x, (void)w, (void)y, (void)k, (void)n, (void)rows, (void)build, (void)s;
#endif
}

// The tensor-core path for `rows` from 1 to `MMA_MAX_ROWS`, through variant `variant`.
template <typename T>
void launch_mma(const void* x, const void* w, void* y, int k, int n, int rows, int variant,
                cudaStream_t s) {
    const float* xf = (const float*)x;
    const uint16_t* wh = (const uint16_t*)w;
    float* yf = (float*)y;
    const int build = variant == 0 ? 0 : 1;
    switch (variant_tiles(variant)) {
        case 4: launch_tiles<T, 4>(xf, wh, yf, k, n, rows, build, s); break;
        case 2: launch_tiles<T, 2>(xf, wh, yf, k, n, rows, build, s); break;
        default: launch_tiles<T, 1>(xf, wh, yf, k, n, rows, build, s); break;
    }
}

template <typename W>
void dispatch(const void* x, const void* w, void* y, int k, int n, int rows, int sms, int cc,
              cudaStream_t s) {
    const float* xf = (const float*)x;
    float* yf = (float*)y;
    switch (rows) {
        case 1: {
            const int blocks = (n + ONE_ROW_GROUPS - 1) / ONE_ROW_GROUPS;
            float_matvec_kernel<W, 1><<<blocks, ONE_ROW_GROUPS * GROUP, 0, s>>>(xf, w, yf, k, n);
            break;
        }
        case 2: launch_rows<W, 2>(xf, w, yf, k, n, sms, cc, s); break;
        case 3: launch_rows<W, 3>(xf, w, yf, k, n, sms, cc, s); break;
        case 4: launch_rows<W, 4>(xf, w, yf, k, n, sms, cc, s); break;
        case 5: launch_rows<W, 5>(xf, w, yf, k, n, sms, cc, s); break;
        case 6: launch_rows<W, 6>(xf, w, yf, k, n, sms, cc, s); break;
        case 7: launch_rows<W, 7>(xf, w, yf, k, n, sms, cc, s); break;
        case 8: launch_rows<W, 8>(xf, w, yf, k, n, sms, cc, s); break;
        case 9: launch_rows<W, 9>(xf, w, yf, k, n, sms, cc, s); break;
        case 10: launch_rows<W, 10>(xf, w, yf, k, n, sms, cc, s); break;
        case 11: launch_rows<W, 11>(xf, w, yf, k, n, sms, cc, s); break;
        case 12: launch_rows<W, 12>(xf, w, yf, k, n, sms, cc, s); break;
        case 13: launch_rows<W, 13>(xf, w, yf, k, n, sms, cc, s); break;
        case 14: launch_rows<W, 14>(xf, w, yf, k, n, sms, cc, s); break;
        case 15: launch_rows<W, 15>(xf, w, yf, k, n, sms, cc, s); break;
        default: launch_rows<W, 16>(xf, w, yf, k, n, sms, cc, s); break;
    }
}

}  // namespace ll_float_matvec

// The caller checks the shape: `k` a multiple of 8, `rows` from 1 to 16.
// `sms` is the card's multiprocessor count.
// `cc` is its compute capability as major * 100 + minor * 10.
// A card that takes half weights through the tensor cores splits each row in 2 parts of the
// weight's type.
// `variant` picks a variant of that path, see `MMA_VARIANTS`, or the CUDA-core kernel below 0.
extern "C" void loken_bf16_matvec(const void* x, const void* w, void* y, int k, int n, int rows,
                                  int sms, int cc, int variant, cudaStream_t stream) {
#if !defined(GGML_USE_HIP)
    if (variant >= 0 && ll_float_matvec::takes_mma(cc) &&
        rows <= ll_float_matvec::MMA_MAX_ROWS) {
        ll_float_matvec::launch_mma<ll_float_matvec::MmaBf16>(x, w, y, k, n, rows, variant,
                                                              stream);
        return;
    }
#endif
    (void)variant;
    ll_float_matvec::dispatch<ll_float_matvec::Bf16>(x, w, y, k, n, rows, sms, cc, stream);
}

extern "C" void loken_f16_matvec(const void* x, const void* w, void* y, int k, int n, int rows,
                                 int sms, int cc, int variant, cudaStream_t stream) {
#if !defined(GGML_USE_HIP)
    if (variant >= 0 && ll_float_matvec::takes_mma(cc) &&
        rows <= ll_float_matvec::MMA_MAX_ROWS) {
        ll_float_matvec::launch_mma<ll_float_matvec::MmaF16>(x, w, y, k, n, rows, variant,
                                                              stream);
        return;
    }
#endif
    (void)variant;
    ll_float_matvec::dispatch<ll_float_matvec::F16>(x, w, y, k, n, rows, sms, cc, stream);
}

// The variants of a half weight's product on a card of compute capability `cc`.
// A card with more than 1 takes half weights through the tensor cores.
extern "C" int loken_float_matvec_variants(int cc) {
    return ll_float_matvec::takes_mma(cc) ? ll_float_matvec::MMA_VARIANTS : 1;
}

// The groups of 8 weight rows a warp of the tensor-core variant `variant` takes.
extern "C" int loken_float_matvec_tiles(int variant) {
    return ll_float_matvec::variant_tiles(variant);
}

extern "C" void loken_f32_matvec(const void* x, const void* w, void* y, int k, int n, int rows,
                                 int sms, int cc, cudaStream_t stream) {
    ll_float_matvec::dispatch<ll_float_matvec::F32>(x, w, y, k, n, rows, sms, cc, stream);
}
