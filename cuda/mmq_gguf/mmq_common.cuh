// What the MMQ kernels need beyond the block formats themselves: the architecture
// capabilities they branch on, the tile geometry, and the fast integer division the inner
// loop leans on.
//
// The formats are NOT here. `gguf.cuh` states them once for the whole project - the twelve
// block layouts, their dot products, their dequantisers - and this header includes it.
#pragma once

#include "../moe/gguf.cuh"

#include <cstdint>
#include <cstdio>
#include <climits>

#include "cuda_fp16.h"
#include "cuda_bf16.h"

// ============================================================
// Basic macros
// ============================================================

#define MATRIX_ROW_PADDING 512
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define GGML_CUDA_MAX_DEVICES 16

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

// The formats these kernels dispatch on: the id a GGUF file stores, and the three numbers a
// kernel asks of each - values to a block, how many of them a stored item covers, and how many
// items fit an int.
//
// Only these: an id this build cannot serve never reaches a kernel, because the Rust side
// refuses it while reading the file and names it there. One row per format, because the id and
// the geometry were two lists over the same formats and a row could be in one and not the
// other. f16 is not blocked - two of them fit an int, which is the only one of the three
// numbers that means anything for it.
#define GGUF_FORMATS(X)                        \
    X(F16,   1,  1,     1,     2)              \
    X(Q4_0,  2,  QK4_0, QR4_0, QI4_0)          \
    X(Q4_1,  3,  QK4_1, QR4_1, QI4_1)          \
    X(Q5_0,  6,  QK5_0, QR5_0, QI5_0)          \
    X(Q5_1,  7,  QK5_1, QR5_1, QI5_1)          \
    X(Q8_0,  8,  QK8_0, QR8_0, QI8_0)          \
    X(Q8_1,  9,  QK8_1, QR8_1, QI8_1)          \
    X(Q2_K, 10,  QK_K,  QR2_K, QI2_K)          \
    X(Q3_K, 11,  QK_K,  QR3_K, QI3_K)          \
    X(Q4_K, 12,  QK_K,  QR4_K, QI4_K)          \
    X(Q5_K, 13,  QK_K,  QR5_K, QI5_K)          \
    X(Q6_K, 14,  QK_K,  QR6_K, QI6_K)          \
    X(Q8_K, 15,  QK_K,  1,     1)

enum ggml_type {
#define GGUF_TYPE_ID(name, id, qk_v, qr_v, qi_v) GGML_TYPE_##name = id,
    GGUF_FORMATS(GGUF_TYPE_ID)
#undef GGUF_TYPE_ID
};

/// Asking for a trait of a format nothing serves is a compile error, which is the answer
/// wanted: a kernel is instantiated for a format or it is not built at all.
template <ggml_type type>
struct ggml_cuda_type_traits;

#define GGUF_TYPE_TRAITS(name, id, qk_v, qr_v, qi_v)    \
    template <>                                         \
    struct ggml_cuda_type_traits<GGML_TYPE_##name> {    \
        static constexpr int qk = qk_v;                 \
        static constexpr int qr = qr_v;                 \
        static constexpr int qi = qi_v;                 \
    };
GGUF_FORMATS(GGUF_TYPE_TRAITS)
#undef GGUF_TYPE_TRAITS

// The block geometry comes from `gguf.cuh`, included above: it states each format once - the
// values in a block, the ratio, the ints a dot product reads - and every one of those names was
// spelled a second time here. Two spellings of a constant agree until one of them is corrected.

// ============================================================
// Architecture detection
// ============================================================

#define GGML_CUDA_CC_PASCAL       600
#define GGML_CUDA_CC_VOLTA        700
#define GGML_CUDA_CC_TURING       750
#define GGML_CUDA_CC_AMPERE       800
#define GGML_CUDA_CC_BLACKWELL    1200
#define GGML_CUDA_CC_RUBIN        1300

#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc) (cc < GGML_CUDA_CC_OFFSET_MTHREADS)

/// The newest architecture this build has code for that the card can still run.
///
/// A kernel asks this rather than `__CUDA_ARCH__` because a card newer than anything compiled
/// runs the newest code object there is, and a capability check against its own generation
/// would then be answering about code that is not in the binary. `-1` means the card is older
/// than everything compiled, which is a card this build cannot serve.
#ifdef __CUDA_ARCH_LIST__
constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    int best = 0;
    for (const int compiled : {__CUDA_ARCH_LIST__}) {
        if (compiled <= arch && compiled > best) {
            best = compiled;
        }
    }
    return best == 0 ? -1 : best;
}
#else
/// Outside a device compilation there is no list, and the answer is the card itself.
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

// FP16 availability
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif

// The one NVIDIA tensor-core generation these kernels branch on. Turing is where the integer
// `mma.sync` this matmul is built on appears; the generations above it were named here too and
// nothing asked about them, because what a kernel needs to know is whether the instruction
// exists, and `compiled_at_least` answers the rest.
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif

// AMD tensor-core availability.
//
// Forty-six places in these kernels ask `#if defined(AMD_MFMA_AVAILABLE)` or its WMMA twin, and
// until now nothing anywhere defined either. Under nvcc that is correct - those arms are not
// for this compiler - but it also meant a HIP build would skip every one of them and produce a
// binary with no AMD tensor-core path at all, silently, which is not what forty-six guarded
// arms are for.
//
// HIP names its target with `__gfx*__` rather than a number, so that is what decides here. The
// two instructions are not interchangeable and neither is universal: CDNA has MFMA and no WMMA,
// RDNA3 and later have WMMA and no MFMA, and RDNA1 and RDNA2 have neither - which is why the
// non-tensor arm below stays and is not an oversight.
#if defined(__HIP_DEVICE_COMPILE__)
  #if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
      defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
    #define AMD_MFMA_AVAILABLE
  #endif
  #if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
      defined(__gfx1200__) || defined(__gfx1201__)
    #define AMD_WMMA_AVAILABLE
  #endif
#endif

/// Whether this build has code the card can run, at or above a generation.
///
/// The question is about the BINARY and not the card: a card newer than anything compiled runs
/// the newest object there is, so what matters is the newest compiled generation the card can
/// still run. `below` bounds the answer for a capability a later generation takes away.
static bool compiled_at_least(const int cc, const int generation, const int below = INT_MAX) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }
    const int compiled = ggml_cuda_highest_compiled_arch(cc);
    return compiled >= generation && compiled < below;
}

/// This one asks about the CARD rather than the binary: half-precision tensor cores are a
/// property of the silicon, and the fallback for a card without them is not another kernel but
/// another dtype.
static bool fp16_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA;
}

/// AMD's matrix cores, for the day this builds under a HIP toolchain. Under nvcc there are none
/// - not "not yet compiled for", none - so these answer from the vendor and not from the list.
static bool amd_mfma_available(const int /*cc*/) { return false; }
static bool amd_wmma_available(const int /*cc*/) { return false; }

static bool turing_mma_available(const int cc) {
    return compiled_at_least(cc, GGML_CUDA_CC_TURING);
}

static bool blackwell_mma_available(const int cc) {
    return compiled_at_least(cc, GGML_CUDA_CC_BLACKWELL, GGML_CUDA_CC_RUBIN);
}

// ============================================================
// Device helpers
// ============================================================

/// The lanes a warp-wide exchange spans, as a constant expression. The shuffles state it
/// directly as `WARP_SIZE`; the tile geometry and the launch bounds ask through this because
/// they are computed at compile time.
static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return WARP_SIZE;
}

/// The body of a dispatch arm the running architecture has no instruction for.
///
/// Such an arm is compiled regardless - it is one branch of a template another architecture
/// takes - so it needs a body, and the body has to be one that cannot return a wrong answer.
/// Reaching it means the arm was selected for a card whose instruction set does not hold it:
/// no input causes that and no retry fixes it, so it names the site and stops the thread.
[[noreturn]] static __device__ void no_device_code(const char * source_file, const int source_line,
                                                   const char * kernel, const int running_arch,
                                                   const char * built_for) {
    printf("%s:%d: %s has no instructions for architecture %d; this build holds {%s}\n",
           source_file, source_line, kernel, running_arch, built_for);
    __trap();
    // Naming itself keeps it off the unused list in the host pass, where the macro below
    // expands to nothing and nothing calls it.
    UNUSED(no_device_code);
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE
#endif

/// Opt one kernel into a dynamic shared-memory allocation larger than the default cap.
///
/// The driver wants this per kernel and per device, and the amount is decided by the kernel's
/// own template arguments, so asking a second time on the same device can only repeat the
/// first answer. Each expansion therefore carries its own record of the devices already asked
/// - one bit each, since a process addresses at most `GGML_CUDA_MAX_DEVICES` of them.
#define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                           \
    do {                                                                                       \
        static_assert(GGML_CUDA_MAX_DEVICES <= 32, "one bit per device, and no bits to spare"); \
        static unsigned int already_asked = 0;                                                 \
        int device = 0;                                                                        \
        cudaGetDevice(&device);                                                                \
        const unsigned int bit = 1u << device;                                                 \
        if ((already_asked & bit) == 0) {                                                       \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes); \
            already_asked |= bit;                                                              \
        }                                                                                      \
    } while (0)

// ============================================================
// Dividing by a number the host knows and the device does not
//
// The inner loop divides by the same few values on every iteration - blocks to a row, tiles to
// a channel - and integer division is among the slowest things a lane can be asked to do. The
// host converts each divisor into a multiply-and-shift once, and the device applies it: for a
// divisor d, the pair (mp, L) is chosen so that (mulhi(n, mp) + n) >> L equals n / d for every
// 32-bit n. The construction is the round-up variant of Granlund-Möller. The divisor rides
// along in the third slot so the remainder follows from the quotient without a second pair.
// ============================================================
static inline uint3 init_fastdiv_values(uint32_t d) {
    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && ((uint32_t)1 << L) < d) {
        L++;
    }
    uint32_t mp = (uint32_t) ((((uint64_t)1) << 32) * ((((uint64_t)1) << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fdv) {
    const uint32_t hi = __umulhi(n, fdv.x);
    return (hi + n) >> fdv.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fdv) {
    return n - fastdiv(n, fdv) * fdv.z;
}

static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fdv) {
    const uint32_t q = fastdiv(n, fdv);
    const uint32_t m = n - q * fdv.z;
    return make_uint2(q, m);
}

// ============================================================
// Additional macros and helpers
// ============================================================

template<typename... Args>
__host__ __device__ constexpr inline void ggml_unused_vars_impl(Args&&...) noexcept {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif
}

/// The widest unit a copy of this width can move at a time.
template <int bytes> struct copy_unit;
template <> struct copy_unit<1>  { using type = char;  };
template <> struct copy_unit<2>  { using type = short; };
template <> struct copy_unit<4>  { using type = int;   };
template <> struct copy_unit<8>  { using type = int2;  };
template <> struct copy_unit<16> { using type = int4;  };

/// Move `nbytes` between registers and shared memory.
///
/// `alignment` is what the caller can promise about BOTH addresses; zero means the whole copy
/// is one aligned move. Asking for a wider unit than the addresses support is not slower, it is
/// wrong, which is why the width is the caller's promise and not the copy's size.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void copy_bytes(
        void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment != 0,
                  "a copy this wide has to say what it is aligned to");
    constexpr int unit = alignment == 0 ? nbytes : alignment;
    static_assert(nbytes % unit == 0, "the copy does not divide into whole units");
    using unit_t = typename copy_unit<unit>::type;
#pragma unroll
    for (int i = 0; i < nbytes / unit; ++i) {
        ((unit_t *) dst)[i] = ((const unit_t *) src)[i];
    }
}
