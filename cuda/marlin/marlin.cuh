#pragma once

// The register array every fragment in this GEMM is cut from, and the one load that fills a
// pipeline stage.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME loken_w4a16
#endif

namespace MARLIN_NAMESPACE_NAME {

/// A fixed-length array of registers.
///
/// The storage is a plain array and the struct carries nothing beside it, so the type stays an
/// aggregate: a fragment goes straight to inline PTX as a list of registers, and a shared
/// memory word read as an `I4` names the four the `mma` will consume. Every index the kernel
/// writes is a compile-time constant, which is what keeps `regs` in registers instead of
/// spilling the array to local memory.
template <typename T, int n>
struct Vec {
    T regs[n];
    __device__ T& operator[](int i) { return regs[i]; }
};

/// The four registers one sixteen-byte load lands in.
using I4 = Vec<int, 4>;

/// Rounded up, over counts that are never negative.
constexpr int div_ceil(int num, int den) { return (num + den - 1) / den; }

// -----------------------------------------------------------------------------------------
// Filling a stage.
//
// Every load the kernel issues moves the same sixteen bytes per thread - eight activations,
// thirty-two weights, or eight scales - so there is one copy here and not a family of them
// sized by the caller. It bypasses L1, because a tile is read once and never revisited and
// caching it would only evict something that will be. The predicate rides inside the
// instruction instead of a branch around it, so a warp whose last lanes have nothing left to
// fetch does not diverge.
//
// A stage needs three things done to it - fill it, close the batch of fills, wait an older
// batch down - and `StageCopy` is the one place that says how, so it is also the one place
// the architecture is asked. The two definitions below differ in their bodies alone; the
// names the kernel calls are written once underneath and read the same either way.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

/// From Ampere the fill is asynchronous, which is what lets several stages be in flight: the
/// thread issues the load and keeps computing on the stage it already holds, and a wait is
/// where it finally stops.
struct StageCopy {
    /// The shared-memory operand of a fill, which is a window offset and not a generic pointer.
    static __device__ inline uint32_t stage_offset(void* smem_ptr) {
        return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    }

    /// Sixteen bytes from global to shared, issued when `pred` and awaited later.
    static __device__ inline void fill(void* smem_ptr, const void* glob_ptr, bool pred) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %0, 0;"
                     " @p cp.async.cg.shared.global [%1], [%2], 16; }\n"
                     :
                     : "r"(static_cast<int>(pred)), "r"(stage_offset(smem_ptr)), "l"(glob_ptr));
    }

    /// Close the batch of fills issued since the last one closed, so a later wait can name it.
    static __device__ inline void close_batch() { asm volatile("cp.async.commit_group;\n" ::); }

    /// Block until all but the `keep` most recent batches have landed - `keep` is how deep the
    /// pipeline is allowed to stay, not how long to wait.
    template <int keep>
    static __device__ inline void wait_down_to() {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(keep));
    }
};

#else

/// Before Ampere the same sixteen bytes travel through registers and have landed by the time
/// the fill returns, so nothing is ever outstanding: there is no batch to close, and no depth
/// to wait down to.
struct StageCopy {
    static __device__ inline void fill(void* smem_ptr, const void* glob_ptr, bool pred) {
        if (pred) {
            *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(glob_ptr);
        }
    }

    static __device__ inline void close_batch() {}

    template <int keep>
    static __device__ inline void wait_down_to() {}
};

#endif

/// A load that may run past the end of the matrix, and so carries the predicate that says
/// whether it is one of the lanes with something left to fetch.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred) {
    StageCopy::fill(smem_ptr, glob_ptr, pred);
}

/// A load whose address is known to be inside the matrix, and so needs no predicate.
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
    StageCopy::fill(smem_ptr, glob_ptr, true);
}

__device__ inline void cp_async_fence() { StageCopy::close_batch(); }

template <int n>
__device__ inline void cp_async_wait() {
    StageCopy::wait_down_to<n>();
}

}  // namespace MARLIN_NAMESPACE_NAME
