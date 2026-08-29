// The host side of the W4A16 GEMM: pick a tile, give the block the shared memory that tile
// needs, and walk the batch in chunks the compiled kernels can take.
//
//   A      = f16 [m, k] row-major (lda = k), 16-byte aligned
//   B      = repacked u4 [k/16, n*2] int32, integer zero point
//   scales = f16 [k/group_size, n], lane-permuted
//   zp     = u4  [k/group_size, n/8] int32, permuted and interleaved
//   C      = f16 [m, n]
//   group_size = 128; there is no act-order and no bias.

#include "kernel.h"
#include <cuda_runtime.h>

namespace MARLIN_NAMESPACE_NAME {

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

/// One compiled tile shape, and the kernel that runs it.
///
/// The instantiation list and the search list are the same list. A shape the search returns
/// therefore exists by construction, which is why nothing below reports a missing kernel.
struct CompiledShape {
    /// Threads per block.
    int threads;
    /// The tile, in 16-wide blocks of m, n and k.
    int m_blocks, n_blocks, k_blocks;
    /// An 8-row tile, for a batch that does not fill sixteen.
    bool half_tile;
    MarlinFuncPtr fn;
};

#define LOKEN_MARLIN_SHAPE(TH, TMB, TNB, TKB, M8) \
    {TH, TMB, TNB, TKB, M8, Marlin<TH, TMB, TNB, TKB, M8>},

/// Widest tile first: a wide tile reads each weight once for more output. The search walks down
/// the list when the problem does not divide by the tile or the tile does not fit.
static const CompiledShape compiled_shapes[] = {
    LOKEN_MARLIN_SHAPE(256, 1, 8, 8, true)      //
    LOKEN_MARLIN_SHAPE(128, 1, 8, 4, true)      //
    LOKEN_MARLIN_SHAPE(128, 1, 4, 8, true)      //
    LOKEN_MARLIN_SHAPE(256, 1, 8, 8, false)     //
    LOKEN_MARLIN_SHAPE(128, 1, 8, 4, false)     //
    LOKEN_MARLIN_SHAPE(128, 1, 4, 8, false)     //
    LOKEN_MARLIN_SHAPE(256, 2, 16, 4, false)    //
    LOKEN_MARLIN_SHAPE(128, 2, 8, 4, false)     //
    LOKEN_MARLIN_SHAPE(128, 2, 4, 8, false)     //
};

#undef LOKEN_MARLIN_SHAPE

/// Shared memory one block of `s` needs, counted the way the kernel lays it out.
///
/// The weight tile and the reduction buffer never live at once, so they share a region sized by
/// whichever is larger; the activations, the scales and the zero points each get their own,
/// `stages` deep.
static int shared_bytes(const CompiledShape& s, int stages, int group_size) {
    const int tb_m = s.half_tile ? 8 : s.m_blocks * 16;
    const int tb_n = s.n_blocks * 16;
    const int tb_k = s.k_blocks * 16;

    const int sh_a = stages * tb_m * tb_k * 2;                        // f16
    const int sh_b = stages * (tb_k * tb_n / 8) * 4;                  // 4 bits, eight to an int
    const int sh_red = tb_m * (tb_n + 8) * 2;                         // f16, one row of padding
    const int sh_s = stages * div_ceil(tb_k, group_size) * tb_n * 2;  // f16
    const int sh_zp = sh_s / 4;                                       // 4 bits against f16

    return (sh_b > sh_red ? sh_b : sh_red) + sh_a + sh_s + sh_zp;
}

/// The tile this problem gets, or null when none of them divides it and fits.
///
/// One refinement on "widest that fits": when the widest tile would leave multiprocessors with
/// nothing to do, a narrower one cuts the same work into more blocks. Every compiled shape
/// already clears the minimum tile and thread count, so there is nothing else to check.
static const CompiledShape* choose_shape(int prob_n, int prob_k, int m_blocks, bool half_tile,
                                         int group_size, int stages, int smem_budget, int sms) {
    const CompiledShape* widest = nullptr;
    const CompiledShape* narrowest = nullptr;
    for (const CompiledShape& s : compiled_shapes) {
        if (s.m_blocks != m_blocks || s.half_tile != half_tile) continue;
        if (prob_n % (s.n_blocks * 16) != 0 || prob_k % (s.k_blocks * 16) != 0) continue;
        if (shared_bytes(s, stages, group_size) > smem_budget) continue;
        if (!widest) widest = &s;
        if (!narrowest || s.n_blocks < narrowest->n_blocks) narrowest = &s;
    }
    if (widest && prob_n / (widest->n_blocks * 16) * 4 <= sms) return narrowest;
    return widest;
}

/// Opt one kernel into the full shared-memory carveout, once per device.
///
/// Decode calls the launcher hundreds of times a token and the attribute is sticky, so the
/// table is what keeps the driver call off the hot path. Two threads racing here set the same
/// attribute to the same value.
static bool grant_shared_memory(MarlinFuncPtr kernel, int bytes) {
    constexpr int max_entries = 64;
    static void* seen_kernel[max_entries] = {};
    static int seen_dev[max_entries] = {};

    int cur_dev = 0;
    cudaGetDevice(&cur_dev);
    for (int i = 0; i < max_entries && seen_kernel[i]; i++) {
        if (seen_kernel[i] == (void*) kernel && seen_dev[i] == cur_dev) return true;
    }
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
        cudaSuccess) {
        return false;
    }
    for (int i = 0; i < max_entries; i++) {
        if (!seen_kernel[i]) {
            seen_kernel[i] = (void*) kernel;
            seen_dev[i] = cur_dev;
            break;
        }
    }
    return true;
}

}  // namespace MARLIN_NAMESPACE_NAME

// GEMM C[m,n] = A[m,k] . dequant(B) for the repacked AWQ layout.
//   c_tmp: f32 scratch for the partial-tile reduction (>= sms*64*256 floats).
//   locks: int workspace (>= sms entries), ZEROED at allocation; the kernel restores the zeros
//          so it can be reused without clearing it again.
//   sms / max_shared_mem: the device's multiprocessor count and its per-block opt-in shared
//          memory, queried once on the Rust side; -1 asks this function to query them.
// Returns 0 on success, negative on error:
//   -1 bad problem shape, -2 unsupported group size, -3 no tile fits, -5 CUDA error.
extern "C" int loken_marlin_awq_f16_gemm(const void* A, const void* B, const void* scales,
                                         const void* zp, void* C, float* c_tmp, int* locks,
                                         int prob_m, int prob_n, int prob_k, int num_groups,
                                         int sms, int max_shared_mem, void* stream) {
    using namespace MARLIN_NAMESPACE_NAME;

    if (prob_m <= 0 || prob_n <= 0 || prob_k <= 0) return -1;
    if (prob_k % 16 != 0 || prob_n % 64 != 0) return -1;
    if (num_groups <= 0 || prob_k % num_groups != 0) return -2;
    const int group_size = prob_k / num_groups;
    if (group_size != 128) return -2;  // the only group the kernels are compiled for

    if (sms <= 0 &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        return -5;
    }
    if (max_shared_mem <= 0 &&
        cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) !=
            cudaSuccess) {
        return -5;
    }

    constexpr int stages = 4;
    const int lda = prob_k;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    const int4* A_ptr = reinterpret_cast<const int4*>(A);
    const int4* B_ptr = reinterpret_cast<const int4*>(B);
    int4* C_ptr = reinterpret_cast<int4*>(C);
    int4* C_tmp_ptr = reinterpret_cast<int4*>(c_tmp);
    const int4* s_ptr = reinterpret_cast<const int4*>(scales);
    const int4* zp_ptr = reinterpret_cast<const int4*>(zp);

    // Two 16-row blocks is the tallest tile compiled - the decode and speculative-verify
    // regime. A taller batch is several launches of that height.
    constexpr int max_m_blocks = 2;
    const int max_chunks = prob_n <= 4096 ? 128 : 16;

    int rest_m = prob_m;
    while (rest_m) {
        int chunks = rest_m / (max_m_blocks * 16);
        if (chunks > max_chunks) chunks = max_chunks;
        const int prob_m_split = chunks > 0 ? chunks * (max_m_blocks * 16) : rest_m;

        const int m_blocks = min(div_ceil(prob_m_split, 16), max_m_blocks);
        const bool half_tile = prob_m_split <= 8;

        // The tile comes from the device - its multiprocessor count and its opt-in shared
        // memory - and from the problem's own shape. There is nothing here for a caller to set.
        // The budget keeps half a kilobyte back: the driver reserves a little of the carveout
        // for itself, and a block that asks for every byte does not launch.
        const CompiledShape* shape = choose_shape(prob_n, prob_k, m_blocks, half_tile, group_size,
                                                  stages, max_shared_mem - 512, sms);
        if (!shape) return -3;
        if (!grant_shared_memory(shape->fn, max_shared_mem)) return -5;

        shape->fn<<<sms, shape->threads, max_shared_mem, cu_stream>>>(
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, zp_ptr, prob_m_split, prob_n, prob_k, lda,
            locks, max_shared_mem);
        if (cudaGetLastError() != cudaSuccess) return -5;

        A_ptr += prob_m_split * (lda / 8);
        C_ptr += prob_m_split * (prob_n / 8);
        rest_m -= prob_m_split;
    }
    return 0;
}
