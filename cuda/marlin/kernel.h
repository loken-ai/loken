#pragma once

// The one kernel this build compiles, declared for the units that instantiate it and for the
// launcher that takes its address.

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME loken_w4a16
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "scalar_type.hpp"

// A declaration, an explicit instantiation and a function-pointer type all have to repeat the
// argument list, so it is written here once, in the two halves it falls into. The definition
// spells the same list out with a note on each argument; this is only the shape of the call.

/// The memory a launch hands over: the two operands, the output, the scratch a split tile
/// leaves its partial sums in, and the two quantisation tables.
#define MARLIN_KERNEL_TENSORS                \
    const int4* __restrict__ A,              \
        const int4* __restrict__ B,          \
        int4* __restrict__ C,                \
        int4* __restrict__ C_tmp,            \
        const int4* __restrict__ scales_ptr, \
        const int4* __restrict__ zp_ptr

/// The problem those cover, then the two things only a launch knows: one lock word per tile
/// that blocks share, and how much shared memory the block was given.
#define MARLIN_KERNEL_EXTENTS                    \
    int prob_m, int prob_n, int prob_k, int lda, \
        int* locks,                              \
        int max_shared_mem

#define MARLIN_KERNEL_PARAMS MARLIN_KERNEL_TENSORS, MARLIN_KERNEL_EXTENTS

namespace MARLIN_NAMESPACE_NAME {

// The five parameters a launch varies - how the output tile is cut, and how wide it is. What
// the kernel took as a parameter beyond these is fixed by what the build instantiates and is
// a constant in its body, so this signature says which kernel it is.
template <const int threads,          // threads in a threadblock
          const int thread_m_blocks,  // 16x16 blocks along m (batch)
          const int thread_n_blocks,  // ... along n (output)
          const int thread_k_blocks,  // ... along k (reduction)
          const bool m_block_size_8   // an 8-row tile, only when thread_m_blocks == 1
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
