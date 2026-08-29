// Marlin W4A16 kernel instantiations - AWQ (u4 + int zero-point), f16
// activation/output/scales, group_size=128 (group_blocks=8), 4-stage cp.async
// pipeline. This unit: thread_m_blocks=1 full-16-row variants (M = 9..16).
//
// The tile shapes here are the ones the launcher's table searches; they are the same
// list, so a shape the search returns exists by construction.
// clang-format off

#include "kernel.h"
#include "marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

// (thread_k=128, thread_n=128, threads=256)
template __global__ void Marlin<256, 1, 8, 8, false>( MARLIN_KERNEL_PARAMS );

// (thread_k=64, thread_n=128, threads=128)
template __global__ void Marlin<128, 1, 8, 4, false>( MARLIN_KERNEL_PARAMS );

// (thread_k=128, thread_n=64, threads=128)
template __global__ void Marlin<128, 1, 4, 8, false>( MARLIN_KERNEL_PARAMS );

}
