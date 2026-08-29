#pragma once

// The tensor-core multiply-accumulate the GEMM is built on.
//
// One instruction computes a 16x8 output tile from a 16x16 slice of A and a 16x8 slice of B,
// accumulating into C in place. The operands live in registers spread across the warp in a
// layout the instruction fixes; what this header does is name which registers go where.
//
// Turing has no k=16 form of it, so there the same product is two k=8 instructions over the
// halves of A and of B. That is not a slow path - it is the same arithmetic issued twice.
//
// Whichever shape is issued, and whichever width it accumulates in, the operands arrive the
// same way: four registers of A and two of B. Only the destination differs - two registers of
// packed halves, or four floats. So the register plumbing is written once per destination
// width, and an instruction is then a shape plus a note of which of those registers it reads.

#include <cuda_fp16.h>

#include <type_traits>

#include "marlin_dtypes.cuh"
#include "scalar_type.hpp"

namespace MARLIN_NAMESPACE_NAME {

/// `D = A*B + C`, over four registers of A and two of B.
///
/// Accumulating in f16 halves the register cost of C, which is what lets a block hold a wider
/// output tile; it costs precision in the tail of a long reduction, so the caller decides.
template <bool accumulate_in_f16>
__device__ inline void mma_f16(const uint32_t* a, const uint32_t* b, void* c_ptr) {
    // The destination occupies `%0` onwards and the operands follow it, so each width numbers
    // the same registers differently - which is the whole of what the two emissions differ in.
    // `slice` is the A and B operand pair an instruction reads, in that numbering: all of them
    // for the k=16 shape, half of each for a k=8 shape.
#define LOKEN_MMA_F16_ACC(shape, slice)                                                 \
    asm volatile("mma.sync.aligned." shape ".row.col.f16.f16.f16.f16 "                  \
                 "{%0,%1}, " slice ", {%8,%9};\n"                                       \
                 : "=r"(c[0]), "=r"(c[1])                                               \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),    \
                   "r"(c[0]), "r"(c[1]))

#define LOKEN_MMA_F32_ACC(shape, slice)                                                 \
    asm volatile("mma.sync.aligned." shape ".row.col.f32.f16.f16.f32 "                  \
                 "{%0,%1,%2,%3}, " slice ", {%10,%11,%12,%13};\n"                       \
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])                       \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),    \
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]))

    if constexpr (accumulate_in_f16) {
        uint32_t* c = reinterpret_cast<uint32_t*>(c_ptr);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
        LOKEN_MMA_F16_ACC("m16n8k8", "{%2,%3}, {%6}");
        LOKEN_MMA_F16_ACC("m16n8k8", "{%4,%5}, {%7}");
#else
        LOKEN_MMA_F16_ACC("m16n8k16", "{%2,%3,%4,%5}, {%6,%7}");
#endif
    } else {
        float* c = reinterpret_cast<float*>(c_ptr);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
        LOKEN_MMA_F32_ACC("m16n8k8", "{%4,%5}, {%8}");
        LOKEN_MMA_F32_ACC("m16n8k8", "{%6,%7}, {%9}");
#else
        LOKEN_MMA_F32_ACC("m16n8k16", "{%4,%5,%6,%7}, {%8,%9}");
#endif
    }

#undef LOKEN_MMA_F16_ACC
#undef LOKEN_MMA_F32_ACC
}

/// A fragment's registers, for handing to the instruction as an operand list.
template <typename Frag>
__device__ inline const uint32_t* regs_of(const Frag& frag) {
    static_assert(sizeof(Frag) % sizeof(uint32_t) == 0, "a fragment is whole registers");
    return reinterpret_cast<const uint32_t*>(&frag);
}

/// Both products below name `f16` operands, so the carrier has to be that.
template <loken::ScalarTypeId type_id>
constexpr bool carried_as_f16 =
    std::is_same<typename MarlinScalarType<type_id>::scalar_t, half>::value;

/// The activation tile against one weight fragment.
template <loken::ScalarTypeId type_id, bool use_fp16_accum>
__device__ inline void mma(const typename MarlinScalarType<type_id>::FragA& a_frag,
                           const typename MarlinScalarType<type_id>::FragB& frag_b,
                           typename MarlinScalarType<type_id>::FragC& frag_c) {
    static_assert(carried_as_f16<type_id>, "the fragments here are f16");
    mma_f16<use_fp16_accum>(regs_of(a_frag), regs_of(frag_b), &frag_c);
}

/// The same product with the operands' roles exchanged: the WEIGHTS play A and the activation
/// plays B.
///
/// A is sixteen wide and a weight fragment is eight, so two of them are interleaved to fill
/// it - which is why this takes `frag_b` and `frag_b2`. It computes a narrow output tile
/// without anything being transposed in memory.
template <loken::ScalarTypeId type_id, bool use_fp16_accum>
__device__ inline void mma_trans(const typename MarlinScalarType<type_id>::FragA& a_frag,
                                 const typename MarlinScalarType<type_id>::FragB& frag_b,
                                 const typename MarlinScalarType<type_id>::FragB& frag_b2,
                                 typename MarlinScalarType<type_id>::FragC& frag_c) {
    static_assert(carried_as_f16<type_id>, "the fragments here are f16");
    const uint32_t* b = regs_of(frag_b);
    const uint32_t* b2 = regs_of(frag_b2);
    const uint32_t lhs[4] = {b[0], b2[0], b[1], b2[1]};
    mma_f16<use_fp16_accum>(lhs, regs_of(a_frag), &frag_c);
}

}  // namespace MARLIN_NAMESPACE_NAME
