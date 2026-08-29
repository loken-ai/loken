#pragma once

/// What one thread holds of the tensor-core instruction this GEMM is built on, and the
/// arithmetic the kernel asks of the type it carries.
///
/// `mma.sync.aligned.m16n8k16` multiplies a 16x16 tile by a 16x8 one across a whole warp, and
/// every operand is spread over the warp's 32 lanes. So a lane's share of each is the tile's
/// size divided by 32, and half-precision values travel two to a register - which is where the
/// register counts below come from, rather than from four numbers written out.
///
/// Only float16 is carried. The bfloat16 and fp8 arms of the table this replaces, the
/// second-scale fragment and the by-carrier alias were reachable from nothing, so they are not
/// here: what a header declares is what the kernel asks it for.

#include "marlin.cuh"
#include "scalar_type.hpp"
#include <cuda_fp16.h>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME loken_w4a16
#endif

namespace MARLIN_NAMESPACE_NAME {

/// The instruction's tile, which fixes every fragment size below.
struct MmaTile {
    static constexpr int m = 16, n = 8, k = 16;
    static constexpr int lanes = 32;
    /// Halves per lane for a tile of `rows x cols`, in registers of two.
    static constexpr int pairs(int rows, int cols) { return rows * cols / lanes / 2; }
};

/// A lane's share of each operand, for halves travelling two to a register.
///
/// The accumulator is the exception to the pairing: `mma` accumulates in float32 whatever its
/// operands are, so `FragC` counts values and not pairs. `FragS` is odd for another reason  - 
/// it is not an operand of the instruction at all, but one group's scales, four of these to
/// the sixteen bytes a load moves.
struct HalfFragments {
    using FragA = Vec<half2, MmaTile::pairs(MmaTile::m, MmaTile::k)>;
    using FragB = Vec<half2, MmaTile::pairs(MmaTile::k, MmaTile::n)>;
    using FragC = Vec<float, MmaTile::m * MmaTile::n / MmaTile::lanes>;
    using FragS = Vec<half2, 1>;
    /// The zero points a lane subtracts from its B operand, dequantised eight at a time - the
    /// same shape as A because they are read the same way.
    using FragZP = Vec<half2, MmaTile::pairs(MmaTile::m, MmaTile::k)>;
};

/// The conversions the kernel performs on a carried value: one reading a scale, three writing
/// the accumulator back out.
struct HalfArithmetic {
    static __device__ inline float num2float(const half x) { return __half2float(x); }
    static __device__ inline half2 num2num2(const half x) { return __half2half2(x); }
    static __device__ inline half2 nums2num2(const half a, const half b) {
        return __halves2half2(a, b);
    }
    static __host__ __device__ inline half float2num(const float x) { return __float2half(x); }
};

/// A carrier is its fragment shapes, its arithmetic, and the two names the kernel spells its
/// values by. The primary template is declared and never defined, so an id the build has no
/// carrier for is a compile error that names the id rather than an empty class that fails
/// later on a missing member.
template <long scalar_type_id>
class MarlinScalarType;

/// float16: the carrier the weights are dequantised into and the activations arrive in.
template <>
class MarlinScalarType<loken::kFloat16.id()> : public HalfFragments, public HalfArithmetic {
 public:
    using scalar_t = half;
    using scalar_t2 = half2;
    /// What `dequant` writes through: a 32-bit word IS a pair of halves here.
    using scalar_32bit_t = half2;
};

}  // namespace MARLIN_NAMESPACE_NAME
