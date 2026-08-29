/// Four-bit weights into f16, by arithmetic on the bit pattern.

#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

/// Three-input logic in one instruction.
///
/// `lut` is the truth table: bit `i` of it is the result for the input combination whose bits
/// are `i`. The convention is that `a`, `b` and `c` stand for the constants 0xf0, 0xcc and
/// 0xaa, so any expression written in those three constants evaluates to its own table.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

/// Four of the eight 4-bit weights packed in `q`, as two `half2` - each one biased by 1024.
///
/// An f16 whose exponent says 2^10 and whose mantissa holds a small integer *is* that integer
/// plus 1024, because at that exponent the mantissa counts in units of one. So dropping a
/// nibble into the mantissa of the constant 0x6400 is the whole conversion: one instruction,
/// no integer-to-float, no table.
///
/// The bias is never removed here. The kernel subtracts a zero point that came through this
/// same function, and 1024 cancels against 1024 - a weight format with no zero point would be
/// the one that had to pay for the subtraction.
///
/// Two weights convert at once because the halves of a `half2` are sixteen bits apart and so
/// are nibbles 0 and 4 of the word; shifting by four takes nibbles 1 and 5. Called on `q` and
/// on `q >> 8`, the pair reads the word in the order 0 4 1 5 2 6 3 7, which is why the repack
/// stores the eight weights shuffled: stored that way, they arrive in fragment order.
__device__ inline void dequant_u4_biased(int q, half2* frag_b) {
  constexpr int NIBBLE_PER_HALF = 0x000f000f;
  constexpr int BIAS_1024 = 0x64006400;
  constexpr int A_AND_B_OR_C = (0xf0 & 0xcc) | 0xaa;

  const int lo = lop3<A_AND_B_OR_C>(q, NIBBLE_PER_HALF, BIAS_1024);
  const int hi = lop3<A_AND_B_OR_C>(q >> 4, NIBBLE_PER_HALF, BIAS_1024);
  frag_b[0] = *reinterpret_cast<const half2*>(&lo);
  frag_b[1] = *reinterpret_cast<const half2*>(&hi);
}

#endif

}  // namespace MARLIN_NAMESPACE_NAME
