#pragma once

#include <cstdint>

// What a Marlin template argument needs to know about a numeric type.
//
// A kernel is selected on four of these - the activation, the weight, the output and the
// scales - and it asks them three things: how wide the type is, whether it is a particular
// one, and an integer identity it can be passed as a template argument. That is the whole of
// the surface, measured: `.id()`, `.size_bits()`, `from_id`, and equality against a named
// constant.
//
// The identity is an encoding of the description rather than a registry number, so a type is
// its own name: the kernel that reads it back through `from_id` gets the same description,
// and nothing outside this build ever sees it.

namespace loken {

using ScalarTypeId = int64_t;

struct ScalarType {
    /// How a value is laid out inside its bits.
    enum Kind : int64_t {
        Unsigned = 0,  // an unsigned integer, optionally offset by `bias`
        Signed = 1,    // a two's-complement integer
        Float = 2,     // an IEEE-like float, split into `exponent` and mantissa bits
    };

    int64_t kind;
    int64_t bits;
    /// For `Unsigned`, the value subtracted after decoding - a u4 with bias 8 spans -8..7.
    int64_t bias;
    /// For `Float`, the width of the exponent field; the mantissa takes what is left, less
    /// the sign bit for the signed forms.
    int64_t exponent;

    constexpr int size_bits() const { return static_cast<int>(bits); }

    /// The four fields, packed so that distinct descriptions get distinct ids and the packing
    /// is reversible. Sixteen bits each is far more than any of them uses.
    constexpr ScalarTypeId id() const {
        return (kind << 48) | (exponent << 32) | (bias << 16) | bits;
    }

    static constexpr ScalarType from_id(ScalarTypeId v) {
        return ScalarType{(v >> 48) & 0xFFFF, v & 0xFFFF, (v >> 16) & 0xFFFF,
                          (v >> 32) & 0xFFFF};
    }

    constexpr bool operator==(const ScalarType& o) const { return id() == o.id(); }
    constexpr bool operator!=(const ScalarType& o) const { return id() != o.id(); }

    static constexpr ScalarType uint(int64_t bits, int64_t bias = 0) {
        return ScalarType{Unsigned, bits, bias, 0};
    }
    /// `bits` counts the sign bit; the mantissa is what remains after the exponent.
    static constexpr ScalarType float_(int64_t exponent, int64_t mantissa) {
        return ScalarType{Float, 1 + exponent + mantissa, 0, exponent};
    }
};

// The two types the kernel names. Nine more were listed here as the shapes a quantised
// weight can take; nothing named them, not even a branch the instantiation discards, so a
// format that arrives brings its own line back along with the kernel that reads it.
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kFloat16 = ScalarType::float_(5, 10);

}  // namespace loken
