//! DeepSeek V4.1 block-scaled fp8/fp4 weight dequantization (bet phase 7).
//!
//! The released weights are fp8 (e4m3) in 32x32 blocks with a power-of-two (ue8m0) scale per
//! block, and the experts fp4 (e2m1, two per byte) with a ue8m0 scale per 32 elements along the
//! input dimension. Dequantizing is decoding the low-precision value and multiplying its block
//! scale, exactly as the reference convert.py does. Because the scale is a power of two the product
//! is exact in f32; only the final cast to bf16 rounds, so loken's dequant is bit-for-bit the
//! reference's.
//!
//! The reference is `notes/deepseek-oracle`; the dequant is judged against a dump in the test.

/// The e2m1 grid indexed by the low three bits of a nibble (exponent in bits 2-1, mantissa bit 0).
const FP4_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode one OCP e4m3fn fp8 byte to f32 (1 sign, 4 exp bias-7, 3 mantissa; S1111.111 = NaN).
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 0x0F && mant == 7.0 {
        return f32::NAN;
    }
    let v = if exp == 0 {
        (mant / 8.0) * 2f32.powi(-6)
    } else {
        (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    };
    sign * v
}

/// Decode one e2m1 fp4 nibble (bit 3 sign, bits 2-1 exp, bit 0 mantissa) to f32.
#[inline]
pub fn e2m1_to_f32(nib: u8) -> f32 {
    let mag = FP4_GRID[(nib & 0x07) as usize];
    if nib & 0x08 != 0 {
        -mag
    } else {
        mag
    }
}

/// Decode one ue8m0 scale byte: a power of two, `2^(e - 127)`; the all-ones byte is NaN.
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else {
        2f32.powi(b as i32 - 127)
    }
}

/// Dequantize a block-scaled fp8 weight. `bytes` is [out, in] e4m3, `scale` [out/block, in/block]
/// the power-of-two block scales already decoded to f32. Returns [out, in] f32.
pub fn dequant_fp8_blocked(
    bytes: &[u8],
    scale: &[f32],
    out: usize,
    in_dim: usize,
    block: usize,
) -> Vec<f32> {
    let sib = in_dim / block;
    let mut w = vec![0f32; out * in_dim];
    for i in 0..out {
        let sr = (i / block) * sib;
        for j in 0..in_dim {
            w[i * in_dim + j] = e4m3_to_f32(bytes[i * in_dim + j]) * scale[sr + j / block];
        }
    }
    w
}

/// Dequantize a block-scaled fp4 weight. `bytes` is [out, in/2] packed e2m1 (element 2j in the low
/// nibble, 2j+1 in the high), `scale` [out, in/block] the power-of-two scales. Returns [out, in].
pub fn dequant_fp4_blocked(
    bytes: &[u8],
    scale: &[f32],
    out: usize,
    in_dim: usize,
    block: usize,
) -> Vec<f32> {
    let packed = in_dim / 2;
    let sib = in_dim / block;
    let mut w = vec![0f32; out * in_dim];
    for i in 0..out {
        let sr = i * sib;
        for j in 0..in_dim {
            let byte = bytes[i * packed + j / 2];
            let nib = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            w[i * in_dim + j] = e2m1_to_f32(nib) * scale[sr + j / block];
        }
    }
    w
}
