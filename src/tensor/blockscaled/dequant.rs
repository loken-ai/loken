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

/// Every e4m3 byte decoded, as `e4m3_to_f32` decodes it.
fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|b| e4m3_to_f32(b as u8)))
}

/// Every e2m1 nibble decoded, as `e2m1_to_f32` decodes it.
const FP4_VALUES: [f32; 16] = {
    let mut t = [0f32; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = if i & 8 != 0 {
            -FP4_GRID[i & 7]
        } else {
            FP4_GRID[i & 7]
        };
        i += 1;
    }
    t
};

/// Dequantize a block-scaled fp8 weight. `bytes` is [out, in] e4m3, `scale` [out/block, in/block]
/// the power-of-two block scales already decoded to f32. Returns [out, in] f32, rows decoded
/// across the rayon pool.
pub fn dequant_fp8_blocked(
    bytes: &[u8],
    scale: &[f32],
    out: usize,
    in_dim: usize,
    block: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let sib = in_dim / block;
    let table = e4m3_table();
    let mut w = vec![0f32; out * in_dim];
    w.par_chunks_mut(in_dim.max(1))
        .enumerate()
        .for_each(|(i, row)| {
            let sr = (i / block) * sib;
            let src = &bytes[i * in_dim..(i + 1) * in_dim];
            for (j, (o, &b)) in row.iter_mut().zip(src).enumerate() {
                *o = table[b as usize] * scale[sr + j / block];
            }
        });
    w
}

/// Dequantize a block-scaled fp4 weight. `bytes` is [out, in/2] packed e2m1 (element 2j in the low
/// nibble, 2j+1 in the high), `scale` [out, in/block] the power-of-two scales. Returns [out, in],
/// rows decoded across the rayon pool.
pub fn dequant_fp4_blocked(
    bytes: &[u8],
    scale: &[f32],
    out: usize,
    in_dim: usize,
    block: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let packed = in_dim / 2;
    let sib = in_dim / block;
    let mut w = vec![0f32; out * in_dim];
    w.par_chunks_mut(in_dim.max(1))
        .enumerate()
        .for_each(|(i, row)| {
            let src = &bytes[i * packed..(i + 1) * packed];
            let scales = &scale[i * sib..(i + 1) * sib];
            for (j, pair) in row.chunks_exact_mut(2).enumerate() {
                let byte = src[j];
                pair[0] = FP4_VALUES[(byte & 0x0F) as usize] * scales[2 * j / block];
                pair[1] = FP4_VALUES[(byte >> 4) as usize] * scales[(2 * j + 1) / block];
            }
        });
    w
}

/// `dequant_fp4_blocked` in column order: value `(i, j)` at `j * out + i`, so each column of the
/// weight is contiguous. Blocks of `block` columns are decoded across the rayon pool.
pub fn dequant_fp4_blocked_columns(
    bytes: &[u8],
    scale: &[f32],
    out: usize,
    in_dim: usize,
    block: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    if !block.is_multiple_of(2) {
        let rows = dequant_fp4_blocked(bytes, scale, out, in_dim, block);
        let mut w = vec![0f32; out * in_dim];
        w.par_chunks_mut(out.max(1))
            .enumerate()
            .for_each(|(j, column)| {
                for (i, v) in column.iter_mut().enumerate() {
                    *v = rows[i * in_dim + j];
                }
            });
        return w;
    }
    let packed = in_dim / 2;
    let sib = in_dim / block;
    let mut w = vec![0f32; out * in_dim];
    w.par_chunks_mut((out * block).max(1))
        .enumerate()
        .for_each(|(cb, columns)| {
            for i in 0..out {
                let s = scale[i * sib + cb];
                let src = &bytes[i * packed + cb * block / 2..i * packed + (cb + 1) * block / 2];
                for (k, &byte) in src.iter().enumerate() {
                    columns[2 * k * out + i] = FP4_VALUES[(byte & 0x0F) as usize] * s;
                    columns[(2 * k + 1) * out + i] = FP4_VALUES[(byte >> 4) as usize] * s;
                }
            }
        });
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every decoder gives each value the per-element decoding gives it, bit for bit, the column
    /// order included, for an even block and an odd one.
    #[test]
    fn the_decoders_agree_with_the_element_decoding() {
        let (out, in_dim) = (6usize, 64usize);
        let bytes: Vec<u8> = (0..out * in_dim).map(|i| (i * 37 + 11) as u8).collect();
        for block in [32usize, 2, 1] {
            let sib = in_dim / block;
            let scale: Vec<f32> = (0..out * sib)
                .map(|i| e8m0_to_f32(120 + (i % 9) as u8))
                .collect();
            let want = |i: usize, j: usize| {
                let byte = bytes[i * in_dim / 2 + j / 2];
                let nib = if j.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                e2m1_to_f32(nib) * scale[i * sib + j / block]
            };
            let rows = dequant_fp4_blocked(&bytes[..out * in_dim / 2], &scale, out, in_dim, block);
            let cols =
                dequant_fp4_blocked_columns(&bytes[..out * in_dim / 2], &scale, out, in_dim, block);
            for i in 0..out {
                for j in 0..in_dim {
                    assert_eq!(rows[i * in_dim + j].to_bits(), want(i, j).to_bits());
                    assert_eq!(cols[j * out + i].to_bits(), want(i, j).to_bits());
                }
            }
        }
        let block = 32;
        let (out, in_dim) = (64usize, 64usize);
        let scale: Vec<f32> = (0..4).map(|i| e8m0_to_f32(125 + i as u8)).collect();
        let fp8: Vec<u8> = (0..out * in_dim).map(|i| (i * 13 + 5) as u8).collect();
        let got = dequant_fp8_blocked(&fp8, &scale, out, in_dim, block);
        for i in 0..out {
            for j in 0..in_dim {
                let want = e4m3_to_f32(fp8[i * in_dim + j]) * scale[(i / block) * 2 + j / block];
                assert_eq!(got[i * in_dim + j].to_bits(), want.to_bits());
            }
        }
    }
}
