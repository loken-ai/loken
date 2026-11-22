//! Multiplying the rows of a quantised weight by powers of two through their scale fields.
//!
//! A block format holds each value as codes times float scale fields.
//! Multiplying every scale field of a block by `2^e` multiplies every value of the block by
//! `2^e`, exactly, while each field stays a normal number of its width.
//! A dense format multiplies each value.
//! The codes never change.

use super::GgmlDType;
use half::{bf16, f16};

/// Where a block keeps its float scale fields.
enum Fields {
    /// f16 fields at these byte offsets of the block.
    Half(&'static [usize]),
    /// An f32 field at this byte offset.
    Single(usize),
    /// An E8M0 exponent byte at this offset: the scale is `2^(byte - 127)`.
    Exponent(usize),
    /// Every value is an f32.
    DenseF32,
    /// Every value is an f16.
    DenseF16,
    /// Every value is a bf16.
    DenseBf16,
}

/// The scale fields of `dtype`.
/// `None` for a format whose blocks carry none, or whose scales are not whole f16, f32 or E8M0
/// fields: IQ1_M splits its f16 across 4 words and NVFP4 holds E4M3 scales.
fn fields(dtype: GgmlDType) -> Option<Fields> {
    use GgmlDType::*;
    Some(match dtype {
        F32 => Fields::DenseF32,
        F16 => Fields::DenseF16,
        BF16 => Fields::DenseBf16,
        Q4_0 | Q5_0 | Q8_0 | Q2_0 | Iq4Nl | Iq4Xs | Iq2Xxs | Iq2S | Iq3Xxs | Iq3S | Iq1S => {
            Fields::Half(&[0])
        }
        Tq1_0 => Fields::Half(&[52]),
        Tq2_0 => Fields::Half(&[64]),
        Q4_1 | Q5_1 | Q8_1 | Q4K | Q5K => Fields::Half(&[0, 2]),
        Q2K => Fields::Half(&[80, 82]),
        Q3K => Fields::Half(&[108]),
        Q6K => Fields::Half(&[208]),
        Q8K => Fields::Single(0),
        MxFp4 => Fields::Exponent(0),
        I8 | Iq1M | NvFp4 => return None,
    })
}

/// `v` times `2^e` when the product is an f16 value, exactly, `Some(v)` for a zero.
/// A subnormal value or product is kept when no bit is lost.
fn half_times(v: f16, e: i32) -> Option<f16> {
    if v.to_f32() == 0.0 {
        return Some(v);
    }
    let want = v.to_f64() * 2f64.powi(e);
    let out = f16::from_f64(want);
    (v.is_finite() && out.is_finite() && out.to_f64() == want).then_some(out)
}

/// `v` times `2^e` when both are normal bf16 values, `Some(v)` for a zero.
fn bf16_times(v: bf16, e: i32) -> Option<bf16> {
    if v.to_f32() == 0.0 {
        return Some(v);
    }
    let out = bf16::from_f64(v.to_f64() * 2f64.powi(e));
    (v.is_normal() && out.is_normal()).then_some(out)
}

/// `v` times `2^e` when both are normal f32 values, `Some(v)` for a zero.
fn single_times(v: f32, e: i32) -> Option<f32> {
    if v == 0.0 {
        return Some(v);
    }
    let out = (f64::from(v) * 2f64.powi(e)) as f32;
    (v.is_normal() && out.is_normal()).then_some(out)
}

/// Multiplies the 2-byte value at `at` of `bytes` by `2^e` through `times`.
fn scale_two<T>(
    bytes: &mut [u8],
    at: usize,
    e: i32,
    read: fn([u8; 2]) -> T,
    write: fn(T) -> [u8; 2],
    times: fn(T, i32) -> Option<T>,
) -> bool {
    let v = read([bytes[at], bytes[at + 1]]);
    match times(v, e) {
        Some(out) => {
            bytes[at..at + 2].copy_from_slice(&write(out));
            true
        }
        None => false,
    }
}

/// Multiplies one row of `dtype` by `2^e`, in place.
///
/// `false` when a field of the row cannot take the factor exactly: a subnormal field, or one
/// the factor would take out of the normal range.
/// The row may then be partly scaled: the caller discards it.
fn scale_row(dtype: GgmlDType, row: &mut [u8], e: i32) -> bool {
    if e == 0 {
        return true;
    }
    let Some(kind) = fields(dtype) else {
        return false;
    };
    let f16_read = |b: [u8; 2]| f16::from_le_bytes(b);
    match kind {
        Fields::DenseF32 => row.chunks_exact_mut(4).all(|c| {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            single_times(v, e).is_some_and(|out| {
                c.copy_from_slice(&out.to_le_bytes());
                true
            })
        }),
        Fields::DenseF16 => (0..row.len() / 2)
            .all(|i| scale_two(row, 2 * i, e, f16_read, f16::to_le_bytes, half_times)),
        Fields::DenseBf16 => (0..row.len() / 2).all(|i| {
            scale_two(
                row,
                2 * i,
                e,
                bf16::from_le_bytes,
                bf16::to_le_bytes,
                bf16_times,
            )
        }),
        Fields::Half(offsets) => row.chunks_exact_mut(dtype.type_size()).all(|block| {
            offsets
                .iter()
                .all(|&at| scale_two(block, at, e, f16_read, f16::to_le_bytes, half_times))
        }),
        Fields::Single(at) => row.chunks_exact_mut(dtype.type_size()).all(|block| {
            let v = f32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
            single_times(v, e).is_some_and(|out| {
                block[at..at + 4].copy_from_slice(&out.to_le_bytes());
                true
            })
        }),
        Fields::Exponent(at) => row.chunks_exact_mut(dtype.type_size()).all(|block| {
            // 255 is not a scale; every other byte is 2^(byte - 127).
            let b = i32::from(block[at]);
            let out = b + e;
            if b == 255 || !(0..=254).contains(&out) {
                return false;
            }
            block[at] = out as u8;
            true
        }),
    }
}

/// `bytes`, rows of `row_len` values of `dtype`, with row `r` multiplied by `2^exponents[r]`.
///
/// `None` when a row cannot take its factor exactly, or when the bytes do not hold
/// `exponents.len()` whole rows of whole blocks.
pub fn scale_rows_pow2(
    dtype: GgmlDType,
    bytes: &[u8],
    row_len: usize,
    exponents: &[i32],
) -> Option<Vec<u8>> {
    let block = dtype.block_size();
    if block == 0 || !row_len.is_multiple_of(block) {
        return None;
    }
    let row_bytes = row_len / block * dtype.type_size();
    if bytes.len() != row_bytes * exponents.len() {
        return None;
    }
    let mut out = bytes.to_vec();
    out.chunks_exact_mut(row_bytes)
        .zip(exponents)
        .all(|(row, &e)| scale_row(dtype, row, e))
        .then_some(out)
}

#[cfg(test)]
mod tests {
    use super::super::{from_float_bytes, to_float_bytes};
    use super::*;

    /// Values of either sign over a few binades, their scale fields far from the subnormal range.
    fn values(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(97);
                (((h >> 8) % 2001) as f32 / 1000.0 - 1.0) * 16.0
            })
            .collect()
    }

    /// Every format the host encodes: each row scaled by its power of two dequantises to the
    /// plain row's values times that power, bit for bit.
    #[test]
    fn a_scaled_row_dequantises_to_the_row_times_its_power_bit_for_bit() {
        use GgmlDType::*;
        let formats = [
            F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, Iq4Nl,
            Iq4Xs, Iq2Xxs, Iq2S, Iq3Xxs, Iq3S, Q2_0, MxFp4, Iq1S, Tq1_0, Tq2_0,
        ];
        let (rows, row_len) = (6usize, 512usize);
        let exponents = [-3, -1, 0, 1, 2, 4];
        let xs = values(rows * row_len);
        let mut tested = 0;
        for dtype in formats {
            let Ok(bytes) = from_float_bytes(dtype, &xs) else {
                continue;
            };
            tested += 1;
            let scaled = scale_rows_pow2(dtype, &bytes, row_len, &exponents)
                .unwrap_or_else(|| panic!("{dtype:?}: the rows refuse their factors"));
            let (mut plain, mut out) = (vec![0f32; xs.len()], vec![0f32; xs.len()]);
            to_float_bytes(dtype, &bytes, &mut plain).unwrap();
            to_float_bytes(dtype, &scaled, &mut out).unwrap();
            for (r, &e) in exponents.iter().enumerate() {
                for c in 0..row_len {
                    let i = r * row_len + c;
                    let want = plain[i] * 2f32.powi(e);
                    assert_eq!(
                        out[i].to_bits(),
                        want.to_bits(),
                        "{dtype:?} row {r} column {c}: {} against {want}",
                        out[i]
                    );
                }
            }
        }
        assert!(tested >= 20, "only {tested} formats encoded");
    }

    /// A factor that takes a scale field out of the normal range is refused.
    #[test]
    fn a_factor_past_the_range_of_a_field_is_refused() {
        let xs = values(64);
        let bytes = from_float_bytes(GgmlDType::Q8_0, &xs).unwrap();
        assert!(scale_rows_pow2(GgmlDType::Q8_0, &bytes, 64, &[40]).is_none());
        assert!(scale_rows_pow2(GgmlDType::Q8_0, &bytes, 64, &[-40]).is_none());
        assert!(scale_rows_pow2(GgmlDType::I8, &[0u8; 64], 64, &[1]).is_none());
        assert!(scale_rows_pow2(GgmlDType::Q8_0, &bytes, 64, &[0]).is_some());
    }
}
