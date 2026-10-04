//! The quantised cache rows in the form they came from.
//!
//! The decode state holds its KV as f32, but every value is a point of a small grid times a block
//! scale: the window rows are e4m3 with a power-of-two scale per block of 32, the compressed KV is
//! e2m1 with an f32 scale per block of 16, and the index keys are e2m1 with a power-of-two scale
//! per block of 32. These functions write such rows as codes plus scales and read them back to
//! the exact f32 bits.
//!
//! Each `pack_*` verifies its own result: it unpacks what it packed and compares every bit, and
//! gives `None` for any array that is not on the grid it was asked for, so a caller falls back to
//! raw f32 and the round trip stays exact whatever the data was.

const FP4_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP4_MAX: f32 = 6.0;
const FP8_MAX: f32 = 448.0;

/// The 127 finite non-negative e4m3 values by code.
fn e4m3_table() -> &'static [f32; 127] {
    static T: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 127];
        for (code, v) in t.iter_mut().enumerate() {
            let (e, m) = ((code >> 3) as i32, (code & 7) as f32);
            *v = if e == 0 {
                m * 2f32.powi(-9)
            } else {
                (1.0 + m / 8.0) * 2f32.powi(e - 7)
            };
        }
        t
    })
}

fn pow2_ceil(x: f32) -> f32 {
    (x.log2().ceil()).exp2()
}

fn exponent_of(scale: f32) -> Option<i8> {
    let e = scale.log2();
    (e.fract() == 0.0 && (-128.0..=127.0).contains(&e)).then_some(e as i8)
}

/// Bytes `pack_fp8` writes for `n` values.
pub fn fp8_len(n: usize) -> usize {
    n / 32 * 33
}

/// Bytes `pack_fp4` writes for `n` values in blocks of `block`.
pub fn fp4_len(n: usize, block: usize, pow2: bool) -> usize {
    n / block * (block / 2 + if pow2 { 1 } else { 4 })
}

/// e4m3 values with a power-of-two scale per block of 32: one exponent byte then 32 codes
/// (sign in the top bit) per block.
pub fn pack_fp8(values: &[f32]) -> Option<Vec<u8>> {
    if !values.len().is_multiple_of(32) {
        return None;
    }
    let table = e4m3_table();
    let mut out = Vec::with_capacity(fp8_len(values.len()));
    for block in values.chunks_exact(32) {
        let amax = block.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if !amax.is_finite() {
            return None;
        }
        let scale = if amax == 0.0 {
            1.0
        } else {
            pow2_ceil(amax / FP8_MAX)
        };
        out.push(exponent_of(scale)? as u8);
        for &v in block {
            let mag = v.abs() / scale;
            let code = table.partition_point(|&t| t < mag);
            if code >= table.len() || table[code] != mag {
                return None;
            }
            out.push(code as u8 | if v.is_sign_negative() { 0x80 } else { 0 });
        }
    }
    (unpack_fp8(&out, values.len())
        .iter()
        .zip(values)
        .all(|(a, b)| a.to_bits() == b.to_bits()))
    .then_some(out)
}

/// The values `pack_fp8` wrote. `bytes` must be `fp8_len(n)` long.
pub fn unpack_fp8(bytes: &[u8], n: usize) -> Vec<f32> {
    let table = e4m3_table();
    let mut out = Vec::with_capacity(n);
    for block in bytes.chunks_exact(33) {
        let scale = f32::from(block[0] as i8).exp2();
        for &c in &block[1..] {
            let m = table[(c & 0x7F).min(126) as usize] * scale;
            out.push(if c & 0x80 != 0 { -m } else { m });
        }
    }
    out
}

/// e2m1 values in blocks of `block` (even): a scale then one nibble per value (sign in the top
/// bit, grid index below). The scale is an exponent byte when `pow2`, the f32 bits otherwise.
pub fn pack_fp4(values: &[f32], block: usize, pow2: bool) -> Option<Vec<u8>> {
    if block == 0 || !block.is_multiple_of(2) || !values.len().is_multiple_of(block) {
        return None;
    }
    let mut out = Vec::with_capacity(fp4_len(values.len(), block, pow2));
    for blk in values.chunks_exact(block) {
        let mut best_mag = 0f32;
        for &v in blk {
            if !v.is_finite() {
                return None;
            }
            best_mag = best_mag.max(v.abs());
        }
        let scale = if best_mag == 0.0 {
            Some(if pow2 { 1.0 } else { 0.0 })
        } else if pow2 {
            // The scale the quantiser would pick for a block whose largest value sits on 6, or
            // one binade lower where the largest rounded to 4 or 3.
            let s = pow2_ceil(best_mag / FP4_MAX);
            [s, s * 2.0, s * 0.5]
                .into_iter()
                .find(|&c| block_fits(blk, c))
        } else {
            // The largest value is some grid point times the scale; try each as the source.
            FP4_GRID[1..].iter().rev().find_map(|&g| {
                let s = best_mag / g;
                [0i32, 1, -1, 2, -2]
                    .into_iter()
                    .map(|d| f32::from_bits((s.to_bits() as i64 + d as i64) as u32))
                    .find(|&c| c.is_finite() && block_fits(blk, c))
            })
        };
        let scale = scale?;
        if pow2 {
            out.push(exponent_of(scale)? as u8);
        } else {
            out.extend_from_slice(&scale.to_bits().to_le_bytes());
        }
        for pair in blk.chunks_exact(2) {
            let nib = |v: f32| -> u8 {
                let idx = FP4_GRID
                    .iter()
                    .position(|&g| (g.copysign(v) * scale).to_bits() == v.to_bits())
                    .unwrap_or(0) as u8;
                idx | if v.is_sign_negative() { 8 } else { 0 }
            };
            out.push(nib(pair[0]) | nib(pair[1]) << 4);
        }
    }
    (unpack_fp4(&out, values.len(), block, pow2)
        .iter()
        .zip(values)
        .all(|(a, b)| a.to_bits() == b.to_bits()))
    .then_some(out)
}

/// Whether every value of `blk` is a grid point times `scale`, sign of zero included.
fn block_fits(blk: &[f32], scale: f32) -> bool {
    blk.iter().all(|&v| {
        FP4_GRID
            .iter()
            .any(|&g| (g.copysign(v) * scale).to_bits() == v.to_bits())
    })
}

/// The values `pack_fp4` wrote.
pub fn unpack_fp4(bytes: &[u8], n: usize, block: usize, pow2: bool) -> Vec<f32> {
    let head = if pow2 { 1 } else { 4 };
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(head + block / 2) {
        let scale = if pow2 {
            f32::from(chunk[0] as i8).exp2()
        } else {
            f32::from_bits(u32::from_le_bytes(chunk[..4].try_into().unwrap()))
        };
        for &b in &chunk[head..] {
            for nib in [b & 15, b >> 4] {
                let m = FP4_GRID[(nib & 7) as usize] * scale;
                out.push(if nib & 8 != 0 { -m } else { m });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model::deepseek_v41::attention::act_quant_fp8_e4m3;
    use crate::inference::model::deepseek_v41::band::fp4_act_quant;
    use crate::tensor::{Device, Tensor};

    /// Values with a spread of magnitudes, zeros of both signs and a few exact grid points.
    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                match i % 37 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => u * 1e-7,
                    3 => u * 3000.0,
                    _ => u * 4.0 * (1.0 + (i % 7) as f32),
                }
            })
            .collect()
    }

    fn quant(f: impl Fn(&Tensor) -> crate::tensor::Result<Tensor>, v: &[f32]) -> Vec<f32> {
        let t = Tensor::from_vec(v.to_vec(), (v.len() / 64, 64), &Device::Cpu).unwrap();
        f(&t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn the_quantisers_outputs_pack_and_come_back_to_the_bit() {
        for seed in 0..40 {
            let raw = data(64 * 24, seed);
            let q8 = quant(|t| act_quant_fp8_e4m3(t, 32), &raw);
            let p = pack_fp8(&q8).expect("fp8 rows are on the grid");
            assert_eq!(p.len(), fp8_len(q8.len()));
            assert!(same(&unpack_fp8(&p, q8.len()), &q8), "fp8 seed {seed}");
            for (block, pow2) in [(16, false), (32, true)] {
                let q4 = quant(|t| fp4_act_quant(t, block, pow2), &raw);
                let p = pack_fp4(&q4, block, pow2)
                    .unwrap_or_else(|| panic!("fp4 block {block} pow2 {pow2} seed {seed}"));
                assert_eq!(p.len(), fp4_len(q4.len(), block, pow2));
                assert!(same(&unpack_fp4(&p, q4.len(), block, pow2), &q4));
            }
        }
    }

    #[test]
    fn values_off_the_grid_are_refused_not_rounded() {
        let raw = data(64 * 4, 3);
        assert!(pack_fp8(&raw).is_none());
        assert!(pack_fp4(&raw, 16, false).is_none());
        assert!(pack_fp4(&raw, 32, true).is_none());
        assert!(pack_fp8(&raw[..31]).is_none());
        assert!(pack_fp4(&[f32::NAN; 16], 16, false).is_none());
    }

    #[test]
    fn a_packed_row_costs_a_fraction_of_the_floats() {
        assert_eq!(fp8_len(512), 528);
        assert_eq!(fp4_len(512, 16, false), 384);
        assert_eq!(fp4_len(128, 32, true), 68);
    }
}
