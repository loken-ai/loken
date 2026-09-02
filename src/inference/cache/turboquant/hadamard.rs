//! Randomized Hadamard rotation: a random sign flip followed by a Walsh-Hadamard
//! transform.
//!
//! Both halves are orthogonal, so the product is orthogonal, and the transform costs
//! `n log n` additions and subtractions - no multiplies on the data, and no matrix. What
//! it buys is outlier spreading: a single large coordinate in the input is smeared across
//! every coordinate of the output, so a group scale is no longer set by whichever channel
//! happened to be large. That is the property a 3-bit scalar quantiser needs to exist at
//! all.
//!
//! For K the rotation costs nothing in accuracy. Attention scores are `q . k`, and an
//! orthogonal `R` leaves a dot product alone: `q . k == (Rq) . (Rk)`. So K is rotated once
//! when it is written and Q is rotated once per step when it is read, and no correction
//! term is ever needed.

use super::rng::split_mix_64;

/// The largest power of two that divides `n`, capped at `n` itself.
///
/// A Walsh-Hadamard transform of length `m` exists when `m` is a power of two. Head
/// dimensions are usually 64 or 128 and this is the whole vector, but 80 and 96 occur, and
/// a rotation that refused to run on those would be a rotation that quietly excludes model
/// families. For those the vector is cut into `n / m` contiguous blocks of the largest
/// power-of-two divisor and each block is transformed on its own. The result is still
/// orthogonal - a block-diagonal of orthogonal blocks is orthogonal - it simply spreads an
/// outlier over `m` coordinates rather than over `n`.
pub fn hadamard_block_len(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    // `n & n.wrapping_neg()` isolates the lowest set bit, which is exactly the largest
    // power of two dividing n.
    n & n.wrapping_neg()
}

/// The unnormalised Walsh-Hadamard butterfly, in place, over one power-of-two block.
///
/// Adds and subtracts only. Applying it twice multiplies the block by its length, which is
/// why the public entry points carry the `1/sqrt(m)` scaling.
fn wht_unnormalised(v: &mut [f32]) {
    let m = v.len();
    debug_assert!(
        m.is_power_of_two(),
        "WHT block length must be a power of two"
    );
    let mut h = 1;
    while h < m {
        let mut i = 0;
        while i < m {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += h << 1;
        }
        h <<= 1;
    }
}

/// A rotation of a fixed length, reproducible from its seed.
///
/// Holds the sign pattern and the block length so that a caller does not re-derive either
/// per vector. The same `(len, seed)` always yields the same rotation, in this process and
/// in the next one.
#[derive(Clone, Debug)]
pub struct HadamardRotation {
    /// Vector length this rotation is built for.
    len: usize,
    /// Length of each independently transformed block; equals `len` when `len` is a power
    /// of two.
    block: usize,
    /// `1 / sqrt(block)`, applied once per pass so that the transform is orthogonal.
    norm: f32,
    /// One sign per coordinate, `+1.0` or `-1.0`.
    signs: Vec<f32>,
}

impl HadamardRotation {
    /// Build the rotation for vectors of `len` coordinates from `seed`.
    pub fn new(len: usize, seed: u64) -> Self {
        let block = hadamard_block_len(len);
        let mut state = seed;
        let mut signs = Vec::with_capacity(len);
        for _ in 0..len {
            // One bit per coordinate, taken from the high end of the word where SplitMix64's
            // avalanche is strongest.
            let bit = split_mix_64(&mut state) >> 63;
            signs.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        Self {
            len,
            block,
            norm: if block == 0 {
                1.0
            } else {
                1.0 / (block as f32).sqrt()
            },
            signs,
        }
    }

    /// The vector length this rotation applies to.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the rotation covers no coordinates at all.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The block length actually transformed - smaller than [`len`](Self::len) only when
    /// `len` is not a power of two.
    pub fn block_len(&self) -> usize {
        self.block
    }

    /// Apply the rotation in place: sign flip, then the normalised transform.
    pub fn apply(&self, v: &mut [f32]) {
        assert_eq!(v.len(), self.len, "rotation applied to the wrong length");
        for (x, s) in v.iter_mut().zip(self.signs.iter()) {
            *x *= *s;
        }
        self.transform_only(v);
    }

    /// Apply the inverse rotation in place: the transform, then the same sign flip.
    ///
    /// Both halves are involutions - a sign flip undoes itself, and the normalised
    /// Walsh-Hadamard transform is its own inverse - so the inverse of `H.D` is `D.H` with
    /// no new tables.
    pub fn apply_inverse(&self, v: &mut [f32]) {
        assert_eq!(v.len(), self.len, "rotation applied to the wrong length");
        self.transform_only(v);
        for (x, s) in v.iter_mut().zip(self.signs.iter()) {
            *x *= *s;
        }
    }

    /// The normalised transform on its own, without the sign flip.
    fn transform_only(&self, v: &mut [f32]) {
        if self.block <= 1 {
            return;
        }
        for chunk in v.chunks_mut(self.block) {
            wht_unnormalised(chunk);
            for x in chunk.iter_mut() {
                *x *= self.norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm2(v: &[f32]) -> f64 {
        v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()
    }

    /// A deterministic pseudo-random vector, so the tests do not depend on a generator.
    fn sample_vector(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                let u = (split_mix_64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                (u * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn turboquant_hadamard_is_its_own_inverse() {
        for &len in &[1usize, 2, 8, 64, 80, 96, 128, 256] {
            let rot = HadamardRotation::new(len, 0xDEAD_BEEF);
            let original = sample_vector(len, 7);
            let mut v = original.clone();
            rot.apply(&mut v);
            rot.apply_inverse(&mut v);
            for (a, b) in v.iter().zip(original.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "len {len}: round trip drifted, {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn turboquant_hadamard_preserves_norm() {
        for &len in &[2usize, 8, 64, 80, 96, 128, 256] {
            let rot = HadamardRotation::new(len, 12345);
            let mut v = sample_vector(len, 99);
            let before = norm2(&v);
            rot.apply(&mut v);
            let after = norm2(&v);
            let rel = (after - before).abs() / before;
            assert!(rel < 1e-6, "len {len}: norm changed by {rel} relative");
        }
    }

    #[test]
    fn turboquant_hadamard_preserves_dot_products() {
        // The property the K path rests on: q.k survives the rotation, so no correction
        // term is needed anywhere.
        for &len in &[64usize, 96, 128] {
            let rot = HadamardRotation::new(len, 4242);
            let q = sample_vector(len, 1);
            let k = sample_vector(len, 2);
            let before: f64 = q
                .iter()
                .zip(k.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let mut qr = q.clone();
            let mut kr = k.clone();
            rot.apply(&mut qr);
            rot.apply(&mut kr);
            let after: f64 = qr
                .iter()
                .zip(kr.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            assert!(
                (after - before).abs() < 1e-4 * before.abs().max(1.0),
                "len {len}: dot product moved, {before} vs {after}"
            );
        }
    }

    #[test]
    fn turboquant_rotation_is_reproducible_from_its_seed() {
        // The cache is written by one call and read by another. Two rotations built from
        // the same seed must be the same rotation, or a stored K and a live Q do not meet.
        let a = HadamardRotation::new(128, 0x5EED);
        let b = HadamardRotation::new(128, 0x5EED);
        let c = HadamardRotation::new(128, 0x5EED + 1);
        let src = sample_vector(128, 3);
        let (mut va, mut vb, mut vc) = (src.clone(), src.clone(), src.clone());
        a.apply(&mut va);
        b.apply(&mut vb);
        c.apply(&mut vc);
        assert_eq!(va, vb, "same seed produced a different rotation");
        assert_ne!(va, vc, "different seeds produced the same rotation");
    }

    #[test]
    fn turboquant_hadamard_spreads_a_lone_outlier() {
        // The reason the rotation is here at all. One coordinate a hundred times the rest
        // sets the group scale for everybody; after rotation no coordinate dominates.
        let len = 128;
        let rot = HadamardRotation::new(len, 777);
        let mut v = vec![1.0f32; len];
        v[13] = 100.0;
        let peak_before = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let rms_before = (norm2(&v) / len as f64).sqrt() as f32;
        rot.apply(&mut v);
        let peak_after = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let rms_after = (norm2(&v) / len as f64).sqrt() as f32;
        let crest_before = peak_before / rms_before;
        let crest_after = peak_after / rms_after;
        assert!(
            crest_after < crest_before / 4.0,
            "rotation did not flatten the outlier: crest {crest_before} -> {crest_after}"
        );
    }

    #[test]
    fn turboquant_block_len_matches_the_largest_power_of_two_divisor() {
        assert_eq!(hadamard_block_len(128), 128);
        assert_eq!(hadamard_block_len(64), 64);
        assert_eq!(hadamard_block_len(96), 32);
        assert_eq!(hadamard_block_len(80), 16);
        assert_eq!(hadamard_block_len(1), 1);
    }
}
