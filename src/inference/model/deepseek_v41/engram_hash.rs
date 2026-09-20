//! The engram hash parameters derived the way the reference derives them, so a checkpoint that
//! ships only the tables (the released one) hashes exactly as it was trained.
//!
//! The reference draws one multiplier per (layer, look-back) from numpy's default generator
//! seeded per layer, and takes each (n-gram size, head) bucket modulus as the next unused prime
//! above the engram vocabulary size. Both are reproduced here bit for bit: the seed sequence,
//! the PCG64 stream and the bounded draw are numpy's algorithms, checked against numpy's output.

/// numpy `SeedSequence(entropy)` with the default pool of four 32-bit words.
struct SeedSequence {
    pool: [u32; 4],
}

impl SeedSequence {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;

    fn new(entropy: u64) -> Self {
        // The entropy as little-endian 32-bit words, trailing zero word dropped.
        let mut words = vec![entropy as u32, (entropy >> 32) as u32];
        if words[1] == 0 {
            words.pop();
        }
        let mut hash_const = Self::INIT_A;
        let mut hash = |mut value: u32| -> u32 {
            value ^= hash_const;
            hash_const = hash_const.wrapping_mul(Self::MULT_A);
            value = value.wrapping_mul(hash_const);
            value ^= value >> Self::XSHIFT;
            value
        };
        let mix = |x: u32, y: u32| -> u32 {
            let r = Self::MIX_MULT_L
                .wrapping_mul(x)
                .wrapping_sub(Self::MIX_MULT_R.wrapping_mul(y));
            r ^ (r >> Self::XSHIFT)
        };
        let mut pool = [0u32; 4];
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hash(words.get(i).copied().unwrap_or(0));
        }
        for src in 0..4 {
            for dst in 0..4 {
                if src != dst {
                    pool[dst] = mix(pool[dst], hash(pool[src]));
                }
            }
        }
        for &w in words.iter().skip(4) {
            for slot in pool.iter_mut() {
                *slot = mix(*slot, hash(w));
            }
        }
        Self { pool }
    }

    /// `generate_state(n, dtype=uint64)`: 2n hashed 32-bit words, paired little-endian.
    fn state_u64(&self, n: usize) -> Vec<u64> {
        let mut hash_const = Self::INIT_B;
        let mut words = Vec::with_capacity(2 * n);
        for i in 0..2 * n {
            let mut v = self.pool[i % 4];
            v ^= hash_const;
            hash_const = hash_const.wrapping_mul(Self::MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^= v >> Self::XSHIFT;
            words.push(v);
        }
        words
            .chunks(2)
            .map(|w| (w[0] as u64) | ((w[1] as u64) << 32))
            .collect()
    }
}

/// numpy `PCG64` (XSL-RR, 128-bit state), seeded from a `SeedSequence`.
struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    fn seeded(seed: &SeedSequence) -> Self {
        let s = seed.state_u64(4);
        let initstate = ((s[0] as u128) << 64) | s[1] as u128;
        let initseq = ((s[2] as u128) << 64) | s[3] as u128;
        let mut g = Self {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        let rot = (self.state >> 122) as u32;
        (hi ^ lo).rotate_right(rot)
    }

    /// `integers(0, high, dtype=int64)`: Lemire's bounded draw over `[0, high)`.
    fn below(&mut self, high: u64) -> u64 {
        let range = high; // exclusive bound; numpy works on the inclusive `high - 1` plus one
        if range == 0 {
            return 0;
        }
        let mut m = (self.next_u64() as u128) * (range as u128);
        let mut leftover = m as u64;
        if leftover < range {
            let threshold = range.wrapping_neg() % range;
            while leftover < threshold {
                m = (self.next_u64() as u128) * (range as u128);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// The reference's `compute_hash_multipliers`: per engram layer, `max_ngram` odd multipliers
/// bounded so that a compressed id times a multiplier fits in a signed 64-bit product.
pub fn multipliers(
    layer_ids: &[usize],
    max_ngram: usize,
    compressed_vocab: usize,
) -> Vec<Vec<i64>> {
    let bound = ((i64::MAX as u64) / (compressed_vocab as u64).max(1) / 2).max(1);
    layer_ids
        .iter()
        .map(|&l| {
            let mut g = Pcg64::seeded(&SeedSequence::new(10007 * l as u64));
            (0..max_ngram)
                .map(|_| (g.below(bound) as i64) * 2 + 1)
                .collect()
        })
        .collect()
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// The reference's bucket layout: per layer, per n-gram size, per head, the next unused prime
/// above `vocab - 1`; and the offset of each column's bucket range in the layer's table.
pub fn primes_and_offsets(
    n_layers: usize,
    max_ngram: usize,
    n_heads: usize,
    vocab: u64,
) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    let mut seen = std::collections::HashSet::new();
    let mut primes = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let mut layer = Vec::with_capacity((max_ngram - 1) * n_heads);
        for _ in 1..max_ngram {
            let mut current = vocab - 1;
            for _ in 0..n_heads {
                current += 1;
                while !is_prime(current) || seen.contains(&current) {
                    current += 1;
                }
                seen.insert(current);
                layer.push(current);
            }
        }
        primes.push(layer);
    }
    let offsets = primes
        .iter()
        .map(|layer| {
            let mut acc = 0u64;
            layer
                .iter()
                .map(|&p| {
                    let o = acc;
                    acc += p;
                    o
                })
                .collect()
        })
        .collect();
    (primes, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy's own numbers for `SeedSequence(10007)` and `PCG64` seeded from it.
    #[test]
    fn seed_sequence_and_pcg64_match_numpy() {
        let seq = SeedSequence::new(10007);
        assert_eq!(
            seq.state_u64(4),
            vec![
                0x77d32a348ad70d07,
                0x993ed609b84a4fa5,
                0xd5f972dec8698e49,
                0x7d79ebaebcf148de
            ]
        );
        let mut g = Pcg64::seeded(&seq);
        assert_eq!(g.inc, 228559182809729053680923842009152786877);
        assert_eq!(g.state, 78698009995358412790897477648799726887);
        assert_eq!(
            [g.next_u64(), g.next_u64(), g.next_u64()],
            [
                15187255322829266483,
                959186003677246517,
                7126631698937927467
            ]
        );
    }

    /// The released model's multipliers (also carried by a mainline conversion) for its two
    /// engram layers, and the bounded draws numpy makes for the first.
    #[test]
    fn multipliers_match_the_release() {
        let m = multipliers(&[1, 14], 4, 99092);
        assert_eq!(
            m[0],
            vec![
                76632096046245,
                4839876093313,
                35959672319349,
                73987337458391
            ]
        );
        assert_eq!(
            m[1],
            vec![
                67716810739261,
                51510806800915,
                30921347202721,
                82619226485591
            ]
        );
    }

    /// The released layout: consecutive unused primes above 16,000,000, three n-gram sizes of
    /// eight heads per layer, offsets summing to each table's row count.
    #[test]
    fn primes_and_offsets_match_the_release() {
        let (p, o) = primes_and_offsets(2, 4, 8, 16_000_000);
        assert_eq!(
            &p[0][..6],
            &[16000057, 16000079, 16000081, 16000097, 16000121, 16000129]
        );
        assert_eq!(&p[1][21..], &[16000877, 16000879, 16000889]);
        assert_eq!(o[0][1], 16000057);
        assert_eq!(o[1][0], 0);
        assert_eq!(p[0].iter().sum::<u64>(), 384006168);
        assert_eq!(p[1].iter().sum::<u64>(), 384016682);
    }
}
