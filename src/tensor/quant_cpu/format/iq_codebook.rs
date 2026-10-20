//! What a grid encoder needs of its grid, for the formats whose codes name points of `D` levels.
//!
//! A group of `D` values rounds to one level each; the levels form a pattern, `BITS` bits a level.
//! The pattern names a point when the grid holds it. When it does not, the encoder searches the
//! points nearest it: those at the `DISTANCES` least squared distances, in order of distance and
//! then of index - ggml's neighbour lists (MIT), built the same way.

use super::*;

/// A grid as the encoder reads it.
pub(super) struct Codebook<const D: usize> {
    /// Each point's levels.
    pub levels: Vec<[u8; D]>,
    /// Each point's stored magnitudes, what the format's reader multiplies by its scale.
    pub bytes: Vec<[u8; D]>,
    /// The point a pattern names, or -1.
    pub index: Vec<i16>,
    /// For a pattern the grid does not hold and the encoder can reach, the points nearest it.
    pub near: Vec<Vec<u16>>,
    /// Bits per level in a pattern.
    pub bits: u32,
}

impl<const D: usize> Codebook<D> {
    /// The codebook of a grid whose points have `bytes` as magnitudes and `levels` as levels.
    /// Patterns with a level at or past `reach` are never searched and get no list.
    pub fn new(
        bytes: Vec<[u8; D]>,
        levels: Vec<[u8; D]>,
        bits: u32,
        reach: u8,
        distances: usize,
    ) -> Self {
        let patterns = 1usize << (bits as usize * D);
        let mut index = vec![-1i16; patterns];
        let pattern = |l: &[u8; D]| {
            l.iter()
                .enumerate()
                .fold(0usize, |p, (i, &v)| p | (v as usize) << (bits as usize * i))
        };
        for (n, l) in levels.iter().enumerate() {
            index[pattern(l)] = n as i16;
        }
        let mask = (1usize << bits) - 1;
        let near = (0..patterns)
            .map(|p| {
                let at: [u8; D] =
                    std::array::from_fn(|i| ((p >> (bits as usize * i)) & mask) as u8);
                if index[p] >= 0 || at.iter().any(|&l| l >= reach) {
                    return Vec::new();
                }
                let mut by_distance: Vec<(i32, u16)> = levels
                    .iter()
                    .enumerate()
                    .map(|(n, l)| {
                        let d = (0..D).map(|i| (l[i] as i32 - at[i] as i32).pow(2)).sum();
                        (d, n as u16)
                    })
                    .collect();
                by_distance.sort_unstable();
                let (mut distinct, mut last) = (0, -1);
                by_distance
                    .into_iter()
                    .take_while(|&(d, _)| {
                        if d != last {
                            distinct += 1;
                            last = d;
                        }
                        distinct <= distances
                    })
                    .map(|(_, n)| n)
                    .collect()
            })
            .collect();
        Self {
            levels,
            bytes,
            index,
            near,
            bits,
        }
    }

    /// The pattern `levels` form.
    pub fn pattern(&self, levels: &[u8]) -> usize {
        levels
            .iter()
            .enumerate()
            .fold(0, |p, (i, &l)| p | (l as usize) << (self.bits as usize * i))
    }

    /// The point among the pattern's near points that leaves `x` the least error at `scale`, each
    /// value's error weighed by `w`; a point's value is `2 l + 1` steps of the scale.
    pub fn nearest(&self, p: usize, x: &[f32], w: &[f32], scale: f32) -> usize {
        let (mut at, mut best) = (usize::MAX, f32::MAX);
        for &n in &self.near[p] {
            let l = &self.levels[n as usize];
            let mut err = 0f32;
            for i in 0..D {
                let d = scale * (2 * l[i] as i32 + 1) as f32 - x[i];
                err += w[i] * d * d;
            }
            if err < best {
                (at, best) = (n as usize, err);
            }
        }
        at
    }

    /// The point a group of values takes at `scale`, `inverse` being the step its levels are
    /// rounded at: the point its rounded levels name, or the nearest the grid holds. The second
    /// value says whether the grid held the rounded pattern.
    pub fn place(
        &self,
        x: &[f32],
        w: &[f32],
        (inverse, scale): (f32, f32),
        top: u8,
    ) -> (usize, bool) {
        let l: [u8; D] = std::array::from_fn(|i| {
            nearest_int(0.5 * (inverse * x[i] - 1.0)).clamp(0, top as i32) as u8
        });
        let p = self.pattern(&l);
        match self.index[p] {
            n if n >= 0 => (n as usize, true),
            _ => (self.nearest(p, x, w, scale), false),
        }
    }

    /// The codebook as one table, for a card that runs the same search: the point each pattern
    /// names or -1, an offset per pattern into the lists of near points and one past the last,
    /// each point's levels, each point's magnitudes, then the lists.
    pub fn table(&self) -> Vec<i32> {
        let mut out: Vec<i32> = self.index.iter().map(|&i| i as i32).collect();
        let mut at = 0i32;
        for near in &self.near {
            out.push(at);
            at += near.len() as i32;
        }
        out.push(at);
        out.extend(self.levels.iter().flatten().map(|&l| l as i32));
        out.extend(self.bytes.iter().flatten().map(|&b| b as i32));
        out.extend(self.near.iter().flatten().map(|&n| n as i32));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every point is its own pattern, and a pattern has near points exactly when the grid does
    /// not hold it and the encoder can reach it.
    fn holds<const D: usize>(book: &Codebook<D>, reach: u8) {
        for (n, l) in book.levels.iter().enumerate() {
            assert_eq!(book.index[book.pattern(l)], n as i16);
        }
        assert_eq!(
            book.index.iter().filter(|&&i| i >= 0).count(),
            book.levels.len()
        );
        let mask = (1usize << book.bits) - 1;
        for (p, near) in book.near.iter().enumerate() {
            let reachable =
                (0..D).all(|i| ((p >> (book.bits as usize * i)) & mask) < reach as usize);
            assert_eq!(
                near.is_empty(),
                book.index[p] >= 0 || !reachable,
                "pattern {p}"
            );
        }
    }

    #[test]
    fn the_grid_codebooks_name_their_points() {
        holds(super::super::iq3_s::codebook(), 8);
        holds(super::super::iq2_s::codebook(), 3);
    }
}
