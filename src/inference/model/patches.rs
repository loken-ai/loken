//! Cutting an image into patches, and putting it back.
//!
//! Every transformer that reads an image starts by cutting it into square patches and
//! flattening each one into a token. The cut is the same everywhere; what differs is the order
//! the values inside a token end up in, and that is not a free choice - it is what the patch
//! projection's weights were trained against, so a checkpoint that expects one and is handed
//! the other produces a plausible image of nothing in particular.
//!
//! Both orders in the tree are here, named after what varies slowest inside a token.

use crate::tensor::{Result, Tensor};

/// The order the values of one patch are flattened in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Channel first: all of one channel's pixels, then the next channel's. FLUX and the SigLIP
    /// towers read their patches this way.
    ByChannel,
    /// Position first: one pixel's channels together, then the next pixel's. Z-Image reads its
    /// patches this way.
    ByPixel,
}

/// `[b, c, h, w]` into `[b, (h/patch).(w/patch), c.patch²]`.
///
/// `h` and `w` must divide by `patch`; a caller that pads does so before getting here, because
/// where the padding goes is a property of the model and not of the cut.
pub fn cut(x: &Tensor, patch: usize, order: Order) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if patch == 0 || h % patch != 0 || w % patch != 0 {
        crate::tensor::bail!("cut: a {h}x{w} image does not divide into {patch}x{patch} patches");
    }
    let (rows, cols) = (h / patch, w / patch);
    let square = x.reshape((b, c, rows, patch, cols, patch))?;
    let ordered = match order {
        // (b, rows, cols, c, ph, pw)
        Order::ByChannel => square.permute((0, 2, 4, 1, 3, 5))?,
        // (b, rows, cols, ph, pw, c)
        Order::ByPixel => square.permute((0, 2, 4, 3, 5, 1))?,
    };
    ordered.reshape((b, rows * cols, c * patch * patch))
}

/// The inverse of [`cut`]: `[b, tokens, c.patch²]` back to `[b, c, rows.patch, cols.patch]`.
pub fn weave(
    x: &Tensor,
    rows: usize,
    cols: usize,
    patch: usize,
    channels: usize,
    order: Order,
) -> Result<Tensor> {
    let (b, tokens, dim) = x.dims3()?;
    if tokens < rows * cols {
        crate::tensor::bail!("weave: {tokens} tokens cannot fill a {rows}x{cols} grid");
    }
    if dim != channels * patch * patch {
        crate::tensor::bail!(
            "weave: a token of {dim} values is not {channels} channels of {patch}x{patch}"
        );
    }
    // A model that padded its sequence to a tile boundary hands back the padding too.
    let x = x.narrow(1, 0, rows * cols)?;
    let (spread, back) = match order {
        Order::ByChannel => (
            x.reshape((b, rows, cols, channels, patch, patch))?,
            [0, 3, 1, 4, 2, 5],
        ),
        Order::ByPixel => (
            x.reshape((b, rows, cols, patch, patch, channels))?,
            [0, 5, 1, 3, 2, 4],
        ),
    };
    spread
        .permute((back[0], back[1], back[2], back[3], back[4], back[5]))?
        .reshape((b, channels, rows * patch, cols * patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn ramp(b: usize, c: usize, h: usize, w: usize) -> Tensor {
        let v: Vec<f32> = (0..b * c * h * w).map(|i| i as f32).collect();
        Tensor::from_vec(v, (b, c, h, w), &Device::Cpu).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Where each value lands, computed from the definition rather than from a permutation.
    ///
    /// A round trip cannot say this: cutting and weaving with the same wrong order gives the
    /// image back unharmed, and the checkpoint is the only thing that would notice. So the cut
    /// is held against an index walked by hand, for both orders, on a picture whose every value
    /// is distinct and whose sides differ.
    #[test]
    fn the_values_of_a_patch_land_where_the_order_says() {
        let (b, c, h, w, patch) = (1usize, 3usize, 2usize, 4usize, 2usize);
        let x = ramp(b, c, h, w);
        let (rows, cols) = (h / patch, w / patch);
        let at = |ci: usize, hi: usize, wi: usize| (ci * h * w + hi * w + wi) as f32;

        let mut by_channel = Vec::new();
        let mut by_pixel = Vec::new();
        for r in 0..rows {
            for col in 0..cols {
                for ci in 0..c {
                    for ph in 0..patch {
                        for pw in 0..patch {
                            by_channel.push(at(ci, r * patch + ph, col * patch + pw));
                        }
                    }
                }
                for ph in 0..patch {
                    for pw in 0..patch {
                        for ci in 0..c {
                            by_pixel.push(at(ci, r * patch + ph, col * patch + pw));
                        }
                    }
                }
            }
        }

        assert_eq!(flat(&cut(&x, patch, Order::ByChannel).unwrap()), by_channel);
        assert_eq!(flat(&cut(&x, patch, Order::ByPixel).unwrap()), by_pixel);
        // And the two orders really are different, or the test above proves nothing.
        assert_ne!(by_channel, by_pixel);
    }

    /// Weaving undoes cutting, in either order and at either patch size.
    #[test]
    fn weaving_puts_back_what_cutting_took_apart() {
        for order in [Order::ByChannel, Order::ByPixel] {
            for (c, h, w, patch) in [(3usize, 4usize, 6usize, 2usize), (2, 6, 6, 3), (1, 2, 2, 2)] {
                let x = ramp(2, c, h, w);
                let tokens = cut(&x, patch, order).unwrap();
                let back = weave(&tokens, h / patch, w / patch, patch, c, order).unwrap();
                assert_eq!(back.dims(), x.dims(), "{order:?} {c}x{h}x{w}/{patch}");
                assert_eq!(flat(&back), flat(&x), "{order:?} {c}x{h}x{w}/{patch}");
            }
        }
    }

    /// A sequence padded past the grid is cut back to it rather than reshaped around it.
    #[test]
    fn weaving_drops_the_padding_a_model_added() {
        let (c, h, w, patch) = (2usize, 4usize, 4usize, 2usize);
        let x = ramp(1, c, h, w);
        let tokens = cut(&x, patch, Order::ByChannel).unwrap();
        let padded = Tensor::cat(&[&tokens, &tokens], 1).unwrap();
        let back = weave(&padded, h / patch, w / patch, patch, c, Order::ByChannel).unwrap();
        assert_eq!(flat(&back), flat(&x));
    }
}
