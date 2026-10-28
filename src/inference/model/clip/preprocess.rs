//! The image preprocessing a Hugging Face `preprocessor_config.json` states: resize, centre
//! crop, rescale and normalise, in that order, with the resampling filter it names.
//!
//! The resize is Pillow's: a separable convolution, horizontal pass then vertical, each
//! rounded back to 8 bits with Pillow's fixed-point coefficients, so the pixels the tower
//! reads are the ones the reference processor hands its model.

use serde_json::Value;

/// How the image is resized before the crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resize {
    /// The shorter side to this many pixels, the longer one in proportion (truncated).
    ShortestEdge(usize),
    /// Exactly `height` by `width`.
    Exact { height: usize, width: usize },
}

/// Pillow's resampling filters, by the number a config names them with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Lanczos,
    Bilinear,
    Bicubic,
    Box,
    Hamming,
}

impl Filter {
    /// Pillow's `Image.Resampling` value.
    fn from_pil(n: u64) -> Result<Self, String> {
        Ok(match n {
            0 => Self::Nearest,
            1 => Self::Lanczos,
            2 => Self::Bilinear,
            3 => Self::Bicubic,
            4 => Self::Box,
            5 => Self::Hamming,
            other => return Err(format!("resampling filter {other} is not one of Pillow's")),
        })
    }

    /// Half the width of the kernel at unit scale.
    fn support(self) -> f64 {
        match self {
            Self::Nearest | Self::Box => 0.5,
            Self::Bilinear | Self::Hamming => 1.0,
            Self::Bicubic => 2.0,
            Self::Lanczos => 3.0,
        }
    }

    /// The kernel at `x`, as Pillow defines it.
    fn weight(self, x: f64) -> f64 {
        use std::f64::consts::PI;
        /// The free parameter of Pillow's cubic kernel.
        const CUBIC_A: f64 = -0.5;
        let sinc = |x: f64| {
            if x == 0.0 {
                1.0
            } else {
                (x * PI).sin() / (x * PI)
            }
        };
        match self {
            Self::Nearest | Self::Box => {
                if x > -0.5 && x <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Self::Hamming => {
                let x = x.abs();
                if x == 0.0 {
                    1.0
                } else if x >= 1.0 {
                    0.0
                } else {
                    let x = x * PI;
                    x.sin() / x * (0.54 + 0.46 * x.cos())
                }
            }
            Self::Bicubic => {
                let x = x.abs();
                if x < 1.0 {
                    ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * CUBIC_A
                } else {
                    0.0
                }
            }
            Self::Lanczos => {
                if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Fractional bits of Pillow's 8-bit resampling coefficients.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// The transformers default when a config names no `rescale_factor`.
const DEFAULT_RESCALE: f64 = 1.0 / 255.0;

/// One output sample of a resampling pass: the first source sample it reads and the
/// fixed-point weights of the samples from there.
struct Tap {
    first: usize,
    weights: Vec<i32>,
}

/// Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc` for `input` samples to `output`.
fn taps(input: usize, output: usize, filter: Filter) -> Vec<Tap> {
    let scale = input as f64 / output as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ss = 1.0 / filterscale;
    let one = f64::from(1u32 << PRECISION_BITS);
    (0..output)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            // C's `(int)` truncates toward zero, as `as` does.
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let raw: Vec<f64> = (first..end)
                .map(|x| filter.weight((x as f64 - center + 0.5) * ss))
                .collect();
            let total: f64 = raw.iter().sum();
            let weights = raw
                .iter()
                .map(|w| {
                    let w = if total != 0.0 { w / total } else { *w };
                    if w < 0.0 {
                        (-0.5 + w * one) as i32
                    } else {
                        (0.5 + w * one) as i32
                    }
                })
                .collect();
            Tap { first, weights }
        })
        .collect()
}

/// A fixed-point sum back to 8 bits, rounded and clamped as Pillow's `clip8` does.
fn clip8(sum: i64) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Interleaved RGB8 `w` x `h` resized to `ow` x `oh` with Pillow's arithmetic.
pub fn resample(
    rgb: &[u8],
    (w, h): (usize, usize),
    (ow, oh): (usize, usize),
    filter: Filter,
) -> Vec<u8> {
    if filter == Filter::Nearest {
        // Pillow resizes NEAREST through an affine transform: the source sample under each
        // output sample's centre.
        let mut out = Vec::with_capacity(ow * oh * 3);
        for y in 0..oh {
            let sy = (((y as f64 + 0.5) * h as f64 / oh as f64) as usize).min(h - 1);
            for x in 0..ow {
                let sx = (((x as f64 + 0.5) * w as f64 / ow as f64) as usize).min(w - 1);
                let s = (sy * w + sx) * 3;
                out.extend_from_slice(&rgb[s..s + 3]);
            }
        }
        return out;
    }
    let round = 1i64 << (PRECISION_BITS - 1);
    // Horizontal pass first, skipped when the width is kept, as Pillow skips it.
    let wide = if ow == w {
        rgb.to_vec()
    } else {
        let taps = taps(w, ow, filter);
        let mut out = vec![0u8; ow * h * 3];
        for y in 0..h {
            let row = &rgb[y * w * 3..(y + 1) * w * 3];
            for (x, tap) in taps.iter().enumerate() {
                for c in 0..3 {
                    let sum = tap.weights.iter().enumerate().fold(round, |acc, (k, &wk)| {
                        acc + i64::from(row[(tap.first + k) * 3 + c]) * i64::from(wk)
                    });
                    out[(y * ow + x) * 3 + c] = clip8(sum);
                }
            }
        }
        out
    };
    if oh == h {
        return wide;
    }
    let taps = taps(h, oh, filter);
    let mut out = vec![0u8; ow * oh * 3];
    for (y, tap) in taps.iter().enumerate() {
        for x in 0..ow {
            for c in 0..3 {
                let sum = tap.weights.iter().enumerate().fold(round, |acc, (k, &wk)| {
                    acc + i64::from(wide[((tap.first + k) * ow + x) * 3 + c]) * i64::from(wk)
                });
                out[(y * ow + x) * 3 + c] = clip8(sum);
            }
        }
    }
    out
}

/// What a `preprocessor_config.json` asks of an image.
#[derive(Debug, Clone, PartialEq)]
pub struct Preprocess {
    resize: Option<Resize>,
    filter: Filter,
    /// `(height, width)`.
    crop: Option<(usize, usize)>,
    rescale: Option<f64>,
    normalize: Option<([f32; 3], [f32; 3])>,
}

/// One whole-number side of a size object.
fn side(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|n| n as usize)
}

/// Three channel statistics, or one for all three.
fn channels(v: &Value) -> Option<[f32; 3]> {
    match v {
        Value::Array(a) if a.len() == 3 => {
            let f = |i: usize| a[i].as_f64().map(|x| x as f32);
            Some([f(0)?, f(1)?, f(2)?])
        }
        Value::Number(n) => n.as_f64().map(|x| [x as f32; 3]),
        _ => None,
    }
}

impl Preprocess {
    /// Read from the config's JSON. A step runs when its `do_*` flag says so; a missing flag
    /// counts as set when the step's parameters are there. An integer `size` is the shortest
    /// edge and an integer `crop_size` a square, as CLIP's processor reads them.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let flag =
            |key: &str, present: bool| v.get(key).and_then(Value::as_bool).unwrap_or(present);
        let size = v.get("size");
        let resize = match size {
            Some(s) if flag("do_resize", true) => Some(match s {
                Value::Number(n) => {
                    Resize::ShortestEdge(n.as_u64().ok_or("`size` must be a whole number")? as usize)
                }
                Value::Object(_) => match (
                    side(s, "height"),
                    side(s, "width"),
                    side(s, "shortest_edge"),
                ) {
                    (Some(height), Some(width), _) => Resize::Exact { height, width },
                    (_, _, Some(edge)) if s.get("longest_edge").is_none() => {
                        Resize::ShortestEdge(edge)
                    }
                    _ => return Err(format!("`size` {s} is not a shape this server resizes to")),
                },
                other => return Err(format!("`size` {other} is not a shape")),
            }),
            _ => None,
        };
        let filter = match v.get("resample").and_then(Value::as_u64) {
            Some(n) => Filter::from_pil(n)?,
            None => Filter::Bilinear,
        };
        let crop = match v.get("crop_size") {
            Some(c) if flag("do_center_crop", true) => Some(match c {
                Value::Number(n) => {
                    let s = n.as_u64().ok_or("`crop_size` must be a whole number")? as usize;
                    (s, s)
                }
                _ => (
                    side(c, "height").ok_or("`crop_size` names no height")?,
                    side(c, "width").ok_or("`crop_size` names no width")?,
                ),
            }),
            _ => None,
        };
        let rescale = flag("do_rescale", true).then(|| {
            v.get("rescale_factor")
                .and_then(Value::as_f64)
                .unwrap_or(DEFAULT_RESCALE)
        });
        let stats = v
            .get("image_mean")
            .and_then(channels)
            .zip(v.get("image_std").and_then(channels));
        let normalize = if flag("do_normalize", stats.is_some()) {
            Some(stats.ok_or("`do_normalize` is set without `image_mean` and `image_std`")?)
        } else {
            None
        };
        Ok(Self {
            resize,
            filter,
            crop,
            rescale,
            normalize,
        })
    }

    /// The `(height, width)` every image comes out at, when the steps fix it.
    pub fn output_size(&self) -> Option<(usize, usize)> {
        match (self.crop, self.resize) {
            (Some(c), _) => Some(c),
            (None, Some(Resize::Exact { height, width })) => Some((height, width)),
            _ => None,
        }
    }

    /// The size a `w` x `h` image is resized to, `(width, height)`.
    fn resized(&self, w: usize, h: usize) -> (usize, usize) {
        match self.resize {
            None => (w, h),
            Some(Resize::Exact { height, width }) => (width, height),
            Some(Resize::ShortestEdge(edge)) => {
                let (short, long) = if w <= h { (w, h) } else { (h, w) };
                let long = (edge as f64 * long as f64 / short as f64) as usize;
                if w <= h {
                    (edge, long)
                } else {
                    (long, edge)
                }
            }
        }
    }

    /// Interleaved RGB8 `w` x `h` -> channel-first `[3, height, width]` floats, with the
    /// height and width.
    pub fn apply(
        &self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        if w == 0 || h == 0 || rgb.len() < w * h * 3 {
            return Err(format!(
                "expected a {w}x{h} RGB image ({} bytes), got {} bytes",
                w * h * 3,
                rgb.len()
            ));
        }
        let (rw, rh) = self.resized(w, h);
        let resized = if (rw, rh) == (w, h) {
            rgb[..w * h * 3].to_vec()
        } else {
            resample(rgb, (w, h), (rw, rh), self.filter)
        };
        let (ch, cw) = self.crop.unwrap_or((rh, rw));
        // The crop's corner, as transformers places it; an image smaller than the crop is
        // centred in a zero border.
        let corner = |from: usize, to: usize| -> i64 {
            if from >= to {
                ((from - to) as f64 / 2.0) as i64
            } else {
                -(((to - from) as f64 / 2.0).ceil() as i64)
            }
        };
        let (top, left) = (corner(rh, ch), corner(rw, cw));
        let plane = ch * cw;
        let mut out = vec![0f32; 3 * plane];
        for y in 0..ch {
            let sy = y as i64 + top;
            for x in 0..cw {
                let sx = x as i64 + left;
                let inside = (0..rh as i64).contains(&sy) && (0..rw as i64).contains(&sx);
                for c in 0..3 {
                    let byte = if inside {
                        resized[((sy as usize) * rw + sx as usize) * 3 + c]
                    } else {
                        0
                    };
                    // transformers rescales in double precision and stores single; the
                    // normalisation runs in single.
                    let mut v = match self.rescale {
                        Some(factor) => (f64::from(byte) * factor) as f32,
                        None => f32::from(byte),
                    };
                    if let Some((mean, std)) = &self.normalize {
                        v = (v - mean[c]) / std[c];
                    }
                    out[c * plane + y * cw + x] = v;
                }
            }
        }
        Ok((out, ch, cw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A 2:1 box downscale averages pairs and rounds half up, through Pillow's fixed point.
    #[test]
    fn a_box_halving_averages_pairs() {
        let row: Vec<u8> = [10u8, 20, 30, 41].iter().flat_map(|&v| [v, v, v]).collect();
        let out = resample(&row, (4, 1), (2, 1), Filter::Box);
        assert_eq!(out, vec![15, 15, 15, 36, 36, 36]);
    }

    /// Every filter's weights sum to one, so a flat image stays flat at any scale.
    #[test]
    fn a_flat_image_stays_flat() {
        let rgb = vec![77u8; 13 * 7 * 3];
        for f in [
            Filter::Lanczos,
            Filter::Bilinear,
            Filter::Bicubic,
            Filter::Box,
            Filter::Hamming,
            Filter::Nearest,
        ] {
            for size in [(5, 3), (29, 17), (13, 3)] {
                let out = resample(&rgb, (13, 7), size, f);
                assert_eq!(out.len(), size.0 * size.1 * 3);
                assert!(out.iter().all(|&v| v == 77), "{f:?} to {size:?}");
            }
        }
    }

    /// The cubic kernel is one at zero, zero at the integers, and negative in its outer lobe.
    #[test]
    fn the_cubic_kernel_has_its_shape() {
        let f = Filter::Bicubic;
        assert_eq!(f.weight(0.0), 1.0);
        assert_eq!(f.weight(1.0), 0.0);
        assert_eq!(f.weight(2.0), 0.0);
        assert!(f.weight(1.5) < 0.0);
        assert_eq!(f.weight(-0.5), f.weight(0.5));
    }

    /// CLIP's config: an integer size is the shortest edge, then a square crop.
    #[test]
    fn a_shortest_edge_config_resizes_then_crops() {
        let p = Preprocess::from_json(&json!({
            "crop_size": 224, "do_center_crop": true, "do_normalize": true, "do_resize": true,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "resample": 3, "size": 224
        }))
        .unwrap();
        assert_eq!(p.filter, Filter::Bicubic);
        assert_eq!(p.output_size(), Some((224, 224)));
        assert_eq!(p.resized(640, 480), (298, 224));
        assert_eq!(p.resized(480, 640), (224, 298));
        assert_eq!(p.rescale, Some(DEFAULT_RESCALE));
    }

    /// SigLIP's config: an exact size, no crop, statistics of one half.
    #[test]
    fn an_exact_size_config_needs_no_crop() {
        let p = Preprocess::from_json(&json!({
            "do_normalize": true, "do_rescale": true, "do_resize": true,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "resample": 3, "rescale_factor": 0.00392156862745098,
            "size": {"height": 224, "width": 224}
        }))
        .unwrap();
        assert_eq!(p.crop, None);
        assert_eq!(p.output_size(), Some((224, 224)));
        assert_eq!(p.resized(640, 480), (224, 224));
        // A white pixel lands at one and a black one at minus one.
        let (out, h, w) = p.apply(&[255, 255, 255, 0, 0, 0], 2, 1).unwrap();
        assert_eq!((h, w), (224, 224));
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[223] + 1.0).abs() < 1e-6);
    }

    /// The crop takes the centre, its corner truncated toward the top left.
    #[test]
    fn the_crop_takes_the_centre() {
        let p = Preprocess::from_json(&json!({"crop_size": {"height": 1, "width": 2},
            "do_rescale": false}))
        .unwrap();
        let rgb: Vec<u8> = (0u8..5).flat_map(|v| [v, v, v]).collect();
        let (out, h, w) = p.apply(&rgb, 5, 1).unwrap();
        assert_eq!((h, w), (1, 2));
        assert_eq!(&out[..2], &[1.0, 2.0]);
    }

    /// A shape the processor resizes to by other rules is refused rather than guessed.
    #[test]
    fn an_unknown_size_is_refused() {
        assert!(Preprocess::from_json(&json!({"size": {"longest_edge": 512}})).is_err());
        assert!(Preprocess::from_json(&json!({"resample": 9})).is_err());
    }
}
