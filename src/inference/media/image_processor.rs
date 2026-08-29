use crate::tensor::{DType, Device, Tensor};
use anyhow::{anyhow, Result};
use image::DynamicImage;

/// Decode image bytes, APPLYING the EXIF orientation.
///
/// THE decode every input path must use. A photograph from a phone is almost always
/// stored in the sensor's native landscape layout with an EXIF Orientation tag saying
/// how to rotate it, and `image` does not apply that tag on decode. Skipping it means
/// an upright portrait arrives rotated 90 degrees: the edit is composed sideways, the
/// face detector finds nothing (it is not rotation invariant), and the result comes
/// back turned over - all without a single error.
///
/// Orientation also SWAPS width and height for the quarter turns, so anything that
/// measures an image must measure it after this, not before.
/// What a rejection tells the caller they COULD have sent.
///
/// Every image on every endpoint decodes through the function below, so this one string
/// is the whole server's answer. It exists because the rejections said only "cannot
/// decode the image" - true, and useless: the caller cannot tell whether the file is
/// corrupt, or a format nobody here reads, and has nothing to try next.
///
/// The list is the decoder's actual set. AVIF and JPEG-XL are named as ABSENT on purpose:
/// they are the two a user is most likely to have and most likely to assume work.
pub const ACCEPTED_IMAGE_FORMATS: &str =
    "Accepted: PNG, JPEG, WebP, GIF, BMP, TIFF, ICO, TGA, QOI, PNM, HDR, EXR, DDS \
     (not AVIF or JPEG-XL - convert those first)";

pub fn decode_image_oriented(bytes: &[u8]) -> Result<DynamicImage> {
    // Bound the decode so a tiny "decompression bomb" (a small file declaring
    // e.g. 60000x60000) can't OOM-abort the process - no CatchPanic covers an
    // allocator abort. 16384^2 covers any real photo/8K screenshot; 512 MB decode
    // budget is generous for those and rejects the pathological ones.
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| {
            anyhow!(
                "could not read the image header: {e}. {}",
                ACCEPTED_IMAGE_FORMATS
            )
        })?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(512 * 1024 * 1024);
    reader.limits(limits);
    use image::ImageDecoder as _;
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| anyhow!("could not read the image: {e}. {}", ACCEPTED_IMAGE_FORMATS))?;
    // A missing or unreadable tag is not an error: it just means no rotation.
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).map_err(|e| {
        anyhow!(
            "could not decode the image: {e}. {}",
            ACCEPTED_IMAGE_FORMATS
        )
    })?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Decode a base64-encoded image string into a DynamicImage, EXIF orientation applied.
pub fn decode_base64_image(base64_str: &str) -> Result<DynamicImage> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_str)
        .map_err(|e| anyhow!("Failed to decode base64 image: {}", e))?;
    decode_image_oriented(&bytes)
}

/// Preprocess an image for Moondream vision model.
/// Resizes to 378x378, normalizes with mean=0.5/std=0.5, returns tensor (1, 3, 378, 378).
/// Follows the reference's own load_image()
pub fn preprocess_moondream(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    // Bilinear matches ollama/llama.cpp clip (which uses RESIZE_ALGO_BILINEAR
    // for moondream and produces correct captions). A bicubic experiment
    // was inconclusive - it did NOT fix the residual prompt-sensitive
    // "urn/ale" hallucination - so the resize is not the diff; reverted.
    let resized = image.resize_to_fill(378, 378, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let data = rgb.into_raw();

    // Shape: (378, 378, 3) -> permute to (3, 378, 378)
    let tensor = Tensor::from_vec(data, (378, 378, 3), &Device::Cpu)?.permute((2, 0, 1))?;

    // Normalize: (pixel / 255.0 - 0.5) / 0.5
    //
    // Since mean and std are uniform 0.5 across all channels, this
    // collapses algebraically to a single affine: pixel * (1/127.5) - 1
    //   = (pixel/255 - 0.5) / 0.5
    //   = (pixel/255) * 2 - 1
    //   = pixel * (2/255) - 1
    //
    // The original 3-kernel pipeline (div + broadcast_sub + broadcast_div)
    // allocated two intermediate buffers and dispatched two extra
    // element-wise kernels for what is mathematically a single
    // multiply-add. The affine path matches Tensor::affine's "x * mul + add"
    // signature exactly, so the substrate issues one launch + one alloc.
    let normalized = tensor.to_dtype(DType::F32)?.affine(2.0 / 255.0, -1.0)?;

    // Add batch dimension: (3, 378, 378) -> (1, 3, 378, 378)
    let batched = normalized.unsqueeze(0)?;

    // Move to target device
    let result = batched.to_device(device)?;
    Ok(result)
}

/// Decode a base64 image and preprocess it for Moondream in one step.
pub fn prepare_moondream_image(base64_str: &str, device: &Device) -> Result<Tensor> {
    let image = decode_base64_image(base64_str)?;
    preprocess_moondream(&image, device)
}

/// Qwen3-VL / qwen35moe image preprocessing parameters.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35VisionPreproc {
    pub patch_size: usize,    // 16
    pub merge_size: usize,    // 2 (spatial_merge_size)
    pub temporal: usize,      // 2 (temporal_patch_size; still image -> frame duplicated)
    pub channels: usize,      // 3
    pub shortest_edge: usize, // area lower bound (px)
    pub longest_edge: usize,  // area upper bound (px)
}
impl Default for Qwen35VisionPreproc {
    fn default() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal: 2,
            channels: 3,
            shortest_edge: 64 << 10,
            longest_edge: 2 << 20,
        }
    }
}

/// Qwen smart-resize: round each dim to a multiple of `factor`, then scale so the
/// pixel area lands within [shortest_edge, longest_edge]. Matches ollama qwen3vl.
fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    shortest: usize,
    longest: usize,
) -> (usize, usize) {
    let (hf, wf, ff) = (h as f64, w as f64, factor as f64);
    let mut hbar = ((hf / ff).round_ties_even() * ff) as usize;
    let mut wbar = ((wf / ff).round_ties_even() * ff) as usize;
    hbar = hbar.max(factor);
    wbar = wbar.max(factor);
    if hbar * wbar > longest {
        let beta = ((h * w) as f64 / longest as f64).sqrt();
        hbar = ((hf / beta / ff).floor() * ff).max(ff) as usize;
        wbar = ((wf / beta / ff).floor() * ff).max(ff) as usize;
    } else if hbar * wbar < shortest {
        let beta = (shortest as f64 / (h * w) as f64).sqrt();
        hbar = ((hf * beta / ff).ceil() * ff) as usize;
        wbar = ((wf * beta / ff).ceil() * ff) as usize;
    }
    (hbar, wbar)
}

/// Preprocess an image for qwen35moe vision. Returns (pixel_values, (grid_h, grid_w))
/// where pixel_values is [num_patches, patch_dim] F32 on `device`,
/// patch_dim = channels.temporal.patch.patch (= 1536), num_patches = grid_h.grid_w.
/// Patch order is 2x2-merge-grouped (so the merger's later reshape groups the right
/// 2x2 block); per patch the layout is [c][temporal][py][px], temporal frames are the
/// still frame duplicated. Normalize = pixel.2/255 - 1 (mean=std=0.5).
pub fn preprocess_qwen35vl(
    image: &DynamicImage,
    p: Qwen35VisionPreproc,
    device: &Device,
) -> Result<(Tensor, (usize, usize))> {
    let factor = p.patch_size * p.merge_size;
    let (h0, w0) = (image.height() as usize, image.width() as usize);
    if h0 < factor || w0 < factor {
        // upscale tiny images to at least one merge-block
        // (smart_resize would panic in ollama; we just clamp).
    }
    let (hbar, wbar) = smart_resize(
        h0.max(factor),
        w0.max(factor),
        factor,
        p.shortest_edge,
        p.longest_edge,
    );
    let resized = image.resize_exact(
        wbar as u32,
        hbar as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let raw = rgb.as_raw(); // HWC, row-major
    let (h, w, c) = (hbar, wbar, p.channels);
    // CHW normalized
    let mut chw = vec![0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let v = raw[(y * w + x) * 3 + ch] as f32 * (2.0 / 255.0) - 1.0;
                chw[ch * h * w + y * w + x] = v;
            }
        }
    }
    let grid_h = h / p.patch_size;
    let grid_w = w / p.patch_size;
    let patch_dim = c * p.temporal * p.patch_size * p.patch_size;
    let num_patches = grid_h * grid_w;
    let mut out = vec![0f32; num_patches * patch_dim];
    let ms = p.merge_size;
    let ps = p.patch_size;
    let frame = ps * ps;
    let mut pidx = 0usize;
    let mut gh = 0;
    while gh < grid_h {
        let mut gw = 0;
        while gw < grid_w {
            for mh in 0..ms {
                for mw in 0..ms {
                    let base = pidx * patch_dim;
                    for ch in 0..c {
                        let ch_off = base + ch * p.temporal * frame;
                        for py in 0..ps {
                            for px in 0..ps {
                                let y = (gh + mh) * ps + py;
                                let x = (gw + mw) * ps + px;
                                out[ch_off + py * ps + px] = chw[ch * h * w + y * w + x];
                            }
                        }
                        // duplicate first frame across temporal frames
                        for t in 1..p.temporal {
                            let dst = ch_off + t * frame;
                            out.copy_within(ch_off..ch_off + frame, dst);
                        }
                    }
                    pidx += 1;
                }
            }
            gw += ms;
        }
        gh += ms;
    }
    let t = Tensor::from_vec(out, (num_patches, patch_dim), &Device::Cpu)?.to_device(device)?;
    Ok((t, (grid_h, grid_w)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::IndexOp;
    use base64::Engine;

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// Build a 1x1 PNG containing a single mid-grey pixel.
    /// Used as a minimal valid input for the decode path.
    fn one_pixel_png() -> Vec<u8> {
        let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_fn(1, 1, |_, _| {
            image::Rgb([128u8, 128u8, 128u8])
        });
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("encode png");
        bytes
    }

    #[test]
    fn decode_base64_image_round_trips_a_valid_png() {
        let png = one_pixel_png();
        let img = decode_base64_image(&b64(&png)).expect("decode");
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
    }

    #[test]
    fn decode_base64_image_rejects_garbage_base64() {
        // Not valid base64 at all.
        assert!(decode_base64_image("!!!").is_err());
        // Valid base64 but not an image.
        assert!(decode_base64_image(&b64(b"hello world")).is_err());
        // Empty.
        assert!(decode_base64_image("").is_err());
    }

    #[test]
    fn preprocess_moondream_normalises_to_unit_range_per_documented_formula() {
        // The affine `pixel * (2/255) - 1` MUST produce values
        // identical (to within f32 precision) to the original
        // `(pixel/255 - 0.5) / 0.5` pipeline. Pin both endpoints
        // and the midpoint:
        //   pixel=0   -> -1.0
        //   pixel=255 -> +1.0  (within f32 ε)
        //   pixel=128 -> ~0.0039  (= 128*2/255 - 1)
        //
        // Construct a 3x1x1 image with one red/green/blue pixel each
        // value so all three channels exercise the same normalisation.
        let img =
            image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(3, 1, |x, _| match x {
                0 => image::Rgb([0u8, 0, 0]),
                1 => image::Rgb([128u8, 128, 128]),
                _ => image::Rgb([255u8, 255, 255]),
            }));
        let t = preprocess_moondream(&img, &Device::Cpu).expect("preprocess");
        // (1, 3, 378, 378) after resize. The 3-pixel source gets
        // upscaled to fill 378x378 via Triangle filter - so we sample
        // just one location and check the three channels are within
        // the expected per-pixel range.
        let dims = t.dims4().expect("4d shape");
        assert_eq!(dims, (1, 3, 378, 378));
        // Range check: all values must be in [-1, 1].
        let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &flat {
            assert!(
                *v >= -1.0 - 1e-5 && *v <= 1.0 + 1e-5,
                "value {v} outside [-1, 1]"
            );
        }
        // Mean over a 378x378 patch of a 3-pixel src is hard to pin
        // exactly because of the resize filter, but the OVERALL mean
        // across all pixels should be near zero (because 0 ↔ 255
        // input maps to -1 ↔ 1 symmetrically around 0).
        let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
        assert!(
            mean.abs() < 0.1,
            "global mean {mean} not near zero - symmetry broken"
        );
    }

    #[test]
    fn smart_resize_scales_into_area_bounds_and_factor_multiple() {
        // tiny image -> scaled UP to >= shortest_edge area, dims multiple of 32
        let (h, w) = smart_resize(64, 64, 32, 64 << 10, 2 << 20);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h * w >= 64 << 10, "area {} below shortest", h * w);
        // huge image -> scaled DOWN below longest_edge
        let (h2, w2) = smart_resize(8000, 8000, 32, 64 << 10, 2 << 20);
        assert_eq!(h2 % 32, 0);
        assert!(h2 * w2 <= 2 << 20, "area {} above longest", h2 * w2);
    }

    #[test]
    fn preprocess_qwen35vl_produces_correct_patch_shape() {
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(64, 64, |x, _| {
            image::Rgb([(x % 256) as u8, 128, 200])
        }));
        let (t, (gh, gw)) = preprocess_qwen35vl(&img, Qwen35VisionPreproc::default(), &Device::Cpu)
            .expect("preprocess");
        let (np, pd) = t.dims2().expect("2d");
        assert_eq!(pd, 3 * 2 * 16 * 16, "patch_dim");
        assert_eq!(np, gh * gw, "num_patches == grid_h*grid_w");
        // 64x64 -> area 4096 < 65536 -> upscaled to 256x256 -> grid 16x16 = 256 patches
        assert_eq!((gh, gw), (16, 16));
        // values in [-1, 1]
        let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|v| *v >= -1.0 - 1e-5 && *v <= 1.0 + 1e-5));
        // temporal duplication: frame 0 and frame 1 of channel 0, patch 0 must match
        let p0 = t.i(0).unwrap().to_vec1::<f32>().unwrap();
        let frame = 16 * 16;
        assert_eq!(
            &p0[0..frame],
            &p0[frame..2 * frame],
            "temporal frames not duplicated"
        );
    }

    #[test]
    fn preprocess_moondream_matches_explicit_mean_std_formula() {
        // Build a single-pixel image at each documented input value,
        // verify the affine yields the same number the old three-kernel
        // pipeline did. Computed by hand: y = (x/255 - 0.5) / 0.5.
        let test_cases = [
            (0u8, -1.0_f32),   // 0/255 = 0 -> -1
            (64, -0.4980392),  // 64/255 = 0.251 -> 0.251*2-1 = -0.498
            (128, 0.00392157), // 128/255 ≈ 0.502 -> 0.502*2-1 = 0.00392
            (191, 0.49803925), // ~0.749 -> 0.498
            (255, 1.0),
        ];
        for (px, want) in test_cases {
            let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(1, 1, |_, _| {
                image::Rgb([px, px, px])
            }));
            let t = preprocess_moondream(&img, &Device::Cpu).expect("preprocess");
            // Sample the (0, 0, 0, 0) position. The single source pixel
            // gets replicated to fill 378x378 so any position works.
            let v = t.i((0, 0, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
            assert!((v - want).abs() < 1e-4, "px={px}: got {v}, want {want}");
        }
    }
}

#[cfg(test)]
mod orientation_tests {
    use super::*;

    /// Build a JPEG carrying an EXIF Orientation tag, so the test needs no fixture.
    /// `orientation` 6 is the common phone-portrait case: stored landscape, displayed
    /// rotated 90 degrees clockwise.
    fn jpeg_with_orientation(w: u32, h: u32, orientation: u16) -> Vec<u8> {
        // A distinctive image: the top-left quadrant is white, the rest black, so a
        // rotation is detectable from the pixels alone.
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let on = x < w / 2 && y < h / 2;
                img.put_pixel(
                    x,
                    y,
                    image::Rgb(if on { [255, 255, 255] } else { [0, 0, 0] }),
                );
            }
        }
        let mut jpg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut jpg, image::ImageFormat::Jpeg)
            .expect("encode jpeg");
        let jpg = jpg.into_inner();

        // Splice a minimal APP1/EXIF segment carrying only the Orientation tag in
        // after SOI. Little-endian TIFF header, one IFD entry, no next-IFD.
        let mut exif: Vec<u8> = b"Exif\0\0".to_vec();
        let tiff_start = exif.len();
        exif.extend_from_slice(b"II*\0");
        exif.extend_from_slice(&8u32.to_le_bytes()); // IFD0 at offset 8
        exif.extend_from_slice(&1u16.to_le_bytes()); // one entry
        exif.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
        exif.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        exif.extend_from_slice(&1u32.to_le_bytes()); // count
        exif.extend_from_slice(&orientation.to_le_bytes());
        exif.extend_from_slice(&[0, 0]); // pad the 4-byte value field
        exif.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        let _ = tiff_start;

        let mut out = Vec::with_capacity(jpg.len() + exif.len() + 4);
        out.extend_from_slice(&jpg[..2]); // SOI
        out.extend_from_slice(&[0xFF, 0xE1]); // APP1
        out.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&exif);
        out.extend_from_slice(&jpg[2..]);
        out
    }

    /// The bug: a phone portrait is stored landscape with a rotate tag, and every
    /// path that skipped the tag composed the edit sideways and lost the face.
    #[test]
    fn a_rotated_photo_is_decoded_upright() {
        // Orientation 6 = rotate 90 CW on display, so 40x20 stored becomes 20x40.
        let bytes = jpeg_with_orientation(40, 20, 6);
        let plain = image::load_from_memory(&bytes).expect("plain decode");
        assert_eq!(
            (plain.width(), plain.height()),
            (40, 20),
            "sanity: the raw decode keeps the stored layout, which is the bug"
        );
        let fixed = decode_image_oriented(&bytes).expect("oriented decode");
        assert_eq!(
            (fixed.width(), fixed.height()),
            (20, 40),
            "the quarter turn must swap width and height"
        );
    }

    /// An image with no EXIF at all must come through untouched.
    #[test]
    fn an_untagged_image_is_unchanged() {
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_pixel(7, 3, image::Rgb([10, 20, 30]))
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode");
        let bytes = png.into_inner();
        let img = decode_image_oriented(&bytes).expect("decode");
        assert_eq!((img.width(), img.height()), (7, 3));
    }

    /// SOURCE GATE: an input path that decodes with the raw `load_from_memory` skips
    /// the orientation. That was the defect at six sites; the scan keeps a seventh
    /// from appearing. Outputs we produced ourselves carry no EXIF and may opt out
    /// with `ORIENTATION-OK: <why>`.
    #[test]
    fn no_input_path_decodes_without_the_orientation() {
        fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                // Dev binaries under bin/ read files the operator hands them directly.
                if p.is_dir() {
                    if p.file_name().is_some_and(|n| n == "bin") {
                        continue;
                    }
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    let Ok(src) = std::fs::read_to_string(&p) else {
                        continue;
                    };
                    for (i, line) in src.lines().enumerate() {
                        let needle = concat!("image::load_", "from_memory");
                        if line.contains(needle)
                            && !line.contains("ORIENTATION-OK:")
                            && !src
                                .lines()
                                .nth(i.saturating_sub(1))
                                .is_some_and(|l| l.contains("ORIENTATION-OK:"))
                            // The test above calls it deliberately, to show the bug.
                            && !line.contains("plain decode")
                        {
                            out.push(format!("{}:{}: {}", p.display(), i + 1, line.trim()));
                        }
                    }
                }
            }
        }
        let mut offenders = Vec::new();
        walk(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src")
                .as_path(),
            &mut offenders,
        );
        assert!(
            offenders.is_empty(),
            "decode input images with `image_processor::decode_image_oriented`, which \
             applies the EXIF rotation; a raw decode returns a phone photo sideways:\n{}",
            offenders.join("\n")
        );
    }
}
