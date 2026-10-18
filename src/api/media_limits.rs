//! Per-request caps on the media a chat request hands the model.
//!
//! Every chat route reaches the encoders through the same admission, which checks the
//! caps there: the count of images, the pixels each image declares in its header (read
//! before the image is decoded) and the seconds of the audio clip (read from the
//! container's header where it states them, and from the decoded frames otherwise). A
//! request past a cap is refused with the cap's key and the value it carried.

use crate::inference::media::image_processor::DECODE_MAX_ALLOC_BYTES;
use base64::Engine as _;

/// Images one request hands the vision encoder by default: the most one message carries
/// on the wire.
pub const DEFAULT_MAX_IMAGES: usize = 16;

/// Bytes a decoded pixel holds at RGBA with 8 bits per channel.
const BYTES_PER_PIXEL_RGBA8: u64 = 4;

/// Channels and rate of the clip the default audio cap is sized for: stereo at the
/// highest common capture rate.
const AUDIO_REF_CHANNELS: u64 = 2;
const AUDIO_REF_RATE: u64 = 48_000;

/// Bytes a decoded audio sample holds (planar f32).
const BYTES_PER_AUDIO_SAMPLE: u64 = 4;

/// The pixels one image may declare by default: what the decoder's allocation budget holds
/// at RGBA8.
pub fn default_max_image_pixels() -> u64 {
    DECODE_MAX_ALLOC_BYTES / BYTES_PER_PIXEL_RGBA8
}

/// The seconds of audio one request may carry by default: what the same allocation budget
/// holds of a stereo clip at 48 kHz, decoded to f32.
pub fn default_max_audio_seconds() -> f64 {
    let bytes_per_second = AUDIO_REF_CHANNELS * AUDIO_REF_RATE * BYTES_PER_AUDIO_SAMPLE;
    (DECODE_MAX_ALLOC_BYTES / bytes_per_second) as f64
}

/// The caps one request is held to.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaLimits {
    /// Images one request hands the vision encoder.
    pub max_images: usize,
    /// Pixels (width times height) one image may declare.
    pub max_image_pixels: u64,
    /// Seconds of audio one request hands the speech tower.
    pub max_audio_seconds: f64,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_images: DEFAULT_MAX_IMAGES,
            max_image_pixels: default_max_image_pixels(),
            max_audio_seconds: default_max_audio_seconds(),
        }
    }
}

impl MediaLimits {
    /// Refuses more images than `max_images`, or a base64 image whose header declares more
    /// pixels than `max_image_pixels`. An image whose header does not read passes: the
    /// decoder refuses it with its own reason.
    pub fn check_images(&self, images: &[String]) -> Result<(), String> {
        if images.len() > self.max_images {
            return Err(format!(
                "the request carries {} images and `max_images` allows {}",
                images.len(),
                self.max_images
            ));
        }
        for (i, b64) in images.iter().enumerate() {
            let payload = crate::api::types::base64_payload(b64).trim();
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(payload) {
                self.check_image_bytes(i + 1, &bytes)?;
            }
        }
        Ok(())
    }

    /// Refuses an image file whose header declares more pixels than `max_image_pixels`;
    /// `ordinal` names it in the reason.
    pub fn check_image_bytes(&self, ordinal: usize, bytes: &[u8]) -> Result<(), String> {
        let Some((w, h)) = image_dimensions(bytes) else {
            return Ok(());
        };
        let pixels = u64::from(w) * u64::from(h);
        if pixels > self.max_image_pixels {
            return Err(format!(
                "image {ordinal} is {w}x{h}, {pixels} pixels, and `max_image_pixels` allows {}",
                self.max_image_pixels
            ));
        }
        Ok(())
    }

    /// Refuses a clip longer than `max_audio_seconds`.
    pub fn check_audio_seconds(&self, seconds: f64) -> Result<(), String> {
        if seconds > self.max_audio_seconds {
            return Err(format!(
                "the clip lasts {seconds:.2} seconds and `max_audio_seconds` allows {}",
                self.max_audio_seconds
            ));
        }
        Ok(())
    }
}

/// Width and height an image file declares in its header, without decoding its pixels.
fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::new(w, h);
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn limits() -> MediaLimits {
        MediaLimits {
            max_images: 2,
            max_image_pixels: 64 * 64,
            max_audio_seconds: 10.0,
            ..MediaLimits::default()
        }
    }

    #[test]
    fn defaults_derive_from_the_decode_budget() {
        let d = MediaLimits::default();
        assert_eq!(d.max_images, DEFAULT_MAX_IMAGES);
        assert!(d.max_audio_seconds > 0.0);
    }

    #[test]
    fn the_image_count_cap_passes_at_the_cap_and_refuses_past_it() {
        let one = b64(&png(8, 8));
        assert!(limits().check_images(&[one.clone(), one.clone()]).is_ok());
        let err = limits()
            .check_images(&[one.clone(), one.clone(), one])
            .unwrap_err();
        assert!(
            err.contains("3 images") && err.contains("`max_images` allows 2"),
            "{err}"
        );
    }

    #[test]
    fn the_pixel_cap_passes_at_the_cap_and_refuses_past_it() {
        assert!(limits().check_images(&[b64(&png(64, 64))]).is_ok());
        let data_url = format!("data:image/png;base64,{}", b64(&png(65, 64)));
        let err = limits().check_images(&[data_url]).unwrap_err();
        assert!(
            err.contains("65x64")
                && err.contains("4160 pixels")
                && err.contains("`max_image_pixels` allows 4096"),
            "{err}"
        );
    }

    #[test]
    fn an_unreadable_image_is_left_to_the_decoder() {
        assert!(limits().check_images(&[b64(b"not an image")]).is_ok());
    }

    #[test]
    fn the_audio_cap_passes_at_the_cap_and_refuses_past_it() {
        assert!(limits().check_audio_seconds(10.0).is_ok());
        let err = limits().check_audio_seconds(10.01).unwrap_err();
        assert!(
            err.contains("10.01 seconds") && err.contains("`max_audio_seconds` allows 10"),
            "{err}"
        );
    }
}
