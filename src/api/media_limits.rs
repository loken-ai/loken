//! Per-request caps on the media a chat request hands the model, and the fetch of images
//! given by URL.
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

/// Seconds a fetch of an image by URL may take by default.
pub const DEFAULT_FETCH_TIMEOUT_SECS: u64 = 30;

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

/// The caps one request is held to, and where an image given by URL may come from.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaLimits {
    /// Images one request hands the vision encoder.
    pub max_images: usize,
    /// Pixels (width times height) one image may declare.
    pub max_image_pixels: u64,
    /// Seconds of audio one request hands the speech tower.
    pub max_audio_seconds: f64,
    /// Hosts, or `host:port` pairs, an image given by URL is fetched from.
    pub fetch_hosts: Vec<String>,
    /// Seconds a fetch may take, connection to last byte.
    pub fetch_timeout_secs: u64,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_images: DEFAULT_MAX_IMAGES,
            max_image_pixels: default_max_image_pixels(),
            max_audio_seconds: default_max_audio_seconds(),
            fetch_hosts: Vec::new(),
            fetch_timeout_secs: DEFAULT_FETCH_TIMEOUT_SECS,
        }
    }
}

impl MediaLimits {
    /// Bytes a fetched image file may hold: its largest allowed pixel count at RGBA8.
    pub fn max_image_bytes(&self) -> u64 {
        self.max_image_pixels.saturating_mul(BYTES_PER_PIXEL_RGBA8)
    }

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

    /// Whether `url`'s host, or its `host:port`, is in `fetch_hosts`.
    fn host_allowed(&self, url: &reqwest::Url) -> bool {
        host_in(&self.fetch_hosts, url)
    }

    /// The image at `url`, fetched over http or https from a host in `fetch_hosts`,
    /// following redirects only to such hosts, held to `max_image_bytes` and to the pixel
    /// cap. Answers the bytes as base64 and the media type the host gave, if any.
    pub async fn fetch_image(&self, url: &str) -> Result<(String, Option<String>), String> {
        let parsed = reqwest::Url::parse(url).map_err(|e| format!("image URL '{url}': {e}"))?;
        match parsed.scheme() {
            "http" | "https" => {}
            other => {
                return Err(format!(
                    "image URL scheme '{other}' is not fetched; use http or https"
                ))
            }
        }
        if !self.host_allowed(&parsed) {
            return Err(format!(
                "host '{}' is not in `image_fetch_hosts`",
                parsed.host_str().unwrap_or("")
            ));
        }
        let hosts = self.fetch_hosts.clone();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.fetch_timeout_secs))
            .redirect(reqwest::redirect::Policy::custom(move |attempt| {
                if host_in(&hosts, attempt.url()) {
                    attempt.follow()
                } else {
                    let host = attempt.url().host_str().unwrap_or("").to_string();
                    attempt.error(format!(
                        "redirect to host '{host}', which is not in `image_fetch_hosts`"
                    ))
                }
            }))
            .build()
            .map_err(|e| format!("image fetch client: {e}"))?;
        let mut resp = client
            .get(parsed)
            .send()
            .await
            .map_err(|e| format!("fetching image '{url}': {}", error_chain(&e)))?;
        if !resp.status().is_success() {
            return Err(format!("fetching image '{url}' answered {}", resp.status()));
        }
        let max = self.max_image_bytes();
        if let Some(len) = resp.content_length().filter(|&l| l > max) {
            return Err(format!(
                "image '{url}' is {len} bytes and `max_image_pixels` allows files of {max}"
            ));
        }
        let media_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let mut bytes = Vec::new();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| format!("fetching image '{url}': {}", error_chain(&e)))?
        {
            if (bytes.len() + chunk.len()) as u64 > max {
                return Err(format!(
                    "image '{url}' is over {max} bytes, the file size `max_image_pixels` allows"
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        self.check_image_bytes(1, &bytes)?;
        Ok((
            base64::engine::general_purpose::STANDARD.encode(&bytes),
            media_type,
        ))
    }
}

/// Whether `url`'s host, or its `host:port`, is one of `hosts`, compared without case.
fn host_in(hosts: &[String], url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    let with_port = url.port_or_known_default().map(|p| format!("{host}:{p}"));
    hosts.iter().any(|h| {
        h.eq_ignore_ascii_case(host)
            || with_port
                .as_deref()
                .is_some_and(|hp| h.eq_ignore_ascii_case(hp))
    })
}

/// An error and its sources, joined, so a redirect refusal reaches the client.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        out.push_str(": ");
        out.push_str(&s.to_string());
        src = s.source();
    }
    out
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
        assert_eq!(d.max_image_bytes(), DECODE_MAX_ALLOC_BYTES);
        assert!(d.max_audio_seconds > 0.0);
        assert!(
            d.fetch_hosts.is_empty(),
            "nothing is fetched unless configured"
        );
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

    /// A one-route server on loopback answering `body` at `/img.png`, with `/away`
    /// redirecting to `away`.
    async fn serve(body: Vec<u8>, away: String) -> std::net::SocketAddr {
        use axum::routing::get;
        let app = axum::Router::new()
            .route(
                "/img.png",
                get(move || {
                    let body = body.clone();
                    async move { ([(axum::http::header::CONTENT_TYPE, "image/png")], body) }
                }),
            )
            .route(
                "/away",
                get(move || {
                    let away = away.clone();
                    async move { axum::response::Redirect::temporary(&away) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr
    }

    fn allowing(host: &str) -> MediaLimits {
        MediaLimits {
            fetch_hosts: vec![host.to_string()],
            ..limits()
        }
    }

    #[tokio::test]
    async fn an_allowed_host_is_fetched() {
        let bytes = png(16, 16);
        let addr = serve(bytes.clone(), String::new()).await;
        let (data, media_type) = allowing("127.0.0.1")
            .fetch_image(&format!("http://{addr}/img.png"))
            .await
            .unwrap();
        assert_eq!(data, b64(&bytes));
        assert_eq!(media_type.as_deref(), Some("image/png"));
        // A `host:port` entry matches too.
        assert!(allowing(&addr.to_string())
            .fetch_image(&format!("http://{addr}/img.png"))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn a_host_outside_the_allowlist_is_refused_by_name() {
        let addr = serve(png(16, 16), String::new()).await;
        let err = allowing("images.example")
            .fetch_image(&format!("http://{addr}/img.png"))
            .await
            .unwrap_err();
        assert!(
            err.contains("host '127.0.0.1' is not in `image_fetch_hosts`"),
            "{err}"
        );
        let err = MediaLimits::default()
            .fetch_image(&format!("http://{addr}/img.png"))
            .await
            .unwrap_err();
        assert!(
            err.contains("'127.0.0.1'"),
            "an empty allowlist fetches nothing: {err}"
        );
    }

    #[tokio::test]
    async fn a_redirect_outside_the_allowlist_is_refused() {
        let addr = serve(png(16, 16), "http://localhost:9/img.png".to_string()).await;
        let err = allowing("127.0.0.1")
            .fetch_image(&format!("http://{addr}/away"))
            .await
            .unwrap_err();
        assert!(err.contains("host 'localhost'"), "{err}");
    }

    #[tokio::test]
    async fn a_fetched_image_is_held_to_the_caps() {
        let addr = serve(png(65, 64), String::new()).await;
        let err = allowing("127.0.0.1")
            .fetch_image(&format!("http://{addr}/img.png"))
            .await
            .unwrap_err();
        assert!(err.contains("`max_image_pixels` allows 4096"), "{err}");
        let tiny = MediaLimits {
            max_image_pixels: 1,
            ..allowing("127.0.0.1")
        };
        let err = tiny
            .fetch_image(&format!("http://{addr}/img.png"))
            .await
            .unwrap_err();
        assert!(err.contains("bytes"), "{err}");
    }

    #[tokio::test]
    async fn a_non_http_scheme_is_refused() {
        let err = allowing("127.0.0.1")
            .fetch_image("file:///etc/hostname")
            .await
            .unwrap_err();
        assert!(err.contains("scheme 'file'"), "{err}");
    }
}
