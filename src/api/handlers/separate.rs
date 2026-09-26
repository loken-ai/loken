//! Music source separation over HTTP: a mix in, its stems out.
//!
//! The separation model itself was already ported and driven by a CLI, but nothing
//! served it - a finished capability no client could reach. This is the route.
//!
//! Two things the CLI could take for granted and a server cannot: the upload is any
//! container at any rate, not a 44.1 kHz WAV, and the model has to be shared rather
//! than loaded per invocation. Separation is also genuinely slow (it runs a
//! transformer over overlapping 8 s windows of the whole track), so the handler
//! bounds the input length rather than letting one request occupy the GPU for an hour.

use axum::extract::{Multipart, State};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::handlers::APIServer;
use crate::inference::codec::melband::{MelBandRoformer, SAMPLE_RATE};

// No cap on track length. Separation cost is linear in duration, and the model is
// windowed, so a long track is slower but not different in kind - there is no length at
// which the result stops being correct. The old 12-minute refusal existed because a long
// request "looks like a hang", which is a reason to report progress, not to decline the
// work: an album side is exactly what someone separating stems is holding.

/// The resident model.
///
/// Loaded once and kept: the checkpoint is hundreds of megabytes and a per-request
/// load would dominate the separation itself.
// Only a SUCCESS is remembered. `get_or_init` over a `Result` caches the failure too,
// so a first request made before the weights were installed made every later request
// fail for the life of the process - telling the user to install what they had just
// installed. An error now propagates without being written down.
static NET: std::sync::OnceLock<MelBandRoformer> = std::sync::OnceLock::new();

/// The catalogue name of the separation model.
pub(crate) const SEPARATION_MODEL: &str = "melband-roformer";

/// Where the resident separation model sits; none when it has not been loaded.
pub(crate) fn resident_parts() -> Option<crate::inference::serve::progress::placement::Parts> {
    NET.get().map(|net| vec![(String::new(), net.placement())])
}

fn model(hf_models_dir: &str) -> Result<&'static MelBandRoformer, String> {
    if let Some(m) = NET.get() {
        return Ok(m);
    }
    // The loadable form is the safetensors; the .ckpt beside it is the original torch
    // checkpoint and is not what the loader reads.
    let w =
        std::path::Path::new(hf_models_dir).join("melband-roformer/MelBandRoformer.safetensors");
    if !w.exists() {
        tracing::warn!("source separation weights missing at {}", w.display());
        return Err("source separation needs the Mel-Band RoFormer weights, which are not installed; see the model documentation for how to fetch them"
            .to_string());
    }
    // Placement through the fleet's authority, never a hardcoded device: the model is
    // ~600 MB resident plus its per-window activations.
    let dev =
        crate::inference::place::vram_manager::pick_device_for_aux("source separation", 3 << 30)
            .map(|(_, _, d)| d)
            .unwrap_or(crate::tensor::Device::Cpu);
    tracing::info!("source separation: loading Mel-Band RoFormer on {dev:?}");
    let net = MelBandRoformer::load(w.to_str().unwrap_or_default(), &dev)
        .map_err(|e| format!("separation model load: {e}"))?;
    // Two concurrent first-requests can both load; the loser's copy is dropped here, which
    // frees it. Rare enough to be worth less than a lock held across a checkpoint load.
    Ok(NET.get_or_init(|| net))
}

/// Which stems the caller wants back.
#[derive(Clone, Copy, PartialEq)]
enum Want {
    Vocals,
    Instrumental,
    Both,
}

impl Want {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "" | "both" | "all" => Ok(Self::Both),
            "vocals" | "vocal" | "voice" => Ok(Self::Vocals),
            "instrumental" | "instruments" | "accompaniment" | "music" => Ok(Self::Instrumental),
            other => Err(format!(
                "`stems`: expected vocals, instrumental or both; got `{other}`"
            )),
        }
    }
}

/// POST /v1/audio/separate - multipart with `file` (the mix) and optional `stems`.
///
/// Returns base64 WAVs, one per requested stem, so a client can take just the one it
/// wants without paying for the other on the wire.
pub(crate) async fn audio_separate(state: State<APIServer>, mut multipart: Multipart) -> Response {
    use base64::Engine as _;
    let err = |code: axum::http::StatusCode, msg: String| -> Response {
        (
            code,
            Json(crate::api::handlers::openai::openai_error_body(code, msg)),
        )
            .into_response()
    };

    let (mut audio, mut want) = (None::<Vec<u8>>, Want::Both);
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return err(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("multipart parse: {e}"),
                )
            }
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "audio" => match field.bytes().await {
                Ok(b) => {
                    // The one audio endpoint that accepted an UNBOUNDED upload, and the
                    // one that does the most work per byte. Its siblings all cap here;
                    // this one relied on the global body limit, whose rejection names
                    // neither the cap nor what was sent.
                    if let Err(e) = crate::api::handlers::audio::validate_audio_input_size(b.len())
                    {
                        return err(axum::http::StatusCode::PAYLOAD_TOO_LARGE, e);
                    }
                    audio = Some(b.to_vec())
                }
                Err(e) => {
                    return err(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("read `file`: {e}"),
                    )
                }
            },
            "stems" => {
                let text = field.text().await.unwrap_or_default();
                match Want::parse(&text) {
                    Ok(w) => want = w,
                    Err(e) => return err(axum::http::StatusCode::BAD_REQUEST, e),
                }
            }
            _ => {}
        }
    }
    let Some(bytes) = audio else {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            "separation needs a `file` field holding the mix".into(),
        );
    };

    let dir = state.huggingface_models_dir.clone();
    let started = std::time::Instant::now();
    // Decode, separate and mix down are all blocking compute.
    // One media job at a time: the separation takes a card for its windows.
    let _media_guard = state.media_lock_for(SEPARATION_MODEL, "separation").await;
    let done = tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, Vec<u8>, f32), String> {
        let (left, right) = decode_to_stereo_44k(&bytes)?;
        let frames = left.len();
        let seconds = frames as f32 / SAMPLE_RATE as f32;
        if frames == 0 {
            return Err("the uploaded audio is empty".into());
        }
        // The model takes planar stereo: all of the left channel, then all of the right.
        let mut mix = Vec::with_capacity(2 * frames);
        mix.extend_from_slice(&left);
        mix.extend_from_slice(&right);

        let net = model(&dir)?;
        let vocals = net
            .separate(&mix, false)
            .map_err(|e| format!("separation: {e}"))?;
        // The model emits ONE stem; the accompaniment is what is left of the mix. That
        // also guarantees the two stems sum back to the original exactly.
        let inst: Vec<f32> = mix.iter().zip(&vocals).map(|(a, b)| a - b).collect();
        let wav = |planar: &[f32]| -> Vec<u8> {
            crate::inference::media::audio_io::write_wav_planar(planar, 2, frames, SAMPLE_RATE)
        };
        Ok((wav(&vocals), wav(&inst), seconds))
    })
    .await;

    let (vocals, inst, seconds) = match done {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            let code = if e.contains("needs ") {
                axum::http::StatusCode::SERVICE_UNAVAILABLE
            } else {
                axum::http::StatusCode::BAD_REQUEST
            };
            return err(code, e);
        }
        Err(e) => {
            return err(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("separation task: {e}"),
            )
        }
    };

    let b64 = base64::engine::general_purpose::STANDARD;
    let mut body = serde_json::json!({
        "created": chrono::Utc::now().timestamp(),
        "duration_s": seconds,
        "sample_rate": SAMPLE_RATE,
        "render_ms": started.elapsed().as_millis() as u64,
    });
    if want != Want::Instrumental {
        body["vocals"] = serde_json::json!(b64.encode(&vocals));
    }
    if want != Want::Vocals {
        body["instrumental"] = serde_json::json!(b64.encode(&inst));
    }
    tracing::info!(
        "Audio separate: {:.1}s of audio in {} ms",
        seconds,
        started.elapsed().as_millis()
    );
    Json(body).into_response()
}

/// Decode any supported container to 44.1 kHz stereo planar.
///
/// The separation model is trained at 44.1 kHz stereo and both properties matter: the
/// stereo field is part of how it distinguishes a centred vocal from the accompaniment,
/// so the mono decoder used for transcription would throw away the signal this depends
/// on. A mono source is duplicated rather than rejected.
fn decode_to_stereo_44k(bytes: &[u8]) -> Result<(Vec<f32>, Vec<f32>), String> {
    crate::api::handlers::audio::decode_audio_stereo(bytes, SAMPLE_RATE).map_err(|e| {
        format!(
            "{e}. {}",
            crate::api::handlers::audio::ACCEPTED_AUDIO_FORMATS
        )
    })
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// End-to-end through the handler's OWN path: decode, separate, and the identity
    /// the endpoint promises - the two stems sum back to the mix exactly.
    ///
    /// That last property is not decoration. The accompaniment is computed as
    /// `mix - vocals`, so if the stems ever stopped summing it would mean the decode
    /// and the model disagreed about layout or length, which is precisely the kind of
    /// wiring error that still produces plausible audio.
    #[test]
    #[ignore = "needs the separation weights and a mix at /tmp/sep/mix.wav"]
    fn separates_a_real_mix_and_the_stems_sum_back() {
        let path = std::path::Path::new("/tmp/sep/mix.wav");
        if !path.exists() {
            println!("no mix at {}; skipping", path.display());
            return;
        }
        let dir = crate::config::Config::load_test().get_hf_models_dir();
        let bytes = std::fs::read(path).expect("read mix");
        let (l, r) = decode_to_stereo_44k(&bytes).expect("decode");
        let frames = l.len();
        println!("mix: {:.1}s stereo", frames as f32 / SAMPLE_RATE as f32);
        let mut mix = Vec::with_capacity(2 * frames);
        mix.extend_from_slice(&l);
        mix.extend_from_slice(&r);

        let net = model(dir.to_str().unwrap()).expect("model");
        let t0 = std::time::Instant::now();
        let vocals = net.separate(&mix, false).expect("separate");
        println!(
            "separated {:.1}s of audio in {:.1}s",
            frames as f32 / SAMPLE_RATE as f32,
            t0.elapsed().as_secs_f32()
        );
        assert_eq!(
            vocals.len(),
            mix.len(),
            "the stem must match the mix in length"
        );

        let inst: Vec<f32> = mix.iter().zip(&vocals).map(|(a, b)| a - b).collect();
        let worst = mix
            .iter()
            .zip(vocals.iter().zip(&inst))
            .map(|(m, (v, i))| (m - (v + i)).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "the stems do not sum back to the mix (worst {worst:.2e})"
        );

        // And it must have actually SEPARATED something: a model that returned the mix
        // (or silence) would satisfy the sum above trivially.
        let energy = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let (em, ev, ei) = (energy(&mix), energy(&vocals), energy(&inst));
        println!("energy - mix {em:.5}, vocals {ev:.5}, instrumental {ei:.5}");
        assert!(ev > 1e-9, "the vocal stem is silent");
        assert!(ei > 1e-9, "the instrumental stem is silent");
        assert!(
            ev < em * 0.98,
            "the vocal stem is essentially the whole mix"
        );
    }

    #[test]
    fn stem_selection_accepts_the_names_a_client_would_use() {
        assert!(matches!(Want::parse("").unwrap(), Want::Both));
        assert!(matches!(Want::parse("both").unwrap(), Want::Both));
        assert!(matches!(Want::parse(" Vocals ").unwrap(), Want::Vocals));
        assert!(matches!(
            Want::parse("accompaniment").unwrap(),
            Want::Instrumental
        ));
        // A typo is reported rather than silently treated as "both", which would send
        // back a stem the caller did not ask for and never told them why.
        assert!(Want::parse("drums").is_err());
    }

    /// Mono must be duplicated, not rejected: the model needs two channels and a mono
    /// upload is a normal thing for a client to send.
    #[test]
    fn a_mono_wav_becomes_stereo_at_the_model_rate() {
        let frames = SAMPLE_RATE as usize / 10;
        let samples: Vec<f32> = (0..frames)
            .map(|i| (i as f32 / frames as f32) * 2.0 - 1.0)
            .collect();
        let wav =
            crate::inference::media::audio_io::write_wav_planar(&samples, 1, frames, SAMPLE_RATE);
        let (l, r) = decode_to_stereo_44k(&wav).expect("decode");
        assert_eq!(l.len(), frames);
        assert_eq!(l, r, "a mono source must feed both channels");
    }

    /// A rate that is not the model's must be converted, or the separation runs on a
    /// pitch-shifted track and the stems come back subtly wrong rather than failing.
    #[test]
    fn a_wav_at_another_rate_is_resampled_to_the_model_rate() {
        let src_rate = 22_050u32;
        let frames = src_rate as usize / 10;
        let samples: Vec<f32> = (0..2 * frames).map(|i| (i % 100) as f32 / 100.0).collect();
        let wav =
            crate::inference::media::audio_io::write_wav_planar(&samples, 2, frames, src_rate);
        let (l, _) = decode_to_stereo_44k(&wav).expect("decode");
        let want = frames * SAMPLE_RATE as usize / src_rate as usize;
        assert!(
            (l.len() as i64 - want as i64).abs() <= 2,
            "expected about {want} frames after resampling, got {}",
            l.len()
        );
    }

    #[test]
    fn a_non_audio_upload_is_reported() {
        assert!(decode_to_stereo_44k(b"this is not audio").is_err());
        assert!(decode_to_stereo_44k(&[]).is_err());
    }
}
