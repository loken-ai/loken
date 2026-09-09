//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Decode arbitrary audio bytes (WAV, MP3, FLAC, OGG-Vorbis, ...) ->
/// mono f32 PCM at whisper's 16 kHz sample rate. WAV goes through hound
/// for low overhead; everything else is dispatched to symphonia.
///
/// Takes `Vec<u8>` by value so the symphonia branch can move the buffer
/// into Cursor without cloning. symphonia's MediaSourceStream requires
/// `Box<dyn MediaSource + Send + Sync + 'static>`, so we need an owned
/// Cursor - a borrowed `&[u8]` slice can't satisfy 'static. For a 5 MB
/// MP3 upload that's a 5 MB heap copy saved per request. The WAV branch
/// just lends `&bytes` since hound is happy with a Cursor<&[u8]>.
/// Any container the server reads, as interleaved stereo at the model's rate.
///
/// Built on the mono decoder that already handles six containers: it resamples to 16 kHz
/// for speech, so the samples are re-stretched here rather than decoded twice. Mono is
/// duplicated to both channels, which is what a mono source means to a stereo model.
pub(super) fn decode_any_to_stereo_44k(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let (mono16k, rate) = decode_audio_to_mono_f32_16k(bytes.to_vec())?;
    if mono16k.is_empty() {
        anyhow::bail!("the audio decoded to nothing");
    }
    let target = crate::inference::model::stable_audio::SAMPLE_RATE;
    let src_rate = if rate == 0 { 16_000 } else { rate };
    let ratio = target as f64 / src_rate as f64;
    let out_frames = ((mono16k.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_frames * 2);
    for i in 0..out_frames {
        // Linear interpolation between the two neighbouring source samples.
        let pos = i as f64 / ratio;
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let a = *mono16k.get(i0).unwrap_or(&0.0);
        let b = *mono16k.get(i0 + 1).unwrap_or(&a);
        let v = a + (b - a) * frac;
        out.push(v);
        out.push(v);
    }
    Ok(out)
}

pub(super) fn decode_audio_to_mono_f32_16k(bytes: Vec<u8>) -> anyhow::Result<(Vec<f32>, u32)> {
    let kind = detect_audio_format(&bytes);
    match kind {
        AudioFormat::Wav => decode_wav_to_mono_f32_16k(&bytes),
        AudioFormat::Other => decode_symphonia_to_mono_f32_16k(bytes),
    }
}

/// Coarse audio format detector. Distinguishes "this is a WAV (hound
/// path)" from "this is anything else (symphonia path)". We don't
/// strictly need a finer-grained classification - symphonia probes the
/// container itself.
pub(super) enum AudioFormat {
    Wav,
    Other,
}

pub(super) fn detect_audio_format(bytes: &[u8]) -> AudioFormat {
    // RIFF****WAVE -> classic WAV.
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return AudioFormat::Wav;
    }
    AudioFormat::Other
}

/// Decode WAV bytes -> mono f32 PCM at 16 kHz. Handles any input sample
/// rate by passing through rubato's polyphase resampler (anti-aliased).
pub(super) fn decode_wav_to_mono_f32_16k(bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    use std::io::Cursor;
    let mut reader = hound::WavReader::new(Cursor::new(bytes))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        anyhow::bail!("zero-channel WAV");
    }

    // Branch by sample format up front. The previous code converted
    // Float WAV samples to i32 just to divide them back to f32 in
    // the next step - a redundant pass that also cost 1 ULP at the
    // i32 round-trip. Each branch produces the per-sample f32 in
    // [-1, 1] directly so the mono-down-mix consumes a single
    // pre-normalised stream.
    let normalized: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_amp = match spec.bits_per_sample {
                8 => 1i64 << 7,
                16 => 1i64 << 15,
                24 => 1i64 << 23,
                32 => 1i64 << 31,
                b => anyhow::bail!("unsupported bit depth: {b}"),
            } as f32;
            reader
                .samples::<i32>()
                .map(|r| r.map(|s| s as f32 / max_amp))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };

    let mono: Vec<f32> = if channels == 1 {
        normalized
    } else {
        let inv_ch = 1.0 / channels as f32;
        normalized
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() * inv_ch)
            .collect()
    };

    let target_sr = crate::inference::engine::audio_engine::WHISPER_SAMPLE_RATE;
    let resampled = if spec.sample_rate == target_sr {
        mono
    } else {
        resample_to_16k(&mono, spec.sample_rate, target_sr)?
    };
    Ok((resampled, target_sr))
}

/// Decode a WAV into interleaved STEREO f32 at 44.1 kHz (Stable Audio's
/// native rate). Mono input is duplicated to both channels; other rates are
/// resampled per channel.
pub(super) fn decode_wav_to_stereo_44k(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    use std::io::Cursor;
    // WAV directly; anything else through the general container decoder.
    //
    // Every audio field reads the six formats its file picker offers. Accepting WAV alone
    // here would reject an MP3 with a raw decoder error, on a control whose own dialog
    // said it was fine.
    let mut reader = match hound::WavReader::new(Cursor::new(bytes)) {
        Ok(r) => r,
        Err(hound_err) => {
            return decode_any_to_stereo_44k(bytes).map_err(|e| {
                anyhow::anyhow!("{e} (as a WAV: {hound_err}). {ACCEPTED_AUDIO_FORMATS}")
            })
        }
    };
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        anyhow::bail!("zero-channel WAV");
    }
    let normalized: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max_amp = match spec.bits_per_sample {
                8 => 1i64 << 7,
                16 => 1i64 << 15,
                24 => 1i64 << 23,
                32 => 1i64 << 31,
                b => anyhow::bail!("unsupported bit depth: {b}"),
            } as f32;
            reader
                .samples::<i32>()
                .map(|r| r.map(|s| s as f32 / max_amp))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    // Split to L/R (mono duplicates; >2 channels take the first two).
    let frames = normalized.len() / channels;
    let (mut l, mut r) = (Vec::with_capacity(frames), Vec::with_capacity(frames));
    for fr in normalized.chunks_exact(channels) {
        l.push(fr[0]);
        r.push(if channels > 1 { fr[1] } else { fr[0] });
    }
    let target = crate::inference::model::stable_audio::SAMPLE_RATE;
    let (l, r) = if spec.sample_rate == target {
        (l, r)
    } else {
        (
            resample_to_16k(&l, spec.sample_rate, target)?,
            resample_to_16k(&r, spec.sample_rate, target)?,
        )
    };
    let mut out = Vec::with_capacity(l.len() * 2);
    for i in 0..l.len().min(r.len()) {
        out.push(l[i]);
        out.push(r[i]);
    }
    Ok(out)
}

/// Fold an AudioBuffer<f32> into a mono f32 accumulator.
///
/// Writes the per-frame channel sum (then mean) directly into the
/// tail of `out` instead of allocating a fresh `acc` Vec per packet.
/// A 30 s MP3 emits ~1300 packets - that's ~1300 allocations
/// eliminated from the per-transcription hot path. Mono input takes
/// a fast path that extends directly from the channel slice (no
/// per-sample accumulation loop at all).
///
/// Lifted to module scope so unit tests can construct synthetic
/// AudioBuffers and exercise both the mono fast path and the
/// multi-channel sum/scale path without going through a full
/// symphonia decode.
pub(super) fn append_audio_buffer_as_mono(
    buf: &symphonia::core::audio::AudioBuffer<f32>,
    out: &mut Vec<f32>,
) {
    use symphonia::core::audio::Signal;
    let chans = buf.spec().channels.count();
    let frames = buf.frames();
    if frames == 0 || chans == 0 {
        return;
    }
    // Mono fast path: each frame already has exactly one sample.
    if chans == 1 {
        out.extend_from_slice(buf.chan(0));
        return;
    }
    // Multi-channel: extend the output Vec in-place, sum channels
    // directly into the new region, then scale by 1/chans.
    let start = out.len();
    out.resize(start + frames, 0.0);
    let dst = &mut out[start..];
    for ch in 0..chans {
        let plane = buf.chan(ch);
        for (i, s) in plane.iter().enumerate() {
            dst[i] += *s;
        }
    }
    let inv = 1.0 / chans as f32;
    for v in dst.iter_mut() {
        *v *= inv;
    }
}

/// Decode MP3/FLAC/OGG/AAC/MP4-audio via symphonia -> mono f32 PCM at
/// 16 kHz. Symphonia probes the container itself; we just hand it bytes.
///
/// Strategy: convert each decoded packet to an `AudioBuffer<f32>` using
/// symphonia's sample-format helpers, then average channels to mono.
pub(super) fn decode_symphonia_to_mono_f32_16k(bytes: Vec<u8>) -> anyhow::Result<(Vec<f32>, u32)> {
    use std::io::Cursor;
    use symphonia::core::audio::{AudioBuffer, AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::conv::FromSample;
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use symphonia::core::sample::Sample;

    // Move the input Vec into the Cursor directly - no .to_vec() clone.
    // MediaSourceStream::new requires Box<dyn MediaSource + Send + Sync
    // + 'static>; Cursor<Vec<u8>> satisfies that, Cursor<&[u8]> can't.
    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let hint = Hint::new();
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("symphonia probe: {e}"))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no decodable audio track"))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow::anyhow!("symphonia codec init: {e}"))?;
    let src_sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("unknown source sample rate"))?;

    let mut mono: Vec<f32> = Vec::new();

    // Convert any AudioBufferRef into an owned AudioBuffer<f32> by
    // sample-format dispatch, then average to mono.
    fn fold<S>(src: &AudioBuffer<S>, out: &mut Vec<f32>)
    where
        S: Sample,
        f32: FromSample<S>,
    {
        let spec = *src.spec();
        let frames = src.frames();
        let mut dst = AudioBuffer::<f32>::new(frames as u64, spec);
        dst.render_reserved(Some(frames));
        src.convert(&mut dst);
        append_audio_buffer_as_mono(&dst, out);
    }

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(anyhow::anyhow!("symphonia next_packet: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::IoError(_)) | Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow::anyhow!("symphonia decode: {e}")),
        };
        match decoded {
            AudioBufferRef::F32(buf) => append_audio_buffer_as_mono(&buf, &mut mono),
            AudioBufferRef::F64(buf) => fold(&buf, &mut mono),
            AudioBufferRef::U8(buf) => fold(&buf, &mut mono),
            AudioBufferRef::U16(buf) => fold(&buf, &mut mono),
            AudioBufferRef::U24(buf) => fold(&buf, &mut mono),
            AudioBufferRef::U32(buf) => fold(&buf, &mut mono),
            AudioBufferRef::S8(buf) => fold(&buf, &mut mono),
            AudioBufferRef::S16(buf) => fold(&buf, &mut mono),
            AudioBufferRef::S24(buf) => fold(&buf, &mut mono),
            AudioBufferRef::S32(buf) => fold(&buf, &mut mono),
        }
    }

    let target_sr = crate::inference::engine::audio_engine::WHISPER_SAMPLE_RATE;
    let out = if src_sr == target_sr {
        mono
    } else {
        resample_to_16k(&mono, src_sr, target_sr)?
    };
    Ok((out, target_sr))
}

/// Polyphase resampler (rubato FftFixedIn) -> 16 kHz mono. Used when the
/// uploaded WAV isn't already at whisper's required sample rate. Produces
/// anti-aliased output suitable for ASR.
///
/// Uses rubato's polyphase FFT resampler (FftFixedInOut). Important
/// implementation details to keep audio bit-accurate end-to-end:
///   * The resampler has an inherent latency of `chunk_size_out / 2`
///     frames; we drain it by calling `process_partial(None)` after the
///     real input, then drop that many output frames from the front.
///   * Input is fed in fixed-size chunks (`input_frames_next()`); the
///     final chunk is zero-padded via `process_partial` so the tail of
///     the audio actually reaches the output.
pub(super) fn resample_to_16k(
    src: &[f32],
    src_rate: u32,
    dst_rate: u32,
) -> anyhow::Result<Vec<f32>> {
    use rubato::{FftFixedInOut, Resampler};

    if src.is_empty() {
        return Ok(Vec::new());
    }
    // A corrupt WAV can carry sample_rate==0; guard before it reaches rubato /
    // the capacity math (src_rate as f64 in the denominator -> inf -> huge alloc).
    if src_rate == 0 {
        anyhow::bail!("invalid audio: sample rate is 0");
    }
    if src_rate == dst_rate {
        return Ok(src.to_vec());
    }

    // 1024 frames at 44.1 kHz ≈ 23 ms - small enough that the warm-up
    // latency is short, large enough that FFT overhead amortizes.
    let mut resampler = FftFixedInOut::<f32>::new(
        src_rate as usize,
        dst_rate as usize,
        1024, // desired input chunk; rubato may round up
        1,    // channels (mono)
    )?;

    let in_chunk = resampler.input_frames_next();
    let out_delay = resampler.output_delay();
    // Expected output length, computed from the *true* input length so
    // the trailing zero-padding doesn't appear in the result.
    let expected = (src.len() as f64 * dst_rate as f64 / src_rate as f64).round() as usize;

    let mut output: Vec<f32> = Vec::with_capacity(expected + out_delay);
    // Reusable output buffer; rubato's process() convenience method
    // allocates Vec<Vec<f32>> on every call. With a 30s 44.1 kHz input
    // and 1024-frame chunks that's ~1300 redundant allocations per
    // transcription request. Pre-allocate once and reuse via
    // process_into_buffer.
    let mut out_buf: Vec<Vec<f32>> = resampler.output_buffer_allocate(true);

    let mut input_pos = 0usize;
    while input_pos + in_chunk <= src.len() {
        let slice = &src[input_pos..input_pos + in_chunk];
        let in_buf: [&[f32]; 1] = [slice];
        let (_, out_len) = resampler
            .process_into_buffer(&in_buf, &mut out_buf, None)
            .map_err(|e| anyhow::anyhow!("rubato resample: {e}"))?;
        output.extend_from_slice(&out_buf[0][..out_len]);
        input_pos += in_chunk;
    }
    // Drain remaining < chunk_size frames + flush latency tail.
    // First call: pass the residual. Subsequent calls: pass None to
    // pump rubato's internal buffer. process_partial_into_buffer still
    // allocates a temp zero-padded input vec internally each call  -
    // unavoidable without re-implementing the partial path - but is
    // bounded to O(out_delay / out_chunk) calls, not O(src.len() /
    // in_chunk), so the bulk of allocations are gone.
    let mut tail_done = input_pos >= src.len();
    // None disambiguation: process_partial_into_buffer has a generic
    // Vin parameter that's unused when wave_in is None. Help the
    // compiler infer it.
    let none_in: Option<&[&[f32]]> = None;
    while output.len() < expected + out_delay {
        let (_, out_len) = if tail_done {
            resampler
                .process_partial_into_buffer(none_in, &mut out_buf, None)
                .map_err(|e| anyhow::anyhow!("rubato resample flush: {e}"))?
        } else {
            let tail: &[f32] = &src[input_pos..];
            tail_done = true;
            input_pos = src.len();
            let in_buf: [&[f32]; 1] = [tail];
            resampler
                .process_partial_into_buffer(Some(&in_buf), &mut out_buf, None)
                .map_err(|e| anyhow::anyhow!("rubato resample tail: {e}"))?
        };
        if out_len == 0 {
            break;
        }
        output.extend_from_slice(&out_buf[0][..out_len]);
    }

    // Drop the half-window warmup latency from the front.
    if output.len() > out_delay {
        output.drain(..out_delay);
    }
    output.truncate(expected);
    Ok(output)
}

/// Generative audio (text->sound) - fills the API gap where the media models
/// were CLI-only. JSON body:
///   { "model": "ezaudio", "prompt": "rolling thunder with rain",
///     "seconds": 5, "steps": 50, "cfg": 3.0, "seed": 0 }
/// Returns { data: [{ b64_json: "<wav base64>", content_type: "audio/wav" }] },
/// mirroring /v1/images/generations. Dispatch by model name: `ezaudio` ->
/// text->SFX (OpenSound EzAudio, MIT, commercial-OK). ACE-Step music + Wan
/// video land next (they need the CLI orchestration extracted into a lib fn).
/// The caption a music render is actually conditioned on.
///
/// The composer decides whether a track has vocals from what the caption says, so lyrics
/// against an instrumental caption ("pads, percussions") produce an instrumental for most
/// seeds - the reported "my lyrics are ignored". When lyrics are present and the caption
/// names no voice, say so explicitly.
fn music_caption(prompt: &str, lyrics: &str) -> String {
    const VOCAL_TERMS: [&str; 12] = [
        "vocal",
        "voice",
        "voix",
        "sing",
        "sung",
        "chant",
        "choir",
        "chorus",
        "rap",
        "acapella",
        "a cappella",
        "spoken",
    ];
    let lowered = prompt.to_lowercase();
    if lyrics.trim().is_empty() || VOCAL_TERMS.iter().any(|t| lowered.contains(t)) {
        return prompt.to_string();
    }
    format!("{prompt}, with clear lead vocals singing the written lyrics")
}

/// Where a music render writes, and the configuration it writes under.
///
/// The streaming and non-streaming routes ask for the same render and differ only in how
/// they report its progress, so the request becomes a configuration once here: written out
/// at both, a knob wired on one route is a knob missing from the other.
fn acestep_render_config(
    b: &serde_json::Value,
    prompt: &str,
    seconds: f32,
    steps: usize,
    cfg: f32,
    seed: u64,
    loop_mode: bool,
) -> (std::path::PathBuf, serde_json::Value) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let out_path =
        std::env::temp_dir().join(format!("acestep_api_{}_{}.wav", std::process::id(), nanos));
    let codes_per_section = ((seconds as usize) * 5).clamp(20, 3000);
    let lyrics = str_field(b, "lyrics");
    let mut config = serde_json::json!({
        "caption": music_caption(prompt, &lyrics),
        "lyrics": lyrics,
        "out": out_path.to_string_lossy().to_string(),
        "bpm": b.get("bpm").and_then(|v| v.as_i64()).unwrap_or(120),
        "seed": seed,
        "codes_per_section": codes_per_section,
        "dit_steps": steps.max(1),
        "dit_cfg": cfg,
    });
    let Some(obj) = config.as_object_mut() else {
        return (out_path, config);
    };
    // DiT checkpoint choice: turbo (8-step distilled, default) vs the 50-step
    // sft/base quality checkpoints, in 2B and 4B (xl) sizes.
    if let Some(m) = b.get("dit_model").and_then(|v| v.as_str()) {
        let gguf = match m.to_ascii_lowercase().as_str() {
            "turbo" => Some("acestep-v15-turbo-Q8_0.gguf"),
            "sft" => Some("acestep-v15-sft-Q8_0.gguf"),
            "base" => Some("acestep-v15-base-Q8_0.gguf"),
            "xl-turbo" => Some("acestep-v15-xl-turbo-Q8_0.gguf"),
            "xl-sft" | "xl" => Some("acestep-v15-xl-sft-Q8_0.gguf"),
            "xl-base" => Some("acestep-v15-xl-base-Q8_0.gguf"),
            _ => None,
        };
        if let Some(g) = gguf {
            obj.insert("dit_gguf".to_string(), serde_json::json!(g));
        }
    }
    // Renderer knobs the CLI configs already exploit - passed through when present
    // so the API/GUI can reach the full engine (long tracks, tonality, negatives).
    for key in ["negative_prompt", "keyscale", "language", "time_signature"] {
        if let Some(v) = b.get(key).and_then(|v| v.as_str()) {
            if !v.trim().is_empty() {
                obj.insert(key.to_string(), serde_json::json!(v));
            }
        }
    }
    for key in ["temperature", "top_p", "cfg_scale"] {
        if let Some(v) = b.get(key).and_then(|v| v.as_f64()) {
            obj.insert(key.to_string(), serde_json::json!(v));
        }
    }
    // Length floor: force the LM toward the FULL requested duration instead of
    // stopping at its natural song end (the "asked 120s, got 45s" surprise).
    if loop_mode
        || b.get("force_duration")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    {
        obj.insert(
            "min_codes".to_string(),
            serde_json::json!(codes_per_section),
        );
    }
    (out_path, config)
}

/// The two knobs Stable Audio takes beyond the ones every engine here shares.
///
/// Its guidance default is the model's own recipe rather than the shared 3.0, and
/// `negative_prompt` is what steers its unconditional branch. Both of its routes read them
/// the same way, so they are read in one place.
fn stable_audio_knobs(b: &serde_json::Value) -> (f32, String) {
    let cfg = b
        .get("cfg")
        .and_then(|v| v.as_f64())
        .map_or(7.0, |v| v as f32);
    (cfg, str_field(b, "negative_prompt"))
}

/// The string a request field holds, empty when it is absent or is not a string.
fn str_field(b: &serde_json::Value, key: &str) -> String {
    b.get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The first of several spellings a field arrives under, trimmed.
///
/// Callers disagree about what to call the same thing - a caption is `prompt` to some and
/// `input` to others, what to steer away from is `negative_prompt` or `negative` - and
/// both spellings are honoured. The order here is which one wins when a request carries
/// two; a spelling that is present but holds something other than a string still wins,
/// because the caller named that field and falling through to the other would answer a
/// different request than the one asked.
fn trimmed_field(b: &serde_json::Value, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| b.get(*key))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// The progress event every render route emits while it works.
fn rendering_event(
    phase: &str,
    step: usize,
    total: usize,
    since: std::time::Instant,
    node: Option<&str>,
) -> String {
    serde_json::json!({
        "status": "rendering",
        "phase": phase,
        "phase_label": crate::inference::serve::progress::label(phase),
        "step": step, "total": total,
        "elapsed_ms": since.elapsed().as_millis() as u64,
        "node": node,
    })
    .to_string()
}

/// The first event of a streamed render: the model, and the node that renders it, null
/// when this one runs alone.
fn started_event(model: &str, node: Option<&str>) -> String {
    serde_json::json!({"status": "started", "model": model, "node": node}).to_string()
}

/// The terminal event: one clip, how long it took, and what it cost when that could be
/// measured at all.
fn done_event(b64: String, sample_rate: u32, render_ms: u64, energy_j: Option<f64>) -> String {
    let mut done = serde_json::json!({
        "status": "done",
        "data": [{"b64_json": b64, "content_type": "audio/wav", "sample_rate": sample_rate}],
        "render_ms": render_ms,
    });
    if let Some(j) = energy_j {
        done["energy_j"] = serde_json::json!((j * 10.0).round() / 10.0);
    }
    done.to_string()
}

/// The event a route ends on when there is no clip to send.
fn error_event(message: String) -> String {
    serde_json::json!({"status": "error", "error": message}).to_string()
}

pub(crate) async fn audio_generations(
    _state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    // The job goes to a node whose catalogue holds the model when this one's does not.
    {
        let requested = crate::api::handlers::catalogue::sound_entry(&body.0);
        let holds = |n: &crate::distributed::membership::NodeState| {
            crate::api::handlers::catalogue::serves_sound(n, &requested)
        };
        let served_here = holds(&_state.local_node_state().await);
        if let Some(relayed) = crate::api::handlers::route_media_to_holder(
            &_state,
            &headers,
            if requested.is_empty() {
                "a sound model"
            } else {
                &requested
            },
            served_here,
            true,
            holds,
            &crate::api::handlers::AUDIO_GENERATIONS,
            &body.0,
        )
        .await
        {
            return relayed;
        }
    }
    let err_resp = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    // Pressure protocol (vram_manager): hot demand = the biggest component the
    // requested engine will place. ACE-Step: the larger of the 4B LM / DiT
    // checkpoints. Stable Audio: its whole resident (DiT+decoder checkpoint +
    // t5-base) - it stays loaded between requests. The reserve mirrors the
    // free-VRAM gate the loaders apply, so "fits" here means "will actually
    // place on GPU there" (a 0-reserve fit test passed on a card where the
    // loader's own gate then pushed the DiT to CPU - a minutes-long render).
    {
        let requested = str_field(&body.0, "model").to_ascii_lowercase();
        let hot = if requested.contains("stable-audio") || requested.contains("stable_audio") {
            crate::inference::model::stable_audio::resident_bytes()
        } else {
            let sz = |n: &str| {
                std::fs::metadata(crate::inference::model::acestep::fsq::acestep_gguf(n))
                    .map(|m| m.len())
                    .unwrap_or(0)
            };
            // The language model whole, with its KV cache and step reserve, or the
            // denoiser's weights: whichever is larger is what a card must hold.
            let lm =
                crate::inference::model::acestep::fsq::acestep_gguf("acestep-5Hz-lm-4B-Q8_0.gguf");
            crate::inference::model::acestep::lm::placement_demand(lm.to_str().unwrap_or(""))
                .max(sz("acestep-v15-turbo-Q8_0.gguf"))
        };
        // A node whose cards cannot hold the render whole hands it to a peer that holds
        // the model, before reclaiming anything here.
        let requested = crate::api::handlers::catalogue::sound_entry(&body.0);
        let holds = |n: &crate::distributed::membership::NodeState| {
            crate::api::handlers::catalogue::serves_sound(n, &requested)
        };
        if let Some(relayed) = crate::api::handlers::route_media_to_holder(
            &_state,
            &headers,
            if requested.is_empty() {
                "a sound model"
            } else {
                &requested
            },
            true,
            _state.card_holds(hot),
            holds,
            &crate::api::handlers::AUDIO_GENERATIONS,
            &body.0,
        )
        .await
        {
            return relayed;
        }
        crate::inference::place::vram_manager::ensure_gpu_headroom("music", hot, 2 << 30).await;
    }

    let b = body.0;
    let prompt = trimmed_field(&b, &["prompt", "input"]);
    if prompt.is_empty() {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            "prompt (or input) must not be empty".into(),
        );
    }
    let model = b
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("ezaudio")
        .to_string();
    let lower = model.to_lowercase();
    let is_music = lower.contains("ace") || lower.contains("music");
    let is_sao = lower.contains("stable-audio") || lower.contains("stable_audio");
    // One media job at a time, this one included: a music render shares the cards with
    // the image and video engines.
    let _media_guard = _state.media_lock_for(&model, "sound").await;
    // Music (ACE-Step) proves 6+ minute tracks; EzAudio SFX is trained on ~10 s
    // clips and its latent is seconds*50 frames - a shared 600 s clamp would let
    // an SFX request allocate a 30k-frame latent for guaranteed-garbage output.
    // Stable Audio Open's training window is ~47.5 s (2_097_152 samples).
    let seconds_raw = b.get("seconds").and_then(|v| v.as_f64()).unwrap_or(5.0);
    let mut seconds = if is_music {
        seconds_raw.clamp(0.5, 600.0) as f32
    } else if is_sao {
        seconds_raw.clamp(1.0, 47.0) as f32
    } else {
        seconds_raw.clamp(0.5, 30.0) as f32
    };
    // Say so. Asking for a five-minute effect and getting thirty seconds with no word
    // about it reads as the server ignoring the request - the limit belongs to what the
    // model was trained on, and that is worth one line.
    if (seconds as f64 - seconds_raw).abs() > 0.01 {
        info!(
            "audio: {seconds_raw:.1}s is outside what this model was trained for; \
             rendering {seconds:.1}s"
        );
    }
    // Loop mode (music): `loop: true` + `bars: N` renders a BAR-EXACT segment at the
    // requested bpm/time signature, generates one extra beat of tail, and equal-power
    // crossfades that tail into the head - the returned WAV loops seamlessly and its
    // length is exactly bars * beats * 60/bpm (what a sampler/DAW expects).
    let loop_mode =
        (is_music || is_sao) && b.get("loop").and_then(|v| v.as_bool()).unwrap_or(false);
    // What the guidance steers AWAY from. The music branch already read this; the SFX
    // branch conditioned its uncond pass on the empty string no matter what was sent.
    let negative_body = trimmed_field(&b, &["negative_prompt", "negative"]);
    let loop_bars = b
        .get("bars")
        .and_then(|v| v.as_u64())
        .unwrap_or(4)
        .clamp(1, 64) as f32;
    let loop_bpm = b
        .get("bpm")
        .and_then(|v| v.as_i64())
        .unwrap_or(120)
        .clamp(40, 300) as f32;
    let beats_per_bar = b
        .get("time_signature")
        .and_then(|v| v.as_str())
        .and_then(|t| t.split('/').next())
        .and_then(|n| n.trim().parse::<f32>().ok())
        .filter(|n| (1.0..=16.0).contains(n))
        .unwrap_or(4.0);
    let beat_secs = 60.0 / loop_bpm;
    let loop_len_secs = loop_bars * beats_per_bar * beat_secs;
    if loop_mode {
        // Render the loop body + one extra beat (the crossfade donor) + a safety
        // margin: the ACE renderer's finalize pass trims leading/trailing silence
        // (measured ~3 s on short clips) and plays an intro/decaying outro, so it
        // needs a wide margin; Stable Audio returns the raw window, so one extra
        // beat plus a small pad suffices (and its window caps at ~47 s).
        seconds = if is_sao {
            (loop_len_secs + beat_secs + 2.0).clamp(1.0, 47.0)
        } else {
            (loop_len_secs + beat_secs + 8.0).clamp(0.5, 600.0)
        };
    }
    let steps = b
        .get("steps")
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let cfg = b.get("cfg").and_then(|v| v.as_f64()).unwrap_or(3.0) as f32;
    let seed = b.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
    let mask = b.get("mask").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let t0 = std::time::Instant::now();

    // MIDI (symbolic, MIDI-LLM) returns a .mid file - distinct response shape (no sample rate).
    if lower.contains("midi") {
        // Symbolic rendering is its own feature: without it there is no renderer to
        // call, and answering with audio would be a different format than the one
        // this branch exists to return.
        #[cfg(feature = "midi")]
        {
            let max_tokens = b
                .get("max_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(2046)
                .clamp(6, 8192) as usize;
            let temperature = b.get("temperature").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let topp = b.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.98) as f32;
            let prompt_m = prompt.clone();
            let res = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
                crate::inference::media::midi::render_midi(
                    &prompt_m,
                    "MIDI-LLM_Llama-3.2-1B.Q8_0.gguf",
                    "cuda",
                    max_tokens,
                    temperature,
                    topp,
                    seed,
                )
                .map_err(|e| e.to_string())
            })
            .await;
            let mid = match res {
                Ok(Ok(m)) => m,
                Ok(Err(e)) => {
                    return err_resp(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("midi render failed: {e}"),
                    )
                }
                Err(e) => {
                    return err_resp(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("render task panicked: {e}"),
                    )
                }
            };
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&mid);
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            return Json(serde_json::json!({
                "created": created,
                "model": model,
                "data": [{ "b64_json": b64, "content_type": "audio/midi", "bytes": mid.len() }],
                "render_ms": t0.elapsed().as_millis() as u64,
            }))
            .into_response();
        }
        #[cfg(not(feature = "midi"))]
        return err_resp(
            axum::http::StatusCode::NOT_IMPLEMENTED,
            format!("model '{model}' needs the midi feature, which this build does not carry"),
        );
    }

    let is_sfx = lower.contains("ezaudio") || lower.contains("sfx") || lower.contains("sound");
    if !is_music && !is_sfx && !is_sao {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            format!("model '{model}' not supported - use 'ezaudio' (SFX), 'stable-audio' (SFX/loops), 'ace-step' (music), or 'midi'"),
        );
    }

    // Music streaming (SSE): per-DiT-step progress then the final clip. Music only - its DiT
    // trajectory has a progress hook; the EzAudio SFX path doesn't (yet).
    if is_music && b.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        use axum::response::sse::Event;
        let (out_path, cfg_json) =
            acestep_render_config(&b, &prompt, seconds, steps, cfg, seed, loop_mode);
        let model_s = model.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, usize, usize)>(64);
        let out_read = out_path.clone();
        let energy = crate::energy_report::begin();
        // A music render is the longest job in the fleet; without this it runs to
        // completion after the listener has gone.
        let cancel_m = crate::inference::serve::cancel::CancelToken::new();
        let guard_m = crate::inference::serve::cancel::CancelGuard::new(cancel_m.clone());
        let load_tx = tx.clone();
        // The job record follows the render: every count and every placed part lands in
        // it, which is what a listing of this node shows while the render runs.
        let record = _media_guard.reporter();
        let load_record = record.clone();
        let place_fn = record.placement_fn();
        let cancel_load = cancel_m.clone();
        let handle = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
            let mcfg: crate::inference::model::acestep::music::MusicConfig =
                serde_json::from_value(cfg_json).map_err(|e| format!("bad config: {e}"))?;
            // The render reports its own sampling steps; this reports the phase BEFORE them,
            // where the weights are read - minutes, on this route, that used to arrive as a
            // single "loading" with no position in it. The weight readers count into whatever
            // reporter is published, so nothing in the music stack has to know about this.
            let _counts = crate::inference::serve::progress::scoped::publish(
                crate::inference::serve::progress::per_percent(std::sync::Arc::new(
                    move |phase: &str, done: usize, total: usize| {
                        load_record.note(phase, done, total);
                        // Dropped rather than queued when the client is behind: a stale count
                        // is worth nothing, and blocking the loader to deliver one would make
                        // the load slower.
                        let _ = load_tx.try_send((phase.to_string(), done, total));
                    },
                )),
            );
            let _placed = crate::inference::serve::progress::placement::publish(place_fn);
            // A weight reader that sees the token stops a load nobody waits for.
            let _stop = crate::inference::serve::cancel::scoped::publish(&cancel_load);
            let cb = move |phase: &str, step: usize, total: usize| {
                record.note(phase, step, total);
                let _ = tx.blocking_send((phase.to_string(), step, total));
                cancel_m.bail()
            };
            crate::inference::model::acestep::music::render_with_progress(&mcfg, Some(&cb))
                .map_err(|e| e.to_string())?;
            std::fs::read(&out_read).map_err(|e| format!("read rendered wav: {e}"))
        });
        let node_s = _state.node_name();
        let t0m = std::time::Instant::now();
        let stream = async_stream::stream! {
            // Lives with the stream, not the handler call - see the stable-audio branch.
            let _cancel_guard = guard_m;
            // So does the media gate and the job record: the render runs on past the
            // handler, and both must last as long as it does.
            let _media_guard = _media_guard;
            yield Ok::<_, axum::Error>(Event::default().data(started_event(&model_s, node_s.as_deref())));
            while let Some((phase, step, total)) = rx.recv().await {
                yield Ok(Event::default().data(rendering_event(&phase, step, total, t0m, node_s.as_deref())));
            }
            let result = handle.await;
            _cancel_guard.disarm();
            let _ = std::fs::remove_file(&out_path);
            match result {
                Ok(Ok(wav)) => {
                    let wav = if loop_mode {
                        match seamless_loop_wav(&wav, loop_len_secs, beat_secs, 48_000) {
                            Ok(w) => w,
                            Err(e) => {
                                tracing::warn!("loop post-process failed ({e}); raw render returned");
                                wav
                            }
                        }
                    } else {
                        wav
                    };
                    use base64::Engine;
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
                    let energy_j = crate::energy_report::end_measured(
                        energy, "music", "[/v1/audio/generations]");
                    let render_ms = t0m.elapsed().as_millis() as u64;
                    yield Ok(Event::default().data(done_event(b64, 48_000, render_ms, energy_j)));
                }
                Ok(Err(e)) => yield Ok(Event::default().data(error_event(e))),
                Err(e) => yield Ok(Event::default().data(error_event(format!("render task panicked: {e}")))),
            }
        };
        return axum::response::Sse::new(stream).into_response();
    }

    // Stable Audio streaming (SSE): per-denoise-step progress then the final clip,
    // mirroring the music branch (render's on_step hook feeds the channel).
    // Optional audio-to-audio init (both SAO branches): `init_audio` = b64 WAV,
    // `init_noise_level` = variation strength (schedule sigma_max; ~1 stays
    // close to the source, ~10+ reinterprets it).
    let sao_init: Option<(Vec<f32>, f32)> = if is_sao && b.get("init_audio").is_some() {
        match decode_b64_field(&b, "init_audio").and_then(|bytes| {
            decode_wav_to_stereo_44k(&bytes).map_err(|e| format!("init_audio: {e}"))
        }) {
            Ok(pcm) if !pcm.is_empty() => {
                let level = b
                    .get("init_noise_level")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1.0);
                Some((pcm, level))
            }
            Ok(_) => {
                return err_resp(
                    axum::http::StatusCode::BAD_REQUEST,
                    "init_audio is empty".into(),
                )
            }
            Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
        }
    } else {
        None
    };

    if is_sao && b.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        use axum::response::sse::Event;
        let (sao_cfg, negative) = stable_audio_knobs(&b);
        let sao_prompt = prompt.clone();
        let sao_init_s = sao_init.clone();
        let model_s = model.clone();
        // The render outlives the request unless the sampler can see the token.
        let cancel_s = crate::inference::serve::cancel::CancelToken::new();
        let guard_s = crate::inference::serve::cancel::CancelGuard::new(cancel_s.clone());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, usize, usize)>(64);
        let energy = crate::energy_report::begin();
        let load_tx = tx.clone();
        let record = _media_guard.reporter();
        let load_record = record.clone();
        let handle = tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32), String> {
            // Same as the music branch: the sampler counts its own steps, this counts the
            // weights read before the first one exists to count.
            let _counts = crate::inference::serve::progress::scoped::publish(
                crate::inference::serve::progress::per_percent(std::sync::Arc::new(
                    move |phase: &str, done: usize, total: usize| {
                        load_record.note(phase, done, total);
                        let _ = load_tx.try_send((phase.to_string(), done, total));
                    },
                )),
            );
            crate::inference::model::stable_audio::render_with_init(
                &sao_prompt,
                &negative,
                seconds,
                steps,
                sao_cfg,
                seed,
                sao_init_s.as_ref().map(|(pcm, lvl)| (pcm.as_slice(), *lvl)),
                |phase: &str, step, total| {
                    record.note(phase, step, total);
                    let _ = tx.blocking_send((phase.to_string(), step, total));
                    cancel_s.bail()
                },
            )
            .map_err(|e| e.to_string())
        });
        let node_s = _state.node_name();
        let t0m = std::time::Instant::now();
        let stream = async_stream::stream! {
            // The guard must live as long as the STREAM, not the handler call: it
            // is what turns "the client stopped reading" into a stopped sampler.
            let _cancel_guard = guard_s;
            let _media_guard = _media_guard;
            yield Ok::<_, axum::Error>(Event::default().data(started_event(&model_s, node_s.as_deref())));
            while let Some((phase, step, total)) = rx.recv().await {
                yield Ok(Event::default().data(rendering_event(&phase, step, total, t0m, node_s.as_deref())));
            }
            let rendered = handle.await;
            _cancel_guard.disarm();
            match rendered {
                Ok(Ok((pcm, rate))) => {
                    let mut wav = pcm_to_wav_ch(&pcm, rate, 2);
                    if loop_mode {
                        match seamless_loop_wav(&wav, loop_len_secs, beat_secs, rate as usize) {
                            Ok(w) => wav = w,
                            Err(e) => {
                                tracing::warn!("loop post-process failed ({e}); raw render returned");
                            }
                        }
                    }
                    use base64::Engine;
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
                    let energy_j = crate::energy_report::end_measured(
                        energy, "sfx", "[/v1/audio/generations]");
                    let render_ms = t0m.elapsed().as_millis() as u64;
                    yield Ok(Event::default().data(done_event(b64, rate, render_ms, energy_j)));
                }
                Ok(Err(e)) => yield Ok(Event::default().data(error_event(e))),
                Err(e) => yield Ok(Event::default().data(error_event(format!("render task panicked: {e}")))),
            }
        };
        return axum::response::Sse::new(stream).into_response();
    }

    // EzAudio SFX streaming (SSE): same event shape as the music/stable-audio
    // branches, driven by the denoise-step callback.
    if is_sfx && b.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        use axum::response::sse::Event;
        let sfx_prompt = prompt.clone();
        let sfx_negative = negative_body.clone();
        let model_s = model.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, usize, usize)>(64);
        let energy = crate::energy_report::begin();
        let handle = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
            crate::inference::model::ezaudio::pipeline::render_with_progress(
                &sfx_prompt,
                seconds,
                steps,
                cfg,
                seed,
                mask,
                &sfx_negative,
                |phase: &str, step, total| {
                    let _ = tx.blocking_send((phase.to_string(), step, total));
                },
            )
            .map(|(pcm, _)| pcm)
            .map_err(|e| e.to_string())
        });
        let node_s = _state.node_name();
        let t0m = std::time::Instant::now();
        let stream = async_stream::stream! {
            yield Ok::<_, axum::Error>(Event::default().data(started_event(&model_s, node_s.as_deref())));
            while let Some((phase, step, total)) = rx.recv().await {
                yield Ok(Event::default().data(rendering_event(&phase, step, total, t0m, node_s.as_deref())));
            }
            match handle.await {
                Ok(Ok(pcm)) => {
                    let b64 = pcm_to_wav_base64(&pcm, 24_000);
                    let energy_j = crate::energy_report::end_measured(
                        energy, "sfx", "[/v1/audio/generations]");
                    let render_ms = t0m.elapsed().as_millis() as u64;
                    yield Ok(Event::default().data(done_event(b64, 24_000, render_ms, energy_j)));
                }
                Ok(Err(e)) => yield Ok(Event::default().data(error_event(e))),
                Err(e) => yield Ok(Event::default().data(error_event(format!("render task panicked: {e}")))),
            }
        };
        return axum::response::Sse::new(stream).into_response();
    }

    let t0_all = std::time::Instant::now();
    let energy_all = crate::energy_report::begin();
    let (b64, sr): (String, u32) = if is_music {
        // ACE-Step text->music (48 kHz stereo). render() writes a WAV to cfg.out;
        // build MusicConfig from the request (serde fills the rest via defaults),
        // render to a unique tempfile, read it back, base64 it.
        let (out_path, cfg_json) =
            acestep_render_config(&b, &prompt, seconds, steps, cfg, seed, loop_mode);
        let out_read = out_path.clone();
        let record = _media_guard.reporter();
        let load_record = record.clone();
        let place_fn = record.placement_fn();
        let res = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
            let mcfg: crate::inference::model::acestep::music::MusicConfig =
                serde_json::from_value(cfg_json).map_err(|e| format!("bad config: {e}"))?;
            let _counts = crate::inference::serve::progress::scoped::publish(
                crate::inference::serve::progress::per_percent(std::sync::Arc::new(
                    move |phase: &str, done: usize, total: usize| {
                        load_record.note(phase, done, total)
                    },
                )),
            );
            let _placed = crate::inference::serve::progress::placement::publish(place_fn);
            let cb = move |phase: &str, step: usize, total: usize| {
                record.note(phase, step, total);
                Ok(())
            };
            crate::inference::model::acestep::music::render_with_progress(&mcfg, Some(&cb))
                .map_err(|e| e.to_string())?;
            std::fs::read(&out_read).map_err(|e| format!("read rendered wav: {e}"))
        })
        .await;
        let wav = match res {
            Ok(Ok(w)) => w,
            Ok(Err(e)) => {
                let _ = std::fs::remove_file(&out_path);
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("ace-step render failed: {e}"),
                );
            }
            Err(e) => {
                let _ = std::fs::remove_file(&out_path);
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("render task panicked: {e}"),
                );
            }
        };
        let _ = std::fs::remove_file(&out_path);
        let wav = if loop_mode {
            match seamless_loop_wav(&wav, loop_len_secs, beat_secs, 48_000) {
                Ok(w) => w,
                Err(e) => {
                    tracing::warn!("loop post-process failed ({e}); returning the raw render");
                    wav
                }
            }
        } else {
            wav
        };
        use base64::Engine;
        (
            base64::engine::general_purpose::STANDARD.encode(&wav),
            48_000,
        )
    } else if is_sao {
        // Stable Audio Open: stereo 44.1 kHz, v-diffusion, up to the ~47 s
        // training window. CFG default is the model's recipe (7), not the
        // shared 3.0; `negative_prompt` steers the uncond branch.
        let (sao_cfg, negative) = stable_audio_knobs(&b);
        let sao_prompt = prompt.clone();
        let sao_init_s = sao_init.clone();
        let cancel_n = crate::inference::serve::cancel::CancelToken::new();
        let guard_n = crate::inference::serve::cancel::CancelGuard::new(cancel_n.clone());
        let rendered = tokio::task::spawn_blocking(move || {
            crate::inference::model::stable_audio::render_with_init(
                &sao_prompt,
                &negative,
                seconds,
                steps,
                sao_cfg,
                seed,
                sao_init_s.as_ref().map(|(pcm, lvl)| (pcm.as_slice(), *lvl)),
                |_, _, _| cancel_n.bail(),
            )
        })
        .await;
        guard_n.disarm();
        let (pcm, rate) = match rendered {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("stable-audio render failed: {e}"),
                )
            }
            Err(e) => {
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("render task panicked: {e}"),
                )
            }
        };
        use base64::Engine;
        let mut wav = pcm_to_wav_ch(&pcm, rate, 2);
        if loop_mode {
            match seamless_loop_wav(&wav, loop_len_secs, beat_secs, rate as usize) {
                Ok(w) => wav = w,
                Err(e) => {
                    tracing::warn!("loop post-process failed ({e}); returning the raw render");
                }
            }
        }
        (base64::engine::general_purpose::STANDARD.encode(&wav), rate)
    } else {
        // EzAudio text->SFX. render() returns (pcm_f32, n_samples); native rate
        // 24 kHz (the 2nd value is the SAMPLE COUNT, not the rate).
        let sfx_negative = negative_body.clone();
        let rendered = tokio::task::spawn_blocking(move || {
            crate::inference::model::ezaudio::pipeline::render(
                &prompt,
                seconds,
                steps,
                cfg,
                seed,
                mask,
                &sfx_negative,
            )
        })
        .await;
        let (pcm, _n_samples) = match rendered {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("ezaudio render failed: {e}"),
                )
            }
            Err(e) => {
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("render task panicked: {e}"),
                )
            }
        };
        (pcm_to_wav_base64(&pcm, 24_000), 24_000)
    };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let energy_j = crate::energy_report::end_measured(
        energy_all,
        if is_music { "music" } else { "sfx" },
        "[/v1/audio/generations]",
    );
    let mut body = serde_json::json!({
        "created": created,
        "model": model,
        "data": [{
            "b64_json": b64,
            "content_type": "audio/wav",
            "seconds": seconds,
            "sample_rate": sr,
        }],
        "render_ms": t0_all.elapsed().as_millis() as u64,
    });
    if let Some(j) = energy_j {
        body["energy_j"] = serde_json::json!((j * 10.0).round() / 10.0);
    }
    Json(body).into_response()
}

/// Max accepted bytes on caller-uploaded audio (the multipart
/// `file` field on /v1/audio/transcriptions + /translations, and
/// the base64-encoded `images[0]` audio attachment on the
/// /api/chat ASR path). Sized so an hour of lossless audio goes
/// through as-is: a 60-min mono 16 kHz WAV is ~115 MB raw, and
/// asking the caller to transcode first is asking them to lose
/// material to save the server nothing.
/// Turn a rendered clip into a seamless BAR-EXACT loop: keep `loop_secs` of audio,
/// then equal-power crossfade the following `fade` (up to one beat) of extra tail
/// INTO the loop's head - when the loop wraps, the junction has already been blended
/// with the material that follows it, so there is no click or energy dip. Operates on
/// a canonical s16le stereo WAV (44-byte header) at the given sample rate.
pub(super) fn seamless_loop_wav(
    wav: &[u8],
    loop_secs: f32,
    beat_secs: f32,
    sr: usize,
) -> Result<Vec<u8>, String> {
    const CH: usize = 2;
    if wav.len() < 44 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err("not a canonical WAV".into());
    }
    let data = &wav[44..];
    let n_frames = data.len() / (2 * CH);
    let loop_frames = (loop_secs * sr as f32).round() as usize;
    if loop_frames == 0 || loop_frames > n_frames {
        return Err(format!(
            "loop needs {loop_frames} frames, render has {n_frames}"
        ));
    }
    let fade_frames = ((beat_secs * sr as f32).round() as usize)
        .min(n_frames - loop_frames)
        .min(loop_frames / 2);
    // Cut the loop from the STEADY part of the render: the model plays an intro at
    // the start and a decaying outro at the end (it knows the song is ending), so a
    // window starting at 0 loops into a fade-out. Slide the window as late as the
    // donor tail allows while keeping one fade of material after it, but never past
    // the midpoint of the slack (stays clear of the outro decay).
    let slack = n_frames - loop_frames - fade_frames;
    let start = (slack / 2).min(2 * sr);
    let sample = |frame: usize, ch: usize| -> f32 {
        let i = ((start + frame) * CH + ch) * 2;
        i16::from_le_bytes([data[i], data[i + 1]]) as f32
    };
    let mut out = Vec::with_capacity(44 + loop_frames * CH * 2);
    out.extend_from_slice(&wav[..44]);
    for f in 0..loop_frames {
        for ch in 0..CH {
            let v = if f < fade_frames && fade_frames > 0 {
                // Head blended with the tail that follows the loop point:
                // equal-power so the summed energy stays flat across the seam.
                let t = f as f32 / fade_frames as f32;
                let (a, bmix) = (
                    (t * std::f32::consts::FRAC_PI_2).sin(),
                    (t * std::f32::consts::FRAC_PI_2).cos(),
                );
                sample(f, ch) * a + sample(loop_frames + f, ch) * bmix
            } else {
                sample(f, ch)
            };
            out.extend_from_slice(&(v.clamp(-32768.0, 32767.0) as i16).to_le_bytes());
        }
    }
    // Fix the RIFF/data sizes for the shortened payload.
    let data_len = (loop_frames * CH * 2) as u32;
    out[4..8].copy_from_slice(&(36 + data_len).to_le_bytes());
    out[40..44].copy_from_slice(&data_len.to_le_bytes());
    Ok(out)
}

/// What a rejected upload tells the caller they COULD have sent.
///
/// The decoders here take six containers, and the rejection named none of them - so a
/// caller with an Opus voice note or a 24-bit WAV learns only that it "failed", with
/// nothing to try next. Opus is called out because `.ogg` suggests it works: the Ogg
/// container decodes, the Opus codec inside it does not, and that is not guessable.
pub const ACCEPTED_AUDIO_FORMATS: &str =
    "Accepted: WAV, MP3, FLAC, OGG (Vorbis - not Opus), M4A/AAC";

/// Max accepted bytes of uploaded audio.
///
/// A backstop against a runaway body, not a statement about how long a recording may
/// be. At 25 MB - OpenAI's documented figure - a lossless rip of a single album track
/// was already refused, so transcription and separation rejected exactly the material
/// they exist for. Duration is bounded by what the model can do, and that belongs in
/// the handler that knows the model, not in a byte count here.
pub(crate) const AUDIO_INPUT_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Max accepted character count for TTS input text. The synth path
/// splits long input into sentences and concatenates the audio, so the
/// engine itself has no 4096 limit as OpenAI's tts-1 does; this generous cap only guards
/// against accidental multi-hour jobs. Shared between /v1/audio/speech and the /api/chat TTS
/// path so the two surfaces can't drift.
pub(crate) const TTS_INPUT_MAX_CHARS: usize = 50_000;

/// Boundary check for TTS input text length. Returns the documented
/// "split client-side" hint so OpenAI SDK users get the same error
/// shape on both /v1/audio/speech and /api/chat TTS routes.
pub(super) fn validate_tts_input(input: &str) -> Result<(), String> {
    if input.trim().is_empty() {
        return Err("input must not be empty".to_string());
    }
    let chars = input.chars().count();
    if chars > TTS_INPUT_MAX_CHARS {
        return Err(format!(
            "input is {chars} chars; cap is {TTS_INPUT_MAX_CHARS} - split into multiple requests"
        ));
    }
    Ok(())
}

/// Boundary check for caller-uploaded audio bytes. Sibling to
/// validate_image_input_size; centralised so the multipart audio
/// endpoint and the chat-ASR path can't drift on the cap.
pub(crate) fn validate_audio_input_size(byte_len: usize) -> Result<(), String> {
    if byte_len > AUDIO_INPUT_MAX_BYTES {
        Err(format!(
            "audio file too large: {byte_len} bytes > {} MB cap (compress as MP3/FLAC/OGG)",
            AUDIO_INPUT_MAX_BYTES / (1024 * 1024)
        ))
    } else {
        Ok(())
    }
}

// ------------------------------------------------------------
// OpenAI-compatible TTS - /v1/audio/speech
// ------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
pub(super) struct OpenAISpeechRequest {
    #[serde(default)]
    pub(super) model: Option<String>,
    pub(super) input: String,
    /// OpenAI voice preset name. Maps to a Parler `voice_description`
    /// internally; users can also pass `voice_description` directly.
    #[serde(default)]
    pub(super) voice: Option<String>,
    #[serde(default)]
    pub(super) voice_description: Option<String>,
    /// gpt-4o-mini-tts `instructions` field: free-form text describing
    /// the desired delivery ("Speak in a calm whisper", "Sound
    /// excited"). When set, takes precedence over the `voice` preset
    /// - folded into the Parler description with the voice as a
    /// timbre prefix.
    #[serde(default)]
    pub(super) instructions: Option<String>,
    /// The language to SPEAK, as a BCP-47 code ("fr", "en").
    ///
    /// Reaching French used to mean knowing to type `kyutai` or `piper/fr_FR-...` into
    /// the MODEL field - the language smuggled through a model id, which is a workaround
    /// and not an interface. With this, a caller says what they want spoken and the
    /// server picks a backend that can speak it; an explicit `model` still wins, because
    /// naming one is an instruction.
    #[serde(default)]
    pub(super) language: Option<String>,
    #[serde(default)]
    pub(super) response_format: Option<String>,
    /// Playback speed. Parler doesn't expose a direct speed knob, so the
    /// value is folded into the voice description ("a slow / fast speaker").
    #[serde(default)]
    pub(super) speed: Option<f32>,
    /// Same negative-sentinel-tolerant seed deserializer as the
    /// chat/image surfaces (e5a12db).
    #[serde(
        default,
        deserialize_with = "crate::api::types::deserialize_optional_seed"
    )]
    pub(super) seed: Option<u64>,
    #[serde(default)]
    pub(super) max_steps: Option<u32>,
    /// LogitsProcessor temperature for the parler decoder. 0.0 = pure
    /// greedy (default, deterministic output); higher values add
    /// variability between calls.
    #[serde(default)]
    pub(super) temperature: Option<f64>,
    /// Nucleus-sampling cutoff. Only used when temperature > 0.
    #[serde(default)]
    pub(super) top_p: Option<f64>,
    /// When true, split the input into sentences, synthesise each in
    /// turn, and stream the audio bytes as they're produced. Cuts
    /// latency-to-first-byte for long input. Same response Content-Type
    /// as the non-streaming variant; the bytes are concatenated PCM
    /// (or a single WAV file with the data chunk size left at 0xFFFFFFFF
    /// for `wav`).
    #[serde(default)]
    pub(super) stream: Option<bool>,
    /// How the response is DELIVERED: `audio` (default) is a body of audio, `sse` is an
    /// event stream that says what the server is doing before it has any audio to send.
    /// See [`speech_delivery`] for why this is a field of its own rather than `stream`.
    #[serde(default)]
    pub(super) stream_format: Option<String>,
}

/// How a `/v1/audio/speech` response reaches the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpeechDelivery {
    /// A body of audio - one clip, or the byte stream `stream: true` asks for.
    Audio,
    /// An event stream: phase events while the checkpoint loads and while the text is
    /// spoken, then one terminal event carrying the clip.
    Events,
}

/// Read the delivery the caller asked for.
///
/// `stream` was ALREADY taken on this route, and it means something else: send the audio
/// bytes as they are produced. Overloading it would have changed what every existing
/// caller gets, so the event stream is opted into by OpenAI's own field for exactly this,
/// `stream_format`, whose absent value is the behaviour that shipped.
///
/// An unrecognised value is REFUSED rather than treated as `audio`: a typo that silently
/// returns a body of audio to a client waiting for events reads as a hung server, which is
/// the harder failure to diagnose.
pub(super) fn speech_delivery(stream_format: Option<&str>) -> Result<SpeechDelivery, String> {
    match stream_format.map(str::trim) {
        None | Some("") => Ok(SpeechDelivery::Audio),
        Some(v) if v.eq_ignore_ascii_case("audio") => Ok(SpeechDelivery::Audio),
        Some(v) if v.eq_ignore_ascii_case("sse") => Ok(SpeechDelivery::Events),
        Some(other) => Err(format!(
            "stream_format '{other}' is not supported; use 'audio' (a body of audio, the \
             default) or 'sse' (an event stream carrying load and synthesis progress)"
        )),
    }
}

/// An identifier a client can cancel this synthesis by. Distinct from the media routes'
/// `r`-prefixed ids so two live jobs can never collide in the shared registry.
pub(super) fn next_speech_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!("s{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

/// A spawned task that must not outlive the request that wanted it.
///
/// The load is driven from inside the SSE generator so its progress can be forwarded while
/// it happens, which means it runs in a task of its own - and nothing links that task to
/// the connection. A client that stops reading would leave the checkpoint loading for
/// nobody, and the engine would answer the NEXT request on behalf of a request that has
/// gone. Aborting drops the load future, which fires the cancel guard it carries.
pub(super) struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Map OpenAI voice presets -> a Parler description that produces a
/// roughly comparable timbre. Falls back to "alloy" (neutral) for unknown
/// names.
/// Known TTS voice preset names - kept in sync with the `voice_meta`
/// table in `audio_voices`. Used by request validation to fail fast on
/// typo'd voice names instead of silently producing an "alloy" voice.
pub(super) const KNOWN_VOICES: &[&str] = &[
    "alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "ballad", "coral", "sage", "verse",
];

pub(super) fn openai_voice_to_description(voice: &str, speed: f32) -> String {
    let speed_phrase = if speed < 0.8 {
        "slowly with deliberate pacing"
    } else if speed > 1.2 {
        "quickly with brisk pacing"
    } else {
        "at a natural moderate pace"
    };
    let core = match voice.to_lowercase().as_str() {
        // Original tts-1 / tts-1-hd presets
        "echo" => "A male speaker with a deep, warm voice",
        "fable" => "A male British-accented speaker, expressive and clear",
        "onyx" => "A male speaker with a low, authoritative tone",
        "nova" => "A female speaker with a bright, friendly voice",
        "shimmer" => "A female speaker, soft and warm",
        // gpt-4o-mini-tts presets (added Sept 2024)
        "ash" => "A male speaker with a calm, measured baritone",
        "ballad" => "A female speaker with a melodic, lyrical lilt",
        "coral" => "A female speaker with a soft, friendly cadence",
        "sage" => "A male speaker with a thoughtful, reflective tone",
        "verse" => "A female speaker with a poetic, expressive voice",
        // "alloy" and unknowns
        _ => "A clear, neutral English speaker",
    };
    format!(
        "{core}, delivering speech {speed_phrase}. The recording is high quality with the speaker's voice close-up and crisp.",
    )
}

pub(super) fn pcm_to_wav_bytes(pcm: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    // 16-bit PCM WAV. We hand-encode to avoid pulling another dep.
    let n_samples = pcm.len() as u32;
    let byte_rate = sample_rate * 2; // 1 channel * 2 bytes/sample
    let block_align: u16 = 2;
    let data_size = n_samples * 2;
    let chunk_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt subchunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    // Same write-into-pre-sized-region pattern as pcm_to_wav_base64
    // - avoids N extend_from_slice calls in the per-sample loop.
    let data_start = out.len();
    out.resize(data_start + pcm.len() * 2, 0);
    write_i16_le_into(&mut out[data_start..], pcm);
    Ok(out)
}

/// Constant bitrate of the MP3 the speech endpoint returns; the encoder snaps it to the
/// nearest value the MPEG version of the checkpoint's sample rate allows.
const MP3_BITRATE_KBPS: u32 = 96;

/// Mono f32 samples as one MP3 stream: the whole tail padded, the info header in front.
/// A sample rate no MPEG version carries (anything but 8 to 48 kHz on the standard
/// ladder) is refused by the encoder, and the refusal reaches the client.
pub(super) fn pcm_to_mp3_bytes(pcm: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let mut enc = rusty_mp3::Mp3Encoder::new(rusty_mp3::Mp3EncoderConfig {
        bitrate_kbps: MP3_BITRATE_KBPS,
        vbr_quality: None,
    });
    enc.push_pcm_f32(pcm, 1, sample_rate)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    enc.finish();
    let mut out = Vec::with_capacity(pcm.len() / 8);
    loop {
        match enc.next_packet() {
            Ok(packet) => out.extend_from_slice(&packet),
            Err(rusty_mp3::error::Error::Eof) => break,
            Err(rusty_mp3::error::Error::Again) => break,
            Err(e) => return Err(anyhow::anyhow!("{e}")),
        }
    }
    Ok(out)
}

pub(super) fn pcm_to_raw_le_bytes(pcm: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; pcm.len() * 2];
    write_i16_le_into(&mut out, pcm);
    out
}

/// Write f32 samples in [-1, 1] as 16-bit little-endian PCM into `dst`.
/// `dst.len()` must be exactly `pcm.len() * 2`. Out-of-range samples
/// saturate at i16::MAX / i16::MIN - matches the convention used by
/// hound's int-format writer and ffmpeg's `pcm_s16le` encoder.
pub(super) fn write_i16_le_into(dst: &mut [u8], pcm: &[f32]) {
    debug_assert_eq!(dst.len(), pcm.len() * 2);
    for (i, &s) in pcm.iter().enumerate() {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let bytes = v.to_le_bytes();
        dst[2 * i] = bytes[0];
        dst[2 * i + 1] = bytes[1];
    }
}

/// Linearly resample `pcm` to play `speed`x faster (or slower). speed=2.0
/// halves the duration; speed=0.5 doubles it. Pitch shifts along with
/// rate, matching the simple-resample semantics OpenAI's reference uses
/// for its `speed` knob. Linear interpolation is fine for the
/// [0.25, 4.0] range and 24/44.1 kHz speech audio.
pub(super) fn apply_speed_linear(pcm: &[f32], speed: f32) -> Vec<f32> {
    if pcm.is_empty() || !speed.is_finite() || speed <= 0.0 {
        return pcm.to_vec();
    }
    let n_in = pcm.len();
    let n_out = ((n_in as f32) / speed).round() as usize;
    if n_out == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let src = i as f32 * speed;
        let lo = src.floor() as usize;
        if lo + 1 >= n_in {
            out.push(pcm[n_in - 1]);
        } else {
            let frac = src - lo as f32;
            out.push(pcm[lo] * (1.0 - frac) + pcm[lo + 1] * frac);
        }
    }
    out
}

pub(crate) async fn audio_speech(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let err_resp = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let req: OpenAISpeechRequest = match serde_json::from_value(body.0.clone()) {
        Ok(r) => r,
        Err(e) => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid request body: {e}"),
            )
        }
    };
    if let Err(e) = validate_tts_input(&req.input) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }

    // WHICH BACKEND CAN SPEAK THIS. A named model is an instruction and wins; a language
    // with no model picks an engine that can say it. Neither silently substitutes: a
    // language nothing here speaks is REFUSED, because the alternative - the default
    // English model reading French - returns 200 and sounds wrong, which is the harder
    // failure to diagnose.
    let requested_model = match (req.model.as_deref(), req.language.as_deref()) {
        (Some(m), _) if !m.trim().is_empty() => Some(m.to_string()),
        (_, Some(lang)) if !lang.trim().is_empty() => {
            let code = lang.trim().to_lowercase();
            let bare = code.split('-').next().unwrap_or(&code);
            match speakable_backend(bare) {
                Some(m) => {
                    info!("speech: language '{lang}' -> {m}");
                    Some(m)
                }
                None => {
                    return err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!(
                            "no voice here speaks '{lang}'. Speakable: en, fr (Kyutai); \
                             any language with a Piper voice installed, by naming it as \
                             the model (e.g. \"piper/de_DE-thorsten-medium\")"
                        ),
                    )
                }
            }
        }
        _ => req.model.clone(),
    };
    // The voice goes to a node whose catalogue holds its backend when this one's does not.
    if let Some(wanted) = requested_model.as_deref() {
        let holds = |n: &crate::distributed::membership::NodeState| {
            crate::api::handlers::catalogue::serves_speech(n, wanted)
        };
        let served_here = holds(&state.local_node_state().await);
        if let Some(relayed) = crate::api::handlers::route_media_to_holder(
            &state,
            &headers,
            wanted,
            served_here,
            true,
            holds,
            &crate::api::handlers::AUDIO_SPEECH,
            &body.0,
        )
        .await
        {
            return relayed;
        }
    }
    // The voice loads whole on one card: an idle resident is reclaimed to make room, and a
    // node whose cards cannot hold it hands the request to a peer that has the voice.
    if let Some(wanted) = requested_model.as_deref() {
        use crate::api::handlers::catalogue;
        let demand =
            catalogue::listed_size(&state, |id| catalogue::speech_entry_matches(id, wanted))
                .await
                .map(catalogue::whole_load_demand)
                .unwrap_or(0);
        let holds =
            |n: &crate::distributed::membership::NodeState| catalogue::serves_speech(n, wanted);
        if let Some(relayed) = crate::api::handlers::route_media_to_holder(
            &state,
            &headers,
            wanted,
            true,
            state.card_holds(demand),
            holds,
            &crate::api::handlers::AUDIO_SPEECH,
            &body.0,
        )
        .await
        {
            return relayed;
        }
        crate::inference::place::vram_manager::ensure_gpu_headroom(
            "tts",
            demand,
            catalogue::WHOLE_LOAD_RESERVE,
        )
        .await;
    }
    let _job = state.media_note(requested_model.as_deref().unwrap_or("speech"), "speech");

    // Validate `voice` against the documented preset list - unless
    // the caller is overriding via our `voice_description` extension,
    // in which case the voice field is ignored downstream.
    // Without this, an unknown voice (e.g. "fred", "marvin", typos)
    // silently routes to the "alloy" default in openai_voice_to_description,
    // so callers expecting a specific voice get the wrong one with no
    // warning.
    if req.voice_description.is_none() {
        if let Some(v) = req.voice.as_deref() {
            let lower = v.to_lowercase();
            if !KNOWN_VOICES.contains(&lower.as_str()) {
                return err_resp(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!(
                        "voice '{v}' is unknown; valid presets: {} (or pass `voice_description` to override)",
                        KNOWN_VOICES.join(", ")
                    ),
                );
            }
        }
    }

    let response_format = req
        .response_format
        .as_deref()
        .map(str::to_lowercase)
        .unwrap_or_else(|| "wav".to_string());
    match response_format.as_str() {
        "wav" | "pcm" | "mp3" => {}
        // opus/aac/flac would each need an encoder of their own; a clear error
        // beats the wrong content-type.
        other => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "response_format '{other}' not supported; use 'wav', 'pcm' or 'mp3' (rate follows the loaded checkpoint - mini-v1 ships 44.1 kHz, large-v1 24 kHz)"
                ),
            );
        }
    };

    // Speed validation - OpenAI rejects out-of-range with 400. Was
    // silently clamping (lenient but surprising). Now explicit error
    // for the obvious-bug case (speed > 4 or < 0.25); the default 1.0
    // and the boundaries themselves are accepted.
    let speed = req.speed.unwrap_or(1.0);
    if !speed.is_finite() || !(0.25..=4.0).contains(&speed) {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            format!("speed must be in [0.25, 4.0]; got {speed}"),
        );
    }
    // Description precedence (most specific wins):
    //   1. explicit voice_description (our extension)
    //   2. OpenAI gpt-4o-mini-tts instructions (folded with voice timbre)
    //   3. voice preset alone
    //
    // Cap to 1024 chars (~220 T5 tokens). Parler's T5 encoder pass
    // scales with description length; absurdly long descriptions
    // tie up CPU on the encoder for marginal voice-timbre detail.
    const DESC_MAX_CHARS: usize = 1024;
    let description = if let Some(mut d) = req.voice_description.clone() {
        if d.chars().count() > DESC_MAX_CHARS {
            d = d.chars().take(DESC_MAX_CHARS).collect();
        }
        d
    } else if let Some(instr) = req
        .instructions
        .as_ref()
        .map(|s| {
            if s.chars().count() > DESC_MAX_CHARS {
                s.chars().take(DESC_MAX_CHARS).collect::<String>()
            } else {
                s.to_string()
            }
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        let voice = req.voice.clone().unwrap_or_else(|| "alloy".to_string());
        let timbre = openai_voice_to_description(&voice, speed);
        format!("{timbre} {instr}")
    } else {
        let voice = req.voice.clone().unwrap_or_else(|| "alloy".to_string());
        openai_voice_to_description(&voice, speed)
    };

    // The backend resolved above: the caller's model when they named one, otherwise the
    // one that can speak the language they asked for.
    let model = requested_model.clone();
    if let Some(name) = model.as_deref() {
        if let Err(e) = validate_model_id(name) {
            return e.into_response();
        }
    }
    let delivery = match speech_delivery(req.stream_format.as_deref()) {
        Ok(d) => d,
        Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
    };

    let params = TtsSynthParams {
        voice_description: description,
        // Clamp via the finite-aware helper so NaN/Infinity in top_p
        // (e.g. `{"top_p": "NaN"}`) doesn't propagate into the sampler
        // - Rust's `f64::clamp` returns NaN for NaN inputs, which
        // would silently disable nucleus filtering at runtime.
        // Default 0.95 mirrors Parler-TTS's official inference recipe;
        // pure greedy (None) was loop-prone on short inputs.
        top_p: Some(
            req.top_p
                .map(|v| clamp_finite_f64(v, 0.0, 1.0, 0.95))
                .unwrap_or(0.95),
        ),
        // Default 1.0 matches Parler-TTS's official inference recipe
        // (do_sample=True, temperature=1.0). Greedy (temperature=0.0)
        // is loop-prone on short inputs: the model gets stuck on the
        // final phoneme because argmax keeps picking "stay on the
        // current sound" and never triggers the pad-EOS break.
        // Range [0, 2] still allowed via explicit caller value.
        temperature: clamp_finite_f64(req.temperature.unwrap_or(1.0), 0.0, 2.0, 1.0),
        // When the caller didn't supply a seed, draw a fresh one from
        // the system clock so successive `temperature > 0` calls
        // produce varied output instead of all using seed=0.
        // `temperature = 0` keeps the path deterministic regardless
        // (argmax ignores the RNG).
        seed: req.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        }),
        // Clamp to a sensible band: 16 (≈ 0.3 s of audio) prevents
        // 0/runt outputs; 4096 (≈ 80 s) caps a single sentence so a
        // pathological input can't tie the engine up indefinitely.
        max_steps: req
            .max_steps
            .map(|v| (v as usize).clamp(16, 4096))
            .unwrap_or(1024),
    };

    // The event stream owns its load: it has to START before the checkpoint is read, or
    // the phase it exists to report has already happened by the time the client is
    // connected. Every other delivery loads here, exactly as it always did.
    if delivery == SpeechDelivery::Events {
        return tts_events_response(
            state.0.clone(),
            model,
            req.input.clone(),
            params,
            response_format,
            speed,
        );
    }

    if let Err(e) = ensure_tts_model_loaded(&state, model.as_deref()).await {
        return err_resp(http_status_for_load_error(&e), e);
    }

    if req.stream.unwrap_or(false) {
        return tts_stream_response(
            state.tts_engine.clone(),
            req.input.clone(),
            params,
            &response_format,
            speed,
        )
        .await;
    }

    // Non-stream path: chunk long input by sentence and concatenate the
    // PCM so a single synth call doesn't get truncated by the decoder's
    // per-call step cap. Short input (<= one sentence) hits the engine
    // once with no overhead. Sentence chunks share the encoded-
    // description cache so this is essentially free.
    let synth_start = std::time::Instant::now();
    let energy = crate::energy_report::begin();
    let sentences = split_into_sentences(&req.input);
    let mut all_pcm: Vec<f32> = Vec::new();
    let mut sample_rate = crate::inference::engine::tts_engine::TTS_SAMPLE_RATE;
    for (idx, sentence) in sentences.into_iter().enumerate() {
        let mut p = params.clone();
        // Cap per-sentence so a runaway sentence can't dominate; the
        // total length is bounded by `sentences.len() * cap` which is
        // generally fine for OpenAI-shaped inputs.
        p.max_steps = p.max_steps.min(512);
        // ONE seed for every sentence: Parler derives the speaker identity from
        // the sampling path, so a per-sentence seed advance (the old "varied
        // delivery" behavior) audibly CHANGED THE VOICE mid-narration - the
        // user-reported "voices are not respected". Stable identity wins.
        let _ = idx;
        match state.tts_engine.synthesize(sentence, p).await {
            Ok(r) => {
                sample_rate = r.sample_rate;
                all_pcm.extend(r.pcm);
            }
            Err(e) => {
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("tts synth: {e}"),
                )
            }
        }
    }
    crate::energy_report::end(energy, "tts", "[/v1/audio/speech]");
    // Apply playback speed by time-domain resampling (same trick OpenAI's
    // reference uses: `speed=2.0` shortens duration by 2x and shifts pitch
    // up by an octave). The voice-description hint above coaxes parler
    // into a faster *delivery*; resampling guarantees a measurable
    // duration change so clients that check sample counts get the
    // expected result.
    let pcm_speed = if (speed - 1.0).abs() < 1e-3 {
        all_pcm
    } else {
        apply_speed_linear(&all_pcm, speed)
    };
    let result = TtsResult {
        pcm: pcm_speed,
        sample_rate,
    };

    let (body, content_type, ext): (Vec<u8>, String, &str) = match response_format.as_str() {
        "pcm" => (
            pcm_to_raw_le_bytes(&result.pcm),
            format!("audio/L16; rate={}; channels=1", result.sample_rate),
            "pcm",
        ),
        "mp3" => match pcm_to_mp3_bytes(&result.pcm, result.sample_rate) {
            Ok(bytes) => (bytes, "audio/mpeg".to_string(), "mp3"),
            Err(e) => {
                return err_resp(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("mp3 encode: {e}"),
                )
            }
        },
        _ => match pcm_to_wav_bytes(&result.pcm, result.sample_rate) {
            Ok(bytes) => (bytes, "audio/wav".to_string(), "wav"),
            Err(e) => {
                return err_resp(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("wav encode: {e}"),
                )
            }
        },
    };

    let synth_ms = synth_start.elapsed().as_secs_f64() * 1000.0;
    let audio_s = result.pcm.len() as f64 / result.sample_rate as f64;
    info!(
        "Audio speech: voice={voice} format={ext} audio_duration={audio_s:.1}s synth={synth_ms:.0}ms",
        voice = req.voice.as_deref().unwrap_or("alloy"),
    );
    // Set Content-Disposition so browsers + curl -OJ pick a sensible
    // filename instead of the route name. Inline disposition keeps
    // <audio> playback flowing without download prompts.
    let cd = format!("inline; filename=\"speech.{ext}\"");
    let st = format!("synthesize;dur={synth_ms:.1}");
    let mut headers = axum::http::HeaderMap::new();
    if let Ok(hv) = axum::http::HeaderValue::from_str(&content_type) {
        headers.insert(axum::http::header::CONTENT_TYPE, hv);
    }
    if let Ok(hv) = axum::http::HeaderValue::from_str(&cd) {
        headers.insert(axum::http::header::CONTENT_DISPOSITION, hv);
    }
    if let Ok(hv) = axum::http::HeaderValue::from_str(&st) {
        headers.insert("server-timing", hv);
    }
    (headers, body).into_response()
}

/// Ensure the TtsEngine has the requested parler-tts checkpoint loaded.
/// Unloads + reloads when the request asks for a different one (mini-v1
/// ↔ large-v1, or a custom HF id) so the next synth call speaks in the
/// right voice family.
pub(super) async fn ensure_tts_model_loaded(
    state: &APIServer,
    requested: Option<&str>,
) -> Result<(), String> {
    ensure_tts_model_loaded_reporting(state, requested, LoadWatch::default()).await
}

/// [`ensure_tts_model_loaded`], with the weights it reads counted into `watch` - and
/// stoppable through it.
///
/// The watch is what a STREAMING route forwards to its client. Without one the loaders
/// publish no reporter, so every tensor count is dropped on the floor, which is why this
/// route used to answer a cold checkpoint with a minute of silence.
pub(super) async fn ensure_tts_model_loaded_reporting(
    state: &APIServer,
    requested: Option<&str>,
    watch: LoadWatch,
) -> Result<(), String> {
    // Kyutai tts-1.6b-en_fr path (multilingual en/fr Delayed-Streams TTS): a precomputed
    // tts-voices embedding selected by name. Checked before pocket-tts so "kyutai" wins.
    if let Some(voice) = requested.and_then(kyutai_voice_name) {
        let canonical = format!("kyutai-{}", voice.as_deref().unwrap_or("default"));
        let loaded = state.tts_engine.loaded_name().await;
        if loaded.as_deref() != Some(canonical.as_str()) {
            if loaded.is_some() {
                state.tts_engine.unload().await;
            }
            state
                .tts_engine
                .load_kyutai_reporting(voice.clone(), canonical, watch)
                .await
                .map_err(|e| format!("kyutai load: {e:#}"))?;
        }
        return Ok(());
    }

    // Kyutai pocket-tts path: resolve the voice WAV under the configured models
    // dir and load the flow-matching backend.
    if let Some(voice) = requested.and_then(pocket_tts_voice_name) {
        let canonical = format!("pocket-tts-{voice}");
        let loaded = state.tts_engine.loaded_name().await;
        if loaded.as_deref() != Some(canonical.as_str()) {
            if loaded.is_some() {
                state.tts_engine.unload().await;
            }
            let voice_wav = pocket_tts_voice_path(&state.huggingface_models_dir, &voice);
            if !voice_wav.is_file() {
                return Err(format!(
                    "pocket-tts reference voice not found: {} (place a 24 kHz mono WAV at \
                     <hf_models_dir>/pocket-tts-voices/{voice}.wav)",
                    voice_wav.display()
                ));
            }
            state
                .tts_engine
                .load_pocket_tts_reporting(voice_wav, canonical, watch)
                .await
                .map_err(|e| format!("pocket-tts load: {e:#}"))?;
        }
        return Ok(());
    }

    // Native Piper (VITS) path: resolve the voice + its .onnx under the
    // configured HF models dir, and load the full-Rust backend.
    if let Some(voice) = requested.and_then(piper_voice_name) {
        let canonical = format!("piper/{voice}");
        let loaded = state.tts_engine.loaded_name().await;
        if loaded.as_deref() != Some(canonical.as_str()) {
            if loaded.is_some() {
                state.tts_engine.unload().await;
            }
            let onnx = piper_onnx_path(&state.huggingface_models_dir, &voice);
            if !onnx.is_file() {
                return Err(format!(
                    "piper voice not found: {} (expected under <hf_models_dir>/piper/{voice}/)",
                    onnx.display()
                ));
            }
            state
                .tts_engine
                .load_piper_reporting(onnx, canonical, watch)
                .await
                .map_err(|e| format!("piper load: {e:#}"))?;
        }
        return Ok(());
    }

    let loaded = state.tts_engine.loaded_name().await;
    let needs_reload = match (loaded.as_deref(), requested) {
        (Some(cur), Some(req)) => {
            // Normalize: compare on the suffix after `/` so bare names
            // ("parler-tts-mini-v1") match the canonical
            // ("parler-tts/parler-tts-mini-v1").
            let cur_tail = cur.rsplit('/').next().unwrap_or(cur);
            let req_tail = req.rsplit('/').next().unwrap_or(req);
            cur_tail != req_tail
        }
        // Switching from a Piper/pocket-tts/kyutai voice to a Parler request -> reload.
        (Some(cur), None) => {
            cur.starts_with("piper/") || cur.starts_with("pocket-tts") || cur.starts_with("kyutai")
        }
        _ => false,
    };
    if needs_reload {
        state.tts_engine.unload().await;
    }
    if !state.tts_engine.is_loaded().await {
        state
            .tts_engine
            .load_parler_reporting(requested, watch)
            .await
            .map_err(|e| format!("tts load: {e:#}"))?;
    }
    Ok(())
}

/// Split text into rough sentence chunks (period/!/?). Falls back to a
/// length cap so a single very long "sentence" still streams in
/// pieces. Used by streaming TTS to lower latency-to-first-byte.
pub(super) fn split_into_sentences(text: &str) -> Vec<String> {
    const HARD_CAP_CHARS: usize = 240;
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        // Sentence terminators across major scripts:
        //   ASCII (en/fr/de/es/...): . ! ?
        //   CJK (zh/ja/ko):          。 ！ ？  ︒ (small variants)
        //   Devanagari/Arabic/etc:   ।  ؟ ؛
        //   Plus paragraph break.
        let is_terminator = matches!(
            ch,
            '.' | '!' | '?' | '\n'
                | '\u{3002}' // 。 ideographic full stop
                | '\u{ff01}' // ！ fullwidth exclamation
                | '\u{ff1f}' // ？ fullwidth question
                | '\u{0964}' // ।  devanagari danda (Hindi)
                | '\u{061f}' // ؟ Arabic question mark
                | '\u{06d4}' // ۔  Urdu full stop
        );
        if is_terminator || cur.chars().count() >= HARD_CAP_CHARS {
            let trimmed = cur.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    if out.is_empty() && !text.trim().is_empty() {
        out.push(text.trim().to_string());
    }
    out
}

/// Build a WAV header with the data-chunk size left at 0xFFFFFFFF (the
/// classic "streaming WAV" trick). Most players, including ffmpeg and
/// libsndfile, accept this and read until the connection closes.
pub(super) fn wav_streaming_header(sample_rate: u32) -> Vec<u8> {
    let mut hdr = Vec::with_capacity(44);
    hdr.extend_from_slice(b"RIFF");
    hdr.extend_from_slice(&u32::MAX.to_le_bytes()); // chunk size - unknown
    hdr.extend_from_slice(b"WAVE");
    hdr.extend_from_slice(b"fmt ");
    hdr.extend_from_slice(&16u32.to_le_bytes());
    hdr.extend_from_slice(&1u16.to_le_bytes());
    hdr.extend_from_slice(&1u16.to_le_bytes());
    hdr.extend_from_slice(&sample_rate.to_le_bytes());
    hdr.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    hdr.extend_from_slice(&2u16.to_le_bytes());
    hdr.extend_from_slice(&16u16.to_le_bytes());
    hdr.extend_from_slice(b"data");
    hdr.extend_from_slice(&u32::MAX.to_le_bytes()); // data size - unknown
    hdr
}

pub(super) async fn tts_stream_response(
    engine: std::sync::Arc<crate::inference::engine::TtsEngine>,
    input: String,
    base_params: TtsSynthParams,
    response_format: &str,
    speed: f32,
) -> axum::response::Response {
    use async_stream::stream;
    use axum::body::Body;
    use axum::response::IntoResponse;

    let sentences = split_into_sentences(&input);
    let format = response_format.to_string();
    // Engine is loaded by the time we get here (caller calls
    // ensure_tts_model_loaded first), so we can size the WAV header to
    // the actual per-checkpoint sample rate.
    let sample_rate = engine
        .loaded_sample_rate()
        .await
        .unwrap_or(crate::inference::engine::tts_engine::TTS_SAMPLE_RATE);
    let content_type = if format == "mp3" {
        "audio/mpeg".to_string()
    } else if format == "pcm" {
        format!("audio/L16; rate={sample_rate}; channels=1")
    } else {
        "audio/wav".to_string()
    };
    let ext = match format.as_str() {
        "pcm" => "pcm",
        "mp3" => "mp3",
        _ => "wav",
    };
    let cd = format!("inline; filename=\"speech.{ext}\"");

    let sentence_count = sentences.len();
    let stream_started = std::time::Instant::now();
    let body_stream = stream! {
        // Emit the streaming WAV header up front for the wav format so
        // the client can start playback before the first sentence is
        // ready.
        if format == "wav" {
            yield Ok::<Vec<u8>, std::io::Error>(wav_streaming_header(sample_rate));
        }
        let mut bytes_total: usize = 0;
        for (idx, sentence) in sentences.into_iter().enumerate() {
            let mut params = base_params.clone();
            // Each sentence is a self-contained synth call; cap the
            // per-sentence steps tighter so a runaway one doesn't
            // monopolize streaming.
            params.max_steps = params.max_steps.min(512);
            // Advance the seed per sentence so `temperature > 0` produces
            // varied output across the stream instead of all sentences
            // sharing the base seed.
            params.seed = params.seed.wrapping_add(idx as u64);
            match engine.synthesize(sentence, params).await {
                Ok(r) => {
                    // Apply playback-speed time-domain resampling per
                    // chunk so the streaming path matches the non-stream
                    // path's duration semantics. Sentence boundaries
                    // already carry natural silence, so per-chunk
                    // resample doesn't introduce audible glitches in
                    // the [0.25, 4.0] range.
                    let pcm = if (speed - 1.0).abs() < 1e-3 {
                        r.pcm
                    } else {
                        apply_speed_linear(&r.pcm, speed)
                    };
                    // One MP3 stream per sentence: its frames leave as soon as the
                    // sentence is synthesised, and decoders take the streams back to back.
                    let bytes = if format == "mp3" {
                        match pcm_to_mp3_bytes(&pcm, sample_rate) {
                            Ok(b) => b,
                            Err(e) => {
                                tracing::error!("TTS streaming: mp3 encode: {e}");
                                break;
                            }
                        }
                    } else {
                        pcm_to_raw_le_bytes(&pcm)
                    };
                    bytes_total += bytes.len();
                    yield Ok(bytes);
                }
                Err(e) => {
                    // Inline an error marker. There's no robust way to
                    // surface a typed error mid-audio-stream, so emit a
                    // brief silence and log.
                    tracing::error!("TTS streaming: {e}");
                    let silence = vec![0u8; (sample_rate as usize) * 2 / 4]; // 0.25 s
                    bytes_total += silence.len();
                    yield Ok(silence);
                    break;
                }
            }
        }
        let total_ms = stream_started.elapsed().as_millis();
        let audio_s = bytes_total as f64 / 2.0 / sample_rate as f64;
        tracing::info!(
            "Audio speech stream: sentences={sentence_count} audio_duration={audio_s:.1}s total={total_ms}ms"
        );
    };

    let body = Body::from_stream(body_stream);
    (
        [
            (axum::http::header::CONTENT_TYPE, content_type),
            (axum::http::header::CONTENT_DISPOSITION, cd),
        ],
        body,
    )
        .into_response()
}

/// The event-stream variant of `/v1/audio/speech`.
///
/// The defect this exists for: the checkpoint was read by `ensure_tts_model_loaded` BEFORE
/// the response body existed, so a cold engine answered a client with a minute of nothing -
/// indistinguishable, from the outside, from a wedged server. So the load happens INSIDE
/// the generator, and the counts the weight readers already produce are forwarded as they
/// happen. Event vocabulary is the one `/v1/audio/generations` and `/v1/video/generations`
/// already speak: `status` plus `phase` / `phase_label` / `step` / `total`.
pub(super) fn tts_events_response(
    state: APIServer,
    model: Option<String>,
    input: String,
    params: TtsSynthParams,
    response_format: String,
    speed: f32,
) -> axum::response::Response {
    use crate::inference::serve::progress::{label, phase};
    use axum::response::sse::{Event, KeepAlive, Sse};
    use axum::response::IntoResponse;

    let render_id = next_speech_id();
    let cancel = crate::inference::serve::cancel::CancelToken::new();
    let id_for_event = render_id.clone();
    let t0 = std::time::Instant::now();
    // Every event names the node that speaks, null when it runs alone.
    let node = state.node_name();
    let ev = move |mut v: serde_json::Value| {
        v["node"] = serde_json::json!(node);
        Ok::<Event, std::convert::Infallible>(Event::default().data(v.to_string()))
    };

    let stream = async_stream::stream! {
        // The registry entry lives with the STREAM, not with this call. Built in the
        // handler it would be dropped the moment the response was constructed, and the id
        // the first event carries would already name a render nobody could stop - measured.
        let _reg = crate::inference::serve::cancel::registry::Entry::new(&render_id, &cancel);
        yield ev(serde_json::json!({
            "status": "started",
            "model": model,
            "render_id": id_for_event,
            "format": response_format,
        }));

        // Phase 1: the checkpoint.
        {
            // Announced before anything is read, because some backends read through a
            // loader that counts and some do not: a phase with no count in it still has to
            // say it has begun, or a Piper voice would look like a stall.
            yield ev(serde_json::json!({
                "status": "loading",
                "phase": phase::LOAD_MODEL,
                "phase_label": label(phase::LOAD_MODEL),
                "step": 0, "total": 0,
                "elapsed_ms": t0.elapsed().as_millis() as u64,
            }));
            let (load_tx, mut load_rx) = tokio::sync::mpsc::channel::<(String, usize, usize)>(64);
            let report: crate::inference::serve::progress::SharedProgressFn =
                crate::inference::serve::progress::per_percent(std::sync::Arc::new(
                    move |ph: &str, done: usize, total: usize| {
                        // Dropped rather than queued when the client is behind: a stale
                        // count is worth nothing, and blocking the loader to deliver one
                        // would make the load itself slower.
                        let _ = load_tx.try_send((ph.to_string(), done, total));
                    },
                ));
            let watch = LoadWatch { report: Some(report), cancel: Some(cancel.clone()) };
            let load_state = state.clone();
            let load_model = model.clone();
            let mut load_task = AbortOnDrop(tokio::spawn(async move {
                ensure_tts_model_loaded_reporting(&load_state, load_model.as_deref(), watch).await
            }));
            // Drained to CHANNEL CLOSE, not to a "done" the loader sends: the sender lives
            // in the load, so the close is the honest end of it.
            while let Some((ph, done, total)) = load_rx.recv().await {
                yield ev(serde_json::json!({
                    "status": "loading",
                    "phase": ph,
                    "phase_label": label(&ph),
                    "step": done, "total": total,
                    "elapsed_ms": t0.elapsed().as_millis() as u64,
                }));
            }
            let outcome = match (&mut load_task.0).await {
                Ok(r) => r,
                Err(e) => Err(format!("tts load task failed: {e}")),
            };
            if let Err(e) = outcome {
                yield ev(serde_json::json!({"status": "error", "error": e}));
                return;
            }
        }

        // Phase 2: the speech. Chunked by sentence for the same reason the non-stream path
        // is - a single synth call is capped - and counted in those chunks, which is a unit
        // the caller can relate to the text it sent.
        let sentences = split_into_sentences(&input);
        let total = sentences.len();
        let mut rate = state
            .tts_engine
            .loaded_sample_rate()
            .await
            .unwrap_or(crate::inference::engine::tts_engine::TTS_SAMPLE_RATE);
        let energy = crate::energy_report::begin();
        let mut all_pcm: Vec<f32> = Vec::new();
        let mut failed: Option<String> = None;
        yield ev(serde_json::json!({
            "status": "synthesizing",
            "phase": phase::SYNTHESIZE,
            "phase_label": label(phase::SYNTHESIZE),
            "step": 0, "total": total,
            "elapsed_ms": t0.elapsed().as_millis() as u64,
        }));
        for (idx, sentence) in sentences.into_iter().enumerate() {
            // Cancelled BY NAME between chunks. Dropping the connection stops the synth on
            // its own (the engine's guard lives in the call's frame); this is the half a
            // client can act on without hanging up, which is what the id was published for.
            if cancel.is_cancelled() {
                failed = Some("synthesis cancelled".to_string());
                break;
            }
            let mut p = params.clone();
            // Same per-sentence cap the non-stream path uses, and the SAME seed for every
            // sentence: Parler derives the speaker identity from the sampling path, so
            // advancing it per sentence audibly changes the voice mid-narration.
            p.max_steps = p.max_steps.min(512);
            match state.tts_engine.synthesize(sentence, p).await {
                Ok(r) => {
                    rate = r.sample_rate;
                    all_pcm.extend(r.pcm);
                }
                Err(e) => {
                    failed = Some(format!("tts synth: {e}"));
                    break;
                }
            }
            yield ev(serde_json::json!({
                "status": "synthesizing",
                "phase": phase::SYNTHESIZE,
                "phase_label": label(phase::SYNTHESIZE),
                "step": idx + 1, "total": total,
                "elapsed_ms": t0.elapsed().as_millis() as u64,
            }));
        }
        if let Some(e) = failed {
            yield ev(serde_json::json!({"status": "error", "error": e}));
            return;
        }

        // Same speed semantics as the body-of-audio path: resampling, so the duration
        // actually changes rather than only the delivery being described as faster.
        let pcm = if (speed - 1.0).abs() < 1e-3 {
            all_pcm
        } else {
            apply_speed_linear(&all_pcm, speed)
        };
        let (bytes, content_type) = match response_format.as_str() {
            "pcm" => (
                pcm_to_raw_le_bytes(&pcm),
                format!("audio/L16; rate={rate}; channels=1"),
            ),
            _ => match pcm_to_wav_bytes(&pcm, rate) {
                Ok(b) => (b, "audio/wav".to_string()),
                Err(e) => {
                    yield ev(serde_json::json!({"status": "error", "error": format!("wav encode: {e}")}));
                    return;
                }
            },
        };
        let audio_s = pcm.len() as f64 / rate.max(1) as f64;
        let b64 = {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        };
        let energy_j = crate::energy_report::end_measured(energy, "tts", "[/v1/audio/speech]");
        let mut done = serde_json::json!({
            "status": "done",
            "data": [{
                "b64_json": b64,
                "content_type": content_type,
                "sample_rate": rate,
            }],
            "audio_s": (audio_s * 10.0).round() / 10.0,
            "render_ms": t0.elapsed().as_millis() as u64,
        });
        if let Some(j) = energy_j {
            done["energy_j"] = serde_json::json!((j * 10.0).round() / 10.0);
        }
        tracing::info!(
            "Audio speech events: chunks={total} audio_duration={audio_s:.1}s total={}ms",
            t0.elapsed().as_millis()
        );
        yield ev(done);
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub(crate) async fn audio_voices(
    state: axum::extract::State<APIServer>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    // (id, gender, family, accent). family distinguishes the original
    // 6 tts-1 voices from the gpt-4o-mini-tts additions so clients can
    // surface them in two sections.
    let voice_meta: &[(&str, &str, &str, &str)] = &[
        ("alloy", "neutral", "tts-1", "en"),
        ("echo", "male", "tts-1", "en"),
        ("fable", "male", "tts-1", "en-GB"),
        ("onyx", "male", "tts-1", "en"),
        ("nova", "female", "tts-1", "en"),
        ("shimmer", "female", "tts-1", "en"),
        ("ash", "male", "gpt-4o-mini-tts", "en"),
        ("ballad", "female", "gpt-4o-mini-tts", "en"),
        ("coral", "female", "gpt-4o-mini-tts", "en"),
        ("sage", "male", "gpt-4o-mini-tts", "en"),
        ("verse", "female", "gpt-4o-mini-tts", "en"),
    ];
    let voices: Vec<serde_json::Value> = voice_meta
        .iter()
        .map(|(v, gender, family, accent)| {
            serde_json::json!({
                "voice": v,
                "description": openai_voice_to_description(v, 1.0),
                "gender": gender,
                "family": family,
                "accent": accent,
                // `alloy` is the OpenAI documented default when the
                // caller omits the `voice` field on /v1/audio/speech.
                // Surfacing the flag lets voice-picker UIs preselect
                // it without hardcoding the name.
                "is_default": *v == "alloy",
            })
        })
        .collect();
    // Report the *loaded* checkpoint's name + sample_rate when a model
    // is warm; fall back to the default mini-v1 / 44.1 kHz otherwise so
    // pre-load discovery still returns useful values.
    let loaded_name = state.tts_engine.loaded_name().await;
    let loaded_sr = state.tts_engine.loaded_sample_rate().await;
    let loaded = loaded_name.is_some();
    Json(serde_json::json!({
        "object": "list",
        "data": voices,
        // Surface the actual loaded model vs the default fallback so
        // clients can tell warm-cache state from cold.
        "loaded": loaded,
        "model_hint": loaded_name.unwrap_or_else(|| "parler-tts/parler-tts-mini-v1".to_string()),
        "sample_rate": loaded_sr.unwrap_or(crate::inference::engine::tts_engine::TTS_SAMPLE_RATE),
        "channels": 1,
        // Self-describing hints so clients don't need to read source
        // to know what's supported on this server.
        "supported_response_formats": ["wav", "pcm"],
        "max_input_chars": 4096,
        "max_steps_cap": 4096,
        "supports_streaming": true,
        "supports_instructions": true,
    }))
    .into_response()
}
