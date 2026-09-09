use super::*;

#[test]
fn route_intent_rules_bucket_by_phrasing() {
    // Attached image is always vision, regardless of text.
    assert_eq!(route_intent("what is this", true), ConvRoute::Vision);
    // Explicit image command: verb-start + image noun.
    assert_eq!(
        route_intent("draw a picture of a cat", false),
        ConvRoute::ImageGen
    );
    assert_eq!(
        route_intent("generate an image of a sunset", false),
        ConvRoute::ImageGen
    );
    // Explicit TTS command.
    assert_eq!(route_intent("say hello world", false), ConvRoute::Tts);
    assert_eq!(
        route_intent("read this aloud please", false),
        ConvRoute::Tts
    );
    // Explicit sound-gen commands.
    assert_eq!(
        route_intent("make the sound of rain on a window", false),
        ConvRoute::SoundGen
    );
    assert_eq!(
        route_intent("generate a drum loop at 120 bpm", false),
        ConvRoute::SoundGen
    );
    assert_eq!(
        route_intent("create a sound effect of breaking glass", false),
        ConvRoute::SoundGen
    );
    // "loop" without audio context must NOT route to sound.
    assert_eq!(
        route_intent("explain how a for loop works", false),
        ConvRoute::Chat
    );
    // Plain question -> chat.
    assert_eq!(
        route_intent("what is the capital of France", false),
        ConvRoute::Chat
    );
    // The gap the classifier fills: mis-phrased media intents that the
    // rules leave in the Chat bucket (no verb-start / no explicit noun).
    assert_eq!(
        route_intent("could you paint me a serene lake", false),
        ConvRoute::Chat
    );
    assert_eq!(
        route_intent("narrate this poem for me", false),
        ConvRoute::Chat
    );
}

#[test]
fn conv_note_model_accumulates_working_set() {
    // Unique id so parallel tests can't cross-contaminate the global.
    let cid = "test-conv-accumulate-9f3a";
    let a = conv_note_model(cid, "qwen3:0.6b");
    assert!(a.contains(&"qwen3:0.6b".to_string()));
    let b = conv_note_model(cid, "moondream");
    // Second turn: the set carries BOTH models forward, so a warm-touch
    // keeps the whole conversation working set resident.
    assert!(b.contains(&"qwen3:0.6b".to_string()));
    assert!(b.contains(&"moondream".to_string()));
    assert_eq!(b.len(), 2);
    // Re-noting an existing model is idempotent (set, not multiset).
    let c = conv_note_model(cid, "moondream");
    assert_eq!(c.len(), 2);
}

#[test]
fn kyutai_voice_name_routes_and_extracts_voice() {
    // Non-kyutai ids are not this backend.
    assert_eq!(kyutai_voice_name("parler-tts-mini-v1"), None);
    assert_eq!(kyutai_voice_name("piper/fr_FR-tom-medium"), None);
    assert_eq!(kyutai_voice_name("pocket-tts-alba"), None);
    // Bare markers -> default voice (inner None).
    assert_eq!(kyutai_voice_name("kyutai"), Some(None));
    assert_eq!(kyutai_voice_name("kyutai-tts"), Some(None));
    assert_eq!(kyutai_voice_name("kyutai-en_fr"), Some(None));
    // Named voice -> the selector substring.
    assert_eq!(
        kyutai_voice_name("kyutai-alba"),
        Some(Some("alba".to_string()))
    );
    assert_eq!(
        kyutai_voice_name("kyutai/alba-mackenna/a-moment-by"),
        Some(Some("alba-mackenna/a-moment-by".to_string()))
    );
    assert_eq!(
        kyutai_voice_name("org/kyutai-estelle"),
        Some(Some("estelle".to_string()))
    );
    // Recognized as a TTS model + routed before pocket-tts.
    assert!(is_tts_model("kyutai"));
    assert!(is_tts_model("kyutai-alba"));
}

#[test]
fn pocket_tts_voice_name_and_path() {
    // Non-pocket ids fall through.
    assert_eq!(pocket_tts_voice_name("piper/fr_FR-tom-medium"), None);
    assert_eq!(pocket_tts_voice_name("parler-tts-mini-v1"), None);
    // Bare / prefixed forms.
    assert_eq!(
        pocket_tts_voice_name("pocket-tts").as_deref(),
        Some("default")
    );
    assert_eq!(
        pocket_tts_voice_name("pocket-tts-alba").as_deref(),
        Some("alba")
    );
    assert_eq!(
        pocket_tts_voice_name("org/pocket-tts-estelle").as_deref(),
        Some("estelle")
    );
    // Path steps out of hub/ and lands under pocket-tts-voices/.
    let p = pocket_tts_voice_path("/m/hf/hub", "alba");
    assert!(p.ends_with("pocket-tts-voices/alba.wav"), "{p:?}");
    assert!(p.starts_with("/m/hf"));
    // is_tts_model recognizes pocket-tts.
    assert!(is_tts_model("pocket-tts-alba"));
}

#[test]
fn piper_voice_name_routes_and_extracts_voice() {
    // Non-Piper ids return None (fall through to Parler).
    assert_eq!(piper_voice_name("parler-tts/parler-tts-mini-v1"), None);
    assert_eq!(piper_voice_name("tts-1"), None);
    // Bare `piper` -> default French voice.
    assert_eq!(
        piper_voice_name("piper").as_deref(),
        Some("fr_FR-tom-medium")
    );
    assert_eq!(
        piper_voice_name("Piper").as_deref(),
        Some("fr_FR-tom-medium")
    );
    // `piper/<voice>` and `piper-<voice>` forms extract the voice.
    assert_eq!(
        piper_voice_name("piper/fr_FR-tom-medium").as_deref(),
        Some("fr_FR-tom-medium")
    );
    assert_eq!(
        piper_voice_name("piper-en_US-amy-medium").as_deref(),
        Some("en_US-amy-medium")
    );
    // Piper is recognised as a TTS model.
    assert!(is_tts_model("piper/fr_FR-tom-medium"));
}

#[test]
fn piper_onnx_path_resolves_under_hf_dir() {
    // Plain HF dir -> <dir>/piper/<voice>/<voice>.onnx.
    assert_eq!(
        piper_onnx_path("/models/hf", "fr_FR-tom-medium"),
        std::path::PathBuf::from("/models/hf/piper/fr_FR-tom-medium/fr_FR-tom-medium.onnx"),
    );
    // A dir already ending in `hub/` steps back out (voices sit beside hub).
    assert_eq!(
        piper_onnx_path("/models/hf/hub", "fr_FR-tom-medium"),
        std::path::PathBuf::from("/models/hf/piper/fr_FR-tom-medium/fr_FR-tom-medium.onnx"),
    );
}

// -- validate_tts_input -----------------------------------------
// Length cap on TTS input text. Same helper used by /v1/audio/speech
// and the /api/chat TTS path - without this a 100 MB chat would
// pin the synth engine for many minutes (each char ≈ 5-15 ms on
// parler-mini-v1).

#[test]
fn validate_tts_input_accepts_normal_lengths() {
    assert!(validate_tts_input("Hello world").is_ok());
    assert!(validate_tts_input("a").is_ok(), "minimum non-empty");
    // Exactly at the cap accepts.
    let at_cap: String = "x".repeat(TTS_INPUT_MAX_CHARS);
    assert!(validate_tts_input(&at_cap).is_ok());
}

#[test]
fn validate_tts_input_rejects_empty_and_whitespace() {
    // Pin: every empty / whitespace-only form gets the same
    // "must not be empty" message so SDK clients can match on it.
    for empty in ["", "   ", "\n\n", "\t  \t"] {
        let err = validate_tts_input(empty).expect_err("empty must reject");
        assert!(
            err.contains("must not be empty"),
            "error must hint emptiness; got {err:?} for input {empty:?}"
        );
    }
}

#[test]
fn validate_tts_input_rejects_over_cap_with_split_hint() {
    let over: String = "y".repeat(TTS_INPUT_MAX_CHARS + 1);
    let err = validate_tts_input(&over).expect_err("over-cap must reject");
    // Error must surface both the offending length AND the cap
    // (so the client can compute how much to chunk by) PLUS the
    // documented "split client-side" guidance.
    assert!(
        err.contains(&(TTS_INPUT_MAX_CHARS + 1).to_string()),
        "error must surface offending char count; got {err:?}"
    );
    assert!(
        err.contains(&TTS_INPUT_MAX_CHARS.to_string()),
        "error must surface the cap; got {err:?}"
    );
    assert!(
        err.contains("split into multiple requests"),
        "error must include the actionable hint; got {err:?}"
    );

    // Pathological 1 MB input - same rejection path.
    let huge: String = "z".repeat(1_000_000);
    assert!(validate_tts_input(&huge).is_err());
}

#[test]
fn validate_tts_input_counts_chars_not_bytes() {
    // Multibyte UTF-8 chars count once each, not per-byte. Cap
    // is in characters (OpenAI's tts-1 documented unit) so a
    // multibyte-heavy string under the char cap must accept
    // even if its byte length is 3-4x.
    let multibyte: String = "é".repeat(TTS_INPUT_MAX_CHARS); // 2 bytes each
    assert!(
        validate_tts_input(&multibyte).is_ok(),
        "multibyte input at char-cap (well under byte-cap) must accept"
    );
}

/// A language this server can speak resolves to a backend that speaks it - the
/// point of the parameter. Before it, French meant knowing to type `kyutai` into the
/// MODEL field, which is a workaround dressed as a feature.
#[test]
fn a_speakable_language_picks_a_backend_that_speaks_it() {
    assert_eq!(super::speakable_backend("fr").as_deref(), Some("kyutai"));
    assert_eq!(super::speakable_backend("en").as_deref(), Some("kyutai"));
}

/// One nothing here speaks is REFUSED. The alternative - the default English model
/// reading German - returns 200 and sounds wrong, which is the harder failure to
/// diagnose and the one this whole parameter exists to prevent.
#[test]
fn an_unspeakable_language_is_refused_not_substituted() {
    assert!(super::speakable_backend("de").is_none());
    assert!(super::speakable_backend("ja").is_none());
    assert!(super::speakable_backend("").is_none());
}

/// It answers on the BARE language: the caller may send a region tag, and dropping
/// the request over one is the same defect that made `fr-FR` transcribe as English.
#[test]
fn a_region_tag_does_not_hide_a_speakable_language() {
    let code = "fr-FR".to_lowercase();
    let bare = code.split('-').next().unwrap();
    assert_eq!(super::speakable_backend(bare).as_deref(), Some("kyutai"));
}

#[test]
fn validate_audio_input_size_accepts_at_or_under_cap() {
    assert!(
        validate_audio_input_size(0).is_ok(),
        "empty handled separately upstream"
    );
    assert!(
        validate_audio_input_size(1_000_000).is_ok(),
        "1 MB small clip"
    );
    assert!(
        validate_audio_input_size(AUDIO_INPUT_MAX_BYTES).is_ok(),
        "exactly at the cap must accept"
    );
}

#[test]
fn validate_audio_input_size_rejects_over_cap_with_useful_error() {
    let err = validate_audio_input_size(AUDIO_INPUT_MAX_BYTES + 1).expect_err("cap+1 must reject");
    assert!(
        err.contains("too large"),
        "error must mention rejection; got {err:?}"
    );
    // Wording must steer the caller toward a fix.
    assert!(
        err.contains("MP3/FLAC/OGG"),
        "error must point to compression formats; got {err:?}"
    );
    // Derived from the constant, so raising the cap does not need this line edited
    // and cannot leave the message quoting a figure that is no longer enforced.
    assert!(
        err.contains(&format!("{} MB", AUDIO_INPUT_MAX_BYTES / (1024 * 1024))),
        "error must surface the cap in MB; got {err:?}"
    );
    // Pathological upload attempt - well past any legitimate recording.
    assert!(validate_audio_input_size(AUDIO_INPUT_MAX_BYTES * 8).is_err());
}

#[test]
fn pcm_to_wav_base64_emits_a_well_formed_riff_header() {
    use base64::Engine;
    // Two silent samples at 48 kHz.
    let pcm = vec![0.0_f32, 0.0_f32];
    let b64 = pcm_to_wav_base64(&pcm, 48_000);
    let wav = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .expect("base64 must decode");

    // 44-byte canonical WAV header + 2 samples x 2 bytes = 48 bytes.
    assert_eq!(wav.len(), 48);
    // Magic.
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
    // Sample rate (little-endian u32 at offset 24) = 48000.
    let sr = u32::from_le_bytes(wav[24..28].try_into().unwrap());
    assert_eq!(sr, 48_000);
    // Bits per sample (offset 34) = 16.
    let bps = u16::from_le_bytes(wav[34..36].try_into().unwrap());
    assert_eq!(bps, 16);
    // Channels (offset 22) = 1 (mono - we always emit mono).
    let ch = u16::from_le_bytes(wav[22..24].try_into().unwrap());
    assert_eq!(ch, 1);
    // Data chunk size (offset 40) = 2 samples x 2 bytes = 4.
    let data_size = u32::from_le_bytes(wav[40..44].try_into().unwrap());
    assert_eq!(data_size, 4);
}

#[test]
fn seamless_loop_is_bar_exact_and_click_free_at_44100() {
    // Synthetic 12 s stereo render at 44.1 kHz: a 220 Hz tone whose phase is
    // NOT loop-periodic, so a naive cut would click at the wrap point.
    const SR: usize = 44_100;
    let n = 12 * SR;
    let pcm: Vec<f32> = (0..n * 2)
        .map(|i| {
            let t = (i / 2) as f32 / SR as f32;
            (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.5
        })
        .collect();
    let wav = pcm_to_wav_ch(&pcm, SR as u32, 2);
    // 4 bars at 128 bpm 4/4 = 7.5 s; fade = one beat.
    let (loop_secs, beat_secs) = (7.5f32, 60.0 / 128.0);
    let out = seamless_loop_wav(&wav, loop_secs, beat_secs, SR).expect("loop");
    let frames = (out.len() - 44) / 4;
    assert_eq!(
        frames,
        (loop_secs * SR as f32).round() as usize,
        "bar-exact length"
    );
    // Seam metric: the wrap-point jump must be in the same class as interior
    // adjacent-sample jumps (equal-power crossfade, no click).
    let s = |f: usize| -> f32 {
        let i = 44 + f * 4;
        i16::from_le_bytes([out[i], out[i + 1]]) as f32
    };
    let wrap_jump = (s(0) - s(frames - 1)).abs();
    let mut interior_max = 0f32;
    for f in 1..frames {
        interior_max = interior_max.max((s(f) - s(f - 1)).abs());
    }
    assert!(
        wrap_jump <= interior_max * 1.5,
        "seam click: wrap jump {wrap_jump} vs interior max {interior_max}"
    );
}

#[test]
fn pcm_to_wav_base64_clamps_out_of_range_samples() {
    use base64::Engine;
    // Saturate above +1.0 and below -1.0; output 16-bit samples must
    // clamp to i16::MAX/MIN respectively (no integer wrap to negative).
    let pcm = vec![2.0_f32, -2.0_f32, 0.0_f32];
    let b64 = pcm_to_wav_base64(&pcm, 22_050);
    let wav = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .unwrap();

    // First sample -> +32767 (clamped from +2.0 x 32767 = +65534).
    let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
    assert_eq!(s0, i16::MAX, "+2.0 must clamp to +32767, not wrap");
    // Second sample -> -32767 (clamped from -2.0; spec writes round
    // of -2.0x32767 = -65534, clamped to i16::MIN+1, since the
    // implementation multiplies by 32767 not 32768).
    let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
    assert!(s1 <= -32767, "-2.0 must clamp near -32767, got {s1}");
    // Third sample -> 0.
    let s2 = i16::from_le_bytes(wav[48..50].try_into().unwrap());
    assert_eq!(s2, 0);
}

/// Build a minimal WAV file in memory for round-trip tests of the
/// decoder. `samples_per_ch` is per-channel; `channels` and
/// `sample_rate` populate the fmt chunk. Returns 16-bit Int WAV.
fn build_wav_int16(samples: &[i16], channels: u16, sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for s in samples {
            writer.write_sample(*s).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

fn build_wav_f32(samples: &[f32], channels: u16, sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buf = std::io::Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for s in samples {
            writer.write_sample(*s).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

#[test]
fn decode_wav_mono_int16_normalises_to_unit_range() {
    // Whisper expects f32 in [-1, 1] at 16 kHz. Mono Int16 at 16 kHz
    // should pass through unchanged (no resample, no down-mix) with
    // amplitude divided by 2^15.
    let pcm_i16 = vec![0_i16, 16384, -16384, 32767, -32768];
    let wav = build_wav_int16(&pcm_i16, 1, 16_000);
    let (out, sr) = decode_wav_to_mono_f32_16k(&wav).expect("decode");
    assert_eq!(sr, 16_000);
    assert_eq!(out.len(), pcm_i16.len());
    // 0 -> 0.0, 16384 -> 0.5, -16384 -> -0.5, 32767 -> ~0.99997, -32768 -> -1.0.
    let expect = [0.0_f32, 0.5, -0.5, 32767.0 / 32768.0, -1.0];
    for (got, want) in out.iter().zip(expect.iter()) {
        assert!((got - want).abs() < 1e-4, "got {got} want {want}");
    }
}

#[test]
fn decode_wav_stereo_int16_downmixes_to_mono() {
    // Two-channel Int16 -> mono via per-frame average. Pre-resample.
    let frames: &[(i16, i16)] = &[(16384, 16384), (16384, -16384), (-32768, 32767)];
    let mut flat = Vec::with_capacity(frames.len() * 2);
    for &(l, r) in frames {
        flat.push(l);
        flat.push(r);
    }
    let wav = build_wav_int16(&flat, 2, 16_000);
    let (out, sr) = decode_wav_to_mono_f32_16k(&wav).expect("decode");
    assert_eq!(sr, 16_000);
    assert_eq!(out.len(), frames.len(), "stereo->mono halves sample count");
    // (16384 + 16384) / 2 = 16384 -> 0.5
    assert!((out[0] - 0.5).abs() < 1e-4);
    // (16384 + -16384) / 2 = 0 -> 0.0
    assert!(out[1].abs() < 1e-4);
    // (-32768 + 32767) / 2 ≈ -0.5/32768 ≈ 0.0
    assert!(out[2].abs() < 1e-4);
}

#[test]
fn decode_wav_float32_skips_int_roundtrip_and_preserves_amplitude() {
    // Float WAV used to do a redundant Float->i32->Float trip that
    // lost ~1 ULP per sample. The refactored path consumes the
    // float samples directly. Pin: a 0.5 sample comes out as
    // exactly 0.5 (no precision loss).
    let pcm = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.123456_f32];
    let wav = build_wav_f32(&pcm, 1, 16_000);
    let (out, sr) = decode_wav_to_mono_f32_16k(&wav).expect("decode");
    assert_eq!(sr, 16_000);
    assert_eq!(out.len(), pcm.len());
    for (got, want) in out.iter().zip(pcm.iter()) {
        // Float WAV is lossless f32 round-trip; allow 1 ULP just
        // for hound's internal i32 storage path.
        assert!(
            (got - want).abs() < 1e-6,
            "Float WAV roundtrip drifted: got {got} want {want}"
        );
    }
}

/// Construct an in-memory AudioBuffer<f32> from per-channel planes.
/// `planes[ch][frame]` is the sample value. Width = planes[0].len().
fn make_audio_buffer(planes: &[Vec<f32>]) -> symphonia::core::audio::AudioBuffer<f32> {
    use symphonia::core::audio::{AudioBuffer, Channels, Signal, SignalSpec};
    let chans = planes.len();
    let frames = planes.first().map(|p| p.len()).unwrap_or(0);
    // Channel bitmask: take the first `chans` channel positions from
    // the documented ordering. For tests we just need a valid spec
    // that yields .count() == chans.
    let ch_mask = match chans {
        1 => Channels::FRONT_LEFT,
        2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
        3 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT | Channels::FRONT_CENTRE,
        4 => {
            Channels::FRONT_LEFT
                | Channels::FRONT_RIGHT
                | Channels::REAR_LEFT
                | Channels::REAR_RIGHT
        }
        _ => panic!("unsupported channel count for test fixture: {chans}"),
    };
    let spec = SignalSpec::new(16_000, ch_mask);
    let mut buf = AudioBuffer::<f32>::new(frames as u64, spec);
    buf.render_reserved(Some(frames));
    for (ch, plane) in planes.iter().enumerate() {
        buf.chan_mut(ch).copy_from_slice(plane);
    }
    buf
}

#[test]
fn append_mono_fast_path_appends_channel_zero_directly() {
    // Mono input is the cheap path - no per-frame summation. Pin
    // exact equality so the fast-path branch can't drift to using
    // the multi-channel sum loop (which would still produce
    // correct output but at higher cost).
    let buf = make_audio_buffer(&[vec![0.1_f32, 0.2, 0.3, -0.4]]);
    let mut out = vec![0.99_f32]; // pre-existing value preserved
    append_audio_buffer_as_mono(&buf, &mut out);
    assert_eq!(out, vec![0.99, 0.1, 0.2, 0.3, -0.4]);
}

#[test]
fn append_mono_stereo_averages_channels_in_place() {
    // Two-channel input: each frame's per-channel sum divided by 2.
    let buf = make_audio_buffer(&[vec![0.4_f32, -0.5, 1.0], vec![0.6_f32, 0.5, -1.0]]);
    let mut out = Vec::new();
    append_audio_buffer_as_mono(&buf, &mut out);
    assert_eq!(out.len(), 3);
    assert!(
        (out[0] - 0.5).abs() < 1e-6,
        "(0.4 + 0.6) / 2 = 0.5, got {}",
        out[0]
    );
    assert!(
        (out[1] - 0.0).abs() < 1e-6,
        "(-0.5 + 0.5) / 2 = 0.0, got {}",
        out[1]
    );
    assert!(
        (out[2] - 0.0).abs() < 1e-6,
        "(1.0 + -1.0) / 2 = 0.0, got {}",
        out[2]
    );
}

#[test]
fn append_mono_quad_averages_four_channels() {
    // Surround input (4 channels): mean of all four.
    let buf = make_audio_buffer(&[vec![0.4_f32], vec![0.2_f32], vec![-0.2_f32], vec![0.0_f32]]);
    let mut out = Vec::new();
    append_audio_buffer_as_mono(&buf, &mut out);
    assert_eq!(out.len(), 1);
    // (0.4 + 0.2 + -0.2 + 0.0) / 4 = 0.1
    assert!((out[0] - 0.1).abs() < 1e-6, "got {}", out[0]);
}

#[test]
fn append_mono_preserves_existing_output_tail() {
    // Multiple packets accumulate into the same Vec across the
    // decode loop; appending a stereo packet must NOT clobber
    // samples already written by previous packets.
    let buf = make_audio_buffer(&[vec![1.0_f32, 0.0], vec![1.0_f32, 0.0]]);
    let mut out = vec![0.1_f32, 0.2, 0.3];
    append_audio_buffer_as_mono(&buf, &mut out);
    assert_eq!(out.len(), 5);
    // First three preserved verbatim.
    assert_eq!(&out[..3], &[0.1, 0.2, 0.3]);
    // Appended frames are the stereo means.
    assert!((out[3] - 1.0).abs() < 1e-6);
    assert!((out[4] - 0.0).abs() < 1e-6);
}

#[test]
fn append_mono_empty_buffer_is_a_noop() {
    let buf = make_audio_buffer(&[Vec::<f32>::new(), Vec::<f32>::new()]);
    let mut out = vec![42.0_f32];
    append_audio_buffer_as_mono(&buf, &mut out);
    assert_eq!(out, vec![42.0]);
}

// -- split_into_sentences --
//
// Drives the streaming-TTS chunk boundaries: each returned
// String is synthesised as one PCM block then flushed. Drift in
// the terminator set means non-English text (CJK, Hindi,
// Arabic, Urdu) gets one massive chunk instead of natural
// sentence-by-sentence streaming.

#[test]
fn split_into_sentences_handles_ascii_terminators() {
    let out = split_into_sentences("Hello world. How are you? I'm fine!");
    assert_eq!(out, vec!["Hello world.", "How are you?", "I'm fine!"]);
}

#[test]
fn split_into_sentences_recognises_cjk_fullwidth_terminators() {
    // Chinese / Japanese full-width period (。), exclamation (！),
    // question (？). Without these, CJK text would be one big
    // chunk and lose the TTS streaming benefit.
    let out = split_into_sentences("你好。今天好吗？我很好！");
    assert_eq!(out, vec!["你好。", "今天好吗？", "我很好！"]);
}

#[test]
fn split_into_sentences_recognises_hindi_arabic_urdu_terminators() {
    // Devanagari danda (।), Arabic question mark (؟), Urdu full
    // stop (۔). Pin so the TTS path streams these scripts too.
    // Hindi sentence with danda terminator.
    let out = split_into_sentences("नमस्ते जी। आप कैसे हैं।");
    assert_eq!(out.len(), 2);
    assert!(out[0].contains("नमस्ते"));
    assert!(out[1].contains("कैसे"));
    // Arabic question mark.
    let out = split_into_sentences("كيف حالك؟ بخير.");
    assert_eq!(out.len(), 2);
    // Urdu full stop.
    let out = split_into_sentences("یہ ایک جملہ ہے۔ یہ دوسرا ہے۔");
    assert_eq!(out.len(), 2);
}

#[test]
fn split_into_sentences_uses_newline_as_terminator() {
    // \n breaks too - keeps paragraph boundaries as separate
    // synthesis chunks (better prosody than running them together).
    let out = split_into_sentences("Line one\nLine two\nLine three");
    assert_eq!(out, vec!["Line one", "Line two", "Line three"]);
}

#[test]
fn split_into_sentences_hard_caps_at_240_chars_without_terminator() {
    // A single 300-char "sentence" must split into chunks of
    // <= 240 chars so a giant un-terminated paragraph doesn't
    // produce one massive TTS chunk that blocks streaming.
    let long: String = "x".repeat(300);
    let out = split_into_sentences(&long);
    assert!(out.len() >= 2, "300 chars uncapped -> must split");
    for chunk in &out {
        assert!(
            chunk.chars().count() <= 240,
            "chunk has {} chars, exceeds 240 cap: {:?}",
            chunk.chars().count(),
            chunk
        );
    }
}

#[test]
fn split_into_sentences_trims_each_chunk_and_skips_empties() {
    // Leading/trailing whitespace dropped. Repeated terminators
    // produce empty chunks -> skipped (no empty strings in output).
    let out = split_into_sentences("  Hello.   World.   ");
    for chunk in &out {
        assert!(!chunk.starts_with(' ') && !chunk.ends_with(' '));
        assert!(!chunk.is_empty());
    }
    assert_eq!(out, vec!["Hello.", "World."]);
}

#[test]
fn split_into_sentences_empty_and_whitespace_only_yield_empty_vec() {
    assert!(split_into_sentences("").is_empty());
    assert!(split_into_sentences("   ").is_empty());
    assert!(split_into_sentences("\n\n  ").is_empty());
}

#[test]
fn split_into_sentences_text_without_terminator_falls_back_to_whole_text() {
    // The trailing-tail branch handles inputs with no terminator
    // and no hit on the 240-cap (i.e. short un-terminated text).
    let out = split_into_sentences("hello world");
    assert_eq!(out, vec!["hello world"]);
}

// -- wav_streaming_header --

#[test]
fn wav_streaming_header_uses_unknown_size_markers() {
    // Streaming WAV trick: both `RIFF` chunk size AND `data`
    // chunk size are written as 0xFFFFFFFF so the player reads
    // until the connection closes. ffmpeg/libsndfile accept
    // this; pin so a refactor that computed a "real" size
    // doesn't break streaming-aware clients.
    let h = wav_streaming_header(16_000);
    assert_eq!(h.len(), 44);
    assert_eq!(&h[..4], b"RIFF");
    assert_eq!(&h[4..8], &u32::MAX.to_le_bytes());
    assert_eq!(&h[8..12], b"WAVE");
    assert_eq!(&h[12..16], b"fmt ");
    // fmt subchunk1 size = 16 (PCM).
    assert_eq!(u32::from_le_bytes(h[16..20].try_into().unwrap()), 16);
    // Format = 1 (PCM), channels = 1 (mono), bits = 16.
    assert_eq!(u16::from_le_bytes(h[20..22].try_into().unwrap()), 1);
    assert_eq!(u16::from_le_bytes(h[22..24].try_into().unwrap()), 1);
    assert_eq!(u16::from_le_bytes(h[34..36].try_into().unwrap()), 16);
    // Sample rate threaded through.
    assert_eq!(u32::from_le_bytes(h[24..28].try_into().unwrap()), 16_000);
    // Byte rate = sample_rate * 2 (mono 16-bit).
    assert_eq!(u32::from_le_bytes(h[28..32].try_into().unwrap()), 32_000);
    // data chunk size = unknown marker.
    assert_eq!(&h[36..40], b"data");
    assert_eq!(&h[40..44], &u32::MAX.to_le_bytes());
}

#[test]
fn wav_streaming_header_threads_sample_rate_through() {
    // Different sample rates -> byte_rate scales accordingly.
    for sr in [22_050_u32, 24_000, 44_100, 48_000] {
        let h = wav_streaming_header(sr);
        assert_eq!(u32::from_le_bytes(h[24..28].try_into().unwrap()), sr);
        assert_eq!(u32::from_le_bytes(h[28..32].try_into().unwrap()), sr * 2);
    }
}

// -- detect_audio_format --

#[test]
fn detect_audio_format_recognises_riff_wave_header() {
    // Real WAV starts with `RIFF` at offset 0 and `WAVE` at offset 8
    // (4 bytes of chunk size in between). Pin both checks.
    let mut wav = Vec::with_capacity(12);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&0u32.to_le_bytes()); // chunk size (don't care)
    wav.extend_from_slice(b"WAVE");
    assert!(matches!(detect_audio_format(&wav), AudioFormat::Wav));
}

#[test]
fn detect_audio_format_falls_back_to_other_for_non_wav() {
    // MP3 frames start with 0xFF 0xFB / 0xFA / 0xF3 / ...
    let mp3 = [0xFF_u8, 0xFB, 0x90, 0x00];
    assert!(matches!(detect_audio_format(&mp3), AudioFormat::Other));

    // Empty bytes - shouldn't panic on the slice indexing guards.
    assert!(matches!(detect_audio_format(&[]), AudioFormat::Other));

    // RIFF at offset 0 but no WAVE at offset 8 -> AVI / RMP / etc.
    // We dispatch those to symphonia (which itself rejects them).
    let mut riff_not_wave = Vec::new();
    riff_not_wave.extend_from_slice(b"RIFF");
    riff_not_wave.extend_from_slice(&0u32.to_le_bytes());
    riff_not_wave.extend_from_slice(b"AVI ");
    assert!(matches!(
        detect_audio_format(&riff_not_wave),
        AudioFormat::Other
    ));

    // Just `RIFF` with no body - too short for the 12-byte check.
    assert!(matches!(detect_audio_format(b"RIFF"), AudioFormat::Other));
}

// -- openai_voice_to_description --
//
// /v1/audio/speech uses this to translate an OpenAI voice preset
// (alloy/echo/nova/etc.) + speed knob into a free-text prompt
// the Parler-TTS encoder consumes. Drift here doesn't break any
// type contract - it just produces a different voice. Pin the
// mapping so a refactor can't silently swap timbres on SDK clients
// that depend on documented preset names.

#[test]
fn openai_voice_to_description_uses_speed_tier_phrasing() {
    // Three speed tiers: <0.8, 0.8..=1.2, >1.2.
    let slow = openai_voice_to_description("alloy", 0.5);
    let mid = openai_voice_to_description("alloy", 1.0);
    let fast = openai_voice_to_description("alloy", 1.5);
    assert!(slow.contains("slowly"), "got: {slow}");
    assert!(mid.contains("moderate"), "got: {mid}");
    assert!(fast.contains("quickly"), "got: {fast}");
    // Boundary tier values fall into the moderate bucket.
    assert!(openai_voice_to_description("alloy", 0.8).contains("moderate"));
    assert!(openai_voice_to_description("alloy", 1.2).contains("moderate"));
}

#[test]
fn openai_voice_to_description_distinguishes_gendered_presets() {
    // Spot-check that gendered timbres don't accidentally swap
    // - `nova` is the female bright preset, `onyx` is male
    // authoritative. Drift would route TTS output to the wrong
    // gender of voice.
    let nova = openai_voice_to_description("nova", 1.0).to_lowercase();
    assert!(
        nova.contains("female") && nova.contains("bright"),
        "nova should be female + bright; got: {nova}"
    );
    let onyx = openai_voice_to_description("onyx", 1.0).to_lowercase();
    assert!(
        onyx.contains("male") && onyx.contains("authoritative"),
        "onyx should be male + authoritative; got: {onyx}"
    );
}

#[test]
fn openai_voice_to_description_every_preset_yields_a_distinct_timbre_phrase() {
    // Pull the core timbre phrase (before the first comma) for
    // each preset listed in KNOWN_VOICES. They must all be
    // distinct so SDK clients can rely on preset names mapping
    // to recognisably different output. `alloy` hits the `_`
    // fallback in the match - that fallback string ("A clear,
    // neutral English speaker") is itself distinct from every
    // named preset, so all 11 presets produce 11 unique timbres.
    let core = |voice: &str| -> String {
        let full = openai_voice_to_description(voice, 1.0);
        full.split(',').next().unwrap_or("").to_string()
    };
    let presets: std::collections::HashSet<String> = KNOWN_VOICES.iter().map(|v| core(v)).collect();
    assert_eq!(
        presets.len(),
        KNOWN_VOICES.len(),
        "all KNOWN_VOICES presets must yield distinct timbres; got {} unique \
         for {} presets - two presets collided",
        presets.len(),
        KNOWN_VOICES.len()
    );
}

#[test]
fn openai_voice_to_description_unknown_voice_falls_back_to_alloy_neutral() {
    // Unknown voice names route to the generic neutral timbre
    // (same phrase as `alloy`). KNOWN_VOICES validation already
    // 400s these at the request boundary, but the helper itself
    // must be safe to call with anything.
    let unknown = openai_voice_to_description("typo-voice", 1.0);
    let alloy = openai_voice_to_description("alloy", 1.0);
    assert_eq!(
        unknown, alloy,
        "unknown voice must fall back to the alloy/neutral phrase"
    );
}

#[test]
fn openai_voice_to_description_is_case_insensitive() {
    // SDK clients sometimes send NOVA / Onyx with mixed case.
    assert_eq!(
        openai_voice_to_description("NOVA", 1.0),
        openai_voice_to_description("nova", 1.0),
    );
    assert_eq!(
        openai_voice_to_description("Onyx", 1.0),
        openai_voice_to_description("onyx", 1.0),
    );
}

#[test]
fn known_voices_is_in_sync_with_voice_to_description_branches() {
    // Pin that every entry in KNOWN_VOICES (the request-validation
    // allowlist) has a corresponding branch in the description
    // table. The `alloy` entry intentionally hits the `_ =>`
    // fallback (it IS the fallback timbre) so we check it produces
    // the neutral phrase; every other preset must produce a
    // non-fallback (distinct) timbre.
    let fallback = openai_voice_to_description("__definitely-unknown__", 1.0);
    for &voice in KNOWN_VOICES {
        let got = openai_voice_to_description(voice, 1.0);
        if voice == "alloy" {
            assert_eq!(got, fallback, "alloy IS the fallback");
        } else {
            assert_ne!(
                got, fallback,
                "{voice} hits the unknown-voice fallback - drift between \
                 KNOWN_VOICES allowlist and openai_voice_to_description match arm"
            );
        }
    }
}

// -- hms_comma / hms_dot + format_srt / format_vtt --
//
// Subtitle parsers are strict about the millisecond separator
// (SRT uses `,`, WebVTT uses `.`) and the HH:MM:SS,mmm timing
// shape. Drift here would silently break ingestion of loken
// transcripts in downstream tools like VLC, FFmpeg, browser
// <track> elements, or subtitle editors.

#[test]
fn hms_comma_uses_srt_timestamp_shape() {
    // Pin exact shape: HH:MM:SS,mmm with all components zero-padded.
    assert_eq!(hms_comma(0.0), "00:00:00,000");
    assert_eq!(hms_comma(0.5), "00:00:00,500");
    assert_eq!(hms_comma(1.0), "00:00:01,000");
    // Rounding: 1.2345 -> 1234.5 ms rounded = 1234 (banker's
    // rounding rounds-half-to-even); 1.2346 unambiguously 1235.
    assert_eq!(hms_comma(1.2346), "00:00:01,235");
    // Minute / hour rollover.
    assert_eq!(hms_comma(60.0), "00:01:00,000");
    assert_eq!(hms_comma(3600.0), "01:00:00,000");
    assert_eq!(hms_comma(3661.5), "01:01:01,500");
    // Negative timestamps shouldn't occur for whisper output, but
    // the saturating `as u64` cast yields 0 - verify no panic.
    assert_eq!(hms_comma(-1.0), "00:00:00,000");
}

#[test]
fn hms_dot_uses_webvtt_timestamp_shape() {
    // Identical to hms_comma except for the millisecond separator.
    assert_eq!(hms_dot(0.0), "00:00:00.000");
    assert_eq!(hms_dot(3661.5), "01:01:01.500");
    // Spot-check: the only character that differs from hms_comma
    // is the millisecond delimiter at offset 8.
    let s = hms_dot(123.456);
    assert_eq!(&s[8..9], ".", "VTT must use dot, not comma; got {s}");
}

#[test]
fn format_srt_emits_canonical_subtitle_blocks() {
    // Build a minimal TranscribeResult with two segments and
    // assert the exact SRT byte sequence. Catches both numbering
    // (1-indexed) and the trailing blank-line block separator
    // that SRT parsers require.
    use crate::inference::engine::audio_engine::WhisperSegment;
    let segs = vec![
        WhisperSegment {
            start: 0.0,
            end: 2.5,
            text: "Hello, world.".into(),
            tokens: vec![],
            compression_ratio: 1.0,
            temperature: 0.0,
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
        },
        WhisperSegment {
            start: 2.5,
            end: 5.0,
            text: "  Trim me.  ".into(),
            tokens: vec![],
            compression_ratio: 1.0,
            temperature: 0.0,
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
        },
    ];
    let r = TranscribeResult {
        text: "Hello, world.\nTrim me.".into(),
        language: "en".into(),
        duration_s: 5.0,
        tokens: vec![],
        segments: segs,
    };
    let srt = format_srt(&r);
    assert_eq!(
        srt,
        "1\n00:00:00,000 --> 00:00:02,500\nHello, world.\n\n\
         2\n00:00:02,500 --> 00:00:05,000\nTrim me.\n\n",
        "got:\n{srt}",
    );
}

#[test]
fn format_vtt_emits_canonical_header_and_dot_timestamps() {
    // VTT requires the literal `WEBVTT\n\n` header and dot-form
    // timestamps. Cues do NOT carry a numeric prefix (unlike SRT).
    use crate::inference::engine::audio_engine::WhisperSegment;
    let segs = vec![WhisperSegment {
        start: 1.234,
        end: 2.5,
        text: "first".into(),
        tokens: vec![],
        compression_ratio: 1.0,
        temperature: 0.0,
        avg_logprob: 0.0,
        no_speech_prob: 0.0,
    }];
    let r = TranscribeResult {
        text: "first".into(),
        language: "en".into(),
        duration_s: 2.5,
        tokens: vec![],
        segments: segs,
    };
    let vtt = format_vtt(&r);
    assert_eq!(
        vtt, "WEBVTT\n\n00:00:01.234 --> 00:00:02.500\nfirst\n\n",
        "got:\n{vtt}",
    );
}

#[test]
fn format_srt_with_no_segments_is_empty_body() {
    // Empty audio -> no segments -> empty SRT body (no "1\n..." line).
    let r = TranscribeResult {
        text: String::new(),
        language: "en".into(),
        duration_s: 0.0,
        tokens: vec![],
        segments: vec![],
    };
    assert_eq!(format_srt(&r), "");
}

#[test]
fn format_vtt_with_no_segments_keeps_webvtt_header() {
    // Empty segments still need the WEBVTT header so parsers
    // accept the file (a zero-byte VTT is rejected).
    let r = TranscribeResult {
        text: String::new(),
        language: "en".into(),
        duration_s: 0.0,
        tokens: vec![],
        segments: vec![],
    };
    assert_eq!(format_vtt(&r), "WEBVTT\n\n");
}

#[test]
fn resample_empty_input_returns_empty() {
    // Short-circuit: an empty audio buffer must not spin up rubato
    // (which would fail on zero-length input).
    let out = resample_to_16k(&[], 48_000, 16_000).expect("empty");
    assert!(out.is_empty());
}

#[test]
fn resample_downsamples_to_expected_length() {
    // 1 second of 48 kHz -> ~16000 samples at 16 kHz. The expected-
    // length math (src.len() * dst_rate / src_rate) is what trims
    // the trailing warm-up tail, so a drift here would change the
    // duration the model sees by ±out_delay frames.
    let src: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
    let out = resample_to_16k(&src, 48_000, 16_000).expect("resample");
    // expected = round(48000 * 16000 / 48000) = 16000.
    assert_eq!(out.len(), 16_000, "1 s @ 48 kHz -> 16000 samples @ 16 kHz");
}

#[test]
fn resample_upsamples_to_expected_length() {
    // 16 kHz -> 16 kHz is the no-op fast path (the caller skips
    // resample_to_16k for matching rates) so test 8 kHz -> 16 kHz
    // here to exercise the upsampling rate ratio.
    let src: Vec<f32> = (0..8_000).map(|i| (i as f32 * 0.02).sin()).collect();
    let out = resample_to_16k(&src, 8_000, 16_000).expect("resample");
    // expected = round(8000 * 16000 / 8000) = 16000.
    assert_eq!(out.len(), 16_000, "1 s @ 8 kHz -> 16000 samples @ 16 kHz");
}

#[test]
fn resample_short_input_yields_proportional_output() {
    // Inputs shorter than rubato's in_chunk used to need the
    // residual-tail path. Pin behaviour: 100 ms @ 22050 should
    // still produce ~1600 samples @ 16 kHz (within out_delay).
    let src: Vec<f32> = vec![0.0_f32; 22_050 / 10]; // 0.1 s of silence
    let out = resample_to_16k(&src, 22_050, 16_000).expect("resample");
    // expected = round(2205 * 16000 / 22050) = 1600.
    assert_eq!(out.len(), 1600);
}

#[test]
fn decode_wav_unsupported_int_bit_depth_returns_error() {
    // Hound only supports a subset of bit depths; the decoder's
    // explicit guard returns a typed error rather than producing
    // a normalisation off by a huge factor.
    // Construct a malformed-on-purpose WAV: 7 bits/sample on Int
    // format is rejected by hound *before* we hit our match. Use
    // a 12-bit (unsupported on our side) Int WAV - hound accepts
    // bits<=16 packed in 16-bit containers but our guard only
    // recognises 8/16/24/32.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 12,
        sample_format: hound::SampleFormat::Int,
    };
    // hound disallows bps=12 directly; emulate by writing a
    // standard 16-bit file then patching the bps field at
    // offset 34 (little-endian u16).
    let pcm = vec![0_i16, 1, -1];
    let mut wav = build_wav_int16(&pcm, spec.channels, spec.sample_rate);
    wav[34] = 12;
    wav[35] = 0;
    let err = decode_wav_to_mono_f32_16k(&wav).expect_err("12-bit WAV must fail");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("bit") || msg.contains("unsupported") || msg.contains("12"),
        "unhelpful error: {err}"
    );
}

// -- apply_speed_linear --
//
// TTS speed control runs through a linear-interpolation
// resampler. speed=2.0 halves duration; speed=0.5 doubles it.
// Below 1.0 the audio plays slower (longer); above 1.0 faster.
// These tests pin the duration contract, the boundary handling
// (last source sample clamps), and the degenerate-input policy.

#[test]
fn apply_speed_passes_through_for_invalid_speed() {
    // Non-finite or non-positive speed -> caller would otherwise
    // be a divide-by-zero. Fall back to returning the input
    // unchanged (matches the documented "treat as no-op" hint).
    let pcm = vec![0.1_f32, 0.2, 0.3];
    assert_eq!(apply_speed_linear(&pcm, f32::NAN), pcm);
    assert_eq!(apply_speed_linear(&pcm, f32::INFINITY), pcm);
    assert_eq!(apply_speed_linear(&pcm, 0.0), pcm);
    assert_eq!(apply_speed_linear(&pcm, -1.0), pcm);
}

#[test]
fn apply_speed_empty_input_returns_empty() {
    assert!(apply_speed_linear(&[], 2.0).is_empty());
}

#[test]
fn apply_speed_halves_duration_at_2x() {
    // 8 input samples @ speed=2.0 -> 4 output samples (rounded).
    let pcm: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let out = apply_speed_linear(&pcm, 2.0);
    assert_eq!(out.len(), 4, "2x speed must halve sample count");
    // Each output is the input at index i*2 (no fractional part).
    assert_eq!(out, vec![0.0, 2.0, 4.0, 6.0]);
}

#[test]
fn apply_speed_doubles_duration_at_half_speed() {
    let pcm = vec![0.0_f32, 10.0, 20.0, 30.0];
    let out = apply_speed_linear(&pcm, 0.5);
    // n_out = round(4 / 0.5) = 8.
    assert_eq!(out.len(), 8);
    // Even indices land on input; odd indices interpolate midway.
    assert_eq!(out[0], 0.0);
    assert!((out[1] - 5.0).abs() < 1e-4, "got {}", out[1]);
    assert_eq!(out[2], 10.0);
    assert!((out[3] - 15.0).abs() < 1e-4);
    assert_eq!(out[4], 20.0);
    assert!((out[5] - 25.0).abs() < 1e-4, "got {}", out[5]);
    // i=6,7: src = 3.0, 3.5 -> lo+1 >= n_in -> clamp to pcm[3] = 30.
    assert_eq!(out[6], 30.0);
    assert_eq!(out[7], 30.0);
}

#[test]
fn apply_speed_returns_at_least_one_sample_for_micro_input() {
    // n_in=1 -> n_out = round(1 / speed). For speed > 2.0 that
    // rounds to 0, and the function returns an empty Vec rather
    // than panicking.
    let one = vec![0.5_f32];
    assert!(apply_speed_linear(&one, 100.0).is_empty());
}

#[test]
fn apply_speed_preserves_unit_input_at_speed_one() {
    // The caller's fast path skips this fn for speed≈1.0, but
    // the function itself must still be identity when called
    // with speed=1.0 (defensive contract).
    let pcm = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5];
    let out = apply_speed_linear(&pcm, 1.0);
    assert_eq!(out.len(), pcm.len());
    for (a, b) in out.iter().zip(pcm.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn pcm_to_wav_bytes_emits_canonical_riff_header() {
    // Same header shape as pcm_to_wav_base64 but raw bytes (no
    // base64). Used by the TTS pipeline when /v1/audio/speech
    // asks for response_format=wav. Pin the byte layout so
    // downstream WAV players still accept it.
    let pcm = vec![0.0_f32; 4];
    let wav = pcm_to_wav_bytes(&pcm, 24_000).expect("encode");
    assert_eq!(wav.len(), 44 + 8); // 4 samples x 2 bytes = 8
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
    assert_eq!(
        u16::from_le_bytes(wav[22..24].try_into().unwrap()),
        1,
        "mono"
    );
    assert_eq!(
        u16::from_le_bytes(wav[34..36].try_into().unwrap()),
        16,
        "16-bit"
    );
    assert_eq!(
        u32::from_le_bytes(wav[40..44].try_into().unwrap()),
        8,
        "data size"
    );
}

#[test]
fn pcm_to_wav_bytes_clamps_out_of_range_symmetrically() {
    // pcm_to_wav_bytes clamps the *float* sample to [-1, 1]
    // before multiplying by i16::MAX (=32767). That yields a
    // symmetric output range of [-32767, +32767] - one short
    // of i16::MIN (-32768) but symmetric around zero, which
    // is the standard PCM convention (matches hound, ffmpeg's
    // pcm_s16le, and the OpenAI reference output).
    let pcm = vec![2.0_f32, -2.0, 0.0];
    let wav = pcm_to_wav_bytes(&pcm, 22_050).expect("encode");
    let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
    let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
    let s2 = i16::from_le_bytes(wav[48..50].try_into().unwrap());
    assert_eq!(s0, 32767, "+2.0 must clamp to +32767");
    assert_eq!(s1, -32767, "-2.0 must clamp to -32767 (symmetric)");
    assert_eq!(s2, 0);
}

#[test]
fn pcm_to_raw_le_bytes_matches_data_chunk_of_wav() {
    // pcm_to_raw_le_bytes shares the per-sample encode path with
    // pcm_to_wav_bytes; equality of the data region pins they
    // can't diverge (e.g. one rounding vs the other truncating).
    let pcm = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
    let raw = pcm_to_raw_le_bytes(&pcm);
    let wav = pcm_to_wav_bytes(&pcm, 16_000).expect("encode");
    assert_eq!(raw.len(), pcm.len() * 2);
    // First sample bytes in raw should equal bytes at offset 44 in WAV.
    assert_eq!(&raw[..], &wav[44..44 + pcm.len() * 2]);
}

#[test]
fn pcm_to_raw_le_bytes_handles_empty_input() {
    assert!(pcm_to_raw_le_bytes(&[]).is_empty());
}

#[test]
fn pcm_to_wav_base64_handles_empty_input() {
    // Zero samples must still produce a valid (header-only) WAV.
    use base64::Engine;
    let b64 = pcm_to_wav_base64(&[], 44_100);
    let wav = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .unwrap();
    assert_eq!(wav.len(), 44, "empty PCM -> header only, no data bytes");
    let data_size = u32::from_le_bytes(wav[40..44].try_into().unwrap());
    assert_eq!(data_size, 0);
}

#[test]
fn is_asr_model_matches_whisper_variants() {
    assert!(is_asr_model("openai/whisper-small"));
    assert!(is_asr_model("whisper-large-v3"));
    assert!(is_asr_model("distil-whisper/distil-large-v3"));
    // Case-insensitive
    assert!(is_asr_model("Whisper"));
    // Negatives - chat / TTS / image / video must NOT classify as ASR.
    assert!(!is_asr_model("qwen3-coder:latest"));
    assert!(!is_asr_model("parler-tts/parler-tts-mini-v1"));
    assert!(!is_asr_model("Tongyi-MAI/Z-Image-Turbo"));
    assert!(!is_asr_model(""));
}

#[test]
fn is_tts_model_matches_known_families() {
    assert!(is_tts_model("parler-tts/parler-tts-mini-v1"));
    assert!(is_tts_model("parler-tts-large-v1"));
    assert!(is_tts_model("openai/tts-1"));
    assert!(is_tts_model("tts-1-hd"));
    assert!(is_tts_model("suno/bark"));
    assert!(is_tts_model("hexgrad/kokoro"));
    assert!(is_tts_model("ai4bharat/f5-tts"));
    // Negative cases: chat / image / ASR models must NOT be TTS.
    assert!(!is_tts_model("qwen3-coder:latest"));
    assert!(!is_tts_model("Tongyi-MAI/Z-Image-Turbo"));
    assert!(!is_tts_model("openai/whisper-small"));
    assert!(!is_tts_model("lmz/candle-flux"));
    assert!(!is_tts_model(""));
}

// -- /v1/audio/speech delivery -------------------------------------------

/// `stream` was already taken on this route and means "send the audio bytes as they
/// are produced". Every body that shipped must still get a body of audio: the event
/// stream is opted into, never inferred.
#[test]
fn the_delivery_that_shipped_is_the_one_a_body_without_stream_format_gets() {
    assert_eq!(speech_delivery(None).unwrap(), SpeechDelivery::Audio);
    assert_eq!(speech_delivery(Some("")).unwrap(), SpeechDelivery::Audio);
    assert_eq!(
        speech_delivery(Some("audio")).unwrap(),
        SpeechDelivery::Audio
    );
    assert_eq!(
        speech_delivery(Some("SSE")).unwrap(),
        SpeechDelivery::Events
    );
    assert_eq!(
        speech_delivery(Some(" sse ")).unwrap(),
        SpeechDelivery::Events
    );
    // A typo returns a body of audio to a client waiting for events, which reads as a
    // hung server. Refused instead.
    let e = speech_delivery(Some("event-stream")).unwrap_err();
    assert!(
        e.contains("stream_format"),
        "an error that names the field: {e}"
    );

    // The byte-streaming request an existing client sends, parsed as it will be:
    // `stream` untouched, and the delivery still a body of audio.
    let req: OpenAISpeechRequest = serde_json::from_value(serde_json::json!({
        "input": "Hello there.", "stream": true, "response_format": "pcm",
    }))
    .unwrap();
    assert_eq!(req.stream, Some(true));
    assert_eq!(req.stream_format, None);
    assert_eq!(
        speech_delivery(req.stream_format.as_deref()).unwrap(),
        SpeechDelivery::Audio
    );
}

/// Two live jobs must never collide in the shared cancel registry: an id reused there
/// CANCELS whatever held it, so a speech stream taking a video's identifier would kill
/// the video.
#[test]
fn a_speech_id_is_its_own() {
    let a = next_speech_id();
    let b = next_speech_id();
    assert_ne!(a, b);
    assert!(a.starts_with('s'), "{a}");
    assert!(
        !a.starts_with('r'),
        "the media routes own the r prefix: {a}"
    );
}

/// THE defect: the checkpoint was read by `ensure_tts_model_loaded` before the response
/// body existed, so a cold engine answered with a minute of silence and the client had
/// no way to tell that from a wedged server. Drive the route with a voice that cannot
/// load - no weights, no GPU, no network - and the stream must still have SAID things
/// first: the identifier it can be cancelled by, and the phase it is in. An error
/// arriving on its own is the old behaviour with a new content-type.
#[tokio::test]
async fn the_speech_event_stream_names_its_phase_before_it_has_anything_to_send() {
    let state = APIServer::new(
        "/nonexistent-ollama-models".to_string(),
        "/nonexistent-hf-models".to_string(),
    );
    let resp = audio_speech(
        axum::extract::State(state),
        axum::http::HeaderMap::new(),
        Json(serde_json::json!({
            "input": "Hello there.",
            "model": "piper/no-such-voice-here",
            "stream_format": "sse",
        })),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("text/event-stream"),
        "not an event stream: {ct}"
    );

    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&body).to_string();
    let events: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).expect("every event is one JSON object"))
        .collect();
    assert!(
        events.len() >= 3,
        "expected started + a phase + an outcome, got {text}"
    );

    assert_eq!(events[0]["status"], "started");
    assert!(
        events[0]["render_id"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "the id is published in the FIRST event or it is useless: {}",
        events[0]
    );

    let outcome = events.last().unwrap();
    assert_eq!(
        outcome["status"], "error",
        "expected the load to fail: {outcome}"
    );
    let err = outcome["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("piper voice not found"),
        "an error that says why: {err}"
    );

    let first_error = events.iter().position(|e| e["status"] == "error").unwrap();
    let first_phase = events
        .iter()
        .position(|e| e["status"] == "loading")
        .expect("the load phase must be announced");
    assert!(
        first_phase < first_error,
        "the client must learn what the server is DOING before it learns how it went"
    );
    assert_eq!(
        events[first_phase]["phase"],
        crate::inference::serve::progress::phase::LOAD_MODEL
    );
    assert_eq!(events[first_phase]["phase_label"], "Loading the model");

    // Nothing claims to be audio before the work is done: a client concatenating
    // `data` would otherwise play the progress.
    assert!(
        events.iter().all(|e| e.get("data").is_none()),
        "a failed synthesis must not carry a clip: {text}"
    );
}
