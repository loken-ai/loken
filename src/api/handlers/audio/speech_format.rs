//! The formats `/v1/audio/speech` answers, one table for the parser, the error message,
//! the `Content-Type` and the encoder.

use super::*;

/// How a format leaves the server while the sentences of a request are still being spoken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Streaming {
    /// The container is a header followed by raw samples.
    Raw,
    /// Each sentence is a complete stream of its own and the streams follow one another.
    Chained,
    /// The container needs the whole clip before its first byte.
    Whole,
}

/// One format of the speech route.
pub(super) struct SpeechFormat {
    pub name: &'static str,
    pub content_type: &'static str,
    pub streaming: Streaming,
    /// Interleaved samples, channel count and sample rate to the bytes of one clip.
    encode: fn(&[f32], u16, u32) -> anyhow::Result<Vec<u8>>,
}

/// Every format served. The first is the default.
pub(super) const SPEECH_FORMATS: [SpeechFormat; 6] = [
    SpeechFormat {
        name: "wav",
        content_type: "audio/wav",
        streaming: Streaming::Raw,
        encode: |pcm, channels, rate| Ok(pcm_to_wav_ch(pcm, rate, channels)),
    },
    SpeechFormat {
        name: "pcm",
        content_type: "audio/L16",
        streaming: Streaming::Raw,
        encode: |pcm, _, _| Ok(pcm_to_raw_le_bytes(pcm)),
    },
    SpeechFormat {
        name: "mp3",
        content_type: "audio/mpeg",
        streaming: Streaming::Chained,
        encode: pcm_to_mp3_bytes,
    },
    SpeechFormat {
        name: "flac",
        content_type: "audio/flac",
        streaming: Streaming::Whole,
        encode: |pcm, channels, rate| {
            pcm16_to_flac_bytes(&pcm_to_i16(pcm), channels, rate).map_err(anyhow::Error::msg)
        },
    },
    SpeechFormat {
        name: "opus",
        content_type: "audio/ogg; codecs=opus",
        streaming: Streaming::Chained,
        encode: pcm_to_opus_bytes,
    },
    SpeechFormat {
        name: "aac",
        content_type: "audio/aac",
        streaming: Streaming::Chained,
        encode: pcm_to_aac_bytes,
    },
];

impl SpeechFormat {
    /// The format a request names, WAV when it names none; an unknown name is refused with a
    /// message naming it and listing the formats served.
    pub(super) fn parse(name: Option<&str>) -> Result<&'static SpeechFormat, String> {
        let Some(name) = name else {
            return Ok(&SPEECH_FORMATS[0]);
        };
        let lower = name.to_ascii_lowercase();
        SPEECH_FORMATS
            .iter()
            .find(|f| f.name == lower)
            .ok_or_else(|| {
                format!(
                    "response_format '{lower}' not supported; use {}",
                    Self::list()
                )
            })
    }

    /// The names served, quoted and joined for a message.
    pub(super) fn list() -> String {
        let quoted: Vec<String> = SPEECH_FORMATS
            .iter()
            .map(|f| format!("'{}'", f.name))
            .collect();
        match quoted.split_last() {
            Some((last, head)) if !head.is_empty() => format!("{} or {last}", head.join(", ")),
            _ => quoted.join(""),
        }
    }

    /// The `Content-Type` of a clip with this rate and channel count; raw samples carry both
    /// as parameters.
    pub(super) fn content_type_for(&self, rate: u32, channels: u16) -> String {
        if self.name == "pcm" {
            format!("{}; rate={rate}; channels={channels}", self.content_type)
        } else {
            self.content_type.to_string()
        }
    }

    /// One whole clip in this format.
    pub(super) fn encode(&self, pcm: &[f32], channels: u16, rate: u32) -> anyhow::Result<Vec<u8>> {
        (self.encode)(pcm, channels, rate)
    }

    /// The error a streamed request gets when the container cannot be cut into chunks.
    pub(super) fn stream_refusal(&self) -> Option<String> {
        (self.streaming == Streaming::Whole).then(|| {
            let streamable: Vec<String> = SPEECH_FORMATS
                .iter()
                .filter(|f| f.streaming != Streaming::Whole)
                .map(|f| format!("'{}'", f.name))
                .collect();
            format!(
                "response_format '{}' cannot be streamed: its container is written once the \
                 whole clip is known; use {} with stream, or drop stream",
                self.name,
                streamable.join(", ")
            )
        })
    }
}

/// Samples in [-1, 1] as 16-bit integers, saturating, the way the raw and WAV writers do.
fn pcm_to_i16(pcm: &[f32]) -> Vec<i16> {
    pcm.iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

/// The rate a clip is resampled to when its own is one the codec does not take.
const FALLBACK_RATE: u32 = 48_000;

/// Interleaved samples at `rate`, or at `FALLBACK_RATE` when the codec does not take `rate`.
/// Each channel goes through the resampler the audio inputs use.
fn at_codec_rate(
    pcm: &[f32],
    channels: u16,
    rate: u32,
    takes: impl Fn(u32) -> bool,
) -> anyhow::Result<(std::borrow::Cow<'_, [f32]>, u32)> {
    if takes(rate) {
        return Ok((std::borrow::Cow::Borrowed(pcm), rate));
    }
    let ch = usize::from(channels.max(1));
    let planes: Vec<Vec<f32>> = (0..ch)
        .map(|c| {
            let plane: Vec<f32> = pcm.iter().skip(c).step_by(ch).copied().collect();
            crate::inference::media::audio_io::resample(&plane, rate, FALLBACK_RATE)
        })
        .collect::<anyhow::Result<_>>()?;
    let frames = planes.first().map_or(0, Vec::len);
    let mut out = Vec::with_capacity(frames * ch);
    for i in 0..frames {
        out.extend(planes.iter().map(|p| p[i]));
    }
    Ok((std::borrow::Cow::Owned(out), FALLBACK_RATE))
}

/// Sample rates an Opus stream carries.
const OPUS_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, FALLBACK_RATE];

/// Interleaved samples as an Ogg Opus stream at the encoder's own default bit rate.
/// A rate Opus does not take is resampled to 48 kHz.
/// The stream is gapless: the encoder delay is flushed and the last granule position marks
/// where the audio ends.
pub(super) fn pcm_to_opus_bytes(
    pcm: &[f32],
    channels: u16,
    sample_rate: u32,
) -> anyhow::Result<Vec<u8>> {
    use opus_pure::{Application, OggOpusWriter, OpusEncoder, OpusHead, MAX_PACKET_BYTES};
    let (pcm, rate) = at_codec_rate(pcm, channels, sample_rate, |r| OPUS_RATES.contains(&r))?;
    let ch = usize::from(channels);
    let enc_err = |e: opus_pure::Error| anyhow::anyhow!("{e}");
    let mut encoder = OpusEncoder::new(rate as i32, ch, Application::Audio).map_err(enc_err)?;
    let head = OpusHead::for_encoder(&encoder, rate);
    // Every Opus rate divides 48 kHz, so a sample at `rate` is a whole number of granule ticks.
    let ticks = (FALLBACK_RATE / rate) as usize;
    let total = pcm.len() / ch;
    let frame = (rate / 50) as usize;
    let lookahead = (head.pre_skip as usize).div_ceil(ticks);
    let frames = (total + lookahead).div_ceil(frame).max(1);
    let final_granule = u64::from(head.pre_skip) + (total * ticks) as u64;

    let mut writer = OggOpusWriter::new(Vec::new(), head).map_err(enc_err)?;
    let mut packet = vec![0u8; MAX_PACKET_BYTES];
    let per_frame = frame * ch;
    let mut block = vec![0.0f32; per_frame];
    for i in 0..frames {
        let start = (i * per_frame).min(pcm.len());
        let end = (start + per_frame).min(pcm.len());
        block[..end - start].copy_from_slice(&pcm[start..end]);
        block[end - start..].fill(0.0);
        let n = encoder
            .encode(&block, frame, &mut packet)
            .map_err(enc_err)?;
        if i + 1 == frames {
            let duration = final_granule.saturating_sub(writer.granule() as u64);
            writer
                .write_packet_with_duration(&packet[..n], duration as u32)
                .map_err(enc_err)?;
        } else {
            writer.write_packet(&packet[..n]).map_err(enc_err)?;
        }
    }
    writer.finish().map_err(enc_err)
}

/// Interleaved samples as an AAC-LC stream in ADTS frames at the encoder's own default bit
/// rate per channel. A rate ADTS cannot name is resampled to 48 kHz.
pub(super) fn pcm_to_aac_bytes(
    pcm: &[f32],
    channels: u16,
    sample_rate: u32,
) -> anyhow::Result<Vec<u8>> {
    use oxideav_aac::adts::ADTS_SAMPLE_RATES_HZ;
    use oxideav_aac::codec_encoder::DEFAULT_BITRATE_PER_CHANNEL;
    use oxideav_aac::encoder::{EncoderConfig, StreamEncoder};
    // The last entry of the ADTS table has no scalefactor bands in the encoder.
    let ladder = &ADTS_SAMPLE_RATES_HZ[..ADTS_SAMPLE_RATES_HZ.len() - 1];
    let (pcm, rate) = at_codec_rate(pcm, channels, sample_rate, |r| ladder.contains(&r))?;
    let channel_count =
        u8::try_from(channels).map_err(|_| anyhow::anyhow!("{channels} channels not supported"))?;
    let mut encoder = StreamEncoder::new(EncoderConfig {
        sample_rate: rate,
        channels: channel_count,
        bitrate: DEFAULT_BITRATE_PER_CHANNEL * u32::from(channels),
    })
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    encoder
        .encode_all(&pcm_to_i16(&pcm))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second of three tones at `rate`, interleaved over `channels` channels with the tones
    /// scaled per channel so the channels differ.
    fn tones(rate: u32, channels: u16) -> Vec<f32> {
        let mut out = Vec::new();
        for i in 0..rate {
            let t = i as f32 / rate as f32;
            let base = 0.3 * (t * 440.0 * std::f32::consts::TAU).sin()
                + 0.2 * (t * 1_250.0 * std::f32::consts::TAU).sin()
                + 0.1 * (t * 3_100.0 * std::f32::consts::TAU).sin();
            for c in 0..channels {
                out.push(base / (1.0 + f32::from(c)));
            }
        }
        out
    }

    fn named(name: &str) -> &'static SpeechFormat {
        SpeechFormat::parse(Some(name)).unwrap()
    }

    /// Signal-to-noise ratio in dB of `got` against `want` after the lag, within `0..=max_lag`
    /// samples, that best aligns them, the first and last block left out because a transform
    /// codec has not settled there.
    fn aligned_snr_db(want: &[f32], got: &[f32], max_lag: usize) -> f64 {
        const EDGE: usize = 2_048;
        let window = want.len().min(8_192);
        let lag = (0..=max_lag)
            .max_by(|&a, &b| {
                let dot = |l: usize| -> f64 {
                    (0..window)
                        .filter_map(|i| got.get(i + l).map(|g| f64::from(*g) * f64::from(want[i])))
                        .sum()
                };
                dot(a).total_cmp(&dot(b))
            })
            .unwrap();
        let end = want.len().min(got.len() - lag) - EDGE;
        let (mut signal, mut noise) = (0.0f64, 0.0f64);
        for i in EDGE..end {
            let w = f64::from(want[i]);
            let e = w - f64::from(got[i + lag]);
            signal += w * w;
            noise += e * e;
        }
        10.0 * (signal / noise.max(1e-30)).log10()
    }

    /// The SNR a uniform quantiser of `bits_per_sample` bits reaches on a full-scale ramp,
    /// 6.02 dB per bit.
    fn floor_db(bitrate_bps: u32, rate: u32, channels: u16) -> f64 {
        6.02 * f64::from(bitrate_bps) / (f64::from(rate) * f64::from(channels))
    }

    /// Opus bytes back to interleaved samples at `rate`, trimmed as RFC 7845 says.
    fn decode_opus(bytes: &[u8], rate: i32) -> (Vec<f32>, usize) {
        use opus_pure::{OggOpusReader, Trim, MAX_PACKET_SAMPLES};
        let mut reader = OggOpusReader::new(std::io::Cursor::new(bytes)).unwrap();
        let head = reader.head().clone();
        let channels = usize::from(head.channel_count);
        let mut decoder = head.decoder(rate).unwrap();
        let mut trim = Trim::new(&head, rate, channels).unwrap();
        let mut block = vec![0.0f32; MAX_PACKET_SAMPLES * channels];
        let mut out = Vec::new();
        for packet in reader.packets() {
            let packet = packet.unwrap();
            let n = decoder
                .decode(&packet.data, MAX_PACKET_SAMPLES, &mut block)
                .unwrap();
            out.extend_from_slice(trim.keep(&packet, &block[..n * channels]));
        }
        (out, channels)
    }

    #[test]
    fn parse_defaults_to_wav_and_is_case_blind() {
        assert_eq!(SpeechFormat::parse(None).unwrap().name, "wav");
        assert_eq!(SpeechFormat::parse(Some("OPUS")).unwrap().name, "opus");
    }

    #[test]
    fn an_unknown_format_is_refused_with_every_name_served() {
        let err = SpeechFormat::parse(Some("ogg")).err().unwrap();
        assert!(err.contains("'ogg'"), "{err}");
        for f in &SPEECH_FORMATS {
            assert!(err.contains(&format!("'{}'", f.name)), "{err}");
        }
    }

    #[test]
    fn each_format_has_its_content_type() {
        let want = [
            ("wav", "audio/wav"),
            ("pcm", "audio/L16; rate=24000; channels=1"),
            ("mp3", "audio/mpeg"),
            ("flac", "audio/flac"),
            ("opus", "audio/ogg; codecs=opus"),
            ("aac", "audio/aac"),
        ];
        assert_eq!(want.len(), SPEECH_FORMATS.len());
        for (name, content_type) in want {
            assert_eq!(named(name).content_type_for(24_000, 1), content_type);
        }
    }

    #[test]
    fn only_a_container_written_at_the_end_refuses_to_stream() {
        for f in &SPEECH_FORMATS {
            let refusal = f.stream_refusal();
            assert_eq!(refusal.is_some(), f.name == "flac", "{}", f.name);
        }
        let err = named("flac").stream_refusal().unwrap();
        assert!(err.contains("'flac'") && err.contains("'opus'"), "{err}");
    }

    #[test]
    fn flac_returns_the_samples_exactly() {
        for rate in [24_000, 44_100] {
            let pcm = tones(rate, 1);
            let bytes = named("flac").encode(&pcm, 1, rate).unwrap();
            let (planar, channels, got_rate) = decode_audio_planar(&bytes).unwrap();
            assert_eq!((channels, got_rate), (1, rate));
            assert_eq!(planar.len(), pcm.len());
            for (i, (&got, &want)) in planar.iter().zip(&pcm).enumerate() {
                let want = (want * i16::MAX as f32) as i16;
                assert_eq!(
                    (got * 32_768.0).round() as i32,
                    i32::from(want),
                    "{rate} {i}"
                );
            }
        }
    }

    #[test]
    fn the_transform_codecs_return_the_signal_above_their_floor() {
        let mp3_bps = MP3_BITRATE_KBPS * 1000;
        let aac_bps = oxideav_aac::codec_encoder::DEFAULT_BITRATE_PER_CHANNEL;
        for rate in [24_000u32, 44_100] {
            let pcm = tones(rate, 1);
            for (name, bps) in [("mp3", mp3_bps), ("aac", aac_bps)] {
                let bytes = named(name).encode(&pcm, 1, rate).unwrap();
                let (planar, channels, got_rate) = decode_audio_planar(&bytes).unwrap();
                assert_eq!((channels, got_rate), (1, rate), "{name} {rate}");
                // The codec adds its delay and pads the tail to a whole frame.
                assert!(planar.len() >= pcm.len(), "{name} {rate}: {}", planar.len());
                assert!(planar.len() < pcm.len() + 4 * 2_048, "{name} {rate}");
                let snr = aligned_snr_db(&pcm, &planar, 2 * 2_048);
                let floor = floor_db(bps, rate, 1);
                assert!(
                    snr >= floor,
                    "{name} {rate}: {snr:.1} dB under {floor:.1} dB"
                );
            }
        }
    }

    #[test]
    fn opus_at_an_opus_rate_keeps_rate_length_and_signal() {
        let rate = 24_000;
        let pcm = tones(rate, 1);
        let bytes = named("opus").encode(&pcm, 1, rate).unwrap();
        assert_eq!(&bytes[..4], b"OggS");
        let (got, channels) = decode_opus(&bytes, rate as i32);
        assert_eq!(channels, 1);
        assert_eq!(got.len(), pcm.len());
        let bitrate = opus_pure::OpusEncoder::new(rate as i32, 1, opus_pure::Application::Audio)
            .unwrap()
            .bitrate_bps;
        let snr = aligned_snr_db(&pcm, &got, 0);
        let floor = floor_db(bitrate as u32, rate, 1);
        assert!(snr >= floor, "{snr:.1} dB under {floor:.1} dB");
    }

    #[test]
    fn opus_at_another_rate_is_resampled_to_48_khz() {
        let rate = 44_100;
        let pcm = tones(rate, 1);
        let bytes = named("opus").encode(&pcm, 1, rate).unwrap();
        let (got, _) = decode_opus(&bytes, FALLBACK_RATE as i32);
        let want = crate::inference::media::audio_io::resample(&pcm, rate, FALLBACK_RATE).unwrap();
        assert_eq!(got.len(), want.len());
        let bitrate =
            opus_pure::OpusEncoder::new(FALLBACK_RATE as i32, 1, opus_pure::Application::Audio)
                .unwrap()
                .bitrate_bps;
        let snr = aligned_snr_db(&want, &got, 0);
        let floor = floor_db(bitrate as u32, FALLBACK_RATE, 1);
        assert!(snr >= floor, "{snr:.1} dB under {floor:.1} dB");
    }

    #[test]
    fn opus_and_aac_keep_both_channels() {
        let rate = 24_000;
        let pcm = tones(rate, 2);
        let opus = named("opus").encode(&pcm, 2, rate).unwrap();
        let (got, channels) = decode_opus(&opus, rate as i32);
        assert_eq!((channels, got.len()), (2, pcm.len()));
        let aac = named("aac").encode(&pcm, 2, rate).unwrap();
        let (planar, channels, got_rate) = decode_audio_planar(&aac).unwrap();
        assert_eq!((channels, got_rate), (2, rate));
        // The left channel carries twice the level of the right.
        let frames = planar.len() / 2;
        let peak = |s: &[f32]| s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let ratio = peak(&planar[..frames]) / peak(&planar[frames..]);
        assert!((1.5..2.5).contains(&ratio), "{ratio}");
    }

    #[test]
    fn every_format_encodes_an_empty_clip() {
        for f in &SPEECH_FORMATS {
            f.encode(&[], 1, 24_000)
                .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        }
    }
}
