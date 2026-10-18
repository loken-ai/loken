//! The container a rendered clip is answered in: WAV as rendered, MP3 through the speech
//! route's encoder, or FLAC through the lossless encoder below.

use super::*;

/// The formats `/v1/audio/generations` answers, named by `response_format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClipFormat {
    Wav,
    Mp3,
    Flac,
}

impl ClipFormat {
    /// The format a request names, WAV when it names none; an unknown name is refused with a
    /// message naming it.
    pub(crate) fn parse(name: Option<&str>) -> Result<Self, String> {
        match name.map(str::to_ascii_lowercase).as_deref() {
            None | Some("wav") => Ok(Self::Wav),
            Some("mp3") => Ok(Self::Mp3),
            Some("flac") => Ok(Self::Flac),
            Some(other) => Err(format!(
                "response_format '{other}' not supported; use 'wav', 'mp3' or 'flac'"
            )),
        }
    }

    pub(crate) fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Flac => "audio/flac",
        }
    }
}

/// A rendered WAV clip in `format`: WAV passes through untouched, MP3 and FLAC are encoded
/// from its samples at its rate and channel count. A 16-bit clip reaches FLAC sample for
/// sample; any other depth is quantised to 16 bits the way the WAV writer does.
pub(crate) fn encode_wav_clip(wav: Vec<u8>, format: ClipFormat) -> Result<Vec<u8>, String> {
    if format == ClipFormat::Wav {
        return Ok(wav);
    }
    let read_err = |e: hound::Error| format!("read rendered wav: {e}");
    let mut reader = hound::WavReader::new(std::io::Cursor::new(&wav)).map_err(read_err)?;
    let spec = reader.spec();
    match format {
        ClipFormat::Wav => Ok(wav),
        ClipFormat::Mp3 => {
            let (pcm, _, _) =
                read_wav_interleaved(&wav).map_err(|e| format!("read rendered wav: {e}"))?;
            pcm_to_mp3_bytes(&pcm, spec.channels, spec.sample_rate)
                .map_err(|e| format!("mp3 encode: {e}"))
        }
        ClipFormat::Flac => {
            let ints: Vec<i16> =
                if spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16 {
                    reader
                        .samples::<i16>()
                        .collect::<Result<_, _>>()
                        .map_err(read_err)?
                } else {
                    let (pcm, _, _) = read_wav_interleaved(&wav)
                        .map_err(|e| format!("read rendered wav: {e}"))?;
                    pcm.iter()
                        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                        .collect()
                };
            pcm16_to_flac_bytes(&ints, spec.channels, spec.sample_rate)
        }
    }
}

/// Samples per FLAC frame: the reference encoder's default block size.
const FLAC_BLOCK: usize = 4096;
/// Highest fixed-predictor order FLAC defines.
const FLAC_MAX_FIXED_ORDER: usize = 4;
/// Highest Rice parameter the four-bit residual coding carries; 15 is the escape code.
const FLAC_MAX_RICE: u32 = 14;
/// Channel counts a FLAC stream may carry.
const FLAC_CHANNELS: std::ops::RangeInclusive<u16> = 1..=8;
/// Sample rates the 20-bit STREAMINFO field carries.
const FLAC_MAX_RATE: u32 = (1 << 20) - 1;

/// Interleaved 16-bit samples as a FLAC stream: STREAMINFO, then fixed-size frames whose
/// channels are coded independently, each subframe by the fixed predictor of the order with
/// the smallest residual and one Rice partition. The MD5 field is zero, which the format
/// defines as not computed.
pub(crate) fn pcm16_to_flac_bytes(
    pcm: &[i16],
    channels: u16,
    sample_rate: u32,
) -> Result<Vec<u8>, String> {
    if !FLAC_CHANNELS.contains(&channels) {
        return Err(format!("flac encode: {channels} channels not supported"));
    }
    if sample_rate == 0 || sample_rate > FLAC_MAX_RATE {
        return Err(format!(
            "flac encode: sample rate {sample_rate} not supported"
        ));
    }
    let ch = usize::from(channels);
    let frames = pcm.len() / ch;
    let mut out = Vec::with_capacity(pcm.len());
    out.extend_from_slice(b"fLaC");
    // Metadata block header: last block, type 0 (STREAMINFO), 34 bytes.
    out.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
    let mut info = BitWriter::default();
    info.put(FLAC_BLOCK as u64, 16);
    info.put(FLAC_BLOCK as u64, 16);
    info.put(0, 24); // minimum frame size, unknown
    info.put(0, 24); // maximum frame size, unknown
    info.put(u64::from(sample_rate), 20);
    info.put(u64::from(channels - 1), 3);
    info.put(15, 5); // bits per sample minus one
    info.put(frames as u64, 36);
    out.extend_from_slice(&info.finish());
    out.extend_from_slice(&[0u8; 16]); // MD5, not computed

    let mut channel = vec![0i32; FLAC_BLOCK];
    for (index, start) in (0..frames).step_by(FLAC_BLOCK).enumerate() {
        let len = FLAC_BLOCK.min(frames - start);
        let mut frame = BitWriter::default();
        frame.put(0b1111_1111_1111_1000, 16); // sync, reserved, fixed block size
        frame.put(0b0111, 4); // block size minus one follows on 16 bits
        frame.put(0b0000, 4); // sample rate from STREAMINFO
        frame.put(u64::from(channels - 1), 4); // independent channels
        frame.put(0b100, 3); // 16 bits per sample
        frame.put(0, 1);
        put_utf8_number(&mut frame, index as u64);
        frame.put((len - 1) as u64, 16);
        let crc8 = crc8(frame.bytes());
        frame.put(u64::from(crc8), 8);
        for c in 0..ch {
            for (i, s) in channel[..len].iter_mut().enumerate() {
                *s = i32::from(pcm[(start + i) * ch + c]);
            }
            put_fixed_subframe(&mut frame, &channel[..len]);
        }
        let mut bytes = frame.finish();
        let crc16 = crc16(&bytes);
        bytes.extend_from_slice(&crc16.to_be_bytes());
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

/// One FIXED subframe: warm-up samples verbatim, then the residual of the order that leaves
/// the smallest absolute sum, Rice-coded in a single partition.
fn put_fixed_subframe(w: &mut BitWriter, x: &[i32]) {
    let max_order = FLAC_MAX_FIXED_ORDER.min(x.len().saturating_sub(1));
    let residual = |order: usize, i: usize| -> i32 {
        match order {
            0 => x[i],
            1 => x[i] - x[i - 1],
            2 => x[i] - 2 * x[i - 1] + x[i - 2],
            3 => x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3],
            _ => x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4],
        }
    };
    let cost = |order: usize| -> u64 {
        (order..x.len())
            .map(|i| u64::from(residual(order, i).unsigned_abs()))
            .sum()
    };
    let order = (0..=max_order).min_by_key(|&o| cost(o)).unwrap_or(0);
    // The Rice parameter that codes this residual in the fewest bits.
    let zigzag = |r: i32| ((r << 1) ^ (r >> 31)) as u32;
    let rice = (0..=FLAC_MAX_RICE)
        .min_by_key(|&k| {
            (order..x.len())
                .map(|i| u64::from(zigzag(residual(order, i)) >> k) + 1 + u64::from(k))
                .sum::<u64>()
        })
        .unwrap_or(0);

    w.put(0, 1);
    w.put(0b001000 | order as u64, 6); // FIXED, order in the low bits
    w.put(0, 1); // no wasted bits
    for &s in &x[..order] {
        w.put(u64::from(s as u16), 16);
    }
    w.put(0b00, 2); // Rice coding, four-bit parameters
    w.put(0, 4); // partition order 0
    w.put(u64::from(rice), 4);
    for i in order..x.len() {
        let u = zigzag(residual(order, i));
        w.put_zeros(u >> rice);
        w.put(1, 1);
        w.put(u64::from(u) & ((1u64 << rice) - 1), rice);
    }
}

/// A frame number in the format's UTF-8-like coding: one byte below 0x80, else a lead byte
/// of `t + 1` ones, a zero and the high bits, then `t` continuation bytes of six bits each.
fn put_utf8_number(w: &mut BitWriter, n: u64) {
    if n < 0x80 {
        w.put(n, 8);
        return;
    }
    let mut tail = 1u32;
    while tail < 6 && n >= 1u64 << (5 * tail + 6) {
        tail += 1;
    }
    let lead = ((0xFFu64 << (7 - tail)) & 0xFF) | (n >> (6 * tail));
    w.put(lead, 8);
    for k in (0..tail).rev() {
        w.put(0x80 | ((n >> (6 * k)) & 0x3F), 8);
    }
}

/// CRC-8, polynomial x^8 + x^2 + x + 1, as the frame header carries it.
fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |mut crc, &b| {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
        crc
    })
}

/// CRC-16, polynomial x^16 + x^15 + x^2 + 1, as the frame footer carries it.
fn crc16(data: &[u8]) -> u16 {
    data.iter().fold(0u16, |mut crc, &b| {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
        crc
    })
}

/// Big-endian bit packer: bits fill each byte from its most significant end.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl BitWriter {
    /// The low `n` bits of `v`, most significant first.
    fn put(&mut self, v: u64, n: u32) {
        if n > 32 {
            self.put(v >> 32, n - 32);
            self.put(v & 0xFFFF_FFFF, 32);
            return;
        }
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (v & ((1u64 << n) - 1));
        self.bits += n;
        while self.bits >= 8 {
            self.bits -= 8;
            self.bytes.push((self.acc >> self.bits) as u8);
        }
        self.acc &= (1u64 << self.bits) - 1;
    }

    fn put_zeros(&mut self, mut n: u32) {
        while n > 0 {
            let step = n.min(32);
            self.put(0, step);
            n -= step;
        }
    }

    /// The whole bytes written so far.
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The bytes with the last partial byte padded with zeros.
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            let pad = 8 - self.bits;
            self.put(0, pad);
        }
        self.bytes
    }
}

/// A rendered WAV clip in `format`, base64-encoded for a `b64_json` field.
pub(crate) fn clip_b64(wav: Vec<u8>, format: ClipFormat) -> Result<String, String> {
    use base64::Engine;
    let bytes = encode_wav_clip(wav, format)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stereo 16-bit WAV of `frames` frames: a sine on the left, low-level noise on the right.
    fn stereo_wav(frames: usize, rate: u32) -> (Vec<u8>, Vec<i16>) {
        let mut ints = Vec::with_capacity(frames * 2);
        let mut lcg = 1u32;
        for i in 0..frames {
            let t = i as f32 / rate as f32;
            ints.push(((t * 440.0 * std::f32::consts::TAU).sin() * 12_000.0) as i16);
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ints.push(((lcg >> 16) as i16) >> 4);
        }
        let floats: Vec<f32> = ints
            .iter()
            .map(|&v| f32::from(v) / i16::MAX as f32)
            .collect();
        (pcm_to_wav_ch(&floats, rate, 2), ints)
    }

    #[test]
    fn parse_defaults_to_wav_and_names_an_unknown_format() {
        assert_eq!(ClipFormat::parse(None), Ok(ClipFormat::Wav));
        assert_eq!(ClipFormat::parse(Some("MP3")), Ok(ClipFormat::Mp3));
        assert_eq!(ClipFormat::parse(Some("flac")), Ok(ClipFormat::Flac));
        let err = ClipFormat::parse(Some("opus")).unwrap_err();
        assert!(err.contains("'opus'"), "{err}");
    }

    #[test]
    fn each_format_carries_its_magic() {
        let (wav, _) = stereo_wav(10_000, 48_000);
        let as_wav = encode_wav_clip(wav.clone(), ClipFormat::Wav).unwrap();
        assert_eq!(&as_wav[..4], b"RIFF");
        let mp3 = encode_wav_clip(wav.clone(), ClipFormat::Mp3).unwrap();
        assert!(
            mp3.starts_with(b"ID3") || (mp3[0] == 0xFF && mp3[1] & 0xE0 == 0xE0),
            "mp3 starts {:02x?}",
            &mp3[..4]
        );
        let flac = encode_wav_clip(wav, ClipFormat::Flac).unwrap();
        assert_eq!(&flac[..4], b"fLaC");
    }

    #[test]
    fn flac_decodes_back_to_the_same_samples() {
        // Not a multiple of the block size, so the last frame is a short one.
        let frames = 2 * FLAC_BLOCK + 123;
        let (wav, ints) = stereo_wav(frames, 44_100);
        let wav_ints: Vec<i16> = hound::WavReader::new(std::io::Cursor::new(&wav))
            .unwrap()
            .samples::<i16>()
            .map(Result::unwrap)
            .collect();
        assert_eq!(wav_ints.len(), ints.len());
        let flac = encode_wav_clip(wav.clone(), ClipFormat::Flac).unwrap();
        assert!(flac.len() < wav.len(), "{} >= {}", flac.len(), wav.len());
        let (planar, channels, rate) = decode_audio_planar(&flac).unwrap();
        assert_eq!((channels, rate), (2, 44_100));
        assert_eq!(planar.len(), wav_ints.len());
        for c in 0..2 {
            for i in 0..frames {
                let got = (planar[c * frames + i] * 32_768.0).round() as i32;
                assert_eq!(got, i32::from(wav_ints[i * 2 + c]), "channel {c} frame {i}");
            }
        }
    }
}
