//! The binary frame an activation travels in, and the rules that make a bad one fail loudly.
//!
//! Today a `LayerRequest` goes out through `reqwest.json(..)`, so the tensor - a `Vec<u8>` -
//! is serialised as a JSON array of decimal numbers. A 4096-wide f16 activation is 8 KB of
//! payload and leaves as roughly 30 KB of ASCII digits, to be parsed again at the far end,
//! once per layer boundary per token. On a pipeline that crosses a host on every token that
//! cost is the transport, not the model.
//!
//! A frame is a fixed header followed by the raw bytes, length-prefixed so a reader knows how
//! much to pull off a stream before it has parsed anything. Control traffic stays on HTTP/JSON,
//! where a self-describing format earns its keep; activations do not.
//!
//! The header is deliberately redundant: it carries the shape AND the payload length, so a
//! frame whose two halves disagree can be rejected. A JSON body cannot even express that
//! inconsistency - it has no length to disagree with - so a truncated or mismatched tensor
//! arrives as a shorter array and is silently accepted as a smaller one.

use std::io::{self, Read, Write};

/// Frame magic, so a stream that is not this protocol fails on the first frame rather than
/// on some later field that happened to parse.
const MAGIC: u32 = 0x4C4D_5731; // "LMW1"

/// Bumped when the layout changes. A peer speaking another version is refused, not guessed at.
const VERSION: u16 = 1;

/// Refuse a frame that claims an implausible payload rather than trying to allocate it: a
/// corrupt or hostile length is otherwise an out-of-memory abort on the receiving node.
/// One activation is a few tens of kilobytes and one KV block a few megabytes; a gigabyte is
/// far outside both and still far below what a node can hold.
const MAX_PAYLOAD: u64 = 1 << 30;

/// The element type of a payload. Encoded as one byte - the string "F16" cost three and could
/// hold anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WireDType {
    F32 = 1,
    F16 = 2,
    BF16 = 3,
    /// Already-quantised bytes, carried through untouched.
    Q8 = 4,
}

impl WireDType {
    pub fn size_bytes(self) -> usize {
        match self {
            WireDType::F32 => 4,
            WireDType::F16 | WireDType::BF16 => 2,
            WireDType::Q8 => 1,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => WireDType::F32,
            2 => WireDType::F16,
            3 => WireDType::BF16,
            4 => WireDType::Q8,
            _ => return None,
        })
    }
}

/// What travels ahead of the bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    /// Which in-flight request this belongs to, so one connection can carry many.
    pub request_id: u64,
    /// Which layer boundary produced it.
    pub layer_id: u32,
    pub dtype: WireDType,
    pub shape: Vec<u32>,
}

impl FrameHeader {
    /// How many bytes the shape and dtype say the payload must be.
    ///
    /// Returned as `u64` and computed with checked arithmetic: a shape of `[u32::MAX, u32::MAX]`
    /// overflows `usize` on a 32-bit target, and an overflow here would wrap to a small number
    /// that then agrees with a short payload.
    pub fn expected_payload_len(&self) -> Option<u64> {
        let elems = self
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(u64::from(d)))?;
        elems.checked_mul(self.dtype.size_bytes() as u64)
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Write one frame: header, then the payload verbatim.
///
/// The payload is borrowed and written straight through - it is never copied into an
/// intermediate buffer, which is the whole point of leaving JSON behind.
pub fn write_frame<W: Write>(w: &mut W, h: &FrameHeader, payload: &[u8]) -> io::Result<()> {
    match h.expected_payload_len() {
        Some(n) if n == payload.len() as u64 => {}
        Some(n) => {
            return Err(invalid(format!(
                "frame refuses to encode: shape {:?} of {:?} needs {n} bytes, payload has {}",
                h.shape,
                h.dtype,
                payload.len()
            )))
        }
        None => return Err(invalid("frame refuses to encode: shape overflows")),
    }
    if payload.len() as u64 > MAX_PAYLOAD {
        return Err(invalid("frame refuses to encode: payload above the cap"));
    }

    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&h.request_id.to_le_bytes())?;
    w.write_all(&h.layer_id.to_le_bytes())?;
    w.write_all(&[h.dtype as u8])?;
    let rank = u8::try_from(h.shape.len())
        .map_err(|_| invalid("frame refuses to encode: rank above 255"))?;
    w.write_all(&[rank])?;
    for d in &h.shape {
        w.write_all(&d.to_le_bytes())?;
    }
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
    Ok(())
}

/// Read one frame. Fails rather than returning a partial or implausible one.
pub fn read_frame<R: Read>(r: &mut R) -> io::Result<(FrameHeader, Vec<u8>)> {
    let mut u32b = [0u8; 4];
    let mut u64b = [0u8; 8];
    let mut u16b = [0u8; 2];
    let mut u8b = [0u8; 1];

    r.read_exact(&mut u32b)?;
    if u32::from_le_bytes(u32b) != MAGIC {
        return Err(invalid("not a frame: magic mismatch"));
    }
    r.read_exact(&mut u16b)?;
    let version = u16::from_le_bytes(u16b);
    if version != VERSION {
        return Err(invalid(format!(
            "frame version {version}, this build speaks {VERSION}"
        )));
    }
    r.read_exact(&mut u64b)?;
    let request_id = u64::from_le_bytes(u64b);
    r.read_exact(&mut u32b)?;
    let layer_id = u32::from_le_bytes(u32b);
    r.read_exact(&mut u8b)?;
    let dtype = WireDType::from_u8(u8b[0])
        .ok_or_else(|| invalid(format!("unknown dtype tag {}", u8b[0])))?;
    r.read_exact(&mut u8b)?;
    let rank = u8b[0] as usize;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        r.read_exact(&mut u32b)?;
        shape.push(u32::from_le_bytes(u32b));
    }
    r.read_exact(&mut u64b)?;
    let len = u64::from_le_bytes(u64b);

    // Check the length BEFORE allocating: a corrupt frame claiming a terabyte must not be
    // answered with a terabyte request to the allocator.
    if len > MAX_PAYLOAD {
        return Err(invalid(format!("frame claims {len} bytes, above the cap")));
    }
    let h = FrameHeader {
        request_id,
        layer_id,
        dtype,
        shape,
    };
    match h.expected_payload_len() {
        Some(n) if n == len => {}
        Some(n) => {
            return Err(invalid(format!(
                "frame disagrees with itself: shape needs {n} bytes, header says {len}"
            )))
        }
        None => return Err(invalid("frame shape overflows")),
    }

    let mut payload = vec![0u8; len as usize];
    // `read_exact` is what makes a truncated frame an error instead of a shorter tensor.
    r.read_exact(&mut payload)?;
    Ok((h, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_activation(hidden: usize) -> (FrameHeader, Vec<u8>) {
        // Values chosen to include the ones a lossy or lexical round trip damages: a
        // subnormal, a negative zero, an infinity and a NaN all survive a byte copy and
        // none of them survive a detour through decimal text.
        let mut bytes = Vec::with_capacity(hidden * 2);
        for i in 0..hidden {
            let v = match i % 5 {
                0 => half::f16::from_f32(-0.0),
                1 => half::f16::from_bits(0x0001), // smallest subnormal
                2 => half::f16::INFINITY,
                3 => half::f16::NAN,
                _ => half::f16::from_f32(i as f32 * 0.001),
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let h = FrameHeader {
            request_id: 0xDEAD_BEEF_CAFE_1234,
            layer_id: 27,
            dtype: WireDType::F16,
            shape: vec![1, hidden as u32],
        };
        (h, bytes)
    }

    /// The phase-1 invariant: a `[1, hidden]` f16 activation comes back bit-exact.
    #[test]
    fn a_hidden_activation_round_trips_bit_exact() {
        let (h, payload) = f16_activation(4096);
        let mut buf = Vec::new();
        write_frame(&mut buf, &h, &payload).unwrap();

        let (got_h, got_payload) = read_frame(&mut buf.as_slice()).unwrap();
        assert_eq!(got_h, h, "the header must survive unchanged");
        assert_eq!(
            got_payload, payload,
            "byte for byte, NaN and subnormals included"
        );
    }

    /// Many requests share one connection, so frames must be readable back to back and each
    /// must carry its own identity - otherwise a multiplexed reply lands on the wrong request.
    #[test]
    fn frames_read_back_to_back_keep_their_identity() {
        let mut buf = Vec::new();
        for (i, hidden) in [8usize, 4096, 16].iter().enumerate() {
            let (mut h, p) = f16_activation(*hidden);
            h.request_id = i as u64 + 1;
            h.layer_id = i as u32 * 10;
            write_frame(&mut buf, &h, &p).unwrap();
        }
        let mut r = buf.as_slice();
        for (i, hidden) in [8usize, 4096, 16].iter().enumerate() {
            let (h, p) = read_frame(&mut r).unwrap();
            assert_eq!(h.request_id, i as u64 + 1);
            assert_eq!(h.layer_id, i as u32 * 10);
            assert_eq!(p.len(), hidden * 2);
        }
        assert!(r.is_empty(), "nothing left over between frames");
    }

    /// A frame cut short is the failure JSON cannot report: a shorter array is just a
    /// smaller tensor. Here it has to be an error.
    #[test]
    fn a_truncated_frame_is_an_error_not_a_smaller_tensor() {
        let (h, payload) = f16_activation(1024);
        let mut buf = Vec::new();
        write_frame(&mut buf, &h, &payload).unwrap();
        buf.truncate(buf.len() - 64);
        let err = read_frame(&mut buf.as_slice()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    /// The header carries the shape and the length twice over on purpose.
    #[test]
    fn a_frame_that_disagrees_with_itself_is_refused() {
        let (h, payload) = f16_activation(64);
        let mut buf = Vec::new();
        write_frame(&mut buf, &h, &payload).unwrap();
        // Rewrite the declared length: header ends with the u64 length before the payload.
        let len_at = buf.len() - payload.len() - 8;
        buf[len_at..len_at + 8].copy_from_slice(&(payload.len() as u64 - 2).to_le_bytes());
        let err = read_frame(&mut buf.as_slice()).unwrap_err();
        assert!(
            format!("{err}").contains("disagrees with itself"),
            "got: {err}"
        );

        // And the encoder refuses the same inconsistency before it reaches the wire.
        let mut short = Vec::new();
        assert!(write_frame(&mut short, &h, &payload[..payload.len() - 2]).is_err());
    }

    /// A corrupt length must not become an allocation.
    #[test]
    fn an_implausible_length_is_refused_before_allocating() {
        let (h, payload) = f16_activation(8);
        let mut buf = Vec::new();
        write_frame(&mut buf, &h, &payload).unwrap();
        let len_at = buf.len() - payload.len() - 8;
        buf[len_at..len_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let err = read_frame(&mut buf.as_slice()).unwrap_err();
        assert!(format!("{err}").contains("above the cap"), "got: {err}");
    }

    /// A stream that is not this protocol, and a peer speaking another version, both fail on
    /// the first frame rather than on a field that happened to parse.
    #[test]
    fn a_foreign_stream_and_a_foreign_version_both_fail_immediately() {
        let junk = b"HTTP/1.1 200 OK\r\n\r\n{\"shape\":[1,4096]}";
        let err = read_frame(&mut &junk[..]).unwrap_err();
        assert!(format!("{err}").contains("magic mismatch"), "got: {err}");

        let (h, payload) = f16_activation(4);
        let mut buf = Vec::new();
        write_frame(&mut buf, &h, &payload).unwrap();
        buf[4..6].copy_from_slice(&(VERSION + 1).to_le_bytes());
        let err = read_frame(&mut buf.as_slice()).unwrap_err();
        assert!(format!("{err}").contains("this build speaks"), "got: {err}");
    }

    /// What the frame is for: the same activation costs a fraction of its JSON form.
    #[test]
    fn the_frame_is_smaller_than_the_json_it_replaces() {
        let (h, payload) = f16_activation(4096);
        let mut buf = Vec::new();
        write_frame(&mut buf, &h, &payload).unwrap();

        // What the current transport sends: `Vec<u8>` as a JSON array of decimal numbers.
        let json = serde_json::to_vec(&serde_json::json!({
            "shape": [1, 4096],
            "dtype": "F16",
            "data": payload,
        }))
        .unwrap();

        assert!(
            buf.len() < payload.len() + 64,
            "header is a fixed handful of bytes"
        );
        assert!(
            (json.len() as f64) > 2.0 * buf.len() as f64,
            "json {} vs frame {}",
            json.len(),
            buf.len()
        );
    }
}
