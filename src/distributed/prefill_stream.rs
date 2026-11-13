//! The streamed answer of `/api/cluster/prefill`: the peer's reading of the prompt as it goes,
//! then the handover.
//!
//! The asking instance names [`MEDIA_TYPE`] in its `Accept` header.
//! A peer that knows it answers in that type.
//! A peer that does not answers the handover alone as `application/octet-stream`.
//!
//! The stream is a sequence of records, each a line of JSON:
//!
//! - `{"done":n,"total":m}`: the peer has read `n` of the `m` tokens it reads.
//! - `{"error":"..."}`: the reading failed, and the stream ends.
//! - `{"handover":n}`: the last record, followed by the `n` bytes of the handover.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, Read};

/// The media type of the streamed answer.
pub const MEDIA_TYPE: &str = "application/vnd.loken.prefill-stream";

/// One record of the stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Record {
    Progress { done: u64, total: u64 },
    Error { error: String },
    Handover { handover: u64 },
}

/// The line of `record`, its newline included.
pub fn line(record: &Record) -> Vec<u8> {
    let mut out = serde_json::to_vec(record).unwrap_or_default();
    out.push(b'\n');
    out
}

/// Reads a stream from `reader`, calling `progress` with each progress record's `(done, total)`
/// as it arrives.
/// Returns the handover's bytes, or the error the peer or the stream gave.
pub fn read(
    mut reader: impl BufRead,
    mut progress: impl FnMut(u64, u64),
) -> Result<Vec<u8>, String> {
    let mut text = Vec::new();
    loop {
        text.clear();
        let n = reader
            .read_until(b'\n', &mut text)
            .map_err(|e| format!("prefill stream: {e}"))?;
        if n == 0 {
            return Err("prefill stream: it ended before the handover".to_string());
        }
        let record: Record = serde_json::from_slice(&text)
            .map_err(|e| format!("prefill stream: a record does not parse: {e}"))?;
        match record {
            Record::Progress { done, total } => progress(done, total),
            Record::Error { error } => return Err(error),
            Record::Handover { handover } => {
                let mut bytes = Vec::with_capacity(handover as usize);
                let got = reader
                    .by_ref()
                    .take(handover)
                    .read_to_end(&mut bytes)
                    .map_err(|e| format!("prefill stream: {e}"))?;
                if got as u64 != handover {
                    return Err(format!(
                        "prefill stream: {got} bytes of a handover of {handover}"
                    ));
                }
                return Ok(bytes);
            }
        }
    }
}

/// Whether the `Accept` header value `accept` names [`MEDIA_TYPE`].
pub fn accepted(accept: &str) -> bool {
    accept
        .split(',')
        .filter_map(|range| range.split(';').next())
        .any(|range| range.trim().eq_ignore_ascii_case(MEDIA_TYPE))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The last record and the handover's bytes after it.
    fn payload(handover: Vec<u8>) -> Vec<u8> {
        let mut out = line(&Record::Handover {
            handover: handover.len() as u64,
        });
        out.extend_from_slice(&handover);
        out
    }

    #[test]
    fn a_stream_gives_its_progress_in_order_then_the_handover() {
        let handover: Vec<u8> = (0..=255u8).chain(*b"\n{").collect();
        let mut stream = line(&Record::Progress { done: 0, total: 9 });
        stream.extend(line(&Record::Progress { done: 4, total: 9 }));
        stream.extend(payload(handover.clone()));
        let mut seen = Vec::new();
        let got = read(stream.as_slice(), |d, t| seen.push((d, t))).unwrap();
        assert_eq!(seen, vec![(0, 9), (4, 9)]);
        assert_eq!(got, handover);
    }

    #[test]
    fn an_error_record_or_a_short_stream_fails_the_read() {
        let stream = line(&Record::Error {
            error: "no model".to_string(),
        });
        assert_eq!(
            read(stream.as_slice(), |_, _| {}),
            Err("no model".to_string())
        );
        let mut cut = payload(vec![1, 2, 3]);
        cut.pop();
        assert!(read(cut.as_slice(), |_, _| {}).is_err());
        assert!(read(&b""[..], |_, _| {}).is_err());
    }

    #[test]
    fn the_media_type_is_found_among_the_accepted_ones() {
        assert!(accepted(MEDIA_TYPE));
        assert!(accepted(&format!(
            "{MEDIA_TYPE}; q=1, application/octet-stream; q=0.5"
        )));
        assert!(!accepted("application/octet-stream"));
        assert!(!accepted("*/*"));
    }
}
