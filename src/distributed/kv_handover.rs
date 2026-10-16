//! A prompt's KV cache handed from the instance that read the prompt to the instance that decodes
//! it: the tokens it holds, the cache's format and every layer's key and value rows.
//!
//! The decoding instance imports the rows as a snapshot of its own cache, which its prefix reuse
//! then restores like any other, so the prompt is not read twice. Both instances serve the same
//! model: the layout carried here is checked against the importer's before a row is read.
//!
//! Layout on the wire, little-endian: the magic, a header length and the header as JSON, then for
//! each layer a byte (1 when the layer holds rows), the count of key values and the keys as f32,
//! the count of value values and the values as f32.

use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"LKVH";

/// The rows of every layer, `None` for a layer that holds none: keys then values, as f32.
pub type LayerRows = Vec<Option<(Vec<f32>, Vec<f32>)>>;

/// What the header says of the rows that follow it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    /// The tokens the rows hold, from the start of the sequence.
    pub tokens: Vec<u32>,
    /// Positions of the cache the rows fill.
    pub kv_len: usize,
    /// The cache's element format, as the snapshot import names it.
    pub dtype: u8,
    /// The cache's shape (layers, kv heads, head width) packed as the importer packs its own.
    pub layout: u64,
    pub layers: usize,
}

/// A handed-over cache.
#[derive(Debug, Clone, PartialEq)]
pub struct Handover {
    pub header: Header,
    pub rows: LayerRows,
}

/// `handover` as the bytes the wire carries.
pub fn encode(handover: &Handover) -> Vec<u8> {
    let header = serde_json::to_vec(&handover.header).expect("a header serialises");
    let values: usize = handover
        .rows
        .iter()
        .flatten()
        .map(|(k, v)| k.len() + v.len())
        .sum();
    let mut out = Vec::with_capacity(8 + header.len() + handover.rows.len() * 17 + values * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&header);
    for layer in &handover.rows {
        match layer {
            None => out.push(0),
            Some((k, v)) => {
                out.push(1);
                for part in [k, v] {
                    out.extend_from_slice(&(part.len() as u64).to_le_bytes());
                    for x in part {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

/// The handover `bytes` carry, or why they are not one.
pub fn decode(bytes: &[u8]) -> Result<Handover, String> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], String> {
        let end = at
            .checked_add(n)
            .filter(|&e| e <= bytes.len())
            .ok_or("the handover ends early")?;
        let part = &bytes[at..end];
        at = end;
        Ok(part)
    };
    if take(4)? != MAGIC {
        return Err("not a KV handover".into());
    }
    let header_len = u32::from_le_bytes(take(4)?.try_into().expect("four bytes")) as usize;
    let header: Header =
        serde_json::from_slice(take(header_len)?).map_err(|e| format!("handover header: {e}"))?;
    let mut rows = Vec::with_capacity(header.layers);
    for _ in 0..header.layers {
        match take(1)?[0] {
            0 => rows.push(None),
            1 => {
                let mut read = || -> Result<Vec<f32>, String> {
                    let n = u64::from_le_bytes(take(8)?.try_into().expect("eight bytes")) as usize;
                    let raw = take(
                        n.checked_mul(4)
                            .ok_or("the handover's row count overflows")?,
                    )?;
                    Ok(raw
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
                        .collect())
                };
                let k = read()?;
                let v = read()?;
                rows.push(Some((k, v)));
            }
            flag => return Err(format!("handover layer flag {flag}")),
        }
    }
    if at != bytes.len() {
        return Err("the handover has bytes past its last layer".into());
    }
    Ok(Handover { header, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Handover {
        Handover {
            header: Header {
                tokens: vec![1, 4, 2, 9],
                kv_len: 4,
                dtype: 0,
                layout: (3 << 40) | (2 << 20) | 8,
                layers: 3,
            },
            rows: vec![
                Some((vec![0.5, -1.25, 3.0], vec![2.0, f32::MIN_POSITIVE, -0.0])),
                None,
                Some((vec![], vec![7.5])),
            ],
        }
    }

    /// A handover comes back as it left, bit for bit.
    #[test]
    fn a_handover_round_trips() {
        let h = sample();
        let back = decode(&encode(&h)).unwrap();
        assert_eq!(back.header, h.header);
        let bits = |r: &LayerRows| -> Vec<Option<(Vec<u32>, Vec<u32>)>> {
            r.iter()
                .map(|l| {
                    l.as_ref().map(|(k, v)| {
                        (
                            k.iter().map(|x| x.to_bits()).collect(),
                            v.iter().map(|x| x.to_bits()).collect(),
                        )
                    })
                })
                .collect()
        };
        assert_eq!(bits(&back.rows), bits(&h.rows));
    }

    /// A cut, padded or foreign body is refused rather than read as rows.
    #[test]
    fn a_damaged_handover_is_refused() {
        let bytes = encode(&sample());
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(decode(&longer).is_err());
        assert!(decode(b"NOPE").is_err());
    }
}
