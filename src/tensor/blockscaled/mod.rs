//! Weights read in place in the formats a released checkpoint ships: e4m3 fp8 with a scale per
//! tile, e2m1 fp4 with a scale per block of inputs, and bf16. Each is a `[out, in]` weight whose
//! product with an activation runs straight from the mapping.

pub mod bf16;
pub mod dequant;
pub mod fp4;
pub mod fp8;

/// The block-scaled fp8 tensors `scripts/fp8_block_vectors.py` writes, read for the tests of
/// the readers.
#[cfg(test)]
pub(crate) mod test_vectors {
    use std::collections::HashMap;

    /// One stored tensor: its shape and raw bytes.
    pub struct Stored {
        pub shape: Vec<usize>,
        pub bytes: Vec<u8>,
    }

    impl Stored {
        pub fn f32s(&self) -> Vec<f32> {
            self.bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    }

    /// Every tensor of the vector file by name.
    pub fn block_vectors() -> HashMap<String, Stored> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/vectors/fp8_block/blocks.safetensors");
        let file = std::fs::read(path).unwrap();
        let len = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&file[8..8 + len]).unwrap();
        let mut out = HashMap::new();
        for (name, v) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let shape = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let at = |i: usize| v["data_offsets"][i].as_u64().unwrap() as usize;
            let bytes = file[8 + len + at(0)..8 + len + at(1)].to_vec();
            out.insert(name.clone(), Stored { shape, bytes });
        }
        out
    }
}
