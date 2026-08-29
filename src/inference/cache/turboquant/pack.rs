//! The 8-in-3-bytes bit layout.
//!
//! Three bits do not divide a byte, so the unit of storage is eight values: eight
//! three-bit indices are twenty-four bits, which is exactly three bytes with nothing left
//! over and no index straddling the end of the group. Anything smaller either wastes bits
//! or splits an index across a group boundary, and a kernel that has to reassemble an
//! index from two groups is a kernel that reads twice.
//!
//! Bit order is little-endian within the group: index `i` occupies bits `[3i, 3i + 3)` of
//! a twenty-four bit word, and the word is stored low byte first. Stated here because it
//! is the one thing a device kernel would have to agree with exactly, and disagreeing
//! about it produces plausible-looking values rather than an error.

/// Values in one packed group.
pub const GROUP_VALUES: usize = 8;

/// Bytes one packed group occupies.
pub const GROUP_BYTES: usize = 3;

/// Pack eight three-bit indices into three bytes.
///
/// Indices above seven are a caller bug; they are masked rather than allowed to corrupt
/// their neighbours, and a debug build asserts.
pub fn pack_group(indices: &[u8; GROUP_VALUES]) -> [u8; GROUP_BYTES] {
    let mut word: u32 = 0;
    for (i, &v) in indices.iter().enumerate() {
        debug_assert!(v < 8, "index {v} does not fit in three bits");
        word |= ((v & 0x7) as u32) << (3 * i);
    }
    [word as u8, (word >> 8) as u8, (word >> 16) as u8]
}

/// Unpack three bytes back into eight three-bit indices.
pub fn unpack_group(bytes: &[u8; GROUP_BYTES]) -> [u8; GROUP_VALUES] {
    let word = bytes[0] as u32 | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16);
    std::array::from_fn(|i| ((word >> (3 * i)) & 0x7) as u8)
}

/// Pack a whole run of indices, whose length must be a multiple of [`GROUP_VALUES`].
pub fn pack_run(indices: &[u8], out: &mut Vec<u8>) {
    assert_eq!(
        indices.len() % GROUP_VALUES,
        0,
        "a packed run is a whole number of eight-value groups"
    );
    for chunk in indices.chunks_exact(GROUP_VALUES) {
        let group: [u8; GROUP_VALUES] = std::array::from_fn(|i| chunk[i]);
        out.extend_from_slice(&pack_group(&group));
    }
}

/// Unpack a whole run, the inverse of [`pack_run`].
pub fn unpack_run(bytes: &[u8], out: &mut Vec<u8>) {
    assert_eq!(
        bytes.len() % GROUP_BYTES,
        0,
        "a packed run is a whole number of three-byte groups"
    );
    for chunk in bytes.chunks_exact(GROUP_BYTES) {
        let group: [u8; GROUP_BYTES] = [chunk[0], chunk[1], chunk[2]];
        out.extend_from_slice(&unpack_group(&group));
    }
}

/// Bytes needed to hold `values` three-bit indices.
pub fn packed_len(values: usize) -> usize {
    assert_eq!(
        values % GROUP_VALUES,
        0,
        "a packed run is a whole number of eight-value groups"
    );
    values / GROUP_VALUES * GROUP_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::cache::turboquant::rng::split_mix_64;

    #[test]
    fn turboquant_pack_round_trips_random_indices() {
        let mut state = 0xA5A5_1234_u64;
        for _ in 0..20_000 {
            let indices: [u8; GROUP_VALUES] =
                std::array::from_fn(|_| (split_mix_64(&mut state) & 0x7) as u8);
            let packed = pack_group(&indices);
            assert_eq!(unpack_group(&packed), indices, "group round trip failed");
        }
    }

    #[test]
    fn turboquant_pack_covers_every_code_in_every_slot() {
        // A layout bug that only shows on one slot is the usual kind. Every index value in
        // every position, with the other seven held at a value that would mask a leak.
        for slot in 0..GROUP_VALUES {
            for v in 0..8u8 {
                let mut indices = [7u8; GROUP_VALUES];
                indices[slot] = v;
                assert_eq!(
                    unpack_group(&pack_group(&indices)),
                    indices,
                    "slot {slot} value {v}"
                );
                let mut indices = [0u8; GROUP_VALUES];
                indices[slot] = v;
                assert_eq!(
                    unpack_group(&pack_group(&indices)),
                    indices,
                    "slot {slot} value {v} against zeros"
                );
            }
        }
    }

    #[test]
    fn turboquant_pack_run_round_trips_and_has_the_stated_size() {
        let mut state = 99u64;
        let values = 8 * 977;
        let indices: Vec<u8> = (0..values)
            .map(|_| (split_mix_64(&mut state) & 0x7) as u8)
            .collect();
        let mut packed = Vec::new();
        pack_run(&indices, &mut packed);
        assert_eq!(packed.len(), packed_len(values));
        assert_eq!(packed.len() * 8, values * 3, "not three bits per value");
        let mut back = Vec::new();
        unpack_run(&packed, &mut back);
        assert_eq!(back, indices);
    }

    #[test]
    fn turboquant_pack_bit_layout_is_the_documented_one() {
        // Pins the wire format itself, not just its self-consistency: a kernel written
        // against the doc comment must produce these exact bytes.
        assert_eq!(pack_group(&[1, 0, 0, 0, 0, 0, 0, 0]), [0x01, 0x00, 0x00]);
        assert_eq!(pack_group(&[0, 1, 0, 0, 0, 0, 0, 0]), [0x08, 0x00, 0x00]);
        assert_eq!(pack_group(&[0, 0, 1, 0, 0, 0, 0, 0]), [0x40, 0x00, 0x00]);
        assert_eq!(pack_group(&[0, 0, 0, 1, 0, 0, 0, 0]), [0x00, 0x02, 0x00]);
        // Slot 7 starts at bit 21, so it is the top three bits of the last byte.
        assert_eq!(pack_group(&[0, 0, 0, 0, 0, 0, 0, 1]), [0x00, 0x00, 0x20]);
        assert_eq!(pack_group(&[0, 0, 0, 0, 0, 0, 0, 7]), [0x00, 0x00, 0xE0]);
        assert_eq!(pack_group(&[7; GROUP_VALUES]), [0xFF, 0xFF, 0xFF]);
        assert_eq!(pack_group(&[0; GROUP_VALUES]), [0x00, 0x00, 0x00]);
    }
}
