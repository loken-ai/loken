use super::*;

/// `values`, `rows` of `width`, written in loken's format with `dtype` (f32, f16 or bf16).
fn encode(values: &[f32], rows: u32, width: u32, dtype: Dtype) -> Vec<u8> {
    let code = [Dtype::F32, Dtype::F16, Dtype::Bf16]
        .iter()
        .position(|&d| d == dtype)
        .expect("loken's format carries f32, f16 and bf16") as u8;
    let mut out = MAGIC.to_vec();
    out.extend(VERSION.to_le_bytes());
    out.extend([code, 0]);
    out.extend(rows.to_le_bytes());
    out.extend(width.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    for &v in values {
        match dtype {
            Dtype::F16 => out.extend(half::f16::from_f32(v).to_le_bytes()),
            Dtype::Bf16 => out.extend(half::bf16::from_f32(v).to_le_bytes()),
            _ => out.extend(v.to_le_bytes()),
        }
    }
    let crc = crc32(&out);
    out.extend(crc.to_le_bytes());
    out
}

/// The CRC-32 trailer of `payload` written again over everything before it.
fn reseal(mut payload: Vec<u8>) -> Vec<u8> {
    let body = payload.len() - CRC_BYTES;
    let crc = crc32(&payload[..body]);
    payload[body..].copy_from_slice(&crc.to_le_bytes());
    payload
}

fn values_of(bytes: Vec<u8>, width: usize) -> Result<Vec<f32>, String> {
    PromptEmbeds::from_bytes(bytes)?.into_values(width, 1 << 20)
}

const VALUES: [f32; 6] = [0.5, -1.25, 3.0, 0.0, 2.5, -0.75];

/// Each dtype gives back the values it carries; a half-precision payload's values take half
/// the bytes of the f32 one.
#[test]
fn loken_format_round_trips_each_dtype() {
    for dtype in [Dtype::F32, Dtype::F16, Dtype::Bf16] {
        let payload = encode(&VALUES, 2, 3, dtype);
        assert_eq!(
            payload.len(),
            HEADER_BYTES + VALUES.len() * dtype.size() + CRC_BYTES
        );
        assert_eq!(values_of(payload, 3).unwrap(), VALUES, "{dtype:?}");
    }
}

/// The magic's first four bytes are a NaN as an f32, so no bare f32 payload starts with them.
#[test]
fn the_magic_is_never_a_bare_f32_payload() {
    assert!(f32::from_le_bytes([MAGIC[0], MAGIC[1], MAGIC[2], MAGIC[3]]).is_nan());
}

/// A payload one byte short or one byte long is refused on its length, before its checksum.
#[test]
fn loken_format_refuses_a_truncated_or_padded_payload() {
    let payload = encode(&VALUES, 2, 3, Dtype::F16);
    let mut short = payload.clone();
    short.pop();
    let err = values_of(short, 3).unwrap_err();
    assert!(err.contains("truncated"), "{err}");
    let mut long = payload;
    long.push(0);
    assert!(values_of(long, 3).is_err());
    for n in 0..HEADER_BYTES + CRC_BYTES {
        assert!(values_of(MAGIC.iter().copied().cycle().take(n).collect(), 3).is_err());
    }
}

/// A flipped bit anywhere, header included, fails the checksum or the header's own checks.
#[test]
fn loken_format_refuses_any_corrupted_byte() {
    let payload = encode(&VALUES, 2, 3, Dtype::F16);
    for at in 0..payload.len() {
        let mut bad = payload.clone();
        bad[at] ^= 0x10;
        assert!(values_of(bad, 3).is_err(), "byte {at}");
    }
    // f16 and bf16 take the same bytes: the checksum covers the dtype.
    let mut swapped = payload;
    swapped[10] = 2;
    let err = values_of(swapped, 3).unwrap_err();
    assert!(err.contains("checksum"), "{err}");
}

/// Header fields out of range are refused with what they hold, without an allocation sized by
/// them.
#[test]
fn loken_format_refuses_a_bad_header() {
    let payload = encode(&VALUES, 2, 3, Dtype::F32);
    let with = |at: usize, bytes: &[u8]| {
        let mut p = payload.clone();
        p[at..at + bytes.len()].copy_from_slice(bytes);
        values_of(reseal(p), 3).unwrap_err()
    };
    assert!(with(8, &2u16.to_le_bytes()).contains("version 2"));
    assert!(with(10, &[7]).contains("dtype code 7"));
    assert!(with(11, &[1]).contains("reserved"));
    assert!(with(20, &[1]).contains("reserved"));
    let huge = with(12, &[0xff; 8]);
    assert!(huge.contains("4294967295 rows"), "{huge}");
    assert!(with(12, &0u32.to_le_bytes()).contains("holds"));
}

/// Rows checked against the model: their width, their count against the context window, and
/// each value finite.
#[test]
fn rows_are_checked_against_the_model() {
    let embeds = |p: Vec<u8>| PromptEmbeds::from_bytes(p).unwrap();
    let wide = embeds(encode(&VALUES, 2, 3, Dtype::F32)).into_values(4, 64);
    assert!(wide.unwrap_err().contains("3 wide"));
    let long = embeds(encode(&VALUES, 2, 3, Dtype::F32)).into_values(3, 2);
    assert!(long.unwrap_err().contains("context window"));
    assert!(embeds(encode(&VALUES, 2, 3, Dtype::F32))
        .into_values(3, 3)
        .is_ok());
    for bad in [f32::NAN, f32::INFINITY] {
        let mut v = VALUES;
        v[4] = bad;
        let err = embeds(encode(&v, 2, 3, Dtype::F16))
            .into_values(3, 64)
            .unwrap_err();
        assert!(err.contains("row 1"), "{err}");
    }
    // A bf16 value past f16's range is finite in bf16.
    let big = embeds(encode(&[1.0e30; 3], 1, 3, Dtype::Bf16));
    assert!(big.into_values(3, 64).is_ok());
}

/// Bare f32 values take the model's width and must fill whole rows; JSON rows must all be one
/// width.
#[test]
fn bare_values_and_json_rows_are_checked() {
    let raw: Vec<u8> = VALUES.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(values_of(raw.clone(), 3).unwrap(), VALUES);
    assert!(values_of(raw.clone(), 4).is_err());
    assert!(values_of(raw[..5].to_vec(), 3).is_err());
    assert!(values_of(Vec::new(), 3).is_err());
    let ragged = serde_json::json!([[1.0, 2.0], [3.0]]);
    assert!(PromptEmbeds::from_json(&ragged).is_err());
    assert!(PromptEmbeds::from_json(&serde_json::json!([])).is_err());
    assert!(PromptEmbeds::from_json(&serde_json::json!([[]])).is_err());
    let huge = PromptEmbeds::from_json(&serde_json::json!([[1.0e300]])).unwrap();
    assert!(huge.into_values(1, 64).is_err());
}

// torch.save, as torch's writer lays it out.

fn push_str(p: &mut Vec<u8>, s: &str) {
    p.push(b'X');
    p.extend((s.len() as u32).to_le_bytes());
    p.extend(s.as_bytes());
}

fn push_int(p: &mut Vec<u8>, v: usize) {
    if v < 1 << 8 {
        p.extend([b'K', v as u8]);
    } else if v < 1 << 16 {
        p.push(b'M');
        p.extend((v as u16).to_le_bytes());
    } else {
        p.push(b'J');
        p.extend((v as i32).to_le_bytes());
    }
}

fn push_tuple(p: &mut Vec<u8>, items: &[usize]) {
    match items.len() {
        0 => p.push(b')'),
        n @ 1..=3 => {
            items.iter().for_each(|&v| push_int(p, v));
            p.push(0x84 + n as u8);
        }
        _ => {
            p.push(b'(');
            items.iter().for_each(|&v| push_int(p, v));
            p.push(b't');
        }
    }
}

/// The pickle `torch.save` writes for one tensor over storage `key`.
fn tensor_pickle(
    storage: &str,
    key: &str,
    numel: usize,
    offset: usize,
    size: &[usize],
    stride: &[usize],
) -> Vec<u8> {
    let mut p = vec![0x80, 2];
    p.extend(b"ctorch._utils\n_rebuild_tensor_v2\nq\x00(");
    p.push(b'(');
    push_str(&mut p, "storage");
    p.extend(format!("ctorch\n{storage}\nq\x01").as_bytes());
    push_str(&mut p, key);
    push_str(&mut p, "cpu");
    push_int(&mut p, numel);
    p.extend(b"tq\x02Q");
    push_int(&mut p, offset);
    push_tuple(&mut p, size);
    push_tuple(&mut p, stride);
    p.push(0x89);
    p.extend(b"ccollections\nOrderedDict\nq\x03)Rq\x04tq\x05Rq\x06.");
    p
}

/// A stored zip of `entries`, with zip64 end records when `zip64`.
fn zip(entries: &[(&str, &[u8])], zip64: bool) -> Vec<u8> {
    let (mut out, mut central) = (Vec::new(), Vec::new());
    for (name, data) in entries {
        let offset = out.len() as u32;
        let crc = crc32(data);
        let len = data.len() as u32;
        out.extend(0x0403_4b50u32.to_le_bytes());
        out.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend(crc.to_le_bytes());
        out.extend(len.to_le_bytes());
        out.extend(len.to_le_bytes());
        out.extend((name.len() as u16).to_le_bytes());
        out.extend([0, 0]);
        out.extend(name.as_bytes());
        out.extend(*data);
        central.extend(0x0201_4b50u32.to_le_bytes());
        central.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend(crc.to_le_bytes());
        central.extend(len.to_le_bytes());
        central.extend(len.to_le_bytes());
        central.extend((name.len() as u16).to_le_bytes());
        central.extend([0; 12]);
        central.extend(offset.to_le_bytes());
        central.extend(name.as_bytes());
    }
    let (cd, cd_size, n) = (out.len(), central.len(), entries.len());
    out.extend(central);
    if zip64 {
        let rec = out.len() as u64;
        out.extend(0x0606_4b50u32.to_le_bytes());
        out.extend(44u64.to_le_bytes());
        out.extend([45, 0, 45, 0]);
        out.extend([0; 8]);
        out.extend((n as u64).to_le_bytes());
        out.extend((n as u64).to_le_bytes());
        out.extend((cd_size as u64).to_le_bytes());
        out.extend((cd as u64).to_le_bytes());
        out.extend(0x0706_4b50u32.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(rec.to_le_bytes());
        out.extend(1u32.to_le_bytes());
    }
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend([0; 4]);
    out.extend((n as u16).to_le_bytes());
    out.extend((n as u16).to_le_bytes());
    out.extend((cd_size as u32).to_le_bytes());
    out.extend((cd as u32).to_le_bytes());
    out.extend([0, 0]);
    out
}

/// The archive `torch.save` writes for a pickle over one storage.
fn torch_archive(pickle: &[u8], storage: &[u8]) -> Vec<u8> {
    zip(
        &[
            ("archive/data.pkl", pickle),
            ("archive/byteorder", &b"little"[..]),
            ("archive/data/0", storage),
            ("archive/version", &b"3\n"[..]),
        ],
        false,
    )
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The 2-by-3 f32 tensor of `VALUES`, contiguous.
fn plain_tensor() -> Vec<u8> {
    let pickle = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    torch_archive(&pickle, &f32_bytes(&VALUES))
}

#[test]
fn torch_reads_a_contiguous_tensor_in_two_or_three_dimensions() {
    assert_eq!(values_of(plain_tensor(), 3).unwrap(), VALUES);
    let pickle = tensor_pickle("FloatStorage", "0", 6, 0, &[1, 2, 3], &[6, 3, 1]);
    let batch = torch_archive(&pickle, &f32_bytes(&VALUES));
    assert_eq!(values_of(batch, 3).unwrap(), VALUES);
    let pickle = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    let storage = f32_bytes(&VALUES);
    let entries: [(&str, &[u8]); 2] = [("t/data.pkl", &pickle[..]), ("t/data/0", &storage[..])];
    assert_eq!(values_of(zip(&entries, true), 3).unwrap(), VALUES);
}

/// A transposed tensor and a slice at an offset read through their strides.
#[test]
fn torch_reads_a_strided_view() {
    // Storage [[0.5, -1.25], [3.0, 0.0], [2.5, -0.75]] read transposed.
    let pickle = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[1, 2]);
    let t = values_of(torch_archive(&pickle, &f32_bytes(&VALUES)), 3).unwrap();
    assert_eq!(t, [0.5, 3.0, 2.5, -1.25, 0.0, -0.75]);
    // The last row of a 3-by-2 storage.
    let pickle = tensor_pickle("FloatStorage", "0", 6, 4, &[1, 2], &[2, 1]);
    let s = values_of(torch_archive(&pickle, &f32_bytes(&VALUES)), 2).unwrap();
    assert_eq!(s, [2.5, -0.75]);
}

#[test]
fn torch_reads_each_float_dtype() {
    let cases: [(&str, Vec<u8>); 3] = [
        (
            "HalfStorage",
            VALUES
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect(),
        ),
        (
            "BFloat16Storage",
            VALUES
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect(),
        ),
        (
            "DoubleStorage",
            VALUES
                .iter()
                .flat_map(|&v| f64::from(v).to_le_bytes())
                .collect(),
        ),
    ];
    for (storage, bytes) in cases {
        let pickle = tensor_pickle(storage, "0", 6, 0, &[2, 3], &[3, 1]);
        assert_eq!(
            values_of(torch_archive(&pickle, &bytes), 3).unwrap(),
            VALUES,
            "{storage}"
        );
    }
    let pickle = tensor_pickle("LongStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    let err = values_of(torch_archive(&pickle, &[0; 48]), 3).unwrap_err();
    assert!(err.contains("LongStorage"), "{err}");
}

/// Every prefix of a genuine archive, and every prefix of its pickle sealed in a valid archive,
/// is refused without a panic.
#[test]
fn torch_refuses_every_truncation() {
    let whole = plain_tensor();
    for n in 0..whole.len() {
        assert!(values_of(whole[..n].to_vec(), 3).is_err(), "prefix {n}");
    }
    let pickle = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    for n in 0..pickle.len() {
        let archive = torch_archive(&pickle[..n], &f32_bytes(&VALUES));
        assert!(values_of(archive, 3).is_err(), "pickle prefix {n}");
    }
}

/// A flipped byte anywhere in a genuine archive either leaves the values as they were or is
/// refused, never a panic and never other values.
#[test]
fn torch_never_reads_a_corrupted_archive_as_other_values() {
    let whole = plain_tensor();
    for at in 0..whole.len() {
        for flip in [0x01, 0x80, 0xff] {
            let mut bad = whole.clone();
            bad[at] ^= flip;
            if let Ok(v) = values_of(bad, 3) {
                assert_eq!(v, VALUES, "byte {at} flipped by {flip:#x}");
            }
        }
    }
}

/// Globals, opcodes and calls outside a plain tensor's pickle are refused, with the global
/// named.
#[test]
fn torch_refuses_hostile_pickles() {
    let storage = f32_bytes(&VALUES);
    let refused = |pickle: &[u8]| values_of(torch_archive(pickle, &storage), 3).unwrap_err();
    for (module, name) in [
        ("os", "system"),
        ("builtins", "eval"),
        ("pickle", "loads"),
        ("torch", "load"),
        ("torch._utils", "_rebuild_parameter"),
    ] {
        let p = [
            &b"\x80\x02c"[..],
            module.as_bytes(),
            b"\n",
            name.as_bytes(),
            b"\nX\x02\x00\x00\x00idtR.",
        ]
        .concat();
        let err = refused(&p);
        assert!(err.contains(&format!("{module}.{name}")), "{err}");
    }
    // STACK_GLOBAL naming a forbidden callable.
    let mut p = vec![0x80, 4];
    push_str(&mut p, "os");
    push_str(&mut p, "system");
    p.extend([0x93, b'.']);
    assert!(refused(&p).contains("os.system"));
    // Opcodes no tensor pickle uses: INST, OBJ, BUILD, BINBYTES, SETITEMS, NONE.
    for op in *b"iobBuN" {
        let err = refused(&[0x80, 2, op, b'.']);
        assert!(err.contains("opcode"), "{err}");
    }
    // A nested call: the storage argument is itself a rebuilt tensor.
    let inner = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    let mut p = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(".to_vec();
    p.extend(&inner[2..inner.len() - 1]);
    push_int(&mut p, 0);
    push_tuple(&mut p, &[2, 3]);
    push_tuple(&mut p, &[3, 1]);
    p.extend(b"\x89ccollections\nOrderedDict\n)RtR.");
    assert!(refused(&p).contains("arguments"));
    // The metadata dictionary a plain tensor never carries.
    let mut p = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    let at = p.len() - b"tq\x05Rq\x06.".len();
    p.splice(at..at, *b"}");
    assert!(refused(&p).contains("opcode"));
    // A persistent reference that is not a storage, and a key that walks the archive.
    assert!(refused(b"\x80\x02X\x01\x00\x00\x00xQ.").contains("persistent"));
    let p = tensor_pickle("FloatStorage", "../0", 6, 0, &[2, 3], &[3, 1]);
    assert!(refused(&p).contains("persistent"));
    // Bytes after STOP, and a stop with nothing built.
    let mut p = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    p.push(b'.');
    assert!(refused(&p).contains("after its end"));
    assert!(refused(b"\x80\x02K\x01.").contains("one tensor"));
}

/// Lengths and counts past what one tensor needs are refused before anything is sized by them.
#[test]
fn torch_refuses_oversized_lengths() {
    let storage = f32_bytes(&VALUES);
    let refused = |pickle: &[u8]| values_of(torch_archive(pickle, &storage), 3).unwrap_err();
    assert!(refused(b"\x80\x02X\xff\xff\xff\xffab.").contains("string of"));
    let mut long = vec![0x80, 2, 0x8a, 200];
    long.extend([0; 200]);
    long.push(b'.');
    assert!(refused(&long).contains("integer of 200 bytes"));
    let deep = [vec![0x80, 2], vec![b'('; 300], vec![b'.']].concat();
    assert!(refused(&deep).contains("stack"));
    let mut memo = vec![0x80, 2, b'K', 1];
    for slot in 0..300u32 {
        memo.push(b'r');
        memo.extend(slot.to_le_bytes());
    }
    memo.push(b'.');
    assert!(refused(&memo).contains("memo"));
    // A tuple doubled through the memo forty times stays cheap: values share their contents.
    let mut doubling = vec![0x80, 2, b')', b'q', 0];
    for _ in 0..40 {
        doubling.extend([b'h', 0, b'h', 0, 0x86, b'q', 0]);
    }
    doubling.push(b'.');
    assert!(refused(&doubling).contains("one tensor"));
    // A pickle past its cap.
    let big = vec![b'K'; 1 << 17];
    assert!(refused(&big).contains("pickle is"));
    // A storage view reaching past its entry, and an entry shorter than its reference says.
    let p = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[4, 1]);
    assert!(refused(&p).contains("past the end"));
    let p = tensor_pickle("FloatStorage", "0", 7, 0, &[2, 3], &[3, 1]);
    assert!(refused(&p).contains("storage holds"));
    // A storage length past usize, as a LONG1 of eight bytes.
    let mut p = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    let numel = b"cpuK\x06t";
    let at = p.windows(numel.len()).position(|w| w == numel).unwrap() + 3;
    p.splice(at..at + 2, [0x8a, 8, 0, 0, 0, 0, 0, 0, 0, 0x40]);
    assert!(refused(&p).contains("storage holds"));
}

/// Archives torch does not write are refused: compressed, big-endian, with two pickles, with
/// a missing storage, with too many entries, in the legacy format.
#[test]
fn torch_refuses_archives_torch_does_not_write() {
    let pickle = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
    let storage = f32_bytes(&VALUES);
    let mut compressed = plain_tensor();
    compressed[8] = 8;
    let at = (0..compressed.len())
        .find(|&i| compressed[i..].starts_with(&0x0201_4b50u32.to_le_bytes()))
        .unwrap();
    compressed[at + 10] = 8;
    assert!(values_of(compressed, 3).unwrap_err().contains("compressed"));
    let (pickle, storage) = (&pickle[..], &storage[..]);
    let big_endian = zip(
        &[
            ("archive/data.pkl", pickle),
            ("archive/byteorder", &b"big"[..]),
            ("archive/data/0", storage),
        ],
        false,
    );
    assert!(values_of(big_endian, 3).unwrap_err().contains("big-endian"));
    let two = zip(
        &[
            ("a/data.pkl", pickle),
            ("b/data.pkl", pickle),
            ("a/data/0", storage),
        ],
        false,
    );
    assert!(values_of(two, 3).unwrap_err().contains("2 `data.pkl`"));
    let twice = zip(
        &[
            ("archive/data.pkl", pickle),
            ("archive/data.pkl", pickle),
            ("archive/data/0", storage),
        ],
        false,
    );
    assert!(values_of(twice, 3).unwrap_err().contains("twice"));
    let missing = zip(&[("archive/data.pkl", pickle)], false);
    assert!(values_of(missing, 3).unwrap_err().contains("lacks"));
    let names: Vec<String> = (0..100).map(|i| format!("archive/x{i}")).collect();
    let mut many: Vec<(&str, &[u8])> = names.iter().map(|n| (n.as_str(), &b""[..])).collect();
    many.push(("archive/data.pkl", pickle));
    assert!(values_of(zip(&many, false), 3)
        .unwrap_err()
        .contains("entries"));
    let legacy = [&TORCH_LEGACY_MAGIC[..], b"rest"].concat();
    assert!(values_of(legacy, 3).unwrap_err().contains("legacy"));
    // A batch of two, and a 1-D tensor, are not a prompt.
    let p = tensor_pickle("FloatStorage", "0", 6, 0, &[2, 1, 3], &[3, 3, 1]);
    assert!(values_of(torch_archive(&p, storage), 3)
        .unwrap_err()
        .contains("shape"));
    let p = tensor_pickle("FloatStorage", "0", 6, 0, &[6], &[1]);
    assert!(values_of(torch_archive(&p, storage), 3)
        .unwrap_err()
        .contains("shape"));
}

/// The files `scripts/prompt_embeds_vectors.py` writes with torch and with the reference
/// encoder of loken's format, each read back to the values torch holds.
#[test]
fn genuine_files_read_back_to_their_values() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/prompt_embeds");
    let manifest = std::fs::read(dir.join("manifest.json")).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; run scripts/prompt_embeds_vectors.py {}",
            dir.display(),
            dir.display()
        )
    });
    let manifest: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let file = case["file"].as_str().unwrap();
        let width = case["width"].as_u64().unwrap() as usize;
        let expected: Vec<f32> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let bytes = std::fs::read(dir.join(file)).unwrap();
        let values = values_of(bytes, width).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(values, expected, "{file}");
    }
}
