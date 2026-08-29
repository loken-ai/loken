//! Dependency-free reader for PyTorch `.pt`/`.pth` checkpoints -> native tensors.
//!
//! A torch checkpoint is a ZIP archive (entries stored UNCOMPRESSED) containing a
//! pickle (`<root>/data.pkl`) describing the `state_dict` plus one raw little-endian
//! storage blob per tensor at `<root>/data/<key>`. We parse the ZIP central directory
//! and a minimal subset of the pickle opcodes ourselves - no `zip`/`byteorder`
//! dependency - and rebuild `crate::tensor::Tensor` values the same way
//! `safetensors_io` does. Pickle-VM logic mirrors the well-known torch format
//! but is written here over std alone.

use super::{CpuStorage, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;

// -- minimal stored-ZIP reader ------------------------------------------------
struct ZipEntry {
    offset: u64,
    size: u64,
} // offset/size of the raw (stored) data

struct Zip {
    data: memmap2::Mmap,
    entries: HashMap<String, ZipEntry>,
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

impl Zip {
    fn open(path: &str) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|e| Error(format!("pth open {path}: {e}")))?;
        let data =
            unsafe { memmap2::Mmap::map(&f) }.map_err(|e| Error(format!("pth mmap: {e}")))?;
        let n = data.len();
        // Find the End-Of-Central-Directory record (sig 0x06054b50), scanning back over
        // the (<=64KB) trailing comment.
        let mut eocd = None;
        let lo = n.saturating_sub(65_557);
        for i in (lo..=n.saturating_sub(22)).rev() {
            if u32le(&data, i) == 0x0605_4b50 {
                eocd = Some(i);
                break;
            }
        }
        let eocd = eocd.ok_or_else(|| Error("pth: no ZIP EOCD".into()))?;
        let mut cd_count = u16le(&data, eocd + 10) as u64;
        let mut cd_off = u32le(&data, eocd + 16) as u64;
        // ZIP64: the 32-bit fields saturate to 0xFFFF/0xFFFFFFFF -> follow the locator.
        if cd_count == 0xFFFF || cd_off == 0xFFFF_FFFF {
            // ZIP64 EOCD locator (sig 0x07064b50) sits 20 bytes before the EOCD.
            let loc = eocd - 20;
            if u32le(&data, loc) == 0x0706_4b50 {
                let z64 = u64le(&data, loc + 8) as usize;
                if u32le(&data, z64) == 0x0606_4b50 {
                    cd_count = u64le(&data, z64 + 32);
                    cd_off = u64le(&data, z64 + 48);
                }
            }
        }
        let mut entries = HashMap::new();
        let mut p = cd_off as usize;
        for _ in 0..cd_count {
            if u32le(&data, p) != 0x0201_4b50 {
                return Err(Error("pth: bad central dir".into()));
            }
            let method = u16le(&data, p + 10);
            let mut comp_size = u32le(&data, p + 20) as u64;
            let name_len = u16le(&data, p + 28) as usize;
            let extra_len = u16le(&data, p + 30) as usize;
            let comment_len = u16le(&data, p + 32) as usize;
            let mut lho = u32le(&data, p + 42) as u64;
            let name = String::from_utf8_lossy(&data[p + 46..p + 46 + name_len]).to_string();
            // ZIP64 extra field (id 0x0001) carries 64-bit sizes/offset when 32-bit saturated.
            if comp_size == 0xFFFF_FFFF || lho == 0xFFFF_FFFF {
                let mut ep = p + 46 + name_len;
                let end = ep + extra_len;
                while ep + 4 <= end {
                    let id = u16le(&data, ep);
                    let sz = u16le(&data, ep + 4 - 2) as usize;
                    let mut dp = ep + 4;
                    if id == 0x0001 {
                        // order: uncompressed, compressed, lho (each present only if saturated)
                        let usize32 = u32le(&data, p + 24); // uncompressed (skip)
                        if usize32 == 0xFFFF_FFFF {
                            dp += 8;
                        }
                        if comp_size == 0xFFFF_FFFF {
                            comp_size = u64le(&data, dp);
                            dp += 8;
                        }
                        if lho == 0xFFFF_FFFF {
                            lho = u64le(&data, dp);
                        }
                    }
                    ep += 4 + sz;
                }
            }
            let _ = method; // torch uses STORED; data offset computed from the local header
                            // Local file header: 30-byte fixed + name + extra -> then the stored data.
            let lh = lho as usize;
            let lname = u16le(&data, lh + 26) as usize;
            let lextra = u16le(&data, lh + 28) as usize;
            let doff = lh + 30 + lname + lextra;
            entries.insert(
                name,
                ZipEntry {
                    offset: doff as u64,
                    size: comp_size,
                },
            );
            p += 46 + name_len + extra_len + comment_len;
        }
        Ok(Self { data, entries })
    }
    fn read(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .get(name)
            .map(|e| &self.data[e.offset as usize..(e.offset + e.size) as usize])
    }
    fn find_suffix(&self, suffix: &str) -> Option<String> {
        self.entries.keys().find(|k| k.ends_with(suffix)).cloned()
    }
}

// -- minimal pickle VM (torch state_dict subset) ------------------------------
/// The pickle object model, as far as a checkpoint needs it.
///
/// Some payloads are carried but never read back: a `REDUCE` or a `BUILD` is followed for
/// its effect on the stack, and its arguments matter only while that opcode executes.
/// Dropping them from the type would make the parser lie about what a pickle stream holds.
#[derive(Debug, Clone)]
enum Obj {
    Class(String, String), // module, class
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    None,
    Tuple(Vec<Obj>),
    List(Vec<Obj>),
    Dict(Vec<(Obj, Obj)>),
    Mark,
    PersId(Box<Obj>),
    Reduce(Box<Obj>, Box<Obj>),
    Build(Box<Obj>, Box<Obj>),
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.b[self.p];
        self.p += 1;
        v
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn line(&mut self) -> String {
        let start = self.p;
        while self.b[self.p] != b'\n' {
            self.p += 1;
        }
        let s = String::from_utf8_lossy(&self.b[start..self.p]).to_string();
        self.p += 1;
        s
    }
    fn u16(&mut self) -> u16 {
        let v = u16le(self.b, self.p);
        self.p += 2;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32le(self.b, self.p);
        self.p += 4;
        v
    }
    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }
}

fn run_pickle(buf: &[u8]) -> Result<Obj> {
    let mut r = Reader { b: buf, p: 0 };
    let mut stack: Vec<Obj> = Vec::with_capacity(256);
    let mut memo: HashMap<u32, Obj> = HashMap::new();
    let err = |m: &str| Error(format!("pth pickle: {m}"));
    loop {
        let op = r.u8();
        match op {
            0x80 => {
                r.u8();
            } // PROTO
            0x95 => {
                r.take(8);
            } // FRAME (8-byte len)
            b'c' => {
                let m = r.line();
                let c = r.line();
                stack.push(Obj::Class(m, c));
            } // GLOBAL
            0x93 => {
                // STACK_GLOBAL
                let c = stack.pop().ok_or_else(|| err("stack_global"))?;
                let m = stack.pop().ok_or_else(|| err("stack_global"))?;
                if let (Obj::Str(m), Obj::Str(c)) = (m, c) {
                    stack.push(Obj::Class(m, c));
                } else {
                    return Err(err("stack_global types"));
                }
            }
            b'q' => {
                let i = r.u8() as u32;
                memo.insert(i, stack.last().cloned().ok_or_else(|| err("binput"))?);
            } // BINPUT
            b'r' => {
                let i = r.u32();
                memo.insert(i, stack.last().cloned().ok_or_else(|| err("lbinput"))?);
            } // LONG_BINPUT
            0x94 => {
                memo.insert(
                    memo.len() as u32,
                    stack.last().cloned().ok_or_else(|| err("memoize"))?,
                );
            } // MEMOIZE
            b'h' => {
                let i = r.u8() as u32;
                stack.push(memo.get(&i).cloned().ok_or_else(|| err("binget"))?);
            } // BINGET
            b'j' => {
                let i = r.u32();
                stack.push(memo.get(&i).cloned().ok_or_else(|| err("lbinget"))?);
            } // LONG_BINGET
            b'K' => {
                let v = r.u8();
                stack.push(Obj::Int(v as i64));
            } // BININT1
            b'M' => {
                let v = r.u16();
                stack.push(Obj::Int(v as i64));
            } // BININT2
            b'J' => {
                let v = r.i32();
                stack.push(Obj::Int(v as i64));
            } // BININT
            0x8a => {
                let n = r.u8() as usize;
                let mut v: i64 = 0;
                for i in 0..n {
                    v |= (r.u8() as i64) << (i * 8);
                }
                stack.push(Obj::Int(v));
            } // LONG1
            b'G' => {
                let bytes = r.take(8);
                stack.push(Obj::Float(f64::from_be_bytes(bytes.try_into().unwrap())));
            } // BINFLOAT
            b'X' => {
                let n = r.u32() as usize;
                let s = String::from_utf8_lossy(r.take(n)).to_string();
                stack.push(Obj::Str(s));
            } // BINUNICODE
            0x8c => {
                let n = r.u8() as usize;
                let s = String::from_utf8_lossy(r.take(n)).to_string();
                stack.push(Obj::Str(s));
            } // SHORT_BINUNICODE
            b'N' => stack.push(Obj::None),
            0x88 => stack.push(Obj::Bool(true)),
            0x89 => stack.push(Obj::Bool(false)),
            b'(' => stack.push(Obj::Mark),
            b')' => stack.push(Obj::Tuple(vec![])), // EMPTY_TUPLE
            0x85 => {
                let a = stack.pop().ok_or_else(|| err("t1"))?;
                stack.push(Obj::Tuple(vec![a]));
            } // TUPLE1
            0x86 => {
                let b = stack.pop().ok_or_else(|| err("t2"))?;
                let a = stack.pop().ok_or_else(|| err("t2"))?;
                stack.push(Obj::Tuple(vec![a, b]));
            } // TUPLE2
            0x87 => {
                let c = stack.pop().ok_or_else(|| err("t3"))?;
                let b = stack.pop().ok_or_else(|| err("t3"))?;
                let a = stack.pop().ok_or_else(|| err("t3"))?;
                stack.push(Obj::Tuple(vec![a, b, c]));
            } // TUPLE3
            b't' => {
                let v = pop_mark(&mut stack)?;
                stack.push(Obj::Tuple(v));
            } // TUPLE
            b']' => stack.push(Obj::List(vec![])),  // EMPTY_LIST
            b'e' => {
                let items = pop_mark(&mut stack)?;
                if let Some(Obj::List(l)) = stack.last_mut() {
                    l.extend(items);
                }
            } // APPENDS
            b'a' => {
                let it = stack.pop().ok_or_else(|| err("append"))?;
                if let Some(Obj::List(l)) = stack.last_mut() {
                    l.push(it);
                }
            } // APPEND
            b'}' => stack.push(Obj::Dict(vec![])),  // EMPTY_DICT
            b's' => {
                let v = stack.pop().ok_or_else(|| err("si"))?;
                let k = stack.pop().ok_or_else(|| err("si"))?;
                if let Some(Obj::Dict(d)) = stack.last_mut() {
                    d.push((k, v));
                }
            } // SETITEM
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if let Some(Obj::Dict(d)) = stack.last_mut() {
                    for ch in items.chunks(2) {
                        d.push((ch[0].clone(), ch[1].clone()));
                    }
                }
            } // SETITEMS
            b'Q' => {
                let id = stack.pop().ok_or_else(|| err("persid"))?;
                stack.push(Obj::PersId(Box::new(id)));
            } // BINPERSID
            b'R' => {
                let args = stack.pop().ok_or_else(|| err("reduce"))?;
                let call = stack.pop().ok_or_else(|| err("reduce"))?;
                stack.push(reduce(call, args));
            } // REDUCE
            0x81 => {
                let args = stack.pop().ok_or_else(|| err("newobj"))?;
                let call = stack.pop().ok_or_else(|| err("newobj"))?;
                stack.push(Obj::Reduce(Box::new(call), Box::new(args)));
            } // NEWOBJ
            b'b' => {
                let args = stack.pop().ok_or_else(|| err("build"))?;
                let obj = stack.pop().ok_or_else(|| err("build"))?;
                stack.push(build(obj, args));
            } // BUILD
            b'.' => break,                          // STOP
            other => return Err(err(&format!("unsupported opcode 0x{other:02x}"))),
        }
    }
    stack.pop().ok_or_else(|| err("empty stack at STOP"))
}

fn pop_mark(stack: &mut Vec<Obj>) -> Result<Vec<Obj>> {
    let idx = stack
        .iter()
        .rposition(|o| matches!(o, Obj::Mark))
        .ok_or_else(|| Error("pth pickle: no mark".into()))?;
    let v = stack.split_off(idx + 1);
    stack.pop(); // the mark
    Ok(v)
}

fn reduce(call: Obj, args: Obj) -> Obj {
    if let Obj::Class(m, c) = &call {
        if m == "collections" && (c == "OrderedDict" || c == "defaultdict") {
            return Obj::Dict(vec![]);
        }
    }
    Obj::Reduce(Box::new(call), Box::new(args))
}

fn build(obj: Obj, args: Obj) -> Obj {
    match (obj, args) {
        (Obj::Dict(mut d), Obj::Dict(mut a)) => {
            d.append(&mut a);
            Obj::Dict(d)
        }
        (obj, args) => Obj::Build(Box::new(obj), Box::new(args)),
    }
}

// torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...) -> (key, dtype, shape)
struct TInfo {
    storage_key: String,
    dtype: DType,
    shape: Vec<usize>,
    offset: usize,
}

fn tensor_info(name: &str, value: Obj) -> Option<TInfo> {
    // unwrap _rebuild_parameter / _rebuild_from_type_v2 wrappers, then _rebuild_tensor_v2
    let (call, args) = match value {
        Obj::Reduce(c, a) => (*c, *a),
        _ => return None,
    };
    let (call, args) = match call {
        Obj::Class(m, c) if m == "torch._utils" && c == "_rebuild_parameter" => {
            if let Obj::Tuple(mut t) = args {
                if let Obj::Reduce(c2, a2) = t.remove(0) {
                    (*c2, *a2)
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        c => (c, args),
    };
    match call {
        Obj::Class(m, c) if m == "torch._utils" && c == "_rebuild_tensor_v2" => {}
        _ => return None,
    }
    let mut t = if let Obj::Tuple(t) = args {
        t
    } else {
        return None;
    };
    // args: storage, storage_offset, size, stride, ...
    let size = as_usize_vec(&t.get(2).cloned()?)?;
    // storage_offset is in ELEMENTS into the (possibly shared) storage; tensors saved as
    // views share one storage at different offsets, so this must be honored.
    let offset = match t.get(1) {
        Some(Obj::Int(i)) => *i as usize,
        _ => 0,
    };
    let storage = match t.remove(0) {
        Obj::PersId(p) => *p,
        _ => return None,
    };
    // storage persid tuple: ('storage', StorageClass, key, location, numel)
    let st = if let Obj::Tuple(st) = storage {
        st
    } else {
        return None;
    };
    let dtype = match st.get(1)? {
        Obj::Class(_, c) => match c.as_str() {
            "FloatStorage" => DType::F32,
            "DoubleStorage" => DType::F64,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "ByteStorage" => DType::U8,
            "LongStorage" => DType::I64,
            _ => return None,
        },
        _ => return None,
    };
    let key = match st.get(2)? {
        Obj::Str(s) => s.clone(),
        _ => return None,
    };
    let _ = name;
    Some(TInfo {
        storage_key: key,
        dtype,
        shape: size,
        offset,
    })
}

fn as_usize_vec(o: &Obj) -> Option<Vec<usize>> {
    if let Obj::Tuple(t) = o {
        t.iter()
            .map(|x| {
                if let Obj::Int(i) = x {
                    Some(*i as usize)
                } else {
                    None
                }
            })
            .collect()
    } else {
        None
    }
}

fn bytes_to_storage(data: &[u8], dt: DType) -> Result<CpuStorage> {
    // A scalar map over every element is the dominant CPU cost of loading a big checkpoint
    // - Wan's 11 GB bf16 umT5-XXL is 5.5 B elements, tens of seconds on one thread - so it
    // runs in parallel. rayon's indexed `collect` preserves order, and the result is
    // bit-identical to the sequential form.
    use rayon::prelude::*;
    Ok(match dt {
        DType::F32 => CpuStorage::F32(
            data.par_chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        DType::F16 => CpuStorage::F16(
            data.par_chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
        DType::BF16 => CpuStorage::BF16(
            data.par_chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
        DType::F64 => CpuStorage::F64(
            data.par_chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::U8 => CpuStorage::U8(data.to_vec()),
        DType::I64 => CpuStorage::I64(
            data.par_chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        other => return Err(Error(format!("pth: unsupported storage dtype {other:?}"))),
    })
}

/// Read a torch `.pt`/`.pth` checkpoint to a name -> host native `Tensor` map.
/// Honors each tensor's `storage_offset` (tensors saved as views share one storage);
/// assumes contiguous (row-major) layout from that offset.
pub fn read_pt(path: &str) -> Result<HashMap<String, Tensor>> {
    let zip = Zip::open(path)?;
    let pkl_name = zip
        .find_suffix("data.pkl")
        .ok_or_else(|| Error("pth: no data.pkl".into()))?;
    let root = pkl_name.strip_suffix("data.pkl").unwrap_or("").to_string(); // e.g. "1m_clean/"
    let obj = run_pickle(
        zip.read(&pkl_name)
            .ok_or_else(|| Error("pth: data.pkl unreadable".into()))?,
    )?;
    // Top-level may be the state_dict OrderedDict, or {'state_dict': {...}} / {'model': {...}}.
    let dict = unwrap_state_dict(obj).ok_or_else(|| Error("pth: no state_dict dict".into()))?;
    let mut out = HashMap::new();
    for (k, v) in dict {
        let name = if let Obj::Str(s) = k { s } else { continue };
        if let Some(ti) = tensor_info(&name, v) {
            let raw = zip
                .read(&format!("{root}data/{}", ti.storage_key))
                .ok_or_else(|| Error(format!("pth: storage {} missing", ti.storage_key)))?;
            let nelem: usize = ti.shape.iter().product();
            let esz = ti.dtype.size_in_bytes();
            // honor storage_offset: tensors saved as views share one storage at byte
            // `offset.elem_size`; reading from 0 would corrupt every non-first view.
            let start = ti.offset * esz;
            let end = (start + nelem * esz).min(raw.len());
            let storage = bytes_to_storage(&raw[start.min(raw.len())..end], ti.dtype)?;
            out.insert(name, Tensor::from_storage(storage, Shape::from(ti.shape))?);
        }
    }
    Ok(out)
}

fn unwrap_state_dict(obj: Obj) -> Option<Vec<(Obj, Obj)>> {
    match obj {
        Obj::Dict(d) => {
            // direct state_dict (keys are tensor names) vs a wrapper {'state_dict'/'model': {...}}
            if d.iter()
                .any(|(k, _)| matches!(k, Obj::Str(s) if s == "state_dict" || s == "model"))
            {
                for (k, v) in d {
                    if let Obj::Str(s) = &k {
                        if s == "state_dict" || s == "model" {
                            if let Obj::Dict(inner) = v {
                                return Some(inner);
                            }
                        }
                    }
                }
                None
            } else {
                Some(d)
            }
        }
        Obj::Build(o, _) => unwrap_state_dict(*o),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Loads the EzAudio VAE .pt and reports the inventory. Run:
    //   cargo test --release -p server pth::tests::probe_ezaudio_vae -- --ignored --nocapture
    #[test]
    #[ignore = "needs the EzAudio VAE .pt under the HF cache"]
    fn probe_ezaudio_vae() {
        let p = format!(
            "{}/.cache/huggingface/hub/models--OpenSound--EzAudio/snapshots/main/ckpts/vae/1m.pt",
            std::env::var("HOME").unwrap()
        );
        let t = read_pt(&p).expect("read .pt");
        println!("tensors: {}", t.len());
        let mut names: Vec<_> = t.keys().cloned().collect();
        names.sort();
        for n in names.iter().take(30) {
            println!("  {n}  {:?} {:?}", t[n].dims(), t[n].dtype());
        }
    }
}
