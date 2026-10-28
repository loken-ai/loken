//! `torch.save` of one tensor, read without unpickling: the zip archive torch writes, and the
//! pickle opcodes that rebuild a plain tensor over one storage of a float dtype. Any other global,
//! opcode, persistent reference, argument or shape is refused; nothing in the payload is run.

use std::collections::HashMap;
use std::rc::Rc;

use super::{Dtype, Layout, Packed};

/// Entries an archive may hold; torch writes about seven for one tensor.
const MAX_ENTRIES: usize = 64;
/// Bytes `data.pkl` may hold; the pickle of one tensor takes a few hundred.
const MAX_PICKLE_BYTES: usize = 64 * 1024;
/// Values the pickle machine's stack may hold at once.
const MAX_STACK: usize = 256;
/// Memo slots the pickle machine may fill.
const MAX_MEMO: usize = 256;
/// Bytes of a global's module or name, of a storage key, or of a string argument.
const MAX_NAME: usize = 64;
/// Bytes of the widest integer a `LONG1` may carry.
const MAX_LONG_BYTES: usize = 8;

const MALFORMED: &str = "`prompt_embeds` is a truncated or malformed torch.save archive";

// Zip records, as torch's writer lays them out.
const LOCAL_SIG: u32 = 0x0403_4b50;
const LOCAL_BYTES: usize = 30;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const CENTRAL_BYTES: usize = 46;
const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD_BYTES: usize = 22;
const ZIP64_LOCATOR_SIG: u32 = 0x0706_4b50;
const ZIP64_LOCATOR_BYTES: usize = 20;
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
const ZIP64_EXTRA_ID: u16 = 0x0001;
const FLAG_ENCRYPTED: u16 = 1;
const METHOD_STORED: u16 = 0;
/// The value a 32-bit zip field holds when its real value sits in the zip64 extra field.
const ZIP64_U32: u32 = u32::MAX;

// The pickle opcodes a tensor's pickle uses, protocols 2 to 5.
const PROTO: u8 = 0x80;
const FRAME: u8 = 0x95;
const GLOBAL: u8 = b'c';
const STACK_GLOBAL: u8 = 0x93;
const BINPUT: u8 = b'q';
const LONG_BINPUT: u8 = b'r';
const MEMOIZE: u8 = 0x94;
const BINGET: u8 = b'h';
const LONG_BINGET: u8 = b'j';
const MARK: u8 = b'(';
const EMPTY_TUPLE: u8 = b')';
const TUPLE: u8 = b't';
const TUPLE1: u8 = 0x85;
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;
const BINUNICODE: u8 = b'X';
const SHORT_BINUNICODE: u8 = 0x8c;
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const BININT: u8 = b'J';
const LONG1: u8 = 0x8a;
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;
const BINPERSID: u8 = b'Q';
const REDUCE: u8 = b'R';
const STOP: u8 = b'.';
/// The pickle protocols torch may write.
const PROTOCOLS: std::ops::RangeInclusive<u8> = 2..=5;

fn le<const N: usize>(b: &[u8], at: usize) -> Option<[u8; N]> {
    b.get(at..at.checked_add(N)?)?.try_into().ok()
}
fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    le(b, at).map(u16::from_le_bytes)
}
fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    le(b, at).map(u32::from_le_bytes)
}
fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    le(b, at).map(u64::from_le_bytes)
}
fn index(v: u64) -> Option<usize> {
    usize::try_from(v).ok()
}

/// One stored entry of the archive: its name, CRC-32, length and where its bytes start.
#[derive(Debug)]
struct Entry {
    name: String,
    crc: u32,
    size: usize,
    data: usize,
}

/// The archive's central directory: entry count, offset and size, from the end record or the
/// zip64 record it points to. The end record must close the payload exactly.
fn central_directory(b: &[u8]) -> Result<(usize, usize, usize), String> {
    let last = b.len().checked_sub(EOCD_BYTES).ok_or(MALFORMED)?;
    let first = last.saturating_sub(usize::from(u16::MAX));
    let eocd = (first..=last)
        .rev()
        .find(|&at| {
            u32_at(b, at) == Some(EOCD_SIG)
                && u16_at(b, at + 20).map(|c| at + EOCD_BYTES + usize::from(c)) == Some(b.len())
        })
        .ok_or("`prompt_embeds` is not a whole zip archive: no end of central directory")?;
    if u16_at(b, eocd + 4) != Some(0) || u16_at(b, eocd + 6) != Some(0) {
        return Err("`prompt_embeds` is a zip archive split over several disks".into());
    }
    let locator = eocd
        .checked_sub(ZIP64_LOCATOR_BYTES)
        .filter(|&at| u32_at(b, at) == Some(ZIP64_LOCATOR_SIG));
    let (count, offset, size) = match locator {
        Some(at) => {
            let rec = u64_at(b, at + 8).and_then(index).ok_or(MALFORMED)?;
            if u32_at(b, rec) != Some(ZIP64_EOCD_SIG)
                || u32_at(b, rec + 16) != Some(0)
                || u32_at(b, rec + 20) != Some(0)
            {
                return Err(MALFORMED.into());
            }
            (
                u64_at(b, rec + 32).ok_or(MALFORMED)?,
                u64_at(b, rec + 48).ok_or(MALFORMED)?,
                u64_at(b, rec + 40).ok_or(MALFORMED)?,
            )
        }
        None => (
            u64::from(u16_at(b, eocd + 10).ok_or(MALFORMED)?),
            u64::from(u32_at(b, eocd + 16).ok_or(MALFORMED)?),
            u64::from(u32_at(b, eocd + 12).ok_or(MALFORMED)?),
        ),
    };
    let count = index(count).filter(|&c| c <= MAX_ENTRIES).ok_or_else(|| {
        format!(
            "`prompt_embeds` archive holds {count} entries; a tensor takes at most {MAX_ENTRIES}"
        )
    })?;
    let offset = index(offset).ok_or(MALFORMED)?;
    let size = index(size).ok_or(MALFORMED)?;
    match offset.checked_add(size) {
        Some(end) if end <= eocd => Ok((count, offset, size)),
        _ => Err(MALFORMED.into()),
    }
}

/// The 64-bit values of a zip64 extra field for the 32-bit fields that hold the marker, in the
/// field's order: uncompressed size, compressed size, local header offset.
fn zip64_values(extra: &[u8], wanted: [bool; 3]) -> Option<[Option<u64>; 3]> {
    let mut at = 0;
    while at < extra.len() {
        let id = u16_at(extra, at)?;
        let len = usize::from(u16_at(extra, at + 2)?);
        let body = extra.get(at + 4..at + 4 + len)?;
        if id == ZIP64_EXTRA_ID {
            let mut out = [None; 3];
            let mut read = 0;
            for (slot, want) in out.iter_mut().zip(wanted) {
                if want {
                    *slot = Some(u64_at(body, read)?);
                    read += 8;
                }
            }
            return Some(out);
        }
        at += 4 + len;
    }
    None
}

/// Every entry of the archive, each stored uncompressed, unencrypted, named once, its local
/// header matching the central one and its bytes lying before the central directory.
fn entries(b: &[u8]) -> Result<Vec<Entry>, String> {
    let (count, cd, cd_size) = central_directory(b)?;
    let cd_end = cd + cd_size;
    let mut out: Vec<Entry> = Vec::with_capacity(count);
    let mut at = cd;
    for _ in 0..count {
        let field = |off: usize| u16_at(b, at + off).ok_or(MALFORMED);
        let word = |off: usize| u32_at(b, at + off).ok_or(MALFORMED);
        if word(0)? != CENTRAL_SIG {
            return Err(MALFORMED.into());
        }
        let (flags, method) = (field(8)?, field(10)?);
        let crc = word(16)?;
        let (comp, uncomp, local) = (word(20)?, word(24)?, word(42)?);
        let (n, e, k) = (
            usize::from(field(28)?),
            usize::from(field(30)?),
            usize::from(field(32)?),
        );
        let disk = field(34)?;
        let name_at = at + CENTRAL_BYTES;
        let next = name_at + n + e + k;
        if next > cd_end {
            return Err(MALFORMED.into());
        }
        let name = &b[name_at..name_at + n];
        let extra = &b[name_at + n..name_at + n + e];
        let wanted = [uncomp == ZIP64_U32, comp == ZIP64_U32, local == ZIP64_U32];
        let [big_uncomp, big_comp, big_local] = if wanted.contains(&true) {
            zip64_values(extra, wanted).ok_or(MALFORMED)?
        } else {
            [None; 3]
        };
        let uncomp = big_uncomp.unwrap_or(u64::from(uncomp));
        let comp = big_comp.unwrap_or(u64::from(comp));
        let local = index(big_local.unwrap_or(u64::from(local))).ok_or(MALFORMED)?;
        if flags & FLAG_ENCRYPTED != 0 {
            return Err("`prompt_embeds` archive holds an encrypted entry".into());
        }
        if method != METHOD_STORED {
            return Err(
                "`prompt_embeds` archive holds a compressed entry; torch.save stores \
                        its entries uncompressed"
                    .into(),
            );
        }
        if comp != uncomp || disk != 0 {
            return Err(MALFORMED.into());
        }
        let name = std::str::from_utf8(name)
            .map_err(|_| MALFORMED)?
            .to_string();
        if out.iter().any(|x| x.name == name) {
            return Err(format!(
                "`prompt_embeds` archive names {:?} twice",
                truncated(&name)
            ));
        }
        if u32_at(b, local) != Some(LOCAL_SIG) {
            return Err(MALFORMED.into());
        }
        let local_name = usize::from(u16_at(b, local + 26).ok_or(MALFORMED)?);
        let local_extra = usize::from(u16_at(b, local + 28).ok_or(MALFORMED)?);
        let local_name_at = local + LOCAL_BYTES;
        if b.get(local_name_at..local_name_at + local_name) != Some(name.as_bytes()) {
            return Err(MALFORMED.into());
        }
        let data = local_name_at + local_name + local_extra;
        let size = index(uncomp).ok_or(MALFORMED)?;
        match data.checked_add(size) {
            Some(end) if end <= cd => {}
            _ => return Err(MALFORMED.into()),
        }
        out.push(Entry {
            name,
            crc,
            size,
            data,
        });
        at = next;
    }
    Ok(out)
}

/// An entry's bytes, refused when they fail the CRC-32 the archive records for them.
fn contents<'a>(b: &'a [u8], e: &Entry) -> Result<&'a [u8], String> {
    let bytes = &b[e.data..e.data + e.size];
    if super::crc32(bytes) != e.crc {
        return Err(format!(
            "`prompt_embeds` archive entry {:?} fails its checksum: the payload is corrupted",
            truncated(&e.name)
        ));
    }
    Ok(bytes)
}

/// At most `MAX_NAME` bytes of a string the payload carries, for an error message.
fn truncated(s: &str) -> &str {
    let mut end = s.len().min(MAX_NAME);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The tensor `torch.save` wrote in `bytes`, as a view over its storage entry: a shape of
/// (tokens, width) or (1, tokens, width), any strides and offset that stay inside the storage.
pub(super) fn read(bytes: Vec<u8>) -> Result<Packed, String> {
    const PICKLE: &str = "data.pkl";
    let entries = entries(&bytes)?;
    let pickles: Vec<&Entry> = entries
        .iter()
        .filter(|e| e.name == PICKLE || e.name.ends_with(&format!("/{PICKLE}")))
        .collect();
    let [pickle] = pickles.as_slice() else {
        return Err(format!(
            "`prompt_embeds` archive holds {} `{PICKLE}` entries, not one",
            pickles.len()
        ));
    };
    let prefix = &pickle.name[..pickle.name.len() - PICKLE.len()];
    if let Some(order) = entries
        .iter()
        .find(|e| e.name.strip_prefix(prefix) == Some("byteorder"))
    {
        if contents(&bytes, order)? != b"little" {
            return Err(
                "`prompt_embeds` tensor was saved big-endian; save it little-endian".into(),
            );
        }
    }
    if pickle.size > MAX_PICKLE_BYTES {
        return Err(format!(
            "`prompt_embeds` pickle is {} bytes; one tensor's takes under {MAX_PICKLE_BYTES}",
            pickle.size
        ));
    }
    let tensor = unpickle(contents(&bytes, pickle)?)?;
    let storage = &tensor.storage;
    let storage_name = format!("{prefix}data/{}", storage.key);
    let entry = entries
        .iter()
        .find(|e| e.name == storage_name)
        .ok_or_else(|| {
            format!("`prompt_embeds` archive lacks the storage entry {storage_name:?}")
        })?;
    if storage.numel.checked_mul(storage.dtype.size()) != Some(entry.size) {
        return Err(format!(
            "`prompt_embeds` storage holds {} bytes, not the {} {} values its reference declares",
            entry.size,
            storage.numel,
            storage.dtype.name()
        ));
    }
    contents(&bytes, entry)?;
    let ((rows, width), (row_stride, col_stride)) =
        match (tensor.size.as_slice(), tensor.stride.as_slice()) {
            (&[r, w], &[rs, cs]) | (&[1, r, w], &[_, rs, cs]) => ((r, w), (rs, cs)),
            (size, _) => {
                return Err(format!(
                    "`prompt_embeds` must be a tensor of shape (tokens, hidden_size) or \
                     (1, tokens, hidden_size); got {size:?}"
                ))
            }
        };
    let (start, end) = (entry.data, entry.data + entry.size);
    Packed::new(
        bytes,
        start,
        end,
        Layout {
            dtype: storage.dtype,
            rows,
            width,
            offset: tensor.offset,
            row_stride,
            col_stride,
        },
    )
}

/// A global the restricted unpickler resolves.
#[derive(Clone, Copy, Debug)]
enum Global {
    RebuildTensorV2,
    Storage(Dtype),
    OrderedDict,
}

/// The storage a persistent reference names: its dtype, its key in the archive and its length
/// in elements.
#[derive(Debug)]
struct Storage {
    dtype: Dtype,
    key: String,
    numel: usize,
}

/// A tensor `_rebuild_tensor_v2` would build.
#[derive(Debug)]
struct Tensor {
    storage: Rc<Storage>,
    offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

/// A value on the unpickler's stack. Clones share their contents, so a memo read costs the
/// same whatever it holds.
#[derive(Clone, Debug)]
enum Value {
    Mark,
    Global(Global),
    Str(Rc<str>),
    Int(i64),
    /// A boolean; the rebuilt tensor's `requires_grad`, which the reader does not use.
    Bool,
    Tuple(Rc<[Value]>),
    Storage(Rc<Storage>),
    OrderedDict,
    Tensor(Rc<Tensor>),
}

/// The global `module.name` stands for, refused unless it is one a tensor's pickle names.
fn global(module: &str, name: &str) -> Result<Global, String> {
    match (module, name) {
        ("torch._utils", "_rebuild_tensor_v2") => Ok(Global::RebuildTensorV2),
        ("collections", "OrderedDict") => Ok(Global::OrderedDict),
        ("torch", "FloatStorage") => Ok(Global::Storage(Dtype::F32)),
        ("torch", "HalfStorage") => Ok(Global::Storage(Dtype::F16)),
        ("torch", "BFloat16Storage") => Ok(Global::Storage(Dtype::Bf16)),
        ("torch", "DoubleStorage") => Ok(Global::Storage(Dtype::F64)),
        ("torch", storage) if storage.ends_with("Storage") => Err(format!(
            "`prompt_embeds` tensor is a {:?}; the reader takes float32, float16, bfloat16 and \
             float64",
            truncated(storage)
        )),
        _ => Err(format!(
            "`prompt_embeds` pickle names `{}.{}`, which is not part of a tensor's pickle",
            truncated(module),
            truncated(name)
        )),
    }
}

/// The non-negative integer `v` holds.
fn count(v: &Value) -> Option<usize> {
    match v {
        Value::Int(i) => usize::try_from(*i).ok(),
        _ => None,
    }
}

/// The non-negative integers of a tuple.
fn counts(v: &Value) -> Option<Vec<usize>> {
    match v {
        Value::Tuple(items) => items.iter().map(count).collect(),
        _ => None,
    }
}

/// The storage a persistent reference `('storage', <dtype>Storage, key, location, numel)` names.
fn storage_of(pid: &Value) -> Result<Value, String> {
    const BAD: &str = "`prompt_embeds` pickle holds a persistent reference that is not a storage";
    let Value::Tuple(items) = pid else {
        return Err(BAD.into());
    };
    match &items[..] {
        [Value::Str(tag), Value::Global(Global::Storage(dtype)), Value::Str(key), Value::Str(_), numel]
            if &**tag == "storage"
                && !key.is_empty()
                && key.bytes().all(|c| c.is_ascii_alphanumeric()) =>
        {
            Ok(Value::Storage(Rc::new(Storage {
                dtype: *dtype,
                key: key.to_string(),
                numel: count(numel).ok_or(BAD)?,
            })))
        }
        _ => Err(BAD.into()),
    }
}

/// What calling `f` on `args` builds: an empty `OrderedDict`, or a tensor from
/// `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad, backward_hooks)` with no
/// hooks. Any other call is refused.
fn call(f: &Value, args: &Value) -> Result<Value, String> {
    const BAD_ARGS: &str =
        "`prompt_embeds` pickle rebuilds a tensor with arguments a plain tensor does not have";
    let Value::Tuple(args) = args else {
        return Err(BAD_ARGS.into());
    };
    match f {
        Value::Global(Global::OrderedDict) if args.is_empty() => Ok(Value::OrderedDict),
        Value::Global(Global::RebuildTensorV2) => match &args[..] {
            [Value::Storage(storage), offset, size, stride, Value::Bool, Value::OrderedDict] => {
                let offset = count(offset).ok_or(BAD_ARGS)?;
                let size = counts(size).ok_or(BAD_ARGS)?;
                let stride = counts(stride).ok_or(BAD_ARGS)?;
                if size.len() != stride.len() {
                    return Err(BAD_ARGS.into());
                }
                Ok(Value::Tensor(Rc::new(Tensor {
                    storage: storage.clone(),
                    offset,
                    size,
                    stride,
                })))
            }
            _ => Err(BAD_ARGS.into()),
        },
        _ => Err(
            "`prompt_embeds` pickle calls something other than a tensor's rebuild function".into(),
        ),
    }
}

/// The stack machine of the restricted unpickler over the bytes of `data.pkl`.
struct Machine<'a> {
    p: &'a [u8],
    at: usize,
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
}

impl<'a> Machine<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.p.len());
        let end = end.ok_or("`prompt_embeds` pickle is truncated")?;
        let out = &self.p[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// A string of `n` bytes, refused past `MAX_NAME` or when it is not UTF-8.
    fn text(&mut self, n: usize) -> Result<&'a str, String> {
        if n > MAX_NAME {
            return Err(format!(
                "`prompt_embeds` pickle holds a string of {n} bytes; a tensor's are under \
                 {MAX_NAME}"
            ));
        }
        std::str::from_utf8(self.take(n)?)
            .map_err(|_| "`prompt_embeds` pickle holds a string that is not UTF-8".into())
    }

    /// The text up to the next newline, for `GLOBAL`.
    fn line(&mut self) -> Result<&'a str, String> {
        let rest = &self.p[self.at..];
        let n = rest
            .iter()
            .take(MAX_NAME + 1)
            .position(|&c| c == b'\n')
            .ok_or("`prompt_embeds` pickle names a global past its length limit or truncated")?;
        let s = self.text(n)?;
        self.at += 1;
        Ok(s)
    }

    fn push(&mut self, v: Value) -> Result<(), String> {
        if self.stack.len() >= MAX_STACK {
            return Err("`prompt_embeds` pickle overflows the unpickler's stack".into());
        }
        self.stack.push(v);
        Ok(())
    }

    /// The top value, never a mark.
    fn pop(&mut self) -> Result<Value, String> {
        match self.stack.pop() {
            Some(Value::Mark) | None => Err("`prompt_embeds` pickle pops an empty stack".into()),
            Some(v) => Ok(v),
        }
    }

    fn pop_str(&mut self) -> Result<Rc<str>, String> {
        match self.pop()? {
            Value::Str(s) => Ok(s),
            _ => Err("`prompt_embeds` pickle names a global with a value that is not text".into()),
        }
    }

    fn pop_n(&mut self, n: usize) -> Result<Vec<Value>, String> {
        let mut out = (0..n).map(|_| self.pop()).collect::<Result<Vec<_>, _>>()?;
        out.reverse();
        Ok(out)
    }

    fn pop_to_mark(&mut self) -> Result<Vec<Value>, String> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or("`prompt_embeds` pickle builds a tuple without a mark")?;
        let out = self.stack.split_off(mark + 1);
        self.stack.pop();
        Ok(out)
    }

    fn memo_put(&mut self, slot: u32) -> Result<(), String> {
        let top = match self.stack.last() {
            Some(Value::Mark) | None => {
                return Err("`prompt_embeds` pickle memoizes nothing".into());
            }
            Some(v) => v.clone(),
        };
        if self.memo.len() >= MAX_MEMO && !self.memo.contains_key(&slot) {
            return Err("`prompt_embeds` pickle fills more memo slots than a tensor's".into());
        }
        self.memo.insert(slot, top);
        Ok(())
    }

    fn memo_get(&mut self, slot: u32) -> Result<(), String> {
        let v = self
            .memo
            .get(&slot)
            .cloned()
            .ok_or("`prompt_embeds` pickle reads a memo slot it never filled")?;
        self.push(v)
    }

    fn tuple(&mut self, items: Vec<Value>) -> Result<(), String> {
        self.push(Value::Tuple(items.into()))
    }
}

/// The tensor the pickle `p` rebuilds, read by a machine that knows only the opcodes, globals
/// and calls of a plain tensor's pickle. The pickle must end at its `STOP` with that tensor alone
/// on the stack.
fn unpickle(p: &[u8]) -> Result<Rc<Tensor>, String> {
    let mut m = Machine {
        p,
        at: 0,
        stack: Vec::new(),
        memo: HashMap::new(),
    };
    loop {
        match m.byte()? {
            PROTO => {
                let v = m.byte()?;
                if !PROTOCOLS.contains(&v) {
                    return Err(format!("`prompt_embeds` pickle is protocol {v}"));
                }
            }
            FRAME => {
                m.take(8)?;
            }
            GLOBAL => {
                let module = m.line()?;
                let name = m.line()?;
                let g = global(module, name)?;
                m.push(Value::Global(g))?;
            }
            STACK_GLOBAL => {
                let name = m.pop_str()?;
                let module = m.pop_str()?;
                let g = global(&module, &name)?;
                m.push(Value::Global(g))?;
            }
            BINPUT => {
                let slot = m.byte()?;
                m.memo_put(u32::from(slot))?;
            }
            LONG_BINPUT => {
                let slot = m.u32()?;
                m.memo_put(slot)?;
            }
            MEMOIZE => {
                let slot = u32::try_from(m.memo.len()).map_err(|_| MALFORMED)?;
                m.memo_put(slot)?;
            }
            BINGET => {
                let slot = m.byte()?;
                m.memo_get(u32::from(slot))?;
            }
            LONG_BINGET => {
                let slot = m.u32()?;
                m.memo_get(slot)?;
            }
            MARK => m.push(Value::Mark)?,
            EMPTY_TUPLE => m.tuple(Vec::new())?,
            TUPLE => {
                let items = m.pop_to_mark()?;
                m.tuple(items)?;
            }
            op @ (TUPLE1 | TUPLE2 | TUPLE3) => {
                let items = m.pop_n(usize::from(op - TUPLE1) + 1)?;
                m.tuple(items)?;
            }
            BINUNICODE => {
                let n = index(u64::from(m.u32()?)).ok_or(MALFORMED)?;
                let s = m.text(n)?;
                m.push(Value::Str(s.into()))?;
            }
            SHORT_BINUNICODE => {
                let n = usize::from(m.byte()?);
                let s = m.text(n)?;
                m.push(Value::Str(s.into()))?;
            }
            BININT1 => {
                let v = m.byte()?;
                m.push(Value::Int(i64::from(v)))?;
            }
            BININT2 => {
                let v = m.u16()?;
                m.push(Value::Int(i64::from(v)))?;
            }
            BININT => {
                let v = m.u32()? as i32;
                m.push(Value::Int(i64::from(v)))?;
            }
            LONG1 => {
                let n = usize::from(m.byte()?);
                if n > MAX_LONG_BYTES {
                    return Err(format!(
                        "`prompt_embeds` pickle holds an integer of {n} bytes; the widest a \
                         tensor needs is {MAX_LONG_BYTES}"
                    ));
                }
                let b = m.take(n)?;
                let fill = if b.last().is_some_and(|&x| x & 0x80 != 0) {
                    0xff
                } else {
                    0
                };
                let mut wide = [fill; MAX_LONG_BYTES];
                wide[..n].copy_from_slice(b);
                m.push(Value::Int(i64::from_le_bytes(wide)))?;
            }
            NEWTRUE => m.push(Value::Bool)?,
            NEWFALSE => m.push(Value::Bool)?,
            BINPERSID => {
                let pid = m.pop()?;
                let storage = storage_of(&pid)?;
                m.push(storage)?;
            }
            REDUCE => {
                let args = m.pop()?;
                let f = m.pop()?;
                let built = call(&f, &args)?;
                m.push(built)?;
            }
            STOP => {
                if m.at != p.len() {
                    return Err("`prompt_embeds` pickle has bytes after its end".into());
                }
                return match (m.stack.pop(), m.stack.is_empty()) {
                    (Some(Value::Tensor(t)), true) => Ok(t),
                    _ => Err("`prompt_embeds` pickle does not hold one tensor".into()),
                };
            }
            op => {
                return Err(format!(
                    "`prompt_embeds` pickle uses opcode {op:#04x}, which is not part of a \
                     tensor's pickle"
                ))
            }
        }
    }
}
