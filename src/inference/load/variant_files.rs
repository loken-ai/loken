//! A further file of the checkpoint a model is served from: the same routed experts quantised to
//! another format, every other tensor as the model's first file holds it. Checked at load before
//! any expert is read from it.
//!
//! The two files are of one checkpoint when they carry the same architecture and tokenizer, the
//! same routed experts (layer, projection and expert) at the same shapes, and every other tensor
//! of the first file at the same shape, format and bytes. A table read row by row is compared on
//! evenly spread spans, as many bytes in all as the largest tensor compared whole. The further
//! file may carry tensors the first does not: they are never read.

use crate::tensor::quantized::gguf_source::GgufSource;
use crate::tensor::{Error, Result};
use std::collections::BTreeMap;

/// The routed expert a tensor holds, by stack and expert: `(stack, Some(e))` for one stored as
/// its own tensor (`blk.3.ffn_gate_exps.17`), `(stack, None)` for a whole stack
/// (`blk.3.ffn_gate_exps.weight`). `None` for a tensor that is not a routed expert.
fn routed(name: &str) -> Option<(String, Option<usize>)> {
    if !name.contains("_exps") {
        return None;
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    match base.rsplit_once('.') {
        Some((stack, e)) if stack.ends_with("_exps") => e
            .parse()
            .ok()
            .map(|e| (stack.to_string(), Some(e)))
            .or(Some((base.to_string(), None))),
        _ => Some((base.to_string(), None)),
    }
}

/// Every routed expert a file holds, `(stack, expert)`, with its shape.
fn routed_experts<G: GgufSource>(g: &G) -> BTreeMap<(String, usize), Vec<usize>> {
    let mut out = BTreeMap::new();
    for name in g.tensor_names() {
        let (Some((stack, e)), Some(info)) = (routed(&name), g.info(&name)) else {
            continue;
        };
        let dims = info.shape.dims();
        match e {
            Some(e) => {
                out.insert((stack, e), dims.to_vec());
            }
            None if dims.len() == 3 => {
                for e in 0..dims[0] {
                    out.insert((stack.clone(), e), dims[1..].to_vec());
                }
            }
            None => {
                out.insert((stack, 0), dims.to_vec());
            }
        }
    }
    out
}

/// Whether `len` bytes at `offset` of `file` equal `want`, read in pieces of `piece` bytes; the pages read are
/// dropped from the page cache once compared, so the check leaves the memory as it found it.
fn file_equals(file: &std::fs::File, offset: u64, want: &[u8], piece: usize) -> Result<bool> {
    use std::os::unix::fs::FileExt;
    let piece = piece.max(1);
    let mut buf = vec![0u8; piece.min(want.len())];
    let mut at = 0usize;
    let mut same = true;
    while at < want.len() {
        let n = piece.min(want.len() - at);
        file.read_exact_at(&mut buf[..n], offset + at as u64)
            .map_err(|e| Error::msg(format!("expert variants: read: {e}")))?;
        if buf[..n] != want[at..at + n] {
            same = false;
            break;
        }
        at += n;
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // Safety: a valid descriptor; advice only.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                want.len() as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    Ok(same)
}

/// Check that `variant` is of the checkpoint `primary` is, as the module says; `rows(name)`
/// names the tables read row by row. The error says what differs.
pub fn check_variant<A: GgufSource, B: GgufSource>(
    primary: &A,
    variant: &B,
    rows: &dyn Fn(&str) -> bool,
) -> Result<()> {
    let refuse = |why: String| Err(Error::msg(format!("expert variants: {why}")));
    let meta_a = |k: &str| primary.metadata().get(k).map(|v| format!("{v:?}"));
    let meta_b = |k: &str| variant.metadata().get(k).map(|v| format!("{v:?}"));
    let arch = "general.architecture";
    if meta_a(arch) != meta_b(arch) || meta_a(arch).is_none() {
        return refuse(format!(
            "the architectures differ ({:?} and {:?})",
            meta_a(arch),
            meta_b(arch)
        ));
    }
    let mut keys: Vec<&String> = primary
        .metadata()
        .keys()
        .chain(variant.metadata().keys())
        .filter(|k| k.starts_with("tokenizer."))
        .collect();
    keys.sort();
    keys.dedup();
    if let Some(k) = keys.iter().find(|k| meta_a(k) != meta_b(k)) {
        return refuse(format!("the tokenizers differ at {k}"));
    }
    let (ea, eb) = (routed_experts(primary), routed_experts(variant));
    if ea.is_empty() {
        return refuse("the first file has no routed experts".into());
    }
    if let Some((k, shape)) = ea.iter().find(|(k, s)| eb.get(*k) != Some(*s)) {
        return refuse(format!(
            "expert {} of {} is {:?} in the first file and {:?} in the other",
            k.1,
            k.0,
            shape,
            eb.get(k)
        ));
    }
    if let Some(k) = eb.keys().find(|k| !ea.contains_key(*k)) {
        return refuse(format!(
            "expert {} of {} is only in the other file",
            k.1, k.0
        ));
    }
    // Every other tensor of the first file: same shape and format, then the same bytes.
    let mut others: Vec<String> = primary
        .tensor_names()
        .into_iter()
        .filter(|n| routed(n).is_none())
        .collect();
    others.sort();
    let mut whole = Vec::new();
    let mut by_rows = Vec::new();
    for name in others {
        let (Some(a), Some(b)) = (primary.info(&name), variant.info(&name)) else {
            return refuse(format!("the other file has no tensor {name}"));
        };
        if a.shape.dims() != b.shape.dims() || a.ggml_dtype != b.ggml_dtype {
            return refuse(format!(
                "{name} is {:?} {:?} in the first file and {:?} {:?} in the other",
                a.ggml_dtype,
                a.shape.dims(),
                b.ggml_dtype,
                b.shape.dims()
            ));
        }
        let (Some((ma, oa, la)), Some((_, ob, lb)), Some(file)) = (
            primary.mapped_range(&name)?,
            variant.mapped_range(&name)?,
            variant.mapped_file(&name),
        ) else {
            return refuse(format!("{name} is not mapped in both files"));
        };
        if la != lb {
            return refuse(format!("{name} is {la} bytes and {lb} bytes"));
        }
        let entry = (name.clone(), ma, oa, la, file, ob);
        if rows(&name) {
            by_rows.push(entry);
        } else {
            whole.push(entry);
        }
    }
    let differs = |name: &str| {
        refuse(format!(
            "{name} holds other bytes in the two files: they are not of one checkpoint"
        ))
    };
    let largest = whole.iter().map(|w| w.3).max().unwrap_or(0);
    // The tensors compared whole, spread over the cores: the reads of both files overlap.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(whole.len().max(1));
    // One tile of reading in flight over all the workers.
    let piece = crate::inference::offload::WEIGHT_TILE_BYTES / workers;
    let found: std::sync::Mutex<Option<Result<String>>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((name, map, at, len, file, other_at)) = whole.get(i) else {
                    return;
                };
                if found.lock().unwrap().is_some() {
                    return;
                }
                match file_equals(file, *other_at as u64, &map[*at..*at + *len], piece) {
                    Ok(true) => {}
                    Ok(false) => *found.lock().unwrap() = Some(Ok(name.clone())),
                    Err(e) => *found.lock().unwrap() = Some(Err(e)),
                }
            });
        }
    });
    match found.into_inner().unwrap() {
        Some(Ok(name)) => return differs(&name),
        Some(Err(e)) => return Err(e),
        None => {}
    }
    // A table read by rows: spans of one read piece each, evenly spread, `largest` bytes in all.
    let page = crate::tensor::mapped::page_size().unwrap_or(4096).max(1);
    let span = piece.max(page) / page * page;
    for (name, map, at, len, file, other_at) in &by_rows {
        let spans = len.div_ceil(span);
        let take = (largest / span).clamp(1, spans);
        for i in 0..take {
            let start = i * spans / take * span;
            let n = span.min(len - start);
            let want = &map[at + start..at + start + n];
            if !file_equals(file, (other_at + start) as u64, want, span)? {
                return differs(name);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::{open_mapped, Value};
    use crate::tensor::quantized::GgmlDType;

    fn entry(name: &str, dims: Vec<usize>, seed: f32) -> GgufEntry {
        let n: usize = dims.iter().product();
        GgufEntry {
            name: name.to_string(),
            dims,
            dtype: GgmlDType::F32,
            data: (0..n)
                .flat_map(|i| (i as f32 * 0.01 + seed).to_le_bytes())
                .collect(),
        }
    }

    fn write(tag: &str, tokens: &str, entries: Vec<GgufEntry>) -> std::path::PathBuf {
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("toy".into()),
            ),
            (
                "tokenizer.ggml.model".to_string(),
                Value::String(tokens.into()),
            ),
        ];
        let path =
            std::env::temp_dir().join(format!("loken-variant-{tag}-{}.gguf", std::process::id()));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        path
    }

    /// Units in one file, a stack in the other; the other tensors as given.
    fn pair(
        tag: &str,
        other_tokens: &str,
        other_norm_seed: f32,
        other_experts: usize,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut a = vec![entry("blk.0.attn_norm.weight", vec![8], 0.0)];
        for e in 0..2 {
            a.push(entry(&format!("blk.0.ffn_gate_exps.{e}"), vec![4, 8], 1.0));
        }
        let b = vec![
            entry("blk.0.attn_norm.weight", vec![8], other_norm_seed),
            entry("blk.0.ffn_gate_exps.weight", vec![other_experts, 4, 8], 2.0),
            entry("blk.0.extra", vec![2], 0.0),
        ];
        (
            write(&format!("{tag}a"), "gpt2", a),
            write(&format!("{tag}b"), other_tokens, b),
        )
    }

    fn check(paths: (std::path::PathBuf, std::path::PathBuf)) -> Result<()> {
        let (a, b) = (
            open_mapped(&paths.0).unwrap(),
            open_mapped(&paths.1).unwrap(),
        );
        let _ = std::fs::remove_file(&paths.0);
        let _ = std::fs::remove_file(&paths.1);
        check_variant(&a, &b, &|_| false)
    }

    #[test]
    fn a_file_of_the_same_checkpoint_is_accepted() {
        check(pair("same", "gpt2", 0.0, 2)).unwrap();
    }

    #[test]
    fn other_bytes_another_tokenizer_or_other_experts_are_refused() {
        let e = check(pair("bytes", "gpt2", 0.5, 2))
            .unwrap_err()
            .to_string();
        assert!(e.contains("not of one checkpoint"), "{e}");
        let e = check(pair("tok", "llama", 0.0, 2)).unwrap_err().to_string();
        assert!(e.contains("tokenizers differ"), "{e}");
        let e = check(pair("count", "gpt2", 0.0, 3))
            .unwrap_err()
            .to_string();
        assert!(e.contains("only in the other file"), "{e}");
    }

    #[test]
    fn expert_names_are_read_as_stack_and_expert() {
        assert_eq!(
            routed("blk.3.ffn_gate_exps.17"),
            Some(("blk.3.ffn_gate_exps".into(), Some(17)))
        );
        assert_eq!(
            routed("blk.3.ffn_down_exps.weight"),
            Some(("blk.3.ffn_down_exps".into(), None))
        );
        assert_eq!(routed("blk.3.ffn_down_shexp.weight"), None);
    }
}
