//! A teacher-forced quality instrument for any model the generic route serves.
//!
//! One fixed token sequence is run through the loaded model the way a prefill runs it, in
//! chunks at increasing positions, keeping the logits of every position: each position's
//! distribution comes from the true tokens before it. Per text the instrument reports the mean
//! negative log-likelihood of the true next token (nats per token) and its perplexity, and
//! records each position's top-20 log-probabilities in a compact dump. A run given the dump of
//! another run on the same tokens also reports its top-1 agreement with that run and the mean
//! KL divergence of its distribution from the other's over the other's top-20.
//!
//! Invariants: a text's rows come from one pass over the same tokens whatever the chunking, the
//! placement or the number of cards; the dump holds the tokens it was made from and a run
//! against it refuses a different sequence.
//!
//! The model run is a driver the caller writes: it loads a model as the server does, reads texts,
//! and hands both to `score_pass`; no file or model name lives here.
//!
//! Held-out text rule (`select_files`): candidates are the repository's `src/**/*.rs` files for
//! a code text and `docs/**/*.md` for a documentation text, at least `MIN_FILE_BYTES` long. A
//! candidate whose distinct substantial lines are mostly present in the calibration text is
//! dropped. The rest are ordered by the SHA-256 of their relative path, which is uncorrelated
//! with their content, and read in that order until the text holds enough characters; the run
//! cuts it to its token budget.

use super::*;
use crate::inference::engine::decode_step::VERIFY_ROWS;
use crate::tensor::DType;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::Path;

/// Positions' recorded candidates.
pub const TOP: usize = 20;

/// Logit rows moved to the host and analysed at a time.
const ROW_BLOCK: usize = 128;

/// A candidate text file shorter than this is not a text.
pub const MIN_FILE_BYTES: usize = 2000;

/// A candidate with at least this share of its distinct substantial lines in the calibration
/// text is dropped.
pub const CALIBRATION_OVERLAP: f64 = 0.5;

/// A line shorter than this (trimmed) says too little to count as shared.
const SUBSTANTIAL_LINE: usize = 20;

/// What one row of logits says: its log-sum-exp and its `TOP` best tokens.
struct RowStat {
    lse: f64,
    ids: [u32; TOP],
    lp: [f64; TOP],
}

fn analyse(row: &[f32]) -> RowStat {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    let k = TOP.min(row.len().saturating_sub(1));
    if k > 0 {
        idx.select_nth_unstable_by(k - 1, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    }
    let mut head: Vec<u32> = idx[..TOP.min(row.len())].to_vec();
    head.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    let mut ids = [0u32; TOP];
    let mut lp = [f64::NEG_INFINITY; TOP];
    for (i, &t) in head.iter().enumerate() {
        ids[i] = t;
        lp[i] = row[t as usize] as f64 - lse;
    }
    RowStat { lse, ids, lp }
}

/// KL(p || q) over the `TOP` tokens of `p`, both renormalised over those tokens; `p` and `q`
/// are log-probabilities.
pub fn kl_over_top(p: &[f64; TOP], q: &[f64; TOP]) -> f64 {
    let norm = |v: &[f64; TOP]| -> Vec<f64> {
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
        let s: f64 = e.iter().sum();
        e.into_iter().map(|x| x / s).collect()
    };
    let (p, q) = (norm(p), norm(q));
    p.iter()
        .zip(&q)
        .map(|(p, q)| {
            if *p > 0.0 {
                p * (p / q.max(1e-300)).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// The top-20 of a pass: its tokens and, per position, the best tokens and their
/// log-probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct Dump {
    pub tokens: Vec<u32>,
    pub ids: Vec<[u32; TOP]>,
    pub lp: Vec<[f32; TOP]>,
    /// Per position the log-probability of the true next token; NaN at the last position.
    pub true_lp: Vec<f32>,
}

const MAGIC: &[u8; 4] = b"LKQ2";

impl Dump {
    fn new(tokens: &[u32]) -> Self {
        Self {
            tokens: tokens.to_vec(),
            ids: Vec::new(),
            lp: Vec::new(),
            true_lp: Vec::new(),
        }
    }

    /// Little-endian: magic, `TOP`, position count, the tokens, then per position `TOP` ids,
    /// `TOP` f32 log-probabilities and the f32 log-probability of the true next token.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        out.write_all(MAGIC)?;
        out.write_all(&(TOP as u32).to_le_bytes())?;
        out.write_all(&(self.tokens.len() as u64).to_le_bytes())?;
        for t in &self.tokens {
            out.write_all(&t.to_le_bytes())?;
        }
        for ((ids, lp), t) in self.ids.iter().zip(&self.lp).zip(&self.true_lp) {
            for i in ids {
                out.write_all(&i.to_le_bytes())?;
            }
            for l in lp {
                out.write_all(&l.to_le_bytes())?;
            }
            out.write_all(&t.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read(mut input: impl Read) -> std::io::Result<Self> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let mut magic = [0u8; 4];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("not a quality dump"));
        }
        let mut w = [0u8; 4];
        input.read_exact(&mut w)?;
        if u32::from_le_bytes(w) as usize != TOP {
            return Err(bad("a dump of another width"));
        }
        let mut n8 = [0u8; 8];
        input.read_exact(&mut n8)?;
        let n = u64::from_le_bytes(n8) as usize;
        let mut u32s = |count: usize| -> std::io::Result<Vec<[u8; 4]>> {
            let mut buf = vec![0u8; count * 4];
            input.read_exact(&mut buf)?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect())
        };
        let tokens: Vec<u32> = u32s(n)?.into_iter().map(u32::from_le_bytes).collect();
        let body = u32s(n * (TOP * 2 + 1))?;
        let mut ids = Vec::with_capacity(n);
        let mut lp = Vec::with_capacity(n);
        let mut true_lp = Vec::with_capacity(n);
        for row in body.chunks_exact(TOP * 2 + 1) {
            ids.push(std::array::from_fn(|j| u32::from_le_bytes(row[j])));
            lp.push(std::array::from_fn(|j| f32::from_le_bytes(row[TOP + j])));
            true_lp.push(f32::from_le_bytes(row[TOP * 2]));
        }
        Ok(Self {
            tokens,
            ids,
            lp,
            true_lp,
        })
    }
}

/// Sums over the positions scored so far.
#[derive(Debug, Default, Clone)]
pub struct Accum {
    pub positions: usize,
    pub predicted: usize,
    pub nll_sum: f64,
    pub top1_hits: usize,
    pub kl_sum: f64,
    pub compared: usize,
}

impl Accum {
    /// Mean negative log-likelihood of the true next token, nats per token.
    pub fn nll(&self) -> f64 {
        self.nll_sum / self.predicted.max(1) as f64
    }

    pub fn perplexity(&self) -> f64 {
        self.nll().exp()
    }

    /// Share of positions whose best token is the reference's, when a reference was given.
    pub fn top1(&self) -> Option<f64> {
        (self.compared > 0).then(|| self.top1_hits as f64 / self.compared as f64)
    }

    /// Mean KL(reference || run) over the reference's top-20, when a reference was given.
    pub fn kl(&self) -> Option<f64> {
        (self.compared > 0).then(|| self.kl_sum / self.compared as f64)
    }
}

/// Score `rows` (row-major, `vocab` wide) as the positions `first..` of `tokens`: row `r`
/// predicts `tokens[first + r + 1]`. Appends each row's top-20 to `dump`.
pub fn score_rows(
    rows: &[f32],
    vocab: usize,
    first: usize,
    tokens: &[u32],
    reference: Option<&Dump>,
    acc: &mut Accum,
    dump: &mut Dump,
) {
    let stats: Vec<(RowStat, Option<f64>)> = rows
        .par_chunks(vocab)
        .enumerate()
        .map(|(r, row)| {
            let s = analyse(row);
            let true_lp = tokens
                .get(first + r + 1)
                .map(|&t| row[t as usize] as f64 - s.lse);
            (s, true_lp)
        })
        .collect();
    for (r, (s, true_lp)) in stats.iter().enumerate() {
        let p = first + r;
        acc.positions += 1;
        if let Some(lp) = true_lp {
            acc.predicted += 1;
            acc.nll_sum -= lp;
        }
        if let Some(refd) = reference {
            let row = &rows[r * vocab..(r + 1) * vocab];
            let rids = &refd.ids[p];
            let at: [f64; TOP] = std::array::from_fn(|j| row[rids[j] as usize] as f64 - s.lse);
            let p_ref: [f64; TOP] = std::array::from_fn(|j| refd.lp[p][j] as f64);
            acc.compared += 1;
            acc.top1_hits += (s.ids[0] == rids[0]) as usize;
            acc.kl_sum += kl_over_top(&p_ref, &at);
        }
        dump.true_lp.push(true_lp.map_or(f32::NAN, |l| l as f32));
        dump.ids.push(s.ids);
        dump.lp.push(std::array::from_fn(|j| s.lp[j] as f32));
    }
}

/// Chunk lengths covering `total` tokens at `chunk` each. A tail no longer than a draft
/// verify would run as a decode step, not as the batch a prefill runs, so it joins the chunk
/// before it.
pub fn chunk_sizes(total: usize, chunk: usize) -> Vec<usize> {
    let chunk = chunk.max(1);
    let mut sizes: Vec<usize> = (0..total.div_ceil(chunk))
        .map(|i| chunk.min(total - i * chunk))
        .collect();
    if sizes.len() > 1 && *sizes.last().unwrap() <= VERIFY_ROWS {
        let tail = sizes.pop().unwrap();
        *sizes.last_mut().unwrap() += tail;
    }
    sizes
}

/// One teacher-forced pass over `tokens`. `forward` is the model's multi-position forward
/// (`[1, n]` ids at a position to `[1, n, vocab]` logits) and `reset` empties its cache.
/// Returns the sums and the pass's dump.
pub fn score_pass(
    forward: &mut dyn FnMut(&Tensor, usize) -> crate::tensor::Result<Tensor>,
    reset: &mut dyn FnMut(),
    device: &Device,
    tokens: &[u32],
    chunk: usize,
    reference: Option<&Dump>,
) -> crate::tensor::Result<(Accum, Dump)> {
    if let Some(r) = reference {
        if r.tokens != tokens {
            crate::tensor::bail!("the reference dump was made from other tokens");
        }
    }
    reset();
    let mut acc = Accum::default();
    let mut dump = Dump::new(tokens);
    let mut at = 0usize;
    for len in chunk_sizes(tokens.len(), chunk) {
        let ids = Tensor::from_vec(tokens[at..at + len].to_vec(), (1, len), device)?;
        let logits = forward(&ids, at)?;
        let vocab = logits.dim(2)?;
        let mut done = 0usize;
        while done < len {
            let n = ROW_BLOCK.min(len - done);
            let block = logits
                .narrow(1, done, n)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            score_rows(
                &block,
                vocab,
                at + done,
                tokens,
                reference,
                &mut acc,
                &mut dump,
            );
            done += n;
        }
        at += len;
    }
    Ok((acc, dump))
}

/// The files of `candidates` (relative path, content) that make a held-out text of at least
/// `want_chars` characters under the rule in the module comment, in reading order.
pub fn select_files(
    candidates: &[(String, String)],
    calibration: &str,
    want_chars: usize,
) -> Vec<String> {
    let substantial = |s: &str| -> HashSet<String> {
        s.lines()
            .map(str::trim)
            .filter(|l| l.len() >= SUBSTANTIAL_LINE)
            .map(String::from)
            .collect()
    };
    let calibrated = substantial(calibration);
    let mut ranked: Vec<(String, &String, &String)> = candidates
        .iter()
        .filter(|(_, text)| text.len() >= MIN_FILE_BYTES)
        .filter(|(_, text)| {
            let lines = substantial(text);
            let shared = lines.iter().filter(|l| calibrated.contains(*l)).count();
            (shared as f64) < CALIBRATION_OVERLAP * lines.len().max(1) as f64
        })
        .map(|(path, text)| {
            let key: String = Sha256::digest(path.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            (key, path, text)
        })
        .collect();
    ranked.sort();
    let mut chosen = Vec::new();
    let mut chars = 0usize;
    for (_, path, text) in ranked {
        if chars >= want_chars {
            break;
        }
        chars += text.len();
        chosen.push(path.clone());
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random logits in [-4, 4).
    fn logits(rows: usize, vocab: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..rows * vocab)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
            })
            .collect()
    }

    /// The NLL against a plain f64 evaluation of the softmax, written separately.
    #[test]
    fn nll_matches_an_independent_f64_computation() {
        let (n, vocab) = (12, 37);
        let rows = logits(n, vocab, 7);
        let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 5 + 3) % vocab as u32).collect();
        let mut acc = Accum::default();
        let mut dump = Dump::new(&tokens);
        score_rows(&rows, vocab, 0, &tokens, None, &mut acc, &mut dump);
        let mut want = 0f64;
        for i in 0..n - 1 {
            let z: f64 = rows[i * vocab..(i + 1) * vocab]
                .iter()
                .map(|&x| (x as f64).exp())
                .sum();
            let p = (rows[i * vocab + tokens[i + 1] as usize] as f64).exp() / z;
            want += -p.ln();
        }
        want /= (n - 1) as f64;
        assert_eq!(acc.predicted, n - 1);
        let from_dump: f64 = dump.true_lp[..n - 1]
            .iter()
            .map(|&x| -(x as f64))
            .sum::<f64>()
            / (n - 1) as f64;
        assert!((from_dump - acc.nll()).abs() < 1e-5);
        assert!(
            (acc.nll() - want).abs() < 1e-12,
            "{} against {want}",
            acc.nll()
        );
        assert!((acc.perplexity() - want.exp()).abs() < 1e-9);
    }

    #[test]
    fn a_run_against_itself_agrees_and_has_no_divergence() {
        let (n, vocab) = (9, 64);
        let rows = logits(n, vocab, 11);
        let tokens: Vec<u32> = (0..n as u32).collect();
        let mut acc = Accum::default();
        let mut dump = Dump::new(&tokens);
        score_rows(&rows, vocab, 0, &tokens, None, &mut acc, &mut dump);
        let mut again = Accum::default();
        let mut scratch = Dump::new(&tokens);
        score_rows(
            &rows,
            vocab,
            0,
            &tokens,
            Some(&dump),
            &mut again,
            &mut scratch,
        );
        assert_eq!(again.top1(), Some(1.0));
        assert!(
            again.kl().unwrap().abs() < 1e-6,
            "kl {}",
            again.kl().unwrap()
        );
    }

    /// A perturbed run diverges, and swapping two tokens' values between two distributions
    /// that otherwise coincide diverges the same each way.
    #[test]
    fn a_perturbed_run_diverges_and_a_swap_is_symmetric() {
        let vocab = 40usize;
        let base: Vec<f32> = (0..vocab).map(|i| 5.0 - i as f32 * 0.2).collect();
        let mut swapped = base.clone();
        swapped.swap(0, 1);
        let kl_of = |reference: &[f32], run: &[f32]| -> (f64, Option<f64>) {
            let tokens = [0u32];
            let mut acc = Accum::default();
            let mut ref_dump = Dump::new(&tokens);
            score_rows(reference, vocab, 0, &tokens, None, &mut acc, &mut ref_dump);
            let mut scored = Accum::default();
            let mut scratch = Dump::new(&tokens);
            score_rows(
                run,
                vocab,
                0,
                &tokens,
                Some(&ref_dump),
                &mut scored,
                &mut scratch,
            );
            (scored.kl().unwrap(), scored.top1())
        };
        let (forward, top_f) = kl_of(&base, &swapped);
        let (backward, _) = kl_of(&swapped, &base);
        assert!(forward > 0.0);
        assert_eq!(top_f, Some(0.0));
        assert!(
            (forward - backward).abs() < 1e-6,
            "{forward} against {backward}"
        );
        let noise: Vec<f32> = base
            .iter()
            .zip(logits(1, vocab, 3))
            .map(|(b, n)| b + n * 0.5)
            .collect();
        assert!(kl_of(&base, &noise).0 > 0.0);
    }

    #[test]
    fn a_dump_survives_its_file_format() {
        let (n, vocab) = (6, 50);
        let tokens: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
        let mut acc = Accum::default();
        let mut dump = Dump::new(&tokens);
        score_rows(
            &logits(n, vocab, 5),
            vocab,
            0,
            &tokens,
            None,
            &mut acc,
            &mut dump,
        );
        let mut bytes = Vec::new();
        dump.write(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 16 + n * 4 + n * (TOP * 8 + 4));
        let back = Dump::read(&bytes[..]).unwrap();
        assert_eq!(
            (&back.tokens, &back.ids, &back.lp),
            (&dump.tokens, &dump.ids, &dump.lp)
        );
        let bits = |d: &Dump| d.true_lp.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&back), bits(&dump));
        assert!(dump.true_lp[n - 1].is_nan() && dump.true_lp[..n - 1].iter().all(|x| *x < 0.0));
        assert!(Dump::read(&b"nope"[..]).is_err());
    }

    #[test]
    fn chunks_cover_the_text_and_leave_no_decode_sized_tail() {
        for (total, chunk) in [
            (1500, 512),
            (2200, 2048),
            (100, 512),
            (520, 512),
            (1024, 512),
        ] {
            let sizes = chunk_sizes(total, chunk);
            assert_eq!(sizes.iter().sum::<usize>(), total);
            if sizes.len() > 1 {
                assert!(
                    *sizes.last().unwrap() > VERIFY_ROWS,
                    "{total}/{chunk}: {sizes:?}"
                );
            }
        }
        assert_eq!(chunk_sizes(1500, 512), vec![512, 512, 476]);
        assert_eq!(chunk_sizes(520, 512), vec![520]);
    }

    #[test]
    fn selection_is_deterministic_and_skips_calibrated_files() {
        let body = |tag: &str| {
            (0..100)
                .map(|i| format!("let value_{tag}_{i} = compute({i});\n"))
                .collect::<String>()
        };
        let files: Vec<(String, String)> = ["a", "b", "c", "d"]
            .iter()
            .map(|t| (format!("src/{t}.rs"), body(t)))
            .chain([("src/tiny.rs".to_string(), "fn f() {}".to_string())])
            .collect();
        let all = select_files(&files, "", usize::MAX);
        assert_eq!(all.len(), 4);
        assert_eq!(all, select_files(&files, "", usize::MAX));
        assert!(!all.contains(&"src/tiny.rs".to_string()));
        let calibrated = select_files(&files, &body("b"), usize::MAX);
        assert_eq!(calibrated.len(), 3);
        assert!(!calibrated.contains(&"src/b.rs".to_string()));
        let one = select_files(&files, "", 1);
        assert_eq!(one, all[..1]);
    }

    /// The toy model on the host, scored through the instrument.
    fn toy() -> crate::inference::generic_transformer::GenericHeteroTransformer {
        use crate::inference::generic_transformer::GenericHeteroTransformer;
        use crate::inference::place::layer_executor::HeteroPlan;
        use crate::tensor::quantized::gguf_file;
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/qwen4exp/toy.gguf");
        let file = std::fs::File::open(path).unwrap();
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
        let mut content =
            gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
                .unwrap();
        content.mmap_owner = Some(mmap.clone() as Arc<dyn std::any::Any + Send + Sync>);
        let layers = content.metadata["qwen4exp.block_count"].to_u32().unwrap() as usize;
        GenericHeteroTransformer::from_gguf_with_kv_quant(
            content,
            &mmap[..],
            &HashMap::new(),
            &HeteroPlan::forced_gpu(layers, 0, 0),
            KvQuant::Off,
            None,
        )
        .unwrap()
    }

    fn toy_tokens(vocab: usize) -> Vec<u32> {
        (0..70u32).map(|i| (i * 7 + 1) % vocab as u32).collect()
    }

    fn toy_vocab(
        model: &mut crate::inference::generic_transformer::GenericHeteroTransformer,
    ) -> usize {
        let one = Tensor::from_vec(vec![1u32], (1, 1), &Device::Cpu).unwrap();
        model.forward(&one, 0).unwrap().elem_count()
    }

    /// The scorer against a direct f64 evaluation of the same logits, chunked and whole, and
    /// the divergence of the chunked run from the whole one.
    #[test]
    fn the_toy_is_scored_the_same_whole_and_chunked() {
        let mut model = toy();
        let vocab = {
            model.trim_kv(0);
            toy_vocab(&mut model)
        };
        let tokens = toy_tokens(vocab);
        let ids = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &Device::Cpu).unwrap();
        model.trim_kv(0);
        let all = model
            .forward_all(&ids, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut want = 0f64;
        for i in 0..tokens.len() - 1 {
            let row = &all[i * vocab..(i + 1) * vocab];
            let m = row.iter().copied().fold(f32::MIN, f32::max) as f64;
            let z: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
            want -= row[tokens[i + 1] as usize] as f64 - m - z.ln();
        }
        want /= (tokens.len() - 1) as f64;

        let cell = std::cell::RefCell::new(model);
        let run = |chunk: usize, reference: Option<&Dump>| {
            score_pass(
                &mut |x, p| cell.borrow_mut().forward_all(x, p),
                &mut || cell.borrow_mut().trim_kv(0),
                &Device::Cpu,
                &tokens,
                chunk,
                reference,
            )
            .unwrap()
        };
        let (whole, whole_dump) = run(tokens.len(), None);
        assert!(
            (whole.nll() - want).abs() < 1e-9,
            "{} against {want}",
            whole.nll()
        );
        let (chunked, _) = run(24, Some(&whole_dump));
        assert!(
            (chunked.nll() - whole.nll()).abs() < 0.05,
            "{} against {}",
            chunked.nll(),
            whole.nll()
        );
        assert!(chunked.kl().unwrap() < 0.05, "kl {}", chunked.kl().unwrap());
        let (itself, _) = run(tokens.len(), Some(&whole_dump));
        assert_eq!(itself.top1(), Some(1.0));
        assert!(itself.kl().unwrap().abs() < 1e-9);
        let other: Vec<u32> = tokens.iter().rev().copied().collect();
        let refused = score_pass(
            &mut |x, p| cell.borrow_mut().forward_all(x, p),
            &mut || cell.borrow_mut().trim_kv(0),
            &Device::Cpu,
            &other,
            tokens.len(),
            Some(&whole_dump),
        );
        assert!(refused.is_err());
    }

    /// A run whose logits are perturbed diverges from the exact run and agrees less.
    #[test]
    fn a_perturbed_toy_run_diverges_from_the_exact_one() {
        let mut model = toy();
        model.trim_kv(0);
        let vocab = toy_vocab(&mut model);
        let tokens = toy_tokens(vocab);
        let cell = std::cell::RefCell::new(model);
        let (_, exact) = score_pass(
            &mut |x, p| cell.borrow_mut().forward_all(x, p),
            &mut || cell.borrow_mut().trim_kv(0),
            &Device::Cpu,
            &tokens,
            tokens.len(),
            None,
        )
        .unwrap();
        let (perturbed, _) = score_pass(
            &mut |x, p| {
                let logits = cell.borrow_mut().forward_all(x, p)?;
                let (b, n, v) = logits.dims3()?;
                let noise = Tensor::from_vec(logits_noise(n * v), (b, n, v), &Device::Cpu)?;
                logits.add(&noise)
            },
            &mut || cell.borrow_mut().trim_kv(0),
            &Device::Cpu,
            &tokens,
            tokens.len(),
            Some(&exact),
        )
        .unwrap();
        assert!(perturbed.kl().unwrap() > 0.0);
        assert!(perturbed.top1().unwrap() < 1.0 || perturbed.kl().unwrap() > 1e-4);
    }

    fn logits_noise(len: usize) -> Vec<f32> {
        logits(1, len, 99).into_iter().map(|x| x * 0.5).collect()
    }
}
