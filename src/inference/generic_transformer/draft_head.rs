//! The draft head: one more layer of the model's own kind, fed a token's embedding fused with the
//! hidden state before it, that predicts the token after.
//!
//! The head sees pairs `(token at p + 1, hidden at p)` and writes its own cache over the
//! positions, so it keeps no recurrent state: cutting it back is cutting its cache. Its output
//! goes through its own final mixer and the model's output head.

use super::hyper::HyperMixer;
use super::projection::QMatMul;
use super::{GenericHeteroTransformer, GenericTransformerLayer, LayerDevice};
use crate::tensor::layer::RmsNorm;
use crate::tensor::{DType, Device, Result, Tensor, D};

pub(super) struct DraftHead {
    pub(super) layer: GenericTransformerLayer,
    pub(super) layer_dev: LayerDevice,
    pub(super) norm_embedding: RmsNorm,
    /// `[streams * dim]` F32, the scale of the stream-wise norm the hidden state goes through.
    pub(super) hidden_gamma: Tensor,
    pub(super) fc_embedding: QMatMul,
    pub(super) fc_hidden: QMatMul,
    /// Collapses the layer's streams into the row the output head reads.
    pub(super) mixer: HyperMixer,
    pub(super) streams: usize,
    pub(super) dim: usize,
    pub(super) eps: f64,
    /// Positions the head's cache holds.
    pub(super) len: usize,
    /// Positions that hold a pair of the sequence itself, not one the head guessed ahead.
    pub(super) committed: usize,
    /// The last row of the prompt chunk before, whose pair waits for the next chunk's first token.
    pub(super) carried: Option<Tensor>,
    /// Rows of the prompt the head guessed right, and the rows it was asked, this request.
    pub(super) fit: (u64, u64),
    /// The layer's indexer as it stood with `committed` pairs read: what a cut back to that
    /// length returns it to.
    pub(super) indexer_mark: Option<(usize, super::qsa::IndexerCache)>,
    /// The vocabulary the head guesses in: the first rows of the output projection, which the
    /// tokenizer numbers in order of frequency, as a view of the card's projection; `None` for
    /// the whole of it.
    pub(super) vocab: Option<(usize, crate::tensor::quantized::QMatMul)>,
    /// The head's guesses as the kernels of a recorded step.
    #[cfg(feature = "cuda")]
    pub(super) step: Option<DraftStep>,
}

/// The head's guesses as the kernels of recorded steps, replayed with the tokens' embeddings,
/// the rows' positions and tables refreshed before each replay: one graph per count of pairs
/// takes the pairs through the layer and leaves the last row; one names the token after the
/// row; one takes a guessed token with the row through the layer and names the next.
#[cfg(feature = "cuda")]
pub(super) struct DraftStep {
    /// The tokens' embeddings, `[1, VERIFY_ROWS, dim]` F32.
    emb: Tensor,
    /// The hidden rows the pairs read, `[1, VERIFY_ROWS, streams * dim]` F32.
    rows_in: Tensor,
    /// The row a guess reads and the row it writes, `[1, 1, streams * dim]` F32.
    hidden: Tensor,
    /// The token named, `[1]` U32, by the card's argmax.
    ids: Tensor,
    /// The softmax mass relative to the top logit, `[1, 1]` F32.
    read: Tensor,
    /// The round's committed tokens, `[VERIFY_ROWS]` U32, whose rows the card gathers.
    tok_ids: Tensor,
    /// Guesses told so far, `[1]` U32 on the card, and what the host last read of it.
    counter: Tensor,
    told: Told,
    seen: u32,
    /// The rows' positions, `[VERIFY_ROWS]` i32.
    pos: Tensor,
    /// The pairs through the layer, by their count, the most recently recorded last: at most
    /// [`DRAFT_PAIRS_GRAPHS`] of them, each reserving memory on the card.
    pairs: Vec<(usize, HeldGraph)>,
    /// The head over the row alone.
    head: Option<HeldGraph>,
    /// The layer over the named token and the row at row `j` of the round, then the head.
    steps: Vec<Option<HeldGraph>>,
    /// The buffers the graphs were recorded against: the cache's keys, the indexer's births and
    /// the rows of the vocabulary guessed in.
    key: Option<(crate::tensor::ops::traits::TensorId, u64, usize)>,
    /// Replays that failed since the load.
    failures: u32,
    /// Microseconds a guess's phases took (the preparation, the pairs, the head, the read,
    /// the steps), and the guesses they count.
    spent_us: [u64; 5],
    spent_n: u32,
    /// The recorded guesses are given up on: a capture failed or a replay disagreed.
    blocked: bool,
}

/// Replays that may fail before the head's guesses stay eager.
#[cfg(feature = "cuda")]
const DRAFT_RECORD_FAILURES: u32 = 3;
/// Rows of each prompt chunk the head's fit is measured over.
const DRAFT_FIT_ROWS: usize = 32;

/// Write a tensor's rows as little-endian f32 under `dir`, once per name, for a reference run.
/// Where the probes write the tensors they are asked for (`probe_dump_dir` in the
/// configuration), set once at the load; none by default.
static PROBE_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// Name the directory the probes write to; the first call holds.
pub fn set_probe_dir(dir: Option<String>) {
    let _ = PROBE_DIR.set(dir);
}

/// The directory the probes write to, if the configuration names one.
pub(super) fn probe_dir() -> Option<&'static str> {
    PROBE_DIR.get().and_then(|d| d.as_deref())
}

/// The rows `n` of a vocabulary of `rows` the head guesses in for a prompt of `tokens`: the
/// least of `n / rows * share + (tokens at or past n) / tokens`, over the cuts that keep a
/// prompt token in.
fn vocab_cut(tokens: &[u32], rows: usize, share: f32) -> usize {
    if tokens.is_empty() || share <= 0.0 {
        return rows;
    }
    let mut ids: Vec<usize> = tokens
        .iter()
        .map(|&t| t as usize)
        .filter(|&t| t < rows)
        .collect();
    ids.sort_unstable();
    let total = tokens.len() as f64;
    let mut best = (f64::from(share), rows);
    for (i, &id) in ids.iter().enumerate() {
        if i + 1 < ids.len() && ids[i + 1] == id {
            continue;
        }
        let n = id + 1;
        let past = (ids.len() - i - 1) as f64 + (tokens.len() - ids.len()) as f64;
        let cost = n as f64 / rows as f64 * f64::from(share) + past / total;
        if cost < best.0 {
            best = (cost, n);
        }
    }
    best.1
}

pub(super) fn dump_rows(dir: &str, name: &str, t: &Tensor) {
    let path = std::path::Path::new(dir).join(format!("{name}.f32"));
    if path.exists() {
        return;
    }
    if let Ok(v) = t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
    {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = std::fs::write(&path, bytes);
        let _ = std::fs::write(path.with_extension("shape"), format!("{:?}", t.dims()));
    }
}

fn dump_u32(dir: &str, name: &str, v: &[u32]) {
    let path = std::path::Path::new(dir).join(format!("{name}.u32"));
    if path.exists() {
        return;
    }
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let _ = std::fs::write(&path, bytes);
}
/// Guesses between two reports of a guess's phases.
const DRAFT_REPORT_EVERY: u32 = 64;
/// Pairs graphs the head keeps at once: a draft's pairs are the tokens the last verify kept
/// and one more, at most a verify's rows, and every count has its graph.
#[cfg(feature = "cuda")]
const DRAFT_PAIRS_GRAPHS: usize = crate::inference::engine::decode_step::VERIFY_ROWS;

/// A recorded graph the model holds: launched only by the thread that holds the model, on the
/// stream it was captured on, which is what its handles need.
#[cfg(feature = "cuda")]
pub(super) struct HeldGraph(crate::tensor::cuda_ext::CapturedGraph);

#[cfg(feature = "cuda")]
unsafe impl Send for HeldGraph {}
#[cfg(feature = "cuda")]
unsafe impl Sync for HeldGraph {}

/// What a recorded step of the head runs.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum DraftKind {
    /// `n` pairs through the layer; the last row is left for the head.
    Pairs(usize),
    /// The head over the row.
    Head,
    /// The token the head named with the row through the layer at row `j` of the round's
    /// staged positions, then the head.
    Step(usize),
}

/// What a guess costs to read: the token named, the softmax mass and the count of guesses
/// told, written by the card into page-locked memory the host spins on.
#[cfg(feature = "cuda")]
struct Told {
    host: *mut u32,
    device: u64,
}

#[cfg(feature = "cuda")]
unsafe impl Send for Told {}
#[cfg(feature = "cuda")]
unsafe impl Sync for Told {}

#[cfg(feature = "cuda")]
impl Told {
    const WORDS: usize = 4;
    const TOKEN: usize = 0;
    const MASS: usize = 1;
    const SEQ: usize = 2;

    fn new(dev: &Device) -> Result<Self> {
        let (host, device) = super::expert_tier::mapped(dev, Self::WORDS * 4)?;
        Ok(Self {
            host: host as *mut u32,
            device,
        })
    }

    fn word(&self, at: usize) -> u32 {
        // SAFETY: the mapping holds `WORDS` words for as long as this lives; the card writes
        // them, so each read is a fresh one.
        unsafe { std::ptr::read_volatile(self.host.add(at)) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for Told {
    fn drop(&mut self) {
        // SAFETY: allocated by `mapped` and freed once here.
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

/// The card address of `t`'s first element.
#[cfg(feature = "cuda")]
fn card_address<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor) -> Result<u64> {
    use cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        crate::tensor::StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            Ok(slice.device_ptr(slice.stream()).0
                + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
        }
        _ => crate::tensor::bail!("a guess's buffer is on the card"),
    }
}

impl DraftHead {
    /// The head's input for `emb` `[1, n, dim]` and `hidden` `[1, n, streams * dim]`: the
    /// embedding normed and projected, the hidden state normed as one row of `streams * dim`
    /// values and each stream projected, the first added to every stream of the second.
    fn fuse(&self, emb: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let (b, n, _) = hidden.dims3()?;
        let e = self.norm_embedding.forward(emb)?;
        let e = self
            .fc_embedding
            .forward(&e.reshape((b * n, self.dim))?)?
            .reshape((b, n, 1, self.dim))?;
        let ms = hidden.sqr()?.mean_keepdim(D::Minus1)?;
        let inv = (ms + self.eps)?.sqrt()?.recip()?;
        let h = hidden
            .broadcast_mul(&inv)?
            .broadcast_mul(&self.hidden_gamma.to_device(&hidden.device())?)?;
        let h = self
            .fc_hidden
            .forward(&h.reshape((b * n * self.streams, self.dim))?)?
            .reshape((b, n, self.streams, self.dim))?;
        h.broadcast_add(&e)?
            .reshape((b, n, self.streams * self.dim))
    }
}

impl DraftHead {
    /// Remember the indexer as it stands: `committed` pairs are read.
    fn mark_indexer(&mut self) {
        let mark = self
            .layer
            .sparse
            .as_ref()
            .and_then(|s| s.snapshot_cache().ok());
        self.indexer_mark = mark.map(|c| (self.committed, c));
    }
}

impl GenericHeteroTransformer {
    /// Whether the model came with a draft head.
    pub fn has_draft_head(&self) -> bool {
        self.draft.is_some()
    }

    /// The expert caches' mailboxes as they stand, one line per layer with one, the draft head's
    /// last: what a wait on the card that lasts is told by.
    pub fn tier_mailboxes(&self) -> Vec<String> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| (format!("layer {i}"), l))
            .chain(
                self.draft
                    .as_ref()
                    .map(|d| ("draft head".to_string(), &d.layer)),
            )
            .filter_map(|(name, l)| {
                let tier = l.moe.as_ref()?.tier.as_ref()?;
                Some(format!("{name}: {}", tier.mailbox_state()?))
            })
            .collect()
    }

    /// Before a token runs through the expert caches' layers: an error while a card's tier
    /// worker stands stalled or failed, a stall being lifted once the worker serves again.
    pub fn tier_check(&self) -> crate::tensor::Result<()> {
        self.tier_workers.iter().try_for_each(|w| w.check())
    }

    /// Why a card's tier worker left its mailboxes released, if one did: the step that ran
    /// through them is not to be trusted.
    pub fn tier_failure(&self) -> Option<String> {
        self.tier_workers.iter().find_map(|w| w.failure())
    }

    /// Positions the draft head's cache holds.
    pub fn draft_len(&self) -> usize {
        self.draft.as_ref().map_or(0, |d| d.len)
    }

    /// Run the head over the pairs `(tokens[i], hidden[i])` that sit at positions `at..`, and
    /// return the wide rows it wrote, `[1, n, streams * dim]`. `hidden` is the model's own
    /// last-layer state of the position before each token. With `kv_only` the layer stops after
    /// its token mixer: the rows leave their keys and values in its caches, and what is returned
    /// is not the head's output.
    pub fn draft_rows(
        &mut self,
        tokens: &[u32],
        hidden: &Tensor,
        at: usize,
        kv_only: bool,
    ) -> Result<Tensor> {
        let n = tokens.len();
        let (dev, edev) = {
            let d = self.draft.as_ref().ok_or_else(|| {
                crate::tensor::Error::msg("this model has no draft head".to_string())
            })?;
            let dev = match &d.layer_dev {
                LayerDevice::Cuda(i) => self.cuda_devices.get(i).cloned().unwrap_or(Device::Cpu),
                LayerDevice::Cpu => Device::Cpu,
            };
            (dev, self.embeddings.embeddings().device().clone())
        };
        let ids = Tensor::from_vec(tokens.to_vec(), (1, n), &edev)?;
        let mut emb = self.embeddings.forward(&ids)?.to_dtype(DType::F32)?;
        if let Some(scale) = self.config.embed_scale {
            emb = (emb * scale)?;
        }
        let emb = emb.to_device(&dev)?;
        let wide = hidden.elem_count() / n.max(1);
        let hidden = hidden
            .to_dtype(DType::F32)?
            .reshape((1, n, wide))?
            .to_device(&dev)?;
        let mut draft = self.draft.take().expect("checked above");
        let x = draft.fuse(&emb, &hidden);
        if !kv_only {
            if let (Some(dir), Ok(x)) = (probe_dir(), x.as_ref()) {
                dump_rows(dir, "x", x);
            }
        }
        let first = self.layers.len();
        // The head's layer stands after the model's own for the pass, so the loop that runs the
        // stack - masks, segment hops, the layer's own forward - runs it unchanged.
        draft.layer.kv_only = kv_only;
        self.layers.push(draft.layer);
        self.layer_devs.push(draft.layer_dev.clone());
        let out = x.and_then(|x| self.run_layers_in(x, None, at, n, first, first + 1));
        let layer_dev = self.layer_devs.pop().expect("pushed above");
        draft.layer = self.layers.pop().expect("pushed above");
        draft.layer.kv_only = false;
        draft.layer_dev = layer_dev;
        draft.len = draft.len.max(at + n);
        self.draft = Some(draft);
        if !kv_only {
            if let (Some(dir), Ok((wide, _))) = (probe_dir(), out.as_ref()) {
                dump_rows(dir, "wide", wide);
            }
        }
        out.map(|(wide, _)| wide)
    }

    /// The logits of the rows `wide` `[1, n, streams * dim]` the head wrote, `[n, vocab]`.
    pub fn draft_logits(&self, wide: &Tensor) -> Result<Tensor> {
        let draft = self
            .draft
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("this model has no draft head".to_string()))?;
        let (b, n, _) = wide.dims3()?;
        let rows = wide.to_device(&draft.mixer.device())?;
        let (normed, _) = draft.mixer.mix(&rows)?;
        let normed = normed.reshape((b * n, draft.dim))?;
        let proj = draft
            .vocab
            .as_ref()
            .map(|(_, q)| q)
            .or(self.output_proj_cuda_qmm.as_ref());
        match (proj, &self.output_proj_dev) {
            (Some(proj), Some(dev)) => proj.forward(&normed.to_device(dev)?),
            _ => self.output_proj.forward(&normed.to_device(&Device::Cpu)?),
        }
    }

    /// The head guesses in the first `n` tokens of the vocabulary, `n` chosen for the prompt:
    /// it minimises the share of the projection read, weighted by what the projection costs a
    /// token, plus the share of the prompt's tokens past `n`, which the head could not guess.
    pub fn draft_vocab_for(&mut self, prompt: &[u32]) -> Result<()> {
        let rows = match &self.output_proj_cuda_qmm {
            Some(crate::tensor::quantized::QMatMul::QTensor(q)) => q.shape().dims()[0],
            _ => return Ok(()),
        };
        let n = vocab_cut(prompt, rows, self.head_share);
        tracing::info!(
            "the draft head guesses in the first {n} of {rows} tokens (the projection {:.1} percent of a token's reads)",
            100.0 * self.head_share
        );
        let vocab = if n >= rows {
            None
        } else {
            match self
                .output_proj_cuda_qmm
                .as_ref()
                .and_then(|q| q.row_prefix(n).transpose())
            {
                Some(q) => Some((n, q?)),
                None => None,
            }
        };
        if let Some(d) = self.draft.as_mut() {
            d.vocab = vocab;
        }
        Ok(())
    }

    /// Read a prompt chunk `ids` that starts at `index_pos`, whose last-layer rows are `hidden`
    /// `[1, n, streams * dim]`: the pair `(ids[i + 1], hidden[i])` is what the head reads at its
    /// next position, the chunk's last row waiting for the token that follows it.
    pub(super) fn draft_ingest(
        &mut self,
        ids: &[u32],
        hidden: &Tensor,
        index_pos: usize,
    ) -> Result<()> {
        if index_pos == 0 {
            self.draft_reset();
        }
        let c = ids.len();
        let wide = hidden.elem_count() / c.max(1);
        let hidden = hidden.to_dtype(DType::F32)?.reshape((1, c, wide))?;
        let Some(d) = self.draft.as_mut() else {
            return Ok(());
        };
        let carried = d.carried.take();
        let committed = d.committed;
        let head = hidden.narrow(1, 0, c - 1)?;
        let (tokens, rows) = match carried {
            Some(prev) => (
                ids.to_vec(),
                Tensor::cat(&[&prev.to_device(&head.device())?, &head], 1)?,
            ),
            None => (ids[1..].to_vec(), head),
        };
        let last = hidden.narrow(1, c - 1, 1)?;
        if !tokens.is_empty() {
            // The head's fit to the prompt: the chunk's last rows run whole, their guesses
            // against the tokens that followed them, a measure of the head apart from the
            // reply; the rows before them fill the cache alone.
            let probe = tokens.len().min(DRAFT_FIT_ROWS);
            let probe = if probe >= 2 { probe } else { 0 };
            let from = tokens.len() - probe;
            if let (Some(dir), true) = (probe_dir(), committed == 0) {
                dump_rows(dir, "rows", &rows);
                dump_u32(dir, "ids", ids);
            }
            if from > 0 {
                self.draft_rows(&tokens[..from], &rows.narrow(1, 0, from)?, committed, true)?;
            }
            if probe >= 2 {
                let wide = self.draft_rows(
                    &tokens[from..],
                    &rows.narrow(1, from, probe)?,
                    committed + from,
                    false,
                )?;
                let guessed = self
                    .draft_logits(&wide)?
                    .to_dtype(DType::F32)?
                    .argmax_keepdim(1)?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                if let (Some(dir), true) = (probe_dir(), committed == 0) {
                    dump_u32(dir, "guessed", &guessed);
                }
                let hits = guessed
                    .iter()
                    .zip(&tokens[from + 1..])
                    .filter(|(g, t)| g == t)
                    .count() as u64;
                if let Some(d) = self.draft.as_mut() {
                    d.fit.0 += hits;
                    d.fit.1 += (probe - 1) as u64;
                    tracing::info!(
                        "draft head fit over the prompt: {} of {} rows guessed right",
                        d.fit.0,
                        d.fit.1
                    );
                }
            }
        }
        if let Some(d) = self.draft.as_mut() {
            d.committed = committed + tokens.len();
            d.carried = Some(last);
            d.mark_indexer();
        }
        Ok(())
    }

    /// Start the head's history over: the next pairs it is given are the first it holds.
    pub fn draft_reset(&mut self) {
        self.draft_trim(0);
        if let Some(d) = self.draft.as_mut() {
            d.fit = (0, 0);
            d.committed = 0;
            d.carried = None;
            d.indexer_mark = None;
        }
    }

    /// Up to `k` tokens that follow the sequence, guessed by the head; with `k` zero the head
    /// reads the pairs alone.
    ///
    /// `tokens` are the tokens the model has been given since the last call plus the one it has
    /// yet to be given, and the rows of `last_feature` the hidden states that went before them:
    /// the pair `(tokens[i], hidden[i])` is what the head reads at its position `committed + i`.
    /// What the head guessed ahead the last time is cut off first, and the pairs are read in one
    /// pass; each further guess reads the one before it with the head's own output as hidden.
    pub fn draft_tokens(&mut self, tokens: &[u32], k: usize, min_odds: f32) -> Result<Vec<u32>> {
        let n = tokens.len();
        let Some(feature) = self.last_feature.take() else {
            return Ok(Vec::new());
        };
        let rows = if feature.rank() == 3 {
            feature.dim(1)?
        } else {
            1
        };
        if n == 0 || rows < n {
            return Ok(Vec::new());
        }
        let committed = self.draft.as_ref().map_or(0, |d| d.committed);
        self.draft_trim(committed);
        let feature = if feature.rank() == 3 {
            feature.narrow(1, 0, n)?
        } else {
            feature
        };
        // Under the stage profile, what the pairs, each guess's layer, its logits and its odds
        // cost, with the card drained at each mark.
        let prof = crate::inference::offload::stage_prof_enabled();
        let mark = |name: &'static str, since: &mut std::time::Instant, dev: &Device| {
            if prof {
                let _ = dev.synchronize();
                crate::inference::offload::prof_record(name, since.elapsed().as_nanos() as u64);
                *since = std::time::Instant::now();
            }
        };
        let dev = feature.device().clone();
        let mut since = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        if let Some(out) = self.draft_tokens_recorded(tokens, &feature, committed, k, min_odds)? {
            mark("draft: recorded", &mut since, &dev);
            return Ok(out);
        }
        let wide = self.draft_rows(tokens, &feature, committed, false)?;
        mark("draft: the pairs", &mut since, &dev);
        if let Some(d) = self.draft.as_mut() {
            d.committed = committed + n;
            d.mark_indexer();
        }
        let mut at = committed + n;
        let mut last = wide.narrow(1, n - 1, 1)?;
        let mut out = Vec::with_capacity(k);
        for step in 0..k {
            let logits = self.draft_logits(&last)?.to_dtype(DType::F32)?;
            mark("draft: logits", &mut since, &dev);
            // The token and the odds the head gives it come back from the card in one read: the
            // odds are the top's share of the softmax, one over the mass relative to the top.
            let top = logits.max_keepdim(1)?;
            let mass = logits.broadcast_sub(&top)?.exp()?.sum_keepdim(1)?;
            let token = logits.argmax_keepdim(1)?.to_dtype(DType::F32)?;
            let read = Tensor::cat(&[&token, &mass], 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let token = read[0] as u32;
            // A row of the verify costs what the policy measured: a token the head gives less
            // than those odds is not worth one, nor are the tokens that would follow it.
            let odds = 1.0 / read[1].max(1.0);
            mark("draft: token and odds", &mut since, &dev);
            if odds < min_odds {
                break;
            }
            out.push(token);
            if step + 1 == k {
                break;
            }
            last = self.draft_rows(&[token], &last, at, false)?;
            mark("draft: a guess's layer", &mut since, &dev);
            at += 1;
        }
        Ok(out)
    }

    /// Cut the head's cache back to `len` positions.
    pub fn draft_trim(&mut self, len: usize) {
        if let Some(d) = self.draft.as_mut() {
            d.layer.trim_kv_to(len);
            if let Some(sparse) = d.layer.sparse.as_mut() {
                // The card keeps its buffers: the recorded guesses read them.
                match &d.indexer_mark {
                    Some((at, cache)) if *at == len => sparse.restore_cache(cache, true),
                    _ if len == 0 => sparse.reset(),
                    _ => {}
                }
                #[cfg(feature = "cuda")]
                sparse.card_cut(len);
            }
            d.len = d.len.min(len);
        }
    }
}

#[cfg(feature = "cuda")]
impl GenericHeteroTransformer {
    /// The card the head's layer runs on, when it runs on one and the head's logits are there.
    fn draft_card(&self) -> Option<Device> {
        let d = self.draft.as_ref()?;
        let dev = d.layer.cos.device().clone();
        let head_there = match (&self.output_proj_cuda_qmm, &self.output_proj_dev) {
            (Some(_), Some(head)) => head.same_device(&dev),
            _ => false,
        };
        (dev.is_cuda() && head_there && d.layer.hyper.is_some() && d.layer.ngram.is_none())
            .then_some(dev)
    }

    /// Ready the buffers of recorded rows at `at`: the embeddings of `tokens` (none for the head
    /// alone), the positions, the layer's tables and its room for the rows. `false` when the
    /// guesses are not recorded.
    fn draft_step_prepare(&mut self, tokens: &[u32], at: usize) -> Result<bool> {
        use crate::inference::engine::decode_step::VERIFY_ROWS;
        let Some(dev) = self.draft_card() else {
            return Ok(false);
        };
        if self
            .draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .is_some_and(|s| s.blocked)
        {
            return Ok(false);
        }
        // A head that has read no token yet has no cache to record against: the eager draft
        // writes its first rows.
        if self
            .draft
            .as_ref()
            .is_some_and(|d| d.layer.kv_cache.k_buffer().is_none())
        {
            return Ok(false);
        }
        let rows = tokens.len().max(1);
        // Every position a request can reach.
        let cap = self.config.effective_max_context();
        if rows > VERIFY_ROWS || at + rows > cap {
            return Ok(false);
        }
        // The workspaces the kernels keep are made before a recording, never inside one.
        self.ensure_graph_workspaces();
        let on_card = self.embed_table_on(&dev).is_some();
        let emb = if tokens.is_empty() || on_card {
            None
        } else {
            let edev = self.embeddings.embeddings().device().clone();
            let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &edev)?;
            let mut e = self.embeddings.forward(&ids)?.to_dtype(DType::F32)?;
            if let Some(scale) = self.config.embed_scale {
                e = (e * scale)?;
            }
            Some(e.to_device(&dev)?)
        };
        let tok_ids = match (tokens.is_empty(), on_card) {
            (false, true) => Some(Tensor::from_vec(tokens.to_vec(), (tokens.len(),), &dev)?),
            _ => None,
        };
        let d = self.draft.as_mut().expect("checked above");
        let (streams, dim) = (d.streams, d.dim);
        if d.step.is_none() {
            d.step = Some(DraftStep {
                emb: Tensor::zeros_on((1, VERIFY_ROWS, dim), DType::F32, &dev)?,
                rows_in: Tensor::zeros_on((1, VERIFY_ROWS, streams * dim), DType::F32, &dev)?,
                hidden: Tensor::zeros_on((1, 1, streams * dim), DType::F32, &dev)?,
                ids: Tensor::zeros_on(1, DType::U32, &dev)?,
                read: Tensor::zeros_on((1, 1), DType::F32, &dev)?,
                tok_ids: Tensor::zeros_on(VERIFY_ROWS, DType::U32, &dev)?,
                counter: Tensor::zeros_on(1, DType::U32, &dev)?,
                told: Told::new(&dev)?,
                seen: 0,
                pos: Tensor::zeros_on(VERIFY_ROWS, DType::I32, &dev)?,
                pairs: Vec::new(),
                head: None,
                steps: (0..crate::inference::engine::decode_step::DRAFT_MAX)
                    .map(|_| None)
                    .collect(),
                key: None,
                failures: 0,
                blocked: false,
                spent_us: [0; 5],
                spent_n: 0,
            });
        }
        let step = d.step.as_mut().expect("made above");
        if let Some(e) = emb {
            step.emb.slice_set(&e, 1, 0)?;
        }
        if let Some(t) = tok_ids {
            step.tok_ids.slice_set(&t, 0, 0)?;
        }
        let positions: Vec<i32> = (0..VERIFY_ROWS).map(|r| (at + r) as i32).collect();
        step.pos
            .slice_set(&Tensor::new(&positions[..], &dev)?, 0, 0)?;
        d.layer.verify_pos_i32 = Some(step.pos.clone());
        d.layer.verify_keep_ready(&dev)?;
        let (cos, sin) = d.layer.rope_table().rows(at, VERIFY_ROWS, &dev)?;
        match (&d.layer.verify_rope_cos, &d.layer.verify_rope_sin) {
            (Some(c), Some(s)) => {
                c.slice_set(&cos, 0, 0)?;
                s.slice_set(&sin, 0, 0)?;
            }
            _ => {
                d.layer.verify_rope_cos = Some(cos.force_contiguous()?);
                d.layer.verify_rope_sin = Some(sin.force_contiguous()?);
            }
        }
        // The cache grows to the positions this step reaches, its room doubling as it does, and
        // the indexer's room follows it: the buffers move seldom, and a move ends the recording.
        if d.layer.kv_cache.max_seq_len_padded() < at + rows {
            d.layer.kv_cache.reserve(at + rows)?;
        }
        let room = d.layer.kv_cache.max_seq_len_padded();
        let births = match d.layer.sparse.as_mut() {
            Some(sparse) => {
                sparse.card_room_for(room)?;
                sparse.card_hold_rows(at, rows)?;
                sparse.card_births()
            }
            None => 0,
        };
        // Buffers that moved since the recording end it: the graphs are recorded again.
        let keys = d
            .layer
            .kv_cache
            .k_buffer()
            .ok_or_else(|| crate::tensor::Error::msg("the head's cache holds no token"))?
            .id();
        let rows_guessed = d.vocab.as_ref().map_or(0, |(n, _)| *n);
        let step = d.step.as_mut().expect("made above");
        if step.key != Some((keys, births, rows_guessed)) {
            step.pairs.clear();
            step.head = None;
            step.steps.iter_mut().for_each(|g| *g = None);
            step.key = Some((keys, births, rows_guessed));
        }
        Ok(true)
    }

    /// The kernels of a recorded step of `kind`; the token named and the softmax mass are left
    /// in their buffers by the head.
    fn draft_step_forward(&mut self, kind: DraftKind) -> Result<()> {
        let _order = crate::tensor::quantized::DecodeOrder::enter();
        let _in_place = crate::inference::moe_cuda::in_place_state(true);
        let embed_scale = self.config.embed_scale;
        let table = self.draft_card().and_then(|dev| {
            self.embed_cards
                .iter()
                .find(|t| t.device().same_device(&dev))
                .cloned()
        });
        let d = self
            .draft
            .as_mut()
            .ok_or_else(|| crate::tensor::Error::msg("this model has no draft head".to_string()))?;
        let (emb, rows_in, hidden, ids, read, tok_ids, counter, told_device) = {
            let s = d
                .step
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("the guess was not prepared"))?;
            (
                s.emb.clone(),
                s.rows_in.clone(),
                s.hidden.clone(),
                s.ids.clone(),
                s.read.clone(),
                s.tok_ids.clone(),
                s.counter.clone(),
                s.told.device,
            )
        };
        // The rows of the tokens, decoded on the card from its table where it has one.
        let gather = |ids: &Tensor, into: &Tensor| -> Result<()> {
            if let Some(table) = table.as_ref() {
                crate::inference::moe_cuda::embed_rows_gguf(table, ids, into)?;
                if let Some(scale) = embed_scale {
                    into.slice_set(&(into.clone() * scale)?, 1, 0)?;
                }
            }
            Ok(())
        };
        match kind {
            DraftKind::Pairs(n) => {
                gather(&tok_ids.narrow(0, 0, n)?, &emb.narrow(1, 0, n)?)?;
                let x = d.fuse(&emb.narrow(1, 0, n)?, &rows_in.narrow(1, 0, n)?)?;
                let out = d.layer.forward_wide_graph_rows_last(&x, n)?;
                hidden.slice_set(&out.to_dtype(DType::F32)?, 1, 0)?;
                return Ok(());
            }
            DraftKind::Step(j) => {
                // The token the head named, from the card's own buffer: no host in between.
                if table.is_some() {
                    gather(&ids, &emb.narrow(1, j, 1)?)?;
                } else {
                    crate::tensor::bail!("a recorded step needs the embedding table on the card");
                }
                let x = d.fuse(&emb.narrow(1, j, 1)?, &hidden)?;
                let out = d.layer.forward_wide_graph_rows_at(&x, j, 1)?;
                hidden.slice_set(&out.to_dtype(DType::F32)?, 1, 0)?;
            }
            DraftKind::Head => {}
        }
        let xn = d.mixer.normed_on_card(&hidden)?;
        let (mixed, _) = d.mixer.mix_on_card(&xn, false)?;
        let normed = mixed.reshape((1, d.dim))?;
        let prefix = self
            .draft
            .as_ref()
            .and_then(|d| d.vocab.as_ref())
            .map(|(_, q)| q.clone());
        let logits = match (
            prefix.as_ref().or(self.output_proj_cuda_qmm.as_ref()),
            &self.output_proj_cuda,
        ) {
            (Some(qmm), _) => qmm.forward(&normed)?,
            (None, Some(proj)) => normed.to_dtype(DType::F16)?.matmul_t(proj)?,
            (None, None) => crate::tensor::bail!("a recorded guess needs the card's head"),
        }
        .to_dtype(DType::F32)?;
        crate::inference::moe_cuda::top_and_mass_into(&logits, &ids, &read)?;
        // The token and the mass to the host's mapping, then one more guess told: the host
        // spins on the count and reads the two words behind it.
        let cd = ids.device().as_cuda_device()?;
        let stream = cd.0.stream().cu_stream();
        let host_words = self
            .draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .map(|s| s.told.host)
            .ok_or_else(|| crate::tensor::Error::msg("the guess was not prepared"))?;
        let tell_copy = |from: u64, word: usize| -> Result<()> {
            let r = unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    host_words.add(word) as *mut std::ffi::c_void,
                    from,
                    4,
                    stream,
                )
            };
            if r != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                crate::tensor::bail!("the guess was not told to the host: {r:?}");
            }
            Ok(())
        };
        tell_copy(card_address::<u32>(&ids)?, Told::TOKEN)?;
        tell_copy(card_address::<f32>(&read)?, Told::MASS)?;
        unsafe {
            crate::inference::moe_cuda::tier_graph::tell(
                card_address::<u32>(&counter)? as *mut u32,
                (told_device as *mut u32).wrapping_add(Told::SEQ),
                stream as i64,
            );
        }
        Ok(())
    }

    /// The token the step named and the head's odds on it.
    fn draft_step_read(&mut self) -> Result<(u32, f32)> {
        let dev = self.draft_card().expect("prepared");
        let stream = dev.as_cuda_device()?.0.stream().cu_stream();
        let step = self
            .draft
            .as_mut()
            .and_then(|d| d.step.as_mut())
            .ok_or_else(|| crate::tensor::Error::msg("the guess was not prepared"))?;
        // The card tells each guess by its count: spin until the count moves. Once the stream is
        // idle without it having moved, the words are what they are; a stream in error never
        // moves it, and its error ends the wait.
        let mut spins = 0u32;
        loop {
            let seq = step.told.word(Told::SEQ);
            if seq != step.seen {
                step.seen = seq;
                break;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) {
                use cudarc::driver::sys::cudaError_enum as E;
                match unsafe { cudarc::driver::sys::cuStreamQuery(stream) } {
                    E::CUDA_ERROR_NOT_READY => {}
                    E::CUDA_SUCCESS => {
                        step.seen = step.told.word(Told::SEQ);
                        break;
                    }
                    e => {
                        return Err(crate::tensor::Error::msg(format!(
                            "the guess's stream failed: {e:?}"
                        )))
                    }
                }
            }
            std::hint::spin_loop();
        }
        let token = step.told.word(Told::TOKEN);
        let mass = f32::from_bits(step.told.word(Told::MASS));
        Ok((token, 1.0 / mass.max(1.0)))
    }

    /// The row the last step wrote, on the host.
    fn draft_step_row(&self) -> Result<Vec<f32>> {
        self.draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .ok_or_else(|| crate::tensor::Error::msg("the guess was not prepared"))?
            .hidden
            .flatten_all()?
            .to_vec1::<f32>()
    }

    /// A step of `kind` as a recorded graph: recorded on first use, checked against the eager
    /// run, replayed after. `false` when the graphs are given up on.
    fn draft_step_run(&mut self, kind: DraftKind, at: usize) -> Result<bool> {
        let have = self
            .draft
            .as_ref()
            .and_then(|d| d.step.as_ref())
            .is_some_and(|s| match kind {
                DraftKind::Pairs(n) => s.pairs.iter().any(|(m, _)| *m == n),
                DraftKind::Head => s.head.is_some(),
                DraftKind::Step(j) => s.steps.get(j).is_some_and(Option::is_some),
            });
        if have {
            let step = self
                .draft
                .as_ref()
                .and_then(|d| d.step.as_ref())
                .expect("checked above");
            let g = match kind {
                DraftKind::Pairs(n) => {
                    &step
                        .pairs
                        .iter()
                        .find(|(m, _)| *m == n)
                        .expect("checked above")
                        .1
                }
                DraftKind::Head => step.head.as_ref().expect("checked above"),
                DraftKind::Step(j) => step.steps[j].as_ref().expect("checked above"),
            };
            if let Err(e) = g.0.launch() {
                let step = self
                    .draft
                    .as_mut()
                    .and_then(|d| d.step.as_mut())
                    .expect("checked above");
                step.failures += 1;
                step.pairs.clear();
                step.head = None;
                step.steps.iter_mut().for_each(|g| *g = None);
                step.blocked = step.failures >= DRAFT_RECORD_FAILURES;
                tracing::warn!(
                    "the replay of the draft's {} at pos={at} failed: {e:?}; {}",
                    match kind {
                        DraftKind::Pairs(n) => format!("pairs of {n}"),
                        DraftKind::Head => "head".to_string(),
                        DraftKind::Step(j) => format!("step {j}"),
                    },
                    if step.blocked {
                        "the guesses stay eager"
                    } else {
                        "recorded again"
                    }
                );
                return Ok(false);
            }
            return Ok(true);
        }
        let dev = self.draft_card().expect("prepared");
        let stream = dev.as_cuda_device()?.0.stream().clone();
        // The eager run takes the kernels the recording will take, so that what they keep is
        // made before the recording and the two compute alike.
        // The pairs advance the head's recurrent state in place: the state before the eager
        // run is put back before the replay that proves the recording, so the replay runs
        // the rows once, as the eager run did, and leaves the state the eager run left.
        let state_before = match kind {
            DraftKind::Pairs(_) => {
                let d = self.draft.as_mut().expect("prepared");
                let keep = d.layer.mixer.checkpoint()?;
                if let Some(keep) = keep.as_ref() {
                    d.layer.mixer.save_in_place(keep)?;
                }
                keep
            }
            _ => None,
        };
        // A step reads the hidden row and the token the round before left on the card and
        // writes both: they are put back before the replay that proves the recording.
        let step_before = match kind {
            DraftKind::Step(_) => {
                let s = self
                    .draft
                    .as_ref()
                    .and_then(|d| d.step.as_ref())
                    .expect("prepared");
                Some(((&s.hidden * 1.0)?, (&s.ids.to_dtype(DType::F32)? * 1.0)?))
            }
            _ => None,
        };
        super::expert_tier::kernels_of_a_recording(true);
        let eager = self.draft_step_forward(kind);
        super::expert_tier::kernels_of_a_recording(false);
        eager?;
        // What the eager run left: the token named, or the row written.
        let want_token = match kind {
            DraftKind::Pairs(_) => None,
            _ => Some(self.draft_step_read()?.0),
        };
        let want_row = match kind {
            DraftKind::Pairs(_) => Some(self.draft_step_row()?),
            _ => None,
        };
        let _ = stream.synchronize();
        if let Some(keep) = state_before.as_ref() {
            let d = self.draft.as_mut().expect("prepared");
            d.layer.mixer.restore_in_place(keep)?;
            let _ = stream.synchronize();
        }
        if let Some((hidden, ids)) = step_before.as_ref() {
            self.draft_trim(at);
            let s = self
                .draft
                .as_ref()
                .and_then(|d| d.step.as_ref())
                .expect("prepared");
            s.hidden.slice_set(hidden, 1, 0)?;
            s.ids.slice_set(&ids.to_dtype(DType::U32)?, 0, 0)?;
            let _ = stream.synchronize();
        }
        let captured = (|| -> Result<Option<crate::tensor::cuda_ext::CapturedGraph>> {
            crate::tensor::cuda_ext::begin_capture(&stream)?;
            self.draft_step_forward(kind)?;
            crate::tensor::cuda_ext::end_capture(&stream)
        })();
        let verdict: std::result::Result<crate::tensor::cuda_ext::CapturedGraph, String> =
            match captured {
                Ok(Some(g)) => {
                    let _ = g.upload();
                    // A graph that reads what it writes back is launched once; the others
                    // twice, as a recording that keeps an allocation of its own cannot be
                    // launched again, and says so here rather than later.
                    let launched = match kind {
                        DraftKind::Step(_) | DraftKind::Pairs(_) => g.launch(),
                        DraftKind::Head => g.launch().and_then(|()| g.launch()),
                    };
                    match launched {
                        Ok(()) => {
                            let agrees = match (want_token, want_row) {
                                (Some(t), _) => self.draft_step_read().map(|got| {
                                    (got.0 == t, format!("token {} against {t}", got.0))
                                }),
                                (None, Some(row)) => self.draft_step_row().map(|got| {
                                    let apart = got
                                        .iter()
                                        .zip(&row)
                                        .map(|(a, b)| (a - b).abs())
                                        .fold(0f32, f32::max);
                                    (
                                        crate::inference::engine::decode_step::replay_agrees(
                                            &got, &row,
                                        ),
                                        format!("rows {apart:.3e} apart at most"),
                                    )
                                }),
                                _ => Ok((false, "nothing to compare".to_string())),
                            };
                            match agrees {
                                Ok((true, _)) => Ok(g),
                                Ok((false, what)) => Err(format!(
                                    "the replay of {} does not agree with the eager run: {what}",
                                    match kind {
                                        DraftKind::Pairs(n) => format!("pairs of {n}"),
                                        DraftKind::Head => "the head".to_string(),
                                        DraftKind::Step(j) => format!("step {j}"),
                                    }
                                )),
                                Err(e) => Err(format!("{e}")),
                            }
                        }
                        Err(e) => {
                            let (mut alloc, mut free) = (0usize, 0usize);
                            if let Ok(nodes) = g.nodes(1 << 16) {
                                if let Ok(types) = g.node_types(&nodes) {
                                    for t in types {
                                        match t {
                                            cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => alloc += 1,
                                            cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => free += 1,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            Err(format!(
                                "launch: {e:?} ({alloc} allocations, {free} frees in the graph)"
                            ))
                        }
                    }
                }
                Ok(None) => Err("the capture recorded nothing".into()),
                Err(e) => {
                    let _ = crate::tensor::cuda_ext::end_capture(&stream);
                    let _ = stream.synchronize();
                    Err(format!("capture: {e}"))
                }
            };
        let step = self
            .draft
            .as_mut()
            .and_then(|d| d.step.as_mut())
            .expect("prepared");
        match verdict {
            Ok(g) => {
                let (mut alloc, mut free) = (0usize, 0usize);
                if let Ok(nodes) = g.nodes(1 << 16) {
                    if let Ok(types) = g.node_types(&nodes) {
                        for t in types {
                            match t {
                                cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => alloc += 1,
                                cudarc::driver::sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => free += 1,
                                _ => {}
                            }
                        }
                    }
                }
                tracing::info!(
                    "draft {} recorded at pos={at}: {} nodes, {alloc} allocations, {free} frees",
                    match kind {
                        DraftKind::Pairs(n) => format!("pairs of {n}"),
                        DraftKind::Head => "head".to_string(),
                        DraftKind::Step(j) => format!("step {j}"),
                    },
                    g.num_nodes().unwrap_or(0)
                );
                match kind {
                    DraftKind::Pairs(n) => {
                        step.pairs.retain(|(m, _)| *m != n);
                        if step.pairs.len() >= DRAFT_PAIRS_GRAPHS {
                            step.pairs.remove(0);
                        }
                        step.pairs.push((n, HeldGraph(g)));
                    }
                    DraftKind::Head => step.head = Some(HeldGraph(g)),
                    DraftKind::Step(j) => step.steps[j] = Some(HeldGraph(g)),
                }
                Ok(true)
            }
            Err(why) => {
                tracing::warn!("draft guesses stay eager: {why}");
                step.blocked = true;
                step.pairs.clear();
                step.head = None;
                step.steps.iter_mut().for_each(|g| *g = None);
                Ok(false)
            }
        }
    }

    /// The recorded steps wrote the layer's rows up to `len`: the caches say so.
    fn draft_step_settle(&mut self, len: usize) {
        if let Some(d) = self.draft.as_mut() {
            d.layer.kv_cache.sync_len(len);
            if let Some(sparse) = d.layer.sparse.as_mut() {
                sparse.card_confirm(len);
            }
            d.len = d.len.max(len);
        }
    }

    /// Up to `k` tokens guessed as recorded steps: the pairs `(tokens[i], feature[i])` at
    /// `committed` onward through the layer, then the guesses. `None` when the guesses are not
    /// recorded; the head's bookkeeping is then the caller's.
    fn draft_tokens_recorded(
        &mut self,
        tokens: &[u32],
        feature: &Tensor,
        committed: usize,
        k: usize,
        min_odds: f32,
    ) -> Result<Option<Vec<u32>>> {
        let n = tokens.len();
        let mut spent = [0u64; 5];
        let mut since = std::time::Instant::now();
        let mut mark = |phase: usize, since: &mut std::time::Instant| {
            spent[phase] += since.elapsed().as_micros() as u64;
            *since = std::time::Instant::now();
        };
        if !self.draft_step_prepare(tokens, committed)? {
            return Ok(None);
        }
        {
            let dev = self.draft_card().expect("prepared");
            let wide = feature.elem_count() / n.max(1);
            let rows = feature
                .to_dtype(DType::F32)?
                .reshape((1, n, wide))?
                .to_device(&dev)?;
            let step = self
                .draft
                .as_mut()
                .and_then(|d| d.step.as_mut())
                .expect("prepared");
            step.rows_in.slice_set(&rows, 1, 0)?;
        }
        mark(0, &mut since);
        if !self.draft_step_run(DraftKind::Pairs(n), committed)? {
            return Ok(None);
        }
        let mut at = committed + n;
        self.draft_step_settle(at);
        if let Some(d) = self.draft.as_mut() {
            d.committed = at;
            d.mark_indexer();
        }
        mark(1, &mut since);
        // The steps' positions and rotation rows, staged once for the round: step `j` runs at
        // row `j` from here, its token taken from the guess before it on the card.
        let at0 = at;
        if k > 1 && !self.draft_step_prepare(&[], at0)? {
            return Ok(None);
        }
        mark(0, &mut since);
        let mut out = Vec::with_capacity(k);
        let mut token: Option<u32> = None;
        for j in 0..k {
            let kind = match token {
                None => DraftKind::Head,
                Some(_) => DraftKind::Step(j - 1),
            };
            if !self.draft_step_run(kind, at)? {
                break;
            }
            if let DraftKind::Step(_) = kind {
                at += 1;
                self.draft_step_settle(at);
                mark(4, &mut since);
            } else {
                mark(2, &mut since);
            }
            let (t, odds) = self.draft_step_read()?;
            mark(3, &mut since);
            // A row of the verify costs what the policy measured: a token the head gives less
            // than those odds is not worth one, nor are the tokens that would follow it.
            if odds < min_odds {
                break;
            }
            out.push(t);
            token = Some(t);
        }
        self.draft_step_settle(at);
        if let Some(step) = self.draft.as_mut().and_then(|d| d.step.as_mut()) {
            for (sum, us) in step.spent_us.iter_mut().zip(spent) {
                *sum += us;
            }
            step.spent_n += 1;
            if step.spent_n == DRAFT_REPORT_EVERY {
                let n = u64::from(step.spent_n);
                let t = step.spent_us;
                tracing::info!(
                    "the head's guesses, {n}: preparation {} us, pairs {}, head {}, read {}, steps {}",
                    t[0] / n,
                    t[1] / n,
                    t[2] / n,
                    t[3] / n,
                    t[4] / n
                );
                step.spent_us = [0; 5];
                step.spent_n = 0;
            }
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod vocab_tests {
    use super::vocab_cut;

    /// The cut keeps the prompt's tokens when they are worth the rows they cost, drops a rare
    /// one far down the vocabulary, and keeps everything when the projection costs nothing.
    #[test]
    fn the_cut_weighs_rows_against_the_tokens_past_it() {
        let rows = 1000;
        let mut prompt: Vec<u32> = (0..200).map(|i| (i * 7 % 300) as u32).collect();
        assert_eq!(vocab_cut(&prompt, rows, 0.1), 298);
        prompt.push(900);
        assert_eq!(
            vocab_cut(&prompt, rows, 0.1),
            298,
            "one token past is cheaper than 600 rows"
        );
        assert_eq!(vocab_cut(&prompt, rows, 0.0), rows);
        assert_eq!(vocab_cut(&[], rows, 0.1), rows);
    }
}
