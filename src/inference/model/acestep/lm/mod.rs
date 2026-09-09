//! ACE-Step 1.5 - 5Hz audio-code LM ( M1). The autoregressive Qwen3 causal LM
//! (`acestep-5Hz-lm-4B-Q8_0.gguf`, arch `acestep-lm`) that samples audio codes from
//! the text/lyrics prompt. CPU-Q8 path: weights stay Q8 via `QKernelMatMul::forward_slice_cpu`
//! (the engine's AVX2 dot GEMV - ~4 GB resident, not ~17 GB eager-F32), with a
//! per-token F32 KV cache for O(n)/step decode. Reuses the validated Qwen3 conventions
//! (rms_norm + qk-norm RMS-over-D, NEOX RoPE, GQA). 4B config: H2560, 36L, 32q/8kv,
//! hd128, ffn9728, vocab 217204, θ1e6, tied lm_head, all full (causal) attention.
//!
//! This is the MODEL (prefill/decode -> logits). CFG + top-p/top-k sampling and the
//! decode loop live in the pipeline orchestration (cond+uncond, temp 0.85, cfg 2.0).

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::quantized::QKernelMatMul;
use crate::tensor::{DType, Device, Result, Tensor};
use std::cell::RefCell;

/// Ordinal key for a device (CPU = usize::MAX) - used to key the per-device
/// reusable scratch (RoPE tables, rms-norm `ones`) so a layer on a given GPU
/// reuses one upload instead of rebuilding a host Vec per row/layer/token.
fn dev_key(d: &Device) -> usize {
    match d {
        #[cfg(feature = "cuda")]
        Device::Cuda(c) => c.ordinal(),
        _ => usize::MAX,
    }
}

/// Pre-allocated KV-cache seq capacity (prompt + decoded codes). Matches the
/// `max_ctx` KV reserve horizon used when planning device placement in `from_gguf`.
/// The autoregressive position is asserted to stay within this bound.
const MAX_SEQ: usize = 8192;

/// One Qwen3 decoder layer (Q8 weights) + its growing on-device KV cache.
pub struct LmLayer {
    // Norm weights uploaded ONCE as device tensors (on this layer's `device`):
    // rebuilt-per-token from a host Vec would re-introduce a host↔GPU hop.
    input_ln_t: Tensor, // [1,1,H]
    post_ln_t: Tensor,  // [1,1,H]
    q: QKernelMatMul,
    k: QKernelMatMul,
    v: QKernelMatMul,
    o: QKernelMatMul,
    q_norm_t: Tensor,
    k_norm_t: Tensor, // [1,1,1,d]
    gate: QKernelMatMul,
    up: QKernelMatMul,
    down: QKernelMatMul,
    // Device this layer's weights live on (HeteroPlan segment): GPU0/GPU1/CPU.
    // The activation tensor stays on device across the whole layer; only a
    // cross-segment boundary moves it (once), so per-token host hops are gone.
    device: Device,
}

pub struct Qwen3Lm {
    embed: Tensor, // F32 [V, H] (on `device`) - vocab gather (rows) + tied lm_head matmul
    // Tied lm_head restricted to the only tokens the audio decoder can sample: the
    // contiguous row block `[cand_base, V)` covering EOS + the audio-code range. Decode
    // sampling never looks at the text-vocab prefix, so the per-token lm_head matmul +
    // host readback run over this ~30%-size slice instead of the full vocab. A contiguous
    // row sub-block, so it shares `embed`'s storage (no extra upload). Values identical.
    embed_audio: Tensor, // F32 [V - cand_base, H]
    layers: Vec<LmLayer>,
    norm_t: Tensor, // [1,1,H] final norm on the primary device
    // Single-sequence KV cache (post-RoPE keys/values), ON-DEVICE per layer.
    // Pre-allocated `[1,nkv,MAX_SEQ,d]` buffers written in place at the current
    // position (no per-token realloc/copy or kernel-launch flood); attention reads
    // a `narrow(2,0,len)` view. Lazily allocated on the first token (so a layer on
    // a given device only allocates its buffer there). Kept on `Qwen3Lm` (not
    // `LmLayer`) so the immutable per-layer weights and the mutable cache borrow
    // independently. `cached` = current seq length.
    k_cache: Vec<Option<Tensor>>,
    v_cache: Vec<Option<Tensor>>,
    cached: usize,
    device: Device,
    // Reusable per-device decode scratch (interior-mutable: the hot decode path
    // borrows `&self`). Built lazily on first use per device, then reused for the
    // whole stream - removes the per-row/per-layer/per-token host Vec build +
    // upload that pegged a host core (the launch-bound symptom).
    //   rope: NEOX cos/sin tables `[MAX_SEQ, d/2]` for every absolute position;
    //         a token at `pos` reads `narrow(0,pos,1)` (no host rebuild/upload).
    //   ones: rms-norm reduction vector `ones[n,1]` per distinct n (hidden / d).
    scratch: RefCell<Vec<DevScratch>>,
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// Feed-forward width. Kept because the decode step's transient footprint - and so the
    /// capture arena that has to hold it - is set by it as much as by the hidden size.
    pub ffn: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub eos: u32,
    /// Top-k cutoff for audio-code sampling (0 = disabled; the reference default). Applied
    /// before top-p. Set by the caller before generation.
    pub top_k: usize,
}

/// One section of a continuous style morph (see `generate_cfg_morph`): the caption/lyrics
/// that condition this stretch of the stream, the CoT metadata block, and how many codes
/// to emit before swapping to the next section's conditioning.
pub struct MorphSection<'a> {
    pub caption: &'a str,
    pub lyrics: &'a str,
    pub cot: &'a str,
    pub n_codes: usize,
}

/// True when a formatted error MESSAGE indicates a CUDA out-of-memory condition. The
/// native substrate tags device-allocation OOM (`[oom]` + the driver's `out of memory`
/// text); cuBLAS/cuDNN OOM surfaces as `*_STATUS_ALLOC_FAILED`. The check is on the
/// Display string so it works for any error type (`native::Error`, `Box<dyn Error>`, ...).
pub fn msg_is_oom(s: &str) -> bool {
    let s = s.to_ascii_lowercase();
    s.contains("[oom]")
        || s.contains("out of memory")
        || s.contains("out_of_memory")
        || s.contains("alloc_failed")
        || s.contains("alloc failed")
}

/// Run one render UNIT with the graceful-degradation backstop shared by every generative
/// pipeline (ACE-Step audio, Wan video): if the unit hits CUDA OOM (a concurrent VRAM
/// consumer, a resolution whose activation peak overflows the current split, ...) escalate
/// the process-global degradation level - which makes the unit's next attempt reserve more
/// VRAM headroom and re-plan with a more balanced multi-GPU split (CPU only as the last
/// resort) - then re-run the unit. The unit MUST reload its models so the new placement
/// takes effect. Non-OOM errors propagate immediately. Bounded to a few attempts so a true
/// impossibility (OOM even on CPU) finally errors out instead of looping forever.
pub fn oom_retry<T, E: std::fmt::Display>(
    label: &str,
    mut f: impl FnMut() -> std::result::Result<T, E>,
) -> std::result::Result<T, String> {
    let mut attempt = 0u32;
    loop {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                let msg = e.to_string();
                if attempt < 5 && msg_is_oom(&msg) {
                    attempt += 1;
                    let lvl = crate::inference::place::vram_manager::vram_degrade();
                    let cpu = crate::inference::place::vram_manager::vram_force_cpu();
                    eprintln!(
                        "[oom_retry] {label} hit CUDA OOM - degrading (level {lvl}{}) and retrying",
                        if cpu {
                            ", CPU placement"
                        } else {
                            ", +VRAM reserve"
                        }
                    );
                    continue;
                }
                return Err(format!("{label}: {e}"));
            }
        }
    }
}

/// Reusable per-device decode scratch (one per device a layer lives on).
struct DevScratch {
    key: usize,
    // NEOX cos/sin `[MAX_SEQ, d/2]` for absolute positions 0..MAX_SEQ.
    rope_cos: Tensor,
    rope_sin: Tensor,
    // rms-norm reduction `ones[n,1]` per distinct n (small, keyed by n).
    ones: Vec<(usize, Tensor)>,
}

/// RMS-norm over the LAST dim of `x`, weighted by `w` (broadcast), kept on-device.
/// `x` is any shape `[..,n]`; `w` broadcasts as `[..,n]` (e.g. `[1,1,n]`/`[1,1,1,n]`).
/// `ones` is a reusable `[n,1]` device vector (cached per device/n) used to sum the
/// last dim as a matmul (the native reduce has no CUDA kernel - it would round-trip
/// to host; per rowxlayerxtoken that flood pegs a host core).
fn rms_norm_t(x: &Tensor, w: &Tensor, ones: &Tensor, eps: f32) -> Result<Tensor> {
    // The native substrate now has a FUSED rms_norm CUDA kernel (cuda::rms_norm_f32)  -
    // numerically the same `x/rms(x).w` as the old matmul(ones) reduction, but ONE kernel
    // instead of matmul+affine+sqrt+recip+broadcastx2. Kills ~128 matmul launches/code on the
    // launch-bound audio-LM decode (nsys: 2.89M cuLaunchKernel). `ones` is now unused.
    let _ = ones;
    x.rms_norm(w, eps)
}

/// NEOX RoPE on `x` `[.., d]` given precomputed `cos`/`sin` `[.., d/2]` (broadcastable):
/// out1 = x1.cos - x2.sin, out2 = x1.sin + x2.cos, concatenated. (No `sub` op on the
/// native tensor -> fold the minus into x2 via `affine(-1,0)`.)
fn rope_neox(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Decode hot path (rank-4 [B,heads,seq,head_dim]): the native FUSED NeoX rope kernel
    // (cuda::rope_f32) - same split-half rotation as the hand-rolled chain below but ONE kernel
    // instead of narrowx2+broadcast_mulx2+affine+cat (~7 launches x q,k x 32L x 2CFG per code),
    // cutting the launch-bound audio-LM decode further (cf the fused rms_norm -28% win).
    if x.rank() == 4 {
        return x.rope(cos, sin);
    }
    let last = x.rank() - 1;
    let d = x.dims()[last];
    let half = d / 2;
    let x1 = x.narrow(last, 0, half)?;
    let x2 = x.narrow(last, half, half)?;
    let o1 = x1
        .broadcast_mul(cos)?
        .add(&x2.broadcast_mul(sin)?.affine(-1.0, 0.0)?)?;
    let o2 = x1.broadcast_mul(sin)?.add(&x2.broadcast_mul(cos)?)?;
    Tensor::cat(&[&o1, &o2], last)
}

/// Full NEOX cos/sin tables `[MAX_SEQ, d/2]` on `dev` - built ONCE per device at
/// load/first-use; a token at absolute position `pos` reads row `narrow(0,pos,1)`
/// (no per-token host rebuild + upload). freq_j = θ^(-2j/d); angle = pos.freq_j.
fn rope_tables_full(d: usize, theta: f32, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = d / 2;
    let mut cos = vec![0f32; MAX_SEQ * half];
    let mut sin = vec![0f32; MAX_SEQ * half];
    for pos in 0..MAX_SEQ {
        for j in 0..half {
            let a = pos as f32 * theta.powf(-2.0 * (j as f32) / (d as f32));
            cos[pos * half + j] = a.cos();
            sin[pos * half + j] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec_f32(cos, vec![MAX_SEQ, half])?.to_device(dev)?,
        Tensor::from_vec_f32(sin, vec![MAX_SEQ, half])?.to_device(dev)?,
    ))
}

/// Two independent KV caches (row 0 = conditional, row 1 = unconditional) for batched-CFG
/// decode. Kept OUTSIDE `LmLayer` so the single-sequence `prefill`/`generate` paths are
/// untouched. Per layer, per row: post-RoPE keys + values, grown one token per step.
pub struct CfgKv {
    // Per layer, per row: on-device pre-allocated post-RoPE keys/values
    // [1,nkv,MAX_SEQ,d], written in place at the current position (see `attn_row`).
    k: Vec<[Option<Tensor>; 2]>,
    v: Vec<[Option<Tensor>; 2]>,
    cached: [usize; 2],
    #[cfg(feature = "cuda")]
    graph: Option<DecodeGraph>,
}

impl CfgKv {
    fn new(n_layers: usize) -> Self {
        CfgKv {
            k: (0..n_layers).map(|_| [None, None]).collect(),
            v: (0..n_layers).map(|_| [None, None]).collect(),
            cached: [0, 0],
            #[cfg(feature = "cuda")]
            graph: None,
        }
    }
}

/// A captured CUDA graph for the steady-state B=2 CFG decode step, plus the stable
/// ARENA input buffers refreshed in place per replay. The growing-KV problem is
/// removed by capturing over a FIXED context capacity `cap` (a 256-bucket): the per-
/// token-varying state is only (a) the embedded token `x`, (b) the per-row RoPE
/// cos/sin rows, (c) the per-row KV write offset (a device i32 index buffer the
/// `scatter_set` reads), and (d) the per-row additive attention mask (-inf beyond the
/// row's current length). All four are arena buffers refreshed via `slice_set`; the
/// graph shape is constant within the bucket, so one capture replays for ~256 tokens.
#[cfg(feature = "cuda")]
struct DecodeGraph {
    graph: cudarc::driver::CudaGraph,
    cap: usize,     // captured context capacity (multiple of GRAPH_BUCKET)
    x: Tensor,      // [2,1,H]   embedded token (both rows = same chosen token)
    cos: Tensor,    // [2,1,1,half] per-row RoPE cos
    sin: Tensor,    // [2,1,1,half] per-row RoPE sin
    widx: Tensor,   // [2] i32 KV write offset per row (scatter index)
    mask: Tensor,   // [2,1,1,cap] additive causal/length mask per row
    hidden: Tensor, // [2,1,H]  graph output (final-norm hidden)
}

/// Context-capacity bucket for the decode graph: recapture only when the running
/// length crosses a multiple of this, otherwise replay (refresh buffers + launch).
/// Trade-off - a larger bucket means fewer recaptures (each costs an eager warmup
/// pass) but attention over more (mostly masked) positions every step; the masked
/// compute outweighs the recapture saving here, so a tight bucket wins.
#[cfg(feature = "cuda")]
const GRAPH_BUCKET: usize = 256;

/// Rows one decode step carries: the conditioned and the unconditioned trajectory, batched
/// so each weight is read once. Every transient of a step is this many times a single
/// row's, which is what makes it part of the placement arithmetic rather than a detail of
/// the sampler.
const CFG_ROWS: usize = 2;

/// The checkpoint's geometry: width, heads, KV heads, head size, feed-forward width,
/// layers and vocabulary.
const LM_GEOMETRY: (usize, usize, usize, usize, usize, usize, usize) =
    (2560, 32, 8, 128, 9728, 36, 217204);

/// The context the KV cache is reserved for.
const LM_KV_HORIZON: u64 = 8192;

/// F32 KV bytes one layer holds over the reserve horizon.
fn kv_bytes_per_layer() -> u64 {
    let (_, _, nkv, d, _, _, _) = LM_GEOMETRY;
    2 * nkv as u64 * d as u64 * LM_KV_HORIZON * 4
}

/// What a card must hold to run the language model whole: the weights, the KV cache the
/// plan reserves for every layer, and the step reserve. This is the figure the pressure
/// protocol has to ask for; the weights alone let a card pass that then spilled a third
/// of the layers onto the host.
pub fn placement_demand(path: &str) -> u64 {
    let (h, _, _, _, ffn, n_layers, _) = LM_GEOMETRY;
    let weights = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    weights
        + n_layers as u64 * kv_bytes_per_layer()
        + crate::inference::place::audio_demand::lm_reserve(n_layers, h, ffn, CFG_ROWS)
}

impl Qwen3Lm {
    /// Load the 4B LM (Q8 weights kept quantized) from its GGUF.
    pub fn from_gguf(path: &str) -> Result<Self> {
        let (h, nh, nkv, d, ffn, n_layers, vocab) = LM_GEOMETRY;
        // Standard placement (same mechanism as every other model): build a HeteroPlan
        // from the real free VRAM (pack-first on the fastest GPU, else greedy split,
        // remainder -> CPU) instead of an ad-hoc all-on-GPU0. The per-layer activation is
        // a host Vec<f32>, so CPU-spilled / second-GPU layers need no extra transfer code.
        // A checkpoint that cannot be measured cannot be planned for. This used to answer
        // with a typed 4.2 GB, which let the planner approve a card on a figure nobody
        // had measured and then fail obscurely when the same unreadable file was opened a
        // few lines later. Say it here, where the reason is still in hand.
        let model_size = std::fs::metadata(path).map(|m| m.len()).map_err(|e| {
            crate::tensor::Error(format!(
                "ace-step LM: cannot size the checkpoint {path}: {e}"
            ))
        })?;
        let kv_per_layer = kv_bytes_per_layer(); // F32 KV over the reserve horizon
                                                 // Headroom left free on each card after weights+KV: covers the decode graph
                                                 // arena + transient scratch AND a moderate concurrent VRAM consumer, so
                                                 // placement spills to CPU/another GPU before a post-probe alloc can OOM
                                                 // mid-render. Sized from the step this checkpoint will actually capture rather
                                                 // than fixed - the arena has to hold every layer's transients at once, so a deeper
                                                 // or wider checkpoint needs proportionally more and used to be handed the same
                                                 // figure as the one the reserve was validated on. Ample-VRAM placement at that
                                                 // reference geometry is unchanged (the model still packs on GPU0).
        let reserve: u64 =
            crate::inference::place::audio_demand::lm_reserve(n_layers, h, ffn, CFG_ROWS);
        let mut cudas = crate::inference::place::vram_manager::probe_under_pressure(reserve);
        // SINGLE-CUDA(+CPU) plans only: neither the decode graph (single-device arena) nor the
        // eager per-layer path produces correct output when layers span TWO cuda cards today
        // (graph capture aborts on the cross-device scatter_set; eager runs but yields no
        // valid codes). Keep the fastest card, spill the remainder to CPU - the historically
        // validated topology - until the cross-CUDA decode is actually implemented+validated.
        cudas.truncate(1);
        let cuda_budget: Vec<(usize, u64)> = cudas.iter().map(|(i, f, _)| (*i, *f)).collect();
        let plan = HeteroPlan::calculate_with_kv_reserve(
            // The budgets below ALREADY exclude the reserve: `probe_under_pressure` probes through
            // `probe_cuda_devices`, which returns `stable_free - reserve`. Passing it again here
            // subtracted it TWICE - invisible at half a gigabyte, and fatal at twelve, where it
            // took both cards to zero usable and sent a whole video DiT to the host.
            n_layers,
            model_size,
            &cuda_budget,
            &[],
            1.0,
            kv_per_layer,
            0,
        );
        let dev_of_kind = |k: DeviceKind| -> Device {
            match k {
                DeviceKind::Cuda(idx) => cudas
                    .iter()
                    .find(|(i, _, _)| *i == idx)
                    .map(|(_, _, dv)| dv.clone())
                    .unwrap_or(Device::Cpu),
                _ => Device::Cpu,
            }
        };
        let layer_device = |l: usize| -> Device {
            plan.segments
                .iter()
                .find(|s| l >= s.layer_start && l < s.layer_end)
                .map(|s| dev_of_kind(s.kind))
                .unwrap_or(Device::Cpu)
        };
        let primary = cudas
            .first()
            .map(|(_, _, dv)| dv.clone())
            .unwrap_or(Device::Cpu);
        eprintln!(
            "[ace-lm] placement: {}",
            plan.segments
                .iter()
                .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>()
                .join(" ")
        );

        // CUDA-graph capture of the decode step (the launch-bound lever) requires the
        // per-tensor read/write EVENTS to be off: a captured kernel reading a tensor
        // created with tracking ON would wait on that tensor's pre-capture event ->
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION. Disable tracking on every device used by
        // this model BEFORE any weight/KV/scratch tensor is created, so they are all
        // capture-clean. The whole ace-lm pipeline runs serially on each device's
        // primary stream, so manual cross-stream sync is not needed (same contract as
        // the LLM graph-decode paths). acestep-only - does not touch LLM placement.
        #[cfg(feature = "cuda")]
        for (_, _, dv) in &cudas {
            if let Device::Cuda(c) = dv {
                unsafe {
                    c.context().disable_event_tracking();
                }
                // BLOCKING sync (yield the host core) instead of the default SPIN: the
                // autoregressive decode is serial (each token waits on the GPU before
                // sampling the next), so once the per-token launch flood is replaced by a
                // CUDA-graph replay, the host has nothing to do but wait - a spinning sync
                // would peg a core at ~100% doing nothing. Blocking sync lets that core
                // idle while the GPU runs. acestep-only.
                let _ = c.context().set_blocking_synchronize();
            }
        }

        let vb = crate::inference::cache::qvb::from_gguf_cached(path, &primary)?;
        // Upload a norm weight as an on-device tensor of the given shape (so the
        // per-token rms_norm broadcast-mul stays on device, no host rebuild).
        let norm_t = |name: String, n: usize, shape: Vec<usize>, dev: &Device| -> Result<Tensor> {
            vb.get_f32(n, &name)?.reshape(shape)?.to_device(dev)
        };
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            // Same as the DiT: the mapped builder is instant and this loop is the load.
            // The LM is the larger of the two checkpoints a music render reads, so leaving
            // it uncounted left the longest part of the longest job silent.
            crate::inference::serve::progress::scoped::note(
                crate::inference::serve::progress::phase::LOAD_MODEL,
                l,
                n_layers,
            );
            let p = format!("model.layers.{l}");
            let ldev = layer_device(l);
            let qm = |nm: &str, ind: usize, outd: usize| -> Result<QKernelMatMul> {
                vb.qmatmul_on(ind, outd, &format!("{p}.{nm}.weight"), &ldev)
            };
            layers.push(LmLayer {
                input_ln_t: norm_t(
                    format!("{p}.input_layernorm.weight"),
                    h,
                    vec![1, 1, h],
                    &ldev,
                )?,
                post_ln_t: norm_t(
                    format!("{p}.post_attention_layernorm.weight"),
                    h,
                    vec![1, 1, h],
                    &ldev,
                )?,
                q: qm("self_attn.q_proj", h, nh * d)?,
                k: qm("self_attn.k_proj", h, nkv * d)?,
                v: qm("self_attn.v_proj", h, nkv * d)?,
                o: qm("self_attn.o_proj", nh * d, h)?,
                q_norm_t: norm_t(
                    format!("{p}.self_attn.q_norm.weight"),
                    d,
                    vec![1, 1, 1, d],
                    &ldev,
                )?,
                k_norm_t: norm_t(
                    format!("{p}.self_attn.k_norm.weight"),
                    d,
                    vec![1, 1, 1, d],
                    &ldev,
                )?,
                gate: qm("mlp.gate_proj", h, ffn)?,
                up: qm("mlp.up_proj", h, ffn)?,
                down: qm("mlp.down_proj", ffn, h)?,
                device: ldev.clone(),
            });
        }
        // embed_tokens [V,H] as F32 (vocab gather + tied lm_head) on the primary device.
        let embed = vb.get_f32((vocab, h), "model.embed_tokens.weight")?;
        // Candidate-only lm_head slice: rows [cand_base, V) = EOS (151645) + the audio-code
        // block (AUDIO_CODE_BASE..V). Contiguous row sub-block -> a zero-copy view into embed.
        let cand_base = (151645u32).min(AUDIO_CODE_BASE) as usize;
        // `.contiguous()`: materialize the row block at offset 0 - `matmul_t` reads from the
        // tensor's base pointer, so an offset narrow view would otherwise read the wrong rows.
        let embed_audio = embed
            .narrow(0, cand_base, vocab - cand_base)?
            .contiguous()?;
        let norm_t = norm_t("model.norm.weight".to_string(), h, vec![1, 1, h], &primary)?;
        Ok(Qwen3Lm {
            embed,
            embed_audio,
            layers,
            norm_t,
            k_cache: (0..n_layers).map(|_| None).collect(),
            v_cache: (0..n_layers).map(|_| None).collect(),
            cached: 0,
            scratch: RefCell::new(Vec::new()),
            device: primary,
            hidden: h,
            n_head: nh,
            n_kv: nkv,
            head_dim: d,
            ffn,
            vocab,
            rope_theta: 1e6,
            eos: 151645,
            top_k: 0,
        })
    }

    pub fn reset(&mut self) {
        for c in &mut self.k_cache {
            *c = None;
        }
        for c in &mut self.v_cache {
            *c = None;
        }
        self.cached = 0;
    }

    fn eps(&self) -> f32 {
        1e-6
    }

    /// The NEOX RoPE cos/sin row for absolute position `pos` on `dev`, shaped
    /// `[1,1,1,d/2]` to broadcast against `[1,nh,1,d]` queries. Reuses the per-device
    /// `[MAX_SEQ,d/2]` table (built once), so no host Vec + upload per token.
    fn rope_row(&self, pos: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let key = dev_key(dev);
        {
            let s = self.scratch.borrow();
            if let Some(e) = s.iter().find(|e| e.key == key) {
                let half = self.head_dim / 2;
                let cos = e.rope_cos.narrow(0, pos, 1)?.reshape(vec![1, 1, 1, half])?;
                let sin = e.rope_sin.narrow(0, pos, 1)?.reshape(vec![1, 1, 1, half])?;
                return Ok((cos, sin));
            }
        }
        let (cos, sin) = rope_tables_full(self.head_dim, self.rope_theta, dev)?;
        self.scratch.borrow_mut().push(DevScratch {
            key,
            rope_cos: cos,
            rope_sin: sin,
            ones: Vec::new(),
        });
        self.rope_row(pos, dev)
    }

    /// The reusable rms-norm reduction vector `ones[n,1]` on `dev` (cached per n),
    /// so `rms_norm_t` never allocates a fresh ones-vector per call.
    fn ones_for(&self, n: usize, dev: &Device) -> Result<Tensor> {
        let key = dev_key(dev);
        {
            let s = self.scratch.borrow();
            if let Some(e) = s.iter().find(|e| e.key == key) {
                if let Some((_, t)) = e.ones.iter().find(|(m, _)| *m == n) {
                    return Ok(t.clone());
                }
            }
        }
        // ensure the device's scratch exists (rope_row creates it), then add `ones[n]`.
        let _ = self.rope_row(0, dev)?;
        let ones = Tensor::zeros_on(vec![n, 1], DType::F32, dev)?.affine(0.0, 1.0)?;
        let mut s = self.scratch.borrow_mut();
        let e = s
            .iter_mut()
            .find(|e| e.key == key)
            .expect("scratch present after rope_row");
        e.ones.push((n, ones.clone()));
        Ok(ones)
    }

    /// Per-row GQA causal self-attention, fully ON-DEVICE. `q_r` `[1,nh,1,d]`,
    /// `k_new`/`v_new` `[1,nkv,1,d]` (this token, post-RoPE), `kc`/`vc` the running
    /// per-row caches: pre-allocated `[1,nkv,MAX_SEQ,d]` buffers written in place at
    /// seq offset `pos` (O(1)/token, no realloc-and-copy). Attention reads the
    /// `narrow(2,0,pos+1)` view. Returns the output reshaped to `[1,1,nh.d]`.
    fn attn_row(
        &self,
        q_r: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        pos: usize,
        kc: &mut Option<Tensor>,
        vc: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let nrep = nh / nkv;
        let scale = 1.0 / (d as f32).sqrt();
        assert!(
            pos < MAX_SEQ,
            "ace-lm KV position {pos} exceeds MAX_SEQ {MAX_SEQ}"
        );
        // write this token's K/V into the pre-allocated cache at seq offset `pos`
        // (in-place device memcpy); allocate the buffer lazily on the first token.
        if kc.is_none() {
            let dev = k_new.device();
            let dt = k_new.dtype();
            *kc = Some(Tensor::zeros_on(vec![1, nkv, MAX_SEQ, d], dt, &dev)?);
            *vc = Some(Tensor::zeros_on(vec![1, nkv, MAX_SEQ, d], dt, &dev)?);
        }
        let kbuf = kc.as_ref().unwrap();
        let vbuf = vc.as_ref().unwrap();
        kbuf.slice_set(k_new, 2, pos)?;
        vbuf.slice_set(v_new, 2, pos)?;
        let len = pos + 1;
        let k = kbuf.narrow(2, 0, len)?;
        let v = vbuf.narrow(2, 0, len)?;
        // GQA attention WITHOUT materializing the nrep-repeated KV (the old per-token
        // narrow+cat repeat was a second O(n²) launch flood pegging the host). Group the
        // nh query heads into nkv groups of nrep - query head h reads kv head h/nrep, which
        // is exactly a contiguous reshape [1,nh,1,d] -> [1,nkv,nrep,d] - then run one batched
        // matmul per kv head (batch dim = nkv). Same per-head math/reduction -> bit-identical.
        let o = if nrep == 1 {
            let scores = q_r.matmul_t(&k)?.affine(scale, 0.0)?.softmax_last_dim()?;
            scores.matmul(&v)? // [1,nh,1,d]
        } else {
            let qg = q_r.contiguous()?.reshape(vec![1, nkv, nrep, d])?;
            let kg = k.contiguous()?; // [1,nkv,len,d]
            let vg = v.contiguous()?;
            let scores = qg.matmul_t(&kg)?.affine(scale, 0.0)?.softmax_last_dim()?; // [1,nkv,nrep,len]
            scores.matmul(&vg)?.reshape(vec![1, nh, 1, d])? // [1,nkv,nrep,d] -> [1,nh,1,d]
        };
        // [1,nh,1,d] -> [1,1,nh.d]
        o.transpose(1, 2)?.contiguous()?.reshape(vec![1, 1, nh * d])
    }

    /// CAPTURE-SAFE per-row GQA attention: a FIXED-shape variant of `attn_row` whose
    /// per-token-varying values all come from refreshable DEVICE buffers (so the op
    /// sequence is constant and replays inside a CUDA graph). The K/V write offset is
    /// `widx_r` (a `[1]` i32 index buffer the `scatter_set` reads), and the running
    /// length is encoded in `mask_r` (`[1,1,1,cap]`, additive: 0 for valid positions,
    /// -inf beyond the current token) added to the scores before softmax. Attention
    /// reads the FIXED `narrow(2,0,cap)` view (constant shape). Math is identical to
    /// `attn_row` (masked-out positions contribute 0 after softmax).
    #[cfg(feature = "cuda")]
    fn attn_row_capped(
        &self,
        q_r: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        widx_r: &Tensor,
        mask_r: &Tensor,
        cap: usize,
        kbuf: &Tensor,
        vbuf: &Tensor,
    ) -> Result<Tensor> {
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let nrep = nh / nkv;
        let scale = 1.0 / (d as f32).sqrt();
        // write this token's K/V at the device-resident row offset (scatter along the
        // seq dim, index read from `widx_r` -> no offset baked into the graph).
        kbuf.scatter_set(widx_r, k_new, 2)?;
        vbuf.scatter_set(widx_r, v_new, 2)?;
        let k = kbuf.narrow(2, 0, cap)?;
        let v = vbuf.narrow(2, 0, cap)?;
        let o = if nrep == 1 {
            let scores = q_r
                .matmul_t(&k)?
                .affine(scale, 0.0)?
                .broadcast_add(mask_r)?
                .softmax_last_dim()?;
            scores.matmul(&v)?
        } else {
            let qg = q_r.contiguous()?.reshape(vec![1, nkv, nrep, d])?;
            let kg = k.contiguous()?; // [1,nkv,cap,d]
            let vg = v.contiguous()?;
            // scores [1,nkv,nrep,cap] + mask [1,1,1,cap] (broadcast over kv/nrep)
            let scores = qg
                .matmul_t(&kg)?
                .affine(scale, 0.0)?
                .broadcast_add(mask_r)?
                .softmax_last_dim()?;
            scores.matmul(&vg)?.reshape(vec![1, nh, 1, d])?
        };
        o.transpose(1, 2)?.contiguous()?.reshape(vec![1, 1, nh * d])
    }

    /// CAPTURE-SAFE decoder layer (B=2 CFG): mirrors `decode_layer_ondevice` but uses
    /// `attn_row_capped` with the per-row device buffers (`cos`/`sin`/`widx`/`mask`),
    /// so the whole op sequence is fixed-shape and graph-capturable.
    #[cfg(feature = "cuda")]
    fn decode_layer_graph(
        &self,
        l: &LmLayer,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        widx: &Tensor,
        mask: &Tensor,
        cap: usize,
        kc: &mut [Option<Tensor>],
        vc: &mut [Option<Tensor>],
    ) -> Result<Tensor> {
        let (h, nh, nkv, d) = (self.hidden, self.n_head, self.n_kv, self.head_dim);
        let eps = self.eps();
        let ones_h = self.ones_for(h, &l.device)?;
        let ones_d = self.ones_for(d, &l.device)?;
        let xl = x.to_device(&l.device)?;
        let norm = rms_norm_t(&xl, &l.input_ln_t, &ones_h, eps)?; // [2,1,H]
        let q = l.q.forward(&norm)?;
        let k = l.k.forward(&norm)?;
        let v = l.v.forward(&norm)?;
        let mut attn_rows = Vec::with_capacity(2);
        for r in 0..2 {
            let cos_r = cos.narrow(0, r, 1)?;
            let sin_r = sin.narrow(0, r, 1)?;
            let widx_r = widx.narrow(0, r, 1)?;
            let mask_r = mask.narrow(0, r, 1)?;
            let qr = q.narrow(0, r, 1)?.reshape(vec![1, nh, 1, d])?;
            let qr = rms_norm_t(&qr, &l.q_norm_t, &ones_d, eps)?;
            let qr = rope_neox(&qr, &cos_r, &sin_r)?;
            let kr = k.narrow(0, r, 1)?.reshape(vec![1, nkv, 1, d])?;
            let kr = rms_norm_t(&kr, &l.k_norm_t, &ones_d, eps)?;
            let kr = rope_neox(&kr, &cos_r, &sin_r)?;
            let vr = v.narrow(0, r, 1)?.reshape(vec![1, nkv, 1, d])?;
            let kbuf = kc[r]
                .as_ref()
                .expect("graph KV buffer must be pre-allocated");
            let vbuf = vc[r]
                .as_ref()
                .expect("graph KV buffer must be pre-allocated");
            let ao = self.attn_row_capped(&qr, &kr, &vr, &widx_r, &mask_r, cap, kbuf, vbuf)?;
            attn_rows.push(ao);
        }
        let aref: Vec<&Tensor> = attn_rows.iter().collect();
        let attn = Tensor::cat(&aref, 0)?; // [2,1,nh.d]
        let ao = l.o.forward(&attn)?;
        let xl = xl.add(&ao)?;
        let norm2 = rms_norm_t(&xl, &l.post_ln_t, &ones_h, eps)?;
        let g = l.gate.forward(&norm2)?;
        let u = l.up.forward(&norm2)?;
        let ff = g.silu()?.mul(&u)?;
        let down = l.down.forward(&ff)?;
        let xl = xl.add(&down)?;
        xl.to_device(&self.device)
    }

    /// Run the B=2 CFG decode through all layers with the capture-safe buffers, then
    /// the final norm -> hidden `[2,1,H]`. The `kc`/`vc` per-layer KV buffers must be
    /// pre-allocated (the prefill path does this) before this runs under capture.
    #[cfg(feature = "cuda")]
    fn forward_graph(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        widx: &Tensor,
        mask: &Tensor,
        cap: usize,
        kc: &mut [[Option<Tensor>; 2]],
        vc: &mut [[Option<Tensor>; 2]],
    ) -> Result<Tensor> {
        let mut xl = x.clone();
        for (li, l) in self.layers.iter().enumerate() {
            xl = self.decode_layer_graph(
                l,
                &xl,
                cos,
                sin,
                widx,
                mask,
                cap,
                &mut kc[li],
                &mut vc[li],
            )?;
        }
        let ones_h = self.ones_for(self.hidden, &self.device)?;
        rms_norm_t(&xl, &self.norm_t, &ones_h, self.eps())
    }

    /// ONE decoder layer, activation kept ON-DEVICE. `x` is `[B,1,H]` on the PRIMARY
    /// device (B = number of CFG rows). `positions[r]` is row r's absolute position;
    /// `kc`/`vc` are this layer's per-row caches (`[B][...]`). The Q8 projections run
    /// batched (`QKernelMatMul::forward` reads each weight once for the whole batch, on the
    /// layer's device); qk-norm + NEOX RoPE + GQA attention run per row. Returns `[B,1,H]`.
    fn decode_layer_ondevice(
        &self,
        l: &LmLayer,
        x: &Tensor,
        positions: &[usize],
        kc: &mut [Option<Tensor>],
        vc: &mut [Option<Tensor>],
    ) -> Result<Tensor> {
        let (h, nh, nkv, d) = (self.hidden, self.n_head, self.n_kv, self.head_dim);
        let b = x.dims()[0];
        let eps = self.eps();
        // Reusable rms-norm reduction vectors on this layer's device (cached): one for
        // the hidden norms (n=H), one for the qk-norms (n=d). Built once, not per call.
        let ones_h = self.ones_for(h, &l.device)?;
        let ones_d = self.ones_for(d, &l.device)?;
        // move the activation to this layer's device (HeteroPlan segment) - once.
        let xl = x.to_device(&l.device)?;
        // --- self-attn ---
        let norm = rms_norm_t(&xl, &l.input_ln_t, &ones_h, eps)?; // [B,1,H]
        let q = l.q.forward(&norm)?; // [B,1,nh.d]
        let k = l.k.forward(&norm)?; // [B,1,nkv.d]
        let v = l.v.forward(&norm)?; // [B,1,nkv.d]
        let mut attn_rows = Vec::with_capacity(b);
        for r in 0..b {
            let (cos, sin) = self.rope_row(positions[r], &l.device)?;
            // q row -> [1,nh,1,d], qk-norm over d, RoPE.
            let qr = q.narrow(0, r, 1)?.reshape(vec![1, nh, 1, d])?;
            let qr = rms_norm_t(&qr, &l.q_norm_t, &ones_d, eps)?;
            let qr = rope_neox(&qr, &cos, &sin)?;
            // k/v row -> [1,nkv,1,d].
            let kr = k.narrow(0, r, 1)?.reshape(vec![1, nkv, 1, d])?;
            let kr = rms_norm_t(&kr, &l.k_norm_t, &ones_d, eps)?;
            let kr = rope_neox(&kr, &cos, &sin)?;
            let vr = v.narrow(0, r, 1)?.reshape(vec![1, nkv, 1, d])?;
            let ao = self.attn_row(&qr, &kr, &vr, positions[r], &mut kc[r], &mut vc[r])?; // [1,1,nh.d]
            attn_rows.push(ao);
        }
        let aref: Vec<&Tensor> = attn_rows.iter().collect();
        let attn = Tensor::cat(&aref, 0)?; // [B,1,nh.d]
        let ao = l.o.forward(&attn)?; // [B,1,H]
        let xl = xl.add(&ao)?;
        // --- MLP (SwiGLU) ---
        let norm2 = rms_norm_t(&xl, &l.post_ln_t, &ones_h, eps)?;
        let g = l.gate.forward(&norm2)?;
        let u = l.up.forward(&norm2)?;
        let ff = g.silu()?.mul(&u)?;
        let down = l.down.forward(&ff)?;
        let xl = xl.add(&down)?;
        // back to the primary device for the next layer / final norm.
        let _ = h;
        xl.to_device(&self.device)
    }

    /// Run `x` `[B,1,H]` through all layers with the given per-row positions and
    /// per-layer/per-row caches, then the final norm. Returns the hidden `[B,1,H]`.
    fn forward_ondevice(
        &self,
        x: &Tensor,
        positions: &[usize],
        kc: &mut [Vec<Option<Tensor>>],
        vc: &mut [Vec<Option<Tensor>>],
    ) -> Result<Tensor> {
        let mut xl = x.clone();
        for (li, l) in self.layers.iter().enumerate() {
            xl = self.decode_layer_ondevice(l, &xl, positions, &mut kc[li], &mut vc[li])?;
        }
        let ones_h = self.ones_for(self.hidden, &self.device)?;
        rms_norm_t(&xl, &self.norm_t, &ones_h, self.eps())
    }

    /// Gather the F32 embedding row for `token` as an on-device tensor `[1,1,H]`.
    fn embed_row(&self, token: u32) -> Result<Tensor> {
        self.embed
            .narrow(0, token as usize, 1)?
            .reshape(vec![1, 1, self.hidden])
    }

    /// Prefill the prompt `tokens` (builds the single-seq KV cache); returns the hidden
    /// state `[H]` after the last token. Call `reset()` first for a fresh sequence.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut last = None;
        for &t in tokens {
            let row = self.embed_row(t)?;
            last = Some(self.forward_single(&row)?);
        }
        Ok(last.unwrap_or_default())
    }

    /// One token through all layers into the single-sequence cache -> hidden `[H]`.
    /// `pos` = current cache length (absolute position of this token).
    fn forward_single(&mut self, embed_row: &Tensor) -> Result<Vec<f32>> {
        let pos = self.cached;
        // Temporarily move the caches out so the immutable-self forward can borrow them.
        let mut kc = std::mem::take(&mut self.k_cache)
            .into_iter()
            .map(|c| vec![c])
            .collect::<Vec<_>>();
        let mut vc = std::mem::take(&mut self.v_cache)
            .into_iter()
            .map(|c| vec![c])
            .collect::<Vec<_>>();
        let out = self.forward_ondevice(embed_row, &[pos], &mut kc, &mut vc);
        self.k_cache = kc.into_iter().map(|mut v| v.pop().flatten()).collect();
        self.v_cache = vc.into_iter().map(|mut v| v.pop().flatten()).collect();
        let h = out?;
        self.cached += 1;
        Ok(h.reshape(vec![self.hidden])?.to_vec_f32())
    }

    /// One token into an EXTERNAL per-row cache (`kv` row `row`) -> hidden `[H]`. Math
    /// identical to the single-seq path; the cache lives in `CfgKv` so the two CFG
    /// sequences are kept apart. Used to prefill cond (row 0) + uncond (row 1).
    fn forward_into_kv(&self, row: usize, embed_row: &Tensor, kv: &mut CfgKv) -> Result<Vec<f32>> {
        let pos = kv.cached[row];
        let mut kc: Vec<Vec<Option<Tensor>>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, _)| vec![kv.k[li][row].take()])
            .collect();
        let mut vc: Vec<Vec<Option<Tensor>>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, _)| vec![kv.v[li][row].take()])
            .collect();
        let out = self.forward_ondevice(embed_row, &[pos], &mut kc, &mut vc);
        for (li, mut c) in kc.into_iter().enumerate() {
            kv.k[li][row] = c.pop().flatten();
        }
        for (li, mut c) in vc.into_iter().enumerate() {
            kv.v[li][row] = c.pop().flatten();
        }
        let h = out?;
        kv.cached[row] += 1;
        Ok(h.reshape(vec![self.hidden])?.to_vec_f32())
    }

    /// Batched (B=2) decode step: both rows feed the SAME chosen token at their own
    /// positions/caches (`kv`). Each layer's projections run batched -> ONE weight read
    /// for cond+uncond. Per-row qk-norm/RoPE/attention mirror the single-seq path, so each
    /// row equals the serial path (modulo the GPU batched-MMVQ rounding). The hidden state
    /// stays ON-DEVICE (`[2,1,H]`) - no host readback; the caller computes logits straight
    /// from the tensor (`logits_t`), so the per-token decode does only ONE host transfer
    /// (the final logits), not a hidden download + a re-upload per row.
    fn forward_b2_t(&self, rows2: &Tensor, kv: &mut CfgKv) -> Result<Tensor> {
        let pos = [kv.cached[0], kv.cached[1]];
        let mut kc: Vec<Vec<Option<Tensor>>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, _)| vec![kv.k[li][0].take(), kv.k[li][1].take()])
            .collect();
        let mut vc: Vec<Vec<Option<Tensor>>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, _)| vec![kv.v[li][0].take(), kv.v[li][1].take()])
            .collect();
        let out = self.forward_ondevice(rows2, &pos, &mut kc, &mut vc);
        for (li, mut c) in kc.into_iter().enumerate() {
            kv.k[li][1] = c.pop().flatten();
            kv.k[li][0] = c.pop().flatten();
        }
        for (li, mut c) in vc.into_iter().enumerate() {
            kv.v[li][1] = c.pop().flatten();
            kv.v[li][0] = c.pop().flatten();
        }
        let hidden = out?; // [2,1,H]
        kv.cached[0] += 1;
        kv.cached[1] += 1;
        Ok(hidden)
    }

    /// First token id covered by the candidate-only lm_head slice (`embed_audio`): EOS or
    /// the audio-code base, whichever is lower. Candidate logits are indexed `id - cand_base`.
    fn cand_base(&self) -> usize {
        (self.eos as usize).min(AUDIO_CODE_BASE as usize)
    }

    /// Candidate-only tied lm_head from an ON-DEVICE hidden `[..,H]`: logits over the
    /// `[cand_base, V)` slice only (EOS + audio codes), `[rows.(V-cand_base)]` on host.
    /// ~70% less matmul + readback than `logits_t`; the text-vocab prefix is never sampled.
    fn logits_audio_t(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let rows = hidden.elem_count() / self.hidden;
        let x = hidden
            .reshape(vec![rows, self.hidden])?
            .to_device(&self.device)?;
        Ok(x.matmul_t(&self.embed_audio)?.to_vec_f32())
    }

    /// Candidate-only lm_head from a HOST hidden `[H]` (the prefill output) - same slice
    /// as `logits_audio_t`, used to seed the first decode step before the on-device loop.
    fn logits_audio(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let x = Tensor::from_vec_f32(hidden.to_vec(), (1, self.hidden))?.to_device(&self.device)?;
        Ok(x.matmul_t(&self.embed_audio)?.to_vec_f32())
    }

    /// Build the capture-safe per-token DEVICE input buffers for the B=2 CFG decode at
    /// positions `pos` (row 0 / row 1) for a captured context capacity `cap`:
    ///   x    [2,1,H]      embed of `chosen` (both rows identical)
    ///   cos/sin [2,1,1,half]   per-row RoPE row (pos-dependent)
    ///   widx [2,nkv,1,d]  per-row scatter index (all = pos_r) - the KV write offset
    ///   mask [2,1,1,cap]  additive mask, 0 for j<=pos_r else -inf (the running length)
    #[cfg(feature = "cuda")]
    fn graph_inputs(
        &self,
        chosen: u32,
        pos: [usize; 2],
        cap: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let (h, nkv, d) = (self.hidden, self.n_kv, self.head_dim);
        let half = d / 2;
        let dev = &self.device;
        let row = self.embed_row(chosen)?; // [1,1,H]
        let x = Tensor::cat(&[&row, &row], 0)?; // [2,1,H]
                                                // per-row RoPE rows (gathered from the precomputed table), stacked.
        let mut cos_rows = Vec::with_capacity(2);
        let mut sin_rows = Vec::with_capacity(2);
        for r in 0..2 {
            let (c, s) = self.rope_row(pos[r], dev)?; // [1,1,1,half]
            cos_rows.push(c);
            sin_rows.push(s);
        }
        let cos = Tensor::cat(&[&cos_rows[0], &cos_rows[1]], 0)?; // [2,1,1,half]
        let sin = Tensor::cat(&[&sin_rows[0], &sin_rows[1]], 0)?;
        let _ = (h, half);
        // scatter index: [2,nkv,1,d], each row filled with its write offset pos_r.
        let mut widx = vec![0i64; 2 * nkv * d];
        for r in 0..2 {
            for e in 0..nkv * d {
                widx[r * nkv * d + e] = pos[r] as i64;
            }
        }
        let widx = Tensor::from_vec_i64(widx, vec![2, nkv, 1, d])?.to_device(dev)?;
        // additive mask: [2,1,1,cap], 0 for valid (j <= pos_r), -inf beyond.
        let mut mask = vec![0f32; 2 * cap];
        for r in 0..2 {
            for j in 0..cap {
                if j > pos[r] {
                    mask[r * cap + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec_f32(mask, vec![2, 1, 1, cap])?.to_device(dev)?;
        Ok((x, cos, sin, widx, mask))
    }

    /// Captured-graph B=2 CFG decode step. Mirrors `continuous_serve`'s capture/replay:
    /// capture once per ctx-capacity bucket (warmup eager -> this token's real result),
    /// then replay (refresh the arena input buffers + launch) for the rest of the bucket
    /// - removing the per-token kernel-launch flood that pegs a host core. The growing
    /// KV is handled by a fixed `cap` window + a refreshed length mask (see `graph_inputs`
    /// / `attn_row_capped`). Returns hidden `[2,1,H]` on-device. On ANY capture/replay
    /// error the caller falls back to the eager `forward_b2_t`.
    #[cfg(feature = "cuda")]
    fn forward_b2_graph(&self, chosen: u32, kv: &mut CfgKv) -> Result<Tensor> {
        use cudarc::driver::sys::{CUgraphInstantiate_flags_enum, CUstreamCaptureMode};
        // The decode graph is SINGLE-DEVICE by construction (its arena buffers live on the
        // primary device and every captured op must run there). Under a hetero plan that
        // spilled layers onto another CUDA card, a capture attempt aborts MID-STREAM on the
        // first cross-device op (scatter_set) - and an aborted capture leaves the stream in a
        // corrupted capture state that makes the SUBSEQUENT eager fallback fault with
        // CUDA_ILLEGAL_ADDRESS. Refuse the graph up front; the eager path migrates
        // activations per layer and handles multi-device correctly.
        if self
            .layers
            .iter()
            .any(|l| l.device.location() != self.device.location())
        {
            return Err(crate::tensor::Error(
                "graph: layers span multiple devices (eager only)".into(),
            ));
        }
        let err = |s: String| crate::tensor::Error(s);
        // The native CUDA device backing this model (the whole ace-lm stack is native).
        let cd = match &self.device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(err("graph: model not on CUDA".into())),
        };
        let dev = self.device.clone();
        let ctx = cd.context().clone();
        let stream = cd.stream().clone();
        let pos = [kv.cached[0], kv.cached[1]];
        let cap = (pos.iter().copied().max().unwrap() + 1).div_ceil(GRAPH_BUCKET) * GRAPH_BUCKET;
        if cap > MAX_SEQ {
            return Err(err(format!("graph: cap {cap} > MAX_SEQ {MAX_SEQ}")));
        }
        // KV buffers MUST already exist (lazy alloc inside capture is illegal). The
        // prefill wrote tokens into them, so they do - verify before proceeding.
        for li in 0..self.layers.len() {
            for r in 0..2 {
                if kv.k[li][r].is_none() || kv.v[li][r].is_none() {
                    return Err(err("graph: KV not pre-allocated".into()));
                }
            }
        }
        let (x, cos, sin, widx, mask) = self.graph_inputs(chosen, pos, cap)?;

        // -- replay (same bucket) --
        if let Some(g) = kv.graph.as_ref() {
            if g.cap == cap {
                g.x.slice_set(&x, 0, 0)?;
                g.cos.slice_set(&cos, 0, 0)?;
                g.sin.slice_set(&sin, 0, 0)?;
                g.widx.slice_set(&widx, 0, 0)?;
                g.mask.slice_set(&mask, 0, 0)?;
                stream
                    .synchronize()
                    .map_err(|e| err(format!("sync: {e}")))?;
                g.graph
                    .launch()
                    .map_err(|e| err(format!("graph launch: {e:?}")))?;
                stream
                    .synchronize()
                    .map_err(|e| err(format!("sync: {e}")))?;
                let hidden = g.hidden.clone();
                kv.cached[0] += 1;
                kv.cached[1] += 1;
                return Ok(hidden);
            }
            // bucket changed -> drop the old graph + arena, recapture below.
            kv.graph = None;
            ctx.free_capture_arena();
        }

        // -- capture (new bucket) --
        // Pin a cuBLAS workspace ONCE (process lifetime) so the in-graph F32 attention-
        // score gemms don't grab a per-call workspace during capture (an in-capture
        // cuBLAS alloc invalidates it). Guarded so recaptures don't leak a workspace.
        static CUBLAS_WS_PINNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        if CUBLAS_WS_PINNED.get().is_none() {
            use cudarc::cublas::sys::cublasSetWorkspace_v2;
            use cudarc::driver::result::malloc_sync;
            // Sized from the bucket the graph is captured at: the workspace backs the
            // score GEMMs of one step, so a wider bucket means wider score matrices and
            // more split-k staging behind them. A fixed figure is right for one bucket.
            let bytes =
                crate::inference::place::audio_demand::lm_cublas_workspace_bytes(GRAPH_BUCKET)
                    as usize;
            let raw = *cd.blas().map_err(|e| err(e.0))?.handle();
            let ws = unsafe { malloc_sync(bytes) }
                .map_err(|e| err(format!("cublas ws malloc: {e:?}")))?;
            unsafe { cublasSetWorkspace_v2(raw, ws as *mut std::ffi::c_void, bytes) }
                .result()
                .map_err(|e| err(format!("cublas set ws: {e:?}")))?;
            let _ = CUBLAS_WS_PINNED.set(());
        }
        // WARMUP eager pass: loads every kernel module (lazy load in capture is illegal)
        // AND computes THIS token's real KV write + hidden (capture only RECORDS).
        let eager_hidden =
            self.forward_graph(&x, &cos, &sin, &widx, &mask, cap, &mut kv.k, &mut kv.v)?;
        stream
            .synchronize()
            .map_err(|e| err(format!("sync: {e}")))?;
        // Arm a modest arena (decode activations are tiny) and re-home the stable input
        // buffers into it so their VAs stay stable across replays.
        let free = ctx.mem_get_info().map(|(f, _)| f).unwrap_or(0);
        // The arena holds EVERY transient alloc of the decode pass at once (it never frees
        // mid-capture), so it is a function of the LAYER COUNT and the width, not a fixed
        // slab: the family ships more than one checkpoint, and the deeper one was being
        // handed the shallower one's arena. Still bounded by what is free, because an
        // arena that cannot be allocated is worse than a small one.
        let arena_cap = crate::inference::place::audio_demand::lm_graph_arena_bytes(
            self.layers.len(),
            self.hidden,
            self.ffn,
            CFG_ROWS,
        ) as usize;
        // What to try when half of free VRAM is less than the arena wants: a share of the
        // arena rather than a fixed slab, so the fallback follows the model too.
        let arena_floor =
            arena_cap / crate::inference::place::audio_demand::ARENA_HEADROOM_SHARE as usize;
        let mut arena_bytes = arena_cap.min((free / 2).max(arena_floor));
        // Arena alloc can OOM if a concurrent consumer grabbed VRAM after the free
        // probe. Shrink-retry before giving up; if even a small arena won't fit, bubble
        // the error so the decode loop falls back to the eager (arena-free) path.
        loop {
            eprintln!(
                "[ace-lm] graph arena {}MB (free {}MB)",
                arena_bytes >> 20,
                free >> 20
            );
            match ctx.begin_capture_arena(arena_bytes) {
                Ok(()) => break,
                Err(e) => {
                    let oom = format!("{e:?}")
                        .to_ascii_lowercase()
                        .contains("out of memory");
                    if oom
                        && arena_bytes
                            > arena_cap
                                / crate::inference::place::audio_demand::ARENA_FLOOR_SHARE as usize
                    {
                        arena_bytes /= 2;
                        eprintln!(
                            "[ace-lm] graph arena alloc OOM - retrying smaller ({}MB)",
                            arena_bytes >> 20
                        );
                        continue;
                    }
                    let tag = if oom { "[oom] " } else { "" };
                    return Err(err(format!("{tag}begin arena: {e:?}")));
                }
            }
        }
        let arena_copy = |t: &Tensor| -> Result<Tensor> {
            let a = Tensor::zeros_on(t.dims().to_vec(), t.dtype(), &dev)?;
            a.slice_set(t, 0, 0)?;
            Ok(a)
        };
        let x = arena_copy(&x)?;
        let cos = arena_copy(&cos)?;
        let sin = arena_copy(&sin)?;
        let widx = arena_copy(&widx)?;
        let mask = arena_copy(&mask)?;
        // Settle ALL pre-capture stream work (arena copies, input prep) so the captured
        // region has no live dependency on uncaptured same-/cross-stream work
        // (CUDA_ERROR_STREAM_CAPTURE_ISOLATION otherwise).
        stream
            .synchronize()
            .map_err(|e| err(format!("sync: {e}")))?;
        let inst_flags: CUgraphInstantiate_flags_enum = unsafe { std::mem::transmute(0u32) };
        let mut hidden_out: Option<Tensor> = None;
        let cap_res: std::result::Result<_, crate::tensor::Error> = (|| {
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| err(format!("begin_capture: {e:?}")))?;
            let hf = self.forward_graph(&x, &cos, &sin, &widx, &mask, cap, &mut kv.k, &mut kv.v)?;
            hidden_out = Some(hf);
            stream
                .end_capture(inst_flags)
                .map_err(|e| err(format!("end_capture: {e:?}")))?
                .ok_or_else(|| err("graph capture: null graph".into()))
        })();
        let (_peak, overflow) = ctx.end_capture_arena();
        let graph = match cap_res {
            Ok(g) if overflow == 0 => {
                eprintln!(
                    "[ace-lm] graph captured (cap={cap}) arena peak {}MB",
                    _peak >> 20
                );
                g
            }
            Ok(g) => {
                drop(g);
                ctx.free_capture_arena();
                return Err(err(format!(
                    "graph capture: arena overflow ({}MB over, peak {}MB)",
                    overflow >> 20,
                    _peak >> 20
                )));
            }
            Err(e) => {
                let _ = stream.end_capture(inst_flags);
                let _ = stream.synchronize();
                ctx.free_capture_arena();
                return Err(e);
            }
        };
        let hidden = hidden_out.ok_or_else(|| err("graph: hidden not produced".into()))?;
        let out = eager_hidden; // this token's result is the eager warmup's output
        kv.graph = Some(DecodeGraph {
            graph,
            cap,
            x,
            cos,
            sin,
            widx,
            mask,
            hidden,
        });
        kv.cached[0] += 1;
        kv.cached[1] += 1;
        Ok(out)
    }

    /// Decode one step: feed `token` at the next position -> hidden `[H]`.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        let row = self.embed_row(token)?;
        self.forward_single(&row)
    }

    /// Tied lm_head: hidden `[H]` -> logits `[V]` (= embed . hiddenᵀ).
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let x = Tensor::from_vec_f32(hidden.to_vec(), (1, self.hidden))?.to_device(&self.device)?;
        Ok(x.matmul_t(&self.embed)?.to_vec_f32()) // [1,H].[V,H]ᵀ = [1,V]
    }

    /// Generate audio codes from a caption + lyrics (cond-only, no CFG yet).
    /// `cot_yaml` is the metadata block (build_cot_yaml). Samples over the audio-code
    /// vocab [AUDIO_CODE_BASE, +AUDIO_CODE_COUNT) + EOS with temperature/top-p until
    /// EOS or `max_codes`. Returns FSQ code indices (tok - AUDIO_CODE_BASE).
    pub fn generate(
        &mut self,
        tok: &tokenizers::Tokenizer,
        caption: &str,
        lyrics: &str,
        cot_yaml: &str,
        max_codes: usize,
        seed: u64,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        let prompt = build_lm_prompt_with_cot(tok, caption, lyrics, cot_yaml)?;
        self.reset();
        let mut hidden = self.prefill(&prompt)?;
        let mut codes = Vec::new();
        let mut rng = seed.max(1);
        loop {
            let logits = self.logits(&hidden)?;
            let chosen = sample_audio(
                &logits,
                0,
                self.eos,
                temperature,
                top_p,
                self.top_k,
                &mut rng,
            );
            if chosen == self.eos {
                break;
            }
            codes.push(chosen - AUDIO_CODE_BASE);
            if codes.len() >= max_codes {
                break;
            }
            hidden = self.step(chosen)?;
        }
        Ok(codes)
    }

    /// Classifier-free-guidance generation (oracle lm_cfg_scale, default 2.0): runs a
    /// conditional pass (this model) and an unconditional pass (`uncond`, a 2nd instance
    /// with separate KV state, prompted with the bare negative prompt + empty CoT), and
    /// combines `logit = uncond + cfg.(cond - uncond)` per step before sampling. CFG is
    /// what makes the model actually follow the lyrics (cond-only renders instrumental).
    /// `cfg_scale <= 1.0` falls back to cond-only.
    pub fn generate_cfg(
        &mut self,
        uncond: &mut Qwen3Lm,
        tok: &tokenizers::Tokenizer,
        caption: &str,
        lyrics: &str,
        cot_yaml: &str,
        neg: &str,
        max_codes: usize,
        seed: u64,
        temperature: f32,
        top_p: f32,
        cfg_scale: f32,
    ) -> Result<Vec<u32>> {
        if cfg_scale <= 1.0 {
            return self.generate(
                tok,
                caption,
                lyrics,
                cot_yaml,
                max_codes,
                seed,
                temperature,
                top_p,
            );
        }
        let cond_prompt = build_lm_prompt_with_cot(tok, caption, lyrics, cot_yaml)?;
        let uncond_prompt = build_lm_prompt_uncond(tok, neg)?;
        self.reset();
        uncond.reset();
        let mut hc = self.prefill(&cond_prompt)?;
        let mut hu = uncond.prefill(&uncond_prompt)?;
        let mut codes = Vec::new();
        let mut rng = seed.max(1);
        loop {
            let lc = self.logits(&hc)?;
            let lu = uncond.logits(&hu)?;
            let chosen = sample_audio_cfg(
                &lc,
                &lu,
                0,
                self.eos,
                cfg_scale,
                temperature,
                top_p,
                self.top_k,
                &mut rng,
                false,
            );
            if chosen == self.eos {
                break;
            }
            codes.push(chosen - AUDIO_CODE_BASE);
            if codes.len() >= max_codes {
                break;
            }
            hc = self.step(chosen)?;
            hu = uncond.step(chosen)?;
        }
        Ok(codes)
    }

    /// Batched CFG: ONE Qwen3Lm instance carries BOTH the conditional (row 0) and unconditional
    /// (row 1) passes as a 2-row batch, so each layer's 4B weight is read ONCE per step for both
    /// - ~half the decode weight-bandwidth (the dominant cost) vs the two-instance `generate_cfg`.
    /// Prefill is per-row (forward_into_kv, identical math); only the decode loop is batched.
    /// Bit-identical to `generate_cfg` for the same seed (asserted by the cfg_batched_matches_serial
    /// A/B). `cfg_scale <= 1.0` -> cond-only `generate`.
    pub fn generate_cfg_batched(
        &mut self,
        tok: &tokenizers::Tokenizer,
        caption: &str,
        lyrics: &str,
        cot_yaml: &str,
        neg: &str,
        max_codes: usize,
        min_codes: usize,
        seed: u64,
        temperature: f32,
        top_p: f32,
        cfg_scale: f32,
    ) -> Result<Vec<u32>> {
        if cfg_scale <= 1.0 {
            return self.generate(
                tok,
                caption,
                lyrics,
                cot_yaml,
                max_codes,
                seed,
                temperature,
                top_p,
            );
        }
        let cond_prompt = build_lm_prompt_with_cot(tok, caption, lyrics, cot_yaml)?;
        let uncond_prompt = build_lm_prompt_uncond(tok, neg)?;
        let mut kv = CfgKv::new(self.layers.len());
        // Prefill each row into its own cache (single-row, bit-identical to serial prefill).
        let mut hc = Vec::new();
        for &t in &cond_prompt {
            let row = self.embed_row(t)?;
            hc = self.forward_into_kv(0, &row, &mut kv)?;
        }
        let mut hu = Vec::new();
        for &t in &uncond_prompt {
            let row = self.embed_row(t)?;
            hu = self.forward_into_kv(1, &row, &mut kv)?;
        }
        // First logits from the prefill hidden (host). Subsequent steps keep the hidden
        // ON-DEVICE (forward_b2_t) and compute logits straight from the tensor (logits_t),
        // so the per-token host work is one logits download - no hidden round-trip.
        let base = self.cand_base();
        let mut lc = self.logits_audio(&hc)?;
        let mut lu = self.logits_audio(&hu)?;
        let mut codes = Vec::new();
        let mut rng = seed.max(1);
        // CUDA-graph capture/replay of the steady-state decode step removes the per-token
        // kernel-launch flood that pegs a host core (36 layers x 2 CFG rows x ~15 ops).
        // Any capture or replay error falls back to eager for the rest of the run.
        #[cfg(feature = "cuda")]
        let mut use_graph = self.device.is_cuda();
        let _prof = std::env::var("ACE_LM_PROF").is_ok();
        let (mut _t_s, mut _t_f, mut _t_l) = (0f64, 0f64, 0f64);
        loop {
            let _ps = std::time::Instant::now();
            let chosen = sample_audio_cfg(
                &lc,
                &lu,
                base,
                self.eos,
                cfg_scale,
                temperature,
                top_p,
                self.top_k,
                &mut rng,
                codes.len() < min_codes,
            );
            if _prof {
                _t_s += _ps.elapsed().as_secs_f64();
            }
            if chosen == self.eos {
                break;
            }
            codes.push(chosen - AUDIO_CODE_BASE);
            if codes.len() >= max_codes {
                break;
            }
            let _pf = std::time::Instant::now();
            // both sequences feed the SAME chosen token; the hidden state stays on-device.
            let hidden = {
                #[cfg(feature = "cuda")]
                {
                    if use_graph {
                        match self.forward_b2_graph(chosen, &mut kv) {
                            Ok(h) => h,
                            Err(e) => {
                                eprintln!(
                                    "[ace-lm] graph decode failed ({e}); falling back to eager"
                                );
                                use_graph = false;
                                kv.graph = None;
                                if let Device::Cuda(c) = &self.device {
                                    c.context().free_capture_arena();
                                }
                                let row = self.embed_row(chosen)?;
                                let rows2 = Tensor::cat(&[&row, &row], 0)?;
                                self.forward_b2_t(&rows2, &mut kv)?
                            }
                        }
                    } else {
                        let row = self.embed_row(chosen)?;
                        let rows2 = Tensor::cat(&[&row, &row], 0)?;
                        self.forward_b2_t(&rows2, &mut kv)?
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let row = self.embed_row(chosen)?;
                    let rows2 = Tensor::cat(&[&row, &row], 0)?;
                    self.forward_b2_t(&rows2, &mut kv)?
                }
            };
            if _prof {
                _t_f += _pf.elapsed().as_secs_f64();
            }
            let _pl = std::time::Instant::now();
            let both = self.logits_audio_t(&hidden)?; // [2.(V-cand_base)] (cond row 0, uncond row 1)
            let v = both.len() / 2;
            lc = both[0..v].to_vec();
            lu = both[v..2 * v].to_vec();
            if _prof {
                _t_l += _pl.elapsed().as_secs_f64();
            }
        }
        if _prof {
            let n = codes.len().max(1) as f64;
            eprintln!("[ace-lm-prof] {} codes | sample(host)={:.1}ms ({:.2}ms/c) | forward(launch)={:.1}ms ({:.2}ms/c) | logits+download(sync)={:.1}ms ({:.2}ms/c)",
                codes.len(), _t_s*1e3, _t_s*1e3/n, _t_f*1e3, _t_f*1e3/n, _t_l*1e3, _t_l*1e3/n);
        }
        // Release the captured graph BEFORE the arena it references (the graph's nodes
        // point into the arena allocation), then free the arena.
        #[cfg(feature = "cuda")]
        {
            kv.graph = None;
            if let Device::Cuda(c) = &self.device {
                c.context().free_capture_arena();
            }
        }
        Ok(codes)
    }

    /// Continuous multi-style MORPH: ONE autoregressive code stream whose conditioning caption
    /// changes between sections. Per section we re-prefill `[chat-prompt(caption_i, lyrics_i) +
    /// the codes generated so far]` and continue - the prior codes (in context) carry the
    /// musical continuity while the new caption drifts the style, so the styles transition
    /// FLUIDLY within one stream (not a concatenation / cross-fade of independent clips). One
    /// continuous code stream -> one detok->DiT->VAE render. CFG as in `generate_cfg`.
    pub fn generate_cfg_morph(
        &mut self,
        uncond: &mut Qwen3Lm,
        tok: &tokenizers::Tokenizer,
        sections: &[MorphSection],
        neg: &str,
        seed: u64,
        temperature: f32,
        top_p: f32,
        cfg_scale: f32,
    ) -> Result<Vec<u32>> {
        let use_cfg = cfg_scale > 1.0;
        let mut all: Vec<u32> = Vec::new();
        let mut prior: Vec<u32> = Vec::new(); // generated audio-code TOKENS (AUDIO_CODE_BASE + idx)
        let mut rng = seed.max(1);
        for sec in sections {
            // re-prefill: new caption's prompt + the codes already generated (the continuity).
            let mut cond_p = build_lm_prompt_with_cot(tok, sec.caption, sec.lyrics, sec.cot)?;
            cond_p.extend_from_slice(&prior);
            self.reset();
            let mut hc = self.prefill(&cond_p)?;
            let mut hu = Vec::new();
            if use_cfg {
                let mut unc_p = build_lm_prompt_uncond(tok, neg)?;
                unc_p.extend_from_slice(&prior);
                uncond.reset();
                hu = uncond.prefill(&unc_p)?;
            }
            let mut made = 0usize;
            loop {
                let chosen = if use_cfg {
                    let (lc, lu) = (self.logits(&hc)?, uncond.logits(&hu)?);
                    sample_audio_cfg(
                        &lc,
                        &lu,
                        0,
                        self.eos,
                        cfg_scale,
                        temperature,
                        top_p,
                        self.top_k,
                        &mut rng,
                        false,
                    )
                } else {
                    sample_audio(
                        &self.logits(&hc)?,
                        0,
                        self.eos,
                        temperature,
                        top_p,
                        self.top_k,
                        &mut rng,
                    )
                };
                if chosen == self.eos {
                    break;
                }
                all.push(chosen - AUDIO_CODE_BASE);
                prior.push(chosen);
                made += 1;
                if made >= sec.n_codes {
                    break;
                }
                hc = self.step(chosen)?;
                if use_cfg {
                    hu = uncond.step(chosen)?;
                }
            }
        }
        Ok(all)
    }
}

/// Build the ACE-Step LM tokenizer: reuse `build_tokenizer_from_gguf` for the
/// vocab/merges, but override the pre-tokenizer with Qwen's (the GGUF default
/// applied by that helper is the GPT-2 ByteLevel regex, which splits e.g. ":\n\n"
/// into ":"+"\n\n"; Qwen keeps punctuation+newlines together via
/// `[^\s\p{L}\p{N}]+[\r\n]*`). Sequence[Split(qwen_regex, Isolated), ByteLevel(no-regex)].
pub fn acestep_tokenizer(gguf_path: &str) -> Result<tokenizers::Tokenizer> {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    use tokenizers::SplitDelimiterBehavior;
    let f = std::fs::File::open(gguf_path)
        .map_err(|e| crate::tensor::Error(format!("open {gguf_path}: {e}")))?;
    let content = crate::tensor::quantized::gguf_file::read_mapped_file(&f)
        .map_err(|e| crate::tensor::Error(format!("gguf: {e}")))?;
    let mut tok = crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&content)
        .map_err(|e| crate::tensor::Error(format!("tokenizer: {e}")))?;
    let qwen = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split = Split::new(
        SplitPattern::Regex(qwen.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| crate::tensor::Error(format!("split: {e}")))?;
    let bl = ByteLevel::new(false, false, false);
    let seq = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(bl),
    ]);
    tok.with_pre_tokenizer(Some(seq));
    Ok(tok)
}

pub const AUDIO_CODE_BASE: u32 = 151669;
pub const AUDIO_CODE_COUNT: u32 = 65535;
const IM_START: u32 = 151644;
const IM_END: u32 = 151645;
const THINK: u32 = 151667;
const THINK_END: u32 = 151668;
const LM_INSTRUCTION: &str = "Generate audio semantic tokens based on the given conditions:";

/// Qwen3 chat-template prompt with injected CoT (oracle build_lm_prompt_with_cot):
/// text segments via the HF tokenizer (no special tokens), special IDs inserted.
fn build_lm_prompt_with_cot(
    tok: &tokenizers::Tokenizer,
    caption: &str,
    lyrics: &str,
    cot_yaml: &str,
) -> Result<Vec<u32>> {
    let enc = |s: &str| -> Result<Vec<u32>> {
        Ok(tok
            .encode(s, false)
            .map_err(|e| crate::tensor::Error(format!("bpe: {e}")))?
            .get_ids()
            .to_vec())
    };
    let mut ids = vec![IM_START];
    ids.extend(enc(&format!(
        "system\n# Instruction\n{LM_INSTRUCTION}\n\n"
    ))?);
    ids.push(IM_END);
    ids.extend(enc("\n")?);
    ids.push(IM_START);
    ids.extend(enc(&format!(
        "user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    ))?);
    ids.push(IM_END);
    ids.extend(enc("\n")?);
    ids.push(IM_START);
    ids.extend(enc("assistant\n")?);
    ids.push(THINK);
    ids.extend(enc(&format!("\n{cot_yaml}"))?);
    ids.push(THINK_END);
    ids.extend(enc("\n\n")?);
    Ok(ids)
}

/// Unconditional prompt for CFG (oracle build_lm_prompt_uncond_with_cot): bare user
/// (optional negative prompt) + empty CoT - the training CFG-dropout distribution.
fn build_lm_prompt_uncond(tok: &tokenizers::Tokenizer, neg: &str) -> Result<Vec<u32>> {
    let enc = |s: &str| -> Result<Vec<u32>> {
        Ok(tok
            .encode(s, false)
            .map_err(|e| crate::tensor::Error(format!("bpe: {e}")))?
            .get_ids()
            .to_vec())
    };
    let mut ids = vec![IM_START];
    ids.extend(enc(&format!(
        "system\n# Instruction\n{LM_INSTRUCTION}\n\n"
    ))?);
    ids.push(IM_END);
    ids.extend(enc("\n")?);
    ids.push(IM_START);
    ids.extend(enc(&format!("user\n{neg}"))?);
    ids.push(IM_END);
    ids.extend(enc("\n")?);
    ids.push(IM_START);
    ids.extend(enc("assistant\n")?);
    ids.push(THINK);
    ids.extend(enc("\n\n")?);
    ids.push(THINK_END);
    ids.extend(enc("\n\n")?);
    Ok(ids)
}

/// Metadata CoT block (oracle build_cot_yaml: sorted keys, caption 80-col wrapped).
pub fn build_cot_yaml(
    bpm: i32,
    caption: &str,
    duration: i32,
    keyscale: &str,
    language: &str,
    timesignature: &str,
) -> String {
    let wrap = |key: &str, val: &str| -> String {
        let mut result = format!("{key}:");
        let mut col = key.len() + 1;
        for word in val.split(' ') {
            if col > 80 {
                result.push_str("\n  ");
                col = 2;
            } else {
                result.push(' ');
                col += 1;
            }
            result.push_str(word);
            col += word.len();
        }
        result.push('\n');
        result
    };
    let mut y = String::new();
    if bpm > 0 {
        y += &format!("bpm: {bpm}\n");
    }
    if !caption.is_empty() {
        y += &wrap("caption", caption);
    }
    if duration > 0 {
        y += &format!("duration: {duration}\n");
    }
    if !keyscale.is_empty() {
        y += &format!("keyscale: {keyscale}\n");
    }
    if !language.is_empty() {
        y += &format!("language: {language}\n");
    }
    if !timesignature.is_empty() {
        y += &format!("timesignature: {timesignature}\n");
    }
    y
}

/// Temperature + top-p sample restricted to the audio-code range + EOS. Returns the
/// chosen token id (EOS or AUDIO_CODE_BASE+idx). `rng` is a simple LCG state.
fn sample_audio(
    logits: &[f32],
    base: usize,
    eos: u32,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rng: &mut u64,
) -> u32 {
    // candidate ids: EOS + the audio-code block. `base` = the token id of `logits[0]`
    // (0 for full-vocab logits, `cand_base` for the candidate-only lm_head slice).
    let at = |id: u32| logits[id as usize - base];
    let mut cand: Vec<(u32, f32)> = Vec::with_capacity(AUDIO_CODE_COUNT as usize + 1);
    cand.push((eos, at(eos)));
    for c in 0..AUDIO_CODE_COUNT {
        let id = AUDIO_CODE_BASE + c;
        cand.push((id, at(id)));
    }
    // temperature + softmax
    let t = temperature.max(1e-4);
    let mx = cand
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    for c in cand.iter_mut() {
        c.1 = ((c.1 - mx) / t).exp();
    }
    let sum: f32 = cand.iter().map(|&(_, p)| p).sum();
    for c in cand.iter_mut() {
        c.1 /= sum;
    }
    // top-k then top-p nucleus
    // Partial top-prefix selection instead of a full O(v.log v) sort of all 65k candidates
    // per token (that sort was ~half the host-sampling cost). top_k/top_p only need the
    // highest-probability prefix: grow a top-N window (via O(v) select_nth) until it covers
    // top_p (or reaches top_k), then sort ONLY that window. Bit-identical result to
    // sort-all -> top_k -> top_p, since top_p reads a descending prefix and the window is it.
    let cmp =
        |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
    let cap = if top_k > 0 {
        top_k.min(cand.len())
    } else {
        cand.len()
    };
    let mut n = cap.min(256).max(1);
    loop {
        if n >= cand.len() {
            cand.sort_by(cmp);
            break;
        }
        cand.select_nth_unstable_by(n - 1, cmp);
        let covered = cand[..n].iter().map(|&(_, p)| p).sum::<f32>() >= top_p;
        if covered || n >= cap {
            cand.truncate(n);
            cand.sort_by(cmp);
            break;
        }
        n = (n * 4).min(cap);
    }
    let mut cum = 0.0f32;
    let mut cut = cand.len();
    for (i, &(_, p)) in cand.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cut = i + 1;
            break;
        }
    }
    cand.truncate(cut);
    let renorm: f32 = cand.iter().map(|&(_, p)| p).sum();
    // LCG draw
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let r = ((*rng >> 33) as f32 / (1u64 << 31) as f32) * renorm;
    let mut acc = 0.0f32;
    for &(id, p) in &cand {
        acc += p;
        if acc >= r {
            return id;
        }
    }
    cand.last().map(|&(id, _)| id).unwrap_or(eos)
}

/// CFG sample: combine cond/uncond logits as `lu + scale.(lc - lu)` over the
/// EOS+audio candidate set, then temperature/top-p/LCG sample (mirrors sample_audio).
fn sample_audio_cfg(
    lc: &[f32],
    lu: &[f32],
    base: usize,
    eos: u32,
    cfg_scale: f32,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rng: &mut u64,
    ban_eos: bool,
) -> u32 {
    // `base` = token id of lc[0]/lu[0] (0 for full vocab, `cand_base` for the slice).
    let cfg = |id: u32| {
        let i = id as usize - base;
        lu[i] + cfg_scale * (lc[i] - lu[i])
    };
    let mut cand: Vec<(u32, f32)> = Vec::with_capacity(AUDIO_CODE_COUNT as usize + 1);
    // `ban_eos` enforces a minimum-length floor: EOS gets a -∞ logit so the LM keeps
    // emitting audio codes until the caller's `min_codes` is reached (forces the full
    // requested duration when the LM would otherwise end the song early).
    cand.push((eos, if ban_eos { f32::NEG_INFINITY } else { cfg(eos) }));
    for c in 0..AUDIO_CODE_COUNT {
        let id = AUDIO_CODE_BASE + c;
        cand.push((id, cfg(id)));
    }
    let t = temperature.max(1e-4);
    let mx = cand
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    for c in cand.iter_mut() {
        c.1 = ((c.1 - mx) / t).exp();
    }
    let sum: f32 = cand.iter().map(|&(_, p)| p).sum();
    for c in cand.iter_mut() {
        c.1 /= sum;
    }
    // Partial top-prefix selection instead of a full O(v.log v) sort of all 65k candidates
    // per token (that sort was ~half the host-sampling cost). top_k/top_p only need the
    // highest-probability prefix: grow a top-N window (via O(v) select_nth) until it covers
    // top_p (or reaches top_k), then sort ONLY that window. Bit-identical result to
    // sort-all -> top_k -> top_p, since top_p reads a descending prefix and the window is it.
    let cmp =
        |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
    let cap = if top_k > 0 {
        top_k.min(cand.len())
    } else {
        cand.len()
    };
    let mut n = cap.min(256).max(1);
    loop {
        if n >= cand.len() {
            cand.sort_by(cmp);
            break;
        }
        cand.select_nth_unstable_by(n - 1, cmp);
        let covered = cand[..n].iter().map(|&(_, p)| p).sum::<f32>() >= top_p;
        if covered || n >= cap {
            cand.truncate(n);
            cand.sort_by(cmp);
            break;
        }
        n = (n * 4).min(cap);
    }
    let mut cum = 0.0f32;
    let mut cut = cand.len();
    for (i, &(_, p)) in cand.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cut = i + 1;
            break;
        }
    }
    cand.truncate(cut);
    let renorm: f32 = cand.iter().map(|&(_, p)| p).sum();
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let r = ((*rng >> 33) as f32 / (1u64 << 31) as f32) * renorm;
    let mut acc = 0.0f32;
    for &(id, p) in &cand {
        acc += p;
        if acc >= r {
            return id;
        }
    }
    cand.last().map(|&(id, _)| id).unwrap_or(eos)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod demand_tests {
    /// A card that holds the weights alone does not hold the model: the KV cache and the
    /// step reserve come with them, so the demand is always more than the file.
    #[test]
    fn the_placement_demand_exceeds_the_weights() {
        let dir = std::env::temp_dir().join(format!("ace-lm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lm.gguf");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();
        let demand = super::placement_demand(path.to_str().unwrap());
        assert!(demand > 4096 + super::kv_bytes_per_layer());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
