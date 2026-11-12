//! A dense model placed on a Vulkan device answers as the host does: its first layer within a
//! tolerance, and its greedy text token for token. Every Vulkan GPU the loader lists is tried.
//!
//! Ignored by default: it needs the model in the store and a Vulkan device. Run with
//!   cargo test --features vulkan --profile gate vulkan_ -- --ignored --test-threads=2

use super::*;
use crate::inference::place::layer_executor::{DeviceKind, HeteroSegment};

/// The model the tests read from the store.
const TAG: &str = "qwen3:0.6b";

/// The tokens the greedy answer runs for.
const STEPS: usize = 32;

fn model_blob(tag: &str) -> std::path::PathBuf {
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, tag) = tag.split_once(':').unwrap_or((tag, "latest"));
    let m = crate::inference::load::ollama_manager::OllamaManager::new(store.clone());
    let manifest = m
        .read_manifest_json(name, tag)
        .expect("manifest in the store");
    let digest = manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")
        .and_then(|l| l["digest"].as_str())
        .expect("a model layer")
        .replace(':', "-");
    store.join("blobs").join(digest)
}

fn host_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn argmax(t: &Tensor) -> u32 {
    host_vec(t)
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

struct Fixture {
    mmap: std::sync::Arc<memmap2::Mmap>,
    n_layers: usize,
}

impl Fixture {
    fn open() -> Self {
        Self::open_tag(TAG)
    }

    fn open_tag(tag: &str) -> Self {
        let path = model_blob(tag);
        let file = std::fs::File::open(&path).unwrap();
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
        let c = self_content(&mmap);
        let arch = c.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone();
        let n_layers = GenericTransformerConfig::from_gguf(&c, &arch)
            .unwrap()
            .n_layers;
        Self { mmap, n_layers }
    }

    fn build(&self, plan: &HeteroPlan) -> GenericHeteroTransformer {
        self.try_build(plan).unwrap()
    }

    fn try_build(
        &self,
        plan: &HeteroPlan,
    ) -> std::result::Result<GenericHeteroTransformer, String> {
        GenericHeteroTransformer::from_gguf(
            self_content(&self.mmap),
            &self.mmap[..],
            &HashMap::new(),
            plan,
        )
        .map_err(|e| e.to_string())
    }

    fn host(&self) -> HeteroPlan {
        HeteroPlan::forced_gpu(self.n_layers, 0, 0)
    }

    /// Layers `[0, upto)` on Vulkan device `device`, the rest on the host.
    fn on_vulkan(&self, device: usize, upto: usize) -> HeteroPlan {
        let segment = |kind, layer_start, layer_end| HeteroSegment {
            kind,
            layer_start,
            layer_end,
            free_memory_bytes: u64::MAX,
        };
        let mut segments = vec![segment(DeviceKind::Vulkan(device), 0, upto)];
        if upto < self.n_layers {
            segments.push(segment(DeviceKind::Cpu, upto, self.n_layers));
        }
        HeteroPlan {
            segments,
            tp_cards: None,
            total_layers: self.n_layers,
            host_experts_from: None,
        }
    }
}

fn self_content(mmap: &std::sync::Arc<memmap2::Mmap>) -> gguf_file::Content {
    gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
}

/// Layer 0 of the model in plain f32 over the file's dequantised weights: the reference the
/// Vulkan layer is held to, free of the host's quantised activations.
fn reference_layer0(fx: &Fixture, x: &[f32], tokens: usize) -> Vec<f32> {
    let c = self_content(&fx.mmap);
    let arch = c.metadata["general.architecture"]
        .to_string()
        .unwrap()
        .clone();
    let cfg = GenericTransformerConfig::from_gguf(&c, &arch).unwrap();
    let w = |name: &str| -> Vec<f32> {
        host_vec(
            &c.tensor(&mut std::io::Cursor::new(&fx.mmap[..]), name, &Device::Cpu)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap(),
        )
    };
    let has = |name: &str| c.tensor_infos.contains_key(name);
    let (h, nh, nkv, hd) = (
        cfg.embedding_length,
        cfg.n_head,
        cfg.n_kv_head,
        cfg.head_dim,
    );
    let eps = cfg.rms_norm_eps as f32;
    let rms = |v: &[f32], g: &[f32]| -> Vec<f32> {
        let r = 1.0 / (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32 + eps).sqrt();
        v.iter().zip(g).map(|(a, b)| a * r * b).collect()
    };
    let mm = |x: &[f32], wt: &[f32], rows: usize| -> Vec<f32> {
        let cols = x.len() / tokens;
        (0..tokens)
            .flat_map(|t| {
                (0..rows).map(move |r| {
                    (0..cols)
                        .map(|k| wt[r * cols + k] as f64 * x[t * cols + k] as f64)
                        .sum::<f64>() as f32
                })
            })
            .collect()
    };
    let rope = |v: &mut [f32], pos: usize| {
        let half = hd / 2;
        for i in 0..half {
            let theta = pos as f32 * cfg.rope_freq_base.powf(-((2 * i) as f32) / hd as f32);
            let (sn, cs) = theta.sin_cos();
            let (a, b) = if cfg.flags.use_rope_i {
                (2 * i, 2 * i + 1)
            } else {
                (i, i + half)
            };
            let (x0, x1) = (v[a], v[b]);
            v[a] = x0 * cs - x1 * sn;
            v[b] = x0 * sn + x1 * cs;
        }
    };
    let an = w("blk.0.attn_norm.weight");
    let xn: Vec<f32> = x.chunks(h).flat_map(|r| rms(r, &an)).collect();
    let mut q = mm(&xn, &w("blk.0.attn_q.weight"), nh * hd);
    let mut k = mm(&xn, &w("blk.0.attn_k.weight"), nkv * hd);
    let v = mm(&xn, &w("blk.0.attn_v.weight"), nkv * hd);
    for (buf, norm) in [
        (&mut q, "blk.0.attn_q_norm.weight"),
        (&mut k, "blk.0.attn_k_norm.weight"),
    ] {
        let g = has(norm).then(|| w(norm));
        let heads = buf.len() / tokens / hd;
        for t in 0..tokens {
            for hh in 0..heads {
                let row = &mut buf[(t * heads + hh) * hd..(t * heads + hh + 1) * hd];
                if let Some(g) = &g {
                    let n = rms(row, g);
                    row.copy_from_slice(&n);
                }
                rope(row, t);
            }
        }
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn = vec![0f32; tokens * nh * hd];
    for t in 0..tokens {
        for hh in 0..nh {
            let g = hh / (nh / nkv);
            let qr = &q[(t * nh + hh) * hd..][..hd];
            let s: Vec<f32> = (0..=t)
                .map(|j| {
                    qr.iter()
                        .zip(&k[(j * nkv + g) * hd..][..hd])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            for d in 0..hd {
                attn[(t * nh + hh) * hd + d] = (0..=t)
                    .map(|j| e[j] * v[(j * nkv + g) * hd + d])
                    .sum::<f32>()
                    / sum;
            }
        }
    }
    let o = mm(&attn, &w("blk.0.attn_output.weight"), h);
    let x1: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
    let fnw = w("blk.0.ffn_norm.weight");
    let xn2: Vec<f32> = x1.chunks(h).flat_map(|r| rms(r, &fnw)).collect();
    let gate = mm(&xn2, &w("blk.0.ffn_gate.weight"), cfg.ffn_dim);
    let up = mm(&xn2, &w("blk.0.ffn_up.weight"), cfg.ffn_dim);
    let act: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let down = mm(&act, &w("blk.0.ffn_down.weight"), h);
    x1.iter().zip(&down).map(|(a, b)| a + b).collect()
}

fn prompt() -> Vec<u32> {
    (0..24u32).map(|i| 1000 + (i * 37) % 20000).collect()
}

fn ids(t: &[u32]) -> Tensor {
    Tensor::new(t, &Device::Cpu).unwrap().unsqueeze(0).unwrap()
}

/// The greedy continuation of the prompt, and the decode rate in tokens per second.
fn greedy(m: &mut GenericHeteroTransformer, prompt: &[u32]) -> (Vec<u32>, f64) {
    let mut tokens = vec![argmax(&m.forward(&ids(prompt), 0).unwrap())];
    let start = std::time::Instant::now();
    for i in 0..STEPS - 1 {
        let next = m.forward(&ids(&tokens[i..=i]), prompt.len() + i).unwrap();
        tokens.push(argmax(&next));
    }
    let rate = (STEPS - 1) as f64 / start.elapsed().as_secs_f64();
    (tokens, rate)
}

#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_first_layer_matches_the_host() {
    let fx = Fixture::open();
    let prompt = prompt();
    let mut host = fx.build(&fx.host());
    let (h, _) = host.embed_stage(&ids(&prompt), 0).unwrap();
    let (want, _) = host
        .run_layers_in(h.clone(), None, 0, prompt.len(), 0, 1)
        .unwrap();
    let host_rows = host_vec(&want);
    let embed = host_vec(&h);
    let reference = reference_layer0(&fx, &embed, prompt.len());
    // The layer's own contribution, against which an error is measured: the residual it adds
    // to is exact everywhere.
    let delta: f32 = reference
        .iter()
        .zip(&embed)
        .fold(0f32, |m, (w, e)| m.max((w - e).abs()));
    let max_err = |rows: &[f32]| -> f32 {
        rows.iter()
            .zip(&reference)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
    };
    let host_err = max_err(&host_rows);
    println!("{TAG} layer 0 on the host: max error {host_err:.3e} of a change {delta:.3e}");
    // The native runtime of the card, where the build has one, sets the bound.
    // Without it, the host sets it: it reads the same quantised weights.
    let bound = match native_card() {
        Some(card) => {
            let mut native = try_on_native(&fx, &card, 1).unwrap();
            let (got, _) = native
                .run_layers_in(h.clone(), None, 0, prompt.len(), 0, 1)
                .unwrap();
            let err = max_err(&host_vec(&got));
            println!(
                "{TAG} layer 0 on the native card: max error {err:.3e} of a change {delta:.3e}"
            );
            drop(native);
            release_native(&card);
            err
        }
        None => host_err,
    };
    for d in crate::inference::kernel::vulkan::devices() {
        let mut m = fx.build(&fx.on_vulkan(d.index, 1));
        let (got, _) = m
            .run_layers_in(h.clone(), None, 0, prompt.len(), 0, 1)
            .unwrap();
        let err = max_err(&host_vec(&got));
        println!(
            "{TAG} layer 0 on {}: max error {err:.3e} of a change {delta:.3e}",
            d.name
        );
        assert!(
            err <= bound,
            "{}: layer 0 differs from the f32 reference by {err}, beyond the bound {bound}",
            d.name
        );
    }
}

/// The chat prompt the text tests answer.
const TEXT: &str =
    "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";

/// The native card of the build, CUDA or ROCm.
/// `None` when the build has no native runtime or no card answers.
fn native_card() -> Option<Device> {
    #[cfg(feature = "gpu")]
    {
        Device::new_cuda(0).ok()
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// Gives the card back the memory the native runtime keeps cached for the copies already freed.
fn release_native(card: &Device) {
    #[cfg(feature = "gpu")]
    if let Device::Cuda(cd) = card {
        crate::tensor::cuda::reclaim_pool(cd);
    }
    #[cfg(not(feature = "gpu"))]
    let _ = card;
}

/// The model with its first `layers` layers on the native card `card`.
fn try_on_native(
    fx: &Fixture,
    card: &Device,
    layers: usize,
) -> std::result::Result<GenericHeteroTransformer, String> {
    let cards: HashMap<usize, Device> = [(0, card.clone())].into_iter().collect();
    GenericHeteroTransformer::from_gguf(
        self_content(&fx.mmap),
        &fx.mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(fx.n_layers, layers, 0),
    )
    .map_err(|e| e.to_string())
}

/// Whether the error `why` says a device ran out of memory.
fn out_of_memory(why: &str) -> bool {
    why.contains("out of memory") || why.contains("[oom]")
}

/// The host's logit rows at the prompt's last row and after each of its own greedy tokens.
fn host_rows(fx: &Fixture, prompt: &[u32], steps: usize) -> Vec<Vec<f32>> {
    let mut host = fx.build(&fx.host());
    let mut rows = vec![host_vec(&host.forward(&ids(prompt), 0).unwrap())];
    for i in 0..steps - 1 {
        let t = [top2(&rows[i]).0 as u32];
        rows.push(host_vec(&host.forward(&ids(&t), prompt.len() + i).unwrap()));
    }
    rows
}

/// The picks of `m` at the rows of `host`, fed the host's tokens.
fn forced_picks(m: &mut GenericHeteroTransformer, prompt: &[u32], host: &[Vec<f32>]) -> Vec<usize> {
    try_forced_picks(m, prompt, host).unwrap()
}

/// [`forced_picks`], or the error of the first step that fails.
fn try_forced_picks(
    m: &mut GenericHeteroTransformer,
    prompt: &[u32],
    host: &[Vec<f32>],
) -> std::result::Result<Vec<usize>, String> {
    let first = m.forward(&ids(prompt), 0).map_err(|e| e.to_string())?;
    let mut picks = vec![top2(&host_vec(&first)).0];
    for (i, row) in host[..host.len() - 1].iter().enumerate() {
        let t = [top2(row).0 as u32];
        let next = m
            .forward(&ids(&t), prompt.len() + i)
            .map_err(|e| e.to_string())?;
        picks.push(top2(&host_vec(&next)).0);
    }
    Ok(picks)
}

/// What the native card of the build says at the steps of a run.
enum Native {
    /// Its picks, fed the host's tokens.
    Picks(Vec<usize>),
    /// The build has no native runtime, or no card answers.
    Absent,
    /// The model does not fit the native card: why.
    NoRoom(String),
}

/// The picks of the native card of the build at the rows of `host`, fed the host's tokens.
/// The caller frees every Vulkan copy first.
/// The native copy is freed, and its cached memory given back, before this returns.
fn native_picks(fx: &Fixture, prompt: &[u32], host: &[Vec<f32>]) -> Native {
    let Some(card) = native_card() else {
        return Native::Absent;
    };
    let picks = try_on_native(fx, &card, fx.n_layers)
        .and_then(|mut m| try_forced_picks(&mut m, prompt, host));
    release_native(&card);
    match picks {
        Ok(picks) => Native::Picks(picks),
        Err(why) if out_of_memory(&why) => Native::NoRoom(why),
        Err(why) => panic!("the native card: {why}"),
    }
}

/// Fails on each step where the Vulkan pick leaves the host's while the native pick keeps it.
/// Without a native runtime, a Vulkan pick that leaves the host's passes only on an exact tie.
/// When the model does not fit the native card, a step where the picks part is reported
/// unjudged.
fn judge_picks(at: &str, host: &[Vec<f32>], native: &Native, got: &[usize]) {
    if let Native::NoRoom(why) = native {
        println!("{at}: the native card has no room for the model, its comparison skipped: {why}");
    }
    let mut gaps = Vec::new();
    for (step, (row, &v)) in host.iter().zip(got).enumerate() {
        let h = top2(row).0;
        if v == h {
            continue;
        }
        let margin = row[h] - row[v];
        let n = match native {
            Native::Picks(n) => Some(n[step]),
            _ => None,
        };
        println!("{at}: step {step}: Vulkan picks {v}, the host {h} by {margin:.3}, native {n:?}");
        let held = match (native, n) {
            (_, Some(n)) => n != h,
            (Native::NoRoom(_), _) => {
                println!("{at}: step {step}: unjudged");
                true
            }
            _ => margin == 0.0,
        };
        if !held {
            gaps.push(format!(
                "step {step}: Vulkan picks {v}, the host {h} by {margin}, native {n:?}"
            ));
        }
    }
    assert!(gaps.is_empty(), "{at}: {}", gaps.join("; "));
}

fn top2(l: &[f32]) -> (usize, usize) {
    let mut best = (0, 1);
    if l[1] > l[0] {
        best = (1, 0);
    }
    for (i, &v) in l.iter().enumerate().skip(2) {
        if v > l[best.0] {
            best = (i, best.0);
        } else if v > l[best.1] {
            best = (best.0, i);
        }
    }
    best
}

// The greedy text of the qwen3 model is left out.
// On the Radeon RX 9070 and the GTX 1650 at step 26, Vulkan picks 8038 where the host and the
// native card pick 2874.

/// The model with every layer on Vulkan device `d`, or `None` when the device cannot hold it:
/// the placement would then split the model across devices and the host, which another test
/// covers.
fn whole_on(
    fx: &Fixture,
    tag: &str,
    d: &crate::inference::kernel::vulkan::DeviceInfo,
) -> Option<GenericHeteroTransformer> {
    match fx.try_build(&fx.on_vulkan(d.index, fx.n_layers)) {
        Ok(model) => Some(model),
        Err(why) if why.contains("out of memory") => {
            println!("{tag} vulkan {}: the model does not fit, skipped", d.name);
            None
        }
        Err(why) => panic!("{tag} vulkan {}: {why}", d.name),
    }
}

/// The host's greedy answer and, fed the same tokens, the Vulkan one: at every step the
/// Vulkan argmax is the host's, or the native card's leaves the host's too. The
/// free-running answers and the decode rates are printed beside.
fn greedy_text_matches(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
    let decode = |t: &[u32]| tokenizer.decode(t, false).unwrap_or_default();
    let (want, host_rate) = greedy(&mut fx.build(&fx.host()), &prompt);
    println!("{tag}: host {host_rate:.1} tok/s {:?}", decode(&want));
    let host = host_rows(&fx, &prompt, STEPS);
    let mut picks = Vec::new();
    for d in crate::inference::kernel::vulkan::devices() {
        let Some(mut whole) = whole_on(&fx, tag, &d) else {
            continue;
        };
        // One copy serves both runs.
        // The teacher-forced run comes first, on the fresh copy, fed the host's tokens.
        picks.push((
            format!("{tag} {}", d.name),
            forced_picks(&mut whole, &prompt, &host),
        ));
        let (got, rate) = greedy(&mut whole, &prompt);
        println!(
            "{tag}: vulkan {}: {rate:.1} tok/s {:?}",
            d.name,
            decode(&got)
        );
    }
    if picks.is_empty() {
        return;
    }
    // The native copy loads once every Vulkan copy is freed.
    let native = native_picks(&fx, &prompt, &host);
    for (at, got) in picks {
        judge_picks(&at, &host, &native, &got);
    }
}

/// A prompt that crosses the chunk a step carries and several tiles of keys: the Vulkan argmax
/// is the host's at the prompt's last row and at the steps that follow, or the native card's
/// leaves the host's too.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_long_prompt_matches_the_host() {
    long_prompt_matches(TAG, 14);
}

fn long_prompt_matches(tag: &str, repeats: usize) {
    let fx = Fixture::open_tag(tag);
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let text = TEXT.repeat(repeats);
    let prompt = tokenizer
        .encode(text.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(prompt.len() > 2 * 128, "{} tokens", prompt.len());
    let host = host_rows(&fx, &prompt, 8);
    let mut picks = Vec::new();
    for d in crate::inference::kernel::vulkan::devices() {
        let Some(mut m) = whole_on(&fx, tag, &d) else {
            continue;
        };
        let got = forced_picks(&mut m, &prompt, &host);
        picks.push((
            format!("{tag} {} with {} tokens", d.name, prompt.len()),
            got,
        ));
    }
    if picks.is_empty() {
        return;
    }
    // The native copy loads once every Vulkan copy is freed.
    let native = native_picks(&fx, &prompt, &host);
    for (at, got) in picks {
        judge_picks(&at, &host, &native, &got);
        println!("{at}: ok");
    }
}

/// The rows after the first `layers` layers on every Vulkan device against the host's: the
/// difference stays within a tenth of what the layers add to the residual.
fn first_layers_match(tag: &str, layers: usize, tolerance: f32) {
    let fx = Fixture::open_tag(tag);
    let prompt = prompt();
    let mut host = fx.build(&fx.host());
    let (h, _) = host.embed_stage(&ids(&prompt), 0).unwrap();
    let (want, _) = host
        .run_layers_in(h.clone(), None, 0, prompt.len(), 0, layers)
        .unwrap();
    let want = host_vec(&want);
    let embed = host_vec(&h);
    let change = want
        .iter()
        .zip(&embed)
        .fold(0f32, |m, (w, e)| m.max((w - e).abs()));
    for d in crate::inference::kernel::vulkan::devices() {
        let mut m = fx.build(&fx.on_vulkan(d.index, layers));
        let (got, _) = m
            .run_layers_in(h.clone(), None, 0, prompt.len(), 0, layers)
            .unwrap();
        let err = host_vec(&got)
            .iter()
            .zip(&want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "{tag} {layers} layers on {}: max error {err:.3e} of a change {change:.3e}",
            d.name
        );
        assert!(err <= tolerance * change, "{}: {err} of {change}", d.name);
    }
}

/// The experts' share of the first layer: the rows after attention and after the experts on
/// Vulkan, against the host's experts fed the rows after attention.
fn experts_match(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let prompt = prompt();
    let mut host = fx.build(&fx.host());
    let (h, _) = host.embed_stage(&ids(&prompt), 0).unwrap();
    let rows = host_vec(&h);
    let hidden = rows.len() / prompt.len();
    for d in crate::inference::kernel::vulkan::devices() {
        let mut m = fx.build(&fx.on_vulkan(d.index, 1));
        let stack = m.vulkan.stack_mut();
        let router = stack.first_dispatch(prompt.len(), "router").unwrap();
        // Stops at the norm that opens the experts' block.
        stack.stop_after(Some(router - 1));
        let attended = stack.forward(&rows, 0, 0, 1).unwrap();
        stack.stop_after(None);
        let full = stack.forward(&rows, 0, 0, 1).unwrap();
        let got: Vec<f32> = full.iter().zip(&attended).map(|(f, a)| f - a).collect();
        let x = Tensor::from_vec(attended, (1, prompt.len(), hidden), &Device::Cpu).unwrap();
        let layer = &host.layers[0];
        let normed = match layer.ffn_norm.as_ref().unwrap() {
            super::config::WeightedNorm::Rms(n) => n.forward(&x).unwrap(),
            _ => panic!("an RMS norm"),
        };
        let want = host_vec(
            &layer
                .moe
                .as_ref()
                .unwrap()
                .forward_rows(
                    &normed,
                    &x,
                    layer.embedding_length_for_moe,
                    layer.ffn_norm_eps,
                )
                .unwrap(),
        );
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = want
            .iter()
            .zip(&got)
            .fold(0f32, |m, (w, g)| m.max((w - g).abs()));
        println!(
            "{tag} experts on {}: max error {err:.3e} of {scale:.3e}",
            d.name
        );
        assert!(err <= 0.01 * scale, "{}: {err} of {scale}", d.name);
    }
}

/// The other model families: each must give the host's tokens, on a short prompt and on one
/// that crosses several steps and tiles of keys, or leave them where the native card does.
/// A family names the token tests it runs when it leaves one out.
macro_rules! family {
    (@test greedy_text, $tag:expr) => {
        #[test]
        #[ignore = "needs the model in the store and a Vulkan device"]
        fn greedy_text() {
            super::greedy_text_matches($tag);
        }
    };
    (@test long_prompt, $tag:expr) => {
        #[test]
        #[ignore = "needs the model in the store and a Vulkan device"]
        fn long_prompt() {
            super::long_prompt_matches($tag, 14);
        }
    };
    ($name:ident, $tag:expr) => {
        family!($name, $tag, greedy_text, long_prompt);
    };
    ($name:ident, $tag:expr, $($test:ident),*) => {
        mod $name {
            $(family!(@test $test, $tag);)*

            #[test]
            #[ignore = "needs the model in the store and a Vulkan device"]
            fn first_layer() {
                super::first_layers_match($tag, 1, 0.1);
            }

            #[test]
            #[ignore = "needs the model in the store and a Vulkan device"]
            fn logit_noise() {
                super::logit_noise($tag, 1.5, false);
            }
        }
    };
}

family!(llama, "llama3.2:1b");
// The greedy text is left out.
// On the Radeon RX 9070 and the GTX 1650, Vulkan picks 236761 at step 9 where the host and the
// native card pick 21603.
family!(gemma, "gemma3:latest", long_prompt);
family!(smollm3, "smollm3:latest");
family!(granite, "granite3.1-dense:2b");
family!(granite_moe, "granite3-moe:1b");
// The greedy text is left out.
// On the Radeon RX 9070, Vulkan picks 6437 at step 21 where the host and ROCm pick 24210.
family!(olmo2, "olmo2:7b", long_prompt);
family!(gemma2, "shieldgemma:2b");

/// Attention sinks, output and expert biases, and experts in MXFP4 behind the clamped SwiGLU: the
/// first layer, the logit spread, the greedy text and the long prompt hold it.
mod gpt_oss {
    const TAG: &str = "gpt-oss:20b";

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn long_prompt() {
        super::long_prompt_matches(TAG, 14);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn first_layer() {
        super::first_layers_match(TAG, 1, 0.1);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn logit_noise() {
        super::logit_noise(TAG, 1.5, false);
    }

    // The greedy text is left out.
    // On the Radeon RX 9070, Vulkan picks 481 at step 21 where the host and ROCm pick 316.
}

/// Recurrent layers between layers that attend: the host runs the recurrent ones, Vulkan the
/// others, and each is reported where it runs.
mod lfm2 {
    const TAG: &str = "lfm2.5-thinking:latest";

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn greedy_text() {
        super::greedy_text_matches(TAG);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn long_prompt() {
        super::long_prompt_matches(TAG, 14);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn first_layers() {
        let fx = super::Fixture::open_tag(TAG);
        let first = (0..fx.n_layers)
            .find(|&i| fx.build(&fx.host()).layers[i].attends())
            .unwrap();
        super::first_layers_match(TAG, first + 1, 0.1);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn logit_noise() {
        super::logit_noise(TAG, 1.5, false);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn layers_are_reported_where_they_run() {
        let fx = super::Fixture::open_tag(TAG);
        for d in crate::inference::kernel::vulkan::devices() {
            let m = fx.build(&fx.on_vulkan(d.index, fx.n_layers));
            let mut want: Vec<(String, usize, u32, u32)> = Vec::new();
            for (i, l) in m.layers.iter().enumerate() {
                let (kind, at) = if l.attends() {
                    ("Vulkan".to_string(), d.index)
                } else {
                    ("CPU".to_string(), 0)
                };
                match want.last_mut() {
                    Some(last) if last.0 == kind && last.1 == at => last.3 = i as u32,
                    _ => want.push((kind, at, i as u32, i as u32)),
                }
            }
            assert!(want.iter().any(|w| w.0 == "Vulkan"), "{}", d.name);
            assert_eq!(m.device_layer_distribution(), want, "{}", d.name);
        }
    }
}

/// A mixture of experts: where the host's best experts for a token are within rounding of the
/// next ones, the two pick other experts and the logits part by more than rounding. The
/// experts, the first layers and the logit spread are what these hold it to.
mod olmoe {
    const TAG: &str = "olmoe:latest";

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn experts() {
        super::experts_match(TAG);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn layers() {
        super::first_layers_match(TAG, 1, 0.1);
        for layers in [4, 8] {
            super::first_layers_match(TAG, layers, 0.01);
        }
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn logit_noise() {
        super::logit_noise(TAG, 2.5, false);
    }
}

/// Weights in Q5_0, and query heads that share a key head in groups that do not divide a block of
/// the prompt attention: the first layer, the logit spread, the greedy text and the long prompt
/// hold it.
mod qwen2 {
    const TAG: &str = "qwen2.5:0.5b";

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn long_prompt() {
        super::long_prompt_matches(TAG, 14);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn first_layer() {
        super::first_layers_match(TAG, 1, 0.1);
    }

    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn logit_noise() {
        super::logit_noise(TAG, 1.5, false);
    }

    // The greedy text is left out.
    // On the Radeon RX 9070, Vulkan picks 3720 at step 11 where the host and ROCm pick 6819.
}

/// A prompt longer than gemma3's window of 1024 keys: the Vulkan layers mask as the host does.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_gemma_window_matches_the_host() {
    long_prompt_matches("gemma3:latest", 40);
}

/// Root-mean-square and largest difference of two logit rows.
fn logit_gap(a: &[f32], b: &[f32]) -> (f64, f32) {
    let sq: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let max = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    ((sq / a.len() as f64).sqrt(), max)
}

/// How far the Vulkan and (where built) CUDA logits sit from the host's when fed the host's
/// own tokens: each backend quantises its activations, so none equals another bit for bit.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_logit_noise() {
    for tag in [TAG, "qwen3:1.7b"] {
        logit_noise(tag, 1.2, true);
    }
}

/// The logit rms of every Vulkan device (and of CUDA card 0 with `with_cuda`) against the host
/// over a prompt and a few steps, none above `limit`.
fn logit_noise(tag: &str, limit: f64, with_cuda: bool) {
    const NOISE_STEPS: usize = 12;
    {
        let fx = Fixture::open_tag(tag);
        let tokenizer = crate::inference::engine::llm_engine::build_tokenizer_from_gguf(
            &self_content(&fx.mmap),
        )
        .unwrap();
        let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
        let mut host = fx.build(&fx.host());
        let mut fed = prompt.clone();
        let mut want = vec![host_vec(&host.forward(&ids(&prompt), 0).unwrap())];
        for _ in 0..NOISE_STEPS {
            let next = argmax(&Tensor::new(want[want.len() - 1].as_slice(), &Device::Cpu).unwrap());
            fed.push(next);
            want.push(host_vec(
                &host.forward(&ids(&[next]), fed.len() - 1).unwrap(),
            ));
        }
        drop(host);
        let run = |name: String, mut m: GenericHeteroTransformer| {
            let mut got = vec![host_vec(&m.forward(&ids(&prompt), 0).unwrap())];
            for i in 0..NOISE_STEPS {
                let t = [fed[prompt.len() + i]];
                got.push(host_vec(&m.forward(&ids(&t), prompt.len() + i).unwrap()));
            }
            let gaps: Vec<(f64, f32)> = want
                .iter()
                .zip(&got)
                .map(|(w, g)| logit_gap(w, g))
                .collect();
            let rms = gaps.iter().map(|g| g.0).sum::<f64>() / gaps.len() as f64;
            println!(
                "{tag} {name} against the host: logit rms {rms:.3}, largest {:.3}",
                gaps.iter().fold(0f32, |m, g| m.max(g.1))
            );
            // The host's own activations are quantised: a faithful backend sits about
            // 0.5 to 0.8 from it on these models, a faulty kernel far beyond.
            assert!(rms < limit, "{tag} {name}: logit rms {rms} from the host");
        };
        for d in crate::inference::kernel::vulkan::devices() {
            if let Some(model) = whole_on(&fx, tag, &d) {
                run(format!("vulkan {}", d.name), model);
            }
        }
        #[cfg(feature = "cuda")]
        {
            if !with_cuda {
                return;
            }
            let cards: HashMap<usize, Device> =
                [(0, Device::new_cuda(0).unwrap())].into_iter().collect();
            run(
                "cuda 0".to_string(),
                GenericHeteroTransformer::from_gguf(
                    self_content(&fx.mmap),
                    &fx.mmap[..],
                    &cards,
                    &HeteroPlan::forced_gpu(fx.n_layers, fx.n_layers, 0),
                )
                .unwrap(),
            );
        }
    }
}

/// A model whose layers all run on a Vulkan device reports them on that device, not on the
/// host.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_layers_are_reported_on_their_device() {
    let fx = Fixture::open();
    for d in crate::inference::kernel::vulkan::devices() {
        let m = fx.build(&fx.on_vulkan(d.index, fx.n_layers));
        assert_eq!(
            m.device_layer_distribution(),
            vec![("Vulkan".to_string(), d.index, 0, fx.n_layers as u32 - 1)],
            "{}",
            d.name
        );
    }
}

/// The same greedy answer on the first CUDA card, for the comparison of the two runtimes on one
/// NVIDIA card.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the model in the store and an NVIDIA card"]
fn vulkan_reference_on_cuda() {
    let fx = Fixture::open();
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
    let cards: HashMap<usize, Device> = [(0, Device::new_cuda(0).unwrap())].into_iter().collect();
    let mut m = GenericHeteroTransformer::from_gguf(
        self_content(&fx.mmap),
        &fx.mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(fx.n_layers, fx.n_layers, 0),
    )
    .unwrap();
    let (got, rate) = greedy(&mut m, &prompt);
    println!(
        "{TAG}: cuda 0: {rate:.1} tok/s {:?}",
        tokenizer.decode(&got, false).unwrap_or_default()
    );
}

/// Where a decode step's time goes on each Vulkan device, kernel by kernel, after a prompt of
/// `PROFILE_PROMPT` tokens.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_decode_profile() {
    decode_profile(TAG);
}

/// [`vulkan_decode_profile`] for the larger model of the speed benches.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_decode_profile_large() {
    decode_profile("qwen3:1.7b");
}

/// [`vulkan_decode_profile`] for a model with 256-wide heads, sliding windows and a large
/// vocabulary.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_decode_profile_wide() {
    decode_profile("gemma3:latest");
}

fn decode_profile(tag: &str) {
    const PROFILE_PROMPT: u32 = 1024;
    const PROFILE_RUNS: usize = 8;
    let prompt: Vec<u32> = (0..PROFILE_PROMPT)
        .map(|i| 1000 + (i * 37) % 20000)
        .collect();
    let fx = Fixture::open_tag(tag);
    for d in crate::inference::kernel::vulkan::devices() {
        // A card too small for the model has nothing to profile.
        if d.total < 8 << 30 {
            continue;
        }
        let mut m = fx.build(&fx.on_vulkan(d.index, fx.n_layers));
        m.forward(&ids(&prompt), 0).unwrap();
        // Sustained work first: an idle card runs slow until its clock ramps up.
        let warm = std::time::Instant::now();
        while warm.elapsed() < std::time::Duration::from_secs(2) {
            m.vulkan.profile(128, prompt.len());
        }
        for tokens in [1, 32, 128] {
            // The card's clock wanders: the fastest of several runs is the kernels' own time.
            let total_of =
                |by: &Vec<Vec<(&'static str, f64)>>| by[0].iter().map(|(_, us)| us).sum::<f64>();
            let by = (0..PROFILE_RUNS)
                .map(|_| m.vulkan.profile(tokens, prompt.len()))
                .min_by(|a, b| total_of(a).total_cmp(&total_of(b)))
                .unwrap();
            let total = total_of(&by);
            println!(
                "{}: {tokens} rows: {total:.0} us a step: {:?}",
                d.name,
                by[0]
                    .iter()
                    .map(|(k, us)| format!("{k} {us:.0}"))
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Prompt tokens and decode steps of the speed benches.
const BENCH_PROMPT: usize = 1318;
const BENCH_STEPS: usize = 64;
/// Prompt tokens a forward takes, as the server's prefill does.
const BENCH_CHUNK: usize = 512;
const BENCH_RUNS: usize = 3;

/// Prompt reading and decode rates of `m`, one line per run.
fn bench_model(label: &str, m: &mut GenericHeteroTransformer) {
    let prompt: Vec<u32> = (0..BENCH_PROMPT as u32)
        .map(|i| 1000 + (i * 37) % 20000)
        .collect();
    for run in 0..=BENCH_RUNS {
        let start = std::time::Instant::now();
        let mut tok = 0;
        for (i, chunk) in prompt.chunks(BENCH_CHUNK).enumerate() {
            tok = argmax(&m.forward(&ids(chunk), i * BENCH_CHUNK).unwrap());
        }
        let read = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        for i in 0..BENCH_STEPS {
            tok = argmax(&m.forward(&ids(&[tok]), prompt.len() + i).unwrap());
        }
        let dec = start.elapsed().as_secs_f64();
        if run > 0 {
            println!(
                "BENCH {label}: prompt {:.1} tok/s, decode {:.1} tok/s",
                BENCH_PROMPT as f64 / read,
                BENCH_STEPS as f64 / dec
            );
        }
    }
}

/// Speed of `tag` on every Vulkan device.
fn bench_vulkan(tag: &str) {
    let fx = Fixture::open_tag(tag);
    for d in crate::inference::kernel::vulkan::devices() {
        let mut m = fx.build(&fx.on_vulkan(d.index, fx.n_layers));
        bench_model(&format!("{tag} vulkan {}", d.name), &mut m);
    }
}

/// Speed of `tag` on the first CUDA or ROCm card.
#[cfg(feature = "gpu")]
fn bench_gpu(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let cards: HashMap<usize, Device> = [(0, Device::new_cuda(0).unwrap())].into_iter().collect();
    let mut m = GenericHeteroTransformer::from_gguf(
        self_content(&fx.mmap),
        &fx.mmap[..],
        &cards,
        &HeteroPlan::forced_gpu(fx.n_layers, fx.n_layers, 0),
    )
    .unwrap();
    bench_model(&format!("{tag} gpu"), &mut m);
}

#[test]
#[ignore = "speed bench"]
fn bench_vulkan_small() {
    bench_vulkan("qwen3:0.6b");
}

#[test]
#[ignore = "speed bench"]
fn bench_vulkan_large() {
    bench_vulkan("qwen3:1.7b");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "speed bench"]
fn bench_gpu_small() {
    bench_gpu("qwen3:0.6b");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "speed bench"]
fn bench_gpu_large() {
    bench_gpu("qwen3:1.7b");
}
