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
        let path = model_blob(TAG);
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
        GenericHeteroTransformer::from_gguf(
            self_content(&self.mmap),
            &self.mmap[..],
            &HashMap::new(),
            plan,
        )
        .unwrap()
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
    println!(
        "{TAG} layer 0 on the host: max error {:.3e} of a change {delta:.3e}",
        max_err(&host_rows)
    );
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
            err <= delta * 1e-3,
            "{}: layer 0 differs from the f32 reference by {err} against a change of {delta}",
            d.name
        );
    }
}

/// The chat prompt the text tests answer.
const TEXT: &str =
    "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";

/// Logits closer than this between the host's two best tokens are a tie: which one a backend
/// picks there is rounding, not a fault.
const TIE: f32 = 0.1;

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

/// The host's greedy answer and, fed the same tokens, the Vulkan one: at every step the
/// Vulkan argmax is the host's, unless the host's two best tokens tie. The free-running answers
/// and the decode rates are printed beside.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_greedy_text_matches_the_host() {
    let fx = Fixture::open();
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
    let decode = |t: &[u32]| tokenizer.decode(t, false).unwrap_or_default();
    let (want, host_rate) = greedy(&mut fx.build(&fx.host()), &prompt);
    println!("{TAG}: host {host_rate:.1} tok/s {:?}", decode(&want));
    for d in crate::inference::kernel::vulkan::devices() {
        let plan = fx.on_vulkan(d.index, fx.n_layers);
        let (got, rate) = greedy(&mut fx.build(&plan), &prompt);
        println!(
            "{TAG}: vulkan {}: {rate:.1} tok/s {:?}",
            d.name,
            decode(&got)
        );
        // Teacher-forced: fresh models both fed the host's tokens.
        let mut host = fx.build(&fx.host());
        let mut m = fx.build(&plan);
        let mut fed = prompt.clone();
        let mut h = host_vec(&host.forward(&ids(&prompt), 0).unwrap());
        let mut v = host_vec(&m.forward(&ids(&prompt), 0).unwrap());
        for step in 0..STEPS {
            let (h1, h2) = top2(&h);
            let (v1, _) = top2(&v);
            let margin = h[h1] - h[h2];
            assert!(
                v1 == h1 || margin < TIE,
                "{}: step {step}: Vulkan picks {v1}, the host {h1} by {margin}",
                d.name
            );
            if v1 != h1 {
                println!(
                    "{}: step {step}: a tie ({margin:.3}) broken the other way",
                    d.name
                );
            }
            fed.push(h1 as u32);
            let t = [h1 as u32];
            h = host_vec(&host.forward(&ids(&t), fed.len() - 1).unwrap());
            v = host_vec(&m.forward(&ids(&t), fed.len() - 1).unwrap());
        }
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

/// Where a decode step's time goes on each Vulkan device, kernel by kernel.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_decode_profile() {
    let fx = Fixture::open();
    for d in crate::inference::kernel::vulkan::devices() {
        let mut m = fx.build(&fx.on_vulkan(d.index, fx.n_layers));
        m.forward(&ids(&prompt()), 0).unwrap();
        let by = m.vulkan.profile(prompt().len());
        let total: f64 = by[0].iter().map(|(_, us)| us).sum();
        println!(
            "{}: {total:.0} us a step: {:?}",
            d.name,
            by[0]
                .iter()
                .map(|(k, us)| format!("{k} {us:.0}"))
                .collect::<Vec<_>>()
        );
    }
}
