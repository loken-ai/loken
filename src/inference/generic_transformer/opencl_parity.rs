//! A dense model placed on an OpenCL device answers as the host does: its first layer within a
//! tolerance, and its greedy text token for token. Every OpenCL GPU the loader lists is tried.
//!
//! Ignored by default: it needs the model in the store and an OpenCL device. Run with
//!   cargo test --lib --features opencl --profile gate opencl_ -- --ignored --test-threads=2

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

    /// Layers `[0, upto)` on OpenCL device `device`, the rest on the host.
    fn on_opencl(&self, device: usize, upto: usize) -> HeteroPlan {
        let segment = |kind, layer_start, layer_end| HeteroSegment {
            kind,
            layer_start,
            layer_end,
            free_memory_bytes: u64::MAX,
        };
        let mut segments = vec![segment(DeviceKind::OpenCL(device), 0, upto)];
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

/// Layers `[0, layers)` of the model in plain f32 over the file's dequantised weights: the
/// reference the OpenCL layers are held to, free of the host's quantised activations.
fn reference_layers(fx: &Fixture, x: &[f32], tokens: usize, layers: usize) -> Vec<f32> {
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
    let mm = |x: &[f32], wt: &[f32], rows: usize, bias: Option<&[f32]>| -> Vec<f32> {
        let cols = x.len() / tokens;
        (0..tokens)
            .flat_map(|t| {
                (0..rows).map(move |r| {
                    (0..cols)
                        .map(|k| wt[r * cols + k] as f64 * x[t * cols + k] as f64)
                        .sum::<f64>() as f32
                        + bias.map_or(0.0, |b| b[r])
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
    let mut x = x.to_vec();
    for li in 0..layers {
        let p = format!("blk.{li}");
        let bias = |n: &str| has(&format!("{p}.{n}")).then(|| w(&format!("{p}.{n}")));
        let an = w(&format!("{p}.attn_norm.weight"));
        let xn: Vec<f32> = x.chunks(h).flat_map(|r| rms(r, &an)).collect();
        let mut q = mm(
            &xn,
            &w(&format!("{p}.attn_q.weight")),
            nh * hd,
            bias("attn_q.bias").as_deref(),
        );
        let mut k = mm(
            &xn,
            &w(&format!("{p}.attn_k.weight")),
            nkv * hd,
            bias("attn_k.bias").as_deref(),
        );
        let v = mm(
            &xn,
            &w(&format!("{p}.attn_v.weight")),
            nkv * hd,
            bias("attn_v.bias").as_deref(),
        );
        for (buf, norm) in [
            (&mut q, "attn_q_norm.weight"),
            (&mut k, "attn_k_norm.weight"),
        ] {
            let g = bias(norm);
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
        let o = mm(&attn, &w(&format!("{p}.attn_output.weight")), h, None);
        let x1: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
        let fnw = w(&format!("{p}.ffn_norm.weight"));
        let xn2: Vec<f32> = x1.chunks(h).flat_map(|r| rms(r, &fnw)).collect();
        let gate = mm(&xn2, &w(&format!("{p}.ffn_gate.weight")), cfg.ffn_dim, None);
        let up = mm(&xn2, &w(&format!("{p}.ffn_up.weight")), cfg.ffn_dim, None);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let down = mm(&act, &w(&format!("{p}.ffn_down.weight")), h, None);
        x = x1.iter().zip(&down).map(|(a, b)| a + b).collect();
    }
    x
}

/// The logits of the last row after all layers, in plain f32.
fn reference_logits(fx: &Fixture, tokens: &[u32]) -> Vec<f32> {
    let c = self_content(&fx.mmap);
    let cfg = GenericTransformerConfig::from_gguf(
        &c,
        &c.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let w = |name: &str| -> Vec<f32> {
        host_vec(
            &c.tensor(&mut std::io::Cursor::new(&fx.mmap[..]), name, &Device::Cpu)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap(),
        )
    };
    let h = cfg.embedding_length;
    let embed = w("token_embd.weight");
    let x: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| embed[t as usize * h..(t as usize + 1) * h].to_vec())
        .collect();
    let out = reference_layers(fx, &x, tokens.len(), fx.n_layers);
    let last = &out[(tokens.len() - 1) * h..];
    let g = w("output_norm.weight");
    let r =
        1.0 / (last.iter().map(|a| a * a).sum::<f32>() / h as f32 + cfg.rms_norm_eps as f32).sqrt();
    let normed: Vec<f32> = last.iter().zip(&g).map(|(a, b)| a * r * b).collect();
    let head = if c.tensor_infos.contains_key("output.weight") {
        w("output.weight")
    } else {
        embed
    };
    head.chunks(h)
        .map(|row| {
            row.iter()
                .zip(&normed)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn reference_layer0(fx: &Fixture, x: &[f32], tokens: usize) -> Vec<f32> {
    reference_layers(fx, x, tokens, 1)
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
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_first_layer_matches_the_host() {
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
    for d in devices() {
        let mut m = fx.build(&fx.on_opencl(d.0, 1));
        let (got, _) = m
            .run_layers_in(h.clone(), None, 0, prompt.len(), 0, 1)
            .unwrap();
        let err = max_err(&host_vec(&got));
        println!(
            "{TAG} layer 0 on {}: max error {err:.3e} of a change {delta:.3e}",
            d.1
        );
        assert!(
            err <= max_err(&host_rows),
            "{}: layer 0 differs from the f32 reference by {err} against a change of {delta}",
            d.1
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

/// The host's greedy answer and, fed the same tokens, the OpenCL one: at every step the
/// OpenCL argmax is the host's, unless the host's two best tokens tie. The free-running answers
/// and the decode rates are printed beside.
#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_greedy_text_matches_the_host() {
    let fx = Fixture::open();
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
    let decode = |t: &[u32]| tokenizer.decode(t, false).unwrap_or_default();
    let (want, host_rate) = greedy(&mut fx.build(&fx.host()), &prompt);
    println!("{TAG}: host {host_rate:.1} tok/s {:?}", decode(&want));
    for d in devices() {
        let plan = fx.on_opencl(d.0, fx.n_layers);
        let (got, rate) = greedy(&mut fx.build(&plan), &prompt);
        println!("{TAG}: opencl {}: {rate:.1} tok/s {:?}", d.1, decode(&got));
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
                "{}: step {step}: OpenCL picks {v1}, the host {h1} by {margin}",
                d.1
            );
            if v1 != h1 {
                println!(
                    "{}: step {step}: a tie ({margin:.3}) broken the other way",
                    d.1
                );
            }
            fed.push(h1 as u32);
            let t = [h1 as u32];
            h = host_vec(&host.forward(&ids(&t), fed.len() - 1).unwrap());
            v = host_vec(&m.forward(&ids(&t), fed.len() - 1).unwrap());
        }
    }
}

/// A prompt that crosses the chunk a step carries and several tiles of keys: the OpenCL argmax
/// is the host's at the prompt's last row and at the steps that follow, unless the host's two
/// best tokens tie.
#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_long_prompt_matches_the_host() {
    let fx = Fixture::open();
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let text = TEXT.repeat(14);
    let prompt = tokenizer
        .encode(text.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(prompt.len() > 2 * 128, "{} tokens", prompt.len());
    for d in devices() {
        let mut host = fx.build(&fx.host());
        let mut m = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        let mut fed = prompt.clone();
        let mut h = host_vec(&host.forward(&ids(&prompt), 0).unwrap());
        let mut v = host_vec(&m.forward(&ids(&prompt), 0).unwrap());
        for step in 0..8 {
            let (h1, h2) = top2(&h);
            let (v1, _) = top2(&v);
            let margin = h[h1] - h[h2];
            assert!(
                v1 == h1 || margin < TIE,
                "{}: step {step} of {} tokens: OpenCL picks {v1}, the host {h1} by {margin}",
                d.1,
                prompt.len()
            );
            fed.push(h1 as u32);
            let t = [h1 as u32];
            h = host_vec(&host.forward(&ids(&t), fed.len() - 1).unwrap());
            v = host_vec(&m.forward(&ids(&t), fed.len() - 1).unwrap());
        }
        println!("{}: {} tokens: ok", d.1, prompt.len());
    }
}

/// The OpenCL GPUs the loader lists, as `(index, name)`.
fn devices() -> Vec<(usize, String)> {
    let found: Vec<(usize, String)> = crate::inference::kernel::opencl::enumerate_opencl_devices()
        .into_iter()
        .enumerate()
        .map(|(i, id)| {
            (
                i,
                opencl3::device::Device::new(id)
                    .name()
                    .unwrap_or_else(|_| "unknown".to_string()),
            )
        })
        .collect();
    assert!(!found.is_empty(), "no OpenCL device: these tests need one");
    found
}

/// The decode rate of a model: a prompt, then greedy steps, tokens per second.
fn bench(label: &str, m: &mut GenericHeteroTransformer, prompt: &[u32]) {
    let start = std::time::Instant::now();
    let mut next = argmax(&m.forward(&ids(prompt), 0).unwrap());
    let prefill = start.elapsed().as_secs_f64();
    let steps = 64;
    let start = std::time::Instant::now();
    for i in 0..steps {
        next = argmax(&m.forward(&ids(&[next]), prompt.len() + i).unwrap());
    }
    let rate = steps as f64 / start.elapsed().as_secs_f64();
    println!(
        "{label}: prefill {} tokens {:.1} tok/s, decode {rate:.1} tok/s",
        prompt.len(),
        prompt.len() as f64 / prefill
    );
}

/// A decode step with one launch to a layer gives the logits of the launches it replaces.
fn single_launch_against_launches(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
    for d in devices() {
        let mut one = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        let mut many = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        println!(
            "{tag} on {}: single launch available: {}, {} workgroups",
            d.1,
            one.opencl.single_launch(),
            one.opencl.single_launch_groups()
        );
        if !one.opencl.single_launch() {
            continue;
        }
        many.opencl.set_single_launch(false);
        let mut fed = prompt.clone();
        let mut a = host_vec(&one.forward(&ids(&prompt), 0).unwrap());
        let mut b = host_vec(&many.forward(&ids(&prompt), 0).unwrap());
        for step in 0..STEPS {
            let gap = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            println!("{tag} step {step}: largest logit difference {gap:.2e}");
            // Both quantise the activations as the host does, so a rounding that lands on the
            // other side of a boundary moves a logit by about what the host's own does.
            let (b1, b2) = top2(&b);
            let (a1, _) = top2(&a);
            assert!(gap < 4.0, "step {step}: {gap}");
            assert!(
                a1 == b1 || b[b1] - b[b2] < 1.0,
                "step {step}: {a1} against {b1}"
            );
            let next = b1 as u32;
            fed.push(next);
            a = host_vec(&one.forward(&ids(&[next]), fed.len() - 1).unwrap());
            b = host_vec(&many.forward(&ids(&[next]), fed.len() - 1).unwrap());
        }
    }
}

#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_single_launch_matches_qwen3() {
    single_launch_against_launches("qwen3:0.6b");
}

#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_single_launch_matches_qwen25() {
    single_launch_against_launches("qwen2.5:0.5b");
}

fn bench_tag(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    // A short prompt for the decode rate, a long one for the prompt rate.
    for repeat in [1usize, 24] {
        let text = TEXT.repeat(repeat);
        let prompt = tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        bench(&format!("{tag} host"), &mut fx.build(&fx.host()), &prompt);
        for d in devices() {
            let mut m = fx.build(&fx.on_opencl(d.0, fx.n_layers));
            if m.opencl.single_launch() {
                m.opencl.set_single_launch(false);
                bench(
                    &format!("{tag} opencl {}, a launch to a phase", d.1),
                    &mut m,
                    &prompt,
                );
                m.opencl.set_single_launch(true);
                let mut m2 = fx.build(&fx.on_opencl(d.0, fx.n_layers));
                bench(
                    &format!("{tag} opencl {}, a launch to a layer", d.1),
                    &mut m2,
                    &prompt,
                );
            } else {
                bench(&format!("{tag} opencl {}", d.1), &mut m, &prompt);
            }
        }
    }
}

#[test]
#[ignore = "speed bench"]
fn bench_opencl_qwen3() {
    bench_tag("qwen3:0.6b");
}

#[test]
#[ignore = "speed bench"]
fn bench_opencl_qwen25() {
    bench_tag("qwen2.5:0.5b");
}

/// The logits of the last prompt row against the f32 reference: the OpenCL stack is held to
/// the reference as closely as the host is, or closer.
fn logits_against_reference(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
    let want = reference_logits(&fx, &prompt);
    let gap = |a: &[f32]| {
        a.iter()
            .zip(&want)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    };
    let host = host_vec(&fx.build(&fx.host()).forward(&ids(&prompt), 0).unwrap());
    let host_gap = gap(&host);
    println!("{tag}: host against the reference {host_gap:.3}");
    for d in devices() {
        let got = host_vec(
            &fx.build(&fx.on_opencl(d.0, fx.n_layers))
                .forward(&ids(&prompt), 0)
                .unwrap(),
        );
        let g = gap(&got);
        println!("{tag}: {} against the reference {g:.3}", d.1);
        assert!(
            g <= host_gap.max(0.05),
            "{}: {g} against the host's {host_gap}",
            d.1
        );
    }
}

#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_logits_match_the_reference_qwen3() {
    logits_against_reference("qwen3:0.6b");
}

#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_logits_match_the_reference_qwen25() {
    logits_against_reference("qwen2.5:0.5b");
}

/// Where a single-launch layer first departs from the launches it replaces: the hidden state
/// after each of the first layers of a decode step.
#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_single_launch_layer_by_layer() {
    let fx = Fixture::open_tag("qwen3:0.6b");
    let prompt: Vec<u32> = (0..9u32).map(|i| 1000 + (i * 37) % 20000).collect();
    for d in devices() {
        let mut one = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        let mut many = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        if !one.opencl.single_launch() {
            continue;
        }
        many.opencl.set_single_launch(false);
        one.forward(&ids(&prompt), 0).unwrap();
        many.forward(&ids(&prompt), 0).unwrap();
        let next = [prompt[prompt.len() - 1] + 1];
        let pos = prompt.len();
        let (h, _) = one.embed_stage(&ids(&next), pos).unwrap();
        for upto in [4, 5, 6, 7, 8] {
            let a = one.opencl.run(h.clone(), pos, 0, upto).unwrap();
            let b = many.opencl.run(h.clone(), pos, 0, upto).unwrap();
            let (a, b) = (host_vec(&a), host_vec(&b));
            let gap = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            let scale = b.iter().fold(0f32, |m, y| m.max(y.abs()));
            println!("after {upto} layers: largest difference {gap:.3e} of {scale:.3e}");
        }
    }
}
