//! A dense model placed on an OpenCL device answers as the host does: its first layer within a
//! tolerance, and its greedy text token for token. Every OpenCL GPU the loader lists is tried.
//!
//! Ignored by default: it needs the model in the store and an OpenCL device. Run with
//!   cargo test --lib --features opencl --profile gate opencl_ -- --ignored --test-threads=2

use super::native_judge::*;
use super::*;
use crate::inference::place::layer_executor::{DeviceKind, HeteroSegment};
use crate::tensor::quantized::GgmlDType;

/// The model the tests read from the store.
const TAG: &str = "qwen3:0.6b";

/// The tokens the greedy answer runs for.
const STEPS: usize = 32;

impl Fixture {
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

/// How a product reads its activation rows, from its weights' type and the rows' length.
type Read = fn(GgmlDType, &[f32], usize) -> Vec<f32>;

/// Activation rows read as they are, for the products in plain f32.
fn exact(_: GgmlDType, x: &[f32], _: usize) -> Vec<f32> {
    x.to_vec()
}

/// Activation rows read as the OpenCL products read them.
const CARD: Read = crate::inference::kernel::opencl_stack::host_quantised;

/// Layers `[0, layers)` of the model in f32 over dequantised weights, each product reading its
/// activations through `read`.
/// The weights are the file's, or, with `held`, each in the format the layers of that built
/// model give it, as a device that reads the file holds it.
/// With [`exact`] and the file's weights it is the reference the OpenCL layers are held to,
/// free of the host's quantised activations.
/// With [`CARD`] and a model built for an OpenCL device it is that device's arithmetic,
/// computed on the host.
fn reference_layers(
    fx: &Fixture,
    x: &[f32],
    tokens: usize,
    layers: usize,
    read: Read,
    held: Option<&GenericHeteroTransformer>,
) -> Vec<f32> {
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
    let mm = |x: &[f32], name: &str, rows: usize, bias: Option<&[f32]>| -> Vec<f32> {
        let cols = x.len() / tokens;
        let (wt, dtype) = match held {
            Some(m) => {
                let stored = c
                    .tensor(&mut std::io::Cursor::new(&fx.mmap[..]), name, &Device::Cpu)
                    .unwrap();
                let q = super::loader::in_layer_format(&m.layers, name, stored).unwrap();
                (host_vec(&q.dequantize(&Device::Cpu).unwrap()), q.dtype())
            }
            None => (w(name), c.tensor_infos[name].ggml_dtype),
        };
        let x = read(dtype, x, cols);
        (0..tokens * rows)
            .into_par_iter()
            .map(|i| {
                let (t, r) = (i / rows, i % rows);
                (0..cols)
                    .map(|k| wt[r * cols + k] as f64 * x[t * cols + k] as f64)
                    .sum::<f64>() as f32
                    + bias.map_or(0.0, |b| b[r])
            })
            .collect()
    };
    // The file's frequency factors divide each pair's frequency.
    let factors = has("rope_freqs.weight").then(|| w("rope_freqs.weight"));
    let rope = |v: &mut [f32], pos: usize| {
        let half = hd / 2;
        for i in 0..half {
            let factor = factors.as_ref().map_or(1.0, |f| f[i]);
            let theta =
                pos as f32 * cfg.rope_freq_base.powf(-((2 * i) as f32) / hd as f32) / factor;
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
            &format!("{p}.attn_q.weight"),
            nh * hd,
            bias("attn_q.bias").as_deref(),
        );
        let mut k = mm(
            &xn,
            &format!("{p}.attn_k.weight"),
            nkv * hd,
            bias("attn_k.bias").as_deref(),
        );
        let v = mm(
            &xn,
            &format!("{p}.attn_v.weight"),
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
        let o = mm(&attn, &format!("{p}.attn_output.weight"), h, None);
        let x1: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
        let fnw = w(&format!("{p}.ffn_norm.weight"));
        let xn2: Vec<f32> = x1.chunks(h).flat_map(|r| rms(r, &fnw)).collect();
        let gate = mm(&xn2, &format!("{p}.ffn_gate.weight"), cfg.ffn_dim, None);
        let up = mm(&xn2, &format!("{p}.ffn_up.weight"), cfg.ffn_dim, None);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let down = mm(&act, &format!("{p}.ffn_down.weight"), h, None);
        x = x1.iter().zip(&down).map(|(a, b)| a + b).collect();
    }
    x
}

/// The logits of the last row after all layers, in plain f32.
fn reference_logits(fx: &Fixture, tokens: &[u32]) -> Vec<f32> {
    reference_rows(fx, tokens, tokens.len() - 1, exact, None).remove(0)
}

/// The logits of rows `[from, tokens.len())` after all layers, each product reading its
/// activations through `read` and its weights as [`reference_layers`] does.
fn reference_rows(
    fx: &Fixture,
    tokens: &[u32],
    from: usize,
    read: Read,
    held: Option<&GenericHeteroTransformer>,
) -> Vec<Vec<f32>> {
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
    drop(embed);
    let out = reference_layers(fx, &x, tokens.len(), fx.n_layers, read, held);
    let g = w("output_norm.weight");
    let head_name = if c.tensor_infos.contains_key("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let head = w(head_name);
    out[from * h..]
        .chunks(h)
        .map(|last| {
            let r = 1.0
                / (last.iter().map(|a| a * a).sum::<f32>() / h as f32 + cfg.rms_norm_eps as f32)
                    .sqrt();
            let normed: Vec<f32> = last.iter().zip(&g).map(|(a, b)| a * r * b).collect();
            let normed = read(c.tensor_infos[head_name].ggml_dtype, &normed, h);
            head.par_chunks(h)
                .map(|row| {
                    row.iter()
                        .zip(&normed)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>() as f32
                })
                .collect()
        })
        .collect()
}

/// The picks of the OpenCL arithmetic computed on the host at the rows of `host`, fed the
/// host's tokens, with the weights of the model placed whole on OpenCL device `device`.
fn card_picks(fx: &Fixture, device: usize, prompt: &[u32], host: &[Vec<f32>]) -> Vec<usize> {
    let m = fx.build(&fx.on_opencl(device, fx.n_layers));
    let mut tokens = prompt.to_vec();
    tokens.extend(host[..host.len() - 1].iter().map(|row| top2(row).0 as u32));
    reference_rows(fx, &tokens, prompt.len() - 1, CARD, Some(&m))
        .iter()
        .map(|row| top2(row).0)
        .collect()
}

fn reference_layer0(
    fx: &Fixture,
    x: &[f32],
    tokens: usize,
    read: Read,
    held: Option<&GenericHeteroTransformer>,
) -> Vec<f32> {
    reference_layers(fx, x, tokens, 1, read, held)
}

fn prompt() -> Vec<u32> {
    (0..24u32).map(|i| 1000 + (i * 37) % 20000).collect()
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
    let fx = Fixture::open_tag(TAG);
    let prompt = prompt();
    let mut host = fx.build(&fx.host());
    let (h, _) = host.embed_stage(&ids(&prompt), 0).unwrap();
    let (want, _) = host
        .run_layers_in(h.clone(), None, 0, prompt.len(), 0, 1)
        .unwrap();
    let host_rows = host_vec(&want);
    let embed = host_vec(&h);
    let reference = reference_layer0(&fx, &embed, prompt.len(), exact, None);
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
    let mut beyond = Vec::new();
    for d in devices() {
        let mut m = fx.build(&fx.on_opencl(d.0, 1));
        let (got, _) = m
            .run_layers_in(h.clone(), None, 0, prompt.len(), 0, 1)
            .unwrap();
        let got = host_vec(&got);
        // The device's arithmetic computed on the host: the error its own products give.
        let card = reference_layer0(&fx, &embed, prompt.len(), CARD, Some(&m));
        let card_err = max_err(&card);
        println!(
            "{TAG} layer 0 in the arithmetic of {} on the host: max error {card_err:.3e}",
            d.1
        );
        // The native runtime of the card, where the build has one, sets the bound.
        // Without it, the device's arithmetic computed on the host sets it.
        let bound = fx.first_layer_bound(TAG, &h, prompt.len(), card_err, max_err);
        let err = max_err(&got);
        let off = got
            .iter()
            .zip(&card)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "{TAG} layer 0 on {}: max error {err:.3e} of a change {delta:.3e}, {off:.3e} from its arithmetic on the host",
            d.1
        );
        if err > bound {
            beyond.push(format!("{}: error {err}, bound {bound}", d.1));
        }
    }
    assert!(
        beyond.is_empty(),
        "layer 0 differs from the f32 reference beyond its bound on {}",
        beyond.join("; ")
    );
}

/// The chat prompt the text tests answer.
const TEXT: &str =
    "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";

/// The host's greedy answer and, fed the same tokens, the OpenCL one: at every step the
/// OpenCL argmax is the host's, or the native card's leaves the host's too. The free-running
/// answers and the decode rates are printed beside.
#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_greedy_text_matches_the_host() {
    greedy_text_against_host(TAG);
}

/// The greedy text test on a model whose RoPE has frequency factors.
#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_greedy_text_matches_the_host_llama32() {
    greedy_text_against_host("llama3.2:1b");
}

fn greedy_text_against_host(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let prompt = tokenizer.encode(TEXT, false).unwrap().get_ids().to_vec();
    let decode = |t: &[u32]| tokenizer.decode(t, false).unwrap_or_default();
    let (want, host_rate) = greedy(&mut fx.build(&fx.host()), &prompt);
    println!("{tag}: host {host_rate:.1} tok/s {:?}", decode(&want));
    let host = fx.host_rows(&prompt, STEPS);
    let mut picks = Vec::new();
    for d in devices() {
        let mut whole = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        // One copy serves both runs.
        // The teacher-forced run comes first, on the fresh copy, fed the host's tokens.
        picks.push((
            format!("{tag} opencl {}", d.1),
            forced_picks(&mut whole, &prompt, &host),
        ));
        let (got, rate) = greedy(&mut whole, &prompt);
        println!("{tag}: opencl {}: {rate:.1} tok/s {:?}", d.1, decode(&got));
    }
    // The native copy loads once every OpenCL copy is freed.
    let native = native_or_card(&fx, &prompt, &host);
    judge_devices(&host, &native, &picks);
}

/// A prompt that crosses the chunk a step carries and several tiles of keys: the OpenCL argmax
/// is the host's at the prompt's last row and at the steps that follow, or the native card's
/// leaves the host's too.
#[test]
#[ignore = "needs the model in the store and an OpenCL device"]
fn opencl_long_prompt_matches_the_host() {
    let fx = Fixture::open_tag(TAG);
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
    let host = fx.host_rows(&prompt, 8);
    let mut picks = Vec::new();
    for d in devices() {
        let mut m = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        picks.push((
            format!("{TAG} opencl {} with {} tokens", d.1, prompt.len()),
            forced_picks(&mut m, &prompt, &host),
        ));
    }
    // The native copy loads once every OpenCL copy is freed.
    let native = native_or_card(&fx, &prompt, &host);
    judge_devices(&host, &native, &picks);
}

/// The native card's picks at the rows of `host`.
/// Without a native card, the picks of the OpenCL arithmetic computed on the host, with the
/// weights of the model placed whole on the first OpenCL device.
fn native_or_card(fx: &Fixture, prompt: &[u32], host: &[Vec<f32>]) -> Native {
    match fx.native_picks(prompt, host) {
        Native::Absent => Native::Arithmetic(card_picks(fx, devices()[0].0, prompt, host)),
        n => n,
    }
}

/// Judges the picks of every device, and fails naming each device whose picks do not hold.
fn judge_devices(host: &[Vec<f32>], native: &Native, picks: &[(String, Vec<usize>)]) {
    let failed: Vec<String> = picks
        .iter()
        .filter_map(|(at, got)| {
            let gaps = pick_gaps(at, host, native, got);
            println!("{at}: {}", if gaps.is_empty() { "ok" } else { "fails" });
            (!gaps.is_empty()).then(|| format!("{at}: {}", gaps.join("; ")))
        })
        .collect();
    assert!(failed.is_empty(), "{}", failed.join(" | "));
}

/// Every OpenCL GPU, as `(index, name)`.
/// A card a native runtime also drives is one of them: the tests compare both on it.
fn devices() -> Vec<(usize, String)> {
    let found: Vec<(usize, String)> = crate::inference::kernel::opencl::layer_devices()
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
    let cpu = cpu_seconds();
    let start = std::time::Instant::now();
    for i in 0..steps {
        next = argmax(&m.forward(&ids(&[next]), prompt.len() + i).unwrap());
    }
    let rate = steps as f64 / start.elapsed().as_secs_f64();
    let cpu_ms = (cpu_seconds() - cpu) * 1e3 / steps as f64;
    println!(
        "{label}: prefill {} tokens {:.1} tok/s, decode {rate:.1} tok/s, {cpu_ms:.1} ms of CPU a token",
        prompt.len(),
        prompt.len() as f64 / prefill
    );
}

/// The CPU time the process has used, every thread counted, in seconds.
fn cpu_seconds() -> f64 {
    // SAFETY: getrusage fills the zeroed struct it is given.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
    t(u.ru_utime) + t(u.ru_stime)
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
            "{tag} on {}: single launch available: {}, {} workgroups, {}",
            d.1,
            one.opencl.single_launch(),
            one.opencl.single_launch_groups(),
            one.opencl.single_launch_facts()
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
            // Both quantise the activations alike, so a rounding that lands on the other side
            // of a boundary moves a logit by about what one quantised product does.
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
                for _ in 0..2 {
                    let mut m2 = fx.build(&fx.on_opencl(d.0, fx.n_layers));
                    bench(
                        &format!(
                            "{tag} opencl {}, a launch to a layer of {} workgroups",
                            d.1,
                            m2.opencl.single_launch_groups()
                        ),
                        &mut m2,
                        &prompt,
                    );
                }
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

/// Where the time of a prompt goes on each OpenCL device: each stage of the stack waits for the
/// device and takes its time, against the same prompt untraced.
#[test]
#[ignore = "speed bench"]
fn opencl_prompt_trace() {
    let fx = Fixture::open_tag(TAG);
    let tokenizer =
        crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&self_content(&fx.mmap))
            .unwrap();
    let text = TEXT.repeat(24);
    let prompt = tokenizer
        .encode(text.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    for d in devices() {
        let mut m = fx.build(&fx.on_opencl(d.0, fx.n_layers));
        m.forward(&ids(&prompt), 0).unwrap();
        let start = std::time::Instant::now();
        m.forward(&ids(&prompt), 0).unwrap();
        let plain = start.elapsed().as_secs_f64();
        m.opencl.start_trace();
        let start = std::time::Instant::now();
        m.forward(&ids(&prompt), 0).unwrap();
        let traced = start.elapsed().as_secs_f64() * 1e3;
        let stages = m.opencl.take_trace();
        let in_stack = stages.iter().map(|s| s.1).sum::<f64>() / 1e3;
        println!(
            "{TAG} on {}: {} tokens, {:.1} tok/s untraced; traced {traced:.0} ms, {in_stack:.0} ms in the stack",
            d.1,
            prompt.len(),
            prompt.len() as f64 / plain
        );
        for (name, us, n) in stages {
            println!(
                "  {name:<20} {:>9.1} ms {n:>6} passes {:>9.1} us a pass",
                us / 1e3,
                us / n as f64
            );
        }
    }
}
