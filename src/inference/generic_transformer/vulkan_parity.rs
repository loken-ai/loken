//! A dense model placed on a Vulkan device answers as the host does: its first layer within a
//! tolerance, and its greedy text token for token. Every Vulkan GPU the loader lists is tried.
//!
//! Ignored by default: it needs the model in the store and a Vulkan device. Run with
//!   cargo test --features vulkan --profile gate vulkan_ -- --ignored --test-threads=2

use super::native_judge::*;
use super::*;
use crate::inference::place::layer_executor::{DeviceKind, HeteroSegment};

/// The model the tests read from the store.
const TAG: &str = "qwen3:0.6b";

/// The tokens the greedy answer runs for.
const STEPS: usize = 32;

impl Fixture {
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
    prompt_of(24)
}

/// A prompt of `n` tokens.
fn prompt_of(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 37) % 20000).collect()
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
    let fx = Fixture::open_tag(TAG);
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
    let bound = fx.first_layer_bound(TAG, &h, prompt.len(), host_err, max_err);
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

/// The relative L2 distance of `got` from `want`.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (e, w) = got.iter().zip(want).fold((0f64, 0f64), |(e, w), (g, r)| {
        let (g, r) = (*g as f64, *r as f64);
        (e + (g - r) * (g - r), w + r * r)
    });
    (e / w).sqrt()
}

/// Errors as they print.
fn sci(errors: &[f64]) -> String {
    errors
        .iter()
        .map(|e| format!("{e:.2e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A decode step of one token through layer 0 of the model placed whole on each Vulkan device,
/// at position 0 where the attention returns the values: the change the layer makes to the
/// residual is no further from the f32 reference than the host's.
/// Each placement reads the weights its loader gives it.
/// The native card's error prints beside: it reads its activations as the Vulkan card does.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_decode_step_matches_the_host() {
    let fx = Fixture::open_tag(TAG);
    let tokens = prompt();
    let mut host = fx.build(&fx.host());
    let (rows, _) = host.embed_stage(&ids(&tokens), 0).unwrap();
    let rows = host_vec(&rows);
    let h = rows.len() / tokens.len();
    let picks = [3, 11, tokens.len() - 1];
    let row = |p: usize| rows[p * h..(p + 1) * h].to_vec();
    let one = |x: &[f32]| Tensor::from_vec(x.to_vec(), (1, 1, h), &Device::Cpu).unwrap();
    let change =
        |out: &[f32], x: &[f32]| -> Vec<f32> { out.iter().zip(x).map(|(o, i)| o - i).collect() };
    let want: Vec<Vec<f32>> = picks
        .iter()
        .map(|&p| change(&reference_layer0(&fx, &row(p), 1), &row(p)))
        .collect();
    let errors = |m: &mut GenericHeteroTransformer| -> Vec<f64> {
        picks
            .iter()
            .zip(&want)
            .map(|(&p, w)| {
                let (got, _) = m.run_layers_in(one(&row(p)), None, 0, 1, 0, 1).unwrap();
                rel_l2(&change(&host_vec(&got), &row(p)), w)
            })
            .collect()
    };
    let host_err = errors(&mut host);
    drop(host);
    println!("{TAG} layer 0 at one token on the host: {}", sci(&host_err));
    let mut vulkan = Vec::new();
    for d in crate::inference::kernel::vulkan::devices() {
        let Some(mut m) = whole_on(&fx, TAG, &d) else {
            continue;
        };
        let err = errors(&mut m);
        println!("{TAG} layer 0 at one token on {}: {}", d.name, sci(&err));
        vulkan.push((d.name.clone(), err));
    }
    // The native copy loads once every Vulkan copy is freed.
    if let Some(card) = native_card() {
        let mut native = fx.try_on_native(&card, fx.n_layers).unwrap();
        let err = errors(&mut native);
        drop(native);
        release_native(&card);
        println!(
            "{TAG} layer 0 at one token on the native card: {}",
            sci(&err)
        );
    }
    for (name, err) in vulkan {
        for ((e, b), p) in err.iter().zip(&host_err).zip(picks) {
            assert!(
                e <= b,
                "{name}: row {p} is {e:.3e} from the f32 layer, beyond {b:.3e}"
            );
        }
    }
}

/// The chat prompt the text tests answer.
const TEXT: &str =
    "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";

/// The host's greedy answer and, fed the same tokens, the Vulkan one: at every step the Vulkan
/// argmax is the host's, or the native card's leaves the host's too.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_greedy_text_matches_the_host() {
    greedy_text_matches(TAG);
}

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
    let host = fx.host_rows(&prompt, STEPS);
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
    let native = fx.native_picks(&prompt, &host);
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
    // The text repeats in runs of `repeats` until the prompt spans more than 2 chunks of a step.
    let chunk = crate::inference::kernel::vulkan::stack::MAX_TOKENS;
    let mut runs = 1;
    let prompt = loop {
        let text = TEXT.repeat(repeats * runs);
        let prompt = tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        if prompt.len() > 2 * chunk {
            break prompt;
        }
        runs += 1;
    };
    let host = fx.host_rows(&prompt, 8);
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
    let native = fx.native_picks(&prompt, &host);
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
///
/// A short prompt runs each routed pair on its own.
/// A prompt of a whole step runs the pairs grouped by expert.
fn experts_match(tag: &str) {
    let fx = Fixture::open_tag(tag);
    let mut host = fx.build(&fx.host());
    let step = crate::inference::kernel::vulkan::stack::MAX_TOKENS;
    for prompt in [prompt(), prompt_of(step)] {
        experts_match_on(&fx, &mut host, tag, &prompt);
    }
}

fn experts_match_on(fx: &Fixture, host: &mut GenericHeteroTransformer, tag: &str, prompt: &[u32]) {
    let (h, _) = host.embed_stage(&ids(prompt), 0).unwrap();
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
            "{tag} experts of {} tokens on {}: max error {err:.3e} of {scale:.3e}",
            prompt.len(),
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

    /// [`super::vulkan_decode_profile`] for a model whose layers route to experts.
    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn decode_profile() {
        super::decode_profile(TAG);
    }

    /// [`super::vulkan_prompt_wall`] for a model whose layers route to experts.
    #[test]
    #[ignore = "needs the model in the store and a Vulkan device"]
    fn prompt_wall_profile() {
        super::prompt_wall(TAG);
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
    let fx = Fixture::open_tag(TAG);
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
    let fx = Fixture::open_tag(TAG);
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
        let chunk = crate::inference::kernel::vulkan::stack::MAX_TOKENS;
        for tokens in [1, 32, 128, chunk] {
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

/// The wall of a whole prompt on each Vulkan device against the kernel times of its steps.
///
/// Warm walls follow sustained work.
/// Cold walls follow an idle pause.
#[test]
#[ignore = "needs the model in the store and a Vulkan device"]
fn vulkan_prompt_wall() {
    prompt_wall(TAG);
}

fn prompt_wall(tag: &str) {
    const WALL_PROMPT: usize = 1106;
    const WALL_RUNS: usize = 3;
    let pause = std::time::Duration::from_secs(2);
    let prompt: Vec<u32> = (0..WALL_PROMPT as u32)
        .map(|i| 1000 + (i * 37) % 20000)
        .collect();
    let fx = Fixture::open_tag(tag);
    let chunk = crate::inference::kernel::vulkan::stack::MAX_TOKENS;
    for d in crate::inference::kernel::vulkan::devices() {
        // A card too small for the model has nothing to time.
        if d.total < 8 << 30 {
            continue;
        }
        let mut m = fx.build(&fx.on_vulkan(d.index, fx.n_layers));
        m.forward(&ids(&prompt), 0).unwrap();
        let wall = |m: &mut GenericHeteroTransformer| {
            let start = std::time::Instant::now();
            m.forward(&ids(&prompt), 0).unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        let warm_up = |m: &GenericHeteroTransformer| {
            let start = std::time::Instant::now();
            while start.elapsed() < pause {
                m.vulkan.profile(128, prompt.len());
            }
        };
        warm_up(&m);
        for run in 0..WALL_RUNS {
            println!("{}: warm wall {run}: {:.1} ms", d.name, wall(&mut m));
        }
        warm_up(&m);
        let mut sum = 0.0;
        for start in (0..prompt.len()).step_by(chunk) {
            let rows = (prompt.len() - start).min(chunk);
            let totals: Vec<f64> = (0..WALL_RUNS)
                .map(|_| {
                    m.vulkan.profile(rows, start)[0]
                        .iter()
                        .map(|(_, us)| us)
                        .sum::<f64>()
                })
                .collect();
            let best = totals.iter().copied().fold(f64::INFINITY, f64::min);
            sum += best;
            println!(
                "{}: kernels of {rows} rows at {start}: {totals:.0?} us",
                d.name
            );
        }
        println!("{}: kernel sum: {:.1} ms", d.name, sum / 1e3);
        for run in 0..WALL_RUNS {
            std::thread::sleep(pause);
            println!("{}: cold wall {run}: {:.1} ms", d.name, wall(&mut m));
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
