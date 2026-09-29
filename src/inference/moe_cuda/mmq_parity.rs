//! The expert-batched tiled matmul, judged on random weights for every format it serves.
//!
//! Three answers to the same SiLU expert block: the float one over the dequantised weights,
//! the one the decode kernels give, and the one the expert-batched tile kernels give. Both
//! kernel families quantise the activation to q8_1 in groups of 32, which leaves each about
//! two hundredths of the largest output from the float answer on random weights; the tile
//! kernels must be no further from it than the decode kernels are.

use super::mmq_experts::{Glu, MmqRouting};
use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{Device, Tensor};
use std::sync::Arc;

const FORMATS: [GgmlDType; 6] = [
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8_0,
    GgmlDType::Q5_0,
    GgmlDType::MxFp4,
];

fn card(what: &str) -> Option<Device> {
    match crate::tensor::cuda::CudaDevice::new(0) {
        Ok(d) => Some(Device::Cuda(d)),
        Err(_) => {
            eprintln!("no CUDA device; {what} NOT covered by this run");
            None
        }
    }
}

/// A deterministic stream of values in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() * n as f32) as usize).min(n - 1)
    }
}

/// `topk` distinct experts per token with their router weights, and the stable sort by
/// expert: `(slots, experts, weights)`, the weights in (token, choice) order.
fn routing(rng: &mut Lcg, tokens: usize, topk: usize, e: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut pairs = Vec::with_capacity(tokens * topk);
    let mut weights = Vec::with_capacity(tokens * topk);
    for t in 0..tokens {
        let mut chosen: Vec<usize> = Vec::with_capacity(topk);
        while chosen.len() < topk {
            let c = rng.below(e);
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        for (s, &c) in chosen.iter().enumerate() {
            pairs.push((c as u32, (t * topk + s) as u32));
            weights.push(0.05 + rng.next() * 0.3);
        }
    }
    pairs.sort_by_key(|p| p.0);
    (
        pairs.iter().map(|p| p.1).collect(),
        pairs.iter().map(|p| p.0).collect(),
        weights,
    )
}

fn random(rng: &mut Lcg, n: usize, spread: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.next() - 0.5) * 2.0 * spread).collect()
}

/// The largest difference relative to the largest magnitude of `want`.
fn gap(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    let scale = want.iter().map(|x| x.abs()).fold(1e-6f32, f32::max);
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs() / scale)
        .fold(0.0f32, f32::max)
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

struct Block {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    gate_up: Arc<QTensor>,
    down: Arc<QTensor>,
    x: Tensor,
    slots: Tensor,
    experts: Tensor,
    /// `[tokens, topk]`: the router's choices the sorted routing was built from.
    choices: Tensor,
    weights: Tensor,
    host: (Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>),
    tokens: usize,
    topk: usize,
    e: usize,
    hidden: usize,
    inter: usize,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn build(
        dev: &Device,
        rng: &mut Lcg,
        dtype: GgmlDType,
        tokens: usize,
        topk: usize,
        e: usize,
        hidden: usize,
        inter: usize,
    ) -> Self {
        let host = Device::Cpu;
        let gw = random(rng, e * inter * hidden, 0.08);
        let uw = random(rng, e * inter * hidden, 0.08);
        let dw = random(rng, e * hidden * inter, 0.08);
        let mut guw = Vec::with_capacity(2 * gw.len());
        for x in 0..e {
            let span = x * inter * hidden..(x + 1) * inter * hidden;
            guw.extend_from_slice(&gw[span.clone()]);
            guw.extend_from_slice(&uw[span]);
        }
        let q = |w: Vec<f32>, shape: (usize, usize, usize)| {
            let t = Tensor::from_vec(w, shape, &host).unwrap();
            Arc::new(QTensor::quantize_onto(&t, dtype, dev).unwrap())
        };
        let xs = random(rng, tokens * hidden, 1.0);
        let (slots, experts, weights) = routing(rng, tokens, topk, e);
        let m = slots.len();
        Self {
            gate: q(gw, (e, inter, hidden)),
            up: q(uw, (e, inter, hidden)),
            gate_up: q(guw, (e, 2 * inter, hidden)),
            down: q(dw, (e, hidden, inter)),
            x: Tensor::from_vec(xs.clone(), (tokens, hidden), dev).unwrap(),
            slots: Tensor::from_vec(slots.clone(), m, dev).unwrap(),
            experts: Tensor::from_vec(experts.clone(), m, dev).unwrap(),
            choices: {
                let mut c = vec![0u32; m];
                for (&pair, &e) in slots.iter().zip(&experts) {
                    c[pair as usize] = e;
                }
                Tensor::from_vec(c, (tokens, topk), dev).unwrap()
            },
            weights: Tensor::from_vec(weights.clone(), m, dev).unwrap(),
            host: (xs, slots, experts, weights),
            tokens,
            topk,
            e,
            hidden,
            inter,
        }
    }

    fn mmq(&self, fused: bool, tile: i32) -> Vec<f32> {
        let plan = MmqRouting::new(&self.slots, &self.experts, self.topk, self.e)
            .unwrap()
            .with_tile_width(tile);
        self.mmq_over(&plan, fused)
    }

    /// The same, grouping the rows on the device from the router's choices.
    fn mmq_grouped_on_card(&self) -> Vec<f32> {
        let plan = MmqRouting::from_choices(&self.choices, self.e).unwrap();
        self.mmq_over(&plan, false)
    }

    fn mmq_over(&self, plan: &MmqRouting, fused: bool) -> Vec<f32> {
        let h = if fused {
            plan.gate_up_silu(&self.x, &self.gate_up, None).unwrap()
        } else {
            plan.gate_up_silu(&self.x, &self.gate, Some(&self.up))
                .unwrap()
        };
        to_vec(&plan.down_fold(&h, &self.down, &self.weights, None).unwrap())
    }

    fn decode(&self) -> Vec<f32> {
        let h = super::gemm::moe_gemm_gguf_gate_up_silu_mul(
            &self.x,
            &self.gate,
            &self.up,
            &self.slots,
            &self.experts,
            self.topk,
        )
        .unwrap();
        to_vec(
            &super::gemm::moe_gemm_gguf_down_reduce(
                &h,
                &self.down,
                &self.slots,
                &self.experts,
                &self.weights,
                self.topk,
                self.tokens,
                None,
                None,
                None,
            )
            .unwrap(),
        )
    }

    /// The float answer over the dequantised weights.
    fn float(&self) -> Vec<f32> {
        let host = Device::Cpu;
        let deq = |w: &QTensor| to_vec(&w.dequantize(&host).unwrap());
        let (g, u, d) = (deq(&self.gate), deq(&self.up), deq(&self.down));
        let (xs, slots, experts, weights) = &self.host;
        let (hd, it) = (self.hidden, self.inter);
        let mut out = vec![0f32; self.tokens * hd];
        for (&pair, &e) in slots.iter().zip(experts) {
            let (pair, e) = (pair as usize, e as usize);
            let t = pair / self.topk;
            let x = &xs[t * hd..(t + 1) * hd];
            let h: Vec<f32> = (0..it)
                .map(|r| {
                    let row = (e * it + r) * hd;
                    let gv: f32 = g[row..row + hd].iter().zip(x).map(|(a, b)| a * b).sum();
                    let uv: f32 = u[row..row + hd].iter().zip(x).map(|(a, b)| a * b).sum();
                    gv / (1.0 + (-gv).exp()) * uv
                })
                .collect();
            for r in 0..hd {
                let row = (e * hd + r) * it;
                let dv: f32 = d[row..row + it].iter().zip(&h).map(|(a, b)| a * b).sum();
                out[t * hd + r] += weights[pair] * dv;
            }
        }
        out
    }
}

#[test]
fn the_expert_batched_matmul_agrees_with_the_decode_kernels_and_the_float_answer() {
    let Some(dev) = card("the expert-batched matmul is") else {
        return;
    };
    let mut rng = Lcg(7);
    let mut failures = Vec::new();
    for dtype in FORMATS {
        for &(tokens, topk, e) in &[
            (1usize, 2usize, 4usize),
            (7, 4, 8),
            (45, 6, 16),
            (300, 8, 32),
        ] {
            let b = Block::build(&dev, &mut rng, dtype, tokens, topk, e, 512, 256);
            let want = b.float();
            let decode = gap(&want, &b.decode());
            for fused in [false, true] {
                let mmq = gap(&want, &b.mmq(fused, 0));
                if mmq > 1.25 * decode + 0.005 {
                    failures.push(format!(
                        "{dtype:?} tokens={tokens} topk={topk} E={e} fused={fused}: \
                         {mmq} from the float answer, the decode kernels {decode}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every tile width the launcher can pick gives the same answer, and the same one twice.
#[test]
fn the_expert_batched_matmul_is_the_same_at_every_tile_width_and_on_every_run() {
    let Some(dev) = card("the expert-batched tile widths are") else {
        return;
    };
    let mut rng = Lcg(11);
    for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        let b = Block::build(&dev, &mut rng, dtype, 700, 8, 16, 512, 256);
        let reference = b.mmq(false, 0);
        // The card's grouping leaves the rows of an expert in any order; a row's result does
        // not depend on where in its expert's run it sits.
        assert_eq!(
            reference,
            b.mmq_grouped_on_card(),
            "{dtype:?}: grouped on the card differs"
        );
        assert_eq!(reference, b.mmq(false, 0), "{dtype:?}: two runs differ");
        // Tensor-core tiles from 48 columns up come in steps of sixteen.
        for tile in (8..=128).step_by(8).filter(|w| *w < 48 || w % 16 == 0) {
            assert_eq!(
                reference,
                b.mmq(false, tile),
                "{dtype:?}: tile {tile} differs"
            );
        }
    }
}

/// gpt-oss's block: the clamped SwiGLU with per-expert gate and up biases, and a per-expert
/// down bias, against the decode kernels that fold the same three into their epilogues.
#[test]
fn the_expert_batched_swiglu_oai_block_agrees_with_the_decode_kernels() {
    let Some(dev) = card("the expert-batched gpt-oss block is") else {
        return;
    };
    let (alpha, limit) = (1.702f32, 7.0f32);
    let mut rng = Lcg(13);
    let mut failures = Vec::new();
    for dtype in [GgmlDType::MxFp4, GgmlDType::Q8_0] {
        // A k that is not a whole number of tile passes, as gpt-oss's is.
        for &(tokens, topk, e, hidden, inter) in &[
            (9usize, 4usize, 8usize, 576usize, 320usize),
            (200, 4, 32, 576, 320),
        ] {
            let b = Block::build(&dev, &mut rng, dtype, tokens, topk, e, hidden, inter);
            let mut t = |n: usize| Tensor::from_vec(random(&mut rng, n, 0.5), n, &dev).unwrap();
            let gb = t(e * inter).reshape((e, inter)).unwrap();
            let ub = t(e * inter).reshape((e, inter)).unwrap();
            let db = t(e * hidden).reshape((e, hidden)).unwrap();
            let h = super::gemm::moe_gemm_gguf_gate_up_swiglu_oai(
                &b.x,
                &b.gate,
                &b.up,
                Some(&gb),
                Some(&ub),
                &b.slots,
                &b.experts,
                topk,
                f64::from(alpha),
                f64::from(limit),
            )
            .unwrap();
            let decode = to_vec(
                &super::gemm::moe_gemm_gguf_down_reduce(
                    &h,
                    &b.down,
                    &b.slots,
                    &b.experts,
                    &b.weights,
                    topk,
                    tokens,
                    None,
                    Some(&db),
                    None,
                )
                .unwrap(),
            );
            let plan = MmqRouting::from_choices(&b.choices, e).unwrap();
            let glu = Glu::SwigluOai {
                gate_bias: Some(&gb),
                up_bias: Some(&ub),
                alpha,
                limit,
            };
            let hs = plan.gate_up(&b.x, &b.gate, Some(&b.up), glu).unwrap();
            let got = to_vec(
                &plan
                    .down_fold_biased(&hs, &b.down, &b.weights, None, Some(&db))
                    .unwrap(),
            );
            let g = gap(&decode, &got);
            if g >= 0.02 {
                failures.push(format!(
                    "{dtype:?} tokens={tokens} E={e}: {g} from the decode kernels"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Kernel times on a qwen3-coder shaped layer (128 experts, 8 per token, 2048 hidden, 768
/// intermediate), the expert-batched path against the decode kernels, per format and tile
/// width. Run with `--ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check"]
fn expert_batched_matmul_timings() {
    let Some(dev) = card("the timings are") else {
        return;
    };
    let cuda = dev.as_cuda_device().unwrap();
    let (e, topk, hidden, inter) = (
        std::env::var("MOE_E")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128usize),
        std::env::var("MOE_TOPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8usize),
        std::env::var("MOE_HIDDEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048usize),
        std::env::var("MOE_INTER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(768usize),
    );
    let token_counts: Vec<usize> = std::env::var("MOE_TOKENS")
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|_| vec![4, 8, 16, 32, 64, 128, 512, 2048, 4096]);
    let tiles: Vec<i32> = std::env::var("MOE_TILES")
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|_| vec![0]);
    let mut rng = Lcg(3);
    let time = |f: &dyn Fn()| -> f64 {
        f();
        cuda.synchronize().unwrap();
        let reps = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        cuda.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::Q5K,
        GgmlDType::Q8_0,
    ] {
        for &tokens in &token_counts {
            let b = Block::build(&dev, &mut rng, dtype, tokens, topk, e, hidden, inter);
            let h_dec = super::gemm::moe_gemm_gguf_gate_up_silu_mul(
                &b.x, &b.gate, &b.up, &b.slots, &b.experts, topk,
            )
            .unwrap();
            let dec_gu = time(&|| {
                super::gemm::moe_gemm_gguf_gate_up_silu_mul(
                    &b.x, &b.gate, &b.up, &b.slots, &b.experts, topk,
                )
                .unwrap();
            });
            let dec_dn = time(&|| {
                super::gemm::moe_gemm_gguf_down_reduce(
                    &h_dec, &b.down, &b.slots, &b.experts, &b.weights, topk, tokens, None, None,
                    None,
                )
                .unwrap();
            });
            for &tile in &tiles {
                let plan = MmqRouting::new(&b.slots, &b.experts, topk, e)
                    .unwrap()
                    .with_tile_width(tile);
                let h = plan.gate_up_silu(&b.x, &b.gate, Some(&b.up)).unwrap();
                let mmq_gu = time(&|| {
                    plan.gate_up_silu(&b.x, &b.gate, Some(&b.up)).unwrap();
                });
                let mmq_dn = time(&|| {
                    plan.down_fold(&h, &b.down, &b.weights, None).unwrap();
                });
                let mmq_plan = time(&|| {
                    MmqRouting::new(&b.slots, &b.experts, topk, e).unwrap();
                });
                println!(
                    "{dtype:?} tokens={tokens:>5} tile={tile:>3}: decode gate_up {dec_gu:8.3} ms \
                     down {dec_dn:8.3} ms | mmq gate_up {mmq_gu:8.3} ms down {mmq_dn:8.3} ms \
                     plan {mmq_plan:6.3} ms"
                );
            }
        }
    }
}
