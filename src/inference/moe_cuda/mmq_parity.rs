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
    routing_with_hot(rng, tokens, topk, e, None)
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

    /// A block whose stacks are random blocks of a format that cannot be quantised from floats:
    /// the scale is fixed and the codes are random, which every code of these formats allows.
    #[allow(clippy::too_many_arguments)]
    fn build_from_blocks(
        dev: &Device,
        rng: &mut Lcg,
        dtype: GgmlDType,
        tokens: usize,
        topk: usize,
        e: usize,
        hidden: usize,
        inter: usize,
    ) -> Self {
        Self::build_blocks(dev, rng, dtype, tokens, topk, e, hidden, inter, false)
    }

    /// [`Self::build_from_blocks`], with the scale of every expert the routing does not pick set
    /// to NaN when `poison_unrouted`: what a transit buffer holds of an expert not sent to it.
    #[allow(clippy::too_many_arguments)]
    fn build_blocks(
        dev: &Device,
        rng: &mut Lcg,
        dtype: GgmlDType,
        tokens: usize,
        topk: usize,
        e: usize,
        hidden: usize,
        inter: usize,
        poison_unrouted: bool,
    ) -> Self {
        const SCALE_F16: [u8; 2] = [0x00, 0x24]; // 2^-6
        const SCALE_NAN: [u8; 2] = [0xFF, 0xFF];
        let early = poison_unrouted.then(|| routing(rng, tokens, topk, e));
        let routed: Vec<bool> = match &early {
            Some((_, picked, _)) => (0..e).map(|x| picked.contains(&(x as u32))).collect(),
            None => vec![true; e],
        };
        let mut stack = |rows: usize, k: usize| -> Arc<QTensor> {
            let per_expert = rows * (k / dtype.block_size());
            let blocks = e * per_expert;
            let mut data: Vec<u8> = (0..blocks * dtype.type_size())
                .map(|_| (rng.next() * 256.0) as u8)
                .collect();
            for (i, block) in data.chunks_exact_mut(dtype.type_size()).enumerate() {
                let scale = if routed[i / per_expert] {
                    SCALE_F16
                } else {
                    SCALE_NAN
                };
                block[..2].copy_from_slice(&scale);
            }
            Arc::new(QTensor::from_ggml_bytes(dtype, &data, vec![e, rows, k], dev).unwrap())
        };
        let gate = stack(inter, hidden);
        let up = stack(inter, hidden);
        let down = stack(hidden, inter);
        let xs = random(rng, tokens * hidden, 1.0);
        let (slots, experts, weights) = early.unwrap_or_else(|| routing(rng, tokens, topk, e));
        let m = slots.len();
        Self {
            gate_up: gate.clone(),
            gate,
            up,
            down,
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
        use rayon::prelude::*;
        // Each pair's product on its own, then summed into its token in the pairs' order.
        let products: Vec<(usize, Vec<f32>)> = slots
            .par_iter()
            .zip(experts.par_iter())
            .map(|(&pair, &e)| {
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
                let y = (0..hd)
                    .map(|r| {
                        let row = (e * hd + r) * it;
                        let dv: f32 = d[row..row + it].iter().zip(&h).map(|(a, b)| a * b).sum();
                        weights[pair] * dv
                    })
                    .collect();
                (t, y)
            })
            .collect();
        let mut out = vec![0f32; self.tokens * hd];
        for (t, y) in products {
            for (o, v) in out[t * hd..(t + 1) * hd].iter_mut().zip(y) {
                *o += v;
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

/// The codebook formats through the expert-batched tile kernels, judged as the others are: no
/// further from the float answer over the dequantised weights than the decode kernels are.
#[test]
fn the_expert_batched_matmul_serves_the_codebook_formats() {
    let Some(dev) = card("the codebook formats are") else {
        return;
    };
    let mut rng = Lcg(23);
    let mut failures = Vec::new();
    for dtype in [
        GgmlDType::Iq4Nl,
        GgmlDType::Iq4Xs,
        GgmlDType::Iq3Xxs,
        GgmlDType::Iq3S,
        GgmlDType::Iq2S,
        GgmlDType::Q2_0,
    ] {
        // From a handful of tokens, where an expert has a row or two, to hundreds, at the
        // launcher's own tile width and at every width it can be forced to.
        for &(tokens, topk, e) in &[
            (1usize, 2usize, 4usize),
            (5, 4, 16),
            (45, 6, 16),
            (300, 8, 32),
        ] {
            let b = Block::build_from_blocks(&dev, &mut rng, dtype, tokens, topk, e, 512, 256);
            let want = b.float();
            let decode = gap(&want, &b.decode());
            for tile in [0, 8, 16, 32, 64, 128] {
                let mmq = gap(&want, &b.mmq(false, tile));
                if mmq.is_nan() || mmq > 1.25 * decode + 0.005 {
                    failures.push(format!(
                        "{dtype:?} tokens={tokens} topk={topk} E={e} tile={tile}: {mmq} from the \
                         float answer, the decode kernels {decode}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A format whose block is shorter than a pass of the tile kernel, on a row that is not a whole
/// number of passes: the pass reads past the row's end, into experts the routing does not pick and
/// whose scales are not finite, and must add nothing.
#[test]
fn the_expert_batched_matmul_serves_rows_of_a_partial_pass() {
    let Some(dev) = card("rows of a partial pass are") else {
        return;
    };
    let mut rng = Lcg(31);
    let mut failures = Vec::new();
    let shapes = [
        (1usize, 2usize, 4usize),
        (5, 4, 16),
        (45, 6, 16),
        (300, 8, 32),
        (12, 10, 256),
        (3, 10, 256),
    ];
    for (dtype, &(tokens, topk, e)) in [GgmlDType::Q2_0, GgmlDType::Iq4Nl]
        .into_iter()
        .flat_map(|d| shapes.iter().map(move |s| (d, s)))
    {
        let b = Block::build_blocks(&dev, &mut rng, dtype, tokens, topk, e, 320, 192, true);
        let want = b.float();
        let decode = gap(&want, &b.decode());
        for tile in [0, 8, 16, 32, 64, 128] {
            let mmq = gap(&want, &b.mmq(false, tile));
            if mmq.is_nan() || mmq > 1.25 * decode + 0.005 {
                failures.push(format!(
                    "{dtype:?} tokens={tokens} topk={topk} E={e} tile={tile}: {mmq} from the \
                     float answer, the decode kernels {decode}"
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

/// The shape of a wide mixture: experts, experts per token, hidden width, expert width.
const WIDE: (usize, usize, usize, usize) = (256, 10, 2560, 640);

/// An `[e, rows, k]` stack of random `dtype` blocks with a finite scale, as raw bytes.
fn random_blocks(rng: &mut Lcg, dtype: GgmlDType, e: usize, rows: usize, k: usize) -> Vec<u8> {
    const SCALE_F16: [u8; 2] = [0x00, 0x24]; // 2^-6
    let mut data = vec![0u8; e * rows * (k / dtype.block_size()) * dtype.type_size()];
    for word in data.chunks_mut(3) {
        let bits = (rng.next() * (1u32 << 24) as f32) as u32;
        for (b, byte) in word.iter_mut().enumerate() {
            *byte = (bits >> (8 * b)) as u8;
        }
    }
    for block in data.chunks_exact_mut(dtype.type_size()) {
        block[..2].copy_from_slice(&SCALE_F16);
    }
    data
}

/// [`routing`], the first expert of every token `hot` when given.
fn routing_with_hot(
    rng: &mut Lcg,
    tokens: usize,
    topk: usize,
    e: usize,
    hot: Option<usize>,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut pairs = Vec::with_capacity(tokens * topk);
    let mut weights = Vec::with_capacity(tokens * topk);
    for t in 0..tokens {
        let mut chosen: Vec<usize> = hot.into_iter().collect();
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

/// The routed rows of one wide-mixture layer chunk on the card.
struct WideChunk {
    x: Tensor,
    slots: Tensor,
    experts: Tensor,
    weights: Tensor,
}

impl WideChunk {
    fn new(dev: &Device, rng: &mut Lcg, tokens: usize, hot: Option<usize>) -> Self {
        let (e, topk, hidden, _) = WIDE;
        let (slots, experts, weights) = routing_with_hot(rng, tokens, topk, e, hot);
        let m = slots.len();
        Self {
            x: Tensor::from_vec(random(rng, tokens * hidden, 1.0), (tokens, hidden), dev).unwrap(),
            slots: Tensor::from_vec(slots, m, dev).unwrap(),
            experts: Tensor::from_vec(experts, m, dev).unwrap(),
            weights: Tensor::from_vec(weights, m, dev).unwrap(),
        }
    }

    fn plan(&self) -> MmqRouting {
        MmqRouting::new(&self.slots, &self.experts, WIDE.1, WIDE.0).unwrap()
    }
}

/// The gate and up products launched together, over a grid sized for the expected longest run,
/// against the first launches (a width for the mean run, a grid for every token on one expert,
/// gate and up apart), a fused gate||up stack and other tile widths: every product of a column
/// sums the same k blocks in the same order, so all of them agree bit for bit. One expert takes
/// every token, a run far longer than the grid is sized for.
#[test]
fn the_grouped_launches_match_the_first_launches_bit_for_bit() {
    let Some(dev) = card("the grouped launches are") else {
        return;
    };
    let (e, _, hidden, inter) = WIDE;
    let mut rng = Lcg(43);
    let tokens = 1000;
    let chunk = WideChunk::new(&dev, &mut rng, tokens, Some(7));
    let down = random_blocks(&mut rng, GgmlDType::Iq4Nl, e, hidden, inter);
    let down =
        QTensor::from_ggml_bytes(GgmlDType::Iq4Nl, &down, vec![e, hidden, inter], &dev).unwrap();
    for dtype in [GgmlDType::Iq2S, GgmlDType::Iq3Xxs] {
        let gate = random_blocks(&mut rng, dtype, e, inter, hidden);
        let up = random_blocks(&mut rng, dtype, e, inter, hidden);
        let per_expert = gate.len() / e;
        let mut gate_up = Vec::with_capacity(2 * gate.len());
        for x in 0..e {
            gate_up.extend_from_slice(&gate[x * per_expert..(x + 1) * per_expert]);
            gate_up.extend_from_slice(&up[x * per_expert..(x + 1) * per_expert]);
        }
        let stack = |bytes: &[u8], rows: usize| {
            QTensor::from_ggml_bytes(dtype, bytes, vec![e, rows, hidden], &dev).unwrap()
        };
        let (gate, up, gate_up) = (
            stack(&gate, inter),
            stack(&up, inter),
            stack(&gate_up, 2 * inter),
        );
        let run = |plan: MmqRouting, fused: bool| -> (Vec<f32>, Vec<f32>) {
            let h = match fused {
                true => plan.gate_up_silu(&chunk.x, &gate_up, None).unwrap(),
                false => plan.gate_up_silu(&chunk.x, &gate, Some(&up)).unwrap(),
            };
            let out = plan.down_fold(&h, &down, &chunk.weights, None).unwrap();
            (to_vec(&h), to_vec(&out))
        };
        let want = run(chunk.plan().with_first_launches(), false);
        assert!(want.1.iter().all(|v| v.is_finite()) && want.1.iter().any(|&v| v != 0.0));
        let cases = [
            ("grouped", run(chunk.plan(), false)),
            ("fused stack", run(chunk.plan(), true)),
            ("width 8", run(chunk.plan().with_tile_width(8), false)),
            ("width 128", run(chunk.plan().with_tile_width(128), false)),
        ];
        for (what, got) in cases {
            assert!(got.0 == want.0, "{dtype:?} {what}: gate and up differ");
            assert!(
                got.1 == want.1,
                "{dtype:?} {what}: the folded result differs"
            );
        }
    }
}

/// Kernel times of one wide-mixture layer chunk (256 experts, ten per token, 2560 hidden, 640
/// intermediate; gate and up in a low-bit format, down in IQ4_NL), the first launches against
/// the grouped ones, with the rate of the int8 work. Run with `--ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check"]
fn wide_mixture_layer_chunk_timings() {
    let Some(dev) = card("the timings are") else {
        return;
    };
    let cuda = dev.as_cuda_device().unwrap();
    let (e, topk, hidden, inter) = WIDE;
    let mut rng = Lcg(5);
    let time = |f: &dyn Fn()| -> f64 {
        f();
        cuda.synchronize().unwrap();
        let reps = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        cuda.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    let stack = |dtype: GgmlDType, rows: usize, k: usize, rng: &mut Lcg| -> QTensor {
        let data = random_blocks(rng, dtype, e, rows, k);
        QTensor::from_ggml_bytes(dtype, &data, vec![e, rows, k], &dev).unwrap()
    };
    let down = stack(GgmlDType::Iq4Nl, hidden, inter, &mut rng);
    for dtype in [GgmlDType::Iq2S, GgmlDType::Iq3Xxs] {
        let gate = stack(dtype, inter, hidden, &mut rng);
        let up = stack(dtype, inter, hidden, &mut rng);
        for tokens in [1000usize, 4000] {
            let chunk = WideChunk::new(&dev, &mut rng, tokens, None);
            let flops = (3 * tokens * topk * hidden * inter * 2) as f64;
            for (what, first) in [("first", true), ("grouped", false)] {
                let plan = || match first {
                    true => chunk.plan().with_first_launches(),
                    false => chunk.plan(),
                };
                let p = plan();
                let h = p.gate_up_silu(&chunk.x, &gate, Some(&up)).unwrap();
                let gu = time(&|| {
                    p.gate_up_silu(&chunk.x, &gate, Some(&up)).unwrap();
                });
                let dn = time(&|| {
                    p.down_fold(&h, &down, &chunk.weights, None).unwrap();
                });
                let all = time(&|| {
                    let p = plan();
                    let h = p.gate_up_silu(&chunk.x, &gate, Some(&up)).unwrap();
                    p.down_fold(&h, &down, &chunk.weights, None).unwrap();
                });
                println!(
                    "{dtype:?}+Iq4Nl rows={tokens:>5} {what:>7}: gate_up {gu:7.3} ms down {dn:7.3} ms \
                     chunk {all:7.3} ms ({:.1} TOPS)",
                    flops / (all * 1e-3) / 1e12
                );
            }
        }
    }
}
