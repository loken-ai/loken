//! The MoE prefill kernels judged against a reduction the host does its own way.
//!
//! `prefill_parity` next door sets the two expert GEMMs against each other, which is the right
//! instrument for a dequantiser - the weight error cancels - and the wrong one for everything
//! else: two device paths that agree on where a row lives agree wrongly. So the reference here
//! is built the other way round. The Q4_K stack is dequantised on the host, the activation is
//! kept in f32, and the reduction is done in f64 by the host's own loop. Nothing the kernel
//! reads decides what the reference computes.
//!
//! WHAT THE ROW INDEX MEANS. A routed pair has two numbers: where the sort put it, and the
//! `(token, choice)` slot it came from. The down projection's input holds one row per pair
//! indexed by SLOT - that is where `loken_moe_gemm_gguf_gate_up_silu_mul` writes it
//! (`output_row(all_outputs, routed.token, ...)`), where the IMMA gate‖up writes it, and where
//! the dp4a down projection reads it (`activation_row(all_inputs, routed.token, ...)`). Three
//! call sites, one convention; the reference below follows it, so a kernel that reads a row by
//! its sorted position instead is measured against what it was supposed to read.
//!
//! WHY THE TOLERANCE IS ALMOST NOTHING. The kernels quantise their activation to q8_1 before
//! multiplying, and that step normally costs a percent. It costs nothing here because the
//! activation is built to survive it exactly: every value is `n . 2⁻⁶` with `|n| <= 127` and one
//! `|n| = 127` in each block of thirty-two, so the quantiser's `d = amax/127` is exactly `2⁻⁶`,
//! `round(x/d)` is exactly `n`, and both halves of the block's `ds` pair - the scale and the
//! block sum - are exact in f16. The card therefore multiplies the very numbers the host does,
//! its integer products are exact in int32 (`|Σ q₄.q₈| <= 15.127.32`, well inside 2²⁴), and the
//! only gap left is the order the f32 additions happen in. That is what the bound below sizes:
//! at most one half-ulp per weight value plus the warp reduction and the top-k scatter, against
//! the magnitudes actually summed. It comes out around 10⁻⁵ of the row - sharp enough that a
//! misplaced row, which is a different vector entirely, cannot hide under it.

use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{Device, Tensor};

/// Half an f32 ulp, relative: the most one rounding can move a sum.
const HALF_ULP_F32: f64 = 5.960_464_477_539_063e-8;

/// The most `__expf` is documented to miss by, relative - two ulp of f32.
const EXPF_ULP: f64 = 2.0 * 2.0 * HALF_ULP_F32;

/// Values per q8_1 block, and per Q4_K sub-block: the same thirty-two either way.
const BLOCK: usize = 32;

/// How many times a launch is repeated when the question is whether it answers the same twice.
const REPEATS: usize = 6;

/// A reproducible bit source. The cases have to be identical from run to run - a judge whose
/// input moves cannot tell a racy kernel from a fresh case.
struct Bits(u64);

impl Bits {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
    fn upto(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A value in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.upto(1 << 20) as f32) / 524_288.0 - 1.0
    }
}

/// `rows x k` of activation that the device quantiser reproduces exactly.
///
/// See the module header for why that is possible and what it buys. The invariant each block of
/// thirty-two carries: every value is an integer multiple of `2⁻⁶`, the largest magnitude is
/// exactly `127 . 2⁻⁶`, and the block's sum is a multiple of `2⁻⁶` below 2¹¹ of them - which is
/// the range f16 holds without rounding.
fn exact_q8_1_rows(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    assert_eq!(
        k % BLOCK,
        0,
        "an activation row is a whole number of blocks"
    );
    let mut bits = Bits::new(seed);
    let mut out = vec![0.0f32; rows * k];
    for row in 0..rows {
        for block in 0..k / BLOCK {
            let base = row * k + block * BLOCK;
            // Everything but the peak stays inside ±50, so thirty-two of them sum to less than
            // 2¹¹ steps and the block sum stays exact in f16.
            let mut codes = [0i32; BLOCK];
            for c in codes.iter_mut() {
                *c = bits.upto(101) as i32 - 50;
            }
            let peak = bits.upto(BLOCK as u64) as usize;
            codes[peak] = if bits.upto(2) == 0 { 127 } else { -127 };
            for (i, c) in codes.iter().enumerate() {
                out[base + i] = (*c as f32) / 64.0;
            }
        }
    }
    out
}

/// An expert stack of dense weights, before quantisation.
///
/// The per-row scale varies so that the superblocks do not all land on the same Q4_K footing  -
/// a stack of uniform magnitude would exercise one scale and call it the format.
fn expert_stack(experts: usize, rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut bits = Bits::new(seed);
    let mut out = vec![0.0f32; experts * rows * k];
    for r in 0..experts * rows {
        let scale = 0.05 + 0.45 * ((r % 7) as f32) / 6.0;
        for i in 0..k {
            out[r * k + i] = bits.unit() * scale;
        }
    }
    out
}

/// A routing whose expert runs are `n_tokens` long, sorted by expert as every expert kernel
/// requires, with `topk` experts and each token visiting all of them once.
///
/// Long runs are deliberate. The IMMA kernels carry a four-entry expert list per eight-pair
/// tile and drop anything past it, so a routing with short runs would put a second defect in
/// the same measurement; [`crowded_runs`] asks that question on its own.
///
/// Returns the slot of each sorted pair and the expert it names. The slot list is a
/// permutation of `0..n_tokens.topk` and is the identity only when `topk == 1`, which is what
/// makes the difference between "sorted position" and "slot" visible at all.
fn long_runs(n_tokens: usize, topk: usize) -> (Vec<u32>, Vec<u32>, usize) {
    let mut slots = Vec::with_capacity(n_tokens * topk);
    let mut experts = Vec::with_capacity(n_tokens * topk);
    for e in 0..topk {
        for t in 0..n_tokens {
            slots.push((t * topk + e) as u32);
            experts.push(e as u32);
        }
    }
    (slots, experts, topk)
}

/// The same pair count laid out so that each expert holds `run` of them, over slots that are
/// their own sorted positions.
///
/// The identity slot list is the point. It is the one routing under which "sorted position" and
/// "slot" name the same row, so a kernel that confused the two still answers correctly here and
/// the only thing left for this case to measure is the four-entry expert list an IMMA tile
/// carries - one variable at a time.
fn crowded_runs(n_tokens: usize, topk: usize, run: usize) -> (Vec<u32>, Vec<u32>, usize) {
    let size_m = n_tokens * topk;
    let slots: Vec<u32> = (0..size_m as u32).collect();
    let experts: Vec<u32> = (0..size_m).map(|i| (i / run) as u32).collect();
    let n_experts = experts.last().map_or(0, |e| *e as usize + 1);
    (slots, experts, n_experts)
}

/// The largest offset a Q4_K sub-block can carry, read off its dequantised values.
///
/// A sub-block dequantises to `A.q - B` with `q` a four-bit code, so `A <= hi - lo` whenever two
/// codes differ and `B = A.q_min - lo`, giving `|B| <= 15.(hi - lo) + max(|hi|, |lo|)`. The
/// kernel forms `A.Σq.x` and `B.Σx` as separate f32 terms, so this is what the rounding bound
/// needs and the only thing it is used for - a loose answer costs a part in ten million.
fn offset_ceiling(sub: &[f32]) -> f64 {
    let hi = sub.iter().fold(f32::MIN, |a, b| a.max(*b)) as f64;
    let lo = sub.iter().fold(f32::MAX, |a, b| a.min(*b)) as f64;
    15.0 * (hi - lo) + hi.abs().max(lo.abs())
}

/// One dequantised weight row, and what it takes to judge a dot product against it.
struct WeightRow {
    values: Vec<f32>,
    /// [`offset_ceiling`] per sub-block, held so the bound does not rebuild it per pair.
    offsets: Vec<f64>,
}

impl WeightRow {
    fn split(dequantised: &[f32], rows: usize, k: usize) -> Vec<Self> {
        (0..rows)
            .map(|r| {
                let values = dequantised[r * k..(r + 1) * k].to_vec();
                let offsets = values.chunks(BLOCK).map(offset_ceiling).collect();
                Self { values, offsets }
            })
            .collect()
    }

    /// The exact dot product against `x`, and the most the card's f32 order can move it.
    ///
    /// The slack is `½ulp x (one rounding per weight value, plus the warp reduction) x` the
    /// magnitude the kernel actually sums - which is the two Q4_K sub-terms, not their
    /// difference, since cancellation between them is real and does not shrink the error.
    fn dot_and_slack(&self, x: &[f32], block_sums: &[f32]) -> (f64, f64) {
        let mut dot = 0.0f64;
        let mut mass = 0.0f64;
        for (w, xi) in self.values.iter().zip(x) {
            let p = (*w as f64) * (*xi as f64);
            dot += p;
            mass += p.abs();
        }
        for (b, sum) in self.offsets.iter().zip(block_sums) {
            mass += 2.0 * b * (*sum as f64).abs();
        }
        (dot, HALF_ULP_F32 * ((self.values.len() + 8) as f64) * mass)
    }
}

/// `|Σ x|` over each block of thirty-two, for every row: the factor the Q4_K offset term rides.
fn block_sums(x: &[f32], rows: usize, k: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|r| {
            x[r * k..(r + 1) * k]
                .chunks(BLOCK)
                .map(|c| c.iter().sum::<f32>())
                .collect()
        })
        .collect()
}

/// A CUDA device, or the reason this run judges nothing.
///
/// The IMMA kernels are built on `m16n8k32`, which arrived with Ampere; below that the family
/// ships no code and a launch would only report the card, so the skip says so out loud.
fn ampere_card() -> Option<Device> {
    let dev = crate::tensor::cuda::CudaDevice::new(0).ok()?;
    if !dev.has_ampere_tensor_cores() {
        eprintln!("no m16n8k32 on this card; the IMMA MoE kernels are NOT covered by this run");
        return None;
    }
    Some(Device::Cuda(dev))
}

/// Quantise a dense stack to Q4_K on the card, and hand back what the host reads it as.
fn q4k_pair(dev: &Device, dense: &[f32], dims: (usize, usize, usize)) -> (QTensor, Vec<f32>) {
    let t = Tensor::from_vec(dense.to_vec(), dims, &Device::Cpu).expect("stack tensor");
    let q = QTensor::quantize(&t, GgmlDType::Q4K)
        .expect("quantise to Q4_K")
        .to_device(dev)
        .expect("upload the expert stack");
    let deq = q
        .dequantize(&Device::Cpu)
        .expect("dequantise on the host")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("read back");
    (q, deq)
}

/// The worst breach of a per-element bound, and where it sits.
///
/// A judge that only reported "they differ" would leave a misplaced row and a drifting sum
/// looking alike, so this says how many elements broke their bound and by what multiple of it.
fn breaches(got: &[f32], want: &[f64], bound: &[f64], width: usize) -> (usize, f64, String) {
    let mut count = 0usize;
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for i in 0..want.len() {
        let gap = (got[i] as f64 - want[i]).abs();
        let allowed = bound[i].max(f64::MIN_POSITIVE);
        if gap > allowed {
            count += 1;
            if gap / allowed > worst {
                worst = gap / allowed;
                worst_at = i;
            }
        }
    }
    if count == 0 {
        return (0, 0.0, String::new());
    }
    let scale = want.iter().fold(1e-12f64, |a, b| a.max(b.abs()));
    let rel = (got[worst_at] as f64 - want[worst_at]).abs() / scale;
    (
        count,
        worst,
        format!(
            "{count} of {} elements are outside their bound; the worst is row {}, column {}, \
             {:.6} against {:.6} - {rel:.4} of the largest element and {worst:.0}x what f32 \
             ordering allows",
            want.len(),
            worst_at / width,
            worst_at % width,
            got[worst_at],
            want[worst_at],
        ),
    )
}

/// Whether several runs of one launch wrote the same bits, and if not, how far apart they are.
///
/// The distinction is the point. A kernel that reassociates a float sum differs in the last
/// bits and stays inside what f32 ordering allows; a kernel with a race differs by whole
/// values. So this returns the count of elements that moved at all, the widest move relative to
/// the largest element, and the move of every element - the last so the caller can hold each
/// one against its own bound rather than against a single number for the whole tensor.
fn run_gaps(runs: &[Vec<f32>]) -> (usize, f64, Vec<f64>) {
    let first = &runs[0];
    let scale = first.iter().fold(1e-12f32, |a, b| a.max(b.abs())) as f64;
    let mut gaps = vec![0.0f64; first.len()];
    let mut differing = 0usize;
    let mut worst = 0.0f64;
    for i in 0..first.len() {
        for run in &runs[1..] {
            if run[i].to_bits() != first[i].to_bits() {
                gaps[i] = gaps[i].max((run[i] as f64 - first[i] as f64).abs());
            }
        }
        if gaps[i] > 0.0 {
            differing += 1;
            worst = worst.max(gaps[i]);
        }
    }
    (differing, worst / scale, gaps)
}

// ------------------------------------------------------------
// The down projection and its top-k reduction
// ------------------------------------------------------------

/// One down-projection case: the tensors on the card, and everything the host needs to say what
/// the answer is without asking the card anything.
struct DownCase {
    weights: QTensor,
    input: Tensor,
    slots_dev: Tensor,
    experts_dev: Tensor,
    tw_dev: Tensor,
    slots: Vec<u32>,
    experts: Vec<u32>,
    tw: Vec<f32>,
    x: Vec<f32>,
    rows: Vec<WeightRow>,
    sums: Vec<Vec<f32>>,
    n_tokens: usize,
    topk: usize,
    hidden: usize,
    k: usize,
}

impl DownCase {
    fn build(
        dev: &Device,
        n_tokens: usize,
        topk: usize,
        hidden: usize,
        k: usize,
        routing: (Vec<u32>, Vec<u32>, usize),
        seed: u64,
    ) -> Self {
        let (slots, experts, n_experts) = routing;
        let size_m = n_tokens * topk;
        assert_eq!(slots.len(), size_m);

        let dense = expert_stack(n_experts, hidden, k, seed);
        let (weights, deq) = q4k_pair(dev, &dense, (n_experts, hidden, k));

        // One activation row per routed pair, laid out by SLOT - the convention the module
        // header names, and the one every producer of these rows writes.
        let x = exact_q8_1_rows(size_m, k, seed ^ 0x5AD1);
        let tw: Vec<f32> = (0..size_m)
            .map(|i| 0.05 + ((i % 13) as f32) * 0.061)
            .collect();

        Self {
            input: Tensor::from_vec(x.clone(), (size_m, k), dev).expect("upload activation"),
            slots_dev: Tensor::from_vec(slots.clone(), size_m, dev).expect("upload slots"),
            experts_dev: Tensor::from_vec(experts.clone(), size_m, dev).expect("upload experts"),
            tw_dev: Tensor::from_vec(tw.clone(), size_m, dev).expect("upload routing weights"),
            rows: WeightRow::split(&deq, n_experts * hidden, k),
            sums: block_sums(&x, size_m, k),
            weights,
            slots,
            experts,
            tw,
            x,
            n_tokens,
            topk,
            hidden,
            k,
        }
    }

    /// What the host says the card should have written, and the room f32 ordering leaves.
    fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let mut want = vec![0.0f64; self.n_tokens * self.hidden];
        let mut bound = vec![0.0f64; self.n_tokens * self.hidden];
        for pair in 0..self.slots.len() {
            let slot = self.slots[pair] as usize;
            let expert = self.experts[pair] as usize;
            let token = slot / self.topk;
            let scale = self.tw[slot] as f64;
            let xrow = &self.x[slot * self.k..(slot + 1) * self.k];
            let sums = &self.sums[slot];
            for col in 0..self.hidden {
                let (dot, slack) = self.rows[expert * self.hidden + col].dot_and_slack(xrow, sums);
                let at = token * self.hidden + col;
                want[at] += scale * dot;
                // The kernel folds each pair in with an atomicAdd, so the scatter costs one
                // more rounding per contributor on top of the dot product's own.
                bound[at] +=
                    scale.abs() * (slack + HALF_ULP_F32 * ((self.topk + 1) as f64) * dot.abs());
            }
        }
        (want, bound)
    }

    fn launch(&self) -> Vec<f32> {
        super::moe_q4k_imma_m8_down_reduce(
            &self.input,
            &self.weights,
            &self.slots_dev,
            &self.experts_dev,
            &self.tw_dev,
            self.topk,
            self.n_tokens,
            None,
        )
        .expect("the IMMA down projection refused the case")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("read back")
    }

    fn label(&self) -> String {
        format!(
            "size_m={} (tokens={} topk={}) hidden={} K={}",
            self.n_tokens * self.topk,
            self.n_tokens,
            self.topk,
            self.hidden,
            self.k
        )
    }
}

/// The shapes the down projection is judged at: either side of the sixty-four the dispatcher
/// switches on, the tile edges around it, and one wide case.
///
/// `size_m` is `tokens x topk`, so each row names both. The hidden width alternates between a
/// multiple of the kernel's sixteen-row tile and one that is not, which is what makes the tail
/// guard part of the measurement rather than an assumption.
const DOWN_SHAPES: [(usize, usize, usize); 6] = [
    // (tokens, topk, hidden) -> size_m = 32, 63, 64, 65, 128, 1024
    (8, 4, 128),
    (9, 7, 100),
    (8, 8, 128),
    (13, 5, 100),
    (16, 8, 128),
    (128, 8, 128),
];

#[test]
fn the_imma_down_projection_agrees_with_a_host_reduction_of_the_dequantised_weights() {
    let Some(dev) = ampere_card() else { return };
    let mut failures: Vec<String> = Vec::new();
    // Held for the whole run: the repack cache is keyed by device pointer, so a stack freed
    // here can hand its address to the next one and be answered from the first one's entry.
    let mut keep: Vec<QTensor> = Vec::new();

    for (i, &(tokens, topk, hidden)) in DOWN_SHAPES.iter().enumerate() {
        let case = DownCase::build(
            &dev,
            tokens,
            topk,
            hidden,
            512,
            long_runs(tokens, topk),
            0xD0_0000 + i as u64,
        );
        let (want, bound) = case.reference();
        let got = case.launch();
        let (count, _, where_) = breaches(&got, &want, &bound, hidden);
        if count > 0 {
            failures.push(format!("{}: {where_}", case.label()));
        }
        keep.push(case.weights);
    }

    assert!(
        failures.is_empty(),
        "the IMMA down projection does not compute what the host says it should:\n  {}",
        failures.join("\n  ")
    );
}

/// The same judgement where an eight-pair tile holds more experts than the kernel keeps room
/// for - four - so that a drop there is not mistaken for anything else.
#[test]
fn the_imma_down_projection_serves_every_expert_in_a_crowded_tile() {
    let Some(dev) = ampere_card() else { return };
    let mut failures: Vec<String> = Vec::new();
    let mut keep: Vec<QTensor> = Vec::new();

    // Runs of two put four experts in a tile - the last that fits - and runs of one put eight.
    for (i, &run) in [4usize, 2, 1].iter().enumerate() {
        let case = DownCase::build(
            &dev,
            16,
            8,
            128,
            512,
            crowded_runs(16, 8, run),
            0xC0_0000 + i as u64,
        );
        let (want, bound) = case.reference();
        let got = case.launch();
        let (count, _, where_) = breaches(&got, &want, &bound, 128);
        if count > 0 {
            failures.push(format!(
                "{} pairs per expert, {} experts in an eight-pair tile: {where_}",
                run,
                8 / run
            ));
        }
        keep.push(case.weights);
    }

    assert!(
        failures.is_empty(),
        "the IMMA down projection loses experts when a tile holds several:\n  {}",
        failures.join("\n  ")
    );
}

/// Repeats of one down-projection launch, and whether what moves between them is the reduction
/// order or something wider.
///
/// The two answers need different fixes, so they are separated rather than lumped under "not
/// reproducible". A pair reaches its token through an `atomicAdd`, so the order the `topk`
/// contributions arrive in belongs to the card and the last bits follow it - that is the window
/// [`DownCase::reference`] already sizes, taken twice over since two runs may sit at opposite
/// ends of it. A move wider than that window is not an order, it is a race: two launches that
/// read or wrote different memory. The count of elements that moved at all is reported either
/// way, because bit-identical and merely-close are worth telling apart even when both pass.
#[test]
fn the_imma_down_projection_repeats_within_its_reduction_order() {
    let Some(dev) = ampere_card() else { return };
    let case = DownCase::build(&dev, 16, 8, 128, 512, long_runs(16, 8), 0xDE_7E01);
    let (_, bound) = case.reference();
    let runs: Vec<Vec<f32>> = (0..REPEATS).map(|_| case.launch()).collect();
    let (differing, worst_rel, gaps) = run_gaps(&runs);
    eprintln!(
        "{}: over {REPEATS} launches, {differing} of {} elements were not bit-identical, \
         the widest by {worst_rel:.3e} of the largest element",
        case.label(),
        gaps.len()
    );
    let racy: Vec<usize> = (0..gaps.len())
        .filter(|&i| gaps[i] > 2.0 * bound[i])
        .collect();
    assert!(
        racy.is_empty(),
        "{}: {} of {} elements moved between identical launches by more than the atomicAdd \
         order can account for - element {} moved by {:.3e} against a reduction-order window \
         of {:.3e}. That is a race, not an ordering.",
        case.label(),
        racy.len(),
        gaps.len(),
        racy[0],
        gaps[racy[0]],
        2.0 * bound[racy[0]]
    );
}

// ------------------------------------------------------------
// The fused gate‖up projection with SiLU
// ------------------------------------------------------------

/// One gate‖up case. Same reference discipline as [`DownCase`]: two dequantised stacks, an
/// activation the quantiser reproduces exactly, and the epilogue evaluated in f64.
struct SiluCase {
    gate: QTensor,
    up: QTensor,
    input: Tensor,
    slots_dev: Tensor,
    experts_dev: Tensor,
    slots: Vec<u32>,
    experts: Vec<u32>,
    x: Vec<f32>,
    gate_rows: Vec<WeightRow>,
    up_rows: Vec<WeightRow>,
    sums: Vec<Vec<f32>>,
    n_tokens: usize,
    topk: usize,
    width: usize,
    k: usize,
}

impl SiluCase {
    fn build(
        dev: &Device,
        n_tokens: usize,
        topk: usize,
        width: usize,
        k: usize,
        routing: (Vec<u32>, Vec<u32>, usize),
        seed: u64,
    ) -> Self {
        let (slots, experts, n_experts) = routing;
        let g_dense = expert_stack(n_experts, width, k, seed);
        let u_dense = expert_stack(n_experts, width, k, seed ^ 0x11FF);
        let (gate, g_deq) = q4k_pair(dev, &g_dense, (n_experts, width, k));
        let (up, u_deq) = q4k_pair(dev, &u_dense, (n_experts, width, k));

        // Gate and up read one activation row per REAL TOKEN - the `topk` pairs of a token
        // share it - which is one fewer index than the down projection has to get right.
        let x = exact_q8_1_rows(n_tokens, k, seed ^ 0x5AD2);

        Self {
            input: Tensor::from_vec(x.clone(), (n_tokens, k), dev).expect("upload activation"),
            slots_dev: Tensor::from_vec(slots.clone(), slots.len(), dev).expect("upload slots"),
            experts_dev: Tensor::from_vec(experts.clone(), experts.len(), dev)
                .expect("upload experts"),
            gate_rows: WeightRow::split(&g_deq, n_experts * width, k),
            up_rows: WeightRow::split(&u_deq, n_experts * width, k),
            sums: block_sums(&x, n_tokens, k),
            gate,
            up,
            slots,
            experts,
            x,
            n_tokens,
            topk,
            width,
            k,
        }
    }

    /// The reference, one row per routed pair, indexed by slot.
    ///
    /// The tolerance carries the epilogue as well as the two dot products: SiLU's derivative
    /// never exceeds `1.1`, so a drift of `Eg` in the gate moves the product by at most
    /// `1.1.Eg.|up|`, and the kernel's sigmoid is the hardware exponential, which is documented
    /// to two ulp.
    fn reference(&self) -> (Vec<f64>, Vec<f64>) {
        let mut want = vec![0.0f64; self.slots.len() * self.width];
        let mut bound = vec![0.0f64; self.slots.len() * self.width];
        for pair in 0..self.slots.len() {
            let slot = self.slots[pair] as usize;
            let expert = self.experts[pair] as usize;
            let token = slot / self.topk;
            let xrow = &self.x[token * self.k..(token + 1) * self.k];
            let sums = &self.sums[token];
            for col in 0..self.width {
                let at = expert * self.width + col;
                let (g, g_slack) = self.gate_rows[at].dot_and_slack(xrow, sums);
                let (u, u_slack) = self.up_rows[at].dot_and_slack(xrow, sums);
                let silu = g / (1.0 + (-g).exp());
                let out = slot * self.width + col;
                want[out] = silu * u;
                bound[out] = 1.1 * g_slack * u.abs()
                    + silu.abs() * u_slack
                    + (silu * u).abs() * (EXPF_ULP + HALF_ULP_F32);
            }
        }
        (want, bound)
    }

    fn launch(&self) -> Vec<f32> {
        super::moe_gemm_gguf_gate_up_silu_mul(
            &self.input,
            &self.gate,
            &self.up,
            &self.slots_dev,
            &self.experts_dev,
            self.topk,
        )
        .expect("the fused SiLU gate‖up refused the case")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("read back")
    }

    fn label(&self) -> String {
        format!(
            "size_m={} (tokens={} topk={}) N={} K={}",
            self.slots.len(),
            self.n_tokens,
            self.topk,
            self.width,
            self.k
        )
    }
}

/// The same span of `size_m` the down projection is judged over - the sub-threshold shapes are
/// the point here, since that is where this kernel runs on its own.
const SILU_SHAPES: [(usize, usize, usize); 7] = [
    // (tokens, topk, N) -> size_m = 3, 8, 32, 63, 64, 65, 128
    (3, 1, 128),
    (4, 2, 100),
    (8, 4, 128),
    (9, 7, 100),
    (8, 8, 128),
    (13, 5, 100),
    (16, 8, 128),
];

#[test]
fn the_fused_silu_gate_up_agrees_with_a_host_dot_of_the_dequantised_weights() {
    let Some(dev) = ampere_card() else { return };
    let mut failures: Vec<String> = Vec::new();
    let mut keep: Vec<QTensor> = Vec::new();

    for (i, &(tokens, topk, width)) in SILU_SHAPES.iter().enumerate() {
        let case = SiluCase::build(
            &dev,
            tokens,
            topk,
            width,
            512,
            long_runs(tokens, topk),
            0x51_0000 + i as u64,
        );
        let (want, bound) = case.reference();
        let got = case.launch();
        let (count, _, where_) = breaches(&got, &want, &bound, width);
        if count > 0 {
            failures.push(format!("{}: {where_}", case.label()));
        }
        keep.push(case.gate);
        keep.push(case.up);
    }

    assert!(
        failures.is_empty(),
        "the fused SiLU gate‖up does not compute what the host says it should:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn the_fused_silu_gate_up_writes_the_same_bits_every_run() {
    let Some(dev) = ampere_card() else { return };
    let case = SiluCase::build(&dev, 16, 8, 128, 512, long_runs(16, 8), 0x51_7E01);
    let runs: Vec<Vec<f32>> = (0..REPEATS).map(|_| case.launch()).collect();
    let (differing, worst_rel, gaps) = run_gaps(&runs);
    assert_eq!(
        differing,
        0,
        "{}: {REPEATS} identical launches did not write identical bits - {differing} of {} \
         elements moved, by up to {worst_rel:.3e} of the largest. This kernel stores each row \
         once, with no accumulation to reorder, so any movement at all is a race.",
        case.label(),
        gaps.len()
    );
}
