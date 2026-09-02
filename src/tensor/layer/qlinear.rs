//! A linear projection over a quantised weight, with an optional bias and optional adapters.
//!
//! The order is **weight, then adapter, then bias**, matching [`layer::Linear`]. It is not
//! arbitrary: all three arrangements compute `Wx + b + delta`, but float addition is not
//! associative, so a projection that adds them in another order disagrees in the last bits  -
//! and over the depth of a transformer that is enough to move a sampled token.
//!
//! Adapters are kept beside the weight, never merged into it. Merging into a quantised weight
//! would mean dequantising, adding, and requantising, which loses exactly the precision the
//! format exists to preserve. Held separately, the blob is untouched, several adapters compose
//! by addition, and detaching one is a drop rather than a reload.
//!
//! Any number of them fold into a single `([1, in, R], [1, R, out])` pair when attached, so the
//! forward costs two matmuls whether one adapter is loaded or five.

use super::{Linear, LoraDelta};
// The kernel-backed matmul, which the `quant` module re-exports under the shorter name
// `QMatMul` - the same type, not the enum of that name in `quantized`.
use super::super::quantized::QKernelMatMul;
use super::{DType, Error, Result, Tensor};

/// Where the projection's weight is held.
pub enum Weight {
    /// Block-quantised, computed by the production kernels.
    Quant(QKernelMatMul),
    /// Dense, held at the dtype it was loaded in.
    ///
    /// The activation is cast to that dtype for the matmul and the product cast back to F32,
    /// so the GEMM runs at the weight's precision - BF16 on a card - while the elementwise
    /// arithmetic around it stays F32. Casting the weight up instead would double what it
    /// occupies, on the path chosen precisely because memory was the constraint.
    ///
    /// The `Linear` is a bare weight here: the bias and the adapters belong to the `QLinear`
    /// around it, so there is one place each lives rather than two that must agree.
    Dense(Linear, DType),
}

impl Weight {
    /// The projection alone, without the bias or the adapters the `QLinear` around it holds.
    ///
    /// A fused SwiGLU needs the gate and the up projection raw, because it multiplies them
    /// before anything is added to either - so it asks for the matmul and nothing else.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Weight::Quant(w) => w.forward(x),
            Weight::Dense(l, wdtype) => {
                let xw = if x.dtype() == *wdtype {
                    x.clone()
                } else {
                    x.to_dtype(*wdtype)?
                };
                let y = l.forward(&xw)?;
                if y.dtype() == DType::F32 {
                    Ok(y)
                } else {
                    y.to_dtype(DType::F32)
                }
            }
        }
    }
}

/// A linear projection: `y = weight(x) + lora(x) + bias`.
pub struct QLinear {
    weight: Weight,
    bias: Option<Tensor>,
    /// Attached adapters, kept rather than merged: merging into a quantised weight would mean
    /// dequantising, adding and requantising, which loses precision the format exists to keep.
    lora: Vec<LoraDelta>,
    /// The attached adapters folded into one pair, so the forward cost does not grow with
    /// their number.
    fused: Option<(Tensor, Tensor)>,
    /// Input and output features, so a wrongly shaped adapter is refused when it is attached
    /// rather than producing a shape error deep in a forward.
    in_dim: usize,
    out_dim: usize,
}

impl QLinear {
    pub fn new(weight: Weight, bias: Option<Tensor>, in_dim: usize, out_dim: usize) -> Self {
        Self {
            weight,
            bias,
            lora: Vec::new(),
            fused: None,
            in_dim,
            out_dim,
        }
    }

    pub fn weight(&self) -> &Weight {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Attach an adapter, refusing one that does not fit.
    ///
    /// Checked here rather than at the forward because a mis-shaped adapter attached at load
    /// would otherwise fail on the first token of the first request, by which point the error
    /// names a matmul rather than the adapter file that caused it.
    pub fn add_lora(&mut self, delta: LoraDelta) -> Result<()> {
        let (din, r) = delta.down.shape().dims2()?;
        let (r2, dout) = delta.up.shape().dims2()?;
        if din != self.in_dim || dout != self.out_dim || r != r2 {
            return Err(Error(format!(
                "lora: down {din}x{r} up {r2}x{dout} does not fit a {}x{} projection",
                self.in_dim, self.out_dim
            )));
        }
        self.lora.push(delta);
        self.fused = super::super::lora::fuse_loras(&self.lora, self.in_dim, self.out_dim);
        Ok(())
    }

    pub fn clear_lora(&mut self) {
        self.lora.clear();
        self.fused = None;
    }

    pub fn lora_count(&self) -> usize {
        self.lora.len()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.weight.forward(x)?;

        if let Some((down, up)) = &self.fused {
            let dims = x.dims().to_vec();
            let k = *dims
                .last()
                .ok_or_else(|| Error("qlinear: rank-0 input".into()))?;
            let rows = x.elem_count() / k;
            // The adapter runs in F32 whatever the base weight's format: it is a rank-r
            // correction over dimensions in the thousands, so its cost is a rounding error on
            // the projection, and computing it at the base's precision would throw away the
            // accuracy the adapter was trained at for nothing.
            let xf = x.to_dtype(super::DType::F32)?.reshape(vec![1, rows, k])?;
            let mut odims = dims;
            *odims
                .last_mut()
                .ok_or_else(|| Error("qlinear: rank-0 input".into()))? = self.out_dim;
            let delta = xf.matmul(down)?.matmul(up)?.reshape(odims)?;
            y = y.add(&delta.to_dtype(y.dtype())?)?;
        }

        match &self.bias {
            None => Ok(y),
            Some(b) => y.broadcast_add(b),
        }
    }
}

impl super::super::ops::traits::Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, xs)
    }
}

/// Written out rather than derived: the quantised weight holds a device blob and a kernel
/// workspace, neither of which has a useful rendering, and printing a projection should not
/// mean printing megabytes. What identifies one here is its shape and where its weight lives.
impl std::fmt::Debug for QLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QLinear")
            .field("in", &self.in_dim)
            .field("out", &self.out_dim)
            .field(
                "weight",
                &match &self.weight {
                    Weight::Quant(_) => "quantised",
                    Weight::Dense(..) => "dense",
                },
            )
            .field("bias", &self.bias.is_some())
            .field("adapters", &self.lora.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QHostTensor};
    use crate::tensor::{Device, Tensor};

    /// `k` inputs, `n` outputs, and an input whose sum is known.
    ///
    /// The weight is a plain dense tensor rather than a quantised blob: what this module adds
    /// over the matmul it wraps is the adapter and the bias, and a Q8_0 round trip would only
    /// put quantisation error between the assertion and the thing being asserted.
    fn fixture(k: usize, n: usize) -> (QLinear, Tensor, f32) {
        // [out, in], the layout a projection weight is stored in. The pattern is deliberately
        // NOT zero-mean against the input below: a weight whose period is coprime with the
        // input's sums to nothing over a long row, and a test whose expected output is 1e-8 is
        // measuring float noise rather than the projection.
        let w: Vec<f32> = (0..n * k)
            .map(|i| 0.05 + ((i % 17) as f32) / 64.0)
            .collect();
        let wt = Tensor::from_vec_f32(w, vec![n, k]).unwrap();
        let xs: Vec<f32> = (0..k).map(|i| (i % 5) as f32 * 0.1).collect();
        let sum_x = xs.iter().sum();
        let x = Tensor::from_vec_f32(xs, vec![1, k]).unwrap();
        (
            QLinear::new(
                Weight::Dense(Linear::new(wt, None).unwrap(), DType::F32),
                None,
                k,
                n,
            ),
            x,
            sum_x,
        )
    }

    fn ones_adapter(k: usize, n: usize, r: usize, scale: f32) -> LoraDelta {
        LoraDelta {
            down: Tensor::from_vec_f32(vec![1.0; k * r], vec![k, r]).unwrap(),
            up: Tensor::from_vec_f32(vec![1.0; r * n], vec![r, n]).unwrap(),
            scale,
        }
    }

    /// Two adapters must move the output by the sum of what each moves it by.
    ///
    /// This is the property the four hand-written copies of this type did not have: each held
    /// a single `Option`, so attaching a second adapter replaced the first instead of composing
    /// with it - silently, since replacing one adapter with another still produces an output
    /// that has visibly been adapted.
    #[test]
    fn adapters_compose_rather_than_replace() {
        let (k, n) = (256usize, 32usize);
        let (mut lin, x, sum_x) = fixture(k, n);
        let base = lin.forward(&x).unwrap().to_vec_f32();

        lin.add_lora(ones_adapter(k, n, 2, 0.25)).unwrap();
        lin.add_lora(ones_adapter(k, n, 3, 0.5)).unwrap();
        assert_eq!(lin.lora_count(), 2);

        // An all-ones adapter of rank r and scale s adds `s * r * sum(x)` to every output.
        let expected = 0.25 * 2.0 * sum_x + 0.5 * 3.0 * sum_x;
        let got = lin.forward(&x).unwrap().to_vec_f32();
        for i in 0..n {
            let moved = got[i] - base[i];
            assert!(
                (moved - expected).abs() < 1e-2,
                "out[{i}] moved by {moved}, expected {expected} from both adapters"
            );
        }

        lin.clear_lora();
        assert_eq!(lin.forward(&x).unwrap().to_vec_f32(), base);
    }

    /// The bias is added AFTER the adapter, and the test would pass either way if it only
    /// checked the total - so it checks that removing the adapter leaves exactly the bias.
    #[test]
    fn the_bias_survives_an_adapter_being_detached() {
        let (k, n) = (256usize, 32usize);
        let (lin, x, _) = fixture(k, n);
        let unbiased = lin.forward(&x).unwrap().to_vec_f32();

        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mut lin = QLinear::new(
            lin.weight,
            Some(Tensor::from_vec_f32(bias.clone(), vec![n]).unwrap()),
            k,
            n,
        );
        lin.add_lora(ones_adapter(k, n, 2, 0.25)).unwrap();
        lin.clear_lora();

        let got = lin.forward(&x).unwrap().to_vec_f32();
        for i in 0..n {
            assert!(
                (got[i] - (unbiased[i] + bias[i])).abs() < 1e-4,
                "out[{i}] = {} but the weight gave {} and the bias is {}",
                got[i],
                unbiased[i],
                bias[i]
            );
        }
    }

    /// The quantised holder projects the same vector, within its own quantisation error.
    ///
    /// Not a formality: the dense path casts the activation to the weight's dtype and the
    /// product back, and a cast applied in the wrong place changes the answer without failing.
    ///
    /// The tolerance is Q8_0's, not a convenience: an 8-bit block format reconstructs a weight
    /// to within about a percent, so the two answers differ by the format's own error and
    /// nothing else. Demanding equality would be demanding that quantisation be lossless.
    #[test]
    fn a_quantised_weight_projects_like_a_dense_one() {
        let (k, n) = (256usize, 32usize);
        let (dense, x, _) = fixture(k, n);
        let via_dense = dense.forward(&x).unwrap().to_vec_f32();

        let w: Vec<f32> = (0..n * k)
            .map(|i| 0.05 + ((i % 17) as f32) / 64.0)
            .collect();
        let bytes = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Q8_0, &w).unwrap();
        let qt = QHostTensor::from_bytes(&bytes, GgmlDType::Q8_0, vec![n, k]).unwrap();
        let quantised = QLinear::new(
            Weight::Quant(
                QKernelMatMul::from_qtensor_on(std::sync::Arc::new(qt), &Device::Cpu).unwrap(),
            ),
            None,
            k,
            n,
        );
        let via_quant = quantised.forward(&x).unwrap().to_vec_f32();

        for i in 0..n {
            let (a, b) = (via_quant[i], via_dense[i]);
            assert!(
                (a - b).abs() <= 2e-2 * b.abs().max(1.0),
                "out[{i}]: quantised {a} against dense {b}"
            );
        }
    }

    /// An adapter that does not fit the projection is refused when it is attached.
    #[test]
    fn a_wrongly_shaped_adapter_is_refused() {
        let (k, n) = (256usize, 32usize);
        let (mut lin, _, _) = fixture(k, n);
        // Right rank, wrong input width.
        let bad = ones_adapter(k / 2, n, 2, 1.0);
        assert!(
            lin.add_lora(bad).is_err(),
            "a {}-wide adapter fitted a {k}-wide projection",
            k / 2
        );
        assert_eq!(
            lin.lora_count(),
            0,
            "a refused adapter must not be retained"
        );
    }
}

// ---------------------------------------------------------------------------
// Builders over a quantised weight store.
//
// The compat loader offers these and `QVarBuilder` did not, which is why the files that need
// them stayed on it. They are thin - `get_f32` already fetches and dequantises in one step  -
// and thin is the point: what kept those files where they were was an absent name, not an
// absent capability.
// ---------------------------------------------------------------------------

/// Widen, activate, narrow, over quantised weights.
///
/// The dense `Mlp` next door is the same three steps; the two cannot be one because the
/// projections are different types and bridging them would dequantise, which is a change of
/// arithmetic and not of shape. Two vision towers carried this triple inline.
#[derive(Debug)]
pub struct QMlp {
    fc1: QLinear,
    act: crate::tensor::ops::Activation,
    fc2: QLinear,
}

impl QMlp {
    pub fn new(fc1: QLinear, act: crate::tensor::ops::Activation, fc2: QLinear) -> Self {
        Self { fc1, act, fc2 }
    }

    /// The widening projection.
    ///
    /// A caller that cannot run [`forward`](crate::tensor::Module::forward) whole - one that
    /// slabs the intermediate to hold a memory budget, or that dispatches each projection to
    /// a device queue itself - needs the two steps apart. They are the same weights either way.
    pub fn fc1(&self) -> &QLinear {
        &self.fc1
    }

    /// The narrowing projection.
    pub fn fc2(&self) -> &QLinear {
        &self.fc2
    }

    /// The activation between them.
    pub fn act(&self) -> crate::tensor::ops::Activation {
        self.act
    }

    /// The intermediate width - what makes a slabbed buffer big, so what a memory budget has
    /// to be derived from. Read from the projection rather than carried alongside it.
    pub fn hidden(&self) -> usize {
        self.fc1.out_dim()
    }
}

impl crate::tensor::Module for QMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

/// The gated feed-forward over quantised projections: `down(silu(gate(x)) * up(x))`.
///
/// The quantised twin of [`layer::SwiGlu`](super::SwiGlu), for the same reason [`QMlp`] is
/// the twin of `Mlp`: the projections are a different type, and bridging them would
/// dequantise.
#[derive(Debug)]
pub struct QSwiGlu {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl QSwiGlu {
    pub fn new(gate: QLinear, up: QLinear, down: QLinear) -> Self {
        Self { gate, up, down }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.silu()?;
        let rhs = self.up.forward(xs)?;
        self.down.forward(&lhs.mul(&rhs)?)
    }
}

impl crate::tensor::Module for QSwiGlu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

use super::super::quantized::QVarBuilder;
use super::{Embedding, LayerNorm};

/// `{prefix}.weight` as a quantised projection, with `{prefix}.bias` dequantised beside it.
pub fn qlinear(in_dim: usize, out_dim: usize, vb: &QVarBuilder) -> Result<QLinear> {
    let weight = vb.qmatmul(in_dim, out_dim, "weight")?;
    let bias = vb.get_f32(out_dim, "bias")?;
    Ok(QLinear::new(
        Weight::Quant(weight),
        Some(bias),
        in_dim,
        out_dim,
    ))
}

/// The same, for a projection the checkpoint publishes without a bias.
pub fn qlinear_no_bias(in_dim: usize, out_dim: usize, vb: &QVarBuilder) -> Result<QLinear> {
    let weight = vb.qmatmul(in_dim, out_dim, "weight")?;
    Ok(QLinear::new(Weight::Quant(weight), None, in_dim, out_dim))
}

/// Whichever of the two the caller's architecture says, decided by a flag rather than by two
/// call sites that must be kept in step.
pub fn qlinear_b(in_dim: usize, out_dim: usize, bias: bool, vb: &QVarBuilder) -> Result<QLinear> {
    if bias {
        qlinear(in_dim, out_dim, vb)
    } else {
        qlinear_no_bias(in_dim, out_dim, vb)
    }
}

/// A layer norm read from a quantised store.
///
/// Norms are dequantised rather than kept in blocks: they are a vector per layer against a
/// matrix per projection, so the memory saved is nothing and the precision lost is applied to
/// every element that passes through.
pub fn q_layer_norm(size: usize, eps: f64, vb: &QVarBuilder) -> Result<LayerNorm> {
    let weight = vb.get_f32(size, "weight")?;
    let bias = vb.get_f32(size, "bias")?;
    Ok(LayerNorm::new(weight, Some(bias), eps as f32))
}

/// An embedding table read from a quantised store, dequantised for the lookup.
pub fn q_embedding(vocab: usize, dim: usize, vb: &QVarBuilder) -> Result<Embedding> {
    Ok(Embedding::new(vb.get_f32((vocab, dim), "weight")?))
}
