//! Activations, and the enum that dispatches over them by name.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Sigmoid: `1 / (1 + exp(-x))`. Drop-in for `sigmoid`.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.device().is_cuda() && x.dtype() == DType::F32 {
            return crate::inference::kernel::fused::fused_sigmoid_f32(x);
        }
    }
    (x.neg()?.exp()? + 1.0)?.recip()
}

/// SiLU / swish: `x * sigmoid(x) = x / (1 + exp(-x))`. Drop-in for
/// `silu`.
/// Gated feed-forward activation: `silu(gate) * up`, in one pass.
///
/// The two projections leave their matmuls in half precision. Widening each,
/// activating the gate, multiplying and narrowing the result back each wrote the
/// whole wide intermediate before the next step read it, so the tensor was
/// walked several times for one elementwise result. Here the widen, the
/// activation, the product and the narrow all ride along with a single pass over
/// the two inputs, and only the result is written.
///
/// The activation keeps the reciprocal-then-multiply form used by the separate
/// step, which a division would not reproduce bit-for-bit, so results are
/// identical to the unfused chain. Anything not on this path falls back to it.
/// `out[i] = silu(gate[i]) * up[i]` over one contiguous chunk. The activation's
/// exponential is the cost here; running it eight lanes at a time with a vector
/// exp (instead of a per-element library call) and converting the half-precision
/// inputs and output a vector at a time removes the per-element function-call
/// and scalar-convert overhead. The scalar tail keeps the same recip-then-mul
/// form as the reference so the two agree bit-for-bit off the vector path.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn swiglu_chunk(g: &[half::f16], u: &[half::f16], o: &mut [half::f16]) {
    use core::arch::x86_64::*;
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= o.len() {
            let gv = _mm256_cvtph_ps(_mm_loadu_si128(g.as_ptr().add(i) as *const __m128i));
            let uv = _mm256_cvtph_ps(_mm_loadu_si128(u.as_ptr().add(i) as *const __m128i));
            // silu(g) = g / (1 + exp(-g))
            let e = crate::inference::kernel::cpu_decode_exec::simd::exp8(_mm256_sub_ps(zero, gv));
            let sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
            let res = _mm256_mul_ps(_mm256_mul_ps(gv, sig), uv);
            _mm_storeu_si128(
                o.as_mut_ptr().add(i) as *mut __m128i,
                _mm256_cvtps_ph::<0>(res),
            );
            i += 8;
        }
        while i < o.len() {
            let gv = g[i].to_f32();
            let s = gv * (1.0 / (1.0 + (-gv).exp()));
            o[i] = half::f16::from_f32(s * u[i].to_f32());
            i += 1;
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn swiglu_chunk(g: &[half::f16], u: &[half::f16], o: &mut [half::f16]) {
    for i in 0..o.len() {
        let gv = g[i].to_f32();
        let s = gv * (1.0 / (1.0 + (-gv).exp()));
        o[i] = half::f16::from_f32(s * u[i].to_f32());
    }
}

/// Gated feed-forward activation from a packed projection: the input is
/// `[.., 2i]` = `[gate | up]` in the last dim, the output is `[.., i]` =
/// `silu(gate) * up`, produced in one pass.
///
/// Done as separate ops, the gate half is copied to the host, run through a
/// scalar library exponential element by element to form `silu(gate)` as a
/// whole new tensor, then a second pass multiplies it by the up half. Reading
/// the packed tensor once and writing only the result - with a vector
/// exponential and the product folded in - removes the host round-trip, the
/// per-element scalar exp and the intermediate tensor. The activation keeps the
/// reciprocal-then-multiply form of the separate `silu`; only the exponential
/// changes (a vector polynomial rather than the scalar library call), so the
/// result matches to within its last bits. Returns `None` off this path (non-CPU,
/// non-F32, odd last dim, or a view that is not contiguous), and the caller keeps
/// its existing chain.
pub fn split_silu_mul_f32(up_states: &Tensor) -> Result<Option<Tensor>> {
    if !matches!(up_states.device(), Device::Cpu) || up_states.dtype() != DType::F32 {
        return Ok(None);
    }
    let last = up_states.dim(D::Minus1)?;
    if last == 0 || last % 2 != 0 {
        return Ok(None);
    }
    let i = last / 2;
    let data = match up_states.f32_data() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    if data.len() % last != 0 {
        return Ok(None);
    }
    let rows = data.len() / last;
    let mut out = vec![0f32; rows * i];
    // Shared spin-pool, NOT rayon: the fused gated-FFN activation runs once per
    // layer on the prefill critical path between quantized GEMMs; a second rayon
    // pool here made the caller drive its bridge serially while the GEMM pool's
    // workers idled (cf. the silu/quantize fixes).
    crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, i, &|r, o| {
        silu_mul_row(&data[r * last..r * last + last], i, o);
    });
    let mut odims = up_states.dims().to_vec();
    *odims.last_mut().unwrap() = i;
    Ok(Some(Tensor::from_vec(out, odims, &up_states.device())?))
}

/// One row of the gated activation: `out[c] = silu(gate[c]) * up[c]`, where the
/// row is `[gate(0..i) | up(i..2i)]`. Vectorized with the reciprocal-then-
/// multiply silu and a vector exp; a scalar tail keeps the same form.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn silu_mul_row(row: &[f32], i: usize, o: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let g = row.as_ptr();
        let u = row.as_ptr().add(i);
        let mut c = 0;
        while c + 8 <= i {
            let gv = _mm256_loadu_ps(g.add(c));
            let uv = _mm256_loadu_ps(u.add(c));
            // silu(g) = g * (1 / (1 + exp(-g)))
            let e = crate::inference::kernel::cpu_decode_exec::simd::exp8(_mm256_sub_ps(zero, gv));
            let sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
            _mm256_storeu_ps(
                o.as_mut_ptr().add(c),
                _mm256_mul_ps(_mm256_mul_ps(gv, sig), uv),
            );
            c += 8;
        }
        while c < i {
            let gv = *g.add(c);
            o[c] = (gv * (1.0 / (1.0 + (-gv).exp()))) * *u.add(c);
            c += 1;
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn silu_mul_row(row: &[f32], i: usize, o: &mut [f32]) {
    for c in 0..i {
        let gv = row[c];
        o[c] = (gv * (1.0 / (1.0 + (-gv).exp()))) * row[i + c];
    }
}

pub fn swiglu(gate: &Tensor, up: &Tensor, out_dtype: DType) -> Result<Tensor> {
    if matches!(gate.device(), Device::Cpu)
        && gate.dtype() == DType::F16
        && up.dtype() == DType::F16
        && out_dtype == DType::F16
        && gate.dims() == up.dims()
    {
        let g = gate.f16_data()?;
        let u = up.f16_data()?;
        if g.len() == u.len() {
            use rayon::prelude::*;
            let n = g.len();
            let mut out: Vec<half::f16> = Vec::with_capacity(n);
            // Fully overwritten below, so skip the zero-fill.
            #[allow(clippy::uninit_vec)]
            unsafe {
                out.set_len(n)
            };
            // Chunk large enough to amortize task dispatch.
            const CHUNK: usize = 8192;
            out.par_chunks_mut(CHUNK)
                .zip(g.par_chunks(CHUNK))
                .zip(u.par_chunks(CHUNK))
                .for_each(|((o, gc), uc)| {
                    swiglu_chunk(gc, uc, o);
                });
            return Tensor::from_vec(out, gate.dims().to_vec(), &gate.device());
        }
    }
    // Fallback: the separate steps, unchanged.
    let g = silu(&gate.to_dtype(DType::F32)?)?;
    let u = up.to_dtype(DType::F32)?;
    (g * u)?.to_dtype(out_dtype)
}

/// Fused gpt-oss `swiglu_oai` gated activation in ONE pass over the f32 rows,
/// folding the optional per-column gate/up bias. Replaces the ~13 separate
/// elementwise tensor passes (two bias adds, the affine/relu clamp chains, the
/// sigmoid, and two muls - each allocating a full intermediate) that profiling
/// put at ~22% of gpt-oss CPU prefill. Bit-identical to that chain: every affine
/// has `|mul| == 1` (exact) except the `alpha.x` scale, which mirrors
/// `affine(alpha as f32, 0.0)`, and the sigmoid is `1/(exp(-.)+1)` with the same
/// `f32::exp`, so greedy tokens are unchanged. `gate`/`up` are `[.., n]` f32;
/// biases, if present, are `[n]`.
pub fn swiglu_oai(
    gate: &Tensor,
    up: &Tensor,
    gbias: Option<&Tensor>,
    ubias: Option<&Tensor>,
    alpha: f64,
    limit: f64,
) -> Result<Tensor> {
    let dims = gate.dims().to_vec();
    let n = *dims.last().expect("swiglu_oai: gate must be at least 1-D");
    let g = gate.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let u = up.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let to_vec = |b: Option<&Tensor>| -> Result<Option<Vec<f32>>> {
        Ok(match b {
            Some(t) => Some(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        })
    };
    let gb = to_vec(gbias)?;
    let ub = to_vec(ubias)?;
    let l = limit as f32;
    let a = alpha as f32;
    let rows = g.len() / n;
    let mut out = vec![0f32; g.len()];
    for r in 0..rows {
        let base = r * n;
        for i in 0..n {
            let idx = base + i;
            let gi = g[idx] + gb.as_ref().map_or(0.0, |b| b[i]);
            let ui = u[idx] + ub.as_ref().map_or(0.0, |b| b[i]);
            // x = min(gi, l): affine(-1,l) -> relu -> affine(-1,l).
            let t = (-1.0f32 * gi + l).max(0.0);
            let x = -1.0f32 * t + l;
            // gg = min(max(ui, -l), l): the four affine/relu steps of the chain.
            let s = (1.0f32 * ui + l).max(0.0);
            let s = 1.0f32 * s + (-l);
            let s = (-1.0f32 * s + l).max(0.0);
            let gg = -1.0f32 * s + l;
            // act = x * sigmoid(alpha.x); sigmoid = 1/(exp(-.)+1).
            let xa = a * x;
            let sig = 1.0f32 / ((-xa).exp() + 1.0);
            out[idx] = (x * sig) * (gg + 1.0f32);
        }
    }
    Tensor::from_vec(out, dims, &gate.device())
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.device().is_cuda() && x.dtype() == DType::F32 {
            return crate::inference::kernel::fused::fused_silu_f32(x);
        }
    }
    // CPU F32 fused: x / (1 + exp(-x)) in a single pass (vs neg/exp/add/recip/mul
    // = 5 allocating ops); runs over the wide FFN intermediate once per layer.
    if matches!(x.device(), Device::Cpu) && x.dtype() == DType::F32 {
        let mut out = x.flatten_all()?.to_vec1::<f32>()?;
        // recip-then-mul mirrors the chain (x * sigmoid(x)) exactly.
        let f = |v: &mut f32| *v *= 1.0 / (1.0 + (-*v).exp());
        if out.len() >= 4096 {
            // Shared spin-pool, NOT rayon: this runs over the wide FFN intermediate
            // once per layer, on the prefill critical path between quantized GEMMs.
            // On rayon it woke a second threadpool whose bridge the caller drove
            // serially while the GEMM pool's workers idled - move it onto the one
            // pool so there is a single set of workers (cf. the flash-attn/quantize
            // fixes). Scalar exp kept verbatim -> bit-identical.
            crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, 1 << 13, &|_, d| {
                for v in d.iter_mut() {
                    *v *= 1.0 / (1.0 + (-*v).exp());
                }
            });
        } else {
            out.iter_mut().for_each(f);
        }
        return Tensor::from_vec(out, x.dims().to_vec(), &x.device());
    }
    x.broadcast_mul(&sigmoid(x)?)
}

/// The activations checkpoints name, and what each one is.
///
/// The GELU family is three entries and not one on purpose: `gelu` in a config file means
/// the error-function form, `gelu_new` and `gelu_pytorch_tanh` the tanh approximation, and a
/// checkpoint that says one while getting the other drifts a few thousandths per layer.
/// `QuickGelu` is a third approximation again - `x.σ(1.702x)`, what CLIP was trained with.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Silu,
    Sigmoid,
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
    #[serde(alias = "quick_gelu")]
    QuickGelu,
}

impl Activation {
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Silu => silu(xs),
            Self::Sigmoid => sigmoid(xs),
            Self::GeluPytorchTanh => xs.gelu(),
            Self::QuickGelu => xs.mul(&xs.scale(1.702)?.sigmoid()?),
        }
    }
}

impl crate::tensor::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.apply(xs)
    }
}
