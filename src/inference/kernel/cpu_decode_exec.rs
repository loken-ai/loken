//! Zero-allocation, single-pool dense-CPU decode executor.
//!
//! Root cause of the CPU dense-decode gap vs ollama (measured): our per-op
//! Tensor dispatch allocates a fresh buffer for every one of ~240 ops/token and
//! leaves cores idle in the serial glue between ops, while ollama runs the whole
//! token on one persistent threadpool over reused work-buffers. This module is
//! the foundation of the replacement: a `DecodeArena` of persistent scratch
//! buffers (fixed decode shapes, reused every token) plus slice-based versions
//! of the cheap elementwise ops, so the dense decode forward can run with zero
//! per-token allocation. The heavy ops already have slice primitives
//! (`quant_cpu::matmul_bytes`, `cpu_q8_kv::CpuQ8Kv::attention`).
//!
//! Built and validated bottom-up (this commit = the arena + elementwise
//! primitives, unit-tested vs naive references); the layer-forward assembly and
//! wiring follow, validated for coherence on the real model at each step.

/// Per-layer constant decode parameters, materialised as plain `f32` slices
/// once on the first decode token and reused thereafter. The norm weights and
/// QKV/qk-norm biases never change, but reading them straight off the quantized
/// `WeightedNorm`/bias tensors needs a `to_vec1` allocation+copy each call  -
/// ~5 per layer x n_layers per token of pure repeated conversion. Caching them
/// makes the decode path genuinely zero-alloc.
pub struct DecodeNormCache {
    pub attn_norm_w: Vec<f32>,
    pub attn_eps: f32,
    pub ffn_norm_w: Vec<f32>,
    pub ffn_eps: f32,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub q_norm_w: Option<(Vec<f32>, f32)>,
    pub k_norm_w: Option<(Vec<f32>, f32)>,
    // gemma4 sandwich-norm: post-attention and post-FFN RMSNorm (learned weight).
    pub post_attn_norm_w: Option<(Vec<f32>, f32)>,
    pub post_ffn_norm_w: Option<(Vec<f32>, f32)>,
    // gemma4 per-layer-embedding (PLE) post-norm (learned weight) + per-layer
    // output scale (`[hidden]` broadcast multiply, altup_correct_scale).
    pub ple_post_norm_w: Option<(Vec<f32>, f32)>,
    pub ple_output_scale: Option<Vec<f32>>,
}

/// Persistent per-token scratch for one dense decode step. Allocated once for
/// the model's fixed decode shapes and reused every token - no per-op alloc.
pub struct DecodeArena {
    pub hidden: usize,
    pub ffn: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    // reusable buffers
    pub norm: Vec<f32>,    // [hidden] norm output
    pub qkvfull: Vec<f32>, // [(n_head + 2*n_kv_head)*head_dim] for fused-qkv split
    pub q: Vec<f32>,       // [n_head*head_dim]
    pub k: Vec<f32>,       // [n_kv_head*head_dim]
    pub v: Vec<f32>,       // [n_kv_head*head_dim]
    pub attn: Vec<f32>,    // [n_head*head_dim] attention output
    pub proj: Vec<f32>,    // [hidden] o-proj output
    pub gateup: Vec<f32>,  // [2*ffn] fused gate‖up (or gate=[0..ffn], up=[ffn..2ffn] for split)
    pub act: Vec<f32>,     // [ffn] silu(gate)*up
    pub ple: Vec<f32>,     // [ple_dim] gemma4 per-layer-embedding gate output (lazily sized)
}

impl DecodeArena {
    /// `ffn` = intermediate_size (the FFN hidden dim, NOT 2x).
    pub fn new(
        hidden: usize,
        ffn: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            hidden,
            ffn,
            n_head,
            n_kv_head,
            head_dim,
            norm: vec![0.0; hidden],
            qkvfull: vec![0.0; (n_head + 2 * n_kv_head) * head_dim],
            q: vec![0.0; n_head * head_dim],
            k: vec![0.0; n_kv_head * head_dim],
            v: vec![0.0; n_kv_head * head_dim],
            attn: vec![0.0; n_head * head_dim],
            proj: vec![0.0; hidden],
            gateup: vec![0.0; 2 * ffn],
            act: vec![0.0; ffn],
            ple: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2/FMA vectorized elementwise decode primitives.
//
// The per-token dense decode runs these on every layer (silu/gelu over
// ffn≈14336, rms_norm over hidden≈5120). ggml vectorizes them (8-lane SIMD +
// a fast polynomial exp/tanh); a scalar `f32::exp`/`tanh` per element is ~8 ns
// each and dominated the non-GEMV decode overhead (~573k scalar exp/token for
// mistral: ffn 14336 x 40 layers). These use a Cephes-accuracy 8-lane exp
// (`exp8`, ~1 ulp) so parity vs the scalar reference stays well under 1e-3.
//
// Gated on compile-time `target_feature="avx2"` (build uses target-cpu=native).
// A scalar fallback (`*_scalar`) is kept for non-AVX2 targets and as the
// reference the unit tests / micro-bench validate against.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod simd {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// 8-lane single-precision exp, Cephes `exp256_ps` polynomial (~1 ulp).
    /// Matches `f32::exp` to a few x1e-7 relative over the finite range.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp8(x: __m256) -> __m256 {
        let hi = _mm256_set1_ps(88.376_26);
        let lo = _mm256_set1_ps(-88.376_26);
        let log2ef = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(0.693_359_4);
        let c2 = _mm256_set1_ps(-2.121_944_4e-4);

        let mut x = _mm256_min_ps(hi, _mm256_max_ps(lo, x));
        // fx = floor(x*log2ef + 0.5)
        let fx = _mm256_floor_ps(_mm256_fmadd_ps(x, log2ef, half));
        // x -= fx*c1; x -= fx*c2   (Cody-Waite range reduction)
        x = _mm256_fnmadd_ps(fx, c1, x);
        x = _mm256_fnmadd_ps(fx, c2, x);
        let z = _mm256_mul_ps(x, x);
        // Horner polynomial for exp(x) on the reduced range.
        let mut y = _mm256_set1_ps(1.987_569_1e-4);
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.398_199_9e-3));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.333_452e-3));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.166_579_6e-2));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.666_666_5e-1));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5e-1));
        y = _mm256_fmadd_ps(y, z, x);
        y = _mm256_add_ps(y, one);
        // build 2^fx by injecting the exponent bits
        let imm0 = _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(0x7f)),
            23,
        );
        _mm256_mul_ps(y, _mm256_castsi256_ps(imm0))
    }

    /// Horizontal sum of an 8-lane vector.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let sh = _mm_movehl_ps(s, s);
        let s = _mm_add_ps(s, sh);
        let sh = _mm_shuffle_ps(s, s, 0x1);
        _mm_cvtss_f32(_mm_add_ss(s, sh))
    }

    /// Fused softmax numerator: in-place `x[i] = exp(x[i] - max)`, returns Σx[i].
    /// One pass - the exp and the sum-reduction share the same 8-lane sweep (an
    /// AVX2 partial-sum accumulator, horizontal-added once, plus a scalar tail).
    /// This is the attention softmax over the (windowed/full) score vector; at
    /// long ctx it's ~seq exps x n_heads x n_layers per token, the 2.5K cliff.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp_sub_max_sum(x: &mut [f32], max: f32) -> f32 {
        let n = x.len();
        let vmax = _mm256_set1_ps(max);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let e = exp8(_mm256_sub_ps(v, vmax));
            _mm256_storeu_ps(x.as_mut_ptr().add(i), e);
            acc = _mm256_add_ps(acc, e);
            i += 8;
        }
        let mut sum = hsum(acc);
        while i < n {
            let e = (x[i] - max).exp();
            x[i] = e;
            sum += e;
            i += 1;
        }
        sum
    }

    /// `out[i] = silu(gate[i]) * up[i]`, `silu(z)=z/(1+e^-z)`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
        let n = gate.len();
        let one = _mm256_set1_ps(1.0);
        let neg = _mm256_set1_ps(-1.0);
        let mut i = 0;
        while i + 8 <= n {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let u = _mm256_loadu_ps(up.as_ptr().add(i));
            let e = exp8(_mm256_mul_ps(g, neg)); // e^-g
            let sig = _mm256_div_ps(g, _mm256_add_ps(one, e));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(sig, u));
            i += 8;
        }
        while i < n {
            let g = gate[i];
            out[i] = (g / (1.0 + (-g).exp())) * up[i];
            i += 1;
        }
    }

    /// Core `out[i] = gelu_tanh(gate[i]) * up[i]` over raw pointers so it can be
    /// used both distinct and in-place (`gate == out`). Uses the identity
    /// `0.5*g*(1+tanh(y)) = g*sigmoid(2y)` (mathematically exact) so it needs
    /// one exp instead of a tanh, `y = C*(g + 0.044715 g³)`.
    ///
    /// # Safety
    /// `gate`/`up` readable and `out` writable for `n` f32; in-place aliasing of
    /// `gate` and `out` is sound (each lane is read before its own store).
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gelu_core(gate: *const f32, up: *const f32, out: *mut f32, n: usize) {
        let one = _mm256_set1_ps(1.0);
        let neg = _mm256_set1_ps(-1.0);
        let c2 = _mm256_set1_ps(1.595_769); // 2*sqrt(2/pi)
        let a = _mm256_set1_ps(0.044715);
        let mut i = 0;
        while i + 8 <= n {
            let g = _mm256_loadu_ps(gate.add(i));
            let u = _mm256_loadu_ps(up.add(i));
            let g2 = _mm256_mul_ps(g, g);
            // poly = g + a*g³ = fma(a*g², g, g)
            let poly = _mm256_fmadd_ps(_mm256_mul_ps(a, g2), g, g);
            let w = _mm256_mul_ps(c2, poly); // 2y
            let s = _mm256_div_ps(one, _mm256_add_ps(one, exp8(_mm256_mul_ps(w, neg))));
            _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_mul_ps(g, s), u));
            i += 8;
        }
        while i < n {
            let g = *gate.add(i);
            let s = 1.0 / (1.0 + (-(1.595_769 * (g + 0.044715 * g * g * g))).exp());
            *out.add(i) = g * s * *up.add(i);
            i += 1;
        }
    }

    /// `out[i] = gelu_tanh(gate[i]) * up[i]` (distinct buffers).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gelu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
        gelu_core(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), gate.len());
    }

    /// In-place `g[i] = gelu_tanh(g[i]) * ple[i]`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gelu_mul_inplace(g: &mut [f32], ple: &[f32]) {
        let n = g.len();
        gelu_core(g.as_ptr(), ple.as_ptr(), g.as_mut_ptr(), n);
    }

    /// `out[i] = x[i] * scale * w[i]`, with `scale = 1/sqrt(mean(x²)+eps)`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
        let n = x.len();
        // sum of squares (2 accumulators to hide FMA latency)
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let a = _mm256_loadu_ps(x.as_ptr().add(i));
            let b = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            acc0 = _mm256_fmadd_ps(a, a, acc0);
            acc1 = _mm256_fmadd_ps(b, b, acc1);
            i += 16;
        }
        let mut ss = hsum(_mm256_add_ps(acc0, acc1));
        while i < n {
            ss += x[i] * x[i];
            i += 1;
        }
        let scale = 1.0 / (ss / n as f32 + eps).sqrt();
        let vs = _mm256_set1_ps(scale);
        let mut i = 0;
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            let wv = _mm256_loadu_ps(w.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_mul_ps(_mm256_mul_ps(xv, vs), wv),
            );
            i += 8;
        }
        while i < n {
            out[i] = x[i] * scale * w[i];
            i += 1;
        }
    }

    /// NeoX-style RoPE in place over `[n_heads, head_dim]`. Rotates each
    /// `(i, i+half)` pair by `(cos[i], sin[i])`: `a' = a.c - b.s`, `b' = a.s + b.c`.
    /// FMA-fused rotation (parity vs scalar < 1e-6). `half >= 8` in practice
    /// (head_dim >= 16), so the 8-wide stores to `[base+i]` and `[base+i+half]`
    /// never overlap.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rope_neox(
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        n_heads: usize,
        head_dim: usize,
    ) {
        let half = head_dim / 2;
        for h in 0..n_heads {
            let base = h * head_dim;
            let mut i = 0;
            while i + 8 <= half {
                let a = _mm256_loadu_ps(x.as_ptr().add(base + i));
                let b = _mm256_loadu_ps(x.as_ptr().add(base + i + half));
                let c = _mm256_loadu_ps(cos.as_ptr().add(i));
                let s = _mm256_loadu_ps(sin.as_ptr().add(i));
                // a' = a.c - b.s = fmsub(a, c, b.s)
                let na = _mm256_fmsub_ps(a, c, _mm256_mul_ps(b, s));
                // b' = a.s + b.c = fmadd(a, s, b.c)
                let nb = _mm256_fmadd_ps(a, s, _mm256_mul_ps(b, c));
                _mm256_storeu_ps(x.as_mut_ptr().add(base + i), na);
                _mm256_storeu_ps(x.as_mut_ptr().add(base + i + half), nb);
                i += 8;
            }
            while i < half {
                let (c, s) = (cos[i], sin[i]);
                let a = x[base + i];
                let b = x[base + i + half];
                x[base + i] = a * c - b * s;
                x[base + i + half] = a * s + b * c;
                i += 1;
            }
        }
    }

    /// Greedy argmax over `v`, first-max-wins tie-break - BIT-IDENTICAL to the
    /// scalar `for (i,&x){ if x>bv {bv=x;best=i} }`. Per-lane running max with a
    /// strictly-greater (`_CMP_GT_OQ`) blend keeps, in each lane, the lowest
    /// index achieving that lane's max; the 8-lane horizontal reduce breaks ties
    /// by lowest index; the tail updates only on strictly-greater. NaN never
    /// updates (ordered compare = scalar `>`), so ties/NaNs match exactly.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn argmax(v: &[f32]) -> u32 {
        let n = v.len();
        if n == 0 {
            return 0;
        }
        let mut maxv = _mm256_set1_ps(f32::NEG_INFINITY);
        // lane index vector, starts [0,1,..,7], += 8 each step
        let mut curidx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let eight = _mm256_set1_epi32(8);
        let mut idxv = curidx;
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_loadu_ps(v.as_ptr().add(i));
            let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(x, maxv); // x > maxv
            maxv = _mm256_blendv_ps(maxv, x, mask);
            idxv = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(idxv),
                _mm256_castsi256_ps(curidx),
                mask,
            ));
            curidx = _mm256_add_epi32(curidx, eight);
            i += 8;
        }
        // spill the 8 lanes and reduce: max value, ties -> lowest stored index.
        let mut vals = [0f32; 8];
        let mut idxs = [0i32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), maxv);
        _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idxv);
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for l in 0..8 {
            // strictly-greater value, OR equal value with a lower index
            if vals[l] > bv || (vals[l] == bv && (idxs[l] as usize) < best) {
                bv = vals[l];
                best = idxs[l] as usize;
            }
        }
        // if no SIMD iterations ran, `best`/`bv` are the -inf sentinel; the tail
        // scalar pass below (from i=0) then reproduces the scalar loop exactly.
        while i < n {
            if v[i] > bv {
                bv = v[i];
                best = i;
            }
            i += 1;
        }
        best as u32
    }
}

/// RMSNorm: `out[i] = x[i] * rsqrt(mean(x²)+eps) * w[i]`. Matches
/// `ops::rms_norm` (mean over the row, eps inside the sqrt). Zero-alloc.
pub fn rms_norm_slice(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(w.len(), n);
    debug_assert_eq!(out.len(), n);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: avx2+fma guaranteed present by the target_feature cfg.
    unsafe {
        simd::rms_norm(x, w, eps, out)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut ss = 0f32;
        for &v in x {
            ss += v * v;
        }
        let scale = 1.0 / (ss / n as f32 + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] * scale * w[i];
        }
    }
}

/// Fused SwiGLU: `out[i] = silu(gate[i]) * up[i]`, `silu(z)=z/(1+e^-z)`. Zero-alloc.
pub fn silu_mul_slice(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len();
    debug_assert_eq!(up.len(), n);
    debug_assert_eq!(out.len(), n);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: avx2+fma guaranteed present by the target_feature cfg.
    unsafe {
        simd::silu_mul(gate, up, out)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for i in 0..n {
        let g = gate[i];
        out[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// GeGLU: `gelu(gate) * up` (gemma's FFN). Uses the tanh GELU approximation to
/// match the tanh-approx GELU the Tensor path uses.
pub fn gelu_mul_slice(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len();
    debug_assert_eq!(up.len(), n);
    debug_assert_eq!(out.len(), n);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: avx2+fma guaranteed present by the target_feature cfg.
    unsafe {
        simd::gelu_mul(gate, up, out)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        const C: f32 = 0.797_884_56; // sqrt(2/pi)
        for i in 0..n {
            let g = gate[i];
            let t = (C * (g + 0.044715 * g * g * g)).tanh();
            out[i] = 0.5 * g * (1.0 + t) * up[i];
        }
    }
}

/// In-place `g[i] = gelu_tanh(g[i]) * ple[i]` - gemma4's PLE gate. Matches the
/// Tensor path's `gate.gelu().mul(ple_input)` (`.gelu()` = tanh approx).
pub fn gelu_mul_inplace(g: &mut [f32], ple: &[f32]) {
    debug_assert_eq!(g.len(), ple.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: avx2+fma guaranteed present by the target_feature cfg.
    unsafe {
        simd::gelu_mul_inplace(g, ple)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        const C: f32 = 0.797_884_56; // sqrt(2/pi)
        for i in 0..g.len() {
            let v = g[i];
            let t = (C * (v + 0.044715 * v * v * v)).tanh();
            g[i] = 0.5 * v * (1.0 + t) * ple[i];
        }
    }
}

/// In-place gemma4 PLE output scale, matching the Tensor path's `broadcast_mul`:
/// a length-1 `s` is a SCALAR broadcast over all of `x` (gemma4's per-layer
/// `out_scale`, shape `{1}`); a length-`x` `s` is per-element. Zero-alloc.
pub fn mul_inplace(x: &mut [f32], s: &[f32]) {
    if s.len() == 1 {
        let sc = s[0];
        for v in x.iter_mut() {
            *v *= sc;
        }
    } else {
        debug_assert_eq!(x.len(), s.len());
        for i in 0..x.len() {
            x[i] *= s[i];
        }
    }
}

/// In-place residual add: `acc[i] += add[i]`. Zero-alloc. Kept as a plain scalar
/// loop: the micro-bench shows this trivial elementwise add already
/// auto-vectorizes (a hand AVX2 version was only ~1.03-1.16x = measurement
/// noise), so it is NOT a missed vectorization lever - unlike silu/gelu (exp)
/// and argmax (data-dependent branch) which the compiler can't vectorize.
pub fn residual_add(acc: &mut [f32], add: &[f32]) {
    debug_assert_eq!(acc.len(), add.len());
    for i in 0..acc.len() {
        acc[i] += add[i];
    }
}

/// Fused softmax numerator + sum: in-place `x[i] = exp(x[i] - max)`, returns Σ.
/// This is the exp-heavy inner loop of the CPU attention softmax scans
/// (cpu_f16_kv / cpu_q8_kv): at long ctx it runs ~seq exps per head per layer
/// per token. AVX2 exp8 (Cephes ~1 ulp) with a fused partial-sum sweep; a scalar
/// tail + non-AVX2 fallback keep it exact-order-identical to the reference so the
/// normalized probabilities match to <1e-5. Caller supplies the running max.
pub fn exp_sub_max_sum(x: &mut [f32], max: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: avx2+fma guaranteed present by the target_feature cfg.
    unsafe {
        simd::exp_sub_max_sum(x, max)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum = 0f32;
        for v in x.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }
        sum
    }
}

/// Apply NeoX-style RoPE in place to a `[n_heads, head_dim]` slice, using
/// precomputed `cos`/`sin` of length `head_dim/2` for the current position.
/// Rotates the (i, i+head_dim/2) pairs - the GGUF/llama convention used by our
/// `ops::rope` (neox). Zero-alloc. Vectorized (AVX2, FMA rotation);
/// parity vs scalar < 1e-6.
pub fn rope_neox_slice(x: &mut [f32], cos: &[f32], sin: &[f32], n_heads: usize, head_dim: usize) {
    let half = head_dim / 2;
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: avx2+fma guaranteed present by the target_feature cfg.
    unsafe {
        simd::rope_neox(x, cos, sin, n_heads, head_dim)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let (c, s) = (cos[i], sin[i]);
            let a = x[base + i];
            let b = x[base + i + half];
            x[base + i] = a * c - b * s;
            x[base + i + half] = a * s + b * c;
        }
    }
}

/// Greedy argmax over `v` (first-max-wins tie-break: lowest index among equal
/// maxima), BIT-IDENTICAL to `for (i,&x){ if x>bv {bv=x;best=i} }`. AVX2 8-lane
/// running-max + index blend, then an 8-way horizontal reduce. Hot: once per
/// decoded token over the full vocab (qwen3 151936, gemma4 262144). NaN never
/// wins (ordered compare), matching the scalar `>`.
pub fn argmax_f32(v: &[f32]) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: avx2+fma guaranteed present by the target_feature cfg.
    unsafe {
        simd::argmax(v)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i;
            }
        }
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sizes chosen NOT multiples of 8 so the vectorized tail loop is exercised.
    #[test]
    fn rms_norm_matches_naive() {
        // 5123 = decode hidden-ish, not a multiple of 8.
        for &n in &[7usize, 128, 5123] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 2.0).collect();
            let w: Vec<f32> = (0..n)
                .map(|i| 1.0 + (i as f32 * 0.007).cos() * 0.1)
                .collect();
            let eps = 1e-5;
            let mut out = vec![0f32; n];
            rms_norm_slice(&x, &w, eps, &mut out);
            let ss: f32 = x.iter().map(|v| v * v).sum();
            let scale = 1.0 / (ss / n as f32 + eps).sqrt();
            let mut maxd = 0f32;
            for i in 0..n {
                maxd = maxd.max((out[i] - x[i] * scale * w[i]).abs());
            }
            // scalar-vs-vectorized reduction order differ slightly; well under 1e-3.
            assert!(maxd < 1e-4, "rms_norm n={n} maxdiff={maxd:e}");
        }
    }

    // Softmax numerator (exp(x-max)+sum) must match the scalar reference on the
    // NORMALIZED probabilities to <1e-5 at real attention shapes, incl. seq=2500
    // (the long-ctx cliff) and non-mult-of-8 tails. exp8 is ~1 ulp so the maxdiff
    // is dominated by the sum-reduction order - still far under tolerance.
    #[test]
    fn exp_sub_max_sum_matches_naive() {
        for &n in &[1usize, 7, 8, 63, 512, 2500] {
            // scores spanning a wide range so max-subtraction + exp are stressed.
            let scores: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.031).sin() * 6.0) + (i as f32 * 0.002))
                .collect();
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // scalar reference: probabilities
            let ref_exp: Vec<f32> = scores.iter().map(|x| (x - mx).exp()).collect();
            let ref_sum: f32 = ref_exp.iter().sum();
            let ref_p: Vec<f32> = ref_exp.iter().map(|e| e / ref_sum).collect();

            // vectorized: in-place exp + fused sum
            let mut p = scores.clone();
            let sum = exp_sub_max_sum(&mut p, mx);
            let inv = 1.0 / sum;
            let mut maxd = 0f32;
            for i in 0..n {
                maxd = maxd.max((p[i] * inv - ref_p[i]).abs());
            }
            assert!(maxd < 1e-5, "exp_sub_max_sum n={n} prob maxdiff={maxd:e}");
            // sum itself close (relative) so downstream 1/sum matches.
            assert!(
                (sum - ref_sum).abs() / ref_sum < 1e-5,
                "exp_sub_max_sum n={n} sum {sum} vs {ref_sum}"
            );
        }
    }

    #[test]
    fn silu_mul_matches_naive() {
        // wide input range incl. large |g| to stress the exp approximation; tail.
        for &n in &[5usize, 64, 14337] {
            let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin() * 12.0).collect();
            let u: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
            let mut out = vec![0f32; n];
            silu_mul_slice(&g, &u, &mut out);
            let mut maxd = 0f32;
            for i in 0..n {
                let s = g[i] / (1.0 + (-g[i]).exp());
                maxd = maxd.max((out[i] - s * u[i]).abs());
            }
            assert!(maxd < 1e-4, "silu_mul n={n} maxdiff={maxd:e}");
        }
    }

    #[test]
    fn gelu_mul_matches_naive() {
        const C: f32 = 0.797_884_56;
        for &n in &[5usize, 64, 14337] {
            let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin() * 8.0).collect();
            let u: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
            let mut out = vec![0f32; n];
            gelu_mul_slice(&g, &u, &mut out);
            // in-place variant must match the distinct one.
            let mut gi = g.clone();
            gelu_mul_inplace(&mut gi, &u);
            let mut maxd = 0f32;
            let mut maxd_inplace = 0f32;
            for i in 0..n {
                let t = (C * (g[i] + 0.044715 * g[i] * g[i] * g[i])).tanh();
                let expect = 0.5 * g[i] * (1.0 + t) * u[i];
                maxd = maxd.max((out[i] - expect).abs());
                maxd_inplace = maxd_inplace.max((gi[i] - out[i]).abs());
            }
            assert!(maxd < 1e-3, "gelu_mul n={n} maxdiff={maxd:e}");
            assert!(
                maxd_inplace < 1e-6,
                "gelu_mul_inplace n={n} diff={maxd_inplace:e}"
            );
        }
    }

    #[test]
    fn rope_neox_is_a_rotation() {
        // A rotation preserves the norm of each (i, i+half) pair.
        let hd = 8usize;
        let mut x: Vec<f32> = (0..hd).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let orig = x.clone();
        let cos: Vec<f32> = (0..hd / 2).map(|i| (i as f32 * 0.2).cos()).collect();
        let sin: Vec<f32> = (0..hd / 2).map(|i| (i as f32 * 0.2).sin()).collect();
        rope_neox_slice(&mut x, &cos, &sin, 1, hd);
        for i in 0..hd / 2 {
            let n0 = orig[i] * orig[i] + orig[i + hd / 2] * orig[i + hd / 2];
            let n1 = x[i] * x[i] + x[i + hd / 2] * x[i + hd / 2];
            assert!((n0 - n1).abs() < 1e-5, "rope must preserve pair norm");
        }
    }

    // Scalar reference for RoPE (the non-AVX2 fallback), used to check parity.
    fn rope_scalar(x: &mut [f32], cos: &[f32], sin: &[f32], n_heads: usize, head_dim: usize) {
        let half = head_dim / 2;
        for h in 0..n_heads {
            let base = h * head_dim;
            for i in 0..half {
                let (c, s) = (cos[i], sin[i]);
                let a = x[base + i];
                let b = x[base + i + half];
                x[base + i] = a * c - b * s;
                x[base + i + half] = a * s + b * c;
            }
        }
    }

    #[test]
    fn rope_neox_matches_scalar() {
        // real decode shapes + a non-8-multiple half to exercise the tail.
        for &(nh, hd) in &[(32usize, 128usize), (8, 64), (1, 96), (4, 20)] {
            let half = hd / 2;
            let mut xv: Vec<f32> = (0..nh * hd)
                .map(|i| (i as f32 * 0.011).sin() * 3.0)
                .collect();
            let mut xs = xv.clone();
            let cos: Vec<f32> = (0..half).map(|i| (i as f32 * 0.03).cos()).collect();
            let sin: Vec<f32> = (0..half).map(|i| (i as f32 * 0.03).sin()).collect();
            rope_neox_slice(&mut xv, &cos, &sin, nh, hd);
            rope_scalar(&mut xs, &cos, &sin, nh, hd);
            let md = xv
                .iter()
                .zip(&xs)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(md < 1e-5, "rope nh={nh} hd={hd} maxdiff={md:e}");
        }
    }

    fn argmax_scalar_ref(v: &[f32]) -> u32 {
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i;
            }
        }
        best as u32
    }

    #[test]
    fn argmax_bit_identical_to_scalar() {
        // 1) crafted tie inputs: first (lowest-index) max must win.
        let tie = vec![1.0f32, 3.0, 2.0, 3.0, 3.0, 0.0, 3.0]; // max 3.0 first at idx 1
        assert_eq!(argmax_f32(&tie), 1);
        assert_eq!(argmax_f32(&tie), argmax_scalar_ref(&tie));

        // tie that straddles an 8-lane boundary (idx 3 and idx 11 both = 9.0).
        let mut straddle = vec![0.0f32; 20];
        straddle[3] = 9.0;
        straddle[11] = 9.0;
        straddle[17] = 9.0;
        assert_eq!(argmax_f32(&straddle), 3);
        assert_eq!(argmax_f32(&straddle), argmax_scalar_ref(&straddle));

        // 2) max is in the tail (n not a multiple of 8).
        let mut tail = vec![-1.0f32; 19];
        tail[18] = 5.0;
        assert_eq!(argmax_f32(&tail), 18);
        assert_eq!(argmax_f32(&tail), argmax_scalar_ref(&tail));

        // 3) all-equal -> index 0.
        let eqv = vec![2.5f32; 100];
        assert_eq!(argmax_f32(&eqv), 0);

        // 4) NaN never wins (ordered compare matches scalar `>`).
        let mut withnan = vec![1.0f32, 2.0, f32::NAN, 4.0, f32::NAN, 3.0, 4.0, 0.0, 4.0];
        assert_eq!(argmax_f32(&withnan), argmax_scalar_ref(&withnan)); // 4.0 first at idx 3
        assert_eq!(argmax_f32(&withnan), 3);
        withnan[3] = -10.0;
        assert_eq!(argmax_f32(&withnan), argmax_scalar_ref(&withnan));

        // 5) all -inf -> index 0 (matches scalar sentinel).
        let ninf = vec![f32::NEG_INFINITY; 40];
        assert_eq!(argmax_f32(&ninf), 0);

        // 6) short slices (< 8, no SIMD iteration).
        for k in 0..8usize {
            let s: Vec<f32> = (0..k).map(|i| (i as f32 * 0.7).sin()).collect();
            assert_eq!(argmax_f32(&s), argmax_scalar_ref(&s), "short k={k}");
        }

        // 7) fuzz over vocab-sized random data - exhaustive bit-identity.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for &n in &[151936usize, 262144, 4096, 33] {
            let data: Vec<f32> = (0..n)
                .map(|_| ((rng() >> 40) as f32 / 16_777_216.0 - 0.5) * 20.0)
                .collect();
            assert_eq!(argmax_f32(&data), argmax_scalar_ref(&data), "fuzz n={n}");
        }
    }
}
