//! Qwen3-VL-8B text encoder (text tower only) - produces the `[seq, 4096]` conditioning
//! hidden states for the Boogu-Image DiT. The Qwen3-VL text tower IS a Qwen3-8B dense decoder
//! (verified against Qwen3-8B config.json + the checkpoint header): 36 layers, hidden 4096,
//! 32 Q / 8 KV heads (GQA n_rep=4), head_dim 128, SwiGLU ffn 12288, **QK-RMSNorm** (the Qwen3
//! addition vs Qwen2.5), rope theta 1e6, RMSNorm eps 1e-6.
//!
//! Weights are fp8_scaled safetensors (big matmuls fp8 e4m3 x per-tensor `weight_scale`,
//! norms/embeddings bf16). The projection weights are re-block-quantized to Q8_0 and kept
//! ~1 byte resident on-device (via `fp8_scaled::load_qvarbuilder` -> QVarBuilder/QKernelMatMul), so the
//! ~33 GB F32 tower fits a single GPU at ~9 GB. Text-encoder activations are normal-magnitude, so
//! the standard quantized matmul (MMQ/MMVQ) is used directly. The vision tower is ignored
//! (text-to-image conditions on the caption only).

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::quantized::{GgmlDType, QKernelMatMul, QVarBuilder};
use crate::tensor::{Device, DeviceLocation, Error, Result, Tensor as NT};
use std::collections::HashMap;

/// Flatten a `HeteroPlan` into one `Device` per layer (the plan owns placement - NEVER hardcode a
/// device). `DeviceKind::Cuda(i)` resolves via the probed `cuda_devices` map; everything else (CPU,
/// and OpenCL which these models don't plan for) lands on CPU.
fn plan_layer_devices(plan: &HeteroPlan, cuda_devices: &HashMap<usize, Device>) -> Vec<Device> {
    let mut out = Vec::with_capacity(plan.total_layers);
    for seg in &plan.segments {
        let dev = match seg.kind {
            DeviceKind::Cuda(i) => cuda_devices.get(&i).cloned().unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        };
        for _ in seg.layer_start..seg.layer_end {
            out.push(dev.clone());
        }
    }
    out
}

/// Config for the Qwen3-8B text tower.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub hidden: usize,   // 4096
    pub n_layers: usize, // 36
    pub n_head: usize,   // 32
    pub n_kv: usize,     // 8
    pub head_dim: usize, // 128
    pub ffn: usize,      // 12288
    pub theta: f32,      // 1e6
    pub eps: f32,        // 1e-6
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hidden: 4096,
            n_layers: 36,
            n_head: 32,
            n_kv: 8,
            head_dim: 128,
            ffn: 12288,
            theta: 1_000_000.0,
            eps: 1e-6,
        }
    }
}

impl Config {
    /// Qwen3-4B - the text encoder FLUX.2 Klein conditions on. The SAME decoder as the 8B tower
    /// above (36 layers, 32 Q / 8 KV, head_dim 128, QK-RMSNorm, rope 1e6), only narrower, so it
    /// loads and runs through this module unchanged.
    pub fn qwen3_4b() -> Self {
        Config {
            hidden: 2560,
            n_layers: 36,
            n_head: 32,
            n_kv: 8,
            head_dim: 128,
            ffn: 9728,
            theta: 1_000_000.0,
            eps: 1e-6,
        }
    }
}

/// Quantized linear (no bias - Qwen3 attn/mlp projections are bias-free). The Q8_0 weight
/// (`[out, in]`) is kept ~1 byte resident on-device; the matmul dequantizes on the fly.
struct QLin {
    qm: QKernelMatMul,
}

impl QLin {
    fn load(vb: &QVarBuilder, dev: &Device, name: &str) -> Result<Self> {
        Ok(QLin {
            qm: vb.qmatmul_auto(&format!("{name}.weight"), dev)?,
        })
    }
    fn forward(&self, x: &NT) -> Result<NT> {
        // x [s, in] @ w [in, out] -> [s, out]. Normal-magnitude activations -> the standard
        // quantized matmul (MMQ prefill / MMVQ decode, and the CPU quantized path) is exact enough.
        self.qm.forward(x)
    }
}

struct Layer {
    /// Device this layer's weights (and its slice of the forward) live on - assigned by the plan.
    device: Device,
    input_norm: NT, // RMSNorm gamma [hidden]
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    q_norm: NT, // QK-norm gamma [head_dim]
    k_norm: NT,
    post_norm: NT, // RMSNorm gamma [hidden]
    gate: QLin,
    up: QLin,
    down: QLin,
}

pub struct Qwen3VlTextEncoder {
    cfg: Config,
    /// First layer's device (embed_tokens lives here; the forward starts here).
    input_device: Device,
    /// Last layer's device (final_norm lives here; the forward output ends here).
    embed_tokens: NT, // [vocab, hidden] F32, on input_device
    layers: Vec<Layer>,
    final_norm: NT, // model.norm gamma [hidden], on output_device
}

impl Qwen3VlTextEncoder {
    pub fn dim(&self) -> usize {
        self.cfg.hidden
    }

    /// Device the forward starts on (where `embed_tokens` lives), so a caller can report the
    /// placement the plan actually chose rather than the one it asked for.
    pub fn input_device(&self) -> &Device {
        &self.input_device
    }

    /// Load the fp8_scaled Qwen3-VL text tower, placing each layer on the device the `plan` assigns
    /// (adaptive multi-GPU+CPU, fastest-first, spill to next GPU/CPU - NO hardcoded device). The
    /// QVarBuilder is parsed once to CPU; its QHostTensor blobs are device-independent, so
    /// `qmatmul_auto(.., dev)` / `get_f32_auto_on(.., dev)` place each layer's weights per device.
    pub fn load(
        path: &str,
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
    ) -> Result<Self> {
        Self::load_cancellable(path, cuda_devices, plan, None)
    }

    /// [`Self::load`] with cooperative cancellation: checked per tensor during the fp8 decode
    /// and per block during assembly, so an abandoned load stops in seconds.
    pub fn load_cancellable(
        path: &str,
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Self> {
        Self::load_cfg(&[path], Config::default(), cuda_devices, plan, cancel)
    }

    /// [`Self::load_cancellable`] over an explicit config and a SHARDED checkpoint.
    ///
    /// Same decoder, different width and file layout: FLUX.2 Klein conditions on Qwen3-4B, which
    /// ships as two diffusers shards rather than one fp8 file. Neither the walk nor the forward
    /// changes, so the only thing that had to become a parameter is the config.
    pub fn load_cfg(
        paths: &[&str],
        cfg: Config,
        cuda_devices: &HashMap<usize, Device>,
        plan: &HeteroPlan,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Self> {
        // SAFETY: `load_qvarbuilder` copies every tensor out of the mmap before returning.
        let vb = unsafe {
            crate::inference::load::fp8_scaled::load_qvarbuilder_cancellable(
                paths,
                GgmlDType::Q8_0,
                &Device::Cpu,
                cancel,
            )
        }?;
        let layer_dev = plan_layer_devices(plan, cuda_devices);
        if layer_dev.len() != cfg.n_layers {
            return Err(Error(format!(
                "qwen3vl: plan has {} layers, expected {}",
                layer_dev.len(),
                cfg.n_layers
            )));
        }
        let input_device = layer_dev.first().cloned().unwrap_or(Device::Cpu);
        let output_device = layer_dev.last().cloned().unwrap_or(Device::Cpu);
        // fp8_scaled::canonical_name strips the `model.` wrapper prefix, so the QVarBuilder keys
        // are `layers.N.*` / `embed_tokens.weight` / `norm.weight` (no `model.`).
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if let Some(c) = cancel {
                c.bail()?;
            }
            let dev = &layer_dev[i];
            let g = |name: &str| -> Result<NT> { vb.get_f32_auto_on(name, dev) };
            let p = format!("layers.{i}");
            layers.push(Layer {
                device: dev.clone(),
                input_norm: g(&format!("{p}.input_layernorm.weight"))?,
                q: QLin::load(&vb, dev, &format!("{p}.self_attn.q_proj"))?,
                k: QLin::load(&vb, dev, &format!("{p}.self_attn.k_proj"))?,
                v: QLin::load(&vb, dev, &format!("{p}.self_attn.v_proj"))?,
                o: QLin::load(&vb, dev, &format!("{p}.self_attn.o_proj"))?,
                q_norm: g(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: g(&format!("{p}.self_attn.k_norm.weight"))?,
                post_norm: g(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: QLin::load(&vb, dev, &format!("{p}.mlp.gate_proj"))?,
                up: QLin::load(&vb, dev, &format!("{p}.mlp.up_proj"))?,
                down: QLin::load(&vb, dev, &format!("{p}.mlp.down_proj"))?,
            });
        }
        Ok(Qwen3VlTextEncoder {
            embed_tokens: vb.get_f32_auto_on("embed_tokens.weight", &input_device)?,
            final_norm: vb.get_f32_auto_on("norm.weight", &output_device)?,
            layers,
            cfg,
            input_device,
        })
    }

    /// 1-D rope cos/sin `[s, head_dim/2]` on CPU (the forward moves it to each layer's device). For
    /// a text-only sequence the Qwen3-VL M-RoPE axes all carry the same sequential position, so it
    /// reduces to standard 1-D rope.
    fn rope_cos_sin(&self, s: usize) -> Result<(NT, NT)> {
        let (d, theta) = (self.cfg.head_dim, self.cfg.theta);
        let half = d / 2;
        let inv = crate::inference::model::rope::inverse_frequencies(d, theta);
        let (mut cos, mut sin) = (vec![0f32; s * half], vec![0f32; s * half]);
        for i in 0..s {
            for j in 0..half {
                let a = i as f32 * inv[j];
                cos[i * half + j] = a.cos();
                sin[i * half + j] = a.sin();
            }
        }
        Ok((
            NT::from_vec_f32(cos, (s, half))?,
            NT::from_vec_f32(sin, (s, half))?,
        ))
    }

    /// `x [s, hidden]`, returns `[s, hidden]` after the causal self-attention block. `mask`, when
    /// present, REPLACES the built-in causal fill and must already encode it (the kernel treats
    /// `do_causal` and an explicit mask as alternatives, not as things it intersects).
    fn self_attn(
        &self,
        l: &Layer,
        x: &NT,
        s: usize,
        cos: &NT,
        sin: &NT,
        mask: Option<&NT>,
    ) -> Result<NT> {
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (nh, nkv, hd, eps) = (
            self.cfg.n_head,
            self.cfg.n_kv,
            self.cfg.head_dim,
            self.cfg.eps,
        );
        // Project, reshape to [s, heads, hd], QK-RMSNorm over hd, then rope on [1, heads, s, hd].
        let roped = |lin: &QLin, norm: Option<&NT>, heads: usize| -> Result<NT> {
            let mut t = lin.forward(x)?.reshape((s, heads, hd))?;
            if let Some(g) = norm {
                t = t.rms_norm(g, eps)?; // per-head RMSNorm over head_dim
            }
            let t = t.transpose(0, 1)?.unsqueeze(0)?.contiguous()?; // [1, heads, s, hd]
            match norm {
                Some(_) => t.rope(cos, sin),
                None => Ok(t),
            }
        };
        let q = roped(&l.q, Some(&l.q_norm), nh)?;
        let k = roped(&l.k, Some(&l.k_norm), nkv)?;
        let v = roped(&l.v, None, nkv)?;
        let n_rep = nh / nkv;
        let (k, v) = (repeat_kv(k, n_rep)?, repeat_kv(v, n_rep)?);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let attn = match mask {
            Some(m) => sdpa(&q, &k, &v, Some(m), false, scale, 1.0)?,
            None => sdpa(&q, &k, &v, None, true, scale, 1.0)?, // causal
        };
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * hd))?;
        l.o.forward(&ao)
    }

    fn layer(
        &self,
        l: &Layer,
        x: &NT,
        s: usize,
        cos: &NT,
        sin: &NT,
        mask: Option<&NT>,
    ) -> Result<NT> {
        let eps = self.cfg.eps;
        let h = x.rms_norm(&l.input_norm, eps)?;
        let x = x.add(&self.self_attn(l, &h, s, cos, sin, mask)?)?;
        let h2 = x.rms_norm(&l.post_norm, eps)?;
        let ff = l.gate.forward(&h2)?.silu()?.mul(&l.up.forward(&h2)?)?;
        x.add(&l.down.forward(&ff)?)
    }

    /// Run the token ids through the tower -> `[seq, hidden]` (post final RMSNorm). Layers may sit on
    /// different devices (per the plan): the hidden state hops device at each boundary via the
    /// sync -> to_device -> sync pattern, and the rope is pre-materialized on each distinct device.
    /// When every layer shares one device (the fits-one-GPU case) NO hops happen -> identical to the
    /// single-device path.
    pub fn forward(&self, ids: &[u32]) -> Result<NT> {
        let (x, _) = self.run(ids, &[], None)?;
        // x is on output_device (last layer); final_norm lives there too.
        x.rms_norm(&self.final_norm, self.cfg.eps)
    }

    /// Hidden states after selected blocks, using HuggingFace's `output_hidden_states` indexing:
    /// `k` means `hidden_states[k]`, the output of the k-th block (index 0 would be the embedding
    /// output, before any block).
    ///
    /// Returned BEFORE the final RMSNorm, because that is what HF exposes for INTERMEDIATE
    /// layers - the norm is applied only to the last hidden state. A consumer that conditions on
    /// mid-stack layers (FLUX.2 Klein takes 9, 18 and 27) would otherwise get normalised tensors
    /// where the reference has raw residual-stream ones, which rescales the whole conditioning.
    ///
    /// Each tap comes back on the INPUT device, so a caller can concatenate them whatever the
    /// plan did with the layers in between.
    pub fn forward_taps(&self, ids: &[u32], taps: &[usize]) -> Result<Vec<NT>> {
        Ok(self.run(ids, taps, None)?.1)
    }

    /// [`Self::forward_taps`] for a RIGHT-PADDED batch: `n_real` says how many leading ids are
    /// real, and everything at or beyond it is padding that no position may attend to.
    ///
    /// This is not cosmetic. Causal attention already keeps a real token from seeing a pad, so
    /// the real positions are identical either way - but the PAD positions are not, and a
    /// consumer that conditions on the whole padded sequence (FLUX.2 Klein takes all 512) feeds
    /// them to its DiT. Measured against the reference, leaving them unmasked put the real
    /// tokens at corr 0.999994 and the padded tail at 0.05.
    pub fn forward_taps_padded(
        &self,
        ids: &[u32],
        taps: &[usize],
        n_real: usize,
    ) -> Result<Vec<NT>> {
        Ok(self.run(ids, taps, Some(n_real))?.1)
    }

    /// The shared layer loop. `taps` are 1-based block indices (see `forward_taps`); `n_real`,
    /// when set, marks the end of the real tokens in a right-padded sequence.
    fn run(&self, ids: &[u32], taps: &[usize], n_real: Option<usize>) -> Result<(NT, Vec<NT>)> {
        let s = ids.len();
        let idx = NT::from_vec_u32(ids.to_vec(), (s,))?.to_device(&self.input_device)?;
        let mut x = self.embed_tokens.index_select(&idx, 0)?; // [s, hidden] on input_device
        let (cos_cpu, sin_cpu) = self.rope_cos_sin(s)?;
        // Pre-materialize the rope on every distinct device the plan uses.
        let mut rope: HashMap<DeviceLocation, (NT, NT)> = HashMap::new();
        for l in &self.layers {
            let loc = l.device.location();
            if let std::collections::hash_map::Entry::Vacant(e) = rope.entry(loc) {
                e.insert((cos_cpu.to_device(&l.device)?, sin_cpu.to_device(&l.device)?));
            }
        }
        // One mask carrying BOTH conditions: a key is visible when it is not in the future AND
        // not padding. The kernel picks either its built-in causal fill or an explicit mask, so
        // the two cannot be combined by passing both.
        let mask: Option<HashMap<DeviceLocation, NT>> = match n_real {
            Some(n) if n < s => {
                let mut d = vec![0f32; s * s];
                for q in 0..s {
                    for k in 0..s {
                        if k > q || k >= n {
                            d[q * s + k] = f32::NEG_INFINITY;
                        }
                    }
                }
                let cpu = NT::from_vec_f32(d, (s, s))?;
                let mut per_dev = HashMap::new();
                for l in &self.layers {
                    let loc = l.device.location();
                    if let std::collections::hash_map::Entry::Vacant(e) = per_dev.entry(loc) {
                        e.insert(cpu.to_device(&l.device)?);
                    }
                }
                Some(per_dev)
            }
            _ => None,
        };
        let mut out = Vec::with_capacity(taps.len());
        for (i, l) in self.layers.iter().enumerate() {
            if x.device().location() != l.device.location() {
                x.device().synchronize()?;
                x = x.to_device(&l.device)?;
                l.device.synchronize()?;
            }
            let (cos, sin) = &rope[&l.device.location()];
            let m = mask.as_ref().map(|per_dev| &per_dev[&l.device.location()]);
            x = self.layer(l, &x, s, cos, sin, m)?;
            if taps.contains(&(i + 1)) {
                out.push(x.to_device(&self.input_device)?);
            }
        }
        Ok((x, out))
    }

    /// Encode a text prompt -> `[seq', hidden]` conditioning, dropping the fixed template prefix
    /// (`drop_idx` tokens).
    pub fn encode_text(
        &self,
        tok: &tokenizers::Tokenizer,
        text: &str,
        drop_idx: usize,
    ) -> Result<NT> {
        let ids: Vec<u32> = tok
            .encode(text, true)
            .map_err(|e| Error(format!("tokenize: {e}")))?
            .get_ids()
            .to_vec();
        let hid = self.forward(&ids)?;
        let s = ids.len();
        let keep = s.saturating_sub(drop_idx).max(1);
        hid.narrow(0, s - keep, keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Loads the ~10GB fp8 encoder and runs one forward. Ignored (heavy, needs local weights):
    //   cargo test -p loken --profile fast native_qwen3vl_textenc -- --ignored --nocapture
    #[test]
    #[ignore]
    fn qwen3vl_one_forward() {
        let path = format!(
            "{}/boogu/text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
            crate::inference::cache::hf::models_dir()
        );
        // Plan across whatever CUDA devices are present (fastest-first, spill to CPU); the plan
        // decides placement - the test never hardcodes a device.
        let mut cuda_devices = HashMap::new();
        let cuda_list: Vec<(usize, u64)> =
            crate::inference::place::device_probe::probe_cuda_devices(0)
                .into_iter()
                .map(|(i, f, d)| {
                    cuda_devices.insert(i, d);
                    (i, f)
                })
                .collect();
        let sz = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let plan = HeteroPlan::calculate(Config::default().n_layers, sz, &cuda_list, &[], 1.0);
        println!("qwen3vl encoder smoke test, plan: {:?}", plan.segments);
        let enc =
            Qwen3VlTextEncoder::load(&path, &cuda_devices, &plan).expect("load qwen3vl encoder");
        let ids: Vec<u32> = vec![9707, 1879, 11, 419, 374, 264, 1273]; // arbitrary token ids
        let out = enc.forward(&ids).expect("forward");
        assert_eq!(out.shape().dims(), &[ids.len(), 4096]);
        let v = out.to_vec_f32();
        assert!(
            v.iter().all(|z| z.is_finite()),
            "encoder output has non-finite values"
        );
        println!("qwen3vl encoder forward ok: out[0..4]={:?}", &v[..4]);
    }
}
