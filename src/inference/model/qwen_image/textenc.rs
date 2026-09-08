//! Qwen2.5-VL text encoder - produces the `encoder_hidden_states [S, 3584]` that condition the
//! Qwen-Image / Qwen-Image-Edit DiT. The VL model's TEXT backbone is a Qwen2.5 transformer
//! (arch `qwen2vl`): 28 causal layers, hidden 3584, 28 q / 4 kv heads, head_dim 128, ffn 18944,
//! SwiGLU, RMSNorm eps 1e-6, RoPE theta 1e6, **qkv-bias**, **no qk-norm** (unlike Qwen3).
//!
//! Weights stay QUANTIZED on-device via qmatmul (QVarBuilder::from_gguf + qmatmul_on) - a 7B Q4
//! is ~4.7GB and fits a 16GB GPU; the ACE-Step detok_* helpers would dequantize (~14GB -> OOM).
//! Small tensors (biases, norms, the token-embed table) are dequantized to F32 via QTensor.
//!
//! For text-only conditioning the qwen2vl M-RoPE reduces to standard 1D RoPE. The pipeline takes
//! the LAST hidden layer (post final norm) and drops the template prefix downstream.

use crate::tensor::quantized::QVarBuilder;
use crate::tensor::{Device, Tensor};

/// `y = qm(x) (+ bias)` - weight kept quantized on-device.
use crate::tensor::layer::qlinear::{QLinear, Weight};

struct Qwen2Layer {
    /// Device this layer's weights + its slice of the forward live on (assigned by the HeteroPlan).
    device: Device,
    attn_norm: Tensor,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    ffn_norm: Tensor,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

pub struct Qwen2TextEncoder {
    embed_tokens: Tensor, // [V, 3584] F32 (dequantized for row-gather), on input_device
    layers: Vec<Qwen2Layer>,
    norm: Tensor, // [3584], on output_device
    hidden: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    rope_theta: f32,
    /// First layer's device (embed_tokens + the forward start here).
    device: Device,
    /// Last layer's device (final norm + the forward output land here).
    output_device: Device,
}

/// Flatten a `HeteroPlan` into one `Device` per layer (the plan owns placement - NEVER hardcode a
/// device). `DeviceKind::Cuda(i)` resolves via the probed `cuda_devices` map; CPU/OpenCL -> CPU.
fn plan_layer_devices(
    plan: &crate::inference::place::layer_executor::HeteroPlan,
    cuda_devices: &std::collections::HashMap<usize, Device>,
) -> Vec<Device> {
    use crate::inference::place::layer_executor::DeviceKind;
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

/// Dequantize a small GGUF tensor (norm/bias/embed) to a native F32 tensor on `device`.
fn dq_t(
    content: &crate::tensor::quantized::gguf_file::Content,
    file: &mut std::fs::File,
    device: &Device,
    name: &str,
) -> crate::tensor::Result<Tensor> {
    use crate::tensor::{DType as CDType, Device as CDevice};
    let dq = content
        .tensor(file, name, &CDevice::Cpu)
        .and_then(|t| t.dequantize(&CDevice::Cpu))
        .and_then(|t| t.to_dtype(CDType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| crate::tensor::Error(format!("dq_t {name}: {e}")))?;
    let dims = content
        .tensor_infos
        .get(name)
        .map(|i| i.shape.dims().to_vec())
        .unwrap_or_else(|| vec![dq.len()]);
    Tensor::from_vec_f32(dq, dims)?.to_device(device)
}

impl Qwen2TextEncoder {
    pub fn dim(&self) -> usize {
        self.hidden
    }

    /// Load a `qwen2vl` GGUF (llama.cpp names) with quantized linears on `device`. The vision tower
    /// (if present) is ignored - only the text `blk.{i}.*` layers + token_embd + output_norm.
    pub fn from_gguf_hetero(
        path: &str,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
    ) -> crate::tensor::Result<Self> {
        use crate::tensor::quantized::gguf_file;
        let mut f = std::fs::File::open(path)?;
        let content = gguf_file::read_mapped_file(&f)?;
        // Parse once; QHostTensor blobs are device-independent -> qmatmul_on/dq_t place per layer.
        let vb = crate::inference::cache::qvb::from_gguf_cached(path, &Device::Cpu)?;
        let (hidden, nh, nkv, hd) = (3584usize, 28usize, 4usize, 128usize);
        let layer_dev = plan_layer_devices(plan, cuda_devices);
        if layer_dev.len() != 28 {
            return Err(crate::tensor::Error(format!(
                "qwen2 encoder: plan has {} layers, expected 28",
                layer_dev.len()
            )));
        }
        let input_device = layer_dev.first().cloned().unwrap_or(Device::Cpu);
        let output_device = layer_dev.last().cloned().unwrap_or(Device::Cpu);
        let ql = |vb: &QVarBuilder,
                  name: &str,
                  ind: usize,
                  outd: usize,
                  bias: bool,
                  file: &mut std::fs::File,
                  dev: &Device|
         -> crate::tensor::Result<QLinear> {
            let qm = vb.qmatmul_on(ind, outd, &format!("{name}.weight"), dev)?;
            let b = if bias {
                dq_t(&content, file, dev, &format!("{name}.bias")).ok()
            } else {
                None
            };
            Ok(QLinear::new(Weight::Quant(qm), b, ind, outd))
        };
        let mut layers = Vec::with_capacity(28);
        for l in 0..28usize {
            let dev = &layer_dev[l];
            let p = format!("blk.{l}");
            layers.push(Qwen2Layer {
                device: dev.clone(),
                attn_norm: dq_t(&content, &mut f, dev, &format!("{p}.attn_norm.weight"))?,
                q: ql(
                    &vb,
                    &format!("{p}.attn_q"),
                    hidden,
                    nh * hd,
                    true,
                    &mut f,
                    dev,
                )?,
                k: ql(
                    &vb,
                    &format!("{p}.attn_k"),
                    hidden,
                    nkv * hd,
                    true,
                    &mut f,
                    dev,
                )?,
                v: ql(
                    &vb,
                    &format!("{p}.attn_v"),
                    hidden,
                    nkv * hd,
                    true,
                    &mut f,
                    dev,
                )?,
                o: ql(
                    &vb,
                    &format!("{p}.attn_output"),
                    nh * hd,
                    hidden,
                    false,
                    &mut f,
                    dev,
                )?,
                ffn_norm: dq_t(&content, &mut f, dev, &format!("{p}.ffn_norm.weight"))?,
                gate: ql(
                    &vb,
                    &format!("{p}.ffn_gate"),
                    hidden,
                    18944,
                    false,
                    &mut f,
                    dev,
                )?,
                up: ql(
                    &vb,
                    &format!("{p}.ffn_up"),
                    hidden,
                    18944,
                    false,
                    &mut f,
                    dev,
                )?,
                down: ql(
                    &vb,
                    &format!("{p}.ffn_down"),
                    18944,
                    hidden,
                    false,
                    &mut f,
                    dev,
                )?,
            });
        }
        let embed_tokens = dq_t(&content, &mut f, &input_device, "token_embd.weight")?;
        let norm = dq_t(&content, &mut f, &output_device, "output_norm.weight")?;
        Ok(Qwen2TextEncoder {
            embed_tokens,
            layers,
            norm,
            hidden,
            n_head: nh,
            n_kv: nkv,
            head_dim: hd,
            rope_theta: 1e6,
            device: input_device,
            output_device,
        })
    }

    /// Single-device compat wrapper: all 28 layers on `device`.
    pub fn from_gguf(path: &str, device: &Device) -> crate::tensor::Result<Self> {
        let mut map = std::collections::HashMap::new();
        let plan = match device.location() {
            crate::tensor::DeviceLocation::Cuda { gpu_id } => {
                map.insert(gpu_id, device.clone());
                crate::inference::place::layer_executor::HeteroPlan::forced_gpu(28, 28, gpu_id)
            }
            _ => {
                crate::inference::place::layer_executor::HeteroPlan::calculate(28, 0, &[], &[], 1.0)
            }
        };
        Self::from_gguf_hetero(path, &map, &plan)
    }

    /// Device the encoder consumes its input on (the first layer's device).
    pub fn input_device(&self) -> &Device {
        &self.device
    }
    /// Device the encoder emits its output on (the last layer's device, where the final norm lives).
    pub fn output_device(&self) -> &Device {
        &self.output_device
    }

    fn rms_eps(&self) -> f32 {
        1e-6
    }

    fn qkv_roped(
        &self,
        lin: &QLinear,
        x: &Tensor,
        n_heads: usize,
        s: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> crate::tensor::Result<Tensor> {
        let d = self.head_dim;
        let proj = lin.forward(x)?; // WITH bias for q/k/v
        let q = proj
            .reshape((s, n_heads, d))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        q.rope(cos, sin) // rotate-half rope with the precomputed (M-RoPE) cos/sin [s, d/2]
    }

    fn self_attn(
        &self,
        l: &Qwen2Layer,
        x: &Tensor,
        s: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> crate::tensor::Result<Tensor> {
        use crate::inference::model::acestep::ops::{repeat_kv, sdpa};
        let (nh, nkv, d) = (self.n_head, self.n_kv, self.head_dim);
        let q = self.qkv_roped(&l.q, x, nh, s, cos, sin)?;
        let k = self.qkv_roped(&l.k, x, nkv, s, cos, sin)?;
        let v =
            l.v.forward(x)?
                .reshape((s, nkv, d))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
        let nrep = nh / nkv;
        let (k, v) = (repeat_kv(k, nrep)?, repeat_kv(v, nrep)?);
        let scale = 1.0f32 / (d as f32).sqrt();
        let attn = sdpa(&q, &k, &v, None, true, scale, 1.0)?; // causal
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((s, nh * d))?;
        l.o.forward(&ao)
    }

    fn enc_layer(
        &self,
        l: &Qwen2Layer,
        x: &Tensor,
        s: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> crate::tensor::Result<Tensor> {
        let eps = self.rms_eps();
        let norm = x.rms_norm(&l.attn_norm, eps)?;
        let x = x.add(&self.self_attn(l, &norm, s, cos, sin)?)?;
        let norm2 = x.rms_norm(&l.ffn_norm, eps)?;
        let gate = l.gate.forward(&norm2)?;
        let up = l.up.forward(&norm2)?;
        let ff = gate.silu()?.mul(&up)?;
        x.add(&l.down.forward(&ff)?)
    }

    /// Build M-RoPE cos/sin `[S, head_dim/2]` from 3D positions `[S][t,h,w]`. Freq `j` uses the
    /// temporal/height/width position per the mrope_section split (`[16,24,24]` for the 7B, half=64).
    fn mrope_cos_sin(&self, positions: &[[i64; 3]]) -> crate::tensor::Result<(Tensor, Tensor)> {
        const SEC: [usize; 3] = [16, 24, 24]; // mrope_section (sums to head_dim/2 = 64)
        let (d, s) = (self.head_dim, positions.len());
        let half = d / 2;
        let inv: Vec<f32> = (0..half)
            .map(|j| 1.0 / self.rope_theta.powf(2.0 * j as f32 / d as f32))
            .collect();
        let axis = |j: usize| {
            if j < SEC[0] {
                0
            } else if j < SEC[0] + SEC[1] {
                1
            } else {
                2
            }
        };
        let (mut cos, mut sin) = (vec![0f32; s * half], vec![0f32; s * half]);
        for i in 0..s {
            for j in 0..half {
                let a = positions[i][axis(j)] as f32 * inv[j];
                cos[i * half + j] = a.cos();
                sin[i * half + j] = a.sin();
            }
        }
        // Built on CPU; the forward moves them to each layer's device.
        Ok((
            Tensor::from_vec_f32(cos, (s, half))?,
            Tensor::from_vec_f32(sin, (s, half))?,
        ))
    }

    /// Encode a text prompt -> conditioning embeddings `[S', 3584]`. Applies the Qwen-Image
    /// chat template, tokenizes (BPE via the `tokenizers` crate), runs the encoder, and drops
    /// the fixed system-prompt prefix (`drop_idx` tokens). `image_pad` inserts the VL vision
    /// placeholder span (edit mode); pass 0 for text-only (drop_idx must match the template).
    pub fn encode_text(
        &self,
        tok: &tokenizers::Tokenizer,
        text: &str,
        drop_idx: usize,
    ) -> crate::tensor::Result<Tensor> {
        let ids: Vec<u32> = tok
            .encode(text, true)
            .map_err(|e| crate::tensor::Error(format!("tokenize: {e}")))?
            .get_ids()
            .to_vec();
        let hid = self.forward(&ids)?; // [S, 3584]
        let s = ids.len();
        let keep = s.saturating_sub(drop_idx).max(1);
        hid.narrow(0, s - keep, keep) // drop the leading `drop_idx` template tokens
    }

    /// DEBUG: forward with EXACT token ids + 3D positions, splicing `vision` rows at `image_token`
    /// positions. Isolates the LLM+M-RoPE+splice from tokenization/position-computation.
    pub fn forward_debug(
        &self,
        ids: &[u32],
        positions: &[[i64; 3]],
        vision: &Tensor,
        image_token: u32,
        drop: usize,
    ) -> crate::tensor::Result<Tensor> {
        let mut rows: Vec<Tensor> = Vec::with_capacity(ids.len());
        let mut vi = 0usize;
        for &id in ids {
            if id == image_token {
                rows.push(vision.narrow(0, vi, 1)?);
                vi += 1;
            } else {
                rows.push(self.embed_tokens.narrow(0, id as usize, 1)?);
            }
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let embeds = Tensor::cat(&refs, 0)?;
        let hid = self.forward_embeds_pos(&embeds, positions)?;
        let s = hid.dim(0)?;
        let keep = s.saturating_sub(drop).max(1);
        hid.narrow(0, s - keep, keep)
    }

    /// Row-gather token embeddings `[S,3584]`.
    fn embed_ids(&self, ids: &[u32]) -> crate::tensor::Result<Tensor> {
        let rows: Vec<Tensor> = ids
            .iter()
            .map(|&id| self.embed_tokens.narrow(0, id as usize, 1))
            .collect::<crate::tensor::Result<_>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 0)
    }

    /// Run the encoder layers over pre-built input embeddings `[S,3584]` with 3D M-RoPE positions.
    pub fn forward_embeds_pos(
        &self,
        embeds: &Tensor,
        positions: &[[i64; 3]],
    ) -> crate::tensor::Result<Tensor> {
        use crate::tensor::DeviceLocation;
        let s = embeds.dim(0)?;
        let (cos0, sin0) = self.mrope_cos_sin(positions)?; // on CPU
                                                           // Cache cos/sin on each distinct layer device so a layer reads only on-device tensors.
        let mut rope: std::collections::HashMap<DeviceLocation, (Tensor, Tensor)> =
            std::collections::HashMap::new();
        for l in &self.layers {
            let loc = l.device.location();
            if let std::collections::hash_map::Entry::Vacant(e) = rope.entry(loc) {
                e.insert((cos0.to_device(&l.device)?, sin0.to_device(&l.device)?));
            }
        }
        // Walk layers; move `hid` to a layer's device only when it differs (single-device plan ->
        // no-op -> identical to the single-device path). Final norm lives on the last layer's device.
        let mut hid = embeds.to_device(&self.device)?;
        let mut cur = self.device.clone();
        for l in &self.layers {
            if l.device.location() != cur.location() {
                cur.synchronize()?;
                hid = hid.to_device(&l.device)?;
                l.device.synchronize()?;
                cur = l.device.clone();
            }
            let (cos, sin) = &rope[&l.device.location()];
            hid = self.enc_layer(l, &hid, s, cos, sin)?;
        }
        hid.rms_norm(&self.norm, self.rms_eps())
    }

    /// Qwen-Image-EDIT conditioning: splice the vision-tower embeds `[N,3584]` into the edit prompt
    /// at the `<|vision_start|>...<|vision_end|>` span, run the LLM, drop the `drop_idx` system prefix.
    /// `instruction` is the user's edit text. Matches `_get_qwen_prompt_embeds` (image-aware).
    pub fn encode_edit(
        &self,
        tok: &tokenizers::Tokenizer,
        vision: &Tensor,
        instruction: &str,
        llm_h: usize,
        llm_w: usize,
    ) -> crate::tensor::Result<Tensor> {
        // Prefix WITHOUT vision_start; vision_start is kept in the conditioning (matches the reference
        // drop=prefix-len, which keeps [vision_start ; image ; vision_end instruction]).
        const PRE: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
        let vstart = "<|vision_start|>";
        let tail = format!("<|vision_end|>{instruction}<|im_end|>\n<|im_start|>assistant\n");
        let ids = |t: &str, add: bool| -> crate::tensor::Result<Vec<u32>> {
            Ok(tok
                .encode(t, add)
                .map_err(|e| crate::tensor::Error(format!("tok: {e}")))?
                .get_ids()
                .to_vec())
        };
        let pre_ids = ids(PRE, true)?;
        let vs_ids = ids(vstart, false)?;
        let tail_ids = ids(&tail, false)?;
        let (pre, vs, tailt) = (
            self.embed_ids(&pre_ids)?,
            self.embed_ids(&vs_ids)?,
            self.embed_ids(&tail_ids)?,
        );
        let embeds = Tensor::cat(&[&pre, &vs, vision, &tailt], 0)?; // [pre ; vision_start ; image ; tail]
        let _ = (llm_h, llm_w);
        // The diffusers pipeline calls the text encoder WITHOUT position_ids -> the Qwen2.5-VL text
        // model defaults to 1D `arange` for ALL tokens (M-RoPE degenerates; verified byte-exact vs
        // the real model: 1D -> conditioning cosine 0.984, mrope-3D -> only 0.78). So: plain 1D.
        let s = embeds.dim(0)?;
        let pos: Vec<[i64; 3]> = (0..s).map(|i| [i as i64; 3]).collect();
        let hid = self.forward_embeds_pos(&embeds, &pos)?;
        let s = hid.dim(0)?;
        let keep = s.saturating_sub(pre_ids.len()).max(1); // drop the system+user prefix, keep vision_start onward
        hid.narrow(0, s - keep, keep)
    }

    /// Encode token ids -> last hidden states `[S, 3584]`, post final RMSNorm. Text -> 1D M-RoPE
    /// (t=h=w=position), which is the standard 1D rope.
    pub fn forward(&self, token_ids: &[u32]) -> crate::tensor::Result<Tensor> {
        let embeds = self.embed_ids(token_ids)?;
        let positions: Vec<[i64; 3]> = (0..token_ids.len()).map(|i| [i as i64; 3]).collect();
        self.forward_embeds_pos(&embeds, &positions)
    }
}
