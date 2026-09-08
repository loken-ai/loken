//! EzAudio DiT (the UDiT diffusion denoiser) - full-Rust on the native substrate.
//!
//! EzAudio's denoiser is a U-ViT-shaped diffusion transformer ("UDiT"): a 1-D conv
//! patch-embed, a symmetric tower of `in_blocks` -> `mid_block` -> `out_blocks` with
//! U-Net long skip connections (early block outputs concatenated into the matching
//! late block), and a conv final head. It predicts the `v`-parameterized velocity
//! field for one diffusion step, conditioned on the diffusion timestep (AdaLN) and on
//! the T5 text encoding (cross-attention).
//!
//! Per-block math vs the ACE-Step DiT differs in five ways, all reflected here:
//!   * norms are **LayerNorm** (mean-subtracting, weight+bias), not RMSNorm;
//!   * the qk-norm is a **LayerNorm over the head_dim** (weight+bias), not RMS;
//!   * the MLP is **GEGLU** (`x . gelu(gate)`), not SwiGLU;
//!   * time conditioning is **AdaLN-SOLA** = a SHARED modulation `time_ada(temb)`
//!     (one-for-all, computed once) + a per-block low-rank correction
//!     `lora_b(lora_a(temb))` + the per-block constant `scale_shift_table` bias;
//!   * self-attention is **full / non-causal** with shared RoPE and **no GQA**
//!     (k/v have the same head count as q).
//!
//! Config (ckpts/ezaudio-l.yml): embed_dim 1024, depth 24 (12 in + 1 mid + 11 out...),
//! num_heads 16 (head_dim 64), mlp_ratio 4 geglu, context_dim 1024, in_chans 257,
//! out_chans 128, patch_size 1. v-prediction. Geometry is resolved from the real `.pt`
//! tensor shapes at load (not hard-coded) so a different EzAudio size still loads.
//!
//! ✅ ORACLE-VALIDATED (`bin/ezaudio_oracle` vs the torch reference): every
//! forward stage matches cos≈1.0 (patch_embed, AdaLN-SOLA modulation, qk-norm+RoPE attn,
//! GEGLU, cross-attn, block/mid/out outputs, final). The one fidelity bug found & fixed:
//! the block residual gate is `(1 - gate)`, not `gate` (blocks.py `DiTBlock._forward`).
//! Set `EZ_DUMP_DIR` to re-dump per-stage activations for the A/B.

use crate::inference::model::ezaudio::vae::ezaudio_pt;
use crate::tensor::layer::{same_length_1d, Conv1d, Conv1dConfig, LayerNorm, Linear};
use crate::tensor::{DType, Device, Error, Result, Tensor};

const LN_EPS: f32 = 1e-5; // LayerNorm default eps (norm_layer = layernorm)

/// Dev-only per-stage activation dump (env `EZ_DUMP_DIR`) for the torch oracle A/B.
/// No-op unless the env var is set. Writes raw little-endian f32 to `{dir}/{tag}.bin`.
fn ez_dump(tag: &str, t: &Tensor) {
    if let Ok(dir) = std::env::var("EZ_DUMP_DIR") {
        let v = t.to_vec_f32();
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let _ = std::fs::write(format!("{dir}/{tag}.bin"), bytes);
    }
}

/// One attention sub-module (self- or cross-attention). `to_{q,k,v}` are bias-free
/// projections; `proj` (the output) carries a bias. `norm_q`/`norm_k` are the qk-norm
/// LayerNorms over the head_dim. `inv_freq` is `Some` only for self-attention (the
/// shared RoPE table source); cross-attention applies no positional rotation.
struct Attn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    proj: Linear,
    norm_q: LayerNorm,
    norm_k: LayerNorm,
    inv_freq: Option<Vec<f32>>,
}

/// The AdaLN-SOLA modulation source for one block. The full per-block 6-way
/// modulation is `scale_shift_table (constant) + time_ada(c) (shared) + lora_b(lora_a(c))
/// (low-rank per-block)`, where `c` is the timestep conditioning vector. `time_ada(c)`
/// is computed once at the model level and passed in; this struct holds the two
/// per-block terms.
struct AdaLn {
    scale_shift_table: Tensor, // [1, 6.H] (flattened from [6, H])
    lora_a: Linear,            // H -> rank.6   (bias-free)
    lora_b: Linear,            // rank.6 -> 6.H (bias-free)
    lora_scale: f32,           // ada_lora_alpha / ada_lora_rank
}

/// One UDiT block: self-attn (AdaLN-modulated + gated), cross-attn to the text context
/// (plain LayerNorm, ungated residual), GEGLU MLP (AdaLN-modulated + gated). `skip` is
/// `Some` for `out_blocks` (the U-Net long skip: concat the saved early-block output,
/// LayerNorm over the 2.H concat, project back to H).
struct Block {
    norm1: LayerNorm,        // pre self-attn
    norm2: LayerNorm,        // pre cross-attn (query side)
    norm3: LayerNorm,        // pre MLP
    norm_context: LayerNorm, // context (cross-attn k/v side)
    attn: Attn,
    cross_attn: Attn,
    mlp_in: Linear,  // H -> 2.inner (GEGLU: value half | gate half)
    mlp_out: Linear, // inner -> H
    adaln: AdaLn,
    skip: Option<(LayerNorm, Linear)>, // (skip_norm over 2H, skip_linear 2H->H)
}

/// The EzAudio UDiT velocity-field predictor.
pub struct EzAudioDiT {
    patch_embed: Conv1d, // [in_chans -> H], kernel = patch_size
    // context_embed: Linear -> SiLU -> Linear (projects the T5 encoding into H).
    ctx_embed_0: Linear,
    ctx_embed_2: Linear,
    // time_embed: sinusoid(t, 256) -> Linear -> SiLU -> Linear -> temb [H].
    time_embed_0: Linear,
    time_embed_2: Linear,
    time_ada: Linear,       // H -> 6.H   (shared AdaLN modulation, with bias)
    time_ada_final: Linear, // H -> 2.H   (final-block shift/scale)
    in_blocks: Vec<Block>,
    mid_block: Block,
    out_blocks: Vec<Block>,
    final_norm: LayerNorm, // [H]
    final_linear: Linear,  // H -> out_chans.patch_size
    final_conv: Conv1d,    // out_chans -> out_chans, kernel 3 pad 1
    hidden: usize,
    n_heads: usize,
    head_dim: usize,
    in_chans: usize,
    out_chans: usize,
    /// The learned MAE `mask_embed` `[out_chans]` (top-level `mask_embed` in the `.pt`):
    /// the conditioning tensor that fills the `gt` block of the faithful 257-ch input for
    /// pure text->audio (no reference latent). Absmax ~0.017.
    mask_embed: Vec<f32>,
    /// One device per block, in forward order: the in-blocks, the middle, the
    /// out-blocks. The stem and the head stay on `device`.
    block_devices: Vec<Device>,
    device: Device,
}

impl EzAudioDiT {
    /// Resolve + load the EzAudio S3 large DiT from the configured HF hub dir
    /// (`ckpts/s3/ezaudio_s3_l.pt`).
    pub fn load_s3_large() -> Result<Self> {
        let p = ezaudio_pt("ckpts/s3/ezaudio_s3_l.pt");
        Self::from_pt(
            p.to_str()
                .ok_or_else(|| Error("ezaudio-dit: bad path".into()))?,
        )
    }

    /// Load the DiT from a torch `.pt` (full-Rust [`crate::tensor::pth`] reader).
    /// All weights are coerced to F32 on the placement device.
    pub fn from_pt(path: &str) -> Result<Self> {
        let m = crate::tensor::pth::read_pt(path)?;
        // The file is right here; a written-down size is a fact about one build of it.
        let model_size = std::fs::metadata(path).map(|x| x.len()).unwrap_or(0);
        let dev = crate::inference::model::acestep::vae::vae_best_device(model_size);
        eprintln!("[ezaudio-dit] loading {} tensors from {path}", m.len());

        // THE CARD A BLOCK LANDS ON, rewritten before each block is built.
        //
        // Every tensor used to go to one device, so this per-step tower could not use
        // a second card and, when it fit none, went to the host in its entirety. The
        // fetch helpers below read this cell rather than a fixed device, which keeps
        // the change to where the blocks are built instead of threading a device
        // through nine closures.
        let cur_dev = std::cell::RefCell::new(dev.clone());

        // -- tensor / layer fetch helpers (host .pt -> F32 device tensors) --
        let get = |name: &str| -> Result<Tensor> {
            m.get(name)
                .ok_or_else(|| Error(format!("ezaudio-dit: missing tensor `{name}`")))?
                .to_dtype(DType::F32)?
                .to_device(&cur_dev.borrow())
        };
        let lin = |name: &str| -> Result<Linear> {
            Linear::new(
                get(&format!("{name}.weight"))?,
                Some(get(&format!("{name}.bias"))?),
            )
        };
        let lin_nb =
            |name: &str| -> Result<Linear> { Linear::new(get(&format!("{name}.weight"))?, None) };
        let ln = |name: &str| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                get(&format!("{name}.weight"))?,
                Some(get(&format!("{name}.bias"))?),
                LN_EPS,
            ))
        };

        // -- geometry from the real tensor shapes --
        let pe_w = get("model.patch_embed.proj.weight")?; // [H, in_chans, patch]
        let (hidden, in_chans, patch) = {
            let d = pe_w.dims();
            (d[0], d[1], d[2])
        };
        let qn = get("model.in_blocks.0.attn.norm_q.weight")?; // [head_dim]
        let head_dim = qn.dims()[0];
        let n_heads = hidden / head_dim;
        let out_chans = get("model.final_block.linear.weight")?.dims()[0]; // [out_chans.patch, H]
                                                                           // SOLA rank: lora_a is [rank.6, H] -> rank = rows / 6.
        let lora_a_rows = get("model.in_blocks.0.adaln.lora_a.weight")?.dims()[0];
        let lora_rank = lora_a_rows / 6;
        // ada_lora_alpha == ada_lora_rank in the EzAudio config ⟹ scale 1.0; kept explicit.
        let lora_scale = 1.0f32;
        let n_in = (0..)
            .take_while(|i| m.contains_key(&format!("model.in_blocks.{i}.norm1.weight")))
            .count();
        let n_out = (0..)
            .take_while(|i| m.contains_key(&format!("model.out_blocks.{i}.norm1.weight")))
            .count();
        eprintln!("[ezaudio-dit] geom: H={hidden} heads={n_heads} hd={head_dim} in_ch={in_chans} out_ch={out_chans} patch={patch} rank={lora_rank} in_blocks={n_in} out_blocks={n_out}");

        // -- per-block loader --
        let load_attn = |prefix: &str, has_rope: bool| -> Result<Attn> {
            let inv_freq = if has_rope {
                Some(get(&format!("{prefix}.rotary.inv_freq"))?.to_vec_f32())
            } else {
                None
            };
            Ok(Attn {
                to_q: lin_nb(&format!("{prefix}.to_q"))?,
                to_k: lin_nb(&format!("{prefix}.to_k"))?,
                to_v: lin_nb(&format!("{prefix}.to_v"))?,
                proj: lin(&format!("{prefix}.proj"))?,
                norm_q: ln(&format!("{prefix}.norm_q"))?,
                norm_k: ln(&format!("{prefix}.norm_k"))?,
                inv_freq,
            })
        };
        let load_block = |prefix: &str, is_out: bool| -> Result<Block> {
            let sst = get(&format!("{prefix}.adaln.scale_shift_table"))?; // [6, H]
            let sst = sst.reshape((1, 6 * hidden))?;
            let skip = if is_out {
                Some((
                    ln(&format!("{prefix}.skip_norm"))?,
                    lin(&format!("{prefix}.skip_linear"))?,
                ))
            } else {
                None
            };
            Ok(Block {
                norm1: ln(&format!("{prefix}.norm1"))?,
                norm2: ln(&format!("{prefix}.norm2"))?,
                norm3: ln(&format!("{prefix}.norm3"))?,
                norm_context: ln(&format!("{prefix}.norm_context"))?,
                attn: load_attn(&format!("{prefix}.attn"), true)?,
                cross_attn: load_attn(&format!("{prefix}.cross_attn"), false)?,
                mlp_in: lin(&format!("{prefix}.mlp.net.0.proj"))?,
                mlp_out: lin(&format!("{prefix}.mlp.net.2"))?,
                adaln: AdaLn {
                    scale_shift_table: sst,
                    lora_a: lin_nb(&format!("{prefix}.adaln.lora_a"))?,
                    lora_b: lin_nb(&format!("{prefix}.adaln.lora_b"))?,
                    lora_scale,
                },
                skip,
            })
        };

        // Spread the tower over every card, the same mechanism the rest of the fleet
        // uses. The blocks are all the same width here, so a uniform plan is the right
        // shape - unlike a UNet, where the deep stages dwarf the shallow ones.
        let n_blocks = n_in + 1 + n_out;
        let block_devices: Vec<Device> = {
            use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
            let cudas = crate::inference::place::vram_manager::probe_under_pressure(0);
            let budget: Vec<(usize, u64)> = cudas.iter().map(|(i, fr, _)| (*i, *fr)).collect();
            let plan = HeteroPlan::calculate(n_blocks, model_size, &budget, &[], 1.0);
            let mut out = vec![Device::Cpu; n_blocks];
            for seg in &plan.segments {
                let d = match seg.kind {
                    DeviceKind::Cuda(i) => cudas
                        .iter()
                        .find(|(j, _, _)| *j == i)
                        .map(|(_, _, d)| d.clone())
                        .unwrap_or(Device::Cpu),
                    _ => Device::Cpu,
                };
                for slot in
                    out[seg.layer_start.min(n_blocks)..seg.layer_end.min(n_blocks)].iter_mut()
                {
                    *slot = d.clone();
                }
            }
            out
        };
        eprintln!(
            "[ezaudio-dit] {n_blocks} blocks placed across {} device(s)",
            {
                let mut seen: Vec<String> = block_devices
                    .iter()
                    .map(|d| format!("{:?}", d.location()))
                    .collect();
                seen.sort();
                seen.dedup();
                seen.len()
            }
        );

        let mut in_blocks = Vec::with_capacity(n_in);
        for i in 0..n_in {
            *cur_dev.borrow_mut() = block_devices[i].clone();
            in_blocks.push(load_block(&format!("model.in_blocks.{i}"), false)?);
        }
        *cur_dev.borrow_mut() = block_devices[n_in].clone();
        let mid_block = load_block("model.mid_block", false)?;
        let mut out_blocks = Vec::with_capacity(n_out);
        for i in 0..n_out {
            *cur_dev.borrow_mut() = block_devices[n_in + 1 + i].clone();
            out_blocks.push(load_block(&format!("model.out_blocks.{i}"), true)?);
        }
        // The stem and the head stay with the primary, so a caller always gets its
        // result back where it handed the input in.
        *cur_dev.borrow_mut() = dev.clone();

        // patch_embed conv: weight [H, in_chans, patch], kernel = patch_size.
        let patch_embed = Conv1d::new(
            pe_w,
            Some(get("model.patch_embed.proj.bias")?),
            Conv1dConfig {
                padding: 0,
                stride: patch,
                dilation: 1,
                groups: 1,
            },
        );
        // final conv: [out_chans, out_chans, 3] pad 1.
        let fc_w = get("model.final_block.final_layer.weight")?;
        let fc_k = fc_w.dims()[2];
        let final_conv = Conv1d::new(
            fc_w,
            Some(get("model.final_block.final_layer.bias")?),
            same_length_1d(fc_k, 1),
        );

        Ok(Self {
            patch_embed,
            ctx_embed_0: lin("model.context_embed.0")?,
            ctx_embed_2: lin("model.context_embed.2")?,
            time_embed_0: lin("model.time_embed.mlp.0")?,
            time_embed_2: lin("model.time_embed.mlp.2")?,
            time_ada: lin("model.time_ada")?,
            time_ada_final: lin("model.time_ada_final")?,
            in_blocks,
            mid_block,
            out_blocks,
            block_devices,
            final_norm: ln("model.final_block.norm")?,
            final_linear: lin("model.final_block.linear")?,
            final_conv,
            hidden,
            n_heads,
            head_dim,
            in_chans,
            out_chans,
            mask_embed: get("mask_embed")?.to_vec_f32(),
            device: dev,
        })
    }

    /// The learned MAE `mask_embed` `[out_chans]` - the `gt`/condition block used to build
    /// the faithful UDiT input for pure text->audio (Stage-4 `ncm` assembly).
    pub fn mask_embed(&self) -> &[f32] {
        &self.mask_embed
    }

    /// The full DiT input channel count (UDiT `in_chans`, e.g. 257) - the width Stage-4
    /// must assemble (noisy latent ⊕ condition ⊕ mask).
    pub fn in_chans(&self) -> usize {
        self.in_chans
    }

    /// Build NEOX RoPE cos/sin tables from a stored `inv_freq` (`[head_dim/2]`):
    /// `cos[p,j] = cos(p.inv_freq[j])`, `sin` likewise. Returns `([S, hd/2], [S, hd/2])`
    /// device tensors. RoPE is "shared" across blocks (all blocks store the same table).
    fn rope_tables(&self, inv_freq: &[f32], s: usize) -> Result<(Tensor, Tensor)> {
        let half = inv_freq.len();
        let (mut cos, mut sin) = (vec![0f32; s * half], vec![0f32; s * half]);
        for p in 0..s {
            for j in 0..half {
                let a = p as f32 * inv_freq[j];
                cos[p * half + j] = a.cos();
                sin[p * half + j] = a.sin();
            }
        }
        Ok((
            Tensor::from_vec_f32(cos, (s, half))?.to_device(&self.device)?,
            Tensor::from_vec_f32(sin, (s, half))?.to_device(&self.device)?,
        ))
    }

    /// One attention pass (tokens-major). `xq` is the query source `[Sq, H]`; `xkv` is the
    /// key/value source `[Sk, H]` (== `xq` for self-attention). Reshapes to heads, applies
    /// the qk-norm LayerNorm over the head_dim, optional RoPE (self-attn only, from
    /// `cos`/`sin`), then full bidirectional SDPA, then the output projection. Returns
    /// `[Sq, H]`.
    fn attention(
        &self,
        a: &Attn,
        xq: &Tensor,
        xkv: &Tensor,
        sq: usize,
        sk: usize,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        use crate::inference::model::acestep::ops::sdpa;
        let (nh, hd) = (self.n_heads, self.head_dim);
        // [S, H] -> [1, nh, S, hd]
        let to_heads = |t: Tensor, s: usize| -> Result<Tensor> {
            t.reshape((s, nh, hd))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()
        };
        let mut q = to_heads(a.to_q.forward(xq)?, sq)?;
        let mut k = to_heads(a.to_k.forward(xkv)?, sk)?;
        let v = to_heads(a.to_v.forward(xkv)?, sk)?;
        // qk-norm: LayerNorm over the head_dim (weight+bias).
        q = a.norm_q.forward(&q)?;
        k = a.norm_k.forward(&k)?;
        if let Some((cos, sin)) = rope {
            q = q.rope(cos, sin)?;
            k = k.rope(cos, sin)?;
        }
        let scale = 1.0f32 / (hd as f32).sqrt();
        let attn = sdpa(&q, &k, &v, None, false, scale, 1.0)?; // [1, nh, sq, hd]
        let ao = attn.transpose(1, 2)?.contiguous()?.reshape((sq, nh * hd))?;
        a.proj.forward(&ao)
    }

    /// GEGLU MLP: `proj(x)` -> split into (value, gate) halves -> `value . gelu(gate)` -> down.
    fn geglu(&self, b: &Block, x: &Tensor) -> Result<Tensor> {
        let h = b.mlp_in.forward(x)?; // [S, 2.inner]
        let inner = h.dims()[1] / 2;
        let value = h.narrow(1, 0, inner)?;
        let gate = h.narrow(1, inner, inner)?;
        // nn.GELU() default = exact (erf) gelu.
        let act = value.mul(&gate.gelu_erf()?)?;
        b.mlp_out.forward(&act)
    }

    /// `modulate(norm) = norm . (1 + scale) + shift`, per-channel broadcast over tokens.
    fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(shift)
    }

    /// One UDiT block. `x` `[S, H]`; `t0` the shared `time_ada(c)` `[1, 6H]`; `c` the
    /// timestep conditioning `[1, H]` (drives the per-block SOLA low-rank term); `ctx`
    /// the embedded text `[Sc, H]`; `rope` the shared self-attn cos/sin. `skip` is the
    /// saved early-block output for an `out_block` (`None` for in/mid).
    fn block_forward(
        &self,
        b: &Block,
        mut x: Tensor,
        t0: &Tensor,
        c: &Tensor,
        ctx: &Tensor,
        rope: (&Tensor, &Tensor),
        s: usize,
        sc: usize,
        skip: Option<&Tensor>,
        tag: &str,
    ) -> Result<Tensor> {
        let h = self.hidden;
        // U-Net skip: concat([x, skip]) -> skip_norm (over 2H) -> skip_linear (-> H).
        if let (Some((skip_norm, skip_linear)), Some(skip_x)) = (&b.skip, skip) {
            let cat = Tensor::cat(&[&x, skip_x], 1)?; // [S, 2H]
            x = skip_linear.forward(&skip_norm.forward(&cat)?)?;
        }
        // AdaLN-SOLA: ada[6H] = scale_shift_table + time_ada(c) + lora_scale.lora_b(lora_a(c)).
        let sola = b
            .adaln
            .lora_b
            .forward(&b.adaln.lora_a.forward(c)?)?
            .affine(b.adaln.lora_scale, 0.0)?;
        let ada = b.adaln.scale_shift_table.add(t0)?.add(&sola)?; // [1, 6H]
        if tag == "b0" {
            ez_dump("b0_adaln", &ada);
        }
        let part = |i: usize| ada.narrow(1, i * h, h);
        // chunk order: shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp.
        let (shift_sa, scale_sa, gate_sa) = (part(0)?, part(1)?, part(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (part(3)?, part(4)?, part(5)?);

        // self-attn (AdaLN-modulated norm -> attn -> (1-gate) residual). The reference gates
        // the residual with (1 - gate), NOT gate (blocks.py DiTBlock._forward).
        let n1 = b.norm1.forward(&x)?;
        if tag == "b0" {
            ez_dump("b0_norm1", &n1);
        }
        let norm_sa = Self::modulate(&n1, &scale_sa, &shift_sa)?;
        let sa = self.attention(&b.attn, &norm_sa, &norm_sa, s, s, Some(rope))?;
        if tag == "b0" {
            ez_dump("b0_attn", &sa);
        }
        x = x.add(&sa.broadcast_mul(&gate_sa.affine(-1.0, 1.0)?)?)?;

        // cross-attn (plain LayerNorm query + context, ungated residual).
        let q_in = b.norm2.forward(&x)?;
        let kv_in = b.norm_context.forward(ctx)?;
        let ca = self.attention(&b.cross_attn, &q_in, &kv_in, s, sc, None)?;
        if tag == "b0" {
            ez_dump("b0_cross", &ca);
        }
        x = x.add(&ca)?;

        // GEGLU MLP (AdaLN-modulated norm -> geglu -> (1-gate) residual).
        let norm_mlp = Self::modulate(&b.norm3.forward(&x)?, &scale_mlp, &shift_mlp)?;
        let mlp = self.geglu(b, &norm_mlp)?;
        if tag == "b0" {
            ez_dump("b0_mlp", &mlp);
        }
        x = x.add(&mlp.broadcast_mul(&gate_mlp.affine(-1.0, 1.0)?)?)?;
        if !tag.is_empty() {
            ez_dump(&format!("{tag}_out"), &x);
        }
        Ok(x)
    }

    /// Predict the velocity `v` for one diffusion step. `x_noisy` is the full DiT input
    /// `[in_chans, T]` (S4 assembles it: noisy latent ⊕ reference/condition latent ⊕ mask
    /// = `in_chans` channels). `t` is the diffusion timestep (embedded as-is; the caller
    /// owns the 0..1 vs 0..1000 convention). `ctx` is the T5 text encoding `[Sc, H]`.
    /// Returns `v` `[out_chans, T]`.
    pub fn forward(&self, x_noisy: &Tensor, t: f32, ctx: &Tensor) -> Result<Tensor> {
        let h = self.hidden;
        let dims = x_noisy.dims();
        if dims.len() != 2 || dims[0] != self.in_chans {
            return Err(Error(format!(
                "ezaudio-dit: x_noisy must be [{}, T], got {:?}",
                self.in_chans, dims
            )));
        }
        let t_len = dims[1];
        let x_dev = x_noisy.to_device(&self.device)?;

        // patch embed: [in_chans, T] -> [1, in_chans, T] conv -> [1, H, S] -> [S, H].
        let pe = self.patch_embed.forward(&x_dev.unsqueeze(0)?)?; // [1, H, S]
        let s = pe.dims()[2];
        let mut x = pe.squeeze(0)?.transpose(0, 1)?.contiguous()?; // [S, H]
        ez_dump("patch_embed", &x);

        // timestep -> sinusoid(256) -> MLP -> temb [1, H]; c = silu(temb) drives AdaLN.
        let sin = crate::inference::model::acestep::dit::timestep_embedding(t, 256, 10000.0); // [256]
        let temb_in = Tensor::from_vec_f32(sin, (1, 256))?.to_device(&self.device)?;
        let temb = self
            .time_embed_2
            .forward(&self.time_embed_0.forward(&temb_in)?.silu()?)?; // [1, H]
        let c = temb.silu()?; // conditioning for the AdaLN projections
        let t0 = self.time_ada.forward(&c)?; // [1, 6H] shared modulation
        let final_mod = self.time_ada_final.forward(&c)?; // [1, 2H]

        // context embed: Linear -> SiLU -> Linear.
        let ctx_dev = ctx.to_device(&self.device)?;
        let ctx_e = self
            .ctx_embed_2
            .forward(&self.ctx_embed_0.forward(&ctx_dev)?.silu()?)?; // [Sc, H]
        let sc = ctx_e.dims()[0];

        // shared RoPE table (all self-attn blocks share it).
        let inv_freq = self.in_blocks[0]
            .attn
            .inv_freq
            .clone()
            .ok_or_else(|| Error("ezaudio-dit: missing self-attn rope".into()))?;
        // Built once; each block takes it on ITS card below, since the blocks no
        // longer all live on the same one.
        let (cos, sin_t) = self.rope_tables(&inv_freq, s)?;

        // U-ViT tower: in_blocks (push outputs) -> mid -> out_blocks (pop skips, LIFO).
        // Each block runs where its weights are. The activation follows, and so does
        // the conditioning - it is small, and carrying it keeps the block code
        // device-agnostic. A skip is pushed on the way down and popped on the way up,
        // so its two ends are placed independently and it has to be moved when used.
        let mut skips: Vec<Tensor> = Vec::with_capacity(self.in_blocks.len());
        let on = |t: &Tensor, d: &Device| -> Result<Tensor> { t.to_device(d) };
        for (i, b) in self.in_blocks.iter().enumerate() {
            let tag = match i {
                0 => "b0",
                1 => "b1",
                _ => "",
            };
            let d = &self.block_devices[i];
            let (t0d, cd, ctxd) = (on(&t0, d)?, on(&c, d)?, on(&ctx_e, d)?);
            let (cosd, sind) = (on(&cos, d)?, on(&sin_t, d)?);
            x = self.block_forward(
                b,
                on(&x, d)?,
                &t0d,
                &cd,
                &ctxd,
                (&cosd, &sind),
                s,
                sc,
                None,
                tag,
            )?;
            skips.push(x.clone());
        }
        {
            let d = &self.block_devices[self.in_blocks.len()];
            let (t0d, cd, ctxd) = (on(&t0, d)?, on(&c, d)?, on(&ctx_e, d)?);
            let (cosd, sind) = (on(&cos, d)?, on(&sin_t, d)?);
            x = self.block_forward(
                &self.mid_block,
                on(&x, d)?,
                &t0d,
                &cd,
                &ctxd,
                (&cosd, &sind),
                s,
                sc,
                None,
                "mid",
            )?;
        }
        let out_base = self.in_blocks.len() + 1;
        for (i, b) in self.out_blocks.iter().enumerate() {
            let skip = skips
                .pop()
                .ok_or_else(|| Error("ezaudio-dit: skip stack underflow".into()))?;
            let tag = if i == 0 { "out0" } else { "" };
            let d = &self.block_devices[out_base + i];
            let (t0d, cd, ctxd) = (on(&t0, d)?, on(&c, d)?, on(&ctx_e, d)?);
            let (cosd, sind) = (on(&cos, d)?, on(&sin_t, d)?);
            x = self.block_forward(
                b,
                on(&x, d)?,
                &t0d,
                &cd,
                &ctxd,
                (&cosd, &sind),
                s,
                sc,
                Some(&on(&skip, d)?),
                tag,
            )?;
        }
        // Back where the caller handed it in, for the head.
        x = x.to_device(&self.device)?;

        // final: modulate(final_norm(x)) with (shift, scale) -> linear -> conv head.
        let shift = final_mod.narrow(1, 0, h)?;
        let scale = final_mod.narrow(1, h, h)?;
        let xn = Self::modulate(&self.final_norm.forward(&x)?, &scale, &shift)?;
        let xl = self.final_linear.forward(&xn)?; // [S, out_chans.patch]
                                                  // [S, out_chans.patch] -> [1, out_chans.patch, S] -> conv -> [out_chans, T].
        let xc = xl.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let out = self.final_conv.forward(&xc)?; // [1, out_chans, T]
        debug_assert_eq!(out.dims(), &[self.out_chans, t_len]);
        let out = out.squeeze(0)?;
        ez_dump("final", &out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    #[test]
    #[ignore = "needs EzAudio ckpts/s3/ezaudio_s3_l.pt (config hf_models_dir)"]
    fn dump_dit_keys() {
        let p = ezaudio_pt("ckpts/s3/ezaudio_s3_l.pt");
        println!("path: {}", p.display());
        let m = crate::tensor::pth::read_pt(p.to_str().unwrap()).unwrap();
        let mut keys: Vec<_> = m.keys().cloned().collect();
        keys.sort();
        println!("{} tensors total", keys.len());
        let group_idx = |k: &str, g: &str| -> Option<usize> {
            k.split(&format!("{g}."))
                .nth(1)?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        };
        let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
        for g in ["in_blocks", "out_blocks", "mid_block"] {
            let mx = keys.iter().filter_map(|k| group_idx(k, g)).max();
            if let Some(mx) = mx {
                counts.insert(g, mx + 1);
            } else if keys.iter().any(|k| k.contains(&format!(".{g}."))) {
                counts.insert(g, 1);
            }
        }
        println!("== HEAD / non-block tensors ==");
        for k in &keys {
            let in_block = ["in_blocks", "out_blocks", "mid_block"]
                .iter()
                .any(|g| k.contains(&format!(".{g}.")));
            if !in_block {
                println!("  {k}  {:?}", m[k].dims());
            }
        }
        for g in ["in_blocks", "out_blocks"] {
            println!("== {g} (count {:?}) - block 0 ==", counts.get(g));
            for k in &keys {
                if k.contains(&format!(".{g}.0.")) {
                    println!("  {k}  {:?}", m[k].dims());
                }
            }
        }
        println!("== mid_block (all keys) ==");
        for k in &keys {
            if k.contains(".mid_block.") {
                println!("  {k}  {:?}", m[k].dims());
            }
        }
        println!("block counts: {counts:?}");
    }

    #[test]
    #[ignore = "needs EzAudio ckpts/s3/ezaudio_s3_l.pt; sanity forward (shape + finite)"]
    fn sanity_forward() {
        let dit = EzAudioDiT::load_s3_large().unwrap();
        let (ic, oc, t) = (dit.in_chans, dit.out_chans, 64usize);
        // random noisy DiT input [in_chans, T] (deterministic pseudo-random).
        let xv: Vec<f32> = (0..ic * t)
            .map(|i| (i as f32 * 0.137).sin() * 0.5)
            .collect();
        let x = Tensor::from_vec_f32(xv, (ic, t)).unwrap();
        // tiny T5-shaped context [Sc, H].
        let sc = 5usize;
        let cv: Vec<f32> = (0..sc * dit.hidden)
            .map(|i| (i as f32 * 0.071).cos() * 0.3)
            .collect();
        let ctx = Tensor::from_vec_f32(cv, (sc, dit.hidden)).unwrap();

        let v = dit.forward(&x, 500.0, &ctx).unwrap();
        let d = v.dims();
        println!("ezaudio-dit forward out dims: {d:?}");
        assert_eq!(d.len(), 2);
        assert_eq!(d[0], oc, "out_chans");
        assert_eq!(d[1], t, "T preserved (patch_size {} ⟹ S==T here)", t);
        let vv = v.to_vec_f32();
        assert!(vv.iter().all(|x| x.is_finite()), "non-finite output");
        let mean = vv.iter().sum::<f32>() / vv.len() as f32;
        let var = vv.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vv.len() as f32;
        println!(
            "✅ EzAudio DiT forward: [{}, {}] all finite, mean={:.4} std={:.4}",
            d[0],
            d[1],
            mean,
            var.sqrt()
        );
    }
}
