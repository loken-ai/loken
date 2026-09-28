//! Kyutai `tts-1.6b-en_fr` - the conditioner (component 7a of the port).
//!
//! Produces the two conditioning tensors the Helium LM consumes each step:
//!   - `condition_sum` [1, dim] = control-LUT + cfg-LUT (a shared additive offset),
//!   - `condition_cross` [MAX_SPEAKERS*T_voice, dim] = speaker-voice projection + sinusoidal
//!     position emb (the cross-attention source `ca`).
//!
//! speaker_wavs (TensorConditioner): a precomputed voice embedding `[C, T_voice]` from a
//! `kyutai/tts-voices` `.safetensors` (key `speaker_wavs`, shape `[1, C, T_voice]`) is
//! placed in speaker slot 0 of `MAX_SPEAKERS`, projected to the model width, masked
//! positions filled with `learnt_padding`, then a sin/cos positional embedding is added
//! (fuser `cross_attention_pos_emb`, scale 1). cfg/control (LUTConditioner): a class index
//! -> `embed` lookup -> `output_proj` -> model width; the index is the value's place in the
//! conditioner's `possible_values` in the checkpoint's `config.json`.

use crate::inference::model::kyutai::config::KyutaiConfig;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

/// Speaker slots the cross-attention source holds, of which the first carries the voice.
const MAX_SPEAKERS: usize = 5;
const MAX_PERIOD: f32 = 10_000.0;

/// The classes a conditioner's LUT holds a row for, in row order.
fn lut_values(cfg: &KyutaiConfig, conditioner: &str) -> Result<Vec<String>> {
    cfg.strings(&["conditioners", conditioner, "lut", "possible_values"])
}

pub struct KyutaiConditioner {
    speaker_proj: Tensor,  // [dim, voice_dim]
    speaker_pad: Tensor,   // [dim] learnt_padding
    cfg_embed: Tensor,     // [n_bins + 1, cfg_dim]
    cfg_proj: Tensor,      // [dim, cfg_dim]
    control_embed: Tensor, // [n_bins + 1, dim]
    control_proj: Tensor,  // [dim, dim]
    /// The cfg coefficients the cfg LUT holds a row for, in row order.
    cfg_values: Vec<f32>,
    /// The control LUT's row for the "ok" class.
    control_ok: usize,
    dim: usize,
    voice_dim: usize,
    device: Device,
}
impl KyutaiConditioner {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let cfg = KyutaiConfig::read(path)?;
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let c = vb.pp("condition_provider").pp("conditioners");
        let [dim, voice_dim] = c.shape_of("speaker_wavs.output_proj.weight")?[..] else {
            return Err(crate::tensor::Error(
                "kyutai: speaker_wavs.output_proj.weight is not a matrix".into(),
            ));
        };
        let lut_rows = |name: &str| -> Result<usize> {
            Ok(cfg.usize(&["conditioners", name, "lut", "n_bins"])? + 1)
        };
        let cfg_dim = cfg.usize(&["conditioners", "cfg", "lut", "dim"])?;
        let control_dim = cfg.usize(&["conditioners", "control", "lut", "dim"])?;
        let cfg_values = lut_values(&cfg, "cfg")?
            .iter()
            .map(|v| {
                v.parse::<f32>()
                    .map_err(|e| crate::tensor::Error(format!("kyutai cfg value {v}: {e}")))
            })
            .collect::<Result<Vec<f32>>>()?;
        let control_ok = lut_values(&cfg, "control")?
            .iter()
            .position(|v| v == "ok")
            .ok_or_else(|| crate::tensor::Error("kyutai: no \"ok\" control class".into()))?;
        Ok(Self {
            speaker_proj: c.get((dim, voice_dim), "speaker_wavs.output_proj.weight")?,
            speaker_pad: c
                .get((1, 1, dim), "speaker_wavs.learnt_padding")?
                .reshape(dim)?,
            cfg_embed: c.get((lut_rows("cfg")?, cfg_dim), "cfg.embed.weight")?,
            cfg_proj: c.get((dim, cfg_dim), "cfg.output_proj.weight")?,
            control_embed: c.get((lut_rows("control")?, control_dim), "control.embed.weight")?,
            control_proj: c.get((dim, control_dim), "control.output_proj.weight")?,
            cfg_values,
            control_ok,
            dim,
            voice_dim,
            device,
        })
    }

    /// The cfg LUT row for `cfg_coef`, when the checkpoint was trained with it.
    pub fn cfg_index(&self, cfg_coef: f32) -> Option<usize> {
        self.cfg_values
            .iter()
            .position(|&v| (v - cfg_coef).abs() < 1e-4)
    }

    fn lut(&self, embed: &Tensor, proj: &Tensor, idx: usize) -> Result<Tensor> {
        let id = Tensor::from_vec_u32(vec![idx as u32], vec![1])?.to_device(&self.device)?;
        embed.index_select(&id, 0)?.matmul_t(proj) // [1, dim]
    }

    /// `condition_sum` [1, dim] = control("ok") + cfg(cfg_coef). Both LUT lookups are
    /// masked True (values always provided) so no learnt_padding applies.
    pub fn condition_sum(&self, cfg_coef: f32) -> Result<Tensor> {
        let ci = self.cfg_index(cfg_coef).ok_or_else(|| {
            crate::tensor::Error(format!("kyutai: no cfg conditioning for {cfg_coef}"))
        })?;
        let cfg = self.lut(&self.cfg_embed, &self.cfg_proj, ci)?;
        let ctrl = self.lut(&self.control_embed, &self.control_proj, self.control_ok)?;
        ctrl.add(&cfg)
    }

    /// Sinusoidal position embedding `[T, dim]` = cat(cos(phase), sin(phase)).
    fn sin_embedding(&self, t: usize) -> Result<Tensor> {
        let dim = self.dim;
        let half = dim / 2;
        let mut v = vec![0f32; t * dim];
        for p in 0..t {
            for i in 0..half {
                let phase = p as f32 / MAX_PERIOD.powf(i as f32 / (half as f32 - 1.0));
                v[p * dim + i] = phase.cos();
                v[p * dim + half + i] = phase.sin();
            }
        }
        Tensor::from_vec_f32(v, (t, dim))?.to_device(&self.device)
    }

    /// `condition_cross` [MAX_SPEAKERS*T_voice, dim]. `voice: [voice_dim, T_voice]`
    /// (the raw `speaker_wavs` tensor). Slot 0 is filled, the others are learnt_padding;
    /// a sinusoidal position embedding is added over the whole length.
    pub fn condition_cross(&self, voice: &Tensor) -> Result<Tensor> {
        let voice_dim = self.voice_dim;
        let (vd, tv) = voice.shape().dims2()?;
        if vd != voice_dim {
            return Err(crate::tensor::Error(format!(
                "kyutai: a voice of width {vd} for a conditioner of width {voice_dim}"
            )));
        }
        let total = MAX_SPEAKERS * tv;
        // Build [total, voice_dim]: rows 0..tv = voice transposed, rest zero.
        let vt = voice.transpose(0, 1)?.contiguous()?; // [tv, voice_dim]
        let vt_host = vt.flatten_all()?.to_vec1_f32()?;
        let mut mat = vec![0f32; total * voice_dim];
        mat[..tv * voice_dim].copy_from_slice(&vt_host);
        let mat = Tensor::from_vec_f32(mat, (total, voice_dim))?.to_device(&self.device)?;
        let proj = mat.matmul_t(&self.speaker_proj)?; // [total, dim]
                                                      // Mask: rows < tv keep proj, else learnt_padding.
        let maskf: Vec<f32> = (0..total).map(|r| if r < tv { 1.0 } else { 0.0 }).collect();
        let m = Tensor::from_vec_f32(maskf, (total, 1))?.to_device(&self.device)?;
        let inv = m.affine(-1.0, 1.0)?; // 1 - mask
        let pad = self.speaker_pad.reshape((1, self.dim))?;
        let cross = proj.broadcast_mul(&m)?.add(&pad.broadcast_mul(&inv)?)?;
        cross.add(&self.sin_embedding(total)?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Load a voice embedding `[C, T_voice]` from a `kyutai/tts-voices` `.safetensors`, where
/// it is stored as `[1, C, T_voice]`.
pub fn load_voice(path: &str, device: &Device) -> Result<Tensor> {
    let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, device)? };
    let [batch, width, len] = vb.shape_of("speaker_wavs")?[..] else {
        return Err(crate::tensor::Error(format!(
            "{path}: speaker_wavs is not [1, C, T]"
        )));
    };
    vb.get((batch, width, len), "speaker_wavs")?
        .reshape((width, len))
}
