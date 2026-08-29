//! Kyutai `tts-1.6b-en_fr` - the conditioner (component 7a of the port).
//!
//! Produces the two conditioning tensors the Helium LM consumes each step:
//!   - `condition_sum` [1, 2048]  = control-LUT + cfg-LUT  (a shared additive offset),
//!   - `condition_cross` [625, 2048] = speaker-voice projection + sinusoidal position emb
//!     (the cross-attention source `ca`).
//!
//! speaker_wavs (TensorConditioner): a precomputed voice embedding `[512, T_voice]` from a
//! `kyutai/tts-voices` `.safetensors` (key `speaker_wavs`, shape `[1, 512, T_voice]`) is
//! placed in speaker slot 0 of `MAX_SPEAKERS` (5), projected 512->2048, masked positions
//! filled with `learnt_padding`, then a sin/cos positional embedding is added (fuser
//! `cross_attention_pos_emb`, scale 1). cfg/control (LUTConditioner): a class index ->
//! `embed` lookup -> `output_proj` -> 2048; cfg "2.0" = index 2, control "ok" = index 0.

use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

const DIM: usize = 2048;
const VOICE_DIM: usize = 512;
const CFG_DIM: usize = 16;
const MAX_SPEAKERS: usize = 5;
const MAX_PERIOD: f32 = 10_000.0;

/// cfg_coef -> LUT index (valid_cfg_conditionings: 1.0..4.0 by 0.5).
pub fn cfg_index(cfg_coef: f32) -> Option<usize> {
    let steps = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    steps.iter().position(|&v| (v - cfg_coef).abs() < 1e-4)
}

pub struct KyutaiConditioner {
    speaker_proj: Tensor,  // [2048, 512]
    speaker_pad: Tensor,   // [2048] learnt_padding
    cfg_embed: Tensor,     // [8, 16]
    cfg_proj: Tensor,      // [2048, 16]
    control_embed: Tensor, // [2, 2048]
    control_proj: Tensor,  // [2048, 2048]
    device: Device,
}
impl KyutaiConditioner {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let c = vb.pp("condition_provider").pp("conditioners");
        Ok(Self {
            speaker_proj: c.get((DIM, VOICE_DIM), "speaker_wavs.output_proj.weight")?,
            speaker_pad: c
                .get((1, 1, DIM), "speaker_wavs.learnt_padding")?
                .reshape(DIM)?,
            cfg_embed: c.get((8, CFG_DIM), "cfg.embed.weight")?,
            cfg_proj: c.get((DIM, CFG_DIM), "cfg.output_proj.weight")?,
            control_embed: c.get((2, DIM), "control.embed.weight")?,
            control_proj: c.get((DIM, DIM), "control.output_proj.weight")?,
            device,
        })
    }

    fn lut(&self, embed: &Tensor, proj: &Tensor, idx: usize) -> Result<Tensor> {
        let id = Tensor::from_vec_u32(vec![idx as u32], vec![1])?.to_device(&self.device)?;
        embed.index_select(&id, 0)?.matmul_t(proj) // [1, DIM]
    }

    /// `condition_sum` [1, DIM] = control("ok") + cfg(cfg_coef). Both LUT lookups are
    /// masked True (values always provided) so no learnt_padding applies.
    pub fn condition_sum(&self, cfg_coef: f32) -> Result<Tensor> {
        let ci = cfg_index(cfg_coef).expect("unsupported cfg_coef");
        let cfg = self.lut(&self.cfg_embed, &self.cfg_proj, ci)?;
        let ctrl = self.lut(&self.control_embed, &self.control_proj, 0)?;
        ctrl.add(&cfg)
    }

    /// Sinusoidal position embedding `[T, DIM]` = cat(cos(phase), sin(phase)).
    fn sin_embedding(&self, t: usize) -> Result<Tensor> {
        let half = DIM / 2;
        let mut v = vec![0f32; t * DIM];
        for p in 0..t {
            for i in 0..half {
                let phase = p as f32 / MAX_PERIOD.powf(i as f32 / (half as f32 - 1.0));
                v[p * DIM + i] = phase.cos();
                v[p * DIM + half + i] = phase.sin();
            }
        }
        Tensor::from_vec_f32(v, (t, DIM))?.to_device(&self.device)
    }

    /// `condition_cross` [MAX_SPEAKERS*T_voice, DIM]. `voice: [VOICE_DIM, T_voice]`
    /// (the raw `speaker_wavs` tensor). Slot 0 is filled, slots 1..4 are learnt_padding;
    /// a sinusoidal position embedding is added over the whole length.
    pub fn condition_cross(&self, voice: &Tensor) -> Result<Tensor> {
        let (vd, tv) = voice.shape().dims2()?;
        assert_eq!(vd, VOICE_DIM);
        let total = MAX_SPEAKERS * tv;
        // Build [total, VOICE_DIM]: rows 0..tv = voice transposed, rest zero.
        let vt = voice.transpose(0, 1)?.contiguous()?; // [tv, VOICE_DIM]
        let vt_host = vt.flatten_all()?.to_vec1_f32()?;
        let mut mat = vec![0f32; total * VOICE_DIM];
        mat[..tv * VOICE_DIM].copy_from_slice(&vt_host);
        let mat = Tensor::from_vec_f32(mat, (total, VOICE_DIM))?.to_device(&self.device)?;
        let proj = mat.matmul_t(&self.speaker_proj)?; // [total, DIM]
                                                      // Mask: rows < tv keep proj, else learnt_padding.
        let maskf: Vec<f32> = (0..total).map(|r| if r < tv { 1.0 } else { 0.0 }).collect();
        let m = Tensor::from_vec_f32(maskf, (total, 1))?.to_device(&self.device)?;
        let inv = m.affine(-1.0, 1.0)?; // 1 - mask
        let pad = self.speaker_pad.reshape((1, DIM))?;
        let cross = proj.broadcast_mul(&m)?.add(&pad.broadcast_mul(&inv)?)?;
        cross.add(&self.sin_embedding(total)?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Read the `speaker_wavs` tensor shape `[1, 512, T_voice]` from a safetensors header
/// (8-byte LE header length + JSON), returning `T_voice`.
fn voice_len(path: &str) -> Result<usize> {
    let bytes = std::fs::read(path).map_err(|e| crate::tensor::Error(e.to_string()))?;
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr = std::str::from_utf8(&bytes[8..8 + hlen])
        .map_err(|e| crate::tensor::Error(e.to_string()))?;
    // find "speaker_wavs" ... "shape":[1,512,T]
    let key = hdr
        .find("\"speaker_wavs\"")
        .ok_or_else(|| crate::tensor::Error("no speaker_wavs".into()))?;
    let shp = hdr[key..]
        .find("\"shape\":[")
        .ok_or_else(|| crate::tensor::Error("no shape".into()))?
        + key
        + 9;
    let end = hdr[shp..].find(']').unwrap() + shp;
    let dims: Vec<usize> = hdr[shp..end]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Ok(*dims.last().unwrap())
}

/// Load a voice embedding `[512, T_voice]` from a `kyutai/tts-voices` `.safetensors`.
pub fn load_voice(path: &str, device: &Device) -> Result<Tensor> {
    let tv = voice_len(path)?;
    let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, device)? };
    // stored as [1, 512, T_voice]; drop the batch dim.
    vb.get((1, VOICE_DIM, tv), "speaker_wavs")?
        .reshape((VOICE_DIM, tv))
}
