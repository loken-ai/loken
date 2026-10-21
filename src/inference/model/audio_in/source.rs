//! Where an audio tower's tensors come from.
//!
//! Both sources are read under the names an `mmproj` GGUF gives them. A checkpoint's own
//! names (`audio_tower.layers.N.self_attn.q_proj.weight` and the rest) are translated to
//! those, so the encoder and the projector are written once.

use crate::tensor::quantized::gguf_file::MappedGguf;
use crate::tensor::safetensors::MmapedSafetensors;
use crate::tensor::{DType, Device, Error, Result, Tensor};

/// A file or set of files holding one tower's weights.
pub enum Source {
    Gguf(MappedGguf),
    Checkpoint(MmapedSafetensors),
    /// A checkpoint in the vendor's own naming, which stores no position table: the encoder's
    /// positions are the fixed sinusoids of `positions` rows.
    Native {
        st: MmapedSafetensors,
        positions: usize,
    },
}

/// The prefix a native layout gives its speech tower's tensors.
pub const NATIVE_TOWER: &str = "mm_whisper_embeddings.";

/// The name a native layout gives the tensor an `mmproj` calls `name`, `None` for a name the
/// translation does not cover (including the position table, which is computed).
pub fn native_name(name: &str) -> Option<String> {
    let (stem, kind) = name.rsplit_once('.')?;
    let encoder = format!("{NATIVE_TOWER}whisper_encoder");
    if let Some(n) = stem.strip_prefix("a.conv1d.") {
        let i = n.parse::<usize>().ok()?.checked_sub(1)?;
        return Some(format!("{encoder}.conv_layers.{i}.{kind}"));
    }
    if stem == "a.post_ln" {
        return Some(format!("{encoder}.transformer.norm.{kind}"));
    }
    if let Some(rest) = stem.strip_prefix("a.blk.") {
        let (n, part) = rest.split_once('.')?;
        let part = match part {
            "attn_q" => "attention.wq",
            "attn_k" => "attention.wk",
            "attn_v" => "attention.wv",
            "attn_out" => "attention.wo",
            "ln1" => "attention_norm",
            "ln2" => "ffn_norm",
            "ffn_up" => "feed_forward.w1",
            "ffn_down" => "feed_forward.w2",
            _ => return None,
        };
        return Some(format!("{encoder}.transformer.layers.{n}.{part}.{kind}"));
    }
    let index = match stem {
        "mm.a.mlp.1" => 0,
        "mm.a.mlp.2" => 2,
        _ => return None,
    };
    Some(format!(
        "{NATIVE_TOWER}audio_language_projection.{index}.{kind}"
    ))
}

/// The encoder's fixed positions `[len, channels]`: sines then cosines of geometrically spaced
/// timescales, as the Whisper encoder defines them.
pub fn sinusoids(len: usize, channels: usize, dev: &Device) -> Result<Tensor> {
    let half = channels / 2;
    let step = (10000f64).ln() / (half.max(2) - 1) as f64;
    let mut table = vec![0f32; len * channels];
    for t in 0..len {
        for i in 0..half {
            let angle = t as f64 * (-step * i as f64).exp();
            table[t * channels + i] = angle.sin() as f32;
            table[t * channels + half + i] = angle.cos() as f32;
        }
    }
    Tensor::from_vec(table, (len, channels), dev)
}

/// The name a checkpoint gives the tensor an `mmproj` calls `name`, `None` for a name the
/// translation does not cover.
pub fn checkpoint_name(name: &str) -> Option<String> {
    let (stem, kind) = name.rsplit_once('.')?;
    let tail = |s: &str| format!("{s}.{kind}");
    if let Some(n) = stem.strip_prefix("a.conv1d.") {
        return Some(format!("audio_tower.conv{n}.{kind}"));
    }
    if stem == "a.position_embd" {
        return Some(tail("audio_tower.embed_positions"));
    }
    if stem == "a.post_ln" {
        return Some(tail("audio_tower.layer_norm"));
    }
    if let Some(rest) = stem.strip_prefix("a.blk.") {
        let (n, part) = rest.split_once('.')?;
        let part = match part {
            "attn_q" => "self_attn.q_proj",
            "attn_k" => "self_attn.k_proj",
            "attn_v" => "self_attn.v_proj",
            "attn_out" => "self_attn.out_proj",
            "ln1" => "self_attn_layer_norm",
            "ln2" => "final_layer_norm",
            "ffn_up" => "fc1",
            "ffn_down" => "fc2",
            _ => return None,
        };
        return Some(format!("audio_tower.layers.{n}.{part}.{kind}"));
    }
    let part = match stem {
        "mm.a.mlp.1" => "linear_1",
        "mm.a.mlp.2" => "linear_2",
        "mm.a.norm_pre" => "ln_pre",
        "mm.a.norm_mid" => "ln_mid",
        _ => return None,
    };
    Some(format!("multi_modal_projector.{part}.{kind}"))
}

/// Reads tensors of a [`Source`] as F32 on one device.
pub(crate) struct TensorReader<'a> {
    pub source: &'a Source,
    pub dev: &'a Device,
}

impl TensorReader<'_> {
    /// The tensor `name` as F32 on the device, with its dimensions as the files give them:
    /// weights `[out, in]`, convolutions `[out, in, k]`.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        if let (Source::Native { st, positions }, "a.position_embd.weight") = (self.source, name) {
            let width = st
                .shape_of(&native_name("a.post_ln.weight").unwrap_or_default())
                .and_then(|d| d.first().copied())
                .ok_or_else(|| Error(format!("{name}: the encoder's width is unknown")))?;
            return sinusoids(*positions, width, self.dev);
        }
        // CONVERSION: an audio tower's weights are held F32; the tower runs once per clip.
        let host = match self.source {
            Source::Gguf(f) => f
                .tensor(name, &Device::Cpu)
                .map_err(|e| Error(format!("{name}: {e}")))?
                .dequantize(&Device::Cpu)?,
            Source::Checkpoint(st) | Source::Native { st, .. } => {
                let n = if matches!(self.source, Source::Native { .. }) {
                    native_name(name)
                } else {
                    checkpoint_name(name)
                }
                .ok_or_else(|| Error(format!("{name}: no checkpoint name")))?;
                st.load(&n, &Device::Cpu)
                    .map_err(|e| Error(format!("{n}: {e}")))?
            }
        };
        // CONVERSION: the checkpoint's dtype widens to the F32 the tower is held in.
        host.to_dtype(DType::F32)?.to_device(self.dev)
    }

    /// A vector tensor (a bias or a norm weight), flattened to `[n]`.
    pub fn vector(&self, name: &str) -> Result<Tensor> {
        let t = self.get(name)?;
        let n = t.elem_count();
        t.reshape((n,))
    }

    /// A matrix tensor, checked against the shape the geometry implies.
    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Tensor> {
        let t = self.get(name)?;
        if t.dims() != [rows, cols] {
            return Err(Error(format!(
                "{name}: shape {:?}, the geometry implies [{rows}, {cols}]",
                t.dims()
            )));
        }
        Ok(t)
    }

    /// The shape of a tensor without reading it.
    pub fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        match self.source {
            Source::Gguf(f) => f.tensor_infos.get(name).map(|i| i.shape.dims().to_vec()),
            Source::Checkpoint(st) => st.shape_of(&checkpoint_name(name)?),
            Source::Native { st, positions } => {
                if name == "a.position_embd.weight" {
                    let width = st
                        .shape_of(&native_name("a.post_ln.weight")?)?
                        .first()
                        .copied()?;
                    return Some(vec![*positions, width]);
                }
                st.shape_of(&native_name(name)?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{checkpoint_name, native_name, sinusoids};
    use crate::tensor::Device;

    #[test]
    fn a_checkpoint_name_is_translated_part_by_part() {
        let t = |a: &str, b: &str| assert_eq!(checkpoint_name(a).as_deref(), Some(b));
        t("a.conv1d.2.bias", "audio_tower.conv2.bias");
        t(
            "a.position_embd.weight",
            "audio_tower.embed_positions.weight",
        );
        t("a.post_ln.bias", "audio_tower.layer_norm.bias");
        t(
            "a.blk.7.attn_out.weight",
            "audio_tower.layers.7.self_attn.out_proj.weight",
        );
        t("a.blk.7.ffn_down.bias", "audio_tower.layers.7.fc2.bias");
        t("mm.a.mlp.1.weight", "multi_modal_projector.linear_1.weight");
        t(
            "mm.a.norm_mid.weight",
            "multi_modal_projector.ln_mid.weight",
        );
        assert_eq!(checkpoint_name("token_embd.weight"), None);
    }

    #[test]
    fn a_native_name_is_translated_part_by_part() {
        let t = |a: &str, b: &str| assert_eq!(native_name(a).as_deref(), Some(b));
        let e = "mm_whisper_embeddings.whisper_encoder";
        t("a.conv1d.1.bias", &format!("{e}.conv_layers.0.bias"));
        t("a.conv1d.2.weight", &format!("{e}.conv_layers.1.weight"));
        t("a.post_ln.bias", &format!("{e}.transformer.norm.bias"));
        t(
            "a.blk.7.attn_out.weight",
            &format!("{e}.transformer.layers.7.attention.wo.weight"),
        );
        t(
            "a.blk.7.ffn_down.bias",
            &format!("{e}.transformer.layers.7.feed_forward.w2.bias"),
        );
        t(
            "mm.a.mlp.2.weight",
            "mm_whisper_embeddings.audio_language_projection.2.weight",
        );
        assert_eq!(native_name("a.position_embd.weight"), None);
    }

    #[test]
    fn the_sinusoids_are_sines_then_cosines() {
        let t = sinusoids(3, 4, &Device::Cpu).unwrap();
        let v = t.to_vec2::<f32>().unwrap();
        assert_eq!(v[0], vec![0.0, 0.0, 1.0, 1.0]);
        // Two timescales: 1 and 1/10000 (the geometric step spans 10000 over one interval).
        assert!((v[1][0] - 1f32.sin()).abs() < 1e-6);
        assert!((v[1][1] - 1e-4f32.sin()).abs() < 1e-6);
        assert!((v[2][2] - 2f32.cos()).abs() < 1e-6);
    }
}
