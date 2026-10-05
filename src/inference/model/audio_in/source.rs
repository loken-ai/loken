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
        let host = match self.source {
            Source::Gguf(f) => f
                .tensor(name, &Device::Cpu)
                .map_err(|e| Error(format!("{name}: {e}")))?
                .dequantize(&Device::Cpu)?,
            Source::Checkpoint(st) => {
                let n = checkpoint_name(name)
                    .ok_or_else(|| Error(format!("{name}: no checkpoint name")))?;
                st.load(&n, &Device::Cpu)
                    .map_err(|e| Error(format!("{n}: {e}")))?
            }
        };
        // CONVERSION: an audio tower's weights are held F32; the tower runs once per clip.
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::checkpoint_name;

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
}
