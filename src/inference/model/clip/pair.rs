//! A text tower and a vision tower trained to put a caption and its image at the same
//! point, read from a Hugging Face checkpoint directory: `config.json` with a `text_config`
//! and a `vision_config`, the safetensors, `tokenizer.json` and `preprocessor_config.json`.
//!
//! Two layouts, told apart by the tensors the checkpoint stores:
//!
//! | Layout    | Marker tensors                          | Text                                                     | Image                                     |
//! |-----------|-----------------------------------------|----------------------------------------------------------|-------------------------------------------|
//! | Projected | `text_projection`, `visual_projection` | causal, pooled at the end-of-text token, projected       | class token through `post_layernorm`, projected |
//! | Headed    | `text_model.head`, `vision_model.head` | bidirectional over a row padded to the position table, last position through the head | attention-pool head |
//!
//! Every vector is L2-normalised. Runs on the host in f32.

use crate::inference::model::clip::preprocess::Preprocess;
use crate::inference::model::clip::{text, vision};
use crate::tensor::layer::{linear, linear_no_bias, Linear};
use crate::tensor::ops::Activation;
use crate::tensor::{DType, Device, Tensor, VarBuilder};
use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Which of the two layouts a checkpoint stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Projected,
    Headed,
}

impl Layout {
    /// The layout the tensor names mark, if any.
    fn of(has: impl Fn(&str) -> bool) -> Option<Self> {
        let tower = has("text_model.embeddings.token_embedding.weight")
            && has("vision_model.embeddings.patch_embedding.weight");
        if !tower {
            None
        } else if has("text_projection.weight") && has("visual_projection.weight") {
            Some(Self::Projected)
        } else if has("text_model.head.weight") && has("vision_model.head.probe") {
            Some(Self::Headed)
        } else {
            None
        }
    }

    /// What a tower config leaves out, as transformers' config classes default it for the
    /// towers of this layout: `(text heads, vision heads, norm epsilon, activation)`.
    fn defaults(self) -> (usize, usize, f32, Activation) {
        match self {
            Self::Projected => (8, 12, 1e-5, Activation::QuickGelu),
            Self::Headed => (12, 12, 1e-6, Activation::GeluPytorchTanh),
        }
    }
}

/// Where the text tower's sequence is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextPool {
    /// The highest token id of the row: transformers' rule for a config whose `eos_token_id`
    /// is [`LEGACY_EOS_ID`].
    HighestId,
    /// The first position holding this id.
    FirstOf(u32),
    /// The last position of a row padded with this id to the position table.
    Last { pad: u32 },
}

/// The `eos_token_id` early CLIP configs carry; transformers pools those at the highest id.
const LEGACY_EOS_ID: u64 = 2;

impl TextPool {
    /// The position `ids` is pooled at.
    fn position(self, ids: &[u32]) -> Result<usize> {
        match self {
            Self::HighestId => {
                let top = ids.iter().copied().max().unwrap_or(0);
                Ok(ids.iter().position(|&t| t == top).unwrap_or(0))
            }
            Self::FirstOf(eos) => ids
                .iter()
                .position(|&t| t == eos)
                .ok_or_else(|| anyhow!("the tokenizer added no end-of-text token {eos}")),
            Self::Last { .. } => Ok(ids.len().saturating_sub(1)),
        }
    }
}

/// The ids a text tower of `length` positions reads: cut to the table keeping the final
/// (end-of-text) token, and padded to it when the pooling reads the last position.
fn fit_ids(mut ids: Vec<u32>, length: usize, pool: TextPool) -> Vec<u32> {
    if ids.len() > length {
        let last = ids[ids.len() - 1];
        ids.truncate(length - 1);
        ids.push(last);
    }
    if let TextPool::Last { pad } = pool {
        ids.resize(length, pad);
    }
    ids
}

/// The id of the token `tokenizer_config.json` names under `key`, given as a string or as an
/// object with a `content`.
fn named_token(dir: &Path, tokenizer: &Tokenizer, key: &str) -> Option<u32> {
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?)
            .ok()?;
    let token = match config.get(key)? {
        Value::String(s) => s.as_str(),
        other => other.get("content")?.as_str()?,
    };
    tokenizer.token_to_id(token)
}

/// The safetensors files of a checkpoint directory, sorted.
fn weight_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// The bytes of a checkpoint directory's safetensors files.
pub fn weight_bytes(dir: &Path) -> u64 {
    weight_files(dir)
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum()
}

/// Whether `dir` holds a text-image pair this module loads: the two tower configs, the
/// tokenizer, the preprocessor config, and tensors of one of the two layouts.
pub fn claims_checkpoint_dir(dir: &Path) -> bool {
    if !dir.is_dir()
        || !dir.join("tokenizer.json").is_file()
        || !dir.join("preprocessor_config.json").is_file()
    {
        return false;
    }
    let towers = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .is_some_and(|c| c.get("text_config").is_some() && c.get("vision_config").is_some());
    if !towers {
        return false;
    }
    let files = weight_files(dir);
    if files.is_empty() {
        return false;
    }
    crate::inference::load::shapes::Shapes::read(&files)
        .is_ok_and(|shapes| Layout::of(|n| shapes.contains(n)).is_some())
}

/// A tower config's value under `key`, else the layout's default.
fn usize_or(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(Value::as_u64)
        .map_or(default, |v| v as usize)
}

fn eps_or(config: &Value, default: f32) -> f32 {
    config
        .get("layer_norm_eps")
        .and_then(Value::as_f64)
        .map_or(default, |v| v as f32)
}

fn activation_or(config: &Value, default: Activation) -> Result<Activation> {
    match config.get("hidden_act") {
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| anyhow!("activation {v} is not one this server runs: {e}")),
        None => Ok(default),
    }
}

/// A text-image pair, resident on the host.
pub struct DualEncoder {
    text: text::Transformer,
    text_head: Linear,
    text_pool: TextPool,
    /// Rows of the text tower's position table: the longest sequence it reads.
    text_length: usize,
    tokenizer: Tokenizer,
    vision: vision::VisionTower,
    /// The projection after the vision tower's pooling, when the layout has one.
    vision_head: Option<Linear>,
    preprocess: Preprocess,
    dim: usize,
}

impl DualEncoder {
    /// Load the pair stored in `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let config: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let empty = Value::Object(Default::default());
        let text_config = config.get("text_config").unwrap_or(&empty);
        let vision_config = config.get("vision_config").unwrap_or(&empty);
        let files = weight_files(dir);
        let device = Device::Cpu;
        let vb = unsafe { VarBuilder::from_files(&files, DType::F32, &device) }?;
        let layout = Layout::of(|n| vb.contains(n)).ok_or_else(|| {
            anyhow!(
                "{}: no text-image tower layout in its tensors",
                dir.display()
            )
        })?;
        let (text_heads, vision_heads, eps, activation) = layout.defaults();

        let t = vb.pp("text_model");
        let token = t
            .pp("embeddings")
            .pp("token_embedding")
            .shape_of("weight")?;
        let (vocab_size, embed_dim) = (token[0], token[1]);
        let heads = usize_or(text_config, "num_attention_heads", text_heads);
        if heads == 0 || embed_dim % heads != 0 {
            bail!("text tower: {embed_dim} wide does not split into {heads} heads");
        }
        let text_cfg = text::Config {
            vocab_size,
            embed_dim,
            activation: activation_or(text_config, activation)?,
            intermediate_size: t
                .pp("encoder")
                .pp("layers")
                .pp("0")
                .pp("mlp")
                .pp("fc1")
                .shape_of("weight")?[0],
            max_position_embeddings: t
                .pp("embeddings")
                .pp("position_embedding")
                .shape_of("weight")?[0],
            num_hidden_layers: t.pp("encoder").count_children("layers"),
            num_attention_heads: heads,
            layer_norm_eps: eps_or(text_config, eps),
            causal: layout == Layout::Projected,
        };
        let text_length = text_cfg.max_position_embeddings;
        let text_tower = text::Transformer::new(t.clone(), &text_cfg)?;

        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("{}: tokenizer: {e}", dir.display()))?;
        // The pooling below decides the length; a padding or truncation the file carries
        // would decide it first.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("tokenizer: {e}"))?;

        let text_head_vb = match layout {
            Layout::Projected => vb.pp("text_projection"),
            Layout::Headed => t.pp("head"),
        };
        let text_dim = text_head_vb.shape_of("weight")?[0];
        let (text_head, text_pool) = match layout {
            Layout::Projected => {
                let eos = match text_config.get("eos_token_id").and_then(Value::as_u64) {
                    Some(id) => Some(id),
                    None => named_token(dir, &tokenizer, "eos_token").map(u64::from),
                };
                let pool = match eos {
                    Some(LEGACY_EOS_ID) => TextPool::HighestId,
                    Some(id) => TextPool::FirstOf(u32::try_from(id)?),
                    None => {
                        bail!("text tower: no end-of-text token in the config or the tokenizer")
                    }
                };
                (linear_no_bias(embed_dim, text_dim, &text_head_vb)?, pool)
            }
            Layout::Headed => {
                let pad = named_token(dir, &tokenizer, "pad_token")
                    .or_else(|| {
                        text_config
                            .get("pad_token_id")
                            .and_then(Value::as_u64)
                            .and_then(|id| u32::try_from(id).ok())
                    })
                    .ok_or_else(|| {
                        anyhow!("text tower: no padding token in the tokenizer or the config")
                    })?;
                (
                    linear(embed_dim, text_dim, &text_head_vb)?,
                    TextPool::Last { pad },
                )
            }
        };

        let vision_cfg = vision::Config {
            heads: usize_or(vision_config, "num_attention_heads", vision_heads),
            layer_norm_eps: eps_or(vision_config, eps),
            activation: activation_or(vision_config, activation)?,
            pooled: true,
        };
        let v = vb.pp("vision_model");
        let width = v
            .pp("embeddings")
            .pp("patch_embedding")
            .shape_of("weight")?[0];
        if vision_cfg.heads == 0 || width % vision_cfg.heads != 0 {
            bail!(
                "vision tower: {width} wide does not split into {} heads",
                vision_cfg.heads
            );
        }
        let vision_tower = vision::VisionTower::new(&v, &vision_cfg, &device)?;
        let (vision_head, image_dim) = match layout {
            Layout::Projected => {
                let p = vb.pp("visual_projection");
                let out = p.shape_of("weight")?[0];
                (Some(linear_no_bias(width, out, &p)?), out)
            }
            Layout::Headed => (None, width),
        };
        if image_dim != text_dim {
            bail!("the text vectors are {text_dim} wide and the image vectors {image_dim}");
        }

        let preprocess = Preprocess::from_json(&serde_json::from_str(&std::fs::read_to_string(
            dir.join("preprocessor_config.json"),
        )?)?)
        .map_err(|e| anyhow!("preprocessor_config.json: {e}"))?;
        let side = vision_tower.image_size();
        if preprocess.output_size() != Some((side, side)) {
            bail!(
                "preprocessor_config.json makes images of {:?}; the vision tower reads {side}x{side}",
                preprocess.output_size()
            );
        }
        Ok(Self {
            text: text_tower,
            text_head,
            text_pool,
            text_length,
            tokenizer,
            vision: vision_tower,
            vision_head,
            preprocess,
            dim: text_dim,
        })
    }

    /// Width of the vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn text_ids(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            bail!("the text has no tokens");
        }
        Ok(fit_ids(ids, self.text_length, self.text_pool))
    }

    /// One text -> its L2-normalised vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.text_ids(text)?;
        let at = self.text_pool.position(&ids)?;
        let states = self.text.final_states(&ids)?;
        let pooled = self.text_head.forward(&states.narrow(0, at, 1)?)?;
        let mut v = pooled.flatten_all()?.to_vec_f32();
        crate::inference::model::embedding::l2_normalize(&mut v);
        Ok(v)
    }

    /// One interleaved RGB8 image -> its L2-normalised vector.
    pub fn embed_image(&self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        let (pixels, ph, pw) = self.preprocess.apply(rgb, w, h).map_err(|e| anyhow!(e))?;
        let pixels = Tensor::from_vec_f32(pixels, vec![1, 3, ph, pw])?;
        let pooled = self.vision.pooled(&pixels)?;
        let pooled = match &self.vision_head {
            Some(head) => head.forward(&pooled)?,
            None => pooled,
        };
        let mut v = pooled.flatten_all()?.to_vec_f32();
        crate::inference::model::embedding::l2_normalize(&mut v);
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout is read off the tensor names; towers alone, or a projection without its
    /// pair, mark none.
    #[test]
    fn the_layout_is_read_off_the_tensors() {
        let towers = [
            "text_model.embeddings.token_embedding.weight",
            "vision_model.embeddings.patch_embedding.weight",
        ];
        let with = |extra: &[&str]| {
            let names: Vec<&str> = towers.iter().chain(extra).copied().collect();
            Layout::of(|n| names.contains(&n))
        };
        assert_eq!(
            with(&["text_projection.weight", "visual_projection.weight"]),
            Some(Layout::Projected)
        );
        assert_eq!(
            with(&["text_model.head.weight", "vision_model.head.probe"]),
            Some(Layout::Headed)
        );
        assert_eq!(with(&["text_projection.weight"]), None);
        assert_eq!(with(&[]), None);
        assert_eq!(Layout::of(|n| n == "text_projection.weight"), None);
    }

    /// A text over the table keeps its end-of-text token; a headed tower pads to the table.
    #[test]
    fn ids_are_cut_and_padded_to_the_table() {
        let pool = TextPool::FirstOf(9);
        assert_eq!(fit_ids(vec![1, 2, 3, 4, 9], 3, pool), vec![1, 2, 9]);
        assert_eq!(fit_ids(vec![1, 9], 3, pool), vec![1, 9]);
        let pad = TextPool::Last { pad: 0 };
        assert_eq!(fit_ids(vec![5, 1], 4, pad), vec![5, 1, 0, 0]);
        assert_eq!(fit_ids(vec![5, 6, 7, 8, 1], 4, pad), vec![5, 6, 7, 1]);
    }

    /// Each pooling reads its own position: the first highest id, the first end-of-text
    /// token, or the last row.
    #[test]
    fn each_pooling_reads_its_position() {
        let ids = [49406, 320, 49407, 49407];
        assert_eq!(TextPool::HighestId.position(&ids).unwrap(), 2);
        assert_eq!(TextPool::FirstOf(49407).position(&ids).unwrap(), 2);
        assert!(TextPool::FirstOf(7).position(&ids).is_err());
        assert_eq!(TextPool::Last { pad: 1 }.position(&ids).unwrap(), 3);
    }

    /// A config's own values win over the layout's defaults; a missing one takes them.
    #[test]
    fn config_values_win_over_defaults() {
        let c = serde_json::json!({"num_attention_heads": 16, "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh"});
        assert_eq!(usize_or(&c, "num_attention_heads", 12), 16);
        assert_eq!(eps_or(&c, 1e-5), 1e-6);
        assert_eq!(
            activation_or(&c, Activation::QuickGelu).unwrap(),
            Activation::GeluPytorchTanh
        );
        let empty = serde_json::json!({});
        assert_eq!(usize_or(&empty, "num_attention_heads", 12), 12);
        assert_eq!(
            activation_or(&empty, Activation::QuickGelu).unwrap(),
            Activation::QuickGelu
        );
        assert!(activation_or(
            &serde_json::json!({"hidden_act": "swish7"}),
            Activation::Gelu
        )
        .is_err());
    }

    /// Against the real checkpoints the test configuration's Hugging Face tree holds: a text
    /// and an image give unit vectors of one width, and a caption sits nearer its picture than
    /// a wrong caption does.
    #[test]
    #[ignore = "needs a CLIP or SigLIP snapshot in the Hugging Face tree"]
    fn a_caption_is_nearer_its_picture() {
        let hf = crate::config::Config::load_test().get_hf_models_dir();
        let snapshots: Vec<PathBuf> = [
            "openai/clip-vit-large-patch14",
            "google/siglip-base-patch16-224",
        ]
        .iter()
        .filter_map(|id| {
            let (org, name) = id.split_once('/')?;
            let hub = format!("models--{org}--{name}");
            [hf.clone(), hf.join("hub")].iter().find_map(|root| {
                let mut snaps: Vec<PathBuf> = std::fs::read_dir(root.join(&hub).join("snapshots"))
                    .ok()?
                    .flatten()
                    .map(|e| e.path())
                    .collect();
                snaps.sort();
                snaps.pop()
            })
        })
        .collect();
        if snapshots.is_empty() {
            println!(
                "no CLIP or SigLIP snapshot under {}; skipping",
                hf.display()
            );
            return;
        }
        for dir in snapshots {
            println!("{}", dir.display());
            assert!(claims_checkpoint_dir(&dir));
            let m = DualEncoder::load(&dir).expect("load");
            let (w, h) = (160usize, 120usize);
            let red: Vec<u8> = (0..w * h).flat_map(|_| [220u8, 20, 20]).collect();
            let img = m.embed_image(&red, w, h).unwrap();
            let right = m.embed_text("a red square").unwrap();
            let wrong = m.embed_text("a photo of a dog on a beach").unwrap();
            assert_eq!((img.len(), right.len()), (m.dim(), m.dim()));
            let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
            assert!((dot(&img, &img) - 1.0).abs() < 1e-4);
            println!(
                "right {:.4} wrong {:.4}",
                dot(&img, &right),
                dot(&img, &wrong)
            );
            assert!(dot(&img, &right) > dot(&img, &wrong));
        }
    }
}
