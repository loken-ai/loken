//! The dimensions of a latent attention layer and of the sparse indexer beside it, read from the
//! file's header.
//!
//! A layer projects its hidden state to a low-rank query and to 1 latent row per token. The
//! latent row is the only thing the layer caches: the keys and values of every head are
//! rebuilt from it. A sparse indexer may select, per query, the tokens the layer attends to.

use super::config::{meta_u32, meta_u32_array};
use super::gguf_file;

/// The sparse indexer of a model: how it scores tokens and which layers run it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMeta {
    /// Query heads of the indexer; the key is shared by all of them.
    pub heads: usize,
    /// Width of an indexer head.
    pub head_dim: usize,
    /// Tokens a query attends to at most.
    pub top_k: usize,
    /// Per layer: `true` when the layer runs its own indexer, `false` when it reuses the
    /// selection of the nearest earlier layer that does.
    pub full: Vec<bool>,
}

impl IndexMeta {
    /// The layer whose selection layer `il` uses: itself when it runs an indexer, else the
    /// nearest earlier one that does. `None` when no layer up to `il` runs one.
    pub fn source_layer(&self, il: usize) -> Option<usize> {
        (0..=il.min(self.full.len().saturating_sub(1)))
            .rev()
            .find(|&l| self.full[l])
    }
}

/// The latent attention's dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaMeta {
    /// Rank of the query's low-rank factor; `None` for a query projected in 1 step.
    pub q_lora_rank: Option<usize>,
    /// Width of the latent row a token caches beside its rotated position key.
    pub kv_lora_rank: usize,
    /// Width of a head's key that is not rotated.
    pub nope_dim: usize,
    /// Width of the rotated part of a head's key, shared by all heads.
    pub rope_dim: usize,
    /// Width of a head's value.
    pub v_dim: usize,
    /// The sparse indexer, when the file declares one.
    pub index: Option<IndexMeta>,
}

impl MlaMeta {
    /// Width of a head's query and key.
    pub fn qk_dim(&self) -> usize {
        self.nope_dim + self.rope_dim
    }

    /// The metadata of `arch`; `Ok(None)` for a file without a latent rank, an error for a file
    /// whose keys contradict each other.
    pub fn from_gguf(
        content: &gguf_file::Content,
        arch: &str,
    ) -> crate::tensor::Result<Option<Self>> {
        let Some(kv_lora_rank) = meta_u32(content, arch, "attention.kv_lora_rank") else {
            return Ok(None);
        };
        let need = |key: &str| {
            meta_u32(content, arch, key)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    crate::tensor::Error::msg(format!(
                        "latent attention without the key {arch}.{key}"
                    ))
                })
        };
        let rope_dim = need("rope.dimension_count")?;
        let key_len = need("attention.key_length_mla")?;
        let v_dim = need("attention.value_length_mla")?;
        let n_layers = need("block_count")?;
        if rope_dim == 0 || rope_dim % 2 != 0 || rope_dim >= key_len {
            return Err(crate::tensor::Error::msg(format!(
                "latent attention keys of {key_len} lanes cannot hold {rope_dim} rotated lanes"
            )));
        }
        let index = match meta_u32(content, arch, "attention.indexer.head_count") {
            None => None,
            Some(heads) => {
                let head_dim = need("attention.indexer.key_length")?;
                let top_k = need("attention.indexer.top_k")?;
                // A file that lists no layer types runs the indexer on every layer.
                let full = match meta_u32_array(content, arch, "attention.indexer.types") {
                    Some(types) if types.len() == n_layers => {
                        types.into_iter().map(|t| t != 0).collect()
                    }
                    Some(types) => {
                        return Err(crate::tensor::Error::msg(format!(
                            "{} indexer layer types for {n_layers} layers",
                            types.len()
                        )))
                    }
                    None => vec![true; n_layers],
                };
                if full.first() != Some(&true) {
                    return Err(crate::tensor::Error::msg(
                        "the first layer reuses the selection of an indexer no layer before it ran",
                    ));
                }
                if head_dim < rope_dim || heads == 0 || top_k == 0 {
                    return Err(crate::tensor::Error::msg(format!(
                        "an indexer of {heads} heads of {head_dim} lanes cannot rotate {rope_dim}"
                    )));
                }
                Some(IndexMeta {
                    heads: heads as usize,
                    head_dim,
                    top_k,
                    full,
                })
            }
        };
        Ok(Some(Self {
            q_lora_rank: meta_u32(content, arch, "attention.q_lora_rank").map(|r| r as usize),
            kv_lora_rank: kv_lora_rank as usize,
            nope_dim: key_len - rope_dim,
            rope_dim,
            v_dim,
            index,
        }))
    }
}

/// The per-layer indexer kinds of a configuration: its own list when it gives one, else the
/// pattern string (`F` runs the indexer, `S` reuses), else the schedule `freq` and `offset`
/// describe, under which the layers before `offset` run the indexer and then every `freq`-th
/// one does.
pub fn indexer_layers(
    n_layers: usize,
    listed: Option<&[String]>,
    pattern: Option<&str>,
    freq: usize,
    offset: usize,
) -> Vec<bool> {
    if let Some(list) = listed {
        return list.iter().map(|t| t == "full").collect();
    }
    if let Some(p) = pattern {
        return p.chars().map(|c| c == 'F').collect();
    }
    let freq = freq.max(1);
    (0..n_layers)
        .map(|i| (i + 1).saturating_sub(offset) % freq == 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == 'F').collect()
    }

    #[test]
    fn the_schedule_runs_the_first_layers_then_every_freq_th() {
        let types = indexer_layers(78, None, None, 4, 3);
        let full: Vec<usize> = (0..78).filter(|&i| types[i]).collect();
        let mut want = vec![0, 1, 2];
        want.extend((6..78).step_by(4));
        assert_eq!(full, want);
        assert_eq!(full.len(), 21);
    }

    #[test]
    fn a_pattern_and_a_list_override_the_schedule() {
        assert_eq!(indexer_layers(4, None, Some("FSSF"), 1, 0), flags("FSSF"));
        let listed = ["full", "shared", "full"].map(String::from);
        assert_eq!(
            indexer_layers(3, Some(&listed), Some("SSS"), 1, 0),
            flags("FSF")
        );
    }

    #[test]
    fn the_source_of_a_shared_layer_is_the_nearest_earlier_full_one() {
        let meta = IndexMeta {
            heads: 2,
            head_dim: 8,
            top_k: 4,
            full: flags("FSSFS"),
        };
        let sources: Vec<_> = (0..5).map(|l| meta.source_layer(l)).collect();
        assert_eq!(sources, [Some(0), Some(0), Some(0), Some(3), Some(3)]);
    }
}

#[cfg(test)]
mod release_config_tests {
    use super::*;
    use crate::inference::generic_transformer::GenericTransformerConfig;
    use crate::inference::load::awq_loader::{
        config_from_value, synth_gguf_content, AwqConfigJson, AwqTensorPresence,
    };
    use crate::tensor::quantized::gguf_file::Value;

    /// The header keys of a latent attention do not depend on the tag they are written under.
    const ARCH: &str = "llama";

    /// The `config.json` of the release, its list of unconverted modules cut short.
    fn release() -> AwqConfigJson {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/vectors/glm/config/config.json");
        let v = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        config_from_value(v, "NoClass".into(), ARCH.into()).unwrap()
    }

    fn full_layers() -> Vec<usize> {
        let mut want = vec![0, 1, 2];
        want.extend((6..75).step_by(4));
        want
    }

    #[test]
    fn the_release_configuration_is_read_as_a_latent_attention_with_an_indexer() {
        let cfg = release();
        assert_eq!(cfg.gguf_arch, ARCH);
        assert_eq!(cfg.n_layers, 78);
        assert_eq!(cfg.head_dim, Some(256));
        let content = synth_gguf_content(&cfg, AwqTensorPresence::default());
        let meta = MlaMeta::from_gguf(&content, ARCH).unwrap().unwrap();
        assert_eq!(meta.q_lora_rank, Some(2048));
        assert_eq!(meta.kv_lora_rank, 512);
        assert_eq!((meta.nope_dim, meta.rope_dim, meta.v_dim), (192, 64, 256));
        assert_eq!(meta.qk_dim(), 256);
        let index = meta.index.unwrap();
        assert_eq!((index.heads, index.head_dim, index.top_k), (32, 128, 2048));
        let full: Vec<usize> = (0..78).filter(|&i| index.full[i]).collect();
        assert_eq!(full, full_layers());
        assert_eq!(full.len(), 21);
        assert_eq!(index.source_layer(77), Some(74));
    }

    #[test]
    fn the_schedule_gives_the_listed_layers_when_the_list_is_absent() {
        let mut cfg = release();
        cfg.root.as_object_mut().unwrap().remove("indexer_types");
        let content = synth_gguf_content(&cfg, AwqTensorPresence::default());
        let meta = MlaMeta::from_gguf(&content, ARCH).unwrap().unwrap();
        let index = meta.index.unwrap();
        let full: Vec<usize> = (0..78).filter(|&i| index.full[i]).collect();
        assert_eq!(full, full_layers());
    }

    #[test]
    fn the_mixture_fields_reach_the_generic_configuration() {
        let cfg = release();
        let content = synth_gguf_content(&cfg, AwqTensorPresence::default());
        let config = GenericTransformerConfig::from_gguf(&content, ARCH).unwrap();
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.rope_dim, Some(64));
        assert_eq!(config.rope_freq_base, 8_000_000.0);
        assert!(config.mla.is_some());
        assert!(config.expert_sigmoid_gating);
        assert!(config.expert_weights_norm);
        assert_eq!(config.expert_weights_scale, 2.5);
        assert_eq!(config.flags.n_experts, 256);
        assert_eq!(config.flags.n_experts_used, 8);
        assert_eq!(config.flags.expert_ffn_dim, 2048);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-12);
        assert!(config.indexer.is_none(), "not the block-sparse indexer");
    }

    #[test]
    fn a_file_with_contradicting_keys_is_refused() {
        let cfg = release();
        let mut content = synth_gguf_content(&cfg, AwqTensorPresence::default());
        let key = format!("{ARCH}.attention.indexer.types");
        content
            .metadata
            .insert(key.clone(), Value::Array(vec![Value::Bool(false); 78]));
        assert!(MlaMeta::from_gguf(&content, ARCH).is_err());
        content
            .metadata
            .insert(key.clone(), Value::Array(vec![Value::Bool(true); 3]));
        assert!(MlaMeta::from_gguf(&content, ARCH).is_err());
    }
}
