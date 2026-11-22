//! Part of `impl LlmEngine`: the classification head of the loaded model.
//!
//! The head reads the final hidden states the loaded model returns for its embeddings, so it
//! runs on the model the generic route placed.

use super::*;
use crate::inference::model::reranker::head::{self, ClassifierHead, Pool};

/// The head of the checkpoint the load opened, `None` when it holds none.
///
/// A GGUF file holds `cls` and `cls.output`, its labels under `{arch}.classifier.output_labels`.
/// A safetensors checkpoint holds `score` or `classifier`, its labels in `id2label`.
pub(super) fn read_classifier(
    content: &gguf_file::Content,
    arch: &str,
    mmap: &Mmap,
    awq: &Option<(String, Vec<String>)>,
    split: &Option<crate::tensor::quantized::gguf_source::SplitGguf>,
) -> Option<ClassifierHead> {
    use crate::tensor::quantized::gguf_source::{GgufMeta, GgufSource};
    let values = |t: Tensor| -> Option<(Vec<f32>, Vec<usize>)> {
        let dims = t.dims().to_vec();
        let v = t
            .to_dtype(crate::tensor::DType::F32)
            .ok()?
            .flatten_all()
            .ok()?
            .to_vec1::<f32>()
            .ok()?;
        Some((v, dims))
    };
    if let Some((dir, shards)) = awq {
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(shards) }.ok()?;
        let names = st.names();
        let fetch = |n: &str| {
            names
                .contains(&n)
                .then(|| st.load(n, &Device::Cpu).ok())
                .flatten()
                .and_then(values)
        };
        let layers = head::SAFETENSORS_HEADS
            .iter()
            .find_map(|stem| head::read_layers(&[(stem, false)], &fetch))?;
        let config: Option<serde_json::Value> =
            std::fs::read_to_string(std::path::Path::new(dir).join("config.json"))
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok());
        let labels = config.as_ref().and_then(head::id2label);
        return Some(ClassifierHead::new(layers, labels, Pool::Last, None));
    }
    let fetch = |n: &str| -> Option<(Vec<f32>, Vec<usize>)> {
        let q = match split {
            Some(s) => {
                s.info(n)?;
                s.tensor(n, &Device::Cpu).ok()?
            }
            None => {
                content.tensor_infos.get(n)?;
                content
                    .tensor(&mut std::io::Cursor::new(&mmap[..]), n, &Device::Cpu)
                    .ok()?
            }
        };
        values(q.dequantize(&Device::Cpu).ok()?)
    };
    let layers = head::read_layers(&head::GGUF_LAYERS, &fetch)?;
    let labels = content
        .metadata
        .get(&format!("{arch}{}", head::GGUF_LABELS_SUFFIX))
        .and_then(|v| v.to_vec().ok())
        .and_then(|a| {
            a.iter()
                .map(|v| v.to_string().ok().cloned())
                .collect::<Option<Vec<String>>>()
        });
    let pool = Pool::of_gguf(
        content
            .metadata
            .get(&format!("{arch}.pooling_type"))
            .and_then(|v| v.to_u64().ok()),
    );
    let template = get_gguf_string(content, head::GGUF_PAIR_TEMPLATE);
    Some(ClassifierHead::new(layers, labels, pool, template))
}

/// What a head reads: one text, or a pair of texts.
enum HeadInput {
    Text(String),
    Pair(String, String),
}

impl LlmEngine {
    /// The label names of the loaded model's classification head.
    /// `None` when no model is loaded or the model has no head.
    pub async fn classifier_labels(&self) -> Option<Vec<String>> {
        let guard = self.model_state.lock().await;
        guard
            .as_ref()?
            .classifier
            .as_ref()
            .map(|h| h.labels.clone())
    }

    /// The head's logits for `input`, read at the positions the head pools.
    pub async fn classify(&self, input: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut rows = self
            .head_logits(HeadInput::Text(input.to_string()), false)
            .await?;
        rows.pop().ok_or_else(|| "classify: no logits".into())
    }

    /// The head's logits for each token of `input`, in order.
    pub async fn classify_tokens(
        &self,
        input: &str,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        self.head_logits(HeadInput::Text(input.to_string()), true)
            .await
    }

    /// The head's logits for the pair `(query, document)`.
    ///
    /// The pair is written in the prompt the model declares, else encoded as a pair by the
    /// tokenizer.
    pub async fn classify_pair(
        &self,
        query: &str,
        document: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let input = HeadInput::Pair(query.to_string(), document.to_string());
        let mut rows = self.head_logits(input, false).await?;
        rows.pop().ok_or_else(|| "classify: no logits".into())
    }

    /// Whether the vocabulary holds the `yes` and `no` tokens the generative reranker reads.
    pub async fn answers_yes_no(&self) -> bool {
        let guard = self.model_state.lock().await;
        guard.as_ref().is_some_and(|s| {
            let has = |w: &str, ws: &str| {
                s.tokenizer.token_to_id(w).is_some() || s.tokenizer.token_to_id(ws).is_some()
            };
            has("yes", "Ġyes") && has("no", "Ġno")
        })
    }

    /// The head's logits for `input`: one row per token when `per_token`, else one row for
    /// the pooled state.
    async fn head_logits(
        &self,
        input: HeadInput,
        per_token: bool,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<Vec<Vec<f32>>> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("classify: no model loaded"))?;
            let head = state
                .classifier
                .clone()
                .ok_or_else(|| anyhow!("the model has no classification head"))?;
            let encoding = match &input {
                HeadInput::Text(t) => super::encode_prompt(&state.tokenizer, t),
                HeadInput::Pair(q, d) => match head.pair_prompt(q, d) {
                    Some(p) => super::encode_prompt(&state.tokenizer, &p),
                    None => state.tokenizer.encode((q.as_str(), d.as_str()), true),
                },
            }
            .map_err(|e| anyhow!("classify encode: {e}"))?;
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            if ids.is_empty() {
                return Err(anyhow!("classify: empty token sequence"));
            }
            let x = Tensor::new(ids.as_slice(), &state.device)?.unsqueeze(0)?;
            let rows = if !per_token && head.pool == Pool::Last {
                vec![state.model.embed_last_hidden(&x)?]
            } else {
                state.model.embed_all_hidden(&x)?
            };
            if per_token {
                return rows
                    .iter()
                    .map(|r| head.logits(r).map_err(|e| anyhow!(e)))
                    .collect();
            }
            let pooled = head
                .pool
                .pool(&rows)
                .ok_or_else(|| anyhow!("classify: no hidden state"))?;
            Ok(vec![head.logits(&pooled).map_err(|e| anyhow!(e))?])
        })
        .await;
        match inner {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Value;
    use crate::tensor::quantized::GgmlDType;

    fn f32_entry(name: &str, dims: Vec<usize>, values: &[f32]) -> GgufEntry {
        GgufEntry {
            name: name.into(),
            dims,
            dtype: GgmlDType::F32,
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// A GGUF head is read with its shapes, its labels, its pooling and its pair template, and
    /// gives the logits its weights define.
    #[test]
    fn a_gguf_head_is_read_with_its_labels_pooling_and_template() {
        let dir = std::env::temp_dir().join(format!("loken-cls-head-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.gguf");
        let entries = vec![
            f32_entry("cls.weight", vec![2, 2], &[1.0, 0.0, 0.0, 1.0]),
            f32_entry("cls.bias", vec![2], &[0.0, 0.5]),
            f32_entry(
                "cls.output.weight",
                vec![3, 2],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            f32_entry("cls.output.bias", vec![3], &[0.0, 0.0, -1.0]),
        ];
        let labels = ["a", "b", "c"].map(|l| Value::String(l.into())).to_vec();
        let metadata = vec![
            ("general.architecture".into(), Value::String("llama".into())),
            ("llama.pooling_type".into(), Value::U32(1)),
            (
                "llama.classifier.output_labels".into(),
                Value::Array(labels),
            ),
            (
                head::GGUF_PAIR_TEMPLATE.into(),
                Value::String("{query}|{document}".into()),
            ),
        ];
        write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let content = gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..])).unwrap();
        let head = read_classifier(&content, "llama", &mmap, &None, &None).expect("a head");
        assert_eq!(head.labels, ["a", "b", "c"]);
        assert_eq!(head.pool, Pool::Mean);
        assert_eq!(head.hidden(), 2);
        assert_eq!(head.pair_prompt("q", "d").as_deref(), Some("q|d"));
        let got = head.logits(&[0.0, 0.0]).unwrap();
        let t = 0.5f32.tanh();
        let want = [0.0, t, t - 1.0];
        assert!(
            got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6),
            "{got:?} {want:?}"
        );
        let plain = dir.join("plain.gguf");
        write_gguf_with_metadata(&plain, &metadata[..1], &entries[..0]).unwrap();
        let file = std::fs::File::open(&plain).unwrap();
        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let content = gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..])).unwrap();
        assert!(read_classifier(&content, "llama", &mmap, &None, &None).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
