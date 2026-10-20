//! A generic transformer as one segment of a stack that several nodes run in turn.
//!
//! The first segment embeds the tokens and runs its layers, an interior segment runs its
//! layers over the boundary it receives, and the last runs its layers and the output head.
//! Each segment keeps the KV cache of its own layers, so a boundary is the only thing that
//! crosses between nodes. The boundary of a prefill chunk and of a decode step is the same
//! tensor, `[1, tokens, hidden]`.

#[allow(unused_imports)]
use super::*;

/// What the layers of a model cost to run one token, measured on this node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerRates {
    /// Milliseconds one layer takes on a card, `None` when no layer runs on one.
    pub card_ms: Option<f64>,
    /// Milliseconds one layer takes on the host, `None` when no layer runs there.
    pub host_ms: Option<f64>,
}

/// The middle value of `v`, which is sorted by the call.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

impl GenericHeteroTransformer {
    /// Why this model cannot be cut into segments, or `None` when it can. A segment holds only
    /// what its own layers need, so a model whose layers read state kept outside them stays
    /// whole.
    pub fn segment_unsupported_reason(&self) -> Option<&'static str> {
        if !self.vulkan.is_empty() {
            return Some("its layers run through Vulkan");
        }
        if self.ple_dim > 0 {
            return Some("its layers take per-layer inputs built from the tokens");
        }
        if self.config.shared_kv_layers > 0 {
            return Some("its later layers read the KV cache of earlier ones");
        }
        if self.config.hyper.is_some() || self.ngram.is_some() {
            return Some("its residual stream is wide and mixed outside the layers");
        }
        if self.spliced_vision.is_some() {
            return Some("it carries an image tower that splices into the embeddings");
        }
        if !self.adapters.is_empty() {
            return Some("an adapter is applied to its layers");
        }
        if self.layers.iter().any(|l| !l.attends()) {
            return Some("a layer keeps a recurrent state that a cache trim cannot cut");
        }
        if self
            .layers
            .iter()
            .any(|l| l.moe.as_ref().is_some_and(|m| m.experts_on_host()))
        {
            return Some("routed experts are held in host memory");
        }
        None
    }

    /// Elements one layer's cache holds per token: K and V of every KV head.
    pub fn kv_elems_per_token(&self) -> u64 {
        let c = &self.config;
        (2 * c.n_kv_head * c.head_dim) as u64
    }

    /// How the layers' caches are stored.
    pub fn kv_quant(&self) -> crate::inference::engine::llm_engine::KvQuant {
        self.config.kv_quant
    }

    /// Layers this model runs.
    pub fn segment_layers(&self) -> usize {
        self.layers.len()
    }

    /// The layers `[from, to)` of the stack this model holds, and the stack's length.
    pub fn segment_range(&self) -> (usize, usize, usize) {
        (
            self.first_layer,
            self.first_layer + self.layers.len(),
            self.config.n_layers,
        )
    }

    /// The boundary the first segment hands on: the embeddings of `ids` through every layer
    /// this model holds, `[1, tokens, hidden]` on the host.
    pub fn segment_head_forward(&mut self, ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, seq_len) = ids.dims2()?;
        let (hidden, ple) = self.embed_stage(ids, index_pos)?;
        let (hidden, _) = self.run_layers(hidden, ple, index_pos, seq_len)?;
        hidden.to_device(&Device::Cpu)
    }

    /// The boundary an interior segment hands on: the boundary it received through every
    /// layer this model holds, on the host.
    pub fn segment_interior_forward(&mut self, hidden: Tensor, index_pos: usize) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        self.mark_prefill(seq_len);
        let (hidden, _) = self.run_layers(hidden, None, index_pos, seq_len)?;
        hidden.to_device(&Device::Cpu)
    }

    /// The logits the last segment answers with: the boundary it received through every layer
    /// this model holds, the final norm and the output head, `[1, vocab]` for the last token.
    pub fn segment_tail_forward(&mut self, hidden: Tensor, index_pos: usize) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        self.mark_prefill(seq_len);
        let (hidden, on_cuda) = self.run_layers(hidden, None, index_pos, seq_len)?;
        self.output_stage(hidden, on_cuda, seq_len, false)
    }

    /// [`Self::segment_tail_forward`] for a verify: the logits of every token of the boundary,
    /// `[1, tokens, vocab]`.
    pub fn segment_tail_forward_all(&mut self, hidden: Tensor, index_pos: usize) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        self.mark_prefill(seq_len);
        let (hidden, on_cuda) = self.run_layers(hidden, None, index_pos, seq_len)?;
        self.output_stage(hidden, on_cuda, seq_len, true)
    }

    fn mark_prefill(&self, seq_len: usize) {
        crate::tensor::quant_cpu::set_small_model_prefill(
            seq_len > 1 && self.config.embedding_length <= 1024,
        );
    }

    /// Keep the layers `[from, to)` of a model loaded whole and release the rest. A segment
    /// that does not end the stack gives up the output head with them.
    pub fn narrow_to_layers(&mut self, from: usize, to: usize) -> Result<()> {
        let n = self.layers.len();
        if self.first_layer != 0 || n != self.config.n_layers {
            crate::tensor::bail!("only a model holding the whole stack narrows to a share");
        }
        if from >= to || to > n {
            crate::tensor::bail!("layers [{from}, {to}) are not within the {n} this model holds");
        }
        self.layers.truncate(to);
        self.layers.drain(..from);
        self.layer_devs.truncate(to);
        self.layer_devs.drain(..from);
        if to < n {
            self.output_proj_cuda = None;
            self.output_proj_cuda_qmm = None;
            self.output_norm_cuda_weight = None;
            self.output_proj_dev = None;
            self.graph_logits_buffer = None;
        }
        self.mask_cache.clear();
        Ok(())
    }

    /// What one layer costs on a card and on the host, measured by running decode steps
    /// through each run of layers that shares a device and reading the result back. The cache
    /// is left empty.
    pub fn measure_layer_ms(&mut self) -> Result<LayerRates> {
        // Steps the kernels compile and the caches grow in, then the steps that are timed.
        const WARM: usize = 2;
        const TIMED: usize = 7;
        let mut runs: Vec<(LayerDevice, usize, usize)> = Vec::new();
        for (i, dev) in self.layer_devs.iter().enumerate() {
            match runs.last_mut() {
                Some((d, _, end)) if d == dev => *end = i + 1,
                _ => runs.push((dev.clone(), i, i + 1)),
            }
        }
        let ids = Tensor::from_vec(vec![0u32], (1usize, 1usize), &Device::Cpu)?;
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); runs.len()];
        for step in 0..WARM + TIMED {
            let (hidden, _) = self.embed_stage(&ids, step)?;
            for (k, (_, from, upto)) in runs.iter().enumerate() {
                let started = std::time::Instant::now();
                let (out, _) = self.run_layers_in(hidden.clone(), None, step, 1, *from, *upto)?;
                out.to_device(&Device::Cpu)?;
                if step >= WARM {
                    samples[k].push(started.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        self.trim_kv(0);
        let mut rate = |card: bool| -> Option<f64> {
            let (ms, layers) = runs
                .iter()
                .zip(samples.iter_mut())
                .filter(|((dev, _, _), _)| matches!(dev, LayerDevice::Cuda(_)) == card)
                .fold((0.0, 0usize), |(ms, n), ((_, from, upto), s)| {
                    (ms + median(s), n + upto - from)
                });
            (layers > 0).then(|| ms / layers as f64)
        };
        Ok(LayerRates {
            card_ms: rate(true),
            host_ms: rate(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::tiny_model;
    use crate::inference::place::layer_executor::HeteroPlan;

    const LAYERS: usize = 3;

    /// The model at `path` holding only the layers `[from, to)`, each read once and placed by
    /// a plan over its own share.
    fn share(path: &std::path::Path, from: usize, to: usize) -> GenericHeteroTransformer {
        let plan = HeteroPlan::calculate_with_kv(to - from, 1 << 20, &[], &[], 1.0, 0)
            .shifted(from, LAYERS);
        let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
        let bytes = g.mmap.clone();
        GenericHeteroTransformer::from_gguf(
            g.content,
            &bytes[..],
            &std::collections::HashMap::new(),
            &plan,
        )
        .unwrap()
    }

    fn ids(v: &[u32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1usize, v.len()), &Device::Cpu).unwrap()
    }

    fn row(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The prompt as a prefill chunk, then single decode steps.
    fn steps() -> Vec<(Vec<u32>, usize)> {
        vec![
            (vec![1, 4, 2, 9], 0),
            (vec![3], 4),
            (vec![7], 5),
            (vec![5], 6),
        ]
    }

    /// Three parts loaded from their own layers answer as the whole model does, bit for bit,
    /// for a prefill chunk and the decode steps after it.
    #[test]
    fn shares_loaded_alone_run_as_the_whole_model_does() {
        let path = tiny_model::write("share", LAYERS);
        let mut whole = tiny_model::load(&path, LAYERS);
        let mut first = share(&path, 0, 1);
        let mut middle = share(&path, 1, 2);
        let mut last = share(&path, 2, 3);
        assert_eq!(first.segment_layers(), 1);
        for (toks, pos) in steps() {
            let want = row(&whole.forward(&ids(&toks), pos).unwrap());
            let h = first.segment_head_forward(&ids(&toks), pos).unwrap();
            let h = middle.segment_interior_forward(h, pos).unwrap();
            let got = row(&last.segment_tail_forward(h, pos).unwrap());
            assert_eq!(got, want, "step at position {pos}");
        }
        tiny_model::remove(&path);
    }

    /// A model loaded whole and narrowed to a share is the share loaded alone.
    #[test]
    fn a_narrowed_model_runs_as_the_whole_model_does() {
        let path = tiny_model::write("narrow", LAYERS);
        let mut whole = tiny_model::load(&path, LAYERS);
        let mut head = tiny_model::load(&path, LAYERS);
        head.narrow_to_layers(0, 2).unwrap();
        let mut tail = tiny_model::load(&path, LAYERS);
        tail.narrow_to_layers(2, 3).unwrap();
        assert_eq!((head.segment_layers(), tail.segment_layers()), (2, 1));
        for (toks, pos) in steps() {
            let want = row(&whole.forward(&ids(&toks), pos).unwrap());
            let h = head.segment_head_forward(&ids(&toks), pos).unwrap();
            let got = row(&tail.segment_tail_forward(h, pos).unwrap());
            assert_eq!(got, want, "step at position {pos}");
        }
        assert!(head.narrow_to_layers(1, 1).is_err());
        assert!(head.narrow_to_layers(0, 3).is_err());
        tiny_model::remove(&path);
    }

    /// A share cut back and run again from an earlier position answers as the whole model
    /// does after the same cut.
    #[test]
    fn a_trimmed_cache_resumes_as_the_whole_model_does() {
        let path = tiny_model::write("trim", LAYERS);
        let mut whole = tiny_model::load(&path, LAYERS);
        let mut head = share(&path, 0, 2);
        let mut tail = share(&path, 2, 3);
        let run = |whole: &mut GenericHeteroTransformer,
                   head: &mut GenericHeteroTransformer,
                   tail: &mut GenericHeteroTransformer,
                   toks: &[u32],
                   pos: usize| {
            let want = row(&whole.forward(&ids(toks), pos).unwrap());
            let h = head.segment_head_forward(&ids(toks), pos).unwrap();
            assert_eq!(row(&tail.segment_tail_forward(h, pos).unwrap()), want);
        };
        run(&mut whole, &mut head, &mut tail, &[1, 4, 2, 9, 3], 0);
        for m in [&mut whole, &mut head, &mut tail] {
            m.trim_kv(2);
        }
        run(&mut whole, &mut head, &mut tail, &[6, 8], 2);
        tiny_model::remove(&path);
    }

    /// A model of plain attention layers can be cut; the reasons it cannot are named.
    #[test]
    fn a_plain_model_can_be_cut() {
        let path = tiny_model::write("plain", LAYERS);
        let m = tiny_model::load(&path, LAYERS);
        assert_eq!(m.segment_unsupported_reason(), None);
        tiny_model::remove(&path);
    }

    /// What a layer costs is measured on the host and the cache is left empty.
    #[test]
    fn layer_cost_is_measured_and_leaves_the_cache_empty() {
        let path = tiny_model::write("rates", LAYERS);
        let mut m = tiny_model::load(&path, LAYERS);
        let rates = m.measure_layer_ms().unwrap();
        assert_eq!(rates.card_ms, None);
        assert!(rates.host_ms.is_some_and(|ms| ms > 0.0));
        assert_eq!(m.kv_len().unwrap_or(0), 0);
        tiny_model::remove(&path);
    }
}

#[cfg(test)]
mod layer_major_tests {
    use crate::inference::generic_transformer::tiny_model;
    use crate::tensor::{Device, Tensor};

    fn ids(v: &[u32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1usize, v.len()), &Device::Cpu).unwrap()
    }

    fn row(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Every chunk through a layer before the next layer, or each chunk through every layer:
    /// the last token's logits are the same, and so is the state the decode steps that follow
    /// read.
    #[test]
    fn a_prompt_goes_through_the_layers_layer_by_layer_as_chunk_by_chunk() {
        let path = tiny_model::write("layer-major", 4);
        let prompt: Vec<u32> = (0..13u32).map(|i| (i * 7 + 3) % 11).collect();
        for chunk in [2usize, 5, 13] {
            let mut by_chunk = tiny_model::load(&path, 4);
            let mut by_layer = tiny_model::load(&path, 4);
            let mut want = None;
            let mut off = 0;
            while off < prompt.len() {
                let n = chunk.min(prompt.len() - off);
                want = Some(row(&by_chunk
                    .forward(&ids(&prompt[off..off + n]), off)
                    .unwrap()));
                off += n;
            }
            let got = row(&by_layer
                .forward_prefill_layer_major(&ids(&prompt), 0, chunk)
                .unwrap());
            assert_eq!(got, want.unwrap(), "chunk {chunk}");
            for (i, t) in [4u32, 9, 1].into_iter().enumerate() {
                let a = row(&by_chunk.forward(&ids(&[t]), 13 + i).unwrap());
                let b = row(&by_layer.forward(&ids(&[t]), 13 + i).unwrap());
                assert_eq!(a, b, "chunk {chunk}, decode step {i}");
            }
        }
        tiny_model::remove(&path);
    }
}
