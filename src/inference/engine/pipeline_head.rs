//! The backend of a node that runs the first layers of a generic transformer and hands the
//! rest to the node after it.
//!
//! The head keeps the KV cache of its own layers and sends the boundary of every forward, a
//! prefill chunk or a decode step alike, to the next segment, which answers with the logits.
//! A verify batch asks the last segment for the logits of every position; a cache shift is
//! refused, since it moves positions the later segments hold too.

use super::model_backend::{BoxedModelBackend, Capabilities, ModelBackend, PrefillShape};
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::serve::pipeline_segment::SegmentLink;
use crate::tensor::Tensor;

pub(crate) struct PipelineHeadBackend {
    model: GenericHeteroTransformer,
    link: SegmentLink,
    /// Why the segments after this one can no longer be trusted to hold the head's sequence:
    /// the link failed, or a cut did not reach them. Nothing more is served once it is set.
    broken: Option<String>,
}

impl PipelineHeadBackend {
    /// Make `model` the head of a pipeline: connect to the segment at `addr`, which runs layers
    /// `[cut, n)`. A model loaded whole keeps layers `[0, cut)` and gives up the rest; one
    /// loaded as the share `[0, cut)` is already the head. A model that cannot be cut, or a link
    /// that does not open, hands the model back untouched with the reason.
    pub(crate) fn connect(
        mut model: GenericHeteroTransformer,
        addr: std::net::SocketAddr,
        cut: usize,
    ) -> Result<Self, (Box<GenericHeteroTransformer>, crate::tensor::Error)> {
        let (first, held_to, n) = model.segment_range();
        let whole = first == 0 && held_to == n;
        let refused = if let Some(why) = model.segment_unsupported_reason() {
            Some(format!("this model cannot be a pipeline head: {why}"))
        } else if cut == 0 || cut >= n {
            Some(format!("pipeline head: cut {cut} must be in 1..{n}"))
        } else if !whole && (first, held_to) != (0, cut) {
            Some(format!(
                "pipeline head: holds layers [{first}, {held_to}), not [0, {cut})"
            ))
        } else {
            None
        };
        if let Some(msg) = refused {
            return Err((Box::new(model), crate::tensor::Error::msg(msg)));
        }
        let link = match SegmentLink::connect(addr) {
            Ok(link) => link,
            Err(e) => {
                let msg = format!("pipeline head: connect {addr}: {e}");
                return Err((Box::new(model), crate::tensor::Error::msg(msg)));
            }
        };
        if whole {
            if let Err(e) = model.narrow_to_layers(0, cut) {
                return Err((Box::new(model), e));
            }
            crate::inference::place::ladder::release_pools();
        }
        tracing::info!("pipeline head: layers [0, {cut}) of {n} here, the rest at {addr}");
        Ok(Self {
            model,
            link,
            broken: None,
        })
    }

    fn check(&self) -> crate::tensor::Result<()> {
        match &self.broken {
            Some(why) => crate::tensor::bail!("pipeline: {why}"),
            None => Ok(()),
        }
    }

    /// Cut the caches of every segment back to `new_len`; a cut that does not reach the
    /// others ends the pipeline.
    fn cut(&mut self, new_len: usize) {
        self.model.trim_kv(new_len);
        if let Err(e) = self.link.cut(new_len) {
            self.broken = Some(format!(
                "a cut to {new_len} did not reach the next segment: {e}"
            ));
        }
    }
}

impl ModelBackend for PipelineHeadBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            trim_kv: true,
            pld: true,
            ..Capabilities::default()
        }
    }

    fn kv_capacity(&self) -> Option<usize> {
        Some(self.model.kv_capacity())
    }

    fn unusable(&self) -> Option<String> {
        self.broken.clone()
    }

    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.check()?;
        let hidden = self.model.segment_head_forward(x, index_pos)?;
        match self.link.boundary(&hidden, index_pos) {
            Ok(logits) => Ok(logits),
            Err(e) => {
                // The segments hold the sequence up to the position this forward started at.
                self.cut(index_pos);
                Err(e)
            }
        }
    }

    fn forward_all(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.check()?;
        let hidden = self.model.segment_head_forward(x, index_pos)?;
        match self.link.boundary_all(&hidden, index_pos) {
            Ok(logits) => Ok(logits),
            Err(e) => {
                self.cut(index_pos);
                Err(e)
            }
        }
    }

    fn widest_ffn(&self) -> usize {
        self.model.widest_ffn()
    }

    fn prefill_shape(&self) -> PrefillShape {
        self.model.prefill_shape()
    }

    fn trim_kv(&mut self, new_len: usize) {
        self.cut(new_len);
    }

    fn reset_kv_from(&mut self, keep: usize) {
        self.cut(keep);
    }

    fn kv_len(&self) -> Option<usize> {
        self.model.kv_len()
    }

    fn placement(&self) -> Option<crate::inference::place::Placement> {
        Some(self.model.placement())
    }

    #[cfg(feature = "gpu")]
    fn cuda_device_ordinals(&self) -> std::collections::HashSet<usize> {
        self.model.cuda_device_ordinals()
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        self.model.device_layer_distribution()
    }

    fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
        Err(self)
    }

    fn into_batched(
        self: Box<Self>,
        _eos: u32,
        _max_batch: usize,
    ) -> Result<
        std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>,
        BoxedModelBackend,
    > {
        Err(self)
    }

    fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend> {
        Err(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::model_backend::GenericBackend;
    use super::*;
    use crate::inference::generic_transformer::tiny_model;
    use crate::inference::place::layer_executor::HeteroPlan;
    use crate::tensor::Device;

    const LAYERS: usize = 4;

    fn ids(v: &[u32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1usize, v.len()), &Device::Cpu).unwrap()
    }

    fn row(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// A head over a tail on loopback serves a prompt, its chunked prefill and its decode
    /// steps as the whole model does, resumes from a cut cache, and refuses a verify batch.
    #[test]
    fn a_head_serves_as_the_whole_model_does() {
        let path = tiny_model::write("backend", LAYERS);
        let mut whole = GenericBackend(tiny_model::load(&path, LAYERS));
        let plan = HeteroPlan::calculate_with_kv(2, 1 << 20, &[], &[], 1.0, 0).shifted(2, LAYERS);
        let g = crate::tensor::quantized::gguf_file::open_mapped(&path).unwrap();
        let bytes = g.mmap.clone();
        let tail = GenericHeteroTransformer::from_gguf(
            g.content,
            &bytes[..],
            &std::collections::HashMap::new(),
            &plan,
        )
        .unwrap();
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            crate::inference::serve::pipeline_segment::serve_segment(tail, None, listener).unwrap();
        });

        let mut head = match PipelineHeadBackend::connect(tiny_model::load(&path, LAYERS), addr, 2)
        {
            Ok(head) => head,
            Err((_, e)) => panic!("{e}"),
        };
        assert!(head.capabilities().trim_kv && !head.capabilities().shift_kv);

        // A prompt prefilled in chunks of two, then decode steps.
        let prompt = ids(&[1, 4, 2, 9, 3, 5]);
        let want = row(&whole.forward_prefill_with_chunk(&prompt, 0, 6, 2).unwrap());
        let got = row(&head.forward_prefill_with_chunk(&prompt, 0, 6, 2).unwrap());
        assert_eq!(got, want, "prefill");
        for (i, t) in [8u32, 2, 7].into_iter().enumerate() {
            let want = row(&whole.forward(&ids(&[t]), 6 + i).unwrap());
            let got = row(&head.forward(&ids(&[t]), 6 + i).unwrap());
            assert_eq!(got, want, "decode step {i}");
        }
        assert_eq!(head.kv_len(), whole.kv_len());
        // A verify batch: the logits of every token of the batch, as the whole model gives them,
        // then the cache cut back over its rejected tail.
        let batch = ids(&[5, 1, 8]);
        let want = row(&whole.forward_all(&batch, 9).unwrap());
        let got = row(&head.forward_all(&batch, 9).unwrap());
        assert_eq!(got, want, "verify batch");
        whole.trim_kv(10);
        head.trim_kv(10);
        let want = row(&whole.forward(&ids(&[3]), 10).unwrap());
        let got = row(&head.forward(&ids(&[3]), 10).unwrap());
        assert_eq!(got, want, "after the rejected tail");
        // The cache cut back to the prompt's first four tokens, then resumed.
        whole.trim_kv(4);
        head.trim_kv(4);
        assert_eq!(head.kv_len(), Some(4));
        let want = row(&whole.forward(&ids(&[6, 10]), 4).unwrap());
        let got = row(&head.forward(&ids(&[6, 10]), 4).unwrap());
        assert_eq!(got, want, "resumed after a cut");
        assert!(head.broken.is_none());
        drop(head);
        server.join().unwrap();
        tiny_model::remove(&path);
    }

    /// A tail that goes away leaves the head unusable, with the reason, so the model can be
    /// loaded again where the nodes still are.
    #[test]
    fn a_head_whose_tail_is_gone_says_it_is_unusable() {
        let path = tiny_model::write("gone", LAYERS);
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let tail = std::thread::spawn(move || {
            // Accept the head, then hang up on it.
            drop(listener.accept().unwrap());
        });
        let mut head = match PipelineHeadBackend::connect(tiny_model::load(&path, LAYERS), addr, 2)
        {
            Ok(head) => head,
            Err((_, e)) => panic!("{e}"),
        };
        tail.join().unwrap();
        assert!(head.unusable().is_none());
        assert!(head.forward(&ids(&[1, 4]), 0).is_err());
        let why = head.unusable().expect("the lost tail is named");
        assert!(why.contains("did not reach the next segment"), "{why}");
        assert!(
            head.forward(&ids(&[1]), 0).is_err(),
            "nothing more is served"
        );
        tiny_model::remove(&path);
    }

    /// A model loaded as the share `[0, cut)` is the head as it stands; a share that is not
    /// the one the cut names is refused.
    #[test]
    fn a_share_is_a_head_only_for_its_own_cut() {
        let path = tiny_model::write("head-share", LAYERS);
        let share = |from: usize, to: usize| {
            let plan = HeteroPlan::calculate_with_kv(to - from, 1 << 20, &[], &[], 1.0, 0)
                .shifted(from, LAYERS);
            let g = crate::tensor::quantized::gguf_file::open_mapped(&path).unwrap();
            let bytes = g.mmap.clone();
            GenericHeteroTransformer::from_gguf(
                g.content,
                &bytes[..],
                &std::collections::HashMap::new(),
                &plan,
            )
            .unwrap()
        };
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let tail = share(2, 4);
        let server = std::thread::spawn(move || {
            crate::inference::serve::pipeline_segment::serve_segment(tail, None, listener).unwrap();
        });
        let Err((_, e)) = PipelineHeadBackend::connect(share(0, 2), addr, 3) else {
            panic!("a share of two layers was made the head of a cut at three");
        };
        assert!(e.to_string().contains("holds layers [0, 2)"), "{e}");
        let Err((_, e)) = PipelineHeadBackend::connect(share(1, 3), addr, 3) else {
            panic!("a share that does not start the stack was made the head");
        };
        assert!(e.to_string().contains("holds layers [1, 3)"), "{e}");
        let mut head = match PipelineHeadBackend::connect(share(0, 2), addr, 2) {
            Ok(head) => head,
            Err((_, e)) => panic!("{e}"),
        };
        let mut whole = GenericBackend(tiny_model::load(&path, LAYERS));
        let want = row(&whole.forward(&ids(&[1, 4, 2]), 0).unwrap());
        let got = row(&head.forward(&ids(&[1, 4, 2]), 0).unwrap());
        assert_eq!(got, want);
        drop(head);
        server.join().unwrap();
        tiny_model::remove(&path);
    }

    /// A model that is cut at its end, or at its start, stays whole and says why.
    #[test]
    fn a_cut_outside_the_stack_is_refused_and_the_model_comes_back() {
        let path = tiny_model::write("refuse", LAYERS);
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 9));
        for cut in [0, LAYERS] {
            let Err((model, e)) =
                PipelineHeadBackend::connect(tiny_model::load(&path, LAYERS), addr, cut)
            else {
                panic!("a cut at {cut} was accepted");
            };
            assert!(e.to_string().contains("must be in"), "{e}");
            assert_eq!(model.segment_layers(), LAYERS);
        }
        tiny_model::remove(&path);
    }
}
