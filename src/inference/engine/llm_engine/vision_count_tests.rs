//! The prompt count of a request, through the engine on the small generic model: a repeated text
//! prompt, and a request whose image rows are spliced over a placeholder token.
//!
//! The toy backend expands each placeholder to the merged grid of its item and prefills the
//! expanded ids, so the cache holds the rows a vision tower would write.

use super::block_draft_tests::{engine_holding, params};
use super::*;
use crate::inference::engine::model_backend::ModelBackend;
use crate::inference::generic_transformer::block_draft_tests::{generic_of, toy_files};

/// The band layout of the small file.
const LAYOUT: (&[u32], &[u32], &[u32]) = (&[0, 2, 0], &[1], &[1]);

/// The placeholder token of an image.
const SENTINEL: u32 = 5;

/// The patch grid of the toy image: a merged grid of 4 by 3 rows.
const GRID: (usize, usize) = (8, 6);

/// The generic model of the small file, with each placeholder spliced as its merged grid.
struct SplicedToy(GenericBackend);

impl ModelBackend for SplicedToy {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn forward_prefill_chunked(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> crate::tensor::Result<Tensor> {
        self.0.forward_prefill_chunked(x, index_pos)
    }

    fn trim_kv(&mut self, new_len: usize) {
        self.0.trim_kv(new_len)
    }

    fn reset_kv_from(&mut self, keep: usize) {
        self.0.reset_kv_from(keep)
    }

    fn kv_len(&self) -> Option<usize> {
        self.0.kv_len()
    }

    fn qwen35_image_token(&self) -> Option<u32> {
        Some(SENTINEL)
    }

    fn forward_qwen35_vision(
        &mut self,
        prompt_tokens: &[u32],
        items: &[(Tensor, (usize, usize))],
        device: &Device,
    ) -> crate::tensor::Result<(Tensor, usize)> {
        let mut grids = items.iter().map(|(_, (gh, gw))| (gh / 2) * (gw / 2));
        let mut ids = Vec::new();
        for &t in prompt_tokens {
            match (t == SENTINEL).then(|| grids.next()).flatten() {
                Some(n) => ids.extend(std::iter::repeat_n(t, n)),
                None => ids.push(t),
            }
        }
        let x = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let logits = self.0.forward_prefill_chunked(&x, 0)?;
        Ok((logits, ids.len()))
    }

    fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
        Err(self)
    }

    fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend> {
        Err(self)
    }
}

/// An engine holding the spliced toy.
async fn engine() -> LlmEngine {
    let (file, _) = toy_files(LAYOUT, false);
    let model = generic_of(file);
    let num_layers = model.segment_layers();
    engine_holding(Box::new(SplicedToy(GenericBackend(model))), num_layers)
        .await
        .0
}

impl LlmEngine {
    /// Gives the next request one Qwen3.5 image of `GRID`, whatever the model reads.
    pub(crate) async fn attach_qwen35_image(&self) {
        let px =
            Tensor::zeros_on((1usize, 4usize), crate::tensor::DType::F32, &Device::Cpu).unwrap();
        let mut guard = self.model_state.lock().await;
        guard.as_mut().unwrap().qwen35_vision = vec![(px, GRID)];
    }
}

/// Gives the next request one image of `GRID`.
async fn attach_image(engine: &LlmEngine) {
    engine.attach_qwen35_image().await;
}

/// A prompt of 6 tokens, 1 of them the placeholder.
const PROMPT: &str = "t1 t2 t5 t3 t4 t6";

/// The rows the toy image takes beyond its placeholder.
const IMAGE_EXTRA: u64 = ((GRID.0 / 2) * (GRID.1 / 2) - 1) as u64;

/// A request with an image counts the rows of its merged grid, through the whole and the
/// streamed answer.
#[tokio::test(flavor = "multi_thread")]
async fn an_image_counts_its_rows_in_the_prompt_tokens() {
    let engine = engine().await;
    attach_image(&engine).await;
    let whole = engine.generate(PROMPT, params(3)).await.unwrap();
    assert_eq!(whole.prompt_eval_count, 6 + IMAGE_EXTRA);

    attach_image(&engine).await;
    let mut rx = engine.generate_stream(PROMPT, params(3)).await.unwrap();
    while let Some(chunk) = rx.recv().await {
        chunk.unwrap();
    }
    let stats = engine.take_last_stream_stats().unwrap();
    assert_eq!(stats.prompt_eval_count, 6 + IMAGE_EXTRA);
}

/// A request without an image counts its tokens.
#[tokio::test(flavor = "multi_thread")]
async fn a_text_prompt_counts_its_tokens() {
    let engine = engine().await;
    let whole = engine.generate(PROMPT, params(3)).await.unwrap();
    assert_eq!(whole.prompt_eval_count + whole.cached_prompt_tokens, 6);
}

/// A text prompt streamed twice is read once.
/// The repeat counts each position once, as served from the cache, in its own counts and in
/// the request's figures alike.
#[tokio::test(flavor = "multi_thread")]
async fn a_repeated_prompt_counts_each_position_once() {
    let (file, _) = toy_files(LAYOUT, false);
    let (engine, _) = super::block_draft_tests::engine_of(generic_of(file)).await;
    let mut counts = Vec::new();
    for _ in 0..2 {
        let probe = std::sync::Arc::new(crate::request_metrics::RequestProbe::default());
        let params = GenerationParams {
            probe: Some(probe.clone()),
            ..params(3)
        };
        let mut rx = engine.generate_stream(PROMPT, params).await.unwrap();
        while let Some(chunk) = rx.recv().await {
            chunk.unwrap();
        }
        let stats = engine.take_last_stream_stats().unwrap();
        let figures = probe.outcome().unwrap().figures;
        counts.push((
            (stats.prompt_eval_count, stats.cached_prompt_tokens),
            (figures.prompt_read(), figures.cached_prompt_tokens),
        ));
    }
    assert_eq!(
        counts[0],
        ((6, 0), (6, 0)),
        "the first request reads its prompt"
    );
    assert_eq!(counts[1], ((0, 6), (0, 6)), "the repeat reads nothing");
}

/// The request's own figures count the prompt the answer reports, image rows included, on the
/// whole and the streamed answer.
#[tokio::test(flavor = "multi_thread")]
async fn the_request_figures_count_the_prompt_the_answer_reports() {
    let engine = engine().await;
    for streamed in [false, true] {
        attach_image(&engine).await;
        let probe = std::sync::Arc::new(crate::request_metrics::RequestProbe::default());
        let params = GenerationParams {
            probe: Some(probe.clone()),
            ..params(3)
        };
        let (read, cached) = if streamed {
            let mut rx = engine.generate_stream(PROMPT, params).await.unwrap();
            while let Some(chunk) = rx.recv().await {
                chunk.unwrap();
            }
            let stats = engine.take_last_stream_stats().unwrap();
            (stats.prompt_eval_count, stats.cached_prompt_tokens)
        } else {
            let whole = engine.generate(PROMPT, params).await.unwrap();
            (whole.prompt_eval_count, whole.cached_prompt_tokens)
        };
        let figures = probe.outcome().unwrap().figures;
        assert_eq!(figures.prompt_tokens, read + cached, "streamed {streamed}");
        assert_eq!(figures.prompt_read(), read, "streamed {streamed}");
    }
}

/// A request with an image and token rules is decoded with its image: the sequential path
/// counts the rows of the image and writes the answer the other paths write.
#[tokio::test(flavor = "multi_thread")]
async fn an_image_with_token_rules_is_decoded_with_its_image() {
    let engine = engine().await;
    attach_image(&engine).await;
    let plain = engine.generate(PROMPT, params(3)).await.unwrap();
    attach_image(&engine).await;
    let mut ruled = params(3);
    ruled.rules.cache_salt = Some("a salt".to_string());
    assert!(ruled.rules.decodes_sequentially());
    let sequential = engine.generate(PROMPT, ruled).await.unwrap();
    assert_eq!(sequential.prompt_eval_count, 6 + IMAGE_EXTRA);
    assert_eq!(sequential.text, plain.text);
}
