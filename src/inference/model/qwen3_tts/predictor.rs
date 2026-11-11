//! The code predictor: the codes of a frame past group 0.
//!
//! It is a transformer of the generic route with no head, its cache restarted every frame.
//! Its first pass takes 2 rows: the talker's normed row and the talker's codec row of group 0.
//! Each later pass takes the code predictor's row of the code the pass before drew.
//! Every row it takes goes through its input projection first.
//! The head of pass `s` gives the logits of group `s + 1`.

use super::config::Sampling;
use super::glue::{HostTable, Linear};
use super::{sampling, SubCodes};
use crate::inference::generic_transformer::{EmbeddingStep, GenericHeteroTransformer};
use crate::tensor::{Device, Error, Result, Tensor};

/// The class of the code predictor nested in the checkpoint.
pub const PREDICTOR_CLASS: &str = "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration";

/// The tensor name of the input projection.
const INPUT: &str = "talker.code_predictor.small_to_mtp_projection";

/// The tensor name of the head of pass `s`.
fn head(s: usize) -> String {
    format!("talker.code_predictor.lm_head.{s}")
}

/// The code predictor, its input projection, its heads and its tables.
pub struct CodePredictor {
    model: GenericHeteroTransformer,
    input: Linear,
    heads: Vec<Linear>,
    /// The table of each group past group 0, read by row.
    tables: Vec<HostTable>,
    sampling: Sampling,
}

impl CodePredictor {
    /// Bytes the input projection and the `passes` heads of `st` take held, which sit beside
    /// the code predictor's last layer.
    pub fn glue_bytes(st: &crate::tensor::safetensors::MmapedSafetensors, passes: usize) -> u64 {
        Linear::bytes(st, INPUT)
            + (0..passes)
                .map(|s| Linear::bytes(st, &head(s)))
                .sum::<u64>()
    }

    /// The code predictor `model` with the projection and the heads of `st` on `device`.
    pub fn new(
        model: GenericHeteroTransformer,
        st: &crate::tensor::safetensors::MmapedSafetensors,
        tables: Vec<HostTable>,
        sampling: Sampling,
        device: &Device,
    ) -> Result<Self> {
        let heads = (0..tables.len())
            .map(|s| Linear::load(st, &head(s), false, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            model,
            input: Linear::load(st, INPUT, true, device)?,
            heads,
            tables,
            sampling,
        })
    }

    /// The device the projection and the heads run on.
    pub fn device(&self) -> Device {
        self.input.device()
    }

    /// Draw the codes of the frames that follow by `sampling`.
    pub fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    /// Where the code predictor's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        super::layer_runs(&self.model)
    }

    /// The codes of a frame past group 0, `pick` choosing the code of each pass from its logits.
    ///
    /// `past` is the talker's normed row and `group0` its codec row of group 0.
    /// The cache restarts at position 0.
    pub fn run(
        &mut self,
        past: &Tensor,
        group0: &Tensor,
        pick: &mut dyn FnMut(usize, &[f32]) -> Result<u32>,
    ) -> Result<Vec<u32>> {
        let device = self.input.device();
        let width = self.tables.first().map_or(0, |t| t.width());
        let rows = Tensor::cat(
            &[
                &past.to_device(&device)?.reshape((1, width))?,
                &group0.to_device(&device)?.reshape((1, width))?,
            ],
            0,
        )?;
        let first = self.input.forward(&rows)?;
        let mut out: EmbeddingStep = self.model.forward_embeddings_prefill(&first, false)?;
        let mut codes = Vec::with_capacity(self.heads.len());
        for s in 0..self.heads.len() {
            let logits = super::host(&self.heads[s].forward(&out.normed)?)?;
            let code = pick(s, &logits)?;
            codes.push(code);
            if s + 1 < self.heads.len() {
                let row = self.input.forward(&self.tables[s].rows(&[code])?)?;
                out = self.model.forward_embedding_step(&row, s + 2, false)?;
            }
        }
        Ok(codes)
    }

    /// The logits of every pass of a frame fed the codes `codes`, `codes[s]` following pass `s`.
    pub fn logits_fed(
        &mut self,
        past: &Tensor,
        group0: &Tensor,
        codes: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        if codes.len() != self.heads.len() {
            return Err(Error::msg(format!(
                "a frame holds {} codes past group 0, {} given",
                self.heads.len(),
                codes.len()
            )));
        }
        let mut all = Vec::new();
        self.run(past, group0, &mut |s, logits| {
            all.push(logits.to_vec());
            Ok(codes[s])
        })?;
        Ok(all)
    }
}

impl SubCodes for CodePredictor {
    fn sub_codes(
        &mut self,
        past: &Tensor,
        group0: &Tensor,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Vec<u32>> {
        let sampling = self.sampling;
        self.run(past, group0, &mut |_, logits| {
            sampling::draw(logits, &sampling, rng)
        })
    }
}
