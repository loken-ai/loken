//! Qwen3-TTS: a talker over the codes of a speech codec, and its code predictor.
//!
//! The talker is a transformer of the generic route.
//! Each frame, it draws the code of group 0 from the row its last step gives.
//! The other groups of the frame come from a source of sub-codes, the code predictor.
//! The next talker row is the sum of the frame's code rows and the pad text row.
//!
//! The prompt rows and the frame loop follow the reference generation of the checkpoint's
//! publisher, in its non-streaming layout: the whole text sits in the prompt.

pub mod config;
pub mod decoder;
mod glue;
pub mod predictor;
pub mod prompt;
pub mod sampling;
#[cfg(test)]
mod tests;
pub mod text;

pub use config::{Generation, Sampling, SpeechIds};

use crate::inference::generic_transformer::class_load::{self, Rows};
use crate::inference::generic_transformer::{EmbeddingStep, GenericHeteroTransformer};
use crate::inference::serve::progress::placement::Placed;
use crate::tensor::{DType, Device, Error, Result, Tensor};
use glue::{HostTable, Linear, Shard};
use rand::SeedableRng;

/// What a request asks of the talker.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    /// The text to speak.
    pub text: &'a str,
    /// The words that describe the voice.
    pub design: Option<&'a str>,
    /// A language the checkpoint names, `None` leaving it to the model.
    pub language: Option<&'a str>,
    /// The seed of every draw.
    pub seed: u64,
}

/// Values a request sets in place of the checkpoint's sampling values, each left `None` to keep
/// the checkpoint's.
///
/// The temperature, top-k and top-p apply to the talker and to the code predictor.
/// A temperature of 0 takes the most probable code.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Overrides {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Frames an utterance may hold at most, under the checkpoint's own limit.
    pub max_frames: Option<usize>,
}

impl Overrides {
    /// `sampling` with the values the request sets.
    pub fn apply(&self, sampling: Sampling) -> Sampling {
        let temperature = self.temperature.unwrap_or(sampling.temperature);
        Sampling {
            sample: sampling.sample && temperature > 0.0,
            temperature,
            top_k: self.top_k.unwrap_or(sampling.top_k),
            top_p: self.top_p.unwrap_or(sampling.top_p),
        }
    }
}

/// Why the frame loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The talker drew the end of speech.
    End,
    /// The utterance reached the frames the request allows.
    Asked,
    /// The utterance reached the frames the cache and the checkpoint allow.
    Budget,
}

/// The frames of an utterance and why it stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utterance {
    pub frames: Vec<Vec<u32>>,
    pub stop: Stop,
}

/// The runs of the layers of `model`, by device.
fn layer_runs(model: &GenericHeteroTransformer) -> Vec<Placed> {
    model
        .device_layer_distribution()
        .into_iter()
        .map(|(kind, id, start, end)| Placed {
            device: match kind.as_str() {
                "CUDA" => crate::tensor::DeviceLocation::Cuda { gpu_id: id },
                _ => crate::tensor::DeviceLocation::Cpu,
            },
            layer_start: start as usize,
            layer_end: end as usize + 1,
            bytes: 0,
        })
        .collect()
}

/// The source of the codes of a frame past group 0.
pub trait SubCodes {
    /// The codes of groups 1 and up of a frame, in order.
    ///
    /// `past` is the talker's normed row the frame's group 0 code was drawn from.
    /// `group0` is the talker's codec row of that code.
    fn sub_codes(
        &mut self,
        past: &Tensor,
        group0: &Tensor,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Vec<u32>>;
}

/// The talker, the tables and the projection its rows are built from, and the sampling values.
pub struct Talker {
    model: GenericHeteroTransformer,
    tokenizer: tokenizers::Tokenizer,
    ids: SpeechIds,
    generation: Generation,
    /// The text table, read by row.
    text_table: HostTable,
    /// The text projection: `fc2(silu(fc1(row)))`.
    text_fc1: Linear,
    text_fc2: Linear,
    /// The code predictor's table of each group past group 0, read by row.
    groups: Vec<HostTable>,
    /// The pad text row, on the host.
    pad_row: Vec<f32>,
    /// Rows the talker's cache holds.
    context: usize,
}

/// `t` as F32 values on the host.
fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn read_json(path: &std::path::Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::msg(format!("{}: {e}", path.display())))
}

/// The tensor name of the text projection's layer `fc`.
fn text_projection(fc: &str) -> String {
    format!("talker.text_projection.{fc}")
}

impl Talker {
    /// The talker of the checkpoint at `dir`, its cache holding `rows`.
    ///
    /// The talker loads through `HeteroPlan` and the ladder.
    /// The text projection sits on the device of its last layer, counted in its demand.
    pub fn load(dir: &std::path::Path, rows: Rows) -> Result<Self> {
        Self::load_beside(dir, rows, None)
    }

    /// [`Self::load`] with the text projection on `glue`, which another demand counts, when
    /// given.
    fn load_beside(dir: &std::path::Path, rows: Rows, glue: Option<&Device>) -> Result<Self> {
        let root = read_json(&dir.join("config.json"))?;
        let ids = SpeechIds::read(&root)?;
        let generation = Generation::read(&read_json(&dir.join("generation_config.json"))?)?;
        let class = root
            .get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|c| c.as_str())
            .ok_or_else(|| Error::msg("config.json names no class"))?;
        let (_, shards) = crate::inference::load::awq_loader::checkpoint_shards(dir)
            .map_err(|e| Error::msg(e.to_string()))?;
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards)? };
        let fcs = ["linear_fc1", "linear_fc2"];
        let glue_bytes = match glue {
            Some(_) => 0,
            None => Self::glue_bytes(&st),
        };
        let loaded = class_load::load_class(dir, class, rows, glue_bytes)?;
        let glue = glue.unwrap_or(&loaded.last_device);
        let fc = |n: &str| Linear::load(&st, &text_projection(n), true, glue);
        let mapped = Shard::map_all(&shards)?;
        let groups = (0..ids.groups - 1)
            .map(|g| {
                HostTable::open(
                    &mapped,
                    &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let tokenizer = crate::inference::engine::llm_engine::build_tokenizer_from_bpe_files(dir)
            .map_err(|e| Error::msg(e.to_string()))?;
        let mut talker = Self {
            model: loaded.model,
            tokenizer,
            text_table: HostTable::open(&mapped, "talker.model.text_embedding.weight")?,
            text_fc1: fc(fcs[0])?,
            text_fc2: fc(fcs[1])?,
            groups,
            pad_row: Vec::new(),
            ids,
            generation,
            context: loaded.context,
        };
        talker.pad_row = host(&talker.text_rows(&[talker.ids.tts_pad])?)?;
        Ok(talker)
    }

    /// Bytes the text projection of `st` takes held.
    fn glue_bytes(st: &crate::tensor::safetensors::MmapedSafetensors) -> u64 {
        ["linear_fc1", "linear_fc2"]
            .iter()
            .map(|fc| Linear::bytes(st, &text_projection(fc)))
            .sum()
    }

    /// Rows the talker's cache holds.
    pub fn context(&self) -> usize {
        self.context
    }

    /// The ids the prompt and the loop use.
    pub fn ids(&self) -> &SpeechIds {
        &self.ids
    }

    /// The sampling values of the checkpoint.
    pub fn generation(&self) -> &Generation {
        &self.generation
    }

    /// The text rows of `ids` through the text projection, `[ids.len(), hidden]` F32.
    pub fn text_rows(&self, ids: &[u32]) -> Result<Tensor> {
        let x = self.text_table.rows(ids)?;
        let h = self.text_fc1.forward(&x)?;
        self.text_fc2.forward(&crate::tensor::ops::silu(&h)?)
    }

    /// The talker's codec rows of `ids`, `[ids.len(), hidden]` F32 on the host.
    pub fn codec_rows(&self, ids: &[u32]) -> Result<Tensor> {
        self.model
            .token_rows(ids)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)
    }

    /// The prompt of `rows`, `[rows.len(), hidden]` F32 on the host.
    pub fn prompt_rows(&self, rows: &[prompt::Row]) -> Result<Tensor> {
        let width = self.text_table.width();
        let text_ids: Vec<u32> = rows.iter().filter_map(|r| r.text).collect();
        let codec_ids: Vec<u32> = rows.iter().filter_map(|r| r.codec).collect();
        let text = host(&self.text_rows(&text_ids)?)?;
        let codec = host(&self.codec_rows(&codec_ids)?)?;
        let (mut text, mut codec) = (text.chunks_exact(width), codec.chunks_exact(width));
        let mut out = vec![0f32; rows.len() * width];
        for (row, o) in rows.iter().zip(out.chunks_exact_mut(width)) {
            let text = row.text.and_then(|_| text.next());
            let codec = row.codec.and_then(|_| codec.next());
            for part in [text, codec].into_iter().flatten() {
                o.iter_mut().zip(part).for_each(|(a, b)| *a += b);
            }
        }
        Tensor::from_vec(out, (rows.len(), width), &Device::Cpu)
    }

    /// The talker row that follows the frame `codes`: the talker's row of group 0, the code
    /// predictor's row of each later group, and the pad text row, summed.
    pub fn frame_input(&self, codes: &[u32]) -> Result<Tensor> {
        if codes.len() != self.ids.groups {
            return Err(Error::msg(format!(
                "a frame holds {} codes, {} given",
                self.ids.groups,
                codes.len()
            )));
        }
        let mut row = host(&self.codec_rows(&codes[..1])?)?;
        for (table, code) in self.groups.iter().zip(&codes[1..]) {
            table.add_row(*code, &mut row)?;
        }
        row.iter_mut().zip(&self.pad_row).for_each(|(a, b)| *a += b);
        let width = row.len();
        Tensor::from_vec(row, (1, width), &Device::Cpu)
    }

    /// The prefill of the prompt `rows`.
    pub fn prefill(&mut self, rows: &Tensor) -> Result<EmbeddingStep> {
        self.model.forward_embeddings_prefill(rows, true)
    }

    /// The step of the talker row `input` at position `pos`.
    pub fn step(&mut self, input: &Tensor, pos: usize) -> Result<EmbeddingStep> {
        self.model.forward_embedding_step(input, pos, true)
    }

    /// The prompt rows of `request`.
    pub fn request_rows(&self, request: &Request<'_>) -> Result<Tensor> {
        let turns = prompt::Turns::of(&self.tokenizer, request.text, request.design)?;
        let language = self.ids.language(request.language)?;
        self.prompt_rows(&prompt::rows(&self.ids, &turns, language))
    }

    /// The frames of the utterance of `request`, each of a code per group.
    pub fn generate_with(
        &mut self,
        request: &Request<'_>,
        sub: &mut dyn SubCodes,
    ) -> Result<Vec<Vec<u32>>> {
        let rows = self.request_rows(request)?;
        self.frames(&rows, request.seed, sub)
    }

    /// The frames that follow the prompt `rows`, every draw seeded by `seed`.
    ///
    /// The loop stops at the end of speech, at the checkpoint's frame limit, or when the cache
    /// is full.
    /// The frame of the end of speech is not given.
    pub fn frames(
        &mut self,
        rows: &Tensor,
        seed: u64,
        sub: &mut dyn SubCodes,
    ) -> Result<Vec<Vec<u32>>> {
        let utterance =
            self.frames_with(rows, seed, sub, &Overrides::default(), &mut |_| Ok(()))?;
        Ok(utterance.frames)
    }

    /// Frames an utterance after a prompt of `rows` rows may hold: the checkpoint's limit and
    /// the room the cache leaves, 0 when the prompt fills the cache.
    pub fn budget(&self, rows: usize) -> usize {
        self.generation
            .max_frames
            .min(self.context.saturating_sub(rows))
    }

    /// [`Self::frames`] with the values `overrides` sets, and why the loop stopped.
    ///
    /// `sink` takes each frame as soon as it is drawn.
    /// The loop checks the cancel token of the thread before each frame.
    /// An error of `sink` stops the loop with that error.
    /// A prompt that leaves the cache no room gives no frame and stops at the budget.
    pub fn frames_with(
        &mut self,
        rows: &Tensor,
        seed: u64,
        sub: &mut dyn SubCodes,
        overrides: &Overrides,
        sink: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<Utterance> {
        let n = rows.dim(0)?;
        let budget = self.budget(n);
        let (limit, cap) = match overrides.max_frames {
            Some(asked) if asked < budget => (asked, Stop::Asked),
            _ => (budget, Stop::Budget),
        };
        if limit == 0 {
            return Ok(Utterance {
                frames: Vec::new(),
                stop: cap,
            });
        }
        let talker = overrides.apply(self.generation.talker);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let first = self.prefill(rows)?;
        let mut last = first;
        let mut drawn: Vec<u32> = Vec::new();
        let mut frames = Vec::new();
        for frame in 0..limit {
            crate::inference::serve::cancel::scoped::bail()?;
            let logits = last
                .logits
                .as_ref()
                .ok_or_else(|| Error::msg("the talker gave no logits"))?;
            let mut logits = host(logits)?;
            sampling::talker_logits(
                &mut logits,
                &self.ids,
                frame,
                &drawn,
                self.generation.repetition_penalty,
            );
            let code = sampling::draw(&logits, &talker, &mut rng)?;
            if code == self.ids.codec_eos {
                return Ok(Utterance {
                    frames,
                    stop: Stop::End,
                });
            }
            drawn.push(code);
            let group0 = self.codec_rows(&[code])?;
            let mut codes = vec![code];
            codes.extend(sub.sub_codes(&last.normed, &group0, &mut rng)?);
            let input = self.frame_input(&codes)?;
            sink(&codes)?;
            frames.push(codes);
            if frame + 1 < limit {
                last = self.step(&input, n + frame)?;
            }
        }
        Ok(Utterance { frames, stop: cap })
    }

    /// Where the talker's layers sit, by device.
    pub fn placement(&self) -> Vec<Placed> {
        layer_runs(&self.model)
    }
}

/// The prefix of the code predictor's tensors.
const PREDICTOR_PREFIX: &str = "talker.code_predictor.";

/// Bytes of the 3 blocks of a checkpoint, as stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blocks {
    /// Every tensor of the checkpoint's weights outside the code predictor.
    pub talker: u64,
    /// The tensors under the code predictor's prefix.
    pub predictor: u64,
    /// The decoder's tensors in the codec's file.
    pub decoder: u64,
}

impl Blocks {
    /// The blocks of the checkpoint at `dir`, read from the headers of its weights.
    pub fn of(dir: &std::path::Path) -> Result<Self> {
        let (_, shards) = crate::inference::load::awq_loader::checkpoint_shards(dir)
            .map_err(|e| Error::msg(e.to_string()))?;
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards)? };
        let (mut talker, mut predictor) = (0u64, 0u64);
        for (name, bytes) in st.byte_sizes() {
            match name.starts_with(PREDICTOR_PREFIX) {
                true => predictor += bytes as u64,
                false => talker += bytes as u64,
            }
        }
        Ok(Self {
            talker,
            predictor,
            decoder: decoder::decoder_bytes(&decoder::weights_of(dir))?,
        })
    }

    /// Bytes of the 3 blocks together.
    pub fn total(&self) -> u64 {
        self.talker + self.predictor + self.decoder
    }
}

/// The talker and its code predictor.
pub struct Qwen3Tts {
    pub talker: Talker,
    pub predictor: predictor::CodePredictor,
}

impl Qwen3Tts {
    /// The talker and the code predictor of the checkpoint at `dir`, the talker's cache
    /// holding `rows`.
    ///
    /// The code predictor loads first through `HeteroPlan` and the ladder.
    /// Its cache holds the 2 rows of its first pass and a row for each later pass of a frame.
    /// Its demand holds its input projection, its heads and the talker's text projection,
    /// beside its last layer.
    /// The talker then loads through `HeteroPlan` and the ladder.
    pub fn load(dir: &std::path::Path, rows: Rows) -> Result<Self> {
        let root = read_json(&dir.join("config.json"))?;
        let ids = SpeechIds::read(&root)?;
        let generation = Generation::read(&read_json(&dir.join("generation_config.json"))?)?;
        let (_, shards) = crate::inference::load::awq_loader::checkpoint_shards(dir)
            .map_err(|e| Error::msg(e.to_string()))?;
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards)? };
        let passes = ids.groups - 1;
        let beside = predictor::CodePredictor::glue_bytes(&st, passes) + Talker::glue_bytes(&st);
        let frame_rows = Rows::Exactly(2 + passes - 1);
        let loaded = class_load::load_class(dir, predictor::PREDICTOR_CLASS, frame_rows, beside)?;
        let talker = Talker::load_beside(dir, rows, Some(&loaded.last_device))?;
        let predictor = predictor::CodePredictor::new(
            loaded.model,
            &st,
            talker.groups.clone(),
            generation.predictor,
            &loaded.last_device,
        )?;
        Ok(Self { talker, predictor })
    }

    /// The device the code predictor's projection and heads run on.
    pub fn predictor_device(&self) -> Device {
        self.predictor.device()
    }

    /// The frames of the utterance of `request`, each of a code per group.
    pub fn generate(&mut self, request: &Request<'_>) -> Result<Vec<Vec<u32>>> {
        self.talker.generate_with(request, &mut self.predictor)
    }

    /// The utterance of `request` in one call, with the values `overrides` sets, each frame
    /// given to `sink` as soon as it is drawn.
    pub fn utterance(
        &mut self,
        request: &Request<'_>,
        overrides: &Overrides,
        sink: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<Utterance> {
        let predictor = overrides.apply(self.talker.generation.predictor);
        self.predictor.set_sampling(predictor);
        let rows = self.talker.request_rows(request)?;
        self.talker
            .frames_with(&rows, request.seed, &mut self.predictor, overrides, sink)
    }

    /// The speech of `request`, each frame given to `sink` as soon as it is drawn.
    ///
    /// A text is spoken in one call when the talker's cache holds its prompt rows and the
    /// checkpoint's frame limit.
    /// Otherwise a text of several parts is spoken a part per call: its paragraphs, else its
    /// sentences, each under the same rule.
    /// Each part is spoken with the request's seed, its frames following the frames before.
    /// A text spoken in one call ends where that call ends.
    /// The utterance holds every frame given and the stop of the last call.
    pub fn speak_streamed(
        &mut self,
        request: &Request<'_>,
        overrides: &Overrides,
        sink: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<Utterance> {
        let rows = self.talker.request_rows(request)?.dim(0)?;
        let holds = self.talker.context >= rows + self.talker.generation.max_frames;
        let parts = text::parts(request.text);
        if holds || parts.len() < 2 {
            let utterance = self.utterance(request, overrides, sink)?;
            if utterance.stop == Stop::Budget {
                if utterance.frames.is_empty() {
                    return Err(self.no_room(request));
                }
                tracing::warn!(
                    "a streamed call stopped at its budget of {} frames, the checkpoint's limit being {} and the talker's cache {} rows",
                    utterance.frames.len(),
                    self.talker.generation.max_frames,
                    self.talker.context
                );
            }
            return Ok(utterance);
        }
        let mut whole = Utterance {
            frames: Vec::new(),
            stop: Stop::End,
        };
        for part in &parts {
            let part = Request {
                text: part,
                ..*request
            };
            let spoken = self.speak_streamed(&part, overrides, sink)?;
            whole.frames.extend(spoken.frames);
            whole.stop = spoken.stop;
        }
        Ok(whole)
    }

    /// The error of a text whose prompt leaves the talker's cache no frame.
    fn no_room(&self, request: &Request<'_>) -> Error {
        Error::msg(format!(
            "a text of {} characters leaves the talker's cache of {} rows no frame",
            request.text.chars().count(),
            self.talker.context
        ))
    }

    /// The frames of the speech of `request`, the text split when one call cannot hold it.
    ///
    /// A call that stops at its budget leaves the text to its parts: its paragraphs, else its
    /// sentences.
    /// Each part is spoken with the request's seed, its frames following the frames before.
    /// A text that splits no further keeps the frames its budget held.
    pub fn speak(&mut self, request: &Request<'_>, overrides: &Overrides) -> Result<Vec<Vec<u32>>> {
        let whole = self.utterance(request, overrides, &mut |_| Ok(()))?;
        if whole.stop != Stop::Budget {
            return Ok(whole.frames);
        }
        let parts = text::parts(request.text);
        if parts.len() < 2 {
            if whole.frames.is_empty() {
                return Err(self.no_room(request));
            }
            return Ok(whole.frames);
        }
        let mut frames = Vec::new();
        for part in &parts {
            let part = Request {
                text: part,
                ..*request
            };
            frames.extend(self.speak(&part, overrides)?);
        }
        Ok(frames)
    }

    /// Where the layers of the talker and of the code predictor sit, by device.
    pub fn placement(&self) -> [Vec<Placed>; 2] {
        [self.talker.placement(), self.predictor.placement()]
    }
}
