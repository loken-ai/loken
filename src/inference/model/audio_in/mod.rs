//! Audio as a prompt's input: a speech encoder and a projector that turn a waveform into
//! embeddings the language model reads in place of a placeholder.
//!
//! * [`mel`]: the log-mel features the encoder is trained on.
//! * [`encoder`]: the speech encoder, read from an `mmproj` GGUF or a checkpoint.
//! * [`projector`]: stacking and the network into the model's embedding width.
//!
//! Every dimension is read from the files. What they do not say (the projector kind, the
//! placeholder, whether clips are padded to whole windows) is in `model_facts`.

pub mod encoder;
pub mod mel;
pub mod projector;
pub mod source;

use crate::inference::model::mmproj::MmprojAudio;
use crate::inference::model_facts::{audio_facts, AudioFacts};
use crate::tensor::quantized::gguf_file;
use crate::tensor::{Device, Error, Result, Tensor};
use encoder::Encoder;
use mel::MelSpec;
use projector::Projector;
use source::{Source, TensorReader};
use std::path::{Path, PathBuf};

/// The epsilon of the encoder's LayerNorms when the source does not carry one: the
/// framework default the checkpoints were trained with.
const LAYER_NORM_EPS: f32 = 1e-5;

/// A speech tower: waveform in, language-model embeddings out.
pub struct AudioTower {
    encoder: Encoder,
    projector: Projector,
    facts: AudioFacts,
    geo: MmprojAudio,
    spec: MelSpec,
}

impl AudioTower {
    /// The tower of an `mmproj` GGUF that carries an audio encoder.
    pub fn from_mmproj(path: &Path, dev: &Device) -> Result<Self> {
        let file =
            gguf_file::open_mapped(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let geo = MmprojAudio::read(&file)?;
        let source = Source::Gguf(file);
        Self::build(&source, geo, dev)
    }

    /// The tower of a checkpoint directory whose `config.json` declares an `audio_config`.
    pub fn from_checkpoint(dir: &Path, dev: &Device) -> Result<Self> {
        let config = read_config(dir)?;
        let audio = config.get("audio_config").ok_or_else(|| {
            Error(format!(
                "{}: config.json has no audio_config",
                dir.display()
            ))
        })?;
        let projector_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error(format!("{}: config.json has no model_type", dir.display())))?
            .to_string();
        let number = |keys: &[&str]| -> Result<usize> {
            keys.iter()
                .find_map(|k| audio.get(*k).and_then(|v| v.as_u64()))
                .map(|v| v as usize)
                .ok_or_else(|| Error(format!("audio_config has none of {keys:?}")))
        };
        let shards = weight_files(dir)?;
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards)? };
        let source = Source::Checkpoint(st);
        let reader = TensorReader {
            source: &source,
            dev,
        };
        let width = number(&["d_model", "hidden_size"])?;
        let shape = |name: &str| {
            reader
                .shape_of(name)
                .ok_or_else(|| Error(format!("{name}: missing from the checkpoint")))
        };
        let stack = shape("mm.a.mlp.1.weight")?[1] / width.max(1);
        let projection = shape("mm.a.mlp.2.weight")?[0];
        let geo = MmprojAudio {
            projector_type,
            width,
            heads: number(&["encoder_attention_heads", "num_attention_heads"])?,
            blocks: number(&["encoder_layers", "num_hidden_layers"])?,
            feed_forward: number(&["encoder_ffn_dim", "intermediate_size"])?,
            n_mels: number(&["num_mel_bins"])?,
            stack,
            projection,
            eps: LAYER_NORM_EPS,
        };
        if let Some(declared) = config.get("stack_factor").and_then(|v| v.as_u64()) {
            if declared as usize != stack {
                return Err(Error(format!(
                    "config.json stack_factor {declared}, the projector's shape implies {stack}"
                )));
            }
        }
        Self::build(&source, geo, dev)
    }

    fn build(source: &Source, geo: MmprojAudio, dev: &Device) -> Result<Self> {
        let facts = *audio_facts(&geo.projector_type).ok_or_else(|| {
            Error(format!(
                "no audio tower is implemented for projector type {:?}",
                geo.projector_type
            ))
        })?;
        let reader = TensorReader { source, dev };
        let encoder = Encoder::load(&reader, &geo)?;
        let projector = Projector::load(&reader, &geo, facts.projector)?;
        Ok(Self {
            encoder,
            projector,
            facts,
            spec: MelSpec::speech(geo.n_mels),
            geo,
        })
    }

    pub fn facts(&self) -> &AudioFacts {
        &self.facts
    }

    /// Samples per second the clip must arrive at.
    pub fn sample_rate(&self) -> usize {
        self.spec.sample_rate
    }

    /// Width of the embeddings: the language model's embedding width.
    pub fn embedding_width(&self) -> usize {
        self.geo.projection
    }

    /// The samples a clip is extended to before its spectrogram: whole windows when the
    /// model was trained on those, else the hop (and at least two hops).
    fn padded_len(&self, samples: usize) -> usize {
        if self.facts.pads_to_window {
            samples.div_ceil(self.spec.window_samples()).max(1) * self.spec.window_samples()
        } else {
            samples.max(2 * self.spec.hop).div_ceil(self.spec.hop) * self.spec.hop
        }
    }

    /// Embeddings a clip of `samples` yields, known before the encoder runs: what the KV
    /// window has to hold.
    pub fn embeds_for(&self, samples: usize) -> usize {
        let frames = self.padded_len(samples) / self.spec.hop;
        let window = self.spec.window_frames();
        let (full, rest) = (frames / window, frames % window);
        full * self.projector.rows_for(window.div_ceil(2))
            + if rest > 0 {
                self.projector.rows_for(rest.div_ceil(2))
            } else {
                0
            }
    }

    /// Encoder frames `[ceil(T / 2), width]` of one window's spectrogram `[1, n_mels, T]`.
    pub fn encode_window(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel)
    }

    /// Embeddings `[ceil(F / stack), width]` of `F` encoder frames.
    pub fn project(&self, frames: &Tensor) -> Result<Tensor> {
        self.projector.forward(frames)
    }

    /// The spectrogram of mono `pcm` at [`Self::sample_rate`] as this model is fed it,
    /// `[n_mels, frames]` row-major, and its frame count.
    pub fn features(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let mut audio = pcm.to_vec();
        audio.resize(self.padded_len(pcm.len()), 0.0);
        mel::log_mel(&audio, &self.spec)
    }

    /// Embeddings `[1, n, width]` of mono `pcm` at [`Self::sample_rate`], windows in order.
    pub fn embed(&self, pcm: &[f32]) -> Result<Tensor> {
        let (mel, frames) = self.features(pcm);
        let window = self.spec.window_frames();
        let dev = self.encoder_device();
        let mut parts = Vec::new();
        let mut from = 0;
        while from < frames {
            let n = window.min(frames - from);
            let slice = mel::window_of(&mel, self.geo.n_mels, frames, from, n);
            let t = Tensor::from_vec(slice, (1, self.geo.n_mels, n), &dev)?;
            parts.push(self.project(&self.encode_window(&t)?)?);
            from += n;
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 0)?.unsqueeze(0)
    }

    fn encoder_device(&self) -> Device {
        self.encoder.device().clone()
    }
}

fn read_config(dir: &Path) -> Result<serde_json::Value> {
    let path = dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))
}

/// The safetensors shards of a checkpoint directory, in name order.
fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error(format!("{}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error(format!("{}: no safetensors file", dir.display())));
    }
    Ok(files)
}

/// Whether `dir` is a checkpoint whose configuration declares a speech tower this build
/// reads: an `audio_config` and a model type the facts table has a row for.
pub fn checkpoint_has_tower(dir: &Path) -> bool {
    read_config(dir).is_ok_and(|c| {
        c.get("audio_config").is_some()
            && c.get("model_type")
                .and_then(|v| v.as_str())
                .is_some_and(|t| audio_facts(t).is_some())
    })
}

/// Whether the GGUF at `path` is an `mmproj` carrying a speech tower this build reads.
pub fn mmproj_has_tower(path: &Path) -> bool {
    gguf_file::open_header(path)
        .ok()
        .and_then(|c| MmprojAudio::read(&c).ok())
        .is_some_and(|g| audio_facts(&g.projector_type).is_some())
}

/// Bytes the tower of the `mmproj` at `path` holds once loaded: every tensor is F32 then.
pub fn resident_bytes(path: &Path) -> u64 {
    gguf_file::open_header(path).map_or(0, |c| {
        c.tensor_infos
            .values()
            .map(|i| i.shape.elem_count() as u64 * std::mem::size_of::<f32>() as u64)
            .sum()
    })
}

/// Reading the toy towers and references the parity gates are written against.
#[cfg(test)]
pub(crate) mod gate {
    use crate::tensor::safetensors::MmapedSafetensors;
    use crate::tensor::{Device, Tensor};
    use std::path::{Path, PathBuf};

    /// Where the committed vectors of a toy tower live.
    pub fn vectors(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/vectors/audio_in")
            .join(name)
    }

    /// A named tensor of a vectors file, on the host.
    pub fn tensor(dir: &Path, name: &str) -> Tensor {
        let st = unsafe { MmapedSafetensors::new(dir.join("vectors.safetensors")) }.unwrap();
        st.load(name, &Device::Cpu).unwrap()
    }

    pub fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The largest absolute difference, and the largest magnitude of the reference.
    pub fn compare(got: &Tensor, want: &Tensor) -> (f32, f32) {
        assert_eq!(got.dims(), want.dims(), "shapes differ");
        let (g, w) = (values(got), values(want));
        let diff = g
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = w.iter().map(|v| v.abs()).fold(0f32, f32::max);
        (diff, scale)
    }
}

/// The `mmproj` route reads what the checkpoint route reads: each toy checkpoint is rewritten
/// as an `mmproj` GGUF (the canonical names, the `clip.audio.*` keys) and the two towers must
/// give the same embeddings.
#[cfg(test)]
mod route_tests {
    use super::gate::{compare, tensor, vectors};
    use super::source::checkpoint_name;
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Value;
    use crate::tensor::quantized::GgmlDType;
    use crate::tensor::safetensors::MmapedSafetensors;

    fn canonical_names(blocks: usize, gated: bool) -> Vec<String> {
        let mut names = Vec::new();
        for kind in ["weight", "bias"] {
            for part in ["a.conv1d.1", "a.conv1d.2", "a.post_ln"] {
                names.push(format!("{part}.{kind}"));
            }
        }
        names.push("a.position_embd.weight".to_string());
        for i in 0..blocks {
            for part in [
                "attn_q", "attn_v", "attn_out", "ln1", "ln2", "ffn_up", "ffn_down",
            ] {
                for kind in ["weight", "bias"] {
                    names.push(format!("a.blk.{i}.{part}.{kind}"));
                }
            }
            names.push(format!("a.blk.{i}.attn_k.weight"));
        }
        names.extend(["mm.a.mlp.1.weight", "mm.a.mlp.2.weight"].map(String::from));
        if gated {
            names.extend(["mm.a.norm_pre.weight", "mm.a.norm_mid.weight"].map(String::from));
        }
        names
    }

    fn rewrite_as_mmproj(
        family: &str,
        projector_type: &str,
        feed_forward: u32,
        gated: bool,
    ) -> PathBuf {
        let dir = vectors(family);
        let st = unsafe { MmapedSafetensors::new(dir.join("model.safetensors")) }.unwrap();
        let entries: Vec<GgufEntry> = canonical_names(2, gated)
            .into_iter()
            .map(|name| {
                let t = st
                    .load(&checkpoint_name(&name).unwrap(), &Device::Cpu)
                    .unwrap();
                let data = t
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                GgufEntry {
                    dims: t.dims().to_vec(),
                    name,
                    dtype: GgmlDType::F32,
                    data,
                }
            })
            .collect();
        let s = |v: &str| Value::String(v.to_string());
        let metadata = vec![
            ("general.architecture".to_string(), s("clip")),
            ("clip.has_audio_encoder".to_string(), Value::Bool(true)),
            ("clip.projector_type".to_string(), s(projector_type)),
            ("clip.audio.embedding_length".to_string(), Value::U32(32)),
            (
                "clip.audio.feed_forward_length".to_string(),
                Value::U32(feed_forward),
            ),
            ("clip.audio.block_count".to_string(), Value::U32(2)),
            ("clip.audio.attention.head_count".to_string(), Value::U32(4)),
            ("clip.audio.num_mel_bins".to_string(), Value::U32(16)),
            (
                "clip.audio.projector.stack_factor".to_string(),
                Value::U32(4),
            ),
            ("clip.audio.projection_dim".to_string(), Value::U32(24)),
            (
                "clip.audio.attention.layer_norm_epsilon".to_string(),
                Value::F32(1e-5),
            ),
        ];
        let path = std::env::temp_dir().join(format!(
            "loken-audio-in-{family}-{}.gguf",
            std::process::id()
        ));
        write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
        path
    }

    fn the_routes_agree(family: &str, projector_type: &str, feed_forward: u32, gated: bool) {
        let path = rewrite_as_mmproj(family, projector_type, feed_forward, gated);
        let dir = vectors(family);
        let from_file = AudioTower::from_mmproj(&path, &Device::Cpu).unwrap();
        let from_dir = AudioTower::from_checkpoint(&dir, &Device::Cpu).unwrap();
        let declared = mmproj_has_tower(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            declared,
            "the header of {family} names a tower this build reads"
        );
        let mel = tensor(&dir, "mel").unsqueeze(0).unwrap();
        let a = from_file
            .project(&from_file.encode_window(&mel).unwrap())
            .unwrap();
        let b = from_dir
            .project(&from_dir.encode_window(&mel).unwrap())
            .unwrap();
        let (diff, scale) = compare(&a, &b);
        assert!(
            diff <= 1e-6 * scale.max(1.0),
            "{family}: {diff} off, scale {scale}"
        );
        assert_eq!(from_file.embedding_width(), 24);
    }

    #[test]
    fn a_gelu_projector_reads_the_same_from_an_mmproj() {
        the_routes_agree("voxtral", "voxtral", 128, false);
    }

    #[test]
    fn a_gated_projector_reads_the_same_from_an_mmproj() {
        the_routes_agree("ultravox", "ultravox", 64, true);
    }
}
