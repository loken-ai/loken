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
use crate::inference::model_facts::{audio_facts, audio_type_of_native, AudioFacts};
use crate::tensor::quantized::gguf_file;
use crate::tensor::{Device, Error, Result, Tensor};
use encoder::Encoder;
use mel::MelSpec;
use projector::Projector;
use source::{Source, TensorReader};
use std::path::Path;

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

    /// The tower of a checkpoint directory: a `config.json` that declares an `audio_config`,
    /// or a native layout's `params.json` that declares a speech tower.
    pub fn from_checkpoint(dir: &Path, dev: &Device) -> Result<Self> {
        let (native, shards) = checkpoint_layout(dir)?;
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards)? };
        let params = Params::read(dir, native)?;
        let source = if native {
            Source::Native {
                st,
                positions: params.positions,
            }
        } else {
            Source::Checkpoint(st)
        };
        let reader = TensorReader {
            source: &source,
            dev,
        };
        let shape = |name: &str| {
            reader
                .shape_of(name)
                .ok_or_else(|| Error(format!("{name}: missing from the checkpoint")))
        };
        let stack = shape("mm.a.mlp.1.weight")?[1] / params.width.max(1);
        let projection = shape("mm.a.mlp.2.weight")?[0];
        let geo = MmprojAudio {
            projector_type: params.projector_type,
            width: params.width,
            heads: params.heads,
            blocks: params.blocks,
            feed_forward: params.feed_forward,
            n_mels: params.n_mels,
            stack,
            projection,
            eps: LAYER_NORM_EPS,
        };
        if let Some(declared) = params.stack_factor {
            if declared != stack {
                return Err(Error(format!(
                    "{}: the declared stack factor is {declared}, the projector's shape implies {stack}",
                    dir.display()
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

/// The geometry a checkpoint's configuration declares for its speech tower.
struct Params {
    projector_type: String,
    width: usize,
    heads: usize,
    blocks: usize,
    feed_forward: usize,
    n_mels: usize,
    stack_factor: Option<usize>,
    /// Rows of the encoder's position table.
    positions: usize,
}

impl Params {
    fn read(dir: &Path, native: bool) -> Result<Self> {
        let file = if native { "params.json" } else { "config.json" };
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let missing = |what: &str| Error(format!("{}: {what}", path.display()));
        if native {
            let projector_type = audio_type_of_native(&json).ok_or_else(|| {
                missing("no speech tower the facts table has a row for under multimodal")
            })?;
            let key = audio_facts(projector_type)
                .and_then(|f| f.native_key)
                .unwrap_or_default();
            let encoder = json
                .pointer(&format!("/multimodal/{key}/encoder_args"))
                .ok_or_else(|| missing("no encoder_args"))?;
            let number = |o: &serde_json::Value, k: &str| {
                o.get(k)
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .ok_or_else(|| missing(&format!("no {k}")))
            };
            let encoding = encoder
                .get("audio_encoding_args")
                .ok_or_else(|| missing("no audio_encoding_args"))?;
            return Ok(Self {
                projector_type: projector_type.to_string(),
                width: number(encoder, "dim")?,
                heads: number(encoder, "n_heads")?,
                blocks: number(encoder, "n_layers")?,
                feed_forward: number(encoder, "hidden_dim")?,
                n_mels: number(encoding, "num_mel_bins")?,
                stack_factor: json
                    .pointer(&format!(
                        "/multimodal/{key}/downsample_args/downsample_factor"
                    ))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize),
                positions: number(encoder, "max_source_positions")?,
            });
        }
        let audio = json
            .get("audio_config")
            .ok_or_else(|| missing("no audio_config"))?;
        let projector_type = json
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| missing("no model_type"))?
            .to_string();
        let number = |keys: &[&str]| -> Result<usize> {
            keys.iter()
                .find_map(|k| audio.get(*k).and_then(|v| v.as_u64()))
                .map(|v| v as usize)
                .ok_or_else(|| missing(&format!("audio_config has none of {keys:?}")))
        };
        Ok(Self {
            projector_type,
            width: number(&["d_model", "hidden_size"])?,
            heads: number(&["encoder_attention_heads", "num_attention_heads"])?,
            blocks: number(&["encoder_layers", "num_hidden_layers"])?,
            feed_forward: number(&["encoder_ffn_dim", "intermediate_size"])?,
            n_mels: number(&["num_mel_bins"])?,
            stack_factor: json
                .get("stack_factor")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            positions: 0,
        })
    }
}

/// Whether `dir` reads in the vendor's native layout, and its weight files; the error names
/// what the directory lacks.
fn checkpoint_layout(dir: &Path) -> Result<(bool, Vec<String>)> {
    crate::inference::load::awq_loader::checkpoint_shards(dir).map_err(|e| Error(e.to_string()))
}

/// Whether `dir` is a checkpoint whose configuration declares a speech tower this build
/// reads: an `audio_config` and a model type the facts table has a row for, or a native
/// layout's `multimodal` object the table has a row for.
pub fn checkpoint_has_tower(dir: &Path) -> bool {
    let json = |file: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(dir.join(file)).ok()?).ok()
    };
    if let Some(c) = json("config.json") {
        return c.get("audio_config").is_some()
            && c.get("model_type")
                .and_then(|v| v.as_str())
                .is_some_and(|t| audio_facts(t).is_some());
    }
    json("params.json").is_some_and(|p| audio_type_of_native(&p).is_some())
}

/// Whether the GGUF at `path` is an `mmproj` carrying a speech tower this build reads.
pub fn mmproj_has_tower(path: &Path) -> bool {
    gguf_file::open_header(path)
        .ok()
        .and_then(|c| MmprojAudio::read(&c).ok())
        .is_some_and(|g| audio_facts(&g.projector_type).is_some())
}

/// Bytes the tower at `path` (an `mmproj` file or a checkpoint directory) holds once loaded:
/// every tensor is F32 then.
pub fn resident_bytes(path: &Path) -> u64 {
    if path.is_dir() {
        let Ok((native, shards)) = checkpoint_layout(path) else {
            return 0;
        };
        let Ok(st) = (unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards) })
        else {
            return 0;
        };
        return st
            .names()
            .iter()
            .filter(|n| {
                if native {
                    n.starts_with(source::NATIVE_TOWER) && !n.ends_with("tok_embeddings.weight")
                } else {
                    n.starts_with("audio_tower.") || n.starts_with("multi_modal_projector.")
                }
            })
            .filter_map(|n| st.shape_of(n))
            .map(|d| d.iter().product::<usize>() as u64 * std::mem::size_of::<f32>() as u64)
            .sum();
    }
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
    use std::path::PathBuf;

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

    /// Writes `tensors` (name, dims, values) as one F32 safetensors file.
    fn write_checkpoint(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let bytes: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, _, v)| v.iter().flat_map(|x| x.to_le_bytes()).collect())
            .collect();
        let views: Vec<_> = tensors
            .iter()
            .zip(&bytes)
            .map(|((n, d, _), b)| {
                (
                    n.clone(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, d.clone(), b)
                        .unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, path).unwrap();
    }

    const TOY_PARAMS: &str = r#"{"multimodal": {"whisper_model_args": {
        "encoder_args": {"dim": 32, "n_layers": 2, "head_dim": 8, "hidden_dim": 128,
            "n_heads": 4, "vocab_size": 51866, "max_source_positions": 1500,
            "audio_encoding_args": {"num_mel_bins": 16}},
        "downsample_args": {"downsample_factor": 4}}}}"#;

    /// The toy Voxtral tower written in both layouts of a checkpoint (the position table of
    /// the framework's layout being the sinusoids the native one computes) gives the same
    /// embeddings, and each layout is told apart by its files.
    #[test]
    fn both_checkpoint_layouts_give_the_same_tower() {
        let toy = vectors("voxtral");
        let st = unsafe { MmapedSafetensors::new(toy.join("model.safetensors")) }.unwrap();
        let read = |name: &str| -> (Vec<usize>, Vec<f32>) {
            let t = st.load(name, &Device::Cpu).unwrap();
            (t.dims().to_vec(), super::gate::values(&t))
        };
        let root = std::env::temp_dir().join(format!("loken-layouts-{}", std::process::id()));
        let (framework, native) = (root.join("framework"), root.join("native"));
        std::fs::create_dir_all(&framework).unwrap();
        std::fs::create_dir_all(&native).unwrap();
        let mut fw = Vec::new();
        let mut nv = Vec::new();
        for name in canonical_names(2, false) {
            let (dims, values) = if name == "a.position_embd.weight" {
                let table = super::source::sinusoids(1500, 32, &Device::Cpu).unwrap();
                (table.dims().to_vec(), super::gate::values(&table))
            } else {
                read(&checkpoint_name(&name).unwrap())
            };
            fw.push((
                checkpoint_name(&name).unwrap(),
                dims.clone(),
                values.clone(),
            ));
            if let Some(n) = super::source::native_name(&name) {
                nv.push((n, dims, values));
            }
        }
        write_checkpoint(&framework.join("model.safetensors"), &fw);
        write_checkpoint(&native.join("consolidated.safetensors"), &nv);
        std::fs::copy(toy.join("config.json"), framework.join("config.json")).unwrap();
        std::fs::write(native.join("params.json"), TOY_PARAMS).unwrap();

        assert!(checkpoint_has_tower(&framework));
        assert!(checkpoint_has_tower(&native));
        assert!(!checkpoint_layout(&framework).unwrap().0);
        assert!(checkpoint_layout(&native).unwrap().0);
        assert_eq!(resident_bytes(&framework) > 0, resident_bytes(&native) > 0);
        let a = AudioTower::from_checkpoint(&framework, &Device::Cpu).unwrap();
        let b = AudioTower::from_checkpoint(&native, &Device::Cpu).unwrap();
        let mel = tensor(&toy, "mel").unsqueeze(0).unwrap();
        let ea = a.project(&a.encode_window(&mel).unwrap()).unwrap();
        let eb = b.project(&b.encode_window(&mel).unwrap()).unwrap();
        let (diff, _) = compare(&ea, &eb);
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(diff, 0.0);
        assert_eq!(b.embedding_width(), 24);
    }

    /// A directory with neither layout is refused with what it lacks of each.
    #[test]
    fn a_directory_of_no_known_layout_is_refused_by_what_it_lacks() {
        let dir = std::env::temp_dir().join(format!("loken-no-layout-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"architectures": ["X"]}"#).unwrap();
        let message = match AudioTower::from_checkpoint(&dir, &Device::Cpu) {
            Ok(_) => panic!("a directory with no weights loaded"),
            Err(e) => e.to_string(),
        };
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            message.contains("numbered *.safetensors shards"),
            "{message}"
        );
        assert!(message.contains("params.json"), "{message}");
        assert!(message.contains("consolidated*.safetensors"), "{message}");
        assert!(!message.contains("lacks: config.json"), "{message}");
    }
}
