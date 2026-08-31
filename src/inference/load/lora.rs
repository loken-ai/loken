//! LoRA adapters for the image models.
//!
//! A LoRA is a rank-r correction to a handful of projections, trained on a style, a
//! character or a concept. It is tens of MB against a checkpoint's several GB, which is
//! why people keep libraries of them - and why "can I use my LoRAs" decides whether a
//! server is usable at all.
//!
//! APPLIED, NOT MERGED. The delta could be folded into the base weight, and every
//! reference implementation offers that. It is the wrong choice here: our image weights
//! are resident QUANTISED (Q8_0 from GGUF, fp8 re-quantised to Q8_0 for the Ray
//! checkpoints), so merging would mean dequantise, add, requantise - paying the memory
//! of a dense copy and taking a rounding pass over every affected tensor. Applying
//! `((x @ down) @ up) * scale` at runtime leaves the base untouched, lets several
//! adapters compose by addition, and makes unloading one a drop rather than a reload.
//! The arithmetic is negligible: r is 8-128 against dimensions in the thousands.
//!
//! KEY NAMING. The ecosystem writes LoRA keys by flattening the module path with
//! underscores and prefixing the network: `attn1.to_q` inside
//! `down_blocks.1.attentions.0.transformer_blocks.0` becomes
//! `lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q`. That mapping
//! is not invertible - `down_blocks` and `down.blocks` flatten identically - so this
//! goes in the only direction that is well defined: from OUR module paths to the
//! expected key. A module that is not in the file simply has no adapter.

use crate::tensor::lora::LoraDelta;
use crate::tensor::{DType, Device, Error, Result, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The directory adapters are served from, fixed once at startup from the config.
///
/// Requests name an adapter; they never carry a path. A server that opened
/// `safetensors` at a caller-supplied path would be handing out a file-read probe:
/// the error text alone distinguishes "no such file" from "not a tensor file", which
/// is enough to walk a filesystem. Confining resolution to one directory removes the
/// question instead of trying to filter it.
static LORA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Set the adapter directory. Called once during startup; later calls are ignored.
pub fn set_lora_dir(dir: PathBuf) {
    let _ = LORA_DIR.set(dir);
}

pub fn lora_dir() -> Option<&'static Path> {
    LORA_DIR.get().map(PathBuf::as_path)
}

/// Turn a requested adapter name into a path inside the configured directory.
///
/// The name must be a plain file name: no separator, no `..`, no absolute path. The
/// `.safetensors` suffix is optional so callers can say `lcm-lora-sdxl`. The resolved
/// path is canonicalised and re-checked against the canonical directory, which is what
/// actually stops a symlink inside the directory from pointing back out of it.
pub fn resolve(name: &str) -> std::result::Result<PathBuf, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("lora: empty name".to_string());
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!(
            "lora '{name}': name a file in the server's lora directory, not a path"
        ));
    }
    let Some(dir) = lora_dir() else {
        return Err("lora: no lora directory is configured on this server".to_string());
    };
    let root = dir.canonicalize().map_err(|_| {
        format!(
            "lora: the configured directory {} does not exist",
            dir.display()
        )
    })?;
    let mut cand = root.join(name);
    if cand.extension().is_none() {
        cand.set_extension("safetensors");
    }
    let full = cand
        .canonicalize()
        .map_err(|_| format!("lora '{name}': not found in {}", root.display()))?;
    if !full.starts_with(&root) {
        return Err(format!(
            "lora '{name}': resolves outside the lora directory"
        ));
    }
    Ok(full)
}

/// The adapters available to callers, by name, sorted. Empty if unconfigured.
pub fn available() -> Vec<String> {
    let Some(dir) = lora_dir() else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|e| {
            e.path()
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();
    out.sort();
    out
}

/// Which architecture an adapter was trained for, from its own tensor names.
///
/// Read from the header alone, so it costs nothing and cannot disagree with the file.
/// Callers need it because an adapter applied to the wrong family matches NOTHING and
/// fails the render - which is what a user gets today by picking any adapter from a
/// list that offers all of them whatever model is selected. Eight failed renders in a
/// row, and the only clue is a message about module names.
///
/// `None` when the layout is not one of the two shapes shipped here; such an adapter is
/// still offered, because refusing to name it is not a reason to hide it.
pub fn target_family(name: &str) -> Option<&'static str> {
    use std::io::Read;
    let path = resolve(name).ok()?;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).ok()?;
    let hlen = u64::from_le_bytes(len8) as usize;
    // not-a-vram-size: a bound on a header length read out of the file.
    if hlen == 0 || hlen > 32 * 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; hlen];
    f.read_exact(&mut buf).ok()?;
    let hdr = std::str::from_utf8(&buf).ok()?;
    // The dual/single-stream block naming is unique to one family; the UNet's
    // down/up-block numbering to the other.
    if hdr.contains("double_blocks") || hdr.contains("single_blocks") {
        Some("flux")
    } else if hdr.contains("down_blocks") || hdr.contains("up_blocks") {
        Some("sdxl")
    } else {
        None
    }
}

/// The sampling an adapter REQUIRES, when it carries one.
///
/// Most adapters change what a model draws and leave how it is sampled alone. A
/// consistency adapter is the other kind: it does not restyle anything, it retrains the
/// model to reach an image in a handful of steps at almost no guidance. Run at the base
/// model's own recipe - twenty-five steps at guidance seven - it does not produce "the
/// same picture, better": it produces an overcooked, washed-out one, and reads as an
/// adapter that degrades rather than one that was never given the regime it needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingRegime {
    pub steps: usize,
    pub guidance: f64,
}

/// Read the regime an adapter declares, from the FILE.
///
/// safetensors carries a `__metadata__` object, and a consistency adapter says so there:
/// the one shipped here records `"modelspec.merged_from": "sdxl_LCM_lora"`. Reading it
/// beats matching on the filename, which is a label anyone can change and which the
/// other adapter here does not carry at all - it has no metadata whatsoever.
///
/// The filename is still consulted as a FALLBACK, for exactly that case: a file with no
/// metadata cannot declare anything, and refusing to look at its name would mean a
/// correctly-named adapter silently getting the wrong recipe.
///
/// The figures are the published recipe for this family of adapters: four steps, and a
/// guidance of one - a consistency model is trained to need none, and pushing it re-adds
/// the contrast the distillation removed.
pub fn sampling_regime(name: &str) -> Option<SamplingRegime> {
    is_consistency(header_text(name).as_deref(), name).then_some(CONSISTENCY)
}

/// The regime this family of adapters is published with: four steps, and a guidance of
/// one - a consistency model is trained to need none, and pushing it re-adds the contrast
/// the distillation removed.
const CONSISTENCY: SamplingRegime = SamplingRegime {
    steps: 4,
    guidance: 1.0,
};

/// Does this adapter retrain the sampling rather than the style?
///
/// The FILE is asked first. The name is a fallback for a file that declares nothing -
/// which is not hypothetical: of the two adapters shipped here, one carries a full
/// modelspec and the other has no metadata at all.
fn is_consistency(header: Option<&str>, name: &str) -> bool {
    let declared = header.is_some_and(|h| {
        let lower = h.to_lowercase();
        lower.contains("lcm") || lower.contains("consistency")
    });
    declared || name.to_lowercase().contains("lcm")
}

/// The safetensors header of an adapter, as text.
fn header_text(name: &str) -> Option<String> {
    use std::io::Read;
    let path = resolve(name).ok()?;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).ok()?;
    let hlen = u64::from_le_bytes(len8) as usize;
    // not-a-vram-size: a bound on a header length read out of the file.
    if hlen == 0 || hlen > 32 * 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; hlen];
    f.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// The `alpha/rank` convention: a LoRA stores `alpha` and the effective contribution is
/// scaled by `alpha / rank`, so that changing the rank does not change the strength.
/// Files that omit alpha are trained at `alpha == rank`, i.e. a scale of 1.
const DEFAULT_ALPHA_IS_RANK: f32 = 1.0;

/// One loaded adapter file: the low-rank pairs, keyed by the flattened module name.
pub struct LoraFile {
    /// `lora_unet_..._to_q` -> (down `[r, in]`, up `[out, r]`, alpha/rank)
    entries: HashMap<String, (Tensor, Tensor, f32)>,
}

impl LoraFile {
    /// Read a `.safetensors` adapter onto `device`.
    ///
    /// Both the diffusers layout (`.lora_down.weight` / `.lora_up.weight`) and the PEFT
    /// one (`.lora_A.weight` / `.lora_B.weight`) appear in the wild; both are accepted
    /// because a user's folder will contain both and neither is "wrong".
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let vb = unsafe { crate::tensor::VarBuilder::from_files(&[path], DType::F32, device) }?;
        let names = vb.tensor_names();

        // Collect the halves first: a pair is only usable once both sides are present.
        let mut downs: HashMap<String, String> = HashMap::new();
        let mut ups: HashMap<String, String> = HashMap::new();
        let mut alphas: HashMap<String, String> = HashMap::new();
        // THREE naming conventions are in circulation and a user's folder will hold all
        // of them. Diffusers writes `.lora_down/.lora_up`, PEFT writes `.lora_A/.lora_B`,
        // and the XLabs Flux adapters write a bare `.down/.up` under a `processor`
        // subpath. Accepting only the first two meant an XLabs file did not even LOAD -
        // it produced no pairs at all, so the adapter reported as matching nothing.
        for n in &names {
            if let Some(base) = n
                .strip_suffix(".lora_down.weight")
                .or_else(|| n.strip_suffix(".lora_A.weight"))
                .or_else(|| n.strip_suffix(".down.weight"))
            {
                downs.insert(base.to_string(), n.clone());
            } else if let Some(base) = n
                .strip_suffix(".lora_up.weight")
                .or_else(|| n.strip_suffix(".lora_B.weight"))
                .or_else(|| n.strip_suffix(".up.weight"))
            {
                ups.insert(base.to_string(), n.clone());
            } else if let Some(base) = n.strip_suffix(".alpha") {
                alphas.insert(base.to_string(), n.clone());
            }
        }

        let mut entries = HashMap::new();
        for (base, dname) in downs {
            let Some(uname) = ups.get(&base) else {
                continue;
            };
            let down = vb.get_by_name(&dname)?;
            let up = vb.get_by_name(uname)?;
            // down is [r, in], up is [out, r] as stored.
            let r = down.dims()[0];
            let scale = match alphas.get(&base) {
                Some(a) => {
                    let alpha = vb.get_by_name(a)?.to_device(&Device::Cpu)?.to_vec_f32();
                    alpha.first().copied().unwrap_or(r as f32) / r.max(1) as f32
                }
                None => DEFAULT_ALPHA_IS_RANK,
            };
            entries.insert(base, (down, up, scale));
        }
        Ok(Self { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The adapter for a module path, already in this layer's transposed convention.
    ///
    /// `module_path` is OUR name (`down_blocks.1...attn1.to_q`); the flattening to the
    /// file's key happens here, in the one direction that is unambiguous.
    pub fn delta_for(&self, module_path: &str, strength: f32) -> Result<Option<LoraDelta>> {
        self.delta_for_key(
            &format!("lora_unet_{}", module_path.replace('.', "_")),
            strength,
        )
    }

    /// The adapter stored under an EXACT key, for layouts that are not the flattened
    /// `lora_unet_` one - the XLabs Flux adapters key on the checkpoint path itself.
    pub fn delta_for_key(&self, key: &str, strength: f32) -> Result<Option<LoraDelta>> {
        let Some((down, up, scale)) = self.entries.get(key) else {
            return Ok(None);
        };
        // CONVOLUTION adapters are 4-D and are not this. A real SDXL LoRA carries both:
        // the LCM adapter has 739 linear and 49 conv entries. Transposing a 4-D tensor
        // as if it were a matrix would either error or, worse, succeed on the wrong axes
        // and corrupt the projection - so they are skipped explicitly, and applying them
        // is a separate piece of work rather than something to fake here.
        if down.dims().len() != 2 || up.dims().len() != 2 {
            return Ok(None);
        }
        // Stored [r, in] and [out, r]; the layer wants [in, r] and [r, out].
        let down_t = down.transpose(0, 1)?.contiguous()?;
        let up_t = up.transpose(0, 1)?.contiguous()?;
        Ok(Some(LoraDelta {
            down: down_t,
            up: up_t,
            scale: scale * strength,
        }))
    }

    /// The CONVOLUTION adapter stored for a module path, in its native 4-D layout.
    ///
    /// Kept separate from `delta_for` because the two are not interchangeable: a linear
    /// delta is two matrices the layer multiplies, a convolution delta is two kernels it
    /// convolves, and the transpose that makes the first usable would corrupt the second.
    /// `down` is `[r, in, kh, kw]` and `up` is `[out, r, 1, 1]`, exactly as stored.
    pub fn conv_delta_for(&self, module_path: &str, strength: f32) -> Result<Option<LoraDelta>> {
        let key = format!("lora_unet_{}", module_path.replace('.', "_"));
        let Some((down, up, scale)) = self.entries.get(&key) else {
            return Ok(None);
        };
        if down.dims().len() != 4 || up.dims().len() != 4 {
            return Ok(None);
        }
        Ok(Some(LoraDelta {
            down: down.clone(),
            up: up.clone(),
            scale: scale * strength,
        }))
    }

    /// Move the adapters onto `device` (they load where the file was read).
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let mut entries = HashMap::with_capacity(self.entries.len());
        for (k, (d, u, s)) in &self.entries {
            entries.insert(k.clone(), (d.to_device(device)?, u.to_device(device)?, *s));
        }
        Ok(Self { entries })
    }

    /// Names present in the file, for diagnosing a LoRA that matched nothing.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }
}

/// Error helper so callers can report a LoRA that matched no module at all - which is
/// the failure people actually hit (a SD1.5 adapter on an SDXL checkpoint), and which is
/// otherwise silent: the render simply looks unchanged.
pub fn no_match_error(path: &str, file: &LoraFile) -> Error {
    let sample: Vec<&String> = file.keys().take(3).collect();
    Error(format!(
        "lora '{path}': none of its {} modules matched this model. First keys: {sample:?} - \
         this usually means the adapter was trained for a different architecture",
        file.len()
    ))
}

/// The keys an adapter file may use for one Flux projection, in the order they are
/// tried: kohya's flattened `lora_unet_<path>` first, then the XLabs `processor.*`
/// layout.
///
/// It lives here because the scheme was written TWICE - once for the split transformer,
/// once for the facade one - and two hand-copies of a key scheme diverge silently. The
/// symptom is not a crash: it is an adapter that matches on one path and nothing on the
/// other, which reads to a user as "that LoRA does not work with this model".
pub struct FluxKey {
    pub path: String,
    pub alt: String,
}

/// The projections of one DOUBLE block that can carry an adapter: image qkv, image proj,
/// text qkv, text proj. The XLabs layout numbers the streams 1 and 2 in that order.
pub fn double_block_keys(i: usize) -> [FluxKey; 4] {
    let p = format!("double_blocks.{i}");
    [
        FluxKey {
            path: format!("{p}.img_attn.qkv"),
            alt: format!("{p}.processor.qkv_lora1"),
        },
        FluxKey {
            path: format!("{p}.img_attn.proj"),
            alt: format!("{p}.processor.proj_lora1"),
        },
        FluxKey {
            path: format!("{p}.txt_attn.qkv"),
            alt: format!("{p}.processor.qkv_lora2"),
        },
        FluxKey {
            path: format!("{p}.txt_attn.proj"),
            alt: format!("{p}.processor.proj_lora2"),
        },
    ]
}

/// The projections of one SINGLE block that can carry an adapter: linear1 then linear2.
/// These blocks restart their own numbering, as the files key them.
pub fn single_block_keys(i: usize) -> [FluxKey; 2] {
    let p = format!("single_blocks.{i}");
    [
        FluxKey {
            path: format!("{p}.linear1"),
            alt: format!("{p}.processor.qkv_lora"),
        },
        FluxKey {
            path: format!("{p}.linear2"),
            alt: format!("{p}.processor.proj_lora"),
        },
    ]
}

#[cfg(test)]
mod flux_key_tests {
    use super::{double_block_keys, single_block_keys};

    /// The exact strings. An adapter is matched BY NAME, so a change here changes which
    /// files work - it has to be deliberate rather than incidental.
    #[test]
    fn a_double_block_names_both_streams() {
        let k = double_block_keys(7);
        assert_eq!(k[0].path, "double_blocks.7.img_attn.qkv");
        assert_eq!(k[0].alt, "double_blocks.7.processor.qkv_lora1");
        assert_eq!(k[3].path, "double_blocks.7.txt_attn.proj");
        assert_eq!(k[3].alt, "double_blocks.7.processor.proj_lora2");
    }

    #[test]
    fn a_single_block_uses_its_own_numbering() {
        let k = single_block_keys(0);
        assert_eq!(k[0].path, "single_blocks.0.linear1");
        assert_eq!(k[0].alt, "single_blocks.0.processor.qkv_lora");
        assert_eq!(k[1].path, "single_blocks.0.linear2");
        assert_eq!(k[1].alt, "single_blocks.0.processor.proj_lora");
    }

    /// No two projections may share a key: one adapter tensor landing on two projections
    /// is a quietly different model.
    #[test]
    fn every_projection_has_its_own_key() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..19 {
            for k in double_block_keys(i) {
                assert!(seen.insert(k.path.clone()), "duplicate {}", k.path);
                assert!(seen.insert(k.alt.clone()), "duplicate {}", k.alt);
            }
        }
        for i in 0..38 {
            for k in single_block_keys(i) {
                assert!(seen.insert(k.path.clone()), "duplicate {}", k.path);
                assert!(seen.insert(k.alt.clone()), "duplicate {}", k.alt);
            }
        }
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    /// A name that is not a plain file name must be refused BEFORE anything touches the
    /// filesystem, and refused the same way whether or not the target exists - otherwise
    /// the error text alone answers "does this path exist on your server?".
    #[test]
    fn a_path_is_not_a_name() {
        for bad in [
            "../../etc/passwd",
            "/etc/passwd",
            "sub/dir/adapter",
            "..",
            "ok/../../escape",
            "",
            "   ",
        ] {
            let e = resolve(bad).expect_err("must refuse");
            assert!(
                !e.contains("not found"),
                "{bad}: refused with an existence-revealing message: {e}"
            );
        }
    }

    /// With no directory configured the feature is off, not wide open.
    #[test]
    fn an_unconfigured_server_serves_no_adapters() {
        // `lora_dir` is process-global and set once at startup; this asserts the
        // behaviour of the unset branch without racing a test that sets it.
        if lora_dir().is_none() {
            assert!(resolve("anything").is_err());
            assert!(available().is_empty());
        }
    }
}

#[cfg(test)]
mod consistency_tests {
    use super::{is_consistency, CONSISTENCY};

    /// The file declares it. This is the header the adapter shipped here actually
    /// carries, and reading it beats matching a filename anyone can change.
    #[test]
    fn a_file_that_declares_itself_is_believed() {
        let hdr = r#"{"__metadata__":{"modelspec.merged_from":"sdxl_LCM_lora",
                      "modelspec.architecture":"stable-diffusion-xl-v1-base/lora"}}"#;
        assert!(is_consistency(Some(hdr), "whatever-it-was-renamed-to"));
    }

    /// And a file with NO metadata still gets the right answer from its name - the other
    /// adapter here has none at all, so refusing to look at names would leave a
    /// correctly-named adapter running at the base model's recipe.
    #[test]
    fn a_file_with_no_metadata_falls_back_to_its_name() {
        assert!(is_consistency(None, "lcm-lora-sdxl"));
        assert!(is_consistency(Some("{}"), "LCM-Lora-SDXL"));
    }

    /// A STYLE adapter must not be mistaken for one: forcing four steps at guidance one
    /// on a realism LoRA would flatten every render it touches.
    #[test]
    fn a_style_adapter_carries_no_regime() {
        let hdr = r#"{"__metadata__":{}}"#;
        assert!(!is_consistency(Some(hdr), "flux-realism-xlabs"));
        assert!(!is_consistency(None, "flux-realism-xlabs"));
        assert!(!is_consistency(None, "add-detail-xl"));
    }

    /// The regime itself: few steps, almost no guidance. A regression to the base
    /// model's twenty-five at seven is the failure this exists to prevent.
    #[test]
    fn the_regime_is_the_published_one() {
        assert_eq!(CONSISTENCY.steps, 4);
        assert!(
            CONSISTENCY.guidance <= 2.0,
            "a consistency model needs almost no guidance"
        );
    }
}

/// What attaching a set of adapters did.
///
/// `skipped` is not a detail: an adapter trained for a projection this checkpoint fuses, or
/// for a layer count it does not have, matches nothing - and a caller told only "attached"
/// would believe it took effect.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AdapterReport {
    /// Adapter names now live on the model, in the order they were applied.
    pub attached: Vec<String>,
    /// Projections that took a delta.
    pub projections: usize,
    /// Entries in the files that matched no projection.
    pub unmatched: usize,
}

/// The names a PEFT adapter may use for one projection of one transformer layer.
///
/// Three prefixes are in circulation for the same tensor: PEFT wraps the model twice when it
/// saves from a `PeftModel`, once when it saves from the inner model, and a hand-written
/// adapter often carries neither. Trying all three is cheaper than making a user rename their
/// file, and they cannot collide - each is a strict prefix of a full key.
pub fn transformer_keys(layer: usize, projection: &str) -> [String; 3] {
    [
        format!("base_model.model.model.layers.{layer}.{projection}"),
        format!("base_model.model.layers.{layer}.{projection}"),
        format!("model.layers.{layer}.{projection}"),
    ]
}

/// How many entries a file holds, for reporting what matched nothing.
impl LoraFile {
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod transformer_key_tests {
    use super::transformer_keys;

    /// The three prefixes PEFT writes for the same tensor, exactly. A typo here means an
    /// adapter loads, matches nothing, and is reported as attached to zero projections - which
    /// reads to a user as "my fine-tune does nothing".
    #[test]
    fn the_three_prefixes_are_what_peft_writes() {
        let keys = transformer_keys(7, "self_attn.q_proj");
        assert_eq!(keys[0], "base_model.model.model.layers.7.self_attn.q_proj");
        assert_eq!(keys[1], "base_model.model.layers.7.self_attn.q_proj");
        assert_eq!(keys[2], "model.layers.7.self_attn.q_proj");
    }

    /// Two layers never produce the same key, or an adapter for layer 1 would land on layer 11.
    #[test]
    fn no_two_layers_share_a_key() {
        let mut seen = std::collections::HashSet::new();
        for layer in 0..64 {
            for p in ["self_attn.q_proj", "mlp.down_proj"] {
                for key in transformer_keys(layer, p) {
                    assert!(seen.insert(key.clone()), "{key} produced twice");
                }
            }
        }
    }
}
