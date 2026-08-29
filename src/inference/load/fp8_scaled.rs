//! fp8-scaled safetensors bridge for the native image engines.
//!
//! The Ray / MutantSparrow checkpoint collection ships fine-tuned drop-in
//! variants of the base architectures this server already runs (Rayflux /
//! Rayflux-Krea over Flux, RayQwest over Qwen-Image, Rayzist over Z-Image).
//! They are distributed as `fp8_scaled` safetensors: the big projection
//! weights are stored as OCP fp8 (e4m3 / e5m2) paired with a companion
//! per-tensor `weight_scale` F32 scalar, and the true weight is
//! `fp8_value * weight_scale`. Small tensors (norm scales, biases, embedders)
//! are bf16 / f32.
//!
//! The native image engines load their transformer through a
//! [`QVarBuilder`](crate::tensor::quantized::QVarBuilder) walk that was
//! wired for quantized GGUF, NOT for safetensors -- so an fp8-scaled Ray
//! checkpoint does not load through the current Flux / Qwen-Image / Z-Image
//! paths. This module is that bridge: it reuses the existing
//! [`SafeTensorsLoader`](crate::tensor::safetensors_io::SafeTensorsLoader)
//! (which already decodes fp8 / bf16 to F32), folds each `weight_scale`, remaps
//! the checkpoint's module names onto the unprefixed names the base-arch
//! builder walk expects, and hands back a `QVarBuilder` the engine consumes
//! unchanged.
//!
//! It is generic across the three families -- the only per-family knob is the
//! `weight_dtype` (Q8_0 for Flux to keep full kernel coverage; F32 for the
//! Qwen-Image DiT, whose activation magnitudes overflow the quantized path).
//!
//! PER-FAMILY INTEGRATION DESIGN
//! ------------------------------
//! How each engine loads its transformer today, and where an fp8-scaled Ray
//! checkpoint slots in:
//!
//! * FLUX (Rayflux / Rayflux-Krea) -- WIRED here.
//!   - Today: `image_engine::load_flux_schnell` downloads a quantized GGUF and
//!     builds `native::quantized::QVarBuilder::from_gguf`, then `flux_native::Flux::
//!     new(cfg, vb)`. The builder walk fetches UNPREFIXED reference-Flux names:
//!     `img_in.weight`/`.bias`, `txt_in.*`, `double_blocks.{i}.{img,txt}_{mod,
//!     attn,mlp,norm}...`, `single_blocks.{i}...`, `final_layer...`. Projection
//!     weights are 2-D `[out,in]` (GGUF block dtypes); norm scales / biases are
//!     small F32.
//!   - fp8 slot-in: when the selected Flux checkpoint is a `.safetensors`, the
//!     engine calls [`load_qvarbuilder`] (this module) with `GgmlDType::Q8_0`
//!     instead of `from_gguf`. Same `Flux::new`, same names -- only the builder
//!     source changes. That branch lives in `load_flux_schnell`'s native
//!     single-device path.
//!
//! * QWEN-IMAGE (RayQwest) -- primitive ready, wiring is a follow-up.
//!   - Today: `image_engine::load_qwen_image` downloads bf16 diffusers shards
//!     (`transformer/diffusion_pytorch_model-*.safetensors`) and builds the DiT
//!     via `native_varbuilder::VarBuilder::from_mmaped_safetensors` (dense
//!     bf16/f32). `native_qwen_image_dit` also has a `load_f32` path that reads
//!     from a `name -> tensor` closure. Names are diffusers-style, UNPREFIXED:
//!     `transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}.weight`, `.norm*`,
//!     `.img_mlp/.txt_mlp...`; weights are stored `[out,in]` and transposed to
//!     `[in,out]` on load.
//!   - fp8 slot-in: feed the DiT's `load_f32` closure from [`folded_tensors`]
//!     (a `HashMap` of the folded F32 map). Use `GgmlDType::F32` (NOT Q8_0):
//!     the Qwen-Image DiT's AdaLN activations reach ~1e2-1e9 and overflow the
//!     quantized MMQ path (see `QKernelMatMul::forward_dequant_f32`).
//!
//! * Z-IMAGE (Rayzist) -- primitive ready, wiring is a follow-up.
//!   - Today: `image_engine` loads Z-Image transformer safetensors via
//!     `native_varbuilder::VarBuilder::from_mmaped_safetensors` /
//!     `HeteroZImage::from_safetensors` (dense bf16). Names are the Z-Image
//!     transformer scheme (`zimage_native` / `zimage_transformer`).
//!   - fp8 slot-in: same as Qwen -- build the transformer from a folded F32
//!     state dict via [`folded_tensors`], or add an fp8-aware safetensors
//!     builder mirroring `from_mmaped_safetensors`.
//!
//! Detection lives in `api::handlers::media`: `is_ray_fp8_model`,
//! `is_qwen_image_model`/`is_z_image_model` (extended for RayQwest/Rayzist), and
//! `image_family` already routes rayflux -> flux (via its `flux` substring).
//!
//! TENSOR-NAME MAPPING (verify once real weights exist): the base-arch builder
//! walks unprefixed names (`img_in.weight`, `double_blocks.0.img_attn.qkv.weight`
//! for Flux; `transformer_blocks.N...` for Qwen-Image / Z-Image). Diffusion
//! safetensors commonly wrap the transformer under `model.diffusion_model.` (or
//! `diffusion_model.` / `model.`); [`canonical_name`] strips those. If a Ray
//! checkpoint uses a DIFFERENT scheme (e.g. diffusers `single_transformer_blocks`
//! naming instead of the reference Flux `single_blocks`), the strip is not
//! enough and a per-family rename table is needed -- dump the header JSON of the
//! real checkpoint (`python -c "import json,sys; \
//! print(list(json.load(open(p,'rb').read ...)))"` on the safetensors header)
//! and diff the key set against what `Flux::new` / the DiT loader fetch.

use crate::tensor::quantized::{GgmlDType, QVarBuilder};
use crate::tensor::safetensors_io::SafeTensorsLoader;
use crate::tensor::{DType, Device, Result, Tensor};

/// Suffix that marks a per-tensor fp8 dequant scale companion. A weight
/// `foo.weight` pairs with `foo.weight_scale` (i.e. the scale name is the
/// weight name with `.weight` -> `.weight_scale`).
const SCALE_SUFFIX: &str = ".weight_scale";

/// Wrapper prefixes a diffusion safetensors checkpoint may put the transformer
/// under. The base-arch builder walks UNPREFIXED module names, so strip the
/// wrapper (longest / most-specific first).
const STRIP_PREFIXES: &[&str] = &["model.diffusion_model.", "diffusion_model.", "model."];

/// Strip a known wrapper prefix from a checkpoint tensor name, yielding the
/// unprefixed module name the base-arch builder walk expects. Names without a
/// known wrapper pass through unchanged.
pub fn canonical_name(raw: &str) -> &str {
    for p in STRIP_PREFIXES {
        if let Some(rest) = raw.strip_prefix(p) {
            return rest;
        }
    }
    raw
}

/// The `weight_scale` companion name for a matmul weight, or `None` if the name
/// is not a `.weight`. `foo.weight` -> `foo.weight_scale`.
fn scale_name_of(weight: &str) -> Option<String> {
    weight
        .strip_suffix(".weight")
        .map(|_| format!("{weight}_scale"))
}

/// Every tensor's shape from a safetensors header (empty on any error). Header-only read
/// (8-byte length prefix + the JSON), so this costs nothing on a 40 GB checkpoint.
fn header_shapes(path: &std::path::Path) -> Vec<Vec<u64>> {
    use std::io::Read;
    (|| -> std::io::Result<Vec<Vec<u64>>> {
        let mut f = std::fs::File::open(path)?;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)?;
        let n = u64::from_le_bytes(len8) as usize;
        if n > 256 * 1024 * 1024 {
            return Ok(Vec::new());
        }
        let mut hdr = vec![0u8; n];
        f.read_exact(&mut hdr)?;
        let v: serde_json::Value = serde_json::from_slice(&hdr)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut out = Vec::new();
        if let Some(o) = v.as_object() {
            for (k, t) in o {
                if k == "__metadata__" {
                    continue;
                }
                if let Some(shape) = t.get("shape").and_then(|s| s.as_array()) {
                    out.push(shape.iter().filter_map(serde_json::Value::as_u64).collect());
                }
            }
        }
        Ok(out)
    })()
    .unwrap_or_default()
}

/// Element count of the largest 2-D tensor in a safetensors header (0 on any error). Header-only
/// read - used to size the dequant scratch (F32 + BF16 copies of one weight) in a runtime-VRAM
/// reserve without loading the checkpoint.
pub fn largest_2d_elems(path: &std::path::Path) -> u64 {
    header_shapes(path)
        .iter()
        .filter(|s| s.len() == 2)
        .map(|s| s.iter().product::<u64>())
        .max()
        .unwrap_or(0)
}

/// Total parameter count of a safetensors checkpoint (0 on any error), header-only.
///
/// This is what a resident-size estimate must be built on, NOT a fraction of the file:
/// the same network ships at 1 byte per weight (fp8) or 2 (bf16), so a ratio calibrated
/// on one container is off by a factor of two on the other - and a doubled estimate
/// splits a model across cards that would have held it whole.
pub fn total_elems(path: &std::path::Path) -> u64 {
    header_shapes(path)
        .iter()
        .map(|s| s.iter().product::<u64>())
        .sum()
}

/// True when the safetensors header contains a tensor whose name includes `needle`.
/// Reads ONLY the JSON header (8-byte length prefix + header bytes), never the weights -
/// cheap enough to probe a 12+ GB checkpoint for arch markers (e.g. `guidance_in` =>
/// dev-family Flux) before deciding a config.
pub fn header_contains(path: &std::path::Path, needle: &str) -> bool {
    use std::io::Read;
    (|| -> std::io::Result<bool> {
        let mut f = std::fs::File::open(path)?;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8)?;
        let n = u64::from_le_bytes(len8) as usize;
        // Safetensors headers are well under this; refuse absurd values instead of allocating.
        if n > 256 * 1024 * 1024 {
            return Ok(false);
        }
        let mut hdr = vec![0u8; n];
        f.read_exact(&mut hdr)?;
        let v: serde_json::Value = serde_json::from_slice(&hdr)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(v.as_object()
            .map(|o| o.keys().any(|k| k.contains(needle)))
            .unwrap_or(false))
    })()
    .unwrap_or(false)
}

/// Load one tensor by its raw (in-file) name as F32, folding its per-tensor
/// `weight_scale` if the checkpoint carries one. This is the reusable seam: the
/// fp8 decode lives in `SafeTensorsLoader`; the scale fold lives here. Matches
/// the fold convention used by the Boogu DiT loader (scalar `weight_scale`,
/// `t * scalar`).
fn load_folded(loader: &SafeTensorsLoader, raw_name: &str) -> Result<Tensor> {
    let mut t = loader.load(raw_name)?.to_dtype(DType::F32)?;
    if let Some(scale_name) = scale_name_of(raw_name) {
        if loader.contains(&scale_name) {
            let s = loader.load(&scale_name)?.to_dtype(DType::F32)?;
            if let Some(&scalar) = s.to_vec_f32().first() {
                t = t.affine(scalar, 0.0)?;
            }
        }
    }
    Ok(t)
}

/// A load count that survives a fan-out to the thread pool.
///
/// A reporter published with `progress::scoped` belongs to the thread that published it, so
/// a rayon worker sees nothing - and the decode below is entirely rayon. Taking a clone of
/// the reporter on the owning thread and calling it from the workers is the explicit
/// hand-off that contract asks for; without it the biggest checkpoints in the fleet are
/// exactly the ones that report nothing while they load.
struct FanOutCount {
    report: Option<crate::inference::serve::progress::SharedProgressFn>,
    done: std::sync::atomic::AtomicUsize,
    total: usize,
}

impl FanOutCount {
    /// Takes the reporter in force on THIS thread; `total` is the work about to be spread.
    fn new(total: usize) -> Self {
        Self {
            report: crate::inference::serve::progress::scoped::current(),
            done: std::sync::atomic::AtomicUsize::new(0),
            total,
        }
    }

    /// One tensor done, from whichever worker finished it.
    fn tick(&self) {
        let Some(f) = self.report.as_ref() else {
            return;
        };
        let done = self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        // Through `note` so the count is clamped by the same rule as every other phase.
        let f = |p: &str, d: usize, t: usize| f(p, d, t);
        crate::inference::serve::progress::note(
            Some(&f),
            crate::inference::serve::progress::phase::LOAD_MODEL,
            done,
            self.total,
        );
    }
}

/// Every transformer tensor of an fp8-scaled checkpoint as
/// `(canonical_name, folded_f32_tensor)` pairs: fp8 / bf16 decoded to F32, each
/// `weight_scale` folded in, names de-prefixed. `weight_scale` companions are
/// consumed (folded) and not emitted. The tensors are host-side F32 here; the
/// device placement / block-quant happens in [`QVarBuilder::from_named_tensors`].
///
/// Kept separate from [`load_qvarbuilder`] so the decode + fold + rename path is
/// unit-testable without building a `QVarBuilder`.
pub fn folded_tensors(loader: &SafeTensorsLoader) -> Result<Vec<(String, Tensor)>> {
    folded_tensors_cancellable(loader, None)
}

/// [`folded_tensors`] with cooperative cancellation: checked once per tensor, so an abandoned
/// load (client disconnected) stops within one tensor's decode instead of chewing through the
/// remaining 10+ GB.
pub fn folded_tensors_cancellable(
    loader: &SafeTensorsLoader,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<Vec<(String, Tensor)>> {
    use rayon::prelude::*;
    let mut names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
    names.sort(); // deterministic order (aids reproducibility / debugging)
                  // Decode fp8/bf16 -> F32 and fold the per-tensor scale in parallel: this is ~10-19 B independent
                  // per-tensor decodes for a diffusion checkpoint, the other half of the load cost besides the
                  // Q8_0 quantize. Reads are from the read-only mmap, so they parallelize safely. Order-independent
                  // (each entry carries its own name), and `names` is pre-sorted for reproducibility.
    let counted = FanOutCount::new(names.iter().filter(|n| !n.ends_with(SCALE_SUFFIX)).count());
    names
        .par_iter()
        .filter(|raw| !raw.ends_with(SCALE_SUFFIX))
        .map(|raw| -> Result<(String, Tensor)> {
            if let Some(c) = cancel {
                c.bail()?;
            }
            let out = (canonical_name(raw).to_string(), load_folded(loader, raw)?);
            counted.tick();
            Ok(out)
        })
        .collect::<Result<Vec<_>>>()
}

/// Load an fp8-scaled safetensors checkpoint into a [`QVarBuilder`] the native
/// image engines consume. `paths` are the checkpoint shard(s) (a single file or
/// a sharded set). `weight_dtype` selects the block dtype for 2-D projection
/// weights (Q8_0 = full kernel coverage; F32 = exact dense).
///
/// # Safety
/// The backing files must not be mutated while mapped (the `SafeTensorsLoader`
/// mmap contract). The mapping is dropped before this returns -- every tensor is
/// copied out during the fold.
pub unsafe fn load_qvarbuilder<P: AsRef<std::path::Path>>(
    paths: &[P],
    weight_dtype: GgmlDType,
    device: &Device,
) -> Result<QVarBuilder> {
    unsafe { load_qvarbuilder_cancellable(paths, weight_dtype, device, None) }
}

/// [`load_qvarbuilder`] with cooperative cancellation (per-tensor during decode+fold; the
/// quantize pass inside `from_named_tensors` follows in one bounded step).
///
/// # Safety
/// Same contract as [`load_qvarbuilder`].
pub unsafe fn load_qvarbuilder_cancellable<P: AsRef<std::path::Path>>(
    paths: &[P],
    weight_dtype: GgmlDType,
    device: &Device,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<QVarBuilder> {
    // Sidecar cache: the fp8 decode + block quantize is a one-time conversion, yet it used to
    // run at full parallelism on EVERY load (minutes of all-core work per model swap). Persist
    // the converted tensors as a GGUF next to the checkpoint on first load; later loads read
    // the quantized bytes straight from disk. Falls back to the in-memory path whenever the
    // sidecar cannot be written (read-only dir, no space) or read back.
    if let Some(sidecar) = sidecar_path(paths, weight_dtype) {
        if sidecar_fresh(&sidecar, paths) {
            match crate::inference::cache::qvb::from_gguf_cached(&sidecar, device) {
                Ok(vb) => {
                    tracing::info!("fp8: loaded converted sidecar {}", sidecar.display());
                    return Ok(vb);
                }
                Err(e) => {
                    tracing::warn!(
                        "fp8: sidecar {} unreadable ({e}); reconverting",
                        sidecar.display()
                    );
                    let _ = std::fs::remove_file(&sidecar);
                }
            }
        }
        let loader = unsafe { SafeTensorsLoader::multi(paths) }?;
        let entries = quantized_entries(&loader, weight_dtype, cancel)?;
        let need: u64 = entries.iter().map(|e| e.data.len() as u64).sum::<u64>() + (1 << 30);
        if fs_available_bytes(&sidecar) < need {
            tracing::warn!(
                "fp8: not enough free space for a {:.1} GB sidecar next to {}; keeping in-memory",
                need as f64 / 1e9,
                sidecar.display()
            );
            return QVarBuilder::from_quantized_entries(entries, device);
        }
        return match crate::tensor::gguf_write::write_gguf(&sidecar, &entries) {
            Ok(()) => {
                tracing::info!(
                    "fp8: wrote sidecar {} ({:.1} GB); later loads skip the conversion",
                    sidecar.display(),
                    entries.iter().map(|e| e.data.len() as f64).sum::<f64>() / 1e9
                );
                // Serve (and host-cache) the bytes we already hold instead of re-reading the
                // file that was just written.
                let vb = QVarBuilder::from_quantized_entries(entries, device)?;
                crate::inference::cache::qvb::insert_for_path(&sidecar, &vb);
                Ok(vb)
            }
            Err(e) => {
                tracing::warn!("fp8: sidecar write failed ({e}); continuing in-memory");
                QVarBuilder::from_quantized_entries(entries, device)
            }
        };
    }
    let loader = unsafe { SafeTensorsLoader::multi(paths) }?;
    let tensors = folded_tensors_cancellable(&loader, cancel)?;
    if let Some(c) = cancel {
        c.bail()?;
    }
    QVarBuilder::from_named_tensors(tensors, weight_dtype, device)
}

/// Free bytes on the filesystem holding `path`'s parent (0 when the query fails, which
/// disables the sidecar write rather than risking filling the model disk).
fn fs_available_bytes(path: &std::path::Path) -> u64 {
    let Some(dir) = path.parent() else { return 0 };
    let Ok(cstr) = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()) else {
        return 0;
    };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: cstr is a valid NUL-terminated path and st is a properly sized out-param.
    if unsafe { libc::statvfs(cstr.as_ptr(), &mut st) } != 0 {
        return 0;
    }
    (st.f_bavail as u64).saturating_mul(st.f_frsize as u64)
}

/// `<stem>.<dtype>.gguf` next to the checkpoint - single-shard checkpoints only (the sharded
/// case would need a merged name; none of the fp8 collections ship sharded today).
/// Path of the converted-GGUF sidecar for an fp8 checkpoint, when one applies.
/// Public so loaders that only speak GGUF (the hetero CPU-spill path) can be
/// pointed at the conversion instead of failing on the safetensors magic.
pub fn sidecar_for<P: AsRef<std::path::Path>>(
    path: P,
    weight_dtype: GgmlDType,
) -> Option<std::path::PathBuf> {
    sidecar_path(&[path], weight_dtype)
}

/// Ensure the sidecar GGUF exists for `path` (converting once if needed) and
/// return it. The conversion is the same one `load_qvarbuilder_cancellable`
/// performs and caches, so this costs nothing when a load already happened.
pub fn ensure_sidecar<P: AsRef<std::path::Path>>(
    path: P,
    weight_dtype: GgmlDType,
) -> Result<std::path::PathBuf> {
    let p = path.as_ref();
    let sidecar = sidecar_for(p, weight_dtype)
        .ok_or_else(|| crate::tensor::Error::msg("fp8: no sidecar path for this checkpoint"))?;
    if sidecar_fresh(&sidecar, &[p]) {
        return Ok(sidecar);
    }
    // Converting on the CPU writes the sidecar as a side effect; the returned
    // builder is dropped (the caller wants the FILE, per-device builders come next).
    let _ = unsafe { load_qvarbuilder_cancellable(&[p], weight_dtype, &Device::Cpu, None) }?;
    if sidecar_fresh(&sidecar, &[p]) {
        Ok(sidecar)
    } else {
        Err(crate::tensor::Error::msg(
            "fp8: sidecar conversion produced no usable file",
        ))
    }
}

fn sidecar_path<P: AsRef<std::path::Path>>(
    paths: &[P],
    weight_dtype: GgmlDType,
) -> Option<std::path::PathBuf> {
    if paths.len() != 1 {
        return None;
    }
    let p = paths[0].as_ref();
    let stem = p.file_stem()?.to_str()?;
    let tag = format!("{weight_dtype:?}").to_ascii_lowercase();
    Some(p.with_file_name(format!("{stem}.{tag}.gguf")))
}

/// A sidecar serves only if it is non-empty and no older than every source shard (a
/// re-downloaded checkpoint invalidates the conversion).
fn sidecar_fresh<P: AsRef<std::path::Path>>(sidecar: &std::path::Path, paths: &[P]) -> bool {
    let Ok(meta) = std::fs::metadata(sidecar) else {
        return false;
    };
    if meta.len() == 0 {
        return false;
    }
    let Ok(side_m) = meta.modified() else {
        return false;
    };
    paths.iter().all(|p| {
        std::fs::metadata(p.as_ref())
            .and_then(|m| m.modified())
            .is_ok_and(|src_m| src_m <= side_m)
    })
}

/// Decode + fold + block-quantize every tensor into GGUF-ready entries, streaming per tensor
/// (the F32 intermediate is dropped as soon as its quantized bytes exist - the peak host
/// footprint is the quantized model plus a few in-flight F32 tensors, not the full F32 model).
/// The quantization rule mirrors `QVarBuilder::from_named_tensors`: 2-D projection weights go
/// to `weight_dtype` when the input dim tiles cleanly, everything else stays dense F32.
fn quantized_entries(
    loader: &SafeTensorsLoader,
    weight_dtype: GgmlDType,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<Vec<crate::tensor::gguf_write::GgufEntry>> {
    use rayon::prelude::*;
    let mut names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
    names.sort();
    let counted = FanOutCount::new(names.iter().filter(|n| !n.ends_with(SCALE_SUFFIX)).count());
    names
        .par_iter()
        .filter(|raw| !raw.ends_with(SCALE_SUFFIX))
        .map(|raw| -> Result<crate::tensor::gguf_write::GgufEntry> {
            if let Some(c) = cancel {
                c.bail()?;
            }
            let t = load_folded(loader, raw)?;
            let dims = t.dims().to_vec();
            let f32s = t.flatten_all()?.to_vec1::<f32>()?;
            let block = weight_dtype.block_size();
            let use_weight_dtype = dims.len() == 2
                && weight_dtype != GgmlDType::F32
                && block > 0
                && dims[1] % block == 0;
            let dtype = if use_weight_dtype {
                weight_dtype
            } else {
                GgmlDType::F32
            };
            let data = crate::tensor::quant_cpu::from_float_bytes(dtype, &f32s)?;
            counted.tick();
            Ok(crate::tensor::gguf_write::GgufEntry {
                name: canonical_name(raw).to_string(),
                dims,
                dtype,
                data,
            })
        })
        .collect::<Result<Vec<_>>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};

    #[test]
    fn canonical_name_strips_wrappers() {
        assert_eq!(
            canonical_name("model.diffusion_model.img_in.weight"),
            "img_in.weight"
        );
        assert_eq!(
            canonical_name("diffusion_model.double_blocks.0.img_mod.lin.weight"),
            "double_blocks.0.img_mod.lin.weight"
        );
        assert_eq!(
            canonical_name("model.final_layer.linear.weight"),
            "final_layer.linear.weight"
        );
        // already unprefixed -> unchanged
        assert_eq!(canonical_name("txt_in.weight"), "txt_in.weight");
    }

    #[test]
    fn scale_name_convention() {
        assert_eq!(
            scale_name_of("img_in.weight").as_deref(),
            Some("img_in.weight_scale")
        );
        assert_eq!(scale_name_of("img_in.bias"), None);
    }

    /// End-to-end proof on a synthesized fp8-scaled fixture: fp8 e4m3 decode +
    /// `weight_scale` fold + wrapper-prefix rename, using hand-picked fp8 byte
    /// patterns with known exact values (so the assertion is independent of the
    /// loader's own decode).
    #[test]
    fn fp8_decode_scale_fold_and_rename() {
        // fp8 e4m3 (bias 7): byte = S EEEE MMM.
        //   0x38 = 0 0111 000 -> (1+0)*2^(7-7) =  1.0
        //   0x40 = 0 1000 000 -> (1+0)*2^(8-7) =  2.0
        //   0x30 = 0 0110 000 -> (1+0)*2^(6-7) =  0.5
        //   0xB8 = 1 0111 000 -> -1.0
        let weight_bytes: [u8; 4] = [0x38, 0x40, 0x30, 0xB8]; // [2,2] row-major
        let scale: f32 = 2.0;
        let scale_bytes = scale.to_le_bytes();
        // A bias tensor left as bf16 -> should decode to F32, no scale.
        let bias_vals: [half::bf16; 2] = [half::bf16::from_f32(0.25), half::bf16::from_f32(-4.0)];
        let mut bias_bytes = Vec::with_capacity(4);
        for v in bias_vals {
            bias_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let tensors = vec![
            (
                "model.diffusion_model.img_in.weight".to_string(),
                TensorView::new(Dtype::F8_E4M3, vec![2, 2], &weight_bytes).unwrap(),
            ),
            (
                "model.diffusion_model.img_in.weight_scale".to_string(),
                TensorView::new(Dtype::F32, vec![], &scale_bytes).unwrap(),
            ),
            (
                "model.diffusion_model.img_in.bias".to_string(),
                TensorView::new(Dtype::BF16, vec![2], &bias_bytes).unwrap(),
            ),
        ];

        let dir = std::env::temp_dir().join(format!("fp8_bridge_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ray_flux_fixture.safetensors");
        safetensors::serialize_to_file(tensors, None, &path).unwrap();

        let loader = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();

        // 1. weight: fp8 decoded * scale, renamed.
        let folded = folded_tensors(&loader).unwrap();
        let by_name: std::collections::HashMap<_, _> = folded.into_iter().collect();
        assert!(
            by_name.contains_key("img_in.weight"),
            "wrapper prefix not stripped"
        );
        assert!(by_name.contains_key("img_in.bias"));
        // weight_scale companion must NOT be emitted as its own tensor.
        assert!(!by_name.keys().any(|k| k.ends_with(SCALE_SUFFIX)));

        let w = by_name["img_in.weight"].to_vec_f32();
        // decoded [1,2,0.5,-1] * scale 2 = [2,4,1,-2]
        assert_eq!(
            w,
            vec![2.0, 4.0, 1.0, -2.0],
            "fp8 decode * weight_scale fold"
        );

        let b = by_name["img_in.bias"].to_vec_f32();
        assert_eq!(b, vec![0.25, -4.0], "bf16 bias decode (no scale)");

        // 2. Bridge into a QVarBuilder and re-fetch through the engine-facing API.
        let vb = QVarBuilder::from_named_tensors(
            by_name.clone().into_iter().collect::<Vec<_>>(),
            GgmlDType::Q8_0,
            &Device::Cpu,
        )
        .unwrap();
        // bias is 1-D -> served dense F32, exact.
        let bias_t = vb.get_f32(2, "img_in.bias").unwrap();
        assert_eq!(bias_t.to_vec_f32(), vec![0.25, -4.0]);
        // weight is 2-D with in-dim 2 (not a multiple of the Q8_0 block of 32)
        // -> kept dense F32 -> the qmatmul must still build and be exact.
        let mm = vb.qmatmul(2, 2, "img_in.weight").unwrap();
        // x = identity rows -> output rows are the weight rows (w @ x^T semantics).
        let x = Tensor::from_vec_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let y = mm.forward(&x).unwrap().to_vec_f32();
        // w = [[2,4],[1,-2]]; x row0=[1,0] -> [2,1], x row1=[0,1] -> [4,-2].
        assert_eq!(
            y,
            vec![2.0, 1.0, 4.0, -2.0],
            "dense F32 qmatmul over folded weight"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The sidecar round-trip must be BIT-IDENTICAL to the in-memory path: first load writes
    /// the converted GGUF, second load reads it back, and every tensor fetched through the
    /// engine-facing API matches exactly. A quantized weight (in-dim = one Q8_0 block) proves
    /// the block bytes survive the disk round-trip, not just the dense F32 ones.
    #[test]
    fn sidecar_roundtrip_bit_identical() {
        let dim = 32; // one Q8_0 block per row -> the weight actually quantizes
        let vals: Vec<f32> = (0..dim * dim).map(|i| ((i % 61) as f32) - 30.0).collect();
        let mut w_bytes = Vec::with_capacity(vals.len() * 2);
        for v in &vals {
            w_bytes.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
        }
        let bias: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        let mut b_bytes = Vec::with_capacity(bias.len() * 2);
        for v in &bias {
            b_bytes.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
        }
        let tensors = vec![
            (
                "model.diffusion_model.blk.weight".to_string(),
                TensorView::new(Dtype::BF16, vec![dim, dim], &w_bytes).unwrap(),
            ),
            (
                "model.diffusion_model.blk.bias".to_string(),
                TensorView::new(Dtype::BF16, vec![dim], &b_bytes).unwrap(),
            ),
        ];
        let dir = std::env::temp_dir().join(format!("fp8_sidecar_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.safetensors");
        safetensors::serialize_to_file(tensors, None, &path).unwrap();

        // Reference: the pure in-memory bridge.
        let loader = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();
        let folded = folded_tensors(&loader).unwrap();
        let ref_vb =
            QVarBuilder::from_named_tensors(folded, GgmlDType::Q8_0, &Device::Cpu).unwrap();
        drop(loader);

        // First cached load: converts + writes the sidecar.
        let side = sidecar_path(&[&path], GgmlDType::Q8_0).unwrap();
        assert!(!side.exists());
        let vb1 =
            unsafe { load_qvarbuilder_cancellable(&[&path], GgmlDType::Q8_0, &Device::Cpu, None) }
                .unwrap();
        assert!(side.exists(), "first load must write the sidecar");
        // Second cached load: reads the sidecar back.
        let vb2 =
            unsafe { load_qvarbuilder_cancellable(&[&path], GgmlDType::Q8_0, &Device::Cpu, None) }
                .unwrap();

        let x = Tensor::from_vec_f32(
            (0..dim).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            vec![1, dim],
        )
        .unwrap();
        let y_ref = ref_vb
            .qmatmul(dim, dim, "blk.weight")
            .unwrap()
            .forward(&x)
            .unwrap()
            .to_vec_f32();
        for (tag, vb) in [("written", &vb1), ("read-back", &vb2)] {
            let y = vb
                .qmatmul(dim, dim, "blk.weight")
                .unwrap()
                .forward(&x)
                .unwrap()
                .to_vec_f32();
            assert_eq!(
                y, y_ref,
                "{tag} sidecar qmatmul must be bit-identical to in-memory"
            );
            let b = vb.get_f32(dim, "blk.bias").unwrap().to_vec_f32();
            assert_eq!(b, bias, "{tag} sidecar bias must be exact");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
