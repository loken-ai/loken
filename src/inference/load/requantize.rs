//! Requantise a GGUF into a new one, tensor by tensor, carrying its metadata through.
//!
//! Why per-tensor and not a whole-model type: decode is bound by the bytes read per token, and
//! those bytes are not evenly useful. In a dense transformer the feed-forward projections carry
//! most of the weight, so moving them one step down buys most of the saving while attention and
//! the embedding keep the quality a uniform step would have spent everywhere.
//!
//! The blob is named by the sha256 of what was actually written, never by an invented digest: a
//! manifest that points at nothing loads on no engine, and the client answers the miss by pulling
//! gigabytes from the network. The two `deepseek-r1:70b-ffnq3*` tags that shipped a fabricated
//! digest are exactly that failure, and are why this exists in the engine rather than a script.

use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{DType, Device, Error, Result};
use std::path::Path;

/// A per-tensor override: every tensor whose dotted name carries `component` as one of its parts
/// is written as `dtype`. Component "ffn_down" matches "blk.12.ffn_down.weight" and nothing else.
pub struct TensorRule {
    pub component: String,
    pub dtype: GgmlDType,
}

/// A quantisation type from its ggml name (`q3_K`, `q4_0`, `f16`, ...). Case-insensitive on the
/// `k`. `None` for a name no format matches, which the caller turns into a request error rather
/// than a silent fallback.
pub fn parse_dtype(name: &str) -> Option<GgmlDType> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        "q2_k" => GgmlDType::Q2K,
        "iq2_xxs" => GgmlDType::Iq2Xxs,
        "q3_k" => GgmlDType::Q3K,
        "q4_k" => GgmlDType::Q4K,
        "q5_k" => GgmlDType::Q5K,
        "q6_k" => GgmlDType::Q6K,
        "mxfp4" => GgmlDType::MxFp4,
        _ => return None,
    })
}

/// Whether a tensor can take a block-quantised type: two-dimensional, and its row length divides
/// the block. A 1-D norm or bias, or a row that does not divide, keeps its source type - the same
/// rule the checkpoint converter uses, so the two produce the same shape of output.
fn quantisable(dims: &[usize], dtype: GgmlDType) -> bool {
    let block = dtype.block_size();
    dims.len() == 2 && dtype != GgmlDType::F32 && block > 0 && dims[1].is_multiple_of(block)
}

/// The type a tensor is written as: a rule on one of its name components wins; otherwise the base
/// type when one is given and the tensor is quantisable; otherwise the source type, unchanged.
///
/// `base` is `None` for a requant that only touches the ruled tensors and copies everything else
/// verbatim - what recreating a tag like ffnq3 needs, since its source is already a Q4_K_M mix
/// (q6_K on some tensors) that a uniform base would flatten and shrink.
fn target_dtype(
    name: &str,
    dims: &[usize],
    src: GgmlDType,
    base: Option<GgmlDType>,
    rules: &[TensorRule],
) -> GgmlDType {
    for r in rules {
        if name.split('.').any(|c| c == r.component) {
            return if quantisable(dims, r.dtype) {
                r.dtype
            } else {
                src
            };
        }
    }
    match base {
        Some(b) if quantisable(dims, b) => b,
        _ => src,
    }
}

/// A no-op progress sink, for callers that do not report progress (the tests).
pub fn no_progress(_done: usize, _total: usize) {}

/// Which target types have a device quantise kernel. The GPU path is taken only when every
/// requantised tensor lands on one of these; the set grows as kernels are added (q8_0 today).
#[cfg(feature = "cuda")]
fn gpu_target_supported(t: GgmlDType) -> bool {
    matches!(
        t,
        GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q3K | GgmlDType::Q2K
    )
}

/// Read `src_blob`, requantise per `base` and `rules`, write `dst_gguf`. Every tensor is either
/// copied byte for byte (its type is unchanged) or dequantised to f32 and requantised to its
/// target; the model's whole metadata block is carried through unchanged. Tensors are emitted in
/// name order so the same input and rules produce the same bytes, and therefore the same digest.
///
/// `progress(done, total)` is called once per finished tensor, from a worker thread, so a caller
/// can stream progress. The per-tensor dequantise-and-requantise is the whole cost and every
/// tensor is independent, so the work runs across the rayon pool; peak memory is the pool width
/// times the largest tensor's f32 expansion.
pub fn requantize_gguf(
    src_blob: &Path,
    dst_gguf: &Path,
    base: Option<GgmlDType>,
    rules: &[TensorRule],
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<()> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // The sanctioned zero-copy reader: the source is mapped, not parsed into committed heap, and
    // each tensor is read as a view over the mapping. A plain `Content::read` here is the defect
    // the placement invariants forbid - a weight read that copies the whole file.
    let mapped = crate::tensor::quantized::gguf_file::open_mapped(src_blob)?;
    let dev = Device::Cpu;

    let mut names: Vec<String> = mapped.tensor_infos.keys().cloned().collect();
    names.sort();
    let total = names.len();
    let done = AtomicUsize::new(0);

    // Build one entry, quantising through `quantize` (CPU or GPU). A tensor whose type is
    // unchanged is copied byte for byte; otherwise it is dequantised to f32 and requantised.
    let build = |name: &str,
                 quantize: &(dyn Fn(GgmlDType, &[f32]) -> Result<Vec<u8>> + Sync)|
     -> Result<GgufEntry> {
        let info = mapped
            .tensor_infos
            .get(name)
            .ok_or_else(|| Error::msg(format!("requantize: tensor {name} vanished")))?;
        let dims = info.shape.dims().to_vec();
        let src = info.ggml_dtype;
        let target = target_dtype(name, &dims, src, base, rules);
        let q = mapped.tensor(name, &dev)?;
        let (dtype, data) = if target == src {
            (src, q.data()?.into_owned())
        } else {
            // dequantize returns the source's own float dtype - F32 from a quantised tensor, but
            // F16 or BF16 from a half-precision one. Cast to F32 so the quantiser always gets the
            // element type it reads.
            let f32s = q
                .dequantize(&dev)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            (target, quantize(target, &f32s)?)
        };
        Ok(GgufEntry {
            name: name.to_string(),
            dims,
            dtype,
            data,
        })
    };

    let cpu_quantize = |t: GgmlDType, f: &[f32]| crate::tensor::quant_cpu::from_float_bytes(t, f);

    // The CPU path runs the tensors across the rayon pool; par_iter().collect() into a Vec keeps
    // name order, so the bytes and therefore the digest stay stable.
    let cpu_parallel = || -> Result<Vec<GgufEntry>> {
        names
            .par_iter()
            .map(|name| {
                let e = build(name, &cpu_quantize)?;
                progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
                Ok(e)
            })
            .collect()
    };

    // The GPU path runs the tensors one at a time - the GPU is the parallelism, and mixing rayon
    // workers with device launches is what this avoids. It is taken only when a device is present
    // and every requantised tensor's target has a device kernel; anything else falls to the CPU
    // pool. A copied (unchanged) tensor needs no kernel, so it never blocks the choice.
    #[cfg(feature = "cuda")]
    let entries: Vec<GgufEntry> = {
        let all_on_gpu = names.iter().all(|name| {
            let info = &mapped.tensor_infos[name];
            let src = info.ggml_dtype;
            let target = target_dtype(name, info.shape.dims(), src, base, rules);
            target == src || gpu_target_supported(target)
        });
        match crate::tensor::cuda::CudaDevice::new(0) {
            Ok(cdev) if all_on_gpu => {
                let gpu_quantize = move |t: GgmlDType, f: &[f32]| -> Result<Vec<u8>> {
                    match t {
                        GgmlDType::Q8_0 => crate::tensor::cuda::gpu_quantize_q8_0(&cdev, f),
                        GgmlDType::Q4K => crate::tensor::cuda::gpu_quantize_q4_k(&cdev, f),
                        GgmlDType::Q3K => crate::tensor::cuda::gpu_quantize_q3_k(&cdev, f),
                        GgmlDType::Q2K => crate::tensor::cuda::gpu_quantize_q2_k(&cdev, f),
                        // Not reached while all_on_gpu gates entry, but keeps the closure total.
                        other => crate::tensor::quant_cpu::from_float_bytes(other, f),
                    }
                };
                let mut out = Vec::with_capacity(total);
                for (i, name) in names.iter().enumerate() {
                    out.push(build(name, &gpu_quantize)?);
                    progress(i + 1, total);
                }
                out
            }
            _ => cpu_parallel()?,
        }
    };
    #[cfg(not(feature = "cuda"))]
    let entries: Vec<GgufEntry> = cpu_parallel()?;

    // Sorted by key: the source metadata is a hash map whose iteration order is not stable
    // across reads, and an unstable key order would give the same model a different digest each
    // time. Order is nothing to a reader that looks keys up, and everything to a stable digest.
    let mut metadata: Vec<(String, Value)> = mapped
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    metadata.sort_by(|a, b| a.0.cmp(&b.0));
    write_gguf_with_metadata(dst_gguf, &metadata, &entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Content;

    fn f32_bytes(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A 2-D weight moves to the base type, a 1-D norm keeps its type, and the metadata survives.
    /// Run twice, the bytes are identical - the digest a blob is named by is stable for the same
    /// input and rules, which is the property a fabricated digest never had.
    #[test]
    fn requantises_the_weights_keeps_the_norms_and_the_metadata() {
        let dir = std::env::temp_dir();
        let src = dir.join(format!("loken-requant-src-{}.gguf", std::process::id()));
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("qwen3".into()),
            ),
            ("qwen3.block_count".to_string(), Value::U32(1)),
        ];
        // A 2x256 weight (row divides every block) and a 256 norm (1-D, never quantised).
        let weight: Vec<f32> = (0..512).map(|i| (i as f32) * 0.001 - 0.25).collect();
        let norm: Vec<f32> = vec![1.0; 256];
        let entries = vec![
            GgufEntry {
                name: "blk.0.ffn_down.weight".to_string(),
                dims: vec![2, 256],
                dtype: GgmlDType::F32,
                data: f32_bytes(&weight),
            },
            GgufEntry {
                name: "blk.0.attn_norm.weight".to_string(),
                dims: vec![256],
                dtype: GgmlDType::F32,
                data: f32_bytes(&norm),
            },
            // An F16 weight: dequantise returns F16, and the quantiser needs F32. This is the
            // case a real model hit that the all-F32 fixture could not.
            GgufEntry {
                name: "blk.0.ffn_up.weight".to_string(),
                dims: vec![2, 256],
                dtype: GgmlDType::F16,
                data: weight
                    .iter()
                    .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
                    .collect(),
            },
        ];
        write_gguf_with_metadata(&src, &md, &entries).unwrap();

        let out = |n: &str| dir.join(n);
        let a = out(&format!("loken-requant-a-{}.gguf", std::process::id()));
        let b = out(&format!("loken-requant-b-{}.gguf", std::process::id()));
        let rules = vec![TensorRule {
            component: "ffn_down".to_string(),
            dtype: GgmlDType::Q3K,
        }];
        // base q8_0, but the rule sends ffn_down to q3_K.
        requantize_gguf(&src, &a, Some(GgmlDType::Q8_0), &rules, &no_progress).unwrap();
        requantize_gguf(&src, &b, Some(GgmlDType::Q8_0), &rules, &no_progress).unwrap();

        // Reproducible: identical bytes on both runs.
        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "requantise is not reproducible"
        );

        let mut f = std::fs::File::open(&a).unwrap();
        let c = Content::read(&mut f).unwrap();
        for p in [&src, &a, &b] {
            let _ = std::fs::remove_file(p);
        }
        assert_eq!(
            c.tensor_infos["blk.0.ffn_down.weight"].ggml_dtype,
            GgmlDType::Q3K,
            "the ffn weight took the rule's type"
        );
        assert_eq!(
            c.tensor_infos["blk.0.attn_norm.weight"].ggml_dtype,
            GgmlDType::F32,
            "the 1-D norm kept its type"
        );
        assert_eq!(
            c.tensor_infos["blk.0.ffn_up.weight"].ggml_dtype,
            GgmlDType::Q8_0,
            "an F16 weight requantised through the F32 cast"
        );
        assert_eq!(
            c.metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok().cloned())
                .as_deref(),
            Some("qwen3"),
            "metadata carried through"
        );
    }

    /// A requant whose every target is q8_0 or a copy takes the GPU path when a device is present
    /// and the CPU path otherwise; either way it is reproducible and reads back with the right
    /// types. Bytes are not compared to a fixed digest: GPU and CPU are quality-equivalent, not
    /// bit-identical, so only run-to-run stability (whichever path this machine took) is asserted.
    #[test]
    fn q8_0_only_requant_is_reproducible_on_whichever_path() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let src = dir.join(format!("loken-requant-g-src-{pid}.gguf"));
        let md = vec![(
            "general.architecture".to_string(),
            Value::String("qwen3".into()),
        )];
        let weight: Vec<f32> = (0..512).map(|i| (i as f32) * 0.002 - 0.5).collect();
        let norm: Vec<f32> = vec![1.0; 256];
        let entries = vec![
            GgufEntry {
                name: "blk.0.ffn_down.weight".to_string(),
                dims: vec![2, 256],
                dtype: GgmlDType::F32,
                data: f32_bytes(&weight),
            },
            GgufEntry {
                name: "blk.0.attn_norm.weight".to_string(),
                dims: vec![256],
                dtype: GgmlDType::F32,
                data: f32_bytes(&norm),
            },
        ];
        write_gguf_with_metadata(&src, &md, &entries).unwrap();

        let a = dir.join(format!("loken-requant-g-a-{pid}.gguf"));
        let b = dir.join(format!("loken-requant-g-b-{pid}.gguf"));
        // No rules: the 2-D weight goes to q8_0 (a device-kernel target), the 1-D norm is copied.
        requantize_gguf(&src, &a, Some(GgmlDType::Q8_0), &[], &no_progress).unwrap();
        requantize_gguf(&src, &b, Some(GgmlDType::Q8_0), &[], &no_progress).unwrap();

        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "q8_0 requant is not reproducible"
        );
        let mut f = std::fs::File::open(&a).unwrap();
        let c = Content::read(&mut f).unwrap();
        for p in [&src, &a, &b] {
            let _ = std::fs::remove_file(p);
        }
        assert_eq!(
            c.tensor_infos["blk.0.ffn_down.weight"].ggml_dtype,
            GgmlDType::Q8_0,
            "the 2-D weight became q8_0"
        );
        assert_eq!(
            c.tensor_infos["blk.0.attn_norm.weight"].ggml_dtype,
            GgmlDType::F32,
            "the 1-D norm was copied unchanged"
        );
    }
}
