//! Zero-copy quantized CPU storage: a typed view into block data owned
//! by something else - a parent `QTensor` (per-expert slices) or an mmap'd
//! GGUF. The native `QTensor` storage enum carries owned-or-view block bytes;
//! matmul/dequantize route through the `quant_cpu` dot engine, so a view
//! behaves exactly like an owned tensor without duplicating the block data.

use std::sync::Arc;

use crate::tensor::{self, quantized as nquant};

/// Quant dtypes the view machinery supports (everything the `quant_cpu`
/// engine can serve as a weight).
pub fn native_view_supported(dtype: nquant::GgmlDType) -> bool {
    crate::tensor::quant_cpu::supports(dtype)
}

/// Zero-copy native QTensor viewing tensor `name` directly in the mmap'd GGUF
/// file ( phase C). Returns `Ok(None)` when the tensor is absent or its
/// dtype has no view path - the caller falls back to the copying loader
/// (`Content::tensor`).
pub fn native_gguf_mmap_view(
    content: &nquant::gguf_file::Content,
    mmap: &Arc<memmap2::Mmap>,
    name: &str,
) -> tensor::Result<Option<nquant::QHostTensor>> {
    let Some(info) = content.tensor_infos.get(name) else {
        return Ok(None);
    };
    let dtype = info.ggml_dtype;
    if !native_view_supported(dtype) {
        return Ok(None);
    }
    let elem = info.elem_count();
    if elem % dtype.block_size() != 0 {
        return Ok(None);
    }
    let byte_len = info.size_in_bytes();
    let start = (content.tensor_data_offset + info.offset) as usize;
    if start + byte_len > mmap.len() {
        return Err(tensor::Error(format!(
            "native_gguf_mmap_view: tensor {name} range {start}+{byte_len} exceeds file size {}",
            mmap.len()
        )));
    }
    let base = mmap.as_ptr();
    // SAFETY: the check above rejects `start + byte_len > mmap.len()`, so the range is
    // in bounds of this mapping, and the Arc<Mmap> passed as owner keeps it mapped for
    // as long as the view lives.
    unsafe {
        nquant::QHostTensor::view(
            Arc::new(mmap.clone()) as Arc<dyn std::any::Any + Send + Sync>,
            base,
            start,
            byte_len,
            dtype,
            info.shape.dims().to_vec(),
        )
    }
    .map(Some)
}

/// Zero-copy native view of expert `e` of a `[E, R, C]` expert stack: blocks
/// are expert-major / row-major, so each expert is one contiguous byte range.
/// The returned `[R, C]` QTensor pins the parent via its owner Arc.
pub fn native_expert_view(
    parent: &Arc<nquant::QHostTensor>,
    e: usize,
) -> tensor::Result<nquant::QHostTensor> {
    native_expert_view_pinning(
        parent,
        Arc::new(parent.clone()) as Arc<dyn std::any::Any + Send + Sync>,
        e,
    )
}

// ===========================================================================
// Facade twins: same signatures over the compat types, implemented on the
// native view machinery above.
// ===========================================================================

/// Zero-copy facade QTensor viewing tensor `name` in the mmap'd GGUF file.
/// The view stays host-side (CPU device, no blob upload).
pub fn gguf_mmap_view(
    content: &crate::tensor::quantized::gguf_file::Content,
    mmap: &Arc<memmap2::Mmap>,
    name: &str,
) -> crate::tensor::Result<Option<crate::tensor::quantized::QTensor>> {
    let Some(info) = content.tensor_infos.get(name) else {
        return Ok(None);
    };
    let dtype = info.ggml_dtype;
    if !native_view_supported(dtype) {
        return Ok(None);
    }
    let elem = info.elem_count();
    if elem % dtype.block_size() != 0 {
        return Ok(None);
    }
    let byte_len = info.size_in_bytes();
    let start = (content.tensor_data_offset + info.offset) as usize;
    if start + byte_len > mmap.len() {
        return Err(crate::tensor::Error::msg(format!(
            "gguf_mmap_view: tensor {name} range {start}+{byte_len} exceeds file size {}",
            mmap.len()
        )));
    }
    let base = mmap.as_ptr();
    // SAFETY: same bound as above - `start + byte_len <= mmap.len()` was just checked,
    // and the owner Arc pins the mapping for the view's lifetime.
    let qt = unsafe {
        nquant::QHostTensor::view(
            Arc::new(mmap.clone()) as Arc<dyn std::any::Any + Send + Sync>,
            base,
            start,
            byte_len,
            dtype,
            info.shape.dims().to_vec(),
        )
    }
    .map_err(|e| crate::tensor::Error::msg(e.0))?;
    Ok(Some(crate::tensor::quantized::QTensor::from_native(
        Arc::new(qt),
        &crate::tensor::Device::Cpu,
    )?))
}

/// Zero-copy facade view of expert `e` of a CPU `[E, R, C]` expert stack.
pub fn expert_view(
    parent: &Arc<crate::tensor::quantized::QTensor>,
    e: usize,
) -> crate::tensor::Result<crate::tensor::quantized::QTensor> {
    // Pin the FACADE parent (which owns the native QTensor) in the view.
    let nqt = native_expert_view_pinning(
        parent.native_qtensor(),
        Arc::new(parent.clone()) as Arc<dyn std::any::Any + Send + Sync>,
        e,
    )
    .map_err(|e| crate::tensor::Error::msg(e.0))?;
    crate::tensor::quantized::QTensor::from_native(Arc::new(nqt), &crate::tensor::Device::Cpu)
}

/// `native_expert_view` with an explicit owner to pin.
fn native_expert_view_pinning(
    parent: &Arc<nquant::QHostTensor>,
    owner: Arc<dyn std::any::Any + Send + Sync>,
    e: usize,
) -> tensor::Result<nquant::QHostTensor> {
    let [n_e, r, c] = parent.dims[..] else {
        return Err(tensor::Error(format!(
            "expert_view: parent must be 3-D [E,R,C], got {:?}",
            parent.dims
        )));
    };
    if e >= n_e {
        return Err(tensor::Error(format!(
            "expert_view: expert {e} out of {n_e}"
        )));
    }
    let dtype = parent.dtype;
    let row_bytes = (c / dtype.block_size()) * dtype.type_size();
    let expert_bytes = r * row_bytes;
    let base = parent.data().as_ptr();
    // SAFETY: `e < n_e` was checked, and the parent holds `n_e * expert_bytes` bytes, so
    // `[e * expert_bytes, +expert_bytes)` lies inside its allocation; `owner` pins the
    // parent that owns those bytes.
    unsafe {
        nquant::QHostTensor::view(
            owner,
            base,
            e * expert_bytes,
            expert_bytes,
            dtype,
            vec![r, c],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu;

    /// Per-expert views must dequantize to the exact same f32 bits as the
    /// corresponding rows of the parent stack.
    #[test]
    fn native_expert_view_matches_parent_rows() {
        let e = 3usize;
        let (r, c) = (4usize, 64usize);
        let data: Vec<f32> = (0..e * r * c).map(|i| (i as f32 * 0.37).sin()).collect();

        let bytes = quant_cpu::from_float_bytes(nquant::GgmlDType::Q8_0, &data).unwrap();
        let nqt = Arc::new(
            nquant::QHostTensor::from_bytes(&bytes, nquant::GgmlDType::Q8_0, vec![e, r, c])
                .unwrap(),
        );
        let full = nqt.dequantize_f32().unwrap();

        for i in 0..e {
            let nv = native_expert_view(&nqt, i).unwrap();
            assert_eq!(nv.dims, vec![r, c]);
            let got = nv.dequantize_f32().unwrap();
            let want = &full[i * r * c..(i + 1) * r * c];
            assert_eq!(got.len(), want.len());
            for (j, (g, w)) in got.iter().zip(want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "expert {i} idx {j}: {g} vs {w}");
            }
        }
    }

    /// Native mmap view over a real GGUF blob (skips without a model store):
    /// view bytes and dequant must match the copying loader exactly.
    #[test]
    fn native_gguf_mmap_view_matches_copy() {
        let dir = crate::config::Config::load_test()
            .get_ollama_models_dir()
            .join("blobs");
        let Ok(rd) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut candidates: Vec<_> = rd
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                let mut f = std::fs::File::open(&p).ok()?;
                let mut magic = [0u8; 4];
                std::io::Read::read_exact(&mut f, &mut magic).ok()?;
                (magic == nquant::gguf_file::layout::MAGIC_BYTES)
                    .then_some((e.metadata().ok()?.len(), p))
            })
            .collect();
        candidates.sort();
        let Some((_, path)) = candidates.into_iter().next() else {
            return;
        };

        let mut f = std::fs::File::open(&path).unwrap();
        let content = nquant::gguf_file::Content::read(&mut f).unwrap();
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });

        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        let mut checked = 0usize;
        for name in names {
            if content.tensor_infos[name].elem_count() > 8_000_000 || checked >= 4 {
                continue;
            }
            let Some(view) = native_gguf_mmap_view(&content, &mmap, name).unwrap() else {
                continue;
            };
            let copied = content.host_tensor(&mut f, name).unwrap();
            assert_eq!(view.data(), copied.data(), "{name} view bytes");
            assert_eq!(view.dims, copied.dims, "{name} dims");
            let a = view.dequantize_f32().unwrap();
            let b = copied.dequantize_f32().unwrap();
            assert_eq!(a.len(), b.len(), "{name}");
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{name} idx {i}");
            }
            checked += 1;
        }
        assert!(checked > 0, "no viewable tensor exercised in {path:?}");
    }
}
