//! Reading a tensor's values back as host numbers.

use super::{DType, Result, Tensor};

/// A tensor's elements as one flat f32 row on the host.
///
/// Widening to F32 first is what makes the read dtype-agnostic: the caller never has to know
/// whether the stream is F16, BF16 or F32. Flattening packs any view - a narrow, a transpose  - 
/// before its elements are copied, so what comes back is the tensor's logical order and not
/// its storage order. On a device tensor this is a download, so it belongs to load time and to
/// host decode paths, not inside a device forward.
///
/// Not the same read as [`Tensor::to_vec_f32`], which takes the storage as it stands: that one
/// is rank- and layout-blind, and answers an empty vector when a download fails where this
/// answers an error.
pub fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}
