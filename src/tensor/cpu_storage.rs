//! CPU storage for the native tensor: one contiguous typed buffer.

use super::DType;
use half::{bf16, f16};

/// Declare the buffers a CPU tensor can hold, and derive what is asked of one.
///
/// A row is `Variant(element) => dtype`. The enum, the dtype and the length were three lists
/// over the same nine variants; a tenth carrier would have had to be added to all three.
macro_rules! cpu_storages {
    ($($variant:ident($elem:ty) => $dtype:ident;)+) => {
        #[derive(Debug, Clone)]
        pub enum CpuStorage {
            $($variant(Vec<$elem>),)+
        }

        impl CpuStorage {
            pub fn dtype(&self) -> DType {
                match self {
                    $(Self::$variant(_) => DType::$dtype,)+
                }
            }

            pub fn len(&self) -> usize {
                match self {
                    $(Self::$variant(v) => v.len(),)+
                }
            }
        }
    };
}

cpu_storages! {
    U8(u8) => U8;
    U32(u32) => U32;
    I16(i16) => I16;
    I32(i32) => I32;
    I64(i64) => I64;
    BF16(bf16) => BF16;
    F16(f16) => F16;
    F32(f32) => F32;
    F64(f64) => F64;
}

impl CpuStorage {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// View as f32, converting if needed (reduction-friendly accessor used by
    /// the early CPU op implementations; typed fast paths come with the ops).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        // Widen half-precision weights in parallel above a size gate - this pass runs over
        // billions of elements for big encoders (umT5-XXL bf16 ≈ 5.5 B) and was a load-time
        // bottleneck. Small tensors stay sequential to dodge fork overhead; the
        // indexed collect preserves order, so every path is bit-identical to the scalar map.
        // Half->f32 uses the matmul spin-pool (not rayon): on F16 models this cast runs
        // per elementwise op, and rayon's idle workers steal-spin through the matmul pool.
        const PAR: usize = 1 << 16;
        const CH: usize = 1 << 16;
        match self {
            Self::U8(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I16(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I64(v) => v.iter().map(|&x| x as f32).collect(),
            Self::BF16(v) => widen(v, PAR, CH, bf16::to_f32),
            Self::F16(v) => widen(v, PAR, CH, f16::to_f32),
            Self::F32(v) => v.clone(),
            Self::F64(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }
}

/// Widen a half-precision buffer to f32.
///
/// Parallel above `par` elements and sequential below it: this pass runs over billions of
/// elements for a large encoder and was a load-time bottleneck, while the fork costs more than
/// it saves on a small tensor. The indexed write preserves order, so both paths are
/// bit-identical to the scalar map.
///
/// On the matmul spin-pool rather than rayon: on half-precision models this cast runs once per
/// elementwise op, and rayon's idle workers steal-spin through the matmul pool.
fn widen<T: Copy + Send + Sync>(v: &[T], par: usize, chunk: usize, to_f32: fn(T) -> f32) -> Vec<f32> {
    if v.len() < par {
        return v.iter().map(|x| to_f32(*x)).collect();
    }
    let mut out = vec![0f32; v.len()];
    crate::tensor::quant_cpu::pool_par_chunks_mut(&mut out, chunk, &|c, d| {
        for (j, o) in d.iter_mut().enumerate() {
            *o = to_f32(v[c * chunk + j]);
        }
    });
    out
}
