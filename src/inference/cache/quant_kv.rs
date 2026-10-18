//! A layer's quantised KV cache, whichever precision it stores.
//!
//! The lifecycle (length, trim, reset, device position, graph state) is the same for every
//! precision and is answered here once; the attention kernels differ, so a caller that needs
//! one reaches the precision through `q8()` / `q4()`.

use super::q4_kv::{Q4KvCache, Q4KvRaw};
use super::q8_kv::{Q8KvCache, Q8KvRaw};
use crate::tensor::cuda_ext::CudaSlice;
use crate::tensor::{DType, Tensor};
use anyhow::Result;

pub enum QuantKv {
    Q8(Q8KvCache),
    Q4(Q4KvCache),
}

/// A quantised cache's stored bytes for the positions it held, copied aside in the cache's
/// own packed form.
pub enum QuantKvSnapshot {
    Q8(Q8KvRaw),
    Q4(Q4KvRaw),
}

impl QuantKvSnapshot {
    pub fn len(&self) -> usize {
        match self {
            QuantKvSnapshot::Q8(r) => r.len(),
            QuantKvSnapshot::Q4(r) => r.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device memory the copy holds.
    pub fn device_bytes(&self) -> u64 {
        match self {
            QuantKvSnapshot::Q8(r) => r.device_bytes(),
            QuantKvSnapshot::Q4(r) => r.device_bytes(),
        }
    }

    /// The held positions as `[1, n_kv, len, head_dim]` K and V in `dtype`.
    pub fn dequantize_kv(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        match self {
            QuantKvSnapshot::Q8(r) => r.dequantize_kv(dtype),
            QuantKvSnapshot::Q4(r) => r.dequantize_kv(dtype),
        }
    }

    /// Tokens `[from, to)` as host f32 rows, token-major `[n, n_kv, head_dim]`, K then V.
    pub fn host_rows(&self, from: usize, to: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let (k, v) = self.dequantize_kv(DType::F32)?;
        let rows = |t: &Tensor| -> Result<Vec<f32>> {
            Ok(t.narrow(2, from, to - from)?
                .permute((0, 2, 1, 3))?
                .contiguous()?
                .to_device(&crate::tensor::Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        Ok((rows(&k)?, rows(&v)?))
    }
}

impl QuantKv {
    pub fn q8(&self) -> Option<&Q8KvCache> {
        match self {
            QuantKv::Q8(c) => Some(c),
            QuantKv::Q4(_) => None,
        }
    }

    pub fn q8_mut(&mut self) -> Option<&mut Q8KvCache> {
        match self {
            QuantKv::Q8(c) => Some(c),
            QuantKv::Q4(_) => None,
        }
    }

    pub fn q4(&self) -> Option<&Q4KvCache> {
        match self {
            QuantKv::Q4(c) => Some(c),
            QuantKv::Q8(_) => None,
        }
    }

    pub fn q4_mut(&mut self) -> Option<&mut Q4KvCache> {
        match self {
            QuantKv::Q4(c) => Some(c),
            QuantKv::Q8(_) => None,
        }
    }

    pub fn current_seq_len(&self) -> usize {
        match self {
            QuantKv::Q8(c) => c.current_seq_len(),
            QuantKv::Q4(c) => c.current_seq_len(),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        match self {
            QuantKv::Q8(c) => c.append(k, v),
            QuantKv::Q4(c) => c.append(k, v),
        }
    }

    /// Device bytes a raw snapshot of the first `len` positions takes.
    pub fn raw_bytes(&self, len: usize) -> usize {
        match self {
            QuantKv::Q8(c) => c.raw_bytes(len),
            QuantKv::Q4(c) => c.raw_bytes(len),
        }
    }

    /// The stored bytes of the positions held, copied device to device; `None` when empty.
    pub fn snapshot_raw(&self) -> Result<Option<QuantKvSnapshot>> {
        Ok(match self {
            QuantKv::Q8(c) => c.snapshot_raw()?.map(QuantKvSnapshot::Q8),
            QuantKv::Q4(c) => c.snapshot_raw()?.map(QuantKvSnapshot::Q4),
        })
    }

    /// Makes `snap` the whole content of the cache, bit for bit; an error when it was taken
    /// from a cache of another precision.
    pub fn restore_raw(&mut self, snap: &QuantKvSnapshot) -> Result<()> {
        match (self, snap) {
            (QuantKv::Q8(c), QuantKvSnapshot::Q8(r)) => c.restore_raw(r),
            (QuantKv::Q4(c), QuantKvSnapshot::Q4(r)) => c.restore_raw(r),
            _ => Err(anyhow::anyhow!(
                "quantised KV restore: the snapshot's precision differs from the cache's"
            )),
        }
    }

    /// The whole history as `[1, n_kv, len, head_dim]` K and V in `dtype`.
    pub fn dequantize_kv(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        match self {
            QuantKv::Q8(c) => c.dequantize_kv(dtype),
            QuantKv::Q4(c) => c.dequantize_kv(dtype),
        }
    }

    /// Shortens to `new_len`. A trim the precision cannot honour exactly empties the cache,
    /// which is always correct and only loses the reuse.
    pub fn trim_to(&mut self, new_len: usize) {
        match self {
            QuantKv::Q8(c) => c.trim_to(new_len),
            QuantKv::Q4(c) => {
                if let Err(e) = c.trim_to(new_len) {
                    tracing::warn!("Q4 KV trim_to({new_len}) failed: {e}; resetting cache");
                    c.reset();
                }
            }
        }
    }

    pub fn reset(&mut self) {
        match self {
            QuantKv::Q8(c) => c.reset(),
            QuantKv::Q4(c) => c.reset(),
        }
    }

    pub fn update_graph_state(&mut self, pos: usize) -> Result<()> {
        match self {
            QuantKv::Q8(c) => c.update_graph_state(pos),
            QuantKv::Q4(c) => c.update_graph_state(pos),
        }
    }

    pub fn cur_pos_dev_ptr(&self) -> Option<&CudaSlice<i32>> {
        match self {
            QuantKv::Q8(c) => c.cur_pos_dev_ptr(),
            QuantKv::Q4(c) => c.cur_pos_dev_ptr(),
        }
    }

    pub fn mark_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.mark_graph_captured(),
            QuantKv::Q4(c) => c.mark_graph_captured(),
        }
    }

    pub fn clear_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.clear_graph_captured(),
            QuantKv::Q4(c) => c.clear_graph_captured(),
        }
    }

    pub fn resume_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.resume_graph_captured(),
            QuantKv::Q4(c) => c.resume_graph_captured(),
        }
    }

    pub fn suspend_graph_captured(&mut self) {
        match self {
            QuantKv::Q8(c) => c.suspend_graph_captured(),
            QuantKv::Q4(c) => c.suspend_graph_captured(),
        }
    }

    /// Times the buffers were made anew: a graph recorded against them is stale once this moves.
    pub fn buffer_moves(&self) -> u64 {
        match self {
            QuantKv::Q8(c) => c.buffer_moves(),
            QuantKv::Q4(c) => c.buffer_moves(),
        }
    }

    pub fn graph_seq_limit(&self) -> Option<usize> {
        match self {
            QuantKv::Q8(c) => c.graph_seq_limit(),
            QuantKv::Q4(c) => c.graph_seq_limit(),
        }
    }

    pub fn set_seq_len_for_graph(&mut self, len: usize) {
        match self {
            QuantKv::Q8(c) => c.set_seq_len_for_graph(len),
            QuantKv::Q4(c) => c.set_seq_len_for_graph(len),
        }
    }

    pub fn max_seq_padded(&self) -> usize {
        match self {
            QuantKv::Q8(c) => c.max_seq_padded(),
            QuantKv::Q4(c) => c.max_seq_padded(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{QuantKv, QuantKvSnapshot};
    use crate::inference::cache::q4_kv::Q4KvCache;
    use crate::inference::cache::q8_kv::Q8KvCache;
    use crate::tensor::quantized::GgmlDType;
    use crate::tensor::{DType, Device, Tensor};

    const N_KV: usize = 2;
    const HEAD_DIM: usize = 64;

    fn cuda() -> Option<Device> {
        let dev = Device::new_cuda(0).ok()?;
        Tensor::zeros_on((1,), DType::F32, &dev).ok()?;
        Some(dev)
    }

    fn append_random(c: &mut QuantKv, n: usize, dev: &Device) {
        let k = Tensor::randn(0.0f32, 1.0, (1, N_KV, n, HEAD_DIM), dev).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, N_KV, n, HEAD_DIM), dev).unwrap();
        c.append(&k, &v).unwrap();
    }

    /// The history's K and V as host bits, the exact values a read returns.
    fn read_bits(c: &QuantKv) -> (Vec<u32>, Vec<u32>) {
        let (k, v) = c.dequantize_kv(DType::F32).unwrap();
        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };
        (bits(&k), bits(&v))
    }

    /// A raw snapshot restored after more positions were appended reads back bit for bit, in
    /// the cache it came from and in a fresh one.
    fn restores_bit_exact(make: impl Fn(&Device) -> QuantKv, held: usize, more: usize) {
        let Some(dev) = cuda() else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let mut c = make(&dev);
        append_random(&mut c, held, &dev);
        let before = read_bits(&c);
        let snap = c.snapshot_raw().unwrap().expect("a filled cache snapshots");
        assert_eq!(snap.len(), held);
        assert_eq!(snap.device_bytes(), c.raw_bytes(held) as u64);
        append_random(&mut c, more, &dev);
        assert_ne!(read_bits(&c).0.len(), before.0.len());
        c.restore_raw(&snap).unwrap();
        assert_eq!(c.current_seq_len(), held);
        assert!(
            read_bits(&c) == before,
            "restored K/V differ from the original"
        );
        let mut fresh = make(&dev);
        fresh.restore_raw(&snap).unwrap();
        assert!(
            read_bits(&fresh) == before,
            "K/V restored into a fresh cache differ"
        );
        let (k_rows, _) = snap.host_rows(0, held).unwrap();
        assert_eq!(k_rows.len(), held * N_KV * HEAD_DIM);
    }

    #[test]
    fn q8_raw_snapshot_restores_bit_exact() {
        // Past the initial capacity, so the fresh cache grows on restore.
        let held = crate::inference::cache::KV_WORKING_WINDOW_TOKENS + 37;
        restores_bit_exact(
            |d| QuantKv::Q8(Q8KvCache::new(2 * held, N_KV, HEAD_DIM, d).unwrap()),
            held,
            45,
        );
    }

    #[test]
    fn q4_raw_snapshot_restores_bit_exact() {
        let block = GgmlDType::Q4_0.block_size();
        let make = |d: &Device| QuantKv::Q4(Q4KvCache::new(1024, N_KV, HEAD_DIM, d).unwrap());
        // Inside a block, so the residual travels; then on a block boundary.
        restores_bit_exact(make, 3 * block + 5, block + 7);
        restores_bit_exact(make, 2 * block, 9);
    }

    #[test]
    fn q8_raw_snapshot_holds_the_packed_bytes_of_its_length() {
        let Some(dev) = cuda() else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let held = 77;
        let mut c = QuantKv::Q8(Q8KvCache::new(1024, N_KV, HEAD_DIM, &dev).unwrap());
        append_random(&mut c, held, &dev);
        let snap = c.snapshot_raw().unwrap().unwrap();
        let q8 = GgmlDType::Q8_0;
        let per_position = N_KV * (HEAD_DIM / q8.block_size()) * q8.type_size();
        assert_eq!(snap.device_bytes(), (2 * held * per_position) as u64);
        assert!(matches!(snap, QuantKvSnapshot::Q8(_)));
        let empty = QuantKv::Q8(Q8KvCache::new(1024, N_KV, HEAD_DIM, &dev).unwrap());
        assert!(empty.snapshot_raw().unwrap().is_none());
    }
}
