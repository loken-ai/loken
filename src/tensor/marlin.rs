//! The host side of the W4A16 tensor-core GEMM: the load-time repack, and the layer that
//! calls the kernel.
//!
//! ## What arrives
//!
//! An AutoAWQ GEMM linear, as the safetensors carry it:
//!   - `qweight`: `[K, N/8]` i32 - eight output channels per int, channel `n` at nibble
//!     `AWQ_ORDER[n]`;
//!   - `qzeros` : `[K/gs, N/8]` i32, the same nibble order;
//!   - `scales` : `[K/gs, N]` f16.
//!
//! The weight is `(q - z) * s`: an unsigned 4-bit value, an unsigned 4-bit zero point, and one
//! f16 scale per group of `gs` inputs. Only `gs = 128` is compiled.
//!
//! ## What leaves
//!
//! The same numbers, in the order the kernel's fragments want to be read in - the repack is a
//! pure reorder, so nothing is lost and nothing is computed:
//!   - `b_q`   : `[K/16, N*16/8]` u32 - 16x16 tiles, then the fragment permutation, eight
//!     nibbles per u32;
//!   - `scales`: `[K/gs, N]` f16, permuted 64 lanes at a time;
//!   - `zp`    : `[K/gs, N/8]` u32, the same 64-lane permutation composed with the dequantiser's
//!     own interleave, packed like the weights.
//!
//! Why those orders are what they are is `cuda/marlin/DESIGN.md`; this module produces them.
//!
//! ## What the kernel is for
//!
//! One to thirty-two rows of activation. At a single row it wins on the memory pipeline rather
//! than the arithmetic - the weights stage through shared memory with no dequantisation in the
//! load path - and from two rows up the tensor cores keep 4-bit throughput where a mat-vec
//! kernel has none left to give.

use half::f16;

use super::{Error, Result};

/// Marlin tile edge (16x16 weight tiles).
pub const MARLIN_TILE: usize = 16;
/// The only group size with compiled kernel instantiations (group_blocks=8).
pub const MARLIN_GROUP_SIZE: usize = 128;
/// N must divide by this (smallest thread_n tile).
pub const MIN_THREAD_N: usize = 64;
/// Largest thread_n tile (sizes the fp32-reduce scratch).
pub const MAX_THREAD_N: usize = 256;
/// Per-launch-chunk M ceiling compiled (thread_m_blocks <= 2); larger M loops
/// in chunks of 32 inside the launcher.
pub const MAX_M_PER_CHUNK: usize = 32;

/// AWQ nibble order: output channel `n` (within a packed i32) lives at nibble
/// `AWQ_ORDER[n]`. Mirrors `inference::load::awq::AWQ_ORDER` (kept local so the
/// tensor substrate does not depend on the inference layer).
const AWQ_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Marlin 4-bit dequant interleave (pairs of half2 in the fragment).
const DEQUANT_INTERLEAVE_U4: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// An AWQ linear repacked into the Marlin kernel layout (host copy).
pub struct MarlinAwqRepacked {
    /// Input features.
    pub k: usize,
    /// Output features.
    pub n: usize,
    /// Quantization group size (128).
    pub group_size: usize,
    /// `[K/16 * N*2]` u32 - tile-permuted packed weights.
    pub b_q: Vec<u32>,
    /// `[K/gs * N]` f16 - lane-permuted scales.
    pub scales: Vec<f16>,
    /// `[K/gs * N/8]` u32 - permuted + interleaved packed zero-points.
    pub zp: Vec<u32>,
}

/// Where each of the 1024 weights of a 16x64 tile group comes from.
///
/// Two orders composed: the one the tensor-core fragments are read in, and the one the
/// dequantiser emits its four values in. Composing them here means the kernel does neither.
fn weight_perm_u4() -> Vec<usize> {
    let mut perm: Vec<usize> = Vec::with_capacity(1024);
    for i in 0..32usize {
        let col = i / 4;
        let mut perm1 = [0usize; 8];
        let mut idx = 0;
        for block in [0usize, 1] {
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1[idx] = 16 * row + col + 8 * block;
                idx += 1;
            }
        }
        for j in 0..4usize {
            for &p in &perm1 {
                perm.push(p + 256 * j);
            }
        }
    }
    // Interleave for the in-kernel u4 dequant: groups of 8, order [0,2,4,6,1,3,5,7].
    let mut out = vec![0usize; 1024];
    for g in 0..128usize {
        for j in 0..8usize {
            out[g * 8 + j] = perm[g * 8 + DEQUANT_INTERLEAVE_U4[j]];
        }
    }
    out
}

/// Where each of 64 consecutive scales comes from: the eight-by-eight transpose that puts a
/// warp's quarter of the columns where its lanes will look for them.
fn scale_perm_grouped() -> [usize; 64] {
    let mut p = [0usize; 64];
    for i in 0..8usize {
        for j in 0..8usize {
            p[i * 8 + j] = i + 8 * j;
        }
    }
    p
}

/// The same 64 lanes as the scales, composed with the dequantiser's interleave.
///
/// A zero point goes through the same unpacking as a weight, so it needs the weight's shuffle
/// as well as the scale's - one permutation rather than two passes.
fn zero_point_perm() -> [usize; 64] {
    let scale = scale_perm_grouped();
    let mut p = [0usize; 64];
    for c in 0..8usize {
        for j in 0..8usize {
            p[c * 8 + j] = scale[c * 8 + DEQUANT_INTERLEAVE_U4[j]];
        }
    }
    p
}

/// Undo the AWQ nibble order: eight output channels per int, channel `n` at nibble
/// `AWQ_ORDER[n]`, becoming one channel per byte in channel order.
fn unpack_awq(packed: &[i32], out: &mut [u8]) {
    for (i, &word) in packed.iter().enumerate() {
        let w = word as u32;
        for j in 0..8 {
            out[i * 8 + j] = ((w >> (4 * AWQ_ORDER[j])) & 0xF) as u8;
        }
    }
}

/// Eight nibbles to an int, value `i` at bits `4i` - the order the kernel's dequantiser reads.
fn pack_nibbles(src: &[u8]) -> Vec<u32> {
    src.chunks_exact(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |v, (i, &q)| v | (q as u32) << (4 * i))
        })
        .collect()
}

/// Apply `perm` inside every chunk of `perm.len()` values.
fn permute_chunks<T: Copy + Default>(src: &[T], perm: &[usize]) -> Vec<T> {
    let mut out = vec![T::default(); src.len()];
    for (chunk, out) in out.chunks_exact_mut(perm.len()).enumerate() {
        let base = chunk * perm.len();
        for (j, o) in out.iter_mut().enumerate() {
            *o = src[base + perm[j]];
        }
    }
    out
}

/// Reorder one AWQ linear into the layout the kernel reads. Lossless: every output is an
/// input, moved.
pub fn repack_awq_to_marlin(
    qweight: &[i32],
    qzeros: &[i32],
    scales: &[f16],
    k: usize,
    n: usize,
    group_size: usize,
) -> Result<MarlinAwqRepacked> {
    if group_size != MARLIN_GROUP_SIZE {
        return Err(Error(format!(
            "marlin repack: group_size {group_size} unsupported (only {MARLIN_GROUP_SIZE} compiled)"
        )));
    }
    if !k.is_multiple_of(MARLIN_TILE) || !k.is_multiple_of(group_size) {
        return Err(Error(format!(
            "marlin repack: K={k} not a multiple of 16/group"
        )));
    }
    if !n.is_multiple_of(MIN_THREAD_N) {
        return Err(Error(format!(
            "marlin repack: N={n} not a multiple of {MIN_THREAD_N}"
        )));
    }
    let pcols = n / 8;
    let groups = k / group_size;
    if qweight.len() != k * pcols || qzeros.len() != groups * pcols || scales.len() != groups * n {
        return Err(Error("marlin repack: input tensor shape mismatch".into()));
    }

    use rayon::prelude::*;

    // The weights, in three moves. A `[K,N]` array of nibbles becomes `[K/16, N/16, 16, 16]`
    // tiles, each tile group of 1024 takes the fragment permutation, and the result packs eight
    // to an int. The rows are disjoint, and this runs once per linear at model load, where
    // serially it is what the load time is spent on.
    let mut plain = vec![0u8; k * n];
    plain
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, out)| unpack_awq(&qweight[row * pcols..(row + 1) * pcols], out));

    let mut tiled = vec![0u8; k * n];
    tiled
        .par_chunks_mut(n * MARLIN_TILE)
        .enumerate()
        .for_each(|(kt, out)| {
            for nt in 0..n / MARLIN_TILE {
                for r in 0..MARLIN_TILE {
                    for c in 0..MARLIN_TILE {
                        out[nt * MARLIN_TILE * MARLIN_TILE + r * MARLIN_TILE + c] =
                            plain[(kt * MARLIN_TILE + r) * n + nt * MARLIN_TILE + c];
                    }
                }
            }
        });

    // A tile row is `N*16` wide and N divides by 64, so it is a whole number of tile groups.
    let b_q = pack_nibbles(&permute_chunks(&tiled, &weight_perm_u4()));

    // The scales move as they are; the zero points take the weights' shuffle as well, and are
    // packed like the weights.
    let s_out = permute_chunks(scales, &scale_perm_grouped());

    let mut zq = vec![0u8; groups * n];
    unpack_awq(qzeros, &mut zq);
    let zp = pack_nibbles(&permute_chunks(&zq, &zero_point_perm()));

    Ok(MarlinAwqRepacked {
        k,
        n,
        group_size,
        b_q,
        scales: s_out,
        zp,
    })
}

impl MarlinAwqRepacked {
    /// Device bytes the uploaded layer will occupy (weights + metadata; the
    /// fp32-reduce scratch is shared per device, not counted here).
    pub fn device_bytes(&self) -> usize {
        self.b_q.len() * 4 + self.scales.len() * 2 + self.zp.len() * 4
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use cudarc::driver::{CudaSlice, CudaView, DevicePtr};
    use std::sync::Arc;

    extern "C" {
        /// cuda/marlin/marlin_launcher.cu - returns 0 on success.
        fn loken_marlin_awq_f16_gemm(
            a: *const std::ffi::c_void,
            b: *const std::ffi::c_void,
            scales: *const std::ffi::c_void,
            zp: *const std::ffi::c_void,
            c: *mut std::ffi::c_void,
            c_tmp: *mut f32,
            locks: *mut i32,
            prob_m: i32,
            prob_n: i32,
            prob_k: i32,
            num_groups: i32,
            sms: i32,
            max_shared_mem: i32,
            stream: *mut std::ffi::c_void,
        ) -> i32;
    }

    /// Kernel scratch shared by every Marlin layer on one stream: the fp32 partial-tile
    /// reduce buffer and the lock array.
    ///
    /// Shared because a stream serialises its own launches and the kernel restores the locks
    /// to zero after use; one per layer would cost several megabytes times some hundreds of
    /// linears.
    pub struct MarlinScratch {
        /// fp32 partial-tile reduce scratch, `sms * 64 * MAX_THREAD_N` floats.
        c_tmp: CudaSlice<f32>,
        /// Lock array (zero-initialized; the kernel restores zeros after use).
        locks: CudaSlice<i32>,
    }

    /// The scratch every layer on one STREAM shares. `Weak` so unloading a model - dropping
    /// its layers - gives the device memory back.
    ///
    /// Per stream and not per card: the locks and the partial-sum buffer are ordered by
    /// nothing except the stream the kernel runs on, so two streams sharing them would have
    /// one launch reading the other's partial tiles. A card runs more than one stream as soon
    /// as anything else on it builds its own context, which the image and audio engines do.
    fn scratch_for(
        dev: &Arc<super::super::cuda::CudaDevice>,
        nsm: usize,
    ) -> Result<Arc<MarlinScratch>> {
        use std::sync::{Mutex, OnceLock, Weak};
        static REG: OnceLock<Mutex<Vec<(usize, Weak<MarlinScratch>)>>> = OnceLock::new();
        let reg = REG.get_or_init(|| Mutex::new(Vec::new()));
        let mut reg = reg
            .lock()
            .map_err(|_| Error("marlin scratch lock poisoned".into()))?;
        let stream = dev.stream();
        let key = stream.cu_stream() as usize;
        if let Some((_, w)) = reg.iter().find(|(k, _)| *k == key) {
            if let Some(s) = w.upgrade() {
                return Ok(s);
            }
        }
        let err = |e: cudarc::driver::DriverError| Error(format!("marlin scratch alloc: {e}"));
        let scratch = Arc::new(MarlinScratch {
            c_tmp: stream
                .alloc_zeros::<f32>(nsm * 64 * MAX_THREAD_N)
                .map_err(err)?,
            locks: stream.alloc_zeros::<i32>(nsm * 4).map_err(err)?,
        });
        reg.retain(|(k, w)| *k != key && w.strong_count() > 0);
        reg.push((key, Arc::downgrade(&scratch)));
        Ok(scratch)
    }

    /// A device-resident Marlin AWQ linear: repacked weights + a handle on the
    /// device-shared kernel scratch. `forward` computes `C[m,n] = A[m,k] . W`
    /// for f16 row-major `A` (lda = k).
    pub struct MarlinAwqLayer {
        pub k: usize,
        pub n: usize,
        pub group_size: usize,
        b_q: CudaSlice<u32>,
        scales: CudaSlice<f16>,
        zp: CudaSlice<u32>,
        scratch: Arc<MarlinScratch>,
        sms: i32,
        max_shared_mem: i32,
    }

    impl std::fmt::Debug for MarlinAwqLayer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "MarlinAwqLayer(k={}, n={}, gs={})",
                self.k, self.n, self.group_size
            )
        }
    }

    impl MarlinAwqLayer {
        /// Upload a repacked linear to `dev`.
        pub fn new(
            dev: &Arc<super::super::cuda::CudaDevice>,
            r: &MarlinAwqRepacked,
        ) -> Result<Self> {
            let stream = dev.stream();
            let info = dev.mmq_device_info()?;
            let err = |e: cudarc::driver::DriverError| Error(format!("marlin upload: {e}"));
            let b_q = stream.clone_htod(&r.b_q).map_err(err)?;
            let scales = stream.clone_htod(&r.scales).map_err(err)?;
            let zp = stream.clone_htod(&r.zp).map_err(err)?;
            let scratch = scratch_for(dev, info.nsm as usize)?;
            Ok(Self {
                k: r.k,
                n: r.n,
                group_size: r.group_size,
                b_q,
                scales,
                zp,
                scratch,
                sms: info.nsm,
                max_shared_mem: info.smpbo as i32,
            })
        }

        /// Raw-pointer launch core shared by the slice/view entry points.
        fn launch(
            &self,
            dev: &Arc<super::super::cuda::CudaDevice>,
            a_ptr: *const std::ffi::c_void,
            m: usize,
            c: &mut CudaSlice<f16>,
        ) -> Result<()> {
            if c.len() < m * self.n {
                return Err(Error("marlin forward: output buffer too small".into()));
            }
            let stream = dev.stream();
            let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
            let b_ptr = self.b_q.device_ptr(stream).0 as *const std::ffi::c_void;
            let s_ptr = self.scales.device_ptr(stream).0 as *const std::ffi::c_void;
            let z_ptr = self.zp.device_ptr(stream).0 as *const std::ffi::c_void;
            let c_ptr = c.device_ptr(stream).0 as *mut std::ffi::c_void;
            let ct_ptr = self.scratch.c_tmp.device_ptr(stream).0 as *mut f32;
            let l_ptr = self.scratch.locks.device_ptr(stream).0 as *mut i32;
            let rc = unsafe {
                loken_marlin_awq_f16_gemm(
                    a_ptr,
                    b_ptr,
                    s_ptr,
                    z_ptr,
                    c_ptr,
                    ct_ptr,
                    l_ptr,
                    m as i32,
                    self.n as i32,
                    self.k as i32,
                    (self.k / self.group_size) as i32,
                    self.sms,
                    self.max_shared_mem,
                    stream_ptr,
                )
            };
            if rc != 0 {
                return Err(Error(format!(
                    "marlin gemm failed rc={rc} (m={m} n={} k={})",
                    self.n, self.k
                )));
            }
            Ok(())
        }

        /// `C[m,n] = A[m,k] . W` - A f16 row-major contiguous, C f16 row-major
        /// (caller-allocated, `m*n` elements). Enqueues on the device stream.
        pub fn forward_into(
            &self,
            dev: &Arc<super::super::cuda::CudaDevice>,
            a: &CudaSlice<f16>,
            m: usize,
            c: &mut CudaSlice<f16>,
        ) -> Result<()> {
            if a.len() < m * self.k {
                return Err(Error("marlin forward: input buffer too small".into()));
            }
            let a_ptr = a.device_ptr(dev.stream()).0 as *const std::ffi::c_void;
            self.launch(dev, a_ptr, m, c)
        }

        /// [`forward_into`] over a borrowed `CudaView` (e.g. a facade tensor's
        /// storage via `cuda_ext::f16_slice_of`).
        pub fn forward_view_into(
            &self,
            dev: &Arc<super::super::cuda::CudaDevice>,
            a: &CudaView<'_, f16>,
            m: usize,
            c: &mut CudaSlice<f16>,
        ) -> Result<()> {
            if a.len() < m * self.k {
                return Err(Error("marlin forward: input view too small".into()));
            }
            let a_ptr = a.device_ptr(dev.stream()).0 as *const std::ffi::c_void;
            self.launch(dev, a_ptr, m, c)
        }

        /// Allocating variant of [`forward_into`].
        pub fn forward(
            &self,
            dev: &Arc<super::super::cuda::CudaDevice>,
            a: &CudaSlice<f16>,
            m: usize,
        ) -> Result<CudaSlice<f16>> {
            let stream = dev.stream();
            let mut c = unsafe { stream.alloc::<f16>(m * self.n) }
                .map_err(|e| Error(format!("marlin out alloc: {e}")))?;
            self.forward_into(dev, a, m, &mut c)?;
            Ok(c)
        }

        /// Allocating variant of [`forward_view_into`].
        pub fn forward_view(
            &self,
            dev: &Arc<super::super::cuda::CudaDevice>,
            a: &CudaView<'_, f16>,
            m: usize,
        ) -> Result<CudaSlice<f16>> {
            let stream = dev.stream();
            let mut c = unsafe { stream.alloc::<f16>(m * self.n) }
                .map_err(|e| Error(format!("marlin out alloc: {e}")))?;
            self.forward_view_into(dev, a, m, &mut c)?;
            Ok(c)
        }
    }

    #[cfg(test)]
    impl MarlinAwqLayer {
        /// The partial-tile reduce scratch, read back as it stands.
        ///
        /// This buffer answers, from the device, the one question a problem shape does not
        /// answer on its own: whether a launch cut its output tiles along k and had several
        /// blocks add their pieces together. The kernel's `global_reduce_fp32` is the only
        /// writer, and it is called only when a tile is shared; the buffer is allocated zeroed
        /// and the kernel never restores it. So a buffer still entirely zero after a launch is
        /// a launch on which every block owned a whole tile.
        pub(crate) fn reduction_scratch(
            &self,
            dev: &Arc<super::super::cuda::CudaDevice>,
        ) -> Result<Vec<f32>> {
            dev.stream()
                .clone_dtoh(&self.scratch.c_tmp)
                .map_err(|e| Error(format!("marlin scratch read: {e}")))
        }
    }
}

#[cfg(feature = "cuda")]
pub use device::{MarlinAwqLayer, MarlinScratch};

#[cfg(test)]
mod tests {
    use super::*;

    /// Both lane permutations must be true permutations of their domain.
    #[test]
    fn perms_are_valid() {
        let wp = weight_perm_u4();
        assert_eq!(wp.len(), 1024);
        let mut seen = vec![false; 1024];
        for &p in &wp {
            assert!(p < 1024 && !seen[p], "weight perm not a permutation");
            seen[p] = true;
        }
        let sp = scale_perm_grouped();
        let mut seen = [false; 64];
        for &p in &sp {
            assert!(p < 64 && !seen[p], "scale perm not a permutation");
            seen[p] = true;
        }
    }

    /// Repack output shapes and content stability. The GPU judge below catches a layout that
    /// moved; this one catches a size that did.
    #[test]
    fn repack_shapes() {
        let (k, n, gs) = (256usize, 128usize, 128usize);
        let pcols = n / 8;
        let groups = k / gs;
        let qweight: Vec<i32> = (0..k * pcols)
            .map(|i| (i as i32).wrapping_mul(2654435761u32 as i32))
            .collect();
        let qzeros: Vec<i32> = (0..groups * pcols)
            .map(|i| (i as i32).wrapping_mul(40503))
            .collect();
        let scales: Vec<f16> = (0..groups * n)
            .map(|i| f16::from_f32(0.01 + (i % 17) as f32 * 1e-3))
            .collect();
        let r = repack_awq_to_marlin(&qweight, &qzeros, &scales, k, n, gs).unwrap();
        assert_eq!(r.b_q.len(), (k / 16) * n * 2);
        assert_eq!(r.scales.len(), groups * n);
        assert_eq!(r.zp.len(), groups * pcols);
        // The repack is a pure reorder: nibble multiset must be preserved.
        let mut hist_in = [0usize; 16];
        for &w in &qweight {
            let mut v = w as u32;
            for _ in 0..8 {
                hist_in[(v & 0xF) as usize] += 1;
                v >>= 4;
            }
        }
        let mut hist_out = [0usize; 16];
        for &w in &r.b_q {
            let mut v = w;
            for _ in 0..8 {
                hist_out[(v & 0xF) as usize] += 1;
                v >>= 4;
            }
        }
        assert_eq!(hist_in, hist_out, "weight nibble multiset changed");
    }

    /// Nibbles, zero points and scales, packed into the AWQ layout the loader is handed.
    ///
    /// The generator keeps the plain `[k][n]` nibble and zero-point arrays and derives the
    /// packed form from them, so the reference never reads the packed bytes back: a repack
    /// that misplaces a nibble has nowhere to hide.
    ///
    /// ## Why the operand ranges are what they are
    ///
    /// They are picked so that the entire product-and-sum chain is EXACT, which is what lets
    /// the comparison below be bit-identity rather than a tolerance:
    ///
    ///   - an activation is a multiple of `2^-9` with `|a| <= 2^-2` - 128 steps either side of
    ///     zero, and exact in f16 (the ulp of `[2^-3, 2^-2)` is `2^-13`);
    ///   - a weight is `(q - z) * s` with `q - z` in `[-15, 15]` and `s` in
    ///     `{2^-6, 2^-7, 2^-8}`, so it is a multiple of `2^-8` with `|w| <= 15 * 2^-6` - 60
    ///     steps either side of zero, and exact in f16 for the same reason;
    ///   - every product is therefore a multiple of `2^-17`, and every partial sum of `K` of
    ///     them is bounded by `K * 2^-2 * 15 * 2^-6`. At the largest K here, 1024, that is 60,
    ///     and `60 / 2^-17 = 7.9e6` fits inside f32's exact integer range of `2^24 = 1.7e7`.
    ///
    /// That bound is the whole tolerance derivation. Because every partial sum is exactly
    /// representable, f32 addition over these terms is associative, so it does not matter in
    /// what order the two sides add: the kernel adds in tensor-core fragment order and, when a
    /// tile is split along k, adds f32 partials handed over by other blocks on top, while the
    /// reference adds straight along K. The only rounding left anywhere is the single f16 store
    /// of the finished accumulator, and both sides round the same f32 value. The permitted
    /// difference is zero.
    ///
    /// Nothing cancels between the two sides. The reference does not share a rounded quantity
    /// with the kernel - it starts from `q`, `z` and `s` and reconstructs the weight itself  -
    /// and each of those varies enough to be judged: `z` differs per column and per group, so
    /// a kernel that dropped the zero point would fail; `s` takes three different values, so a
    /// kernel that read a neighbouring group's scale would disagree on two columns in three;
    /// `q` and `a` vary along K, so a permutation that moved a nibble or a k index would fail.
    #[cfg(feature = "cuda")]
    #[derive(Clone)]
    struct AwqLinear {
        k: usize,
        n: usize,
        group_size: usize,
        /// `[k][n]` unsigned 4-bit weights.
        q: Vec<u8>,
        /// `[k/gs][n]` unsigned 4-bit zero points.
        z: Vec<u8>,
        /// `[k/gs][n]` scales, powers of two - the AWQ layout stores these as they are.
        s: Vec<f16>,
        qweight: Vec<i32>,
        qzeros: Vec<i32>,
    }

    #[cfg(feature = "cuda")]
    impl AwqLinear {
        fn generate(k: usize, n: usize, group_size: usize, seed: u32) -> Self {
            let groups = k / group_size;
            let q: Vec<u8> = (0..k * n)
                .map(|i| (hash(i, seed) >> 17) as u8 & 0xF)
                .collect();
            let z: Vec<u8> = (0..groups * n)
                .map(|i| (hash(i + k * n, seed) >> 19) as u8 & 0xF)
                .collect();
            let s: Vec<f16> = (0..groups * n)
                .map(|i| f16::from_f32(1.0 / (64u32 << (hash(i + 2 * k * n, seed) % 3)) as f32))
                .collect();

            let mut me = Self {
                k,
                n,
                group_size,
                q,
                z,
                s,
                qweight: Vec::new(),
                qzeros: Vec::new(),
            };
            me.repack();
            me
        }

        /// Rebuild both packed arrays from the plain ones, from scratch - so a perturbation of
        /// `q` or `z` reaches the device through exactly the path the originals took.
        fn repack(&mut self) {
            let pcols = self.n / 8;
            self.qweight = vec![0i32; self.k * pcols];
            for ki in 0..self.k {
                for ni in 0..self.n {
                    self.qweight[ki * pcols + ni / 8] |=
                        (self.q[ki * self.n + ni] as i32) << (4 * AWQ_ORDER[ni % 8]);
                }
            }
            self.qzeros = vec![0i32; (self.k / self.group_size) * pcols];
            for g in 0..self.k / self.group_size {
                for ni in 0..self.n {
                    self.qzeros[g * pcols + ni / 8] |=
                        (self.z[g * self.n + ni] as i32) << (4 * AWQ_ORDER[ni % 8]);
                }
            }
        }

        /// A copy with one operand disturbed - the negative controls below.
        fn perturbed(&self, disturb: impl FnOnce(&mut Self)) -> Self {
            let mut c = self.clone();
            disturb(&mut c);
            c.repack();
            c
        }

        fn upload(&self, dev: &std::sync::Arc<crate::tensor::cuda::CudaDevice>) -> MarlinAwqLayer {
            let r = repack_awq_to_marlin(
                &self.qweight,
                &self.qzeros,
                &self.s,
                self.k,
                self.n,
                self.group_size,
            )
            .expect("repack");
            MarlinAwqLayer::new(dev, &r).expect("upload")
        }

        /// The weight the kernel is supposed to reconstruct: `(q - z) * s`, in f16 because that
        /// is where the kernel forms it.
        fn weight(&self, ki: usize, ni: usize) -> f32 {
            let g = ki / self.group_size;
            let q = self.q[ki * self.n + ni] as f32;
            let z = self.z[g * self.n + ni] as f32;
            f16::from_f32((q - z) * self.s[g * self.n + ni].to_f32()).to_f32()
        }

        /// `A . W` on the host, from the plain nibbles: f32 accumulation along K, stored once
        /// in f16. Independent of the kernel in method as well as in code - it dequantises
        /// every weight up front and never forms a tile, a fragment or a partial sum.
        fn reference(&self, a: &[f16], m: usize) -> Vec<f16> {
            use rayon::prelude::*;
            let w: Vec<f32> = (0..self.k * self.n)
                .into_par_iter()
                .map(|i| self.weight(i / self.n, i % self.n))
                .collect();
            let mut c = vec![f16::ZERO; m * self.n];
            c.par_chunks_mut(self.n).enumerate().for_each(|(mi, row)| {
                for (ni, out) in row.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for ki in 0..self.k {
                        acc += a[mi * self.k + ki].to_f32() * w[ki * self.n + ni];
                    }
                    *out = f16::from_f32(acc);
                }
            });
            c
        }
    }

    #[cfg(feature = "cuda")]
    fn hash(i: usize, seed: u32) -> u32 {
        (i as u32)
            .wrapping_add(seed)
            .wrapping_mul(2_246_822_519)
            .rotate_left(13)
            .wrapping_mul(3_266_489_917)
    }

    /// Activations in the range the exactness argument on [`AwqLinear`] assumes: multiples of
    /// `2^-9`, never larger than `2^-2`, and varying along K so a misread k index shows.
    #[cfg(feature = "cuda")]
    fn activations(len: usize, seed: u32) -> Vec<f16> {
        (0..len)
            .map(|i| f16::from_f32(((hash(i, seed) >> 20) as i32 & 0xFF) as f32 / 512.0 - 0.25))
            .collect()
    }

    /// How many outputs disagree, and by how much at worst.
    #[cfg(feature = "cuda")]
    fn disagreements(got: &[f16], want: &[f16]) -> (usize, f32) {
        let n = got
            .iter()
            .zip(want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, w)| (g.to_f32() - w.to_f32()).abs())
            .fold(0f32, f32::max);
        (n, worst)
    }

    /// Whether this device runs the f32-accumulating kernel. Turing accumulates in f16, which
    /// rounds every partial sum and puts the bit-identity below out of reach; the comparison
    /// there would have to be a different one, so say so rather than pretend to cover it.
    #[cfg(feature = "cuda")]
    fn accumulates_in_f32(dev: &std::sync::Arc<crate::tensor::cuda::CudaDevice>) -> bool {
        dev.mmq_device_info().is_ok_and(|i| i.cc != 750)
    }

    /// The W4A16 GEMM, judged against the weights it was handed.
    ///
    /// The comparison is bit-identity, and [`AwqLinear`] derives why it is allowed to be: on
    /// these operands every product and every partial sum is exactly representable, so the two
    /// sides may add in any order and must still land on the same f32 before the single f16
    /// store. A kernel one accumulator out fails this.
    ///
    /// The batch heights are the shapes the launcher dispatches: 8 takes the half-tile
    /// instance, 16 and 32 the one- and two-block ones, 40 splits into a 32-row chunk plus a
    /// remainder - two launches, the second on a different kernel - and `32 * nsm` is one
    /// output tile per multiprocessor, which is the only height here that does NOT split its
    /// tiles along k. Which of them split is not asserted here; it is measured, on the device,
    /// by the test below.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_w4a16_gemm_matches_the_weights_it_was_given() {
        use crate::tensor::cuda::CudaDevice;
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the W4A16 kernel is NOT covered by this run");
            return;
        };
        if !accumulates_in_f32(&dev) {
            eprintln!("f16-accumulating kernel; bit-identity does not apply, NOT covered");
            return;
        }
        let nsm = dev.mmq_device_info().expect("device info").nsm as usize;
        let stream = dev.stream();

        // The narrow shape is one tile column and four scale groups; the wide one is four tile
        // columns and eight, so a block that walked into a neighbouring column's weights or a
        // neighbouring group's scales has somewhere to be caught.
        for (k, n) in [(512usize, 64usize), (1024, 256)] {
            let w = AwqLinear::generate(k, n, MARLIN_GROUP_SIZE, (k * n) as u32);
            let layer = w.upload(&dev);

            // Two negative controls, each disturbing one operand and each rebuilt and uploaded
            // through the ordinary path. A nibble is the finest thing the repack places; a
            // group scale is the finest thing the k-split has to get right, because a slice is
            // one scale group wide and a block that took the wrong one would read a plausible
            // number rather than a wrong-looking one.
            let flipped_nibble = w.perturbed(|c| c.q[0] ^= 1).upload(&dev);
            let halved_scale = w
                .perturbed(|c| {
                    let i = (c.k / c.group_size - 1) * c.n + c.n / 2;
                    c.s[i] = f16::from_f32(c.s[i].to_f32() / 2.0);
                })
                .upload(&dev);

            let mut heights = vec![8usize, 16, 32, 40];
            if n == MIN_THREAD_N {
                heights.push(MAX_M_PER_CHUNK * nsm);
            }
            for m in heights {
                let a = activations(m * k, m as u32);
                let want = w.reference(&a, m);
                let a_dev = stream.clone_htod(&a).expect("upload a");

                let got = stream
                    .clone_dtoh(&layer.forward(&dev, &a_dev, m).expect("gemm"))
                    .expect("download c");
                let (wrong, worst) = disagreements(&got, &want);
                assert_eq!(
                    wrong,
                    0,
                    "m={m} k={k} n={n}: {wrong}/{} outputs differ, worst {worst:e}",
                    got.len()
                );

                // The same comparison, against a kernel fed one disturbed operand, must find
                // the disturbance. This is what says the equality above is a judgement and not
                // two names for one buffer.
                for (what, off) in [("nibble", &flipped_nibble), ("scale", &halved_scale)] {
                    let off_got = stream
                        .clone_dtoh(&off.forward(&dev, &a_dev, m).expect("gemm"))
                        .expect("download c");
                    let (seen, _) = disagreements(&off_got, &want);
                    eprintln!(
                        "m={m} k={k} n={n}: a changed {what} moves {seen}/{} outputs",
                        off_got.len()
                    );
                    assert!(
                        seen > 0,
                        "m={m} k={k} n={n}: a changed {what} left every output identical"
                    );
                }
            }
        }
    }

    /// Where the stream-K split is, measured rather than argued.
    ///
    /// The kernel cuts its work in two parts: whole output tiles a block owns alone, and a
    /// remainder cut ALONG K so several blocks share a tile and add their pieces through an f32
    /// scratch. Which part a launch lands in is decided inside the kernel from `gridDim.x`, so
    /// no problem shape states it - but the scratch does, because the reduction is its only
    /// writer and it is allocated zeroed.
    ///
    /// One linear, two batch heights, and the criterion is the tile count against the
    /// multiprocessor count. At `m = 32 * nsm` the batch becomes `nsm` independent one-tile
    /// problems and N=64 gives one tile column, so there are exactly as many tiles as blocks
    /// and every block keeps its own: the scratch must be untouched. At `m = 16` there is one
    /// tile for the whole grid, so k is cut between blocks: the scratch must have been written.
    ///
    /// Both directions are asserted on purpose. An instrument that can only report the split
    /// cannot distinguish a kernel that always splits from a shape that reaches it.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_cross_block_reduction_runs_only_when_a_tile_is_split() {
        use crate::tensor::cuda::CudaDevice;
        // A private context, so the scratch this layer picks up is one nothing has launched on
        // and is still holding the zeros it was allocated with.
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the W4A16 stream-K path is NOT covered by this run");
            return;
        };
        let nsm = dev.mmq_device_info().expect("device info").nsm as usize;
        // Above 128 tiles the launcher splits the batch into several launches and the tall
        // height stops being one tile per block; no such device is to hand to re-derive on.
        if nsm > 128 {
            eprintln!("nsm={nsm} exceeds the launcher's chunk ceiling; NOT covered by this run");
            return;
        }

        let (k, n) = (512usize, MIN_THREAD_N);
        let w = AwqLinear::generate(k, n, MARLIN_GROUP_SIZE, 7);
        let layer = w.upload(&dev);
        let stream = dev.stream();

        let unsplit = MAX_M_PER_CHUNK * nsm;
        for (m, splits) in [(unsplit, false), (16usize, true)] {
            let a = activations(m * k, m as u32);
            let a_dev = stream.clone_htod(&a).expect("upload a");
            let got = stream
                .clone_dtoh(&layer.forward(&dev, &a_dev, m).expect("gemm"))
                .expect("download c");

            // Whatever path it took, it still has to be right.
            if accumulates_in_f32(&dev) {
                let (wrong, worst) = disagreements(&got, &w.reference(&a, m));
                assert_eq!(wrong, 0, "m={m}: {wrong} outputs differ, worst {worst:e}");
            }

            let touched = layer
                .reduction_scratch(&dev)
                .expect("read scratch")
                .iter()
                .filter(|v| **v != 0.0)
                .count();
            eprintln!("m={m} on {nsm} multiprocessors: {touched} partial sums in the scratch");
            if splits {
                assert!(
                    touched > 0,
                    "m={m} k={k} n={n} on {nsm} multiprocessors: one output tile for the whole \
                     grid, yet no block left a partial sum - the k-split never ran, so the \
                     tests above judge the single-slice path only"
                );
            } else {
                assert_eq!(
                    touched, 0,
                    "m={m} k={k} n={n} on {nsm} multiprocessors: one tile per block, yet {touched} \
                     partial sums were written - the scratch cannot tell the two paths apart"
                );
            }
        }
    }
}
