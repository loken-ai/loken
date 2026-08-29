//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// The device dtypes, one row per format: the storage variant (which is also the [`DType`]
/// and the [`CpuStorage`] variant of the same name), the Rust element type it holds, the
/// accessor that borrows it, and how that accessor reads in an error.
///
/// Everything asked of a device buffer - its dtype, its length, how it is allocated, uploaded,
/// copied between cards and read back - is derived from this one list. Those were eight
/// separate tables over the same eight formats, which is eight places to forget a format in
/// and no way to notice.
macro_rules! device_dtypes {
    ($($variant:ident, $ty:ty, $accessor:ident, $name:literal, $what:literal;)+) => {
        impl CudaStorage {
            pub fn dtype(&self) -> DType {
                match self {
                    $(Self::$variant(_) => DType::$variant,)+
                }
            }

            pub fn len(&self) -> usize {
                match self {
                    $(Self::$variant(s) => s.len(),)+
                }
            }

            $(
                #[doc = concat!("Borrow the ", $name, " device slice (", $what, ").")]
                pub fn $accessor(&self) -> Result<&CudaSlice<$ty>> {
                    match self {
                        Self::$variant(s) => Ok(s),
                        other => Err(Error(format!(
                            concat!("expected ", $name, " cuda storage, got {}"),
                            other.dtype()
                        ))),
                    }
                }
            )+

            /// Allocate a zeroed device buffer directly (async memset on the
            /// device's stream - no host alloc, no H2D copy). The decode hot path
            /// allocates kernel output buffers through `Tensor::zeros` every
            /// layer/token; the previous CPU-vec + upload route cost a synchronous
            /// `cuMemcpyHtoDAsync` per call.
            pub fn zeros(dev: &CudaDevice, dtype: DType, n: usize) -> Result<Self> {
                let stream = dev.stream();
                Ok(match dtype {
                    $(DType::$variant => Self::$variant(with_oom_retry(dev, "alloc_zeros", || {
                        stream.alloc_zeros(n)
                    })?),)+
                    DType::F64 => {
                        return Err(Error("cuda alloc_zeros: f64 not device-supported".into()))
                    }
                })
            }

            /// Uninitialized typed device buffer (callers must overwrite every
            /// element - the cat/assembly outputs do).
            pub fn alloc_uninit(dev: &CudaDevice, dtype: DType, n: usize) -> Result<Self> {
                let stream = dev.stream();
                // SAFETY: documented contract - every element is written by the
                // caller's follow-up kernels before any read.
                Ok(match dtype {
                    $(DType::$variant => Self::$variant(with_oom_retry(dev, "alloc", || unsafe {
                        stream.alloc(n)
                    })?),)+
                    DType::F64 => return Err(Error("cuda alloc: f64 not device-supported".into())),
                })
            }

            /// Upload host storage to the device.
            pub fn upload(dev: &CudaDevice, cpu: &CpuStorage) -> Result<Self> {
                let stream = dev.stream();
                Ok(match cpu {
                    $(CpuStorage::$variant(v) => {
                        Self::$variant(with_oom_retry(dev, "upload", || stream.clone_htod(v))?)
                    })+
                    CpuStorage::F64(_) => {
                        return Err(Error("cuda upload: f64 not device-supported".into()))
                    }
                })
            }

            /// Download device storage to the host.
            pub fn download(&self, dev: &CudaDevice) -> Result<CpuStorage> {
                let stream = dev.stream();
                let err = |e| Error(format!("cuda download: {e}"));
                Ok(match self {
                    $(Self::$variant(s) => {
                        CpuStorage::$variant(stream.clone_dtoh(s).map_err(err)?)
                    })+
                })
            }

            /// Cross-GPU peer copy onto `target`'s stream (cuMemcpyPeerAsync via
            /// cudarc's cross-context `memcpy_dtod`). Caller must have synced the
            /// SOURCE device first - the copy is ordered only on the target stream.
            pub fn peer_copy(&self, src_dev: &CudaDevice, target: &CudaDevice) -> Result<Self> {
                if !Self::can_access_peer(src_dev.ordinal(), target.ordinal()) {
                    // Host bounce: download on the source device, upload on the target.
                    // Correct on every topology, and the only option without P2P.
                    let host = self.download(src_dev)?;
                    return Self::upload(target, &host);
                }
                let stream = target.stream();
                let err = |e| Error(format!("cuda peer copy: {e}"));
                Ok(match self {
                    $(Self::$variant(s) => {
                        // SAFETY: fully overwritten by the memcpy below.
                        let mut dst: CudaSlice<$ty> = with_oom_retry(target, "peer copy", || {
                            unsafe { stream.alloc(s.len()) }
                        })?;
                        stream.memcpy_dtod(s, &mut dst).map_err(err)?;
                        Self::$variant(dst)
                    })+
                })
            }
        }
    };
}

device_dtypes! {
    U8,   u8,          as_u8_slice,   "u8",   "raw byte buffers";
    U32,  u32,         as_u32_slice,  "u32",  "sort indices / token ids";
    I16,  i16,         as_i16_slice,  "i16",  "narrow integer buffers";
    I32,  i32,         as_i32_slice,  "i32",  "AWQ qweight/qzeros";
    I64,  i64,         as_i64_slice,  "i64",  "graph kv positions";
    F16,  half::f16,   as_f16_slice,  "f16",  "kernel-launch boundary";
    BF16, half::bf16,  as_bf16_slice, "bf16", "kernel-launch boundary";
    F32,  f32,         as_f32_slice,  "f32",  "kernel-launch boundary";
}

impl CudaStorage {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `src` can address `dst`'s memory directly (cached per ordered pair).
    /// Consumer GPUs on plain PCIe frequently report P2P as NOT supported; a
    /// device-to-device memcpy issued anyway does not fail at the call - it
    /// corrupts the stream and surfaces later as `CUDA_ERROR_LAUNCH_FAILED` from
    /// an unrelated synchronize, which is how a cross-GPU model split failed
    /// mid-forward with no usable error site.
    fn can_access_peer(src: usize, dst: usize) -> bool {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<(usize, usize), bool>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut g = cache.lock().unwrap_or_else(|e| e.into_inner());
        *g.entry((src, dst)).or_insert_with(|| {
            let mut can: std::ffi::c_int = 0;
            let ok = unsafe {
                cudarc::driver::sys::cuDeviceCanAccessPeer(
                    &mut can,
                    src as std::ffi::c_int,
                    dst as std::ffi::c_int,
                )
            } == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;
            let can = ok && can != 0;
            if !can {
                tracing::info!(
                    "cuda: GPU{src} cannot address GPU{dst} directly (no P2P);                      cross-GPU tensor moves bounce through host memory"
                );
            }
            can
        })
    }

    /// [`Self::upload`], made safe against the caller freeing the host buffer.
    ///
    /// `clone_htod` issues the copy ASYNC on the stream and returns while the borrowed
    /// host Vec may still be the DMA source; the caller then frees it, the allocator
    /// reuses the pages, and the device receives whatever landed there next. Measured
    /// twice: the loader's RoPE tables (NaN from block zero on a slower machine) and
    /// stable-audio's per-forward tables (a fixed seed answering different audio on
    /// every call). The barrier costs microseconds once per host upload and buys back
    /// the only property a seed promises.
    pub fn upload_host_safe(dev: &CudaDevice, cpu: &CpuStorage) -> Result<Self> {
        let out = Self::upload(dev, cpu)?;
        dev.stream()
            .synchronize()
            .map_err(|e| Error(format!("upload barrier: {e}")))?;
        Ok(out)
    }
}

/// One thread per element, the launch shape every kernel in [`ELEMENTWISE_SRC`] is written
/// for: it derives its own index from `blockIdx.x * blockDim.x + threadIdx.x` and returns
/// when that index passes the element count. The block width is the launcher's to choose,
/// so it is chosen here rather than at each launch site.
pub(crate) fn elementwise_launch(n: usize) -> LaunchConfig {
    const BLOCK: usize = 256;
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK) as u32, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Small f32 ops with no production-kernel equivalent (those live in
/// fused_kernels for the shapes the decode path fuses; these are the plain
/// element-wise forms the native tensor needs).
pub(super) const ELEMENTWISE_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// One thread per element, the element read into `v`, one expression written back. Most of
// the maps below are exactly that and nothing else, so the shell is stated once here and
// each of them contributes only the expression that makes it what it is.
#define NATIVE_MAP(NAME, TIN, TOUT, EXPR)                                                    \
extern "C" __global__ void NAME(const TIN* x, TOUT* y, int n) {                               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                            \
    if (i < n) { TIN v = x[i]; y[i] = (EXPR); }                                               \
}

// Gather through permuted strides: decompose the output's flat index by the output dims,
// fastest axis first, and accumulate the input offset from the matching input stride.
// Permute and broadcast_as are pure data movement, so each element width gets this same
// index math over its own type.
#define NATIVE_PERMUTE(NAME, T)                                                              \
extern "C" __global__ void NAME(                                                              \
    const T* x, T* y, const int* odims, const int* in_strides_perm, int rank, int n) {        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                            \
    if (i >= n) return;                                                                       \
    int rem = i;                                                                              \
    int src = 0;                                                                              \
    for (int ax = rank - 1; ax >= 0; ax--) {                                                  \
        int idx = rem % odims[ax];                                                            \
        rem /= odims[ax];                                                                     \
        src += idx * in_strides_perm[ax];                                                     \
    }                                                                                         \
    y[i] = x[src];                                                                            \
}

// Copy one cat operand into the output: src viewed as [outer, src_row], dst rows are
// dst_stride apart starting at element offset dst_off (replaces the per-outer-slab
// dtod-copy loop - one launch per operand). Cat is pure data movement too, so every dtype
// routes through the kernel of its element width.
#define NATIVE_CAT_COPY(NAME, T)                                                             \
extern "C" __global__ void NAME(                                                              \
    const T* src, T* dst, long long n,                                                        \
    int src_row, int dst_stride, int dst_off) {                                               \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (i >= n) return;                                                                       \
    long long r = i / src_row;                                                                \
    int j = (int)(i % src_row);                                                               \
    dst[r * dst_stride + dst_off + j] = src[i];                                               \
}

// Block-wide pairwise sum of two shared accumulators: halve the active thread count each
// round, adding the far half into the near one, with the whole block meeting before the
// first round and between rounds. Both normalisations below reduce a sum and a sum of
// squares together this way.
#define NATIVE_BLOCK_REDUCE_PAIR(a, b)                                                       \
    __syncthreads();                                                                          \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                                            \
        if (threadIdx.x < s) {                                                                \
            a[threadIdx.x] += a[threadIdx.x + s];                                             \
            b[threadIdx.x] += b[threadIdx.x + s];                                             \
        }                                                                                     \
        __syncthreads();                                                                      \
    }

NATIVE_MAP(native_cast_f32_bf16, float, __nv_bfloat16, __float2bfloat16(v))
NATIVE_MAP(native_cast_bf16_f32, __nv_bfloat16, float, __bfloat162float(v))
NATIVE_MAP(native_sin_f32, float, float, sinf(v))
NATIVE_MAP(native_cos_f32, float, float, cosf(v))
// narrow over one dim of a contiguous tensor, dtype-agnostic (byte copy):
// y[r*row_bytes + j] = x[r*src_row_bytes + off_bytes + j]
extern "C" __global__ void native_slice_u8(
    const unsigned char* x, unsigned char* y,
    long long off_bytes, long long src_row_bytes, long long row_bytes,
    long long n_rows) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_rows * row_bytes;
    if (i >= total) return;
    long long r = i / row_bytes;
    long long j = i % row_bytes;
    y[i] = x[r * src_row_bytes + off_bytes + j];
}
// im2col for 1-D conv: one (batch,group) slab. col[(ci*k+kk)*l_out+lo]
// Restricted to output columns [lo_off, lo_off+lo_len) so callers can bound
// the col-buffer transient (same contract as native_im2col2d_f32 below).
extern "C" __global__ void native_im2col1d_f32(
    const float* x, float* col,
    int c_in_g, int l, int k, int l_out,
    int lo_off, int lo_len,
    int padding, int stride, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = c_in_g * k * lo_len;
    if (idx >= total) return;
    int lo = lo_off + idx % lo_len;
    if (lo >= l_out) return;
    int kk = (idx / lo_len) % k;
    int ci = idx / (lo_len * k);
    int pos = lo * stride + kk * dilation;
    float v = 0.0f;
    if (pos >= padding && pos - padding < l) v = x[ci * l + pos - padding];
    col[idx] = v;
}
// im2col for 2-D conv: one (batch,group) slab, restricted to output columns
// [col_off, col_off+col_len) so callers can bound the col-buffer transient.
// col[prow*col_len + (o - col_off)]
extern "C" __global__ void native_im2col2d_f32(
    const float* x, float* col,
    int c_in_g, int h, int w, int kh, int kw,
    int w_out, int col_off, int col_len,
    int padding, int stride, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int patch = c_in_g * kh * kw;
    int total = patch * col_len;
    if (idx >= total) return;
    int o = col_off + idx % col_len;
    int prow = idx / col_len;
    int wo = o % w_out;
    int ho = o / w_out;
    int kj = prow % kw;
    int ki = (prow / kw) % kh;
    int ci = prow / (kw * kh);
    int hi = ho * stride + ki * dilation;
    int wi = wo * stride + kj * dilation;
    float v = 0.0f;
    if (hi >= padding && wi >= padding && hi - padding < h && wi - padding < w)
        v = x[ci * h * w + (hi - padding) * w + (wi - padding)];
    col[idx] = v;
}
NATIVE_MAP(native_relu_f32, float, float, fmaxf(v, 0.0f))
NATIVE_MAP(native_cast_f32_f16, float, __half, __float2half(v))
NATIVE_MAP(native_cast_f16_f32, __half, float, __half2float(v))
extern "C" __global__ void native_add_f32(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
extern "C" __global__ void native_mul_f32(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}
NATIVE_MAP(native_silu_f32, float, float, v / (1.0f + expf(-v)))
extern "C" __global__ void native_gelu_f32(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // gelu_fwd's exact evaluation order (x_sq, x_cube, alpha):
        // a different association rounds differently.
        float v = x[i];
        float x_sq = v * v;
        float x_cube = x_sq * v;
        float alpha = v + 0.044715f * x_cube;
        y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * alpha));
    }
}
// Exact (erf) GELU on-device - the gelu_erf CUDA path previously round-tripped to the CPU
// (to_device(Cpu) -> erf -> back), which on a big FFN activation is a ~100MB D2H+H2D per call.
NATIVE_MAP(native_gelu_erf_f32, float, float, 0.5f * v * (1.0f + erff(v * 0.7071067811865476f)))
// -- f16 ops computed IN HALF, mirroring the upstream kernel set exactly:
// the facade's f16 affine/silu/gelu round per half-op (x*mul+add in __half;
// silu = x / (1 + hexp(-x)); gelu's tanh chain in half with tanhf upcast).
// The previous f32-cast detour rounded once at the end - up to ~4e-2 rel
// difference, which seeded the per-layer parity drift.
extern "C" __global__ void native_affine_f16(const __half* x, __half* y, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __half mul = __float2half(alpha);
    __half add = __float2half(beta);
    if (i < n) y[i] = x[i] * mul + add;
}
NATIVE_MAP(native_silu_f16, __half, __half, v / (__half(1.0f) + hexp(-v)))
extern "C" __global__ void native_gelu_f16(const __half* x, __half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half v = x[i];
        __half x_sq = v * v;
        __half x_cube = x_sq * v;
        __half alpha = v + __half(0.044715f) * x_cube;
        __half t = __float2half(tanhf(__half2float(__half(0.7978845608028654f) * alpha)));
        y[i] = __half(0.5f) * v * (__half(1.0f) + t);
    }
}
extern "C" __global__ void native_div_f16(const __half* a, const __half* b, __half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}
// Direct half binary ops (f16 elementwise is computed IN HALF - the
// f32-cast detour cost 2 extra kernels + allocs per op on the decode path).
extern "C" __global__ void native_add_f16(const __half* a, const __half* b, __half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
extern "C" __global__ void native_mul_f16(const __half* a, const __half* b, __half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}
// Tail-aligned broadcast in half (the rope/mask/norm-weight shapes).
extern "C" __global__ void native_badd_tail_f16(const __half* a, const __half* b, __half* y, int n, int bn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i % bn];
}
extern "C" __global__ void native_bmul_tail_f16(const __half* a, const __half* b, __half* y, int n, int bn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i % bn];
}
// -- bf16 ops: compute through f32 and round once. For add/mul/div/affine a
// single f32 rounding is BIT-IDENTICAL to native bf16 arithmetic (operands
// are exact in f32, the one round-to-nearest lands on the same bf16), and it
// compiles on any NVRTC target arch. Saves the cast-detour's 2 extra kernels
// + allocs per op on bf16-heavy decode paths.
extern "C" __global__ void native_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}
extern "C" __global__ void native_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(__bfloat162float(a[i]) * __bfloat162float(b[i]));
}
extern "C" __global__ void native_div_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(__bfloat162float(a[i]) / __bfloat162float(b[i]));
}
extern "C" __global__ void native_badd_tail_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, int n, int bn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i % bn]));
}
extern "C" __global__ void native_bmul_tail_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, int n, int bn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(__bfloat162float(a[i]) * __bfloat162float(b[i % bn]));
}
extern "C" __global__ void native_affine_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // the reference bf16 affine rounds mul and add separately (per half-op).
    if (i < n) {
        float m = __bfloat162float(__float2bfloat16(__bfloat162float(x[i]) * __bfloat162float(__float2bfloat16(alpha))));
        y[i] = __float2bfloat16(m + __bfloat162float(__float2bfloat16(beta)));
    }
}
// Direct division (single rounding - the reference `bdiv` semantics; the old
// mul-by-recip route rounded twice).
extern "C" __global__ void native_div_f32(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}
extern "C" __global__ void native_scale_f32(const float* x, float* y, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * alpha;
}
// rhs broadcast over the TRAILING dims of lhs (right-aligned, the transformer
// cases: [cols] weight over [..., cols]; [1,1,s,s] mask over [b,h,s,s]).
extern "C" __global__ void native_badd_tail_f32(const float* a, const float* b, float* y, int n, int bn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i % bn];
}
extern "C" __global__ void native_bmul_tail_f32(const float* a, const float* b, float* y, int n, int bn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i % bn];
}
// dims/strides come in device memory. The f32 permute, and the same gather for 1/2/4/8-byte dtypes, so non-f32 broadcast_as
// (u8 masks, f16 tables, i64 positions) stays on-device.
NATIVE_PERMUTE(native_permute_f32, float)
NATIVE_PERMUTE(native_permute_w8, unsigned char)
NATIVE_PERMUTE(native_permute_w16, unsigned short)
NATIVE_PERMUTE(native_permute_w32, unsigned int)
NATIVE_PERMUTE(native_permute_w64, unsigned long long)
// RoPE, both layouts. x/y: [bh, seq, d]; cos/sin: [seq, d/2].
extern "C" __global__ void native_rope_f32(
    const float* x, const float* c, const float* s, float* y,
    int bh, int seq, int d, int interleaved) {
    int half = d / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bh * seq * half;
    if (i >= total) return;
    int p = i % half;
    int t = (i / half) % seq;
    int b = i / (half * seq);
    const float* xr = x + ((long)b * seq + t) * d;
    float* yr = y + ((long)b * seq + t) * d;
    float cv = c[t * half + p];
    float sv = s[t * half + p];
    if (interleaved) {
        float x0 = xr[2 * p], x1 = xr[2 * p + 1];
        yr[2 * p] = x0 * cv - x1 * sv;
        yr[2 * p + 1] = x0 * sv + x1 * cv;
    } else {
        float x0 = xr[p], x1 = xr[p + half];
        yr[p] = x0 * cv - x1 * sv;
        yr[p + half] = x0 * sv + x1 * cv;
    }
}
NATIVE_MAP(native_exp_f32, float, float, expf(v))
extern "C" __global__ void native_affine_f32(const float* x, float* y, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * alpha + beta;
}
// Per-channel bias over NCHW-style layouts: y[i] = a[i] + b[(i / spatial) % c].
extern "C" __global__ void native_bias_chw_f32(
    const float* a, const float* b, float* y, int n, int spatial, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[(i / spatial) % c];
}
// Nearest-neighbor 2-D upsample on [bc, h, w] -> [bc, oh, ow]
// (src index = floor(dst * in / out), the standard nearest mapping).
extern "C" __global__ void native_upsample2d_f32(
    const float* x, float* y, int bc, int h, int w, int oh, int ow) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)bc * oh * ow;
    if (i >= total) return;
    int wo = i % ow;
    int ho = (i / ow) % oh;
    long ci = i / ((long)ow * oh);
    int hi = (int)((long)ho * h / oh);
    int wi = (int)((long)wo * w / ow);
    y[i] = x[(ci * h + hi) * w + wi];
}
// Bilinear 2-D resample on [bc, h, w] -> [bc, oh, ow], HALF-PIXEL centres
// (align_corners = false), which is what both PyTorch and ONNX's
// pytorch_half_pixel mode use. Corner-aligned sampling shifts the image by half an
// output pixel, which is visible as a drift when it feeds a face paste-back.
extern "C" __global__ void native_upsample_bilinear2d_f32(
    const float* x, float* y, int bc, int h, int w, int oh, int ow) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)bc * oh * ow;
    if (i >= total) return;
    int wo = i % ow;
    int ho = (i / ow) % oh;
    long ci = i / ((long)ow * oh);
    float sy = ((float)ho + 0.5f) * ((float)h / (float)oh) - 0.5f;
    float sx = ((float)wo + 0.5f) * ((float)w / (float)ow) - 0.5f;
    if (sy < 0.0f) sy = 0.0f;
    if (sx < 0.0f) sx = 0.0f;
    int y0 = (int)floorf(sy);
    int x0 = (int)floorf(sx);
    int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
    int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
    float fy = sy - (float)y0;
    float fx = sx - (float)x0;
    const float* p = x + ci * (long)h * w;
    float v00 = p[(long)y0 * w + x0], v01 = p[(long)y0 * w + x1];
    float v10 = p[(long)y1 * w + x0], v11 = p[(long)y1 * w + x1];
    y[i] = (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy)
         + (v10 * (1.0f - fx) + v11 * fx) * fy;
}
// Zero-pad ONE dim, viewed as [outer, d_in, inner] -> [outer, d_out, inner].
extern "C" __global__ void native_pad_dim_f32(
    const float* x, float* y, int outer, int d_in, int d_out, int left, int inner) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)outer * d_out * inner;
    if (i >= total) return;
    int ii = i % inner;
    int j = (i / inner) % d_out;
    long o = i / ((long)inner * d_out);
    float v = 0.0f;
    if (j >= left && j - left < d_in) v = x[(o * d_in + (j - left)) * inner + ii];
    y[i] = v;
}
// Snake activation on [b, c, l]: y = x + inv_alpha[c] * sin(alpha[c]*x)^2
// (fused - the per-channel broadcasts aren't tail-aligned).
extern "C" __global__ void native_snake1d_f32(
    const float* x, const float* alpha, const float* inv_alpha, float* y,
    long n, int l, int c) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)((i / l) % c);
    float v = x[i];
    float s = sinf(alpha[ch] * v);
    y[i] = v + inv_alpha[ch] * s * s;
}
// Direct transposed 1-D convolution (stride upsampling, dilation 1,
// groups 1): one thread per output element, gathering the contributing
// input taps. w layout: [c_in, c_out, k].
extern "C" __global__ void native_convt1d_f32(
    const float* x, const float* w, float* y,
    int c_in, int c_out, int l_in, int l_out, int k, int stride, int padding) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)c_out * l_out;
    if (i >= total) return;
    int lo = (int)(i % l_out);
    int co = (int)(i / l_out);
    float acc = 0.0f;
    for (int kk = 0; kk < k; kk++) {
        int pos = lo + padding - kk;
        if (pos < 0 || pos % stride != 0) continue;
        int li = pos / stride;
        if (li >= l_in) continue;
        for (int ci = 0; ci < c_in; ci++) {
            acc += x[ci * l_in + li] * w[(ci * c_out + co) * k + kk];
        }
    }
    y[i] = acc;
}
// col2im for a GEMM-decomposed transposed 1-D conv (N=1, dilation 1). `colT`
// holds the per-tap contributions `colT[(kk*c_out+oc), ti]` (the cuBLAS hgemm
// output W_rearranged.x, F16); this overlap-adds them into the output and adds
// bias - the c_in contraction already happened in the GEMM, so here we only
// gather over the k taps (no inner channel loop), the cheap bandwidth half.
extern "C" __global__ void native_col2im1d_f16(
    const __half* colT, const float* bias, float* y,
    int c_out, int l_out, int k, int t_in, int stride, int padding) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)c_out * l_out;
    if (i >= total) return;
    int lo = (int)(i % l_out);
    int co = (int)(i / l_out);
    float acc = 0.0f;
    for (int kk = 0; kk < k; kk++) {
        int pos = lo + padding - kk;
        if (pos < 0 || pos % stride != 0) continue;
        int li = pos / stride;
        if (li >= t_in) continue;
        acc += __half2float(colT[((long)(kk * c_out + co)) * t_in + li]);
    }
    y[i] = acc + bias[co];
}
NATIVE_MAP(native_sigmoid_f32, float, float, 1.0f / (1.0f + expf(-v)))
// General N-d broadcast binary: the output flat index is decomposed by the
// output dims (row-major); each operand is gathered through per-axis
// effective strides (0 on its broadcast axes). meta = [odims | a_strides |
// b_strides], one device array. op: 0=add, 1=mul (uniform branch).
extern "C" __global__ void native_bbin_nd_f32(
    const float* a, const float* b, float* y,
    const int* meta, int rank, long long n, int op) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long rem = i;
    long long ia = 0, ib = 0;
    for (int ax = rank - 1; ax >= 0; ax--) {
        int idx = (int)(rem % meta[ax]);
        rem /= meta[ax];
        ia += (long long)idx * meta[rank + ax];
        ib += (long long)idx * meta[2 * rank + ax];
    }
    float r;
    if (op == 0) r = a[ia] + b[ib];
    else if (op == 1) r = a[ia] * b[ib];
    else r = a[ia] / b[ib];
    y[i] = r;
}
NATIVE_MAP(native_tanh_f32, float, float, tanhf(v))
NATIVE_MAP(native_abs_f32, float, float, fabsf(v))
NATIVE_MAP(native_recip_f32, float, float, 1.0f / v)
NATIVE_MAP(native_sqrt_f32, float, float, sqrtf(v))
extern "C" __global__ void native_pow_f32(const float* x, float* y, float e, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = powf(x[i], e);
}
// Reduce over the middle `red` axis: input [outer, red, inner] -> output [outer, inner].
// opcode 0 = sum (scale=1 gives sum, scale=1/red gives mean), 1 = max. One thread per output.
extern "C" __global__ void native_reduce_f32(
    const float* x, float* y, int outer, int red, int inner, int opcode, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;
    int o = idx / inner, i = idx % inner;
    const float* base = x + ((long)o * red) * inner + i;
    float acc = (opcode == 1) ? __int_as_float(0xff800000) : 0.0f; // -inf (NVRTC has no <math.h>)
    for (int r = 0; r < red; ++r) {
        float v = base[(long)r * inner];
        acc = (opcode == 1) ? fmaxf(acc, v) : (acc + v);
    }
    y[idx] = acc * scale;
}
extern "C" __global__ void native_bdiv_tail_f32(const float* a, const float* b, float* y, int n, int bn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i % bn];
}
// element select: cond non-zero -> on_true (u8 / u32 condition variants)
extern "C" __global__ void native_where_u8_f32(
    const unsigned char* c, const float* t, const float* f, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = c[i] ? t[i] : f[i];
}
extern "C" __global__ void native_where_u32_f32(
    const unsigned int* c, const float* t, const float* f, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = c[i] ? t[i] : f[i];
}
// inverse of native_slice_u8: write src rows INTO dst at a byte offset
// (in-place slice_set - the O(1) KV append / graph-buffer update;
// dtype-agnostic byte copy)
extern "C" __global__ void native_slice_set_u8(
    const unsigned char* x, unsigned char* y,
    long long off_bytes, long long dst_row_bytes, long long row_bytes,
    long long n_rows) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_rows * row_bytes;
    if (i >= total) return;
    long long r = i / row_bytes;
    long long j = i % row_bytes;
    y[r * dst_row_bytes + off_bytes + j] = x[i];
}
// scatter_set along one dim: dst[(o*dst_d + idx[e])*inner + ii] = src[e]
// (in-place; idx has src's shape; elem-size-agnostic byte copy)
extern "C" __global__ void native_scatter_set_i64(
    const unsigned char* src, unsigned char* dst, const long long* idx,
    long long n, int inner, int src_d, int dst_d, int esize) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    long long ii = e % inner;
    long long o = e / ((long long)inner * src_d);
    long long t = idx[e];
    if (t < 0 || t >= dst_d) return;
    long long de = ((o * dst_d + t) * inner + ii) * esize;
    long long se = e * esize;
    for (int b = 0; b < esize; b++) dst[de + b] = src[se + b];
}
extern "C" __global__ void native_scatter_set_u32(
    const unsigned char* src, unsigned char* dst, const unsigned int* idx,
    long long n, int inner, int src_d, int dst_d, int esize) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    long long ii = e % inner;
    long long o = e / ((long long)inner * src_d);
    long long t = idx[e];
    if (t >= dst_d) return;
    long long de = ((o * dst_d + t) * inner + ii) * esize;
    long long se = e * esize;
    for (int b = 0; b < esize; b++) dst[de + b] = src[se + b];
}
// gather along one dim (non-dim dims equal - the decode-path form):
// out[e] = src[(o*src_d + idx[e])*inner + ii]
extern "C" __global__ void native_gather_i64(
    const unsigned char* src, const long long* idx, unsigned char* out,
    long long n, int inner, int idx_d, int src_d, int esize) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    long long ii = e % inner;
    long long o = e / ((long long)inner * idx_d);
    long long t = idx[e];
    if (t < 0 || t >= src_d) t = 0; // clamp instead of OOB-reading
    long long se = ((o * src_d + t) * inner + ii) * esize;
    for (int b = 0; b < esize; b++) out[e * esize + b] = src[se + b];
}
extern "C" __global__ void native_gather_u32(
    const unsigned char* src, const unsigned int* idx, unsigned char* out,
    long long n, int inner, int idx_d, int src_d, int esize) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    long long ii = e % inner;
    long long o = e / ((long long)inner * idx_d);
    long long t = idx[e];
    if (t >= src_d) t = 0; // clamp instead of OOB-reading
    long long se = ((o * src_d + t) * inner + ii) * esize;
    for (int b = 0; b < esize; b++) out[e * esize + b] = src[se + b];
}
// index_select along one dim: out viewed [outer, idx_len, inner], src viewed
// [outer, src_d, inner], ids 1-D [idx_len] (one index per OUTPUT ROW - unlike
// gather's per-element idx). The token-embedding lookup form: previously a
// full-table host bounce (qwen3.5 vocab-248320 table = ~2 GB D2H per token).
extern "C" __global__ void native_index_select_u32(
    const unsigned char* src, const unsigned int* ids, unsigned char* out,
    long long n, int inner, int idx_len, int src_d, int esize) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    long long ii = e % inner;
    long long rest = e / inner;
    long long r = rest % idx_len;
    long long o = rest / idx_len;
    long long t = ids[r];
    if (t >= src_d) t = 0; // clamp instead of OOB-reading
    long long se = ((o * src_d + t) * inner + ii) * esize;
    for (int b = 0; b < esize; b++) out[e * esize + b] = src[se + b];
}
extern "C" __global__ void native_index_select_i64(
    const unsigned char* src, const long long* ids, unsigned char* out,
    long long n, int inner, int idx_len, int src_d, int esize) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    long long ii = e % inner;
    long long rest = e / inner;
    long long r = rest % idx_len;
    long long o = rest / idx_len;
    long long t = ids[r];
    if (t < 0 || t >= src_d) t = 0; // clamp instead of OOB-reading
    long long se = ((o * src_d + t) * inner + ii) * esize;
    for (int b = 0; b < esize; b++) out[e * esize + b] = src[se + b];
}
// LayerNorm over the last dim: one block per row (the inference fused-
// layernorm pattern). bias may alias w when has_bias == 0.
extern "C" __global__ void native_layer_norm_f32(
    const float* x, const float* w, const float* bias, float* y,
    int cols, float eps, int has_bias) {
    extern __shared__ float smem[]; // [2 * blockDim] floats
    const float* xr = x + (long long)blockIdx.x * cols;
    float* yr = y + (long long)blockIdx.x * cols;
    float lsum = 0.0f, lsq = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = xr[j];
        lsum += v;
        lsq += v * v;
    }
    float* ssum = smem;
    float* ssq = smem + blockDim.x;
    ssum[threadIdx.x] = lsum;
    ssq[threadIdx.x] = lsq;
    NATIVE_BLOCK_REDUCE_PAIR(ssum, ssq)
    float mean = ssum[0] / cols;
    float var = ssq[0] / cols - mean * mean;
    float inv = rsqrtf(fmaxf(var, 0.0f) + eps);
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float r = (xr[j] - mean) * inv * w[j];
        if (has_bias) r += bias[j];
        yr[j] = r;
    }
}
// The f32 cat copy, and the same copy per element width - f16/bf16/u8/i64 cats
// previously host-bounced.
NATIVE_CAT_COPY(native_cat_copy_f32, float)
NATIVE_CAT_COPY(native_cat_copy_w8, unsigned char)
NATIVE_CAT_COPY(native_cat_copy_w16, unsigned short)
NATIVE_CAT_COPY(native_cat_copy_w64, unsigned long long)
// GroupNorm on [b, c, spatial]: one block per (batch, group) slab of
// cg = c/groups channels. Double accumulators so the single-pass
// E[x^2]-mean^2 variance stays accurate on multi-megabyte slabs.
extern "C" __global__ void native_group_norm_f32(
    const float* x, const float* w, const float* bias, float* y,
    int num_groups, int cg, int spatial, float eps) {
    int g = blockIdx.x % num_groups;
    long slab = (long)cg * spatial;
    long base = (long)blockIdx.x * slab;
    __shared__ double ssum[256];
    __shared__ double ssq[256];
    double lsum = 0.0, lsq = 0.0;
    for (long i = threadIdx.x; i < slab; i += blockDim.x) {
        float v = x[base + i];
        lsum += v;
        lsq += (double)v * v;
    }
    ssum[threadIdx.x] = lsum;
    ssq[threadIdx.x] = lsq;
    NATIVE_BLOCK_REDUCE_PAIR(ssum, ssq)
    float mean = (float)(ssum[0] / slab);
    float var = (float)(ssq[0] / slab) - mean * mean;
    float inv = rsqrtf(fmaxf(var, 0.0f) + eps);
    for (long i = threadIdx.x; i < slab; i += blockDim.x) {
        int ch = g * cg + (int)(i / spatial);
        y[base + i] = (x[base + i] - mean) * inv * w[ch] + bias[ch];
    }
}
"#;

// --- GPU op launchers - same kernels + launch configs as the live
// --- decode path's fused_kernels launchers, fed from native storage.
