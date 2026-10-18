//! The tiled quantised matmul: its workspace, the dtypes it serves, and its launchers.
//!
//! MMQ quantises the activation to q8_1 into a scratch buffer, so unlike the mat-vec path it
//! needs a per-call workspace - which is why its caller must be able to do without it.

use super::*;

#[cfg(feature = "gpu")]
#[derive(Default)]
pub(super) struct MmqWorkspace {
    pub(super) main: Option<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>,
    pub(super) fixup: Option<crate::tensor::gpu_runtime::driver::CudaSlice<u8>>,
    /// Marks the last work enqueued against these buffers, so the next taker can queue
    /// behind it. There is ONE workspace per device and any number of streams on a device  -
    /// an LLM and an image engine sharing a card, two requests in flight - and the mutex
    /// below only orders the *host*, which stops mattering the moment a launch returns.
    pub(super) last_use: Option<crate::tensor::gpu_runtime::driver::CudaEvent>,
}

#[cfg(feature = "gpu")]
pub(super) static MMQ_WORKSPACES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, MmqWorkspace>>,
> = std::sync::OnceLock::new();

/// Drop the per-device MMQ scratch workspaces (model unload - returns the
/// reserved VRAM; re-created lazily on the next batched matmul).
#[cfg(feature = "gpu")]
pub fn release_mmq_workspaces() {
    // try_lock, NOT lock: the emergency-reclaim path runs INSIDE an OOM'd
    // `mmq_forward`, which holds this mutex across its workspace use (the
    // borrowed slices must outlive the kernel launch). A blocking lock there
    // self-deadlocks the forward - and froze the whole server for hours when
    // a Flux generation OOM'd mid-MMQ (every later unload queued behind it).
    // Skipping under contention is also the only CORRECT behaviour: the
    // holder is actively using its workspace, and dropping it would free
    // device memory a queued kernel still addresses.
    if let Some(m) = MMQ_WORKSPACES.get() {
        if let Ok(mut g) = m.try_lock() {
            g.clear();
        }
    }
}

/// The activation quantisers, one row per scale layout a tile kernel can read.
///
/// They differ in what they write into the scratch and not at all in how they are called, so
/// the call is stated once - here, next to the pointer type the dispatch below holds them in,
/// which is the other place the same thirteen arguments used to be spelled out.
macro_rules! mmq_quantizers {
    ($($(#[$note:meta])* $name:ident;)+) => {
        #[cfg(feature = "gpu")]
        pub(crate) type MmqQuantizeLauncher = unsafe extern "C" fn(
            x: *const std::ffi::c_void,
            ids: *const i32,
            vy: *mut std::ffi::c_void,
            type_x: i32,
            ne00: i64,
            s01: i64,
            s02: i64,
            s03: i64,
            ne0: i64,
            ne1: i64,
            ne2: i64,
            ne3: i64,
            stream: *mut std::ffi::c_void,
        );

        #[cfg(feature = "gpu")]
        extern "C" {
            $(
                $(#[$note])*
                fn $name(
                    x: *const std::ffi::c_void,
                    ids: *const i32,
                    vy: *mut std::ffi::c_void,
                    type_x: i32,
                    ne00: i64,
                    s01: i64,
                    s02: i64,
                    s03: i64,
                    ne0: i64,
                    ne1: i64,
                    ne2: i64,
                    ne3: i64,
                    stream: *mut std::ffi::c_void,
                );
            )+
        }
    };
}

mmq_quantizers! {
    /// One scale per 32 values.
    launch_mmq_quantize_q8_1_D4;
    /// A scale and a sum per 32 values, for weights whose blocks carry an offset.
    launch_mmq_quantize_q8_1_DS4;
    /// Two scales and six partial sums, for a superblock with a minimum per sub-block.
    launch_mmq_quantize_q8_1_D2S6;
}

/// Everything this module knows about a weight format, from one row per format.
///
/// The four things the caller asks - is it served, how long is its block, which activation
/// quantiser its tile kernel reads, and which tile kernel that is - used to be four lists over
/// the same ten formats, and a format could be compiled into the kernel crate and missing from
/// one of them. Here a format is a row or it is nothing: the `extern` declaration and the four
/// answers are all generated from it.
macro_rules! mmq_formats {
    ($($dtype:ident => $qk:expr, $quantize:ident, $matmul:ident, $moe:ident;)+) => {
        /// The pointer type the dispatch below holds a tile kernel in - the same call the
        /// declarations under it make, said once.
        #[cfg(feature = "gpu")]
        type MmqMatmulLauncher = unsafe extern "C" fn(
            tmp_fixup: *mut std::ffi::c_void,
            x: *const std::ffi::c_void,
            y: *const std::ffi::c_void,
            dst: *mut std::ffi::c_void,
            ncols_x: i64,
            nrows_x: i64,
            ncols_y: i64,
            stride_row_x: i64,
            stride_col_dst: i64,
            cc: i32,
            nsm: i32,
            smpbo: i64,
            warp_size: i32,
            stream: *mut std::ffi::c_void,
        );

        /// The pointer type the expert-batched tile kernels are held in.
        #[cfg(feature = "gpu")]
        pub(crate) type MmqMoeLauncher = unsafe extern "C" fn(
            x: *const std::ffi::c_void,
            x_second: *const std::ffi::c_void,
            dst_second: i64,
            y: *const std::ffi::c_void,
            dst: *mut std::ffi::c_void,
            expert_bounds: *const i32,
            ids_dst: *const i32,
            stride_expert_x: i64,
            ncols_x: i64,
            nrows_x: i64,
            ncols_y: i64,
            stride_row_x: i64,
            stride_col_dst: i64,
            n_experts: i32,
            ncols_max: i64,
            ncols_hint: i64,
            mmq_x: i32,
            cc: i32,
            smpbo: i64,
            warp_size: i32,
            stream: *mut std::ffi::c_void,
        ) -> i32;

        #[cfg(feature = "gpu")]
        extern "C" {
            $(
                fn $moe(
                    x: *const std::ffi::c_void,
                    x_second: *const std::ffi::c_void,
                    dst_second: i64,
                    y: *const std::ffi::c_void,
                    dst: *mut std::ffi::c_void,
                    expert_bounds: *const i32,
                    ids_dst: *const i32,
                    stride_expert_x: i64,
                    ncols_x: i64,
                    nrows_x: i64,
                    ncols_y: i64,
                    stride_row_x: i64,
                    stride_col_dst: i64,
                    n_experts: i32,
                    ncols_max: i64,
                    ncols_hint: i64,
                    mmq_x: i32,
                    cc: i32,
                    smpbo: i64,
                    warp_size: i32,
                    stream: *mut std::ffi::c_void,
                ) -> i32;
            )+
        }

        #[cfg(feature = "gpu")]
        extern "C" {
            $(
                fn $matmul(
                    tmp_fixup: *mut std::ffi::c_void,
                    x: *const std::ffi::c_void,
                    y: *const std::ffi::c_void,
                    dst: *mut std::ffi::c_void,
                    ncols_x: i64,
                    nrows_x: i64,
                    ncols_y: i64,
                    stride_row_x: i64,
                    stride_col_dst: i64,
                    cc: i32,
                    nsm: i32,
                    smpbo: i64,
                    warp_size: i32,
                    stream: *mut std::ffi::c_void,
                );
            )+
        }

        /// Whether a tile kernel exists for this weight format.
        #[cfg(feature = "gpu")]
        pub(crate) fn mmq_supports(dtype: GgmlDType) -> bool {
            matches!(dtype, $(GgmlDType::$dtype)|+)
        }

        /// Values per block - what the activation must be quantised in multiples of.
        #[cfg(feature = "gpu")]
        pub(crate) fn mmq_qk(dtype: GgmlDType) -> usize {
            match dtype {
                $(GgmlDType::$dtype => $qk,)+
                _ => 256,
            }
        }

        /// The activation quantiser whose scale layout this format's tile kernel reads.
        #[cfg(feature = "gpu")]
        pub(crate) fn mmq_quantize_launcher(dtype: GgmlDType) -> Result<MmqQuantizeLauncher> {
            Ok(match dtype {
                $(GgmlDType::$dtype => $quantize,)+
                other => return Err(Error(format!("mmq: unsupported dtype {other:?}"))),
            })
        }

        /// The expert-batched tile kernel of a format in the table.
        #[cfg(feature = "gpu")]
        fn mmq_moe_launcher_listed(dtype: GgmlDType) -> Result<MmqMoeLauncher> {
            Ok(match dtype {
                $(GgmlDType::$dtype => $moe,)+
                other => return Err(Error(format!("mmq: unsupported dtype {other:?}"))),
            })
        }

        /// The tile kernel itself.
        #[cfg(feature = "gpu")]
        pub(crate) fn mmq_launcher(dtype: GgmlDType) -> Result<MmqMatmulLauncher> {
            Ok(match dtype {
                $(GgmlDType::$dtype => $matmul,)+
                other => return Err(Error(format!("mmq: unsupported dtype {other:?}"))),
            })
        }
    };
}

mmq_formats! {
    //         block  activation quantiser           tile kernel            expert-batched
    Q4_0  =>  32, launch_mmq_quantize_q8_1_DS4,   launch_mmq_gguf_q4_0,  launch_mmq_moe_gguf_q4_0;
    Q4_1  =>  32, launch_mmq_quantize_q8_1_DS4,   launch_mmq_gguf_q4_1,  launch_mmq_moe_gguf_q4_1;
    Q5_0  =>  32, launch_mmq_quantize_q8_1_D4,    launch_mmq_gguf_q5_0,  launch_mmq_moe_gguf_q5_0;
    Q5_1  =>  32, launch_mmq_quantize_q8_1_DS4,   launch_mmq_gguf_q5_1,  launch_mmq_moe_gguf_q5_1;
    Q8_0  =>  32, launch_mmq_quantize_q8_1_D4,    launch_mmq_gguf_q8_0,  launch_mmq_moe_gguf_q8_0;
    Q2K   => 256, launch_mmq_quantize_q8_1_D2S6,  launch_mmq_gguf_q2_k,  launch_mmq_moe_gguf_q2_k;
    Q3K   => 256, launch_mmq_quantize_q8_1_D4,    launch_mmq_gguf_q3_k,  launch_mmq_moe_gguf_q3_k;
    Q4K   => 256, launch_mmq_quantize_q8_1_DS4,   launch_mmq_gguf_q4_k,  launch_mmq_moe_gguf_q4_k;
    Q5K   => 256, launch_mmq_quantize_q8_1_DS4,   launch_mmq_gguf_q5_k,  launch_mmq_moe_gguf_q5_k;
    Q6K   => 256, launch_mmq_quantize_q8_1_D4,    launch_mmq_gguf_q6_k,  launch_mmq_moe_gguf_q6_k;
}

/// The formats only a mixture stores, which have no dense entry point: each is declared once
/// here, with the expert-batched launcher built for it and the block it quantises its
/// activation by.
#[cfg(feature = "gpu")]
macro_rules! expert_only_formats {
    ($($dtype:ident => $launcher:ident;)+) => {
        #[cfg(feature = "gpu")]
        extern "C" {
            $(fn $launcher(
                x: *const std::ffi::c_void,
                x_second: *const std::ffi::c_void,
                dst_second: i64,
                y: *const std::ffi::c_void,
                dst: *mut std::ffi::c_void,
                expert_bounds: *const i32,
                ids_dst: *const i32,
                stride_expert_x: i64,
                ncols_x: i64,
                nrows_x: i64,
                ncols_y: i64,
                stride_row_x: i64,
                stride_col_dst: i64,
                n_experts: i32,
                ncols_max: i64,
                ncols_hint: i64,
                mmq_x: i32,
                cc: i32,
                smpbo: i64,
                warp_size: i32,
                stream: *mut std::ffi::c_void,
            ) -> i32;)+
        }

        /// The expert-batched launcher of a format only mixtures store.
        #[cfg(feature = "gpu")]
        fn expert_only_launcher(dtype: GgmlDType) -> Option<MmqMoeLauncher> {
            match dtype {
                $(GgmlDType::$dtype => Some($launcher),)+
                _ => None,
            }
        }
    };
}

#[cfg(feature = "gpu")]
expert_only_formats! {
    MxFp4  => launch_mmq_moe_gguf_mxfp4;
    Iq4Nl  => launch_mmq_moe_gguf_iq4_nl;
    Iq4Xs  => launch_mmq_moe_gguf_iq4_xs;
    Iq3Xxs => launch_mmq_moe_gguf_iq3_xxs;
    Iq3S   => launch_mmq_moe_gguf_iq3_s;
    Iq2S   => launch_mmq_moe_gguf_iq2_s;
    Q2_0   => launch_mmq_moe_gguf_q2_0;
}

/// Whether an expert-batched tile kernel exists for this weight format: every format the
/// table above serves, and the formats only mixtures store, which have no dense entry point.
#[cfg(feature = "gpu")]
pub(crate) fn mmq_moe_supports(dtype: GgmlDType) -> bool {
    mmq_supports(dtype) || expert_only_launcher(dtype).is_some()
}

/// Values per block of an expert stack of `dtype`.
#[cfg(feature = "gpu")]
pub(crate) fn mmq_moe_qk(dtype: GgmlDType) -> usize {
    if expert_only_launcher(dtype).is_some() {
        dtype.block_size()
    } else {
        mmq_qk(dtype)
    }
}

/// The activation quantiser an expert stack of `dtype` reads: the formats only mixtures store
/// are symmetric, one scale per 32 values.
#[cfg(feature = "gpu")]
pub(crate) fn mmq_moe_quantize_launcher(dtype: GgmlDType) -> Result<MmqQuantizeLauncher> {
    if expert_only_launcher(dtype).is_some() {
        Ok(launch_mmq_quantize_q8_1_D4)
    } else {
        mmq_quantize_launcher(dtype)
    }
}

/// The expert-batched tile kernel: every expert of a stack against its own sorted run of
/// activation columns in one launch.
#[cfg(feature = "gpu")]
pub(crate) fn mmq_moe_launcher(dtype: GgmlDType) -> Result<MmqMoeLauncher> {
    match expert_only_launcher(dtype) {
        Some(launcher) => Ok(launcher),
        None => mmq_moe_launcher_listed(dtype),
    }
}
