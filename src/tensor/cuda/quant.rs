//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// F32 activations -> q8_1 blocks (row padded to MATRIX_ROW_PADDING=512),
/// the input format every mmvq GEMV kernel consumes.
pub fn quantize_q8_1(dev: &CudaDevice, x: &CudaSlice<f32>, k: usize) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows(dev, x, k, 1)
}

/// Multi-row variant: quantize `rows` consecutive activation rows of `k`
/// f32s each into per-row-padded q8_1 blocks (the 2-D grid the production
/// quantize kernel was built for - `blockIdx.y` indexes the row).
pub fn quantize_q8_1_rows(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_t(dev, x, "mmvq_gguf_quantize_q8_1_f32", k, rows)
}

/// F16-activation variant (the fork's `launch_mmvq_gguf_quantize_q8_1_f16`):
/// F16-carrier models (qwen3.5 DeltaNet projections et al) quantize without
/// an intermediate f32 cast, exactly like `fast_mmvq::try_fwd`.
pub fn quantize_q8_1_rows_f16(
    dev: &CudaDevice,
    x: &CudaSlice<half::f16>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_t(dev, x, "mmvq_gguf_quantize_q8_1_f16", k, rows)
}

/// BF16-activation variant (mirrors `launch_mmvq_gguf_quantize_q8_1_bf16`).
pub fn quantize_q8_1_rows_bf16(
    dev: &CudaDevice,
    x: &CudaSlice<half::bf16>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_t(dev, x, "mmvq_gguf_quantize_q8_1_bf16", k, rows)
}

/// View-input variants (zero-copy narrow views quantize straight from their
/// offset range in the source buffer - no materialization, perf).
pub fn quantize_q8_1_rows_f32_view(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<'_, f32>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_view_t(dev, x, "mmvq_gguf_quantize_q8_1_f32", k, rows)
}

pub fn quantize_q8_1_rows_f16_view(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<'_, half::f16>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_view_t(dev, x, "mmvq_gguf_quantize_q8_1_f16", k, rows)
}

pub fn quantize_q8_1_rows_bf16_view(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<'_, half::bf16>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_view_t(dev, x, "mmvq_gguf_quantize_q8_1_bf16", k, rows)
}

pub(super) fn quantize_q8_1_rows_view_t<T: cudarc::driver::DeviceRepr>(
    dev: &CudaDevice,
    x: &cudarc::driver::CudaView<'_, T>,
    kernel: &str,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    use crate::tensor::quantized::MATRIX_ROW_PADDING;
    const QUANT_BLOCK: usize = 256;
    let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    let bytes = rows * (k_padded / 32 * 36); // block_q8_1 = 32 i8 + 2 f16
    let func = dev.mmvq_fn(kernel)?;
    let stream = dev.stream();
    // plain (uninit) alloc: the quantize grid covers every padded block and
    // writes all 36 bytes of each (zero data in the pad lanes), so the
    // per-call memset was pure launch/API overhead on the decode path.
    let out = with_oom_retry(dev, "q8_1", || unsafe { stream.alloc::<u8>(bytes) })?;
    let cfg = LaunchConfig {
        grid_dim: (k_padded.div_ceil(QUANT_BLOCK) as u32, rows as u32, 1),
        block_dim: (QUANT_BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (k_i, kp_i) = (k as i32, k_padded as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&k_i);
    b.arg(&kp_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("q8_1 launch: {e}")))?;
    Ok(out)
}

pub(super) fn quantize_q8_1_rows_t<T: cudarc::driver::DeviceRepr>(
    dev: &CudaDevice,
    x: &CudaSlice<T>,
    kernel: &str,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    use crate::tensor::quantized::MATRIX_ROW_PADDING;
    const QUANT_BLOCK: usize = 256;
    let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    let bytes = rows * (k_padded / 32 * 36); // block_q8_1 = 32 i8 + 2 f16
    let func = dev.mmvq_fn(kernel)?;
    let stream = dev.stream();
    // plain (uninit) alloc: the quantize grid covers every padded block and
    // writes all 36 bytes of each (zero data in the pad lanes), so the
    // per-call memset was pure launch/API overhead on the decode path.
    let out = with_oom_retry(dev, "q8_1", || unsafe { stream.alloc::<u8>(bytes) })?;
    let cfg = LaunchConfig {
        grid_dim: (k_padded.div_ceil(QUANT_BLOCK) as u32, rows as u32, 1),
        block_dim: (QUANT_BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (k_i, kp_i) = (k as i32, k_padded as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&k_i);
    b.arg(&kp_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("q8_1 launch: {e}")))?;
    Ok(out)
}

/// Quantized GEMV: `[n, k] (GGML blocks) x [k] (q8_1) -> [n] f32` via the
/// production `mmvq_gguf_{tag}_f32_plain_cuda1` kernel (launch dims mirror
/// the live launcher: grid=(n,1,1), block=(32,4,1)).
pub fn mmvq_gemv_f32(
    dev: &CudaDevice,
    tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &CudaSlice<u8>,
    k: usize,
    n: usize,
) -> Result<CudaSlice<f32>> {
    mmvq_f32(dev, tag, weight_blob, q8_1, k, n, 1, false)
}

/// Batched MMVQ: `[n, k] (GGML blocks) x [b_size, k] (q8_1 rows) -> [b_size, n]`
/// f32 via the production `mmvq_gguf_{tag}_f32_plain_cuda{b}` kernels (
/// Launch geometry mirrors the fork's host launcher
/// `launch_mmvq_gguf_*_plain` exactly: rows_per_block = b<=1 ? 1 : 2,
/// nwarps = b<=4 ? 4 : 2 - so rows 2..=8 (spec-decode/PLD `forward_all`,
/// short prefills) take the SAME kernel family as the facade instead of MMQ.
pub fn mmvq_f32(
    dev: &CudaDevice,
    tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &CudaSlice<u8>,
    k: usize,
    n: usize,
    b_size: usize,
    smallk: bool,
) -> Result<CudaSlice<f32>> {
    mmvq_t::<f32>(dev, tag, "f32", weight_blob, q8_1, k, n, b_size, smallk)
}

/// Dequantize a whole K-quant weight blob -> a dense F32 device buffer, using the ggml
/// `dequantize_block_{q4_K,q5_K,q6_K}_f32` kernels (one CUDA block per 256-elem super-block).
/// `kernel` = the kernel name, `threads` = its block size (q4_K:32, q5_K/q6_K:64), `total` = n.k.
/// The result feeds a parallel cuBLAS GEMM (F32 accum) - the DiT's ~1e9 activations overflow the
/// quantized MMQ path, and MMVQ caps at 5 rows/launch, so this is the high-GPU-util correct path.
pub fn dequantize_kquant_f32(
    dev: &CudaDevice,
    kernel: &str,
    weight_blob: &CudaSlice<u8>,
    total: usize,
    threads: u32,
) -> Result<CudaSlice<f32>> {
    let func = dev.quantized_fn(kernel)?;
    let stream = dev.stream();
    let dst = with_oom_retry(dev, "dequant", || unsafe { stream.alloc::<f32>(total) })?;
    let nb = total / 256; // QK_K super-blocks
    let cfg = LaunchConfig {
        grid_dim: (nb as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&func);
    b.arg(weight_blob);
    b.arg(&dst);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("{kernel} launch: {e}")))?;
    Ok(dst)
}

/// Whole-weight Q8_0 -> F32 dequant on GPU. Q8_0 is NOT a 256-wide K-quant (its blocks are 32
/// wide), so it needs its own launcher: the kernel takes the block count `nb32 = total/32` and each
/// CUDA block (32 threads) dequantizes 256 elements (8 q8_0 blocks), so grid = total/256. Lets the
/// Q8_0 projections (whose in-dim is a multiple of 32 but not the K-quant 256) run the same
/// GPU-dequant + BF16-cuBLAS GEMM path as the K-quants instead of a silent CPU fallback.
pub fn dequantize_q8_0_f32(
    dev: &CudaDevice,
    weight_blob: &CudaSlice<u8>,
    total: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.quantized_fn("dequantize_block_q8_0_f32")?;
    let stream = dev.stream();
    let dst = with_oom_retry(dev, "dequant q8_0", || unsafe {
        stream.alloc::<f32>(total)
    })?;
    let nb32 = (total / 32) as i32;
    let grid = total.div_ceil(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&func);
    b.arg(weight_blob);
    b.arg(&dst);
    b.arg(&nb32);
    unsafe { b.launch(cfg) }
        .map_err(|e| Error(format!("dequantize_block_q8_0_f32 launch: {e}")))?;
    Ok(dst)
}

/// F16-output batched MMVQ (the fork's `plain_launcher_f16` family).
/// `smallk` selects the decode-only 4-rows/block `plain_smallk_cuda1`
/// kernel - caller must ensure `b_size == 1 && n % 4 == 0` and apply the
/// ggml dispatch threshold `(k/256) < 4*vdr` (`quantized/fast_mmvq.rs`).
pub fn mmvq_f16(
    dev: &CudaDevice,
    tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &CudaSlice<u8>,
    k: usize,
    n: usize,
    b_size: usize,
    smallk: bool,
) -> Result<CudaSlice<half::f16>> {
    mmvq_t::<half::f16>(dev, tag, "f16", weight_blob, q8_1, k, n, b_size, smallk)
}

/// BF16-output batched MMVQ (no smallk - the fork only wires smallk for f16).
pub fn mmvq_bf16(
    dev: &CudaDevice,
    tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &CudaSlice<u8>,
    k: usize,
    n: usize,
    b_size: usize,
) -> Result<CudaSlice<half::bf16>> {
    mmvq_t::<half::bf16>(dev, tag, "bf16", weight_blob, q8_1, k, n, b_size, false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mmvq_t<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    dev: &CudaDevice,
    tag: &str,
    dst_tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &CudaSlice<u8>,
    k: usize,
    n: usize,
    b_size: usize,
    smallk: bool,
) -> Result<CudaSlice<T>> {
    use crate::tensor::quantized::MATRIX_ROW_PADDING;
    if b_size == 0 || b_size > 8 {
        return Err(Error(format!("mmvq: b_size {b_size} not in 1..=8")));
    }
    if smallk && (b_size != 1 || n % 4 != 0) {
        return Err(Error(format!(
            "mmvq smallk requires b_size==1 && n%4==0 (got b_size={b_size}, n={n})"
        )));
    }
    let kernel = if smallk {
        format!("mmvq_gguf_{tag}_{dst_tag}_plain_smallk_cuda1")
    } else {
        format!("mmvq_gguf_{tag}_{dst_tag}_plain_cuda{b_size}")
    };
    let func = dev.mmvq_fn(&kernel)?;
    let stream = dev.stream();
    let dst = with_oom_retry(dev, "gemv", || unsafe { stream.alloc::<T>(n * b_size) })?;
    let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    let stride_col_y = (k_padded / 32) as i32;
    let stride_col_dst = n as i32;
    // Launch geometry mirrors the fork's C launchers exactly:
    // plain: rows_per_block = b<=1 ? 1 : 2, nwarps = b<=4 ? 4 : 2;
    // smallk (MMVQ_LAUNCHER_SMALLK_PLAIN): rows_per_block = 4, nwarps = 4.
    let rows_per_block: usize = if smallk {
        4
    } else if b_size <= 1 {
        1
    } else {
        2
    };
    let nwarps: u32 = if smallk || b_size <= 4 { 4 } else { 2 };
    let grid_x = if smallk {
        n / rows_per_block
    } else {
        n.div_ceil(rows_per_block)
    };
    let cfg = LaunchConfig {
        grid_dim: (grid_x as u32, 1, 1),
        block_dim: (32, nwarps, 1),
        shared_mem_bytes: 0,
    };
    let (k_i, n_i) = (k as i32, n as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(weight_blob);
    b.arg(q8_1);
    b.arg(&dst);
    b.arg(&k_i);
    b.arg(&n_i);
    b.arg(&stride_col_y);
    b.arg(&stride_col_dst);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("{kernel} launch: {e}")))?;
    Ok(dst)
}

/// Device-side dtype casts (no host roundtrip).
pub fn cast_f32_to_f16(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<half::f16>> {
    let func = dev.elementwise_fn("native_cast_f32_f16")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "cast", || unsafe { stream.alloc::<half::f16>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("cast launch: {e}")))?;
    Ok(out)
}

pub fn cast_f16_to_f32(
    dev: &CudaDevice,
    x: &CudaSlice<half::f16>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn("native_cast_f16_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "cast", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("cast launch: {e}")))?;
    Ok(out)
}

/// What resolving a kernel by name costs, per launch.
///
/// The mat-vec launcher builds its symbol with `format!` and then asks the module for it on
/// every call. A decode step launches it once per projection per layer - of the order of two
/// hundred times per token - so if that pair costs microseconds it is microseconds the card
/// spends idle. Measured rather than argued.
#[cfg(test)]
mod launch_lookup_cost {
    #[test]
    #[ignore = "measurement: needs a CUDA device"]
    fn what_resolving_a_kernel_by_name_costs() {
        use crate::tensor::cuda::CudaDevice;
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device");
            return;
        };
        // Warm the module so this measures the lookup, not the PTX load.
        let _ = dev.mmvq_fn("mmvq_gguf_q4_0_f32_plain_cuda1").unwrap();

        const N: usize = 20_000;
        let start = std::time::Instant::now();
        for i in 0..N {
            let b = 1 + (i % 8);
            let name = format!("mmvq_gguf_q4_0_f32_plain_cuda{b}");
            let f = dev.mmvq_fn(&name).unwrap();
            std::hint::black_box(&f);
        }
        let per = start.elapsed().as_secs_f64() / N as f64;
        eprintln!(
            "format! + load_function: {:.2} us per launch, {:.2} ms per 224 launches (one token \
             through a 32-layer model)",
            per * 1e6,
            per * 224.0 * 1e3
        );
    }
}

/// The five 32-value dequantisers, judged against the codec that wrote the bytes.
#[cfg(test)]
mod block_dequant_parity {
    use super::*;
    use crate::tensor::quantized::GgmlDType;

    /// Each of these kernels turns a stored block back into floats, and the host codec that
    /// produced the block turns it back independently. They read the SAME bytes, so the
    /// quantisation error is common to both and drops out: what is left is one multiply-add
    /// per value, done twice.
    ///
    /// The two do not associate it the same way - the device folds the offset into a constant
    /// and adds it, the host subtracts the midpoint before scaling - so this is a tolerance and
    /// not an equality. It is sized to the block's own scale: a single misread bit moves a
    /// value by a whole `d`, four orders of magnitude above what is allowed here.
    #[test]
    fn the_gpu_block_dequantisers_agree_with_the_host_codec() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the block dequantisers are NOT covered by this run");
            return;
        };

        // Eight blocks per CUDA block of 32 threads, so a whole number of those.
        const TOTAL: usize = 2048;
        let xs: Vec<f32> = (0..TOTAL)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(374_761_393)
                    .rotate_left(11);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * 3.1
            })
            .collect();

        for (dtype, kernel) in [
            (GgmlDType::Q4_0, "dequantize_block_q4_0_f32"),
            (GgmlDType::Q4_1, "dequantize_block_q4_1_f32"),
            (GgmlDType::Q5_0, "dequantize_block_q5_0_f32"),
            (GgmlDType::Q5_1, "dequantize_block_q5_1_f32"),
            (GgmlDType::Q8_0, "dequantize_block_q8_0_f32"),
        ] {
            let bytes = crate::tensor::quant_cpu::from_float_bytes(dtype, &xs)
                .unwrap_or_else(|e| panic!("{dtype:?}: encode: {e}"));
            let mut want = vec![0f32; TOTAL];
            crate::tensor::quant_cpu::to_float_bytes(dtype, &bytes, &mut want)
                .unwrap_or_else(|e| panic!("{dtype:?}: host decode: {e}"));

            let stream = dev.stream();
            let blob = stream.clone_htod(&bytes).expect("upload");
            let dst = unsafe { stream.alloc::<f32>(TOTAL) }.expect("alloc");
            let func = dev
                .quantized_fn(kernel)
                .unwrap_or_else(|e| panic!("{kernel} does not resolve: {e}"));
            let nb32 = (TOTAL / 32) as i32;
            let cfg = LaunchConfig {
                grid_dim: (TOTAL as u32 / 256, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&func);
            b.arg(&blob);
            b.arg(&dst);
            b.arg(&nb32);
            unsafe { b.launch(cfg) }.unwrap_or_else(|e| panic!("{kernel} launch: {e}"));
            let got = stream.clone_dtoh(&dst).expect("download");

            // The largest step this format can take between two stored values.
            let step = want.iter().fold(0f32, |m, v| m.max(v.abs())) / 8.0;
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-4 * step,
                "{dtype:?}: the device and the host disagree by {worst:e}, a step being {step:e}"
            );

            // The negative control: one nibble of one block, and the check must break.
            let mut off = bytes.clone();
            *off.last_mut().unwrap() ^= 0x11;
            let mut other = vec![0f32; TOTAL];
            crate::tensor::quant_cpu::to_float_bytes(dtype, &off, &mut other)
                .unwrap_or_else(|e| panic!("{dtype:?}: host decode: {e}"));
            let moved = other
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                moved > 1e-4 * step,
                "{dtype:?}: changing a stored value moved nothing, so this check proves nothing"
            );
        }

        // The superblock formats, through the launcher production uses.
        //
        // These carry a different bound because they carry a different arithmetic: the kernels
        // multiply and subtract in HALF while the host codec works in f32, so the two cannot
        // agree past what eleven mantissa bits allow. The bound is derived and not fitted  -
        // four roundings (`dall*sc`, `dmin*m`, the product, the subtraction), each worth at
        // most half an ULP of an operand no larger than the block's range. A misread bit
        // position moves a value by a whole scale, thousands of ULPs, so this still says no.
        const HALF_STEP: f32 = 4.884e-4; // 2^-11
        for (dtype, kernel, threads) in [
            (GgmlDType::Q2K, "dequantize_block_q2_K_f32", 64u32),
            (GgmlDType::Q3K, "dequantize_block_q3_K_f32", 64),
            (GgmlDType::Q4K, "dequantize_block_q4_K_f32", 32),
            (GgmlDType::Q5K, "dequantize_block_q5_K_f32", 64),
            (GgmlDType::Q6K, "dequantize_block_q6_K_f32", 64),
            // Eight values a thread, so a superblock is one warp.
            (GgmlDType::Q8K, "dequantize_block_q8_K_f32", 32),
        ] {
            let bytes = crate::tensor::quant_cpu::from_float_bytes(dtype, &xs)
                .unwrap_or_else(|e| panic!("{dtype:?}: encode: {e}"));
            let mut want = vec![0f32; TOTAL];
            crate::tensor::quant_cpu::to_float_bytes(dtype, &bytes, &mut want)
                .unwrap_or_else(|e| panic!("{dtype:?}: host decode: {e}"));

            let blob = dev.stream().clone_htod(&bytes).expect("upload");
            let dst = dequantize_kquant_f32(&dev, kernel, &blob, TOTAL, threads)
                .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            let got = dev.stream().clone_dtoh(&dst).expect("download");

            let range = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let allowed = 4.0 * HALF_STEP * range;
            let (worst, at) =
                got.iter()
                    .zip(&want)
                    .enumerate()
                    .fold((0f32, 0usize), |(w, at), (i, (g, k))| {
                        let e = (g - k).abs();
                        if e > w {
                            (e, i)
                        } else {
                            (w, at)
                        }
                    });
            assert!(
                worst <= allowed,
                "{dtype:?}: worst disagreement {worst:e} at [{at}] (superblock {}, position {}), \
                 device {} against the host's {}; half precision allows {allowed:e} over a range \
                 of {range:e}",
                at / 256,
                at % 256,
                got[at],
                want[at]
            );
        }
    }

    /// Every output position a superblock kernel is launched for must be written.
    ///
    /// These kernels take one superblock per call and do not read `blockIdx`, while the
    /// launcher gives them one CUDA block per superblock. Read literally that would have every
    /// block write the first 256 outputs and leave the rest untouched. The buffer is filled
    /// with a value the arithmetic cannot produce, and anything still holding it afterwards is
    /// a position nobody wrote.
    #[test]
    fn a_superblock_dequantiser_writes_every_position_it_was_launched_for() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the superblock dequantisers are NOT covered by this run");
            return;
        };
        const TOTAL: usize = 2048;
        const UNWRITTEN: f32 = -123_456.0;
        let xs: Vec<f32> = (0..TOTAL).map(|i| ((i % 17) as f32) * 0.11 - 0.9).collect();

        for (dtype, kernel, threads) in [
            (GgmlDType::Q2K, "dequantize_block_q2_K_f32", 64u32),
            (GgmlDType::Q3K, "dequantize_block_q3_K_f32", 64),
            (GgmlDType::Q4K, "dequantize_block_q4_K_f32", 32),
            (GgmlDType::Q5K, "dequantize_block_q5_K_f32", 64),
            (GgmlDType::Q6K, "dequantize_block_q6_K_f32", 64),
            // Eight values a thread, so a superblock is one warp.
            (GgmlDType::Q8K, "dequantize_block_q8_K_f32", 32),
        ] {
            let bytes = crate::tensor::quant_cpu::from_float_bytes(dtype, &xs)
                .unwrap_or_else(|e| panic!("{dtype:?}: encode: {e}"));
            let stream = dev.stream();
            let blob = stream.clone_htod(&bytes).expect("upload");
            let dst = stream
                .clone_htod(&vec![UNWRITTEN; TOTAL])
                .expect("prefill the output");
            let func = dev
                .quantized_fn(kernel)
                .unwrap_or_else(|e| panic!("{kernel} does not resolve: {e}"));
            let cfg = LaunchConfig {
                grid_dim: ((TOTAL / 256) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = stream.launch_builder(&func);
            b.arg(&blob);
            b.arg(&dst);
            unsafe { b.launch(cfg) }.unwrap_or_else(|e| panic!("{kernel} launch: {e}"));
            let got = stream.clone_dtoh(&dst).expect("download");

            let untouched: Vec<usize> = got
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == UNWRITTEN)
                .map(|(i, _)| i)
                .collect();
            assert!(
                untouched.is_empty(),
                "{dtype:?}: {} of {TOTAL} positions were never written, first at [{}] \
                 (superblock {}, position {})",
                untouched.len(),
                untouched[0],
                untouched[0] / 256,
                untouched[0] % 256
            );
        }
    }
}
