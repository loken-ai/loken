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

/// Quantise an f32 buffer to Q8_0 GGUF blocks on the GPU, returning the block bytes.
///
/// `f32s.len()` must be a multiple of 32. The output is the exact GGUF Q8_0 layout (34 bytes per
/// block: an f16 scale then 32 signed bytes), so the bytes drop straight into a GGUF writer. One
/// warp per block. Values match the CPU quantiser to f32 rounding, and the kernel is deterministic
/// so the same input yields the same bytes every run - what the requant digest needs.
pub fn gpu_quantize_q8_0(dev: &CudaDevice, f32s: &[f32]) -> Result<Vec<u8>> {
    debug_assert_eq!(
        f32s.len() % 32,
        0,
        "q8_0 quantize: {} not a multiple of 32",
        f32s.len()
    );
    let nblocks = f32s.len() / 32;
    let bytes = nblocks * 34; // sizeof(block_q8_0) = f16 + 32 i8
    let stream = dev.stream();
    let x = stream
        .clone_htod(f32s)
        .map_err(|e| Error(format!("q8_0 upload: {e}")))?;
    let out = with_oom_retry(dev, "requant_q8_0", || unsafe { stream.alloc::<u8>(bytes) })?;
    let func = dev.quantized_fn("requant_q8_0")?;
    let cfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = nblocks as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(&x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("requant_q8_0 launch: {e}")))?;
    stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("q8_0 download: {e}")))
}

/// Quantise an f32 buffer to Q4_K GGUF blocks on the GPU, returning the block bytes.
///
/// `f32s.len()` must be a multiple of 256 (the K-quant super-block). Exact GGUF Q4_K layout (144
/// bytes per block), one super-block per CUDA block: each thread fits one 32-value sub-block by the
/// same weighted grid search as the CPU encoder, the sixteen sub-block scales are quantised to six
/// bits, and every value is levelled against the rounded scale. Quality-equivalent to the CPU
/// quantiser and deterministic, so the digest stays stable.
pub fn gpu_quantize_q4_k(dev: &CudaDevice, f32s: &[f32]) -> Result<Vec<u8>> {
    debug_assert_eq!(
        f32s.len() % 256,
        0,
        "q4_K quantize: {} not a multiple of 256",
        f32s.len()
    );
    let nblocks = f32s.len() / 256;
    let bytes = nblocks * 144; // sizeof(block_q4_K) = half2 + 12 + 128
    let stream = dev.stream();
    let x = stream
        .clone_htod(f32s)
        .map_err(|e| Error(format!("q4_K upload: {e}")))?;
    let out = with_oom_retry(dev, "requant_q4_k", || unsafe { stream.alloc::<u8>(bytes) })?;
    let func = dev.quantized_fn("requant_q4_k")?;
    let cfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = nblocks as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(&x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("requant_q4_k launch: {e}")))?;
    stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("q4_K download: {e}")))
}

/// Quantise an f32 buffer to Q3_K GGUF blocks on the GPU, returning the block bytes.
///
/// `f32s.len()` must be a multiple of 256. Exact GGUF Q3_K layout (110 bytes per block), one
/// super-block per CUDA block: each thread fits one 16-value sub-block by the signed grid search
/// and refinement of the CPU encoder, the sixteen scales are quantised to six signed bits, and
/// every value's three-bit code is split into hmask and qs. Quality-equivalent, deterministic.
pub fn gpu_quantize_q3_k(dev: &CudaDevice, f32s: &[f32]) -> Result<Vec<u8>> {
    debug_assert_eq!(
        f32s.len() % 256,
        0,
        "q3_K quantize: {} not a multiple of 256",
        f32s.len()
    );
    let nblocks = f32s.len() / 256;
    let bytes = nblocks * 110; // sizeof(block_q3_K) = 32 + 64 + 12 + 2
    let stream = dev.stream();
    let x = stream
        .clone_htod(f32s)
        .map_err(|e| Error(format!("q3_K upload: {e}")))?;
    let out = with_oom_retry(dev, "requant_q3_k", || unsafe { stream.alloc::<u8>(bytes) })?;
    let func = dev.quantized_fn("requant_q3_k")?;
    let cfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = nblocks as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(&x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("requant_q3_k launch: {e}")))?;
    stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("q3_K download: {e}")))
}

/// Quantise an f32 buffer to Q2_K GGUF blocks on the GPU, returning the block bytes.
///
/// `f32s.len()` must be a multiple of 256. Exact GGUF Q2_K layout (84 bytes per block), one
/// super-block per CUDA block: sixteen 16-value sub-blocks fitted by the same unsigned grid search
/// as the CPU encoder at nmax 3, scales and mins quantised to four bits, codes two bits packed four
/// to a byte. Quality-equivalent, deterministic.
pub fn gpu_quantize_q2_k(dev: &CudaDevice, f32s: &[f32]) -> Result<Vec<u8>> {
    debug_assert_eq!(
        f32s.len() % 256,
        0,
        "q2_K quantize: {} not a multiple of 256",
        f32s.len()
    );
    let nblocks = f32s.len() / 256;
    let bytes = nblocks * 84; // sizeof(block_q2_K) = 16 + 64 + 2 + 2
    let stream = dev.stream();
    let x = stream
        .clone_htod(f32s)
        .map_err(|e| Error(format!("q2_K upload: {e}")))?;
    let out = with_oom_retry(dev, "requant_q2_k", || unsafe { stream.alloc::<u8>(bytes) })?;
    let func = dev.quantized_fn("requant_q2_k")?;
    let cfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = nblocks as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(&x);
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("requant_q2_k launch: {e}")))?;
    stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("q2_K download: {e}")))
}

/// `x` [n, inp] through a block-scaled fp4 weight [out, inp] - `nibbles` and ue8m0 `scales` as the
/// released checkpoint stores them - dequantised on the device and applied at full precision.
/// Returns [n, out], row-major.
pub fn gpu_fp4_linear(
    dev: &CudaDevice,
    nibbles: &[u8],
    scales: &[u8],
    (out, inp): (usize, usize),
    x: &[f32],
) -> Result<Vec<f32>> {
    if inp == 0
        || nibbles.len() != out * inp / 2
        || scales.is_empty()
        || scales.len() % out != 0
        || x.len() % inp != 0
    {
        return Err(Error(format!(
            "fp4 linear: {} nibble bytes, {} scales, {} inputs for [{out}, {inp}]",
            nibbles.len(),
            scales.len(),
            x.len()
        )));
    }
    let block = inp / (scales.len() / out);
    let n = x.len() / inp;
    let total = out * inp;
    let stream = dev.stream();
    let nib = stream
        .clone_htod(nibbles)
        .map_err(|e| alloc_err("fp4 upload", e))?;
    let sc = stream
        .clone_htod(scales)
        .map_err(|e| alloc_err("fp4 upload", e))?;
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("fp4 upload", e))?;
    let w = with_oom_retry(dev, "fp4 dequant", || unsafe { stream.alloc::<f32>(total) })?;
    let func = dev.quantized_fn("dequant_fp4_e8m0")?;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (inp_i, block_i, total_i) = (inp as i64, block as i64, total as i64);
    let mut b = stream.launch_builder(&func);
    b.arg(&nib);
    b.arg(&sc);
    b.arg(&w);
    b.arg(&inp_i);
    b.arg(&block_i);
    b.arg(&total_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("dequant_fp4_e8m0 launch: {e}")))?;
    let y = matmul_nt_f32(dev, &xs, &w, 1, n, inp, out)?;
    stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("fp4 linear download: {e}")))
}

/// `x` [n, inp] through rows `row0 .. row0 + out` of a block-scaled fp8 weight `inp` wide: `bytes` the
/// e4m3fn values of those rows, `scales` the whole weight's ue8m0 tile scales, `block` by `block`.
/// Dequantised on the device and applied at full precision; returns [n, out], row-major.
pub fn gpu_fp8_linear(
    dev: &CudaDevice,
    bytes: &[u8],
    scales: &[u8],
    (out, inp, block, row0): (usize, usize, usize, usize),
    x: &[f32],
) -> Result<Vec<f32>> {
    if inp == 0 || block == 0 || bytes.len() != out * inp || x.len() % inp != 0 {
        return Err(Error(format!(
            "fp8 linear: {} bytes, {} inputs for [{out}, {inp}]",
            bytes.len(),
            x.len()
        )));
    }
    let n = x.len() / inp;
    let total = out * inp;
    let stream = dev.stream();
    let src = stream
        .clone_htod(bytes)
        .map_err(|e| alloc_err("fp8 upload", e))?;
    let sc = stream
        .clone_htod(scales)
        .map_err(|e| alloc_err("fp8 upload", e))?;
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("fp8 upload", e))?;
    let w = with_oom_retry(dev, "fp8 dequant", || unsafe { stream.alloc::<f32>(total) })?;
    let func = dev.quantized_fn("dequant_fp8_tiles")?;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let args = [
        inp as i64,
        block as i64,
        row0 as i64,
        inp.div_ceil(block) as i64,
        total as i64,
    ];
    let mut b = stream.launch_builder(&func);
    b.arg(&src);
    b.arg(&sc);
    b.arg(&w);
    for a in &args {
        b.arg(a);
    }
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("dequant_fp8_tiles launch: {e}")))?;
    let y = matmul_nt_f32(dev, &xs, &w, 1, n, inp, out)?;
    stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("fp8 linear download: {e}")))
}

/// `x` [n, inp] through a GGUF-quantised weight [out, inp] - iq2_xxs, q2_K or q8_0 blocks as the file
/// stores them - dequantised on the device and applied at full precision. Returns [n, out].
/// The IQ2_XXS decode tables on the device: the grid of two hundred and fifty-six eight-byte
/// points, and the hundred and twenty-eight sign patterns. The same for every weight, so a caller
/// that keeps weights there keeps these beside them.
pub fn iq2_xxs_tables(dev: &CudaDevice) -> Result<(CudaSlice<u8>, CudaSlice<u8>)> {
    let stream = dev.stream();
    let grid: Vec<u8> = crate::tensor::quant_cpu::format::IQ2XXS_GRID
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .collect();
    let grid = stream
        .clone_htod(&grid)
        .map_err(|e| alloc_err("iq2_xxs grid upload", e))?;
    let signs = stream
        .clone_htod(&crate::tensor::quant_cpu::format::KSIGNS_IQ2XS)
        .map_err(|e| alloc_err("iq2_xxs signs upload", e))?;
    Ok((grid, signs))
}

/// One row through an IQ2_XXS weight `[out, inp]` whose blocks are already on the device. The
/// weight is read where it lies rather than decoded into a dense copy first, which is what makes
/// keeping it there worth the room.
pub fn gpu_mmv_iq2_xxs(
    dev: &CudaDevice,
    blocks: &CudaSlice<u8>,
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (out, inp): (usize, usize),
    x: &[f32],
) -> Result<Vec<f32>> {
    if inp % 256 != 0 || x.len() != inp {
        return Err(Error(format!(
            "iq2_xxs matvec: [{out}, {inp}] against {} inputs",
            x.len()
        )));
    }
    let stream = dev.stream();
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("iq2_xxs matvec upload", e))?;
    let y = with_oom_retry(dev, "iq2_xxs matvec", || unsafe {
        stream.alloc::<f32>(out)
    })?;
    let func = dev.quantized_fn("mmv_iq2_xxs_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (out as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (ki, ni) = (inp as i32, out as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(blocks);
    b.arg(tables.0);
    b.arg(tables.1);
    b.arg(&xs);
    b.arg(&y);
    b.arg(&ki);
    b.arg(&ni);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("mmv_iq2_xxs_f32 launch: {e}")))?;
    stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("iq2_xxs matvec download: {e}")))
}

/// A routed expert's whole forward for one row, its three weights already on the device: the gate
/// and up products with the SwiGLU between them, then the down product. Returns the rows entering
/// the down projection and the output. Two launches and one crossing each way, where three
/// separate products pay four - and on a decode the crossing, not the arithmetic, is the cost.
#[allow(clippy::too_many_arguments)]
pub fn gpu_expert_row(
    dev: &CudaDevice,
    gate: &CudaSlice<u8>,
    up: &CudaSlice<u8>,
    down: &CudaSlice<u8>,
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (dim, inter): (usize, usize),
    x: &[f32],
    limit: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if dim % 256 != 0 || inter % 256 != 0 || x.len() != dim {
        return Err(Error(format!(
            "expert row: dim {dim}, inter {inter}, {} inputs",
            x.len()
        )));
    }
    let stream = dev.stream();
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("expert row upload", e))?;
    let h = with_oom_retry(dev, "expert rows", || unsafe { stream.alloc::<f32>(inter) })?;
    let func = dev.quantized_fn("mmv_expert_h_iq2_xxs_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (inter as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (dim_i, inter_i) = (dim as i32, inter as i32);
    let mut b = stream.launch_builder(&func);
    b.arg(gate);
    b.arg(up);
    b.arg(tables.0);
    b.arg(tables.1);
    b.arg(&xs);
    b.arg(&h);
    b.arg(&dim_i);
    b.arg(&inter_i);
    b.arg(&limit);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("mmv_expert_h_iq2_xxs_f32 launch: {e}")))?;

    let y = with_oom_retry(dev, "expert output", || unsafe { stream.alloc::<f32>(dim) })?;
    let func = dev.quantized_fn("mmv_q2_k_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (dim as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&func);
    b.arg(down);
    b.arg(&h);
    b.arg(&y);
    b.arg(&inter_i);
    b.arg(&dim_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("mmv_q2_k_f32 launch: {e}")))?;
    // The rows entering the down projection come back beside the output: an observer watching a
    // calibration reads them, and nine kilobytes is not worth a second path.
    let hs = stream
        .clone_dtoh(&h)
        .map_err(|e| Error(format!("expert rows download: {e}")))?;
    let ys = stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("expert row download: {e}")))?;
    Ok((hs, ys))
}

/// The same two kernels as `gpu_expert_row`, over several experts that share one input row - the
/// shape a decode takes, where every active expert of a layer sees the same token. The input goes
/// over once, all the kernels are queued on the one stream, and the outputs come back after they
/// have all run: one host round-trip for the block instead of one per expert, which is what the
/// per-expert path pays. Bit-exact with running them one at a time - the kernels and their inputs
/// are the same, only the host waits less.
#[allow(clippy::type_complexity)]
pub fn gpu_expert_rows(
    dev: &CudaDevice,
    experts: &[(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (dim, inter): (usize, usize),
    x: &[f32],
    limit: f32,
) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
    if dim % 256 != 0 || inter % 256 != 0 || x.len() != dim {
        return Err(Error(format!(
            "expert rows: dim {dim}, inter {inter}, {} inputs",
            x.len()
        )));
    }
    let stream = dev.stream();
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("expert rows upload", e))?;
    let hfun = dev.quantized_fn("mmv_expert_h_iq2_xxs_f32")?;
    let dfun = dev.quantized_fn("mmv_q2_k_f32")?;
    let (dim_i, inter_i) = (dim as i32, inter as i32);
    // Every kernel of every expert is queued before a single byte is read back: the outputs
    // downloaded below wait once, on the last of them, not once per expert.
    let mut bufs: Vec<(CudaSlice<f32>, CudaSlice<f32>)> = Vec::with_capacity(experts.len());
    for (gate, up, down) in experts {
        let h = with_oom_retry(dev, "expert rows h", || unsafe { stream.alloc::<f32>(inter) })?;
        let hcfg = LaunchConfig {
            grid_dim: (inter as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&hfun);
        b.arg(*gate);
        b.arg(*up);
        b.arg(tables.0);
        b.arg(tables.1);
        b.arg(&xs);
        b.arg(&h);
        b.arg(&dim_i);
        b.arg(&inter_i);
        b.arg(&limit);
        unsafe { b.launch(hcfg) }
            .map_err(|e| Error(format!("mmv_expert_h_iq2_xxs_f32 launch: {e}")))?;
        let y = with_oom_retry(dev, "expert rows y", || unsafe { stream.alloc::<f32>(dim) })?;
        let dcfg = LaunchConfig {
            grid_dim: (dim as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&dfun);
        b.arg(*down);
        b.arg(&h);
        b.arg(&y);
        b.arg(&inter_i);
        b.arg(&dim_i);
        unsafe { b.launch(dcfg) }.map_err(|e| Error(format!("mmv_q2_k_f32 launch: {e}")))?;
        bufs.push((h, y));
    }
    // Only the output comes back. The rows entering the down projection (`h`) are read by a
    // calibration observer, which a decode does not run; downloading them for every expert of
    // every layer was a device round-trip spent on nothing. The caller gets an empty `h`.
    let mut out = Vec::with_capacity(bufs.len());
    for (_h, y) in &bufs {
        let ys = stream
            .clone_dtoh(y)
            .map_err(|e| Error(format!("expert rows download: {e}")))?;
        out.push((Vec::new(), ys));
    }
    Ok(out)
}

/// The whole block of experts in ONE pair of launches: the h kernel over `(inter, nexperts)`
/// blocks and the down kernel over `(dim, nexperts)`, each expert's weights reached through a
/// device array of pointers. The card runs all of a layer's experts at once instead of one at a
/// time on the host's clock - the difference between a matvec per crossing and a block per layer.
/// Bit-exact with `gpu_expert_rows`.
#[allow(clippy::type_complexity)]
pub fn gpu_expert_rows_grouped(
    dev: &CudaDevice,
    experts: &[(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (dim, inter): (usize, usize),
    x: &[f32],
    limit: f32,
) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
    use cudarc::driver::DevicePtr;
    if dim % 256 != 0 || inter % 256 != 0 || x.len() != dim {
        return Err(Error(format!(
            "grouped expert rows: dim {dim}, inter {inter}, {} inputs",
            x.len()
        )));
    }
    let n = experts.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let stream = dev.stream();
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("grouped expert upload", e))?;
    // The experts' device addresses, gathered into three arrays the kernels index by blockIdx.y.
    // The guards keep the slices mapped until the launches are queued.
    let mut gate_ptrs = Vec::with_capacity(n);
    let mut up_ptrs = Vec::with_capacity(n);
    let mut down_ptrs = Vec::with_capacity(n);
    let mut guards = Vec::with_capacity(3 * n);
    for (g, u, d) in experts {
        let (gp, gg) = g.device_ptr(&stream);
        let (up, gu) = u.device_ptr(&stream);
        let (dp, gd) = d.device_ptr(&stream);
        gate_ptrs.push(gp);
        up_ptrs.push(up);
        down_ptrs.push(dp);
        guards.push(gg);
        guards.push(gu);
        guards.push(gd);
    }
    let gate_arr = stream
        .clone_htod(&gate_ptrs)
        .map_err(|e| alloc_err("grouped gate ptrs", e))?;
    let up_arr = stream
        .clone_htod(&up_ptrs)
        .map_err(|e| alloc_err("grouped up ptrs", e))?;
    let down_arr = stream
        .clone_htod(&down_ptrs)
        .map_err(|e| alloc_err("grouped down ptrs", e))?;
    let h = with_oom_retry(dev, "grouped h", || unsafe { stream.alloc::<f32>(n * inter) })?;
    let y = with_oom_retry(dev, "grouped y", || unsafe { stream.alloc::<f32>(n * dim) })?;
    let (dim_i, inter_i, n_i) = (dim as i32, inter as i32, n as i32);
    let hfun = dev.quantized_fn("mmv_expert_h_iq2_xxs_grouped_f32")?;
    let hcfg = LaunchConfig {
        grid_dim: (inter as u32, n as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&hfun);
    b.arg(&gate_arr);
    b.arg(&up_arr);
    b.arg(tables.0);
    b.arg(tables.1);
    b.arg(&xs);
    b.arg(&h);
    b.arg(&dim_i);
    b.arg(&inter_i);
    b.arg(&limit);
    b.arg(&n_i);
    unsafe { b.launch(hcfg) }
        .map_err(|e| Error(format!("mmv_expert_h_iq2_xxs_grouped_f32 launch: {e}")))?;
    let dfun = dev.quantized_fn("mmv_q2_k_grouped_f32")?;
    let dcfg = LaunchConfig {
        grid_dim: (dim as u32, n as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&dfun);
    b.arg(&down_arr);
    b.arg(&h);
    b.arg(&y);
    b.arg(&inter_i);
    b.arg(&dim_i);
    b.arg(&n_i);
    unsafe { b.launch(dcfg) }.map_err(|e| Error(format!("mmv_q2_k_grouped_f32 launch: {e}")))?;
    drop(guards);
    let ys_all = stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("grouped expert download: {e}")))?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((Vec::new(), ys_all[i * dim..(i + 1) * dim].to_vec()));
    }
    Ok(out)
}

/// Grouped experts with one activation row per instance: `x` is [n, dim] and instance `i` reads
/// row `i`. A speculative verify makes one instance per (expert, token) it routes, so its whole
/// block of experts runs in one launch. Returns instance `i`'s output row. Bit-exact with
/// `gpu_expert_row` per instance.
pub fn gpu_expert_rows_grouped_multi(
    dev: &CudaDevice,
    experts: &[(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (dim, inter): (usize, usize),
    x: &[f32],
    limit: f32,
) -> Result<Vec<Vec<f32>>> {
    use cudarc::driver::DevicePtr;
    let n = experts.len();
    if dim % 256 != 0 || inter % 256 != 0 || x.len() != n * dim {
        return Err(Error(format!(
            "grouped multi expert rows: dim {dim}, inter {inter}, {} inputs for {n}",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let stream = dev.stream();
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("grouped multi upload", e))?;
    let mut gate_ptrs = Vec::with_capacity(n);
    let mut up_ptrs = Vec::with_capacity(n);
    let mut down_ptrs = Vec::with_capacity(n);
    let mut guards = Vec::with_capacity(3 * n);
    for (g, u, d) in experts {
        let (gp, gg) = g.device_ptr(&stream);
        let (up, gu) = u.device_ptr(&stream);
        let (dp, gd) = d.device_ptr(&stream);
        gate_ptrs.push(gp);
        up_ptrs.push(up);
        down_ptrs.push(dp);
        guards.push(gg);
        guards.push(gu);
        guards.push(gd);
    }
    let gate_arr = stream
        .clone_htod(&gate_ptrs)
        .map_err(|e| alloc_err("grouped multi gate ptrs", e))?;
    let up_arr = stream
        .clone_htod(&up_ptrs)
        .map_err(|e| alloc_err("grouped multi up ptrs", e))?;
    let down_arr = stream
        .clone_htod(&down_ptrs)
        .map_err(|e| alloc_err("grouped multi down ptrs", e))?;
    let h = with_oom_retry(dev, "grouped multi h", || unsafe { stream.alloc::<f32>(n * inter) })?;
    let y = with_oom_retry(dev, "grouped multi y", || unsafe { stream.alloc::<f32>(n * dim) })?;
    let (dim_i, inter_i, n_i) = (dim as i32, inter as i32, n as i32);
    let hfun = dev.quantized_fn("mmv_expert_h_iq2_xxs_grouped_multi_f32")?;
    let hcfg = LaunchConfig {
        grid_dim: (inter as u32, n as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&hfun);
    b.arg(&gate_arr);
    b.arg(&up_arr);
    b.arg(tables.0);
    b.arg(tables.1);
    b.arg(&xs);
    b.arg(&h);
    b.arg(&dim_i);
    b.arg(&inter_i);
    b.arg(&limit);
    b.arg(&n_i);
    unsafe { b.launch(hcfg) }
        .map_err(|e| Error(format!("mmv_expert_h_iq2_xxs_grouped_multi_f32 launch: {e}")))?;
    let dfun = dev.quantized_fn("mmv_q2_k_grouped_f32")?;
    let dcfg = LaunchConfig {
        grid_dim: (dim as u32, n as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&dfun);
    b.arg(&down_arr);
    b.arg(&h);
    b.arg(&y);
    b.arg(&inter_i);
    b.arg(&dim_i);
    b.arg(&n_i);
    unsafe { b.launch(dcfg) }
        .map_err(|e| Error(format!("mmv_q2_k_grouped_f32 (multi) launch: {e}")))?;
    drop(guards);
    let ys_all = stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("grouped multi download: {e}")))?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(ys_all[i * dim..(i + 1) * dim].to_vec());
    }
    Ok(out)
}

pub fn gpu_quant_linear(
    dev: &CudaDevice,
    dtype: crate::tensor::quantized::GgmlDType,
    blocks: &[u8],
    (out, inp): (usize, usize),
    x: &[f32],
) -> Result<Vec<f32>> {
    let total = out * inp;
    let want = total / dtype.block_size() * dtype.type_size();
    if inp == 0 || total % dtype.block_size() != 0 || blocks.len() != want || x.len() % inp != 0 {
        return Err(Error(format!(
            "{dtype:?} linear: {} block bytes ({want} wanted), {} inputs for [{out}, {inp}]",
            blocks.len(),
            x.len()
        )));
    }
    let n = x.len() / inp;
    let stream = dev.stream();
    let blob = stream
        .clone_htod(blocks)
        .map_err(|e| alloc_err("quant linear upload", e))?;
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("quant linear upload", e))?;
    let w = match dtype {
        crate::tensor::quantized::GgmlDType::Q8_0 => dequantize_q8_0_f32(dev, &blob, total)?,
        crate::tensor::quantized::GgmlDType::Q2K => {
            let dst = with_oom_retry(dev, "dequant q2_K", || unsafe {
                stream.alloc::<f32>(total)
            })?;
            let func = dev.quantized_fn("dequant_q2_k_f32")?;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let total_i = total as i64;
            let mut b = stream.launch_builder(&func);
            b.arg(&blob);
            b.arg(&dst);
            b.arg(&total_i);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("dequant_q2_k_f32 launch: {e}")))?;
            dst
        }
        crate::tensor::quantized::GgmlDType::Iq2Xxs => {
            let grid: Vec<u8> = crate::tensor::quant_cpu::format::IQ2XXS_GRID
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect();
            let grid = stream
                .clone_htod(&grid)
                .map_err(|e| alloc_err("iq2_xxs grid upload", e))?;
            let signs = stream
                .clone_htod(&crate::tensor::quant_cpu::format::KSIGNS_IQ2XS)
                .map_err(|e| alloc_err("iq2_xxs signs upload", e))?;
            let dst = with_oom_retry(dev, "dequant iq2_xxs", || unsafe {
                stream.alloc::<f32>(total)
            })?;
            let func = dev.quantized_fn("dequant_iq2_xxs_f32")?;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let total_i = total as i64;
            let mut b = stream.launch_builder(&func);
            b.arg(&blob);
            b.arg(&grid);
            b.arg(&signs);
            b.arg(&dst);
            b.arg(&total_i);
            unsafe { b.launch(cfg) }
                .map_err(|e| Error(format!("dequant_iq2_xxs_f32 launch: {e}")))?;
            dst
        }
        other => {
            return Err(Error(format!(
                "{other:?} has no device dequantiser for a linear"
            )))
        }
    };
    let y = matmul_nt_f32(dev, &xs, &w, 1, n, inp, out)?;
    stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("quant linear download: {e}")))
}

/// Sparse attention with a per-head sink on the device: `q` [s, h, d], `kv` [n, d] one shared head,
/// `idxs` [s, topk] into `kv`, negative for an empty slot. Returns [s, h, d], each query's softmax
/// over its keys and its head's sink, applied to those keys.
pub fn gpu_sparse_attn(
    dev: &CudaDevice,
    q: &[f32],
    kv: &[f32],
    sink: &[f32],
    idxs: &[i32],
    (s, h, d, topk): (usize, usize, usize, usize),
    scale: f32,
) -> Result<Vec<f32>> {
    if q.len() != s * h * d
        || sink.len() != h
        || idxs.len() != s * topk
        || d == 0
        || kv.len() % d != 0
    {
        return Err(Error(format!(
            "sparse attention: {} queries, {} keys, {} sinks, {} slots for s {s} h {h} d {d} topk {topk}",
            q.len(),
            kv.len(),
            sink.len(),
            idxs.len()
        )));
    }
    let n = kv.len() / d;
    if let Some(&bad) = idxs.iter().find(|&&i| i >= n as i32) {
        return Err(Error(format!("sparse attention: slot {bad} past {n} keys")));
    }
    let stream = dev.stream();
    let up = |e| alloc_err("sparse attention upload", e);
    let (kvd, sd, id) = (
        stream.clone_htod(kv).map_err(up)?,
        stream.clone_htod(sink).map_err(up)?,
        stream.clone_htod(idxs).map_err(up)?,
    );
    // Queries go through in chunks whose gathered keys, logits and outputs take at most half of
    // what the card has free now.
    let (free, _) = dev
        .ctx
        .mem_get_info()
        .map_err(|e| Error(format!("sparse attention memory: {e}")))?;
    let per_query = (topk * d + h * topk + 2 * h * d) * std::mem::size_of::<f32>();
    let chunk = (free / 2 / per_query.max(1)).clamp(1, s.max(1));
    let (func_gather, func_softmax) = (
        dev.quantized_fn("gather_slots_f32")?,
        dev.quantized_fn("sink_softmax_f32")?,
    );
    let mut result = vec![0f32; s * h * d];
    for start in (0..s).step_by(chunk) {
        let count = chunk.min(s - start);
        let qd = stream
            .clone_htod(&q[start * h * d..(start + count) * h * d])
            .map_err(up)?;
        let keys = with_oom_retry(dev, "sparse attention keys", || unsafe {
            stream.alloc::<f32>(count * topk * d)
        })?;
        let cfg = |n: usize| LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (start_i, count_i, topk_i, d_i, h_i) =
            (start as i64, count as i64, topk as i64, d as i64, h as i64);
        let mut g = stream.launch_builder(&func_gather);
        g.arg(&kvd);
        g.arg(&id);
        g.arg(&keys);
        g.arg(&start_i);
        g.arg(&count_i);
        g.arg(&topk_i);
        g.arg(&d_i);
        unsafe { g.launch(cfg(count * topk)) }
            .map_err(|e| Error(format!("gather_slots_f32 launch: {e}")))?;
        // [count, h, d] @ [count, topk, d]^T -> [count, h, topk]
        let logits = matmul_nt_f32(dev, &qd, &keys, count, h, d, topk)?;
        let mut sm = stream.launch_builder(&func_softmax);
        sm.arg(&logits);
        sm.arg(&sd);
        sm.arg(&id);
        sm.arg(&start_i);
        sm.arg(&count_i);
        sm.arg(&h_i);
        sm.arg(&topk_i);
        sm.arg(&scale);
        unsafe { sm.launch(cfg(count * h)) }
            .map_err(|e| Error(format!("sink_softmax_f32 launch: {e}")))?;
        // [count, h, topk] @ [count, topk, d] -> [count, h, d]
        let out = matmul_f32(dev, &logits, &keys, count, h, topk, d)?;
        stream
            .memcpy_dtoh(&out, &mut result[start * h * d..(start + count) * h * d])
            .map_err(|e| Error(format!("sparse attention download: {e}")))?;
    }
    Ok(result)
}

/// An indexer's scores on the device: `q` [s, nh, ihd], `k` [g, ihd], `weights` [s, nh]. Returns
/// [s, g]: for each query the sum over heads of relu(q . k) * weight * scale over the compressed
/// positions it can reach, `(i + 1) / ratio` of them, and -inf past those.
#[allow(clippy::too_many_arguments)]
pub fn gpu_index_scores(
    dev: &CudaDevice,
    q: &[f32],
    k: &[f32],
    weights: &[f32],
    (s, nh, ihd, g): (usize, usize, usize, usize),
    ratio: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    if q.len() != s * nh * ihd || k.len() != g * ihd || weights.len() != s * nh || ratio == 0 {
        return Err(Error(format!(
            "index scores: {} queries, {} keys, {} weights for s {s} nh {nh} ihd {ihd} g {g}",
            q.len(),
            k.len(),
            weights.len()
        )));
    }
    let stream = dev.stream();
    let up = |e| alloc_err("index scores upload", e);
    let (qd, kd, wd) = (
        stream.clone_htod(q).map_err(up)?,
        stream.clone_htod(k).map_err(up)?,
        stream.clone_htod(weights).map_err(up)?,
    );
    let out = with_oom_retry(dev, "index scores", || unsafe {
        stream.alloc::<f32>(s * g)
    })?;
    let func = dev.quantized_fn("index_scores_f32")?;
    let cfg = LaunchConfig {
        grid_dim: ((s * g).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let dims = [s as i64, nh as i64, ihd as i64, g as i64, ratio as i64];
    let mut b = stream.launch_builder(&func);
    b.arg(&qd);
    b.arg(&kd);
    b.arg(&wd);
    b.arg(&out);
    for v in &dims {
        b.arg(v);
    }
    b.arg(&scale);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("index_scores_f32 launch: {e}")))?;
    stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("index scores download: {e}")))
}

/// q2_K quantise on the device weighted by `importance`, one weight per column of a row
/// `n_per_row` values wide - the layout the CPU encoder's guided path takes, and its fit.
pub fn gpu_quantize_q2_k_guided(
    dev: &CudaDevice,
    f32s: &[f32],
    importance: &[f32],
    n_per_row: usize,
) -> Result<Vec<u8>> {
    if f32s.len() % 256 != 0 || n_per_row % 256 != 0 || importance.len() != n_per_row {
        return Err(Error(format!(
            "q2_K guided quantize: {} values, {} weights, {n_per_row} per row",
            f32s.len(),
            importance.len()
        )));
    }
    let nblocks = f32s.len() / 256;
    let stream = dev.stream();
    let x = stream
        .clone_htod(f32s)
        .map_err(|e| Error(format!("q2_K upload: {e}")))?;
    let imp = stream
        .clone_htod(importance)
        .map_err(|e| Error(format!("q2_K importance upload: {e}")))?;
    let out = with_oom_retry(dev, "requant_q2_k_guided", || unsafe {
        stream.alloc::<u8>(nblocks * std::mem::size_of::<crate::tensor::quant_cpu::BlockQ2K>())
    })?;
    let func = dev.quantized_fn("requant_q2_k_guided")?;
    let cfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, per_row) = (nblocks as i32, (n_per_row / 256) as i32);
    let (block_step, value_step, no) = (256i64, 1i64, 0i32);
    let unused = with_oom_retry(dev, "requant_q2_k_guided", || unsafe {
        stream.alloc::<f32>(1)
    })?;
    let mut b = stream.launch_builder(&func);
    b.arg(&x);
    b.arg(&block_step);
    b.arg(&value_step);
    b.arg(&imp);
    b.arg(&out);
    b.arg(&unused);
    b.arg(&no);
    b.arg(&n_i);
    b.arg(&per_row);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("requant_q2_k_guided launch: {e}")))?;
    stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("q2_K download: {e}")))
}

/// IQ2_XXS quantise on the device, optionally weighted: `importance` holds one weight per column of
/// a row that `n_per_row` values wide, the same layout the CPU encoder takes. The grid is uploaded
/// with the values so the kernel never carries a second copy of the table.
pub fn gpu_quantize_iq2_xxs(
    dev: &CudaDevice,
    f32s: &[f32],
    importance: Option<&[f32]>,
    n_per_row: usize,
) -> Result<Vec<u8>> {
    if f32s.len() % 256 != 0 || n_per_row % 256 != 0 {
        return Err(Error(format!(
            "iq2_xxs quantize: {} values, {n_per_row} per row, not multiples of 256",
            f32s.len()
        )));
    }
    let nblocks = f32s.len() / 256;
    let stream = dev.stream();
    let grid: Vec<f32> = crate::tensor::quant_cpu::IQ2XXS_GRID
        .iter()
        .flat_map(|g| g.to_le_bytes().map(|b| b as f32))
        .collect();
    let x = stream
        .clone_htod(f32s)
        .map_err(|e| Error(format!("iq2_xxs upload: {e}")))?;
    let has_imp = importance.is_some() as i32;
    let imp_host: Vec<f32> = importance.map_or_else(|| vec![0.0; 256], |w| w.to_vec());
    let imp = stream
        .clone_htod(&imp_host)
        .map_err(|e| Error(format!("iq2_xxs importance upload: {e}")))?;
    let grid = stream
        .clone_htod(&grid)
        .map_err(|e| Error(format!("iq2_xxs grid upload: {e}")))?;
    let out = with_oom_retry(dev, "requant_iq2_xxs", || unsafe {
        stream.alloc::<u8>(nblocks * 66)
    })?;
    let func = dev.quantized_fn("requant_iq2_xxs")?;
    let cfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, per_row, has) = (nblocks as i32, (n_per_row / 256) as i32, has_imp);
    let (block_step, value_step, no) = (256i64, 1i64, 0i32);
    let unused = with_oom_retry(dev, "requant_iq2_xxs", || unsafe { stream.alloc::<f32>(1) })?;
    let mut b = stream.launch_builder(&func);
    b.arg(&x);
    b.arg(&block_step);
    b.arg(&value_step);
    b.arg(&imp);
    b.arg(&grid);
    b.arg(&out);
    b.arg(&unused);
    b.arg(&no);
    b.arg(&n_i);
    b.arg(&per_row);
    b.arg(&has);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("requant_iq2_xxs launch: {e}")))?;
    stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("iq2_xxs download: {e}")))
}

/// `a` (`m` by `k`) times `b` (`k` by `n`), row-major host buffers, into `out` (`m` by `n`), through
/// cuBLAS on `dev` at full precision.
pub fn gpu_matmul_host(
    dev: &CudaDevice,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error(format!(
            "matmul: {} by {} into {} for {m}x{k}x{n}",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    let stream = dev.stream();
    let a = stream
        .clone_htod(a)
        .map_err(|e| Error(format!("matmul upload: {e}")))?;
    let b = stream
        .clone_htod(b)
        .map_err(|e| Error(format!("matmul upload: {e}")))?;
    let c = matmul_f32_inner(dev, &a, &b, 1, m, k, n, false)?;
    stream
        .memcpy_dtoh(&c, out)
        .map_err(|e| Error(format!("matmul download: {e}")))
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
    if smallk && (b_size != 1 || !n.is_multiple_of(4)) {
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

#[cfg(test)]
mod requant_quantize_parity {
    use super::*;
    use crate::tensor::quantized::GgmlDType;

    /// Dequantise a Q8_0 GGUF blob (34 bytes/block: f16 scale, then 32 signed bytes).
    fn dequant_q8_0(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 34 * 32);
        for blk in bytes.chunks_exact(34) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            for &q in &blk[2..34] {
                out.push(d * (q as i8) as f32);
            }
        }
        out
    }

    /// Phase-7 GPU gate: the device Q8_0 quantiser produces a valid, reproducible blob whose
    /// dequantisation matches the CPU quantiser's to within a quantisation step. Not bit-identical
    /// to the CPU (the decided target is quality-equivalent), so the check is a tolerance sized to
    /// each block's own step, plus an exact run-to-run reproducibility check.
    #[test]
    fn gpu_q8_0_is_quality_equivalent_and_reproducible() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the GPU Q8_0 quantiser is NOT covered by this run");
            return;
        };
        const N: usize = 32 * 128;
        let xs: Vec<f32> = (0..N)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(374_761_393)
                    .rotate_left(11);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * 40.0
            })
            .collect();

        let gpu = gpu_quantize_q8_0(&dev, &xs).unwrap();
        let cpu = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Q8_0, &xs).unwrap();
        assert_eq!(gpu.len(), cpu.len(), "blob size differs");

        // Reproducible: the digest names the bytes, so the same input must give the same bytes.
        let gpu2 = gpu_quantize_q8_0(&dev, &xs).unwrap();
        assert_eq!(gpu, gpu2, "GPU quantise is not reproducible");

        let dg = dequant_q8_0(&gpu);
        let dc = dequant_q8_0(&cpu);
        // GPU and CPU compute the same block scale (amax is an exact max, d = amax/127 the same
        // single f32 divide, then the same f16 round), so they agree to within one level - the
        // only slack is a 1-ULP boundary in 1/d that can flip a value sitting exactly on the
        // half-step. That equivalence is the gate; the reproducibility check above is the other.
        for (b, (xb, (gb, cb))) in xs
            .chunks_exact(32)
            .zip(dg.chunks_exact(32).zip(dc.chunks_exact(32)))
            .enumerate()
        {
            let amax = xb.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let step = amax / 127.0;
            for i in 0..32 {
                assert!(
                    (gb[i] - cb[i]).abs() <= step * 1.01 + 1e-6,
                    "block {b} lane {i}: gpu {} vs cpu {} (step {step})",
                    gb[i],
                    cb[i]
                );
            }
        }
    }
}

#[cfg(test)]
mod requant_q4k_parity {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QHostTensor};

    fn dequant_f32(bytes: &[u8], elems: usize) -> Vec<f32> {
        QHostTensor::from_bytes(bytes, GgmlDType::Q4K, vec![elems])
            .unwrap()
            .dequantize_f32()
            .unwrap()
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum::<f32>().max(1e-20);
        (num / den).sqrt()
    }

    /// Phase-7 GPU gate for q4_K: the device quantiser reconstructs the input as well as the CPU
    /// quantiser (the fit can pick a different candidate scale at a near-tie, so this is not a
    /// pointwise match but an equal-quality one), and is reproducible run-to-run.
    #[test]
    fn gpu_q4_k_is_quality_equivalent_and_reproducible() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the GPU q4_K quantiser is NOT covered by this run");
            return;
        };
        const N: usize = 256 * 96;
        let xs: Vec<f32> = (0..N)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(1_013_904_223)
                    .rotate_left(13);
                let u = (h >> 8) as f32 / (1 << 24) as f32 - 0.5;
                // A few scales of magnitude so blocks differ, plus the odd spike.
                u * 6.0 + (if i % 257 == 0 { 30.0 } else { 0.0 })
            })
            .collect();

        let gpu = gpu_quantize_q4_k(&dev, &xs).unwrap();
        let cpu = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Q4K, &xs).unwrap();
        assert_eq!(gpu.len(), cpu.len(), "q4_K blob size differs");

        let gpu2 = gpu_quantize_q4_k(&dev, &xs).unwrap();
        assert_eq!(gpu, gpu2, "GPU q4_K quantise is not reproducible");

        let e_gpu = rel_l2(&dequant_f32(&gpu, N), &xs);
        let e_cpu = rel_l2(&dequant_f32(&cpu, N), &xs);
        // GPU reconstruction no worse than the CPU's by more than a small margin.
        assert!(
            e_gpu <= e_cpu * 1.05 + 1e-4,
            "GPU q4_K worse than CPU: gpu rel_l2 {e_gpu} vs cpu {e_cpu}"
        );
        // And a sane absolute quality (q4_K on this data is well under 10% relative).
        assert!(e_gpu < 0.1, "GPU q4_K reconstruction poor: {e_gpu}");
    }
}

#[cfg(test)]
mod requant_q3k_parity {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QHostTensor};

    fn dequant(bytes: &[u8], elems: usize) -> Vec<f32> {
        QHostTensor::from_bytes(bytes, GgmlDType::Q3K, vec![elems])
            .unwrap()
            .dequantize_f32()
            .unwrap()
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum::<f32>().max(1e-20);
        (num / den).sqrt()
    }

    /// Phase-7 GPU gate for q3_K: the device quantiser reconstructs the input as well as the CPU
    /// quantiser (the signed fit can settle on a different scale at a near-tie, so equal-quality
    /// rather than pointwise), and is reproducible.
    #[test]
    fn gpu_q3_k_is_quality_equivalent_and_reproducible() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the GPU q3_K quantiser is NOT covered by this run");
            return;
        };
        const N: usize = 256 * 96;
        let xs: Vec<f32> = (0..N)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(2_246_822_519)
                    .wrapping_add(2_654_435_761)
                    .rotate_left(15);
                let u = (h >> 8) as f32 / (1 << 24) as f32 - 0.5;
                u * 5.0 + (if i % 401 == 0 { 20.0 } else { 0.0 })
            })
            .collect();

        let gpu = gpu_quantize_q3_k(&dev, &xs).unwrap();
        let cpu = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Q3K, &xs).unwrap();
        assert_eq!(gpu.len(), cpu.len(), "q3_K blob size differs");

        let gpu2 = gpu_quantize_q3_k(&dev, &xs).unwrap();
        assert_eq!(gpu, gpu2, "GPU q3_K quantise is not reproducible");

        let e_gpu = rel_l2(&dequant(&gpu, N), &xs);
        let e_cpu = rel_l2(&dequant(&cpu, N), &xs);
        assert!(
            e_gpu <= e_cpu * 1.05 + 1e-4,
            "GPU q3_K worse than CPU: gpu rel_l2 {e_gpu} vs cpu {e_cpu}"
        );
        assert!(e_gpu < 0.2, "GPU q3_K reconstruction poor: {e_gpu}");
    }
}

#[cfg(test)]
mod requant_q2k_parity {
    use super::*;
    use crate::tensor::quantized::{GgmlDType, QHostTensor};

    fn dequant(bytes: &[u8], elems: usize) -> Vec<f32> {
        QHostTensor::from_bytes(bytes, GgmlDType::Q2K, vec![elems])
            .unwrap()
            .dequantize_f32()
            .unwrap()
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum::<f32>().max(1e-20);
        (num / den).sqrt()
    }

    /// Phase-7 GPU gate for q2_K: equal reconstruction quality to the CPU quantiser, reproducible.
    #[test]
    fn gpu_q2_k_is_quality_equivalent_and_reproducible() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the GPU q2_K quantiser is NOT covered by this run");
            return;
        };
        const N: usize = 256 * 96;
        let xs: Vec<f32> = (0..N)
            .map(|i| {
                let h = (i as u32)
                    .wrapping_mul(40_503)
                    .wrapping_add(2_654_435_761)
                    .rotate_left(17);
                let u = (h >> 8) as f32 / (1 << 24) as f32 - 0.5;
                u * 4.0 + (if i % 509 == 0 { 15.0 } else { 0.0 })
            })
            .collect();

        let gpu = gpu_quantize_q2_k(&dev, &xs).unwrap();
        let cpu = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Q2K, &xs).unwrap();
        assert_eq!(gpu.len(), cpu.len(), "q2_K blob size differs");

        let gpu2 = gpu_quantize_q2_k(&dev, &xs).unwrap();
        assert_eq!(gpu, gpu2, "GPU q2_K quantise is not reproducible");

        let e_gpu = rel_l2(&dequant(&gpu, N), &xs);
        let e_cpu = rel_l2(&dequant(&cpu, N), &xs);
        assert!(
            e_gpu <= e_cpu * 1.05 + 1e-4,
            "GPU q2_K worse than CPU: gpu rel_l2 {e_gpu} vs cpu {e_cpu}"
        );
        assert!(e_gpu < 0.3, "GPU q2_K reconstruction poor: {e_gpu}");
    }

    /// Weighted by an importance, the device encoder leaves the weighted error the CPU encoder
    /// leaves, to rounding, and gives the same bytes twice.
    #[test]
    fn gpu_q2_k_guided_matches_the_cpu_encoder() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the GPU guided q2_K quantiser is NOT covered by this run");
            return;
        };
        use crate::tensor::quant_cpu::{BlockFormat, BlockQ2K};
        let (rows, cols) = (48usize, 512usize);
        let xs: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761).rotate_left(13);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.04
            })
            .collect();
        let imp: Vec<f32> = (0..cols)
            .map(|c| 0.1 + ((c * 37) % 97) as f32 / 50.0)
            .collect();
        let weighted = |bytes: &[u8]| {
            let mut got = vec![0f32; rows * cols];
            crate::tensor::quant_cpu::to_float_bytes(GgmlDType::Q2K, bytes, &mut got).unwrap();
            let (mut num, mut den) = (0f64, 0f64);
            for (i, (g, t)) in got.iter().zip(&xs).enumerate() {
                let w = imp[i % cols] as f64;
                num += w * (*g as f64 - *t as f64).powi(2);
                den += w * (*t as f64).powi(2);
            }
            (num / den).sqrt()
        };
        let gpu = gpu_quantize_q2_k_guided(&dev, &xs, &imp, cols).unwrap();
        assert_eq!(
            gpu,
            gpu_quantize_q2_k_guided(&dev, &xs, &imp, cols).unwrap()
        );
        let mut blocks = vec![BlockQ2K::zeros(); rows * cols / 256];
        BlockQ2K::quantize_guided(&xs, &mut blocks, &imp, cols);
        // Safety: plain block data.
        let cpu = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                std::mem::size_of_val(&blocks[..]),
            )
        };
        let (e_gpu, e_cpu) = (weighted(&gpu), weighted(cpu));
        assert!(
            e_gpu <= e_cpu * 1.01 + 1e-6,
            "guided q2_K: gpu {e_gpu} against cpu {e_cpu}"
        );
    }
}

#[cfg(test)]
mod iq2_xxs_gpu_tests {
    use super::*;
    use crate::tensor::quantized::GgmlDType;

    /// The matvec that reads IQ2_XXS blocks in place against the same product taken from the
    /// dequantised weight. A misread grid index or sign byte moves a value by a whole point,
    /// which no tolerance this tight would let through.
    #[test]
    fn the_iq2_xxs_matvec_matches_the_dequantised_product() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the iq2_xxs matvec is NOT covered by this run");
            return;
        };
        let (out, inp) = (9usize, 768usize);
        let w: Vec<f32> = (0..out * inp)
            .map(|i| {
                (((i as u32).wrapping_mul(2_246_822_519).rotate_left(7) >> 8) as f32
                    / (1 << 24) as f32
                    - 0.5)
                    * 0.06
            })
            .collect();
        let x: Vec<f32> = (0..inp).map(|i| ((i as f32) * 0.17).cos()).collect();
        let blocks = crate::tensor::quant_cpu::from_float_bytes(GgmlDType::Iq2Xxs, &w).unwrap();
        let mut dense = vec![0f32; out * inp];
        crate::tensor::quant_cpu::to_float_bytes(GgmlDType::Iq2Xxs, &blocks, &mut dense).unwrap();
        let want: Vec<f32> = (0..out)
            .map(|o| (0..inp).map(|c| x[c] * dense[o * inp + c]).sum())
            .collect();

        let on_card = dev.stream().clone_htod(&blocks).unwrap();
        let (grid, signs) = iq2_xxs_tables(&dev).unwrap();
        let got = gpu_mmv_iq2_xxs(&dev, &on_card, (&grid, &signs), (out, inp), &x).unwrap();
        for (g, v) in got.iter().zip(&want) {
            assert!(
                (g - v).abs() <= 1e-3 * v.abs().max(1.0),
                "iq2_xxs matvec: {g} against {v}"
            );
        }
    }

    /// A whole expert on the card against the same arithmetic taken from the dequantised weights:
    /// the two products, the clamped SwiGLU between them, and the down product. The clamps are
    /// the reference's, and getting them wrong moves an output by a hair on most rows and by
    /// everything on the few the clamp bites.
    #[test]
    fn an_expert_row_on_the_card_matches_the_dequantised_arithmetic() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the expert row is NOT covered by this run");
            return;
        };
        let (dim, inter, limit) = (512usize, 256usize, 7.0f32);
        let draw = |seed: u32, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    (((i as u32)
                        .wrapping_add(seed)
                        .wrapping_mul(2_654_435_761)
                        .rotate_left(11)
                        >> 8) as f32
                        / (1 << 24) as f32
                        - 0.5)
                        * scale
                })
                .collect()
        };
        let (gw, uw) = (draw(1, inter * dim, 0.07), draw(2, inter * dim, 0.07));
        let dw = draw(3, dim * inter, 0.07);
        let x = draw(4, dim, 1.0);

        let enc = |v: &[f32], d| crate::tensor::quant_cpu::from_float_bytes(d, v).unwrap();
        let dec = |b: &[u8], d, n| {
            let mut o = vec![0f32; n];
            crate::tensor::quant_cpu::to_float_bytes(d, b, &mut o).unwrap();
            o
        };
        let (gb, ub) = (enc(&gw, GgmlDType::Iq2Xxs), enc(&uw, GgmlDType::Iq2Xxs));
        let db = enc(&dw, GgmlDType::Q2K);
        let (gd, ud) = (
            dec(&gb, GgmlDType::Iq2Xxs, inter * dim),
            dec(&ub, GgmlDType::Iq2Xxs, inter * dim),
        );
        let dd = dec(&db, GgmlDType::Q2K, dim * inter);
        let h: Vec<f32> = (0..inter)
            .map(|j| {
                let g: f32 = (0..dim).map(|c| x[c] * gd[j * dim + c]).sum();
                let u: f32 = (0..dim).map(|c| x[c] * ud[j * dim + c]).sum();
                let (g, u) = (g.min(limit), u.clamp(-limit, limit));
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        let want: Vec<f32> = (0..dim)
            .map(|o| (0..inter).map(|c| h[c] * dd[o * inter + c]).sum())
            .collect();

        let stream = dev.stream();
        let (gc, uc, dc) = (
            stream.clone_htod(&gb).unwrap(),
            stream.clone_htod(&ub).unwrap(),
            stream.clone_htod(&db).unwrap(),
        );
        let (grid, signs) = iq2_xxs_tables(&dev).unwrap();
        let (got_h, got) = gpu_expert_row(
            &dev,
            &gc,
            &uc,
            &dc,
            (&grid, &signs),
            (dim, inter),
            &x,
            limit,
        )
        .unwrap();
        for (g, v) in got_h.iter().zip(&h) {
            assert!(
                (g - v).abs() <= 2e-3 * v.abs().max(1.0),
                "expert rows: {g} against {v}"
            );
        }
        for (g, v) in got.iter().zip(&want) {
            assert!(
                (g - v).abs() <= 2e-3 * v.abs().max(1.0),
                "expert row: {g} against {v}"
            );
        }
    }

    /// A GGUF-quantised linear on the card equals the CPU's dequantisation times the input, for each
    /// format the served file carries.
    #[test]
    fn gpu_quant_linear_matches_the_cpu_dequantisation() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card quant linear is NOT covered by this run");
            return;
        };
        let (out, inp, n) = (12usize, 512usize, 5usize);
        let w: Vec<f32> = (0..out * inp)
            .map(|i| {
                (((i as u32).wrapping_mul(2_654_435_761).rotate_left(9) >> 8) as f32
                    / (1 << 24) as f32
                    - 0.5)
                    * 0.08
            })
            .collect();
        let x: Vec<f32> = (0..n * inp).map(|i| ((i as f32) * 0.31).sin()).collect();
        for dtype in [GgmlDType::Iq2Xxs, GgmlDType::Q2K, GgmlDType::Q8_0] {
            let blocks = crate::tensor::quant_cpu::from_float_bytes(dtype, &w).unwrap();
            let mut dense = vec![0f32; out * inp];
            crate::tensor::quant_cpu::to_float_bytes(dtype, &blocks, &mut dense).unwrap();
            let mut want = vec![0f32; n * out];
            for t in 0..n {
                for o in 0..out {
                    want[t * out + o] = (0..inp).map(|c| x[t * inp + c] * dense[o * inp + c]).sum();
                }
            }
            let got = gpu_quant_linear(&dev, dtype, &blocks, (out, inp), &x).unwrap();
            for (g, v) in got.iter().zip(&want) {
                assert!(
                    (g - v).abs() <= 1e-4 * v.abs().max(1.0),
                    "{dtype:?}: {g} vs {v}"
                );
            }
        }
    }

    /// A rectangular row-major product on the card equals the one on the CPU, to rounding.
    #[test]
    fn gpu_matmul_host_matches_the_cpu() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the GPU matmul is NOT covered by this run");
            return;
        };
        let (m, k, n) = (37usize, 53usize, 29usize);
        let value = |i: usize, salt: u32| {
            let h = (i as u32 ^ salt)
                .wrapping_mul(2_654_435_761)
                .rotate_left(11);
            (h >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|i| value(i, 7)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| value(i, 91)).collect();
        let mut want = vec![0f32; m * n];
        crate::inference::load::compensate::cpu_product(&a, &b, &mut want, (m, k, n)).unwrap();
        let mut got = vec![0f32; m * n];
        gpu_matmul_host(&dev, &a, &b, &mut got, (m, k, n)).unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} against {w}");
        }
    }

    /// On the same weights and importance the device encoder leaves the error the CPU encoder
    /// leaves, to rounding, and gives the same bytes twice.
    #[test]
    fn gpu_iq2_xxs_matches_the_cpu_encoder() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the GPU iq2_xxs quantiser is NOT covered by this run");
            return;
        };
        use crate::tensor::quant_cpu::{BlockFormat, BlockIq2Xxs};
        let (rows, cols) = (48usize, 512usize);
        let xs: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761).rotate_left(13);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.04
            })
            .collect();
        let imp: Vec<f32> = (0..cols)
            .map(|c| 0.1 + ((c * 37) % 97) as f32 / 50.0)
            .collect();
        let decode = |bytes: &[u8]| {
            let mut out = vec![0f32; rows * cols];
            crate::tensor::quant_cpu::to_float_bytes(GgmlDType::Iq2Xxs, bytes, &mut out).unwrap();
            out
        };
        let weighted = |got: &[f32]| {
            let (mut num, mut den) = (0f64, 0f64);
            for (i, (g, t)) in got.iter().zip(&xs).enumerate() {
                let w = imp[i % cols] as f64;
                num += w * (*g as f64 - *t as f64).powi(2);
                den += w * (*t as f64).powi(2);
            }
            (num / den).sqrt()
        };
        let gpu = gpu_quantize_iq2_xxs(&dev, &xs, Some(&imp), cols).unwrap();
        assert_eq!(
            gpu,
            gpu_quantize_iq2_xxs(&dev, &xs, Some(&imp), cols).unwrap()
        );
        let mut blocks = vec![BlockIq2Xxs::zeros(); rows * cols / 256];
        BlockIq2Xxs::quantize_guided(&xs, &mut blocks, &imp, cols);
        let cpu: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                std::mem::size_of_val(&blocks[..]),
            )
        }
        .to_vec();
        let (e_gpu, e_cpu) = (weighted(&decode(&gpu)), weighted(&decode(&cpu)));
        assert!(
            e_gpu <= e_cpu * 1.02,
            "GPU iq2_xxs {e_gpu} against CPU {e_cpu}"
        );
    }
}
