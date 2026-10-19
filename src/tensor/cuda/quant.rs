//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;
use crate::tensor::gpu_runtime::driver::CudaView;

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
    x: &crate::tensor::gpu_runtime::driver::CudaView<'_, f32>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_view_t(dev, x, "mmvq_gguf_quantize_q8_1_f32", k, rows)
}

pub fn quantize_q8_1_rows_f16_view(
    dev: &CudaDevice,
    x: &crate::tensor::gpu_runtime::driver::CudaView<'_, half::f16>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_view_t(dev, x, "mmvq_gguf_quantize_q8_1_f16", k, rows)
}

pub fn quantize_q8_1_rows_bf16_view(
    dev: &CudaDevice,
    x: &crate::tensor::gpu_runtime::driver::CudaView<'_, half::bf16>,
    k: usize,
    rows: usize,
) -> Result<CudaSlice<u8>> {
    quantize_q8_1_rows_view_t(dev, x, "mmvq_gguf_quantize_q8_1_bf16", k, rows)
}

pub(super) fn quantize_q8_1_rows_view_t<T: crate::tensor::gpu_runtime::driver::DeviceRepr>(
    dev: &CudaDevice,
    x: &crate::tensor::gpu_runtime::driver::CudaView<'_, T>,
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

pub(super) fn quantize_q8_1_rows_t<T: crate::tensor::gpu_runtime::driver::DeviceRepr>(
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
        || !scales.len().is_multiple_of(out)
        || !x.len().is_multiple_of(inp)
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
    if inp == 0 || block == 0 || bytes.len() != out * inp || !x.len().is_multiple_of(inp) {
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

/// The instances of one expert a warp reads a weight row once for, in the exact expert kernels: a
/// decode runs one instance per expert, a prompt's expert up to this many rows per read.
const EXACT_TILE: usize = 8;

/// Routed experts on the card, computed as the host computes them (see `cx_expert_h_iq2_xxs`):
/// gate and up IQ2_XXS [inter, dim], down Q2_K [dim, inter], all already on the device. Instance
/// `i` runs expert `instances[i].0` over row `instances[i].1` of `xq`, the q8_K32 rows of the
/// inputs on the card; instances of one expert stand together. Returns, on the card, the rows
/// entering the down projections [instances, inter] and the outputs [instances, dim], each the
/// same bits `offload::cpu_experts` forms on the host.
#[allow(clippy::type_complexity)]
pub fn gpu_experts_exact(
    dev: &CudaDevice,
    experts: &[(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (dim, inter): (usize, usize),
    xq: &CudaSlice<u8>,
    instances: &[(usize, usize)],
    limit: f32,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
    use crate::tensor::gpu_runtime::driver::DevicePtr;
    use crate::tensor::quant_cpu::format::BlockQ8K;
    let n = instances.len();
    if dim % 256 != 0
        || inter % 256 != 0
        || n == 0
        || instances.iter().any(|&(e, _)| e >= experts.len())
    {
        return Err(Error(format!(
            "exact experts: dim {dim}, inter {inter}, {n} instances of {} experts",
            experts.len()
        )));
    }
    let stream = dev.stream();
    let mut gate_ptrs = Vec::with_capacity(experts.len());
    let mut up_ptrs = Vec::with_capacity(experts.len());
    let mut down_ptrs = Vec::with_capacity(experts.len());
    let mut guards = Vec::with_capacity(3 * experts.len());
    for (g, u, d) in experts {
        let (gp, gg) = g.device_ptr(stream);
        let (up, gu) = u.device_ptr(stream);
        let (dp, gd) = d.device_ptr(stream);
        gate_ptrs.push(gp);
        up_ptrs.push(up);
        down_ptrs.push(dp);
        guards.push(gg);
        guards.push(gu);
        guards.push(gd);
    }
    // Runs of instances that read one weight, at most a tile long: (expert, first, count).
    let mut tiles: Vec<i32> = Vec::new();
    let mut widest = 0usize;
    let mut start = 0usize;
    while start < n {
        let e = instances[start].0;
        let mut end = start + 1;
        while end < n && end - start < EXACT_TILE && gate_ptrs[instances[end].0] == gate_ptrs[e] {
            end += 1;
        }
        tiles.extend([e as i32, start as i32, (end - start) as i32]);
        widest = widest.max(end - start);
        start = end;
    }
    let ntiles = tiles.len() / 3;
    let rows: Vec<i32> = instances.iter().map(|&(_, r)| r as i32).collect();
    let up_err = |e| alloc_err("exact experts upload", e);
    let gate_arr = stream.clone_htod(&gate_ptrs).map_err(up_err)?;
    let up_arr = stream.clone_htod(&up_ptrs).map_err(up_err)?;
    let down_arr = stream.clone_htod(&down_ptrs).map_err(up_err)?;
    let tiles_d = stream.clone_htod(&tiles).map_err(up_err)?;
    let rows_d = stream.clone_htod(&rows).map_err(up_err)?;
    let h = with_oom_retry(dev, "exact experts h", || unsafe {
        stream.alloc::<f32>(n * inter)
    })?;
    let hq_bytes = n * (inter / 256) * std::mem::size_of::<BlockQ8K>();
    let hq = with_oom_retry(dev, "exact experts hq", || unsafe {
        stream.alloc::<u8>(hq_bytes)
    })?;
    let y = with_oom_retry(dev, "exact experts y", || unsafe {
        stream.alloc::<f32>(n * dim)
    })?;
    let (dim_i, inter_i, tt_i) = (dim as i32, inter as i32, widest as i32);
    let (sbs, sbs2) = (dim / 256, inter / 256);
    let hfun = dev.quantized_fn("cx_expert_h_iq2_xxs")?;
    let hcfg = LaunchConfig {
        grid_dim: (inter as u32, ntiles as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: ((2 * widest * sbs * 8 + 2 * widest * sbs) * F32_BYTES) as u32,
    };
    let mut b = stream.launch_builder(&hfun);
    b.arg(&gate_arr);
    b.arg(&up_arr);
    b.arg(tables.0);
    b.arg(tables.1);
    b.arg(xq);
    b.arg(&rows_d);
    b.arg(&tiles_d);
    b.arg(&h);
    b.arg(&dim_i);
    b.arg(&inter_i);
    b.arg(&limit);
    b.arg(&tt_i);
    unsafe { b.launch(hcfg) }.map_err(|e| Error(format!("cx_expert_h_iq2_xxs launch: {e}")))?;
    let qfun = dev.quantized_fn("cx_quantize_q8_k")?;
    let nblocks = (n * sbs2) as i64;
    let qcfg = LaunchConfig {
        grid_dim: (nblocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&qfun);
    b.arg(&h);
    b.arg(&hq);
    b.arg(&nblocks);
    unsafe { b.launch(qcfg) }.map_err(|e| Error(format!("cx_quantize_q8_k launch: {e}")))?;
    let dfun = dev.quantized_fn("cx_expert_down_q2_k")?;
    let dcfg = LaunchConfig {
        grid_dim: (dim as u32, ntiles as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: ((2 * widest * sbs2 * 16 + widest * sbs2) * F32_BYTES) as u32,
    };
    let mut b = stream.launch_builder(&dfun);
    b.arg(&down_arr);
    b.arg(&hq);
    b.arg(&tiles_d);
    b.arg(&y);
    b.arg(&inter_i);
    b.arg(&dim_i);
    b.arg(&tt_i);
    unsafe { b.launch(dcfg) }.map_err(|e| Error(format!("cx_expert_down_q2_k launch: {e}")))?;
    drop(guards);
    Ok((h, y))
}

/// Bytes of an f32, the unit of the exact kernels' shared memory.
const F32_BYTES: usize = std::mem::size_of::<f32>();

/// `x`, rows of `dim`, on the card as the q8_K32 rows the exact expert kernels read.
pub fn q8k32_rows_to_card(dev: &CudaDevice, x: &[f32]) -> Result<CudaSlice<u8>> {
    dev.stream()
        .clone_htod(&crate::tensor::quant_cpu::format::q8k32_bytes(x))
        .map_err(|e| alloc_err("expert input upload", e))
}

/// A routed expert's whole forward for one row, its three weights already on the device: the
/// rows entering the down projection and the output, the host's bits.
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
    if x.len() != dim {
        return Err(Error(format!("expert row: dim {dim}, {} inputs", x.len())));
    }
    let xq = q8k32_rows_to_card(dev, x)?;
    let (h, y) = gpu_experts_exact(
        dev,
        &[(gate, up, down)],
        tables,
        (dim, inter),
        &xq,
        &[(0, 0)],
        limit,
    )?;
    let stream = dev.stream();
    let hs = stream
        .clone_dtoh(&h)
        .map_err(|e| Error(format!("expert rows download: {e}")))?;
    let ys = stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("expert row download: {e}")))?;
    Ok((hs, ys))
}

/// Several experts that share one input row - a decode's active experts of a layer - in one pair
/// of launches. Returns each expert's output; the rows entering the down projections stay on the
/// card, so `h` comes back empty.
#[allow(clippy::type_complexity)]
pub fn gpu_expert_rows_grouped(
    dev: &CudaDevice,
    experts: &[(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (dim, inter): (usize, usize),
    x: &[f32],
    limit: f32,
) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
    if x.len() != dim {
        return Err(Error(format!(
            "grouped expert rows: dim {dim}, {} inputs",
            x.len()
        )));
    }
    if experts.is_empty() {
        return Ok(Vec::new());
    }
    let xq = q8k32_rows_to_card(dev, x)?;
    let instances: Vec<(usize, usize)> = (0..experts.len()).map(|i| (i, 0)).collect();
    let (_h, y) = gpu_experts_exact(dev, experts, tables, (dim, inter), &xq, &instances, limit)?;
    let ys = dev
        .stream()
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("grouped expert download: {e}")))?;
    Ok(ys.chunks(dim).map(|r| (Vec::new(), r.to_vec())).collect())
}

/// Experts with one input row per instance: `x` is [n, dim] and instance `i`, expert
/// `experts[i]`, reads row `i`. A speculative verify makes one instance per (expert, token) it
/// routes, so its whole block of experts runs in one pair of launches. Returns instance `i`'s
/// output row.
#[allow(clippy::type_complexity)]
pub fn gpu_expert_rows_grouped_multi(
    dev: &CudaDevice,
    experts: &[(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (dim, inter): (usize, usize),
    x: &[f32],
    limit: f32,
) -> Result<Vec<Vec<f32>>> {
    let n = experts.len();
    if x.len() != n * dim {
        return Err(Error(format!(
            "grouped multi expert rows: dim {dim}, {} inputs for {n}",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let xq = q8k32_rows_to_card(dev, x)?;
    let instances: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    let (_h, y) = gpu_experts_exact(dev, experts, tables, (dim, inter), &xq, &instances, limit)?;
    let ys = dev
        .stream()
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("grouped multi download: {e}")))?;
    Ok(ys.chunks(dim).map(<[f32]>::to_vec).collect())
}

/// Bytes of the card a product of `gpu_quant_linear` may hold at once beside the activations:
/// the expanded f32 weights of one tile of output rows and their products. A weight wider than
/// this is multiplied one tile of rows after another, so a card with little room beside what it
/// keeps still takes a prompt's projections.
pub const QUANT_LINEAR_TILE_BYTES: usize = crate::inference::offload::WEIGHT_TILE_BYTES;

/// `x` [n, inp] through a quantised weight [out, inp] on the card, as `[n, out]`, one tile of
/// output rows at a time (see `QUANT_LINEAR_TILE_BYTES`).
pub fn gpu_quant_linear(
    dev: &CudaDevice,
    dtype: crate::tensor::quantized::GgmlDType,
    blocks: &[u8],
    shape: (usize, usize),
    x: &[f32],
) -> Result<Vec<f32>> {
    gpu_quant_linear_tiled(dev, dtype, blocks, shape, x, QUANT_LINEAR_TILE_BYTES)
}

/// `gpu_quant_linear` with a tile of at most `tile_bytes`.
pub fn gpu_quant_linear_tiled(
    dev: &CudaDevice,
    dtype: crate::tensor::quantized::GgmlDType,
    blocks: &[u8],
    (out, inp): (usize, usize),
    x: &[f32],
    tile_bytes: usize,
) -> Result<Vec<f32>> {
    // From the page-locked copy a layer's read-ahead made, where it made one: the mapping's
    // pages are then not in memory. Every upload below completes before its download returns,
    // and the slab is refilled only two layers on.
    let staged = staged_weight(blocks.as_ptr() as usize, blocks.len()).map(|at| {
        // Safety: the slab holds `blocks.len()` bytes of this weight at `at` until its refill.
        unsafe { std::slice::from_raw_parts(at as *const u8, blocks.len()) }
    });
    let blocks = staged.unwrap_or(blocks);
    let bs = dtype.block_size();
    if inp == 0 || inp % bs != 0 || !x.len().is_multiple_of(inp) {
        return gpu_quant_linear_once(dev, dtype, blocks, (out, inp), x);
    }
    let n = x.len() / inp;
    let row_bytes = inp / bs * dtype.type_size();
    let tile = (tile_bytes / ((inp + n) * 4)).clamp(32, out.max(32)) / 32 * 32;
    if tile >= out || blocks.len() != out * row_bytes {
        return gpu_quant_linear_once(dev, dtype, blocks, (out, inp), x);
    }
    let mut y = vec![0f32; n * out];
    let mut o0 = 0;
    while o0 < out {
        let rows = tile.min(out - o0);
        let part = gpu_quant_linear_once(
            dev,
            dtype,
            &blocks[o0 * row_bytes..(o0 + rows) * row_bytes],
            (rows, inp),
            x,
        )?;
        for t in 0..n {
            y[t * out + o0..t * out + o0 + rows].copy_from_slice(&part[t * rows..(t + 1) * rows]);
        }
        o0 += rows;
    }
    Ok(y)
}

/// `total` values of a quantised weight in `blob`, expanded to f32 on the card at full precision.
fn dequantise_f32(
    dev: &CudaDevice,
    dtype: crate::tensor::quantized::GgmlDType,
    blob: &CudaView<'_, u8>,
    total: usize,
) -> Result<CudaSlice<f32>> {
    let stream = dev.stream();
    Ok(match dtype {
        crate::tensor::quantized::GgmlDType::Q8_0 => dequantize_q8_0_view(dev, blob, total)?,
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
            b.arg(blob);
            b.arg(&dst);
            b.arg(&total_i);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("dequant_q2_k_f32 launch: {e}")))?;
            dst
        }
        crate::tensor::quantized::GgmlDType::Q6K => {
            let dst = with_oom_retry(dev, "dequant q6_K", || unsafe {
                stream.alloc::<f32>(total)
            })?;
            let func = dev.quantized_fn("dequant_q6_k_f32")?;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let total_i = total as i64;
            let mut b = stream.launch_builder(&func);
            b.arg(blob);
            b.arg(&dst);
            b.arg(&total_i);
            unsafe { b.launch(cfg) }.map_err(|e| Error(format!("dequant_q6_k_f32 launch: {e}")))?;
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
            b.arg(blob);
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
    })
}

fn gpu_quant_linear_once(
    dev: &CudaDevice,
    dtype: crate::tensor::quantized::GgmlDType,
    blocks: &[u8],
    (out, inp): (usize, usize),
    x: &[f32],
) -> Result<Vec<f32>> {
    let total = out * inp;
    let want = total / dtype.block_size() * dtype.type_size();
    if inp == 0
        || total % dtype.block_size() != 0
        || blocks.len() != want
        || !x.len().is_multiple_of(inp)
    {
        return Err(Error(format!(
            "{dtype:?} linear: {} block bytes ({want} wanted), {} inputs for [{out}, {inp}]",
            blocks.len(),
            x.len()
        )));
    }
    let n = x.len() / inp;
    let stream = dev.stream();
    let upload_started = std::time::Instant::now();
    let blob = stream
        .clone_htod(blocks)
        .map_err(|e| alloc_err("quant linear upload", e))?;
    let xs = stream
        .clone_htod(x)
        .map_err(|e| alloc_err("quant linear upload", e))?;
    crate::inference::offload::prof_record(
        "q linear upload",
        upload_started.elapsed().as_nanos() as u64,
    );
    let launch_started = std::time::Instant::now();
    let w = dequantise_f32(dev, dtype, &blob.as_view(), total)?;
    crate::inference::offload::prof_record(
        "q linear dequant launch",
        launch_started.elapsed().as_nanos() as u64,
    );
    let gemm_started = std::time::Instant::now();
    let y = matmul_nt_f32(dev, &xs, &w, 1, n, inp, out)?;
    let r = stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("quant linear download: {e}")));
    crate::inference::offload::prof_record(
        "q linear gemm and download",
        gemm_started.elapsed().as_nanos() as u64,
    );
    r
}

/// The copy stream of a card, made once: weights go up on it while the compute stream runs.
fn copy_stream(
    dev: &CudaDevice,
) -> Result<std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaStream>> {
    static COPY: std::sync::Mutex<
        Vec<(
            usize,
            std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
        )>,
    > = std::sync::Mutex::new(Vec::new());
    let mut g = COPY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = g.iter().find(|(o, _)| *o == dev.ordinal()) {
        return Ok(s.clone());
    }
    let s = dev
        .ctx
        .new_stream()
        .map_err(|e| Error(format!("copy stream: {e:?}")))?;
    g.push((dev.ordinal(), s.clone()));
    Ok(s)
}

/// A page-locked host buffer the weights of an expert are staged through: copied into here from
/// the mapped file, then sent to the card by a copy that runs beside the card's other work.
struct Pinned {
    ptr: *mut std::ffi::c_void,
    cap: usize,
}

// Safety: the buffer is owned by one holder at a time and freed only with the process.
unsafe impl Send for Pinned {}

static PINNED: std::sync::Mutex<Vec<Pinned>> = std::sync::Mutex::new(Vec::new());

/// A layer's expert weights read ahead into page-locked memory while the layer before computes.
/// Two slabs alternate, one being filled for the next layer while the other is consumed by this
/// one; an upload that finds its weight there copies from it directly, with no read and no copy
/// into a staging buffer of its own.
struct SlabInner {
    ptr: usize,
    cap: usize,
    /// Bumped when the slab is refilled: readers of an earlier fill stop, and its entries vanish.
    generation: u64,
    /// Source address of a weight to its place in the slab and whether it has arrived.
    entries: std::collections::HashMap<usize, (usize, usize, bool)>,
    /// Reader threads still running on this slab.
    active: usize,
}

struct Slab {
    inner: std::sync::Mutex<SlabInner>,
    cv: std::sync::Condvar,
}

fn slabs() -> &'static [Slab; 2] {
    static S: std::sync::OnceLock<[Slab; 2]> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let one = || Slab {
            inner: std::sync::Mutex::new(SlabInner {
                ptr: 0,
                cap: 0,
                generation: 0,
                entries: Default::default(),
                active: 0,
            }),
            cv: std::sync::Condvar::new(),
        };
        [one(), one()]
    })
}

static SLAB_TURN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Start reading the weights at `spans` = (address, length) of mapped files into the next slab,
/// on `readers` threads this call does not wait for, in the order given and as many as the host's
/// free memory allows a quarter of. Answers whether the slab took them; `false` leaves the reads
/// to whoever asks for the weights.
/// The file of `f` opened again for direct I/O, kept per file: `None` where it cannot be.
#[cfg(target_os = "linux")]
fn direct_file(f: &std::fs::File) -> Option<std::sync::Arc<std::fs::File>> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;
    type Opened = std::collections::HashMap<i32, Option<std::sync::Arc<std::fs::File>>>;
    static OPENED: std::sync::Mutex<Option<Opened>> = std::sync::Mutex::new(None);
    let fd = f.as_raw_fd();
    let mut g = OPENED.lock().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(Default::default)
        .entry(fd)
        .or_insert_with(|| {
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(format!("/proc/self/fd/{fd}"))
                .ok()
                .map(std::sync::Arc::new)
        })
        .clone()
}

/// Read at least `need` bytes of `f` from `off` into `buf` with direct I/O; `off`, `buf`'s
/// address and length are on pages. False when direct I/O is refused or the file ends first.
fn read_direct(f: &std::fs::File, buf: &mut [u8], off: u64, need: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::FileExt;
        let Some(d) = direct_file(f) else {
            return false;
        };
        let mut done = 0usize;
        while done < need {
            match d.read_at(&mut buf[done..], off + done as u64) {
                Ok(0) | Err(_) => return false,
                Ok(n) => done += n,
            }
        }
        true
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (f, buf, off, need);
        false
    }
}

/// Whether a slab has taken the weight at `addr`, `len` bytes, read or not: its bytes are, or
/// are about to be, in page-locked memory, and a read of the mapping would read them again.
pub fn is_prestaged(addr: usize, len: usize) -> bool {
    slabs().iter().any(|slab| {
        let g = slab.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.entries.get(&addr).is_some_and(|&(_, l, _)| l == len)
    })
}

pub fn prestage_weights(spans: Vec<(usize, usize)>, readers: usize) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        if spans.is_empty() {
            return false;
        }
        // PLACEMENT-EXEMPT: sizes a host staging area, not a device placement. What the kernel
        // calls available counts the page cache it can give back, which free pages do not.
        let avail = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| {
                m.lines()
                    .find_map(|l| l.strip_prefix("MemAvailable:"))
                    .and_then(|v| v.split_whitespace().next()?.parse::<usize>().ok())
            })
            .map_or(0, |kb| kb * 1024);
        let budget = avail / 4;
        // Each weight is read with direct I/O, around the page-cache, when its file allows it:
        // the read then starts and ends on a page, which the slab leaves room for, and the
        // weight's bytes sit `delta` past the read's start.
        let page = crate::tensor::mapped::page_size().unwrap_or(1);
        let mut taken = 0usize;
        let mut chosen: Vec<(usize, usize, usize)> = Vec::new();
        for &(addr, len) in &spans {
            let delta = crate::tensor::mapped::file_span(addr, len)
                .map_or(0, |(_, off)| off as usize % page);
            let start = taken.div_ceil(page) * page;
            let span = (delta + len).div_ceil(page) * page;
            if start + span > budget {
                break;
            }
            chosen.push((addr, len, start + delta));
            taken = start + span;
        }
        if chosen.is_empty() {
            return false;
        }
        let turn = SLAB_TURN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2;
        let slab = &slabs()[turn];
        let generation = {
            let mut g = slab.inner.lock().unwrap_or_else(|e| e.into_inner());
            // Readers of the fill before stop at their next weight; wait for them, since the
            // slab may be about to move.
            g.generation += 1;
            g.entries.clear();
            while g.active > 0 {
                g = slab.cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            if g.cap < taken {
                // Grown with headroom up to the budget: the layers' fills differ by a few
                // experts, and a page-locked allocation costs far more than the copies it serves.
                let want = (taken + taken / 2).min(budget).max(taken);
                if g.ptr != 0 {
                    let _ = unsafe {
                        crate::tensor::gpu_runtime::driver::result::free_host(
                            g.ptr as *mut std::ffi::c_void,
                        )
                    };
                    g.ptr = 0;
                    g.cap = 0;
                }
                match unsafe { crate::tensor::gpu_runtime::driver::result::malloc_host(want, 0) } {
                    Ok(p) => {
                        g.ptr = p as usize;
                        g.cap = want;
                    }
                    Err(_) => return false,
                }
            }
            for &(addr, len, at) in &chosen {
                g.entries.insert(addr, (at, len, false));
            }
            g.active = readers.max(1);
            g.generation
        };
        let chosen = std::sync::Arc::new(chosen);
        let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..readers.max(1) {
            let (chosen, next) = (chosen.clone(), next.clone());
            std::thread::spawn(move || {
                let slab = &slabs()[turn];
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(addr, len, at)) = chosen.get(i) else {
                        break;
                    };
                    let base = {
                        let g = slab.inner.lock().unwrap_or_else(|e| e.into_inner());
                        if g.generation != generation {
                            break;
                        }
                        g.ptr
                    };
                    let read = crate::tensor::mapped::file_span(addr, len).and_then(|(f, off)| {
                        let page = crate::tensor::mapped::page_size().unwrap_or(1);
                        let delta = off as usize % page;
                        let span = (delta + len).div_ceil(page) * page;
                        // Safety: the chooser reserved `span` bytes of the slab from
                        // `at - delta`, a page boundary, for this weight alone.
                        let whole = unsafe {
                            std::slice::from_raw_parts_mut((base + at - delta) as *mut u8, span)
                        };
                        if read_direct(&f, whole, off - delta as u64, delta + len) {
                            return Some(());
                        }
                        let buf =
                            unsafe { std::slice::from_raw_parts_mut((base + at) as *mut u8, len) };
                        f.read_exact_at(buf, off).ok()
                    });
                    let mut g = slab.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if g.generation == generation {
                        if read.is_some() {
                            if let Some(e) = g.entries.get_mut(&addr) {
                                e.2 = true;
                            }
                        } else {
                            g.entries.remove(&addr);
                        }
                    }
                    slab.cv.notify_all();
                }
                let mut g = slab.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.active -= 1;
                slab.cv.notify_all();
            });
        }
        true
    }
    #[cfg(not(unix))]
    {
        let _ = (spans, readers);
        false
    }
}

/// The page-locked address of the weight at `addr`, `len` bytes, once a slab has read it in;
/// waits for a read that has begun. `None` when no slab holds it.
fn staged_weight(addr: usize, len: usize) -> Option<usize> {
    let started = std::time::Instant::now();
    let found = staged_weight_wait(addr, len);
    if found.is_some() {
        crate::inference::offload::prof_record("slab wait", started.elapsed().as_nanos() as u64);
    }
    found
}

fn staged_weight_wait(addr: usize, len: usize) -> Option<usize> {
    for slab in slabs() {
        let mut g = slab.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match g.entries.get(&addr) {
                Some(&(at, l, true)) if l == len => return Some(g.ptr + at),
                Some(&(_, l, false)) if l == len => {
                    g = slab.cv.wait(g).unwrap_or_else(|e| e.into_inner());
                }
                _ => break,
            }
        }
    }
    None
}

/// Staging buffers in use by one expert's products. Dropping them waits for the card's stream,
/// so a buffer goes back to the pool only once no copy still reads it.
struct Staging<'a> {
    stream: &'a std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    /// The card's copy stream: uploads run there beside the compute stream's kernels, and the
    /// compute stream waits for each one's event before it reads the weight.
    copy: std::sync::Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    held: Vec<Pinned>,
    /// One per upload, recorded behind its copy: waited on instead of the whole stream, which
    /// other threads' work shares.
    copied: Vec<crate::tensor::gpu_runtime::driver::CudaEvent>,
}

impl Drop for Staging<'_> {
    fn drop(&mut self) {
        for e in &self.copied {
            let _ = e.synchronize();
        }
        PINNED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .append(&mut self.held);
    }
}

impl<'a> Staging<'a> {
    /// `bytes` on the card: read into the page cache and mapped in one pass, copied to a
    /// page-locked buffer, and sent asynchronously on the stream.
    fn upload(&mut self, dev: &CudaDevice, bytes: &[u8]) -> Result<CudaSlice<u8>> {
        self.upload_inner(dev, bytes)
    }

    fn upload_inner(&mut self, dev: &CudaDevice, bytes: &[u8]) -> Result<CudaSlice<u8>> {
        // Already in page-locked memory, read ahead: sent from there.
        if let Some(src) = staged_weight(bytes.as_ptr() as usize, bytes.len()) {
            return self.send(dev, src as *const std::ffi::c_void, bytes.len());
        }
        let pin = {
            let mut pool = PINNED.lock().unwrap_or_else(|e| e.into_inner());
            match pool.iter().position(|p| p.cap >= bytes.len()) {
                Some(i) => pool.swap_remove(i),
                None => {
                    let cap = bytes.len().next_power_of_two().max(1 << 20);
                    let ptr =
                        unsafe { crate::tensor::gpu_runtime::driver::result::malloc_host(cap, 0) }
                            .map_err(|e| Error(format!("pinned staging buffer: {e}")))?;
                    Pinned { ptr, cap }
                }
            }
        };
        // Read from the file into the page-locked buffer in one copy: through the mapping, each
        // page of the weight would first be mapped into the process, which costs more than the
        // copy. A weight that is not in a registered mapping is copied from its bytes.
        let t_copy = std::time::Instant::now();
        let from_file = {
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                crate::tensor::mapped::file_span(bytes.as_ptr() as usize, bytes.len()).and_then(
                    |(file, off)| {
                        let buf = unsafe {
                            std::slice::from_raw_parts_mut(pin.ptr as *mut u8, bytes.len())
                        };
                        file.read_exact_at(buf, off).ok()
                    },
                )
            }
            #[cfg(not(unix))]
            {
                None::<()>
            }
        };
        if from_file.is_none() {
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), pin.ptr as *mut u8, bytes.len())
            };
        }
        crate::inference::offload::prof_record("staged copy", t_copy.elapsed().as_nanos() as u64);
        let ptr = pin.ptr;
        self.held.push(pin);
        self.send(dev, ptr, bytes.len())
    }

    /// `len` page-locked bytes at `src` sent to a new buffer on the card. The buffer belongs to
    /// the compute stream, so it is freed behind the kernels that read it; the copy runs on the
    /// copy stream, after the allocation, and the compute stream waits for it.
    fn send(
        &mut self,
        dev: &CudaDevice,
        src: *const std::ffi::c_void,
        len: usize,
    ) -> Result<CudaSlice<u8>> {
        let (stream, copy) = (self.stream, &self.copy);
        let dst = with_oom_retry(dev, "expert weight upload", || unsafe {
            stream.alloc::<u8>(len)
        })?;
        let allocated = stream
            .record_event(None)
            .map_err(|e| Error(format!("staged upload: record: {e:?}")))?;
        copy.wait(&allocated)
            .map_err(|e| Error(format!("staged upload: wait: {e:?}")))?;
        copy.context()
            .bind_to_thread()
            .map_err(|e| Error(format!("staged upload: bind context: {e:?}")))?;
        use crate::tensor::gpu_runtime::driver::DevicePtr;
        let (dptr, guard) = dst.device_ptr(stream);
        let r = unsafe {
            crate::tensor::gpu_runtime::driver::sys::cuMemcpyHtoDAsync_v2(
                dptr,
                src,
                len,
                copy.cu_stream(),
            )
        };
        drop(guard);
        if r != crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS {
            return Err(Error(format!("staged upload: {r:?}")));
        }
        let done = copy
            .record_event(None)
            .map_err(|e| Error(format!("staged upload: record: {e:?}")))?;
        stream
            .wait(&done)
            .map_err(|e| Error(format!("staged upload: wait: {e:?}")))?;
        self.copied.push(done);
        Ok(dst)
    }
}

/// Where a weight of an expert is: in the mapped file, to be staged over, or already on the card.
pub enum ExpertWeight<'a> {
    Host(&'a [u8]),
    Card(std::sync::Arc<CudaSlice<u8>>),
}

/// The three weights of an expert on the card, staged over the ones that are not there yet.
fn expert_weights_on_card(
    dev: &CudaDevice,
    staging: &mut Staging,
    weights: &[ExpertWeight; 3],
) -> Result<[std::sync::Arc<CudaSlice<u8>>; 3]> {
    let mut on = Vec::with_capacity(3);
    for w in weights {
        on.push(match w {
            ExpertWeight::Card(blob) => blob.clone(),
            ExpertWeight::Host(bytes) => std::sync::Arc::new(staging.upload(dev, bytes)?),
        });
    }
    let [g, u, d]: [std::sync::Arc<CudaSlice<u8>>; 3] = on
        .try_into()
        .map_err(|_| Error("an expert has three weights".into()))?;
    Ok([g, u, d])
}

/// One expert over a prompt's rows on the card, whole: `x` [n, dim] through the IQ2_XXS gate and
/// up weights and the Q2_K down weight (`weights` in that order, staged over through page-locked
/// memory where they are not on the card), the host's bits. The rows entering the down
/// projection [n, inter] and the output [n, dim] come back together.
pub fn gpu_expert_tokens(
    dev: &CudaDevice,
    weights: [ExpertWeight; 3],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (inter, dim): (usize, usize),
    x: &[f32],
    limit: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if dim == 0 || !x.len().is_multiple_of(dim) {
        return Err(Error(format!(
            "expert tokens: {} inputs for rows of {dim}",
            x.len()
        )));
    }
    let n = x.len() / dim;
    let stream = dev.stream();
    let mut staging = Staging {
        stream,
        copy: copy_stream(dev)?,
        held: Vec::new(),
        copied: Vec::new(),
    };
    let xq = q8k32_rows_to_card(dev, x)?;
    let [g, u, d] = expert_weights_on_card(dev, &mut staging, &weights)?;
    let instances: Vec<(usize, usize)> = (0..n).map(|r| (0, r)).collect();
    let (h, y) = gpu_experts_exact(
        dev,
        &[(&g, &u, &d)],
        tables,
        (dim, inter),
        &xq,
        &instances,
        limit,
    )?;
    let t_wait = std::time::Instant::now();
    let h_host = stream
        .clone_dtoh(&h)
        .map_err(|e| Error(format!("expert tokens download: {e}")))?;
    let out_host = stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("expert tokens download: {e}")))?;
    crate::inference::offload::prof_record("staged card wait", t_wait.elapsed().as_nanos() as u64);
    Ok((h_host, out_host))
}

/// A layer's routed experts' token context on a card: the layer's input rows [n, dim] as the
/// q8_K32 rows the expert kernels read, sent once for every expert of the layer the card runs.
pub struct LayerBuffers {
    xq: CudaSlice<u8>,
    n: usize,
    dim: usize,
}

/// Open a layer's context on the card from its input rows `x` [n, dim].
pub fn gpu_layer_begin(dev: &CudaDevice, x: &[f32], dim: usize) -> Result<LayerBuffers> {
    if dim == 0 || !x.len().is_multiple_of(dim) {
        return Err(Error(format!(
            "layer context: {} inputs for rows of {dim}",
            x.len()
        )));
    }
    Ok(LayerBuffers {
        xq: q8k32_rows_to_card(dev, x)?,
        n: x.len() / dim,
        dim,
    })
}

/// One expert over its rows of the layer on the card: the rows `toks` are read from the layer's
/// input where it lies, put through the expert as `gpu_expert_tokens` does, and their outputs
/// [toks, dim] come back in the order of `toks`, for the host to add in the order it adds every
/// expert in.
pub fn gpu_expert_layer_rows(
    dev: &CudaDevice,
    weights: [ExpertWeight; 3],
    tables: (&CudaSlice<u8>, &CudaSlice<u8>),
    (inter, dim): (usize, usize),
    layer: &LayerBuffers,
    toks: &[u32],
    limit: f32,
) -> Result<Vec<f32>> {
    if dim != layer.dim || toks.is_empty() || toks.iter().any(|&t| t as usize >= layer.n) {
        return Err(Error(format!(
            "expert rows: {} rows of {dim} against a layer of {} rows of {}",
            toks.len(),
            layer.n,
            layer.dim
        )));
    }
    let stream = dev.stream();
    let mut staging = Staging {
        stream,
        copy: copy_stream(dev)?,
        held: Vec::new(),
        copied: Vec::new(),
    };
    let [g, u, d] = expert_weights_on_card(dev, &mut staging, &weights)?;
    let instances: Vec<(usize, usize)> = toks.iter().map(|&t| (0, t as usize)).collect();
    let (_h, y) = gpu_experts_exact(
        dev,
        &[(&g, &u, &d)],
        tables,
        (dim, inter),
        &layer.xq,
        &instances,
        limit,
    )?;
    let sync_started = std::time::Instant::now();
    let out = stream
        .clone_dtoh(&y)
        .map_err(|e| Error(format!("expert rows download: {e}")))?;
    crate::inference::offload::prof_record(
        "expert card wait",
        sync_started.elapsed().as_nanos() as u64,
    );
    Ok(out)
}

/// A quantised weight [rows, inp] as it lies in the file, `dtype` its format.
pub struct QuantWeight<'a> {
    pub dtype: crate::tensor::quantized::GgmlDType,
    pub bytes: &'a [u8],
    pub rows: usize,
    pub inp: usize,
}

/// What a prompt's attention needs on the card from the shared query latent on.
pub struct PrefillAttention<'a> {
    /// The query latent [n, q_lora] and the projection of it to the heads, [heads * hd, q_lora].
    pub qr: &'a [f32],
    pub wq_b: QuantWeight<'a>,
    /// The rope tables of the queries' positions, [n, rd / 2].
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    pub heads: usize,
    pub hd: usize,
    pub rd: usize,
    /// The window and compressed keys [rows, hd], the per-head sink, and for each query the
    /// `topk` key rows it reads (negative: none).
    pub kv: &'a [f32],
    pub sink: &'a [f32],
    pub idxs: &'a [i32],
    pub topk: usize,
    pub scale: f32,
    /// The grouped output projection: `wo_a` [groups * o_lora, heads * hd / groups], then `wo_b`
    /// [dim, groups * o_lora].
    pub wo_a: QuantWeight<'a>,
    pub wo_b: QuantWeight<'a>,
    pub groups: usize,
}

/// `x` [n, inp] through rows `row0 .. row0 + rows` of the weight held on the card as `blob`, as
/// `[n, rows]`: the rows are expanded to f32 a tile at a time, so a weight wider than the card has
/// room for still goes through.
fn linear_tiled(
    dev: &CudaDevice,
    tile_bytes: usize,
    dtype: crate::tensor::quantized::GgmlDType,
    blob: &CudaSlice<u8>,
    (row0, rows, inp): (usize, usize, usize),
    x: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    let stream = dev.stream();
    let row_bytes = inp / dtype.block_size() * dtype.type_size();
    let tile = (tile_bytes / (inp * 4)).clamp(32, rows.max(32)) / 32 * 32;
    let tile = tile.max(1).min(rows);
    let one = |start: usize, count: usize| -> Result<CudaSlice<f32>> {
        let view = blob.slice((row0 + start) * row_bytes..(row0 + start + count) * row_bytes);
        let w = dequantise_f32(dev, dtype, &view, count * inp)?;
        matmul_nt_f32(dev, x, &w, 1, n, inp, count)
    };
    if tile >= rows {
        return one(0, rows);
    }
    let out = with_oom_retry(dev, "linear tiles", || unsafe {
        stream.alloc::<f32>(n * rows)
    })?;
    let func = dev.quantized_fn("put_cols_f32")?;
    let mut start = 0;
    while start < rows {
        let count = tile.min(rows - start);
        let part = one(start, count)?;
        let cfg = LaunchConfig {
            grid_dim: ((n * count).div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (total, w, dst_w, off) = ((n * count) as i64, count as i64, rows as i64, start as i64);
        let mut b = stream.launch_builder(&func);
        b.arg(&part);
        b.arg(&out);
        b.arg(&total);
        b.arg(&w);
        b.arg(&dst_w);
        b.arg(&off);
        unsafe { b.launch(cfg) }.map_err(|e| Error(format!("put_cols_f32 launch: {e}")))?;
        start += count;
    }
    Ok(out)
}

/// Queries a prompt's attention takes on a card at once. Fixed rather than sized from the memory
/// free at the call: a product's bits follow how many rows it takes, so a chunk that followed the
/// free memory would make a prompt's answer follow what else the card holds.
pub const ATTENTION_QUERY_CHUNK: usize = 256;

/// A prompt's attention on the card from the shared query latent on, in chunks of queries: the
/// queries from the latent, rotated, over the gathered keys with the sink, rotated back, and
/// through the grouped output projection; `[n, dim]` comes back. Nothing but the latent and the
/// result crosses, where each step through the host moved a copy of the queries or the outputs.
pub fn gpu_prefill_attention(dev: &CudaDevice, a: &PrefillAttention) -> Result<Vec<f32>> {
    gpu_prefill_attention_with(dev, a, usize::MAX, QUANT_LINEAR_TILE_BYTES)
}

/// `gpu_prefill_attention` with the chunk of queries capped at `max_chunk` and the tile of an
/// expanded weight at `tile_bytes`.
pub fn gpu_prefill_attention_with(
    dev: &CudaDevice,
    a: &PrefillAttention,
    max_chunk: usize,
    tile_bytes: usize,
) -> Result<Vec<f32>> {
    let (h, hd, rd, topk) = (a.heads, a.hd, a.rd, a.topk);
    let q_lora = a.wq_b.inp;
    let n = a.qr.len() / q_lora.max(1);
    let (p, groups) = (h * hd / a.groups.max(1), a.groups);
    let r = a.wo_a.rows / groups.max(1);
    let dim = a.wo_b.rows;
    if a.qr.len() != n * q_lora
        || a.wq_b.rows != h * hd
        || a.cos.len() != n * (rd / 2)
        || a.sin.len() != a.cos.len()
        || a.idxs.len() != n * topk
        || a.wo_a.inp != p
        || a.wo_b.inp != groups * r
        || !a.kv.len().is_multiple_of(hd)
        || a.sink.len() != h
    {
        return Err(Error(format!(
            "prefill attention: shapes disagree ({n} tokens, {} latent values, {} cos values)",
            a.qr.len(),
            a.cos.len()
        )));
    }
    let keys_n = a.kv.len() / hd;
    if let Some(&bad) = a.idxs.iter().find(|&&i| i >= keys_n as i32) {
        return Err(Error(format!(
            "prefill attention: slot {bad} past {keys_n} keys"
        )));
    }
    let stream = dev.stream();
    let mut staging = Staging {
        stream,
        copy: copy_stream(dev)?,
        held: Vec::new(),
        copied: Vec::new(),
    };
    let up = |e| alloc_err("prefill attention upload", e);
    let blobs = [
        staging.upload(dev, a.wq_b.bytes)?,
        staging.upload(dev, a.wo_a.bytes)?,
        staging.upload(dev, a.wo_b.bytes)?,
    ];
    let (kvd, sd, idd) = (
        stream.clone_htod(a.kv).map_err(up)?,
        stream.clone_htod(a.sink).map_err(up)?,
        stream.clone_htod(a.idxs).map_err(up)?,
    );
    let neg_sin: Vec<f32> = a.sin.iter().map(|v| -v).collect();
    let (cosd, sind, nsind) = (
        stream.clone_htod(a.cos).map_err(up)?,
        stream.clone_htod(a.sin).map_err(up)?,
        stream.clone_htod(&neg_sin).map_err(up)?,
    );
    // Queries go through in chunks of a fixed count (see `ATTENTION_QUERY_CHUNK`).
    let chunk = ATTENTION_QUERY_CHUNK.min(max_chunk).clamp(1, n.max(1));
    let launch1 = |name: &str, total: usize| -> Result<(CudaFunction, LaunchConfig)> {
        Ok((
            dev.quantized_fn(name)?,
            LaunchConfig {
                grid_dim: (total.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
        ))
    };
    let (func_gather, func_softmax) = (
        dev.quantized_fn("gather_slots_f32")?,
        dev.quantized_fn("sink_softmax_f32")?,
    );
    let mut result = vec![0f32; n * dim];
    for start in (0..n).step_by(chunk) {
        let count = chunk.min(n - start);
        let qr = stream
            .clone_htod(&a.qr[start * q_lora..(start + count) * q_lora])
            .map_err(up)?;
        // The queries, rotated.
        let q = linear_tiled(
            dev,
            tile_bytes,
            a.wq_b.dtype,
            &blobs[0],
            (0, h * hd, q_lora),
            &qr,
            count,
        )?;
        let rope = |buf: &CudaSlice<f32>, sin: &CudaSlice<f32>| -> Result<()> {
            let pairs = count * h * (rd / 2);
            let (f, cfg) = launch1("rope_tail_f32", pairs)?;
            let (tab_c, tab_s) = (
                cosd.slice(start * (rd / 2)..),
                sin.slice(start * (rd / 2)..),
            );
            let (pairs_i, h_i, hd_i, rd_i) = (pairs as i64, h as i64, hd as i64, rd as i64);
            let mut b = stream.launch_builder(&f);
            b.arg(buf);
            b.arg(&tab_c);
            b.arg(&tab_s);
            b.arg(&pairs_i);
            b.arg(&h_i);
            b.arg(&hd_i);
            b.arg(&rd_i);
            unsafe { b.launch(cfg) }
                .map_err(|e| Error(format!("rope_tail_f32 launch: {e}")))
                .map(|_| ())
        };
        rope(&q, &sind)?;
        // Sparse attention over the gathered keys: [count, h, hd] against [count, topk, hd].
        let keys = with_oom_retry(dev, "prefill attention keys", || unsafe {
            stream.alloc::<f32>(count * topk * hd)
        })?;
        let cfg = |m: usize| LaunchConfig {
            grid_dim: (m.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (start_i, count_i, topk_i, d_i, h_i) =
            (start as i64, count as i64, topk as i64, hd as i64, h as i64);
        let mut g = stream.launch_builder(&func_gather);
        g.arg(&kvd);
        g.arg(&idd);
        g.arg(&keys);
        g.arg(&start_i);
        g.arg(&count_i);
        g.arg(&topk_i);
        g.arg(&d_i);
        unsafe { g.launch(cfg(count * topk)) }
            .map_err(|e| Error(format!("gather_slots_f32 launch: {e}")))?;
        let logits = matmul_nt_f32(dev, &q, &keys, count, h, hd, topk)?;
        let mut sm = stream.launch_builder(&func_softmax);
        sm.arg(&logits);
        sm.arg(&sd);
        sm.arg(&idd);
        sm.arg(&start_i);
        sm.arg(&count_i);
        sm.arg(&h_i);
        sm.arg(&topk_i);
        sm.arg(&a.scale);
        unsafe { sm.launch(cfg(count * h)) }
            .map_err(|e| Error(format!("sink_softmax_f32 launch: {e}")))?;
        let o = matmul_f32(dev, &logits, &keys, count, h, topk, hd)?;
        // The output rotated back, then the grouped projection.
        rope(&o, &nsind)?;
        let wide = with_oom_retry(dev, "prefill attention groups", || unsafe {
            stream.alloc::<f32>(count * groups * r)
        })?;
        let slice = with_oom_retry(dev, "prefill attention group", || unsafe {
            stream.alloc::<f32>(count * p)
        })?;
        let (take, put) = (
            dev.quantized_fn("take_cols_f32")?,
            dev.quantized_fn("put_cols_f32")?,
        );
        for gi in 0..groups {
            let (tot, w, sw, off) = (
                (count * p) as i64,
                p as i64,
                (h * hd) as i64,
                (gi * p) as i64,
            );
            let mut b = stream.launch_builder(&take);
            b.arg(&o);
            b.arg(&slice);
            b.arg(&tot);
            b.arg(&w);
            b.arg(&sw);
            b.arg(&off);
            unsafe { b.launch(cfg(count * p)) }
                .map_err(|e| Error(format!("take_cols_f32 launch: {e}")))?;
            let y = linear_tiled(
                dev,
                tile_bytes,
                a.wo_a.dtype,
                &blobs[1],
                (gi * r, r, p),
                &slice,
                count,
            )?;
            let (tot, w, dw, off) = (
                (count * r) as i64,
                r as i64,
                (groups * r) as i64,
                (gi * r) as i64,
            );
            let mut b = stream.launch_builder(&put);
            b.arg(&y);
            b.arg(&wide);
            b.arg(&tot);
            b.arg(&w);
            b.arg(&dw);
            b.arg(&off);
            unsafe { b.launch(cfg(count * r)) }
                .map_err(|e| Error(format!("put_cols_f32 launch: {e}")))?;
        }
        let out = linear_tiled(
            dev,
            tile_bytes,
            a.wo_b.dtype,
            &blobs[2],
            (0, dim, groups * r),
            &wide,
            count,
        )?;
        stream
            .memcpy_dtoh(&out, &mut result[start * dim..(start + count) * dim])
            .map_err(|e| Error(format!("prefill attention download: {e}")))?;
    }
    Ok(result)
}

/// A prompt's residual stream `x` [n, hc, d] held on a card for the whole prompt, with the
/// hyper-connection steps that read and rewrite it. The coefficients are a few dozen values per
/// token and cross as such; the sublayers' inputs and outputs cross once each.
pub struct HcStreamDev {
    x: CudaSlice<f32>,
    n: usize,
    hc: usize,
    d: usize,
}

impl HcStreamDev {
    /// Tokens held times values per token.
    pub fn len(&self) -> usize {
        self.n * self.hc * self.d
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upload `x`, which holds `hc` copies of `d` values per token. `hc` is at most the widest the
    /// kernels hold in registers.
    pub fn open(dev: &CudaDevice, x: &[f32], hc: usize, d: usize) -> Result<Self> {
        if hc == 0 || hc > 8 || d == 0 || !x.len().is_multiple_of(hc * d) {
            return Err(Error(format!(
                "hc stream: {} values for {hc} x {d}",
                x.len()
            )));
        }
        let xd = dev
            .stream()
            .clone_htod(x)
            .map_err(|e| alloc_err("hc stream upload", e))?;
        Ok(Self {
            x: xd,
            n: x.len() / (hc * d),
            hc,
            d,
        })
    }

    /// The pre, post and comb coefficients of every token from the stream, as `[n, hc]`, `[n, hc]`
    /// and `[n, hc, hc]`. `func` is the projection [mix_hc, hc * d].
    pub fn mixes(
        &self,
        dev: &CudaDevice,
        func: &[f32],
        scale: &[f32],
        base: &[f32],
        iters: usize,
        norm_eps: f32,
        hc_eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (n, hc, flat) = (self.n, self.hc, self.hc * self.d);
        let mix_hc = (2 + hc) * hc;
        if func.len() != mix_hc * flat || scale.len() != 3 || base.len() != mix_hc {
            return Err(Error(format!(
                "hc mixes: {} projection values, {} scales, {} biases",
                func.len(),
                scale.len(),
                base.len()
            )));
        }
        let stream = dev.stream();
        let up = |e| alloc_err("hc mixes upload", e);
        let (fd, sd, bd) = (
            stream.clone_htod(func).map_err(up)?,
            stream.clone_htod(scale).map_err(up)?,
            stream.clone_htod(base).map_err(up)?,
        );
        let pre = stream.alloc_zeros::<f32>(n * hc).map_err(up)?;
        let post = stream.alloc_zeros::<f32>(n * hc).map_err(up)?;
        let comb = stream.alloc_zeros::<f32>(n * hc * hc).map_err(up)?;
        let func_k = dev.quantized_fn("hc_mixes_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (flat_i, hc_i, mix_i, it_i) = (flat as i64, hc as i32, mix_hc as i32, iters as i32);
        let mut b = stream.launch_builder(&func_k);
        b.arg(&self.x);
        b.arg(&fd);
        b.arg(&sd);
        b.arg(&bd);
        b.arg(&pre);
        b.arg(&post);
        b.arg(&comb);
        b.arg(&flat_i);
        b.arg(&hc_i);
        b.arg(&mix_i);
        b.arg(&it_i);
        b.arg(&norm_eps);
        b.arg(&hc_eps);
        unsafe { b.launch(cfg) }.map_err(|e| Error(format!("hc_mixes_f32 launch: {e}")))?;
        let down = |s: &CudaSlice<f32>| {
            stream
                .clone_dtoh(s)
                .map_err(|e| Error(format!("hc mixes download: {e}")))
        };
        Ok((down(&pre)?, down(&post)?, down(&comb)?))
    }

    /// The stream collapsed by `pre` [n, hc] and RMS-normalised by `w` [d]: the sublayer's input,
    /// `[n, d]`.
    pub fn pre_norm(&self, dev: &CudaDevice, pre: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>> {
        let (n, hc, d) = (self.n, self.hc, self.d);
        if pre.len() != n * hc || w.len() != d {
            return Err(Error(format!(
                "hc pre: {} weights for {n} x {hc}, {} norm values for {d}",
                pre.len(),
                w.len()
            )));
        }
        let stream = dev.stream();
        let up = |e| alloc_err("hc pre upload", e);
        let (pd, wd) = (
            stream.clone_htod(pre).map_err(up)?,
            stream.clone_htod(w).map_err(up)?,
        );
        let out = with_oom_retry(dev, "hc pre", || unsafe { stream.alloc::<f32>(n * d) })?;
        let func = dev.quantized_fn("hc_pre_norm_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (d * 4) as u32,
        };
        let (d_i, hc_i) = (d as i64, hc as i32);
        let mut b = stream.launch_builder(&func);
        b.arg(&self.x);
        b.arg(&pd);
        b.arg(&wd);
        b.arg(&out);
        b.arg(&d_i);
        b.arg(&hc_i);
        b.arg(&eps);
        unsafe { b.launch(cfg) }.map_err(|e| Error(format!("hc_pre_norm_f32 launch: {e}")))?;
        stream
            .clone_dtoh(&out)
            .map_err(|e| Error(format!("hc pre download: {e}")))
    }

    /// Rewrite the stream with the sublayer's output `xo` [n, d] and the coefficients `post`
    /// [n, hc], `comb` [n, hc, hc].
    pub fn post(&mut self, dev: &CudaDevice, xo: &[f32], post: &[f32], comb: &[f32]) -> Result<()> {
        let (n, hc, d) = (self.n, self.hc, self.d);
        if xo.len() != n * d || post.len() != n * hc || comb.len() != n * hc * hc {
            return Err(Error(format!(
                "hc post: {} outputs, {} post, {} comb for {n} x {hc} x {d}",
                xo.len(),
                post.len(),
                comb.len()
            )));
        }
        let stream = dev.stream();
        let up = |e| alloc_err("hc post upload", e);
        let (xd, pd, cd) = (
            stream.clone_htod(xo).map_err(up)?,
            stream.clone_htod(post).map_err(up)?,
            stream.clone_htod(comb).map_err(up)?,
        );
        let func = dev.quantized_fn("hc_post_f32")?;
        let total = n * d;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (total_i, d_i, hc_i) = (total as i64, d as i64, hc as i32);
        let mut b = stream.launch_builder(&func);
        b.arg(&xd);
        b.arg(&self.x);
        b.arg(&pd);
        b.arg(&cd);
        b.arg(&total_i);
        b.arg(&d_i);
        b.arg(&hc_i);
        unsafe { b.launch(cfg) }.map_err(|e| Error(format!("hc_post_f32 launch: {e}")))?;
        // The inputs above are freed when this returns; the launch is ordered before that.
        Ok(())
    }

    /// The stream of tokens `from..from + count`, `[count, hc, d]`.
    pub fn rows(&self, dev: &CudaDevice, from: usize, count: usize) -> Result<Vec<f32>> {
        let w = self.hc * self.d;
        if from + count > self.n {
            return Err(Error(format!("hc rows: {from} + {count} of {}", self.n)));
        }
        let stream = dev.stream();
        let view = self.x.slice(from * w..(from + count) * w);
        stream
            .clone_dtoh(&view)
            .map_err(|e| Error(format!("hc rows download: {e}")))
    }

    /// Replace the stream with `x`, the same shape.
    pub fn replace(&mut self, dev: &CudaDevice, x: &[f32]) -> Result<()> {
        if x.len() != self.n * self.hc * self.d {
            return Err(Error(format!("hc replace: {} values", x.len())));
        }
        self.x = dev
            .stream()
            .clone_htod(x)
            .map_err(|e| alloc_err("hc stream upload", e))?;
        Ok(())
    }
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
        || !kv.len().is_multiple_of(d)
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
    // Queries go through in chunks of a fixed count (see `ATTENTION_QUERY_CHUNK`).
    let chunk = ATTENTION_QUERY_CHUNK.clamp(1, s.max(1));
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
    q_off: usize,
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
    let dims = [
        s as i64,
        nh as i64,
        ihd as i64,
        g as i64,
        ratio as i64,
        q_off as i64,
    ];
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
    if !f32s.len().is_multiple_of(256)
        || !n_per_row.is_multiple_of(256)
        || importance.len() != n_per_row
    {
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
    if !f32s.len().is_multiple_of(256) || !n_per_row.is_multiple_of(256) {
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
fn dequantize_q8_0_view(
    dev: &CudaDevice,
    weight_blob: &CudaView<'_, u8>,
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

pub fn dequantize_q8_0_f32(
    dev: &CudaDevice,
    weight_blob: &CudaSlice<u8>,
    total: usize,
) -> Result<CudaSlice<f32>> {
    dequantize_q8_0_view(dev, &weight_blob.as_view(), total)
}

/// F16-output batched MMVQ (the fork's `plain_launcher_f16` family).
/// `smallk` selects the decode-only small-K kernels - the caller ensures `b_size == 1` and
/// that [`mmvq_smallk_warps`] applies.
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

pub(super) fn mmvq_t<
    T: crate::tensor::gpu_runtime::driver::DeviceRepr
        + crate::tensor::gpu_runtime::driver::ValidAsZeroBits,
>(
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
    if b_size == 0 || b_size > 8 {
        return Err(Error(format!("mmvq: b_size {b_size} not in 1..=8")));
    }
    let stream = dev.stream();
    let mut dst = with_oom_retry(dev, "gemv", || unsafe { stream.alloc::<T>(n * b_size) })?;
    launch_mmvq(
        dev,
        tag,
        dst_tag,
        weight_blob,
        &q8_1.slice(..),
        &mut dst.slice_mut(..),
        k,
        n,
        b_size,
        smallk,
        None,
    )?;
    Ok(dst)
}

/// One-row F32 mat-vec with `bias` (`n` F32 values) added to each output row in the kernel's
/// store: the same bits as the mat-vec followed by a bias add.
#[allow(clippy::too_many_arguments)]
pub fn mmvq_f32_bias(
    dev: &CudaDevice,
    tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &CudaSlice<u8>,
    k: usize,
    n: usize,
    smallk: bool,
    bias: &CudaView<'_, f32>,
) -> Result<CudaSlice<f32>> {
    if bias.len() < n {
        return Err(Error(format!(
            "mmvq bias: {} values for {n} rows",
            bias.len()
        )));
    }
    let stream = dev.stream();
    let mut dst = with_oom_retry(dev, "gemv", || unsafe { stream.alloc::<f32>(n) })?;
    launch_mmvq(
        dev,
        tag,
        "f32",
        weight_blob,
        &q8_1.slice(..),
        &mut dst.slice_mut(..),
        k,
        n,
        1,
        smallk,
        Some(bias),
    )?;
    Ok(dst)
}

/// The largest batch the mat-vec kernels run with the warp count of a single column
/// (`mmvq_nwarps_for` in `mmvq_gguf.cu`). Up to it, every column's blocks meet in the order
/// a one-row launch sums them, so each output row is bit-identical to that row alone.
pub const MMVQ_DECODE_ORDER_BATCH: usize = 4;

/// Batched MMVQ whose every output row equals a one-row launch over that row: the rows go
/// in launches of at most [`MMVQ_DECODE_ORDER_BATCH`], and a lone row takes the kernel a
/// one-row call would (`smallk_single`). `q8_1` holds `rows` quantised activation rows.
#[allow(clippy::too_many_arguments)]
pub fn mmvq_decode_order<
    T: crate::tensor::gpu_runtime::driver::DeviceRepr
        + crate::tensor::gpu_runtime::driver::ValidAsZeroBits,
>(
    dev: &CudaDevice,
    tag: &str,
    dst_tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &CudaSlice<u8>,
    k: usize,
    n: usize,
    rows: usize,
    smallk_single: bool,
) -> Result<CudaSlice<T>> {
    use crate::tensor::quantized::MATRIX_ROW_PADDING;
    let stream = dev.stream();
    let mut dst = with_oom_retry(dev, "gemv", || unsafe { stream.alloc::<T>(n * rows) })?;
    // block_q8_1 = 32 i8 + 2 f16 per 32 activations.
    let q8_row_bytes = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING / 32 * 36;
    let mut first = 0;
    while first < rows {
        let b = MMVQ_DECODE_ORDER_BATCH.min(rows - first);
        launch_mmvq(
            dev,
            tag,
            dst_tag,
            weight_blob,
            &q8_1.slice(first * q8_row_bytes..(first + b) * q8_row_bytes),
            &mut dst.slice_mut(first * n..(first + b) * n),
            k,
            n,
            b,
            b == 1 && smallk_single,
            None,
        )?;
        first += b;
    }
    Ok(dst)
}

/// A mat-vec format's block geometry as `mmvq_gguf.cu` walks it: values per block, 32-bit
/// ints per block, and ints one lane takes per dot product.
fn mmvq_geometry(tag: &str) -> Option<(usize, usize, usize)> {
    Some(match tag {
        "q4_0" | "q4_1" | "q5_0" | "q5_1" => (32, 4, 2),
        "q8_0" => (32, 8, 2),
        "q2_k" | "q3_k" => (256, 16, 1),
        "q4_k" | "q5_k" => (256, 32, 2),
        "q6_k" => (256, 32, 1),
        _ => return None,
    })
}

/// The fewest warps, of 1, 2 or 4, whose lanes cover a row of `k` values of format `tag` in
/// one pass: `(k / qk) * (qi / vdr)` lanes. Four when the row needs four warps or more.
pub fn mmvq_row_warps(tag: &str, k: usize) -> Option<usize> {
    let (qk, qi, vdr) = mmvq_geometry(tag)?;
    let lanes = (k / qk) * (qi / vdr);
    Some(
        [1usize, 2]
            .into_iter()
            .find(|&w| w * 32 >= lanes)
            .unwrap_or(4),
    )
}

/// The warps a one-row small-K mat-vec gives each output row, or `None` when a row takes
/// the four warps more than one pass.
///
/// When the four warps' 128 lanes cover a row in one pass, the small-K kernels apply: a block
/// of four warps serves 4 rows a group of [`mmvq_row_warps`], `n` dividing by the rows of a
/// block. The four-warp small-K kernel takes a row count that the narrower ones do not divide.
pub fn mmvq_smallk_warps(tag: &str, k: usize, n: usize) -> Option<usize> {
    let (qk, qi, vdr) = mmvq_geometry(tag)?;
    if (k / qk) * (qi / vdr) > 128 {
        return None;
    }
    let warps = mmvq_row_warps(tag, k)?;
    match n.is_multiple_of(4 * (4 / warps)) {
        true => Some(warps),
        false => n.is_multiple_of(4).then_some(4),
    }
}

/// One mat-vec launch of `b_size` activation rows into `dst`.
#[allow(clippy::too_many_arguments)]
fn launch_mmvq<T: crate::tensor::gpu_runtime::driver::DeviceRepr>(
    dev: &CudaDevice,
    tag: &str,
    dst_tag: &str,
    weight_blob: &CudaSlice<u8>,
    q8_1: &crate::tensor::gpu_runtime::driver::CudaView<'_, u8>,
    dst: &mut crate::tensor::gpu_runtime::driver::CudaViewMut<'_, T>,
    k: usize,
    n: usize,
    b_size: usize,
    smallk: bool,
    bias: Option<&CudaView<'_, f32>>,
) -> Result<()> {
    use crate::tensor::quantized::MATRIX_ROW_PADDING;
    if bias.is_some() && (b_size != 1 || dst_tag != "f32") {
        return Err(Error(format!(
            "mmvq bias takes one F32 row (got b_size={b_size}, dst {dst_tag})"
        )));
    }
    let smallk_warps = match smallk {
        true => match mmvq_smallk_warps(tag, k, n) {
            Some(warps) if b_size == 1 => warps,
            _ => {
                return Err(Error(format!(
                    "mmvq smallk requires b_size==1, a small k and n%4==0 (got b_size={b_size}, \
                     k={k}, n={n})"
                )))
            }
        },
        false => 4,
    };
    let with_bias = if bias.is_some() { "_bias" } else { "" };
    let kernel = match (smallk, smallk_warps) {
        (true, 4) => format!("mmvq_gguf_{tag}_{dst_tag}_plain_smallk{with_bias}_cuda1"),
        (true, warps) => {
            format!("mmvq_gguf_{tag}_{dst_tag}_plain_smallk_w{warps}{with_bias}_cuda1")
        }
        (false, _) => format!("mmvq_gguf_{tag}_{dst_tag}_plain{with_bias}_cuda{b_size}"),
    };
    let func = dev.mmvq_fn(&kernel)?;
    let stream = dev.stream();
    let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    let stride_col_y = (k_padded / 32) as i32;
    let stride_col_dst = n as i32;
    // Launch geometry mirrors the fork's C launchers exactly:
    // plain: rows_per_block = b<=1 ? 1 : 2, nwarps = b<=4 ? 4 : 2;
    // smallk: four rows to each group of `smallk_warps` warps, nwarps = 4.
    let rows_per_block: usize = if smallk {
        4 * (4 / smallk_warps)
    } else if b_size <= 1 {
        1
    } else {
        2
    };
    let nwarps: u32 = if smallk || b_size <= MMVQ_DECODE_ORDER_BATCH {
        4
    } else {
        2
    };
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
    b.arg(dst);
    b.arg(&k_i);
    b.arg(&n_i);
    b.arg(&stride_col_y);
    b.arg(&stride_col_dst);
    if let Some(bias) = bias {
        b.arg(bias);
    }
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("{kernel} launch: {e}")))?;
    Ok(())
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

/// The small-K mat-vecs and the gate and up pair against the four-warp one-row kernel.
///
/// A narrower row reads the same blocks per lane and sums in the same order, so the outputs
/// are the same bits: a decode that moves to them emits the same tokens.
#[cfg(test)]
mod small_k_bits {
    use super::*;
    use crate::tensor::quant_cpu;
    use crate::tensor::quantized::GgmlDType;

    fn values(len: usize, seed: u32, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((h >> 8) as f32 / (1 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }

    /// Formats and widths that give a row one, two and four warps, and a row of exactly the
    /// four warps' lanes.
    const CASES: [(GgmlDType, &str, usize); 7] = [
        (GgmlDType::Q4_0, "q4_0", 256),
        (GgmlDType::Q5_0, "q5_0", 896),
        (GgmlDType::Q8_0, "q8_0", 896),
        (GgmlDType::Q4K, "q4_k", 256),
        (GgmlDType::Q4K, "q4_k", 1024),
        (GgmlDType::Q6K, "q6_k", 512),
        (GgmlDType::Q4K, "q4_k", 2048),
    ];

    #[test]
    fn every_small_k_width_writes_the_one_row_kernel_bits() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the small-K mat-vec is NOT covered by this run");
            return;
        };
        let stream = dev.stream().clone();
        // 48 rows: whole blocks of 16, 8 and 4 rows, and a tail for the pair's guard.
        let n = 48;
        let mut widths = Vec::new();
        for (dtype, tag, k) in CASES {
            let w = quant_cpu::from_float_bytes(dtype, &values(n * k, 3, 0.6)).unwrap();
            let blob = stream.clone_htod(&w).unwrap();
            let x = stream.clone_htod(&values(k, 11, 2.0)).unwrap();
            let q81 = quantize_q8_1(&dev, &x, k).unwrap();
            let one = stream.clone_dtoh(&mmvq_f32(&dev, tag, &blob, &q81, k, n, 1, false).unwrap());
            let warps = mmvq_smallk_warps(tag, k, n)
                .unwrap_or_else(|| panic!("{tag} k={k}: no small-K kernel for a one-pass row"));
            let small =
                stream.clone_dtoh(&mmvq_f32(&dev, tag, &blob, &q81, k, n, 1, true).unwrap());
            let (one, small) = (one.unwrap(), small.unwrap());
            assert!(
                one.iter().any(|v| *v != 0.0),
                "{tag} k={k}: the one-row kernel wrote zeros"
            );
            for (row, (a, b)) in one.iter().zip(&small).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{tag} k={k} row {row}: {warps}-warp small-K {b} against one-row {a}"
                );
            }
            // The bias kernels: the sum, then the row's bias, as a separate add would.
            let bias = values(n, 17, 1.0);
            let bias_dev = stream.clone_htod(&bias).unwrap();
            for smallk in [false, true] {
                let y = mmvq_f32_bias(&dev, tag, &blob, &q81, k, n, smallk, &bias_dev.slice(..))
                    .unwrap();
                let y = stream.clone_dtoh(&y).unwrap();
                for (row, (a, got)) in one.iter().zip(&y).enumerate() {
                    let want = a + bias[row];
                    assert_eq!(
                        want.to_bits(),
                        got.to_bits(),
                        "{tag} k={k} row {row} small-K {smallk}: biased {got} against {want}"
                    );
                }
            }
            widths.push(warps);
        }
        for warps in [1, 2, 4] {
            assert!(widths.contains(&warps), "no case gives a row {warps} warps");
        }
    }

    #[test]
    fn the_pair_writes_silu_of_the_gate_times_the_up_rows() {
        use crate::tensor::kernel_ffi::{CudaDevice as FfiDevice, CudaStorage, Layout};
        use crate::tensor::quantized::QCudaStorage;
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the gate and up pair is NOT covered by this run");
            return;
        };
        let stream = dev.stream().clone();
        let silu = dev.fused_fn("fused_silu_mul_f32").unwrap();
        // An odd row count: the narrower pairs' last block holds a row past the end.
        let n = 45;
        for (dtype, tag, k) in CASES {
            let gate_w = quant_cpu::from_float_bytes(dtype, &values(n * k, 5, 0.3)).unwrap();
            let up_w = quant_cpu::from_float_bytes(dtype, &values(n * k, 9, 0.6)).unwrap();
            let ffi = FfiDevice(dev.clone());
            let gate_s = QCudaStorage::upload(&ffi, &gate_w, dtype).unwrap();
            let up_s = QCudaStorage::upload(&ffi, &up_w, dtype).unwrap();
            let xs = values(k, 13, 2.0);
            let x = stream.clone_htod(&xs).unwrap();
            let q81 = quantize_q8_1(&dev, &x, k).unwrap();
            let gate =
                mmvq_f32(&dev, tag, gate_s.weight_cuda_slice(), &q81, k, n, 1, false).unwrap();
            let up = mmvq_f32(&dev, tag, up_s.weight_cuda_slice(), &q81, k, n, 1, false).unwrap();
            let want = stream.alloc_zeros::<f32>(n).unwrap();
            let n_i = n as i32;
            let mut b = stream.launch_builder(&silu);
            b.arg(&gate).arg(&up).arg(&want).arg(&n_i);
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(cfg) }.unwrap();
            let want = stream.clone_dtoh(&want).unwrap();

            let rhs = CudaStorage::wrap_cuda_slice(x, ffi.clone());
            let layout = Layout::contiguous_with_offset(crate::tensor::Shape::from(vec![1, k]), 0);
            let shape = crate::tensor::Shape::from(vec![n, k]);
            let (out, _) = crate::tensor::quantized::fast_mmvq::try_fused_silu(
                &up_s, &gate_s, &shape, &rhs, &layout,
            )
            .unwrap()
            .unwrap_or_else(|| panic!("{tag} k={k}: the pair declined a same-format pair"));
            let got = stream
                .clone_dtoh(out.as_cuda_slice::<f32>().unwrap())
                .unwrap();
            for (row, (a, b)) in want.iter().zip(&got).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{tag} k={k} row {row}: pair {b} against the projections apart {a}"
                );
            }
        }
    }
}

/// What the one-row mat-vec kernels reach on a decode's shapes, in weight bytes a second.
///
/// Each shape runs the four-warp one-row kernel and every small-K width whose lanes cover the
/// row, on zeroed weights: the bytes moved are the format's, the values do not matter.
#[cfg(test)]
mod mat_vec_throughput {
    #[test]
    #[ignore = "measurement: needs a CUDA device"]
    fn what_each_one_row_kernel_reaches() {
        use crate::tensor::cuda::CudaDevice;
        use crate::tensor::gpu_runtime::driver::{LaunchConfig, PushKernelArg};
        use crate::tensor::quantized::{GgmlDType, MATRIX_ROW_PADDING};
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device");
            return;
        };
        let stream = dev.stream().clone();
        let ctx = stream.context();
        let shapes = [
            ("q5_0", GgmlDType::Q5_0, 896, 9728),
            ("q5_0", GgmlDType::Q5_0, 896, 896),
            ("q8_0", GgmlDType::Q8_0, 896, 151936),
            ("q8_0", GgmlDType::Q8_0, 1024, 151936),
            ("q4_k", GgmlDType::Q4K, 1024, 3072),
            ("q6_k", GgmlDType::Q6K, 1024, 3072),
            ("q4_k", GgmlDType::Q4K, 2048, 8192),
            ("q6_k", GgmlDType::Q6K, 4864, 896),
            ("q4_k", GgmlDType::Q4K, 4864, 896),
        ];
        for (tag, dtype, k, n) in shapes {
            let bytes = n * (k / dtype.block_size()) * dtype.type_size();
            let weights = stream.alloc_zeros::<u8>(bytes).unwrap();
            let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
            let q8_1 = stream.alloc_zeros::<u8>(k_padded / 32 * 36).unwrap();
            let mut dst = stream.alloc_zeros::<f32>(n).unwrap();
            let lanes_warps = super::mmvq_row_warps(tag, k).unwrap();
            let mut variants = vec![(format!("mmvq_gguf_{tag}_f32_plain_cuda1"), 1)];
            for warps in [4usize, 2, 1] {
                if warps >= lanes_warps && n.is_multiple_of(4 * (4 / warps)) {
                    let name = match warps {
                        4 => format!("mmvq_gguf_{tag}_f32_plain_smallk_cuda1"),
                        w => format!("mmvq_gguf_{tag}_f32_plain_smallk_w{w}_cuda1"),
                    };
                    variants.push((name, 4 * (4 / warps)));
                }
            }
            for (name, rows_per_block) in variants {
                let func = dev.mmvq_fn(&name).unwrap();
                let cfg = LaunchConfig {
                    grid_dim: ((n / rows_per_block) as u32, 1, 1),
                    block_dim: (32, 4, 1),
                    shared_mem_bytes: 0,
                };
                let (k_i, n_i, sy, sd) = (k as i32, n as i32, (k_padded / 32) as i32, n as i32);
                let launch = |dst: &mut crate::tensor::gpu_runtime::driver::CudaSlice<f32>| {
                    let mut b = stream.launch_builder(&func);
                    b.arg(&weights)
                        .arg(&q8_1)
                        .arg(dst)
                        .arg(&k_i)
                        .arg(&n_i)
                        .arg(&sy)
                        .arg(&sd);
                    unsafe { b.launch(cfg) }.unwrap();
                };
                for _ in 0..20 {
                    launch(&mut dst);
                }
                const RUNS: usize = 200;
                // Timing events: the default flags leave timing on.
                let flags =
                    crate::tensor::gpu_runtime::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
                let (a, b) = (
                    ctx.new_event(Some(flags)).unwrap(),
                    ctx.new_event(Some(flags)).unwrap(),
                );
                a.record(&stream).unwrap();
                for _ in 0..RUNS {
                    launch(&mut dst);
                }
                b.record(&stream).unwrap();
                b.synchronize().unwrap();
                let us = a.elapsed_ms(&b).unwrap() as f64 * 1e3 / RUNS as f64;
                eprintln!(
                    "{tag} k={k} n={n}: {name}: {us:.1} us, {:.1} GB/s",
                    bytes as f64 / us / 1e3
                );
            }
        }
        // The gate and up pair: two matrices of the shape, one launch.
        let pairs = [
            ("q5_0", GgmlDType::Q5_0, 896, 4864),
            ("q4_k", GgmlDType::Q4K, 1024, 3072),
            ("q4_k", GgmlDType::Q4K, 2048, 8192),
        ];
        for (tag, dtype, k, n) in pairs {
            let bytes = 2 * n * (k / dtype.block_size()) * dtype.type_size();
            let gate = stream.alloc_zeros::<u8>(bytes / 2).unwrap();
            let up = stream.alloc_zeros::<u8>(bytes / 2).unwrap();
            let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
            let q8_1 = stream.alloc_zeros::<u8>(k_padded / 32 * 36).unwrap();
            let mut dst = stream.alloc_zeros::<f32>(n).unwrap();
            let lanes_warps = super::mmvq_row_warps(tag, k).unwrap();
            for (suffix, warps) in [("", 4usize), ("_w2", 2), ("_w1", 1)] {
                if warps < lanes_warps {
                    continue;
                }
                let name = format!("mmvq_gguf_pair_{tag}_{tag}_silu{suffix}_cuda1");
                let func = dev.mmvq_fn(&name).unwrap();
                let rows_per_block = 4 / warps;
                let cfg = LaunchConfig {
                    grid_dim: (n.div_ceil(rows_per_block) as u32, 1, 1),
                    block_dim: (32, 4, 1),
                    shared_mem_bytes: 0,
                };
                let (k_i, n_i, sy, sd) = (k as i32, n as i32, (k_padded / 32) as i32, n as i32);
                let launch = |dst: &mut crate::tensor::gpu_runtime::driver::CudaSlice<f32>| {
                    let mut b = stream.launch_builder(&func);
                    b.arg(&up).arg(&gate).arg(&q8_1).arg(dst);
                    b.arg(&k_i).arg(&n_i).arg(&sy).arg(&sd);
                    unsafe { b.launch(cfg) }.unwrap();
                };
                for _ in 0..20 {
                    launch(&mut dst);
                }
                const RUNS: usize = 200;
                let flags =
                    crate::tensor::gpu_runtime::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
                let (a, b) = (
                    ctx.new_event(Some(flags)).unwrap(),
                    ctx.new_event(Some(flags)).unwrap(),
                );
                a.record(&stream).unwrap();
                for _ in 0..RUNS {
                    launch(&mut dst);
                }
                b.record(&stream).unwrap();
                b.synchronize().unwrap();
                let us = a.elapsed_ms(&b).unwrap() as f64 * 1e3 / RUNS as f64;
                eprintln!(
                    "pair {tag} k={k} n={n}: {name}: {us:.1} us, {:.1} GB/s",
                    bytes as f64 / us / 1e3
                );
            }
        }
    }
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

    /// Routed experts on the card are the host's bit for bit on every card path: one row, a
    /// decode's block, a verify's instances, a prompt's rows from staged and from resident weights,
    /// and a layer's rows. Where an expert runs must not change what it gives, or a greedy decode
    /// depends on which experts earlier requests left on the card.
    #[test]
    fn card_experts_are_the_hosts_bit_for_bit() {
        use crate::inference::offload::cpu_experts;
        use crate::inference::offload::experts::Expert;
        use crate::inference::offload::projection::Projection;
        use crate::tensor::quantized::QTensor;
        use crate::tensor::{Device, Tensor};
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card experts are NOT covered by this run");
            return;
        };
        let (dim, inter, limit, n) = (1024usize, 512usize, 7.0f32, 11usize);
        let draw = |seed: u32, len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    (((i as u32)
                        .wrapping_add(seed.wrapping_mul(0x9E37_79B9))
                        .wrapping_mul(2_654_435_761)
                        .rotate_left(11)
                        >> 8) as f32
                        / (1 << 24) as f32
                        - 0.5)
                        * scale
                })
                .collect()
        };
        let quant = |w: Vec<f32>, rows: usize, cols: usize, dtype: GgmlDType| {
            let t = Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap();
            Projection::Quant(std::sync::Arc::new(QTensor::quantize(&t, dtype).unwrap()))
        };
        let experts: Vec<std::sync::Arc<Expert>> = (0..3u32)
            .map(|j| {
                std::sync::Arc::new(Expert {
                    w1: quant(
                        draw(10 * j + 1, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Iq2Xxs,
                    ),
                    w3: quant(
                        draw(10 * j + 2, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Iq2Xxs,
                    ),
                    w2: quant(
                        draw(10 * j + 3, dim * inter, 0.08),
                        dim,
                        inter,
                        GgmlDType::Q2K,
                    ),
                })
            })
            .collect();
        // Rows large enough for the clamps to bite, one of zeros, one with a lone extreme.
        let mut x = draw(99, n * dim, 6.0);
        x[4 * dim..5 * dim].fill(0.0);
        x[7 * dim + 300] = 400.0;
        let row = |t: usize| &x[t * dim..(t + 1) * dim];
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();

        // The host: each expert over each row, and the rows entering its down projection.
        let host: Vec<Vec<(Vec<f32>, Vec<f32>)>> = experts
            .iter()
            .map(|e| {
                (0..n)
                    .map(|t| {
                        let xe = Tensor::from_vec(row(t).to_vec(), (1, dim), &Device::Cpu).unwrap();
                        let h = std::cell::RefCell::new(Vec::new());
                        let seen = |r: &[f32]| *h.borrow_mut() = r.to_vec();
                        let out = crate::inference::offload::on_host(|| {
                            e.forward_seen(&xe, limit, Some(&seen))
                        })
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                        (h.into_inner(), out)
                    })
                    .collect()
            })
            .collect();
        // The host's own paths agree with each other first.
        for t in 0..n {
            let outs = cpu_experts::batched(&experts, row(t), limit).unwrap();
            for (j, o) in outs.iter().enumerate() {
                assert_eq!(
                    bits(o),
                    bits(&host[j][t].1),
                    "host block, expert {j} row {t}"
                );
            }
        }
        let all: Vec<Vec<f32>> = (0..experts.len()).map(|_| x.clone()).collect();
        let multi = cpu_experts::multirow(&experts, &all, dim, limit).unwrap();
        for (j, o) in multi.iter().enumerate() {
            for t in 0..n {
                assert_eq!(
                    bits(&o[t * dim..(t + 1) * dim]),
                    bits(&host[j][t].1),
                    "host rows, expert {j} row {t}"
                );
            }
        }

        let stream = dev.stream();
        let bytes = |p: &Projection| match p {
            Projection::Quant(q) => q.data().unwrap().into_owned(),
            _ => unreachable!(),
        };
        let host_bytes: Vec<[Vec<u8>; 3]> = experts
            .iter()
            .map(|e| [bytes(&e.w1), bytes(&e.w3), bytes(&e.w2)])
            .collect();
        let on_card: Vec<[std::sync::Arc<CudaSlice<u8>>; 3]> = host_bytes
            .iter()
            .map(|b| {
                b.clone()
                    .map(|w| std::sync::Arc::new(stream.clone_htod(&w).unwrap()))
            })
            .collect();
        let (grid, signs) = iq2_xxs_tables(&dev).unwrap();
        let tables = (&grid, &signs);
        let refs: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)> =
            on_card.iter().map(|w| (&*w[0], &*w[1], &*w[2])).collect();

        for t in 0..n {
            let (h, out) = gpu_expert_row(
                &dev,
                refs[1].0,
                refs[1].1,
                refs[1].2,
                tables,
                (dim, inter),
                row(t),
                limit,
            )
            .unwrap();
            assert_eq!(bits(&h), bits(&host[1][t].0), "card row h, row {t}");
            assert_eq!(bits(&out), bits(&host[1][t].1), "card row, row {t}");
            let block =
                gpu_expert_rows_grouped(&dev, &refs, tables, (dim, inter), row(t), limit).unwrap();
            for (j, (_, o)) in block.iter().enumerate() {
                assert_eq!(
                    bits(o),
                    bits(&host[j][t].1),
                    "card block, expert {j} row {t}"
                );
            }
        }
        // A verify's instances: each expert over some of the rows, the instances of one expert
        // together, more of them than a tile.
        let picks: Vec<(usize, usize)> = (0..experts.len())
            .flat_map(|j| {
                (0..n)
                    .filter(move |t| (t + j) % 3 != 0)
                    .map(move |t| (j, t))
            })
            .collect();
        let inst: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)> =
            picks.iter().map(|&(j, _)| refs[j]).collect();
        let xs: Vec<f32> = picks
            .iter()
            .flat_map(|&(_, t)| row(t).iter().copied())
            .collect();
        let got =
            gpu_expert_rows_grouped_multi(&dev, &inst, tables, (dim, inter), &xs, limit).unwrap();
        for (k, &(j, t)) in picks.iter().enumerate() {
            assert_eq!(
                bits(&got[k]),
                bits(&host[j][t].1),
                "card instances, expert {j} row {t}"
            );
        }
        // A prompt's rows, from staged weights and from the card's.
        for j in 0..experts.len() {
            for resident in [false, true] {
                let weights = if resident {
                    on_card[j].clone().map(ExpertWeight::Card)
                } else {
                    [
                        ExpertWeight::Host(&host_bytes[j][0]),
                        ExpertWeight::Host(&host_bytes[j][1]),
                        ExpertWeight::Host(&host_bytes[j][2]),
                    ]
                };
                let (h, out) =
                    gpu_expert_tokens(&dev, weights, tables, (inter, dim), &x, limit).unwrap();
                for t in 0..n {
                    assert_eq!(
                        bits(&h[t * inter..(t + 1) * inter]),
                        bits(&host[j][t].0),
                        "card prompt h, expert {j} row {t}"
                    );
                    assert_eq!(
                        bits(&out[t * dim..(t + 1) * dim]),
                        bits(&host[j][t].1),
                        "card prompt, expert {j} row {t} resident {resident}"
                    );
                }
            }
        }
        // A layer's rows, read where the layer's input lies.
        let layer = gpu_layer_begin(&dev, &x, dim).unwrap();
        for j in 0..experts.len() {
            let toks: Vec<u32> = (0..n as u32)
                .filter(|t| (*t as usize + j).is_multiple_of(2))
                .collect();
            let weights = [
                ExpertWeight::Host(&host_bytes[j][0]),
                ExpertWeight::Host(&host_bytes[j][1]),
                ExpertWeight::Host(&host_bytes[j][2]),
            ];
            let out =
                gpu_expert_layer_rows(&dev, weights, tables, (inter, dim), &layer, &toks, limit)
                    .unwrap();
            for (r, &t) in toks.iter().enumerate() {
                assert_eq!(
                    bits(&out[r * dim..(r + 1) * dim]),
                    bits(&host[j][t as usize].1),
                    "card layer, expert {j} row {t}"
                );
            }
        }
    }

    /// Time per expert of a prompt's pass over a layer's rows: weights staged each time, then
    /// resident on the card; one thread, then four.
    #[test]
    #[ignore = "benchmark: run explicitly with --ignored --nocapture"]
    fn bench_expert_layer_rows() {
        let Ok(dev) = CudaDevice::new(0) else { return };
        let (inter, dim, n) = (2304usize, 5120usize, 128usize);
        let enc = |dtype: GgmlDType, len: usize, f: f32| {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32) * f).sin() * 0.05).collect();
            crate::tensor::quant_cpu::from_float_bytes(dtype, &v).unwrap()
        };
        let (gb, ub, db) = (
            enc(GgmlDType::Iq2Xxs, inter * dim, 0.013),
            enc(GgmlDType::Iq2Xxs, inter * dim, 0.017),
            enc(GgmlDType::Q2K, dim * inter, 0.011),
        );
        let x: Vec<f32> = (0..n * dim).map(|i| ((i as f32) * 0.3).sin()).collect();
        let toks: Vec<u32> = (0..31u32).map(|t| t * 3).collect();
        let stream = dev.stream();
        let (gc, uc, dc) = (
            std::sync::Arc::new(stream.clone_htod(&gb).unwrap()),
            std::sync::Arc::new(stream.clone_htod(&ub).unwrap()),
            std::sync::Arc::new(stream.clone_htod(&db).unwrap()),
        );
        let (grid, signs) = iq2_xxs_tables(&dev).unwrap();
        for resident in [false, true] {
            for threads in [1usize, 4] {
                let layer = gpu_layer_begin(&dev, &x, dim).unwrap();
                let calls = 120usize;
                let next = std::sync::atomic::AtomicUsize::new(0);
                let t = std::time::Instant::now();
                std::thread::scope(|s| {
                    for _ in 0..threads {
                        s.spawn(|| loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= calls {
                                break;
                            }
                            let w = if resident {
                                [
                                    ExpertWeight::Card(gc.clone()),
                                    ExpertWeight::Card(uc.clone()),
                                    ExpertWeight::Card(dc.clone()),
                                ]
                            } else {
                                [
                                    ExpertWeight::Host(&gb),
                                    ExpertWeight::Host(&ub),
                                    ExpertWeight::Host(&db),
                                ]
                            };
                            gpu_expert_layer_rows(
                                &dev,
                                w,
                                (&grid, &signs),
                                (inter, dim),
                                &layer,
                                &toks,
                                7.0,
                            )
                            .unwrap();
                        });
                    }
                });
                dev.stream().synchronize().unwrap();
                eprintln!(
                    "resident {resident} threads {threads}: {:.2} ms per expert",
                    t.elapsed().as_secs_f64() * 1000.0 / calls as f64
                );
            }
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
        let (out, inp, n) = (96usize, 512usize, 5usize);
        let w: Vec<f32> = (0..out * inp)
            .map(|i| {
                (((i as u32).wrapping_mul(2_654_435_761).rotate_left(9) >> 8) as f32
                    / (1 << 24) as f32
                    - 0.5)
                    * 0.08
            })
            .collect();
        let x: Vec<f32> = (0..n * inp).map(|i| ((i as f32) * 0.31).sin()).collect();
        for dtype in [
            GgmlDType::Iq2Xxs,
            GgmlDType::Q2K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
        ] {
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
            // The same product a few rows of the weight at a time.
            let tiled =
                gpu_quant_linear_tiled(&dev, dtype, &blocks, (out, inp), &x, 64 * (inp + n) * 4)
                    .unwrap();
            assert_eq!(tiled.len(), got.len());
            for (g, v) in tiled.iter().zip(&want) {
                assert!(
                    (g - v).abs() <= 1e-4 * v.abs().max(1.0),
                    "{dtype:?} tiled: {g} vs {v}"
                );
            }
            for (g, v) in got.iter().zip(&want) {
                assert!(
                    (g - v).abs() <= 1e-4 * v.abs().max(1.0),
                    "{dtype:?}: {g} vs {v}"
                );
            }
        }
    }

    /// The stream's mixes, collapse and update on the card equal the host's, to rounding.
    #[test]
    fn gpu_hc_stream_matches_the_cpu() {
        use crate::inference::model::deepseek_v41::hyper_connections::{hc_mixes, hc_post, hc_pre};
        use crate::tensor::{Device, Tensor};
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card stream is NOT covered by this run");
            return;
        };
        let (n, hc, d) = (37usize, 4usize, 96usize);
        let mix_hc = (2 + hc) * hc;
        let wave = |len: usize, k: f32, amp: f32, off: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * k).sin() * amp + off)
                .collect()
        };
        let x = wave(n * hc * d, 0.173, 1.5, 0.1);
        let func = wave(mix_hc * hc * d, 0.071, 0.2, 0.0);
        let scale = [0.7f32, 1.1, 0.9];
        let base = wave(mix_hc, 0.9, 0.5, 0.0);
        let (iters, norm_eps, hc_eps) = (20usize, 1e-6f32, 1e-6f32);
        let xt = Tensor::from_vec(x.clone(), (1, n, hc, d), &Device::Cpu).unwrap();
        let ft = Tensor::from_vec(func.clone(), (mix_hc, hc * d), &Device::Cpu).unwrap();
        let want = hc_mixes(&xt, &ft, &scale, &base, hc, iters, norm_eps, hc_eps).unwrap();
        let mut s = HcStreamDev::open(&dev, &x, hc, d).unwrap();
        let (pre, post, comb) = s
            .mixes(&dev, &func, &scale, &base, iters, norm_eps, hc_eps)
            .unwrap();
        let close = |a: &[f32], b: &[f32], what: &str| {
            let worst = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let range = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(
                worst <= 2e-5 * range.max(1.0),
                "{what}: worst {worst} of {range}"
            );
        };
        close(&pre, &want.pre.concat(), "pre");
        close(&post, &want.post.concat(), "post");
        close(
            &comb,
            &want
                .comb
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            "comb",
        );
        // The collapse and the update, from the host's own coefficients.
        let w = wave(d, 0.37, 0.3, 1.0);
        let wt = Tensor::from_vec(w.clone(), d, &Device::Cpu).unwrap();
        let collapsed =
            crate::tensor::ops::rms_norm(&hc_pre(&xt, &want.pre).unwrap(), &wt, 1e-6).unwrap();
        let got = s.pre_norm(&dev, &want.pre.concat(), &w, 1e-6).unwrap();
        close(
            &got,
            &collapsed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "collapse",
        );
        let xo = wave(n * d, 0.53, 0.8, 0.0);
        let xot = Tensor::from_vec(xo.clone(), (1, n, d), &Device::Cpu).unwrap();
        let after = hc_post(&xot, &xt, &want.post, &want.comb, hc).unwrap();
        s.post(
            &dev,
            &xo,
            &want.post.concat(),
            &want
                .comb
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        )
        .unwrap();
        close(
            &s.rows(&dev, 0, n).unwrap(),
            &after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "update",
        );
        close(
            &s.rows(&dev, n - 1, 1).unwrap(),
            &after.flatten_all().unwrap().to_vec1::<f32>().unwrap()[(n - 1) * hc * d..],
            "last row",
        );
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
