//! A projection held on a card in the form its file stores.
//!
//! The card keeps the packed bytes, never a dense F32 copy:
//! - GGUF blocks go to the card's quantised matmul, which reads them in place,
//! - block-scaled fp8 and fp4 keep their bytes and ue8m0 scales and expand a tile of rows at a
//!   time into a scratch buffer that one matrix product consumes,
//! - bf16 keeps its 16-bit values and expands a tile the same way,
//! - a dense weight keeps its F32 values, which are its stored form.
//!
//! The scratch of an expansion is bounded by the offload's weight tile.

use crate::tensor::blockscaled::fp8::ScaleFormat;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::cuda::{
    alloc_f32, cast_bf16_to_f32, copy_f32, matmul_nt_f32, put_cols, upload_f32, CudaDevice,
    CudaStorage, QUANT_LINEAR_TILE_BYTES,
};
use crate::tensor::gpu_runtime::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::tensor::quantized::QMatMul;
use crate::tensor::{Device, Error, Result, Tensor};

use std::sync::Arc;

/// How the card holds the values of a weight.
enum Form {
    /// F32 values.
    Dense(CudaSlice<f32>),
    /// GGUF blocks behind the card's quantised matmul.
    Blocks(Box<QMatMul>),
    /// e4m3 values of the rows of a view, with the ue8m0 scales of the whole weight's tiles.
    Fp8 {
        bytes: CudaSlice<u8>,
        scales: CudaSlice<u8>,
        block: usize,
        row0: usize,
    },
    /// e2m1 nibbles with a ue8m0 scale per `block` values along the input.
    Fp4 {
        nibbles: CudaSlice<u8>,
        scales: CudaSlice<u8>,
        block: usize,
    },
    /// bf16 values.
    Bf16(CudaSlice<half::bf16>),
}

/// A weight `[out, inp]` on a card, in its packed form.
pub(super) struct PackedDev {
    dev: Arc<CudaDevice>,
    form: Form,
    out: usize,
    inp: usize,
    /// Bytes the card holds for the weight.
    bytes: usize,
    /// Bytes of F32 values a scratch tile holds.
    tile_bytes: usize,
}

fn upload_u8(dev: &CudaDevice, what: &str, v: &[u8]) -> Result<CudaSlice<u8>> {
    dev.stream()
        .clone_htod(v)
        .map_err(|e| Error::msg(format!("{what} upload: {e}")))
}

fn launch_err(name: &str, e: crate::tensor::gpu_runtime::driver::DriverError) -> Error {
    Error::msg(format!("{name} launch: {e}"))
}

fn grid(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(256).max(1) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl PackedDev {
    /// `p` uploaded as stored.
    pub(super) fn new(dev: &Arc<CudaDevice>, p: &Projection) -> Result<Self> {
        let (out, inp) = match p.dims().as_slice() {
            [out, inp] => (*out, *inp),
            _ => return Err(Error::msg("a card mixer weight is a matrix")),
        };
        let (form, bytes) = match p {
            Projection::Quant(q) => {
                let on_card = q.to_device(&Device::Cuda(dev.clone()))?;
                (
                    Form::Blocks(Box::new(QMatMul::from_arc(Arc::new(on_card))?)),
                    p.bytes(),
                )
            }
            Projection::Fp8(f) if f.scale_format == ScaleFormat::E8m0 => {
                // The scales of the tile rows the view touches, addressed from the first of them.
                let per_row = f.inp.div_ceil(f.block);
                let first = f.row0() / f.block;
                let last = (f.row0() + f.out).div_ceil(f.block);
                let scales = &f.scales.as_slice()[first * per_row..last * per_row];
                (
                    Form::Fp8 {
                        bytes: upload_u8(dev, "fp8 weight", f.view_bytes())?,
                        scales: upload_u8(dev, "fp8 scales", scales)?,
                        block: f.block,
                        row0: f.row0() - first * f.block,
                    },
                    f.view_bytes().len() + scales.len(),
                )
            }
            Projection::Fp4(f) => (
                Form::Fp4 {
                    nibbles: upload_u8(dev, "fp4 weight", f.nibbles.as_slice())?,
                    scales: upload_u8(dev, "fp4 scales", f.scales.as_slice())?,
                    block: inp / (f.scales.len() / out),
                },
                p.bytes(),
            ),
            Projection::Bf16(f) => {
                let values: Vec<half::bf16> = f
                    .bytes
                    .as_slice()
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let up = dev
                    .stream()
                    .clone_htod(&values)
                    .map_err(|e| Error::msg(format!("bf16 weight upload: {e}")))?;
                (Form::Bf16(up), p.bytes())
            }
            // The values of a dense weight, or of fp8 scales stored as F32, are F32 already.
            _ => {
                let values = p.dense_f32()?;
                (
                    Form::Dense(upload_f32(dev, "mixer weight", &values)?),
                    values.len() * std::mem::size_of::<f32>(),
                )
            }
        };
        Ok(Self {
            dev: dev.clone(),
            form,
            out,
            inp,
            bytes,
            tile_bytes: QUANT_LINEAR_TILE_BYTES,
        })
    }

    /// The weight with scratch tiles of `bytes`.
    #[cfg(test)]
    pub(super) fn with_tile_bytes(mut self, bytes: usize) -> Self {
        self.tile_bytes = bytes;
        self
    }

    pub(super) fn out(&self) -> usize {
        self.out
    }

    /// Bytes the card holds for the weight.
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Rows of the weight a scratch tile holds.
    fn tile_rows(&self) -> usize {
        (self.tile_bytes / (self.inp * std::mem::size_of::<f32>()).max(1)).clamp(1, self.out)
    }

    /// Rows `r0..r0 + rows` of the weight expanded to F32, `[rows, inp]`.
    fn expand(&self, r0: usize, rows: usize) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let inp = self.inp;
        let total = rows * inp;
        match &self.form {
            Form::Fp8 {
                bytes,
                scales,
                block,
                row0,
            } => {
                let w = alloc_f32(dev, "fp8 tile", total)?;
                let func = dev.quantized_fn("dequant_fp8_tiles")?;
                let args = [
                    inp as i64,
                    *block as i64,
                    (row0 + r0) as i64,
                    inp.div_ceil(*block) as i64,
                    total as i64,
                ];
                let view = bytes.slice(r0 * inp..(r0 + rows) * inp);
                let mut b = dev.stream().launch_builder(&func);
                b.arg(&view);
                b.arg(scales);
                b.arg(&w);
                for a in &args {
                    b.arg(a);
                }
                unsafe { b.launch(grid(total)) }.map_err(|e| launch_err("dequant_fp8_tiles", e))?;
                Ok(w)
            }
            Form::Fp4 {
                nibbles,
                scales,
                block,
            } => {
                let w = alloc_f32(dev, "fp4 tile", total)?;
                let func = dev.quantized_fn("dequant_fp4_e8m0")?;
                let per_row = inp / block;
                let nib = nibbles.slice(r0 * inp / 2..(r0 + rows) * inp / 2);
                let sc = scales.slice(r0 * per_row..(r0 + rows) * per_row);
                let (inp_i, block_i, total_i) = (inp as i64, *block as i64, total as i64);
                let mut b = dev.stream().launch_builder(&func);
                b.arg(&nib);
                b.arg(&sc);
                b.arg(&w);
                b.arg(&inp_i);
                b.arg(&block_i);
                b.arg(&total_i);
                unsafe { b.launch(grid(total)) }.map_err(|e| launch_err("dequant_fp4_e8m0", e))?;
                Ok(w)
            }
            Form::Bf16(values) => {
                let tile = values.slice(r0 * inp..(r0 + rows) * inp);
                let own = {
                    let mut c = dev
                        .stream()
                        .alloc_zeros::<half::bf16>(total)
                        .map_err(|e| Error::msg(format!("bf16 tile: {e}")))?;
                    dev.stream()
                        .memcpy_dtod(&tile, &mut c.slice_mut(0..total))
                        .map_err(|e| Error::msg(format!("bf16 tile copy: {e}")))?;
                    c
                };
                cast_bf16_to_f32(dev, &own, total)
            }
            Form::Dense(_) | Form::Blocks(_) => Err(Error::msg("this form expands no tile")),
        }
    }

    /// `x` `[rows, inp]` through the weight: `[rows, out]`.
    pub(super) fn apply(&self, x: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>> {
        let dev: &CudaDevice = &self.dev;
        let (out, inp) = (self.out, self.inp);
        if x.len() < rows * inp {
            return Err(Error::msg(format!(
                "linear: {} inputs for [{rows}, {inp}] x [{out}, {inp}]",
                x.len()
            )));
        }
        match &self.form {
            Form::Dense(w) => matmul_nt_f32(dev, x, w, 1, rows, inp, out),
            Form::Blocks(qmm) => {
                let xt = Tensor::from_cuda_storage(
                    CudaStorage::F32(copy_f32(dev, "mixer input", &x.slice(0..rows * inp))?),
                    self.dev.clone(),
                    (rows, inp),
                )?;
                let y = qmm.forward(&xt)?;
                let (ys, _) = y.cuda_f32_slice()?;
                copy_f32(dev, "mixer product", &ys.slice(0..rows * out))
            }
            _ => {
                let tile = self.tile_rows();
                if tile >= out {
                    let w = self.expand(0, out)?;
                    return matmul_nt_f32(dev, x, &w, 1, rows, inp, out);
                }
                let mut y = alloc_f32(dev, "mixer product", rows * out)?;
                for r0 in (0..out).step_by(tile) {
                    let n = tile.min(out - r0);
                    let w = self.expand(r0, n)?;
                    let part = matmul_nt_f32(dev, x, &w, 1, rows, inp, n)?;
                    put_cols(dev, &part, &mut y, (rows, out), (r0, n))?;
                }
                Ok(y)
            }
        }
    }
}
