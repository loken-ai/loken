//! A BF16 weight times a few F32 rows on a card, each weight row read once for all of them.

use super::*;
use crate::tensor::kernel_ffi::CudaDType;
use core::ffi::c_void;

/// Rows one launch of the BF16 product takes.
pub const BF16_MATVEC_ROWS: usize = 8;
/// Weights the kernel reads at a time: a row's length is a multiple of it.
const BF16_PER_LOAD: usize = 8;

fn address<T: CudaDType>(storage: &StorageView, what: &str) -> Result<u64> {
    let slice = match storage {
        StorageView::Cuda(c) => c.as_cuda_slice::<T>()?,
        _ => crate::tensor::bail!("BF16 product: {what} must be on CUDA"),
    };
    Ok(slice.device_ptr(slice.stream()).0)
}

/// `x @ w^T` for `x` `[rows, k]` and a BF16 weight `w` `[n, k]` on a card: `[rows, n]` F32,
/// summed in F32. A row's result does not depend on the rows beside it. `None` for more rows than
/// [`BF16_MATVEC_ROWS`], a row length the kernel does not take, or a weight not held BF16 on a
/// card.
pub fn bf16_matvec(x: &Tensor, w: &Tensor) -> Result<Option<Tensor>> {
    if w.dtype() != DType::BF16 || !w.device().is_cuda() || w.dims().len() != 2 {
        return Ok(None);
    }
    let (n, k) = (w.dims()[0], w.dims()[1]);
    let xd = x.dims();
    if xd.len() != 2 || xd[1] != k || !k.is_multiple_of(BF16_PER_LOAD) {
        return Ok(None);
    }
    let rows = xd[0];
    if rows == 0 || rows > BF16_MATVEC_ROWS {
        return Ok(None);
    }
    let dev = w.device().as_cuda_device()?;
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let w = w.contiguous()?;
    let (x_st, x_lo) = x.storage_and_layout();
    let (w_st, w_lo) = w.storage_and_layout();
    let x_ptr =
        (address::<f32>(&x_st, "activation")? as usize + x_lo.start_offset() * 4) as *const c_void;
    let w_ptr = (address::<half::bf16>(&w_st, "weight")? as usize + w_lo.start_offset() * 2)
        as *const c_void;
    if !(x_ptr as usize).is_multiple_of(16) || !(w_ptr as usize).is_multiple_of(16) {
        return Ok(None);
    }
    let y = unsafe { dev.alloc::<f32>(rows * n) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    unsafe {
        loken_bf16_matvec(
            x_ptr,
            w_ptr,
            y.device_ptr(y.stream()).0 as *mut c_void,
            k as i32,
            n as i32,
            rows as i32,
            stream,
        );
    }
    Ok(Some(tensor_from_cuda_storage(
        CudaStorage::wrap_cuda_slice(y, dev.clone()),
        (rows, n),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                (h >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn a_row_among_others_equals_its_own_launch_and_the_f32_sum() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (n, k) = (37, 2560);
        let w = Tensor::from_vec(values(n * k, 1), (n, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let wide: Vec<f32> = w
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let x: Vec<f32> = values(BF16_MATVEC_ROWS * k, 2);
        let w_card = w.to_device(&dev).unwrap();
        let alone: Vec<Vec<f32>> = (0..BF16_MATVEC_ROWS)
            .map(|r| {
                let xr = Tensor::from_vec(x[r * k..(r + 1) * k].to_vec(), (1, k), &dev).unwrap();
                let y = bf16_matvec(&xr, &w_card)
                    .unwrap()
                    .expect("the kernel takes one row");
                y.flatten_all().unwrap().to_vec1().unwrap()
            })
            .collect();
        for rows in 1..=BF16_MATVEC_ROWS {
            let xs = Tensor::from_vec(x[..rows * k].to_vec(), (rows, k), &dev).unwrap();
            let y: Vec<f32> = bf16_matvec(&xs, &w_card)
                .unwrap()
                .expect("the kernel takes these rows")
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for r in 0..rows {
                assert_eq!(&y[r * n..(r + 1) * n], &alone[r][..], "row {r} of {rows}");
            }
        }
        for r in 0..BF16_MATVEC_ROWS {
            for o in 0..n {
                let want: f64 = (0..k)
                    .map(|i| f64::from(wide[o * k + i]) * f64::from(x[r * k + i]))
                    .sum();
                assert!(
                    (f64::from(alone[r][o]) - want).abs() < 1e-3,
                    "row {r} output {o}"
                );
            }
        }
        let over = Tensor::from_vec(
            values((BF16_MATVEC_ROWS + 1) * k, 3),
            (BF16_MATVEC_ROWS + 1, k),
            &dev,
        )
        .unwrap();
        assert!(bf16_matvec(&over, &w_card).unwrap().is_none());
    }

    #[test]
    #[ignore = "needs an idle CUDA card"]
    fn the_library_product_sums_bf16_operands_in_f32() {
        let dev = Device::new_cuda(0).expect("a card for the card test");
        let (m, n, k) = (300, 37, 2560);
        let bf16 = |v: Vec<f32>, rows| {
            Tensor::from_vec(v, (rows, k), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let (x, w) = (bf16(values(m * k, 4), m), bf16(values(n * k, 5), n));
        let wide = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let (xs, ws) = (wide(&x), wide(&w));
        let y = x
            .to_device(&dev)
            .unwrap()
            .matmul_t_bf16_f32(&w.to_device(&dev).unwrap())
            .unwrap()
            .expect("two BF16 matrices on one card");
        assert_eq!(y.dtype(), DType::F32);
        assert_eq!(y.dims(), &[m, n]);
        let y: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for r in 0..m {
            for o in 0..n {
                let want: f64 = (0..k)
                    .map(|i| f64::from(xs[r * k + i]) * f64::from(ws[o * k + i]))
                    .sum();
                assert!(
                    (f64::from(y[r * n + o]) - want).abs() < 1e-3,
                    "row {r} output {o}"
                );
            }
        }
    }
}
