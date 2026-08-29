//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Launch a 1-elementwise or 2-operand elementwise kernel over `n` elements.
pub(super) fn launch_elementwise(
    dev: &CudaDevice,
    name: &str,
    inputs: &[&CudaSlice<f32>],
    n: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.elementwise_fn(name)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "kernel out", || unsafe { stream.alloc::<f32>(n) })?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut b = stream.launch_builder(&func);
    for inp in inputs {
        b.arg(*inp);
    }
    b.arg(&out);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("{name} launch: {e}")))?;
    Ok(out)
}

pub fn binary_f32(
    dev: &CudaDevice,
    name: &str,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<f32>> {
    launch_elementwise(dev, name, &[a, b], n)
}
