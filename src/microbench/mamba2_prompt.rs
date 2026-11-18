//! The Mamba2 block of one layer over the rows of a prompt chunk on a card, after its input
//! projection: the token kernels launched row by row against one pass for every row.
//! The dimensions are a file's `ssm.*` metadata.
//! Both variants run the functions the layer runs.

use super::{Context, Measured};

/// The Mamba2 dimensions of the file: `ssm.inner_size`, `ssm.state_size`, `ssm.group_count`,
/// `ssm.conv_kernel` and `ssm.time_step_rank` for the heads.
pub fn read_dims(
    path: &std::path::Path,
) -> Result<crate::inference::generic_transformer::mixer::Mamba2Dims, String> {
    let content = crate::tensor::quantized::gguf_file::open_header(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .ok_or_else(|| "the file names no general.architecture".to_string())?;
    let meta = |key: &str| -> Result<usize, String> {
        content
            .metadata
            .get(&format!("{arch}.ssm.{key}"))
            .and_then(|v| v.to_u64().ok())
            .map(|v| v as usize)
            .ok_or_else(|| format!("the file names no {arch}.ssm.{key}"))
    };
    Ok(crate::inference::generic_transformer::mixer::Mamba2Dims {
        d_inner: meta("inner_size")?,
        d_state: meta("state_size")?,
        ngroups: meta("group_count")?,
        d_conv: meta("conv_kernel")?,
        nheads: meta("time_step_rank")?,
    })
}

#[cfg(feature = "gpu")]
pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    use super::stats::{run_for, timed};
    use super::Fail;
    use crate::inference::generic_transformer::mixer::{Mamba2, Mamba2Weights};
    use crate::inference::generic_transformer::projection::QMatMul;
    use crate::tensor::quantized::{GgmlDType, QTensor};
    use crate::tensor::{DType, Device, Tensor};
    use std::time::Duration;

    let card = cx.card.ok_or("mamba2-prompt needs a card")?;
    let path = cx
        .options
        .shapes_from
        .as_ref()
        .ok_or("mamba2-prompt reads its dimensions from a file: give --shapes-from")?;
    let dims = read_dims(path)?;
    let rows = cx.options.rows;
    if rows == 0 || dims.nheads == 0 || dims.ngroups == 0 || dims.d_conv == 0 {
        return Err("--rows and the file's heads, groups and taps must be above zero".to_string());
    }
    let (dx, nh, hd) = (dims.d_xbc(), dims.nheads, dims.headdim());
    let width = dims.d_inner + dx + nh;
    let dev = Device::Cuda(card.dev.clone());
    let tensor = |e: crate::tensor::Error| format!("{e:?}");
    let randn = |std: f32, shape: &[usize]| Tensor::randn(0.0f32, std, shape, &dev).map_err(tensor);
    let a = randn(0.5, &[nh])?
        .abs()
        .and_then(|t| t.affine(-1.0, -0.5))
        .map_err(tensor)?;
    let weights = Mamba2Weights {
        in_proj: QMatMul::from_qtensor(
            // CONVERSION: synthetic weight of a microbench, no stored weight.
            QTensor::quantize_onto(
                &Tensor::zeros_on((width, 1), DType::F32, &Device::Cpu).map_err(tensor)?,
                GgmlDType::F32,
                &Device::Cpu,
            )
            .map_err(tensor)?,
        )
        .map_err(tensor)?,
        conv_w: randn(0.5, &[dx, dims.d_conv])?,
        conv_b: randn(0.2, &[dx])?,
        a,
        d: randn(1.0, &[nh])?,
        dt_bias: randn(0.5, &[nh])?,
        norm_w: randn(1.0, &[dims.ngroups, dims.d_inner / dims.ngroups])?,
    };
    let mamba = Mamba2::new(weights, dims, 1e-5, &dev).map_err(tensor)?;
    let proj = randn(1.0, &[1, rows, width])?;
    let conv = Tensor::zeros_on((1, dx, dims.d_conv), DType::F32, &dev).map_err(tensor)?;
    let h = Tensor::zeros_on((1, nh, hd, dims.d_state), DType::F32, &dev).map_err(tensor)?;
    let target = Duration::from_secs_f64(cx.options.seconds);
    let fail = Fail::default();
    let mut out = Vec::new();
    type Pass<'a> = Box<dyn Fn() -> crate::tensor::Result<(Tensor, Tensor, Tensor)> + 'a>;
    let variants: [(String, Pass); 2] = [
        (
            format!(
                "token steps: the 4 token kernels for each of {rows} rows, then a concatenation"
            ),
            Box::new(|| mamba.card_steps(&proj, conv.clone(), h.clone())),
        ),
        (
            "whole sequence: 1 launch of each of the 4 kernels".to_string(),
            Box::new(|| mamba.card_rows(&proj, conv.clone(), h.clone())),
        ),
    ];
    for (variant, pass) in variants {
        let samples = run_for(
            target,
            timed(|| {
                fail.ok(pass().map(|out| {
                    std::hint::black_box(out);
                }));
                fail.ok(dev.synchronize());
            }),
        );
        fail.check(&variant)?;
        out.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: None,
        });
    }
    if let [old, new] = &mut out[..] {
        new.note = super::prefill_layer::ratio_note(old, new);
    }
    Ok(out)
}
