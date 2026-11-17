//! The gated short convolution of one layer over the rows of a prompt chunk on a card: one
//! launch of the token kernel per row against one launch for every row.
//! The width and the taps are a file's: `embedding_length` and `shortconv.l_cache`.
//! Both variants run the functions the layer runs, on rows in the card's activation format.

use super::{Context, Measured};

/// The taps of the file's short convolution, from `{arch}.shortconv.l_cache`.
pub fn read_taps(path: &std::path::Path) -> Result<usize, String> {
    let content = crate::tensor::quantized::gguf_file::open_header(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .ok_or_else(|| "the file names no general.architecture".to_string())?;
    content
        .metadata
        .get(&format!("{arch}.shortconv.l_cache"))
        .and_then(|v| v.to_u64().ok())
        .map(|v| v as usize)
        .ok_or_else(|| format!("the file names no {arch}.shortconv.l_cache"))
}

#[cfg(feature = "gpu")]
pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    use super::stats::{run_for, timed};
    use super::Fail;
    use crate::inference::generic_transformer::mixer::{short_conv_card, short_conv_card_steps};
    use crate::tensor::{DType, Device, Tensor};
    use std::time::Duration;

    let card = cx.card.ok_or("shortconv-prompt needs a card")?;
    let path = cx
        .options
        .shapes_from
        .as_ref()
        .ok_or("shortconv-prompt reads the taps from a file: give --shapes-from")?;
    let taps = read_taps(path)?;
    let (d, rows) = (cx.shapes.hidden, cx.options.rows);
    if rows == 0 || taps < 2 {
        return Err("--rows must be above zero and the file's taps above 1".to_string());
    }
    let dev = Device::Cuda(card.dev.clone());
    let tensor = |e: crate::tensor::Error| format!("{e:?}");
    let proj = Tensor::randn(0.0f32, 1.0, (1, rows, 3 * d), &dev)
        .and_then(|t| t.to_dtype(DType::F16))
        .map_err(tensor)?;
    let conv_w = Tensor::randn(0.0f32, 0.5, (d, taps), &dev).map_err(tensor)?;
    let state = Tensor::zeros_on((1, d, taps - 1), DType::F32, &dev).map_err(tensor)?;
    let target = Duration::from_secs_f64(cx.options.seconds);
    let fail = Fail::default();
    let mut out = Vec::new();
    type Pass =
        fn(&Tensor, Tensor, &Tensor, usize, usize) -> crate::tensor::Result<(Tensor, Tensor)>;
    let variants: [(String, Pass); 2] = [
        (
            format!("token steps: {rows} launches of the token kernel, then a concatenation"),
            short_conv_card_steps,
        ),
        ("whole sequence: 1 launch".to_string(), short_conv_card),
    ];
    for (variant, pass) in variants {
        let samples = run_for(
            target,
            timed(|| {
                fail.ok(pass(&proj, state.clone(), &conv_w, d, taps).map(|out| {
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
