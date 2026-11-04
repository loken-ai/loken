//! The data movement of one prefill layer: the copies of q, k and v out of the projections, the
//! norms of q and k, the rotation, and the write of k and v into a paged store. The separate
//! launches of the layer loop run against the one fused launch, on synthetic F16 tensors of a
//! model's head shapes, with the launches of each variant counted.
//!
//! A second group runs the activation of the feed-forward block: the 2 copies that cut the gate
//! and the up halves out of the fused projection and `silu_mul_f16`, against the one kernel that
//! reads the fused buffer.
//!
//! A third group runs the stage from the fused input launch to the attention output: the q cast
//! to F32, the copy of q to heads first, the flash kernel and the cast of its output, against a
//! fused launch that writes q in the flash kernel's layout and a flash kernel that stores F16.

use super::stats::{run_for, Summary};
use super::{Measured, Options};

/// The head shapes of an attention layer and the base of its rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heads {
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_base: f32,
}

/// What a file says about the heads, each part when it names it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FileHeads {
    pub n_head: Option<usize>,
    pub n_kv: Option<usize>,
    pub head_dim: Option<usize>,
    pub rope_base: Option<f32>,
}

/// The heads the options and the file give, an option over the file. The key-value heads default
/// to the heads, as a file that names no grouping has none. The head width defaults to the
/// hidden size over the heads, which `hidden` states. The rotation base only sets the angles, so
/// without one the base is 1.
pub fn resolve_heads(
    opts: &Options,
    file: &FileHeads,
    hidden: Option<usize>,
) -> Result<Heads, String> {
    let n_head = opts.heads.or(file.n_head).ok_or(NEEDS_HEADS)?;
    let n_kv = opts.kv_heads.or(file.n_kv).unwrap_or(n_head);
    let head_dim = opts
        .head_dim
        .or(file.head_dim)
        .or_else(|| hidden.filter(|_| n_head > 0).map(|h| h / n_head))
        .ok_or(NEEDS_HEADS)?;
    if n_head == 0 || n_kv == 0 || head_dim == 0 || !n_head.is_multiple_of(n_kv) {
        return Err(format!(
            "heads {n_head}, kv heads {n_kv}, head dim {head_dim}: each must be above zero and the heads a multiple of the kv heads"
        ));
    }
    Ok(Heads {
        n_head,
        n_kv,
        head_dim,
        rope_base: file.rope_base.unwrap_or(1.0),
    })
}

const NEEDS_HEADS: &str =
    "prefill-layer needs --shapes-from, or --heads, --kv-heads and --head-dim";

/// The heads a GGUF header names, with the hidden size.
pub fn read_file(path: &std::path::Path) -> Result<(FileHeads, usize), String> {
    let content = crate::tensor::quantized::gguf_file::open_header(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .ok_or_else(|| "the file names no general.architecture".to_string())?;
    let meta = |key: &str| content.metadata.get(&format!("{arch}.{key}"));
    let meta_u = |key: &str| meta(key).and_then(|v| v.to_u64().ok()).map(|v| v as usize);
    let hidden = meta_u("embedding_length")
        .ok_or_else(|| format!("the file names no {arch}.embedding_length"))?;
    Ok((
        FileHeads {
            n_head: meta_u("attention.head_count"),
            n_kv: meta_u("attention.head_count_kv"),
            head_dim: meta_u("attention.key_length"),
            rope_base: meta("rope.freq_base").and_then(|v| v.to_f32().ok()),
        },
        hidden,
    ))
}

/// The feed-forward length the activation group runs at: the `--inter` option, else the layer
/// length `marlin-rows` uses.
pub fn ffn_len(opts: &Options) -> usize {
    opts.inter
        .unwrap_or(crate::microbench::marlin_rows::DEFAULT_INTER)
}

/// The speed of `new` against `old`, from their median times.
pub fn ratio_note(old: &Measured, new: &Measured) -> Option<String> {
    if old.samples_us.is_empty() || new.samples_us.is_empty() {
        return None;
    }
    let (o, n) = (
        Summary::of(&old.samples_us).median_us,
        Summary::of(&new.samples_us).median_us,
    );
    (n > 0.0).then(|| format!("{:.2}x the speed of `{}` by the median", o / n, old.variant))
}

#[cfg(feature = "gpu")]
pub fn run(cx: &super::Context) -> Result<Vec<Measured>, String> {
    use super::Fail;
    use crate::inference::cache::paged_attention::{
        paged_prefill_attn_seqs, paged_prefill_attn_seqs_head_major, rope_apply, rope_cos_sin,
    };
    use crate::inference::cache::paged_kv::PagedKvStore;
    use crate::inference::kernel::fused::{
        flash_prefill_gqa_varlen_head_major, prefill_rope_kv_write_f16,
        prefill_rope_kv_write_head_major_f16, silu_mul_f16, silu_mul_gu_f16,
    };
    use crate::tensor::layer::RmsNorm;
    use crate::tensor::{DType, Device, Tensor};
    use std::time::Duration;

    let card = cx.card.ok_or("prefill-layer needs a card")?;
    let (file, hidden) = match &cx.options.shapes_from {
        Some(path) => {
            let (f, h) = read_file(path)?;
            (f, Some(h))
        }
        None => (FileHeads::default(), None),
    };
    let heads = resolve_heads(cx.options, &file, hidden)?;
    let Heads {
        n_head: nh,
        n_kv: nkv,
        head_dim: hd,
        rope_base,
    } = heads;
    let rows = cx.options.rows;
    if rows == 0 {
        return Err("--rows must be above zero".to_string());
    }
    let (qd, kvd) = (nh * hd, nkv * hd);
    let target = Duration::from_secs_f64(cx.options.seconds);
    let dev = Device::Cuda(card.dev.clone());
    let tensor = |e: crate::tensor::Error| format!("{e:?}");

    let synth = |width: usize, dtype: DType| -> Result<Tensor, String> {
        Tensor::randn(0.0, 1.0, &[rows, width][..], &dev)
            .and_then(|t| t.to_dtype(dtype))
            .map_err(tensor)
    };
    let (q, k, v) = (
        synth(qd, DType::F16)?,
        synth(kvd, DType::F16)?,
        synth(kvd, DType::F16)?,
    );
    let qkv = synth(qd + 2 * kvd, DType::F16)?;
    let weight = || Tensor::randn(0.0, 1.0, &[hd][..], &dev).map_err(tensor);
    let eps = f32::EPSILON as f64;
    let (qn, kn) = (
        RmsNorm::from_tensor(weight()?, eps),
        RmsNorm::from_tensor(weight()?, eps),
    );
    let positions: Vec<usize> = (0..rows).collect();
    let (rcos, rsin) = rope_cos_sin(&positions, hd, rope_base, &dev, DType::F16).map_err(tensor)?;
    let store = PagedKvStore::new(1, rows, nkv, hd, DType::F16, &dev).map_err(tensor)?;
    let slot_index = store.slot_index(&positions, &dev).map_err(tensor)?;
    let slots_i32: Vec<i32> = positions.iter().map(|&s| s as i32).collect();
    let slot_dev = Tensor::from_vec(slots_i32.clone(), &[rows][..], &dev).map_err(tensor)?;

    // The layer loop's separate launches, from the 3 projected tensors.
    let old = |q: &Tensor,
               k: &Tensor,
               v: &Tensor,
               norms: Option<(&RmsNorm, &RmsNorm)>|
     -> crate::tensor::Result<()> {
        let qh = q.reshape(&[rows, nh, hd][..])?;
        let kh = k.reshape(&[rows, nkv, hd][..])?;
        let (qh, kh) = match norms {
            Some((qn, kn)) => (
                qn.forward(&qh.contiguous()?)?,
                kn.forward(&kh.contiguous()?)?,
            ),
            None => (qh, kh),
        };
        let q_rot = rope_apply(&qh, &rcos, &rsin, false)?;
        let k_rot = rope_apply(&kh, &rcos, &rsin, false)?;
        let v_f = v.reshape(&[rows, nkv, hd][..])?;
        store.write_indexed(
            &slot_index,
            &k_rot.reshape(&[rows, kvd][..])?,
            &v_f.reshape(&[rows, kvd][..])?,
        )?;
        std::hint::black_box((q_rot, k_rot, v_f));
        Ok(())
    };
    // The same from one fused projection: q, k and v are copies cut out of it.
    let old_fused = |norms: Option<(&RmsNorm, &RmsNorm)>| -> crate::tensor::Result<()> {
        old(
            &qkv.narrow(1, 0, qd)?,
            &qkv.narrow(1, qd, kvd)?,
            &qkv.narrow(1, qd + kvd, kvd)?,
            norms,
        )
    };
    // The one fused launch. `None` when the card or a width does not fit it.
    let new = |srcs: [(&Tensor, usize); 3], norms: bool| -> crate::tensor::Result<bool> {
        let weights = norms.then(|| {
            [
                (qn.weight(), qn.eps() as f32),
                (kn.weight(), kn.eps() as f32),
            ]
        });
        let out = prefill_rope_kv_write_f16(
            srcs,
            weights,
            &rcos,
            &rsin,
            false,
            &slot_dev,
            &slots_i32,
            store.k(),
            store.v(),
            nh,
            nkv,
            hd,
        )?;
        let fits = out.is_some();
        std::hint::black_box(out);
        Ok(fits)
    };

    let fail = Fail::default();
    let mut rows_out: Vec<Measured> = Vec::new();
    let mut measure = |variant: String,
                       work: &mut dyn FnMut() -> crate::tensor::Result<()>|
     -> Result<(), String> {
        let samples = run_for(
            target,
            super::stats::timed(|| {
                fail.ok(work());
                fail.ok(dev.synchronize());
            }),
        );
        fail.check(&variant)?;
        rows_out.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: None,
        });
        Ok(())
    };

    let norm_pair = Some((&qn, &kn));
    measure(
        "old, separate projections, q and k norms: 10 launches".to_string(),
        &mut || old(&q, &k, &v, norm_pair),
    )?;
    measure(
        "old, fused projection, q and k norms: 13 launches".to_string(),
        &mut || old_fused(norm_pair),
    )?;
    measure(
        "old, separate projections, no norms: 4 launches".to_string(),
        &mut || old(&q, &k, &v, None),
    )?;
    let mut fused_variant = |variant: String,
                             srcs: [(&Tensor, usize); 3],
                             norms: bool|
     -> Result<(), String> {
        if !new(srcs, norms).map_err(tensor)? {
            rows_out.push(Measured {
                variant,
                bytes: None,
                samples_us: Vec::new(),
                note: Some("the fused launch does not fit this card or these widths".to_string()),
            });
            return Ok(());
        }
        let samples = run_for(
            target,
            super::stats::timed(|| {
                fail.ok(new(srcs, norms).map(|_| ()));
                fail.ok(dev.synchronize());
            }),
        );
        fail.check(&variant)?;
        rows_out.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: None,
        });
        Ok(())
    };
    fused_variant(
        "new, separate projections, q and k norms: 1 launch".to_string(),
        [(&q, 0), (&k, 0), (&v, 0)],
        true,
    )?;
    fused_variant(
        "new, fused projection, q and k norms: 1 launch".to_string(),
        [(&qkv, 0), (&qkv, qd), (&qkv, qd + kvd)],
        true,
    )?;
    fused_variant(
        "new, separate projections, no norms: 1 launch".to_string(),
        [(&q, 0), (&k, 0), (&v, 0)],
        false,
    )?;

    // The activation group: the fused gate and up projection of the layer's feed-forward block.
    let ffn = ffn_len(cx.options);
    if ffn == 0 {
        return Err("--inter must be above zero".to_string());
    }
    let gu = synth(2 * ffn, DType::F16)?;
    let old_act = || -> crate::tensor::Result<()> {
        let g = gu.narrow(1, 0, ffn)?;
        let u = gu.narrow(1, ffn, ffn)?;
        std::hint::black_box(silu_mul_f16(&g, &u)?);
        Ok(())
    };
    let samples = run_for(
        target,
        super::stats::timed(|| {
            fail.ok(old_act());
            fail.ok(dev.synchronize());
        }),
    );
    let variant = "MLP activation, old, 2 narrow copies and silu_mul: 3 launches".to_string();
    fail.check(&variant)?;
    rows_out.push(Measured {
        variant,
        bytes: None,
        samples_us: samples,
        note: None,
    });
    let variant = "MLP activation, new, one kernel from the fused buffer: 1 launch".to_string();
    let fits = silu_mul_gu_f16(&gu, ffn).map_err(tensor)?.is_some();
    if fits {
        let samples = run_for(
            target,
            super::stats::timed(|| {
                fail.ok(silu_mul_gu_f16(&gu, ffn).map(|out| {
                    std::hint::black_box(out);
                }));
                fail.ok(dev.synchronize());
            }),
        );
        fail.check(&variant)?;
        rows_out.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: None,
        });
    } else {
        rows_out.push(Measured {
            variant,
            bytes: None,
            samples_us: Vec::new(),
            note: Some("the fused launch does not fit this card or this width".to_string()),
        });
    }

    // The attention I/O group: the fused input launch and the attention, one sequence of all rows.
    let stage = |head_major: bool| -> crate::tensor::Result<Option<(Tensor, Tensor, Tensor)>> {
        let srcs = [(&qkv, 0), (&qkv, qd), (&qkv, qd + kvd)];
        let weights = Some([
            (qn.weight(), qn.eps() as f32),
            (kn.weight(), kn.eps() as f32),
        ]);
        if head_major {
            prefill_rope_kv_write_head_major_f16(
                srcs,
                weights,
                &rcos,
                &rsin,
                false,
                &slot_dev,
                &slots_i32,
                store.k(),
                store.v(),
                nh,
                nkv,
                hd,
            )
        } else {
            prefill_rope_kv_write_f16(
                srcs,
                weights,
                &rcos,
                &rsin,
                false,
                &slot_dev,
                &slots_i32,
                store.k(),
                store.v(),
                nh,
                nkv,
                hd,
            )
        }
    };
    let scale = 1.0 / (hd as f32).sqrt();
    let lens = [rows];
    let old_attn = || -> crate::tensor::Result<()> {
        if let Some((q, k, v)) = stage(false)? {
            std::hint::black_box(paged_prefill_attn_seqs(
                &q, &k, &v, &lens, nh, nkv, hd, scale,
            )?);
        }
        Ok(())
    };
    let new_attn = || -> crate::tensor::Result<()> {
        if let Some((q, k, v)) = stage(true)? {
            std::hint::black_box(paged_prefill_attn_seqs_head_major(
                &q, &k, &v, &lens, nh, nkv, hd, scale,
            )?);
        }
        Ok(())
    };
    // The new row only means something where both launches run, not their fallback.
    let new_runs = match stage(true).map_err(tensor)? {
        Some((q, k, v)) => flash_prefill_gqa_varlen_head_major(&q, &k, &v, &lens, scale)
            .map_err(tensor)?
            .is_some(),
        None => false,
    };
    let old_runs = stage(false).map_err(tensor)?.is_some();
    let attention_rows: [(&str, bool, &dyn Fn() -> crate::tensor::Result<()>); 2] = [
        (
            "attention I/O, old, fused stage, q cast, q transpose copy, flash, output cast: 5 launches",
            old_runs,
            &old_attn,
        ),
        (
            "attention I/O, new, q written head-major f32 by the fused stage, flash with f16 output: 2 launches",
            new_runs,
            &new_attn,
        ),
    ];
    for (name, runs, work) in attention_rows {
        let variant = name.to_string();
        if !runs {
            rows_out.push(Measured {
                variant,
                bytes: None,
                samples_us: Vec::new(),
                note: Some(
                    "the fused launch or the flash kernel does not fit this card or these widths"
                        .to_string(),
                ),
            });
            continue;
        }
        let samples = run_for(
            target,
            super::stats::timed(|| {
                fail.ok(work());
                fail.ok(dev.synchronize());
            }),
        );
        fail.check(&variant)?;
        rows_out.push(Measured {
            variant,
            bytes: None,
            samples_us: samples,
            note: None,
        });
    }

    // Each new variant is set against the old one on the same inputs.
    for (old_at, new_at) in [(0, 3), (1, 4), (2, 5), (6, 7), (8, 9)] {
        if let Some(note) = rows_out
            .get(old_at)
            .zip(rows_out.get(new_at))
            .and_then(|(o, n)| ratio_note(o, n))
        {
            rows_out[new_at].note = Some(note);
        }
    }
    Ok(rows_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn opts(args: &[&str]) -> Options {
        let mut all = vec!["microbench", "prefill-layer"];
        all.extend_from_slice(args);
        Options::parse_from(all)
    }

    #[test]
    fn the_options_name_the_heads_without_a_file() {
        let o = opts(&["--heads", "32", "--kv-heads", "8", "--head-dim", "128"]);
        let h = resolve_heads(&o, &FileHeads::default(), None).unwrap();
        assert_eq!((h.n_head, h.n_kv, h.head_dim), (32, 8, 128));
    }

    #[test]
    fn a_run_with_neither_a_file_nor_the_heads_is_refused() {
        let err = resolve_heads(&opts(&[]), &FileHeads::default(), None).unwrap_err();
        assert!(err.contains("--shapes-from"));
    }

    #[test]
    fn the_file_fills_what_the_options_leave_and_the_options_win() {
        let file = FileHeads {
            n_head: Some(32),
            n_kv: None,
            head_dim: None,
            rope_base: Some(10000.0),
        };
        let h = resolve_heads(&opts(&[]), &file, Some(4096)).unwrap();
        assert_eq!((h.n_head, h.n_kv, h.head_dim), (32, 32, 128));
        assert_eq!(h.rope_base, 10000.0);
        let h = resolve_heads(&opts(&["--kv-heads", "8"]), &file, Some(4096)).unwrap();
        assert_eq!(h.n_kv, 8);
    }

    #[test]
    fn heads_that_do_not_group_are_refused() {
        let o = opts(&["--heads", "32", "--kv-heads", "5", "--head-dim", "128"]);
        assert!(resolve_heads(&o, &FileHeads::default(), None).is_err());
    }

    #[test]
    fn the_activation_group_runs_at_the_inter_option() {
        assert_eq!(ffn_len(&opts(&["--inter", "12288"])), 12288);
    }

    #[test]
    fn the_ratio_reads_the_medians() {
        let row = |v: &str, s: &[f64]| Measured {
            variant: v.to_string(),
            bytes: None,
            samples_us: s.to_vec(),
            note: None,
        };
        let note = ratio_note(&row("a", &[40.0, 40.0]), &row("b", &[10.0, 10.0])).unwrap();
        assert!(note.starts_with("4.00x"));
        assert!(ratio_note(&row("a", &[]), &row("b", &[1.0])).is_none());
    }
}
