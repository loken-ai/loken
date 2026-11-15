//! `expert-pairs`: 1 layer's routed pairs on a card for a decoded token and for the rows of a
//! verify, at a model's widths and formats. The fused expert kernels of a layer without exact
//! kernels run beside the exact kernels, which are timed 1 launch at a time and in sequence.
//!
//! The weights are synthetic bytes of the stacks' formats, with a finite scale in front of each
//! block; the times do not depend on the values. The stacks hold the layer's experts, as many as
//! the card's free memory takes, and each row routes to distinct experts drawn as a router's
//! choice, so the pairs read their weights from the card's memory as a served model does.

use super::stats::run_for;
use super::{Context, Fail, Measured};
use crate::inference::generic_transformer::expert_tier::ExactPairs;
use crate::tensor::quantized::{GgmlDType, QHostTensor, QTensor};
use crate::tensor::{Device, Tensor};
use std::sync::Arc;
use std::time::Duration;

/// The formats the expert tier serves, by the name a file spells them with.
const SERVED: [GgmlDType; 15] = [
    GgmlDType::Q8_0,
    GgmlDType::Q4K,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q5_0,
    GgmlDType::MxFp4,
    GgmlDType::Q5_1,
    GgmlDType::Iq4Nl,
    GgmlDType::Iq4Xs,
    GgmlDType::Iq3Xxs,
    GgmlDType::Iq3S,
    GgmlDType::Iq2S,
    GgmlDType::Q2_0,
];

/// Pairs a row routes to when no file states it.
const DEFAULT_PAIRS: usize = 8;

/// What the bench runs at: the widths, the formats, and the pairs a row.
struct Layer {
    hidden: usize,
    inter: usize,
    gate: GgmlDType,
    down: GgmlDType,
    pairs: usize,
    /// The layer's routed experts; without a file, as many as the rows' pairs.
    experts: Option<usize>,
}

fn format_of(name: &str) -> Result<GgmlDType, String> {
    SERVED
        .iter()
        .copied()
        .find(|d| d.gguf_name().eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("format {name} is not one the expert tier serves"))
}

/// The layer of the shapes file, or of the options without one.
fn layer(cx: &Context) -> Result<Layer, String> {
    let o = cx.options;
    let Some(path) = &cx.shapes.file else {
        return Ok(Layer {
            hidden: o.hidden,
            inter: o.expert_inter(),
            gate: format_of(&o.gate_format)?,
            down: format_of(&o.down_format)?,
            pairs: DEFAULT_PAIRS,
            experts: None,
        });
    };
    let content = crate::tensor::quantized::gguf_file::open_header(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let layer = cx
        .shapes
        .layer
        .ok_or("the file has no layer of routed experts")?;
    let info = |name: &str| {
        content
            .tensor_infos
            .get(&format!("blk.{layer}.{name}.weight"))
            .ok_or(format!("layer {layer} has no stacked {name}"))
    };
    let (gate, down) = (info("ffn_gate_exps")?, info("ffn_down_exps")?);
    let dims = gate.shape.dims();
    let pairs = content
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".expert_used_count"))
        .and_then(|(_, v)| v.to_u32().ok())
        .ok_or("the file states no expert_used_count")? as usize;
    Ok(Layer {
        hidden: dims[dims.len() - 1],
        inter: dims[dims.len() - 2],
        gate: gate.ggml_dtype,
        down: down.ggml_dtype,
        pairs,
        experts: (dims.len() == 3).then_some(dims[0]),
    })
}

/// `experts` matrices `[rows, cols]` of `dtype` from synthetic bytes, each block's first 2 bytes
/// a small finite f16.
fn stack(
    experts: usize,
    rows: usize,
    cols: usize,
    dtype: GgmlDType,
    seed: u32,
) -> Result<Arc<QTensor>, String> {
    let blocks = experts * rows * cols / dtype.block_size();
    let size = dtype.type_size();
    let mut state = 0x9E37_79B9u32 ^ seed;
    let bytes: Vec<u8> = (0..blocks * size)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match i % size {
                0 => 0x00,
                1 => 0x2C,
                _ => (state >> 24) as u8,
            }
        })
        .collect();
    let host = QHostTensor::from_bytes(&bytes, dtype, vec![experts, rows, cols])
        .map_err(|e| format!("{dtype:?} stack: {e:?}"))?;
    QTensor::from_native(Arc::new(host), &Device::Cpu)
        .map(Arc::new)
        .map_err(|e| format!("{dtype:?} stack: {e:?}"))
}

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let card = cx.card.ok_or("expert-pairs needs a card")?;
    let o = cx.options;
    let l = layer(cx)?;
    let blocks = if o.block.is_empty() {
        vec![1]
    } else {
        o.block.clone()
    };
    if blocks.contains(&0) {
        return Err("--block needs row counts above zero".to_string());
    }
    let rows_max = blocks.iter().copied().max().unwrap_or(1);
    let device = Device::Cuda(card.dev.clone());
    // The layer's experts, as many as the card's free memory holds once, beside the pairs'
    // buffers that 1 more expert's bytes stand for.
    let expert_bytes =
        |d: GgmlDType, rows: usize, cols: usize| rows * cols / d.block_size() * d.type_size();
    let one = 2 * expert_bytes(l.gate, l.inter, l.hidden) + expert_bytes(l.down, l.hidden, l.inter);
    let free = crate::inference::place::room::free_now(&device)
        .ok_or_else(|| "the card does not report its free memory".to_string())?
        as usize;
    let wanted = l.experts.unwrap_or(l.pairs * rows_max);
    let experts = wanted.min((free / one).saturating_sub(1)).max(l.pairs);
    let held = (experts < wanted).then(|| {
        format!("{experts} of the layer's {wanted} experts: what the card's free memory holds")
    });
    let host = [
        stack(experts, l.inter, l.hidden, l.gate, 1)?,
        stack(experts, l.inter, l.hidden, l.gate, 2)?,
        stack(experts, l.hidden, l.inter, l.down, 3)?,
    ];
    // The bytes of 1 expert's gate and up, and of its down.
    let per_expert = |t: &Arc<QTensor>| (t.native_qtensor().data().len() / experts) as u64;
    let (gate_up_bytes, down_bytes) = (
        per_expert(&host[0]) + per_expert(&host[1]),
        per_expert(&host[2]),
    );
    let stream = card.dev.stream().clone();
    // The card's cache: the routings of consecutive calls read more than it holds.
    let cache = card
        .dev
        .context()
        .attribute(crate::tensor::gpu_runtime::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .map_err(|e| format!("card cache size: {e:?}"))?
        .max(0) as u64;
    let (k0, k1) = (
        card.dev
            .new_timed_event()
            .map_err(|e| format!("event: {e:?}"))?,
        card.dev
            .new_timed_event()
            .map_err(|e| format!("event: {e:?}"))?,
    );
    let target = Duration::from_secs_f64(o.seconds);
    let fail = Fail::default();
    // The card's time of the launches `f` queues, between 2 events.
    let timed = |f: &dyn Fn() -> crate::tensor::Result<()>| -> Vec<f64> {
        run_for(target, || {
            fail.ok(k0.record(&stream));
            fail.ok(f());
            fail.ok(k1.record(&stream));
            fail.ok(stream.synchronize());
            fail.value(super::card::elapsed_us(&k0, &k1))
        })
    };
    let mut measured = Vec::new();
    for &rows in &blocks {
        let n = rows * l.pairs;
        let tag = format!("{:?}/{:?}, {rows} x {} pairs", l.gate, l.down, l.pairs);
        // Routings drawn from a fixed seed, each row to `pairs` distinct experts as a router's
        // choice, enough of them that the experts of the others between 2 uses of one exceed the
        // card's cache: a call reads its weights from the card's memory, as a token whose layer
        // ran after others does.
        let call_bytes = n as u64 * (gate_up_bytes + down_bytes);
        let sets = (cache.div_ceil(call_bytes.max(1)) + 1).max(2) as usize;
        let mut state = 0x2545_F491u32 ^ rows as u32;
        let routings: Vec<Vec<u32>> = (0..sets)
            .map(|_| {
                let mut chosen: Vec<u32> = Vec::with_capacity(n);
                for _ in 0..rows {
                    let mut row: Vec<u32> = Vec::with_capacity(l.pairs);
                    while row.len() < l.pairs {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let e = (state >> 8) % experts as u32;
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    chosen.extend(row);
                }
                chosen
            })
            .collect();
        // The routing of the next call.
        let next = std::cell::Cell::new(0usize);
        let turn = || {
            let r = next.get();
            next.set((r + 1) % sets);
            r
        };
        let x = Tensor::from_vec(
            (0..rows * l.hidden)
                .map(|i| (i % 13) as f32 * 0.1 - 0.6)
                .collect::<Vec<f32>>(),
            (rows, l.hidden),
            &device,
        )
        .map_err(|e| format!("{tag}: input: {e:?}"))?;
        let ids = routings
            .iter()
            .map(|r| {
                Tensor::from_vec(r.clone(), n, &Device::Cpu)
                    .and_then(|t| t.to_device(&device))
                    .map_err(|e| format!("{tag}: ids: {e:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let order = Tensor::from_vec((0..n as u32).collect::<Vec<u32>>(), n, &Device::Cpu)
            .and_then(|t| t.to_device(&device))
            .map_err(|e| format!("{tag}: order: {e:?}"))?;
        let w = Tensor::from_vec(vec![1.0 / l.pairs as f32; n], n, &Device::Cpu)
            .and_then(|t| t.to_device(&device))
            .map_err(|e| format!("{tag}: weights: {e:?}"))?;
        let pairs = ExactPairs::new(&device, &host, &x, &routings, l.pairs)
            .map_err(|e| format!("{tag}: exact kernels: {e:?}"))?;
        let owned: Vec<QTensor> = match &pairs {
            Some(_) => Vec::new(),
            None => host
                .iter()
                .map(|t| {
                    QTensor::from_native(t.native_qtensor().clone(), &device)
                        .map_err(|e| format!("{tag}: stacks: {e:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        let card_stacks: Vec<&QTensor> = match &pairs {
            Some(p) => p.card_stacks().iter().collect(),
            None => owned.iter().collect(),
        };
        let (gate_up_read, down_read) =
            (Some(gate_up_bytes * n as u64), Some(down_bytes * n as u64));
        let mut push =
            |variant: &str, bytes: Option<u64>, samples: Vec<f64>, note: Option<String>| {
                measured.push(Measured {
                    variant: format!("{tag}: {variant}"),
                    bytes,
                    samples_us: samples,
                    note,
                })
            };

        let gate_up = |r: usize| {
            crate::inference::moe_cuda::moe_gemm_gguf_gate_up_silu_mul_over(
                &x,
                card_stacks[0],
                card_stacks[1],
                None,
                &order,
                &ids[r],
                l.pairs,
            )
        };
        let down_reduced = |h: &Tensor, r: usize| {
            crate::inference::moe_cuda::moe_gemm_gguf_down_reduce_over(
                h,
                card_stacks[2],
                None,
                &order,
                &ids[r],
                &w,
                l.pairs,
                rows,
                None,
                None,
                None,
            )
        };
        let h = gate_up(0).map_err(|e| format!("{tag}: fused gate and up: {e:?}"))?;
        push(
            "fused gate and up",
            gate_up_read,
            timed(&|| gate_up(turn()).map(drop)),
            held.clone(),
        );
        fail.check("fused gate and up")?;
        push(
            "fused down, pairs",
            down_read,
            timed(&|| {
                crate::inference::moe_cuda::moe_gemm_gguf_down_pairs_over(
                    &h,
                    card_stacks[2],
                    None,
                    &order,
                    &ids[turn()],
                    &w,
                    l.pairs,
                )
                .map(drop)
            }),
            None,
        );
        fail.check("fused down, pairs")?;
        push(
            "fused down, reduced",
            down_read,
            timed(&|| down_reduced(&h, turn()).map(drop)),
            None,
        );
        fail.check("fused down, reduced")?;
        push(
            "fused, gate and up then down reduced",
            Some(call_bytes),
            timed(&|| {
                let r = turn();
                down_reduced(&gate_up(r)?, r).map(drop)
            }),
            None,
        );
        fail.check("fused, gate and up then down reduced")?;

        let Some(p) = &pairs else {
            push(
                "exact kernels",
                None,
                Vec::new(),
                Some("the stacks have no exact form".to_string()),
            );
            continue;
        };
        let step = |f: &dyn Fn() -> crate::tensor::Result<()>| {
            p.select(turn());
            f()
        };
        push(
            "exact quantise x",
            None,
            timed(&|| step(&|| p.quantize_x())),
            None,
        );
        push(
            "exact gate and up",
            gate_up_read,
            timed(&|| step(&|| p.gate_up())),
            None,
        );
        push(
            "exact quantise h",
            None,
            timed(&|| step(&|| p.quantize_h())),
            None,
        );
        push("exact down", down_read, timed(&|| step(&|| p.down())), None);
        push("exact sum", None, timed(&|| step(&|| p.sum())), None);
        push(
            "exact, all 5",
            Some(call_bytes),
            timed(&|| {
                p.select(turn());
                p.quantize_x()?;
                p.gate_up()?;
                p.quantize_h()?;
                p.down()?;
                p.sum()
            }),
            None,
        );
        fail.check("exact kernels")?;
    }
    Ok(measured)
}

/// `host-pairs`: 1 layer's routed pairs on the host cores, at the widths and formats of the
/// layer, through the formats' own dots, as the tier computes the pairs of a layer without exact
/// kernels, and through the exact dots. Each pair's projections also run alone on 1 thread.
///
/// The stacks hold the layer's experts in the process's memory. Consecutive calls take
/// different routings, as many as the layer has experts, so a call reads its weights from memory
/// as a missing expert's are read.
pub fn run_host(cx: &Context) -> Result<Vec<Measured>, String> {
    use crate::inference::generic_transformer::expert_tier::{HostPairs, Projection};
    let o = cx.options;
    let l = layer(cx)?;
    let blocks = if o.block.is_empty() {
        vec![1]
    } else {
        o.block.clone()
    };
    if blocks.contains(&0) {
        return Err("--block needs row counts above zero".to_string());
    }
    let rows_max = blocks.iter().copied().max().unwrap_or(1);
    let experts = l.experts.unwrap_or(l.pairs * rows_max).max(l.pairs);
    let host = [
        stack(experts, l.inter, l.hidden, l.gate, 1)?,
        stack(experts, l.inter, l.hidden, l.gate, 2)?,
        stack(experts, l.hidden, l.inter, l.down, 3)?,
    ];
    let per_expert = |t: &Arc<QTensor>| (t.native_qtensor().data().len() / experts) as u64;
    let (gate_bytes, down_bytes) = (per_expert(&host[0]), per_expert(&host[2]));
    let pair_bytes = 2 * gate_bytes + down_bytes;
    let on_host = HostPairs::new(&host);
    let target = Duration::from_secs_f64(o.seconds);
    let fail = Fail::default();
    let x: Vec<f32> = (0..rows_max * l.hidden)
        .map(|i| (i % 13) as f32 * 0.1 - 0.6)
        .collect();
    let h: Vec<f32> = (0..l.inter).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    // The routings of `rows` rows of `k` distinct experts each, drawn from a fixed seed.
    let routings = |rows: usize, k: usize| -> Vec<Vec<(usize, u32, f32)>> {
        let mut state = 0x2545_F491u32 ^ (rows * k) as u32;
        (0..experts)
            .map(|_| {
                let mut chosen = Vec::with_capacity(rows * k);
                for r in 0..rows {
                    let mut row: Vec<u32> = Vec::with_capacity(k);
                    while row.len() < k {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let e = (state >> 8) % experts as u32;
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    chosen.extend(row.into_iter().map(|e| (r, e, 1.0 / k as f32)));
                }
                chosen
            })
            .collect()
    };
    let next = std::cell::Cell::new(0usize);
    let turn = || {
        let r = next.get();
        next.set((r + 1) % experts);
        r
    };
    let mut measured = Vec::new();
    let names = format!("{:?}/{:?}", l.gate, l.down);
    let mut shapes: Vec<(usize, usize, String)> = vec![(1, 1, "1 pair".to_string())];
    shapes.extend(
        blocks
            .iter()
            .map(|&rows| (rows, l.pairs, format!("{rows} x {} pairs", l.pairs))),
    );
    for (rows, k, what) in shapes {
        let drawn = routings(rows, k);
        let x = &x[..rows * l.hidden];
        let bytes = Some((rows * k) as u64 * pair_bytes);
        measured.push(Measured {
            variant: format!("{names}, {what}: own dots"),
            bytes,
            samples_us: run_for(target, || {
                let pairs = &drawn[turn()];
                let t = std::time::Instant::now();
                std::hint::black_box(fail.value(on_host.own_dots(x, pairs)));
                t.elapsed().as_secs_f64() * 1e6
            }),
            note: None,
        });
        fail.check("own dots")?;
        if !on_host.has_exact() {
            measured.push(Measured {
                variant: format!("{names}, {what}: exact dots"),
                bytes: None,
                samples_us: Vec::new(),
                note: Some("the stacks have no exact form".to_string()),
            });
            continue;
        }
        measured.push(Measured {
            variant: format!("{names}, {what}: exact dots"),
            bytes,
            samples_us: run_for(target, || {
                let pairs = &drawn[turn()];
                let t = std::time::Instant::now();
                std::hint::black_box(fail.value(on_host.exact_dots(x, pairs)));
                t.elapsed().as_secs_f64() * 1e6
            }),
            note: None,
        });
        fail.check("exact dots")?;
    }
    for (p, dtype, input, bytes) in [
        (Projection::Gate, l.gate, &x[..l.hidden], gate_bytes),
        (Projection::Down, l.down, &h[..], down_bytes),
    ] {
        for (own, how) in [(true, "own dot"), (false, "exact dot")] {
            if !own && !on_host.has_exact() {
                continue;
            }
            measured.push(Measured {
                variant: format!("{dtype:?} {p:?} of 1 expert, 1 thread: {how}"),
                bytes: Some(bytes),
                samples_us: run_for(target, || {
                    let e = turn();
                    let t = std::time::Instant::now();
                    std::hint::black_box(on_host.one_thread(p, e, input, own));
                    t.elapsed().as_secs_f64() * 1e6
                }),
                note: None,
            });
        }
    }
    Ok(measured)
}
