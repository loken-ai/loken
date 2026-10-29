//! One routed expert's quantised weights host to card: what a synchronous upload inside a token
//! costs and what overlap buys. The three projections are copied on the compute stream, on a
//! copy stream, and on a copy stream ordered after the compute stream's queue as the expert
//! uploader orders its allocation; from pageable memory and from page-locked memory; with the
//! card idle and with a kernel holding the compute stream for the time of one idle upload. The
//! kernel's own time beside each upload says what the upload cost the card.

use super::card::{copy_stream, device_bytes, elapsed_us, Pinned, Spin};
use super::stats::{run_for, timed, Summary};
use super::{Context, Fail, Measured};
use crate::tensor::gpu_runtime::driver::CudaSlice;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Pageable,
    Pinned,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    Compute,
    Copy,
    /// The copy stream, waiting first for an event recorded on the compute stream.
    CopyAfterCompute,
}

const SOURCES: [(Source, &str); 2] = [(Source::Pageable, "pageable"), (Source::Pinned, "pinned")];
const LANES: [(Lane, &str); 3] = [
    (Lane::Compute, "compute stream"),
    (Lane::Copy, "copy stream"),
    (
        Lane::CopyAfterCompute,
        "copy stream after the compute stream's event",
    ),
];

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let card = cx.card.ok_or("expert-upload needs a card")?;
    let dev = &card.dev;
    let compute = dev.stream().clone();
    let copy = copy_stream(card)?;
    let target = Duration::from_secs_f64(cx.options.seconds);
    let sizes: Vec<usize> = cx
        .shapes
        .projections
        .iter()
        .map(|p| p.bytes as usize)
        .collect();
    let bytes = Some(cx.shapes.expert_bytes());

    let pageable: Vec<Vec<u8>> = sizes
        .iter()
        .map(|&n| (0..n).map(|i| (i % 253) as u8).collect())
        .collect();
    let pinned = sizes
        .iter()
        .zip(&pageable)
        .map(|(&n, src)| {
            let mut p = Pinned::new(card, n)?;
            p.as_mut_slice().copy_from_slice(src);
            Ok::<_, String>(p)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The destinations are allocated on the compute stream once, as the uploader allocates
    // them, and reused: the copies alone are timed.
    let mut on_card = sizes
        .iter()
        .map(|&n| device_bytes(&compute, n))
        .collect::<Result<Vec<CudaSlice<u8>>, _>>()?;
    let done = dev.new_event().map_err(|e| format!("event: {e:?}"))?;
    let k0 = dev.new_event().map_err(|e| format!("event: {e:?}"))?;
    let k1 = dev.new_event().map_err(|e| format!("event: {e:?}"))?;
    let fail = Fail::default();

    // Queue the three copies on the lane and record `done` behind them.
    let upload = |source: Source, lane: Lane, on_card: &mut [CudaSlice<u8>]| {
        let lane_stream = match lane {
            Lane::Compute => &compute,
            Lane::Copy | Lane::CopyAfterCompute => &copy,
        };
        if lane == Lane::CopyAfterCompute {
            match compute.record_event(None) {
                Ok(ready) => fail.ok(copy.wait(&ready)),
                Err(e) => fail.ok(Err(e)),
            }
        }
        for (i, dst) in on_card.iter_mut().enumerate() {
            let src: &[u8] = match source {
                Source::Pageable => &pageable[i],
                Source::Pinned => pinned[i].as_slice(),
            };
            fail.ok(lane_stream.memcpy_htod(src, dst));
        }
        fail.ok(done.record(lane_stream));
    };

    let mut rows = Vec::new();
    let mut idle_pinned_compute_us = None;
    for (source, source_name) in SOURCES {
        for (lane, lane_name) in LANES {
            let samples = run_for(
                target,
                timed(|| {
                    upload(source, lane, &mut on_card);
                    fail.ok(done.synchronize());
                }),
            );
            let variant = format!("upload, {source_name}, {lane_name}, card idle");
            fail.check(&variant)?;
            if source == Source::Pinned && lane == Lane::Compute {
                idle_pinned_compute_us = Some(Summary::of(&samples).median_us);
            }
            rows.push(Measured {
                variant,
                bytes,
                samples_us: samples,
                note: None,
            });
        }
    }

    // The kernel runs as long as one idle upload from page-locked memory on the compute stream.
    let spin = Spin::lasting(card, idle_pinned_compute_us.unwrap_or(0.0))?;
    let alone = run_for(target, || {
        fail.ok(k0.record(&compute));
        fail.ok(spin.launch(&compute));
        fail.ok(k1.record(&compute));
        fail.ok(compute.synchronize());
        elapsed_us(&k0, &k1).unwrap_or(0.0)
    });
    fail.check("kernel alone")?;
    let alone_us = Summary::of(&alone).median_us;
    rows.push(Measured {
        variant: format!("kernel alone, built for {:.0} us", spin.us),
        bytes: None,
        samples_us: alone,
        note: None,
    });

    for (source, source_name) in SOURCES {
        for (lane, lane_name) in LANES {
            let mut kernel_us = Vec::new();
            let samples = run_for(target, || {
                fail.ok(k0.record(&compute));
                fail.ok(spin.launch(&compute));
                fail.ok(k1.record(&compute));
                let t = Instant::now();
                upload(source, lane, &mut on_card);
                fail.ok(done.synchronize());
                let us = t.elapsed().as_secs_f64() * 1e6;
                fail.ok(compute.synchronize());
                kernel_us.push(elapsed_us(&k0, &k1).unwrap_or(0.0));
                us
            });
            let variant = format!("upload, {source_name}, {lane_name}, kernel running");
            fail.check(&variant)?;
            // The warm-up calls recorded kernel times too: the timed ones are the last.
            let kernel_us = kernel_us.split_off(kernel_us.len() - samples.len());
            rows.push(Measured {
                variant,
                bytes,
                samples_us: samples,
                note: None,
            });
            rows.push(Measured {
                variant: format!("kernel beside upload, {source_name}, {lane_name}"),
                bytes: None,
                samples_us: kernel_us,
                note: Some(format!("alone: {alone_us:.1} us")),
            });
        }
    }
    Ok(rows)
}
