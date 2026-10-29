//! One activation row host to card and back: what one host-card crossing costs, the unit behind
//! a layer's count of crossings. Synchronous copies from pageable memory as a decode step makes
//! them, asynchronous copies from page-locked memory waited on with an event, several rows in
//! separate copies against the same rows packed in one, and a synchronisation with nothing
//! queued.

use super::card::{device_bytes, Pinned};
use super::stats::{run_for, timed};
use super::{Context, Fail, Measured};
use std::time::Duration;

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let card = cx.card.ok_or("crossing needs a card")?;
    let dev = &card.dev;
    let stream = dev.stream().clone();
    let row = cx.shapes.row_bytes();
    let pack = cx.options.pack.max(1);
    let total = row * pack;
    let target = Duration::from_secs_f64(cx.options.seconds);

    let host: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
    let mut back = vec![0u8; total];
    let mut pinned_in = Pinned::new(card, total)?;
    pinned_in.as_mut_slice().copy_from_slice(&host);
    let mut pinned_out = Pinned::new(card, total)?;
    let mut on_card = device_bytes(&stream, total)?;
    let wait = dev.new_event().map_err(|e| format!("event: {e:?}"))?;
    let fail = Fail::default();

    let mut rows = Vec::new();
    let mut measure = |variant: String, bytes: Option<u64>, body: &mut dyn FnMut()| {
        let samples = run_for(target, timed(body));
        fail.check(&variant)?;
        rows.push(Measured {
            variant,
            bytes,
            samples_us: samples,
            note: None,
        });
        Ok::<(), String>(())
    };

    measure(
        "h2d, pageable, synchronous".to_string(),
        Some(row as u64),
        &mut || {
            fail.ok(stream.memcpy_htod(&host[..row], &mut on_card.slice_mut(0..row)));
            fail.ok(stream.synchronize());
        },
    )?;
    measure(
        "d2h, pageable, synchronous".to_string(),
        Some(row as u64),
        &mut || {
            fail.ok(stream.memcpy_dtoh(&on_card.slice(0..row), &mut back[..row]));
            fail.ok(stream.synchronize());
        },
    )?;
    measure(
        "round trip, pageable, synchronous".to_string(),
        Some(2 * row as u64),
        &mut || {
            fail.ok(stream.memcpy_htod(&host[..row], &mut on_card.slice_mut(0..row)));
            fail.ok(stream.memcpy_dtoh(&on_card.slice(0..row), &mut back[..row]));
            fail.ok(stream.synchronize());
        },
    )?;
    measure(
        "h2d, pinned, asynchronous, event wait".to_string(),
        Some(row as u64),
        &mut || {
            fail.ok(
                stream.memcpy_htod(&pinned_in.as_slice()[..row], &mut on_card.slice_mut(0..row))
            );
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        "d2h, pinned, asynchronous, event wait".to_string(),
        Some(row as u64),
        &mut || {
            fail.ok(stream.memcpy_dtoh(
                &on_card.slice(0..row),
                &mut pinned_out.as_mut_slice()[..row],
            ));
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        "round trip, pinned, asynchronous, event wait".to_string(),
        Some(2 * row as u64),
        &mut || {
            fail.ok(
                stream.memcpy_htod(&pinned_in.as_slice()[..row], &mut on_card.slice_mut(0..row))
            );
            fail.ok(stream.memcpy_dtoh(
                &on_card.slice(0..row),
                &mut pinned_out.as_mut_slice()[..row],
            ));
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        format!("{pack} rows, separate copies each way, one event wait"),
        Some(2 * total as u64),
        &mut || {
            for i in 0..pack {
                let at = i * row..(i + 1) * row;
                fail.ok(stream.memcpy_htod(
                    &pinned_in.as_slice()[at.clone()],
                    &mut on_card.slice_mut(at.clone()),
                ));
                fail.ok(stream.memcpy_dtoh(
                    &on_card.slice(at.clone()),
                    &mut pinned_out.as_mut_slice()[at],
                ));
            }
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        format!("{pack} rows packed, one copy each way, event wait"),
        Some(2 * total as u64),
        &mut || {
            fail.ok(stream.memcpy_htod(pinned_in.as_slice(), &mut on_card));
            fail.ok(stream.memcpy_dtoh(&on_card, pinned_out.as_mut_slice()));
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    measure(
        "stream synchronise, nothing queued".to_string(),
        None,
        &mut || fail.ok(stream.synchronize()),
    )?;
    measure(
        "event record and wait, nothing queued".to_string(),
        None,
        &mut || {
            fail.ok(wait.record(&stream));
            fail.ok(wait.synchronize());
        },
    )?;
    Ok(rows)
}
