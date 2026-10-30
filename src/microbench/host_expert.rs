//! One routed expert computed on the host cores from memory: gate, up and down products of a
//! decode row, the floor of what a missing expert costs once its pages are in memory. Run at
//! the dimensions of the shapes file, on weights of a fixed pattern.

use super::stats::run_for;
use super::{Context, Measured};
use crate::inference::offload::cpu_experts::bench_miss_ms;
use crate::tensor::quantized::GgmlDType;
use std::time::Duration;

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let dim = cx.shapes.hidden;
    let inter = cx.options.inter;
    let gate = format_of(&cx.options.gate_format)?;
    let down = format_of(&cx.options.down_format)?;
    let target = Duration::from_secs_f64(cx.options.seconds);
    // Each sample is a few repeats of the three products, so the clock is not the measure.
    const REPEATS: usize = 8;
    let mut failed = false;
    let samples = run_for(target, || {
        match bench_miss_ms(dim, inter, gate, down, REPEATS) {
            Some(ms) => ms * 1000.0,
            None => {
                failed = true;
                0.0
            }
        }
    });
    if failed {
        return Err(format!(
            "the kernels do not run dimensions {dim} x {inter} in {} and {}",
            cx.options.gate_format, cx.options.down_format
        ));
    }
    let bytes = (inter * dim / gate.block_size() * gate.type_size()) * 2
        + dim * inter / down.block_size() * down.type_size();
    let mut rows = vec![Measured {
        variant: format!(
            "one expert on the cores, {} gate and up, {} down",
            cx.options.gate_format, cx.options.down_format
        ),
        bytes: Some(bytes as u64),
        samples_us: samples,
        note: Some("the same weights every call, so they stay in the caches".to_string()),
    }];
    rows.extend(distinct(cx, gate, down, bytes, target));
    rows.extend(through_the_decode_path(cx, gate, down, bytes, target));
    Ok(rows)
}

/// Number of distinct experts the calls cycle through, and the experts of one call.
const SET: usize = 96;
const PER_CALL: usize = 4;

/// Experts a decode's layer leaves the host, each call on weights not read since a full cycle
/// of the set, so they come from memory: through the pool's regions, and one per thread. Run on
/// weights in the process's own memory and on weights in the page cache of a mapped file, which
/// is where a served model keeps them.
fn distinct(
    cx: &Context,
    gate: GgmlDType,
    down: GgmlDType,
    bytes: usize,
    target: Duration,
) -> Vec<Measured> {
    let (dim, inter) = (cx.shapes.hidden, cx.options.inter);
    let gate_len = inter * (dim / gate.block_size()) * gate.type_size();
    let down_len = dim * (inter / down.block_size()) * down.type_size();
    let heap: Vec<[Vec<u8>; 3]> = (0..SET)
        .map(|i| {
            [
                vec![(i % 251) as u8; gate_len],
                vec![(i % 241) as u8; gate_len],
                vec![(i % 239) as u8; down_len],
            ]
        })
        .collect();
    let weights: Vec<[&[u8]; 3]> = heap
        .iter()
        .map(|e| [e[0].as_slice(), e[1].as_slice(), e[2].as_slice()])
        .collect();
    let mut rows = measure(cx, gate, down, bytes, target, &weights, "own memory");
    if let Some(mapped) = MappedSet::new(&heap) {
        let weights: Vec<[&[u8]; 3]> = mapped.experts();
        rows.extend(measure(
            cx,
            gate,
            down,
            bytes,
            target,
            &weights,
            "mapped file",
        ));
    }
    rows
}

/// The set written to a file next to the process and mapped back, so its pages are the
/// page cache's, in pages of the kernel's choosing, as a served model's are.
struct MappedSet {
    path: std::path::PathBuf,
    map: memmap2::Mmap,
    spans: Vec<[(usize, usize); 3]>,
}

impl MappedSet {
    fn new(heap: &[[Vec<u8>; 3]]) -> Option<Self> {
        use std::io::Write;
        let path = std::path::PathBuf::from(format!("host-expert-set-{}.bin", std::process::id()));
        let mut file = std::fs::File::create(&path).ok()?;
        let mut spans = Vec::new();
        let mut at = 0usize;
        for e in heap {
            let mut one = [(0, 0); 3];
            for (k, w) in e.iter().enumerate() {
                file.write_all(w).ok()?;
                one[k] = (at, w.len());
                at += w.len();
            }
            spans.push(one);
        }
        file.sync_all().ok()?;
        let file = std::fs::File::open(&path).ok()?;
        // Safety: the file is private to this process and not written again while mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        map.iter().step_by(4096).fold(0u8, |a, b| a ^ b);
        Some(Self { path, map, spans })
    }

    fn experts(&self) -> Vec<[&[u8]; 3]> {
        self.spans
            .iter()
            .map(|e| {
                [
                    &self.map[e[0].0..e[0].0 + e[0].1],
                    &self.map[e[1].0..e[1].0 + e[1].1],
                    &self.map[e[2].0..e[2].0 + e[2].1],
                ]
            })
            .collect()
    }
}

impl Drop for MappedSet {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn measure(
    cx: &Context,
    gate: GgmlDType,
    down: GgmlDType,
    bytes: usize,
    target: Duration,
    weights: &[[&[u8]; 3]],
    memory: &str,
) -> Vec<Measured> {
    use crate::tensor::quant_cpu::{matmul_bytes_multi, matmul_bytes_multi_serial};
    let (dim, inter) = (cx.shapes.hidden, cx.options.inter);
    let (x, h) = (vec![0.05f32; dim], vec![0.05f32; inter]);
    let mut next = 0usize;
    let mut pick = || -> Vec<&[&[u8]; 3]> {
        let at = next;
        next = (next + PER_CALL) % SET;
        (0..PER_CALL).map(|k| &weights[(at + k) % SET]).collect()
    };
    let mut gate_o = vec![vec![0f32; inter]; PER_CALL];
    let mut up_o = vec![vec![0f32; inter]; PER_CALL];
    let mut out_o = vec![vec![0f32; dim]; PER_CALL];
    let xs: Vec<&[f32]> = vec![&x; PER_CALL];
    let hs: Vec<&[f32]> = vec![&h; PER_CALL];
    let note = Some(format!(
        "{PER_CALL} experts per call out of {SET} distinct ones, {} MB each",
        bytes / 1_000_000
    ));
    let mut variants = Vec::new();

    let pooled = run_for(target, || {
        let ex = pick();
        let g: Vec<&[u8]> = ex.iter().map(|e| e[0]).collect();
        let u: Vec<&[u8]> = ex.iter().map(|e| e[1]).collect();
        let d: Vec<&[u8]> = ex.iter().map(|e| e[2]).collect();
        let t = std::time::Instant::now();
        let mut o: Vec<&mut [f32]> = gate_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        let _ = matmul_bytes_multi(gate, dim, inter, &g, &xs, &mut o);
        let mut o: Vec<&mut [f32]> = up_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        let _ = matmul_bytes_multi(gate, dim, inter, &u, &xs, &mut o);
        let mut o: Vec<&mut [f32]> = out_o.iter_mut().map(|v| v.as_mut_slice()).collect();
        let _ = matmul_bytes_multi(down, inter, dim, &d, &hs, &mut o);
        t.elapsed().as_secs_f64() * 1e6
    });
    variants.push(Measured {
        variant: format!("4 experts through the pool, 3 regions, {memory}"),
        bytes: Some((PER_CALL * bytes) as u64),
        samples_us: pooled,
        note: note.clone(),
    });

    let threaded = run_for(target, || {
        let ex = pick();
        let t = std::time::Instant::now();
        std::thread::scope(|s| {
            for ((e, go), (uo, oo)) in ex
                .iter()
                .zip(gate_o.iter_mut())
                .zip(up_o.iter_mut().zip(out_o.iter_mut()))
            {
                let (x, h) = (&x, &h);
                s.spawn(move || {
                    let _ = matmul_bytes_multi_serial(gate, dim, inter, &[e[0]], &[x], &mut [go]);
                    let _ = matmul_bytes_multi_serial(gate, dim, inter, &[e[1]], &[x], &mut [uo]);
                    let _ = matmul_bytes_multi_serial(down, inter, dim, &[e[2]], &[h], &mut [oo]);
                });
            }
        });
        t.elapsed().as_secs_f64() * 1e6
    });
    variants.push(Measured {
        variant: format!("4 experts, one thread each, {memory}"),
        bytes: Some((PER_CALL * bytes) as u64),
        samples_us: threaded,
        note: note.clone(),
    });

    let one = run_for(target, || {
        let ex = pick();
        let t = std::time::Instant::now();
        for (e, (go, (uo, oo))) in ex
            .iter()
            .zip(gate_o.iter_mut().zip(up_o.iter_mut().zip(out_o.iter_mut())))
        {
            let _ = matmul_bytes_multi_serial(gate, dim, inter, &[e[0]], &[&x], &mut [go]);
            let _ = matmul_bytes_multi_serial(gate, dim, inter, &[e[1]], &[&x], &mut [uo]);
            let _ = matmul_bytes_multi_serial(down, inter, dim, &[e[2]], &[&h], &mut [oo]);
        }
        t.elapsed().as_secs_f64() * 1e6
    });
    variants.push(Measured {
        variant: format!("4 experts, one after the other on one thread, {memory}"),
        bytes: Some((PER_CALL * bytes) as u64),
        samples_us: one,
        note,
    });
    variants
}

fn format_of(name: &str) -> Result<GgmlDType, String> {
    match name {
        "iq2_xxs" => Ok(GgmlDType::Iq2Xxs),
        "q2_k" => Ok(GgmlDType::Q2K),
        "q4_k" => Ok(GgmlDType::Q4K),
        "q6_k" => Ok(GgmlDType::Q6K),
        "q8_0" => Ok(GgmlDType::Q8_0),
        other => Err(format!("format {other} is not one the bench knows")),
    }
}

/// Experts built as the store builds them, run by the function a decode calls for its host
/// experts, `k` of them per call out of a set far larger than the caches.
fn through_the_decode_path(
    cx: &Context,
    gate: GgmlDType,
    down: GgmlDType,
    bytes: usize,
    target: Duration,
) -> Vec<Measured> {
    use crate::inference::offload::cpu_experts::batched;
    use crate::inference::offload::experts::Expert;
    use crate::inference::offload::projection::Projection;
    use crate::tensor::quantized::QTensor;
    use crate::tensor::Device;
    use std::sync::Arc;
    const SET: usize = 48;
    let (dim, inter) = (cx.shapes.hidden, cx.options.inter);
    let gate_len = inter * (dim / gate.block_size()) * gate.type_size();
    let down_len = dim * (inter / down.block_size()) * down.type_size();
    let quant = |fill: u8, len: usize, dtype: GgmlDType, dims: Vec<usize>| {
        QTensor::from_ggml_bytes(dtype, &vec![fill; len], dims, &Device::Cpu)
            .ok()
            .map(|q| Projection::Quant(Arc::new(q)))
    };
    let mut experts: Vec<Arc<Expert>> = Vec::new();
    for i in 0..SET {
        let (a, b, c) = (
            quant((i % 251) as u8, gate_len, gate, vec![inter, dim]),
            quant((i % 241) as u8, gate_len, gate, vec![inter, dim]),
            quant((i % 239) as u8, down_len, down, vec![dim, inter]),
        );
        match (a, b, c) {
            (Some(w1), Some(w3), Some(w2)) => experts.push(Arc::new(Expert { w1, w2, w3 })),
            _ => return Vec::new(),
        }
    }
    let x = vec![0.05f32; dim];
    let mut rows = Vec::new();
    for k in [4usize, 6] {
        let mut next = 0usize;
        let samples = run_for(target, || {
            let at = next;
            next = (next + k) % SET;
            let pick: Vec<Arc<Expert>> = (0..k).map(|j| experts[(at + j) % SET].clone()).collect();
            let t = std::time::Instant::now();
            let _ = batched(&pick, &x, 7.0);
            t.elapsed().as_secs_f64() * 1e6
        });
        rows.push(Measured {
            variant: format!("{k} experts through the decode's host path"),
            bytes: Some((k * bytes) as u64),
            samples_us: samples,
            note: Some(format!("{k} experts per call out of {SET} distinct ones")),
        });
    }
    rows
}
