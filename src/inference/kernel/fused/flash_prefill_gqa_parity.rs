use crate::tensor::{DType, Device, Tensor};

/// A fixed pseudo-random fill in [-amp, amp): the same inputs on every run.
fn fill(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * amp
        })
        .collect()
}

struct Case {
    n_head: usize,
    n_kv: usize,
    hd: usize,
    seq_q: usize,
    seq_kv: usize,
    window: usize,
}

/// Exact attention in F64 over F16-rounded K/V (what the cache holds), for query token
/// `i` at position `i + seq_kv - seq_q`. Output `[seq_q, n_head, hd]`.
fn reference(c: &Case, q: &[f32], k: &[f32], v: &[f32], scale: f64) -> Vec<f32> {
    let g = c.n_head / c.n_kv;
    let off = c.seq_kv - c.seq_q;
    let mut out = vec![0f32; c.seq_q * c.n_head * c.hd];
    for h in 0..c.n_head {
        let kvh = h / g;
        for i in 0..c.seq_q {
            let pos = i + off;
            let qi = &q[(h * c.seq_q + i) * c.hd..][..c.hd];
            let keys: Vec<usize> = (0..=pos)
                .filter(|&j| c.window == 0 || j + c.window >= pos)
                .collect();
            let s: Vec<f64> = keys
                .iter()
                .map(|&j| {
                    let kj = &k[(j * c.n_kv + kvh) * c.hd..][..c.hd];
                    qi.iter()
                        .zip(kj)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            let o = &mut out[(i * c.n_head + h) * c.hd..][..c.hd];
            for (idx, &j) in keys.iter().enumerate() {
                let vj = &v[(j * c.n_kv + kvh) * c.hd..][..c.hd];
                for d in 0..c.hd {
                    o[d] += (e[idx] / z * vj[d] as f64) as f32;
                }
            }
        }
    }
    out
}

/// The F32 chain the layer runs without the kernel: K/V widened and repeated per query head,
/// masked scores, softmax, then V. Output `[seq_q, n_head, hd]`.
fn chain(c: &Case, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, dev: &Device) -> Vec<f32> {
    let g = c.n_head / c.n_kv;
    let widen = |t: &Tensor| {
        let t = t.to_dtype(DType::F32).unwrap().contiguous().unwrap();
        crate::tensor::ops::repeat_kv(t, g)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let (k, v) = (widen(k), widen(v));
    let off = c.seq_kv - c.seq_q;
    let mut m = vec![0f32; c.seq_q * c.seq_kv];
    for i in 0..c.seq_q {
        for j in 0..c.seq_kv {
            let pos = i + off;
            if j > pos || (c.window > 0 && j + c.window < pos) {
                m[i * c.seq_kv + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(m, (c.seq_q, c.seq_kv), dev).unwrap();
    let s = q
        .matmul(&k.t().unwrap())
        .unwrap()
        .affine(scale, 0.0)
        .unwrap()
        .broadcast_add(&mask)
        .unwrap();
    let p = crate::tensor::ops::softmax_last_dim(&s).unwrap();
    p.matmul(&v)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The GQA flash prefill against exact attention, over the shapes prefill produces: a first
/// chunk, a later chunk over a longer history, group sizes that do not divide the tile,
/// lengths off the tile grid, both head sizes, and a sliding window.
#[test]
#[ignore = "needs a CUDA device"]
fn flash_prefill_gqa_matches_exact_attention() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let cases = [
        Case {
            n_head: 32,
            n_kv: 8,
            hd: 128,
            seq_q: 77,
            seq_kv: 77,
            window: 0,
        },
        Case {
            n_head: 32,
            n_kv: 8,
            hd: 128,
            seq_q: 100,
            seq_kv: 613,
            window: 0,
        },
        Case {
            n_head: 32,
            n_kv: 8,
            hd: 64,
            seq_q: 130,
            seq_kv: 300,
            window: 0,
        },
        Case {
            n_head: 14,
            n_kv: 2,
            hd: 64,
            seq_q: 45,
            seq_kv: 190,
            window: 0,
        },
        Case {
            n_head: 28,
            n_kv: 4,
            hd: 128,
            seq_q: 33,
            seq_kv: 33,
            window: 0,
        },
        Case {
            n_head: 16,
            n_kv: 16,
            hd: 128,
            seq_q: 70,
            seq_kv: 250,
            window: 0,
        },
        Case {
            n_head: 8,
            n_kv: 4,
            hd: 128,
            seq_q: 90,
            seq_kv: 400,
            window: 100,
        },
    ];
    for c in &cases {
        let scale = 1.0 / (c.hd as f64).sqrt();
        // Q as the layer holds it: [1, n_head, seq_q, hd], F32. Magnitudes large enough
        // that the softmax is peaked, as trained attention is.
        let qv = fill(c.n_head * c.seq_q * c.hd, 1 + c.hd as u64, 3.0);
        // K/V in the cache's stored order [seq_kv, n_kv, hd], rounded to F16.
        let round = |x: Vec<f32>| -> Vec<f32> {
            x.into_iter()
                .map(|a| half::f16::from_f32(a).to_f32())
                .collect()
        };
        let kv = round(fill(c.seq_kv * c.n_kv * c.hd, 2 + c.seq_kv as u64, 3.0));
        let vv = round(fill(c.seq_kv * c.n_kv * c.hd, 3 + c.seq_q as u64, 1.0));
        let q = Tensor::from_vec(qv.clone(), (1, c.n_head, c.seq_q, c.hd), &dev).unwrap();
        let as_view = |x: &[f32]| {
            Tensor::from_vec(x.to_vec(), (1, c.seq_kv, c.n_kv, c.hd), &dev)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
                .permute((0, 2, 1, 3))
                .unwrap()
        };
        let (k, v) = (as_view(&kv), as_view(&vv));
        let got = match super::flash_prefill_gqa(&q, &k, &v, scale as f32, c.window).unwrap() {
            Some(y) => y,
            None => return, // device below the kernel's architecture
        };
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = reference(c, &qv, &kv, &vv, scale);
        let chain = chain(c, &q, &k, &v, scale as f32, &dev);
        let err = |x: &[f32]| {
            let num: f64 = x
                .iter()
                .zip(&want)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();
            let den: f64 = want
                .iter()
                .map(|b| (*b as f64).powi(2))
                .sum::<f64>()
                .max(1e-12);
            (num / den).sqrt()
        };
        let rel_rms = err(&got);
        println!(
            "gqa flash nh={} nkv={} hd={} seq_q={} seq_kv={} window={}: rel_rms={rel_rms:.2e} (F32 chain {:.2e})",
            c.n_head,
            c.n_kv,
            c.hd,
            c.seq_q,
            c.seq_kv,
            c.window,
            err(&chain)
        );
        // F16 operands with F32 accumulation, as every flash attention runs: the error is
        // the rounding of Q and of the probabilities, below one part in ten thousand.
        assert!(
            rel_rms < 5e-4,
            "rel_rms {rel_rms} for nh={} hd={}",
            c.n_head,
            c.hd
        );
    }
}

/// Throughput of the GQA flash prefill over late prefill chunks: a chunk of queries
/// against a growing history, reported as achieved tensor TFLOP/s.
#[test]
#[ignore = "needs a CUDA device; a measurement, not a check"]
fn flash_prefill_gqa_throughput() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    for &(n_head, n_kv, hd) in &[(32usize, 8usize, 128usize), (32, 8, 64)] {
        for &(seq_q, seq_kv) in &[
            (512usize, 512usize),
            (512, 4096),
            (512, 7168),
            (2048, 2048),
            (2048, 7168),
        ] {
            let q = Tensor::from_vec(
                fill(n_head * seq_q * hd, 1, 1.0),
                (1, n_head, seq_q, hd),
                &dev,
            )
            .unwrap();
            let kv = |seed| {
                Tensor::from_vec(
                    fill(seq_kv * n_kv * hd, seed, 1.0),
                    (1, seq_kv, n_kv, hd),
                    &dev,
                )
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
                .permute((0, 2, 1, 3))
                .unwrap()
            };
            let (k, v) = (kv(2), kv(3));
            let scale = 1.0 / (hd as f32).sqrt();
            let run = || {
                super::flash_prefill_gqa(&q, &k, &v, scale, 0)
                    .unwrap()
                    .unwrap()
            };
            let _ = run();
            dev.synchronize().unwrap();
            let iters = 20;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                let _ = run();
            }
            dev.synchronize().unwrap();
            let secs = t.elapsed().as_secs_f64() / iters as f64;
            // Keys visible per query summed over the chunk (causal, offset seq_kv - seq_q).
            let off = (seq_kv - seq_q) as f64;
            let keys = seq_q as f64 * off + (seq_q as f64) * (seq_q as f64 + 1.0) / 2.0;
            let flops = 4.0 * keys * n_head as f64 * hd as f64;
            println!(
                "nh={n_head} nkv={n_kv} hd={hd} seq_q={seq_q} seq_kv={seq_kv}: {:.0} us, {:.1} TFLOP/s",
                secs * 1e6,
                flops / secs / 1e12
            );
        }
    }
}
