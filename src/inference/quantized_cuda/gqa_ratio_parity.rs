//! The Q8 KV decode kernels, judged per head shape against attention computed on the host.
//!
//! Every shape the load gate admits must give the same answer through each decode path: the
//! eager output over given probabilities, the graph's device-position score + fused
//! softmax-output chain, the eager fused softmax-output over raw scores, and the split-K flash
//! decode where it applies. A shape served by the wrong instance reads or writes a neighbouring
//! head and still produces a plausible distribution, so only a reference catches it.
use super::*;
use crate::tensor::kernel_ffi::CudaDevice;

const N_KV: usize = 2;
const SEQ: usize = 40;
const MAX_SEQ: usize = 64;
const HEAD_DIMS: [usize; 4] = [64, 96, 128, 256];
const RATIOS: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 16];
/// Shapes the split-K flash decode is built for; outside them its caller takes the chain.
const FLASH_HEAD_DIMS: [usize; 2] = [64, 128];
const FLASH_MAX_RATIO: usize = 8;

/// Q8_0 blocks laid out `[max_seq, n_kv, head_dim]`, with their dequantised values.
fn q8_blob(hd: usize, salt: usize) -> (Vec<u8>, Vec<f32>) {
    let nblocks = MAX_SEQ * N_KV * (hd / 32);
    let mut blob = Vec::with_capacity(nblocks * 34);
    let mut vals = Vec::with_capacity(nblocks * 32);
    for i in 0..nblocks {
        let d = half::f16::from_f32(0.02 + ((i + salt) % 5) as f32 * 0.003);
        blob.extend_from_slice(&d.to_le_bytes());
        for j in 0..32 {
            let q = (((i * 7 + j * 3 + salt) % 23) as i32 - 11) as i8;
            blob.push(q as u8);
            vals.push(q as f32 * d.to_f32());
        }
    }
    (blob, vals)
}

/// Scaled scores `[n_q, SEQ]` of every query head against its KV head.
fn scores(q: &[f32], k: &[f32], hd: usize, nqpk: usize, scale: f32) -> Vec<f32> {
    let n_q = N_KV * nqpk;
    let mut s = vec![0f32; n_q * SEQ];
    for h in 0..n_q {
        let kv = h / nqpk;
        for t in 0..SEQ {
            s[h * SEQ + t] = (0..hd)
                .map(|d| q[h * hd + d] * k[(t * N_KV + kv) * hd + d])
                .sum::<f32>()
                * scale;
        }
    }
    s
}

fn softmax_rows(s: &[f32]) -> Vec<f32> {
    s.chunks(SEQ)
        .flat_map(|row| {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = row.iter().map(|x| (x - m).exp()).sum();
            row.iter().map(move |x| (x - m).exp() / z)
        })
        .collect()
}

fn attend(p: &[f32], v: &[f32], hd: usize, nqpk: usize) -> Vec<f32> {
    let n_q = N_KV * nqpk;
    let mut out = vec![0f32; n_q * hd];
    for h in 0..n_q {
        let kv = h / nqpk;
        for d in 0..hd {
            out[h * hd + d] = (0..SEQ)
                .map(|t| p[h * SEQ + t] * v[(t * N_KV + kv) * hd + d])
                .sum();
        }
    }
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn every_admitted_head_shape_decodes_like_the_reference() {
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no CUDA device; the Q8 attention kernels NOT covered by this run");
        return;
    };
    let stream = dev.native().stream();
    let pos_dev = stream.clone_htod(&[(SEQ - 1) as i32]).unwrap();
    let host = |s: &CudaStorage| {
        stream
            .clone_dtoh(s.as_cuda_slice::<f32>().unwrap())
            .unwrap()
    };

    let mut failures = Vec::new();
    for hd in HEAD_DIMS {
        let (kb, kf) = q8_blob(hd, 1);
        let (vb, vf) = q8_blob(hd, 4);
        let k_blob = stream.clone_htod(&kb).unwrap();
        let v_blob = stream.clone_htod(&vb).unwrap();
        let scale = 1.0 / (hd as f32).sqrt();
        for nqpk in RATIOS {
            assert!(
                kv_shape_supported(hd, nqpk),
                "hd {hd} nq {nqpk} not admitted"
            );
            let n_q = N_KV * nqpk;
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i * 13 % 97) as f32) * 0.02 - 0.9)
                .collect();
            let raw = scores(&q, &kf, hd, nqpk, 1.0);
            let scaled: Vec<f32> = raw.iter().map(|x| x * scale).collect();
            let probs = softmax_rows(&scaled);
            let want = attend(&probs, &vf, hd, nqpk);
            let q_dev = stream.clone_htod(&q).unwrap();
            let qv = q_dev.slice(..);
            let mut diffs = Vec::new();

            let dev_scores = attn_score_q8_0_f32_dev_pos(
                &k_blob, &qv, &pos_dev, hd, MAX_SEQ, N_KV, nqpk, 0, &dev,
            )
            .unwrap();
            let sv = dev_scores.as_cuda_slice::<f32>().unwrap().slice(..);
            let chain = attn_softmax_output_q8_0_f32_dev_pos(
                &v_blob, &sv, &pos_dev, hd, MAX_SEQ, N_KV, nqpk, scale, 0, &dev,
            )
            .unwrap();
            diffs.push(("graph chain", max_diff(&host(&chain), &want)));

            let p_dev = stream.clone_htod(&probs).unwrap();
            let eager =
                attn_output_q8_0_f32_gqa(&v_blob, &p_dev.slice(..), hd, SEQ, N_KV, nqpk, &dev)
                    .unwrap();
            diffs.push(("eager output", max_diff(&host(&eager), &want)));

            let s_dev = stream.clone_htod(&scaled).unwrap();
            let fused = attn_softmax_output_q8_0_f32_gqa(
                &v_blob,
                &s_dev.slice(..),
                hd,
                SEQ,
                N_KV,
                nqpk,
                &dev,
            )
            .unwrap();
            diffs.push(("eager softmax-output", max_diff(&host(&fused), &want)));

            if FLASH_HEAD_DIMS.contains(&hd) && nqpk <= FLASH_MAX_RATIO {
                let flash = attn_flash_splitk_q8_gqa_decode_dev_pos(
                    &k_blob, &v_blob, &qv, &pos_dev, hd, N_KV, nqpk, SEQ, 1, scale, &dev,
                )
                .unwrap();
                diffs.push(("flash", max_diff(&host(&flash), &want)));
            }

            for (path, d) in diffs {
                if d > 1e-3 {
                    failures.push(format!("hd={hd} nq={nqpk} {path} off by {d:.2e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// The split-K flash decode at the lengths a conversation reaches: its split count grows with
/// the cache, so a shape right at forty positions says nothing about one at two thousand.
#[test]
fn flash_decode_matches_the_reference_at_long_lengths() {
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no CUDA device; the split-K flash decode NOT covered by this run");
        return;
    };
    let stream = dev.native().stream();
    let mut failures = Vec::new();
    for hd in FLASH_HEAD_DIMS {
        for nqpk in [1usize, 4, 8] {
            for seq in [40usize, 1100, 1200, 1500, 2100, 4000] {
                let n_q = N_KV * nqpk;
                let blocks = seq * N_KV * (hd / 32);
                let blob = |salt: usize| {
                    let mut b = Vec::with_capacity(blocks * 34);
                    let mut v = Vec::with_capacity(blocks * 32);
                    for i in 0..blocks {
                        let d = half::f16::from_f32(0.02 + ((i + salt) % 5) as f32 * 0.003);
                        b.extend_from_slice(&d.to_le_bytes());
                        for j in 0..32 {
                            let q = (((i * 7 + j * 3 + salt) % 23) as i32 - 11) as i8;
                            b.push(q as u8);
                            v.push(q as f32 * d.to_f32());
                        }
                    }
                    (b, v)
                };
                let (kb, kf) = blob(1);
                let (vb, vf) = blob(4);
                let scale = 1.0 / (hd as f32).sqrt();
                let q: Vec<f32> = (0..n_q * hd)
                    .map(|i| ((i * 13 % 97) as f32) * 0.02 - 0.9)
                    .collect();
                let mut want = vec![0f32; n_q * hd];
                for h in 0..n_q {
                    let kv = h / nqpk;
                    let s: Vec<f32> = (0..seq)
                        .map(|t| {
                            (0..hd)
                                .map(|d| q[h * hd + d] * kf[(t * N_KV + kv) * hd + d])
                                .sum::<f32>()
                                * scale
                        })
                        .collect();
                    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let z: f32 = s.iter().map(|x| (x - m).exp()).sum();
                    for d in 0..hd {
                        want[h * hd + d] = (0..seq)
                            .map(|t| (s[t] - m).exp() / z * vf[(t * N_KV + kv) * hd + d])
                            .sum();
                    }
                }
                let k_blob = stream.clone_htod(&kb).unwrap();
                let v_blob = stream.clone_htod(&vb).unwrap();
                let q_dev = stream.clone_htod(&q).unwrap();
                let pos_dev = stream.clone_htod(&[(seq - 1) as i32]).unwrap();
                let got = attn_flash_splitk_q8_gqa_decode_dev_pos(
                    &k_blob,
                    &v_blob,
                    &q_dev.slice(..),
                    &pos_dev,
                    hd,
                    N_KV,
                    nqpk,
                    seq,
                    1,
                    scale,
                    &dev,
                )
                .unwrap();
                let got = stream
                    .clone_dtoh(got.as_cuda_slice::<f32>().unwrap())
                    .unwrap();
                let d = max_diff(&got, &want);
                if d > 1e-3 {
                    failures.push(format!("hd={hd} nq={nqpk} seq={seq} off by {d:.2e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
