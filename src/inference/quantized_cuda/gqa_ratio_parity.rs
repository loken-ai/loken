//! The Q8 KV decode kernels, judged per GQA ratio against attention computed on the host.
//!
//! Every ratio the load gate admits must give the same answer through each decode path: the
//! eager score + output chain, the graph's device-position score + fused softmax-output chain,
//! and the split-K flash decode. A ratio served by the wrong instance reads or writes a
//! neighbouring head and still produces a plausible distribution, so only a reference
//! catches it.
use super::*;
use crate::tensor::kernel_ffi::CudaDevice;

const HD: usize = 64;
const N_KV: usize = 2;
const SEQ: usize = 40;
const MAX_SEQ: usize = 64;

/// Q8_0 blocks laid out `[max_seq, n_kv, head_dim]`, with their dequantised values.
fn q8_blob(salt: usize) -> (Vec<u8>, Vec<f32>) {
    let hdb = HD / 32;
    let nblocks = MAX_SEQ * N_KV * hdb;
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

/// softmax(scale * q.k) @ v per query head, over the first `SEQ` positions.
fn reference(q: &[f32], k: &[f32], v: &[f32], nqpk: usize, scale: f32) -> Vec<f32> {
    let n_q = N_KV * nqpk;
    let mut out = vec![0f32; n_q * HD];
    for h in 0..n_q {
        let kv = h / nqpk;
        let at = |t: usize, d: usize| (t * N_KV + kv) * HD + d;
        let s: Vec<f32> = (0..SEQ)
            .map(|t| (0..HD).map(|d| q[h * HD + d] * k[at(t, d)]).sum::<f32>() * scale)
            .collect();
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = e.iter().sum();
        for d in 0..HD {
            out[h * HD + d] = (0..SEQ).map(|t| e[t] / z * v[at(t, d)]).sum();
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
fn every_admitted_gqa_ratio_decodes_like_the_reference() {
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no CUDA device; the Q8 GQA decode kernels NOT covered by this run");
        return;
    };
    let stream = dev.native().stream();
    let (kb, kf) = q8_blob(1);
    let (vb, vf) = q8_blob(4);
    let k_blob = stream.clone_htod(&kb).unwrap();
    let v_blob = stream.clone_htod(&vb).unwrap();
    let pos_dev = stream.clone_htod(&[(SEQ - 1) as i32]).unwrap();
    let scale = 1.0 / (HD as f32).sqrt();
    let host = |s: &CudaStorage| {
        stream
            .clone_dtoh(s.as_cuda_slice::<f32>().unwrap())
            .unwrap()
    };

    let mut failures = Vec::new();
    for nqpk in [1usize, 2, 4, 5, 7, 8] {
        let n_q = N_KV * nqpk;
        let q: Vec<f32> = (0..n_q * HD)
            .map(|i| ((i * 13 % 97) as f32) * 0.02 - 0.9)
            .collect();
        let want = reference(&q, &kf, &vf, nqpk, scale);
        let q_dev = stream.clone_htod(&q).unwrap();
        let qv = q_dev.slice(..);

        // Graph chain: raw scores at the padded stride, then fused softmax + V.
        let scores =
            attn_score_q8_0_f32_dev_pos(&k_blob, &qv, &pos_dev, HD, MAX_SEQ, N_KV, nqpk, 0, &dev)
                .unwrap();
        let sv = scores.as_cuda_slice::<f32>().unwrap().slice(..);
        let chain = attn_softmax_output_q8_0_f32_dev_pos(
            &v_blob, &sv, &pos_dev, HD, MAX_SEQ, N_KV, nqpk, scale, 0, &dev,
        )
        .unwrap();
        let d_chain = max_diff(&host(&chain), &want);

        // Eager output: probabilities from the reference scores, V read at the live length.
        let mut probs = vec![0f32; n_q * SEQ];
        for h in 0..n_q {
            let kv = h / nqpk;
            let s: Vec<f32> = (0..SEQ)
                .map(|t| {
                    (0..HD)
                        .map(|d| q[h * HD + d] * kf[(t * N_KV + kv) * HD + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = s.iter().map(|x| (x - m).exp()).sum();
            for t in 0..SEQ {
                probs[h * SEQ + t] = (s[t] - m).exp() / z;
            }
        }
        let p_dev = stream.clone_htod(&probs).unwrap();
        let eager =
            attn_output_q8_0_f32_gqa(&v_blob, &p_dev.slice(..), HD, SEQ, N_KV, nqpk, &dev).unwrap();
        let d_eager = max_diff(&host(&eager), &want);

        let flash = attn_flash_splitk_q8_gqa_decode_dev_pos(
            &k_blob, &v_blob, &qv, &pos_dev, HD, N_KV, nqpk, SEQ, scale, &dev,
        )
        .unwrap();
        let d_flash = max_diff(&host(&flash), &want);

        eprintln!("nq={nqpk}: chain {d_chain:.2e}  eager {d_eager:.2e}  flash {d_flash:.2e}");
        for (path, d) in [("chain", d_chain), ("eager", d_eager), ("flash", d_flash)] {
            if d > 1e-3 {
                failures.push(format!("nq={nqpk} {path} off by {d:.2e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
