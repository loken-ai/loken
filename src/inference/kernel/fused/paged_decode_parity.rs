//! The paged decode attention a card runs, against the attention computed on the host from the
//! same half-precision cache, over head sizes, head groupings, batch widths and ragged lengths.

use super::*;
use crate::tensor::{Device, Tensor};

/// Deterministic values in `[-amp, amp]`.
fn fill(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn halves(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
}

struct Case {
    n_head: usize,
    n_kv: usize,
    hd: usize,
    lens: Vec<usize>,
}

/// The attention of each sequence's query over its own tokens, read through its block table.
#[allow(clippy::too_many_arguments)]
fn reference(
    c: &Case,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tables: &[Vec<usize>],
    block_size: usize,
    scale: f32,
) -> Vec<f32> {
    let (nh, hd) = (c.n_head, c.hd);
    let feat = c.n_kv * hd;
    let groups = nh / c.n_kv;
    let mut out = vec![0f32; c.lens.len() * nh * hd];
    for (bi, &len) in c.lens.iter().enumerate() {
        for h in 0..nh {
            let qh = &q[(bi * nh + h) * hd..][..hd];
            let off = (h / groups) * hd;
            let slot = |j: usize| tables[bi][j / block_size] * block_size + j % block_size;
            let scores: Vec<f32> = (0..len)
                .map(|j| {
                    let kj = &k[slot(j) * feat + off..][..hd];
                    qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let top = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let w: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let sum: f32 = w.iter().sum();
            let o = &mut out[(bi * nh + h) * hd..][..hd];
            for (j, wj) in w.iter().enumerate() {
                let vj = &v[slot(j) * feat + off..][..hd];
                for d in 0..hd {
                    o[d] += wj / sum * vj[d];
                }
            }
        }
    }
    out
}

#[test]
fn the_paged_decode_attention_matches_the_host() {
    let Ok(dev) = Device::new_cuda(0) else {
        eprintln!("skipping: no card");
        return;
    };
    let block_size = 16;
    let ragged = |b: usize| -> Vec<usize> { (0..b).map(|i| 1 + (i * 37) % 300).collect() };
    let mut cases = Vec::new();
    for hd in [64, 128, 256] {
        for (n_head, n_kv) in [(8, 8), (16, 8), (12, 4), (16, 4), (32, 4)] {
            for lens in [vec![700], vec![5, 1, 33], ragged(64)] {
                cases.push(Case {
                    n_head,
                    n_kv,
                    hd,
                    lens,
                });
            }
        }
    }
    for c in &cases {
        let (b, nh, hd) = (c.lens.len(), c.n_head, c.hd);
        let feat = c.n_kv * hd;
        let blocks_of = |len: usize| len.div_ceil(block_size);
        let total_blocks: usize = c.lens.iter().map(|&l| blocks_of(l)).sum();
        // Each sequence's blocks are spread over the store in reverse order.
        let mut next = total_blocks;
        let tables: Vec<Vec<usize>> = c
            .lens
            .iter()
            .map(|&l| {
                (0..blocks_of(l))
                    .map(|_| {
                        next -= 1;
                        next
                    })
                    .collect()
            })
            .collect();
        let max_blocks = tables.iter().map(Vec::len).max().unwrap();
        let slots = total_blocks * block_size;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = halves(&fill(b * nh * hd, 1, 1.0));
        let k = halves(&fill(slots * feat, 2, 1.0));
        let v = halves(&fill(slots * feat, 3, 1.0));
        let want = reference(c, &q, &k, &v, &tables, block_size, scale);

        let card = |x: &[f32], dims: &[usize]| {
            Tensor::from_vec(x.to_vec(), dims, &dev)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
        };
        let mut flat = vec![0i32; b * max_blocks];
        for (bi, t) in tables.iter().enumerate() {
            for (j, &blk) in t.iter().enumerate() {
                flat[bi * max_blocks + j] = blk as i32;
            }
        }
        let bt = Tensor::from_vec(flat, &[b, max_blocks][..], &dev).unwrap();
        let lens: Vec<i32> = c.lens.iter().map(|&l| l as i32).collect();
        let sl = Tensor::from_vec(lens, &[b][..], &dev).unwrap();
        let got = paged_flash_decode(
            &card(&q, &[b, nh, hd]),
            &card(&k, &[slots, feat]),
            &card(&v, &[slots, feat]),
            &bt,
            &sl,
            scale,
            b,
            nh,
            c.n_kv,
            hd,
            block_size,
            max_blocks,
        )
        .unwrap()
        .expect("the card runs the paged decode")
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 4e-3,
            "hd {hd}, {nh} heads over {} kv heads, lengths {:?}: worst difference {worst}",
            c.n_kv,
            &c.lens[..c.lens.len().min(4)]
        );
    }
}
