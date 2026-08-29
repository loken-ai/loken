use crate::tensor::{DType, Device, Tensor};

/// The fused prefill attention against the chain it would replace.
///
/// Wiring it changed what models answer, so the question is not whether it is fast but
/// by how much and where it differs. This builds both sides from the same inputs and
/// reports the error, so a divergence has a size rather than an anecdote.
#[test]
#[ignore = "needs a CUDA device"]
fn flash_prefill_matches_the_chain_it_replaces() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let (b, nh, nkv, seq, hd) = (1usize, 8usize, 8usize, 64usize, 64usize);
    // A fixed pseudo-random fill: the same inputs on both sides, and the same on
    // every run, so a reported error is the kernel's and not the draw's.
    let mk = |n: usize, seed: u64| {
        let mut s = seed;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        v
    };
    let shape = (b, nh, seq, hd);
    let q = Tensor::from_vec(mk(b * nh * seq * hd, 1), shape, &dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let k = Tensor::from_vec(mk(b * nkv * seq * hd, 2), (b, nkv, seq, hd), &dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let v = Tensor::from_vec(mk(b * nkv * seq * hd, 3), (b, nkv, seq, hd), &dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let scale = 1.0f32 / (hd as f32).sqrt();

    let fused = match super::flash_prefill_f16(&q, &k, &v, nh, nkv, scale, 0).unwrap() {
        Some(y) => y,
        None => return, // shape not covered: nothing to compare
    };

    // The chain: scaled scores, causal mask, softmax, then V.
    let scores = q
        .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
        .unwrap();
    let scores = scores
        .affine(scale, 0.0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let mut m = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                m[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(m, (seq, seq), &dev).unwrap();
    let scores = scores.broadcast_add(&mask).unwrap();
    let probs = crate::tensor::ops::softmax_last_dim(&scores)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let expect = probs.matmul(&v).unwrap();

    let a = fused
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let e = expect
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(a.len(), e.len());
    let num: f32 = a.iter().zip(&e).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = e.iter().map(|y| y * y).sum::<f32>().max(1e-12);
    let rel_rms = (num / den).sqrt();
    let maxabs = a
        .iter()
        .zip(&e)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("flash_prefill vs chain: rel_rms={rel_rms:.6} maxabs={maxabs:.6}");
    // Accuracy is not the bar. Greedy decoding compares logits by rank, and a drift
    // this size, carried through every layer, eventually swaps two neighbours and the
    // continuation changes from there - measured on qwen3:8b, "curious young prince"
    // against "wise and kind queen", each reproducible across restarts. Substituting
    // the kernel on a greedy path needs bit-identity; anything looser is a deliberate
    // numerical change and has to be revalidated per model.
    assert!(
        rel_rms < 1e-3,
        "the fused prefill does not even approximate the chain: {rel_rms}"
    );
    if rel_rms > 0.0 {
        println!(
            "NOT bit-identical: substituting this on a greedy path changes what \
                  models answer, however small the error"
        );
    }
}
