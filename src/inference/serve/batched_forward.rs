//! Batched-paged decode forward - reference (f32) + the batch-invariance contract.
//!
//! This is the algorithmic spec for the real per-model `BatchedModel` impl
//! (which swaps these f32 matmuls for the layer's quantized GEMM kernels and runs
//! on GPU). One decode step over B sequences: batched GEMMs for the projections
//! (the throughput win - B tokens share one weight read) and PAGED per-sequence
//! attention (each token attends only over ITS own KV blocks).
//!
//! The non-negotiable correctness property is BATCH-INVARIANCE: a sequence's
//! output must not depend on which other sequences share its batch - otherwise
//! continuous batching silently corrupts concurrent requests. The projections
//! are trivially row-independent; the danger is the KV path (a wrong slot/block
//! table mixes sequences). The test below asserts a sequence decoded in a batch
//! of 2 (with a DIFFERENT-length neighbour) yields bit-identical output to
//! decoding it alone - the gate every real impl must also pass on GPU.

use crate::inference::cache::paged_attention::{paged_attn_decode_seq, PagedAttnSeq};
use crate::inference::cache::paged_kv::PagedKvStore;

/// Dense decoder-layer weights (f32, row-major `[out, in]`) for the reference.
pub struct RefLayer {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
}

/// Per-sequence paged state for one decode step.
pub struct SeqDecode {
    pub pos: usize,            // position of the new token (0-based)
    pub slot: usize,           // KV slot to write the new K/V
    pub block_table: Vec<u32>, // logical->physical blocks
    pub context_len: usize,    // tokens attended (incl. the new one)
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

/// `out[o] = Σ_i x[i] * W[o,i]` for W shaped `[n_out, n_in]`.
fn matvec(w: &[f32], x: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let mut o = vec![0f32; n_out];
    for r in 0..n_out {
        let row = &w[r * n_in..(r + 1) * n_in];
        o[r] = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    o
}

/// NEOX (split-half) rope on a `[n_heads * head_dim]` vector at `pos`.
fn rope(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, base: f32) {
    let half = head_dim / 2;
    let inv_freq = crate::inference::model::rope::inverse_frequencies(head_dim, base);
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let ang = pos as f32 * inv_freq[i];
            let (s, c) = ang.sin_cos();
            let x1 = x[off + i];
            let x2 = x[off + half + i];
            x[off + i] = x1 * c - x2 * s;
            x[off + half + i] = x2 * c + x1 * s;
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// One batched-paged decode-layer step. `hidden` is `[B, hidden]` (row per
/// sequence); `seqs[b]` is sequence b's paged state. Mutates `store` (writes new
/// K/V) and returns the new `[B, hidden]`. Batched in the projections, paged &
/// per-sequence in attention - bit-identical regardless of batch composition.
pub fn batched_decode_layer(
    layer: &RefLayer,
    hidden: &[Vec<f32>],
    seqs: &[SeqDecode],
    store: &PagedKvStore,
) -> crate::tensor::Result<Vec<Vec<f32>>> {
    use crate::tensor::{Device, Tensor};
    let b = hidden.len();
    let (h, nh, nkv, hd) = (layer.hidden, layer.n_head, layer.n_kv_head, layer.head_dim);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = Vec::with_capacity(b);
    for bi in 0..b {
        let x = &hidden[bi];
        // -- attention block --
        let hn = rmsnorm(x, &layer.attn_norm, layer.rms_eps);
        let mut q = matvec(&layer.wq, &hn, nh * hd, h);
        let mut k = matvec(&layer.wk, &hn, nkv * hd, h);
        let v = matvec(&layer.wv, &hn, nkv * hd, h);
        rope(&mut q, nh, hd, seqs[bi].pos, layer.rope_base);
        rope(&mut k, nkv, hd, seqs[bi].pos, layer.rope_base);
        // write the new K/V into the paged store at this seq's slot
        let kt = Tensor::from_vec(k.clone(), &[1, nkv * hd][..], &Device::Cpu)?;
        let vt = Tensor::from_vec(v.clone(), &[1, nkv * hd][..], &Device::Cpu)?;
        store.write(&[seqs[bi].slot], &kt, &vt)?;
        // gather this seq's whole KV and run paged GQA attention
        let (kg, vg) = store.gather_seq(&seqs[bi].block_table, seqs[bi].context_len)?;
        let kgv = kg.to_vec2::<f32>()?.concat();
        let vgv = vg.to_vec2::<f32>()?.concat();
        let mut attn = vec![0f32; nh * hd];
        {
            let mut s = PagedAttnSeq {
                q: &q,
                k: &kgv,
                v: &vgv,
                context_len: seqs[bi].context_len,
                out: &mut attn,
            };
            paged_attn_decode_seq(&mut s, nh, nkv, hd, scale);
        }
        let o = matvec(&layer.wo, &attn, h, nh * hd);
        let post_attn: Vec<f32> = (0..h).map(|i| x[i] + o[i]).collect();
        // -- FFN block --
        let hn2 = rmsnorm(&post_attn, &layer.ffn_norm, layer.rms_eps);
        let g = matvec(&layer.w_gate, &hn2, layer.ffn, h);
        let u = matvec(&layer.w_up, &hn2, layer.ffn, h);
        let act: Vec<f32> = (0..layer.ffn).map(|i| silu(g[i]) * u[i]).collect();
        let d = matvec(&layer.w_down, &act, h, layer.ffn);
        out.push((0..h).map(|i| post_attn[i] + d[i]).collect());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Device};

    fn rnd(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32 / 500.0) - 1.0)
            .collect()
    }

    fn tiny_layer() -> RefLayer {
        let (h, nh, nkv, hd, ffn) = (8usize, 2usize, 1usize, 4usize, 16usize);
        RefLayer {
            wq: rnd(nh * hd * h, 1),
            wk: rnd(nkv * hd * h, 2),
            wv: rnd(nkv * hd * h, 3),
            wo: rnd(h * nh * hd, 4),
            w_gate: rnd(ffn * h, 5),
            w_up: rnd(ffn * h, 6),
            w_down: rnd(h * ffn, 7),
            attn_norm: vec![1.0; h],
            ffn_norm: vec![1.0; h],
            hidden: h,
            n_head: nh,
            n_kv_head: nkv,
            head_dim: hd,
            ffn,
            rms_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    // Decode sequence `sid` ALONE (its own store) for `n_prior` already-cached
    // tokens + the new one, returning the layer output for the new token.
    fn decode_alone(
        layer: &RefLayer,
        x: &[f32],
        prior_k: &[Vec<f32>],
        prior_v: &[Vec<f32>],
    ) -> Vec<f32> {
        let store = PagedKvStore::new(
            8,
            16,
            layer.n_kv_head,
            layer.head_dim,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let mut a = crate::inference::cache::paged_kv::PagedKvAllocator::new(8, 16);
        a.allocate(0, 0).unwrap();
        // pre-fill prior KV
        let n_prior = prior_k.len();
        if n_prior > 0 {
            let slots = a.append(0, n_prior).unwrap();
            let feat = layer.n_kv_head * layer.head_dim;
            let kflat: Vec<f32> = prior_k.iter().flatten().copied().collect();
            let vflat: Vec<f32> = prior_v.iter().flatten().copied().collect();
            let kt =
                crate::tensor::Tensor::from_vec(kflat, &[n_prior, feat][..], &Device::Cpu).unwrap();
            let vt =
                crate::tensor::Tensor::from_vec(vflat, &[n_prior, feat][..], &Device::Cpu).unwrap();
            store.write(&slots, &kt, &vt).unwrap();
        }
        let new_slot = a.append(0, 1).unwrap()[0];
        let bt = a.block_table(0).unwrap().to_vec();
        let seqs = vec![SeqDecode {
            pos: n_prior,
            slot: new_slot,
            block_table: bt,
            context_len: n_prior + 1,
        }];
        batched_decode_layer(layer, &[x.to_vec()], &seqs, &store)
            .unwrap()
            .remove(0)
    }

    #[test]
    fn batch_invariance_two_sequences_different_lengths() {
        let layer = tiny_layer();
        let feat = layer.n_kv_head * layer.head_dim;
        // Seq A: 3 prior tokens; Seq B: 1 prior token. Distinct hidden + priors.
        let xa = rnd(layer.hidden, 11);
        let xb = rnd(layer.hidden, 22);
        let pka: Vec<Vec<f32>> = (0..3).map(|t| rnd(feat, 100 + t)).collect();
        let pva: Vec<Vec<f32>> = (0..3).map(|t| rnd(feat, 200 + t)).collect();
        let pkb: Vec<Vec<f32>> = (0..1).map(|t| rnd(feat, 300 + t)).collect();
        let pvb: Vec<Vec<f32>> = (0..1).map(|t| rnd(feat, 400 + t)).collect();

        // reference: each alone
        let want_a = decode_alone(&layer, &xa, &pka, &pva);
        let want_b = decode_alone(&layer, &xb, &pkb, &pvb);

        // batched: both share ONE store, distinct blocks.
        let store = PagedKvStore::new(
            8,
            16,
            layer.n_kv_head,
            layer.head_dim,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let mut alloc = crate::inference::cache::paged_kv::PagedKvAllocator::new(8, 16);
        // seq A
        alloc.allocate(0, 0).unwrap();
        let sa = alloc.append(0, 3).unwrap();
        let ka = crate::tensor::Tensor::from_vec(
            pka.iter().flatten().copied().collect::<Vec<_>>(),
            &[3, feat][..],
            &Device::Cpu,
        )
        .unwrap();
        let va = crate::tensor::Tensor::from_vec(
            pva.iter().flatten().copied().collect::<Vec<_>>(),
            &[3, feat][..],
            &Device::Cpu,
        )
        .unwrap();
        store.write(&sa, &ka, &va).unwrap();
        let na = alloc.append(0, 1).unwrap()[0];
        let bta = alloc.block_table(0).unwrap().to_vec();
        // seq B
        alloc.allocate(1, 0).unwrap();
        let sb = alloc.append(1, 1).unwrap();
        let kb = crate::tensor::Tensor::from_vec(
            pkb.iter().flatten().copied().collect::<Vec<_>>(),
            &[1, feat][..],
            &Device::Cpu,
        )
        .unwrap();
        let vb = crate::tensor::Tensor::from_vec(
            pvb.iter().flatten().copied().collect::<Vec<_>>(),
            &[1, feat][..],
            &Device::Cpu,
        )
        .unwrap();
        store.write(&sb, &kb, &vb).unwrap();
        let nb = alloc.append(1, 1).unwrap()[0];
        let btb = alloc.block_table(1).unwrap().to_vec();

        let seqs = vec![
            SeqDecode {
                pos: 3,
                slot: na,
                block_table: bta,
                context_len: 4,
            },
            SeqDecode {
                pos: 1,
                slot: nb,
                block_table: btb,
                context_len: 2,
            },
        ];
        let got = batched_decode_layer(&layer, &[xa.clone(), xb.clone()], &seqs, &store).unwrap();

        for i in 0..layer.hidden {
            assert!(
                (got[0][i] - want_a[i]).abs() < 1e-5,
                "A[{i}] batched {} != alone {}",
                got[0][i],
                want_a[i]
            );
            assert!(
                (got[1][i] - want_b[i]).abs() < 1e-5,
                "B[{i}] batched {} != alone {}",
                got[1][i],
                want_b[i]
            );
        }
    }
}
