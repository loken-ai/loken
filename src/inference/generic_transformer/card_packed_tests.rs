//! The packed forms of a projection on a card, against the host product of the same weights.
//!
//! Every form of a stored weight must run from the bytes the file holds.
//! A card and the host sum in another order, so the products agree within the card versus host
//! policy of `card_mixer_tests.rs`.

use super::card_mixer_tests::{closeness, MAX_REL_L2, MIN_COSINE};
use super::card_packed::PackedDev;
use super::compressed_band::{
    BandCell, BandDims, BandIndexerWeights, CompressedBand, CompressorWeights,
};
use super::latent_mqa::{frequencies, LatentMqa, LatentMqaDims, LatentMqaWeights};
use crate::tensor::blockscaled::bf16::Bf16Weight;
use crate::tensor::blockscaled::fp4::Fp4Weight;
use crate::tensor::blockscaled::fp8::Fp8Weight;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::cuda::upload_f32;
use crate::tensor::mapped::MappedBytes;
use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{Device, Tensor};
use std::sync::Arc;

/// Input and output width of the weights: whole blocks of every form, K-quants included.
const WIDE: usize = 256;

/// Values of order 1, deterministic.
fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * amp
        })
        .collect()
}

fn mapped(bytes: &[u8]) -> MappedBytes {
    let mut m = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
    m[..bytes.len()].copy_from_slice(bytes);
    MappedBytes::new(Arc::new(m.make_read_only().unwrap()), 0, bytes.len()).unwrap()
}

fn dense(out: usize, inp: usize, seed: u64) -> Projection {
    let w = Tensor::from_vec(values(out * inp, seed, 0.4), (out, inp), &Device::Cpu).unwrap();
    Projection::Dense(w)
}

fn dense_of(values: Vec<f32>, out: usize, inp: usize) -> Projection {
    Projection::Dense(Tensor::from_vec(values, (out, inp), &Device::Cpu).unwrap())
}

fn blocks(out: usize, inp: usize, seed: u64, dtype: GgmlDType) -> Projection {
    let w = Tensor::from_vec(values(out * inp, seed, 0.4), (out, inp), &Device::Cpu).unwrap();
    Projection::Quant(Arc::new(QTensor::quantize(&w, dtype).unwrap()))
}

fn fp8(out: usize, inp: usize, seed: u64) -> Projection {
    let block = 32;
    // 0x7F and 0xFF are e4m3's NaN: a checkpoint holds none.
    let bytes: Vec<u8> = (0..out * inp)
        .map(|i| match ((i as u64 * 29 + 7 + seed * 53) % 256) as u8 {
            b if b & 0x7F == 0x7F => b - 1,
            b => b,
        })
        .collect();
    let tiles = out.div_ceil(block) * inp.div_ceil(block);
    let scales: Vec<u8> = (0..tiles)
        .map(|i| (111 + ((i as u64 + seed) % 4)) as u8)
        .collect();
    Projection::Fp8(Arc::new(
        Fp8Weight::new(mapped(&bytes), mapped(&scales), out, inp, block).unwrap(),
    ))
}

fn fp4(out: usize, inp: usize, seed: u64) -> Projection {
    let block = 32;
    let nibbles: Vec<u8> = (0..out * inp / 2)
        .map(|i| ((i as u64 * 37 + 11 + seed * 101) % 251) as u8)
        .collect();
    let scales: Vec<u8> = (0..out * inp / block)
        .map(|i| (119 + ((i as u64 + seed) % 4)) as u8)
        .collect();
    Projection::Fp4(Arc::new(
        Fp4Weight::new(mapped(&nibbles), mapped(&scales), out, inp).unwrap(),
    ))
}

fn bf16(out: usize, inp: usize, seed: u64) -> Projection {
    let bytes: Vec<u8> = values(out * inp, seed, 0.4)
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    Projection::Bf16(Arc::new(Bf16Weight::new(mapped(&bytes), out, inp).unwrap()))
}

/// Every form, as `(name, projection)`, `[out, inp]`.
fn forms(out: usize, inp: usize) -> Vec<(&'static str, Projection)> {
    vec![
        ("f32", dense(out, inp, 1)),
        ("q8_0", blocks(out, inp, 2, GgmlDType::Q8_0)),
        ("q6_k", blocks(out, inp, 3, GgmlDType::Q6K)),
        ("q4_k", blocks(out, inp, 4, GgmlDType::Q4K)),
        ("fp8", fp8(out, inp, 5)),
        ("fp4", fp4(out, inp, 6)),
        ("bf16", bf16(out, inp, 7)),
    ]
}

fn card() -> Option<Arc<crate::tensor::cuda::CudaDevice>> {
    match Device::new_cuda(0) {
        Ok(Device::Cuda(dev)) => Some(dev),
        _ => None,
    }
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// The product of a packed weight on the card agrees with the host product of the same weight,
/// for 1 row, a few rows and a prompt's worth, in whole and in tiles of a few rows.
#[test]
fn every_packed_form_gives_the_host_product_within_the_card_policy() {
    let Some(dev) = card() else {
        return;
    };
    let (out, inp) = (96, WIDE);
    let mut worst = (1f64, 0f64);
    for (name, p) in forms(out, inp) {
        for tile in [usize::MAX, 4 * inp * 4] {
            let packed = PackedDev::new(&dev, &p).unwrap().with_tile_bytes(tile);
            for rows in [1usize, 3, 9] {
                let xs = values(rows * inp, 11 + rows as u64, 1.0);
                let x = Tensor::from_vec(xs.clone(), (rows, inp), &Device::Cpu).unwrap();
                let want = flat(&p.apply_here(&x).unwrap());
                let up = upload_f32(&dev, "x", &xs).unwrap();
                let y = packed.apply(&up, rows).unwrap();
                let got = dev.stream().clone_dtoh(&y).unwrap();
                let (cos, l2) = closeness(&want, &got[..rows * out]);
                worst = (worst.0.min(cos), worst.1.max(l2));
                assert!(
                    cos > MIN_COSINE && l2 < MAX_REL_L2,
                    "{name}, {rows} rows, tile {tile}: cosine {cos}, relative L2 {l2}"
                );
            }
        }
    }
    eprintln!(
        "packed products: worst cosine {}, worst relative L2 {}",
        worst.0, worst.1
    );
}

/// A packed form holds the bytes the file stores and a dense form its F32 values.
#[test]
fn a_packed_form_holds_the_stored_bytes() {
    let Some(dev) = card() else {
        return;
    };
    let (out, inp) = (96, WIDE);
    for (name, p) in forms(out, inp) {
        let packed = PackedDev::new(&dev, &p).unwrap();
        let dense_bytes = out * inp * std::mem::size_of::<f32>();
        match name {
            "f32" => assert_eq!(packed.bytes(), dense_bytes, "{name}"),
            _ => assert_eq!(packed.bytes(), p.bytes(), "{name}"),
        }
        eprintln!(
            "{name}: {} bytes on the card, {} as dense F32, {:.2}x",
            packed.bytes(),
            dense_bytes,
            dense_bytes as f64 / packed.bytes() as f64
        );
    }
}

fn dims() -> LatentMqaDims {
    LatentMqaDims {
        heads: 8,
        head_dim: 64,
        rope_dim: 16,
        o_groups: 2,
        o_lora_rank: 32,
        window: 4,
        eps: 1e-6,
    }
}

fn norm(n: usize, seed: u64) -> Tensor {
    let scale: Vec<f32> = values(n, seed, 0.5).iter().map(|v| 1.0 + v).collect();
    Tensor::from_vec(scale, n, &Device::Cpu).unwrap()
}

/// A window layer over weights `make` gives, on the host or on card 0.
fn layer(make: &dyn Fn(usize, usize, u64) -> Projection, on_card: bool) -> LatentMqa {
    let d = dims();
    let group = d.heads * d.head_dim / d.o_groups;
    let w = LatentMqaWeights {
        wq_a: make(WIDE, WIDE, 1),
        q_norm: norm(WIDE, 2),
        wq_b: make(d.heads * d.head_dim, WIDE, 3),
        wkv: make(d.head_dim, WIDE, 4),
        kv_norm: norm(d.head_dim, 5),
        sink: values(d.heads, 6, 1.0),
        wo_a: make(d.out_width(), group, 7),
    };
    let inv_freq = frequencies(d.rope_dim, 0, 10_000.0, 1.0);
    let mut m = LatentMqa::new(d, w, inv_freq).unwrap();
    if on_card {
        m.on_card(0).unwrap();
    }
    m
}

/// A window layer over packed weights gives the host layer's rows on a prompt and past the
/// window, and the card holds what the host plan counts for it.
#[test]
fn a_window_layer_over_packed_weights_matches_the_host() {
    if card().is_none() {
        return;
    }
    // The grouped output is cut by rows, which fp4 and bf16 views do not allow.
    let makers: Vec<(&str, Box<dyn Fn(usize, usize, u64) -> Projection>)> = vec![
        ("f32", Box::new(dense)),
        ("q8_0", Box::new(|o, i, s| blocks(o, i, s, GgmlDType::Q8_0))),
        ("q6_k", Box::new(|o, i, s| blocks(o, i, s, GgmlDType::Q6K))),
        ("q4_k", Box::new(|o, i, s| blocks(o, i, s, GgmlDType::Q4K))),
        ("fp8", Box::new(fp8)),
    ];
    for (name, make) in makers {
        let mut host = layer(&*make, false);
        let mut card = layer(&*make, true);
        // The same weights decoded to F32 on the host: the answer neither side approximates.
        let mut exact = layer(
            &|o, i, s| {
                let w = make(o, i, s).dense_f32().unwrap();
                dense_of(w, o, i)
            },
            false,
        );
        let stored = host.resident_bytes();
        let held = card.card_bytes();
        eprintln!("{name}: host plan {stored} bytes, card holds {held} bytes");
        assert_eq!(
            card.resident_bytes(),
            held,
            "{name}: the plan counts the card"
        );
        // The norms and the sink are the only bytes beyond the projections.
        let d = dims();
        let vectors = (WIDE + d.head_dim + d.heads) * std::mem::size_of::<f32>();
        assert_eq!(held, stored + vectors, "{name}");

        let (mut cos_min, mut l2_max) = (1f64, 0f64);
        let (mut host_l2, mut card_l2) = (0f64, 0f64);
        let steps: Vec<(usize, usize)> = std::iter::once((0, 5))
            .chain((5..17).map(|p| (p, 1)))
            .chain(std::iter::once((17, 3)))
            .collect();
        for (step, &(pos, seq)) in steps.iter().enumerate() {
            let xs = values(seq * WIDE, 100 + step as u64, 1.0);
            let x = Tensor::from_vec(xs, (1, seq, WIDE), &Device::Cpu).unwrap();
            let want = flat(&host.forward(&x, pos).unwrap());
            let got = flat(&card.forward(&x, pos).unwrap());
            let truth = flat(&exact.forward(&x, pos).unwrap());
            host_l2 = host_l2.max(closeness(&truth, &want).1);
            card_l2 = card_l2.max(closeness(&truth, &got).1);
            let (cos, l2) = closeness(&want, &got);
            cos_min = cos_min.min(cos);
            l2_max = l2_max.max(l2);
            assert!(
                cos > MIN_COSINE && l2 < MAX_REL_L2,
                "{name}, rows {pos}..{}: cosine {cos}, relative L2 {l2}",
                pos + seq
            );
        }
        eprintln!(
            "{name}: worst cosine {cos_min}, worst relative L2 {l2_max}; against the F32 weights: host {host_l2}, card {card_l2}"
        );
    }
}

/// The 3 layers of a band over weights `make` gives: a kv-source that scores and keeps candidate
/// blocks, a scorer inside those blocks, and a layer that reads what both left.
fn bands(make: &dyn Fn(usize, usize, u64) -> Projection, on_card: bool) -> Vec<CompressedBand> {
    let cell = Arc::new(BandCell::default());
    let roles = [
        (true, true, true, false),
        (false, true, false, true),
        (false, false, false, false),
    ];
    let mut layers = Vec::new();
    for (n, (kv, index, source, uses)) in roles.into_iter().enumerate() {
        let n = n as u64;
        let band = BandDims {
            ratio: 2,
            kv_source: kv,
            index_source: index,
            candidate_source: source,
            uses_candidates: uses,
            index_topk: 3,
            index_heads: 2,
            index_head_dim: 32,
            candidate_blocks: 1,
            candidate_block_size: 2,
        };
        let d = dims();
        let hd = d.head_dim;
        let compressor = kv.then(|| CompressorWeights {
            wkv: dense_tensor(hd, WIDE, 20 + n),
            wgate: Some(dense_tensor(hd, WIDE, 30 + n)),
            norm: norm(hd, 40 + n),
        });
        let indexer = index.then(|| BandIndexerWeights {
            wq_b: make(band.index_heads * band.index_head_dim, WIDE, 50 + n),
            weights_proj: dense_tensor(band.index_heads, WIDE, 60 + n),
            wk: kv.then(|| dense_tensor(band.index_head_dim, hd, 70 + n)),
            k_norm: kv.then(|| norm(band.index_head_dim, 80 + n)),
        });
        let mut layer =
            CompressedBand::new(layer(make, false), band, compressor, indexer, cell.clone())
                .unwrap();
        if on_card {
            layer.on_card(0).unwrap();
        }
        layers.push(layer);
    }
    layers
}

fn dense_tensor(out: usize, inp: usize, seed: u64) -> Tensor {
    Tensor::from_vec(values(out * inp, seed, 0.4), (out, inp), &Device::Cpu).unwrap()
}

/// A band over packed weights gives the host band's rows on a prompt, on decode steps past the
/// window and across groups, and on a continued prompt.
#[test]
fn a_band_over_packed_weights_matches_the_host() {
    if card().is_none() {
        return;
    }
    let makers: Vec<(&str, Box<dyn Fn(usize, usize, u64) -> Projection>)> = vec![
        ("f32", Box::new(dense)),
        ("q8_0", Box::new(|o, i, s| blocks(o, i, s, GgmlDType::Q8_0))),
        ("fp8", Box::new(fp8)),
    ];
    for (name, make) in makers {
        let mut host = bands(&*make, false);
        let mut card = bands(&*make, true);
        let (mut cos_min, mut l2_max) = (1f64, 0f64);
        let steps: Vec<(usize, usize)> = std::iter::once((0, 7))
            .chain((7..27).map(|p| (p, 1)))
            .chain(std::iter::once((27, 4)))
            .collect();
        for (step, &(pos, seq)) in steps.iter().enumerate() {
            for layers in [&host, &card] {
                layers[0].cell.open(pos, seq);
            }
            for il in 0..host.len() {
                let xs = values(seq * WIDE, 500 + (step * 7 + il) as u64, 1.0);
                let x = Tensor::from_vec(xs, (1, seq, WIDE), &Device::Cpu).unwrap();
                let want = flat(&host[il].forward(&x, pos).unwrap());
                let got = flat(&card[il].forward(&x, pos).unwrap());
                let (cos, l2) = closeness(&want, &got);
                cos_min = cos_min.min(cos);
                l2_max = l2_max.max(l2);
                assert!(
                    cos > MIN_COSINE && l2 < MAX_REL_L2,
                    "{name}, layer {il}, rows {pos}..{}: cosine {cos}, relative L2 {l2}",
                    pos + seq
                );
            }
        }
        eprintln!("band {name}: worst cosine {cos_min}, worst relative L2 {l2_max}");
    }
}
