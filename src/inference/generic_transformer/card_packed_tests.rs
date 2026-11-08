//! The packed forms of a projection on a card, against the host product of the same weights.
//!
//! Every form of a stored weight must run from the bytes the file holds.
//! The host quantises the rows it multiplies and a card sums in another order, so each product
//! is measured against the exact one: the weight decoded to F32 times the F32 rows.
//! A card passes when it is no farther from the exact product than the host, up to the rounding
//! of F32 (`no_farther` of `card_packed.rs`).
//!
//! A Q4_K or Q6_K weight measures its products and chooses one by the same bar, so the choice
//! is checked here too: by its measurements, by the rule alone, and against each forced route.

use super::card_mixer_tests::{closeness, distances, no_farther};
use super::card_packed::{
    beyond_chance, choose, f32_rounding, pairs_won, route, tied, Choices, Errors, PackedDev, Route,
    Timing,
};
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

/// The product of a packed weight on the card is no farther from the exact product than the host
/// product of the same weight, for 1 row, a few rows and a prompt's worth, in whole and in tiles
/// of a few rows.
#[test]
fn every_packed_form_stays_as_close_to_the_exact_product_as_the_host() {
    let Some(dev) = card() else {
        return;
    };
    let (out, inp) = (96, WIDE);
    let mut worst = (0f64, 0f64);
    for (name, p) in forms(out, inp) {
        let truth = dense_of(p.dense_f32().unwrap(), out, inp);
        for tile in [usize::MAX, 4 * inp * 4] {
            let packed = PackedDev::new(&dev, &p, &Choices::default())
                .unwrap()
                .with_tile_bytes(tile);
            for rows in [1usize, 3, 9] {
                let xs = values(rows * inp, 11 + rows as u64, 1.0);
                let x = Tensor::from_vec(xs.clone(), (rows, inp), &Device::Cpu).unwrap();
                let truth = flat(&truth.apply_here(&x).unwrap());
                let want = flat(&p.apply_here(&x).unwrap());
                let up = upload_f32(&dev, "x", &xs).unwrap();
                let y = packed.apply(&up, rows).unwrap();
                let got = dev.stream().clone_dtoh(&y).unwrap();
                let (host_l2, card_l2) = distances(&truth, &want, &got[..rows * out]);
                worst = (worst.0.max(host_l2), worst.1.max(card_l2));
                assert!(
                    no_farther(host_l2, card_l2, inp),
                    "{name}, {rows} rows, tile {tile}: card {card_l2} from the exact product, host {host_l2}"
                );
            }
        }
    }
    eprintln!(
        "packed products, worst relative L2 from the exact product: host {}, card {}",
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
        let packed = PackedDev::new(&dev, &p, &Choices::default()).unwrap();
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
        m.on_card(0, &Choices::default()).unwrap();
    }
    m
}

/// A window layer over packed weights gives rows no farther from the F32 weights' rows than the
/// host layer's, on a prompt and past the window, and the card holds what the host plan counts
/// for it.
#[test]
fn a_window_layer_over_packed_weights_stays_as_close_to_the_exact_one_as_the_host() {
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
            let (h, c) = distances(&truth, &want, &got);
            host_l2 = host_l2.max(h);
            card_l2 = card_l2.max(c);
        }
        eprintln!("{name}: worst relative L2 from the F32 weights: host {host_l2}, card {card_l2}");
        assert!(
            no_farther(host_l2, card_l2, WIDE),
            "{name}: the card is farther from the F32 weights than the host"
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
            layer.on_card(0, &Choices::default()).unwrap();
        }
        layers.push(layer);
    }
    layers
}

fn dense_tensor(out: usize, inp: usize, seed: u64) -> Tensor {
    Tensor::from_vec(values(out * inp, seed, 0.4), (out, inp), &Device::Cpu).unwrap()
}

/// A band over packed weights gives rows no farther from the F32 weights' rows than the host
/// band's, on a prompt, on decode steps past the window and across groups, and on a continued
/// prompt.
#[test]
fn a_band_over_packed_weights_stays_as_close_to_the_exact_one_as_the_host() {
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
        let mut reference = bands(
            &|o, i, s| dense_of(make(o, i, s).dense_f32().unwrap(), o, i),
            false,
        );
        let (mut host_l2, mut card_l2) = (0f64, 0f64);
        let steps: Vec<(usize, usize)> = std::iter::once((0, 7))
            .chain((7..27).map(|p| (p, 1)))
            .chain(std::iter::once((27, 4)))
            .collect();
        for (step, &(pos, seq)) in steps.iter().enumerate() {
            for layers in [&reference, &host, &card] {
                layers[0].cell.open(pos, seq);
            }
            for il in 0..host.len() {
                let xs = values(seq * WIDE, 500 + (step * 7 + il) as u64, 1.0);
                let x = Tensor::from_vec(xs, (1, seq, WIDE), &Device::Cpu).unwrap();
                let truth = flat(&reference[il].forward(&x, pos).unwrap());
                let want = flat(&host[il].forward(&x, pos).unwrap());
                let got = flat(&card[il].forward(&x, pos).unwrap());
                let (h, c) = distances(&truth, &want, &got);
                host_l2 = host_l2.max(h);
                card_l2 = card_l2.max(c);
            }
        }
        eprintln!(
            "band {name}: worst relative L2 from the F32 weights: host {host_l2}, card {card_l2}"
        );
        assert!(
            no_farther(host_l2, card_l2, WIDE),
            "band {name}: the card is farther from the F32 weights than the host"
        );
    }
}

/// The card's product of a block-quantised weight against the exact one, for the card's quantised
/// matmul (q8_1 rows) and for the form the card holds, next to the host's: worst relative L2 of
/// each. The form the card holds is no farther than the host.
#[test]
fn the_card_blocks_are_as_close_to_the_exact_product_as_the_host() {
    let Some(dev) = card() else {
        return;
    };
    let (out, inp) = (96, 2 * WIDE);
    for (name, dtype, seed) in [
        ("q6_k", GgmlDType::Q6K, 3),
        ("q4_k", GgmlDType::Q4K, 4),
        ("q8_0", GgmlDType::Q8_0, 2),
    ] {
        let p = blocks(out, inp, seed, dtype);
        let (mut old, mut new, mut host_worst) = (0f64, 0f64, 0f64);
        // The weight decoded to F32 against F32 rows: what a card that quantises nothing gives.
        let exact = dense_of(p.dense_f32().unwrap(), out, inp);
        for tile in [usize::MAX, 4 * inp * 4] {
            let packed = PackedDev::new(&dev, &p, &Choices::default())
                .unwrap()
                .with_tile_bytes(tile);
            for rows in [1usize, 3, 9] {
                let xs = values(rows * inp, 11 + rows as u64, 1.0);
                let x = Tensor::from_vec(xs.clone(), (rows, inp), &Device::Cpu).unwrap();
                let truth = flat(&exact.apply_here(&x).unwrap());
                let want = flat(&p.apply_here(&x).unwrap());
                host_worst = host_worst.max(closeness(&truth, &want).1);
                let up = upload_f32(&dev, "x", &xs).unwrap();
                let before = dev
                    .stream()
                    .clone_dtoh(&packed.apply_q8_1(&up, rows).unwrap())
                    .unwrap();
                let after = dev
                    .stream()
                    .clone_dtoh(&packed.apply(&up, rows).unwrap())
                    .unwrap();
                let (_, l0) = distances(&truth, &want, &before[..rows * out]);
                let (host_l2, l1) = distances(&truth, &want, &after[..rows * out]);
                old = old.max(l0);
                new = new.max(l1);
                if dtype != GgmlDType::Q8_0 {
                    assert!(
                        no_farther(host_l2, l1, inp),
                        "{name}, {rows} rows, tile {tile}: card {l1} from the exact product, host {host_l2}"
                    );
                }
            }
        }
        eprintln!(
            "{name}: worst relative L2 from the exact product: host {host_worst}, q8_1 matmul {old}, card blocks {new}"
        );
    }
}

/// The product of `packed` over `xs`, `rows` rows, read back.
fn product_of(
    packed: &PackedDev,
    dev: &Arc<crate::tensor::cuda::CudaDevice>,
    xs: &[f32],
    rows: usize,
) -> Vec<f32> {
    let up = upload_f32(dev, "x", xs).unwrap();
    let y = packed.apply(&up, rows).unwrap();
    let mut got = dev.stream().clone_dtoh(&y).unwrap();
    got.truncate(rows * packed.out());
    got
}

/// The product a block weight measures is admissible by the measured errors, and the fastest
/// admissible one by the measured times, and it is the product `apply` gives.
#[test]
fn the_measured_product_is_admissible_and_is_the_one_applied() {
    let Some(dev) = card() else {
        return;
    };
    route::clear();
    let (out, inp) = (96, 2 * WIDE);
    for (name, dtype, seed) in [("q6_k", GgmlDType::Q6K, 3), ("q4_k", GgmlDType::Q4K, 4)] {
        let p = blocks(out, inp, seed, dtype);
        let packed = PackedDev::new(&dev, &p, &Choices::default()).unwrap();
        for rows in [9usize, 1, 3, 17] {
            let xs = values(rows * inp, 40 + rows as u64, 1.0);
            let got = product_of(&packed, &dev, &xs, rows);
            let (errors, d) = packed.decision(rows).expect("the first product measures");
            let admissible = errors.admissible();
            if admissible.is_empty() {
                let least = errors
                    .routes
                    .iter()
                    .map(|(_, e)| *e)
                    .fold(f64::INFINITY, f64::min);
                let mine = errors.routes.iter().find(|(r, _)| *r == d.route).unwrap().1;
                assert_eq!(
                    mine, least,
                    "{name}: with none admissible the smallest error wins"
                );
            } else {
                assert!(
                    admissible.contains(&d.route),
                    "{name}, {rows} rows: {d:?} is not admissible in {errors:?}"
                );
                // The chosen route is tied with the fastest.
                // No route tied with the fastest is nearer the exact product.
                let error = |r: Route| errors.routes.iter().find(|(x, _)| *x == r).unwrap().1;
                let fastest = d.times.iter().min_by(|a, b| a.median.total_cmp(&b.median));
                let tied = |f: &Timing, t: &Timing| tied(f, t, d.tick);
                if let (Some(f), Some(m)) = (fastest, d.times.iter().find(|t| t.route == d.route)) {
                    assert!(
                        tied(f, m),
                        "{name}, {rows} rows: {d:?} is not tied with {f:?}"
                    );
                    for t in &d.times {
                        assert!(
                            !(tied(f, t) && error(t.route) < error(d.route)),
                            "{name}, {rows} rows: {t:?} is tied and nearer than {d:?}"
                        );
                    }
                }
            }
            route::set(d.route);
            let forced = product_of(&packed, &dev, &xs, rows);
            route::clear();
            assert_eq!(
                got, forced,
                "{name}, {rows} rows: apply gives the product of {:?}",
                d.route
            );
            eprintln!(
                "{name}, {rows} rows: {:?} chosen, errors host {} {:?} over {} rows, times {:?}",
                d.route, errors.host, errors.routes, errors.rows, d.times
            );
        }
    }
}

/// A forced route gives its own product: the host's steps the host's product, the q8_1 route the
/// card's quantised matmul, the F32 route the same product in whole and in tiles.
///
/// The F32 route expands the blocks with the card's kernels, which round a value in half
/// precision, so it is near the exact product without being it.
#[test]
fn each_forced_route_gives_its_product() {
    let Some(dev) = card() else {
        return;
    };
    let (out, inp, rows) = (96, 2 * WIDE, 5);
    for (name, dtype, seed) in [("q6_k", GgmlDType::Q6K, 3), ("q4_k", GgmlDType::Q4K, 4)] {
        let p = blocks(out, inp, seed, dtype);
        let xs = values(rows * inp, 77, 1.0);
        let x = Tensor::from_vec(xs.clone(), (rows, inp), &Device::Cpu).unwrap();
        let host = flat(&p.apply_here(&x).unwrap());
        let exact = flat(
            &dense_of(p.dense_f32().unwrap(), out, inp)
                .apply_here(&x)
                .unwrap(),
        );
        let mut whole: Option<Vec<f32>> = None;
        for tile in [usize::MAX, 4 * inp * 4] {
            let packed = PackedDev::new(&dev, &p, &Choices::default())
                .unwrap()
                .with_tile_bytes(tile);
            route::set(Route::HostSteps);
            let steps = product_of(&packed, &dev, &xs, rows);
            route::set(Route::Q81);
            let q81 = product_of(&packed, &dev, &xs, rows);
            route::set(Route::F32);
            let f32s = product_of(&packed, &dev, &xs, rows);
            route::clear();
            let up = upload_f32(&dev, "x", &xs).unwrap();
            let matmul = dev
                .stream()
                .clone_dtoh(&packed.apply_q8_1(&up, rows).unwrap())
                .unwrap();
            let rounding = f32_rounding(inp);
            let (_, to_host) = closeness(&host, &steps);
            assert!(
                to_host <= rounding,
                "{name}, tile {tile}: host steps {to_host} from the host"
            );
            assert_eq!(
                q81,
                matmul[..rows * out],
                "{name}, tile {tile}: q8_1 is the quantised matmul"
            );
            let (host_l2, to_exact) = distances(&exact, &host, &f32s);
            assert!(
                no_farther(host_l2, to_exact, inp),
                "{name}, tile {tile}: f32 {to_exact} from the exact product, host {host_l2}"
            );
            let first = whole.get_or_insert_with(|| f32s.clone());
            let (_, to_whole) = closeness(first, &f32s);
            assert!(
                to_whole <= rounding,
                "{name}, tile {tile}: tiles {to_whole} from the whole weight"
            );
            assert!(
                packed.decision(rows).is_none(),
                "{name}: a forced route measures nothing"
            );
            eprintln!("{name}, tile {tile}: host steps {to_host} from the host, f32 {to_exact} from the exact product");
        }
    }
}

/// The errors of a unit where the host's steps match the host, the q8_1 route is a little
/// nearer and the F32 route is exact.
fn errors(host: f64, steps: f64, q81: f64, f32s: f64) -> Errors {
    Errors {
        host,
        routes: vec![
            (Route::HostSteps, steps),
            (Route::Q81, q81),
            (Route::F32, f32s),
        ],
        rows: 8,
        terms: 2 * WIDE,
    }
}

/// Timings of one sample each, with no spread.
fn at(times: &[(Route, f64)]) -> Vec<Timing> {
    times.iter().map(|&(r, t)| Timing::of(r, &[t])).collect()
}

/// A product that is slower and not more accurate is never chosen, an inadmissible one never
/// while one is admissible, and with none admissible the smallest error wins.
#[test]
fn a_slower_product_that_is_not_more_accurate_is_never_chosen() {
    let e = errors(0.010, 0.010, 0.009, 0.0);
    let fast_steps = [
        (Route::HostSteps, 10.0),
        (Route::Q81, 50.0),
        (Route::F32, 80.0),
    ];
    assert_eq!(choose(&e, &at(&fast_steps), 0.0), Route::HostSteps);
    let fast_f32 = [
        (Route::HostSteps, 90.0),
        (Route::Q81, 50.0),
        (Route::F32, 20.0),
    ];
    assert_eq!(choose(&e, &at(&fast_f32), 0.0), Route::F32);
    // F32 is as far as the host and far slower.
    let e = errors(0.010, 0.010, 0.009, 0.010);
    let slow_f32 = [
        (Route::HostSteps, 30.0),
        (Route::Q81, 40.0),
        (Route::F32, 1000.0),
    ];
    assert_eq!(choose(&e, &at(&slow_f32), 0.0), Route::HostSteps);
    // The q8_1 route is the fastest and farther than the host.
    let e = errors(0.010, 0.010, 0.020, 0.0);
    let fast_q81 = [
        (Route::HostSteps, 30.0),
        (Route::Q81, 5.0),
        (Route::F32, 60.0),
    ];
    assert_eq!(choose(&e, &at(&fast_q81), 0.0), Route::HostSteps);
    // Within the rounding of an F32 sum of the host is admissible.
    let e = errors(0.010, 0.010 + f32_rounding(2 * WIDE) / 2.0, 0.020, 0.020);
    assert!(e.admissible().contains(&Route::HostSteps));
    // None is admissible: the smallest error, whatever the times.
    let e = errors(0.001, 0.010, 0.030, 0.020);
    assert_eq!(choose(&e, &at(&fast_q81), 0.0), Route::HostSteps);
    assert_eq!(choose(&e, &[], 0.0), Route::HostSteps);
}

/// A pair count is beyond chance when equal routes reach it with a chance below 1 in the pairs.
#[test]
fn a_pair_count_is_beyond_chance_below_one_in_the_pairs() {
    assert!(beyond_chance(1, 1));
    assert!(beyond_chance(2, 2) && !beyond_chance(1, 2));
    assert!(beyond_chance(4, 4) && !beyond_chance(3, 4));
    assert!(beyond_chance(7, 8) && !beyond_chance(6, 8));
    assert!(!beyond_chance(0, 0) && !beyond_chance(0, 4));
    // More pairs need a share of wins that falls toward a half.
    let least = |n: usize| (1..=n).find(|&w| beyond_chance(w, n)).unwrap();
    assert!(least(128) < 96 && least(128) > 64);
}

/// A route faster by its medians but not in the pairs of samples taken back to back is tied with
/// the slower one, and the tie goes to the product nearest the exact one.
#[test]
fn a_route_faster_in_a_slow_moment_of_the_other_is_tied() {
    let e = errors(0.010, 0.010, 0.009, 0.0);
    let steps = Timing::of(Route::HostSteps, &[12.6, 12.6, 12.6, 12.6]);
    // After the first round, the q8_1 route is slower in 2 rounds and faster in 1.
    let q81 = Timing::of(Route::Q81, &[14.1, 15.2, 12.0, 14.1]);
    assert_eq!(pairs_won(&steps, &q81, 0.3), (2, 3));
    assert!(tied(&steps, &q81, 0.3));
    let slow_f32 = Timing::of(Route::F32, &[400.0, 400.0, 400.0, 400.0]);
    let times = [steps.clone(), q81, slow_f32.clone()];
    assert_eq!(choose(&e, &times, 0.3), Route::Q81);
    // Slower in every round, the q8_1 route loses.
    let q81 = Timing::of(Route::Q81, &[14.1, 15.2, 13.0, 14.1]);
    assert!(!tied(&steps, &q81, 0.3));
    assert_eq!(choose(&e, &[steps, q81, slow_f32], 0.3), Route::HostSteps);
    // A slow first run, a scratch buffer's first allocation, is not a pair lost.
    let first_slow = Timing::of(Route::Q81, &[400.0, 86.0, 86.0, 86.0]);
    let f32s = Timing::of(Route::F32, &[295.0, 295.0, 295.0, 295.0]);
    assert!(!tied(&first_slow, &f32s, 0.3));
    let e = errors(0.010, 0.020, 0.009, 0.0);
    assert_eq!(choose(&e, &[first_slow, f32s], 0.3), Route::Q81);
}

/// The choice depends on the measurements alone: not on the order they are listed in, and the
/// errors measured twice on the same rows are the same.
#[test]
fn the_choice_is_deterministic_given_the_same_samples() {
    let e = errors(0.010, 0.010, 0.009, 0.0);
    let times = [
        (Route::HostSteps, 30.0),
        (Route::Q81, 30.0),
        (Route::F32, 30.0),
    ];
    let mut reversed = times;
    reversed.reverse();
    let mut swapped = e.clone();
    swapped.routes.reverse();
    // Equal times fall to the smaller error.
    assert_eq!(choose(&e, &at(&times), 0.0), Route::F32);
    assert_eq!(choose(&e, &at(&reversed), 0.0), Route::F32);
    assert_eq!(choose(&swapped, &at(&times), 0.0), Route::F32);
    // Equal times and errors fall to the order of the routes.
    let e = errors(0.010, 0.010, 0.010, 0.010);
    assert_eq!(choose(&e, &at(&times), 0.0), Route::HostSteps);
    assert_eq!(choose(&e, &at(&reversed), 0.0), Route::HostSteps);

    let Some(dev) = card() else {
        return;
    };
    route::clear();
    let (out, inp, rows) = (96, 2 * WIDE, 9);
    let p = blocks(out, inp, 3, GgmlDType::Q6K);
    let xs = values(rows * inp, 91, 1.0);
    let measured: Vec<(Errors, Route)> = (0..2)
        .map(|_| {
            let packed = PackedDev::new(&dev, &p, &Choices::default()).unwrap();
            product_of(&packed, &dev, &xs, rows);
            let (errors, d) = packed.decision(rows).unwrap();
            eprintln!("measured {:?}, tick {} us", d.times, d.tick);
            (errors, d.route)
        })
        .collect();
    assert_eq!(measured[0], measured[1]);
}

/// Samples of `base` microseconds each moved by `jitter` times a pseudo-random factor in -1..1
/// drawn from `seed`.
fn jittered(base: &[f64], jitter: f64, seed: u64) -> Vec<f64> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    base.iter()
        .map(|&b| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 11) as f64 / (1u64 << 53) as f64;
            b + jitter * (2.0 * u - 1.0)
        })
        .collect()
}

/// The same weights decided twice, with timings perturbed within their noise, give the same
/// route: a tie within the noise falls to the route nearest the exact product, whichever route
/// the samples make the fastest, and a route slower beyond the noise is never chosen for its
/// precision.
#[test]
fn a_decision_made_twice_with_perturbed_timings_gives_the_same_route() {
    // The host's steps as far as the host, q8_1 nearer, F32 nearest.
    let e = errors(0.0165, 0.0165, 0.0075, 0.00018);
    let tick = 0.3;
    // The host's steps and q8_1 within half a microsecond of each other, F32 4 times slower.
    let base = [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let slow = [48.0; 8];
    let mut seen = Vec::new();
    for seed in 0..64u64 {
        let steps = jittered(&base, 0.5, 3 * seed);
        let q81 = jittered(&base, 0.5, 3 * seed + 1);
        let f32s = jittered(&slow, 1.0, 3 * seed + 2);
        let times = [
            Timing::of(Route::HostSteps, &steps),
            Timing::of(Route::Q81, &q81),
            Timing::of(Route::F32, &f32s),
        ];
        let faster = if times[0].median < times[1].median {
            Route::HostSteps
        } else {
            Route::Q81
        };
        seen.push(faster);
        assert_eq!(
            choose(&e, &times, tick),
            Route::Q81,
            "seed {seed}: {times:?}"
        );
    }
    assert!(
        seen.contains(&Route::HostSteps) && seen.contains(&Route::Q81),
        "the perturbations make either route the fastest"
    );
    // A slow route with 1 sample far off, a first allocation, stays slower than the noise.
    let mut outlier = slow.to_vec();
    outlier[0] = 30_000.0;
    let times = [
        Timing::of(Route::HostSteps, &jittered(&base, 1.0, 5)),
        Timing::of(Route::Q81, &jittered(&base, 1.0, 6)),
        Timing::of(Route::F32, &outlier),
    ];
    assert_eq!(choose(&e, &times, tick), Route::Q81, "{times:?}");
    // A route faster beyond the noise keeps its place against a nearer one.
    let times = [
        Timing::of(Route::HostSteps, &jittered(&base, 0.2, 7)),
        Timing::of(Route::Q81, &jittered(&[20.0; 8], 0.2, 8)),
        Timing::of(Route::F32, &jittered(&slow, 0.2, 9)),
    ];
    assert_eq!(choose(&e, &times, tick), Route::HostSteps);
}

/// The weights of one block type and shape share one measurement, and another shape has its own.
#[test]
fn the_weights_of_a_unit_share_one_measurement() {
    let Some(dev) = card() else {
        return;
    };
    route::clear();
    let choices = Choices::default();
    let (out, inp, rows) = (96, 2 * WIDE, 4);
    let a = PackedDev::new(&dev, &blocks(out, inp, 3, GgmlDType::Q6K), &choices).unwrap();
    let b = PackedDev::new(&dev, &blocks(out, inp, 8, GgmlDType::Q6K), &choices).unwrap();
    let wider = PackedDev::new(&dev, &blocks(out, 2 * inp, 9, GgmlDType::Q6K), &choices).unwrap();
    let other = PackedDev::new(&dev, &blocks(out, inp, 4, GgmlDType::Q4K), &choices).unwrap();
    assert!(a.shares_unit(&b));
    assert!(!a.shares_unit(&wider));
    assert!(!a.shares_unit(&other));
    product_of(&a, &dev, &values(rows * inp, 5, 1.0), rows);
    assert!(
        b.decision(rows).is_some(),
        "the unit measured once for both"
    );
    assert!(
        b.decision(4 * rows).is_none(),
        "another size of rows is timed on its own"
    );
    assert!(wider.decision(rows).is_none());
}
