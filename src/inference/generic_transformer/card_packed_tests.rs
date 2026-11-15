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
    beyond_chance, choose, f32_rounding, fastest, pairs_won, route, tied, Choices, Decision,
    Errors, PackedDev, Route, Timing,
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

/// Asserts that decision `d` follows the rule of the measured choice over `errors`.
///
/// The chosen route is admissible, or the nearest when none is.
/// It is tied with the fastest admissible route.
/// No route tied with the fastest is nearer the exact product.
fn follows_the_rule(what: &str, errors: &Errors, d: &Decision) {
    let admissible = errors.admissible();
    let error = |r: Route| errors.routes.iter().find(|(x, _)| *x == r).unwrap().1;
    if admissible.is_empty() {
        let least = errors
            .routes
            .iter()
            .map(|(_, e)| *e)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(
            error(d.route),
            least,
            "{what}: with none admissible the smallest error wins"
        );
        return;
    }
    assert!(
        admissible.contains(&d.route),
        "{what}: {d:?} is not admissible in {errors:?}"
    );
    let fastest = fastest(errors, &d.times);
    let tied = |f: &Timing, t: &Timing| tied(f, t, d.tick);
    if let (Some(f), Some(m)) = (fastest, d.times.iter().find(|t| t.route == d.route)) {
        assert!(tied(f, m), "{what}: {d:?} is not tied with {f:?}");
        for t in &d.times {
            assert!(
                !(tied(f, t) && error(t.route) < error(d.route)),
                "{what}: {t:?} is tied and nearer than {d:?}"
            );
        }
    }
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
            follows_the_rule(&format!("{name}, {rows} rows"), &errors, &d);
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

/// The timings a card recorded for a Q4_K weight of 1 row, where 2 routes share the least median.
///
/// The q8_1 route is nearer than the host's steps, so it leads the ties.
/// F32 is tied with the host's steps but slower than the q8_1 route beyond chance.
#[test]
fn two_routes_of_the_least_median_lead_by_their_error() {
    let e = Errors {
        host: 0.003799352285924926,
        routes: vec![
            (Route::HostSteps, 0.00379935426820409),
            (Route::Q81, 0.0037737262411300853),
            (Route::F32, 0.0005538836322946031),
        ],
        rows: 9,
        terms: 2 * WIDE,
    };
    let tick = 0.31999999191612005;
    let times = vec![
        Timing::of(
            Route::HostSteps,
            &[
                6.496000103652477,
                5.824000108987093,
                13.088000006973743,
                6.047999951988459,
            ],
        ),
        Timing::of(
            Route::Q81,
            &[
                6.047999951988459,
                6.111999973654747,
                6.111999973654747,
                4.70399996265769,
            ],
        ),
        Timing::of(
            Route::F32,
            &[
                8.895999751985073,
                7.903999648988247,
                7.679999805986881,
                7.903999648988247,
            ],
        ),
    ];
    assert_eq!(times[0].median, times[1].median);
    assert!(tied(&times[0], &times[2], tick) && !tied(&times[1], &times[2], tick));
    let route = choose(&e, &times, tick);
    assert_eq!(route, Route::Q81);
    follows_the_rule("q4_k, 1 rows", &e, &Decision { route, times, tick });
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

/// The bits of `v`.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `n` values of `y` read back.
fn read(
    dev: &crate::tensor::cuda::CudaDevice,
    y: &crate::tensor::gpu_runtime::driver::CudaSlice<f32>,
    n: usize,
) -> Vec<f32> {
    let mut v = dev.stream().clone_dtoh(y).unwrap();
    v.truncate(n);
    v
}

/// The shapes of a large latent attention layer's projections: the query factor, the queries,
/// the key row, a group of the output projection and the layer's output projection.
const LARGE: [(usize, usize); 5] = [
    (1280, 5120),
    (32768, 1280),
    (512, 5120),
    (1024, 4096),
    (5120, 8192),
];

/// A product of 1 row on the q8_1 route has the bits of the quantised matmul's forward over a
/// copy of the row, for every block type the route serves, on small shapes and on large ones.
#[test]
fn a_one_row_product_keeps_the_bits_of_the_matmul_forward() {
    let Some(dev) = card() else {
        return;
    };
    let small = [(96usize, 512usize), (40, 1024), (8, 2 * WIDE)];
    let mut cases: Vec<(GgmlDType, (usize, usize))> = Vec::new();
    for dtype in [GgmlDType::Q6K, GgmlDType::Q4K, GgmlDType::Q8_0] {
        cases.extend(small.iter().map(|&s| (dtype, s)));
    }
    cases.extend(LARGE.iter().map(|&s| (GgmlDType::Q6K, s)));
    for (k, &(dtype, (out, inp))) in cases.iter().enumerate() {
        let p = blocks(out, inp, 40 + k as u64, dtype);
        let packed = PackedDev::new(&dev, &p, &Choices::default()).unwrap();
        let xs = values(inp, 90 + k as u64, 1.0);
        let up = upload_f32(&dev, "x", &xs).unwrap();
        route::set(Route::Q81);
        let fused = packed.apply(&up, 1);
        let chain = packed.apply_through_forward(&up, 1);
        route::clear();
        let (fused, chain) = (fused.unwrap(), chain.unwrap());
        assert_eq!(
            bits(&read(&dev, &fused, out)),
            bits(&read(&dev, &chain, out)),
            "{dtype:?} [{out}, {inp}]"
        );
    }
}

/// A window layer of `d` over Q6_K weights on card 0, with a hidden width and a query rank.
fn q6_layer(d: LatentMqaDims, hidden: usize, q_rank: usize) -> LatentMqa {
    layer_of(GgmlDType::Q6K, d, hidden, q_rank)
}

/// A window layer of `d` over `dtype` weights on card 0, with a hidden width and a query rank.
fn layer_of(dtype: GgmlDType, d: LatentMqaDims, hidden: usize, q_rank: usize) -> LatentMqa {
    let make = |o, i, s| blocks(o, i, s, dtype);
    let group = d.heads * d.head_dim / d.o_groups;
    let w = LatentMqaWeights {
        wq_a: make(q_rank, hidden, 1),
        q_norm: norm(q_rank, 2),
        wq_b: make(d.heads * d.head_dim, q_rank, 3),
        wkv: make(d.head_dim, hidden, 4),
        kv_norm: norm(d.head_dim, 5),
        sink: values(d.heads, 6, 1.0),
        wo_a: make(d.out_width(), group, 7),
    };
    let inv_freq = frequencies(d.rope_dim, 0, 10_000.0, 1.0);
    let mut m = LatentMqa::new(d, w, inv_freq).unwrap();
    m.on_card(0, &Choices::default()).unwrap();
    m
}

/// The fused steps of a decoded row on the q8_1 route have the bits of the steps they replace:
/// the query factor and the key row from one quantisation and one launch, and the grouped output
/// projection from one quantisation of the heads' outputs.
#[test]
fn the_fused_steps_of_a_decoded_row_keep_the_bits_of_the_chain() {
    if card().is_none() {
        return;
    }
    let large = LatentMqaDims {
        heads: 64,
        head_dim: 512,
        rope_dim: 64,
        o_groups: 8,
        o_lora_rank: 1024,
        window: 128,
        eps: 1e-6,
    };
    let small = LatentMqaDims {
        heads: 8,
        head_dim: 128,
        rope_dim: 16,
        o_groups: 2,
        o_lora_rank: 64,
        window: 4,
        eps: 1e-6,
    };
    // A hidden width past the small-K mat-vec, which the pair's launch does not serve.
    for (d, hidden, q_rank) in [(small, 8 * WIDE, WIDE), (large, 5120, 1280)] {
        let m = q6_layer(d, hidden, q_rank);
        let card = m.card.clone().unwrap();
        let dev = card.dev().clone();
        let x = upload_f32(&dev, "x", &values(hidden, 7, 1.0)).unwrap();
        let width = d.heads * d.head_dim;
        let o = upload_f32(&dev, "o", &values(width, 8, 1.0)).unwrap();
        let p = width / d.o_groups;
        route::set(Route::Q81);
        let pair = card.factor_and_key_row(&x, 1);
        let chain = card.factor_and_key_by_forward(&x, 1);
        let mut wide = crate::tensor::cuda::alloc_f32(&dev, "wide", d.out_width()).unwrap();
        let grouped = card.grouped_row(&o, (p, d.o_lora_rank), &mut wide);
        let by_columns = card.grouped_by_forward(&o, 1, (width, p, d.o_lora_rank));
        route::clear();
        let (qa, kv) = pair.unwrap().expect("the pair takes the one-row mat-vec");
        let (qa_chain, kv_chain) = chain.unwrap();
        assert_eq!(
            bits(&read(&dev, &qa, q_rank)),
            bits(&read(&dev, &qa_chain, q_rank)),
            "query factor, hidden {hidden}"
        );
        assert_eq!(
            bits(&read(&dev, &kv, d.head_dim)),
            bits(&read(&dev, &kv_chain, d.head_dim)),
            "key row, hidden {hidden}"
        );
        assert!(grouped.unwrap(), "the groups take the one-row mat-vec");
        assert_eq!(
            bits(&read(&dev, &wide, d.out_width())),
            bits(&read(&dev, &by_columns.unwrap(), d.out_width())),
            "grouped output, hidden {hidden}"
        );
    }
}

/// The groups of the output projection in one launch have the bits of a launch a group, and of
/// the quantised matmul's forward over each group's columns.
#[test]
fn the_grouped_output_projection_in_one_launch_keeps_the_bits_of_a_launch_a_group() {
    if card().is_none() {
        return;
    }
    let dims = |heads, head_dim, o_groups, o_lora_rank| LatentMqaDims {
        heads,
        head_dim,
        rope_dim: 64,
        o_groups,
        o_lora_rank,
        window: 4,
        eps: 1e-6,
    };
    // Row counts that no small-K mat-vec serves, so every group takes the one-row kernel.
    let mut cases = Vec::new();
    for dtype in [GgmlDType::Q6K, GgmlDType::Q4K, GgmlDType::Q8_0] {
        cases.push((dtype, dims(16, 512, 2, 98), 8 * WIDE, WIDE));
        cases.push((dtype, dims(24, 256, 3, 98), 8 * WIDE, WIDE));
    }
    cases.push((GgmlDType::Q6K, dims(64, 512, 8, 1024), 5120, 1280));
    for (dtype, d, hidden, q_rank) in cases {
        let m = layer_of(dtype, d, hidden, q_rank);
        let card = m.card.clone().unwrap();
        let dev = card.dev().clone();
        let width = d.heads * d.head_dim;
        let o = upload_f32(&dev, "o", &values(width, 8, 1.0)).unwrap();
        let p = width / d.o_groups;
        let shape = format!("{dtype:?} {} groups [{}, {p}]", d.o_groups, d.o_lora_rank);
        let wide = || crate::tensor::cuda::alloc_f32(&dev, "wide", d.out_width()).unwrap();
        let (mut one, mut each) = (wide(), wide());
        route::set(Route::Q81);
        let in_one = card.groups_in_one_launch();
        let grouped = card.grouped_row(&o, (p, d.o_lora_rank), &mut one);
        let grouped_each = card.grouped_row_each(&o, (p, d.o_lora_rank), &mut each);
        let by_columns = card.grouped_by_forward(&o, 1, (width, p, d.o_lora_rank));
        route::clear();
        assert!(in_one, "{shape}: the groups take one launch");
        assert!(grouped.unwrap() && grouped_each.unwrap(), "{shape}");
        let one = bits(&read(&dev, &one, d.out_width()));
        assert_eq!(one, bits(&read(&dev, &each, d.out_width())), "{shape}");
        assert_eq!(
            one,
            bits(&read(&dev, &by_columns.unwrap(), d.out_width())),
            "{shape}"
        );
    }
}

/// The unit of the groups of an output projection measures its product of 1 row, the groups'
/// launch included, at the first product.
#[test]
fn the_groups_of_an_output_projection_measure_their_launch() {
    if card().is_none() {
        return;
    }
    route::clear();
    let d = LatentMqaDims {
        heads: 16,
        head_dim: 512,
        rope_dim: 64,
        o_groups: 2,
        o_lora_rank: 98,
        window: 4,
        eps: 1e-6,
    };
    let m = q6_layer(d, 8 * WIDE, WIDE);
    let card = m.card.clone().unwrap();
    let dev = card.dev().clone();
    let width = d.heads * d.head_dim;
    let o = upload_f32(&dev, "o", &values(width, 8, 1.0)).unwrap();
    let p = width / d.o_groups;
    assert!(card.group_decision().is_none());
    let mut wide = crate::tensor::cuda::alloc_f32(&dev, "wide", d.out_width()).unwrap();
    card.grouped_by_columns(&o, 1, (width, p, d.o_lora_rank), &mut wide)
        .unwrap();
    let decision = card.group_decision().expect("the first product measures");
    assert!(decision.times.iter().all(|t| t.median > 0.0));
}
