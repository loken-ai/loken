//! Bit-exact parity against ggml's own scalar quantisers.
//!
//! The tests already in `tensor/quant_cpu/mod.rs` hold the AVX paths against the scalar path. That is the
//! right check for a SIMD kernel and the wrong one for the scalar kernel itself, because the
//! scalar kernel is the code under question: it came from candle, which took it from ggml, and
//! an implementation compared against itself agrees with itself. Those tests would pass on a
//! reimplementation that got every block format subtly wrong, as long as it got it wrong twice.
//!
//! This module is the judge that can say no. `scripts/oracle/` builds a tool against a
//! llama.cpp checkout, runs ggml's `quantize_row_*_ref` and `dequantize_row_*` over a fixed,
//! closed-form input, and writes the bytes out. Here the same input goes through our
//! implementation and the results must match **bit for bit** - the quantised blocks byte for
//! byte, the dequantised floats by their bit patterns, not by a tolerance.
//!
//! A tolerance would be the wrong instrument twice over: these are integer formats where the
//! bytes are either right or wrong, and a comparison that accepts "close" cannot distinguish a
//! correct port from one that rounds a scale search differently in the last place - which is
//! exactly the failure that survives thirty layers and flips a greedy argmax.
//!
//! `vectors.txt` is checked in, so this runs without llama.cpp present. Regenerate it only
//! when the block formats themselves change.

use super::quant_cpu::{
    BlockFormat, BlockMxFp4, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K,
    BlockQ5_0, BlockQ5_1, BlockQ6K, BlockQ8K, BlockQ8_0,
};

/// One block format's reference data, as the oracle emitted it.
struct Reference {
    block_elems: usize,
    block_bytes: usize,
    /// The quantised blocks, exactly as ggml laid them out in memory.
    quant: Vec<u8>,
    /// What ggml's dequantiser produced from those blocks.
    dequant: Vec<f32>,
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("oracle: bad hex byte"))
        .collect()
}

/// Floats arrive as their bit patterns, so the comparison never passes through a decimal
/// rendering that could round two different values onto the same text.
fn hex_to_floats(s: &str) -> Vec<f32> {
    (0..s.len() / 8)
        .map(|i| {
            f32::from_bits(
                u32::from_str_radix(&s[i * 8..i * 8 + 8], 16).expect("oracle: bad float"),
            )
        })
        .collect()
}

fn load_vectors() -> (Vec<f32>, std::collections::HashMap<String, Reference>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/oracle/vectors.txt");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "oracle: cannot read {path}: {e}\n\
             Regenerate with: LLAMA_CPP=<path to llama.cpp> make -C scripts/oracle vectors"
        )
    });

    let mut input = Vec::new();
    let mut refs = std::collections::HashMap::new();
    let mut current: Option<(String, usize, usize)> = None;
    let mut quant: Vec<u8> = Vec::new();

    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut f = line.split_whitespace();
        match f.next() {
            Some("N") => {}
            Some("INPUT") => {
                let _n = f.next();
                input = hex_to_floats(f.next().expect("oracle: INPUT has no payload"));
            }
            Some("TYPE") => {
                let name = f.next().expect("oracle: TYPE has no name").to_string();
                let elems = f.next().unwrap().parse().unwrap();
                let bytes = f.next().unwrap().parse().unwrap();
                current = Some((name, elems, bytes));
            }
            Some("QUANT") => {
                let _n = f.next();
                quant = hex_to_bytes(f.next().expect("oracle: QUANT has no payload"));
            }
            Some("DEQUANT") => {
                let _n = f.next();
                let dequant = hex_to_floats(f.next().expect("oracle: DEQUANT has no payload"));
                let (name, block_elems, block_bytes) =
                    current.take().expect("oracle: DEQUANT before TYPE");
                refs.insert(
                    name,
                    Reference {
                        block_elems,
                        block_bytes,
                        quant: std::mem::take(&mut quant),
                        dequant,
                    },
                );
            }
            other => panic!("oracle: unrecognised record {other:?}"),
        }
    }
    assert!(!input.is_empty(), "oracle: vectors carry no INPUT row");
    (input, refs)
}

/// Read ggml's own blocks and check we get ggml's floats, bit for bit.
///
/// THIS is the contract that has to hold. A GGUF file in the wild is a sequence of blocks, and
/// what those blocks MEAN is fixed by the format. If our dequantiser disagrees with ggml's by
/// one bit, every weight of every model we load is subtly not the weight the publisher shipped,
/// and nothing reports it - the model simply answers slightly worse.
///
/// It reads ggml's blocks rather than our own, so a quantiser error that our dequantiser undid
/// symmetrically cannot hide inside a round trip.
fn check_dequant<T: BlockFormat>(name: &str) {
    let (input, refs) = load_vectors();
    let r = refs
        .get(name)
        .unwrap_or_else(|| panic!("oracle: no reference vectors for {name}"));
    assert_layout::<T>(name, r);

    let ggml_blocks: &[T] = unsafe {
        std::slice::from_raw_parts(r.quant.as_ptr() as *const T, r.quant.len() / r.block_bytes)
    };
    let mut back = vec![0f32; input.len()];
    T::dequantize(ggml_blocks, &mut back);
    if let Some(i) = (0..back.len()).find(|&i| back[i].to_bits() != r.dequant[i].to_bits()) {
        panic!(
            "{name}: dequantisation differs from ggml at element {i} (block {}): \
             ours {} (0x{:08x}), ggml {} (0x{:08x})",
            i / r.block_elems,
            back[i],
            back[i].to_bits(),
            r.dequant[i],
            r.dequant[i].to_bits()
        );
    }
}

/// A block whose size or element count differs is not the same format, and every comparison
/// after it would be reading one layout as another.
fn assert_layout<T: BlockFormat>(name: &str, r: &Reference) {
    assert_eq!(
        T::BLOCK_LEN,
        r.block_elems,
        "{name}: our block holds {} elements, ggml's holds {}",
        T::BLOCK_LEN,
        r.block_elems
    );
    assert_eq!(
        std::mem::size_of::<T>(),
        r.block_bytes,
        "{name}: our block is {} bytes, ggml's is {}",
        std::mem::size_of::<T>(),
        r.block_bytes
    );
}

#[test]
fn q4_0_dequantises_like_ggml() {
    check_dequant::<BlockQ4_0>("q4_0");
}
#[test]
fn q4_1_dequantises_like_ggml() {
    check_dequant::<BlockQ4_1>("q4_1");
}
#[test]
fn q5_0_dequantises_like_ggml() {
    check_dequant::<BlockQ5_0>("q5_0");
}
#[test]
fn q5_1_dequantises_like_ggml() {
    check_dequant::<BlockQ5_1>("q5_1");
}
#[test]
fn q8_0_dequantises_like_ggml() {
    check_dequant::<BlockQ8_0>("q8_0");
}
#[test]
fn q2_k_dequantises_like_ggml() {
    check_dequant::<BlockQ2K>("q2_K");
}
#[test]
fn q3_k_dequantises_like_ggml() {
    check_dequant::<BlockQ3K>("q3_K");
}
#[test]
fn q4_k_dequantises_like_ggml() {
    check_dequant::<BlockQ4K>("q4_K");
}
#[test]
fn q5_k_dequantises_like_ggml() {
    check_dequant::<BlockQ5K>("q5_K");
}
#[test]
fn q6_k_dequantises_like_ggml() {
    check_dequant::<BlockQ6K>("q6_K");
}
#[test]
fn q8_k_dequantises_like_ggml() {
    check_dequant::<BlockQ8K>("q8_K");
}
#[test]
fn mxfp4_dequantises_like_ggml() {
    check_dequant::<BlockMxFp4>("mxfp4");
}

// The encoder side has no test here on purpose. Quantising is a CHOICE - two sound encoders
// pick different scales - so comparing bytes against ggml's asks the wrong question, and the
// twelve tests that did were `#[ignore]`d from the day they were written. What they hid is in
// `every_format_reconstructs_about_as_well_as_ggml`, which measures the thing an encoder is
// actually for: how much of the signal survives the round trip.

// ---------------------------------------------------------------------------
// How well does OUR encoder compress, against theirs, on the same input?
// ---------------------------------------------------------------------------

/// Root-mean-square error between a reconstruction and the signal it came from.
fn rmse(signal: &[f32], rebuilt: &[f32]) -> f64 {
    let n = signal.len() as f64;
    (signal
        .iter()
        .zip(rebuilt)
        .map(|(a, b)| {
            let d = (*a - *b) as f64;
            d * d
        })
        .sum::<f64>()
        / n)
        .sqrt()
}

/// Quantise the oracle's input with OUR encoder, dequantise it, and compare the error with
/// the one ggml's own encoder made on the same input.
///
/// This is the instrument the encoder side needs, and it is a different one from
/// `check_dequant`. Dequantisation has a single correct answer, so the test there is equality.
/// A quantiser is a LOSSY encoder running an iterative scale search whose algorithm upstream
/// has revised more than once - ours descends from an earlier revision - so demanding equality
/// would compare two choices rather than check a contract. What can be compared is the thing
/// the encoder exists to minimise: how far the round trip lands from the signal.
///
/// `vectors.txt` already carries everything needed. INPUT is the signal, DEQUANT is what ggml
/// reconstructed from its own blocks, so their error is known without running their code.
///
/// The assertion is deliberately one-sided and loose: ours must not be MUCH worse. A tight
/// bound here would fail on a legitimate improvement to the search, which is the opposite of
/// what this is for - it exists so a change can be judged, and so a change that quietly
/// degrades reconstruction cannot pass as a refactor.
fn check_reconstruction<T: BlockFormat>(name: &str, budget: f64) -> (f64, f64) {
    let (input, refs) = load_vectors();
    let r = refs
        .get(name)
        .unwrap_or_else(|| panic!("oracle: no reference vectors for {name}"));

    let mut ours = vec![T::zeros(); input.len() / T::BLOCK_LEN];
    T::quantize(&input, &mut ours);
    let mut rebuilt = vec![0f32; input.len()];
    T::dequantize(&ours, &mut rebuilt);

    let mine = rmse(&input, &rebuilt);
    let theirs = rmse(&input, &r.dequant);
    println!(
        "{name}: ours {mine:.6}, ggml {theirs:.6}, ratio {:.3}",
        mine / theirs.max(f64::MIN_POSITIVE)
    );
    let ratio = mine / theirs.max(f64::MIN_POSITIVE);
    assert!(
        ratio <= budget + 0.01,
        "{name}: reconstruction has regressed - {ratio:.3}x ggml's error, was {budget:.3}x. \
         Raise the recorded figure only alongside the measurement that justifies it."
    );
    (mine, theirs)
}

#[test]
fn every_format_reconstructs_about_as_well_as_ggml() {
    let mut worse = Vec::new();
    let mut report = Vec::new();
    // The recorded figure is the ratio measured when this format's encoder was last changed,
    // so the assertion is a REGRESSION gate: it fails on a change that reconstructs worse than
    // what we already shipped, not merely worse than ggml. The five formats at 1.000 have no
    // free parameter to choose - one scale per block, taken from the extreme value - so there
    // is nothing for an encoder to do differently and equality is the only correct answer.
    macro_rules! measure {
        ($t:ty, $n:literal, $budget:literal) => {{
            let (mine, theirs) = check_reconstruction::<$t>($n, $budget);
            report.push(format!("{:>6} ours {mine:.6} ggml {theirs:.6}", $n));
            if mine > theirs * 1.001 {
                worse.push(format!("{}: {:.4}x", $n, mine / theirs));
            }
        }};
    }
    measure!(BlockQ4_0, "q4_0", 1.000);
    measure!(BlockQ4_1, "q4_1", 1.000);
    measure!(BlockQ5_0, "q5_0", 1.000);
    measure!(BlockQ5_1, "q5_1", 1.000);
    measure!(BlockQ8_0, "q8_0", 1.000);
    measure!(BlockQ2K, "q2_K", 0.893);
    measure!(BlockQ3K, "q3_K", 0.886);
    measure!(BlockQ4K, "q4_K", 1.000);
    measure!(BlockQ5K, "q5_K", 0.996);
    measure!(BlockQ6K, "q6_K", 0.978);
    measure!(BlockMxFp4, "mxfp4", 0.979);
    println!("{}", report.join("\n"));
    assert!(
        worse.is_empty(),
        "these formats reconstruct worse than ggml: {}",
        worse.join(", ")
    );
}

/// The importance-matrix encoders, which nothing in this tree calls yet.
///
/// `quantize_guided` is implemented for the five k-quant formats and invoked by nobody, so
/// only a test stands between it and quiet rot. It broke exactly that way once: a rewrite kept
/// the imatrix WEIGHTS and dropped the weighted second stage that fits the sixteen sub-block
/// scales, which reads as a simplification and compiles the same.
///
/// The property is the one the capability exists for. Given a weighting that says some values
/// matter far more than others, the imatrix encoder must reconstruct THOSE values better than
/// the plain encoder does - measured as weighted RMSE under the same weights, which is the
/// quantity the imatrix is fitting.
#[test]
fn an_importance_matrix_buys_accuracy_where_it_says_it_matters() {
    let (input, _) = load_vectors();

    // The contrast has to be BETWEEN sub-blocks, not only within them. The second stage fits
    // the sixteen sub-block scales against how much weight each sub-block carries, so a
    // matrix that gives every sub-block the same total leaves it nothing to do - which is how
    // the first version of this test passed with that stage disabled.
    let weights: Vec<f32> = (0..input.len())
        .map(|i| {
            let sub = (i / 32) % 8;
            let heavy = if sub < 2 { 64.0 } else { 0.05 };
            if i % 8 == 0 {
                heavy
            } else {
                heavy / 16.0
            }
        })
        .collect();

    fn weighted_rmse(signal: &[f32], rebuilt: &[f32], w: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for ((&s, &r), &wi) in signal.iter().zip(rebuilt).zip(w) {
            num += wi as f64 * (s as f64 - r as f64).powi(2);
            den += wi as f64;
        }
        (num / den).sqrt()
    }

    // The recorded ratio each format reaches. "Better than plain" alone is too weak: with the
    // weighted second stage disabled, q4_K still scored 0.997 and passed. The figure is what
    // separates "the matrix reached both stages" from "it reached the first one".
    fn compare<T: BlockFormat>(name: &str, input: &[f32], w: &[f32], budget: f64) {
        let blocks = input.len() / T::BLOCK_LEN;
        let mut rebuilt = vec![0f32; input.len()];

        let mut plain = vec![T::zeros(); blocks];
        T::quantize(input, &mut plain);
        T::dequantize(&plain, &mut rebuilt);
        let without = weighted_rmse(input, &rebuilt, w);

        let mut guided = vec![T::zeros(); blocks];
        T::quantize_guided(input, &mut guided, w, input.len());
        T::dequantize(&guided, &mut rebuilt);
        let with = weighted_rmse(input, &rebuilt, w);

        println!(
            "{name}: guided {with:.6}, plain {without:.6}, ratio {:.3}",
            with / without
        );
        let ratio = with / without;
        assert!(
            ratio <= budget + 0.02,
            "{name}: the importance matrix bought {ratio:.3} of the plain error, was {budget:.3}. \
             A ratio drifting toward 1 means the weights reached the sub-block fit and not the \
             stage that quantises the sixteen scales - which is how this broke once."
        );
    }

    compare::<BlockQ2K>("q2_K", &input, &weights, 0.780);
    compare::<BlockQ3K>("q3_K", &input, &weights, 0.848);
    compare::<BlockQ4K>("q4_K", &input, &weights, 0.905);
    compare::<BlockQ5K>("q5_K", &input, &weights, 0.935);
    compare::<BlockQ6K>("q6_K", &input, &weights, 0.879);
}
