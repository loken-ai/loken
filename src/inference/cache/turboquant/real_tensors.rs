//! The real-tensor run: load a small model on CPU, push a prompt through it, and measure
//! the candidate storages against the K and V it actually produced.
//!
//! Synthetic Gaussians would answer a question nobody asked. The rotation is justified by
//! a claim about the *structure* of attention keys - that a few channels are
//! systematically large and set the scale for every group they touch - and that claim can
//! only be checked against keys a model wrote.
//!
//! Ignored by default because it needs a checkpoint on disk, which is the house convention
//! for tests that load weights. Run it with:
//!
//! ```text
//! cargo test --lib --release turboquant_real -- --ignored --nocapture
//! ```

use super::hadamard::HadamardRotation;
use super::measure::{
    evaluate_layer, format_table, rotation_diagnostics, LayerReport, MEASUREMENT_ROTATION_SEED,
};
use crate::config::Config;
use crate::inference::engine::llm_engine::{build_tokenizer_from_gguf, KvQuant};
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::quantized::gguf_file;
use crate::tensor::{DType, Device, Tensor};
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::Cursor;

/// The blob an ollama tag names, or `None` when the manifest is there and the weights are
/// not - a manifest without its layer is an empty measurement, never a small one.
fn ollama_blob(tag: &str) -> Option<std::path::PathBuf> {
    let dir = Config::load_test().get_ollama_models_dir();
    let (name, ver) = tag.split_once(':').unwrap_or((tag, "latest"));
    let text = std::fs::read_to_string(
        dir.join("manifests/registry.ollama.ai/library")
            .join(name)
            .join(ver),
    )
    .ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let digest = v["layers"]
        .as_array()?
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")?["digest"]
        .as_str()?
        .replace(':', "-");
    let blob = dir.join("blobs").join(digest);
    blob.exists().then_some(blob)
}

/// One layer's K and V, pulled out of the cache the forward filled, as `[n_kv_heads,
/// tokens, head_dim]` f32 with the token count truncated to a multiple of 32.
///
/// The truncation is applied once, here, so that every column downstream measures the same
/// slice - the Q4_0 K layout needs whole 32-token windows and the others do not, and a
/// harness that truncated only for that column would be comparing different data.
struct CapturedLayer {
    layer: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    n_kv_heads: usize,
    tokens: usize,
    head_dim: usize,
    dtype: DType,
}

/// Pull `[1, n_kv_heads, seq, head_dim]` out of a cache tensor and flatten it.
fn flatten_kv(t: &Tensor, tokens: usize) -> Option<(Vec<f32>, usize, usize)> {
    let dims = t.dims();
    if dims.len() != 4 || dims[0] != 1 {
        return None;
    }
    let (n_kv_heads, head_dim) = (dims[1], dims[3]);
    let narrowed = t.narrow(2, 0, tokens).ok()?;
    let flat = narrowed
        .to_dtype(DType::F32)
        .ok()?
        .contiguous()
        .ok()?
        .flatten_all()
        .ok()?
        .to_vec1::<f32>()
        .ok()?;
    Some((flat, n_kv_heads, head_dim))
}

#[test]
#[ignore]
fn turboquant_real_cache_tensors_decide_the_scheme() {
    // A small dense model with a 128-wide head, which is the common case and the one the
    // rotation has a full power-of-two transform for.
    const TAG: &str = "qwen3:0.6b";
    // Long enough for several whole 32-token windows and varied enough that the keys are
    // not the keys of one repeated phrase. A prompt that repeats itself would hand the
    // measurement a cache whose rows are near-duplicates, and near-duplicate rows flatter
    // every quantiser in the table equally.
    const PROMPT: &str = "The quantisation of an attention cache is decided by the \
        distribution of its keys, not by the size of its values. A key vector is produced \
        by a projection and then rotated by position, and the projection is trained \
        without any pressure towards an even spread of magnitude across its output \
        channels. What emerges instead, in almost every trained transformer anyone has \
        looked at, is a small handful of channels carrying magnitudes an order or two \
        above the rest, in the same places for every token. Group a run of consecutive \
        channels together and give them one shared scale, and that scale is set by \
        whichever outlier happens to fall inside the group; every other value in the \
        group is then represented with a fraction of the levels it could have had. \
        At eight bits nobody notices. At four bits it costs a little quality and the \
        usual answer is to group along the token axis instead, where the outliers line \
        up. At three bits there are eight levels in total and there is nothing left to \
        give away, which is why a scheme that ignores the problem stops working there \
        rather than degrading gracefully. \
        An orthogonal rotation is the standard way out. Multiply the key by a matrix that \
        preserves lengths and angles, and the energy that sat in one channel is spread \
        across all of them; the rotated coordinates look Gaussian, no group has a \
        dominant member, and one scale per group is finally a reasonable thing to share. \
        The rotation costs nothing in accuracy for keys, because an attention score is a \
        dot product and a dot product is exactly what an orthogonal map leaves alone: \
        rotate the stored key once when it is written, rotate the query once per step \
        when it is read, and the score that comes out is the score that would have come \
        out without either. There is no correction term, no residual to carry alongside \
        the codes, and no sketch to estimate. \
        Values are a different matter. A value vector is averaged against softmax \
        weights rather than dotted with a query, so there is no identity to exploit and \
        no argument from orthogonality that says the rotation is free. Whether spreading \
        the outliers still pays for values is a question about real tensors, and it is \
        settled by measuring rather than by reasoning about it. Explain, carefully and at \
        length, why a rotation applied before a low-bit scalar quantiser changes what \
        that quantiser can represent, what it costs to undo, and under what conditions \
        the whole exercise fails to pay for itself.";

    let Some(path) = ollama_blob(TAG) else {
        eprintln!("turboquant: {TAG} is not on disk, nothing measured");
        return;
    };

    let file = std::fs::File::open(&path).expect("open blob");
    let file_size = file.metadata().expect("stat blob").len();
    // SAFETY: the GGUF blob is a read-only content-addressed file in the model store;
    // mapping it is how every other loader in the tree reads weights, and the mapping is
    // kept alive by the Arc for as long as the parsed Content views it.
    let mmap = std::sync::Arc::new(unsafe { Mmap::map(&file) }.expect("mmap blob"));
    let content = gguf_file::Content::read_mapped(&mut Cursor::new(&mmap[..]), mmap.clone())
        .expect("parse gguf");
    let num_layers = content
        .metadata
        .iter()
        .find(|(k, _)| *k == "block_count" || k.ends_with(".block_count"))
        .and_then(|(_, v)| v.to_u32().ok())
        .expect("gguf block_count") as usize;

    let tokenizer = build_tokenizer_from_gguf(&content).expect("tokenizer");
    let ids: Vec<u32> = tokenizer
        .encode(PROMPT, true)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(
        ids.len() >= 128,
        "prompt tokenised to only {} tokens, which is fewer than four 32-token windows",
        ids.len()
    );

    // Empty device list and empty device map: every layer lands on the host. This is the
    // same constructor the server calls, so the K and V are the ones production writes.
    let cuda_devices: Vec<(usize, u64)> = Vec::new();
    let hetero: HashMap<usize, Device> = HashMap::new();
    let plan = HeteroPlan::calculate_with_kv(num_layers, file_size, &cuda_devices, &[], 1.0, 0);
    let mut model = GenericHeteroTransformer::from_gguf_with_kv_quant(
        content,
        &mmap,
        &hetero,
        &plan,
        KvQuant::Off,
        Some(4096),
    )
    .expect("load model on cpu");

    let n = ids.len();
    let input = Tensor::from_vec(ids, (1, n), &Device::Cpu).expect("input tensor");
    model.forward(&input, 0).expect("forward");

    // Truncate to a multiple of the largest group size any column uses. The per-channel
    // columns group along the token axis, so the token count must divide by 64 as well as
    // by the Q4_0 block of 32 - and it is truncated once, here, so that every column
    // measures the same slice rather than each trimming to suit itself.
    const TOKEN_ALIGNMENT: usize = 64;
    let tokens = n - n % TOKEN_ALIGNMENT;
    assert!(
        tokens >= TOKEN_ALIGNMENT,
        "only {tokens} tokens after truncation"
    );

    let mut captured: Vec<CapturedLayer> = Vec::new();
    for (i, layer) in model.layers().iter().enumerate() {
        let Some((k, v)) = layer.kv_cache.current_kv() else {
            continue;
        };
        let dtype = k.dtype();
        let Some((k, n_kv_heads, head_dim)) = flatten_kv(&k, tokens) else {
            continue;
        };
        let Some((v, _, _)) = flatten_kv(&v, tokens) else {
            continue;
        };
        captured.push(CapturedLayer {
            layer: i,
            k,
            v,
            n_kv_heads,
            tokens,
            head_dim,
            dtype,
        });
    }
    assert!(!captured.is_empty(), "no layer produced a K/V cache");

    let head_dim = captured[0].head_dim;
    let dtype = captured[0].dtype;
    println!(
        "\nturboquant: {TAG}, {} layers captured, {tokens} tokens, cache dtype {dtype:?}",
        captured.len()
    );
    if dtype == DType::F16 {
        // Worth saying out loud: when the cache already stores f16, the f16 column is a
        // tautology and reads as zero error. It is not a bug in the harness.
        println!("turboquant: the cache is already f16, so the f16 column is its own reference");
    }

    // -- Stage isolation, before any table --------------------------------------------
    //
    // A negative result that contradicts a published claim has to be suspected of being
    // our bug first. Two things must hold before the reconstruction numbers mean anything:
    // the rotation must be an exact involution on this data, and it must actually flatten
    // these rows. Either failing would explain the tables without the design being wrong.
    let rotation = HadamardRotation::new(head_dim, MEASUREMENT_ROTATION_SEED);
    println!(
        "\nrotation check (head_dim {head_dim}, block {}):",
        rotation.block_len()
    );
    println!(
        "{:>6}  {:>12}  {:>12}  {:>10} {:>10}  {:>8} {:>8}",
        "tensor", "roundtrip", "norm drift", "kurt.pre", "kurt.post", "crest.pre", "crest.post"
    );
    for (name, values) in [("K", &captured[0].k), ("V", &captured[0].v)] {
        let d = rotation_diagnostics(values, head_dim, &rotation);
        println!(
            "{name:>6}  {:>12.3e}  {:>12.3e}  {:>10.2} {:>10.2}  {:>8.2} {:>8.2}",
            d.max_round_trip_error,
            d.max_relative_norm_change,
            d.kurtosis_before,
            d.kurtosis_after,
            d.crest_before,
            d.crest_after
        );
        // The rotation must not be contributing to the error the tables attribute to three
        // bits. Three bits costs ~1.6e-1 relative; anything the rotation loses has to be
        // orders of magnitude under that, which these bounds enforce.
        assert!(
            d.max_round_trip_error < 1e-4,
            "{name}: rotation round trip is off by {:.3e} of a row RMS - the reported \
             reconstruction error includes a rotation error",
            d.max_round_trip_error
        );
        assert!(
            d.max_relative_norm_change < 1e-4,
            "{name}: rotation changed a row norm by {:.3e} - it is not orthogonal on this \
             data",
            d.max_relative_norm_change
        );
    }
    // Averaged over every layer, so the shape claim is not decided by layer zero.
    let (mut kb, mut ka, mut cb, mut ca) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for c in &captured {
        let d = rotation_diagnostics(&c.k, head_dim, &rotation);
        kb += d.kurtosis_before;
        ka += d.kurtosis_after;
        cb += d.crest_before;
        ca += d.crest_after;
    }
    let n = captured.len() as f64;
    println!(
        "K over all {} layers: excess kurtosis {:.2} -> {:.2}, crest {:.2} -> {:.2}",
        captured.len(),
        kb / n,
        ka / n,
        cb / n,
        ca / n
    );

    // Two group sizes, because the eight-value packing group is the layout unit and not a
    // sensible scale unit: at g = 8 the side information alone costs more than the codes.
    for group in [8usize, 32, 64] {
        if !head_dim.is_multiple_of(group) {
            continue;
        }
        let reports: Vec<LayerReport> = captured
            .iter()
            .map(|c| {
                evaluate_layer(
                    c.layer,
                    &c.k,
                    &c.v,
                    c.n_kv_heads,
                    c.tokens,
                    c.head_dim,
                    group,
                )
            })
            .collect();
        println!("\n===== values per scale: {group} =====");
        println!("{}", format_table(&reports));
    }
}
