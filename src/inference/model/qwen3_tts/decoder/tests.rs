//! The left context and the chunks on toy chains, and the decoder's chunked and streamed decodes
//! on the published checkpoint.
//! The comparison to the reference runs outside the test suite.

use super::*;
use crate::inference::generic_transformer::codec_class_tests;
use crate::inference::load::hf_config::tests::speech_snapshot;
use serde_json::json;

/// A toy codec configuration whose strides give 6 samples a frame.
fn toy_config() -> serde_json::Value {
    json!({
        "output_sample_rate": 60,
        "decode_upsample_rate": 6,
        "decoder_config": {
            "num_quantizers": 2,
            "codebook_size": 4,
            "num_semantic_quantizers": 1,
            "latent_dim": 8,
            "hidden_size": 4,
            "num_attention_heads": 2,
            "head_dim": 2,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "sliding_window": 3,
            "rope_theta": 10000,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 64,
            "upsampling_ratios": [2],
            "decoder_dim": 8,
            "upsample_rates": [3]
        }
    })
}

#[test]
fn the_config_reads_every_size_and_checks_the_strides() {
    let c = CodecConfig::read(&toy_config()).unwrap();
    assert_eq!(c.frame_samples, 6);
    assert_eq!(c.strides().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!((c.heads, c.head_dim, c.window), (2, 2, 3));
    let mut bad = toy_config();
    bad["decode_upsample_rate"] = json!(5);
    assert!(CodecConfig::read(&bad).is_err());
}

#[test]
fn the_left_context_sums_the_reach_of_every_stage() {
    // A conv of 3 taps at the frame rate reads 2 frames back: 12 samples.
    // 2 layers of a window of 3 read 4 frames back: 24 samples.
    // A transposed conv of 4 taps and stride 2 reads 1 frame back, and its output lies up to
    // 3 samples into its input: 9 samples.
    // A conv of 3 taps dilated by 2 at twice the frame rate reads 4 positions of 3 samples.
    // A transposed conv of 6 taps and stride 3 at twice the frame rate reads 1 position back,
    // and its output lies up to 2 samples into its input: 5 samples.
    // A conv of 2 taps at 6 times the frame rate reads 1 sample.
    let stages = [
        Reach::Conv {
            kernel: 3,
            dilation: 1,
        },
        Reach::Window {
            window: 3,
            layers: 2,
        },
        Reach::ConvTr {
            kernel: 4,
            stride: 2,
        },
        Reach::Conv {
            kernel: 3,
            dilation: 2,
        },
        Reach::ConvTr {
            kernel: 6,
            stride: 3,
        },
        Reach::Conv {
            kernel: 2,
            dilation: 1,
        },
    ];
    // 12 + 24 + 9 + 12 + 5 + 1 = 63 samples, 11 frames of 6.
    assert_eq!(left_context(&stages, 6).unwrap(), 11);
    // 12 + 24 + 9 + 12 = 57 samples, 10 frames of 6.
    assert_eq!(left_context(&stages[..4], 6).unwrap(), 10);
    assert_eq!(left_context(&[], 6).unwrap(), 0);
    let odd = [Reach::ConvTr {
        kernel: 8,
        stride: 4,
    }];
    assert!(left_context(&odd, 6).is_err());
}

#[test]
fn the_chunk_is_the_most_frames_within_the_reserve() {
    let square = |n: usize| (n * n) as u64;
    assert_eq!(frames_within(100, 1000, square), Some(10));
    assert_eq!(frames_within(99, 1000, square), Some(9));
    assert_eq!(frames_within(100, 7, square), Some(7));
    assert_eq!(frames_within(0, 7, square), None);
}

#[test]
fn the_passes_keep_every_frame_once_in_order() {
    let cases = [
        (0, 1, 4, 2),
        (0, 10, 4, 2),
        (0, 11, 5, 1),
        (0, 3, 10, 9),
        (6, 10, 4, 2),
    ];
    for (first, total, span, context) in cases {
        let p = passes(first, total, span, context);
        let mut next = first;
        for &(from, start, end) in &p {
            assert_eq!(start, next);
            assert!(end - from <= span);
            assert_eq!(start - from, start.min(context));
            next = end;
        }
        assert_eq!(next, total);
    }
}

/// `t` as F32 values on the host.
fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1_f32()
        .unwrap()
}

/// The largest difference of `got` from `want`, and the largest magnitude of `want`.
fn error(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len());
    assert!(got.iter().all(|v| v.is_finite()));
    let err = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    (err, want.iter().map(|v| v.abs()).fold(0f32, f32::max))
}

/// The tolerance on a value of magnitude `scale` computed in F32.
///
/// A product of `width` terms rounds to about `eps * sqrt(width)` of its scale.
/// The roundings of a chain of `products` add up.
fn tolerance(products: usize, width: usize, scale: f32) -> f32 {
    f32::EPSILON * (width as f32).sqrt() * products as f32 * scale
}

/// The published checkpoint's directory, `None` when the test store lacks it.
fn published() -> Option<std::path::PathBuf> {
    let dir = speech_snapshot();
    if dir.is_none() {
        eprintln!("the published checkpoint is not in the test store");
    }
    dir
}

/// The decoders to test: one built on the host, and the one the ladder places when it places
/// it on a card.
fn decoders(dir: &std::path::Path) -> Vec<CodecDecoder> {
    let mut out = vec![CodecDecoder::on_host(dir).unwrap()];
    let placed = CodecDecoder::load(dir).unwrap();
    if !matches!(placed.device(), Device::Cpu) {
        out.push(placed);
    }
    out
}

/// The products chained up to each stage, and the widest of them.
struct Chain {
    products: usize,
    width: usize,
}

impl Chain {
    fn then(&mut self, products: usize, width: usize) -> &mut Self {
        self.products += products;
        self.width = self.width.max(width);
        self
    }
}

/// `n` frames of codes, each within the codebook, that differ from a frame to the next.
fn sample_frames(dec: &CodecDecoder, n: usize) -> Vec<Vec<u32>> {
    let c = dec.config();
    (0..n)
        .map(|t| {
            (0..c.quantizers)
                .map(|g| ((t * 131 + g * 17 + 5) % c.codebook_size) as u32)
                .collect()
        })
        .collect()
}

/// The waveform of 8 frames has its samples, each within the bound, and the same frames give
/// the same samples twice.
#[test]
fn the_published_waveform_has_its_samples_and_repeats() {
    let Some(dir) = published() else { return };
    for dec in decoders(&dir) {
        let place = format!("{:?}", dec.device());
        let frames = sample_frames(&dec, 8);
        let wave = dec.decode(&frames).unwrap();
        assert_eq!(
            wave.len(),
            frames.len() * dec.config().frame_samples,
            "{place}"
        );
        assert!(wave
            .iter()
            .all(|v| v.is_finite() && v.abs() <= WAVEFORM_BOUND));
        assert_eq!(wave, dec.decode(&frames).unwrap(), "{place}");
    }
}

/// The products from the codes to the waveform, and the widest of them.
fn whole_chain(dec: &CodecDecoder) -> Chain {
    let c = dec.config();
    let mut chain = Chain {
        products: 3,
        width: c.codebook_size * dec.pre_conv.kernel(),
    };
    chain.then(4 * c.layers + 2, c.intermediate.max(c.heads * c.head_dim));
    for u in &dec.upsample {
        chain.then(4, u.widths().1);
    }
    chain.then(1, c.latent * dec.conv_in.kernel());
    for b in &dec.blocks {
        let (c_in, k) = (b.convtr.w.dims()[0], b.convtr.kernel());
        chain.then(1 + 2 * b.units.len(), c_in * k);
    }
    chain.then(1, dec.conv_out.conv_in(1) * dec.conv_out.kernel());
    chain
}

/// A decode in chunks of a few frames past the left context gives the samples of a whole one,
/// within the rounding of F32 through the decoder.
///
/// The back runs in 2 passes over a list of frames that the front decodes whole.
/// The second pass keeps fewer frames than the first.
/// The front runs in 2 passes over a list longer than its left context.
#[test]
fn a_chunked_decode_gives_the_whole_decode() {
    let Some(dir) = published() else { return };
    for dec in decoders(&dir) {
        let place = format!("{:?}", dec.device());
        let derived = dec.chunking();
        eprintln!("{place}: derived chunking {derived:?}");
        let whole = Chunking {
            front_frames: usize::MAX,
            front_context: 0,
            back_frames: usize::MAX,
            back_context: 0,
        };
        let chain = whole_chain(&dec);
        let frames = sample_frames(&dec, derived.back_context + 5);
        let small = Chunking {
            back_frames: derived.back_context + 3,
            ..whole
        };
        let small = Chunking {
            back_context: derived.back_context,
            ..small
        };
        let want = dec.decode_in(&frames, whole).unwrap();
        let got = dec.decode_in(&frames, small).unwrap();
        let (err, scale) = error(&got, &want);
        let tol = tolerance(chain.products, chain.width, scale);
        eprintln!("{place}: back in chunks {err} away at a scale of {scale}, tolerance {tol}");
        assert!(
            err <= tol,
            "{place}: back in chunks {err} away, tolerance {tol}"
        );

        let frames = sample_frames(&dec, derived.front_context + 5);
        let codes = dec.codes(&frames).unwrap();
        let want = host(&dec.front_rows(&codes, whole).unwrap());
        let small = Chunking {
            front_frames: derived.front_context + 3,
            front_context: derived.front_context,
            ..whole
        };
        let got = host(&dec.front_rows(&codes, small).unwrap());
        let (err, scale) = error(&got, &want);
        let c = dec.config();
        let tol = tolerance(4 * c.layers + 5, c.intermediate, scale);
        eprintln!("{place}: front in chunks {err} away at a scale of {scale}, tolerance {tol}");
        assert!(
            err <= tol,
            "{place}: front in chunks {err} away, tolerance {tol}"
        );
    }
}

/// Frames fed to a stream in uneven groups give the samples of a whole decode, within the
/// rounding of F32 through the decoder.
///
/// The back runs in passes of a few frames past its left context, so a group crosses passes.
/// Each group's time is printed.
#[test]
fn a_streamed_decode_gives_the_whole_decode() {
    let Some(dir) = published() else { return };
    for dec in decoders(&dir) {
        let place = format!("{:?}", dec.device());
        let derived = dec.chunking();
        let groups = [1, 3, 2, derived.back_context + 4, 1];
        let frames = sample_frames(&dec, groups.iter().sum());
        let want = dec.decode(&frames).unwrap();
        let small = Chunking {
            back_frames: derived.back_context + 3,
            ..derived
        };
        let mut stream = dec.stream_in(small);
        let mut got = Vec::new();
        let mut at = 0;
        for n in groups {
            let t0 = std::time::Instant::now();
            let wave = stream.push(&frames[at..at + n]).unwrap();
            eprintln!(
                "{place}: {n} frames after {at} in {:.3} s",
                t0.elapsed().as_secs_f64()
            );
            assert_eq!(wave.len(), n * dec.config().frame_samples);
            got.extend(wave);
            at += n;
        }
        let chain = whole_chain(&dec);
        let (err, scale) = error(&got, &want);
        let tol = tolerance(chain.products, chain.width, scale);
        eprintln!("{place}: streamed {err} away at a scale of {scale}, tolerance {tol}");
        assert!(err <= tol, "{place}: streamed {err} away, tolerance {tol}");
    }
}

/// The time of a streamed call past the left context of both parts, on each device: the front
/// over its left context and the new frames, then the back over its left context and the new
/// frames.
#[test]
#[ignore = "a timing, run on demand in a release build"]
fn the_time_of_a_streamed_call_past_the_left_context() {
    let Some(dir) = published() else { return };
    for dec in decoders(&dir) {
        let c = dec.chunking();
        for n in [1, 4, 12, 24] {
            let frames = sample_frames(&dec, c.front_context + n);
            let codes = dec.codes(&frames).unwrap();
            let _ = host(&dec.front(&codes).unwrap());
            let t0 = std::time::Instant::now();
            let rows = dec.front(&codes).unwrap();
            let _ = host(&rows);
            let front = t0.elapsed().as_secs_f64();
            let rows = rows.narrow(
                0,
                c.front_context + n - (c.back_context + n),
                c.back_context + n,
            );
            let rows = rows.unwrap().contiguous().unwrap();
            let t0 = std::time::Instant::now();
            let wave = dec.back_wave_from(&rows, c.back_context, c).unwrap();
            let back = t0.elapsed().as_secs_f64();
            eprintln!(
                "{:?}: a call of {n} frames: front over {} frames {front:.3} s, back over {} frames {back:.3} s, {} samples",
                dec.device(),
                c.front_context + n,
                c.back_context + n,
                wave.len()
            );
        }
    }
}

/// The time of one cold decode of 8 and of 300 frames on each device.
#[test]
#[ignore = "a timing, run on demand in a release build"]
fn the_decode_time_of_8_and_300_frames() {
    let Some(dir) = published() else { return };
    for dec in decoders(&dir) {
        for n in [8, 300] {
            let frames = sample_frames(&dec, n);
            let t0 = std::time::Instant::now();
            let wave = dec.decode(&frames).unwrap();
            eprintln!(
                "{:?}: decode of {n} frames in {:.3} s, {} samples",
                dec.device(),
                t0.elapsed().as_secs_f64(),
                wave.len()
            );
        }
    }
}

/// The front of the transformer as the codec's reference runs it, in F64: the input projection,
/// the layers and their final norm, the output projection.
fn reference_front(
    w: &std::collections::HashMap<String, Vec<f32>>,
    g: &codec_class_tests::Geometry,
    x: &[f32],
) -> Vec<f64> {
    let p = |n: &str| format!("{}{n}", transformer::PREFIX);
    let linear = |name: &str, x: &[f64], width: usize| -> Vec<f64> {
        let wt = &w[&p(&format!("{name}.weight"))];
        let b = &w[&p(&format!("{name}.bias"))];
        x.chunks(width)
            .flat_map(|row| {
                wt.chunks(width)
                    .zip(b)
                    .map(|(r, b)| {
                        r.iter().zip(row).map(|(a, v)| *a as f64 * v).sum::<f64>() + *b as f64
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let x: Vec<f64> = x.iter().map(|v| *v as f64).collect();
    let h = linear("input_proj", &x, g.latent);
    let rows = codec_class_tests::reference_rows(w, g, &h);
    linear("output_proj", &rows, g.hidden)
}

/// The transformer's tensors of the codec's file `file`, as F32 values by name.
fn transformer_weights(file: &std::path::Path) -> std::collections::HashMap<String, Vec<f32>> {
    let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::new(file).unwrap() };
    st.byte_sizes()
        .iter()
        .filter(|(n, _)| n.starts_with(transformer::PREFIX))
        .map(|(n, _)| (n.to_string(), host(&st.load(n, &Device::Cpu).unwrap())))
        .collect()
}

/// The transformer `t` over `frames` rows of the latent width, against the reference of `w`,
/// within the rounding of `eps` through the projections, the layers and the final norm.
fn the_transformer_matches_the_reference(
    t: &Transformer,
    w: &std::collections::HashMap<String, Vec<f32>>,
    g: &codec_class_tests::Geometry,
    frames: usize,
    eps: f64,
    place: &str,
) {
    let mut seed = 9u32;
    let x = crate::inference::generic_transformer::class_load::tests::values(
        &mut seed,
        frames * g.latent,
        0.0,
        1.0,
    );
    let want = reference_front(w, g, &x);
    let xs = Tensor::from_vec(x, (frames, g.latent), &Device::Cpu).unwrap();
    let got = host(&t.forward(&xs).unwrap());
    let (err, scale) = codec_class_tests::error(&got, &want);
    let widest = g.ffn.max(g.heads * g.head_dim).max(g.latent);
    let tol = eps * (widest as f64).sqrt() * (4 * g.layers + 3) as f64 * scale;
    eprintln!("{place}: transformer {err} away at a scale of {scale}, tolerance {tol}");
    assert!(
        err <= tol,
        "{place}: transformer {err} away, tolerance {tol}"
    );
}

/// The host alone, then the cards the ladder sees, each with the machine epsilon of F32: the
/// codec's layers keep the stored F32 on a card.
fn placements() -> Vec<(Vec<(usize, u64)>, f64, &'static str)> {
    let mut out = vec![(Vec::new(), f32::EPSILON as f64, "host")];
    let cards = crate::inference::place::ladder::cards(0);
    if !cards.is_empty() {
        out.push((cards, f32::EPSILON as f64, "cards"));
    }
    out
}

/// The toy's transformer on the generic layers, its projections around them, gives the
/// reference's rows past the window.
#[test]
fn the_toy_transformer_on_the_generic_layers_matches_the_reference() {
    let toy = codec_class_tests::Toy::write("codec-glue", 2, 6);
    let g = codec_class_tests::Geometry::of(&toy.root);
    let file = std::path::PathBuf::from(toy.file());
    let vb = unsafe { VarBuilder::from_files(&[&file], DType::F32, &Device::Cpu).unwrap() };
    let vb = vb.pp(transformer::PREFIX.trim_end_matches('.'));
    for (cards, eps, place) in placements() {
        let layers = Transformer::layers(&toy.root, &file, 64, &cards).unwrap();
        let t = Transformer::new(&vb, layers).unwrap();
        the_transformer_matches_the_reference(&t, &toy.weights, &g, 3 * g.window + 1, eps, place);
    }
}

/// The published transformer on the generic layers gives the reference's rows over more frames
/// than its window, on the host and on the cards.
#[test]
fn the_published_transformer_matches_the_reference() {
    let Some(dir) = published() else { return };
    let file = weights_of(&dir);
    let w = transformer_weights(&file);
    let (root, config) = read_config(&dir.join(CODEC_DIR)).unwrap();
    let g = codec_class_tests::Geometry::of(&root);
    let vb = unsafe { VarBuilder::from_files(&[&file], DType::F32, &Device::Cpu).unwrap() };
    let vb = vb.pp(transformer::PREFIX.trim_end_matches('.'));
    for (cards, eps, place) in placements() {
        let layers = Transformer::layers(&root, &file, config.max_positions, &cards).unwrap();
        let t = Transformer::new(&vb, layers).unwrap();
        eprintln!("{place}: layers on {:?}", t.placement());
        the_transformer_matches_the_reference(&t, &w, &g, g.window + 28, eps, place);
    }
}

/// The dry plans of the published transformer's layers by their own demand: a card that holds
/// half of their weights, a card that holds all of the demand, the host alone.
#[test]
fn the_published_transformer_places_by_its_own_demand() {
    use crate::inference::place::layer_executor::DeviceKind;
    let Some(dir) = published() else { return };
    let (root, config) = read_config(&dir.join(CODEC_DIR)).unwrap();
    let shards = [weights_of(&dir).to_string_lossy().to_string()];
    let c = crate::inference::generic_transformer::class_load::class_demand(
        &root,
        &shards,
        transformer::CODEC_TRANSFORMER_CLASS,
        0,
    )
    .unwrap();
    let d = c.at(config.max_positions as u64);
    let whole =
        d.block_bytes + d.head_bytes + d.layers as u64 * d.kv_bytes_per_layer + d.runtime_reserve;
    eprintln!("demand {d:?}, {whole} bytes in all");
    let on_card = |cards: &[(usize, u64)]| -> usize {
        d.plan(cards)
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .map(|s| s.num_layers())
            .sum()
    };
    let small = on_card(&[(0, d.block_bytes / 2)]);
    let large = on_card(&[(0, 4 * whole)]);
    let host = on_card(&[]);
    eprintln!(
        "layers on the card of {}: small card {small}, large card {large}, host {host}",
        d.layers
    );
    assert!(small < d.layers);
    assert_eq!(large, d.layers);
    assert_eq!(host, 0);
}
