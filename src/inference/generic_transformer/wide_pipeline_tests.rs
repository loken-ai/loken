//! A Sinkhorn wide-residual model cut between 2 layers, on the synthetic model of the frozen V4.1
//! oracle and on the same model with engram layers.
//!
//! Each range is the generic model loaded whole, then narrowed to its layers.
//! Every boundary crosses as its bytes.

use super::compressed_band::BandLeft;
use super::wide_pipeline::{WideBoundary, WideSettle, HEADER_BYTES};
use super::wide_sparse_engram_tests::{generic_engram, v41_engram};
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::*;
use crate::inference::generic_transformer::wide_sparse_toy::prompt;
use crate::inference::model::deepseek_v41::band::SharedAttn;
use crate::inference::model::deepseek_v41::model::oracle_gate::synthetic;
use crate::inference::model::deepseek_v41::model::{
    DecodeBoundary, DecodeRangeIn, DecodeRangeOut, DeepseekV41Model,
};
use crate::inference::serve::pipeline_segment::{serve_segment, SegmentLink};

/// The oracle's 2 layouts, then the 5-layer band of the V4.1 test
/// `a_mid_band_boundary_survives_a_byte_crossing`, whose every band layer is a kv-source and an
/// index-source.
const CUT_LAYOUTS: [(&[u32], &[u32], &[u32]); 3] = [
    LAYOUTS[0],
    LAYOUTS[1],
    (&[0, 2, 2, 2, 0], &[1, 2, 3], &[1, 2, 3]),
];

/// Tokens of the oracle's verified block, as `wide_sparse_oracle_tests` holds it.
const BLOCK: usize = 6;

/// Tokens the oracle's rewound block keeps, as `wide_sparse_oracle_tests` holds it.
const KEPT: usize = 3;

/// The generic model of a layout's file: `(ratios, kv sources, index sources)`.
type Make = fn(&[u32], &[u32], &[u32]) -> GenericHeteroTransformer;

/// The V4.1 model of a layout's file.
type MakeV41 = fn(&[u32], &[u32], &[u32]) -> DeepseekV41Model;

fn plain_v41(ratios: &[u32], kv_src: &[u32], index_src: &[u32]) -> DeepseekV41Model {
    synthetic(ratios, kv_src, index_src, 0).0
}

/// The 2 files of every layout: the oracle's, and the oracle's with engram layers.
const FILES: [(&str, Make, MakeV41); 2] = [
    ("plain", generic, plain_v41),
    ("engram", generic_engram, v41_engram),
];

fn ids(tokens: &[u32]) -> Tensor {
    Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The first offset where `got` and `want` differ, or their common length when one ends first.
fn first_difference(got: &[u8], want: &[u8]) -> Option<usize> {
    match got.iter().zip(want).position(|(a, b)| a != b) {
        Some(i) => Some(i),
        None => (got.len() != want.len()).then(|| got.len().min(want.len())),
    }
}

/// The ranges `[0, cut)` and `[cut, n)` of the model `make` builds.
fn ranges(
    make: &dyn Fn() -> GenericHeteroTransformer,
    cut: usize,
    n: usize,
) -> [GenericHeteroTransformer; 2] {
    let mut head = make();
    head.narrow_to_layers(0, cut).unwrap();
    let mut tail = make();
    tail.narrow_to_layers(cut, n).unwrap();
    [head, tail]
}

/// The tokens through the head and the tail, the boundary crossing as its bytes.
fn split(
    cut: &mut [GenericHeteroTransformer; 2],
    tokens: &[u32],
    first: usize,
    all: bool,
) -> Tensor {
    let [head, tail] = cut;
    let bytes = head
        .wide_head_forward(&ids(tokens), first, all)
        .unwrap()
        .to_bytes()
        .unwrap();
    let got = WideBoundary::from_bytes(&bytes).unwrap();
    assert_eq!(
        got.to_bytes().unwrap(),
        bytes,
        "the bytes of the rebuilt boundary"
    );
    tail.wide_tail_forward(&got, all).unwrap()
}

/// Both ranges cut back to `new_len`, the band settle crossing as its bytes.
fn split_trim(cut: &mut [GenericHeteroTransformer; 2], new_len: usize) {
    let [head, tail] = cut;
    let settle = head
        .wide_trim(new_len)
        .unwrap()
        .map(|s| WideSettle::from_bytes(&s.to_bytes().unwrap()).unwrap());
    tail.wide_trim_after(new_len, settle.as_ref()).unwrap();
}

/// The whole model and the cut model on the same steps: every step gives the same bits.
///
/// The steps are the oracle's: a prefill, the same tokens decoded 1 at a time, a verify onto a
/// prompt, the verify rewound to its start, a shorter verify and the decode after it.
/// Then a verify cut back inside its rows, for 2 kept lengths, and the decode after each.
fn assert_cut_runs_as_whole(
    whole: &mut GenericHeteroTransformer,
    cut: &mut [GenericHeteroTransformer; 2],
    tokens: &[u32],
    label: &str,
) {
    let step = |name: &str, want: Tensor, got: Tensor| {
        assert_eq!(bits(&got), bits(&want), "{label}: {name}");
    };
    step(
        "prefill",
        whole.forward_all(&ids(tokens), 0).unwrap(),
        split(cut, tokens, 0, true),
    );
    for (i, &t) in tokens.iter().enumerate() {
        step(
            &format!("decode at {i}"),
            whole.forward(&ids(&[t]), i).unwrap(),
            split(cut, &[t], i, false),
        );
    }

    let s = tokens.len() - BLOCK;
    let (prompt, block) = tokens.split_at(s);
    step(
        "prompt",
        whole.forward(&ids(prompt), 0).unwrap(),
        split(cut, prompt, 0, false),
    );
    step(
        "verify",
        whole.forward_all(&ids(block), s).unwrap(),
        split(cut, block, s, true),
    );
    whole.trim_kv(s);
    split_trim(cut, s);
    step(
        "rewound verify",
        whole.forward_all(&ids(&block[..KEPT]), s).unwrap(),
        split(cut, &block[..KEPT], s, true),
    );
    for (i, &t) in block[KEPT..].iter().enumerate() {
        let pos = s + KEPT + i;
        step(
            &format!("decode after the rewind at {pos}"),
            whole.forward(&ids(&[t]), pos).unwrap(),
            split(cut, &[t], pos, false),
        );
    }

    for kept in [1, KEPT] {
        step(
            &format!("prompt before keeping {kept}"),
            whole.forward(&ids(prompt), 0).unwrap(),
            split(cut, prompt, 0, false),
        );
        step(
            &format!("verify before keeping {kept}"),
            whole.forward_all(&ids(block), s).unwrap(),
            split(cut, block, s, true),
        );
        whole.trim_kv(s + kept);
        split_trim(cut, s + kept);
        for (i, &t) in block[kept..].iter().enumerate() {
            let pos = s + kept + i;
            step(
                &format!("decode at {pos} after keeping {kept}"),
                whole.forward(&ids(&[t]), pos).unwrap(),
                split(cut, &[t], pos, false),
            );
        }
    }
}

/// For every layout, with and without engram layers, a head and a tail cut at every layer
/// boundary give the bits of the whole model: in the prefill, every decode step, the verify,
/// the rewound verify, the verifies cut back inside their rows and every decode after them.
#[test]
fn every_cut_gives_the_bits_of_the_whole_model() {
    for (name, make, _) in FILES {
        for (ratios, kv_src, index_src) in CUT_LAYOUTS {
            let tokens = prompt();
            let n = ratios.len();
            let build = || make(ratios, kv_src, index_src);
            for at in 1..n {
                let mut whole = build();
                let mut cut = ranges(&build, at, n);
                let label = format!("{name} layout {ratios:?}, cut at {at}");
                assert_cut_runs_as_whole(&mut whole, &mut cut, &tokens, &label);
            }
        }
    }
}

/// For every layout, with and without engram layers, at every layer boundary, the body of each
/// decode step's boundary is the byte format of the V4.1 path's boundary for the same cut, after
/// the same prompt.
/// The V4.1 tail and the generic tail then answer with the same bits.
#[test]
fn a_decode_boundary_is_the_v41_boundary_byte_for_byte() {
    for (name, make, make_v41) in FILES {
        for (ratios, kv_src, index_src) in CUT_LAYOUTS {
            let tokens = prompt();
            let v41 = make_v41(ratios, kv_src, index_src);
            let n = ratios.len();
            assert_eq!(v41.n_layers(), n);
            let s = tokens.len() - BLOCK;
            let build = || make(ratios, kv_src, index_src);
            for at in 1..n {
                let label = format!("{name} layout {ratios:?}, cut at {at}");
                let mut cut = ranges(&build, at, n);
                let mut state = v41.new_decode_state();
                v41.prefill_into(&tokens[..s], &mut state).unwrap();
                split(&mut cut, &tokens[..s], 0, false);
                for (i, &t) in tokens[s..].iter().enumerate() {
                    let pos = s + i;
                    let want = match v41
                        .forward_decode_range(DecodeRangeIn::Token(t), 0, at, &mut state)
                        .unwrap()
                    {
                        DecodeRangeOut::Boundary(b) => b.to_bytes().unwrap(),
                        DecodeRangeOut::Logits(_) => panic!("{label}: the V4.1 head answered"),
                    };
                    let [head, tail] = &mut cut;
                    let sent = head.wide_head_forward(&ids(&[t]), pos, false).unwrap();
                    let bytes = sent.to_bytes().unwrap();
                    let body = sent.body_bytes().unwrap();
                    assert_eq!(&bytes[HEADER_BYTES..], body.as_slice());
                    assert_eq!(
                        first_difference(&body, &want),
                        None,
                        "{label}: the boundary at {pos}, {} bytes for {}",
                        body.len(),
                        want.len()
                    );
                    let resumed = DecodeRangeIn::Resume(DecodeBoundary::from_bytes(&body).unwrap());
                    let v41_logits = match v41.forward_decode_range(resumed, at, n, &mut state) {
                        Ok(DecodeRangeOut::Logits(l)) => l,
                        _ => panic!("{label}: the V4.1 tail did not answer"),
                    };
                    let got = tail.wide_tail_forward(&sent, false).unwrap();
                    assert_eq!(
                        bits(&got),
                        bits(&v41_logits),
                        "{label}: the logits at {pos}"
                    );
                }
            }
        }
    }
}

/// The V4.1 encoder and the generic encoder write the same bytes for the same values, a band
/// value of each kind present.
/// The bytes rebuild the boundary, which neither a byte less nor a byte more does.
#[test]
fn the_v41_encoder_and_the_generic_encoder_write_the_same_bytes() {
    let (streams, hidden, groups, row, key) = (2, 4, 3, 2, 4);
    let values = |n: usize, k: f32| (0..n).map(|i| i as f32 * k - 0.5).collect::<Vec<f32>>();
    let left = BandLeft {
        compressed: Some(values(groups * row, 0.25)),
        index_keys: Some(values(groups * key, 0.125)),
        picks: Some(vec![5, -1, 7]),
        candidates: Some(vec![true, false, true]),
    };
    let wide = WideBoundary {
        first: 9,
        streams,
        hidden,
        residual: values(streams * hidden, 0.5),
        pre: vec![vec![0.25, 0.75]],
        recent: Some(vec![3, 1, 4]),
        widths: (row, key),
        band: vec![left.clone()],
    };
    let tensor = |v: &[f32], dims: (usize, usize, usize)| {
        Tensor::from_vec(v.to_vec(), dims, &Device::Cpu).unwrap()
    };
    let v41 = DecodeBoundary {
        x: Tensor::from_vec(wide.residual.clone(), (1, 1, streams, hidden), &Device::Cpu).unwrap(),
        pre_mix: wide.pre.clone(),
        shared: SharedAttn {
            compress_kv: left
                .compressed
                .as_deref()
                .map(|v| tensor(v, (1, groups, row))),
            index_k: left
                .index_keys
                .as_deref()
                .map(|v| tensor(v, (1, groups, key))),
            topk_idxs: left.picks.clone(),
            candidates: left.candidates.clone(),
        },
        recent: wide.recent.clone(),
    };
    assert_eq!(wide.body_bytes().unwrap(), v41.to_bytes().unwrap());

    let empty = WideBoundary {
        recent: None,
        band: vec![BandLeft::default()],
        ..wide.clone()
    };
    let v41_empty = DecodeBoundary {
        shared: SharedAttn::default(),
        recent: None,
        ..v41
    };
    assert_eq!(empty.body_bytes().unwrap(), v41_empty.to_bytes().unwrap());

    // 2 rows of a verify, each with its own band value.
    let verify = WideBoundary {
        pre: vec![vec![0.25, 0.75], vec![1.0, 0.0]],
        residual: values(2 * streams * hidden, 0.5),
        band: vec![left, BandLeft::default()],
        ..wide
    };
    let bytes = verify.to_bytes().unwrap();
    assert_eq!(WideBoundary::from_bytes(&bytes).unwrap(), verify);
    assert!(WideBoundary::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(WideBoundary::from_bytes(&longer).is_err());
}

/// For every layout, with and without engram layers, 2 sequences interleaved step by step on a
/// cut model, each over its own state on each side, give the bits each gives alone on the whole
/// model.
#[test]
fn interleaved_sequences_on_a_cut_give_the_bits_of_their_runs_alone() {
    for (name, make, _) in FILES {
        for (ratios, kv_src, index_src) in CUT_LAYOUTS {
            let tokens = prompt();
            let n = ratios.len();
            let build = || make(ratios, kv_src, index_src);
            let prompts: [&[u32]; 2] = [&tokens[..5], &tokens[7..10]];
            let steps = &tokens[12..18];
            let alone: Vec<Vec<Vec<u32>>> = prompts
                .iter()
                .map(|&p| {
                    let mut whole = build();
                    let mut rows = vec![bits(&whole.forward(&ids(p), 0).unwrap())];
                    for (i, &t) in steps.iter().enumerate() {
                        rows.push(bits(&whole.forward(&ids(&[t]), p.len() + i).unwrap()));
                    }
                    rows
                })
                .collect();
            for at in 1..n {
                let label = format!("{name} layout {ratios:?}, cut at {at}");
                let [mut head, mut tail] = ranges(&build, at, n);
                let mut seqs: Vec<_> = (0..prompts.len())
                    .map(|_| {
                        [
                            head.sequence_mixers().unwrap(),
                            tail.sequence_mixers().unwrap(),
                        ]
                    })
                    .collect();
                let mut run = |j: usize, toks: &[u32], first: usize| -> Vec<u32> {
                    let [hs, ts] = &mut seqs[j];
                    let sent = head
                        .with_sequence(hs, |m| m.wide_head_forward(&ids(toks), first, false))
                        .unwrap();
                    let got = WideBoundary::from_bytes(&sent.to_bytes().unwrap()).unwrap();
                    bits(
                        &tail
                            .with_sequence(ts, |m| m.wide_tail_forward(&got, false))
                            .unwrap(),
                    )
                };
                for (j, &p) in prompts.iter().enumerate() {
                    assert_eq!(run(j, p, 0), alone[j][0], "{label}: prompt {j}");
                }
                for (i, &t) in steps.iter().enumerate() {
                    for (j, &p) in prompts.iter().enumerate() {
                        assert_eq!(
                            run(j, &[t], p.len() + i),
                            alone[j][i + 1],
                            "{label}: sequence {j}, step {i}"
                        );
                    }
                }
            }
        }
    }
}

/// The head of a chain of segments over loopback, each segment served by `serve_segment`.
fn wire_chain(
    build: &dyn Fn() -> GenericHeteroTransformer,
    bounds: &[usize],
) -> (
    GenericHeteroTransformer,
    SegmentLink,
    Vec<std::thread::JoinHandle<()>>,
) {
    // The last segment first, so each segment's next one listens before it connects.
    let mut next = None;
    let mut threads = Vec::new();
    for w in bounds.windows(2).skip(1).rev() {
        let mut model = build();
        model.narrow_to_layers(w[0], w[1]).unwrap();
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        threads.push(std::thread::spawn(move || {
            serve_segment(model, next, listener).unwrap();
        }));
        next = Some(addr);
    }
    let mut head = build();
    head.narrow_to_layers(0, bounds[1]).unwrap();
    let link = SegmentLink::connect(next.unwrap()).unwrap();
    (head, link, threads)
}

/// For every layout, with and without engram layers, a head and the segments behind it on
/// loopback give the bits of the whole model, for every single cut and for a cut at every layer
/// boundary: a prompt, a verify, a cut inside the verify's rows, the decode after it, and a new
/// prompt.
#[test]
fn segments_on_the_wire_give_the_bits_of_the_whole_model() {
    for (name, make, _) in FILES {
        for (ratios, kv_src, index_src) in CUT_LAYOUTS {
            let tokens = prompt();
            let n = ratios.len();
            let build = || make(ratios, kv_src, index_src);
            let mut chains: Vec<Vec<usize>> = (1..n).map(|at| vec![0, at, n]).collect();
            chains.push((0..=n).collect());
            for bounds in chains {
                let label = format!("{name} layout {ratios:?}, bounds {bounds:?}");
                let mut whole = build();
                let (mut head, mut link, threads) = wire_chain(&build, &bounds);
                let mut step = |what: &str, toks: &[u32], first: usize, all: bool| {
                    let want = match all {
                        true => whole.forward_all(&ids(toks), first).unwrap(),
                        false => whole.forward(&ids(toks), first).unwrap(),
                    };
                    let sent = head.wide_head_forward(&ids(toks), first, all).unwrap();
                    let got = link.wide_boundary(&sent, all).unwrap();
                    assert_eq!(bits(&got), bits(&want), "{label}: {what}");
                };
                let s = tokens.len() - BLOCK;
                let (prompt, block) = tokens.split_at(s);
                step("prompt", prompt, 0, false);
                step("verify", block, s, true);
                let kept = s + KEPT;
                whole.trim_kv(kept);
                let settle = head.wide_trim(kept).unwrap();
                link.cut_with(kept, settle.as_ref()).unwrap();
                let mut step = |what: &str, toks: &[u32], first: usize| {
                    let want = whole.forward(&ids(toks), first).unwrap();
                    let sent = head.wide_head_forward(&ids(toks), first, false).unwrap();
                    let got = link.wide_boundary(&sent, false).unwrap();
                    assert_eq!(bits(&got), bits(&want), "{label}: {what}");
                };
                for (i, &t) in block[KEPT..].iter().enumerate() {
                    step("decode after the cut", &[t], kept + i);
                }
                step("new prompt", &tokens[3..9], 0);
                drop(link);
                for t in threads {
                    t.join().unwrap();
                }
            }
        }
    }
}

/// A model without a Sinkhorn wide residual is refused a wide boundary, and a boundary that
/// does not fit the model is refused.
#[test]
fn a_boundary_that_does_not_fit_is_refused() {
    let path = tiny_model::write("wide-boundary", 2);
    let mut dense = tiny_model::load(&path, 2);
    assert!(dense.wide_head_forward(&ids(&[1, 2]), 0, false).is_err());
    tiny_model::remove(&path);

    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let n = ratios.len();
    let [mut head, mut tail] = ranges(&|| generic(ratios, kv_src, index_src), 1, n);
    let sent = head.wide_head_forward(&ids(&[1, 2]), 0, false).unwrap();
    let wrong = WideBoundary {
        streams: sent.streams + 1,
        ..sent.clone()
    };
    assert!(tail.wide_tail_forward(&wrong, false).is_err());
    let doubled = WideBoundary {
        band: vec![BandLeft::default(); 2],
        ..sent.clone()
    };
    assert!(tail.wide_tail_forward(&doubled, false).is_err());
    let hashed = WideBoundary {
        recent: Some(vec![1, 2]),
        ..sent
    };
    assert!(tail.wide_tail_forward(&hashed, false).is_err());
}
