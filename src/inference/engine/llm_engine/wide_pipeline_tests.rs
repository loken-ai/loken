//! A Sinkhorn wide-residual model cut across 2 nodes on loopback, through the engine: the tail is
//! the share the engine's layer share loads, the head is what the engine makes of its loaded model.
//!
//! The model is the generic model of the small file of the frozen V4.1 oracle.
//! Head plus tail give the bits and the greedy text of the model run whole on 1 node.

use super::block_draft_tests::{collect, engine_of, run};
use super::*;
use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
use crate::inference::engine::pipeline_head::PipelineHeadBackend;
use crate::inference::generic_transformer::block_draft_tests::{generic_of, prompt, toy_files};
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::quantized::gguf_file::MappedGguf;

/// The oracle's band layouts, as `wide_sparse_oracle_tests` holds them: compress ratios, kv
/// source layers, index source layers.
const LAYOUTS: [(&[u32], &[u32], &[u32]); 2] =
    [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];

/// Tokens of the oracle's verified block, as `wide_sparse_oracle_tests` holds it.
const BLOCK: usize = 6;

/// Tokens the oracle's rewound block keeps, as `wide_sparse_oracle_tests` holds it.
const KEPT: usize = 3;

/// Tokens each request generates.
const NEW: usize = 20;

/// The small file of `layout`, without a draft.
fn file(layout: (&[u32], &[u32], &[u32])) -> MappedGguf {
    toy_files(layout, false).0
}

/// The share `[from, to)` of the model of `file`, loaded as the engine loads a layer share: the
/// plan over the share's layers moved to their place in the stack of `n`.
fn share_of(file: MappedGguf, from: usize, to: usize, n: usize) -> GenericHeteroTransformer {
    let MappedGguf { content, mmap, .. } = file;
    let plan = HeteroPlan::calculate_with_kv(to - from, 1 << 20, &[], &[], 1.0, 0).shifted(from, n);
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan).unwrap()
}

/// A tail serving `tail` on a loopback port, as `become_pipeline_tail` serves a share.
///
/// Returns its address and the thread that ends when the head hangs up.
fn serve(tail: GenericHeteroTransformer) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
    assert_eq!(
        tail.segment_unsupported_reason(),
        None,
        "the share the tail loads can be cut"
    );
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        crate::inference::serve::pipeline_segment::serve_segment(tail, None, listener).unwrap();
    });
    (addr, server)
}

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

/// A share of a wide-residual file holds its own layers, takes no draft and can be cut.
#[test]
fn a_share_of_a_wide_file_holds_its_layers() {
    for layout in LAYOUTS {
        let n = layout.0.len();
        for cut in 1..n {
            for (from, to) in [(0, cut), (cut, n)] {
                let share = share_of(toy_files(layout, true).0, from, to, n);
                assert_eq!(share.segment_range(), (from, to, n), "{layout:?}");
                assert!(!share.has_block_draft(), "a share takes no draft");
                assert_eq!(share.segment_unsupported_reason(), None, "{layout:?}");
            }
        }
        let whole = generic_of(toy_files(layout, true).0);
        assert!(
            whole.segment_unsupported_reason().is_some(),
            "a model with a block draft is not cut"
        );
    }
}

/// At every cut, the head backend loaded as the share `[0, cut)` over a tail on loopback gives
/// the bits of the whole model's backend: a prompt, decode steps, a verify cut back inside its
/// rows, the decodes after it, and a new prompt.
#[test]
fn a_wide_head_and_tail_give_the_bits_of_the_whole_backend() {
    let tokens = prompt();
    let s = tokens.len() - BLOCK;
    let (start, block) = tokens.split_at(s);
    for layout in LAYOUTS {
        let n = layout.0.len();
        for cut in 1..n {
            let (addr, server) = serve(share_of(file(layout), cut, n, n));
            let mut head =
                match PipelineHeadBackend::connect(share_of(file(layout), 0, cut, n), addr, cut) {
                    Ok(head) => head,
                    Err((_, e)) => panic!("{e}"),
                };
            let mut whole = GenericBackend(generic_of(file(layout)));
            let label = format!("layout {:?}, cut at {cut}", layout.0);
            assert_eq!(
                head.capabilities().trim_kv,
                whole.capabilities().trim_kv,
                "{label}: the cut of the cache"
            );
            assert_eq!(
                head.capabilities().pld,
                whole.capabilities().pld,
                "{label}: the verify"
            );
            let mut step = |name: &str, f: &dyn Fn(&mut dyn ModelBackend) -> Tensor| {
                let want = bits(&f(&mut whole));
                let got = bits(&f(&mut head));
                assert_eq!(got, want, "{label}: {name}");
            };
            step("prompt", &|m| m.forward(&ids(start), 0).unwrap());
            step("verify", &|m| m.forward_all(&ids(block), s).unwrap());
            let rewound = &|m: &mut dyn ModelBackend| {
                m.trim_kv(s + KEPT);
                m.forward(&ids(&block[KEPT..KEPT + 1]), s + KEPT).unwrap()
            };
            step("decode after the verify cut back", rewound);
            for (i, &t) in block[KEPT + 1..].iter().enumerate() {
                let pos = s + KEPT + 1 + i;
                step(&format!("decode at {pos}"), &|m| {
                    m.forward(&ids(&[t]), pos).unwrap()
                });
            }
            step("new prompt", &|m| {
                m.reset_kv_from(0);
                m.forward(&ids(&tokens), 0).unwrap()
            });
            assert!(head.unusable().is_none(), "{label}");
            drop(head);
            server.join().unwrap();
        }
    }
}

/// At every cut, an engine whose loaded model is made the head of a tail on loopback answers 2
/// greedy requests, plain then streamed, with the tokens and the text of an engine holding the
/// whole model.
#[tokio::test(flavor = "multi_thread")]
async fn a_wide_pipeline_through_the_engine_gives_the_text_of_one_node() {
    for layout in LAYOUTS {
        let n = layout.0.len();
        for cut in 1..n {
            let label = format!("layout {:?}, cut at {cut}", layout.0);
            let (whole, prompt) = engine_of(generic_of(file(layout))).await;
            let want = run(&whole, &prompt, NEW).await;
            let want_streamed = collect(whole, prompt.clone(), NEW).await;

            // The tail ends when the head the engine holds hangs up.
            let (addr, _server) = serve(share_of(file(layout), cut, n, n));
            let (engine, prompt) = engine_of(generic_of(file(layout))).await;
            engine.connect_pipeline_tail(addr, cut).await.unwrap();
            for round in 0..2 {
                let got = run(&engine, &prompt, NEW).await;
                assert_eq!(
                    got.tokens, want.tokens,
                    "{label}: request {round}, the tokens"
                );
                assert_eq!(got.text, want.text, "{label}: request {round}, the text");
            }
            assert_eq!(
                collect(engine, prompt, NEW).await,
                want_streamed,
                "{label}: the streamed text"
            );
        }
    }
}
