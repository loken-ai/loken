//! A model whose cache is cut over cards takes a prompt another instance read, and gives one it
//! read: the rows travel in the order of their positions, and the ranges are cut again for the
//! cards of the instance that takes them.
//! The decode state of a hybrid with such a cache travels between instances too.

use super::state_cache_tests::{argmax, floats, prompt, write_hybrid};
use super::tiny_model;
use super::GenericHeteroTransformer;
use crate::tensor::{Device, Tensor};

const LAYERS: usize = 3;
/// Blocks of the hybrid of the state cache tests.
const HYBRID_LAYERS: usize = 4;

/// The model at `path` of `layers` blocks on card 0, its cache cut over 2 cards when `cut`.
/// A single card holds both ranges when no second one is there.
fn load(path: &std::path::Path, layers: usize, cut: bool) -> GenericHeteroTransformer {
    let mut m = tiny_model::load_on_card(path, layers);
    if cut {
        let second = usize::from(Device::new_cuda(1).is_ok());
        let cards = [
            Device::new_cuda(0).unwrap(),
            Device::new_cuda(second).unwrap(),
        ];
        assert!(m.attach_context_shards(&cards) > 0, "no layer was cut");
    }
    m
}

fn ids(toks: &[u32]) -> Tensor {
    Tensor::new(toks, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
}

/// The largest difference between 2 rows of logits.
fn worst(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// The bound of the hand-over of a layer split across cards (`tp_tests`).
fn close(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let off = worst(got, want);
    assert!(off <= 1e-4 * scale, "{what}: off by {off} of {scale}");
}

/// The logits `got` of a model that took rows another model read, compared with `alone`, the
/// logits that model computes reading the prompt itself, and with `reader`, the logits of the
/// model that read it, all decoding the same tokens.
///
/// The rows of a cache cut over cards and the rows of a whole cache differ in their last bits.
/// The ranges hold F16 rows, which a whole cache may hold wider.
/// A cut layer projects and rotates its keys outside the fused kernels a whole layer uses.
/// So the rows a model takes are not the rows it computes itself, and `got` misses `alone` by
/// what the 2 caches differ by.
///
/// The tightest rule that holds: `got` is no farther from `alone` than `reader` is, or than the
/// bound of `close`.
/// `reader` against `alone` is the difference the 2 caches make reading the same prompt.
/// The hand-over adds none of its own.
fn handed(got: &[f32], alone: &[f32], reader: &[f32], what: &str) {
    let scale = alone.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let (off, apart) = (worst(got, alone), worst(reader, alone));
    eprintln!(
        "{what}: off by {off}, the reader by {apart}, the bound of close {}",
        1e-4 * scale
    );
    assert!(
        off <= apart.max(1e-4 * scale),
        "{what}: off by {off} of {scale}, the reader by {apart}"
    );
}

/// The rows a prompt leaves are handed over between a model whose cache is cut over cards and
/// one whose cache is whole, both ways, and between 2 cut caches.
/// The model that takes them decodes on to the logits it computes reading the prompt itself:
/// bit for bit between 2 cut caches.
#[test]
#[ignore = "needs a CUDA card"]
fn rows_of_a_cut_cache_are_handed_over_both_ways() {
    let path = tiny_model::write("cut-handover", LAYERS);
    let prompt = [1u32, 4, 2, 9, 3, 5];
    let asked = [1u32, 4, 2, 9, 3, 5, 8];
    let check = |from_cut: bool, to_cut: bool| {
        let mut from = load(&path, LAYERS, from_cut);
        let mut to = load(&path, LAYERS, to_cut);
        let mut alone = load(&path, LAYERS, to_cut);
        from.reserve_context(prompt.len());
        from.forward(&ids(&prompt), 0).unwrap();
        from.snapshot_kv(prompt.to_vec(), prompt.len()).unwrap();
        let (index, held) = from.best_kv_snapshot(&asked).unwrap();
        assert_eq!(held, prompt.len(), "the reader kept the prompt");
        let kv = from.export_kv_rows(index, 0, held).unwrap();
        assert_eq!(from.kv_layout(), to.kv_layout(), "one layout for both");
        let dtype = from.kv_snapshot_dtype(index).unwrap_or(0);
        to.import_kv_snapshot(prompt.to_vec(), held, kv.clone(), dtype)
            .unwrap();
        let (imported, _) = to.best_kv_snapshot(&asked).unwrap();
        // A cut cache keeps rows in F16: rows a whole cache held wider come back rounded.
        if from_cut || !to_cut {
            assert!(to.export_kv_rows(imported, 0, held).unwrap() == kv);
        }
        to.restore_kv_snapshot(imported).unwrap();
        // The prefill takes its ranges' blocks as the engine's does, so both read the same blocks.
        alone.reserve_context(prompt.len());
        alone.forward(&ids(&prompt), 0).unwrap();
        for (i, t) in [8u32, 2, 6].into_iter().enumerate() {
            let at = prompt.len() + i;
            let what = format!("token {i} from the cut {from_cut} to the cut {to_cut}");
            let got = floats(&to.forward(&ids(&[t]), at).unwrap());
            let want = floats(&alone.forward(&ids(&[t]), at).unwrap());
            let read = floats(&from.forward(&ids(&[t]), at).unwrap());
            if from_cut == to_cut {
                assert!(got == want, "{what}: not bit for bit");
            } else {
                handed(&got, &want, &read, &what);
            }
        }
    };
    check(false, true);
    check(true, false);
    check(true, true);
    tiny_model::remove(&path);
}

/// A conversation on a model whose cache is cut over cards resumes its second turn from the
/// snapshot of its first, after another prompt took the cache.
#[test]
#[ignore = "needs a CUDA card"]
fn a_cut_cache_resumes_a_conversation_from_its_snapshot() {
    let path = tiny_model::write("cut-turns", LAYERS);
    let first = [1u32, 4, 2, 9, 3, 5, 8];
    let second = [1u32, 4, 2, 9, 3, 5, 8, 2, 6, 7];
    let mut m = load(&path, LAYERS, true);
    let mut alone = load(&path, LAYERS, true);
    m.forward(&ids(&first), 0).unwrap();
    m.snapshot_kv(first.to_vec(), first.len()).unwrap();
    m.forward(&ids(&[7, 7, 6, 3]), 0).unwrap();
    let (index, held) = m.best_kv_snapshot(&second).unwrap();
    assert_eq!(held, first.len(), "the first turn is kept");
    let (_, len) = m.restore_kv_snapshot(index).unwrap();
    assert_eq!(len, first.len());
    close(
        &floats(&m.forward(&ids(&second[len..]), len).unwrap()),
        &floats(&alone.forward(&ids(&second), 0).unwrap()),
        "the second turn",
    );
    tiny_model::remove(&path);
}

/// The decode state of a hybrid whose global layer's cache is cut over cards is handed over
/// between it and the hybrid whole, both ways.
#[test]
#[ignore = "needs a CUDA card"]
fn the_state_of_a_hybrid_with_a_cut_cache_is_handed_over_both_ways() {
    hybrid_state_is_handed_over("cut-state", &|path, cut| load(path, HYBRID_LAYERS, cut));
}

/// A model built by `apart`, the hybrid with its cache cut over cards, and the hybrid whole
/// hand their decode state over to each other.
/// The model that takes it resumes from the reader's logits, then decodes on to the logits it
/// computes reading the prompt itself, as [`handed`] compares them.
fn hybrid_state_is_handed_over(
    tag: &str,
    apart: &dyn Fn(&std::path::Path, bool) -> GenericHeteroTransformer,
) {
    let dir = super::state_cache_tests::scratch_dir(tag);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("hybrid.gguf");
    write_hybrid(&file);
    let tokens = prompt(53);
    for from_cut in [false, true] {
        let mut reader = apart(&file, from_cut);
        let mut decoder = apart(&file, !from_cut);
        let mut alone = apart(&file, !from_cut);
        let what =
            |s: &str| format!("{tag}: {s} after the hand-over from the model apart {from_cut}");
        assert!(reader.carries_handover_state() && decoder.carries_handover_state());
        let (a, b) = (reader.state_layout(1), decoder.state_layout(1));
        assert_eq!(
            (a.kinds, a.geometry),
            (b.kinds, b.geometry),
            "{}",
            what("one layout for both")
        );
        reader.reserve_context(tokens.len());
        let logits = reader.forward(&ids(&tokens), 0).unwrap();
        reader.snapshot_prefix(&tokens, &logits).unwrap();
        let index = reader.state_snapshot_covering(&tokens).unwrap();
        let bytes = reader.export_state(index).unwrap();
        decoder.import_state(tokens.clone(), &bytes).unwrap();
        let (covered, resumed) = decoder.try_restore_prefix(&tokens).unwrap().unwrap();
        assert_eq!(covered, tokens.len());
        assert!(
            floats(&resumed) == floats(&logits),
            "{}",
            what("the reader's logits")
        );
        // The prefill takes its ranges' blocks as the engine's does, so both read the same blocks.
        alone.reserve_context(tokens.len());
        let read = alone.forward(&ids(&tokens), 0).unwrap();
        let mut next = argmax(&floats(&read)) as u32;
        for at in tokens.len()..tokens.len() + 4 {
            let want = floats(&alone.forward(&ids(&[next]), at).unwrap());
            handed(
                &floats(&decoder.forward(&ids(&[next]), at).unwrap()),
                &want,
                &floats(&reader.forward(&ids(&[next]), at).unwrap()),
                &what(&format!("position {at}")),
            );
            next = argmax(&want) as u32;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
