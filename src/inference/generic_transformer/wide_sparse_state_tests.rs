//! The state cache of the generic route over latent window and compressed band layers, on the
//! synthetic model of the frozen V4.1 oracle.
//!
//! A state is captured from a live model, written to bytes, read into a model that never saw
//! the tokens, and both continue on the same tokens.
//! Every continuation gives the same bits on both.

use super::state_cache::*;
use super::wide_sparse_oracle_tests::{generic, LAYOUTS};
use super::*;
use crate::inference::cache::state_disk::StateStore;
use crate::inference::generic_transformer::wide_sparse_toy::VOCAB;

/// Tokens per block of the states under test, as the V4.1 snapshot tests cut them.
const BLOCK: usize = 8;

/// The prompt the V4.1 snapshot tests run.
fn tokens() -> Vec<u32> {
    (0..80).map(|i| (i * 7 + 3) % VOCAB as u32).collect()
}

fn ids(tokens: &[u32]) -> Tensor {
    Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// The identity the states of a test are kept under.
fn state_identity(salt: &[u8]) -> [u8; 32] {
    crate::inference::cache::state_disk::identity(&[b"latent attention states".as_slice(), salt])
}

/// The logits bits of each of `steps` decoded from `pos`.
fn decode(m: &mut GenericHeteroTransformer, steps: &[u32], pos: usize) -> Vec<Vec<u32>> {
    steps
        .iter()
        .enumerate()
        .map(|(i, &t)| bits(&m.forward(&ids(&[t]), pos + i).unwrap()))
        .collect()
}

/// Values the latent layers of `m` hold in their windows, compressed rows and index keys.
fn latent_values(m: &GenericHeteroTransformer) -> usize {
    m.layers
        .iter()
        .map(|l| match &l.mixer {
            mixer::Mixer::LatentMqa(x) => x.state.as_ref().map_or(0, |s| s.window.len()),
            mixer::Mixer::CompressedBand(b) => b.state.as_ref().map_or(0, |s| {
                s.window.window.len() + s.compressed.len() + s.index_keys.len()
            }),
            _ => 0,
        })
        .sum()
}

/// A window layer is a latent layer of its key row's width. A band layer keeps its ratio, and
/// compressed rows only on a kv-source.
#[test]
fn the_latent_layers_are_stored_as_their_own_kinds() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let model = generic(ratios, kv_src, index_src);
        let kinds = model.state_layout(BLOCK).kinds;
        assert_eq!(kinds.len(), ratios.len());
        for (il, kind) in kinds.iter().enumerate() {
            match *kind {
                StateKind::Latent { row } => {
                    assert_eq!(ratios[il], 0, "{ratios:?} layer {il}");
                    assert!(row > 0);
                }
                StateKind::Band {
                    row,
                    ratio,
                    compressed,
                    index,
                } => {
                    let kv = kv_src.contains(&(il as u32));
                    let ix = index_src.contains(&(il as u32));
                    assert_eq!(ratio, ratios[il] as usize, "{ratios:?} layer {il}");
                    assert_eq!(compressed > 0, kv, "{ratios:?} layer {il}");
                    assert_eq!(index > 0, kv && ix, "{ratios:?} layer {il}");
                    assert!(compressed == 0 || compressed == row);
                }
                other => panic!("{ratios:?} layer {il} is {other:?}"),
            }
        }
    }
}

/// A state captured at every split, written to bytes and read into a fresh model, continues
/// as the live model does: the prompt's remainder and 3 decoded tokens give the same bits.
///
/// The splits fall inside the first group, inside and past the window, on and off the block
/// boundaries.
/// The rows go on their grids, so the manifest is a fraction of the f32 state.
/// Too few blocks are refused.
#[test]
fn a_restored_snapshot_decodes_the_bits_of_the_live_model() {
    let tokens = tokens();
    for (ratios, kv_src, index_src) in LAYOUTS {
        for split in [1usize, 2, 3, 5, 17, 32, 33, 41, 64] {
            let mut live = generic(ratios, kv_src, index_src);
            live.forward(&ids(&tokens[..split]), 0).unwrap();
            let layout = live.state_layout(BLOCK);
            let point = live.capture_point(split, &layout).unwrap();
            let nb = split / BLOCK;
            let blocks: Vec<Vec<u8>> = (0..nb)
                .map(|b| live.capture_block(b, &layout).unwrap())
                .collect();
            let manifest = manifest_bytes(&point, &layout);
            let floats = latent_values(&live) * 4;
            assert!(
                manifest.len() < floats / 2 + 4096 + ratios.len() * 8192,
                "{ratios:?} split {split}: {} bytes for {floats} of rows",
                manifest.len()
            );

            if nb > 0 {
                let mut short = generic(ratios, kv_src, index_src);
                let refused = restore_from_bytes(
                    &mut short,
                    &manifest,
                    &blocks[1..],
                    &layout,
                    &tokens[..split],
                );
                assert!(refused.is_err(), "{ratios:?} split {split}: too few blocks");
            }
            let mut restored = generic(ratios, kv_src, index_src);
            restore_from_bytes(&mut restored, &manifest, &blocks, &layout, &tokens[..split])
                .unwrap();

            let rest = &tokens[split..];
            let want = bits(&live.forward(&ids(rest), split).unwrap());
            let got = bits(&restored.forward(&ids(rest), split).unwrap());
            assert_eq!(got, want, "{ratios:?} split {split}: the remainder");
            let steps = [3u32, 9, 11];
            assert_eq!(
                decode(&mut restored, &steps, tokens.len()),
                decode(&mut live, &steps, tokens.len()),
                "{ratios:?} split {split}: decode after the restore"
            );
        }
    }
}

/// A state captured straight after a decode, the way a turn ends, restores and decodes the
/// same bits.
#[test]
fn a_state_left_by_decoding_restores_to_the_bit() {
    let tokens = tokens();
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut live = generic(ratios, kv_src, index_src);
        live.forward(&ids(&tokens[..9]), 0).unwrap();
        decode(&mut live, &tokens[9..20], 9);
        let layout = live.state_layout(BLOCK);
        let point = live.capture_point(20, &layout).unwrap();
        let blocks: Vec<Vec<u8>> = (0..20 / BLOCK)
            .map(|b| live.capture_block(b, &layout).unwrap())
            .collect();
        let mut restored = generic(ratios, kv_src, index_src);
        restore_from_bytes(
            &mut restored,
            &manifest_bytes(&point, &layout),
            &blocks,
            &layout,
            &tokens[..20],
        )
        .unwrap();
        assert_eq!(
            decode(&mut restored, &tokens[20..30], 20),
            decode(&mut live, &tokens[20..30], 20),
            "{ratios:?}"
        );
    }
}

/// The blocks of a state that later grew are byte for byte the earlier ones, so a store keeps
/// each once.
#[test]
fn a_growing_state_keeps_its_blocks() {
    let tokens = tokens();
    for (ratios, kv_src, index_src) in LAYOUTS {
        let mut live = generic(ratios, kv_src, index_src);
        live.forward(&ids(&tokens[..24]), 0).unwrap();
        let layout = live.state_layout(BLOCK);
        let early: Vec<Vec<u8>> = (0..3)
            .map(|b| live.capture_block(b, &layout).unwrap())
            .collect();
        live.forward(&ids(&tokens[24..48]), 24).unwrap();
        for (b, blk) in early.iter().enumerate() {
            assert_eq!(
                &live.capture_block(b, &layout).unwrap(),
                blk,
                "{ratios:?} block {b}"
            );
        }
    }
}

/// A store's block is a whole number of groups at every ratio: derived, it is their least
/// common multiple, and a fixed block that cuts a group is refused.
#[test]
fn a_block_is_a_whole_number_of_groups() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        // Every band layer of the oracle pools at 1 ratio, so the least common multiple is it.
        let granule = ratios.iter().copied().max().unwrap() as usize;
        assert!(ratios.iter().all(|&r| r == 0 || r as usize == granule));
        let identity = state_identity(b"granule");

        let mut derived = generic(ratios, kv_src, index_src);
        let store = Arc::new(StateStore::in_memory(1 << 20, 0));
        assert!(derived.attach_state_cache(store, identity));
        let block = derived.state_cache().unwrap().layout().block_tokens;
        assert_eq!(block, granule, "{ratios:?}");

        let mut cut = generic(ratios, kv_src, index_src);
        let store = Arc::new(StateStore::in_memory(1 << 20, granule + 1));
        assert!(!cut.attach_state_cache(store, identity), "{ratios:?}");
    }
}

/// Through the store, as a served prompt runs: a first prompt kept on its grid and at its
/// end, then a longer prompt restored at the first's end. The remainder and the decode after
/// it give the bits of a twin that read both prompts.
#[test]
fn a_stored_state_restores_through_the_store_and_decodes_the_same_bits() {
    let tokens = tokens();
    let first = &tokens[..41];
    let second = &tokens[..60];
    for (ratios, kv_src, index_src) in LAYOUTS {
        let identity = state_identity(b"store");
        let store = Arc::new(StateStore::in_memory(1 << 26, BLOCK));

        let mut writer = generic(ratios, kv_src, index_src);
        assert!(writer.attach_state_cache(store.clone(), identity));
        let cache = writer.state_cache().unwrap();
        let layout = writer.state_layout(BLOCK);
        let (_, grid) = writer
            .prefill_with_grid(first, 0, BLOCK, BLOCK, &layout)
            .unwrap();
        assert_eq!(
            grid.iter().map(|p| p.pos).collect::<Vec<_>>(),
            vec![8, 16, 24, 32, 40]
        );
        cache.save(&writer, first, grid, None);
        let mut stored = false;
        for _ in 0..600 {
            if store.lengths().contains(&first.len()) {
                stored = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(
            stored,
            "{ratios:?}: no state was stored: {:?}",
            store.lengths()
        );

        let mut restored = generic(ratios, kv_src, index_src);
        assert!(restored.attach_state_cache(store.clone(), identity));
        let covered = restored
            .state_cache()
            .unwrap()
            .restore(&mut restored, second);
        assert_eq!(covered, first.len(), "{ratios:?}");
        let (got, _) = restored
            .prefill_with_grid(second, covered, BLOCK, BLOCK, &layout)
            .unwrap();

        let mut twin = generic(ratios, kv_src, index_src);
        twin.prefill_with_grid(first, 0, BLOCK, BLOCK, &layout)
            .unwrap();
        let (want, _) = twin
            .prefill_with_grid(second, first.len(), BLOCK, BLOCK, &layout)
            .unwrap();
        assert_eq!(bits(&got), bits(&want), "{ratios:?}: the remainder");
        let steps = [5u32, 7, 13];
        assert_eq!(
            decode(&mut restored, &steps, second.len()),
            decode(&mut twin, &steps, second.len()),
            "{ratios:?}: decode after the restore"
        );
    }
}
