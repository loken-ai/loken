//! The hybrid state cache against a small synthetic hybrid: a short convolution, a global
//! attention layer, a second convolution and a windowed attention layer, on the host.

use super::state_cache::*;
use super::*;
use crate::inference::cache::state_disk::StateStore;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const VOCAB: usize = 96;
const WINDOW: usize = 16;
/// Tokens per block of the store under test.
const BLOCK: usize = 64;
/// Tokens per forward of the prefill under test.
const CHUNK: usize = 128;

fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("loken-hybrid-cache-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Deterministic values in `[-amp, amp]`.
fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// A hybrid file written to `path`: blocks 0 and 2 short convolutions, block 1 global attention,
/// block 3 attention over a window.
fn write_hybrid(path: &std::path::Path) {
    let meta = vec![
        (
            "general.architecture".to_string(),
            Value::String("lfm2".into()),
        ),
        (
            "lfm2.attention.head_count".to_string(),
            Value::U32(HEADS as u32),
        ),
        (
            "lfm2.attention.head_count_kv".to_string(),
            Value::Array(
                [0, KV_HEADS, 0, KV_HEADS]
                    .iter()
                    .map(|&n| Value::U32(n as u32))
                    .collect(),
            ),
        ),
        (
            "lfm2.attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(1e-5),
        ),
        (
            "lfm2.attention.sliding_window".to_string(),
            Value::U32(WINDOW as u32),
        ),
        (
            "lfm2.attention.sliding_window_pattern".to_string(),
            Value::Array([0, 0, 0, 1].iter().map(|&n| Value::U32(n)).collect()),
        ),
        ("lfm2.block_count".to_string(), Value::U32(4)),
        (
            "lfm2.embedding_length".to_string(),
            Value::U32(HIDDEN as u32),
        ),
        (
            "lfm2.feed_forward_length".to_string(),
            Value::U32(2 * HIDDEN as u32),
        ),
        ("lfm2.shortconv.l_cache".to_string(), Value::U32(3)),
        ("lfm2.context_length".to_string(), Value::U32(4096)),
    ];
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("token_embd.weight".into(), vec![VOCAB, HIDDEN]),
        ("token_embd_norm.weight".into(), vec![HIDDEN]),
    ];
    for il in 0..4 {
        tensors.push((format!("blk.{il}.attn_norm.weight"), vec![HIDDEN]));
        if il % 2 == 0 {
            tensors.push((
                format!("blk.{il}.shortconv.in_proj.weight"),
                vec![3 * HIDDEN, HIDDEN],
            ));
            tensors.push((format!("blk.{il}.shortconv.conv.weight"), vec![HIDDEN, 3]));
            tensors.push((
                format!("blk.{il}.shortconv.out_proj.weight"),
                vec![HIDDEN, HIDDEN],
            ));
        } else {
            tensors.push((
                format!("blk.{il}.attn_q.weight"),
                vec![HEADS * HEAD_DIM, HIDDEN],
            ));
            tensors.push((
                format!("blk.{il}.attn_k.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
            ));
            tensors.push((
                format!("blk.{il}.attn_v.weight"),
                vec![KV_HEADS * HEAD_DIM, HIDDEN],
            ));
            tensors.push((format!("blk.{il}.attn_q_norm.weight"), vec![HEAD_DIM]));
            tensors.push((format!("blk.{il}.attn_k_norm.weight"), vec![HEAD_DIM]));
            tensors.push((
                format!("blk.{il}.attn_output.weight"),
                vec![HIDDEN, HEADS * HEAD_DIM],
            ));
        }
        tensors.push((format!("blk.{il}.ffn_norm.weight"), vec![HIDDEN]));
        tensors.push((
            format!("blk.{il}.ffn_gate.weight"),
            vec![2 * HIDDEN, HIDDEN],
        ));
        tensors.push((format!("blk.{il}.ffn_up.weight"), vec![2 * HIDDEN, HIDDEN]));
        tensors.push((
            format!("blk.{il}.ffn_down.weight"),
            vec![HIDDEN, 2 * HIDDEN],
        ));
    }
    let entries: Vec<_> = tensors
        .into_iter()
        .enumerate()
        .map(|(i, (name, dims))| {
            let n: usize = dims.iter().product();
            let vals = if name.ends_with("norm.weight") {
                values(n, i as u64, 0.1)
                    .into_iter()
                    .map(|x| 1.0 + x)
                    .collect()
            } else {
                let fan_in = *dims.last().unwrap() as f32;
                values(n, i as u64 + 1, 1.5 / fan_in.sqrt())
            };
            crate::tensor::gguf_write::GgufEntry {
                name,
                dims,
                dtype: GgmlDType::F32,
                data: crate::tensor::quant_cpu::from_float_bytes(GgmlDType::F32, &vals).unwrap(),
            }
        })
        .collect();
    crate::tensor::gguf_write::write_gguf_with_metadata(path, &meta, &entries).unwrap();
}

fn load(path: &std::path::Path) -> GenericHeteroTransformer {
    let file = std::fs::File::open(path).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
    let content =
        gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone())
            .unwrap();
    let plan = crate::inference::place::layer_executor::HeteroPlan::forced_gpu(4, 0, 0);
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan).unwrap()
}

fn open_store(dir: &std::path::Path) -> Arc<StateStore> {
    Arc::new(StateStore::open(dir.to_path_buf(), 0, BLOCK).unwrap())
}

fn attach(model: &mut GenericHeteroTransformer, store: &Arc<StateStore>, salt: &[u8]) {
    let identity = crate::inference::cache::state_disk::identity(&[b"synthetic hybrid", salt]);
    assert!(model.attach_state_cache(store.clone(), identity));
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7 + i / 5) % VOCAB) as u32).collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn floats(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// Waits for the background write to hold a state of `len` tokens.
fn wait_for_len(store: &StateStore, len: usize) {
    for _ in 0..600 {
        if store.lengths().contains(&len) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("no {len}-token state was stored: {:?}", store.lengths());
}

/// The prefill the serving path runs: from `from`, the grid states kept.
fn prefill(m: &mut GenericHeteroTransformer, tokens: &[u32], from: usize) -> (Tensor, Vec<Point>) {
    let layout = m.state_layout(BLOCK);
    m.prefill_with_grid(tokens, from, CHUNK, CHUNK / BLOCK * BLOCK, &layout)
        .unwrap()
}

/// Decodes `steps` tokens greedily and returns every step's logits bits.
fn decode(
    m: &mut GenericHeteroTransformer,
    first: &Tensor,
    mut pos: usize,
    steps: usize,
) -> Vec<Vec<u32>> {
    let mut out = vec![bits(first)];
    let mut next = argmax(&floats(first));
    for _ in 0..steps {
        let x = Tensor::new(&[next as u32][..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = m.forward(&x, pos).unwrap();
        pos += 1;
        next = argmax(&floats(&logits));
        out.push(bits(&logits));
    }
    out
}

struct Fixture {
    dir: std::path::PathBuf,
    file: std::path::PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let dir = scratch_dir(tag);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("hybrid.gguf");
        write_hybrid(&file);
        Self { dir, file }
    }
    fn store_dir(&self) -> std::path::PathBuf {
        self.dir.join("states")
    }
    fn model(&self) -> GenericHeteroTransformer {
        load(&self.file)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The layer kinds are the ones the file's tensors name.
#[test]
fn the_layer_kinds_are_read_from_the_layers() {
    let fx = Fixture::new("kinds");
    let m = fx.model();
    let kinds = m.state_layout(BLOCK).kinds;
    assert_eq!(
        kinds,
        vec![
            StateKind::Mixer,
            StateKind::Global,
            StateKind::Mixer,
            StateKind::Window
        ]
    );
}

/// A model with no recurrent state, or none attached, takes no part.
#[test]
fn a_state_cache_is_refused_without_a_recurrent_state() {
    let fx = Fixture::new("refused");
    let store = open_store(&fx.store_dir());
    let mut m = fx.model();
    assert!(m.state_cache().is_none());
    attach(&mut m, &store, b"");
    assert!(m.state_cache().is_some());
}

/// A state restored from disk continues exactly as the model that wrote it: the logits of the
/// remainder and every decode step after it are the same bits, at a split on the grid and at one
/// off it (the end of the prompt a conversation's next turn extends).
#[test]
fn a_restored_state_continues_bit_for_bit_like_its_live_twin() {
    let fx = Fixture::new("twin");
    let store = open_store(&fx.store_dir());
    let stride = CHUNK / BLOCK * BLOCK;

    // A first turn of 3 strides and a few tokens, written by a cold prefill.
    let first = prompt(3 * stride + 37);
    let mut writer = fx.model();
    attach(&mut writer, &store, b"");
    let cache = writer.state_cache().unwrap();
    let (_, grid) = prefill(&mut writer, &first, 0);
    assert_eq!(
        grid.iter().map(|p| p.pos).collect::<Vec<_>>(),
        vec![stride, 2 * stride, 3 * stride]
    );
    cache.save(&writer, &first, grid, None);
    for len in [stride, 2 * stride, 3 * stride, first.len()] {
        wait_for_len(&store, len);
    }

    // The second turn extends the first. It restores at the end of the first, off the grid.
    let mut second = first.clone();
    second.extend(prompt(90).iter().map(|t| (t + 3) % VOCAB as u32));
    let mut restored = fx.model();
    attach(&mut restored, &store, b"");
    let covered = restored
        .state_cache()
        .unwrap()
        .restore(&mut restored, &second);
    assert_eq!(
        covered,
        first.len(),
        "the longest stored state this prompt extends"
    );
    let (logits_r, _) = prefill(&mut restored, &second, covered);

    let mut twin = fx.model();
    let _ = prefill(&mut twin, &first, 0);
    let (logits_t, _) = prefill(&mut twin, &second, first.len());
    assert_eq!(
        bits(&logits_r),
        bits(&logits_t),
        "logits after the remainder"
    );
    assert_eq!(
        decode(&mut restored, &logits_r, second.len(), 6),
        decode(&mut twin, &logits_t, second.len(), 6),
        "decode after a restore off the grid"
    );

    // A prompt that shares only the first two strides restores on the grid.
    let mut branch = first[..2 * stride + 5].to_vec();
    branch.extend(prompt(70).iter().map(|t| (t + 11) % VOCAB as u32));
    let mut on_grid = fx.model();
    attach(&mut on_grid, &store, b"");
    let covered = on_grid
        .state_cache()
        .unwrap()
        .restore(&mut on_grid, &branch);
    assert_eq!(covered, 2 * stride);
    let (logits_g, _) = prefill(&mut on_grid, &branch, covered);
    let mut twin = fx.model();
    let _ = prefill(&mut twin, &branch[..covered], 0);
    let (logits_t, _) = prefill(&mut twin, &branch, covered);
    assert_eq!(
        bits(&logits_g),
        bits(&logits_t),
        "logits after a restore on the grid"
    );
    assert_eq!(
        decode(&mut on_grid, &logits_g, branch.len(), 6),
        decode(&mut twin, &logits_t, branch.len(), 6),
        "decode after a restore on the grid"
    );
}

/// A restored state plus the remainder is a cold prefill to rounding, with the same argmax.
#[test]
fn a_restored_state_and_its_remainder_equal_a_cold_prefill() {
    let fx = Fixture::new("cold");
    let store = open_store(&fx.store_dir());
    let first = prompt(300);
    let mut writer = fx.model();
    attach(&mut writer, &store, b"");
    let cache = writer.state_cache().unwrap();
    let (_, grid) = prefill(&mut writer, &first, 0);
    cache.save(&writer, &first, grid, None);
    wait_for_len(&store, first.len());

    let mut second = first.clone();
    second.extend(prompt(40).iter().map(|t| (t + 5) % VOCAB as u32));
    let mut restored = fx.model();
    attach(&mut restored, &store, b"");
    let covered = restored
        .state_cache()
        .unwrap()
        .restore(&mut restored, &second);
    assert!(covered > 0);
    let (warm, _) = prefill(&mut restored, &second, covered);

    // A cold prefill with no cache and another chunking.
    let mut cold = fx.model();
    let mut last = None;
    for (i, c) in second.chunks(100).enumerate() {
        let x = Tensor::new(c, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        last = Some(cold.forward(&x, i * 100).unwrap());
    }
    let (warm, cold) = (floats(&warm), floats(&last.unwrap()));
    let worst = warm
        .iter()
        .zip(&cold)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = cold.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(
        worst <= 1e-3 * scale.max(1.0),
        "restored and cold differ by {worst} (scale {scale})"
    );
    assert_eq!(argmax(&warm), argmax(&cold));
}

/// A damaged, foreign or other-version file is refused and dropped, and the model still answers.
#[test]
fn a_damaged_foreign_or_stale_file_is_refused() {
    let fx = Fixture::new("refuse");
    let first = prompt(200);
    let mut second = first.clone();
    second.extend(prompt(30));

    let write = |dir: &std::path::Path| {
        let store = open_store(dir);
        let mut m = fx.model();
        attach(&mut m, &store, b"");
        let cache = m.state_cache().unwrap();
        let (_, grid) = prefill(&mut m, &first, 0);
        cache.save(&m, &first, grid, None);
        wait_for_len(&store, first.len());
    };

    let files = |dir: &std::path::Path, ext: &str, sub: &str| -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir.join(sub))
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == ext))
            .collect()
    };

    // The same bytes under another salt are another scope: nothing restores.
    let dir = fx.store_dir();
    write(&dir);
    {
        let store = open_store(&dir);
        let mut m = fx.model();
        attach(&mut m, &store, b"another salt");
        assert_eq!(m.state_cache().unwrap().restore(&mut m, &second), 0);
    }

    type Damage = (&'static str, fn(&[u8]) -> Vec<u8>);
    let damages: [Damage; 4] = [
        ("truncated", |b| b[..b.len() / 2].to_vec()),
        ("empty", |_| Vec::new()),
        ("flipped", |b| {
            let mut v = b.to_vec();
            let mid = v.len() / 2;
            v[mid] ^= 0x55;
            v
        }),
        ("other version", |b| {
            // A file of another format version with a valid checksum.
            let mut v = b[..b.len() - 16].to_vec();
            v[8..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
            let sum = {
                use sha2::{Digest, Sha256};
                Sha256::digest(&v)[..16].to_vec()
            };
            v.extend(sum);
            v
        }),
    ];
    for (target, ext, sub) in [("manifest", "snap", ""), ("block", "blk", "blocks")] {
        for (what, damage) in &damages {
            let case = fx.dir.join(format!("{target}-{what}").replace(' ', "-"));
            write(&case);
            // Every file of the kind is damaged: a damaged state falls back to a shorter intact
            // one when there is one, so the case leaves none.
            for victim in files(&case, ext, sub) {
                let bytes = std::fs::read(&victim).unwrap();
                std::fs::write(&victim, damage(&bytes)).unwrap();
            }

            let store = open_store(&case);
            let mut m = fx.model();
            attach(&mut m, &store, b"");
            let covered = m.state_cache().unwrap().restore(&mut m, &second);
            // A damaged manifest or block is never used; a damaged block shared by the manifests
            // that name it takes them all down.
            assert_eq!(covered, 0, "{target} {what}: a damaged state was restored");
            // The model is clean and answers like a model that never saw the file.
            let (logits, _) = prefill(&mut m, &second[..80], 0);
            let mut fresh = fx.model();
            let (expected, _) = prefill(&mut fresh, &second[..80], 0);
            assert_eq!(
                bits(&logits),
                bits(&expected),
                "{target} {what}: the model is not clean"
            );
        }
    }
}

/// The packing is exact: values an f16 holds go in f16, any other value keeps its f32 bits.
#[test]
fn rows_round_trip_bit_for_bit() {
    let exact = [0.0f32, -0.0, 1.5, -2.25, 65504.0, 6.1035156e-5];
    let others = [0.1f32, 1.0e-9, f32::MAX, f32::NAN, f32::INFINITY, -3.3];
    for rows in [
        &exact[..],
        &others[..],
        &[&exact[..], &others[..]].concat()[..],
    ] {
        let block = vec![(rows.to_vec(), rows.to_vec())];
        let layout = Layout {
            kinds: vec![StateKind::Global],
            geometry: vec![(1, rows.len())],
            block_tokens: 1,
        };
        let back = decode_block_for_test(&block, &layout);
        for (a, b) in rows.iter().zip(&back.0) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

fn decode_block_for_test(block: &[(Vec<f32>, Vec<f32>)], layout: &Layout) -> (Vec<f32>, Vec<f32>) {
    block_round_trip(block, layout).remove(0)
}

/// With no block size fixed, the model settles one from the disk it writes to and finds the same
/// one after a restart; a state stored under it is restored by the next run.
#[test]
fn a_derived_block_size_is_the_same_after_a_restart() {
    let fx = Fixture::new("derived");
    let dir = fx.store_dir();
    let (block, tokens) = {
        let store = Arc::new(StateStore::open(dir.clone(), 0, 0).unwrap());
        let mut m = fx.model();
        attach(&mut m, &store, b"");
        let cache = m.state_cache().unwrap();
        let block = cache.layout().block_tokens;
        assert!(block >= 1);
        assert!(dir.join("geometry.json").exists());
        // A prompt of two blocks and a little, as far as the model's own context allows.
        let tokens = prompt((2 * block + 10).min(3000));
        if tokens.len() >= 2 * block {
            let layout = m.state_layout(block);
            let stride = block.max(CHUNK);
            let (_, grid) = m
                .prefill_with_grid(&tokens, 0, CHUNK, stride, &layout)
                .unwrap();
            cache.save(&m, &tokens, grid, None);
            wait_for_len(&store, tokens.len());
        }
        (block, tokens)
    };
    let store = Arc::new(StateStore::open(dir, 0, 0).unwrap());
    let mut m = fx.model();
    attach(&mut m, &store, b"");
    assert_eq!(m.state_cache().unwrap().layout().block_tokens, block);
    if tokens.len() >= 2 * block {
        let mut longer = tokens.clone();
        longer.extend(prompt(20));
        assert_eq!(
            m.state_cache().unwrap().restore(&mut m, &longer),
            tokens.len()
        );
    }
}

/// A state handed over as bytes and imported by a second instance resumes with the logits and
/// every decode step of the instance that read the prompt, the prompt being longer than the
/// window of the windowed layer.
#[test]
fn a_handed_over_state_resumes_like_the_instance_that_read_it() {
    let fx = Fixture::new("handover");
    let tokens = prompt(3 * WINDOW + 5);
    let mut reader = fx.model();
    assert!(reader.carries_handover_state());
    let (logits, _) = prefill(&mut reader, &tokens, 0);
    reader.snapshot_prefix(&tokens, &logits).unwrap();
    let index = reader.state_snapshot_covering(&tokens).unwrap();
    let bytes = reader.export_state(index).unwrap();

    let mut decoder = fx.model();
    decoder.import_state(tokens.clone(), &bytes).unwrap();
    let (covered, resumed) = decoder.try_restore_prefix(&tokens).unwrap().unwrap();
    assert_eq!(covered, tokens.len());
    assert_eq!(
        bits(&resumed),
        bits(&logits),
        "logits of the handed-over state"
    );
    assert_eq!(
        decode(&mut decoder, &resumed, tokens.len(), 8),
        decode(&mut reader, &logits, tokens.len(), 8),
        "decode after the handover"
    );
}

/// A handover read for other tokens, damaged, or without the logits a recurrent state resumes
/// from is refused and leaves no snapshot.
#[test]
fn a_mismatched_state_handover_is_refused() {
    let fx = Fixture::new("handover-refused");
    let tokens = prompt(40);
    let mut reader = fx.model();
    let (logits, _) = prefill(&mut reader, &tokens, 0);
    reader.snapshot_prefix(&tokens, &logits).unwrap();
    let index = reader.state_snapshot_covering(&tokens).unwrap();
    let bytes = reader.export_state(index).unwrap();

    let mut decoder = fx.model();
    assert!(decoder.import_state(tokens[..39].to_vec(), &bytes).is_err());
    let mut damaged = bytes.clone();
    damaged[bytes.len() / 2] ^= 1;
    assert!(decoder.import_state(tokens.clone(), &damaged).is_err());
    assert!(decoder
        .import_state(tokens.clone(), &bytes[..bytes.len() - 1])
        .is_err());
    assert!(decoder.import_state(Vec::new(), &bytes).is_err());
    assert!(decoder.try_restore_prefix(&tokens).unwrap().is_none());
}

/// Forwards `tokens` from `from` in chunks of `chunk`, returning the last logits.
fn forward_chunks(
    m: &mut GenericHeteroTransformer,
    tokens: &[u32],
    from: usize,
    chunk: usize,
) -> Tensor {
    let mut last = None;
    let mut pos = from;
    while pos < tokens.len() {
        let n = chunk.min(tokens.len() - pos);
        let x = Tensor::new(&tokens[pos..pos + n], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        last = Some(m.forward(&x, pos).unwrap());
        pos += n;
    }
    last.unwrap()
}

/// The head's pair count, cache length and guesses one decode step after `prompt`.
fn head_after(
    m: &mut GenericHeteroTransformer,
    logits: &Tensor,
    at: usize,
) -> (usize, usize, Vec<u32>) {
    let d = m.draft.as_ref().unwrap();
    let (committed, len) = (d.committed, d.len);
    let next = argmax(&floats(logits)) as u32;
    let x = Tensor::new(&[next], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    m.forward(&x, at).unwrap();
    (committed, len, m.draft_tokens(&[next], 3, 0.0).unwrap())
}

#[test]
fn a_restored_draft_head_drafts_like_one_that_read_the_prompt() {
    let chunk = 5;
    let mut whole = super::wide_tests::toy_with_head();
    whole.set_capture_feature(true);
    let vocab = whole.config.vocab_size as u32;
    let tokens: Vec<u32> = (0..30u32).map(|i| (i * 7 + 2) % vocab).collect();
    // On a chunk boundary: every chunk of the rest is more than a token, which the head reads.
    let cut = 20;
    let logits = forward_chunks(&mut whole, &tokens, 0, chunk);
    let want = head_after(&mut whole, &logits, tokens.len());
    assert_eq!((want.0, want.1), (tokens.len() - 1, tokens.len() - 1));
    assert_eq!(want.2.len(), 3);

    // The state a store keeps at the cut, written and read again.
    let mut writer = super::wide_tests::toy_with_head();
    writer.set_capture_feature(true);
    forward_chunks(&mut writer, &tokens[..cut], 0, chunk);
    let held = writer
        .capture_draft(cut)
        .unwrap()
        .expect("the head holds the cut's pairs");
    let held = draft_round_trip(&held);

    // A twin whose target read the prefix without the head: the head disagrees with the
    // target and is not captured, then takes the stored state and reads the rest.
    let mut restored = super::wide_tests::toy_with_head();
    forward_chunks(&mut restored, &tokens[..cut], 0, chunk);
    assert!(restored.capture_draft(cut).unwrap().is_none());
    restored.draft_reset();
    restored.restore_draft(Some(&held)).unwrap();
    let d = restored.draft.as_ref().unwrap();
    assert_eq!((d.committed, d.len), (cut - 1, cut - 1));
    restored.set_capture_feature(true);
    let logits = forward_chunks(&mut restored, &tokens, cut, chunk);
    assert_eq!(head_after(&mut restored, &logits, tokens.len()), want);
}
