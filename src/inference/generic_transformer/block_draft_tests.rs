//! The block draft of the generic route against the V4.1 draft, on the synthetic model of the
//! frozen V4.1 oracle.
//!
//! The oracle's file carries a draft of 2 stages beside its layers.
//! The draft's keys are the V4.1 writer's, its tensors keep the checkpoint's names.
//! Both paths load from that file: the V4.1 model with its backend and draft, the generic model
//! by the generic loader, which finds the draft in the file.

use super::wide_sparse_oracle_tests::LAYOUTS;
use super::*;
use crate::inference::engine::model_backend::{DeepseekV41Backend, GenericBackend, ModelBackend};
use crate::inference::model::deepseek_v41::dspark::{Dspark, DsparkConfig};
use crate::inference::model::deepseek_v41::model::tests::{
    real_entries_for, real_md_layout, VOCAB,
};
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{open_mapped, MappedGguf};
use crate::tensor::quantized::GgmlDType;

/// Width of a stage's query factor.
const Q_RANK: usize = 32;

/// Width of a stage's experts.
const INTER: usize = 64;

/// The draft the synthetic file carries for a model of hidden width `dim`.
///
/// Its ring is shorter than the prompt, so the prompt wraps it.
pub(crate) fn draft_config(dim: usize) -> DsparkConfig {
    DsparkConfig {
        n_stages: 2,
        block_size: 4,
        noise_token: VOCAB as u32 - 1,
        target_layers: vec![1, 2],
        markov_rank: 8,
        n_routed: 4,
        n_activated: 2,
        window_size: 6,
        dim,
        hc_mult: 4,
        n_heads: 2,
        head_dim: 64,
        rope_head_dim: 16,
        o_groups: 2,
        o_lora_rank: 32,
        gate_temp: 1.0,
        route_scale: 1.5,
        swiglu_limit: 10.0,
        norm_topk: true,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
        rope_factor: 1.0,
        original_seq_len: 0,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        vocab: VOCAB,
    }
}

/// `n` values in `center - amp..center + amp`, drawn from `name` by a 64-bit linear
/// congruential generator.
fn values(n: usize, name: &str, center: f32, amp: f32) -> Vec<f32> {
    let mut s = name.bytes().fold(1469598103934665603u64, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    });
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            center + ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// An F32 tensor `name` of `dims`, rows first.
fn entry(name: String, dims: Vec<usize>, center: f32, amp: f32) -> GgufEntry {
    let n = dims.iter().product();
    let data = values(n, &name, center, amp)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    GgufEntry {
        name,
        dims,
        dtype: GgmlDType::F32,
        data,
    }
}

/// A projection `[out, inp]` whose products keep the scale of their input.
fn matrix(name: String, out: usize, inp: usize) -> GgufEntry {
    entry(name, vec![out, inp], 0.0, (1.0 / inp as f32).sqrt())
}

/// A norm's scale of `n` values near 1.
fn norm(name: String, n: usize) -> GgufEntry {
    entry(name, vec![n], 1.0, 0.1)
}

/// The tensors of the draft `c`, under the checkpoint's names.
pub(crate) fn draft_entries(c: &DsparkConfig) -> Vec<GgufEntry> {
    let (dim, hc) = (c.dim, c.hc_mult);
    let mix = (2 + hc) * hc;
    let heads = c.n_heads * c.head_dim;
    let mut entries = Vec::new();
    for s in 0..c.n_stages {
        let p = |rest: &str| format!("mtp.{s}.{rest}");
        entries.extend([
            matrix(p("attn.wq_a.weight"), Q_RANK, dim),
            norm(p("attn.q_norm.weight"), Q_RANK),
            matrix(p("attn.wq_b.weight"), heads, Q_RANK),
            matrix(p("attn.wkv.weight"), c.head_dim, dim),
            norm(p("attn.kv_norm.weight"), c.head_dim),
            entry(p("attn.attn_sink"), vec![c.n_heads], 0.0, 0.5),
            matrix(
                p("attn.wo_a.weight"),
                c.o_groups * c.o_lora_rank,
                heads / c.o_groups,
            ),
            matrix(p("attn.wo_b.weight"), dim, c.o_groups * c.o_lora_rank),
            norm(p("attn_norm.weight"), dim),
            norm(p("ffn_norm.weight"), dim),
            matrix(p("ffn.gate.weight"), c.n_routed, dim),
            entry(p("ffn.gate.bias"), vec![c.n_routed], 0.0, 0.1),
        ]);
        let mut expert = |stem: String| {
            entries.extend([
                matrix(format!("{stem}.w1.weight"), INTER, dim),
                matrix(format!("{stem}.w3.weight"), INTER, dim),
                matrix(format!("{stem}.w2.weight"), dim, INTER),
            ]);
        };
        expert(p("ffn.shared_experts"));
        for e in 0..c.n_routed {
            expert(p(&format!("ffn.experts.{e}")));
        }
        for part in ["attn", "ffn"] {
            entries.extend([
                matrix(p(&format!("hc_{part}_fn")), mix, hc * dim),
                entry(p(&format!("hc_{part}_scale")), vec![3], 1.0, 0.5),
                entry(p(&format!("hc_{part}_base")), vec![mix], 0.0, 0.5),
            ]);
        }
    }
    let last = c.n_stages - 1;
    entries.extend([
        matrix(
            "mtp.0.main_proj.weight".to_string(),
            dim,
            dim * c.target_layers.len(),
        ),
        norm("mtp.0.main_norm.weight".to_string(), dim),
        norm(format!("mtp.{last}.norm.weight"), dim),
        entry(
            format!("mtp.{last}.markov_head.embed.weight"),
            vec![c.vocab, c.markov_rank],
            0.0,
            1.0,
        ),
        matrix(
            format!("mtp.{last}.markov_head.head.weight"),
            c.vocab,
            c.markov_rank,
        ),
    ]);
    entries
}

/// The synthetic model's file in a band layout, carrying the draft of [`draft_config`] when
/// `with_draft`.
///
/// Mapped twice, so each path takes its own mapping.
pub(crate) fn toy_files(
    (ratios, kv_src, index_src): (&[u32], &[u32], &[u32]),
    with_draft: bool,
) -> (MappedGguf, MappedGguf) {
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let mut md = real_md_layout(ratios, kv_src, index_src);
    let dim = md
        .iter()
        .find(|(k, _)| k.ends_with(".embedding_length"))
        .map(|(_, v)| v.to_u32().unwrap() as usize)
        .unwrap();
    let mut entries = real_entries_for(&bands);
    if with_draft {
        let c = draft_config(dim);
        md.extend(c.to_metadata());
        entries.extend(draft_entries(&c));
    }
    // Unique per call: cargo runs the tests in parallel.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "loken-block-draft-{}-{seq}.gguf",
        std::process::id()
    ));
    write_gguf_with_metadata(&path, &md, &entries).unwrap();
    let files = (open_mapped(&path).unwrap(), open_mapped(&path).unwrap());
    let _ = std::fs::remove_file(&path);
    files
}

/// The synthetic model's file in a band layout without a draft, and beside it a file that carries
/// the draft of [`draft_config`] alone, in a directory of their own.
///
/// Returns the model file's path.
/// The caller removes its directory.
pub(crate) fn toy_with_companion(
    (ratios, kv_src, index_src): (&[u32], &[u32], &[u32]),
) -> std::path::PathBuf {
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let md = real_md_layout(ratios, kv_src, index_src);
    let dim = md
        .iter()
        .find(|(k, _)| k.ends_with(".embedding_length"))
        .map(|(_, v)| v.to_u32().unwrap() as usize)
        .unwrap();
    let c = draft_config(dim);
    // Unique per call: cargo runs the tests in parallel.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "loken-block-draft-companion-{}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    write_gguf_with_metadata(&model, &md, &real_entries_for(&bands)).unwrap();
    write_gguf_with_metadata(
        &dir.join("draft.gguf"),
        &c.to_metadata(),
        &draft_entries(&c),
    )
    .unwrap();
    model
}

/// The generic model of `file`, every layer on the host.
pub(crate) fn generic_of(file: MappedGguf) -> GenericHeteroTransformer {
    let MappedGguf { content, mmap, .. } = file;
    let layers = content
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".block_count"))
        .map(|(_, v)| v.to_u32().unwrap() as usize)
        .unwrap();
    let plan = HeteroPlan::calculate_with_kv(layers, 1 << 20, &[], &[], 1.0, 0);
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan).unwrap()
}

/// The V4.1 model of `file`, every expert resident.
fn v41_of(file: &MappedGguf) -> DeepseekV41Model {
    let mut cfg = DeepseekV41Config::from_gguf(&file.content).unwrap();
    cfg.expert_cache_count = 0;
    DeepseekV41Model::load(file, &cfg).unwrap()
}

/// The prompt of the oracle's synthetic model.
pub(crate) fn prompt() -> Vec<u32> {
    (0..24).map(|i| (i * 7 + 3) % VOCAB as u32).collect()
}

fn ids(tokens: &[u32]) -> Tensor {
    Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap()
}

/// Each row of `vocab` logits of `t`, whatever its rank.
fn rows(t: &Tensor) -> Vec<Vec<f32>> {
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    v.chunks_exact(VOCAB).map(<[f32]>::to_vec).collect()
}

/// The greedy choice: the last of the largest.
fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The rows the capture of `model`'s block draft holds, by position.
fn captured(model: &GenericHeteroTransformer) -> Vec<(usize, Vec<f32>)> {
    model
        .wide_sparse
        .as_ref()
        .and_then(|ws| ws.draft.as_ref())
        .expect("the model carries a block draft")
        .capture
        .rows()
}

/// The generic loader finds the draft the file carries, and a file without one gives a model
/// without one.
#[test]
fn the_generic_model_takes_the_draft_its_file_carries() {
    for layout in LAYOUTS {
        let (file, _) = toy_files(layout, true);
        let model = generic_of(file);
        assert!(model.has_block_draft());
        assert!(GenericBackend(model).capabilities().native_draft);
        let (file, _) = toy_files(layout, false);
        let model = generic_of(file);
        assert!(!model.has_block_draft());
        assert!(!GenericBackend(model).capabilities().native_draft);
    }
}

/// The rows the generic capture holds have the bits of the rows the V4.1 state captures at the
/// draft's target layers: the last window of a prompt, then each decoded token's.
#[test]
fn the_capture_holds_the_bits_the_v41_state_captures() {
    for layout in LAYOUTS {
        let (v41_file, generic_file) = toy_files(layout, true);
        let v41 = v41_of(&v41_file);
        let mut model = generic_of(generic_file);
        let c = DsparkConfig::from_metadata(&v41_file.content.metadata)
            .unwrap()
            .unwrap();
        let mut state = v41.new_decode_state();
        state.capture_layers = c.target_layers.clone();
        state.capture_window = c.window_size;

        let prompt = prompt();
        v41.prefill_into(&prompt, &mut state).unwrap();
        model.forward(&ids(&prompt), 0).unwrap();
        let got = captured(&model);
        let first = prompt.len() - c.window_size;
        assert_eq!(
            got.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            (first..prompt.len()).collect::<Vec<_>>()
        );
        assert_eq!(state.block_hidden.len(), got.len());
        for ((pos, row), want) in got.iter().zip(&state.block_hidden) {
            assert_eq!(bits(row), bits(want), "prompt row {pos}");
        }

        for (j, &t) in [5u32, 9, 13].iter().enumerate() {
            let pos = prompt.len() + j;
            v41.forward_decode(t, &mut state).unwrap();
            model.forward(&ids(&[t]), pos).unwrap();
            let got = captured(&model);
            let (last, row) = got.last().unwrap();
            assert_eq!(*last, pos);
            assert_eq!(
                bits(row),
                bits(&state.captured.concat()),
                "decoded row {pos}"
            );
        }
    }
}

/// The generic draft guesses what the V4.1 draft guesses, round after round, from the same file:
/// after the prompt, after each verify kept whole or cut back, and after a decoded token.
///
/// The verified blocks hold the V4.1 draft's guesses, or tokens the model rejects when it has
/// none.
/// The greedy choice of every verified row is the same on both paths.
#[test]
fn the_generic_draft_guesses_as_the_v41_draft() {
    for layout in LAYOUTS {
        let (v41_file, generic_file) = toy_files(layout, true);
        let draft = Dspark::from_file(&v41_file).unwrap().unwrap();
        let k = draft.cfg.block_size - 1;
        let mut v41 = DeepseekV41Backend::with_draft(v41_of(&v41_file), 0, 128, Some(draft));
        let mut generic = GenericBackend(generic_of(generic_file));
        let vocab = VOCAB as u32;

        let prompt = prompt();
        let want = rows(&v41.forward(&ids(&prompt), 0).unwrap());
        let got = rows(&generic.forward(&ids(&prompt), 0).unwrap());
        let mut next = argmax(want.last().unwrap());
        assert_eq!(argmax(got.last().unwrap()), next);
        let mut pos = prompt.len();
        let (mut asked, mut kept_whole, mut cut) = (0, 0, 0);
        for round in 0..12 {
            let want = v41.native_draft_tokens(next, k);
            let got = generic.native_draft_tokens(next, k);
            assert_eq!(got, want, "round {round}: the guesses");
            asked += usize::from(!want.is_empty());
            assert!(
                generic.native_draft_tokens(next, k).is_empty(),
                "round {round}: a second ask before the cache moves"
            );
            if round % 4 == 3 {
                let want = rows(&v41.forward(&ids(&[next]), pos).unwrap());
                let got = rows(&generic.forward(&ids(&[next]), pos).unwrap());
                let chosen = argmax(want.last().unwrap());
                assert_eq!(argmax(got.last().unwrap()), chosen, "round {round}: decode");
                next = chosen;
                pos += 1;
                continue;
            }
            let mut guess = match want.is_empty() {
                true => vec![(next + 1) % vocab, (next + 2) % vocab],
                false => want,
            };
            // Every 4th round the last guess is wrong, so that a verify is cut back.
            if round % 4 == 1 {
                if let Some(last) = guess.last_mut() {
                    *last = (*last + 1) % vocab;
                }
            }
            let mut block = vec![next];
            block.extend(&guess);
            let want: Vec<u32> = rows(&v41.forward_all(&ids(&block), pos).unwrap())
                .iter()
                .map(|r| argmax(r))
                .collect();
            let got: Vec<u32> = rows(&generic.forward_all(&ids(&block), pos).unwrap())
                .iter()
                .map(|r| argmax(r))
                .collect();
            assert_eq!(got, want, "round {round}: the verified rows");
            let accepted = guess.iter().zip(&want).take_while(|(g, c)| g == c).count();
            let keep = accepted + 1;
            if keep == block.len() {
                kept_whole += 1;
            } else {
                cut += 1;
            }
            v41.trim_kv(pos + keep);
            generic.trim_kv(pos + keep);
            next = want[accepted];
            pos += keep;
        }
        eprintln!("rounds with a guess {asked}, verifies kept whole {kept_whole}, cut back {cut}");
        assert!(cut > 0, "a verify was cut back");
    }
}
