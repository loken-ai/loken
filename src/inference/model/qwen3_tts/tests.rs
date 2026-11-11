//! The prompt layout, the sampling and the frame loop on a toy checkpoint, and the talker's
//! reproducibility on the published checkpoint.
//! The comparison to the reference runs outside the test suite.

use super::*;
use crate::inference::generic_transformer::class_load::tests::{scratch, write_speech_toy};
use crate::inference::load::hf_config::tests::{speech_config, speech_snapshot};
use serde_json::{json, Value as Json};

/// Rows of the toy's text table.
const TEXT_VOCAB: usize = 16;

/// The toy configuration with the ids the prompt and the loop read.
fn toy_config() -> Json {
    let mut root = speech_config();
    root["tts_bos_token_id"] = json!(13);
    root["tts_eos_token_id"] = json!(14);
    root["tts_pad_token_id"] = json!(15);
    let t = &mut root["talker_config"];
    t["codec_pad_id"] = json!(30);
    t["codec_bos_id"] = json!(31);
    t["codec_eos_token_id"] = json!(32);
    t["codec_think_id"] = json!(33);
    t["codec_nothink_id"] = json!(34);
    t["codec_think_bos_id"] = json!(35);
    t["codec_think_eos_id"] = json!(36);
    t["codec_language_id"] = json!({"english": 37});
    t["max_position_embeddings"] = json!(64);
    root
}

/// A toy generation configuration that draws.
fn toy_generation() -> Json {
    json!({
        "do_sample": true,
        "repetition_penalty": 1.05,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "subtalker_dosample": true,
        "subtalker_temperature": 0.9,
        "subtalker_top_p": 1.0,
        "subtalker_top_k": 50,
        "max_new_tokens": 8
    })
}

/// The toy checkpoint written to a scratch directory, with its generation configuration and a
/// byte-level tokenizer of 2 pieces and their merge.
fn toy_dir(tag: &str) -> std::path::PathBuf {
    let dir = scratch(tag);
    write_speech_toy(&dir, &toy_config(), TEXT_VOCAB);
    std::fs::write(
        dir.join("generation_config.json"),
        toy_generation().to_string(),
    )
    .unwrap();
    std::fs::write(dir.join("vocab.json"), r#"{"a": 0, "b": 1, "ab": 2}"#).unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\na b\n").unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"tokenizer_class": "Qwen2Tokenizer", "add_bos_token": false}"#,
    )
    .unwrap();
    dir
}

fn ids() -> SpeechIds {
    SpeechIds::read(&toy_config()).unwrap()
}

#[test]
fn the_prompt_lays_out_design_role_controls_text_and_start() {
    let ids = ids();
    let turns = prompt::Turns {
        design: vec![1, 2],
        role: vec![3],
        text: vec![4, 5],
    };
    let text = |t| prompt::Row {
        text: Some(t),
        codec: None,
    };
    let both = |t, c| prompt::Row {
        text: Some(t),
        codec: Some(c),
    };
    let (pad, bos, eos) = (ids.tts_pad, ids.tts_bos, ids.tts_eos);
    let lang = ids.language(Some("English")).unwrap();
    assert_eq!(lang, Some(37));
    let tail = [
        both(4, ids.codec_pad),
        both(5, ids.codec_pad),
        both(eos, ids.codec_pad),
        both(pad, ids.codec_bos),
    ];
    let mut want = vec![text(1), text(2), text(3)];
    want.extend([
        both(pad, ids.codec_think),
        both(pad, ids.codec_think_bos),
        both(pad, 37),
        both(pad, ids.codec_think_eos),
        both(bos, ids.codec_pad),
    ]);
    want.extend(tail);
    assert_eq!(prompt::rows(&ids, &turns, lang), want);
    let mut want = vec![text(1), text(2), text(3)];
    want.extend([
        both(pad, ids.codec_nothink),
        both(pad, ids.codec_think_bos),
        both(pad, ids.codec_think_eos),
        both(bos, ids.codec_pad),
    ]);
    want.extend(tail);
    assert_eq!(prompt::rows(&ids, &turns, None), want);
    assert!(ids.language(Some("klingon")).is_err());
}

#[test]
fn the_suppressed_ids_are_past_the_codebook_but_the_end_of_speech() {
    let ids = ids();
    let want: Vec<u32> = (24..40).filter(|i| *i != 32).collect();
    assert_eq!(ids.suppressed(), want);
}

#[test]
fn the_end_of_speech_waits_for_the_minimum_frames() {
    let ids = ids();
    let mut row = vec![0f32; ids.talker_vocab];
    row[ids.codec_eos as usize] = 5.0;
    row[7] = 1.0;
    for frame in 0..=sampling::MIN_FRAMES_BEFORE_EOS {
        let mut v = row.clone();
        sampling::talker_logits(&mut v, &ids, frame, &[], 1.0);
        let want = match frame < sampling::MIN_FRAMES_BEFORE_EOS {
            true => 7,
            false => ids.codec_eos,
        };
        assert_eq!(sampling::argmax(&v), want, "frame {frame}");
    }
}

#[test]
fn the_repetition_penalty_counts_each_drawn_code_once() {
    let mut v = vec![2.0, -2.0, 4.0, 1.0];
    sampling::penalise(&mut v, &[0, 1, 0, 2, 2], 2.0);
    assert_eq!(v, vec![1.0, -4.0, 2.0, 1.0]);
}

#[test]
fn a_draw_keeps_the_top_k_and_takes_the_most_probable_when_it_does_not_sample() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let v = vec![0.5, 3.0, 2.9, -1.0];
    let s = Sampling {
        sample: false,
        temperature: 0.9,
        top_k: 50,
        top_p: 1.0,
    };
    assert_eq!(sampling::draw(&v, &s, &mut rng).unwrap(), 1);
    let one = Sampling {
        sample: true,
        top_k: 1,
        ..s
    };
    for _ in 0..8 {
        assert_eq!(sampling::draw(&v, &one, &mut rng).unwrap(), 1);
    }
    let two = Sampling {
        sample: true,
        top_k: 2,
        ..s
    };
    for _ in 0..32 {
        assert!([1, 2].contains(&sampling::draw(&v, &two, &mut rng).unwrap()));
    }
}

/// Sub-codes drawn from the run's generator, each within the codebook.
struct Drawn(usize, usize);

impl SubCodes for Drawn {
    fn sub_codes(
        &mut self,
        _past: &Tensor,
        _group0: &Tensor,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Vec<u32>> {
        let uniform = vec![0f32; self.1];
        let s = Sampling {
            sample: true,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        (0..self.0)
            .map(|_| sampling::draw(&uniform, &s, rng))
            .collect()
    }
}

/// The toy prompt of a few text rows.
fn toy_rows(talker: &Talker) -> Tensor {
    let turns = prompt::Turns {
        design: vec![1, 2],
        role: vec![3],
        text: vec![4, 5, 6],
    };
    talker
        .prompt_rows(&prompt::rows(talker.ids(), &turns, None))
        .unwrap()
}

#[test]
fn a_seeded_run_gives_the_same_codes_twice() {
    let dir = toy_dir("seeded");
    let mut talker = Talker::load(&dir, Rows::Exactly(64)).unwrap();
    let ids = talker.ids().clone();
    let rows = toy_rows(&talker);
    let mut sub = Drawn(ids.groups - 1, ids.codebook);
    let a = talker.frames(&rows, 11, &mut sub).unwrap();
    let b = talker.frames(&rows, 11, &mut sub).unwrap();
    assert_eq!(a, b);
    assert!(!a.is_empty());
    for frame in &a {
        assert_eq!(frame.len(), ids.groups);
        assert!(frame.iter().all(|c| (*c as usize) < ids.codebook));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_loop_stops_where_the_cache_is_full() {
    let dir = toy_dir("full");
    let probe = Talker::load(&dir, Rows::Exactly(64)).unwrap();
    let n = toy_rows(&probe).dim(0).unwrap();
    drop(probe);
    let mut talker = Talker::load(&dir, Rows::Exactly(n + 1)).unwrap();
    let rows = toy_rows(&talker);
    let ids = talker.ids().clone();
    let mut sub = Drawn(ids.groups - 1, ids.codebook);
    let frames = talker.frames(&rows, 3, &mut sub).unwrap();
    assert!(frames.len() <= 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A prompt row is the sum of its text part and its codec part, either of which may be absent.
#[test]
fn a_prompt_row_sums_its_text_and_codec_parts() {
    let dir = toy_dir("rowsum");
    let talker = Talker::load(&dir, Rows::Exactly(64)).unwrap();
    let ids = talker.ids().clone();
    let rows = [
        prompt::Row {
            text: Some(4),
            codec: None,
        },
        prompt::Row {
            text: None,
            codec: Some(ids.codec_pad),
        },
        prompt::Row {
            text: Some(5),
            codec: Some(ids.codec_bos),
        },
    ];
    let got = host(&talker.prompt_rows(&rows).unwrap()).unwrap();
    let width = got.len() / rows.len();
    let text = host(&talker.text_rows(&[4, 5]).unwrap()).unwrap();
    let codec = host(&talker.codec_rows(&[ids.codec_pad, ids.codec_bos]).unwrap()).unwrap();
    for i in 0..width {
        assert_eq!(got[i], text[i]);
        assert_eq!(got[width + i], codec[i]);
        assert_eq!(got[2 * width + i], text[width + i] + codec[width + i]);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The row after a frame is the codec row of group 0, the table row of each later group and the
/// pad text row, summed.
/// A frame of the wrong width is refused.
#[test]
fn the_row_after_a_frame_sums_its_code_rows_and_the_pad_row() {
    let dir = toy_dir("frame_input");
    let tts = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let talker = &tts.talker;
    let ids = talker.ids().clone();
    let codes: Vec<u32> = (0..ids.groups as u32)
        .map(|g| (g + 3) % ids.codebook as u32)
        .collect();
    let got = host(&talker.frame_input(&codes).unwrap()).unwrap();
    let mut want = host(&talker.codec_rows(&codes[..1]).unwrap()).unwrap();
    for (table, code) in talker.groups.iter().zip(&codes[1..]) {
        let row = host(&table.rows(&[*code]).unwrap()).unwrap();
        want.iter_mut().zip(&row).for_each(|(a, b)| *a += b);
    }
    let pad = host(&talker.text_rows(&[ids.tts_pad]).unwrap()).unwrap();
    want.iter_mut().zip(&pad).for_each(|(a, b)| *a += b);
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} {b}");
    }
    assert!(talker.frame_input(&codes[1..]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The published checkpoint's directory, `None` when the test store lacks the checkpoint.
fn published() -> Option<std::path::PathBuf> {
    let dir = speech_snapshot();
    if dir.is_none() {
        eprintln!("the published checkpoint is not in the test store");
    }
    dir
}

/// A request of the published checkpoint.
fn published_request() -> Request<'static> {
    Request {
        text: "Hello, this is a short test.",
        design: Some("A calm young woman with a warm, clear voice."),
        language: Some("english"),
        seed: 0,
    }
}

/// The checkpoint's tokenizer loads from its files and wraps both turns.
#[test]
fn the_published_tokenizer_wraps_both_turns() {
    let Some(dir) = published() else { return };
    let tokenizer = crate::inference::engine::llm_engine::build_tokenizer_from_bpe_files(&dir)
        .unwrap_or_else(|e| panic!("{e}"));
    let request = published_request();
    let turns = prompt::Turns::of(&tokenizer, request.text, request.design).unwrap();
    assert!(!turns.design.is_empty());
    assert!(!turns.text.is_empty());
    assert_eq!(turns.role.len(), 3);
    let bare = prompt::Turns::of(&tokenizer, request.text, None).unwrap();
    assert!(bare.design.is_empty());
    assert_eq!((bare.role, bare.text), (turns.role, turns.text));
}

/// The same request and seed give the same frames twice on the published checkpoint, each of a
/// code per group within the codebook.
#[test]
fn the_published_checkpoint_gives_the_same_frames_twice() {
    let Some(dir) = published() else { return };
    let mut tts = Qwen3Tts::load(&dir, Rows::UpTo(1024)).unwrap();
    let request = published_request();
    let rows = tts.talker.request_rows(&request).unwrap();
    assert_eq!(rows.dim(1).unwrap(), tts.talker.text_table.width());
    let overrides = Overrides {
        max_frames: Some(4),
        ..Overrides::default()
    };
    let a = tts
        .utterance(&request, &overrides, &mut |_| Ok(()))
        .unwrap();
    let b = tts
        .utterance(&request, &overrides, &mut |_| Ok(()))
        .unwrap();
    assert_eq!(a, b);
    assert!(!a.frames.is_empty() && a.frames.len() <= 4);
    let ids = tts.talker.ids();
    for frame in &a.frames {
        assert_eq!(frame.len(), ids.groups);
        assert!(frame.iter().all(|c| (*c as usize) < ids.codebook));
    }
}

/// The talker row and the codec row of group 0 a toy frame starts from.
fn toy_frame(tts: &Qwen3Tts, seed: u32, code: u32) -> (Tensor, Tensor) {
    let width = tts.talker.text_table.width();
    let mut seed = seed;
    let past = crate::inference::generic_transformer::class_load::tests::values(
        &mut seed, width, 0.0, 1.0,
    );
    let past = Tensor::from_vec(past, (1, width), &Device::Cpu).unwrap();
    (past, tts.talker.codec_rows(&[code]).unwrap())
}

/// A frame of the code predictor gives the same logits whatever frames ran before it.
#[test]
fn the_code_predictor_restarts_every_frame() {
    let dir = toy_dir("restart");
    let mut tts = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let passes = tts.talker.ids.groups - 1;
    let codes: Vec<u32> = (1..=passes as u32).collect();
    let (past, group0) = toy_frame(&tts, 5, 3);
    let first = tts.predictor.logits_fed(&past, &group0, &codes).unwrap();
    assert_eq!(first.len(), passes);
    assert!(first
        .iter()
        .all(|l| l.len() == tts.talker.ids.codebook && l.iter().all(|v| v.is_finite())));
    let (other, other0) = toy_frame(&tts, 9, 7);
    let reversed: Vec<u32> = codes.iter().rev().copied().collect();
    tts.predictor
        .logits_fed(&other, &other0, &reversed)
        .unwrap();
    let again = tts.predictor.logits_fed(&past, &group0, &codes).unwrap();
    assert_eq!(first, again);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A seeded utterance of the talker and its code predictor gives the same codes twice.
#[test]
fn a_seeded_utterance_gives_the_same_codes_twice() {
    let dir = toy_dir("utterance");
    let mut tts = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let ids = tts.talker.ids.clone();
    let rows = toy_rows(&tts.talker);
    let a = tts.talker.frames(&rows, 21, &mut tts.predictor).unwrap();
    let b = tts.talker.frames(&rows, 21, &mut tts.predictor).unwrap();
    assert_eq!(a, b);
    assert!(!a.is_empty());
    for frame in &a {
        assert_eq!(frame.len(), ids.groups);
        assert!(frame.iter().all(|c| (*c as usize) < ids.codebook));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The sink takes every frame of the loop in order.
/// A sink that fails stops the loop at that frame with its error.
#[test]
fn the_sink_takes_each_frame_and_its_failure_stops_the_loop() {
    let dir = toy_dir("sink");
    let mut tts = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let rows = toy_rows(&tts.talker);
    let mut seen = Vec::new();
    let all = tts
        .talker
        .frames_with(
            &rows,
            21,
            &mut tts.predictor,
            &Overrides::default(),
            &mut |f| {
                seen.push(f.to_vec());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(seen, all.frames);
    assert!(all.frames.len() >= 2);
    let mut taken = 0;
    let stopped = tts
        .talker
        .frames_with(
            &rows,
            21,
            &mut tts.predictor,
            &Overrides::default(),
            &mut |_| {
                taken += 1;
                match taken {
                    2 => Err(Error::msg("the client is gone")),
                    _ => Ok(()),
                }
            },
        )
        .unwrap_err();
    assert!(
        stopped.to_string().contains("the client is gone"),
        "{stopped}"
    );
    assert_eq!(taken, 2);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A streamed text longer than one call's budget is spoken to its end.
/// Its frames are those of the whole-clip path, given to the sink in order.
#[test]
fn a_streamed_long_text_is_spoken_to_its_end() {
    let dir = toy_dir("streamed_long");
    let request = Request {
        text: "ab ab. ab ab. ab ab.",
        design: None,
        language: None,
        seed: 21,
    };
    let probe = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let n = probe.talker.request_rows(&request).unwrap().dim(0).unwrap();
    drop(probe);
    // The whole text leaves the cache room for 2 frames.
    let mut tts = Qwen3Tts::load(&dir, Rows::Exactly(n + 2)).unwrap();
    let overrides = Overrides::default();
    let one = tts
        .utterance(&request, &overrides, &mut |_| Ok(()))
        .unwrap();
    assert_eq!(one.stop, Stop::Budget);
    let whole = tts.speak(&request, &overrides).unwrap();
    let mut seen = Vec::new();
    let streamed = tts
        .speak_streamed(&request, &overrides, &mut |f| {
            seen.push(f.to_vec());
            Ok(())
        })
        .unwrap();
    assert!(whole.len() > one.frames.len());
    assert_eq!(streamed.frames, whole);
    assert_eq!(seen, whole);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A streamed text of several sentences is spoken in 1 call when the cache holds its prompt
/// and the checkpoint's frame limit.
/// Its frames are those of the whole-clip path.
#[test]
fn a_streamed_text_the_cache_holds_is_spoken_in_one_call() {
    let dir = toy_dir("streamed_whole");
    let request = Request {
        text: "ab ab. ab ab. ab ab.",
        design: None,
        language: None,
        seed: 21,
    };
    let probe = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let n = probe.talker.request_rows(&request).unwrap().dim(0).unwrap();
    let limit = probe.talker.generation().max_frames;
    drop(probe);
    let mut tts = Qwen3Tts::load(&dir, Rows::Exactly(n + limit)).unwrap();
    let overrides = Overrides::default();
    let one = tts
        .utterance(&request, &overrides, &mut |_| Ok(()))
        .unwrap();
    assert_eq!(one.stop, Stop::End);
    let whole = tts.speak(&request, &overrides).unwrap();
    let mut seen = Vec::new();
    let streamed = tts
        .speak_streamed(&request, &overrides, &mut |f| {
            seen.push(f.to_vec());
            Ok(())
        })
        .unwrap();
    assert_eq!(streamed, one);
    assert_eq!(seen, one.frames);
    assert_eq!(whole, one.frames);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_variant_other_than_the_voice_design_is_refused_by_name() {
    let mut root = speech_config();
    assert_eq!(config::served_variant(&root).unwrap(), "voice_design");
    root["tts_model_type"] = json!("custom_voice");
    let err = config::served_variant(&root).unwrap_err().to_string();
    assert!(err.contains("custom_voice"), "{err}");
}

#[test]
fn the_blocks_split_the_tensors_of_the_checkpoint() {
    use crate::inference::media_facts::fixture::safetensors;
    let dir = scratch("blocks");
    std::fs::write(dir.join("config.json"), speech_config().to_string()).unwrap();
    safetensors(
        &dir.join("model.safetensors"),
        &[
            "talker.model.norm.weight",
            "talker.codec_head.weight",
            "talker.code_predictor.model.norm.weight",
        ],
        None,
    );
    safetensors(
        &decoder::weights_of(&dir),
        &["decoder.pre_conv.conv.weight", "encoder.conv.weight"],
        None,
    );
    let blocks = Blocks::of(&dir).unwrap();
    assert_eq!(
        blocks,
        Blocks {
            talker: 8,
            predictor: 4,
            decoder: 4
        }
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The bytes the header of the safetensors file at `path` lists for the tensors `keep` takes.
fn header_bytes(path: &std::path::Path, keep: impl Fn(&str) -> bool) -> u64 {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap();
    let mut len = [0u8; 8];
    file.read_exact(&mut len).unwrap();
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    file.read_exact(&mut header).unwrap();
    let header: serde_json::Map<String, Json> = serde_json::from_slice(&header).unwrap();
    header
        .iter()
        .filter(|(name, _)| *name != "__metadata__" && keep(name))
        .map(|(_, info)| {
            let span = &info["data_offsets"];
            span[1].as_u64().unwrap() - span[0].as_u64().unwrap()
        })
        .sum()
}

#[test]
fn the_published_blocks_sum_to_the_tensors_they_load() {
    let Some(dir) = published() else { return };
    let blocks = Blocks::of(&dir).unwrap();
    let weights = header_bytes(&dir.join("model.safetensors"), |_| true);
    let decoder = header_bytes(&decoder::weights_of(&dir), |n| n.starts_with("decoder."));
    eprintln!("blocks of the published checkpoint: {blocks:?}");
    assert_eq!(blocks.total(), weights + decoder);
    assert!(blocks.talker > blocks.predictor && blocks.predictor > 0);
}
