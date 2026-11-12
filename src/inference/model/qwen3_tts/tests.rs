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
        speaker: None,
        language: Some("english"),
        seed: 0,
        clone: None,
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
        speaker: None,
        language: None,
        seed: 21,
        clone: None,
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
        speaker: None,
        language: None,
        seed: 21,
        clone: None,
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

/// The toy request of 1 text token with `seed`.
fn one_token_request(seed: u64) -> Request<'static> {
    Request {
        text: "ab",
        design: None,
        speaker: None,
        language: None,
        seed,
        clone: None,
    }
}

/// The toy loaded, its median pace set to 1 frame per token for the language and the length
/// of [`one_token_request`], so that an utterance of more than 5 frames overruns.
/// The median holds enough paces to stay at 1 through the tests' own utterances.
fn toy_with_a_slow_median(tag: &str) -> (std::path::PathBuf, Qwen3Tts, Prompt) {
    let dir = toy_dir(tag);
    let mut tts = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let prompt = tts.talker.request_prompt(&one_token_request(0)).unwrap();
    assert_eq!(prompt.text_tokens, 1);
    for _ in 0..64 {
        tts.paces.record(prompt.language, 1, 1);
    }
    (dir, tts, prompt)
}

/// A whole clip that overruns is drawn again with the next seed.
/// The second draw is kept when it does not overrun, else the shorter of the 2.
/// The same request twice gives the same frames.
#[test]
fn a_whole_clip_that_overruns_is_drawn_again_the_same_way_twice() {
    let (dir, mut tts, prompt) = toy_with_a_slow_median("overrun_whole");
    let overrides = Overrides::default();
    let raw = |tts: &mut Qwen3Tts, seed: u64| {
        tts.utterance(&one_token_request(seed), &overrides, &mut |_| Ok(()))
            .unwrap()
    };
    let overruns = |frames: usize| frames > 5 * prompt.text_tokens;
    let flagged: Vec<u64> = (0..64)
        .filter(|s| overruns(raw(&mut tts, *s).frames.len()))
        .take(4)
        .collect();
    assert!(!flagged.is_empty(), "no seed of the toy overruns");
    for seed in flagged {
        let (first, second) = (raw(&mut tts, seed), raw(&mut tts, seed.wrapping_add(1)));
        let want = match overruns(second.frames.len()) {
            true if second.frames.len() >= first.frames.len() => first.frames,
            _ => second.frames,
        };
        let median = tts.paces.median(prompt.language, prompt.text_tokens);
        let once = tts.speak(&one_token_request(seed), &overrides).unwrap();
        let twice = tts.speak(&one_token_request(seed), &overrides).unwrap();
        assert_eq!(once, want, "seed {seed}");
        assert_eq!(twice, once, "seed {seed}");
        assert_eq!(
            tts.paces.median(prompt.language, prompt.text_tokens),
            median,
            "the kept draws leave a median of 1"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A streamed clip that overruns is not drawn again and leaves the median as it was.
#[test]
fn a_streamed_clip_is_never_drawn_again() {
    let (dir, mut tts, prompt) = toy_with_a_slow_median("overrun_streamed");
    let overrides = Overrides::default();
    let seed = (0..64)
        .find(|s| {
            tts.utterance(&one_token_request(*s), &overrides, &mut |_| Ok(()))
                .unwrap()
                .frames
                .len()
                > 5
        })
        .expect("a seed of the toy overruns");
    let request = one_token_request(seed);
    let one = tts
        .utterance(&request, &overrides, &mut |_| Ok(()))
        .unwrap();
    let median = tts.paces.median(prompt.language, prompt.text_tokens);
    let streamed = tts
        .speak_streamed(&request, &overrides, &mut |_| Ok(()))
        .unwrap();
    assert_eq!(streamed, one);
    assert_eq!(
        tts.paces.median(prompt.language, prompt.text_tokens),
        median
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_variant_this_build_does_not_serve_is_refused_by_name() {
    let mut root = speech_config();
    assert_eq!(config::served_variant(&root).unwrap(), Variant::VoiceDesign);
    root["tts_model_type"] = json!("base");
    assert_eq!(config::served_variant(&root).unwrap(), Variant::Base);
    root["tts_model_type"] = json!("custom_voice");
    assert_eq!(config::served_variant(&root).unwrap(), Variant::CustomVoice);
    root["tts_model_type"] = json!("other");
    let err = config::served_variant(&root).unwrap_err().to_string();
    assert!(
        err.contains("other") && err.contains("custom_voice"),
        "{err}"
    );
}

/// The toy configuration of a CustomVoice checkpoint: 2 named speakers, one of a dialect.
fn named_config() -> Json {
    let mut root = toy_config();
    root["tts_model_type"] = json!("custom_voice");
    let t = &mut root["talker_config"];
    t["codec_language_id"] = json!({"english": 37, "chinese": 28, "toy_dialect": 29});
    t["spk_id"] = json!({"Alice": 38, "bob": 39});
    t["spk_is_dialect"] = json!({"Alice": false, "bob": "toy_dialect"});
    root
}

/// The speakers are read from the configuration, found without case, and an unknown name is
/// refused with the names the checkpoint has.
/// A dialect speaker speaks its dialect when the language is left to the model or is the one
/// its dialect belongs to, and another language when asked.
#[test]
fn a_named_speaker_and_its_dialect_come_from_the_configuration() {
    let ids = SpeechIds::read(&named_config()).unwrap();
    let names: Vec<&str> = ids.speakers.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["Alice", "bob"]);
    let alice = ids.speaker("alice").unwrap();
    assert_eq!((alice.codec, alice.dialect.as_deref()), (38, None));
    let bob = ids.speaker(" BOB ").unwrap();
    assert_eq!(
        (bob.codec, bob.dialect.as_deref()),
        (39, Some("toy_dialect"))
    );
    let err = ids.speaker("carol").unwrap_err().to_string();
    assert!(err.contains("carol") && err.contains("Alice, bob"), "{err}");
    let language = |name: Option<&str>, speaker: &config::NamedSpeaker| {
        ids.speaker_language(name, speaker).unwrap()
    };
    assert_eq!(language(None, alice), None);
    assert_eq!(language(Some("chinese"), alice), Some(28));
    assert_eq!(language(None, bob), Some(29));
    assert_eq!(language(Some("chinese"), bob), Some(29));
    assert_eq!(language(Some("english"), bob), Some(37));
    assert!(ids.speaker_language(Some("klingon"), bob).is_err());
    assert!(SpeechIds::read(&toy_config()).unwrap().speakers.is_empty());
}

/// A named prompt: the style turn, the role, the controls and the speaker's codec row beside
/// the pad text row, the start of text, the text and its end, then the start of codes.
#[test]
fn the_named_prompt_puts_the_style_then_the_speaker_row_before_the_start_of_text() {
    let ids = SpeechIds::read(&named_config()).unwrap();
    let text = |t| prompt::Row {
        text: Some(t),
        codec: None,
    };
    let both = |t, c| prompt::Row {
        text: Some(t),
        codec: Some(c),
    };
    let (pad, bos, eos) = (ids.tts_pad, ids.tts_bos, ids.tts_eos);
    let plain = prompt::Turns {
        design: vec![],
        role: vec![3],
        text: vec![4, 5],
    };
    let tail = [
        both(bos, ids.codec_pad),
        both(4, ids.codec_pad),
        both(5, ids.codec_pad),
        both(eos, ids.codec_pad),
        both(pad, ids.codec_bos),
    ];
    let mut want = vec![
        text(3),
        both(pad, ids.codec_nothink),
        both(pad, ids.codec_think_bos),
        both(pad, ids.codec_think_eos),
        both(pad, 38),
    ];
    want.extend(tail);
    assert_eq!(prompt::named_rows(&ids, &plain, None, 38), want);
    let mut want = vec![
        text(3),
        both(pad, ids.codec_think),
        both(pad, ids.codec_think_bos),
        both(pad, 29),
        both(pad, ids.codec_think_eos),
        both(pad, 39),
    ];
    want.extend(tail);
    assert_eq!(prompt::named_rows(&ids, &plain, Some(29), 39), want);
    let styled = prompt::Turns {
        design: vec![1, 2],
        ..plain.clone()
    };
    let got = prompt::named_rows(&ids, &styled, None, 38);
    assert_eq!(got[..2], [text(1), text(2)]);
    assert_eq!(got[2..], prompt::named_rows(&ids, &plain, None, 38)[..]);
}

/// A CustomVoice checkpoint speaks a speaker it names, with or without a style.
/// It refuses a request without a speaker, with an unknown one, or with a clone prompt.
/// Another variant refuses a named speaker.
#[test]
fn a_custom_voice_checkpoint_speaks_a_speaker_it_names() {
    let dir = toy_dir("named");
    std::fs::write(dir.join("config.json"), named_config().to_string()).unwrap();
    let talker = Talker::load(&dir, Rows::Exactly(64)).unwrap();
    assert_eq!(talker.variant(), Variant::CustomVoice);
    let request = Request {
        text: "ab",
        design: None,
        speaker: Some("BOB"),
        language: None,
        seed: 0,
        clone: None,
    };
    let rows = talker.request_rows(&request).unwrap();
    let ids = talker.ids().clone();
    let turns = prompt::Turns::of(&talker.tokenizer, "ab", None).unwrap();
    let want = talker
        .prompt_rows(&prompt::named_rows(&ids, &turns, Some(29), 39))
        .unwrap();
    assert_eq!(host(&rows).unwrap(), host(&want).unwrap());
    let styled = Request {
        design: Some("ab"),
        ..request
    };
    let styled_rows = talker.request_rows(&styled).unwrap();
    assert!(styled_rows.dim(0).unwrap() > rows.dim(0).unwrap());
    let nameless = Request {
        speaker: None,
        ..request
    };
    let err = talker.request_rows(&nameless).unwrap_err().to_string();
    assert!(err.contains("Alice, bob"), "{err}");
    let unknown = Request {
        speaker: Some("carol"),
        ..request
    };
    assert!(talker.request_rows(&unknown).is_err());
    let clone = ClonePrompt {
        speaker: vec![0.0; talker.text_table.width()],
        reference: None,
    };
    let cloned = Request {
        clone: Some(&clone),
        ..request
    };
    let err = talker.request_rows(&cloned).unwrap_err().to_string();
    assert_eq!(err, CUSTOM_VOICE_TAKES_NO_CLONE);
    let _ = std::fs::remove_dir_all(&dir);
    let dir = toy_dir("named_on_design");
    let design = Talker::load(&dir, Rows::Exactly(64)).unwrap();
    assert!(design.request_rows(&request).is_err());
    let _ = std::fs::remove_dir_all(&dir);
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
            "speaker_encoder.fc.weight",
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
            decoder: 4,
            speaker: 4
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

/// A clone prompt without a reference: the role, the controls and the speaker row beside the
/// pad text row, the start of text, the text and its end, then the start of codes.
#[test]
fn the_speaker_only_clone_prompt_puts_the_speaker_row_before_the_start_of_text() {
    use prompt::CloneRow::{Frame, Ids, Speaker};
    let ids = ids();
    let turns = prompt::Turns {
        design: vec![],
        role: vec![3],
        text: vec![4, 5],
    };
    let both = |t, c| {
        Ids(prompt::Row {
            text: Some(t),
            codec: Some(c),
        })
    };
    let (pad, bos, eos) = (ids.tts_pad, ids.tts_bos, ids.tts_eos);
    let role = Ids(prompt::Row {
        text: Some(3),
        codec: None,
    });
    let want = vec![
        role,
        both(pad, ids.codec_think),
        both(pad, ids.codec_think_bos),
        both(pad, 37),
        both(pad, ids.codec_think_eos),
        Speaker,
        both(bos, ids.codec_pad),
        both(4, ids.codec_pad),
        both(5, ids.codec_pad),
        both(eos, ids.codec_pad),
        both(pad, ids.codec_bos),
    ];
    assert_eq!(prompt::clone_rows(&ids, &turns, Some(37), None), want);
    let got = prompt::clone_rows(&ids, &turns, None, None);
    assert_eq!(got[1], both(pad, ids.codec_nothink));
    assert_eq!(got[4], Speaker);
    assert!(!got.iter().any(|r| matches!(r, Frame(_))));
}

/// A clone prompt with a reference: the reference text before the text, then the start of
/// codes and each reference frame beside the pad text row.
#[test]
fn the_in_context_clone_prompt_puts_the_reference_text_then_its_frames() {
    use prompt::CloneRow::{Frame, Ids, Speaker};
    let ids = ids();
    let turns = prompt::Turns {
        design: vec![],
        role: vec![3],
        text: vec![4, 5],
    };
    let both = |t, c| {
        Ids(prompt::Row {
            text: Some(t),
            codec: Some(c),
        })
    };
    let (pad, bos, eos) = (ids.tts_pad, ids.tts_bos, ids.tts_eos);
    let got = prompt::clone_rows(&ids, &turns, None, Some((&[8, 9], 3)));
    let mut want = vec![Ids(prompt::Row {
        text: Some(3),
        codec: None,
    })];
    want.extend([
        both(pad, ids.codec_nothink),
        both(pad, ids.codec_think_bos),
        both(pad, ids.codec_think_eos),
        Speaker,
        both(bos, ids.codec_pad),
    ]);
    for t in [8, 9, 4, 5, eos] {
        want.push(both(t, ids.codec_pad));
    }
    want.push(both(pad, ids.codec_bos));
    want.extend([Frame(0), Frame(1), Frame(2)]);
    assert_eq!(got, want);
}

/// The toy checkpoint written as a Base checkpoint.
fn toy_base_dir(tag: &str) -> std::path::PathBuf {
    let dir = toy_dir(tag);
    let mut root: Json =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    root["tts_model_type"] = json!("base");
    std::fs::write(dir.join("config.json"), root.to_string()).unwrap();
    dir
}

/// A speaker row sums the pad text row and the embedding.
/// A frame row is the row that follows the reference frame.
/// A Base checkpoint refuses a request without a clone prompt or with a design text.
#[test]
fn a_clone_prompt_row_sums_the_pad_row_and_the_speaker_or_the_frame() {
    let dir = toy_base_dir("clone_rows");
    let tts = Qwen3Tts::load(&dir, Rows::Exactly(64)).unwrap();
    let talker = &tts.talker;
    assert_eq!(talker.variant(), Variant::Base);
    let ids = talker.ids().clone();
    let width = talker.text_table.width();
    let speaker: Vec<f32> = (0..width).map(|i| i as f32 / width as f32).collect();
    let frame: Vec<u32> = (0..ids.groups as u32)
        .map(|g| (g + 1) % ids.codebook as u32)
        .collect();
    let rows = [prompt::CloneRow::Speaker, prompt::CloneRow::Frame(0)];
    let got = host(
        &talker
            .clone_prompt_rows(&rows, &speaker, std::slice::from_ref(&frame))
            .unwrap(),
    )
    .unwrap();
    let pad = host(&talker.text_rows(&[ids.tts_pad]).unwrap()).unwrap();
    let next = host(&talker.frame_input(&frame).unwrap()).unwrap();
    for i in 0..width {
        assert_eq!(got[i], pad[i] + speaker[i]);
        assert_eq!(got[width + i], next[i]);
    }
    assert!(talker
        .clone_prompt_rows(&rows, &speaker[1..], std::slice::from_ref(&frame))
        .is_err());
    assert!(talker.clone_prompt_rows(&rows, &speaker, &[]).is_err());
    let clone = ClonePrompt {
        speaker,
        reference: None,
    };
    let request = Request {
        text: "ab",
        design: None,
        speaker: None,
        language: None,
        seed: 0,
        clone: None,
    };
    let refused = talker.request_rows(&request).unwrap_err().to_string();
    assert_eq!(refused, BASE_NEEDS_A_CLONE);
    let designed = Request {
        design: Some("a"),
        clone: Some(&clone),
        ..request
    };
    assert!(talker.request_rows(&designed).is_err());
    let cloned = Request {
        clone: Some(&clone),
        ..request
    };
    assert!(talker.request_rows(&cloned).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A voice design checkpoint refuses a clone prompt.
#[test]
fn a_voice_design_checkpoint_refuses_a_clone_prompt() {
    let dir = toy_dir("design_refuses_clone");
    let talker = Talker::load(&dir, Rows::Exactly(64)).unwrap();
    let clone = ClonePrompt {
        speaker: vec![0.0; talker.text_table.width()],
        reference: None,
    };
    let request = Request {
        text: "ab",
        design: None,
        speaker: None,
        language: None,
        seed: 0,
        clone: Some(&clone),
    };
    assert!(talker.request_rows(&request).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The log-mel spectrogram of `samples` by the definition of each step, in f64.
fn plain_log_mel(samples: &[f32], rate: usize, bands: usize) -> Vec<f64> {
    use speaker::*;
    let pad = (MEL_N_FFT - MEL_HOP) / 2;
    let n = samples.len();
    let x: Vec<f64> = (0..n + 2 * pad)
        .map(|i| {
            let j = i as isize - pad as isize;
            let j = match j {
                j if j < 0 => -j,
                j if j >= n as isize => 2 * (n as isize - 1) - j,
                j => j,
            };
            samples[j as usize] as f64
        })
        .collect();
    let frames = 1 + (x.len() - MEL_N_FFT) / MEL_HOP;
    let bins = MEL_N_FFT / 2 + 1;
    let bank = filter_bank(bands, rate);
    let tau = 2.0 * std::f64::consts::PI;
    let mut out = vec![0f64; bands * frames];
    for f in 0..frames {
        let mag: Vec<f64> = (0..bins)
            .map(|k| {
                let (mut re, mut im) = (0f64, 0f64);
                for t in 0..MEL_N_FFT {
                    let w = 0.5 - 0.5 * (tau * t as f64 / MEL_WINDOW as f64).cos();
                    let v = x[f * MEL_HOP + t] * w;
                    let a = tau * (k * t) as f64 / MEL_N_FFT as f64;
                    re += v * a.cos();
                    im -= v * a.sin();
                }
                (re * re + im * im + MEL_MAGNITUDE_EPS).sqrt()
            })
            .collect();
        for b in 0..bands {
            let sum: f64 = (0..bins).map(|k| bank[b * bins + k] * mag[k]).sum();
            out[b * frames + f] = sum.max(MEL_LOG_FLOOR).ln();
        }
    }
    out
}

/// The mel front end gives the plain f64 computation of each step.
#[test]
fn the_mel_front_end_matches_a_plain_computation() {
    let rate = 24_000;
    let bands = 16;
    let samples: Vec<f32> = (0..3000)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 3100.0 * t).sin()
        })
        .collect();
    let (got, frames) = speaker::log_mel(&samples, rate, bands).unwrap();
    let want = plain_log_mel(&samples, rate, bands);
    assert_eq!(
        frames,
        1 + (samples.len() + speaker::MEL_N_FFT - speaker::MEL_HOP - speaker::MEL_N_FFT)
            / speaker::MEL_HOP
    );
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert!((*a as f64 - b).abs() <= 1e-4 * b.abs().max(1.0), "{a} {b}");
    }
    assert!(speaker::log_mel(&samples[..100], rate, bands).is_err());
}

/// Each band of the filter bank is a triangle of area 1 in hertz, and the bands cover the
/// spectrum in order.
#[test]
fn the_filter_bank_bands_are_normalised_triangles() {
    let rate = 24_000;
    let bands = 32;
    let bank = speaker::filter_bank(bands, rate);
    let bins = speaker::MEL_N_FFT / 2 + 1;
    let hz = rate as f64 / 2.0 / (bins - 1) as f64;
    let mut last_peak = 0;
    for b in 0..bands {
        let row = &bank[b * bins..(b + 1) * bins];
        let peak = (0..bins)
            .max_by(|i, j| row[*i].total_cmp(&row[*j]))
            .unwrap();
        assert!(peak >= last_peak, "band {b}");
        last_peak = peak;
        let area: f64 = row.iter().sum::<f64>() * hz;
        assert!((area - 1.0).abs() < 0.25, "band {b}: {area}");
    }
}
