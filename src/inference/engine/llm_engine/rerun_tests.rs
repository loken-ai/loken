//! A decode step run twice from a copy of the recurrent states, as the probation of a recorded
//! step runs it, on a small hybrid of short convolutions and attention.

use super::decode::Rerun;
use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
use crate::inference::generic_transformer::state_cache_tests::{
    argmax, floats, prompt, write_hybrid,
};
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::tensor::{Device, Tensor};
use std::sync::atomic::Ordering;

/// Blocks of the hybrid file.
const LAYERS: usize = 4;
/// Tokens of the prompt.
const PROMPT: usize = 20;
/// Tokens decoded after it.
const NEW: usize = 12;

/// The hybrid file, written into a directory of its own, removed on drop.
struct File {
    dir: std::path::PathBuf,
    path: std::path::PathBuf,
}

impl File {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("loken-rerun-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hybrid.gguf");
        write_hybrid(&path);
        Self { dir, path }
    }

    /// The model on the host, or every block on card 0.
    fn load(&self, card: Option<Device>) -> GenericBackend {
        use crate::inference::place::layer_executor::HeteroPlan;
        let g = crate::tensor::quantized::gguf_file::open_mapped(&self.path).unwrap();
        let bytes = g.mmap.clone();
        let (plan, cards) = match card {
            Some(dev) => (
                HeteroPlan::forced_gpu(LAYERS, LAYERS, 0),
                std::collections::HashMap::from([(0, dev)]),
            ),
            None => (
                HeteroPlan::forced_gpu(LAYERS, 0, 0),
                std::collections::HashMap::new(),
            ),
        };
        GenericBackend(
            GenericHeteroTransformer::from_gguf_with_kv_quant(
                g.content,
                &bytes[..],
                &cards,
                &plan,
                crate::inference::engine::llm_engine::KvQuant::Off,
                None,
            )
            .unwrap(),
        )
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// How each decode step runs.
#[derive(Clone, Copy)]
enum Step {
    /// Once.
    Once,
    /// Twice when the states were copied, the copy put back between the runs.
    /// The cache row of the first run is cut before the second, as a step written at its
    /// position again leaves it.
    Probation,
    /// Twice, the cache row cut between the runs and the recurrent states left as the first
    /// run left them.
    TwiceUnrestored,
}

/// The greedy tokens after the prompt, and the logits of the last step.
fn decode(b: &mut GenericBackend, step: Step) -> (Vec<usize>, Vec<f32>) {
    let tokens = prompt(PROMPT);
    let x = Tensor::new(&tokens[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut logits = floats(&b.0.forward(&x, 0).unwrap());
    let mut out = Vec::new();
    for i in 0..NEW {
        let next = argmax(&logits);
        out.push(next);
        let pos = PROMPT + i;
        let x = Tensor::new(&[next as u32], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        match step {
            Step::Once => {}
            Step::Probation => {
                let rerun = Rerun::take(&*b, pos);
                if rerun.allows_rerun() {
                    b.0.forward(&x, pos).unwrap();
                    b.0.trim_kv(pos);
                    rerun.restore(b).unwrap();
                }
            }
            Step::TwiceUnrestored => {
                b.0.forward(&x, pos).unwrap();
                b.0.trim_kv(pos);
            }
        }
        logits = floats(&b.0.forward(&x, pos).unwrap());
    }
    (out, logits)
}

/// The tokens of a decode whose copies of the states fail are those of a decode that runs each
/// step once.
/// On a card, where the states are put back in place, so are those of a decode whose copies
/// succeed, and a step run twice without putting the states back gives other logits: the test
/// sees a state advanced twice.
fn a_step_whose_states_were_not_copied_runs_once(card: Option<Device>, tag: &str) {
    let file = File::new(tag);
    let on_card = card.is_some();
    let mut plain = file.load(card.clone());
    assert!(plain.is_recurrent(), "the hybrid carries recurrent states");
    let (want, want_logits) = decode(&mut plain, Step::Once);

    if on_card {
        let mut copied = file.load(card.clone());
        let (got, got_logits) = decode(&mut copied, Step::Probation);
        assert_eq!(got, want, "a step run twice from its copied states");
        assert_eq!(
            got_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the last step's logits, the states put back between the runs"
        );
    }

    let failures: &[usize] = if on_card {
        &[usize::MAX, NEW / 2]
    } else {
        &[usize::MAX]
    };
    for &failing in failures {
        let mut failed = file.load(card.clone());
        failed
            .0
            .failing_checkpoints
            .store(failing, Ordering::Release);
        let (got, got_logits) = decode(&mut failed, Step::Probation);
        assert_eq!(got, want, "{failing} copies of the states fail");
        assert_eq!(
            got_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the last step's logits, {failing} copies failing"
        );
    }

    if !on_card {
        return;
    }
    let mut twice = file.load(card);
    let (_, twice_logits) = decode(&mut twice, Step::TwiceUnrestored);
    assert_ne!(
        twice_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a state advanced twice changes the logits"
    );
}

#[test]
fn on_the_host_a_step_whose_states_were_not_copied_runs_once() {
    a_step_whose_states_were_not_copied_runs_once(None, "host");
}

#[cfg(feature = "gpu")]
#[test]
fn on_a_card_a_step_whose_states_were_not_copied_runs_once() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("skipping: no usable card");
        return;
    };
    a_step_whose_states_were_not_copied_runs_once(Some(card), "card");
}
