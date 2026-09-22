//! A batched decode worker for DeepSeek V4.1.
//!
//! Concurrent requests to one model serialise through a lone `forward_decode`, leaving the card
//! idle between each token. This worker is the sole caller of the model forward: requests submit a
//! prompt, a sampling spec and a token channel, the worker prefills each and then decodes every
//! running sequence together each tick through `forward_decode_batch`, so each active expert is
//! read once for the whole batch. The batch shrinks as sequences finish, so a long generation
//! never holds a finished one's place. It is a throughput lever, not a latency one: each request
//! sees a little less than its lone rate, but many run at once.
//!
//! Ownership: the worker owns the model for its lifetime and is the only thread that calls its
//! forward, so the model needs no lock and its device handles are only ever touched from here.

use super::cache::DecodeState;
use super::model::DeepseekV41Model;
use crate::inference::offload::{with_offload, Offload};
use crate::inference::serve::continuous_batch::SamplingParams;
use crate::inference::serve::continuous_serve::sample_row_host;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;

// The worker streams the shared `CbToken` the continuous-batch server uses, so one serving path
// carries both. Its `Done` finish reason is not read by the stream glue; it is set for parity.
use crate::inference::serve::continuous_serve::CbToken;
use crate::inference::serve::scheduler::FinishReason;

struct Submission {
    prompt: Vec<u32>,
    max_new: usize,
    sampling: SamplingParams,
    tx: Sender<CbToken>,
}

/// One running sequence the worker decodes: its cache, the token it decodes next, how many it has
/// produced, its budget and sampling spec, the prompt and generated tokens (the repeat-penalty
/// window), and the channel it streams back on.
struct Seq {
    state: DecodeState,
    next: u32,
    produced: usize,
    max_new: usize,
    sampling: SamplingParams,
    prompt: Vec<u32>,
    generated: Vec<u32>,
    tx: Sender<CbToken>,
}

/// A handle to submit requests to the worker.
pub struct V41BatchServer {
    submit_tx: Sender<Submission>,
}

// The worker thread is the sole owner and caller of the model; its device handles are only ever
// touched from that one thread, the same contract the generic continuous server rests on.
struct SendModel {
    model: Arc<DeepseekV41Model>,
    offload: Option<Arc<dyn Offload>>,
}
unsafe impl Send for SendModel {}

/// The last `last_n` tokens of (prompt ++ generated) - the window `sample_row_host` reads for the
/// repeat penalty. Empty when the penalty is off.
fn recent_window(prompt: &[u32], generated: &[u32], last_n: usize) -> Vec<u32> {
    if last_n == 0 {
        return Vec::new();
    }
    let total = prompt.len() + generated.len();
    let start = total.saturating_sub(last_n);
    prompt
        .iter()
        .chain(generated.iter())
        .skip(start)
        .copied()
        .collect()
}

impl V41BatchServer {
    /// Start the worker. `offload` is the card placement to run the forwards under (None on a pure
    /// CPU host); `eos` ends a sequence; `max_batch` caps how many sequences decode together.
    pub fn spawn(
        model: Arc<DeepseekV41Model>,
        offload: Option<Arc<dyn Offload>>,
        eos: Option<u32>,
        max_batch: usize,
    ) -> Self {
        let (submit_tx, submit_rx) = mpsc::channel::<Submission>();
        let sm = SendModel { model, offload };
        std::thread::Builder::new()
            .name("dsv41-batch".into())
            .spawn(move || worker(sm, eos, max_batch.max(1), submit_rx))
            .expect("spawn dsv41 batch worker");
        Self { submit_tx }
    }

    /// Submit a greedy request; the returned receiver yields its tokens as they are produced,
    /// ending with a `Done` token.
    pub fn submit(&self, prompt: Vec<u32>, max_new: usize) -> Receiver<CbToken> {
        self.submit_sampled(prompt, max_new, SamplingParams::greedy())
    }

    /// Submit with explicit sampling controls. Dropping the receiver lets the worker reclaim the
    /// slot once the sequence finishes.
    pub fn submit_sampled(
        &self,
        prompt: Vec<u32>,
        max_new: usize,
        sampling: SamplingParams,
    ) -> Receiver<CbToken> {
        let (tx, rx) = mpsc::channel();
        let _ = self.submit_tx.send(Submission {
            prompt,
            max_new,
            sampling,
            tx,
        });
        rx
    }
}

impl crate::inference::serve::continuous_serve::BatchedServer for V41BatchServer {
    fn submit_cb(
        &self,
        prompt: Vec<u32>,
        max_new: usize,
        sampling: SamplingParams,
    ) -> Receiver<CbToken> {
        self.submit_sampled(prompt, max_new, sampling)
    }
}

/// Run `f` under the worker's offload, if it has one.
fn under<R>(offload: &Option<Arc<dyn Offload>>, f: impl FnOnce() -> R) -> R {
    match offload {
        Some(o) => with_offload(o.clone(), f),
        None => f(),
    }
}

/// Draw the token that follows `flat[row]` for `seq`, applying its sampling spec over its
/// repeat-penalty window.
fn sample(seq: &Seq, flat: &[f32], row: usize, vocab: usize) -> u32 {
    let logits = flat[row * vocab..(row + 1) * vocab].to_vec();
    let recent = recent_window(&seq.prompt, &seq.generated, seq.sampling.repeat_last_n);
    sample_row_host(logits, &seq.sampling, &recent, seq.produced)
        .unwrap_or_else(|_| argmax(flat, row, vocab))
}

fn argmax(flat: &[f32], row: usize, vocab: usize) -> u32 {
    flat[row * vocab..(row + 1) * vocab]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Prefill a submission's prompt and add it to the running set, streaming nothing yet. A prefill
/// failure or an empty budget ends the request at once.
fn admit(sm: &SendModel, sub: Submission, active: &mut Vec<Seq>) {
    let mut state = sm.model.new_decode_state();
    let logits = match under(&sm.offload, || {
        sm.model.prefill_into(&sub.prompt, &mut state)
    }) {
        Ok(l) => l,
        Err(_) => {
            let _ = sub.tx.send(CbToken::Done(FinishReason::Stop));
            return;
        }
    };
    let flat = match logits.dims2().and_then(|(rows, vocab)| {
        logits
            .flatten_all()?
            .to_vec1::<f32>()
            .map(|v| (v, rows, vocab))
    }) {
        Ok((v, rows, vocab)) => (v, rows, vocab),
        Err(_) => {
            let _ = sub.tx.send(CbToken::Done(FinishReason::Stop));
            return;
        }
    };
    if sub.max_new == 0 {
        let _ = sub.tx.send(CbToken::Done(FinishReason::Stop));
        return;
    }
    let (v, rows, vocab) = flat;
    let mut seq = Seq {
        state,
        next: 0,
        produced: 0,
        max_new: sub.max_new,
        sampling: sub.sampling,
        prompt: sub.prompt,
        generated: Vec::new(),
        tx: sub.tx,
    };
    // The first token to decode: sampled from the prefill's last position.
    seq.next = sample(&seq, &v, rows - 1, vocab);
    active.push(seq);
}

fn worker(sm: SendModel, eos: Option<u32>, max_batch: usize, rx: Receiver<Submission>) {
    let mut active: Vec<Seq> = Vec::new();
    loop {
        // Idle: block for the next request rather than spin. A closed channel with nothing running
        // ends the worker.
        if active.is_empty() {
            match rx.recv() {
                Ok(sub) => admit(&sm, sub, &mut active),
                Err(_) => return,
            }
        }
        // Fill the batch from whatever else is waiting, up to the width cap.
        while active.len() < max_batch {
            match rx.try_recv() {
                Ok(sub) => admit(&sm, sub, &mut active),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if active.is_empty() {
            continue;
        }
        // One batched decode step over every running sequence.
        let mut batch: Vec<(u32, &mut DecodeState)> =
            active.iter_mut().map(|s| (s.next, &mut s.state)).collect();
        let logits = under(&sm.offload, || sm.model.forward_decode_batch(&mut batch));
        drop(batch);
        let logits = match logits {
            Ok(l) => l,
            Err(_) => {
                for s in active.drain(..) {
                    let _ = s.tx.send(CbToken::Done(FinishReason::Stop));
                }
                continue;
            }
        };
        let (_rows, vocab) = logits.dims2().expect("batched logits are 2-D");
        let flat = logits
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("batched logits read back");
        let mut finished: Vec<usize> = Vec::new();
        for (row, seq) in active.iter_mut().enumerate() {
            let tok = seq.next;
            let _ = seq.tx.send(CbToken::Tok(tok));
            seq.generated.push(tok);
            seq.produced += 1;
            if seq.produced >= seq.max_new || Some(tok) == eos {
                let reason = if Some(tok) == eos {
                    FinishReason::Eos
                } else {
                    FinishReason::Length
                };
                let _ = seq.tx.send(CbToken::Done(reason));
                finished.push(row);
            } else {
                seq.next = sample(seq, &flat, row, vocab);
            }
        }
        for &r in finished.iter().rev() {
            active.remove(r);
        }
    }
}
