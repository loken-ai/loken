//! A reasoning budget: the reasoning of a streamed answer is cut once it has spent its tokens,
//! closed with the closing tag of the chat template, and the answer is generated from there.
//!
//! The cut ends the first generation, waits for it to finish and starts a second one whose prompt is the first prompt,
//! the reasoning so far and the closing tag. The caller sees one stream: the reasoning, the
//! closing tag, then the answer.

use crate::inference::engine::llm_engine::{FinishReason, GenerationParams, GenerationResult};
use std::future::Future;
use tokio::sync::mpsc::{channel, Receiver};

/// How a model writes a reasoning block: the tag that opens it, the tag that closes it, and the
/// text that ends it for the answer to follow, as its template writes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// `<think>` and `</think>`, the line of qwen3, deepseek-r1 and the like.
    Think,
    /// Harmony's `analysis` channel, closed for the `final` channel.
    Harmony,
    /// gemma's `thought` channel.
    Gemma,
}

impl Family {
    fn open(self) -> &'static str {
        match self {
            Family::Think => "<think>",
            Family::Harmony => "<|channel|>analysis<|message|>",
            Family::Gemma => "<|channel>thought\n",
        }
    }

    fn close(self) -> &'static str {
        match self {
            Family::Think => "</think>",
            Family::Harmony => "<|end|>",
            Family::Gemma => "<channel|>",
        }
    }

    /// What ends a reasoning that was cut: its closing tag and what opens the answer.
    fn closing(self) -> &'static str {
        match self {
            Family::Think => "\n</think>\n\n",
            Family::Harmony => "<|end|><|start|>assistant<|channel|>final<|message|>",
            Family::Gemma => "<channel|>",
        }
    }

    /// The family a text is written in. A reasoning the prompt opened is a `<think>` one.
    fn of(text: &str) -> Self {
        if text.contains(Family::Harmony.open()) {
            Family::Harmony
        } else if text.contains(Family::Gemma.open()) {
            Family::Gemma
        } else {
            Family::Think
        }
    }
}

/// Whether `text` ends inside a reasoning block. A reasoning the prompt opened counts from the
/// start of the text.
fn ends_inside(text: &str, opened_by_prompt: bool) -> bool {
    let family = Family::of(text);
    let opened_at = text.rfind(family.open());
    let opened = opened_at.is_some() || (opened_by_prompt && family == Family::Think);
    let closed = text
        .rfind(family.close())
        .is_some_and(|c| opened_at.is_none_or(|o| o + family.open().len() <= c));
    opened && !closed
}

/// What the reasoning may spend, and how the block is told to end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Tokens the reasoning may take, one per chunk of the stream.
    pub tokens: usize,
    /// The prompt already opened the reasoning block, so the stream starts inside it.
    pub opened_by_prompt: bool,
}

/// What a generation continues from: the reasoning it has written so far and the closing tag.
pub struct Continuation {
    /// The text to append to the first prompt.
    pub suffix: String,
    /// The tokens the first generation has produced.
    pub spent: usize,
}

/// `rx` limited by `budget`. While the reasoning is open and under its budget the chunks pass
/// through. When it reaches the budget, `rx` is dropped, which ends its generation, the closing
/// tag is sent, and the stream goes on with what `restart` returns for the continuation.
/// A reasoning that closes by itself, or never opens, is not touched.
pub fn limit<F, Fut>(
    mut rx: Receiver<Result<String, String>>,
    budget: Budget,
    restart: F,
) -> Receiver<Result<String, String>>
where
    F: FnOnce(Continuation) -> Fut + Send + 'static,
    Fut: Future<Output = Result<Receiver<Result<String, String>>, String>> + Send + 'static,
{
    let (tx, out) = channel(64);
    tokio::spawn(async move {
        let mut text = String::new();
        let mut inside = budget.opened_by_prompt;
        let mut spent = 0usize;
        while let Some(item) = rx.recv().await {
            let Ok(chunk) = &item else {
                if tx.send(item).await.is_err() {
                    return;
                }
                continue;
            };
            spent += 1;
            text.push_str(chunk);
            inside = ends_inside(&text, budget.opened_by_prompt);
            if tx.send(item).await.is_err() {
                return;
            }
            if inside && spent >= budget.tokens {
                break;
            }
        }
        if !(inside && spent >= budget.tokens) {
            return;
        }
        // The budget is spent inside the reasoning: end it and go on from there.
        // The first generation ends before the second starts: closing the channel stops it,
        // and the channel is read to its end, which comes when the generation has finished.
        rx.close();
        while rx.recv().await.is_some() {}
        let close = Family::of(&text).closing().to_string();
        if tx.send(Ok(close.clone())).await.is_err() {
            return;
        }
        let continuation = Continuation {
            suffix: format!("{text}{close}"),
            spent,
        };
        match restart(continuation).await {
            Ok(mut next) => {
                while let Some(item) = next.recv().await {
                    if tx.send(item).await.is_err() {
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
            }
        }
    });
    out
}

/// A whole answer under a reasoning budget. The first generation is capped at the budget. One
/// that stops there is continued: from the closing tag when it is still reasoning, from where
/// it stopped when the reasoning had ended. `generate` runs one generation of the prompt with
/// the given text appended.
pub async fn generate<F, Fut>(
    budget: Budget,
    params: GenerationParams,
    generate: F,
) -> Result<GenerationResult, Box<dyn std::error::Error>>
where
    F: Fn(String, GenerationParams) -> Fut,
    Fut: Future<Output = Result<GenerationResult, Box<dyn std::error::Error>>>,
{
    let cap = params.max_tokens;
    let first_cap = cap.map_or(budget.tokens, |c| c.min(budget.tokens));
    let first = generate(
        String::new(),
        GenerationParams {
            max_tokens: Some(first_cap),
            ..params.clone()
        },
    )
    .await?;
    let more = cap.is_none_or(|c| c > first_cap);
    if !(first.finish_reason == FinishReason::MaxTokens && more) {
        return Ok(first);
    }
    let reasoning = ends_inside(&first.text, budget.opened_by_prompt);
    let joined = if reasoning {
        format!("{}{}", first.text, Family::of(&first.text).closing())
    } else {
        first.text.clone()
    };
    let second = generate(
        joined.clone(),
        GenerationParams {
            max_tokens: cap.map(|c| c.saturating_sub(first.eval_count as usize)),
            ..params
        },
    )
    .await?;
    let mut logprobs = first.logprobs;
    logprobs.extend(second.logprobs);
    Ok(GenerationResult {
        tokens: second.tokens,
        text: format!("{joined}{}", second.text),
        prompt_eval_count: first.prompt_eval_count,
        prompt_eval_duration: first.prompt_eval_duration + second.prompt_eval_duration,
        eval_count: first.eval_count + second.eval_count,
        eval_duration: first.eval_duration + second.eval_duration,
        cached_prompt_tokens: first.cached_prompt_tokens,
        finish_reason: second.finish_reason,
        logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream_of(chunks: &[&str]) -> Receiver<Result<String, String>> {
        let (tx, rx) = channel(64);
        for c in chunks {
            tx.try_send(Ok(c.to_string())).unwrap();
        }
        rx
    }

    async fn all(mut rx: Receiver<Result<String, String>>) -> String {
        let mut text = String::new();
        while let Some(item) = rx.recv().await {
            text.push_str(&item.unwrap());
        }
        text
    }

    #[tokio::test]
    async fn reasoning_over_its_budget_is_closed_and_the_answer_goes_on() {
        let first = stream_of(&["a", "b", "c", "d", "e"]);
        let budget = Budget {
            tokens: 3,
            opened_by_prompt: true,
        };
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let kept = seen.clone();
        let out = limit(first, budget, move |c| async move {
            *kept.lock().unwrap() = Some((c.suffix, c.spent));
            Ok(stream_of(&["the answer"]))
        });
        assert_eq!(all(out).await, "abc\n</think>\n\nthe answer");
        let (suffix, spent) = seen.lock().unwrap().clone().unwrap();
        assert_eq!(suffix, "abc\n</think>\n\n");
        assert_eq!(spent, 3);
    }

    #[tokio::test]
    async fn reasoning_that_closes_inside_its_budget_is_left_alone() {
        let first = stream_of(&["hm", "</think>", "\n\n", "the", " answer", "!"]);
        let budget = Budget {
            tokens: 3,
            opened_by_prompt: true,
        };
        let out = limit(first, budget, |_| async {
            Err("not restarted".to_string())
        });
        assert_eq!(all(out).await, "hm</think>\n\nthe answer!");
    }

    #[tokio::test]
    async fn a_stream_that_never_reasons_is_left_alone() {
        let first = stream_of(&["just", " an", " answer", " here"]);
        let budget = Budget {
            tokens: 2,
            opened_by_prompt: false,
        };
        let out = limit(first, budget, |_| async {
            Err("not restarted".to_string())
        });
        assert_eq!(all(out).await, "just an answer here");
    }

    #[tokio::test]
    async fn reasoning_opened_by_the_stream_counts_from_its_opening() {
        let first = stream_of(&["<think>", "x", "y", "z"]);
        let budget = Budget {
            tokens: 3,
            opened_by_prompt: false,
        };
        let out = limit(first, budget, |c| async move {
            assert!(c.suffix.starts_with("<think>xy"));
            Ok(stream_of(&["done"]))
        });
        assert_eq!(all(out).await, "<think>xy\n</think>\n\ndone");
    }

    #[tokio::test]
    async fn a_failed_restart_ends_the_stream_with_its_error() {
        let first = stream_of(&["a", "b"]);
        let budget = Budget {
            tokens: 2,
            opened_by_prompt: true,
        };
        let mut out = limit(first, budget, |_| async { Err("no room".to_string()) });
        let mut items = Vec::new();
        while let Some(i) = out.recv().await {
            items.push(i);
        }
        assert_eq!(items.last(), Some(&Err("no room".to_string())));
    }

    fn result(text: &str, eval: u64, finish: FinishReason) -> GenerationResult {
        GenerationResult {
            tokens: Vec::new(),
            text: text.to_string(),
            prompt_eval_count: 10,
            prompt_eval_duration: 5,
            eval_count: eval,
            eval_duration: 7,
            cached_prompt_tokens: 0,
            finish_reason: finish,
            logprobs: Vec::new(),
        }
    }

    /// A generator that answers from a script and records what each call was given.
    fn scripted(
        script: Vec<GenerationResult>,
    ) -> (
        std::sync::Arc<std::sync::Mutex<Vec<(String, Option<usize>)>>>,
        impl Fn(
            String,
            GenerationParams,
        ) -> std::future::Ready<Result<GenerationResult, Box<dyn std::error::Error>>>,
    ) {
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = calls.clone();
        let script = std::sync::Mutex::new(script.into_iter());
        (calls, move |suffix, params: GenerationParams| {
            log.lock().unwrap().push((suffix, params.max_tokens));
            std::future::ready(Ok(script.lock().unwrap().next().unwrap()))
        })
    }

    #[tokio::test]
    async fn a_whole_reasoning_over_its_budget_is_closed_and_continued() {
        let (calls, run) = scripted(vec![
            result("thinking hard", 4, FinishReason::MaxTokens),
            result("the answer", 3, FinishReason::Eos),
        ]);
        let budget = Budget {
            tokens: 4,
            opened_by_prompt: true,
        };
        let params = GenerationParams {
            max_tokens: Some(20),
            ..Default::default()
        };
        let out = generate(budget, params, run).await.unwrap();
        assert_eq!(out.text, "thinking hard\n</think>\n\nthe answer");
        assert_eq!((out.eval_count, out.prompt_eval_count), (7, 10));
        assert_eq!(out.finish_reason, FinishReason::Eos);
        let calls = calls.lock().unwrap().clone();
        assert_eq!(calls[0], (String::new(), Some(4)));
        assert_eq!(
            calls[1],
            ("thinking hard\n</think>\n\n".to_string(), Some(16))
        );
    }

    #[tokio::test]
    async fn an_answer_cut_by_the_budget_after_its_reasoning_is_continued_not_closed() {
        let (calls, run) = scripted(vec![
            result("hm</think>\n\nhello wor", 4, FinishReason::MaxTokens),
            result("ld", 1, FinishReason::Eos),
        ]);
        let budget = Budget {
            tokens: 4,
            opened_by_prompt: true,
        };
        let params = GenerationParams {
            max_tokens: Some(20),
            ..Default::default()
        };
        let out = generate(budget, params, run).await.unwrap();
        assert_eq!(out.text, "hm</think>\n\nhello world");
        assert_eq!(calls.lock().unwrap()[1].0, "hm</think>\n\nhello wor");
    }

    #[tokio::test]
    async fn a_generation_that_ends_inside_the_budget_is_returned_as_it_is() {
        let (calls, run) = scripted(vec![result("short", 2, FinishReason::Eos)]);
        let budget = Budget {
            tokens: 4,
            opened_by_prompt: true,
        };
        let out = generate(budget, GenerationParams::default(), run)
            .await
            .unwrap();
        assert_eq!(out.text, "short");
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_request_cap_under_the_budget_is_the_cap() {
        let (calls, run) = scripted(vec![result("cut", 3, FinishReason::MaxTokens)]);
        let budget = Budget {
            tokens: 10,
            opened_by_prompt: true,
        };
        let params = GenerationParams {
            max_tokens: Some(3),
            ..Default::default()
        };
        let out = generate(budget, params, run).await.unwrap();
        assert_eq!(out.finish_reason, FinishReason::MaxTokens);
        assert_eq!(
            calls.lock().unwrap().clone(),
            vec![(String::new(), Some(3))]
        );
    }

    #[tokio::test]
    async fn a_harmony_reasoning_is_closed_for_the_final_channel() {
        let first = stream_of(&["<|channel|>analysis<|message|>", "a", "b", "c"]);
        let budget = Budget {
            tokens: 3,
            opened_by_prompt: false,
        };
        let out = limit(first, budget, |_| async { Ok(stream_of(&["done"])) });
        assert_eq!(
            all(out).await,
            "<|channel|>analysis<|message|>ab<|end|><|start|>assistant<|channel|>final<|message|>done"
        );
    }

    #[tokio::test]
    async fn a_gemma_reasoning_is_closed_with_its_own_tag() {
        let first = stream_of(&["<|channel>thought\n", "a", "b", "c"]);
        let budget = Budget {
            tokens: 3,
            opened_by_prompt: false,
        };
        let out = limit(first, budget, |_| async { Ok(stream_of(&["done"])) });
        assert_eq!(all(out).await, "<|channel>thought\nab<channel|>done");
    }

    #[test]
    fn a_text_ends_inside_a_reasoning_only_when_its_block_is_open() {
        assert!(ends_inside("plan", true));
        assert!(!ends_inside("plan</think>answer", true));
        assert!(!ends_inside("answer", false));
        assert!(ends_inside("<think>plan", false));
        assert!(ends_inside("<|channel|>analysis<|message|>plan", false));
        assert!(!ends_inside(
            "<|channel|>analysis<|message|>plan<|end|>",
            false
        ));
        assert!(ends_inside("<|channel>thought\nplan", false));
        assert!(!ends_inside("<|channel>thought\nplan<channel|>x", false));
    }
}
