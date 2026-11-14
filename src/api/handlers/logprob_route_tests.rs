//! The log-probabilities of an answer spell it, one entry per token, on every route that
//! reports them, whole or streamed, on the plain and on the sequential decode.
//!
//! The tiny checkpoint's vocabulary writes one word per token.
//! The entries spell an answer when their tokens are its words, in order.

use super::request_figures_tests::{body_text, data_lines, prompt, server_with_tiny_model, MODEL};
use super::*;
use serde_json::{json, Value};

/// Tokens each answer may write: enough for the decode to repeat itself and draft.
const MAX_TOKENS: u64 = 200;
/// Words of each prompt.
const PROMPT_WORDS: usize = 8;
/// Alternatives asked for each token.
const TOP: u64 = 2;
/// A token the answer to the first prompt writes after a few others.
const END_TOKEN: u32 = 114;

/// What a route answered: its text, the token of each entry, and the tokens it counted.
struct Answer {
    text: String,
    entries: Vec<String>,
    counted: Option<u64>,
}

impl Answer {
    /// What departs from an answer whose entries are its words, in order, and as many as the
    /// tokens counted; `None` when nothing does.
    fn departure(&self, route: &str) -> Option<String> {
        let words: Vec<&str> = self.text.split_whitespace().collect();
        if words.is_empty() {
            return Some(format!("{route}: no answer"));
        }
        if self.entries != words {
            return Some(format!(
                "{route}: entries {:?} do not spell the answer {:?}",
                self.entries, self.text
            ));
        }
        match self.counted {
            Some(counted) if counted != self.entries.len() as u64 => Some(format!(
                "{route}: {} entries for {counted} tokens",
                self.entries.len()
            )),
            _ => None,
        }
    }

    fn assert_spelled(&self, route: &str) {
        if let Some(departure) = self.departure(route) {
            panic!("{departure}");
        }
    }
}

/// The token of each entry of a list of log-probabilities.
fn tokens_of(list: &Value) -> Vec<String> {
    list.as_array()
        .map(|a| {
            a.iter()
                .map(|e| e["token"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// The JSON lines of an NDJSON body.
fn json_lines(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

async fn ollama_generate_answer(state: &APIServer, words: String, stream: bool) -> Answer {
    let request = serde_json::from_value(json!({
        "model": MODEL,
        "prompt": words,
        "stream": stream,
        "logprobs": true,
        "top_logprobs": TOP,
        "options": {"temperature": 0, "num_predict": MAX_TOKENS, "repeat_penalty": 1.0},
    }))
    .unwrap();
    let response = super::ollama_generate(
        axum::extract::State(state.clone()),
        axum::http::HeaderMap::new(),
        OllamaJson(request),
    )
    .await
    .unwrap_or_else(|e| panic!("the generate request is served: {e:?}"));
    let lines = json_lines(&body_text(response).await);
    Answer {
        text: lines
            .iter()
            .filter_map(|l| l["response"].as_str())
            .collect(),
        entries: lines
            .iter()
            .flat_map(|l| tokens_of(&l["logprobs"]))
            .collect(),
        counted: lines.iter().rev().find_map(|l| l["eval_count"].as_u64()),
    }
}

async fn ollama_chat_answer(state: &APIServer, words: String, stream: bool) -> Answer {
    let request = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": words}],
        "stream": stream,
        "logprobs": true,
        "top_logprobs": TOP,
        "options": {"temperature": 0, "num_predict": MAX_TOKENS, "repeat_penalty": 1.0},
    }))
    .unwrap();
    let response = super::ollama_chat(
        axum::extract::State(state.clone()),
        axum::http::HeaderMap::new(),
        OllamaJson(request),
    )
    .await
    .unwrap_or_else(|e| panic!("the chat request is served: {e:?}"));
    let lines = json_lines(&body_text(response).await);
    Answer {
        text: lines
            .iter()
            .filter_map(|l| l["message"]["content"].as_str())
            .collect(),
        entries: lines
            .iter()
            .flat_map(|l| tokens_of(&l["logprobs"]))
            .collect(),
        counted: lines.iter().rev().find_map(|l| l["eval_count"].as_u64()),
    }
}

/// A chat completion; `rules` adds a token rule that sends it to the sequential decode.
async fn chat_completion_answer(
    state: &APIServer,
    words: String,
    stream: bool,
    rules: bool,
) -> Answer {
    let mut body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": words}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "logprobs": true,
        "top_logprobs": TOP,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    if rules {
        body["min_tokens"] = json!(1);
    }
    let response = super::openai::chat_completion_at(
        state.clone(),
        axum::http::HeaderMap::new(),
        serde_json::from_value(body).unwrap(),
        crate::api::gate::Priority::Interactive,
    )
    .await
    .unwrap_or_else(|e| panic!("the chat completion is served: {e:?}"));
    let body = body_text(response).await;
    if !stream {
        let v: Value = serde_json::from_str(&body).unwrap();
        return Answer {
            text: v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_string(),
            entries: tokens_of(&v["choices"][0]["logprobs"]["content"]),
            counted: v["usage"]["completion_tokens"].as_u64(),
        };
    }
    let chunks = data_lines(&body);
    Answer {
        text: chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect(),
        entries: chunks
            .iter()
            .flat_map(|c| tokens_of(&c["choices"][0]["logprobs"]["content"]))
            .collect(),
        counted: chunks
            .iter()
            .rev()
            .find_map(|c| c["usage"]["completion_tokens"].as_u64()),
    }
}

/// A completion; `end` names a token that ends it.
async fn text_completion_answer(
    state: &APIServer,
    words: String,
    stream: bool,
    end: Option<u32>,
) -> Answer {
    let mut body = json!({
        "model": MODEL,
        "prompt": words,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "logprobs": TOP,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    if let Some(end) = end {
        body["stop_token_ids"] = json!([end]);
    }
    let response = super::openai::text_completions_at(
        state.clone(),
        axum::http::HeaderMap::new(),
        body,
        crate::api::gate::Priority::Interactive,
    )
    .await
    .unwrap_or_else(|e| panic!("the completion is served: {e:?}"));
    let body = body_text(response).await;
    let tokens = |v: &Value| -> Vec<String> {
        v["choices"][0]["logprobs"]["tokens"]
            .as_array()
            .map(|a| a.iter().map(|t| t.as_str().unwrap().to_string()).collect())
            .unwrap_or_default()
    };
    if !stream {
        let v: Value = serde_json::from_str(&body).unwrap();
        return Answer {
            text: v["choices"][0]["text"].as_str().unwrap().to_string(),
            entries: tokens(&v),
            counted: v["usage"]["completion_tokens"].as_u64(),
        };
    }
    let chunks = data_lines(&body);
    Answer {
        text: chunks
            .iter()
            .filter_map(|c| c["choices"][0]["text"].as_str())
            .collect(),
        entries: chunks.iter().flat_map(tokens).collect(),
        counted: chunks
            .iter()
            .rev()
            .find_map(|c| c["usage"]["completion_tokens"].as_u64()),
    }
}

/// Each route, whole and streamed, gives one entry per token of its answer, and the entries
/// spell it.
#[tokio::test(flavor = "multi_thread")]
async fn every_route_spells_its_answer_in_its_logprobs() {
    let _one = super::request_figures_tests::ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let words = || prompt(PROMPT_WORDS, vocab, 3);
    let mut departures = Vec::new();
    for stream in [false, true] {
        let shape = if stream { "streamed" } else { "whole" };
        let answers = [
            (
                "/api/generate",
                ollama_generate_answer(&state, words(), stream).await,
            ),
            (
                "/api/chat",
                ollama_chat_answer(&state, words(), stream).await,
            ),
            (
                "/v1/chat/completions",
                chat_completion_answer(&state, words(), stream, false).await,
            ),
            (
                "/v1/completions",
                text_completion_answer(&state, words(), stream, None).await,
            ),
        ];
        for (route, answer) in answers {
            departures.extend(answer.departure(&format!("{route} {shape}")));
        }
    }
    assert!(departures.is_empty(), "{}", departures.join("\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A request with a token rule decodes sequentially, and its entries spell its answer, whole
/// and streamed.
#[tokio::test(flavor = "multi_thread")]
async fn the_sequential_decode_spells_its_answer_in_its_logprobs() {
    let _one = super::request_figures_tests::ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    for stream in [false, true] {
        let shape = if stream { "streamed" } else { "whole" };
        chat_completion_answer(&state, prompt(PROMPT_WORDS, vocab, 3), stream, true)
            .await
            .assert_spelled(&format!("/v1/chat/completions {shape} with a token rule"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2 streams of different prompts sent together each get the entries of their own answer.
/// The first ends on an end token, whose draw gives no entry.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_streams_each_spell_their_own_answer() {
    let _one = super::request_figures_tests::ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let (first, second) = tokio::join!(
        text_completion_answer(
            &state,
            prompt(PROMPT_WORDS, vocab, 3),
            true,
            Some(END_TOKEN)
        ),
        text_completion_answer(&state, prompt(PROMPT_WORDS, vocab, 70), true, None),
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        first.text.split_whitespace().count() < MAX_TOKENS as usize,
        "the first stream ends on its end token: {:?}",
        first.text
    );
    let departures: Vec<String> = [
        first.departure("the first stream"),
        second.departure("the second stream"),
    ]
    .into_iter()
    .flatten()
    .collect();
    assert!(departures.is_empty(), "{}", departures.join("\n"));
}

/// The text of a completion that asks for no log-probabilities.
async fn plain_completion(state: &APIServer, words: String, max_tokens: u64) -> String {
    let body = json!({
        "model": MODEL,
        "prompt": words,
        "max_tokens": max_tokens,
        "temperature": 0,
    });
    let response = super::openai::text_completions_at(
        state.clone(),
        axum::http::HeaderMap::new(),
        body,
        crate::api::gate::Priority::Interactive,
    )
    .await
    .unwrap_or_else(|e| panic!("the completion is served: {e:?}"));
    let v: Value = serde_json::from_str(&body_text(response).await).unwrap();
    v["choices"][0]["text"].as_str().unwrap().to_string()
}

/// A request for log-probabilities that arrives while the batch worker serves another request
/// gets the entries of its own answer.
/// The other request is answered.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_for_logprobs_behind_a_busy_worker_spells_its_answer() {
    let _one = super::request_figures_tests::ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let engine = state.get_engine(MODEL).await.expect("the model is loaded");
    engine.hold_lone_requests_on(true);
    // A short request moves the model into its batch worker, as a server on a card does.
    plain_completion(&state, prompt(PROMPT_WORDS, vocab, 5), 2).await;
    let busy = plain_completion(&state, prompt(PROMPT_WORDS, vocab, 70), MAX_TOKENS);
    let asking = async {
        while engine.in_flight() == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        let in_worker = engine.is_continuous().await;
        let answer =
            chat_completion_answer(&state, prompt(PROMPT_WORDS, vocab, 3), false, false).await;
        (in_worker, answer)
    };
    let (busy, (in_worker, answer)) = tokio::join!(busy, asking);
    let _ = std::fs::remove_dir_all(&dir);
    #[cfg(feature = "gpu")]
    assert!(
        crate::inference::place::device_probe::card_count() == 0 || in_worker,
        "the model was not in its batch worker when the request for log-probabilities came"
    );
    #[cfg(not(feature = "gpu"))]
    let _ = in_worker;
    assert!(!busy.is_empty(), "the busy request was not answered");
    answer.assert_spelled("the request for log-probabilities");
}

/// 2 whole answers of different prompts sent together each get the entries of their own
/// answer, on a model that batches concurrent requests.
#[tokio::test(flavor = "multi_thread")]
async fn overlapping_answers_each_spell_their_own_answer() {
    let _one = super::request_figures_tests::ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let (first, second) = tokio::join!(
        chat_completion_answer(&state, prompt(PROMPT_WORDS, vocab, 3), false, false),
        chat_completion_answer(&state, prompt(PROMPT_WORDS, vocab, 70), false, false),
    );
    let _ = std::fs::remove_dir_all(&dir);
    let departures: Vec<String> = [
        first.departure("the first answer"),
        second.departure("the second answer"),
    ]
    .into_iter()
    .flatten()
    .collect();
    assert!(departures.is_empty(), "{}", departures.join("\n"));
}
