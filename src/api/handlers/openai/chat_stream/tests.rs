use super::*;
use futures::StreamExt;

/// An engine that drew nothing and kept no stats.
struct Silent;

#[async_trait::async_trait]
impl StreamSource for Silent {
    fn take_logprobs(&self) -> Vec<TokenLogprob> {
        Vec::new()
    }

    fn take_last_stream_stats(&self) -> Option<StreamStats> {
        None
    }

    async fn count_tokens(&self, _text: &str) -> Option<usize> {
        None
    }
}

fn run() -> ChatRun {
    ChatRun {
        completion_id: "chatcmpl-test".into(),
        model: "m".into(),
        created_at: 0,
        service_tier: "default".into(),
        wants_logprobs: false,
        ids_as_tokens: false,
        include_reasoning: true,
        tools_active: false,
        tool_format: ToolFormat::Hermes,
        thinking_opened: false,
        stops: Vec::new(),
        keep_stop: false,
        echo_text: String::new(),
        single_tool_call: false,
        max_tokens_cap: None,
        include_usage: true,
        prompt_tokens: 1,
        probe: Arc::new(crate::request_metrics::RequestProbe::default()),
        energy: None,
        stream_started: std::time::Instant::now(),
    }
}

/// The events of a stream fed these engine messages, each as the text it sends.
async fn frames_of(messages: Vec<Result<String, String>>) -> Vec<String> {
    let (tx, rx) = tokio::sync::mpsc::channel(messages.len().max(1));
    for m in messages {
        tx.send(m).await.unwrap();
    }
    drop(tx);
    chat_events(rx, Arc::new(Silent), run())
        .map(|e| format!("{:?}", e.unwrap()))
        .collect()
        .await
}

fn token(text: &str) -> Result<String, String> {
    Ok(text.to_string())
}

fn failure() -> Result<String, String> {
    Err("engine stopped".to_string())
}

#[tokio::test]
async fn an_engine_error_after_tokens_ends_the_stream_with_the_error() {
    let frames = frames_of(vec![token("he"), token("llo"), failure()]).await;
    let last = frames.last().unwrap();
    assert!(last.contains("engine stopped"), "got: {last}");
    assert!(frames.iter().any(|f| f.contains("llo")), "got: {frames:?}");
    assert!(
        !frames.iter().any(|f| f.contains("finish_reason")),
        "got: {frames:?}"
    );
    assert!(
        !frames.iter().any(|f| f.contains("[DONE]")),
        "got: {frames:?}"
    );
}

#[tokio::test]
async fn an_engine_error_before_any_token_ends_the_stream_with_the_error() {
    let frames = frames_of(vec![failure()]).await;
    assert_eq!(frames.len(), 1, "got: {frames:?}");
    assert!(frames[0].contains("engine stopped"), "got: {frames:?}");
}

#[tokio::test]
async fn a_normal_end_closes_with_a_finish_reason_then_done() {
    let frames = frames_of(vec![token("he"), token("llo")]).await;
    let last = frames.last().unwrap();
    assert!(last.contains("[DONE]"), "got: {last}");
    let finish = frames
        .iter()
        .position(|f| f.contains("finish_reason"))
        .expect("a chunk carries the finish reason");
    assert!(finish < frames.len() - 1);
    assert!(!frames.iter().any(|f| f.contains("engine stopped")));
}
