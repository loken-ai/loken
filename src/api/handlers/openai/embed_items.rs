//! What an embeddings request asks to embed, read the way vLLM reads it: `input` as a
//! string, a list of strings, or one conversation; `messages` as one conversation or a list
//! of conversations. A conversation is one vector: its text parts joined by newlines, or its
//! single `image_url` part.

use super::*;
use serde_json::Value;

/// One thing to embed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbedItem {
    Text(String),
    /// An image by its URL: a `data:` URL carrying base64, or an http(s) URL.
    Image(String),
}

/// Whether `v` is a chat message: an object with a `role`.
fn is_message(v: &Value) -> bool {
    v.get("role").is_some_and(Value::is_string)
}

/// Whether `v` is a non-empty list of chat messages.
fn is_conversation(v: &Value) -> bool {
    v.as_array()
        .is_some_and(|a| !a.is_empty() && a.iter().all(is_message))
}

/// The one item a conversation stands for.
fn conversation_item(messages: &[Value]) -> Result<EmbedItem, String> {
    let mut texts: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    for m in messages {
        match m.get("content") {
            Some(Value::String(s)) => texts.push(s.clone()),
            Some(Value::Array(parts)) => {
                for p in parts {
                    match p.get("type").and_then(Value::as_str) {
                        Some("text") => texts.push(
                            p.get("text")
                                .and_then(Value::as_str)
                                .ok_or("a `text` part needs a `text` string")?
                                .to_string(),
                        ),
                        Some("image_url") => {
                            let url = match p.get("image_url") {
                                Some(Value::String(s)) => Some(s.as_str()),
                                Some(o) => o.get("url").and_then(Value::as_str),
                                None => None,
                            };
                            images.push(
                                url.ok_or("an `image_url` part needs an `image_url.url` string")?
                                    .to_string(),
                            );
                        }
                        other => {
                            return Err(format!(
                                "content part type {} is not embedded; send `text` or `image_url`",
                                other.unwrap_or("(none)")
                            ))
                        }
                    }
                }
            }
            Some(Value::Null) | None => {}
            Some(other) => return Err(format!("message content {other} is not text or parts")),
        }
    }
    let text = texts.join("\n");
    match images.as_slice() {
        [] if text.trim().is_empty() => Err("a conversation to embed holds no text".into()),
        [] => Ok(EmbedItem::Text(text)),
        [one] if text.trim().is_empty() => Ok(EmbedItem::Image(one.clone())),
        [_] => Err(
            "a conversation is embedded as text or as one image, not both; send them apart".into(),
        ),
        _ => Err("a conversation is embedded as one image; send each image apart".into()),
    }
}

/// The items `body` asks to embed, in order.
pub(crate) fn embed_items(body: &Value) -> Result<Vec<EmbedItem>, String> {
    if let Some(messages) = body.get("messages") {
        if is_conversation(messages) {
            return Ok(vec![conversation_item(
                messages.as_array().unwrap_or(&Vec::new()),
            )?]);
        }
        let batch = messages
            .as_array()
            .filter(|a| !a.is_empty() && a.iter().all(is_conversation))
            .ok_or("`messages` must be a conversation or a list of conversations")?;
        return batch
            .iter()
            .map(|c| conversation_item(c.as_array().unwrap_or(&Vec::new())))
            .collect();
    }
    match body.get("input") {
        Some(Value::String(s)) => Ok(vec![EmbedItem::Text(s.clone())]),
        Some(input) if is_conversation(input) => Ok(vec![conversation_item(
            input.as_array().unwrap_or(&Vec::new()),
        )?]),
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| EmbedItem::Text(s.to_string()))
                    .ok_or_else(|| "`input` must be a string or array of strings".to_string())
            })
            .collect(),
        _ => Err("`input` must be a string or array of strings".to_string()),
    }
}

/// The bytes of the image `url` names: a `data:` URL's payload, or the body of an http(s)
/// URL fetched under the server's media limits. `ordinal` counts from one, for the message.
pub(crate) async fn image_bytes(
    state: &APIServer,
    url: &str,
    ordinal: usize,
) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let limits = &state.default_inference_config.media_limits;
    let payload = match crate::api::types::data_url_base64(url) {
        Some(p) => p.to_string(),
        None => limits.fetch_image(url).await?.0,
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| format!("image {ordinal}: not base64: {e}"))?;
    limits.check_image_bytes(ordinal, &bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn image(url: &str) -> Value {
        json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]})
    }

    /// `input` keeps its plain forms.
    #[test]
    fn input_strings_are_texts() {
        assert_eq!(
            embed_items(&json!({"input": "a"})).unwrap(),
            [EmbedItem::Text("a".into())]
        );
        assert_eq!(
            embed_items(&json!({"input": ["a", "b"]})).unwrap(),
            [EmbedItem::Text("a".into()), EmbedItem::Text("b".into())]
        );
        assert!(embed_items(&json!({"input": [1]})).is_err());
        assert!(embed_items(&json!({})).is_err());
    }

    /// One conversation is one vector, in `messages` or in `input`.
    #[test]
    fn a_conversation_is_one_item() {
        let one = json!({"messages": [image("data:image/png;base64,AAAA")]});
        assert_eq!(
            embed_items(&one).unwrap(),
            [EmbedItem::Image("data:image/png;base64,AAAA".into())]
        );
        let text = json!({"input": [
            {"role": "user", "content": "a cat"},
            {"role": "user", "content": [{"type": "text", "text": "on a mat"}]}
        ]});
        assert_eq!(
            embed_items(&text).unwrap(),
            [EmbedItem::Text("a cat\non a mat".into())]
        );
    }

    /// A list of conversations is one vector each.
    #[test]
    fn a_batch_of_conversations_is_one_item_each() {
        let batch = json!({"messages": [
            [image("https://x.example/a.png")],
            [{"role": "user", "content": "a dog"}]
        ]});
        assert_eq!(
            embed_items(&batch).unwrap(),
            [
                EmbedItem::Image("https://x.example/a.png".into()),
                EmbedItem::Text("a dog".into())
            ]
        );
    }

    /// Text beside an image, two images, or an unknown part are refused.
    #[test]
    fn a_mixed_conversation_is_refused() {
        let mixed = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "a cat"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}]});
        assert!(embed_items(&mixed).unwrap_err().contains("not both"));
        let two = json!({"messages": [image("data:,a"), image("data:,b")]});
        assert!(embed_items(&two).is_err());
        let audio = json!({"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]});
        assert!(embed_items(&audio).is_err());
        assert!(embed_items(&json!({"messages": "a"})).is_err());
    }
}
