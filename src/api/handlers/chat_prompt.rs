//! The prompt a chat request prefills, built once for every dialect.
//!
//! The OpenAI, Ollama and Messages routes, the token counter and the cluster router all
//! turn a conversation into text the same way: the model's template, the tools rendered
//! in the model's own syntax, the grammar a required call or a structured output
//! imposes, the caller's thinking preference and the vision sentinel a Qwen3-VL prompt
//! needs. A step written out per route is a step one route forgets.

use super::*;
use crate::api::tool_calls::ToolFormat;

/// What a request says about its prompt, whatever dialect it arrived in.
pub(crate) struct PromptSpec<'a> {
    pub messages: &'a [Message],
    pub tools: Option<&'a Vec<Tool>>,
    /// The raw `tool_choice`, OpenAI or Anthropic shape.
    pub tool_choice: Option<&'a serde_json::Value>,
    /// `disabled`, `enabled` or an effort level (`low`, `medium`, `high`); `None` leaves
    /// the template's own behaviour.
    pub thinking: Option<&'a str>,
    /// The grammar the caller's structured-output field asks for. It takes precedence
    /// over the call object a required tool call would impose.
    pub output_grammar: Option<String>,
}

/// The rendered prompt and what the response side needs to read the answer back.
pub(crate) struct PreparedPrompt {
    pub prompt: String,
    pub tools_active: bool,
    pub tool_format: ToolFormat,
    pub grammar: Option<String>,
    /// Whether the template already opened the thinking block.
    pub thinking_opened: bool,
}

/// The token a Qwen3-VL prompt carries where the vision encoder's output is spliced; the
/// engine expands it to the image's patch count at prefill.
const QWEN35_IMAGE_SENTINEL: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// `thinking` from OpenAI's `reasoning_effort`: `none` and `minimal` switch thinking
/// off, the levels pass through, anything else leaves it on.
pub(crate) fn thinking_from_effort(effort: Option<&str>) -> Option<&'static str> {
    let effort = effort?.to_ascii_lowercase();
    Some(match effort.as_str() {
        "none" | "minimal" => "disabled",
        "low" => "low",
        "medium" => "medium",
        "high" => "high",
        _ => "enabled",
    })
}

/// The messages with the vision sentinel put in front of the last user turn that carries
/// images, the only turn whose images reach the encoder.
fn with_vision_sentinel(messages: &[Message]) -> Vec<Message> {
    let mut msgs = messages.to_vec();
    if let Some(um) = msgs
        .iter_mut()
        .rev()
        .find(|m| m.role == "user" && m.images.as_ref().is_some_and(|i| !i.is_empty()))
    {
        um.content = format!("{QWEN35_IMAGE_SENTINEL}{}", um.content);
    }
    msgs
}

/// Whether any user turn carries an image.
pub(crate) fn carries_images(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|m| m.role == "user" && m.images.as_ref().is_some_and(|i| !i.is_empty()))
}

/// The images the vision encoder reads: the last user turn's, empty when it has none.
pub(crate) fn last_user_images(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.images.clone())
        .unwrap_or_default()
}

/// Render a request against a template. Pure: the same inputs give the same prompt on
/// every node, which is what lets a peer be asked about the prompt this node would run.
pub(crate) fn build_prompt(
    spec: &PromptSpec<'_>,
    template: Option<&str>,
    model_name: &str,
    vision_sentinel: bool,
) -> PreparedPrompt {
    let tools_active = crate::api::tool_calls::should_inject_tools(spec.tools, spec.tool_choice);
    let tool_format = crate::api::tool_calls::detect_tool_format(template, model_name);
    let owned;
    let messages: &[Message] = if vision_sentinel && carries_images(spec.messages) {
        owned = with_vision_sentinel(spec.messages);
        &owned
    } else {
        spec.messages
    };
    let mut prompt = match spec.tools.filter(|_| tools_active) {
        Some(tools) if template_takes_tools(template) => {
            format_chat_prompt_with_tools(messages, template, tools)
        }
        Some(tools) => {
            let directive = crate::api::tool_calls::tool_choice_directive(spec.tool_choice);
            let flat = crate::api::tool_calls::flatten_messages(
                messages,
                tool_format,
                tools,
                directive.as_deref(),
            );
            format_chat_prompt(&flat, template)
        }
        None => format_chat_prompt(messages, template),
    };
    apply_thinking_preference(&mut prompt, spec.thinking);
    let thinking_opened = crate::api::thinking::prompt_opens_thinking(&prompt);
    let forced = spec
        .tools
        .filter(|_| tools_active)
        .zip(crate::api::tool_calls::forced_call(spec.tool_choice))
        .map(|(tools, only)| {
            format!(
                "json_schema:{}",
                crate::api::tool_calls::forced_call_schema(tools, only.as_deref())
            )
        });
    PreparedPrompt {
        prompt,
        tools_active,
        tool_format,
        grammar: spec.output_grammar.clone().or(forced),
        thinking_opened,
    }
}

impl APIServer {
    /// The chat template a model is prompted with: the one cached on its loaded engine,
    /// else the one its manifest packages (cached on the engine for next time), else one
    /// inferred from its name.
    pub(crate) async fn chat_template_for(&self, model_name: &str) -> Option<String> {
        let cached = {
            let engines = self.engines.read().await;
            engines
                .iter()
                .find(|e| e.model_id == model_name)
                .and_then(|e| e.chat_template.clone())
        };
        if cached.is_some() {
            return cached;
        }
        let read = self.read_chat_template(model_name);
        if let Some(tmpl) = read.clone() {
            let mut engines = self.engines.write().await;
            if let Some(entry) = engines.iter_mut().find(|e| e.model_id == model_name) {
                entry.chat_template = Some(tmpl);
            }
        }
        read.or_else(|| infer_template_from_model_name(model_name))
    }

    /// The prompt a request prefills on `engine`. With no engine (the model is not
    /// loaded here) the vision sentinel is left out: only a loaded model says whether it
    /// reads one.
    pub(crate) async fn prepare_prompt(
        &self,
        model_name: &str,
        engine: Option<&LlmEngine>,
        spec: &PromptSpec<'_>,
    ) -> PreparedPrompt {
        let template = self.chat_template_for(model_name).await;
        let vision = match engine {
            Some(e) if carries_images(spec.messages) => e.is_qwen35_vision().await,
            _ => false,
        };
        build_prompt(spec, template.as_deref(), model_name, vision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Tool {
        serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}}
        }))
        .unwrap()
    }

    fn user(text: &str, images: Option<Vec<String>>) -> Message {
        let mut m = Message::new("user".to_string(), text.to_string());
        m.images = images;
        m
    }

    fn spec<'a>(
        messages: &'a [Message],
        tools: Option<&'a Vec<Tool>>,
        choice: Option<&'a serde_json::Value>,
    ) -> PromptSpec<'a> {
        PromptSpec {
            messages,
            tools,
            tool_choice: choice,
            thinking: None,
            output_grammar: None,
        }
    }

    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{{ '<|im_start|>assistant\\n' }}";

    /// A required call constrains the generation to the call object in both dialects.
    #[test]
    fn a_required_call_is_a_grammar_in_every_dialect() {
        let msgs = [user("hi", None)];
        let tools = vec![tool("get_weather"), tool("get_time")];
        for choice in [
            serde_json::json!("required"),
            serde_json::json!({"type": "any"}),
        ] {
            let p = build_prompt(
                &spec(&msgs, Some(&tools), Some(&choice)),
                Some(CHATML),
                "qwen3",
                false,
            );
            let g = p.grammar.expect("a required call must set a grammar");
            assert!(g.contains("get_weather") && g.contains("get_time"), "{g}");
        }
        for choice in [
            serde_json::json!({"type": "function", "function": {"name": "get_time"}}),
            serde_json::json!({"type": "tool", "name": "get_time"}),
        ] {
            let p = build_prompt(
                &spec(&msgs, Some(&tools), Some(&choice)),
                Some(CHATML),
                "qwen3",
                false,
            );
            let g = p.grammar.expect("a named call must set a grammar");
            assert!(g.contains("get_time") && !g.contains("get_weather"), "{g}");
        }
        let auto = serde_json::json!("auto");
        let p = build_prompt(
            &spec(&msgs, Some(&tools), Some(&auto)),
            Some(CHATML),
            "qwen3",
            false,
        );
        assert!(p.grammar.is_none());
    }

    /// `none` switches tools off in both dialects, and a structured output wins over a
    /// required call.
    #[test]
    fn none_disables_tools_and_output_grammar_wins() {
        let msgs = [user("hi", None)];
        let tools = vec![tool("f")];
        for choice in [
            serde_json::json!("none"),
            serde_json::json!({"type": "none"}),
        ] {
            let p = build_prompt(
                &spec(&msgs, Some(&tools), Some(&choice)),
                Some(CHATML),
                "qwen3",
                false,
            );
            assert!(!p.tools_active);
            assert!(p.grammar.is_none());
        }
        let required = serde_json::json!("required");
        let mut s = spec(&msgs, Some(&tools), Some(&required));
        s.output_grammar = Some("json_object".to_string());
        let p = build_prompt(&s, Some(CHATML), "qwen3", false);
        assert_eq!(p.grammar.as_deref(), Some("json_object"));
    }

    /// The sentinel goes in front of the last user turn with images, and only when asked.
    #[test]
    fn the_vision_sentinel_marks_the_last_image_turn() {
        let msgs = [
            user("first", Some(vec!["a".to_string()])),
            user("second", Some(vec!["b".to_string()])),
        ];
        let p = build_prompt(&spec(&msgs, None, None), Some(CHATML), "qwen3-vl", true);
        assert_eq!(p.prompt.matches(QWEN35_IMAGE_SENTINEL).count(), 1);
        assert!(p.prompt.contains(&format!("{QWEN35_IMAGE_SENTINEL}second")));
        let p = build_prompt(&spec(&msgs, None, None), Some(CHATML), "qwen3-vl", false);
        assert!(!p.prompt.contains(QWEN35_IMAGE_SENTINEL));
    }

    /// A disabled preference closes the thinking block on every route that builds here.
    #[test]
    fn a_disabled_preference_closes_thinking() {
        let msgs = [user("hi", None)];
        let mut s = spec(&msgs, None, None);
        s.thinking = Some("disabled");
        let p = build_prompt(&s, Some(CHATML), "qwen3", false);
        assert!(
            p.prompt.ends_with("<think>\n\n</think>\n\n"),
            "{}",
            p.prompt
        );
    }

    #[test]
    fn effort_maps_to_a_thinking_preference() {
        assert_eq!(thinking_from_effort(Some("minimal")), Some("disabled"));
        assert_eq!(thinking_from_effort(Some("HIGH")), Some("high"));
        assert_eq!(thinking_from_effort(Some("other")), Some("enabled"));
        assert_eq!(thinking_from_effort(None), None);
    }
}
