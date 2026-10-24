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

/// What a model's files say about how it is prompted: the chat template it declares and
/// the format markers its vocabulary holds (`vocabulary_markers`).
#[derive(Clone, Debug, Default)]
pub(crate) struct ChatFormat {
    pub template: Option<String>,
    pub vocabulary: String,
}

impl ChatFormat {
    /// The template the model is prompted with: the declared one, else the fingerprint
    /// its vocabulary's markers choose.
    pub(crate) fn effective_template(&self) -> Option<String> {
        self.template
            .clone()
            .or_else(|| template_from_vocabulary(&self.vocabulary))
    }
}

/// The chat format a model file or checkpoint directory declares.
///
/// A GGUF file: its header's `tokenizer.chat_template`, and the vocabulary in
/// `tokenizer.ggml.tokens` plus the added tokens of an embedded tokenizer. A directory: the
/// `chat_template` of `tokenizer_config.json` or a `chat_template.jinja`, and the added
/// tokens of both tokenizer files; a directory holding neither is read from the first GGUF
/// file in it.
pub(crate) fn path_chat_format(path: &std::path::Path) -> ChatFormat {
    if path.is_dir() {
        dir_chat_format(path)
    } else {
        gguf_chat_format(path)
    }
}

fn gguf_chat_format(path: &std::path::Path) -> ChatFormat {
    use crate::inference::engine::llm_engine::EMBEDDED_TOKENIZER_KEY;
    let Ok(content) = crate::tensor::quantized::gguf_file::open_header(path) else {
        return ChatFormat::default();
    };
    let md = &content.metadata;
    let text = |key: &str| md.get(key).and_then(|v| v.to_string().ok());
    let embedded: Option<serde_json::Value> =
        text(EMBEDDED_TOKENIZER_KEY).and_then(|j| serde_json::from_str(j).ok());
    let tokens = md
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.to_vec().ok())
        .into_iter()
        .flatten()
        .filter_map(|t| t.to_string().ok().map(String::as_str))
        .chain(embedded.iter().flat_map(added_tokens));
    ChatFormat {
        template: text("tokenizer.chat_template").cloned(),
        vocabulary: vocabulary_markers(tokens),
    }
}

/// The model files of a folder, by name: the GGUF files that are not a vision projector or a
/// draft head.
fn model_files_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
                && !name.starts_with("mmproj")
                && !name.ends_with(".mtp.gguf")
        })
        .collect();
    files.sort();
    files
}

fn dir_chat_format(dir: &std::path::Path) -> ChatFormat {
    let read_json = |name: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(dir.join(name)).ok()?).ok()
    };
    let config = read_json("tokenizer_config.json");
    let tokenizer = read_json("tokenizer.json");
    // A checkpoint that ships its vocabulary as a tekken file lists its control tokens
    // there and nowhere else.
    let tekken = tokenizer
        .is_none()
        .then(|| read_json("tekken.json"))
        .flatten();
    let template = config
        .as_ref()
        .and_then(|c| declared_template(c.get("chat_template")?))
        .or_else(|| std::fs::read_to_string(dir.join("chat_template.jinja")).ok());
    if template.is_none() && config.is_none() && tokenizer.is_none() && tekken.is_none() {
        let mut ggufs = model_files_in(dir);
        if ggufs.is_empty() {
            // A repository that keeps each quantisation in a folder of its own.
            let mut folders: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            folders.sort();
            ggufs = folders
                .iter()
                .map(|f| model_files_in(f))
                .find(|files| !files.is_empty())
                .unwrap_or_default();
        }
        if let Some(first) = ggufs.first() {
            return gguf_chat_format(first);
        }
    }
    let tokens = config
        .iter()
        .chain(tokenizer.iter())
        .chain(tekken.iter())
        .flat_map(added_tokens);
    // A checkpoint whose tokenizer configuration declares no template is a chat model only
    // when it stops on a token that closes a turn; any other is prompted as given.
    let base = template.is_none()
        && config.is_some()
        && !stop_tokens(dir, config.as_ref(), tokenizer.as_ref())
            .iter()
            .any(|t| TURN_ENDS.contains(&t.as_str()));
    ChatFormat {
        template,
        vocabulary: match base {
            true => String::new(),
            false => vocabulary_markers(tokens),
        },
    }
}

/// The tokens that close an assistant's turn in the chat formats `FORMAT_MARKERS` names.
const TURN_ENDS: &[&str] = &[
    "<|im_end|>",
    "<end_of_turn>",
    "<turn|>",
    "<|return|>",
    "<|call|>",
    "<|end|>",
    "<|eot_id|>",
    "<|end_of_role|>",
];

/// The text of the tokens a checkpoint directory stops on: the `eos_token_id` of
/// `generation_config.json` and `config.json`, read through the added tokens of the tokenizer
/// files, and the `eos_token` of the tokenizer configuration.
fn stop_tokens(
    dir: &std::path::Path,
    config: Option<&serde_json::Value>,
    tokenizer: Option<&serde_json::Value>,
) -> Vec<String> {
    let read_json = |name: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(dir.join(name)).ok()?).ok()
    };
    let ids: Vec<u64> = ["generation_config.json", "config.json"]
        .iter()
        .filter_map(|f| read_json(f)?.get("eos_token_id").cloned())
        .flat_map(|v| match v {
            serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_u64()).collect(),
            v => v.as_u64().into_iter().collect::<Vec<_>>(),
        })
        .collect();
    let listed = tokenizer
        .and_then(|t| t.get("added_tokens")?.as_array())
        .into_iter()
        .flatten()
        .filter_map(|t| Some((t.get("id")?.as_u64()?, t.get("content")?.as_str()?)));
    let decoded = config
        .and_then(|c| c.get("added_tokens_decoder")?.as_object())
        .into_iter()
        .flatten()
        .filter_map(|(id, t)| Some((id.parse().ok()?, t.get("content")?.as_str()?)));
    let mut stops: Vec<String> = listed
        .chain(decoded)
        .filter(|(id, _)| ids.contains(id))
        .map(|(_, text)| text.to_string())
        .collect();
    let eos = config.and_then(|c| c.get("eos_token"));
    let eos = eos.and_then(|e| e.as_str().or_else(|| e.get("content")?.as_str()));
    stops.extend(eos.map(str::to_string));
    stops
}

/// A `chat_template` entry: a template, or a list of named ones of which `default` (else
/// the first) writes the conversation.
fn declared_template(v: &serde_json::Value) -> Option<String> {
    if let Some(t) = v.as_str() {
        return Some(t.to_string());
    }
    let named = v.as_array()?;
    let template = |e: &serde_json::Value| e.get("template")?.as_str().map(str::to_string);
    named
        .iter()
        .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
        .or_else(|| named.first())
        .and_then(template)
}

/// The text of a tokenizer file's added tokens: `added_tokens` in `tokenizer.json`,
/// `added_tokens_decoder` in `tokenizer_config.json`, `special_tokens` in a tekken file.
fn added_tokens(json: &serde_json::Value) -> impl Iterator<Item = &str> {
    let special = json
        .get("special_tokens")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|t| t.get("token_str").and_then(|c| c.as_str()));
    let listed = json
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten();
    let decoded = json
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
        .into_iter()
        .flat_map(|m| m.values());
    listed
        .chain(decoded)
        .filter_map(|t| t.get("content").and_then(|c| c.as_str()))
        .chain(special)
}

/// The rendered prompt and what the response side needs to read the answer back.
pub(crate) struct PreparedPrompt {
    pub prompt: String,
    pub tools_active: bool,
    pub tool_format: ToolFormat,
    pub grammar: Option<String>,
    /// Whether the template already opened the thinking block.
    pub thinking_opened: bool,
    /// Bytes of `prompt` the next turn of the conversation renders the same.
    pub stable_prefix: Option<usize>,
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

/// The messages with `sentinel` put in front of the last user turn that carries audio, the
/// only turn whose clip reaches the speech tower.
fn with_audio_sentinel(messages: &[Message], sentinel: &str) -> Vec<Message> {
    let mut msgs = messages.to_vec();
    if let Some(um) = msgs
        .iter_mut()
        .rev()
        .find(|m| m.role == "user" && m.audios.as_ref().is_some_and(|a| !a.is_empty()))
    {
        um.content = format!("{sentinel}{}", um.content);
    }
    msgs
}

/// The template a prompt is rendered with: the format's own, or, for a model prompted in
/// the bracket instruction format, that format's fingerprint from the vocabulary's markers,
/// which routes the render to its writer.
pub(crate) fn prompt_template(format: &ChatFormat, instruction_format: bool) -> Option<String> {
    instruction_format
        .then(|| instruction_markers(&format.vocabulary))
        .flatten()
        .or_else(|| format.effective_template())
}

/// The prompt of a transcription on a model prompted in the bracket instruction format: the
/// clip alone in the user turn, then the language when one is given and the transcription
/// marker, which is what makes the model write the words instead of answering them.
pub(crate) fn transcription_prompt(
    tower: &crate::inference::engine::model_backend::AudioInput,
    language: Option<&str>,
) -> String {
    let language = language.map(|l| format!("lang:{l}")).unwrap_or_default();
    format!(
        "[INST]{}{}[/INST]{language}[TRANSCRIBE]",
        tower.begin, tower.placeholder
    )
}

/// Whether any user turn carries audio.
pub(crate) fn carries_audio(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|m| m.role == "user" && m.audios.as_ref().is_some_and(|a| !a.is_empty()))
}

/// The clip the speech tower reads: the first clip of the last user turn that carries one.
pub(crate) fn last_user_audio(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .filter(|m| m.role == "user")
        .find_map(|m| m.audios.as_ref()?.first())
        .map(String::as_str)
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

/// Render a request against a template, the tool-call format read from the template and
/// then from `vocabulary`, the format markers the model's vocabulary holds. Pure: the same
/// inputs give the same prompt on every node, which is what lets a peer be asked about the
/// prompt this node would run.
pub(crate) fn build_prompt(
    spec: &PromptSpec<'_>,
    template: Option<&str>,
    vocabulary: &str,
    vision_sentinel: bool,
    audio_sentinel: Option<&str>,
) -> PreparedPrompt {
    let tools_active = crate::api::tool_calls::should_inject_tools(spec.tools, spec.tool_choice);
    let tool_format = crate::api::tool_calls::detect_tool_format(template, vocabulary);
    let mut owned: Option<Vec<Message>> = None;
    if vision_sentinel && carries_images(spec.messages) {
        owned = Some(with_vision_sentinel(spec.messages));
    }
    if let Some(sentinel) = audio_sentinel.filter(|_| carries_audio(spec.messages)) {
        owned = Some(with_audio_sentinel(
            owned.as_deref().unwrap_or(spec.messages),
            sentinel,
        ));
    }
    let messages: &[Message] = owned.as_deref().unwrap_or(spec.messages);
    // An answer a grammar constrains from its first token has no room to think: a prompt that
    // opens a thinking block would file the whole constrained answer as reasoning.
    let constrained = spec.output_grammar.is_some()
        || spec
            .tools
            .filter(|_| tools_active)
            .is_some_and(|_| crate::api::tool_calls::forced_call(spec.tool_choice).is_some());
    let thinking = if constrained {
        Some("disabled")
    } else {
        spec.thinking
    };
    let thinking_off = thinking == Some("disabled");
    let render = |messages: &[Message]| -> String {
        let mut prompt = match spec.tools.filter(|_| tools_active) {
            Some(tools) if template_takes_tools(template) => {
                format_chat_prompt_with_tools(messages, template, tools)
            }
            Some(tools) if template_is_harmony(template) => format_harmony(messages, Some(tools)),
            Some(tools) if template_is_mistral_tools(template) => {
                format_mistral_tools(messages, template.unwrap_or_default(), Some(tools))
            }
            Some(tools) => {
                let directive = crate::api::tool_calls::tool_choice_directive(spec.tool_choice);
                let flat = crate::api::tool_calls::flatten_messages(
                    messages,
                    tool_format,
                    tools,
                    directive.as_deref(),
                );
                format_chat_prompt_thinking(&flat, template, thinking_off)
            }
            None => format_chat_prompt_thinking(messages, template, thinking_off),
        };
        apply_thinking_preference(&mut prompt, thinking, template);
        prompt
    };
    let prompt = render(messages);
    let stable_prefix = stable_prefix_len(messages, &prompt, &render);
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
        stable_prefix,
    }
}

/// Bytes of `prompt` that the next turn of its conversation renders the same: the prompt up to
/// where the template opens the assistant's turn differently for an answer than for a turn to
/// be written. Found by rendering the conversation again with an answer after it and keeping
/// what the two have in common. `None` when the conversation already ends on an answer, or the
/// two have nothing to share.
fn stable_prefix_len(
    messages: &[Message],
    prompt: &str,
    render: &dyn Fn(&[Message]) -> String,
) -> Option<usize> {
    if messages.last().is_none_or(|m| m.role == "assistant") {
        return None;
    }
    let mut answered = messages.to_vec();
    answered.push(Message::new("assistant".to_string(), "ok".to_string()));
    let next = render(&answered);
    let mut shared = prompt
        .bytes()
        .zip(next.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    while shared > 0 && !prompt.is_char_boundary(shared) {
        shared -= 1;
    }
    (shared > 0).then_some(shared)
}

impl APIServer {
    /// A model's chat format: the one cached on its loaded engine, else the one read from
    /// its files (cached on the engine for next time).
    pub(crate) async fn chat_format_for(&self, model_name: &str) -> ChatFormat {
        let cached = {
            let engines = self.engines.read().await;
            engines
                .iter()
                .find(|e| e.model_id == model_name)
                .and_then(|e| e.chat_format.clone())
        };
        if let Some(format) = cached {
            return format;
        }
        let read = self.read_chat_format(model_name);
        let mut engines = self.engines.write().await;
        if let Some(entry) = engines.iter_mut().find(|e| e.model_id == model_name) {
            entry.chat_format = Some(read.clone());
        }
        read
    }

    /// The chat template a model is prompted with: its declared one, else the fingerprint
    /// its vocabulary's markers choose.
    pub(crate) async fn chat_template_for(&self, model_name: &str) -> Option<String> {
        self.chat_format_for(model_name).await.effective_template()
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
        let format = self.chat_format_for(model_name).await;
        let vision = match engine {
            Some(e) if carries_images(spec.messages) => e.is_qwen35_vision().await,
            _ => false,
        };
        let tower = match engine {
            Some(e) => e.audio_input().await,
            None => None,
        };
        let template = prompt_template(
            &format,
            tower.as_ref().is_some_and(|a| a.instruction_format),
        );
        // What opens a clip and marks where its embeddings go, as the loaded model's tower
        // declares it.
        let audio = tower
            .filter(|_| carries_audio(spec.messages))
            .map(|a| format!("{}{}", a.begin, a.placeholder));
        build_prompt(
            spec,
            template.as_deref(),
            &format.vocabulary,
            vision,
            audio.as_deref(),
        )
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
                "",
                false,
                None,
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
                "",
                false,
                None,
            );
            let g = p.grammar.expect("a named call must set a grammar");
            assert!(g.contains("get_time") && !g.contains("get_weather"), "{g}");
        }
        let auto = serde_json::json!("auto");
        let p = build_prompt(
            &spec(&msgs, Some(&tools), Some(&auto)),
            Some(CHATML),
            "",
            false,
            None,
        );
        assert!(p.grammar.is_none());
    }

    /// The audio sentinel goes in front of the last user turn with a clip, and only when
    /// the model's tower declares one.
    #[test]
    fn the_audio_sentinel_leads_the_last_turn_with_a_clip() {
        let mut clip = user("what is said?", None);
        clip.audios = Some(vec!["AAAA".to_string()]);
        let msgs = [user("hello", None), clip];
        let p = build_prompt(
            &spec(&msgs, None, None),
            Some(CHATML),
            "",
            false,
            Some("<a><b>"),
        );
        assert_eq!(p.prompt.matches("<a><b>").count(), 1);
        assert!(p.prompt.contains("<a><b>what is said?"));
        let p = build_prompt(&spec(&msgs, None, None), Some(CHATML), "", false, None);
        assert!(!p.prompt.contains("<a><b>"));
        assert_eq!(last_user_audio(&msgs), Some("AAAA"));
    }

    /// A model prompted in the bracket instruction format is rendered by that format's
    /// writer, not by a file template that writes a default system turn; any other model
    /// keeps its file's template.
    #[test]
    fn an_instruction_format_model_ignores_the_files_template() {
        let format = ChatFormat {
            template: Some(
                "{% set default_system_message = 'You are an agent.' %}{{ bos_token }}\
                 [SYSTEM_PROMPT]{{ default_system_message }}[/SYSTEM_PROMPT]\
                 {% for m in messages %}[INST]{{ m.content }}[/INST]{% endfor %}"
                    .to_string(),
            ),
            vocabulary: "[INST][SYSTEM_PROMPT][TOOL_CALLS]".to_string(),
        };
        let mut clip = user("Transcribe this audio", None);
        clip.audios = Some(vec!["AAAA".to_string()]);
        let msgs = [clip];
        let render = |instruction_format: bool| {
            let template = prompt_template(&format, instruction_format);
            build_prompt(
                &spec(&msgs, None, None),
                template.as_deref(),
                &format.vocabulary,
                false,
                Some("[BEGIN_AUDIO][AUDIO]"),
            )
            .prompt
        };
        assert_eq!(
            render(true),
            "[INST][BEGIN_AUDIO][AUDIO]Transcribe this audio[/INST]"
        );
        assert!(render(false).contains("You are an agent."));
    }

    /// A transcription on an instruction-format model holds the clip alone, the language and
    /// the transcription marker; without a language the marker follows the turn directly.
    #[test]
    fn a_transcription_prompt_ends_in_the_marker() {
        let tower = crate::inference::engine::model_backend::AudioInput {
            sample_rate: 16_000,
            placeholder: "[AUDIO]".to_string(),
            begin: "[BEGIN_AUDIO]".to_string(),
            instruction_format: true,
        };
        assert_eq!(
            transcription_prompt(&tower, Some("en")),
            "[INST][BEGIN_AUDIO][AUDIO][/INST]lang:en[TRANSCRIBE]"
        );
        assert_eq!(
            transcription_prompt(&tower, None),
            "[INST][BEGIN_AUDIO][AUDIO][/INST][TRANSCRIBE]"
        );
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
                "",
                false,
                None,
            );
            assert!(!p.tools_active);
            assert!(p.grammar.is_none());
        }
        let required = serde_json::json!("required");
        let mut s = spec(&msgs, Some(&tools), Some(&required));
        s.output_grammar = Some("json_object".to_string());
        let p = build_prompt(&s, Some(CHATML), "", false, None);
        assert_eq!(p.grammar.as_deref(), Some("json_object"));
    }

    /// The sentinel goes in front of the last user turn with images, and only when asked.
    #[test]
    fn the_vision_sentinel_marks_the_last_image_turn() {
        let msgs = [
            user("first", Some(vec!["a".to_string()])),
            user("second", Some(vec!["b".to_string()])),
        ];
        let p = build_prompt(&spec(&msgs, None, None), Some(CHATML), "", true, None);
        assert_eq!(p.prompt.matches(QWEN35_IMAGE_SENTINEL).count(), 1);
        assert!(p.prompt.contains(&format!("{QWEN35_IMAGE_SENTINEL}second")));
        let p = build_prompt(&spec(&msgs, None, None), Some(CHATML), "", false, None);
        assert!(!p.prompt.contains(QWEN35_IMAGE_SENTINEL));
    }

    /// A disabled preference closes the thinking block of a template that has one, and
    /// leaves a template without one as it is.
    #[test]
    fn a_disabled_preference_closes_thinking() {
        let msgs = [user("hi", None)];
        let mut s = spec(&msgs, None, None);
        s.thinking = Some("disabled");
        let thinking = format!("{{# enable_thinking #}}{CHATML}");
        let p = build_prompt(&s, Some(&thinking), "", false, None);
        assert!(
            p.prompt.ends_with("<think>\n\n</think>\n\n"),
            "{}",
            p.prompt
        );
        let p = build_prompt(&s, Some(CHATML), "", false, None);
        assert!(
            p.prompt.ends_with("<|im_start|>assistant\n"),
            "{}",
            p.prompt
        );
    }

    /// A grammar-constrained answer is the content, whatever the template does by default:
    /// the prompt closes the thinking block so the constrained text is not read as reasoning.
    #[test]
    fn a_grammar_closes_thinking_for_the_answer_it_constrains() {
        let msgs = [user("hi", None)];
        let mut s = spec(&msgs, None, None);
        s.output_grammar = Some("json_object".to_string());
        let thinking = format!("{{# enable_thinking #}}{CHATML}");
        let p = build_prompt(&s, Some(&thinking), "", false, None);
        assert!(!p.thinking_opened, "{}", p.prompt);
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

    /// A repository that keeps each quantisation in a folder of its own is prompted with the
    /// template its files carry.
    #[test]
    fn a_template_in_a_quantisation_folder_is_found() {
        use crate::tensor::quantized::gguf_file::Value;
        let dir = std::env::temp_dir().join(format!("loken-quant-dir-{}", std::process::id()));
        let folder = dir.join("Q4");
        std::fs::create_dir_all(&folder).unwrap();
        let header = [
            (
                "general.architecture".to_string(),
                Value::String("llama".into()),
            ),
            (
                "tokenizer.chat_template".to_string(),
                Value::String("{{ messages[0].content }}".into()),
            ),
        ];
        crate::tensor::gguf_write::write_gguf_with_metadata(&folder.join("m.gguf"), &header, &[])
            .unwrap();
        let format = path_chat_format(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(
            format.template.as_deref(),
            Some("{{ messages[0].content }}")
        );
    }

    /// A checkpoint whose only vocabulary file is a tekken file is prompted in the
    /// instruction format its control tokens name.
    #[test]
    fn a_tekken_vocabulary_chooses_the_instruction_format() {
        let dir = std::env::temp_dir().join(format!("loken-tekken-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tekken = serde_json::json!({
            "special_tokens": [
                {"rank": 1, "token_str": "<s>", "is_control": true},
                {"rank": 3, "token_str": "[INST]", "is_control": true},
                {"rank": 4, "token_str": "[/INST]", "is_control": true},
                {"rank": 17, "token_str": "[SYSTEM_PROMPT]", "is_control": true},
            ],
            "vocab": [],
        });
        std::fs::write(dir.join("tekken.json"), tekken.to_string()).unwrap();
        let format = path_chat_format(&dir);
        std::fs::remove_dir_all(&dir).ok();
        let template = format
            .effective_template()
            .expect("a template from the vocabulary");
        let p = build_prompt(
            &spec(&[user("hi", None)], None, None),
            Some(&template),
            "",
            false,
            None,
        );
        assert!(p.prompt.contains("[INST]hi[/INST]"), "{}", p.prompt);
    }

    /// A checkpoint whose tokenizer configuration declares no template is prompted as given
    /// when it stops on the plain end of text, and in its vocabulary's format when it stops
    /// on the token that closes that format's turn.
    #[test]
    fn a_checkpoint_declaring_no_template_is_prompted_by_its_stop_tokens() {
        let format_with = |eos: u64| {
            let dir = std::env::temp_dir().join(format!("loken-base-{eos}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let config = serde_json::json!({
                "eos_token": "<eos>",
                "added_tokens_decoder": {
                    "1": {"content": "<eos>", "special": true},
                    "106": {"content": "<start_of_turn>", "special": true},
                    "107": {"content": "<end_of_turn>", "special": true},
                },
            });
            std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
            let generation = serde_json::json!({"eos_token_id": [1, eos]});
            std::fs::write(dir.join("generation_config.json"), generation.to_string()).unwrap();
            let format = path_chat_format(&dir);
            std::fs::remove_dir_all(&dir).ok();
            format
        };
        assert_eq!(format_with(1).effective_template(), None);
        assert!(format_with(107)
            .effective_template()
            .is_some_and(|t| t.contains("<start_of_turn>")));
    }

    /// The part of a prompt the conversation's next turn renders the same ends where the template
    /// opens the assistant's turn differently for an answer than for a turn to be written.
    #[test]
    fn the_shared_part_ends_where_the_assistants_turn_opens() {
        let render = |messages: &[Message]| -> String {
            let mut out = String::new();
            for m in messages {
                if m.role == "assistant" {
                    out += &format!(
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n",
                        m.content
                    );
                } else {
                    out += &format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content);
                }
            }
            out + "<|im_start|>assistant\n<think>\n"
        };
        let first = [user("question", None)];
        let prompt = render(&first);
        let shared = stable_prefix_len(&first, &prompt, &render).unwrap();
        assert_eq!(
            &prompt[..shared],
            "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );

        // The next turn's prompt begins with the shared part, whatever the answer was.
        let mut next = first.to_vec();
        next.push(Message::new(
            "assistant".to_string(),
            "a long answer".to_string(),
        ));
        next.push(user("more", None));
        assert!(render(&next).starts_with(&prompt[..shared]));
        let again = stable_prefix_len(&next, &render(&next), &render).unwrap();
        assert!(render(&next)[..again].starts_with(&prompt[..shared]));
    }

    /// A conversation that already ends on an answer has no turn to share.
    #[test]
    fn a_conversation_ending_on_an_answer_shares_nothing() {
        let render = |_: &[Message]| String::from("x");
        let ended = [Message::new("assistant".to_string(), "done".to_string())];
        assert_eq!(stable_prefix_len(&ended, "x", &render), None);
    }
}
