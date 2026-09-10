//! Chat-prompt templating: template fingerprint routing plus the
//! per-family formatters (ChatML/Gemma/Phi/GLM4/DeepSeek/...).

use super::*;

/// Does a raw /api/generate prompt already contain markers from *this* model's
/// chat template family? If yes, the caller pre-templated and we must not re-wrap.
///
/// We detect the template family from a distinctive substring of the template
/// string, then look only for that family's markers in the prompt - this avoids
/// false positives where the prompt legitimately contains markers from an
/// unrelated family (e.g., a DeepSeek prompt quoting `[INST]` as literal text).
pub(crate) fn prompt_already_templated(prompt: &str, template: &str) -> bool {
    // (template fingerprint, prompt markers that identify a pre-wrapped prompt).
    // Fingerprints mirror the family detection in `format_chat_prompt`.
    const FAMILIES: &[(&str, &[&str])] = &[
        ("<｜User｜>", &["<｜User｜>", "<｜Assistant｜>"]),
        ("<｜Assistant｜>", &["<｜User｜>", "<｜Assistant｜>"]),
        ("<|im_start|>", &["<|im_start|>", "<|im_end|>"]),
        ("<start_of_turn>", &["<start_of_turn>", "<end_of_turn>"]),
        ("<|end|>", &["<|user|>", "<|assistant|>", "<|system|>"]),
        ("[gMASK]", &["[gMASK]", "<sop>"]),
        ("[INST]", &["[INST]", "[/INST]"]),
        // Moondream Q/A template - manifest contains literal
        // "Question:" + "Answer:" markers. Pre-templated prompts that
        // already include those should not be re-wrapped.
        ("Question:", &["Question:", "Answer:"]),
    ];
    FAMILIES
        .iter()
        .find(|(fp, _)| template.contains(fp))
        .is_some_and(|(_, markers)| markers.iter().any(|m| prompt.contains(m)))
}

/// Synthesize a chat template fingerprint when the Ollama manifest omits
/// the template layer. We return a minimal string containing the
/// distinctive marker that `format_chat_prompt` looks for, so it routes
/// to the right `format_*` family. Examples in the wild:
/// - gemma3/gemma4 sometimes ship without a template layer.
/// - phi3/phi4 - `<|user|>` / `<|end|>` - usually has template.
/// - qwen / chatml - `<|im_start|>` - usually has template.
///
/// Add new fallbacks here as discovered.
pub(crate) fn infer_template_from_model_name(model_id: &str) -> Option<String> {
    let lower = model_id.to_lowercase();
    let pick = |fingerprint: &'static str| Some(fingerprint.to_string());
    if lower.contains("gemma") {
        // gemma4 (entire family - confirmed gemma4:latest AND gemma4:26b)
        // ships the Harmony-style vocab (asymmetric `<|turn>` / `<turn|>`
        // tags at IDs 105/106) instead of the legacy `<start_of_turn>` /
        // `<end_of_turn>`. The legacy template strings BPE-split into 7
        // pieces ("<", "start", "_", "of", "_", "turn", ">") - the model
        // never sees a real special token and emits XML-style closing
        // tags in response (`</start_of_turn>` etc).
        // gemma3 and earlier still use the legacy vocab.
        if lower.contains("gemma4") {
            return pick("<|turn>");
        }
        // Gemma3/2/legacy: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
        return pick("<start_of_turn>");
    }
    if lower.contains("qwen") || lower.contains("chatml") {
        return pick("<|im_start|>");
    }
    // lfm2 / lfm2moe (LiquidAI) ship no template layer (Ollama uses a built-in
    // Go renderer). They use ChatML (`<|im_start|>`/`<|im_end|>`, BOS
    // `<|startoftext|>` added by the tokenizer). Without this they fall to the
    // [INST] default and the model echoes/loops.
    if lower.contains("lfm2") {
        return pick("<|im_start|>");
    }
    if lower.contains("moondream") {
        // Moondream Q/A template: ` Question: <prompt>\n\n Answer: <response>`.
        // Detection token "Question:" matches the format_moondream_qa dispatch
        // in format_chat_prompt. Without this fallback, a moondream variant
        // whose Modelfile template isn't reachable would get the [INST]
        // default -> base-phi2 LM emits gibberish on instruction-shaped
        // prompts.
        return pick("Question: Answer:");
    }
    // Llama 3 and the families that adopted its header block. Only a fallback: a model
    // whose manifest declares its template is formatted from that declaration.
    if lower.contains("llama3") || lower.contains("llama-3") {
        return pick("<|start_header_id|>");
    }
    if lower.contains("phi3") || lower.contains("phi4") || lower.contains("phi") {
        return pick("<|end|><|user|><|assistant|><|system|>");
    }
    if lower.contains("deepseek") {
        return pick("<｜User｜><｜Assistant｜>");
    }
    if lower.contains("glm") {
        return pick("[gMASK]");
    }
    None
}

/// Format chat messages into a prompt string using Mistral/Llama chat template.
/// Combines system message + conversation history into a single prompt.
/// Apply a model-specific chat template to messages.
/// Close the assistant turn with an empty thinking block when the caller asked for no
/// reasoning.
///
/// What `enable_thinking=false` renders to for the ChatML reasoning families: an empty
/// `<think></think>` right after the assistant opener, which the model reads as "the
/// thinking is already done" and answers directly. Applied only when the rendered prompt
/// really ends with that opener, so a non-reasoning template is left alone.
///
/// Here rather than in a handler because both the chat and the generate endpoints have to
/// honour the same request field, and the one that did not was silently reasoning anyway.
pub(crate) fn apply_thinking_preference(prompt: &mut String, thinking: Option<&str>) {
    const OPENER: &str = "<|im_start|>assistant\n";
    const HARMONY_SYSTEM: &str = "<|start|>system<|message|>";
    // gpt-oss reads its effort from a `Reasoning:` line of the system turn and cannot
    // stop reasoning altogether: `disabled` becomes its lowest level.
    if let Some(at) = prompt.find(HARMONY_SYSTEM) {
        let level = match thinking {
            Some("low") | Some("medium") | Some("high") => thinking,
            Some("disabled") | Some("none") | Some("minimal") => Some("low"),
            _ => None,
        };
        if let Some(level) = level {
            let line = format!("Reasoning: {level}");
            let head = at + HARMONY_SYSTEM.len();
            if let Some(rel) = prompt[head..].find("Reasoning: ") {
                let start = head + rel;
                let end = prompt[start..]
                    .find('\n')
                    .map(|n| start + n)
                    .unwrap_or(prompt.len());
                prompt.replace_range(start..end, &line);
            } else {
                prompt.insert_str(head, &format!("{line}\n"));
            }
        }
        return;
    }
    if thinking == Some("disabled") && prompt.ends_with(OPENER) {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
}

/// Renders the Go template of an Ollama modelfile for one exchange: the `.System`,
/// `.Prompt` and `.Response` fields, `if`/`else`/`end` on them, and the `-` trims.
/// `None` for any other construct - `range`, `with`, `.Messages`, functions - which a
/// caller then answers with the model's own template.
pub(crate) fn render_go_template(tmpl: &str, system: Option<&str>, prompt: &str) -> Option<String> {
    let field = |name: &str| -> Option<&str> {
        match name {
            ".System" => Some(system.unwrap_or("")),
            ".Prompt" => Some(prompt),
            ".Response" => Some(""),
            _ => None,
        }
    };
    let mut out = String::new();
    // Each `if` pushes whether its branch is live; `else` flips it; `end` pops.
    let mut live: Vec<bool> = Vec::new();
    let mut trim_next = false;
    let mut rest = tmpl;
    loop {
        let Some(open) = rest.find("{{") else {
            let tail = if trim_next { rest.trim_start() } else { rest };
            if live.iter().all(|&l| l) {
                out.push_str(tail);
            }
            break;
        };
        let text = &rest[..open];
        let text = if trim_next { text.trim_start() } else { text };
        let after = &rest[open + 2..];
        let close = after.find("}}")?;
        let mut action = after[..close].trim();
        let trim_before = action.starts_with('-');
        trim_next = action.ends_with('-');
        action = action.trim_start_matches('-').trim_end_matches('-').trim();
        let text = if trim_before { text.trim_end() } else { text };
        if live.iter().all(|&l| l) {
            out.push_str(text);
        }
        let emitting = live.iter().all(|&l| l);
        if let Some(cond) = action.strip_prefix("if ") {
            let value = field(cond.trim())?;
            live.push(!value.is_empty());
        } else if action == "else" {
            let last = live.last_mut()?;
            *last = !*last;
        } else if action == "end" {
            live.pop()?;
        } else if action.starts_with('.') {
            let value = field(action)?;
            if emitting {
                out.push_str(value);
            }
        } else {
            return None;
        }
        rest = &after[close + 2..];
    }
    if !live.is_empty() {
        return None;
    }
    Some(out)
}

/// Renders a Jinja chat template - the one the weights file carries under
/// `tokenizer.chat_template`, which is the upstream source of truth for every
/// packager rather than one distributor's wrapper.
///
/// Everything below this function recognises a template by a characteristic
/// token and reproduces its shape by hand, falling through to a generic
/// `[INST]` form when nothing matches. That approximation is why models whose
/// format nobody had fingerprinted closed their turn after a dozen tokens: we
/// held the right template and rendered something else. Evaluating it removes
/// the whole class instead of adding one more fingerprint per model that fails.
/// The Python string methods chat templates call on `message.content` and friends -
/// `startswith`, `split`, `rstrip` - which the template language does not carry. A
/// template that cannot call them renders nothing, and the model silently receives an
/// approximation of its own format.
/// The dict methods a chat template reaches for, on anything that is not a string.
///
/// Templates are written against Python dicts and call `keys`, `values`, `items` and
/// `get` on the tool schemas they are handed. minijinja has none of them, so a template
/// that walks a schema renders nothing and the model receives a reconstruction of its own
/// format instead.
fn python_mapping_methods(
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error, ErrorKind, Value};
    let pairs = || -> Result<Vec<(Value, Value)>, Error> {
        let mut out = Vec::new();
        for key in value.try_iter()? {
            let item = value.get_item(&key)?;
            out.push((key, item));
        }
        Ok(out)
    };
    Ok(match method {
        "keys" => Value::from(value.try_iter()?.collect::<Vec<_>>()),
        "values" => Value::from(pairs()?.into_iter().map(|(_, v)| v).collect::<Vec<_>>()),
        "items" => Value::from(
            pairs()?
                .into_iter()
                .map(|(k, v)| Value::from(vec![k, v]))
                .collect::<Vec<_>>(),
        ),
        // Python returns the default, or None, rather than raising.
        "get" => {
            let key = args.first().cloned().unwrap_or_default();
            match value.get_item(&key) {
                Ok(v) if !v.is_undefined() => v,
                _ => args.get(1).cloned().unwrap_or_default(),
            }
        }
        _ => {
            return Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("{} has no method named {method}", value.kind()),
            ))
        }
    })
}

fn python_string_methods(
    _state: &minijinja::State,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error, ErrorKind, Value};
    let Some(s) = value.as_str() else {
        return python_mapping_methods(value, method, args);
    };
    let text = |i: usize| -> Result<&str, Error> {
        args.get(i).and_then(Value::as_str).ok_or_else(|| {
            Error::new(
                ErrorKind::MissingArgument,
                format!("{method}: argument {} must be a string", i + 1),
            )
        })
    };
    let chars = |i: usize| args.get(i).and_then(Value::as_str);
    // Python strips whitespace without an argument, any of the given characters with one.
    let cut = |c: char| match chars(0) {
        Some(set) => set.contains(c),
        None => c.is_whitespace(),
    };
    Ok(match method {
        "startswith" => Value::from(s.starts_with(text(0)?)),
        "endswith" => Value::from(s.ends_with(text(0)?)),
        "strip" => Value::from(s.trim_matches(cut)),
        "lstrip" => Value::from(s.trim_start_matches(cut)),
        "rstrip" => Value::from(s.trim_end_matches(cut)),
        "lower" => Value::from(s.to_lowercase()),
        "upper" => Value::from(s.to_uppercase()),
        "replace" => Value::from(s.replace(text(0)?, text(1)?)),
        "find" => Value::from(s.find(text(0)?).map(|i| i as i64).unwrap_or(-1)),
        "count" => Value::from(s.matches(text(0)?).count()),
        "split" => {
            let limit = args.get(1).and_then(|v| i64::try_from(v.clone()).ok());
            let parts: Vec<Value> = match (chars(0), limit) {
                (Some(sep), Some(n)) if n >= 0 => {
                    s.splitn(n as usize + 1, sep).map(Value::from).collect()
                }
                (Some(sep), _) => s.split(sep).map(Value::from).collect(),
                (None, _) => s.split_whitespace().map(Value::from).collect(),
            };
            Value::from(parts)
        }
        "join" => {
            let items: Result<Vec<String>, Error> = args
                .first()
                .ok_or_else(|| Error::new(ErrorKind::MissingArgument, "join: nothing to join"))?
                .try_iter()?
                .map(|v| Ok(v.to_string()))
                .collect();
            Value::from(items?.join(s))
        }
        _ => {
            return Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("string has no method named {method}"),
            ))
        }
    })
}

fn render_jinja_template(
    tmpl: &str,
    messages: &[Message],
    add_generation_prompt: bool,
    tools: Option<&[Tool]>,
) -> Option<String> {
    use minijinja::{context, Environment, Value};
    // Messages go in whole: a tool-using template reads `tool_calls`, `tool_call_id`
    // and `name` off them, with the call's arguments as an object, the way the
    // upstream templates were written against.
    let msgs: Vec<Value> = messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::json!({"role": m.role, "content": m.content});
            if let Some(calls) = m.tool_calls.as_ref() {
                let calls: Vec<serde_json::Value> = calls
                    .iter()
                    .map(|c| {
                        let name = c
                            .function
                            .as_ref()
                            .map(|f| f.name.clone())
                            .unwrap_or_default();
                        let raw = c
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.clone())
                            .unwrap_or_default();
                        let arguments = serde_json::from_str::<serde_json::Value>(&raw)
                            .unwrap_or(serde_json::Value::String(raw));
                        serde_json::json!({
                            "id": c.id,
                            "type": c.r#type,
                            "function": {"name": name, "arguments": arguments},
                            "name": name,
                            "arguments": arguments,
                        })
                    })
                    .collect();
                obj["tool_calls"] = serde_json::Value::Array(calls);
            }
            if let Some(id) = m.tool_call_id.as_ref() {
                obj["tool_call_id"] = serde_json::json!(id);
            }
            if let Some(n) = m.name.as_ref() {
                obj["name"] = serde_json::json!(n);
            }
            Value::from_serialize(obj)
        })
        .collect();
    let tools_value: Option<Value> = tools.map(|t| {
        Value::from_serialize(
            t.iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": tool.r#type,
                        "function": {
                            "name": tool.function.as_ref().map(|f| f.name.clone()),
                            "description": tool.function.as_ref().and_then(|f| f.description.clone()),
                            "parameters": tool.function.as_ref().and_then(|f| f.parameters.clone()),
                        }
                    })
                })
                .collect::<Vec<_>>(),
        )
    });
    let mut env = Environment::new();
    env.set_unknown_method_callback(python_string_methods);
    // Chat templates call this to reject malformed conversations. Without it the
    // render fails outright, so map it to an error the caller turns into a
    // fallback rather than a panic.
    // Functions and methods these templates reach for that minijinja does not
    // ship. Each was named by the fallback log rather than guessed: a template
    // that cannot call them renders nothing and the model silently receives an
    // approximation of its own format.
    // Templates serialise a schema fragment straight into the prompt with it. Without
    // it the tool block renders nothing.
    env.add_filter(
        "tojson",
        |v: minijinja::Value| -> Result<String, minijinja::Error> {
            serde_json::to_string(&v).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })
        },
    );
    env.add_function("strftime_now", |fmt: String| -> String {
        // The templates use it to stamp a date into the system prompt. The exact
        // day does not change the model's behaviour, and reading the clock here
        // would make a prompt - and therefore a greedy answer - depend on when it
        // was sent, which no test could then reproduce.
        let _ = fmt;
        "01 Jan 2025".to_string()
    });
    env.add_filter("startswith", |s: String, prefix: String| {
        s.starts_with(&prefix)
    });
    env.add_filter(
        "get",
        |m: minijinja::Value, key: String| -> minijinja::Value {
            m.get_item(&minijinja::Value::from(key)).unwrap_or_default()
        },
    );
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    if let Err(e) = env.add_template("chat", tmpl) {
        tracing::warn!(
            "chat template does not parse ({e}); falling back to a reconstructed \
                        format, which is an approximation of what the model asked for"
        );
        return None;
    }
    let t = match env.get_template("chat") {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("chat template vanished after parsing ({e}); falling back");
            return None;
        }
    };
    // Undefined rather than none when there are no tools. Every template guards with
    // `if not tools is defined`, and a none that is defined walks straight past that
    // guard into `tools | length`, which is where the render dies.
    let tools_value = tools_value.unwrap_or(Value::UNDEFINED);
    match t.render(context! {
        messages => msgs,
        add_generation_prompt => add_generation_prompt,
        tools => tools_value,
    }) {
        Ok(r) => Some(r),
        Err(e) => {
            // Silence here is the defect this reports. A template that fails to
            // render sends the model a hand-rolled approximation instead of the
            // format it declared, and the only visible symptom is an answer that
            // stops after a few tokens - which reads as a model quirk, not a bug.
            tracing::warn!(
                "chat template failed to render ({e}); falling back to a \
                            reconstructed format - the model will receive something \
                            other than the format it declares"
            );
            None
        }
    }
}

/// Detects the template format from characteristic tokens and applies it.
/// Falls back to a generic Llama-style format if no template is provided.
/// Whether a Jinja template renders tools itself, so a conversation with tools can be
/// given to it whole instead of flattened into text.
pub(crate) fn template_takes_tools(template: Option<&str>) -> bool {
    template.is_some_and(|t| t.contains("{%") && t.contains("tools"))
}

/// `format_chat_prompt` with the tools handed to a template that renders them.
pub(crate) fn format_chat_prompt_with_tools(
    messages: &[Message],
    template: Option<&str>,
    tools: &[Tool],
) -> String {
    if let Some(tmpl) = template {
        if template_takes_tools(Some(tmpl)) {
            if let Some(rendered) = render_jinja_template(tmpl, messages, true, Some(tools)) {
                return rendered;
            }
        }
    }
    format_chat_prompt(messages, template)
}

pub(crate) fn format_chat_prompt(messages: &[Message], template: Option<&str>) -> String {
    // Extract system prompt and last user message for legacy templates
    let system = messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .unwrap_or("");
    let prompt = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    if let Some(tmpl) = template {
        // A Jinja template is the model's own statement of its format, so
        // evaluate it rather than pattern-match it. `{%` is the discriminator:
        // Jinja has statement blocks, the Go templates a distributor wraps its
        // models in do not.
        if tmpl.contains("{%") {
            if let Some(rendered) = render_jinja_template(tmpl, messages, true, None) {
                return rendered;
            }
        }
        // Detect template type from characteristic tokens
        if tmpl.contains("<|im_start|>") {
            // ChatML: Qwen, InternLM, SmolLM, StableLM
            return format_chatml(messages);
        } else if tmpl.contains("<|turn>") {
            // gemma4:26b Harmony-style vocab
            return format_gemma_harmony(messages);
        } else if tmpl.contains("<start_of_turn>") {
            // Gemma
            return format_gemma(messages);
        } else if tmpl.contains("<|system|>") && tmpl.contains("<|end|>") {
            // Phi
            return format_phi(messages);
        } else if tmpl.contains("[gMASK]") {
            // GLM4
            return format_glm4(system, prompt);
        } else if tmpl.contains("<｜User｜>") || tmpl.contains("<｜Assistant｜>") {
            // DeepSeek (R1, V3): unicode pipe tokens
            return format_deepseek(messages);
        } else if tmpl.contains("Question:") && tmpl.contains("Answer:") {
            // Moondream-style "Question: ... Answer:" template. Matches the
            // Ollama moondream manifest template exactly:
            //   {{ if .Prompt }} Question: {{ .Prompt }}\n\n{{ end }} Answer: {{ .Response }}
            // The default [INST] fallback produces gibberish on phi2-base
            // (moondream's LM) because base phi2 wasn't taught the chat
            // markers - so this branch is required for coherent captioning.
            return format_moondream_qa(messages);
        } else if tmpl.contains("<|start_header_id|>") {
            // Llama 3.x: a header block per turn. Without this branch the declared
            // template matches nothing and the message falls through to the [INST]
            // default below - which is Llama 2's format, from a different family. The
            // model then echoes the markers it was never taught into its own answer.
            return format_llama3(messages);
        } else if tmpl.contains("<|start_of_role|>") {
            // IBM Granite 3.x: <|start_of_role|>{role}<|end_of_role|>{content}
            // <|end_of_text|> per turn, then an assistant role opener. The default
            // [INST] fallback feeds Granite alien tokens: the 1B MoE degenerates to
            // garbage (echoes [/INST]); the larger dense model tolerates it but still
            // drifts from the reference.
            return format_granite(messages);
        }
        // If template doesn't match known patterns, fall through to default
    }

    // Default: Llama/Mistral [INST] format
    let mut out = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {}
            // OpenAI's "developer" role (o1-mini and later) is
            // semantically a system-prompt extension. Fold its content
            // into the next user turn the same way the legacy system
            // prefix works.
            "developer" => {}
            "user" => {
                if system.is_empty() || !out.is_empty() {
                    out.push_str(&format!("[INST] {} [/INST]", msg.content));
                } else {
                    out.push_str(&format!("[INST] {}\n\n{} [/INST]", system, msg.content));
                }
            }
            "assistant" => out.push_str(&msg.content),
            // tool / function call results - fold as a user-turn so
            // the content reaches the model instead of being silently
            // dropped. Tag the role so the model sees it as observed
            // tool output rather than user input.
            other => {
                out.push_str(&format!("[INST] ({other}) {} [/INST]", msg.content));
            }
        }
    }
    if out.is_empty() {
        prompt.to_string()
    } else {
        out
    }
}

/// ChatML format: <|im_start|>role\ncontent<|im_end|>
/// IBM Granite 3.x chat format:
///   <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n   (per turn)
///   <|start_of_role|>assistant<|end_of_role|>                           (generation opener)
/// The final assistant turn (if any) is a continuation: no closing token, no opener.
/// `tool` maps to `tool_response`, `developer` folds into `system`.
/// Llama 3.x: `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`,
/// closed by an empty assistant header the model continues from.
///
/// The opening `<|begin_of_text|>` is left to the tokenizer, which adds it.
fn format_llama3(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "developer" => "system",
            "tool" => "ipython",
            r => r,
        };
        out.push_str("<|start_header_id|>");
        out.push_str(role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(msg.content.trim());
        out.push_str("<|eot_id|>");
    }
    if messages
        .last()
        .map(|m| m.role != "assistant")
        .unwrap_or(true)
    {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    out
}

fn format_granite(messages: &[Message]) -> String {
    let mut out = String::new();
    let last = messages.len().saturating_sub(1);
    for (i, msg) in messages.iter().enumerate() {
        let role = match msg.role.as_str() {
            "tool" => "tool_response",
            "developer" => "system",
            r => r,
        };
        out.push_str("<|start_of_role|>");
        out.push_str(role);
        out.push_str("<|end_of_role|>");
        out.push_str(&msg.content);
        if !(i == last && role == "assistant") {
            out.push_str("<|end_of_text|>\n");
        }
    }
    if messages
        .last()
        .map(|m| m.role != "assistant")
        .unwrap_or(true)
    {
        out.push_str("<|start_of_role|>assistant<|end_of_role|>");
    }
    out
}

fn format_chatml(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Gemma format: <start_of_turn>user\ncontent<end_of_turn>
fn format_gemma(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = if msg.role == "assistant" {
            "model"
        } else {
            "user"
        };
        out.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, msg.content
        ));
    }
    out.push_str("<start_of_turn>model\n");
    out
}

/// gemma4:26b Harmony format: <|turn>role<turn|>content<|turn>model<turn|>
/// Vocab has asymmetric delimiters (id=105 `<|turn>` opens, id=106 `<turn|>`
/// closes). Verified empirically: `<|turn>user<turn|>What is 2+2?<|turn>model<turn|>`
/// -> clean English completion. Bare prompts -> degenerate HTML (matches Ollama).
fn format_gemma_harmony(messages: &[Message]) -> String {
    // gemma4: id=105 `<|turn>` opens a turn, id=106 `<turn|>` closes it.
    // Per the model's own GGUF chat_template (Jinja), the structure is
    //   <|turn>{role}\n{content}<turn|>\n
    // repeated per message, then `<|turn>model\n` to open generation.
    // The earlier "<|turn>{role}<turn|>{content}" form (content OUTSIDE the
    // turn, no newlines) was malformed - it orphaned the user text between
    // turns and produced incoherent output on gemma4:12b. roles: user stays
    // `user`, assistant -> `model`, system stays `system`.
    let mut out = String::new();
    for msg in messages {
        let role = if msg.role == "assistant" {
            "model"
        } else {
            msg.role.as_str()
        };
        out.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, msg.content));
    }
    out.push_str("<|turn>model\n");
    out
}

/// Moondream Q/A format: ` Question: <prompt>\n\n Answer: `
/// Exact byte layout from the Ollama moondream manifest template (leading
/// space before "Question:" and "Answer:" included).
fn format_moondream_qa(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg.role.as_str() {
            // No explicit system role in moondream's template - fold into
            // the question text so the LM still sees it.
            "system" | "developer" => {
                if !msg.content.is_empty() {
                    out.push_str(&format!(" Question: {}\n\n Answer: \n", msg.content));
                }
            }
            "user" => {
                out.push_str(&format!(" Question: {}\n\n Answer: ", msg.content));
            }
            "assistant" => {
                out.push_str(&msg.content);
                out.push('\n');
            }
            other => {
                out.push_str(&format!(
                    " Question: ({other}) {}\n\n Answer: ",
                    msg.content
                ));
            }
        }
    }
    if out.is_empty() {
        // Caller passed empty messages - leave a bare "Answer:" so the
        // model still sees the prompt-completion marker. Avoids producing
        // an empty string that downstream code might treat as a load-only
        // request.
        " Answer: ".to_string()
    } else {
        out
    }
}

/// Phi format: <|system|>\ncontent<|end|>\n<|user|>\ncontent<|end|>\n<|assistant|>
fn format_phi(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role, msg.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

/// GLM4 format: `[gMASK]<sop><|system|>\ncontent<|user|>\ncontent<|assistant|>`
fn format_glm4(system: &str, prompt: &str) -> String {
    let mut out = "[gMASK]<sop>".to_string();
    if !system.is_empty() {
        out.push_str(&format!("<|system|>\n{}", system));
    }
    out.push_str(&format!("<|user|>\n{}<|assistant|>\n", prompt));
    out
}

/// DeepSeek format (R1/V3): <｜User｜>content<｜Assistant｜>content<｜end▁of▁sentence｜>
/// System prompt is prepended verbatim before the first user turn.
fn format_deepseek(messages: &[Message]) -> String {
    let mut out = String::new();
    // System prompt first (no special tokens around it)
    // OpenAI's `developer` role is a system-prompt extension; append
    // its content alongside any system message before the first user
    // turn (DeepSeek folds both into the prefix the same way).
    if let Some(sys) = messages.iter().find(|m| m.role == "system") {
        out.push_str(&sys.content);
    }
    for dev in messages.iter().filter(|m| m.role == "developer") {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&dev.content);
    }
    let non_system: Vec<&Message> = messages
        .iter()
        .filter(|m| m.role != "system" && m.role != "developer")
        .collect();
    let last_idx = non_system.len().saturating_sub(1);
    for (i, msg) in non_system.iter().enumerate() {
        match msg.role.as_str() {
            "user" => {
                out.push_str("<｜User｜>");
                out.push_str(&msg.content);
            }
            "assistant" => {
                out.push_str("<｜Assistant｜>");
                out.push_str(&msg.content);
                if i != last_idx {
                    out.push_str("<｜end▁of▁sentence｜>");
                }
            }
            // tool / function: surface as user-turn observed output
            // rather than silently dropping. Tag the role inline so
            // the model can distinguish from genuine user input.
            other => {
                out.push_str("<｜User｜>");
                out.push_str(&format!("({other}) {}", msg.content));
            }
        }
    }
    // Append assistant marker if last message wasn't from assistant
    if non_system.last().map(|m| m.role.as_str()) != Some("assistant") {
        out.push_str("<｜Assistant｜>");
    }
    out
}

#[cfg(test)]
mod tests {
    /// The constructs the Qwen3-Coder template is built out of. Each one of these
    /// failing sent the model a reconstruction of its own format, with the tool block
    /// missing, and the only symptom was an agent that never called a tool.
    #[test]
    fn templates_can_walk_a_tool_schema() {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(super::python_string_methods);
        env.add_filter(
            "tojson",
            |v: minijinja::Value| -> Result<String, minijinja::Error> {
                serde_json::to_string(&v).map_err(|e| {
                    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
                })
            },
        );
        let ctx = minijinja::context! {
            d => minijinja::Value::from_serialize(serde_json::json!({"a": 1, "b": 2})),
        };
        let render = |src: &str| env.render_str(src, ctx.clone()).unwrap();
        assert_eq!(render("{{ d.keys() | list | join(',') }}"), "a,b");
        assert_eq!(render("{{ d.values() | list | join(',') }}"), "1,2");
        assert_eq!(
            render("{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}"),
            "a=1;b=2;"
        );
        assert_eq!(
            render("{{ d.get('a') }}|{{ d.get('z', 'none') }}"),
            "1|none"
        );
        assert_eq!(render("{{ d | tojson }}"), r#"{"a":1,"b":2}"#);
    }

    /// A conversation with no tools has to walk past the template's own guard. minijinja
    /// counts a none as defined, so passing one there skips the guard and dies on the
    /// length test the guard exists to prevent.
    #[test]
    fn a_template_with_no_tools_takes_the_empty_path() {
        let msgs = [crate::api::types::Message::new(
            "user".to_string(),
            "hello".to_string(),
        )];
        let tmpl = "{%- if not tools is defined %}{%- set tools = [] %}{%- endif %}\
                    {%- if tools is iterable and tools | length > 0 %}TOOLS\
                    {%- else %}NONE{%- endif %}{{ messages[0].content }}";
        let out = super::render_jinja_template(tmpl, &msgs, false, None);
        assert_eq!(out.as_deref(), Some("NONEhello"));
    }

    #[test]
    fn templates_can_call_python_string_methods() {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(super::python_string_methods);
        let render = |src: &str| env.render_str(src, minijinja::context! {}).unwrap();
        assert_eq!(
            render("{{ '<tool_response>x'.startswith('<tool_response>') }}"),
            "True"
        );
        assert_eq!(render("{{ 'a</think>b'.split('</think>') | last }}"), "b");
        assert_eq!(
            render("{{ 'a</think>b</think>c'.split('</think>', 1) | length }}"),
            "2"
        );
        assert_eq!(
            render("{{ ' x \n'.rstrip() }}|{{ 'xxy'.lstrip('x') }}"),
            " x|y"
        );
        assert_eq!(
            render("{{ 'A b'.lower() }} {{ 'a-b'.replace('-', '+') }}"),
            "a b a+b"
        );
        assert_eq!(
            render("{{ ', '.join(['a', 'b']) }} {{ 'abc'.find('c') }}"),
            "a, b 2"
        );
        assert!(env
            .render_str("{{ 'x'.casefold() }}", minijinja::context! {})
            .is_err());
    }

    use super::*;

    #[test]
    fn prompt_already_templated_detects_each_family() {
        // ChatML - qwen, llama-3 community tunes
        assert!(prompt_already_templated(
            "<|im_start|>user\nhi<|im_end|>",
            "<|im_start|>{{role}}\n{{content}}<|im_end|>",
        ));
        // Gemma
        assert!(prompt_already_templated(
            "<start_of_turn>user\nhi<end_of_turn>",
            "<start_of_turn>{{role}}\n{{content}}<end_of_turn>",
        ));
        // Mistral / Llama-2 [INST]
        assert!(prompt_already_templated(
            "[INST] hi [/INST]",
            "[INST] {{content}} [/INST]"
        ));
    }

    #[test]
    fn prompt_already_templated_rejects_mismatched_family() {
        // Template fingerprint matches ChatML but prompt has gemma markers  -
        // we should NOT report "already templated" because re-applying ChatML
        // is still required.
        assert!(!prompt_already_templated(
            "<start_of_turn>user\nhi<end_of_turn>",
            "<|im_start|>{{role}}\n{{content}}<|im_end|>",
        ));
        // Plain user prompt - no markers anywhere.
        assert!(!prompt_already_templated(
            "just a question",
            "<|im_start|>..."
        ));
    }

    fn msg(role: &str, content: &str) -> crate::api::types::Message {
        crate::api::types::Message::new(role.to_string(), content.to_string())
    }

    #[test]
    fn format_chatml_wraps_each_message_and_opens_assistant() {
        let out = format_chatml(&[msg("system", "be helpful"), msg("user", "hi")]);
        assert!(out.contains("<|im_start|>system\nbe helpful<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        // Must end with the open assistant marker so the model continues
        // a turn rather than starting a new conversation.
        assert!(out.ends_with("<|im_start|>assistant\n"), "got: {out}");
    }

    #[test]
    fn format_gemma_renames_assistant_to_model_role() {
        // Gemma uses 'model' on the assistant turn; the wrapper must
        // remap so an OpenAI-shaped conversation still produces a valid
        // gemma prompt.
        let out = format_gemma(&[
            msg("user", "Q1?"),
            msg("assistant", "A1"),
            msg("user", "Q2?"),
        ]);
        assert!(out.contains("<start_of_turn>user\nQ1?<end_of_turn>"));
        assert!(out.contains("<start_of_turn>model\nA1<end_of_turn>"));
        assert!(out.contains("<start_of_turn>user\nQ2?<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn format_gemma_harmony_uses_asymmetric_turn_delimiters() {
        // gemma4: id=105 `<|turn>` opens, id=106 `<turn|>` closes. Content
        // goes INSIDE the turn with newlines, per the model's GGUF
        // chat_template: `<|turn>{role}\n{content}<turn|>\n` + `<|turn>model\n`.
        let out = format_gemma_harmony(&[msg("user", "What is 2+2?")]);
        assert_eq!(out, "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n");
    }

    #[test]
    fn format_phi_wraps_every_role_and_opens_assistant() {
        // Phi format applies the same `<|role|>\ncontent<|end|>\n`
        // shape to every message - including `developer` or `tool`
        // roles a future caller might surface. Pin this so a
        // role-renaming change has to update the test.
        let out = format_phi(&[
            msg("system", "S"),
            msg("user", "U"),
            msg("assistant", "A"),
            msg("user", "U2"),
        ]);
        assert!(out.contains("<|system|>\nS<|end|>"));
        assert!(out.contains("<|user|>\nU<|end|>"));
        assert!(out.contains("<|assistant|>\nA<|end|>"));
        assert!(out.contains("<|user|>\nU2<|end|>"));
        assert!(out.ends_with("<|assistant|>\n"), "got: {out}");
    }

    #[test]
    fn format_glm4_prepends_system_and_opens_assistant() {
        // GLM4: `[gMASK]<sop>` is the BOS pair; system + user
        // segments follow. The assistant marker at the end keeps
        // generation flowing into a model turn.
        let with_sys = format_glm4("ground truth", "question?");
        assert!(with_sys.starts_with("[gMASK]<sop>"));
        assert!(with_sys.contains("<|system|>\nground truth"));
        assert!(with_sys.contains("<|user|>\nquestion?<|assistant|>\n"));

        // Empty system: skip the `<|system|>` block but still emit
        // user + assistant.
        let no_sys = format_glm4("", "just ask");
        assert!(no_sys.starts_with("[gMASK]<sop>"));
        assert!(!no_sys.contains("<|system|>"));
        assert!(no_sys.contains("<|user|>\njust ask<|assistant|>\n"));
    }

    #[test]
    fn format_deepseek_folds_system_developer_and_eos_between_turns() {
        // DeepSeek wraps user with `<｜User｜>` and assistant with
        // `<｜Assistant｜>`, inserting `<｜end▁of▁sentence｜>` *between*
        // assistant turns but NOT after the final one (the model is
        // expected to continue from the final assistant marker).
        let out = format_deepseek(&[
            msg("system", "SYS"),
            msg("developer", "DEV"),
            msg("user", "Q1"),
            msg("assistant", "A1"),
            msg("user", "Q2"),
        ]);
        // System prefix comes first, no special wrapping.
        assert!(out.starts_with("SYS"), "got: {out}");
        // Developer content concatenated after a newline.
        assert!(out.contains("SYS\nDEV"), "got: {out}");
        // User and assistant wraps applied.
        assert!(out.contains("<｜User｜>Q1"));
        assert!(out.contains("<｜Assistant｜>A1"));
        assert!(out.contains("<｜User｜>Q2"));
        // Intra-turn EOS appears after A1 (mid-stream), but the
        // string ends with the bare assistant marker so the model
        // generates the next reply.
        assert!(out.contains("<｜Assistant｜>A1<｜end▁of▁sentence｜>"));
        assert!(out.ends_with("<｜Assistant｜>"), "got: {out}");
    }

    #[test]
    fn format_deepseek_omits_trailing_assistant_when_last_is_assistant() {
        // If the conversation ends on an assistant message (e.g.
        // when the caller is just rendering history without
        // requesting a continuation), don't append an extra
        // `<｜Assistant｜>` marker.
        let out = format_deepseek(&[msg("user", "Q"), msg("assistant", "final reply")]);
        assert!(out.ends_with("final reply"), "got: {out}");
        // Exactly one assistant marker - the one wrapping the reply.
        assert_eq!(out.matches("<｜Assistant｜>").count(), 1, "got: {out}");
        // No mid-stream EOS because this is the last turn.
        assert!(!out.contains("<｜end▁of▁sentence｜>"), "got: {out}");
    }

    #[test]
    fn format_deepseek_tags_unknown_roles_inline_as_user_turn() {
        // tool/function results shouldn't be silently dropped. They
        // get a `<｜User｜>(role) content` tag so the model can
        // distinguish observed output from genuine user input.
        let out = format_deepseek(&[
            msg("user", "call f"),
            msg("tool", "result=42"),
            msg("user", "what now?"),
        ]);
        assert!(out.contains("<｜User｜>call f"));
        assert!(out.contains("<｜User｜>(tool) result=42"));
        assert!(out.contains("<｜User｜>what now?"));
        // Ends with assistant marker (last message was a user turn).
        assert!(out.ends_with("<｜Assistant｜>"), "got: {out}");
    }

    #[test]
    fn format_chat_prompt_routes_by_template_fingerprint() {
        let messages = vec![msg("user", "hi")];
        // ChatML
        let out = format_chat_prompt(&messages, Some("<|im_start|>..."));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        // Gemma (legacy)
        let out = format_chat_prompt(&messages, Some("<start_of_turn>..."));
        assert!(out.contains("<start_of_turn>user\nhi<end_of_turn>"));
        // Gemma Harmony (must win over legacy gemma in the dispatch order).
        // Structure per the model's GGUF chat_template:
        //   <|turn>{role}\n{content}<turn|>\n ... <|turn>model\n
        let out = format_chat_prompt(&messages, Some("<|turn>{{role}}<turn|>{{content}}"));
        assert!(
            out.starts_with("<|turn>user\nhi<turn|>\n<|turn>model\n"),
            "got: {out}"
        );
        // Phi
        let out = format_chat_prompt(&messages, Some("<|system|>...<|end|>"));
        assert!(out.contains("<|user|>\nhi<|end|>"));
        // DeepSeek (full-width pipes)
        let out = format_chat_prompt(&messages, Some("<｜User｜>...<｜Assistant｜>"));
        assert!(out.contains("hi"));
        // GLM4
        let out = format_chat_prompt(&messages, Some("[gMASK]<sop>..."));
        assert!(out.starts_with("[gMASK]<sop>"));
    }

    /// A declared Llama 3 template must reach its own formatter. It used to match no
    /// fingerprint and fall through to the [INST] default - another family's format,
    /// which the model echoes back instead of answering.
    #[test]
    fn a_llama3_template_is_not_formatted_as_mistral() {
        let msgs = vec![
            Message {
                role: "system".into(),
                content: "be terse".into(),
                images: None,
                audios: None,
                thinking: None,
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: "user".into(),
                content: "hello".into(),
                images: None,
                audios: None,
                thinking: None,
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let out = format_chat_prompt(
            &msgs,
            Some("<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}"),
        );
        assert!(
            !out.contains("[INST]"),
            "fell through to the Mistral default: {out}"
        );
        assert!(
            out.starts_with("<|start_header_id|>system<|end_header_id|>\n\nbe terse<|eot_id|>"),
            "got: {out}"
        );
        assert!(
            out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            "got: {out}"
        );
        // The name-inferred fallback must reach the same formatter.
        assert_eq!(
            infer_template_from_model_name("llama3.2:1b").as_deref(),
            Some("<|start_header_id|>")
        );
    }

    #[test]
    fn format_chat_prompt_falls_through_to_inst_format_on_unknown_template() {
        let messages = vec![msg("system", "be terse"), msg("user", "hello")];
        let out = format_chat_prompt(&messages, Some("some-novel-template-spec"));
        // Unknown template fingerprint -> llama-style [INST] with system
        // folded into the first user turn.
        assert!(
            out.starts_with("[INST] be terse\n\nhello [/INST]"),
            "got: {out}"
        );
    }

    #[test]
    fn format_chat_prompt_handles_unknown_role_via_tag_prefix() {
        // tool / function-call role: should NOT be silently dropped  -
        // wrap as a tagged user turn so the content reaches the model.
        let messages = vec![
            msg("user", "use the calculator"),
            msg("assistant", "ok"),
            msg("tool", "result=42"),
        ];
        let out = format_chat_prompt(&messages, None);
        assert!(
            out.contains("[INST] (tool) result=42 [/INST]"),
            "got: {out}"
        );
    }

    #[test]
    fn infer_template_from_model_name_maps_known_families() {
        let pick = |name: &str| infer_template_from_model_name(name).unwrap_or_default();
        // All gemma4 (latest, 26b, instruct...) -> Harmony template
        // (per b605672: GGUF vocab confirmed asymmetric `<|turn>` IDs 105/106
        // across the entire gemma4 family, NOT the legacy gemma3 tokens).
        assert_eq!(pick("gemma4:26b"), "<|turn>");
        assert_eq!(pick("Gemma4-26b-instruct"), "<|turn>");
        assert_eq!(pick("gemma4:latest"), "<|turn>");
        // gemma3 and earlier - legacy template.
        assert_eq!(pick("gemma3:latest"), "<start_of_turn>");
        assert_eq!(pick("gemma:2b"), "<start_of_turn>");
        // ChatML family
        assert_eq!(pick("qwen3-coder:30b"), "<|im_start|>");
        assert_eq!(pick("chatml-finetune"), "<|im_start|>");
        // Phi family - fingerprint includes all role markers so
        // format_chat_prompt can locate them.
        assert!(pick("phi3:14b").contains("<|user|>"));
        assert!(pick("phi4-mini").contains("<|user|>"));
        // DeepSeek - full-width brackets.
        assert!(pick("deepseek-r1:32b").contains("<｜User｜>"));
        // Unknown model -> None.
        assert!(infer_template_from_model_name("mystery-model").is_none());
    }
}

#[cfg(test)]
mod go_template_tests {
    use super::*;

    #[test]
    fn renders_the_common_ollama_shape() {
        let t = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n{{ .Response }}";
        assert_eq!(
            render_go_template(t, Some("be brief"), "hi").unwrap(),
            "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            render_go_template(t, None, "hi").unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(render_go_template("{{ range .Messages }}x{{ end }}", None, "hi").is_none());
    }

    #[test]
    fn harmony_effort_is_written_into_the_system_turn() {
        let mut p =
            "<|start|>system<|message|>You are ChatGPT.\nReasoning: medium\n<|end|>".to_string();
        apply_thinking_preference(&mut p, Some("high"));
        assert!(p.contains("Reasoning: high"));
        assert!(!p.contains("Reasoning: medium"));
        let mut q = "<|start|>system<|message|>You are ChatGPT.<|end|>".to_string();
        apply_thinking_preference(&mut q, Some("disabled"));
        assert!(q.starts_with("<|start|>system<|message|>Reasoning: low\n"));
    }
}
