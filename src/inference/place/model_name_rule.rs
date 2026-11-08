//! What the model-name gates share: telling a criterion applied by name from a detection.
//!
//! No criterion is applied model by model: behaviour comes from tensors, metadata or the
//! chat template, never from a table or a test on a model name.
//!
//! Code specific to a family carries that family's real name: a module, a directory, a
//! type, a function, a test fixture. A literal is accepted too when it is compared with a
//! value the model's own files declare: the `model_type` or `architectures` of its
//! configuration, a tensor name, a metadata key, a header field. That is detection: the
//! file says what it is.
//!
//! A literal compared with any other value is refused: a request, a file name, a tag, an
//! alias, a catalogue id, or an operand whose origin the gate cannot read.

/// Words that say a value is read from the model's own files, as whole snake or camel case
/// tokens of an identifier.
const READ_FROM_THE_FILES: &[&str] = &[
    "architecture",
    "architectures",
    "arch",
    "config",
    "cfg",
    "metadata",
    "meta",
    "header",
    "tensor",
    "tensors",
    "gguf",
    "safetensors",
    "pipeline",
    "class",
];

/// Words that say a value comes from outside the files, as whole tokens of an identifier.
const COMES_FROM_OUTSIDE: &[&str] = &[
    "name",
    "names",
    "tag",
    "tags",
    "id",
    "ids",
    "alias",
    "aliases",
    "catalogue",
    "catalog",
    "path",
    "file",
    "filename",
    "repo",
    "request",
    "req",
    "prompt",
    "model",
    "slug",
    "label",
    "lower",
    "query",
];

/// The lowercase tokens of an identifier: split at `_` and at a lower to upper boundary.
fn tokens(ident: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in ident.chars() {
        if (c == '_' || (c.is_ascii_uppercase() && prev_lower)) && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if c != '_' {
            cur.push(c.to_ascii_lowercase());
        }
        prev_lower = c.is_ascii_lowercase() || c.is_ascii_digit();
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The code of a line with every string literal emptied and its `//` comment cut.
fn code_of(line: &str) -> String {
    let mut out = String::new();
    let mut in_string = false;
    let mut escaped = false;
    let chars: Vec<char> = line.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => {
                    in_string = false;
                    out.push('"');
                }
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
            out.push('"');
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            break;
        } else {
            out.push(c);
        }
    }
    out
}

/// The identifiers of a piece of code.
fn identifiers(code: &str) -> Vec<&str> {
    code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| w.chars().next().is_some_and(|c| !c.is_ascii_digit()))
        .collect()
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The statement head of line `at`: the nearest line above with a smaller indentation, a
/// function signature excepted (its parameters name the caller, not the comparison).
fn head<'a>(lines: &[&'a str], at: usize) -> Option<&'a str> {
    let here = indent(lines.get(at)?);
    lines[..at]
        .iter()
        .rev()
        .take(60)
        .filter(|l| !l.trim().is_empty())
        .find(|l| indent(l) < here)
        .copied()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("fn ")
                || t.starts_with("pub fn ")
                || t.starts_with("pub(crate) fn ")
                || t.starts_with("async fn ")
                || t.starts_with("pub async fn "))
        })
}

/// Whether the comparison on `lines[at]` reads a value the model's own files declare, and
/// nothing that comes from outside them. The operands are the identifiers of the line and of
/// the statement head above it (a `match` scrutinee, a `const` name).
pub(crate) fn reads_the_model_itself(lines: &[&str], at: usize) -> bool {
    let mut code = code_of(lines.get(at).copied().unwrap_or(""));
    if let Some(h) = head(lines, at) {
        code.push(' ');
        code.push_str(&code_of(h));
    }
    let (mut reads, mut outside) = (false, false);
    for ident in identifiers(&code) {
        let toks = tokens(ident);
        let declared = ident.ends_with("model_type")
            || toks
                .iter()
                .any(|t| READ_FROM_THE_FILES.contains(&t.as_str()));
        if declared {
            reads = true;
        } else if toks
            .iter()
            .any(|t| COMES_FROM_OUTSIDE.contains(&t.as_str()))
        {
            outside = true;
        }
    }
    reads && !outside
}

#[test]
fn a_comparison_reads_the_files_or_it_does_not() {
    let reads = [
        "if cfg.model_type == \"qwen3_tts\" {",
        "if hf.architectures.iter().any(|a| a == \"Qwen2ForCausalLM\") {",
        "let is = tensor_name.starts_with(\"qwen3.\");",
    ];
    for line in reads {
        assert!(reads_the_model_itself(&[line], 0), "refused: {line}");
    }
    let decides = [
        "if name.contains(\"qwen\") {",
        "if req.model.contains(\"qwen\") {",
        "if config.model_name == \"qwen3\" {",
        "if lower.contains(\"gemma\") {",
        "if file_name.ends_with(\"llama\") {",
    ];
    for line in decides {
        assert!(!reads_the_model_itself(&[line], 0), "accepted: {line}");
    }
    let lines = [
        "match config.model_type.as_str() {",
        "    \"qwen3_tts\" => 1,",
    ];
    assert!(reads_the_model_itself(&lines, 1));
    let lines = ["match name {", "    \"qwen3_tts\" => 1,"];
    assert!(!reads_the_model_itself(&lines, 1));
    let lines = [
        "fn f(req: &Request) {",
        "    if cfg.model_type == \"qwen3_tts\" {",
    ];
    assert!(reads_the_model_itself(&lines, 1));
}
