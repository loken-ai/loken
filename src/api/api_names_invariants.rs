//! No text model or family name decides anything in the API: the chat template decides how
//! a conversation and its tools are written and read, the vocabulary stands in when there
//! is no template, and the file's declared architecture is the family reported.
//!
//! The gate reads every production source under `src/api` and fails on a string literal
//! naming a text model family that sits in a comparison: a `contains`, `starts_with`,
//! `ends_with`, `strip_prefix`, `==` or `!=` operand, or a match pattern. Media families are
//! the ones `media_family` names; they are outside this gate.

use std::path::{Path, PathBuf};

/// Text model and family names, lowercase. A literal holds one when the name starts it or
/// follows a character that is not a letter or digit.
const NAMES: &[&str] = &[
    "bge",
    "codellama",
    "codestral",
    "deepseek",
    "devstral",
    "falcon",
    "gemma",
    "glm",
    "gpt-oss",
    "granite",
    "guardian",
    "internvl",
    "lfm",
    "llama",
    "llava",
    "minicpm",
    "mistral",
    "mixtral",
    "moondream",
    "nemotron",
    "nomic",
    "olmo",
    "phi",
    "pixtral",
    "qwen",
    "safeguard",
    "shieldgemma",
    "shieldstral",
    "smollm",
    "stablelm",
    "starcoder",
];

/// The code of a source file the gate reads: everything before its test module, with
/// comments removed.
fn production_lines(src: &str) -> Vec<String> {
    let code = match src.find("#[cfg(test)]\nmod ") {
        Some(at) => &src[..at],
        None => src,
    };
    code.lines().map(strip_comment).collect()
}

/// A line without its `//` comment, a `//` inside a string literal kept.
fn strip_comment(line: &str) -> String {
    let line = line.replace("'\"'", "' '");
    let mut in_string = false;
    let mut escaped = false;
    let chars: Vec<char> = line.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            return chars[..i].iter().collect();
        }
    }
    line
}

/// The string literals of a line, with the code before and after each.
fn literals(line: &str) -> Vec<(&str, &str, &str)> {
    let mut out = Vec::new();
    let mut start = None;
    let mut escaped = false;
    for (i, c) in line.char_indices() {
        match start {
            Some(s) => match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => {
                    out.push((&line[..s], &line[s + 1..i], &line[i + 1..]));
                    start = None;
                }
                _ => {}
            },
            None if c == '"' => start = Some(i),
            None => {}
        }
    }
    out
}

/// Whether a literal between `before` and `after` is compared against.
fn compared(before: &str, after: &str) -> bool {
    let before = before.trim_end();
    let after = after.trim_start();
    [
        "contains(",
        "starts_with(",
        "ends_with(",
        "strip_prefix(",
        "strip_suffix(",
        "==",
        "!=",
    ]
    .iter()
    .any(|op| before.ends_with(op))
        || after.starts_with("=>")
        || (after.starts_with('|') && !after.starts_with("||"))
}

/// The text model name a literal holds, if any.
fn named(literal: &str) -> Option<&'static str> {
    let lower = literal.to_lowercase();
    if crate::api::handlers::media_family(&lower).is_some() {
        return None;
    }
    NAMES.iter().copied().find(|name| {
        lower.match_indices(name).any(|(at, _)| {
            lower[..at]
                .chars()
                .next_back()
                .is_none_or(|c| !c.is_ascii_alphanumeric())
        })
    })
}

/// The name comparisons in one source, as `line: code` strings.
fn violations(src: &str) -> Vec<String> {
    let mut found = Vec::new();
    for (n, line) in production_lines(src).iter().enumerate() {
        for (before, literal, after) in literals(line) {
            if let Some(name) = named(literal) {
                if compared(before, after) {
                    found.push(format!("{}: {name}: {}", n + 1, line.trim()));
                }
            }
        }
    }
    found
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir)
        .expect("src/api is readable")
        .flatten()
    {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if stem != "tests" && !stem.ends_with("_tests") && stem != "api_names_invariants" {
                out.push(path);
            }
        }
    }
}

#[test]
fn no_model_name_decides_anything_in_the_api() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api");
    let mut files = Vec::new();
    sources(&root, &mut files);
    assert!(!files.is_empty(), "no source under {}", root.display());
    let mut failures = Vec::new();
    for file in files {
        let src = std::fs::read_to_string(&file).expect("source is readable");
        for v in violations(&src) {
            failures.push(format!("{}:{v}", file.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "a model name decides behaviour; read the template, vocabulary or header instead:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn the_gate_sees_every_comparison_shape() {
    let caught = [
        r#"if lower.contains("gemma4") {"#,
        r#"    m.starts_with("qwen3") => x,"#,
        r#"if name == "llama3.2:1b" {"#,
        r#"    "phi3" | "phi4" => Format::Phi,"#,
        r#"    "deepseek-r1" => 1,"#,
        r#"let x = m.ends_with("-GLM");"#,
    ];
    for line in caught {
        assert!(!violations(line).is_empty(), "missed: {line}");
    }
    let passed = [
        r#"// if lower.contains("gemma") {"#,
        r#"let s = "<|im_start|>";"#,
        r#"Self::LlamaGuard => "llama-guard","#,
        r#"if t.contains("graphic") {"#,
        r#"if lower.contains("qwen-image") {"#,
        r#"println!("loading gemma"); // "gemma" =>"#,
        "#[cfg(test)]\nmod tests {\n    x.contains(\"gemma\");\n}",
    ];
    for line in passed {
        assert!(violations(line).is_empty(), "false alarm: {line}");
    }
}
