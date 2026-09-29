//! No model, family or publisher name decides anything in the API: the chat template decides
//! how a conversation and its tools are written and read, the vocabulary stands in when there
//! is no template, the file's declared architecture is the family reported, and a media
//! checkpoint's own files say what it is (`inference/media_facts.rs`).
//!
//! The gate reads every production source under `src/api` and fails on a string literal
//! naming a model family or its publisher that sits in a comparison: a `contains`,
//! `starts_with`, `ends_with`, `strip_prefix`, `eq_ignore_ascii_case`, `==` or `!=` operand,
//! or a match pattern. A name may still be a value - an alias the catalogue looks up, an
//! entry it registers - but never an operand.

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

/// Media model, family and publisher names, lowercase. A literal holds one when the name
/// starts it or follows a character that is not a letter or digit, and ends it or comes
/// before a character that is not a lowercase letter: short words like `wan` or `t5` are
/// not caught inside `want` or `t5x`.
const MEDIA_NAMES: &[&str] = &[
    "ace-step",
    "acestep",
    "bark",
    "black-forest-labs",
    "boogu",
    "causvid",
    "city96",
    "clip",
    "coqui",
    "ezaudio",
    "flux",
    "flash",
    "fusionx",
    "kokoro",
    "kontext",
    "klein",
    "kyutai",
    "lcm",
    "lightx2v",
    "mt5",
    "myshell",
    "openvoice",
    "parler",
    "piper",
    "pocket-tts",
    "qwen-image",
    "qwen_image",
    "rayburn",
    "rayctifier",
    "rayflux",
    "raymnants",
    "rayqwest",
    "rayzist",
    "sdxl",
    "stable-audio",
    "stable_audio",
    "stable-cascade",
    "stable-diffusion",
    "systran",
    "t5",
    "tongyi",
    "wan",
    "whisper",
    "xtts",
    "z-image",
    "z_image",
    "zimage",
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
        "eq_ignore_ascii_case(",
        "==",
        "!=",
    ]
    .iter()
    .any(|op| before.ends_with(op))
        || after.starts_with("=>")
        || (after.starts_with('|') && !after.starts_with("||"))
}

/// The model, family or publisher name a literal holds, if any.
fn named(literal: &str) -> Option<&'static str> {
    let lower = literal.to_lowercase();
    let starts_word = |at: usize| {
        lower[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_ascii_alphanumeric())
    };
    let ends_word = |end: usize| {
        lower[end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_lowercase())
    };
    NAMES
        .iter()
        .copied()
        .find(|name| lower.match_indices(name).any(|(at, _)| starts_word(at)))
        .or_else(|| {
            MEDIA_NAMES.iter().copied().find(|name| {
                lower
                    .match_indices(name)
                    .any(|(at, m)| starts_word(at) && ends_word(at + m.len()))
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
        "a model name decides behaviour; read the template, vocabulary, header or the \
         checkpoint's files instead:\n  {}",
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
        r#"if lower.contains("qwen-image") {"#,
        r#"if model.starts_with("wan") {"#,
        r#"    "flux2" | "flux" => 1,"#,
        r#"if r.eq_ignore_ascii_case("whisper-1") {"#,
        r#"let t = name.contains("kyutai");"#,
    ];
    for line in caught {
        assert!(!violations(line).is_empty(), "missed: {line}");
    }
    let passed = [
        r#"// if lower.contains("gemma") {"#,
        r#"let s = "<|im_start|>";"#,
        r#"Self::LlamaGuard => "llama-guard","#,
        r#"if t.contains("graphic") {"#,
        r#"if t.contains("want") || t.contains("clipboard") {"#,
        r#"const ALIASES: &[(&str, &str)] = &[("kyutai", "kyutai/tts")];"#,
        r#"println!("loading gemma"); // "gemma" =>"#,
        "#[cfg(test)]\nmod tests {\n    x.contains(\"gemma\");\n}",
    ];
    for line in passed {
        assert!(violations(line).is_empty(), "false alarm: {line}");
    }
}
