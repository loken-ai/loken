//! No model or family name decides anything outside the facts table.
//!
//! What a model is comes from its file: the tensors a block carries, their shapes, the
//! metadata keys. The few facts a file does not carry sit in one table,
//! `inference/model_facts.rs`, keyed by the file's `general.architecture`; what a media
//! checkpoint's files declare is read in `inference/media_facts.rs`, keyed by the pipeline
//! class, model type or architecture they carry. Anywhere else, a string literal naming an
//! architecture or a model family inside a comparison, a match pattern or a list of names
//! is a decision taken on a name, and this gate refuses it.
//!
//! Writing an identifier is not deciding on one: a converter that stores
//! `general.architecture` in the file it produces, or a translation whose RESULT is an
//! architecture tag, places the literal where a value is built (a constructor argument, the
//! right of `=>`), which the gate does not count. Reading a file-format identifier that
//! shares its spelling with a family (a `tokenizer.ggml.pre` value, an HF `architectures`
//! class) sits under an `ARCH-NAME-EXEMPT:` comment that says so; the exemption covers the
//! statement below it.

use super::placement_invariants::{fleet_sources, production};

/// The tables: the files allowed to decide by an identifier a checkpoint's files carry.
const FACTS_TABLES: [&str; 2] = ["inference/model_facts.rs", "inference/media_facts.rs"];

/// Trees this gate does not read yet. `api/` routes requests by model and carries its own
/// coverage for now; extend this gate to it once that lands.
const NOT_YET_GATED: &[&str] = &["api/"];

/// Words that begin a model family's name. A literal equal to one, or to one followed by
/// anything but a lowercase letter (`gemma4`, `llama-bpe`, `Qwen2ForCausalLM`), names a
/// family.
const FAMILY_WORDS: &[&str] = &[
    "llama",
    "mistral",
    "qwen",
    "gemma",
    "phi",
    "olmo",
    "granite",
    "lfm",
    "nemotron",
    "deepseek",
    "gpt-oss",
    "gptoss",
    "moondream",
    "falcon",
    "starcoder",
    "stablelm",
    "chatglm",
    "glm",
    "dbrx",
    "internlm",
    "orion",
    "mpt",
    "bloom",
    "ernie",
    "smollm",
    "yi",
    "refact",
    "codeshell",
    "exaone",
    "cohere",
    "command-r",
    "mamba",
    "jamba",
    "rwkv",
    "hunyuan",
    "minimax",
    "baichuan",
    "mixtral",
    "magistral",
    "devstral",
    "codestral",
    // Media families, checkpoints and their publishers.
    "flux",
    "kontext",
    "klein",
    "sdxl",
    "zimage",
    "z-image",
    "z_image",
    "boogu",
    "wan",
    "causvid",
    "fusionx",
    "lightx2v",
    "whisper",
    "parler",
    "kyutai",
    "pocket-tts",
    "piper",
    "xtts",
    "openvoice",
    "bark",
    "kokoro",
    "acestep",
    "ace-step",
    "ezaudio",
    "stable-audio",
    "stable_audio",
    "stable-diffusion",
    "rayflux",
    "rayzist",
    "rayqwest",
    "raymnants",
    "rayctifier",
    "rayburn",
    "lcm",
    "black-forest-labs",
    "tongyi",
    "coqui",
    "myshell",
    "systran",
    "city96",
];

/// Files that still decide by a name, with how many literals. The list only shrinks.
///
/// `distributed/layer_scheduler.rs` estimates a model's geometry from its id.
const NAME_DEBT: &[(&str, usize)] = &[("distributed/layer_scheduler.rs", 1)];

/// Written on or above a statement that reads a file-format identifier.
const EXEMPT: &str = "ARCH-NAME-EXEMPT:";

/// Whether a literal names an architecture of the table or a model family.
fn names_a_model(lit: &str) -> bool {
    let l = lit.to_ascii_lowercase();
    crate::inference::model_facts::architectures().any(|a| a == l)
        || FAMILY_WORDS.iter().any(|w| {
            l.strip_prefix(w)
                .is_some_and(|rest| rest.chars().next().is_none_or(|c| !c.is_ascii_lowercase()))
        })
}

/// A string literal of a source: its line, its column span on that line, its content, and
/// the bracket it sits in.
struct Literal {
    line: usize,
    start: usize,
    end: usize,
    text: String,
    inside: Option<char>,
}

/// The string literals of `text` outside comments, and the bracket depth at the end of each
/// line.
fn scan(text: &str) -> (Vec<Literal>, Vec<usize>) {
    let chars: Vec<char> = text.chars().collect();
    let mut literals = Vec::new();
    let mut depth_at_end = Vec::new();
    let mut stack: Vec<char> = Vec::new();
    let (mut line, mut col) = (0usize, 0usize);
    let mut i = 0;
    let advance =
        |i: &mut usize, line: &mut usize, col: &mut usize, depth: usize, ends: &mut Vec<usize>| {
            if chars[*i] == '\n' {
                ends.push(depth);
                *line += 1;
                *col = 0;
            } else {
                *col += 1;
            }
            *i += 1;
        };
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
            }
            continue;
        }
        if c == '/' && next == Some('*') {
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
            }
            i += 2;
            col += 2;
            continue;
        }
        if c == '\'' {
            // A char literal; a lifetime has no closing quote two or three places on.
            let close = match next {
                Some('\\') => chars[i + 2..]
                    .iter()
                    .position(|&x| x == '\'')
                    .map(|p| i + 2 + p),
                Some(_) if chars.get(i + 2) == Some(&'\'') => Some(i + 2),
                _ => None,
            };
            if let Some(close) = close {
                col += close + 1 - i;
                i = close + 1;
                continue;
            }
        }
        let raw_hashes = (c == 'r'
            && !chars
                .get(i.wrapping_sub(1))
                .is_some_and(|p| p.is_alphanumeric() || *p == '_'))
        .then(|| chars[i + 1..].iter().take_while(|&&x| x == '#').count())
        .filter(|&h| chars.get(i + 1 + h) == Some(&'"'));
        if c == '"' || raw_hashes.is_some() {
            let hashes = raw_hashes.unwrap_or(0);
            let (start_line, start_col) = (line, col);
            let open = if raw_hashes.is_some() { 2 + hashes } else { 1 };
            for _ in 0..open {
                advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
            }
            let mut content = String::new();
            while i < chars.len() {
                let ch = chars[i];
                if raw_hashes.is_none() && ch == '\\' {
                    content.push(ch);
                    advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
                    if i < chars.len() {
                        content.push(chars[i]);
                        advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
                    }
                    continue;
                }
                if ch == '"'
                    && chars[i + 1..]
                        .iter()
                        .take(hashes)
                        .filter(|&&x| x == '#')
                        .count()
                        == hashes
                {
                    for _ in 0..=hashes {
                        advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
                    }
                    break;
                }
                content.push(ch);
                advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
            }
            if line == start_line {
                literals.push(Literal {
                    line: start_line,
                    start: start_col,
                    end: col,
                    text: content,
                    inside: stack.last().copied(),
                });
            }
            continue;
        }
        match c {
            '(' | '[' | '{' => stack.push(c),
            ')' | ']' | '}' => {
                stack.pop();
            }
            _ => {}
        }
        advance(&mut i, &mut line, &mut col, stack.len(), &mut depth_at_end);
    }
    depth_at_end.push(stack.len());
    (literals, depth_at_end)
}

/// Whether the literal at `start..end` of `line` is compared: on a line with `==`/`!=` or a
/// string test, a match pattern ahead of the line's first `=>`, or an alternative beside `|`.
fn compared(line: &str, start: usize, end: usize) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let before: String = chars[..start.min(chars.len())].iter().collect();
    let after: String = chars[end.min(chars.len())..].iter().collect();
    const TESTS: [&str; 9] = [
        "==",
        "!=",
        ".contains(",
        ".starts_with(",
        ".ends_with(",
        ".strip_prefix(",
        ".strip_suffix(",
        "matches!(",
        ".find(",
    ];
    TESTS.iter().any(|t| line.contains(t))
        || after.contains("=>") && !before.contains("=>")
        || after.trim_start().starts_with('|') && !after.trim_start().starts_with("||")
        || before.trim_end().ends_with('|') && !before.trim_end().ends_with("||")
}

/// Every literal of `text` that names a model inside a comparison or a list, as
/// `(line number, literal)`. Statements under an [`EXEMPT`] comment are skipped.
fn name_decisions(text: &str) -> Vec<(usize, String)> {
    let text = production(text);
    let lines: Vec<&str> = text.lines().collect();
    let (literals, depth_at_end) = scan(text);
    // The lines each exemption covers: from its comment through the line that closes the
    // statement below it.
    let mut exempt = vec![false; lines.len()];
    let mut l = 0;
    while l < lines.len() {
        if !lines[l].contains(EXEMPT) {
            l += 1;
            continue;
        }
        let base = if l == 0 { 0 } else { depth_at_end[l - 1] };
        let mut k = l;
        while k < lines.len() {
            exempt[k] = true;
            let t = lines[k].trim_end();
            let closes = depth_at_end[k] <= base
                && !t.trim_start().starts_with("//")
                && (t.ends_with(';') || t.ends_with('}') || t.ends_with(')') || t.ends_with(','));
            if closes {
                break;
            }
            k += 1;
        }
        l = k + 1;
    }
    literals
        .iter()
        .filter(|lit| !exempt.get(lit.line).copied().unwrap_or(false))
        .filter(|lit| names_a_model(&lit.text))
        .filter(|lit| {
            lit.inside == Some('[')
                || compared(lines.get(lit.line).unwrap_or(&""), lit.start, lit.end)
        })
        .map(|lit| (lit.line + 1, lit.text.clone()))
        .collect()
}

#[test]
fn a_name_is_caught_where_it_decides_and_nowhere_else() {
    let decides = [
        "fn f(a: &str) -> bool { a == \"gemma4\" }",
        "fn f(a: &str) -> bool { a.starts_with(\"gemma\") }",
        "fn f(a: &str) -> u8 { match a {\n    \"llama\" |\n    \"mistral\" => 1,\n    _ => 0,\n} }",
        "const L: &[&str] = &[\n    \"qwen3\",\n];",
        "fn f(a: Option<&str>) -> bool { a == Some(\"gptoss\") }",
        "fn f(a: &str) -> bool { matches!(a, \"olmoe\" | \"x\") }",
        "fn f(a: &str) -> bool { a.contains(\"flux\") }",
        "fn f(a: &str) -> bool { a.starts_with(\"wan-\") }",
    ];
    for src in decides {
        assert!(!name_decisions(src).is_empty(), "missed: {src}");
    }
    let writes = [
        "fn f() -> V { V::String(\"qwen3\".into()) }",
        "fn f(hf: &str) -> &str { match hf { \"X\" => \"qwen2\", _ => \"\" } }",
        "fn f() { log(\"gemma4 loaded\"); }",
        "// a == \"gemma4\" in a comment\nfn f() {}",
        "fn f(a: &str) -> bool { a == \"gemmaless\" && a == \"yield\" }",
        "fn f(a: &str) -> bool { a == \"want\" || a == \"fluxion\" }",
        "fn f(p: &str) -> u8 {\n    // ARCH-NAME-EXEMPT: a tokenizer.ggml.pre value.\n    let x = match p {\n        \"llama3\" => 1,\n        _ => 0,\n    };\n    x\n}",
    ];
    for src in writes {
        assert!(
            name_decisions(src).is_empty(),
            "flagged: {src}: {:?}",
            name_decisions(src)
        );
    }
    // The exemption stops at the end of its statement.
    let after = "fn f(p: &str) -> u8 {\n    // ARCH-NAME-EXEMPT: a tokenizer.ggml.pre value.\n    let x = match p {\n        \"llama3\" => 1,\n        _ => 0,\n    };\n    if p == \"qwen3\" { 2 } else { x }\n}";
    assert_eq!(name_decisions(after), vec![(7, "qwen3".to_string())]);
}

/// NO NAME DECIDES OUTSIDE THE FACTS TABLE.
#[test]
fn no_architecture_name_decides_outside_the_facts_table() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let debt: std::collections::BTreeMap<&str, usize> = NAME_DEBT.iter().copied().collect();
    let mut problems = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for path in fleet_sources() {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        let table = FACTS_TABLES.contains(&rel.as_str());
        if table || NOT_YET_GATED.iter().any(|d| rel.starts_with(d)) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let found = name_decisions(&text);
        match debt.get(rel.as_str()) {
            Some(&was) => {
                seen.insert(rel.clone());
                if found.len() != was {
                    problems.push(format!(
                        "{rel}: {} name decisions, debt says {was}",
                        found.len()
                    ));
                }
            }
            None => {
                for (line, lit) in found {
                    problems.push(format!("{rel}:{line}: {lit:?} decides by name"));
                }
            }
        }
    }
    for file in debt.keys().filter(|f| !seen.contains(**f)) {
        problems.push(format!("{file}: gone; remove it from NAME_DEBT"));
    }
    assert!(
        problems.is_empty(),
        "behaviour comes from the file's tensors and metadata, read in {FACTS_TABLES:?}:\n{}",
        problems.join("\n")
    );
}
