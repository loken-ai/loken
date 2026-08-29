//! The one rule that cannot be re-litigated: user content never reaches a log.
//!
//! Prompts, generated text, captions and raw token ids are the user's data. A log
//! line carrying any of them persists it somewhere the user did not choose - a file,
//! a journal, a shipped bundle - and a `debug!` is not an exemption, because debug
//! logging is exactly what gets turned on when something goes wrong.
//!
//! Token ids count as content: they decode back to the exact text.
//!
//! This module holds no runtime code. It holds the GATE, because a rule that is only
//! written down is a rule that comes back.

#[cfg(test)]
mod tests {
    /// Every source file in the crate, so the scan cannot miss a new one.
    fn sources() -> Vec<(String, String)> {
        fn walk(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    if let Ok(s) = std::fs::read_to_string(&p) {
                        out.push((p.to_string_lossy().into_owned(), s));
                    }
                }
            }
        }
        let mut out = Vec::new();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        // Both crates that can log: the server and the GUI. A rule that stops at a
        // crate boundary is not a rule.
        for dir in [root.join("src"), root.join("../gui/src")] {
            walk(&dir, &mut out);
        }
        out
    }

    /// Names that hold user content wherever they appear.
    const CONTENT: [&str; 11] = [
        "prompt",
        "text",
        "caption",
        "negative",
        "generated",
        "decoded",
        "completion",
        "message",
        // Specific containers of token IDS, which decode back to the text. The bare
        // word "token" is not here: it appears in every count in the codebase.
        "next_token",
        "token_ids",
        "generated_token_ids",
    ];

    /// Split a logging macro's arguments into its format literal and the expressions
    /// after it. Only the expressions can carry a value; the literal only hints.
    fn split_args(args: &str) -> (String, String) {
        let bytes: Vec<char> = args.chars().collect();
        let Some(open) = bytes.iter().position(|c| *c == '"') else {
            return (String::new(), args.to_string());
        };
        let mut i = open + 1;
        while i < bytes.len() {
            if bytes[i] == '"' && bytes[i - 1] != '\\' {
                break;
            }
            i += 1;
        }
        let literal: String = bytes[open + 1..i.min(bytes.len())].iter().collect();
        let rest: String = bytes[(i + 1).min(bytes.len())..].iter().collect();
        (literal, rest)
    }

    /// A logging macro invocation on one line, with everything it was handed.
    fn log_call(line: &str) -> Option<&str> {
        for m in ["info!(", "warn!(", "error!(", "debug!(", "trace!("] {
            if let Some(i) = line.find(m) {
                return Some(&line[i + m.len()..]);
            }
        }
        None
    }

    /// Whole-word identifier match, so `text_encoder` is not `text`.
    fn names_a_content_value(expr: &str) -> bool {
        let mut word = String::new();
        let mut hit = false;
        for c in expr.chars().chain(std::iter::once(' ')) {
            if c.is_alphanumeric() || c == '_' {
                word.push(c.to_ascii_lowercase());
                continue;
            }
            if CONTENT.contains(&word.as_str()) {
                hit = true;
            }
            word.clear();
        }
        hit
    }

    /// SOURCE GATE: no log line may interpolate user content.
    ///
    /// The scan is deliberately crude - it flags a logging macro whose arguments
    /// mention a content-bearing name without an obvious size accessor. That
    /// over-flags, which is the right direction for this rule: a false positive costs
    /// a `.len()` or an explicit exemption, a false negative writes someone's prompt
    /// to disk. Mark a reviewed line with `PRIVACY-OK: <why>`.
    #[test]
    fn no_log_line_carries_user_content() {
        // Accessors that reduce content to a measurement rather than reproducing it.
        let measured = [
            ".len()",
            ".count()",
            ".chars().count()",
            "_len",
            "_count",
            "_tokens",
            "_chars",
            "is_empty()",
            "is_some()",
            "is_none()",
            ".iter().count()",
        ];
        let mut offenders = Vec::new();
        for (path, src) in sources() {
            // This file describes the rule; its own prose names the forbidden things.
            if path.ends_with("privacy.rs") {
                continue;
            }
            for (i, line) in src.lines().enumerate() {
                let Some(args) = log_call(line) else { continue };
                // The exemption may sit on the line itself or on the comment above it,
                // which is where the justification naturally goes.
                let exempt = line.contains("PRIVACY-OK:")
                    || i.checked_sub(1)
                        .and_then(|j| src.lines().nth(j))
                        .is_some_and(|prev| prev.contains("PRIVACY-OK:"));
                if exempt || !args.contains('{') {
                    continue;
                }
                let (literal, exprs) = split_args(args);
                // Two tells, because neither alone is enough. An expression NAMING a
                // content value is the direct case. And a message that talks about a
                // prompt or generated text while debug-formatting a value beside it is
                // the indirect case - the value was renamed (`head`, `slice`) but the
                // sentence gives it away.
                let direct =
                    names_a_content_value(&exprs) && !measured.iter().any(|m| exprs.contains(m));
                let indirect = {
                    let l = literal.to_lowercase();
                    literal.contains("{:?}") && CONTENT.iter().any(|c| l.contains(c))
                };
                if direct || indirect {
                    offenders.push(format!("{path}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "a log line interpolates user content (prompts, generated text and token ids are \
             all content - log a size, or mark the line PRIVACY-OK: <why>):\n{}",
            offenders.join("\n")
        );
    }
}
