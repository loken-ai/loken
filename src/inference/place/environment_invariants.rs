//! No setting comes from the process environment: the configuration names what an operator
//! decides and the engine derives the rest, so a variable would be a second, hidden
//! configuration.

/// Lines that read the process environment, by file, as they stood before this rule. A file
/// not listed reads none, and a listed file reads no more than its count: a setting comes from
/// the configuration or is derived, never from a variable.
const ENV_READS: &[(&str, usize)] = &[
    ("api/handlers/model_manager.rs", 1),
    ("api/handlers/openai/mod.rs", 2),
    ("api/handlers/prompt_format.rs", 1),
    ("api/handlers/tests.rs", 1),
    ("bin/loken.rs", 1),
    ("bin/lokend.rs", 2),
    ("config.rs", 2),
    ("inference/engine/audio_engine.rs", 1),
    ("inference/engine/llm_engine/gguf.rs", 2),
    ("inference/engine/tts_engine.rs", 1),
    ("inference/generic_transformer/card_host_parity.rs", 7),
    ("inference/generic_transformer/verify_decode_parity.rs", 3),
    ("inference/load/huggingface_manager.rs", 1),
    ("inference/model/boogu/dit.rs", 2),
    ("inference/model/clip/vision.rs", 1),
    ("inference/model/deepseek_v41/dspark.rs", 4),
    ("inference/model/sdxl/pipeline.rs", 1),
    ("inference/model/sdxl/vae.rs", 1),
    ("inference/model/stable_audio/tests.rs", 6),
    ("inference/model/wan/dit/layers.rs", 1),
    ("inference/model/zimage/vae.rs", 1),
    ("inference/moe_cuda/mmq_parity.rs", 6),
    ("inference/quantized_cuda/awq.rs", 2),
    ("inference/sample/flow_unipc.rs", 1),
    ("tensor/pth.rs", 1),
    ("tensor/safetensors_io.rs", 2),
];

/// The calls that read the process environment, spelled in two parts so this file does not
/// count itself.
fn environment_calls() -> [String; 3] {
    ["var(", "var_os("]
        .map(|f| format!("{}::{f}", "env"))
        .into_iter()
        .chain([format!("option_{}!(", "env")])
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Whether a line of code (comment removed) reads the process environment at run time.
fn reads_environment(line: &str) -> bool {
    let code = line.split("//").next().unwrap_or("");
    environment_calls()
        .iter()
        .any(|p| code.contains(p.as_str()))
}

/// The environment reads of one file's text.
fn environment_reads(text: &str) -> usize {
    text.lines().filter(|l| reads_environment(l)).count()
}

fn source_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            source_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn no_setting_comes_from_the_process_environment() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    source_files(&root, &mut files);
    let mut over = Vec::new();
    for f in files {
        let rel = f
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let allowed = ENV_READS
            .iter()
            .find(|(p, _)| *p == rel)
            .map_or(0, |(_, n)| *n);
        let found = environment_reads(&std::fs::read_to_string(&f).unwrap());
        if found > allowed {
            over.push(format!(
                "{rel}: {found} environment reads, {allowed} allowed"
            ));
        }
    }
    assert!(over.is_empty(), "{}", over.join("\n"));
}

#[test]
fn the_environment_scanner_counts_a_read_and_skips_a_comment() {
    let [var, var_os, _] = environment_calls();
    assert_eq!(
        environment_reads(&format!("let a = std::{var}(\"X\");\nlet b = 1;")),
        1
    );
    assert_eq!(
        environment_reads(&format!("let k = std::{var_os}(\"X\");")),
        1
    );
    assert_eq!(
        environment_reads(&format!("// std::{var}(\"X\")\nlet b = 1;")),
        0
    );
}
