//! Shared scaffolding for the per-model-type render CLIs (the `*_render` bins): uniform TOML
//! config loading, energy-metered execution, and error/exit handling - so each bin stays a thin
//! wrapper over its inference engine instead of re-copying the same boilerplate. Modality-specific
//! concerns (model-path resolution, OOM/VRAM ladders, HF-cache wiring) stay in the bins.

use serde::de::DeserializeOwned;

pub type DynErr = Box<dyn std::error::Error>;

/// Read the TOML config from `argv[1]` (the bin's only positional argument) into `T`. On a missing
/// argument or any read/parse error, print a uniform message + `usage` to stderr and `exit(1)`.
pub fn load_toml<T: DeserializeOwned>(bin: &str, usage: &str) -> T {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("{bin}: missing config file\n{usage}");
            std::process::exit(1);
        }
    };
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("{bin}: cannot read `{path}`: {e}\n{usage}");
        std::process::exit(1);
    });
    toml::from_str(&text).unwrap_or_else(|e| {
        eprintln!("{bin}: invalid TOML in `{path}`: {e}");
        std::process::exit(1);
    })
}

/// The integer/float config field names for [`load_toml_cli`], so `--key value` overrides coerce
/// to the right TOML type (a bare string would fail to deserialize into a numeric field). Every
/// other flag is passed through as a string. Field names use the struct's snake_case spelling;
/// `--kebab-case` flags are accepted and normalized.
pub struct NumKeys<'a> {
    pub ints: &'a [&'a str],
    pub floats: &'a [&'a str],
}

/// Like [`load_toml`] but the positional TOML path is OPTIONAL and any field can be set or
/// overridden with `--key value` / `--key=value` flags that WIN over the file (and over the
/// struct's serde defaults when no file is given). `nums` classifies the numeric fields so their
/// CLI values coerce correctly. `--help`/`-h` prints `usage` and exits 0. All error paths print a
/// uniform `{bin}: ...` message + `usage` and `exit(1)`.
pub fn load_toml_cli<T: DeserializeOwned>(bin: &str, usage: &str, nums: NumKeys) -> T {
    let mut toml_path: Option<String> = None;
    let mut overrides: Vec<(String, String)> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "-h" || a == "--help" {
            println!("{usage}");
            std::process::exit(0);
        }
        if let Some(flag) = a.strip_prefix("--") {
            let (k, v) = match flag.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => match it.next() {
                    Some(v) => (flag.to_string(), v),
                    None => {
                        eprintln!("{bin}: flag --{flag} needs a value\n{usage}");
                        std::process::exit(1);
                    }
                },
            };
            overrides.push((k, v));
        } else if toml_path.is_none() {
            toml_path = Some(a);
        } else {
            eprintln!("{bin}: unexpected argument `{a}`\n{usage}");
            std::process::exit(1);
        }
    }
    let text = match &toml_path {
        Some(p) => std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("{bin}: cannot read `{p}`: {e}\n{usage}");
            std::process::exit(1);
        }),
        None => String::new(),
    };
    let mut root: toml::Value = if text.trim().is_empty() {
        toml::Value::Table(Default::default())
    } else {
        toml::from_str(&text).unwrap_or_else(|e| {
            eprintln!(
                "{bin}: invalid TOML in `{}`: {e}",
                toml_path.as_deref().unwrap_or("?")
            );
            std::process::exit(1);
        })
    };
    {
        let tbl = match root.as_table_mut() {
            Some(t) => t,
            None => {
                eprintln!("{bin}: TOML root must be a table");
                std::process::exit(1);
            }
        };
        for (k, v) in overrides {
            let key = k.replace('-', "_");
            let val = if nums.ints.contains(&key.as_str()) {
                v.parse::<i64>()
                    .map(toml::Value::Integer)
                    .unwrap_or_else(|_| {
                        eprintln!("{bin}: --{key} expects an integer, got `{v}`");
                        std::process::exit(1);
                    })
            } else if nums.floats.contains(&key.as_str()) {
                v.parse::<f64>()
                    .map(toml::Value::Float)
                    .unwrap_or_else(|_| {
                        eprintln!("{bin}: --{key} expects a number, got `{v}`");
                        std::process::exit(1);
                    })
            } else {
                toml::Value::String(v)
            };
            tbl.insert(key, val);
        }
    }
    root.try_into().unwrap_or_else(|e| {
        eprintln!("{bin}: invalid config after overrides: {e}");
        std::process::exit(1);
    })
}

/// Print `"{bin}: {err}"` to stderr and `exit(1)`.
pub fn bail(bin: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("{bin}: {err}");
    std::process::exit(1);
}
