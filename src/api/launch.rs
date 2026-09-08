//! What a coding agent needs to reach a lokend, in the form each agent reads.
//!
//! Claude Code takes its endpoint from the environment of the process; Cline reads two JSON
//! files under its own directory. Both are shaped here from one [`Launch`], and the binary only
//! spawns the agent.

use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// The token Claude Code presents when the daemon does not ask for a key.
const PLACEHOLDER_TOKEN: &str = "loken";

/// The Cline provider these files select. Cline speaks to the daemon over the Ollama API.
const CLINE_PROVIDER: &str = "ollama";

/// A coding agent the client can start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Claude,
    Cline,
}

impl Tool {
    pub fn parse(name: &str) -> Option<Tool> {
        match name {
            "claude" => Some(Tool::Claude),
            "cline" => Some(Tool::Cline),
            _ => None,
        }
    }

    /// The executable on the PATH.
    pub fn binary(&self) -> &'static str {
        match self {
            Tool::Claude => "claude",
            Tool::Cline => "cline",
        }
    }

    /// The one-line way to install it when the binary is missing.
    pub fn install_hint(&self) -> &'static str {
        match self {
            Tool::Claude => "curl -fsSL https://claude.ai/install.sh | bash",
            Tool::Cline => "npm install -g cline",
        }
    }
}

/// Where the agent goes and what it asks for.
#[derive(Debug, Clone)]
pub struct Launch {
    /// The daemon's address, without a trailing slash.
    pub server: String,
    /// The model every request names.
    pub model: String,
    /// The key the daemon expects, when it requires one.
    pub api_key: Option<String>,
    /// The window the daemon loads the model with, when known.
    pub context: Option<u32>,
}

impl Launch {
    pub fn new(server: &str, model: &str, api_key: Option<String>) -> Launch {
        Launch {
            server: server.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            context: None,
        }
    }

    pub fn with_context(mut self, context: Option<u32>) -> Launch {
        self.context = context;
        self
    }
}

/// The environment Claude Code reads: the Messages API endpoint, the credential, the model
/// behind every tier so that no request leaves for the vendor, and the window it compacts
/// within when the daemon told it.
pub fn claude_env(launch: &Launch) -> Vec<(String, String)> {
    let token = launch
        .api_key
        .clone()
        .unwrap_or_else(|| PLACEHOLDER_TOKEN.to_string());
    let mut env = vec![
        ("ANTHROPIC_BASE_URL".to_string(), launch.server.clone()),
        ("ANTHROPIC_API_KEY".to_string(), String::new()),
        ("ANTHROPIC_AUTH_TOKEN".to_string(), token),
    ];
    for tier in [
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ] {
        env.push((tier.to_string(), launch.model.clone()));
    }
    if let Some(context) = launch.context {
        env.push((
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS".to_string(),
            context.to_string(),
        ));
    }
    env
}

/// The arguments Claude Code takes on its command line.
pub fn claude_args(launch: &Launch) -> Vec<String> {
    vec!["--model".to_string(), launch.model.clone()]
}

/// Cline's provider file and its global state, under the home directory.
pub fn cline_paths(home: &Path) -> (PathBuf, PathBuf) {
    let data = home.join(".cline").join("data");
    (
        data.join("settings").join("providers.json"),
        data.join("globalState.json"),
    )
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

/// Cline's provider file with the daemon selected as the provider in use. Other providers in
/// the file are kept.
pub fn cline_providers(existing: Value, launch: &Launch, now: &str) -> Value {
    let mut config = object(existing);
    let mut providers = object(config.remove("providers").unwrap_or(Value::Null));
    let mut provider = object(providers.remove(CLINE_PROVIDER).unwrap_or(Value::Null));
    let mut settings = object(provider.remove("settings").unwrap_or(Value::Null));
    settings.insert("provider".into(), json!(CLINE_PROVIDER));
    settings.insert("model".into(), json!(launch.model));
    settings.insert("baseUrl".into(), json!(format!("{}/v1", launch.server)));
    match &launch.api_key {
        Some(key) => settings.insert("apiKey".into(), json!(key)),
        None => settings.remove("apiKey"),
    };
    provider.insert("settings".into(), Value::Object(settings));
    provider.insert("tokenSource".into(), json!("manual"));
    provider.insert("updatedAt".into(), json!(now));
    providers.insert(CLINE_PROVIDER.into(), Value::Object(provider));
    config.insert("version".into(), json!(1));
    config.insert("lastUsedProvider".into(), json!(CLINE_PROVIDER));
    config.insert("providers".into(), Value::Object(providers));
    Value::Object(config)
}

/// Cline's global state with both modes on the daemon and the welcome screen behind.
pub fn cline_global_state(existing: Value, launch: &Launch) -> Value {
    let mut config = object(existing);
    config.insert("ollamaBaseUrl".into(), json!(launch.server));
    for mode in ["actMode", "planMode"] {
        config.insert(format!("{mode}ApiProvider"), json!(CLINE_PROVIDER));
        config.insert(format!("{mode}OllamaModelId"), json!(launch.model));
        config.insert(format!("{mode}OllamaBaseUrl"), json!(launch.server));
    }
    config.insert("welcomeViewCompleted".into(), json!(true));
    Value::Object(config)
}

fn read_json(path: &Path) -> std::io::Result<Value> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Value::Object(Map::new())),
        Err(e) => Err(e),
    }
}

/// Write one file, keeping the previous content next to it with a `.bak` suffix.
fn write_json(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if path.exists() {
        std::fs::copy(path, path.with_extension("json.bak"))?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
}

/// Point Cline at the daemon. Returns the files written.
pub fn write_cline(home: &Path, launch: &Launch) -> std::io::Result<Vec<PathBuf>> {
    let (providers, state) = cline_paths(home);
    let now = chrono::Utc::now().to_rfc3339();
    write_json(
        &providers,
        &cline_providers(read_json(&providers)?, launch, &now),
    )?;
    write_json(&state, &cline_global_state(read_json(&state)?, launch))?;
    Ok(vec![providers, state])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch() -> Launch {
        Launch::new("http://localhost:11435/", "qwen3:8b", None)
    }

    #[test]
    fn claude_gets_the_endpoint_and_one_model_for_every_tier() {
        let env = claude_env(&launch());
        let get = |k: &str| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("ANTHROPIC_BASE_URL"), Some("http://localhost:11435"));
        assert_eq!(get("ANTHROPIC_API_KEY"), Some(""));
        assert_eq!(get("ANTHROPIC_AUTH_TOKEN"), Some("loken"));
        assert_eq!(get("ANTHROPIC_DEFAULT_HAIKU_MODEL"), Some("qwen3:8b"));
        assert_eq!(get("CLAUDE_CODE_SUBAGENT_MODEL"), Some("qwen3:8b"));
        assert_eq!(get("CLAUDE_CODE_MAX_CONTEXT_TOKENS"), None);
        assert_eq!(claude_args(&launch()), ["--model", "qwen3:8b"]);
    }

    #[test]
    fn a_known_window_bounds_the_session() {
        let env = claude_env(&launch().with_context(Some(32768)));
        assert!(env.contains(&(
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS".to_string(),
            "32768".to_string()
        )));
    }

    #[test]
    fn a_key_becomes_the_token() {
        let l = Launch::new("http://h:1", "m", Some("k".into()));
        let env = claude_env(&l);
        assert!(env.contains(&("ANTHROPIC_AUTH_TOKEN".to_string(), "k".to_string())));
        let p = cline_providers(Value::Null, &l, "t");
        assert_eq!(p["providers"]["ollama"]["settings"]["apiKey"], "k");
    }

    #[test]
    fn cline_keeps_the_other_providers_and_selects_the_daemon() {
        let existing = json!({
            "version": 1,
            "lastUsedProvider": "anthropic",
            "providers": {
                "anthropic": {"settings": {"apiKey": "sk"}},
                "ollama": {"settings": {"apiKey": "old", "model": "x"}, "updatedAt": "then"}
            }
        });
        let p = cline_providers(existing, &launch(), "now");
        assert_eq!(p["lastUsedProvider"], "ollama");
        assert_eq!(p["providers"]["anthropic"]["settings"]["apiKey"], "sk");
        let ollama = &p["providers"]["ollama"];
        assert_eq!(ollama["settings"]["model"], "qwen3:8b");
        assert_eq!(ollama["settings"]["baseUrl"], "http://localhost:11435/v1");
        assert!(ollama["settings"].get("apiKey").is_none());
        assert_eq!(ollama["updatedAt"], "now");
        assert_eq!(ollama["tokenSource"], "manual");
    }

    #[test]
    fn cline_global_state_puts_both_modes_on_the_daemon() {
        let s = cline_global_state(json!({"other": 1}), &launch());
        assert_eq!(s["other"], 1);
        assert_eq!(s["actModeOllamaModelId"], "qwen3:8b");
        assert_eq!(s["planModeOllamaBaseUrl"], "http://localhost:11435");
        assert_eq!(s["welcomeViewCompleted"], true);
    }

    #[test]
    fn the_files_land_under_the_home_directory() {
        let home = std::env::temp_dir().join(format!("loken-launch-{}", std::process::id()));
        let written = write_cline(&home, &launch()).unwrap();
        assert_eq!(written, {
            let (a, b) = cline_paths(&home);
            vec![a, b]
        });
        let again = write_cline(&home, &launch()).unwrap();
        assert!(again[0].with_extension("json.bak").exists());
        std::fs::remove_dir_all(&home).unwrap();
    }
}
