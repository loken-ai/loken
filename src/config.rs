//! Configuration file loader
//!
//! Loads configuration from config.toml file

use crate::tensor::DType;
use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::inference::engine::llm_engine::InferenceConfig;

/// Server configuration
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Require an API key on every request that is not a health check.
    ///
    /// Off by default so an existing local install keeps working untouched. It has to
    /// be ON before the port is reachable from anywhere the operator does not control:
    /// the API can pull models and DELETE them, so "no auth" is not merely a quota
    /// question.
    #[serde(default)]
    pub require_auth: bool,
    /// Accepted API keys. Empty with `require_auth = true` refuses every request,
    /// which is the safe direction: a typo in the config locks the door rather than
    /// opening it.
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Origins a browser may call from when auth is on. Empty means none - browser
    /// clients then need a proxy, which is the correct default for a credentialed API.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    /// Requests per minute per client, 0 = unlimited.
    ///
    /// Authentication answers WHO may call; this answers HOW MUCH. They are different
    /// problems: one valid key running a loop can hold every GPU indefinitely, and no
    /// amount of key checking stops it.
    #[serde(default)]
    pub rate_limit_per_minute: usize,
    /// How many requests may arrive back-to-back before the per-minute rate applies.
    /// A UI that opens a page fires several calls at once and is not abuse; without
    /// slack, a limit low enough to matter would break normal use.
    #[serde(default = "default_rate_burst")]
    pub rate_limit_burst: usize,
}

fn default_rate_burst() -> usize {
    10
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 11435,
            require_auth: false,
            api_keys: Vec::new(),
            allowed_origins: Vec::new(),
            rate_limit_per_minute: 0,
            rate_limit_burst: default_rate_burst(),
        }
    }
}

/// Tokens per block of the decode-state cache's hash chain when the configuration names none.
pub const DEFAULT_STATE_CACHE_BLOCK_TOKENS: usize = 256;

fn default_state_cache_block_tokens() -> usize {
    DEFAULT_STATE_CACHE_BLOCK_TOKENS
}

/// Inference configuration from TOML file. Only what loken cannot decide itself: the model it
/// serves by default, where things live, and the features an operator switches on. Sampling,
/// placement, threads and cache formats are decided from the model and the machine.
#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
pub struct InferenceConfigToml {
    pub model_id: String,
    #[serde(default)]
    pub kv_disk_dir: Option<String>,
    #[serde(default)]
    pub kv_disk_budget_gb: f64,
    /// Directory of the on-disk cache of DeepSeek V4.1 decode states, which lets a prompt that
    /// begins with a prefix seen before skip prefilling it. Off unless set. The files hold
    /// state derived from prompts (KV tensors and hidden rows, no text), so they are as private
    /// as the prompts.
    #[serde(default)]
    pub state_cache_dir: Option<String>,
    /// Bytes the decode-state cache may hold, in GiB; 0 is no limit.
    #[serde(default)]
    pub state_cache_budget_gb: f64,
    /// Tokens per block of the cache's hash chain. Changing it orphans the stored states.
    #[serde(default = "default_state_cache_block_tokens")]
    pub state_cache_block_tokens: usize,
    /// Mixed into every state's identity: states written under another salt are never read.
    #[serde(default)]
    pub state_cache_salt: String,
    #[serde(default)]
    pub files_dir: Option<String>,
    #[serde(default)]
    pub moderation_model: Option<String>,
    #[serde(default)]
    pub public_url: Option<String>,
    /// For a model whose routed experts stream from disk: the share of a token's experts whose
    /// host compute may finish a layer late, overlapping the next layer's attention. Costs a
    /// little accuracy; 0 keeps every layer exact.
    #[serde(default)]
    pub expert_deferral: f32,
    /// For a streamed model: the largest mean divergence (KL, in nats) from the exact output the
    /// operator accepts in exchange for speed. The engine measures what each way of giving up
    /// experts costs on a bundled text at load and picks the fastest that fits; 0 keeps the
    /// output exact.
    #[serde(default)]
    pub quality_budget: f32,
    /// Keys this version no longer reads, kept so a config written for an older one still
    /// loads; each is reported once at startup.
    #[serde(flatten, skip_serializing)]
    pub obsolete: std::collections::BTreeMap<String, toml::Value>,
}

/// Energy-reporting configuration (`[energy]` section). All optional with sane
/// defaults so existing config.toml files keep working unchanged.
#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
pub struct EnergyConfig {
    /// Measure + report energy on every request. Default: true (systematic).
    pub enabled: Option<bool>,
    /// Grid carbon intensity (gCO₂eq / kWh) for the CO₂ estimate. Default: France ~50.
    pub carbon_intensity: Option<f64>,
    /// CPU package TDP (W) used to estimate CPU energy when RAPL is unreadable
    /// (root-only kernels). Default: 0 = no estimate (report measured domains only).
    pub cpu_tdp_w: Option<f64>,
    /// Water footprint of electricity generation (liters / kWh). Default: ~1.8.
    pub water_l_per_kwh: Option<f64>,
}

/// Root configuration structure
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct Config {
    pub server: Option<ServerConfig>,
    pub inference: InferenceConfigToml,
    pub ollama_models_dir: Option<String>,
    pub huggingface_models_dir: Option<String>,
    /// Where LoRA adapters are kept. Requests name an adapter, never a path, and the
    /// name is resolved inside this directory - see `native_lora::resolve`.
    pub lora_dir: Option<String>,
    #[serde(default)]
    pub energy: Option<EnergyConfig>,
    /// Clustering policy. Absent = this node runs alone, and every cluster path is skipped.
    #[serde(default)]
    pub cluster: Option<ClusterConfigToml>,
}

/// What clustering needs told to it. The fabric is discovered; only policy is configured.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ClusterConfigToml {
    /// Nodes announcing another name are ignored, so two clusters can share a network
    /// without merging - a development machine and a production node on one switch would
    /// otherwise route each other's requests, and nothing would report it.
    #[serde(default = "default_cluster_name")]
    pub name: String,
    /// How this node calls itself. Empty = derive from the hostname, which is what makes a
    /// three-machine setup work with no per-machine configuration at all.
    #[serde(default)]
    pub node_id: String,
    /// Where peers should reach this node. Empty = advertise nothing and stay a client of
    /// the others: a node behind NAT can still use a cluster without being usable BY it.
    #[serde(default)]
    pub advertise: String,
    /// Seeds for peers multicast cannot reach - another subnet. Discovery adds to this.
    #[serde(default)]
    pub join: Vec<String>,
    #[serde(default = "default_gossip_ms")]
    pub gossip_interval_ms: u64,
    /// Refuse to move a request below this predicted speedup. Moving one costs a second
    /// failure domain and a relayed stream that the arithmetic does not model.
    #[serde(default = "default_min_speedup")]
    pub min_speedup: f64,
}

fn default_cluster_name() -> String {
    "default".to_string()
}
fn default_gossip_ms() -> u64 {
    1000
}
fn default_min_speedup() -> f64 {
    1.15
}

impl Config {
    /// Resolve the energy-reporting settings (defaults applied).
    pub fn energy_settings(&self) -> crate::energy_report::EnergySettings {
        let e = self.energy.clone().unwrap_or_default();
        crate::energy_report::EnergySettings {
            enabled: e.enabled.unwrap_or(true),
            carbon_intensity: e
                .carbon_intensity
                .unwrap_or(crate::energy_report::DEFAULT_CARBON_INTENSITY),
            cpu_tdp_w: e.cpu_tdp_w.unwrap_or(0.0),
            water_l_per_kwh: e
                .water_l_per_kwh
                .unwrap_or(crate::energy_report::DEFAULT_WATER_L_PER_KWH),
        }
    }
}

impl Config {
    /// Load configuration from TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file {:?}: {}", path, e))?;

        let config: Config =
            toml::from_str(&content).map_err(|e| anyhow!("Failed to parse config file: {}", e))?;

        Ok(config)
    }

    /// Load from default location, searching multiple paths:
    /// 1. ./config.toml (current working directory)
    /// 2. Next to the executable
    /// 3. User config dir (~/.config/loken/config.toml, or %APPDATA%\loken on Windows)
    pub fn load_default() -> Result<Self> {
        // Try CWD first
        let cwd = Path::new("config.toml");
        if cwd.exists() {
            return Self::load(cwd);
        }

        // Try next to the executable
        if let Ok(exe) = std::env::current_exe() {
            let exe_dir = exe.parent().unwrap_or(Path::new(".")).join("config.toml");
            if exe_dir.exists() {
                return Self::load(&exe_dir);
            }
            // A build tree puts the binary several levels below the directory that holds
            // the configuration, so walk up from it. Without this the file is found only
            // when the server happens to be started from the right directory, and a
            // configuration silently replaced by defaults changes what the engine does -
            // quantized GPU matmul is off by default, which sends large models to the host.
            let mut up = exe.parent();
            for _ in 0..4 {
                let Some(dir) = up else { break };
                let candidate = dir.join("config.toml");
                if candidate.exists() {
                    return Self::load(&candidate);
                }
                up = dir.parent();
            }
        }

        // Try the user config directory.
        if let Some(config_dir) = dirs::config_dir() {
            let user_config = config_dir.join("loken").join("config.toml");
            if user_config.exists() {
                return Self::load(&user_config);
            }
        }

        Err(anyhow!(
            "config.toml not found in CWD, exe dir, or user config dir"
        ))
    }

    /// Convert to InferenceConfig
    pub fn to_inference_config(&self) -> InferenceConfig {
        let toml_config = &self.inference;

        InferenceConfig {
            model_id: toml_config.model_id.clone(),
            context_length: 0,
            dtype: DType::F16, // Default to F16
            device_index: None,
            kv_disk_dir: toml_config.kv_disk_dir.clone(),
            kv_disk_budget_gb: toml_config.kv_disk_budget_gb,
            state_cache_dir: toml_config.state_cache_dir.clone(),
            state_cache_budget_gb: toml_config.state_cache_budget_gb,
            state_cache_block_tokens: toml_config.state_cache_block_tokens,
            state_cache_salt: toml_config.state_cache_salt.clone(),
            files_dir: toml_config.files_dir.clone(),
            moderation_model: toml_config.moderation_model.clone(),
            public_url: toml_config.public_url.clone(),
            expert_deferral: toml_config.expert_deferral,
            quality_budget: toml_config.quality_budget,
            progress_callback: None, // No progress callback from config file
            models_dir: None,        // Will be set by APIServer
            disable_arc_layers: false,
            disable_cuda: false,
            force_cuda_only_layers: false, // Not exposed in config.toml, use programmatically for testing
            kv_quant: crate::inference::engine::llm_engine::KvQuant::Off,
        }
    }

    /// Get server config
    pub fn server(&self) -> ServerConfig {
        self.server.clone().unwrap_or_default()
    }

    /// Get OS-aware default Ollama models directory
    pub fn default_ollama_models_dir() -> PathBuf {
        #[cfg(target_os = "windows")]
        {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".ollama").join("models")
        }
        #[cfg(not(target_os = "windows"))]
        {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".ollama")
                .join("models")
        }
    }

    /// Get OS-aware default HuggingFace models directory
    pub fn default_hf_models_dir() -> PathBuf {
        #[cfg(target_os = "windows")]
        {
            let app_data = std::env::var("APPDATA").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(app_data).join("huggingface").join("hub")
        }
        #[cfg(not(target_os = "windows"))]
        {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
                .join("huggingface")
                .join("hub")
        }
    }

    /// Get the configured Ollama models directory (or default)
    pub fn get_ollama_models_dir(&self) -> PathBuf {
        self.ollama_models_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(Self::default_ollama_models_dir)
    }

    /// Get the configured LoRA directory (or a `loras` sibling of the model store).
    pub fn get_lora_dir(&self) -> PathBuf {
        self.lora_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let hf = self.get_hf_models_dir();
                hf.parent().unwrap_or(&hf).join("loras")
            })
    }

    /// Get the configured HuggingFace models directory (or default)
    pub fn get_hf_models_dir(&self) -> PathBuf {
        self.huggingface_models_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(Self::default_hf_models_dir)
    }

    /// The separate configuration the ignored tests read, so that they resolve the model
    /// trees on the machine they run on rather than carrying a path of their own. Anchored
    /// to the crate directory, since a test's working directory is not the repository's.
    /// Falls back to the platform defaults when it is absent.
    pub fn load_test() -> Self {
        let p = format!("{}/config.test.toml", env!("CARGO_MANIFEST_DIR"));
        Self::load(Path::new(&p)).unwrap_or_else(|_| Config {
            server: None,
            inference: InferenceConfigToml {
                model_id: "test".into(),
                ..Default::default()
            },
            ollama_models_dir: None,
            huggingface_models_dir: None,
            lora_dir: None,
            energy: None,
            cluster: None,
        })
    }

    /// Save configuration back to TOML file
    pub fn save(&self, path: &Path) -> Result<()> {
        let toml_string = toml::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize config: {}", e))?;
        std::fs::write(path, toml_string)
            .map_err(|e| anyhow!("Failed to write config file: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let toml_str = r#"
            [server]
            host = "0.0.0.0"
            port = 8080

            [inference]
            model_id = "test-model"
            temperature = 0.8
            force_gpu_layers = 12
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server().port, 8080);
        assert_eq!(config.inference.model_id, "test-model");
        // Keys loken now decides still load, and are held aside to be reported.
        let obsolete: Vec<&str> = config
            .inference
            .obsolete
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(obsolete, ["force_gpu_layers", "temperature"]);
    }
}

/// The stores the running daemon was configured with, installed once at start.
///
/// A loader that resolves a checkpoint without a server handle - the sound, video and
/// music pipelines among them - reads these, so that every family looks in the store the
/// configuration names. Without them, one such loader read the test configuration and
/// looked for weights in a store the daemon was never given.
static STORES: std::sync::OnceLock<(PathBuf, PathBuf)> = std::sync::OnceLock::new();

/// Record the Ollama and Hugging Face stores of this process. The first call wins; a
/// server built twice in one process, as tests do, keeps the first.
pub fn install_stores(ollama: &str, huggingface: &str) {
    let _ = STORES.set((PathBuf::from(ollama), PathBuf::from(huggingface)));
}

fn stores() -> (PathBuf, PathBuf) {
    if let Some(installed) = STORES.get() {
        return installed.clone();
    }
    let cfg = Config::load_default().unwrap_or_else(|_| Config::load_test());
    (cfg.get_ollama_models_dir(), cfg.get_hf_models_dir())
}

/// The Ollama store of this process: installed, else the default configuration, else the
/// test one.
pub fn ollama_models_dir() -> PathBuf {
    stores().0
}

/// The Hugging Face store of this process, resolved like [`ollama_models_dir`].
pub fn hf_models_dir() -> PathBuf {
    stores().1
}

#[cfg(test)]
mod store_tests {
    #[test]
    fn the_first_installed_stores_are_what_every_loader_reads() {
        // Another test in this process may have built a server first; whichever stores
        // were installed first are the ones every later reader sees.
        super::install_stores("/stores/ollama", "/stores/hf");
        let (ollama, hf) = (super::ollama_models_dir(), super::hf_models_dir());
        assert!(!ollama.as_os_str().is_empty() && !hf.as_os_str().is_empty());
        super::install_stores("/other/ollama", "/other/hf");
        assert_eq!(super::ollama_models_dir(), ollama);
        assert_eq!(super::hf_models_dir(), hf);
    }
}
