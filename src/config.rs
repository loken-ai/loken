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

/// Inference configuration from TOML file
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct InferenceConfigToml {
    pub model_id: String,
    pub model_source: Option<String>, // "ollama" or "huggingface"
    pub max_tokens: Option<usize>,
    pub context_length: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    pub device_index: Option<usize>,
    pub draft_model: Option<String>,
    pub draft_device_index: Option<usize>,
    #[serde(default)]
    pub kv_shift_reuse: bool,
    #[serde(default)]
    pub kv_snapshots: usize,
    #[serde(default)]
    pub kv_disk_dir: Option<String>,
    #[serde(default)]
    pub kv_disk_budget_gb: f64,
    pub max_gpu_memory_fraction: Option<f64>,

    // Performance settings
    pub force_gpu_layers: Option<usize>,
    pub use_quantized_gpu: Option<bool>,
    pub cpu_threads: Option<usize>,

    // Device testing/optimization
    pub disable_arc_layers: Option<bool>, // Skip Arc/OpenCL layers (test mode)

    // KV-cache storage quantization: "off" (default), "q8", or "q4".
    pub kv_quant: Option<String>,

    /// Continuous batching (paged KV) for CONCURRENT request throughput.
    /// Default OFF: single-stream decode is faster on the serial path;
    /// turn on for multi-client serving (aggregate throughput regime).
    /// GPU-only; ignored on CPU models.
    pub continuous_batching: Option<bool>,
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
            max_tokens: toml_config.max_tokens.unwrap_or(2048),
            context_length: toml_config.context_length.unwrap_or(4096),
            temperature: toml_config.temperature.unwrap_or(0.7) as f32,
            top_p: toml_config.top_p.unwrap_or(0.9) as f32,
            top_k: toml_config.top_k.unwrap_or(50),
            seed: toml_config.seed.unwrap_or(42),
            dtype: DType::F16, // Default to F16
            device_index: toml_config.device_index,
            draft_model: toml_config.draft_model.clone(),
            draft_device_index: toml_config.draft_device_index,
            kv_shift_reuse: toml_config.kv_shift_reuse,
            kv_snapshots: toml_config.kv_snapshots,
            kv_disk_dir: toml_config.kv_disk_dir.clone(),
            kv_disk_budget_gb: toml_config.kv_disk_budget_gb,
            max_gpu_memory_fraction: toml_config.max_gpu_memory_fraction.unwrap_or(0.9),

            // Performance settings
            force_gpu_layers: toml_config.force_gpu_layers,
            use_quantized_gpu: toml_config.use_quantized_gpu.unwrap_or(true),
            cpu_threads: toml_config.cpu_threads.unwrap_or(0),
            progress_callback: None, // No progress callback from config file
            models_dir: None,        // Will be set by APIServer
            disable_arc_layers: toml_config.disable_arc_layers.unwrap_or(false),
            disable_cuda: false,
            force_cuda_only_layers: false, // Not exposed in config.toml, use programmatically for testing
            repeat_penalty: 1.1,           // Ollama default
            repeat_last_n: 64,             // Ollama default
            kv_quant: parse_kv_quant(toml_config.kv_quant.as_deref()),
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
                model_source: None,
                max_tokens: None,
                context_length: None,
                temperature: None,
                top_p: None,
                top_k: None,
                seed: None,
                device_index: None,
                draft_model: None,
                draft_device_index: None,
                kv_shift_reuse: false,
                kv_snapshots: 0,
                kv_disk_dir: None,
                kv_disk_budget_gb: 0.0,
                max_gpu_memory_fraction: None,
                force_gpu_layers: None,
                use_quantized_gpu: None,
                cpu_threads: None,
                disable_arc_layers: None,
                kv_quant: None,
                continuous_batching: None,
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

/// Parse a `kv_quant` string into the engine enum. Accepts the same
/// aliases as the runtime `options.kv_quant` override on the API
/// (extract_kv_quant_override in api/handlers/) so the same string in
/// config.toml and in a per-request override resolves identically.
/// Unknown / missing -> Off (no quantization, F-dtype KV cache).
fn parse_kv_quant(raw: Option<&str>) -> crate::inference::engine::llm_engine::KvQuant {
    use crate::inference::engine::llm_engine::KvQuant;
    let Some(s) = raw else {
        return KvQuant::Off;
    };
    match s.trim().to_ascii_lowercase().as_str() {
        "off" | "none" | "f16" | "f32" => KvQuant::Off,
        "q8" | "q8_0" => KvQuant::Q8,
        "q4" | "q4_0" => KvQuant::Q4,
        _ => KvQuant::Off,
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
            max_tokens = 128
            temperature = 0.8
            force_gpu_layers = 12
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server().port, 8080);
        assert_eq!(config.inference.model_id, "test-model");
        assert_eq!(config.inference.force_gpu_layers, Some(12));
    }

    #[test]
    fn parse_kv_quant_matches_handler_aliases() {
        use crate::inference::engine::llm_engine::KvQuant;
        // None / unknown / explicit-off variants
        assert!(matches!(parse_kv_quant(None), KvQuant::Off));
        assert!(matches!(parse_kv_quant(Some("off")), KvQuant::Off));
        assert!(matches!(parse_kv_quant(Some("NONE")), KvQuant::Off));
        assert!(matches!(parse_kv_quant(Some("f16")), KvQuant::Off));
        assert!(matches!(parse_kv_quant(Some("F32")), KvQuant::Off));
        // Q8 aliases
        assert!(matches!(parse_kv_quant(Some("q8")), KvQuant::Q8));
        assert!(matches!(parse_kv_quant(Some("Q8")), KvQuant::Q8));
        assert!(matches!(parse_kv_quant(Some("Q8_0")), KvQuant::Q8));
        // Q4 aliases
        assert!(matches!(parse_kv_quant(Some("q4")), KvQuant::Q4));
        assert!(matches!(parse_kv_quant(Some("Q4_0")), KvQuant::Q4));
        // Whitespace-tolerant (TOML strings may carry trailing whitespace
        // from in-file comments or copy-paste mishaps).
        assert!(matches!(parse_kv_quant(Some("  q4  ")), KvQuant::Q4));
        // Unknown falls back to Off (safest - never silently downgrade
        // precision the user didn't ask for).
        assert!(matches!(parse_kv_quant(Some("q2")), KvQuant::Off));
        assert!(matches!(parse_kv_quant(Some("nonsense")), KvQuant::Off));
    }

    #[test]
    fn to_inference_config_respects_kv_quant_aliases() {
        // Parse a minimal config TOML that carries the alias; verifies that
        // both the TOML deserialiser and to_inference_config() round-trip
        // 'Q8_0' (handler-side alias) into the engine enum.
        let toml_str = r#"
            [inference]
            model_id = "x"
            kv_quant = "Q8_0"
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        let inferred = config.to_inference_config();
        use crate::inference::engine::llm_engine::KvQuant;
        assert!(matches!(inferred.kv_quant, KvQuant::Q8));
    }
}
