//! Configuration file loader
//!
//! Loads configuration from config.toml file

use crate::tensor::DType;
use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::inference::engine::llm_engine::InferenceConfig;

/// Server configuration: a field the file leaves out takes its [`Default`] value.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(default)]
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

fn default_replicas() -> usize {
    1
}

/// Seconds a conversation's kept state stays after its last turn.
pub const DEFAULT_KV_LEASE_SECS: u64 = 1800;

/// The model served when the configuration names none.
pub const DEFAULT_MODEL_ID: &str = "ollama:qwen3:0.6b";

fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}

impl Default for InferenceConfigToml {
    /// Every field at its serde default, as a configuration without an `[inference]` table reads.
    fn default() -> Self {
        toml::from_str("").expect("every field of the inference table has a default")
    }
}

/// Inference configuration from TOML file. Only what loken cannot decide itself: the model it
/// serves by default, where things live, and the features an operator switches on. Sampling,
/// placement, threads and cache formats are decided from the model and the machine.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct InferenceConfigToml {
    #[serde(default = "default_model_id")]
    pub model_id: String,
    /// Copies of a model to serve side by side, each whole on a card of its own, so concurrent
    /// requests run in parallel. One is the model once; a model that does not fit a single
    /// card is served once whatever this says.
    #[serde(default = "default_replicas")]
    pub replicas: usize,
    #[serde(default)]
    pub kv_disk_dir: Option<String>,
    /// The Vulkan devices placement may give layers to, by index or by a part of their name.
    /// Empty offers the ones no other runtime of the build drives.
    #[serde(default)]
    pub vulkan_devices: Vec<String>,
    /// The instance that reads this one's long prompts: it hands back their KV cache and this
    /// one decodes. Both serve the same model. None reads every prompt here.
    #[serde(default)]
    pub prefill_peer: Option<String>,
    /// Whether the cache of each attention layer is cut into ranges of positions over every
    /// card, so a context no single card holds is served whole. Off unless set.
    #[serde(default)]
    pub context_parallel: bool,
    /// How closely the output follows the published model, against speed and memory:
    /// `reference`, `balanced` (the default) or `fast`. It chooses the weight format of float
    /// checkpoints, the cache format, the shifted context, and the value of
    /// `recorded_drafts`, `quality` and `expert_variants` where those are left unset.
    #[serde(default)]
    pub fidelity: crate::inference::engine::llm_engine::Fidelity,
    /// Directory of the on-disk cache of DeepSeek V4.1 decode states, which lets a prompt that
    /// begins with a prefix seen before skip prefilling it. Off unless set. The files hold
    /// state derived from prompts (KV tensors and hidden rows, no text), so they are as private
    /// as the prompts.
    #[serde(default)]
    pub state_cache_dir: Option<String>,
    #[serde(default)]
    pub files_dir: Option<String>,
    #[serde(default)]
    pub moderation_model: Option<String>,
    #[serde(default)]
    pub public_url: Option<String>,
    /// Hosts an image given by URL may be fetched from. Empty fetches nothing.
    #[serde(default)]
    pub image_fetch_hosts: Vec<String>,
    /// Keys this version no longer reads, kept so a config written for an older one still
    /// loads; each is reported once at startup.
    #[serde(flatten, skip_serializing)]
    pub obsolete: std::collections::BTreeMap<String, toml::Value>,
}

/// Energy-reporting configuration (`[energy]` section). All optional with sane
/// defaults so existing config.toml files keep working unchanged.
#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
pub struct EnergyConfig {
    /// Grid carbon intensity (gCO₂eq / kWh) for the CO₂ estimate. Default: France ~50.
    pub carbon_intensity: Option<f64>,
}

/// Root configuration structure
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct Config {
    pub server: Option<ServerConfig>,
    #[serde(default)]
    pub inference: InferenceConfigToml,
    /// The Ollama stores: one directory, or a list of them read as one catalogue.
    pub ollama_models_dir: Option<DirList>,
    /// The Hugging Face stores, as `ollama_models_dir`.
    pub huggingface_models_dir: Option<DirList>,
    /// Where LoRA adapters are kept. Requests name an adapter, never a path, and the
    /// name is resolved inside this directory - see `native_lora::resolve`.
    pub lora_dir: Option<String>,
    #[serde(default)]
    pub energy: Option<EnergyConfig>,
    /// Clustering policy. Absent = this node runs alone, and every cluster path is skipped.
    #[serde(default)]
    pub cluster: Option<ClusterConfigToml>,
}

/// One model directory or several, as `ollama_models_dir` and `huggingface_models_dir` take
/// them: a string, or a list of strings. The first listed receives the models built here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirList(pub Vec<String>);

impl DirList {
    /// A single directory.
    pub fn one(dir: impl Into<String>) -> Self {
        Self(vec![dir.into()])
    }

    /// The directories as paths, empty entries left out.
    pub fn paths(&self) -> Vec<PathBuf> {
        self.0
            .iter()
            .filter(|d| !d.trim().is_empty())
            .map(PathBuf::from)
            .collect()
    }
}

impl<'de> Deserialize<'de> for DirList {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Form {
            One(String),
            Many(Vec<String>),
        }
        Ok(match Form::deserialize(d)? {
            Form::One(dir) => DirList(vec![dir]),
            Form::Many(dirs) => DirList(dirs),
        })
    }
}

impl serde::Serialize for DirList {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self.0.as_slice() {
            [one] => s.serialize_str(one),
            many => serde::Serialize::serialize(many, s),
        }
    }
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
            carbon_intensity: e
                .carbon_intensity
                .unwrap_or(crate::energy_report::DEFAULT_CARBON_INTENSITY),
            ..Default::default()
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
        let fidelity = toml_config.fidelity;

        InferenceConfig {
            model_id: toml_config.model_id.clone(),
            context_length: 0,
            dtype: DType::F16, // Default to F16
            device_index: None,
            replicas: toml_config.replicas.max(1),
            kv_disk_dir: toml_config.kv_disk_dir.clone(),
            prefill_peer: toml_config.prefill_peer.clone(),
            context_parallel: toml_config.context_parallel,
            recorded_drafts: fidelity.recorded_drafts(),
            probe_dump_dir: None,
            expert_profile: None,
            expert_profile_save: None,
            // No fixed budget. The store holds what its disk allows.
            kv_disk_budget_gb: 0.0,
            kv_lease_secs: DEFAULT_KV_LEASE_SECS,
            state_cache_dir: toml_config.state_cache_dir.clone(),
            // The limit is derived from the free space of the cache directory's filesystem.
            state_cache_budget_gb: 0.0,
            files_dir: toml_config.files_dir.clone(),
            moderation_model: toml_config.moderation_model.clone(),
            public_url: toml_config.public_url.clone(),
            media_limits: crate::api::media_limits::MediaLimits {
                fetch_hosts: toml_config.image_fetch_hosts.clone(),
                ..Default::default()
            },
            quality: fidelity.quality(),
            expert_variants: fidelity.expert_variants(),
            fidelity,
            progress_callback: None, // No progress callback from config file
            models_dir: None,        // Will be set by APIServer
            disable_arc_layers: false,
            disable_cuda: false,
            force_cuda_only_layers: false, // Not exposed in config.toml, use programmatically for testing
            kv_quant: crate::inference::engine::llm_engine::KvQuant::Off,
            layer_share: None,
            vulkan_devices: toml_config.vulkan_devices.clone(),
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

    /// The configured Ollama models directories in the order listed, or the default one.
    pub fn get_ollama_models_dirs(&self) -> Vec<PathBuf> {
        let dirs = self
            .ollama_models_dir
            .as_ref()
            .map(DirList::paths)
            .unwrap_or_default();
        if dirs.is_empty() {
            vec![Self::default_ollama_models_dir()]
        } else {
            dirs
        }
    }

    /// The first configured Ollama models directory (or default): where models built here go.
    pub fn get_ollama_models_dir(&self) -> PathBuf {
        self.get_ollama_models_dirs().swap_remove(0)
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

    /// The configured HuggingFace models directories in the order listed, or the default one.
    pub fn get_hf_models_dirs(&self) -> Vec<PathBuf> {
        let dirs = self
            .huggingface_models_dir
            .as_ref()
            .map(DirList::paths)
            .unwrap_or_default();
        if dirs.is_empty() {
            vec![Self::default_hf_models_dir()]
        } else {
            dirs
        }
    }

    /// The first configured HuggingFace models directory (or default).
    pub fn get_hf_models_dir(&self) -> PathBuf {
        self.get_hf_models_dirs().swap_remove(0)
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

    #[test]
    fn a_file_without_an_inference_table_keeps_what_it_sets() {
        let config: Config = toml::from_str(
            r#"
            ollama_models_dir = "/models"
            [server]
            port = 11436
            "#,
        )
        .unwrap();
        assert_eq!(config.ollama_models_dir, Some(DirList::one("/models")));
        assert_eq!(config.server().port, 11436);
        assert_eq!(config.inference.model_id, DEFAULT_MODEL_ID);
        assert_eq!(config.inference.replicas, 1);
    }

    #[test]
    fn a_models_directory_is_one_string_or_a_list() {
        let old: Config = toml::from_str(
            "ollama_models_dir = \"/disk1/ollama\"\nhuggingface_models_dir = \"/disk1/hf\"\n",
        )
        .unwrap();
        assert_eq!(
            old.get_ollama_models_dirs(),
            [PathBuf::from("/disk1/ollama")]
        );
        assert_eq!(old.get_ollama_models_dir(), PathBuf::from("/disk1/ollama"));
        assert_eq!(old.get_hf_models_dir(), PathBuf::from("/disk1/hf"));
        let many: Config = toml::from_str(
            "ollama_models_dir = [\"/disk1/ollama\", \"/disk2/ollama\"]\n\
             huggingface_models_dir = [\"/disk2/hf\", \"\", \"/disk1/hf\"]\n",
        )
        .unwrap();
        assert_eq!(
            many.get_ollama_models_dirs(),
            [
                PathBuf::from("/disk1/ollama"),
                PathBuf::from("/disk2/ollama")
            ]
        );
        assert_eq!(many.get_ollama_models_dir(), PathBuf::from("/disk1/ollama"));
        assert_eq!(
            many.get_hf_models_dirs(),
            [PathBuf::from("/disk2/hf"), PathBuf::from("/disk1/hf")]
        );
        // An empty list is the default directory, as an absent key is.
        let empty: Config = toml::from_str("ollama_models_dir = []\n").unwrap();
        assert_eq!(
            empty.get_ollama_models_dirs(),
            [Config::default_ollama_models_dir()]
        );
        // A single directory is written back in the old form, several as a list.
        #[derive(serde::Serialize)]
        struct Written {
            ollama_models_dir: DirList,
        }
        let one = toml::to_string(&Written {
            ollama_models_dir: DirList::one("/disk1/ollama"),
        })
        .unwrap();
        assert_eq!(one.trim(), "ollama_models_dir = \"/disk1/ollama\"");
        let list = many.ollama_models_dir.clone().unwrap();
        let written = toml::to_string(&Written {
            ollama_models_dir: list.clone(),
        })
        .unwrap();
        let again: Config = toml::from_str(&written).unwrap();
        assert_eq!(again.ollama_models_dir, Some(list));
    }

    #[test]
    fn keys_that_loken_decides_load_and_are_held_aside() {
        let config: Config = toml::from_str(
            "[inference]\nmodel_id = \"m\"\nquality = \"fast\"\nkv_lease_secs = 5\nmax_images = 2\n",
        )
        .unwrap();
        let held: Vec<&str> = config
            .inference
            .obsolete
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(held, ["kv_lease_secs", "max_images", "quality"]);
        // The fidelity and the engine decide these values.
        let resolved = config.to_inference_config();
        assert_eq!(resolved.quality, resolved.fidelity.quality());
        assert_eq!(resolved.kv_lease_secs, DEFAULT_KV_LEASE_SECS);
        assert_eq!(
            resolved.media_limits.max_images,
            crate::api::media_limits::DEFAULT_MAX_IMAGES
        );
    }

    #[test]
    fn the_fidelity_level_sets_the_drafts_the_quality_and_the_expert_files() {
        use crate::inference::engine::llm_engine::{Fidelity, QualityLevel};
        use crate::inference::place::expert_variants::VariantChoice;
        let read = |body: &str| -> Result<Config, toml::de::Error> {
            toml::from_str(&format!("[inference]\nmodel_id = \"m\"\n{body}"))
        };
        let resolved = |body: &str| read(body).unwrap().to_inference_config();

        // No key is `balanced`, and `balanced` is today's behaviour.
        let default = resolved("");
        assert_eq!(default.fidelity, Fidelity::Balanced);
        assert!(!default.recorded_drafts);
        assert_eq!(default.quality, QualityLevel::Exact);
        assert_eq!(default.expert_variants, VariantChoice::Plan);

        let reference = resolved("fidelity = \"reference\"\n");
        assert_eq!(reference.fidelity, Fidelity::Reference);
        assert!(!reference.recorded_drafts);
        assert_eq!(reference.quality, QualityLevel::Exact);
        assert_eq!(reference.expert_variants, VariantChoice::Largest);

        let fast = resolved("fidelity = \"fast\"\n");
        assert_eq!(fast.fidelity, Fidelity::Fast);
        assert!(fast.recorded_drafts);
        assert_eq!(fast.quality, QualityLevel::Fast);
        assert_eq!(fast.expert_variants, VariantChoice::Smallest);

        // The level decides these values. The file does not.
        let ignored = resolved(
            "fidelity = \"fast\"\nrecorded_drafts = false\nquality = \"exact\"\n\
             expert_variants = \"plan\"\n",
        );
        assert!(ignored.recorded_drafts);
        assert_eq!(ignored.quality, QualityLevel::Fast);
        assert_eq!(ignored.expert_variants, VariantChoice::Smallest);

        // Any other level is a startup error that names the three.
        let err = read("fidelity = \"exact\"\n").unwrap_err().to_string();
        for name in ["reference", "balanced", "fast"] {
            assert!(err.contains(name), "{err}");
        }
    }
}

/// The stores the running daemon was configured with, installed once at start.
///
/// A loader that resolves a checkpoint without a server handle - the sound, video and
/// music pipelines among them - reads these, so that every family looks in the store the
/// configuration names. Without them, one such loader read the test configuration and
/// looked for weights in a store the daemon was never given.
/// The directories of each kind, in the order the configuration lists them.
type Stores = (Vec<PathBuf>, Vec<PathBuf>);

static STORES: std::sync::OnceLock<Stores> = std::sync::OnceLock::new();

/// Record the Ollama and Hugging Face directories of this process, in the order the
/// configuration lists them; the daemon calls it once at start. The first call wins. A server
/// built in a test installs nothing, so a test that opens a real checkpoint reads the test
/// configuration's stores.
pub fn install_stores(ollama: &[PathBuf], huggingface: &[PathBuf]) {
    let _ = STORES.set((ollama.to_vec(), huggingface.to_vec()));
}

fn stores() -> Stores {
    if let Some(installed) = STORES.get() {
        return installed.clone();
    }
    let cfg = Config::load_default().unwrap_or_else(|_| Config::load_test());
    (cfg.get_ollama_models_dirs(), cfg.get_hf_models_dirs())
}

/// Every Ollama directory of this process, in the order listed: installed, else the default
/// configuration's, else the test one's. Never empty.
pub fn ollama_models_dirs() -> Vec<PathBuf> {
    stores().0
}

/// Every Hugging Face directory of this process, resolved like [`ollama_models_dirs`].
pub fn hf_models_dirs() -> Vec<PathBuf> {
    stores().1
}

/// The first Ollama directory of this process: where models built here go.
pub fn ollama_models_dir() -> PathBuf {
    stores().0.swap_remove(0)
}

/// The first Hugging Face directory of this process.
pub fn hf_models_dir() -> PathBuf {
    stores().1.swap_remove(0)
}

#[cfg(test)]
mod store_tests {
    #[test]
    fn the_first_installed_stores_are_what_every_loader_reads() {
        // The stores installed are the test configuration's, the ones every test that opens a
        // real checkpoint reads, so this test changes nothing for them.
        let cfg = super::Config::load_test();
        super::install_stores(&cfg.get_ollama_models_dirs(), &cfg.get_hf_models_dirs());
        let (ollama, hf) = (super::ollama_models_dirs(), super::hf_models_dirs());
        assert!(!ollama.is_empty() && !hf.is_empty());
        assert_eq!(super::ollama_models_dir(), ollama[0]);
        assert_eq!(super::hf_models_dir(), hf[0]);
        super::install_stores(
            &[std::path::PathBuf::from("/other/ollama")],
            &[std::path::PathBuf::from("/other/hf")],
        );
        assert_eq!(super::ollama_models_dirs(), ollama);
        assert_eq!(super::hf_models_dirs(), hf);
    }
}
