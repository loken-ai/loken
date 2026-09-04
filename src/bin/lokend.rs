//! The LOKEN daemon: `lokend serve`.
//!
//! Starts the HTTP server and handles model management and inference requests over the
//! Ollama- and OpenAI-compatible protocols.

use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use loken::{
    api::APIServer,
    config::Config,
    cpu::detect_cpu_topology,
    distributed::{DeviceManager, HardwareTopology},
    gpu::{GPUManagerImpl, GPUManagerInterface},
};

/// The loken daemon: serves every modality over one HTTP port.
#[derive(Parser)]
#[command(name = "lokend")]
#[command(
    about = "Serve language, image, audio and video models locally. OpenAI- and Ollama-compatible."
)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP server
    Serve {
        /// HTTP server port
        #[arg(short, long, default_value = "11435")]
        port: u16,

        /// Ollama model store to serve from (the directory holding `manifests/` and
        /// `blobs/`). Defaults to the platform's usual location.
        #[arg(short, long)]
        models_dir: Option<String>,

        /// Enable verbose logging
        #[arg(short, long)]
        verbose: bool,

        /// Default keep-alive duration for loaded models (in minutes).
        /// Use -1 to keep models loaded indefinitely.
        /// Can also be set via OLLAMA_KEEP_ALIVE environment variable.
        /// Examples: "5", "10m", "1h", "30s", "-1" (forever)
        #[arg(long, value_name = "DURATION")]
        keep_alive: Option<String>,

        /// Run CPU-only: ignore all GPUs and place every layer on the CPU.
        /// Lets one cuda-built binary serve either GPU or CPU without a rebuild.
        #[arg(long)]
        cpu: bool,
    },
}

// vec_init_then_push fires inside main() on the #[cfg]-gated feature_flags
// builder (see ~line 198) - clippy doesn't reason across cfg attributes.
// Allowing on the #[cfg]-gated push site itself doesn't propagate; the
// fn-level allow is the documented escape hatch.
#[allow(clippy::vec_init_then_push)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Keep freed large scratch buffers resident in the malloc arena instead of
    // munmap'ing them back to the OS. CPU prefill re-allocates big per-layer
    // activation/output buffers (e.g. a 2B model's ~16 MB gate/up output) every
    // layer; returning them to the OS on free forces the kernel to re-fault fresh
    // anonymous pages on the next layer, and concurrent faults contend on the mm
    // page-table lock - measured ~18% of granite-2B CPU prefill in page faults +
    // native_queued_spin_lock_slowpath. Retaining the arena reuses the already-
    // faulted pages: granite-2B prefill 3735 -> 3059 ms (-18%), flipping from
    // 1.11x behind ollama to 0.92x ahead. glibc-specific; a no-op elsewhere.
    #[cfg(target_env = "gnu")]
    unsafe {
        libc::mallopt(libc::M_TRIM_THRESHOLD, -1);
        libc::mallopt(libc::M_MMAP_THRESHOLD, 256 * 1024 * 1024);
    }

    // cuBLAS reserves a per-handle workspace on first use; without an
    // explicit size the runtime picks one that yields non-deterministic
    // output across runs at temp=0 (observed on gemma4:26b - same
    // prompt, different generated tokens). Setting :4096:8 reserves
    // 4 KiB x 8 buffers per handle and pins the algo to a deterministic
    // selector. Must be set BEFORE any CUDA context is created (i.e.,
    // before the first cublas call from any thread).
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").is_err() {
        std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
    }

    let cli = Cli::parse();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                // `server` is THIS binary's own target. Without it the default filter
                // admits only the library, so everything the startup path says - the
                // address it bound, and the warning that the API is unauthenticated -
                // was dropped before reaching the terminal. A diagnostic nobody can see
                // is not a diagnostic, and a security warning nobody can see is worse.
                .unwrap_or_else(|_| "lokend=info,loken=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    match cli.command {
        Commands::Serve {
            port,
            models_dir,
            verbose,
            keep_alive,
            cpu,
        } => {
            if verbose {
                warn!("Verbose logging enabled");
            }

            // Must be set before ANY GPUManagerImpl is constructed so detection
            // reports zero GPUs and placement uses the CPU path.
            if cpu {
                loken::gpu::set_force_cpu(true);
                info!("🖥️  --cpu: GPU disabled, serving on CPU only");
            }

            info!("Starting lokend on port {}", port);
            info!(
                "Models directory: {}",
                models_dir.clone().unwrap_or_else(|| {
                    loken::config::Config::default_ollama_models_dir()
                        .to_string_lossy()
                        .to_string()
                })
            );

            // Load configuration from config.toml
            info!("");
            info!("📝 Loading configuration from config.toml...");
            let mut config = match Config::load_default() {
                Ok(config) => {
                    info!("✅ Successfully loaded config.toml");
                    config
                }
                Err(e) => {
                    warn!("⚠️  Failed to load config.toml: {}. Using defaults.", e);
                    warn!(
                        "   Make sure config.toml exists in the current directory: {:?}",
                        std::env::current_dir().unwrap_or_default()
                    );
                    // Create default config if file doesn't exist
                    Config {
                        cluster: None,
                        server: None,
                        lora_dir: None,
                        inference: loken::config::InferenceConfigToml {
                            model_id: "ollama:qwen3:0.6b".to_string(),
                            model_source: Some("ollama".to_string()),
                            max_tokens: Some(512),
                            context_length: Some(4096),
                            temperature: Some(0.7),
                            top_p: Some(0.9),
                            top_k: Some(50),
                            seed: Some(42),
                            device_index: None,
                            draft_model: None,
                            draft_device_index: None,
                            kv_shift_reuse: false,
                            max_gpu_memory_fraction: Some(0.9),
                            force_gpu_layers: None,
                            use_quantized_gpu: Some(false),
                            cpu_threads: Some(0),
                            disable_arc_layers: None,
                            kv_quant: None,
                            continuous_batching: None,
                        },
                        ollama_models_dir: None,
                        huggingface_models_dir: None,
                        energy: None,
                    }
                }
            };

            // Before anything else can touch rayon. Whoever builds the global pool first wins,
            // and until now that was never this call: it sat at the top of the first model
            // load, found the pool already up, and warned into a log nobody reads while every
            // host kernel ran on one thread per logical core instead of per physical one.
            loken::inference::engine::llm_engine::configure_thread_pool(
                config.inference.cpu_threads.unwrap_or(0),
            );

            if let Some(dir) = models_dir.as_ref() {
                config.ollama_models_dir = Some(dir.clone());
            }

            // Energy reporting: initialise the global settings from config so every
            // request measures + reports energy (systematic, awareness-oriented).
            let energy_settings = config.energy_settings();
            loken::energy_report::init_settings(energy_settings.clone());
            if energy_settings.enabled {
                info!(
                    "⚡ Energy reporting: ENABLED (carbon intensity {:.0} gCO₂/kWh{})",
                    energy_settings.carbon_intensity,
                    if energy_settings.cpu_tdp_w > 0.0 {
                        format!(
                            ", CPU estimate {:.0} W when RAPL unavailable",
                            energy_settings.cpu_tdp_w
                        )
                    } else {
                        String::new()
                    }
                );
            } else {
                info!("⚡ Energy reporting: disabled ([energy] enabled=false in config)");
            }

            let inference_config = config.to_inference_config();

            // Log loaded configuration
            info!("✅ Configuration loaded:");
            info!("   • Model: {}", inference_config.model_id);
            info!("   • Max tokens: {}", inference_config.max_tokens);
            info!("   • Temperature: {}", inference_config.temperature);

            // Performance settings
            if let Some(forced_layers) = inference_config.force_gpu_layers {
                info!(
                    "   🎯 OVERRIDE: force_gpu_layers = {} (from config.toml)",
                    forced_layers
                );
            } else {
                info!("   • GPU layers: AUTO (will calculate based on available memory)");
            }

            if inference_config.use_quantized_gpu {
                info!("   ⚠️  use_quantized_gpu = true (EXPERIMENTAL - may crash!)");
            } else {
                info!("   • Quantized GPU: Disabled (safe f16 dequantization)");
            }

            info!(
                "   • CPU threads: {}",
                if inference_config.cpu_threads == 0 {
                    "AUTO".to_string()
                } else {
                    inference_config.cpu_threads.to_string()
                }
            );
            info!("");

            // Initialize NVML and detect GPUs
            info!("🔧 Initializing NVML (NVIDIA Management Library)...");
            let gpu_manager = GPUManagerImpl::new();

            // Start GPU detection
            info!("🔍 Detecting GPUs...");
            if let Err(e) = gpu_manager.detect_gpus().await {
                warn!("GPU detection failed: {}", e);
            } else {
                let devices = gpu_manager.get_devices();
                if devices.is_empty() {
                    info!("📋 No NVIDIA GPUs detected");
                } else {
                    info!("📋 Detected {} NVIDIA GPU(s):", devices.len());
                    for device in &devices {
                        let mem_mb = device
                            .memory_info()
                            .map(|m| m.total / (1024 * 1024))
                            .unwrap_or(0);
                        info!("   ✓ {} ({} MB)", device.name(), mem_mb);
                    }
                }
            }

            // Surface compile-time feature set so operators can see at
            // a glance whether they're getting the CUDA, OpenCL,
            // or CPU-only build they expect. Each push is #[cfg]-gated;
            // the Vec stays empty for a minimal CPU-only build.
            // (vec_init_then_push allow lives at fn-level: clippy doesn't
            // reason across cfg gates and the per-statement allow doesn't
            // attach to the right span here.)
            let mut feature_flags: Vec<&'static str> = Vec::new();
            #[cfg(feature = "cuda")]
            feature_flags.push("cuda");
            #[cfg(feature = "opencl")]
            feature_flags.push("opencl");
            #[cfg(feature = "cpu")]
            feature_flags.push("cpu");
            if feature_flags.is_empty() {
                info!("🧱 Compiled features: (none - minimal CPU-only build)");
            } else {
                info!("🧱 Compiled features: {}", feature_flags.join(", "));
            }

            // Check CUDA availability for inference
            #[cfg(feature = "cuda")]
            let cuda_available = true;
            #[cfg(not(feature = "cuda"))]
            let cuda_available = false;

            if cuda_available {
                info!("🚀 CUDA Available: Yes (GPU acceleration enabled)");
            } else {
                info!("🚀 CUDA Available: No (running in CPU-only mode)");
                #[cfg(not(feature = "cuda"))]
                info!("   💡 To enable CUDA, build with: cargo build --features cuda");
            }

            // Detect CPU topology (P-cores vs E-cores)
            info!("");
            let _cpu_topology = detect_cpu_topology();

            // Run comprehensive hardware discovery for distributed inference
            info!("");
            let mut device_manager = DeviceManager::new();
            if let Err(e) = device_manager.detect_devices() {
                warn!("Device detection failed: {}", e);
            }
            let topology = HardwareTopology::from_device_manager(&device_manager);
            topology.log_summary();

            // Get configured directories from config or use OS-aware defaults
            let ollama_models_dir = config.get_ollama_models_dir().to_string_lossy().to_string();
            let huggingface_models_dir = config.get_hf_models_dir().to_string_lossy().to_string();

            // Adapters are named, not pathed: fix the one directory they resolve inside
            // before any request can arrive.
            let lora_dir = config.get_lora_dir();
            let n_loras = {
                loken::inference::load::lora::set_lora_dir(lora_dir.clone());
                loken::inference::load::lora::available().len()
            };
            info!(
                "LoRA directory: {} ({n_loras} adapters)",
                lora_dir.display()
            );

            // Create API server with config and optional keep_alive override
            let mut api_server = if let Some(ref keep_alive_str) = keep_alive {
                // Parse the keep_alive value
                let keep_alive_minutes = loken::api::parse_keep_alive(keep_alive_str).unwrap_or(5); // Default to 5 minutes if parsing fails
                info!("Custom keep_alive set to {} minutes", keep_alive_minutes);
                APIServer::with_config_and_keep_alive(
                    ollama_models_dir,
                    huggingface_models_dir,
                    inference_config,
                    keep_alive_minutes,
                )
            } else {
                APIServer::with_config(ollama_models_dir, huggingface_models_dir, inference_config)
            };

            // -- Join the cluster, if one is configured ------------------------------
            //
            // Absent section = this node runs alone and nothing below happens, which is what
            // keeps the single-machine path exactly what it was.
            if let Some(cc) = config.cluster.clone() {
                use loken::distributed::{cluster, cluster_runtime, discovery};
                // A node id nobody had to type: the hostname is already unique on a network
                // and is what an operator reads in a log anyway.
                let node_id = if cc.node_id.is_empty() {
                    hostname_or("node")
                } else {
                    cc.node_id.clone()
                };
                let advertise = if cc.advertise.is_empty() {
                    format!("http://{}:{}", node_id, port)
                } else {
                    cc.advertise.clone()
                };
                let cfg = cluster::ClusterConfig {
                    join: cc.join.clone(),
                    gossip_interval_ms: cc.gossip_interval_ms,
                    min_speedup: cc.min_speedup,
                    node_id: node_id.clone(),
                };
                let handle = cluster::Cluster::new(cfg);
                api_server.attach_cluster(handle.clone());

                let book = std::sync::Arc::new(std::sync::Mutex::new(
                    discovery::PeerBook::with_seeds(cc.join.clone()),
                ));
                cluster_runtime::spawn_discovery(
                    cc.name.clone(),
                    node_id.clone(),
                    advertise.clone(),
                    book.clone(),
                    std::time::Duration::from_millis(cc.gossip_interval_ms),
                );
                cluster_runtime::spawn_gossip(handle, book, api_server.cluster_started());
                info!(
                    "🌐 Cluster '{}' as {node_id}, advertising {advertise}, {} seed(s), \
                     hand-over above {:.2}x",
                    cc.name,
                    cc.join.len(),
                    cc.min_speedup
                );
            }

            // Start periodic stats monitoring
            api_server.start_stats_monitoring();
            info!("Stats monitoring started (logging every 10 seconds)");

            // Apply the configured authentication policy BEFORE the router is built -
            // the middleware captures it, so configuring afterwards would compile,
            // change nothing, and look done.
            {
                let (required, keys, origins, rpm, burst) = config
                    .server
                    .as_ref()
                    .map(|sv| {
                        (
                            sv.require_auth,
                            sv.api_keys.clone(),
                            sv.allowed_origins.clone(),
                            sv.rate_limit_per_minute,
                            sv.rate_limit_burst,
                        )
                    })
                    .unwrap_or((false, Vec::new(), Vec::new(), 0, 10));
                api_server.configure_auth(required, &keys, &origins, rpm, burst);
                if rpm > 0 {
                    info!("Rate limit: {rpm} requests/minute per client, burst {burst}");
                } else {
                    info!(
                        "Rate limit OFF (set [server] rate_limit_per_minute to cap per-client \
                         request rate)"
                    );
                }
                if required {
                    if keys.is_empty() {
                        warn!(
                            "[server] require_auth is on with no api_keys: every request \
                             will be refused. That is the safe direction for a typo, but \
                             it is probably not what was meant."
                        );
                    } else {
                        info!(
                            "API authentication ON ({} key(s), {} allowed browser origin(s))",
                            keys.len(),
                            origins.len()
                        );
                    }
                } else {
                    info!("API authentication OFF (set [server] require_auth = true to enable)");
                }
            }

            // Create router
            let router = api_server.create_router();

            // Start server. Disable Nagle on every accepted connection: each
            // streamed token is a small write, and with Nagle on the kernel can
            // hold it waiting for the peer's ACK, adding ~1-2 ms/token of
            // streaming latency for clients that ACK lazily. Go's net/http (what
            // Ollama uses) sets TCP_NODELAY by default; hyper does not. tap_io
            // applies it per-connection.
            use axum::serve::ListenerExt;
            // BIND WHERE THE CONFIG SAYS. This was hardcoded to 0.0.0.0, so the
            // server listened on every interface whatever `[server] host` declared -
            // and the default it advertises is 127.0.0.1. A config that promises
            // loopback and a process that answers the whole network is the kind of gap
            // nobody notices until the machine is reachable from somewhere it should
            // not be.
            // `[server]` is optional in the file; its absence means the documented
            // default, which is loopback.
            let host = config
                .server
                .as_ref()
                .map(|s| s.host.trim().to_string())
                .filter(|h| !h.is_empty())
                .unwrap_or_else(|| "127.0.0.1".to_string());
            let host = host.as_str();
            // Loopback means only this machine can reach it; anything else is at least
            // the local network. The API has no authentication, and it can DELETE
            // models, so say so plainly rather than leaving it to be discovered.
            let loopback = host == "127.0.0.1" || host == "::1" || host == "localhost";
            if !loopback {
                warn!(
                    "The API is listening on {host} - reachable beyond this machine - and \
                     it is NOT authenticated. Anything that can reach this port can \
                     generate, pull models and DELETE them. Bind 127.0.0.1 in \
                     [server] host, or put it behind something that authenticates."
                );
            }
            let listener = tokio::net::TcpListener::bind(format!("{host}:{}", port))
                .await?
                .tap_io(|stream| {
                    if let Err(e) = stream.set_nodelay(true) {
                        tracing::debug!("set_nodelay on incoming connection failed: {e:#}");
                    }
                });

            info!("Server listening on http://{host}:{}", port);
            info!("Ollama-compatible endpoints:");
            info!("  GET  /api/tags          - List available models");
            info!("  POST /api/pull          - Pull a model (Ollama or HuggingFace)");
            info!("  DELETE /api/delete      - Delete a model");
            info!("  POST /api/show          - Show model info");
            info!("  POST /api/chat          - Chat completion (streaming supported)");
            info!("  POST /api/generate      - Generate text (streaming supported)");
            info!("  POST /api/embed         - Generate embeddings");
            info!("  GET  /api/ps            - List running models");
            info!("  GET  /api/version       - Server version");
            info!("  (Model load: use POST /api/generate with an empty prompt)");
            info!("");
            info!("OpenAI-compatible endpoints:");
            info!("  GET  /v1/models               - List models");
            info!("  POST /v1/chat/completions     - Chat completion");
            info!("  POST /v1/completions          - Text completion");
            info!("  POST /v1/embeddings           - Embeddings");
            info!("  POST /v1/rerank               - Rerank documents");
            info!("  POST /v1/messages             - Anthropic Messages protocol");
            info!("");
            info!("Generative media:");
            info!("  POST /v1/images/generations   - Generate an image");
            info!("  POST /v1/images/edits         - Edit an image");
            info!("  POST /v1/images/variations    - Vary an image");
            info!("  POST /v1/audio/speech         - Text to speech");
            info!("  POST /v1/audio/transcriptions - Transcribe audio");
            info!("  POST /v1/audio/translations   - Translate audio");
            info!("  POST /v1/audio/generations    - Generate music or sound");
            info!("  POST /v1/audio/separate       - Separate audio sources");
            info!("  GET  /v1/audio/voices         - List available voices");
            info!("  POST /v1/video/generations    - Generate a video");
            info!("  GET  /v1/renders              - Track a running render");
            info!("");
            info!("  GET  /health - Health check");

            // Graceful shutdown: wait for SIGINT / SIGTERM, then stop
            // accepting new connections and let in-flight requests
            // drain. Without this, a `systemctl stop` mid-generation
            // would drop the streaming response on the floor.
            // Stop accepting, let in-flight WORK finish, and exit even if a socket
            // stays open. `with_graceful_shutdown` alone waits on connections, so one
            // idle keep-alive client is enough to hang the stop for good - which is
            // what happens in practice, and it costs the clean unload this path exists
            // to give. Racing it against the drain watcher below bounds that.
            let drained = drain_when_idle(api_server.clone());
            tokio::pin!(drained);
            tokio::select! {
                r = axum::serve(listener, router)
                    .with_graceful_shutdown(shutdown_signal()) => r?,
                () = &mut drained => info!(
                    "no work left in flight; exiting without waiting for open connections"
                ),
            }
            // Exit rather than return: unwinding waits on the runtime's blocking tasks and on
            // CUDA teardown, which can outlast the port a replacement needs to bind.
            info!("stopped");
            std::process::exit(0);
        }
    }

    Ok(())
}

/// Resolves once a stop was asked for AND nothing is being served any more.
///
/// The pair to `shutdown_signal`: that one says when to stop taking work, this one says
/// when the work is done. Polled rather than notified, because "nothing in flight" is a
/// state the gates already publish and no event announces.
/// A stop waits for WORK, not for sockets - and a second signal ends the wait.
///
/// A render is legitimately long, so waiting for one is the right default: dropping a video
/// at the last step to save a few seconds of shutdown helps nobody. But the wait had no
/// escape, and a stop sent mid-render therefore never returned. An operator who restarts
/// anyway - the natural thing to do when a stop appears hung - is left with a process that
/// still holds every byte of its VRAM, so the NEW one plans against whatever is left. Three
/// such processes were found alive at once holding 16 GB between them, and the placements
/// measured against them were wrong in a way nothing in the log explained.
///
/// The escape is a second signal rather than a grace period, because no single duration is
/// right for both a chat completion and an hour-long render: any constant either abandons
/// real work or fails to bound the wait. The operator knows which one they are in; the
/// process does not. What the process CAN do is say what it is waiting for, which is the
/// other half of the fix - a silent wait is indistinguishable from a hang, and that is the
/// reading that produced the orphans.
async fn drain_when_idle(api: APIServer) {
    let mut stops = StopSignals::install();
    let first = stops.next().await;
    tracing::info!("{first}: no longer accepting work, waiting for what is in flight");
    tracing::info!("send it again to exit immediately and abandon in-flight work");

    // Slow enough to be free, fast enough that a stop feels immediate.
    const POLL: std::time::Duration = std::time::Duration::from_millis(250);
    // A wait nobody can explain is a wait nobody trusts. Reported on a period long enough
    // not to fill a log during a normal drain and short enough that a long one is visibly
    // PROGRESSING rather than stuck.
    const REPORT_EVERY: std::time::Duration = std::time::Duration::from_secs(15);
    let started = std::time::Instant::now();
    let mut next_report = REPORT_EVERY;

    loop {
        if !api.work_in_flight().await {
            tracing::info!("drained in {:.1}s", started.elapsed().as_secs_f32());
            return;
        }
        tokio::select! {
            again = stops.next() => {
                tracing::warn!(
                    "{again} while draining after {:.0}s: exiting now, in-flight work is \
                     abandoned",
                    started.elapsed().as_secs_f32()
                );
                return;
            }
            () = tokio::time::sleep(POLL) => {}
        }
        if started.elapsed() >= next_report {
            tracing::info!(
                "still draining after {:.0}s (a render finishes when it finishes; send the \
                 stop signal again to exit now)",
                started.elapsed().as_secs_f32()
            );
            next_report += REPORT_EVERY;
        }
    }
}

/// Resolves when either SIGINT (Ctrl-C) or SIGTERM arrives. On
/// Windows, only Ctrl-C fires since SIGTERM has no native equivalent.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received SIGINT, draining in-flight requests..."),
        _ = terminate => tracing::info!("Received SIGTERM, draining in-flight requests..."),
    }
}

/// Both stop signals as ONE source that can be awaited repeatedly.
///
/// The listeners are installed once and kept for the process's life, so a second signal
/// delivered while the drain is already running is QUEUED rather than missed. Registering a
/// fresh listener after the first signal instead would leave a window - short, but exactly
/// the one an operator hits pressing Ctrl-C twice - in which the second signal is dropped
/// and the stop the operator asked for never happens.
struct StopSignals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
}

impl StopSignals {
    fn install() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            Self {
                interrupt: signal(SignalKind::interrupt()).expect("install SIGINT handler"),
                terminate: signal(SignalKind::terminate()).expect("install SIGTERM handler"),
            }
        }
        #[cfg(not(unix))]
        Self {}
    }

    /// Resolves on the next stop signal, naming it for the log.
    async fn next(&mut self) -> &'static str {
        #[cfg(unix)]
        {
            tokio::select! {
                _ = self.interrupt.recv() => "SIGINT",
                _ = self.terminate.recv() => "SIGTERM",
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c()
                .await
                .expect("install Ctrl-C handler");
            "Ctrl-C"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert()
    }

    #[test]
    fn test_serve_command() {
        let args = vec![
            "lokend",
            "serve",
            "--port",
            "9000",
            "--models-dir",
            "/tmp/models",
            "--verbose",
        ];

        let cli = Cli::parse_from(args);

        match cli.command {
            Commands::Serve {
                port,
                models_dir,
                verbose,
                keep_alive,
                ..
            } => {
                assert_eq!(port, 9000);
                assert_eq!(models_dir.as_deref(), Some("/tmp/models"));
                assert!(verbose);
                assert!(keep_alive.is_none());
            }
        }
    }

    #[test]
    fn test_keep_alive_flag() {
        let args = vec!["lokend", "serve", "--keep-alive", "10m"];

        let cli = Cli::parse_from(args);

        match cli.command {
            Commands::Serve { keep_alive, .. } => {
                assert_eq!(keep_alive, Some("10m".to_string()));
            }
        }
    }
}

/// The machine's name, or a fallback. Used as the node id when none is configured, so a
/// cluster of three machines needs no per-machine configuration at all.
fn hostname_or(fallback: &str) -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_else(|| fallback.to_string())
}
