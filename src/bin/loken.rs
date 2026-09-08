//! The loken client: everything a lokend can be asked to do, from a terminal.
//!
//! Every subcommand but the daemon's own is an HTTP call, which is why this binary is a
//! fraction of the daemon's size - it carries no kernels and no model code.

use clap::{Parser, Subcommand};
use std::time::Instant;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use loken::api::{Client, Message, OllamaChatRequest, OllamaGenerateRequest};

/// The loken client.
#[derive(Parser)]
#[command(name = "loken")]
#[command(about = "Talk to a lokend - pull and manage models, chat, generate. Ollama-compatible.")]
#[command(version)]
struct Cli {
    /// Address of the lokend to talk to. 11434 reaches an Ollama instead.
    #[arg(short, long, default_value = "http://localhost:11435")]
    server: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download a model into the daemon's store, from HuggingFace or the Ollama registry
    Pull {
        /// Model ID (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0 or llama3)
        model: String,
        /// Model source: "ollama" or "huggingface" (required)
        #[arg(long, value_parser = ["ollama", "huggingface"])]
        source: Option<String>,
    },

    /// Load a model into memory, so the next request does not pay for it
    Load {
        /// Model ID
        model: String,
    },

    /// List the models the daemon holds
    List,

    /// Remove a model from the store
    Delete {
        /// Model ID
        model: String,
    },

    /// Check a model has every file it needs
    Validate {
        /// Model ID
        model: String,
    },

    /// Re-download whatever a model is missing
    Repair {
        /// Model ID
        model: String,
    },

    /// Chat with a model
    Chat {
        /// Model ID
        #[arg(short, long)]
        model: Option<String>,

        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Message text (for positional arguments)
        message: Vec<String>,
    },

    /// Generate from a prompt, without a chat template
    Generate {
        /// Model ID
        #[arg(short, long)]
        model: Option<String>,

        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Message text (for positional arguments)
        message: Vec<String>,
    },

    /// Show models worth starting from
    Popular,

    /// Show what is currently loaded in memory
    Ps,

    /// Start a coding agent on this daemon: claude or cline
    Launch {
        /// The agent to start
        #[arg(value_parser = ["claude", "cline"])]
        tool: String,

        /// Model the agent talks to; the one loaded model when there is exactly one
        #[arg(short, long)]
        model: Option<String>,

        /// Write the agent's configuration and stop
        #[arg(long)]
        config: bool,

        /// Arguments handed to the agent, after --
        #[arg(last = true)]
        args: Vec<String>,
    },
}

/// The model a launched agent talks to, with the window it is loaded with when it is: the one
/// asked for, else the one loaded model, else the one model held.
async fn launch_model(
    client: &Client,
    asked: Option<String>,
) -> Result<(String, Option<u32>), String> {
    let loaded = client
        .list_loaded_models()
        .await
        .map_err(|e| format!("cannot ask the daemon what is loaded: {e}"))?;
    let window = |name: &str| {
        loaded
            .models
            .iter()
            .find(|m| m.model == name)
            .and_then(|m| m.context_length)
    };
    if let Some(model) = asked {
        let window = window(&model);
        return Ok((model, window));
    }
    if let [one] = loaded.models.as_slice() {
        return Ok((one.model.clone(), one.context_length));
    }
    let held = client
        .list_models()
        .await
        .map_err(|e| format!("cannot list the daemon's models: {e}"))?;
    if let [one] = held.models.as_slice() {
        return Ok((one.name.clone(), None));
    }
    let names: Vec<&str> = if loaded.models.is_empty() {
        held.models.iter().map(|m| m.name.as_str()).collect()
    } else {
        loaded.models.iter().map(|m| m.model.as_str()).collect()
    };
    let has = match names.len() {
        0 => "no model".to_string(),
        n if n > NAMES_SHOWN => format!("{n} models; loken list shows them"),
        _ => names.join(", "),
    };
    Err(format!("pass --model; the daemon has {has}"))
}

/// How many candidate models an error names before pointing at the list.
const NAMES_SHOWN: usize = 8;

/// Configure the agent, then hand the terminal to it unless only the configuration was asked.
async fn launch(
    client: &Client,
    server: &str,
    tool: &str,
    model: Option<String>,
    config_only: bool,
    extra: Vec<String>,
) -> Result<i32, String> {
    use loken::api::launch::{self as setup, Tool};
    let tool = Tool::parse(tool).ok_or_else(|| format!("unknown agent {tool}"))?;
    let (model, window) = launch_model(client, model).await?;
    let api_key = std::env::var("LOKEN_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let launch = setup::Launch::new(server, &model, api_key).with_context(window);

    let mut command = std::process::Command::new(tool.binary());
    match tool {
        Tool::Claude => {
            let env = setup::claude_env(&launch);
            if config_only {
                for (name, value) in &env {
                    println!("export {name}={value:?}");
                }
                println!(
                    "{} {}",
                    tool.binary(),
                    setup::claude_args(&launch).join(" ")
                );
                return Ok(0);
            }
            command.envs(env).args(setup::claude_args(&launch));
        }
        Tool::Cline => {
            let home = dirs::home_dir().ok_or("no home directory")?;
            for path in setup::write_cline(&home, &launch).map_err(|e| e.to_string())? {
                println!("wrote {}", path.display());
            }
            if config_only {
                return Ok(0);
            }
        }
    }
    match window {
        Some(window) => println!("{} on {model} at {server}, window {window}", tool.binary()),
        None => println!("{} on {model} at {server}", tool.binary()),
    }
    match command.args(extra).status() {
        Ok(status) => Ok(status.code().unwrap_or(1)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(format!(
            "{} is not installed. Install it with:\n  {}",
            tool.binary(),
            tool.install_hint()
        )),
        Err(e) => Err(format!("cannot start {}: {e}", tool.binary())),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize tracing
    if cli.verbose {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "loken=debug".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "loken=info".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    let server_addr = cli.server.clone();
    let client = Client::new(cli.server);

    match cli.command {
        Commands::Pull { model, source } => {
            let source = source.unwrap_or_else(|| {
                // Auto-detect based on format: contains "/" = huggingface, else = ollama
                if model.contains('/') {
                    "huggingface".to_string()
                } else {
                    "ollama".to_string()
                }
            });

            info!(
                "Requesting server to pull model: {} (source: {})",
                model, source
            );

            match client.pull_model_with_source(&model, &source).await {
                Ok(response) => {
                    println!("✅ Model pull initiated!");
                    println!("  Source: {}", source);
                    println!("  Status: {}", response.status);
                    if let Some(digest) = response.digest {
                        println!("  Digest: {}", digest);
                    }
                    if let Some(total) = response.total {
                        println!("  Size: {} MB", total / (1024 * 1024));
                    }
                }
                Err(e) => {
                    error!("Failed to pull model: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Load { model } => {
            info!("Requesting server to load model: {}", model);

            match client.load_model(&model).await {
                Ok(response) => {
                    println!("✅ {}", response.message);
                    println!("  Model: {}", response.model);
                    println!("  Status: {}", response.status);
                }
                Err(e) => {
                    eprintln!("❌ Failed to load model: {}", e);
                    eprintln!("Make sure the server is running and the model exists.");
                    std::process::exit(1);
                }
            }
        }

        Commands::List => {
            info!("Requesting model list from server");

            match client.list_models().await {
                Ok(response) => {
                    if response.models.is_empty() {
                        println!("No models downloaded yet.");
                        println!("Use 'loken pull <model-id>' to download a model.");
                    } else {
                        println!("Downloaded models:");
                        for model in response.models {
                            println!(
                                "  {} - {} (modified: {})",
                                model.name, model.size, model.modified_at
                            );
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to list models: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Delete { model } => {
            info!("Requesting server to delete model: {}", model);

            match client.delete_model(&model).await {
                Ok(()) => {
                    println!("✅ Model {} deleted successfully", model);
                }
                Err(e) => {
                    error!("Failed to delete model: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Validate { model } => {
            info!("Validating model: {}", model);

            match client.validate_model(&model).await {
                Ok(result) => {
                    if result.is_valid {
                        println!("✅ Model '{}' is valid", model);
                        println!(
                            "  GGUF: {}, SafeTensors: {}",
                            result.has_gguf, result.has_safetensors
                        );
                        println!(
                            "  Total size: {} bytes ({:.2} MB)",
                            result.total_size,
                            result.total_size as f64 / (1024.0 * 1024.0)
                        );
                        if !result.files.is_empty() {
                            println!("  Files:");
                            for file in result.files {
                                println!("    - {}", file);
                            }
                        }
                    } else {
                        println!("❌ Model '{}' is broken", model);
                        println!("  {}", result.message);
                        println!("  Use 'loken repair {}' to re-download", model);
                    }
                }
                Err(e) => {
                    error!("Failed to validate model: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Repair { model } => {
            info!("Repairing model: {}", model);

            match client.repair_model(&model).await {
                Ok(metadata) => {
                    println!("✅ Model '{}' repaired successfully", model);
                    println!("  Name: {}", metadata.name);
                    println!(
                        "  Size: {} bytes ({:.2} MB)",
                        metadata.size,
                        metadata.size as f64 / (1024.0 * 1024.0)
                    );
                    println!("  Files: {:?}", metadata.files);
                }
                Err(e) => {
                    error!("Failed to repair model: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Chat {
            model,
            prompt,
            message,
        } => {
            let model_name =
                model.unwrap_or_else(|| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());
            let message_text = if !message.is_empty() {
                message.join(" ")
            } else if let Some(p) = prompt {
                p
            } else {
                error!("No message provided. Use --prompt or provide message text.");
                std::process::exit(1);
            };

            info!("Sending chat request to server for model: {}", model_name);

            let request = OllamaChatRequest::new(
                model_name.clone(),
                vec![Message::new("user".to_string(), message_text)],
            );

            let start_time = Instant::now();

            match client.chat(&request).await {
                Ok(response) => {
                    println!("Assistant: {}", response.message.content);
                    if let Some(tokens) = response.eval_count {
                        println!("\n[Tokens: {}]", tokens);
                    }
                }
                Err(e) => {
                    error!("Failed to chat: {}", e);
                    std::process::exit(1);
                }
            }

            let elapsed = start_time.elapsed();
            info!("Request completed in {:.2}s", elapsed.as_secs_f32());
        }

        Commands::Generate {
            model,
            prompt,
            message,
        } => {
            let model_name =
                model.unwrap_or_else(|| "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());
            let prompt_text = if !message.is_empty() {
                message.join(" ")
            } else if let Some(p) = prompt {
                p
            } else {
                error!("No prompt provided. Use --prompt or provide text.");
                std::process::exit(1);
            };

            info!(
                "Sending generate request to server for model: {}",
                model_name
            );

            let request = OllamaGenerateRequest::new(model_name.clone(), prompt_text);

            let start_time = Instant::now();

            match client.generate(&request).await {
                Ok(response) => {
                    println!("{}\n", response.response);
                    if let Some(tokens) = response.eval_count {
                        println!("[Model: {}, Tokens: {}]", response.model, tokens);
                    } else {
                        println!("[Model: {}]", response.model);
                    }
                }
                Err(e) => {
                    error!("Failed to generate: {}", e);
                    std::process::exit(1);
                }
            }

            let elapsed = start_time.elapsed();
            info!("Request completed in {:.2}s", elapsed.as_secs_f32());
        }

        Commands::Popular => {
            // A short, deliberately generic starting point rather than a catalogue. Anything
            // this list names will be superseded, so it stays small and says where to look
            // instead of pretending to rank what is current.
            println!("A few models to start with, from the HuggingFace Hub:");
            println!();
            let models = [
                (
                    "Qwen/Qwen3-0.6B",
                    "Small enough for a laptop CPU, and still answers coherently",
                ),
                (
                    "Qwen/Qwen3-8B",
                    "A general-purpose model that fits one consumer card",
                ),
                (
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "Instruction-tuned, widely used as a baseline",
                ),
                (
                    "microsoft/Phi-4-mini-instruct",
                    "Strong for its size, useful when VRAM is the constraint",
                ),
            ];

            for (id, desc) in models {
                println!("  {}", id);
                println!("    {}", desc);
                println!();
            }
            println!("Use 'loken pull <model-id>' to download a model.");
        }

        Commands::Ps => {
            info!("Requesting loaded models from server");

            match client.list_loaded_models().await {
                Ok(response) => {
                    if response.models.is_empty() {
                        println!("No models currently loaded in memory.");
                        println!("Use 'loken load <model-id>' to load a model.");
                    } else {
                        println!("Loaded models (running in memory):");
                        for model in response.models {
                            println!("  {} - {}", model.model, model.status);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: Failed to list loaded models: {}", e);
                    eprintln!("Make sure the server is running at {}", server_addr);
                    std::process::exit(1);
                }
            }
        }

        Commands::Launch {
            tool,
            model,
            config,
            args,
        } => match launch(&client, &server_addr, &tool, model, config, args).await {
            Ok(code) => std::process::exit(code),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    }

    Ok(())
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
    fn test_pull_command() {
        let args = vec![
            "loken",
            "pull",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--source",
            "huggingface",
        ];
        let cli = Cli::parse_from(args);

        match cli.command {
            Commands::Pull { model, source } => {
                assert_eq!(model, "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
                assert_eq!(source, Some("huggingface".to_string()));
            }
            _ => panic!("Expected Pull command"),
        }
    }

    #[test]
    fn test_chat_command() {
        let args = vec![
            "loken",
            "chat",
            "--model",
            "TinyLlama-1.1B",
            "--prompt",
            "Hello world",
        ];
        let cli = Cli::parse_from(args);

        match cli.command {
            Commands::Chat { model, prompt, .. } => {
                assert_eq!(model, Some("TinyLlama-1.1B".to_string()));
                assert_eq!(prompt, Some("Hello world".to_string()));
            }
            _ => panic!("Expected Chat command"),
        }
    }

    #[test]
    fn launch_takes_the_agent_a_model_and_its_own_arguments() {
        let cli = Cli::parse_from([
            "loken", "launch", "claude", "-m", "qwen3:8b", "--", "-p", "hi",
        ]);
        match cli.command {
            Commands::Launch {
                tool, model, args, ..
            } => {
                assert_eq!(tool, "claude");
                assert_eq!(model, Some("qwen3:8b".to_string()));
                assert_eq!(args, ["-p", "hi"]);
            }
            _ => panic!("Expected Launch command"),
        }
        assert!(Cli::try_parse_from(["loken", "launch", "codex"]).is_err());
    }
}
