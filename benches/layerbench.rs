//! The layer bench: replay decode steps through a few layers of a model file, placed as a served
//! model is, and report where a token's time goes. Run as
//! `cargo bench --features cuda,microbench --bench layerbench -- <file> --layers <range> [options]`.
//! See docs/dev/layer-bench.md.

use clap::Parser;
use loken::inference::bench::{args::Args, parse_layers, run, Controls};

fn main() {
    let a = Args::parse();
    let layers = parse_layers(&a.layers).unwrap_or_else(|e| fail(&e.to_string()));
    let controls = Controls {
        file: a.file,
        layers,
        tokens: a.tokens,
        warm: a.warm,
        placement: a.placement,
        page_cache: a.page_cache,
        readahead: a.readahead,
        card_experts: a.card_experts,
        guess_hit: a.guess_hit,
        seed: a.seed,
        expert_profile: a.expert_profile,
        json: a.json,
    };
    let record = run(&controls).unwrap_or_else(|e| fail(&e.to_string()));
    print!("{}", record.table());
    if let Some(path) = &controls.json {
        if let Err(e) = std::fs::write(path, record.to_json()) {
            fail(&format!("{}: {e}", path.display()));
        }
        println!("record written to {}", path.display());
    }
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}
