//! The microbenches' command line. Run as
//! `cargo bench --features cuda,microbench --bench microbench -- <name> [options]`, and
//! `-- list` to name the benches. One mechanism per bench, its variants side by side, no model
//! loaded and no server started. See docs/guides/microbench.md.

use clap::Parser;
use loken::microbench::{benches, run, Options};

fn main() {
    let opts = Options::parse();
    if opts.name == "list" {
        for b in benches() {
            let card = if b.needs_card { " (needs a card)" } else { "" };
            println!("{:<15} {}{card}", b.name, b.about);
        }
        return;
    }
    let record = match run(&opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    print!("{}", record.table());
    if let Some(path) = &opts.json {
        if let Err(e) = std::fs::write(path, record.to_json()) {
            eprintln!("error: {}: {e}", path.display());
            std::process::exit(1);
        }
        println!("record written to {}", path.display());
    }
}
