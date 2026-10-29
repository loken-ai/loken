//! The layer bench's command line: `cargo bench --features cuda,microbench --bench layerbench -- <file> --layers <range> [options]`.

use super::{PageCache, Placement};

/// Replay decode steps through a few layers of a model file, placed as a served model is, and
/// report where a token's time goes. A developer's measurement: no server is involved.
#[derive(clap::Parser, Debug)]
pub struct Args {
    /// The model file; the parts of a split file are found beside it
    pub file: std::path::PathBuf,

    /// The layers to hold, `from..to` (the end excluded) or one index
    #[arg(long)]
    pub layers: String,

    /// Decode steps timed
    #[arg(long, default_value_t = 32)]
    pub tokens: usize,

    /// Decode steps run before the timed ones
    #[arg(long, default_value_t = 4)]
    pub warm: usize,

    /// Where the layers go: auto (as served), host (no card), cards (no host tier)
    #[arg(long, default_value = "auto")]
    pub placement: Placement,

    /// The page cache of the held layers' experts when the steps start: warm or cold
    #[arg(long, default_value = "warm")]
    pub page_cache: PageCache,

    /// The expert read-ahead depth, in layers; without it the model's own controller
    #[arg(long)]
    pub readahead: Option<usize>,

    /// Routed experts per held layer the cards keep; without it the share a served layer gets
    #[arg(long)]
    pub card_experts: Option<usize>,

    /// The share of a token's experts its read-ahead guess names, 0 to 1
    #[arg(long, default_value_t = 1.0)]
    pub guess_hit: f64,

    /// The seed of the synthetic stream and of the routing drawn
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// An expert profile whose counts give the routing heat, in place of the file's prior
    #[arg(long)]
    pub expert_profile: Option<std::path::PathBuf>,

    /// Write the JSON record here
    #[arg(long)]
    pub json: Option<std::path::PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The controls parse with their defaults and their overrides.
    #[test]
    fn arguments_parse() {
        let a = Args::parse_from(["layerbench", "m.gguf", "--layers", "10..13"]);
        assert_eq!(a.layers, "10..13");
        assert_eq!((a.tokens, a.warm), (32, 4));
        assert_eq!(a.placement, Placement::Auto);
        assert_eq!(a.page_cache, PageCache::Warm);
        assert_eq!(a.readahead, None);
        assert_eq!(a.guess_hit, 1.0);
        let a = Args::parse_from([
            "layerbench",
            "m.gguf",
            "--layers",
            "7",
            "--placement",
            "cards",
            "--page-cache",
            "cold",
            "--readahead",
            "2",
            "--card-experts",
            "40",
            "--tokens",
            "8",
        ]);
        assert_eq!(a.placement, Placement::Cards);
        assert_eq!(a.page_cache, PageCache::Cold);
        assert_eq!(
            (a.readahead, a.card_experts, a.tokens),
            (Some(2), Some(40), 8)
        );
        assert!(Args::try_parse_from([
            "layerbench",
            "m.gguf",
            "--layers",
            "1",
            "--placement",
            "gpu"
        ])
        .is_err());
    }
}
