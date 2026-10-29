//! Microbenches: one mechanism each, isolated, on synthetic data of real shapes, its variants
//! side by side in one run so that one run settles one hypothesis. No model file is loaded and
//! no server runs. A bench runs each variant for a target wall time and reports the median and
//! spread of the time per call, the bandwidth where bytes move, and a JSON record that carries
//! the build, the machine and the card, so two commits compare.
//!
//! The benches are driven by `benches/microbench.rs`, compiled with the `microbench` feature
//! alone, and reach the crate's own primitives: the card's streams and events, the mapped-file
//! advice, the GGUF header reader. A bench joins with a module and a line in [`benches`].

pub mod record;
pub mod shapes;
pub mod stats;

#[cfg(feature = "gpu")]
pub mod card;
#[cfg(feature = "gpu")]
pub mod crossing;
pub mod expert_read;
#[cfg(feature = "gpu")]
pub mod expert_upload;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// What a run is told: the bench, the shapes, and how long each variant runs.
#[derive(clap::Parser, Clone, Debug, Serialize, Deserialize, PartialEq)]
#[command(name = "microbench")]
#[command(about = "Time one mechanism of the inference server at a real model's shapes")]
pub struct Options {
    /// The bench to run, or `list`
    pub name: String,

    /// A GGUF whose metadata and tensor table give the shapes; no weight is loaded
    #[arg(long)]
    pub shapes_from: Option<PathBuf>,

    /// The layer whose routed experts give the expert shapes; without it the first layer that
    /// has routed experts
    #[arg(long)]
    pub layer: Option<usize>,

    /// The expert of that layer the file-reading variants start at
    #[arg(long, default_value_t = 0)]
    pub expert: usize,

    /// One tensor of the file in place of the layer's experts: its bytes are the span read or
    /// uploaded
    #[arg(long)]
    pub tensor: Option<String>,

    /// Elements of one activation row, without a file
    #[arg(long, default_value_t = 4096)]
    pub hidden: usize,

    /// The hyper-connection multiple, without a file: rows the residual stream carries per token
    #[arg(long, default_value_t = 1)]
    pub hc: usize,

    /// Bytes per element of an activation row
    #[arg(long, default_value_t = 4)]
    pub element_bytes: usize,

    /// The expert's feed-forward length, without a file
    #[arg(long, default_value_t = 2048)]
    pub inter: usize,

    /// Bits per weight of a stored expert, without a file: 4.5 for q4_k, 8.5 for q8_0
    #[arg(long, default_value_t = 4.5)]
    pub weight_bits: f64,

    /// The bytes of each projection, without a file: one value or three, in place of the
    /// feed-forward length and the bits per weight
    #[arg(long, value_delimiter = ',')]
    pub projection_bytes: Vec<u64>,

    /// Rows the packed variants of `crossing` move in one copy
    #[arg(long, default_value_t = 8)]
    pub pack: usize,

    /// The file `expert-read` reads; without it the shapes file
    #[arg(long)]
    pub file: Option<PathBuf>,

    /// Where in that file the first expert read starts, when the file is not the shapes file
    #[arg(long)]
    pub offset: Option<u64>,

    /// Threads of the threaded variants; without it the processors the host reports
    #[arg(long)]
    pub threads: Option<usize>,

    /// The card the card benches run on
    #[arg(long, default_value_t = 0)]
    pub card: usize,

    /// Wall time each variant runs for, in seconds, after a warm-up of a tenth of it
    #[arg(long, default_value_t = 1.0)]
    pub seconds: f64,

    /// Write the JSON record here
    #[arg(long)]
    pub json: Option<PathBuf>,

    /// What `cargo bench` appends to the command line; nothing here reads it
    #[arg(long, hide = true)]
    #[serde(skip)]
    pub bench: bool,
}

/// One variant's calls, as a bench answers them.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    pub variant: String,
    /// The bytes one call moves or reads, when the variant moves any.
    pub bytes: Option<u64>,
    /// The time of each timed call, in microseconds. Empty when the variant did not run.
    pub samples_us: Vec<f64>,
    /// Why the variant did not run, or what its figures leave out.
    pub note: Option<String>,
}

/// What a bench runs with.
pub struct Context<'a> {
    pub options: &'a Options,
    pub shapes: &'a shapes::Shapes,
    /// The card opened for a bench that needs one.
    #[cfg(feature = "gpu")]
    pub card: Option<&'a card::Card>,
}

/// One bench: its name on the command line, what it answers, and the run.
pub struct Bench {
    pub name: &'static str,
    pub about: &'static str,
    pub needs_card: bool,
    pub run: fn(&Context) -> Result<Vec<Measured>, String>,
}

/// Every bench, in the order `list` shows them.
pub fn benches() -> Vec<Bench> {
    let mut all = Vec::new();
    #[cfg(feature = "gpu")]
    {
        all.push(Bench {
            name: "crossing",
            about: "one activation row host to card and back: synchronous, asynchronous, packed, and a bare synchronisation",
            needs_card: true,
            run: crossing::run,
        });
        all.push(Bench {
            name: "expert-upload",
            about: "one routed expert's weights host to card: compute or copy stream, pageable or pinned, the card idle or busy",
            needs_card: true,
            run: expert_upload::run,
        });
    }
    all.push(Bench {
        name: "expert-read",
        about: "one routed expert read from disk with its pages dropped first: faults, advice, populate, direct reads",
        needs_card: false,
        run: expert_read::run,
    });
    all
}

/// The first error a sequence of calls met, kept so that a timing closure stays a closure.
#[derive(Default)]
pub struct Fail(std::cell::RefCell<Option<String>>);

impl Fail {
    /// Note `r`'s error, the first one only.
    pub fn ok<E: std::fmt::Debug>(&self, r: Result<(), E>) {
        if let Err(e) = r {
            let mut g = self.0.borrow_mut();
            if g.is_none() {
                *g = Some(format!("{e:?}"));
            }
        }
    }

    /// `Err` naming `what` once an error was noted.
    pub fn check(&self, what: &str) -> Result<(), String> {
        match self.0.borrow().as_ref() {
            Some(e) => Err(format!("{what}: {e}")),
            None => Ok(()),
        }
    }
}

/// Run the bench `opts` names and answer its record. The table is printed by the caller.
pub fn run(opts: &Options) -> Result<record::Record, String> {
    let bench = benches()
        .into_iter()
        .find(|b| b.name == opts.name)
        .ok_or_else(|| format!("no bench `{}`; `list` names them", opts.name))?;
    if !opts.seconds.is_finite() || opts.seconds <= 0.0 {
        return Err("--seconds must be above zero".to_string());
    }
    let shapes = shapes::resolve(opts)?;
    #[cfg(feature = "gpu")]
    let card = if bench.needs_card {
        Some(card::open(opts.card)?)
    } else {
        None
    };
    let cx = Context {
        options: opts,
        shapes: &shapes,
        #[cfg(feature = "gpu")]
        card: card.as_ref(),
    };
    let measured = (bench.run)(&cx)?;
    #[cfg(feature = "gpu")]
    let card_record = card.as_ref().map(|c| record::CardRecord {
        ordinal: c.dev.ordinal(),
        name: c.name.clone(),
    });
    #[cfg(not(feature = "gpu"))]
    let card_record = None;
    Ok(record::Record::new(
        bench.name,
        opts,
        &shapes,
        card_record,
        measured,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn the_options_parse_the_shapes_and_swallow_cargo_bench_flag() {
        let o = Options::parse_from([
            "microbench",
            "crossing",
            "--hidden",
            "7168",
            "--hc",
            "4",
            "--pack",
            "13",
            "--seconds",
            "0.5",
            "--bench",
        ]);
        assert_eq!(o.name, "crossing");
        assert_eq!((o.hidden, o.hc, o.pack), (7168, 4, 13));
        assert_eq!(o.seconds, 0.5);
        assert!(o.bench);
        assert!(o.shapes_from.is_none());
        let o = Options::parse_from(["microbench", "list"]);
        assert_eq!(o.name, "list");
        assert!(Options::try_parse_from(["microbench"]).is_err());
    }

    #[test]
    fn the_options_serialise_without_the_cargo_flag() {
        let o = Options::parse_from(["microbench", "expert-read", "--threads", "3"]);
        let json = serde_json::to_string(&o).unwrap();
        assert!(json.contains("\"threads\":3"));
        assert!(!json.contains("\"bench\""));
    }

    #[test]
    fn an_unknown_bench_and_a_zero_target_are_refused() {
        let o = Options::parse_from(["microbench", "nothing"]);
        assert!(run(&o).unwrap_err().contains("no bench `nothing`"));
        let o = Options::parse_from(["microbench", "expert-read", "--seconds", "0"]);
        assert!(run(&o).unwrap_err().contains("--seconds"));
    }

    #[test]
    fn every_bench_has_a_distinct_name() {
        let names: Vec<&str> = benches().iter().map(|b| b.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len());
        assert!(names.contains(&"expert-read"));
    }

    #[test]
    fn the_first_failure_is_the_one_kept() {
        let f = Fail::default();
        assert!(f.check("x").is_ok());
        f.ok::<&str>(Ok(()));
        f.ok(Err("first"));
        f.ok(Err("second"));
        assert_eq!(f.check("copy").unwrap_err(), "copy: \"first\"");
    }
}
