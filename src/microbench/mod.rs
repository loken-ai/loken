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

pub mod cache_policy;
pub mod cache_replay;
#[cfg(feature = "gpu")]
pub mod card;
#[cfg(feature = "gpu")]
pub mod crossing;
#[cfg(feature = "gpu")]
pub mod decode_inputs;
pub mod decode_layer;
pub mod decode_layout;
pub mod dequant_gemm;
#[cfg(feature = "gpu")]
pub mod expert_ahead;
pub mod expert_read;
#[cfg(feature = "gpu")]
pub mod expert_upload;
pub mod host_expert;
pub mod marlin_rows;
pub mod media_overlap;
pub mod parler_step;
pub mod prefill_layer;
pub mod schedule_host;

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

    /// The feed-forward length, without a file: the expert's for the expert benches, the layer's
    /// for `marlin-rows`; each bench has its own default when this is absent
    #[arg(long)]
    pub inter: Option<usize>,

    /// Weights per quantisation group (`marlin-rows`)
    #[arg(long, default_value_t = 128)]
    pub group_size: usize,

    /// Row counts the GEMM is timed at, 1 variant each (`marlin-rows`)
    #[arg(long, value_delimiter = ',', default_values_t = vec![1, 8, 16, 32, 33, 64, 65, 130, 260, 512])]
    pub rows_list: Vec<usize>,

    /// The row count whose layer time is set against the time of 1 pass (`marlin-rows`)
    #[arg(long, default_value_t = 130)]
    pub compare_rows: usize,

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

    /// The file `expert-read` reads; without it the shapes file.
    /// For `cache-policy`, a host tier's recorded trace, replayed in place of draws from a profile
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

    /// An expert profile, the routing counts per layer a served model writes: the heat
    /// `cache-policy` draws its routing from
    #[arg(long)]
    pub profile: Option<PathBuf>,

    /// The hottest experts per layer the cards hold, which a host tier never serves
    /// (`cache-policy`)
    #[arg(long, default_value_t = 58)]
    pub card_slots: usize,

    /// Slots per layer the host tier is tried with (`cache-policy`)
    #[arg(long, value_delimiter = ',', default_values_t = vec![94, 130, 170])]
    pub slots: Vec<usize>,

    /// The share of a token's experts the read-ahead guess names, one variant for each
    /// (`cache-policy`, `expert-ahead`)
    #[arg(long, value_delimiter = ',', default_values_t = vec![1.0])]
    pub guess_hit: Vec<f64>,

    /// Tokens verified together, a draft's block: the experts the block routes to are read once
    /// for all of it, one variant for each size (`cache-policy`)
    #[arg(long, value_delimiter = ',', default_values_t = vec![1])]
    pub block: Vec<usize>,

    /// The share of a draft token the verification accepts after the one before it was
    /// accepted (`cache-policy`)
    #[arg(long, default_value_t = 0.8)]
    pub accept: f64,

    /// Tokens routed per variant (`cache-policy`)
    #[arg(long, default_value_t = 20000)]
    pub tokens: usize,

    /// The rate the disk reads experts at, in GB/s, to turn misses into time (`cache-policy`)
    #[arg(long, default_value_t = 6.0)]
    pub disk_gbs: f64,

    /// The format of an expert's gate and up weights, for `host-expert`
    #[arg(long, default_value = "iq2_xxs")]
    pub gate_format: String,

    /// The format of an expert's down weights, for `host-expert`
    #[arg(long, default_value = "q2_k")]
    pub down_format: String,

    /// Rows of the prompt batch one layer of `prefill-layer` moves
    #[arg(long, default_value_t = 130)]
    pub rows: usize,

    /// Query heads, without a file (`prefill-layer`, `marlin-rows`)
    #[arg(long)]
    pub heads: Option<usize>,

    /// Key-value heads, without a file: the heads when absent for `prefill-layer`
    /// (`prefill-layer`, `marlin-rows`)
    #[arg(long)]
    pub kv_heads: Option<usize>,

    /// Elements of one head, without a file: the `prefill-layer` and `marlin-rows` shapes, and the
    /// size of the rope rows of `schedule-host` and `decode-inputs`; each bench has its own
    /// default when this is absent
    #[arg(long)]
    pub head_dim: Option<usize>,

    /// Tokens of each sequence's prompt (`schedule-host`)
    #[arg(long, default_value_t = 53)]
    pub prompt_tokens: usize,

    /// Running sequences, one decode row each (`schedule-host`, `decode-inputs`)
    #[arg(long, default_value_t = 64)]
    pub batch: usize,

    /// Tokens of context per sequence at the step timed (`schedule-host`, `decode-inputs`)
    #[arg(long, default_value_t = 512)]
    pub context: usize,

    /// Tokens per block of the paged cache (`schedule-host`, `decode-inputs`)
    #[arg(long, default_value_t = crate::inference::serve::continuous_serve::CB_BLOCK_SIZE)]
    pub block_size: usize,

    /// The width of a block table in `decode-inputs`, in place of the blocks of the context
    #[arg(long)]
    pub blocks_per_row: Option<usize>,

    /// Layers of the stack a step runs (`decode-layer`, `expert-ahead`)
    #[arg(long, default_value_t = crate::microbench::decode_layer::DEFAULT_LAYERS)]
    pub layers: usize,

    /// Entries of the vocabulary the head matmul and the argmax run over (`decode-layer`)
    #[arg(long, default_value_t = crate::microbench::decode_layer::DEFAULT_VOCAB)]
    pub vocab: usize,

    /// Codebooks of the decoder, one embedding and one head each (`parler-step`)
    #[arg(long, default_value_t = 9)]
    pub codebooks: usize,

    /// The rope base of the rope rows (`schedule-host`, `decode-inputs`)
    #[arg(long, default_value_t = 10000.0)]
    pub rope_base: f32,

    /// Write the JSON record here
    #[arg(long)]
    pub json: Option<PathBuf>,

    /// What `cargo bench` appends to the command line; nothing here reads it
    #[arg(long, hide = true)]
    #[serde(skip)]
    pub bench: bool,
}

/// The feed-forward length the expert benches use when `--inter` is absent.
pub const DEFAULT_EXPERT_INTER: usize = 2048;

/// The elements of one attention head when `--head-dim` is absent and the bench has no file.
pub const DEFAULT_HEAD_DIM: usize = 128;

impl Options {
    /// The head size that sizes the rope rows: `--head-dim`, or the default.
    pub fn rope_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(DEFAULT_HEAD_DIM)
    }

    /// The expert's feed-forward length: `--inter`, or the expert benches' default.
    pub fn expert_inter(&self) -> usize {
        self.inter.unwrap_or(DEFAULT_EXPERT_INTER)
    }
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
#[cfg_attr(not(feature = "gpu"), allow(clippy::vec_init_then_push))]
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
            name: "decode-inputs",
            about: "the input upload between two decode steps of a batch: per-array uploads and copies, with and without synchronisations, packed",
            needs_card: true,
            run: decode_inputs::run,
        });
        all.push(Bench {
            name: "decode-layer",
            about: "the non-GEMM kernels of a batched decode step, one by one against the bytes they must move, and one layer of them in a graph",
            needs_card: true,
            run: decode_layer::run,
        });
        all.push(Bench {
            name: "dequant-gemm",
            about: "the 4-bit AWQ Marlin GEMM against the weight dequantised to f16 once and a dense f16 GEMM, by row count: the rows from which dequantising pays",
            needs_card: true,
            run: dequant_gemm::run,
        });
        all.push(Bench {
            name: "marlin-rows",
            about: "the 4-bit AWQ Marlin GEMM of one layer's 4 projections by row count: passes over the weights and streaming rate",
            needs_card: true,
            run: marlin_rows::run,
        });
        all.push(Bench {
            name: "expert-ahead",
            about: "the experts of the next layer copied to the card while the current layer runs: no prefetch, the last token's set, the early router, by experts missing and lead time",
            needs_card: true,
            run: expert_ahead::run,
        });
        all.push(Bench {
            name: "expert-upload",
            about: "one routed expert's weights host to card: compute or copy stream, pageable or pinned, the card idle or busy",
            needs_card: true,
            run: expert_upload::run,
        });
        all.push(Bench {
            name: "prefill-layer",
            about: "the data movement of one prefill layer, the copies, q and k norms, rotation and store write: separate launches against one fused launch",
            needs_card: true,
            run: prefill_layer::run,
        });
    }
    #[cfg(all(feature = "gpu", feature = "audio"))]
    all.push(Bench {
        name: "parler-step",
        about: "one decode step of a Parler-TTS decoder, its projections in F32 and in F16",
        needs_card: true,
        run: parler_step::run,
    });
    #[cfg(feature = "gpu")]
    all.push(Bench {
        name: "media-overlap",
        about: "2 denoise steps of 2 requests on 1 card: one after the other, on 2 streams, and as 1 step of twice the rows",
        needs_card: true,
        run: media_overlap::run,
    });
    all.push(Bench {
        name: "host-expert",
        about: "one routed expert computed on the host cores from memory: the floor of a miss once its pages are there",
        needs_card: false,
        run: host_expert::run,
    });
    all.push(Bench {
        name: "cache-policy",
        about: "the experts a host tier misses per token, for a policy and a size, on a routing drawn from an expert profile",
        needs_card: false,
        run: cache_policy::run,
    });
    all.push(Bench {
        name: "schedule-host",
        about: "the host work between 2 decode steps of the continuous-batching worker: schedule, items, decode inputs",
        needs_card: false,
        run: schedule_host::run,
    });
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

    /// The value of `r`, or its default once its error is noted.
    pub fn value<T: Default, E: std::fmt::Debug>(&self, r: Result<T, E>) -> T {
        match r {
            Ok(v) => v,
            Err(e) => {
                self.ok(Err::<(), E>(e));
                T::default()
            }
        }
    }

    /// The value of `r` once it is `Ok`, `None` after noting its error.
    pub fn some<T, E: std::fmt::Debug>(&self, r: Result<T, E>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                self.ok(Err::<(), E>(e));
                None
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
