//! The layer bench: a decode replayed through a few layers of a model file, with the placement
//! a served model gets, so that where a token's time goes is read in minutes rather than after
//! a load of the whole model.
//!
//! The layers are held from the file's mapping alone. The input is a synthetic stream, and the
//! routing is the model's own distribution replayed from the file's prior or an expert profile,
//! so the experts read, uploaded and missed are the ones a served decode reads. The report is
//! the stage profile the server keeps, the crossings at the card seam, the expert tiers'
//! counters and the kernel's counts of faults and disk reads, as a table and as a JSON record
//! that carries the build, the machine and the controls.
//!
//! A model joins through [`subject::Subject`]: the first is the streamed MoE the offload
//! serves, and another streamed or resident model adds an implementation and a line in
//! [`subject::open`].

pub mod args;
pub mod counters;
pub mod deepseek;
pub mod report;
pub mod routing;
pub mod subject;

use std::path::PathBuf;

/// Where the layers' weights and experts go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// The placement a served model gets: the cards present take what they hold in service,
    /// the host tier keeps the rest.
    Auto,
    /// No card: every step on the host, the experts through the host tier.
    Host,
    /// The cards as in service, with no host tier: an expert off the cards is read from the file.
    Cards,
}

impl std::str::FromStr for Placement {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "host" => Ok(Self::Host),
            "cards" => Ok(Self::Cards),
            other => Err(format!("placement `{other}` is not auto, host or cards")),
        }
    }
}

/// The page cache's state for the held layers' experts when the steps start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageCache {
    /// Every routed expert of the held layers read into memory first.
    Warm,
    /// Every routed expert of the held layers dropped from memory first.
    Cold,
}

impl std::str::FromStr for PageCache {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "warm" => Ok(Self::Warm),
            "cold" => Ok(Self::Cold),
            other => Err(format!("page cache `{other}` is not warm or cold")),
        }
    }
}

/// What a run is told.
#[derive(Clone, Debug, PartialEq)]
pub struct Controls {
    /// The model file.
    pub file: PathBuf,
    /// The layers held, `[from, to)`.
    pub layers: (usize, usize),
    /// Decode steps timed.
    pub tokens: usize,
    /// Decode steps run before the timed ones, for the kernels and caches to settle.
    pub warm: usize,
    pub placement: Placement,
    pub page_cache: PageCache,
    /// The expert read-ahead depth, in layers; `None` leaves it to the model's controller.
    pub readahead: Option<usize>,
    /// Routed experts per held layer the cards keep; `None` gives each layer the share a served
    /// model's layers get.
    pub card_experts: Option<usize>,
    /// The share of a token's experts its read-ahead guess names.
    pub guess_hit: f64,
    /// The seed of the synthetic stream and routing.
    pub seed: u64,
    /// An expert profile whose counts give the routing heat.
    pub expert_profile: Option<PathBuf>,
    /// Where the JSON record goes; `None` prints the table alone.
    pub json: Option<PathBuf>,
}

/// The layers `from..to` or a single index `i` as `[i, i + 1)`.
pub fn parse_layers(s: &str) -> Result<(usize, usize), String> {
    let bad = |why: &str| format!("layers `{s}`: {why}");
    let (from, to) = match s.split_once("..") {
        Some((a, b)) => (
            a.trim()
                .parse::<usize>()
                .map_err(|_| bad("the first layer is not a number"))?,
            b.trim()
                .parse::<usize>()
                .map_err(|_| bad("the end is not a number"))?,
        ),
        None => {
            let i = s
                .trim()
                .parse::<usize>()
                .map_err(|_| bad("not a layer index nor a range from..to"))?;
            (i, i + 1)
        }
    };
    if from >= to {
        return Err(bad("the range is empty"));
    }
    Ok((from, to))
}

impl Controls {
    /// The tokens scripted: the warm steps and the timed ones.
    pub fn steps(&self) -> usize {
        self.warm + self.tokens
    }
}

/// Run the bench under `controls`: load, place, replay, and answer the record. The table is
/// printed by the caller.
pub fn run(controls: &Controls) -> Result<report::Record, String> {
    use crate::inference::offload::{stage_prof_enable, stage_prof_snapshot};
    if controls.tokens == 0 {
        return Err("no token to time".to_string());
    }
    let mut subject = subject::open(controls)?;
    let model = subject.describe();
    let heat = subject.routing_heat();
    let activated = model.activated;
    let script = routing::script(
        &heat,
        activated,
        controls.steps(),
        controls.guess_hit,
        controls.seed,
    );
    subject.script(script)?;
    subject.page_cache(controls.page_cache)?;
    let crossings = subject.crossings();

    for step in 0..controls.warm {
        subject.step(step)?;
    }
    stage_prof_enable(true);
    crossings.reset();
    let tiers_before = subject.tier_stats();
    let process_before = counters::ProcessCounters::now();
    let mut ms = Vec::with_capacity(controls.tokens);
    for step in controls.warm..controls.steps() {
        let started = std::time::Instant::now();
        subject.step(step)?;
        ms.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let process = counters::ProcessCounters::now().since(&process_before);
    let tiers = subject.tier_stats();
    stage_prof_enable(false);
    let stages: Vec<report::StageRow> = stage_prof_snapshot()
        .into_iter()
        .map(|(stage, ns, calls)| report::StageRow {
            stage: stage.to_string(),
            total_us: ns / 1000,
            calls,
        })
        .collect();
    let expert_uploads = stages
        .iter()
        .filter(|r| r.stage == "card expert upload")
        .map(|r| r.calls)
        .sum();
    let steps = crossings
        .snapshot()
        .into_iter()
        .map(|(step, count)| report::StepRow {
            step: step.to_string(),
            count,
        })
        .collect();
    let model = subject.describe();
    Ok(report::Record {
        build: report::Build::this(),
        machine: report::Machine::this(),
        disk: report::Disk::of(&controls.file),
        controls: report::ControlsRecord::from(controls),
        model,
        timing: report::Timing::of(ms),
        stages,
        counters: report::Counters {
            steps,
            expert_uploads,
            tier_hits: tiers.0.saturating_sub(tiers_before.0),
            tier_misses: tiers.1.saturating_sub(tiers_before.1),
            tier_evictions: tiers.2.saturating_sub(tiers_before.2),
            process,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A range and a single index parse; an empty or malformed one is refused with its text.
    #[test]
    fn layers_parse_as_a_range_or_one_index() {
        assert_eq!(parse_layers("10..13"), Ok((10, 13)));
        assert_eq!(parse_layers(" 3 .. 4 "), Ok((3, 4)));
        assert_eq!(parse_layers("7"), Ok((7, 8)));
        assert!(parse_layers("5..5").unwrap_err().contains("empty"));
        assert!(parse_layers("5..2").unwrap_err().contains("empty"));
        assert!(parse_layers("a..2").unwrap_err().contains("a..2"));
        assert!(parse_layers("x").unwrap_err().contains("not a layer index"));
    }

    /// The placement and page cache names parse and name themselves back.
    #[test]
    fn placement_and_page_cache_parse() {
        for p in [Placement::Auto, Placement::Host, Placement::Cards] {
            assert_eq!(p.name().parse::<Placement>(), Ok(p));
        }
        for c in [PageCache::Warm, PageCache::Cold] {
            assert_eq!(c.name().parse::<PageCache>(), Ok(c));
        }
        assert!("gpu".parse::<Placement>().is_err());
        assert!("hot".parse::<PageCache>().is_err());
    }
}
