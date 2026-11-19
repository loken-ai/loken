//! The Sinkhorn wide-residual mixture (the DeepSeek V4.1 family) as a bench subject: a range of
//! its layers loaded through the generic route, placed as the engine places a model, and decoded
//! from a synthetic boundary with scripted routes.

use super::counters::Crossings;
use super::report::ModelRecord;
use super::routing::Script;
use super::{Controls, PageCache, Placement};
use crate::inference::engine::llm_engine::{KvQuant, PREFILL_CHUNK_TOKENS};
use crate::inference::generic_transformer::expert_profile::ExpertProfile;
use crate::inference::generic_transformer::layer_bench::{StepIn, WideFacts};
use crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta;
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::place::expert_variants::VariantChoice;
use crate::inference::place::ladder::{cards_of, load_placed, Demand};
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::quantized::gguf_source::SplitGguf;
use crate::tensor::Device;
use std::collections::HashMap;
use std::sync::Arc;

/// Where the routing heat came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeatSource {
    Profile,
    Prior,
    Uniform,
}

impl HeatSource {
    fn name(self) -> &'static str {
        match self {
            Self::Profile => "the expert profile",
            Self::Prior => "the file's routing prior",
            Self::Uniform => "a uniform draw (no prior, no profile)",
        }
    }
}

pub struct WideSubject {
    model: GenericHeteroTransformer,
    architecture: String,
    from: usize,
    to: usize,
    /// Layers of the stack.
    layers: usize,
    /// Whether the plan puts a layer on a card.
    cards_opened: bool,
    /// The profile's counts per held layer, when one was read.
    profile_heat: Option<Vec<Vec<u64>>>,
    heat: HeatSource,
    /// The raw ids of the positions run so far, which the engram rows of a step read.
    history: Vec<u32>,
    pos: usize,
    rng: super::routing::Lcg,
    /// The generic route has no card seam to count: the counts stay empty.
    crossings: Arc<Crossings>,
}

impl WideSubject {
    /// The layers `[from, to)` of `src` loaded and placed under `controls`.
    pub fn open(src: SplitGguf, controls: &Controls, architecture: String) -> Result<Self, String> {
        let (from, to) = controls.layers;
        if controls.placement == Placement::Cards {
            return Err(
                "placement cards: the generic route keeps the host tier a served load keeps \
                 for the experts of a wide residual, so it places as auto does; ask for auto"
                    .to_string(),
            );
        }
        if controls.readahead.is_some() {
            return Err(
                "readahead: the generic route has no read-ahead depth to pin for the experts of \
                 a wide residual"
                    .to_string(),
            );
        }
        let content = src.merged_content();
        let meta = WideSparseMeta::from_file(&content).map_err(|e| format!("config: {e}"))?;
        let n = meta.n_layers;
        if to > n {
            return Err(format!(
                "layers [{from}, {to}) reach past the {n} of the stack"
            ));
        }
        // A band layer reads the compressed keys a source layer below it publishes within the
        // token: a range can only start where that state is empty or made.
        let starts_clean = |l: usize| {
            meta.compress_ratios.get(l).copied().unwrap_or(0) == 0
                || meta.kv_source_layers.contains(&l)
        };
        if from != 0 && !starts_clean(from) {
            let starts: Vec<usize> = (0..n).filter(|&l| l == 0 || starts_clean(l)).collect();
            return Err(format!(
                "layer {from} reads the compressed keys of a layer below it; a range starts at one of {starts:?}"
            ));
        }
        let profile = match &controls.expert_profile {
            Some(path) => {
                let profile = ExpertProfile::read(path)
                    .map_err(|e| format!("expert profile {}: {e}", path.display()))?;
                if profile.layers != n || profile.experts != meta.mixture.routed {
                    return Err(format!(
                        "expert profile {}: {} layers of {} experts, the model has {} of {}",
                        path.display(),
                        profile.layers,
                        profile.experts,
                        n,
                        meta.mixture.routed
                    ));
                }
                Some(profile)
            }
            None => None,
        };

        let started = std::time::Instant::now();
        let (mut model, plan) = Self::load(&src, &content, &meta, controls)?;
        tracing::info!(
            "layers [{from}, {to}) of {n} held in {:.1}s",
            started.elapsed().as_secs_f32()
        );
        let placement = plan
            .segments
            .iter()
            .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!("{architecture} loaded: layers {placement}");

        // The prior a companion file declares, then the profile's ranking, as the engine does.
        model.locate_routing_prior(&controls.file);
        if let Some(profile) = &profile {
            model
                .prime_routing_heat(profile)
                .map_err(|e| format!("expert profile: {e}"))?;
        }
        let facts = model
            .bench_facts()
            .ok_or_else(|| "the loaded model is not a wide residual".to_string())?;
        if let Some(per_layer) = controls.card_experts {
            // The room each card gives the routed experts: the share of the held layers asked for.
            let rooms = model.wide_card_rooms().to_vec();
            let bytes = per_layer as u64 * (to - from) as u64 * facts.expert_bytes;
            model.set_wide_card_rooms(
                rooms
                    .iter()
                    .enumerate()
                    .map(|(i, &(ordinal, _))| (ordinal, if i == 0 { bytes } else { 0 }))
                    .collect(),
            );
        }
        // The cards that pay keep the hottest experts, each in the variant the policy gives it.
        model.place_wide_experts(VariantChoice::default());

        let profile_heat = profile.as_ref().map(|p| {
            (from..to)
                .map(|l| p.layer(l).iter().map(|&c| u64::from(c)).collect())
                .collect()
        });
        let heat = match (&profile_heat, model.routing_prior()) {
            (Some(_), _) => HeatSource::Profile,
            (None, prior) if prior.iter().any(|l| !l.is_empty()) => HeatSource::Prior,
            _ => HeatSource::Uniform,
        };
        let cards_opened = plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::Cuda(_)));
        Ok(Self {
            model,
            architecture,
            from,
            to,
            layers: n,
            cards_opened,
            profile_heat,
            heat,
            history: Vec::new(),
            pos: 0,
            rng: super::routing::Lcg::new(controls.seed),
            crossings: Arc::new(Crossings::default()),
        })
    }

    /// The model of the layers `controls.layers`, on the plan the engine gives a share of a wide
    /// residual: the mixers on the cards the machine offers when they hold them, else the host.
    ///
    /// The cards' room is priced for the held layers' mixers and the engine's planning window of
    /// the cache.
    fn load(
        src: &SplitGguf,
        content: &crate::tensor::quantized::gguf_file::Content,
        meta: &WideSparseMeta,
        controls: &Controls,
    ) -> Result<(GenericHeteroTransformer, HeteroPlan), String> {
        let (from, to) = controls.layers;
        let held = to - from;
        let n = meta.n_layers;
        let ctx = Some(controls.steps() + 1);
        let cards_wanted = controls.placement == Placement::Auto && crate::gpu::cards_allowed();
        let mixers =
            crate::inference::generic_transformer::wide_sparse_plan::card_mixer_bytes(content)
                .ok_or_else(|| "the file describes no wide residual".to_string())?;
        let demand = Demand {
            layers: held,
            block_bytes: mixers.per_layer[from..to].iter().sum(),
            head_bytes: 0,
            kv_bytes_per_layer: meta
                .kv_bytes_per_layer(crate::inference::cache::KV_WORKING_WINDOW_TOKENS),
            runtime_reserve: crate::inference::place::runtime_demand::llm_work_reserve(
                meta.d_model,
                PREFILL_CHUNK_TOKENS,
            ),
        };
        // Without the cards the ladder plans every held layer on the host.
        let cards = match cards_wanted {
            true => {
                crate::inference::place::llm::offer(
                    crate::inference::place::llm::OWNER,
                    None,
                    demand.whole(),
                    mixers.largest_upload,
                )
                .cards
            }
            false => Vec::new(),
        };
        let mut last = None;
        let m = load_placed(
            "layer bench",
            &demand,
            &cards,
            || false,
            |plan| {
                let placed = plan.shifted(from, n);
                let mut devices: HashMap<usize, Device> = HashMap::new();
                for card in cards_of(plan) {
                    devices.insert(card, Device::new_cuda(card)?);
                }
                let mut m = GenericHeteroTransformer::from_split_gguf(
                    src,
                    &devices,
                    &placed,
                    KvQuant::Off,
                    ctx,
                    false,
                    false,
                )?;
                let (kv, reserve) = (demand.kv_bytes_per_layer, demand.runtime_reserve);
                m.note_wide_card_rooms(&placed, &[], kv, reserve);
                last = Some(placed);
                Ok::<_, crate::tensor::Error>(m)
            },
        )
        .map_err(|e| format!("load layers [{from}, {to}): {e}"))?;
        let placed = last.ok_or_else(|| "no plan was built".to_string())?;
        Ok((m, placed))
    }

    fn facts(&self) -> WideFacts {
        self.model.bench_facts().unwrap_or(WideFacts {
            layers: self.layers,
            held: (self.from, self.to),
            routed: 0,
            activated: 0,
            vocab: 0,
            expert_bytes: 0,
            card_experts: Vec::new(),
            host_slots: Vec::new(),
        })
    }
}

impl super::subject::Subject for WideSubject {
    fn describe(&self) -> ModelRecord {
        let facts = self.facts();
        ModelRecord {
            architecture: self.architecture.clone(),
            layers: facts.layers,
            held: facts.held,
            routed_experts: facts.routed,
            activated: facts.activated,
            expert_bytes: facts.expert_bytes,
            cards_opened: self.cards_opened,
            host_slots_per_layer: facts.host_slots,
            card_experts_per_layer: facts.card_experts,
            heat_source: self.heat.name().to_string(),
        }
    }

    fn routing_heat(&self) -> Vec<Vec<u64>> {
        if let Some(h) = &self.profile_heat {
            return h.clone();
        }
        let routed = self.facts().routed;
        let heat = self.model.routing_heat();
        (self.from..self.to)
            .map(|l| match heat.get(l) {
                Some(order) if !order.is_empty() => super::routing::heat_from_order(order, routed),
                _ => vec![0; routed],
            })
            .collect()
    }

    fn script(&mut self, script: Script) -> Result<(), String> {
        self.model
            .bench_script(script.routes)
            .map_err(|e| format!("script: {e}"))
    }

    fn page_cache(&mut self, state: PageCache) -> Result<(), String> {
        let started = std::time::Instant::now();
        self.model.bench_page_cache(state == PageCache::Warm);
        tracing::info!(
            "page cache {} for the held layers' experts in {:.1}s",
            state.name(),
            started.elapsed().as_secs_f32()
        );
        Ok(())
    }

    fn crossings(&self) -> Arc<Crossings> {
        self.crossings.clone()
    }

    fn step(&mut self, _step: usize) -> Result<(), String> {
        let vocab = self.facts().vocab.max(1);
        let token = self.rng.below(vocab) as u32;
        self.history.push(token);
        let input = if self.from == 0 {
            StepIn::Token(token)
        } else {
            // A stream drawn in [-1, 1), as the first layer's output stands in for the layers
            // below the range.
            let values: Vec<f32> = (0..self.model.bench_stream_width())
                .map(|_| (self.rng.next_f64() * 2.0 - 1.0) as f32)
                .collect();
            StepIn::Stream {
                values,
                history: &self.history,
            }
        };
        self.model
            .bench_step(self.pos, input)
            .map_err(|e| format!("decode step: {e}"))?;
        self.pos += 1;
        Ok(())
    }

    fn tier_stats(&self) -> (u64, u64, u64) {
        self.model.host_tier_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::super::subject::Subject;
    use super::*;
    use crate::inference::generic_transformer::wide_sparse_toy::{
        real_entries_for, real_md_layout,
    };
    use crate::tensor::gguf_write::write_gguf_with_metadata;

    /// A synthetic model file in the given band layout, written where the test runs.
    fn synthetic_file(name: &str, ratios: &[u32], kv_src: &[u32]) -> std::path::PathBuf {
        let bands: Vec<bool> = (0..ratios.len())
            .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)))
            .collect();
        let md = real_md_layout(ratios, kv_src, kv_src);
        let entries = real_entries_for(&bands);
        let path = std::env::temp_dir().join(format!(
            "loken-layer-bench-{name}-{}.gguf",
            std::process::id()
        ));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        path
    }

    fn controls(file: std::path::PathBuf, layers: (usize, usize)) -> Controls {
        Controls {
            file,
            layers,
            tokens: 3,
            warm: 1,
            placement: Placement::Host,
            page_cache: PageCache::Warm,
            readahead: None,
            card_experts: None,
            guess_hit: 1.0,
            seed: 5,
            expert_profile: None,
            json: None,
        }
    }

    fn subject(file: &std::path::Path, layers: (usize, usize)) -> Result<WideSubject, String> {
        let src = SplitGguf::open(file).unwrap();
        WideSubject::open(src, &controls(file.to_path_buf(), layers), "test".into())
    }

    /// Two layers of a four-layer synthetic model are held alone, and the run reports the model
    /// as the generic route loaded it, the stage profile and the counters on the host.
    #[test]
    fn held_layers_replay_scripted_routes_on_the_host() {
        let file = synthetic_file("held", &[0, 0, 0, 0], &[]);
        let c = controls(file.clone(), (1, 3));
        let record = super::super::run(&c).unwrap();
        println!("{}", record.table());
        let _ = std::fs::remove_file(&file);
        assert_eq!(record.model.held, (1, 3));
        assert_eq!(record.model.layers, 4);
        assert!(record.model.routed_experts > record.model.activated);
        assert!(record.model.expert_bytes > 0);
        assert!(!record.model.cards_opened);
        assert_eq!(record.model.card_experts_per_layer, vec![0, 0]);
        assert_eq!(record.model.host_slots_per_layer, vec![0, 0]);
        assert_eq!(record.timing.ms_per_token.len(), 3);
        for stage in ["token mixer", "experts"] {
            assert!(record.stages.iter().any(|s| s.stage == stage), "{stage}");
        }
        assert!(record
            .counters
            .steps
            .iter()
            .all(|s| s.count.taken == 0 && s.count.bytes_out == 0));
        assert_eq!(record.controls.placement, "host");
        assert_eq!(
            record.model.heat_source,
            "a uniform draw (no prior, no profile)"
        );
        // The JSON parser may read the last bit of a float differently, so numbers compare to a
        // relative 1e-12 and everything else exactly.
        let back: serde_json::Value = serde_json::from_str(&record.to_json()).unwrap();
        assert!(same_json(&back, &serde_json::to_value(&record).unwrap()));
    }

    /// Two JSON values are the same when their numbers agree to a relative 1e-12 and the rest is equal.
    fn same_json(a: &serde_json::Value, b: &serde_json::Value) -> bool {
        use serde_json::Value::*;
        match (a, b) {
            (Number(x), Number(y)) => match (x.as_f64(), y.as_f64()) {
                (Some(x), Some(y)) => (x - y).abs() <= 1e-12 * x.abs().max(y.abs()),
                _ => x == y,
            },
            (Array(x), Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same_json(x, y))
            }
            (Object(x), Object(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w)))
            }
            _ => a == b,
        }
    }

    /// The range starting the stack runs from a token, the range ending it runs to the logits,
    /// and one inside takes a synthetic stream.
    #[test]
    fn every_kind_of_range_runs() {
        let file = synthetic_file("kinds", &[0, 0, 0, 0], &[]);
        for layers in [(0, 4), (0, 2), (2, 4), (1, 3)] {
            let mut s = subject(&file, layers).unwrap();
            let heat = s.routing_heat();
            assert_eq!(heat.len(), layers.1 - layers.0);
            let script = super::super::routing::script(&heat, s.facts().activated, 3, 1.0, 5);
            s.script(script).unwrap();
            for step in 0..3 {
                s.step(step).unwrap_or_else(|e| panic!("{layers:?}: {e}"));
            }
            assert_eq!(s.pos, 3);
        }
        let _ = std::fs::remove_file(&file);
    }

    /// A range starting at a band layer that reads another's keys is refused, and the refusal
    /// names where a range may start.
    #[test]
    fn a_range_cannot_start_inside_a_band() {
        let file = synthetic_file("band", &[0, 4, 4, 0], &[1]);
        let err = subject(&file, (2, 4)).err().unwrap();
        assert!(err.contains("[0, 1, 3]"), "{err}");
        assert!(subject(&file, (1, 4)).is_ok());
        assert!(subject(&file, (3, 5)).err().unwrap().contains("past the 4"));
        let _ = std::fs::remove_file(&file);
    }

    /// A profile of the model's shape gives the heat its counts; one of another shape is refused.
    #[test]
    fn a_profile_gives_the_heat_its_counts() {
        let file = synthetic_file("profile", &[0, 0, 0], &[]);
        let plain = subject(&file, (0, 2)).unwrap();
        let (layers, experts) = (plain.layers, plain.facts().routed);
        let counts: Vec<u32> = (0..layers * experts)
            .map(|i| (i % experts) as u32)
            .collect();
        let path = std::env::temp_dir().join(format!(
            "loken-layer-bench-profile-{}.counts",
            std::process::id()
        ));
        ExpertProfile {
            layers,
            experts,
            counts: counts.clone(),
        }
        .write(&path)
        .unwrap();
        let mut c = controls(file.clone(), (0, 2));
        c.expert_profile = Some(path.clone());
        let s = WideSubject::open(SplitGguf::open(&file).unwrap(), &c, "test".into()).unwrap();
        assert_eq!(s.describe().heat_source, "the expert profile");
        let want: Vec<u64> = counts[..experts].iter().map(|&c| u64::from(c)).collect();
        assert_eq!(s.routing_heat()[0], want);
        ExpertProfile {
            layers: layers + 1,
            experts,
            counts: vec![0; (layers + 1) * experts],
        }
        .write(&path)
        .unwrap();
        let err = WideSubject::open(SplitGguf::open(&file).unwrap(), &c, "test".into())
            .err()
            .unwrap();
        assert!(err.contains("the model has 3 of"), "{err}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&file);
    }

    /// The controls the generic route has no counterpart for are refused by name.
    #[test]
    fn controls_without_a_generic_counterpart_are_refused() {
        let file = synthetic_file("refused", &[0, 0], &[]);
        let mut cards = controls(file.clone(), (0, 2));
        cards.placement = Placement::Cards;
        let mut ahead = controls(file.clone(), (0, 2));
        ahead.readahead = Some(2);
        for (what, c) in [("placement cards", cards), ("readahead", ahead)] {
            let err = WideSubject::open(SplitGguf::open(&file).unwrap(), &c, "test".into())
                .err()
                .unwrap();
            assert!(err.starts_with(what), "{err}");
        }
        let _ = std::fs::remove_file(&file);
    }

    /// The page cache of the held layers' experts is asked warm and cold without a failure.
    #[test]
    fn the_page_cache_is_asked_warm_and_cold() {
        let file = synthetic_file("pages", &[0, 0], &[]);
        let mut s = subject(&file, (0, 2)).unwrap();
        s.page_cache(PageCache::Cold).unwrap();
        s.page_cache(PageCache::Warm).unwrap();
        let _ = std::fs::remove_file(&file);
    }

    /// Two subjects of the same file and seed draw the same routes.
    #[test]
    fn the_seed_fixes_the_routes() {
        let file = synthetic_file("seed", &[0, 0], &[]);
        let one = subject(&file, (0, 2)).unwrap();
        let two = subject(&file, (0, 2)).unwrap();
        let (h1, h2) = (one.routing_heat(), two.routing_heat());
        assert_eq!(h1, h2);
        let k = one.facts().activated;
        let a = super::super::routing::script(&h1, k, 4, 1.0, 5);
        let b = super::super::routing::script(&h2, k, 4, 1.0, 5);
        assert_eq!(a.routes, b.routes);
        let _ = std::fs::remove_file(&file);
    }

    /// The released file in the store, 2 layers from the start on the host: the record is read by
    /// hand. Ignored by default: it maps a file of hundreds of GB.
    /// Run with
    ///   CUDA_VISIBLE_DEVICES= cargo test --release --no-default-features --features cpu,microbench --lib -- --ignored the_released_file --nocapture
    #[test]
    #[ignore]
    fn the_released_file_runs_two_layers_on_the_host() {
        use crate::inference::generic_transformer::wide_sparse_real_tests::{blob, TAG};
        let mut c = controls(blob(TAG), (0, 2));
        c.tokens = 4;
        let record = super::super::run(&c).unwrap();
        println!("{}", record.table());
        println!("{}", record.to_json());
    }
}
