//! The streamed MoE as a bench subject: a few of its layers held from the file, placed as the
//! engine places the whole, and decoded from a synthetic boundary with scripted routes.

use super::counters::{Counting, Crossings};
use super::report::ModelRecord;
use super::routing::Script;
use super::{Controls, PageCache, Placement};
use crate::inference::model::deepseek_v41::cache::DecodeState;
use crate::inference::model::deepseek_v41::model::{
    DecodeBoundary, DecodeRangeIn, DeepseekV41Model,
};
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::inference::offload::streamed::{Demand, Streamed};
use crate::inference::offload::Offload;
use crate::tensor::quantized::gguf_source::SplitGguf;
use crate::tensor::{Device, Tensor};
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

pub struct DeepseekSubject {
    model: DeepseekV41Model,
    cfg: DeepseekV41Config,
    state: DecodeState,
    architecture: String,
    from: usize,
    to: usize,
    /// The card placement, kept for the run; `None` on the host alone.
    streamed: Option<Streamed>,
    /// The offload every step goes through: the counting seam over the cards, or over nothing.
    offload: Arc<Counting>,
    /// The profile's counts per held layer, when one was read.
    profile_heat: Option<Vec<Vec<u64>>>,
    heat: HeatSource,
    rng: super::routing::Lcg,
}

impl DeepseekSubject {
    /// The layers `[from, to)` of `src` loaded and placed under `controls`.
    pub fn open(src: SplitGguf, controls: &Controls, architecture: String) -> Result<Self, String> {
        let (from, to) = controls.layers;
        let mut cfg = DeepseekV41Config::from_meta(&src).map_err(|e| format!("config: {e}"))?;
        if to > cfg.n_layers {
            return Err(format!(
                "layers [{from}, {to}) reach past the {} of the stack",
                cfg.n_layers
            ));
        }
        // A band layer reads the compressed keys a source layer below it publishes within the
        // token: a range can only start where that state is empty or made.
        let starts_clean = from == 0
            || cfg.compress_ratios.get(from).copied().unwrap_or(0) == 0
            || cfg.kv_source_layers.contains(&from);
        if !starts_clean {
            let starts: Vec<usize> = (0..cfg.n_layers)
                .filter(|&l| {
                    cfg.compress_ratios.get(l).copied().unwrap_or(0) == 0
                        || cfg.kv_source_layers.contains(&l)
                })
                .collect();
            return Err(format!(
                "layer {from} reads the compressed keys of a layer below it; a range starts at one of {starts:?}"
            ));
        }
        if cfg.hot_experts.iter().all(Vec::is_empty) {
            if let Some(prior) = crate::inference::model::deepseek_v41::sibling::prior_beside(
                &controls.file,
                cfg.n_layers,
                cfg.n_routed_experts,
            ) {
                cfg.hot_experts = prior;
            }
        }
        let started = std::time::Instant::now();
        let mut model = DeepseekV41Model::load_layers(&src, &cfg, from, to)
            .map_err(|e| format!("load layers [{from}, {to}): {e}"))?;
        tracing::info!(
            "layers [{from}, {to}) of {} held in {:.1}s",
            cfg.n_layers,
            started.elapsed().as_secs_f32()
        );

        // The routing heat: the profile's counts when one is given, else the file's prior.
        let mut profile_heat = None;
        let mut heat = if cfg.hot_experts.iter().any(|l| !l.is_empty()) {
            HeatSource::Prior
        } else {
            HeatSource::Uniform
        };
        let ranks = match &controls.expert_profile {
            Some(path) => {
                use crate::inference::generic_transformer::expert_profile::ExpertProfile;
                let profile = ExpertProfile::read(path)
                    .map_err(|e| format!("expert profile {}: {e}", path.display()))?;
                if profile.layers != cfg.n_layers || profile.experts != cfg.n_routed_experts {
                    return Err(format!(
                        "expert profile {}: {} layers of {} experts, the model has {} of {}",
                        path.display(),
                        profile.layers,
                        profile.experts,
                        cfg.n_layers,
                        cfg.n_routed_experts
                    ));
                }
                let e = profile.experts;
                profile_heat = Some(
                    (from..to)
                        .map(|l| {
                            profile.counts[l * e..(l + 1) * e]
                                .iter()
                                .map(|&c| u64::from(c))
                                .collect()
                        })
                        .collect(),
                );
                heat = HeatSource::Profile;
                crate::inference::place::expert_variants::ranks_from_counts(
                    &profile.counts,
                    profile.layers,
                    profile.experts,
                    &cfg.hot_experts,
                )
            }
            None => None,
        };
        model.set_variant_policy(Default::default(), ranks);

        // The host tier as the engine sizes it: the memory left, divided over every layer of the
        // stack, so a held layer keeps what it would keep in service.
        let slots = model
            .size_expert_cache_to_memory(&cfg)
            .map_err(|e| format!("host tier: {e}"))?;
        tracing::info!("host tier: {slots} experts per layer");
        if controls.placement == Placement::Cards {
            model.set_expert_cache(0);
        }

        let streamed = match controls.placement {
            Placement::Host => None,
            Placement::Auto | Placement::Cards => Self::place(&model, &cfg, controls)?,
        };
        if let Some(s) = &streamed {
            model.offload_experts(Some(s.lanes.clone()));
        }
        model.pin_readahead(controls.readahead);
        let offload = Arc::new(Counting::new(streamed.as_ref().map(|s| s.offload.clone())));
        let state = model.new_decode_state();
        Ok(Self {
            model,
            cfg,
            state,
            architecture,
            from,
            to,
            streamed,
            offload,
            profile_heat,
            heat,
            rng: super::routing::Lcg::new(controls.seed),
        })
    }

    /// The cards opened for the held layers with the room a served model's layers get: the
    /// weights every token reads are set aside as if the whole stack were held, and the experts'
    /// room is the held layers' share of what the stack's experts get, or `card_experts` per
    /// layer when asked. `None` when no card takes the placement.
    fn place(
        model: &DeepseekV41Model,
        cfg: &DeepseekV41Config,
        controls: &Controls,
    ) -> Result<Option<Streamed>, String> {
        use crate::inference::engine::model_backend::PREFILL_TRANSIENT_TOKENS;
        let (from, to) = controls.layers;
        let held = to - from;
        let n = cfg.n_layers;
        let transient = model.transient_bytes(PREFILL_TRANSIENT_TOKENS);
        let ceiling: usize = crate::inference::place::room::card_rooms()
            .iter()
            .map(|c| (c.total as usize).saturating_sub(transient))
            .sum();
        let expert_bytes = model.expert_bytes().map_err(|e| e.to_string())?;
        let layer_part = model.resident_layer_bytes();
        let resident_held = model.resident_path_bytes();
        let resident_full = resident_held - layer_part + layer_part / held * n;
        let served_per_layer = if expert_bytes == 0 {
            0
        } else {
            ceiling.saturating_sub(resident_full) / n / expert_bytes
        };
        let per_layer = controls.card_experts.unwrap_or(served_per_layer);
        let expert_room = per_layer * held * expert_bytes;
        let always_read = ceiling.saturating_sub(expert_room).max(resident_held);
        tracing::info!(
            "cards: {} MB ceiling, {} MB set aside, room for {per_layer} experts per held layer ({served_per_layer} in service)",
            ceiling / 1_000_000,
            always_read / 1_000_000
        );
        use crate::inference::offload::streamed::StreamedModel;
        let prior: Vec<Vec<usize>> = model
            .routing_prior()
            .iter()
            .enumerate()
            .map(|(l, ranked)| {
                if (from..to).contains(&l) {
                    ranked.iter().copied().take(per_layer).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let fetch = |layer: usize, id: usize| model.expert(layer, id);
        let choose = |card_bytes: u64| {
            let _ = DeepseekV41Model::place_variants(model, card_bytes);
        };
        let placed = Streamed::open(&Demand {
            always_read,
            transient,
            concurrency: model.active_experts(),
            layers: n,
            prior: &prior,
            fetch: &fetch,
            choose: &choose,
        });
        if placed.is_none() {
            tracing::warn!("no card took the placement: the host runs every step");
        }
        Ok(placed)
    }

    /// The boundary a step starts from when the range starts inside the stack: a stream of
    /// values drawn in `[-1, 1)`, collapsed as the first layer collapses the embedding, no band
    /// state, and the engram's recent tokens pushed with a token drawn from the vocabulary.
    fn synthetic_boundary(&mut self) -> Result<DecodeBoundary, String> {
        let (dim, hc) = (self.cfg.d_model, self.cfg.hc_mult);
        let values: Vec<f32> = (0..hc * dim)
            .map(|_| (self.rng.next_f64() * 2.0 - 1.0) as f32)
            .collect();
        let x =
            Tensor::from_vec(values, (1, 1, hc, dim), &Device::Cpu).map_err(|e| e.to_string())?;
        let mut pre = vec![0f32; hc];
        pre[0] = 1.0;
        let token = self.rng.below(self.cfg.vocab_size.max(1)) as u32;
        let recent = self
            .cfg
            .engram
            .as_ref()
            .map(|ecfg| self.state.engram.push(ecfg, token));
        Ok(DecodeBoundary {
            x,
            pre_mix: vec![pre],
            shared: Default::default(),
            recent,
        })
    }
}

impl super::subject::Subject for DeepseekSubject {
    fn describe(&self) -> ModelRecord {
        let tiers = self.model.expert_tiers();
        ModelRecord {
            architecture: self.architecture.clone(),
            layers: self.cfg.n_layers,
            held: (self.from, self.to),
            routed_experts: self.cfg.n_routed_experts,
            activated: self.cfg.n_activated_experts,
            expert_bytes: self.model.expert_bytes().unwrap_or(0) as u64,
            cards_opened: self.streamed.is_some(),
            card_experts_per_layer: tiers.iter().map(|t| t.0).collect(),
            host_slots_per_layer: tiers.iter().map(|t| t.1).collect(),
            heat_source: self.heat.name().to_string(),
        }
    }

    fn routing_heat(&self) -> Vec<Vec<u64>> {
        if let Some(h) = &self.profile_heat {
            return h.clone();
        }
        use crate::inference::offload::streamed::StreamedModel;
        let prior = self.model.routing_prior();
        (self.from..self.to)
            .map(|l| match prior.get(l) {
                Some(order) if !order.is_empty() => {
                    super::routing::heat_from_order(order, self.cfg.n_routed_experts)
                }
                _ => vec![0; self.cfg.n_routed_experts],
            })
            .collect()
    }

    fn script(&mut self, script: Script) -> Result<(), String> {
        for (i, layer) in (self.from..self.to).enumerate() {
            let routes = script.routes.get(i).cloned().unwrap_or_default();
            let guesses = script.guesses.get(i).cloned().unwrap_or_default();
            self.model
                .script_routes(layer, routes, guesses)
                .map_err(|e| format!("layer {layer}: {e}"))?;
        }
        Ok(())
    }

    fn page_cache(&mut self, state: PageCache) -> Result<(), String> {
        let started = std::time::Instant::now();
        for layer in self.from..self.to {
            let Some(store) = self.model.expert_store(layer) else {
                continue;
            };
            for id in 0..store.count() {
                let expert = store
                    .view(id)
                    .map_err(|e| format!("layer {layer} expert {id}: {e}"))?;
                match state {
                    PageCache::Cold => expert.dont_need(),
                    PageCache::Warm => expert.populate(),
                }
            }
        }
        tracing::info!(
            "page cache {} for the held layers' experts in {:.1}s",
            state.name(),
            started.elapsed().as_secs_f32()
        );
        Ok(())
    }

    fn crossings(&self) -> Arc<Crossings> {
        self.offload.crossings.clone()
    }

    fn step(&mut self, _step: usize) -> Result<(), String> {
        let input = if self.from == 0 {
            DecodeRangeIn::Token(self.rng.below(self.cfg.vocab_size.max(1)) as u32)
        } else {
            DecodeRangeIn::Resume(self.synthetic_boundary()?)
        };
        let (from, to) = (self.from, self.to);
        let model = &self.model;
        let state = &mut self.state;
        let offload: Arc<dyn Offload> = self.offload.clone();
        crate::inference::offload::with_offload(offload, || {
            model.forward_decode_range(input, from, to, state)
        })
        .map_err(|e| format!("decode step: {e}"))?;
        // A range to the end of the stack advances the position itself.
        if to < self.cfg.n_layers {
            self.state.pos += 1;
        }
        Ok(())
    }

    fn tier_stats(&self) -> (u64, u64, u64) {
        self.model
            .expert_store_stats()
            .iter()
            .fold((0, 0, 0), |(h, m, e), s| {
                (h + s.hits, m + s.misses, e + s.evictions)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::super::subject::Subject;
    use super::*;
    use crate::inference::model::deepseek_v41::model::tests::{real_entries_for, real_md_layout};
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
            readahead: Some(1),
            card_experts: None,
            guess_hit: 1.0,
            seed: 5,
            expert_profile: None,
            json: None,
        }
    }

    /// Two layers of a four-layer synthetic model are held alone, the scripted routes are the
    /// ones the layers take, and the run reports stages and counters on the host.
    #[test]
    fn held_layers_replay_scripted_routes_on_the_host() {
        let file = synthetic_file("held", &[0, 0, 0, 0], &[]);
        let c = controls(file.clone(), (1, 3));
        let record = super::super::run(&c).unwrap();
        println!("{}", record.table());
        let _ = std::fs::remove_file(&file);
        assert_eq!(record.model.held, (1, 3));
        assert_eq!(record.model.layers, 4);
        assert!(!record.model.cards_opened);
        assert_eq!(record.timing.ms_per_token.len(), 3);
        assert!(record.stages.iter().any(|s| s.stage == "attention"));
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
    }

    /// The routes the script gives a layer are the ones its gate reports having taken.
    #[test]
    fn the_scripted_routes_are_the_routes_taken() {
        let file = synthetic_file("script", &[0, 0, 0], &[]);
        let c = controls(file.clone(), (0, 2));
        let src = SplitGguf::open(&file).unwrap();
        let mut s = DeepseekSubject::open(src, &c, "test".into()).unwrap();
        let _ = std::fs::remove_file(&file);
        let heat = s.routing_heat();
        assert_eq!(heat.len(), 2);
        let script = super::super::routing::script(&heat, s.cfg.n_activated_experts, 2, 1.0, 5);
        let want = script.routes.clone();
        s.script(script).unwrap();
        s.step(0).unwrap();
        let routed = s.model.last_routing();
        for (l, layer) in routed.iter().enumerate() {
            let got: Vec<usize> = layer[0].iter().map(|&(e, _)| e).collect();
            let mut expect = want[l][0].clone();
            let mut got_sorted = got.clone();
            expect.sort_unstable();
            got_sorted.sort_unstable();
            assert_eq!(got_sorted, expect, "layer {l}");
        }
        s.step(1).unwrap();
        assert_eq!(s.state.pos, 2);
    }

    /// A range starting at a band layer that reads another's keys is refused, and the refusal
    /// names where a range may start.
    #[test]
    fn a_range_cannot_start_inside_a_band() {
        let file = synthetic_file("band", &[0, 4, 4, 0], &[1]);
        let c = controls(file.clone(), (2, 4));
        let src = SplitGguf::open(&file).unwrap();
        let err = DeepseekSubject::open(src, &c, "test".into()).err().unwrap();
        assert!(err.contains("[0, 1, 3]"), "{err}");
        let c = controls(file.clone(), (1, 4));
        let src = SplitGguf::open(&file).unwrap();
        assert!(DeepseekSubject::open(src, &c, "test".into()).is_ok());
        let c = controls(file.clone(), (3, 5));
        let src = SplitGguf::open(&file).unwrap();
        assert!(DeepseekSubject::open(src, &c, "test".into())
            .err()
            .unwrap()
            .contains("past the 4"));
        let _ = std::fs::remove_file(&file);
    }
}
