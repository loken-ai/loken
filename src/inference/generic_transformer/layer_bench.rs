//! What the layer bench reads and drives on a loaded Sinkhorn wide-residual model.
//!
//! The bench holds a range of the stack, replays decode steps through it from a synthetic
//! boundary, and imposes the routes of its own script on the routers of the held layers.
//! Everything else the bench reports is read from the loaded model.

use super::compressed_band::BandLeft;
use super::wide_pipeline::WideBoundary;
use super::wide_sparse_layer::{onto_first, RoutedFfn, TokenLog};
use super::*;

/// What a loaded wide-residual model holds, as the bench reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct WideFacts {
    /// Layers of the stack.
    pub layers: usize,
    /// The layers held, `[from, to)`.
    pub held: (usize, usize),
    pub routed: usize,
    pub activated: usize,
    pub vocab: usize,
    /// Bytes of 1 routed expert as the layer reads it.
    pub expert_bytes: u64,
    /// Per held layer, the routed experts the cards keep.
    pub card_experts: Vec<usize>,
    /// Per held layer, the routed experts the host tier keeps at most, 0 without a tier.
    pub host_slots: Vec<usize>,
}

/// What 1 decode step starts from.
pub enum StepIn<'a> {
    /// A token, for a range that starts the stack.
    Token(u32),
    /// A residual of `bench_stream_width` values and the raw ids of the positions up to this
    /// one, for a range that starts inside the stack.
    Stream {
        values: Vec<f32>,
        history: &'a [u32],
    },
}

/// The advice that asks the kernel to read a range in as a read would.
const POPULATE_READ: libc::c_int = 22;

impl RoutedFfn {
    /// Queues `routes` for the single-row forwards to come, 1 per forward.
    ///
    /// A route names `used` distinct experts of the layer.
    /// A route that does not is refused, and the queue stays as it was.
    pub fn script_routes(&self, routes: Vec<Vec<usize>>) -> Result<()> {
        for route in &routes {
            let distinct: std::collections::HashSet<usize> = route.iter().copied().collect();
            if route.len() != self.used
                || distinct.len() != route.len()
                || route.iter().any(|&e| e >= self.experts.len())
            {
                crate::tensor::bail!(
                    "a scripted route names {} experts, {} distinct, of {}: {} are routed to at once",
                    route.len(),
                    distinct.len(),
                    self.experts.len(),
                    self.used
                );
            }
        }
        let mut queue = self.lossy.script.lock().unwrap_or_else(|e| e.into_inner());
        self.lossy
            .scripted
            .fetch_add(routes.len(), std::sync::atomic::Ordering::Relaxed);
        queue.extend(routes);
        Ok(())
    }
}

/// Reads the storage of `p` into memory now.
fn populate(p: &crate::tensor::blockscaled::projection::Projection) {
    match p.storage_span() {
        Some((start, len)) => crate::tensor::mapped::advise_span(start, len, POPULATE_READ, false),
        None => p.will_need(),
    }
}

impl GenericHeteroTransformer {
    /// The routed feed-forwards of the layers held, in layer order.
    fn held_ffns(&self) -> Vec<&RoutedFfn> {
        self.layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .map(|b| &b.ffn)
            .collect()
    }

    /// What the model holds, `None` for a model that is not a Sinkhorn wide residual.
    pub fn bench_facts(&self) -> Option<WideFacts> {
        let ws = self.wide_sparse.as_ref()?;
        let (from, to, layers) = self.segment_range();
        let ffns = self.held_ffns();
        Some(WideFacts {
            layers,
            held: (from, to),
            routed: ws.meta.mixture.routed,
            activated: ws.meta.mixture.used,
            vocab: ws.meta.vocab,
            expert_bytes: ffns
                .first()
                .and_then(|f| f.variant_bytes(0, 0))
                .unwrap_or(0),
            card_experts: ffns.iter().map(|f| f.card.count()).collect(),
            host_slots: ffns
                .iter()
                .map(|f| f.host.as_ref().map_or(0, |k| k.tier().capacity()))
                .collect(),
        })
    }

    /// The host tier's hits, misses and experts let go since the load, over the held layers.
    pub fn host_tier_stats(&self) -> (u64, u64, u64) {
        self.held_ffns()
            .iter()
            .filter_map(|f| f.host.as_ref().map(|k| k.stats()))
            .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
    }

    /// Values of the residual a step that starts inside the stack is given.
    pub fn bench_stream_width(&self) -> usize {
        self.stream_width()
    }

    /// Queues the routes of each held layer, in layer order: per layer, 1 route per step.
    pub fn bench_script(&self, routes: Vec<Vec<Vec<usize>>>) -> Result<()> {
        let ffns = self.held_ffns();
        if routes.len() != ffns.len() {
            crate::tensor::bail!(
                "routes for {} layers, the model holds {}",
                routes.len(),
                ffns.len()
            );
        }
        for (ffn, layer) in ffns.into_iter().zip(routes) {
            ffn.script_routes(layer)?;
        }
        Ok(())
    }

    /// Reads the routed experts of the held layers into memory (`warm`) or drops them from it.
    pub fn bench_page_cache(&self, warm: bool) {
        for ffn in self.held_ffns() {
            for e in &ffn.experts {
                for p in [&e.gate, &e.up, &e.down] {
                    match warm {
                        true => populate(p),
                        false => p.dont_need(),
                    }
                }
            }
        }
    }

    /// The boundary a step that starts inside the stack is given: `values` as the residual of the
    /// row at `pos`, collapsed onto stream 0, with no band state and the compressed ids the
    /// engram rows of that position read.
    fn bench_boundary(
        &self,
        pos: usize,
        values: Vec<f32>,
        history: &[u32],
    ) -> Result<WideBoundary> {
        let ws = self
            .wide_sparse
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("a bench step needs a wide residual"))?;
        let width = self.stream_width();
        if values.len() != width || history.len() <= pos {
            crate::tensor::bail!(
                "a boundary of {} values and {} ids for a row of {width} values at position {pos}",
                values.len(),
                history.len()
            );
        }
        let recent = ws.engram.as_ref().map(|meta| {
            let start = TokenLog::window_start(meta, pos);
            history[start..=pos]
                .iter()
                .map(|&t| meta.compress(t))
                .collect()
        });
        Ok(WideBoundary {
            first: pos,
            streams: ws.streams,
            hidden: width / ws.streams.max(1),
            residual: values,
            pre: vec![onto_first(ws.streams)],
            recent,
            widths: (0, 0),
            band: vec![BandLeft::default()],
        })
    }

    /// Runs the decode step at position `pos` through the layers held.
    ///
    /// A range that ends the stack runs to the logits.
    pub fn bench_step(&mut self, pos: usize, input: StepIn<'_>) -> Result<()> {
        let (from, to, layers) = self.segment_range();
        match (input, from == 0, to == layers) {
            (StepIn::Token(t), true, true) => {
                self.forward(&Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?, pos)?;
            }
            (StepIn::Token(t), true, false) => {
                let ids = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
                self.wide_head_forward(&ids, pos, false)?;
            }
            (StepIn::Stream { values, history }, false, to_end) => {
                let b = self.bench_boundary(pos, values, history)?;
                match to_end {
                    true => self.wide_tail_forward(&b, false).map(drop)?,
                    false => self.wide_interior_forward(&b, false).map(drop)?,
                }
            }
            _ => crate::tensor::bail!("a range from layer {from} takes the other kind of input"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::moe_weights::sqrt_softplus_weigh;
    use super::super::wide_sparse_oracle_tests::generic;
    use super::*;

    /// A route that is queued is the choice of the next single row, weighed by the gate's row,
    /// and the row after it is the gate's own choice again.
    #[test]
    fn a_queued_route_replaces_one_single_row_choice() {
        let model = generic(&[0, 2, 0], &[1], &[1]);
        let ffn = model.held_ffns()[0];
        let n = ffn.experts.len();
        let row: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let own = ffn.route(std::slice::from_ref(&row));
        let route: Vec<usize> = (0..n).rev().take(ffn.used).collect();
        assert_ne!(own[0].0, route);
        ffn.script_routes(vec![route.clone()]).unwrap();
        // A batch of rows is never scripted and leaves the queue as it is.
        let batch = ffn.route(&[row.clone(), row.clone()]);
        assert_eq!(batch[0], own[0]);
        let scripted = ffn.route(std::slice::from_ref(&row));
        let want = sqrt_softplus_weigh(&row, route, ffn.temperature, ffn.normalize);
        assert_eq!(scripted[0], want);
        assert_eq!(ffn.route(std::slice::from_ref(&row)), own);
    }

    /// A route that does not name `used` distinct experts of the layer is refused whole.
    #[test]
    fn a_route_off_the_layer_is_refused() {
        let model = generic(&[0, 2, 0], &[1], &[1]);
        let ffn = model.held_ffns()[0];
        let n = ffn.experts.len();
        let ok: Vec<usize> = (0..ffn.used).collect();
        let short = ok[..ffn.used - 1].to_vec();
        let repeated = vec![0; ffn.used];
        let beyond: Vec<usize> = (n..n + ffn.used).collect();
        for bad in [short, repeated, beyond] {
            assert!(ffn.script_routes(vec![ok.clone(), bad]).is_err());
        }
        assert_eq!(
            ffn.lossy
                .scripted
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    /// The facts are those of the toy as loaded: its stack, its experts, nothing on a card.
    #[test]
    fn the_facts_are_those_of_the_loaded_toy() {
        let model = generic(&[0, 2, 0], &[1], &[1]);
        let f = model.bench_facts().unwrap();
        assert_eq!((f.layers, f.held), (3, (0, 3)));
        assert!(f.routed > f.activated && f.expert_bytes > 0 && f.vocab > 0);
        assert_eq!(f.card_experts, vec![0, 0, 0]);
    }

    /// A share of a stack with engram layers runs steps from a synthetic stream: the boundary
    /// carries the compressed ids the engram layer inside the share reads.
    #[test]
    fn a_share_with_an_engram_layer_runs_from_a_synthetic_stream() {
        use super::super::wide_sparse_engram_tests::engram_file;
        use crate::inference::place::layer_executor::HeteroPlan;
        use crate::tensor::quantized::gguf_file::MappedGguf;
        for share in [(0, 2), (1, 3), (2, 3)] {
            let MappedGguf { content, mmap, .. } = engram_file(&[0, 2, 0], &[1], &[1]);
            let plan = HeteroPlan::calculate_with_kv(share.1 - share.0, 1 << 20, &[], &[], 1.0, 0)
                .shifted(share.0, 3);
            let mut model =
                GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan)
                    .unwrap();
            assert_eq!(model.segment_range(), (share.0, share.1, 3));
            let facts = model.bench_facts().unwrap();
            let width = model.bench_stream_width();
            let mut history = Vec::new();
            for pos in 0..5 {
                let token = (pos as u32 * 7 + 3) % facts.vocab as u32;
                history.push(token);
                let input = match share.0 {
                    0 => StepIn::Token(token),
                    _ => StepIn::Stream {
                        values: (0..width).map(|i| ((i + pos) as f32 * 0.1).sin()).collect(),
                        history: &history,
                    },
                };
                model
                    .bench_step(pos, input)
                    .unwrap_or_else(|e| panic!("{share:?} at {pos}: {e}"));
            }
        }
    }
}
