//! The routing heat of a Sinkhorn wide-residual model: per layer, its routed experts from the
//! most to the least routed to.
//!
//! The heat is the ranking of a routing profile that counted a route.
//! The prior orders the experts the profile never counted.
//! Without such a profile, the heat is the prior.
//!
//! The prior is the one the model file carries.
//! A model file that carries none takes the one a companion file beside it declares.

use super::expert_profile::ExpertProfile;
use super::wide_sparse_sibling as sibling;
use super::GenericHeteroTransformer;
use crate::inference::place::expert_variants::ranks_from_counts;
use crate::tensor::Result;

impl GenericHeteroTransformer {
    /// The prior of a Sinkhorn wide-residual model, per layer of the stack.
    ///
    /// Empty for a model of another kind.
    pub fn routing_prior(&self) -> &[Vec<usize>] {
        self.wide_sparse
            .as_ref()
            .map_or(&[], |ws| ws.prior.as_slice())
    }

    /// The heat of a Sinkhorn wide-residual model, per layer of the stack: the ranking a
    /// profile gave, else the prior.
    ///
    /// Empty for a model of another kind.
    pub fn routing_heat(&self) -> &[Vec<usize>] {
        match self.wide_sparse.as_ref() {
            Some(ws) => ws.heat.as_deref().unwrap_or(&ws.prior),
            None => &[],
        }
    }

    /// Takes the prior a companion file beside `model` declares, when the model file carries
    /// none.
    ///
    /// A companion is found by its metadata keys, whatever its name.
    /// A companion whose prior does not fit the model's layers and experts is passed over.
    /// Returns whether a companion gave the prior.
    pub fn locate_routing_prior(&mut self, model: &std::path::Path) -> bool {
        let Some(ws) = self.wide_sparse.as_mut() else {
            return false;
        };
        if !ws.prior.iter().all(Vec::is_empty) {
            return false;
        }
        match sibling::prior_beside(model, ws.meta.n_layers, ws.meta.mixture.routed) {
            Some(prior) => {
                ws.prior = prior;
                true
            }
            None => false,
        }
    }

    /// Ranks each layer's experts by the counts of `profile`, the prior ordering the experts it
    /// never counted.
    ///
    /// A profile of another shape than the model's layers and routed experts is refused, naming
    /// both shapes.
    /// A profile that counted no route leaves the heat the prior.
    /// A model of another kind is left as it is.
    pub fn prime_routing_heat(&mut self, profile: &ExpertProfile) -> Result<()> {
        let Some(ws) = self.wide_sparse.as_mut() else {
            return Ok(());
        };
        let (layers, experts) = (ws.meta.n_layers, ws.meta.mixture.routed);
        if profile.layers != layers || profile.experts != experts {
            crate::tensor::bail!(
                "the profile counts {} layers of {} experts, the model has {layers} layers of \
                 {experts} experts",
                profile.layers,
                profile.experts,
            );
        }
        ws.heat = ranks_from_counts(&profile.counts, layers, experts, &ws.prior);
        let used = ws.meta.mixture.used;
        let shares: Vec<Option<Vec<f64>>> = (0..layers)
            .map(|l| super::wide_always_read::count_shares(profile.layer(l), used))
            .collect();
        ws.shares = shares.iter().any(Option::is_some).then_some(shares);
        Ok(())
    }
}
