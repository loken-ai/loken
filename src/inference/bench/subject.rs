//! A model under the bench: what it holds, how it is placed, and one decode step at a time. The
//! bench drives any model through this; the file's metadata decides which implementation opens.

use super::counters::Crossings;
use super::report::ModelRecord;
use super::routing::Script;
use super::{Controls, PageCache};
use std::sync::Arc;

pub trait Subject {
    /// What is held and where its experts went, as it stands now.
    fn describe(&self) -> ModelRecord;

    /// Per held layer, the heat of every routed expert: the profile's counts, the prior's order
    /// as a heat, or all zeros for a uniform draw.
    fn routing_heat(&self) -> Vec<Vec<u64>>;

    /// Give every held layer the routes and guesses of the steps to come, in layer order.
    fn script(&mut self, script: Script) -> Result<(), String>;

    /// Put the held layers' routed experts in the asked state of the page cache.
    fn page_cache(&mut self, state: PageCache) -> Result<(), String>;

    /// The counts of the steps at the card seam, kept across the run.
    fn crossings(&self) -> Arc<Crossings>;

    /// Run decode step `step` through the held layers.
    fn step(&mut self, step: usize) -> Result<(), String>;

    /// The expert tiers' counters summed over the held layers: hits, misses, evictions.
    fn tier_stats(&self) -> (u64, u64, u64);
}

/// The subject the file at `controls.file` names, opened and placed under `controls`. A file of
/// an architecture the bench has no subject for is refused by name.
pub fn open(controls: &Controls) -> Result<Box<dyn Subject>, String> {
    use crate::tensor::quantized::gguf_source::SplitGguf;
    let path = controls.file.display();
    let src = SplitGguf::open(&controls.file).map_err(|e| format!("open {path}: {e}"))?;
    let content = src.merged_content();
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    if crate::inference::model::deepseek_v41::claims_gguf(&content) {
        return Ok(Box::new(super::deepseek::DeepseekSubject::open(
            src, controls, arch,
        )?));
    }
    Err(format!(
        "{path}: the layer bench has no subject for the architecture `{arch}`"
    ))
}
