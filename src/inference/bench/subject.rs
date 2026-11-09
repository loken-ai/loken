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

/// The implementation a file is read by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// A Sinkhorn wide-residual mixture, loaded through the generic route.
    GenericWide,
}

/// The arm a file with this `content` takes, `None` when the bench has no subject for it.
pub fn arm_of(content: &crate::tensor::quantized::gguf_file::Content) -> Option<Arm> {
    use crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta;
    WideSparseMeta::claims(content).then_some(Arm::GenericWide)
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
    match arm_of(&content) {
        Some(Arm::GenericWide) => Ok(Box::new(super::deepseek::WideSubject::open(
            src, controls, arch,
        )?)),
        None => Err(format!(
            "{path}: the layer bench has no subject for the architecture `{arch}`"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::wide_sparse_toy::{
        real_entries_for, real_md_layout,
    };
    use crate::tensor::gguf_write::write_gguf_with_metadata;
    use crate::tensor::quantized::gguf_file::{open_mapped, Value};

    fn controls(file: std::path::PathBuf) -> Controls {
        Controls {
            file,
            layers: (0, 2),
            tokens: 1,
            warm: 0,
            placement: super::super::Placement::Host,
            page_cache: PageCache::Warm,
            readahead: None,
            card_experts: None,
            guess_hit: 1.0,
            seed: 1,
            expert_profile: None,
            json: None,
        }
    }

    fn path_of(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "loken-bench-subject-{name}-{}.gguf",
            std::process::id()
        ))
    }

    /// A wide-residual file takes the generic arm, and the arm opens it into a subject.
    #[test]
    fn a_wide_residual_file_takes_the_generic_arm() {
        let path = path_of("wide");
        let md = real_md_layout(&[0, 0], &[], &[]);
        write_gguf_with_metadata(&path, &md, &real_entries_for(&[false, false])).unwrap();
        let g = open_mapped(&path).unwrap();
        assert_eq!(arm_of(&g.content), Some(Arm::GenericWide));
        let subject = open(&controls(path.clone()));
        let _ = std::fs::remove_file(&path);
        let record = subject.expect("the generic arm opens the file").describe();
        assert_eq!(record.held, (0, 2));
        assert_eq!(record.layers, 2);
    }

    /// A file of another architecture takes no arm and is refused by the architecture's name.
    #[test]
    fn another_architecture_is_refused_by_name() {
        let path = path_of("other");
        let md = vec![(
            "general.architecture".to_string(),
            Value::String("other".into()),
        )];
        write_gguf_with_metadata(&path, &md, &[]).unwrap();
        let err = open(&controls(path.clone())).err().expect("no arm");
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("`other`"), "{err}");
    }
}
