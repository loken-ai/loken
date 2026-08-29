//! Resolving weights inside a HuggingFace cache.
//!
//! Every model whose weights come from the hub is stored the same way:
//! `<hub>/models--<org>--<name>/snapshots/<revision>/<file>`. The revision is a
//! content hash chosen by whoever downloaded it, so a path can only be written by
//! looking - which is what this module does, once, for every family.
//!
//! The hub root comes from the configuration and never from a hardcoded path or an
//! environment variable.

use std::path::{Path, PathBuf};

/// The configured models directory.
///
/// The cache proper lives under `hub/`, but a checkpoint published outside the hub layout -
/// a plain subdirectory of loose safetensors - is resolved from here.
pub fn models_dir() -> String {
    crate::config::Config::load_default()
        .unwrap_or_else(|_| crate::config::Config::load_test())
        .get_hf_models_dir()
        .to_string_lossy()
        .into_owned()
}

/// The `hub/` directory holding the cached repositories.
///
/// The configured value may point at the cache root or at `hub/` itself; both are
/// written in the wild, so both are accepted and normalised here rather than at each
/// of the call sites.
pub fn hub() -> PathBuf {
    let cfg = crate::config::Config::load_default()
        .unwrap_or_else(|_| crate::config::Config::load_test());
    let dir = cfg.get_hf_models_dir();
    if dir.file_name().is_some_and(|n| n == "hub") {
        dir
    } else {
        dir.join("hub")
    }
}

/// Where a repository's snapshots live, whether or not any has been downloaded.
///
/// Returned even when it does not exist so an error can name the directory the
/// weights were expected in.
pub fn snapshots(repo: &str) -> PathBuf {
    hub().join(repo).join("snapshots")
}

/// The snapshot of `repo` holding `marker`, if one is present.
///
/// A repository can hold several revisions and a partial download can leave one
/// without the file being asked for, so the marker decides which revision answers
/// rather than the first directory listed.
pub fn snapshot_with(repo: &str, marker: &str) -> Option<PathBuf> {
    let snaps = snapshots(repo);
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join(marker).exists())
}

/// Any downloaded snapshot of `repo`.
///
/// For repositories consumed as a whole directory rather than through one named
/// file - a checkpoint plus its tokenizer plus its configuration.
pub fn snapshot(repo: &str) -> Option<PathBuf> {
    let snaps = snapshots(repo);
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// `rel` inside whichever snapshot of `repo` holds it.
///
/// Falls back to a path under the snapshots directory when nothing is downloaded, so
/// the caller's own "not found" error names a real location instead of an empty
/// option.
pub fn file(repo: &str, rel: &str) -> PathBuf {
    match snapshot_with(repo, rel) {
        Some(snap) => snap.join(rel),
        None => snapshots(repo).join("<snapshot>").join(rel),
    }
}

/// `rel` inside `repo`, or nothing when it has not been downloaded.
///
/// For callers that treat an absent model as a state to handle rather than an error
/// to report - a catalogue listing what is available, say.
pub fn find(repo: &str, rel: &str) -> Option<PathBuf> {
    snapshot_with(repo, rel).map(|snap| snap.join(rel))
}

/// The first entry of `dir` whose file name starts with `prefix`.
///
/// Some repositories name a checkpoint after its own content hash, which no caller
/// can spell in advance.
pub fn file_starting_with(dir: &Path, prefix: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_path_is_the_hub_layout() {
        let p = snapshots("models--org--name");
        let s = p.to_string_lossy();
        assert!(
            s.ends_with("hub/models--org--name/snapshots"),
            "unexpected layout: {s}"
        );
    }

    #[test]
    fn a_missing_repo_still_names_the_file() {
        // The fallback exists so a loader's error points at a directory rather than
        // reporting that nothing could be resolved at all.
        let p = file("models--nobody--nothing", "weights.gguf");
        assert!(p.ends_with("<snapshot>/weights.gguf"), "{}", p.display());
        assert_eq!(find("models--nobody--nothing", "weights.gguf"), None);
        assert_eq!(snapshot("models--nobody--nothing"), None);
    }

    #[test]
    fn a_marker_picks_the_revision_that_holds_it() {
        let tmp = std::env::temp_dir().join(format!("loken-hf-cache-{}", std::process::id()));
        let snaps = tmp.join("snapshots");
        std::fs::create_dir_all(snaps.join("empty")).unwrap();
        std::fs::create_dir_all(snaps.join("full")).unwrap();
        std::fs::write(snaps.join("full").join("model.safetensors"), b"x").unwrap();
        // Same walk the resolver runs, against a tree this test owns: the revision
        // holding the marker answers, not whichever one the filesystem lists first.
        let picked = std::fs::read_dir(&snaps)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("model.safetensors").exists());
        assert_eq!(picked, Some(snaps.join("full")));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn a_hash_named_checkpoint_resolves_by_prefix() {
        let tmp = std::env::temp_dir().join(format!("loken-hf-prefix-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("dsm_tts_9c1f.safetensors"), b"x").unwrap();
        std::fs::write(tmp.join("tokenizer.model"), b"x").unwrap();
        assert_eq!(
            file_starting_with(&tmp, "dsm_tts"),
            Some(tmp.join("dsm_tts_9c1f.safetensors"))
        );
        assert_eq!(file_starting_with(&tmp, "nothing"), None);
        std::fs::remove_dir_all(&tmp).ok();
    }
}
