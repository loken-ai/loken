//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl LlmEngine {
    /// Extract models root path from manifest path
    /// Path: {root}/manifests/{registry}/library/{model_name}/{tag}
    pub(super) fn extract_models_root(
        manifest_path: &std::path::Path,
    ) -> Option<std::path::PathBuf> {
        let mut current = manifest_path;
        while let Some(parent) = current.parent() {
            if current.file_name().is_some_and(|n| n == "manifests") {
                return Some(parent.to_path_buf());
            }
            current = parent;
        }
        None
    }

    /// Find blob file for a given digest in the models root
    pub(super) fn find_blob_for_digest(
        models_root: &std::path::Path,
        digest: &str,
    ) -> Option<std::path::PathBuf> {
        // Split digest into algorithm and hash (e.g., "sha256:abc123" -> "sha256", "abc123")
        let (algo, hash) = if let Some(colon_pos) = digest.find(':') {
            digest.split_at(colon_pos)
        } else {
            ("sha256", digest)
        };
        let hash = hash.strip_prefix(':').unwrap_or(hash);

        // Ollama blob storage format: blobs/{algorithm}-{hash} (single file, not directory)
        let blob_filename = format!("{}-{}", algo, hash);
        let blob_path = models_root.join("blobs").join(&blob_filename);
        if blob_path.exists() && blob_path.is_file() {
            return Some(blob_path);
        }

        // Also try OCI format: blobs/{algorithm}/{hash} for compatibility
        let oci_blob_path = models_root.join("blobs").join(algo).join(hash);
        if oci_blob_path.exists() && oci_blob_path.is_file() {
            return Some(oci_blob_path);
        }

        None
    }

    /// Load model blob from Ollama manifest file
    pub(super) fn load_from_manifest(&self, manifest_path: &std::path::Path) -> Option<PathBuf> {
        if let Ok(content) = std::fs::read_to_string(manifest_path) {
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
                // Try to get the first layer's digest (should be the model file)
                if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
                    if let Some(layer) = layers.first() {
                        if let Some(digest) = layer.get("digest").and_then(|d| d.as_str()) {
                            // Extract models root
                            if let Some(root) = Self::extract_models_root(manifest_path) {
                                debug!("🔍 Looking for blob: {}", digest);
                                if let Some(blob_path) = Self::find_blob_for_digest(&root, digest) {
                                    debug!("📍 Found model blob at: {}", blob_path.display());
                                    return Some(blob_path);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Resolve the projector blob (if any) for a model. Walks the same
    /// Ollama manifest tree as `find_model_file` but scans layers for
    /// `application/vnd.ollama.image.projector` (vision models like
    /// moondream/llava ship the vision tower as a separate blob).
    /// Returns None for text-only models.
    pub(super) fn find_projector_blob_for_model(&self, model_id: &str) -> Option<PathBuf> {
        let models_dir = self.config.models_dir.as_ref()?;
        if !models_dir.is_dir() {
            return None;
        }
        // Build the canonical Ollama manifest path. This covers the
        // standard `library/{name}/{tag}` and namespaced `{ns}/{name}/{tag}`
        // layouts - same prefixes find_model_file walks but without the
        // full directory scan since we only need to find the projector
        // for an already-loaded model.
        let (model_name, tag) = if model_id.contains(':') {
            let parts: Vec<&str> = model_id.split(':').collect();
            (
                parts[0].to_lowercase(),
                parts.get(1).copied().unwrap_or("latest").to_string(),
            )
        } else {
            (model_id.to_lowercase(), "latest".to_string())
        };
        let manifests = models_dir.join("manifests");
        let mut candidate_manifests: Vec<PathBuf> = Vec::new();
        if let Ok(registry_entries) = std::fs::read_dir(&manifests) {
            for re in registry_entries.flatten() {
                let rp = re.path();
                if !rp.is_dir() {
                    continue;
                }
                if model_name.contains('/') {
                    let parts: Vec<&str> = model_name.splitn(2, '/').collect();
                    candidate_manifests.push(rp.join(parts[0]).join(parts[1]).join(&tag));
                    if tag != "latest" {
                        candidate_manifests.push(rp.join(parts[0]).join(parts[1]).join("latest"));
                    }
                }
                let lib = rp.join("library").join(&model_name);
                candidate_manifests.push(lib.join(&tag));
                if tag != "latest" {
                    candidate_manifests.push(lib.join("latest"));
                }
            }
        }
        for manifest_path in candidate_manifests {
            if !manifest_path.is_file() {
                continue;
            }
            let content = match std::fs::read_to_string(&manifest_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let manifest: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let layers = match manifest.get("layers").and_then(|l| l.as_array()) {
                Some(l) => l,
                None => continue,
            };
            for layer in layers {
                let mtype = layer
                    .get("mediaType")
                    .and_then(|m| m.as_str())
                    .unwrap_or("");
                if mtype != "application/vnd.ollama.image.projector" {
                    continue;
                }
                let digest = match layer.get("digest").and_then(|d| d.as_str()) {
                    Some(d) => d,
                    None => continue,
                };
                let root = match Self::extract_models_root(&manifest_path) {
                    Some(r) => r,
                    None => continue,
                };
                if let Some(blob) = Self::find_blob_for_digest(&root, digest) {
                    debug!("📍 Found projector blob: {}", blob.display());
                    return Some(blob);
                }
            }
        }
        // Direct-path GGUF (no Ollama manifest): a sibling `mmproj*.gguf` in
        // the model file's directory IS the projector - the llama.cpp/HF
        // convention (model.gguf + mmproj-*.gguf side by side, e.g. Pixtral).
        let model_file = self.find_model_file(model_id)?;
        let dir = model_file.parent()?;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                let Some(name) = p.file_name().map(|n| n.to_string_lossy().to_lowercase()) else {
                    continue;
                };
                if name.starts_with("mmproj") && name.ends_with(".gguf") && p != model_file {
                    debug!("📍 Found sibling mmproj projector: {}", p.display());
                    return Some(p);
                }
            }
        }
        None
    }

    /// Find model file by name
    pub(super) fn find_model_file(&self, model_id: &str) -> Option<PathBuf> {
        // AWQ (HF safetensors) checkpoint: the model reference is a
        // directory whose config.json declares quant_method==awq (a direct
        // snapshot dir or an HF hub dir with snapshots/<hash>). Return the
        // resolved dir; `load_model` detects it and uses the AWQ loader. This
        // only fires for real AWQ dirs, so the Ollama/GGUF path is untouched.
        if let Some(dir) = resolve_awq_dir(model_id) {
            return Some(PathBuf::from(dir));
        }
        {
            // HF repo-id (`org/name`) -> hub dir under the configured models_dir's
            // huggingface/hub cache, plus the configured huggingface_models_dir.
            let mut roots: Vec<PathBuf> = Vec::new();
            if let Some(md) = &self.config.models_dir {
                roots.push(md.join("huggingface").join("hub"));
                roots.push(md.clone());
                if let Some(dir) = resolve_awq_dir(&md.to_string_lossy()) {
                    return Some(PathBuf::from(dir));
                }
            }
            // The configured HuggingFace hub root (from config.toml, OS-default
            // fallback) - never a hardcoded absolute path.
            if let Ok(cfg) = crate::config::Config::load_default() {
                let hf = cfg.get_hf_models_dir();
                let hub = if hf.file_name().is_some_and(|n| n == "hub") {
                    hf
                } else {
                    hf.join("hub")
                };
                roots.push(hub);
            }
            if let Some(dir) = resolve_awq_repo(model_id, &roots) {
                return Some(PathBuf::from(dir));
            }
            if let Some(file) = resolve_hub_gguf(model_id, &roots) {
                return Some(file);
            }
            if let Some(dir) = resolve_hub_checkpoint(model_id, &roots) {
                return Some(dir);
            }
        }
        // Try to find GGUF file in models directory
        if let Some(models_dir) = &self.config.models_dir {
            // If models_dir points directly to a GGUF file, use it
            if models_dir.exists() && models_dir.is_file() {
                if models_dir.extension().is_some_and(|e| e == "gguf") {
                    return Some(models_dir.clone());
                }
                // Otherwise treat as manifest
                return self.load_from_manifest(models_dir);
            }

            // models_dir is the Ollama root - try to find the manifest
            // Ollama structure: ~/.ollama/models/manifests/{registry}/library/{model_name}/{tag}
            if models_dir.is_dir() {
                debug!(
                    "📂 Looking for Ollama manifest in: {}",
                    models_dir.display()
                );

                // Parse model_id to construct manifest path
                // Examples: "tinyllama:latest" or "devstral-small-2:latest"
                let (model_name, tag) = if model_id.contains(':') {
                    let parts: Vec<&str> = model_id.split(':').collect();
                    (
                        parts[0].to_lowercase(),
                        parts.get(1).copied().unwrap_or("latest"),
                    )
                } else {
                    (model_id.to_lowercase(), "latest")
                };

                debug!("🔍 Looking for model: '{}' with tag: '{}'", model_name, tag);

                // Check if manifests directory exists
                let manifests_dir = models_dir.join("manifests");
                debug!("📂 Manifests dir exists: {}", manifests_dir.exists());

                if manifests_dir.exists() {
                    // Scan for all registries in manifests/ and look for the model
                    if let Ok(registry_entries) = std::fs::read_dir(&manifests_dir) {
                        for registry_entry in registry_entries.flatten() {
                            if let Ok(registry_type) = registry_entry.file_type() {
                                if registry_type.is_dir() {
                                    let registry_path = registry_entry.path();
                                    if let Some(registry_name) =
                                        registry_path.file_name().and_then(|n| n.to_str())
                                    {
                                        debug!("📂 Found registry: {}", registry_name);

                                        // For namespaced models (e.g., "x/flux2-klein"),
                                        // try {namespace}/{model}/{tag} directly under registry
                                        if model_name.contains('/') {
                                            let parts: Vec<&str> =
                                                model_name.splitn(2, '/').collect();
                                            let model_dir =
                                                registry_path.join(parts[0]).join(parts[1]);
                                            if model_dir.exists() {
                                                debug!(
                                                    "📂 Found namespaced model dir: {}",
                                                    model_dir.display()
                                                );
                                                let manifest_path = model_dir.join(tag);
                                                if manifest_path.exists() && manifest_path.is_file()
                                                {
                                                    debug!("✅ Found manifest!");
                                                    if let Some(blob) =
                                                        self.load_from_manifest(&manifest_path)
                                                    {
                                                        return Some(blob);
                                                    }
                                                }
                                                if tag != "latest" {
                                                    let manifest_path_latest =
                                                        model_dir.join("latest");
                                                    if manifest_path_latest.exists()
                                                        && manifest_path_latest.is_file()
                                                    {
                                                        debug!("✅ Found manifest (latest)!");
                                                        if let Some(blob) = self.load_from_manifest(
                                                            &manifest_path_latest,
                                                        ) {
                                                            return Some(blob);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // Try library/{model_name}/{tag} (official models)
                                        let library_dir = registry_path.join("library");
                                        if library_dir.exists() {
                                            let model_dir = library_dir.join(&model_name);
                                            if model_dir.exists() {
                                                debug!(
                                                    "📂 Found model dir at registry '{}': {}",
                                                    registry_name,
                                                    model_dir.display()
                                                );

                                                // Try with specified tag
                                                let manifest_path = model_dir.join(tag);
                                                debug!(
                                                    "🔍 Trying manifest: {}",
                                                    manifest_path.display()
                                                );
                                                if manifest_path.exists() && manifest_path.is_file()
                                                {
                                                    debug!("✅ Found manifest!");
                                                    if let Some(blob) =
                                                        self.load_from_manifest(&manifest_path)
                                                    {
                                                        return Some(blob);
                                                    }
                                                }

                                                // Try with "latest" tag
                                                if tag != "latest" {
                                                    let manifest_path_latest =
                                                        model_dir.join("latest");
                                                    debug!(
                                                        "🔍 Trying manifest (latest): {}",
                                                        manifest_path_latest.display()
                                                    );
                                                    if manifest_path_latest.exists()
                                                        && manifest_path_latest.is_file()
                                                    {
                                                        debug!("✅ Found manifest!");
                                                        if let Some(blob) = self.load_from_manifest(
                                                            &manifest_path_latest,
                                                        ) {
                                                            return Some(blob);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    debug!("❌ Model not found in any registry");
                }
            }

            // Try direct path first
            let direct_path = models_dir.join(model_id);
            if direct_path.exists() && direct_path.is_file() {
                return Some(direct_path);
            }

            // Try with .gguf extension
            let gguf_path = direct_path.with_extension("gguf");
            if gguf_path.exists() && gguf_path.is_file() {
                return Some(gguf_path);
            }

            // Try in subdirectory (Ollama cache style)
            let subdir_path = models_dir.join(model_id).join("model.gguf");
            if subdir_path.exists() && subdir_path.is_file() {
                return Some(subdir_path);
            }

            // Try HuggingFace style
            let hf_path = models_dir.join(model_id).join("snapshots");
            if hf_path.exists() && hf_path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&hf_path) {
                    for entry in entries.flatten() {
                        if let Ok(file_type) = entry.file_type() {
                            if file_type.is_dir() {
                                if let Ok(snapshot_entries) = std::fs::read_dir(entry.path()) {
                                    for snapshot_entry in snapshot_entries.flatten() {
                                        let path = snapshot_entry.path();
                                        if path.extension().is_some_and(|e| e == "gguf") {
                                            return Some(path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

/// The GGUF of a hub repository id `org/name[:selector]` inside the cache roots: any snapshot
/// of the repository, at any depth, since a repository keeps each quantisation in its own
/// directory. The selector, when given, must appear in the path below the snapshot; a split
/// model resolves to its first part. Several candidates left with no selector to tell them
/// apart resolve to none, listed in the log, rather than to an arbitrary one.
pub(super) fn resolve_hub_gguf(model_id: &str, roots: &[PathBuf]) -> Option<PathBuf> {
    let (base, selector) = match model_id.split_once(':') {
        Some((b, s)) => (b, Some(s)),
        None => (model_id, None),
    };
    let (org, name) = base.split_once('/')?;
    if org.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    let hub = format!("models--{org}--{name}");
    let mut found: Vec<PathBuf> = Vec::new();
    for root in roots {
        let snapshots = root.join(&hub).join("snapshots");
        let Ok(snaps) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for snap in snaps.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
            let mut stack = vec![snap.clone()];
            while let Some(dir) = stack.pop() {
                let Ok(entries) = std::fs::read_dir(&dir) else {
                    continue;
                };
                for p in entries.flatten().map(|e| e.path()) {
                    if p.is_dir() {
                        stack.push(p);
                    } else if p.extension().is_some_and(|e| e == "gguf") {
                        let rel = p
                            .strip_prefix(&snap)
                            .unwrap_or(&p)
                            .to_string_lossy()
                            .to_string();
                        let selected = selector.is_none_or(|s| rel.contains(s));
                        if selected && split_part(&p).is_none_or(|n| n == 1) {
                            found.push(p);
                        }
                    }
                }
            }
        }
    }
    found.sort();
    found.dedup();
    match found.as_slice() {
        [one] => Some(one.clone()),
        [] => None,
        many => {
            warn!(
                "{model_id}: {} GGUF files in the cache; name one with {base}:<selector>: {}",
                many.len(),
                many.iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            None
        }
    }
}

/// The snapshot directory of a hub repository id whose checkpoint is a set of safetensors shards
/// with the model's own `config.json`: the loader that serves it is chosen from that config.
pub(super) fn resolve_hub_checkpoint(model_id: &str, roots: &[PathBuf]) -> Option<PathBuf> {
    let (org, name) = model_id.split_once('/')?;
    if org.is_empty() || name.is_empty() || name.contains('/') || name.contains(':') {
        return None;
    }
    let hub = format!("models--{org}--{name}");
    for root in roots {
        let Ok(snaps) = std::fs::read_dir(root.join(&hub).join("snapshots")) else {
            continue;
        };
        let mut dirs: Vec<PathBuf> = snaps
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for snap in dirs {
            let has_shards = std::fs::read_dir(&snap)
                .map(|rd| {
                    rd.flatten()
                        .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                })
                .unwrap_or(false);
            if has_shards && snap.join("config.json").is_file() {
                return Some(snap);
            }
        }
    }
    None
}

/// The part number of a split GGUF named `<stem>-NNNNN-of-MMMMM.gguf`, `None` for a whole file.
fn split_part(path: &std::path::Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    let (head, tail) = stem.rsplit_once("-of-")?;
    if tail.is_empty() || !tail.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (_, part) = head.rsplit_once('-')?;
    if part.len() != tail.len() {
        return None;
    }
    part.parse().ok()
}

#[cfg(test)]
mod hub_gguf_tests {
    use super::*;

    fn touch(p: &std::path::Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"").unwrap();
    }

    /// A whole file resolves; a split resolves to its first part; two quantisations need the
    /// selector; a name with no slash is not a repository id.
    #[test]
    fn resolves_whole_split_and_selected_files() {
        let root = std::env::temp_dir().join(format!("loken-hub-gguf-{}", std::process::id()));
        let snap = root.join("models--org--one").join("snapshots").join("abc");
        touch(&snap.join("one-q4.gguf"));
        let snap2 = root.join("models--org--two").join("snapshots").join("def");
        touch(&snap2.join("Q2_K").join("two-Q2_K-00001-of-00003.gguf"));
        touch(&snap2.join("Q2_K").join("two-Q2_K-00002-of-00003.gguf"));
        touch(&snap2.join("Q2_K").join("two-Q2_K-00003-of-00003.gguf"));
        touch(&snap2.join("Q4_K").join("two-Q4_K.gguf"));
        let roots = vec![root.clone()];

        assert_eq!(
            resolve_hub_gguf("org/one", &roots),
            Some(snap.join("one-q4.gguf"))
        );
        assert_eq!(resolve_hub_gguf("org/two", &roots), None);
        assert_eq!(
            resolve_hub_gguf("org/two:Q2_K", &roots),
            Some(snap2.join("Q2_K").join("two-Q2_K-00001-of-00003.gguf"))
        );
        assert_eq!(
            resolve_hub_gguf("org/two:Q4_K", &roots),
            Some(snap2.join("Q4_K").join("two-Q4_K.gguf"))
        );
        assert_eq!(resolve_hub_gguf("org/three", &roots), None);
        assert_eq!(resolve_hub_gguf("one", &roots), None);
        assert_eq!(
            split_part(std::path::Path::new("a-00002-of-00007.gguf")),
            Some(2)
        );
        assert_eq!(split_part(std::path::Path::new("a-of-b.gguf")), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}
