//! Companion GGUF files of a model file: the routing prior a model that carries none borrows, and
//! the draft it does not hold.
//!
//! A blob of an Ollama store takes its companions from the manifests that name it.
//! A plain file outside a store takes them from its directory.
//! Among those files, a companion is found by the metadata keys it declares, whatever its name.
//! Scanning reads only each candidate's key table, so a directory of large model files costs a
//! few header reads, not their weights.

use super::wide_sparse_names::ARCH;
use crate::tensor::gguf_write::write_gguf_with_metadata;
use crate::tensor::quantized::gguf_file::{open_mapped_unadvised, Value};
use crate::tensor::{Error, Result};
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

/// The largest key a header can plausibly declare; anything longer is not a GGUF key table.
const MAX_KEY_BYTES: u64 = 1 << 16;

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn skip<R: Seek>(r: &mut R, bytes: u64) -> std::io::Result<()> {
    r.seek_relative(i64::try_from(bytes).map_err(|_| std::io::ErrorKind::InvalidData)?)
}

/// Step over one metadata value of GGUF type `ty`.
fn skip_value<R: Read + Seek>(r: &mut R, ty: u32) -> std::io::Result<()> {
    match ty {
        0 | 1 | 7 => skip(r, 1),
        2 | 3 => skip(r, 2),
        4..=6 => skip(r, 4),
        10..=12 => skip(r, 8),
        8 => {
            let n = read_u64(r)?;
            skip(r, n)
        }
        9 => {
            let element = read_u32(r)?;
            let count = read_u64(r)?;
            match element {
                8 | 9 => {
                    for _ in 0..count {
                        skip_value(r, element)?;
                    }
                    Ok(())
                }
                0 | 1 | 7 => skip(r, count),
                2 | 3 => skip(r, count * 2),
                4..=6 => skip(r, count * 4),
                10..=12 => skip(r, count * 8),
                _ => Err(std::io::ErrorKind::InvalidData.into()),
            }
        }
        _ => Err(std::io::ErrorKind::InvalidData.into()),
    }
}

/// Whether the GGUF at `path` declares a metadata key starting with `prefix`; false for a file
/// that is not a GGUF or cannot be read.
pub fn declares(path: &Path, prefix: &str) -> bool {
    let scan = || -> std::io::Result<bool> {
        let mut r = BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Ok(false);
        }
        let _version = read_u32(&mut r)?;
        let _tensors = read_u64(&mut r)?;
        let keys = read_u64(&mut r)?;
        for _ in 0..keys {
            let n = read_u64(&mut r)?;
            if n > MAX_KEY_BYTES {
                return Ok(false);
            }
            let mut key = vec![0u8; n as usize];
            r.read_exact(&mut key)?;
            if key.starts_with(prefix.as_bytes()) {
                return Ok(true);
            }
            let ty = read_u32(&mut r)?;
            skip_value(&mut r, ty)?;
        }
        r.stream_position()?;
        Ok(false)
    };
    scan().unwrap_or(false)
}

/// The suffix of the tag a conversion stores a model's draft companion under: `<name>:<tag>` keeps
/// its draft at `<name>:<tag>-dspark`.
pub const COMPANION_TAG_SUFFIX: &str = "-dspark";

/// The media type of the layer that names a manifest's model file.
const MODEL_LAYER: &str = "application/vnd.ollama.image.model";

/// The root of the Ollama store `blob` sits in: the directory holding `blobs/` and
/// `manifests/`.
///
/// `None` for a file outside a store.
fn store_root(blob: &Path) -> Option<PathBuf> {
    blob.ancestors()
        .skip(1)
        .take(2)
        .find(|d| d.file_name().is_some_and(|n| n == "blobs"))
        .and_then(Path::parent)
        .filter(|root| root.join("manifests").is_dir())
        .map(Path::to_path_buf)
}

/// The blob of `digest` in the store at `root`, when the store holds it.
fn blob_of(root: &Path, digest: &str) -> Option<PathBuf> {
    let (algo, hash) = digest.split_once(':').unwrap_or(("sha256", digest));
    [
        root.join("blobs").join(format!("{algo}-{hash}")),
        root.join("blobs").join(algo).join(hash),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// The layers of the manifest at `path`, as `(media type, digest)`.
fn manifest_layers(path: &Path) -> Vec<(String, String)> {
    let Some(manifest) = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    else {
        return Vec::new();
    };
    let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) else {
        return Vec::new();
    };
    layers
        .iter()
        .filter_map(|l| {
            let media = l.get("mediaType")?.as_str()?;
            let digest = l.get("digest")?.as_str()?;
            Some((media.to_string(), digest.to_string()))
        })
        .collect()
}

/// Every manifest of the store at `root`: `manifests/<registry>/<namespace>/<name>/<tag>`.
fn store_manifests(root: &Path) -> Vec<PathBuf> {
    let children = |p: &Path| -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = std::fs::read_dir(p)
            .map(|it| it.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        out.sort();
        out
    };
    let mut out = Vec::new();
    for registry in children(&root.join("manifests")) {
        for namespace in children(&registry) {
            for name in children(&namespace) {
                out.extend(children(&name).into_iter().filter(|p| p.is_file()));
            }
        }
    }
    out
}

/// The files of the store at `root` that the manifests naming `blob` as their model list beside
/// it.
///
/// Each such manifest gives its further layers, then the model of its companion tag.
fn store_companions(root: &Path, blob: &Path) -> Vec<PathBuf> {
    let me = std::fs::canonicalize(blob).ok();
    let same = |p: &Path| std::fs::canonicalize(p).ok() == me;
    let mut out: Vec<PathBuf> = Vec::new();
    for manifest in store_manifests(root) {
        let layers = manifest_layers(&manifest);
        let names_blob = layers.iter().any(|(media, digest)| {
            media == MODEL_LAYER && blob_of(root, digest).is_some_and(|p| same(&p))
        });
        if !names_blob {
            continue;
        }
        let companion_tag = manifest.file_name().map(|tag| {
            let mut tag = tag.to_os_string();
            tag.push(COMPANION_TAG_SUFFIX);
            manifest.with_file_name(tag)
        });
        let companion_model = companion_tag
            .iter()
            .flat_map(|m| manifest_layers(m))
            .filter(|(media, _)| media == MODEL_LAYER);
        for (_, digest) in layers.iter().cloned().chain(companion_model) {
            match blob_of(root, &digest) {
                Some(p) if !same(&p) && !out.contains(&p) => out.push(p),
                _ => {}
            }
        }
    }
    out
}

/// The files `model` may take a companion from, other than `model` itself.
///
/// A blob of an Ollama store takes the files the manifests naming it list.
/// A plain file takes the files of its directory, in name order.
fn companion_candidates(model: &Path) -> Vec<PathBuf> {
    if let Some(root) = store_root(model) {
        return store_companions(&root, model);
    }
    let Some(dir) = model.parent() else {
        return Vec::new();
    };
    let me = std::fs::canonicalize(model).ok();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| std::fs::canonicalize(p).ok() != me)
        .collect();
    found.sort();
    found
}

/// The companion files of `model` whose metadata declares a key starting with `prefix`.
pub fn find_companions(model: &Path, prefix: &str) -> Vec<PathBuf> {
    companion_candidates(model)
        .into_iter()
        .filter(|p| declares(p, prefix))
        .collect()
}

fn prior_key(layer: usize) -> String {
    format!("{}.hot_experts.{layer}", ARCH)
}

/// The routing prior a companion file of `model` declares: for each of `n_layers` layers its
/// experts from the most routed-to down, every id below `n_routed`. `None` when no companion
/// declares one that fits this model; a companion that does not fit is passed over.
pub fn prior_beside(model: &Path, n_layers: usize, n_routed: usize) -> Option<Vec<Vec<usize>>> {
    let key = format!("{}.hot_experts.", ARCH);
    for path in find_companions(model, &key) {
        let Ok(file) = open_mapped_unadvised(&path) else {
            continue;
        };
        let md = &file.content.metadata;
        let lists: Option<Vec<Vec<usize>>> = (0..n_layers)
            .map(|l| {
                md.get(&prior_key(l))?
                    .to_vec()
                    .ok()?
                    .iter()
                    .map(|v| v.to_u32().ok().map(|x| x as usize))
                    .collect::<Option<Vec<usize>>>()
            })
            .collect();
        match lists {
            Some(lists) if lists.iter().flatten().all(|&e| e < n_routed) => {
                tracing::info!(
                    "routing prior taken from {} ({n_layers} layers)",
                    path.display()
                );
                return Some(lists);
            }
            _ => tracing::warn!(
                "{} declares a routing prior that does not fit this model; passed over",
                path.display()
            ),
        }
    }
    None
}

/// The routing prior as metadata: for each layer, its experts from the most routed-to on the
/// calibration corpus downwards.
///
/// A loader reads the head of each list to fill its cards before the first request.
pub fn hot_experts_metadata(lists: &[Vec<u32>]) -> Vec<(String, Value)> {
    lists
        .iter()
        .enumerate()
        .map(|(l, ids)| {
            (
                format!("{ARCH}.hot_experts.{l}"),
                Value::Array(ids.iter().map(|&x| Value::U32(x)).collect()),
            )
        })
        .collect()
}

/// Write a companion file holding only a routing prior: `lists[l]` ranks layer `l`'s experts.
pub fn write_prior(path: &Path, lists: &[Vec<u32>]) -> Result<()> {
    let mut md = vec![(
        "general.architecture".to_string(),
        Value::String(ARCH.to_string()),
    )];
    md.extend(hot_experts_metadata(lists));
    write_gguf_with_metadata(path, &md, &[]).map_err(|e| Error::msg(format!("{e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("loken-sibling-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn lists(layers: usize, experts: u32) -> Vec<Vec<u32>> {
        (0..layers)
            .map(|l| {
                (0..experts)
                    .rev()
                    .map(|e| (e + l as u32) % experts)
                    .collect()
            })
            .collect()
    }

    /// A companion is found by what its metadata declares: a decoy with other keys, a file that is
    /// no GGUF, and the model itself are passed over, whatever the names.
    #[test]
    fn a_prior_is_found_beside_the_model_by_its_keys() {
        let d = dir("found");
        std::fs::write(d.join("model.bin"), b"not a gguf at all").unwrap();
        std::fs::write(d.join("a-note"), b"GGUF but short").unwrap();
        write_gguf_with_metadata(
            &d.join("decoy"),
            &[("general.architecture".into(), Value::String("other".into()))],
            &[],
        )
        .unwrap();
        write_prior(&d.join("zz-side"), &lists(3, 8)).unwrap();
        let got = prior_beside(&d.join("model.bin"), 3, 8).expect("a prior");
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], vec![7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(got[1], vec![0, 7, 6, 5, 4, 3, 2, 1]);
        // The companion is not its own sibling.
        assert!(prior_beside(&d.join("zz-side"), 3, 8).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A store whose `blobs/` holds 2 models: `a:t` with its draft at `a:t-dspark` and its prior
    /// in a further layer, and `b:t`, whose blob sorts first and declares a draft and a prior.
    /// Returns the store and the blobs of `a:t`, of its draft, of its prior and of `b:t`.
    fn two_model_store(name: &str) -> (PathBuf, [PathBuf; 4]) {
        let d = dir(name);
        let blobs = d.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let blob = |hex: &str| blobs.join(format!("sha256-{hex}"));
        let draft_key = |v: u32| vec![(format!("{ARCH}.dspark.block_size"), Value::U32(v))];
        std::fs::write(blob("cc"), b"model a").unwrap();
        write_gguf_with_metadata(&blob("dd"), &draft_key(1), &[]).unwrap();
        write_prior(&blob("ee"), &lists(3, 8)).unwrap();
        let mut b_md = draft_key(2);
        b_md.extend(hot_experts_metadata(&lists(3, 8)));
        write_gguf_with_metadata(&blob("aa"), &b_md, &[]).unwrap();
        let tags = d.join("manifests").join("registry.example").join("library");
        let manifest = |name: &str, tag: &str, layers: &[(&str, &str)]| {
            let dir = tags.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            let layers: Vec<serde_json::Value> = layers
                .iter()
                .map(|(media, hex)| {
                    serde_json::json!({"mediaType": media, "digest": format!("sha256:{hex}"), "size": 1})
                })
                .collect();
            std::fs::write(
                dir.join(tag),
                serde_json::json!({ "layers": layers }).to_string(),
            )
            .unwrap();
        };
        manifest(
            "a",
            "t",
            &[
                (MODEL_LAYER, "cc"),
                ("application/vnd.loken.image.model.variant", "ee"),
            ],
        );
        manifest("a", "t-dspark", &[(MODEL_LAYER, "dd")]);
        manifest("b", "t", &[(MODEL_LAYER, "aa")]);
        (d, [blob("cc"), blob("dd"), blob("ee"), blob("aa")])
    }

    /// In a store, a model takes its draft from its companion tag and its prior from its own
    /// manifest, never from another model's blob in the same directory.
    #[test]
    fn a_store_blob_takes_its_companions_from_its_manifests() {
        let (d, [a, a_draft, a_prior, b]) = two_model_store("store");
        let draft = format!("{ARCH}.dspark.");
        assert_eq!(find_companions(&a, &draft), vec![a_draft]);
        let prior = format!("{ARCH}.hot_experts.");
        assert_eq!(find_companions(&a, &prior), vec![a_prior]);
        assert_eq!(
            prior_beside(&a, 3, 8).expect("a prior")[0],
            vec![7, 6, 5, 4, 3, 2, 1, 0]
        );
        // The other model lists no companion, though its own blob declares both keys.
        assert!(find_companions(&b, &draft).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A blob that no manifest names takes no companion from the store.
    #[test]
    fn a_store_blob_no_manifest_names_has_no_companion() {
        let (d, [_, _, _, b]) = two_model_store("unnamed");
        let orphan = d.join("blobs").join("sha256-ff");
        std::fs::copy(&b, &orphan).unwrap();
        assert!(find_companions(&orphan, &format!("{ARCH}.dspark.")).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A prior of another shape is passed over silently, and a moved one is simply not there.
    #[test]
    fn a_prior_that_does_not_fit_or_is_gone_gives_none() {
        let d = dir("fit");
        std::fs::write(d.join("model.bin"), b"x").unwrap();
        write_prior(&d.join("side"), &lists(3, 8)).unwrap();
        assert!(
            prior_beside(&d.join("model.bin"), 4, 8).is_none(),
            "layer count"
        );
        assert!(
            prior_beside(&d.join("model.bin"), 3, 6).is_none(),
            "expert count"
        );
        std::fs::remove_file(d.join("side")).unwrap();
        assert!(prior_beside(&d.join("model.bin"), 3, 8).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }
}
