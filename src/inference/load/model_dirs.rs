//! Model directories on several disks, read as one catalogue.
//!
//! The configuration may list several Ollama stores and several Hugging Face stores. Together
//! they form one catalogue, and the order of the list is the order of preference:
//!
//! - A model held by several directories is read from the first of them in the list.
//! - A pull lands on the directory that already holds the model when it has room for the
//!   rest, so that the blobs models share stay on one disk; otherwise on the first directory
//!   in the list with room for the whole model.
//! - A model built here from uploaded or converted blobs is written to the first directory
//!   listed.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// What a pull destination is chosen from, for one directory.
#[derive(Debug, Clone)]
pub struct DirFacts {
    pub dir: PathBuf,
    /// Free bytes on its filesystem; `None` when unreadable.
    pub free: Option<u64>,
    /// Bytes of the model already in the directory.
    pub held: u64,
    /// Whether the directory already holds the model: its manifest or weights, its repository.
    pub holds: bool,
}

/// The directories a manager built on `dir` reads: every configured directory of its kind when
/// `dir` is one of them, else `dir` alone.
pub fn catalogue_of(dir: PathBuf, configured: Vec<PathBuf>) -> Vec<PathBuf> {
    if configured.contains(&dir) {
        configured
    } else {
        vec![dir]
    }
}

/// The directory a pull of `size` bytes lands on, as an index into `facts`, which are in the
/// configured order. A directory has room when its free space covers the bytes of the model it
/// lacks. The first directory that holds the model wins when it has room for the rest;
/// otherwise the first directory with room. The error names each directory's free space.
pub fn choose_destination(facts: &[DirFacts], size: u64) -> Result<usize, String> {
    let fits = |f: &DirFacts| {
        f.free
            .is_some_and(|free| free >= size.saturating_sub(f.held))
    };
    if let Some(i) = facts.iter().position(|f| f.holds && fits(f)) {
        return Ok(i);
    }
    if let Some(i) = facts.iter().position(fits) {
        return Ok(i);
    }
    Err(format!(
        "no model directory has room for {size} bytes: {}",
        facts
            .iter()
            .map(|f| match f.free {
                Some(free) => format!("{} has {free} bytes free", f.dir.display()),
                None => format!("{} has no readable free space", f.dir.display()),
            })
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The closest existing directory at or above `dir`, where a directory not yet created will be.
fn existing_ancestor(dir: &Path) -> Option<&Path> {
    dir.ancestors()
        .find(|p| !p.as_os_str().is_empty() && p.exists())
}

/// Free bytes on the filesystem a directory is, or will be, created on.
pub fn free_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let at = existing_ancestor(dir)?;
    let c = std::ffi::CString::new(at.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a valid NUL-terminated path and `st` a valid out parameter.
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

/// What a pull destination is chosen from: each directory's free space and what
/// `holding(dir)` says of the model there: `(bytes held, holds the model)`.
pub fn facts(dirs: &[PathBuf], holding: impl Fn(&Path) -> (u64, bool)) -> Vec<DirFacts> {
    dirs.iter()
        .map(|d| {
            let (held, holds) = holding(d);
            DirFacts {
                dir: d.clone(),
                free: free_bytes(d),
                held,
                holds,
            }
        })
        .collect()
}

/// Which of several copies of `model` is read: the first of `holders`, which are in the
/// configured order.
pub fn explain_copies(model: &str, holders: &[PathBuf]) -> String {
    let list = holders
        .iter()
        .map(|d| d.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    match holders.first() {
        Some(used) => format!(
            "{model}: copies in {list}; {} is read, the first in the configured order",
            used.display()
        ),
        None => format!("{model}: no copy"),
    }
}

/// Logs once per model which of several copies is read; `holders` in the configured order.
pub fn report_copies(model: &str, holders: &[PathBuf]) {
    static REPORTED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    if holders.len() < 2 {
        return;
    }
    let mut reported = REPORTED.lock().unwrap_or_else(|e| e.into_inner());
    if reported
        .get_or_insert_with(HashSet::new)
        .insert(model.to_string())
    {
        tracing::info!("{}", explain_copies(model, holders));
    }
}

/// The directories of one kind in the order a model held by several of them is read: the
/// configured order.
pub fn read_order(dirs: &[PathBuf]) -> Vec<PathBuf> {
    dirs.to_vec()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A fresh directory under the build's own target directory, standing for one disk.
    pub(crate) fn scratch(tag: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch")
            .join("model_dirs")
            .join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn dir(name: &str, free: u64) -> DirFacts {
        DirFacts {
            dir: PathBuf::from(name),
            free: Some(free),
            held: 0,
            holds: false,
        }
    }

    #[test]
    fn a_pull_lands_on_the_first_directory_with_room() {
        let facts = [dir("/a", 10), dir("/b", 1000), dir("/c", 1000)];
        assert_eq!(choose_destination(&facts, 500), Ok(1));
    }

    #[test]
    fn a_directory_holding_the_model_keeps_it() {
        let mut holder = dir("/second", 100);
        holder.held = 450;
        holder.holds = true;
        let facts = [dir("/first", 1000), holder.clone()];
        // 50 bytes lacking fit in 100 free: the shared blobs stay where they are.
        assert_eq!(choose_destination(&facts, 500), Ok(1));
        // Without room for the rest, the first directory with room takes the whole model.
        holder.free = Some(10);
        let facts = [dir("/first", 1000), holder];
        assert_eq!(choose_destination(&facts, 500), Ok(0));
    }

    #[test]
    fn no_room_anywhere_names_every_directory() {
        let mut unreadable = dir("/b", 0);
        unreadable.free = None;
        let err = choose_destination(&[dir("/a", 10), unreadable], 500).unwrap_err();
        assert!(err.contains("/a has 10 bytes free"), "{err}");
        assert!(err.contains("/b has no readable free space"), "{err}");
    }

    #[test]
    fn the_copy_read_is_the_first_listed() {
        let said = explain_copies("m:1b", &[PathBuf::from("/a"), PathBuf::from("/b")]);
        assert!(said.contains("copies in /a, /b; /a is read"), "{said}");
        assert_eq!(
            read_order(&[PathBuf::from("/b"), PathBuf::from("/a")]),
            [PathBuf::from("/b"), PathBuf::from("/a")]
        );
    }

    #[test]
    fn free_space_is_read_where_a_directory_will_be() {
        let d = scratch("free").join("not-yet").join("created");
        assert!(free_bytes(&d).is_some_and(|f| f > 0));
    }

    #[test]
    fn a_configured_directory_reads_the_whole_list() {
        let list = vec![PathBuf::from("/a"), PathBuf::from("/b")];
        assert_eq!(catalogue_of(PathBuf::from("/b"), list.clone()), list);
        assert_eq!(
            catalogue_of(PathBuf::from("/c"), list),
            vec![PathBuf::from("/c")]
        );
    }
}
