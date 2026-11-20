//! The commit and the version a build is stamped with, as the build script decides them.
//!
//! A `git archive` export has no repository.
//! It carries `.git_archival.txt`, whose placeholders `git archive` fills because
//! `.gitattributes` marks the file `export-subst`.
//! In a checkout the file holds the placeholders as written, and they say nothing.
//!
//! The build script includes this file.
//! The library compiles it for its tests only.

/// What `.git_archival.txt` says about the commit an export was made from.
#[derive(Debug, PartialEq, Eq)]
pub struct Export {
    /// The full commit hash.
    pub commit: String,
    /// The refs that pointed at the commit, as `git log --format=%D` lists them.
    pub refs: Vec<String>,
}

impl Export {
    /// Whether the tag `tag` pointed at the commit.
    pub fn tagged(&self, tag: &str) -> bool {
        self.refs
            .iter()
            .any(|r| r.strip_prefix("tag: ") == Some(tag))
    }
}

/// The export the text of `.git_archival.txt` describes.
/// `None` when the text holds no filled commit, as in a checkout.
pub fn parse_archival(text: &str) -> Option<Export> {
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name))
            .map(str::trim)
    };
    let node = field("node:")?;
    if node.len() != 40 || !node.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let refs = field("ref-names:")
        .filter(|r| !r.contains("$Format"))
        .map(|r| {
            r.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Some(Export {
        commit: node.to_string(),
        refs,
    })
}

/// The version a build answers with.
///
/// A commit carrying the release tag `v<release>` is that release.
/// Any other known commit is pre-release work on top of it, `<release>-dev+g<commit>` in SemVer
/// form.
/// A build that knows neither its commit nor its tags is taken as the release it was cut from.
pub fn version(release: &str, commit: Option<&str>, at_release_tag: Option<bool>) -> String {
    match (commit, at_release_tag) {
        (Some(commit), Some(false)) => format!("{release}-dev+g{commit}"),
        _ => release.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NODE: &str = "f877f72d4c0ffee0123456789abcdef012345678";

    /// A filled file gives its commit and the refs that pointed at it.
    #[test]
    fn a_filled_archival_file_gives_its_commit_and_refs() {
        let export = parse_archival(&format!(
            "node: {NODE}\nref-names: HEAD -> main, tag: v0.5.2, origin/main\n"
        ))
        .expect("an export");
        assert_eq!(export.commit, NODE);
        assert!(export.tagged("v0.5.2"));
        assert!(!export.tagged("v0.5.1"));
    }

    /// The file of a checkout, a damaged file and no file say nothing.
    #[test]
    fn placeholders_and_damage_give_no_export() {
        assert_eq!(
            parse_archival("node: $Format:%H$\nref-names: $Format:%D$\n"),
            None
        );
        assert_eq!(parse_archival("node: f877f72d4\n"), None);
        assert_eq!(parse_archival(""), None);
    }

    /// A filled commit with no refs is an untagged commit.
    #[test]
    fn a_commit_without_refs_carries_no_tag() {
        let export = parse_archival(&format!("node: {NODE}\nref-names: \n")).expect("an export");
        assert!(export.refs.is_empty());
        assert!(!export.tagged("v0.5.2"));
    }

    /// A release tag gives the release, any other known commit a development version, and a
    /// build that knows nothing the release it was cut from.
    #[test]
    fn the_version_tells_a_release_from_development() {
        assert_eq!(version("0.5.2", Some("f877f72d4"), Some(true)), "0.5.2");
        assert_eq!(
            version("0.5.2", Some("f877f72d4"), Some(false)),
            "0.5.2-dev+gf877f72d4"
        );
        assert_eq!(version("0.5.2", None, None), "0.5.2");
        assert_eq!(version("0.5.2", None, Some(false)), "0.5.2");
    }
}
