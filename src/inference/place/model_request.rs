//! What to load, and where the weights live - carried together.
//!
//! These travelled as two adjacent `String` arguments through the loaders, and the loaders
//! disagreed about their order: some took the name first, some the directory. One pair
//! straddled both conventions, so a model NAME reached the download root and thirteen
//! gigabytes were fetched into a directory named after the model, beside the copy the
//! configuration already pointed at. Nothing failed, because a relative path is a valid path.
//!
//! Carrying them together makes that unspellable: the fields are named at every call site and
//! there is no order left to get wrong. It lives here rather than beside any one engine
//! because the same pair travels through image, audio, speech and text alike.

/// What to load, and where the weights live.
///
/// These travelled as two adjacent `String` parameters, and the loaders below disagreed about
/// their order - three took the name first, two the directory. One pair straddled both, so a
/// model name reached the download root and thirteen gigabytes landed in a directory named
/// after the model, beside the copy the configuration already pointed at. Nothing failed,
/// because a relative path is a valid path.
///
/// Carrying them together is what makes that unspellable: the fields are named at every call
/// site, and there is no order left to get wrong.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// The directory the configuration resolved - the root of the weight store, never a model.
    pub models_dir: String,
    /// The model as the caller asked for it, which may be a tag, a repo id or a local name.
    pub name: String,
}

impl ModelRequest {
    pub fn new(models_dir: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            models_dir: models_dir.into(),
            name: name.into(),
        }
    }
}
