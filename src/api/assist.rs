//! Optional collaborators for the media endpoints.
//!
//! Each one is absent by default. A handler that finds nothing here does its own work and
//! nothing else happens - so every endpoint below reads the same on a server that has none
//! of them, which is the ordinary case.

use std::sync::{Arc, OnceLock, RwLock};

/// A rectangle of an image, in fractions of its width and height, so it survives a resize.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Who chose the protected rectangle, because the two answers deserve different treatment.
#[derive(Clone, Copy, Debug)]
pub enum Chosen {
    /// The caller named it. Compositing it is doing as asked, wherever the edit has moved
    /// things since.
    ByCaller,
    /// It was derived from the source image. It describes where something WAS, which only
    /// stays true while the edit leaves the composition alone.
    ByAssist,
}

/// Reads the source image of an edit.
///
/// An edit re-renders the whole frame, so anything the instruction did not ask to change
/// can still drift. On its own the endpoint offers the plain contract: the caller names a
/// rectangle and gets it composited back.
pub trait EditAssist: Send + Sync {
    /// Turn a region spec into a rectangle. `spec` is whatever the caller wrote where
    /// coordinates were expected; the endpoint itself reads only `x,y,w,h`.
    fn resolve_spec(&self, spec: &str, source: &image::RgbImage) -> Result<Region, String>;

    /// A clause describing what the source shows, folded into an instruction that names no
    /// subject. `None` leaves the instruction exactly as written.
    ///
    /// Never fails a request: the instruction going through as written is what happened
    /// before any of this existed.
    fn subject_clause(&self, _source: &image::RgbImage) -> Option<String> {
        None
    }

    /// Put the protected region of `source` back into the rendered `b64` image.
    ///
    /// Returns the new base64 PNG. The default composites the rectangle, which is what the
    /// server does on its own; an assist overrides it to do something that survives the
    /// render having moved.
    fn restore(
        &self,
        b64: &str,
        source: &image::RgbImage,
        region: Region,
        chosen: Chosen,
    ) -> Result<String, String>;
}

/// Reads a generative request for conditioning of its own.
///
/// Handed the raw body, it returns an opaque token that the image and video paths carry
/// back to it. Nothing here inspects that token, and no field name it may read is written
/// down in this crate.
pub trait RequestConditioner: Send + Sync {
    /// Does this request carry conditioning this implementation understands? The returned
    /// value is passed to [`Self::apply_to_frame`] for every image produced.
    ///
    /// `Err` refuses the request - the caller asked for something that could not be
    /// prepared, and rendering without it would silently ignore what they asked for.
    fn prepare(&self, req: &serde_json::Value) -> Result<Option<Conditioning>, String>;

    /// Apply prepared conditioning to one rendered image, given as base64 PNG.
    ///
    /// Used by both the still-image and the video path; video calls it once per frame.
    fn apply_to_frame(&self, prepared: &Conditioning, b64: &str) -> Result<String, String>;

    /// Apply prepared conditioning to a batch of raw RGB frames, in place.
    ///
    /// Returns how many frames were changed. The video path calls this once per clip:
    /// frames come out of the decoder as raw bytes, and round-tripping each one through
    /// PNG only to hand it back would cost more than the conditioning does.
    ///
    /// A frame it cannot handle must be LEFT ALONE rather than failed: a clip is real work
    /// already done, and throwing it away over one frame is worse than the frame.
    fn apply_to_frames(
        &self,
        _prepared: &Conditioning,
        _frames: &mut [Vec<u8>],
        _width: usize,
        _height: usize,
    ) -> Result<usize, String> {
        Ok(0)
    }

    /// Something the caller should know about what the conditioning did, folded into the
    /// response. `None` says nothing.
    ///
    /// Exists because the caller cannot see any of this: a request whose conditioning
    /// quietly did less than it was asked looks exactly like one that did all of it.
    fn note(&self, _prepared: &Conditioning) -> Option<String> {
        None
    }
}

/// An opaque handle to prepared conditioning: stored and forwarded, never opened.
#[derive(Clone)]
pub struct Conditioning(Arc<dyn std::any::Any + Send + Sync>);

impl Conditioning {
    /// Wrap a value. Called by the implementation that will later unwrap it.
    pub fn new<T: Send + Sync + 'static>(v: T) -> Self {
        Self(Arc::new(v))
    }

    /// Recover the value, or `None` if it was not of that type.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for Conditioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Conditioning(..)")
    }
}

/// Reworks synthesized speech for a request the speech backends cannot serve alone.
pub trait SpeechPostprocessor: Send + Sync {
    /// Which base voice to synthesize with, when this request is one it handles.
    /// `Ok(None)` leaves the normal voice selection alone.
    fn base_voice(
        &self,
        req: &serde_json::Value,
        detected_language: &str,
    ) -> Result<Option<String>, String>;

    /// Transform the synthesized audio. Returns the new samples and their sample rate.
    fn apply(
        &self,
        req: &serde_json::Value,
        pcm: Vec<f32>,
        sample_rate: u32,
    ) -> futures::future::BoxFuture<'static, Result<(Vec<f32>, u32), String>>;

    /// What to report back as the voice that was used.
    fn label(&self) -> String {
        "postprocessed".to_string()
    }
}

/// A media kind a client can render a form for without knowing what it is.
///
/// A client knows its own panels by name and cannot know a server's, so a panel that is
/// not built in describes itself and the client builds the form from the description. The
/// model catalogue already works this way one level down: `capabilities` on a model is
/// what the pickers are built from.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KindDescriptor {
    /// Stable identifier. A client that DOES know this kind renders its own panel.
    pub id: String,
    /// Short name for the picker.
    pub label: String,
    /// One line explaining what it does, for a tooltip.
    pub tip: String,
    /// Whether the kind takes a text prompt: `"none"`, `"optional"` or `"required"`.
    pub prompt: String,
    /// Files the caller supplies.
    pub inputs: Vec<KindInput>,
    /// Scalars and choices.
    #[serde(default)]
    pub controls: Vec<KindControl>,
    /// Where to send it.
    pub submit: KindSubmit,
    /// What comes back.
    pub output: KindOutput,
}

/// A file the caller supplies. `kind` is `"image"`, `"image[]"` or `"audio"`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KindInput {
    pub id: String,
    pub label: String,
    pub kind: String,
    #[serde(default)]
    pub required: bool,
    /// For a list input, the fewest entries that make the request valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<usize>,
}

/// A scalar or a choice. `kind` is `"int"`, `"float"`, `"bool"`, `"enum"` or `"string"`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KindControl {
    pub id: String,
    pub label: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
    /// `(wire value, label)` pairs, for an enum.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<(String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

/// Where the form posts. `encoding` is `"json"` or `"multipart"`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KindSubmit {
    pub path: String,
    pub encoding: String,
}

/// What comes back. `kind` is `"image"`, `"audio"` or `"video"`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KindOutput {
    pub kind: String,
    /// Numbers worth showing next to the result, read out of the response by `id`.
    ///
    /// This is what lets a client display a measurement it has no concept of: the label
    /// and the format travel with the number, so the panel says what it means without the
    /// client knowing what was measured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics: Vec<KindMetric>,
}

/// One number in a response, named and formatted for display.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KindMetric {
    pub id: String,
    pub label: String,
    /// A pattern like `"0.00"`, so the client need not guess a precision.
    pub format: String,
}

static DECLARED_KINDS: OnceLock<RwLock<Vec<KindDescriptor>>> = OnceLock::new();

macro_rules! registry {
    ($(#[$doc:meta])* $slot:ident, $reg:ident, $get:ident, $t:ident) => {
        static $slot: OnceLock<RwLock<Option<Arc<dyn $t>>>> = OnceLock::new();

        $(#[$doc])*
        pub fn $reg(v: Arc<dyn $t>) {
            let cell = $slot.get_or_init(|| RwLock::new(None));
            *cell.write().unwrap_or_else(|e| e.into_inner()) = Some(v);
        }

        /// What is set, if anything.
        pub fn $get() -> Option<Arc<dyn $t>> {
            $slot
                .get()?
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
    };
}

registry! {
    /// Set the assist consulted by `POST /v1/images/edits`. Replaces any previous one.
    EDIT_ASSIST, set_edit_assist, edit_assist, EditAssist
}

registry! {
    /// Set the conditioner consulted by the image and video paths.
    CONDITIONER, set_request_conditioner, request_conditioner, RequestConditioner
}

registry! {
    /// Set the post-processor consulted by the speech path.
    SPEECH_POSTPROCESSOR, set_speech_postprocessor, speech_postprocessor, SpeechPostprocessor
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Nothing;
    impl EditAssist for Nothing {
        fn resolve_spec(&self, _: &str, _: &image::RgbImage) -> Result<Region, String> {
            Err("no".into())
        }
        fn restore(
            &self,
            b: &str,
            _: &image::RgbImage,
            _: Region,
            _: Chosen,
        ) -> Result<String, String> {
            Ok(b.to_string())
        }
    }

    /// Absent is the DEFAULT, not an error state: a server that sets none of these must
    /// see `None` and take its own path.
    #[test]
    fn an_unset_collaborator_is_absent() {
        assert!(request_conditioner().is_none());
        assert!(speech_postprocessor().is_none());
    }

    #[test]
    fn one_that_is_set_is_found_and_replaced() {
        assert!(edit_assist().is_none());
        set_edit_assist(Arc::new(Nothing));
        assert!(edit_assist().is_some());
        // Setting again REPLACES rather than accumulating: two assists disagreeing about
        // a region would make the outcome depend on the order they arrived in.
        set_edit_assist(Arc::new(Nothing));
        assert!(edit_assist().is_some());
    }

    /// The opaque handle must survive the round trip, and must not hand back a value of a
    /// type it was never given - a mistyped downcast returning something plausible would
    /// send one implementation's conditioning into another's.
    #[test]
    fn conditioning_round_trips_but_only_for_its_own_type() {
        let c = Conditioning::new(vec![1u8, 2, 3]);
        assert_eq!(c.get::<Vec<u8>>().map(|v| v.len()), Some(3));
        assert!(c.get::<String>().is_none());
    }
}
