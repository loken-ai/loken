//! What each image family acts on of a request, and the refusal of what it does not.
//!
//! A parameter a family has no use for is refused before anything runs: a render that
//! silently drops a LoRA, a negative prompt or a mask looks like a render that honoured
//! it, and the caller has no way to tell.

use super::*;

/// The optional request parameters a family acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaCapabilities {
    pub negative_prompt: bool,
    pub loras: bool,
    pub sampler: bool,
    pub scheduler: bool,
    pub control_image: bool,
    pub step_reuse: bool,
    pub input_image: bool,
    pub mask: bool,
    pub regions: bool,
}

/// A request parameter the resident family cannot honour.
///
/// Returned unwrapped inside the engine error, so a caller can downcast to it and answer
/// with a client error rather than a server one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedParameter {
    pub family: &'static str,
    pub parameter: &'static str,
}

impl std::fmt::Display for UnsupportedParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the {} image family does not support `{}` - retry without it, or with a \
             model that does",
            self.family, self.parameter
        )
    }
}

impl std::error::Error for UnsupportedParameter {}

/// Box an engine error for the caller, keeping an [`UnsupportedParameter`] as itself so it
/// can be downcast; anyhow's own boxing hides the concrete type.
pub(super) fn boxed_engine_error(e: anyhow::Error) -> Box<dyn std::error::Error + Send + Sync> {
    match e.downcast::<UnsupportedParameter>() {
        Ok(refusal) => Box::new(refusal),
        Err(e) => e.into(),
    }
}

impl MediaCapabilities {
    const NONE: Self = Self {
        negative_prompt: false,
        loras: false,
        sampler: false,
        scheduler: false,
        control_image: false,
        step_reuse: false,
        input_image: false,
        mask: false,
        regions: false,
    };

    /// What the loaded family's render path reads. Exhaustive, so a new family cannot be
    /// added without stating what it honours.
    pub fn of(model: &LoadedImageModel) -> Self {
        match model {
            LoadedImageModel::Flux(_) => Self {
                loras: true,
                step_reuse: true,
                input_image: true,
                mask: true,
                regions: true,
                ..Self::NONE
            },
            LoadedImageModel::ZImage(_) => Self {
                input_image: true,
                ..Self::NONE
            },
            LoadedImageModel::QwenImage(_) => Self {
                negative_prompt: true,
                step_reuse: true,
                input_image: true,
                ..Self::NONE
            },
            LoadedImageModel::Flux2(_) | LoadedImageModel::Boogu(_) => Self::NONE,
            LoadedImageModel::Sdxl(_) => Self {
                negative_prompt: true,
                loras: true,
                sampler: true,
                scheduler: true,
                control_image: true,
                input_image: true,
                ..Self::NONE
            },
        }
    }

    /// The first parameter `params` asks for that this family does not honour.
    ///
    /// A blank negative prompt asks for nothing - it is what a family without one does
    /// anyway - and a step-reuse threshold of zero is the exact render.
    pub fn refuse_unsupported(
        &self,
        family: &'static str,
        params: &ImageGenParams,
    ) -> Result<(), UnsupportedParameter> {
        let asked = [
            (
                "negative_prompt",
                self.negative_prompt,
                params
                    .negative_prompt
                    .as_deref()
                    .is_some_and(|s| !s.trim().is_empty()),
            ),
            ("loras", self.loras, !params.loras.is_empty()),
            ("sampler", self.sampler, params.sampler.is_some()),
            ("scheduler", self.scheduler, params.scheduler.is_some()),
            (
                "control_image",
                self.control_image,
                params.control_image.is_some(),
            ),
            ("step_reuse", self.step_reuse, params.step_reuse > 0.0),
            ("image", self.input_image, params.input_image.is_some()),
            ("mask", self.mask, params.mask.is_some()),
            ("regions", self.regions, !params.regions.is_empty()),
        ];
        match asked
            .into_iter()
            .find(|&(_, honoured, requested)| requested && !honoured)
        {
            Some((parameter, ..)) => Err(UnsupportedParameter { family, parameter }),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SDXL: MediaCapabilities = MediaCapabilities {
        negative_prompt: true,
        loras: true,
        sampler: true,
        scheduler: true,
        control_image: true,
        step_reuse: false,
        input_image: true,
        mask: false,
        regions: false,
    };

    #[test]
    fn a_plain_request_is_honoured_by_every_family() {
        let plain = ImageGenParams::default();
        assert!(MediaCapabilities::NONE
            .refuse_unsupported("flux2", &plain)
            .is_ok());
    }

    #[test]
    fn a_parameter_the_family_ignores_is_refused_by_name() {
        let with_lora = ImageGenParams {
            loras: vec![("style.safetensors".into(), 1.0)],
            ..Default::default()
        };
        assert_eq!(
            MediaCapabilities::NONE.refuse_unsupported("boogu", &with_lora),
            Err(UnsupportedParameter {
                family: "boogu",
                parameter: "loras"
            })
        );
        assert!(SDXL.refuse_unsupported("sdxl", &with_lora).is_ok());

        let with_reuse = ImageGenParams {
            step_reuse: 0.05,
            ..Default::default()
        };
        assert_eq!(
            SDXL.refuse_unsupported("sdxl", &with_reuse)
                .map_err(|e| e.parameter),
            Err("step_reuse")
        );
    }

    #[test]
    fn a_blank_negative_prompt_asks_for_nothing() {
        let blank = ImageGenParams {
            negative_prompt: Some("  ".into()),
            ..Default::default()
        };
        assert!(MediaCapabilities::NONE
            .refuse_unsupported("flux2", &blank)
            .is_ok());
        let real = ImageGenParams {
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(MediaCapabilities::NONE
            .refuse_unsupported("flux2", &real)
            .is_err());
    }

    /// The handlers receive engine errors boxed; the refusal must survive that trip
    /// intact so they can map it to a client error.
    #[test]
    fn the_refusal_can_be_recovered_from_the_boxed_engine_error() {
        let e = anyhow::Error::new(UnsupportedParameter {
            family: "zimage",
            parameter: "mask",
        });
        let boxed = boxed_engine_error(e);
        let got = boxed
            .downcast_ref::<UnsupportedParameter>()
            .expect("downcast");
        assert_eq!(got.parameter, "mask");
    }
}
