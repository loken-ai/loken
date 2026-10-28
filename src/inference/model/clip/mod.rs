//! CLIP: the text encoder FLUX pools, the ViT vision tower, and the pair of them that embeds
//! texts and images in one space (CLIP's and SigLIP's layout).
//!
//! `crate::inference::model::clip::<part>`

pub mod pair;
pub mod preprocess;
pub mod text;
pub mod vision;
