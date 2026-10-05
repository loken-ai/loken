//! The Voxtral tower against the reference at toy size.
//!
//! The vectors hold a seeded checkpoint in the reference's own layout and what the
//! reference computes from it: encoder frames and projected embeddings. The tower is read
//! from that checkpoint and must agree at rounding level.

use crate::inference::model::audio_in::gate::{compare, tensor, vectors};
use crate::inference::model::audio_in::AudioTower;
use crate::tensor::Device;

/// Agreement as a fraction of the reference's largest magnitude.
const TOLERANCE: f32 = 2e-4;

#[test]
fn the_encoder_and_projector_match_the_reference() {
    let dir = vectors("voxtral");
    let tower = AudioTower::from_checkpoint(&dir, &Device::Cpu).unwrap();
    let mel = tensor(&dir, "mel");
    let mel = mel.unsqueeze(0).unwrap();
    let frames = tower.encode_window(&mel).unwrap();
    let (diff, scale) = compare(&frames, &tensor(&dir, "frames"));
    assert!(
        diff <= TOLERANCE * scale,
        "encoder frames: {diff} off, scale {scale}"
    );
    let embeds = tower.project(&frames).unwrap();
    let (diff, scale) = compare(&embeds, &tensor(&dir, "embeds"));
    assert!(
        diff <= TOLERANCE * scale,
        "embeddings: {diff} off, scale {scale}"
    );
}

/// A clip is padded to whole windows: one second of audio fills one window, and the count the
/// KV window is checked against is the count the tower yields.
#[test]
fn a_short_clip_fills_a_whole_window() {
    let dir = vectors("voxtral");
    let tower = AudioTower::from_checkpoint(&dir, &Device::Cpu).unwrap();
    let clip: Vec<f32> = (0..tower.sample_rate())
        .map(|i| ((i * 7 % 101) as f32) / 101.0 - 0.5)
        .collect();
    let embeds = tower.embed(&clip).unwrap();
    assert_eq!(embeds.dim(1).unwrap(), tower.embeds_for(clip.len()));
    assert_eq!(embeds.dims(), [1, 375, tower.embedding_width()]);
    let longer = vec![0.1f32; 31 * tower.sample_rate()];
    assert_eq!(tower.embeds_for(longer.len()), 2 * 375);
}
