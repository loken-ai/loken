//! The Ultravox tower against the reference at toy size.
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
    let dir = vectors("ultravox");
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

/// A clip is padded to a whole hop only: the embeddings follow its length, and the count the
/// KV window is checked against is the count the tower yields.
#[test]
fn the_embeddings_follow_the_length_of_the_clip() {
    let dir = vectors("ultravox");
    let tower = AudioTower::from_checkpoint(&dir, &Device::Cpu).unwrap();
    let rate = tower.sample_rate();
    let clip: Vec<f32> = (0..13 * rate / 10)
        .map(|i| ((i * 7 % 101) as f32) / 101.0 - 0.5)
        .collect();
    let embeds = tower.embed(&clip).unwrap();
    // 1.3 s is 130 frames, 65 encoder frames, stacked by 4 with the last group padded.
    assert_eq!(embeds.dims(), [1, 17, tower.embedding_width()]);
    assert_eq!(tower.embeds_for(clip.len()), 17);
    // A clip over the window continues in a second, shorter one.
    let longer = vec![0.1f32; 31 * rate];
    // 3100 frames: a full window of 1500 encoder frames, then 100 frames of 50 encoder frames.
    assert_eq!(
        tower.embeds_for(longer.len()),
        1500usize.div_ceil(4) + 50usize.div_ceil(4)
    );
}
