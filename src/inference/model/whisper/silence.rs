//! The cut of a recording longer than 1 encoder window into windows that end at silences.
//!
//! The level of a mel frame is the mean of its log-mel bands.
//! The silence level of a recording splits its frame levels into a quiet class and a loud
//! class, the split that maximises the variance between the 2 classes.
//! A frame below that level is silent.
//!
//! A window that does not reach the end of the recording ends inside a silent run.
//! The run is the one whose length, times the frames the window keeps up to its middle, is the
//! largest.
//! The window ends at the middle of that run.

/// The level of each of the first `content` frames of the row-major `[n_mels, frames]` log-mel
/// spectrogram `mel`: the mean of its bands.
pub fn frame_levels(mel: &[f32], n_mels: usize, frames: usize, content: usize) -> Vec<f32> {
    let content = content.min(frames);
    let mut levels = vec![0f32; content];
    if n_mels == 0 {
        return levels;
    }
    for band in mel.chunks(frames).take(n_mels) {
        for (level, value) in levels.iter_mut().zip(band) {
            *level += value;
        }
    }
    levels.iter_mut().for_each(|l| *l /= n_mels as f32);
    levels
}

/// The level that splits `levels` into the 2 classes with the largest variance between them.
///
/// A frame below the answer is silent.
/// None when the levels hold fewer than 2 distinct values.
pub fn silence_level(levels: &[f32]) -> Option<f32> {
    let mut sorted: Vec<f64> = levels.iter().map(|&l| l as f64).collect();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let total: f64 = sorted.iter().sum();
    let mut below = 0f64;
    let mut best: Option<(f64, usize)> = None;
    for k in 1..n {
        below += sorted[k - 1];
        if sorted[k - 1] == sorted[k] {
            continue;
        }
        let (w0, w1) = (k as f64, (n - k) as f64);
        let gap = below / w0 - (total - below) / w1;
        let between = w0 * w1 * gap * gap;
        if best.is_none_or(|(b, _)| between > b) {
            best = Some((between, k));
        }
    }
    best.map(|(_, k)| ((sorted[k - 1] + sorted[k]) / 2.0) as f32)
}

/// The frames the window that starts at frame `seek` keeps, at most `window`.
///
/// A window that reaches the last frame keeps every frame up to it.
/// Any other window ends inside a silent run after the speech it opens on.
/// The silent run the window opens on belongs to the cut before it.
/// A window without such a run keeps `window` frames.
pub fn window_length(levels: &[f32], silence: Option<f32>, seek: usize, window: usize) -> usize {
    let end = seek + window;
    if end >= levels.len() {
        return levels.len().saturating_sub(seek);
    }
    let Some(level) = silence else {
        return window;
    };
    let silent = |f: usize| levels[f] < level;
    let mut lead = seek;
    while lead < end && silent(lead) {
        lead += 1;
    }
    let mut best: Option<(usize, usize)> = None;
    let mut f = lead;
    while f <= end {
        if !silent(f) {
            f += 1;
            continue;
        }
        let first = f;
        while f <= end && silent(f) {
            f += 1;
        }
        // The run reaches the window's end: the window keeps every frame.
        if f > end {
            return window;
        }
        let middle = (first + f) / 2;
        let score = (f - first) * (middle - seek);
        if best.is_none_or(|(s, _)| score >= s) {
            best = Some((score, middle));
        }
    }
    best.map_or(window, |(_, middle)| middle - seek)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The levels of a synthetic recording: a tone, a silence, a tone, a silence, a tone.
    ///
    /// Each frame is 10 ms of 16 kHz audio.
    /// The levels are read off a log-mel spectrogram of the waveform.
    fn tone_silence_tone(spans: &[(usize, bool)]) -> Vec<f32> {
        use crate::inference::model::whisper::{audio, HOP_LENGTH, N_FFT, SAMPLE_RATE};
        let samples: Vec<f32> = spans
            .iter()
            .flat_map(|&(frames, tone)| {
                (0..frames * HOP_LENGTH).map(move |i| match tone {
                    true => {
                        let t = i as f32 / SAMPLE_RATE as f32;
                        0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    }
                    false => 0.0,
                })
            })
            .collect();
        let n_mels = 8usize;
        let bins = N_FFT / 2 + 1;
        let mut filters = vec![0f32; n_mels * bins];
        for band in 0..n_mels {
            for bin in band * bins / n_mels..(band + 1) * bins / n_mels {
                filters[band * bins + bin] = 1.0;
            }
        }
        let mel = audio::log_mel_spectrogram_(&samples, &filters, N_FFT, HOP_LENGTH, n_mels);
        let frames = mel.len() / n_mels;
        frame_levels(&mel, n_mels, frames, samples.len().div_ceil(HOP_LENGTH))
    }

    /// A window over tone, silence, tone ends inside the silence.
    #[test]
    fn a_window_ends_inside_the_silence_between_two_tones() {
        // 2 s of tone, 0.5 s of silence, 2 s of tone; the window holds 3 s.
        let levels = tone_silence_tone(&[(200, true), (50, false), (200, true)]);
        let level = silence_level(&levels).expect("tone and silence differ");
        assert!(levels[100] > level && levels[225] < level);
        let kept = window_length(&levels, Some(level), 0, 300);
        assert!(
            (205..245).contains(&kept),
            "the cut at frame {kept} is inside the silence"
        );
        // The next window opens on the rest of the silence and reaches the end.
        assert_eq!(
            window_length(&levels, Some(level), kept, 300),
            levels.len() - kept
        );
    }

    /// Of 2 silences, the window ends in the one that weighs more: its length times the frames
    /// it leaves.
    #[test]
    fn a_window_prefers_the_longer_silence_unless_it_wastes_the_window() {
        let mut levels = vec![1.0f32; 400];
        levels[100..140].iter_mut().for_each(|l| *l = 0.0);
        levels[280..284].iter_mut().for_each(|l| *l = 0.0);
        let level = silence_level(&levels);
        // 40 frames ending near 120 outweigh 4 frames near 282.
        assert_eq!(window_length(&levels, level, 0, 300), 120);
        // A window that opens on a silence keeps it for the cut before.
        assert_eq!(window_length(&levels, level, 110, 200), 282 - 110);
    }

    /// A window without silence, or over a recording of 1 level, keeps every frame it may.
    #[test]
    fn a_window_without_silence_keeps_its_full_length() {
        let flat = vec![0.5f32; 500];
        assert_eq!(silence_level(&flat), None);
        assert_eq!(window_length(&flat, None, 0, 300), 300);
        assert_eq!(window_length(&flat, None, 300, 300), 200);
        let mut loud_end = vec![1.0f32; 500];
        loud_end[..10].iter_mut().for_each(|l| *l = 0.0);
        let level = silence_level(&loud_end);
        assert_eq!(window_length(&loud_end, level, 0, 300), 300);
    }
}
