//! The time of each word of a transcript.
//!
//! The decoder's cross attention, read on the heads the checkpoint names, says which audio
//! position each text token looks at.
//! Dynamic time warping over those weights gives a monotonic path from the first token to the
//! last, and a word starts where the path reaches its first token.
//!
//! A checkpoint that names no head times its words from its timestamp tokens instead.
//! Each segment's span is shared among its words by their token counts.

/// A word of a transcript and the seconds it spans.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

/// The width, in audio positions, of the median filter the reference alignment smooths each
/// head's weights with.
pub const MEDIAN_WIDTH: usize = 7;

/// `[rows, cols]` softmax of the row-major `scores`, each row over its first `cols` columns
/// of `stride`.
pub fn softmax_rows(scores: &[f32], rows: usize, stride: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let row = &scores[r * stride..r * stride + cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        out.extend(exps.iter().map(|&e| e / sum));
    }
    out
}

/// The median of each run of `width` values of `row` centred on each position.
///
/// The row is mirrored at its ends, the edge value not repeated.
/// A row no longer than half the width is kept as it is.
fn median_filter(row: &[f32], width: usize) -> Vec<f32> {
    let half = width / 2;
    let n = row.len();
    if half == 0 || n <= half {
        return row.to_vec();
    }
    let mirror = |i: isize| -> f32 {
        let last = n as isize - 1;
        let j = match i {
            i if i < 0 => -i,
            i if i > last => 2 * last - i,
            i => i,
        };
        row[j as usize]
    };
    let mut window = Vec::with_capacity(width);
    (0..n as isize)
        .map(|c| {
            window.clear();
            window.extend((c - half as isize..=c + half as isize).map(mirror));
            window.sort_by(f32::total_cmp);
            window[half]
        })
        .collect()
}

/// The alignment matrix of a sequence: `[tokens, positions]`, row-major.
///
/// `heads` holds 1 weight matrix per alignment head, `[tokens, positions]` row-major, each row a
/// softmax over the audio positions.
/// Each head is standardised over the tokens at each position, then smoothed over the positions
/// by a median filter of `median_width`.
/// The matrix is the mean of the heads.
pub fn alignment_matrix(
    heads: &[Vec<f32>],
    tokens: usize,
    positions: usize,
    median_width: usize,
) -> Vec<f32> {
    let mut matrix = vec![0f32; tokens * positions];
    if heads.is_empty() || tokens == 0 {
        return matrix;
    }
    for head in heads {
        let mut standard = head.clone();
        for p in 0..positions {
            let column = (0..tokens).map(|t| head[t * positions + p]);
            let mean = column.clone().sum::<f32>() / tokens as f32;
            let var = column.map(|x| (x - mean) * (x - mean)).sum::<f32>() / tokens as f32;
            let std = var.sqrt();
            for t in 0..tokens {
                let x = head[t * positions + p] - mean;
                standard[t * positions + p] = if std > 0.0 { x / std } else { 0.0 };
            }
        }
        for t in 0..tokens {
            let row = &standard[t * positions..(t + 1) * positions];
            let smooth = median_filter(row, median_width);
            for (m, v) in matrix[t * positions..(t + 1) * positions]
                .iter_mut()
                .zip(smooth)
            {
                *m += v;
            }
        }
    }
    let n = heads.len() as f32;
    matrix.iter_mut().for_each(|m| *m /= n);
    matrix
}

/// The cheapest monotonic path through the row-major `[rows, cols]` `cost`, from its first
/// cell to its last.
///
/// The path is returned as the row and the column of each step, in order.
/// A step moves down, right or both at once.
/// A tie between moves is broken towards moving right.
pub fn dtw(cost: &[f32], rows: usize, cols: usize) -> (Vec<usize>, Vec<usize>) {
    if rows == 0 || cols == 0 {
        return (Vec::new(), Vec::new());
    }
    let w = cols + 1;
    let mut acc = vec![f32::INFINITY; (rows + 1) * w];
    // 0 moves diagonally, 1 moves down, 2 moves right.
    let mut trace = vec![2u8; (rows + 1) * w];
    acc[0] = 0.0;
    for j in 1..=cols {
        for i in 1..=rows {
            let c0 = acc[(i - 1) * w + j - 1];
            let c1 = acc[(i - 1) * w + j];
            let c2 = acc[i * w + j - 1];
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            acc[i * w + j] = cost[(i - 1) * cols + j - 1] + c;
            trace[i * w + j] = t;
        }
    }
    for j in 0..=cols {
        trace[j] = 2;
    }
    for i in 0..=rows {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (rows, cols);
    let (mut path_rows, mut path_cols) = (Vec::new(), Vec::new());
    while i > 0 || j > 0 {
        path_rows.push(i.saturating_sub(1));
        path_cols.push(j.saturating_sub(1));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path_rows.reverse();
    path_cols.reverse();
    (path_rows, path_cols)
}

/// The audio position at which a path through an alignment matrix reaches each of its rows.
///
/// The path is the one [`dtw`] returns.
/// The answer holds 1 position per row.
pub fn row_starts(path_rows: &[usize], path_cols: &[usize]) -> Vec<usize> {
    path_rows
        .iter()
        .zip(path_cols)
        .enumerate()
        .filter(|(k, (r, _))| *k == 0 || path_rows[k - 1] != **r)
        .map(|(_, (_, &c))| c)
        .collect()
}

/// The words of `tokens` and the tokens each spans, in order.
///
/// `decode` turns tokens into text.
/// The tokens are first cut into the shortest runs that decode to whole characters.
/// When `spaced`, a run that opens with white space opens a new word and any other run joins the
/// word before it.
/// Otherwise each run is a word.
pub fn split_words(
    tokens: &[u32],
    decode: impl Fn(&[u32]) -> String,
    spaced: bool,
) -> Vec<(String, Vec<u32>)> {
    const REPLACEMENT: char = '\u{fffd}';
    let full: Vec<char> = decode(tokens).chars().collect();
    let mut runs: Vec<(String, Vec<u32>)> = Vec::new();
    let mut current: Vec<u32> = Vec::new();
    let mut offset = 0usize;
    for &token in tokens {
        current.push(token);
        let decoded = decode(&current);
        let whole = match decoded.chars().position(|c| c == REPLACEMENT) {
            None => true,
            Some(at) => full.get(offset + at) == Some(&REPLACEMENT),
        };
        if whole {
            offset += decoded.chars().count();
            runs.push((decoded, std::mem::take(&mut current)));
        }
    }
    if !current.is_empty() {
        runs.push((decode(&current), current));
    }
    if !spaced {
        return runs;
    }
    let mut words: Vec<(String, Vec<u32>)> = Vec::new();
    for (text, ids) in runs {
        match words.last_mut() {
            Some((word, word_ids)) if !text.starts_with(char::is_whitespace) => {
                word.push_str(&text);
                word_ids.extend(ids);
            }
            _ => words.push((text, ids)),
        }
    }
    words
}

/// Whether the words of `text` are separated by white space.
///
/// The scripts that write no space between words are told by their Unicode blocks: Han,
/// Hiragana, Katakana, Thai, Lao, Khmer and Myanmar.
/// Any other text is spaced, a single word included.
pub fn is_spaced(text: &str) -> bool {
    let unspaced = |c: char| {
        matches!(c as u32,
            0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0x3040..=0x309F | 0x30A0..=0x30FF
            | 0x0E00..=0x0E7F | 0x0E80..=0x0EFF | 0x1780..=0x17FF | 0x1000..=0x109F)
    };
    !text.chars().any(unspaced)
}

/// The times of `words` from the positions at which the alignment path reaches each token.
///
/// `starts` holds 1 position per token of the words, then 1 for the token that closes the
/// transcript.
/// A word spans from the position of its first token to the position of the token after its
/// last.
/// A position lasts `position_seconds`.
pub fn timed_by_alignment(
    words: &[(String, Vec<u32>)],
    starts: &[usize],
    position_seconds: f32,
) -> Vec<TimedWord> {
    let mut out = Vec::with_capacity(words.len());
    let mut at = 0usize;
    for (word, ids) in words {
        let next = at + ids.len();
        let (Some(&from), Some(&to)) = (starts.get(at), starts.get(next)) else {
            break;
        };
        out.push(TimedWord {
            word: word.trim().to_string(),
            start: from as f32 * position_seconds,
            end: to.max(from) as f32 * position_seconds,
        });
        at = next;
    }
    out
}

/// The times of `words` from the span of the segment that holds them: `start` to `end`
/// seconds, shared among the words by their token counts.
pub fn timed_by_tokens(words: &[(String, Vec<u32>)], start: f32, end: f32) -> Vec<TimedWord> {
    let total: usize = words.iter().map(|(_, ids)| ids.len()).sum();
    if total == 0 {
        return Vec::new();
    }
    let span = (end - start).max(0.0);
    let at = |tokens: usize| start + span * tokens as f32 / total as f32;
    let mut before = 0usize;
    words
        .iter()
        .map(|(word, ids)| {
            let from = at(before);
            before += ids.len();
            TimedWord {
                word: word.trim().to_string(),
                start: from,
                end: at(before),
            }
        })
        .collect()
}

/// Append to `all` the words of a window that starts `offset` seconds into the recording.
///
/// The words of the window carry times from the window's start.
/// They are moved to times from the recording's start.
/// Each time is held between the end of the word before it and `end`, the second the
/// window's audio ends.
/// A word without text is dropped.
pub fn stitch_words(all: &mut Vec<TimedWord>, window: Vec<TimedWord>, offset: f32, end: f32) {
    for word in window {
        if word.word.is_empty() {
            continue;
        }
        let floor = all.last().map_or(offset, |w| w.end.max(offset)).min(end);
        let start = (word.start + offset).clamp(floor, end);
        let stop = (word.end + offset).clamp(start, end);
        all.push(TimedWord {
            word: word.word,
            start,
            end: stop,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path through a matrix whose cheap cells form a staircase follows the staircase.
    #[test]
    fn the_warping_path_follows_the_cheap_cells() {
        // 3 tokens over 6 positions: each token costs its distance to its own centre.
        // Token 0 is centred between positions 0 and 1, token 1 between 2 and 3, token 2
        // between 4 and 5.
        let (rows, cols) = (3usize, 6usize);
        let cost: Vec<f32> = (0..rows)
            .flat_map(|t| (0..cols).map(move |p| (p as f32 - (2 * t) as f32 - 0.5).abs()))
            .collect();
        let (path_rows, path_cols) = dtw(&cost, rows, cols);
        assert_eq!(path_rows.first(), Some(&0));
        assert_eq!(path_cols.first(), Some(&0));
        assert_eq!(path_rows.last(), Some(&(rows - 1)));
        assert_eq!(path_cols.last(), Some(&(cols - 1)));
        for k in 1..path_rows.len() {
            assert!(path_rows[k] >= path_rows[k - 1] && path_cols[k] >= path_cols[k - 1]);
        }
        assert_eq!(row_starts(&path_rows, &path_cols), vec![0, 2, 4]);
    }

    /// The alignment of synthetic attention places each word where its tokens look.
    #[test]
    fn synthetic_attention_times_each_word_where_it_looks() {
        // 2 words of 1 token, then the closing token, over 9 positions of 0.02 s.
        // Each token looks hardest at its own position and less the further a position is.
        let (tokens, positions) = (3usize, 9usize);
        let looks = [1.0f32, 4.0, 7.0];
        let scores: Vec<f32> = (0..tokens)
            .flat_map(|t| (0..positions).map(move |p| -(p as f32 - looks[t]).powi(2)))
            .collect();
        let head = softmax_rows(&scores, tokens, positions, positions);
        let matrix = alignment_matrix(&[head.clone(), head], tokens, positions, 1);
        let cost: Vec<f32> = matrix.iter().map(|m| -m).collect();
        let (path_rows, path_cols) = dtw(&cost, tokens, positions);
        let starts = row_starts(&path_rows, &path_cols);
        assert_eq!(starts, vec![0, 3, 6]);
        let words = vec![
            (" hello".to_string(), vec![1]),
            (" world".to_string(), vec![2]),
        ];
        let timed = timed_by_alignment(&words, &starts, 0.02);
        assert_eq!(timed.len(), 2);
        assert_eq!(timed[0].word, "hello");
        assert!((timed[0].start - 0.0).abs() < 1e-6 && (timed[0].end - 0.06).abs() < 1e-6);
        assert!((timed[1].start - 0.06).abs() < 1e-6 && (timed[1].end - 0.12).abs() < 1e-6);
    }

    /// The median filter mirrors the row at its ends and removes a lone spike.
    #[test]
    fn the_median_filter_removes_a_lone_spike() {
        let row = [0.0, 0.0, 9.0, 0.0, 0.0];
        assert_eq!(median_filter(&row, 3), vec![0.0; 5]);
        let ramp = [1.0, 2.0, 3.0];
        assert_eq!(median_filter(&ramp, 3), vec![2.0, 2.0, 2.0]);
        assert_eq!(median_filter(&ramp, 7), ramp.to_vec());
    }

    /// Words open at white space, and a token that ends inside a character waits for the rest.
    #[test]
    fn words_open_at_white_space_and_keep_whole_characters() {
        let pieces = [" Hel", "lo", ",", " w", "\u{fffd}", "\u{fffd}rld"];
        // Tokens 4 and 5 together decode to "ö": each alone decodes to a replacement.
        let decode = |ids: &[u32]| -> String {
            let mut out = String::new();
            let mut i = 0;
            while i < ids.len() {
                if ids[i] == 4 && ids.get(i + 1) == Some(&5) {
                    out.push_str("örld");
                    i += 2;
                } else {
                    out.push_str(pieces[ids[i] as usize]);
                    i += 1;
                }
            }
            out
        };
        let words = split_words(&[0, 1, 2, 3, 4, 5], decode, true);
        let texts: Vec<&str> = words.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(texts, vec![" Hello,", " wörld"]);
        assert_eq!(words[1].1, vec![3, 4, 5]);
        let runs = split_words(&[0, 1, 2], decode, false);
        assert_eq!(runs.len(), 3);
        assert!(is_spaced(" Hello, world."));
        assert!(is_spaced(" Unbelievable."));
        assert!(!is_spaced(" こんにちは"));
        assert!(!is_spaced(" 你好世界"));
    }

    /// A segment's span is shared among its words by their token counts.
    #[test]
    fn a_segment_span_is_shared_by_token_count() {
        let words = vec![
            (" one".to_string(), vec![1]),
            (" three".to_string(), vec![2, 3, 4]),
        ];
        let timed = timed_by_tokens(&words, 1.0, 3.0);
        assert_eq!(timed[0].word, "one");
        assert!((timed[0].start - 1.0).abs() < 1e-6 && (timed[0].end - 1.5).abs() < 1e-6);
        assert!((timed[1].start - 1.5).abs() < 1e-6 && (timed[1].end - 3.0).abs() < 1e-6);
    }

    /// The words of 2 windows land at their times in the recording, in order, inside each
    /// window's audio.
    #[test]
    fn the_words_of_two_windows_stitch_in_recording_time() {
        let word = |w: &str, start: f32, end: f32| TimedWord {
            word: w.to_string(),
            start,
            end,
        };
        let mut all = Vec::new();
        stitch_words(
            &mut all,
            vec![word("a", 0.5, 1.0), word("b", 1.0, 29.0)],
            0.0,
            27.4,
        );
        // The second window opens at 27.4 s, and its first word overlaps the last one.
        stitch_words(
            &mut all,
            vec![
                word("c", -0.2, 0.4),
                word("", 0.4, 0.5),
                word("d", 0.6, 5.0),
            ],
            27.4,
            30.0,
        );
        let got: Vec<(&str, f32, f32)> = all
            .iter()
            .map(|w| (w.word.as_str(), w.start, w.end))
            .collect();
        let want = [
            ("a", 0.5, 1.0),
            ("b", 1.0, 27.4),
            ("c", 27.4, 27.8),
            ("d", 28.0, 30.0),
        ];
        assert_eq!(got.len(), want.len());
        for ((gw, gs, ge), (ww, ws, we)) in got.iter().zip(want) {
            assert_eq!(*gw, ww);
            assert!((gs - ws).abs() < 1e-4 && (ge - we).abs() < 1e-4, "{got:?}");
        }
    }
}
