//! Reading a video clip into the frames a vision model samples from it.
//!
//! ffprobe counts the clip's frames and reads its rate, ffmpeg decodes it to RGB, and the frames
//! the sampling keeps are held; both must be on the PATH. Frames are spread evenly over the clip
//! at a rate the model sets, within a floor and a ceiling on their number, and grouped in runs
//! of the model's temporal patch, each run stamped with the mean time of its frames.

use std::io::Read;
use std::process::{Command, Stdio};

/// How a model samples a clip.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    /// Frames kept per second of the clip.
    pub fps: f64,
    /// The fewest and the most frames kept.
    pub min_frames: usize,
    pub max_frames: usize,
    /// Frames one temporal patch takes.
    pub temporal: usize,
}

/// The frames sampled from a clip, in order, with their indices in the clip.
pub struct Clip {
    pub frames: Vec<image::RgbImage>,
    pub indices: Vec<usize>,
    /// The clip's frame rate.
    pub fps: f64,
}

impl Clip {
    /// The time of each temporal patch of the clip (see [`patch_times`]).
    pub fn patch_times(&self, temporal: usize) -> Vec<f64> {
        patch_times(&self.indices, self.fps, temporal)
    }
}

/// The time of each temporal patch of the frames `indices` of a clip at `fps`, in seconds: the
/// mean of its first and last frames' times. A last patch short of frames repeats the clip's
/// last frame, as its pixels do.
pub fn patch_times(indices: &[usize], fps: f64, temporal: usize) -> Vec<f64> {
    let t = temporal.max(1);
    let mut idx = indices.to_vec();
    while !idx.len().is_multiple_of(t) {
        idx.push(*idx.last().unwrap_or(&0));
    }
    idx.chunks(t)
        .map(|c| (c[0] as f64 + c[t - 1] as f64) / 2.0 / fps)
        .collect()
}

/// The indices of the frames kept from a clip of `total` frames at `clip_fps`: `fps` per second,
/// within the floor and the ceiling and never more than the clip holds, spread evenly from the
/// first frame to the last.
pub fn sample_indices(total: usize, clip_fps: f64, s: &Sampling) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    let by_rate = (total as f64 / clip_fps * s.fps) as usize;
    let n = by_rate
        .max(s.min_frames)
        .min(s.max_frames)
        .min(total)
        .max(1);
    if n == 1 {
        return vec![0];
    }
    let last = (total - 1) as f64;
    (0..n)
        .map(|i| (i as f64 * last / (n - 1) as f64).round_ties_even() as usize)
        .collect()
}

/// The frames `s` keeps from the clip in `bytes`, in any container ffmpeg reads.
pub fn read_clip(bytes: &[u8], s: &Sampling) -> Result<Clip, String> {
    with_clip_file(bytes, |path| read_clip_file(path, s))
}

/// The indices of the frames `s` keeps from the clip in `bytes`, and the clip's frame rate,
/// without decoding it.
pub fn probe_clip(bytes: &[u8], s: &Sampling) -> Result<(Vec<usize>, f64), String> {
    with_clip_file(bytes, |path| {
        let probe = probe_file(path)?;
        Ok((sample_indices(probe.frames, probe.fps, s), probe.fps))
    })
}

/// `f` over the clip in `bytes` written to a file: a container may keep its index at the end,
/// so ffmpeg reads the clip from a file rather than a pipe.
fn with_clip_file<T>(
    bytes: &[u8],
    f: impl FnOnce(&std::path::Path) -> Result<T, String>,
) -> Result<T, String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("loken_clip_{}_{nanos}", std::process::id()));
    std::fs::write(&path, bytes).map_err(|e| format!("clip: {e}"))?;
    let out = f(&path);
    let _ = std::fs::remove_file(&path);
    out
}

/// What ffprobe says of a clip's first video stream.
struct Probe {
    width: usize,
    height: usize,
    fps: f64,
    frames: usize,
}

fn probe_file(path: &std::path::Path) -> Result<Probe, String> {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_packets",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("ffprobe: {e} (ffmpeg must be on the PATH to read a video)"))?;
    if !probe.status.success() {
        return Err(format!(
            "ffprobe: {}",
            String::from_utf8_lossy(&probe.stderr).trim()
        ));
    }
    let text = String::from_utf8_lossy(&probe.stdout);
    let field = |k: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .map(str::trim)
    };
    let dim = |k: &str| -> Result<usize, String> {
        field(k)
            .and_then(|v| v.parse().ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| format!("the clip states no {k}"))
    };
    let fps = field("avg_frame_rate")
        .and_then(|r| {
            let (n, d) = r.split_once('/')?;
            let (n, d) = (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?);
            (n > 0.0 && d > 0.0).then_some(n / d)
        })
        .ok_or("the clip states no frame rate")?;
    Ok(Probe {
        width: dim("width")?,
        height: dim("height")?,
        fps,
        frames: dim("nb_read_packets")?,
    })
}

fn read_clip_file(path: &std::path::Path, s: &Sampling) -> Result<Clip, String> {
    let Probe {
        width: w,
        height: h,
        fps,
        frames: total,
    } = probe_file(path)?;
    let indices = sample_indices(total, fps, s);

    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let mut out = child.stdout.take().ok_or("ffmpeg: no output")?;
    let mut frames = Vec::with_capacity(indices.len());
    let mut kept = Vec::with_capacity(indices.len());
    let mut buf = vec![0u8; w * h * 3];
    let mut wanted = indices.iter().peekable();
    let mut n = 0usize;
    while wanted.peek().is_some() {
        if out.read_exact(&mut buf).is_err() {
            break;
        }
        while wanted.peek() == Some(&&n) {
            let frame = image::RgbImage::from_raw(w as u32, h as u32, buf.clone())
                .ok_or("ffmpeg: a frame of the wrong size")?;
            frames.push(frame);
            kept.push(n);
            wanted.next();
        }
        n += 1;
    }
    drop(out);
    let _ = child.kill();
    let _ = child.wait();
    if frames.is_empty() {
        return Err("the clip holds no frame ffmpeg could decode".into());
    }
    Ok(Clip {
        frames,
        indices: kept,
        fps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: Sampling = Sampling {
        fps: 2.0,
        min_frames: 4,
        max_frames: 768,
        temporal: 2,
    };

    /// The frames a 4-second, 24-frame-per-second clip keeps, as the reference keeps them.
    #[test]
    fn a_clip_keeps_its_frames_at_the_model_rate() {
        assert_eq!(
            sample_indices(96, 24.0, &QWEN),
            vec![0, 14, 27, 41, 54, 68, 81, 95]
        );
    }

    /// A short clip keeps the floor, never more frames than it holds.
    #[test]
    fn a_short_clip_keeps_the_floor_and_no_more_than_it_has() {
        assert_eq!(sample_indices(24, 24.0, &QWEN).len(), 4);
        assert_eq!(sample_indices(3, 24.0, &QWEN), vec![0, 1, 2]);
    }

    /// Each patch is stamped with the mean time of its frames; a short last patch repeats the
    /// last frame.
    #[test]
    fn each_patch_takes_the_mean_time_of_its_frames() {
        let clip = Clip {
            frames: Vec::new(),
            indices: vec![0, 14, 27, 41, 54, 68, 81, 95],
            fps: 24.0,
        };
        let times: Vec<String> = clip
            .patch_times(2)
            .iter()
            .map(|t| format!("{t:.1}"))
            .collect();
        assert_eq!(times, ["0.3", "1.4", "2.5", "3.7"]);
        let short = Clip {
            frames: Vec::new(),
            indices: vec![0, 24, 48],
            fps: 24.0,
        };
        assert_eq!(short.patch_times(2), vec![0.5, 2.0]);
    }
}
