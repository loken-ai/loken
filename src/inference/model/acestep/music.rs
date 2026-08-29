//! ACE-Step 1.5 turbo text->music pipeline - shared render used by the acestep_render CLI
//!
//! Reads a TOML prompt file at RUNTIME (caption + lyrics + params, and optionally a list of
//! `[[styles]]` sections) and renders a 48 kHz stereo WAV. Edit the file and re-run - NO
//! recompile. Models resolve via `config.test.toml` (dev-tool convention; no hardcoded paths,
//! no env vars). The prompt path is a CLI argument (argv, not an env var).
//!
//! Usage:   acestep_render [prompt.toml]        (default: ./acestep_prompt.toml)
//!
//! Style MORPH: with several `[[styles]]`, the styles transition fluidly WITHIN one track  - 
//! `generate_cfg_morph` emits ONE continuous autoregressive code stream whose conditioning
//! caption changes between sections (each section re-prefills [new caption + the codes so far]
//! and continues), so the model keeps its own musical line while drifting the style. That single
//! stream is rendered by ONE detok->DiT->VAE pass. (Not a concatenation/cross-fade of clips.)

use crate::inference::model::acestep::cond::CondModel;
use crate::inference::model::acestep::dit::{DitModel, XattnMorph};
use crate::inference::model::acestep::fsq::{acestep_gguf, DetokModel, TokEncoder};
use crate::inference::model::acestep::lm::{acestep_tokenizer, build_cot_yaml, Qwen3Lm};
use crate::inference::model::acestep::pipeline::{
    build_context, cond_emb_apply, dit_latent_cfg, CfgType, Dcw, DcwMode, DitSample, DualCond,
    RepaintMask, Solver,
};
use crate::inference::model::acestep::textenc::TextEncoder;
use crate::inference::model::acestep::vae::{
    decode_wav_s16le, encode_wav_s16le, OobleckDecoder, OobleckEncoder,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct MusicConfig {
    /// enc-conditioning caption (overall/medley guidance; the per-region style is carried by the
    /// morph codes). Falls back to the joined per-style captions when empty.
    #[serde(default)]
    caption: String,
    /// Lyrics for a single-style track (per-style lyrics override).
    #[serde(default)]
    lyrics: String,
    #[serde(default = "d_bpm")]
    bpm: i32,
    #[serde(default = "d_seed")]
    seed: u64,
    #[serde(default = "d_cfg")]
    cfg_scale: f32,
    #[serde(default = "d_temp")]
    temperature: f32,
    #[serde(default = "d_topp")]
    top_p: f32,
    /// LM audio-code top-k cutoff (applied before top-p). 0 (default) = disabled, matching the
    /// reference. A positive value restricts sampling to the k most likely codes.
    #[serde(default)]
    top_k: usize,
    #[serde(default = "d_lang")]
    language: String,
    #[serde(default = "d_ts")]
    time_signature: String,
    /// Musical key/scale directive (e.g. "C minor", "A major", "D dorian"). Empty = let the
    /// model choose freely (can drift / sound off-key, especially in the intro). Setting it
    /// pins the tonality so the generated notes stay in one key across the whole track + all
    /// morph sections. Fed to BOTH the LM CoT (note generation) and the DiT conditioning.
    #[serde(default)]
    keyscale: String,
    /// Negative prompt for classifier-free guidance (the unconditional pass): generation is
    /// pushed AWAY from this. The robust way to FORBID a trait - caption negation ("no talking")
    /// is weakly followed by the model; put what you DON'T want here instead (e.g. "spoken word,
    /// talking, rap, monotone, speech"). Empty (default) = the plain empty-caption uncond.
    #[serde(default)]
    negative_prompt: String,
    /// Codes emitted per style section before swapping caption (≈ section seconds x 5 Hz).
    #[serde(default = "d_cps")]
    codes_per_section: usize,
    /// Minimum codes the LM must emit before EOS is allowed (a length floor). 0 (default)
    /// = let the LM stop the song naturally. A positive value bans EOS until reached, forcing
    /// the track toward the full requested duration even when the LM would end early - use it
    /// to guarantee a fixed length (and to stress-test the model past its natural song length).
    #[serde(default)]
    min_codes: usize,
    #[serde(default = "d_out")]
    out: String,
    /// DiT checkpoint GGUF (resolved via `acestep_gguf`). Default = the validated turbo
    /// model. Set to "acestep-v15-sft-Q8_0.gguf" (higher-quality 2B 50-step base) or
    /// "acestep-v15-xl-sft-Q8_0.gguf" (4B XL) for the non-turbo checkpoints.
    #[serde(default = "d_dit_gguf")]
    dit_gguf: String,
    /// DiT flow-matching Euler steps. Turbo is distilled to 8; the base/sft checkpoints
    /// want ~50. Default 8 (turbo).
    #[serde(default = "d_dit_steps")]
    dit_steps: usize,
    /// DiT classifier-free guidance scale. 1.0 = OFF (turbo, distilled CFG-free). The
    /// non-turbo checkpoints want CFG ON (~4.0-7.0): each Euler step then evaluates the
    /// velocity for the conditional AND a null (empty caption/lyric) encoding and blends
    /// `v = v_uncond + cfg.(v_cond - v_uncond)`.
    #[serde(default = "d_dit_cfg")]
    dit_cfg: f32,
    /// Flow-matching timestep shift. 0 (default) = auto by checkpoint (turbo 3.0, base/sft
    /// 1.0 - the reference values). Set explicitly to override. Only matters with the DiT.
    #[serde(default)]
    dit_shift: f32,
    /// DiT guidance algorithm (only when `dit_cfg > 1.0`): "apg" (default, ACE-Step's
    /// reference - decouples magnitude from direction so a strong scale doesn't saturate),
    /// "cfg" (vanilla `v = u + s.(c-u)`), or "cfg_star" (optimal-scale blend + zero-init).
    #[serde(default = "d_cfg_type")]
    cfg_type: String,
    /// Fraction of DiT steps (centred window) where guidance is applied. Reference 0.5
    /// (middle half) - guidance off in the first/last quarter keeps the intro/outro clean.
    #[serde(default = "d_gi")]
    guidance_interval: f32,
    /// >0 linearly decays the guidance scale `dit_cfg`->`min_guidance_scale` across the
    /// window. 0 (default) = constant `dit_cfg` inside the window.
    #[serde(default)]
    guidance_interval_decay: f32,
    /// Floor the guidance decay reaches at the end of the window (reference 3.0).
    #[serde(default = "d_mgs")]
    min_guidance_scale: f32,
    /// cfg_star only: zero the prediction for steps `i <= cfg_zero_steps` (reference 1).
    #[serde(default = "d_zero_steps")]
    cfg_zero_steps: usize,
    /// Scheduler mean-shift strength. 0 (default) = neutral (preserves the turbo path).
    /// The reference default is 10 (sharpens features); set it on the non-turbo configs.
    #[serde(default)]
    omega_scale: f32,
    /// Dual-condition guidance: when BOTH > 0, replace single-scale CFG with a 3-way blend
    /// `(1-gt).uncond + (gt-gl).text-only + gl.cond` (gt=text, gl=lyric) - lets you weight
    /// how much the model follows the caption vs the lyrics separately. 0/0 (default) = off.
    #[serde(default)]
    guidance_scale_text: f32,
    #[serde(default)]
    guidance_scale_lyric: f32,
    /// ODE/SDE solver: "euler" (default, fast on-device path), "heun" (2nd order, ~2x DiT
    /// time, sharper), or "pingpong" (stochastic SDE). Non-euler use the generalized loop.
    #[serde(default = "d_solver")]
    solver: String,
    /// DCW wavelet-domain SNR-bias correction (CVPR 2026), applied after each solver step. 0
    /// (default) = off. `dcw_mode`: "low" (default, strong at high noise), "high", "double"
    /// (uses dcw_high_scaler for the high band), "pix" (no wavelet). Skipped for pingpong.
    #[serde(default)]
    dcw_scaler: f32,
    #[serde(default)]
    dcw_high_scaler: f32,
    #[serde(default = "d_dcw_mode")]
    dcw_mode: String,
    /// Explicit descending sigma schedule (CSV, e.g. "0.97,0.76,0.61,...,0"); overrides
    /// `dit_steps`/`dit_shift`. A trailing 0 (the x0 target) is dropped. Empty = auto.
    #[serde(default)]
    custom_timesteps: String,
    /// Post-DiT latent tuning before VAE decode: `latent = latent.rescale + shift`. Defaults
    /// 1.0 / 0.0 (identity).
    #[serde(default = "d_one")]
    latent_rescale: f32,
    #[serde(default)]
    latent_shift: f32,
    /// Output loudness handling. -1 (default) = the safety peak limiter (scale down only if it
    /// would clip). >=0 = loudness-maximizing percentile normalization (ACE-Step `peak_clip`):
    /// 0 = peak-normalize, 10 = clip top 0.001% (the reference default), up to 999.
    #[serde(default = "d_peak_clip")]
    peak_clip: i32,
    /// Task mode (all the non-text2music ones need `input_audio`, which is FSQ-encoded to the
    /// semantic codes that drive the DiT - the LM is skipped):
    ///   "text2music" (default) . "cover" (recompose) . "lego" (generate a `track` stem) .
    ///   "extract" (isolate a `track` stem) . "complete" (complete with `track`).
    /// Stem tasks need a stem-trained checkpoint to work well. (audio2audio/repaint/extend are
    /// selected by `input_audio` + their own fields, independently of this.)
    #[serde(default = "d_task")]
    task: String,
    /// Stem name for lego/extract/complete (e.g. "vocals", "drums", "bass", "guitar"). Empty =
    /// the trackless instruction variant.
    #[serde(default)]
    track: String,
    /// DiT cross-attention instruction line (the task verb). Empty (default) = "Fill the audio
    /// semantic mask based on the given conditions:" (text2music); auto-switches to the repaint
    /// instruction in repaint/extend mode. Set explicitly for cover/stem instructions, e.g.
    /// "Generate the vocals track based on the audio context:".
    #[serde(default)]
    dit_instruction: String,
    /// audio2audio: path to a reference WAV (16-bit PCM). When set, the track is generated as
    /// a VARIATION of this audio (SDEdit) - its encoded latent seeds the DiT trajectory and the
    /// output length follows the reference. Empty (default) = plain text2music.
    #[serde(default)]
    input_audio: String,
    /// audio2audio strength in [0,1]: how far to push away from the reference. 0 ≈ keep the
    /// reference, 1 ≈ ignore it (full re-generation). Sets sigma_max = 1 - strength.
    #[serde(default = "d_refstr")]
    ref_audio_strength: f32,
    /// retake: a second seed whose init noise is blended with `seed`'s as
    /// `cos(v).noise(seed) + sin(v).noise(retake_seed)`, v = retake_variance.π/2 - a controlled
    /// variation of the same song. 0 (default) = off. Ignored in audio2audio mode.
    #[serde(default)]
    retake_seed: u64,
    /// retake blend amount in [0,1]: 0 = identical to `seed`, 1 = fully `retake_seed`.
    #[serde(default)]
    retake_variance: f32,
    /// repaint/inpaint (needs `input_audio`): regenerate only the region [repaint_start,
    /// repaint_end] seconds while keeping the rest of the reference. -1 (default) = off.
    #[serde(default = "d_neg1")]
    repaint_start: f32,
    #[serde(default = "d_neg1")]
    repaint_end: f32,
    /// extend/outpaint (needs `input_audio`): keep the whole reference and generate this many
    /// extra seconds appended after it. 0 (default) = off. Takes precedence over repaint_*.
    #[serde(default)]
    extend_seconds: f32,
    /// LoRA adapter: path to an `adapter_model.safetensors` (ACE-Step 1.5). Applied as a
    /// low-rank delta on the DiT attention projections at forward. Empty (default) = none.
    /// ⚠️ the adapter's hidden dim must match the checkpoint (2B adapters need a 2B `dit_gguf`).
    #[serde(default)]
    adapter: String,
    /// LoRA strength (x the adapter's alpha/rank, assumed 1 when it ships no config). 1.0 default.
    #[serde(default = "d_one")]
    adapter_scale: f32,
    /// Style sections - empty = one plain track from the top-level caption/lyrics.
    #[serde(default)]
    styles: Vec<Style>,
}

#[derive(Deserialize, Clone)]
struct Style {
    caption: String,
    #[serde(default = "d_bpm")]
    bpm: i32,
    #[serde(default)]
    tag: String,
    #[serde(default)]
    lyrics: String,
}

fn d_bpm() -> i32 {
    120
}
fn d_seed() -> u64 {
    7
}
fn d_cfg() -> f32 {
    2.0
}
fn d_temp() -> f32 {
    0.85
}
fn d_topp() -> f32 {
    0.9
}
fn d_lang() -> String {
    "en".into()
}
fn d_ts() -> String {
    "4".into()
}
fn d_cps() -> usize {
    120
}
fn d_out() -> String {
    "results/acestep/lyrics_edm.wav".into()
}
fn d_dit_gguf() -> String {
    "acestep-v15-turbo-Q8_0.gguf".into()
}
fn d_dit_steps() -> usize {
    8
}
fn d_dit_cfg() -> f32 {
    1.0
}
fn d_cfg_type() -> String {
    "apg".into()
}
fn d_gi() -> f32 {
    0.5
}
fn d_mgs() -> f32 {
    3.0
}
fn d_zero_steps() -> usize {
    1
}
fn d_one() -> f32 {
    1.0
}
fn d_peak_clip() -> i32 {
    -1
}
fn d_solver() -> String {
    "euler".into()
}
fn d_refstr() -> f32 {
    0.5
}
fn d_neg1() -> f32 {
    -1.0
}
fn d_task() -> String {
    "text2music".into()
}
fn d_dcw_mode() -> String {
    "low".into()
}

/// Trim the trailing sub-threshold tail (the VAE's residual noise floor exposed after the
/// music ends) and apply a short cosine fade-out so the track ends at exactly zero. Input
/// and output are planar `[c.t]` (channel-major, the decoder/encoder layout). Returns the
/// possibly-shortened buffer + its new frame count.
fn finalize_audio(
    planar: &[f32],
    c: usize,
    t: usize,
    sr: usize,
    peak_clip: i32,
) -> (Vec<f32>, usize) {
    if c == 0 || t == 0 {
        return (planar.to_vec(), t);
    }
    let amp = |ti: usize| {
        (0..c)
            .map(|ci| planar[ci * t + ti].abs())
            .fold(0f32, f32::max)
    };
    // Last frame above ~-48 dBFS; keep a short guard for any natural decay. Trailing-only
    // (scans from the end), so quiet passages mid-track are untouched.
    let thresh = 0.004f32;
    let mut end = t;
    while end > 0 && amp(end - 1) < thresh {
        end -= 1;
    }
    let guard = (0.03 * sr as f32) as usize;
    let nt = if end == 0 { t } else { (end + guard).min(t) }; // all-quiet -> keep as-is
    let fade = ((0.08 * sr as f32) as usize).min(nt);
    let mut out = vec![0f32; c * nt];
    for ci in 0..c {
        for ti in 0..nt {
            let mut s = planar[ci * t + ti];
            if fade > 0 && ti + fade > nt {
                let x = (nt - ti) as f32 / fade as f32; // 1 at fade start -> ~0 at the end
                s *= 0.5 * (1.0 - (std::f32::consts::PI * x).cos());
            }
            out[ci * nt + ti] = s;
        }
    }
    if peak_clip >= 0 {
        // Loudness-maximizing percentile normalization (ACE-Step `peak_clip`, acestep.cpp
        // audio-io.h): scale so the target percentile (1 - peak_clip/1e6) hits full scale,
        // hard-clipping whatever is above. peak_clip 0 = peak-normalize (no clip); 10 = clip
        // the top 0.001%. Opt-in via the render config.
        let pc = peak_clip.clamp(0, 999);
        let mut abs: Vec<f32> = out.iter().map(|s| s.abs()).collect();
        if !abs.is_empty() {
            let pct = 1.0 - pc as f64 / 1_000_000.0;
            let idx = (((abs.len() - 1) as f64) * pct) as usize;
            abs.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
            let r = abs[idx];
            if r >= 1e-6 {
                let gain = 1.0 / r;
                for s in out.iter_mut() {
                    *s = (*s * gain).clamp(-1.0, 1.0);
                }
            }
        }
    } else {
        // Default safety peak gain: with CFG enabled the model can decode hot - the summed
        // layers exceed full scale and would hard-clip (saturate, worst in the highs) at the
        // int16 write. Scale the whole buffer down (never up) so the peak sits just under full
        // scale, preserving the mix and its dynamics. A no-op for material already within range.
        let peak = out.iter().fold(0f32, |m, &s| m.max(s.abs()));
        let ceiling = 0.97f32;
        if peak > ceiling {
            let g = ceiling / peak;
            for s in out.iter_mut() {
                *s *= g;
            }
        }
    }
    (out, nt)
}

type DynErr = Box<dyn std::error::Error>;

/// Shared graceful-degradation backstop (extracted to the library so the ACE-Step and Wan
/// renders drive the SAME VRAM-degrade ladder): on a CUDA OOM it escalates the global
/// degradation level - more reserve / a more balanced multi-GPU split, CPU as a last resort
/// - and re-runs the unit (which reloads its models so the new placement takes effect).
use crate::inference::model::acestep::lm::oom_retry;

pub fn render(cfg: &MusicConfig) -> Result<(), DynErr> {
    render_with_progress(cfg, None)
}

/// Like [`render`], plus a per-DiT-step progress callback `progress(step, total)` for streaming UIs.
pub fn render_with_progress(
    cfg: &MusicConfig,
    progress: Option<&crate::inference::serve::progress::ProgressTryFn<'_>>,
) -> Result<(), DynErr> {
    use crate::inference::serve::progress as ph;
    // The checkpoints are gigabytes and load before a single step exists to count.
    ph::try_note(progress, ph::phase::LOAD_MODEL, 0, 0)?;
    let g = |n: &str| acestep_gguf(n).to_str().unwrap().to_string();
    let (lm_g, dit_g) = (g("acestep-5Hz-lm-4B-Q8_0.gguf"), g(&cfg.dit_gguf));
    println!(
        "[acestep_render] DiT checkpoint: {} (steps={}, cfg={})",
        cfg.dit_gguf, cfg.dit_steps, cfg.dit_cfg
    );
    let (emb_g, vae_g) = (g("Qwen3-Embedding-0.6B-Q8_0.gguf"), g("vae-BF16.gguf"));
    // keyscale: the DiT metas want a non-empty token ("N/A" when unset); the LM CoT wants the
    // raw value ("" = omit the line). A set `keyscale` pins the tonality for both.
    let (lang, ts) = (cfg.language.as_str(), cfg.time_signature.as_str());
    let ks = if cfg.keyscale.trim().is_empty() {
        "N/A"
    } else {
        cfg.keyscale.trim()
    };

    let sections: Vec<Style> = if cfg.styles.is_empty() {
        vec![Style {
            caption: cfg.caption.clone(),
            bpm: cfg.bpm,
            tag: String::new(),
            lyrics: cfg.lyrics.clone(),
        }]
    } else {
        cfg.styles.clone()
    };
    let t0 = std::time::Instant::now();

    let medley = if cfg.caption.trim().is_empty() {
        sections
            .iter()
            .map(|s| s.caption.as_str())
            .collect::<Vec<_>>()
            .join("; then ")
    } else {
        cfg.caption.clone()
    };
    let mut full_lyr = String::new();
    for s in &sections {
        if !s.tag.is_empty() {
            full_lyr.push_str(&s.tag);
            full_lyr.push('\n');
        }
        if !s.lyrics.trim().is_empty() {
            full_lyr.push_str(s.lyrics.trim());
            full_lyr.push('\n');
        }
    }
    let full_lyr = if full_lyr.trim().is_empty() {
        cfg.lyrics.trim().to_string()
    } else {
        full_lyr.trim().to_string()
    };
    let total_secs = (cfg.codes_per_section / 5).max(1) as i32 * sections.len() as i32;
    let lm_cot = build_cot_yaml(cfg.bpm, &medley, total_secs, cfg.keyscale.trim(), lang, ts);

    // audio2audio (SDEdit): encode the reference WAV up front so its latent length drives the
    // generation, and (later) seeds the DiT trajectory. The encoder runs on CPU (one-shot).
    let ref_latent_full: Option<(Vec<f32>, usize)> = if !cfg.input_audio.trim().is_empty() {
        let bytes = std::fs::read(&cfg.input_audio)
            .map_err(|e| -> DynErr { format!("input_audio `{}`: {e}", cfg.input_audio).into() })?;
        let (planar, ch, sr) = decode_wav_s16le(&bytes)?;
        let frames = planar.len() / ch.max(1);
        // The encoder expects stereo: duplicate mono, or take the first two channels.
        let stereo: Vec<f32> = if ch == 2 {
            planar
        } else {
            let mut s = vec![0f32; 2 * frames];
            for c in 0..2 {
                let src = c.min(ch - 1);
                for f in 0..frames {
                    s[c * frames + f] = planar[src * frames + f];
                }
            }
            s
        };
        println!(
            "[acestep_render] audio2audio: ref {ch}ch @ {sr}Hz, {frames} samples, strength {}",
            cfg.ref_audio_strength
        );
        let enc = OobleckEncoder::from_gguf(&vae_g, 1e-12)?;
        let (lat, _c, t_ref) = enc.encode_chunked(&stereo, 2, frames, 512, 64)?;
        println!("[acestep_render] audio2audio: ref latent {t_ref} frames");
        Some((lat, t_ref))
    } else {
        None
    };
    let task = cfg.task.to_ascii_lowercase();
    // Tasks that drive the DiT from the reference's FSQ codes (the LM is skipped): cover + stems.
    let source_ctx = matches!(task.as_str(), "cover" | "lego" | "extract" | "complete");
    // FSQ-encode the reference's VAE latent -> semantic codes. Transpose the VAE latent
    // channel-major [64.t] -> frame-major [t.64] for TokEncoder.
    let cover_codes: Option<Vec<u32>> = if source_ctx {
        let (full, t_ref) = ref_latent_full
            .as_ref()
            .ok_or_else(|| -> DynErr { format!("task=\"{task}\" requires input_audio").into() })?;
        let mut fm = vec![0f32; 64 * t_ref];
        for c in 0..64 {
            for ti in 0..*t_ref {
                fm[ti * 64 + c] = full[c * *t_ref + ti];
            }
        }
        let tok = TokEncoder::from_gguf(&dit_g)?;
        let codes = tok.encode(&fm, *t_ref, &vec![0f32; 5 * 64])?;
        println!(
            "[acestep_render] {task}: {} FSQ codes from the reference (LM skipped)",
            codes.len()
        );
        Some(codes)
    } else {
        None
    };
    // Output length follows the reference (+ any extend tail): codes = round(t_total / 5). Not for
    // cover (codes are set directly from the FSQ encode).
    let ref_t_ref: Option<usize> = ref_latent_full.as_ref().map(|(_, t)| *t);
    let forced_codes: Option<usize> = if source_ctx {
        None
    } else {
        ref_latent_full.as_ref().map(|(_, t_ref)| {
            let extra = (cfg.extend_seconds.max(0.0) * 25.0) as usize;
            ((t_ref + extra + 2) / 5).max(1)
        })
    };

    // 1) LM - ONE generation with the FULL (tagged) lyrics + a medley caption, so the lyrics are
    //    aligned and sung across the WHOLE track. (The caption-swap morph put lyrics only at the
    //    start - re-prefill breaks lyric alignment; here the LM stays a single coherent pass.) The
    //    style MORPH is done later in the DiT, not here. OOM-backstopped: a memory-pressure crash
    //    in load/decode reloads the LM with more reserve / on CPU and retries.
    let mut codes = if let Some(cc) = cover_codes {
        cc
    } else {
        oom_retry("LM", || -> Result<_, DynErr> {
            let lm_tok = acestep_tokenizer(&lm_g)?;
            let mut lm = Qwen3Lm::from_gguf(&lm_g)?;
            lm.top_k = cfg.top_k;
            // Batched CFG: ONE instance carries cond+uncond as a 2-row batch -> each weight read once.
            // audio2audio pins the length to the reference (max=min=forced_codes) so t matches t_ref.
            let max_codes = forced_codes.unwrap_or(cfg.codes_per_section * sections.len());
            let min_codes = forced_codes
                .unwrap_or(cfg.min_codes.min(max_codes))
                .min(max_codes);
            let codes = lm.generate_cfg_batched(
                &lm_tok,
                &medley,
                &full_lyr,
                &lm_cot,
                cfg.negative_prompt.trim(),
                max_codes,
                min_codes,
                cfg.seed,
                cfg.temperature,
                cfg.top_p,
                cfg.cfg_scale,
            )?;
            drop(lm);
            Ok(codes)
        })?
    };
    if codes.len() % 2 == 1 {
        codes.pop();
    } // DiT patchify pairs frames -> even count
    if codes.len() < 2 {
        return Err("no audio codes generated".into());
    }
    println!(
        "[acestep_render] {} codes ({}) in {:.1}s",
        codes.len(),
        if source_ctx {
            "FSQ-encoded reference"
        } else {
            "LM"
        },
        t0.elapsed().as_secs_f32()
    );
    let t = codes.len() * 5;
    // Place the reference latent into the generated length `t` (channel-major): the first
    // t_ref frames are the reference, anything beyond (extend tail) stays silent (0). Cover uses
    // the reference only for its FSQ codes (the context), NOT as an SDEdit init -> no ref_latent.
    let ref_latent: Option<Vec<f32>> = if source_ctx {
        None
    } else {
        ref_latent_full.as_ref().map(|(full, t_ref)| {
            let mut r = vec![0f32; 64 * t];
            for c in 0..64 {
                for ti in 0..t.min(*t_ref) {
                    r[c * t + ti] = full[c * t_ref + ti];
                }
            }
            r
        })
    };

    // 2) detok -> context (one continuous timeline). OOM-backstopped.
    let context = oom_retry("detok", || -> Result<_, DynErr> {
        let detok = DetokModel::from_gguf(&dit_g)?;
        Ok(build_context(&detok.decode(&codes)?, t, t, &[]))
    })?;

    // 3+4) Encoders + DiT: build per-style cross-attention encs, then the 8-step Euler latent.
    //    Grouped into ONE OOM-backstopped stage because the DiT model is shared by the enc
    //    application and the trajectory - a memory-pressure crash reloads the encoders + DiT
    //    (with more reserve / on CPU) and recomputes this whole stage. Numerically identical on
    //    the no-pressure path (deterministic, seed-driven).
    let mut latent = oom_retry("DiT", || -> Result<_, DynErr> {
        // Per-style encs for the TIME-VARYING DiT cross-attention. The lyrics (and timbre) are the
        // SAME for every style (global); only the CAPTION differs -> each style's enc carries the
        // full lyrics + that style. cond_emb each, concatenate -> `enc` + per-style token `bounds`.
        let enc_tok = acestep_tokenizer(&emb_g)?;
        let dur = (t / 5).max(1) as i32;
        let lyric_str = format!("# Languages\n{lang}\n\n# Lyric\n{full_lyr}<|endoftext|>");
        let enc_ids = |x: &str| -> Result<Vec<u32>, DynErr> {
            Ok(enc_tok
                .encode(x, true)
                .map_err(|e| -> DynErr { e.to_string().into() })?
                .get_ids()
                .to_vec())
        };
        let lyric_ids = enc_ids(&lyric_str)?;
        let timbre: Vec<f32> = {
            use crate::tensor::{DType, Device};
            let mut f = std::fs::File::open(&dit_g)?;
            let c = crate::tensor::quantized::gguf_file::read_mapped_file(&f)?;
            c.tensor(&mut f, "silence_latent", &Device::Cpu)?
                .dequantize(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?[..64]
                .to_vec()
        };
        let te = TextEncoder::from_gguf(&emb_g)?;
        let lyric_embed = te.embed_lookup(&lyric_ids)?;
        let cond = CondModel::from_gguf(&dit_g)?;
        // Placed for THIS clip: the denoiser attends over the whole latent timeline, so
        // `t` is what decides whether a card can hold the render. A fixed reserve would be
        // the right answer for exactly one length.
        let mut dit = DitModel::from_gguf_for(&dit_g, t)?;
        if !cfg.adapter.trim().is_empty() {
            let n = dit.apply_lora(cfg.adapter.trim(), cfg.adapter_scale)?;
            println!(
                "[acestep_render] LoRA: {n} attn deltas from {} (scale {})",
                cfg.adapter, cfg.adapter_scale
            );
        }
        // DiT instruction verb (the task): explicit override, else the repaint instruction in
        // repaint/extend mode, else the default text2music fill. Matches acestep.cpp task-types.h.
        let repaint_active = ref_latent.is_some()
            && (cfg.extend_seconds > 0.0
                || (cfg.repaint_start >= 0.0 && cfg.repaint_end > cfg.repaint_start));
        // Task instruction (acestep.cpp task-types.h). Stem tasks take the optional `track` name.
        let trk = cfg.track.trim();
        let instr: String = if !cfg.dit_instruction.trim().is_empty() {
            cfg.dit_instruction.trim().to_string()
        } else if source_ctx {
            match task.as_str() {
                "lego" => {
                    if trk.is_empty() {
                        "Generate the track based on the audio context:".into()
                    } else {
                        format!("Generate the {trk} track based on the audio context:")
                    }
                }
                "extract" => {
                    if trk.is_empty() {
                        "Extract the track from the audio:".into()
                    } else {
                        format!("Extract the {trk} track from the audio:")
                    }
                }
                "complete" => {
                    if trk.is_empty() {
                        "Complete the input track:".into()
                    } else {
                        format!("Complete the input track with {trk}:")
                    }
                }
                _ => "Generate audio semantic tokens based on the given conditions:".into(), // cover
            }
        } else if repaint_active {
            "Repaint the mask area based on the given conditions:".to_string()
        } else {
            "Fill the audio semantic mask based on the given conditions:".to_string()
        };
        let (mut enc, mut enc_bounds): (Vec<f32>, Vec<usize>) = (Vec::new(), vec![0]);
        for s in &sections {
            let text_str = format!("# Instruction\n{instr}\n\n# Caption\n{}\n\n# Metas\n- bpm: {}\n- timesignature: {ts}\n- keyscale: {ks}\n- duration: {dur} seconds\n<|endoftext|>\n", s.caption, s.bpm);
            let text_ids = enc_ids(&text_str)?;
            let text_hidden = te.forward(&text_ids)?;
            let (enc_hidden, s_i) = cond.forward(
                &text_hidden,
                text_ids.len(),
                &lyric_embed,
                lyric_ids.len(),
                Some((&timbre, 1)),
            )?;
            let enc_i = cond_emb_apply(&dit, &enc_hidden, s_i)?;
            enc.extend_from_slice(&enc_i);
            enc_bounds.push(enc_bounds.last().unwrap() + s_i);
        }
        let s_total = *enc_bounds.last().unwrap();
        if sections.len() >= 2 {
            dit.xattn_morph = Some(XattnMorph { enc_bounds });
            println!(
                "[acestep_render] DiT time-varying style morph: {} styles, enc_S={s_total}",
                sections.len()
            );
        }

        // dual-condition guidance is active only when BOTH scales are set; it also needs the
        // uncond enc, so build that whenever dual is on (even if dit_cfg <= 1).
        let dual_active = cfg.guidance_scale_text > 0.0 && cfg.guidance_scale_lyric > 0.0;
        // Classifier-free guidance: the UNCONDITIONAL (null) enc = empty caption + empty
        // lyrics through the SAME encoders (mirrors the LM's cond/uncond CFG). Built when
        // dit_cfg > 1.0 (the non-turbo checkpoints) or for dual-condition; turbo leaves it None.
        let uncond_enc: Option<(Vec<f32>, usize)> = if cfg.dit_cfg > 1.0 || dual_active {
            let u_lyric_ids = enc_ids(&format!("# Languages\n{lang}\n\n# Lyric\n<|endoftext|>"))?;
            let u_lyric_embed = te.embed_lookup(&u_lyric_ids)?;
            let u_text_str = format!("# Instruction\n{instr}\n\n# Caption\n\n\n# Metas\n- bpm: {}\n- timesignature: {ts}\n- keyscale: {ks}\n- duration: {dur} seconds\n<|endoftext|>\n", cfg.bpm);
            let u_text_ids = enc_ids(&u_text_str)?;
            let u_text_hidden = te.forward(&u_text_ids)?;
            let (u_enc_hidden, u_s) = cond.forward(
                &u_text_hidden,
                u_text_ids.len(),
                &u_lyric_embed,
                u_lyric_ids.len(),
                Some((&timbre, 1)),
            )?;
            let u_enc = cond_emb_apply(&dit, &u_enc_hidden, u_s)?;
            println!(
                "[acestep_render] DiT CFG ON (scale {}): uncond enc_S={u_s}",
                cfg.dit_cfg
            );
            Some((u_enc, u_s))
        } else {
            None
        };

        // dual-condition: a TEXT-ONLY enc = the real caption (medley) with EMPTY lyrics. Single
        // (no per-style morph) - cross-attn handles its own S. Blended in the sampler with cond+uncond.
        let dual: Option<DualCond> = if dual_active {
            let tl_lyric_ids = enc_ids(&format!("# Languages\n{lang}\n\n# Lyric\n<|endoftext|>"))?;
            let tl_lyric_embed = te.embed_lookup(&tl_lyric_ids)?;
            let tl_text_str = format!("# Instruction\n{instr}\n\n# Caption\n{}\n\n# Metas\n- bpm: {}\n- timesignature: {ts}\n- keyscale: {ks}\n- duration: {dur} seconds\n<|endoftext|>\n", medley, cfg.bpm);
            let tl_text_ids = enc_ids(&tl_text_str)?;
            let tl_hidden = te.forward(&tl_text_ids)?;
            let (tl_enc_hidden, tl_s) = cond.forward(
                &tl_hidden,
                tl_text_ids.len(),
                &tl_lyric_embed,
                tl_lyric_ids.len(),
                Some((&timbre, 1)),
            )?;
            let tl_enc = cond_emb_apply(&dit, &tl_enc_hidden, tl_s)?;
            println!(
                "[acestep_render] dual-condition: gs_text={} gs_lyric={} (text-only enc_S={tl_s})",
                cfg.guidance_scale_text, cfg.guidance_scale_lyric
            );
            Some(DualCond {
                text_enc: tl_enc,
                text_s: tl_s,
                gs_text: cfg.guidance_scale_text,
                gs_lyric: cfg.guidance_scale_lyric,
            })
        } else {
            None
        };

        // N(0,1) noise -> Euler -> latent.
        let mut rng = cfg.seed.max(1);
        let mut u01 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((rng >> 33) as f64 / (1u64 << 31) as f64) as f32).clamp(1e-7, 1.0 - 1e-7)
        };
        let mut noise: Vec<f32> = (0..t * 64)
            .map(|_| {
                let (u1, u2) = (u01(), u01());
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect();
        // retake: blend a second seed's init noise (controlled variation; text2music only).
        if cfg.retake_variance > 0.0 && cfg.retake_seed > 0 && ref_latent.is_none() {
            let v = cfg.retake_variance.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
            let (cv, sv) = (v.cos(), v.sin());
            let mut r2 = cfg.retake_seed.max(1);
            let mut u = || {
                r2 = r2.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((r2 >> 33) as f64 / (1u64 << 31) as f64) as f32).clamp(1e-7, 1.0 - 1e-7)
            };
            for i in 0..noise.len() {
                let (u1, u2) = (u(), u());
                let r = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                noise[i] = cv * noise[i] + sv * r;
            }
            println!(
                "[acestep_render] retake: seed {} variance {} (blend {:.2})",
                cfg.retake_seed, cfg.retake_variance, sv
            );
        }
        let t1 = std::time::Instant::now();
        let uncond_ref = uncond_enc.as_ref().map(|(e, s)| (e.as_slice(), *s));
        // shift: 0 ⟹ auto (turbo 3.0, base/sft 1.0 - the reference per-checkpoint values).
        let shift = if cfg.dit_shift > 0.0 {
            cfg.dit_shift
        } else if cfg.dit_gguf.contains("turbo") {
            3.0
        } else {
            1.0
        };
        let extend_mode = ref_latent.is_some() && cfg.extend_seconds > 0.0;
        let repaint_mode = ref_latent.is_some()
            && (extend_mode || (cfg.repaint_start >= 0.0 && cfg.repaint_end > cfg.repaint_start));
        // audio2audio (SDEdit): seed x at σ_max from the reference latent and truncate the
        // schedule to start there. x = ref.(1-σ_max) + noise.σ_max; σ_max = 1 - strength.
        // Skipped in repaint mode, which keeps full noise + a full schedule and masks per step.
        let a2a_sched: Option<Vec<f32>> = match (ref_latent.as_ref(), repaint_mode) {
            (Some(rl), false) => {
                let sigma_max = (1.0 - cfg.ref_audio_strength).clamp(0.02, 1.0);
                for i in 0..noise.len() {
                    noise[i] = rl[i] * (1.0 - sigma_max) + noise[i] * sigma_max;
                }
                let steps_used = ((sigma_max * cfg.dit_steps as f32) as usize).max(1);
                let s: Vec<f32> = (0..steps_used)
                    .map(|i| {
                        let tl = sigma_max * (1.0 - i as f32 / steps_used as f32);
                        shift * tl / (1.0 + (shift - 1.0) * tl)
                    })
                    .collect();
                println!(
                    "[acestep_render] audio2audio: sigma_max={sigma_max:.3}, {steps_used} steps"
                );
                Some(s)
            }
            _ => None,
        };
        // repaint: keep frames outside [start,end] (latent frames @ 25 Hz), regenerate inside.
        let repaint = if repaint_mode {
            let rl = ref_latent.as_ref().unwrap();
            let keep: Vec<bool> = if extend_mode {
                // keep the whole reference, generate the appended tail.
                let tref = ref_t_ref.unwrap_or(0).min(t);
                println!(
                    "[acestep_render] extend: keep [0,{tref}) of {t}, outpaint the {} frame tail",
                    t - tref
                );
                (0..t).map(|ti| ti < tref).collect()
            } else {
                let sf = ((cfg.repaint_start * 25.0) as usize).min(t);
                let ef = ((cfg.repaint_end * 25.0) as usize).min(t);
                println!("[acestep_render] repaint: regenerate frames [{sf},{ef}) of {t}");
                (0..t).map(|ti| ti < sf || ti >= ef).collect()
            };
            Some(RepaintMask {
                ref_latent: rl.clone(),
                keep,
            })
        } else {
            None
        };
        let cfg_type = match cfg.cfg_type.to_ascii_lowercase().as_str() {
            "cfg" => CfgType::Cfg,
            "cfg_star" | "cfg-star" | "cfgstar" => CfgType::CfgStar,
            _ => CfgType::Apg,
        };
        let solver = match cfg.solver.to_ascii_lowercase().as_str() {
            "heun" => Solver::Heun,
            "pingpong" | "ping-pong" | "sde" => Solver::Pingpong,
            _ => Solver::Euler,
        };
        // custom_timesteps: explicit descending sigmas (drop a trailing x0≈0). Sets `steps`
        // to the schedule length so the guidance window stays consistent.
        // audio2audio's truncated schedule takes precedence; otherwise an explicit custom_timesteps.
        let custom_sched: Option<Vec<f32>> = a2a_sched.or_else(|| {
            let mut v: Vec<f32> = cfg
                .custom_timesteps
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();
            if v.last().is_some_and(|&x| x.abs() < 1e-6) {
                v.pop();
            }
            if v.len() >= 2 {
                Some(v)
            } else {
                None
            }
        });
        let steps = custom_sched.as_ref().map_or(cfg.dit_steps, |v| v.len());
        let dcw = if cfg.dcw_scaler > 0.0 || cfg.dcw_high_scaler > 0.0 {
            let mode = match cfg.dcw_mode.to_ascii_lowercase().as_str() {
                "high" => DcwMode::High,
                "double" => DcwMode::Double,
                "pix" => DcwMode::Pix,
                _ => DcwMode::Low,
            };
            println!(
                "[acestep_render] DCW: mode={mode:?} scaler={} high_scaler={}",
                cfg.dcw_scaler, cfg.dcw_high_scaler
            );
            Some(Dcw {
                mode,
                scaler: cfg.dcw_scaler,
                high_scaler: cfg.dcw_high_scaler,
            })
        } else {
            None
        };
        let sp = DitSample {
            steps,
            cfg: cfg.dit_cfg,
            shift,
            cfg_type,
            guidance_interval: cfg.guidance_interval,
            guidance_interval_decay: cfg.guidance_interval_decay,
            min_guidance_scale: cfg.min_guidance_scale,
            zero_steps: cfg.cfg_zero_steps,
            omega_scale: cfg.omega_scale,
            solver,
            seed: cfg.seed,
            custom_sched,
            repaint,
            dual,
            dcw,
        };
        if cfg.dit_cfg > 1.0 {
            println!("[acestep_render] DiT guidance: type={:?} scale={} interval={} decay={} min={} omega={} shift={shift}",
                     cfg_type, cfg.dit_cfg, cfg.guidance_interval, cfg.guidance_interval_decay, cfg.min_guidance_scale, cfg.omega_scale);
        } else {
            println!("[acestep_render] DiT: no guidance (cfg<=1), shift={shift}");
        }
        ph::try_note(progress, ph::phase::DENOISE, 0, 0)?;
        let latent = dit_latent_cfg(
            &dit, &context, &enc, s_total, uncond_ref, &noise, t, &sp, progress,
        )?;
        println!(
            "[acestep_render] DiT Euler: {} steps in {:.1}s",
            cfg.dit_steps,
            t1.elapsed().as_secs_f32()
        );
        drop(dit);
        Ok(latent)
    })?;

    // Post-DiT latent tuning before VAE decode (identity at the 1.0/0.0 defaults).
    for v in latent.iter_mut() {
        *v = *v * cfg.latent_rescale + cfg.latent_shift;
    }

    // 5) VAE (chunked, OOM-safe) -> WAV. The chunked decode already halves its chunk on CUDA OOM;
    //    the backstop additionally reloads the VAE (more reserve / on CPU) if even the floor chunk
    //    won't fit, so the decode always completes.
    let (audio, c_audio, t_audio) = oom_retry("VAE", || -> Result<_, DynErr> {
        let lat_cm: Vec<f32> = {
            let mut v = vec![0f32; 64 * t];
            for ti in 0..t {
                for c in 0..64 {
                    v[c * t + ti] = latent[ti * 64 + c];
                }
            }
            v
        };
        ph::try_note(progress, ph::phase::DECODE, 0, t)?;
        let vae = OobleckDecoder::from_gguf(&vae_g, 1e-12)?;
        let t2 = std::time::Instant::now();
        // The decode walks the latent in chunks and now says where it is. Reported through
        // a plain closure because the try-form can refuse (a cancelled render), and a
        // decoder has no business unwinding on a progress report - the refusal is honoured
        // at the next step boundary instead.
        let on_chunk = |done: usize, total: usize| {
            let _ = ph::try_note(progress, ph::phase::DECODE, done, total);
        };
        let on_chunk: &dyn Fn(usize, usize) = &on_chunk;
        // The chunk and its overlap are the window the decode's reserve was sized on, so
        // they are read from the same place rather than typed twice.
        let r = vae.decode_chunked_reporting(
            &lat_cm,
            64,
            t,
            crate::inference::place::audio_demand::DECODE_CHUNK_FRAMES,
            crate::inference::place::audio_demand::DECODE_OVERLAP_FRAMES,
            Some(on_chunk),
        )?;
        println!("[acestep_render] VAE: {:.1}s", t2.elapsed().as_secs_f32());
        Ok(r)
    })?;
    // Clean the tail: the VAE emits a steady low-level noise floor even after the music
    // ends, exposed (and audible) once the track fades out. Trim the trailing sub-threshold
    // floor (keeping a short guard for natural decay) and apply a short cosine fade-out so
    // the file ends at exactly zero - removes the end-of-track hiss and any boundary click.
    let (audio, t_audio) = finalize_audio(&audio, c_audio, t_audio, 48000, cfg.peak_clip);
    let wav = encode_wav_s16le(&audio, c_audio, t_audio, 48000);
    if let Some(dir) = std::path::Path::new(&cfg.out).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&cfg.out, &wav)?;
    println!("[acestep_render] ✅ wrote {}: {c_audio}ch x {t_audio} ({:.1}s, {} styles morph) total {:.1}s",
             cfg.out, t_audio as f32 / 48000.0, sections.len(), t0.elapsed().as_secs_f32());
    Ok(())
}
