//! A patched time series encoder with a quantile head.
//!
//! The series is scaled by its own mean and spread, cut into patches of a fixed length, and
//! each patch is embedded with its time position and its observed mask. A learned register
//! row follows the context, then one row per patch of the horizon, embedded from its time
//! position alone. One encoder pass runs over the whole sequence: per block, attention along
//! time with rotary positions, attention across the series of a group, and a feed-forward.
//! The horizon's rows are read out as every quantile of every future step, then scaled back.

use crate::tensor::layer as nl;
use crate::tensor::layer::{Linear, RmsNorm};
use crate::tensor::{self, DType, Device, Tensor, VarBuilder};
use serde::Deserialize;

// What a checkpoint states when its configuration does not.
crate::serde_defaults! {
    default_layer_norm_epsilon: f64 = 1e-6;
    default_rope_theta: f32 = 10000.0;
    default_max_output_patches: usize = 1;
}

fn default_activation() -> String {
    "relu".to_string()
}

/// The forecasting block of the configuration: patching, horizon and quantile levels.
#[derive(Debug, Clone, Deserialize)]
pub struct Patching {
    pub context_length: usize,
    pub input_patch_size: usize,
    pub input_patch_stride: usize,
    pub output_patch_size: usize,
    pub quantiles: Vec<f32>,
    #[serde(default)]
    pub use_reg_token: bool,
    #[serde(default)]
    pub use_arcsinh: bool,
    #[serde(default = "default_max_output_patches")]
    pub max_output_patches: usize,
    #[serde(default)]
    pub time_encoding_scale: Option<f32>,
}

/// The checkpoint's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_activation")]
    pub dense_act_fn: String,
    pub chronos_config: Patching,
}

impl Config {
    /// The longest horizon one encoder pass answers.
    pub fn max_horizon(&self) -> usize {
        self.chronos_config.max_output_patches * self.chronos_config.output_patch_size
    }

    /// Rows one pass attends over at the longest context and horizon.
    pub fn max_rows(&self) -> usize {
        let p = &self.chronos_config;
        p.context_length.div_ceil(p.input_patch_size.max(1))
            + usize::from(p.use_reg_token)
            + p.max_output_patches
    }
}

/// The feed-forward activations a checkpoint may name.
#[derive(Debug, Clone, Copy)]
enum Act {
    Relu,
    Gelu,
    GeluTanh,
    Silu,
}

impl Act {
    fn named(name: &str) -> tensor::Result<Self> {
        Ok(match name {
            "relu" => Act::Relu,
            "gelu" => Act::Gelu,
            "gelu_new" | "gelu_pytorch_tanh" => Act::GeluTanh,
            "silu" | "swish" => Act::Silu,
            other => tensor::bail!("unsupported feed-forward activation `{other}`"),
        })
    }

    fn apply(self, x: &Tensor) -> tensor::Result<Tensor> {
        match self {
            Act::Relu => x.relu(),
            Act::Gelu => x.gelu_erf(),
            Act::GeluTanh => x.gelu(),
            Act::Silu => x.silu(),
        }
    }
}

/// `out(act(hidden(x))) + residual(x)`: the patch embedding and the quantile head.
#[derive(Debug, Clone)]
struct Residual {
    hidden: Linear,
    output: Linear,
    residual: Linear,
    act: Act,
}

impl Residual {
    fn load(
        vb: &VarBuilder,
        input: usize,
        hidden: usize,
        out: usize,
        act: Act,
    ) -> tensor::Result<Self> {
        Ok(Self {
            hidden: nl::linear(input, hidden, &vb.pp("hidden_layer"))?,
            output: nl::linear(hidden, out, &vb.pp("output_layer"))?,
            residual: nl::linear(input, out, &vb.pp("residual_layer"))?,
            act,
        })
    }

    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        let h = self.act.apply(&self.hidden.forward(x)?)?;
        self.output.forward(&h)?.add(&self.residual.forward(x)?)
    }
}

/// Four bias-free projections of one attention.
#[derive(Debug, Clone)]
struct Projections {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl Projections {
    fn load(vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        let inner = cfg.num_heads * cfg.d_kv;
        let one = |n: &str, from, to| nl::linear_no_bias(from, to, &vb.pp(n));
        Ok(Self {
            q: one("q", cfg.d_model, inner)?,
            k: one("k", cfg.d_model, inner)?,
            v: one("v", cfg.d_model, inner)?,
            o: one("o", inner, cfg.d_model)?,
        })
    }
}

/// One encoder block: time attention, group attention, feed-forward, each pre-normed and
/// added back.
#[derive(Debug, Clone)]
struct Block {
    time_norm: RmsNorm,
    time: Projections,
    group_norm: RmsNorm,
    group: Projections,
    ff_norm: RmsNorm,
    wi: Linear,
    wo: Linear,
}

impl Block {
    fn load(vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        let eps = cfg.layer_norm_epsilon as f32;
        let norm = |vb: &VarBuilder| nl::rms_norm(cfg.d_model, eps, &vb.pp("layer_norm"));
        let (t, g, f) = (vb.pp("layer.0"), vb.pp("layer.1"), vb.pp("layer.2"));
        Ok(Self {
            time_norm: norm(&t)?,
            time: Projections::load(&t.pp("self_attention"), cfg)?,
            group_norm: norm(&g)?,
            group: Projections::load(&g.pp("self_attention"), cfg)?,
            ff_norm: norm(&f)?,
            wi: nl::linear_no_bias(cfg.d_model, cfg.d_ff, &f.pp("mlp.wi"))?,
            wo: nl::linear_no_bias(cfg.d_ff, cfg.d_model, &f.pp("mlp.wo"))?,
        })
    }
}

/// What one forecast answers: the quantile levels, and per level one value per future step.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    pub levels: Vec<f32>,
    pub values: Vec<Vec<f32>>,
}

impl Forecast {
    /// The values of the level closest to the median.
    pub fn median(&self) -> &[f32] {
        let i = self
            .levels
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - 0.5).abs().total_cmp(&(b.1 - 0.5).abs()))
            .map_or(0, |(i, _)| i);
        &self.values[i]
    }
}

/// The encoder and its two heads, on the device it was built on.
#[derive(Debug, Clone)]
pub struct Forecaster {
    cfg: Config,
    act: Act,
    input_embed: Residual,
    output_embed: Residual,
    register: Option<Tensor>,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    device: Device,
}

/// A series scaled by its mean and spread over the observed points, as the encoder reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scale {
    loc: f32,
    scale: f32,
    arcsinh: bool,
}

/// The spread a constant series is divided by.
const FLAT_SCALE: f32 = 1e-5;

impl Scale {
    fn of(series: &[f32], arcsinh: bool) -> Self {
        let seen: Vec<f32> = series.iter().copied().filter(|v| v.is_finite()).collect();
        let n = seen.len().max(1) as f32;
        let loc = if seen.is_empty() {
            0.0
        } else {
            seen.iter().sum::<f32>() / n
        };
        let var = seen.iter().map(|v| (v - loc) * (v - loc)).sum::<f32>() / n;
        let scale = if seen.is_empty() { 1.0 } else { var.sqrt() };
        let scale = if scale == 0.0 { FLAT_SCALE } else { scale };
        Self {
            loc,
            scale,
            arcsinh,
        }
    }

    fn apply(&self, v: f32) -> f32 {
        let s = (v - self.loc) / self.scale;
        if self.arcsinh {
            s.asinh()
        } else {
            s
        }
    }

    fn invert(&self, v: f32) -> f32 {
        let v = if self.arcsinh { v.sinh() } else { v };
        v * self.scale + self.loc
    }
}

impl Forecaster {
    /// Build from a checkpoint's tensors. `vb` is rooted at the checkpoint's top level.
    pub fn load(vb: &VarBuilder, cfg: Config) -> tensor::Result<Self> {
        let p = &cfg.chronos_config;
        if p.input_patch_size == 0 || p.input_patch_size != p.input_patch_stride {
            tensor::bail!("only non-overlapping patches are supported");
        }
        if p.input_patch_size != p.output_patch_size {
            tensor::bail!("the input and output patches must be the same length");
        }
        if p.quantiles.is_empty() {
            tensor::bail!("the checkpoint declares no quantile levels");
        }
        let act = Act::named(&cfg.dense_act_fn)?;
        let patch = p.input_patch_size;
        let input_embed = Residual::load(
            &vb.pp("input_patch_embedding"),
            3 * patch,
            cfg.d_ff,
            cfg.d_model,
            act,
        )?;
        let output_embed = Residual::load(
            &vb.pp("output_patch_embedding"),
            cfg.d_model,
            cfg.d_ff,
            p.quantiles.len() * patch,
            act,
        )?;
        let register = if p.use_reg_token {
            let table = vb.get((2, cfg.d_model), "shared.weight")?;
            Some(table.narrow(0, 1, 1)?.reshape((1, 1, cfg.d_model))?)
        } else {
            None
        };
        let blocks = (0..cfg.num_layers)
            .map(|i| Block::load(&vb.pp(format!("encoder.block.{i}")), &cfg))
            .collect::<tensor::Result<Vec<_>>>()?;
        let final_norm = nl::rms_norm(
            cfg.d_model,
            cfg.layer_norm_epsilon as f32,
            &vb.pp("encoder.final_layer_norm"),
        )?;
        Ok(Self {
            device: vb.device().clone(),
            cfg,
            act,
            input_embed,
            output_embed,
            register,
            blocks,
            final_norm,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The quantile levels the head answers.
    pub fn levels(&self) -> &[f32] {
        &self.cfg.chronos_config.quantiles
    }

    /// Forecast `horizon` steps after `series`. A non-finite value is a missing point. The
    /// context is the series' last `context_length` points; the horizon is at most
    /// [`Config::max_horizon`].
    pub fn forecast(&self, series: &[f32], horizon: usize) -> tensor::Result<Forecast> {
        let p = &self.cfg.chronos_config;
        let patch = p.input_patch_size;
        if horizon == 0 || horizon > self.cfg.max_horizon() {
            tensor::bail!(
                "horizon {horizon} is outside 1..={} for this checkpoint",
                self.cfg.max_horizon()
            );
        }
        let series = &series[series.len().saturating_sub(p.context_length)..];
        if !series.iter().any(|v| v.is_finite()) {
            tensor::bail!("the series holds no observed value");
        }
        let scale = Scale::of(series, p.use_arcsinh);
        let time_scale = p.time_encoding_scale.unwrap_or(p.context_length as f32);

        // The context, left-padded with missing points to whole patches.
        let n_ctx = series.len().div_ceil(patch);
        let pad = n_ctx * patch - series.len();
        let mut rows = Vec::with_capacity(n_ctx * 3 * patch);
        let mut observed = Vec::with_capacity(n_ctx);
        for i in 0..n_ctx {
            let at = |j: usize| (i * patch + j).checked_sub(pad).map(|k| series[k]);
            let start = -((n_ctx * patch) as f32) + (i * patch) as f32;
            rows.extend((0..patch).map(|j| (start + j as f32) / time_scale));
            let mut any = false;
            for j in 0..patch {
                rows.push(match at(j) {
                    Some(v) if v.is_finite() => scale.apply(v),
                    _ => 0.0,
                });
            }
            for j in 0..patch {
                let seen = at(j).is_some_and(f32::is_finite);
                any |= seen;
                rows.push(f32::from(u8::from(seen)));
            }
            observed.push(any);
        }
        let ctx = Tensor::from_vec(rows, (1, n_ctx, 3 * patch), &self.device)?;
        let mut parts = vec![self.input_embed.forward(&ctx)?];
        if let Some(reg) = &self.register {
            parts.push(reg.clone());
            observed.push(true);
        }

        // The horizon: its time positions, with nothing observed.
        let n_out = horizon.div_ceil(patch);
        let mut future = Vec::with_capacity(n_out * 3 * patch);
        for i in 0..n_out {
            future.extend((0..patch).map(|j| ((i * patch + j) as f32) / time_scale));
            future.extend(std::iter::repeat_n(0.0, 2 * patch));
        }
        let future = Tensor::from_vec(future, (1, n_out, 3 * patch), &self.device)?;
        parts.push(self.input_embed.forward(&future)?);
        observed.extend(std::iter::repeat_n(true, n_out));
        let refs: Vec<&Tensor> = parts.iter().collect();
        let mut x = Tensor::cat(&refs, 1)?;

        let rows = observed.len();
        let mask = if observed.iter().all(|o| *o) {
            None
        } else {
            let cols: Vec<f32> = observed
                .iter()
                .map(|o| if *o { 0.0 } else { f32::MIN })
                .collect();
            let table: Vec<f32> = (0..rows).flat_map(|_| cols.iter().copied()).collect();
            Some(Tensor::from_vec(table, (1, 1, rows, rows), &self.device)?)
        };
        let (cos, sin) = self.rotary(rows)?;
        for block in &self.blocks {
            x = self.block(block, &x, mask.as_ref(), &cos, &sin)?;
        }
        let x = self.final_norm.forward(&x)?;
        let head = self
            .output_embed
            .forward(&x.narrow(1, rows - n_out, n_out)?.contiguous()?)?;
        let flat = head.flatten_all()?.to_vec1::<f32>()?;

        // Row `i` holds, per level `q`, the patch's steps at `q * patch + j`.
        let q_count = p.quantiles.len();
        let values = (0..q_count)
            .map(|q| {
                (0..horizon)
                    .map(|h| {
                        let (i, j) = (h / patch, h % patch);
                        scale.invert(flat[i * q_count * patch + q * patch + j])
                    })
                    .collect()
            })
            .collect();
        Ok(Forecast {
            levels: p.quantiles.clone(),
            values,
        })
    }

    /// The rotation tables for `rows` positions: `[rows, d_kv / 2]` each.
    fn rotary(&self, rows: usize) -> tensor::Result<(Tensor, Tensor)> {
        let inv =
            crate::inference::model::rope::inverse_frequencies(self.cfg.d_kv, self.cfg.rope_theta);
        let half = inv.len();
        let mut cos = Vec::with_capacity(rows * half);
        let mut sin = Vec::with_capacity(rows * half);
        for pos in 0..rows {
            for f in &inv {
                let a = pos as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos, (rows, half), &self.device)?,
            Tensor::from_vec(sin, (rows, half), &self.device)?,
        ))
    }

    fn block(
        &self,
        b: &Block,
        x: &Tensor,
        mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> tensor::Result<Tensor> {
        let (_, rows, _) = x.dims3()?;
        let (heads, dk) = (self.cfg.num_heads, self.cfg.d_kv);
        let split = |t: Tensor| -> tensor::Result<Tensor> {
            t.reshape((1, rows, heads, dk))?
                .transpose(1, 2)?
                .contiguous()
        };

        // Attention along time, rotary positions on queries and keys, scores unscaled.
        let h = b.time_norm.forward(x)?;
        let q = crate::tensor::ops::rope(&split(b.time.q.forward(&h)?)?, cos, sin)?;
        let k = crate::tensor::ops::rope(&split(b.time.k.forward(&h)?)?, cos, sin)?;
        let v = split(b.time.v.forward(&h)?)?;
        let a = crate::inference::model::sdpa::sdpa(&q, &k, &v, mask, false, 1.0, 1.0)?
            .transpose(1, 2)?
            .reshape((1, rows, heads * dk))?;
        let x = x.add(&b.time.o.forward(&a)?)?;

        // Attention across the series of a group. One series is a group of one: each row
        // attends to itself alone, so the softmax is one and the output is its value.
        let h = b.group_norm.forward(&x)?;
        let x = x.add(&b.group.o.forward(&b.group.v.forward(&h)?)?)?;

        let h = b.ff_norm.forward(&x)?;
        let h = b.wo.forward(&self.act.apply(&b.wi.forward(&h)?)?)?;
        x.add(&h)
    }
}

/// Load the checkpoint in `dir` on `device`.
pub fn load_dir(dir: &std::path::Path, device: &Device) -> anyhow::Result<Forecaster> {
    let cfg = read_config(dir)?;
    let weights = weight_files(dir)?;
    let vb = unsafe { VarBuilder::from_files(&weights, DType::F32, device)? };
    Ok(Forecaster::load(&vb, cfg)?)
}

/// The configuration a checkpoint directory carries.
pub fn read_config(dir: &std::path::Path) -> anyhow::Result<Config> {
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("read {}: {e}", dir.join("config.json").display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// The safetensors files of a checkpoint directory, sorted.
pub fn weight_files(dir: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        anyhow::bail!("{} holds no safetensors weights", dir.display());
    }
    Ok(files)
}

#[cfg(test)]
mod fixed_set;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_round_trips() {
        let s = Scale::of(&[1.0, 2.0, f32::NAN, 3.0], true);
        assert!((s.loc - 2.0).abs() < 1e-6);
        for v in [-4.0f32, 0.5, 2.0, 9.0] {
            assert!((s.invert(s.apply(v)) - v).abs() < 1e-4);
        }
        assert_eq!(Scale::of(&[5.0, 5.0], false).scale, FLAT_SCALE);
    }

    #[test]
    fn median_is_the_level_nearest_one_half() {
        let f = Forecast {
            levels: vec![0.1, 0.5, 0.9],
            values: vec![vec![1.0], vec![2.0], vec![3.0]],
        };
        assert_eq!(f.median(), &[2.0]);
    }
}
