//! A patched time series transformer that reads several series of a group at once.
//!
//! Every row (a target or a covariate) is normalized causally per patch, and each patch is
//! embedded from its values and its missing points. Learned query tokens follow the context,
//! one per patch of the horizon. A stack of layers runs attention along time (rotary
//! positions on part of each head) and attention across the rows of a group, each gated by a
//! sigmoid of its input, then a SwiGLU feed-forward. The query tokens are read out as a
//! monotone fan of quantiles per step, in the frame of the last context patch.
//!
//! With covariates, the window covers the context and the horizon block, so a covariate
//! known over the horizon is read as keys; masks keep the targets out of the covariates and
//! every row's future out of the targets. [`decode`] wraps one block into a forecast.

pub mod decode;
pub mod frame;

use crate::tensor::layer::{Linear, RmsNorm};
use crate::tensor::{self, DType, Device, Tensor, VarBuilder};
use serde::Deserialize;

pub use decode::{Forecaster, Request};

crate::serde_defaults! {
    default_rope_theta: f64 = 10000.0;
    default_rope_partial: f64 = 0.5;
    default_norm_eps: f64 = 1e-6;
    default_minimum_scale: f64 = 1e-6;
    default_true: bool = true;
    default_width_factor: f32 = 2.0;
    default_gate: String = "sdpa_sigmoid".to_string();
}

/// The checkpoint's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<String>,
    pub quantile_knots: Vec<f32>,
    pub block_size: usize,
    pub context_length: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_partial")]
    pub rope_partial_factor: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_gate")]
    pub time_attn_gate: String,
    #[serde(default = "default_gate")]
    pub variate_attn_gate: String,
    #[serde(default = "default_minimum_scale")]
    pub minimum_scale: f64,
    #[serde(default = "default_true")]
    pub use_asinh: bool,
    #[serde(default)]
    pub stabilize_with_global: bool,
    #[serde(default = "default_true")]
    pub covariate_enabled: bool,
    #[serde(default = "default_true")]
    pub attend_missing_patches: bool,
    #[serde(default = "default_width_factor")]
    pub quantile_width_factor: f32,
}

/// The attentions a layer kind runs, in order, before its feed-forward: `true` along time.
fn layer_axes(kind: &str) -> tensor::Result<&'static [bool]> {
    Ok(match kind {
        "time" => &[true],
        "variate" => &[false],
        "time_variate" => &[true, false],
        "variate_time" => &[false, true],
        other => tensor::bail!("unknown layer kind `{other}`"),
    })
}

/// Whether an attention output is gated by a sigmoid of its input.
fn gated(gate: &str) -> tensor::Result<bool> {
    Ok(match gate {
        "none" => false,
        "sdpa_sigmoid" => true,
        other => tensor::bail!("unknown attention gate `{other}`"),
    })
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads.max(1)
    }

    /// Head dimensions the rotary embedding turns; the rest pass through.
    pub fn rope_dim(&self) -> usize {
        (self.head_dim() as f64 * self.rope_partial_factor) as usize
    }

    /// The index of the median level.
    pub fn median_index(&self) -> Option<usize> {
        self.quantile_knots.iter().position(|q| *q == 0.5)
    }

    /// The longest horizon a request may ask for: past the context length the forecast
    /// reads nothing but its own medians.
    pub fn max_horizon(&self) -> usize {
        self.context_length
    }

    /// Whether some time attention is followed by both another time attention and a
    /// variate attention, the path a covariate's horizon takes to reach a target.
    pub fn supports_known_covariates(&self) -> bool {
        if !self.covariate_enabled {
            return false;
        }
        let ops: Vec<bool> = self
            .block_schedule
            .iter()
            .filter_map(|k| layer_axes(k).ok())
            .flatten()
            .copied()
            .collect();
        (0..ops.len()).any(|i| {
            let rest = &ops[i + 1..];
            ops[i] && rest.iter().any(|t| *t) && !rest.iter().all(|t| *t)
        })
    }

    /// Tokens one row carries at the longest context: the context patches, then a horizon
    /// block of queries; a window with covariates carries both twice.
    pub fn max_tokens(&self) -> usize {
        2 * (self.context_length.div_ceil(self.patch_size.max(1)) + self.block_size)
    }

    fn validate(&self) -> tensor::Result<()> {
        if self.patch_size == 0
            || self.num_heads == 0
            || !self.d_model.is_multiple_of(self.num_heads)
        {
            tensor::bail!("the model width does not split into its heads");
        }
        let rope = self.rope_dim();
        if rope == 0 || !rope.is_multiple_of(2) {
            tensor::bail!("the rotary part of a head must be even and not empty");
        }
        if self.median_index().is_none() || !self.quantile_knots.windows(2).all(|w| w[0] < w[1]) {
            tensor::bail!("the quantile levels must increase and hold the median");
        }
        if self.stabilize_with_global {
            tensor::bail!("the advancing frame of a decoded block needs per-series statistics");
        }
        for kind in &self.block_schedule {
            layer_axes(kind)?;
        }
        gated(&self.time_attn_gate)?;
        gated(&self.variate_attn_gate)?;
        Ok(())
    }
}

/// What a row stands for in a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Target,
    /// A covariate known over the context only.
    Past,
    /// A covariate known over the context and the horizon.
    Known,
}

/// The rows of one decode block, packed: `rows` rows of `len` context points each, plus the
/// horizon block of the covariates known over it.
#[derive(Debug, Clone)]
pub struct Packed {
    pub rows: usize,
    pub len: usize,
    pub values: Vec<f32>,
    pub observed: Vec<bool>,
    /// The group of each row; rows of one group attend to each other.
    pub group: Vec<usize>,
    /// `None` when every row is a target.
    pub roles: Option<Vec<Role>>,
    /// Per row, `horizon_patches * patch_size` values over the block, zero off known rows.
    pub known_future: Option<(Vec<f32>, Vec<bool>)>,
}

/// `out(silu(hidden(x))) + residual(x)`: the patch embedding and the quantile head.
#[derive(Debug, Clone)]
struct Residual {
    hidden: Linear,
    output: Linear,
    residual: Linear,
}

impl Residual {
    fn load(vb: &VarBuilder) -> tensor::Result<Self> {
        let one = |name: &str| linear(&vb.pp(name));
        Ok(Self {
            hidden: one("hidden_layer")?,
            output: one("output_layer")?,
            residual: one("residual_layer")?,
        })
    }

    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        let h = self.hidden.forward(x)?.silu()?;
        self.output.forward(&h)?.add(&self.residual.forward(x)?)
    }
}

/// One attention, along time or across rows.
#[derive(Debug, Clone)]
struct Attention {
    along_time: bool,
    norm: RmsNorm,
    /// Queries, keys and values, heads-major in the output.
    qkv: Linear,
    gate: Option<Linear>,
    o: Linear,
}

/// One layer: its attentions, then the feed-forward.
#[derive(Debug, Clone)]
struct Layer {
    attentions: Vec<Attention>,
    mlp_norm: RmsNorm,
    /// Gate then up projection, fused.
    w12: Linear,
    w3: Linear,
}

/// The tensor `name` under `vb`, at the shape the checkpoint stores it.
fn stored(vb: &VarBuilder, name: &str) -> tensor::Result<Tensor> {
    vb.get(vb.shape_of(name)?, name)
}

/// A linear layer of `vb`, its bias when the checkpoint has one.
fn linear(vb: &VarBuilder) -> tensor::Result<Linear> {
    let bias = if vb.contains("bias") {
        Some(stored(vb, "bias")?)
    } else {
        None
    };
    Linear::new(stored(vb, "weight")?, bias)
}

/// The fused projection's rows, stored as `(qkv, head_dim, heads)`, reordered so each head's
/// dimensions are contiguous: `(qkv, heads, head_dim)`.
fn heads_major(w: &Tensor, heads: usize, head_dim: usize) -> tensor::Result<Tensor> {
    let (out, inp) = w.dims2()?;
    let parts = out / (heads * head_dim);
    w.reshape((parts, head_dim, heads, inp))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((out, inp))
}

/// The network on the device it was built on.
#[derive(Debug, Clone)]
pub struct Network {
    cfg: Config,
    median: usize,
    embed: Residual,
    query: Tensor,
    layers: Vec<Layer>,
    out_norm: RmsNorm,
    head: Residual,
    device: Device,
}

/// The hidden states of one block, recorded for a parity check.
pub type Trace = Vec<(String, Tensor)>;

impl Network {
    /// Build from a checkpoint's tensors. `vb` is rooted at the checkpoint's top level.
    pub fn load(vb: &VarBuilder, cfg: Config) -> tensor::Result<Self> {
        cfg.validate()?;
        let eps = cfg.norm_eps as f32;
        let (heads, hd) = (cfg.num_heads, cfg.head_dim());
        let norm = |vb: VarBuilder| -> tensor::Result<RmsNorm> {
            Ok(RmsNorm::new(vb.get(cfg.d_model, "scale")?, eps))
        };
        let mut layers = Vec::with_capacity(cfg.block_schedule.len());
        for (i, kind) in cfg.block_schedule.iter().enumerate() {
            let vb = vb.pp(format!("transformer.layers.{i}"));
            let mut attentions = Vec::new();
            for (j, along_time) in layer_axes(kind)?.iter().enumerate() {
                let a = vb.pp(format!("attns.{j}"));
                let p = a.pp("wQKV");
                let bias = if p.contains("bias") {
                    let b = stored(&p, "bias")?;
                    let b = b.reshape((b.dim(0)?, 1))?;
                    Some(heads_major(&b, heads, hd)?.flatten_all()?)
                } else {
                    None
                };
                let qkv = Linear::new(heads_major(&stored(&p, "weight")?, heads, hd)?, bias)?;
                let gate_kind = if *along_time {
                    &cfg.time_attn_gate
                } else {
                    &cfg.variate_attn_gate
                };
                let gate = if gated(gate_kind)? {
                    Some(linear(&a.pp("_gate"))?)
                } else {
                    None
                };
                attentions.push(Attention {
                    along_time: *along_time,
                    norm: norm(vb.pp(format!("attn_norms.{j}")))?,
                    qkv,
                    gate,
                    o: linear(&a.pp("wO"))?,
                });
            }
            layers.push(Layer {
                attentions,
                mlp_norm: norm(vb.pp("mlp_norm"))?,
                w12: linear(&vb.pp("mlp.w12"))?,
                w3: linear(&vb.pp("mlp.w3"))?,
            });
        }
        let query = vb
            .get(cfg.d_model, "representation.forecast_query")?
            .reshape((1, 1, cfg.d_model))?;
        Ok(Self {
            median: cfg.median_index().unwrap_or(0),
            embed: Residual::load(&vb.pp("representation.patch_proj"))?,
            query,
            layers,
            out_norm: norm(vb.pp("transformer.out_norm"))?,
            head: Residual::load(&vb.pp("head.projection"))?,
            device: vb.device().clone(),
            cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Normalize every row of a window causally per patch; a known covariate's row reads its
    /// whole-row frame instead. Returns the normalized values and each patch's `(loc, scale)`.
    fn normalize(
        &self,
        values: &[f32],
        observed: &[bool],
        rows: usize,
        roles: Option<&[Role]>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let len = values.len() / rows.max(1);
        let n = len / c.patch_size;
        let mut scaled = vec![0f32; values.len()];
        let (mut locs, mut scales) = (vec![0f32; rows * n], vec![0f32; rows * n]);
        for r in 0..rows {
            let (v, o) = (&values[r * len..][..len], &observed[r * len..][..len]);
            let known = roles.is_some_and(|roles| roles[r] == Role::Known);
            let (loc, scale) = if known {
                let (loc, scale, indicator) = frame::row_frame(v, o, c.minimum_scale);
                for t in 0..len {
                    scaled[r * len + t] =
                        frame::normalize(v[t], loc, scale, o[t], c.use_asinh && !indicator);
                }
                (vec![loc; n], vec![scale; n])
            } else {
                let (loc, scale) = frame::patch_frames(v, o, c.patch_size, c.minimum_scale);
                for t in 0..len {
                    let p = t / c.patch_size;
                    scaled[r * len + t] =
                        frame::normalize(v[t], loc[p], scale[p], o[t], c.use_asinh);
                }
                (loc, scale)
            };
            locs[r * n..][..n].copy_from_slice(&loc);
            scales[r * n..][..n].copy_from_slice(&scale);
        }
        (scaled, locs, scales)
    }

    /// Forecast one block of `horizon_patches` patches after `packed`. Returns raw-space
    /// quantiles `[rows][horizon_patches * patch_size][levels]`, flattened, non-decreasing
    /// along the levels. `trace` collects the embedding and each layer's output.
    pub fn predict_block(
        &self,
        packed: &Packed,
        horizon_patches: usize,
        mut trace: Option<&mut Trace>,
    ) -> tensor::Result<Vec<f32>> {
        let c = &self.cfg;
        let ps = c.patch_size;
        let (rows, len) = (packed.rows, packed.len);
        if len % ps != 0 || len == 0 || horizon_patches == 0 {
            tensor::bail!("a block needs whole context patches and at least one horizon patch");
        }
        let roles = packed
            .roles
            .as_deref()
            .filter(|r| c.covariate_enabled && r.iter().any(|r| *r != Role::Target));
        let ll = len / ps;
        // The window: the context, and with covariates the horizon block after it.
        let (values, observed, n, n_dec) = match roles {
            None => (
                packed.values.clone(),
                packed.observed.clone(),
                ll,
                horizon_patches,
            ),
            Some(_) => {
                if horizon_patches > c.block_size {
                    tensor::bail!(
                        "a block with covariates decodes at most {} patches",
                        c.block_size
                    );
                }
                let h = horizon_patches * ps;
                let zeros = (vec![0f32; rows * h], vec![false; rows * h]);
                let (fv, fo) = packed.known_future.as_ref().unwrap_or(&zeros);
                let w = len + h;
                let mut v = Vec::with_capacity(rows * w);
                let mut o = Vec::with_capacity(rows * w);
                for r in 0..rows {
                    v.extend_from_slice(&packed.values[r * len..][..len]);
                    v.extend_from_slice(&fv[r * h..][..h]);
                    o.extend_from_slice(&packed.observed[r * len..][..len]);
                    o.extend_from_slice(&fo[r * h..][..h]);
                }
                (v, o, ll + horizon_patches, ll + horizon_patches)
            }
        };
        let w = n * ps;
        let (scaled, locs, scales) = self.normalize(&values, &observed, rows, roles);

        // The patch embedding reads each patch's values, then its missing points.
        let mut patches = Vec::with_capacity(rows * n * 2 * ps);
        for r in 0..rows {
            for p in 0..n {
                let at = r * w + p * ps;
                patches.extend_from_slice(&scaled[at..at + ps]);
                patches.extend(
                    observed[at..at + ps]
                        .iter()
                        .map(|o| f32::from(u8::from(!o))),
                );
            }
        }
        let x_enc =
            self.embed
                .forward(&Tensor::from_vec(patches, (rows, n, 2 * ps), &self.device)?)?;
        if let Some(t) = trace.as_deref_mut() {
            t.push(("embed".into(), x_enc.clone()));
        }
        let x_dec = self
            .query
            .broadcast_as((rows, n_dec, c.d_model))?
            .contiguous()?;
        let mut x = Tensor::cat(&[&x_enc, &x_dec], 1)?;

        // Which patches are keys: observed, or after the row's first observation.
        let obs_patch: Vec<bool> = (0..rows * n)
            .map(|i| {
                observed[(i / n) * w + (i % n) * ps..][..ps]
                    .iter()
                    .any(|o| *o)
            })
            .collect();
        let mut key_valid = obs_patch.clone();
        if c.attend_missing_patches {
            for r in 0..rows {
                let mut started = false;
                for p in 0..n {
                    started |= obs_patch[r * n + p];
                    key_valid[r * n + p] = started;
                }
            }
        }
        let time_mask = self.time_mask(rows, n, ll, n_dec, &key_valid, roles)?;
        let variate_mask = self.variate_mask(rows, n, n_dec, &key_valid, &packed.group, roles)?;
        let first_obs = first_observed(rows, n, &obs_patch, &packed.group);
        let rope = self.rotary(rows, n, n_dec, roles.is_some(), &first_obs)?;

        for (i, layer) in self.layers.iter().enumerate() {
            for a in &layer.attentions {
                let h = a.norm.forward(&x)?;
                let out = if a.along_time {
                    self.attend_time(a, &h, &time_mask, &rope)?
                } else {
                    let ht = h.transpose(0, 1)?.contiguous()?;
                    let out = match &variate_mask {
                        Some(mask) => self.attend(a, &ht, Some(mask), None)?,
                        None => self.own_value(a, &ht)?,
                    };
                    out.transpose(0, 1)?.contiguous()?
                };
                x = x.add(&a.o.forward(&out)?)?;
            }
            let h = layer.mlp_norm.forward(&x)?;
            let gu = layer.w12.forward(&h)?;
            let hidden = c.mlp_hidden_dim;
            let gate = gu.narrow(2, 0, hidden)?;
            let up = gu.narrow(2, hidden, hidden)?;
            x = x.add(&layer.w3.forward(&gate.mul(&up.silu()?)?)?)?;
            if let Some(t) = trace.as_deref_mut() {
                t.push((format!("layer.{i}"), x.clone()));
            }
        }
        let x = self.out_norm.forward(&x)?;
        let raw = self
            .head
            .forward(&x.narrow(1, n, n_dec)?.contiguous()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Quantiles of the horizon patches, back in the frame of the last context patch.
        let q = c.quantile_knots.len();
        let skip = n_dec - horizon_patches;
        let steps = horizon_patches * ps;
        let mut out = Vec::with_capacity(rows * steps * q);
        for r in 0..rows {
            let (loc, scale) = (locs[r * n + ll - 1], scales[r * n + ll - 1]);
            for t in 0..steps {
                let at = ((r * n_dec + skip) * ps + t) * q;
                let mut step = raw[at..at + q].to_vec();
                frame::monotone(
                    &mut step,
                    &c.quantile_knots,
                    self.median,
                    c.quantile_width_factor,
                );
                for v in step.iter_mut() {
                    let u = if c.use_asinh { v.sinh() } else { *v };
                    *v = u * scale + loc;
                }
                frame::clamp_nonfinite(&mut step);
                out.extend_from_slice(&step);
            }
        }
        Ok(out)
    }

    /// The additive time masks `[1, 1, S, S]` over `[context ; queries]`, each with the rows
    /// that share it.
    ///
    /// The context reads itself causally over valid keys (a known covariate reads all of
    /// its valid keys) and never the queries. Without covariates every query reads the
    /// valid context and the queries before it. With them the window covers `n` patches
    /// of which the first `ll` are context: a query of the horizon reads the valid context
    /// and the horizon queries up to itself; any other query reads only itself.
    fn time_mask(
        &self,
        rows: usize,
        n: usize,
        ll: usize,
        n_dec: usize,
        key_valid: &[bool],
        roles: Option<&[Role]>,
    ) -> tensor::Result<Vec<(Vec<usize>, Tensor)>> {
        let s = n + n_dec;
        let mut distinct: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();
        for r in 0..rows {
            let valid = &key_valid[r * n..][..n];
            let all_keys = roles.is_some_and(|roles| roles[r] == Role::Known);
            let mut m = vec![f32::NEG_INFINITY; s * s];
            let mut open = |i: usize, j: usize| m[i * s + j] = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i == j || (valid[j] && (all_keys || j <= i)) {
                        open(i, j);
                    }
                }
            }
            for d in 0..n_dec {
                let i = n + d;
                open(i, i);
                let horizon = roles.is_none() || d >= ll;
                if !horizon {
                    continue;
                }
                let context = if roles.is_none() { n } else { ll };
                for j in 0..context {
                    if valid[j] {
                        open(i, j);
                    }
                }
                let first = if roles.is_none() { 0 } else { ll };
                for e in first..=d {
                    open(i, n + e);
                }
            }
            match distinct.iter_mut().find(|(_, t)| *t == m) {
                Some((members, _)) => members.push(r),
                None => distinct.push((vec![r], m)),
            }
        }
        distinct
            .into_iter()
            .map(|(members, m)| Ok((members, Tensor::from_vec(m, (1, 1, s, s), &self.device)?)))
            .collect()
    }

    /// The additive variate mask `[S, 1, rows, rows]`, or `None` when no row can read
    /// another. A row reads a row of its group (on context slots, a valid key), except that
    /// a covariate never reads a target and a known covariate never reads a row that is not
    /// one. Every row reads itself.
    fn variate_mask(
        &self,
        rows: usize,
        n: usize,
        n_dec: usize,
        key_valid: &[bool],
        group: &[usize],
        roles: Option<&[Role]>,
    ) -> tensor::Result<Option<Tensor>> {
        let s = n + n_dec;
        let forbid = |q: usize, k: usize| -> bool {
            roles.is_some_and(|roles| {
                let target = |r: usize| roles[r] == Role::Target;
                let known = |r: usize| roles[r] == Role::Known;
                (!target(q) && target(k)) || (known(q) && !known(k))
            })
        };
        let mut any = false;
        let mut table = vec![f32::NEG_INFINITY; s * rows * rows];
        for slot in 0..s {
            for q in 0..rows {
                for k in 0..rows {
                    let valid = slot >= n || key_valid[k * n + slot];
                    let open = q == k || (group[q] == group[k] && valid && !forbid(q, k));
                    any |= open && q != k;
                    if open {
                        table[(slot * rows + q) * rows + k] = 0.0;
                    }
                }
            }
        }
        if !any {
            return Ok(None);
        }
        Ok(Some(Tensor::from_vec(
            table,
            (s, 1, rows, rows),
            &self.device,
        )?))
    }

    /// The rotary tables `[rows][S, head_dim / 2]` of each row's time positions, counted
    /// from its group's first observed patch. With covariates a query shares its patch's
    /// position; without, the queries continue after the context.
    fn rotary(
        &self,
        rows: usize,
        n: usize,
        n_dec: usize,
        window: bool,
        first_obs: &[usize],
    ) -> tensor::Result<Rotary> {
        let hd = self.cfg.head_dim();
        let rope = self.cfg.rope_dim();
        let half = hd / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| {
                if 2 * i < rope {
                    (1.0 / self.cfg.rope_theta.powf((2 * i) as f64 / rope as f64)) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let s = n + n_dec;
        let table = |first: usize| -> tensor::Result<(Tensor, Tensor)> {
            let mut cos = Vec::with_capacity(s * half);
            let mut sin = Vec::with_capacity(s * half);
            for i in 0..s {
                let p = if window && i >= n { i - n } else { i };
                let pos = p as f32 - first as f32;
                for f in &inv {
                    let a = pos * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
            Ok((
                Tensor::from_vec(cos, (s, half), &self.device)?,
                Tensor::from_vec(sin, (s, half), &self.device)?,
            ))
        };
        if first_obs.iter().all(|f| *f == first_obs[0]) {
            return Ok(Rotary::Shared(table(first_obs[0])?));
        }
        Ok(Rotary::PerRow(
            (0..rows)
                .map(|r| table(first_obs[r]))
                .collect::<tensor::Result<_>>()?,
        ))
    }

    /// Attention along time: the rows that share a mask attend together, and the outputs
    /// go back in row order.
    fn attend_time(
        &self,
        a: &Attention,
        h: &Tensor,
        masks: &[(Vec<usize>, Tensor)],
        rope: &Rotary,
    ) -> tensor::Result<Tensor> {
        if let [(_, mask)] = masks {
            return self.attend(a, h, Some(mask), Some(rope));
        }
        let rows = h.dim(0)?;
        let mut order = Vec::with_capacity(rows);
        let mut parts = Vec::with_capacity(masks.len());
        for (members, mask) in masks {
            let ids: Vec<u32> = members.iter().map(|r| *r as u32).collect();
            let ids = Tensor::from_vec(ids, members.len(), &self.device)?;
            let sub = h.index_select(&ids, 0)?;
            let part_rope = rope.select(members);
            parts.push(self.attend(a, &sub, Some(mask), Some(&part_rope))?);
            order.extend_from_slice(members);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let out = Tensor::cat(&refs, 0)?;
        let mut back = vec![0u32; rows];
        for (at, r) in order.iter().enumerate() {
            back[*r] = at as u32;
        }
        out.index_select(&Tensor::from_vec(back, rows, &self.device)?, 0)
    }

    /// The attention of a row that reads only itself: its value, gated.
    fn own_value(&self, a: &Attention, h: &Tensor) -> tensor::Result<Tensor> {
        let d = self.cfg.d_model;
        let v = a.qkv.forward(h)?.narrow(2, 2 * d, d)?.contiguous()?;
        match &a.gate {
            Some(g) => v.mul(&g.forward(h)?.sigmoid()?),
            None => Ok(v),
        }
    }

    /// One attention over `h` `[batch, seq, D]`: the batch is the rows along time, the
    /// patches across rows. Returns `[batch, seq, D]` before the output projection.
    fn attend(
        &self,
        a: &Attention,
        h: &Tensor,
        mask: Option<&Tensor>,
        rope: Option<&Rotary>,
    ) -> tensor::Result<Tensor> {
        let (b, s, d) = h.dims3()?;
        let (heads, hd) = (self.cfg.num_heads, self.cfg.head_dim());
        let qkv = a.qkv.forward(h)?;
        let split = |i: usize| -> tensor::Result<Tensor> {
            qkv.narrow(2, i * d, d)?
                .reshape((b, s, heads, hd))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (mut q, mut k, v) = (split(0)?, split(1)?, split(2)?);
        if let Some(rope) = rope {
            q = rope.apply(&q)?;
            k = rope.apply(&k)?;
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let out = crate::inference::model::sdpa::sdpa(&q, &k, &v, mask, false, scale, 1.0)?
            .transpose(1, 2)?
            .reshape((b, s, d))?;
        match &a.gate {
            Some(g) => out.mul(&g.forward(h)?.sigmoid()?),
            None => Ok(out),
        }
    }
}

/// Per row, the first patch any row of its group observes; 0 for a group that observes none.
fn first_observed(rows: usize, n: usize, obs_patch: &[bool], group: &[usize]) -> Vec<usize> {
    (0..rows)
        .map(|r| {
            (0..n)
                .find(|p| (0..rows).any(|o| group[o] == group[r] && obs_patch[o * n + p]))
                .unwrap_or(0)
        })
        .collect()
}

/// Rotary tables shared by every row, or one per row.
enum Rotary {
    Shared((Tensor, Tensor)),
    PerRow(Vec<(Tensor, Tensor)>),
}

impl Rotary {
    /// The tables of `rows`, in that order.
    fn select(&self, rows: &[usize]) -> Rotary {
        match self {
            Rotary::Shared(t) => Rotary::Shared(t.clone()),
            Rotary::PerRow(tables) => {
                Rotary::PerRow(rows.iter().map(|r| tables[*r].clone()).collect())
            }
        }
    }

    /// Rotate `x` `[rows, heads, S, head_dim]`.
    fn apply(&self, x: &Tensor) -> tensor::Result<Tensor> {
        match self {
            Rotary::Shared((cos, sin)) => crate::tensor::ops::rope(x, cos, sin),
            Rotary::PerRow(tables) => {
                let parts = tables
                    .iter()
                    .enumerate()
                    .map(|(r, (cos, sin))| {
                        crate::tensor::ops::rope(&x.narrow(0, r, 1)?.contiguous()?, cos, sin)
                    })
                    .collect::<tensor::Result<Vec<_>>>()?;
                let refs: Vec<&Tensor> = parts.iter().collect();
                Tensor::cat(&refs, 0)
            }
        }
    }
}

/// Load the checkpoint in `dir` on `device`.
pub fn load_dir(dir: &std::path::Path, device: &Device) -> anyhow::Result<Forecaster> {
    let cfg = read_config(dir)?;
    let weights = crate::inference::model::chronos2::weight_files(dir)?;
    let vb = unsafe { VarBuilder::from_files(&weights, DType::F32, device)? };
    Ok(Forecaster::new(Network::load(&vb, cfg)?))
}

/// The configuration a checkpoint directory carries, when it describes this network.
pub fn read_config(dir: &std::path::Path) -> anyhow::Result<Config> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

#[cfg(test)]
mod tests;
