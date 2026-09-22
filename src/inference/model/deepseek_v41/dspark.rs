//! DSpark: the checkpoint's native multi-token draft, stored under `mtp.0/1/2`. Three stages read
//! the backbone's attention input at the target layers, mix it into a block of `block_size` draft
//! positions, and predict that block in one pass. A speculative decode drafts with it and verifies
//! the block against the backbone. The reference is `inference/model.py` (`forward_spec`,
//! `DSparkBlock`, `DSparkAttention`), which defines the forward but does not wire it.

use super::block::HcWeights;
use super::moe::Moe;
use super::safetensors_source::SafeTensorsSource;
use super::source::WeightSource;
use crate::inference::offload::experts::Expert;
use crate::inference::offload::projection::Projection;
use crate::inference::offload::store::{ExpertSet, ExpertStore};
use crate::tensor::{Error, Result, Tensor};
use std::path::{Path, PathBuf};

/// The fixed shape of the draft, read from the checkpoint config.
pub struct DsparkConfig {
    pub block_size: usize,
    pub noise_token: u32,
    pub target_layers: Vec<usize>,
    pub markov_rank: usize,
    pub n_routed: usize,
    pub n_activated: usize,
    pub window_size: usize,
    pub dim: usize,
    pub hc_mult: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub gate_temp: f32,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub norm_topk: bool,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub original_seq_len: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub vocab: usize,
}

/// One DSpark stage's attention: the same MLA projections a band layer has, but its keys and values
/// come from the backbone's hidden (`main_x`) written into a sliding window, and the block's own
/// queries attend over that window.
pub struct DsparkAttn {
    pub wq_a: Projection,
    pub q_norm: Tensor,
    pub wq_b: Projection,
    pub wkv: Projection,
    pub kv_norm: Tensor,
    pub attn_sink: Tensor,
    pub wo_a: Projection,
    pub wo_b: Projection,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub eps: f32,
}

/// A DSpark stage: attention over the backbone window, then its own routed experts, each wrapped in
/// the same hyper-connections a backbone block uses.
pub struct DsparkStage {
    pub attn: DsparkAttn,
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub moe: Moe,
    pub hc_attn: HcWeights,
    pub hc_ffn: HcWeights,
}

/// The full DSpark draft head: the three stages, the projection that reads the backbone hidden into
/// the stages' input, and the markov head that refines the block position by position. The backbone
/// embedding and output head are shared, passed in.
pub struct Dspark {
    pub stages: Vec<DsparkStage>,
    pub main_proj: Projection,
    pub main_norm: Tensor,
    pub final_norm: Tensor,
    pub markov_embed: Tensor, // [vocab, markov_rank]
    pub markov_head: Projection,
    pub cfg: DsparkConfig,
}

fn proj(g: &SafeTensorsSource, name: &str) -> Result<Projection> {
    g.projection(name)?
        .map(Ok)
        .unwrap_or_else(|| Ok(Projection::Dense(g.dense_f32(name)?)))
}

fn hc(g: &SafeTensorsSource, stage: usize, which: &str) -> Result<HcWeights> {
    Ok(HcWeights {
        func: g.dense_f32(&format!("mtp.{stage}.hc_{which}_fn"))?,
        scale: g.dense_f32(&format!("mtp.{stage}.hc_{which}_scale"))?,
        base: g.dense_f32(&format!("mtp.{stage}.hc_{which}_base"))?,
    })
}

fn attn(g: &SafeTensorsSource, stage: usize, cfg: &DsparkConfig) -> Result<DsparkAttn> {
    let p = format!("mtp.{stage}.attn");
    Ok(DsparkAttn {
        wq_a: proj(g, &format!("{p}.wq_a.weight"))?,
        q_norm: g.dense_f32(&format!("{p}.q_norm.weight"))?,
        wq_b: proj(g, &format!("{p}.wq_b.weight"))?,
        wkv: proj(g, &format!("{p}.wkv.weight"))?,
        kv_norm: g.dense_f32(&format!("{p}.kv_norm.weight"))?,
        attn_sink: g.dense_f32(&format!("{p}.attn_sink"))?,
        wo_a: proj(g, &format!("{p}.wo_a.weight"))?,
        wo_b: proj(g, &format!("{p}.wo_b.weight"))?,
        n_heads: cfg.n_heads,
        head_dim: cfg.head_dim,
        rope_head_dim: cfg.rope_head_dim,
        o_groups: cfg.o_groups,
        o_lora_rank: cfg.o_lora_rank,
        window_size: cfg.window_size,
        eps: cfg.rms_eps,
    })
}

fn stage(g: &SafeTensorsSource, s: usize, cfg: &DsparkConfig) -> Result<DsparkStage> {
    let p = format!("mtp.{s}.ffn");
    let loader = g.experts_named(&format!("{p}.experts"), cfg.n_routed, false);
    let moe = Moe {
        gate_weight: g.dense_f32(&format!("{p}.gate.weight"))?,
        gate_bias: g.dense_f32(&format!("{p}.gate.bias"))?,
        experts: ExpertSet::Streamed(ExpertStore::new(loader, 0)),
        // A stage has no shared expert; a zero projection leaves the shared contribution at nothing.
        shared: Expert {
            w1: proj(g, &format!("mtp.{s}.attn.wkv.weight"))?,
            w2: proj(g, &format!("mtp.{s}.attn.wkv.weight"))?,
            w3: proj(g, &format!("mtp.{s}.attn.wkv.weight"))?,
        },
        has_shared: false,
        n_routed: cfg.n_routed,
        n_activated: cfg.n_activated,
        dim: cfg.dim,
        gate_temp: cfg.gate_temp,
        route_scale: cfg.route_scale,
        swiglu_limit: cfg.swiglu_limit,
        norm_topk: cfg.norm_topk,
        last_routing: std::sync::Mutex::new(Vec::new()),
        observer: std::sync::RwLock::new(None),
        offload: std::sync::RwLock::new(None),
    };
    Ok(DsparkStage {
        attn: attn(g, s, cfg)?,
        attn_norm: g.dense_f32(&format!("mtp.{s}.attn_norm.weight"))?,
        ffn_norm: g.dense_f32(&format!("mtp.{s}.ffn_norm.weight"))?,
        moe,
        hc_attn: hc(g, s, "attn")?,
        hc_ffn: hc(g, s, "ffn")?,
    })
}

impl DsparkConfig {
    /// Read the draft's shape from the checkpoint's `config.json` (`text_config`).
    pub fn from_config_json(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| Error::msg(format!("dspark config.json: {e}")))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| Error::msg(format!("dspark config: {e}")))?;
        let t = v.get("text_config").unwrap_or(&v);
        let u = |k: &str| -> Result<usize> {
            t.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| Error::msg(format!("dspark config: missing {k}")))
        };
        let f = |k: &str, d: f32| {
            t.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(d)
        };
        let target_layers = t
            .get("dspark_target_layer_ids")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        Ok(DsparkConfig {
            block_size: u("dspark_block_size")?,
            noise_token: u("dspark_noise_token_id")? as u32,
            target_layers,
            markov_rank: u("dspark_markov_rank")?,
            n_routed: u("dspark_n_routed_experts")?,
            n_activated: u("dspark_num_experts_per_tok")?,
            window_size: u("sliding_window")?,
            dim: u("hidden_size")?,
            hc_mult: u("hc_mult")?,
            n_heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            rope_head_dim: u("qk_rope_head_dim")?,
            o_groups: u("o_groups")?,
            o_lora_rank: u("o_lora_rank")?,
            gate_temp: f("gate_temp", 1.0),
            route_scale: f("routed_scaling_factor", 1.0),
            swiglu_limit: f("swiglu_limit", 0.0),
            norm_topk: t
                .get("norm_topk_prob")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            rms_eps: f("rms_norm_eps", 1e-6),
            rope_theta: f("rope_theta", 10000.0),
            rope_factor: t
                .get("rope_scaling")
                .and_then(|r| r.get("factor"))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1.0),
            original_seq_len: t
                .get("rope_scaling")
                .and_then(|r| r.get("original_max_position_embeddings"))
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(0),
            hc_sinkhorn_iters: u("hc_sinkhorn_iters").unwrap_or(20),
            hc_eps: f("hc_eps", 1e-6),
            vocab: u("vocab_size")?,
        })
    }
}

/// The released DeepSeek-V4.1 checkpoint under the Hugging Face hub cache, if it is on this machine:
/// its safetensors hold the DSpark weights the served GGUF does not. Searched under the standard
/// hub-cache roots (HF_HOME, the hub-cache env vars) rather than any fixed path.
pub fn find_source() -> Option<(SafeTensorsSource, PathBuf)> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(h) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(h));
    }
    if let Ok(h) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        roots.push(PathBuf::from(h));
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        roots.push(Path::new(&h).join("hub"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(Path::new(&home).join(".cache/huggingface/hub"));
    }
    for root in roots {
        let snaps = root.join("models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots");
        let Ok(rd) = std::fs::read_dir(&snaps) else {
            continue;
        };
        for e in rd.flatten() {
            let dir = e.path();
            if dir.join("config.json").exists() {
                if let Ok(s) = SafeTensorsSource::open(&dir) {
                    return Some((s, dir));
                }
            }
        }
    }
    None
}

use super::attention::{act_quant_fp8_e4m3, rope_partial};
use super::band::sparse_attn;
use super::hyper_connections::{hc_mixes, hc_post, hc_pre};
use super::model::rope_table;
use crate::tensor::ops::rms_norm;
use crate::tensor::Device;

/// The draft's running state: a window of the backbone's recent key/values per stage, written every
/// decode step, that the draft's queries attend over. Kept beside the backbone's decode state.
pub struct DsparkState {
    /// `[stage][slot]` one rope'd kv row (`head_dim` floats); `slot = position % window_size`.
    pub windows: Vec<Vec<Vec<f32>>>,
    pub seen: usize,
}

impl DsparkState {
    pub fn new(n_stages: usize, window: usize, head_dim: usize) -> Self {
        DsparkState {
            windows: (0..n_stages)
                .map(|_| vec![vec![0f32; head_dim]; window.max(1)])
                .collect(),
            seen: 0,
        }
    }
}

impl DsparkAttn {
    /// The grouped output projection, as a band layer's: split the heads into `o_groups`, each
    /// through its own slice of `wo_a`, then all through `wo_b`.
    fn grouped_out(&self, o: &Tensor, s: usize, dim: usize) -> Result<Tensor> {
        let p = self.n_heads * self.head_dim / self.o_groups;
        let og = o.reshape((s, self.o_groups, p))?;
        let mut parts = Vec::with_capacity(self.o_groups);
        for g in 0..self.o_groups {
            let slice = og.narrow(1, g, 1)?.reshape((s, p))?;
            let wa = self.wo_a.rows(g * self.o_lora_rank, self.o_lora_rank)?;
            parts.push(wa.apply(&slice)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let o = Tensor::cat(&refs, 1)?;
        self.wo_b.apply(&o)?.reshape((1, s, dim))
    }
}

impl Dspark {
    /// One decode step's draft: update the stage windows from the backbone hidden, then predict the
    /// next `block_size` tokens. `main_hidden` is the concatenated target-layer hidden, `token` the
    /// token just committed, `start_pos` its position. Returns `block_size` draft tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_draft(
        &self,
        st: &mut DsparkState,
        main_hidden: &[f32],
        token: u32,
        start_pos: usize,
        embed: &Tensor,
        head: &Projection,
    ) -> Result<Vec<u32>> {
        let c = &self.cfg;
        let (dim, hc, rd, win) = (c.dim, c.hc_mult, c.rope_head_dim, c.window_size);
        let bs = c.block_size;
        let eps = c.rms_eps;
        // main_x = main_norm(main_proj(main_hidden)) - the backbone hidden read into the stages.
        let mh = Tensor::from_vec(
            main_hidden.to_vec(),
            (1, dim * c.target_layers.len()),
            &Device::Cpu,
        )?;
        let main_x = rms_norm(&self.main_proj.apply(&mh)?, &self.main_norm, eps)?; // [1, dim]
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static DUMPED: AtomicBool = AtomicBool::new(false);
            if !DUMPED.swap(true, Ordering::Relaxed) {
                let d = "/mnt/data/deepseek-v41-study/wsmeasure";
                let w = |name: &str, v: &[f32]| {
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = std::fs::write(format!("{d}/{name}.bin"), bytes);
                };
                w("dbg_main_hidden", main_hidden);
                w("dbg_main_x", &main_x.flatten_all()?.to_vec1::<f32>()?);
                w("dbg_token", &[token as f32, start_pos as f32]);
                tracing::info!(
                    "DSPARK-DUMP main_hidden+main_x written, token={token} pos={start_pos}"
                );
            }
        }

        // Rope tables long enough for the window positions and this block.
        let n = start_pos + 1 + bs;
        let (cos, sin) = rope_table(rd, n, c.original_seq_len, c.rope_theta, c.rope_factor)?;

        self.update_windows(st, &main_x, start_pos, &cos, &sin)?;
        let coverage = win.min(start_pos + 1);

        // The draft block: the committed token, then the noise placeholder for the rest.
        let mut ids = vec![c.noise_token; bs];
        ids[0] = token;
        let mut rows = Vec::with_capacity(bs);
        for &t in &ids {
            rows.push(embed.narrow(0, t as usize, 1)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let emb = Tensor::cat(&refs, 0)?.reshape((1, bs, dim))?;
        let mut x = emb
            .unsqueeze(2)?
            .broadcast_as((1, bs, hc, dim))?
            .contiguous()?;
        let mut pre_mix: Vec<Vec<f32>> = (0..bs)
            .map(|_| {
                let mut v = vec![0f32; hc];
                v[0] = 1.0;
                v
            })
            .collect();

        let _t_stages = std::time::Instant::now();
        let mut _t_attn = 0u128;
        let mut _t_moe = 0u128;
        for (s, stage) in self.stages.iter().enumerate() {
            let (nx, ta, tm) = self.stage_forward_timed(
                stage,
                &x,
                &mut pre_mix,
                s,
                st,
                start_pos,
                coverage,
                &cos,
                &sin,
            )?;
            x = nx;
            _t_attn += ta;
            _t_moe += tm;
        }
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed).is_multiple_of(8) {
                tracing::info!(
                    "DSPARK-PROF stages={}us attn={}us moe={}us",
                    _t_stages.elapsed().as_micros(),
                    _t_attn,
                    _t_moe
                );
            }
        }

        // Collapse and read the block's logits, then refine each position with the markov head and
        // sample the block.
        let h = hc_pre(&x, &pre_mix)?;
        let h = rms_norm(&h, &self.final_norm, eps)?;
        let logits = head.apply(&h.reshape((bs, dim))?)?; // [bs, vocab]
        let mut logits = logits.to_vec2::<f32>()?;
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static D2: AtomicBool = AtomicBool::new(false);
            if !D2.swap(true, Ordering::Relaxed) {
                let d = "/mnt/data/deepseek-v41-study/wsmeasure";
                let w = |name: String, v: &[f32]| {
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = std::fs::write(format!("{d}/{name}.bin"), bytes);
                };
                for (s, win) in st.windows.iter().enumerate() {
                    w(format!("dbg_window{s}"), &win.concat());
                }
                w("dbg_rawlogits0".into(), &logits[0]);
                w("dbg_rawlogits1".into(), &logits[1]);
                w(
                    "dbg_argmax".into(),
                    &logits.iter().map(|l| argmax(l) as f32).collect::<Vec<_>>(),
                );
                tracing::info!(
                    "DSPARK-DUMP2 windows+rawlogits written; block argmax before markov = {:?}",
                    logits.iter().map(|l| argmax(l)).collect::<Vec<_>>()
                );
            }
        }
        let mut out = Vec::with_capacity(bs);
        let mut prev = token;
        for i in 0..bs {
            // markov bias: head(embed(prev)) added to this position's logits.
            let e = self.markov_embed.narrow(0, prev as usize, 1)?; // [1, rank]
            let bias = self
                .markov_head
                .apply(&e)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (l, b) in logits[i].iter_mut().zip(&bias) {
                *l += *b;
            }
            let next = argmax(&logits[i]);
            out.push(next);
            prev = next;
        }
        Ok(out)
    }

    /// Write this position's backbone kv into each stage's window. Called every decode step so the
    /// window holds the recent `window_size` positions the draft attends over.
    fn update_windows(
        &self,
        st: &mut DsparkState,
        main_x: &Tensor,
        start_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<()> {
        let c = &self.cfg;
        let (rd, win) = (c.rope_head_dim, c.window_size);
        let cos1 = cos.narrow(0, start_pos, 1)?;
        let sin1 = sin.narrow(0, start_pos, 1)?;
        for (s, stage) in self.stages.iter().enumerate() {
            let a = &stage.attn;
            let kv = rms_norm(&a.wkv.apply(main_x)?, &a.kv_norm, c.rms_eps)?;
            let kv = kv.reshape((1, 1, 1, a.head_dim))?;
            let kv = rope_partial(&kv, &cos1, &sin1, rd)?;
            let kv = act_quant_fp8_e4m3(&kv, 32)?;
            st.windows[s][start_pos % win] = kv.flatten_all()?.to_vec1::<f32>()?;
        }
        st.seen = (start_pos + 1).max(st.seen);
        Ok(())
    }

    /// Maintain the windows from the backbone hidden captured at a committed position, without
    /// drafting - the per-step upkeep the draft relies on.
    pub fn update_window_from_hidden(
        &self,
        st: &mut DsparkState,
        main_hidden: &[f32],
        start_pos: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let mh = Tensor::from_vec(
            main_hidden.to_vec(),
            (1, c.dim * c.target_layers.len()),
            &Device::Cpu,
        )?;
        let main_x = rms_norm(&self.main_proj.apply(&mh)?, &self.main_norm, c.rms_eps)?;
        let n = start_pos + 1;
        let (cos, sin) = rope_table(
            c.rope_head_dim,
            n,
            c.original_seq_len,
            c.rope_theta,
            c.rope_factor,
        )?;
        self.update_windows(st, &main_x, start_pos, &cos, &sin)
    }

    /// `stage_forward` with attention and MoE timed, for the draft profile.
    #[allow(clippy::too_many_arguments)]
    fn stage_forward_timed(
        &self,
        stage: &DsparkStage,
        x: &Tensor,
        pre_mix: &mut Vec<Vec<f32>>,
        s: usize,
        st: &DsparkState,
        start_pos: usize,
        coverage: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, u128, u128)> {
        let c = &self.cfg;
        let eps = c.rms_eps;
        let am = hc_mixes(
            x,
            &stage.hc_attn.func,
            &stage.hc_attn.scale.flatten_all()?.to_vec1::<f32>()?,
            &stage.hc_attn.base.flatten_all()?.to_vec1::<f32>()?,
            c.hc_mult,
            c.hc_sinkhorn_iters,
            eps,
            c.hc_eps,
        )?;
        let xin = rms_norm(&hc_pre(x, pre_mix)?, &stage.attn_norm, eps)?;
        let t = std::time::Instant::now();
        let xattn = self.attn_forward(&stage.attn, &xin, s, st, start_pos, coverage, cos, sin)?;
        let ta = t.elapsed().as_micros();
        let x = hc_post(&xattn, x, &am.post, &am.comb, c.hc_mult)?;
        *pre_mix = am.pre;
        let fm = hc_mixes(
            &x,
            &stage.hc_ffn.func,
            &stage.hc_ffn.scale.flatten_all()?.to_vec1::<f32>()?,
            &stage.hc_ffn.base.flatten_all()?.to_vec1::<f32>()?,
            c.hc_mult,
            c.hc_sinkhorn_iters,
            eps,
            c.hc_eps,
        )?;
        let xin = rms_norm(&hc_pre(&x, pre_mix)?, &stage.ffn_norm, eps)?;
        let t = std::time::Instant::now();
        let xffn = stage.moe.forward(&xin)?;
        let tm = t.elapsed().as_micros();
        let out = hc_post(&xffn, &x, &fm.post, &fm.comb, c.hc_mult)?;
        *pre_mix = fm.pre;
        Ok((out, ta, tm))
    }

    /// One stage: attention over the backbone window (from `main_x`) plus the block's own keys, then
    /// the stage's routed experts, each wrapped in hyper-connections.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn stage_forward(
        &self,
        stage: &DsparkStage,
        x: &Tensor,
        pre_mix: &mut Vec<Vec<f32>>,
        s: usize,
        st: &DsparkState,
        start_pos: usize,
        coverage: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let eps = c.rms_eps;
        // Attention half.
        let am = hc_mixes(
            x,
            &stage.hc_attn.func,
            &stage.hc_attn.scale.flatten_all()?.to_vec1::<f32>()?,
            &stage.hc_attn.base.flatten_all()?.to_vec1::<f32>()?,
            c.hc_mult,
            c.hc_sinkhorn_iters,
            eps,
            c.hc_eps,
        )?;
        let xin = rms_norm(&hc_pre(x, pre_mix)?, &stage.attn_norm, eps)?; // [1, bs, dim]
        let xattn = self.attn_forward(&stage.attn, &xin, s, st, start_pos, coverage, cos, sin)?;
        let x = hc_post(&xattn, x, &am.post, &am.comb, c.hc_mult)?;
        *pre_mix = am.pre;
        // FFN half.
        let fm = hc_mixes(
            &x,
            &stage.hc_ffn.func,
            &stage.hc_ffn.scale.flatten_all()?.to_vec1::<f32>()?,
            &stage.hc_ffn.base.flatten_all()?.to_vec1::<f32>()?,
            c.hc_mult,
            c.hc_sinkhorn_iters,
            eps,
            c.hc_eps,
        )?;
        let xin = rms_norm(&hc_pre(&x, pre_mix)?, &stage.ffn_norm, eps)?;
        let xffn = stage.moe.forward(&xin)?;
        let out = hc_post(&xffn, &x, &fm.post, &fm.comb, c.hc_mult)?;
        *pre_mix = fm.pre;
        Ok(out)
    }

    /// The stage's windowed attention: queries from the block, keys/values from the stage window
    /// (the backbone's recent kv) plus the block's own kv.
    #[allow(clippy::too_many_arguments)]
    fn attn_forward(
        &self,
        a: &DsparkAttn,
        xin: &Tensor,
        s: usize,
        st: &DsparkState,
        start_pos: usize,
        coverage: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let (rd, win, bs) = (c.rope_head_dim, c.window_size, c.block_size);
        let (nh, hd) = (a.n_heads, a.head_dim);
        let x2 = xin.reshape((bs, c.dim))?;
        // Queries for the block positions, rope'd at start_pos + 1 + i.
        let qr = rms_norm(&a.wq_a.apply(&x2)?, &a.q_norm, a.eps)?;
        let q = a.wq_b.apply(&qr)?.reshape((1, bs, nh, hd))?;
        let qcos = cos.narrow(0, start_pos + 1, bs)?;
        let qsin = sin.narrow(0, start_pos + 1, bs)?;
        let q = rope_partial(&q, &qcos, &qsin, rd)?;
        // The block's own keys, rope'd at the same positions.
        let kv = rms_norm(&a.wkv.apply(&x2)?, &a.kv_norm, a.eps)?.reshape((1, bs, 1, hd))?;
        let kv =
            act_quant_fp8_e4m3(&rope_partial(&kv, &qcos, &qsin, rd)?, 32)?.reshape((bs, hd))?;
        // Concatenate the window (coverage rows, ring order 0..coverage) and the block keys.
        let mut kvbuf: Vec<f32> = Vec::with_capacity((win + bs) * hd);
        for slot in 0..win {
            kvbuf.extend_from_slice(&st.windows[s][slot]);
        }
        let kvblock = kv.flatten_all()?.to_vec1::<f32>()?;
        kvbuf.extend_from_slice(&kvblock);
        let n_kv = win + bs;
        let kv = Tensor::from_vec(kvbuf, (1, n_kv, hd), &Device::Cpu)?;
        // Each block position attends to the covered window rows and the whole block's keys.
        let idxs: Vec<i32> = {
            let mut base: Vec<i32> = (0..coverage as i32).collect();
            base.extend((win as i32)..(win as i32 + bs as i32));
            let topk = base.len();
            let mut all = Vec::with_capacity(bs * topk);
            for _ in 0..bs {
                all.extend_from_slice(&base);
            }
            all
        };
        let topk = coverage + bs;
        let sink = a.attn_sink.flatten_all()?.to_vec1::<f32>()?;
        let o = sparse_attn(&q, &kv, &sink, &idxs, topk, a.softmax_scale())?;
        let o = rope_partial(&o, &qcos, &qsin.neg()?, rd)?;
        a.grouped_out(&o, bs, c.dim)
    }
}

impl DsparkAttn {
    fn softmax_scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

impl Dspark {
    /// Open the checkpoint from the hub cache and load the draft, or `None` when it is not present.
    pub fn open() -> Option<Dspark> {
        let (src, dir) = find_source()?;
        let cfg = DsparkConfig::from_config_json(&dir).ok()?;
        match Dspark::load(&src, cfg) {
            Ok(d) => Some(d),
            Err(e) => {
                tracing::warn!("DSpark draft not loaded: {e}");
                None
            }
        }
    }

    /// Load the three stages and the shared draft heads from the released checkpoint.
    pub fn load(g: &SafeTensorsSource, cfg: DsparkConfig) -> Result<Self> {
        let n = 3;
        let mut stages = Vec::with_capacity(n);
        for s in 0..n {
            stages.push(stage(g, s, &cfg)?);
        }
        let last = n - 1;
        Ok(Dspark {
            stages,
            main_proj: proj(g, "mtp.0.main_proj.weight")?,
            main_norm: g.dense_f32("mtp.0.main_norm.weight")?,
            final_norm: g.dense_f32(&format!("mtp.{last}.norm.weight"))?,
            markov_embed: g.dense_f32(&format!("mtp.{last}.markov_head.embed.weight"))?,
            markov_head: proj(g, &format!("mtp.{last}.markov_head.head.weight"))?,
            cfg,
        })
    }
}
