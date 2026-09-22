//! Loading the always-resident path of deepseek_v41 from a `WeightSource`.
//!
//! The loaders ask a source for tensors by the reference implementation's names and assemble the
//! phase structs: attention (ratio-0 or band), the MoE's gate, shared expert and routed-expert
//! loader, norms, hyper-connections, embed and head. Weights arrive block-quantised (fp8/fp4 or a
//! k-quant); the reference forward runs in f32, so what is resident is dequantised on load, while
//! the routed and shared experts are read in place when the source offers a view.
//!
//! Per layer `layers.{i}.`: `attn_norm.weight`, `ffn_norm.weight`, `hc_{attn,ffn}_{fn,scale,base}`,
//! `attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b}.weight`, `attn.attn_sink`,
//! `attn.compressor.{norm,wkv,wgate}.weight` and `attn.indexer.{wq_b,weights_proj,wk,k_norm}.weight`
//! on band layers, `ffn.gate.{weight,bias}`, `ffn.shared_experts.w{1,2,3}.weight`,
//! `ffn.experts.{e}.w{1,2,3}.weight`, `engram.{embed,wkv}.weight`, `engram.{q,k}_weight`.

use super::attention::Ratio0Attention;
use super::band::BandAttention;
use super::block::{Block, HcWeights, LayerAttn};
use super::moe::Moe;
use super::source::WeightSource;
use super::DeepseekV41Config;
use crate::inference::offload::experts::Expert;
use crate::inference::offload::projection::Projection;
use crate::inference::offload::store::{ExpertSet, ExpertStore};
use crate::tensor::quantized::gguf_source::GgufSource;
use crate::tensor::{DType, Device, Result, Tensor};

/// Load one GGUF tensor to an f32 CPU tensor, dequantised. The GGUF source's dense path.
pub fn load_f32<S: GgufSource>(g: &S, name: &str) -> Result<Tensor> {
    g.tensor(name, &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .to_dtype(DType::F32)
}

/// A tensor the source may not carry, dense f32 when it does.
fn dense_opt<S: WeightSource + ?Sized>(g: &S, name: &str) -> Result<Option<Tensor>> {
    if g.shape(name).is_some() {
        g.dense_f32(name).map(Some)
    } else {
        Ok(None)
    }
}

/// A projection read in place when the source offers it, dense f32 otherwise.
pub(super) fn projection<S: WeightSource + ?Sized>(g: &S, name: &str) -> Result<Projection> {
    match g.projection(name)? {
        Some(p) => Ok(p),
        None => Ok(Projection::Dense(g.dense_f32(name)?)),
    }
}

/// The always-resident attention weights of a ratio-0 layer.
pub fn load_ratio0_attention<S: WeightSource + ?Sized>(
    g: &S,
    layer: usize,
    cfg: &DeepseekV41Config,
) -> Result<Ratio0Attention> {
    let p = format!("layers.{layer}.attn");
    Ok(Ratio0Attention {
        wq_a: projection(g, &format!("{p}.wq_a.weight"))?,
        q_norm: g.dense_f32(&format!("{p}.q_norm.weight"))?,
        wq_b: projection(g, &format!("{p}.wq_b.weight"))?,
        wkv: projection(g, &format!("{p}.wkv.weight"))?,
        kv_norm: g.dense_f32(&format!("{p}.kv_norm.weight"))?,
        attn_sink: g.dense_f32(&format!("{p}.attn_sink"))?,
        wo_a: projection(g, &format!("{p}.wo_a.weight"))?,
        wo_b: projection(g, &format!("{p}.wo_b.weight"))?,
        n_heads: cfg.n_head,
        head_dim: cfg.head_dim,
        rope_head_dim: cfg.rope_head_dim,
        o_groups: cfg.o_groups,
        o_lora_rank: cfg.o_lora_rank,
        window_size: cfg.window_size,
        eps: cfg.rms_eps as f32,
    })
}

/// The MoE block: the always-resident gate and bias, the shared expert, and the routed experts
/// behind the source's loader - read in place where the storage allows, kept in a store that
/// holds every expert built (`cfg.expert_cache_count` bounds it when set).
pub fn load_moe<S: WeightSource + ?Sized>(
    g: &S,
    layer: usize,
    cfg: &DeepseekV41Config,
) -> Result<Moe> {
    let p = format!("layers.{layer}.ffn");
    let shared = Expert {
        w1: projection(g, &format!("{p}.shared_experts.w1.weight"))?,
        w2: projection(g, &format!("{p}.shared_experts.w2.weight"))?,
        w3: projection(g, &format!("{p}.shared_experts.w3.weight"))?,
    };
    let loader = g.experts(layer, cfg.n_routed_experts)?;
    Ok(Moe {
        gate_weight: g.dense_f32(&format!("{p}.gate.weight"))?,
        gate_bias: g.dense_f32(&format!("{p}.gate.bias"))?,
        experts: ExpertSet::Streamed(ExpertStore::new(loader, cfg.expert_cache_count)),
        shared,
        has_shared: true,
        n_routed: cfg.n_routed_experts,
        n_activated: cfg.n_activated_experts,
        dim: cfg.d_model,
        gate_temp: cfg.gate_temp,
        route_scale: cfg.route_scale,
        swiglu_limit: cfg
            .swiglu_limits
            .get(layer)
            .copied()
            .unwrap_or(cfg.swiglu_limit),
        norm_topk: cfg.norm_topk,
        last_routing: std::sync::Mutex::new(Vec::new()),
        observer: std::sync::RwLock::new(None),
        offload: std::sync::RwLock::new(None),
    })
}

/// A band layer's attention: the base projections of every layer, plus the compressor of a
/// kv-source and the indexer of an index-source.
pub fn load_band_attention<S: WeightSource + ?Sized>(
    g: &S,
    layer: usize,
    cfg: &DeepseekV41Config,
) -> Result<BandAttention> {
    use super::band::{Compressor, Indexer};
    let p = format!("layers.{layer}.attn");
    let ratio = cfg.compress_ratios.get(layer).copied().unwrap_or(0);
    let is_kv_source = cfg.kv_source_layers.contains(&layer);
    let is_index_source = cfg.index_source_layers.contains(&layer);
    let cand = cfg.candidate_source_layer;
    // A kv-source owns its compressor; an index-source owns its indexer; a reader has neither.
    let compressor = if is_kv_source {
        Some(Compressor {
            norm: g.dense_f32(&format!("{p}.compressor.norm.weight"))?,
            wkv: g.dense_f32(&format!("{p}.compressor.wkv.weight"))?,
            wgate: dense_opt(g, &format!("{p}.compressor.wgate.weight"))?,
            ratio,
            head_dim: cfg.head_dim,
            eps: cfg.rms_eps as f32,
        })
    } else {
        None
    };
    let indexer = if is_index_source {
        Some(Indexer {
            wq_b: projection(g, &format!("{p}.indexer.wq_b.weight"))?,
            weights_proj: g.dense_f32(&format!("{p}.indexer.weights_proj.weight"))?,
            wk: dense_opt(g, &format!("{p}.indexer.wk.weight"))?,
            k_norm: dense_opt(g, &format!("{p}.indexer.k_norm.weight"))?,
            n_heads: cfg.index_n_heads,
            index_head_dim: cfg.index_head_dim,
            rope_head_dim: cfg.rope_head_dim,
            index_topk: cfg.index_topk,
            eps: cfg.rms_eps as f32,
            owns_k: is_kv_source,
            is_candidate_source: cand >= 0 && cand as usize == layer,
            uses_candidates: cand >= 0 && (cand as usize) < layer,
            candidate_topk_blocks: cfg.candidate_topk_blocks,
            candidate_block_size: cfg.candidate_block_size,
        })
    } else {
        None
    };
    Ok(BandAttention {
        wq_a: projection(g, &format!("{p}.wq_a.weight"))?,
        q_norm: g.dense_f32(&format!("{p}.q_norm.weight"))?,
        wq_b: projection(g, &format!("{p}.wq_b.weight"))?,
        wkv: projection(g, &format!("{p}.wkv.weight"))?,
        kv_norm: g.dense_f32(&format!("{p}.kv_norm.weight"))?,
        attn_sink: g.dense_f32(&format!("{p}.attn_sink"))?,
        wo_a: projection(g, &format!("{p}.wo_a.weight"))?,
        wo_b: projection(g, &format!("{p}.wo_b.weight"))?,
        compressor,
        indexer,
        n_heads: cfg.n_head,
        head_dim: cfg.head_dim,
        rope_head_dim: cfg.rope_head_dim,
        o_groups: cfg.o_groups,
        o_lora_rank: cfg.o_lora_rank,
        window_size: cfg.window_size,
        ratio,
        index_topk: cfg.index_topk,
        eps: cfg.rms_eps as f32,
    })
}

/// A whole band block. Same MoE, norms and hyper-connections as a ratio-0 block; only the
/// attention differs.
pub fn load_band_block<S: WeightSource + ?Sized>(
    g: &S,
    layer: usize,
    cfg: &DeepseekV41Config,
) -> Result<Block> {
    let mut b = load_ratio0_block(g, layer, cfg)?;
    b.attn = LayerAttn::Band(load_band_attention(g, layer, cfg)?);
    Ok(b)
}

/// A whole ratio-0 block: attention, the MoE, the two norms and the hyper-connection coefficients.
pub fn load_ratio0_block<S: WeightSource + ?Sized>(
    g: &S,
    layer: usize,
    cfg: &DeepseekV41Config,
) -> Result<Block> {
    let p = format!("layers.{layer}");
    let hc = |kind: &str| -> Result<HcWeights> {
        Ok(HcWeights {
            func: g.dense_f32(&format!("{p}.hc_{kind}_fn"))?,
            scale: g.dense_f32(&format!("{p}.hc_{kind}_scale"))?,
            base: g.dense_f32(&format!("{p}.hc_{kind}_base"))?,
        })
    };
    Ok(Block {
        attn: LayerAttn::Ratio0(load_ratio0_attention(g, layer, cfg)?),
        moe: load_moe(g, layer, cfg)?,
        attn_norm: g.dense_f32(&format!("{p}.attn_norm.weight"))?,
        ffn_norm: g.dense_f32(&format!("{p}.ffn_norm.weight"))?,
        hc_attn: hc("attn")?,
        hc_ffn: hc("ffn")?,
        hc_mult: cfg.hc_mult,
        hc_sinkhorn_iters: cfg.hc_sinkhorn_iters,
        norm_eps: cfg.rms_eps as f32,
        hc_eps: cfg.hc_eps,
    })
}

#[cfg(test)]
mod tests {
    use super::super::band::SharedAttn;
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::open_mapped;
    use crate::tensor::quantized::gguf_file::Value;
    use crate::tensor::quantized::GgmlDType;

    fn f32e(name: &str, dims: Vec<usize>) -> GgufEntry {
        let n: usize = dims.iter().product();
        let data: Vec<u8> = (0..n)
            .flat_map(|i| (i as f32 * 0.001 - 0.1).to_le_bytes())
            .collect();
        GgufEntry {
            name: name.to_string(),
            dims,
            dtype: GgmlDType::F32,
            data,
        }
    }

    /// The loader reads a ratio-0 layer's attention from a GGUF by the name contract, assembles the
    /// struct at the config's shapes, and the assembled attention runs to a finite output. This
    /// gates the name mapping and loading plumbing; the numbers are gated by the attention fixture.
    #[test]
    fn loads_a_ratio0_attention_and_runs() {
        // The toy shapes the parity fixtures use.
        let (dim, nh, hd, rd, qlora, olora, ogroups) = (256usize, 4, 128, 32, 64, 64, 4);
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("deepseek_v41".into()),
            ),
            ("deepseek_v41.block_count".to_string(), Value::U32(1)),
            (
                "deepseek_v41.embedding_length".to_string(),
                Value::U32(dim as u32),
            ),
            (
                "deepseek_v41.attention.head_count".to_string(),
                Value::U32(nh as u32),
            ),
            (
                "deepseek_v41.attention.key_length".to_string(),
                Value::U32(hd as u32),
            ),
            ("deepseek_v41.expert_count".to_string(), Value::U32(8)),
        ];
        let entries = vec![
            f32e("blk.0.attn_q_a.weight", vec![qlora, dim]),
            f32e("blk.0.attn_q_a_norm.weight", vec![qlora]),
            f32e("blk.0.attn_q_b.weight", vec![nh * hd, qlora]),
            f32e("blk.0.attn_kv.weight", vec![hd, dim]),
            f32e("blk.0.attn_kv_norm.weight", vec![hd]),
            f32e("blk.0.attn_sink", vec![nh]),
            f32e(
                "blk.0.attn_o_a.weight",
                vec![ogroups * olora, nh * hd / ogroups],
            ),
            f32e("blk.0.attn_o_b.weight", vec![dim, ogroups * olora]),
            // a token_embd so the file is a plausible model, unused by this loader
            f32e("token_embd.weight", vec![1, dim]),
        ];
        let path =
            std::env::temp_dir().join(format!("loken-dsv41-load-{}.gguf", std::process::id()));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let mut cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        // The synthetic header only carries the shapes the parser requires; set the rest to the
        // toy's values the loader reads.
        cfg.rope_head_dim = rd;
        cfg.q_lora_rank = qlora;
        cfg.o_lora_rank = olora;
        cfg.o_groups = ogroups;
        cfg.window_size = 32;
        cfg.rms_eps = 1e-20;

        let attn = load_ratio0_attention(&g, 0, &cfg).unwrap();
        assert_eq!(attn.wq_b.dims(), &[nh * hd, qlora]);
        assert_eq!(attn.wo_a.dims(), &[ogroups * olora, nh * hd / ogroups]);

        // A no-rotation rope table (cos=1, sin=0), just to exercise the assembled forward.
        let s = 16usize;
        let cos = Tensor::from_vec(vec![1f32; s * rd / 2], (s, rd / 2), &Device::Cpu).unwrap();
        let sin = Tensor::from_vec(vec![0f32; s * rd / 2], (s, rd / 2), &Device::Cpu).unwrap();
        let x = Tensor::from_vec(
            (0..s * dim).map(|i| (i as f32 * 0.01).sin()).collect(),
            (1, s, dim),
            &Device::Cpu,
        )
        .unwrap();
        let out = attn.forward_prefill(&x, &cos, &sin).unwrap();
        assert_eq!(out.dims(), &[1, s, dim]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "assembled attention produced non-finite output"
        );
    }

    /// The loader reads a whole ratio-0 block - attention, MoE (gate, shared expert, the stacked
    /// routed experts), norms and hyper-connections - by the name contract, and the assembled
    /// block runs to a finite hc-copy stream. Numbers stay gated by the block parity fixture.
    #[test]
    fn loads_a_ratio0_block_and_runs() {
        let (dim, nh, hd, rd, qlora, olora, og) = (256usize, 4, 128, 32, 64, 64, 4);
        let (er, inter, hc, mix) = (8usize, 256usize, 4usize, 24usize);
        let u = |k: &str, v: u32| (format!("deepseek_v41.{k}"), Value::U32(v));
        let f = |k: &str, v: f32| (format!("deepseek_v41.{k}"), Value::F32(v));
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("deepseek_v41".into()),
            ),
            u("block_count", 1),
            u("embedding_length", dim as u32),
            u("attention.head_count", nh as u32),
            u("attention.key_length", hd as u32),
            u("expert_count", er as u32),
            u("expert_used_count", 2),
            u("expert_shared_count", 1),
            u("expert_feed_forward_length", inter as u32),
            f("expert_weights_scale", 1.5),
            f("expert_swiglu_limit", 10.0),
            u("hyper_connection_mult", hc as u32),
            u("hyper_connection_sinkhorn_iters", 20),
        ];
        let entries = vec![
            f32e("blk.0.attn_q_a.weight", vec![qlora, dim]),
            f32e("blk.0.attn_q_a_norm.weight", vec![qlora]),
            f32e("blk.0.attn_q_b.weight", vec![nh * hd, qlora]),
            f32e("blk.0.attn_kv.weight", vec![hd, dim]),
            f32e("blk.0.attn_kv_norm.weight", vec![hd]),
            f32e("blk.0.attn_sink", vec![nh]),
            f32e("blk.0.attn_o_a.weight", vec![og * olora, nh * hd / og]),
            f32e("blk.0.attn_o_b.weight", vec![dim, og * olora]),
            f32e("blk.0.attn_norm.weight", vec![dim]),
            f32e("blk.0.ffn_norm.weight", vec![dim]),
            f32e("blk.0.ffn_gate_inp.weight", vec![er, dim]),
            f32e("blk.0.ffn_gate_inp.bias", vec![er]),
            f32e("blk.0.ffn_gate_shexp.weight", vec![inter, dim]),
            f32e("blk.0.ffn_up_shexp.weight", vec![inter, dim]),
            f32e("blk.0.ffn_down_shexp.weight", vec![dim, inter]),
            f32e("blk.0.ffn_gate_exps.weight", vec![er, inter, dim]),
            f32e("blk.0.ffn_up_exps.weight", vec![er, inter, dim]),
            f32e("blk.0.ffn_down_exps.weight", vec![er, dim, inter]),
            f32e("blk.0.hc_attn_fn.weight", vec![mix, hc * dim]),
            f32e("blk.0.hc_attn_scale", vec![3]),
            f32e("blk.0.hc_attn_base", vec![mix]),
            f32e("blk.0.hc_ffn_fn.weight", vec![mix, hc * dim]),
            f32e("blk.0.hc_ffn_scale", vec![3]),
            f32e("blk.0.hc_ffn_base", vec![mix]),
            f32e("token_embd.weight", vec![1, dim]),
        ];
        let path =
            std::env::temp_dir().join(format!("loken-dsv41-blk-{}.gguf", std::process::id()));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let mut cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        cfg.rope_head_dim = rd;
        cfg.q_lora_rank = qlora;
        cfg.o_lora_rank = olora;
        cfg.o_groups = og;
        cfg.window_size = 32;
        cfg.rms_eps = 1e-20;
        assert_eq!(cfg.moe_inter_dim, inter);
        assert!(cfg.norm_topk);

        let block = load_ratio0_block(&g, 0, &cfg).unwrap();
        assert_eq!(block.moe.experts.count(), er);
        assert_eq!(
            block.moe.experts.get(0).unwrap().w1.dims(),
            vec![inter, dim]
        );
        assert_eq!(block.hc_attn.func.dims(), &[mix, hc * dim]);

        // Run the assembled block: an hc-copy stream in, identity pre-mix, no-rotation rope.
        let s = 16usize;
        let x = Tensor::from_vec(
            (0..s * hc * dim)
                .map(|i| (i as f32 * 0.01).sin() * 0.1)
                .collect(),
            (1, s, hc, dim),
            &Device::Cpu,
        )
        .unwrap();
        let pre_mix: Vec<Vec<f32>> = (0..s).map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect();
        let cos = Tensor::from_vec(vec![1f32; s * rd / 2], (s, rd / 2), &Device::Cpu).unwrap();
        let sin = Tensor::from_vec(vec![0f32; s * rd / 2], (s, rd / 2), &Device::Cpu).unwrap();

        let mut shared = SharedAttn::default();
        let (out, ffn_pre) = block
            .forward_prefill(&x, &pre_mix, &cos, &sin, &mut shared)
            .unwrap();
        assert_eq!(out.dims(), &[1, s, hc, dim]);
        assert_eq!(ffn_pre.len(), s);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "assembled block produced non-finite output"
        );
    }

    /// A self-sourcing band block (ratio 2, kv + index source) loads its compressor and indexer by
    /// the name contract and runs to a finite hc-copy stream.
    #[test]
    fn loads_a_band_block_and_runs() {
        let (dim, nh, hd, rd, qlora, olora, og) = (256usize, 4, 128, 32, 64, 64, 4);
        let (er, inter, hc, mix) = (8usize, 256usize, 4usize, 24usize);
        let (inh, ihd, itopk) = (4usize, 64usize, 16usize);
        let u = |k: &str, v: u32| (format!("deepseek_v41.{k}"), Value::U32(v));
        let f = |k: &str, v: f32| (format!("deepseek_v41.{k}"), Value::F32(v));
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("deepseek_v41".into()),
            ),
            u("block_count", 1),
            u("embedding_length", dim as u32),
            u("attention.head_count", nh as u32),
            u("attention.key_length", hd as u32),
            u("expert_count", er as u32),
            u("expert_used_count", 2),
            u("expert_shared_count", 1),
            u("expert_feed_forward_length", inter as u32),
            f("expert_weights_scale", 1.5),
            f("expert_swiglu_limit", 10.0),
            u("hyper_connection_mult", hc as u32),
            u("hyper_connection_sinkhorn_iters", 20),
            (
                "deepseek_v41.attention.compress_ratios".to_string(),
                Value::Array(vec![Value::U32(2)]),
            ),
            (
                "deepseek_v41.attention.kv_source_layers".to_string(),
                Value::Array(vec![Value::U32(0)]),
            ),
            (
                "deepseek_v41.attention.index_source_layers".to_string(),
                Value::Array(vec![Value::U32(0)]),
            ),
            u("attention.index_head_count", inh as u32),
            u("attention.index_key_length", ihd as u32),
            u("attention.index_topk", itopk as u32),
        ];
        let base = vec![
            f32e("blk.0.attn_q_a.weight", vec![qlora, dim]),
            f32e("blk.0.attn_q_a_norm.weight", vec![qlora]),
            f32e("blk.0.attn_q_b.weight", vec![nh * hd, qlora]),
            f32e("blk.0.attn_kv.weight", vec![hd, dim]),
            f32e("blk.0.attn_kv_norm.weight", vec![hd]),
            f32e("blk.0.attn_sink", vec![nh]),
            f32e("blk.0.attn_o_a.weight", vec![og * olora, nh * hd / og]),
            f32e("blk.0.attn_o_b.weight", vec![dim, og * olora]),
            f32e("blk.0.attn_norm.weight", vec![dim]),
            f32e("blk.0.ffn_norm.weight", vec![dim]),
            f32e("blk.0.ffn_gate_inp.weight", vec![er, dim]),
            f32e("blk.0.ffn_gate_inp.bias", vec![er]),
            f32e("blk.0.ffn_gate_shexp.weight", vec![inter, dim]),
            f32e("blk.0.ffn_up_shexp.weight", vec![inter, dim]),
            f32e("blk.0.ffn_down_shexp.weight", vec![dim, inter]),
            f32e("blk.0.ffn_gate_exps.weight", vec![er, inter, dim]),
            f32e("blk.0.ffn_up_exps.weight", vec![er, inter, dim]),
            f32e("blk.0.ffn_down_exps.weight", vec![er, dim, inter]),
            f32e("blk.0.hc_attn_fn.weight", vec![mix, hc * dim]),
            f32e("blk.0.hc_attn_scale", vec![3]),
            f32e("blk.0.hc_attn_base", vec![mix]),
            f32e("blk.0.hc_ffn_fn.weight", vec![mix, hc * dim]),
            f32e("blk.0.hc_ffn_scale", vec![3]),
            f32e("blk.0.hc_ffn_base", vec![mix]),
            f32e("token_embd.weight", vec![1, dim]),
        ];
        let extra = vec![
            f32e("blk.0.attn_compressor_norm.weight", vec![hd]),
            f32e("blk.0.attn_compressor_kv.weight", vec![hd, dim]),
            f32e("blk.0.attn_compressor_gate.weight", vec![hd, dim]),
            f32e("blk.0.attn_indexer_q_b.weight", vec![inh * ihd, qlora]),
            f32e("blk.0.attn_indexer_weights.weight", vec![inh, dim]),
            f32e("blk.0.attn_indexer_k.weight", vec![ihd, hd]),
            f32e("blk.0.attn_indexer_k_norm.weight", vec![ihd]),
        ];
        let entries: Vec<_> = base.into_iter().chain(extra).collect();
        let path =
            std::env::temp_dir().join(format!("loken-dsv41-band-{}.gguf", std::process::id()));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let mut cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
        cfg.rope_head_dim = rd;
        cfg.q_lora_rank = qlora;
        cfg.o_lora_rank = olora;
        cfg.o_groups = og;
        cfg.window_size = 32;
        cfg.rms_eps = 1e-20;
        assert_eq!(cfg.compress_ratios, vec![2]);
        assert_eq!(cfg.index_topk, itopk);

        let block = load_band_block(&g, 0, &cfg).unwrap();
        assert!(matches!(block.attn, LayerAttn::Band(_)));

        let s = 64usize;
        let x = Tensor::from_vec(
            (0..s * hc * dim)
                .map(|i| (i as f32 * 0.01).sin() * 0.1)
                .collect(),
            (1, s, hc, dim),
            &Device::Cpu,
        )
        .unwrap();
        let pre_mix: Vec<Vec<f32>> = (0..s).map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect();
        let cos = Tensor::from_vec(vec![1f32; s * rd / 2], (s, rd / 2), &Device::Cpu).unwrap();
        let sin = Tensor::from_vec(vec![0f32; s * rd / 2], (s, rd / 2), &Device::Cpu).unwrap();

        let mut shared = SharedAttn::default();
        let (out, _pre) = block
            .forward_prefill(&x, &pre_mix, &cos, &sin, &mut shared)
            .unwrap();
        assert_eq!(out.dims(), &[1, s, hc, dim]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "assembled band block produced non-finite output"
        );
    }
}
