//! Voxtral (audio->text) - the audio front-end: a Whisper-large-v3 encoder + a 2-layer GELU
//! projector that maps encoder frames into the Ministral-3B embedding space (3072).
//!
//! Loaded from the llama.cpp `mmproj-Voxtral-Mini-3B-*.gguf` (`clip`/`mmproj`, projector_type
//! "voxtral"). Tensors: `a.*` = the Whisper encoder, `mm.*` = the projector. The audio config
//! (from the GGUF metadata): embedding 1280, 32 blocks, 20 heads, ffn 5120, 128 mel bins,
//! projector stack_factor 4 -> projection_dim 3072.
//!
//! Whisper encoder (non-causal, learned position embedding, LayerNorm pre-norm, GELU MLP):
//!   mel [1,128,T] -> conv1(k3,p1,s1)+gelu -> conv2(k3,p1,s2)+gelu -> transpose -> +pos_embd[:T/2]
//!   -> 32x[ln1->MHA->res, ln2->FFN->res] -> post_ln -> [1, T/2, 1280].
//! Projector (Voxtral): stack 4 frames -> [T/8, 5120] -> mlp.1(5120->3072) -> gelu -> mlp.2(3072->3072).
//!
//! Self-contained on the native substrate (mirrors native_qwen25_vision's mmproj-dequant pattern):
//! all `a.*`/`mm.*` tensors are dequantized to F32 at load and the forward is explicit matmuls.
//! The native GGUF reader reverses ggml `ne`, so weights arrive `[out,in]`, conv `[out,in,k]`,
//! pos `[n_ctx,d]` - exactly the layout `lin`/`Conv1d` expect.

use crate::tensor::layer::{same_length_1d, Conv1d, Conv1dConfig};
use crate::tensor::{Device, Result, Tensor};

const DIM: usize = 1280; // audio hidden (d_model)
const HEADS: usize = 20;
const HD: usize = DIM / HEADS; // 64
const LAYERS: usize = 32;
const STACK: usize = 4; // projector frame stack
const EPS: f32 = 1e-5;

fn err(m: String) -> crate::tensor::Error {
    crate::tensor::Error(m)
}

/// `x [n,in] . wᵀ` (`w` is `[out,in]`) -> `[n,out]`, optional bias.
fn lin(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&w.transpose(0, 1)?.contiguous()?)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

fn layernorm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.layer_norm(w, Some(b), EPS)
}

struct Block {
    ln1_w: Tensor,
    ln1_b: Tensor,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
    o_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
}

impl Block {
    /// x `[seq, DIM]` (single sequence, non-causal full attention).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq = x.dim(0)?;
        // pre-norm attention
        let h = layernorm(x, &self.ln1_w, &self.ln1_b)?;
        let q = lin(&h, &self.q_w, Some(&self.q_b))?.reshape((seq, HEADS, HD))?;
        let k = lin(&h, &self.k_w, None)?.reshape((seq, HEADS, HD))?;
        let v = lin(&h, &self.v_w, Some(&self.v_b))?.reshape((seq, HEADS, HD))?;
        // [seq,H,HD] -> [H,seq,HD]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = 1.0f32 / (HD as f32).sqrt();
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)?
            .affine(scale, 0.0)?; // [H,seq,seq]
        let probs = scores.softmax_last_dim()?;
        let o = probs.matmul(&v)?; // [H,seq,HD]
        let o = o.transpose(0, 1)?.contiguous()?.reshape((seq, DIM))?; // [seq,DIM]
        let attn = lin(&o, &self.o_w, Some(&self.o_b))?;
        let x = x.add(&attn)?;
        // pre-norm FFN
        let h = layernorm(&x, &self.ln2_w, &self.ln2_b)?;
        let h = lin(&h, &self.up_w, Some(&self.up_b))?.gelu_erf()?;
        let h = lin(&h, &self.down_w, Some(&self.down_b))?;
        x.add(&h)
    }
}

pub struct VoxtralAudio {
    conv1: Conv1d,
    conv2: Conv1d,
    pos_embd: Tensor, // [NCTX, DIM]
    blocks: Vec<Block>,
    post_ln_w: Tensor,
    post_ln_b: Tensor,
    mm1_w: Tensor, // [PROJ, DIM*STACK]  (5120->3072)
    mm2_w: Tensor, // [PROJ, PROJ]
    dev: Device,
}

impl VoxtralAudio {
    pub fn load_mmproj(path: &str, dev: &Device) -> Result<Self> {
        use crate::tensor::quantized::gguf_file;
        use crate::tensor::DType as CDType;
        use crate::tensor::Device as CDevice;
        let mut f = std::fs::File::open(path).map_err(|e| err(format!("open {path}: {e}")))?;
        let content = gguf_file::read_mapped_file(&f).map_err(|e| err(format!("gguf: {e}")))?;
        // dequant a GGUF tensor -> (F32 values, native dims). The native reader reverses ggml `ne`,
        // so weights come back `[out,in]`, conv `[out,in,k]`, pos `[n_ctx,d]`.
        let mut dqd = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let t = content
                .tensor(&mut f, name, &CDevice::Cpu)
                .and_then(|t| t.dequantize(&CDevice::Cpu))
                .and_then(|t| t.to_dtype(CDType::F32))
                .map_err(|e| err(format!("dq {name}: {e}")))?;
            let dims = t.dims().to_vec();
            let v = t
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| err(format!("vec {name}: {e}")))?;
            Ok((v, dims))
        };
        let t2 = |v: Vec<f32>, r: usize, c: usize| -> Result<Tensor> {
            Tensor::from_vec_f32(v, (r, c)).and_then(|t| t.to_device(dev))
        };
        let t1 = |v: Vec<f32>| -> Result<Tensor> {
            let n = v.len();
            Tensor::from_vec_f32(v, (n,)).and_then(|t| t.to_device(dev))
        };

        // conv stem: native reader gives [out,in,k].
        let (c1w, c1d) = dqd("a.conv1d.1.weight")?; // [1280,128,3]
        let (c1b, _) = dqd("a.conv1d.1.bias")?;
        let (c2w, c2d) = dqd("a.conv1d.2.weight")?; // [1280,1280,3]
        let (c2b, _) = dqd("a.conv1d.2.bias")?;
        let conv1w = Tensor::from_vec_f32(c1w, (c1d[0], c1d[1], c1d[2]))?.to_device(dev)?;
        let conv2w = Tensor::from_vec_f32(c2w, (c2d[0], c2d[1], c2d[2]))?.to_device(dev)?;
        let conv1 = Conv1d::new(conv1w, Some(t1(c1b)?), same_length_1d(3, 1));
        let conv2 = Conv1d::new(
            conv2w,
            Some(t1(c2b)?),
            Conv1dConfig {
                stride: 2,
                ..same_length_1d(3, 1)
            },
        );

        // position embedding [NCTX, DIM]
        let (pe, ped) = dqd("a.position_embd.weight")?;
        let pos_embd = Tensor::from_vec_f32(pe, (ped[0], ped[1]))?.to_device(dev)?;

        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("a.blk.{i}");
            let (qw, _) = dqd(&format!("{p}.attn_q.weight"))?;
            let (qb, _) = dqd(&format!("{p}.attn_q.bias"))?;
            let (kw, _) = dqd(&format!("{p}.attn_k.weight"))?;
            let (vw, _) = dqd(&format!("{p}.attn_v.weight"))?;
            let (vb, _) = dqd(&format!("{p}.attn_v.bias"))?;
            let (ow, _) = dqd(&format!("{p}.attn_out.weight"))?;
            let (ob, _) = dqd(&format!("{p}.attn_out.bias"))?;
            let (l1w, _) = dqd(&format!("{p}.ln1.weight"))?;
            let (l1b, _) = dqd(&format!("{p}.ln1.bias"))?;
            let (l2w, _) = dqd(&format!("{p}.ln2.weight"))?;
            let (l2b, _) = dqd(&format!("{p}.ln2.bias"))?;
            let (uw, ud) = dqd(&format!("{p}.ffn_up.weight"))?; // [5120,1280]
            let (ub, _) = dqd(&format!("{p}.ffn_up.bias"))?;
            let (dw, dd) = dqd(&format!("{p}.ffn_down.weight"))?; // [1280,5120]
            let (db, _) = dqd(&format!("{p}.ffn_down.bias"))?;
            blocks.push(Block {
                ln1_w: t1(l1w)?,
                ln1_b: t1(l1b)?,
                q_w: t2(qw, DIM, DIM)?,
                q_b: t1(qb)?,
                k_w: t2(kw, DIM, DIM)?,
                v_w: t2(vw, DIM, DIM)?,
                v_b: t1(vb)?,
                o_w: t2(ow, DIM, DIM)?,
                o_b: t1(ob)?,
                ln2_w: t1(l2w)?,
                ln2_b: t1(l2b)?,
                up_w: t2(uw, ud[0], ud[1])?,
                up_b: t1(ub)?,
                down_w: t2(dw, dd[0], dd[1])?,
                down_b: t1(db)?,
            });
        }
        let (plw, _) = dqd("a.post_ln.weight")?;
        let (plb, _) = dqd("a.post_ln.bias")?;
        let (m1, m1d) = dqd("mm.a.mlp.1.weight")?; // [3072,5120]
        let (m2, m2d) = dqd("mm.a.mlp.2.weight")?; // [3072,3072]
        Ok(Self {
            conv1,
            conv2,
            pos_embd,
            blocks,
            post_ln_w: t1(plw)?,
            post_ln_b: t1(plb)?,
            mm1_w: t2(m1, m1d[0], m1d[1])?,
            mm2_w: t2(m2, m2d[0], m2d[1])?,
            dev: dev.clone(),
        })
    }

    /// Encode ONE <=30 s chunk. `mel [1, 128, T]` (T <= 3000) -> audio embeds `[frames/STACK, PROJ]`
    /// where frames = T/2 (conv2 stride-2). Positions reset per chunk (absolute pos-embd).
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mel = mel.to_device(&self.dev)?;
        // conv stem
        let x = self.conv1.forward(&mel)?.gelu_erf()?; // [1,1280,T]
        let x = self.conv2.forward(&x)?.gelu_erf()?; // [1,1280,T/2]
        let x = x.transpose(1, 2)?.contiguous()?; // [1,T/2,1280]
        let frames = x.dim(1)?;
        let x = x.reshape((frames, DIM))?; // single sequence
        let pos = self.pos_embd.narrow(0, 0, frames)?; // [frames,1280]
        let mut x = x.add(&pos)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        let x = layernorm(&x, &self.post_ln_w, &self.post_ln_b)?; // [frames,1280]
                                                                  // projector: stack STACK consecutive frames -> [frames/STACK, DIM*STACK]
        let keep = (frames / STACK) * STACK;
        let x = x.narrow(0, 0, keep)?.reshape((keep / STACK, DIM * STACK))?;
        let x = lin(&x, &self.mm1_w, None)?.gelu_erf()?; // [T/STACK, 3072]
        lin(&x, &self.mm2_w, None) // [T/STACK, 3072]
    }

    /// Projector stack factor (frames collapsed per audio embed) - for computing valid-embed counts.
    pub fn stack_factor(&self) -> usize {
        STACK
    }
}
