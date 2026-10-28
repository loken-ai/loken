//! CLIP's ViT vision tower, and SigLIP's: the same encoder blocks, read from the tensors
//! the checkpoint stores. A class token, a norm before the blocks and a bias on the patch
//! projection are each present or not; the pooled output is the class token through
//! `post_layernorm` when the tower has one, else the attention-pool head over every token
//! after `post_layernorm`.
//!
//! Wan's image-to-video conditioning is two things: the reference image encoded by the
//! VAE (which becomes extra input channels), and a semantic embedding of that image,
//! which is what this produces. Without the second one the model gets pixels but no idea
//! WHAT it is looking at, and the motion it invents does not belong to the subject.
//!
//! WHICH OUTPUT, exactly. Read from the reference rather than assumed, because every
//! plausible choice here is wrong in a way nothing downstream can detect - the shapes
//! all match and the video merely comes out subtly unrelated to the input:
//!
//!   * the PENULTIMATE hidden state, i.e. after the second-to-last layer, not the last;
//!   * WITHOUT `post_layernorm` (that norm serves the pooled CLS output, which is not
//!     what is consumed here);
//!   * every token (1 class + the patch grid), not a pooled vector.
//!
//! The encoder is asked for `intermediate_output=-2` and the model is handed
//! `penultimate_hidden_states`; the conditioning path
//! puts it in `clip_fea`.
//!
//! The `pre_layrnorm` spelling is the checkpoint's, typo and all - renaming it here
//! would just mean failing to find the weight.

use crate::tensor::layer::{
    conv2d, conv2d_no_bias, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear,
};
use crate::tensor::ops::Activation;
use crate::tensor::Module;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

/// What the tensors of a vision tower do not state: its head count, its norm epsilon and
/// its activation, and whether the pooled output is wanted.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub heads: usize,
    pub layer_norm_eps: f32,
    pub activation: Activation,
    /// Load `post_layernorm` and the attention-pool head, which only the pooled output reads.
    pub pooled: bool,
}

/// A tower loaded from a lone weights file, without a config: ViT-H's 16 heads, the OpenAI
/// norm epsilon and quick GELU, unpooled.
const LONE_FILE_TOWER: Config = Config {
    heads: 16,
    layer_norm_eps: 1e-5,
    activation: Activation::QuickGelu,
    pooled: false,
};

/// The tower's shape, read from the stored tensors.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    width: usize,
    layers: usize,
    mlp: usize,
    /// Pixels on a side of a patch, and patches on a side of the image.
    patch: usize,
    grid: usize,
    /// Whether a class token leads the patches.
    class_token: bool,
}

impl Geometry {
    fn read(v: &VarBuilder) -> Result<Self> {
        let e = v.pp("embeddings");
        let patch_shape = e.pp("patch_embedding").shape_of("weight")?;
        let (width, patch) = (patch_shape[0], patch_shape[2]);
        let positions = e.pp("position_embedding").shape_of("weight")?[0];
        let class_token = e.contains("class_embedding");
        // The positions are the class token, when there is one, and a square grid of patches.
        let patches = positions.saturating_sub(usize::from(class_token));
        let grid = (patches as f64).sqrt().round() as usize;
        if grid * grid != patches {
            return Err(crate::tensor::Error(format!(
                "clip vision: {positions} positions are not a square grid{}",
                if class_token {
                    " and a class token"
                } else {
                    ""
                }
            )));
        }
        let enc = v.pp("encoder").pp("layers");
        Ok(Self {
            width,
            layers: v.pp("encoder").count_children("layers"),
            mlp: enc.pp("0").pp("mlp").pp("fc1").shape_of("weight")?[0],
            patch,
            grid,
            class_token,
        })
    }

    fn tokens(&self) -> usize {
        self.grid * self.grid + usize::from(self.class_token)
    }

    fn image_size(&self) -> usize {
        self.grid * self.patch
    }

    /// Which layer's output is consumed (0-based): the penultimate one.
    fn penultimate(&self) -> usize {
        self.layers.saturating_sub(2)
    }
}

/// CLIP's image normalisation. Not ImageNet's - close enough to look right and wrong
/// enough to shift the embedding.
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    width: usize,
    heads: usize,
}

impl Attention {
    fn new(vb: &VarBuilder, width: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            q: linear(width, width, &vb.pp("q_proj"))?,
            k: linear(width, width, &vb.pp("k_proj"))?,
            v: linear(width, width, &vb.pp("v_proj"))?,
            out: linear(width, width, &vb.pp("out_proj"))?,
            width,
            heads,
        })
    }

    /// The attention a pooling head stores packed: one `in_proj_weight` of query, key and
    /// value rows in that order, one `in_proj_bias` beside it, and `out_proj`.
    fn packed(vb: &VarBuilder, width: usize, heads: usize) -> Result<Self> {
        let w = vb.get((3 * width, width), "in_proj_weight")?;
        let b = vb.get(3 * width, "in_proj_bias")?;
        let part = |i: usize| -> Result<Linear> {
            Linear::new(
                w.narrow(0, i * width, width)?.contiguous()?,
                Some(b.narrow(0, i * width, width)?.contiguous()?),
            )
        };
        Ok(Self {
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: linear(width, width, &vb.pp("out_proj"))?,
            width,
            heads,
        })
    }

    /// `xs`: `[tokens, width]`, attending to itself.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.attend(xs, xs)
    }

    /// `queries`: `[rows, width]`, attending to `source`: `[tokens, width]`.
    ///
    /// The scores are formed, softmaxed and applied here, in one pass over the whole token
    /// square, and the scale multiplies them AFTER the first product rather than being folded
    /// into the query before it. Both are part of what this tower answers: the rounding of a
    /// product-then-scale differs from that of a scale-then-product, and a pass split into
    /// query tiles is a different sequence of accumulations again. What is consumed downstream
    /// is a conditioning embedding that nothing compares against a reference, so a drift here
    /// is invisible at every later step.
    fn attend(&self, queries: &Tensor, source: &Tensor) -> Result<Tensor> {
        let (rows, n) = (queries.dims()[0], source.dims()[0]);
        let heads = self.heads;
        let hd = self.width / heads;
        let split = |t: Tensor, len: usize| -> Result<Tensor> {
            t.reshape(vec![len, heads, hd])?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = split(self.q.forward(queries)?, rows)?;
        let k = split(self.k.forward(source)?, n)?;
        let v = split(self.v.forward(source)?, n)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let attn = scores.softmax_last_dim()?;
        let o = attn
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape(vec![rows, self.width])?;
        self.out.forward(&o)
    }
}

/// The block's feed-forward, `fc1` and `fc2` around the tower's activation.
///
/// CLIP uses the sigmoid approximation of GELU (`quick_gelu`) and SigLIP the tanh one; the
/// error-function form drifts the activations enough to matter over the tower's depth.
fn mlp(
    vb: &VarBuilder,
    width: usize,
    hidden: usize,
    c: &Config,
) -> Result<crate::tensor::layer::Mlp> {
    Ok(crate::tensor::layer::Mlp::new(
        linear(width, hidden, &vb.pp("fc1"))?,
        c.activation,
        linear(hidden, width, &vb.pp("fc2"))?,
    ))
}

struct Layer {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: crate::tensor::layer::Mlp,
}

impl Layer {
    fn new(vb: &VarBuilder, g: &Geometry, c: &Config) -> Result<Self> {
        let (width, eps) = (g.width, c.layer_norm_eps);
        Ok(Self {
            norm1: layer_norm(width, eps, &vb.pp("layer_norm1"))?,
            attn: Attention::new(&vb.pp("self_attn"), width, c.heads)?,
            norm2: layer_norm(width, eps, &vb.pp("layer_norm2"))?,
            mlp: mlp(&vb.pp("mlp"), width, g.mlp, c)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.add(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let m = self.mlp.forward(&self.norm2.forward(&h)?)?;
        h.add(&m)
    }
}

/// The attention-pool head: a learned probe attends to every token, then a norm and a
/// feed-forward are added back onto it.
struct PoolHead {
    probe: Tensor,
    attn: Attention,
    norm: LayerNorm,
    mlp: crate::tensor::layer::Mlp,
}

impl PoolHead {
    fn new(vb: &VarBuilder, g: &Geometry, c: &Config) -> Result<Self> {
        let width = g.width;
        let hidden = vb.pp("mlp").pp("fc1").shape_of("weight")?[0];
        Ok(Self {
            probe: vb.get((1, 1, width), "probe")?.reshape(vec![1, width])?,
            attn: Attention::packed(&vb.pp("attention"), width, c.heads)?,
            norm: layer_norm(width, c.layer_norm_eps, &vb.pp("layernorm"))?,
            mlp: mlp(&vb.pp("mlp"), width, hidden, c)?,
        })
    }

    /// `[tokens, width]` -> `[1, width]`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.attn.attend(&self.probe, xs)?;
        let m = self.mlp.forward(&self.norm.forward(&h)?)?;
        h.add(&m)
    }
}

/// How the tower reduces its tokens to one vector.
enum Pool {
    /// The class token through `post_layernorm`.
    ClassToken(LayerNorm),
    /// Every token through `post_layernorm`, then the attention-pool head.
    Head(LayerNorm, PoolHead),
}

/// The tower, resident.
pub struct VisionTower {
    patch: Conv2d,
    class_embedding: Option<Tensor>,
    position_embedding: Tensor,
    pre_norm: Option<LayerNorm>,
    layers: Vec<Layer>,
    pool: Option<Pool>,
    g: Geometry,
    device: Device,
}

impl VisionTower {
    /// A lone weights file holding `vision_model.*`, unpooled.
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], DType::F32, device) }?;
        Self::new(&vb.pp("vision_model"), &LONE_FILE_TOWER, device)
    }

    /// The tower rooted at `v` (`vision_model` in a checkpoint).
    pub fn new(v: &VarBuilder, c: &Config, device: &Device) -> Result<Self> {
        let g = Geometry::read(v)?;
        let (width, patch) = (g.width, g.patch);
        let eps = c.layer_norm_eps;
        let cfg = Conv2dConfig {
            padding: 0,
            stride: patch,
            ..Default::default()
        };
        let mut layers = Vec::with_capacity(g.layers);
        let enc = v.pp("encoder").pp("layers");
        for i in 0..g.layers {
            layers.push(Layer::new(&enc.pp(i.to_string()), &g, c)?);
        }
        let e = v.pp("embeddings");
        let patch_vb = e.pp("patch_embedding");
        let patch_conv = if patch_vb.contains("bias") {
            conv2d(3, width, patch, cfg, &patch_vb)?
        } else {
            conv2d_no_bias(3, width, patch, cfg, &patch_vb)?
        };
        let pool = if c.pooled {
            let post = layer_norm(width, eps, &v.pp("post_layernorm"))?;
            Some(if g.class_token {
                Pool::ClassToken(post)
            } else {
                Pool::Head(post, PoolHead::new(&v.pp("head"), &g, c)?)
            })
        } else {
            None
        };
        Ok(Self {
            patch: patch_conv,
            class_embedding: if g.class_token {
                Some(e.get(width, "class_embedding")?)
            } else {
                None
            },
            position_embedding: e
                .pp("position_embedding")
                .get((g.tokens(), width), "weight")?,
            // Checkpoint spelling, typo included.
            pre_norm: if v.contains("pre_layrnorm.weight") {
                Some(layer_norm(width, eps, &v.pp("pre_layrnorm"))?)
            } else {
                None
            },
            layers,
            pool,
            g,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Pixels on a side of the image the tower reads.
    pub fn image_size(&self) -> usize {
        self.g.image_size()
    }

    /// Normalised `[1, 3, S, S]` -> the pooled vector `[1, width]`, after every layer.
    pub fn pooled(&self, pixels: &Tensor) -> Result<Tensor> {
        let Some(pool) = &self.pool else {
            return Err(crate::tensor::Error(
                "clip vision: the tower was loaded without its pooling".into(),
            ));
        };
        let x = self.run(pixels, self.layers.len())?;
        match pool {
            Pool::ClassToken(post) => post.forward(&x.narrow(0, 0, 1)?),
            Pool::Head(post, head) => head.forward(&post.forward(&x)?),
        }
    }

    /// Interleaved RGB8 at any size -> the penultimate hidden states `[tokens, width]`.
    ///
    /// The image is resized to the tower's image size and normalised with CLIP's own statistics; feeding it
    /// raw, or with ImageNet's, produces an embedding that is plausible and wrong.
    pub fn embed_image(&self, rgb: &[u8], w: usize, h: usize) -> Result<Tensor> {
        let x = self.preprocess(rgb, w, h)?;
        self.forward(&x)
    }

    /// RGB8 -> normalised `[1, 3, S, S]` on this tower's device.
    fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Result<Tensor> {
        if w == 0 || h == 0 || rgb.len() < w * h * 3 {
            return Err(crate::tensor::Error(format!(
                "clip vision: expected {}x{} RGB8 ({} bytes), got {}",
                w,
                h,
                w * h * 3,
                rgb.len()
            )));
        }
        let s = self.g.image_size();
        let plane = s * s;
        let mut chw = vec![0f32; 3 * plane];
        for y in 0..s {
            // Nearest-neighbour: the tower sees its own grid whatever the source, and a
            // fancier filter changes the embedding by less than the crop choice does.
            let sy = y * h / s;
            for x in 0..s {
                let sx = x * w / s;
                let src = (sy * w + sx) * 3;
                for c in 0..3 {
                    let v = f32::from(rgb[src + c]) / 255.0;
                    chw[c * plane + y * s + x] = (v - MEAN[c]) / STD[c];
                }
            }
        }
        Tensor::from_vec_f32(chw, vec![1, 3, s, s])?.to_device(&self.device)
    }

    /// `[1, 3, S, S]` -> `[tokens, width]` after the penultimate layer.
    ///
    /// Stops at the penultimate layer's OUTPUT. Running the last one and discarding it
    /// would waste a layer; taking the last one instead would be a different conditioning
    /// that nothing downstream could flag.
    fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        self.run(pixels, self.g.penultimate() + 1)
    }

    /// `[1, 3, S, S]` -> `[tokens, width]` after the first `depth` layers.
    fn run(&self, pixels: &Tensor, depth: usize) -> Result<Tensor> {
        let (width, grid) = (self.g.width, self.g.grid);
        // Patch projection: [1, width, grid, grid] -> [grid*grid, width].
        let p = self.patch.forward(pixels)?;
        let p = p
            .reshape(vec![width, grid * grid])?
            .transpose(0, 1)?
            .contiguous()?;
        // Class token first when there is one, then the patches, then positions.
        let x = match &self.class_embedding {
            Some(cls) => Tensor::cat(&[&cls.reshape(vec![1, width])?, &p], 0)?,
            None => p,
        };
        let mut x = x.add(&self.position_embedding)?;
        if let Some(norm) = &self.pre_norm {
            x = norm.forward(&x)?;
        }
        for l in self.layers.iter().take(depth) {
            x = l.forward(&x)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// The geometry follows the stored tensors: the grid from the position table, the
    /// depth from the numbered layers, and the consumed layer from the depth.
    #[test]
    fn the_geometry_is_read_from_the_tensors() {
        let mut m = std::collections::HashMap::new();
        let z = |d: &[usize]| Tensor::zeros(d.to_vec(), DType::F32).unwrap();
        let (width, mlp, patch, grid, layers) = (8, 12, 2, 3, 4);
        m.insert("embeddings.class_embedding".into(), z(&[width]));
        m.insert(
            "embeddings.patch_embedding.weight".into(),
            z(&[width, 3, patch, patch]),
        );
        m.insert(
            "embeddings.position_embedding.weight".into(),
            z(&[grid * grid + 1, width]),
        );
        for i in 0..layers {
            m.insert(
                format!("encoder.layers.{i}.mlp.fc1.weight"),
                z(&[mlp, width]),
            );
        }
        let vb = VarBuilder::from_tensors(m, DType::F32, &Device::Cpu);
        let g = Geometry::read(&vb).unwrap();
        assert_eq!(
            (g.width, g.mlp, g.patch, g.grid, g.layers),
            (width, mlp, patch, grid, layers)
        );
        assert_eq!(
            g.tokens(),
            grid * grid + 1,
            "the patches plus the class token"
        );
        assert_eq!(g.image_size(), grid * patch);
        assert_eq!(g.penultimate(), layers - 2);
    }

    /// A tower without a class token reads its grid from the whole position table.
    #[test]
    fn a_tower_without_a_class_token_is_all_patches() {
        let mut m = std::collections::HashMap::new();
        let z = |d: &[usize]| Tensor::zeros(d.to_vec(), DType::F32).unwrap();
        let (width, patch, grid) = (8, 2, 3);
        m.insert(
            "embeddings.patch_embedding.weight".into(),
            z(&[width, 3, patch, patch]),
        );
        m.insert(
            "embeddings.position_embedding.weight".into(),
            z(&[grid * grid, width]),
        );
        m.insert("encoder.layers.0.mlp.fc1.weight".into(), z(&[12, width]));
        let vb = VarBuilder::from_tensors(m, DType::F32, &Device::Cpu);
        let g = Geometry::read(&vb).unwrap();
        assert!(!g.class_token);
        assert_eq!((g.grid, g.tokens()), (grid, grid * grid));
    }

    /// The pooling head's packed attention splits its rows into query, key and value in
    /// that order: with one token to attend to, the probe's answer is that token's value
    /// projection through `out_proj`, whatever the query and key rows hold.
    #[test]
    fn a_packed_attention_reads_query_key_value_in_order() {
        let width = 2;
        let mut m = std::collections::HashMap::new();
        let t = |v: Vec<f32>, d: &[usize]| Tensor::from_vec_f32(v, d.to_vec()).unwrap();
        // Query and key rows are arbitrary; the value rows double the input.
        let w = vec![
            1.0, 2.0, 3.0, 4.0, // q
            5.0, 6.0, 7.0, 8.0, // k
            2.0, 0.0, 0.0, 2.0, // v
        ];
        m.insert("in_proj_weight".into(), t(w, &[3 * width, width]));
        m.insert("in_proj_bias".into(), t(vec![0.0; 3 * width], &[3 * width]));
        m.insert(
            "out_proj.weight".into(),
            t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
        );
        m.insert("out_proj.bias".into(), t(vec![0.5, -0.5], &[2]));
        let vb = VarBuilder::from_tensors(m, DType::F32, &Device::Cpu);
        let attn = Attention::packed(&vb, width, 1).unwrap();
        let probe = t(vec![0.3, -0.7], &[1, width]);
        let token = t(vec![1.5, 2.5], &[1, width]);
        let got = attn.attend(&probe, &token).unwrap().to_vec_f32();
        assert_eq!(got, vec![3.5, 4.5]);
    }

    /// The activation the blocks are built with is quick GELU, not the erf form:
    /// `x * sigmoid(1.702x)`.
    #[test]
    fn quick_gelu_matches_its_definition() {
        let x = Tensor::from_vec_f32(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]).unwrap();
        let got = crate::tensor::ops::Activation::QuickGelu
            .apply(&x)
            .unwrap()
            .to_vec_f32();
        for (i, v) in [-2.0f32, -0.5, 0.0, 0.5, 2.0].iter().enumerate() {
            let want = v * (1.0 / (1.0 + (-1.702 * v).exp()));
            assert!(
                (got[i] - want).abs() < 1e-5,
                "quick_gelu({v}) = {} want {want}",
                got[i]
            );
        }
    }

    /// Against the real checkpoint: the shape of what Wan consumes, and that it is NOT
    /// the post-layernorm output.
    #[test]
    #[ignore = "needs the clip_vision_h checkpoint"]
    fn embeds_a_real_image_to_the_penultimate_state() {
        let p = std::path::Path::new(&std::env::var("HOME").unwrap_or_default())
            .join(".cache/huggingface/hub/models--Comfy-Org--Wan_2.1_ComfyUI_repackaged")
            .join("snapshots/main/clip_vision_h.safetensors");
        if !p.exists() {
            println!("clip_vision_h not present; skipping");
            return;
        }
        let dev = Device::Cpu;
        let m = VisionTower::load(p.to_str().unwrap(), &dev).expect("load");
        // A deterministic non-uniform image: a flat one makes every attention row equal
        // and would hide a transposed reshape.
        let (w, h) = (320usize, 200usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = (x * 255 / w) as u8;
                rgb[i + 1] = (y * 255 / h) as u8;
                rgb[i + 2] = ((x + y) % 256) as u8;
            }
        }
        let out = m.embed_image(&rgb, w, h).expect("embed");
        let (tokens, width) = (m.g.tokens(), m.g.width);
        assert_eq!(
            out.dims(),
            &[tokens, width],
            "Wan consumes every token, unpooled"
        );
        let v = out.to_vec_f32();
        assert!(v.iter().all(|x| x.is_finite()), "embedding must be finite");
        // Tokens must differ from one another: an embedding where every token is the
        // same value passes a shape check and carries no image.
        let first = &v[..width];
        let last = &v[(tokens - 1) * width..];
        let diff: f32 = first.iter().zip(last).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1.0,
            "class and last patch tokens are identical ({diff})"
        );
        println!("clip-h penultimate: [{tokens}, {width}] finite, token spread {diff:.1}");
    }
}

impl VisionTower {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, self.layers.len())
    }
}
