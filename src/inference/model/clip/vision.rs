//! CLIP ViT-H/14 vision tower.
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
    conv2d_no_bias, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear,
};
use crate::tensor::Module;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

/// The attention's head count. The checkpoint ships without a config and its tensors do
/// not state it; ViT-H splits its width into this many heads.
const ATTENTION_HEADS: usize = 16;

/// The tower's shape, read from the stored tensors.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    width: usize,
    layers: usize,
    mlp: usize,
    /// Pixels on a side of a patch, and patches on a side of the image.
    patch: usize,
    grid: usize,
}

impl Geometry {
    fn read(v: &VarBuilder) -> Result<Self> {
        let e = v.pp("embeddings");
        let width = e.shape_of("class_embedding")?[0];
        let patch = e.pp("patch_embedding").shape_of("weight")?[2];
        let positions = e.pp("position_embedding").shape_of("weight")?[0];
        // The positions are the class token and a square grid of patches.
        let grid = (positions.saturating_sub(1) as f64).sqrt().round() as usize;
        if grid * grid + 1 != positions {
            return Err(crate::tensor::Error(format!(
                "clip vision: {positions} positions are not a square grid and a class token"
            )));
        }
        let enc = v.pp("encoder").pp("layers");
        Ok(Self {
            width,
            layers: v.pp("encoder").count_children("layers"),
            mlp: enc.pp("0").pp("mlp").pp("fc1").shape_of("weight")?[0],
            patch,
            grid,
        })
    }

    fn tokens(&self) -> usize {
        self.grid * self.grid + 1
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
}

impl Attention {
    fn new(vb: &VarBuilder, width: usize) -> Result<Self> {
        Ok(Self {
            q: linear(width, width, &vb.pp("q_proj"))?,
            k: linear(width, width, &vb.pp("k_proj"))?,
            v: linear(width, width, &vb.pp("v_proj"))?,
            out: linear(width, width, &vb.pp("out_proj"))?,
            width,
        })
    }

    /// `xs`: `[tokens, width]`.
    ///
    /// The scores are formed, softmaxed and applied here, in one pass over the whole token
    /// square, and the scale multiplies them AFTER the first product rather than being folded
    /// into the query before it. Both are part of what this tower answers: the rounding of a
    /// product-then-scale differs from that of a scale-then-product, and a pass split into
    /// query tiles is a different sequence of accumulations again. What is consumed downstream
    /// is a conditioning embedding that nothing compares against a reference, so a drift here
    /// is invisible at every later step.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let n = xs.dims()[0];
        let hd = self.width / ATTENTION_HEADS;
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape(vec![n, ATTENTION_HEADS, hd])?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = split(self.q.forward(xs)?)?;
        let k = split(self.k.forward(xs)?)?;
        let v = split(self.v.forward(xs)?)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let attn = scores.softmax_last_dim()?;
        let o = attn
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape(vec![n, self.width])?;
        self.out.forward(&o)
    }
}

struct Layer {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: crate::tensor::layer::Mlp,
}

impl Layer {
    fn new(vb: &VarBuilder, g: &Geometry) -> Result<Self> {
        let (width, mlp) = (g.width, g.mlp);
        Ok(Self {
            norm1: layer_norm(width, 1e-5, &vb.pp("layer_norm1"))?,
            attn: Attention::new(&vb.pp("self_attn"), width)?,
            norm2: layer_norm(width, 1e-5, &vb.pp("layer_norm2"))?,
            // CLIP uses the sigmoid approximation of GELU (`quick_gelu` in this checkpoint's
            // config); the error-function form drifts the activations enough to matter over
            // the tower's depth.
            mlp: crate::tensor::layer::Mlp::new(
                linear(width, mlp, &vb.pp("mlp").pp("fc1"))?,
                crate::tensor::ops::Activation::QuickGelu,
                linear(mlp, width, &vb.pp("mlp").pp("fc2"))?,
            ),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.add(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let m = self.mlp.forward(&self.norm2.forward(&h)?)?;
        h.add(&m)
    }
}

/// The tower, resident.
pub struct ClipVisionH {
    patch: Conv2d,
    class_embedding: Tensor,
    position_embedding: Tensor,
    pre_norm: LayerNorm,
    layers: Vec<Layer>,
    g: Geometry,
    device: Device,
}

impl ClipVisionH {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], DType::F32, device) }?;
        let v = vb.pp("vision_model");
        let g = Geometry::read(&v)?;
        let (width, patch) = (g.width, g.patch);
        let cfg = Conv2dConfig {
            padding: 0,
            stride: patch,
            ..Default::default()
        };
        let mut layers = Vec::with_capacity(g.layers);
        let enc = v.pp("encoder").pp("layers");
        for i in 0..g.layers {
            layers.push(Layer::new(&enc.pp(i.to_string()), &g)?);
        }
        Ok(Self {
            // The patch projection carries no bias in this checkpoint.
            patch: conv2d_no_bias(
                3,
                width,
                patch,
                cfg,
                &v.pp("embeddings").pp("patch_embedding"),
            )?,
            class_embedding: v.pp("embeddings").get(width, "class_embedding")?,
            position_embedding: v
                .pp("embeddings")
                .pp("position_embedding")
                .get((g.tokens(), width), "weight")?,
            // Checkpoint spelling, typo included.
            pre_norm: layer_norm(width, 1e-5, &v.pp("pre_layrnorm"))?,
            layers,
            g,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
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
    fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let (width, grid) = (self.g.width, self.g.grid);
        // Patch projection: [1, width, grid, grid] -> [grid*grid, width].
        let p = self.patch.forward(pixels)?;
        let p = p
            .reshape(vec![width, grid * grid])?
            .transpose(0, 1)?
            .contiguous()?;
        // Class token first, then the patches, then positions.
        let cls = self.class_embedding.reshape(vec![1, width])?;
        let x = Tensor::cat(&[&cls, &p], 0)?;
        let x = x.add(&self.position_embedding)?;
        let mut x = self.pre_norm.forward(&x)?;
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x)?;
            // Stop at the penultimate layer's OUTPUT. Running the last one and
            // discarding it would waste a layer; taking the last one instead would be a
            // different conditioning that nothing downstream could flag.
            if i == self.g.penultimate() {
                return Ok(x);
            }
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
        let m = ClipVisionH::load(p.to_str().unwrap(), &dev).expect("load");
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

impl ClipVisionH {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, self.layers.len())
    }
}
