//! Image embeddings spliced into the text stream at a run of placeholder tokens, with the
//! multi-axis rotary positions of the Qwen3-VL family.
//!
//! A text token at scalar position `p` rotates at `(p, p, p)`, which is the one-axis
//! rotation, so text-only requests never come here. An image run of a `gh x gw` merged grid
//! starting at `p0` gives its `i`-th token `(p0, p0 + i / gw, p0 + i % gw)`, and the text
//! after it resumes at `p0 + gw`. Rotary pair `j` rotates by axis `j % 3`.

#[allow(unused_imports)]
use super::*;

/// The vision tower whose output replaces the placeholder run, and the placeholder's id.
pub struct SplicedVision {
    pub tower: crate::inference::model::qwen35::vision::Qwen35Vision,
    pub image_token: u32,
}

/// The speech tower whose output replaces the placeholder of a prompt.
#[cfg(feature = "audio")]
pub struct SplicedAudio {
    pub tower: crate::inference::model::audio_in::AudioTower,
}

/// Per-token `(t, h, w)` positions of `ids`: every run of `image_token` spans the merged
/// grid `grids[k]` of the `k`-th image.
pub fn mrope_positions(ids: &[u32], image_token: u32, grids: &[(usize, usize)]) -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(ids.len());
    let mut p: i32 = 0;
    let mut image = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token {
            let (gh, gw) = grids.get(image).copied().unwrap_or((1, 1));
            let n = (gh * gw).min(ids.len() - i);
            for k in 0..n {
                out.push([p, p + (k / gw) as i32, p + (k % gw) as i32]);
            }
            p += gw as i32;
            i += n;
            image += 1;
        } else {
            out.push([p, p, p]);
            p += 1;
            i += 1;
        }
    }
    out
}

/// Rotary tables `[seq, rope_dim / 2]` for positions `pos`, pair `j` taking axis `j % 3`.
pub fn mrope_cos_sin(
    pos: &[[i32; 3]],
    rope_dim: usize,
    base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = rope_dim / 2;
    let inv = crate::inference::model::rope::inverse_frequencies(rope_dim, base);
    let mut cos = vec![0f32; pos.len() * half];
    let mut sin = vec![0f32; pos.len() * half];
    for (p, thw) in pos.iter().enumerate() {
        for (j, &f) in inv.iter().enumerate() {
            let a = thw[j % 3] as f32 * f;
            cos[p * half + j] = a.cos();
            sin[p * half + j] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (pos.len(), half), device)?,
        Tensor::from_vec(sin, (pos.len(), half), device)?,
    ))
}

impl GenericHeteroTransformer {
    /// The placeholder token a spliced image expands, when the model carries a tower.
    pub fn spliced_image_token(&self) -> Option<u32> {
        self.spliced_vision.as_ref().map(|v| v.image_token)
    }

    /// Prefills `prompt` with its one placeholder expanded to the image's merged grid and
    /// the tower's embeddings over it. Returns the last logits and the next decode position,
    /// the logical position past every prefilled one rather than the sequence length.
    pub fn forward_spliced_image(
        &mut self,
        prompt: &[u32],
        pixels: &Tensor,
        gh: usize,
        gw: usize,
    ) -> Result<(Tensor, usize)> {
        let vision = self
            .spliced_vision
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("the model carries no vision tower"))?;
        let token = vision.image_token;
        let img = vision.tower.forward(pixels, gh, gw)?;
        let n_img = img.dim(0)?;
        // The tower merges the patch grid in squares of two.
        let (mh, mw) = (gh / 2, gw / 2);
        if n_img != mh * mw {
            return Err(crate::tensor::Error::msg(format!(
                "the tower gave {n_img} rows for a {mh}x{mw} merged grid"
            )));
        }
        let start = prompt.iter().position(|&t| t == token).ok_or_else(|| {
            crate::tensor::Error::msg("image placeholder missing from the prompt")
        })?;
        let mut ids = prompt[..start].to_vec();
        ids.extend(std::iter::repeat_n(token, n_img));
        ids.extend_from_slice(&prompt[start + 1..]);

        let emb_device = self.embeddings.embeddings().device().clone();
        let id_tensor = Tensor::new(ids.as_slice(), &emb_device)?.unsqueeze(0)?;
        let text = self.embeddings.forward(&id_tensor)?;
        let hidden = text.dim(2)?;
        let img = img
            .to_device(&emb_device)?
            .to_dtype(text.dtype())?
            .reshape((1, n_img, hidden))?;
        let mut parts = vec![text.narrow(1, 0, start)?, img];
        let after = ids.len() - start - n_img;
        if after > 0 {
            parts.push(text.narrow(1, start + n_img, after)?);
        }
        let spliced = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;

        let pos = mrope_positions(&ids, token, &[(mh, mw)]);
        let next = pos
            .iter()
            .flat_map(|p| p.iter().copied())
            .max()
            .unwrap_or(-1) as usize
            + 1;
        let chunk = crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS;
        let mut logits = None;
        let mut off = 0;
        while off < ids.len() {
            let n = chunk.min(ids.len() - off);
            for layer in self.layers.iter_mut() {
                if matches!(layer.mixer, super::mixer::Mixer::Attention) {
                    let rope_dim = match layer.rope_dim {
                        0 => layer.head_dim,
                        d => d,
                    };
                    let (cos, sin) = mrope_cos_sin(
                        &pos[off..off + n],
                        rope_dim,
                        layer.rope_base,
                        &layer.cos.device(),
                    )?;
                    layer.rope_rows = Some((
                        cos.to_dtype(layer.cos.dtype())?,
                        sin.to_dtype(layer.sin.dtype())?,
                    ));
                }
            }
            let out =
                self.forward_layers_and_output(spliced.narrow(1, off, n)?, None, off, n, false);
            for layer in self.layers.iter_mut() {
                layer.rope_rows = None;
            }
            logits = Some(out?);
            off += n;
        }
        let logits = logits.ok_or_else(|| crate::tensor::Error::msg("empty spliced prefill"))?;
        Ok((logits, next))
    }
}

#[cfg(feature = "audio")]
impl GenericHeteroTransformer {
    /// Gives the model a speech tower. The tower's embeddings must be as wide as the model's.
    pub fn set_audio_tower(
        &mut self,
        tower: crate::inference::model::audio_in::AudioTower,
    ) -> Result<()> {
        if tower.embedding_width() != self.config.embedding_length {
            return Err(crate::tensor::Error::msg(format!(
                "the speech tower projects to {} and the model is {} wide",
                tower.embedding_width(),
                self.config.embedding_length
            )));
        }
        self.spliced_audio = Some(SplicedAudio { tower });
        Ok(())
    }

    /// The speech tower, when the model carries one.
    pub fn audio_tower(&self) -> Option<&crate::inference::model::audio_in::AudioTower> {
        self.spliced_audio.as_ref().map(|a| &a.tower)
    }
}

impl GenericHeteroTransformer {
    /// Prefills `prompt` with the token at `at` replaced by `embeds`, `[1, n, width]`.
    /// Returns the last logits and the sequence length, which is the next decode position.
    pub fn forward_spliced_audio(
        &mut self,
        prompt: &[u32],
        at: usize,
        embeds: &Tensor,
    ) -> Result<(Tensor, usize)> {
        if at >= prompt.len() {
            return Err(crate::tensor::Error::msg(
                "the audio placeholder is past the end of the prompt".to_string(),
            ));
        }
        let emb_device = self.embeddings.embeddings().device().clone();
        let ids = |part: &[u32]| -> Result<Tensor> { Tensor::new(part, &emb_device)?.unsqueeze(0) };
        let total = prompt.len() - 1 + embeds.dim(1)?;
        let logits =
            self.forward_with_audio_embeds(&ids(&prompt[..at])?, embeds, &ids(&prompt[at + 1..])?)?;
        Ok((logits, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_positions_collapse_to_one_axis() {
        let dev = Device::Cpu;
        let (rope_dim, base, n) = (16usize, 1e7f32, 6usize);
        let pos: Vec<[i32; 3]> = (0..n as i32).map(|p| [p, p, p]).collect();
        let (c, s) = mrope_cos_sin(&pos, rope_dim, base, &dev).unwrap();
        let (c1, s1) =
            crate::inference::model::rope::precomput_freqs_cis_host(rope_dim, n, base, &dev)
                .unwrap();
        let v = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let d = |a: Vec<f32>, b: Vec<f32>| {
            a.iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        assert!(d(v(c), v(c1)) < 1e-5);
        assert!(d(v(s), v(s1)) < 1e-5);
    }

    #[test]
    fn an_image_run_takes_its_grid_positions() {
        let img = 7u32;
        let ids = [10u32, 20, img, img, img, img, 30];
        assert_eq!(
            mrope_positions(&ids, img, &[(2, 2)]),
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                [4, 4, 4],
            ]
        );
    }
}
