//! The residual of a single decoded row of a Sinkhorn wide-residual model on a card.
//!
//! A layer whose 2 Sinkhorn mixers and 2 norms the load keeps on a card holds there:
//! - each mixer's projection, scales and biases,
//! - the norm of the token mixer's input and the norm of the feed-forward's input.
//!
//! A decode step of 1 row keeps its residual on the card through such a layer:
//! - the mixes, the collapse with its norm and the post of both sublayers run on the card,
//! - a token mixer that runs its output projection on the same card reads its input there and
//!   leaves its output there, its key row and the rows its host state keeps come down,
//! - another token mixer takes its input down and its output up,
//! - the feed-forward's input comes down and its output goes up.
//!
//! The residual stays on the card from one such layer to the next on the same card.
//! An engram layer takes it down, adds the engram on the host and puts it back.
//! A layer without them, and a layer not admitted, take it down and run on the host.
//! A residual that ends on the card that keeps the head is collapsed and normed there, once
//! admitted, and only the logits and the pre mix come down.
//! Else the head reads the residual and the pre mix the last layer left, on the host.
//!
//! The first decoded row of a layer runs on the host.
//! It measures, on its own values, the card's products of both mixers against the exact ones:
//! - the dot products of the residual with the projection, and of the residual with itself,
//! - the collapse with its norm,
//! - the post.
//!
//! The layer is admitted when each card product is no farther from the exact one than the host's
//! product (`card_packed::no_farther`).
//! A layer not admitted runs on the host for the rest of the load.
//!
//! A card call that fails before the token mixer runs leaves the layer to the host for the step.
//! The residual a step starts from is the embedding of its token, so no card copy outlives its
//! step.

use super::wide_sparse_layer::SinkhornBlock;
use crate::tensor::Result;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// The layer's products have not been measured.
const UNMEASURED: u8 = 0;
/// The layer's products are no farther from the exact ones than the host's.
const ADMITTED: u8 = 1;
/// The layer runs on the host.
const REFUSED: u8 = 2;

/// Whether the final collapse and norm of a decoded row run on the card that keeps the head.
///
/// The first decoded row whose residual reaches that card measures them against the exact ones.
/// They are admitted when no farther from the exact ones than the host's.
#[derive(Default)]
pub struct HeadCollapse {
    verdict: AtomicU8,
    /// The bits of the sum of the host's and the card's distances to the exact collapse.
    apart: AtomicU64,
}

impl HeadCollapse {
    /// Whether the card's collapse was admitted, `None` before a decoded row measured it.
    pub fn verdict(&self) -> Option<bool> {
        match self.verdict.load(Ordering::Relaxed) {
            ADMITTED => Some(true),
            REFUSED => Some(false),
            _ => None,
        }
    }

    /// The sum of the host's and the card's relative distances to the exact collapse, as the
    /// measure found them: a bound on the card's relative distance to the host's collapse.
    ///
    /// 0 before a decoded row measured them.
    pub fn apart(&self) -> f64 {
        f64::from_bits(self.apart.load(Ordering::Relaxed))
    }

    #[cfg(feature = "gpu")]
    fn set(&self, admitted: bool, apart: f64) {
        let v = if admitted { ADMITTED } else { REFUSED };
        self.apart.store(apart.to_bits(), Ordering::Relaxed);
        self.verdict.store(v, Ordering::Relaxed);
    }
}

/// A layer's 2 Sinkhorn mixers and 2 norms on a card, when the load keeps them there.
#[derive(Default)]
pub struct CardSinkhorn {
    #[cfg(feature = "gpu")]
    held: Option<gpu::Held>,
    verdict: AtomicU8,
    /// The bits of the sum, over the measured products, of the host's and the card's distances
    /// to the exact ones.
    apart: AtomicU64,
}

impl CardSinkhorn {
    /// The ordinal of the card that keeps the mixers, `None` when the host keeps them.
    pub fn ordinal(&self) -> Option<usize> {
        #[cfg(feature = "gpu")]
        {
            self.held.as_ref().map(|h| h.dev.ordinal())
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    /// Bytes the card holds for the mixers and the norms.
    pub fn bytes(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.held.as_ref().map_or(0, |h| h.bytes)
        }
        #[cfg(not(feature = "gpu"))]
        0
    }

    /// Drops the mixers and the norms from the card: the bytes they held, `None` when the host
    /// keeps them.
    ///
    /// The host runs the layer's mixes from then on.
    pub fn release(&mut self) -> Option<u64> {
        self.ordinal()?;
        Some(std::mem::take(self).bytes() as u64)
    }

    /// Whether the card's products were admitted, `None` before a decoded row measured them.
    pub fn verdict(&self) -> Option<bool> {
        match self.verdict.load(Ordering::Relaxed) {
            ADMITTED => Some(true),
            REFUSED => Some(false),
            _ => None,
        }
    }

    /// The sum, over the products the admission measured, of the host's and the card's relative
    /// distances to the exact ones: a bound on how far apart the card's and the host's products
    /// of the layer are.
    ///
    /// 0 before a decoded row measured them.
    pub fn apart(&self) -> f64 {
        f64::from_bits(self.apart.load(Ordering::Relaxed))
    }

    /// Whether a decoded row's residual stays on the card through the layer.
    pub fn serves(&self) -> bool {
        self.ordinal().is_some() && self.verdict() == Some(true)
    }

    /// Whether the next decoded row of the layer measures the card's products.
    pub fn measures(&self) -> bool {
        self.ordinal().is_some() && self.verdict().is_none()
    }

    /// The mixers and the norms of `block` on card `ordinal`, not yet measured.
    pub(super) fn keep(ordinal: usize, block: &SinkhornBlock) -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            Ok(Self {
                held: Some(gpu::Held::new(ordinal, block)?),
                verdict: AtomicU8::new(UNMEASURED),
                apart: AtomicU64::new(0),
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ordinal, block);
            crate::tensor::bail!("a build without cards keeps the Sinkhorn mixers on the host")
        }
    }
}

#[cfg(feature = "gpu")]
pub(super) mod gpu {
    use super::super::card_packed::no_farther;
    use super::super::compressed_band::BandCell;
    use super::super::hyper::sinkhorn::SinkhornMixer;
    use super::super::mixer::Mixer;
    use super::super::wide::staged;
    use super::super::wide_sparse_layer::{
        attention_half, engram_row, ffn_half, in_sequence, onto_first, token_mixer_out,
        SequenceMixers, SinkhornBlock,
    };
    use super::super::GenericHeteroTransformer;
    use super::{CardSinkhorn, ADMITTED, REFUSED};
    use crate::tensor::cuda::{pack_download, upload_f32, CudaDevice, DecodeStreamDev};
    use crate::tensor::gpu_runtime::driver::CudaSlice;
    use crate::tensor::ops::rms_norm;
    use crate::tensor::{Device, Error, Result, Tensor};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// 1 Sinkhorn mixer on a card.
    pub(in super::super) struct Mix {
        func: CudaSlice<f32>,
        scale: CudaSlice<f32>,
        base: CudaSlice<f32>,
        iters: usize,
        hc_eps: f32,
        norm_eps: f32,
    }

    impl Mix {
        fn new(dev: &CudaDevice, m: &SinkhornMixer) -> Result<Self> {
            let (func, scale, base) = m.weights();
            let func = func.flatten_all()?.to_vec1::<f32>()?;
            let (iters, hc_eps, norm_eps) = m.epsilons();
            Ok(Self {
                func: upload_f32(dev, "Sinkhorn projection", &func)?,
                scale: upload_f32(dev, "Sinkhorn scales", scale)?,
                base: upload_f32(dev, "Sinkhorn biases", base)?,
                iters,
                hc_eps,
                norm_eps,
            })
        }

        fn floats(&self) -> usize {
            self.func.len() + self.scale.len() + self.base.len()
        }

        /// The coefficients of the stream `s` as it stands, kept in the stream.
        fn mixes(&self, dev: &CudaDevice, s: &mut DecodeStreamDev) -> Result<()> {
            s.mixes(
                dev,
                &self.func.slice(0..self.func.len()),
                &self.scale.slice(0..self.scale.len()),
                &self.base.slice(0..self.base.len()),
                self.iters,
                self.norm_eps,
                self.hc_eps,
            )
        }
    }

    /// What a card keeps of a layer.
    pub(in super::super) struct Held {
        pub(super) dev: Arc<CudaDevice>,
        attn: Mix,
        ffn: Mix,
        attn_norm: CudaSlice<f32>,
        ffn_norm: CudaSlice<f32>,
        /// The epsilon of both norms.
        eps: f32,
        hc: usize,
        d: usize,
        pub(super) bytes: usize,
    }

    /// A product's relative distance to the exact one, for the host and for the card, over sums
    /// of `terms` terms.
    struct Distance {
        what: &'static str,
        host: f64,
        card: f64,
        terms: usize,
    }

    /// The relative L2 distance of `got` to `exact`.
    fn distance(exact: &[f64], got: &[f32]) -> f64 {
        let (mut dd, mut ww) = (0f64, 0f64);
        for (&w, &g) in exact.iter().zip(got) {
            dd += (w - g as f64) * (w - g as f64);
            ww += w * w;
        }
        (dd / ww.max(f64::MIN_POSITIVE)).sqrt()
    }

    impl Held {
        pub(super) fn new(ordinal: usize, block: &SinkhornBlock) -> Result<Self> {
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("Sinkhorn mixers on a card need a CUDA device"));
            };
            let vector = |t: &Tensor| super::super::card_mixers::vector(&dev, t);
            let attn = Mix::new(&dev, &block.attn_mix)?;
            let ffn = Mix::new(&dev, &block.ffn_mix)?;
            let attn_norm = vector(&block.attn_norm)?;
            let ffn_norm = vector(&block.ffn_norm)?;
            let floats = attn.floats() + ffn.floats() + attn_norm.len() + ffn_norm.len();
            Ok(Self {
                attn,
                ffn,
                attn_norm,
                ffn_norm,
                eps: block.eps,
                hc: block.attn_mix.streams(),
                d: block.attn_mix.dim(),
                bytes: floats * std::mem::size_of::<f32>(),
                dev,
            })
        }

        /// The sublayer input of the stream `s`: mixes, collapse and norm on the card, the
        /// collapse weights committed, the input read back.
        fn input(
            &self,
            s: &mut DecodeStreamDev,
            mix: &Mix,
            norm: &CudaSlice<f32>,
        ) -> Result<Tensor> {
            let dev: &CudaDevice = &self.dev;
            mix.mixes(dev, s)?;
            let i = s.collapse(dev, &norm.slice(0..self.d), self.eps)?;
            let host = pack_download(dev, &[i.slice(0..self.d)])?;
            s.commit_pre();
            Tensor::from_vec(host, (1, 1, self.d), &Device::Cpu)
        }

        /// The sublayer input of the stream `s`, left on the card: mixes, collapse and norm, the
        /// collapse weights committed.
        fn input_dev(
            &self,
            s: &mut DecodeStreamDev,
            mix: &Mix,
            norm: &CudaSlice<f32>,
        ) -> Result<CudaSlice<f32>> {
            let dev: &CudaDevice = &self.dev;
            mix.mixes(dev, s)?;
            let i = s.collapse(dev, &norm.slice(0..self.d), self.eps)?;
            s.commit_pre();
            Ok(i)
        }

        /// The sublayer output `out` folded into the stream `s`.
        fn fold(&self, s: &mut DecodeStreamDev, out: &Tensor) -> Result<()> {
            let dev: &CudaDevice = &self.dev;
            let up = upload_f32(
                dev,
                "sublayer output",
                &out.flatten_all()?.to_vec1::<f32>()?,
            )?;
            s.post(dev, &up.slice(0..self.d))
        }

        /// The distances of the card's products of `mixer`, kept as `mix` and `norm_dev`, to the
        /// exact ones, beside the host's, over the residual `r` `[1, 1, streams * hidden]`
        /// collapsed by `pre`.
        #[allow(clippy::too_many_arguments)]
        fn measure(
            &self,
            mixer: &SinkhornMixer,
            mix: &Mix,
            norm_dev: &CudaSlice<f32>,
            norm: &Tensor,
            r: &Tensor,
            pre: &[f32],
            out: &mut Vec<Distance>,
        ) -> Result<()> {
            let dev: &CudaDevice = &self.dev;
            let (hc, d) = (self.hc, self.d);
            let flat = hc * d;
            let x = r.flatten_all()?.to_vec1::<f32>()?;
            let w = mixer.weights().0.flatten_all()?.to_vec1::<f32>()?;
            let rows = w.len() / flat.max(1);
            let mut s = DecodeStreamDev::open(dev, &x, hc, d)?;
            s.replace(dev, &x, pre)?;

            // The dot products, then the residual with itself.
            let mut exact: Vec<f64> = w
                .chunks_exact(flat)
                .map(|wm| wm.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum())
                .collect();
            exact.push(x.iter().map(|&v| v as f64 * v as f64).sum());
            let host = mixer.single_dots(&x)?;
            let card = s.mixed_dots(dev, &mix.func.slice(0..mix.func.len()))?;
            out.push(Distance {
                what: "dots",
                host: distance(&exact[..rows], &host[..rows]),
                card: distance(&exact[..rows], &card[..rows]),
                terms: flat,
            });
            out.push(Distance {
                what: "square",
                host: distance(&exact[rows..], &host[rows..]),
                card: distance(&exact[rows..], &card[rows..]),
                terms: flat,
            });

            // The collapse and its norm.
            let wn = norm.flatten_all()?.to_vec1::<f32>()?;
            let v: Vec<f64> = (0..d)
                .map(|k| (0..hc).map(|c| pre[c] as f64 * x[c * d + k] as f64).sum())
                .collect();
            let ms = v.iter().map(|a| a * a).sum::<f64>() / d as f64;
            let den = (ms + self.eps as f64).sqrt();
            let exact: Vec<f64> = v
                .iter()
                .zip(&wn)
                .map(|(a, &g)| a / den * g as f64)
                .collect();
            let collapsed = rms_norm(&mixer.pre(r, &[pre.to_vec()])?, norm, self.eps)?;
            let host = collapsed.flatten_all()?.to_vec1::<f32>()?;
            let i = s.collapse(dev, &norm_dev.slice(0..d), self.eps)?;
            let card = pack_download(dev, &[i.slice(0..d)])?;
            out.push(Distance {
                what: "input",
                host: distance(&exact, &host),
                card: distance(&exact, &card),
                terms: d,
            });

            // The post of the collapsed input, with the host's coefficients.
            let m = mixer.mixes(r)?;
            let (post, comb) = (&m.post[0], m.comb[0].concat());
            let exact: Vec<f64> = (0..hc)
                .flat_map(|j| {
                    let (x, host, comb) = (&x, &host, &comb);
                    (0..d).map(move |k| {
                        let mut acc = post[j] as f64 * host[k] as f64;
                        for c in 0..hc {
                            acc += comb[c * hc + j] as f64 * x[c * d + k] as f64;
                        }
                        acc
                    })
                })
                .collect();
            let folded = mixer.post(&collapsed, r, &m.post, &m.comb)?;
            let host_post = folded.flatten_all()?.to_vec1::<f32>()?;
            s.set_post_comb(dev, post, &comb)?;
            let up = upload_f32(dev, "probe output", &host)?;
            s.post(dev, &up.slice(0..d))?;
            let card = s.rows(dev)?;
            out.push(Distance {
                what: "post",
                host: distance(&exact, &host_post),
                card: distance(&exact, &card),
                terms: hc + 1,
            });
            Ok(())
        }
    }

    impl CardSinkhorn {
        /// Measures the card's products of both mixers of `block`, layer `layer` of the stack,
        /// on the values of a decoded row, and admits or refuses the layer.
        ///
        /// `attn` is the residual the token mixer's half read and the pre mix it collapsed with.
        /// `ffn` is the same for the feed-forward's half.
        pub(in super::super) fn admit(
            &self,
            block: &SinkhornBlock,
            layer: usize,
            attn: (&Tensor, &[f32]),
            ffn: (&Tensor, &[f32]),
        ) {
            let Some(h) = self.held.as_ref() else {
                return;
            };
            let mut found = Vec::new();
            let measured = h
                .measure(
                    &block.attn_mix,
                    &h.attn,
                    &h.attn_norm,
                    &block.attn_norm,
                    attn.0,
                    attn.1,
                    &mut found,
                )
                .and_then(|_| {
                    h.measure(
                        &block.ffn_mix,
                        &h.ffn,
                        &h.ffn_norm,
                        &block.ffn_norm,
                        ffn.0,
                        ffn.1,
                        &mut found,
                    )
                });
            let admitted = match measured {
                Ok(()) => {
                    let ok = found.iter().all(|m| no_farther(m.host, m.card, m.terms));
                    let apart: f64 = found.iter().map(|m| m.host + m.card).sum();
                    self.apart.store(apart.to_bits(), Ordering::Relaxed);
                    let listed = found
                        .iter()
                        .map(|m| format!("{} {:.3e} / {:.3e}", m.what, m.host, m.card))
                        .collect::<Vec<_>>()
                        .join(", ");
                    tracing::info!(
                        "layer {layer}: card Sinkhorn mixers {}: distance to the exact products, host / card, token mixer then feed-forward: {listed}",
                        if ok { "admitted" } else { "refused, the host runs them" }
                    );
                    ok
                }
                Err(e) => {
                    tracing::warn!(
                        "layer {layer}: card Sinkhorn mixers not measured, the host runs them: {e}"
                    );
                    false
                }
            };
            let verdict = if admitted { ADMITTED } else { REFUSED };
            self.verdict.store(verdict, Ordering::Relaxed);
        }

        /// The host's and the card's seconds for both mixers of `block` over the residual row
        /// `x`, each sublayer with its input down and its output up, as `timed` times them.
        pub(in super::super) fn time_layer(
            &self,
            block: &SinkhornBlock,
            x: &[f32],
            timed: &mut dyn FnMut(
                &mut dyn FnMut() -> Result<()>,
            ) -> std::result::Result<f64, String>,
        ) -> std::result::Result<(f64, f64), String> {
            let text = |e: Error| e.to_string();
            let h = self.held.as_ref().ok_or("the mixers are not on a card")?;
            let r = Tensor::from_vec(x.to_vec(), (1, 1, x.len()), &Device::Cpu).map_err(text)?;
            let pre = vec![onto_first(h.hc)];
            let halves = [
                (&block.attn_mix, &block.attn_norm),
                (&block.ffn_mix, &block.ffn_norm),
            ];
            let host_secs = timed(&mut || {
                for (mix, norm) in halves {
                    let m = mix.mixes(&r)?;
                    let input = rms_norm(&mix.pre(&r, &pre)?, norm, block.eps)?;
                    mix.post(&input, &r, &m.post, &m.comb)?;
                }
                Ok(())
            })?;
            let dev: &CudaDevice = &h.dev;
            let mut s = DecodeStreamDev::open(dev, x, h.hc, h.d).map_err(text)?;
            let card_secs = timed(&mut || {
                for (mix, norm) in [(&h.attn, &h.attn_norm), (&h.ffn, &h.ffn_norm)] {
                    let input = h.input(&mut s, mix, norm)?;
                    h.fold(&mut s, &input)?;
                }
                s.pre_host(dev).map(drop)
            })?;
            Ok((host_secs, card_secs))
        }
    }

    /// Where a decoded row's residual is between 2 layers, with the pre mix the next sublayer
    /// collapses with.
    enum Residual {
        Host(Tensor, Vec<f32>),
        Card(DecodeStreamDev, Arc<CudaDevice>),
    }

    impl Residual {
        /// The residual `[1, 1, streams * hidden]` and the pre mix on the host.
        fn down(self) -> Result<(Tensor, Vec<f32>)> {
            match self {
                Residual::Host(r, pre) => Ok((r, pre)),
                Residual::Card(s, dev) => {
                    let x = s.rows(&dev)?;
                    let pre = s.pre_host(&dev)?;
                    let n = x.len();
                    Ok((Tensor::from_vec(x, (1, 1, n), &Device::Cpu)?, pre))
                }
            }
        }

        /// The residual on the card of `h`.
        fn onto(self, h: &Held) -> Result<Self> {
            match self {
                Residual::Card(s, dev) if dev.ordinal() == h.dev.ordinal() => {
                    Ok(Residual::Card(s, dev))
                }
                other => {
                    let (r, pre) = other.down()?;
                    let x = r.flatten_all()?.to_vec1::<f32>()?;
                    let mut s = DecodeStreamDev::open(&h.dev, &x, h.hc, h.d)?;
                    s.replace(&h.dev, &x, &pre)?;
                    Ok(Residual::Card(s, Arc::clone(&h.dev)))
                }
            }
        }
    }

    /// How far a layer's step on the card went.
    enum Step {
        Done,
        /// A card call failed before the token mixer ran: the stream holds the residual the
        /// layer found, with its engram added when `engram_added`.
        Early {
            engram_added: bool,
            why: Error,
        },
    }

    /// Layer `il` over the stream `s` on the card of `h`, for the row of sequence `seq` at `pos`.
    fn card_layer(
        (block, mixer): (&SinkhornBlock, &mut Mixer),
        il: usize,
        h: &Held,
        s: &mut DecodeStreamDev,
        (pos, seq, cell): (usize, &mut SequenceMixers, &BandCell),
    ) -> Result<Step> {
        let dev: &CudaDevice = &h.dev;
        let mut engram_added = false;
        if block.engram.is_some() {
            let found = s.rows(dev).and_then(|x| Ok((x, s.pre_host(dev)?)));
            let (x, pre) = match found {
                Ok(v) => v,
                Err(why) => return Ok(Step::Early { engram_added, why }),
            };
            let n = x.len();
            let r = engram_row(
                block,
                Tensor::from_vec(x, (1, 1, n), &Device::Cpu)?,
                seq,
                pos,
            )?;
            s.replace(dev, &r.flatten_all()?.to_vec1::<f32>()?, &pre)?;
            engram_added = true;
        }
        // A mixer that decodes on the stream's card reads its input and leaves its output there.
        // The rows its state keeps wait on the card for the feed-forward's download.
        let mut pending = None;
        if mixer.decodes_on(dev.ordinal()) {
            let input = staged("mix_token", &Device::Cpu, false, || {
                h.input_dev(s, &h.attn, &h.attn_norm)
            });
            let input = match input {
                Ok(i) => i,
                Err(why) => return Ok(Step::Early { engram_added, why }),
            };
            let (out, rows) = in_sequence(mixer, il, seq, cell, |mixer| {
                staged("token mixer", &Device::Cpu, false, || {
                    mixer.decode_dev(input.slice(0..h.d), pos)
                })
            })?;
            staged("combine", &Device::Cpu, false, || {
                s.post(dev, &out.slice(0..h.d))
            })?;
            pending = Some(rows);
        } else {
            let input = staged("mix_token", &Device::Cpu, false, || {
                h.input(s, &h.attn, &h.attn_norm)
            });
            let input = match input {
                Ok(i) => i,
                Err(why) => return Ok(Step::Early { engram_added, why }),
            };
            let out = in_sequence(mixer, il, seq, cell, |mixer| {
                token_mixer_out(block, mixer, &input, pos)
            })?;
            staged("combine", &Device::Cpu, false, || h.fold(s, &out))?;
        }
        // A router on the stream's card reads the feed-forward's input there.
        // Its logits come down with the input and the rows the mixer's state keeps.
        let (input, logits) = match block.ffn.card.router_on(dev.ordinal()) {
            Some(router) => staged("mix_ffn", &Device::Cpu, false, || {
                let i = h.input_dev(s, &h.ffn, &h.ffn_norm)?;
                let y = router.logits_dev(&i.slice(0..h.d))?;
                let mut parts = vec![i.slice(0..h.d), y.slice(0..router.out())];
                if let Some(p) = &pending {
                    parts.extend(p.views());
                }
                let host = pack_download(dev, &parts)?;
                let (front, kept) = host.split_at(h.d + router.out());
                if let Some(p) = &pending {
                    in_sequence(mixer, il, seq, cell, |mixer| {
                        mixer.keep_decoded(&p.keep, kept)
                    })?;
                }
                let input = Tensor::from_vec(front[..h.d].to_vec(), (1, 1, h.d), &Device::Cpu)?;
                Ok((input, Some(front[h.d..].to_vec())))
            })?,
            None => {
                let input = staged("mix_ffn", &Device::Cpu, false, || {
                    let input = h.input(s, &h.ffn, &h.ffn_norm)?;
                    if let Some(p) = &pending {
                        let kept = p.download()?;
                        in_sequence(mixer, il, seq, cell, |mixer| {
                            mixer.keep_decoded(&p.keep, &kept)
                        })?;
                    }
                    Ok(input)
                })?;
                (input, None)
            }
        };
        let out = staged("experts", &Device::Cpu, false, || {
            block.ffn.forward_routed(&input, logits)
        })?;
        staged("combine", &Device::Cpu, false, || h.fold(s, &out))?;
        Ok(Step::Done)
    }

    /// Layer `il` on the host over the residual `r` collapsed by `pre`, for the row of sequence
    /// `seq` at `pos`: the residual after the layer and the pre mix it left.
    ///
    /// The engram is added unless `engram_added`.
    /// A layer whose card products are not yet measured measures them on the row's values.
    fn host_layer(
        (block, mixer): (&SinkhornBlock, &mut Mixer),
        il: usize,
        (r, pre, engram_added): (Tensor, Vec<f32>, bool),
        (pos, seq, cell): (usize, &mut SequenceMixers, &BandCell),
        stack_layer: usize,
    ) -> Result<(Tensor, Vec<f32>)> {
        let r = match engram_added {
            true => r,
            false => engram_row(block, r, seq, pos)?,
        };
        let pres = vec![pre];
        let (mid, attn_pre, _) = in_sequence(mixer, il, seq, cell, |mixer| {
            attention_half(block, mixer, &r, &pres, pos)
        })?;
        let (next, ffn_pre) = ffn_half(block, &mid, &attn_pre)?;
        if block.card_mix.measures() {
            let attn = (&r, pres[0].as_slice());
            let ffn = (&mid, attn_pre.first().map_or(&[][..], Vec::as_slice));
            block.card_mix.admit(block, stack_layer, attn, ffn);
        }
        let pre = ffn_pre.into_iter().next().unwrap_or_default();
        Ok((next, pre))
    }

    impl GenericHeteroTransformer {
        /// Whether a layer keeps its Sinkhorn mixers on a card.
        pub(in super::super) fn layers_keep_sinkhorn_on_cards(&self) -> bool {
            self.layers.iter().any(|l| {
                l.sinkhorn
                    .as_ref()
                    .is_some_and(|b| b.card_mix.ordinal().is_some())
            })
        }

        /// The logits `[1, vocab]` of the residual `s` on the card `dev` that keeps the head,
        /// before the output bias, scale and softcap, and the pre mix in hand.
        ///
        /// `None` leaves the head to the host:
        /// - the head is on another card or on the host, or a feature capture reads the residual,
        /// - the card's collapse is not admitted, or this row measures it,
        /// - a card call fails.
        fn head_on_card(
            &self,
            s: &DecodeStreamDev,
            dev: &Arc<CudaDevice>,
        ) -> Option<(Tensor, Vec<f32>)> {
            let ws = self.wide_sparse.as_ref()?;
            if self.capture_feature || ws.card_head.ordinal() != Some(dev.ordinal()) {
                return None;
            }
            let collapse = &self.layers.last()?.sinkhorn.as_ref()?.ffn_mix;
            match ws.head_collapse.verdict() {
                Some(true) => {}
                Some(false) => return None,
                None => {
                    let mut apart = 0.0;
                    let admitted = match measure_head(s, dev, collapse, &ws.norm, ws.eps) {
                        Ok((host, card)) => {
                            let ok = no_farther(host, card, s.d());
                            apart = host + card;
                            tracing::info!(
                                "final collapse on card {} {}: distance to the exact one, host / card: {host:.3e} / {card:.3e}",
                                dev.ordinal(),
                                if ok { "admitted" } else { "refused, the host runs it" }
                            );
                            ok
                        }
                        Err(e) => {
                            tracing::warn!(
                                "final collapse on card {} not measured, the host runs it: {e}",
                                dev.ordinal()
                            );
                            false
                        }
                    };
                    ws.head_collapse.set(admitted, apart);
                    return None;
                }
            }
            let run = || -> Result<Option<(Tensor, Vec<f32>)>> {
                let d = s.d();
                let norm =
                    upload_f32(dev, "final norm", &ws.norm.flatten_all()?.to_vec1::<f32>()?)?;
                let h = s.collapse(dev, &norm.slice(0..d), ws.eps)?;
                let Some((y, vocab)) = ws.card_head.apply_dev(&h)? else {
                    return Ok(None);
                };
                let host = pack_download(dev, &[y.slice(0..vocab), s.pre_view()])?;
                let logits = Tensor::from_vec(host[..vocab].to_vec(), (1, vocab), &Device::Cpu)?;
                Ok(Some((logits, host[vocab..].to_vec())))
            };
            match run() {
                Ok(out) => out,
                Err(e) => {
                    tracing::warn!(
                        "card {}: head not run, the host runs it: {e}",
                        dev.ordinal()
                    );
                    None
                }
            }
        }

        /// The decode step of the row `row` `[1, 1, streams * hidden]` of sequence `seq` at
        /// `pos`, through every layer, with the pre mix the last layer left.
        ///
        /// A layer that serves on its card keeps the residual there.
        /// A residual that ends on the card that keeps the head gives the head's logits.
        pub(in super::super) fn decode_row_on_cards(
            &mut self,
            row: Tensor,
            pos: usize,
            seq: &mut SequenceMixers,
            cell: &BandCell,
        ) -> Result<Decoded> {
            let streams = self.wide_sparse.as_ref().map_or(1, |ws| ws.streams);
            let r = row
                .to_device(&Device::Cpu)?
                .to_dtype(crate::tensor::DType::F32)?;
            let mut res = Residual::Host(r.contiguous()?, onto_first(streams));
            for il in 0..self.layers.len() {
                let stack_layer = self.first_layer + il;
                let layer = &mut self.layers[il];
                let block = layer
                    .sinkhorn
                    .as_ref()
                    .ok_or_else(|| Error::msg("a Sinkhorn layer without its block"))?;
                let mixer = &mut layer.mixer;
                let mut engram_added = false;
                if block.card_mix.serves() {
                    if let Some(h) = block.card_mix.held.as_ref() {
                        let mut at = res.onto(h)?;
                        let Residual::Card(s, _) = &mut at else {
                            return Err(Error::msg("the residual did not reach the card"));
                        };
                        match card_layer((block, &mut *mixer), il, h, s, (pos, &mut *seq, cell))? {
                            Step::Done => {
                                res = at;
                                continue;
                            }
                            Step::Early {
                                engram_added: added,
                                why,
                            } => {
                                tracing::warn!(
                                    "layer {stack_layer}: the card's step failed, the host runs the layer: {why}"
                                );
                                engram_added = added;
                                res = at;
                            }
                        }
                    }
                }
                let (r, pre) = res.down()?;
                let (next, pre) = host_layer(
                    (block, mixer),
                    il,
                    (r, pre, engram_added),
                    (pos, &mut *seq, cell),
                    stack_layer,
                )?;
                res = Residual::Host(next, pre);
            }
            if let Residual::Card(s, dev) = &res {
                let head = staged(
                    "head",
                    &Device::Cpu,
                    false,
                    || Ok(self.head_on_card(s, dev)),
                )?;
                if let Some((logits, pre)) = head {
                    return Ok(Decoded::Logits(logits, pre));
                }
            }
            let (r, pre) = res.down()?;
            Ok(Decoded::Rows(r, pre))
        }
    }

    /// What a decode step of a row through every layer gives, with the pre mix the last layer
    /// left.
    pub(in super::super) enum Decoded {
        /// The residual after the last layer, on the host.
        Rows(Tensor, Vec<f32>),
        /// The head's logits `[1, vocab]`, before the output bias, scale and softcap.
        Logits(Tensor, Vec<f32>),
    }

    /// The distances of the host's and the card's final collapse and norm of the residual `s` to
    /// the exact one: `collapse` is a Sinkhorn mixer of the model and `norm` the final norm.
    fn measure_head(
        s: &DecodeStreamDev,
        dev: &CudaDevice,
        collapse: &SinkhornMixer,
        norm: &Tensor,
        eps: f32,
    ) -> Result<(f64, f64)> {
        let (hc, d) = (s.hc(), s.d());
        let x = s.rows(dev)?;
        let pre = s.pre_host(dev)?;
        let wn = norm.flatten_all()?.to_vec1::<f32>()?;
        let v: Vec<f64> = (0..d)
            .map(|k| (0..hc).map(|c| pre[c] as f64 * x[c * d + k] as f64).sum())
            .collect();
        let ms = v.iter().map(|a| a * a).sum::<f64>() / d as f64;
        let den = (ms + eps as f64).sqrt();
        let exact: Vec<f64> = v
            .iter()
            .zip(&wn)
            .map(|(a, &g)| a / den * g as f64)
            .collect();
        let r = Tensor::from_vec(x, (1, 1, hc * d), &Device::Cpu)?;
        let host = rms_norm(&collapse.pre(&r, &[pre])?, norm, eps)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let up = upload_f32(dev, "final norm", &wn)?;
        let i = s.collapse(dev, &up.slice(0..d), eps)?;
        let card = pack_download(dev, &[i.slice(0..d)])?;
        Ok((distance(&exact, &host), distance(&exact, &card)))
    }
}
