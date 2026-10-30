//! A decode token's layers on the cards: the stream, the attention with its caches, the router
//! and the shared expert, with one crossing out and one back per layer (see `DecodeStream`).
//!
//! A layer runs on the card that keeps its dense weights; the stream moves between cards when
//! two layers' weights sit on different ones. A layer's cache rows are the card's copy of the
//! host's: a token appends to the copy first, the host gets the rows at the layer's crossing,
//! and the copy is checked against the host's counts and newest rows before every use, so a
//! rewind, a restore or a host-run token is seen and the copy rebuilt.

use super::*;
use crate::inference::offload::{
    CacheAppend, Crossing, DecodeAttn, DecodeRouter, DecodeStream, DevCache, HcDev, HostCache,
};
use crate::tensor::cuda::{
    alloc_i32, apply_candidate_mask, candidate_mask, compress_pool, copy_f32, fp4_round, fp8_round, gather_ring, index_scores_dev,
    pack_download, rope_rows, router_select, sparse_attention_dev, topk_select, upload_f32,
    upload_i32, DecodeStreamDev, GrowDev, RingDev,
};
use crate::tensor::cuda_ext::{f32_slice_of, tensor_from_f32_slice, TensorCudaSlice};
use crate::tensor::gpu_runtime::driver::CudaView;
use crate::tensor::ops::traits::TensorId;

/// The block of values an fp8 window row shares a scale over, and the blocks of the fp4
/// compressed rows (raw scale) and index keys (power-of-two scale): the reference's cache
/// formats, as the host path rounds them.
const FP8_BLOCK: usize = 32;
const FP4_KV_BLOCK: usize = 16;
const FP4_INDEX_BLOCK: usize = 32;

/// What the host cache is expected to hold once the rows the card appended have reached it.
struct Expect {
    window: usize,
    groups: usize,
    group_rows: usize,
    last_window: Vec<f32>,
    last_group: Option<Vec<f32>>,
}

impl Expect {
    fn matches(&self, host: &dyn HostCache) -> bool {
        let same = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        self.window == host.window_rows()
            && self.groups == host.groups()
            && self.group_rows == host.group_rows()
            && host
                .last_window_row()
                .is_some_and(|r| same(r, &self.last_window))
            && match (&self.last_group, host.last_group_row()) {
                (None, None) => true,
                (Some(a), Some(b)) => same(a, b),
                _ => false,
            }
    }
}

/// The rows the last attention appended, kept until the host reads them.
struct Pending {
    window_row: CudaSlice<f32>,
    group_row: Option<CudaSlice<f32>>,
    comp_row: Option<CudaSlice<f32>>,
    index_row: Option<CudaSlice<f32>>,
    topk: Option<CudaSlice<f32>>,
}

/// One layer's attention rows on a card.
struct LayerCacheDev {
    ordinal: usize,
    ring: RingDev,
    comp: GrowDev,
    index_k: GrowDev,
    group: GrowDev,
    n_groups: usize,
    expect: Option<Expect>,
    pending: Option<Pending>,
}

impl LayerCacheDev {
    /// The card's copy of `host`, rows uploaded whole.
    fn from_host(
        card: &Card,
        host: &dyn HostCache,
        window: usize,
        hd: usize,
        ihd: usize,
        dim: usize,
    ) -> Result<Self> {
        let dev = &card.dev;
        let groups = host.groups();
        let comp = host.compressed();
        let keys = host.index_keys();
        if (groups > 0 && comp.len() != groups * hd)
            || (!keys.is_empty() && keys.len() != groups * ihd)
        {
            return Err(Error::msg(format!(
                "a layer cache of {groups} groups holds {} compressed values and {} index keys",
                comp.len(),
                keys.len()
            )));
        }
        Ok(Self {
            ordinal: card.ordinal,
            ring: RingDev::open(dev, &host.window_flat(), window, hd)?,
            comp: GrowDev::open(dev, comp, hd)?,
            index_k: GrowDev::open(dev, keys, ihd.max(1))?,
            group: GrowDev::open(dev, &host.group_flat(), dim)?,
            n_groups: groups,
            expect: None,
            pending: None,
        })
    }

    /// Read the pending rows back and note what the host holds once it appends them.
    fn take(&mut self, dev: &CudaDevice) -> Result<Option<CacheAppend>> {
        let Some(p) = self.pending.take() else {
            return Ok(None);
        };
        let hd = p.window_row.len();
        let mut parts: Vec<CudaView<'_, f32>> = vec![p.window_row.slice(0..hd)];
        let mut lens = vec![hd];
        for s in [&p.group_row, &p.comp_row, &p.index_row, &p.topk]
            .into_iter()
            .flatten()
        {
            parts.push(s.slice(0..s.len()));
            lens.push(s.len());
        }
        let flat = pack_download(dev, &parts)?;
        let mut at = 0;
        let mut next = |len: usize| {
            let v = flat[at..at + len].to_vec();
            at += len;
            v
        };
        let window_row = next(lens[0]);
        let mut k = 1;
        let mut opt = |present: bool| {
            if present {
                let v = next(lens[k]);
                k += 1;
                Some(v)
            } else {
                None
            }
        };
        let group_row = opt(p.group_row.is_some());
        let comp_row = opt(p.comp_row.is_some());
        let index_row = opt(p.index_row.is_some());
        let topk = opt(p.topk.is_some()).map(|v| v.iter().map(|&f| f as i32).collect());
        self.expect = Some(Expect {
            window: self.ring.len(),
            groups: self.n_groups,
            group_rows: self.group.len(),
            last_window: window_row.clone(),
            last_group: if comp_row.is_some() {
                None
            } else {
                group_row.clone()
            },
        });
        Ok(Some(CacheAppend {
            window_row,
            group_row,
            comp_row,
            index_row,
            topk,
        }))
    }
}

/// The three rope rows of one position: cosines, sines, negated sines.
struct RopeRows {
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    neg: CudaSlice<f32>,
}

/// The cosine, sine and negated sine rows kept under `key`, `half` values each.
fn rope_views<'r>(
    rope: &'r HashMap<(TensorId, usize), RopeRows>,
    key: (TensorId, usize),
    half: usize,
) -> (CudaView<'r, f32>, CudaView<'r, f32>, CudaView<'r, f32>) {
    let r = &rope[&key];
    (
        r.cos.slice(0..half),
        r.sin.slice(0..half),
        r.neg.slice(0..half),
    )
}

/// A device tensor's values as a kernel argument.
fn dview(t: &Tensor) -> Result<TensorCudaSlice<f32>> {
    f32_slice_of(t)
}

fn tensor_of(dev: &Device, s: CudaSlice<f32>, shape: (usize, usize)) -> Result<Tensor> {
    tensor_from_f32_slice(s, shape, dev)
}

impl Card {
    /// `t` on this card as an f32 device tensor, kept by its storage. `None` when the card has no
    /// room for it.
    fn keep_tensor(&self, t: &Tensor) -> Option<Arc<Tensor>> {
        let key = t.id();
        if let Some(k) = self
            .tensors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Some(k.clone());
        }
        let bytes = t.elem_count() * F32;
        if !self.take_room(bytes, true) {
            return None;
        }
        let sent = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Tensor> {
            t.contiguous()?
                .to_dtype(crate::tensor::DType::F32)?
                .to_device(&Device::Cuda(self.dev.clone()))
        }));
        let Ok(Ok(d)) = sent else {
            self.give_room(bytes);
            self.say_once("a dense weight of the decode could not go over");
            return None;
        };
        let d = Arc::new(d);
        self.tensors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, d.clone());
        Some(d)
    }

    /// `p` on this card as a product, kept.
    fn keep_proj(&self, p: &Projection) -> Option<Arc<QMatMul>> {
        match p {
            Projection::Quant(q) => self.kept(q),
            Projection::Dense(w) => self.kept_dense(w),
            _ => None,
        }
    }

    /// The hyper-connection weights of a sublayer on this card.
    fn keep_hc(&self, hc: &HcDev) -> Option<[Arc<Tensor>; 3]> {
        Some([
            self.keep_tensor(hc.func)?,
            self.keep_tensor(hc.scale)?,
            self.keep_tensor(hc.base)?,
        ])
    }

    /// Run `mixes` of `s` with `hc`'s weights.
    fn mixes(&self, s: &mut DecodeStreamDev, w: &[Arc<Tensor>; 3], hc: &HcDev) -> Result<()> {
        let (f, sc, b) = (dview(&w[0])?, dview(&w[1])?, dview(&w[2])?);
        s.mixes(
            &self.dev,
            &f.view()?,
            &sc.view()?,
            &b.view()?,
            hc.iters,
            hc.norm_eps,
            hc.hc_eps,
        )
    }

    /// The attention sublayer on this card. `None` when a weight of it cannot be kept here.
    #[allow(clippy::too_many_arguments)]
    fn decode_attention(
        &self,
        s: &mut DecodeStreamDev,
        rope: &mut HashMap<(TensorId, usize), RopeRows>,
        topk_in: Option<&(usize, usize, Arc<CudaSlice<i32>>)>,
        candidates: &mut Option<(usize, Arc<CudaSlice<f32>>)>,
        hc: &HcDev,
        a: &DecodeAttn,
        cache: &mut DevCache,
        host: &dyn HostCache,
        source: Option<(&DevCache, &dyn HostCache)>,
    ) -> Option<Result<Option<Arc<CudaSlice<i32>>>>> {
        let hcw = self.keep_hc(hc)?;
        let norm = self.keep_tensor(a.norm)?;
        let q_norm = self.keep_tensor(a.q_norm)?;
        let kv_norm = self.keep_tensor(a.kv_norm)?;
        let sink = self.keep_tensor(a.sink)?;
        let m_qa = self.keep_proj(a.wq_a)?;
        let m_qb = self.keep_proj(a.wq_b)?;
        let m_kv = self.keep_proj(a.wkv)?;
        let mut wo = Vec::with_capacity(a.o_groups);
        for g in 0..a.o_groups {
            wo.push(self.keep_proj(&a.wo_a.rows(g * a.o_lora, a.o_lora).ok()?)?);
        }
        let wob = self.keep_proj(a.wo_b)?;
        let comp = match &a.compressor {
            Some(c) => Some((
                self.keep_tensor(c.norm)?,
                self.kept_dense(c.wkv)?,
                match c.wgate {
                    Some(g) => Some(self.kept_dense(g)?),
                    None => None,
                },
            )),
            None => None,
        };
        let idx = match &a.indexer {
            Some(i) => Some((
                self.keep_proj(i.wq_b)?,
                self.kept_dense(i.weights_proj)?,
                match (i.wk, i.k_norm) {
                    (Some(wk), Some(kn)) => Some((self.kept_dense(wk)?, self.keep_tensor(kn)?)),
                    _ => None,
                },
            )),
            None => None,
        };
        let (nh, hd, rd, dim) = (a.n_heads, a.head_dim, a.rope_head_dim, a.dim);
        let ihd = a.indexer.as_ref().map_or(0, |i| i.head_dim);
        let itopk = if a.ratio == 0 { 0 } else { a.index_topk };
        let gtot = host.groups().max(1);
        let need = (6 * dim
            + 8 * nh * hd
            + (a.window + itopk) * (hd + nh + 2)
            + 2 * a.ratio.max(1) * dim
            + gtot
            + 2 * host.groups() * (hd + ihd))
            * F32;
        let dev = Device::Cuda(self.dev.clone());
        let cd = &self.dev;
        let ran = self.attempt("a decode layer's attention", need, || {
            // The card's rows of this layer, rebuilt when they do not match the host's.
            let current = cache
                .0
                .as_ref()
                .and_then(|b| b.downcast_ref::<LayerCacheDev>())
                .is_some_and(|lc| {
                    lc.ordinal == self.ordinal
                        && lc.pending.is_none()
                        && lc.expect.as_ref().is_some_and(|e| e.matches(host))
                });
            if !current {
                let built = LayerCacheDev::from_host(self, host, a.window, hd, ihd, dim)?;
                cache.0 = Some(Box::new(built));
            }
            let lc = cache
                .0
                .as_mut()
                .and_then(|b| b.downcast_mut::<LayerCacheDev>())
                .ok_or_else(|| Error::msg("the layer's card copy is missing"))?;
            // The rope rows of this position for this table.
            let rows_at =
                |rope: &mut HashMap<(TensorId, usize), RopeRows>, pos: usize| -> Result<()> {
                    let key = (a.cos.id(), pos);
                    if let std::collections::hash_map::Entry::Vacant(slot) = rope.entry(key) {
                        let row = |t: &Tensor| -> Result<Vec<f32>> {
                            t.narrow(0, pos, 1)?.flatten_all()?.to_vec1::<f32>()
                        };
                        let (c, sn) = (row(a.cos)?, row(a.sin)?);
                        let neg: Vec<f32> = sn.iter().map(|v| -v).collect();
                        slot.insert(RopeRows {
                            cos: upload_f32(cd, "rope rows", &c)?,
                            sin: upload_f32(cd, "rope rows", &sn)?,
                            neg: upload_f32(cd, "rope rows", &neg)?,
                        });
                    }
                    Ok(())
                };
            rows_at(rope, a.pos)?;
            let half = rd / 2;
            let table = a.cos.id();

            self.mixes(s, &hcw, hc)?;
            let xin = s.collapse(cd, &dview(&norm)?.view()?, hc.norm_eps)?;
            let xin_view = xin.slice(0..dim);
            let xin_t = tensor_of(&dev, copy_f32(cd, "layer input", &xin_view)?, (1, dim))?;

            // The query: latent, normed, to the heads, rotated.
            let qr = rms_norm(&m_qa.forward(&xin_t)?, &q_norm, a.eps)?;
            let q = m_qb.forward(&qr)?;
            let mut qs = copy_f32(cd, "queries", &dview(&q)?.view()?)?;
            {
                let (c, sn, _) = rope_views(rope, (table, a.pos), half);
                rope_rows(cd, &mut qs, nh, hd, rd, &c, &sn)?;
            }
            // This token's key: normed, rotated, rounded to fp8, appended to the window.
            let kv = rms_norm(&m_kv.forward(&xin_t)?, &kv_norm, a.eps)?;
            let mut kvs = copy_f32(cd, "key", &dview(&kv)?.view()?)?;
            {
                let (c, sn, _) = rope_views(rope, (table, a.pos), half);
                rope_rows(cd, &mut kvs, 1, hd, rd, &c, &sn)?;
            }
            fp8_round(cd, &mut kvs, FP8_BLOCK)?;
            lc.ring.append(cd, &kvs.slice(0..hd))?;
            let mut pending = Pending {
                window_row: kvs,
                group_row: None,
                comp_row: None,
                index_row: None,
                topk: None,
            };

            // A kv-source buffers the raw row and pools a group when the buffer fills.
            if let Some((c_norm, m_cwkv, m_cgate)) = &comp {
                lc.group.append(cd, &xin_view)?;
                pending.group_row = Some(copy_f32(cd, "group row", &xin_view)?);
                if lc.group.len() == a.ratio {
                    let g = lc.n_groups;
                    let xg = tensor_of(
                        &dev,
                        copy_f32(cd, "group", &lc.group.view())?,
                        (a.ratio, dim),
                    )?;
                    let kvg = m_cwkv.forward(&xg)?;
                    let pooled = match m_cgate {
                        Some(gate) => {
                            let score = gate.forward(&xg)?;
                            compress_pool(
                                cd,
                                &dview(&kvg)?.view()?,
                                &dview(&score)?.view()?,
                                a.ratio,
                                hd,
                            )?
                        }
                        None if a.ratio == 1 => copy_f32(cd, "latent", &dview(&kvg)?.view()?)?,
                        None => return Err(Error::msg("compressor without a gate at ratio > 1")),
                    };
                    let latent = rms_norm(&tensor_of(&dev, pooled, (1, hd))?, c_norm, a.eps)?;
                    rows_at(rope, g * a.ratio)?;
                    let mut lr = copy_f32(cd, "compressed row", &dview(&latent)?.view()?)?;
                    {
                        let (c, sn, _) = rope_views(rope, (table, g * a.ratio), half);
                        rope_rows(cd, &mut lr, 1, hd, rd, &c, &sn)?;
                    }
                    fp4_round(cd, &mut lr, FP4_KV_BLOCK, false)?;
                    lc.comp.append(cd, &lr.slice(0..hd))?;
                    pending.comp_row = Some(lr);
                    if let Some((_, _, Some((m_wk, k_norm)))) = &idx {
                        let ieps = a.indexer.as_ref().map_or(a.eps, |i| i.eps);
                        let k = rms_norm(&m_wk.forward(&latent)?, k_norm, ieps)?;
                        let mut ks = copy_f32(cd, "index key", &dview(&k)?.view()?)?;
                        {
                            let (c, sn, _) = rope_views(rope, (table, g * a.ratio), half);
                            rope_rows(cd, &mut ks, 1, ihd, rd, &c, &sn)?;
                        }
                        fp4_round(cd, &mut ks, FP4_INDEX_BLOCK, true)?;
                        lc.index_k.append(cd, &ks.slice(0..ihd))?;
                        pending.index_row = Some(ks);
                    }
                    lc.group.clear();
                    lc.n_groups += 1;
                }
            }

            // The compressed rows and index keys this layer attends over: its own or its source's.
            let w = lc.ring.len();
            let src = match source {
                Some((d, h)) => {
                    let sc =
                        d.0.as_ref()
                            .and_then(|b| b.downcast_ref::<LayerCacheDev>())
                            .filter(|sc| {
                                sc.ordinal == self.ordinal
                                    && sc.pending.is_none()
                                    && sc.expect.as_ref().is_some_and(|e| e.matches(h))
                            });
                    match sc {
                        Some(sc) => Some(sc),
                        None => {
                            return Err(Error::msg("the kv-source's rows are not on this card"))
                        }
                    }
                }
                None => None,
            };
            let mut published: Option<Arc<CudaSlice<i32>>> = None;
            let slots: Option<Arc<CudaSlice<i32>>> = match (&a.indexer, &idx) {
                (Some(ix), Some((m_iqb, m_wproj, own))) => {
                    let (keys, gtot) = match (own, src) {
                        (Some(_), _) => (lc.index_k.view(), lc.n_groups),
                        (None, Some(sc)) => (sc.index_k.view(), sc.n_groups),
                        (None, None) => {
                            return Err(Error::msg("an index-source without index keys"))
                        }
                    };
                    let iq = m_iqb.forward(&qr)?;
                    let mut iqs = copy_f32(cd, "index queries", &dview(&iq)?.view()?)?;
                    {
                        let (c, sn, _) = rope_views(rope, (table, a.pos), half);
                        rope_rows(cd, &mut iqs, ix.n_heads, ihd, rd, &c, &sn)?;
                    }
                    fp4_round(cd, &mut iqs, FP4_INDEX_BLOCK, true)?;
                    let wts = m_wproj.forward(&xin_t)?;
                    let scale = (ihd as f32).powf(-0.5) * (ix.n_heads as f32).powf(-0.5);
                    let clen = ((a.pos + 1) / a.ratio.max(1)).min(gtot);
                    let (ints, floats) = if gtot == 0 {
                        (
                            upload_i32(cd, "top-k", &vec![-1i32; ix.topk])?,
                            upload_f32(cd, "top-k", &vec![-1f32; ix.topk])?,
                        )
                    } else {
                        let mut scores = index_scores_dev(
                            cd,
                            &iqs,
                            &keys,
                            &dview(&wts)?.view()?,
                            (ix.n_heads, ihd, gtot),
                            a.ratio.max(1),
                            scale,
                            a.pos,
                        )?;
                        // The two-level candidate pre-filter: a source publishes the blocks its
                        // scores keep, a reader scores only inside them.
                        if ix.is_candidate_source {
                            let mask = candidate_mask(
                                cd,
                                &scores,
                                clen,
                                ix.candidate_block_size,
                                ix.candidate_topk_blocks,
                            )?;
                            *candidates = Some((self.ordinal, Arc::new(mask)));
                        } else if ix.uses_candidates {
                            match candidates {
                                Some((ord, mask)) if *ord == self.ordinal && mask.len() >= gtot => {
                                    apply_candidate_mask(cd, &mut scores, mask)?
                                }
                                _ => {
                                    return Err(Error::msg(
                                        "the candidate mask is not on this card",
                                    ))
                                }
                            }
                        }
                        topk_select(cd, &scores, clen, ix.topk, w)?
                    };
                    pending.topk = Some(floats);
                    let ints = Arc::new(ints);
                    published = Some(ints.clone());
                    Some(ints)
                }
                _ if a.ratio == 0 => None,
                _ => match topk_in {
                    Some((ord, lay, t)) if *ord == self.ordinal && Some(*lay) == a.index_source => {
                        Some(t.clone())
                    }
                    _ => return Err(Error::msg("the index-source's top-k is not on this card")),
                },
            };

            // One attention over the window ring and the compressed slots.
            let topk = w + itopk;
            let stream = cd.stream();
            let mut idxs = alloc_i32(cd, "attention slots", topk)?;
            let win: Vec<i32> = (0..w as i32).collect();
            if w > 0 {
                stream
                    .memcpy_htod(&win, &mut idxs.slice_mut(0..w))
                    .map_err(|e| Error::msg(format!("attention slots: {e}")))?;
            }
            if let Some(sl) = &slots {
                if itopk > 0 {
                    stream
                        .memcpy_dtod(&sl.slice(0..itopk), &mut idxs.slice_mut(w..w + itopk))
                        .map_err(|e| Error::msg(format!("attention slots: {e}")))?;
                }
            }
            let comp_rows: Option<&GrowDev> = if a.ratio == 0 {
                None
            } else if comp.is_some() {
                Some(&lc.comp)
            } else {
                Some(
                    &src.ok_or_else(|| Error::msg("a reader without a kv-source"))?
                        .comp,
                )
            };
            let keys = gather_ring(cd, &lc.ring, comp_rows, &idxs, topk)?;
            let mut o = sparse_attention_dev(
                cd,
                &qs,
                &keys,
                &dview(&sink)?.view()?,
                &idxs,
                nh,
                hd,
                topk,
                a.scale,
            )?;
            {
                let (c, _, neg) = rope_views(rope, (table, a.pos), half);
                rope_rows(cd, &mut o, nh, hd, rd, &c, &neg)?;
            }
            // The grouped output projection, then the stream updated.
            let p = nh * hd / a.o_groups;
            let og = tensor_of(&dev, o, (a.o_groups, p))?;
            let mut parts = Vec::with_capacity(a.o_groups);
            for (g, m) in wo.iter().enumerate() {
                parts.push(m.forward(&og.narrow(0, g, 1)?)?);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            let out = wob.forward(&Tensor::cat(&refs, 1)?)?;
            s.post(cd, &dview(&out)?.view()?)?;
            s.commit_pre();
            lc.pending = Some(pending);
            Ok(published)
        });
        Some(match ran {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => {
                *cache = DevCache::default();
                Err(e)
            }
            None => {
                *cache = DevCache::default();
                Err(Error::msg("the card declined the layer's attention"))
            }
        })
    }

    /// The FFN sublayer up to its crossing on this card, and the shared expert's output started
    /// behind it. `None` when a weight cannot be kept here.
    #[allow(clippy::too_many_arguments)]
    fn decode_route(
        &self,
        s: &mut DecodeStreamDev,
        hc: &HcDev,
        r: &DecodeRouter,
        ahead: Option<(&HcDev, &DecodeRouter)>,
        cache: &mut DevCache,
    ) -> Option<Result<(Crossing, Option<Tensor>)>> {
        let hcw = self.keep_hc(hc)?;
        let norm = self.keep_tensor(r.norm)?;
        let gate = self.kept_dense(r.gate)?;
        let bias = self.keep_tensor(r.bias)?;
        let shared = match r.shared {
            Some((w1, w3, w2, limit)) => Some((
                self.keep_proj(w1)?,
                self.keep_proj(w3)?,
                self.keep_proj(w2)?,
                limit,
            )),
            None => None,
        };
        let ahead_w = match ahead {
            Some((h, a)) => Some((
                self.keep_hc(h)?,
                self.keep_tensor(a.norm)?,
                self.kept_dense(a.gate)?,
                self.keep_tensor(a.bias)?,
                a,
            )),
            None => None,
        };
        let (dim, k) = (s.d(), r.n_activated);
        let inter = r
            .shared
            .map_or(0, |(w1, _, _, _)| w1.dims().first().copied().unwrap_or(0));
        let need = (8 * dim + 2 * r.n_routed + 4 * k + 2 * inter) * F32;
        let dev = Device::Cuda(self.dev.clone());
        let cd = &self.dev;
        let ran = self.attempt("a decode layer's router", need, || {
            self.mixes(s, &hcw, hc)?;
            let xin = s.collapse(cd, &dview(&norm)?.view()?, hc.norm_eps)?;
            let xin_view = xin.slice(0..dim);
            let xin_t = tensor_of(&dev, copy_f32(cd, "ffn input", &xin_view)?, (1, dim))?;
            let z = gate.forward(&xin_t)?;
            let sel = router_select(
                cd,
                &dview(&z)?.view()?,
                &dview(&bias)?.view()?,
                r.n_routed,
                k,
                r.gate_temp,
                r.norm_topk,
                r.route_scale,
            )?;
            // The read-ahead's guess for a layer below, from this stream collapsed with that
            // layer's own mixes.
            let predicted = match &ahead_w {
                Some((h, n, g, b, a)) => {
                    let (f, sc, bs) = (dview(&h[0])?, dview(&h[1])?, dview(&h[2])?);
                    let pre = s.mixes_pre_scratch(
                        cd,
                        &f.view()?,
                        &sc.view()?,
                        &bs.view()?,
                        hc.iters,
                        hc.norm_eps,
                        hc.hc_eps,
                    )?;
                    let xa = s.collapse_with(cd, &pre, &dview(n)?.view()?, hc.norm_eps)?;
                    let xa_t = tensor_of(&dev, xa, (1, dim))?;
                    let za = g.forward(&xa_t)?;
                    Some(router_select(
                        cd,
                        &dview(&za)?.view()?,
                        &dview(b)?.view()?,
                        a.n_routed,
                        a.n_activated,
                        a.gate_temp,
                        a.norm_topk,
                        a.route_scale,
                    )?)
                }
                None => None,
            };
            let (post, comb) = s.post_comb();
            let mut parts: Vec<CudaView<'_, f32>> = vec![xin_view, sel.slice(0..2 * k), post, comb];
            let lc = cache
                .0
                .as_mut()
                .and_then(|b| b.downcast_mut::<LayerCacheDev>());
            // The rows the attention appended go in the same download; their lengths and
            // which of them are present are noted so the download is read back in order.
            let mut rows_len = Vec::new();
            let mut present = [false; 4];
            if let Some(p) = lc.as_ref().and_then(|lc| lc.pending.as_ref()) {
                present = [
                    p.group_row.is_some(),
                    p.comp_row.is_some(),
                    p.index_row.is_some(),
                    p.topk.is_some(),
                ];
                for sl in [
                    Some(&p.window_row),
                    p.group_row.as_ref(),
                    p.comp_row.as_ref(),
                    p.index_row.as_ref(),
                    p.topk.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    parts.push(sl.slice(0..sl.len()));
                    rows_len.push(sl.len());
                }
            }
            let pred_k = ahead.map_or(0, |(_, a)| a.n_activated);
            if let Some(pr) = &predicted {
                parts.push(pr.slice(0..pred_k.min(pr.len())));
            }
            let flat = pack_download(cd, &parts)?;
            drop(parts);
            // The shared expert runs while the host runs its experts.
            let shared_out = match &shared {
                Some((m1, m3, m2, limit)) => {
                    let g = m1.forward(&xin_t)?;
                    let u = m3.forward(&xin_t)?;
                    let h = if *limit > 0.0 {
                        let hi = Tensor::full(*limit, 1, &dev)?;
                        let lo = Tensor::full(-*limit, 1, &dev)?;
                        let g = g.broadcast_minimum(&hi)?;
                        let u = u.broadcast_maximum(&lo)?.broadcast_minimum(&hi)?;
                        g.silu()?.mul(&u)?
                    } else {
                        g.silu()?.mul(&u)?
                    };
                    Some(m2.forward(&h)?)
                }
                None => None,
            };
            // The crossing, read in the order it was packed.
            let mut at = 0;
            let mut next = |len: usize| {
                let v = flat[at..at + len].to_vec();
                at += len;
                v
            };
            let xin_h = next(dim);
            let sel_h = next(2 * k);
            let routing: Vec<(usize, f32)> =
                (0..k).map(|i| (sel_h[i] as usize, sel_h[k + i])).collect();
            let post_h = next(s.hc());
            let comb_h = next(s.hc() * s.hc());
            let append = match lc {
                Some(lc) if !rows_len.is_empty() => {
                    lc.pending = None;
                    let mut i = 0;
                    let mut row = |there: bool| {
                        if there {
                            let v = next(rows_len[i]);
                            i += 1;
                            Some(v)
                        } else {
                            None
                        }
                    };
                    let window_row = row(true).expect("the window row is always present");
                    let group_row = row(present[0]);
                    let comp_row = row(present[1]);
                    let index_row = row(present[2]);
                    let topk = row(present[3]).map(|v| v.iter().map(|&f| f as i32).collect());
                    lc.expect = Some(Expect {
                        window: lc.ring.len(),
                        groups: lc.n_groups,
                        group_rows: lc.group.len(),
                        last_window: window_row.clone(),
                        last_group: if comp_row.is_some() {
                            None
                        } else {
                            group_row.clone()
                        },
                    });
                    Some(CacheAppend {
                        window_row,
                        group_row,
                        comp_row,
                        index_row,
                        topk,
                    })
                }
                _ => None,
            };
            let predicted = match &predicted {
                Some(_) => next(pred_k).iter().map(|&f| f as usize).collect(),
                None => Vec::new(),
            };
            Ok((
                Crossing {
                    xin: xin_h,
                    routing,
                    post: post_h,
                    comb: comb_h,
                    append,
                    predicted,
                },
                shared_out,
            ))
        });
        Some(match ran {
            Some(r) => r,
            None => Err(Error::msg("the card declined the layer's router")),
        })
    }

    /// The FFN output folded into the stream: the shared expert's output plus the routed sum.
    fn decode_finish(
        &self,
        s: &mut DecodeStreamDev,
        shared: Option<&Tensor>,
        routed: &[f32],
    ) -> Result<()> {
        let dim = s.d();
        if routed.len() != dim {
            return Err(Error::msg(format!(
                "a routed sum of {} values for {dim}",
                routed.len()
            )));
        }
        let dev = Device::Cuda(self.dev.clone());
        let cd = &self.dev;
        let need = 3 * dim * F32;
        let ran = self.attempt("a decode layer's update", need, || {
            let rd = tensor_of(&dev, upload_f32(cd, "routed sum", routed)?, (1, dim))?;
            let xo = match shared {
                Some(sh) => sh.add(&rd)?,
                None => rd,
            };
            s.post(cd, &dview(&xo)?.view()?)?;
            s.commit_pre();
            Ok(())
        });
        match ran {
            Some(r) => r,
            None => Err(Error::msg("the card declined the layer's update")),
        }
    }
}

/// A decode token's stream on the cards (see `DecodeStream`).
pub struct CardsDecode {
    cards: Vec<Arc<Card>>,
    /// The card holding the stream now.
    at: usize,
    s: DecodeStreamDev,
    /// This token's rope rows on the current card, by table and position.
    rope: HashMap<(TensorId, usize), RopeRows>,
    /// The top-k the last index-source published this token: its card, its layer, the slots.
    topk: Option<(usize, usize, Arc<CudaSlice<i32>>)>,
    /// The candidate mask the last candidate source published this token: its card, the flags.
    candidates: Option<(usize, Arc<CudaSlice<f32>>)>,
    /// The shared expert's output of the FFN in flight, awaiting the routed sum.
    shared_out: Option<Option<Tensor>>,
}

impl CardsDecode {
    /// The stream on the first card with room for it.
    pub fn open(cards: &[Arc<Card>], x: &[f32], hc: usize, d: usize) -> Option<Self> {
        for (at, card) in cards.iter().enumerate() {
            let opened = card.attempt("a decode token's stream", x.len() * 4 * F32, || {
                DecodeStreamDev::open(&card.dev, x, hc, d)
            });
            if let Some(Ok(s)) = opened {
                return Some(Self {
                    cards: cards.to_vec(),
                    at,
                    s,
                    rope: HashMap::new(),
                    topk: None,
                    candidates: None,
                    shared_out: None,
                });
            }
        }
        None
    }

    /// Move the stream to card `to`: its values and collapse weights cross through the host.
    fn move_to(&mut self, to: usize) -> Result<()> {
        if to == self.at {
            return Ok(());
        }
        let from = &self.cards[self.at];
        let (x, pre) = (self.s.rows(&from.dev)?, self.s.pre_host(&from.dev)?);
        let card = &self.cards[to];
        let mut s = card
            .attempt("a decode token's stream", x.len() * 4 * F32, || {
                DecodeStreamDev::open(&card.dev, &x, self.s.hc(), self.s.d())
            })
            .ok_or_else(|| Error::msg("the next card has no room for the stream"))??;
        s.replace(&card.dev, &x, &pre)?;
        self.s = s;
        self.at = to;
        self.rope.clear();
        Ok(())
    }
}

impl DecodeStream for CardsDecode {
    fn attention(
        &mut self,
        layer: usize,
        hc: &HcDev,
        attn: &DecodeAttn,
        cache: &mut DevCache,
        host: &dyn HostCache,
        source: Option<(&DevCache, &dyn HostCache)>,
    ) -> Result<()> {
        // The card holding the stream first, then the others, the stream moving to the one
        // that keeps the layer's weights.
        let order: Vec<usize> = std::iter::once(self.at)
            .chain((0..self.cards.len()).filter(|&c| c != self.at))
            .collect();
        for c in order {
            let card = self.cards[c].clone();
            if c != self.at {
                // A card is tried only when it keeps the layer's first weight already or has
                // room; the move itself is cheap beside a layer.
                if card.keep_proj(attn.wq_a).is_none() {
                    continue;
                }
                self.move_to(c)?;
            }
            let topk_in = self.topk.as_ref();
            let ran = card.decode_attention(
                &mut self.s,
                &mut self.rope,
                topk_in,
                &mut self.candidates,
                hc,
                attn,
                cache,
                host,
                source,
            );
            match ran {
                Some(Ok(published)) => {
                    if let Some(t) = published {
                        self.topk = Some((card.ordinal, layer, t));
                    }
                    return Ok(());
                }
                Some(Err(e)) => return Err(e),
                None => continue,
            }
        }
        Err(Error::msg("no card keeps the layer's attention weights"))
    }

    fn moe_route(
        &mut self,
        hc: &HcDev,
        r: &DecodeRouter,
        ahead: Option<(&HcDev, &DecodeRouter)>,
        cache: &mut DevCache,
    ) -> Result<Crossing> {
        let card = self.cards[self.at].clone();
        let (crossing, shared) = card
            .decode_route(&mut self.s, hc, r, ahead, cache)
            .ok_or_else(|| Error::msg("the card cannot keep the layer's router"))??;
        self.shared_out = Some(shared);
        Ok(crossing)
    }

    fn candidates(&mut self) -> Result<Option<Vec<bool>>> {
        let Some((ordinal, mask)) = &self.candidates else {
            return Ok(None);
        };
        let card = self
            .cards
            .iter()
            .find(|c| c.ordinal == *ordinal)
            .ok_or_else(|| Error::msg("the candidate mask's card is gone"))?;
        let flags = card
            .dev
            .stream()
            .clone_dtoh(mask.as_ref())
            .map_err(|e| Error(format!("candidate mask download: {e}")))?;
        Ok(Some(flags.into_iter().map(|f| f != 0.0).collect()))
    }

    fn moe_finish(&mut self, routed: &[f32]) -> Result<()> {
        let card = self.cards[self.at].clone();
        let shared = self
            .shared_out
            .take()
            .ok_or_else(|| Error::msg("an FFN finished before it was routed"))?;
        card.decode_finish(&mut self.s, shared.as_ref(), routed)
    }

    fn add(&mut self, delta: &[f32], post: &[f32]) -> Result<()> {
        let card = self.cards[self.at].clone();
        self.s.add(&card.dev, delta, post)
    }

    fn take_append(&mut self, cache: &mut DevCache) -> Result<Option<CacheAppend>> {
        let Some(lc) = cache
            .0
            .as_mut()
            .and_then(|b| b.downcast_mut::<LayerCacheDev>())
        else {
            return Ok(None);
        };
        let card = self
            .cards
            .iter()
            .find(|c| c.ordinal == lc.ordinal)
            .cloned()
            .ok_or_else(|| Error::msg("the layer's card is gone"))?;
        lc.take(&card.dev)
    }

    fn pre(&mut self) -> Result<Vec<f32>> {
        self.s.pre_host(&self.cards[self.at].dev)
    }

    fn rows(&mut self) -> Result<Vec<f32>> {
        self.s.rows(&self.cards[self.at].dev)
    }

    fn replace(&mut self, x: &[f32], pre: &[f32]) -> Result<()> {
        let card = self.cards[self.at].clone();
        self.s.replace(&card.dev, x, pre)
    }
}
