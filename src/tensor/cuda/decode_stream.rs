//! A decode token's residual stream and one layer's attention state on a card.
//!
//! The stream `x` [hc, d] and the hyper-connection coefficients it yields stay on the device from
//! one layer to the next: a sublayer's input is collapsed from it there and its output folded
//! back there. Beside the stream, a layer keeps its window ring, its compressed rows and its
//! index keys on the card, one row appended per token. Every kernel here does what the host step
//! it stands in for does, in the host's order of float operations.

use super::*;
use crate::tensor::gpu_runtime::driver::{CudaView, PushKernelArg};

const THREADS: usize = 256;

/// One thread per element of `total`.
fn grid_for(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (total.div_ceil(THREADS).max(1) as u32, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One block of `threads` threads with `shared` bytes of shared memory.
fn one_block(threads: usize, shared: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared as u32,
    }
}

fn launch_err(name: &str, e: crate::tensor::gpu_runtime::driver::DriverError) -> Error {
    Error(format!("{name} launch: {e}"))
}

/// `n` f32 values on the device, uninitialised.
pub fn alloc_f32(dev: &CudaDevice, what: &str, n: usize) -> Result<CudaSlice<f32>> {
    let stream = dev.stream();
    with_oom_retry(dev, what, || unsafe { stream.alloc::<f32>(n.max(1)) })
}

/// `n` i32 values on the device, uninitialised.
pub fn alloc_i32(dev: &CudaDevice, what: &str, n: usize) -> Result<CudaSlice<i32>> {
    let stream = dev.stream();
    with_oom_retry(dev, what, || unsafe { stream.alloc::<i32>(n.max(1)) })
}

/// `v` uploaded to the device.
pub fn upload_f32(dev: &CudaDevice, what: &str, v: &[f32]) -> Result<CudaSlice<f32>> {
    if v.is_empty() {
        return alloc_f32(dev, what, 1);
    }
    dev.stream().clone_htod(v).map_err(|e| alloc_err(what, e))
}

/// `v` uploaded to the device.
pub fn upload_i32(dev: &CudaDevice, what: &str, v: &[i32]) -> Result<CudaSlice<i32>> {
    let stream = dev.stream();
    if v.is_empty() {
        return with_oom_retry(dev, what, || unsafe { stream.alloc::<i32>(1) });
    }
    stream.clone_htod(v).map_err(|e| alloc_err(what, e))
}

/// A device copy of `src`, owned.
pub fn copy_f32(dev: &CudaDevice, what: &str, src: &CudaView<'_, f32>) -> Result<CudaSlice<f32>> {
    let mut out = alloc_f32(dev, what, src.len())?;
    if !src.is_empty() {
        dev.stream()
            .memcpy_dtod(src, &mut out.slice_mut(0..src.len()))
            .map_err(|e| Error(format!("{what} copy: {e}")))?;
    }
    Ok(out)
}

/// `parts` laid end to end in one device buffer and read back in one download.
pub fn pack_download(dev: &CudaDevice, parts: &[CudaView<'_, f32>]) -> Result<Vec<f32>> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = alloc_f32(dev, "decode crossing", total)?;
    let stream = dev.stream();
    let mut at = 0;
    for p in parts {
        if !p.is_empty() {
            stream
                .memcpy_dtod(p, &mut out.slice_mut(at..at + p.len()))
                .map_err(|e| Error(format!("decode crossing pack: {e}")))?;
        }
        at += p.len();
    }
    let mut host = stream
        .clone_dtoh(&out)
        .map_err(|e| Error(format!("decode crossing download: {e}")))?;
    host.truncate(total);
    Ok(host)
}

/// The residual stream of one token, `[hc, d]`, with the hyper-connection coefficients of the
/// sublayer in hand.
pub struct DecodeStreamDev {
    x: CudaSlice<f32>,
    /// The collapse weights the next sublayer input is taken with, and the ones the latest mixes
    /// produced for the sublayer after it.
    pre: CudaSlice<f32>,
    pre_next: CudaSlice<f32>,
    post: CudaSlice<f32>,
    comb: CudaSlice<f32>,
    hc: usize,
    d: usize,
}

impl DecodeStreamDev {
    /// Upload `x` [hc, d]; the first sublayer collapses with the identity (copy 0 alone).
    pub fn open(dev: &CudaDevice, x: &[f32], hc: usize, d: usize) -> Result<Self> {
        if hc == 0 || hc > 8 || d == 0 || x.len() != hc * d {
            return Err(Error(format!(
                "decode stream: {} values for {hc} x {d}",
                x.len()
            )));
        }
        let mut identity = vec![0f32; hc];
        identity[0] = 1.0;
        Ok(Self {
            x: upload_f32(dev, "decode stream", x)?,
            pre: upload_f32(dev, "decode stream pre", &identity)?,
            pre_next: alloc_f32(dev, "decode stream pre", hc)?,
            post: alloc_f32(dev, "decode stream post", hc)?,
            comb: alloc_f32(dev, "decode stream comb", hc * hc)?,
            hc,
            d,
        })
    }

    pub fn hc(&self) -> usize {
        self.hc
    }

    pub fn d(&self) -> usize {
        self.d
    }

    /// The hyper-connection coefficients of the stream as it stands: `pre` for the sublayer after
    /// the next, `post` and `comb` for the next, kept on the device. `func` is [(2 + hc) hc, hc d],
    /// `scale` [3], `base` [(2 + hc) hc].
    #[allow(clippy::too_many_arguments)]
    pub fn mixes(
        &mut self,
        dev: &CudaDevice,
        func: &CudaView<'_, f32>,
        scale: &CudaView<'_, f32>,
        base: &CudaView<'_, f32>,
        iters: usize,
        norm_eps: f32,
        hc_eps: f32,
    ) -> Result<()> {
        let (pre, post, comb) = (&mut self.pre_next, &mut self.post, &mut self.comb);
        launch_mixes(
            dev, &self.x, self.hc, self.d, func, scale, base, iters, norm_eps, hc_eps, pre, post,
            comb,
        )
    }

    /// The `pre` coefficients the stream would yield with another sublayer's weights, in a buffer
    /// of their own: the stream's own coefficients are untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn mixes_pre_scratch(
        &self,
        dev: &CudaDevice,
        func: &CudaView<'_, f32>,
        scale: &CudaView<'_, f32>,
        base: &CudaView<'_, f32>,
        iters: usize,
        norm_eps: f32,
        hc_eps: f32,
    ) -> Result<CudaSlice<f32>> {
        let mut pre = alloc_f32(dev, "decode stream pre", self.hc)?;
        let mut post = alloc_f32(dev, "decode stream post", self.hc)?;
        let mut comb = alloc_f32(dev, "decode stream comb", self.hc * self.hc)?;
        launch_mixes(
            dev, &self.x, self.hc, self.d, func, scale, base, iters, norm_eps, hc_eps, &mut pre,
            &mut post, &mut comb,
        )?;
        Ok(pre)
    }

    /// The stream collapsed by `pre` [hc] and RMS-normalised by `w` [d]: a sublayer's input [d].
    pub fn collapse_with(
        &self,
        dev: &CudaDevice,
        pre: &CudaSlice<f32>,
        w: &CudaView<'_, f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>> {
        let (hc, d) = (self.hc, self.d);
        if w.len() != d {
            return Err(Error(format!(
                "decode stream collapse: {} norm values for {d}",
                w.len()
            )));
        }
        let shared = d * std::mem::size_of::<f32>();
        if shared > 48 * 1024 {
            return Err(Error(format!(
                "decode stream collapse: a row of {d} values does not fit a block's shared memory"
            )));
        }
        let stream = dev.stream();
        let out = alloc_f32(dev, "decode stream collapse", d)?;
        let func = dev.quantized_fn("hc_pre_norm_f32")?;
        let (d_i, hc_i) = (d as i64, hc as i32);
        let mut b = stream.launch_builder(&func);
        b.arg(&self.x);
        b.arg(pre);
        b.arg(w);
        b.arg(&out);
        b.arg(&d_i);
        b.arg(&hc_i);
        b.arg(&eps);
        unsafe { b.launch(one_block(THREADS, shared)) }
            .map_err(|e| launch_err("hc_pre_norm_f32", e))?;
        Ok(out)
    }

    /// The next sublayer's input: the stream collapsed by the weights in hand and normalised.
    pub fn collapse(
        &self,
        dev: &CudaDevice,
        w: &CudaView<'_, f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>> {
        self.collapse_with(dev, &self.pre, w, eps)
    }

    /// The weights the latest mixes produced become the ones the next collapse uses.
    pub fn commit_pre(&mut self) {
        std::mem::swap(&mut self.pre, &mut self.pre_next);
    }

    /// Fold a sublayer's output `xo` [d] into the stream with the `post` and `comb` in hand.
    pub fn post(&mut self, dev: &CudaDevice, xo: &CudaView<'_, f32>) -> Result<()> {
        let (hc, d) = (self.hc, self.d);
        if xo.len() != d {
            return Err(Error(format!(
                "decode stream post: {} outputs for {d}",
                xo.len()
            )));
        }
        let stream = dev.stream();
        let func = dev.quantized_fn("hc_post_f32")?;
        let (total_i, d_i, hc_i) = (d as i64, d as i64, hc as i32);
        let mut b = stream.launch_builder(&func);
        b.arg(xo);
        b.arg(&mut self.x);
        b.arg(&self.post);
        b.arg(&self.comb);
        b.arg(&total_i);
        b.arg(&d_i);
        b.arg(&hc_i);
        unsafe { b.launch(grid_for(d)) }
            .map(|_| ())
            .map_err(|e| launch_err("hc_post_f32", e))
    }

    /// Add `delta` [d] to every copy of the stream, copy c scaled by `post[c]`: what a sublayer's
    /// `post` would have folded in had `delta` been part of its output.
    pub fn add(&mut self, dev: &CudaDevice, delta: &[f32], post: &[f32]) -> Result<()> {
        let (hc, d) = (self.hc, self.d);
        if delta.len() != d || post.len() != hc {
            return Err(Error(format!(
                "decode stream add: {} values and {} weights for {hc} x {d}",
                delta.len(),
                post.len()
            )));
        }
        let dd = upload_f32(dev, "decode stream add", delta)?;
        let pd = upload_f32(dev, "decode stream add", post)?;
        let stream = dev.stream();
        let func = dev.quantized_fn("hc_add_f32")?;
        let (d_i, hc_i) = (d as i64, hc as i32);
        let mut b = stream.launch_builder(&func);
        b.arg(&mut self.x);
        b.arg(&dd);
        b.arg(&pd);
        b.arg(&d_i);
        b.arg(&hc_i);
        unsafe { b.launch(grid_for(hc * d)) }
            .map(|_| ())
            .map_err(|e| launch_err("hc_add_f32", e))
    }

    /// The post and comb coefficients in hand, `[hc]` then `[hc, hc]`, as device views.
    pub fn post_comb(&self) -> (CudaView<'_, f32>, CudaView<'_, f32>) {
        (
            self.post.slice(0..self.hc),
            self.comb.slice(0..self.hc * self.hc),
        )
    }

    /// The collapse weights in hand, on the host.
    pub fn pre_host(&self, dev: &CudaDevice) -> Result<Vec<f32>> {
        dev.stream()
            .clone_dtoh(&self.pre)
            .map_err(|e| Error(format!("decode stream pre download: {e}")))
    }

    /// The whole stream, `[hc, d]`, on the host.
    pub fn rows(&self, dev: &CudaDevice) -> Result<Vec<f32>> {
        dev.stream()
            .clone_dtoh(&self.x)
            .map_err(|e| Error(format!("decode stream download: {e}")))
    }

    /// Replace the stream and the collapse weights in hand.
    pub fn replace(&mut self, dev: &CudaDevice, x: &[f32], pre: &[f32]) -> Result<()> {
        if x.len() != self.hc * self.d || pre.len() != self.hc {
            return Err(Error(format!(
                "decode stream replace: {} values and {} weights",
                x.len(),
                pre.len()
            )));
        }
        self.x = upload_f32(dev, "decode stream", x)?;
        self.pre = upload_f32(dev, "decode stream pre", pre)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_mixes(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    hc: usize,
    d: usize,
    func: &CudaView<'_, f32>,
    scale: &CudaView<'_, f32>,
    base: &CudaView<'_, f32>,
    iters: usize,
    norm_eps: f32,
    hc_eps: f32,
    pre: &mut CudaSlice<f32>,
    post: &mut CudaSlice<f32>,
    comb: &mut CudaSlice<f32>,
) -> Result<()> {
    let flat = hc * d;
    let mix_hc = (2 + hc) * hc;
    if func.len() != mix_hc * flat || scale.len() != 3 || base.len() != mix_hc {
        return Err(Error(format!(
            "decode stream mixes: {} projection values, {} scales, {} biases",
            func.len(),
            scale.len(),
            base.len()
        )));
    }
    let stream = dev.stream();
    let dots = dev.quantized_fn("hc_mixed_dots_f32")?;
    let gates = dev.quantized_fn("hc_mixed_gates_f32")?;
    let mut mixed = alloc_f32(dev, "decode stream mixed", mix_hc + 1)?;
    let (flat_i, hc_i, mix_i, it_i) = (flat as i64, hc as i32, mix_hc as i32, iters as i32);
    let mut b = stream.launch_builder(&dots);
    b.arg(x);
    b.arg(func);
    b.arg(&mut mixed);
    b.arg(&flat_i);
    b.arg(&mix_i);
    let cfg = LaunchConfig {
        grid_dim: (mix_hc as u32 + 1, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }.map_err(|e| launch_err("hc_mixed_dots_f32", e))?;
    let mut b = stream.launch_builder(&gates);
    b.arg(&mixed);
    b.arg(scale);
    b.arg(base);
    b.arg(pre);
    b.arg(post);
    b.arg(comb);
    b.arg(&flat_i);
    b.arg(&hc_i);
    b.arg(&mix_i);
    b.arg(&it_i);
    b.arg(&norm_eps);
    b.arg(&hc_eps);
    unsafe { b.launch(one_block(32, 0)) }
        .map(|_| ())
        .map_err(|e| launch_err("hc_mixed_gates_f32", e))
}

/// A sliding window of `d`-wide rows on the device: the last `cap` rows appended, oldest first
/// from `start`.
pub struct RingDev {
    rows: CudaSlice<f32>,
    cap: usize,
    d: usize,
    start: usize,
    len: usize,
}

impl RingDev {
    /// A ring of `cap` rows holding `rows` (flat, oldest first, at most `cap` of them).
    pub fn open(dev: &CudaDevice, rows: &[f32], cap: usize, d: usize) -> Result<Self> {
        let cap = cap.max(1);
        if d == 0 || !rows.len().is_multiple_of(d) || rows.len() / d > cap {
            return Err(Error(format!(
                "window ring: {} values for {cap} rows of {d}",
                rows.len()
            )));
        }
        let mut buf = alloc_f32(dev, "window ring", cap * d)?;
        if !rows.is_empty() {
            dev.stream()
                .memcpy_htod(rows, &mut buf.slice_mut(0..rows.len()))
                .map_err(|e| Error(format!("window ring upload: {e}")))?;
        }
        Ok(Self {
            rows: buf,
            cap,
            d,
            start: 0,
            len: rows.len() / d,
        })
    }

    /// Append `row` [d], the oldest row leaving once the ring is full.
    pub fn append(&mut self, dev: &CudaDevice, row: &CudaView<'_, f32>) -> Result<()> {
        if row.len() != self.d {
            return Err(Error(format!(
                "window ring: a row of {} for {}",
                row.len(),
                self.d
            )));
        }
        let slot = if self.len < self.cap {
            let s = (self.start + self.len) % self.cap;
            self.len += 1;
            s
        } else {
            let s = self.start;
            self.start = (self.start + 1) % self.cap;
            s
        };
        dev.stream()
            .memcpy_dtod(
                row,
                &mut self.rows.slice_mut(slot * self.d..(slot + 1) * self.d),
            )
            .map_err(|e| Error(format!("window ring append: {e}")))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn rows(&self) -> &CudaSlice<f32> {
        &self.rows
    }

    /// The newest row, `[d]`.
    pub fn last(&self) -> Option<CudaView<'_, f32>> {
        if self.len == 0 {
            return None;
        }
        let slot = (self.start + self.len - 1) % self.cap;
        Some(self.rows.slice(slot * self.d..(slot + 1) * self.d))
    }
}

/// Rows of `d` values appended one at a time on the device, the buffer doubling as they come.
pub struct GrowDev {
    rows: CudaSlice<f32>,
    cap: usize,
    len: usize,
    d: usize,
}

impl GrowDev {
    pub fn open(dev: &CudaDevice, flat: &[f32], d: usize) -> Result<Self> {
        if d == 0 || !flat.len().is_multiple_of(d) {
            return Err(Error(format!(
                "device rows: {} values for rows of {d}",
                flat.len()
            )));
        }
        let len = flat.len() / d;
        let cap = len.max(16);
        let mut rows = alloc_f32(dev, "device rows", cap * d)?;
        if !flat.is_empty() {
            dev.stream()
                .memcpy_htod(flat, &mut rows.slice_mut(0..flat.len()))
                .map_err(|e| Error(format!("device rows upload: {e}")))?;
        }
        Ok(Self { rows, cap, len, d })
    }

    pub fn append(&mut self, dev: &CudaDevice, row: &CudaView<'_, f32>) -> Result<()> {
        if row.len() != self.d {
            return Err(Error(format!(
                "device rows: a row of {} for {}",
                row.len(),
                self.d
            )));
        }
        let stream = dev.stream();
        if self.len == self.cap {
            let cap = self.cap * 2;
            let mut bigger = alloc_f32(dev, "device rows", cap * self.d)?;
            stream
                .memcpy_dtod(
                    &self.rows.slice(0..self.len * self.d),
                    &mut bigger.slice_mut(0..self.len * self.d),
                )
                .map_err(|e| Error(format!("device rows grow: {e}")))?;
            self.rows = bigger;
            self.cap = cap;
        }
        stream
            .memcpy_dtod(
                row,
                &mut self
                    .rows
                    .slice_mut(self.len * self.d..(self.len + 1) * self.d),
            )
            .map_err(|e| Error(format!("device rows append: {e}")))?;
        self.len += 1;
        Ok(())
    }

    /// Forget the rows: the buffer stays for the next ones.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn view(&self) -> CudaView<'_, f32> {
        self.rows
            .slice(0..(self.len * self.d).max(1).min(self.cap * self.d))
    }

    /// Row `i`, `[d]`.
    pub fn row(&self, i: usize) -> CudaView<'_, f32> {
        self.rows.slice(i * self.d..(i + 1) * self.d)
    }
}

/// Block-wise fp8 (e4m3) round trip of `x` in place, `block` values per scale.
pub fn fp8_round(dev: &CudaDevice, x: &mut CudaSlice<f32>, block: usize) -> Result<()> {
    let total = x.len();
    if block == 0 || !total.is_multiple_of(block) {
        return Err(Error(format!(
            "fp8 round: {total} values in blocks of {block}"
        )));
    }
    let func = dev.quantized_fn("fp8_round_f32")?;
    let (total_i, block_i) = (total as i64, block as i64);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(x);
    b.arg(&total_i);
    b.arg(&block_i);
    unsafe { b.launch(grid_for(total / block)) }
        .map(|_| ())
        .map_err(|e| launch_err("fp8_round_f32", e))
}

/// Block-wise fp4 (e2m1) round trip of `x` in place, `block` values per scale, the scale rounded
/// up to a power of two when `pow2` is set.
pub fn fp4_round(dev: &CudaDevice, x: &mut CudaSlice<f32>, block: usize, pow2: bool) -> Result<()> {
    let total = x.len();
    if block == 0 || !total.is_multiple_of(block) {
        return Err(Error(format!(
            "fp4 round: {total} values in blocks of {block}"
        )));
    }
    let func = dev.quantized_fn("fp4_round_f32")?;
    let (total_i, block_i, pow2_i) = (total as i64, block as i64, pow2 as i32);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(x);
    b.arg(&total_i);
    b.arg(&block_i);
    b.arg(&pow2_i);
    unsafe { b.launch(grid_for(total / block)) }
        .map(|_| ())
        .map_err(|e| launch_err("fp4_round_f32", e))
}

/// Rotate the last `rd` values of each of the `h` heads of `x` [h, hd] in place by the angles
/// whose cosines and sines are `cos` and `sin` [rd / 2].
pub fn rope_rows(
    dev: &CudaDevice,
    x: &mut CudaSlice<f32>,
    h: usize,
    hd: usize,
    rd: usize,
    cos: &CudaView<'_, f32>,
    sin: &CudaView<'_, f32>,
) -> Result<()> {
    let half = rd / 2;
    if x.len() < h * hd || rd > hd || cos.len() < half || sin.len() < half {
        return Err(Error(format!(
            "rope: {} values for {h} heads of {hd}, {rd} rotated, tables of {} and {}",
            x.len(),
            cos.len(),
            sin.len()
        )));
    }
    if half == 0 {
        return Ok(());
    }
    let func = dev.quantized_fn("rope_tail_f32")?;
    let (total_i, h_i, hd_i, rd_i) = ((half * h) as i64, h as i64, hd as i64, rd as i64);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(x);
    b.arg(cos);
    b.arg(sin);
    b.arg(&total_i);
    b.arg(&h_i);
    b.arg(&hd_i);
    b.arg(&rd_i);
    unsafe { b.launch(grid_for(half * h)) }
        .map(|_| ())
        .map_err(|e| launch_err("rope_tail_f32", e))
}

/// `ratio` tokens' latents `kv` [ratio, hd] pooled by a per-channel softmax of `score` [ratio, hd]
/// over the tokens: `[hd]`.
pub fn compress_pool(
    dev: &CudaDevice,
    kv: &CudaView<'_, f32>,
    score: &CudaView<'_, f32>,
    ratio: usize,
    hd: usize,
) -> Result<CudaSlice<f32>> {
    if kv.len() != ratio * hd || score.len() != ratio * hd || ratio == 0 {
        return Err(Error(format!(
            "compress pool: {} latents and {} scores for {ratio} x {hd}",
            kv.len(),
            score.len()
        )));
    }
    let out = alloc_f32(dev, "compress pool", hd)?;
    let func = dev.quantized_fn("compress_pool_f32")?;
    let (ratio_i, hd_i) = (ratio as i64, hd as i64);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(kv);
    b.arg(score);
    b.arg(&out);
    b.arg(&ratio_i);
    b.arg(&hd_i);
    unsafe { b.launch(grid_for(hd)) }.map_err(|e| launch_err("compress_pool_f32", e))?;
    Ok(out)
}

/// One query's indexer scores over `g` compressed positions: `q` [nh, ihd], `k` [g, ihd],
/// `w` [nh]; the sum over heads of relu(q . k) * w * scale where the query at `pos` reaches,
/// `(pos + 1) / ratio` positions, -inf past. `[g]`.
#[allow(clippy::too_many_arguments)]
pub fn index_scores_dev(
    dev: &CudaDevice,
    q: &CudaSlice<f32>,
    k: &CudaView<'_, f32>,
    w: &CudaView<'_, f32>,
    (nh, ihd, g): (usize, usize, usize),
    ratio: usize,
    scale: f32,
    pos: usize,
) -> Result<CudaSlice<f32>> {
    if q.len() < nh * ihd || k.len() < g * ihd || w.len() < nh || ratio == 0 {
        return Err(Error(format!(
            "index scores: {} queries, {} keys, {} weights for nh {nh} ihd {ihd} g {g}",
            q.len(),
            k.len(),
            w.len()
        )));
    }
    let out = alloc_f32(dev, "index scores", g)?;
    if g == 0 {
        return Ok(out);
    }
    let func = dev.quantized_fn("index_scores_f32")?;
    let dims = [
        1i64,
        nh as i64,
        ihd as i64,
        g as i64,
        ratio as i64,
        pos as i64,
    ];
    let mut b = dev.stream().launch_builder(&func);
    b.arg(q);
    b.arg(k);
    b.arg(w);
    b.arg(&out);
    for v in &dims {
        b.arg(v);
    }
    b.arg(&scale);
    unsafe { b.launch(grid_for(g)) }.map_err(|e| launch_err("index_scores_f32", e))?;
    Ok(out)
}

/// The `k` largest of `scores[0 .. reach)` as slots `index + offset`, -1 past the ones found: as
/// ints for the attention and as floats for the host. The set a descending stable sort keeps.
pub fn topk_select(
    dev: &CudaDevice,
    scores: &CudaSlice<f32>,
    reach: usize,
    k: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, CudaSlice<f32>)> {
    if reach > scores.len() {
        return Err(Error(format!(
            "top-k: {reach} reachable of {} scores",
            scores.len()
        )));
    }
    let stream = dev.stream();
    let out = with_oom_retry(dev, "top-k slots", || unsafe {
        stream.alloc::<i32>(k.max(1))
    })?;
    let outf = alloc_f32(dev, "top-k slots", k)?;
    let func = dev.quantized_fn("topk_select_f32")?;
    let (reach_i, k_i, off_i) = (reach as i64, k as i64, offset as i64);
    let mut b = stream.launch_builder(&func);
    b.arg(scores);
    b.arg(&reach_i);
    b.arg(&k_i);
    b.arg(&off_i);
    b.arg(&out);
    b.arg(&outf);
    unsafe { b.launch(one_block(1024, 0)) }.map_err(|e| launch_err("topk_select_f32", e))?;
    Ok((out, outf))
}

/// A token's routed experts from its gate logits `z` [n] and the selection bias: `[2k]`, the
/// chosen experts as floats then their weights (see the kernel).
#[allow(clippy::too_many_arguments)]
pub fn router_select(
    dev: &CudaDevice,
    z: &CudaView<'_, f32>,
    bias: &CudaView<'_, f32>,
    n: usize,
    k: usize,
    temp: f32,
    norm_topk: bool,
    route_scale: f32,
) -> Result<CudaSlice<f32>> {
    if z.len() < n || bias.len() < n || k == 0 || k > n {
        return Err(Error(format!(
            "router: {} logits and {} biases for {n} experts, {k} chosen",
            z.len(),
            bias.len()
        )));
    }
    let shared = n * std::mem::size_of::<f32>();
    if shared > 48 * 1024 {
        return Err(Error(format!(
            "router: {n} experts do not fit a block's shared memory"
        )));
    }
    let out = alloc_f32(dev, "router", 2 * k)?;
    let func = dev.quantized_fn("router_select_f32")?;
    let (n_i, k_i, norm_i) = (n as i64, k as i64, norm_topk as i32);
    let threads = n.div_ceil(32).max(1).min(32) * 32;
    let mut b = dev.stream().launch_builder(&func);
    b.arg(z);
    b.arg(bias);
    b.arg(&n_i);
    b.arg(&k_i);
    b.arg(&temp);
    b.arg(&norm_i);
    b.arg(&route_scale);
    b.arg(&out);
    unsafe { b.launch(one_block(threads, shared)) }
        .map_err(|e| launch_err("router_select_f32", e))?;
    Ok(out)
}

/// The keys of one query gathered from the window `ring` and the compressed rows `comp`: slot
/// `s` below the ring's length is ring row s (oldest first), from there on compressed row
/// `s - len`, a negative slot a row of zeros. `[topk, d]`.
pub fn gather_ring(
    dev: &CudaDevice,
    ring: &RingDev,
    comp: Option<&GrowDev>,
    idxs: &CudaSlice<i32>,
    topk: usize,
) -> Result<CudaSlice<f32>> {
    let d = ring.d;
    if idxs.len() < topk {
        return Err(Error(format!("gather: {} slots for {topk}", idxs.len())));
    }
    let out = alloc_f32(dev, "gathered keys", topk * d)?;
    let comp_view = match comp {
        Some(c) => c.view(),
        None => ring.rows.slice(0..d),
    };
    let func = dev.quantized_fn("gather_ring_f32")?;
    let (start_i, cap_i, w_i, topk_i, d_i) = (
        ring.start as i64,
        ring.cap as i64,
        ring.len as i64,
        topk as i64,
        d as i64,
    );
    let mut b = dev.stream().launch_builder(&func);
    b.arg(&ring.rows);
    b.arg(&start_i);
    b.arg(&cap_i);
    b.arg(&w_i);
    b.arg(&comp_view);
    b.arg(idxs);
    b.arg(&out);
    b.arg(&topk_i);
    b.arg(&d_i);
    unsafe { b.launch(grid_for(topk * d)) }.map_err(|e| launch_err("gather_ring_f32", e))?;
    Ok(out)
}

/// One query's sparse attention with a per-head sink over `keys` [topk, d] already gathered:
/// `q` [h, d], `sink` [h], `idxs` [topk] (negative: empty). `[h, d]`.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attention_dev(
    dev: &CudaDevice,
    q: &CudaSlice<f32>,
    keys: &CudaSlice<f32>,
    sink: &CudaView<'_, f32>,
    idxs: &CudaSlice<i32>,
    h: usize,
    d: usize,
    topk: usize,
    scale: f32,
) -> Result<CudaSlice<f32>> {
    if q.len() < h * d || keys.len() < topk * d || sink.len() < h || idxs.len() < topk {
        return Err(Error(format!(
            "sparse attention: {} queries, {} keys, {} sinks, {} slots for h {h} d {d} topk {topk}",
            q.len(),
            keys.len(),
            sink.len(),
            idxs.len()
        )));
    }
    let logits = matmul_nt_f32(dev, q, keys, 1, h, d, topk)?;
    let func = dev.quantized_fn("sink_softmax_f32")?;
    let (start_i, count_i, h_i, topk_i) = (0i64, 1i64, h as i64, topk as i64);
    let mut b = dev.stream().launch_builder(&func);
    b.arg(&logits);
    b.arg(sink);
    b.arg(idxs);
    b.arg(&start_i);
    b.arg(&count_i);
    b.arg(&h_i);
    b.arg(&topk_i);
    b.arg(&scale);
    unsafe { b.launch(super::warp_cfg(h)) }.map_err(|e| launch_err("sink_softmax_f32", e))?;
    matmul_f32(dev, &logits, keys, 1, h, topk, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Device, Tensor};

    fn wave(len: usize, k: f32, amp: f32) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * k).sin() * amp).collect()
    }

    fn download(dev: &CudaDevice, s: &CudaSlice<f32>) -> Vec<f32> {
        dev.stream().clone_dtoh(s).unwrap()
    }

    /// The card's fp8 and fp4 round trips give the host's bits on rows of varied magnitude.
    #[test]
    fn the_card_rounds_as_the_host_does() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card rounding is NOT covered by this run");
            return;
        };
        let rows = 7usize;
        let v: Vec<f32> = (0..rows * 128)
            .map(|i| {
                let r = i / 128;
                ((i as f32) * 0.37).sin() * 10f32.powi(r as i32 - 3)
                    + if i % 97 == 0 { 400.0 } else { 0.0 }
            })
            .collect();
        let t = Tensor::from_vec(v.clone(), (rows, 128), &Device::Cpu).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let want8 = crate::inference::model::deepseek_v41::attention::act_quant_fp8_e4m3(&t, 32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut x = upload_f32(&dev, "t", &v).unwrap();
        fp8_round(&dev, &mut x, 32).unwrap();
        assert_eq!(bits(&download(&dev, &x)), bits(&want8), "fp8");
        for (block, pow2) in [(16usize, false), (32, true)] {
            let want4 = crate::inference::model::deepseek_v41::band::fp4_act_quant(&t, block, pow2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut x = upload_f32(&dev, "t", &v).unwrap();
            fp4_round(&dev, &mut x, block, pow2).unwrap();
            assert_eq!(bits(&download(&dev, &x)), bits(&want4), "fp4 block {block}");
        }
    }

    /// The card's top-k keeps the set a stable descending sort keeps, ties to the lower index,
    /// within the reachable positions, and fills the rest with -1.
    #[test]
    fn the_card_selects_the_hosts_top_k() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card top-k is NOT covered by this run");
            return;
        };
        for (g, reach, k) in [
            (5000usize, 4321usize, 64usize),
            (40, 40, 64),
            (3, 3, 16),
            (7, 0, 4),
        ] {
            // Ties: every third score repeats.
            let scores: Vec<f32> = (0..g).map(|i| ((i / 3) as f32 * 0.731).sin()).collect();
            let sd = upload_f32(&dev, "s", &scores).unwrap();
            let (out, outf) = topk_select(&dev, &sd, reach, k, 10).unwrap();
            let got: Vec<i32> = dev.stream().clone_dtoh(&out).unwrap();
            let gotf = download(&dev, &outf);
            let mut order: Vec<usize> = (0..reach).collect();
            order.sort_by(|&a, &c| scores[c].partial_cmp(&scores[a]).unwrap());
            let mut want: Vec<i32> = order.iter().take(k).map(|&t| (t + 10) as i32).collect();
            want.resize(k, -1);
            let mut got_sorted: Vec<i32> = got[..k.min(reach)].to_vec();
            got_sorted.sort_unstable();
            let mut want_sorted = want[..k.min(reach)].to_vec();
            want_sorted.sort_unstable();
            assert_eq!(got_sorted, want_sorted, "g {g} reach {reach} k {k}");
            assert!(got[k.min(reach)..].iter().all(|&v| v == -1));
            assert_eq!(gotf.iter().map(|&v| v as i32).collect::<Vec<_>>(), got);
        }
    }

    /// The card's router chooses the host's experts with the host's weights.
    #[test]
    fn the_card_routes_as_the_host_does() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card router is NOT covered by this run");
            return;
        };
        let (n, k, temp, route_scale) = (384usize, 6usize, 1.3f32, 2.5f32);
        let z = wave(n, 0.173, 3.0);
        let bias = wave(n, 0.91, 0.2);
        let softplus = |v: f32| if v > 20.0 { v } else { v.exp().ln_1p() };
        let scores: Vec<f32> = z.iter().map(|&v| softplus(v / temp).sqrt()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            (scores[b] + bias[b])
                .partial_cmp(&(scores[a] + bias[a]))
                .unwrap()
        });
        let chosen = &order[..k];
        for norm in [false, true] {
            let mut w: Vec<f32> = chosen.iter().map(|&e| scores[e]).collect();
            if norm {
                let s: f32 = w.iter().sum::<f32>() + 1e-20;
                for v in w.iter_mut() {
                    *v /= s;
                }
            }
            let zd = upload_f32(&dev, "z", &z).unwrap();
            let bd = upload_f32(&dev, "b", &bias).unwrap();
            let out = router_select(
                &dev,
                &zd.slice(0..n),
                &bd.slice(0..n),
                n,
                k,
                temp,
                norm,
                route_scale,
            )
            .unwrap();
            let got = download(&dev, &out);
            let ids: Vec<usize> = got[..k].iter().map(|&v| v as usize).collect();
            assert_eq!(ids, chosen, "norm {norm}");
            for (g, wi) in got[k..].iter().zip(&w) {
                let want = wi * route_scale;
                assert!(
                    (g - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "{g} vs {want}"
                );
            }
        }
    }

    /// The pooled latent of a group on the card equals the host's compressor pooling.
    #[test]
    fn the_card_pools_a_group_as_the_host_does() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card pooling is NOT covered by this run");
            return;
        };
        let (ratio, hd) = (4usize, 96usize);
        let kv = wave(ratio * hd, 0.31, 1.0);
        let score = wave(ratio * hd, 0.17, 2.0);
        let mut want = vec![0f32; hd];
        for c in 0..hd {
            let mut mx = f32::NEG_INFINITY;
            for r in 0..ratio {
                mx = mx.max(score[r * hd + c]);
            }
            let (mut den, mut acc) = (0f32, 0f32);
            for r in 0..ratio {
                let w = (score[r * hd + c] - mx).exp();
                den += w;
                acc += w * kv[r * hd + c];
            }
            want[c] = acc / den;
        }
        let kvd = upload_f32(&dev, "kv", &kv).unwrap();
        let sd = upload_f32(&dev, "s", &score).unwrap();
        let got = download(
            &dev,
            &compress_pool(
                &dev,
                &kvd.slice(0..kv.len()),
                &sd.slice(0..score.len()),
                ratio,
                hd,
            )
            .unwrap(),
        );
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{g} vs {w}");
        }
    }

    /// A ring appended past its capacity holds the newest rows oldest first, and the gather
    /// reads them in that order followed by the compressed rows.
    #[test]
    fn the_ring_and_the_gather_keep_the_hosts_order() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card ring is NOT covered by this run");
            return;
        };
        let (cap, d) = (4usize, 8usize);
        let mut ring = RingDev::open(&dev, &[], cap, d).unwrap();
        let mut comp = GrowDev::open(&dev, &[], d).unwrap();
        let mut host_ring: std::collections::VecDeque<Vec<f32>> = Default::default();
        let mut host_comp: Vec<Vec<f32>> = Vec::new();
        for t in 0..7 {
            let row: Vec<f32> = (0..d).map(|i| (t * 10 + i) as f32).collect();
            let rd = upload_f32(&dev, "row", &row).unwrap();
            ring.append(&dev, &rd.slice(0..d)).unwrap();
            host_ring.push_back(row.clone());
            while host_ring.len() > cap {
                host_ring.pop_front();
            }
            if t % 2 == 1 {
                let crow: Vec<f32> = row.iter().map(|v| -v).collect();
                let cd = upload_f32(&dev, "crow", &crow).unwrap();
                comp.append(&dev, &cd.slice(0..d)).unwrap();
                host_comp.push(crow);
            }
        }
        let w = ring.len();
        assert_eq!(w, host_ring.len());
        let mut idxs: Vec<i32> = (0..w as i32).collect();
        idxs.push(-1);
        idxs.push((w + 2) as i32);
        idxs.push(w as i32);
        let id = upload_i32(&dev, "idxs", &idxs).unwrap();
        let got = download(
            &dev,
            &gather_ring(&dev, &ring, Some(&comp), &id, idxs.len()).unwrap(),
        );
        let mut want: Vec<f32> = host_ring.iter().flatten().copied().collect();
        want.extend(std::iter::repeat_n(0f32, d));
        want.extend(&host_comp[2]);
        want.extend(&host_comp[0]);
        assert_eq!(got, want);
    }
}
