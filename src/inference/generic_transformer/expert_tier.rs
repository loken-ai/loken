//! A layer's routed experts kept in host memory, with an adaptive cache of them on a card.
//!
//! The experts stay in the file's mapping. The card holds `slots` of them in stacks of the same
//! format and shape as the host's, so the expert kernels that read a stack on a card read a slot
//! as they read an expert: the routed ids are rewritten to slot numbers. A routed expert that
//! has no slot is computed on the host from the mapping, beside the card's own launch, and its
//! product joins the card's. Every routed expert's use is counted, and a missing expert that
//! the counts show to be hotter than the coldest slot not in use this token is copied into it
//! by a worker, which never holds up a token.
//!
//! The counts decay: when `n_experts` tokens have routed through a layer, each count halves, so
//! the cache follows the routing of the text in hand rather than the routing since the load.

use super::moe_weights::{ExpertRouting, MoeWeights};
use crate::tensor::quantized::{GgmlDType, QCudaStorage, QHostTensor, QTensor};
use crate::tensor::{Device, Result, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// What the tier knows of each expert and slot, under one lock: every access is one token's
/// bookkeeping, short next to the launches around it.
struct TierState {
    /// Slot holding each expert, `-1` when it has none.
    slot_of: Vec<i32>,
    /// Expert in each slot; `-1` for a slot never filled and `-2` for one being filled.
    owner: Vec<i32>,
    /// Decayed count of the times each expert was routed to.
    heat: Vec<f32>,
    /// Tokens since the counts last halved.
    since_decay: usize,
}

/// One layer's cache.
pub struct LayerTier {
    dev: Device,
    /// Gate, up and down stacks as the file stores them, `[n_experts, ..]`.
    host: [Arc<QTensor>; 3],
    /// The same stacks on the card, `[slots, ..]`.
    card: [QTensor; 3],
    /// Bytes of one expert in each stack.
    expert_bytes: [usize; 3],
    n_experts: usize,
    state: Mutex<TierState>,
    hits: AtomicU64,
    routed: AtomicU64,
    promotions: AtomicU64,
    /// The layer that reports the cache's hit rate.
    leader: std::sync::atomic::AtomicBool,
    /// What a wide batch of rows reads instead: the whole layer, on the card for the one launch.
    stage: Option<Stage>,
    /// The stream this layer's copies into the cache run on.
    copy: Arc<cudarc::driver::CudaStream>,
    /// A copy into the cache is queued or running.
    promoting: std::sync::atomic::AtomicBool,
    /// Microseconds a decoded token took here, smoothed: what a batch's rows cost one by one.
    decode_us: AtomicU64,
    /// What a decoded token moves, made on the first.
    scratch: Mutex<Option<Scratch>>,
}

/// What one decoded token moves between the card and the host, in page-locked memory and in
/// buffers on the card that stay where they are: the copies are asynchronous and the token waits
/// once, for the last of those that come back.
struct Scratch {
    pinned: *mut u8,
    /// Floats of the hidden row, and the routed experts per token, that the buffers are sized for.
    hidden: usize,
    k: usize,
    /// On the card: the slot of each routed pair, its weight if the card runs it, and the
    /// product the host sends back.
    slots: Tensor,
    weights: Tensor,
    product: Tensor,
}

// Safety: the page-locked block is used by one decode at a time, under the tier's lock.
unsafe impl Send for Scratch {}

impl Scratch {
    fn new(dev: &Device, hidden: usize, k: usize) -> Result<Self> {
        dev.as_cuda_device()?
            .0
            .context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        let bytes = 4 * (2 * hidden + 4 * k);
        let pinned = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
            .map_err(|e| crate::tensor::Error::msg(format!("pinned scratch: {e}")))?
            as *mut u8;
        let host = Device::Cpu;
        Ok(Self {
            pinned,
            hidden,
            k,
            slots: Tensor::from_vec(vec![0u32; k], (k,), &host)?.to_device(dev)?,
            weights: Tensor::from_vec(vec![0f32; k], (1, k), &host)?.to_device(dev)?,
            product: Tensor::from_vec(vec![0f32; hidden], (1, hidden), &host)?.to_device(dev)?,
        })
    }

    /// The row the card sends the host, the choice's ids and weights, then what goes back:
    /// the product, the slots and the weights the card runs.
    fn parts(
        &self,
    ) -> (
        &mut [f32],
        &mut [u32],
        &mut [f32],
        &mut [f32],
        &mut [u32],
        &mut [f32],
    ) {
        let (h, k) = (self.hidden, self.k);
        unsafe {
            let f = self.pinned as *mut f32;
            let u = self.pinned as *mut u32;
            (
                std::slice::from_raw_parts_mut(f, h),
                std::slice::from_raw_parts_mut(u.add(h), k),
                std::slice::from_raw_parts_mut(f.add(h + k), k),
                std::slice::from_raw_parts_mut(f.add(h + 2 * k), h),
                std::slice::from_raw_parts_mut(u.add(2 * h + 2 * k), k),
                std::slice::from_raw_parts_mut(f.add(2 * h + 3 * k), k),
            )
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe { cudarc::driver::result::free_host(self.pinned as *mut std::ffi::c_void) }.ok();
    }
}

/// The card's address of the first element of `t`, which a copy reads or writes.
fn address_of<T: crate::tensor::kernel_ffi::CudaDType>(t: &Tensor) -> Result<u64> {
    use cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        crate::tensor::StorageView::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            Ok(slice.device_ptr(slice.stream()).0
                + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
        }
        _ => crate::tensor::bail!("an expert tier buffer is on the card"),
    }
}

/// The card's room for a layer's experts to pass through, and the copies that fill it.
pub struct CardStager {
    dev: Device,
    /// Bytes per microsecond the host sends to this card, measured when the stager is made.
    bandwidth: f64,
    copy: Arc<cudarc::driver::CudaStream>,
    buffers: [StageBuffer; 2],
}

/// One of two sets of arenas, a stack each, that consecutive layers take turns over.
struct StageBuffer {
    arenas: [Arc<cudarc::driver::CudaSlice<u8>>; 3],
    /// Recorded when the layer that read the buffer last has queued all its kernels: the next
    /// fill waits for it.
    free: Mutex<Option<cudarc::driver::CudaEvent>>,
}

/// A layer's view of its buffer: the three stacks as tensors over the arenas, and the copies in
/// flight into them.
struct Stage {
    stager: Arc<CardStager>,
    buffer: usize,
    stacks: [QTensor; 3],
    /// The fill's end, when one has been queued and not yet read.
    ready: Mutex<Option<cudarc::driver::CudaEvent>>,
    /// Host ranges, per stack, that are page-locked: copies from them are asynchronous.
    pinned: [Option<(usize, usize)>; 3],
}

/// The stacks' dims with the expert axis replaced by `n`.
fn with_experts(host: &QTensor, n: usize) -> Vec<usize> {
    let mut dims = host.shape().dims().to_vec();
    dims[0] = n;
    dims
}

/// The first `n` experts of `host` on `dev`: a view of the host's own bytes, so the host holds no
/// second copy of what the card was filled from.
fn first_experts(host: &Arc<QTensor>, n: usize, dev: &Device) -> Result<QTensor> {
    let inner = host.native_qtensor();
    let bytes = inner.data();
    let per_expert = bytes.len() / host.shape().dims()[0];
    let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(inner.clone());
    let view = unsafe {
        QHostTensor::view(
            owner,
            bytes.as_ptr(),
            0,
            n * per_expert,
            host.dtype(),
            with_experts(host, n),
        )
    }
    .map_err(|e| crate::tensor::Error::msg(e.0))?;
    QTensor::from_native(Arc::new(view), dev)
}

impl super::GenericHeteroTransformer {
    /// Give every layer whose experts the plan left on the host a cache of them on its card:
    /// `slots` takes what each card has left of its budget, `free_before` being what the card
    /// had free before the load.
    pub(super) fn attach_expert_tiers(
        &mut self,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        free_before: &std::collections::HashMap<usize, u64>,
    ) -> Result<()> {
        use super::LayerDevice;
        use crate::inference::place::{expert_cache, layer_executor::DeviceKind};
        for (&ordinal, dev) in cuda_devices {
            let tiered: Vec<usize> = (0..self.layers.len())
                .filter(|&l| self.layer_devs[l] == LayerDevice::Cuda(ordinal))
                .filter(|&l| {
                    self.layers[l]
                        .moe
                        .as_ref()
                        .is_some_and(|m| m.experts_on_host() && serves(m))
                })
                .collect();
            let Some(&first) = tiered.first() else {
                continue;
            };
            // Room for the layers' experts to pass through: two layers at once, each as wide
            // as the widest stack of any layer.
            let mut widest = [0usize; 3];
            for &l in &tiered {
                let m = self.layers[l]
                    .moe
                    .as_ref()
                    .expect("a tiered layer has experts");
                let stacks = [
                    m.gate_exps_sep.as_ref().expect("separate stacks"),
                    m.up_exps_sep.as_ref().expect("separate stacks"),
                    &m.down_exps,
                ];
                for i in 0..3 {
                    widest[i] = widest[i].max(stacks[i].native_qtensor().data().len());
                }
            }
            let bytes = expert_bytes(
                self.layers[first]
                    .moe
                    .as_ref()
                    .expect("a tiered layer has experts"),
            );
            let budget = plan
                .segments
                .iter()
                .filter(|s| s.kind == DeviceKind::Cuda(ordinal))
                .map(|s| s.free_memory_bytes)
                .max()
                .unwrap_or(0);
            let before = free_before.get(&ordinal).copied().unwrap_or(0);
            // What the wide batches the experts now run on the card for keep live: the chunk the
            // prefill is to run at. The plan already leaves a reserve free for the work of a
            // forward, so the cache gives up only what the chunk needs beyond it.
            let shape = self.prefill_shape_with(true);
            let chunk_bytes = (shape.bytes_per_token as u64)
                .saturating_mul(shape.target_chunk as u64)
                .saturating_add(shape.fixed_bytes as u64);
            let held: u64 = 2 * widest.iter().sum::<usize>() as u64
                + chunk_bytes.saturating_sub(before.saturating_sub(budget));
            let slots = expert_cache::slots_per_layer(
                budget.saturating_sub(held),
                before,
                expert_cache::free_now(dev),
                tiered.len(),
                bytes,
            );
            if slots == 0 {
                continue;
            }
            let stager = CardStager::new(dev, widest)?;
            let mut pin_left = expert_cache::host_pin_budget();
            for (rank, &l) in tiered.iter().enumerate() {
                let moe = self.layers[l]
                    .moe
                    .as_mut()
                    .expect("a tiered layer has experts");
                let mut tier = LayerTier::new(moe, dev, slots)?;
                let layer_bytes = (tier.expert_bytes.iter().sum::<usize>() * tier.n_experts) as u64;
                let pin = layer_bytes <= pin_left;
                pin_left = pin_left.saturating_sub(if pin { layer_bytes } else { 0 });
                tier.stage = Some(Stage::new(&stager, rank % 2, &tier.host, pin)?);
                if l == first {
                    tier.leader.store(true, Ordering::Relaxed);
                }
                moe.tier = Some(Arc::new(tier));
            }
            tracing::info!(
                "expert tier on card {ordinal}: {} layers, {} experts each cached of {}, {:.2} GB",
                tiered.len(),
                slots.min(self.layers[first].moe.as_ref().map_or(0, |m| m.n_experts)),
                self.layers[first].moe.as_ref().map_or(0, |m| m.n_experts),
                (tiered.len() * slots * bytes) as f64 / 1e9
            );
        }
        Ok(())
    }
}

impl LayerTier {
    /// This tier with room to read a wide batch's experts whole: `buffer` of `stager`, the
    /// host stacks page-locked when `pin`.
    pub fn with_stage(
        mut self,
        stager: &Arc<CardStager>,
        buffer: usize,
        pin: bool,
    ) -> Result<Self> {
        self.stage = Some(Stage::new(stager, buffer, &self.host, pin)?);
        Ok(self)
    }
}

pub use super::moe_weights::prefill_bytes_per_token;

/// What a layer's tier can serve: separate gate and up stacks read by a SiLU, no biases, and a
/// format both the card's expert kernels and the host's dot read.
pub fn serves(moe: &MoeWeights) -> bool {
    let (Some(gate), Some(up)) = (moe.gate_exps_sep.as_ref(), moe.up_exps_sep.as_ref()) else {
        return false;
    };
    let kernels = |t: &QTensor| {
        crate::inference::moe_cuda::gemm::expert_kernels_serve(t.dtype())
            && crate::inference::moe_cpu::cpu_qmatmul_supported(t.dtype())
    };
    moe.use_silu
        && !moe.gelu
        && !moe.up_relu2
        && moe.swiglu_clamp.is_none()
        && moe.gate_exps_bias.is_none()
        && moe.up_exps_bias.is_none()
        && moe.down_exps_bias.is_none()
        && moe.decode_slots.is_some()
        && gate.dtype() == up.dtype()
        && kernels(gate)
        && kernels(&moe.down_exps)
}

/// Whether the routed experts of the file `content` read as a tier serves them: in every layer
/// that has experts, separate gate and up stacks of one format, no biases, and a format that the
/// card's kernels and the host's dot both read.
pub fn header_serves(content: &crate::tensor::quantized::gguf_file::Content) -> bool {
    let kernels = |d: GgmlDType| {
        crate::inference::moe_cuda::gemm::expert_kernels_serve(d)
            && crate::inference::moe_cpu::cpu_qmatmul_supported(d)
    };
    let mut layers = 0;
    for (name, gate) in &content.tensor_infos {
        let Some(prefix) = name.strip_suffix(".ffn_gate_exps.weight") else {
            continue;
        };
        let (up, down) = (
            content
                .tensor_infos
                .get(&format!("{prefix}.ffn_up_exps.weight")),
            content
                .tensor_infos
                .get(&format!("{prefix}.ffn_down_exps.weight")),
        );
        let biased = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .any(|n| {
                content
                    .tensor_infos
                    .contains_key(&format!("{prefix}.{n}.bias"))
            });
        let (Some(up), Some(down)) = (up, down) else {
            return false;
        };
        if biased
            || up.ggml_dtype != gate.ggml_dtype
            || !kernels(gate.ggml_dtype)
            || !kernels(down.ggml_dtype)
        {
            return false;
        }
        layers += 1;
    }
    layers > 0
}

/// Bytes of one expert of `moe` across its three stacks.
pub fn expert_bytes(moe: &MoeWeights) -> usize {
    let stacks = [
        moe.gate_exps_sep.as_ref(),
        moe.up_exps_sep.as_ref(),
        Some(&moe.down_exps),
    ];
    stacks
        .iter()
        .flatten()
        .map(|t| t.native_qtensor().data().len() / t.shape().dims()[0].max(1))
        .sum()
}

impl LayerTier {
    /// A cache of `slots` experts on `dev` for the layer `moe`, filled with its first experts.
    pub fn new(moe: &MoeWeights, dev: &Device, slots: usize) -> Result<Self> {
        let slots = slots.min(moe.n_experts);
        let host = [
            moe.gate_exps_sep
                .clone()
                .expect("a tier serves separate gate and up stacks"),
            moe.up_exps_sep
                .clone()
                .expect("a tier serves separate gate and up stacks"),
            moe.down_exps.clone(),
        ];
        let card = [
            first_experts(&host[0], slots, dev)?,
            first_experts(&host[1], slots, dev)?,
            first_experts(&host[2], slots, dev)?,
        ];
        let expert_bytes = [0, 1, 2]
            .map(|i| host[i].native_qtensor().data().len() / host[i].shape().dims()[0].max(1));
        let mut slot_of = vec![-1i32; moe.n_experts];
        let mut owner = vec![-1i32; slots];
        for s in 0..slots {
            slot_of[s] = s as i32;
            owner[s] = s as i32;
        }
        Ok(Self {
            dev: dev.clone(),
            host,
            card,
            expert_bytes,
            n_experts: moe.n_experts,
            state: Mutex::new(TierState {
                slot_of,
                owner,
                heat: vec![0.0; moe.n_experts],
                since_decay: 0,
            }),
            hits: AtomicU64::new(0),
            routed: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            leader: std::sync::atomic::AtomicBool::new(false),
            stage: None,
            copy: dev
                .as_cuda_device()?
                .0
                .context()
                .new_stream()
                .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))?,
            promoting: std::sync::atomic::AtomicBool::new(false),
            decode_us: AtomicU64::new(0),
            scratch: Mutex::new(None),
        })
    }

    /// Wait for the copy in flight, if any, to land.
    pub fn settle(&self) {
        while self.promoting.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }

    /// Experts the card holds.
    pub fn slots(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .owner
            .len()
    }

    /// Whether `expert` is on the card now.
    pub fn resident(&self, expert: usize) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).slot_of[expert] >= 0
    }

    /// (routed experts, of which the card held, copies made) since the load.
    pub fn counters(&self) -> (u64, u64, u64) {
        (
            self.routed.load(Ordering::Relaxed),
            self.hits.load(Ordering::Relaxed),
            self.promotions.load(Ordering::Relaxed),
        )
    }

    /// One decoded token's expert block: `xs` `[1, hidden]` and the router's choice `ids`,
    /// `weights` `[1, k]`, all on the card. Returns `[1, hidden]` on the card.
    pub(super) fn decode(
        self: &Arc<Self>,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let started = std::time::Instant::now();
        let out = self.decode_token(moe, xs, ids, weights);
        let us = started.elapsed().as_micros() as u64;
        let before = self.decode_us.load(Ordering::Relaxed);
        // A weighted mean that forgets a token in sixteen.
        self.decode_us.store(
            if before == 0 {
                us
            } else {
                (before * 15 + us) / 16
            },
            Ordering::Relaxed,
        );
        out
    }

    fn decode_token(
        self: &Arc<Self>,
        moe: &MoeWeights,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let k = moe.n_experts_used;
        let host = Device::Cpu;
        let dev = xs.device().clone();
        let hidden = xs.elem_count();
        let cd = dev.as_cuda_device()?;
        let stream = cd.0.stream().clone();
        let mut guard = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(Scratch::new(&dev, hidden, k)?);
        }
        let scratch = guard.as_ref().expect("made above");
        let (h_x, h_ids, h_w, h_out, h_slots, h_hw) = scratch.parts();
        // The hidden row and the router's choice come to the host together, behind the work that
        // made them, and the token waits once.
        let x_card = xs.contiguous()?;
        let ids_card = ids.to_dtype(crate::tensor::DType::U32)?.contiguous()?;
        let w_card = weights.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        for (dst, src, bytes) in [
            (
                h_x.as_mut_ptr() as *mut std::ffi::c_void,
                address_of::<f32>(&x_card)?,
                4 * hidden,
            ),
            (
                h_ids.as_mut_ptr() as *mut std::ffi::c_void,
                address_of::<u32>(&ids_card)?,
                4 * k,
            ),
            (
                h_w.as_mut_ptr() as *mut std::ffi::c_void,
                address_of::<f32>(&w_card)?,
                4 * k,
            ),
        ] {
            driver("choice to host", unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(dst, src, bytes, stream.cu_stream())
            })?;
        }
        let prof = crate::inference::offload::stage_prof_enabled();
        let t_wait = prof.then(std::time::Instant::now);
        stream
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("choice to host: {e:?}")))?;
        if let Some(t) = t_wait {
            crate::inference::offload::prof_record(
                "tier wait for the choice",
                t.elapsed().as_nanos() as u64,
            );
        }
        let (ids_h, w_h): (Vec<u32>, Vec<f32>) = (h_ids.to_vec(), h_w.to_vec());
        // Partition by what the card holds now, and count the use.
        let mut slot_ids = vec![0u32; k];
        let mut hit_weights = vec![0f32; k];
        let (mut miss_ids, mut miss_weights) = (Vec::new(), Vec::new());
        let mut in_use = vec![false; self.slots()];
        let mut is_hit = vec![false; k];
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for (i, (&e, &w)) in ids_h.iter().zip(&w_h).enumerate() {
                st.heat[e as usize] += 1.0;
                match st.slot_of[e as usize] {
                    s if s >= 0 => {
                        slot_ids[i] = s as u32;
                        hit_weights[i] = w;
                        in_use[s as usize] = true;
                        is_hit[i] = true;
                    }
                    _ => {
                        miss_ids.push(e);
                        miss_weights.push(w);
                    }
                }
            }
            st.since_decay += 1;
            if st.since_decay >= self.n_experts {
                st.since_decay = 0;
                st.heat.iter_mut().for_each(|h| *h *= 0.5);
            }
        }
        let hits = k - miss_ids.len();
        let routed = self.routed.fetch_add(k as u64, Ordering::Relaxed) + k as u64;
        self.hits.fetch_add(hits as u64, Ordering::Relaxed);
        // The leader says how the cache is doing, once per window of tokens as wide as the
        // experts it chooses among.
        let window = (self.n_experts * k) as u64;
        if self.leader.load(Ordering::Relaxed) && routed % window < k as u64 {
            let (_, h, p) = self.counters();
            tracing::info!(
                "expert tier layer sample: {:.1}% of {} routed experts on the card, {} copies",
                100.0 * h as f64 / routed as f64,
                routed,
                p
            );
        }

        // The card's share is launched first, and runs while the host does its own.
        let card_out = if hits > 0 {
            // A routed pair whose expert is missing reads a slot that is in use and weighs zero.
            let filler = (0..k)
                .find(|&i| is_hit[i])
                .map(|i| slot_ids[i])
                .unwrap_or(0);
            for i in 0..k {
                if !is_hit[i] {
                    slot_ids[i] = filler;
                }
            }
            h_slots.copy_from_slice(&slot_ids);
            h_hw.copy_from_slice(&hit_weights);
            for (dst, src, bytes) in [
                (
                    address_of::<u32>(&scratch.slots)?,
                    h_slots.as_ptr() as *const std::ffi::c_void,
                    4 * k,
                ),
                (
                    address_of::<f32>(&scratch.weights)?,
                    h_hw.as_ptr() as *const std::ffi::c_void,
                    4 * k,
                ),
            ] {
                driver("slots to card", unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, stream.cu_stream())
                })?;
            }
            let routing = ExpertRouting {
                slots: moe.decode_slots.clone().expect("a tier layer is on a card"),
                experts: scratch.slots.clone(),
                topk: k,
            };
            let h = routing.gate_up_silu(xs, &self.card[0], &self.card[1])?;
            Some(routing.down_reduce(&h, &self.card[2], &scratch.weights, 1, false, None, None)?)
        } else {
            None
        };

        // The host's share: the experts the card does not hold, from the mapping.
        let t_host = prof.then(std::time::Instant::now);
        let host_out = if miss_ids.is_empty() {
            false
        } else {
            let m = miss_ids.len();
            let x = Tensor::from_vec(h_x.to_vec(), (1, hidden), &host)?;
            let routing =
                ExpertRouting::group(&Tensor::from_vec(miss_ids.clone(), (1, m), &host)?, m)?;
            let h = routing.host_gate_up_silu(&x, &self.host[0], &self.host[1])?;
            let w = Tensor::from_vec(miss_weights.clone(), (1, m), &host)?;
            let product = routing.host_down_reduce(&h, &self.host[2], &w, 1, None)?;
            h_out.copy_from_slice(&product.flatten_all()?.to_vec1::<f32>()?);
            driver("product to card", unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    address_of::<f32>(&scratch.product)?,
                    h_out.as_ptr() as *const std::ffi::c_void,
                    4 * hidden,
                    stream.cu_stream(),
                )
            })?;
            true
        };
        if let Some(t) = t_host {
            crate::inference::offload::prof_record(
                "tier host experts",
                t.elapsed().as_nanos() as u64,
            );
        }
        // Fresh tensors: the buffers on the card are this layer's, written again by the next token.
        let out = match (card_out, host_out) {
            (Some(c), true) => (c.to_dtype(crate::tensor::DType::F32)? + &scratch.product)?,
            (Some(c), false) => c,
            (None, true) => scratch.product.affine(1.0, 0.0)?,
            (None, false) => unreachable!("a token routes to at least one expert"),
        };
        drop(guard);

        self.promote(&miss_ids, &in_use);
        Ok(out)
    }

    /// Hand the worker the hottest missing expert if it is hotter than the coldest slot that
    /// this token did not read. The copy runs beside the tokens that follow: the expert is
    /// routed to the card only once it has landed.
    fn promote(self: &Arc<Self>, missing: &[u32], in_use: &[bool]) {
        if self.promoting.swap(true, Ordering::AcqRel) {
            return;
        }
        let chosen = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let hot = missing
                .iter()
                .max_by(|&&a, &&b| st.heat[a as usize].total_cmp(&st.heat[b as usize]))
                .copied();
            let victim = (0..st.owner.len())
                .filter(|&s| !in_use[s] && st.owner[s] != -2)
                .min_by(|&a, &b| slot_heat(&st, a).total_cmp(&slot_heat(&st, b)));
            match (hot, victim) {
                (Some(hot), Some(victim)) if st.heat[hot as usize] > slot_heat(&st, victim) => {
                    let old = st.owner[victim];
                    if old >= 0 {
                        st.slot_of[old as usize] = -1;
                    }
                    st.owner[victim] = -2;
                    Some((hot as usize, victim))
                }
                _ => None,
            }
        };
        let Some((expert, slot)) = chosen else {
            self.promoting.store(false, Ordering::Release);
            return;
        };
        let tier = self.clone();
        copier(self.dev.as_cuda_device().map_or(0, |d| d.0.ordinal()))
            .send(Box::new(move || {
                let landed = tier.copy_in(expert, slot);
                let mut st = tier.state.lock().unwrap_or_else(|e| e.into_inner());
                match landed {
                    Ok(()) => {
                        st.owner[slot] = expert as i32;
                        st.slot_of[expert] = slot as i32;
                        tier.promotions.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        tracing::warn!("expert tier: copying expert {expert} to slot {slot}: {e}");
                        st.owner[slot] = -1;
                    }
                }
                drop(st);
                tier.promoting.store(false, Ordering::Release);
            }))
            .ok();
    }

    /// Expert `expert` of the host stacks into slot `slot` of the card's, and the copy done.
    fn copy_in(&self, expert: usize, slot: usize) -> Result<()> {
        let cd = self.dev.as_cuda_device()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        for i in 0..3 {
            let eb = self.expert_bytes[i];
            let data = self.host[i].native_qtensor().data();
            let dst = self.card[i].device_ptr()? as u64 + (slot * eb) as u64;
            self.upload(i, &data, expert * eb, eb, dst, &self.copy)?;
        }
        self.copy
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("slot copy: {e:?}")))
    }
}

/// The thread of a card that runs the copies into its caches, one at a time.
fn copier(ordinal: usize) -> std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>> {
    use std::collections::HashMap;
    use std::sync::{mpsc, OnceLock};
    static WORKERS: OnceLock<Mutex<HashMap<usize, mpsc::Sender<Box<dyn FnOnce() + Send>>>>> =
        OnceLock::new();
    let mut workers = WORKERS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    workers
        .entry(ordinal)
        .or_insert_with(|| {
            let (tx, rx) = mpsc::channel::<Box<dyn FnOnce() + Send>>();
            std::thread::Builder::new()
                .name(format!("expert-copy-{ordinal}"))
                .spawn(move || {
                    for job in rx {
                        job();
                    }
                })
                .expect("a thread for the expert copies");
            tx
        })
        .clone()
}

/// CUDA driver result as a tensor error.
fn driver(what: &str, r: cudarc::driver::sys::cudaError_enum) -> Result<()> {
    if r == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(crate::tensor::Error::msg(format!("{what}: {r:?}")))
    }
}

impl CardStager {
    /// Room on `dev` for a layer's experts to pass through: two sets of arenas, each a stack
    /// wide enough for the widest of the layers, `widest` bytes per stack.
    pub fn new(dev: &Device, widest: [usize; 3]) -> Result<Arc<Self>> {
        let cd = dev.as_cuda_device()?;
        let copy =
            cd.0.context()
                .new_stream()
                .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))?;
        let buffer = || -> Result<StageBuffer> {
            Ok(StageBuffer {
                arenas: [
                    QCudaStorage::arena(&cd, widest[0])
                        .map_err(|e| crate::tensor::Error::msg(e.to_string()))?,
                    QCudaStorage::arena(&cd, widest[1])
                        .map_err(|e| crate::tensor::Error::msg(e.to_string()))?,
                    QCudaStorage::arena(&cd, widest[2])
                        .map_err(|e| crate::tensor::Error::msg(e.to_string()))?,
                ],
                free: Mutex::new(None),
            })
        };
        let buffers = [buffer()?, buffer()?];
        let bandwidth = measure_upload_bandwidth(dev, &buffers[0].arenas[0], widest[0])?;
        Ok(Arc::new(Self {
            dev: dev.clone(),
            bandwidth,
            copy,
            buffers,
        }))
    }
}

/// Bytes per microsecond that page-locked host memory reaches the card at: a copy into `arena`,
/// the best of a few.
fn measure_upload_bandwidth(
    dev: &Device,
    arena: &Arc<cudarc::driver::CudaSlice<u8>>,
    arena_bytes: usize,
) -> Result<f64> {
    use cudarc::driver::DevicePtr;
    let cd = dev.as_cuda_device()?;
    cd.0.context()
        .bind_to_thread()
        .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
    let bytes = arena_bytes.min(64 << 20);
    let host = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
        .map_err(|e| crate::tensor::Error::msg(format!("pinned probe: {e}")))?;
    let dst = arena.device_ptr(arena.stream()).0;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let r = unsafe { cudarc::driver::sys::cuMemcpyHtoD_v2(dst, host, bytes) };
        let us = t.elapsed().as_secs_f64() * 1e6;
        if r == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            best = best.min(us);
        }
    }
    unsafe { cudarc::driver::result::free_host(host) }.ok();
    Ok(if best.is_finite() && best > 0.0 {
        bytes as f64 / best
    } else {
        0.0
    })
}

/// Page-lock `[ptr, ptr + len)` as far as whole pages allow, and give the locked range.
fn lock_pages(ptr: usize, len: usize) -> Option<(usize, usize)> {
    let page = crate::tensor::mapped::page_size()?;
    let lo = ptr.div_ceil(page) * page;
    let hi = (ptr + len) / page * page;
    if hi <= lo {
        return None;
    }
    // Read-only: the mapping is the file's.
    const READ_ONLY: u32 = 8;
    let r = unsafe {
        cudarc::driver::sys::cuMemHostRegister_v2(lo as *mut std::ffi::c_void, hi - lo, READ_ONLY)
    };
    (r == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS).then_some((lo, hi))
}

impl Stage {
    fn new(
        stager: &Arc<CardStager>,
        buffer: usize,
        host: &[Arc<QTensor>; 3],
        pin: bool,
    ) -> Result<Self> {
        let cd = stager.dev.as_cuda_device()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        let mk = |i: usize| -> QTensor {
            let inner = host[i].native_qtensor().clone();
            let len = inner.data().len();
            let storage = QCudaStorage::sharing(
                stager.buffers[buffer].arenas[i].clone(),
                len,
                host[i].dtype(),
                &cd,
            );
            QTensor::over_card(inner, &stager.dev, storage)
        };
        let pinned = [0, 1, 2].map(|i| {
            if !pin {
                return None;
            }
            let data = host[i].native_qtensor().data();
            lock_pages(data.as_ptr() as usize, data.len())
        });
        Ok(Self {
            stager: stager.clone(),
            buffer,
            stacks: [mk(0), mk(1), mk(2)],
            ready: Mutex::new(None),
            pinned,
        })
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        for r in self.pinned.iter().flatten() {
            unsafe { cudarc::driver::sys::cuMemHostUnregister(r.0 as *mut std::ffi::c_void) };
        }
    }
}

impl LayerTier {
    /// Whether a batch of `rows` pays for the whole layer's experts to come to the card: when
    /// the rows' routed pairs outnumber the experts, nearly every expert is used.
    ///
    /// The rows run one by one through the cache where that is cheaper: the copy of what the card
    /// does not hold, at the rate the host reaches the card, against the rows at the pace tokens
    /// have been decoded here. Before any token was decoded the copy is taken.
    pub fn stream_pays(&self, rows: usize, used: usize) -> bool {
        let Some(stage) = self.stage.as_ref() else {
            return false;
        };
        if rows * used < self.n_experts {
            return false;
        }
        let per_token = self.decode_us.load(Ordering::Relaxed) as f64;
        let held = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .slot_of
            .iter()
            .filter(|&&s| s >= 0)
            .count();
        let missing = (self.n_experts - held) * self.expert_bytes.iter().sum::<usize>();
        let copy_us = missing as f64 / stage.stager.bandwidth.max(1e-9);
        per_token == 0.0 || (rows as f64) * per_token > copy_us
    }

    /// Start the copies that fill this layer's buffer with every expert: the ones the card
    /// holds from the card, the others from the host's mapping. Nothing waits for them here.
    /// A fill already queued and not yet read stands.
    pub fn prefetch(&self) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        let mut ready = stage.ready.lock().unwrap_or_else(|e| e.into_inner());
        if ready.is_some() {
            return Ok(());
        }
        let cd = self.dev.as_cuda_device()?;
        cd.0.context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("bind context: {e:?}")))?;
        let copy = &stage.stager.copy;
        let buffer = &stage.stager.buffers[stage.buffer];
        if let Some(free) = buffer.free.lock().unwrap_or_else(|e| e.into_inner()).take() {
            copy.wait(&free)
                .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
        }
        let slot_of = self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .slot_of
            .clone();
        use cudarc::driver::DevicePtr;
        for i in 0..3 {
            let eb = self.expert_bytes[i];
            let arena = &buffer.arenas[i];
            let base = arena.device_ptr(arena.stream()).0 as u64;
            let slots = self.card[i].device_ptr()? as u64;
            let data = self.host[i].native_qtensor().data();
            // The kernels read whole blocks past the end of a stack, against activations padded
            // with zeros: that tail is zero, as every blob's is, and not what the layer before
            // left in the arena - a scale in it that is not finite would poison the zeros.
            driver("stage tail", unsafe {
                cudarc::driver::sys::cuMemsetD8Async(
                    base + data.len() as u64,
                    0,
                    crate::tensor::quantized::BLOB_TAIL_PAD_BYTES,
                    copy.cu_stream(),
                )
            })?;
            let mut e = 0;
            while e < self.n_experts {
                if slot_of[e] >= 0 {
                    driver("slot to stage", unsafe {
                        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            base + (e * eb) as u64,
                            slots + (slot_of[e] as usize * eb) as u64,
                            eb,
                            copy.cu_stream(),
                        )
                    })?;
                    e += 1;
                    continue;
                }
                // The run of experts the card does not hold, in one copy.
                let first = e;
                while e < self.n_experts && slot_of[e] < 0 {
                    e += 1;
                }
                self.upload(
                    i,
                    &data,
                    first * eb,
                    (e - first) * eb,
                    base + (first * eb) as u64,
                    copy,
                )?;
            }
        }
        *ready = Some(
            copy.record_event(None)
                .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?,
        );
        Ok(())
    }

    /// `len` bytes of stack `i` from `start` to the card at `dst`: from the page-locked part
    /// of the range as it stands, the rest through the driver's own staging.
    fn upload(
        &self,
        i: usize,
        data: &[u8],
        start: usize,
        len: usize,
        dst: u64,
        copy: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let (a, b) = (
            data.as_ptr() as usize + start,
            data.as_ptr() as usize + start + len,
        );
        let pinned = self.stage.as_ref().and_then(|s| s.pinned[i]);
        let (lo, hi) = pinned.map_or((b, b), |(lo, hi)| (lo.clamp(a, b), hi.clamp(a, b)));
        let (lo, hi) = if lo < hi { (lo, hi) } else { (b, b) };
        for (from, to) in [(a, lo), (lo, hi), (hi, b)] {
            if to > from {
                driver("expert to stage", unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        dst + (from - a) as u64,
                        from as *const std::ffi::c_void,
                        to - from,
                        copy.cu_stream(),
                    )
                })?;
            }
        }
        Ok(())
    }

    /// The layer's experts as stacks on the card, once the fill has landed: the compute stream
    /// waits for it. The fill is started here if nothing started it.
    pub(super) fn stacks(&self) -> Result<&[QTensor; 3]> {
        self.prefetch()?;
        let stage = self.stage.as_ref().expect("a staged layer");
        let event = stage
            .ready
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .expect("the fill was queued");
        let cd = self.dev.as_cuda_device()?;
        cd.0.stream()
            .wait(&event)
            .map_err(|e| crate::tensor::Error::msg(format!("wait: {e:?}")))?;
        Ok(&stage.stacks)
    }

    /// Forget the fills and holds of the layers' passes through the buffers. A pass cut short
    /// leaves a fill queued for a layer that never ran, and layers share two buffers, so the
    /// next pass would read another layer's experts as its own. The copies already issued
    /// are waited for first, so nothing writes a buffer after it is handed on.
    pub(super) fn reset_stage(&self) -> Result<()> {
        let Some(stage) = self.stage.as_ref() else {
            return Ok(());
        };
        stage
            .stager
            .copy
            .synchronize()
            .map_err(|e| crate::tensor::Error::msg(format!("copy stream: {e:?}")))?;
        *stage.ready.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *stage.stager.buffers[stage.buffer]
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        Ok(())
    }

    /// The layer's kernels are queued: the buffer is free once they have run.
    pub(super) fn release_stacks(&self) -> Result<()> {
        let stage = self.stage.as_ref().expect("a staged layer");
        let cd = self.dev.as_cuda_device()?;
        let event =
            cd.0.stream()
                .record_event(None)
                .map_err(|e| crate::tensor::Error::msg(format!("record: {e:?}")))?;
        *stage.stager.buffers[stage.buffer]
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(event);
        Ok(())
    }
}

/// Heat of the expert a slot holds; an empty slot is the coldest there is.
fn slot_heat(st: &TierState, slot: usize) -> f32 {
    match st.owner[slot] {
        e if e >= 0 => st.heat[e as usize],
        _ => -1.0,
    }
}

/// The dtype of a layer's stacks, for a log line.
pub fn stack_dtype(moe: &MoeWeights) -> Option<GgmlDType> {
    moe.gate_exps_sep.as_ref().map(|t| t.dtype())
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::tensor::quantized::QMatMul;
    use crate::tensor::DType;

    fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u64 + 1)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(40503));
                (((h >> 8) % 2001) as f32 / 1000.0 - 1.0) * scale
            })
            .collect()
    }

    /// Fill the arenas of `stager` with 0xFF, which read as f16 is not a number.
    fn poison(stager: &CardStager) {
        use cudarc::driver::DevicePtr;
        let cd = stager.dev.as_cuda_device().unwrap();
        cd.0.context().bind_to_thread().unwrap();
        for buffer in &stager.buffers {
            for arena in &buffer.arenas {
                let base = arena.device_ptr(arena.stream()).0;
                let r = unsafe { cudarc::driver::sys::cuMemsetD8_v2(base, 0xFF, arena.len()) };
                assert_eq!(r, cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS);
            }
        }
        cd.0.synchronize().unwrap();
    }

    /// A softmax-routed SiLU layer of `experts` experts on `dev` for its router and slots, the
    /// experts themselves on the host.
    fn layer(dev: &Device, experts: usize, used: usize, hidden: usize, ffn: usize) -> MoeWeights {
        layer_of(dev, experts, used, hidden, ffn, [GgmlDType::Q8_0; 2])
    }

    /// [`layer`] with the formats of the gate and up stacks and of the down stack chosen.
    fn layer_of(
        dev: &Device,
        experts: usize,
        used: usize,
        hidden: usize,
        ffn: usize,
        [gate_up, down]: [GgmlDType; 2],
    ) -> MoeWeights {
        let host = Device::Cpu;
        let stack = |seed, rows, cols, dtype| {
            let w = Tensor::from_vec(
                values(experts * rows * cols, seed, 0.5),
                (experts, rows, cols),
                &host,
            )
            .unwrap();
            QTensor::quantize_onto(&w, dtype, &host).unwrap()
        };
        MoeWeights {
            gate_inp: QMatMul::Tensor(
                Tensor::from_vec(values(experts * hidden, 7, 0.5), (experts, hidden), dev).unwrap(),
            ),
            gate_inp_scale: Tensor::ones(hidden, DType::F32, dev).unwrap(),
            gate_inp_bias: None,
            gate_exps_bias: None,
            up_exps_bias: None,
            down_exps_bias: None,
            swiglu_clamp: None,
            gate_up_exps: None,
            down_exps: Arc::new(stack(3, hidden, ffn, down)),
            gate_exps_sep: Some(Arc::new(stack(1, ffn, hidden, gate_up))),
            up_exps_sep: Some(Arc::new(stack(2, ffn, hidden, gate_up))),
            down_exps_scale: None,
            decode_slots: dev.runs_as_card().then(|| {
                Tensor::from_vec((0..used as u32).collect::<Vec<_>>(), (used,), &host)
                    .unwrap()
                    .to_device(dev)
                    .unwrap()
            }),
            n_experts: experts,
            n_experts_used: used,
            expert_ffn_dim: ffn,
            router_prenormed: true,
            use_silu: true,
            gelu: false,
            norm_topk: true,
            sigmoid_routing: None,
            up_relu2: false,
            shared_expert: None,
            gate_up_exps_f32: None,
            tier: None,
            down_exps_f32: None,
        }
    }

    /// A decoded token through the tier gives what the host's expert loop gives, whatever the
    /// cache holds: nothing hot, some of it, then the hot experts after the worker has copied them.
    #[test]
    fn a_token_through_the_tier_matches_the_host_whatever_the_cache_holds() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        let mut tiered = layer(&card, experts, used, hidden, ffn);
        let slots = 6;
        assert!(serves(&tiered));
        tiered.tier = Some(Arc::new(LayerTier::new(&tiered, &card, slots).unwrap()));
        let tier = tiered.tier.clone().unwrap();
        assert_eq!(tier.slots(), slots);
        // A few distinct inputs, repeated: their experts become the hot set.
        let inputs: Vec<Vec<f32>> = (0..3).map(|s| values(hidden, 100 + s, 1.0)).collect();
        let mut hit_rate_first = None;
        for token in 0..60 {
            let x = &inputs[token % inputs.len()];
            let on_host = Tensor::from_vec(x.clone(), (1, 1, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x.clone(), (1, 1, hidden), &card).unwrap();
            let want = reference
                .forward(&on_host, &on_host, hidden, 1e-5, false)
                .unwrap();
            let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got = tiered
                .forward(&on_card, &on_card, hidden, 1e-5, false)
                .unwrap();
            let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            tier.settle();
            let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 2e-2 * peak,
                "token {token}: max diff {worst} of {peak}"
            );
            if token == inputs.len() - 1 {
                let (routed, hits, _) = tier.counters();
                hit_rate_first = Some(hits as f64 / routed as f64);
            }
        }
        let (routed, hits, promoted) = tier.counters();
        assert!(promoted > 0, "the hot experts were never copied in");
        assert!(
            hits as f64 / routed as f64 > hit_rate_first.unwrap(),
            "the cache did not learn: {hits} of {routed}"
        );
    }

    /// A wide batch through the tier gives what the host's loop gives, with the layer's experts
    /// copied to the card from the host and from the cache, and with the host range locked or not.
    #[test]
    fn a_wide_batch_through_the_tier_matches_the_host() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        let (experts, used, hidden, ffn) = (32usize, 4usize, 64usize, 64usize);
        let reference = layer(&Device::Cpu, experts, used, hidden, ffn);
        for (pin, rows) in [(false, 24usize), (true, 24), (false, 1520), (true, 3000)] {
            let mut tiered = layer(&card, experts, used, hidden, ffn);
            let widest = [0, 1, 2].map(|i| {
                let t = [
                    tiered.gate_exps_sep.as_ref().unwrap(),
                    tiered.up_exps_sep.as_ref().unwrap(),
                    &tiered.down_exps,
                ][i];
                t.native_qtensor().data().len()
            });
            let stager = CardStager::new(&card, widest).unwrap();
            // What an earlier layer left in the arenas: bytes whose f16 scales are not finite.
            poison(&stager);
            // Some experts on the card, the rest to come from the host.
            let tier = LayerTier::new(&tiered, &card, 9)
                .unwrap()
                .with_stage(&stager, 0, pin)
                .unwrap();
            assert!(tier.stream_pays(rows, used));
            tiered.tier = Some(Arc::new(tier));
            let x = values(rows * hidden, 55, 1.0);
            let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
            let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
            let want = reference
                .forward(&on_host, &on_host, hidden, 1e-5, true)
                .unwrap();
            let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            // Twice, since the second fill comes after the first buffer was released.
            for _ in 0..2 {
                let got = tiered
                    .forward(&on_card, &on_card, hidden, 1e-5, true)
                    .unwrap();
                let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 2e-2 * peak,
                    "pin {pin}: max diff {worst} of {peak}"
                );
            }
        }
    }

    /// The same, in the formats of a real file and at the row counts where the expert-batched
    /// kernels take over from the sorted launches.
    #[test]
    fn a_wide_batch_matches_the_host_in_k_quants_across_the_batching_threshold() {
        let Ok(card) = Device::new_cuda(0) else {
            eprintln!("skipping: no usable CUDA device");
            return;
        };
        // The dimensions of a real file: a hidden width that is not a multiple of the expert width.
        let (experts, used, hidden, ffn) = (32usize, 8usize, 2560usize, 640usize);
        for formats in [
            [GgmlDType::Q4K, GgmlDType::Q5_0],
            [GgmlDType::Q4K, GgmlDType::Q8_0],
        ] {
            let reference = layer_of(&Device::Cpu, experts, used, hidden, ffn, formats);
            let mut tiered = layer_of(&card, experts, used, hidden, ffn, formats);
            let widest = [0, 1, 2].map(|i| {
                let t = [
                    tiered.gate_exps_sep.as_ref().unwrap(),
                    tiered.up_exps_sep.as_ref().unwrap(),
                    &tiered.down_exps,
                ][i];
                t.native_qtensor().data().len()
            });
            let stager = CardStager::new(&card, widest).unwrap();
            poison(&stager);
            let tier = LayerTier::new(&tiered, &card, 10)
                .unwrap()
                .with_stage(&stager, 0, false)
                .unwrap();
            tiered.tier = Some(Arc::new(tier));
            for rows in [100usize, 700, 900, 1520] {
                let x = values(rows * hidden, 77, 1.0);
                let on_host = Tensor::from_vec(x.clone(), (1, rows, hidden), &Device::Cpu).unwrap();
                let on_card = Tensor::from_vec(x, (1, rows, hidden), &card).unwrap();
                let want = reference
                    .forward(&on_host, &on_host, hidden, 1e-5, true)
                    .unwrap();
                let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let got = tiered
                    .forward(&on_card, &on_card, hidden, 1e-5, true)
                    .unwrap();
                let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "{formats:?} rows {rows}: non-finite output"
                );
                let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst <= 3e-2 * peak,
                    "{formats:?} rows {rows}: max diff {worst} of {peak}"
                );
            }
        }
    }
}
