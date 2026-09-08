use tracing::{info, warn};

/// Available host RAM in bytes via sysinfo's `MemAvailable` reading.
/// `MemAvailable` (`/proc/meminfo`) already excludes pages held by
/// other processes and currently-loaded models, so the value reflects
/// what a fresh allocation can realistically claim without forcing
/// swap. Mirrors the two-step pattern used in
/// `distributed::device_manager::available_memory_for_model`.
fn available_host_ram_bytes() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Safety headroom subtracted from `available_host_ram_bytes()` when budgeting the CPU spill.
///
/// It covers forward-pass activations, the page cache for mmap'd GGUF weights, one concurrent
/// unload-and-load swap, and whatever else the machine is running. All of those scale with the
/// machine, which is why this is a share of it rather than the flat 8 GB it was: on a 16 GB host
/// that figure was half the RAM and the planner would place nothing on the CPU at all, while on
/// a 256 GB one it left the page cache to be squeezed.
///
/// A quarter of the host, and nothing else. The floor and cap this carried were two more
/// hand-written sizes, and the floor was the same defect in miniature: at 4 GiB it reserved half
/// of an 8 GB machine, which is exactly what the flat figure did to a 16 GB one. Erring large is
/// still right - a per-segment OOM in the loader is far cheaper than swapping the host - but a
/// share errs large on its own, without a number chosen for a machine nobody measured.
fn host_ram_safety_headroom_bytes() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory() / 4
}

/// What a component may take on the host right now without pushing the machine into
/// swap: the available RAM less the safety headroom. Every CPU fallback is admitted
/// against this, the same figure the hetero planner budgets its CPU segment with.
pub fn host_spill_budget_bytes() -> u64 {
    available_host_ram_bytes().saturating_sub(host_ram_safety_headroom_bytes())
}

// LayerExecutor (stub) removed - the module's real value is the
// DeviceKind / DevicePlan / HeteroPlan / HeteroSegment types and
// the create_causal_mask helper, all of which are used widely. The
// LayerExecutor struct + execute_layer method had zero callers and
// did not implement real per-device dispatch.
// `DevicePlan` lived here and was removed with the two endpoints that were its only
// consumers. It was a SECOND planner: a fraction-of-memory layer split, with a branch
// that assumed "~2 GB per layer" whenever the model size was unknown - a hand-written
// memory size in a placement path, and an answer that could not agree with `HeteroPlan`
// below, which is what the loader actually runs. One question, one planner.

// ============================================================================
// Heterogeneous Device Planning (CUDA + wgpu + CPU)
// ============================================================================

/// Device kind for layer assignment (CUDA, OpenCL, or CPU)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cuda(usize),   // CUDA device index
    OpenCL(usize), // OpenCL device index
    Cpu,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(idx) => write!(f, "CUDA({})", idx),
            Self::OpenCL(idx) => write!(f, "opencl({})", idx),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

/// A contiguous segment of layers assigned to a single device
#[derive(Debug, Clone)]
pub struct HeteroSegment {
    pub kind: DeviceKind,
    pub layer_start: usize,
    pub layer_end: usize, // exclusive (layer_start..layer_end)
    pub free_memory_bytes: u64,
}

impl HeteroSegment {
    pub fn num_layers(&self) -> usize {
        self.layer_end.saturating_sub(self.layer_start)
    }
}

/// Heterogeneous device plan: assigns layer ranges to devices (CUDA, wgpu, CPU)
#[derive(Debug, Clone)]
pub struct HeteroPlan {
    pub segments: Vec<HeteroSegment>,
    pub total_layers: usize,
}

impl HeteroPlan {
    /// Calculate layer assignment across heterogeneous devices.
    ///
    /// Algorithm:
    /// 1. Collect all devices (CUDA, wgpu) with their free memory
    /// 2. Sort by free memory descending
    /// 3. Greedily assign layers proportional to each device's VRAM
    /// 4. Remaining layers go to CPU
    pub fn calculate(
        total_layers: usize,
        model_size_bytes: u64,
        cuda_devices: &[(usize, u64)], // (index, free_bytes)
        wgpu_devices: &[(usize, u64)], // (index, free_bytes) - approximate
        dequant_expansion_wgpu: f64,   // wgpu uses dequantized (3.5x for Q4->F16)
    ) -> Self {
        Self::calculate_with_kv(
            total_layers,
            model_size_bytes,
            cuda_devices,
            wgpu_devices,
            dequant_expansion_wgpu,
            /*kv_bytes_per_layer*/ 0,
        )
    }

    /// Like `calculate`, but reserves `kv_bytes_per_layer` bytes per
    /// layer on each GPU device for the KV cache and activations.
    /// Without this reservation, plans that fully use 95% of free VRAM
    /// for weights leave nothing for the per-layer K/V tensors and
    /// the loader OOMs after weights load (observed on deepseek-r1:32b
    /// at ctx=32K with kv_quant=Q8: weights spec was 50/14 across 2x16GB,
    /// post-load Phase 2 OOM on GPU 0).
    ///
    /// `kv_bytes_per_layer` should be computed from
    /// `2 * head_count_kv * head_dim * max_seq_len * kv_dtype_bytes`.
    pub fn calculate_with_kv(
        total_layers: usize,
        model_size_bytes: u64,
        cuda_devices: &[(usize, u64)],
        wgpu_devices: &[(usize, u64)],
        dequant_expansion_wgpu: f64,
        kv_bytes_per_layer: u64,
    ) -> Self {
        Self::calculate_with_kv_reserve(
            total_layers,
            model_size_bytes,
            cuda_devices,
            wgpu_devices,
            dequant_expansion_wgpu,
            kv_bytes_per_layer,
            0,
        )
    }

    /// Like `calculate_with_kv`, but reserves `gpu_runtime_reserve_bytes` of
    /// non-weight headroom on every CUDA device before placing layers. This is
    /// the cuBLAS handle/workspace + chunked-prefill activation peak that isn't
    /// in `model_size_bytes` - without it a model whose weights "just fit" one
    /// card packs all layers there, then OOMs at the first request when cuBLAS
    /// initialises (the idle second GPU never gets used). Reserving up front
    /// makes the pack-first check honest and spills onto the next GPU instead.
    pub fn calculate_with_kv_reserve(
        total_layers: usize,
        model_size_bytes: u64,
        cuda_devices: &[(usize, u64)],
        wgpu_devices: &[(usize, u64)],
        dequant_expansion_wgpu: f64,
        kv_bytes_per_layer: u64,
        gpu_runtime_reserve_bytes: u64,
    ) -> Self {
        // Every device with TWO figures: the budget the fill works against, and the
        // CAPACITY the card actually has. They differ by the flat runtime reserve alone,
        // and that difference is the whole distance between a model on the cards and a
        // model on the host - see the relaxation pass at the end of this function.
        let mut devices: Vec<(DeviceKind, u64, u64)> = Vec::new();

        // CUDA devices use quantized weights (expansion = 1.0). Subtract the
        // runtime reserve from each card's budget so neither the pack-first
        // check nor the greedy fill commits a GPU's last few hundred MB that
        // cuBLAS + the prefill activation will need at runtime.
        for (idx, free_mem) in cuda_devices {
            let budget = free_mem.saturating_sub(gpu_runtime_reserve_bytes);
            info!(
                "  CUDA device {}: {:.1} GB (- {:.1} GB runtime reserve = {:.1} GB usable)",
                idx,
                *free_mem as f64 / 1e9,
                gpu_runtime_reserve_bytes as f64 / 1e9,
                budget as f64 / 1e9
            );
            devices.push((DeviceKind::Cuda(*idx), budget, *free_mem));
        }

        // OpenCL devices use dequantized weights (expansion = ~3.5)
        for (idx, free_mem) in wgpu_devices {
            // Effective memory reduced by dequant expansion
            let effective_mem = (*free_mem as f64 / dequant_expansion_wgpu) as u64;
            info!(
                "  OpenCL device {}: {} GB (effective {} GB after dequant expansion)",
                idx,
                *free_mem as f64 / 1e9,
                effective_mem as f64 / 1e9
            );
            devices.push((DeviceKind::OpenCL(*idx), effective_mem, effective_mem));
        }

        // Sort by performance priority: CUDA first (fastest class), then OpenCL (slower).
        // WITHIN a kind the caller's order is PRESERVED: every fleet call site builds its
        // list from `device_probe::probe_cuda_devices`, which ranks by measured compute
        // throughput (cores x clock) - so the greedy fill packs the fastest card first on
        // ANY topology. Re-sorting by device index here (the old behaviour) silently
        // assumed index order == speed order, which only holds by convention. Free memory
        // is intentionally not an ordering signal (banned heuristic): a card without room
        // simply receives fewer/no layers from the budget fill below.
        devices.sort_by_key(|(k, _, _)| match k {
            DeviceKind::Cuda(_) => 0,
            DeviceKind::OpenCL(_) => 1,
            DeviceKind::Cpu => 2,
        });

        let weights_per_layer = if total_layers > 0 {
            model_size_bytes / total_layers as u64
        } else {
            0
        };
        // Per-layer cost on a GPU = weights + KV cache. CPU layers don't
        // pay the KV cost here (their KV is system RAM, not budgeted).
        let bytes_per_layer = weights_per_layer.saturating_add(kv_bytes_per_layer);
        info!(
            "  Model: {} GB total, ~{} MB/layer (weights {} MB + kv {} MB), {} layers",
            model_size_bytes as f64 / 1e9,
            bytes_per_layer as f64 / 1e6,
            weights_per_layer as f64 / 1e6,
            kv_bytes_per_layer as f64 / 1e6,
            total_layers
        );

        // Pack-first: if the best single device's budget holds the whole
        // model PLUS its KV cache, place all layers there. Splitting across
        // devices introduces cross-device transfers, so avoid it when one
        // GPU can hold everything safely.
        let total_kv_bytes = (total_layers as u64).saturating_mul(kv_bytes_per_layer);
        let single_gpu_required = model_size_bytes.saturating_add(total_kv_bytes);
        if let Some(&(best_kind, best_mem, _)) = devices.first() {
            if single_gpu_required > 0 && best_mem >= single_gpu_required {
                info!("  Pack-first: {:?} has {:.1} GB budget >= model {:.1} GB, placing all {} layers on one device",
                    best_kind, best_mem as f64 / 1e9,
                    model_size_bytes as f64 / 1e9, total_layers);
                let segments = vec![HeteroSegment {
                    kind: best_kind,
                    layer_start: 0,
                    layer_end: total_layers,
                    free_memory_bytes: best_mem,
                }];
                info!("HeteroPlan: 1 segment: {}:0-{}", best_kind, total_layers);
                return HeteroPlan {
                    segments,
                    total_layers,
                };
            }
        }

        // Greedy allocation: fill fastest device first, overflow to next.
        // Counted per device rather than emitted as segments straight away, because two
        // passes below still move blocks between devices and a layer range can only be
        // written once the counts are final.
        let mut counts: Vec<usize> = vec![0; devices.len()];
        let mut placed = 0usize;

        for (i, (device_kind, device_memory, _)) in devices.iter().enumerate() {
            if placed >= total_layers {
                break;
            }

            // `device_memory` is already the caller's chosen budget (they applied
            // `max_gpu_memory_fraction` upstream). Trust it - applying a second
            // safety fraction here would silently tighten the budget and starve
            // the fastest GPU of layers.
            let max_layers = if bytes_per_layer > 0 {
                (device_memory / bytes_per_layer) as usize
            } else {
                total_layers
            };
            // A device whose budget holds ZERO layers gets zero - never force one on it. The
            // old `.max(1)` here put a layer on a card that could not hold it "to make
            // progress", which is exactly how an 11.5 GB single-segment model got planned onto
            // a 4 GB-free card and OOMed at the weight upload. Devices that fit nothing are
            // skipped; whatever remains after the loop spills to CPU (the honest floor).
            let layers_for_device = max_layers.min(total_layers - placed);
            if layers_for_device == 0 {
                info!(
                    "  {:?}: {} GB VRAM -> max 0 layers, skipped",
                    device_kind,
                    *device_memory as f64 / 1e9
                );
                continue;
            }

            info!(
                "  {:?}: {} GB VRAM -> max {} layers, assigned {} (layers {}-{})",
                device_kind,
                *device_memory as f64 / 1e9,
                max_layers,
                layers_for_device,
                placed,
                placed + layers_for_device
            );

            counts[i] = layers_for_device;
            placed += layers_for_device;
        }

        // Even out the CUDA segments before anything else looks at them.
        //
        // The greedy fill packs the fastest card to its budget, and the budget is already
        // free-memory MINUS the activation reserve - so the card that ends up holding most
        // of the model is left with exactly the reserve and not one byte more. Any
        // under-estimate of a forward's peak is then an out-of-memory, and an estimate of a
        // diffusion forward is a model of several transient buffers, not a measurement.
        // Seen directly: a 14B video denoiser planned 33 blocks onto the first card and
        // died on a cast, twice, while the second card sat with gigabytes free.
        //
        // Redistributing the SAME layers across the SAME cards in proportion to their
        // budgets changes nothing about which devices are used or how many layers reach the
        // GPU - only that each card ends up equally full, so the headroom is shared instead
        // of all of it landing on the card with the least work to do. It is also the better
        // shape for the transfer at the boundary, which happens once wherever the split is.
        {
            let cuda: Vec<usize> = devices
                .iter()
                .enumerate()
                .filter(|(i, (k, _, _))| counts[*i] > 0 && matches!(k, DeviceKind::Cuda(_)))
                .map(|(i, _)| i)
                .collect();
            if cuda.len() >= 2 {
                let total: usize = cuda.iter().map(|i| counts[*i]).sum();
                let budget_sum: u128 = cuda.iter().map(|i| devices[*i].1 as u128).sum();
                if budget_sum > 0 && total > 0 {
                    let mut assigned = 0usize;
                    for (n, i) in cuda.iter().enumerate() {
                        let share = if n + 1 == cuda.len() {
                            total - assigned
                        } else {
                            let want =
                                (total as u128 * devices[*i].1 as u128 / budget_sum) as usize;
                            // Never leave a card empty and never overrun what is left.
                            want.clamp(1, total - assigned - (cuda.len() - n - 1))
                        };
                        counts[*i] = share;
                        assigned += share;
                    }
                    // AND NEVER MORE THAN A CARD WAS BUDGETED FOR.
                    //
                    // A proportional share is a RATIO applied to a whole number of blocks, and
                    // the rounding does not know what a card holds: two cards of 5 GB and 2 GB
                    // handed thirty blocks of a twelve gigabyte model come out 5/13, which puts
                    // 5.2 GB on the smaller one. The greedy fill above had it right at 5/8 and
                    // this pass undid it - an over-commitment invisible until the weights are
                    // uploaded, and then an out-of-memory in the loader.
                    //
                    // So the share is capped at what the card's own budget holds and the
                    // remainder is offered to the cards that still have room, fastest first.
                    // A feasible arrangement always exists, because the blocks being divided
                    // are exactly the ones the budgets accepted a moment ago. Whatever cannot
                    // be placed after that is genuinely unplaced: `placed` is recounted so the
                    // relaxation below - which may dip into the runtime reserve - and the host
                    // spill after it both see the truth.
                    if bytes_per_layer > 0 {
                        let room = |i: usize| (devices[i].1 / bytes_per_layer) as usize;
                        let mut excess = 0usize;
                        for i in &cuda {
                            let r = room(*i);
                            if counts[*i] > r {
                                excess += counts[*i] - r;
                                counts[*i] = r;
                            }
                        }
                        for i in &cuda {
                            if excess == 0 {
                                break;
                            }
                            let take = room(*i).saturating_sub(counts[*i]).min(excess);
                            counts[*i] += take;
                            excess -= take;
                        }
                        placed = counts.iter().sum();
                    }
                    info!(
                        "  Balanced across {} CUDA devices: {}",
                        cuda.len(),
                        cuda.iter()
                            .map(|i| format!("{}:{}", devices[*i].0, counts[*i]))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
            }
        }

        // THE HOST IS THE LAST RESORT, AND A RESERVE IS NOT A REASON TO REACH IT.
        //
        // The fill above works against `free - runtime reserve`, and that reserve is
        // charged to EVERY card, because any of them may be the one running a block. On a
        // machine with several cards that is the reserve counted once per card while a
        // forward only ever needs it once: two 16 GB cards asked to keep 7 GB free each
        // have 18 GB of budget between them for a 32 GB fleet, and a 12.3 GB model then
        // misses by fifty megabytes and puts a block on the PROCESSOR - with both cards
        // holding gigabytes.
        //
        // A block on the host is not a slower placement, it is a request that never
        // finishes: minutes per step, no error, the client gives up. Against that, a card
        // dipping into its own headroom is the better answer - and if the dip is too deep
        // the result is an out-of-memory, which is visible, attributable, and already has
        // a re-plan cascade behind it.
        //
        // So the last blocks are pushed back onto the cards, fastest first, up to what
        // each card PHYSICALLY has. Only what no card can hold at all reaches the host,
        // which makes the rule exactly: the sum of the cards decides, not the sum of the
        // budgets. The per-layer cost (a KV cache) is NOT relaxed here - it is a real
        // allocation that grows with the blocks placed, so it stays inside
        // `bytes_per_layer` and bounds this pass too. Only the flat reserve gives way,
        // and callers that pass none (the probe already removed it) see no change at all.
        if placed < total_layers && bytes_per_layer > 0 {
            for (i, (kind, _, capacity)) in devices.iter().enumerate() {
                if placed >= total_layers {
                    break;
                }
                let room = (capacity / bytes_per_layer) as usize;
                let extra = room.saturating_sub(counts[i]).min(total_layers - placed);
                if extra == 0 {
                    continue;
                }
                info!(
                    "  {}: {} more layer(s) taken out of the {:.1} GB runtime reserve rather \
                     than sent to the host",
                    kind,
                    extra,
                    gpu_runtime_reserve_bytes as f64 / 1e9,
                );
                counts[i] += extra;
                placed += extra;
            }
        }

        // Counts are final: lay them out as contiguous layer ranges, in device order.
        let mut segments = Vec::new();
        let mut layer_idx = 0usize;
        for (i, (kind, budget, _)) in devices.iter().enumerate() {
            if counts[i] == 0 {
                continue;
            }
            segments.push(HeteroSegment {
                kind: *kind,
                layer_start: layer_idx,
                layer_end: layer_idx + counts[i],
                free_memory_bytes: *budget,
            });
            layer_idx += counts[i];
        }

        // Remaining layers -> CPU. Cap at available host RAM minus
        // host_ram_safety_headroom_bytes() so the loader can't
        // transparently OOM the host. Without this, a 70 B-class
        // model spilling 30+ GB to CPU on a 64 GB box silently
        // exhausts swap and freezes the machine.
        if layer_idx < total_layers {
            let cpu_layers = total_layers - layer_idx;
            let cpu_bytes_needed = (cpu_layers as u64).saturating_mul(weights_per_layer);
            let mut host_available = available_host_ram_bytes();
            let mut cpu_budget = host_available.saturating_sub(host_ram_safety_headroom_bytes());
            if cpu_bytes_needed > cpu_budget {
                // RAM pressure: before refusing, ask the host-cache registry to drop idle
                // long-lived caches (umT5 staging, CPU-staged VAEs, ...) - they re-fill on
                // their next use, whereas a refused CPU segment degrades or fails THIS load.
                let shortfall = cpu_bytes_needed - cpu_budget;
                let freed = crate::inference::place::vram_manager::reclaim_host_ram(shortfall);
                if freed > 0 {
                    host_available = available_host_ram_bytes();
                    cpu_budget = host_available.saturating_sub(host_ram_safety_headroom_bytes());
                }
            }
            if cpu_bytes_needed > cpu_budget {
                warn!(
                    "  Host-RAM guard: planned {} CPU layers need {:.1} GB but only {:.1} GB available \
                     (after {} GB safety headroom). The loader will refuse this segment via the \
                     host-RAM budget check.",
                    cpu_layers,
                    cpu_bytes_needed as f64 / 1e9,
                    cpu_budget as f64 / 1e9,
                    host_ram_safety_headroom_bytes() / (1024 * 1024 * 1024),
                );
            }
            segments.push(HeteroSegment {
                kind: DeviceKind::Cpu,
                layer_start: layer_idx,
                layer_end: total_layers,
                // Real CPU budget (not u64::MAX) so callers see the
                // constraint. cpu_budget already subtracts the
                // safety headroom.
                free_memory_bytes: cpu_budget,
            });
        }

        // Log the plan
        info!(
            "HeteroPlan: {} segments: {}",
            segments.len(),
            segments
                .iter()
                .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>()
                .join(", ")
        );

        HeteroPlan {
            segments,
            total_layers,
        }
    }

    /// Build a plan that puts exactly `gpu_layers` on CUDA device `cuda_idx`
    /// and the remainder on CPU, regardless of actual VRAM.
    /// Useful for testing or explicit user overrides via `force_gpu_layers`.
    pub fn forced_gpu(total_layers: usize, gpu_layers: usize, cuda_idx: usize) -> Self {
        let gpu_layers = gpu_layers.min(total_layers);
        let mut segments = Vec::new();
        if gpu_layers > 0 {
            segments.push(HeteroSegment {
                kind: DeviceKind::Cuda(cuda_idx),
                layer_start: 0,
                layer_end: gpu_layers,
                free_memory_bytes: u64::MAX,
            });
        }
        if gpu_layers < total_layers {
            // Even in the forced-placement path, the host can be
            // OOM-ed by an over-eager `--force-gpu-layers` value.
            // Report the same host-RAM budget the planner uses so
            // the downstream loader's free_memory_bytes check fires
            // before the host swaps to disk.
            let host_available = available_host_ram_bytes();
            let cpu_budget = host_available.saturating_sub(host_ram_safety_headroom_bytes());
            segments.push(HeteroSegment {
                kind: DeviceKind::Cpu,
                layer_start: gpu_layers,
                layer_end: total_layers,
                free_memory_bytes: cpu_budget,
            });
        }
        info!(
            "HeteroPlan (forced): {} GPU layers + {} CPU layers",
            gpu_layers,
            total_layers - gpu_layers
        );
        HeteroPlan {
            segments,
            total_layers,
        }
    }

    /// Force layers to spread proportionally across ALL the given CUDA
    /// devices, deliberately bypassing the pack-first single-GPU
    /// collapse. This is the OOM-recovery fallback: when an optimistic
    /// single-GPU placement (weights fit, but the first-prefill
    /// activation peak overflows) raises CUDA_ERROR_OUT_OF_MEMORY, the
    /// loader retries with this split instead of pre-reserving a magic
    /// activation margin in the plan. Layers are apportioned by each
    /// device's free VRAM so the faster/roomier card carries more.
    /// `cuda_devices` is `(index, free_bytes)`, lowest index treated as
    /// primary. Falls back to an even split if all free figures are 0.
    ///
    /// Shares are weighted by each device's COMPUTE THROUGHPUT, not by its free
    /// VRAM. The cards a machine mixes are rarely equal, and the pipeline runs
    /// every token through every segment, so an even split makes the fast card
    /// wait on the slow one at each boundary. Free VRAM says nothing about
    /// speed: two cards of the same capacity can differ by a factor in cores x
    /// clock, and apportioning by capacity hands them identical shares. Where
    /// no throughput figure is available the free-VRAM proportion remains the
    /// fallback.
    pub fn split_across_cuda(total_layers: usize, cuda_devices: &[(usize, u64)]) -> Self {
        let weights: Vec<u64> = {
            #[cfg(feature = "cuda")]
            {
                let probed = crate::inference::place::device_probe::probe_cuda_gpus(1.0);
                cuda_devices
                    .iter()
                    .map(|&(idx, _)| {
                        probed
                            .iter()
                            .find(|g| g.index == idx)
                            .map(|g| g.perf_score)
                            .unwrap_or(0)
                    })
                    .collect()
            }
            #[cfg(not(feature = "cuda"))]
            {
                vec![0; cuda_devices.len()]
            }
        };
        Self::split_across_cuda_weighted(total_layers, cuda_devices, &weights)
    }

    /// Apportioning core of [`split_across_cuda`], split out so the share
    /// arithmetic can be exercised without a GPU present. `weights` is parallel
    /// to `cuda_devices`; all-zero means "no throughput information", which
    /// falls back to the free-VRAM proportion.
    pub fn split_across_cuda_weighted(
        total_layers: usize,
        cuda_devices: &[(usize, u64)],
        weights: &[u64],
    ) -> Self {
        if cuda_devices.is_empty() || total_layers == 0 {
            return HeteroPlan {
                segments: Vec::new(),
                total_layers,
            };
        }
        let mut devs: Vec<(usize, u64, u64)> = cuda_devices
            .iter()
            .enumerate()
            .map(|(i, &(idx, free))| (idx, free, weights.get(i).copied().unwrap_or(0)))
            .collect();
        devs.sort_by_key(|&(idx, _, _)| idx);
        let total_weight: u64 = devs.iter().map(|&(_, _, w)| w).sum();
        // No throughput figures: keep the previous capacity-proportional split.
        let devs: Vec<(usize, u64)> = if total_weight > 0 {
            devs.iter().map(|&(idx, _, w)| (idx, w)).collect()
        } else {
            devs.iter().map(|&(idx, f, _)| (idx, f)).collect()
        };
        // The share each device gets; `free` below stays the real VRAM figure,
        // which the segment carries for the loader regardless of what drove the
        // apportioning.
        let shares: Vec<(usize, u64)> = devs;
        let total_share: u64 = shares.iter().map(|&(_, s)| s).sum();
        let free_of = |idx: usize| {
            cuda_devices
                .iter()
                .find(|&&(i, _)| i == idx)
                .map(|&(_, f)| f)
                .unwrap_or(0)
        };
        let mut segments = Vec::new();
        let mut assigned = 0usize;
        for (i, &(idx, share)) in shares.iter().enumerate() {
            if assigned >= total_layers {
                break;
            }
            let remaining = total_layers - assigned;
            let n = if i == shares.len() - 1 {
                // Last device takes the remainder (no rounding orphans).
                remaining
            } else if total_share > 0 {
                // Round to nearest: truncating costs the fastest card a layer
                // on almost every ratio, which is the share that mattered most.
                let num = (total_layers as u128) * (share as u128) + (total_share as u128) / 2;
                ((num / (total_share as u128)) as usize)
                    .max(1)
                    .min(remaining)
            } else {
                // Neither throughput nor capacity known: even split.
                (total_layers / shares.len()).max(1).min(remaining)
            };
            segments.push(HeteroSegment {
                kind: DeviceKind::Cuda(idx),
                layer_start: assigned,
                layer_end: assigned + n,
                free_memory_bytes: free_of(idx),
            });
            assigned += n;
        }
        info!(
            "HeteroPlan (forced CUDA split): {} layers across {} GPUs: {}",
            total_layers,
            segments.len(),
            segments
                .iter()
                .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>()
                .join(", ")
        );
        HeteroPlan {
            segments,
            total_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gb(n: u64) -> u64 {
        n * 1024 * 1024 * 1024
    }

    #[test]
    fn forced_split_weights_shares_by_throughput_not_capacity() {
        // Two cards of the SAME capacity but different speed: capacity would
        // hand them equal shares and the fast card would idle at every layer
        // boundary waiting on the slow one.
        let plan = HeteroPlan::split_across_cuda_weighted(
            40,
            &[(0, gb(16)), (1, gb(16))],
            &[22_240_000, 12_030_000], // ~1.85x apart
        );
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[0].num_layers(), 26);
        assert_eq!(plan.segments[1].num_layers(), 14);
        // The segment still reports the device's real VRAM, not its weight.
        assert_eq!(plan.segments[0].free_memory_bytes, gb(16));
    }

    #[test]
    fn forced_split_falls_back_to_capacity_without_throughput() {
        let plan = HeteroPlan::split_across_cuda_weighted(40, &[(0, gb(24)), (1, gb(8))], &[0, 0]);
        assert_eq!(plan.segments[0].num_layers(), 30);
        assert_eq!(plan.segments[1].num_layers(), 10);
    }

    #[test]
    fn forced_split_covers_every_layer_exactly_once() {
        for w in [[1u64, 1], [50, 1], [1, 50]] {
            let plan = HeteroPlan::split_across_cuda_weighted(40, &[(0, gb(16)), (1, gb(16))], &w);
            assert_eq!(plan.segments[0].layer_start, 0);
            assert_eq!(plan.segments.last().unwrap().layer_end, 40);
            let total: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
            assert_eq!(total, 40, "weights {w:?} lost or duplicated layers");
        }
    }

    #[test]
    fn pack_first_collapses_to_single_device_when_it_fits() {
        // Classic LLM-style call: model fits inside one GPU's budget,
        // so pack-first puts all layers there to avoid the cross-device
        // transfer cost.
        let plan = HeteroPlan::calculate(
            32,
            gb(10),                      // 10 GB model
            &[(0, gb(15)), (1, gb(15))], // both GPUs have 15 GB
            &[],
            1.0,
        );
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].kind, DeviceKind::Cuda(0));
        assert_eq!(plan.segments[0].num_layers(), 32);
    }

    #[test]
    fn pack_first_skipped_when_kv_cost_pushes_total_past_single_device() {
        // Z-Image regression case (3e3208b): a 12 GB model fits the
        // 13 GB single-GPU budget, but the per-step activation peak
        // (modeled as kv_bytes_per_layer) pushes the total need past
        // any single device. Plan must split across both GPUs.
        let kv = gb(6) / 30; // 6 GB activation peak amortised over 30 layers
        let plan =
            HeteroPlan::calculate_with_kv(30, gb(12), &[(0, gb(13)), (1, gb(13))], &[], 1.0, kv);
        assert!(
            plan.segments.len() >= 2,
            "expected split, got {} segments",
            plan.segments.len()
        );
        // Both segments must be on CUDA, not CPU.
        let kinds: Vec<_> = plan.segments.iter().map(|s| s.kind).collect();
        assert!(matches!(kinds[0], DeviceKind::Cuda(_)));
        assert!(matches!(kinds[1], DeviceKind::Cuda(_)));
        // Total layers preserved (no orphans).
        assert_eq!(plan.total_layers, 30);
        let assigned: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
        assert_eq!(assigned, 30);
    }

    #[test]
    fn multi_gpu_distributes_proportionally_when_one_does_not_fit() {
        // Model too big for any single GPU -> split. Both have equal
        // budget so layers should split roughly 50/50.
        let plan = HeteroPlan::calculate(
            30,
            gb(20), // 20 GB model
            &[(0, gb(12)), (1, gb(12))],
            &[],
            1.0,
        );
        let cuda_seg_count = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .count();
        assert_eq!(cuda_seg_count, 2, "should use both CUDA devices");
        // Each segment should be roughly half of total layers (±2).
        for seg in &plan.segments {
            if matches!(seg.kind, DeviceKind::Cuda(_)) {
                let n = seg.num_layers();
                assert!((10..=20).contains(&n), "lopsided split: {n} layers");
            }
        }
    }

    #[test]
    fn three_gpus_all_used_when_model_exceeds_any_pair() {
        // N-GPU invariant: the planner is a list algorithm, not a "primary +
        // secondary" pair. A model larger than any two budgets must spread
        // across all three cards with zero CPU spill and no orphaned layers.
        let plan = HeteroPlan::calculate(
            30,
            gb(30),
            &[(0, gb(12)), (1, gb(12)), (2, gb(12))],
            &[],
            1.0,
        );
        let cuda_segs: Vec<_> = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .collect();
        assert_eq!(
            cuda_segs.len(),
            3,
            "all three GPUs must carry layers: {:?}",
            plan.segments.iter().map(|s| s.kind).collect::<Vec<_>>()
        );
        assert!(
            !plan
                .segments
                .iter()
                .any(|s| matches!(s.kind, DeviceKind::Cpu)),
            "no CPU spill expected with 36 GB of GPU budget for a 30 GB model"
        );
        let assigned: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
        assert_eq!(assigned, 30);
        // Caller order preserved: the budget list is perf-ranked by the caller,
        // segments must follow it, not re-sort by index or free bytes.
        let order: Vec<_> = cuda_segs.iter().map(|s| s.kind).collect();
        assert_eq!(
            order,
            vec![
                DeviceKind::Cuda(0),
                DeviceKind::Cuda(1),
                DeviceKind::Cuda(2)
            ]
        );
    }

    #[test]
    fn zero_fit_device_skipped_in_larger_fleet() {
        // Four cards, one with a budget too small for a single layer: the
        // greedy fill must SKIP it (no forced 1-layer segment) and the other
        // three must absorb every layer.
        let plan = HeteroPlan::calculate(
            30,
            gb(30),
            &[(0, gb(12)), (1, gb(1) / 2), (2, gb(12)), (3, gb(12))],
            &[],
            1.0,
        );
        assert!(
            !plan.segments.iter().any(|s| s.kind == DeviceKind::Cuda(1)),
            "zero-fit device must be skipped: {:?}",
            plan.segments.iter().map(|s| s.kind).collect::<Vec<_>>()
        );
        let assigned: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
        assert_eq!(assigned, 30);
        assert!(
            !plan
                .segments
                .iter()
                .any(|s| matches!(s.kind, DeviceKind::Cpu)),
            "remaining three cards hold 36 GB for a 30 GB model - no CPU spill"
        );
    }

    #[test]
    fn z_image_user_2x17gb_setup_balances_layers_with_vae_headroom() {
        // Pins the placement against a TWO-CARD host whose cards differ in
        // speed and in usable budget after headroom - the case the packing
        // rule exists for. Iteration history baked into the test docstring so
        // a regression is self-explaining:
        //
        // - Before: 5 GB per_device + 9 GB inflate. Placed
        //   8 + 16 + 6_CPU. Slow due to PCIe-bounced CPU layers.
        //   User: "Image generation has become extremely slow".
        // - Attempt 1: zero inflate. Placed 26 + 4 + 0.
        //   GPU0 filled to 15.7 GB at gen (92% of free), VAE decode
        //   then OOMed needing 1.5 GB on 1.1 GB free.
        //   User: "Image generation error: DriverError(CUDA_OOM)".
        // - Attempt 2 (this test): 5 GB per_device + 6 GB
        //   inflate (200 MB/layer scaling). Places ~17 + 13. GPU0
        //   uses ~14.5 GB at gen -> 2.3 GB margin for VAE decode.
        //   All on GPU AND fits.
        //
        // The 6 GB inflate models per-layer NON-WEIGHT cost
        // (residual buffers + allocator fragmentation + activation
        // pressure that scales with layers placed) at ~200 MB/layer
        // measured empirically from the OOM gen log.
        let cuda = &[(0, 10_600_000_000u64), (1, 11_400_000_000u64)];
        // 6 GB total inflate -> 200 MB/layer. NO LONGER what the Z-Image loader does:
        // it charged its runtime reserve per card AND again per layer, so the card
        // holding most of the model paid it over and over, and a demand learned from an
        // exhaustion put sixteen of thirty blocks on the host. The reserve is charged
        // once now, via `calculate_with_kv_reserve`. This case stays because the
        // PLANNER's per-layer accounting is still used by callers with a genuine
        // per-layer cost (a KV cache), and it is that arithmetic under test here.
        let kv = 6_000_000_000u64 / 30;
        let plan = HeteroPlan::calculate_with_kv(30, gb(12), cuda, &[], 1.0, kv);

        let cuda_segs: Vec<_> = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .collect();
        let cpu_segs: Vec<_> = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cpu))
            .collect();

        // CRITICAL CONSTRAINT 1: zero CPU spill. Combined post-headroom
        // budget (~22 GB) exceeds 12 GB weights + per-layer scaling
        // (~6 GB) = ~18 GB needed. All on GPU.
        assert_eq!(
            cpu_segs.len(),
            0,
            "regression: layers spilled to CPU; segments={:?}",
            plan.segments.iter().map(|s| s.kind).collect::<Vec<_>>()
        );

        // CRITICAL CONSTRAINT 2: VAE decode safety. The most-pressured
        // GPU must hold <= 18 layers - empirically, 18 layers at gen
        // uses ~13.7 GB on GPU0 (400 MB weights + 261 MB non-weight
        // measured = 661 MB/layer x 18 = 11.9 GB + 1.8 GB workspace
        // + embed). Plus ~1.5 GB VAE decode peak = 15.2 GB. On a
        // 16.8 GB free card that's 1.6 GB margin - safe under
        // allocator fragmentation. Cap at 18 to lock this in.
        for seg in &cuda_segs {
            assert!(
                seg.num_layers() <= 18,
                "GPU {:?} got {} layers; VAE decode would OOM (need <= 18 \
                 to leave ~1.5 GB headroom for VAE peak on 17 GB cards)",
                seg.kind,
                seg.num_layers()
            );
        }

        // Every layer placed (no orphans).
        let assigned: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
        assert_eq!(assigned, 30, "all 30 layers must be placed");
    }

    #[test]
    fn z_image_single_24gb_gpu_packs_all_layers() {
        // Single card with room to spare. Pack-first
        // should put all 30 layers on the one GPU - no PCIe traffic,
        // maximum perf when one device can safely hold the whole
        // model + activations + VAE.
        // 22 GB free - 5 GB per_device_act_budget = 17 GB budget.
        // 17 GB / 600 MB per_layer (weights+inflate) = 28 layers fit
        // by the budget, but we need ALL 30 -> relies on pack-first.
        // Pack-first triggers when single device fits weights AND
        // total_inflate (6 GB) -> needs 18 GB; 17 GB budget exceeded.
        // Hmm - pack-first won't engage. Test what actually happens.
        let cuda = &[(0, 17_000_000_000u64)];
        let kv = 6_000_000_000u64 / 30;
        let plan = HeteroPlan::calculate_with_kv(30, gb(12), cuda, &[], 1.0, kv);

        // With 17 GB budget and 600 MB/layer effective, max 28 layers
        // on the single GPU; 2 spill to CPU. That's actually the
        // SAFE answer - 28 layers x 661 MB real cost = 18.5 GB
        // would OOM on a card with 22 GB free + VAE decode pressure.
        // Pin that the plan distributes safely.
        let gpu_layers: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .map(|s| s.num_layers())
            .sum();
        assert!(
            gpu_layers >= 27 && gpu_layers <= 30,
            "expected most layers on GPU, got {gpu_layers}"
        );
        let assigned: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
        assert_eq!(assigned, 30);
    }

    #[test]
    fn z_image_tight_vram_still_spills_to_cpu_safely() {
        // Worst case: single 12 GB GPU under heavy contention, only
        // 4 GB free after budget. Half the model can't fit - must
        // spill to CPU cleanly without panic / orphan layers.
        let cuda = &[(0, 4_000_000_000u64)];
        let kv = 6_000_000_000u64 / 30;
        let plan = HeteroPlan::calculate_with_kv(30, gb(12), cuda, &[], 1.0, kv);

        // 4 GB / 600 MB per_layer = 6 layers on GPU, rest on CPU.
        let gpu_layers: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(_)))
            .map(|s| s.num_layers())
            .sum();
        let cpu_layers: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum();
        assert_eq!(gpu_layers + cpu_layers, 30);
        assert!(
            gpu_layers >= 5 && gpu_layers <= 8,
            "expected ~6 layers on GPU, got {gpu_layers}"
        );
        assert!(cpu_layers > 0, "expected CPU spill, got none");
    }

    #[test]
    fn cpu_overflow_when_gpus_full() {
        // 30 GB model, only 10 GB of GPU memory total -> most layers
        // must spill to CPU. Locks in that the planner doesn't silently
        // truncate or panic when there isn't enough GPU.
        let plan = HeteroPlan::calculate(40, gb(30), &[(0, gb(10))], &[], 1.0);
        let cpu_layers: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum();
        assert!(cpu_layers > 0, "expected CPU overflow, got none");
        let assigned: usize = plan.segments.iter().map(|s| s.num_layers()).sum();
        assert_eq!(assigned, 40, "must place every layer somewhere");
    }

    #[test]
    fn cpu_overflow_capped_by_host_ram_not_u64_max() {
        // Host-RAM guard: even when CPU is the only place left for
        // layers, the segment must surface a real budget so the
        // downstream loader can refuse over-spill instead of letting
        // the kernel swap to disk. Locks in that:
        //   1. The CPU segment's free_memory_bytes is < u64::MAX.
        //   2. It does not exceed (host_available - safety_headroom).
        //
        // Snapshot host_available BEFORE the calculate() - otherwise
        // concurrent test allocations between calculate() and the
        // assertion-time reading can decrease the right side and
        // trip a spurious failure when run as part of the full suite.
        let host_available_pre = available_host_ram_bytes();
        let plan = HeteroPlan::calculate(40, gb(30), &[(0, gb(10))], &[], 1.0);
        let cpu_seg = plan
            .segments
            .iter()
            .find(|s| matches!(s.kind, DeviceKind::Cpu))
            .expect("expected CPU spill segment");
        assert!(
            cpu_seg.free_memory_bytes < u64::MAX,
            "CPU segment must carry a real RAM budget, not u64::MAX"
        );
        // Tolerance: calculate()'s internal reading can drift from the
        // pre-snapshot by up to a few hundred MB even on a quiet system.
        // 1 GB slack absorbs that without weakening the guard meaningfully
        // (the safety headroom is 8 GB).
        let max_allowed = host_available_pre.saturating_sub(host_ram_safety_headroom_bytes());
        const SLACK: u64 = 1024 * 1024 * 1024;
        assert!(
            cpu_seg.free_memory_bytes <= max_allowed.saturating_add(SLACK),
            "CPU budget {} must not exceed (pre-snapshot available {} - headroom {} + slack {})",
            cpu_seg.free_memory_bytes,
            host_available_pre,
            host_ram_safety_headroom_bytes(),
            SLACK,
        );
    }

    #[test]
    fn forced_gpu_cpu_segment_also_capped_by_host_ram() {
        // The forced-placement path (--force-gpu-layers) must apply
        // the same guard; an over-eager user value otherwise drives
        // the host into swap.
        let plan = HeteroPlan::forced_gpu(40, 10, 0);
        let cpu_seg = plan
            .segments
            .iter()
            .find(|s| matches!(s.kind, DeviceKind::Cpu))
            .expect("forced_gpu(40,10) must produce a CPU segment");
        assert!(
            cpu_seg.free_memory_bytes < u64::MAX,
            "forced_gpu CPU segment must carry a real RAM budget"
        );
    }

    #[test]
    fn forced_gpu_places_exactly_what_was_asked() {
        let plan = HeteroPlan::forced_gpu(40, 25, 0);
        assert_eq!(plan.total_layers, 40);
        let cuda_layers: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(0)))
            .map(|s| s.num_layers())
            .sum();
        assert_eq!(cuda_layers, 25);
        // Forced plans bypass VRAM bookkeeping -> memory marked as u64::MAX
        // so callers don't accidentally feed it back as a real budget.
        let cuda_seg = plan
            .segments
            .iter()
            .find(|s| matches!(s.kind, DeviceKind::Cuda(0)))
            .unwrap();
        assert_eq!(cuda_seg.free_memory_bytes, u64::MAX);
    }

    /// THE DEFECT. A per-card reserve is not a reason to put a block on the processor.
    ///
    /// The reported render, in the figures the loader hands the planner: two 16.4 GB
    /// cards, a 7.02 GB forward, and a 10.63 GB block stack. The first card has already
    /// been charged 1.68 GB for the stem it carries beside its blocks, the second 6.5 GB
    /// for the caption encoder - and then the reserve comes off BOTH, which leaves 10.58
    /// GB of budget for 10.63 GB of blocks. Fifty megabytes short of thirty, so twenty
    /// nine went to the cards and ONE to the host - where a diffusion step takes minutes
    /// and the render simply never arrives.
    #[test]
    fn a_reserve_charged_to_every_card_never_sends_a_block_to_the_host() {
        let cards = [(0, 14_717_000_000u64), (1, 9_900_000_000u64)];
        let stack = 10_627_245_040u64;
        let plan =
            HeteroPlan::calculate_with_kv_reserve(30, stack, &cards, &[], 1.0, 0, 7_020_000_000);
        assert!(
            !plan
                .segments
                .iter()
                .any(|s| matches!(s.kind, DeviceKind::Cpu)),
            "24.6 GB of cards for a 10.6 GB stack and a block still went to the host: {:?}",
            plan.segments
                .iter()
                .map(|s| (s.kind, s.layer_start, s.layer_end))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            plan.segments.iter().map(|s| s.num_layers()).sum::<usize>(),
            30
        );
        // And each card must hold what it was given: the relaxation dips into the
        // reserve, it does not overrun the card.
        let per_layer = stack / 30;
        for s in &plan.segments {
            let free = cards
                .iter()
                .find(|(i, _)| s.kind == DeviceKind::Cuda(*i))
                .unwrap()
                .1;
            assert!(
                s.num_layers() as u64 * per_layer <= free,
                "{:?} was given {} layers, more than the card holds",
                s.kind,
                s.num_layers()
            );
        }
    }

    /// The relaxation stops at what the cards PHYSICALLY hold - it is not "ignore the
    /// reserve", it is "the sum of the cards decides". Without this the fix above would
    /// be a planner that places 30 GB of weights on 10 GB of silicon.
    #[test]
    fn the_host_still_takes_what_no_card_can_hold() {
        let plan = HeteroPlan::calculate_with_kv_reserve(
            30,
            gb(30),
            &[(0, gb(5)), (1, gb(5))],
            &[],
            1.0,
            0,
            gb(2),
        );
        let cpu: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum();
        assert!(cpu > 0, "10 GB of cards cannot hold a 30 GB model");
        let gpu: usize = plan
            .segments
            .iter()
            .filter(|s| !matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum();
        // 10 GB of cards at 1 GB a layer: ten blocks, and not one more.
        assert_eq!(gpu, 10, "the cards took {gpu} of 30 blocks");
        assert_eq!(
            plan.segments.iter().map(|s| s.num_layers()).sum::<usize>(),
            30
        );
    }

    /// A per-LAYER cost is a real allocation and is NOT relaxed: it grows with the blocks
    /// placed, so a card that would OOM on the KV cache of the blocks it was handed must
    /// still spill. Only the flat reserve gives way.
    #[test]
    fn the_relaxation_leaves_the_per_layer_cost_alone() {
        // One card, 10 GB free, no flat reserve at all: 500 MB of weights + 500 MB of KV
        // per layer means ten blocks fit and twenty cannot, reserve or no reserve.
        let plan = HeteroPlan::calculate_with_kv_reserve(
            30,
            30 * 500_000_000,
            &[(0, 10_000_000_000)],
            &[],
            1.0,
            500_000_000,
            0,
        );
        let gpu: usize = plan
            .segments
            .iter()
            .filter(|s| !matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum();
        assert_eq!(gpu, 10, "the KV cache must bound the fill, not the reserve");
    }

    /// Callers whose probe already removed the reserve pass none here, and for them the
    /// relaxation cannot fire at all - their budget IS the card. Pins that the change is
    /// scoped to the sites that hand the planner a flat reserve.
    #[test]
    fn a_plan_without_a_flat_reserve_is_unchanged() {
        let plan = HeteroPlan::calculate_with_kv(
            30,
            gb(12),
            &[(0, 4_000_000_000), (1, 4_000_000_000)],
            &[],
            1.0,
            0,
        );
        // 8 GB of budget, 12 GB of weights: 9 blocks a card, 12 to the host - exactly
        // what the greedy fill said before the relaxation existed.
        let cpu: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cpu))
            .map(|s| s.num_layers())
            .sum();
        assert_eq!(cpu, 12);
    }

    #[test]
    fn forced_gpu_caps_at_total_layers() {
        // gpu_layers > total should clamp, not overflow.
        let plan = HeteroPlan::forced_gpu(10, 9999, 0);
        let cuda_layers: usize = plan
            .segments
            .iter()
            .filter(|s| matches!(s.kind, DeviceKind::Cuda(0)))
            .map(|s| s.num_layers())
            .sum();
        assert_eq!(cuda_layers, 10);
        // No CPU segment when GPU absorbs everything.
        assert!(!plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::Cpu)));
    }
}

#[cfg(test)]
mod reserve_double_subtraction_gate {
    /// The reserve must be subtracted ONCE.
    ///
    /// Every placement site probes with `probe_under_pressure(reserve)` or
    /// `probe(reserve)`, both of which return `stable_free - reserve` already, and then
    /// hands the budgets to `calculate_with_kv_reserve`. Passing the reserve there too
    /// removes it a second time. At half a gigabyte that merely wastes a little; at
    /// twelve - a thirty-second video clip's activation demand - it took two 16 GB cards
    /// to ZERO usable and sent the whole DiT to the host, which is how "multi-GPU is not
    /// possible" looked from the outside.
    ///
    /// A source gate because the shape is invisible at either call site on its own: each
    /// one reads as correct, and only the pair is wrong. It looked only at files naming
    /// `probe_under_pressure` and so never read `hetero_place`, which probes by the other
    /// name and had been subtracting its reserve twice from every TTS placement since it
    /// was written. It reads every file under `src/inference` now.
    #[test]
    fn no_site_passes_a_reserve_it_has_already_subtracted() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/inference");
        let mut files = Vec::new();
        fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        walk(&root, &mut files);
        // Not the placement GATE module: it is compiled only under test, and it holds a
        // table of the probe and planner names as strings so it can search the tree for
        // them. Reading it here made the mere mention of `probe_under_pressure` in that
        // table turn its own fixtures into offenders.
        files.retain(|f| !f.ends_with("placement_invariants.rs"));
        let mut offenders = Vec::new();
        for f in &files {
            let Ok(text) = std::fs::read_to_string(f) else {
                continue;
            };
            let names_the_net_probe = text.contains("probe_under_pressure");
            // The call is written across lines; join the statement before matching, and
            // drop the line comments first - a sentence ABOUT a probe is not a call to
            // one, and a stray bracket in prose used to end the argument list early and
            // hand this gate half a comment to reason about.
            let flat: String = text
                .lines()
                .map(|l| match l.find("//") {
                    Some(p) => &l[..p],
                    None => l,
                })
                .collect::<Vec<_>>()
                .join(" ");
            for (i, part) in flat.match_indices("calculate_with_kv_reserve(") {
                let tail: String = flat[i..].chars().take(400).collect();
                let Some(end) = tail.find(')') else { continue };
                let args = &tail[part.len()..end];
                // What the SAME piece of code did just above this call. A probe handed the
                // very identifier that is about to be handed to the planner is the figure
                // twice over - and it has to be the same neighbourhood, because a file can
                // hold two placements, one probing net and one probing at zero, and only
                // the first of them is wrong.
                let start = flat[..i]
                    .char_indices()
                    .rev()
                    .take(600)
                    .last()
                    .map(|(p, _)| p)
                    .unwrap_or(0);
                let before = &flat[start..i];
                // The LAST argument is the reserve.
                if let Some(last) = args.rsplit(',').next() {
                    let last = last.trim();
                    // A LITERAL is the answer this gate wants, so it is never an offender -
                    // and `probe(0)` a few lines up must not read as "the same figure
                    // twice" just because both are written `0`.
                    let is_name = !last.is_empty()
                        && last.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                        && !last.starts_with(|c: char| c.is_ascii_digit());
                    let probed_net = is_name
                        && (before.contains(&format!("probe({last})"))
                            || before.contains(&format!("probe_under_pressure({last})")));
                    // Named, it is the one the probe already removed.
                    let named_reserve =
                        names_the_net_probe && (last.contains("reserve") || last == "act");
                    if named_reserve || probed_net {
                        offenders.push(format!("{}: `{last}`", f.display()));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "a placement passes a reserve to the planner that its probe already removed - \
             pass 0 there, and say so:\n{}",
            offenders.join("\n")
        );
    }
}
