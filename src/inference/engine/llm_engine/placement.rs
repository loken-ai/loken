//! Where a language model's layers go, decided from its header and the devices before any
//! weight is read.
//!
//! The load builds the plan [`plan`] returns.
//! The placement preview reports the same plan without building it.

use super::loading::{beside_the_tower, kv_cache_bytes, kv_cache_format, load_staging, ModelDims};
use super::*;
use crate::inference::place::ladder::Demand;
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::inference::place::preview::{HostShare, PlacementPreview};

/// What the configuration asks of a placement, read once from the engine.
#[derive(Debug, Clone)]
pub(crate) struct Ask {
    /// The model's name: the owner the cards are read for.
    pub(crate) model_id: String,
    /// The card the configuration pins the model to.
    pub(crate) pin: Option<usize>,
    /// No card may take a layer.
    pub(crate) cards_off: bool,
    /// The cache format asked for.
    pub(crate) kv_quant: KvQuant,
    pub(crate) fidelity: Fidelity,
    /// The window asked for, 0 for the model's own.
    pub(crate) context_length: usize,
    /// The layers `[from, to)` of a node's share of a stack.
    pub(crate) layer_share: Option<(usize, usize)>,
    /// What one prefill row was measured to keep live on a card, 0 before any measure.
    pub(crate) prefill_row_bytes: u64,
    /// Bytes each card offers on top of its free memory: what models unloaded before the load
    /// give back there.
    pub(crate) given_back: Vec<(usize, u64)>,
}

impl LlmEngine {
    /// What this engine's configuration asks of a placement.
    pub(crate) fn placement_ask(&self) -> Ask {
        Ask {
            model_id: self.config.model_id.clone(),
            pin: self.config.device_index,
            // No allowed backend drives a card: every layer goes to the host or to the
            // Vulkan and OpenCL devices.
            cards_off: self.config.disable_cuda || !crate::gpu::cards_allowed(),
            kv_quant: self.config.kv_quant,
            fidelity: self.config.fidelity,
            context_length: self.config.context_length,
            layer_share: self.config.layer_share,
            prefill_row_bytes: self
                .prefill_row_bytes
                .load(std::sync::atomic::Ordering::Relaxed),
            given_back: Vec::new(),
        }
    }

    /// The placement a load of this engine's model would build now, without reading a weight.
    ///
    /// The cards are read as a load reads them, under the load admission.
    /// The other models on the cards are not asked for the memory they keep.
    pub async fn placement_preview(&self) -> Result<PlacementPreview, String> {
        self.plan_now(preview_of).await
    }

    /// The plan a load of this engine's model would build now beside the models resident on
    /// the cards, without reading a weight, as [`Self::placement_preview`] reads it.
    pub async fn dry_plan(&self) -> Result<DryPlan, String> {
        self.plan_now(dry_plan_of).await
    }

    /// The plan a load of this engine's model would build once the bytes `given_back` holds
    /// for each card are free there, as [`Self::dry_plan`] reads it.
    ///
    /// It does not wait for a load that runs: it answers an error then.
    pub async fn dry_plan_given(&self, given_back: Vec<(usize, u64)>) -> Result<DryPlan, String> {
        let Some(admission) = crate::inference::place::vram_manager::try_load_admission() else {
            return Err("a load runs".to_string());
        };
        let ask = Ask {
            given_back,
            ..self.placement_ask()
        };
        self.plan_with(ask, dry_plan_of, Some(admission)).await
    }

    /// What `read` reports of the plan a load of this engine's model would build now.
    async fn plan_now<T: Send + 'static>(&self, read: Report<T>) -> Result<T, String> {
        self.plan_with(self.placement_ask(), read, None).await
    }

    /// What `read` reports of the plan a load as `ask` asks would build now.
    /// It runs under `admission`, else under the load admission it waits for.
    async fn plan_with<T: Send + 'static>(
        &self,
        ask: Ask,
        read: Report<T>,
        admission: Option<tokio::sync::MutexGuard<'static, ()>>,
    ) -> Result<T, String> {
        let model_id = self.config.model_id.clone();
        let model_file = self
            .find_model_file(&model_id)
            .ok_or_else(|| format!("model file not found for {model_id}"))?;
        if let Some(why) =
            crate::inference::load::awq_loader::text_route_refusal(&model_file.to_string_lossy())
        {
            return Err(why);
        }
        let model_file =
            discover::composite_text_dir(&model_file, &self.hub_roots()).unwrap_or(model_file);
        if crate::inference::model::clip::pair::claims_checkpoint_dir(&model_file)
            || crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta::claims_checkpoint(&model_file)
        {
            return Err(format!(
                "{model_id} is not placed by layers through the language model route"
            ));
        }
        let file_len = std::fs::metadata(&model_file)
            .map_err(|e| format!("{}: {e}", model_file.display()))?
            .len();
        let _admission = match admission {
            Some(held) => held,
            None => crate::inference::place::vram_manager::load_admission().await,
        };
        tokio::task::spawn_blocking(move || plan_file(&model_file, file_len, &ask, read))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())
    }
}

/// What is read of a layered plan: the opened model, the plan and the configuration's ask.
type Report<T> = fn(&Source<'_>, &Layers, &Ask) -> T;

/// What `read` reports of the plan of `model_file`, `file_len` bytes on disk, placed as `ask`
/// says.
fn plan_file<T>(
    model_file: &std::path::Path,
    file_len: u64,
    ask: &Ask,
    read: Report<T>,
) -> AnyResult<T> {
    let Opened {
        mmap,
        content,
        awq,
        split,
        file_size,
    } = open(model_file, file_len)?;
    let weights = generic_weights(&mmap, &awq, &split);
    let arch = get_gguf_string(&content, "general.architecture").unwrap_or_default();
    let dims = ModelDims::read(&content, &arch);
    let window = window(&content, &arch, &dims, ask);
    let source = Source {
        content: &content,
        arch: &arch,
        dims: &dims,
        weights: &weights,
        file_size,
        awq_dir: awq.as_ref().map(|(dir, _)| dir.as_str()),
        split: split.is_some(),
    };
    let placement = plan(&source, &window, ask, false)?;
    let layers = placement.layers.as_ref().ok_or_else(|| {
        anyhow!(
            "{} is placed whole on one device by its own loader",
            ask.model_id
        )
    })?;
    Ok(read(&source, layers, ask))
}

/// The preview of the plan of `layers`.
fn preview_of(source: &Source<'_>, layers: &Layers, ask: &Ask) -> PlacementPreview {
    let per_layer = source
        .weights
        .layer_dense_and_expert_bytes(source.content, source.dims.num_layers)
        .filter(|_| ask.layer_share.is_none());
    layers.preview(
        &ask.model_id,
        source.arch,
        source.file_size,
        per_layer.as_deref(),
    )
}

/// The plan of `layers` as a choice between models reads it.
fn dry_plan_of(_: &Source<'_>, layers: &Layers, _: &Ask) -> DryPlan {
    let Some(plan) = layers.planned_rung() else {
        return DryPlan::default();
    };
    let on_host = plan
        .segments
        .iter()
        .any(|s| s.kind == DeviceKind::Cpu && s.layer_end > s.layer_start)
        || plan
            .host_experts_from
            .is_some_and(|from| from < plan.total_layers);
    DryPlan {
        on_cards: !on_host,
        cards: crate::inference::place::ladder::cards_of(&plan),
        predicted_tok_s: layers.decode_secs(&plan).map(|secs| 1.0 / secs),
        signature: layers.signature(&plan),
    }
}

/// `cards` with the bytes `given_back` holds added to each card's offer.
/// A card that offered nothing joins the cards, ranked by speed, the pinned card `pin` first.
fn with_given_back(
    cards: &[(usize, u64)],
    given_back: &[(usize, u64)],
    pin: Option<usize>,
) -> Vec<(usize, u64)> {
    let mut out = cards.to_vec();
    let mut joined = false;
    for &(card, bytes) in given_back {
        if pin.is_some_and(|p| p != card) {
            continue;
        }
        match out.iter_mut().find(|(idx, _)| *idx == card) {
            Some(entry) => entry.1 = entry.1.saturating_add(bytes),
            None if bytes > 0 => {
                out.push((card, bytes));
                joined = true;
            }
            None => {}
        }
    }
    match joined {
        true => crate::inference::place::ladder::prefer(
            &crate::inference::place::ladder::fastest_first(&out),
            pin,
        ),
        false => out,
    }
}

/// A plan a load would build now, read without loading.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DryPlan {
    /// No block and no routed expert of the plan sits on the host.
    pub on_cards: bool,
    /// The cards the plan uses.
    pub cards: Vec<usize>,
    /// The decode rate the plan's estimate predicts, `None` when the estimate cannot say.
    pub predicted_tok_s: Option<f64>,
    /// The cards the plan uses and the weight bytes it keeps on the host.
    pub signature: crate::inference::place::decode_time::PlanSignature,
}

/// A model's header and its weights, as the cards will hold them.
pub(crate) struct Opened {
    pub(crate) mmap: std::sync::Arc<Mmap>,
    pub(crate) content: gguf_file::Content,
    /// The checkpoint directory and its shards, for a safetensors checkpoint.
    pub(crate) awq: Option<(String, Vec<String>)>,
    /// The parts of a split set, the draft head beside the model included.
    pub(crate) split: Option<crate::tensor::quantized::gguf_source::SplitGguf>,
    /// Bytes the cards and the host place: the shards, the parts of a set as held, the file
    /// otherwise, less the tables read by row.
    pub(crate) file_size: u64,
}

/// Opens `model_file`, `file_len` bytes on disk, and reads its header.
/// Only the header's pages are read.
pub(crate) fn open(model_file: &std::path::Path, file_len: u64) -> AnyResult<Opened> {
    // --- 0. AWQ (HF safetensors) ---
    // Its layers come from the shards; its header is synthesised from the checkpoint's
    // config, and from there it takes the same route as a GGUF file.
    let awq = resolve_awq_dir(&model_file.to_string_lossy())
        .map(|dir| awq_checkpoint(&dir).map(|(content, shards)| (dir, content, shards)))
        .transpose()?;

    // --- 1. Parse GGUF header via mmap ---
    let (mmap, content, awq) = match awq {
        Some((dir, content, shards)) => {
            // No GGUF file backs this model: a one-byte anonymous map stands in for
            // the bytes the GGUF-only readers would take.
            let map = memmap2::MmapOptions::new()
                .len(1)
                .map_anon()?
                .make_read_only()?;
            (std::sync::Arc::new(map), content, Some((dir, shards)))
        }
        None => {
            debug!("📂 Opening GGUF file: {}", model_file.display());
            let file = std::fs::File::open(model_file).map_err(|e| {
                let err_msg = format!("Failed to open GGUF file at {}: {e}", model_file.display());
                error!("❌ {}", err_msg);
                anyhow!(err_msg)
            })?;

            // Memory-map the file for efficient random access and OS read-ahead
            debug!("🔗 Memory-mapping GGUF file...");
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                let err_msg = format!("Failed to mmap GGUF file: {e}");
                error!("❌ {}", err_msg);
                anyhow!(err_msg)
            })?;
            // Arc so loaders can hand out zero-copy file views: a view holds an Arc clone,
            // keeping the mapping alive past this scope.
            // No page is advised here.
            // The advice is given per layer once placement is known, see
            // `advise_pages_by_placement`: WillNeed for what the host will read every token,
            // DontNeed for what the cards already hold.
            let mmap = std::sync::Arc::new(mmap);
            // The mapping is known with its file: page-cache advice on a range of it
            // reaches the file's pages.
            crate::tensor::mapped::register_file(&mmap, &std::sync::Arc::new(file));
            crate::inference::place::device_probe::debug_vram_by_card("after-mmap-parse");
            debug!("✓ GGUF file mmap'd: {} bytes", mmap.len());

            debug!("📖 Parsing GGUF header...");
            let mmap_parse_t = std::time::Instant::now();
            let mut content =
                gguf_file::Content::read_mapped(&mut Cursor::new(&mmap[..]), mmap.clone())
                    .map_err(|e| {
                        let err_msg = format!("GGUF parse error: {e}");
                        error!("❌ {}", err_msg);
                        anyhow!(err_msg)
                    })?;
            // Back the tensor data with the Arc<Mmap> so the loader reads weights as
            // zero-copy views (no host copy, the Arc pins the mmap) instead of copying
            // each tensor into owned host bytes.
            content.mmap_owner =
                Some(mmap.clone() as std::sync::Arc<dyn std::any::Any + Send + Sync>);
            debug!(
                "✓ GGUF header parsed in {:.1}ms",
                mmap_parse_t.elapsed().as_secs_f64() * 1000.0
            );
            (mmap, content, None)
        }
    };
    // A split set is one model: the union of its parts is the header and each tensor is
    // read from the part that holds it. A part whose siblings are missing is an error.
    // The draft head the model ships beside its layers is one more part of the set.
    let draft_head = awq
        .is_none()
        .then(|| crate::tensor::quantized::gguf_source::draft_head_beside(model_file))
        .flatten();
    let split = match (
        awq.is_none(),
        crate::inference::generic_transformer::is_a_split_part(&content) || draft_head.is_some(),
    ) {
        (true, true) => match crate::tensor::quantized::gguf_source::SplitGguf::open_with(
            model_file,
            draft_head.as_slice(),
        )
        .and_then(|mut s| {
            // A head stored under the checkpoint's names is read under the loader's.
            if draft_head.is_some() {
                let head = s.parts().len() - 1;
                let arch = get_gguf_string(&content, "general.architecture").unwrap_or_default();
                let listing =
                    get_gguf_u32(&content, &format!("{arch}.block_count")).and_then(|n| {
                        crate::inference::load::mtp_head::checkpoint_listing(
                            &s.parts()[head].content,
                            n as usize,
                            get_gguf_u32(&content, &format!("{arch}.attention.indexer.key_length"))?
                                as usize,
                        )
                    });
                if let Some(listing) = listing {
                    s.relist(head, &listing)?;
                }
            }
            Ok(s)
        }) {
            Ok(s) => Some(s),
            Err(e) => {
                error!("❌ {e}");
                return Err(anyhow!("{e}"));
            }
        },
        _ => None,
    };
    let content = match &split {
        Some(s) => s.merged_content(),
        None => content,
    };
    // What the cards will hold: the shards of an AWQ checkpoint, the parts of a set, the
    // file otherwise.
    let file_size = match (&awq, &split) {
        (Some((_, shards)), _) => shards
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum(),
        // A tensor held in another format than stored counts as it is held.
        (None, Some(s)) => s.held_bytes(),
        (None, None) => file_len,
    };
    // A table read by row is never placed: the cards and the host answer for the rest.
    let file_size = file_size.saturating_sub(
        crate::inference::generic_transformer::ngram::demand_paged_bytes(
            &content,
            &get_gguf_string(&content, "general.architecture").unwrap_or_default(),
        ),
    );
    Ok(Opened {
        mmap,
        content,
        awq,
        split,
        file_size,
    })
}

/// Where the generic route reads the layers of an opened model from.
pub(crate) fn generic_weights<'a>(
    mmap: &'a std::sync::Arc<Mmap>,
    awq: &'a Option<(String, Vec<String>)>,
    split: &'a Option<crate::tensor::quantized::gguf_source::SplitGguf>,
) -> GenericWeights<'a> {
    match (awq, split) {
        (Some((dir, shards)), _) => GenericWeights::Awq { dir, shards },
        (None, Some(s)) => GenericWeights::GgufSplit { split: s },
        (None, None) => GenericWeights::Gguf {
            mmap_bytes: &mmap[..],
            mmap,
        },
    }
}

/// The context a model serves and the format of its cache.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Window {
    /// The positions the model reaches: its trained context, cut where it would scale its
    /// queries by position.
    pub(crate) context_length: usize,
    /// The window asked for, else the model's own.
    pub(crate) user_context_length: usize,
    /// The configuration asked for a window.
    pub(crate) window_asked: bool,
    /// The cache format the layers build.
    pub(crate) kv_quant: KvQuant,
    /// The quantised attention kernels serve the model's head shape.
    pub(crate) kv_quant_supported: bool,
    /// Query heads sharing one key and value head.
    pub(crate) n_q_per_kv: usize,
    /// An unquantised card cache keeps 16-bit elements.
    pub(crate) narrow_cache: bool,
}

/// The context and the cache format of the model `content` describes, as `ask` configures it.
pub(crate) fn window(
    content: &gguf_file::Content,
    arch: &str,
    dims: &ModelDims,
    ask: &Ask,
) -> Window {
    let facts = crate::inference::model_facts::facts_or_default(arch);
    let trained =
        trained_context(content, arch).unwrap_or(crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
    // The context stops where the model would start scaling its queries by position.
    let context_length = match super::gguf::unscaled_query_reach(content, arch) {
        Some(reach) if reach < trained => {
            warn!(
                "{arch}: the context stops at {reach} tokens of the {trained} trained: past \
                 them the model scales its queries by position, which the attention does not apply"
            );
            reach
        }
        _ => trained,
    };
    // No window asked for: the model's own.
    let window_asked = ask.context_length != 0;
    let user_context_length = if ask.context_length == 0 {
        context_length
    } else {
        ask.context_length
    };
    let ModelDims {
        num_heads,
        num_kv_heads,
        head_dim,
        ..
    } = *dims;
    // The Q8/Q4 attention kernels are instantiated for this model's (head_dim, n_q_per_kv); a
    // shape outside what they can hold downgrades to kv_quant=Off below.
    let n_q_per_kv = if num_kv_heads > 0 {
        num_heads / num_kv_heads
    } else {
        1
    };
    // An architecture whose output a quantised cache degrades keeps it unquantised.
    let quantized_kv_degrades = facts.quantized_kv_degrades;
    // The quantised attention kernels do not cap scores: a model that caps them keeps the
    // unfused attention that does.
    let caps_attention = content
        .metadata
        .contains_key(&format!("{arch}.attn_logit_softcapping"));
    // Nor do they fold attention sinks into the softmax: a model with sinks keeps the
    // F-dtype cache its sink attention reads.
    let has_attn_sinks = content.tensor_infos.contains_key("blk.0.attn_sinks")
        || content.tensor_infos.contains_key("blk.0.attn_sinks.weight");
    let kv_quant_supported =
        crate::inference::cache::kv_core::kv_shape_supported(head_dim, n_q_per_kv)
            && !quantized_kv_degrades
            && !caps_attention
            && !has_attn_sinks;
    // The KV precision follows what the quantised kernels can serve and the fidelity level.
    // The models kept off a quantised cache are decided above by what they are (a head shape
    // the kernels do not cover, capped attention scores, attention sinks, a cache precision
    // the architecture's facts rule out), not by a list of names.
    let kv_quant = {
        let chosen = kv_cache_format(ask.kv_quant, kv_quant_supported, ask.fidelity);
        if chosen != ask.kv_quant && matches!(chosen, KvQuant::Off) {
            warn!(
                "⚠️  Q8/Q4 KV cache kernels don't support head_dim={} x n_q_per_kv={}. Auto-downgrading kv_quant to Off for {}.",
                head_dim, n_q_per_kv, ask.model_id
            );
        } else if chosen != ask.kv_quant {
            info!(
                "Auto-promoting kv_quant Off -> {chosen:?} for arch={arch} (fidelity={})",
                ask.fidelity.name()
            );
        }
        chosen
    };
    // Bytes per element of the cache the layers will build: F16 for an unquantised card cache
    // of an architecture that keeps it so (or one with sinks) where the level narrows it, else
    // F32.
    let narrow_cache = (quantized_kv_degrades || has_attn_sinks) && ask.fidelity.narrows();
    Window {
        context_length,
        user_context_length,
        window_asked,
        kv_quant,
        kv_quant_supported,
        n_q_per_kv,
        narrow_cache,
    }
}

/// What a placement reads of an opened model.
pub(crate) struct Source<'a> {
    pub(crate) content: &'a gguf_file::Content,
    pub(crate) arch: &'a str,
    pub(crate) dims: &'a ModelDims,
    pub(crate) weights: &'a GenericWeights<'a>,
    /// Bytes the cards and the host place, see [`Opened::file_size`].
    pub(crate) file_size: u64,
    /// The directory of a safetensors checkpoint.
    pub(crate) awq_dir: Option<&'a str>,
    /// The model is a split set.
    pub(crate) split: bool,
}

/// The devices a language model is offered and where its layers go on them.
pub(crate) struct Placement {
    /// What one card keeps free beside the model for its cache and its work.
    pub(crate) runtime_reserve: u64,
    /// Why every layer stays on the host, when the model says so.
    pub(crate) host_only: Option<&'static str>,
    /// The cards offered, as read.
    pub(crate) offer: crate::inference::place::llm::Offer,
    /// The cards the plan divides the layers over, each with the bytes it offers.
    pub(crate) cards: Vec<(usize, u64)>,
    /// The layered plan; `None` for a file its own loader places whole.
    pub(crate) layers: Option<Layers>,
}

/// The layered plan of the generic route.
pub(crate) struct Layers {
    /// What the model asks of each device.
    pub(crate) demand: Demand,
    /// The context the plan holds the cache for.
    pub(crate) planned_ctx: u64,
    /// The context a cache on a device is built for.
    pub(crate) cache_ctx: usize,
    /// The context a cache on the host alone is built for.
    pub(crate) host_ctx: usize,
    /// The cache format on a device.
    pub(crate) kv_quant: KvQuant,
    /// An unquantised card cache keeps 16-bit elements.
    pub(crate) narrow_cache: bool,
    /// The plan the load tries first, before the accelerators take the host's blocks.
    pub(crate) first_plan: HeteroPlan,
    /// The embedding table joins the first layer's card on [`Layers::first_plan`].
    pub(crate) stem_fits: bool,
    /// The Vulkan and OpenCL devices that take the blocks a rung leaves on the host.
    pub(crate) accelerators: Vec<(DeviceKind, u64)>,
    /// The cards the plan divides the layers over.
    pub(crate) cards: Vec<(usize, u64)>,
    /// The routed experts of a layer, when the model has them.
    pub(crate) routed: Option<crate::inference::place::tensor_parallel::RoutedExperts>,
    /// Whether the block bytes hold the routed experts: false for a wide residual.
    pub(crate) experts_in_blocks: bool,
}

/// Whether a file is the legacy vision model its own loader places whole.
pub(crate) fn places_whole(content: &gguf_file::Content) -> bool {
    content
        .tensor_infos
        .contains_key("text_model.transformer.embd.wte.weight")
}

/// The placement of `source` with `window`, as `ask` configures it.
/// `reclaim` lets the offer ask the other models on the cards for the memory they keep.
pub(crate) fn plan(
    source: &Source<'_>,
    window: &Window,
    ask: &Ask,
    reclaim: bool,
) -> AnyResult<Placement> {
    let Source {
        content,
        arch,
        dims,
        weights: generic_weights,
        file_size,
        awq_dir,
        split,
    } = *source;
    let Window {
        context_length,
        user_context_length,
        window_asked,
        kv_quant,
        narrow_cache,
        ..
    } = *window;
    let num_layers = dims.num_layers;
    let layer_share = ask.layer_share;

    // --- The cards the placement offers this model ---
    //
    // Each card offers its free memory under the fleet's pressure, less the staging a load of
    // this checkpoint churns through; `place::llm::offer` makes every listed card's context
    // before it measures it, so a card reads the same on every load. What one card must hold
    // for the model to live on it alone is the weights and the runtime reserve: the cache at
    // the reserve window, the cuBLAS workspace and a prefill chunk's activations, sized from
    // the model's own dimensions.
    let runtime_reserve_bytes = dims.runtime_reserve(context_length);
    let staging = load_staging(content, arch);
    // A model whose layers only the host runs is offered no card and no accelerator.
    let host_only = crate::inference::generic_transformer::host_only_reason(content, arch);
    if let Some(why) = host_only {
        info!("🖥️  {arch}: every layer stays on the host: {why}");
    }
    let offer = if ask.cards_off {
        debug!("CUDA disabled by config (disable_cuda=true)");
        crate::inference::place::llm::Offer::default()
    } else if host_only.is_some() {
        crate::inference::place::llm::Offer::default()
    } else {
        crate::inference::place::llm::offer_reading(
            &ask.model_id,
            ask.pin,
            file_size.saturating_add(runtime_reserve_bytes),
            staging,
            reclaim,
        )
    };
    crate::inference::place::device_probe::debug_vram_by_card("after-offer");
    // A model offered no card is offered none of what other models give back.
    let offered = match ask.given_back.is_empty() || ask.cards_off || host_only.is_some() {
        true => offer.cards.clone(),
        false => with_given_back(&offer.cards, &ask.given_back, ask.pin),
    };
    let cuda_devices: Vec<(usize, u64)> = beside_the_tower(
        offered,
        crate::inference::generic_transformer::splices_vision_tower(content, arch)
            .then(|| crate::inference::model::qwen35::vision::card_bytes(content)),
    );
    info!(
        "📊 cards offered (free less staging): {}; staging {:.2} GB, runtime reserve {:.2} GB",
        cuda_devices
            .iter()
            .map(|(i, b)| format!("{i}: {:.2} GB", *b as f64 / 1e9))
            .collect::<Vec<_>>()
            .join(", "),
        staging as f64 / 1e9,
        runtime_reserve_bytes as f64 / 1e9
    );
    if places_whole(content) {
        return Ok(Placement {
            runtime_reserve: runtime_reserve_bytes,
            host_only,
            offer,
            cards: cuda_devices,
            layers: None,
        });
    }

    // A config needs the RoPE pairing the file does not carry: an architecture the facts table
    // does not know is refused here, before any weight is placed.
    // A Sinkhorn wide residual reads its rotation from its own description.
    if !crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta::claims(content) {
        crate::inference::model_facts::rope_pairing(arch).map_err(|e| {
            let err = format!("cannot build {arch:?} through the generic transformer: {e}");
            error!("❌ {err}");
            anyhow!(err)
        })?;
    }
    // Per-layer KV cache memory cost, so each card holds its weights and its cache.
    let native_kv_bytes: u64 = if narrow_cache { 2 } else { 4 };
    // A layer with sliding-window attention caps its cache at the window: the average over
    // the pattern prices a layer, not the worst case.
    let sliding_window =
        get_gguf_u32(content, &format!("{arch}.attention.sliding_window")).unwrap_or(0) as u64;
    let pattern: Vec<bool> = match content
        .metadata
        .get(&format!("{arch}.attention.sliding_window_pattern"))
    {
        Some(gguf_file::Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                gguf_file::Value::U8(n) => *n != 0,
                gguf_file::Value::Bool(b) => *b,
                gguf_file::Value::U32(n) => *n != 0,
                _ => true,
            })
            .collect(),
        _ => Vec::new(),
    };
    // A quantised cache (Q4KvCache / Q8KvCache) starts at the working window and grows on
    // demand up to the configured context, and is planned at that context; an F-dtype cache
    // pre-allocates in full.
    let lazy_kv_initial_cap = crate::inference::cache::KV_WORKING_WINDOW_TOKENS as u64;
    // What one layer's cache costs at a given context, sliding windows included: a windowed
    // layer never holds more than its window.
    // Only the attending layers keep a cache; a recurrent mixer keeps a state of fixed size
    // instead, so the cache is spread over the attending share.
    let attn_layers =
        crate::inference::generic_transformer::mixer::attention_layer_count(content, num_layers)
            as u64;
    let over_attending = |bytes: u64| -> u64 {
        if attn_layers < num_layers as u64 {
            bytes.saturating_mul(attn_layers) / num_layers.max(1) as u64
        } else {
            bytes
        }
    };
    // A Sinkhorn wide residual keeps its own windows and compressed rows: its description
    // prices the cache.
    let wide_meta =
        crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta::claims(content)
            .then(|| {
                crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta::from_file(content)
                    .ok()
            })
            .flatten();
    let kv_bytes_at = |ctx: u64| -> u64 {
        if let Some(meta) = &wide_meta {
            return meta.kv_bytes_per_layer(ctx as usize);
        }
        let avg = if !pattern.is_empty() && sliding_window > 0 {
            let n_swa = pattern.iter().filter(|&&is_swa| is_swa).count() as u64;
            let n_global = pattern.len() as u64 - n_swa;
            (n_swa.saturating_mul(ctx.min(sliding_window)) + n_global.saturating_mul(ctx))
                / pattern.len() as u64
        } else {
            ctx
        };
        over_attending(kv_cache_bytes(
            kv_quant,
            native_kv_bytes,
            2u64 * dims.num_kv_heads as u64 * dims.head_dim as u64 * avg,
        ))
    };
    // What the model asks of the cards; the ladder in `place` turns it into a plan, whole on
    // one card when it fits there, and recovers from running out of card memory. The cache is
    // part of what has to fit: a placement sized on the weights alone runs out once the
    // context fills.
    if layer_share.is_some() && awq_dir.is_some() {
        return Err(anyhow!(
            "a layer share is read from a GGUF file, not a checkpoint directory"
        ));
    }
    if let Some((from, to)) = layer_share {
        if from >= to || to > num_layers {
            return Err(anyhow!(
                "layers [{from}, {to}) are not within the {num_layers} of {arch}"
            ));
        }
    }
    // A wide residual puts its token mixers on a card and keeps everything else on the host:
    // the mixers are the blocks the cards hold.
    let card_mixers = layer_share
        .is_none()
        .then(|| crate::inference::generic_transformer::wide_sparse_plan::card_mixer_bytes(content))
        .flatten();
    let (block_bytes, head_bytes) = match &card_mixers {
        Some(mixers) => (mixers.total(), 0),
        None => generic_weights
            .block_and_head_bytes(content, layer_share, num_layers)
            .unwrap_or((file_size, 0)),
    };
    // The adapters a request may select sit beside the layers they correct.
    let (from, to) = layer_share.unwrap_or((0, num_layers));
    let adapter_widths = crate::inference::load::lora::AdapterWidths {
        hidden: dims.hidden_size,
        qkv: (dims.num_heads + 2 * dims.num_kv_heads) * dims.head_dim,
    };
    let adapter_bytes = crate::inference::load::lora::text_adapter_bytes(from..to, adapter_widths);
    if adapter_bytes > 0 {
        info!(
            "placement holds {:.1} MB beside the layers for the adapters requests may select",
            adapter_bytes as f64 / 1e6
        );
    }
    let block_bytes = block_bytes + adapter_bytes;
    let demand_at = |ctx: u64| Demand {
        layers: layer_share.map_or(num_layers, |(from, to)| to - from),
        block_bytes,
        head_bytes,
        kv_bytes_per_layer: kv_bytes_at(ctx),
        runtime_reserve: dims.work_reserve(),
    };

    // Choose the context that yields the best PLAN, not the largest number.
    // An F-dtype cache honouring a context the cards cannot hold does not fail: it pushes
    // layers onto the host, and a host layer costs far more than a context window nobody asked
    // for. So the placement takes the largest context whose plan keeps every layer on the
    // cards, halving down to the working window.
    let ctx_for_planning: u64 = if window_asked && matches!(kv_quant, KvQuant::Q4 | KvQuant::Q8) {
        // The cache grows to the declared context while a long prompt prefills on cards that
        // hold the weights, so the plan keeps that room: a model that then does not fit takes
        // the ladder's rungs, not a prefill that runs out of memory part-way. A window nobody
        // declared takes the else branch: it never pushes a layer off the cards.
        user_context_length.min(context_length) as u64
    } else {
        let full = user_context_length as u64;
        crate::inference::place::llm::context_the_cards_hold(
            full,
            lazy_kv_initial_cap,
            &cuda_devices,
            crate::inference::place::ladder::card_bandwidth,
            window_asked,
            demand_at,
        )
    };
    let demand = demand_at(ctx_for_planning);
    // A model the cards hold whole with room for a full prefill chunk beside each card's
    // layers keeps that room: its prompts then cross every layer in whole chunks. One the room
    // would push onto the host or onto more cards keeps the smaller reserve.
    let demand = {
        let roomy = Demand {
            runtime_reserve: dims.full_chunk_reserve(ask.prefill_row_bytes),
            ..demand
        };
        let on_cards = |d: &Demand| {
            let plan = d.plan(&cuda_devices);
            !plan.segments.is_empty()
                && plan
                    .segments
                    .iter()
                    .all(|s| matches!(s.kind, DeviceKind::Cuda(_)))
        };
        let cards =
            |d: &Demand| crate::inference::place::ladder::cards_of(&d.plan(&cuda_devices)).len();
        if roomy.runtime_reserve > demand.runtime_reserve
            && on_cards(&roomy)
            && cards(&roomy) <= cards(&demand)
        {
            info!(
                "placement keeps {:.2} GB a card for whole prefill chunks",
                roomy.runtime_reserve as f64 / 1e9
            );
            roomy
        } else {
            demand
        }
    };
    // Cap user_context_length at the model's GGUF context: the model cannot generate beyond its
    // trained context. The plan chose its context because the cards can hold it; the cache
    // must be allocated at that size or the plan promised room the allocation then takes back.
    // A quantised cache was planned at its initial cap and grows on demand, so its ceiling is
    // the configured context; an F-dtype cache is allocated in full at the context the plan
    // chose.
    let effective_ctx = if matches!(kv_quant, KvQuant::Q4 | KvQuant::Q8) {
        user_context_length.min(context_length)
    } else {
        user_context_length
            .min(context_length)
            .min(ctx_for_planning as usize)
    };

    // A dense model that does not sit on one card may split every layer across a group of
    // cards instead of placing them layer by layer: when the file's layers split, the group
    // holds it, and the measured reads and links predict it faster.
    // What a row costs through a layer: a multiply and an add per weight of the layers'
    // matrices.
    // A mixture layer's experts: a row reads and multiplies the ones it routes to.
    let blocks = || {
        content
            .tensor_infos
            .iter()
            .filter(|(name, _)| name.starts_with("blk."))
    };
    let expert_count = get_gguf_u32(content, &format!("{arch}.expert_count")).unwrap_or(0) as usize;
    let expert_used =
        get_gguf_u32(content, &format!("{arch}.expert_used_count")).unwrap_or(0) as usize;
    // Per layer over the layers the header describes: a synthesised header (an AWQ
    // checkpoint's) describes the first layer only.
    let described = blocks()
        .filter_map(|(name, _)| {
            name.strip_prefix("blk.")?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .collect::<std::collections::HashSet<usize>>()
        .len()
        .clamp(1, num_layers.max(1));
    let per_layer = |v: f64| v / described as f64;
    let routed = (expert_count > 0 && expert_used > 0).then(|| {
        let bytes: u64 = blocks()
            .filter(|(name, _)| name.contains("_exps."))
            .map(|(_, t)| {
                t.shape.elem_count() as u64 * t.ggml_dtype.type_size() as u64
                    / t.ggml_dtype.block_size() as u64
            })
            .sum();
        crate::inference::place::tensor_parallel::RoutedExperts {
            bytes: per_layer(bytes as f64) as u64,
            experts: expert_count,
            used: expert_used,
        }
    });
    let routed_share = routed.map_or(0.0, |r| r.used as f64 / r.experts as f64);
    let row_flops = 2.0
        * per_layer(
            blocks()
                .map(|(name, t)| {
                    let n = t.shape.elem_count() as f64;
                    match (name.contains("_exps."), t.shape.dims().len()) {
                        (true, _) => n * routed_share,
                        (false, 2) => n,
                        _ => 0.0,
                    }
                })
                .sum::<f64>(),
        );
    // A context nobody asked for gives way to the split: the cache is halved, down to the
    // working window, until each card holds its share of it beside its share of the weights.
    // An asked context is kept.
    let mut tp_ctx = None;
    let tp_plan = (layer_share.is_none() && !split)
        .then(|| {
            let rates = crate::inference::place::expert_parallel::card_read_rates();
            let weight = |idx: usize| rates.get(&idx).copied().unwrap_or(0.0);
            let holds = |d: &Demand| {
                (2..=cuda_devices.len()).any(|ranks| {
                    crate::inference::place::llm::tensor_parallel(
                        d,
                        &cuda_devices,
                        ranks,
                        routed.is_some(),
                        weight,
                    )
                    .is_some()
                })
            };
            let mut ctx = ctx_for_planning;
            let at = |ctx: u64| Demand {
                kv_bytes_per_layer: kv_bytes_at(ctx),
                ..demand
            };
            while !window_asked && ctx > lazy_kv_initial_cap && !holds(&at(ctx)) {
                ctx = (ctx / 2).max(lazy_kv_initial_cap);
            }
            let d = at(ctx);
            if !holds(&d) {
                info!("stays layer by layer: no group of cards holds a split's shares at context {ctx}");
                return None;
            }
            let p = crate::inference::place::tensor_parallel::plan(
                &d,
                &cuda_devices,
                (dims.hidden_size * std::mem::size_of::<f32>()) as u64,
                row_flops,
                routed,
                super::PREFILL_CHUNK_TOKENS,
                |ranks| match awq_dir {
                    Some(dir) => crate::inference::load::awq_loader::parse_config_json(dir)
                        .map_err(|e| e.to_string())
                        .map_or_else(Some, |cfg| {
                            crate::inference::generic_transformer::tp_awq_refusal(
                                content,
                                arch,
                                ranks,
                                cfg.group_size,
                            )
                        }),
                    None => {
                        crate::inference::generic_transformer::tp_file_refusal(content, arch, ranks)
                    }
                },
            );
            if p.is_some() && ctx < ctx_for_planning {
                info!("KV context {ctx_for_planning} -> {ctx}: each card holds its share of the split there");
                tp_ctx = Some(ctx);
            }
            p
        })
        .flatten();
    let effective_ctx = tp_ctx.map_or(effective_ctx, |c| effective_ctx.min(c as usize));

    // The embedding table joins the first layer's card when the plan can hold it there
    // without moving a layer.
    let stem_bytes = generic_weights.stem_bytes(content);
    let (first_plan, stem_fits) = demand.plan_stem_if_free(&cuda_devices, stem_bytes);
    // A plan that sends whole layers to the host is replaced, for a mixture of experts, by one
    // that sends only routed experts there: every layer's attention and cache then stay on a
    // card.
    // Whether the routed experts left on the host can have a cache of them on the cards: the
    // file says, from its stacks' formats.
    let cached_experts = crate::inference::generic_transformer::expert_tier::header_serves(content);
    let expert_split = generic_weights
        .layer_dense_and_expert_bytes(content, num_layers)
        .filter(|_| layer_share.is_none() && card_mixers.is_none())
        .and_then(|layers| {
            demand.plan_host_experts(&first_plan, &cuda_devices, &layers, cached_experts)
        });
    if let Some(p) = &expert_split {
        info!(
            "📦 routed experts of layers {}-{} kept in host memory; every layer's attention stays on a card",
            p.host_experts_from.unwrap_or(num_layers),
            num_layers
        );
    }
    let (first_plan, stem_fits) = match expert_split {
        Some(p) => (p, false),
        None => (first_plan, stem_fits),
    };
    // A token passes through the cards in turn: the fastest one holds the most it can. The
    // embedding table then stays off the card the sequence starts on.
    let ordered = demand.for_one_stream(first_plan.clone(), &cuda_devices);
    let (first_plan, stem_fits) =
        if crate::inference::place::dry_plan::same_plan(&ordered, &first_plan) {
            (first_plan, stem_fits)
        } else {
            (ordered, false)
        };
    let (first_plan, stem_fits) = match tp_plan {
        Some(p) => (p, false),
        None => (first_plan, stem_fits),
    };
    // The Vulkan devices take the layers a rung leaves on the host before the host does.
    // The OpenCL GPUs no CUDA card stands for take their share after them.
    let accelerators = if host_only.is_some() {
        Vec::new()
    } else {
        crate::inference::place::ladder::layer_accelerators(runtime_reserve_bytes)
    };
    Ok(Placement {
        runtime_reserve: runtime_reserve_bytes,
        host_only,
        offer,
        cards: cuda_devices.clone(),
        layers: Some(Layers {
            demand,
            planned_ctx: ctx_for_planning,
            cache_ctx: effective_ctx,
            host_ctx: user_context_length.min(context_length),
            kv_quant,
            narrow_cache,
            first_plan,
            stem_fits,
            accelerators,
            cards: cuda_devices,
            routed,
            experts_in_blocks: card_mixers.is_none(),
        }),
    })
}

impl Layers {
    /// The cache format and the context the layers build on `plan`.
    /// A plan on the host alone keeps an unquantised cache that grows to the whole window.
    pub(crate) fn cache_for(&self, plan: &HeteroPlan) -> (KvQuant, usize) {
        let on_accelerator = plan
            .segments
            .iter()
            .any(|s| matches!(s.kind, DeviceKind::Vulkan(_) | DeviceKind::OpenCL(_)));
        let host_only =
            crate::inference::place::ladder::cards_of(plan).is_empty() && !on_accelerator;
        if host_only {
            (KvQuant::Off, self.host_ctx)
        } else {
            (self.kv_quant, self.cache_ctx)
        }
    }

    /// The plan the load builds when no build runs out of memory.
    pub(crate) fn planned_rung(&self) -> Option<HeteroPlan> {
        crate::inference::place::ladder::planned_rung(
            self.first_plan.clone(),
            &self.demand,
            &self.cards,
            &self.accelerators,
        )
    }

    /// The routed sets of the layers on `plan`, with no measure of a set.
    fn routed_sets(
        &self,
        plan: &HeteroPlan,
    ) -> Option<crate::inference::place::decode_time::RoutedSets> {
        self.routed.map(|r| {
            crate::inference::place::decode_time::RoutedSets::on(
                plan,
                r,
                self.experts_in_blocks,
                None,
            )
        })
    }

    /// The cards `plan` uses and the weight bytes it keeps on the host.
    pub(crate) fn signature(
        &self,
        plan: &HeteroPlan,
    ) -> crate::inference::place::decode_time::PlanSignature {
        self.demand.signature(plan, self.routed_sets(plan).as_ref())
    }

    /// Seconds a decoded token is estimated to take on `plan`, `None` when the estimate cannot
    /// say.
    pub(crate) fn decode_secs(&self, plan: &HeteroPlan) -> Option<f64> {
        let sets = self.routed_sets(plan);
        self.demand
            .decode_secs(
                plan,
                &crate::inference::place::ladder::card_bandwidth,
                &crate::inference::place::host_read::host_bandwidth,
                sets.as_ref(),
            )
            .filter(|secs| *secs > 0.0)
    }

    /// The report of the plan the load builds.
    /// `per_layer` gives each layer's dense and routed expert bytes, when the source can say.
    fn preview(
        &self,
        model: &str,
        arch: &str,
        file_size: u64,
        per_layer: Option<&[(u64, u64)]>,
    ) -> PlacementPreview {
        let Some(plan) = self.planned_rung() else {
            return PlacementPreview {
                model: model.to_string(),
                architecture: arch.to_string(),
                layers: self.demand.layers,
                weight_bytes: file_size,
                context_length: 0,
                kv_format: String::new(),
                devices: Vec::new(),
                host: HostShare::default(),
                warnings: Vec::new(),
                refused: Some(
                    "the model does not fit the allowed backends, and the host is not one of them"
                        .to_string(),
                ),
            };
        };
        let (kv, ctx) = self.cache_for(&plan);
        let kv_format = match kv {
            KvQuant::Q8 => "q8_0",
            KvQuant::Q4 => "q4_0",
            KvQuant::Off if self.narrow_cache => "f16",
            KvQuant::Off => "f32",
        };
        let offered = |kind: DeviceKind| match kind {
            DeviceKind::Cuda(idx) => self.cards.iter().find(|(i, _)| *i == idx).map(|(_, b)| *b),
            DeviceKind::Cpu => None,
            other => self
                .accelerators
                .iter()
                .find(|(k, _)| *k == other)
                .map(|(_, b)| *b),
        };
        crate::inference::place::preview::build(
            crate::inference::place::preview::Facts {
                model,
                architecture: arch,
                plan: &plan,
                demand: &self.demand,
                per_layer: per_layer.filter(|_| self.experts_in_blocks),
                routed_layer_bytes: self.routed.as_ref().map(|r| r.bytes),
                experts_in_blocks: self.experts_in_blocks,
                weight_bytes: file_size,
                context_length: ctx,
                kv_format,
            },
            offered,
        )
    }
}

#[cfg(test)]
mod given_back_tests {
    use super::with_given_back;

    /// The bytes given back join the offer of their card, and a pinned model takes only its card's.
    #[test]
    fn the_room_given_back_joins_the_offer_of_its_card() {
        let cards = [(0, 10), (1, 4)];
        assert_eq!(
            with_given_back(&cards, &[(1, 6)], None),
            vec![(0, 10), (1, 10)]
        );
        assert_eq!(with_given_back(&cards, &[(1, 6)], Some(0)), cards.to_vec());
        let joined = with_given_back(&[(0, 10)], &[(1, 6), (2, 0)], None);
        assert_eq!(joined.len(), 2);
        assert!(joined.contains(&(1, 6)));
    }
}
