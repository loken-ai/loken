//! What the parity tests of the Vulkan and OpenCL layers share: the model read from the store,
//! and the native card of the build, against which a gap from the host is judged.
//!
//! A first layer is held to the error of the native card in the same run.
//! A greedy pick that leaves the host's passes only when the native card leaves it at the
//! same step, fed the same tokens.
//! Without a native card, a test may judge against the device's own arithmetic computed on
//! the host.

use super::*;

pub(super) fn model_blob(tag: &str) -> std::path::PathBuf {
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, tag) = tag.split_once(':').unwrap_or((tag, "latest"));
    let m = crate::inference::load::ollama_manager::OllamaManager::new(store.clone());
    let manifest = m
        .read_manifest_json(name, tag)
        .expect("manifest in the store");
    let digest = manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")
        .and_then(|l| l["digest"].as_str())
        .expect("a model layer")
        .replace(':', "-");
    store.join("blobs").join(digest)
}

pub(super) fn host_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

pub(super) fn argmax(t: &Tensor) -> u32 {
    top2(&host_vec(t)).0 as u32
}

/// The two largest values of `l`, by index, the largest first.
/// Of equal values the first wins.
pub(super) fn top2(l: &[f32]) -> (usize, usize) {
    let mut best = (0, 1);
    if l[1] > l[0] {
        best = (1, 0);
    }
    for (i, &v) in l.iter().enumerate().skip(2) {
        if v > l[best.0] {
            best = (i, best.0);
        } else if v > l[best.1] {
            best = (best.0, i);
        }
    }
    best
}

pub(super) fn ids(t: &[u32]) -> Tensor {
    Tensor::new(t, &Device::Cpu).unwrap().unsqueeze(0).unwrap()
}

pub(super) fn self_content(mmap: &std::sync::Arc<memmap2::Mmap>) -> gguf_file::Content {
    gguf_file::Content::read_mapped(&mut std::io::Cursor::new(&mmap[..]), mmap.clone()).unwrap()
}

/// A model of the store, mapped once.
pub(super) struct Fixture {
    pub(super) mmap: std::sync::Arc<memmap2::Mmap>,
    pub(super) n_layers: usize,
}

impl Fixture {
    pub(super) fn open_tag(tag: &str) -> Self {
        let path = model_blob(tag);
        let file = std::fs::File::open(&path).unwrap();
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
        let c = self_content(&mmap);
        let arch = c.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone();
        let n_layers = GenericTransformerConfig::from_gguf(&c, &arch)
            .unwrap()
            .n_layers;
        Self { mmap, n_layers }
    }

    pub(super) fn build(&self, plan: &HeteroPlan) -> GenericHeteroTransformer {
        self.try_build(plan).unwrap()
    }

    pub(super) fn try_build(
        &self,
        plan: &HeteroPlan,
    ) -> std::result::Result<GenericHeteroTransformer, String> {
        GenericHeteroTransformer::from_gguf(
            self_content(&self.mmap),
            &self.mmap[..],
            &HashMap::new(),
            plan,
        )
        .map_err(|e| e.to_string())
    }

    pub(super) fn host(&self) -> HeteroPlan {
        HeteroPlan::forced_gpu(self.n_layers, 0, 0)
    }

    /// The model with its first `layers` layers on the native card `card`.
    pub(super) fn try_on_native(
        &self,
        card: &Device,
        layers: usize,
    ) -> std::result::Result<GenericHeteroTransformer, String> {
        let cards: HashMap<usize, Device> = [(0, card.clone())].into_iter().collect();
        GenericHeteroTransformer::from_gguf(
            self_content(&self.mmap),
            &self.mmap[..],
            &cards,
            &HeteroPlan::forced_gpu(self.n_layers, layers, 0),
        )
        .map_err(|e| e.to_string())
    }

    /// The host's logit rows at the prompt's last row and after each of its own greedy tokens.
    pub(super) fn host_rows(&self, prompt: &[u32], steps: usize) -> Vec<Vec<f32>> {
        let mut host = self.build(&self.host());
        let mut rows = vec![host_vec(&host.forward(&ids(prompt), 0).unwrap())];
        for i in 0..steps - 1 {
            let t = [top2(&rows[i]).0 as u32];
            rows.push(host_vec(&host.forward(&ids(&t), prompt.len() + i).unwrap()));
        }
        rows
    }

    /// The picks of the native card of the build at the rows of `host`, fed the host's tokens.
    /// The caller frees every other copy on the card first.
    /// The native copy is freed, and its cached memory given back, before this returns.
    pub(super) fn native_picks(&self, prompt: &[u32], host: &[Vec<f32>]) -> Native {
        let Some(card) = native_card() else {
            return Native::Absent;
        };
        let picks = self
            .try_on_native(&card, self.n_layers)
            .and_then(|mut m| try_forced_picks(&mut m, prompt, host));
        release_native(&card);
        match picks {
            Ok(picks) => Native::Picks(picks),
            Err(why) if out_of_memory(&why) => Native::NoRoom(why),
            Err(why) => panic!("the native card: {why}"),
        }
    }

    /// The largest error `max_err` gives the first layer over the hidden rows `h` of `tokens`
    /// positions on the native card, which is the bound of the first-layer tests.
    /// `fallback` stands in for it without a native card.
    pub(super) fn first_layer_bound(
        &self,
        tag: &str,
        h: &Tensor,
        tokens: usize,
        fallback: f32,
        max_err: impl Fn(&[f32]) -> f32,
    ) -> f32 {
        let Some(card) = native_card() else {
            return fallback;
        };
        let mut native = self.try_on_native(&card, 1).unwrap();
        let (got, _) = native
            .run_layers_in(h.clone(), None, 0, tokens, 0, 1)
            .unwrap();
        let err = max_err(&host_vec(&got));
        println!("{tag} layer 0 on the native card: max error {err:.3e}");
        drop(native);
        release_native(&card);
        err
    }
}

/// The native card of the build, CUDA or ROCm.
/// `None` when the build has no native runtime or no card answers.
pub(super) fn native_card() -> Option<Device> {
    #[cfg(feature = "gpu")]
    {
        Device::new_cuda(0).ok()
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// Gives the card back the memory the native runtime keeps cached for the copies already freed.
pub(super) fn release_native(card: &Device) {
    #[cfg(feature = "gpu")]
    if let Device::Cuda(cd) = card {
        crate::tensor::cuda::reclaim_pool(cd);
    }
    #[cfg(not(feature = "gpu"))]
    let _ = card;
}

/// Whether the error `why` says a device ran out of memory.
pub(super) fn out_of_memory(why: &str) -> bool {
    why.contains("out of memory") || why.contains("[oom]")
}

/// The picks of `m` at the rows of `host`, fed the host's tokens.
pub(super) fn forced_picks(
    m: &mut GenericHeteroTransformer,
    prompt: &[u32],
    host: &[Vec<f32>],
) -> Vec<usize> {
    try_forced_picks(m, prompt, host).unwrap()
}

/// [`forced_picks`], or the error of the first step that fails.
pub(super) fn try_forced_picks(
    m: &mut GenericHeteroTransformer,
    prompt: &[u32],
    host: &[Vec<f32>],
) -> std::result::Result<Vec<usize>, String> {
    let first = m.forward(&ids(prompt), 0).map_err(|e| e.to_string())?;
    let mut picks = vec![top2(&host_vec(&first)).0];
    for (i, row) in host[..host.len() - 1].iter().enumerate() {
        let t = [top2(row).0 as u32];
        let next = m
            .forward(&ids(&t), prompt.len() + i)
            .map_err(|e| e.to_string())?;
        picks.push(top2(&host_vec(&next)).0);
    }
    Ok(picks)
}

/// What the native card of the build says at the steps of a run.
pub(super) enum Native {
    /// Its picks, fed the host's tokens.
    Picks(Vec<usize>),
    /// No native card: the picks of the device's own arithmetic computed on the host, fed the
    /// host's tokens.
    #[cfg_attr(not(feature = "opencl"), allow(dead_code))]
    Arithmetic(Vec<usize>),
    /// The build has no native runtime, or no card answers.
    Absent,
    /// The model does not fit the native card: why.
    NoRoom(String),
}

/// Fails on each step where the pick of the device `at` leaves the host's while the native
/// pick keeps it.
/// Without a native runtime, a pick that leaves the host's passes only on an exact tie.
/// When the model does not fit the native card, a step where the picks part is reported
/// unjudged.
pub(super) fn judge_picks(at: &str, host: &[Vec<f32>], native: &Native, got: &[usize]) {
    let gaps = pick_gaps(at, host, native, got);
    assert!(gaps.is_empty(), "{at}: {}", gaps.join("; "));
}

/// The steps where the pick of the device `at` leaves the host's while the native pick, or the
/// pick of the device's own arithmetic, keeps it.
/// Without either, a pick that leaves the host's holds only on an exact tie.
/// When the model does not fit the native card, a step where the picks part is reported
/// unjudged.
pub(super) fn pick_gaps(
    at: &str,
    host: &[Vec<f32>],
    native: &Native,
    got: &[usize],
) -> Vec<String> {
    if let Native::NoRoom(why) = native {
        println!("{at}: the native card has no room for the model, its comparison skipped: {why}");
    }
    let mut gaps = Vec::new();
    for (step, (row, &v)) in host.iter().zip(got).enumerate() {
        let h = top2(row).0;
        if v == h {
            continue;
        }
        let margin = row[h] - row[v];
        let n = match native {
            Native::Picks(n) | Native::Arithmetic(n) => Some(n[step]),
            _ => None,
        };
        println!("{at}: step {step}: picks {v}, the host {h} by {margin:.3}, native {n:?}");
        let held = match (native, n) {
            (_, Some(n)) => n != h,
            (Native::NoRoom(_), _) => {
                println!("{at}: step {step}: unjudged");
                true
            }
            _ => margin == 0.0,
        };
        if !held {
            gaps.push(format!(
                "step {step}: picks {v}, the host {h} by {margin}, native {n:?}"
            ));
        }
    }
    gaps
}
