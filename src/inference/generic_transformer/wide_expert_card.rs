//! The routed experts of a wide-residual layer that a card keeps.
//!
//! A card keeps an expert's 3 weights in the blocks the file stores.
//! It runs them with the expert kernels that give the host's bits (`gpu_experts_exact`).
//! Only the formats those kernels read are kept: gate and up in `IQ2_XXS` with down in `Q2_K`,
//! or gate and up in `IQ3_XXS` with down in `Q3_K`.
//!
//! A layer's experts on 1 card run in 1 call.
//! The layer's rows go up once, every (expert, row) pair runs, and the outputs come back once.
//! A layer whose experts sit on several cards makes 1 call per card.

use super::block_scaled_experts::BlockScaledExpert;
use crate::tensor::Result;

/// The routed experts of 1 layer that the cards keep, by expert id.
///
/// Empty on the host and in a build without cards.
#[derive(Default)]
pub struct CardExperts {
    #[cfg(feature = "gpu")]
    sets: Vec<gpu::CardSet>,
    /// (expert, row) pairs the cards ran since the load.
    ran: std::sync::atomic::AtomicU64,
}

impl CardExperts {
    /// Whether no card keeps an expert of the layer.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Experts the cards keep.
    pub fn count(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.sets.iter().map(|s| s.experts.len()).sum()
        }
        #[cfg(not(feature = "gpu"))]
        0
    }

    /// Whether a card keeps expert `e`.
    pub fn holds(&self, e: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.sets.iter().any(|s| s.experts.contains_key(&e))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = e;
            false
        }
    }

    /// (expert, row) pairs the cards ran since the load.
    pub fn pairs_run(&self) -> u64 {
        self.ran.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Per card ordinal, the bytes of experts kept there.
    pub fn bytes_by_card(&self) -> Vec<(usize, u64)> {
        #[cfg(feature = "gpu")]
        {
            self.sets.iter().map(|s| (s.ordinal, s.bytes)).collect()
        }
        #[cfg(not(feature = "gpu"))]
        Vec::new()
    }

    /// The outputs of the experts `ids`, kept on the cards, over their rows.
    ///
    /// `rows[i]` are the row indices of `x` `[n, dim]` that expert `ids[i]` runs over.
    /// Answers per expert its outputs `[rows[i].len(), dim]` row-major, in the order of `ids`.
    /// `None` when a card call fails: the caller runs them on the host.
    pub fn run(
        &self,
        ids: &[usize],
        rows: &[Vec<usize>],
        x: &[f32],
        dim: usize,
        limit: f32,
    ) -> Option<Vec<Vec<f32>>> {
        #[cfg(feature = "gpu")]
        {
            let mut out: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
            for set in &self.sets {
                let mine: Vec<usize> = (0..ids.len())
                    .filter(|&i| set.experts.contains_key(&ids[i]))
                    .collect();
                if mine.is_empty() {
                    continue;
                }
                let ids_here: Vec<usize> = mine.iter().map(|&i| ids[i]).collect();
                let rows_here: Vec<Vec<usize>> = mine.iter().map(|&i| rows[i].clone()).collect();
                match set.run(&ids_here, &rows_here, x, dim, limit) {
                    Ok(ys) => {
                        let pairs: usize = rows_here.iter().map(Vec::len).sum();
                        self.ran
                            .fetch_add(pairs as u64, std::sync::atomic::Ordering::Relaxed);
                        for (&i, y) in mine.iter().zip(ys) {
                            out[i] = Some(y);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "card {}: routed experts not run, the host runs them: {e}",
                            set.ordinal
                        );
                        return None;
                    }
                }
            }
            out.into_iter().collect()
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (ids, rows, x, dim, limit);
            None
        }
    }

    /// Keeps `experts` (id and weights) on card `ordinal`, beside what the layer keeps.
    ///
    /// Refused when their format has no kernel that gives the host's bits, or the upload fails.
    #[cfg(feature = "gpu")]
    pub fn keep(&mut self, ordinal: usize, experts: &[(usize, &BlockScaledExpert)]) -> Result<()> {
        let set = gpu::CardSet::new(ordinal, experts)?;
        match self.sets.iter_mut().find(|s| s.ordinal == ordinal) {
            Some(held) => held.absorb(set),
            None => self.sets.push(set),
        }
        Ok(())
    }
}

/// Whether the cards have a kernel that gives the host's bits for `expert`'s format and shape.
pub fn served(expert: &BlockScaledExpert) -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu::format_of(expert).is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = expert;
        false
    }
}

/// The host's outputs of `experts` over the rows `x` `[n, dim]`, each over all the rows.
///
/// The experts run at once, as a layer's forward runs them.
pub fn host_outputs(
    experts: &[&BlockScaledExpert],
    x: &crate::tensor::Tensor,
    limit: f32,
) -> Result<Vec<Vec<f32>>> {
    use rayon::prelude::*;
    experts
        .par_iter()
        .map(|e| e.forward(x, limit)?.flatten_all()?.to_vec1::<f32>())
        .collect()
}

#[cfg(feature = "gpu")]
pub(super) mod gpu {
    use super::BlockScaledExpert;
    use crate::tensor::blockscaled::projection::Projection;
    use crate::tensor::cuda::{
        expert_tables, gpu_experts_exact, q8k32_rows_to_card, CudaDevice, ExpertFormat,
        ExpertTables,
    };
    use crate::tensor::gpu_runtime::driver::CudaSlice;
    use crate::tensor::quant_cpu::QK_K;
    use crate::tensor::{Device, Error, Result};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// The kernels' format of `expert`, `None` when no kernel reads it or its shape.
    ///
    /// The kernels read the rows as q8_K blocks: `dim` and `inter` are whole blocks of it.
    pub(crate) fn format_of(expert: &BlockScaledExpert) -> Option<ExpertFormat> {
        let format = ExpertFormat::of(
            expert.gate.quant_dtype()?,
            expert.up.quant_dtype()?,
            expert.down.quant_dtype()?,
        )?;
        // The gate is `[inter, dim]`.
        let dims = expert.gate.dims();
        let (inter, dim) = (*dims.first()?, *dims.get(1)?);
        (dim % QK_K == 0 && inter % QK_K == 0).then_some(format)
    }

    /// The stored blocks of a projection in a GGUF format.
    fn blocks(p: &Projection) -> Result<std::borrow::Cow<'_, [u8]>> {
        match p {
            Projection::Quant(q) => q.data(),
            _ => Err(Error::msg("a card expert is stored in GGUF blocks")),
        }
    }

    /// Experts of 1 format kept on 1 card.
    pub(crate) struct CardSet {
        pub(crate) ordinal: usize,
        dev: Arc<CudaDevice>,
        tables: Arc<ExpertTables>,
        /// `(dim, inter)`.
        dims: (usize, usize),
        /// Per expert id, gate, up and down blocks.
        pub(crate) experts: BTreeMap<usize, [CudaSlice<u8>; 3]>,
        /// Bytes of the blocks kept.
        pub(crate) bytes: u64,
    }

    impl CardSet {
        /// `experts` uploaded to card `ordinal`.
        ///
        /// Refused when they are not all of 1 format and shape the kernels read.
        pub(crate) fn new(ordinal: usize, experts: &[(usize, &BlockScaledExpert)]) -> Result<Self> {
            let Some(&(_, first)) = experts.first() else {
                return Err(Error::msg("no expert to keep on a card"));
            };
            let format = format_of(first)
                .ok_or_else(|| Error::msg("no card kernel reads these experts' format"))?;
            let dims = first.gate.dims();
            let dims = (dims[1], dims[0]);
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("card experts need a CUDA device"));
            };
            let tables = Arc::new(expert_tables(&dev, format)?);
            let mut set = Self {
                ordinal,
                dev,
                tables,
                dims,
                experts: BTreeMap::new(),
                bytes: 0,
            };
            for &(id, e) in experts {
                if format_of(e) != Some(format) || (e.gate.dims()[1], e.gate.dims()[0]) != dims {
                    return Err(Error::msg(format!(
                        "expert {id} differs in format or shape from the others on card {ordinal}"
                    )));
                }
                let stream = set.dev.stream();
                let mut up = |p: &Projection| -> Result<CudaSlice<u8>> {
                    let b = blocks(p)?;
                    set.bytes += b.len() as u64;
                    stream
                        .clone_htod(b.as_ref())
                        .map_err(|err| Error::msg(format!("expert {id} upload: {err}")))
                };
                let weights = [up(&e.gate)?, up(&e.up)?, up(&e.down)?];
                set.experts.insert(id, weights);
            }
            Ok(set)
        }

        /// Takes the experts of `other`, kept on the same card.
        pub(crate) fn absorb(&mut self, other: CardSet) {
            self.bytes += other.bytes;
            self.experts.extend(other.experts);
        }

        /// The outputs of `ids` over their `rows` of `x` `[n, dim]`, in the order of `ids`.
        pub(crate) fn run(
            &self,
            ids: &[usize],
            rows: &[Vec<usize>],
            x: &[f32],
            dim: usize,
            limit: f32,
        ) -> Result<Vec<Vec<f32>>> {
            if dim != self.dims.0 {
                return Err(Error::msg(format!(
                    "rows of {dim} for experts of {}",
                    self.dims.0
                )));
            }
            let weights = ids
                .iter()
                .map(|e| {
                    self.experts
                        .get(e)
                        .map(|[g, u, d]| (g, u, d))
                        .ok_or_else(|| Error::msg(format!("expert {e} is not on the card")))
                })
                .collect::<Result<Vec<_>>>()?;
            // Instances of 1 expert stand together, its rows in row order.
            let instances: Vec<(usize, usize)> = rows
                .iter()
                .enumerate()
                .flat_map(|(i, r)| r.iter().map(move |&t| (i, t)))
                .collect();
            if instances.is_empty() {
                return Ok(vec![Vec::new(); ids.len()]);
            }
            let xq = q8k32_rows_to_card(&self.dev, x)?;
            let (_h, y) = gpu_experts_exact(
                &self.dev,
                &weights,
                &self.tables,
                self.dims,
                &xq,
                &instances,
                limit,
            )?;
            let flat = self
                .dev
                .stream()
                .clone_dtoh(&y)
                .map_err(|e| Error::msg(format!("card expert outputs: {e}")))?;
            let mut at = 0;
            Ok(rows
                .iter()
                .map(|r| {
                    let n = r.len() * dim;
                    let part = flat[at..at + n].to_vec();
                    at += n;
                    part
                })
                .collect())
        }
    }
}
