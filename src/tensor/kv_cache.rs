//! Key/value caches: what a decode step appends to and reads back.
//!
//! Two strategies live here because they cost differently, not because they differ. One keeps a
//! buffer sized ahead and writes each step into it in place, growing in chunks when the sequence
//! outruns it; the other holds what it has and concatenates each step onto it. The first pays
//! once per growth and nothing per step; the second pays a copy of the whole cache per step but
//! never reserves what it will not use. A test below holds them against each other.

use super::{Device, Error, Result, Tensor};

/// One growing buffer along one axis.
///
/// The buffer is reserved whole on the first append and written into in place afterwards, so a
/// step costs what it writes rather than a copy of everything before it. A sequence that
/// outruns the reservation extends it by whole `chunk`s - the only time anything is copied.
///
/// The reservation is not tracked separately: it is the buffer's own length along `axis`, so
/// there is no second number that could disagree with the tensor.
#[derive(Debug, Clone)]
pub struct Cache {
    reserved: Option<Tensor>,
    axis: usize,
    filled: usize,
    chunk: usize,
}

impl Cache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            reserved: None,
            axis: dim,
            filled: 0,
            chunk: max_seq_len.max(1),
        }
    }

    pub fn dim(&self) -> usize {
        self.axis
    }

    pub fn current_seq_len(&self) -> usize {
        self.filled
    }

    /// How many steps fit before the next growth. Before the first append that is the initial
    /// reservation; afterwards it is what the buffer actually holds.
    pub fn max_seq_len(&self) -> usize {
        match &self.reserved {
            Some(t) => t.dims()[self.axis],
            None => self.chunk,
        }
    }

    pub fn all_data(&self) -> &Option<Tensor> {
        &self.reserved
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        self.reserved
            .as_ref()
            .map(|t| t.narrow(self.axis, 0, self.filled))
            .transpose()
    }

    pub fn reset(&mut self) {
        self.filled = 0;
        self.reserved = None;
    }

    /// An independent copy, buffer included.
    ///
    /// The derived `Clone` shares the storage behind the tensor, so an append through either
    /// handle would write into both; a snapshot taken to be reused by another request must not
    /// move when the request it came from continues. `affine(1, 0)` is the dtype-preserving way
    /// to ask for fresh storage holding the same values.
    pub fn deep_copy(&self) -> Result<Self> {
        Ok(Self {
            reserved: self
                .reserved
                .as_ref()
                .map(|t| t.affine(1.0, 0.0))
                .transpose()?,
            ..self.clone()
        })
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let step = src.dim(self.axis)?;
        let wanted = self.filled + step;

        // How long the buffer has to be to take this step: whole chunks, in one go, so that a
        // step several times the chunk size still costs a single copy.
        let have = self.reserved.as_ref().map_or(0, |t| t.dims()[self.axis]);
        if wanted > have {
            let short_by = wanted - have;
            let grown = have + self.chunk * short_by.div_ceil(self.chunk);
            let mut shape = src.dims().to_vec();
            shape[self.axis] = grown - have;
            let tail = Tensor::zeros_on(shape, src.dtype(), &src.device())?;
            self.reserved = Some(match self.reserved.take() {
                Some(head) => Tensor::cat(&[&head, &tail], self.axis)?,
                None => tail,
            });
        }

        // `reserved` was just made long enough, either now or by an earlier append.
        match self.reserved.as_mut() {
            Some(buf) => buf.slice_set(src, self.axis, self.filled)?,
            None => return Err(Error::msg("kv cache: reserved nothing to append into")),
        }
        self.filled = wanted;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    k: Cache,
    v: Cache,
}

impl KvCache {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            k: Cache::new(dim, max_seq_len),
            v: Cache::new(dim, max_seq_len),
        }
    }

    pub fn k_cache(&self) -> &Cache {
        &self.k
    }

    pub fn v_cache(&self) -> &Cache {
        &self.v
    }

    /// Both halves, in the order every caller names them.
    ///
    /// K and V are appended to and dropped together - no operation here touches one without
    /// the other - so the ones that drive both are written once over the pair.
    fn halves(&mut self) -> [&mut Cache; 2] {
        [&mut self.k, &mut self.v]
    }

    /// What has been written to K, narrowed to the part in use.
    ///
    /// Composed on top of the accessor above rather than reaching for the field again: there is
    /// one way into each half, so how a half is held has one place to change.
    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k_cache().current_data()
    }

    /// What has been written to V, narrowed to the part in use.
    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v_cache().current_data()
    }

    /// Nothing here appends to one half without the other, so either answers for both.
    pub fn current_seq_len(&self) -> usize {
        self.k_cache().current_seq_len()
    }

    /// Independent deep copy of both K and V backing buffers (see `Cache::deep_copy`).
    pub fn deep_copy(&self) -> Result<Self> {
        Ok(Self {
            k: self.k.deep_copy()?,
            v: self.v.deep_copy()?,
        })
    }

    /// Append without materializing the in-use narrows. Decode paths whose
    /// attention kernel takes an explicit `kv_len` + strides (flash decode)
    /// read the FULL backing buffers instead - `current_data()` costs a
    /// device narrow-COPY of the whole growing cache per call on this
    /// (packed, view-less on middle dims) substrate.
    pub fn append_write(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        for (half, step) in self.halves().into_iter().zip([k, v]) {
            half.append(step)?;
        }
        Ok(())
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.append_write(k, v)?;
        // Both appends just created the buffers, so neither read can come back empty.
        match (self.k()?, self.v()?) {
            (Some(k), Some(v)) => Ok((k, v)),
            _ => Err(Error::msg("kv cache: appended and then read back nothing")),
        }
    }

    /// Drops both buffers. The next append reserves again from nothing.
    pub fn reset(&mut self) {
        for half in self.halves() {
            half.reset();
        }
    }
}

/// The same cache, holding exactly what it has been given.
///
/// Every append copies the whole cache into a larger tensor, which is what the [`Cache`] above
/// exists to avoid - but nothing is reserved, so a request that stops early has cost only what
/// it used. The multi-GPU MoE path takes this one, where the reservation would be made on every
/// card. Inference only: each append detaches, so no step holds the previous one alive.
#[derive(Debug, Clone)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
}

impl ConcatKvCache {
    pub fn new(dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            dim,
        }
    }

    /// The length is not tracked. It is what the stored tensor measures along `dim`, so there
    /// is no second number that could disagree with it.
    pub fn current_seq_len(&self) -> usize {
        let Some(k) = &self.k else { return 0 };
        k.dims().get(self.dim).copied().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Detach: KV caches are inference-only (break the BackpropOp chain).
        let (k, v) = (k.contiguous()?.detach(), v.contiguous()?.detach());
        let axis = self.dim;
        let k = Self::grow(&mut self.k, k, axis)?;
        let v = Self::grow(&mut self.v, v, axis)?;
        Ok((k, v))
    }

    /// Put `step` after whatever `held` already holds, keep the result, and hand it back.
    ///
    /// Written once for both halves because a key and a value are extended the same way. The
    /// concatenation is detached again: it is what the next step will be placed after, and an
    /// attached one would hold every step before it alive.
    fn grow(held: &mut Option<Tensor>, step: Tensor, axis: usize) -> Result<Tensor> {
        let whole = match held.as_ref() {
            Some(before) => Tensor::cat(&[before, &step], axis)?.detach(),
            None => step,
        };
        *held = Some(whole.clone());
        Ok(whole)
    }

    /// A cache with nothing in it is a new cache.
    pub fn reset(&mut self) {
        *self = Self::new(self.dim);
    }

    pub fn trim_to(&mut self, new_len: usize) -> Result<()> {
        if new_len == 0 {
            self.reset();
            return Ok(());
        }
        if new_len >= self.current_seq_len() {
            return Ok(());
        }
        if let Some(k) = &self.k {
            self.k = Some(k.narrow(self.dim, 0, new_len)?.contiguous()?);
        }
        if let Some(v) = &self.v {
            self.v = Some(v.narrow(self.dim, 0, new_len)?.contiguous()?);
        }
        Ok(())
    }

    pub fn k(&self) -> Option<&Tensor> {
        self.k.as_ref()
    }

    pub fn v(&self) -> Option<&Tensor> {
        self.v.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(seq: usize, dim_size: usize, value: f32) -> Tensor {
        let v: Vec<f32> = (0..seq * dim_size)
            .map(|i| value + i as f32 * 0.001)
            .collect();
        Tensor::from_vec(v, (1, 1, seq, dim_size), &Device::Cpu).unwrap()
    }

    fn rows(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The two caches are one cache.
    ///
    /// One reserves and writes in place, the other copies and grows; a decode step must not be
    /// able to tell which it was given. The sequence below crosses the reservation on purpose,
    /// because the growth is where the in-place one has something the other does not - and
    /// where an off-by-one would leave a step written past the end of what is read back.
    #[test]
    fn the_two_caches_answer_the_same_thing() {
        const DIM: usize = 2;
        const WIDTH: usize = 4;
        // Reserve less than the sequence will need, so the growth path runs.
        let mut growing = KvCache::new(DIM, 3);
        let mut concat = ConcatKvCache::new(DIM);

        for (i, seq) in [2usize, 1, 3, 1, 4].into_iter().enumerate() {
            let k = step(seq, WIDTH, i as f32);
            let v = step(seq, WIDTH, 100.0 + i as f32);
            let (gk, gv) = growing.append(&k, &v).unwrap();
            let (ck, cv) = concat.append(&k, &v).unwrap();

            assert_eq!(gk.dims(), ck.dims(), "after append {i}: shapes differ");
            assert_eq!(rows(&gk), rows(&ck), "after append {i}: keys differ");
            assert_eq!(rows(&gv), rows(&cv), "after append {i}: values differ");
            assert_eq!(
                growing.current_seq_len(),
                concat.current_seq_len(),
                "after append {i}: lengths differ"
            );
        }
    }

    /// A deep copy is independent of what it was copied from.
    ///
    /// The derived clone shares the buffer, so a later append through either handle would write
    /// into both - which is exactly what a snapshot taken for another request must not do.
    #[test]
    fn a_deep_copy_does_not_move_when_the_original_does() {
        let mut cache = KvCache::new(2, 8);
        let (k, v) = (step(2, 4, 1.0), step(2, 4, 2.0));
        cache.append(&k, &v).unwrap();
        let snapshot = cache.deep_copy().unwrap();
        let before = rows(&snapshot.k().unwrap().unwrap());

        cache.append(&step(2, 4, 9.0), &step(2, 4, 9.0)).unwrap();

        assert_eq!(snapshot.current_seq_len(), 2, "the snapshot grew");
        assert_eq!(
            rows(&snapshot.k().unwrap().unwrap()),
            before,
            "the snapshot's keys changed when the original was appended to"
        );
    }
}
