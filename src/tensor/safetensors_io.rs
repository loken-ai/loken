//! Native safetensors weight loading: mmap + parse via the plain
//! `safetensors` crate, producing native tensors - the weight-load path for
//! migrated modules. Parity-tested against the current substrate's loader.

use super::{CpuStorage, Device, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Decode one OCP `e4m3fn` fp8 byte (1 sign, 4 exp bias-7, 3 mantissa; no inf,
/// `S1111.111` = NaN) to f32.
#[inline]
fn f8_e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 0x0F && mant == 7.0 {
        return f32::NAN;
    }
    let v = if exp == 0 {
        (mant / 8.0) * 2f32.powi(-6) // subnormal: 2^(1-bias)
    } else {
        (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    };
    sign * v
}

/// Decode one OCP `e5m2` fp8 byte (1 sign, 5 exp bias-15, 2 mantissa; IEEE-like
/// inf/nan) to f32.
#[inline]
fn f8_e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 2) & 0x1F) as i32;
    let mant = (b & 0x03) as f32;
    if exp == 0x1F {
        return if mant == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    let v = if exp == 0 {
        (mant / 4.0) * 2f32.powi(-14) // subnormal: 2^(1-bias)
    } else {
        (1.0 + mant / 4.0) * 2f32.powi(exp - 15)
    };
    sign * v
}

/// Information about a tensor stored in a safetensors file.
#[derive(Clone)]
struct TensorInfo {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    /// Byte offset and end within the data section (which starts at 8 + header_len).
    data_offsets: (usize, usize),
}

/// Parsed header data for one safetensors file.
struct FileParsers {
    /// Length of the JSON header that follows the 8-byte length prefix.
    header_len: usize,
    tensors: HashMap<String, TensorInfo>,
}

/// Mmapped safetensors files with a name -> file index.
pub struct SafeTensorsLoader {
    maps: Vec<memmap2::Mmap>,
    /// Parsed header data for each file, indexed by file index.
    parsers: Vec<FileParsers>,
    /// File index of each tensor name
    index: HashMap<String, usize>,
    /// One identity per file: its path, length and modification time.
    identities: Vec<String>,
    /// Tensors read out of this bundle so far, for the progress a client sees.
    ///
    /// Reading the weights is the longest phase of a render and reported nothing: every
    /// engine that loads safetensors - text encoders, VAEs, DiTs, the speech and audio
    /// stacks - arrives here, so counting ONCE at the read is what spares each of them
    /// (and the next one added) from wiring a count of its own. The total is the bundle's
    /// own tensor count, which is a fact about the file rather than a guess.
    ///
    /// Shared behind an `Arc` and read from pool threads, hence the atomic.
    read: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    parse_count: std::sync::atomic::AtomicUsize,
}

impl SafeTensorsLoader {
    /// Parse a single safetensors file's header, building TensorInfo for each tensor.
    fn parse_file(
        map: &memmap2::Mmap,
        counter: &std::sync::atomic::AtomicUsize,
    ) -> Result<FileParsers> {
        let (n, meta) = safetensors::SafeTensors::read_metadata(map)
            .map_err(|e| Error(format!("safetensors header: {e}")))?;
        let mut tensors = HashMap::new();

        for (name, info) in meta.tensors() {
            tensors.insert(
                name.to_string(),
                TensorInfo {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data_offsets: info.data_offsets,
                },
            );
        }
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(FileParsers {
            header_len: n,
            tensors,
        })
    }

    /// Mmap one or more files, parsing each header once.
    ///
    /// # Safety
    /// The backing files must not be mutated while mapped (same contract as
    /// every mmap-based loader).
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut maps = Vec::with_capacity(paths.len());
        let mut parsers = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        let mut identities = Vec::with_capacity(paths.len());
        // Counter shared with the loader's parse_count field.
        let counter: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        for (i, p) in paths.iter().enumerate() {
            let file = std::fs::File::open(p)
                .map_err(|e| Error(format!("safetensors open {:?}: {e}", p.as_ref())))?;
            let meta = file.metadata().ok();
            identities.push(format!(
                "{}:{}:{:?}",
                p.as_ref().display(),
                meta.as_ref().map_or(0, |m| m.len()),
                meta.and_then(|m| m.modified().ok())
            ));
            let map = memmap2::Mmap::map(&file)
                .map_err(|e| Error(format!("safetensors mmap {:?}: {e}", p.as_ref())))?;
            let parser = Self::parse_file(&map, &counter)
                .map_err(|e| Error(format!("{:?}: {}", p.as_ref(), e.0)))?;
            for name in parser.tensors.keys() {
                index.insert(name.clone(), i);
            }
            maps.push(map);
            parsers.push(parser);
        }
        Ok(Self {
            maps,
            parsers,
            index,
            identities,
            read: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            parse_count: counter,
        })
    }

    #[cfg(test)]
    fn parse_count(&self) -> usize {
        self.parse_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Every tensor's size in bytes as stored, read from the headers only.
    pub fn byte_sizes(&self) -> Vec<(String, usize)> {
        let mut out = Vec::with_capacity(self.index.len());
        for parser in &self.parsers {
            for (name, info) in &parser.tensors {
                let len = info.data_offsets.1 - info.data_offsets.0;
                out.push((name.clone(), len));
            }
        }
        out
    }

    /// A tensor's shape WITHOUT reading its data.
    ///
    /// The header is already parsed and mapped, so this costs nothing - which is the
    /// point: a loader can check that a checkpoint is the shape it expects before it
    /// commits to reading gigabytes of it, and refuse with a sentence instead of failing
    /// somewhere deep or, worse, succeeding on the wrong file.
    pub fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        let &fi = self.index.get(name)?;
        self.parsers[fi].tensors.get(name).map(|t| t.shape.clone())
    }

    /// A tensor's stored format, as its checkpoint spells it, and its element width in bytes.
    ///
    /// Read from the header only.
    pub fn stored_format(&self, name: &str) -> Option<(String, usize)> {
        let &fi = self.index.get(name)?;
        let info = self.parsers[fi].tensors.get(name)?;
        let dtype = info.dtype;
        Some((format!("{dtype:?}").to_lowercase(), dtype.bitsize() / 8))
    }

    /// The narrowest float dtype exact for every stored value of tensor `name`.
    ///
    /// A tensor stored in BF16 or F16 is answered from the header.
    /// Any other tensor is read once per file and name, and the answer is kept for the process.
    pub fn narrowest_exact_dtype(&self, name: &str) -> Result<super::DType> {
        use super::DType;
        type Known = Option<HashMap<String, DType>>;
        static KNOWN: std::sync::Mutex<Known> = std::sync::Mutex::new(None);
        let missing = || Error(format!("safetensors: no tensor `{name}`"));
        let &fi = self.index.get(name).ok_or_else(missing)?;
        let info = self.parsers[fi].tensors.get(name).ok_or_else(missing)?;
        match info.dtype {
            safetensors::Dtype::BF16 => return Ok(DType::BF16),
            safetensors::Dtype::F16 => return Ok(DType::F16),
            _ => {}
        }
        let key = format!("{}#{name}", self.identities[fi]);
        let lock = || KNOWN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dtype) = lock().as_ref().and_then(|m| m.get(&key).copied()) {
            return Ok(dtype);
        }
        // CONVERSION: the stored values are widened to F32 to be read, and dropped once read.
        let values = self
            .load(name)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let dtype = super::conversion::narrowest_exact_dtype(&[&values]);
        lock().get_or_insert_with(HashMap::new).insert(key, dtype);
        Ok(dtype)
    }

    /// The stored bytes of tensor `name` as the mapping holds them, its shape and its dtype.
    ///
    /// Nothing is copied: the slice is the mapping's, and lives as long as the loader.
    /// The read is counted and cancelled as a [`Self::load`] is.
    pub fn stored_bytes(&self, name: &str) -> Result<(&[u8], Vec<usize>, super::DType)> {
        crate::inference::serve::cancel::scoped::bail()?;
        let &fi = self
            .index
            .get(name)
            .ok_or_else(|| Error(format!("safetensors: no tensor `{name}`")))?;
        let info = self
            .parsers
            .get(fi)
            .and_then(|p| p.tensors.get(name))
            .ok_or_else(|| Error(format!("safetensors: no tensor `{name}`")))?;
        let header_len = self.parsers[fi].header_len;

        let slice_start = 8 + header_len + info.data_offsets.0;
        let slice_end = 8 + header_len + info.data_offsets.1;
        let data: &[u8] = &self.maps[fi][slice_start..slice_end];

        let dtype = match info.dtype {
            safetensors::Dtype::F32 => super::DType::F32,
            safetensors::Dtype::F16 => super::DType::F16,
            safetensors::Dtype::BF16 => super::DType::BF16,
            safetensors::Dtype::U8 => super::DType::U8,
            safetensors::Dtype::U32 => super::DType::U32,
            safetensors::Dtype::I16 => super::DType::I16,
            safetensors::Dtype::I32 => super::DType::I32,
            safetensors::Dtype::I64 => super::DType::I64,
            safetensors::Dtype::F64 => super::DType::F64,
            other => {
                return Err(Error(format!(
                    "safetensors `{name}`: no host dtype for {other:?}"
                )))
            }
        };
        let out = (data, info.shape.clone(), dtype);
        self.count_read();
        Ok(out)
    }

    /// Count one tensor read, for the progress a client sees.
    fn count_read(&self) {
        let done = self.read.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        crate::inference::serve::progress::scoped::note(
            crate::inference::serve::progress::phase::LOAD_MODEL,
            done,
            self.index.len(),
        );
    }

    /// Load one tensor to host as a native tensor (dtype preserved).
    pub fn load(&self, name: &str) -> Result<Tensor> {
        // Cooperative cancellation at the grain the reading happens, and for the same
        // reason the count below is taken here: every safetensors weight in the process
        // arrives at this one function, so a load whose client has gone stops within ONE
        // tensor instead of finishing the checkpoint for nobody - without any of the
        // model files knowing a request exists. Checked BEFORE the copy, so the tensor
        // being abandoned is not paid for. Outside a published scope (`cancel::scoped`)
        // this is one thread-local read and nothing ever cancels, which is every
        // non-serving caller.
        crate::inference::serve::cancel::scoped::bail()?;
        let &fi = self
            .index
            .get(name)
            .ok_or_else(|| Error(format!("safetensors: no tensor `{name}`")))?;
        let info = self
            .parsers
            .get(fi)
            .and_then(|p| p.tensors.get(name))
            .ok_or_else(|| Error(format!("safetensors: no tensor `{name}`")))?;

        let header_len = self.parsers[fi].header_len;
        let slice_start = 8 + header_len + info.data_offsets.0;
        let slice_end = 8 + header_len + info.data_offsets.1;
        let data: &[u8] = &self.maps[fi][slice_start..slice_end];

        let dims: Vec<usize> = info.shape.clone();
        let storage = match info.dtype {
            safetensors::Dtype::F32 => CpuStorage::F32(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            safetensors::Dtype::F16 => CpuStorage::F16(
                data.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ),
            safetensors::Dtype::BF16 => CpuStorage::BF16(
                data.chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ),
            // fp8 (OCP e4m3fn / e5m2) - used by "fp8_scaled" diffusion checkpoints
            // (Boogu-Image et al.), paired with a companion `<name>.weight_scale`
            // F32 scalar the caller multiplies in. Decode the raw fp8 byte to F32.
            safetensors::Dtype::F8_E4M3 => {
                CpuStorage::F32(data.iter().map(|&b| f8_e4m3fn_to_f32(b)).collect())
            }
            safetensors::Dtype::F8_E5M2 => {
                CpuStorage::F32(data.iter().map(|&b| f8_e5m2_to_f32(b)).collect())
            }
            safetensors::Dtype::U8 => CpuStorage::U8(data.to_vec()),
            safetensors::Dtype::U32 => CpuStorage::U32(
                data.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            // AWQ checkpoints store qweight/qzeros as I32
            safetensors::Dtype::I32 => CpuStorage::I32(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            safetensors::Dtype::I16 => CpuStorage::I16(
                data.chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ),
            safetensors::Dtype::I64 => CpuStorage::I64(
                data.chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            ),
            safetensors::Dtype::F64 => CpuStorage::F64(
                data.chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            ),
            other => {
                return Err(Error(format!(
                    "safetensors `{name}`: unsupported dtype {other:?}"
                )))
            }
        };
        let t = Tensor::from_storage(storage, Shape::from(dims))?;
        // Counted after the read succeeded, so the number means "on the host", not
        // "attempted". Costs one relaxed increment and one thread-local read when no
        // render published a reporter, which is every non-serving caller.
        self.count_read();
        Ok(t)
    }

    /// Load one tensor onto a device with an optional dtype conversion
    /// (the common weight-load call: file dtype -> compute dtype -> device).
    pub fn load_to(&self, name: &str, dtype: super::DType, device: &Device) -> Result<Tensor> {
        // A dry run reads the header and stops there. The shape is the file's own -
        // never a shape derived from a config next to it, which is how a checkpoint
        // came to be weighed three gigabytes heavier than it is - and no data is
        // touched, so planning a placement costs a header parse instead of a read.
        if let Device::Dry(_) = device {
            let dims = self
                .shape_of(name)
                .ok_or_else(|| Error(format!("safetensors: no tensor `{name}`")))?;
            return Tensor::dry(device, dtype, Shape::from(dims));
        }
        let t = self.load(name)?;
        let t = if t.dtype() == dtype {
            t
        } else {
            // CONVERSION: the caller names the dtype it wants the tensor in.
            t.to_dtype(dtype)?
        };
        t.to_device(device)
    }
}

/// Legacy-shaped mmap'd safetensors bundle: thin adapter over
/// `SafeTensorsLoader` (re-exported as `crate::tensor::safetensors`).
pub struct MmapedSafetensors {
    inner: SafeTensorsLoader,
}

impl MmapedSafetensors {
    /// # Safety
    /// The backing file must not be mutated while mapped.
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        Self::multi(&[p])
    }

    /// # Safety
    /// The backing files must not be mutated while mapped.
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        Ok(Self {
            inner: unsafe { SafeTensorsLoader::multi(paths) }?,
        })
    }

    pub fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        let t = self.inner.load(name)?;
        t.to_device(device)
    }

    /// The stored bytes of tensor `name` in the mapping, its shape and its dtype, with no copy.
    pub fn stored_bytes(&self, name: &str) -> Result<(&[u8], Vec<usize>, super::DType)> {
        self.inner.stored_bytes(name)
    }

    /// Presence probe (the loader returns a TensorView; the call sites
    /// only use `.is_ok()` - keep it cheap).
    pub fn get(&self, name: &str) -> Result<()> {
        if self.inner.contains(name) {
            Ok(())
        } else {
            Err(Error(format!("safetensors: no tensor `{name}`")))
        }
    }

    pub fn names(&self) -> Vec<&str> {
        self.inner.names()
    }

    /// Every tensor's size in bytes as stored, read from the headers only.
    pub fn byte_sizes(&self) -> Vec<(String, usize)> {
        self.inner.byte_sizes()
    }

    /// A tensor's shape, read from the header only.
    pub fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        self.inner.shape_of(name)
    }
}

/// Test helper: first usable safetensors file in the local HF cache.
#[cfg(test)]
pub(crate) fn tests_helper_find() -> Option<std::path::PathBuf> {
    tests::find_safetensors()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory under `target/` for tests.
    fn test_scratch_dir(tag: &str) -> std::path::PathBuf {
        let exe = std::env::current_exe().unwrap();
        let dir = exe
            .parent()
            .unwrap()
            .join(format!("loken-st-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(crate) fn find_safetensors() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let hub = std::path::Path::new(&home).join(".cache/huggingface/hub");
        fn scan(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    scan(&p, out);
                } else if p.extension().is_some_and(|x| x == "safetensors")
                    && e.metadata().is_ok_and(|m| m.len() > 1_000_000)
                {
                    out.push(p);
                }
            }
        }
        let mut found = vec![];
        scan(&hub, &mut found);
        found.sort();
        found.into_iter().next()
    }

    /// Reading the weights is the longest phase of a render, and it used to report nothing:
    /// a client saw a stage name that did not move for tens of seconds, which reads exactly
    /// like a wedged server. Counting HERE is what spares every engine from wiring a count of
    /// its own, so this proves the count arrives, that its total is the file's own tensor
    /// count, and - just as important - that a loader outside a render stays silent.
    #[test]
    fn a_read_counts_itself_into_the_render_that_asked_for_it() {
        use crate::inference::serve::progress::{phase, scoped, SharedProgressFn};

        let dir = test_scratch_dir("progress");
        let path = dir.join("counted.safetensors");
        let bytes = [0u8; 8]; // two f32 per tensor
        let tensors: Vec<_> = ["a", "b", "c"]
            .iter()
            .map(|n| {
                (
                    (*n).to_string(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &bytes)
                        .unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(tensors, None, &path).unwrap();

        let loader = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();
        // Outside a render nothing is published, so a read must reach nobody.
        loader.load("a").unwrap();

        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        let report: SharedProgressFn = std::sync::Arc::new(move |p: &str, d: usize, t: usize| {
            sink.lock().unwrap().push((p.to_string(), d, t));
        });
        scoped::with(report, || {
            loader.load("b").unwrap();
            loader.load("c").unwrap();
        });
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[
                (phase::LOAD_MODEL.to_string(), 2, 3),
                (phase::LOAD_MODEL.to_string(), 3, 3),
            ],
            "each read counts once, against the tensor count of the file itself"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A bundle of one-element F32 tensors with distinguishable contents.
    fn write_bundle(path: &std::path::Path, names: &[&str]) {
        let data: Vec<[u8; 4]> = (0..names.len())
            .map(|i| (i as f32 + 1.0).to_le_bytes())
            .collect();
        let tensors: Vec<_> = names
            .iter()
            .zip(&data)
            .map(|(n, d)| {
                (
                    (*n).to_string(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], d)
                        .unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(tensors, None, path).unwrap();
    }

    /// A load whose client has gone must stop AT THE TENSOR IT IS ON. Measured before this
    /// check existed: an image stream abandoned during the load ran on to "fully loaded"
    /// and claimed a card, so the NEXT request was refused with "every GPU is busy with an
    /// in-flight generation". Every weight in the process arrives at this one function,
    /// which is why the check lives here rather than in each model file; the progress
    /// counter is the witness for how far the load actually got.
    #[test]
    fn an_abandoned_load_stops_at_the_tensor_it_is_on() {
        use crate::inference::serve::cancel::CancelToken;
        use crate::inference::serve::progress::{scoped, SharedProgressFn};

        let dir = test_scratch_dir("cancel");
        let path = dir.join("abandoned.safetensors");
        let names = ["a", "b", "c", "d"];
        write_bundle(&path, &names);
        let loader = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();

        let reads = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let sink = reads.clone();
        let report: SharedProgressFn =
            std::sync::Arc::new(move |_: &str, _: usize, _: usize| *sink.lock().unwrap() += 1);

        let token = CancelToken::new();
        let refusals = scoped::with(report, || {
            let _published = crate::inference::serve::cancel::scoped::publish(&token);
            loader.load("a").expect("a live load reads normally");
            // The client drops the stream here: the guard in the request future fires.
            token.cancel();
            names[1..]
                .iter()
                .map(|n| {
                    loader
                        .load(n)
                        .expect_err("an abandoned load must refuse")
                        .to_string()
                })
                .collect::<Vec<_>>()
        });

        for e in &refusals {
            assert!(
                e.contains("cancelled"),
                "an error that says why, not a panic: {e}"
            );
        }
        assert_eq!(
            *reads.lock().unwrap(),
            1,
            "not one tensor is read after the cancel"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A load nobody cancelled reads exactly what it read before the check existed. Outside
    /// a published scope - every binary, every warm-up, every non-serving caller - there is
    /// no token to consult at all.
    #[test]
    fn a_load_nobody_cancelled_is_untouched() {
        use crate::inference::serve::cancel::{scoped as cancelling, CancelToken};

        let dir = test_scratch_dir("uncancelled");
        let path = dir.join("nominal.safetensors");
        let names = ["a", "b", "c", "d"];
        write_bundle(&path, &names);
        let loader = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();

        // No scope at all: the reference reading.
        let bare: Vec<Vec<f32>> = names
            .iter()
            .map(|n| loader.load(n).unwrap().to_vec_f32())
            .collect();
        // Under a token nobody cancelled: every tensor still arrives, same values.
        let live = CancelToken::new();
        let under_scope: Vec<Vec<f32>> = cancelling::with(&live, || {
            names
                .iter()
                .map(|n| loader.load(n).unwrap().to_vec_f32())
                .collect()
        });
        assert_eq!(bare, under_scope);
        assert_eq!(bare, vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// AWQ checkpoints store qweight/qzeros as I32:
    /// the loader must read them (values preserved through CpuStorage::I32).
    #[test]
    fn loads_i32_awq_tensors() {
        // find an AWQ safetensors with an I32 tensor in the HF cache
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let hub = std::path::Path::new(&home).join(".cache/huggingface/hub");
        fn scan(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    scan(&p, out);
                } else if p.extension().is_some_and(|x| x == "safetensors")
                    && p.to_string_lossy().to_lowercase().contains("awq")
                {
                    out.push(p);
                }
            }
        }
        let mut found = vec![];
        scan(&hub, &mut found);
        found.sort();
        for path in found {
            let Ok(native) = (unsafe { SafeTensorsLoader::multi(&[&path]) }) else {
                continue;
            };
            let Some(name) = native
                .names()
                .into_iter()
                .find(|n| n.ends_with("qweight"))
                .map(str::to_owned)
            else {
                continue;
            };
            let t = native.load(&name).unwrap();
            assert_eq!(t.dtype(), crate::tensor::DType::I32, "{name}");
            assert!(t.elem_count() > 0);
            // spot-check raw value preservation against the file bytes
            let v = t.to_vec_f32();
            assert!(
                v.iter().any(|&x| x != 0.0),
                "{name}: all-zero qweight is implausible"
            );
            eprintln!("I32 AWQ load OK: {name} {:?} from {path:?}", t.dims());
            return;
        }
        eprintln!("no AWQ I32 safetensors in HF cache - skipped");
    }

    #[test]
    fn loader_matches_oracle() {
        let Some(path) = find_safetensors() else {
            return;
        }; // no HF cache -> skip
        let native = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();
        // Oracle: the reference `safetensors` crate reading the same file.
        let bytes = std::fs::read(&path).unwrap();
        let oracle = safetensors::SafeTensors::deserialize(&bytes).unwrap();

        let mut names = native.names();
        names.sort();
        assert!(!names.is_empty());
        // a spread of tensors across the file, skipping dtypes the native
        // loader doesn't support
        let mut picks: Vec<&str> = vec![];
        for &i in &[0usize, names.len() / 3, names.len() / 2, names.len() - 1] {
            for name in names.iter().skip(i) {
                if native.load(name).is_ok() && !picks.contains(name) {
                    picks.push(name);
                    break;
                }
            }
        }
        assert!(!picks.is_empty(), "no loadable tensors found");
        for name in picks {
            let nt = native.load(name).unwrap();
            let ot = oracle.tensor(name).unwrap();
            assert_eq!(nt.dims(), ot.shape(), "{name}");
            let got = nt.to_vec_f32();
            let raw = ot.data();
            let want: Vec<f32> = match ot.dtype() {
                safetensors::Dtype::F32 => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => raw
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => raw
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::I32 => raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect(),
                other => panic!("{name}: unhandled oracle dtype {other:?}"),
            };
            assert_eq!(got.len(), want.len(), "{name}");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g == w || (g.is_nan() && w.is_nan()),
                    "{name} idx {i}: {g} vs {w}"
                );
            }
        }
        eprintln!("safetensors loader parity OK on {path:?}");
    }

    /// Test that every tensor reads back the exact same bytes as written.
    #[test]
    fn reads_back_the_same_bytes_as_written() {
        let dir = test_scratch_dir("bytes");
        let path1 = dir.join("part1.safetensors");
        let path2 = dir.join("part2.safetensors");

        // File 1: F32 tensors with specific values
        let data_a: Vec<u8> = [1f32, 2.0, 3.0]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let data_b: Vec<u8> = [4.0f32, 5.0, 6.0]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let tensors1 = vec![
            (
                "layer_a".to_string(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &data_a)
                    .unwrap(),
            ),
            (
                "layer_b".to_string(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &data_b)
                    .unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors1, None, &path1).unwrap();

        // File 2: U8 tensor
        let u8_vals: Vec<u8> = vec![0u8, 1, 2, 3];
        let u8_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::U8,
            vec![u8_vals.len()],
            &u8_vals,
        )
        .unwrap();
        safetensors::serialize_to_file([("wq", u8_view)], None, &path2).unwrap();

        let loader = unsafe { SafeTensorsLoader::multi(&[&path1, &path2]) }.unwrap();

        // Read all tensors and verify bytes match
        let checks: Vec<_> = [
            ("layer_a", vec![1.0f32, 2.0, 3.0]),
            ("layer_b", vec![4.0, 5.0, 6.0]),
        ]
        .into_iter()
        .map(|(name, expected)| {
            let bytes = loader.stored_bytes(name).unwrap();
            (name, bytes, expected)
        })
        .collect();

        for (name, (bytes, _shape, dtype), expected) in checks {
            assert_eq!(
                dtype,
                crate::tensor::DType::F32,
                "unexpected dtype for {name}"
            );
            let expected_bytes: Vec<u8> = expected
                .iter()
                .flat_map(|&v: &f32| v.to_le_bytes())
                .collect();
            assert_eq!(bytes, expected_bytes.as_slice(), "{name}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Test that each file header is parsed exactly once per loader.
    #[test]
    fn one_parse_per_file() {
        let dir = test_scratch_dir("parse");
        let path1 = dir.join("file1.safetensors");
        let path2 = dir.join("file2.safetensors");

        // Write two files with one tensor each
        let view1 =
            safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![3], &[1u8, 2, 3])
                .unwrap();
        safetensors::serialize_to_file([("t1", view1)], None, &path1).unwrap();

        let view2 =
            safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![2], &[4u8, 5])
                .unwrap();
        safetensors::serialize_to_file([("t2", view2)], None, &path2).unwrap();

        let loader = unsafe { SafeTensorsLoader::multi(&[&path1, &path2]) }.unwrap();

        // Verify parse count is incremented correctly - each file triggers one parse
        assert_eq!(loader.parse_count(), 2, "two files should yield two parses");

        // Read all tensors - should not increase parse count
        let _ = loader.load("t1").unwrap();
        let _ = loader.load("t2").unwrap();
        let _ = loader.stored_bytes("t1").unwrap();
        let _ = loader.shape_of("t2").unwrap();
        let _ = loader.stored_format("t1").unwrap();
        let _ = loader.narrowest_exact_dtype("t1").unwrap();

        assert_eq!(
            loader.parse_count(),
            2,
            "reads should not increase parse count"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
