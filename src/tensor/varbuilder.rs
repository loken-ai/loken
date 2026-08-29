//! Where the weights come from: a prefix-walking loader over safetensors files, or over
//! tensors a caller already holds.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::prefix::Prefix;
use super::safetensors_io::{self, SafeTensorsLoader};
use super::{DType, Device, Error, Result, Shape, Tensor};
use std::sync::Arc;

/// Where a builder reads weights from.
///
/// `Files` is the serving path. `Memory` exists because a caller sometimes already HAS the
/// tensors - a test fixture, or weights assembled rather than read - and the alternative is
/// writing them to a temporary file for the sole purpose of reading them back.
#[derive(Clone)]
enum Source {
    Files(Arc<SafeTensorsLoader>),
    Memory(Arc<std::collections::HashMap<String, Tensor>>),
}

impl Source {
    fn contains(&self, name: &str) -> bool {
        match self {
            Source::Files(l) => l.contains(name),
            Source::Memory(m) => m.contains_key(name),
        }
    }

    fn names(&self) -> Vec<String> {
        match self {
            Source::Files(l) => l.names().into_iter().map(str::to_string).collect(),
            Source::Memory(m) => m.keys().cloned().collect(),
        }
    }

    fn load_to(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        match self {
            Source::Files(l) => l.load_to(name, dtype, device),
            Source::Memory(m) => {
                let t = m.get(name).ok_or_else(|| {
                    Error(format!(
                        "VarBuilder: no tensor `{name}` in the supplied map"
                    ))
                })?;
                let t = if t.dtype() == dtype {
                    t.clone()
                } else {
                    t.to_dtype(dtype)?
                };
                t.to_device(device)
            }
        }
    }
}

/// Rewrites a resolved tensor path before lookup. Lets a module written against
/// one checkpoint dialect read another without duplicating the module: the SDXL
/// checkpoints ship the original LDM names (`encoder.down.0.block.0.conv1`) while
/// the diffusers-shaped VAE code here expects `encoder.down_blocks.0.resnets.0.conv1`.
pub type NameRewrite = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Prefix-walking weight loader: `vb.pp("attn").get(shape, "weight")` reads
/// `attn.weight` at the builder's dtype, onto the builder's device.
#[derive(Clone)]
pub struct VarBuilder {
    loader: Source,
    prefix: Prefix,
    dtype: DType,
    device: Device,
    /// Applied to the FULL path at lookup time, after prefix concatenation.
    rename: Option<NameRewrite>,
}

impl VarBuilder {
    /// # Safety
    /// Same mmap contract as [`SafeTensorsLoader::multi`].
    pub unsafe fn from_files<P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let loader = Source::Files(Arc::new(SafeTensorsLoader::multi(paths)?));
        Ok(Self {
            loader,
            prefix: Prefix::root(),
            dtype,
            device: device.clone(),
            rename: None,
        })
    }

    /// Build over tensors already in hand, rather than over files.
    pub fn from_tensors(
        map: std::collections::HashMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Self {
        Self {
            loader: Source::Memory(Arc::new(map)),
            prefix: Prefix::root(),
            dtype,
            device: device.clone(),
            rename: None,
        }
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        Self {
            loader: self.loader.clone(),
            prefix: self.prefix.join(s),
            dtype: self.dtype,
            device: self.device.clone(),
            rename: self.rename.clone(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Same files/prefix, different load target (dtype/device) - e.g. a big
    /// embedding table kept on host while the rest of the stack lives on GPU.
    pub fn to(&self, dtype: DType, device: &Device) -> Self {
        // A DRY builder stays dry, whatever device is asked for. Loaders stage
        // weights on the host before uploading them - `to(F32, Cpu)`, transpose,
        // cast, `to_device` - and a dry run that honoured the host half would read
        // and rewrite twelve gigabytes to answer a question about placement. The
        // rule is one line and it is the whole of it: a dry run allocates nothing
        // ANYWHERE, host included.
        let device = if self.device.is_dry() {
            self.device.clone()
        } else {
            device.clone()
        };
        Self {
            loader: self.loader.clone(),
            prefix: self.prefix.clone(),
            dtype,
            device,
            rename: self.rename.clone(),
        }
    }

    /// This builder's current path, e.g. `down_blocks.1.attentions.0`.
    ///
    /// A module that needs to be addressed LATER by name - a LoRA looks its targets up
    /// by path - records this at construction, so the name comes from the same source
    /// the weights did instead of being reconstructed by string surgery afterwards.
    pub fn prefix(&self) -> &str {
        self.prefix.as_str()
    }

    /// Read weights through `f`, which maps this module's expected name to the name
    /// the FILE uses. Composes with `pp`: the rewrite sees the full path.
    pub fn with_rename(&self, f: NameRewrite) -> Self {
        Self {
            rename: Some(f),
            ..self.clone()
        }
    }

    /// The path actually looked up in the file, after any rename.
    fn resolved(&self, name: &str) -> String {
        let path = self.prefix.path(name);
        match &self.rename {
            Some(f) => f(&path),
            None => path,
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.loader.contains(&self.resolved(name))
    }

    /// Every tensor name in the backing file(s), unprefixed.
    ///
    /// For files whose CONTENTS decide what to do - a LoRA is a set of arbitrary module
    /// names, not a fixed layout - the names have to be discoverable rather than known
    /// in advance.
    pub fn tensor_names(&self) -> Vec<String> {
        self.loader.names()
    }

    /// Fetch by ABSOLUTE name, bypassing the prefix and shape check.
    pub fn get_by_name(&self, name: &str) -> Result<Tensor> {
        self.loader.load_to(name, self.dtype, &self.device)
    }

    /// Fetch `{prefix}.{name}`, validate shape, coerce dtype, move to device.
    pub fn get<S: Into<super::Shape>>(&self, shape: S, name: &str) -> Result<Tensor> {
        let path = self.resolved(name);
        let t = self.loader.load_to(&path, self.dtype, &self.device)?;
        let shape: super::Shape = shape.into();
        if t.shape() == &shape {
            return Ok(t);
        }
        // A 1x1 convolution and a linear projection are the SAME operation, and
        // checkpoint dialects disagree about which to store: the LDM-era VAEs write
        // their attention projections as `[c, c, 1, 1]` convs where the diffusers
        // shape is `[c, c]`. Accept that when the trailing difference is only
        // unit dimensions - the reshape is a pure view, no data moves.
        let want = shape.dims();
        let got = t.dims();
        if t.elem_count() == shape.elem_count()
            && got.len() > want.len()
            && got[..want.len()] == *want
            && got[want.len()..].iter().all(|d| *d == 1)
        {
            return t.reshape(want.to_vec());
        }
        Err(Error(format!(
            "VarBuilder: shape mismatch for `{path}`: expected {:?}, got {:?}",
            shape.dims(),
            t.dims()
        )))
    }
}
