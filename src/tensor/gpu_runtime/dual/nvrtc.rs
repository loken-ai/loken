//! Run-time kernel compilation over both runtimes: NVRTC for an NVIDIA target, hipRTC for an
//! AMD one. The target is the architecture the options name (`gfx...` is AMD's naming), or the
//! runtime bound on the calling thread when they name none. The compiled code loads on a card of
//! the runtime that compiled it.

pub mod sys {
    /// Whether a runtime's kernel compiler opens.
    ///
    /// # Safety
    /// None: it only opens the libraries.
    pub unsafe fn is_culib_present() -> bool {
        unsafe {
            (super::super::cuda_usable() && cudarc::nvrtc::sys::is_culib_present())
                || crate::tensor::gpu_runtime::hip::nvrtc::sys::is_culib_present()
        }
    }
}

pub mod safe {
    use std::path::PathBuf;

    use super::super::{bound, Runtime};
    use crate::tensor::gpu_runtime::hip::nvrtc as hn;
    use cudarc::nvrtc as cn;

    pub use hn::CompileOptions;

    /// Why a compilation failed, as the compiler that ran it says.
    #[derive(Debug)]
    pub enum CompileError {
        Cuda(cn::CompileError),
        Hip(hn::CompileError),
    }

    impl std::fmt::Display for CompileError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                CompileError::Cuda(e) => write!(f, "{e}"),
                CompileError::Hip(e) => write!(f, "{e}"),
            }
        }
    }

    impl std::error::Error for CompileError {}

    #[derive(Debug, Clone)]
    enum Kind {
        Text(String),
        File(PathBuf),
        Binary(Vec<u8>),
        Cuda(cn::Ptx),
        Hip(hn::Ptx),
    }

    /// Compiled device code, or text that may hold it.
    #[derive(Debug, Clone)]
    pub struct Ptx(Kind);

    impl Ptx {
        /// Text read from a file when it is loaded.
        pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
            Self(Kind::File(path.into()))
        }

        /// Text: PTX for an NVIDIA card, a code object's tagged form for an AMD card.
        pub fn from_src<S: Into<String>>(src: S) -> Self {
            Self(Kind::Text(src.into()))
        }

        /// A compiled image.
        pub fn from_binary(data: Vec<u8>) -> Self {
            Self(Kind::Binary(data))
        }

        /// The text form; empty when a file cannot be read.
        pub fn to_src(&self) -> String {
            match &self.0 {
                Kind::Text(s) => s.clone(),
                Kind::File(p) => std::fs::read_to_string(p).unwrap_or_default(),
                Kind::Binary(b) => hn::Ptx::from_binary(b.clone()).to_src(),
                Kind::Cuda(p) => p.to_src(),
                Kind::Hip(p) => p.to_src(),
            }
        }

        /// The compiled image's bytes, when this holds one in memory.
        pub fn as_bytes(&self) -> Option<&[u8]> {
            match &self.0 {
                Kind::Binary(b) => Some(b),
                Kind::Cuda(p) => p.as_bytes(),
                Kind::Hip(p) => p.as_bytes(),
                Kind::Text(_) | Kind::File(_) => None,
            }
        }

        /// The code as NVIDIA's runtime loads it.
        pub(crate) fn into_cuda(self) -> cn::Ptx {
            match self.0 {
                Kind::Text(s) => cn::Ptx::from_src(s),
                Kind::File(p) => cn::Ptx::from_file(p),
                Kind::Binary(b) => cn::Ptx::from_binary(b),
                Kind::Cuda(p) => p,
                Kind::Hip(p) => cn::Ptx::from_src(p.to_src()),
            }
        }

        /// The code as the HIP runtime loads it.
        pub(crate) fn into_hip(self) -> hn::Ptx {
            match self.0 {
                Kind::Text(s) => hn::Ptx::from_src(s),
                Kind::File(p) => hn::Ptx::from_file(p),
                Kind::Binary(b) => hn::Ptx::from_binary(b),
                Kind::Cuda(p) => hn::Ptx::from_src(p.to_src()),
                Kind::Hip(p) => p,
            }
        }
    }

    impl<S: Into<String>> From<S> for Ptx {
        fn from(value: S) -> Self {
            Self::from_src(value)
        }
    }

    /// The runtime that compiles for `arch`: AMD names its targets `gfx...`.
    fn compiler(arch: Option<&str>) -> Runtime {
        match arch {
            Some(a) if a.starts_with("gfx") => Runtime::Hip,
            Some(_) => Runtime::Cuda,
            None => bound(),
        }
    }

    /// Compiles `src` with default options for the bound runtime's current card.
    pub fn compile_ptx<S: AsRef<str>>(src: S) -> Result<Ptx, CompileError> {
        compile_ptx_with_opts(src, Default::default())
    }

    /// Compiles `src` for the architecture `opts.arch` names, or for the bound runtime's
    /// current card.
    pub fn compile_ptx_with_opts<S: AsRef<str>>(
        src: S,
        opts: CompileOptions,
    ) -> Result<Ptx, CompileError> {
        match compiler(opts.arch) {
            Runtime::Hip => hn::compile_ptx_with_opts(src, opts)
                .map(|p| Ptx(Kind::Hip(p)))
                .map_err(CompileError::Hip),
            Runtime::Cuda => {
                let opts = cn::CompileOptions {
                    ftz: opts.ftz,
                    prec_sqrt: opts.prec_sqrt,
                    prec_div: opts.prec_div,
                    fmad: opts.fmad,
                    options: opts.options,
                    use_fast_math: opts.use_fast_math,
                    maxrregcount: opts.maxrregcount,
                    include_paths: opts.include_paths,
                    arch: opts.arch,
                    name: opts.name,
                };
                cn::compile_ptx_with_opts(src, opts)
                    .map(|p| Ptx(Kind::Cuda(p)))
                    .map_err(CompileError::Cuda)
            }
        }
    }
}

pub use safe::*;
