//! Run-time kernel compilation under the NVRTC names, done by hipRTC. The output is a code
//! object for the architecture the runtime reports for a device, which
//! [`CudaContext::load_module`](super::driver::CudaContext::load_module) loads.

pub mod sys {
    use std::ffi::{c_char, c_int};

    crate::tensor::gpu_runtime::hip::dynamic_api! {
        /// The entry points of the hipRTC library.
        pub(crate) struct Hiprtc from "hiprtc" {
            fn hiprtcGetErrorString(r: c_int) -> *const c_char;
            fn hiprtcCreateProgram(
                prog: *mut *mut std::ffi::c_void,
                src: *const c_char,
                name: *const c_char,
                num_headers: c_int,
                headers: *const *const c_char,
                include_names: *const *const c_char,
            ) -> c_int;
            fn hiprtcCompileProgram(
                prog: *mut std::ffi::c_void,
                num_options: c_int,
                options: *const *const c_char,
            ) -> c_int;
            fn hiprtcGetProgramLogSize(prog: *mut std::ffi::c_void, n: *mut usize) -> c_int;
            fn hiprtcGetProgramLog(prog: *mut std::ffi::c_void, log: *mut c_char) -> c_int;
            fn hiprtcGetCodeSize(prog: *mut std::ffi::c_void, n: *mut usize) -> c_int;
            fn hiprtcGetCode(prog: *mut std::ffi::c_void, code: *mut c_char) -> c_int;
            fn hiprtcDestroyProgram(prog: *mut *mut std::ffi::c_void) -> c_int;
        }
    }

    /// Whether the hipRTC library opens.
    pub unsafe fn is_culib_present() -> bool {
        Hiprtc::present()
    }
}

pub mod safe {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::path::PathBuf;

    use super::sys::Hiprtc;
    use crate::tensor::gpu_runtime::hip::driver::{result, sys as driver_sys};

    /// A hipRTC status other than success, or one of the codes below for a failure before it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NvrtcError(pub c_int);

    impl NvrtcError {
        /// The library does not open or lacks an entry point.
        pub const LIBRARY_MISSING: Self = Self(-1);
        /// No device reports an architecture to compile for.
        pub const NO_TARGET: Self = Self(-2);
    }

    impl std::fmt::Display for NvrtcError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let text = Hiprtc::get()
                .and_then(|api| api.hiprtcGetErrorString)
                .map(|g| unsafe { g(self.0) })
                .filter(|p| !p.is_null())
                .map(|p| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
            match text {
                Some(t) => write!(f, "hiprtc error {}: {t}", self.0),
                None => write!(f, "hiprtc error {}", self.0),
            }
        }
    }

    /// Why a compilation failed.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum CompileError {
        CreationError(NvrtcError),
        CompileError {
            nvrtc: NvrtcError,
            options: Vec<String>,
            log: CString,
        },
        GetLogError(NvrtcError),
        GetPtxError(NvrtcError),
        DestroyError(NvrtcError),
    }

    impl std::fmt::Display for CompileError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{self:?}")
        }
    }

    impl std::error::Error for CompileError {}

    /// The options of a compilation, under NVRTC's names. Each one is given its hipRTC
    /// spelling; those without a counterpart are left out.
    #[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
    pub struct CompileOptions {
        pub ftz: Option<bool>,
        pub prec_sqrt: Option<bool>,
        pub prec_div: Option<bool>,
        pub fmad: Option<bool>,
        pub options: Vec<String>,
        pub use_fast_math: Option<bool>,
        pub maxrregcount: Option<usize>,
        pub include_paths: Vec<String>,
        pub arch: Option<&'static str>,
        pub name: Option<String>,
    }

    /// The CUDA headers the kernels include, supplied with their HIP meaning, and the HIP
    /// headers they include whose content hipRTC already puts before every source.
    const HEADERS: [(&str, &str); 7] = [
        ("hip/hip_runtime.h", "#pragma once\n"),
        ("hip/hip_fp16.h", "#pragma once\n"),
        (
            "loken_hip.h",
            include_str!("../../../../cuda/hip/loken_hip.h"),
        ),
        ("cuda.h", include_str!("../../../../cuda/hip/cuda.h")),
        (
            "cuda_runtime.h",
            include_str!("../../../../cuda/hip/cuda_runtime.h"),
        ),
        (
            "cuda_fp16.h",
            include_str!("../../../../cuda/hip/cuda_fp16.h"),
        ),
        (
            "cuda_bf16.h",
            include_str!("../../../../cuda/hip/cuda_bf16.h"),
        ),
    ];

    impl CompileOptions {
        /// The hipRTC options for a compilation targeting `target`.
        pub(crate) fn build(self, target: &str) -> Vec<String> {
            // The CUDA spellings the kernels use reach them through the mapping header, put
            // before every source as the ahead-of-time build does.
            let mut out = vec![
                format!("--offload-arch={target}"),
                "-include".into(),
                "loken_hip.h".into(),
            ];
            if self.ftz == Some(true) {
                out.push("-fgpu-flush-denormals-to-zero".into());
            }
            match (self.use_fast_math, self.fmad) {
                (Some(true), _) => out.push("-ffast-math".into()),
                (_, Some(true)) => out.push("-ffp-contract=fast".into()),
                (_, Some(false)) => out.push("-ffp-contract=off".into()),
                _ => {}
            }
            // A directory holding a header supplied here would shadow it with the CUDA one.
            for path in self.include_paths.iter().filter(|p| {
                !HEADERS
                    .iter()
                    .any(|(name, _)| std::path::Path::new(p).join(name).is_file())
            }) {
                out.push(format!("-I{path}"));
            }
            for option in self.options {
                match option.as_str() {
                    "--use_fast_math" | "-use_fast_math" => out.push("-ffast-math".into()),
                    "-default-device"
                    | "--device-as-default-execution-space"
                    | "-lineinfo"
                    | "--generate-line-info" => {}
                    o if o.starts_with("--std=") => out.push(format!("-std={}", &o[6..])),
                    o if o.starts_with("--gpu-architecture") || o.starts_with("-arch") => {}
                    o if o.starts_with("--include-path=") => out.push(format!("-I{}", &o[15..])),
                    o if o.starts_with("--define-macro=") => out.push(format!("-D{}", &o[15..])),
                    _ => out.push(option),
                }
            }
            out
        }
    }

    /// The tag that opens the text form of a code object, followed by its bytes in hex.
    const CODE_OBJECT_TAG: &str = "//hip-code-object:";

    #[derive(Debug, Clone)]
    enum Kind {
        Src(String),
        File(PathBuf),
        CodeObject(Vec<u8>),
    }

    /// Compiled device code, or text that may hold it. A code object round-trips through
    /// [Ptx::to_src] and [Ptx::from_src] as tagged hex text.
    #[derive(Debug, Clone)]
    pub struct Ptx(Kind);

    fn decode(text: &str) -> Option<Vec<u8>> {
        let hex = text.strip_prefix(CODE_OBJECT_TAG)?.trim_end().as_bytes();
        if hex.len() % 2 != 0 {
            return None;
        }
        hex.chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok())
            .collect()
    }

    fn encode(bytes: &[u8]) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(CODE_OBJECT_TAG.len() + bytes.len() * 2);
        s.push_str(CODE_OBJECT_TAG);
        for b in bytes {
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    impl Ptx {
        /// Text read from a file when it is loaded.
        pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
            Self(Kind::File(path.into()))
        }

        /// Text: a code object's tagged form, or source the device cannot load.
        pub fn from_src<S: Into<String>>(src: S) -> Self {
            let src = src.into();
            match decode(&src) {
                Some(bytes) => Self(Kind::CodeObject(bytes)),
                None => Self(Kind::Src(src)),
            }
        }

        /// A code object.
        pub fn from_binary(data: Vec<u8>) -> Self {
            Self(Kind::CodeObject(data))
        }

        /// The text form; empty when a file cannot be read.
        pub fn to_src(&self) -> String {
            match &self.0 {
                Kind::Src(s) => s.clone(),
                Kind::File(p) => std::fs::read_to_string(p).unwrap_or_default(),
                Kind::CodeObject(bytes) => encode(bytes),
            }
        }

        /// The code object's bytes, when this holds one in memory.
        pub fn as_bytes(&self) -> Option<&[u8]> {
            match &self.0 {
                Kind::CodeObject(bytes) => Some(bytes),
                _ => None,
            }
        }

        /// The code object to load: held, tagged in text, or a file's bytes.
        pub(crate) fn code_object(&self) -> Option<Vec<u8>> {
            match &self.0 {
                Kind::CodeObject(bytes) => Some(bytes.clone()),
                Kind::Src(s) => decode(s),
                Kind::File(p) => {
                    let bytes = std::fs::read(p).ok()?;
                    std::str::from_utf8(&bytes)
                        .ok()
                        .and_then(decode)
                        .or(Some(bytes))
                }
            }
        }
    }

    impl<S: Into<String>> From<S> for Ptx {
        fn from(value: S) -> Self {
            Self::from_src(value)
        }
    }

    /// Compiles `src` with default options for the current device.
    pub fn compile_ptx<S: AsRef<str>>(src: S) -> Result<Ptx, CompileError> {
        compile_ptx_with_opts(src, Default::default())
    }

    /// The architecture to compile for: `arch` when a device reports it, otherwise the current
    /// device's.
    fn target(arch: Option<&str>) -> Result<String, NvrtcError> {
        let count = result::device::get_count().unwrap_or(0);
        let names: Vec<String> = (0..count)
            .filter_map(|d| result::device::gcn_arch_name(d).ok())
            .collect();
        if let Some(a) = arch.filter(|a| names.iter().any(|n| n == a)) {
            return Ok(a.to_string());
        }
        let mut current: c_int = 0;
        let _ = unsafe { driver_sys::hip!(hipGetDevice(&mut current)) };
        result::device::gcn_arch_name(current).map_err(|_| NvrtcError::NO_TARGET)
    }

    /// Compiles `src` into a code object for the architecture `opts.arch` names when a device
    /// reports it, the current device's otherwise.
    pub fn compile_ptx_with_opts<S: AsRef<str>>(
        src: S,
        opts: CompileOptions,
    ) -> Result<Ptx, CompileError> {
        let api = Hiprtc::get().ok_or(CompileError::CreationError(NvrtcError::LIBRARY_MISSING))?;
        let missing = CompileError::CreationError(NvrtcError::LIBRARY_MISSING);
        let create = api.hiprtcCreateProgram.ok_or(missing.clone())?;
        let compile = api.hiprtcCompileProgram.ok_or(missing.clone())?;
        let destroy = api.hiprtcDestroyProgram.ok_or(missing.clone())?;
        let target = target(opts.arch).map_err(CompileError::CreationError)?;

        let src =
            CString::new(src.as_ref()).map_err(|_| CompileError::CreationError(NvrtcError(1)))?;
        let name = CString::new(opts.name.clone().unwrap_or_else(|| "kernel.cu".into()))
            .map_err(|_| CompileError::CreationError(NvrtcError(1)))?;
        let header_names: Vec<CString> = HEADERS
            .iter()
            .map(|(n, _)| CString::new(*n).expect("header name"))
            .collect();
        let header_texts: Vec<CString> = HEADERS
            .iter()
            .map(|(_, t)| CString::new(*t).expect("header text"))
            .collect();
        let name_ptrs: Vec<*const c_char> = header_names.iter().map(|c| c.as_ptr()).collect();
        let text_ptrs: Vec<*const c_char> = header_texts.iter().map(|c| c.as_ptr()).collect();

        let mut prog: *mut c_void = std::ptr::null_mut();
        let r = unsafe {
            create(
                &mut prog,
                src.as_ptr(),
                name.as_ptr(),
                HEADERS.len() as c_int,
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if r != 0 {
            return Err(CompileError::CreationError(NvrtcError(r)));
        }

        let mut options = opts.build(&target);
        if let Some(dir) = rocm_include_dir() {
            options.push(format!("-I{}", dir.display()));
        }
        let option_c: Vec<CString> = options
            .iter()
            .filter_map(|o| CString::new(o.as_str()).ok())
            .collect();
        let option_ptrs: Vec<*const c_char> = option_c.iter().map(|c| c.as_ptr()).collect();
        let r = unsafe { compile(prog, option_ptrs.len() as c_int, option_ptrs.as_ptr()) };
        let out = if r != 0 {
            Err(CompileError::CompileError {
                nvrtc: NvrtcError(r),
                options,
                log: program_log(api, prog),
            })
        } else {
            code(api, prog).map(Ptx::from_binary)
        };
        let d = unsafe { destroy(&mut prog) };
        match (out, d) {
            (Ok(_), d) if d != 0 => Err(CompileError::DestroyError(NvrtcError(d))),
            (out, _) => out,
        }
    }

    /// The include directory of the ROCm install the loaded hipRTC library belongs to: the
    /// `include` beside the library's directory, when it holds the HIP headers.
    fn rocm_include_dir() -> Option<PathBuf> {
        static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
        DIR.get_or_init(|| {
            Hiprtc::get()?;
            let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
            maps.lines()
                .filter_map(|l| l.split_whitespace().nth(5))
                .filter(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .is_some_and(|f| f.to_string_lossy().starts_with("libhiprtc.so"))
                })
                .filter_map(|p| std::fs::canonicalize(p).ok())
                .filter_map(|p| Some(p.parent()?.parent()?.join("include")))
                .find(|d| d.join("hip").join("hip_bf16.h").is_file())
        })
        .clone()
    }

    fn program_log(api: &Hiprtc, prog: *mut c_void) -> CString {
        let (Some(size), Some(get)) = (api.hiprtcGetProgramLogSize, api.hiprtcGetProgramLog) else {
            return CString::default();
        };
        let mut n = 0usize;
        if unsafe { size(prog, &mut n) } != 0 || n == 0 {
            return CString::default();
        }
        let mut buf = vec![0u8; n];
        if unsafe { get(prog, buf.as_mut_ptr() as *mut c_char) } != 0 {
            return CString::default();
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        buf.truncate(end);
        CString::new(buf).unwrap_or_default()
    }

    fn code(api: &Hiprtc, prog: *mut c_void) -> Result<Vec<u8>, CompileError> {
        let missing = CompileError::GetPtxError(NvrtcError::LIBRARY_MISSING);
        let size = api.hiprtcGetCodeSize.ok_or(missing.clone())?;
        let get = api.hiprtcGetCode.ok_or(missing)?;
        let mut n = 0usize;
        let r = unsafe { size(prog, &mut n) };
        if r != 0 {
            return Err(CompileError::GetPtxError(NvrtcError(r)));
        }
        let mut buf = vec![0u8; n];
        let r = unsafe { get(prog, buf.as_mut_ptr() as *mut c_char) };
        if r != 0 {
            return Err(CompileError::GetPtxError(NvrtcError(r)));
        }
        Ok(buf)
    }
}

pub use safe::*;
