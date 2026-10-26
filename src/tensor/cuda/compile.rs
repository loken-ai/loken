//! Run-time compilation of the kernels, to the form the card's driver loads.
//!
//! The kernels are compiled to PTX, which the driver turns into machine code for the card and
//! keeps in its own cache. A driver older than the NVRTC library cannot read the PTX that library
//! writes, so for such a driver an NVIDIA card's kernels are compiled to the card's machine code
//! instead, which any driver of the same major version loads. That machine code is kept in the
//! user's cache directory, keyed by the source, the options and the NVRTC version, so a later
//! start loads it instead of compiling it again.

use crate::tensor::gpu_runtime::nvrtc::{CompileOptions, Ptx};

/// `src` compiled for the target `opts.arch` names: PTX, or the machine code of an NVIDIA card
/// when the driver is older than the NVRTC library.
pub fn compile(src: impl AsRef<str>, opts: CompileOptions) -> Result<Ptx, String> {
    #[cfg(feature = "cuda")]
    if let Some(sm) = opts.arch.and_then(machine_arch) {
        if driver_older_than_nvrtc() {
            return cubin(src.as_ref(), opts, &sm);
        }
    }
    crate::tensor::gpu_runtime::nvrtc::compile_ptx_with_opts(src, opts).map_err(|e| e.to_string())
}

/// The machine architecture of an NVIDIA virtual one: `sm_89` for `compute_89`.
#[cfg(feature = "cuda")]
fn machine_arch(arch: &str) -> Option<String> {
    arch.strip_prefix("compute_").map(|cc| format!("sm_{cc}"))
}

/// Whether the driver supports an older CUDA than the NVRTC library the process opened.
#[cfg(feature = "cuda")]
pub fn driver_older_than_nvrtc() -> bool {
    static OLDER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OLDER.get_or_init(|| {
        let mut driver = 0;
        let (mut major, mut minor) = (0, 0);
        let read = unsafe {
            cudarc::driver::sys::cuDriverGetVersion(&mut driver)
                == cudarc::driver::sys::CUresult::CUDA_SUCCESS
                && cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor)
                    == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS
        };
        let older = read && driver < major * 1000 + minor * 10;
        if older {
            tracing::info!(
                "the driver supports CUDA {}.{} and NVRTC is {major}.{minor}: kernels are compiled \
                 to the cards' machine code",
                driver / 1000,
                driver % 1000 / 10
            );
        }
        older
    })
}

/// `src` compiled to the machine code of `sm`, with the options of `opts`: read from the cache
/// when an earlier compile of the same source and options left it there.
#[cfg(feature = "cuda")]
fn cubin(src: &str, opts: CompileOptions, sm: &str) -> Result<Ptx, String> {
    let mut args: Vec<String> = Vec::new();
    for (flag, v) in [
        ("ftz", opts.ftz),
        ("prec-sqrt", opts.prec_sqrt),
        ("prec-div", opts.prec_div),
        ("fmad", opts.fmad),
    ] {
        if let Some(v) = v {
            args.push(format!("--{flag}={v}"));
        }
    }
    if opts.use_fast_math == Some(true) {
        args.push("--fmad=true".into());
    }
    if let Some(n) = opts.maxrregcount {
        args.push(format!("--maxrregcount={n}"));
    }
    args.extend(
        opts.include_paths
            .iter()
            .map(|p| format!("--include-path={p}")),
    );
    args.push(format!("--gpu-architecture={sm}"));
    args.extend(opts.options.iter().cloned());

    let cached = cache_path(src, &args);
    if let Some(image) = cached.as_ref().and_then(|p| std::fs::read(p).ok()) {
        return Ok(Ptx::from_binary(image));
    }
    let image = nvrtc_cubin(src, opts.name.as_deref(), &args)?;
    if let Some(path) = cached {
        store(&path, &image);
    }
    Ok(Ptx::from_binary(image))
}

/// Where the machine code of `src` compiled with `args` is kept: a file named by the digest of
/// the NVRTC version, the arguments and the source. `None` without a user cache directory.
#[cfg(feature = "cuda")]
fn cache_path(src: &str, args: &[String]) -> Option<std::path::PathBuf> {
    use sha2::{Digest, Sha256};
    let (mut major, mut minor) = (0, 0);
    let _ = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    let mut h = Sha256::new();
    h.update(format!("nvrtc {major}.{minor}\0"));
    for a in args {
        h.update(a.as_bytes());
        h.update([0]);
    }
    h.update(src.as_bytes());
    let name: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Some(
        dirs::cache_dir()?
            .join("loken")
            .join("kernels")
            .join(format!("{name}.cubin")),
    )
}

/// Writes `image` to `path` through a temporary file renamed into place, so a reader never sees
/// a partial file. A failure only costs the next start a compile.
#[cfg(feature = "cuda")]
fn store(path: &std::path::Path, image: &[u8]) {
    let Some(dir) = path.parent() else { return };
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(&tmp, image))
        .and_then(|()| std::fs::rename(&tmp, path));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        tracing::warn!("kernel cache: cannot write {}: {e}", path.display());
    }
}

/// NVRTC's machine code for `src` compiled with `args`.
#[cfg(feature = "cuda")]
fn nvrtc_cubin(src: &str, name: Option<&str>, args: &[String]) -> Result<Vec<u8>, String> {
    use cudarc::nvrtc::{result, sys};
    let source = std::ffi::CString::new(src).map_err(|e| e.to_string())?;
    let name = name
        .map(std::ffi::CString::new)
        .transpose()
        .map_err(|e| e.to_string())?;
    let prog = result::create_program(&source, name.as_deref()).map_err(|e| format!("{e:?}"))?;
    let built = (|| {
        if let Err(e) = unsafe { result::compile_program(prog, args) } {
            let log = unsafe { result::get_program_log(prog) }
                .map(|l| {
                    unsafe { std::ffi::CStr::from_ptr(l.as_ptr()) }
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default();
            return Err(format!("{e:?}: {log}"));
        }
        let mut size = 0usize;
        unsafe { sys::nvrtcGetCUBINSize(prog, &mut size) }
            .result()
            .map_err(|e| format!("{e:?}"))?;
        let mut image = vec![0u8; size];
        unsafe { sys::nvrtcGetCUBIN(prog, image.as_mut_ptr().cast()) }
            .result()
            .map_err(|e| format!("{e:?}"))?;
        Ok(image)
    })();
    let _ = unsafe { result::destroy_program(prog) };
    built
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn the_cache_names_the_source_and_the_options() {
        let args = |a: &str| vec![a.to_string()];
        let Some(a) = cache_path("src", &args("-x")) else {
            return;
        };
        assert_eq!(cache_path("src", &args("-x")), Some(a.clone()));
        assert_ne!(cache_path("src2", &args("-x")), Some(a.clone()));
        assert_ne!(cache_path("src", &args("-y")), Some(a));
    }

    #[test]
    fn each_nvidia_card_loads_the_machine_code_nvrtc_writes_for_it() {
        const SRC: &str = "extern \"C\" __global__ void one(float* x) { x[threadIdx.x] = 1.0f; }";
        for ordinal in 0..8 {
            let Some(sm) = crate::inference::quantized_cuda::nvrtc_arch_of_ordinal(ordinal)
                .and_then(machine_arch)
            else {
                continue;
            };
            let args = vec![format!("--gpu-architecture={sm}")];
            let image = nvrtc_cubin(SRC, None, &args).expect("nvrtc writes the machine code");
            let dev = crate::tensor::cuda::CudaDevice::new(ordinal).expect("the card opens");
            let module = dev
                .context()
                .load_module(Ptx::from_binary(image))
                .expect("the driver loads the machine code");
            module
                .load_function("one")
                .expect("the kernel is in the module");
        }
    }
}
