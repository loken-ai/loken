// Build script for compiling CUDA kernels
// This script runs during the cargo build process when the cuda feature is enabled

fn main() {
    println!("cargo:rerun-if-changed=src/inference/sampling_kernels.cu");
    println!("cargo:rerun-if-changed=src/inference/sampling_kernels.h");

    // Add cuDNN library search path
    #[cfg(feature = "cuda")]
    {
        if cfg!(target_os = "windows") {
            // cuDNN 9.x on Windows installs to a separate directory
            for version in &["v9.20", "v9.10", "v9.9", "v9.8"] {
                for cuda_ver in &["13.2", "12.9", "12.6"] {
                    let cudnn_path = format!(
                        "C:/Program Files/NVIDIA/CUDNN/{}/lib/{}/x64",
                        version, cuda_ver
                    );
                    if std::path::Path::new(&cudnn_path).is_dir() {
                        println!("cargo:rustc-link-search=native={}", cudnn_path);
                    }
                }
            }
        } else {
            // Linux: probe standard cuDNN locations
            let cuda_path = std::env::var("CUDA_PATH")
                .or_else(|_| std::env::var("CUDA_HOME"))
                .unwrap_or_else(|_| "/usr/local/cuda".to_string());

            // Standard CUDA lib directory
            let cuda_lib64 = format!("{}/lib64", cuda_path);
            if std::path::Path::new(&cuda_lib64).is_dir() {
                println!("cargo:rustc-link-search=native={}", cuda_lib64);
            }

            // cuDNN standalone installs (e.g. /usr/local/cuda/cudnn-linux-x86_64-9.x.x.x_cudaXX-archive/lib)
            if let Ok(entries) = std::fs::read_dir(&cuda_path) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("cudnn") && name_str.contains("linux") {
                        let cudnn_lib = entry.path().join("lib");
                        if cudnn_lib.is_dir() {
                            println!("cargo:rustc-link-search=native={}", cudnn_lib.display());
                        }
                    }
                }
            }

            // System library paths
            for path in &["/usr/lib/x86_64-linux-gnu", "/usr/lib64"] {
                if std::path::Path::new(path).is_dir() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }

            // CUDNN_LIB env override
            if let Ok(cudnn_lib) = std::env::var("CUDNN_LIB") {
                if std::path::Path::new(&cudnn_lib).is_dir() {
                    println!("cargo:rustc-link-search=native={}", cudnn_lib);
                }
            }
        }
    }

    // WMMA/IMMA tensor-core kernels that NVRTC cannot JIT-compile (they use
    // `mma.sync` + cuda_runtime.h <<<>>> host launchers). These are relocated
    // out of the vendored fork and compiled here with NVCC into a static lib,
    // then driven via `extern "C"` from inference::quantized_cuda.
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=cuda/q4k_mmvq_imma.cu");
        compile_imma_kernels();
        compile_mmq_kernels();
        compile_marlin_kernels();
        compile_flashdecode_tc_kernels();
    }
}

/// NO PTX FALLBACK. Deliberate, and paid for once.
///
/// A `-gencode=arch=compute_80,code=compute_80` alongside the per-arch SASS looks like
/// free portability: the driver JITs it on any card at or above sm_80, so a binary built
/// here would also run elsewhere. It is not free. On sm_120 the JIT of that PTX produces
/// a kernel that RUNS and returns wrong numbers - the whole Flux transformer went to
/// NaN and every image came out pure black, with no error anywhere in the pipeline.
///
/// Measured, same checkpoint, 512x512:
///
///     with compute_80 PTX      4 981 byte PNG, mean 0.00, std 0.00
///     SASS only               664 921 byte PNG, mean 127.6, std 63.1
///
/// The note on `compile_flashdecode_tc_kernels` below already said this - "the PTX-only
/// compute_80 IMMA build cannot host the 16x16x16 f16 MMA forward-JIT cleanly on
/// sm_120" - and I honoured it there while adding the same PTX to MMQ and Marlin, which
/// run the same class of tensor-core kernel.
///
/// If a binary must run on a card other than the build machine's, the answer is an
/// explicit list of REAL architectures (`code=sm_80,sm_86,sm_89,sm_90,sm_120a`), each
/// compiled and verified - not one low PTX standing in for all of them. A kernel that
/// is absent fails loudly; a kernel that JITs into wrong results does not.
/// Compile the query-head-packed tensor-core flash-DECODE kernel (task ,
/// cuda/flashdecode_tc/) into a static lib. Uses nvcuda::wmma m16n8k16 HMMA +
/// cp.async-class staging, so it needs real per-arch SASS (compute>=90a suffix,
/// like the Marlin build) - the PTX-only compute_80 IMMA build cannot host the
/// 16x16x16 f16 MMA forward-JIT cleanly on sm_120. Falls back to a compute_90a
/// PTX when no GPU is visible at build time. Driven via `extern "C"` from
/// inference::flash_decode_tc.
#[cfg(feature = "cuda")]

/// The gencode flags every kernel family ships with: each REAL architecture the fleet can
/// hold, as SASS, plus the arch-specific 'a' variants where the ISA needs them. One arch -
/// the build machine's - is how a binary stops being copyable: SASS for sm_120a neither runs
/// nor re-JITs on sm_89, and the failure is a kernel that computes nothing, not an error.
#[cfg(feature = "cuda")]
fn portable_gencodes() -> Vec<String> {
    [
        // Turing. A GTX 1650 or an RTX 20-series card has no SASS below this line and cannot
        // JIT the PTX either, which only ever moves UPWARD - so leaving sm_75 out is not a
        // slower card, it is a card that loads nothing at all.
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_120a,code=sm_120a",
        // PTX fallback for architectures that do not exist yet: a card outside the SASS list
        // JITs this instead of failing to load. Known architectures never touch it - exact
        // SASS wins - which is what makes it safe to carry despite the JIT-garbage history:
        // that story was PTX standing ALONE for a real card, not PTX behind a full list.
        "-gencode=arch=compute_80,code=compute_80",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// The subset of [`portable_gencodes`] at or above a compute capability, for a family whose
/// ISA does not exist below it.
///
/// Two instructions draw that line today: bf16 WMMA fragments and the `m16n8k32` integer MMA,
/// both Ampere. A kernel using either is an ACCELERATED variant of work the generic quantised
/// path also does, so a card below the line loses speed, not capability - provided the call
/// site declines instead of launching a kernel that was never compiled for it.
#[cfg(feature = "cuda")]
fn gencodes_from(floor: u32) -> Vec<String> {
    portable_gencodes()
        .into_iter()
        .filter(|g| {
            g.rsplit("code=sm_")
                .next()
                .and_then(|s| s.trim_end_matches('a').parse::<u32>().ok())
                .is_none_or(|cc| cc * 10 >= floor)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn compile_flashdecode_tc_kernels() {
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {nvcc:?}; set CUDA_PATH (needed for flashdecode_tc kernels)");
    }
    let gencodes = portable_gencodes();
    // Both the tensor-core flash-DECODE (Q8 KV) and the tensor-core flash-PREFILL
    // (F16 KV) kernels live in this lib - same WMMA HMMA path, same gencode.
    println!("cargo:rerun-if-changed=cuda/flashdecode_tc");
    println!("cargo:rerun-if-changed=cuda/flash_prefill_f16.cu");
    println!("cargo:rerun-if-changed=cuda/flash_dit_bf16.cu");
    // The DiT kernel's 52 KB static shared-memory tile exceeds the pre-Hopper limit, so it
    // ships for the architectures that can hold it; a card below that never runs a DiT
    // through this path. The LLM kernels ship for every real architecture in the fleet.
    let dit_gencodes: Vec<String> = portable_gencodes()
        .into_iter()
        .filter(|g| g.contains("90a") || g.contains("120a"))
        .collect();
    let srcs = [
        (
            "cuda/flashdecode_tc/flashdecode_tc.cu",
            "flashdecode_tc.o",
            &gencodes,
        ),
        (
            "cuda/flash_prefill_f16.cu",
            "flash_prefill_f16.o",
            &gencodes,
        ),
        // Non-causal BF16 flash attention for the diffusion transformers - same WMMA path,
        // a block-wide K/V tile instead of a per-warp one.
        ("cuda/flash_dit_bf16.cu", "flash_dit_bf16.o", &dit_gencodes),
    ];
    let mut objs = Vec::new();
    for (src, obj_name, gencodes) in srcs {
        let obj = out_dir.join(obj_name);
        let status = Command::new(&nvcc)
            .args([
                "-c",
                src,
                "-o",
                obj.to_str().unwrap(),
                "-O3",
                "--expt-relaxed-constexpr",
                "-std=c++17",
                "-Xcompiler",
                "-fPIC",
            ])
            .args(gencodes)
            .status()
            .unwrap_or_else(|e| panic!("failed to run nvcc for {src}: {e}"));
        assert!(status.success(), "nvcc failed for {src}");
        objs.push(obj);
    }

    let lib = out_dir.join("libloken_flashdecode_tc.a");
    // `ar rcs` REPLACES the members it is given and leaves the rest alone, so an object
    // whose source was deleted stays in the archive and keeps linking - which is exactly
    // how a removed kernel produced a duplicate symbol. Start from nothing.
    let _ = std::fs::remove_file(&lib);
    let mut ar_args: Vec<String> = vec!["rcs".into(), lib.to_str().unwrap().into()];
    ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
    let ar = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("failed to run ar");
    assert!(ar.success(), "ar failed for libloken_flashdecode_tc.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=loken_flashdecode_tc");
}

/// Compile the Marlin W4A16 tensor-core GEMM kernels (vLLM/IST-DASLab lineage,
/// cuda/marlin/) into a static lib. Self-contained: detected-arch SASS like the
/// MMQ build (the kernels select cp.async/mma paths from `__CUDA_ARCH__`),
/// compute_80 PTX fallback when no GPU is visible. Driven via `extern "C"`
/// from tensor/marlin.rs.
#[cfg(feature = "cuda")]
fn compile_marlin_kernels() {
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {nvcc:?}; set CUDA_PATH (needed for Marlin kernels)");
    }

    let gencode_list = portable_gencodes();

    let sources = [
        "cuda/marlin/marlin_awq_f16_m8.cu",
        "cuda/marlin/marlin_awq_f16_m16.cu",
        "cuda/marlin/marlin_awq_f16_m32.cu",
        "cuda/marlin/marlin_launcher.cu",
    ];
    println!("cargo:rerun-if-changed=cuda/marlin");

    // Each unit is a heavy template expansion - compile in parallel.
    let handles: Vec<std::thread::JoinHandle<PathBuf>> = sources
        .iter()
        .map(|src| {
            let src = src.to_string();
            let nvcc = nvcc.clone();
            let out_dir = out_dir.clone();
            let gencode_list = gencode_list.clone();
            std::thread::spawn(move || {
                let stem = std::path::Path::new(&src)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap();
                let obj = out_dir.join(format!("{stem}.o"));
                let status = Command::new(&nvcc)
                    .args(&gencode_list)
                    .args([
                        "-c",
                        &src,
                        "-o",
                        obj.to_str().unwrap(),
                        "-O3",
                        "--expt-relaxed-constexpr",
                        // The launcher references __global__ template
                        // instantiations living in the sibling units; this
                        // restores cross-TU resolution (same flag vLLM uses
                        // for its marlin build).
                        "-static-global-template-stub=false",
                        "-std=c++17",
                        "-I",
                        "cuda/marlin",
                        "-Xcompiler",
                        "-fPIC",
                    ])
                    .status()
                    .unwrap_or_else(|e| panic!("failed to run nvcc for {src}: {e}"));
                assert!(status.success(), "nvcc failed for {src}");
                obj
            })
        })
        .collect();
    let objs: Vec<PathBuf> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let lib = out_dir.join("libloken_marlin.a");
    // `ar rcs` REPLACES the members it is given and leaves the rest alone, so an object
    // whose source was deleted stays in the archive and keeps linking - which is exactly
    // how a removed kernel produced a duplicate symbol. Start from nothing.
    let _ = std::fs::remove_file(&lib);
    let mut ar_args: Vec<String> = vec!["rcs".into(), lib.to_str().unwrap().into()];
    ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
    let ar = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("failed to run ar");
    assert!(ar.success(), "ar failed for libloken_marlin.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=loken_marlin");
}

/// Compile the quantized prefill GEMM (MMQ) kernels into a static lib.
///
/// These tile-template kernels pick their tile geometry from the compiled arch list
/// (`__CUDA_ARCH_LIST__`), so they are built for every architecture in
/// [`portable_gencodes`] as real SASS - NOT for the local card, and with no `compute_80`
/// PTX to JIT forward from. Both of those were true once; see that function's header for
/// the measurement that ended the PTX fallback. Driven via `extern "C"` from
/// tensor/quant.rs.
#[cfg(feature = "cuda")]
fn compile_mmq_kernels() {
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {nvcc:?}; set CUDA_PATH (needed for MMQ kernels)");
    }

    // arch >= 90 needs the 'a' (arch-specific) suffix for its full ISA.
    let gencode_list = portable_gencodes();

    let sources = [
        "cuda/mmq_gguf/mmq_quantize.cu",
        "cuda/mmq_gguf/mmq_instance_q4_0.cu",
        "cuda/mmq_gguf/mmq_instance_q4_1.cu",
        "cuda/mmq_gguf/mmq_instance_q5_0.cu",
        "cuda/mmq_gguf/mmq_instance_q5_1.cu",
        "cuda/mmq_gguf/mmq_instance_q8_0.cu",
        "cuda/mmq_gguf/mmq_instance_q2_k.cu",
        "cuda/mmq_gguf/mmq_instance_q3_k.cu",
        "cuda/mmq_gguf/mmq_instance_q4_k.cu",
        "cuda/mmq_gguf/mmq_instance_q5_k.cu",
        "cuda/mmq_gguf/mmq_instance_q6_k.cu",
    ];
    println!("cargo:rerun-if-changed=cuda/mmq_gguf");

    // The instance files are heavy template expansions (~20s each)  -
    // compile them in parallel.
    let handles: Vec<std::thread::JoinHandle<PathBuf>> = sources
        .iter()
        .map(|src| {
            let src = src.to_string();
            let nvcc = nvcc.clone();
            let out_dir = out_dir.clone();
            let gencode_list = gencode_list.clone();
            std::thread::spawn(move || {
                let stem = std::path::Path::new(&src)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap();
                let obj = out_dir.join(format!("{stem}.o"));
                let status = Command::new(&nvcc)
                    .args(&gencode_list)
                    .args([
                        "-c",
                        &src,
                        "-o",
                        obj.to_str().unwrap(),
                        "-O3",
                        "--expt-relaxed-constexpr",
                        "-std=c++17",
                        "-Xcompiler",
                        "-fPIC",
                    ])
                    .status()
                    .unwrap_or_else(|e| panic!("failed to run nvcc for {src}: {e}"));
                assert!(status.success(), "nvcc failed for {src}");
                obj
            })
        })
        .collect();
    let objs: Vec<PathBuf> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let lib = out_dir.join("libloken_mmq.a");
    // `ar rcs` REPLACES the members it is given and leaves the rest alone, so an object
    // whose source was deleted stays in the archive and keeps linking - which is exactly
    // how a removed kernel produced a duplicate symbol. Start from nothing.
    let _ = std::fs::remove_file(&lib);
    let mut ar_args: Vec<String> = vec!["rcs".into(), lib.to_str().unwrap().into()];
    ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
    let ar = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("failed to run ar");
    assert!(ar.success(), "ar failed for libloken_mmq.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=loken_mmq");
}

/// Compile the relocated tensor-core (IMMA/WMMA) CUDA kernels into a static lib.
/// Emits PTX for `compute_80` (the IMMA `m16n8k32.s8` MMA needs Ampere+), which
/// the driver JITs forward to the actual GPU at load - so one build serves every
/// sm_80+ device without per-arch SASS.
#[cfg(feature = "cuda")]
fn compile_imma_kernels() {
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {nvcc:?}; set CUDA_PATH (needed for IMMA kernels)");
    }

    // All relocated tensor-core .cu files. The MoE ones include the shared
    // `gguf.cuh`, resolved via `-I cuda/moe`. Each file's extern symbols are
    // `loken_`-prefixed to avoid clashing with the upstream kernel set' copies.
    // Ampere drew a line through this set: bf16 WMMA fragments and the `m16n8k32` integer MMA
    // do not exist below sm_80. Each of these is an accelerated variant of work the generic
    // quantised path also does, so a Turing card loses speed here, not capability - provided
    // the call site declines rather than launching a kernel with no code for it.
    let ampere_only = [
        "cuda/q4k_mmvq_imma.cu",
        "cuda/moe/dense_q4k_imma_m8.cu",
        "cuda/moe/dense_q5k_imma_m8.cu",
        "cuda/moe/moe_q4k_imma_m8.cu",
        "cuda/moe/moe_q4k_imma_m8_down.cu",
        "cuda/moe/moe_wmma_gguf.cu",
    ];
    let sources = [
        "cuda/fused_norm_f16.cu",
        "cuda/gptoss_flash_decode.cu",
        "cuda/flash_decode_f16.cu",
        "cuda/flash_decode_tiled_probe.cu",
        "cuda/sampling.cu",
        "cuda/moe/add_rms_norm.cu",
        "cuda/moe/rms_qmatmul.cu",
        "cuda/moe/moe_gguf.cu",
        "cuda/moe/attn_post_qkv.cu",
        "cuda/moe/kv_residual_scatter.cu",
        "cuda/moe/gated_delta_net.cu",
        "cuda/moe/fused_conv_silu.cu",
        "cuda/moe/zgate_rmsnorm.cu",
        "cuda/moe/deltanet_gate.cu",
        "cuda/moe/l2norm_gqa.cu",
        "cuda/moe/head_rmsnorm.cu",
        "cuda/moe/gate_gemv.cu",
        "cuda/moe/gate_topk.cu",
        "cuda/moe/lfm2_shortconv_f16.cu",
    ];
    // These shipped as compute_80 PTX alone. PTX only ever JITs UPWARD, so that was not a
    // portable build with a slow path for older cards - it was a build from which no Turing
    // card could load a single kernel. Real SASS per architecture, like every other family.
    let work: Vec<(&str, Vec<String>)> = sources
        .iter()
        .map(|s| (*s, portable_gencodes()))
        .chain(ampere_only.iter().map(|s| (*s, gencodes_from(800))))
        .collect();
    let handles: Vec<std::thread::JoinHandle<PathBuf>> = work
        .into_iter()
        .map(|(src, gencodes)| {
            println!("cargo:rerun-if-changed={src}");
            let nvcc = nvcc.clone();
            let out_dir = out_dir.clone();
            std::thread::spawn(move || {
                let stem = std::path::Path::new(src)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap();
                let obj = out_dir.join(format!("{stem}.o"));
                let status = Command::new(&nvcc)
                    .args(&gencodes)
                    .args([
                        "-c",
                        src,
                        "-o",
                        obj.to_str().unwrap(),
                        "-O3",
                        "--use_fast_math",
                        "-std=c++17",
                        "-I",
                        "cuda/moe",
                        "-Xcompiler",
                        "-fPIC",
                    ])
                    .status()
                    .unwrap_or_else(|e| panic!("failed to run nvcc for {src}: {e}"));
                assert!(status.success(), "nvcc failed for {src}");
                obj
            })
        })
        .collect();
    let objs: Vec<PathBuf> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let lib = out_dir.join("libloken_imma.a");
    // `ar rcs` REPLACES the members it is given and leaves the rest alone, so an object
    // whose source was deleted stays in the archive and keeps linking - which is exactly
    // how a removed kernel produced a duplicate symbol. Start from nothing.
    let _ = std::fs::remove_file(&lib);
    let mut ar_args: Vec<String> = vec!["rcs".into(), lib.to_str().unwrap().into()];
    ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
    let ar = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("failed to run ar");
    assert!(ar.success(), "ar failed for libloken_imma.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=loken_imma");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}
