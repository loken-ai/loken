// Build script for compiling CUDA kernels
// This script runs during the cargo build process when the cuda feature is enabled

fn main() {
    // The commit these bytes came from, stamped at build time so a running server can say
    // which build is answering. A version number alone cannot tell two builds of one release
    // apart, which is exactly what a benchmark row has to do. Read here rather than at run
    // time: a deployed binary has no checkout beside it.
    // The repository's own directory and the one its worktrees share: a worktree's `.git` is a
    // file naming them. Cargo reruns this script on every build for a watched path that does
    // not exist, so only paths that do are watched.
    let (git_dir, common_dir) = std::process::Command::new("git")
        .args(["rev-parse", "--git-dir", "--git-common-dir"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|text| {
            let mut lines = text.lines().map(str::to_string);
            Some((lines.next()?, lines.next()?))
        })
        .unwrap_or_else(|| (".git".to_string(), ".git".to_string()));
    let watch = |path: String| {
        if std::path::Path::new(&path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    };
    watch(format!("{git_dir}/HEAD"));
    if let Ok(head) = std::fs::read_to_string(format!("{git_dir}/HEAD")) {
        // A branch moves under a fixed HEAD, so the ref it names is what has to be watched.
        if let Some(reference) = head.strip_prefix("ref: ").map(str::trim) {
            watch(format!("{common_dir}/{reference}"));
        }
    }
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short=9", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        // A source tarball has no repository, and saying so is better than an empty string
        // that reads as a build nobody stamped.
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=LOKEN_COMMIT={commit}");

    // The version a build answers with. A commit carrying the release tag `v<version>` is that
    // release; any other commit is pre-release work on top of it and says so in SemVer form,
    // `<version>-dev+g<commit>`. A source archive without a repository is taken as the release
    // it was cut from.
    watch(format!("{common_dir}/refs/tags"));
    watch(format!("{common_dir}/packed-refs"));
    let release = env!("CARGO_PKG_VERSION");
    let at_release_tag = std::process::Command::new("git")
        .args(["tag", "--points-at", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|tags| tags.lines().any(|t| t.trim() == format!("v{release}")));
    let version = match at_release_tag {
        Some(false) if commit != "unknown" => format!("{release}-dev+g{commit}"),
        _ => release.to_string(),
    };
    println!("cargo:rustc-env=LOKEN_VERSION={version}");

    link_cxx_runtime_statically();

    println!("cargo:rerun-if-changed=src/inference/sampling_kernels.cu");
    println!("cargo:rerun-if-changed=src/inference/sampling_kernels.h");

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
    // The same kernels for AMD cards, through HIP: alone when ROCm is the only runtime, beside
    // the CUDA ones under a prefix of their own when both are.
    #[cfg(feature = "rocm")]
    compile_hip_kernels();
}

/// Kernels in NVIDIA's Ampere tensor-core instructions (bf16 warp-matrix, `m16n8k32` integer MMA).
#[cfg(any(feature = "cuda", feature = "rocm"))]
const KERNELS_AMPERE_ONLY: [&str; 6] = [
    "cuda/q4k_mmvq_imma.cu",
    "cuda/moe/dense_q4k_imma_m8.cu",
    "cuda/moe/dense_q5k_imma_m8.cu",
    "cuda/moe/moe_q4k_imma_m8.cu",
    "cuda/moe/moe_q4k_imma_m8_down.cu",
    "cuda/moe/moe_wmma_gguf.cu",
];

/// The kernel families every card serves, compiled per architecture.
#[cfg(any(feature = "cuda", feature = "rocm"))]
const KERNELS: [&str; 30] = [
    "cuda/fused_norm_f16.cu",
    "cuda/gptoss_flash_decode.cu",
    "cuda/flash_decode_f16.cu",
    "cuda/flash_decode_tiled_probe.cu",
    "cuda/sampling.cu",
    "cuda/moe/add_rms_norm.cu",
    "cuda/moe/rms_qmatmul.cu",
    "cuda/moe/moe_gguf.cu",
    "cuda/moe/embed_rows_q8_0.cu",
    "cuda/moe/moe_mmq.cu",
    "cuda/moe/attn_post_qkv.cu",
    "cuda/moe/kv_residual_scatter.cu",
    "cuda/moe/gated_delta_net.cu",
    "cuda/moe/gated_delta_net_chunked.cu",
    "cuda/moe/fused_conv_silu.cu",
    "cuda/moe/zgate_rmsnorm.cu",
    "cuda/moe/hyper_residual.cu",
    "cuda/moe/gdn_prep.cu",
    "cuda/moe/qk_prep.cu",
    "cuda/moe/shared_expert.cu",
    "cuda/moe/indexer.cu",
    "cuda/moe/deltanet_gate.cu",
    "cuda/moe/l2norm_gqa.cu",
    "cuda/moe/head_rmsnorm.cu",
    "cuda/moe/gate_gemv.cu",
    "cuda/moe/bf16_matvec.cu",
    "cuda/moe/gate_topk.cu",
    "cuda/moe/tier_graph.cu",
    "cuda/moe/tp_local.cu",
    "cuda/moe/lfm2_shortconv_f16.cu",
];

/// The quantised matrix products, one translation unit per weight format.
#[cfg(any(feature = "cuda", feature = "rocm"))]
const KERNELS_MMQ: [&str; 18] = [
    "cuda/mmq_gguf/mmq_quantize.cu",
    "cuda/mmq_gguf/mmq_instance_q4_0.cu",
    "cuda/mmq_gguf/mmq_instance_q4_1.cu",
    "cuda/mmq_gguf/mmq_instance_q5_0.cu",
    "cuda/mmq_gguf/mmq_instance_q5_1.cu",
    "cuda/mmq_gguf/mmq_instance_q8_0.cu",
    "cuda/mmq_gguf/mmq_instance_q2_0.cu",
    "cuda/mmq_gguf/mmq_instance_q2_k.cu",
    "cuda/mmq_gguf/mmq_instance_q3_k.cu",
    "cuda/mmq_gguf/mmq_instance_q4_k.cu",
    "cuda/mmq_gguf/mmq_instance_q5_k.cu",
    "cuda/mmq_gguf/mmq_instance_q6_k.cu",
    "cuda/mmq_gguf/mmq_instance_mxfp4.cu",
    "cuda/mmq_gguf/mmq_instance_iq4_nl.cu",
    "cuda/mmq_gguf/mmq_instance_iq3_s.cu",
    "cuda/mmq_gguf/mmq_instance_iq3_xxs.cu",
    "cuda/mmq_gguf/mmq_instance_iq2_s.cu",
    "cuda/mmq_gguf/mmq_instance_iq4_xs.cu",
];

/// The AMD targets a portable ROCm build carries code for, oldest first: CDNA (MFMA), RDNA2,
/// RDNA3, RDNA3.5 and RDNA4 (WMMA from RDNA3 on). The runtime loads the object for the card it
/// finds; a card none of them covers has no GPU code in this build.
#[cfg(feature = "rocm")]
const ROCM_ARCHS: [&str; 11] = [
    "gfx908", "gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151",
    "gfx1200", "gfx1201",
];

/// The kernel families every AMD card serves, compiled with hipcc for the targets of
/// `LOKEN_ROCM_ARCHS` (comma separated) or, without it, every target of [`ROCM_ARCHS`]. The
/// sources are the CUDA build's, read through `cuda/hip`, which gives the CUDA names their HIP
/// meaning; the kernels in NVIDIA's own instructions are left out, and the `_amd` sources
/// (`cuda/marlin/marlin_amd.cu`, `cuda/moe/imma_amd.cu`, `cuda/moe/moe_wmma_amd.cu`,
/// `cuda/flash_prefill_gqa_amd.cu`, `cuda/flash_dit_amd.cu`) give them AMD versions with the
/// same entry points. HIP has no static runtime: a ROCm build links `libamdhip64`.
///
/// Beside the CUDA kernels, every symbol the objects define takes the prefix `loken_hip_`, so the
/// two libraries do not clash, and every call they make into `libamdhip64` goes through a
/// trampoline that the library opened at start resolves (see [`hip_runtime_trampolines`]): the
/// binary then starts on a machine without ROCm.
#[cfg(feature = "rocm")]
fn compile_hip_kernels() {
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=LOKEN_ROCM_ARCHS");
    let rocm = std::env::var("ROCM_PATH")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let out = Command::new("sh")
                .args(["-c", "command -v hipcc"])
                .output()
                .ok()?;
            let hipcc = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
            Some(
                std::fs::canonicalize(hipcc)
                    .ok()?
                    .parent()?
                    .parent()?
                    .to_path_buf(),
            )
        });
    let Some(rocm) = rocm else {
        panic!("hipcc not found: set ROCM_PATH or put hipcc on PATH (needed for the AMD kernels)");
    };
    let hipcc = rocm.join("bin").join("hipcc");
    assert!(hipcc.exists(), "hipcc not found at {hipcc:?}");
    let archs: Vec<String> = match std::env::var("LOKEN_ROCM_ARCHS") {
        Ok(list) => list
            .split(',')
            .map(|a| a.trim().to_string())
            .filter(|a| !a.is_empty())
            .collect(),
        Err(_) => ROCM_ARCHS.iter().map(|a| a.to_string()).collect(),
    };
    let nvidia_only: Vec<&str> = KERNELS_AMPERE_ONLY.to_vec();
    let mut sources: Vec<&str> = KERNELS
        .iter()
        .chain(KERNELS_MMQ.iter())
        .copied()
        .filter(|s| !nvidia_only.contains(s))
        .collect();
    sources.extend([
        "cuda/flashdecode_tc/flashdecode_tc.cu",
        "cuda/flash_prefill_f16.cu",
        "cuda/flash_prefill_gqa_amd.cu",
        "cuda/flash_dit_amd.cu",
        "cuda/marlin/marlin_amd.cu",
        "cuda/moe/imma_amd.cu",
        "cuda/moe/moe_wmma_amd.cu",
    ]);
    println!("cargo:rerun-if-changed=cuda/hip");
    let mut flags: Vec<String> = vec!["-x".into(), "hip".into()];
    flags.extend(archs.iter().map(|a| format!("--offload-arch={a}")));
    flags.extend(
        [
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-include",
            "cuda/hip/loken_hip.h",
            "-I",
            "cuda/hip",
            "-I",
            "cuda",
            "-I",
            "cuda/moe",
            "-I",
            "cuda/mmq_gguf",
        ]
        .iter()
        .map(|f| f.to_string()),
    );
    let handles: Vec<std::thread::JoinHandle<PathBuf>> = sources
        .into_iter()
        .map(|src| {
            println!("cargo:rerun-if-changed={src}");
            let (hipcc, out_dir, flags) = (hipcc.clone(), out_dir.clone(), flags.clone());
            std::thread::spawn(move || {
                let stem = std::path::Path::new(src)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap();
                let obj = out_dir.join(format!("hip_{stem}.o"));
                compile_kernel(&hipcc, src, &flags, &obj);
                obj
            })
        })
        .collect();
    #[allow(unused_mut)]
    let mut objs: Vec<PathBuf> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    #[cfg(feature = "cuda")]
    {
        objs = prefix_hip_symbols(&objs, &rocm.join("lib").join("libamdhip64.so"), &out_dir);
    }

    let lib = out_dir.join("libloken_hip_kernels.a");
    let _ = std::fs::remove_file(&lib);
    let mut ar_args: Vec<String> = vec!["rcs".into(), lib.to_str().unwrap().into()];
    ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
    let ar = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("failed to run ar");
    assert!(ar.success(), "ar failed for libloken_hip_kernels.a");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=loken_hip_kernels");
    if cfg!(not(feature = "cuda")) {
        println!(
            "cargo:rustc-link-search=native={}",
            rocm.join("lib").display()
        );
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }
}

/// The names `nm` lists for `obj` with `flags`, one per symbol.
#[cfg(all(feature = "rocm", feature = "cuda"))]
fn symbol_names(obj: &std::path::Path, flags: &[&str]) -> Vec<String> {
    let out = std::process::Command::new("nm")
        .args(flags)
        .arg("-P")
        .arg(obj)
        .output()
        .unwrap_or_else(|e| panic!("failed to run nm on {obj:?}: {e}"));
    assert!(out.status.success(), "nm failed on {obj:?}");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().next().map(str::to_string))
        .collect()
}

/// Renames the HIP objects' symbols for a binary that also carries the CUDA kernels: each
/// defined global `s` becomes `loken_hip_s`, and each call into `libamdhip64` becomes a call to
/// `loken_hiprt_s`, a trampoline added to the returned objects.
#[cfg(all(feature = "rocm", feature = "cuda"))]
fn prefix_hip_symbols(
    objs: &[std::path::PathBuf],
    hip_runtime: &std::path::Path,
    out_dir: &std::path::Path,
) -> Vec<std::path::PathBuf> {
    use std::collections::BTreeSet;
    let runtime: BTreeSet<String> = symbol_names(hip_runtime, &["-D", "--defined-only"])
        .into_iter()
        .map(|s| s.split('@').next().unwrap_or_default().to_string())
        .collect();
    let mut defined = BTreeSet::new();
    let mut undefined = BTreeSet::new();
    for obj in objs {
        defined.extend(symbol_names(obj, &["--defined-only", "-g"]));
        undefined.extend(symbol_names(obj, &["-u"]));
    }
    let calls: Vec<String> = undefined
        .into_iter()
        .filter(|s| !defined.contains(s) && runtime.contains(s))
        .collect();
    let mut map = String::new();
    for s in &defined {
        map.push_str(&format!("{s} loken_hip_{s}\n"));
    }
    for s in &calls {
        map.push_str(&format!("{s} loken_hiprt_{s}\n"));
    }
    let map_file = out_dir.join("hip_symbols.map");
    std::fs::write(&map_file, map).expect("symbol map");
    let mut renamed: Vec<std::path::PathBuf> = objs
        .iter()
        .map(|obj| {
            let out = obj.with_extension("prefixed.o");
            let status = std::process::Command::new("objcopy")
                .arg(format!("--redefine-syms={}", map_file.display()))
                .arg(obj)
                .arg(&out)
                .status()
                .expect("failed to run objcopy");
            assert!(status.success(), "objcopy failed on {obj:?}");
            out
        })
        .collect();
    renamed.push(hip_runtime_trampolines(&calls, out_dir));
    renamed
}

/// The highest major version tried when `libamdhip64` has no unversioned name on the search
/// path, as the run-time HIP module tries it.
#[cfg(all(feature = "rocm", feature = "cuda"))]
const HIP_MAX_SONAME_MAJOR: u32 = 16;

/// An object holding one trampoline `loken_hiprt_s` per HIP runtime entry point `s` the kernels
/// call, and a constructor that runs before the kernels' own and resolves them in `libamdhip64`
/// when it opens. An unresolved trampoline returns zero: the kernels' registration then does
/// nothing, and no AMD card is listed to launch them on.
#[cfg(all(feature = "rocm", feature = "cuda"))]
fn hip_runtime_trampolines(calls: &[String], out_dir: &std::path::Path) -> std::path::PathBuf {
    use std::fmt::Write;
    let mut c = String::new();
    let _ = writeln!(c, "#include <dlfcn.h>");
    let _ = writeln!(c, "#include <stdio.h>");
    let _ = writeln!(c, "void* loken_hiprt_table[{}];", calls.len().max(1));
    let _ = writeln!(c, "static const char* const loken_hiprt_names[] = {{");
    for s in calls {
        let _ = writeln!(c, "    \"{s}\",");
    }
    let _ = writeln!(c, "    0,\n}};");
    let _ = writeln!(
        c,
        r#"__attribute__((constructor(101))) static void loken_hiprt_resolve(void) {{
    void* h = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
    char name[32];
    for (int v = {HIP_MAX_SONAME_MAJOR}; !h && v >= 0; --v) {{
        snprintf(name, sizeof name, "libamdhip64.so.%d", v);
        h = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    }}
    if (!h) return;
    for (int i = 0; loken_hiprt_names[i]; ++i)
        loken_hiprt_table[i] = dlsym(h, loken_hiprt_names[i]);
}}"#
    );
    let _ = writeln!(c, "__asm__(\".text\\n\"");
    for (i, s) in calls.iter().enumerate() {
        let t = format!("loken_hiprt_{s}");
        let off = i * 8;
        for line in [
            format!(".globl {t}"),
            format!(".type {t}, @function"),
            format!("{t}:"),
            format!("    movq loken_hiprt_table+{off}(%rip), %rax"),
            "    testq %rax, %rax".to_string(),
            "    je 1f".to_string(),
            "    jmp *%rax".to_string(),
            "1:  xorl %eax, %eax".to_string(),
            "    ret".to_string(),
        ] {
            let _ = writeln!(c, "    \"{line}\\n\"");
        }
    }
    let _ = writeln!(c, ");");
    let src = out_dir.join("hip_runtime_trampolines.c");
    std::fs::write(&src, c).expect("trampoline source");
    let obj = out_dir.join("hip_runtime_trampolines.o");
    let status = std::process::Command::new("cc")
        .args(["-c", "-O2", "-fPIC"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to run cc");
    assert!(status.success(), "cc failed on the HIP runtime trampolines");
    obj
}

/// With `LOKEN_STATIC_CXX` set, links the C++ runtime into the binary so it needs no
/// libstdc++ on the machine that runs it. The archive is found through the compiler that will
/// link it; when that compiler has none, the build keeps the shared library.
fn link_cxx_runtime_statically() {
    println!("cargo:rerun-if-env-changed=LOKEN_STATIC_CXX");
    if std::env::var_os("LOKEN_STATIC_CXX").is_none() {
        return;
    }
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "g++".to_string());
    let archive = std::process::Command::new(compiler)
        .arg("-print-file-name=libstdc++.a")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| std::path::PathBuf::from(text.trim()));
    match archive.as_deref().and_then(std::path::Path::parent) {
        Some(dir) if archive.as_ref().is_some_and(|a| a.is_file()) => {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static=stdc++");
        }
        _ => println!("cargo:warning=LOKEN_STATIC_CXX set but libstdc++.a was not found"),
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
/// The gencode flags every kernel family ships with: each REAL architecture the fleet can
/// hold, as SASS, plus the arch-specific 'a' variants where the ISA needs them. One arch -
/// the build machine's - is how a binary stops being copyable: SASS for sm_120a neither runs
/// nor re-JITs on sm_89, and the failure is a kernel that computes nothing, not an error.
#[cfg(feature = "cuda")]
fn portable_gencodes() -> Vec<String> {
    // A development build for the machine's own cards: `LOKEN_CUDA_ARCHS` names the
    // architectures to build SASS for, comma separated (`120a`, `89`), and the kernels compile
    // once instead of once an architecture. A binary built this way does not leave the machine.
    println!("cargo:rerun-if-env-changed=LOKEN_CUDA_ARCHS");
    if let Ok(list) = std::env::var("LOKEN_CUDA_ARCHS") {
        let archs: Vec<String> = list
            .split(',')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(|a| format!("-gencode=arch=compute_{a},code=sm_{a}"))
            .collect();
        if !archs.is_empty() {
            return archs;
        }
    }
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
    let kept: Vec<String> = portable_gencodes()
        .into_iter()
        .filter(|g| {
            g.rsplit("code=sm_")
                .next()
                .and_then(|s| s.trim_end_matches('a').parse::<u32>().ok())
                .is_none_or(|cc| cc * 10 >= floor)
        })
        .collect();
    // A development build for cards below the floor still has to link these kernels: they carry
    // the PTX of the floor architecture, which no card of the build loads.
    if kept.is_empty() {
        let cc = floor / 10;
        return vec![format!("-gencode=arch=compute_{cc},code=compute_{cc}")];
    }
    kept
}

/// Compile the query-head-packed tensor-core flash-DECODE kernel (cuda/flashdecode_tc/,
/// see there) into a static lib. Uses nvcuda::wmma m16n8k16 HMMA +
/// cp.async-class staging, so it needs real per-arch SASS (compute>=90a suffix,
/// like the Marlin build) - the PTX-only compute_80 IMMA build cannot host the
/// 16x16x16 f16 MMA forward-JIT cleanly on sm_120. Falls back to a compute_90a
/// PTX when no GPU is visible at build time. Driven via `extern "C"` from
/// inference::flash_decode_tc.
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
    println!("cargo:rerun-if-changed=cuda/flash_prefill_gqa.cu");
    println!("cargo:rerun-if-changed=cuda/flash_dit_bf16.cu");
    // The DiT kernel's 52 KB static shared-memory tile exceeds the pre-Hopper limit, so it
    // ships for the architectures that can hold it; a card below that never runs a DiT
    // through this path. The LLM kernels ship for every real architecture in the fleet.
    let dit_gencodes: Vec<String> = portable_gencodes()
        .into_iter()
        .filter(|g| g.contains("90a") || g.contains("120a"))
        .collect();
    // A development build for cards below that line still has to link the kernel: it carries
    // the PTX of the newest architecture, which no card of the build loads.
    let dit_gencodes = if dit_gencodes.is_empty() {
        vec!["-gencode=arch=compute_120a,code=compute_120a".to_string()]
    } else {
        dit_gencodes
    };
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
        // Causal GQA prefill attention with mma.sync fragments in registers.
        (
            "cuda/flash_prefill_gqa.cu",
            "flash_prefill_gqa.o",
            &gencodes,
        ),
        // Non-causal BF16 flash attention for the diffusion transformers - same WMMA path,
        // a block-wide K/V tile instead of a per-warp one.
        ("cuda/flash_dit_bf16.cu", "flash_dit_bf16.o", &dit_gencodes),
    ];
    let mut objs = Vec::new();
    for (src, obj_name, gencodes) in srcs {
        let obj = out_dir.join(obj_name);
        let mut flags: Vec<String> = [
            "-O3",
            "--expt-relaxed-constexpr",
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
        ]
        .iter()
        .map(|f| f.to_string())
        .collect();
        flags.extend(gencodes.iter().map(|g| g.to_string()));
        compile_kernel(&nvcc, src, &flags, &obj);
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
                let mut flags = gencode_list.clone();
                flags.extend(
                    [
                        "-O3",
                        "--expt-relaxed-constexpr",
                        // The launcher references __global__ template instantiations living in
                        // the sibling units; this restores cross-TU resolution (same flag vLLM
                        // uses for its marlin build).
                        "-static-global-template-stub=false",
                        "-std=c++17",
                        "-I",
                        "cuda/marlin",
                        "-Xcompiler",
                        "-fPIC",
                    ]
                    .iter()
                    .map(|f| f.to_string()),
                );
                compile_kernel(&nvcc, &src, &flags, &obj);
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

    let sources = KERNELS_MMQ;
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
                let mut flags = gencode_list.clone();
                flags.extend(
                    [
                        "-O3",
                        "--expt-relaxed-constexpr",
                        "-std=c++17",
                        "-Xcompiler",
                        "-fPIC",
                    ]
                    .iter()
                    .map(|f| f.to_string()),
                );
                compile_kernel(&nvcc, &src, &flags, &obj);
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
    let ampere_only = KERNELS_AMPERE_ONLY;
    let sources = KERNELS;
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
                let mut flags = gencodes.clone();
                flags.extend(
                    [
                        "-O3",
                        "--use_fast_math",
                        "-std=c++17",
                        "-I",
                        "cuda/moe",
                        "-Xcompiler",
                        "-fPIC",
                    ]
                    .iter()
                    .map(|f| f.to_string()),
                );
                compile_kernel(&nvcc, src, &flags, &obj);
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
    // The runtime the kernel objects launch through is linked in, so the binary has no
    // dependency on libcudart. The driver stays a run-time library: the static runtime opens
    // libcuda with dlopen on its first call, which is what lets a machine without a card start.
    println!("cargo:rustc-link-lib=static=cudart_static");
    for system in ["dl", "rt", "pthread"] {
        println!("cargo:rustc-link-lib={system}");
    }
}

/// Compile `src` into `obj` with `flags`, unless a build of the same sources and flags already
/// did: objects are kept by content in `kernel-cache` beside the profile directories of the
/// target, so a change to one kernel recompiles that kernel, and another profile, or a fresh
/// checkout of the same sources, takes the objects it needs from the cache.
///
/// The key covers the compiler's version, the flags, the source and every header it includes.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn compile_kernel(nvcc: &std::path::Path, src: &str, flags: &[String], obj: &std::path::Path) {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let version = std::process::Command::new(nvcc)
        .arg("--version")
        .output()
        .expect("nvcc --version")
        .stdout;
    version.hash(&mut hasher);
    flags.hash(&mut hasher);
    // The sources the object is made of: the file and what it includes.
    let deps = std::process::Command::new(nvcc)
        .args(flags)
        .args(["-MM", src])
        .output()
        .unwrap_or_else(|e| panic!("failed to run nvcc -MM for {src}: {e}"));
    assert!(
        deps.status.success(),
        "{} -MM failed for {src}: {}",
        nvcc.display(),
        String::from_utf8_lossy(&deps.stderr)
    );
    let rule = String::from_utf8_lossy(&deps.stdout).replace("\\\n", " ");
    // A HIP compiler writes one rule per target with bundle markers between them: what
    // names a file is a dependency, everything else is syntax.
    let mut inputs: Vec<&str> = rule
        .split_once(':')
        .map_or("", |(_, d)| d)
        .split_whitespace()
        .filter(|t| std::path::Path::new(t).is_file())
        .collect();
    inputs.push(src);
    inputs.sort_unstable();
    inputs.dedup();
    for input in inputs {
        input.hash(&mut hasher);
        std::fs::read(input)
            .unwrap_or_else(|e| panic!("cannot read {input}: {e}"))
            .hash(&mut hasher);
    }
    let stem = std::path::Path::new(src)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let cache = obj
        .ancestors()
        .find(|a| a.file_name().is_some_and(|n| n == "build"))
        .and_then(|build| build.parent()?.parent())
        .map(|target| target.join("kernel-cache"))
        .unwrap_or_else(|| obj.parent().unwrap().join("kernel-cache"));
    std::fs::create_dir_all(&cache).expect("kernel cache directory");
    let cached = cache.join(format!("{stem}-{:016x}.o", hasher.finish()));
    if !cached.exists() {
        let partial = cache.join(format!("{stem}-{:016x}.partial", hasher.finish()));
        let status = std::process::Command::new(nvcc)
            .args(["-c", src, "-o", partial.to_str().unwrap()])
            .args(flags)
            .status()
            .unwrap_or_else(|e| panic!("failed to run nvcc for {src}: {e}"));
        assert!(status.success(), "nvcc failed for {src}");
        std::fs::rename(&partial, &cached).expect("kernel cache entry");
    }
    let _ = std::fs::remove_file(obj);
    std::fs::copy(&cached, obj).expect("kernel object from the cache");
}
