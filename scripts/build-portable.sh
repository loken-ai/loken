#!/usr/bin/env bash
# One Linux x86-64 release archive that starts on the widest range of machines.
#
#   scripts/build-portable.sh           # build in a container, package into dist/
#   scripts/build-portable.sh --host    # build with this machine's toolchain instead
#
# What makes it portable:
#   - The CUDA libraries (driver, NVRTC, cuBLAS, NCCL) and the OpenCL loader are opened at run
#     time. A machine without them starts, reports no device, and serves on the CPU.
#   - TLS is rustls with bundled roots: no OpenSSL, no system certificate store.
#   - The CUDA runtime and the C++ runtime are linked in.
#   - The CPU kernels are built for a fixed baseline (CPU_BASELINE), not for the build machine.
#   - The container build links against an old glibc, so the archive runs on that glibc and
#     every newer one. `--host` links against the host's, and the glibc check below reports it.
#
# Needs: git, tar, sha256sum, and either docker or (with --host) a Rust toolchain, a C/C++
# compiler and the CUDA toolkit. The CUDA toolkit is read from CUDA_PATH, else from nvcc.
#
# Settings (environment):
#   BASE_IMAGE       image the container build starts from (default ubuntu:22.04)
#   CPU_BASELINE     value of -C target-cpu for the CPU kernels (default x86-64-v3: AVX2, FMA, BMI2)
#   MAX_GLIBC        highest glibc symbol version the binaries may need (default 2.34)
#   FEATURES         cargo features, comma separated (default: the crate's defaults)
#   CUDA_PATH        CUDA toolkit root
#   CARGO_TARGET_DIR build directory (default: target in the repository)
#   DIST             output directory (default: dist)
#   BUILD_CACHE      cargo home shared between container builds (downloads, git sources)
#   CONTAINER_MEMORY memory limit of the build container, for example 24g
#   KEEP_IMAGE       set to keep the build image after the build
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE=container
case "${1:-}" in
    "") ;;
    --host) MODE=host ;;
    *) echo "usage: $0 [--host]" >&2; exit 2 ;;
esac

BASE_IMAGE="${BASE_IMAGE:-ubuntu:22.04}"
CPU_BASELINE="${CPU_BASELINE:-x86-64-v3}"
MAX_GLIBC="${MAX_GLIBC:-2.34}"
FEATURES="${FEATURES:-}"
DIST="${DIST:-dist}"
ROOT="$PWD"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
BUILD_CACHE="${BUILD_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/loken-portable}"
IMAGE="loken-portable-build"

if [ -z "${CUDA_PATH:-}" ]; then
    nvcc_bin="$(command -v nvcc || true)"
    [ -n "$nvcc_bin" ] || { echo "set CUDA_PATH or put nvcc on PATH" >&2; exit 1; }
    CUDA_PATH="$(dirname "$(dirname "$(readlink -f "$nvcc_bin")")")"
fi
[ -x "$CUDA_PATH/bin/nvcc" ] || { echo "no nvcc under CUDA_PATH=$CUDA_PATH" >&2; exit 1; }
[ -f "$CUDA_PATH/lib64/libcudart_static.a" ] \
    || { echo "no libcudart_static.a under $CUDA_PATH/lib64" >&2; exit 1; }

PROFILE_DIR="$TARGET/portable"
mkdir -p "$PROFILE_DIR" "$DIST"
PROFILE_DIR="$(cd "$PROFILE_DIR" && pwd)"

feature_args=()
[ -n "$FEATURES" ] && feature_args=(--features "$FEATURES")

cargo_build() {
    RUSTFLAGS="-C target-cpu=$CPU_BASELINE" \
    LOKEN_STATIC_CXX=1 \
    CARGO_TARGET_DIR="$PROFILE_DIR" \
    cargo build --release --locked --bins "${feature_args[@]}"
}

if [ "$MODE" = host ]; then
    CUDA_PATH="$CUDA_PATH" cargo_build
    RUSTC_VERSION="$(rustc --version)"
else
    command -v docker >/dev/null || { echo "docker not found (or use --host)" >&2; exit 1; }
    rust_version="$(rustc --version 2>/dev/null | cut -d' ' -f2 || true)"
    rust_version="${rust_version:-stable}"
    docker build -q -t "$IMAGE" - >/dev/null <<EOF
FROM $BASE_IMAGE
ENV DEBIAN_FRONTEND=noninteractive RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:\$PATH
RUN apt-get update \\
 && apt-get install -y --no-install-recommends build-essential pkg-config git curl ca-certificates cmake perl \\
 && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain $rust_version
EOF
    mkdir -p "$BUILD_CACHE"
    git_common="$(cd "$(git rev-parse --git-common-dir)" && pwd)"
    docker run --rm ${CONTAINER_MEMORY:+--memory "$CONTAINER_MEMORY"} \
        -u "$(id -u):$(id -g)" -e HOME=/tmp \
        -e RUSTFLAGS="-C target-cpu=$CPU_BASELINE" -e LOKEN_STATIC_CXX=1 \
        -e CARGO_TARGET_DIR="$PROFILE_DIR" -e CUDA_PATH="$CUDA_PATH" \
        -e PATH="$CUDA_PATH/bin:/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        -v "$ROOT:$ROOT:ro" -v "$git_common:$git_common:ro" \
        -v "$PROFILE_DIR:$PROFILE_DIR" -v "$CUDA_PATH:$CUDA_PATH:ro" \
        -e CARGO_HOME=/cargo-cache -v "$BUILD_CACHE:/cargo-cache" \
        -w "$ROOT" "$IMAGE" \
        cargo build --release --locked --bins "${feature_args[@]}"
    RUSTC_VERSION="$(docker run --rm "$IMAGE" rustc --version)"
    [ -n "${KEEP_IMAGE:-}" ] || docker rmi "$IMAGE" >/dev/null
fi

BIN_DIR="$PROFILE_DIR/release"
[ -x "$BIN_DIR/lokend" ] && [ -x "$BIN_DIR/loken" ] || { echo "build produced no binaries" >&2; exit 1; }

# The check that makes the archive a promise: nothing the CUDA or OpenCL stacks provide is a
# load-time dependency, and no glibc symbol is newer than MAX_GLIBC.
allowed_needed='^(libc\.so\.6|libm\.so\.6|libgcc_s\.so\.1|libdl\.so\.2|libpthread\.so\.0|librt\.so\.1|libz\.so\.1|libzstd\.so\.1|ld-linux-x86-64\.so\.2|linux-vdso\.so\.1)$'
for bin in lokend loken; do
    needed="$(readelf -d "$BIN_DIR/$bin" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p; s/.*Bibliothèque partagée : \[\(.*\)\]/\1/p')"
    unexpected="$(printf '%s\n' "$needed" | grep -Ev "$allowed_needed" || true)"
    [ -z "$unexpected" ] || { echo "$bin needs libraries the target may lack: $unexpected" >&2; exit 1; }
    top="$(objdump -T "$BIN_DIR/$bin" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -Vu | tail -1)"
    newest="$(printf '%s\n%s\n' "$top" "$MAX_GLIBC" | sort -V | tail -1)"
    [ "$newest" = "$MAX_GLIBC" ] \
        || { echo "$bin needs glibc $top, above MAX_GLIBC=$MAX_GLIBC" >&2; exit 1; }
    echo "$bin: needs glibc $top, libraries: $(echo $needed | tr '\n' ' ')"
done

VERSION="$("$BIN_DIR/lokend" --version | awk '{print $2}')"
COMMIT="$(git rev-parse --short=9 HEAD)"
NAME="loken-$VERSION-linux-x86_64"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME"
for bin in lokend loken; do
    strip -o "$STAGE/$NAME/$bin" "$BIN_DIR/$bin"
done
cp LICENSE-MIT LICENSE-APACHE NOTICE.md "$STAGE/$NAME/"

archs="$(grep -o 'code=sm_[0-9]*a\?' build.rs | sed 's/code=//' | sort -u | tr '\n' ' ')"
cuda_version="$(sed -n 's/.*"cuda" *: *{[^}]*"version" *: *"\([^"]*\)".*/\1/p' "$CUDA_PATH/version.json" 2>/dev/null | head -1)"
[ -n "$cuda_version" ] || cuda_version="$("$CUDA_PATH/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
features="${FEATURES:-$(sed -n 's/^default = \[\(.*\)\]/\1/p' Cargo.toml | head -1 | tr -d ' "')}"
cat > "$STAGE/$NAME/BUILD-INFO" <<EOF
version:        $VERSION
commit:         $COMMIT
rust:           $RUSTC_VERSION
cuda toolkit:   $cuda_version (kernels compiled with it; the target needs the CUDA 13 libraries for GPU use)
cuda archs:     $archs (SASS per architecture, plus compute_80 PTX for newer cards)
cuda tested:    sm_120 only; the other architectures are built for, untested on
cpu baseline:   $CPU_BASELINE
features:       $features
glibc needed:   $(objdump -T "$BIN_DIR/lokend" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -Vu | tail -1)
build mode:     $MODE ($BASE_IMAGE)
runtime libs:   none required. CUDA driver, NVRTC, cuBLAS, NCCL and the OpenCL loader are
                opened when present; without them the server runs on the CPU.
EOF

tar --sort=name --owner=0 --group=0 --numeric-owner -C "$STAGE" -czf "$DIST/$NAME.tar.gz" "$NAME"
(cd "$DIST" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" && cat "$NAME.tar.gz.sha256")
ls -l "$DIST/$NAME.tar.gz"
