// The launchers of the kernels written in NVIDIA's own instructions (inline PTX, the bf16
// warp-matrix DiT and expert prefill), for a HIP build that has no code for them; the integer
// matrix products have AMD kernels of their own in `cuda/moe/imma_amd.cu`. Their callers
// decline on any runtime but CUDA, so these are never reached; one that is names itself, then
// stops the process or, for a launcher that reports a status, returns a nonzero one.
#include "loken_hip.h"

#include <cstdio>
#include <cstdlib>

[[noreturn]] static void loken_nvidia_only(const char* name) {
    std::fprintf(stderr, "%s: an NVIDIA-only kernel was reached on a HIP build\n", name);
    std::abort();
}

static int loken_nvidia_only_status(const char* name) {
    std::fprintf(stderr, "%s: an NVIDIA-only kernel was reached on a HIP build\n", name);
    return 1;
}

extern "C" {
int loken_moe_gemm_gguf_prefill(const void*, const uint8_t*, const int32_t*, const int32_t*,
                                const float*, float*, int, int, int, int, int, int, int,
                                cudaStream_t) {
    return loken_nvidia_only_status(__func__);
}
int loken_moe_expert_offsets(const int32_t*, int, int32_t*, int, cudaStream_t) {
    return loken_nvidia_only_status(__func__);
}
void flash_dit_bf16_hd64(const void*, const void*, const void*, void*, int, int, int, int, float,
                         int, int, long long) {
    loken_nvidia_only(__func__);
}
void flash_dit_bf16_hd128(const void*, const void*, const void*, void*, int, int, int, int, float,
                          int, int, long long) {
    loken_nvidia_only(__func__);
}
}
