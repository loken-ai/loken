// The launchers of the kernels written in NVIDIA's own instructions (inline PTX, the bf16
// warp-matrix DiT and expert prefill), for a HIP build that has no code for them; the integer
// matrix products have AMD kernels of their own in `cuda/moe/imma_amd.cu`. Their callers
// decline on any runtime but CUDA, so these are never reached; one that is stops the process
// and names itself rather than return garbage.
#include "loken_hip.h"

#include <cstdio>
#include <cstdlib>

[[noreturn]] static void loken_nvidia_only(const char* name) {
    std::fprintf(stderr, "%s: an NVIDIA-only kernel was reached on a HIP build\n", name);
    std::abort();
}

extern "C" {
void loken_moe_gemm_gguf_prefill(const void*, const uint8_t*, const int32_t*, const int32_t*,
                                 const float*, float*, int, int, int, int, int, int, int,
                                 cudaStream_t) {
    loken_nvidia_only(__func__);
}
void loken_moe_expert_offsets(const int32_t*, int, int32_t*, int, cudaStream_t) {
    loken_nvidia_only(__func__);
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
