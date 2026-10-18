// The launchers of the kernels written in NVIDIA's own instructions (inline PTX, the IMMA and
// Marlin tensor-core paths, the bf16 warp-matrix DiT and expert prefill), for a HIP build that
// has no code for them. Their callers decline on any runtime but CUDA, so these are never
// reached; one that is stops the process and names itself rather than return garbage.
#include "loken_hip.h"

#include <cstdio>
#include <cstdlib>

[[noreturn]] static void loken_nvidia_only(const char* name) {
    std::fprintf(stderr, "%s: an NVIDIA-only kernel was reached on a HIP build\n", name);
    std::abort();
}

extern "C" {
void q4k_mmvq_imma(const void*, const void*, void*, int, int, cudaStream_t) {
    loken_nvidia_only(__func__);
}
void loken_dense_q4k_imma_m8_actmul(const void*, const void*, const void*, float*, int, int, int,
                                    int, cudaStream_t) {
    loken_nvidia_only(__func__);
}
void loken_dense_q4k_imma_m8_plain(const void*, const void*, float*, int, int, int, cudaStream_t) {
    loken_nvidia_only(__func__);
}
void loken_dense_q5k_imma_m8_plain(const void*, const void*, float*, int, int, int, cudaStream_t) {
    loken_nvidia_only(__func__);
}
void loken_dense_q5k_imma_m8_actmul(const void*, const void*, const void*, float*, int, int, int,
                                    int, cudaStream_t) {
    loken_nvidia_only(__func__);
}
void loken_moe_q4k_imma_m8_gate_up(const void*, const void*, const int32_t*, const int32_t*,
                                   float*, int, int, int, int, int, cudaStream_t) {
    loken_nvidia_only(__func__);
}
void loken_moe_q4k_imma_m8_down(const void*, const void*, const int32_t*, const int32_t*,
                                const float*, float*, int, int, int, int, int, cudaStream_t) {
    loken_nvidia_only(__func__);
}
int loken_marlin_awq_f16_gemm(const void*, const void*, const void*, const void*, void*, float*,
                              int*, int, int, int, int, int, int, void*) {
    loken_nvidia_only(__func__);
}
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
