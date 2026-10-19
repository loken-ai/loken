// The CUDA names the kernels are written with, given their HIP meaning, so one source compiles
// with nvcc for NVIDIA cards and with hipcc for AMD ones. Included through the cuda_*.h
// headers of this directory, which a HIP build puts first on the include path.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#if defined(__HIPCC_RTC__)
// hipRTC defines the fixed-width integer types before every source, so the NVRTC shim's own
// definitions are skipped; its float constants are kept.
#define LOKEN_NVRTC_COMPAT_H
// A runtime compilation, as the sources test for it.
#define __CUDACC_RTC__ 1
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
#endif

// The AMD families the GGUF matmul kernels branch on, named from the target being compiled.
#define GGML_USE_HIP
#if defined(__gfx1200__) || defined(__gfx1201__)
#define RDNA4
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define RDNA3
#endif
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
#define RDNA2
#endif
#if defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif
#if defined(__gfx942__) || defined(__gfx950__)
#define CDNA3
#endif
#if defined(__gfx90a__)
#define CDNA2
#endif
#if defined(__gfx908__)
#define CDNA1
#endif
#if defined(CDNA1) || defined(CDNA2) || defined(CDNA3)
#define CDNA
#endif
#if defined(__gfx900__) || defined(__gfx906__)
#define GCN5
#define GCN
#endif

// Runtime API.
#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaFuncSetAttribute(f, a, v) hipFuncSetAttribute(reinterpret_cast<const void*>(f), a, v)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes(attr, f) hipFuncGetAttributes(attr, reinterpret_cast<const void*>(f))
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMalloc hipMalloc
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone

// Types.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

// Device intrinsics CUDA has and HIP spells otherwise.
#define __trap() __builtin_trap()
// s_sleep takes a constant count of 64-clock units, so a time known only at run time is slept
// in its longest step and then one unit at a time.
static __device__ inline void __nanosleep(unsigned int ns) {
    unsigned int units = (ns + 63) / 64;
    for (; units >= 127; units -= 127) __builtin_amdgcn_s_sleep(127);
    for (; units > 0; --units) __builtin_amdgcn_s_sleep(1);
}

// A wave runs all its lanes together: the participation mask of a CUDA warp function has no
// lane to leave out, so the HIP call without a mask is the same operation.
#define __shfl_sync(mask, var, lane, ...) __shfl(var, lane, ##__VA_ARGS__)
#define __shfl_xor_sync(mask, var, lane, ...) __shfl_xor(var, lane, ##__VA_ARGS__)
#define __shfl_down_sync(mask, var, delta, ...) __shfl_down(var, delta, ##__VA_ARGS__)
#define __shfl_up_sync(mask, var, delta, ...) __shfl_up(var, delta, ##__VA_ARGS__)
#define __any_sync(mask, pred) __any(pred)
#define __all_sync(mask, pred) __all(pred)
#define __ballot_sync(mask, pred) __ballot(pred)
#define __syncwarp(...) __builtin_amdgcn_wave_barrier()

// Four signed bytes, lane by lane.
static __device__ __forceinline__ int loken_hip_byte_op(int a, int b, int saturate, int sub) {
    int r;
    const int8_t* x = reinterpret_cast<const int8_t*>(&a);
    const int8_t* y = reinterpret_cast<const int8_t*>(&b);
    int8_t* z = reinterpret_cast<int8_t*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = sub ? int(x[i]) - int(y[i]) : int(x[i]) + int(y[i]);
        if (saturate) d = d > 127 ? 127 : (d < -128 ? -128 : d);
        z[i] = static_cast<int8_t>(d);
    }
    return r;
}
static __device__ __forceinline__ int __vsub4(int a, int b) { return loken_hip_byte_op(a, b, 0, 1); }
static __device__ __forceinline__ int __vsubss4(int a, int b) { return loken_hip_byte_op(a, b, 1, 1); }
static __device__ __forceinline__ int __vadd4(int a, int b) { return loken_hip_byte_op(a, b, 0, 0); }

// The four-way signed byte dot product with accumulate: one instruction where the target has it.
static __device__ __forceinline__ int __dp4a(int a, int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || \
    defined(__gfx950__) || defined(__gfx1030__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__gfx11__) || defined(__gfx12__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8_t* x = reinterpret_cast<const int8_t*>(&a);
    const int8_t* y = reinterpret_cast<const int8_t*>(&b);
    return c + x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
#endif
}
