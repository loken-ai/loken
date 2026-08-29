// NVRTC compatibility shim.
//
// NVRTC compiles CUDA C++ in isolation: it ships the CUDA device headers
// (cuda_fp16.h, cuda_bf16.h, ...) but NOT the host C standard library, so
// `<stdint.h>` (fixed-width integer types) and the `<math.h>` macros
// (INFINITY, NAN) that nvcc pulls in implicitly are unavailable.
//
// This header provides exactly those definitions for the x86-64 CUDA ABI and
// is prepended to the kernel translation unit at NVRTC compile time, so the
// kernel sources themselves stay free of compiler-workaround cruft.
#ifndef LOKEN_NVRTC_COMPAT_H
#define LOKEN_NVRTC_COMPAT_H

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef short              int16_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef long long          int64_t;
typedef unsigned long long uint64_t;

#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif

#endif // LOKEN_NVRTC_COMPAT_H
