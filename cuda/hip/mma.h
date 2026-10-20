// A HIP build reads this in place of CUDA's <mma.h>: the warp matrix API as rocWMMA provides it,
// under the name the kernels use. Fragment shapes and layouts are rocWMMA's: a kernel that
// reads fragment elements by position is not portable this way.
//
// rocWMMA serves the targets with matrix instructions (CDNA, RDNA3, RDNA4). The device pass of
// any other target (RDNA2) gets no API and `LOKEN_NO_WMMA`: a kernel tests it and compiles a
// body that is never launched there, since the host side selects a kernel without fragments for
// such a card. The host pass serves every target and keeps the API.
#pragma once
#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3) && !defined(RDNA4) && !defined(CDNA)
#define LOKEN_NO_WMMA 1
#else
#include <rocwmma/rocwmma.hpp>
namespace nvcuda {
namespace wmma = rocwmma;
}
#endif
