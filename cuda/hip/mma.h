// A HIP build reads this in place of CUDA's <mma.h>: the warp matrix API as rocWMMA provides it,
// under the name the kernels use. Fragment shapes and layouts are rocWMMA's: a kernel that
// reads fragment elements by position is not portable this way.
#pragma once
#include <rocwmma/rocwmma.hpp>
namespace nvcuda {
namespace wmma = rocwmma;
}
