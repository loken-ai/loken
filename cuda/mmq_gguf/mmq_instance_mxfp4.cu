#include "mmq_common.cuh"
#include "mmq_gguf.cuh"
#include "mmq_launch.cuh"

// Experts only: an MXFP4 stack is a mixture's, so the expert-batched entry point is the one
// built for it.
MMQ_MOE_ENTRY_POINT(GGML_TYPE_MXFP4, mxfp4)
