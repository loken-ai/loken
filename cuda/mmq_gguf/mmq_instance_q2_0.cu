#include "mmq_common.cuh"
#include "mmq_gguf.cuh"
#include "mmq_launch.cuh"

// Experts only: q2_0 is a mixture's format, so the expert-batched entry point is the one built
// for it.
MMQ_MOE_ENTRY_POINT(GGML_TYPE_Q2_0, q2_0)
