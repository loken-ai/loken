#include "mmq_common.cuh"
#include "mmq_gguf.cuh"
#include "mmq_launch.cuh"

// Experts only: a codebook format is a mixture's, so the expert-batched entry point is the one
// built for it.
MMQ_MOE_ENTRY_POINT(GGML_TYPE_IQ4_NL, iq4_nl)
