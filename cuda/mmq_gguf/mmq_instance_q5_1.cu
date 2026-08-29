#include "mmq_common.cuh"
#include "mmq_gguf.cuh"
#include "mmq_launch.cuh"

// One translation unit per format, because `mul_mat_q` is a heavy template expansion and the
// ten build in parallel. The launcher itself lives in `mmq_launch.cuh`, written once.
MMQ_ENTRY_POINT(GGML_TYPE_Q5_1, q5_1)
