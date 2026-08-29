// Reference vectors for the GGUF block formats, produced by ggml itself.
//
// It calls ggml's `_ref` entry points - the canonical scalar definitions, not the SIMD variants
// that vary by host - so the vectors describe the format rather than one machine's dispatch.
//
// Everything here is deterministic and closed-form: no RNG, no clock, no input file. The same
// bytes come out anywhere, which is what makes a bit-exact comparison meaningful.
//
// BUILD (llama.cpp is read as a reference; nothing is copied from it):
//   LLAMA_CPP=/path/to/llama.cpp make -C scripts/oracle
//
// The output is written to stdout. Redirect it to `scripts/oracle/vectors.txt` and the
// `oracle_parity` tests will read it.

// ggml-common.h serves several languages from one file and declares nothing until told which.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ggml exposes these directly; declaring them here keeps the build to one translation unit
// plus ggml-quants.c, with no cmake and no library.
void quantize_row_q4_0_ref(const float *, block_q4_0 *, int64_t);
void quantize_row_q4_1_ref(const float *, block_q4_1 *, int64_t);
void quantize_row_q5_0_ref(const float *, block_q5_0 *, int64_t);
void quantize_row_q5_1_ref(const float *, block_q5_1 *, int64_t);
void quantize_row_q8_0_ref(const float *, block_q8_0 *, int64_t);
void quantize_row_q2_K_ref(const float *, block_q2_K *, int64_t);
void quantize_row_q3_K_ref(const float *, block_q3_K *, int64_t);
void quantize_row_q4_K_ref(const float *, block_q4_K *, int64_t);
void quantize_row_q5_K_ref(const float *, block_q5_K *, int64_t);
void quantize_row_q6_K_ref(const float *, block_q6_K *, int64_t);
void quantize_row_q8_K_ref(const float *, block_q8_K *, int64_t);
void quantize_row_mxfp4_ref(const float *, block_mxfp4 *, int64_t);

void dequantize_row_q4_0(const block_q4_0 *, float *, int64_t);
void dequantize_row_q4_1(const block_q4_1 *, float *, int64_t);
void dequantize_row_q5_0(const block_q5_0 *, float *, int64_t);
void dequantize_row_q5_1(const block_q5_1 *, float *, int64_t);
void dequantize_row_q8_0(const block_q8_0 *, float *, int64_t);
void dequantize_row_q2_K(const block_q2_K *, float *, int64_t);
void dequantize_row_q3_K(const block_q3_K *, float *, int64_t);
void dequantize_row_q4_K(const block_q4_K *, float *, int64_t);
void dequantize_row_q5_K(const block_q5_K *, float *, int64_t);
void dequantize_row_q6_K(const block_q6_K *, float *, int64_t);
void dequantize_row_q8_K(const block_q8_K *, float *, int64_t);
void dequantize_row_mxfp4(const block_mxfp4 *, float *, int64_t);

// ggml-quants.c references these two from paths this tool never enters - the IQ formats and
// their assertions. Compiling ggml.c to satisfy them would drag in the whole graph layer for
// two symbols. They abort instead of returning a plausible number: a stub that answers is how
// a reference quietly stops being one.
#include <stdlib.h>
size_t ggml_type_size(int t) {
    fprintf(stderr, "oracle: ggml_type_size(%d) reached - this tool calls only the _ref and "
                    "dequantize_row_* entry points, so no path should need it\n", t);
    abort();
}
const char *ggml_type_name(int t) {
    fprintf(stderr, "oracle: ggml_type_name(%d) reached - see above\n", t);
    abort();
}
size_t ggml_row_size(int t, int64_t n) {
    fprintf(stderr, "oracle: ggml_row_size(%d, %lld) reached - see above\n", t, (long long)n);
    abort();
}
void ggml_abort(const char *file, int line, const char *fmt, ...) {
    fprintf(stderr, "oracle: ggml_abort at %s:%d - %s\n", file, line, fmt);
    abort();
}

// 1024 is a multiple of 32 and of 256, so every block size divides it evenly and no format is
// measured on a partial block.
#define N 1024

// Closed form rather than a generator: the values must be identical in C and in Rust, and a
// shared PRNG is one more thing that can disagree. The mixture is deliberate - a smooth term
// so neighbouring weights correlate the way real ones do, a sawtooth so each block sees a
// different range and the per-block scale search is actually exercised, and one large outlier
// per block, which is where a quantiser's clamping shows up.
static void reference_input(float *dst) {
    for (int i = 0; i < N; i++) {
        float smooth = 2.0f * cosf(0.017f * (float)i);
        float saw = 0.03f * (float)(i % 37) - 0.5f;
        float spike = (i % 256 == 113) ? 6.5f : 0.0f;
        dst[i] = 0.1f + smooth + saw + spike;
    }
}

static void emit_bytes(const char *label, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    printf("%s %zu ", label, n);
    for (size_t i = 0; i < n; i++) {
        printf("%02x", b[i]);
    }
    printf("\n");
}

// Floats go out as their bit patterns. Printing decimals would put the C library's formatting
// between the two implementations, and a comparison that is only as exact as `%.9g` is not a
// bit-exact comparison.
static void emit_floats(const char *label, const float *f, size_t n) {
    printf("%s %zu ", label, n);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &f[i], sizeof(bits));
        printf("%08x", bits);
    }
    printf("\n");
}

#define ROUNDTRIP(NAME, BLK, QFN, DQFN, BLOCK_ELEMS)                                               \
    do {                                                                                           \
        BLK q[N / (BLOCK_ELEMS)];                                                                  \
        float back[N];                                                                             \
        QFN(input, q, N);                                                                          \
        DQFN(q, back, N);                                                                          \
        printf("TYPE %s %d %zu\n", NAME, (BLOCK_ELEMS), sizeof(BLK));                              \
        emit_bytes("QUANT", q, sizeof(q));                                                         \
        emit_floats("DEQUANT", back, N);                                                           \
    } while (0)

int main(void) {
    float input[N];
    reference_input(input);

    printf("# GGUF block-format reference vectors, produced by ggml's own scalar entry points.\n");
    printf("# Regenerate with: LLAMA_CPP=<path> make -C scripts/oracle && "
           "scripts/oracle/gguf_quant_oracle > scripts/oracle/vectors.txt\n");
    printf("N %d\n", N);
    emit_floats("INPUT", input, N);

    ROUNDTRIP("q4_0", block_q4_0, quantize_row_q4_0_ref, dequantize_row_q4_0, QK4_0);
    ROUNDTRIP("q4_1", block_q4_1, quantize_row_q4_1_ref, dequantize_row_q4_1, QK4_1);
    ROUNDTRIP("q5_0", block_q5_0, quantize_row_q5_0_ref, dequantize_row_q5_0, QK5_0);
    ROUNDTRIP("q5_1", block_q5_1, quantize_row_q5_1_ref, dequantize_row_q5_1, QK5_1);
    ROUNDTRIP("q8_0", block_q8_0, quantize_row_q8_0_ref, dequantize_row_q8_0, QK8_0);
    ROUNDTRIP("q2_K", block_q2_K, quantize_row_q2_K_ref, dequantize_row_q2_K, QK_K);
    ROUNDTRIP("q3_K", block_q3_K, quantize_row_q3_K_ref, dequantize_row_q3_K, QK_K);
    ROUNDTRIP("q4_K", block_q4_K, quantize_row_q4_K_ref, dequantize_row_q4_K, QK_K);
    ROUNDTRIP("q5_K", block_q5_K, quantize_row_q5_K_ref, dequantize_row_q5_K, QK_K);
    ROUNDTRIP("q6_K", block_q6_K, quantize_row_q6_K_ref, dequantize_row_q6_K, QK_K);
    ROUNDTRIP("q8_K", block_q8_K, quantize_row_q8_K_ref, dequantize_row_q8_K, QK_K);
    ROUNDTRIP("mxfp4", block_mxfp4, quantize_row_mxfp4_ref, dequantize_row_mxfp4, QK_MXFP4);

    return 0;
}
