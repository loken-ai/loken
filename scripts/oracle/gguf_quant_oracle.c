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
// The type ids, and ggml's own conversions of f16 and E4M3 scales.
#include "ggml-impl.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ggml exposes these directly; declaring them here keeps the build to one translation unit
// plus ggml-quants.c and ggml-cpu/quants.c, with no cmake and no library.
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
void quantize_row_tq1_0_ref(const float *, block_tq1_0 *, int64_t);
void quantize_row_tq2_0_ref(const float *, block_tq2_0 *, int64_t);
void quantize_row_nvfp4_ref(const float *, block_nvfp4 *, int64_t);
// IQ1_S and IQ1_M have no `_ref` quantiser: theirs takes an importance per value and reads the
// codebook tables `iq2xs_init_impl` builds.
void iq2xs_init_impl(enum ggml_type);
size_t quantize_iq1_s(const float *, void *, int64_t, int64_t, const float *);
size_t quantize_iq1_m(const float *, void *, int64_t, int64_t, const float *);

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
void dequantize_row_tq1_0(const block_tq1_0 *, float *, int64_t);
void dequantize_row_tq2_0(const block_tq2_0 *, float *, int64_t);
void dequantize_row_nvfp4(const block_nvfp4 *, float *, int64_t);
void dequantize_row_iq1_s(const block_iq1_s *, float *, int64_t);
void dequantize_row_iq1_m(const block_iq1_m *, float *, int64_t);

// The scalar dot products of ggml's CPU backend, from ggml-cpu/quants.c: a weight row against an
// activation row quantised to q8_K or q8_0.
typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_tq1_0_q8_K_generic(int, float *, size_t, const void *, size_t, const void *,
                                     size_t, int);
void ggml_vec_dot_tq2_0_q8_K_generic(int, float *, size_t, const void *, size_t, const void *,
                                     size_t, int);
void ggml_vec_dot_iq1_s_q8_K_generic(int, float *, size_t, const void *, size_t, const void *,
                                     size_t, int);
void ggml_vec_dot_iq1_m_q8_K_generic(int, float *, size_t, const void *, size_t, const void *,
                                     size_t, int);
void ggml_vec_dot_nvfp4_q8_0_generic(int, float *, size_t, const void *, size_t, const void *,
                                     size_t, int);

// The lookup tables ggml-cpu/quants.c reads, which the CPU backend fills at its start.
float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_f8_e4m3[1 << 8];

// ggml-quants.c references these from paths this tool never enters - the other IQ formats and
// their assertions. Compiling ggml.c to satisfy them would drag in the whole graph layer for
// a few symbols. They abort instead of returning a plausible number: a stub that answers is how
// a reference quietly stops being one.
#include <stdlib.h>
size_t ggml_type_size(enum ggml_type t) {
    fprintf(stderr, "oracle: ggml_type_size(%d) reached - this tool calls only the _ref and "
                    "dequantize_row_* entry points, so no path should need it\n", (int)t);
    abort();
}
const char *ggml_type_name(enum ggml_type t) {
    fprintf(stderr, "oracle: ggml_type_name(%d) reached - see above\n", (int)t);
    abort();
}
// The IQ1 quantisers return the bytes they wrote, so the row size of those 2 formats answers.
size_t ggml_row_size(enum ggml_type t, int64_t n) {
    if (t == GGML_TYPE_IQ1_S) {
        return (size_t)(n / QK_K) * sizeof(block_iq1_s);
    }
    if (t == GGML_TYPE_IQ1_M) {
        return (size_t)(n / QK_K) * sizeof(block_iq1_m);
    }
    fprintf(stderr, "oracle: ggml_row_size(%d, %lld) reached - see above\n", (int)t, (long long)n);
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

// IQ1_S and IQ1_M quantise against an importance per value: every value weighs the same.
#define ROUNDTRIP_IQ1(NAME, BLK, TYPE, QFN, DQFN)                                                  \
    do {                                                                                           \
        BLK q[N / QK_K];                                                                           \
        float back[N];                                                                             \
        float importance[N];                                                                       \
        for (int i = 0; i < N; i++) {                                                              \
            importance[i] = 1.0f;                                                                  \
        }                                                                                          \
        iq2xs_init_impl(TYPE);                                                                     \
        QFN(input, q, 1, N, importance);                                                           \
        DQFN(q, back, N);                                                                          \
        printf("TYPE %s %d %zu\n", NAME, QK_K, sizeof(BLK));                                       \
        emit_bytes("QUANT", q, sizeof(q));                                                         \
        emit_floats("DEQUANT", back, N);                                                           \
    } while (0)

// Bytes in which every value appears once in each run of 256: what a decoder reads when the
// blocks were not written by an encoder.
static void byte_pattern(void *p, size_t n) {
    unsigned char *b = (unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        b[i] = (unsigned char)(i * 151 + 7);
    }
}

// The f16 scale the patterned blocks carry, 1.0, so that every value they decode to is finite.
#define PATTERN_SCALE 0x3c00

// Patterned blocks of a format and what ggml decodes them to, under the name `NAME.bytes`.
// `FIX` sets the scale fields of block `q[j]`.
#define DECODE(NAME, BLK, DQFN, BLOCK_ELEMS, FIX)                                                  \
    do {                                                                                           \
        BLK q[N / (BLOCK_ELEMS)];                                                                  \
        float back[N];                                                                             \
        byte_pattern(q, sizeof(q));                                                                \
        for (size_t j = 0; j < N / (BLOCK_ELEMS); j++) {                                           \
            FIX;                                                                                   \
        }                                                                                          \
        DQFN(q, back, N);                                                                          \
        printf("TYPE %s.bytes %d %zu\n", NAME, (BLOCK_ELEMS), sizeof(BLK));                        \
        emit_bytes("QUANT", q, sizeof(q));                                                         \
        emit_floats("DEQUANT", back, N);                                                           \
    } while (0)

// The dot of a format's blocks of the input against the input itself, quantised as ggml's CPU
// backend quantises an activation for that format. The result goes out as its bit pattern.
static void emit_dot(const char *name, vec_dot_fn dot, const void *weights, const void *acts) {
    float s = 0.0f;
    uint32_t bits;
    dot(N, &s, 0, weights, 0, acts, 0, 1);
    memcpy(&bits, &s, sizeof(bits));
    printf("DOT %s %08x\n", name, bits);
}

int main(void) {
    float input[N];
    reference_input(input);
    for (int i = 0; i < (1 << 16); i++) {
        union {
            uint16_t u16;
            ggml_fp16_t fp16;
        } u = {(uint16_t)i};
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
    }
    for (int i = 0; i < (1 << 8); i++) {
        ggml_table_f32_f8_e4m3[i] = ggml_f8_e4m3_to_fp32((uint8_t)i);
    }

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
    ROUNDTRIP("tq1_0", block_tq1_0, quantize_row_tq1_0_ref, dequantize_row_tq1_0, QK_K);
    ROUNDTRIP("tq2_0", block_tq2_0, quantize_row_tq2_0_ref, dequantize_row_tq2_0, QK_K);
    ROUNDTRIP("nvfp4", block_nvfp4, quantize_row_nvfp4_ref, dequantize_row_nvfp4, QK_NVFP4);
    ROUNDTRIP_IQ1("iq1_s", block_iq1_s, GGML_TYPE_IQ1_S, quantize_iq1_s, dequantize_row_iq1_s);
    ROUNDTRIP_IQ1("iq1_m", block_iq1_m, GGML_TYPE_IQ1_M, quantize_iq1_m, dequantize_row_iq1_m);

    DECODE("tq1_0", block_tq1_0, dequantize_row_tq1_0, QK_K, q[j].d = PATTERN_SCALE);
    DECODE("tq2_0", block_tq2_0, dequantize_row_tq2_0, QK_K, q[j].d = PATTERN_SCALE);
    DECODE("nvfp4", block_nvfp4, dequantize_row_nvfp4, QK_NVFP4, (void)0);
    DECODE("iq1_s", block_iq1_s, dequantize_row_iq1_s, QK_K, q[j].d = PATTERN_SCALE);
    // IQ1_M keeps nibble `w` of its f16 scale in the top nibble of scale word `w`.
    DECODE("iq1_m", block_iq1_m, dequantize_row_iq1_m, QK_K, {
        for (int w = 0; w < 4; w++) {
            unsigned char nibble = (PATTERN_SCALE >> (4 * w)) & 0xF;
            q[j].scales[2 * w + 1] = (unsigned char)((q[j].scales[2 * w + 1] & 0x0F) | (nibble << 4));
        }
    });

    // Every code of NVFP4's unsigned E4M3 scale, as ggml reads it.
    {
        float ue4m3[1 << 8];
        for (int i = 0; i < (1 << 8); i++) {
            ue4m3[i] = ggml_ue4m3_to_fp32((uint8_t)i);
        }
        emit_floats("UE4M3", ue4m3, 1 << 8);
    }

    // The dot of each format above against the input, through ggml's scalar dot products.
    {
        static block_q8_K act_k[N / QK_K];
        static block_q8_0 act_0[N / QK8_0];
        static block_tq1_0 tq1[N / QK_K];
        static block_tq2_0 tq2[N / QK_K];
        static block_nvfp4 nv[N / QK_NVFP4];
        static block_iq1_s iq1s[N / QK_K];
        static block_iq1_m iq1m[N / QK_K];
        float importance[N];
        for (int i = 0; i < N; i++) {
            importance[i] = 1.0f;
        }
        quantize_row_q8_K_ref(input, act_k, N);
        quantize_row_q8_0_ref(input, act_0, N);
        quantize_row_tq1_0_ref(input, tq1, N);
        quantize_row_tq2_0_ref(input, tq2, N);
        quantize_row_nvfp4_ref(input, nv, N);
        quantize_iq1_s(input, iq1s, 1, N, importance);
        quantize_iq1_m(input, iq1m, 1, N, importance);
        emit_dot("tq1_0", ggml_vec_dot_tq1_0_q8_K_generic, tq1, act_k);
        emit_dot("tq2_0", ggml_vec_dot_tq2_0_q8_K_generic, tq2, act_k);
        emit_dot("nvfp4", ggml_vec_dot_nvfp4_q8_0_generic, nv, act_0);
        emit_dot("iq1_s", ggml_vec_dot_iq1_s_q8_K_generic, iq1s, act_k);
        emit_dot("iq1_m", ggml_vec_dot_iq1_m_q8_K_generic, iq1m, act_k);
    }

    return 0;
}
