// Decoder layers over the blocks of a GGUF file, for the OpenCL stack.
//
// The program is built once per row count R: R activation rows share each read of a weight, so
// a decode step (R = 1) and a prompt chunk (R = 8) read the weights once either way.
//
// The quantised formats multiply as the host does: the activation row is quantised into
// 8-bit codes (q8_0 blocks of 32 for the legacy formats, q8_K blocks of 256 for the k-quants),
// the weight codes meet them in integer dot products, and the block scales come out of the
// sum. F16 and F32 weights read the activations as floats.

#ifndef R
#define R 1
#endif

#define WG 64
#define LANES 8
// Slices of a row's groups worked on at once: SLICES * LANES lanes share a row.
#ifndef SLICES
#define SLICES 4
#endif
// Rows a thread takes together, so that each activation load serves RB weight rows.
#ifndef RB
#define RB 4
#endif
#define SLOTS (WG / (LANES * SLICES))
#define ROWS_PER_WG (SLOTS * RB)

#define FLAG_BIAS 1u
#define FLAG_RESIDUAL 2u

#if defined(__opencl_c_integer_dot_product_input_4x8bit_packed)
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
// Four unsigned bytes of a against four signed bytes of b, added to c.
#define DP_US(a, b, c) dot_acc_sat_4x8packed_us_int(a, b, c)
#define DP_SS(a, b, c) dot_acc_sat_4x8packed_ss_int(a, b, c)
#else
inline int dp_us(uint a, uint b, int c) {
    return c + (int)(a & 255u) * (int)(char)(b & 255u) + (int)((a >> 8) & 255u) * (int)(char)(b >> 8) +
           (int)((a >> 16) & 255u) * (int)(char)(b >> 16) + (int)(a >> 24) * (int)(char)(b >> 24);
}
inline int dp_ss(uint a, uint b, int c) {
    return c + (int)(char)(a) * (int)(char)(b) + (int)(char)(a >> 8) * (int)(char)(b >> 8) +
           (int)(char)(a >> 16) * (int)(char)(b >> 16) + (int)(char)(a >> 24) * (int)(char)(b >> 24);
}
#define DP_US(a, b, c) dp_us(a, b, c)
#define DP_SS(a, b, c) dp_ss(a, b, c)
#endif

// Half to float by arithmetic, so that the weights can be read as aligned words.
inline float h2f(uint h) {
    return as_float(((h & 0x8000u) << 16) | ((h & 0x7fffu) << 13)) * 0x1p112f;
}

// The four bytes of a word as floats.
inline float4 lo4(uint v) {
    return convert_float4(as_uchar4(v & 0x0f0f0f0fu));
}
inline float4 hi4(uint v) {
    return convert_float4(as_uchar4((v >> 4) & 0x0f0f0f0fu));
}
inline float4 h4(uint a, uint b) {
    return (float4)(h2f(a & 0xffffu), h2f(a >> 16), h2f(b & 0xffffu), h2f(b >> 16));
}
// The four low or high nibbles of a word, as unsigned bytes.
inline uint lo_nibbles(uint v) {
    return v & 0x0f0f0f0fu;
}
inline uint hi_nibbles(uint v) {
    return (v >> 4) & 0x0f0f0f0fu;
}

// An activation row quantised for the integer formats, one row after another:
//   qk   the q8_K codes, cols of them; dk the scale of each block of 256;
//   sk   the sum of the codes of each sub-block of 32;
//   qz   the q8_0 codes, cols of them; dz the scale of each block of 32.
#ifdef MEGA
// In the single-launch layer the activations are in local memory.
typedef struct {
    __local const char* qk;
    __local const char* qz;
    __local const float* dk;
    __local const int* sk;
    __local const float* dz;
    __local const float* xf;
} XQ;
#else
typedef struct {
    __global const char* qk;
    __global const char* qz;
    __global const float* dk;
    __global const int* sk;
    __global const float* dz;
} XQ;
#endif

#define ACT_ARGS                                                                                 \
    __global const float *x, __global const char *qk, __global const char *qz,                   \
        __global const float *dk, __global const int *sk, __global const float *dz

// Weights meet activations in groups of 256 values, eight sub-blocks of 32. The eight lanes
// of a row share a group: lane i takes values 4i..4i+3 of every sub-block, so the loads of a
// sub-block are one contiguous run across the lanes (a word of codes per lane). A format's
// `grp_<fmt>` adds the group's dot products with each of the nr activation rows to accA[r],
// and those of a second weight, when use_b, to accB[r].
//
//   wa, wb    the weights;           total    their blocks;
//   blk0      the group's first sub-block of 32 values in the matrix;
//   nsub      the sub-blocks the row has left in the group (8, but for a row of 32-value
//             blocks that does not end on a group);
//   g, nb     the group in the row, and the row's sub-blocks;
//   xf, xq    the activations as floats and as codes, at row 0;  cols  their row stride;
//   xw, xs    for a single row, the group's eight code words of this lane and eight scales
//             (legacy) or code sums (k-quants), and xd the block scale (k-quants).
#define GRP_ARGS                                                                                 \
    __global const uchar *wa, __global const uchar *wb, uint use_b, ulong total, ulong blk0,     \
        uint nsub, uint lane, uint g, uint nb, __global const float *xf, XQ xq, const uint *xw,  \
        const float *xs, float xd, uint cols, uint nr, float *accA, float *accB

#define UNROLL _Pragma("unroll")

#define EIGHT(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7)

// The code word, scale and (k-quant) code sum of sub-block C of row r.
#if defined(MEGA)
#define LW(r, C) (((__local const uint*)(xq.qz + 256 * g + 32 * (C)))[lane])
#define LD(r, C) (xq.dz[8 * g + (C)])
#define KW(r, C) (((__local const uint*)(xq.qk + 256 * g + 32 * (C)))[lane])
#define KS(r, C) ((float)xq.sk[8 * g + (C)])
#define KD(r) (xq.dk[g])
#elif R == 1
#define LW(r, C) (xw[C])
#define LD(r, C) (xs[C])
#define KW(r, C) (xw[C])
#define KS(r, C) (xs[C])
#define KD(r) (xd)
#else
#define LW(r, C) (((__global const uint*)(xq.qz + (ulong)(r) * cols + 256 * g + 32 * (C)))[lane])
#define LD(r, C) (xq.dz[(ulong)(r) * nb + 8 * g + (C)])
#define KW(r, C) (((__global const uint*)(xq.qk + (ulong)(r) * cols + 256 * g + 32 * (C)))[lane])
#define KS(r, C) ((float)xq.sk[(ulong)(r) * nb + 8 * g + (C)])
#define KD(r) (xq.dk[(ulong)(r) * (cols / 256) + g])
#endif

// ---- Q8_0: 32 quants per block, then the half scales.
inline void grp_q8_0(GRP_ARGS) {
#define STEP(cc)                                                                                 \
    if (cc < nsub) {                                                                             \
        ulong blk = blk0 + cc;                                                                   \
        float da = h2f(((__global const ushort*)(wa + total * 32))[blk]);                        \
        uint wqa = ((__global const uint*)(wa + blk * 32))[lane];                                \
        float db = 0;                                                                            \
        uint wqb = 0;                                                                            \
        if (use_b) {                                                                             \
            db = h2f(((__global const ushort*)(wb + total * 32))[blk]);                          \
            wqb = ((__global const uint*)(wb + blk * 32))[lane];                                 \
        }                                                                                        \
        UNROLL for (uint r = 0; r < R; r++) {                                                           \
            if (r >= nr) break;                                                                  \
            uint xv = LW(r, cc);                                                                 \
            float sc = LD(r, cc);                                                                \
            accA[r] += da * sc * (float)DP_SS(wqa, xv, 0);                                       \
            if (use_b) accB[r] += db * sc * (float)DP_SS(wqb, xv, 0);                            \
        }                                                                                        \
    }
    EIGHT(STEP)
#undef STEP
}

// ---- Q4_0: 16 bytes of nibbles per block (low = values 0..15, high = 16..31), then the scales.
inline uint q4_0_nibbles(__global const uchar* w, ulong blk, uint lane) {
    uint dw = ((__global const uint*)(w + blk * 16))[lane & 3];
    return lane < 4 ? lo_nibbles(dw) : hi_nibbles(dw);
}
inline void grp_q4_0(GRP_ARGS) {
#define STEP(cc)                                                                                 \
    if (cc < nsub) {                                                                             \
        ulong blk = blk0 + cc;                                                                   \
        float da = h2f(((__global const ushort*)(wa + total * 16))[blk]);                        \
        uint wqa = q4_0_nibbles(wa, blk, lane);                                                  \
        float db = 0;                                                                            \
        uint wqb = 0;                                                                            \
        if (use_b) {                                                                             \
            db = h2f(((__global const ushort*)(wb + total * 16))[blk]);                          \
            wqb = q4_0_nibbles(wb, blk, lane);                                                   \
        }                                                                                        \
        UNROLL for (uint r = 0; r < R; r++) {                                                           \
            if (r >= nr) break;                                                                  \
            uint xv = LW(r, cc);                                                                 \
            float sc = LD(r, cc);                                                                \
            int sa = DP_SS(0x01010101u, xv, 0);                                                  \
            accA[r] += da * sc * (float)(DP_US(wqa, xv, 0) - 8 * sa);                            \
            if (use_b) accB[r] += db * sc * (float)(DP_US(wqb, xv, 0) - 8 * sa);                 \
        }                                                                                        \
    }
    EIGHT(STEP)
#undef STEP
}

// ---- Q5_0: Q4_0's nibbles, then one word of fifth bits per block, then the scales.
inline uint q5_0_codes(__global const uchar* w, ulong total, ulong blk, uint lane) {
    uint n = q4_0_nibbles(w, blk, lane);
    uint qh = (((__global const uint*)(w + total * 16))[blk] >> (4 * lane)) & 15u;
    // Bit k of the four fifth bits to bit 4 of byte k.
    uint top = (qh * 0x00204081u) & 0x01010101u;
    return n | (top << 4);
}
inline void grp_q5_0(GRP_ARGS) {
#define STEP(cc)                                                                                 \
    if (cc < nsub) {                                                                             \
        ulong blk = blk0 + cc;                                                                   \
        float da = h2f(((__global const ushort*)(wa + total * 20))[blk]);                        \
        uint wqa = q5_0_codes(wa, total, blk, lane);                                             \
        float db = 0;                                                                            \
        uint wqb = 0;                                                                            \
        if (use_b) {                                                                             \
            db = h2f(((__global const ushort*)(wb + total * 20))[blk]);                          \
            wqb = q5_0_codes(wb, total, blk, lane);                                              \
        }                                                                                        \
        UNROLL for (uint r = 0; r < R; r++) {                                                           \
            if (r >= nr) break;                                                                  \
            uint xv = LW(r, cc);                                                                 \
            float sc = LD(r, cc);                                                                \
            int sa = DP_SS(0x01010101u, xv, 0);                                                  \
            accA[r] += da * sc * (float)(DP_US(wqa, xv, 0) - 16 * sa);                           \
            if (use_b) accB[r] += db * sc * (float)(DP_US(wqb, xv, 0) - 16 * sa);                \
        }                                                                                        \
    }
    EIGHT(STEP)
#undef STEP
}

// ---- F16 / F32: as stored, against the activations as floats.
#define FLOAT_STEP_BODY(C, QA, QB)                                                                  \
    UNROLL for (uint r = 0; r < R; r++) {                                                               \
        if (r >= nr) break;                                                                      \
        float4 xv = XF(r, C);                                                                   \
        accA[r] += dot(QA, xv);                                                                  \
        if (use_b) accB[r] += dot(QB, xv);                                                       \
    }
#if defined(MEGA)
#define XF(r, C) vload4(8 * (C) + lane, xq.xf + 256 * g)
#elif R == 1
#define XF(r, C) (xc[C])
#else
#define XF(r, C) vload4(8 * (C) + lane, xf + 256 * g + (ulong)(r) * cols)
#endif
inline float4 f16_values(__global const uchar* w, ulong blk, uint lane) {
    uint2 v = ((__global const uint2*)(w + blk * 64))[lane];
    return h4(v.x, v.y);
}
inline void grp_f16(GRP_ARGS, const float4* xc) {
#define STEP(cc)                                                                                 \
    if (cc < nsub) {                                                                             \
        ulong blk = blk0 + cc;                                                                   \
        float4 qa = f16_values(wa, blk, lane);                                                   \
        float4 qb = 0;                                                                           \
        if (use_b) qb = f16_values(wb, blk, lane);                                               \
        FLOAT_STEP_BODY(cc, qa, qb)                                                                  \
    }
    EIGHT(STEP)
#undef STEP
}
inline void grp_f32(GRP_ARGS, const float4* xc) {
#define STEP(cc)                                                                                 \
    if (cc < nsub) {                                                                             \
        ulong blk = blk0 + cc;                                                                   \
        float4 qa = ((__global const float4*)(wa + blk * 128))[lane];                            \
        float4 qb = 0;                                                                           \
        if (use_b) qb = ((__global const float4*)(wb + blk * 128))[lane];                        \
        FLOAT_STEP_BODY(cc, qa, qb)                                                                  \
    }
    EIGHT(STEP)
#undef STEP
}

// ---- Q4_K: 144 bytes per 256 values: half d, half dmin, 12 bytes of 6-bit scales and minimums
// for the 8 sub-blocks, 128 bytes of nibbles; sub-blocks 2g and 2g + 1 share bytes 32g..32g+31,
// the first in the low nibbles and the second in the high ones.
// The scale and minimum of sub-block j from the 12 scale bytes, held as three words.
inline void scale_min_k4(int j, uint s0, uint s1, uint s2, float* sc, float* mn) {
    if (j < 4) {
        uint sh = 8 * j;
        *sc = (float)((s0 >> sh) & 63);
        *mn = (float)((s1 >> sh) & 63);
    } else {
        uint sh = 8 * (j - 4);
        uint lo = (s2 >> sh) & 255;
        *sc = (float)((lo & 0xF) | (((s0 >> sh) >> 6 & 3) << 4));
        *mn = (float)((lo >> 4) | (((s1 >> sh) >> 6 & 3) << 4));
    }
}
inline void grp_q4_k(GRP_ARGS) {
    ulong sb = blk0 >> 3;
    uint4 ha = ((__global const uint4*)(wa + sb * 144))[0];
    uint4 hb = (uint4)(0);
    uint qa[4], qb[4];
    UNROLL for (int k = 0; k < 4; k++) qa[k] = ((__global const uint*)(wa + sb * 144 + 16))[8 * k + lane];
    UNROLL for (int k = 0; k < 4; k++) qb[k] = 0;
    if (use_b) {
        hb = ((__global const uint4*)(wb + sb * 144))[0];
        UNROLL for (int k = 0; k < 4; k++) qb[k] = ((__global const uint*)(wb + sb * 144 + 16))[8 * k + lane];
    }
    float da = h2f(ha.x & 0xffffu), dma = h2f(ha.x >> 16);
    float db = h2f(hb.x & 0xffffu), dmb = h2f(hb.x >> 16);
    float first = lane == 0 ? 1.0f : 0.0f;
#define STEP(cc)                                                                                 \
    {                                                                                            \
        float sca, mna, scb = 0, mnb = 0;                                                        \
        scale_min_k4(cc, ha.y, ha.z, ha.w, &sca, &mna);                                          \
        if (use_b) scale_min_k4(cc, hb.y, hb.z, hb.w, &scb, &mnb);                               \
        uint wqa = ((cc) & 1) ? hi_nibbles(qa[(cc) >> 1]) : lo_nibbles(qa[(cc) >> 1]);           \
        uint wqb = ((cc) & 1) ? hi_nibbles(qb[(cc) >> 1]) : lo_nibbles(qb[(cc) >> 1]);           \
        UNROLL for (uint r = 0; r < R; r++) {                                                           \
            if (r >= nr) break;                                                                  \
            uint xv = KW(r, cc);                                                                 \
            float s = KS(r, cc), y = KD(r);                                                      \
            accA[r] += y * (da * sca * (float)DP_US(wqa, xv, 0) - first * dma * mna * s);        \
            if (use_b)                                                                           \
                accB[r] += y * (db * scb * (float)DP_US(wqb, xv, 0) - first * dmb * mnb * s);    \
        }                                                                                        \
    }
    EIGHT(STEP)
#undef STEP
}

// ---- Q6_K, stored as four arrays over the superblocks: 128 bytes of low nibbles, 64 of high
// bit pairs, 16 signed scales, one half scale. Sub-block c is quarter c & 3 of half c >> 2: its
// values come from the low (c & 3 < 2) or high nibbles of bytes 64h + 32 (c & 1) + l, and the
// bit pair 2 (c & 3) of byte 32h + l.
inline uint six_codes(uint a, uint e, uint shift, uint high) {
    uint lowbits = high ? hi_nibbles(a) : lo_nibbles(a);
    uint top = (e >> shift) & 0x03030303u;
    return lowbits | (top << 4);
}
inline void grp_q6_k(GRP_ARGS) {
    ulong sb = blk0 >> 3;
    float da = h2f(((__global const ushort*)(wa + total * 208))[sb]);
    float db = 0;
    if (use_b) db = h2f(((__global const ushort*)(wb + total * 208))[sb]);
    uint qla[4], qha[2], qlb[4], qhb[2];
    ulong sca[2], scb[2];
    UNROLL for (int h = 0; h < 2; h++) {
        qla[2 * h] = ((__global const uint*)(wa + sb * 128 + 64 * h))[lane];
        qla[2 * h + 1] = ((__global const uint*)(wa + sb * 128 + 64 * h + 32))[lane];
        qha[h] = ((__global const uint*)(wa + total * 128 + sb * 64 + 32 * h))[lane];
        sca[h] = ((__global const ulong*)(wa + total * 192 + sb * 16))[h];
        qlb[2 * h] = qlb[2 * h + 1] = 0;
        qhb[h] = 0;
        scb[h] = 0;
        if (use_b) {
            qlb[2 * h] = ((__global const uint*)(wb + sb * 128 + 64 * h))[lane];
            qlb[2 * h + 1] = ((__global const uint*)(wb + sb * 128 + 64 * h + 32))[lane];
            qhb[h] = ((__global const uint*)(wb + total * 128 + sb * 64 + 32 * h))[lane];
            scb[h] = ((__global const ulong*)(wb + total * 192 + sb * 16))[h];
        }
    }
    uint half_of_lane = lane >> 2;
#define STEP(cc)                                                                                 \
    {                                                                                            \
        uint wqa = six_codes(qla[2 * ((cc) >> 2) + ((cc) & 1)], qha[(cc) >> 2],                  \
                             2 * ((cc) & 3), ((cc) & 3) >> 1);                                   \
        uint wqb = six_codes(qlb[2 * ((cc) >> 2) + ((cc) & 1)], qhb[(cc) >> 2],                  \
                             2 * ((cc) & 3), ((cc) & 3) >> 1);                                   \
        uint so = 8 * (half_of_lane + 2 * ((cc) & 3));                                           \
        float sa = (float)(char)(sca[(cc) >> 2] >> so);                                          \
        float sb2 = (float)(char)(scb[(cc) >> 2] >> so);                                         \
        UNROLL for (uint r = 0; r < R; r++) {                                                           \
            if (r >= nr) break;                                                                  \
            uint xv = KW(r, cc);                                                                 \
            float y = KD(r);                                                                     \
            int s32 = DP_SS(0x01010101u, xv, 0);                                                 \
            accA[r] += y * da * sa * (float)(DP_US(wqa, xv, 0) - 32 * s32);                      \
            if (use_b) accB[r] += y * db * sb2 * (float)(DP_US(wqb, xv, 0) - 32 * s32);          \
        }                                                                                        \
    }
    EIGHT(STEP)
#undef STEP
}

#ifndef MEGA
// Per-class preloads for a decode step: the group's eight activation words and scales.
#if R == 1
#define PRELOAD_LEGACY()                                                                         \
    UNROLL for (int c = 0; c < 8; c++) {                                                                \
        bool live = 8 * g + c < nb;                                                              \
        xw[c] = live ? ((__global const uint*)(qz + 256 * g + 32 * c))[lane] : 0;                \
        xs[c] = live ? dz[8 * g + c] : 0.0f;                                                     \
    }
#define PRELOAD_KQUANT()                                                                         \
    UNROLL for (int c = 0; c < 8; c++) {                                                                \
        bool live = 8 * g + c < nb;                                                              \
        xw[c] = live ? ((__global const uint*)(qk + 256 * g + 32 * c))[lane] : 0;                \
        xs[c] = live ? (float)sk[8 * g + c] : 0.0f;                                              \
    }                                                                                            \
    xd = dk[g];
#define PRELOAD_FLOAT()                                                                          \
    UNROLL for (int c = 0; c < 8; c++)                                                                  \
        xc[c] = (8 * g + c < nb) ? vload4(8 * c + lane, x + 256 * g) : (float4)0;
#else
#define PRELOAD_LEGACY()
#define PRELOAD_KQUANT()
#define PRELOAD_FLOAT()
#endif

// The matrix-vector kernel of a format: WG threads cover ROWS_PER_WG rows, SLICES * LANES
// lanes each; slice s of a row takes its groups s, s + SLICES, ... The output takes the
// optional bias and the optional residual. `PRELOAD` and `EXTRA` say how the format reads its
// activations.
#define MATVEC(NAME, GRP, PRELOAD, EXTRA)                                                        \
    __kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void NAME(                          \
        __global const uchar* w, ACT_ARGS, __global float* out, uint cols, uint rows, uint nr,   \
        __global const float* bias, __global const float* res, uint flags, ulong total) {        \
        __local float part[SLOTS][SLICES * LANES][RB * R];                                       \
        uint tid = get_local_id(0);                                                              \
        uint rl = tid / (SLICES * LANES), sl = tid % (SLICES * LANES);                           \
        uint lane = sl % LANES, slice = sl / LANES;                                              \
        uint row0 = (get_group_id(0) * SLOTS + rl) * RB;                                         \
        XQ xq = {qk, qz, dk, sk, dz};                                                            \
        float acc[RB][R], none[R];                                                               \
        UNROLL for (uint i = 0; i < RB; i++)                                                            \
            UNROLL for (uint r = 0; r < R; r++) acc[i][r] = 0;                                          \
        UNROLL for (uint r = 0; r < R; r++) none[r] = 0;                                                \
        uint nb = cols / 32, ng = (nb + 7) / 8;                                                  \
        for (uint g = slice; g < ng; g += SLICES) {                                              \
            uint xw[8];                                                                          \
            float xs[8], xd = 0;                                                                 \
            float4 xc[8];                                                                        \
            PRELOAD()                                                                            \
            UNROLL for (uint i = 0; i < RB; i++) {                                                      \
                uint row = row0 + i;                                                             \
                if (row < rows)                                                                  \
                    GRP(w, w, 0, total, (ulong)row * nb + 8 * g, min(8u, nb - 8 * g), lane, g,   \
                        nb, x, xq, xw, xs, xd, cols, nr, acc[i], none EXTRA);                    \
            }                                                                                    \
        }                                                                                        \
        UNROLL for (uint i = 0; i < RB; i++)                                                            \
            UNROLL for (uint r = 0; r < R; r++) part[rl][sl][i * R + r] = acc[i][r];                    \
        barrier(CLK_LOCAL_MEM_FENCE);                                                            \
        if (tid < SLOTS * RB * R) {                                                              \
            uint slot = tid / (RB * R), i = (tid / R) % RB, r = tid % R;                         \
            uint orow = (get_group_id(0) * SLOTS + slot) * RB + i;                               \
            if (orow < rows && r < nr) {                                                         \
                float s = 0;                                                                     \
                for (uint l = 0; l < SLICES * LANES; l++) s += part[slot][l][i * R + r];         \
                if (flags & FLAG_BIAS) s += bias[orow];                                          \
                if (flags & FLAG_RESIDUAL) s += res[(ulong)r * rows + orow];                     \
                out[(ulong)r * rows + orow] = s;                                                 \
            }                                                                                    \
        }                                                                                        \
    }

#define NOTHING
#define EXTRA_XC , xc
MATVEC(mv_q8_0, grp_q8_0, PRELOAD_LEGACY, NOTHING)
MATVEC(mv_q4_0, grp_q4_0, PRELOAD_LEGACY, NOTHING)
MATVEC(mv_q5_0, grp_q5_0, PRELOAD_LEGACY, NOTHING)
MATVEC(mv_f16, grp_f16, PRELOAD_FLOAT, EXTRA_XC)
MATVEC(mv_f32, grp_f32, PRELOAD_FLOAT, EXTRA_XC)
MATVEC(mv_q4_k, grp_q4_k, PRELOAD_KQUANT, NOTHING)
MATVEC(mv_q6_k, grp_q6_k, PRELOAD_KQUANT, NOTHING)

// Gate and up projections of one format in a single pass: out = silu(gate . x) * (up . x).
#define MATVEC_GATE_UP(NAME, GRP, PRELOAD, EXTRA)                                                \
    __kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void NAME(                          \
        __global const uchar* wg, __global const uchar* wu, ACT_ARGS, __global float* out,       \
        uint cols, uint rows, uint nr, ulong total) {                                            \
        __local float pg[SLOTS][SLICES * LANES][RB * R];                                         \
        __local float pu[SLOTS][SLICES * LANES][RB * R];                                         \
        uint tid = get_local_id(0);                                                              \
        uint rl = tid / (SLICES * LANES), sl = tid % (SLICES * LANES);                           \
        uint lane = sl % LANES, slice = sl / LANES;                                              \
        uint row0 = (get_group_id(0) * SLOTS + rl) * RB;                                         \
        XQ xq = {qk, qz, dk, sk, dz};                                                            \
        float ag[RB][R], au[RB][R];                                                              \
        UNROLL for (uint i = 0; i < RB; i++)                                                            \
            UNROLL for (uint r = 0; r < R; r++) { ag[i][r] = 0; au[i][r] = 0; }                         \
        uint nb = cols / 32, ng = (nb + 7) / 8;                                                  \
        for (uint g = slice; g < ng; g += SLICES) {                                              \
            uint xw[8];                                                                          \
            float xs[8], xd = 0;                                                                 \
            float4 xc[8];                                                                        \
            PRELOAD()                                                                            \
            UNROLL for (uint i = 0; i < RB; i++) {                                                      \
                uint row = row0 + i;                                                             \
                if (row < rows)                                                                  \
                    GRP(wg, wu, 1, total, (ulong)row * nb + 8 * g, min(8u, nb - 8 * g), lane, g, \
                        nb, x, xq, xw, xs, xd, cols, nr, ag[i], au[i] EXTRA);                    \
            }                                                                                    \
        }                                                                                        \
        UNROLL for (uint i = 0; i < RB; i++)                                                            \
            UNROLL for (uint r = 0; r < R; r++) {                                                       \
                pg[rl][sl][i * R + r] = ag[i][r];                                                \
                pu[rl][sl][i * R + r] = au[i][r];                                                \
            }                                                                                    \
        barrier(CLK_LOCAL_MEM_FENCE);                                                            \
        if (tid < SLOTS * RB * R) {                                                              \
            uint slot = tid / (RB * R), i = (tid / R) % RB, r = tid % R;                         \
            uint orow = (get_group_id(0) * SLOTS + slot) * RB + i;                               \
            if (orow < rows && r < nr) {                                                         \
                float g = 0, v = 0;                                                              \
                for (uint l = 0; l < SLICES * LANES; l++) {                                      \
                    g += pg[slot][l][i * R + r];                                                 \
                    v += pu[slot][l][i * R + r];                                                 \
                }                                                                                \
                out[(ulong)r * rows + orow] = g / (1.0f + exp(-g)) * v;                          \
            }                                                                                    \
        }                                                                                        \
    }

MATVEC_GATE_UP(gu_q8_0, grp_q8_0, PRELOAD_LEGACY, NOTHING)
MATVEC_GATE_UP(gu_q4_0, grp_q4_0, PRELOAD_LEGACY, NOTHING)
MATVEC_GATE_UP(gu_q5_0, grp_q5_0, PRELOAD_LEGACY, NOTHING)
MATVEC_GATE_UP(gu_f16, grp_f16, PRELOAD_FLOAT, EXTRA_XC)
MATVEC_GATE_UP(gu_f32, grp_f32, PRELOAD_FLOAT, EXTRA_XC)
MATVEC_GATE_UP(gu_q4_k, grp_q4_k, PRELOAD_KQUANT, NOTHING)
MATVEC_GATE_UP(gu_q6_k, grp_q6_k, PRELOAD_KQUANT, NOTHING)

// Quantises activation rows the way the host does before a matmul. One workgroup of 256 per
// (row, block of 256 values); `row_stride` is the rows' length, `rows` how many. A block that
// the row ends inside gets only the q8_0 half.
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void quant_rows(
    __global const float* x, __global char* qk, __global char* qz, __global float* dk,
    __global int* sk, __global float* dz, uint cols, uint x_off) {
    __local float red[256];
    __local float ext[256];
    __local int isum[256];
    __local ushort dh[8];
    uint nblk = (cols + 255) / 256;
    uint row = get_group_id(0) / nblk, blk = get_group_id(0) % nblk;
    uint t = get_local_id(0), grp = t >> 5, lane32 = t & 31;
    uint e = blk * 256 + t;
    bool live = e < cols;
    float v = live ? x[x_off + (ulong)row * cols + e] : 0.0f;
    uint nb = cols / 32;

    // q8_0: the scale of each 32 is its largest magnitude over 127, rounded to half.
    red[t] = fabs(v);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = 16; s > 0; s >>= 1) {
        if (lane32 < s) red[t] = fmax(red[t], red[t + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float d32 = red[grp * 32] / 127.0f;
    if (lane32 == 0) vstore_half(d32, 0, (__local half*)&dh[grp]);
    barrier(CLK_LOCAL_MEM_FENCE);
    float inv = d32 != 0.0f ? 1.0f / d32 : 0.0f;
    if (live) qz[(ulong)row * cols + e] = convert_char_sat(round(v * inv));
    if (lane32 == 0 && blk * 8 + grp < nb) dz[(ulong)row * nb + blk * 8 + grp] = vload_half(0, (__local half*)&dh[grp]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // q8_K: the block's extreme, with its sign, maps to -127.
    if ((blk + 1) * 256 <= cols) {
        ext[t] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = 128; s > 0; s >>= 1) {
            if (t < s && fabs(ext[t + s]) > fabs(ext[t])) ext[t] = ext[t + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float extreme = ext[0];
        float invk = extreme != 0.0f ? -127.0f / extreme : 0.0f;
        int q = extreme != 0.0f ? min((int)rint(invk * v), 127) : 0;
        qk[(ulong)row * cols + e] = (char)q;
        if (t == 0) dk[(ulong)row * (cols / 256) + blk] = extreme != 0.0f ? 1.0f / invk : 0.0f;
        isum[t] = q;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = 16; s > 0; s >>= 1) {
            if (lane32 < s) isum[t] += isum[t + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane32 == 0) sk[(ulong)row * nb + blk * 8 + grp] = isum[t];
    }
}

// out[row] = x[row] * rsqrt(mean(x^2) + eps) * w, one workgroup of 128 per row. `x_off` is
// the first value of the first row, so the head can norm the last row alone.
__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void rms_norm_rows(
    __global const float* x, __global const float* w, __global float* out, uint n, float eps,
    uint x_off) {
    __local float red[128];
    uint row = get_group_id(0), tid = get_local_id(0);
    __global const float* xr = x + x_off + (ulong)row * n;
    float s = 0;
    for (uint i = tid; i < n; i += 128) s += xr[i] * xr[i];
    red[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 64; k > 0; k >>= 1) {
        if (tid < k) red[tid] += red[tid + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(red[0] / (float)n + eps);
    for (uint i = tid; i < n; i += 128) out[(ulong)row * n + i] = xr[i] * scale * w[i];
}

// Per-head RMS norm of q and k (when the layer has them), RoPE on both, and the keys and
// values written into the caches at their positions. One workgroup of 128 per (row, slot):
// slots 0..n_head are query heads, then the key heads, then the value heads.
__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void qkv_post(
    __global float* q, __global const float* k, __global const float* v,
    __global float* cache_k, __global float* cache_v, __global const float* cosv,
    __global const float* sinv, __global const float* q_norm, __global const float* k_norm,
    uint n_head, uint n_kv, uint head_dim, uint rope_dim, uint half_tab, uint adjacent,
    uint use_q_norm, uint use_k_norm, float eps, uint index_pos) {
    __local float h[256];
    __local float red[128];
    uint row = get_group_id(0) / (n_head + 2 * n_kv);
    uint slot = get_group_id(0) % (n_head + 2 * n_kv);
    uint tid = get_local_id(0);
    uint pos = index_pos + row;
    if (slot >= n_head + n_kv) {
        uint kh = slot - n_head - n_kv;
        for (uint i = tid; i < head_dim; i += 128)
            cache_v[((ulong)pos * n_kv + kh) * head_dim + i] =
                v[((ulong)row * n_kv + kh) * head_dim + i];
        return;
    }
    bool is_q = slot < n_head;
    uint hh = is_q ? slot : slot - n_head;
    __global const float* src = is_q ? q + ((ulong)row * n_head + hh) * head_dim
                                     : k + ((ulong)row * n_kv + hh) * head_dim;
    __global const float* nw = is_q ? q_norm : k_norm;
    bool norm = is_q ? use_q_norm != 0 : use_k_norm != 0;
    float s = 0;
    for (uint i = tid; i < head_dim; i += 128) {
        float t = src[i];
        h[i] = t;
        s += t * t;
    }
    red[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint kk = 64; kk > 0; kk >>= 1) {
        if (tid < kk) red[tid] += red[tid + kk];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = norm ? rsqrt(red[0] / (float)head_dim + eps) : 1.0f;
    for (uint i = tid; i < head_dim; i += 128) h[i] = h[i] * scale * (norm ? nw[i] : 1.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    __global float* dst = is_q ? q + ((ulong)row * n_head + hh) * head_dim
                               : cache_k + ((ulong)pos * n_kv + hh) * head_dim;
    uint pairs = rope_dim / 2;
    for (uint i = tid; i < head_dim; i += 128) {
        float o = h[i];
        if (i < rope_dim) {
            uint p, other;
            bool first;
            if (adjacent) { p = i / 2; first = (i % 2) == 0; other = first ? i + 1 : i - 1; }
            else { p = i % pairs; first = i < pairs; other = first ? i + pairs : i - pairs; }
            float c = cosv[(ulong)pos * half_tab + p], sn = sinv[(ulong)pos * half_tab + p];
            o = first ? h[i] * c - h[other] * sn : h[other] * sn + h[i] * c;
        }
        dst[i] = o;
    }
}

// Causal attention of every (row, head), 128 keys at a time with a running softmax. The row at
// `index_pos + r` sees positions 0..=index_pos + r.
__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void attention_rows(
    __global const float* q, __global const float* cache_k, __global const float* cache_v,
    __global float* out, uint n_head, uint n_kv, uint head_dim, float scale, uint index_pos) {
    __local float ql[256];
    __local float p[128];
    __local float red[128];
    uint row = get_group_id(0) / n_head, head = get_group_id(0) % n_head;
    uint tid = get_local_id(0);
    uint kvh = head / (n_head / n_kv);
    uint len = index_pos + row + 1;
    for (uint i = tid; i < head_dim; i += 128) ql[i] = q[((ulong)row * n_head + head) * head_dim + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc0 = 0, acc1 = 0;
    float m = -INFINITY, l = 0;
    for (uint t0 = 0; t0 < len; t0 += 128) {
        uint kp = t0 + tid;
        float sc = -INFINITY;
        if (kp < len) {
            __global const float* kr = cache_k + ((ulong)kp * n_kv + kvh) * head_dim;
            float d = 0;
            for (uint i = 0; i < head_dim; i += 4) d += dot(vload4(0, kr + i), vload4(0, ql + i));
            sc = d * scale;
        }
        red[tid] = sc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint kk = 64; kk > 0; kk >>= 1) {
            if (tid < kk) red[tid] = fmax(red[tid], red[tid + kk]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float mt = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        float mn = fmax(m, mt);
        float pr = kp < len ? exp(sc - mn) : 0.0f;
        p[tid] = pr;
        red[tid] = pr;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint kk = 64; kk > 0; kk >>= 1) {
            if (tid < kk) red[tid] += red[tid + kk];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float corr = exp(m - mn);
        l = l * corr + red[0];
        m = mn;
        acc0 *= corr;
        acc1 *= corr;
        uint cnt = min(128u, len - t0);
        for (uint j = 0; j < cnt; j++) {
            __global const float* vr = cache_v + ((ulong)(t0 + j) * n_kv + kvh) * head_dim;
            float pj = p[j];
            if (tid < head_dim) acc0 += pj * vr[tid];
            if (tid + 128 < head_dim) acc1 += pj * vr[tid + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    __global float* o = out + ((ulong)row * n_head + head) * head_dim;
    if (tid < head_dim) o[tid] = acc0 / l;
    if (tid + 128 < head_dim) o[tid + 128] = acc1 / l;
}

// out[i] = silu(gate[i]) * up[i].
__kernel void silu_mul(__global const float* gate, __global const float* up,
                       __global float* out, uint n) {
    uint i = get_global_id(0);
    if (i < n) {
        float g = gate[i];
        out[i] = g / (1.0f + exp(-g)) * up[i];
    }
}
#endif  // MEGA
