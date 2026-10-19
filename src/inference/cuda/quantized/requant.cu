// Requantisation quantize kernels: f32 -> GGUF blocks, on device, exact GGUF byte layout.
//
// One warp (32 threads) per quantised block, grid = block count. The output is the same bytes a
// CPU quantise would write, to f32 rounding: the gate is quality-equivalent and reproducible, not
// bit-identical to the CPU (GPU f32 differs by FMA contraction and reduction order). block_* and
// the QK_* sizes come from gguf.cuh, already in the NVRTC translation unit.

// q8_0: one scale per 32 values, d = amax / 127, qs[i] = round(x[i] / d). No offset, no packing.
extern "C" __global__ void requant_q8_0(
        const float * __restrict__ x, void * __restrict__ vy, const int nblocks) {
    const int blk = blockIdx.x;
    if (blk >= nblocks) return;
    const int lane = threadIdx.x; // 0..31, one value each
    const float xi = x[(size_t) blk * QK8_0 + lane];

    // Block amax across the 32 lanes.
    float amax = fabsf(xi);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    // roundf is round-half-away-from-zero, matching the CPU quantiser's nearest_int.
    int q = (int) roundf(xi * id);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);

    block_q8_0 * y = (block_q8_0 *) vy + blk;
    y->qs[lane] = (int8_t) q;
    if (lane == 0) {
        y->d = __float2half(d);
    }
}

// q4_K: eight 32-value sub-blocks per 256 super-block. Each sub-block is fitted to a (scale, min)
// by the same weighted grid search as the CPU encoder (make_qkx over GRID_STEPS candidates); those
// sixteen numbers are themselves quantised to six bits, scaled by d and dmin. Then every value is
// levelled against the scale as it was ROUNDED into the block, and packed two sub-blocks per span.
// One CUDA block (32 threads) per super-block. Quality-equivalent to the CPU encoder, reproducible.

#define REQ_QK_K 256
#define REQ_SUB 32
#define REQ_GROUPS 8
#define REQ_NMAX 15
#define REQ_LEVELS 63

// nearest_int: round half away from zero, matching the CPU quantiser.
__device__ __forceinline__ int req_nint(float v) { return (int) roundf(v); }
__device__ __forceinline__ int req_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Fit an n-value sub-block (n = 16 or 32) to a (scale, min>=0) with unsigned levels 0..nmax.
// Mirrors fit_scale_and_offset: the make_qkx grid search over GRID_STEPS candidate scales.
__device__ void req_fit_scale_offset(int nmax, int n, const float * x, const float * w, float * out_scale, float * out_min) {
    float offset = x[0], top = x[0], w_total = w[0], wx_total = w[0] * x[0];
    for (int i = 1; i < n; i++) {
        offset = fminf(offset, x[i]);
        top = fmaxf(top, x[i]);
        w_total += w[i];
        wx_total += w[i] * x[i];
    }
    if (offset > 0.0f) offset = 0.0f;
    if (top <= offset) { *out_scale = 0.0f; *out_min = -offset; return; }

    float per_unit = (float) nmax / (top - offset);
    float scale = 1.0f / per_unit;
    float best_error = 0.0f;
    for (int i = 0; i < n; i++) {
        int lv = req_clampi(req_nint(per_unit * (x[i] - offset)), 0, nmax);
        float diff = scale * (float) lv + offset - x[i];
        best_error += w[i] * diff * diff;
    }

    unsigned char levels[REQ_SUB]; // REQ_SUB (32) is the max n
    for (int step = 0; step <= 36; step++) {
        per_unit = (-0.9f + 0.05f * (float) step + (float) nmax) / (top - offset);
        float wl = 0.0f, wl2 = 0.0f, wxl = 0.0f;
        for (int i = 0; i < n; i++) {
            int lv = req_clampi(req_nint(per_unit * (x[i] - offset)), 0, nmax);
            levels[i] = (unsigned char) lv;
            wl += w[i] * (float) lv;
            wl2 += w[i] * (float) lv * (float) lv;
            wxl += w[i] * (float) lv * x[i];
        }
        float denom = w_total * wl2 - wl * wl;
        if (denom > 0.0f) {
            float cs = (w_total * wxl - wx_total * wl) / denom;
            float co = (wl2 * wx_total - wl * wxl) / denom;
            if (co > 0.0f) { co = 0.0f; cs = wxl / wl2; }
            float error = 0.0f;
            for (int i = 0; i < n; i++) {
                float diff = cs * (float) levels[i] + co - x[i];
                error += w[i] * diff * diff;
            }
            if (error < best_error) { best_error = error; scale = cs; offset = co; }
        }
    }
    *out_scale = scale;
    *out_min = -offset;
}

extern "C" __global__ void requant_q4_k(
        const float * __restrict__ x, void * __restrict__ vy, const int nblocks) {
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x; // 0..31
    const float * xb = x + (size_t) sb * REQ_QK_K;
    block_q4_K * y = (block_q4_K *) vy + sb;

    __shared__ float sh_scale[REQ_GROUPS];
    __shared__ float sh_min[REQ_GROUPS];
    __shared__ float sh_dg[REQ_GROUPS];
    __shared__ float sh_mg[REQ_GROUPS];
    __shared__ unsigned char sh_lvl[REQ_QK_K];

    // Phase A: one thread per sub-block fits its (scale, min).
    if (t < REQ_GROUPS) {
        const float * xg = xb + t * REQ_SUB;
        float ss = 0.0f;
        for (int i = 0; i < REQ_SUB; i++) ss += xg[i] * xg[i];
        float rms = sqrtf(ss / (float) REQ_SUB);
        float w[REQ_SUB];
        for (int i = 0; i < REQ_SUB; i++) w[i] = rms + fabsf(xg[i]);
        float s, m;
        req_fit_scale_offset(REQ_NMAX, REQ_SUB, xg, w, &s, &m);
        sh_scale[t] = s;
        sh_min[t] = m;
    }
    __syncthreads();

    // Phase B (one thread): quantise the sixteen sub-block scales/mins to six bits, pack them, and
    // re-derive the per-sub-block d and m as the decoder will read them.
    if (t == 0) {
        float max_scale = 0.0f, max_min = 0.0f;
        for (int g = 0; g < REQ_GROUPS; g++) {
            max_scale = fmaxf(max_scale, sh_scale[g]);
            max_min = fmaxf(max_min, sh_min[g]);
        }
        unsigned char ls[REQ_GROUPS], lm[REQ_GROUPS];
        float inv_s = max_scale > 0.0f ? (float) REQ_LEVELS / max_scale : 0.0f;
        float inv_m = max_min > 0.0f ? (float) REQ_LEVELS / max_min : 0.0f;
        for (int g = 0; g < REQ_GROUPS; g++) {
            ls[g] = (unsigned char) min(req_nint(inv_s * sh_scale[g]), REQ_LEVELS);
            lm[g] = (unsigned char) min(req_nint(inv_m * sh_min[g]), REQ_LEVELS);
        }
        float d = max_scale / (float) REQ_LEVELS;
        float dmin = max_min / (float) REQ_LEVELS;
        // Store d/dmin as the decoder reads them (f16), and re-derive with the rounded value.
        half hd = __float2half(d);
        half hdm = __float2half(dmin);
        y->dm = __halves2half2(hd, hdm);
        float d_used = __half2float(hd);
        float dmin_used = __half2float(hdm);

        // Pack eight six-bit scales and eight six-bit mins into twelve bytes (q4_K/q5_K layout).
        unsigned char sc[12];
        for (int i = 0; i < 4; i++) { sc[i] = ls[i] & 0x3F; sc[i + 4] = lm[i] & 0x3F; }
        for (int i = 4; i < 8; i++) {
            sc[i + 4] = (ls[i] & 0x0F) | ((lm[i] & 0x0F) << 4);
            sc[i - 4] |= (unsigned char) ((ls[i] >> 4) << 6);
            sc[i] |= (unsigned char) ((lm[i] >> 4) << 6);
        }
        for (int i = 0; i < 12; i++) y->scales[i] = sc[i];

        // Unpack back to the rounded six-bit values, and form each sub-block's d and m.
        unsigned char us[REQ_GROUPS], um[REQ_GROUPS];
        for (int i = 0; i < 4; i++) { us[i] = sc[i] & 0x3F; um[i] = sc[i + 4] & 0x3F; }
        for (int i = 4; i < 8; i++) {
            unsigned char low = sc[i + 4];
            us[i] = (low & 0x0F) | ((sc[i - 4] >> 6) << 4);
            um[i] = (low >> 4) | ((sc[i] >> 6) << 4);
        }
        for (int g = 0; g < REQ_GROUPS; g++) {
            sh_dg[g] = d_used * (float) us[g];
            sh_mg[g] = dmin_used * (float) um[g];
        }
    }
    __syncthreads();

    // Phase C: level every value against its sub-block's rounded scale, then pack.
    for (int idx = t; idx < REQ_QK_K; idx += 32) {
        int g = idx / REQ_SUB;
        float dg = sh_dg[g];
        unsigned char lvl = 0;
        if (dg != 0.0f) {
            lvl = (unsigned char) req_clampi(req_nint((xb[idx] + sh_mg[g]) / dg), 0, 15);
        }
        sh_lvl[idx] = lvl;
    }
    __syncthreads();

    // Two sub-blocks share a 32-byte span: even in the low nibble, odd in the high.
    for (int b = t; b < REQ_QK_K / 2; b += 32) {
        int p = b / REQ_SUB;   // pair 0..3
        int j = b % REQ_SUB;   // 0..31
        unsigned char lo = sh_lvl[(2 * p) * REQ_SUB + j];
        unsigned char hi = sh_lvl[(2 * p + 1) * REQ_SUB + j];
        y->qs[b] = (unsigned char) (lo | (hi << 4));
    }
}

// q3_K: sixteen 16-value sub-blocks per super-block, three-bit signed codes centred on zero (no
// minimum). Each sub-block is fitted by the signed grid search + refinement of the CPU encoder
// (fit_signed_scale); the sixteen scales are themselves quantised to six signed bits (bias 32).
// Codes are biased into 0..7: the top bit goes to hmask, the low two are packed four per byte.

// fit_signed_scale for a 16-value sub-block. `ls` receives level+nmax (>=0); returns the scale.
__device__ void req_fit_signed(int nmax, const float * x, unsigned char * ls, const float * w, float * out_scale) {
    float extreme = 0.0f;
    for (int i = 0; i < 16; i++) if (fabsf(x[i]) > fabsf(extreme)) extreme = x[i];
    if (extreme == 0.0f) {
        for (int i = 0; i < 16; i++) ls[i] = (unsigned char) nmax;
        *out_scale = 0.0f;
        return;
    }
    float wxl = 0.0f, wl2 = 0.0f;
    float inv = -(float) nmax / extreme;
    for (int i = 0; i < 16; i++) {
        int lv = req_clampi(req_nint(inv * x[i]), -nmax, nmax - 1);
        ls[i] = (unsigned char) (lv + nmax);
        wxl += w[i] * x[i] * (float) lv;
        wl2 += w[i] * (float) lv * (float) lv;
    }
    float best = wl2 > 0.0f ? wxl * wxl / wl2 : 0.0f;

    for (int r = 0; r < 3; r++) {
        float sinv = wl2 / wxl; // 1 / (wxl/wl2)
        bool unchanged = true;
        float cwxl = 0.0f, cwl2 = 0.0f;
        for (int i = 0; i < 16; i++) {
            int lv = req_clampi(req_nint(sinv * x[i]), -nmax, nmax - 1);
            if (lv + nmax != (int) ls[i]) unchanged = false;
            cwxl += w[i] * x[i] * (float) lv;
            cwl2 += w[i] * (float) lv * (float) lv;
        }
        if (unchanged || cwl2 == 0.0f || cwxl * cwxl <= best * cwl2) break;
        for (int i = 0; i < 16; i++) {
            int lv = req_clampi(req_nint(sinv * x[i]), -nmax, nmax - 1);
            ls[i] = (unsigned char) (lv + nmax);
        }
        wxl = cwxl; wl2 = cwl2; best = wxl * wxl / wl2;
    }

    for (int pass = 0; pass < 5; pass++) {
        int moved = 0;
        for (int i = 0; i < 16; i++) {
            float v = x[i], ww = w[i];
            int l = (int) ls[i] - nmax;
            float nwxl = wxl - ww * v * (float) l;
            if (nwxl <= 0.0f) continue;
            float nwl2 = wl2 - ww * (float) (l * l);
            int to = req_clampi(req_nint(v * nwl2 / nwxl), -nmax, nmax - 1);
            if (to == l) continue;
            nwxl += ww * v * (float) to;
            nwl2 += ww * (float) (to * to);
            if (nwl2 > 0.0f && nwxl * nwxl * wl2 > wxl * wxl * nwl2) {
                ls[i] = (unsigned char) (to + nmax);
                wxl = nwxl; wl2 = nwl2; best = wxl * wxl / wl2;
                moved++;
            }
        }
        if (moved == 0) break;
    }

    for (int s = -4; s <= 4; s++) {
        if (s == 0) continue;
        float sinv = -((float) nmax + 0.1f * (float) s) / extreme;
        float cwxl = 0.0f, cwl2 = 0.0f;
        for (int i = 0; i < 16; i++) {
            int lv = req_clampi(req_nint(sinv * x[i]), -nmax, nmax - 1);
            cwxl += w[i] * x[i] * (float) lv;
            cwl2 += w[i] * (float) lv * (float) lv;
        }
        if (cwl2 > 0.0f && cwxl * cwxl > best * cwl2) {
            for (int i = 0; i < 16; i++) {
                int lv = req_clampi(req_nint(sinv * x[i]), -nmax, nmax - 1);
                ls[i] = (unsigned char) (lv + nmax);
            }
            wxl = cwxl; wl2 = cwl2; best = wxl * wxl / wl2;
        }
    }
    *out_scale = wxl / wl2;
}

extern "C" __global__ void requant_q3_k(
        const float * __restrict__ x, void * __restrict__ vy, const int nblocks) {
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x; // 0..31
    const float * xb = x + (size_t) sb * REQ_QK_K;
    block_q3_K * y = (block_q3_K *) vy + sb;
    const int CODE_MAX = 4, SCALE_BIAS = 32, GROUPS = 16, GSUB = 16;

    __shared__ float sh_scale[16];
    __shared__ float sh_dg[16];
    __shared__ unsigned char sh_code[REQ_QK_K]; // biased 0..7
    __shared__ unsigned char sh_low[REQ_QK_K];  // low two bits (code with high bit removed)

    // Phase A: one thread per 16-value sub-block fits its signed scale.
    if (t < GROUPS) {
        const float * xg = xb + t * GSUB;
        float ss = 0.0f;
        for (int i = 0; i < GSUB; i++) ss += xg[i] * xg[i];
        float rms = sqrtf(ss / (float) GSUB);
        float w[16], xl[16];
        unsigned char ls[16];
        for (int i = 0; i < GSUB; i++) { xl[i] = xg[i]; w[i] = rms + fabsf(xg[i]); }
        float s;
        req_fit_signed(CODE_MAX, xl, ls, w, &s);
        sh_scale[t] = s;
    }
    __syncthreads();

    // Phase B (one thread): quantise the sixteen signed scales to six bits (bias 32), pack them,
    // and re-derive each sub-block's signed d as the decoder reads it.
    if (t == 0) {
        float extreme = 0.0f;
        for (int g = 0; g < GROUPS; g++) if (fabsf(sh_scale[g]) > fabsf(extreme)) extreme = sh_scale[g];
        int levels[16];
        float d;
        if (extreme == 0.0f) {
            for (int g = 0; g < GROUPS; g++) levels[g] = 0;
            d = 0.0f;
        } else {
            float inv = -(float) SCALE_BIAS / extreme;
            for (int g = 0; g < GROUPS; g++)
                levels[g] = req_clampi(req_nint(inv * sh_scale[g]), -SCALE_BIAS, SCALE_BIAS - 1) + SCALE_BIAS;
            d = 1.0f / inv;
        }
        half hd = __float2half(d);
        y->d = hd;
        float d_used = __half2float(hd);

        // Pack sixteen six-bit levels into twelve bytes (q3_K layout).
        unsigned char sc[12];
        for (int i = 0; i < 12; i++) sc[i] = 0;
        for (int i = 0; i < 16; i++) {
            int level = levels[i];
            if (i < 8) sc[i] |= (unsigned char) (level & 0x0F);
            else sc[i - 8] |= (unsigned char) ((level & 0x0F) << 4);
            sc[8 + i % 4] |= (unsigned char) ((level >> 4) << (2 * (i / 4)));
        }
        for (int i = 0; i < 12; i++) y->scales[i] = sc[i];

        // Unpack back to the signed six-bit scales (bias 32) and form each sub-block's d.
        for (int g = 0; g < GROUPS; g++) {
            int grp = g / 4, j = g % 4;
            unsigned char byte = sc[j + 4 * (grp % 2)];
            int low = grp < 2 ? (byte & 0x0F) : (byte >> 4);
            int high = (sc[8 + j] >> (2 * grp)) & 0x03;
            int scale = (low | (high << 4)) - 32;
            sh_dg[g] = d_used * (float) scale;
        }
    }
    __syncthreads();

    // Phase C: code every value against its sub-block's signed scale (biased into 0..7).
    for (int idx = t; idx < REQ_QK_K; idx += 32) {
        int g = idx / GSUB;
        float dg = sh_dg[g];
        int code = 0;
        if (dg != 0.0f) {
            code = req_clampi(req_nint(xb[idx] / dg), -CODE_MAX, CODE_MAX - 1) + CODE_MAX;
        }
        sh_code[idx] = (unsigned char) code;
        sh_low[idx] = (unsigned char) (code >= CODE_MAX ? code - CODE_MAX : code);
    }
    __syncthreads();

    // hmask: value v owns bit v/32 of byte v%32, set when its biased code reached CODE_MAX.
    for (int m = t; m < REQ_QK_K / 8; m += 32) {
        unsigned char h = 0;
        for (int p = 0; p < 8; p++) {
            if (sh_code[p * 32 + m] >= CODE_MAX) h |= (unsigned char) (1u << p);
        }
        y->hmask[m] = h;
    }
    // qs: the low two bits, four values (l, l+32, l+64, l+96) to a byte, per 128-half.
    for (int b = t; b < REQ_QK_K / 4; b += 32) {
        int half = b / 32, l = b % 32;
        const unsigned char * q = &sh_low[half * 128];
        y->qs[b] = (unsigned char) (q[l] | (q[l + 32] << 2) | (q[l + 64] << 4) | (q[l + 96] << 6));
    }
}

// q2_K: sixteen 16-value sub-blocks, two-bit unsigned codes with a per-sub-block scale and min,
// each four bits, sharing one byte. The fit is q4_K's (unsigned levels + offset) at nmax 3 and a
// 4-bit scale quant. One super-block per CUDA block. Quality-equivalent, reproducible.
extern "C" __global__ void requant_q2_k(
        const float * __restrict__ x, void * __restrict__ vy, const int nblocks) {
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x; // 0..31
    const float * xb = x + (size_t) sb * REQ_QK_K;
    block_q2_K * y = (block_q2_K *) vy + sb;
    const int GROUPS = 16, GSUB = 16, Q2_NMAX = 3, Q2_LEVELS = 15;

    __shared__ float sh_scale[16];
    __shared__ float sh_min[16];
    __shared__ float sh_sg[16]; // per-group scale (d * 4-bit level)
    __shared__ float sh_og[16]; // per-group offset (dmin * 4-bit level)
    __shared__ unsigned char sh_code[REQ_QK_K];

    // Phase A: one thread per 16-value sub-block fits its (scale, min).
    if (t < GROUPS) {
        const float * xg = xb + t * GSUB;
        float ss = 0.0f;
        for (int i = 0; i < GSUB; i++) ss += xg[i] * xg[i];
        float rms = sqrtf(ss / (float) GSUB);
        float w[16], xl[16];
        for (int i = 0; i < GSUB; i++) { xl[i] = xg[i]; w[i] = rms + fabsf(xg[i]); }
        float s, m;
        req_fit_scale_offset(Q2_NMAX, GSUB, xl, w, &s, &m);
        sh_scale[t] = s;
        sh_min[t] = m;
    }
    __syncthreads();

    // Phase B: quantise the sixteen scales/mins to four bits, pack, re-derive per-group d and m.
    if (t == 0) {
        float max_scale = 0.0f, max_min = 0.0f;
        for (int g = 0; g < GROUPS; g++) {
            max_scale = fmaxf(max_scale, sh_scale[g]);
            max_min = fmaxf(max_min, sh_min[g]);
        }
        float inv_s = max_scale > 0.0f ? (float) Q2_LEVELS / max_scale : 0.0f;
        float inv_m = max_min > 0.0f ? (float) Q2_LEVELS / max_min : 0.0f;
        float d = max_scale / (float) Q2_LEVELS;
        float dmin = max_min / (float) Q2_LEVELS;
        half hd = __float2half(d);
        half hdm = __float2half(dmin);
        y->dm = __halves2half2(hd, hdm);
        float d_used = __half2float(hd);
        float dmin_used = __half2float(hdm);
        for (int g = 0; g < GROUPS; g++) {
            int ls = min(req_nint(inv_s * sh_scale[g]), Q2_LEVELS);
            int lm = min(req_nint(inv_m * sh_min[g]), Q2_LEVELS);
            y->scales[g] = (unsigned char) (ls | (lm << 4));
            sh_sg[g] = d_used * (float) ls;
            sh_og[g] = dmin_used * (float) lm;
        }
    }
    __syncthreads();

    // Phase C: two-bit code per value against its sub-block's rounded scale and min.
    for (int idx = t; idx < REQ_QK_K; idx += 32) {
        int g = idx / GSUB;
        float sc = sh_sg[g];
        unsigned char code = 0;
        if (sc != 0.0f) {
            code = (unsigned char) req_clampi(req_nint((xb[idx] + sh_og[g]) / sc), 0, 3);
        }
        sh_code[idx] = code;
    }
    __syncthreads();

    // Pack qs: four sub-blocks share a byte at shifts 0,2,4,6 (the q2_K placement).
    for (int bi = t; bi < REQ_QK_K / 4; bi += 32) {
        int half = bi / 32;
        int off = bi % 32;
        int par = off >= 16 ? 1 : 0;
        int i = off - 16 * par;
        unsigned char b = 0;
        for (int k = 0; k < 4; k++) {
            int s = half * 8 + par + 2 * k;
            b |= (unsigned char) (sh_code[s * 16 + i] << (2 * k));
        }
        y->qs[bi] = b;
    }
}

// Fit non-negative x[0..n) to levels 0..nmax with no offset against weights w, writing the levels
// into l and returning the scale. Mirrors fit_unsigned_scale: nine candidate scales around the
// largest value, then one-at-a-time refinement of the levels.
__device__ float req_fit_unsigned_scale(int nmax, int n, const float * x, unsigned char * l, const float * w) {
    float top = 0.0f;
    for (int i = 0; i < n; i++) top = fmaxf(top, x[i]);
    if (top == 0.0f) {
        for (int i = 0; i < n; i++) l[i] = 0;
        return 0.0f;
    }
    float inv = (float) nmax / top;
    float best = 0.0f;
    for (int i = 0; i < n; i++) {
        int c = min(req_nint(inv * x[i]), nmax);
        if (c < 0) c = 0;
        float diff = x[i] - (1.0f / inv) * (float) c;
        best += w[i] * diff * diff;
    }
    for (int step = -4; step <= 4; step++) {
        if (step == 0) continue;
        float candidate = (0.1f * (float) step + (float) nmax) / top;
        float e = 0.0f;
        for (int i = 0; i < n; i++) {
            int c = min(req_nint(candidate * x[i]), nmax);
            if (c < 0) c = 0;
            float diff = x[i] - (1.0f / candidate) * (float) c;
            e += w[i] * diff * diff;
        }
        if (e < best) { best = e; inv = candidate; }
    }
    float fwxl = 0.0f, fwl2 = 0.0f;
    for (int i = 0; i < n; i++) {
        int c = min(req_nint(inv * x[i]), nmax);
        if (c < 0) c = 0;
        l[i] = (unsigned char) c;
        fwxl += w[i] * x[i] * (float) c;
        fwl2 += w[i] * (float) c * (float) c;
    }
    for (int iter = 0; iter < 5; iter++) {
        int moved = 0;
        for (int i = 0; i < n; i++) {
            float c = (float) l[i];
            float wxl = fwxl - w[i] * x[i] * c;
            float wl2 = fwl2 - w[i] * c * c;
            if (wxl <= 0.0f || wl2 <= 0.0f) continue;
            int to = min(req_nint(x[i] * wl2 / wxl), nmax);
            if (to < 0) to = 0;
            if (to == (int) l[i]) continue;
            wxl += w[i] * x[i] * (float) to;
            wl2 += w[i] * (float) to * (float) to;
            if (wxl * wxl * fwl2 > fwxl * fwxl * wl2) {
                l[i] = (unsigned char) to;
                fwxl = wxl;
                fwl2 = wl2;
                moved++;
            }
        }
        if (moved == 0) break;
    }
    return fwxl / fwl2;
}

// q2_K weighted by an importance: `imp` holds one weight per column of a row `blocks_per_row` blocks
// wide. A value weighs its column's importance times sqrt(sigma2 + x^2), sigma2 the block's mean
// square; each sub-block's scale and minimum are then quantised against the weight its values
// carry. Mirrors the CPU encoder's guided path. Values are laid out as `requant_iq2_xxs` takes them,
// and with `has_got` their decoded values are written to `got`.
extern "C" __global__ void requant_q2_k_guided(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy,
        float * __restrict__ got, const int has_got,
        const int nblocks, const int blocks_per_row) {
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x; // 0..31
    float xb[REQ_QK_K];
    for (int i = 0; i < REQ_QK_K; i++) xb[i] = x[(size_t) sb * block_step + (size_t) i * value_step];
    const float * ib = imp + (size_t) (sb % blocks_per_row) * REQ_QK_K;
    block_q2_K * y = (block_q2_K *) vy + sb;
    const int GROUPS = 16, GSUB = 16, Q2_NMAX = 3, Q2_LEVELS = 15;

    __shared__ float sh_scale[16];
    __shared__ float sh_min[16];
    __shared__ float sh_carried[16];
    __shared__ float sh_sg[16];
    __shared__ float sh_og[16];
    __shared__ unsigned char sh_code[REQ_QK_K];

    // Phase A: one thread per sub-block fits its (scale, min) under the importance weights.
    if (t < GROUPS) {
        float sigma2 = 0.0f;
        for (int i = 0; i < REQ_QK_K; i++) sigma2 += xb[i] * xb[i];
        sigma2 /= (float) REQ_QK_K;
        const float * xg = xb + t * GSUB;
        float w[16];
        float carried = 0.0f;
        for (int i = 0; i < GSUB; i++) {
            w[i] = ib[t * GSUB + i] * sqrtf(sigma2 + xg[i] * xg[i]);
            carried += w[i];
        }
        float s, m;
        req_fit_scale_offset(Q2_NMAX, GSUB, xg, w, &s, &m);
        sh_scale[t] = s;
        sh_min[t] = m;
        sh_carried[t] = carried;
    }
    __syncthreads();

    // Phase B: the sixteen scales and minimums to four bits each, weighted by what they carry.
    if (t == 0) {
        unsigned char ls[16], lm[16];
        float d = req_fit_unsigned_scale(Q2_LEVELS, GROUPS, sh_scale, ls, sh_carried);
        float dmin = req_fit_unsigned_scale(Q2_LEVELS, GROUPS, sh_min, lm, sh_carried);
        half hd = __float2half(d);
        half hdm = __float2half(dmin);
        y->dm = __halves2half2(hd, hdm);
        float d_used = __half2float(hd);
        float dmin_used = __half2float(hdm);
        for (int g = 0; g < GROUPS; g++) {
            y->scales[g] = (unsigned char) (ls[g] | (lm[g] << 4));
            sh_sg[g] = d_used * (float) ls[g];
            sh_og[g] = dmin_used * (float) lm[g];
        }
    }
    __syncthreads();

    // Phase C: two-bit code per value against its sub-block's rounded scale and min.
    for (int idx = t; idx < REQ_QK_K; idx += 32) {
        int g = idx / GSUB;
        float sc = sh_sg[g];
        unsigned char code = 0;
        if (sc != 0.0f) {
            code = (unsigned char) req_clampi(req_nint((xb[idx] + sh_og[g]) / sc), 0, 3);
        }
        sh_code[idx] = code;
        if (has_got) got[(size_t) sb * block_step + (size_t) idx * value_step] = sc * (float) code - sh_og[g];
    }
    __syncthreads();

    // Pack qs: four sub-blocks share a byte at shifts 0,2,4,6 (the q2_K placement).
    for (int bi = t; bi < REQ_QK_K / 4; bi += 32) {
        int half = bi / 32;
        int off = bi % 32;
        int par = off >= 16 ? 1 : 0;
        int i = off - 16 * par;
        unsigned char b = 0;
        for (int k = 0; k < 4; k++) {
            int s = half * 8 + par + 2 * k;
            b |= (unsigned char) (sh_code[s * 16 + i] << (2 * k));
        }
        y->qs[bi] = b;
    }
}

// iq2_xxs: 256 values in 32 groups of 8, each group one point of a 256-point grid with its signs;
// eight sub-blocks of four groups share a 4-bit scale under the block's f16 scale. One sub-block per
// thread searches its scale over candidates around its largest magnitude, each scored by the least
// error any grid point leaves its groups; thread 0 fixes the block scale and the 4-bit levels; then
// each sub-block picks its points at the scale it was rounded to. The sign of one value per group is
// free - the format fixes the eighth by parity - so an odd group flips its cheapest value. `imp`
// holds one weight per column of a row, `blocks_per_row` blocks of it. With `has_got` the values the
// blocks decode to are written to `got`, in the layout `x` has. Quality-equivalent to the CPU encoder,
// reproducible.
typedef struct {
    half d;
    unsigned short qs[32];
} req_block_iq2_xxs;

__device__ static void req_iq2_group(
        const float * xg, const float * wg, float * v, float * w, unsigned char * signs) {
    int neg = 0, cheapest = 0;
    float least = 3.0e38f;
    unsigned char s = 0;
    for (int k = 0; k < 8; k++) {
        v[k] = fabsf(xg[k]);
        w[k] = wg ? wg[k] : 1.0f;
        if (xg[k] < 0.0f) { neg++; s |= (unsigned char) (1 << k); }
        float cost = w[k] * v[k];
        if (cost < least) { least = cost; cheapest = k; }
    }
    if (neg % 2 == 1) s ^= (unsigned char) (1 << cheapest);
    *signs = s;
}

extern "C" __global__ void requant_iq2_xxs(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, const float * __restrict__ grid, void * __restrict__ vy,
        float * __restrict__ got, const int has_got,
        const int nblocks, const int blocks_per_row, const int has_imp) {
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x;
    // Value i of block sb sits at x[sb * block_step + i * value_step]: rows of 256 laid end to end
    // (256, 1), or one row per block down the columns of a column-major matrix (1, rows).
    float xb[REQ_QK_K];
    for (int i = 0; i < REQ_QK_K; i++) xb[i] = x[(size_t) sb * block_step + (size_t) i * value_step];
    const float * wb = has_imp ? imp + (size_t) (sb % blocks_per_row) * REQ_QK_K : 0;
    req_block_iq2_xxs * y = (req_block_iq2_xxs *) vy + sb;
    const int STEPS = 15;
    const float LOW = 0.45f, HIGH = 1.25f, GRID_MAX = 43.0f, LEVELS = 15.0f;

    __shared__ float sh_scale[8];
    __shared__ float sh_rounded[8];
    __shared__ int sh_level[8];

    // Phase A: sub-block t searches its scale.
    if (t < 8) {
        float amax = 1e-30f;
        for (int k = 0; k < 32; k++) amax = fmaxf(amax, fabsf(xb[32 * t + k]));
        float err[15];
        for (int s = 0; s < STEPS; s++) err[s] = 0.0f;
        for (int l = 0; l < 4; l++) {
            const int g = 4 * t + l;
            float v[8], w[8];
            unsigned char signs;
            req_iq2_group(xb + 8 * g, wb ? wb + 8 * g : 0, v, w, &signs);
            float vv = 0.0f;
            for (int k = 0; k < 8; k++) vv += w[k] * v[k] * v[k];
            float dot[256], gg[256];
            for (int n = 0; n < 256; n++) {
                float d = 0.0f, q = 0.0f;
                for (int k = 0; k < 8; k++) {
                    const float p = grid[8 * n + k];
                    d += w[k] * v[k] * p;
                    q += w[k] * p * p;
                }
                dot[n] = d;
                gg[n] = q;
            }
            for (int s = 0; s < STEPS; s++) {
                const float c = amax * (LOW + (HIGH - LOW) * (float) s / (float) (STEPS - 1)) / GRID_MAX;
                const float a = 2.0f * c, b = c * c;
                float best = 3.0e38f;
                for (int n = 0; n < 256; n++) {
                    const float e = b * gg[n] - a * dot[n];
                    best = e < best ? e : best;
                }
                err[s] += vv + best;
            }
        }
        float best = 3.0e38f, chosen = 0.0f;
        for (int s = 0; s < STEPS; s++) {
            if (err[s] < best) {
                best = err[s];
                chosen = amax * (LOW + (HIGH - LOW) * (float) s / (float) (STEPS - 1)) / GRID_MAX;
            }
        }
        sh_scale[t] = chosen;
    }
    __syncthreads();

    // Phase B: the block scale, and each sub-block's four-bit level under it.
    if (t == 0) {
        float max_scale = 0.0f;
        for (int i = 0; i < 8; i++) max_scale = fmaxf(max_scale, sh_scale[i]);
        half hd = __float2half(max_scale / ((0.5f + LEVELS) * 0.25f));
        y->d = hd;
        const float d = __half2float(hd);
        for (int i = 0; i < 8; i++) {
            int level = d > 0.0f ? req_clampi(req_nint(sh_scale[i] / (d * 0.25f) - 0.5f), 0, 15) : 0;
            sh_level[i] = level;
            sh_rounded[i] = d * (0.5f + (float) level) * 0.25f;
        }
    }
    __syncthreads();

    // Phase C: each sub-block picks its points at the scale it was rounded to, and packs them.
    if (t < 8) {
        const float c = sh_rounded[t];
        const float a = 2.0f * c, b = c * c;
        unsigned int lo = 0, hi = (unsigned int) sh_level[t] << 28;
        for (int l = 0; l < 4; l++) {
            const int g = 4 * t + l;
            float v[8], w[8];
            unsigned char signs;
            req_iq2_group(xb + 8 * g, wb ? wb + 8 * g : 0, v, w, &signs);
            int at = 0;
            float best = 3.0e38f;
            for (int n = 0; n < 256; n++) {
                float d = 0.0f, q = 0.0f;
                for (int k = 0; k < 8; k++) {
                    const float p = grid[8 * n + k];
                    d += w[k] * v[k] * p;
                    q += w[k] * p * p;
                }
                const float e = b * q - a * d;
                if (e < best) { best = e; at = n; }
            }
            lo |= (unsigned int) at << (8 * l);
            hi |= (unsigned int) (signs & 127) << (7 * l);
            if (has_got) {
                for (int k = 0; k < 8; k++) {
                    const float mag = c * grid[8 * at + k];
                    const size_t i = (size_t) (8 * g + k);
                    got[(size_t) sb * block_step + i * value_step] = ((signs >> k) & 1) ? -mag : mag;
                }
            }
        }
        y->qs[4 * t] = (unsigned short) (lo & 0xffff);
        y->qs[4 * t + 1] = (unsigned short) (lo >> 16);
        y->qs[4 * t + 2] = (unsigned short) (hi & 0xffff);
        y->qs[4 * t + 3] = (unsigned short) (hi >> 16);
    }
}

// Block-scaled fp4 to f32: `rows` by `inp` values, each row `inp / 2` bytes of e2m1 nibbles (element
// 2j in the low nibble, 2j + 1 in the high) and `inp / block` ue8m0 scales along the input.
extern "C" __global__ void dequant_fp4_e8m0(
        const unsigned char * __restrict__ nibbles, const unsigned char * __restrict__ scales,
        float * __restrict__ out, const long inp, const long block, const long total) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= total) return;
    static const float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const long r = idx / inp, c = idx % inp;
    const unsigned char byte = nibbles[r * (inp / 2) + c / 2];
    const unsigned char nib = (c % 2) ? (byte >> 4) : (byte & 0x0F);
    const unsigned char e = scales[r * (inp / block) + c / block];
    const float scale = e == 0xFF ? NAN : ldexpf(1.0f, (int) e - 127);
    const float mag = E2M1[nib & 7] * scale;
    out[idx] = (nib & 8) ? -mag : mag;
}

// Block-scaled fp8 to f32: rows `row0 .. row0 + rows` of a weight `inp` wide, `bytes` holding just those
// rows of e4m3fn values and `scales` the whole weight's ue8m0 scales, one per `block` x `block` tile,
// `tiles_per_row` to a row of tiles.
extern "C" __global__ void dequant_fp8_tiles(
        const unsigned char * __restrict__ bytes, const unsigned char * __restrict__ scales,
        float * __restrict__ out, const long inp, const long block, const long row0,
        const long tiles_per_row, const long total) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= total) return;
    const long o = idx / inp, c = idx % inp;
    const unsigned char b = bytes[idx];
    const int exp = (b >> 3) & 0x0F, mant = b & 0x07;
    float v;
    if (exp == 0x0F && mant == 7) {
        v = NAN;
    } else if (exp == 0) {
        v = ldexpf((float) mant, -9);
    } else {
        v = ldexpf(1.0f + (float) mant / 8.0f, exp - 7);
    }
    if (b & 0x80) v = -v;
    const unsigned char e = scales[((row0 + o) / block) * tiles_per_row + c / block];
    out[idx] = e == 0xFF ? NAN : v * ldexpf(1.0f, (int) e - 127);
}

// An f16 stored little-endian at `p`, decoded exactly.
__device__ static float req_f16_at(const unsigned char * p) {
    const unsigned int h = (unsigned int) p[0] | ((unsigned int) p[1] << 8);
    const int e = (h >> 10) & 0x1F;
    const float m = (float) (h & 0x3FF);
    float v = e == 0 ? ldexpf(m, -24) : (e == 31 ? (m == 0.0f ? INFINITY : NAN) : ldexpf(1024.0f + m, e - 25));
    return (h & 0x8000) ? -v : v;
}

// iq2_xxs to f32, one value per thread: block `idx / 256`, sub-block of 32, group of 8. `grid` holds the
// 256 lattice points as 8 bytes each and `signs` the 128 sign patterns their 7-bit selectors name.
extern "C" __global__ void dequant_iq2_xxs_f32(
        const unsigned char * __restrict__ blocks, const unsigned char * __restrict__ grid,
        const unsigned char * __restrict__ signs, float * __restrict__ out, const long total) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= total) return;
    const long ib = idx / 256;
    const int i = (int) (idx % 256), sub = i / 32, g = i / 8 - 4 * sub, j = i % 8;
    const unsigned char * b = blocks + ib * 66;
    const float d = req_f16_at(b);
    const unsigned char * q = b + 2 + 8 * sub;
    const unsigned int lo = (unsigned int) q[0] | ((unsigned int) q[1] << 8) | ((unsigned int) q[2] << 16) | ((unsigned int) q[3] << 24);
    const unsigned int hi = (unsigned int) q[4] | ((unsigned int) q[5] << 8) | ((unsigned int) q[6] << 16) | ((unsigned int) q[7] << 24);
    const float db = d * (0.5f + (float) (hi >> 28)) * 0.25f;
    const unsigned char point = grid[8 * ((lo >> (8 * g)) & 0xff) + j];
    const unsigned char pattern = signs[(hi >> (7 * g)) & 127];
    const float v = db * (float) point;
    out[idx] = ((pattern >> j) & 1) ? -v : v;
}

// One row of an IQ2_XXS weight [n, k] against a vector, its blocks read where they lie: the decode
// of dequant_iq2_xxs_f32, accumulated into a dot product instead of written out. A block per row,
// a thread per sub-block of thirty-two, then a reduction over the block. Reading the two-bit codes
// in place is the point: a decode that writes the row out first moves sixteen times the bytes.
extern "C" __global__ void mmv_iq2_xxs_f32(
        const unsigned char * __restrict__ blocks, const unsigned char * __restrict__ grid,
        const unsigned char * __restrict__ signs, const float * __restrict__ x,
        float * __restrict__ y, const int k, const int n) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const int sbs = k / 256;
    const unsigned char * rowp = blocks + (long) row * (long) sbs * 66;
    float acc = 0.0f;
    for (int u = threadIdx.x; u < sbs * 8; u += blockDim.x) {
        const int ib = u / 8, sub = u % 8;
        const unsigned char * b = rowp + (long) ib * 66;
        const float d = req_f16_at(b);
        const unsigned char * q = b + 2 + 8 * sub;
        const unsigned int lo = (unsigned int) q[0] | ((unsigned int) q[1] << 8) | ((unsigned int) q[2] << 16) | ((unsigned int) q[3] << 24);
        const unsigned int hi = (unsigned int) q[4] | ((unsigned int) q[5] << 8) | ((unsigned int) q[6] << 16) | ((unsigned int) q[7] << 24);
        const float db = d * (0.5f + (float) (hi >> 28)) * 0.25f;
        const float * xp = x + (long) ib * 256 + sub * 32;
        for (int g = 0; g < 4; ++g) {
            const unsigned char * gp = grid + 8 * ((lo >> (8 * g)) & 0xff);
            const unsigned char pattern = signs[(hi >> (7 * g)) & 127];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = db * (float) gp[j];
                acc += (((pattern >> j) & 1) ? -v : v) * xp[g * 8 + j];
            }
        }
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// Routed experts on a card, computed as the host computes them: the IQ2_XXS gate and up products
// against the q8_K32 rows of the input, the SwiGLU, the rows entering the down projection as q8_K
// blocks, and the Q2_K down product against them - the dots of quant_cpu and the SwiGLU of
// offload::experts, value for value. The integer sums are exact in any order; every float step is
// taken in the host's order through the rounding intrinsics, which the compiler never contracts.
// So an expert's product is the same bits on a card as on the host, and where it ran does not
// change the answer.
//
// The launches run a list of instances, an expert over one input row each, in tiles: a run of at
// most `tt` instances of one expert, `(expert, first instance, count)`, so a weight row is read
// once for all of them. One warp per (output row, tile).

#define CX_Q8K32_BYTES 272
#define CX_Q8K_BYTES 292
#define CX_IQ2XXS_BYTES 66
#define CX_Q2K_BYTES 84

// e^x from additions, multiplications and a division alone, as offload::experts::exp takes them.
__device__ static float cx_exp(const float x) {
    if (x != x) return x;
    if (x > 88.72284f) return INFINITY;
    if (x < -103.97208f) return 0.0f;
    const float k = rintf(__fmul_rn(x, 1.44269504088896341f));
    const float r = __fsub_rn(__fsub_rn(x, __fmul_rn(k, 0.6933594f)), __fmul_rn(k, -2.1219444e-4f));
    float p = 1.9875691e-4f;
    p = __fadd_rn(__fmul_rn(p, r), 1.3981999e-3f);
    p = __fadd_rn(__fmul_rn(p, r), 8.333452e-3f);
    p = __fadd_rn(__fmul_rn(p, r), 4.1665796e-2f);
    p = __fadd_rn(__fmul_rn(p, r), 0.16666665f);
    p = __fadd_rn(__fmul_rn(p, r), 0.5f);
    const float y = __fadd_rn(__fadd_rn(__fmul_rn(p, __fmul_rn(r, r)), r), 1.0f);
    const int ki = (int) k;
    const int k1 = ki / 2, k2 = ki - ki / 2;
    return __fmul_rn(__fmul_rn(y, __int_as_float((k1 + 127) << 23)), __int_as_float((k2 + 127) << 23));
}

// silu(g) * u with the gate clamped from above and the up value on both sides, as
// offload::experts::swiglu.
__device__ static float cx_swiglu(float g, float u, const float limit) {
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = u < -limit ? -limit : (u > limit ? limit : u);
    }
    return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, cx_exp(-g))), u);
}

// The 32 signed weights of an IQ2_XXS sub-block, four to an int in the order the activations run.
__device__ static void cx_iq2xxs_sub(
        const unsigned char * q, const unsigned char * grid, const unsigned char * signs,
        int * w, float * scale) {
    const unsigned int lo = (unsigned int) q[0] | ((unsigned int) q[1] << 8) | ((unsigned int) q[2] << 16) | ((unsigned int) q[3] << 24);
    const unsigned int hi = (unsigned int) q[4] | ((unsigned int) q[5] << 8) | ((unsigned int) q[6] << 16) | ((unsigned int) q[7] << 24);
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const unsigned char * gp = grid + 8 * ((lo >> (8 * g)) & 0xff);
        const unsigned char pattern = signs[(hi >> (7 * g)) & 127];
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            int packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int bit = 4 * half + j;
                const int v = ((pattern >> bit) & 1) ? -(int) gp[bit] : (int) gp[bit];
                packed |= (v & 0xff) << (8 * j);
            }
            w[2 * g + half] = packed;
        }
    }
    *scale = __fmul_rn(__fadd_rn(0.5f, (float) (hi >> 28)), 0.25f);
}

// The rows entering the down projections: instance i is expert tiles[3t] over row xrow[i] of `xq`
// (q8_K32 rows of `dim`), written to row i of `h` [instances, inter]. Shared memory holds each
// instance's sub-block terms and block shares: (2 * tt * (dim / 32) + 2 * tt * (dim / 256)) floats.
extern "C" __global__ void cx_expert_h_iq2_xxs(
        const unsigned long long * __restrict__ gate_ptrs,
        const unsigned long long * __restrict__ up_ptrs,
        const unsigned char * __restrict__ grid, const unsigned char * __restrict__ signs,
        const unsigned char * __restrict__ xq, const int * __restrict__ xrow,
        const int * __restrict__ tiles, float * __restrict__ h,
        const int dim, const int inter, const float limit, const int tt) {
    extern __shared__ float cx_shared[];
    const int row = blockIdx.x, tile = blockIdx.y, lane = threadIdx.x;
    const int e = tiles[3 * tile], i0 = tiles[3 * tile + 1], c = tiles[3 * tile + 2];
    const int sbs = dim / 256, nsub = sbs * 8;
    float * tg = cx_shared;
    float * tu = tg + tt * nsub;
    float * sg = tu + tt * nsub;
    float * su = sg + tt * sbs;
    const unsigned char * gate0 = (const unsigned char *) gate_ptrs[e] + (long) row * sbs * CX_IQ2XXS_BYTES;
    const unsigned char * up0 = (const unsigned char *) up_ptrs[e] + (long) row * sbs * CX_IQ2XXS_BYTES;
    for (int u = lane; u < nsub; u += 32) {
        const int ib = u / 8, sub = u % 8;
        int wg[8], wu[8];
        float scg, scu;
        cx_iq2xxs_sub(gate0 + (long) ib * CX_IQ2XXS_BYTES + 2 + 8 * sub, grid, signs, wg, &scg);
        cx_iq2xxs_sub(up0 + (long) ib * CX_IQ2XXS_BYTES + 2 + 8 * sub, grid, signs, wu, &scu);
        for (int j = 0; j < c; ++j) {
            const unsigned char * xb = xq + ((long) xrow[i0 + j] * sbs + ib) * CX_Q8K32_BYTES;
            const int * a = reinterpret_cast<const int *>(xb + 16 + 32 * sub);
            int ig = 0, iu = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int av = a[k];
                ig = __dp4a(wg[k], av, ig);
                iu = __dp4a(wu[k], av, iu);
            }
            const float yd = req_f16_at(xb + 2 * sub);
            tg[j * nsub + u] = __fmul_rn(__fmul_rn(scg, (float) ig), yd);
            tu[j * nsub + u] = __fmul_rn(__fmul_rn(scu, (float) iu), yd);
        }
    }
    __syncwarp();
    for (int p = lane; p < c * sbs; p += 32) {
        const int j = p / sbs, ib = p % sbs;
        float ag = 0.0f, au = 0.0f;
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            ag = __fadd_rn(ag, tg[j * nsub + ib * 8 + sub]);
            au = __fadd_rn(au, tu[j * nsub + ib * 8 + sub]);
        }
        sg[j * sbs + ib] = __fmul_rn(req_f16_at(gate0 + (long) ib * CX_IQ2XXS_BYTES), ag);
        su[j * sbs + ib] = __fmul_rn(req_f16_at(up0 + (long) ib * CX_IQ2XXS_BYTES), au);
    }
    __syncwarp();
    if (lane < c) {
        float g = -0.0f, u = -0.0f;
        for (int ib = 0; ib < sbs; ++ib) {
            g = __fadd_rn(g, sg[lane * sbs + ib]);
            u = __fadd_rn(u, su[lane * sbs + ib]);
        }
        h[(long) (i0 + lane) * inter + row] = cx_swiglu(g, u, limit);
    }
}

// Rows of `x` [rows, k] as q8_K blocks, the host's activation quantiser value for value: the
// extreme is the first value of the largest magnitude, the codes are its -127th parts rounded half
// away from zero and clipped at 127, the scale is the reciprocal, and each sixteen codes are
// summed. One warp per block, eight consecutive values to a lane.
extern "C" __global__ void cx_quantize_q8_k(
        const float * __restrict__ x, unsigned char * __restrict__ out, const long nblocks) {
    const long blk = blockIdx.x;
    if (blk >= nblocks) return;
    const int lane = threadIdx.x;
    const float * xb = x + blk * 256 + lane * 8;
    float v[8];
    float best = 0.0f, bv = 0.0f;
    int bi = 256;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        v[t] = xb[t];
        const float a = fabsf(v[t]);
        if (a > best) { best = a; bv = v[t]; bi = lane * 8 + t; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, off);
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ob > best || (ob == best && oi < bi)) { best = ob; bv = ov; bi = oi; }
    }
    unsigned char * ob = out + blk * CX_Q8K_BYTES;
    signed char * qs = reinterpret_cast<signed char *>(ob + 4);
    short * bsums = reinterpret_cast<short *>(ob + 260);
    int sum = 0;
    if (bv == 0.0f) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) qs[lane * 8 + t] = 0;
        if (lane == 0) *reinterpret_cast<float *>(ob) = 0.0f;
    } else {
        const float inv = __fdiv_rn(-127.0f, bv);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int q = (int) roundf(__fmul_rn(inv, v[t]));
            q = q > 127 ? 127 : q;
            qs[lane * 8 + t] = (signed char) q;
            sum += q;
        }
        if (lane == 0) *reinterpret_cast<float *>(ob) = __fdiv_rn(1.0f, inv);
    }
    const int pair = __shfl_down_sync(0xffffffffu, sum, 1);
    if ((lane & 1) == 0) bsums[lane / 2] = (short) (sum + pair);
}

// The outputs of the down projections: instance i is expert tiles[3t] over row i of `hq` (q8_K
// rows of `inter`), written to row i of `y` [instances, dim]. Shared memory holds each instance's
// sub-block sums and block shares: (2 * tt * (inter / 16)) ints and (tt * (inter / 256)) floats.
extern "C" __global__ void cx_expert_down_q2_k(
        const unsigned long long * __restrict__ down_ptrs,
        const unsigned char * __restrict__ hq, const int * __restrict__ tiles,
        float * __restrict__ y, const int inter, const int dim, const int tt) {
    extern __shared__ int cx_shared_i[];
    const int row = blockIdx.x, tile = blockIdx.y, lane = threadIdx.x;
    const int e = tiles[3 * tile], i0 = tiles[3 * tile + 1], c = tiles[3 * tile + 2];
    const int sbs = inter / 256, nsub = sbs * 16;
    int * isc = cx_shared_i;
    int * ioff = isc + tt * nsub;
    float * st = reinterpret_cast<float *>(ioff + tt * nsub);
    const unsigned char * row0 = (const unsigned char *) down_ptrs[e] + (long) row * sbs * CX_Q2K_BYTES;
    for (int u = lane; u < nsub; u += 32) {
        const int ib = u / 16, s = u % 16, t = s % 8;
        const unsigned char * b = row0 + (long) ib * CX_Q2K_BYTES;
        const int base = 32 * (s / 8) + 16 * (t % 2), shift = 2 * (t / 2);
        int w[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packed |= ((b[16 + base + 4 * k + j] >> shift) & 3) << (8 * j);
            }
            w[k] = packed;
        }
        const int sc = b[s] & 0x0F, mn = b[s] >> 4;
        for (int j = 0; j < c; ++j) {
            const unsigned char * hb = hq + ((long) (i0 + j) * sbs + ib) * CX_Q8K_BYTES;
            const int * a = reinterpret_cast<const int *>(hb + 4 + 16 * s);
            int dot = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) dot = __dp4a(w[k], a[k], dot);
            isc[j * nsub + u] = sc * dot;
            ioff[j * nsub + u] = mn * (int) reinterpret_cast<const short *>(hb + 260)[s];
        }
    }
    __syncwarp();
    for (int p = lane; p < c * sbs; p += 32) {
        const int j = p / sbs, ib = p % sbs;
        int scaled = 0, offset = 0;
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            scaled += isc[j * nsub + ib * 16 + s];
            offset += ioff[j * nsub + ib * 16 + s];
        }
        const unsigned char * b = row0 + (long) ib * CX_Q2K_BYTES;
        const unsigned char * hb = hq + ((long) (i0 + j) * sbs + ib) * CX_Q8K_BYTES;
        const float yd = *reinterpret_cast<const float *>(hb);
        st[j * sbs + ib] = __fmul_rn(yd, __fsub_rn(__fmul_rn(req_f16_at(b + 80), (float) scaled),
                                                  __fmul_rn(req_f16_at(b + 82), (float) offset)));
    }
    __syncwarp();
    if (lane < c) {
        float acc = -0.0f;
        for (int ib = 0; ib < sbs; ++ib) acc = __fadd_rn(acc, st[lane * sbs + ib]);
        y[(long) (i0 + lane) * dim + row] = acc;
    }
}

// q2_K to f32 at full precision, one value per thread: sixteen scale/minimum bytes, then the codes of
// sub-block `s` in the byte span 32 * (s / 8) + 16 * ((s % 8) % 2) at shift 2 * ((s % 8) / 2), then
// the f16 `d` and `dmin`.
extern "C" __global__ void dequant_q2_k_f32(
        const unsigned char * __restrict__ blocks, float * __restrict__ out, const long total) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= total) return;
    const long ib = idx / 256;
    const int i = (int) (idx % 256), sub = i / 16, t = sub % 8;
    const unsigned char * b = blocks + ib * 84;
    const unsigned char sc = b[sub];
    const int base = 32 * (sub / 8) + 16 * (t % 2), shift = 2 * (t / 2);
    const int code = (b[16 + base + i % 16] >> shift) & 3;
    const float d = req_f16_at(b + 80), dmin = req_f16_at(b + 82);
    out[idx] = d * (float) (sc & 0x0F) * (float) code - dmin * (float) (sc >> 4);
}

// q6_K to f32 at full precision, one value per thread: 128 low-nibble bytes, 64 bytes of high
// bit pairs, sixteen signed sub-block scales, then the f16 `d`. Value `i` of a superblock lies in
// half `h = i / 128`, quarter `k = (i % 128) / 32`, lane `l = i % 32`: its low four bits are the
// nibble of `ql[64 h + l + 32 (k % 2)]` (high nibble for k >= 2), its top two the pair `k` of
// `qh[32 h + l]`, and its scale is `sc[8 h + l / 16 + 2 k]`.
extern "C" __global__ void dequant_q6_k_f32(
        const unsigned char * __restrict__ blocks, float * __restrict__ out, const long total) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= total) return;
    const long ib = idx / 256;
    const int i = (int) (idx % 256), h = i / 128, k = (i % 128) / 32, l = i % 32;
    const unsigned char * b = blocks + ib * 210;
    const unsigned char * ql = b + 64 * h;
    const unsigned char * qh = b + 128 + 32 * h;
    const signed char * sc = (const signed char *) (b + 192) + 8 * h;
    const unsigned char lo = ql[l + 32 * (k % 2)];
    const int low = k < 2 ? (lo & 0x0F) : (lo >> 4);
    const int q = (low | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
    const float d = req_f16_at(b + 208);
    out[idx] = d * (float) sc[l / 16 + 2 * k] * (float) q;
}

// ---- A token group through a quantised weight, the codes read where they lie ----------------------

// ---- Hyper-connection steps over a prompt's residual stream x [n, hc, d] ----------------------------

#define HC_MAX 8

// Sum of `v` over the block, valid in thread 0. `scratch` holds one float per warp.
__device__ static float hc_block_sum(float v, float * scratch) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < (int) (blockDim.x + 31) / 32; ++w) total += scratch[w];
    }
    return total;
}

// A token's coefficients: the stream's statistic and the projection of its flattened hc*d values
// onto `fn` [mix_hc, flat], then the pre and post gates and the Sinkhorn-normalised comb matrix.
// One block per token; the projection is read as dot products, the rest runs in thread 0.
extern "C" __global__ void hc_mixes_f32(
        const float * __restrict__ x, const float * __restrict__ fn,
        const float * __restrict__ scale, const float * __restrict__ base,
        float * __restrict__ pre, float * __restrict__ post, float * __restrict__ comb,
        const long flat, const int hc, const int mix_hc, const int iters,
        const float norm_eps, const float hc_eps) {
    __shared__ float scratch[32];
    __shared__ float mixed[(2 + HC_MAX) * HC_MAX + 1];
    const long t = blockIdx.x;
    const float * row = x + t * flat;
    for (int m = 0; m <= mix_hc; ++m) {
        const float * w = m < mix_hc ? fn + (long) m * flat : row;
        float acc = 0.0f;
        for (long j = threadIdx.x; j < flat; j += blockDim.x) acc += row[j] * w[j];
        const float total = hc_block_sum(acc, scratch);
        if (threadIdx.x == 0) mixed[m] = total;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const float rsq = 1.0f / sqrtf(mixed[mix_hc] / (float) flat + norm_eps);
    float cm[HC_MAX][HC_MAX];
    for (int c = 0; c < hc; ++c) {
        const float a = mixed[c] * rsq * scale[0] + base[c];
        pre[t * hc + c] = 1.0f / (1.0f + expf(-a)) + hc_eps;
        const float b = mixed[hc + c] * rsq * scale[1] + base[hc + c];
        post[t * hc + c] = 2.0f / (1.0f + expf(-b));
    }
    for (int i = 0; i < hc; ++i)
        for (int j = 0; j < hc; ++j)
            cm[i][j] = mixed[2 * hc + i * hc + j] * rsq * scale[2] + base[2 * hc + i * hc + j];
    for (int i = 0; i < hc; ++i) {
        float mx = cm[i][0];
        for (int j = 1; j < hc; ++j) mx = fmaxf(mx, cm[i][j]);
        float den = 0.0f;
        for (int j = 0; j < hc; ++j) { cm[i][j] = expf(cm[i][j] - mx); den += cm[i][j]; }
        for (int j = 0; j < hc; ++j) cm[i][j] = cm[i][j] / den + hc_eps;
    }
    for (int it = 0; it < (iters > 0 ? iters : 1); ++it) {
        if (it > 0) {
            for (int i = 0; i < hc; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < hc; ++j) sum += cm[i][j];
                sum += hc_eps;
                for (int j = 0; j < hc; ++j) cm[i][j] /= sum;
            }
        }
        for (int j = 0; j < hc; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < hc; ++i) sum += cm[i][j];
            sum += hc_eps;
            for (int i = 0; i < hc; ++i) cm[i][j] /= sum;
        }
    }
    for (int i = 0; i < hc; ++i)
        for (int j = 0; j < hc; ++j) comb[t * hc * hc + i * hc + j] = cm[i][j];
}

// The stream collapsed by `pre` [n, hc] into one row per token and RMS-normalised by `w` [d]:
// out[t] = rms_norm(sum_c pre[t, c] * x[t, c]) * w. One block per token, the row held in shared memory.
extern "C" __global__ void hc_pre_norm_f32(
        const float * __restrict__ x, const float * __restrict__ pre,
        const float * __restrict__ w, float * __restrict__ out,
        const long d, const int hc, const float eps) {
    extern __shared__ float row[];
    __shared__ float scratch[32];
    const long t = blockIdx.x;
    float ss = 0.0f;
    for (long k = threadIdx.x; k < d; k += blockDim.x) {
        float v = 0.0f;
        for (int c = 0; c < hc; ++c) v += pre[t * hc + c] * x[(t * hc + c) * d + k];
        row[k] = v;
        ss += v * v;
    }
    const float total = hc_block_sum(ss, scratch);
    __shared__ float rsq;
    if (threadIdx.x == 0) rsq = 1.0f / sqrtf(total / (float) d + eps);
    __syncthreads();
    for (long k = threadIdx.x; k < d; k += blockDim.x) out[t * d + k] = row[k] * rsq * w[k];
}

// The sublayer's output `xo` [n, d] spread back over the hc copies, the residual mixed in through
// `comb`, in place: x[t, j] = post[t, j] * xo[t] + sum_i comb[t, i, j] * x[t, i]. One (token, value)
// per thread, which reads its hc residual values before it writes any of them.
extern "C" __global__ void hc_post_f32(
        const float * __restrict__ xo, float * __restrict__ x,
        const float * __restrict__ post, const float * __restrict__ comb,
        const long total, const long d, const int hc) {
    const long i = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (i >= total) return;
    const long t = i / d, k = i % d;
    float res[HC_MAX];
    for (int c = 0; c < hc; ++c) res[c] = x[(t * hc + c) * d + k];
    for (int j = 0; j < hc; ++j) {
        float acc = post[t * hc + j] * xo[i];
        for (int c = 0; c < hc; ++c) acc += comb[(t * hc + c) * hc + j] * res[c];
        x[(t * hc + j) * d + k] = acc;
    }
}

// Rotate the last `rd` values of each head of `x` [n, h, hd] in place, pair (2j, 2j + 1) by the angle
// whose cosine and sine are `cs[t, j]` and `sn[t, j]` for token t: the same float operations as the
// host's rope. One pair per thread.
extern "C" __global__ void rope_tail_f32(
        float * __restrict__ x, const float * __restrict__ cs, const float * __restrict__ sn,
        const long total, const long h, const long hd, const long rd) {
    const long i = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (i >= total) return;
    const long half = rd / 2;
    const long j = i % half, head = (i / half) % h, t = i / (half * h);
    float * p = x + (t * h + head) * hd + hd - rd + 2 * j;
    const float c = cs[t * half + j], s = sn[t * half + j];
    const float x0 = p[0], x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

// Columns `off .. off + w` of `dst` [n, dst_w] set from `src` [n, w], one value per thread.
extern "C" __global__ void put_cols_f32(
        const float * __restrict__ src, float * __restrict__ dst,
        const long total, const long w, const long dst_w, const long off) {
    const long i = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (i >= total) return;
    dst[(i / w) * dst_w + off + i % w] = src[i];
}

// Columns `off .. off + w` of `src` [n, src_w] copied into `dst` [n, w], one value per thread.
extern "C" __global__ void take_cols_f32(
        const float * __restrict__ src, float * __restrict__ dst,
        const long total, const long w, const long src_w, const long off) {
    const long i = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (i >= total) return;
    dst[i] = src[(i / w) * src_w + off + i % w];
}

// Index scores, one (query, compressed position) pair per thread: the sum over heads of
// relu(q . k) * weight * scale, for the positions query `i` can reach ((i + 1) / ratio of them);
// the rest are -inf. `q` [s, nh, ihd], `k` [g, ihd], `w` [s, nh]; the output is [s, g].
extern "C" __global__ void index_scores_f32(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ w,
        float * __restrict__ out, const long s, const long nh, const long ihd, const long g,
        const long ratio, const long q_off, const float scale) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= s * g) return;
    const long i = idx / g, t = idx % g;
    long reach = (i + q_off + 1) / ratio;
    if (reach > g) reach = g;
    if (t >= reach) { out[idx] = -INFINITY; return; }
    const float * kk = k + t * ihd;
    float acc = 0.0f;
    for (long hh = 0; hh < nh; hh++) {
        const float * qq = q + (i * nh + hh) * ihd;
        float dot = 0.0f;
        for (long c = 0; c < ihd; c++) dot += qq[c] * kk[c];
        acc += fmaxf(dot, 0.0f) * w[i * nh + hh] * scale;
    }
    out[idx] = acc;
}

// The keys of each query of a chunk gathered: row `k` of query `i` is `kv[idxs[(start + i) * topk + k]]`,
// zeros for an empty slot. The output is [count, topk, d].
extern "C" __global__ void gather_slots_f32(
        const float * __restrict__ kv, const int * __restrict__ idxs, float * __restrict__ out,
        const long start, const long count, const long topk, const long d) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= count * topk) return;
    const long slot = idxs[start * topk + idx];
    float * o = out + idx * d;
    if (slot < 0) {
        for (long t = 0; t < d; t++) o[t] = 0.0f;
    } else {
        const float * k = kv + slot * d;
        for (long t = 0; t < d; t++) o[t] = k[t];
    }
}

// Logits [count, h, topk] turned in place into attention weights: scaled, softmaxed with the head's
// sink logit, empty slots weighing nothing.
extern "C" __global__ void sink_softmax_f32(
        float * __restrict__ logits, const float * __restrict__ sink, const int * __restrict__ idxs,
        const long start, const long count, const long h, const long topk, const float scale) {
    const long idx = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (idx >= count * h) return;
    const long i = idx / h, hi = idx % h;
    const int * slots = idxs + (start + i) * topk;
    float * l = logits + idx * topk;
    float m = sink[hi];
    for (long k = 0; k < topk; k++) {
        if (slots[k] >= 0) m = fmaxf(m, l[k] * scale);
    }
    float denom = exp(sink[hi] - m);
    for (long k = 0; k < topk; k++) {
        if (slots[k] >= 0) {
            l[k] = exp(l[k] * scale - m);
            denom += l[k];
        } else {
            l[k] = 0.0f;
        }
    }
    for (long k = 0; k < topk; k++) l[k] /= denom;
}
