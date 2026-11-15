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

// Float steps taken one rounding at a time, in the host encoder's order: a product and the sum it
// feeds are never fused, so the search below picks what the host picks.
#if defined(GGML_USE_HIP)
#define RQ_EXACT _Pragma("clang fp contract(off)")
#define RQ_MUL(a, b) ((a) * (b))
#define RQ_ADD(a, b) ((a) + (b))
#define RQ_SUB(a, b) ((a) - (b))
#else
#define RQ_EXACT
#define RQ_MUL(a, b) __fmul_rn((a), (b))
#define RQ_ADD(a, b) __fadd_rn((a), (b))
#define RQ_SUB(a, b) __fsub_rn((a), (b))
#endif

// q3_K guided by an importance: the host encoder's weighting and its two signed fits, step for
// step. `fit_signed_scale` over `n` values to levels -nmax..nmax-1, written into `ls` biased by
// nmax; returns the scale. A zero group leaves every level at 0, as the host does.
__device__ static float rq_score(int nmax, int n, const float * x, const float * w, float inv, float * wl2_out) {
    RQ_EXACT
    float wxl = 0.0f, wl2 = 0.0f;
    for (int i = 0; i < n; i++) {
        const float l = (float) req_clampi(req_nint(RQ_MUL(inv, x[i])), -nmax, nmax - 1);
        wxl = RQ_ADD(wxl, RQ_MUL(RQ_MUL(w[i], x[i]), l));
        wl2 = RQ_ADD(wl2, RQ_MUL(RQ_MUL(w[i], l), l));
    }
    *wl2_out = wl2;
    return wxl;
}

__device__ static float rq_fit_signed(int nmax, int n, const float * x, unsigned char * ls, const float * w) {
    RQ_EXACT
    float extreme = 0.0f;
    for (int i = 0; i < n; i++) if (fabsf(x[i]) > fabsf(extreme)) extreme = x[i];
    if (extreme == 0.0f) {
        for (int i = 0; i < n; i++) ls[i] = 0;
        return 0.0f;
    }
    float inv = -(float) nmax / extreme;
    for (int i = 0; i < n; i++) ls[i] = (unsigned char) (req_clampi(req_nint(RQ_MUL(inv, x[i])), -nmax, nmax - 1) + nmax);
    float wl2, wxl = rq_score(nmax, n, x, w, inv, &wl2);
    float best = RQ_MUL(wxl / wl2, wxl);

    for (int r = 0; r < 3; r++) {
        inv = 1.0f / (wxl / wl2);
        bool unchanged = true;
        for (int i = 0; i < n; i++) {
            if (req_clampi(req_nint(RQ_MUL(inv, x[i])), -nmax, nmax - 1) + nmax != (int) ls[i]) unchanged = false;
        }
        float cwl2, cwxl = rq_score(nmax, n, x, w, inv, &cwl2);
        if (unchanged || cwl2 == 0.0f || RQ_MUL(cwxl, cwxl) <= RQ_MUL(best, cwl2)) break;
        for (int i = 0; i < n; i++) ls[i] = (unsigned char) (req_clampi(req_nint(RQ_MUL(inv, x[i])), -nmax, nmax - 1) + nmax);
        wxl = cwxl; wl2 = cwl2; best = RQ_MUL(wxl / wl2, wxl);
    }

    for (int pass = 0; pass < 5; pass++) {
        int moved = 0;
        for (int i = 0; i < n; i++) {
            const float v = x[i], ww = w[i];
            const int l = (int) ls[i] - nmax;
            float nwxl = RQ_SUB(wxl, RQ_MUL(RQ_MUL(ww, v), (float) l));
            if (nwxl <= 0.0f) continue;
            float nwl2 = RQ_SUB(wl2, RQ_MUL(ww, (float) (l * l)));
            const int to = req_clampi(req_nint(RQ_MUL(v, nwl2) / nwxl), -nmax, nmax - 1);
            if (to == l) continue;
            nwxl = RQ_ADD(nwxl, RQ_MUL(RQ_MUL(ww, v), (float) to));
            nwl2 = RQ_ADD(nwl2, RQ_MUL(ww, (float) (to * to)));
            if (nwl2 > 0.0f && RQ_MUL(RQ_MUL(nwxl, nwxl), wl2) > RQ_MUL(RQ_MUL(wxl, wxl), nwl2)) {
                ls[i] = (unsigned char) (to + nmax);
                wxl = nwxl; wl2 = nwl2; best = RQ_MUL(wxl / wl2, wxl);
                moved++;
            }
        }
        if (moved == 0) break;
    }

    for (int s = -4; s <= 4; s++) {
        if (s == 0) continue;
        const float sinv = -RQ_ADD((float) nmax, RQ_MUL(0.1f, (float) s)) / extreme;
        float cwl2, cwxl = rq_score(nmax, n, x, w, sinv, &cwl2);
        if (cwl2 > 0.0f && RQ_MUL(cwxl, cwxl) > RQ_MUL(best, cwl2)) {
            for (int i = 0; i < n; i++) ls[i] = (unsigned char) (req_clampi(req_nint(RQ_MUL(sinv, x[i])), -nmax, nmax - 1) + nmax);
            wxl = cwxl; wl2 = cwl2; best = RQ_MUL(wxl / wl2, wxl);
        }
    }
    return wxl / wl2;
}

// q3_K weighted by an importance, the host encoder's guided path: a value weighs its column's
// importance times sqrt(sigma2 + x^2), sigma2 twice the block's mean square; each group of sixteen
// fits its signed scale, and the sixteen scales fit six signed bits against the weight each group
// carries. Values are laid out as `requant_iq2_xxs` takes them, and with `has_got` their decoded
// values are written to `got`. The same blocks as the host encoder.
extern "C" __global__ void requant_q3_k_guided(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy,
        float * __restrict__ got, const int has_got,
        const int nblocks, const int blocks_per_row) {
    RQ_EXACT
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x; // 0..31
    const int CODE_MAX = 4, SCALE_BIAS = 32, GROUPS = 16, GSUB = 16;
    __shared__ float sh_x[REQ_QK_K];
    __shared__ float sh_sigma2;
    __shared__ float sh_scale[16];
    __shared__ float sh_carried[16];
    __shared__ float sh_dg[16];
    __shared__ unsigned char sh_code[REQ_QK_K];
    __shared__ unsigned char sh_low[REQ_QK_K];
    for (int i = t; i < REQ_QK_K; i += 32) sh_x[i] = x[(size_t) sb * block_step + (size_t) i * value_step];
    const float * ib = imp + (size_t) (sb % blocks_per_row) * REQ_QK_K;
    block_q3_K * y = (block_q3_K *) vy + sb;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < REQ_QK_K; i++) s = RQ_ADD(s, RQ_MUL(sh_x[i], sh_x[i]));
        sh_sigma2 = RQ_MUL(2.0f, s) / (float) REQ_QK_K;
    }
    __syncthreads();

    if (t < GROUPS) {
        const float * xg = sh_x + t * GSUB;
        float w[16];
        unsigned char ls[16];
        float carried = 0.0f;
        for (int i = 0; i < GSUB; i++) {
            w[i] = RQ_MUL(ib[t * GSUB + i], sqrtf(RQ_ADD(sh_sigma2, RQ_MUL(xg[i], xg[i]))));
            carried = RQ_ADD(carried, w[i]);
        }
        sh_carried[t] = carried;
        sh_scale[t] = rq_fit_signed(CODE_MAX, GSUB, xg, ls, w);
    }
    __syncthreads();

    if (t == 0) {
        unsigned char ls[16];
        const float d = rq_fit_signed(SCALE_BIAS, GROUPS, sh_scale, ls, sh_carried);
        half hd = __float2half(d);
        y->d = hd;
        const float d_used = __half2float(hd);
        unsigned char sc[12];
        for (int i = 0; i < 12; i++) sc[i] = 0;
        for (int i = 0; i < 16; i++) {
            const int level = ls[i];
            if (i < 8) sc[i] |= (unsigned char) (level & 0x0F);
            else sc[i - 8] |= (unsigned char) ((level & 0x0F) << 4);
            sc[8 + i % 4] |= (unsigned char) ((level >> 4) << (2 * (i / 4)));
        }
        for (int i = 0; i < 12; i++) y->scales[i] = sc[i];
        for (int g = 0; g < GROUPS; g++) {
            const int grp = g / 4, j = g % 4;
            const unsigned char byte = sc[j + 4 * (grp % 2)];
            const int low = grp < 2 ? (byte & 0x0F) : (byte >> 4);
            const int high = (sc[8 + j] >> (2 * grp)) & 0x03;
            sh_dg[g] = RQ_MUL(d_used, (float) ((low | (high << 4)) - 32));
        }
    }
    __syncthreads();

    for (int idx = t; idx < REQ_QK_K; idx += 32) {
        const float dg = sh_dg[idx / GSUB];
        int code = 0;
        if (dg != 0.0f) code = req_clampi(req_nint(sh_x[idx] / dg), -CODE_MAX, CODE_MAX - 1) + CODE_MAX;
        sh_code[idx] = (unsigned char) code;
        sh_low[idx] = (unsigned char) (code >= CODE_MAX ? code - CODE_MAX : code);
        if (has_got) got[(size_t) sb * block_step + (size_t) idx * value_step] = RQ_MUL(dg, (float) (code - CODE_MAX));
    }
    __syncthreads();
    for (int m = t; m < REQ_QK_K / 8; m += 32) {
        unsigned char h = 0;
        for (int p = 0; p < 8; p++) if (sh_code[p * 32 + m] >= CODE_MAX) h |= (unsigned char) (1u << p);
        y->hmask[m] = h;
    }
    for (int b = t; b < REQ_QK_K / 4; b += 32) {
        const int half = b / 32, l = b % 32;
        const unsigned char * q = &sh_low[half * 128];
        y->qs[b] = (unsigned char) (q[l] | (q[l + 32] << 2) | (q[l + 64] << 4) | (q[l + 96] << 6));
    }
}

// iq3_xxs: eight sub-blocks of thirty-two, each a 4-bit scale under the block's f16 scale and eight
// groups of four magnitudes, each group one point of a 256-point grid, with seven signs per eight
// values and the eighth by parity. The host encoder's search, step for step: each sub-block tries
// thirty-one scales around its largest magnitude, a group the grid does not hold taking the nearest
// point among the ones `book` lists for it, the scale refitted by least squares; the best fit wins
// and its off-grid groups are placed again at it. One CUDA block of 256 threads per block: thread
// (sub-block, step). `book` is `iq3_xxs_codebook`. `imp` holds one weight per column of a row,
// `blocks_per_row` blocks of it; without one a value weighs its square. With `has_got` the values
// the blocks decode to are written to `got`, in the layout `x` has. The same blocks as the host.
#define IQ3_INDEX 0
#define IQ3_OFFSETS 4096
#define IQ3_LEVELS (IQ3_OFFSETS + 4097)
#define IQ3_BYTES (IQ3_LEVELS + 1024)
#define IQ3_NEAR (IQ3_BYTES + 1024)
#define IQ3_STEPS 31

typedef struct {
    half d;
    unsigned char qs[96];
} req_block_iq3_xxs;

// The pattern of the four levels nearest `x` at `inverse`, the levels written to `lv`.
__device__ static int rq_iq3_round(const float * x, const float inverse, int * lv) {
    RQ_EXACT
    int p = 0;
    for (int i = 0; i < 4; i++) {
        lv[i] = req_clampi(req_nint(RQ_MUL(0.5f, RQ_SUB(RQ_MUL(inverse, x[i]), 1.0f))), 0, 7);
        p |= lv[i] << (3 * i);
    }
    return p;
}

// The point the grid holds for pattern `p`, or the one among its near points that leaves the group
// the least error at `scale`.
__device__ static int rq_iq3_point(const int * __restrict__ book, const int p, const float * x,
        const float * w, const float scale, bool * held) {
    RQ_EXACT
    const int n = book[IQ3_INDEX + p];
    if (n >= 0) { *held = true; return n; }
    *held = false;
    const int from = book[IQ3_OFFSETS + p], to = book[IQ3_OFFSETS + p + 1];
    int at = book[IQ3_NEAR + from];
    float best = INFINITY;
    for (int c = from; c < to; c++) {
        const int m = book[IQ3_NEAR + c];
        const int * l = book + IQ3_LEVELS + 4 * m;
        float err = 0.0f;
        for (int i = 0; i < 4; i++) {
            const float d = RQ_SUB(RQ_MUL(scale, (float) (2 * l[i] + 1)), x[i]);
            err = RQ_ADD(err, RQ_MUL(RQ_MUL(w[i], d), d));
        }
        if (err < best) { best = err; at = m; }
    }
    return at;
}

// `sum w x q` and `sum w q q` for the levels `l` of a sub-block, q = 2 l + 1.
__device__ static float rq_iq3_fit(const float * w, const float * x, const unsigned char * l, float * q2_out) {
    RQ_EXACT
    float qx = 0.0f, q2 = 0.0f;
    for (int i = 0; i < 32; i++) {
        const float q = (float) (2 * (int) l[i] + 1);
        qx = RQ_ADD(qx, RQ_MUL(RQ_MUL(w[i], x[i]), q));
        q2 = RQ_ADD(q2, RQ_MUL(RQ_MUL(w[i], q), q));
    }
    *q2_out = q2;
    return qx;
}

extern "C" __global__ void requant_iq3_xxs(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, const int * __restrict__ book, void * __restrict__ vy,
        float * __restrict__ got, const int has_got,
        const int nblocks, const int blocks_per_row, const int has_imp) {
    RQ_EXACT
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x, sub = t / 32, lane = t % 32;
    const float TOP = 15.0f, STEP = 0.2f, EMPTY = 1e-8f, SCALE_STEPS = 31.0f, SCALE_LIFT = 1.0125f;
    __shared__ float sh_x[REQ_QK_K];
    __shared__ float sh_v[REQ_QK_K];
    __shared__ float sh_w[REQ_QK_K];
    __shared__ float sh_r[REQ_QK_K];
    __shared__ float sh_sigma2;
    __shared__ unsigned int sh_signs[32];
    __shared__ float sh_max[8];
    __shared__ unsigned char sh_trial[8][IQ3_STEPS][32];
    __shared__ unsigned char sh_held[8][IQ3_STEPS];
    __shared__ float sh_qx[8][IQ3_STEPS];
    __shared__ float sh_q2[8][IQ3_STEPS];
    __shared__ float sh_scale[8];
    __shared__ unsigned int sh_word[8];
    __shared__ unsigned char sh_code[64];
    __shared__ float sh_d;

    sh_x[t] = x[(size_t) sb * block_step + (size_t) t * value_step];
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < REQ_QK_K; i++) s = RQ_ADD(s, RQ_MUL(sh_x[i], sh_x[i]));
        sh_sigma2 = RQ_MUL(2.0f, s) / (float) REQ_QK_K;
    }
    __syncthreads();
    {
        const float v = sh_x[t];
        const float w = has_imp
            ? RQ_MUL(imp[(size_t) (sb % blocks_per_row) * REQ_QK_K + t], sqrtf(RQ_ADD(sh_sigma2, RQ_MUL(v, v))))
            : RQ_MUL(v, v);
        sh_w[t] = w;
        sh_r[t] = sqrtf(w);
        sh_v[t] = fabsf(v);
    }
    __syncthreads();

    // Each group of eight given an even count of negatives: an odd one flips its cheapest value.
    if (t < 32) {
        const float * xg = sh_x + 8 * t;
        const float * wg = sh_w + 8 * t;
        unsigned int s = 0;
        for (int i = 0; i < 8; i++) if (xg[i] < 0.0f) s |= 1u << i;
        if (__popc(s) & 1) {
            int cheapest = 0;
            float least = RQ_MUL(RQ_MUL(wg[0], xg[0]), xg[0]);
            for (int i = 1; i < 8; i++) {
                const float cost = RQ_MUL(RQ_MUL(wg[i], xg[i]), xg[i]);
                if (cost < least) { least = cost; cheapest = i; }
            }
            sh_v[8 * t + cheapest] = -sh_v[8 * t + cheapest];
            s ^= 1u << cheapest;
        }
        sh_signs[t] = s & 127;
    }
    __syncthreads();
    if (lane == 0) {
        float m = -INFINITY;
        for (int i = 0; i < 32; i++) m = fmaxf(m, sh_v[32 * sub + i]);
        sh_max[sub] = m;
    }
    __syncthreads();

    const float * xv = sh_v + 32 * sub;
    const float * wv = sh_w + 32 * sub;
    const float * rv = sh_r + 32 * sub;
    const float max = sh_max[sub];
    const bool live = max >= EMPTY;
    if (live && lane < IQ3_STEPS) {
        const float inverse = RQ_ADD(TOP, RQ_MUL((float) (lane - 15), STEP)) / max;
        unsigned char * trial = sh_trial[sub][lane];
        unsigned int held_mask = 0;
        for (int k = 0; k < 8; k++) {
            int lv[4];
            const int p = rq_iq3_round(xv + 4 * k, inverse, lv);
            bool held;
            const int n = rq_iq3_point(book, p, xv + 4 * k, rv + 4 * k, 1.0f / inverse, &held);
            if (held) held_mask |= 1u << k;
            for (int i = 0; i < 4; i++) trial[4 * k + i] = (unsigned char) book[IQ3_LEVELS + 4 * n + i];
        }
        float q2;
        const float qx = rq_iq3_fit(wv, xv, trial, &q2);
        sh_qx[sub][lane] = qx;
        sh_q2[sub][lane] = q2;
        sh_held[sub][lane] = (unsigned char) held_mask;
    }
    __syncthreads();

    if (lane == 0) {
        unsigned int word = 0;
        float scale = 0.0f;
        unsigned char l[32];
        for (int i = 0; i < 32; i++) l[i] = 0;
        if (live) {
            float best = 0.0f;
            scale = max / TOP;
            int chosen = -1;
            for (int s = 0; s < IQ3_STEPS; s++) {
                const float qx = sh_qx[sub][s], q2 = sh_q2[sub][s];
                if (q2 > 0.0f && RQ_MUL(qx, qx) > RQ_MUL(best, q2)) {
                    scale = qx / q2;
                    best = RQ_MUL(scale, qx);
                    chosen = s;
                }
            }
            unsigned int on_grid = 0xff;
            if (chosen >= 0) {
                for (int i = 0; i < 32; i++) l[i] = sh_trial[sub][chosen][i];
                on_grid = sh_held[sub][chosen];
            }
            if (on_grid != 0xff && scale > 0.0f) {
                const float inverse = 1.0f / scale;
                for (int k = 0; k < 8; k++) {
                    if ((on_grid >> k) & 1) continue;
                    int lv[4];
                    const int p = rq_iq3_round(xv + 4 * k, inverse, lv);
                    bool held;
                    const int n = rq_iq3_point(book, p, xv + 4 * k, rv + 4 * k, scale, &held);
                    for (int i = 0; i < 4; i++) l[4 * k + i] = (unsigned char) book[IQ3_LEVELS + 4 * n + i];
                }
                float q2;
                const float qx = rq_iq3_fit(wv, xv, l, &q2);
                if (q2 > 0.0f) scale = qx / q2;
            }
            unsigned int signs[4];
            for (int k = 0; k < 4; k++) signs[k] = sh_signs[4 * sub + k];
            if (scale < 0.0f) {
                scale = -scale;
                for (int k = 0; k < 4; k++) signs[k] = ~signs[k] & 127;
            }
            word = signs[0] | signs[1] << 7 | signs[2] << 14 | signs[3] << 21;
        }
        for (int k = 0; k < 8; k++) {
            int p = 0;
            for (int i = 0; i < 4; i++) p |= (int) l[4 * k + i] << (3 * i);
            sh_code[8 * sub + k] = live ? (unsigned char) book[IQ3_INDEX + p] : 0;
        }
        sh_word[sub] = word;
        sh_scale[sub] = scale;
    }
    __syncthreads();

    req_block_iq3_xxs * y = (req_block_iq3_xxs *) vy + sb;
    if (t == 0) {
        float max_scale = 0.0f;
        for (int i = 0; i < 8; i++) max_scale = fmaxf(max_scale, sh_scale[i]);
        if (max_scale == 0.0f) {
            y->d = __float2half(0.0f);
            for (int i = 0; i < 64; i++) sh_code[i] = 0;
            for (int i = 0; i < 8; i++) sh_word[i] = 0;
            sh_d = 0.0f;
        } else {
            const float d = max_scale / SCALE_STEPS;
            half hd = __float2half(RQ_MUL(d, SCALE_LIFT));
            y->d = hd;
            sh_d = __half2float(hd);
            for (int i = 0; i < 8; i++) {
                const int level = req_clampi(req_nint(RQ_MUL(0.5f, RQ_SUB(sh_scale[i] / d, 1.0f))), 0, 15);
                sh_word[i] |= (unsigned int) level << 28;
            }
        }
    }
    __syncthreads();
    if (t < 64) y->qs[t] = sh_code[t];
    if (t < 32) {
        const unsigned int word = sh_word[t / 4];
        y->qs[64 + t] = (unsigned char) (word >> (8 * (t % 4)));
    }
    if (has_got) {
        const int g = (t % 32) / 8, j = t % 8;
        const unsigned int word = sh_word[sub];
        const float db = RQ_MUL(RQ_MUL(sh_d, RQ_ADD(0.5f, (float) (word >> 28))), 0.5f);
        unsigned int s7 = (word >> (7 * g)) & 127;
        const unsigned int signs = s7 | ((__popc(s7) & 1) << 7);
        const int code = sh_code[8 * sub + 2 * g + (j >= 4)];
        const float mag = RQ_MUL(db, (float) book[IQ3_BYTES + 4 * code + (j % 4)]);
        got[(size_t) sb * block_step + (size_t) t * value_step] = ((signs >> j) & 1) ? -mag : mag;
    }
}

// The grid formats iq3_s and iq2_s: groups of D magnitudes, each group one point of a grid, a sign
// bit per value and a four-bit scale per run under the block's f16 scale. The host encoders'
// searches, step for step: each run tries nineteen scales around its largest magnitude, a group
// the grid does not hold taking the nearest point among the ones `book` lists for it, the scale
// refitted by least squares; the best fit wins and groups are placed again at it. `book` is the
// format's codebook table (`iq3_s_codebook`, `iq2_s_codebook`): the point of each pattern or -1,
// the offsets of the near lists, each point's levels, its magnitudes, then the lists. `imp` holds
// one weight per column of a row, `blocks_per_row` blocks of it. With `has_got` the values the
// blocks decode to are written to `got`, in the layout `x` has. The same blocks as the host.
#define RQ_GRID_STEPS 19

// The point a group of D values takes: the one its rounded levels name, or the nearest the grid
// holds at `scale`. `top` is the highest level; `bits` the bits a level takes in a pattern.
template <int D>
__device__ static int rq_grid_place(const int * __restrict__ book, const int npat, const int npts,
        const int bits, const int top, const float * x, const float * w, const float inverse,
        const float scale, bool * held) {
    RQ_EXACT
    int p = 0;
    for (int i = 0; i < D; i++) {
        const int l = req_clampi(req_nint(RQ_MUL(0.5f, RQ_SUB(RQ_MUL(inverse, x[i]), 1.0f))), 0, top);
        p |= l << (bits * i);
    }
    const int n = book[p];
    if (n >= 0) { *held = true; return n; }
    *held = false;
    const int * offsets = book + npat;
    const int * levels = offsets + npat + 1;
    const int * near = levels + 2 * D * npts;
    int at = -1;
    float best = 3.402823466e38f;
    for (int c = offsets[p]; c < offsets[p + 1]; c++) {
        const int m = near[c];
        const int * l = levels + D * m;
        float err = 0.0f;
        for (int i = 0; i < D; i++) {
            const float d = RQ_SUB(RQ_MUL(scale, (float) (2 * l[i] + 1)), x[i]);
            err = RQ_ADD(err, RQ_MUL(RQ_MUL(w[i], d), d));
        }
        if (err < best) { best = err; at = m; }
    }
    return at;
}

// `sum w x q` over n values of levels `l`, q = 2 l + 1; `sum w q q` into `q2_out`.
__device__ static float rq_grid_fit(const int n, const float * w, const float * x, const unsigned char * l, float * q2_out) {
    RQ_EXACT
    float qx = 0.0f, q2 = 0.0f;
    for (int i = 0; i < n; i++) {
        const float q = (float) (2 * (int) l[i] + 1);
        qx = RQ_ADD(qx, RQ_MUL(RQ_MUL(w[i], x[i]), q));
        q2 = RQ_ADD(q2, RQ_MUL(RQ_MUL(w[i], q), q));
    }
    *q2_out = q2;
    return qx;
}

// One block of a grid format. RUN values a run, D a group; LEVELS levels, BITS bits a level in a
// pattern; NPAT patterns, NPTS points; STEP apart the scales tried; a run below EMPTY is zeros;
// the second placement takes every group (ALL) or the off-grid ones; a value without importance
// weighs its square plus FLOOR times the block's spread. `out_run` writes a run's codes, signs
// and scale into the block; `decode` gives value t of the block from it.
template <int RUN, int D, int LEVELS, int BITS, int NPAT, int NPTS, bool ALL>
__device__ static void rq_grid_block(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, const int * __restrict__ book,
        const int sb, const int blocks_per_row, const int has_imp,
        const float STEP, const float EMPTY, const float FLOOR,
        float * sh_x, float * sh_v, float * sh_w, float * sh_r, float * sh_sigma2,
        unsigned char * sh_sign, float * sh_max, unsigned char * sh_trial, unsigned int * sh_held,
        float * sh_qx, float * sh_q2, float * sh_scale, int * sh_point) {
    RQ_EXACT
    const int RUNS = REQ_QK_K / RUN, GROUPS = RUN / D;
    const float TOP = (float) (2 * LEVELS - 1);
    const int t = threadIdx.x;
    if (t < REQ_QK_K) sh_x[t] = x[(size_t) sb * block_step + (size_t) t * value_step];
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < REQ_QK_K; i++) s = RQ_ADD(s, RQ_MUL(sh_x[i], sh_x[i]));
        *sh_sigma2 = RQ_MUL(2.0f, s) / (float) REQ_QK_K;
    }
    __syncthreads();
    if (t < REQ_QK_K) {
        const float v = sh_x[t];
        const float w = has_imp
            ? RQ_MUL(imp[(size_t) (sb % blocks_per_row) * REQ_QK_K + t], sqrtf(RQ_ADD(*sh_sigma2, RQ_MUL(v, v))))
            : RQ_ADD(RQ_MUL(FLOOR, *sh_sigma2), RQ_MUL(v, v));
        sh_w[t] = w;
        sh_r[t] = sqrtf(w);
        sh_v[t] = fabsf(v);
    }
    if (t < REQ_QK_K / 8) {
        unsigned int s = 0;
        for (int i = 0; i < 8; i++) if (sh_x[8 * t + i] < 0.0f) s |= 1u << i;
        sh_sign[t] = (unsigned char) s;
    }
    __syncthreads();
    if (t < RUNS) {
        float m = sh_v[RUN * t];
        for (int i = 0; i < RUN; i++) m = fmaxf(m, sh_v[RUN * t + i]);
        sh_max[t] = m;
    }
    __syncthreads();

    // Thread (run, step): run t / 32, step t % 32.
    {
        const int run = t / 32, lane = t % 32;
        if (run < RUNS && lane < RQ_GRID_STEPS) {
            const float max = sh_max[run];
            const bool live = EMPTY > 0.0f ? max >= EMPTY : max != 0.0f;
            if (live) {
                const float * xv = sh_v + RUN * run;
                const float * rv = sh_r + RUN * run;
                const float inverse = RQ_ADD(TOP, RQ_MUL((float) (lane - 9), STEP)) / max;
                const float this_scale = 1.0f / inverse;
                unsigned char * trial = sh_trial + (run * RQ_GRID_STEPS + lane) * RUN;
                const int * levels = book + 2 * NPAT + 1;
                unsigned int held_mask = 0;
                for (int k = 0; k < GROUPS; k++) {
                    bool held;
                    const int n = rq_grid_place<D>(book, NPAT, NPTS, BITS, LEVELS - 1, xv + D * k,
                                                   rv + D * k, inverse, this_scale, &held);
                    if (held) held_mask |= 1u << k;
                    for (int i = 0; i < D; i++) trial[D * k + i] = (unsigned char) levels[D * n + i];
                }
                float q2;
                const float qx = rq_grid_fit(RUN, sh_w + RUN * run, xv, trial, &q2);
                sh_qx[run * RQ_GRID_STEPS + lane] = qx;
                sh_q2[run * RQ_GRID_STEPS + lane] = q2;
                sh_held[run * RQ_GRID_STEPS + lane] = held_mask;
            }
        }
    }
    __syncthreads();

    if (t < RUNS) {
        const int run = t;
        const float max = sh_max[run];
        const bool live = EMPTY > 0.0f ? max >= EMPTY : max != 0.0f;
        const float * xv = sh_v + RUN * run;
        const float * rv = sh_r + RUN * run;
        const int * levels = book + 2 * NPAT + 1;
        float scale = 0.0f;
        unsigned char l[RUN];
        for (int i = 0; i < RUN; i++) l[i] = 0;
        if (live) {
            float best = 0.0f;
            scale = max / TOP;
            unsigned int on_grid = ALL ? 0u : (1u << GROUPS) - 1;
            for (int s = 0; s < RQ_GRID_STEPS; s++) {
                const float qx = sh_qx[run * RQ_GRID_STEPS + s], q2 = sh_q2[run * RQ_GRID_STEPS + s];
                if (q2 > 0.0f && RQ_MUL(qx, qx) > RQ_MUL(best, q2)) {
                    scale = qx / q2;
                    best = RQ_MUL(scale, qx);
                    for (int i = 0; i < RUN; i++) l[i] = sh_trial[(run * RQ_GRID_STEPS + s) * RUN + i];
                    on_grid = sh_held[run * RQ_GRID_STEPS + s];
                }
            }
            if (on_grid != (1u << GROUPS) - 1 && scale > 0.0f) {
                const float inverse = 1.0f / scale;
                for (int k = 0; k < GROUPS; k++) {
                    if (!ALL && ((on_grid >> k) & 1)) continue;
                    bool held;
                    const int n = rq_grid_place<D>(book, NPAT, NPTS, BITS, LEVELS - 1, xv + D * k,
                                                   rv + D * k, inverse, scale, &held);
                    for (int i = 0; i < D; i++) l[D * k + i] = (unsigned char) levels[D * n + i];
                }
                float q2;
                const float qx = rq_grid_fit(RUN, sh_w + RUN * run, xv, l, &q2);
                if (q2 > 0.0f) scale = qx / q2;
            }
            if (scale < 0.0f) {
                scale = -scale;
                for (int k = 0; k < RUN / 8; k++) sh_sign[RUN / 8 * run + k] = (unsigned char) ~sh_sign[RUN / 8 * run + k];
            }
            for (int k = 0; k < GROUPS; k++) {
                int p = 0;
                for (int i = 0; i < D; i++) p |= (int) l[D * k + i] << (BITS * i);
                sh_point[GROUPS * run + k] = book[p];
            }
        } else {
            for (int k = 0; k < GROUPS; k++) sh_point[GROUPS * run + k] = 0;
            for (int k = 0; k < RUN / 8; k++) sh_sign[RUN / 8 * run + k] = 0;
        }
        sh_scale[run] = scale;
    }
    __syncthreads();
}

// The block scale over the runs' scales: `d` is their largest over 31 steps, stored times `lift`;
// each run's four-bit level written to `sh_level`. Zero everything when every run is.
__device__ static void rq_grid_scales(const int runs, const float lift, const float * sh_scale,
        int * sh_level, float * sh_d, half * hd_out) {
    RQ_EXACT
    float max_scale = 0.0f;
    for (int i = 0; i < runs; i++) max_scale = fmaxf(max_scale, sh_scale[i]);
    if (max_scale == 0.0f) {
        *hd_out = __float2half(0.0f);
        *sh_d = 0.0f;
        for (int i = 0; i < runs; i++) sh_level[i] = 0;
        return;
    }
    const float d = max_scale / 31.0f;
    const half hd = __float2half(RQ_MUL(d, lift));
    *hd_out = hd;
    *sh_d = __half2float(hd);
    const float inverse = 1.0f / d;
    for (int i = 0; i < runs; i++)
        sh_level[i] = req_clampi(req_nint(RQ_MUL(0.5f, RQ_SUB(RQ_MUL(inverse, sh_scale[i]), 1.0f))), 0, 15);
}

#define RQ_IQ3S_NPAT 4096
#define RQ_IQ3S_NPTS 512
#define RQ_IQ3S_BYTES 110

extern "C" __global__ void requant_iq3_s(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, const int * __restrict__ book, void * __restrict__ vy,
        float * __restrict__ got, const int has_got,
        const int nblocks, const int blocks_per_row, const int has_imp) {
    RQ_EXACT
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x;
    __shared__ float sh_x[REQ_QK_K], sh_v[REQ_QK_K], sh_w[REQ_QK_K], sh_r[REQ_QK_K];
    __shared__ float sh_sigma2, sh_d;
    __shared__ unsigned char sh_sign[32];
    __shared__ float sh_max[8];
    __shared__ unsigned char sh_trial[8 * RQ_GRID_STEPS * 32];
    __shared__ unsigned int sh_held[8 * RQ_GRID_STEPS];
    __shared__ float sh_qx[8 * RQ_GRID_STEPS], sh_q2[8 * RQ_GRID_STEPS];
    __shared__ float sh_scale[8];
    __shared__ int sh_point[64];
    __shared__ int sh_level[8];
    rq_grid_block<32, 4, 8, 3, RQ_IQ3S_NPAT, RQ_IQ3S_NPTS, true>(
        x, block_step, value_step, imp, book, sb, blocks_per_row, has_imp, 0.2f, 0.0f, 0.0f,
        sh_x, sh_v, sh_w, sh_r, &sh_sigma2, sh_sign, sh_max, sh_trial, sh_held, sh_qx, sh_q2,
        sh_scale, sh_point);
    unsigned char * y = (unsigned char *) vy + (size_t) sb * RQ_IQ3S_BYTES;
    if (t == 0) {
        half hd;
        rq_grid_scales(8, 1.033f, sh_scale, sh_level, &sh_d, &hd);
        *(half *) y = hd;
    }
    __syncthreads();
    // d, qs[64], qh[8], signs[32], scales[4]
    if (t < 64) y[2 + t] = (unsigned char) (sh_point[t] & 255);
    if (t < 8) {
        unsigned int h = 0;
        for (int k = 0; k < 8; k++) h |= (unsigned int) (sh_point[8 * t + k] >> 8) << k;
        y[66 + t] = (unsigned char) h;
    }
    if (t < 32) y[74 + t] = sh_sign[t];
    if (t < 4) y[106 + t] = (unsigned char) (sh_level[2 * t] | (sh_level[2 * t + 1] << 4));
    if (has_got && t < REQ_QK_K) {
        RQ_EXACT
        const int run = t / 32, k = (t % 32) / 4, j = t % 4;
        const float db = RQ_MUL(sh_d, (float) (1 + 2 * sh_level[run]));
        const int n = sh_point[8 * run + k];
        const float mag = RQ_MUL(db, (float) book[2 * RQ_IQ3S_NPAT + 1 + 4 * RQ_IQ3S_NPTS + 4 * n + j]);
        got[(size_t) sb * block_step + (size_t) t * value_step] = ((sh_sign[t / 8] >> (t % 8)) & 1) ? -mag : mag;
    }
}

#define RQ_IQ2S_NPAT 65536
#define RQ_IQ2S_NPTS 1024
#define RQ_IQ2S_BYTES 82

extern "C" __global__ void requant_iq2_s(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, const int * __restrict__ book, void * __restrict__ vy,
        float * __restrict__ got, const int has_got,
        const int nblocks, const int blocks_per_row, const int has_imp) {
    RQ_EXACT
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x;
    __shared__ float sh_x[REQ_QK_K], sh_v[REQ_QK_K], sh_w[REQ_QK_K], sh_r[REQ_QK_K];
    __shared__ float sh_sigma2, sh_d;
    __shared__ unsigned char sh_sign[32];
    __shared__ float sh_max[16];
    __shared__ unsigned char sh_trial[16 * RQ_GRID_STEPS * 16];
    __shared__ unsigned int sh_held[16 * RQ_GRID_STEPS];
    __shared__ float sh_qx[16 * RQ_GRID_STEPS], sh_q2[16 * RQ_GRID_STEPS];
    __shared__ float sh_scale[16];
    __shared__ int sh_point[32];
    __shared__ int sh_level[16];
    rq_grid_block<16, 8, 3, 2, RQ_IQ2S_NPAT, RQ_IQ2S_NPTS, false>(
        x, block_step, value_step, imp, book, sb, blocks_per_row, has_imp, 0.1f, 1e-8f, 0.25f,
        sh_x, sh_v, sh_w, sh_r, &sh_sigma2, sh_sign, sh_max, sh_trial, sh_held, sh_qx, sh_q2,
        sh_scale, sh_point);
    unsigned char * y = (unsigned char *) vy + (size_t) sb * RQ_IQ2S_BYTES;
    if (t == 0) {
        half hd;
        rq_grid_scales(16, 0.9875f, sh_scale, sh_level, &sh_d, &hd);
        *(half *) y = hd;
    }
    __syncthreads();
    // d, qs[32] codes, qs[32] signs, qh[8], scales[8]
    if (t < 32) {
        y[2 + t] = (unsigned char) (sh_point[t] & 255);
        y[34 + t] = sh_sign[t];
    }
    if (t < 8) {
        unsigned int h = 0;
        for (int k = 0; k < 4; k++) h |= (unsigned int) (sh_point[4 * t + k] >> 8) << (2 * k);
        y[66 + t] = (unsigned char) h;
        y[74 + t] = (unsigned char) (sh_level[2 * t] | (sh_level[2 * t + 1] << 4));
    }
    if (has_got && t < REQ_QK_K) {
        RQ_EXACT
        const int run = t / 16, g = t / 8, j = t % 8;
        const float db = RQ_MUL(RQ_MUL(sh_d, RQ_ADD(0.5f, (float) sh_level[run])), 0.25f);
        const int n = sh_point[g];
        const float mag = RQ_MUL(db, (float) book[2 * RQ_IQ2S_NPAT + 1 + 8 * RQ_IQ2S_NPTS + 8 * n + j]);
        got[(size_t) sb * block_step + (size_t) t * value_step] = ((sh_sign[g] >> j) & 1) ? -mag : mag;
    }
}

// iq4_nl and iq4_xs: four-bit codes into a sixteen-level table, a scale per run of thirty-two. The
// host encoders' search, step for step: a run's largest magnitude starts on the widest level,
// fifteen scales around it are tried with every value at its nearest level and the scale refitted
// by least squares, and the best fit wins. iq4_nl stores each run's scale as its block's f16;
// iq4_xs stores six-bit steps of one f16 scale a block of 256. One CUDA block of 256 threads per
// 256 values: thread (run, try). `imp` holds one weight per column of a row, `blocks_per_row`
// spans of 256 of it. With `has_got` the values the blocks decode to are written to `got`.
__constant__ signed char rq_iq4_levels[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ static int rq_iq4_index(const float v) {
    if (v <= (float) rq_iq4_levels[0]) return 0;
    if (v >= (float) rq_iq4_levels[15]) return 15;
    int low = 0, high = 15;
    while (high - low > 1) {
        const int mid = (low + high) / 2;
        if (v < (float) rq_iq4_levels[mid]) high = mid; else low = mid;
    }
    return RQ_SUB(v, (float) rq_iq4_levels[high - 1]) < RQ_SUB((float) rq_iq4_levels[high], v) ? high - 1 : high;
}

template <bool XS>
__device__ static void rq_iq4(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy, float * __restrict__ got,
        const int has_got, const int nblocks, const int blocks_per_row, const int has_imp) {
    RQ_EXACT
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x, run = t / 32, lane = t % 32;
    __shared__ float sh_x[REQ_QK_K], sh_w[REQ_QK_K];
    __shared__ float sh_sigma2[8], sh_max[8], sh_scale[8], sh_step[8];
    __shared__ float sh_qx[8][16], sh_q2[8][16];
    __shared__ float sh_d;
    __shared__ int sh_l[8];
    sh_x[t] = x[(size_t) sb * block_step + (size_t) t * value_step];
    __syncthreads();
    // The spread: over the whole block for iq4_xs, over each run of thirty-two for iq4_nl.
    if (XS ? t == 0 : lane == 0) {
        const int from = XS ? 0 : 32 * run, len = XS ? REQ_QK_K : 32;
        float s = 0.0f;
        for (int i = 0; i < len; i++) s = RQ_ADD(s, RQ_MUL(sh_x[from + i], sh_x[from + i]));
        s = RQ_MUL(s, 2.0f / (float) len);
        if (XS) for (int i = 0; i < 8; i++) sh_sigma2[i] = s; else sh_sigma2[run] = s;
    }
    __syncthreads();
    {
        const float v = sh_x[t];
        sh_w[t] = has_imp
            ? RQ_MUL(imp[(size_t) (sb % blocks_per_row) * REQ_QK_K + t], sqrtf(RQ_ADD(sh_sigma2[run], RQ_MUL(v, v))))
            : RQ_MUL(v, v);
    }
    if (lane == 0) {
        float amax = 0.0f, max = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float v = sh_x[32 * run + i];
            if (fabsf(v) > amax) { amax = fabsf(v); max = v; }
        }
        sh_max[run] = amax < 1e-15f ? 0.0f : max;
    }
    __syncthreads();
    const float max = sh_max[run];
    const float * xb = sh_x + 32 * run;
    const float * wb = sh_w + 32 * run;
    // Lanes 0..14 try the scales -7..7 steps from the widest level; lane 15 the widest itself.
    if (max != 0.0f && lane < 16) {
        const float inverse = lane < 15
            ? (float) (lane - 7 + rq_iq4_levels[0]) / max
            : 1.0f / ((-max) / (float) rq_iq4_levels[0]);
        float qx = 0.0f, q2 = 0.0f;
        for (int j = 0; j < 32; j++) {
            const float q = (float) rq_iq4_levels[rq_iq4_index(RQ_MUL(inverse, xb[j]))];
            qx = RQ_ADD(qx, RQ_MUL(RQ_MUL(wb[j], q), xb[j]));
            q2 = RQ_ADD(q2, RQ_MUL(RQ_MUL(wb[j], q), q));
        }
        sh_qx[run][lane] = qx;
        sh_q2[run][lane] = q2;
    }
    __syncthreads();
    if (lane == 0) {
        float d = 0.0f;
        if (max != 0.0f) {
            const float qx0 = sh_qx[run][15], q20 = sh_q2[run][15];
            d = q20 > 0.0f ? qx0 / q20 : 0.0f;
            float best = RQ_MUL(d, qx0);
            for (int s = 0; s < 15; s++) {
                const float qx = sh_qx[run][s], q2 = sh_q2[run][s];
                if (q2 > 0.0f && RQ_MUL(qx, qx) > RQ_MUL(best, q2)) {
                    d = qx / q2;
                    best = RQ_MUL(d, qx);
                }
            }
        }
        sh_scale[run] = d;
    }
    __syncthreads();
    unsigned char * y;
    if (XS) {
        // d, scales_h, scales_l[4], qs[128]
        y = (unsigned char *) vy + (size_t) sb * 136;
        if (t == 0) {
            float widest = 0.0f, max_scale = 0.0f;
            for (int i = 0; i < 8; i++) {
                if (fabsf(sh_scale[i]) > widest) { widest = fabsf(sh_scale[i]); max_scale = sh_scale[i]; }
            }
            const half hd = __float2half(max_scale == 0.0f ? 0.0f : (-max_scale) / 32.0f);
            *(half *) y = hd;
            const float d = __half2float(hd);
            sh_d = d;
            const float inverse = d != 0.0f ? 1.0f / d : 0.0f;
            unsigned int high = 0;
            for (int i = 0; i < 8; i++) {
                const int l = req_clampi(req_nint(RQ_MUL(inverse, sh_scale[i])), -32, 31);
                sh_l[i] = l;
                const float dl = RQ_MUL(d, (float) l);
                sh_step[i] = dl != 0.0f ? 1.0f / dl : 0.0f;
                high |= (unsigned int) ((l + 32) >> 4) << (2 * i);
            }
            *(unsigned short *) (y + 2) = (unsigned short) high;
            for (int i = 0; i < 4; i++)
                y[4 + i] = (unsigned char) (((sh_l[2 * i] + 32) & 15) | (((sh_l[2 * i + 1] + 32) & 15) << 4));
        }
        __syncthreads();
        if (lane < 16) {
            const float step = sh_step[run];
            y[8 + 16 * run + lane] = (unsigned char) (rq_iq4_index(RQ_MUL(step, xb[lane]))
                                                      | (rq_iq4_index(RQ_MUL(step, xb[lane + 16])) << 4));
        }
        if (has_got) {
            const float dl = RQ_MUL(sh_d, (float) sh_l[run]);
            got[(size_t) sb * block_step + (size_t) t * value_step] =
                RQ_MUL(dl, (float) rq_iq4_levels[rq_iq4_index(RQ_MUL(sh_step[run], xb[lane]))]);
        }
    } else {
        // Eight blocks of d, qs[16].
        y = (unsigned char *) vy + ((size_t) sb * 8 + run) * 18;
        __shared__ float sh_dn[8];
        if (lane == 0) {
            const half hd = __float2half(sh_scale[run]);
            *(half *) y = hd;
            const float d = __half2float(hd);
            sh_dn[run] = d;
            sh_step[run] = d != 0.0f ? 1.0f / d : 0.0f;
        }
        __syncthreads();
        const float step = sh_step[run];
        if (lane < 16) {
            y[2 + lane] = (unsigned char) (rq_iq4_index(RQ_MUL(step, xb[lane]))
                                           | (rq_iq4_index(RQ_MUL(step, xb[lane + 16])) << 4));
        }
        if (has_got) {
            got[(size_t) sb * block_step + (size_t) t * value_step] =
                RQ_MUL(sh_dn[run], (float) rq_iq4_levels[rq_iq4_index(RQ_MUL(step, xb[lane]))]);
        }
    }
}

extern "C" __global__ void requant_iq4_xs(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy, float * __restrict__ got,
        const int has_got, const int nblocks, const int blocks_per_row, const int has_imp) {
    rq_iq4<true>(x, block_step, value_step, imp, vy, got, has_got, nblocks, blocks_per_row, has_imp);
}

extern "C" __global__ void requant_iq4_nl(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy, float * __restrict__ got,
        const int has_got, const int nblocks, const int blocks_per_row, const int has_imp) {
    rq_iq4<false>(x, block_step, value_step, imp, vy, got, has_got, nblocks, blocks_per_row, has_imp);
}

// q6_K, q5_K and q4_K as the host encoders write them, with an importance and without. `imp` holds
// one weight per column of a row, `blocks_per_row` blocks of it, read when `has_imp` says it is
// there. With `has_got` the values the blocks decode to are written to `got`, in the layout `x`
// has. One CUDA block of 32 threads per block; the same blocks as the host.

// `rq_fit_signed` and then the scales a wider sweep reaches, a tenth of a level apart from half a
// level to nine tenths either side of the extreme's own, kept where they fit better.
__device__ static float rq_fit_signed_wide(int nmax, int n, const float * x, unsigned char * ls, const float * w) {
    RQ_EXACT
    float scale = rq_fit_signed(nmax, n, x, ls, w);
    float extreme = 0.0f;
    for (int i = 0; i < n; i++) if (fabsf(x[i]) > fabsf(extreme)) extreme = x[i];
    if (extreme == 0.0f) return scale;
    float wxl = 0.0f, wl2 = 0.0f;
    for (int i = 0; i < n; i++) {
        const float l = (float) ((int) ls[i] - nmax);
        wxl = RQ_ADD(wxl, RQ_MUL(RQ_MUL(w[i], x[i]), l));
        wl2 = RQ_ADD(wl2, RQ_MUL(RQ_MUL(w[i], l), l));
    }
    float best = wl2 > 0.0f ? RQ_MUL(wxl, wxl) / wl2 : 0.0f;
    const int wide[10] = {-9, -8, -7, -6, -5, 5, 6, 7, 8, 9};
    for (int k = 0; k < 10; k++) {
        const float inv = -RQ_ADD((float) nmax, RQ_MUL(0.1f, (float) wide[k])) / extreme;
        float cwl2, cwxl = rq_score(nmax, n, x, w, inv, &cwl2);
        if (cwl2 > 0.0f && RQ_MUL(cwxl, cwxl) > RQ_MUL(best, cwl2)) {
            for (int i = 0; i < n; i++) ls[i] = (unsigned char) (req_clampi(req_nint(RQ_MUL(inv, x[i])), -nmax, nmax - 1) + nmax);
            scale = cwxl / cwl2;
            best = RQ_MUL(scale, cwxl);
        }
    }
    return scale;
}

// `rms(x) + |x|` over n values: the weight of a value when no importance says.
__device__ static void rq_magnitude_weights(int n, const float * x, float * w) {
    RQ_EXACT
    float s = 0.0f;
    for (int i = 0; i < n; i++) s = RQ_ADD(s, RQ_MUL(x[i], x[i]));
    const float rms = sqrtf(s / (float) n);
    for (int i = 0; i < n; i++) w[i] = RQ_ADD(rms, fabsf(x[i]));
}

extern "C" __global__ void requant_q6_k_exact(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy, float * __restrict__ got,
        const int has_got, const int nblocks, const int blocks_per_row, const int has_imp) {
    RQ_EXACT
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x;
    __shared__ float sh_x[REQ_QK_K];
    __shared__ float sh_scale[16];
    __shared__ float sh_dg[16];
    __shared__ unsigned char sh_l[REQ_QK_K];
    for (int i = t; i < REQ_QK_K; i += 32) sh_x[i] = x[(size_t) sb * block_step + (size_t) i * value_step];
    __syncthreads();
    if (t < 16) {
        const float * xg = sh_x + 16 * t;
        float w[16];
        if (has_imp) {
            for (int i = 0; i < 16; i++) w[i] = imp[(size_t) (sb % blocks_per_row) * REQ_QK_K + 16 * t + i];
            sh_scale[t] = rq_fit_signed_wide(32, 16, xg, sh_l + 16 * t, w);
        } else {
            rq_magnitude_weights(16, xg, w);
            sh_scale[t] = rq_fit_signed(32, 16, xg, sh_l + 16 * t, w);
        }
    }
    __syncthreads();
    // ql[128], qh[64], scales[16], d
    unsigned char * y = (unsigned char *) vy + (size_t) sb * 210;
    __shared__ float sh_d;
    __shared__ bool sh_zero;
    if (t == 0) {
        float max = 0.0f;
        for (int i = 0; i < 16; i++) if (fabsf(sh_scale[i]) > fabsf(max)) max = sh_scale[i];
        sh_zero = max == 0.0f;
        if (sh_zero) {
            sh_d = 0.0f;
            for (int i = 0; i < 16; i++) sh_dg[i] = 0.0f;
        } else {
            const float iscale = -128.0f / max;
            const half hd = __float2half(1.0f / iscale);
            *(half *) (y + 208) = hd;
            sh_d = __half2float(hd);
            for (int i = 0; i < 16; i++) {
                int s = req_nint(RQ_MUL(iscale, sh_scale[i]));
                s = s > 127 ? 127 : s;
                y[192 + i] = (unsigned char) (signed char) s;
                sh_dg[i] = RQ_MUL(sh_d, (float) (signed char) s);
            }
        }
    }
    __syncthreads();
    if (sh_zero) {
        for (int i = t; i < 210; i += 32) y[i] = 0;
        if (has_got) for (int i = t; i < REQ_QK_K; i += 32) got[(size_t) sb * block_step + (size_t) i * value_step] = 0.0f;
        return;
    }
    for (int i = t; i < REQ_QK_K; i += 32) {
        const float dg = sh_dg[i / 16];
        if (dg != 0.0f) sh_l[i] = (unsigned char) (req_clampi(req_nint(sh_x[i] / dg), -32, 31) + 32);
    }
    __syncthreads();
    for (int half_ = 0; half_ < 2; half_++) {
        const unsigned char * h = sh_l + 128 * half_;
        const int i = t;
        const int a = h[i], b = h[i + 32], c = h[i + 64], e = h[i + 96];
        y[64 * half_ + i] = (unsigned char) ((a & 15) | ((c & 15) << 4));
        y[64 * half_ + i + 32] = (unsigned char) ((b & 15) | ((e & 15) << 4));
        y[128 + 32 * half_ + i] = (unsigned char) ((a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((e >> 4) << 6));
    }
    if (has_got) {
        for (int i = t; i < REQ_QK_K; i += 32) {
            const signed char s = (signed char) y[192 + i / 16];
            got[(size_t) sb * block_step + (size_t) i * value_step] =
                RQ_MUL(RQ_MUL(sh_d, (float) s), (float) ((int) sh_l[i] - 32));
        }
    }
}

// The host's `fit_scales_and_offsets` for one group of n values: levels 0..nmax under
// `scale * l + offset`, offset at most zero, the grid of scales walked from nmax - 0.9 to nmax + 0.9
// in steps of 0.05. Returns the scale; the negated offset into `below`.
__device__ static float rq_fit_scale_offset(int nmax, int n, const float * x, const float * w, float * below) {
    RQ_EXACT
    const float top_level = (float) nmax;
    float offset = x[0], top = x[0], w_total = w[0], wx_total = RQ_MUL(w[0], x[0]);
    for (int i = 1; i < n; i++) {
        if (x[i] < offset) offset = x[i];
        if (x[i] > top) top = x[i];
        w_total = RQ_ADD(w_total, w[i]);
        wx_total = RQ_ADD(wx_total, RQ_MUL(w[i], x[i]));
    }
    if (offset > 0.0f) offset = 0.0f;
    const bool flat = top <= offset;
    float per_unit = top_level / RQ_SUB(top, offset);
    float scale = 1.0f / per_unit;
    float best_error = 0.0f;
    for (int i = 0; i < n; i++) {
        const float l = (float) req_clampi(req_nint(RQ_MUL(per_unit, RQ_SUB(x[i], offset))), 0, nmax);
        const float diff = RQ_SUB(RQ_ADD(RQ_MUL(scale, l), offset), x[i]);
        best_error = RQ_ADD(best_error, RQ_MUL(RQ_MUL(w[i], diff), diff));
    }
    float levels[32];
    for (int step = 0; step <= 36; step++) {
        per_unit = RQ_ADD(RQ_ADD(-0.9f, RQ_MUL(0.05f, (float) step)), top_level) / RQ_SUB(top, offset);
        float wl = 0.0f, wl2 = 0.0f, wxl = 0.0f;
        for (int i = 0; i < n; i++) {
            const float l = (float) req_clampi(req_nint(RQ_MUL(per_unit, RQ_SUB(x[i], offset))), 0, nmax);
            levels[i] = l;
            wl = RQ_ADD(wl, RQ_MUL(w[i], l));
            wl2 = RQ_ADD(wl2, RQ_MUL(w[i], RQ_MUL(l, l)));
            wxl = RQ_ADD(wxl, RQ_MUL(RQ_MUL(w[i], l), x[i]));
        }
        const float denom = RQ_SUB(RQ_MUL(w_total, wl2), RQ_MUL(wl, wl));
        const bool fits = denom > 0.0f;
        float cs = RQ_SUB(RQ_MUL(w_total, wxl), RQ_MUL(wx_total, wl)) / denom;
        float co = RQ_SUB(RQ_MUL(wl2, wx_total), RQ_MUL(wl, wxl)) / denom;
        if (co > 0.0f) { co = 0.0f; cs = wxl / wl2; }
        float error = 0.0f;
        for (int i = 0; i < n; i++) {
            const float diff = RQ_SUB(RQ_ADD(RQ_MUL(cs, levels[i]), co), x[i]);
            error = RQ_ADD(error, RQ_MUL(RQ_MUL(w[i], diff), diff));
        }
        if (fits && !flat && error < best_error) { best_error = error; scale = cs; offset = co; }
    }
    *below = -offset;
    return flat ? 0.0f : scale;
}

// The host's `fit_unsigned_scale`: non-negative `x` to codes 0..nmax with no offset, against
// weights `w`; the codes into `l`, the scale returned.
__device__ static float rq_fit_unsigned(int nmax, int n, const float * x, unsigned char * l, const float * w) {
    RQ_EXACT
    float max = 0.0f;
    for (int i = 0; i < n; i++) max = fmaxf(max, x[i]);
    if (max == 0.0f) {
        for (int i = 0; i < n; i++) l[i] = 0;
        return 0.0f;
    }
#define RQ_CODE(inv, v) min((int) (unsigned char) req_nint(RQ_MUL((inv), (v))), nmax)
    float inv = (float) nmax / max;
    float best;
    {
        const float scale = 1.0f / inv;
        float e = -0.0f;
        for (int i = 0; i < n; i++) {
            const float diff = RQ_SUB(x[i], RQ_MUL(scale, (float) RQ_CODE(inv, x[i])));
            e = RQ_ADD(e, RQ_MUL(RQ_MUL(w[i], diff), diff));
        }
        best = e;
    }
    for (int s = -4; s <= 4; s++) {
        if (s == 0) continue;
        const float candidate = RQ_ADD(RQ_MUL(0.1f, (float) s), (float) nmax) / max;
        const float scale = 1.0f / candidate;
        float e = -0.0f;
        for (int i = 0; i < n; i++) {
            const float diff = RQ_SUB(x[i], RQ_MUL(scale, (float) RQ_CODE(candidate, x[i])));
            e = RQ_ADD(e, RQ_MUL(RQ_MUL(w[i], diff), diff));
        }
        if (e < best) { best = e; inv = candidate; }
    }
    float wxl = 0.0f, wl2 = 0.0f;
    for (int i = 0; i < n; i++) {
        const int c = RQ_CODE(inv, x[i]);
        l[i] = (unsigned char) c;
        wxl = RQ_ADD(wxl, RQ_MUL(RQ_MUL(w[i], x[i]), (float) c));
        wl2 = RQ_ADD(wl2, RQ_MUL(RQ_MUL(w[i], (float) c), (float) c));
    }
#undef RQ_CODE
    for (int pass = 0; pass < 5; pass++) {
        int moved = 0;
        for (int i = 0; i < n; i++) {
            const float v = x[i], ww = w[i], c = (float) l[i];
            float nwxl = RQ_SUB(wxl, RQ_MUL(RQ_MUL(ww, v), c));
            float nwl2 = RQ_SUB(wl2, RQ_MUL(RQ_MUL(ww, c), c));
            if (nwxl <= 0.0f || nwl2 <= 0.0f) continue;
            const int to = min((int) (unsigned char) req_nint(RQ_MUL(v, nwl2) / nwxl), nmax);
            if (to == (int) l[i]) continue;
            nwxl = RQ_ADD(nwxl, RQ_MUL(RQ_MUL(ww, v), (float) to));
            nwl2 = RQ_ADD(nwl2, RQ_MUL(RQ_MUL(ww, (float) to), (float) to));
            if (RQ_MUL(RQ_MUL(nwxl, nwxl), wl2) > RQ_MUL(RQ_MUL(wxl, wxl), nwl2)) {
                l[i] = (unsigned char) to;
                wxl = nwxl; wl2 = nwl2;
                moved++;
            }
        }
        if (moved == 0) break;
    }
    return wxl / wl2;
}

template <bool Q5>
__device__ static void rq_qk_exact(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy, float * __restrict__ got,
        const int has_got, const int nblocks, const int blocks_per_row, const int has_imp) {
    RQ_EXACT
    const int sb = blockIdx.x;
    if (sb >= nblocks) return;
    const int t = threadIdx.x;
    const int NMAX = Q5 ? 31 : 15;
    const int BYTES = Q5 ? 176 : 144;
    __shared__ float sh_x[REQ_QK_K], sh_w[REQ_QK_K];
    __shared__ float sh_sigma2;
    __shared__ float sh_scale[8], sh_min[8], sh_carried[8], sh_dg[8], sh_mg[8];
    for (int i = t; i < REQ_QK_K; i += 32) sh_x[i] = x[(size_t) sb * block_step + (size_t) i * value_step];
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < REQ_QK_K; i++) s = RQ_ADD(s, RQ_MUL(sh_x[i], sh_x[i]));
        sh_sigma2 = RQ_MUL(2.0f, s) / (float) REQ_QK_K;
    }
    __syncthreads();
    if (t < 8) {
        const float * xg = sh_x + 32 * t;
        float * wg = sh_w + 32 * t;
        if (has_imp) {
            for (int i = 0; i < 32; i++)
                wg[i] = RQ_MUL(imp[(size_t) (sb % blocks_per_row) * REQ_QK_K + 32 * t + i],
                               sqrtf(RQ_ADD(sh_sigma2, RQ_MUL(xg[i], xg[i]))));
        } else {
            rq_magnitude_weights(32, xg, wg);
        }
        float c = -0.0f;
        for (int i = 0; i < 32; i++) c = RQ_ADD(c, wg[i]);
        sh_carried[t] = c;
        float below;
        sh_scale[t] = rq_fit_scale_offset(NMAX, 32, xg, wg, &below);
        sh_min[t] = below;
    }
    __syncthreads();
    unsigned char * y = (unsigned char *) vy + (size_t) sb * BYTES;
    if (t == 0) {
        unsigned char ls[8], lm[8];
        float d, dmin;
        if (has_imp) {
            d = rq_fit_unsigned(63, 8, sh_scale, ls, sh_carried);
            dmin = rq_fit_unsigned(63, 8, sh_min, lm, sh_carried);
        } else {
            float max_scale = 0.0f, max_min = 0.0f;
            for (int g = 0; g < 8; g++) { max_scale = fmaxf(sh_scale[g], max_scale); max_min = fmaxf(sh_min[g], max_min); }
            const float is = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
            const float im = max_min > 0.0f ? 63.0f / max_min : 0.0f;
            for (int g = 0; g < 8; g++) {
                ls[g] = (unsigned char) min(req_nint(RQ_MUL(is, sh_scale[g])), 63);
                lm[g] = (unsigned char) min(req_nint(RQ_MUL(im, sh_min[g])), 63);
            }
            d = max_scale / 63.0f;
            dmin = max_min / 63.0f;
        }
        const half hd = __float2half(d), hm = __float2half(dmin);
        *(half *) y = hd;
        *(half *) (y + 2) = hm;
        unsigned char p[12];
        for (int i = 0; i < 4; i++) { p[i] = ls[i] & 63; p[i + 4] = lm[i] & 63; }
        for (int i = 4; i < 8; i++) {
            p[i + 4] = (unsigned char) ((ls[i] & 15) | ((lm[i] & 15) << 4));
            p[i - 4] |= (unsigned char) ((ls[i] >> 4) << 6);
            p[i] |= (unsigned char) ((lm[i] >> 4) << 6);
        }
        for (int i = 0; i < 12; i++) y[4 + i] = p[i];
        const float du = __half2float(hd), mu = __half2float(hm);
        for (int g = 0; g < 8; g++) {
            sh_dg[g] = RQ_MUL(du, (float) (ls[g] & 63));
            sh_mg[g] = RQ_MUL(mu, (float) (lm[g] & 63));
        }
    }
    __syncthreads();
    // Thread t takes value t of each group: its byte of each pair of groups, and for q5_K its qh.
    unsigned char code[8];
    for (int g = 0; g < 8; g++) {
        const float dg = sh_dg[g];
        code[g] = dg == 0.0f ? 0
            : (unsigned char) req_clampi(req_nint(RQ_ADD(sh_x[32 * g + t], sh_mg[g]) / dg), 0, NMAX);
    }
    const int qs = Q5 ? 48 : 16;
    for (int p = 0; p < 4; p++)
        y[qs + 32 * p + t] = (unsigned char) ((code[2 * p] & 15) | ((code[2 * p + 1] & 15) << 4));
    if (Q5) {
        unsigned char h = 0;
        for (int g = 0; g < 8; g++) h |= (unsigned char) ((code[g] >> 4) << g);
        y[16 + t] = h;
    }
    if (has_got) {
        for (int g = 0; g < 8; g++)
            got[(size_t) sb * block_step + (size_t) (32 * g + t) * value_step] =
                RQ_SUB(RQ_MUL(sh_dg[g], (float) code[g]), sh_mg[g]);
    }
}

extern "C" __global__ void requant_q4_k_exact(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy, float * __restrict__ got,
        const int has_got, const int nblocks, const int blocks_per_row, const int has_imp) {
    rq_qk_exact<false>(x, block_step, value_step, imp, vy, got, has_got, nblocks, blocks_per_row, has_imp);
}

extern "C" __global__ void requant_q5_k_exact(
        const float * __restrict__ x, const long block_step, const long value_step,
        const float * __restrict__ imp, void * __restrict__ vy, float * __restrict__ got,
        const int has_got, const int nblocks, const int blocks_per_row, const int has_imp) {
    rq_qk_exact<true>(x, block_step, value_step, imp, vy, got, has_got, nblocks, blocks_per_row, has_imp);
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

// h[i] = swiglu(g[i], u[i]) over a layer's gate and up products, one thread to a value.
extern "C" __global__ void swiglu_limit_f32(
        const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ h,
        const long n, const float limit) {
    const long i = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (i < n) h[i] = cx_swiglu(g[i], u[i], limit);
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

// One 256-value block at `xb` as a q8_K block at `ob`, the host's activation quantiser value for
// value: the extreme is the first value of the largest magnitude, the codes are its -127th parts
// rounded half away from zero and clipped at 127, the scale is the reciprocal, and each sixteen
// codes are summed. A warp per block, eight consecutive values to a lane.
__device__ static void cx_q8k_block(const float * __restrict__ x, unsigned char * __restrict__ ob, const int lane) {
    const float * xb = x + lane * 8;
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

// Rows of `x` [rows, k] as q8_K blocks (see cx_q8k_block). One warp per block.
extern "C" __global__ void cx_quantize_q8_k(
        const float * __restrict__ x, unsigned char * __restrict__ out, const long nblocks) {
    const long blk = blockIdx.x;
    if (blk >= nblocks) return;
    cx_q8k_block(x + blk * 256, out + blk * CX_Q8K_BYTES, threadIdx.x);
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

// cx_expert_h_iq2_xxs with the terms staged a pass of the warp's lanes at a time.
// A pass takes 32 sub-blocks: 4 whole blocks of 8.
// Each instance's 8 terms of a block are added in order before the next pass.
// Shared memory holds each instance's terms of a pass and its block shares:
// (2 * tt * 32 + 2 * tt * (dim / 256)) floats.
extern "C" __global__ void cx_expert_h_iq2_xxs_lanes(
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
    float * tu = tg + tt * 32;
    float * sg = tu + tt * 32;
    float * su = sg + tt * sbs;
    const unsigned char * gate0 = (const unsigned char *) gate_ptrs[e] + (long) row * sbs * CX_IQ2XXS_BYTES;
    const unsigned char * up0 = (const unsigned char *) up_ptrs[e] + (long) row * sbs * CX_IQ2XXS_BYTES;
    for (int u0 = 0; u0 < nsub; u0 += 32) {
        const int u = u0 + lane;
        if (u < nsub) {
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
                tg[j * 32 + lane] = __fmul_rn(__fmul_rn(scg, (float) ig), yd);
                tu[j * 32 + lane] = __fmul_rn(__fmul_rn(scu, (float) iu), yd);
            }
        }
        __syncwarp();
        const int blocks = min(4, sbs - u0 / 8);
        for (int p = lane; p < c * blocks; p += 32) {
            const int j = p / blocks, b = p % blocks, ib = u0 / 8 + b;
            float ag = 0.0f, au = 0.0f;
            #pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                ag = __fadd_rn(ag, tg[j * 32 + b * 8 + sub]);
                au = __fadd_rn(au, tu[j * 32 + b * 8 + sub]);
            }
            sg[j * sbs + ib] = __fmul_rn(req_f16_at(gate0 + (long) ib * CX_IQ2XXS_BYTES), ag);
            su[j * sbs + ib] = __fmul_rn(req_f16_at(up0 + (long) ib * CX_IQ2XXS_BYTES), au);
        }
        __syncwarp();
    }
    if (lane < c) {
        float g = -0.0f, u = -0.0f;
        for (int ib = 0; ib < sbs; ++ib) {
            g = __fadd_rn(g, sg[lane * sbs + ib]);
            u = __fadd_rn(u, su[lane * sbs + ib]);
        }
        h[(long) (i0 + lane) * inter + row] = cx_swiglu(g, u, limit);
    }
}

// cx_expert_down_q2_k with the sums staged a pass of the warp's lanes at a time.
// A pass takes 32 sub-blocks: 2 whole blocks of 16.
// Shared memory holds each instance's sums of a pass and its block shares:
// (2 * tt * 32) ints and (tt * (inter / 256)) floats.
extern "C" __global__ void cx_expert_down_q2_k_lanes(
        const unsigned long long * __restrict__ down_ptrs,
        const unsigned char * __restrict__ hq, const int * __restrict__ tiles,
        float * __restrict__ y, const int inter, const int dim, const int tt) {
    extern __shared__ int cx_shared_i[];
    const int row = blockIdx.x, tile = blockIdx.y, lane = threadIdx.x;
    const int e = tiles[3 * tile], i0 = tiles[3 * tile + 1], c = tiles[3 * tile + 2];
    const int sbs = inter / 256, nsub = sbs * 16;
    int * isc = cx_shared_i;
    int * ioff = isc + tt * 32;
    float * st = reinterpret_cast<float *>(ioff + tt * 32);
    const unsigned char * row0 = (const unsigned char *) down_ptrs[e] + (long) row * sbs * CX_Q2K_BYTES;
    for (int u0 = 0; u0 < nsub; u0 += 32) {
        const int u = u0 + lane;
        if (u < nsub) {
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
                isc[j * 32 + lane] = sc * dot;
                ioff[j * 32 + lane] = mn * (int) reinterpret_cast<const short *>(hb + 260)[s];
            }
        }
        __syncwarp();
        const int blocks = min(2, sbs - u0 / 16);
        for (int p = lane; p < c * blocks; p += 32) {
            const int j = p / blocks, bk = p % blocks, ib = u0 / 16 + bk;
            int scaled = 0, offset = 0;
            #pragma unroll
            for (int s = 0; s < 16; ++s) {
                scaled += isc[j * 32 + bk * 16 + s];
                offset += ioff[j * 32 + bk * 16 + s];
            }
            const unsigned char * b = row0 + (long) ib * CX_Q2K_BYTES;
            const unsigned char * hb = hq + ((long) (i0 + j) * sbs + ib) * CX_Q8K_BYTES;
            const float yd = *reinterpret_cast<const float *>(hb);
            st[j * sbs + ib] = __fmul_rn(yd, __fsub_rn(__fmul_rn(req_f16_at(b + 80), (float) scaled),
                                                      __fmul_rn(req_f16_at(b + 82), (float) offset)));
        }
        __syncwarp();
    }
    if (lane < c) {
        float acc = -0.0f;
        for (int ib = 0; ib < sbs; ++ib) acc = __fadd_rn(acc, st[lane * sbs + ib]);
        y[(long) (i0 + lane) * dim + row] = acc;
    }
}

// Routed experts with IQ3_XXS gate and up and a Q3_K down, computed as the host's 256-bit grid and
// superblock kernels compute them. Those keep eight float accumulators, lane l summing the
// products of the values whose place in each run of thirty-two is 4l to 4l + 3: per run (IQ3_XXS)
// or per block (Q3_K) the lane's integer sum, converted and fused into the lane's accumulator
// under the run's or block's float scale, from +0 in run order, then one horizontal sum. Here a
// thread is one such lane of one instance, eight threads an instance: the integer sums are the
// host's exactly, each fused step is one __fmaf_rn, and the horizontal sum is taken in the host's
// order. So the products are the host's bits, as the IQ2_XXS kernels above give theirs. A warp
// holds four output rows, eight threads each, so a decode's single instance keeps every thread busy.

#define CX_IQ3XXS_BYTES 98
#define CX_Q3K_BYTES 110

#define CX_ROWS_PER_WARP 4
// The most instances of one expert a launch's tile holds (EXACT_TILE): each thread keeps an
// accumulator for each, so a weight lane is decoded once for all of them.
#define CX_TILE 8

// The eight lanes of a group of eight threads added in the order of the host's horizontal sum:
// lane i with lane i + 4, then the first with the third and the second with the fourth, then the
// two. Lane 0 of the group holds the sum; every thread of the warp must take part.
__device__ static float cx_lanes_sum8(float v) {
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 4, 8));
    v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 2, 8));
    return __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, 1, 8));
}

// Lane `l` of run `r` of the IQ3_XXS block at `b`: its four signed weights packed as bytes, and the
// run's odd integer scale.
__device__ static int cx_iq3xxs_lane(
        const unsigned char * b, const int r, const int l, const unsigned char * grid,
        const unsigned char * signs, int * odd) {
    const unsigned char * qs = b + 2;
    const unsigned char * t = qs + 64 + 4 * r;
    const unsigned int word = (unsigned int) t[0] | ((unsigned int) t[1] << 8) | ((unsigned int) t[2] << 16) | ((unsigned int) t[3] << 24);
    const unsigned char * mag = grid + 4 * (int) qs[8 * r + l];
    const unsigned char pattern = signs[(word >> (7 * (l >> 1))) & 127];
    const int shift = 4 * (l & 1);
    int packed = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int v = ((pattern >> (shift + k)) & 1) ? -(int) mag[k] : (int) mag[k];
        packed |= (v & 0xff) << (8 * k);
    }
    *odd = 2 * (int) (word >> 28) + 1;
    return packed;
}

// The rows entering the down projections: instance i is expert tiles[3t] over row xrow[i] of `xq`
// (q8_K32 rows of `dim`), written to row i of `h` [instances, inter]. A warp per (four output
// rows, tile). No shared memory.
extern "C" __global__ void cx_expert_h_iq3_xxs(
        const unsigned long long * __restrict__ gate_ptrs,
        const unsigned long long * __restrict__ up_ptrs,
        const unsigned char * __restrict__ grid, const unsigned char * __restrict__ signs,
        const unsigned char * __restrict__ xq, const int * __restrict__ xrow,
        const int * __restrict__ tiles, float * __restrict__ h,
        const int dim, const int inter, const float limit, const int tt) {
    const int tile = blockIdx.y, lane = threadIdx.x;
    const int e = tiles[3 * tile], i0 = tiles[3 * tile + 1], c = tiles[3 * tile + 2];
    const int sbs = dim / 256, l = lane & 7;
    const int want = blockIdx.x * CX_ROWS_PER_WARP + (lane >> 3);
    const bool live = want < inter;
    const int row = live ? want : 0;
    const unsigned char * gate0 = (const unsigned char *) gate_ptrs[e] + (long) row * sbs * CX_IQ3XXS_BYTES;
    const unsigned char * up0 = (const unsigned char *) up_ptrs[e] + (long) row * sbs * CX_IQ3XXS_BYTES;
    long xr[CX_TILE];
    float tg[CX_TILE], tu[CX_TILE];
    #pragma unroll
    for (int j = 0; j < CX_TILE; ++j) {
        xr[j] = j < c ? xrow[i0 + j] : 0;
        tg[j] = 0.0f;
        tu[j] = 0.0f;
    }
    for (int ib = 0; ib < sbs; ++ib) {
        const unsigned char * gb = gate0 + (long) ib * CX_IQ3XXS_BYTES;
        const unsigned char * ub = up0 + (long) ib * CX_IQ3XXS_BYTES;
        const float unit_g = __fmul_rn(req_f16_at(gb), 0.25f);
        const float unit_u = __fmul_rn(req_f16_at(ub), 0.25f);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int og, ou;
            const int wg = cx_iq3xxs_lane(gb, r, l, grid, signs, &og);
            const int wu = cx_iq3xxs_lane(ub, r, l, grid, signs, &ou);
            #pragma unroll
            for (int j = 0; j < CX_TILE; ++j) {
                if (j < c) {
                    const unsigned char * xb = xq + (xr[j] * sbs + ib) * CX_Q8K32_BYTES;
                    const float yd = req_f16_at(xb + 2 * r);
                    const int a = *reinterpret_cast<const int *>(xb + 16 + 32 * r + 4 * l);
                    tg[j] = __fmaf_rn((float) (og * __dp4a(wg, a, 0)), __fmul_rn(unit_g, yd), tg[j]);
                    tu[j] = __fmaf_rn((float) (ou * __dp4a(wu, a, 0)), __fmul_rn(unit_u, yd), tu[j]);
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < CX_TILE; ++j) {
        if (j < c) {
            const float g = cx_lanes_sum8(tg[j]);
            const float u = cx_lanes_sum8(tu[j]);
            if (live && l == 0) h[(long) (i0 + j) * inter + row] = cx_swiglu(g, u, limit);
        }
    }
}

// Sixteen-value sub-block `s` of a Q3_K block's packed scales, signed.
__device__ static int cx_q3k_scale(const unsigned char * sc, const int s) {
    const int group = s >> 2, j = s & 3;
    const int byte = sc[j + 4 * (group & 1)];
    const int low = group < 2 ? (byte & 0x0F) : (byte >> 4);
    const int high = (sc[8 + j] >> (2 * group)) & 0x03;
    return (low | (high << 4)) - 32;
}

// The outputs of the down projections: instance i is expert tiles[3t] over row i of `hq` (q8_K
// rows of `inter`), written to row i of `y` [instances, dim]. A warp per (four output rows,
// tile). No shared memory.
extern "C" __global__ void cx_expert_down_q3_k(
        const unsigned long long * __restrict__ down_ptrs,
        const unsigned char * __restrict__ hq, const int * __restrict__ tiles,
        float * __restrict__ y, const int inter, const int dim, const int tt) {
    const int tile = blockIdx.y, lane = threadIdx.x;
    const int e = tiles[3 * tile], i0 = tiles[3 * tile + 1], c = tiles[3 * tile + 2];
    const int sbs = inter / 256, l = lane & 7;
    const int want = blockIdx.x * CX_ROWS_PER_WARP + (lane >> 3);
    const bool live = want < dim;
    const int row = live ? want : 0;
    const unsigned char * row0 = (const unsigned char *) down_ptrs[e] + (long) row * sbs * CX_Q3K_BYTES;
    float acc[CX_TILE];
    #pragma unroll
    for (int j = 0; j < CX_TILE; ++j) acc[j] = 0.0f;
    for (int ib = 0; ib < sbs; ++ib) {
        const unsigned char * b = row0 + (long) ib * CX_Q3K_BYTES;
        const float xd = req_f16_at(b + 108);
        int total[CX_TILE];
        #pragma unroll
        for (int j = 0; j < CX_TILE; ++j) total[j] = 0;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const int chunk = m >> 2, ln = m & 3;
            const int sc = cx_q3k_scale(b + 96, 2 * m + (l >> 2));
            int packed = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int at = 4 * l + k;
                const int low = (b[32 + 32 * chunk + at] >> (2 * ln)) & 0x03;
                const int third = ((b[at] >> (4 * chunk + ln)) & 1) ? 0 : 4;
                packed |= ((low - third) & 0xff) << (8 * k);
            }
            #pragma unroll
            for (int j = 0; j < CX_TILE; ++j) {
                if (j < c) {
                    const unsigned char * hb = hq + ((long) (i0 + j) * sbs + ib) * CX_Q8K_BYTES;
                    const int a = *reinterpret_cast<const int *>(hb + 4 + 32 * m + 4 * l);
                    total[j] += sc * __dp4a(packed, a, 0);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < CX_TILE; ++j) {
            if (j < c) {
                const unsigned char * hb = hq + ((long) (i0 + j) * sbs + ib) * CX_Q8K_BYTES;
                const float d = __fmul_rn(*reinterpret_cast<const float *>(hb), xd);
                acc[j] = __fmaf_rn(d, (float) total[j], acc[j]);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < CX_TILE; ++j) {
        if (j < c) {
            const float v = cx_lanes_sum8(acc[j]);
            if (live && l == 0) y[(long) (i0 + j) * dim + row] = v;
        }
    }
}

// The routed experts of a cached layer (generic_transformer::expert_tier), computed as the host
// computes them for the formats of quant_cpu::ExactDot: the gate and up products against the q8_K
// row of the input, the SwiGLU of offload::experts, the row entering the down projection as q8_K
// blocks, and the down product against it times the pair's weight. A 256-value block's integer
// sums are exact in any order; the step folding them and the sum over blocks are the host's,
// through the rounding intrinsics, which the compiler never contracts. So a pair's row is the
// same bits whether its expert sits in a slot of the cache, in the transit room or on the host.
//
// A launch runs instances, one routed pair each: slots[i] is where the pair's expert sits, all
// ones for an instance the launch skips; the pair is pairs[i], or i when the list is null, and
// its input row is pair / k. A slot below `base_slots` is in the stacks at `base`, a slot past
// them in the stacks whose addresses the card reads at `extra` (gate, up, down).
#define TX_SKIP 0xFFFFFFFFu
#define TX_Q2K 0
#define TX_Q4K 1
#define TX_Q6K 2
#define TX_IQ2S 3
#define TX_IQ3XXS 4
#define TX_IQ3S 5
#define TX_IQ4NL 6
#define TX_Q2_0 7
#define TX_Q8_0 8
#define TX_Q5_0 9

// A format's bytes per 256 values, and whether its dot reads q8_K32 blocks and folds a block run
// by run (RUNS) or reads q8_K blocks and folds a block in 1 step; ACT is the activation block's
// bytes.
template <int F> struct TxFmt;
template <> struct TxFmt<TX_Q2K> { static constexpr int BYTES = 84; static constexpr bool RUNS = false; static constexpr int ACT = CX_Q8K_BYTES; };
template <> struct TxFmt<TX_Q4K> { static constexpr int BYTES = 144; static constexpr bool RUNS = false; static constexpr int ACT = CX_Q8K_BYTES; };
template <> struct TxFmt<TX_Q6K> { static constexpr int BYTES = 210; static constexpr bool RUNS = false; static constexpr int ACT = CX_Q8K_BYTES; };
template <> struct TxFmt<TX_IQ2S> { static constexpr int BYTES = 82; static constexpr bool RUNS = true; static constexpr int ACT = CX_Q8K32_BYTES; };
template <> struct TxFmt<TX_Q5_0> { static constexpr int BYTES = 176; static constexpr bool RUNS = true; static constexpr int ACT = CX_Q8K32_BYTES; };
template <> struct TxFmt<TX_Q8_0> { static constexpr int BYTES = 272; static constexpr bool RUNS = true; static constexpr int ACT = CX_Q8K32_BYTES; };
template <> struct TxFmt<TX_Q2_0> { static constexpr int BYTES = 72; static constexpr bool RUNS = true; static constexpr int ACT = CX_Q8K32_BYTES; };
template <> struct TxFmt<TX_IQ4NL> { static constexpr int BYTES = 144; static constexpr bool RUNS = true; static constexpr int ACT = CX_Q8K32_BYTES; };
template <> struct TxFmt<TX_IQ3S> { static constexpr int BYTES = 110; static constexpr bool RUNS = true; static constexpr int ACT = CX_Q8K32_BYTES; };
template <> struct TxFmt<TX_IQ3XXS> { static constexpr int BYTES = 98; static constexpr bool RUNS = true; static constexpr int ACT = CX_Q8K32_BYTES; };

// The f16 at `p`, whatever its alignment, widened to f32: exact, as every f16 is an f32.
__device__ static __forceinline__ float tx_f16(const unsigned char * p) {
    return __half2float(__ushort_as_half((unsigned short) (p[0] | (p[1] << 8))));
}

// Four bytes from `p`, 2-byte aligned, the first in the low byte.
__device__ static __forceinline__ unsigned int tx_u32_2(const unsigned char * p) {
    const unsigned short * h = reinterpret_cast<const unsigned short *>(p);
    return (unsigned int) h[0] | ((unsigned int) h[1] << 16);
}

// `n` words of 4 bytes from `p`, 2-byte aligned, read as `n + 1` aligned words shifted into place.
template <int N> __device__ static __forceinline__ void tx_words(const unsigned char * p, unsigned int * out) {
    const unsigned int shift = 8u * (unsigned int) (reinterpret_cast<unsigned long long>(p) & 3u);
    const unsigned int * a = reinterpret_cast<const unsigned int *>(p - (reinterpret_cast<unsigned long long>(p) & 3u));
    unsigned int prev = a[0];
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        const unsigned int next = a[k + 1];
        out[k] = __funnelshift_r(prev, next, shift);
        prev = next;
    }
}

// The 4 bytes of `x`, each 0..15, looked up in the 16 signed levels `t` held as 4 words.
__device__ static __forceinline__ int tx_lookup16(const unsigned int x, const unsigned int * t) {
    const unsigned int low3 = x & 0x07070707u;
    const unsigned int sel = (low3 & 0x7u) | ((low3 >> 4) & 0x70u) | ((low3 >> 8) & 0x700u) | ((low3 >> 12) & 0x7000u);
    const unsigned int lo = __byte_perm(t[0], t[1], sel);
    const unsigned int hi = __byte_perm(t[2], t[3], sel);
    const unsigned int m = ((x & 0x08080808u) >> 3) * 0xFFu;
    return (int) ((lo & ~m) | (hi & m));
}

// Four bytes from `p`, whatever its alignment, the first in the low byte.
__device__ static __forceinline__ unsigned int tx_u32(const unsigned char * p) {
    return (unsigned int) p[0] | ((unsigned int) p[1] << 8) | ((unsigned int) p[2] << 16) |
           ((unsigned int) p[3] << 24);
}

// A lane's part of a weight block against a q8_K block `a`, the lane's eight values being
// 8 * lane..8 * lane + 8: the codes against the activations under their sub-block's scale, and
// the activation sums under each sub-block's minimum (one lane a sub-block adds those).
template <int F> __device__ static void tx_lane_sums(
        const unsigned char * w, const unsigned char * a, const int lane, int * scaled, int * offset);

template <> __device__ void tx_lane_sums<TX_Q4K>(
        const unsigned char * w, const unsigned char * a, const int lane, int * scaled, int * offset) {
    const int s = lane >> 2, l0 = (lane & 3) * 8, shift = 4 * (s & 1);
    const unsigned char * q = w + 16 + 32 * (s >> 1) + l0;
    const int w0 = (int) ((*reinterpret_cast<const unsigned int *>(q) >> shift) & 0x0F0F0F0Fu);
    const int w1 = (int) ((*reinterpret_cast<const unsigned int *>(q + 4) >> shift) & 0x0F0F0F0Fu);
    const int * ap = reinterpret_cast<const int *>(a + 4 + 8 * lane);
    const int dot = __dp4a(w1, ap[1], __dp4a(w0, ap[0], 0));
    const unsigned char * sc = w + 4;
    int scale, mn;
    if (s < 4) {
        scale = sc[s] & 63;
        mn = sc[s + 4] & 63;
    } else {
        scale = (sc[s + 4] & 0x0F) | ((sc[s - 4] >> 6) << 4);
        mn = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4);
    }
    *scaled = scale * dot;
    const short * bsums = reinterpret_cast<const short *>(a + 260);
    *offset = (lane & 3) == 0 ? mn * ((int) bsums[2 * s] + (int) bsums[2 * s + 1]) : 0;
}

template <> __device__ void tx_lane_sums<TX_Q6K>(
        const unsigned char * w, const unsigned char * a, const int lane, int * scaled, int * offset) {
    const int v0 = 8 * lane, half = v0 >> 7, g = (v0 & 127) >> 5, l0 = v0 & 31;
    const unsigned char * ql = w + 64 * half + 32 * (g & 1) + l0;
    const unsigned char * qh = w + 128 + 32 * half + l0;
    const int nib = 4 * (g >> 1), hs = 2 * g;
    int packed[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int p = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 4 * k + j;
            const int q = (int) (((ql[t] >> nib) & 0x0F) | (((qh[t] >> hs) & 0x03) << 4)) - 32;
            p |= (q & 0xff) << (8 * j);
        }
        packed[k] = p;
    }
    const int * ap = reinterpret_cast<const int *>(a + 4 + v0);
    const int dot = __dp4a(packed[1], ap[1], __dp4a(packed[0], ap[0], 0));
    *scaled = (int) reinterpret_cast<const signed char *>(w + 192)[v0 >> 4] * dot;
    *offset = 0;
}

template <> __device__ void tx_lane_sums<TX_Q2K>(
        const unsigned char * w, const unsigned char * a, const int lane, int * scaled, int * offset) {
    const int s = lane >> 1, l0 = (lane & 1) * 8, t = s & 7;
    const int base = 32 * (s >> 3) + 16 * (t & 1), shift = 2 * (t >> 1);
    const unsigned char * q = w + 16 + base + l0;
    const int w0 = (int) ((*reinterpret_cast<const unsigned int *>(q) >> shift) & 0x03030303u);
    const int w1 = (int) ((*reinterpret_cast<const unsigned int *>(q + 4) >> shift) & 0x03030303u);
    const int * ap = reinterpret_cast<const int *>(a + 4 + 8 * lane);
    const int dot = __dp4a(w1, ap[1], __dp4a(w0, ap[0], 0));
    *scaled = (w[s] & 0x0F) * dot;
    const short * bsums = reinterpret_cast<const short *>(a + 260);
    *offset = (lane & 1) == 0 ? (w[s] >> 4) * (int) bsums[s] : 0;
}

// The four bytes of `mags` as signed bytes, negated where the matching bit of `signs` is set: a
// byte flipped and less 0xFF is its negation.
__device__ static __forceinline__ int tx_signed_quad(const unsigned int mags, const int signs) {
    const unsigned int bits = (unsigned int) signs & 15u;
    const unsigned int mask = ((bits & 1u) | ((bits & 2u) << 7) | ((bits & 4u) << 14) | ((bits & 8u) << 21)) * 0xFFu;
    return (int) __vsub4(mags ^ mask, mask);
}

// A run format's run `rr` of a 256-value block `w`: its 32 signed codes, 4 to an int in value
// order, and the integer scales of its 2 halves of 16 values.
template <int F> __device__ static void tx_run_codes(const unsigned char * w, const int rr, int * q, int * lo, int * hi);

// The grid entries of codes 4 * rr .. 4 * rr + 3, each with 2 high bits from the run's byte, under
// the sign bytes 4 * rr ..; a nibble per half of the run gives its odd scale.
template <> __device__ void tx_run_codes<TX_IQ2S>(const unsigned char * w, const int rr, int * q, int * lo, int * hi) {
    const unsigned int codes = tx_u32_2(w + 2 + 4 * rr), signs = tx_u32_2(w + 34 + 4 * rr);
    const unsigned int qh = w[66 + rr];
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned int code = ((codes >> (8 * l)) & 0xFFu) | ((qh << (8 - 2 * l)) & 0x300u);
        const unsigned long long mags = IQ2S_GRID[code];
        const int sg = (int) ((signs >> (8 * l)) & 0xFFu);
        q[2 * l] = tx_signed_quad((unsigned int) mags, sg & 15);
        q[2 * l + 1] = tx_signed_quad((unsigned int) (mags >> 32), sg >> 4);
    }
    const int sc = w[74 + rr];
    *lo = 1 + 2 * (sc & 0x0F);
    *hi = 1 + 2 * (sc >> 4);
}

// Grid codes 8 * rr .. 8 * rr + 7 under the signs of the run's word, whose top 4 bits give the
// run's scale.
template <> __device__ void tx_run_codes<TX_IQ3XXS>(const unsigned char * w, const int rr, int * q, int * lo, int * hi) {
    const unsigned int c0 = tx_u32_2(w + 2 + 8 * rr), c1 = tx_u32_2(w + 6 + 8 * rr);
    const unsigned int word = tx_u32_2(w + 66 + 4 * rr);
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int signs = KSIGNS_IQ2XS[(word >> (7 * l)) & 127];
        const unsigned int c = l < 2 ? c0 : c1, sh = 16 * (l & 1);
        q[2 * l] = tx_signed_quad(IQ3XXS_GRID[(c >> sh) & 0xFFu], signs & 15);
        q[2 * l + 1] = tx_signed_quad(IQ3XXS_GRID[(c >> (sh + 8)) & 0xFFu], signs >> 4);
    }
    *lo = *hi = 1 + 2 * (int) (word >> 28);
}

// Grid codes 8 * rr .. with their 9th bits from the run's byte, under the sign bytes 4 * rr ..;
// a nibble per run gives its scale.
template <> __device__ void tx_run_codes<TX_IQ3S>(const unsigned char * w, const int rr, int * q, int * lo, int * hi) {
    const unsigned int c0 = tx_u32_2(w + 2 + 8 * rr), c1 = tx_u32_2(w + 6 + 8 * rr);
    const unsigned int qh = w[66 + rr], signs = tx_u32_2(w + 74 + 4 * rr);
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned int c = l < 2 ? c0 : c1, sh = 16 * (l & 1);
        const unsigned int i1 = ((c >> sh) & 0xFFu) | ((qh << (8 - 2 * l)) & 256u);
        const unsigned int i2 = ((c >> (sh + 8)) & 0xFFu) | ((qh << (7 - 2 * l)) & 256u);
        const int sg = (int) ((signs >> (8 * l)) & 0xFFu);
        q[2 * l] = tx_signed_quad(IQ3S_GRID[i1], sg & 15);
        q[2 * l + 1] = tx_signed_quad(IQ3S_GRID[i2], sg >> 4);
    }
    *lo = *hi = 1 + 2 * ((w[106 + (rr >> 1)] >> (4 * (rr & 1))) & 0x0F);
}

// Block `rr` of 8 blocks of 32 values, 18 bytes each: the low nibbles of its 16 bytes are values
// 0..16, the high nibbles 16..32, each a level of KVALUES_IQ4NL.
template <> __device__ void tx_run_codes<TX_IQ4NL>(const unsigned char * w, const int rr, int * q, int * lo, int * hi) {
    const unsigned char * levels = reinterpret_cast<const unsigned char *>(KVALUES_IQ4NL);
    const unsigned int table[4] = {tx_u32(levels), tx_u32(levels + 4), tx_u32(levels + 8), tx_u32(levels + 12)};
    unsigned int words[4];
    tx_words<4>(w + 18 * rr + 2, words);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const unsigned int x = words[k];
        q[k] = tx_lookup16(x & 0x0F0F0F0Fu, table);
        q[k + 4] = tx_lookup16((x >> 4) & 0x0F0F0F0Fu, table);
    }
    *lo = *hi = 1;
}

// Half `rr & 1` of block `rr >> 1` of 4 blocks of 64 values, 18 bytes each: 8 bytes of 4 two-bit
// codes each, lowest first, each the level `code - 1`.
template <> __device__ void tx_run_codes<TX_Q2_0>(const unsigned char * w, const int rr, int * q, int * lo, int * hi) {
    const unsigned char * b = w + 18 * (rr >> 1) + 2 + 8 * (rr & 1);
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const unsigned int x4 = tx_u32_2(b + 4 * k);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned int x = (x4 >> (8 * j)) & 0xFFu;
            const unsigned int codes = (x & 3u) | ((x & 0xCu) << 6) | ((x & 0x30u) << 12) | ((x & 0xC0u) << 18);
            q[4 * k + j] = (int) __vsub4(codes, 0x01010101u);
        }
    }
    *lo = *hi = 1;
}

// Block `rr` of 8 blocks of 32 values, 34 bytes each: its 32 signed codes.
template <> __device__ void tx_run_codes<TX_Q8_0>(const unsigned char * w, const int rr, int * q, int * lo, int * hi) {
    unsigned int words[8];
    tx_words<8>(w + 34 * rr + 2, words);
    #pragma unroll
    for (int k = 0; k < 8; ++k) q[k] = (int) words[k];
    *lo = *hi = 1;
}

// Block `rr` of 8 blocks of 32 values, 22 bytes each (scale, 32 high bits, 16 bytes of nibbles):
// the low nibbles are values 0..16, the high nibbles 16..32; bit `v` of the high bits tops value
// `v`; each value is its 5-bit code less 16.
template <> __device__ void tx_run_codes<TX_Q5_0>(const unsigned char * w, const int rr, int * q, int * lo, int * hi) {
    unsigned int words[5];
    tx_words<5>(w + 22 * rr + 2, words);
    const unsigned int qh = words[0];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const unsigned int x = words[k + 1];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int low = (x >> (4 * h)) & 0x0F0F0F0Fu;
            const unsigned int top = (qh >> (16 * h + 4 * k)) & 0xFu;
            const unsigned int high = ((top & 1u) | ((top & 2u) << 7) | ((top & 4u) << 14) | ((top & 8u) << 21)) << 4;
            q[k + 4 * h] = (int) __vsub4(low | high, 0x10101010u);
        }
    }
    *lo = *hi = 1;
}

// A block's share from its two integer sums, as quant_cpu's `combine` of each format.
template <int F> __device__ static float tx_combine(const unsigned char * w, float yd, int scaled, int offset);
template <> __device__ float tx_combine<TX_Q4K>(const unsigned char * w, float yd, int scaled, int offset) {
    return __fmul_rn(yd, __fsub_rn(__fmul_rn(tx_f16(w), (float) scaled),
                                   __fmul_rn(tx_f16(w + 2), (float) offset)));
}
template <> __device__ float tx_combine<TX_Q6K>(const unsigned char * w, float yd, int scaled, int) {
    return __fmul_rn(yd, __fmul_rn(tx_f16(w + 208), (float) scaled));
}
template <> __device__ float tx_combine<TX_Q2K>(const unsigned char * w, float yd, int scaled, int offset) {
    return __fmul_rn(yd, __fsub_rn(__fmul_rn(tx_f16(w + 80), (float) scaled),
                                   __fmul_rn(tx_f16(w + 82), (float) offset)));
}

// The float weight scale of run `run` of a block, as quant_cpu's `ExactDot` of each run format
// gives it to `exact_block_share`.
template <int F> __device__ static float tx_run_scale(const unsigned char * w, int run);
template <> __device__ float tx_run_scale<TX_IQ2S>(const unsigned char * w, int) {
    return __fmul_rn(tx_f16(w), 0.125f);
}
template <> __device__ float tx_run_scale<TX_IQ3XXS>(const unsigned char * w, int) {
    return __fmul_rn(tx_f16(w), 0.25f);
}
template <> __device__ float tx_run_scale<TX_IQ3S>(const unsigned char * w, int) {
    return tx_f16(w);
}
template <> __device__ float tx_run_scale<TX_IQ4NL>(const unsigned char * w, int run) {
    return tx_f16(w + 18 * run);
}
template <> __device__ float tx_run_scale<TX_Q2_0>(const unsigned char * w, int run) {
    return tx_f16(w + 18 * (run >> 1));
}
template <> __device__ float tx_run_scale<TX_Q8_0>(const unsigned char * w, int run) {
    return tx_f16(w + 34 * run);
}
template <> __device__ float tx_run_scale<TX_Q5_0>(const unsigned char * w, int run) {
    return tx_f16(w + 22 * run);
}

// The bytes of a weight row of `len` values: whole blocks of the format.
template <int F> __device__ static __forceinline__ long tx_row_bytes(const int len) {
    return (long) len * TxFmt<F>::BYTES / 256;
}

// A run format's integer sum of run `rr` of a block: each half's codes against the activation's
// under the half's scale.
template <int F> __device__ static __forceinline__ int tx_run_sum(
        const unsigned char * wb, const unsigned char * ab, const int rr) {
    int q[8], lo, hi;
    tx_run_codes<F>(wb, rr, q, &lo, &hi);
    const int4 a0 = *reinterpret_cast<const int4 *>(ab + 16 + 32 * rr);
    const int4 a1 = *reinterpret_cast<const int4 *>(ab + 32 + 32 * rr);
    const int d0 = __dp4a(q[3], a0.w, __dp4a(q[2], a0.z, __dp4a(q[1], a0.y, __dp4a(q[0], a0.x, 0))));
    const int d1 = __dp4a(q[7], a1.w, __dp4a(q[6], a1.z, __dp4a(q[5], a1.y, __dp4a(q[4], a1.x, 0))));
    return lo * d0 + hi * d1;
}

// A run format's term of run `run` of the weight row `w` against the activation row `a`:
// (ws * s) * da, as quant_cpu's `exact_block_share` forms it.
template <int F> __device__ static __forceinline__ float tx_run_term(
        const unsigned char * w, const unsigned char * a, const int run) {
    const int b = run >> 3, rr = run & 7;
    const unsigned char * wb = w + (long) b * TxFmt<F>::BYTES;
    const unsigned char * ab = a + (long) b * TxFmt<F>::ACT;
    return __fmul_rn(__fmul_rn(tx_run_scale<F>(wb, rr), (float) tx_run_sum<F>(wb, ab, rr)), tx_f16(ab + 2 * rr));
}

// A q8_K format's 2 integer sums over the 4 lane parts `part`, `part + 8`, .. of a block: the
// lane's share of the block, which its 8 lanes add up.
template <int F> __device__ static __forceinline__ void tx_part_sums(
        const unsigned char * wb, const unsigned char * ab, const int part, int * scaled, int * offset) {
    int sc = 0, of = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int s, o;
        tx_lane_sums<F>(wb, ab, part + 8 * j, &s, &o);
        sc += s;
        of += o;
    }
    *scaled = sc;
    *offset = of;
}

// q4_K's 4 lane parts at once, the 12 bytes of its scales and minimums read as 3 words.
template <> __device__ void tx_part_sums<TX_Q4K>(
        const unsigned char * w, const unsigned char * a, const int part, int * scaled, int * offset) {
    const unsigned int * sw = reinterpret_cast<const unsigned int *>(w + 4);
    const unsigned int s0 = sw[0], s1 = sw[1], s2 = sw[2];
    const int half = part >> 2, l0 = (part & 3) * 8, shift = 4 * half;
    const short * bsums = reinterpret_cast<const short *>(a + 260);
    int sc = 0, of = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int s = half + 2 * j;
        const unsigned int * q = reinterpret_cast<const unsigned int *>(w + 16 + 32 * j + l0);
        const int w0 = (int) ((q[0] >> shift) & 0x0F0F0F0Fu);
        const int w1 = (int) ((q[1] >> shift) & 0x0F0F0F0Fu);
        const int * ap = reinterpret_cast<const int *>(a + 4 + 8 * (part + 8 * j));
        const int dot = __dp4a(w1, ap[1], __dp4a(w0, ap[0], 0));
        int scale, mn;
        if (s < 4) {
            scale = (s0 >> (8 * s)) & 63;
            mn = (s1 >> (8 * s)) & 63;
        } else {
            const int t = s - 4;
            scale = ((s2 >> (8 * t)) & 0x0F) | (((s0 >> (8 * t + 6)) & 3) << 4);
            mn = ((s2 >> (8 * t + 4)) & 0x0F) | (((s1 >> (8 * t + 6)) & 3) << 4);
        }
        sc += scale * dot;
        if ((part & 3) == 0) of += mn * ((int) bsums[2 * s] + (int) bsums[2 * s + 1]);
    }
    *scaled = sc;
    *offset = of;
}

// The part sums of 2 matrices `w` and `w2` against the same activation block.
template <int F> __device__ static __forceinline__ void tx_part_sums2(
        const unsigned char * w, const unsigned char * w2, const unsigned char * a, const int part,
        int * s1, int * o1, int * s2, int * o2) {
    tx_part_sums<F>(w, a, part, s1, o1);
    tx_part_sums<F>(w2, a, part, s2, o2);
}

// q4_K's part sums of 2 matrices, each activation word and bsum read once for both.
template <> __device__ void tx_part_sums2<TX_Q4K>(
        const unsigned char * w, const unsigned char * w2, const unsigned char * a, const int part,
        int * s1, int * o1, int * s2, int * o2) {
    const unsigned int * sw = reinterpret_cast<const unsigned int *>(w + 4);
    const unsigned int * sv = reinterpret_cast<const unsigned int *>(w2 + 4);
    const unsigned int a0 = sw[0], a1 = sw[1], a2 = sw[2], b0 = sv[0], b1 = sv[1], b2 = sv[2];
    const int half = part >> 2, l0 = (part & 3) * 8, shift = 4 * half;
    const short * bsums = reinterpret_cast<const short *>(a + 260);
    int sc1 = 0, of1 = 0, sc2 = 0, of2 = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int s = half + 2 * j;
        const unsigned int * q = reinterpret_cast<const unsigned int *>(w + 16 + 32 * j + l0);
        const unsigned int * r = reinterpret_cast<const unsigned int *>(w2 + 16 + 32 * j + l0);
        const int * ap = reinterpret_cast<const int *>(a + 4 + 8 * (part + 8 * j));
        const int x0 = ap[0], x1 = ap[1];
        const int d1 = __dp4a((int) ((q[1] >> shift) & 0x0F0F0F0Fu), x1, __dp4a((int) ((q[0] >> shift) & 0x0F0F0F0Fu), x0, 0));
        const int d2 = __dp4a((int) ((r[1] >> shift) & 0x0F0F0F0Fu), x1, __dp4a((int) ((r[0] >> shift) & 0x0F0F0F0Fu), x0, 0));
        int k1, m1, k2, m2;
        if (s < 4) {
            k1 = (a0 >> (8 * s)) & 63;
            m1 = (a1 >> (8 * s)) & 63;
            k2 = (b0 >> (8 * s)) & 63;
            m2 = (b1 >> (8 * s)) & 63;
        } else {
            const int t = s - 4;
            k1 = ((a2 >> (8 * t)) & 0x0F) | (((a0 >> (8 * t + 6)) & 3) << 4);
            m1 = ((a2 >> (8 * t + 4)) & 0x0F) | (((a1 >> (8 * t + 6)) & 3) << 4);
            k2 = ((b2 >> (8 * t)) & 0x0F) | (((b0 >> (8 * t + 6)) & 3) << 4);
            m2 = ((b2 >> (8 * t + 4)) & 0x0F) | (((b1 >> (8 * t + 6)) & 3) << 4);
        }
        sc1 += k1 * d1;
        sc2 += k2 * d2;
        if ((part & 3) == 0) {
            const int bs = (int) bsums[2 * s] + (int) bsums[2 * s + 1];
            of1 += m1 * bs;
            of2 += m2 * bs;
        }
    }
    *s1 = sc1;
    *o1 = of1;
    *s2 = sc2;
    *o2 = of2;
}

// The 4 integers `a`, `b`, `c`, `d` of the 8 lanes of a block summed over those lanes, held by
// every lane of them: each level of the butterfly passes half of what a lane holds, so the 4
// sums take 4 shuffles, and 4 more hand them to every lane.
__device__ static __forceinline__ void tx_reduce4(int & a, int & b, int & c, int & d) {
    const int lane = threadIdx.x & 31, part = lane & 7;
    const bool up = (part & 4) != 0, up2 = (part & 2) != 0;
    const int k0 = (up ? c : a) + __shfl_xor_sync(0xffffffffu, up ? a : c, 4);
    const int k1 = (up ? d : b) + __shfl_xor_sync(0xffffffffu, up ? b : d, 4);
    int x = (up2 ? k1 : k0) + __shfl_xor_sync(0xffffffffu, up2 ? k0 : k1, 2);
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    const int base = lane - part;
    a = __shfl_sync(0xffffffffu, x, base);
    b = __shfl_sync(0xffffffffu, x, base + 2);
    c = __shfl_sync(0xffffffffu, x, base + 4);
    d = __shfl_sync(0xffffffffu, x, base + 6);
}

// The terms of `rows` weight rows of `len` values from `w`, `row_bytes` apart, against the
// activation row `a`, written to `terms` [rows][T + 1]: a run format's T = len / 32 run terms, a
// q8_K format's T = len / 256 block shares. A run format takes a thread per run; a q8_K format 8
// lanes per block, 4 blocks to a warp at a time.
// With `w2` too, the terms of its rows go to `terms2` in the same pass over the activation.
template <int F> __device__ static void tx_terms(
        const unsigned char * w, const unsigned char * w2, const long row_bytes,
        const unsigned char * a, const int len, const int rows, float * terms, float * terms2) {
    if constexpr (TxFmt<F>::RUNS) {
        const int T = len / 32;
        for (int un = threadIdx.x; un < rows * T; un += blockDim.x) {
            const int row = un / T, run = un - row * T;
            terms[row * (T + 1) + run] = tx_run_term<F>(w + row * row_bytes, a, run);
            if (w2 != nullptr) terms2[row * (T + 1) + run] = tx_run_term<F>(w2 + row * row_bytes, a, run);
        }
    } else {
        const int T = len / 256, units = rows * T, lane = threadIdx.x & 31, part = lane & 7;
        for (int u0 = (threadIdx.x >> 5) * 4; u0 < units; u0 += blockDim.x >> 3) {
            // A lane past the units takes the last one and keeps nothing, so the shuffles see
            // every lane.
            const int unit = u0 + (lane >> 3), uu = min(unit, units - 1);
            const int row = uu / T, b = uu - row * T;
            const long off = row * row_bytes + (long) b * TxFmt<F>::BYTES;
            const unsigned char * ab = a + (long) b * TxFmt<F>::ACT;
            // With `w2`, the 2 matrices' sums read the activation once and are reduced together.
            int s1, o1, s2 = 0, o2 = 0;
            if (w2 != nullptr) {
                tx_part_sums2<F>(w + off, w2 + off, ab, part, &s1, &o1, &s2, &o2);
            } else {
                tx_part_sums<F>(w + off, ab, part, &s1, &o1);
            }
            tx_reduce4(s1, o1, s2, o2);
            const float yd = *reinterpret_cast<const float *>(ab);
            if (part == 0 && unit < units) {
                terms[row * (T + 1) + b] = tx_combine<F>(w + off, yd, s1, o1);
                if (w2 != nullptr) terms2[row * (T + 1) + b] = tx_combine<F>(w2 + off, yd, s2, o2);
            }
        }
    }
}

// The dot of ExactDot from a row's terms: a run format's run terms folded from -0 in run order
// into each block's share, the shares folded from -0 in block order; a q8_K format's shares
// folded from -0 in block order.
template <int F> __device__ static float tx_fold(const float * t, const int len) {
    float acc = -0.0f;
    if constexpr (TxFmt<F>::RUNS) {
        const int T = len / 32;
        for (int b = 0; 8 * b < T; ++b) {
            float share = -0.0f;
            const int n = min(8, T - 8 * b);
            for (int r = 0; r < n; ++r) share = __fadd_rn(share, t[8 * b + r]);
            acc = __fadd_rn(acc, share);
        }
    } else {
        for (int b = 0; b < len / 256; ++b) acc = __fadd_rn(acc, t[b]);
    }
    return acc;
}

// The terms a row of `len` values takes in shared memory, with the pad that keeps the rows'
// folds on distinct banks.
template <int F> __device__ static __forceinline__ int tx_terms_stride(const int len) {
    return (TxFmt<F>::RUNS ? len / 32 : len / 256) + 1;
}

// Where instance i's expert sits in stack `stack`, or null for an instance the launch skips.
__device__ static __forceinline__ const unsigned char * tx_expert(
        const unsigned long long base, const unsigned long long * extra, const int stack,
        const int base_slots, const long long bytes, const unsigned int slot) {
    if (slot == TX_SKIP) return nullptr;
    if ((int) slot < base_slots) return (const unsigned char *) (base + (unsigned long long) slot * bytes);
    return (const unsigned char *) (extra[stack] + (unsigned long long) ((int) slot - base_slots) * bytes);
}

// Each of `n` rows of terms `terms` [n][stride] (the rows of `m` matrices of `rows` rows each)
// folded into its first term, in the order of tx_fold: a run format's blocks each folded into
// their first run by a thread per block, then the shares folded by a thread per row. Every thread
// of the block takes part.
template <int F> __device__ static void tx_fold_rows(float * terms, const int n, const int stride, const int len) {
    if constexpr (TxFmt<F>::RUNS) {
        const int T = len / 32, nsb = (T + 7) / 8;
        for (int un = threadIdx.x; un < n * nsb; un += blockDim.x) {
            const int row = un / nsb, b = un - row * nsb;
            float * t = terms + row * stride + 8 * b;
            const int m = min(8, T - 8 * b);
            float share = -0.0f;
            for (int r = 0; r < m; ++r) share = __fadd_rn(share, t[r]);
            t[0] = share;
        }
        __syncthreads();
        for (int row = threadIdx.x; row < n; row += blockDim.x) {
            float * t = terms + row * stride;
            float acc = -0.0f;
            for (int b = 0; b < nsb; ++b) acc = __fadd_rn(acc, t[8 * b]);
            t[0] = acc;
        }
    } else {
        for (int row = threadIdx.x; row < n; row += blockDim.x) {
            float * t = terms + row * stride;
            t[0] = tx_fold<F>(t, len);
        }
    }
    __syncthreads();
}

// The rows entering the down projections: h[pair, row] = swiglu(gate . x, up . x), bounded at
// `limit` when it is positive. A block per (`rows` output rows, instance): the terms of the gate
// and up rows go to shared memory, 2 * rows * tx_terms_stride(dim) floats, and a thread per row
// folds them.
template <int F> __device__ static void tx_h(
        const unsigned long long gate, const unsigned long long up, const unsigned long long * extra,
        const int base_slots, const long long bytes, const unsigned int * slots,
        const unsigned int * pairs, const int k, const unsigned char * xq, float * h,
        const int dim, const int inter, const float limit, const int rows) {
    extern __shared__ float tx_shared[];
    const int row0 = blockIdx.x * rows, i = blockIdx.y;
    const unsigned int slot = slots[i];
    const unsigned char * g = tx_expert(gate, extra, 0, base_slots, bytes, slot);
    if (g == nullptr) return;
    const unsigned char * u = tx_expert(up, extra, 1, base_slots, bytes, slot);
    const int pair = pairs != nullptr ? (int) pairs[i] : i;
    const int nsb = (dim + 255) / 256, stride = tx_terms_stride<F>(dim);
    const long rb = tx_row_bytes<F>(dim);
    const int here = min(rows, inter - row0);
    const unsigned char * a = xq + (long) (pair / k) * nsb * TxFmt<F>::ACT;
    float * tg = tx_shared;
    float * tu = tx_shared + rows * stride;
    if constexpr (TxFmt<F>::RUNS) {
        tx_terms<F>(g + row0 * rb, u + row0 * rb, rb, a, dim, here, tg, tu);
    } else {
        // 8 lanes to a block, the warp's 4 (row, block) units at a time, gate and up together.
        const int T = dim / 256, units = here * T, lane = threadIdx.x & 31, part = lane & 7;
        for (int u0 = (threadIdx.x >> 5) * 4; u0 < units; u0 += blockDim.x >> 3) {
            // A lane past the units takes the last one and keeps nothing, so the shuffles see
            // every lane.
            const int unit = u0 + (lane >> 3), uu = min(unit, units - 1);
            const int row = uu / T, b = uu - row * T;
            const long off = (long) (row0 + row) * rb + (long) b * TxFmt<F>::BYTES;
            const unsigned char * ab = a + (long) b * TxFmt<F>::ACT;
            int s1, o1, s2, o2;
            tx_part_sums2<F>(g + off, u + off, ab, part, &s1, &o1, &s2, &o2);
            tx_reduce4(s1, o1, s2, o2);
            const float yd = *reinterpret_cast<const float *>(ab);
            if (part == 0 && unit < units) {
                tg[row * stride + b] = tx_combine<F>(g + off, yd, s1, o1);
                tu[row * stride + b] = tx_combine<F>(u + off, yd, s2, o2);
            }
        }
    }
    __syncthreads();
    // The up rows follow the gate rows at `rows * stride`: both fold as rows + here rows.
    tx_fold_rows<F>(tx_shared, rows + here, stride, dim);
    const int t = threadIdx.x;
    if (t < here) h[(long) pair * inter + row0 + t] = cx_swiglu(tg[t * stride], tu[t * stride], limit);
}

// The rows of `h` of the launch's instances as q8_K blocks. A warp per (block, instance).
extern "C" __global__ void tx_quantize_h(
        const float * __restrict__ h, const unsigned int * __restrict__ slots,
        const unsigned int * __restrict__ pairs, unsigned char * __restrict__ hq, const int inter) {
    const int blk = blockIdx.x, i = blockIdx.y;
    if (slots[i] == TX_SKIP) return;
    const int pair = pairs != nullptr ? (int) pairs[i] : i;
    const int nb = inter / 256;
    cx_q8k_block(h + (long) pair * inter + (long) blk * 256, hq + ((long) pair * nb + blk) * CX_Q8K_BYTES, threadIdx.x);
}

// One 256-value block at `x` as a q8_K32 block at `ob`, quant_cpu's `BlockQ8K32::quantize` value
// for value: each run of 32 takes the scale amax / 127 stored as f16, and codes rounded half away
// from zero from the values times the reciprocal of the unrounded scale.
// The block holds `valid` values of `x`; the others read as 0, as the host pads a row.
// A warp per block, 8 consecutive values to a lane, 4 lanes to a run.
__device__ static void tx_q8k32_block(const float * __restrict__ x, unsigned char * __restrict__ ob, const int lane, const int valid) {
    const float * xb = x + lane * 8;
    float v[8];
    float amax = 0.0f;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        v[t] = lane * 8 + t < valid ? xb[t] : 0.0f;
        amax = fmaxf(amax, fabsf(v[t]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float d = __fdiv_rn(amax, 127.0f);
    const float inv = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    signed char * qs = reinterpret_cast<signed char *>(ob + 16);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float r = roundf(__fmul_rn(v[t], inv));
        qs[lane * 8 + t] = (signed char) (r != r ? 0 : (int) fminf(fmaxf(r, -128.0f), 127.0f));
    }
    if ((lane & 3) == 0) {
        reinterpret_cast<unsigned short *>(ob)[lane >> 2] = __half_as_ushort(__float2half_rn(d));
    }
}

// Rows of `x` [rows, len] as rows of ceil(len / 256) q8_K32 blocks (see tx_q8k32_block). One
// warp per block of `nblocks`.
extern "C" __global__ void tx_quantize_q8_k32(
        const float * __restrict__ x, unsigned char * __restrict__ out, const long nblocks,
        const int len) {
    const long blk = blockIdx.x;
    if (blk >= nblocks) return;
    const int per_row = (len + 255) / 256;
    const long row = blk / per_row;
    const int b = (int) (blk % per_row);
    tx_q8k32_block(x + row * len + 256 * b, out + blk * CX_Q8K32_BYTES, threadIdx.x, len - 256 * b);
}

// The rows of `h` of the launch's instances as q8_K32 blocks. A warp per (block, instance).
extern "C" __global__ void tx_quantize_h_q8_k32(
        const float * __restrict__ h, const unsigned int * __restrict__ slots,
        const unsigned int * __restrict__ pairs, unsigned char * __restrict__ hq, const int inter) {
    const int blk = blockIdx.x, i = blockIdx.y;
    if (slots[i] == TX_SKIP) return;
    const int pair = pairs != nullptr ? (int) pairs[i] : i;
    const int nb = (inter + 255) / 256;
    tx_q8k32_block(h + (long) pair * inter + (long) blk * 256, hq + ((long) pair * nb + blk) * CX_Q8K32_BYTES, threadIdx.x,
                   inter - 256 * blk);
}

// The pairs' rows: out[pair, row] = weight[pair] * (down . hq[pair]). A block per (`rows` output
// rows, instance): the terms go to shared memory, rows * tx_terms_stride(inter) floats, and a
// thread per row folds them.
template <int F> __device__ static void tx_down(
        const unsigned long long down, const unsigned long long * extra, const int base_slots,
        const long long bytes, const unsigned int * slots, const unsigned int * pairs,
        const float * weights, const unsigned char * hq, float * out, const int inter, const int dim,
        const int rows) {
    extern __shared__ float tx_shared[];
    const int row0 = blockIdx.x * rows, i = blockIdx.y;
    const unsigned char * d = tx_expert(down, extra, 2, base_slots, bytes, slots[i]);
    if (d == nullptr) return;
    const int pair = pairs != nullptr ? (int) pairs[i] : i;
    const int nsb = (inter + 255) / 256, stride = tx_terms_stride<F>(inter);
    const long rb = tx_row_bytes<F>(inter);
    const int here = min(rows, dim - row0);
    tx_terms<F>(d + row0 * rb, nullptr, rb, hq + (long) pair * nsb * TxFmt<F>::ACT, inter, here, tx_shared, nullptr);
    __syncthreads();
    tx_fold_rows<F>(tx_shared, here, stride, inter);
    const int t = threadIdx.x;
    if (t < here) out[(long) pair * dim + row0 + t] = __fmul_rn(weights[pair], tx_shared[t * stride]);
}

#define TX_ENTRIES(name, F)                                                                          \
extern "C" __global__ void tx_h_##name(                                                              \
        const unsigned long long gate, const unsigned long long up, const unsigned long long * extra,\
        const int base_slots, const long long bytes, const unsigned int * slots,                     \
        const unsigned int * pairs, const int k, const unsigned char * xq, float * h,                \
        const int dim, const int inter, const float limit, const int rows) {                         \
    tx_h<F>(gate, up, extra, base_slots, bytes, slots, pairs, k, xq, h, dim, inter, limit, rows);    \
}                                                                                                    \
extern "C" __global__ void tx_down_##name(                                                           \
        const unsigned long long down, const unsigned long long * extra, const int base_slots,      \
        const long long bytes, const unsigned int * slots, const unsigned int * pairs,               \
        const float * weights, const unsigned char * hq, float * out, const int inter,               \
        const int dim, const int rows) {                                                             \
    tx_down<F>(down, extra, base_slots, bytes, slots, pairs, weights, hq, out, inter, dim, rows);    \
}
TX_ENTRIES(q2_k, TX_Q2K)
TX_ENTRIES(q4_k, TX_Q4K)
TX_ENTRIES(q6_k, TX_Q6K)
TX_ENTRIES(iq2_s, TX_IQ2S)
TX_ENTRIES(iq3_xxs, TX_IQ3XXS)
TX_ENTRIES(iq3_s, TX_IQ3S)
TX_ENTRIES(iq4_nl, TX_IQ4NL)
TX_ENTRIES(q2_0, TX_Q2_0)
TX_ENTRIES(q8_0, TX_Q8_0)
TX_ENTRIES(q5_0, TX_Q5_0)

// A dense projection on a card as the host computes it: the weight rows `w` [n, nsb * block bytes]
// against `rows` q8_K rows `xq` [rows, nsb * CX_Q8K_BYTES], written to y [rows, n]. A warp per
// (output row, group of PX_ROWS activation rows): the weight block is read once for the group,
// and each row takes the steps of tx_combine and tx_fold, the dot of ExactDot.
#define PX_ROWS 4
template <int F> __device__ static void px_matvec(
        const unsigned char * __restrict__ w, const unsigned char * __restrict__ xq,
        float * __restrict__ y, const int nsb, const int n, const int rows) {
    const int row = blockIdx.x, r0 = blockIdx.y * PX_ROWS, lane = threadIdx.x;
    const int cnt = min(PX_ROWS, rows - r0);
    const unsigned char * wr = w + (long) row * nsb * TxFmt<F>::BYTES;
    float acc[PX_ROWS];
    #pragma unroll
    for (int t = 0; t < PX_ROWS; ++t) acc[t] = -0.0f;
    for (int b = 0; b < nsb; ++b) {
        const unsigned char * wb = wr + (long) b * TxFmt<F>::BYTES;
        #pragma unroll
        for (int t = 0; t < PX_ROWS; ++t) {
            if (t < cnt) {
                const unsigned char * ab = xq + ((long) (r0 + t) * nsb + b) * CX_Q8K_BYTES;
                int scaled, offset;
                tx_lane_sums<F>(wb, ab, lane, &scaled, &offset);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    scaled += __shfl_xor_sync(0xffffffffu, scaled, off);
                    offset += __shfl_xor_sync(0xffffffffu, offset, off);
                }
                acc[t] = __fadd_rn(acc[t], tx_combine<F>(wb, *reinterpret_cast<const float *>(ab), scaled, offset));
            }
        }
    }
    if (lane == 0) {
        for (int t = 0; t < cnt; ++t) y[(long) (r0 + t) * n + row] = acc[t];
    }
}

#define PX_ENTRY(name, F)                                                                            \
extern "C" __global__ void px_matvec_##name(                                                         \
        const unsigned char * __restrict__ w, const unsigned char * __restrict__ xq,                 \
        float * __restrict__ y, const int nsb, const int n, const int rows) {                        \
    px_matvec<F>(w, xq, y, nsb, n, rows);                                                            \
}
PX_ENTRY(q4_k, TX_Q4K)
PX_ENTRY(q6_k, TX_Q6K)

// A dense F32 projection of 1 row as the host takes 1 row (blockscaled::projection): output o is
// the fold from -0, in input order, of the products w[o][k] * x[k], each rounded once. A warp per
// output: each lane forms 1 product of a run of 32, and every lane folds the run in order.
extern "C" __global__ void seq_row_f32(
        const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int inp) {
    const int o = blockIdx.x, lane = threadIdx.x;
    const float * wr = w + (long) o * inp;
    float acc = -0.0f;
    for (int k0 = 0; k0 < inp; k0 += 32) {
        const int k = k0 + lane;
        const float p = k < inp ? __fmul_rn(wr[k], x[k]) : 0.0f;
        const int run = min(32, inp - k0);
        for (int j = 0; j < run; ++j) acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, p, j));
    }
    if (lane == 0) y[o] = acc;
}

// A dense F32 projection of n rows, each as the host takes 1 row: output (r, o) is the fold from
// -0, in input order, of the products w[o][k] * x[r][k], each rounded once. A thread per output,
// a block per 16 rows by 16 outputs, the inputs staged 16 at a time in shared memory.
extern "C" __global__ void seq_rows_f32(
        const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int inp, const int out, const int n) {
    __shared__ float ws[16][17];
    __shared__ float xs[16][17];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int o0 = blockIdx.x * 16, r0 = blockIdx.y * 16;
    float acc = -0.0f;
    for (int k0 = 0; k0 < inp; k0 += 16) {
        const int k = k0 + tx;
        ws[ty][tx] = (o0 + ty < out && k < inp) ? w[(long) (o0 + ty) * inp + k] : 0.0f;
        xs[ty][tx] = (r0 + ty < n && k < inp) ? x[(long) (r0 + ty) * inp + k] : 0.0f;
        __syncthreads();
        const int run = min(16, inp - k0);
        for (int j = 0; j < run; ++j) acc = __fadd_rn(acc, __fmul_rn(ws[tx][j], xs[ty][j]));
        __syncthreads();
    }
    if (o0 + tx < out && r0 + ty < n) y[(long) (r0 + ty) * out + o0 + tx] = acc;
}

// After a recorded decode's answer: which pairs the transit room runs (late_slots, the expert's
// place there, all ones for the others) and where each pair's row is read from (src: -1 for the
// card's rows, the pair for the host's). The late pairs are read from the mailbox only when the
// request was waited for. One block.
extern "C" __global__ void tx_sources(
        const unsigned int * __restrict__ card_slots, const unsigned int * __restrict__ counter,
        const volatile unsigned int * live, const volatile unsigned int * late_pairs,
        const volatile unsigned int * late_places, unsigned int * __restrict__ late_slots,
        int * __restrict__ src, const int pairs) {
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        late_slots[p] = TX_SKIP;
        src[p] = card_slots[p] != TX_SKIP ? -1 : p;
    }
    __syncthreads();
    const unsigned int n = counter[1] != 0u ? *live : 0u;
    for (unsigned int j = threadIdx.x; j < n; j += blockDim.x) {
        const unsigned int p = late_pairs[j];
        late_slots[p] = late_places[j];
        src[p] = -1;
    }
}

// out[row] = the sum of the row's k pairs in their order from -0, each pair's row read from
// `card` [pairs, hidden], or from row src[pair] of `host` when src[pair] is not negative.
extern "C" __global__ void tx_sum(
        const float * __restrict__ card, const volatile float * host, const int * __restrict__ src,
        float * __restrict__ out, const int rows, const int k, const int hidden) {
    const long idx = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long) rows * hidden) return;
    const int r = (int) (idx / hidden), c = (int) (idx % hidden);
    float acc = -0.0f;
    for (int j = 0; j < k; ++j) {
        const int p = r * k + j;
        const int s = src[p];
        const float v = s < 0 ? card[(long) p * hidden + c] : host[(long) s * hidden + c];
        acc = __fadd_rn(acc, v);
    }
    out[idx] = acc;
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
    const float d = tx_f16(b + 80), dmin = tx_f16(b + 82);
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
    const float d = tx_f16(b + 208);
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

// The coefficients of a token from its mixed values: the gates and the Sinkhorn-normalised comb
// matrix. `N` is the number of hyper-connection copies when it is known at compile time, so the
// loops unroll and the matrix lives in registers; 0 reads it from `hc_rt`.
template <int N>
__device__ static void hc_epilogue(
        const float * mixed, const float * scale, const float * base,
        float * pre, float * post, float * comb, const long t, const long flat, const int hc_rt,
        const int mix_hc, const int iters, const float norm_eps, const float hc_eps) {
    const int hc = N == 0 ? hc_rt : N;
    constexpr int M = N == 0 ? HC_MAX : N;
    const float rsq = 1.0f / sqrtf(mixed[mix_hc] / (float) flat + norm_eps);
    float cm[M][M];
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

// A token's coefficients: the stream's statistic and the projection of its flattened hc*d values
// onto `fn` [mix_hc, flat], then the pre and post gates and the Sinkhorn-normalised comb matrix.
// One block per token. Each dot product of the projection is read by a warp of its own, so the
// products advance together, whatever the number of warps the block holds; the rest runs in
// thread 0. Launched with a multiple of 32 threads.
extern "C" __global__ void __launch_bounds__(1024) hc_mixes_f32(
        const float * __restrict__ x, const float * __restrict__ fn,
        const float * __restrict__ scale, const float * __restrict__ base,
        float * __restrict__ pre, float * __restrict__ post, float * __restrict__ comb,
        const long flat, const int hc, const int mix_hc, const int iters,
        const float norm_eps, const float hc_eps) {
    __shared__ float mixed[(2 + HC_MAX) * HC_MAX + 1];
    const long t = blockIdx.x;
    const float * row = x + t * flat;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    for (int m = warp; m <= mix_hc; m += warps) {
        const float * w = m < mix_hc ? fn + (long) m * flat : row;
        float acc = 0.0f;
        if ((flat & 3) == 0) {
            // Eight 16-byte loads in flight per lane: the products wait on memory, not on the
            // arithmetic.
            const float4 * r4 = (const float4 *) row;
            const float4 * w4 = (const float4 *) w;
            const long n4 = flat / 4;
            float a[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            long j = lane;
            for (; j + 224 < n4; j += 256) {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const float4 p = r4[j + 32 * u], q = w4[j + 32 * u];
                    a[u] += p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
                }
            }
            for (; j < n4; j += 32) {
                const float4 p = r4[j], q = w4[j];
                a[0] += p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
            }
            acc = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        } else {
            for (long j = lane; j < flat; j += 32) acc += row[j] * w[j];
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) mixed[m] = acc;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (hc == 4) hc_epilogue<4>(mixed, scale, base, pre, post, comb, t, flat, hc, mix_hc, iters, norm_eps, hc_eps);
    else hc_epilogue<0>(mixed, scale, base, pre, post, comb, t, flat, hc, mix_hc, iters, norm_eps, hc_eps);
}

// The mixed values of one token's stream, one block for each of the `mix_hc + 1` dot products
// (the last is the stream against itself), so a decode's single token spreads over as many
// multiprocessors as there are products. `mixed` is [n, mix_hc + 1]. Grid (mix_hc + 1, n).
extern "C" __global__ void hc_mixed_dots_f32(
        const float * __restrict__ x, const float * __restrict__ fn,
        float * __restrict__ mixed, const long flat, const int mix_hc) {
    __shared__ float scratch[32];
    const int m = blockIdx.x;
    const long t = blockIdx.y;
    const float * row = x + t * flat;
    const float * w = m < mix_hc ? fn + (long) m * flat : row;
    float acc = 0.0f;
    if ((flat & 3) == 0) {
        const float4 * r4 = (const float4 *) row;
        const float4 * w4 = (const float4 *) w;
        for (long j = threadIdx.x; j < flat / 4; j += blockDim.x) {
            const float4 p = r4[j], q = w4[j];
            acc += p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
        }
    } else {
        for (long j = threadIdx.x; j < flat; j += blockDim.x) acc += row[j] * w[j];
    }
    const float total = hc_block_sum(acc, scratch);
    if (threadIdx.x == 0) mixed[t * (mix_hc + 1) + m] = total;
}

// The gates and the Sinkhorn-normalised comb matrix from the mixed values of `hc_mixed_dots_f32`.
// One thread per token.
extern "C" __global__ void hc_mixed_gates_f32(
        const float * __restrict__ mixed,
        const float * __restrict__ scale, const float * __restrict__ base,
        float * __restrict__ pre, float * __restrict__ post, float * __restrict__ comb,
        const long flat, const int hc, const int mix_hc, const int iters,
        const float norm_eps, const float hc_eps) {
    const long t = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float * m = mixed + t * (mix_hc + 1);
    if (hc == 4) hc_epilogue<4>(m, scale, base, pre, post, comb, t, flat, hc, mix_hc, iters, norm_eps, hc_eps);
    else hc_epilogue<0>(m, scale, base, pre, post, comb, t, flat, hc, mix_hc, iters, norm_eps, hc_eps);
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

// Index scores of one decoded token, a block for each compressed position and a thread for each
// head: the same products and the same sum over the heads, in head order, as `index_scores_f32`
// gives a single query. Launched with `nh * 4` bytes of shared memory and at most 1024 threads.
extern "C" __global__ void index_scores_heads_f32(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ w,
        float * __restrict__ out, const long nh, const long ihd, const long g,
        const long ratio, const long q_off, const float scale) {
    extern __shared__ float contrib[];
    const long t = blockIdx.x;
    long reach = (q_off + 1) / ratio;
    if (reach > g) reach = g;
    if (t >= reach) {
        if (threadIdx.x == 0) out[t] = -INFINITY;
        return;
    }
    const float * kk = k + t * ihd;
    for (long hh = threadIdx.x; hh < nh; hh += blockDim.x) {
        const float * qq = q + hh * ihd;
        float dot = 0.0f;
        for (long c = 0; c < ihd; c++) dot += qq[c] * kk[c];
        contrib[hh] = fmaxf(dot, 0.0f) * w[hh] * scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (long hh = 0; hh < nh; hh++) acc += contrib[hh];
        out[t] = acc;
    }
}

// The keys of each query of a chunk gathered: row `k` of query `i` is `kv[idxs[(start + i) * topk + k]]`,
// zeros for an empty slot. The output is [count, topk, d].
// The threads stride over the output's values, so neighbouring threads copy neighbouring values.
extern "C" __global__ void gather_slots_f32(
        const float * __restrict__ kv, const int * __restrict__ idxs, float * __restrict__ out,
        const long start, const long count, const long topk, const long d) {
    const long total = count * topk * d;
    const long stride = (long) gridDim.x * (long) blockDim.x;
    for (long e = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x; e < total; e += stride) {
        const long slot = idxs[start * topk + e / d];
        out[e] = slot < 0 ? 0.0f : kv[slot * d + e % d];
    }
}

// Logits [count, h, topk] turned in place into attention weights: scaled, softmaxed with the head's
// sink logit, empty slots weighing nothing. One warp per (token, head), its lanes sharing the
// slots; launched with a multiple of 32 threads.
extern "C" __global__ void sink_softmax_f32(
        float * __restrict__ logits, const float * __restrict__ sink, const int * __restrict__ idxs,
        const long start, const long count, const long h, const long topk, const float scale) {
    const long warps = (long) blockDim.x / 32;
    const long idx = (long) blockIdx.x * warps + (long) (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    if (idx >= count * h) return;
    const long i = idx / h, hi = idx % h;
    const int * slots = idxs + (start + i) * topk;
    float * l = logits + idx * topk;
    float m = sink[hi];
    for (long k = lane; k < topk; k += 32) {
        if (slots[k] >= 0) m = fmaxf(m, l[k] * scale);
    }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float denom = 0.0f;
    for (long k = lane; k < topk; k += 32) {
        if (slots[k] >= 0) {
            l[k] = exp(l[k] * scale - m);
            denom += l[k];
        } else {
            l[k] = 0.0f;
        }
    }
    for (int off = 16; off > 0; off >>= 1) denom += __shfl_xor_sync(0xffffffffu, denom, off);
    denom += exp(sink[hi] - m);
    for (long k = lane; k < topk; k += 32) l[k] /= denom;
}

// Block-scaled fp4 weight to f32 in column order: `packed` is [rows, cols/2] e2m1 (element 2j in
// the low nibble, 2j+1 in the high), `scale` [rows, cols/block] the power-of-two scales already
// decoded. Value (r, c) lands at out[c * rows + r]. Every product is exact in f32, so the values
// are the host decoder's bit for bit. One thread per value, consecutive threads down a column.
extern "C" __global__ void fp4_to_columns(
        const unsigned char * __restrict__ packed, const float * __restrict__ scale,
        float * __restrict__ out, const int rows, const int cols, const int block) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * cols) return;
    const int c = (int) (i / rows), r = (int) (i % rows);
    const unsigned char byte = packed[(size_t) r * (cols / 2) + c / 2];
    const int nib = (c & 1) ? (byte >> 4) : (byte & 0x0F);
    const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float mag = grid[nib & 7];
    const float v = (nib & 8) ? -mag : mag;
    out[i] = v * scale[(size_t) r * (cols / block) + c / block];
}

// ---- A decode token's attention state on the card: rounding, pooling, gathering, selection -------

// e4m3fn rounding of `a`, finite and within [-448, 448]: ties to even, subnormal steps of 2^-9.
__device__ static float dec_e4m3_round(float a) {
    if (a == 0.0f) return 0.0f;
    const float mag = fabsf(a);
    float step;
    if (mag < 0.015625f) {
        step = 0.001953125f;
    } else {
        int e;
        frexpf(mag, &e);
        step = ldexpf(1.0f, e - 4);
    }
    return copysignf(fminf(rintf(mag / step) * step, 448.0f), a);
}

// The nearest e2m1 grid magnitude to `m` (>= 0), the first grid point winning a tie.
__device__ static float dec_e2m1_round(float m) {
    const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float best = grid[0], bestd = fabsf(m - grid[0]);
    for (int i = 1; i < 8; ++i) {
        const float d = fabsf(m - grid[i]);
        if (d < bestd) { bestd = d; best = grid[i]; }
    }
    return best;
}

// The smallest power of two at or above `x` (x > 0), as the host computes it.
__device__ static float dec_pow2_ceil(float x) {
    return exp2f(ceilf(log2f(x)));
}

// Block-wise fp8 (e4m3) round trip of `x` in place: `block` consecutive values share the
// power-of-two scale that puts the block's largest magnitude at or below 448. One thread per block
// of values, the host's operations in the host's order.
extern "C" __global__ void fp8_round_f32(float * __restrict__ x, const long total, const long block) {
    const long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b * block >= total) return;
    float * seg = x + b * block;
    float amax = 0.0f;
    for (long i = 0; i < block; ++i) amax = fmaxf(amax, fabsf(seg[i]));
    const float scale = dec_pow2_ceil(fmaxf(amax / 448.0f, 1e-30f));
    for (long i = 0; i < block; ++i) {
        const float c = fminf(fmaxf(seg[i] / scale, -448.0f), 448.0f);
        seg[i] = dec_e4m3_round(c) * scale;
    }
}

// Block-wise fp4 (e2m1) round trip of `x` in place: `block` consecutive values share the scale
// that puts the block's largest magnitude at 6, rounded up to a power of two when `pow2` is set.
extern "C" __global__ void fp4_round_f32(
        float * __restrict__ x, const long total, const long block, const int pow2) {
    const long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b * block >= total) return;
    float * seg = x + b * block;
    float amax = 0.0f;
    for (long i = 0; i < block; ++i) amax = fmaxf(amax, fabsf(seg[i]));
    float scale = fmaxf(amax / 6.0f, 1e-30f);
    if (pow2) scale = dec_pow2_ceil(scale);
    for (long i = 0; i < block; ++i) {
        const float s = fminf(fmaxf(seg[i] / scale, -6.0f), 6.0f);
        seg[i] = copysignf(dec_e2m1_round(fabsf(s)), s) * scale;
    }
}

// `ratio` tokens' latents `kv` [ratio, hd] pooled into one row by a softmax over the tokens of
// their gate scores `score` [ratio, hd], per channel: one thread per channel.
extern "C" __global__ void compress_pool_f32(
        const float * __restrict__ kv, const float * __restrict__ score, float * __restrict__ out,
        const long ratio, const long hd) {
    const long c = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hd) return;
    float mx = -INFINITY;
    for (long r = 0; r < ratio; ++r) mx = fmaxf(mx, score[r * hd + c]);
    float den = 0.0f, acc = 0.0f;
    for (long r = 0; r < ratio; ++r) {
        const float w = expf(score[r * hd + c] - mx);
        den += w;
        acc += w * kv[r * hd + c];
    }
    out[c] = acc / den;
}

// One query's keys gathered from the window ring and the compressed rows: slot `s` below `w` is
// ring row (start + s) % cap, from `w` on compressed row s - w, a negative slot a row of zeros.
// The output is [topk, d], one value per thread.
extern "C" __global__ void gather_ring_f32(
        const float * __restrict__ ring, const long start, const long cap, const long w,
        const float * __restrict__ comp, const int * __restrict__ idxs, float * __restrict__ out,
        const long topk, const long d) {
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= topk * d) return;
    const long slot = i / d, t = i % d;
    const int s = idxs[slot];
    float v = 0.0f;
    if (s >= 0) v = s < w ? ring[((start + s) % cap) * d + t] : comp[(s - w) * d + t];
    out[i] = v;
}

// The stream `x` [hc, d] with `delta` [d] added to every copy, copy c scaled by `post[c]`.
extern "C" __global__ void hc_add_f32(
        float * __restrict__ x, const float * __restrict__ delta, const float * __restrict__ post,
        const long d, const int hc) {
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long) hc * d) return;
    x[i] += post[i / d] * delta[i % d];
}

// A float's order key: unsigned integers in the order of the floats they come from.
__device__ static unsigned int dec_order_key(float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Sum of `v` over the block, valid in every thread. `scratch` holds one int per warp.
__device__ static int dec_block_sum_int(int v, int * scratch) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int total = 0;
    for (int w = 0; w < (int) (blockDim.x + 31) / 32; ++w) total += scratch[w];
    __syncthreads();
    return total;
}

// Exclusive prefix sum of `v` over the block; `scratch` holds one int per thread.
__device__ static int dec_block_scan(int v, int * scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < (int) blockDim.x; off <<= 1) {
        const int t = threadIdx.x >= (unsigned) off ? scratch[threadIdx.x - off] : 0;
        __syncthreads();
        scratch[threadIdx.x] += t;
        __syncthreads();
    }
    return scratch[threadIdx.x] - v;
}

// The candidate blocks of one query's index scores, level one of the two-level pre-filter: each
// block of `block_size` positions is scored by its best position below `reach`, the block holding
// the newest reachable position is pinned, and the `topk_blocks` best blocks (ties to the lower
// block) that are not -inf are kept. `mask` [width] is 1 in a kept block and 0 elsewhere.
// `bscore` and `chosen` [blocks] are scratch. One block of threads, a multiple of 32.
extern "C" __global__ void candidate_blocks_f32(
        const float * __restrict__ scores, const long width, const long reach,
        const long block_size, const long topk_blocks,
        float * __restrict__ bscore, float * __restrict__ chosen, float * __restrict__ mask) {
    __shared__ float rv[32];
    __shared__ long ri[32];
    __shared__ long picked;
    const long nb = (width + block_size - 1) / block_size;
    for (long j = threadIdx.x; j < nb; j += blockDim.x) {
        float m = -INFINITY;
        const long end = (j + 1) * block_size < width ? (j + 1) * block_size : width;
        for (long p = j * block_size; p < end; ++p) {
            const float v = p < reach ? scores[p] : -INFINITY;
            m = fmaxf(m, v);
        }
        bscore[j] = m;
        chosen[j] = 0.0f;
    }
    __syncthreads();
    const long last = reach > 0 ? (reach - 1) / block_size : 0;
    if (threadIdx.x == 0 && last < nb) bscore[last] = INFINITY;
    __syncthreads();
    const long k = topk_blocks < nb ? topk_blocks : nb;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (long it = 0; it < k; ++it) {
        float bv = -INFINITY;
        long bi = -1;
        for (long j = threadIdx.x; j < nb; j += blockDim.x) {
            if (chosen[j] == 0.0f && bscore[j] > bv) { bv = bscore[j]; bi = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const long oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            bv = lane < warps ? rv[lane] : -INFINITY;
            bi = lane < warps ? ri[lane] : -1;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, bv, off);
                const long oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                picked = bi;
                if (bi >= 0) chosen[bi] = 1.0f;
            }
        }
        __syncthreads();
        if (picked < 0) break;
        __syncthreads();
    }
    __syncthreads();
    for (long p = threadIdx.x; p < width; p += blockDim.x) mask[p] = chosen[p / block_size];
}

// Level two's mask step for one query: every score outside a candidate block set to -inf.
extern "C" __global__ void apply_candidate_mask_f32(
        float * __restrict__ scores, const float * __restrict__ mask, const long width) {
    const long i = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
    if (i < width && mask[i] == 0.0f) scores[i] = -INFINITY;
}

// The `k` largest of `scores[0 .. reach)`, the set a descending stable sort keeps (ties to the
// lower index), as slots `index + offset` in `out` [k] and the same as floats in `outf`, -1 past
// the ones found. One block: the k-th largest key is found by a binary search over the key space
// with a count per step, then the keys above it and enough of the keys equal to it are written.
extern "C" __global__ void topk_select_f32(
        const float * __restrict__ scores, const long reach, const long k, const long offset,
        int * __restrict__ out, float * __restrict__ outf) {
    __shared__ int warps[32];
    __shared__ int scan[1024];
    const int T = blockDim.x;
    const long want = k < reach ? k : reach;
    const long per = (reach + T - 1) / T;
    const long lo_i = (long) threadIdx.x * per;
    const long hi_i = lo_i + per < reach ? lo_i + per : reach;
    unsigned int lo = 0u, hi = 0xFFFFFFFFu;
    if (want > 0) {
        // The upper midpoint, computed without overflowing at the top of the key space.
        while (lo < hi) {
            const unsigned int mid = lo + ((hi - lo) >> 1) + 1u;
            int c = 0;
            for (long i = lo_i; i < hi_i; ++i) c += dec_order_key(scores[i]) >= mid;
            const int total = dec_block_sum_int(c, warps);
            if ((long) total >= want) lo = mid; else hi = mid - 1u;
        }
    }
    const unsigned int thr = lo;
    int above = 0, equal = 0;
    if (want > 0) {
        for (long i = lo_i; i < hi_i; ++i) {
            const unsigned int key = dec_order_key(scores[i]);
            above += key > thr;
            equal += key == thr;
        }
    }
    const int at = dec_block_scan(above, scan);
    const int n_above = dec_block_sum_int(above, warps);
    const int at_eq = dec_block_scan(equal, scan);
    if (want > 0) {
        int wa = at, we = n_above + at_eq;
        for (long i = lo_i; i < hi_i; ++i) {
            const unsigned int key = dec_order_key(scores[i]);
            if (key > thr) {
                out[wa] = (int) (i + offset);
                outf[wa] = (float) (i + offset);
                ++wa;
            } else if (key == thr && we < (int) want) {
                out[we] = (int) (i + offset);
                outf[we] = (float) (i + offset);
                ++we;
            }
        }
    }
    for (long i = want + threadIdx.x; i < k; i += T) {
        out[i] = -1;
        outf[i] = -1.0f;
    }
}

// A token's routed experts from its gate logits `z` [n]: scores sqrt(softplus(z / temp)), the `k`
// largest of score + bias taken in turn (ties to the lower expert), their weights the unbiased
// scores, normalised over the chosen when `norm_topk` is set, times `route_scale`. `out` [2k]
// holds the chosen experts as floats, then their weights. One block, the keys in shared memory.
extern "C" __global__ void router_select_f32(
        const float * __restrict__ z, const float * __restrict__ bias, const long n, const long k,
        const float temp, const int norm_topk, const float route_scale, float * __restrict__ out) {
    extern __shared__ float key[];
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    __shared__ int pick;
    const int T = blockDim.x;
    for (long e = threadIdx.x; e < n; e += T) {
        const float s = z[e] / temp;
        const float sp = s > 20.0f ? s : log1pf(expf(s));
        key[e] = sqrtf(sp) + bias[e];
    }
    __syncthreads();
    for (long slot = 0; slot < k; ++slot) {
        float v = -INFINITY;
        int bi = -1;
        for (long e = threadIdx.x; e < n; e += T) {
            if (key[e] > v) { v = key[e]; bi = (int) e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > v || (ov == v && oi >= 0 && (bi < 0 || oi < bi))) { v = ov; bi = oi; }
        }
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) { best_v[warp] = v; best_i[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = -INFINITY;
            int bb = -1;
            for (int w = 0; w < (T + 31) / 32; ++w) {
                const bool better = best_v[w] > bv
                    || (best_v[w] == bv && best_i[w] >= 0 && (bb < 0 || best_i[w] < bb));
                if (better) { bv = best_v[w]; bb = best_i[w]; }
            }
            pick = bb;
            out[slot] = (float) bb;
            const float s = z[bb] / temp;
            out[k + slot] = sqrtf(s > 20.0f ? s : log1pf(expf(s)));
        }
        __syncthreads();
        if (threadIdx.x == 0) key[pick] = -INFINITY;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (norm_topk && k > 1) {
            float sum = 0.0f;
            for (long slot = 0; slot < k; ++slot) sum += out[k + slot];
            sum += 1e-20f;
            for (long slot = 0; slot < k; ++slot) out[k + slot] /= sum;
        }
        for (long slot = 0; slot < k; ++slot) out[k + slot] *= route_scale;
    }
}
