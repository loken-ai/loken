// One decoder layer of a decode step (one row) in a single launch.
//
// A launch is a grid of workgroups that all stay resident, and the phases of the layer are
// separated by a barrier across the grid: the projections, the attention, the output
// projection, the gate and up projections, the down projection. A phase reads its activations
// from the workgroup's local memory, where it quantised them itself from the previous phase's
// output, and writes the rows its workgroup owns to global memory. This is the layer
// `dense_stack.cl` runs as a dozen launches, with the same arithmetic.
//
// Built only for R == 1 and where the device has device-scope acquire/release atomics.

#define NT 256
// Spins of a workgroup at a barrier before it gives up: a round trip to the counter is a
// microsecond or less, so these are a fraction of a second.
#define SPIN_LIMIT 150000u
#define SL_LANES (SLICES * LANES)
#define SL_SLOTS (NT / SL_LANES)

// Format ids, as the host numbers them.
#define F_Q8_0 0
#define F_Q4_0 1
#define F_Q5_0 2
#define F_F16 3
#define F_F32 4
#define F_Q4_K 5
#define F_Q6_K 6

// A barrier across the grid. `ctr` counts arrivals, one barrier after another, so the target
// of the n-th barrier is n * groups. A workgroup that waits too long gives up and says so in
// `fail`, so that a grid that is not fully resident ends in an error and not a hang.
inline void grid_sync(volatile __global atomic_uint* ctr, uint n, __global uint* fail, uint limit) {
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        uint target = n * (uint)get_num_groups(0);
        atomic_fetch_add_explicit(ctr, 1u, memory_order_acq_rel, memory_scope_device);
        uint spins = 0;
        while (atomic_load_explicit(ctr, memory_order_acquire, memory_scope_device) < target) {
            if (++spins > limit) {
                atomic_fetch_add_explicit((volatile __global atomic_uint*)fail, 1u,
                                          memory_order_relaxed, memory_scope_device);
                break;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

// The activation row in local memory: floats, q8_K and q8_0 codes and their scales.
typedef struct {
    __local float* xf;
    __local char* qk;
    __local char* qz;
    __local float* dk;
    __local int* sk;
    __local float* dz;
    __local float* red;
} Smem;

// Loads `n` values from `src` (normalised by `nw` when given) and quantises them the way
// quant_rows does. Ends with a workgroup barrier.
#define XVAL(i) (keep ? m.xf[i] : src[i])

inline void prep_x(__global const float* src, __global const float* nw, uint use_norm, uint keep,
                   uint n, float eps, Smem m) {
    uint t = get_local_id(0);
    float scale = 1.0f;
    if (use_norm) {
        float s = 0;
        for (uint i = t; i < n; i += NT) {
            float v = src[i];
            s += v * v;
        }
        m.red[t] = s;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = NT / 2; k > 0; k >>= 1) {
            if (t < k) m.red[t] += m.red[t + k];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        scale = rsqrt(m.red[0] / (float)n + eps);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // The floats stay in local memory only for a row that float formats may read; any other
    // is read from `src` again where it is needed (it is not normalised).
    if (keep)
        for (uint i = t; i < n; i += NT) m.xf[i] = use_norm ? src[i] * scale * nw[i] : src[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // q8_0: a scale per 32, the largest magnitude over 127, rounded to half.
    uint nb32 = n / 32;
    for (uint b = t; b < nb32; b += NT) {
        float amax = 0;
        for (uint j = 0; j < 32; j++) amax = fmax(amax, fabs(XVAL(b * 32 + j)));
        float d32 = amax / 127.0f;
        ushort hh;
        vstore_half(d32, 0, (__private half*)&hh);
        m.dz[b] = vload_half(0, (__private half*)&hh);
        float inv = d32 != 0.0f ? 1.0f / d32 : 0.0f;
        for (uint j = 0; j < 32; j++) m.qz[b * 32 + j] = convert_char_sat(round(XVAL(b * 32 + j) * inv));
    }

    // q8_K, for rows of whole blocks of 256: eight threads find each block's extreme (the
    // value of largest magnitude, the first of equals), which maps to -127.
    uint nb256 = n / 256;
    if (n % 256 == 0 && nb256 > 0) {
        // red[t] holds the part's extreme; the combined extreme and its scale follow.
        if (t < nb256 * 8) {
            uint b = t / 8, p = t % 8;
            float e = 0;
            for (uint j = 0; j < 32; j++) {
                float v = XVAL(b * 256 + p * 32 + j);
                if (fabs(v) > fabs(e)) e = v;
            }
            m.red[t] = e;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t < nb256) {
            float e = 0;
            for (uint p = 0; p < 8; p++) {
                float v = m.red[t * 8 + p];
                if (fabs(v) > fabs(e)) e = v;
            }
            float invk = e != 0.0f ? -127.0f / e : 0.0f;
            m.dk[nb256 + t] = invk;
            m.dk[t] = e != 0.0f ? 1.0f / invk : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = t; i < n; i += NT) {
            float invk = m.dk[nb256 + i / 256];
            m.qk[i] = invk != 0.0f ? convert_char_sat(min((int)rint(invk * XVAL(i)), 127)) : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint b = t; b < nb32; b += NT) {
            int s = 0;
            for (uint j = 0; j < 32; j++) s += (int)m.qk[b * 32 + j];
            m.sk[b] = s;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// The partial dot products of lane `lane` of slice `slice` for row `row` of a matrix (and of
// a second, when use_b) against the activations in local memory.
inline void row_partial(uint fmt, __global const uchar* wa, __global const uchar* wb, uint use_b,
                        ulong total, uint row, uint cols, uint lane, uint slice, Smem m,
                        float* accA, float* accB) {
    XQ xq = {m.qk, m.qz, m.dk, m.sk, m.dz, m.xf};
    uint nb = cols / 32, ng = (nb + 7) / 8;
    for (uint g = slice; g < ng; g += SLICES) {
        ulong blk0 = (ulong)row * nb + 8 * g;
        uint nsub = min(8u, nb - 8 * g);
        // The activations are read from local memory where the format needs them.
        const uint* xw = 0;
        const float* xs = 0;
        const float4* xc = 0;
#define CALL(GRP) GRP(wa, wb, use_b, total, blk0, nsub, lane, g, nb, (__global const float*)0, xq, xw, xs, 0.0f, cols, 1, accA, accB)
        switch (fmt) {
            case F_F16: grp_f16(wa, wb, use_b, total, blk0, nsub, lane, g, nb, (__global const float*)0, xq, xw, xs, 0.0f, cols, 1, accA, accB, xc); break;
            case F_F32: grp_f32(wa, wb, use_b, total, blk0, nsub, lane, g, nb, (__global const float*)0, xq, xw, xs, 0.0f, cols, 1, accA, accB, xc); break;
            case F_Q4_K: CALL(grp_q4_k); break;
            case F_Q6_K: CALL(grp_q6_k); break;
            case F_Q8_0: CALL(grp_q8_0); break;
            case F_Q4_0: CALL(grp_q4_0); break;
            default: CALL(grp_q5_0); break;
        }
#undef CALL
    }
}

// A matrix as the kernel receives it.
typedef struct {
    __global const uchar* w;
    ulong total;
    uint fmt;
    uint rows;
    uint cols;
} Mat;

// Sums the SL_LANES partials of each slot into the slot's first thread.
inline float slot_sum(__local float* part, uint slot) {
    float s = 0;
    for (uint l = 0; l < SL_LANES; l++) s += part[slot * SL_LANES + l];
    return s;
}

// The layer's phases, in order. `ctr` is the layer's own counter, `fail` the shared error count.
__kernel __attribute__((reqd_work_group_size(NT, 1, 1))) void layer_step(
    __global const float* xin, __global float* xout, __global float* qbuf, __global float* kbuf,
    __global float* vbuf, __global float* attbuf, __global float* actbuf,
    __global const uchar* wq, __global const uchar* wk, __global const uchar* wv,
    __global const uchar* wo, __global const uchar* wg, __global const uchar* wu,
    __global const uchar* wd,
    uint fq, uint fk, uint fv, uint fo, uint fg, uint fu, uint fd,
    ulong tq, ulong tk, ulong tv, ulong to, ulong tg, ulong tu, ulong td,
    __global const float* attn_norm, __global const float* ffn_norm,
    __global const float* qbias, __global const float* kbias, __global const float* vbias,
    __global const float* qnorm, __global const float* knorm,
    __global float* cache_k, __global float* cache_v,
    __global const float* cosv, __global const float* sinv,
    uint hidden, uint n_head, uint n_kv, uint head_dim, uint ffn, uint rope_dim, uint half_tab,
    uint flags, float eps, float attn_scale, uint pos,
    volatile __global atomic_uint* counters, uint slot_no, __global uint* fail) {
    volatile __global atomic_uint* ctr = counters + slot_no * 8;
    __local float xf[MAXH];
    __local char qk[MAXC];
    __local char qz[MAXC];
    __local float dk[MAXC / 256 + 1 + MAXC / 256 + 1];
    __local int sk[MAXC / 32];
    __local float dz[MAXC / 32];
    __local float red[NT];
    __local float part[NT];
    __local float partb[NT];
    __local float scratch[2 * NT];
    Smem m = {xf, qk, qz, dk, sk, dz, red};

    uint t = get_local_id(0), wg_id = get_group_id(0), nwg = get_num_groups(0);
    uint lane = t % LANES, slice = (t / LANES) % SLICES, slot = t / SL_LANES;
    uint sl = t % SL_LANES;
    uint q_dim = n_head * head_dim, kv_dim = n_kv * head_dim;
    uint barrier_no = 0;

    // ---- Phase 1: q, k, v.
    prep_x(xin, attn_norm, 1, 1, hidden, eps, m);
    {
        uint rows_total = q_dim + 2 * kv_dim;
        uint ntiles = (rows_total + SL_SLOTS - 1) / SL_SLOTS;
        for (uint tile = wg_id; tile < ntiles; tile += nwg) {
            uint row = tile * SL_SLOTS + slot;
            float acc[1] = {0}, none[1] = {0};
            __global const uchar* w = wq;
            ulong total = tq;
            uint fmt = fq, r = row;
            if (row >= q_dim + kv_dim) { w = wv; total = tv; fmt = fv; r = row - q_dim - kv_dim; }
            else if (row >= q_dim) { w = wk; total = tk; fmt = fk; r = row - q_dim; }
            if (row < rows_total) row_partial(fmt, w, w, 0, total, r, hidden, lane, slice, m, acc, none);
            part[t] = acc[0];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (sl == 0 && row < rows_total) {
                float s = slot_sum(part, slot);
                if (row >= q_dim + kv_dim) {
                    if (flags & 4u) s += vbias[r];
                    vbuf[r] = s;
                } else if (row >= q_dim) {
                    if (flags & 2u) s += kbias[r];
                    kbuf[r] = s;
                } else {
                    if (flags & 1u) s += qbias[r];
                    qbuf[r] = s;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    grid_sync(ctr, ++barrier_no, fail, SPIN_LIMIT);

    // ---- Phase 2: attention. Workgroup h takes head h; the current key and value come from
    // the projections, not from the cache, so no workgroup waits for another's cache write.
    for (uint h = wg_id; h < n_head; h += nwg) {
        uint kvh = h / (n_head / n_kv);
        __local float* ql = scratch;               // the head's query, rotated
        __local float* kc = scratch + 256;         // the current key, rotated
        __local float* p = part;                   // scores of a tile
        uint pairs = rope_dim / 2;
        // Query and key through their norms and RoPE.
        for (int which = 0; which < 2; which++) {
            __global const float* src = which == 0 ? qbuf + h * head_dim : kbuf + kvh * head_dim;
            __global const float* nw = which == 0 ? qnorm : knorm;
            bool norm = which == 0 ? (flags & 8u) != 0 : (flags & 16u) != 0;
            __local float* dst = which == 0 ? ql : kc;
            float s = 0;
            for (uint i = t; i < head_dim; i += NT) {
                float v = src[i];
                dst[i] = v;
                s += v * v;
            }
            red[t] = s;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint k = NT / 2; k > 0; k >>= 1) {
                if (t < k) red[t] += red[t + k];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float scale = norm ? rsqrt(red[0] / (float)head_dim + eps) : 1.0f;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint i = t; i < head_dim; i += NT) dst[i] = dst[i] * scale * (norm ? nw[i] : 1.0f);
            barrier(CLK_LOCAL_MEM_FENCE);
            // RoPE reads pairs, so rotate into registers first.
            float rot = 0;
            if (t < head_dim) {
                rot = dst[t];
                if (t < rope_dim) {
                    uint pi, other;
                    bool first;
                    if (flags & 32u) { pi = t / 2; first = (t % 2) == 0; other = first ? t + 1 : t - 1; }
                    else { pi = t % pairs; first = t < pairs; other = first ? t + pairs : t - pairs; }
                    float c = cosv[(ulong)pos * half_tab + pi], sn = sinv[(ulong)pos * half_tab + pi];
                    rot = first ? dst[t] * c - dst[other] * sn : dst[other] * sn + dst[t] * c;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (t < head_dim) dst[t] = rot;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // The cache takes the current key and value, once per key-value head.
        if (h % (n_head / n_kv) == 0) {
            for (uint i = t; i < head_dim; i += NT) {
                cache_k[((ulong)pos * n_kv + kvh) * head_dim + i] = kc[i];
                cache_v[((ulong)pos * n_kv + kvh) * head_dim + i] = vbuf[kvh * head_dim + i];
            }
        }
        // Softmax over positions 0..pos in tiles of NT keys, then the current one.
        uint split = NT / head_dim;                 // threads of a lane of the value sum
        uint d = t % head_dim, part_id = t / head_dim;
        float acc = 0, mx = -INFINITY, l = 0;
        uint len = pos;                              // cached keys
        for (uint t0 = 0; t0 < len + 1; t0 += NT) {
            uint kp = t0 + t;
            float sc = -INFINITY;
            if (kp < len) {
                __global const float* kr = cache_k + ((ulong)kp * n_kv + kvh) * head_dim;
                float dsum = 0;
                for (uint i = 0; i < head_dim; i += 4) dsum += dot(vload4(0, kr + i), vload4(0, ql + i));
                sc = dsum * attn_scale;
            } else if (kp == len) {
                float dsum = 0;
                for (uint i = 0; i < head_dim; i += 4) dsum += dot(vload4(0, kc + i), vload4(0, ql + i));
                sc = dsum * attn_scale;
            }
            red[t] = sc;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint k = NT / 2; k > 0; k >>= 1) {
                if (t < k) red[t] = fmax(red[t], red[t + k]);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float mt = red[0];
            barrier(CLK_LOCAL_MEM_FENCE);
            float mn = fmax(mx, mt);
            float pr = kp <= len ? exp(sc - mn) : 0.0f;
            p[t] = pr;
            red[t] = pr;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint k = NT / 2; k > 0; k >>= 1) {
                if (t < k) red[t] += red[t + k];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            float corr = exp(mx - mn);
            l = l * corr + red[0];
            mx = mn;
            acc *= corr;
            barrier(CLK_LOCAL_MEM_FENCE);
            // Value sum: thread (part_id, d) takes the tile's keys part_id, part_id + split, ...
            if (part_id < split) {
                uint cnt = min((uint)NT, len + 1 - t0);
                for (uint j = part_id; j < cnt; j += split) {
                    uint kpos = t0 + j;
                    float v = kpos < len ? cache_v[((ulong)kpos * n_kv + kvh) * head_dim + d]
                                         : vbuf[kvh * head_dim + d];
                    acc += p[j] * v;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // Merge the parts of the value sum.
        scratch[t] = part_id < split ? acc : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t < head_dim) {
            float s = 0;
            for (uint q = 0; q < split; q++) s += scratch[q * head_dim + t];
            attbuf[h * head_dim + t] = s / l;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    grid_sync(ctr, ++barrier_no, fail, SPIN_LIMIT);

    // ---- Phase 3: output projection and the residual.
    prep_x(attbuf, (__global const float*)0, 0, 0, q_dim, eps, m);
    for (uint tile = wg_id; tile < (hidden + SL_SLOTS - 1) / SL_SLOTS; tile += nwg) {
        uint row = tile * SL_SLOTS + slot;
        float acc[1] = {0}, none[1] = {0};
        if (row < hidden) row_partial(fo, wo, wo, 0, to, row, q_dim, lane, slice, m, acc, none);
        part[t] = acc[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sl == 0 && row < hidden) xout[row] = slot_sum(part, slot) + xin[row];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    grid_sync(ctr, ++barrier_no, fail, SPIN_LIMIT);

    // ---- Phase 4: gate and up, with the SiLU.
    prep_x(xout, ffn_norm, 1, 1, hidden, eps, m);
    for (uint tile = wg_id; tile < (ffn + SL_SLOTS - 1) / SL_SLOTS; tile += nwg) {
        uint row = tile * SL_SLOTS + slot;
        float ag[1] = {0}, au[1] = {0};
        if (row < ffn) row_partial(fg, wg, wu, 1, tg, row, hidden, lane, slice, m, ag, au);
        part[t] = ag[0];
        partb[t] = au[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sl == 0 && row < ffn) {
            float g = slot_sum(part, slot), v = slot_sum(partb, slot);
            actbuf[row] = g / (1.0f + exp(-g)) * v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    grid_sync(ctr, ++barrier_no, fail, SPIN_LIMIT);

    // ---- Phase 5: down projection and the residual, in place on the output.
    prep_x(actbuf, (__global const float*)0, 0, 0, ffn, eps, m);
    for (uint tile = wg_id; tile < (hidden + SL_SLOTS - 1) / SL_SLOTS; tile += nwg) {
        uint row = tile * SL_SLOTS + slot;
        float acc[1] = {0}, none[1] = {0};
        if (row < hidden) row_partial(fd, wd, wd, 0, td, row, ffn, lane, slice, m, acc, none);
        part[t] = acc[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sl == 0 && row < hidden) xout[row] += slot_sum(part, slot);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// The output head: the final norm of the last row and the logits.
__kernel __attribute__((reqd_work_group_size(NT, 1, 1))) void head_step(
    __global const float* xin, __global const float* norm, __global const uchar* w, __global float* logits,
    uint fmt, ulong total, uint rows, uint hidden, float eps) {
    __local float xf[MAXH];
    __local char qk[MAXC];
    __local char qz[MAXC];
    __local float dk[MAXC / 256 + 1 + MAXC / 256 + 1];
    __local int sk[MAXC / 32];
    __local float dz[MAXC / 32];
    __local float red[NT];
    __local float part[NT];
    Smem m = {xf, qk, qz, dk, sk, dz, red};
    uint t = get_local_id(0);
    uint lane = t % LANES, slice = (t / LANES) % SLICES, slot = t / SL_LANES, sl = t % SL_LANES;
    prep_x(xin, norm, 1, 1, hidden, eps, m);
    for (uint tile = get_group_id(0); tile < (rows + SL_SLOTS - 1) / SL_SLOTS; tile += get_num_groups(0)) {
        uint row = tile * SL_SLOTS + slot;
        float acc[1] = {0}, none[1] = {0};
        if (row < rows) row_partial(fmt, w, w, 0, total, row, hidden, lane, slice, m, acc, none);
        part[t] = acc[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sl == 0 && row < rows) logits[row] = slot_sum(part, slot);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
