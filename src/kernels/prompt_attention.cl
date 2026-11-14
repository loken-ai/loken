// Causal attention of a prompt chunk for the OpenCL stack, the keys read once for a tile of
// queries.
//
// The program is built for a tile of TILE queries and TILE keys and for heads of HEAD_DIM lanes.
// The queries of a key head are the chunk's rows times the query heads that share that key head,
// row after row. A workgroup takes TILE of them and one slice of the key tiles they see.
// Each key tile goes into local memory once and meets every query of the workgroup.
//
// With one slice the workgroup writes the attention output itself. With several, each writes its
// value sums, its running maximum and its running sum into `parts`, and `merge_slices` adds the
// slices up.

#define AT 128
// Outputs of a workgroup's tile each thread accumulates.
#define ACC ((TILE * HEAD_DIM + AT - 1) / AT)
// Floats a slice writes for each query: the value sums, then the maximum and the sum.
#define PART (HEAD_DIM + 2)

__kernel __attribute__((reqd_work_group_size(AT, 1, 1))) void prompt_attention(
    __global const float* q, __global const float* cache_k, __global const float* cache_v,
    __global float* out, __global float* parts, uint n_head, uint n_kv, uint rows, float scale,
    uint index_pos, uint slices, uint per_slice) {
    __local float qs[TILE * HEAD_DIM];
    __local float ks[TILE * HEAD_DIM];
    __local float vs[TILE * HEAD_DIM];
    __local float sc[TILE * TILE];
    __local float mx[TILE];
    __local float sm[TILE];
    __local float cr[TILE];
    uint tid = get_local_id(0);
    uint wg = get_group_id(0);
    uint s = wg % slices;
    uint kvh = (wg / slices) % n_kv;
    uint j0 = (wg / slices / n_kv) * TILE;
    uint g = n_head / n_kv;
    uint queries = rows * g;
    for (uint i = tid; i < TILE * HEAD_DIM; i += AT) {
        uint j = j0 + i / HEAD_DIM;
        uint at = ((j / g) * n_head + kvh * g + j % g) * HEAD_DIM + i % HEAD_DIM;
        qs[i] = j < queries ? q[at] * scale : 0.0f;
    }
    for (uint i = tid; i < TILE; i += AT) {
        mx[i] = -INFINITY;
        sm[i] = 0.0f;
    }
    float acc[ACC];
    _Pragma("unroll") for (uint a = 0; a < ACC; a++) acc[a] = 0.0f;
    // The keys the tile's last query sees bound the key tiles.
    uint len = index_pos + (min(j0 + TILE, queries) - 1) / g + 1;
    uint t_end = min((s + 1) * per_slice, (len + TILE - 1) / TILE);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint t = s * per_slice; t < t_end; t++) {
        uint k0 = t * TILE;
        for (uint i = tid; i < TILE * HEAD_DIM; i += AT) {
            uint kp = k0 + i / HEAD_DIM;
            ulong at = ((ulong)kp * n_kv + kvh) * HEAD_DIM + i % HEAD_DIM;
            ks[i] = kp < len ? cache_k[at] : 0.0f;
            vs[i] = kp < len ? cache_v[at] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Score (query, key) of the tile; a key past the query's own row is masked.
        // Each key starts its walk along the lanes at its own offset, so the threads of a key
        // row read different banks.
        for (uint i = tid; i < TILE * TILE; i += AT) {
            uint qi = i / TILE, kk = i % TILE;
            uint j = j0 + qi;
            float d = -INFINITY;
            if (j < queries && k0 + kk <= index_pos + j / g) {
                d = 0.0f;
                uint e = (4 * kk) % HEAD_DIM;
                for (uint n = 0; n < HEAD_DIM; n += 4) {
                    d += dot(vload4(0, qs + qi * HEAD_DIM + e), vload4(0, ks + kk * HEAD_DIM + e));
                    e = e + 4 == HEAD_DIM ? 0 : e + 4;
                }
            }
            sc[i] = d;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // The running softmax of each query: its scores become weights.
        for (uint qi = tid; qi < TILE; qi += AT) {
            float m = mx[qi], mt = m;
            for (uint kk = 0; kk < TILE; kk++) mt = fmax(mt, sc[qi * TILE + kk]);
            float l = 0.0f;
            for (uint kk = 0; kk < TILE; kk++) {
                float p = mt == -INFINITY ? 0.0f : exp(sc[qi * TILE + kk] - mt);
                sc[qi * TILE + kk] = p;
                l += p;
            }
            float c = mt == -INFINITY ? 1.0f : exp(m - mt);
            mx[qi] = mt;
            sm[qi] = sm[qi] * c + l;
            cr[qi] = c;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        _Pragma("unroll") for (uint a = 0; a < ACC; a++) {
            uint o = tid + a * AT;
            if (o < TILE * HEAD_DIM) {
                uint qi = o / HEAD_DIM, d = o % HEAD_DIM;
                float x = acc[a] * cr[qi];
                for (uint kk = 0; kk < TILE; kk++) x += sc[qi * TILE + kk] * vs[kk * HEAD_DIM + d];
                acc[a] = x;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    _Pragma("unroll") for (uint a = 0; a < ACC; a++) {
        uint o = tid + a * AT;
        uint qi = o / HEAD_DIM, d = o % HEAD_DIM;
        uint j = j0 + qi;
        if (o >= TILE * HEAD_DIM || j >= queries) continue;
        if (slices == 1)
            out[((ulong)(j / g) * n_head + kvh * g + j % g) * HEAD_DIM + d] = acc[a] / sm[qi];
        else
            parts[((ulong)wg * TILE + qi) * PART + d] = acc[a];
    }
    if (slices > 1) {
        for (uint qi = tid; qi < TILE; qi += AT) {
            parts[((ulong)wg * TILE + qi) * PART + HEAD_DIM] = mx[qi];
            parts[((ulong)wg * TILE + qi) * PART + HEAD_DIM + 1] = sm[qi];
        }
    }
}

// The attention output of every (row, head) from the slices of `prompt_attention`: each slice's
// sums are rescaled to the largest maximum and added. One workgroup per (row, head).
__kernel __attribute__((reqd_work_group_size(AT, 1, 1))) void merge_slices(
    __global const float* parts, __global float* out, uint n_head, uint n_kv, uint slices) {
    uint row = get_group_id(0) / n_head, head = get_group_id(0) % n_head;
    uint g = n_head / n_kv;
    uint kvh = head / g;
    uint j = row * g + head % g;
    // The first workgroup of the query's tile and key head, and the query in the tile.
    ulong first = ((ulong)(j / TILE) * n_kv + kvh) * slices;
    uint qi = j % TILE;
    float m = -INFINITY;
    for (uint s = 0; s < slices; s++)
        m = fmax(m, parts[((first + s) * TILE + qi) * PART + HEAD_DIM]);
    float l = 0.0f;
    for (uint s = 0; s < slices; s++) {
        __global const float* p = parts + ((first + s) * TILE + qi) * PART;
        l += p[HEAD_DIM] == -INFINITY ? 0.0f : p[HEAD_DIM + 1] * exp(p[HEAD_DIM] - m);
    }
    for (uint d = get_local_id(0); d < HEAD_DIM; d += AT) {
        float x = 0.0f;
        for (uint s = 0; s < slices; s++) {
            __global const float* p = parts + ((first + s) * TILE + qi) * PART;
            if (p[HEAD_DIM] != -INFINITY) x += p[d] * exp(p[HEAD_DIM] - m);
        }
        out[((ulong)row * n_head + head) * HEAD_DIM + d] = x / l;
    }
}
