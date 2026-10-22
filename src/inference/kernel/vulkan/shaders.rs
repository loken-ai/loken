//! The kernels, in WGSL, compiled to SPIR-V with naga.
//!
//! Every kernel shares one layout: storage bindings 0..8 of set 0 and 64 bytes of immediates,
//! read as sixteen `u32` (a float travels as its bits). Weights are read as whole words, or as
//! four-word vectors where the block size allows.
//!
//! Activations that feed a mat-vec travel as int8 blocks of 32 (`xq`, four values a word) with
//! one `vec4<f32>` per block in `xm`: the scale `d`, and the sums of the first and of the last
//! sixteen int8 values. The mat-vec kernels take the integer dot product of the weight
//! bytes with those values.

/// Tokens a lane of the integer mat-mat takes, and tokens a workgroup of it takes (eight lanes
/// across).
pub(super) const MM_TT: u32 = 4;
pub(super) const MM_TOKENS: u32 = 8 * MM_TT;

/// Lanes that run one row of the mat-vec of `kernel`: more where a row has plenty of items
/// and the format's loads are narrow enough to want the extra requests in flight.
pub(super) fn lanes_per_row(kernel: &str) -> u32 {
    match kernel {
        "mv_q4k" | "mv_f16" | "mx_q4k" | "mx_f16" => 32,
        _ => 16,
    }
}

/// The immediates: sixteen words whose meaning each kernel states.
const HEAD: &str = r#"
struct P { a: array<vec4<u32>, 4> }
var<immediate> p: P;
fn arg(i: u32) -> u32 { return p.a[i >> 2u][i & 3u]; }
fn argf(i: u32) -> f32 { return bitcast<f32>(arg(i)); }
fn dot4(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(a.x, b.x) + dot4I8Packed(a.y, b.y) + dot4I8Packed(a.z, b.z) + dot4I8Packed(a.w, b.w);
}
"#;

/// Quantises rows of `n` values to int8 blocks (see the module comment), the scale shared by
/// 128 values when `n` is a multiple of 128 and by 32 otherwise, after an optional
/// transform of the source. Args: rows, n, src_off, eps, flags (1: also write the transformed
/// values as f32 to `xn`), mode (0 copy, 1 RMS norm with weight `b`, 2 `silu(a) * b`, 3 the same with GELU).
/// One workgroup per row; eight values a lane, four lanes a block.
const QUANT: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> xq: array<u32>;
@group(0) @binding(4) var<storage, read_write> xm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> xn: array<vec4<f32>>;
var<workgroup> red: array<f32, 64>;
fn packq(q: vec4<i32>) -> u32 {
    let u = bitcast<vec4<u32>>(q) & vec4<u32>(255u);
    return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);
}
fn val(i: u32, base: u32, mode: u32, scale: f32) -> vec4<f32> {
    let v = a[base + i];
    if (mode == 1u) { return v * scale * b[i]; }
    if (mode == 2u) { return v / (vec4<f32>(1.0) + exp(-v)) * b[base + i]; }
    if (mode == 3u) {
        let u = 0.7978845608 * (v + 0.044715 * v * v * v);
        let e = exp(-2.0 * clamp(u, vec4<f32>(-20.0), vec4<f32>(20.0)));
        return 0.5 * v * (vec4<f32>(1.0) + (vec4<f32>(1.0) - e) / (vec4<f32>(1.0) + e)) * b[base + i];
    }
    return v;
}
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_id) sg: u32,
    @builtin(num_subgroups) sgs: u32,
) {
    let n = arg(1u);
    let n4 = n / 4u;
    let row = wid.x;
    let base = (arg(2u) + row * n) / 4u;
    let mode = arg(5u);
    var scale = 1.0;
    if (mode == 1u) {
        var s = 0.0;
        for (var i = li; i < n4; i = i + 256u) { let v = a[base + i]; s = s + dot(v, v); }
        s = subgroupAdd(s);
        if (lane == 0u) { red[sg] = s; }
        workgroupBarrier();
        var tot = 0.0;
        for (var i = 0u; i < sgs; i = i + 1u) { tot = tot + red[i]; }
        scale = inverseSqrt(tot / f32(n) + argf(3u));
    }
    let wide = (n % 128u) == 0u;
    let units = ((n / 8u) + 255u) / 256u * 256u;
    for (var u = li; u < units; u = u + 256u) {
        let valid = u * 8u < n;
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        if (valid) { v0 = val(2u * u, base, mode, scale); v1 = val(2u * u + 1u, base, mode, scale); }
        let a0 = abs(v0);
        let a1 = abs(v1);
        var mx = max(max(max(a0.x, a0.y), max(a0.z, a0.w)), max(max(a1.x, a1.y), max(a1.z, a1.w)));
        mx = max(mx, subgroupShuffleXor(mx, 1u));
        mx = max(mx, subgroupShuffleXor(mx, 2u));
        if (wide) {
            mx = max(mx, subgroupShuffleXor(mx, 4u));
            mx = max(mx, subgroupShuffleXor(mx, 8u));
        }
        let d = mx / 127.0;
        let inv = select(0.0, 127.0 / mx, mx > 0.0);
        let q0 = vec4<i32>(round(v0 * inv));
        let q1 = vec4<i32>(round(v1 * inv));
        let sm = q0.x + q0.y + q0.z + q0.w + q1.x + q1.y + q1.z + q1.w;
        let s16 = sm + subgroupShuffleXor(sm, 1u);
        let shi = subgroupShuffleXor(s16, 2u);
        if (valid) {
            xq[row * n4 + 2u * u] = packq(q0);
            xq[row * n4 + 2u * u + 1u] = packq(q1);
            if ((u & 3u) == 0u) {
                xm[row * (n / 32u) + (u >> 2u)] = vec4<f32>(d, f32(s16), f32(shi), 0.0);
            }
            if ((arg(4u) & 1u) != 0u) {
                xn[row * n4 + 2u * u] = v0;
                xn[row * n4 + 2u * u + 1u] = v1;
            }
        }
    }
}
"#;

/// `x += rmsnorm(src) * w` for rows of `n` values: the norm that closes a block before its
/// output joins the residual; with flag 1, `x = rmsnorm(src) * w` instead. Args: rows, n, flag,
/// eps. One workgroup of 256 lanes a row.
const POST_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_id) sg: u32,
    @builtin(num_subgroups) sgs: u32,
) {
    let n = arg(1u);
    let n4 = n / 4u;
    let base = wid.x * n4;
    var s = 0.0;
    for (var i = li; i < n4; i = i + 256u) { let v = a[base + i]; s = s + dot(v, v); }
    s = subgroupAdd(s);
    if (lane == 0u) { red[sg] = s; }
    workgroupBarrier();
    var tot = 0.0;
    for (var i = 0u; i < sgs; i = i + 1u) { tot = tot + red[i]; }
    let scale = inverseSqrt(tot / f32(n) + argf(3u));
    let keep = select(1.0, 0.0, arg(2u) == 1u);
    for (var i = li; i < n4; i = i + 256u) { x[base + i] = x[base + i] * keep + a[base + i] * scale * b[i]; }
}
"#;

/// The router of a mixture of experts: for each token, the logits of every expert over the
/// normed activations, their softmax, and the `K` best with their weights, written to `route` as
/// (expert, weight bits). One workgroup of 256 lanes a token; `256 / EP` lanes an expert, `EP`
/// the experts rounded up to a power of two. Args: tokens, hidden, experts, K, flag (1: the
/// chosen weights are renormalised).
const ROUTER: &str = r#"
@group(0) @binding(0) var<storage, read> xnorm: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> rw: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> route: array<vec2<u32>>;
var<workgroup> logit: array<f32, EP>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h4 = arg(1u) / 4u;
    let ne = arg(2u);
    let kk = arg(3u);
    let e = li / (256u / EP);
    let part = li % (256u / EP);
    var s = 0.0;
    if (e < ne) {
        for (var i = part; i < h4; i = i + 256u / EP) { s = s + dot(xnorm[wid.x * h4 + i], rw[e * h4 + i]); }
    }
    SHUFFLE_SUM
    if (part == 0u && e < ne) { logit[e] = s; }
    workgroupBarrier();
    if (li == 0u) {
        var m = -3.0e38;
        for (var j = 0u; j < ne; j = j + 1u) { m = max(m, logit[j]); }
        var tot = 0.0;
        for (var j = 0u; j < ne; j = j + 1u) {
            let p = exp(logit[j] - m);
            logit[j] = p;
            tot = tot + p;
        }
        var chosen = 0.0;
        var picks: array<u32, 16>;
        var weights: array<f32, 16>;
        for (var c = 0u; c < kk; c = c + 1u) {
            var best = -1.0;
            var at = 0u;
            for (var j = 0u; j < ne; j = j + 1u) {
                if (logit[j] > best) { best = logit[j]; at = j; }
            }
            picks[c] = at;
            weights[c] = best / tot;
            chosen = chosen + best / tot;
            logit[at] = -2.0;
        }
        for (var c = 0u; c < kk; c = c + 1u) {
            let w = select(weights[c], weights[c] / chosen, arg(4u) == 1u);
            route[wid.x * kk + c] = vec2<u32>(picks[c], bitcast<u32>(w));
        }
    }
}
"#;

/// The experts' outputs weighed into the residual: `x[t] += sum_c w[t, c] * y[t * K + c]`.
/// Args: tokens, hidden, -, K. One workgroup of 256 lanes a token.
const MOE_COMBINE: &str = r#"
@group(0) @binding(0) var<storage, read> yd: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> route: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h4 = arg(1u) / 4u;
    let kk = arg(3u);
    for (var i = li; i < h4; i = i + 256u) {
        var acc = x[wid.x * h4 + i];
        for (var c = 0u; c < kk; c = c + 1u) {
            acc = acc + bitcast<f32>(route[wid.x * kk + c].y) * yd[(wid.x * kk + c) * h4 + i];
        }
        x[wid.x * h4 + i] = acc;
    }
}
"#;

/// The mat-vec frame: `y[t, r] (+)= sum_k W[r, k] x[t, k] (+ bias[r])` for up to `MAXT` tokens.
/// Args: m, k, tokens, first token, y_off, flags (1 accumulate, 2 bias), -, y_stride. A workgroup is one
/// workgroup of `WG` lanes; sixteen lanes run a row, so it runs `WG / 16` rows. `ITEMS` and
/// `ITEM` are the format's: how many items a row has and what one item adds to `acc[t]`.
const MATVEC: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<WTYPE>;
@group(0) @binding(1) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> xm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> xf: array<vec4<f32>>;
const M4: vec4<u32> = vec4<u32>(0x0F0F0F0Fu);
fn kbyte(i: u32, a: u32, b: u32, c: u32) -> u32 {
    let wd = select(select(a, b, (i >> 2u) == 1u), c, (i >> 2u) == 2u);
    return (wd >> ((i & 3u) * 8u)) & 255u;
}
fn q4k_sm(s: u32, a: u32, b: u32, c: u32) -> vec2<u32> {
    let lo = vec2<u32>(kbyte(s, a, b, c) & 63u, kbyte(s + 4u, a, b, c) & 63u);
    let h4 = kbyte(s + 4u, a, b, c);
    let hi = vec2<u32>((h4 & 15u) | ((kbyte(s & 3u, a, b, c) >> 6u) << 4u), (h4 >> 4u) | ((kbyte(s, a, b, c) >> 6u) << 4u));
    return select(hi, lo, s < 4u);
}
fn sgn6(u: vec4<u32>) -> vec4<u32> {
    return ((u | vec4<u32>(0x80808080u)) - vec4<u32>(0x20202020u)) ^ vec4<u32>(0x80808080u);
}
@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let m = arg(0u);
    let k = arg(1u);
    let nt = arg(2u);
    let tb = arg(3u);
    let raw = (wid.x + wid.y * 32768u) * (WG / LPRu) + lane / LPRu;
    let row = min(raw, m - 1u);
    var acc: array<f32, MAXT>;
    for (var it = lane & (LPRu - 1u); it < ITEMS; it = it + LPRu) {
        ITEM
    }
    for (var t = 0u; t < nt; t = t + 1u) {
        var v = acc[t];
        SHUFFLES
        if (raw < m && (lane & (LPRu - 1u)) == 0u) {
            let flags = arg(5u);
            if ((flags & 2u) != 0u) { v = v + bias[row]; }
            let at = arg(4u) + (tb + t) * arg(7u) + row;
            if ((flags & 1u) != 0u) { v = v + y[at]; }
            y[at] = v;
        }
    }
}
"#;

/// Q4_K, as stored: 9 vectors a block of 256 (d and dmin, 12 bytes of 6-bit scales and mins,
/// 128 bytes of nibbles). An item is sixteen bytes of nibbles of one group of 64 values: the
/// low nibbles are half a sub-block, the high ones half of the next.
const Q4K_ITEM: (&str, &str, &str) = (
    "vec4<u32>",
    "(k / 256u) * 8u",
    r#"
        let b = it >> 3u;
        let j = (it >> 1u) & 3u;
        let h = it & 1u;
        let sb = (row * (k / 256u) + b) * 9u;
        let head = w[sb];
        let wv = w[sb + 1u + 2u * j + h];
        let dm = unpack2x16float(head.x);
        let sm0 = q4k_sm(2u * j, head.y, head.z, head.w);
        let sm1 = q4k_sm(2u * j + 1u, head.y, head.z, head.w);
        let dl0 = dm.x * f32(sm0.x);
        let ml0 = dm.y * f32(sm0.y);
        let dl1 = dm.x * f32(sm1.x);
        let ml1 = dm.y * f32(sm1.y);
        let lo = wv & M4;
        let hi = (wv >> vec4<u32>(4u)) & M4;
        let blk = b * 8u + 2u * j;
        for (var t = 0u; t < nt; t = t + 1u) {
            let xi = (tb + t) * (k / 16u) + blk * 2u + h;
            let xa = xq[xi];
            let xb = xq[xi + 2u];
            let ma = xm[(tb + t) * (k / 32u) + blk];
            let mb = xm[(tb + t) * (k / 32u) + blk + 1u];
            let sa = dot4(lo, xa);
            let sc = dot4(hi, xb);
            acc[t] = acc[t] + ma.x * (dl0 * f32(sa) - ml0 * select(ma.y, ma.z, h == 1u))
                + mb.x * (dl1 * f32(sc) - ml1 * select(mb.y, mb.z, h == 1u));
        }
"#,
);

/// Q6_K, repacked to 14 vectors a block of 256: ql (128 bytes), qh (64), the int8 scales (16),
/// d (a word, padded). An item is sixteen lanes of each of the four quarters of one half block:
/// 64 values on four scales.
const Q6K_ITEM: (&str, &str, &str) = (
    "vec4<u32>",
    "(k / 256u) * 4u",
    r#"
        let b = it >> 2u;
        let n = (it >> 1u) & 1u;
        let hh = it & 1u;
        let sb = (row * (k / 256u) + b) * 14u;
        let wa = w[sb + n * 4u + hh];
        let wb = w[sb + n * 4u + 2u + hh];
        let wh = w[sb + 8u + n * 2u + hh];
        let scw = w[sb + 12u];
        let d = unpack2x16float(w[sb + 13u].x).x;
        let sa = select(scw.xy, scw.zw, n == 1u);
        let s0 = d * f32(extractBits(bitcast<i32>(sa.x), hh * 8u, 8u));
        let s1 = d * f32(extractBits(bitcast<i32>(sa.x), (2u + hh) * 8u, 8u));
        let s2 = d * f32(extractBits(bitcast<i32>(sa.y), hh * 8u, 8u));
        let s3 = d * f32(extractBits(bitcast<i32>(sa.y), (2u + hh) * 8u, 8u));
        let m3 = vec4<u32>(0x03030303u);
        let u1 = sgn6((wa & M4) | ((wh & m3) << vec4<u32>(4u)));
        let u2 = sgn6((wb & M4) | (((wh >> vec4<u32>(2u)) & m3) << vec4<u32>(4u)));
        let u3 = sgn6(((wa >> vec4<u32>(4u)) & M4) | (((wh >> vec4<u32>(4u)) & m3) << vec4<u32>(4u)));
        let u4 = sgn6(((wb >> vec4<u32>(4u)) & M4) | (((wh >> vec4<u32>(6u)) & m3) << vec4<u32>(4u)));
        let blk = b * 8u + n * 4u;
        for (var t = 0u; t < nt; t = t + 1u) {
            let xi = (tb + t) * (k / 16u) + blk * 2u + hh;
            let mi = (tb + t) * (k / 32u) + blk;
            acc[t] = acc[t]
                + s0 * xm[mi].x * f32(dot4(u1, xq[xi]))
                + s1 * xm[mi + 1u].x * f32(dot4(u2, xq[xi + 2u]))
                + s2 * xm[mi + 2u].x * f32(dot4(u3, xq[xi + 4u]))
                + s3 * xm[mi + 3u].x * f32(dot4(u4, xq[xi + 6u]));
        }
"#,
);

/// Q8_0, repacked to 9 words a block of 32: d, then the 32 int8. An item is sixteen of them.
const Q8_0_ITEM: (&str, &str, &str) = (
    "u32",
    "(k / 32u) * 2u",
    r#"
        let blk = it >> 1u;
        let h = it & 1u;
        let sb = (row * (k / 32u) + blk) * 9u;
        let d = unpack2x16float(w[sb]).x;
        let wv = vec4<u32>(w[sb + 1u + h * 4u], w[sb + 2u + h * 4u], w[sb + 3u + h * 4u], w[sb + 4u + h * 4u]);
        for (var t = 0u; t < nt; t = t + 1u) {
            acc[t] = acc[t] + d * xm[(tb + t) * (k / 32u) + blk].x * f32(dot4(wv, xq[(tb + t) * (k / 16u) + blk * 2u + h]));
        }
"#,
);

/// F16, as stored: eight values a vector, read against the f32 activations. An item is a vector.
const F16_ITEM: (&str, &str, &str) = (
    "vec4<u32>",
    "k / 8u",
    r#"
        let wv = w[row * (k / 8u) + it];
        let w0 = unpack2x16float(wv.x);
        let w1 = unpack2x16float(wv.y);
        let w2 = unpack2x16float(wv.z);
        let w3 = unpack2x16float(wv.w);
        for (var t = 0u; t < nt; t = t + 1u) {
            let xi = (tb + t) * (k / 4u) + it * 2u;
            let x0 = xf[xi];
            let x1 = xf[xi + 1u];
            acc[t] = acc[t] + dot(w0, x0.xy) + dot(w1, x0.zw) + dot(w2, x1.xy) + dot(w3, x1.zw);
        }
"#,
);

/// The mat-mat frame for up to 32 tokens per tile row: `y[t, r] (+)= sum_k W[r, k] x[t, k]
/// (+ bias[r])` on a tile of 64 rows and 32 tokens a workgroup of 128 lanes, each lane 4 rows by
/// 4 tokens. A step stages `BK` columns of the weights as int8 words with their scales and the
/// same columns of the activations, then takes the integer dot products. `DOTS` is the format's
/// way of weighing a block's products; the K-quants keep their integer scales and minimums in
/// `si`, their super-block scales in `sd`, and apply the activations' scale once a step.
/// Args as the mat-vec's.
const MATMAT: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<WTYPE>;
@group(0) @binding(1) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> xm: array<vec4<f32>>;
const BK: u32 = BKV;
const NB: u32 = BKV / 32u;
const SV: u32 = BKV / 16u + 1u;
const M4: vec4<u32> = vec4<u32>(0x0F0F0F0Fu);
var<workgroup> sw: array<vec4<u32>, SWN>;
var<workgroup> si: array<vec2<i32>, 512>;
var<workgroup> sd: array<vec2<f32>, 64>;
var<workgroup> ss: array<vec4<f32>, SSN>;
var<workgroup> sx: array<vec4<u32>, SXN>;
var<workgroup> sm: array<vec4<f32>, SMN>;
var<workgroup> sxs: array<i32, SMN>;
fn kbyte(i: u32, a: u32, b: u32, c: u32) -> u32 {
    let wd = select(select(a, b, (i >> 2u) == 1u), c, (i >> 2u) == 2u);
    return (wd >> ((i & 3u) * 8u)) & 255u;
}
fn q4k_sm(s: u32, a: u32, b: u32, c: u32) -> vec2<u32> {
    let lo = vec2<u32>(kbyte(s, a, b, c) & 63u, kbyte(s + 4u, a, b, c) & 63u);
    let h4 = kbyte(s + 4u, a, b, c);
    let hi = vec2<u32>((h4 & 15u) | ((kbyte(s & 3u, a, b, c) >> 6u) << 4u), (h4 >> 4u) | ((kbyte(s, a, b, c) >> 6u) << 4u));
    return select(hi, lo, s < 4u);
}
fn sgn6(u: vec4<u32>) -> vec4<u32> {
    return ((u | vec4<u32>(0x80808080u)) - vec4<u32>(0x20202020u)) ^ vec4<u32>(0x80808080u);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let m = arg(0u);
    let k = arg(1u);
    let nt = arg(2u);
    let row0 = wid.x * 64u;
    let tok0 = wid.y * TOKT;
    let ty = li >> 3u;
    let tx = li & 7u;
    var acc: array<f32, NACC>;
    for (var kk = 0u; kk < k; kk = kk + BK) {
        {
            LOADW
        }
        for (var e = li; e < TOKT * (BK / 16u); e = e + 128u) {
            let tk = e / (BK / 16u);
            let pv = e % (BK / 16u);
            var v = vec4<u32>(0u);
            if (tok0 + tk < nt) { v = xq[(tok0 + tk) * (k / 16u) + kk / 16u + pv]; }
            sx[tk * SV + pv] = v;
        }
        for (var e = li; e < TOKT * NB; e = e + 128u) {
            let tk = e / NB;
            let bb = e % NB;
            var v = vec4<f32>(0.0);
            if (tok0 + tk < nt) { v = xm[(tok0 + tk) * (k / 32u) + kk / 32u + bb]; }
            sm[e] = v;
            sxs[e] = i32(v.y + v.z);
        }
        workgroupBarrier();
        DOTS
        workgroupBarrier();
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let row = row0 + ty * 4u + r;
        for (var t = 0u; t < TTu; t = t + 1u) {
            let tk = tok0 + t * 8u + tx;
            if (row < m && tk < nt) {
                var v = acc[r * TTu + t];
                let flags = arg(5u);
                if ((flags & 2u) != 0u) { v = v + bias[row]; }
                let at = arg(4u) + tk * arg(7u) + row;
                if ((flags & 1u) != 0u) { v = v + y[at]; }
                y[at] = v;
            }
        }
    }
}
"#;

/// The products of a step of formats with a float scale to each block of 32 (Q8_0).
const DOTS_BLOCKS: &str = r#"
        for (var blk = 0u; blk < NB; blk = blk + 1u) {
            var isum: array<i32, NACC>;
            for (var hf = 0u; hf < 2u; hf = hf + 1u) {
                var wr: array<vec4<u32>, 4>;
                var xt: array<vec4<u32>, TT>;
                for (var r = 0u; r < 4u; r = r + 1u) { wr[r] = sw[(ty * 4u + r) * SV + blk * 2u + hf]; }
                for (var t = 0u; t < TTu; t = t + 1u) { xt[t] = sx[(t * 8u + tx) * SV + blk * 2u + hf]; }
                for (var r = 0u; r < 4u; r = r + 1u) {
                    for (var t = 0u; t < TTu; t = t + 1u) {
                        isum[r * TTu + t] = isum[r * TTu + t] + dot4(wr[r], xt[t]);
                    }
                }
            }
            for (var r = 0u; r < 4u; r = r + 1u) {
                let sv = ss[(ty * 4u + r) * NB + blk];
                for (var t = 0u; t < TTu; t = t + 1u) {
                    let xv = sm[(t * 8u + tx) * NB + blk];
                    acc[r * TTu + t] = acc[r * TTu + t] + sv.x * xv.x * f32(isum[r * TTu + t]);
                }
            }
        }
"#;

/// The products of a Q4_K step of four blocks: integer scales and minimums weigh the products
/// and the activations' sums, and the floats come in once.
const DOTS_Q4K: &str = r#"
        var ai: array<i32, NACC>;
        var bi: array<i32, NACC>;
        for (var blk = 0u; blk < NB; blk = blk + 1u) {
            var isum: array<i32, NACC>;
            for (var hf = 0u; hf < 2u; hf = hf + 1u) {
                var wr: array<vec4<u32>, 4>;
                var xt: array<vec4<u32>, TT>;
                for (var r = 0u; r < 4u; r = r + 1u) { wr[r] = sw[(ty * 4u + r) * SV + blk * 2u + hf]; }
                for (var t = 0u; t < TTu; t = t + 1u) { xt[t] = sx[(t * 8u + tx) * SV + blk * 2u + hf]; }
                for (var r = 0u; r < 4u; r = r + 1u) {
                    for (var t = 0u; t < TTu; t = t + 1u) {
                        isum[r * TTu + t] = isum[r * TTu + t] + dot4(wr[r], xt[t]);
                    }
                }
            }
            var xs: array<i32, TT>;
            for (var t = 0u; t < TTu; t = t + 1u) { xs[t] = sxs[(t * 8u + tx) * NB + blk]; }
            for (var r = 0u; r < 4u; r = r + 1u) {
                let sv = si[(ty * 4u + r) * 4u + blk];
                for (var t = 0u; t < TTu; t = t + 1u) {
                    ai[r * TTu + t] = ai[r * TTu + t] + sv.x * isum[r * TTu + t];
                    bi[r * TTu + t] = bi[r * TTu + t] + sv.y * xs[t];
                }
            }
        }
        for (var r = 0u; r < 4u; r = r + 1u) {
            let dd = sd[ty * 4u + r];
            for (var t = 0u; t < TTu; t = t + 1u) {
                let xd = sm[(t * 8u + tx) * NB].x;
                acc[r * TTu + t] = acc[r * TTu + t] + xd * (dd.x * f32(ai[r * TTu + t]) - dd.y * f32(bi[r * TTu + t]));
            }
        }
"#;

/// The products of a Q6_K step of four blocks: an integer scale to each sixteen values.
const DOTS_Q6K: &str = r#"
        var ai: array<i32, NACC>;
        for (var blk = 0u; blk < NB; blk = blk + 1u) {
            var i0: array<i32, NACC>;
            var i1: array<i32, NACC>;
            {
                var wr: array<vec4<u32>, 4>;
                var wr2: array<vec4<u32>, 4>;
                var xt: array<vec4<u32>, TT>;
                var xt2: array<vec4<u32>, TT>;
                for (var r = 0u; r < 4u; r = r + 1u) {
                    wr[r] = sw[(ty * 4u + r) * SV + blk * 2u];
                    wr2[r] = sw[(ty * 4u + r) * SV + blk * 2u + 1u];
                }
                for (var t = 0u; t < TTu; t = t + 1u) {
                    xt[t] = sx[(t * 8u + tx) * SV + blk * 2u];
                    xt2[t] = sx[(t * 8u + tx) * SV + blk * 2u + 1u];
                }
                for (var r = 0u; r < 4u; r = r + 1u) {
                    for (var t = 0u; t < TTu; t = t + 1u) {
                        i0[r * TTu + t] = i0[r * TTu + t] + dot4(wr[r], xt[t]);
                        i1[r * TTu + t] = i1[r * TTu + t] + dot4(wr2[r], xt2[t]);
                    }
                }
            }
            for (var r = 0u; r < 4u; r = r + 1u) {
                let s0 = si[(ty * 4u + r) * 8u + 2u * blk].x;
                let s1 = si[(ty * 4u + r) * 8u + 2u * blk + 1u].x;
                for (var t = 0u; t < TTu; t = t + 1u) {
                    ai[r * TTu + t] = ai[r * TTu + t] + s0 * i0[r * TTu + t] + s1 * i1[r * TTu + t];
                }
            }
        }
        for (var r = 0u; r < 4u; r = r + 1u) {
            let dd = sd[ty * 4u + r];
            for (var t = 0u; t < TTu; t = t + 1u) {
                let xd = sm[(t * 8u + tx) * NB].x;
                acc[r * TTu + t] = acc[r * TTu + t] + xd * dd.x * f32(ai[r * TTu + t]);
            }
        }
"#;

/// The staging of a Q4_K step of 128 columns: a lane takes the 32 bytes of nibbles of one group
/// of 64 (its low nibbles are one block of 32, its high ones the next).
const Q4K_LOAD: (&str, u32, &str, &str) = (
    "vec4<u32>",
    128,
    "Q4K",
    r#"
            let row = li >> 1u;
            let jj = li & 1u;
            let gr = min(row0 + row, m - 1u);
            let b = kk >> 8u;
            let h2 = (kk >> 7u) & 1u;
            let jg = 2u * h2 + jj;
            let sb = (gr * (k / 256u) + b) * 9u;
            let a = w[sb + 1u + 2u * jg];
            let c = w[sb + 2u + 2u * jg];
            let o = row * SV + jj * 4u;
            sw[o] = a & M4;
            sw[o + 1u] = c & M4;
            sw[o + 2u] = (a >> vec4<u32>(4u)) & M4;
            sw[o + 3u] = (c >> vec4<u32>(4u)) & M4;
            let head = w[sb];
            let s0 = q4k_sm(4u * h2 + 2u * jj, head.y, head.z, head.w);
            let s1 = q4k_sm(4u * h2 + 2u * jj + 1u, head.y, head.z, head.w);
            si[row * 4u + 2u * jj] = vec2<i32>(i32(s0.x), i32(s0.y));
            si[row * 4u + 2u * jj + 1u] = vec2<i32>(i32(s1.x), i32(s1.y));
            if (jj == 0u) { sd[row] = unpack2x16float(head.x); }
"#,
);

/// The staging of a Q6_K step of 128 columns, one half block: a lane takes sixteen columns of
/// each of its four quarters.
const Q6K_LOAD: (&str, u32, &str, &str) = (
    "vec4<u32>",
    128,
    "Q6K",
    r#"
            let row = li >> 1u;
            let hh = li & 1u;
            let gr = min(row0 + row, m - 1u);
            let b = kk >> 8u;
            let n = (kk >> 7u) & 1u;
            let sb = (gr * (k / 256u) + b) * 14u;
            let wa = w[sb + n * 4u + hh];
            let wb = w[sb + n * 4u + 2u + hh];
            let wh = w[sb + 8u + n * 2u + hh];
            let m3 = vec4<u32>(0x03030303u);
            let u1 = sgn6((wa & M4) | ((wh & m3) << vec4<u32>(4u)));
            let u2 = sgn6((wb & M4) | (((wh >> vec4<u32>(2u)) & m3) << vec4<u32>(4u)));
            let u3 = sgn6(((wa >> vec4<u32>(4u)) & M4) | (((wh >> vec4<u32>(4u)) & m3) << vec4<u32>(4u)));
            let u4 = sgn6(((wb >> vec4<u32>(4u)) & M4) | (((wh >> vec4<u32>(6u)) & m3) << vec4<u32>(4u)));
            let o = row * SV + hh;
            sw[o] = u1;
            sw[o + 2u] = u2;
            sw[o + 4u] = u3;
            sw[o + 6u] = u4;
            if (hh == 0u) {
                let scw = w[sb + 12u];
                sd[row] = vec2<f32>(unpack2x16float(w[sb + 13u].x).x, 0.0);
                let sa = select(scw.xy, scw.zw, n == 1u);
                for (var i = 0u; i < 4u; i = i + 1u) {
                    let word = bitcast<i32>(select(sa.x, sa.y, i >= 2u));
                    let sh = (i & 1u) * 16u;
                    si[row * 8u + 2u * i] = vec2<i32>(extractBits(word, sh, 8u), 0);
                    si[row * 8u + 2u * i + 1u] = vec2<i32>(extractBits(word, sh + 8u, 8u), 0);
                }
            }
"#,
);

/// The staging of a Q8_0 step of 32 columns: a lane takes half a block.
const Q8_0_LOAD: (&str, u32, &str, &str) = (
    "u32",
    32,
    "BLOCKS",
    r#"
            let row = li >> 1u;
            let hf = li & 1u;
            let gr = min(row0 + row, m - 1u);
            let sb = (gr * (k / 32u) + kk / 32u) * 9u;
            sw[row * SV + hf] = vec4<u32>(w[sb + 1u + hf * 4u], w[sb + 2u + hf * 4u], w[sb + 3u + hf * 4u], w[sb + 4u + hf * 4u]);
            if (hf == 0u) {
                let d = unpack2x16float(w[sb]).x;
                ss[row] = vec4<f32>(d, d, 0.0, 0.0);
            }
"#,
);

/// The F16 mat-mat: 64 rows by 32 tokens a workgroup of 128 lanes, each lane 4 by 4, against
/// the f32 activations; a step stages 32 columns. Args as the mat-vec's.
const MATMAT_F16: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> xf: array<vec4<f32>>;
var<workgroup> sw: array<f32, 2112>;
var<workgroup> sx: array<f32, 1056>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let m = arg(0u);
    let k = arg(1u);
    let nt = arg(2u);
    let row0 = wid.x * 64u;
    let tok0 = wid.y * 32u;
    let ty = li >> 3u;
    let tx = li & 7u;
    var acc: array<f32, 16>;
    for (var kk = 0u; kk < k; kk = kk + 32u) {
        for (var e = li; e < 256u; e = e + 128u) {
            let r = e >> 2u;
            let c = e & 3u;
            let gr = min(row0 + r, m - 1u);
            let v = w[gr * (k / 8u) + kk / 8u + c];
            let a = unpack2x16float(v.x);
            let b = unpack2x16float(v.y);
            let cc = unpack2x16float(v.z);
            let d = unpack2x16float(v.w);
            let o = r * 33u + c * 8u;
            sw[o] = a.x;
            sw[o + 1u] = a.y;
            sw[o + 2u] = b.x;
            sw[o + 3u] = b.y;
            sw[o + 4u] = cc.x;
            sw[o + 5u] = cc.y;
            sw[o + 6u] = d.x;
            sw[o + 7u] = d.y;
        }
        for (var e = li; e < 256u; e = e + 128u) {
            let tk = e >> 3u;
            let c = e & 7u;
            var v = vec4<f32>(0.0);
            if (tok0 + tk < nt) { v = xf[(tok0 + tk) * (k / 4u) + kk / 4u + c]; }
            let o = tk * 33u + c * 4u;
            sx[o] = v.x;
            sx[o + 1u] = v.y;
            sx[o + 2u] = v.z;
            sx[o + 3u] = v.w;
        }
        workgroupBarrier();
        for (var i = 0u; i < 32u; i = i + 1u) {
            var wr: array<f32, 4>;
            var xt: array<f32, 4>;
            for (var r = 0u; r < 4u; r = r + 1u) { wr[r] = sw[(ty * 4u + r) * 33u + i]; }
            for (var t = 0u; t < 4u; t = t + 1u) { xt[t] = sx[(tx * 4u + t) * 33u + i]; }
            for (var r = 0u; r < 4u; r = r + 1u) {
                for (var t = 0u; t < 4u; t = t + 1u) {
                    acc[r * 4u + t] = acc[r * 4u + t] + wr[r] * xt[t];
                }
            }
        }
        workgroupBarrier();
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let row = row0 + ty * 4u + r;
        for (var t = 0u; t < 4u; t = t + 1u) {
            let tk = tok0 + tx * 4u + t;
            if (row < m && tk < nt) {
                var v = acc[r * 4u + t];
                let flags = arg(5u);
                if ((flags & 2u) != 0u) { v = v + bias[row]; }
                let at = arg(4u) + tk * arg(7u) + row;
                if ((flags & 1u) != 0u) { v = v + y[at]; }
                y[at] = v;
            }
        }
    }
}
"#;

/// Causal attention of a block of query rows for a key-value head: a row is a token of one of
/// the `n_head / n_kv` heads that share the head, `R` of them a workgroup of 128 lanes. The keys
/// go through shared memory in tiles of `TK` for all rows, with a running softmax per row. The
/// scores of a tile are `RR` rows by `KK` keys a lane; the weighted values are four rows by
/// eight dimensions a lane. Args: n_head, n_kv, hd, scale, -, tokens, window (0 for none). One workgroup per (key-value
/// head, block of `R / (n_head / n_kv)` tokens); `hd` is at most `HD` and a multiple of 8.
const ATTENTION_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> vc: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> step: array<u32>;
const R: u32 = RROWS;
const TK: u32 = TKEYS;
const HV: u32 = HDV / 8u;
const QS: u32 = HDV / 4u + 1u;
const KS: u32 = HV + 1u;
const PS: u32 = TKEYS + 1u;
const KG: u32 = TKEYS / KKEYS;
const DG: u32 = HV;
const RP: u32 = RROWS / (128u / HV);
var<workgroup> qs: array<vec4<f32>, QSN>;
var<workgroup> kt: array<vec4<u32>, KSN>;
var<workgroup> vt: array<vec4<u32>, KSN>;
var<workgroup> pt: array<f32, PSN>;
var<workgroup> mrow: array<f32, RROWS>;
var<workgroup> lrow: array<f32, RROWS>;
var<workgroup> arow: array<f32, RROWS>;
fn lo4(w: vec4<u32>) -> array<vec4<f32>, 2> {
    return array<vec4<f32>, 2>(
        vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)),
        vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)),
    );
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nh = arg(0u);
    let nkv = arg(1u);
    let hd = arg(2u);
    let scale = argf(3u);
    let ntok = arg(5u);
    let g = wid.x;
    let gs = nh / nkv;
    let per = R / gs;
    let tb = wid.y * per;
    let hv = hd / 8u;
    let p0 = step[0];
    for (var e = li; e < R * hv * 2u; e = e + 128u) {
        let r = e / (hv * 2u);
        let i = e % (hv * 2u);
        let tt = tb + r / gs;
        var v = vec4<f32>(0.0);
        if (tt < ntok) { v = q[((tt * nh + g * gs + r % gs) * hd) / 4u + i]; }
        qs[r * QS + i] = v;
    }
    if (li < R) { mrow[li] = -3.0e38; lrow[li] = 0.0; }
    var o: array<vec4<f32>, ODIM>;
    let kmax = p0 + min(tb + per, ntok);
    let win = arg(6u);
    var kstart = 0u;
    if (win > 0u && p0 + tb + 1u > win) { kstart = p0 + tb + 1u - win; }
    let kg = li % KG;
    let rg = li / KG;
    let dg = li % DG;
    let rp = li / DG;
    for (var k0 = kstart; k0 < kmax; k0 = k0 + TK) {
        workgroupBarrier();
        for (var e = li; e < TK * hv; e = e + 128u) {
            let r = e / hv;
            let c = e % hv;
            let j = min(k0 + r, kmax - 1u);
            let base = (j * nkv + g) * hv + c;
            kt[r * KS + c] = kc[base];
            vt[r * KS + c] = vc[base];
        }
        workgroupBarrier();
        var s: array<f32, SCORES>;
        for (var i = 0u; i < hv; i = i + 1u) {
            var kf: array<array<vec4<f32>, 2>, KKEYS>;
            for (var kk = 0u; kk < KKEYS; kk = kk + 1u) { kf[kk] = lo4(kt[(kg + KG * kk) * KS + i]); }
            for (var rr = 0u; rr < RROWS_LANE; rr = rr + 1u) {
                let qrow = (rg * RROWS_LANE + rr) * QS + 2u * i;
                let qa = qs[qrow];
                let qb = qs[qrow + 1u];
                for (var kk = 0u; kk < KKEYS; kk = kk + 1u) {
                    s[rr * KKEYS + kk] = s[rr * KKEYS + kk] + dot(kf[kk][0], qa) + dot(kf[kk][1], qb);
                }
            }
        }
        for (var rr = 0u; rr < RROWS_LANE; rr = rr + 1u) {
            let r = rg * RROWS_LANE + rr;
            let tq = tb + r / gs;
            for (var kk = 0u; kk < KKEYS; kk = kk + 1u) {
                let key = kg + KG * kk;
                let live = (k0 + key) <= (p0 + tq) && tq < ntok && (win == 0u || (k0 + key) + win > p0 + tq);
                pt[r * PS + key] = select(-3.0e38, s[rr * KKEYS + kk] * scale, live);
            }
        }
        workgroupBarrier();
        {
            let r = li / LPR;
            let part = li % LPR;
            var mt = -3.0e38;
            for (var j = 0u; j < TK / LPR; j = j + 1u) { mt = max(mt, pt[r * PS + part * (TK / LPR) + j]); }
            SHUFFLE_MAX
            let mold = mrow[r];
            let mnew = max(mold, mt);
            let alpha = exp(mold - mnew);
            var ps = 0.0;
            for (var j = 0u; j < TK / LPR; j = j + 1u) {
                let at = r * PS + part * (TK / LPR) + j;
                let sc = pt[at];
                let p = select(0.0, exp(sc - mnew), sc > -1.0e38);
                pt[at] = p;
                ps = ps + p;
            }
            SHUFFLE_SUM
            if (part == 0u) {
                lrow[r] = lrow[r] * alpha + ps;
                mrow[r] = mnew;
                arow[r] = alpha;
            }
        }
        workgroupBarrier();
        for (var rr = 0u; rr < RP; rr = rr + 1u) {
            let a = arow[rp * RP + rr];
            o[2u * rr] = o[2u * rr] * a;
            o[2u * rr + 1u] = o[2u * rr + 1u] * a;
        }
        for (var key = 0u; key < TK; key = key + 1u) {
            let vf = lo4(vt[key * KS + dg]);
            for (var rr = 0u; rr < RP; rr = rr + 1u) {
                let p = pt[(rp * RP + rr) * PS + key];
                o[2u * rr] = o[2u * rr] + p * vf[0];
                o[2u * rr + 1u] = o[2u * rr + 1u] + p * vf[1];
            }
        }
    }
    if (dg < hv) {
        for (var rr = 0u; rr < RP; rr = rr + 1u) {
            let r = rp * RP + rr;
            let tq = tb + r / gs;
            if (tq < ntok) {
                let inv = 1.0 / lrow[r];
                let ob = (tq * nh + g * gs + r % gs) * hd + 8u * dg;
                let a = o[2u * rr] * inv;
                let b = o[2u * rr + 1u] * inv;
                out[ob] = a.x;
                out[ob + 1u] = a.y;
                out[ob + 2u] = a.z;
                out[ob + 3u] = a.w;
                out[ob + 4u] = b.x;
                out[ob + 5u] = b.y;
                out[ob + 6u] = b.z;
                out[ob + 7u] = b.w;
            }
        }
    }
}
"#;

/// Query rows a workgroup of the prompt attention kernel takes.
pub(super) const ATTN_ROWS: u32 = 16;

/// The WGSL of the prompt attention kernel for heads of at most `hd` lanes.
fn attention_tiled(hd: u32) -> String {
    let r = ATTN_ROWS;
    let tk = if hd > 128 { 16 } else { 32 };
    let rr = 2;
    let kk = r * tk / 128 / rr;
    let lpr = 128 / r;
    let shuffles = |op: &str, var: &str| -> String {
        (0..lpr.ilog2())
            .map(|b| {
                format!(
                    "{var} = {op}({var}, subgroupShuffleXor({var}, {}u));\n",
                    1 << b
                )
            })
            .collect()
    };
    ATTENTION_TILED
        .replace(
            "SHUFFLE_MAX",
            &shuffles("max", "mt").replace("= max(mt,", "= max(mt,"),
        )
        .replace(
            "SHUFFLE_SUM",
            &(0..lpr.ilog2())
                .map(|b| format!("ps = ps + subgroupShuffleXor(ps, {}u);\n", 1 << b))
                .collect::<String>(),
        )
        .replace("ODIM", &(2 * r / (128 / (hd / 8))).to_string())
        .replace("RROWS_LANE", &rr.to_string())
        .replace("RROWS", &r.to_string())
        .replace("TKEYS", &tk.to_string())
        .replace("KKEYS", &kk.to_string())
        .replace("SCORES", &(rr * kk).to_string())
        .replace("HDV", &hd.to_string())
        .replace("QSN", &(r * (hd / 4 + 1)).to_string())
        .replace("KSN", &(tk * (hd / 8 + 1)).to_string())
        .replace("PSN", &(r * (tk + 1)).to_string())
        .replace("LPR", &lpr.to_string())
}

/// The per-head preparation of Q or of K and V: an optional RMS norm over the head with its
/// weight, then RoPE at the token's position. Q lands in an f32 buffer, K and V in the f16
/// cache at the token's row. Args: heads, hd, has_norm, eps, rope_dim, adjacent, cache_heads,
/// half_table (columns of the cos and sin tables). One workgroup per (head, token).
const PREP: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> nw: array<f32>;
@group(0) @binding(3) var<storage, read> vsrc: array<f32>;
@group(0) @binding(5) var<storage, read> step: array<u32>;
@group(0) @binding(6) var<storage, read> cosv: array<f32>;
@group(0) @binding(7) var<storage, read> sinv: array<f32>;
var<workgroup> a: array<f32, 512>;
var<workgroup> o: array<f32, 512>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let heads = arg(0u);
    let hd = arg(1u);
    let h = wid.x;
    let t = wid.y;
    let pos = step[0] + t;
    let sb = (t * heads + h) * hd;
    var s = 0.0;
    for (var i = li; i < hd; i = i + 64u) { let v = src[sb + i]; a[i] = v; s = s + v * v; }
    red[li] = s;
    workgroupBarrier();
    for (var k = 32u; k > 0u; k = k >> 1u) {
        if (li < k) { red[li] = red[li] + red[li + k]; }
        workgroupBarrier();
    }
    if (arg(2u) != 0u) {
        let r = inverseSqrt(red[0] / f32(hd) + argf(3u));
        for (var i = li; i < hd; i = i + 64u) { a[i] = a[i] * r * nw[i]; }
    }
    workgroupBarrier();
    let rd = arg(4u);
    let half = rd / 2u;
    let tb = pos * arg(7u);
    for (var i = li; i < hd; i = i + 64u) {
        var v = a[i];
        if (i < rd) {
            if (arg(5u) != 0u) {
                let c = cosv[tb + i / 2u];
                let sn = sinv[tb + i / 2u];
                if ((i & 1u) == 0u) { v = a[i] * c - a[i + 1u] * sn; } else { v = a[i] * c + a[i - 1u] * sn; }
            } else if (i < half) {
                v = a[i] * cosv[tb + i] - a[i + half] * sinv[tb + i];
            } else {
                v = a[i] * cosv[tb + i - half] + a[i - half] * sinv[tb + i - half];
            }
        }
        o[i] = v;
    }
    workgroupBarrier();
    STORE
}
"#;

const PREP_Q_STORE: &str = r#"
    _ = vsrc[0];
    for (var i = li; i < hd; i = i + 64u) { qdst[sb + i] = o[i]; }
"#;

const PREP_Q_DST: &str = r#"
@group(0) @binding(2) var<storage, read_write> qdst: array<f32>;
"#;

const PREP_KV_STORE: &str = r#"
    let row = (pos * arg(6u) + h) * (hd / 2u);
    for (var i = li; i < hd / 2u; i = i + 64u) {
        kc[row + i] = pack2x16float(vec2<f32>(o[2u * i], o[2u * i + 1u]));
        vc[row + i] = pack2x16float(vec2<f32>(vsrc[sb + 2u * i], vsrc[sb + 2u * i + 1u]));
    }
"#;

const PREP_KV_DST: &str = r#"
@group(0) @binding(2) var<storage, read_write> kc: array<u32>;
@group(0) @binding(4) var<storage, read_write> vc: array<u32>;
"#;

/// Causal attention of one head of one token over a slice of the f16 cache: keys `[0, pos]`
/// cut into `splits` slices, one workgroup of 256 lanes per (head, token, slice). Sixteen groups
/// of sixteen lanes take the slice's keys in turn; a group's lanes each own eight dimensions,
/// dot a key with the query by a butterfly and keep a running softmax and output of their own,
/// which the groups then merge. Writes the slice's unnormalised output, its maximum score and
/// its sum of exponentials to `part` (rows of `hd + 2`); [`ATTENTION_JOIN`] puts the slices
/// together. Args: n_head, n_kv, hd, scale, -, splits, window (0 for none). `hd` is at most 128
/// times `CPL` and a multiple of 8; a lane owns `CPL` chunks of eight dimensions.
const ATTENTION: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> vc: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(5) var<storage, read> step: array<u32>;
var<workgroup> qs: array<vec4<f32>, QSN>;
var<workgroup> msh: array<f32, 16>;
var<workgroup> lsh: array<f32, 16>;
var<workgroup> acc_sh: array<f32, ASN>;
fn kdot(kw: vec4<u32>, qa: vec4<f32>, qb: vec4<f32>) -> f32 {
    return dot(unpack2x16float(kw.x), qa.xy) + dot(unpack2x16float(kw.y), qa.zw)
        + dot(unpack2x16float(kw.z), qb.xy) + dot(unpack2x16float(kw.w), qb.zw);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nh = arg(0u);
    let nkv = arg(1u);
    let hd = arg(2u);
    let scale = argf(3u);
    let splits = arg(5u);
    let h = wid.x;
    let t = wid.y;
    let part_ix = wid.z;
    let g = h / (nh / nkv);
    let total = step[0] + t + 1u;
    let win = arg(6u);
    var kst = 0u;
    if (win > 0u && total > win) { kst = total - win; }
    let span = total - kst;
    let lo = kst + span * part_ix / splits;
    let hi = kst + span * (part_ix + 1u) / splits;
    let qb = (t * nh + h) * hd;
    let hv = hd / 8u;
    if (li < hd / 4u) { qs[li] = vec4<f32>(q[qb + li * 4u], q[qb + li * 4u + 1u], q[qb + li * 4u + 2u], q[qb + li * 4u + 3u]); }
    workgroupBarrier();
    let sub = li & 15u;
    let slot = li >> 4u;
    var qa: array<vec4<f32>, CPL>;
    var qb2: array<vec4<f32>, CPL>;
    var cs: array<u32, CPL>;
    for (var cc = 0u; cc < CPL; cc = cc + 1u) {
        let ch = sub + 16u * cc;
        let on = ch < hv;
        let c = min(ch, hv - 1u);
        cs[cc] = c;
        qa[cc] = select(vec4<f32>(0.0), qs[2u * c], on);
        qb2[cc] = select(vec4<f32>(0.0), qs[2u * c + 1u], on);
    }
    var m = -3.0e38;
    var l = 0.0;
    var acc: array<f32, ACCN>;
    for (var j0 = lo; j0 < hi; j0 = j0 + 64u) {
        var s: array<f32, 4>;
        var vw: array<vec4<u32>, VWN>;
        var ok: array<bool, 4>;
        for (var u = 0u; u < 4u; u = u + 1u) {
            let j = j0 + slot + 16u * u;
            let jj = min(j, hi - 1u);
            ok[u] = j < hi;
            var sd = 0.0;
            for (var cc = 0u; cc < CPL; cc = cc + 1u) {
                sd = sd + kdot(kc[(jj * nkv + g) * hv + cs[cc]], qa[cc], qb2[cc]);
                vw[u * CPL + cc] = vc[(jj * nkv + g) * hv + cs[cc]];
            }
            s[u] = sd;
        }
        var mt = m;
        for (var u = 0u; u < 4u; u = u + 1u) {
            var x = s[u];
            x = x + subgroupShuffleXor(x, 8u);
            x = x + subgroupShuffleXor(x, 4u);
            x = x + subgroupShuffleXor(x, 2u);
            x = x + subgroupShuffleXor(x, 1u);
            s[u] = x * scale;
            mt = max(mt, select(-3.0e38, s[u], ok[u]));
        }
        let alpha = exp(m - mt);
        l = l * alpha;
        for (var e = 0u; e < ACCN; e = e + 1u) { acc[e] = acc[e] * alpha; }
        for (var u = 0u; u < 4u; u = u + 1u) {
            let p = select(0.0, exp(s[u] - mt), ok[u]);
            l = l + p;
            for (var ch = 0u; ch < CPL; ch = ch + 1u) {
                let w = vw[u * CPL + ch];
                let a = unpack2x16float(w.x);
                let b = unpack2x16float(w.y);
                let cc = unpack2x16float(w.z);
                let dd = unpack2x16float(w.w);
                acc[ch * 8u] = acc[ch * 8u] + p * a.x;
                acc[ch * 8u + 1u] = acc[ch * 8u + 1u] + p * a.y;
                acc[ch * 8u + 2u] = acc[ch * 8u + 2u] + p * b.x;
                acc[ch * 8u + 3u] = acc[ch * 8u + 3u] + p * b.y;
                acc[ch * 8u + 4u] = acc[ch * 8u + 4u] + p * cc.x;
                acc[ch * 8u + 5u] = acc[ch * 8u + 5u] + p * cc.y;
                acc[ch * 8u + 6u] = acc[ch * 8u + 6u] + p * dd.x;
                acc[ch * 8u + 7u] = acc[ch * 8u + 7u] + p * dd.y;
            }
        }
        m = mt;
    }
    if (sub == 0u) {
        msh[slot] = m;
        lsh[slot] = l;
    }
    for (var ch = 0u; ch < CPL; ch = ch + 1u) {
        if (sub + 16u * ch < hv) {
            for (var e = 0u; e < 8u; e = e + 1u) { acc_sh[slot * hd + (sub + 16u * ch) * 8u + e] = acc[ch * 8u + e]; }
        }
    }
    workgroupBarrier();
    var mm = -3.0e38;
    for (var k = 0u; k < 16u; k = k + 1u) { mm = max(mm, msh[k]); }
    let pb = (((t * nh + h) * splits) + part_ix) * (hd + 2u);
    if (li < hd) {
        var o = 0.0;
        for (var k = 0u; k < 16u; k = k + 1u) { o = o + exp(msh[k] - mm) * acc_sh[k * hd + li]; }
        part[pb + li] = o;
    }
    if (li == 0u) {
        var ls = 0.0;
        for (var k = 0u; k < 16u; k = k + 1u) { ls = ls + exp(msh[k] - mm) * lsh[k]; }
        part[pb + hd] = mm;
        part[pb + hd + 1u] = ls;
    }
}
"#;

/// The WGSL of the decode attention for heads of at most `128 * cpl` lanes.
fn attention(cpl: u32) -> String {
    ATTENTION
        .replace("QSN", &(32 * cpl).to_string())
        .replace("ASN", &(2048 * cpl).to_string())
        .replace("ACCN", &(8 * cpl).to_string())
        .replace("VWN", &(4 * cpl).to_string())
        .replace("CPL", &cpl.to_string())
}

/// The slices of [`ATTENTION`] put together: `out = sum_s exp(m_s - M) o_s / sum_s exp(m_s - M)
/// l_s`. Args: n_head, n_kv, hd, -, -, splits. One workgroup of 256 lanes per (head, token).
const ATTENTION_JOIN: &str = r#"
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nh = arg(0u);
    let hd = arg(2u);
    let splits = arg(5u);
    let base = ((wid.y * nh + wid.x) * splits) * (hd + 2u);
    var m = -3.0e38;
    for (var s = 0u; s < splits; s = s + 1u) { m = max(m, part[base + s * (hd + 2u) + hd]); }
    var l = 0.0;
    var o = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let at = base + s * (hd + 2u);
        let wgt = exp(part[at + hd] - m);
        l = l + wgt * part[at + hd + 1u];
        if (li < hd) { o = o + wgt * part[at + li]; }
    }
    if (li < hd) { out[(wid.y * nh + wid.x) * hd + li] = o / l; }
}
"#;

/// The router kernel for `experts` experts.
pub(super) fn router_kernel(experts: usize) -> &'static str {
    match experts {
        0..=16 => "router16",
        17..=32 => "router32",
        33..=64 => "router64",
        65..=128 => "router128",
        _ => "router256",
    }
}

/// The experts' version of mat-vec kernel `mv`.
pub(super) fn expert_kernel(mv: &str) -> &'static str {
    match mv {
        "mv_q4k" => "mx_q4k",
        "mv_q6k" => "mx_q6k",
        "mv_q8_0" => "mx_q8_0",
        _ => "mx_f16",
    }
}

/// The WGSL of kernel `name` for workgroups of `wg` lanes.
fn source(name: &str, wg: u32) -> anyhow::Result<String> {
    let lanes = lanes_per_row(name);
    let shuffles: String = (0..lanes.ilog2())
        .rev()
        .map(|b| format!("        v = v + subgroupShuffleXor(v, {}u);\n", 1 << b))
        .collect();
    let experts = name.starts_with("mx_");
    let matvec = |(wtype, items, item): (&str, &str, &str)| {
        let frame = if experts {
            MATVEC
                .replace("let nt = arg(2u);", "let nt = 1u;")
                .replace(
                    "let tb = arg(3u);",
                    "let pair = wid.z;\n    let rt = route[pair];\n    let tb = select(pair / arg(2u), pair, arg(6u) == 1u);",
                )
                .replace(
                    "let row = min(raw, m - 1u);",
                    "let row = rt.x * m + min(raw, m - 1u);",
                )
                .replace("let at = arg(4u) + (tb + t) * arg(7u) + row;", "let at = pair * m + raw;")
                .replace(
                    "@group(0) @binding(5) var<storage, read> xf: array<vec4<f32>>;",
                    "@group(0) @binding(5) var<storage, read> xf: array<vec4<f32>>;\n@group(0) @binding(6) var<storage, read> route: array<vec2<u32>>;",
                )
        } else {
            MATVEC.to_string()
        };
        frame
            .replace("WTYPE", wtype)
            .replace("SHUFFLES", &shuffles)
            .replace("LPR", &lanes.to_string())
            .replace("ITEMS", items)
            .replace("ITEM", item)
            .replace("MAXT", &super::stack::MV_TOKENS.to_string())
            .replace("WG", &wg.to_string())
    };
    let matmat = |(wtype, bk, dots, load): (&str, u32, &str, &str)| {
        let s = bk / 16 + 1;
        let dots = match dots {
            "Q4K" => DOTS_Q4K,
            "Q6K" => DOTS_Q6K,
            _ => DOTS_BLOCKS,
        };
        MATMAT
            .replace("DOTS", dots)
            .replace("WTYPE", wtype)
            .replace("LOADW", load)
            .replace("BKV", &bk.to_string())
            .replace("SWN", &(64 * s).to_string())
            .replace("SSN", &(64 * (bk / 32)).to_string())
            .replace("SXN", &(MM_TOKENS * s).to_string())
            .replace("SMN", &(MM_TOKENS * (bk / 32)).to_string())
            .replace("NACC", &(4 * MM_TT).to_string())
            .replace("TOKT", &format!("{}u", MM_TOKENS))
            .replace("TTu", &format!("{}u", MM_TT))
            .replace("TT", &MM_TT.to_string())
    };
    if let Some(body) = super::coop::body(name) {
        return Ok(format!(
            "enable f16;\nenable wgpu_cooperative_matrix;\n{HEAD}{body}"
        ));
    }
    if let Some(ep) = name
        .strip_prefix("router")
        .and_then(|n| n.parse::<u32>().ok())
    {
        let lpe = 256 / ep;
        let sum: String = (0..lpe.ilog2())
            .map(|b| format!("    s = s + subgroupShuffleXor(s, {}u);\n", 1 << b))
            .collect();
        let body = ROUTER
            .replace("SHUFFLE_SUM", &sum)
            .replace("EP", &ep.to_string());
        return Ok(format!("{HEAD}{body}"));
    }
    let body = match name {
        "quant" => QUANT.to_string(),
        "post_norm" => POST_NORM.to_string(),
        "mv_q4k" | "mx_q4k" => matvec(Q4K_ITEM),
        "mv_q6k" | "mx_q6k" => matvec(Q6K_ITEM),
        "mv_q8_0" | "mx_q8_0" => matvec(Q8_0_ITEM),
        "mv_f16" | "mx_f16" => matvec(F16_ITEM),
        "moe_combine" => MOE_COMBINE.to_string(),
        "mm_q4k" => matmat(Q4K_LOAD),
        "mm_q6k" => matmat(Q6K_LOAD),
        "mm_q8_0" => matmat(Q8_0_LOAD),
        "mm_f16" => MATMAT_F16.to_string(),
        "attn_tiled" => attention_tiled(128),
        "attn_tiled_wide" => attention_tiled(256),
        "attention_wide" => attention(2),
        "prep_q" => format!("{PREP_Q_DST}{}", PREP.replace("STORE", PREP_Q_STORE)),
        "prep_kv" => format!("{PREP_KV_DST}{}", PREP.replace("STORE", PREP_KV_STORE)),
        "attention" => attention(1),
        "attention_join" => ATTENTION_JOIN.to_string(),
        other => anyhow::bail!("vulkan: no kernel {other}"),
    };
    Ok(format!("{HEAD}{body}"))
}

/// The SPIR-V words of kernel `name` for workgroups of `wg` lanes.
pub(super) fn spirv(name: &str, wg: u32) -> anyhow::Result<Vec<u32>> {
    let src = source(name, wg)?;
    let module = naga::front::wgsl::parse_str(&src)
        .map_err(|e| anyhow::anyhow!("vulkan: {name}: {}", e.emit_to_string(&src)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::IMMEDIATES
            | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP,
    )
    .subgroup_stages(naga::valid::ShaderStages::COMPUTE)
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)
    .map_err(|e| anyhow::anyhow!("vulkan: {name}: {}", e.emit_to_string(&src)))?;
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        force_loop_bounding: false,
        emit_int_div_checks: false,
        ..Default::default()
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline))
        .map_err(|e| anyhow::anyhow!("vulkan: {name}: {e}"))
}
