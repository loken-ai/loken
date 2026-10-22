//! The cooperative-matrix mat-mat kernels: a workgroup of four subgroups takes 64 rows of the
//! weights against 64 tokens, dequantises a step of `BK` columns of both into f16 tiles in shared
//! memory, multiplies 16 by 16 tiles into f32 accumulators, and writes the result through a
//! shared f32 tile so the bias and the accumulation apply as in the integer kernels.
//!
//! Weights are dequantised to f16 with their scales folded in; activations are the int8 blocks
//! times their scale (or the f32 copy for F16 weights).

pub(super) const ROWS: u32 = 64;
pub(super) const TOKENS: u32 = 64;
/// Columns a step stages.
const BK: u32 = 64;

/// The frame. `STAGEW` writes this lane's share of the weight tile `sa`, `STAGEX` the
/// activation tile `sb`. Args as the mat-vec's.
const FRAME: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<WTYPE>;
@group(0) @binding(1) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> xm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> xf: array<vec4<f32>>;
const BK: u32 = 64u;
const AS: u32 = 72u;
const M4: vec4<u32> = vec4<u32>(0x0F0F0F0Fu);
var<workgroup> sa: array<f16, 4608>;
var<workgroup> sb: array<f16, 4608>;
var<workgroup> sc: array<f32, 4096>;
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
fn sb4(x: u32, i: u32) -> f32 { return f32(extractBits(bitcast<i32>(x), i * 8u, 8u)); }
fn put4(o: u32, x: u32, s: f32) {
    sa[o] = f16(sb4(x, 0u) * s);
    sa[o + 1u] = f16(sb4(x, 1u) * s);
    sa[o + 2u] = f16(sb4(x, 2u) * s);
    sa[o + 3u] = f16(sb4(x, 3u) * s);
}
fn putx(o: u32, x: u32, s: f32) {
    sb[o] = f16(sb4(x, 0u) * s);
    sb[o + 1u] = f16(sb4(x, 1u) * s);
    sb[o + 2u] = f16(sb4(x, 2u) * s);
    sb[o + 3u] = f16(sb4(x, 3u) * s);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let m = arg(0u);
    let k = arg(1u);
    let nt = arg(2u);
    let row0 = wid.x * 64u;
    let tok0 = wid.y * 64u;
    let lr = li >> 1u;
    let hv = li & 1u;
    let gr = min(row0 + lr, m - 1u);
    let wave = li >> 5u;
    let rw = (wave & 1u) * 32u;
    let tw = (wave >> 1u) * 32u;
    var c00 = coop_mat16x16<f32, C>();
    var c01 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>();
    var c11 = coop_mat16x16<f32, C>();
    for (var kk = 0u; kk < k; kk = kk + BK) {
        {
            STAGEW
        }
        {
            STAGEX
        }
        workgroupBarrier();
        for (var s = 0u; s < BK; s = s + 16u) {
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&sa[rw * AS + s], AS);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&sa[(rw + 16u) * AS + s], AS);
            let b0 = coopLoad<coop_mat16x16<f16, B>>(&sb[tw * AS + s], AS);
            let b1 = coopLoad<coop_mat16x16<f16, B>>(&sb[(tw + 16u) * AS + s], AS);
            c00 = coopMultiplyAdd(a0, b0, c00);
            c01 = coopMultiplyAdd(a0, b1, c01);
            c10 = coopMultiplyAdd(a1, b0, c10);
            c11 = coopMultiplyAdd(a1, b1, c11);
        }
        workgroupBarrier();
    }
    coopStore(c00, &sc[tw * 64u + rw], 64u);
    coopStore(c01, &sc[(tw + 16u) * 64u + rw], 64u);
    coopStore(c10, &sc[tw * 64u + rw + 16u], 64u);
    coopStore(c11, &sc[(tw + 16u) * 64u + rw + 16u], 64u);
    workgroupBarrier();
    let flags = arg(5u);
    for (var e = li; e < 4096u; e = e + 128u) {
        let tk = tok0 + (e >> 6u);
        let row = row0 + (e & 63u);
        if (row < m && tk < nt) {
            var v = sc[e];
            if ((flags & 2u) != 0u) { v = v + bias[row]; }
            let at = arg(4u) + tk * arg(7u) + row;
            if ((flags & 1u) != 0u) { v = v + y[at]; }
            y[at] = v;
        }
    }
}
"#;

/// Activations from the int8 blocks: a lane takes 32 columns of one token.
const STAGEX_Q: &str = r#"
            let tk = tok0 + lr;
            let o = lr * AS + hv * 32u;
            if (tk < nt) {
                let d = xm[tk * (k / 32u) + kk / 32u + hv].x;
                let q0 = xq[tk * (k / 16u) + kk / 16u + 2u * hv];
                let q1 = xq[tk * (k / 16u) + kk / 16u + 2u * hv + 1u];
                putx(o, q0.x, d); putx(o + 4u, q0.y, d); putx(o + 8u, q0.z, d); putx(o + 12u, q0.w, d);
                putx(o + 16u, q1.x, d); putx(o + 20u, q1.y, d); putx(o + 24u, q1.z, d); putx(o + 28u, q1.w, d);
            } else {
                for (var i = 0u; i < 32u; i = i + 1u) { sb[o + i] = f16(0.0); }
            }
"#;

/// Activations from the f32 copy: a lane takes 32 columns of one token.
const STAGEX_F: &str = r#"
            let tk = tok0 + lr;
            let o = lr * AS + hv * 32u;
            for (var j = 0u; j < 8u; j = j + 1u) {
                var v = vec4<f32>(0.0);
                if (tk < nt) { v = xf[tk * (k / 4u) + kk / 4u + hv * 8u + j]; }
                sb[o + 4u * j] = f16(v.x);
                sb[o + 4u * j + 1u] = f16(v.y);
                sb[o + 4u * j + 2u] = f16(v.z);
                sb[o + 4u * j + 3u] = f16(v.w);
            }
"#;

/// A Q4_K group of 64 columns: a lane takes sixteen bytes, their low nibbles one block of 32 and
/// the high ones the next.
const STAGEW_Q4K: (&str, &str) = (
    "vec4<u32>",
    r#"
            let jg = (kk >> 6u) & 3u;
            let sbk = (gr * (k / 256u) + (kk >> 8u)) * 9u;
            let a = w[sbk + 1u + 2u * jg + hv];
            let head = w[sbk];
            let dd = unpack2x16float(head.x);
            let s0 = q4k_sm(2u * jg, head.y, head.z, head.w);
            let s1 = q4k_sm(2u * jg + 1u, head.y, head.z, head.w);
            let d0 = dd.x * f32(s0.x);
            let m0 = dd.y * f32(s0.y);
            let d1 = dd.x * f32(s1.x);
            let m1 = dd.y * f32(s1.y);
            let o = lr * AS + hv * 16u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let lo = (a[j] & 0x0F0F0F0Fu);
                let hi = (a[j] >> 4u) & 0x0F0F0F0Fu;
                for (var i = 0u; i < 4u; i = i + 1u) {
                    sa[o + 4u * j + i] = f16(d0 * f32((lo >> (8u * i)) & 255u) - m0);
                    sa[o + 32u + 4u * j + i] = f16(d1 * f32((hi >> (8u * i)) & 255u) - m1);
                }
            }
"#,
);

/// A Q6_K quarter of a half block (64 columns): a lane takes sixteen columns of each of its two
/// thirty-twos.
const STAGEW_Q6K: (&str, &str) = (
    "vec4<u32>",
    r#"
            let n = (kk >> 7u) & 1u;
            let qq = (kk >> 6u) & 1u;
            let sbk = (gr * (k / 256u) + (kk >> 8u)) * 14u;
            let wa = w[sbk + n * 4u + hv];
            let wb = w[sbk + n * 4u + 2u + hv];
            let wh = w[sbk + 8u + n * 2u + hv];
            let m3 = vec4<u32>(0x03030303u);
            let sh = vec4<u32>(4u * qq);
            let u1 = sgn6(((wa >> sh) & M4) | (((wh >> sh) & m3) << vec4<u32>(4u)));
            let u2 = sgn6(((wb >> sh) & M4) | (((wh >> (sh + vec4<u32>(2u))) & m3) << vec4<u32>(4u)));
            let scw = w[sbk + 12u];
            let sa2 = select(scw.xy, scw.zw, n == 1u);
            let d = unpack2x16float(w[sbk + 13u].x).x;
            let c0 = 4u * qq + hv;
            let sc0 = f32(extractBits(bitcast<i32>(select(sa2.x, sa2.y, c0 >= 4u)), (c0 & 3u) * 8u, 8u));
            let c1 = c0 + 2u;
            let sc1 = f32(extractBits(bitcast<i32>(select(sa2.x, sa2.y, c1 >= 4u)), (c1 & 3u) * 8u, 8u));
            let o = lr * AS + hv * 16u;
            put4(o, u1.x, d * sc0);
            put4(o + 4u, u1.y, d * sc0);
            put4(o + 8u, u1.z, d * sc0);
            put4(o + 12u, u1.w, d * sc0);
            put4(o + 32u, u2.x, d * sc1);
            put4(o + 36u, u2.y, d * sc1);
            put4(o + 40u, u2.z, d * sc1);
            put4(o + 44u, u2.w, d * sc1);
"#,
);

/// Two Q8_0 blocks (64 columns): a lane takes one.
const STAGEW_Q8_0: (&str, &str) = (
    "u32",
    r#"
            let sbk = (gr * (k / 32u) + kk / 32u + hv) * 9u;
            let d = unpack2x16float(w[sbk]).x;
            let o = lr * AS + hv * 32u;
            for (var j = 0u; j < 8u; j = j + 1u) { put4(o + 4u * j, w[sbk + 1u + j], d); }
"#,
);

/// F16 weights: a lane takes 32 columns.
const STAGEW_F16: (&str, &str) = (
    "vec4<u32>",
    r#"
            let o = lr * AS + hv * 32u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let v = w[gr * (k / 8u) + kk / 8u + hv * 4u + j];
                let a = unpack2x16float(v.x);
                let b = unpack2x16float(v.y);
                let c = unpack2x16float(v.z);
                let d = unpack2x16float(v.w);
                sa[o + 8u * j] = f16(a.x);
                sa[o + 8u * j + 1u] = f16(a.y);
                sa[o + 8u * j + 2u] = f16(b.x);
                sa[o + 8u * j + 3u] = f16(b.y);
                sa[o + 8u * j + 4u] = f16(c.x);
                sa[o + 8u * j + 5u] = f16(c.y);
                sa[o + 8u * j + 6u] = f16(d.x);
                sa[o + 8u * j + 7u] = f16(d.y);
            }
"#,
);

/// The WGSL (without the common header) of cooperative kernel `name`, if it is one.
pub(super) fn body(name: &str) -> Option<String> {
    let (weights, stagex) = match name {
        "cm_q4k" => (STAGEW_Q4K, STAGEX_Q),
        "cm_q6k" => (STAGEW_Q6K, STAGEX_Q),
        "cm_q8_0" => (STAGEW_Q8_0, STAGEX_Q),
        "cm_f16" => (STAGEW_F16, STAGEX_F),
        _ => return None,
    };
    Some(
        FRAME
            .replace("WTYPE", weights.0)
            .replace("STAGEW", weights.1)
            .replace("STAGEX", stagex),
    )
}

/// The mat-mat of the same format for a device with cooperative matrices.
pub(super) fn kernel_for(mat_mat: &'static str) -> &'static str {
    match mat_mat {
        "mm_q4k" => "cm_q4k",
        "mm_q6k" => "cm_q6k",
        "mm_q8_0" => "cm_q8_0",
        _ => "cm_f16",
    }
}

#[allow(dead_code)]
const _: () = assert!(BK == 64);
