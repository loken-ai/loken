//! The kernels, in WGSL, compiled to SPIR-V with naga.
//!
//! Every kernel shares one layout: storage bindings 0..8 of set 0 and 64 bytes of immediates,
//! read as sixteen `u32` (a float travels as its bits). Weights are read as whole words: the
//! stack repacks the block formats whose blocks are not a multiple of four bytes.

/// The immediates: sixteen words whose meaning each kernel states.
const HEAD: &str = r#"
struct P { a: array<vec4<u32>, 4> }
var<immediate> p: P;
fn arg(i: u32) -> u32 { return p.a[i >> 2u][i & 3u]; }
fn argf(i: u32) -> f32 { return bitcast<f32>(arg(i)); }
"#;

/// Root-mean-square norm of `rows` rows of `n`: `y = x / rms(x) * w`.
/// Args: rows, n, x_off, y_off, eps, w_off. One workgroup per row.
const RMS_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = arg(1u);
    let row = wid.x;
    let xb = arg(2u) + row * n;
    var s = 0.0;
    for (var i = li; i < n; i = i + 256u) { let v = x[xb + i]; s = s + v * v; }
    red[li] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (li < k) { red[li] = red[li] + red[li + k]; }
        workgroupBarrier();
    }
    let r = inverseSqrt(red[0] / f32(n) + argf(4u));
    let yb = arg(3u) + row * n;
    let wb = arg(5u);
    for (var i = li; i < n; i = i + 256u) { y[yb + i] = x[xb + i] * r * w[wb + i]; }
}
"#;

/// The mat-vec frame: `y[t, r] (+)= sum_k W[r, k] x[t, k] (+ bias[r])` for up to eight tokens.
/// Args: m, k, tokens, x_off, y_off, flags (1 accumulate, 2 bias), x_stride, y_stride.
/// A workgroup of 64 runs rows `2 g` and `2 g + 1` of group `g = wid.x + wid.y * 32768`, a
/// subgroup to a row at a time; the lanes of a subgroup split the row and add up with the
/// subgroup sum. `ITEMS` and `ITEM` are the format's: how many items a row has and what one
/// item adds to `acc`.
const MATVEC: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) lanes: u32,
    @builtin(subgroup_id) sg: u32,
    @builtin(num_subgroups) sgs: u32,
) {
    let m = arg(0u);
    let k = arg(1u);
    let nt = arg(2u);
    let xo = arg(3u);
    let xs = arg(6u);
    let items = ITEMS;
    for (var slot = sg; slot < 2u; slot = slot + sgs) {
        let raw = (wid.x + wid.y * 32768u) * 2u + slot;
        let row = min(raw, m - 1u);
        var acc: array<f32, 8>;
        for (var t = 0u; t < 8u; t = t + 1u) { acc[t] = 0.0; }
        for (var it = lane; it < items; it = it + lanes) {
            ITEM
        }
        for (var t = 0u; t < nt; t = t + 1u) {
            let v0 = subgroupAdd(acc[t]);
            if (raw < m && lane == 0u) {
                var v = v0;
                let flags = arg(5u);
                if ((flags & 2u) != 0u) { v = v + bias[row]; }
                let at = arg(4u) + t * arg(7u) + row;
                if ((flags & 1u) != 0u) { v = v + y[at]; }
                y[at] = v;
            }
        }
    }
}
"#;

/// Q4_K, as stored: 36 words a block of 256 (d and dmin, 12 bytes of 6-bit scales and mins,
/// 128 bytes of nibbles). An item is one word of nibbles: four values of sub-block `2 jp` in
/// its low nibbles, four of sub-block `2 jp + 1` in its high ones. Neighbouring lanes read
/// neighbouring words.
const Q4K_ITEM: (&str, &str) = (
    "(k / 256u) * 32u",
    r#"
        let b = it >> 5u;
        let wq = it & 31u;
        let jp = wq >> 3u;
        let base = (row * (k / 256u) + b) * 36u;
        let dm = unpack2x16float(w[base]);
        let s0 = w[base + 1u];
        let s1 = w[base + 2u];
        let s2 = w[base + 3u];
        var sc: vec2<u32>;
        var mn: vec2<u32>;
        if (jp < 2u) {
            sc = vec2<u32>(extractBits(s0, jp * 16u, 6u), extractBits(s0, jp * 16u + 8u, 6u));
            mn = vec2<u32>(extractBits(s1, jp * 16u, 6u), extractBits(s1, jp * 16u + 8u, 6u));
        } else {
            let o = (jp - 2u) * 16u;
            let hi = vec2<u32>(extractBits(s2, o, 8u), extractBits(s2, o + 8u, 8u));
            sc = (hi & vec2<u32>(15u)) | (vec2<u32>(extractBits(s0, o + 6u, 2u), extractBits(s0, o + 14u, 2u)) << vec2<u32>(4u));
            mn = (hi >> vec2<u32>(4u)) | (vec2<u32>(extractBits(s1, o + 6u, 2u), extractBits(s1, o + 14u, 2u)) << vec2<u32>(4u));
        }
        let dl = dm.x * vec2<f32>(sc);
        let ml = dm.y * vec2<f32>(mn);
        let v = w[base + 4u + wq];
        let ql = vec4<f32>(f32(v & 15u), f32((v >> 8u) & 15u), f32((v >> 16u) & 15u), f32((v >> 24u) & 15u));
        let qh = vec4<f32>(f32((v >> 4u) & 15u), f32((v >> 12u) & 15u), f32((v >> 20u) & 15u), f32(v >> 28u));
        let col = b * 256u + jp * 64u + (wq & 7u) * 4u;
        for (var t = 0u; t < nt; t = t + 1u) {
            let xb = xo + t * xs + col;
            let lo = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
            let hi = vec4<f32>(x[xb + 32u], x[xb + 33u], x[xb + 34u], x[xb + 35u]);
            acc[t] = acc[t] + dl.x * dot(ql, lo) - ml.x * (lo.x + lo.y + lo.z + lo.w)
                + dl.y * dot(qh, hi) - ml.y * (hi.x + hi.y + hi.z + hi.w);
        }
"#,
);

/// Q6_K, repacked to 53 words a block: ql (32 words), qh (16), the int8 scales (4), d. An item
/// is four lanes `l .. l + 4` of one half block: a word of each half of its ql, a word of its
/// qh, sixteen values. Neighbouring lanes read neighbouring words.
const Q6K_ITEM: (&str, &str) = (
    "(k / 256u) * 16u",
    r#"
        let b = it >> 4u;
        let n = (it >> 3u) & 1u;
        let lq = it & 7u;
        let base = (row * (k / 256u) + b) * 53u;
        let d = unpack2x16float(w[base + 52u]).x;
        // The scales of lanes `l .. l + 4` are `is`, `is + 2`, `is + 4` and `is + 6`, with
        // `is % 4` 0 or 1: the first two in one word, the last two in the next.
        let is = n * 8u + lq / 4u;
        let sw0 = bitcast<i32>(w[base + 48u + is / 4u]);
        let sw1 = bitcast<i32>(w[base + 49u + is / 4u]);
        let sh = (is % 4u) * 8u;
        let sc0 = d * f32(extractBits(sw0, sh, 8u));
        let sc2 = d * f32(extractBits(sw0, sh + 16u, 8u));
        let sc4 = d * f32(extractBits(sw1, sh, 8u));
        let sc6 = d * f32(extractBits(sw1, sh + 16u, 8u));
        let wa = w[base + n * 16u + lq];
        let wb = w[base + n * 16u + 8u + lq];
        let wh = w[base + 32u + n * 8u + lq];
        let sv = vec4<u32>(0u, 8u, 16u, 24u);
        let av = (vec4<u32>(wa) >> sv) & vec4<u32>(255u);
        let bv = (vec4<u32>(wb) >> sv) & vec4<u32>(255u);
        let hv = (vec4<u32>(wh) >> sv) & vec4<u32>(255u);
        let m32 = vec4<i32>(32);
        let q1 = vec4<f32>(vec4<i32>((av & vec4<u32>(15u)) | ((hv & vec4<u32>(3u)) << vec4<u32>(4u))) - m32);
        let q2 = vec4<f32>(vec4<i32>((bv & vec4<u32>(15u)) | (((hv >> vec4<u32>(2u)) & vec4<u32>(3u)) << vec4<u32>(4u))) - m32);
        let q3 = vec4<f32>(vec4<i32>((av >> vec4<u32>(4u)) | (((hv >> vec4<u32>(4u)) & vec4<u32>(3u)) << vec4<u32>(4u))) - m32);
        let q4 = vec4<f32>(vec4<i32>((bv >> vec4<u32>(4u)) | (((hv >> vec4<u32>(6u)) & vec4<u32>(3u)) << vec4<u32>(4u))) - m32);
        let col = b * 256u + n * 128u + lq * 4u;
        for (var t = 0u; t < nt; t = t + 1u) {
            let xb = xo + t * xs + col;
            let x1 = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
            let x2 = vec4<f32>(x[xb + 32u], x[xb + 33u], x[xb + 34u], x[xb + 35u]);
            let x3 = vec4<f32>(x[xb + 64u], x[xb + 65u], x[xb + 66u], x[xb + 67u]);
            let x4 = vec4<f32>(x[xb + 96u], x[xb + 97u], x[xb + 98u], x[xb + 99u]);
            acc[t] = acc[t] + sc0 * dot(q1, x1) + sc2 * dot(q2, x2) + sc4 * dot(q3, x3) + sc6 * dot(q4, x4);
        }
"#,
);

/// Q8_0, repacked to 9 words a block of 32: d, then the 32 int8. An item is one word of
/// four values.
const Q8_0_ITEM: (&str, &str) = (
    "k / 4u",
    r#"
        let b = it >> 3u;
        let base = (row * (k / 32u) + b) * 9u;
        let d = unpack2x16float(w[base]).x;
        let v = bitcast<i32>(w[base + 1u + (it & 7u)]);
        let q = vec4<f32>(f32(extractBits(v, 0u, 8u)), f32(extractBits(v, 8u, 8u)), f32(extractBits(v, 16u, 8u)), f32(extractBits(v, 24u, 8u)));
        let col = it * 4u;
        for (var t = 0u; t < nt; t = t + 1u) {
            let xb = xo + t * xs + col;
            acc[t] = acc[t] + d * dot(q, vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]));
        }
"#,
);

/// F16, as stored: two values a word. An item is a word.
const F16_ITEM: (&str, &str) = (
    "k / 2u",
    r#"
        let v = unpack2x16float(w[row * (k / 2u) + it]);
        for (var t = 0u; t < nt; t = t + 1u) {
            let xb = xo + t * xs + it * 2u;
            acc[t] = acc[t] + dot(v, vec2<f32>(x[xb], x[xb + 1u]));
        }
"#,
);

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

/// Causal attention of one head of one token over the f16 cache: keys `[0, pos]`, the head's
/// KV group, scores in a global scratch row. Args: n_head, n_kv, hd, scale, ctx.
const ATTENTION: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<u32>;
@group(0) @binding(2) var<storage, read> vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> sc: array<f32>;
@group(0) @binding(5) var<storage, read> step: array<u32>;
var<workgroup> qs: array<f32, 512>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nh = arg(0u);
    let nkv = arg(1u);
    let hd = arg(2u);
    let scale = argf(3u);
    let ctx = arg(4u);
    let h = wid.x;
    let t = wid.y;
    let g = h / (nh / nkv);
    let len = step[0] + t + 1u;
    let qb = (t * nh + h) * hd;
    let sb = (t * nh + h) * ctx;
    let hw = hd / 2u;
    for (var i = li; i < hd; i = i + 128u) { qs[i] = q[qb + i]; }
    workgroupBarrier();
    var mx = -3.0e38;
    for (var j = li; j < len; j = j + 128u) {
        let kb = (j * nkv + g) * hw;
        var d = 0.0;
        for (var i = 0u; i < hw; i = i + 1u) {
            d = d + dot(unpack2x16float(kc[kb + i]), vec2<f32>(qs[2u * i], qs[2u * i + 1u]));
        }
        let s = d * scale;
        sc[sb + j] = s;
        mx = max(mx, s);
    }
    red[li] = mx;
    workgroupBarrier();
    for (var k = 64u; k > 0u; k = k >> 1u) {
        if (li < k) { red[li] = max(red[li], red[li + k]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = li; j < len; j = j + 128u) {
        let e = exp(sc[sb + j] - m);
        sc[sb + j] = e;
        sum = sum + e;
    }
    red[li] = sum;
    storageBarrier();
    workgroupBarrier();
    for (var k = 64u; k > 0u; k = k >> 1u) {
        if (li < k) { red[li] = red[li] + red[li + k]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    for (var i = li; i < hw; i = i + 128u) {
        var acc = vec2<f32>(0.0, 0.0);
        for (var j = 0u; j < len; j = j + 1u) {
            acc = acc + sc[sb + j] * unpack2x16float(vc[(j * nkv + g) * hw + i]);
        }
        out[qb + 2u * i] = acc.x * inv;
        out[qb + 2u * i + 1u] = acc.y * inv;
    }
}
"#;

/// `g = silu(g) * u` over `n` values. Args: n.
const SILU_MUL: &str = r#"
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arg(0u)) {
        let v = g[i];
        g[i] = v / (1.0 + exp(-v)) * u[i];
    }
}
"#;

/// The WGSL of kernel `name`.
fn source(name: &str) -> anyhow::Result<String> {
    let matvec = |(items, item): (&str, &str)| MATVEC.replace("ITEMS", items).replace("ITEM", item);
    let body = match name {
        "rms_norm" => RMS_NORM.to_string(),
        "mv_q4k" => matvec(Q4K_ITEM),
        "mv_q6k" => matvec(Q6K_ITEM),
        "mv_q8_0" => matvec(Q8_0_ITEM),
        "mv_f16" => matvec(F16_ITEM),
        "prep_q" => format!("{PREP_Q_DST}{}", PREP.replace("STORE", PREP_Q_STORE)),
        "prep_kv" => format!("{PREP_KV_DST}{}", PREP.replace("STORE", PREP_KV_STORE)),
        "attention" => ATTENTION.to_string(),
        "silu_mul" => SILU_MUL.to_string(),
        other => anyhow::bail!("vulkan: no kernel {other}"),
    };
    Ok(format!("{HEAD}{body}"))
}

/// The SPIR-V words of kernel `name`.
pub(super) fn spirv(name: &str) -> anyhow::Result<Vec<u32>> {
    let src = source(name)?;
    let module = naga::front::wgsl::parse_str(&src)
        .map_err(|e| anyhow::anyhow!("vulkan: {name}: {}", e.emit_to_string(&src)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::IMMEDIATES
            | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
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
