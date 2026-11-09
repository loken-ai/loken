//! The SwiGLU of a routed expert, bounded at a layer's limit, computed value by value.
//!
//! Its exponential takes the steps the card kernels take, so an expert gives the same bits on
//! the host and on a card.

/// `e^x` in single precision from additions, multiplications and a division alone, each rounded
/// once in the order written: a range reduction by ln 2 and a degree-five polynomial, scaled by
/// two powers of two. The cards compute it with the same steps, so the SwiGLU of an expert is
/// the same bits wherever the expert runs. Within two units in the last place of `exp`.
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const P: [f32; 6] = [
        1.987_569_1e-4,
        1.398_199_9e-3,
        8.333_452e-3,
        4.166_579_6e-2,
        0.166_666_65,
        0.5,
    ];
    if x.is_nan() {
        return x;
    }
    if x > 88.722_84 {
        return f32::INFINITY;
    }
    if x < -103.972_08 {
        return 0.0;
    }
    let k = (x * LOG2E).round_ties_even();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut p = P[0];
    for c in &P[1..] {
        p = p * r + c;
    }
    let y = p * (r * r) + r + 1.0;
    let k = k as i32;
    let (k1, k2) = (k / 2, k - k / 2);
    let two = |e: i32| f32::from_bits(((e + 127) as u32) << 23);
    y * two(k1) * two(k2)
}

/// The SwiGLU of one gate and up value, the gate clamped from above and the up value on both
/// sides when `limit` is positive: `silu(g) * u`, its steps in the order the cards take them.
#[inline]
pub fn swiglu(g: f32, u: f32, limit: f32) -> f32 {
    let (g, u) = if limit > 0.0 {
        (g.min(limit), u.clamp(-limit, limit))
    } else {
        (g, u)
    };
    g / (1.0 + exp(-g)) * u
}
