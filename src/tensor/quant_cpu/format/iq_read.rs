//! The dot product shared by the formats loken reads but does not write: each block is decoded
//! to its values and multiplied against the activation's codes.

use super::*;

/// `sum(x . y)` over blocks of `QK_K` values, `x` decoded by the format's own dequantiser, the
/// activation's runs each under their own scale.
pub(super) fn dot_by_decoding<B: BlockFormat>(xs: &[B], ys: &[BlockQ8K32]) -> f32 {
    let mut values = [0f32; QK_K];
    let mut total = 0f32;
    for (x, y) in xs.iter().zip(ys) {
        B::dequantize(std::slice::from_ref(x), &mut values);
        for ib in 0..QK_K / 32 {
            let sum: f32 = values[32 * ib..32 * ib + 32]
                .iter()
                .zip(&y.qs[32 * ib..32 * ib + 32])
                .map(|(w, &q)| w * q as f32)
                .sum();
            total += sum * y.d[ib].to_f32();
        }
    }
    total
}

/// A format read as runs of thirty-two integer weights, each under a scale of its own, which is
/// what lets a dot stay in integers until the end of a run.
pub(super) trait Runs: BlockFormat<ActivationBlock = BlockQ8K32> {
    /// The scale of run `ib`, block scale included, as the dequantiser applies it.
    fn run_scale(&self, ib: usize) -> f32;
    /// The weights of run `ib` before the scale.
    fn run_values(&self, ib: usize) -> [i8; 32];
}

/// `sum(x . y)` over blocks of `QK_K` values for a format read in runs: each run's weights against
/// the same thirty-two codes of the activation, in integers, under the run's two scales.
pub(super) fn dot_runs<B: Runs>(xs: &[B], ys: &[BlockQ8K32]) -> f32 {
    let mut total = 0f32;
    for (x, y) in xs.iter().zip(ys) {
        for ib in 0..QK_K / 32 {
            let w = x.run_values(ib);
            let a: &[i8] = &y.qs[32 * ib..32 * ib + 32];
            let dot: i32 = w.iter().zip(a).map(|(&w, &a)| w as i32 * a as i32).sum();
            total += x.run_scale(ib) * dot as f32 * y.d[ib].to_f32();
        }
    }
    total
}

/// Sixteen i16 sums of a run of thirty-two weights against thirty-two activation codes, each of two
/// adjacent products, the weights being magnitudes (`mags`, unsigned bytes) under sign bits: byte `i`
/// of the run is negated when bit `i % 8` of byte `i / 8` of `signs` is set. The sign goes onto the
/// activation, which leaves the magnitudes unsigned for the byte multiply. A caller widens the sums
/// under the run's scale with one `madd`, so that a block stays in integers until its end.
#[cfg(target_feature = "avx2")]
#[inline]
pub(super) unsafe fn grid_run_pairs(
    mags: std::arch::x86_64::__m256i,
    signs: u32,
    acts: &[i8],
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let spread = _mm256_shuffle_epi8(
        _mm256_set1_epi32(signs as i32),
        _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        ),
    );
    let bit = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    // 0xFF where the bit is set, which with a one ORed in is -1, and +1 elsewhere.
    let negate = _mm256_or_si256(
        _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit), bit),
        _mm256_set1_epi8(1),
    );
    let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
    _mm256_maddubs_epi16(mags, _mm256_sign_epi8(a, negate))
}

/// Sum of the eight float lanes.
#[cfg(target_feature = "avx2")]
#[inline]
pub(super) unsafe fn sum_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let pair = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let quad = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
    _mm_cvtss_f32(_mm_add_ss(quad, _mm_shuffle_ps(quad, quad, 1)))
}

/// The dot of thirty-two codes of the non-linear table, packed as iq4_nl packs them (byte `j` holds
/// value `j` low and value `j + 16` high), against thirty-two activation codes.
#[inline]
pub(super) fn dot_iq4_run(packed: &[u8; 16], acts: &[i8]) -> i32 {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let table = _mm_loadu_si128(super::iq4_nl::LEVELS.as_ptr() as *const __m128i);
        let bytes = _mm_loadu_si128(packed.as_ptr() as *const __m128i);
        let mask = _mm_set1_epi8(0x0F);
        let low = _mm_shuffle_epi8(table, _mm_and_si128(bytes, mask));
        let high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), mask));
        let w = _mm256_set_m128i(high, low);
        let a = _mm256_loadu_si256(acts.as_ptr() as *const __m256i);
        // Unsigned times signed: the weights' magnitudes against the activations carrying their sign.
        let magnitude = _mm256_sign_epi8(w, w);
        let signed = _mm256_sign_epi8(a, w);
        let pairs = _mm256_maddubs_epi16(magnitude, signed);
        let sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
        let lanes: [i32; 8] = std::mem::transmute(sums);
        return lanes.iter().sum();
    }
    #[allow(unreachable_code)]
    {
        let levels = &super::iq4_nl::LEVELS;
        (0..16)
            .map(|j| {
                levels[(packed[j] & 0x0F) as usize] as i32 * acts[j] as i32
                    + levels[(packed[j] >> 4) as usize] as i32 * acts[j + 16] as i32
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    //! The formats loken only reads, against decoders written from the format definitions and
    //! blocks cut out of a file that stores each of them (`tests/vectors/ggml_low_bit_blocks.json`).

    use super::*;
    use crate::tensor::quant_cpu::to_float_bytes;

    fn vectors() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/vectors/ggml_low_bit_blocks.json");
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn every_low_bit_format_decodes_real_blocks_as_the_reference_does() {
        let v = vectors();
        for (name, dtype) in [
            ("IQ4_NL", GgmlDType::Iq4Nl),
            ("IQ4_XS", GgmlDType::Iq4Xs),
            ("IQ3_XXS", GgmlDType::Iq3Xxs),
            ("IQ3_S", GgmlDType::Iq3S),
            ("IQ2_S", GgmlDType::Iq2S),
            ("Q2_0", GgmlDType::Q2_0),
        ] {
            let e = &v[name];
            let bytes = hex(e["bytes"].as_str().unwrap());
            let want: Vec<f64> = e["decoded"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let mut got = vec![0f32; want.len()];
            to_float_bytes(dtype, &bytes, &mut got).unwrap();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (*g as f64 - w).abs() <= 1e-6 * w.abs().max(1e-3),
                    "{name} value {i}: {g} against {w}"
                );
            }
        }
    }

    #[test]
    fn a_low_bit_dot_equals_the_dot_of_the_decoded_values() {
        let v = vectors();
        for (name, dtype) in [
            ("IQ4_XS", GgmlDType::Iq4Xs),
            ("IQ3_XXS", GgmlDType::Iq3Xxs),
            ("IQ3_S", GgmlDType::Iq3S),
            ("IQ2_S", GgmlDType::Iq2S),
            ("Q2_0", GgmlDType::Q2_0),
        ] {
            let e = &v[name];
            let bytes = hex(e["bytes"].as_str().unwrap());
            let decoded: Vec<f32> = e["decoded"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let act: Vec<f32> = (0..decoded.len())
                .map(|i| ((i as f32) * 0.37).sin() * 2.0)
                .collect();
            let want: f32 = decoded.iter().zip(&act).map(|(a, b)| a * b).sum();
            // The activation is rounded to the format's codes: a few permille of the sum's scale.
            let mut dst = [0f32];
            crate::tensor::quant_cpu::matmul_bytes(
                dtype,
                (1, act.len(), 1),
                &act,
                &bytes,
                &mut dst,
            )
            .unwrap();
            let got = dst[0];
            let scale: f32 = decoded.iter().zip(&act).map(|(a, b)| (a * b).abs()).sum();
            assert!(
                (got - want).abs() < 0.01 * scale,
                "{name}: {got} against {want}"
            );
        }
    }

    /// The card's codebooks, in `cuda/moe/iq_tables.cuh`, are the host's: every entry, read back
    /// from the header the kernels include.
    #[test]
    fn the_card_codebooks_are_the_hosts() {
        use super::super::iq2_xxs::KSIGNS_IQ2XS;
        use super::super::iq4_nl::LEVELS;
        use super::super::iq_grids::{IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID};
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/moe/iq_tables.cuh");
        let text = std::fs::read_to_string(path).unwrap();
        let table = |name: &str| -> Vec<i64> {
            let start = text
                .find(&format!(" {name}["))
                .unwrap_or_else(|| panic!("{name} is not in the header"));
            let body = &text[text[start..].find('{').unwrap() + start + 1..];
            let body = &body[..body.find('}').unwrap()];
            body.split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(|t| {
                    let t = t.trim_end_matches(['u', 'l']);
                    match t.strip_prefix("0x") {
                        Some(h) => u64::from_str_radix(h, 16).unwrap() as i64,
                        None => t.parse().unwrap(),
                    }
                })
                .collect()
        };
        let same = |name: &str, host: Vec<i64>| assert_eq!(table(name), host, "{name}");
        same("IQ2S_GRID", IQ2S_GRID.iter().map(|&v| v as i64).collect());
        same(
            "IQ3XXS_GRID",
            IQ3XXS_GRID.iter().map(|&v| v as i64).collect(),
        );
        same("IQ3S_GRID", IQ3S_GRID.iter().map(|&v| v as i64).collect());
        same(
            "KSIGNS_IQ2XS",
            KSIGNS_IQ2XS.iter().map(|&v| v as i64).collect(),
        );
        same("KVALUES_IQ4NL", LEVELS.iter().map(|&v| v as i64).collect());
    }

    /// What each read-only format's dot costs on one thread, in bytes of weight per second. Not a gate.
    #[test]
    #[ignore = "a measurement: cargo test --release iq_dot_throughput -- --ignored --nocapture"]
    fn iq_dot_throughput() {
        fn run<T: BlockFormat<ActivationBlock = BlockQ8K32>>(name: &str) {
            let blocks = (64 << 20) / std::mem::size_of::<T>();
            let mut state = 0x9E37_79B9u64;
            let mut bytes = vec![0u8; blocks * std::mem::size_of::<T>()];
            for b in bytes.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (state >> 33) as u8;
            }
            for chunk in bytes.chunks_exact_mut(std::mem::size_of::<T>()) {
                chunk[..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            }
            let weights: &[T] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, blocks) };
            let acts = vec![BlockQ8K32::zeros(); blocks];
            let mut sink = 0f32;
            let start = std::time::Instant::now();
            for _ in 0..3 {
                sink += T::dot(std::hint::black_box(weights), std::hint::black_box(&acts));
            }
            let secs = start.elapsed().as_secs_f64() / 3.0;
            eprintln!(
                "{name:8} {:6.2} GB/s of weights, {:5.1} ns a block ({sink})",
                bytes.len() as f64 / secs / 1e9,
                secs * 1e9 / blocks as f64
            );
        }
        run::<super::iq3_xxs::BlockIq3Xxs>("iq3_xxs");
        run::<super::iq3_s::BlockIq3S>("iq3_s");
        run::<super::iq2_s::BlockIq2S>("iq2_s");
        run::<super::iq4_xs::BlockIq4Xs>("iq4_xs");
    }
}
