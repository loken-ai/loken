//! The dot product shared by the formats loken reads but does not write: each block is decoded
//! to its values and multiplied against the activation's codes.

use super::*;

/// `sum(x . y)` over blocks of `QK_K` values, `x` decoded by the format's own dequantiser.
pub(super) fn dot_by_decoding<B: BlockFormat>(xs: &[B], ys: &[BlockQ8K]) -> f32 {
    let mut values = [0f32; QK_K];
    let mut total = 0f32;
    for (x, y) in xs.iter().zip(ys) {
        B::dequantize(std::slice::from_ref(x), &mut values);
        let sum: f32 = values
            .iter()
            .zip(y.qs.iter())
            .map(|(w, &q)| w * q as f32)
            .sum();
        total += sum * y.d;
    }
    total
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
}
