//! A draft verify on the host against the decode steps it stands for: the same tokens from
//! the same cache, once as one multi-row `forward_all`, once one `forward` per token. Greedy
//! output may not depend on how many tokens were drafted, so each row's logits must equal the
//! step's bit for bit, whatever the row count.

use super::tiny_model;
use super::*;
use crate::inference::engine::decode_step::VERIFY_ROWS;

const LAYERS: usize = 2;
const PROMPT: [u32; 5] = [1, 4, 2, 9, 3];
const NEXT: [u32; VERIFY_ROWS] = [7, 5, 8, 6, 2, 11, 30, 41, 17];

fn ids(toks: &[u32]) -> Tensor {
    Tensor::new(toks, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
}

fn rows(t: &Tensor) -> Vec<Vec<f32>> {
    let t = t.to_dtype(DType::F32).unwrap();
    let vocab = t.dims().last().copied().unwrap();
    let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    flat.chunks(vocab).map(<[f32]>::to_vec).collect()
}

/// The logits of each token of `NEXT` decoded alone after the prompt.
fn decode_steps(m: &mut GenericHeteroTransformer) -> Vec<Vec<f32>> {
    m.forward(&ids(&PROMPT), 0).unwrap();
    NEXT.iter()
        .enumerate()
        .map(|(i, &t)| rows(&m.forward(&ids(&[t]), PROMPT.len() + i).unwrap()).remove(0))
        .collect()
}

/// Every verify width from two rows to the widest reproduces the decode steps exactly, and
/// a cache trimmed after a verify decodes on as if the rows had been decoded one by one.
#[test]
fn verify_rows_match_decode_steps_on_the_host() {
    let path = tiny_model::write_quant("verify-rows", LAYERS);
    let mut stepped = tiny_model::load(&path, LAYERS);
    let want = decode_steps(&mut stepped);
    let mut verified = tiny_model::load(&path, LAYERS);
    verified.forward(&ids(&PROMPT), 0).unwrap();
    for width in 2..=VERIFY_ROWS {
        let got = rows(
            &verified
                .forward_all(&ids(&NEXT[..width]), PROMPT.len())
                .unwrap(),
        );
        for (r, row) in got.iter().enumerate() {
            assert!(
                row == &want[r],
                "row {r} of a {width}-row verify differs from its decode step"
            );
        }
        verified.trim_kv(PROMPT.len());
    }
    // Two rows kept from a wider verify, then plain steps from there.
    verified
        .forward_all(&ids(&NEXT[..5]), PROMPT.len())
        .unwrap();
    verified.trim_kv(PROMPT.len() + 2);
    for (i, &t) in NEXT.iter().enumerate().skip(2) {
        let got = rows(&verified.forward(&ids(&[t]), PROMPT.len() + i).unwrap()).remove(0);
        assert!(got == want[i], "step {i} after a trimmed verify differs");
    }
    tiny_model::remove(&path);
}

/// The same requests on a fresh model give the same logits: nothing carried between loads or
/// left to thread timing reaches the arithmetic.
#[test]
fn host_decode_is_reproducible() {
    let path = tiny_model::write_quant("reproducible", LAYERS);
    let first = decode_steps(&mut tiny_model::load(&path, LAYERS));
    let second = decode_steps(&mut tiny_model::load(&path, LAYERS));
    assert!(first == second);
    tiny_model::remove(&path);
}

/// Times the row-exact host kernels against the tiled ones at a dense 8B layer's widths.
#[test]
#[ignore = "timing, run by hand"]
fn bench_rows_kernels() {
    use crate::tensor::quant_cpu as qc;
    use crate::tensor::quantized::GgmlDType;
    let (h, f) = (4096usize, 12288usize);
    let mut seed = 5u32;
    let mut vals = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f32 / 65536.0 - 0.5) * 0.4
            })
            .collect()
    };
    let wq4 = qc::from_float_bytes(GgmlDType::Q4K, &vals(f * h)).unwrap();
    let wq6 = qc::from_float_bytes(GgmlDType::Q6K, &vals(h * f)).unwrap();
    let blocks = qc::cast_blocks::<qc::BlockQ4K>(&wq4).unwrap();
    let x8 = qc::repack_q4k::repack(blocks, f, h / 256);
    let b6 = qc::cast_blocks::<qc::BlockQ6K>(&wq6).unwrap();
    let x6 = qc::repack_q6k::repack(b6, h, f / 256);
    let time = |name: &str, run: &mut dyn FnMut()| {
        run();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            run();
        }
        eprintln!("{name}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    };
    let gu = qc::from_float_bytes(GgmlDType::Q4K, &vals(2 * f * h)).unwrap();
    let norm = |x: &[f32], o: &mut [f32]| o.copy_from_slice(x);
    let act = |g: &[f32], u: &[f32], o: &mut [f32]| {
        crate::inference::kernel::cpu_decode_exec::silu_mul_slice(g, u, o)
    };
    for m in [1usize, 2, 3, 5, 9] {
        let mut xr = vals(m * h);
        time(&format!("ffn rows   m={m}"), &mut || {
            qc::ffn_swiglu_coop_rows(
                &mut xr,
                h,
                f,
                None,
                (GgmlDType::Q4K, h, 2 * f, &gu),
                (GgmlDType::Q6K, f, h, &wq6),
                None,
                &norm,
                &act,
            )
            .unwrap();
        });
        let xh = vals(m * h);
        let xf = vals(m * f);
        let mut o = vec![0f32; m * f];
        let mut od = vec![0f32; m * h];
        time(&format!("q4k rows   m={m}"), &mut || {
            qc::matmul_q4k_plain_rows((m, h, f), &xh, &x8, &mut o).unwrap()
        });
        time(&format!("q4k tiled  m={m}"), &mut || {
            qc::matmul_q4k_plain((m, h, f), &xh, &x8, &mut o).unwrap()
        });
        time(&format!("q4k bytes  m={m}"), &mut || {
            qc::matmul_bytes(GgmlDType::Q4K, (m, h, f), &xh, &wq4, &mut o).unwrap()
        });
        time(&format!("q6k bytes  m={m}"), &mut || {
            qc::matmul_bytes(GgmlDType::Q6K, (m, f, h), &xf, &wq6, &mut od).unwrap()
        });
        if m >= 2 {
            time(&format!("q6k tiled  m={m}"), &mut || {
                qc::matmul_q6k_repacked_tiled((m, f, h), &xf, &x6, &mut od).unwrap()
            });
        }
    }
}

/// The row-exact Q4_K kernel gives each row what the one-row decode GEMV gives it.
#[cfg(target_arch = "x86_64")]
#[test]
fn q4k_rows_kernel_matches_the_one_row_kernel() {
    use crate::tensor::quant_cpu as qc;
    use crate::tensor::quantized::GgmlDType;
    if !crate::cpu::dispatch_avx2() {
        return;
    }
    let (k, n) = (512usize, 64usize);
    let mut seed = 9u32;
    let mut vals = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f32 / 65536.0 - 0.5) * 2.0
            })
            .collect()
    };
    let w = qc::from_float_bytes(GgmlDType::Q4K, &vals(n * k)).unwrap();
    let x8 = qc::repack_q4k::repack(qc::cast_blocks::<qc::BlockQ4K>(&w).unwrap(), n, k / 256);
    for m in 1..=qc::repack_q4k::PLAIN_ROWS_MAX + 2 {
        let x = vals(m * k);
        let mut got = vec![0f32; m * n];
        qc::matmul_q4k_plain_rows((m, k, n), &x, &x8, &mut got).unwrap();
        for r in 0..m {
            let mut want = vec![0f32; n];
            qc::matmul_q4k_plain((1, k, n), &x[r * k..(r + 1) * k], &x8, &mut want).unwrap();
            assert!(got[r * n..(r + 1) * n] == want[..], "row {r} of {m}");
        }
    }
}
