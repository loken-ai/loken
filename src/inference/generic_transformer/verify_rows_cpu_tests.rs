//! A draft verify on the host against the decode steps it stands for: the same tokens from
//! the same cache, once as one multi-row `forward_all`, once one `forward` per token. Greedy
//! output may not depend on how many tokens were drafted, so each row's logits must equal the
//! step's bit for bit, whatever the row count.

use super::tiny_model;
use super::*;
use crate::inference::engine::decode_step::VERIFY_ROWS;
use crate::inference::engine::llm_engine::KvQuant;

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
    let mut stepped = tiny_model::load_q8_kv(&path, LAYERS);
    let want = decode_steps(&mut stepped);
    let mut verified = tiny_model::load_q8_kv(&path, LAYERS);
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
    let first = decode_steps(&mut tiny_model::load_q8_kv(&path, LAYERS));
    let second = decode_steps(&mut tiny_model::load_q8_kv(&path, LAYERS));
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

/// Each host product of a verify gives every row what the one-row decode step gives it, for
/// every stored format and every verify width, at the widths of a dense 1B layer with grouped
/// keys and a vocabulary of 131072: the query, key, feed-forward, down and head projections.
#[test]
fn host_products_give_each_verify_row_its_decode_step() {
    use crate::tensor::quantized::{DecodeOrder, GgmlDType, QMatMul, QTensor};
    let shapes = [
        (2048usize, 2048usize),
        (2048, 1024),
        (2048, 8192),
        (8192, 2048),
        (256, 131072),
    ];
    let formats = [
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q5_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ];
    let mut seed = 31u32;
    let mut vals = |len: usize, scale: f32| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f32 / 65536.0 - 0.5) * scale
            })
            .collect()
    };
    let mut broken = Vec::new();
    for &(k, n) in &shapes {
        let w = Tensor::from_vec(vals(n * k, 0.1), (n, k), &Device::Cpu).unwrap();
        let x = vals(VERIFY_ROWS * k, 2.0);
        for dtype in formats {
            let m = QMatMul::from_qtensor(QTensor::quantize(&w, dtype).unwrap()).unwrap();
            let product = |rows: &[f32], r: usize| -> Vec<f32> {
                let t = Tensor::from_vec(rows.to_vec(), (1, r, k), &Device::Cpu).unwrap();
                m.forward(&t)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let steps: Vec<Vec<f32>> = (0..VERIFY_ROWS)
                .map(|r| product(&x[r * k..(r + 1) * k], 1))
                .collect();
            let _order = DecodeOrder::enter();
            for width in 2..=VERIFY_ROWS {
                let got = product(&x[..width * k], width);
                if let Some(r) = (0..width).find(|&r| got[r * n..(r + 1) * n] != steps[r][..]) {
                    broken.push(format!("{dtype:?} k={k} n={n}: row {r} of {width}"));
                    break;
                }
            }
        }
    }
    assert!(
        broken.is_empty(),
        "rows that differ from their step: {broken:#?}"
    );
}

/// A llama model of `layers` blocks shaped as a dense 1B one with grouped keys: heads 256
/// wide, 4 query heads over 2 key heads, adjacent rotation pairs, a vocabulary of 131072, every
/// matrix stored as Q8_0.
fn write_q8_0_grouped(tag: &str, layers: usize) -> std::path::PathBuf {
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Value;
    use crate::tensor::quantized::GgmlDType;
    let (hidden, heads, kv_heads, head_dim, ffn, vocab) = (1024usize, 4, 2, 256, 2048, 131072);
    let mut seed = 41u32;
    let mut entry = |name: String, dims: Vec<usize>, dtype: GgmlDType| {
        let xs: Vec<f32> = (0..dims.iter().product::<usize>())
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f32 / 65536.0 - 0.5) * 0.4
            })
            .collect();
        let data = match dtype {
            GgmlDType::F32 => xs.iter().flat_map(|v| v.to_le_bytes()).collect(),
            _ => crate::tensor::quant_cpu::from_float_bytes(dtype, &xs).unwrap(),
        };
        GgufEntry {
            name,
            dims,
            dtype,
            data,
        }
    };
    let ones = |name: String, n: usize| GgufEntry {
        name,
        dims: vec![n],
        dtype: GgmlDType::F32,
        data: [1.0f32.to_le_bytes(); 1].repeat(n).concat(),
    };
    let q8 = GgmlDType::Q8_0;
    let mut entries = vec![
        entry("token_embd.weight".into(), vec![vocab, hidden], q8),
        entry("output.weight".into(), vec![vocab, hidden], q8),
        ones("output_norm.weight".into(), hidden),
    ];
    for il in 0..layers {
        entries.push(ones(format!("blk.{il}.attn_norm.weight"), hidden));
        entries.push(ones(format!("blk.{il}.ffn_norm.weight"), hidden));
        for (n, d) in [
            ("attn_q", [heads * head_dim, hidden]),
            ("attn_k", [kv_heads * head_dim, hidden]),
            ("attn_v", [kv_heads * head_dim, hidden]),
            ("attn_output", [hidden, heads * head_dim]),
            ("ffn_gate", [ffn, hidden]),
            ("ffn_up", [ffn, hidden]),
            ("ffn_down", [hidden, ffn]),
        ] {
            entries.push(entry(format!("blk.{il}.{n}.weight"), d.to_vec(), q8));
        }
    }
    let u = |k: &str, v: u32| (format!("llama.{k}"), Value::U32(v));
    let metadata = vec![
        ("general.architecture".into(), Value::String("llama".into())),
        u("block_count", layers as u32),
        u("embedding_length", hidden as u32),
        u("feed_forward_length", ffn as u32),
        u("attention.head_count", heads as u32),
        u("attention.head_count_kv", kv_heads as u32),
        u("attention.key_length", head_dim as u32),
        u("attention.value_length", head_dim as u32),
        u("context_length", 4096),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-6),
        ),
        ("llama.rope.freq_base".into(), Value::F32(1.0e6)),
    ];
    let dir = std::env::temp_dir().join(format!("loken-verify-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.gguf");
    write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
    path
}

/// Every verify width over the Q8_0 grouped model reproduces its decode steps bit for bit, with
/// each cache format, the model loaded by `load`.
fn q8_0_grouped_rows_match(
    tag: &str,
    load: &dyn Fn(&std::path::Path, KvQuant) -> GenericHeteroTransformer,
) {
    let path = write_q8_0_grouped(tag, LAYERS);
    // A prompt as long as a served one, so that the attention sums over many positions.
    let prompt: Vec<u32> = (0..108u32).map(|i| 1000 + (i * 37) % 20000).collect();
    for kv in [KvQuant::Off, KvQuant::Q8] {
        let mut stepped = load(&path, kv);
        stepped.forward(&ids(&prompt), 0).unwrap();
        let want: Vec<Vec<f32>> = NEXT
            .iter()
            .enumerate()
            .map(|(i, &t)| rows(&stepped.forward(&ids(&[t]), prompt.len() + i).unwrap()).remove(0))
            .collect();
        let mut verified = load(&path, kv);
        verified.forward(&ids(&prompt), 0).unwrap();
        for width in 2..=VERIFY_ROWS {
            let got = rows(
                &verified
                    .forward_all(&ids(&NEXT[..width]), prompt.len())
                    .unwrap(),
            );
            for (r, row) in got.iter().enumerate() {
                assert!(
                    row == &want[r],
                    "{tag} {kv:?}: row {r} of a {width}-row verify differs from its decode step"
                );
            }
            verified.trim_kv(prompt.len());
        }
    }
    tiny_model::remove(&path);
}

/// The model at `path` on the devices `cards`, placed by `plan`, with the cache `kv` sized for
/// 4096 positions as a served model's is.
fn load_placed(
    path: &std::path::Path,
    kv: KvQuant,
    cards: &std::collections::HashMap<usize, Device>,
    plan: &crate::inference::place::layer_executor::HeteroPlan,
) -> GenericHeteroTransformer {
    let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
    let bytes = g.mmap.clone();
    GenericHeteroTransformer::from_gguf_with_kv_quant(
        g.content,
        &bytes[..],
        cards,
        plan,
        kv,
        Some(4096),
    )
    .unwrap()
}

/// The Q8_0 grouped model on the host.
#[test]
fn q8_0_grouped_verify_rows_match_decode_steps_on_the_host() {
    let plan = crate::inference::place::layer_executor::HeteroPlan::calculate_with_kv(
        LAYERS,
        1 << 20,
        &[],
        &[],
        1.0,
        0,
    );
    let host = std::collections::HashMap::new();
    q8_0_grouped_rows_match("host", &|p, kv| load_placed(p, kv, &host, &plan));
}

/// The Q8_0 grouped model on card 0.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs an idle CUDA card"]
fn q8_0_grouped_verify_rows_match_decode_steps_on_a_card() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let cards: std::collections::HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    let plan = crate::inference::place::layer_executor::HeteroPlan::forced_gpu(LAYERS, LAYERS, 0);
    q8_0_grouped_rows_match("card", &|p, kv| load_placed(p, kv, &cards, &plan));
}

/// The model the store holds under `tag`, on the host with the cache `kv`.
fn stored_on_host(tag: &str, kv: KvQuant) -> GenericHeteroTransformer {
    stored_on(tag, kv, &std::collections::HashMap::new())
}

/// The model the store holds under `tag` with the cache `kv`, every layer on card 0 of `cards`,
/// on the host when `cards` is empty.
fn stored_on(
    tag: &str,
    kv: KvQuant,
    cards: &std::collections::HashMap<usize, Device>,
) -> GenericHeteroTransformer {
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let (name, version) = tag.split_once(':').unwrap_or((tag, "latest"));
    let m = crate::inference::load::ollama_manager::OllamaManager::new(store.clone());
    let manifest = m.read_manifest_json(name, version).unwrap();
    let digest = manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["mediaType"] == "application/vnd.ollama.image.model")
        .and_then(|l| l["digest"].as_str())
        .unwrap()
        .replace(':', "-");
    let path = store.join("blobs").join(digest);
    let layers = {
        let g = crate::tensor::quantized::gguf_file::open_mapped(&path).unwrap();
        let arch = g.content.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .clone();
        GenericTransformerConfig::from_gguf(&g.content, &arch)
            .unwrap()
            .n_layers
    };
    let plan = match cards.is_empty() {
        true => crate::inference::place::layer_executor::HeteroPlan::calculate_with_kv(
            layers,
            1 << 20,
            &[],
            &[],
            1.0,
            0,
        ),
        false => crate::inference::place::layer_executor::HeteroPlan::forced_gpu(layers, layers, 0),
    };
    load_placed(&path, kv, cards, &plan)
}

/// The stored falcon3 file on the host, its cache unquantised: every verify width reproduces
/// the decode steps bit for bit. Its layers pair adjacent lanes, which the host's fused decode
/// does not take, so its verify runs through the attention of the general forward.
#[test]
#[ignore = "needs falcon3:latest in the store"]
fn a_stored_q8_0_model_verifies_its_decode_steps_on_the_host() {
    let prompt: Vec<u32> = (0..108u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let next: Vec<u32> = (0..VERIFY_ROWS as u32).map(|i| 3000 + i * 11).collect();
    let mut stepped = stored_on_host("falcon3:latest", KvQuant::Off);
    stepped.forward(&ids(&prompt), 0).unwrap();
    let want: Vec<Vec<f32>> = next
        .iter()
        .enumerate()
        .map(|(i, &t)| rows(&stepped.forward(&ids(&[t]), prompt.len() + i).unwrap()).remove(0))
        .collect();
    let mut verified = stored_on_host("falcon3:latest", KvQuant::Off);
    verified.forward(&ids(&prompt), 0).unwrap();
    for width in 2..=VERIFY_ROWS {
        let got = rows(
            &verified
                .forward_all(&ids(&next[..width]), prompt.len())
                .unwrap(),
        );
        for (r, row) in got.iter().enumerate() {
            assert!(
                row == &want[r],
                "row {r} of a {width}-row verify differs from its decode step"
            );
        }
        verified.trim_kv(prompt.len());
    }
}

/// The time of a verify on the host against the plain steps it replaces, on the model the
/// store holds under `falcon3:latest`, its cache unquantised: the median of 9 runs per
/// width.
#[test]
#[ignore = "timing, needs a model in the store"]
fn host_verify_cost_against_decode_steps() {
    let tag = "falcon3:latest";
    let mut m = stored_on_host(tag, KvQuant::Off);
    let prompt: Vec<u32> = (0..108u32).map(|i| 1000 + (i * 37) % 20000).collect();
    m.forward(&ids(&prompt), 0).unwrap();
    let base = prompt.len();
    let toks: Vec<u32> = (0..VERIFY_ROWS as u32).map(|i| 3000 + i * 11).collect();
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut step = Vec::new();
    for _ in 0..9 {
        m.trim_kv(base);
        let s = std::time::Instant::now();
        m.forward(&ids(&toks[..1]), base).unwrap();
        step.push(s.elapsed().as_secs_f64());
    }
    let step = median(step);
    println!("{tag}: plain step {:.2} ms", step * 1e3);
    for rows in [2usize, 4, 8] {
        let mut verify = Vec::new();
        for _ in 0..9 {
            m.trim_kv(base);
            let s = std::time::Instant::now();
            m.forward_all(&ids(&toks[..rows]), base).unwrap();
            verify.push(s.elapsed().as_secs_f64());
        }
        let v = median(verify);
        println!(
            "{tag}: verify of {rows} rows {:.2} ms, {:.2} plain steps",
            v * 1e3,
            v / step
        );
    }
}

/// The stored gemma4 file on the host, a prompt longer than its attention window prefilled in
/// chunks as the server does: every row of every verify width is its decode step bit for bit,
/// and the steps after a cut verify are the steps of a run that never verified.
#[test]
#[ignore = "needs gemma4:latest in the store"]
fn a_stored_gemma4_verifies_its_decode_steps_on_the_host() {
    stored_gemma4_verifies_its_decode_steps(&|| stored_on_host("gemma4:latest", KvQuant::Off));
}

/// [`a_stored_gemma4_verifies_its_decode_steps_on_the_host`] with every layer on card 0.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs gemma4:latest in the store and an idle CUDA card"]
fn a_stored_gemma4_verifies_its_decode_steps_on_a_card() {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let cards: std::collections::HashMap<usize, Device> = [(0usize, card)].into_iter().collect();
    stored_gemma4_verifies_its_decode_steps(&|| stored_on("gemma4:latest", KvQuant::Off, &cards));
}

/// Every row of every verify width of the model `load` builds against its decode step, after a
/// prompt longer than the attention window, and the steps after a cut verify.
fn stored_gemma4_verifies_its_decode_steps(load: &dyn Fn() -> GenericHeteroTransformer) {
    use crate::inference::engine::model_backend::{GenericBackend, ModelBackend};
    let prompt: Vec<u32> = (0..1000u32).map(|i| 1000 + (i * 37) % 20000).collect();
    let next: Vec<u32> = (0..VERIFY_ROWS as u32 + 1).map(|i| 3000 + i * 11).collect();
    let p = prompt.len();
    let mut stepped = GenericBackend(load());
    stepped.forward_prefill_chunked(&ids(&prompt), 0).unwrap();
    let want: Vec<Vec<f32>> = next
        .iter()
        .enumerate()
        .map(|(i, &t)| rows(&stepped.forward(&ids(&[t]), p + i).unwrap()).remove(0))
        .collect();
    drop(stepped);
    let mut m = GenericBackend(load());
    m.forward_prefill_chunked(&ids(&prompt), 0).unwrap();
    m.forward(&ids(&next[..1]), p).unwrap();
    for width in 2..=VERIFY_ROWS {
        let got = rows(&m.forward_all(&ids(&next[1..1 + width]), p + 1).unwrap());
        for (r, row) in got.iter().enumerate() {
            assert!(
                row == &want[1 + r],
                "row {r} of a {width}-row verify differs from its decode step"
            );
        }
        m.trim_kv(p + 1);
    }
    m.forward_all(&ids(&next[1..5]), p + 1).unwrap();
    m.trim_kv(p + 2);
    for (i, &t) in next.iter().enumerate().skip(2) {
        let got = rows(&m.forward(&ids(&[t]), p + i).unwrap()).remove(0);
        assert!(got == want[i], "step {i} after a cut verify differs");
    }
}
