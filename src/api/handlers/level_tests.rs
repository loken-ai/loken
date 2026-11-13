//! The precision of the products on a card stays the process's after a model loads.
//!
//! A model loaded at `balanced` leaves an F32 product of another thread at the IEEE F32
//! reduction the process starts at.

use super::handover_route_tests::{models_dir, MODEL};
use crate::inference::engine::llm_engine::{Fidelity, InferenceConfig, LlmEngine};
use crate::tensor::cuda::{gemm_reduced_precision_bf16, gemm_reduced_precision_f32, GemmScope};
use crate::tensor::{Device, Tensor};

/// Rows, inner width and columns of the product.
const SHAPE: (usize, usize, usize) = (64, 1024, 64);

/// `n` values in [-0.5, 0.5) whose mantissas use every bit, from `seed`.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// The F32 product of 2 fixed matrices on card 0, read back.
fn product() -> Vec<f32> {
    let card = Device::new_cuda(0).unwrap();
    let (m, k, n) = SHAPE;
    let a = Tensor::from_vec(values(m * k, 1), (m, k), &card).unwrap();
    let b = Tensor::from_vec(values(k * n, 2), (k, n), &card).unwrap();
    a.matmul(&b)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_loaded_at_balanced_leaves_the_exact_products_exact() {
    if crate::inference::place::device_probe::card_count() == 0 {
        return;
    }
    let (root, _) = models_dir();
    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: MODEL.to_string(),
        models_dir: Some(root.clone()),
        fidelity: Fidelity::Balanced,
        ..InferenceConfig::default()
    });
    engine
        .load_model()
        .await
        .expect("the tiny checkpoint loads");
    let (levels, after_load) = std::thread::spawn(|| {
        (
            (gemm_reduced_precision_f32(), gemm_reduced_precision_bf16()),
            product(),
        )
    })
    .join()
    .unwrap();
    let exact = {
        let _exact = GemmScope::enter(false);
        product()
    };
    drop(engine);
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(
        levels,
        (false, false),
        "the load changed the process's level"
    );
    assert!(
        after_load == exact,
        "an F32 product after the load is not the IEEE F32 one"
    );
}
