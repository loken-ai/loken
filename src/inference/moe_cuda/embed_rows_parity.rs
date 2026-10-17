//! The card decoder of q8_0 embedding rows against the host dequantisation of the same table.

use super::embed_rows_gguf;
use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{DType, Device, Tensor};

/// Rows gathered in an order with repeats come out as the host dequantisation of those rows,
/// and a format without a decoder is refused.
#[test]
fn q8_0_rows_match_the_host_dequantisation() {
    let Ok(dev) = Device::new_cuda(0) else { return };
    let (vocab, dim) = (40usize, 96usize);
    let vals: Vec<f32> = (0..vocab * dim)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
        .collect();
    let rows = Tensor::from_vec(vals, (vocab, dim), &Device::Cpu).unwrap();
    let host = QTensor::quantize_onto(&rows, GgmlDType::Q8_0, &Device::Cpu).unwrap();
    let card = QTensor::quantize_onto(&rows, GgmlDType::Q8_0, &dev).unwrap();
    let want = host
        .dequantize(&Device::Cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    let picked: Vec<u32> = vec![3, 0, 39, 17, 3, 22, 8];
    let ids = Tensor::from_vec(picked.clone(), picked.len(), &Device::Cpu)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    let out = Tensor::zeros_on((1, picked.len(), dim), DType::F32, &dev).unwrap();
    embed_rows_gguf(&card, &ids, &out).unwrap();
    let got = out
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (r, id) in picked.iter().enumerate() {
        for c in 0..dim {
            let (g, w) = (got[r * dim + c], want[*id as usize][c]);
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                "row {r} col {c}: {g} vs {w}"
            );
        }
    }

    let other = QTensor::quantize_onto(&rows, GgmlDType::Q4K, &dev).unwrap();
    assert!(embed_rows_gguf(&other, &ids, &out).is_err());
}
