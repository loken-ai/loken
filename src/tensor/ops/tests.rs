//! Tests for the ops, moved with them.

use super::*;

mod softmax_masked_tests {
    use super::*;
    #[test]
    fn fused_masked_softmax_bit_exact_vs_unfused() {
        // att [B=1, H=2, Pq=5, Pk=5], causal mask [Pq,Pk] with -inf above diagonal.
        let (h, p) = (2usize, 5usize);
        let att: Vec<f32> = (0..h * p * p)
            .map(|i| (i as f32 * 0.37).sin() * 3.0)
            .collect();
        let mut mask = vec![0f32; p * p];
        for i in 0..p {
            for j in 0..p {
                if j > i {
                    mask[i * p + j] = f32::NEG_INFINITY;
                }
            }
        }
        let att_t = Tensor::from_vec(att, vec![1, h, p, p], &Device::Cpu).unwrap();
        let mask_t = Tensor::from_vec(mask, vec![p, p], &Device::Cpu).unwrap();
        let fused = softmax_last_dim_masked(&att_t, &mask_t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let unfused = softmax_last_dim(&att_t.broadcast_add(&mask_t).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(fused.len(), unfused.len());
        for (a, b) in fused.iter().zip(&unfused) {
            assert_eq!(a.to_bits(), b.to_bits(), "fused {a} != unfused {b}");
        }
    }
}
