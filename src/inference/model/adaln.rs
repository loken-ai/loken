//! Adaptive layer-norm modulation, as the DiTs apply their conditioning.

use crate::tensor::{Result, Tensor};

/// `x * (1 + scale) + shift`, with `shift` and `scale` broadcast over the rows of `x`.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
        .broadcast_add(shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulation_scales_around_one_and_shifts_every_row() {
        let dev = crate::tensor::Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).unwrap();
        let shift = Tensor::from_vec(vec![0.5f32, -1.0], (1, 2), &dev).unwrap();
        let scale = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let y = modulate(&x, &shift, &scale)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(y, vec![2.5, 1.0, 6.5, 3.0]);
    }
}
