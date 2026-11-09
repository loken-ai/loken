//! Routed experts held as projections read in place: block-scaled fp8 or fp4 with their scales,
//! bf16, or any other storage a [`Projection`] holds.
//!
//! An expert is `down(swiglu(gate . x, up . x))`.
//! Its SwiGLU is `bounded_swiglu::swiglu`, bounded at the layer's limit when it has one.
//! The experts run on the host's cores.

use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::ops::bounded_swiglu::swiglu;
use crate::tensor::{DType, Device, Result, Tensor};

/// One routed expert: gate and up `[inter, dim]`, down `[dim, inter]`.
#[derive(Clone)]
pub struct BlockScaledExpert {
    pub gate: Projection,
    pub up: Projection,
    pub down: Projection,
}

impl BlockScaledExpert {
    /// Asks for the 3 projections' mapped bytes to be read in, without waiting.
    pub fn will_need(&self) {
        self.gate.will_need();
        self.up.will_need();
        self.down.will_need();
    }

    /// The expert over the rows `x` `[n, dim]` F32 on the host, `[n, dim]`.
    ///
    /// Each projection runs once over all `n` rows.
    /// The SwiGLU runs value by value between the up and the down projections.
    pub fn forward(&self, x: &Tensor, limit: f32) -> Result<Tensor> {
        let gate = self.gate.apply_here(x)?.to_vec2::<f32>()?;
        let up = self.up.apply_here(x)?.to_vec2::<f32>()?;
        let (n, inter) = (gate.len(), gate.first().map_or(0, Vec::len));
        let mut h = vec![0f32; n * inter];
        for t in 0..n {
            for k in 0..inter {
                h[t * inter + k] = swiglu(gate[t][k], up[t][k], limit);
            }
        }
        let h = Tensor::from_vec(h, (n, inter), &Device::Cpu)?;
        self.down.apply_here(&h)
    }
}

/// The routed experts of a layer, numbered as the router numbers them.
pub struct BlockScaledExperts {
    pub experts: Vec<BlockScaledExpert>,
    /// The bound of the SwiGLU, `0` for none.
    pub swiglu_limit: f32,
}

impl BlockScaledExperts {
    /// Bytes of the experts as stored.
    pub fn bytes(&self) -> usize {
        self.experts
            .iter()
            .map(|e| e.gate.bytes() + e.up.bytes() + e.down.bytes())
            .sum()
    }

    /// The weighted sum of the routed experts of every row, `[rows, dim]` F32 on the host.
    ///
    /// `xs` is `[rows, dim]`, `ids` and `weights` are `[rows, k]` row-major.
    /// Each expert runs once over the rows routed to it, in row order, the experts by id.
    /// A row's pairs are summed in the router's order, from `-0`.
    pub fn forward(&self, xs: &Tensor, ids: &[u32], weights: &[f32], k: usize) -> Result<Tensor> {
        let host = Device::Cpu;
        let xs = xs.to_device(&host)?.to_dtype(DType::F32)?.contiguous()?;
        let (rows, dim) = xs.dims2()?;
        if ids.len() != rows * k || weights.len() != rows * k {
            crate::tensor::bail!(
                "block-scaled experts: {} ids and {} weights for {rows} rows of {k}",
                ids.len(),
                weights.len()
            );
        }
        let mut pairs_of: Vec<Vec<usize>> = vec![Vec::new(); self.experts.len()];
        for (p, &e) in ids.iter().enumerate() {
            match pairs_of.get_mut(e as usize) {
                Some(pairs) => pairs.push(p),
                None => crate::tensor::bail!(
                    "block-scaled experts: expert {e} of {}",
                    self.experts.len()
                ),
            }
        }
        let mut pair_rows = vec![0f32; rows * k * dim];
        for (e, pairs) in pairs_of.iter().enumerate() {
            if pairs.is_empty() {
                continue;
            }
            let picked: Vec<u32> = pairs.iter().map(|&p| (p / k) as u32).collect();
            let picked = Tensor::from_vec(picked, pairs.len(), &host)?;
            let x_e = xs.index_select(&picked, 0)?.contiguous()?;
            let y: Vec<f32> = self.experts[e]
                .forward(&x_e, self.swiglu_limit)?
                .flatten_all()?
                .to_vec1()?;
            for (j, &p) in pairs.iter().enumerate() {
                pair_rows[p * dim..(p + 1) * dim].copy_from_slice(&y[j * dim..(j + 1) * dim]);
            }
        }
        let mut out = vec![-0.0f32; rows * dim];
        for r in 0..rows {
            for j in 0..k {
                let p = r * k + j;
                let w = weights[p];
                for (o, v) in out[r * dim..(r + 1) * dim]
                    .iter_mut()
                    .zip(&pair_rows[p * dim..(p + 1) * dim])
                {
                    *o += w * v;
                }
            }
        }
        Tensor::from_vec(out, (rows, dim), &host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::offload::experts::Expert;
    use crate::tensor::blockscaled::fp4::Fp4Weight;
    use crate::tensor::blockscaled::fp8::Fp8Weight;
    use crate::tensor::mapped::MappedBytes;
    use std::sync::Arc;

    const DIM: usize = 128;
    const INTER: usize = 64;
    const EXPERTS: usize = 4;
    const ROWS: usize = 5;
    const K: usize = 2;

    fn mapped(bytes: &[u8]) -> MappedBytes {
        let mut m = memmap2::MmapMut::map_anon(bytes.len().max(1)).unwrap();
        m[..bytes.len()].copy_from_slice(bytes);
        MappedBytes::new(Arc::new(m.make_read_only().unwrap()), 0, bytes.len()).unwrap()
    }

    /// An fp4 weight `[out, inp]` drawn from `seed`, a scale per block of 32 inputs.
    fn fp4(out: usize, inp: usize, seed: usize) -> Projection {
        let block = 32;
        let nibbles: Vec<u8> = (0..out * inp / 2)
            .map(|i| ((i * 37 + 11 + seed * 101) % 251) as u8)
            .collect();
        let scales: Vec<u8> = (0..out * inp / block)
            .map(|i| (120 + ((i + seed) % 5)) as u8)
            .collect();
        Projection::Fp4(Arc::new(
            Fp4Weight::new(mapped(&nibbles), mapped(&scales), out, inp).unwrap(),
        ))
    }

    /// An fp8 weight `[out, inp]` drawn from `seed`, a scale per tile of 32 by 32.
    fn fp8(out: usize, inp: usize, seed: usize) -> Projection {
        let block = 32;
        // 0x7F and 0xFF are e4m3's NaN: a checkpoint holds none.
        let bytes: Vec<u8> = (0..out * inp)
            .map(|i| match ((i * 29 + 7 + seed * 53) % 256) as u8 {
                b if b & 0x7F == 0x7F => b - 1,
                b => b,
            })
            .collect();
        let tiles = out.div_ceil(block) * inp.div_ceil(block);
        let scales: Vec<u8> = (0..tiles).map(|i| (118 + ((i + seed) % 4)) as u8).collect();
        Projection::Fp8(Arc::new(
            Fp8Weight::new(mapped(&bytes), mapped(&scales), out, inp, block).unwrap(),
        ))
    }

    fn expert_of(make: &dyn Fn(usize, usize, usize) -> Projection, e: usize) -> BlockScaledExpert {
        BlockScaledExpert {
            gate: make(INTER, DIM, 3 * e),
            up: make(INTER, DIM, 3 * e + 1),
            down: make(DIM, INTER, 3 * e + 2),
        }
    }

    fn rows() -> Tensor {
        Tensor::from_vec(
            (0..ROWS * DIM).map(|i| ((i as f32) * 0.37).sin()).collect(),
            (ROWS, DIM),
            &Device::Cpu,
        )
        .unwrap()
    }

    /// Row `r` routes to experts `r` and `r + 1` modulo the experts, weighed apart.
    fn routing() -> (Vec<u32>, Vec<f32>) {
        let ids = (0..ROWS * K)
            .map(|p| ((p / K + p % K) % EXPERTS) as u32)
            .collect();
        let weights = (0..ROWS * K).map(|p| 0.25 + 0.1 * p as f32).collect();
        (ids, weights)
    }

    /// The layer's experts as the offload's `Expert` runs them, over the same rows grouped the same
    /// way, summed in the router's order.
    fn by_the_offload(
        experts: &[BlockScaledExpert],
        limit: f32,
        x: &Tensor,
        ids: &[u32],
        weights: &[f32],
    ) -> Vec<f32> {
        let mut pair_rows = vec![0f32; ROWS * K * DIM];
        for (e, ex) in experts.iter().enumerate() {
            let pairs: Vec<usize> = (0..ROWS * K).filter(|&p| ids[p] as usize == e).collect();
            if pairs.is_empty() {
                continue;
            }
            let old = Expert {
                w1: ex.gate.clone(),
                w2: ex.down.clone(),
                w3: ex.up.clone(),
            };
            let picked: Vec<u32> = pairs.iter().map(|&p| (p / K) as u32).collect();
            let x_e = x
                .index_select(
                    &Tensor::from_vec(picked, pairs.len(), &Device::Cpu).unwrap(),
                    0,
                )
                .unwrap()
                .contiguous()
                .unwrap();
            let y: Vec<f32> = old
                .forward(&x_e, limit)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (j, &p) in pairs.iter().enumerate() {
                pair_rows[p * DIM..(p + 1) * DIM].copy_from_slice(&y[j * DIM..(j + 1) * DIM]);
            }
        }
        let mut out = vec![-0.0f32; ROWS * DIM];
        for p in 0..ROWS * K {
            let r = p / K;
            for c in 0..DIM {
                out[r * DIM + c] += weights[p] * pair_rows[p * DIM + c];
            }
        }
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    fn same_as_the_offload(make: &dyn Fn(usize, usize, usize) -> Projection, limit: f32) {
        if !crate::cpu::hardware_has_avx2() {
            return;
        }
        let set = BlockScaledExperts {
            experts: (0..EXPERTS).map(|e| expert_of(make, e)).collect(),
            swiglu_limit: limit,
        };
        let x = rows();
        let (ids, weights) = routing();
        let got: Vec<f32> = set
            .forward(&x, &ids, &weights, K)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let want = by_the_offload(&set.experts, limit, &x, &ids, &weights);
        assert!(got.iter().any(|v| *v != 0.0), "the experts give nothing");
        assert_eq!(bits(&got), bits(&want));
    }

    /// fp4 experts give the bits of the offload's experts over the same projections.
    #[test]
    fn fp4_experts_give_the_bits_of_the_offload_experts() {
        same_as_the_offload(&fp4, 0.0);
    }

    /// fp8 experts give the bits of the offload's experts over the same projections.
    #[test]
    fn fp8_experts_give_the_bits_of_the_offload_experts() {
        same_as_the_offload(&fp8, 0.0);
    }

    /// Under a SwiGLU bound the experts still give the bits of the offload's experts.
    #[test]
    fn bounded_block_scaled_experts_give_the_bits_of_the_offload_experts() {
        same_as_the_offload(&fp8, 0.5);
        same_as_the_offload(&fp4, 0.5);
    }

    /// An id past the experts is refused, not read.
    #[test]
    fn an_id_past_the_experts_is_refused() {
        let set = BlockScaledExperts {
            experts: vec![expert_of(&fp8, 0)],
            swiglu_limit: 0.0,
        };
        let x = Tensor::zeros((1, DIM), DType::F32).unwrap();
        assert!(set.forward(&x, &[1], &[1.0], 1).is_err());
    }
}
