//! The nn layer's tests, split out with the items they exercise.

use super::*;

mod tests {
    use super::*;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                ((st >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn linear_matches_oracle() {
        let (b, s, i, o) = (2usize, 3usize, 8usize, 5usize);
        let w = data(o * i, 1);
        let bias = data(o, 2);
        let x = data(b * s * i, 3);

        let nl = Linear::new(
            Tensor::from_vec_f32(w.clone(), vec![o, i]).unwrap(),
            Some(Tensor::from_vec_f32(bias.clone(), vec![o]).unwrap()),
        )
        .unwrap();
        let got = nl
            .forward(&Tensor::from_vec_f32(x.clone(), vec![b, s, i]).unwrap())
            .unwrap();

        // Reference: y[r, c] = Σ_j x[r, j] * w[c, j] + bias[c].
        let mut want = vec![0f32; b * s * o];
        for r in 0..b * s {
            for c in 0..o {
                let mut acc = 0f32;
                for j in 0..i {
                    acc += x[r * i + j] * w[c * i + j];
                }
                want[r * o + c] = acc + bias[c];
            }
        }
        let g = got.to_vec_f32();
        for (i, (a, b)) in g.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-5, "idx {i}: {a} vs {b}");
        }
    }

    #[test]
    fn embedding_lookup() {
        let (v, d) = (7usize, 4usize);
        let table = data(v * d, 4);
        let emb = Embedding::new(Tensor::from_vec_f32(table.clone(), vec![v, d]).unwrap());
        let ids = Tensor::from_vec_u32(vec![3, 0, 6], vec![1, 3]).unwrap();
        let out = emb.forward(&ids).unwrap();
        assert_eq!(out.dims(), &[1, 3, 4]);
        let got = out.to_vec_f32();
        for (j, &id) in [3usize, 0, 6].iter().enumerate() {
            for c in 0..d {
                assert_eq!(got[j * d + c], table[id * d + c]);
            }
        }
    }

    /// `from_transposed(W^T)` must run the exact same forward as `new(W)`  - 
    /// the host-staged big-checkpoint load path depends on it.
    #[test]
    fn linear_from_transposed_matches_new() {
        let (rows, in_dim, out_dim) = (3usize, 6usize, 5usize);
        let wv = data(out_dim * in_dim, 31);
        let bv = data(out_dim, 32);
        let xv = data(rows * in_dim, 33);
        let w = Tensor::from_vec_f32(wv.clone(), vec![out_dim, in_dim]).unwrap();
        let b = Tensor::from_vec_f32(bv.clone(), vec![out_dim]).unwrap();
        let x = Tensor::from_vec_f32(xv, vec![rows, in_dim]).unwrap();

        let via_new = Linear::new(w.clone(), Some(b.clone()))
            .unwrap()
            .forward(&x)
            .unwrap();
        let wt = w.transpose(0, 1).unwrap();
        let via_pre = Linear::from_transposed(wt, Some(b))
            .unwrap()
            .forward(&x)
            .unwrap();
        assert_eq!(via_new.dims(), via_pre.dims());
        assert_eq!(via_new.to_vec_f32(), via_pre.to_vec_f32());
    }

    #[test]
    fn varbuilder_walks_real_file() {
        // reuse any HF safetensors; just verify prefix walking + shape check
        let Some(path) = super::super::safetensors_io::tests_helper_find() else {
            return;
        };
        let vb = unsafe { VarBuilder::from_files(&[&path], DType::F32, &Device::Cpu) }.unwrap();
        // find any tensor, split its name into prefix + leaf, walk and load
        let loader = unsafe { SafeTensorsLoader::multi(&[&path]) }.unwrap();
        let mut names = loader
            .names()
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        names.sort();
        let name = names
            .iter()
            .find(|n| n.contains('.') && loader.load(n).is_ok())
            .cloned();
        let Some(name) = name else { return };
        let (prefix, leaf) = name.rsplit_once('.').unwrap();
        let t = loader.load(&name).unwrap();
        let walked = vb.pp(prefix).get(t.dims().to_vec(), leaf).unwrap();
        assert_eq!(walked.dims(), t.dims());
    }
}

mod lora_tests {
    use super::*;
    use crate::tensor::{DType, Device, Tensor};

    fn lin(in_dim: usize, out_dim: usize) -> Linear {
        // Identity-ish base so the delta's effect is easy to read off.
        let mut w = vec![0f32; out_dim * in_dim];
        for i in 0..out_dim.min(in_dim) {
            w[i * in_dim + i] = 1.0;
        }
        Linear::new(
            Tensor::from_vec_f32(w, vec![out_dim, in_dim]).unwrap(),
            None,
        )
        .unwrap()
    }

    /// The adapter has to CHANGE the output, by the amount its own algebra predicts.
    ///
    /// A low-rank term that silently does nothing is the failure mode here: the base
    /// projection still works, every shape checks out, and the only symptom is that the
    /// user's LoRA appears to have no effect.
    #[test]
    fn a_lora_shifts_the_output_by_its_own_algebra() {
        let (din, dout, r) = (4usize, 4usize, 2usize);
        let mut l = lin(din, dout);
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, din]).unwrap();
        let base = l.forward(&x).unwrap().to_vec_f32();
        assert_eq!(base, vec![1.0, 2.0, 3.0, 4.0], "identity base");

        // down [in, r] all ones, up [r, out] all ones: every output gets
        // scale * r * sum(x) = 0.5 * 2 * 10 = 10.
        let down = Tensor::from_vec_f32(vec![1.0; din * r], vec![din, r]).unwrap();
        let up = Tensor::from_vec_f32(vec![1.0; r * dout], vec![r, dout]).unwrap();
        l.add_lora(LoraDelta {
            down,
            up,
            scale: 0.5,
        })
        .unwrap();
        let got = l.forward(&x).unwrap().to_vec_f32();
        for (i, v) in got.iter().enumerate() {
            assert!((v - (base[i] + 10.0)).abs() < 1e-5, "out[{i}] = {v}");
        }

        // Two adapters compose by addition, and clearing restores the base exactly.
        let down2 = Tensor::from_vec_f32(vec![1.0; din * r], vec![din, r]).unwrap();
        let up2 = Tensor::from_vec_f32(vec![1.0; r * dout], vec![r, dout]).unwrap();
        l.add_lora(LoraDelta {
            down: down2,
            up: up2,
            scale: 0.5,
        })
        .unwrap();
        let two = l.forward(&x).unwrap().to_vec_f32();
        for (i, v) in two.iter().enumerate() {
            assert!(
                (v - (base[i] + 20.0)).abs() < 1e-5,
                "composed out[{i}] = {v}"
            );
        }
        assert_eq!(l.lora_count(), 2);
        l.clear_lora();
        assert_eq!(
            l.forward(&x).unwrap().to_vec_f32(),
            base,
            "clear must restore the base"
        );
    }

    /// Adapters of DIFFERENT ranks must still fuse to the sum of their contributions.
    ///
    /// The fusion concatenates `down` along its rank axis and `up` along the matching
    /// one; equal ranks would not distinguish the right axis from the wrong one, and a
    /// wrong axis here still produces a plausible number.
    #[test]
    fn adapters_of_unequal_rank_fuse_to_their_sum() {
        let (din, dout) = (4usize, 4usize);
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, din]).unwrap();
        let mk = |r: usize, scale: f32| LoraDelta {
            down: Tensor::from_vec_f32(vec![1.0; din * r], vec![din, r]).unwrap(),
            up: Tensor::from_vec_f32(vec![1.0; r * dout], vec![r, dout]).unwrap(),
            scale,
        };
        // Each adapter contributes scale * r * sum(x) to every output.
        let mut alone_a = lin(din, dout);
        alone_a.add_lora(mk(2, 0.5)).unwrap();
        let a = alone_a.forward(&x).unwrap().to_vec_f32();
        let mut alone_b = lin(din, dout);
        alone_b.add_lora(mk(3, 0.25)).unwrap();
        let b = alone_b.forward(&x).unwrap().to_vec_f32();

        let mut both = lin(din, dout);
        both.add_lora(mk(2, 0.5)).unwrap();
        both.add_lora(mk(3, 0.25)).unwrap();
        let got = both.forward(&x).unwrap().to_vec_f32();
        let base = lin(din, dout).forward(&x).unwrap().to_vec_f32();
        for i in 0..dout {
            let want = a[i] + b[i] - base[i];
            assert!(
                (got[i] - want).abs() < 1e-5,
                "out[{i}] = {} want {want}",
                got[i]
            );
        }
    }

    /// An adapter at strength zero must leave the output bit-identical to the base.
    ///
    /// It is the natural way to disable one without changing the request, so it has to
    /// cost nothing AND change nothing - not "change nothing to within rounding".
    #[test]
    fn a_zero_strength_adapter_is_exactly_the_base() {
        let (din, dout, r) = (4usize, 4usize, 2usize);
        let x = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, din]).unwrap();
        let mut l = lin(din, dout);
        let base = l.forward(&x).unwrap().to_vec_f32();
        l.add_lora(LoraDelta {
            down: Tensor::from_vec_f32(vec![1.0; din * r], vec![din, r]).unwrap(),
            up: Tensor::from_vec_f32(vec![1.0; r * dout], vec![r, dout]).unwrap(),
            scale: 0.0,
        })
        .unwrap();
        assert_eq!(l.forward(&x).unwrap().to_vec_f32(), base);
        // Still reported as attached: the caller asked for it, it is simply disabled.
        assert_eq!(l.lora_count(), 1);
    }

    /// A mismatched adapter must be refused, not silently applied to the wrong axis.
    #[test]
    fn a_wrongly_shaped_lora_is_refused() {
        let mut l = lin(4, 4);
        let down = Tensor::from_vec_f32(vec![1.0; 8 * 2], vec![8, 2]).unwrap();
        let up = Tensor::from_vec_f32(vec![1.0; 2 * 4], vec![2, 4]).unwrap();
        assert!(l
            .add_lora(LoraDelta {
                down,
                up,
                scale: 1.0
            })
            .is_err());
        // Rank disagreement between the two halves is also a refusal.
        let down = Tensor::from_vec_f32(vec![1.0; 4 * 2], vec![4, 2]).unwrap();
        let up = Tensor::from_vec_f32(vec![1.0; 3 * 4], vec![3, 4]).unwrap();
        assert!(l
            .add_lora(LoraDelta {
                down,
                up,
                scale: 1.0
            })
            .is_err());
        let _ = DType::F32;
        let _ = Device::Cpu;
    }
}

// The in-memory source, which the file-backed `varbuilder_walks_real_file` above cannot
// reach. These three came over with the compat loader they used to test: the source is a
// different arm of the same enum, and dropping its coverage along with the duplicate would
// have left `from_tensors` - the arm a test fixture and eagle both go through - unexercised.

fn map_vb() -> VarBuilder {
    let dev = Device::Cpu;
    let mut m = std::collections::HashMap::new();
    for (name, shape) in [
        ("enc.fc.weight", vec![8usize, 4]),
        ("enc.fc.bias", vec![8]),
        ("enc.ln.weight", vec![8]),
        ("enc.ln.bias", vec![8]),
    ] {
        m.insert(
            name.to_string(),
            Tensor::randn(0f32, 0.1, shape.as_slice(), &dev).unwrap(),
        );
    }
    VarBuilder::from_tensors(m, DType::F32, &dev)
}

#[test]
fn a_prefix_walk_over_tensors_in_hand_resolves_the_same_dotted_paths() {
    let vb = map_vb();
    let enc = vb.pp("enc");
    assert!(enc.contains("fc.weight"));
    assert_eq!(enc.pp("fc").get((8, 4), "weight").unwrap().dims(), &[8, 4]);
    // A shape the caller declares wrongly is refused rather than reinterpreted.
    assert!(enc.pp("fc").get((4, 8), "weight").is_err());
    // So is a name the map does not hold.
    assert!(enc.get(8, "nope").is_err());
}

#[test]
fn the_builders_read_that_source_into_working_layers() {
    let dev = Device::Cpu;
    let vb = map_vb();
    let x = Tensor::randn(0f32, 1.0, (3, 4), &dev).unwrap();
    let y = linear(4, 8, &vb.pp("enc").pp("fc"))
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_eq!(y.dims(), &[3, 8]);
    let z = layer_norm(8, 1e-5, &vb.pp("enc").pp("ln"))
        .unwrap()
        .forward(&y)
        .unwrap();
    assert_eq!(z.dims(), &[3, 8]);
}

#[test]
fn a_padded_conv1d_preserves_its_length() {
    let dev = Device::Cpu;
    let mut m = std::collections::HashMap::new();
    m.insert(
        "w".to_string(),
        Tensor::randn(0f32, 0.1, (6, 4, 3), &dev).unwrap(),
    );
    m.insert("b".to_string(), Tensor::randn(0f32, 0.1, 6, &dev).unwrap());
    let vb = VarBuilder::from_tensors(m, DType::F32, &dev);
    let conv = Conv1d::new(
        vb.get((6, 4, 3), "w").unwrap(),
        Some(vb.get(6, "b").unwrap()),
        Conv1dConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let x = Tensor::randn(0f32, 1.0, (1, 4, 10), &dev).unwrap();
    // padding 1, kernel 3, stride 1 - the length is preserved.
    assert_eq!(conv.forward(&x).unwrap().dims(), &[1, 6, 10]);
}
