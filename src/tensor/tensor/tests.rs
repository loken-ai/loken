use super::*;

#[test]
fn transpose_fast_path_matches_reference() -> Result<()> {
    // The block-copy fast path (last axis kept) and the general per-element
    // gather must both equal a manual index-mapped reference, across shapes
    // that exercise both branches. Covers the attention q/k/v transpose(1,2).
    let cases: &[(&[usize], usize, usize)] = &[
        (&[2, 3, 4, 5], 1, 2), // 4D middle swap -> fast path
        (&[1, 7, 6, 8], 1, 2), // batch=1 attention layout -> fast path
        (&[2, 3, 4, 5], 0, 2), // last axis kept, non-adjacent -> fast path
        (&[2, 3, 4, 5], 2, 3), // swaps last axis -> general gather
        (&[6, 4], 0, 1),       // 2D -> general gather
        (&[3, 5, 4], 0, 2),    // 3D last axis kept -> fast path
    ];
    for &(dims, a, b) in cases {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        let t = Tensor::from_vec_f32(data.clone(), dims.to_vec())?;
        let got = t.transpose(a, b)?.contiguous()?.to_vec_f32();
        // Reference: for each output linear index, map back to input.
        let mut odims = dims.to_vec();
        odims.swap(a, b);
        let ostride = Shape::from(odims.clone()).stride_contiguous();
        let istride = Shape::from(dims.to_vec()).stride_contiguous();
        let mut perm: Vec<usize> = (0..dims.len()).collect();
        perm.swap(a, b);
        let mut want = vec![0f32; n];
        for (o, w) in want.iter_mut().enumerate() {
            let mut rem = o;
            let mut ii = 0usize;
            for (ax, &os) in ostride.iter().enumerate() {
                let idx = rem / os;
                rem %= os;
                ii += idx * istride[perm[ax]];
            }
            *w = data[ii];
        }
        assert_eq!(got, want, "transpose {dims:?} swap({a},{b})");
    }
    Ok(())
}

#[test]
fn storage_ptr_id_tracks_shared_storage() -> Result<()> {
    let a = Tensor::from_vec_f32(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2))?;
    // clones + metadata-only views share the allocation -> same id
    assert_eq!(a.storage_ptr_id(), a.clone().storage_ptr_id());
    assert_eq!(a.storage_ptr_id(), a.reshape(4)?.storage_ptr_id());
    assert_eq!(a.storage_ptr_id(), a.unsqueeze(0)?.storage_ptr_id());
    // a tensor with its own storage differs (`b` held alive alongside
    // `a`, so the two allocations must coexist -> distinct addresses)
    let b = Tensor::from_vec_f32(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2))?;
    assert_ne!(a.storage_ptr_id(), b.storage_ptr_id());
    // data-copying ops allocate fresh storage
    let c = a.scale(2.0)?;
    assert_ne!(a.storage_ptr_id(), c.storage_ptr_id());
    Ok(())
}

#[test]
fn gemm_f16w_matches_upcast_path() -> Result<()> {
    // The direct f16 CPU matmul must be BIT-identical to the old
    // upcast-both-operands-to-f32 route (exact conversions + same
    // accumulation order). Shapes cross the KB=256 / NT=512 tile edges,
    // exercise the 4-row quad + tail loops, and the batched branch.
    let cases: &[(usize, usize, usize, usize)] = &[
        (1, 1, 300, 600),  // decode GEMV, k crosses KB, n crosses NT
        (1, 5, 257, 513),  // quad + tail rows
        (4, 3, 64, 100),   // small batched
        (32, 1, 64, 1024), // batch-parallel branch (lbatch*per >= 1<<20)
    ];
    for &(bsz, m, k, n) in cases {
        let mk = |len: usize, s: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32 * s).sin() * 3.0) as f32)
                .collect()
        };
        let a = Tensor::from_vec_f32(mk(bsz * m * k, 0.37), (bsz, m, k))?.to_dtype(DType::F16)?;
        let b = Tensor::from_vec_f32(mk(bsz * k * n, 0.11), (bsz, k, n))?.to_dtype(DType::F16)?;
        let fast = a.matmul(&b)?;
        let slow = a
            .to_dtype(DType::F32)?
            .matmul(&b.to_dtype(DType::F32)?)?
            .to_dtype(DType::F16)?;
        assert_eq!(fast.dims(), &[bsz, m, n]);
        let (f, s) = (fast.to_vec_f32(), slow.to_vec_f32());
        assert_eq!(
            f, s,
            "f16 matmul diverged from upcast path at {bsz}x{m}x{k}x{n}"
        );
    }
    Ok(())
}

#[test]
fn cos_get_on_dim_contiguous() -> Result<()> {
    let t = Tensor::from_vec_f32(
        (0..24).map(|i| i as f32 * 0.3).collect::<Vec<f32>>(),
        (2, 3, 4),
    )?;
    // cos matches scalar cosf
    for (i, v) in t.cos()?.to_vec_f32().iter().enumerate() {
        assert!((v - (i as f32 * 0.3).cos()).abs() < 1e-6);
    }
    // get_on_dim = narrow+squeeze
    let g = t.get_on_dim(1, 2)?;
    assert_eq!(g.dims(), &[2, 4]);
    let want = t.narrow(1, 2, 1)?.reshape((2, 4))?.to_vec_f32();
    assert_eq!(g.to_vec_f32(), want);
    // contiguous is identity on the packed substrate
    assert_eq!(t.contiguous()?.to_vec_f32(), t.to_vec_f32());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_narrow_matches_cpu() -> Result<()> {
    let Some(dev) = crate::tensor::cuda::CudaDevice::get(0).ok() else {
        return Ok(());
    };
    let gpu = crate::tensor::Device::Cuda(dev);
    let v: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let t_cpu = Tensor::from_vec_f32(v, (4, 5, 6))?;
    let t_gpu = t_cpu.to_device(&gpu)?;
    for (d, s, l) in [(0usize, 1usize, 2usize), (1, 2, 3), (2, 1, 4)] {
        let want = t_cpu.narrow(d, s, l)?.to_vec_f32();
        let got = t_gpu.narrow(d, s, l)?.to_vec_f32();
        assert_eq!(want, got, "narrow dim {d} start {s} len {l}");
    }
    Ok(())
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "idx {i}: {x} vs {y}");
    }
}

/// Deterministic pseudo-random data (no rand dep in tests).
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

// -- hand-rolled reference implementations (the test oracles) ---------

fn ref_strides(dims: &[usize]) -> Vec<usize> {
    let mut st = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * dims[i + 1];
    }
    st
}

fn ref_decompose(flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut rem = flat;
    strides
        .iter()
        .map(|&s| {
            let q = rem / s;
            rem %= s;
            q
        })
        .collect()
}

/// Batched matmul reference: `[b,m,k] @ [b,k,n] -> [b,m,n]`.
fn ref_matmul(a: &[f32], w: &[f32], b: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; b * m * n];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for t in 0..k {
                    acc += a[bi * m * k + i * k + t] * w[bi * k * n + t * n + j];
                }
                out[bi * m * n + i * n + j] = acc;
            }
        }
    }
    out
}

/// Row-wise softmax reference (max-subtracted).
fn ref_softmax_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for c in 0..cols {
            out[r * cols + c] = exps[c] / sum;
        }
    }
    out
}

/// Row-wise RMS-norm reference.
fn ref_rms_norm(x: &[f32], w: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = row[c] * inv * w[c];
        }
    }
    out
}

/// RoPE reference on `[b,h,seq,d]` with `cos/sin [seq, d/2]`.
/// `interleaved=false`: pairs (j, j+d/2); `interleaved=true`: (2j, 2j+1).
fn ref_rope(
    x: &[f32],
    cs: &[f32],
    sn: &[f32],
    b: usize,
    h: usize,
    seq: usize,
    d: usize,
    interleaved: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for bi in 0..b {
        for hi in 0..h {
            for t in 0..seq {
                let base = ((bi * h + hi) * seq + t) * d;
                for j in 0..d / 2 {
                    let (i1, i2) = if interleaved {
                        (2 * j, 2 * j + 1)
                    } else {
                        (j, j + d / 2)
                    };
                    let (c, s) = (cs[t * d / 2 + j], sn[t * d / 2 + j]);
                    let (x1, x2) = (x[base + i1], x[base + i2]);
                    out[base + i1] = x1 * c - x2 * s;
                    out[base + i2] = x1 * s + x2 * c;
                }
            }
        }
    }
    out
}

/// Transpose reference: swap dims `a` and `b` of a contiguous tensor.
fn ref_transpose(x: &[f32], dims: &[usize], a: usize, b: usize) -> (Vec<usize>, Vec<f32>) {
    let mut nd = dims.to_vec();
    nd.swap(a, b);
    let st = ref_strides(dims);
    let nst = ref_strides(&nd);
    let mut out = vec![0f32; x.len()];
    for (flat, &v) in x.iter().enumerate() {
        let mut idx = ref_decompose(flat, &st);
        idx.swap(a, b);
        let nflat: usize = idx.iter().zip(&nst).map(|(i, s)| i * s).sum();
        out[nflat] = v;
    }
    (nd, out)
}

/// Narrow reference: keep `dim` indices in `[start, start+len)`.
fn ref_narrow(
    x: &[f32],
    dims: &[usize],
    dim: usize,
    start: usize,
    len: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut nd = dims.to_vec();
    nd[dim] = len;
    let st = ref_strides(dims);
    let nst = ref_strides(&nd);
    let mut out = vec![0f32; nd.iter().product()];
    for (nflat, o) in out.iter_mut().enumerate() {
        let mut idx = ref_decompose(nflat, &nst);
        idx[dim] += start;
        let flat: usize = idx.iter().zip(&st).map(|(i, s)| i * s).sum();
        *o = x[flat];
    }
    (nd, out)
}

/// Concatenation reference along `dim` (two inputs).
fn ref_cat(a: &[f32], ad: &[usize], b: &[f32], bd: &[usize], dim: usize) -> (Vec<usize>, Vec<f32>) {
    let mut nd = ad.to_vec();
    nd[dim] += bd[dim];
    let nst = ref_strides(&nd);
    let ast = ref_strides(ad);
    let bst = ref_strides(bd);
    let mut out = vec![0f32; nd.iter().product()];
    for (nflat, o) in out.iter_mut().enumerate() {
        let mut idx = ref_decompose(nflat, &nst);
        if idx[dim] < ad[dim] {
            let flat: usize = idx.iter().zip(&ast).map(|(i, s)| i * s).sum();
            *o = a[flat];
        } else {
            idx[dim] -= ad[dim];
            let flat: usize = idx.iter().zip(&bst).map(|(i, s)| i * s).sum();
            *o = b[flat];
        }
    }
    (nd, out)
}

#[test]
fn matmul_matches_oracle() {
    for (b, m, k, n) in [(1usize, 4usize, 6usize, 5usize), (3, 8, 16, 7)] {
        let a = data(b * m * k, 1);
        let w = data(b * k * n, 2);
        let nt = Tensor::from_vec_f32(a.clone(), vec![b, m, k])
            .unwrap()
            .matmul(&Tensor::from_vec_f32(w.clone(), vec![b, k, n]).unwrap())
            .unwrap();
        let want = ref_matmul(&a, &w, b, m, k, n);
        close(&nt.to_vec_f32(), &want, 1e-4);
    }
}

#[test]
fn softmax_matches_oracle() {
    let x = data(2 * 3 * 17, 3);
    let nt = Tensor::from_vec_f32(x.clone(), vec![2, 3, 17])
        .unwrap()
        .softmax_last_dim()
        .unwrap();
    let want = ref_softmax_rows(&x, 2 * 3, 17);
    close(&nt.to_vec_f32(), &want, 1e-6);
}

#[test]
fn transpose_matches_oracle() {
    let x = data(2 * 3 * 4 * 5, 7);
    let nt = Tensor::from_vec_f32(x.clone(), vec![2, 3, 4, 5]).unwrap();
    for (a, b) in [(0usize, 2usize), (1, 3), (2, 3)] {
        let n = nt.transpose(a, b).unwrap();
        let (wd, wv) = ref_transpose(&x, &[2, 3, 4, 5], a, b);
        assert_eq!(n.dims(), &wd[..]);
        close(&n.to_vec_f32(), &wv, 0.0);
    }
}

#[test]
fn narrow_cat_match_oracle() {
    let x = data(3 * 7 * 4, 8);
    let nt = Tensor::from_vec_f32(x.clone(), vec![3, 7, 4]).unwrap();
    let n = nt.narrow(1, 2, 4).unwrap();
    let (wd, wv) = ref_narrow(&x, &[3, 7, 4], 1, 2, 4);
    close(&n.to_vec_f32(), &wv, 0.0);

    let y = data(3 * 2 * 4, 9);
    let ny = Tensor::from_vec_f32(y.clone(), vec![3, 2, 4]).unwrap();
    let n2 = Tensor::cat(&[&n, &ny], 1).unwrap();
    let (wd2, wv2) = ref_cat(&wv, &wd, &y, &[3, 2, 4], 1);
    assert_eq!(n2.dims(), &wd2[..]);
    close(&n2.to_vec_f32(), &wv2, 0.0);
}

#[test]
fn broadcast_matches_oracle() {
    let x = data(2 * 3 * 8, 10);
    let w = data(8, 11);
    let nx = Tensor::from_vec_f32(x.clone(), vec![2, 3, 8]).unwrap();
    let nw = Tensor::from_vec_f32(w.clone(), vec![8]).unwrap();
    let nadd = nx.broadcast_add(&nw).unwrap();
    let nmul = nx.broadcast_mul(&nw).unwrap();
    let wadd: Vec<f32> = x.iter().enumerate().map(|(i, v)| v + w[i % 8]).collect();
    let wmul: Vec<f32> = x.iter().enumerate().map(|(i, v)| v * w[i % 8]).collect();
    close(&nadd.to_vec_f32(), &wadd, 0.0);
    close(&nmul.to_vec_f32(), &wmul, 0.0);
    // middle-axis broadcast: [2,1,8] over [2,3,8]
    let m = data(2 * 8, 12);
    let nm = Tensor::from_vec_f32(m.clone(), vec![2, 1, 8]).unwrap();
    let wmid: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v + m[(i / 24) * 8 + i % 8])
        .collect();
    close(&nx.broadcast_add(&nm).unwrap().to_vec_f32(), &wmid, 0.0);
}

#[test]
fn index_select_matches_oracle() {
    let table = data(11 * 6, 13);
    let ids: Vec<u32> = vec![3, 0, 10, 7, 3];
    let nt = Tensor::from_vec_f32(table.clone(), vec![11, 6]).unwrap();
    let nids = Tensor::from_vec_u32(ids.clone(), vec![5]).unwrap();
    let n = nt.index_select(&nids, 0).unwrap();
    assert_eq!(n.dims(), &[5, 6]);
    let mut want = Vec::with_capacity(5 * 6);
    for &id in &ids {
        want.extend_from_slice(&table[id as usize * 6..(id as usize + 1) * 6]);
    }
    close(&n.to_vec_f32(), &want, 0.0);
}

#[test]
fn rms_norm_matches_oracle() {
    let x = data(4 * 32, 14);
    let w = data(32, 15);
    let n = Tensor::from_vec_f32(x.clone(), vec![4, 32])
        .unwrap()
        .rms_norm(&Tensor::from_vec_f32(w.clone(), vec![32]).unwrap(), 1e-6)
        .unwrap();
    let want = ref_rms_norm(&x, &w, 4, 32, 1e-6);
    close(&n.to_vec_f32(), &want, 1e-5);
}

/// The wide RMSNorm kernel against the narrow one, at the width a decode step uses.
///
/// The oracle above runs 32 columns, below the 256 the wide kernel needs, so it judged the
/// narrow kernel only - the whole suite did, and every decode takes the wide one.
///
/// The dispatch picks by shape: one row takes the wide kernel, sixty-five the narrow one. So
/// the same row is fed to both. They share an accumulation order and the claim is bit
/// identity, so that is what is asserted - a tolerance here would pass a kernel that reduces
/// in a different order. The reference check that follows keeps two kernels agreeing on a
/// wrong answer from reading as a pass.
#[cfg(feature = "cuda")]
#[test]
fn the_wide_rmsnorm_matches_the_narrow_one_bit_for_bit() {
    let Ok(dev) = crate::tensor::cuda::CudaDevice::get(0) else {
        return;
    };
    let gpu = crate::tensor::Device::Cuda(dev);
    const COLS: usize = 5120;
    const ROWS: usize = 65;
    // Without this the test would compare a kernel to itself the day the threshold moves,
    // and pass while judging nothing.
    let (one, _) = crate::tensor::cuda::rms_norm_launch(1, COLS);
    let (many_k, _) = crate::tensor::cuda::rms_norm_launch(ROWS, COLS);
    assert_eq!(one, "fused_rmsnorm_wide_f32");
    assert_ne!(
        one, many_k,
        "both shapes now take {one}: this compares nothing"
    );
    let row = data(COLS, 21);
    let w = data(COLS, 22);
    let weight = |g: &crate::tensor::Device| {
        Tensor::from_vec_f32(w.clone(), vec![COLS])
            .unwrap()
            .to_device(g)
            .unwrap()
    };
    let wide = Tensor::from_vec_f32(row.clone(), vec![1, COLS])
        .unwrap()
        .to_device(&gpu)
        .unwrap()
        .rms_norm(&weight(&gpu), 1e-6)
        .unwrap()
        .to_vec_f32();
    let mut many = Vec::with_capacity(ROWS * COLS);
    for _ in 0..ROWS {
        many.extend_from_slice(&row);
    }
    let narrow = Tensor::from_vec_f32(many, vec![ROWS, COLS])
        .unwrap()
        .to_device(&gpu)
        .unwrap()
        .rms_norm(&weight(&gpu), 1e-6)
        .unwrap()
        .to_vec_f32();
    assert_eq!(wide.len(), COLS);
    for c in 0..COLS {
        assert_eq!(
            wide[c].to_bits(),
            narrow[c].to_bits(),
            "column {c}: wide {} vs narrow {}",
            wide[c],
            narrow[c]
        );
    }
    close(&wide, &ref_rms_norm(&row, &w, 1, COLS, 1e-6), 1e-4);
}

/// The wide add+RMSNorm kernel against the narrow one, both outputs.
///
/// The narrow kernel writes the summed row to global memory and reads it back to normalise;
/// the wide one keeps it in shared memory. Both outputs are compared, because a staging bug
/// could leave `sum_out` right and `norm_out` wrong, or the reverse.
#[cfg(feature = "cuda")]
#[test]
fn the_wide_add_rmsnorm_matches_the_narrow_one_bit_for_bit() {
    use crate::inference::kernel::fused::fused_add_rmsnorm_dual;
    let Ok(dev) = crate::tensor::cuda::CudaDevice::get(0) else {
        return;
    };
    let gpu = crate::tensor::Device::Cuda(dev);
    const COLS: usize = 5120;
    const ROWS: usize = 65;
    let (one, _) = crate::tensor::cuda::add_rms_norm_launch(1, COLS);
    let (many_k, _) = crate::tensor::cuda::add_rms_norm_launch(ROWS, COLS);
    assert_eq!(one, "fused_add_rmsnorm_dual_wide_f32");
    assert_ne!(
        one, many_k,
        "both shapes now take {one}: this compares nothing"
    );

    let row = data(COLS, 23);
    let res = data(COLS, 24);
    let w = data(COLS, 25);
    let on_gpu = |v: &Vec<f32>, rows: usize| {
        let mut all = Vec::with_capacity(rows * COLS);
        for _ in 0..rows {
            all.extend_from_slice(v);
        }
        Tensor::from_vec_f32(all, vec![rows, COLS])
            .unwrap()
            .to_device(&gpu)
            .unwrap()
    };
    let weight = Tensor::from_vec_f32(w.clone(), vec![COLS])
        .unwrap()
        .to_device(&gpu)
        .unwrap();
    let (ws, wn) = fused_add_rmsnorm_dual(&on_gpu(&row, 1), &on_gpu(&res, 1), &weight, 1e-6)
        .expect("wide launch");
    let (ns, nn) = fused_add_rmsnorm_dual(&on_gpu(&row, ROWS), &on_gpu(&res, ROWS), &weight, 1e-6)
        .expect("narrow launch");
    for (what, wide, narrow) in [
        ("sum", ws.to_vec_f32(), ns.to_vec_f32()),
        ("norm", wn.to_vec_f32(), nn.to_vec_f32()),
    ] {
        assert_eq!(wide.len(), COLS);
        for c in 0..COLS {
            assert_eq!(
                wide[c].to_bits(),
                narrow[c].to_bits(),
                "{what} column {c}: wide {} vs narrow {}",
                wide[c],
                narrow[c]
            );
        }
    }
    let summed: Vec<f32> = row.iter().zip(&res).map(|(a, b)| a + b).collect();
    close(
        &wn.to_vec_f32(),
        &ref_rms_norm(&summed, &w, 1, COLS, 1e-6),
        1e-4,
    );
}

#[test]
fn activations_match_oracle() {
    let x = data(2 * 37, 16);
    let nx = Tensor::from_vec_f32(x.clone(), vec![2, 37]).unwrap();
    let wsilu: Vec<f32> = x.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    // tanh-approximated gelu (the substrate's `gelu` semantics)
    let wgelu: Vec<f32> = x
        .iter()
        .map(|&v| 0.5 * v * (1.0 + (0.797_884_56_f32 * (v + 0.044715 * v * v * v)).tanh()))
        .collect();
    close(&nx.silu().unwrap().to_vec_f32(), &wsilu, 1e-6);
    close(&nx.gelu().unwrap().to_vec_f32(), &wgelu, 1e-5);
}

#[test]
fn rope_matches_oracle() {
    let (b, h, seq, d) = (1usize, 3usize, 5usize, 8usize);
    let x = data(b * h * seq * d, 17);
    let cs = data(seq * d / 2, 18)
        .iter()
        .map(|v| v.cos())
        .collect::<Vec<_>>();
    let sn = data(seq * d / 2, 18)
        .iter()
        .map(|v| v.sin())
        .collect::<Vec<_>>();
    let nx = Tensor::from_vec_f32(x.clone(), vec![b, h, seq, d]).unwrap();
    let nc = Tensor::from_vec_f32(cs.clone(), vec![seq, d / 2]).unwrap();
    let ns = Tensor::from_vec_f32(sn.clone(), vec![seq, d / 2]).unwrap();
    close(
        &nx.rope(&nc, &ns).unwrap().to_vec_f32(),
        &ref_rope(&x, &cs, &sn, b, h, seq, d, false),
        1e-6,
    );
    close(
        &nx.rope_i(&nc, &ns).unwrap().to_vec_f32(),
        &ref_rope(&x, &cs, &sn, b, h, seq, d, true),
        1e-6,
    );
}

#[test]
fn reductions_match_oracle() {
    let dims = [2usize, 5, 7];
    let x = data(dims.iter().product(), 19);
    let nx = Tensor::from_vec_f32(x.clone(), dims.to_vec()).unwrap();
    let st = ref_strides(&dims);
    for d in 0..3usize {
        // reference reduce over dim d (keepdim layout = same iteration
        // order with dims[d] collapsed to 1)
        let mut rd = dims.to_vec();
        rd[d] = 1;
        let out_n: usize = rd.iter().product();
        let mut wsum = vec![0f32; out_n];
        let mut wmax = vec![f32::NEG_INFINITY; out_n];
        let rst = ref_strides(&rd);
        for (flat, &v) in x.iter().enumerate() {
            let mut idx = ref_decompose(flat, &st);
            idx[d] = 0;
            let o: usize = idx.iter().zip(&rst).map(|(i, s)| i * s).sum();
            wsum[o] += v;
            wmax[o] = wmax[o].max(v);
        }
        let wmean: Vec<f32> = wsum.iter().map(|v| v / dims[d] as f32).collect();
        close(&nx.sum_keepdim(d).unwrap().to_vec_f32(), &wsum, 1e-5);
        close(&nx.mean_keepdim(d).unwrap().to_vec_f32(), &wmean, 1e-6);
        close(&nx.max_keepdim(d).unwrap().to_vec_f32(), &wmax, 0.0);
        if d == 1 {
            close(&nx.sum(1).unwrap().to_vec_f32(), &wsum, 1e-5);
        }
    }
}

#[test]
fn to_dtype_roundtrip_matches_oracle() {
    let x = data(64, 20);
    let nx = Tensor::from_vec_f32(x.clone(), vec![64]).unwrap();
    for dt in [DType::F16, DType::BF16] {
        let n = nx.to_dtype(dt).unwrap().to_dtype(DType::F32).unwrap();
        let want: Vec<f32> = x
            .iter()
            .map(|&v| match dt {
                DType::F16 => half::f16::from_f32(v).to_f32(),
                _ => half::bf16::from_f32(v).to_f32(),
            })
            .collect();
        close(&n.to_vec_f32(), &want, 0.0);
    }
}

/// MILESTONE: a full pre-norm transformer layer (RMSNorm -> QKV -> RoPE ->
/// causal attention -> proj -> residual -> RMSNorm -> SwiGLU MLP ->
/// residual) computed natively must match the same math hand-rolled in
/// scalar f32. This is the op set a real decode forward uses.
#[test]
fn transformer_layer_matches_oracle() {
    let (b, h, seq, hd) = (1usize, 2usize, 6usize, 16usize);
    let dm = h * hd; // d_model
    let ff = 3 * dm;
    let x0 = data(b * seq * dm, 21);
    let wq = data(dm * dm, 22);
    let wk = data(dm * dm, 23);
    let wv = data(dm * dm, 24);
    let wo = data(dm * dm, 25);
    let wg = data(dm * ff, 26);
    let wu = data(dm * ff, 27);
    let wd = data(ff * dm, 28);
    let n1 = data(dm, 29)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let n2 = data(dm, 30)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let theta = 10000f32;
    let mut cs = vec![0f32; seq * hd / 2];
    let mut sn = vec![0f32; seq * hd / 2];
    for t in 0..seq {
        for i in 0..hd / 2 {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
            cs[t * hd / 2 + i] = (t as f32 * freq).cos();
            sn[t * hd / 2 + i] = (t as f32 * freq).sin();
        }
    }
    // causal mask as additive -inf upper triangle
    let mut mask = vec![0f32; seq * seq];
    for q in 0..seq {
        for k in (q + 1)..seq {
            mask[q * seq + k] = f32::NEG_INFINITY;
        }
    }
    let scale = 1.0 / (hd as f32).sqrt();

    // ---- native ----
    let native_out = {
        let t = |v: &[f32], s: Vec<usize>| Tensor::from_vec_f32(v.to_vec(), s).unwrap();
        let x = t(&x0, vec![b, seq, dm]);
        let xn = x.rms_norm(&t(&n1, vec![dm]), 1e-6).unwrap();
        let proj = |w: &[f32], o: usize| {
            let w2 = t(w, vec![1, dm, o]); // [b, dm, o] weightᵀ layout for matmul
            xn.matmul(&w2).unwrap()
        };
        let q = proj(&wq, dm)
            .reshape(vec![b, seq, h, hd])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k = proj(&wk, dm)
            .reshape(vec![b, seq, h, hd])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = proj(&wv, dm)
            .reshape(vec![b, seq, h, hd])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let qr = q
            .rope(&t(&cs, vec![seq, hd / 2]), &t(&sn, vec![seq, hd / 2]))
            .unwrap();
        let kr = k
            .rope(&t(&cs, vec![seq, hd / 2]), &t(&sn, vec![seq, hd / 2]))
            .unwrap();
        let scores = qr
            .matmul(&kr.transpose(2, 3).unwrap())
            .unwrap()
            .unary_f32(|x| x * scale)
            .unwrap()
            .broadcast_add(&t(&mask, vec![1, 1, seq, seq]))
            .unwrap();
        let attn = scores.softmax_last_dim().unwrap().matmul(&v).unwrap();
        let merged = attn
            .transpose(1, 2)
            .unwrap()
            .reshape(vec![b, seq, dm])
            .unwrap();
        let o = merged.matmul(&t(&wo, vec![1, dm, dm])).unwrap();
        let x1 = x.add(&o).unwrap();
        let x1n = x1.rms_norm(&t(&n2, vec![dm]), 1e-6).unwrap();
        let g = x1n
            .matmul(&t(&wg, vec![1, dm, ff]))
            .unwrap()
            .silu()
            .unwrap();
        let u = x1n.matmul(&t(&wu, vec![1, dm, ff])).unwrap();
        let m = g.mul(&u).unwrap().matmul(&t(&wd, vec![1, ff, dm])).unwrap();
        x1.add(&m).unwrap().to_vec_f32()
    };

    // ---- scalar reference (identical math, hand-rolled) ----
    let oracle_out = {
        // x0 as [seq, dm] (b = 1)
        let xn = ref_rms_norm(&x0, &n1, seq, dm, 1e-6);
        // proj: [seq, dm] @ [dm, o]
        let proj = |xin: &[f32], w: &[f32], o: usize| ref_matmul(xin, w, 1, seq, dm, o);
        let qp = proj(&xn, &wq, dm);
        let kp = proj(&xn, &wk, dm);
        let vp = proj(&xn, &wv, dm);
        // [seq, h, hd] -> [h, seq, hd]
        let to_heads = |p: &[f32]| {
            let mut out = vec![0f32; seq * dm];
            for t in 0..seq {
                for hh in 0..h {
                    for dd in 0..hd {
                        out[(hh * seq + t) * hd + dd] = p[(t * h + hh) * hd + dd];
                    }
                }
            }
            out
        };
        let q = to_heads(&qp);
        let k = to_heads(&kp);
        let v = to_heads(&vp);
        let qr = ref_rope(&q, &cs, &sn, 1, h, seq, hd, false);
        let kr = ref_rope(&k, &cs, &sn, 1, h, seq, hd, false);
        // scores[h, t, t2] -> softmax -> ctx[h, t, hd]
        let mut ctx = vec![0f32; h * seq * hd];
        for hh in 0..h {
            let mut scores = vec![0f32; seq * seq];
            for t in 0..seq {
                for t2 in 0..seq {
                    let mut acc = 0f32;
                    for dd in 0..hd {
                        acc += qr[(hh * seq + t) * hd + dd] * kr[(hh * seq + t2) * hd + dd];
                    }
                    scores[t * seq + t2] = acc * scale + mask[t * seq + t2];
                }
            }
            let probs = ref_softmax_rows(&scores, seq, seq);
            for t in 0..seq {
                for dd in 0..hd {
                    let mut acc = 0f32;
                    for t2 in 0..seq {
                        acc += probs[t * seq + t2] * v[(hh * seq + t2) * hd + dd];
                    }
                    ctx[(hh * seq + t) * hd + dd] = acc;
                }
            }
        }
        // merge heads back to [seq, dm]
        let mut merged = vec![0f32; seq * dm];
        for t in 0..seq {
            for hh in 0..h {
                for dd in 0..hd {
                    merged[(t * h + hh) * hd + dd] = ctx[(hh * seq + t) * hd + dd];
                }
            }
        }
        let o = ref_matmul(&merged, &wo, 1, seq, dm, dm);
        let x1: Vec<f32> = x0.iter().zip(&o).map(|(a, b)| a + b).collect();
        let x1n = ref_rms_norm(&x1, &n2, seq, dm, 1e-6);
        let g: Vec<f32> = ref_matmul(&x1n, &wg, 1, seq, dm, ff)
            .iter()
            .map(|v| v / (1.0 + (-v).exp()))
            .collect();
        let u = ref_matmul(&x1n, &wu, 1, seq, dm, ff);
        let gu: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a * b).collect();
        let m = ref_matmul(&gu, &wd, 1, seq, ff, dm);
        x1.iter().zip(&m).map(|(a, b)| a + b).collect::<Vec<f32>>()
    };

    close(&native_out, &oracle_out, 1e-3);
}

/// Direct-convolution references.
fn ref_conv1d(
    x: &[f32],
    w: &[f32],
    b: usize,
    ci: usize,
    l: usize,
    co: usize,
    k: usize,
    pad: usize,
    stride: usize,
    groups: usize,
) -> (usize, Vec<f32>) {
    let lo = (l + 2 * pad - (k - 1) - 1) / stride + 1;
    let cig = ci / groups;
    let cog = co / groups;
    let mut out = vec![0f32; b * co * lo];
    for bi in 0..b {
        for oc in 0..co {
            let g = oc / cog;
            for j in 0..lo {
                let mut acc = 0f32;
                for icg in 0..cig {
                    let ic = g * cig + icg;
                    for kk in 0..k {
                        let pos = j * stride + kk;
                        if pos < pad || pos - pad >= l {
                            continue;
                        }
                        acc += x[(bi * ci + ic) * l + pos - pad] * w[(oc * cig + icg) * k + kk];
                    }
                }
                out[(bi * co + oc) * lo + j] = acc;
            }
        }
    }
    (lo, out)
}

fn ref_conv2d_3x3(
    x: &[f32],
    w: &[f32],
    b: usize,
    ci: usize,
    h: usize,
    wid: usize,
    co: usize,
    pad: usize,
    stride: usize,
) -> (usize, usize, Vec<f32>) {
    let ho = (h + 2 * pad - 2 - 1) / stride + 1;
    let wo = (wid + 2 * pad - 2 - 1) / stride + 1;
    let mut out = vec![0f32; b * co * ho * wo];
    for bi in 0..b {
        for oc in 0..co {
            for oy in 0..ho {
                for ox in 0..wo {
                    let mut acc = 0f32;
                    for ic in 0..ci {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = oy * stride + ky;
                                let ix = ox * stride + kx;
                                if iy < pad || ix < pad || iy - pad >= h || ix - pad >= wid {
                                    continue;
                                }
                                acc += x[((bi * ci + ic) * h + iy - pad) * wid + ix - pad]
                                    * w[((oc * ci + ic) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                    out[((bi * co + oc) * ho + oy) * wo + ox] = acc;
                }
            }
        }
    }
    (ho, wo, out)
}

fn ref_conv_transpose1d(
    x: &[f32],
    w: &[f32],
    b: usize,
    ci: usize,
    l: usize,
    co: usize,
    k: usize,
    pad: usize,
    opad: usize,
    stride: usize,
) -> (usize, Vec<f32>) {
    let lo = (l - 1) * stride + k + opad - 2 * pad;
    let mut out = vec![0f32; b * co * lo];
    for bi in 0..b {
        for ic in 0..ci {
            for i in 0..l {
                for kk in 0..k {
                    let j = i * stride + kk;
                    if j < pad || j - pad >= lo {
                        continue;
                    }
                    for oc in 0..co {
                        out[(bi * co + oc) * lo + j - pad] +=
                            x[(bi * ci + ic) * l + i] * w[(ic * co + oc) * k + kk];
                    }
                }
            }
        }
    }
    (lo, out)
}

#[test]
fn conv_matches_oracle() {
    // conv1d incl groups + padding + stride
    let (b, ci, l, co, k) = (2usize, 4usize, 11usize, 6usize, 3usize);
    for (pad, stride, groups) in [(0usize, 1usize, 1usize), (1, 2, 1), (1, 1, 2)] {
        let x = data(b * ci * l, 50);
        let w = data(co * (ci / groups) * k, 51);
        let n = Tensor::from_vec_f32(x.clone(), vec![b, ci, l])
            .unwrap()
            .conv1d(
                &Tensor::from_vec_f32(w.clone(), vec![co, ci / groups, k]).unwrap(),
                pad,
                stride,
                1,
                groups,
            )
            .unwrap();
        let (lo, want) = ref_conv1d(&x, &w, b, ci, l, co, k, pad, stride, groups);
        assert_eq!(n.dims(), &[b, co, lo], "p{pad}s{stride}g{groups}");
        close(&n.to_vec_f32(), &want, 1e-4);
    }
    // conv2d
    let (h, w2) = (7usize, 8usize);
    for (pad, stride) in [(0usize, 1usize), (1, 2)] {
        let x = data(b * ci * h * w2, 52);
        let wk = data(co * ci * 9, 53);
        let n = Tensor::from_vec_f32(x.clone(), vec![b, ci, h, w2])
            .unwrap()
            .conv2d(
                &Tensor::from_vec_f32(wk.clone(), vec![co, ci, 3, 3]).unwrap(),
                pad,
                stride,
                1,
                1,
            )
            .unwrap();
        let (ho, wo, want) = ref_conv2d_3x3(&x, &wk, b, ci, h, w2, co, pad, stride);
        assert_eq!(n.dims(), &[b, co, ho, wo], "2d p{pad}s{stride}");
        close(&n.to_vec_f32(), &want, 1e-4);
    }
}

#[test]
fn conv_transpose1d_matches_oracle() {
    let (b, ci, l, co, k) = (1usize, 4usize, 9usize, 3usize, 4usize);
    for (pad, opad, stride) in [(0usize, 0usize, 1usize), (1, 0, 2), (2, 1, 4)] {
        let x = data(b * ci * l, 60);
        let w = data(ci * co * k, 61);
        let n = Tensor::from_vec_f32(x.clone(), vec![b, ci, l])
            .unwrap()
            .conv_transpose1d(
                &Tensor::from_vec_f32(w.clone(), vec![ci, co, k]).unwrap(),
                pad,
                opad,
                stride,
                1,
                1,
            )
            .unwrap();
        let (lo, want) = ref_conv_transpose1d(&x, &w, b, ci, l, co, k, pad, opad, stride);
        assert_eq!(n.dims(), &[b, co, lo], "p{pad}o{opad}s{stride}");
        close(&n.to_vec_f32(), &want, 1e-4);
    }
}

#[test]
fn group_norm_matches_oracle() {
    let (b, c, h, w2, g) = (2usize, 8usize, 3usize, 4usize, 4usize);
    let x = data(b * c * h * w2, 80);
    let wv = data(c, 81)
        .iter()
        .map(|v| 1.0 + 0.1 * v)
        .collect::<Vec<_>>();
    let bv = data(c, 82);
    let n = Tensor::from_vec_f32(x.clone(), vec![b, c, h, w2])
        .unwrap()
        .group_norm(
            g,
            &Tensor::from_vec_f32(wv.clone(), vec![c]).unwrap(),
            &Tensor::from_vec_f32(bv.clone(), vec![c]).unwrap(),
            1e-5,
        )
        .unwrap();
    // reference: per (batch, group) mean/var over (c/g)*h*w elements
    let cg = c / g;
    let hw = h * w2;
    let mut want = vec![0f32; b * c * hw];
    for bi in 0..b {
        for gi in 0..g {
            let mut mean = 0f64;
            let mut var = 0f64;
            for ic in 0..cg {
                for p in 0..hw {
                    mean += x[(bi * c + gi * cg + ic) * hw + p] as f64;
                }
            }
            mean /= (cg * hw) as f64;
            for ic in 0..cg {
                for p in 0..hw {
                    let d = x[(bi * c + gi * cg + ic) * hw + p] as f64 - mean;
                    var += d * d;
                }
            }
            var /= (cg * hw) as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for ic in 0..cg {
                let ch = gi * cg + ic;
                for p in 0..hw {
                    let v = (x[(bi * c + ch) * hw + p] as f64 - mean) * inv;
                    want[(bi * c + ch) * hw + p] = v as f32 * wv[ch] + bv[ch];
                }
            }
        }
    }
    close(&n.to_vec_f32(), &want, 1e-4);
}

#[test]
fn arange_matches_oracle() {
    let n = Tensor::arange(2.0, 7.0).unwrap();
    assert_eq!(n.dims(), &[5]);
    close(&n.to_vec_f32(), &[2.0, 3.0, 4.0, 5.0, 6.0], 0.0);
    assert_eq!(Tensor::arange(3.0, 3.0).unwrap().elem_count(), 0);
    let u = Tensor::arange_u32(3, 9).unwrap();
    assert_eq!(u.to_vec_u32().unwrap(), (3u32..9).collect::<Vec<_>>());
    let i = Tensor::arange_i64(-2, 5).unwrap();
    assert_eq!(i.to_vec_i64().unwrap(), (-2i64..5).collect::<Vec<_>>());
}

#[test]
fn unary_ops_match_oracle() {
    let x = data(2 * 19, 31);
    let xp: Vec<f32> = x.iter().map(|v| v.abs() + 0.3).collect(); // recip/sqrt/powf-safe
    let nx = Tensor::from_vec_f32(x.clone(), vec![2, 19]).unwrap();
    let np = Tensor::from_vec_f32(xp.clone(), vec![2, 19]).unwrap();
    let m = |f: fn(f32) -> f32, v: &[f32]| v.iter().map(|&x| f(x)).collect::<Vec<f32>>();
    close(&nx.tanh().unwrap().to_vec_f32(), &m(f32::tanh, &x), 1e-6);
    close(&nx.abs().unwrap().to_vec_f32(), &m(f32::abs, &x), 0.0);
    close(&np.recip().unwrap().to_vec_f32(), &m(f32::recip, &xp), 1e-6);
    close(&np.sqrt().unwrap().to_vec_f32(), &m(f32::sqrt, &xp), 1e-6);
    close(
        &np.powf(1.7).unwrap().to_vec_f32(),
        &xp.iter().map(|v| v.powf(1.7)).collect::<Vec<_>>(),
        1e-5,
    );
}

#[test]
fn broadcast_div_matches_oracle() {
    let x = data(2 * 3 * 8, 33);
    let w: Vec<f32> = data(8, 34).iter().map(|v| v + 2.0).collect();
    let nx = Tensor::from_vec_f32(x.clone(), vec![2, 3, 8]).unwrap();
    let nw = Tensor::from_vec_f32(w.clone(), vec![8]).unwrap();
    let want: Vec<f32> = x.iter().enumerate().map(|(i, v)| v / w[i % 8]).collect();
    close(&nx.broadcast_div(&nw).unwrap().to_vec_f32(), &want, 0.0);
    // middle-axis: [2,1,8] divisor
    let m: Vec<f32> = data(2 * 8, 35).iter().map(|v| v + 2.0).collect();
    let nm = Tensor::from_vec_f32(m.clone(), vec![2, 1, 8]).unwrap();
    let wmid: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v / m[(i / 24) * 8 + i % 8])
        .collect();
    close(&nx.broadcast_div(&nm).unwrap().to_vec_f32(), &wmid, 0.0);
}

#[test]
fn broadcast_as_matches_oracle() {
    let m = data(2 * 8, 36);
    let n = Tensor::from_vec_f32(m.clone(), vec![2, 1, 8])
        .unwrap()
        .broadcast_as(vec![2, 5, 8])
        .unwrap();
    assert_eq!(n.dims(), &[2, 5, 8]);
    let mut want = vec![0f32; 2 * 5 * 8];
    for a in 0..2 {
        for r in 0..5 {
            for c in 0..8 {
                want[(a * 5 + r) * 8 + c] = m[a * 8 + c];
            }
        }
    }
    close(&n.to_vec_f32(), &want, 0.0);
    // rank-raising: [8] -> [3,8]
    let w = data(8, 37);
    let n = Tensor::from_vec_f32(w.clone(), vec![8])
        .unwrap()
        .broadcast_as(vec![3, 8])
        .unwrap();
    let want: Vec<f32> = (0..24).map(|i| w[i % 8]).collect();
    close(&n.to_vec_f32(), &want, 0.0);
    // incompatible target rejected
    assert!(Tensor::from_vec_f32(w, vec![8])
        .unwrap()
        .broadcast_as(vec![3, 7])
        .is_err());
}

#[test]
fn gather_matches_oracle() {
    let x = data(3 * 5, 38);
    let ids: Vec<u32> = vec![4, 0, 2, 2, 1, 3];
    let n = Tensor::from_vec_f32(x.clone(), vec![3, 5])
        .unwrap()
        .gather(&Tensor::from_vec_u32(ids.clone(), vec![3, 2]).unwrap(), 1)
        .unwrap();
    assert_eq!(n.dims(), &[3, 2]);
    let want: Vec<f32> = (0..6).map(|i| x[(i / 2) * 5 + ids[i] as usize]).collect();
    close(&n.to_vec_f32(), &want, 0.0);
    // 1-D table gather (the MoE expert-scale lookup) with i64 indices
    let t = data(7, 39);
    let ids64: Vec<i64> = vec![6, 1, 1, 0, 5];
    let n = Tensor::from_vec_f32(t.clone(), vec![7])
        .unwrap()
        .gather(&Tensor::from_vec_i64(ids64.clone(), vec![5]).unwrap(), 0)
        .unwrap();
    let want: Vec<f32> = ids64.iter().map(|&i| t[i as usize]).collect();
    close(&n.to_vec_f32(), &want, 0.0);
    // out-of-range index rejected on CPU
    let bad = Tensor::from_vec_u32(vec![9, 0], vec![2]).unwrap();
    assert!(Tensor::from_vec_f32(t, vec![7])
        .unwrap()
        .gather(&bad, 0)
        .is_err());
}

#[test]
fn where_cond_matches_oracle() {
    let cu: Vec<u32> = (0..24).map(|i| (i % 3 == 0) as u32).collect();
    let t = data(24, 40);
    let f = data(24, 41);
    let nt = Tensor::from_vec_f32(t.clone(), vec![4, 6]).unwrap();
    let nf = Tensor::from_vec_f32(f.clone(), vec![4, 6]).unwrap();
    let n = Tensor::from_vec_u32(cu.clone(), vec![4, 6])
        .unwrap()
        .where_cond(&nt, &nf)
        .unwrap();
    let want: Vec<f32> = (0..24)
        .map(|i| if cu[i] != 0 { t[i] } else { f[i] })
        .collect();
    close(&n.to_vec_f32(), &want, 0.0);
    // u8 condition route
    let cb: Vec<u8> = cu.iter().map(|&x| x as u8).collect();
    let n8 = Tensor::from_storage(CpuStorage::U8(cb), vec![4, 6])
        .unwrap()
        .where_cond(&nt, &nf)
        .unwrap();
    close(&n8.to_vec_f32(), &want, 0.0);
    // float condition rejected
    assert!(nt.where_cond(&nt, &nf).is_err());
}

/// Stable arg-sort reference (ties keep original order).
fn ref_argsort_f32(row: &[f32], asc: bool) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let o = row[a as usize].partial_cmp(&row[b as usize]).unwrap();
        if asc {
            o
        } else {
            o.reverse()
        }
    });
    idx
}

#[test]
fn sort_matches_oracle() {
    let x = data(2 * 7, 42);
    let nx = Tensor::from_vec_f32(x.clone(), vec![2, 7]).unwrap();
    for asc in [true, false] {
        let mut widx: Vec<u32> = vec![];
        let mut wval: Vec<f32> = vec![];
        for r in 0..2 {
            let row = &x[r * 7..(r + 1) * 7];
            let idx = ref_argsort_f32(row, asc);
            wval.extend(idx.iter().map(|&i| row[i as usize]));
            widx.extend(idx);
        }
        let (nv, ni) = nx.sort_last_dim(asc).unwrap();
        close(&nv.to_vec_f32(), &wval, 0.0);
        assert_eq!(ni.to_vec_u32().unwrap(), widx, "indices asc={asc}");
        let na = nx.arg_sort_last_dim(asc).unwrap();
        assert_eq!(na.to_vec_u32().unwrap(), widx, "arg_sort asc={asc}");
    }
    // u32 rows (the MoE (expert, token) pair sort)
    let v: Vec<u32> = vec![5, 1, 9, 1, 3, 0, 7];
    let (sv, si) = Tensor::from_vec_u32(v.clone(), vec![7])
        .unwrap()
        .sort_last_dim(true)
        .unwrap();
    let mut idx: Vec<u32> = (0..7).collect();
    idx.sort_by_key(|&i| v[i as usize]);
    let want_v: Vec<u32> = idx.iter().map(|&i| v[i as usize]).collect();
    assert_eq!(sv.to_vec_u32().unwrap(), want_v);
    assert_eq!(si.to_vec_u32().unwrap(), idx);
}

#[test]
fn argmax_matches_oracle() {
    let dims = [2usize, 5, 7];
    let x = data(dims.iter().product(), 43);
    let nx = Tensor::from_vec_f32(x.clone(), dims.to_vec()).unwrap();
    let st = ref_strides(&dims);
    for d in 0..3usize {
        let mut rd = dims.to_vec();
        rd[d] = 1;
        let out_n: usize = rd.iter().product();
        let rst = ref_strides(&rd);
        let mut best = vec![f32::NEG_INFINITY; out_n];
        let mut want = vec![0u32; out_n];
        for (flat, &v) in x.iter().enumerate() {
            let mut idx = ref_decompose(flat, &st);
            let di = idx[d];
            idx[d] = 0;
            let o: usize = idx.iter().zip(&rst).map(|(i, s)| i * s).sum();
            if v > best[o] {
                best[o] = v;
                want[o] = di as u32;
            }
        }
        let n = nx.argmax(d).unwrap();
        let mut want_dims = dims.to_vec();
        want_dims.remove(d);
        assert_eq!(n.dims(), &want_dims[..], "dim {d}");
        assert_eq!(n.to_vec_u32().unwrap(), want, "dim {d}");
        let nk = nx.argmax_keepdim(d).unwrap();
        assert_eq!(nk.dims(), &rd[..], "keepdim {d}");
        assert_eq!(nk.to_vec_u32().unwrap(), want, "keepdim {d}");
    }
}

/// slice_set reference on a flat buffer.
fn ref_slice_set(
    dst: &mut [f32],
    dd: &[usize],
    src: &[f32],
    sd: &[usize],
    dim: usize,
    start: usize,
) {
    let dst_st = ref_strides(dd);
    let src_st = ref_strides(sd);
    for (sflat, &v) in src.iter().enumerate() {
        let mut idx = ref_decompose(sflat, &src_st);
        idx[dim] += start;
        let dflat: usize = idx.iter().zip(&dst_st).map(|(i, s)| i * s).sum();
        dst[dflat] = v;
    }
}

#[test]
fn slice_set_matches_oracle() {
    let dst0 = data(2 * 5 * 3, 44);
    let src = data(2 * 2 * 3, 45);
    let nd = Tensor::from_vec_f32(dst0.clone(), vec![2, 5, 3]).unwrap();
    let alias = nd.reshape(vec![2, 5, 3]).unwrap(); // shares storage
    let ns = Tensor::from_vec_f32(src.clone(), vec![2, 2, 3]).unwrap();
    nd.slice_set(&ns, 1, 2).unwrap();
    let mut want = dst0.clone();
    ref_slice_set(&mut want, &[2, 5, 3], &src, &[2, 2, 3], 1, 2);
    close(&nd.to_vec_f32(), &want, 0.0);
    // in-place: the write is visible through the aliasing view (the
    // KV-cache contract)
    close(&alias.to_vec_f32(), &want, 0.0);
    // dim-0 offset write (the graph cos/sin buffer update)
    let row = data(5 * 3, 46);
    let nr = Tensor::from_vec_f32(row.clone(), vec![1, 5, 3]).unwrap();
    nd.slice_set(&nr, 0, 1).unwrap();
    ref_slice_set(&mut want, &[2, 5, 3], &row, &[1, 5, 3], 0, 1);
    close(&nd.to_vec_f32(), &want, 0.0);
    // shared storage and overflow rejected
    assert!(nd.slice_set(&nd.clone(), 1, 0).is_err());
    assert!(nd.slice_set(&ns, 1, 4).is_err());
}

#[test]
fn slice_set_int_dtypes() {
    // i64 (graph_kv_pos) and u32 storages take the same in-place path
    let nd = Tensor::from_vec_i64(vec![0; 6], vec![6]).unwrap();
    let ns = Tensor::from_vec_i64(vec![7, 8], vec![2]).unwrap();
    nd.slice_set(&ns, 0, 3).unwrap();
    assert_eq!(nd.to_vec_i64().unwrap(), vec![0, 0, 0, 7, 8, 0]);
    let ud = Tensor::from_vec_u32(vec![1; 4], vec![4]).unwrap();
    let us = Tensor::from_vec_u32(vec![9], vec![1]).unwrap();
    ud.slice_set(&us, 0, 0).unwrap();
    assert_eq!(ud.to_vec_u32().unwrap(), vec![9, 1, 1, 1]);
    // dtype mismatch rejected
    assert!(nd.slice_set(&us, 0, 0).is_err());
}

/// scatter_set reference along dim 1 of a `[2, n, 3]` buffer.
fn ref_scatter_set_dim1(dst: &mut [f32], dd: &[usize], idx: &[usize], src: &[f32], sd: &[usize]) {
    let dst_st = ref_strides(dd);
    let src_st = ref_strides(sd);
    for (sflat, &v) in src.iter().enumerate() {
        let mut di = ref_decompose(sflat, &src_st);
        di[1] = idx[sflat];
        let dflat: usize = di.iter().zip(&dst_st).map(|(i, s)| i * s).sum();
        dst[dflat] = v;
    }
}

#[test]
fn scatter_set_matches_oracle() {
    let dst0 = data(2 * 6 * 3, 47);
    let src = data(2 * 2 * 3, 48);
    let idx: Vec<i64> = vec![4, 4, 4, 1, 1, 1, 0, 0, 0, 5, 5, 5];
    let nd = Tensor::from_vec_f32(dst0.clone(), vec![2, 6, 3]).unwrap();
    let ns = Tensor::from_vec_f32(src.clone(), vec![2, 2, 3]).unwrap();
    nd.scatter_set(
        &Tensor::from_vec_i64(idx.clone(), vec![2, 2, 3]).unwrap(),
        &ns,
        1,
    )
    .unwrap();
    let mut want = dst0.clone();
    let idx_us: Vec<usize> = idx.iter().map(|&i| i as usize).collect();
    ref_scatter_set_dim1(&mut want, &[2, 6, 3], &idx_us, &src, &[2, 2, 3]);
    close(&nd.to_vec_f32(), &want, 0.0);
    // u32 index route
    let idxu: Vec<u32> = vec![0, 0, 0, 3, 3, 3, 2, 2, 2, 1, 1, 1];
    nd.scatter_set(
        &Tensor::from_vec_u32(idxu.clone(), vec![2, 2, 3]).unwrap(),
        &ns,
        1,
    )
    .unwrap();
    let idxu_us: Vec<usize> = idxu.iter().map(|&i| i as usize).collect();
    ref_scatter_set_dim1(&mut want, &[2, 6, 3], &idxu_us, &src, &[2, 2, 3]);
    close(&nd.to_vec_f32(), &want, 0.0);
    // out-of-range index rejected on CPU
    let bad = Tensor::from_vec_u32(vec![9; 12], vec![2, 2, 3]).unwrap();
    assert!(nd.scatter_set(&bad, &ns, 1).is_err());
}

#[test]
fn accessors_match_oracle() {
    let x = data(6, 49);
    let n = Tensor::from_vec_f32(x.clone(), vec![2, 3]).unwrap();
    assert_eq!(
        n.to_vec2_f32().unwrap(),
        vec![x[..3].to_vec(), x[3..].to_vec()]
    );
    assert_eq!(n.flatten_all().unwrap().dims(), &[6]);
    assert_eq!(n.flatten_all().unwrap().to_vec1_f32().unwrap(), x);
    assert!(n.to_vec1_f32().is_err()); // rank 2
    assert_eq!(
        Tensor::from_vec_f32(vec![7.5f32], vec![1])
            .unwrap()
            .to_scalar_f32()
            .unwrap(),
        7.5
    );
    let u = Tensor::from_vec_u32(vec![3, 1, 4], vec![3]).unwrap();
    assert_eq!(u.to_vec1_u32().unwrap(), vec![3, 1, 4]);
    assert!(u.to_vec1_i64().is_err()); // strict dtype
    let i = Tensor::from_vec_i64(vec![-2, 9], vec![2]).unwrap();
    assert_eq!(i.to_vec1_i64().unwrap(), vec![-2, 9]);
    assert_eq!(
        Tensor::from_vec_u32(vec![42], vec![1])
            .unwrap()
            .to_scalar_u32()
            .unwrap(),
        42
    );
}

#[test]
fn int_dtype_coverage() {
    // narrow / cat on u32 and i64 storages (token-id housekeeping)
    let v: Vec<u32> = (0..12).collect();
    let nn = Tensor::from_vec_u32(v.clone(), vec![3, 4])
        .unwrap()
        .narrow(1, 1, 2)
        .unwrap();
    // rows of the [3,4] matrix, columns 1..3
    let want_n: Vec<u32> = (0..3)
        .flat_map(|r| vec![v[r * 4 + 1], v[r * 4 + 2]])
        .collect();
    assert_eq!(nn.to_vec_u32().unwrap(), want_n);
    let c = Tensor::cat(&[&nn, &nn], 0).unwrap();
    assert_eq!(c.dims(), &[6, 2]);
    let want_c: Vec<u32> = want_n.iter().chain(&want_n).copied().collect();
    assert_eq!(c.to_vec_u32().unwrap(), want_c);
    let vi: Vec<i64> = (0..10).map(|x| x * 3 - 5).collect();
    let n64 = Tensor::from_vec_i64(vi.clone(), vec![2, 5])
        .unwrap()
        .narrow(1, 2, 3)
        .unwrap();
    let want_64: Vec<i64> = (0..2)
        .flat_map(|r| vi[r * 5 + 2..r * 5 + 5].to_vec())
        .collect();
    assert_eq!(n64.to_vec_i64().unwrap(), want_64);
    // to_dtype integer casts (`as` semantics, i64 range preserved)
    let t = Tensor::from_vec_f32(vec![0.0f32, 1.9, 3.0, 250.7], vec![4]).unwrap();
    assert_eq!(
        t.to_dtype(DType::U32).unwrap().to_vec_u32().unwrap(),
        vec![0, 1, 3, 250]
    );
    assert_eq!(
        t.to_dtype(DType::I64).unwrap().to_vec_i64().unwrap(),
        vec![0, 1, 3, 250]
    );
    let u = Tensor::from_vec_u32(vec![7, 8], vec![2]).unwrap();
    assert_eq!(
        u.to_dtype(DType::I64).unwrap().to_vec_i64().unwrap(),
        vec![7, 8]
    );
    assert_eq!(u.to_dtype(DType::F32).unwrap().to_vec_f32(), vec![7.0, 8.0]);
    let big = Tensor::from_vec_u32(vec![1 << 26], vec![1]).unwrap();
    assert_eq!(
        big.to_dtype(DType::I64).unwrap().to_vec_i64().unwrap(),
        vec![1i64 << 26]
    );
    // index_select: i64 ids + f16 table (embedding shapes)
    let table = data(5 * 4, 50);
    let nt = Tensor::from_vec_f32(table.clone(), vec![5, 4])
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let got = nt
        .index_select(&Tensor::from_vec_i64(vec![4, 0, 2], vec![3]).unwrap(), 0)
        .unwrap();
    assert_eq!(got.dtype(), DType::F16);
    let want: Vec<f32> = [4usize, 0, 2]
        .iter()
        .flat_map(|&r| {
            table[r * 4..(r + 1) * 4]
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        })
        .collect();
    close(&got.to_vec_f32(), &want, 0.0);
}

#[test]
fn add_mul_reshape() {
    let x = data(24, 4);
    let y = data(24, 5);
    let nx = Tensor::from_vec_f32(x.clone(), vec![2, 3, 4]).unwrap();
    let ny = Tensor::from_vec_f32(y.clone(), vec![2, 3, 4]).unwrap();
    let sum = nx.add(&ny).unwrap().to_vec_f32();
    let prod = nx.mul(&ny).unwrap().to_vec_f32();
    for i in 0..24 {
        assert_eq!(sum[i], x[i] + y[i]);
        assert_eq!(prod[i], x[i] * y[i]);
    }
    let r = nx.reshape(vec![6, 4]).unwrap();
    assert_eq!(r.dims(), &[6, 4]);
    assert!(nx.reshape(vec![5, 5]).is_err());
}
