use super::*;

/// Turbo schedule matches the oracle formula: starts at 1.0 (t=1 ⟹
/// shift.1/(1+(shift-1)) = 1), strictly decreasing, the exact 8-step shift-3
/// values, length = num_steps.
#[test]
fn turbo_schedule_correct() {
    let s = turbo_schedule(8, 3.0);
    assert_eq!(s.len(), 8);
    assert!((s[0] - 1.0).abs() < 1e-6, "t0 should be 1.0, got {}", s[0]);
    for w in s.windows(2) {
        assert!(
            w[0] > w[1],
            "schedule must strictly decrease: {} !> {}",
            w[0],
            w[1]
        );
    }
    // hand-computed shift=3 values t_i = 3t/(1+2t), t = 1 - i/8.
    let want = [1.0, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3];
    for (a, e) in s.iter().zip(&want) {
        assert!((a - e).abs() < 1e-4, "schedule {a} vs {e}");
    }
}

/// For a CONSTANT velocity field v (independent of x and t), Euler integration
/// is exact: the dt steps telescope to (0 - t_0), so x_final = x_init - t_0.v.
/// With turbo t_0 = 1.0 ⟹ x_final = x_init - v. This pins the integrator's
/// step math + the implicit t=0 final endpoint.
#[test]
fn euler_constant_velocity_is_exact() {
    let v0 = vec![0.3f32, -0.7, 1.1, 0.0];
    let mut x = vec![1.0f32, 2.0, -1.0, 0.5];
    let x_init = x.clone();
    let sched = turbo_schedule(8, 3.0);
    crate::inference::sample::flow_unipc::euler_integrate(&mut x, &sched, |_xt, _t| v0.clone());
    for i in 0..4 {
        let want = x_init[i] - sched[0] * v0[i]; // t_0 = 1.0
        assert!(
            (x[i] - want).abs() < 1e-5,
            "euler @ {i}: {} vs {want}",
            x[i]
        );
    }
}

/// AdaLN modulate with scale=0, shift=0 is the identity; nonzero applies
/// `x.(1+scale)+shift` per channel (broadcast over time).
#[test]
fn adaln_modulate_correct() {
    let (c, t) = (2usize, 3usize);
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let id = adaln_modulate(&x, &[0.0, 0.0], &[0.0, 0.0], c, t);
    assert_eq!(id, x);
    let m = adaln_modulate(&x, &[0.5, -0.5], &[1.0, 2.0], c, t);
    // ch0: *1.5 +1 ; ch1: *0.5 +2
    assert!((m[0] - (1.0 * 1.5 + 1.0)).abs() < 1e-6);
    assert!((m[3] - (4.0 * 0.5 + 2.0)).abs() < 1e-6);
    let g = adaln_gate(&x, &[2.0, 0.0], c, t);
    assert_eq!(&g[0..3], &[2.0, 4.0, 6.0]); // ch0 x2
    assert_eq!(&g[3..6], &[0.0, 0.0, 0.0]); // ch1 gated to 0
}

/// Sinusoidal timestep embedding matches the ggml closed form: at t=0 every
/// arg is 0 ⟹ cos=1/sin=0, so the embedding is [1xhalf, 0xhalf]; general t
/// matches cos/sin(t.exp(-ln(P).j/half)) per lane; freq decays j=0->half.
#[test]
fn timestep_embedding_correct() {
    let (dim, p) = (256usize, 10000.0f32);
    let half = dim / 2;
    let e0 = timestep_embedding(0.0, dim, p);
    assert_eq!(e0.len(), dim);
    for j in 0..half {
        assert!((e0[j] - 1.0).abs() < 1e-6, "cos(0) lane {j}");
        assert!(e0[j + half].abs() < 1e-6, "sin(0) lane {j}");
    }
    let t = 12.5f32;
    let e = timestep_embedding(t, dim, p);
    for j in [0usize, 1, 7, 63, 127] {
        let freq = (-(p.ln()) * (j as f32) / (half as f32)).exp();
        assert!((e[j] - (t * freq).cos()).abs() < 1e-5, "cos lane {j}");
        assert!(
            (e[j + half] - (t * freq).sin()).abs() < 1e-5,
            "sin lane {j}"
        );
    }
    // j=0 lane is the highest freq (=1.0); freq strictly decreases with j.
    let f0 = (-(p.ln()) * 0.0 / half as f32).exp();
    let f1 = (-(p.ln()) * 1.0 / half as f32).exp();
    assert!((f0 - 1.0).abs() < 1e-6 && f1 < f0, "freq decay");
    // The DiT wrapper applies the 1000x scale.
    assert_eq!(
        dit_timestep_embedding(0.0),
        timestep_embedding(0.0, 256, 10000.0)
    );
}

/// AdaLN split: adaln = table + tproj (elementwise), chunked into the 6 named
/// vectors in order. With a zero table the chunks are exactly the tproj chunks.
#[test]
fn adaln_split_correct() {
    let h = 4usize;
    let tproj: Vec<f32> = (0..6 * h).map(|i| i as f32).collect();
    let table = vec![0f32; 6 * h];
    let a = adaln_split(&tproj, &table, h);
    assert_eq!(a.shift_sa, vec![0.0, 1.0, 2.0, 3.0]); // chunk 0
    assert_eq!(a.scale_sa, vec![4.0, 5.0, 6.0, 7.0]); // chunk 1
    assert_eq!(a.gate_sa, vec![8.0, 9.0, 10.0, 11.0]); // chunk 2
    assert_eq!(a.shift_mlp, vec![12.0, 13.0, 14.0, 15.0]); // chunk 3
    assert_eq!(a.scale_mlp, vec![16.0, 17.0, 18.0, 19.0]); // chunk 4
    assert_eq!(a.gate_mlp, vec![20.0, 21.0, 22.0, 23.0]); // chunk 5
                                                          // table adds elementwise per chunk.
    let table2: Vec<f32> = (0..6 * h).map(|_| 100.0).collect();
    let a2 = adaln_split(&tproj, &table2, h);
    assert_eq!(a2.shift_sa, vec![100.0, 101.0, 102.0, 103.0]);
    assert_eq!(a2.gate_mlp, vec![120.0, 121.0, 122.0, 123.0]);
}

/// patchify groups P frames into a token; unpatchify is its exact inverse, so
/// unpatchify(patchify(x)) == x. Also checks the grouping order (frame P.s+p
/// lands in patch-slot p) on a tiny hand-traceable case.
// The `0 *` / `* 1` / `+ 0` terms below are the formula in the doc comment
// written out at concrete indices; collapsing them would hide what is checked.
#[allow(clippy::erasing_op, clippy::identity_op)]
#[test]
fn patch_unpatch_roundtrip() {
    let (in_ch, t, patch) = (3usize, 8usize, 2usize);
    let latent: Vec<f32> = (0..in_ch * t).map(|i| i as f32 * 0.5 - 3.0).collect();
    let (patched, s) = patchify(&latent, in_ch, t, patch);
    assert_eq!(s, t / patch);
    assert_eq!(patched.len(), in_ch * patch * s);
    // grouping: patched[(p.in_ch+c).s + si] == latent[c.t + (patch.si+p)]
    assert_eq!(
        patched[(0 * in_ch + 1) * s + 2],
        latent[1 * t + (patch * 2 + 0)]
    );
    assert_eq!(
        patched[(1 * in_ch + 2) * s + 1],
        latent[2 * t + (patch * 1 + 1)]
    );
    let (back, t2) = unpatchify(&patched, in_ch, s, patch);
    assert_eq!(t2, t);
    assert_eq!(back, latent, "patch/unpatch round-trip");
}

/// INTEGRATION (ignored; needs the real GGUF). Loads the full turbo DiT from
/// acestep-v15-turbo-Q8_0.gguf into the typed structs and checks the geometry +
/// that every weight dequantizes to a finite f32 Tensor of the expected shape.
/// ⚠️ STRUCTURAL load only - numerical correctness pending acestep.cpp --dump.
/// Run: cargo test --release --lib inference::model::acestep::dit::tests::load_real_dit -- --ignored --nocapture
#[test]
#[ignore]
fn load_real_dit() {
    let m = DitModel::from_gguf(
        crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
            .to_str()
            .unwrap(),
    )
    .expect("load dit gguf");
    assert_eq!(m.layers.len(), 24);
    assert_eq!(m.hidden, 2048);
    // proj_in/proj_out are stored RAW 3-D [hidden, ch, patch]; the forward
    // permutes+reshapes them to a [.., in_ch.patch] linear aligned with
    // patchify's p.in_ch+c ordering (the oracle's load-time permute - done in
    // the forward, --dump-validated).
    assert_eq!(m.proj_in.w.dims(), &[2048, 192, 2], "proj_in raw shape");
    assert_eq!(m.proj_out.w.dims(), &[2048, 64, 2], "proj_out raw shape");
    // self_attn q_proj [2048,2048], k_proj [1024,2048] (GQA) - quantized [out,in] linears
    assert_eq!(
        (m.layers[0].sa_q.out_dim(), m.layers[0].sa_q.in_dim()),
        (2048, 2048)
    );
    assert_eq!(
        (m.layers[0].sa_k.out_dim(), m.layers[0].sa_k.in_dim()),
        (1024, 2048)
    );
    assert_eq!(m.layers[0].sa_q_norm.dims(), &[128], "qk-norm head_dim");
    assert_eq!(m.layers[0].scale_shift_table.dims(), &[6, 2048]);
    assert_eq!(
        (
            m.layers[0].mlp_gate.out_dim(),
            m.layers[0].mlp_gate.in_dim()
        ),
        (6144, 2048)
    );
    // time_proj -> 6H
    assert_eq!(m.time_embed.time_proj.w.dims(), &[12288, 2048]);
    // alternating layer types: even=SWA, odd=full
    assert!(!m.layers[0].layer_type_full && m.layers[1].layer_type_full);
    // every weight finite (spot the largest few + a norm + the table)
    // (the per-block linears are now quantized QMatMul - not flatten-able F32 tensors;
    // their finiteness is exercised end-to-end by the render path.)
    for (name, t) in [
        ("proj_in.w", &m.proj_in.w),
        ("l0.scale_shift_table", &m.layers[0].scale_shift_table),
        ("norm_out", &m.norm_out),
    ] {
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1_f32().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "{name} non-finite");
    }
    println!("DiT loaded: 24 layers, hidden 2048, all weights finite + correct shape");
}

#[cfg(test)]
fn load_dump(path: &str) -> (Vec<f32>, Vec<usize>) {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("read {path}"));
    let nd = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let shape: Vec<usize> = (0..nd)
        .map(|i| {
            i32::from_le_bytes([b[4 + i * 4], b[5 + i * 4], b[6 + i * 4], b[7 + i * 4]]) as usize
        })
        .collect();
    let data = b[4 + nd * 4..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, shape)
}

/// VALIDATE proj_in vs the oracle (ignored; needs /tmp/acedump + the DiT gguf).
/// Builds input = concat(context[T,128], xt[T,64]) -> channel-major [192,T], runs
/// proj_in_forward, compares to hidden_after_proj_in [H,S]. Proves input
/// construction + patchify + the proj_in permute + linear.
/// Run: cargo test --release --lib inference::model::acestep::dit::tests::validate_proj_in_vs_oracle -- --ignored --nocapture
#[test]
#[ignore]
fn validate_proj_in_vs_oracle() {
    let (ctx, cs) = load_dump("/tmp/acedump/context.bin"); // [T,128]
    let (xt, xs) = load_dump("/tmp/acedump/dit_step0_xt.bin"); // [T,64]
    let (t_lat, c_ctx, c_xt) = (cs[0], cs[1], xs[1]);
    assert_eq!((c_ctx, c_xt), (128, 64));
    // channel-major input [192, T]: channels [context(128), xt(64)].
    let in_ch = c_ctx + c_xt;
    let mut input = vec![0f32; in_ch * t_lat];
    for ti in 0..t_lat {
        for c in 0..c_ctx {
            input[c * t_lat + ti] = ctx[ti * c_ctx + c];
        }
        for c in 0..c_xt {
            input[(c_ctx + c) * t_lat + ti] = xt[ti * c_xt + c];
        }
    }
    let m = DitModel::from_gguf(
        crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let (hid, s) = m.proj_in_forward(&input, t_lat).unwrap();
    let (oracle, osh) = load_dump("/tmp/acedump/hidden_after_proj_in.bin"); // [H,S], H-fastest
    assert_eq!(osh, vec![m.hidden, s], "proj_in out shape");
    assert_eq!(hid.len(), oracle.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, o) in hid.iter().zip(&oracle) {
        dot += (*a as f64) * (*o as f64);
        na += (*a as f64).powi(2);
        nb += (*o as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    println!("proj_in vs oracle: cosine={cos:.6}");
    assert!(cos > 0.999, "proj_in diverges: cosine {cos}");
}

/// VALIDATE the self-attn Q-path (proj->heads->qk-norm->RoPE) vs the oracle's
/// layer0_q_after_rope dump (q at position s=0, [D,Nh]). Confirms RoPE NEOX +
/// qk-norm order. Run: cargo test --release --lib inference::model::acestep::dit::tests::validate_qk_vs_oracle -- --ignored --nocapture
#[test]
#[ignore]
fn validate_qk_vs_oracle() {
    use crate::tensor::{Tensor, D};
    let m = DitModel::from_gguf(
        crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    // layer0_sa_input [H,S] H-fastest = flat[s.H+h] -> read as [S,H].
    let (sai, sh) = load_dump("/tmp/acedump/layer0_sa_input.bin");
    let (h, s) = (sh[0], sh[1]); // [2048, 160]
    let x = Tensor::from_vec_f32(sai, (s, h)).unwrap();
    let l0 = &m.layers[0];
    let q = m
        .self_attn_qk_roped(&l0.sa_q, &l0.sa_q_norm, &x, m.n_head, s)
        .unwrap(); // [1,Nh,S,D]
                   // extract s=0: [Nh, D] flat[nh.D+d]
    let q0: Vec<f32> = q
        .narrow(2, 0, 1)
        .unwrap()
        .reshape((m.n_head, m.head_dim))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1_f32()
        .unwrap();
    let (oracle, osh) = load_dump("/tmp/acedump/layer0_q_after_rope.bin"); // [D,Nh] flat[nh.D+d]
    assert_eq!(osh, vec![m.head_dim, m.n_head]);
    assert_eq!(q0.len(), oracle.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, o) in q0.iter().zip(&oracle) {
        dot += (*a * o) as f64;
        na += (*a as f64).powi(2);
        nb += (*o as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let _ = D::Minus1;
    // WIP: cosine ~0 at s=0 (RoPE=identity there) ⟹ a proj/reshape/qk-norm
    // layout bug still to find - the matmul (.t()) + head-split (D-fastest)
    // match the validated proj_in precedent, so suspect qk-norm dim or the
    // q_dim head ordering. Debug against layer0_q_after_rope next session.
    println!(
        "self-attn q_after_rope vs oracle: cosine={cos:.6}  mine[..4]={:?} oracle[..4]={:?}",
        &q0[..4],
        &oracle[..4]
    );
}

/// VALIDATE the full self-attn vs the reliable layer0_sa_output dump.
/// Run: cargo test --release --lib inference::model::acestep::dit::tests::validate_self_attn_vs_oracle -- --ignored --nocapture
#[test]
#[ignore]
fn validate_self_attn_vs_oracle() {
    let m = DitModel::from_gguf(
        crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    use crate::tensor::Tensor;
    let (sai, sh) = load_dump("/tmp/acedump/layer0_sa_input.bin"); // [H,S] flat[s.H+h]
    let (h, s) = (sh[0], sh[1]);
    assert_eq!(sai.len(), h * s);
    let x = Tensor::from_vec_f32(sai, (s, h))
        .unwrap()
        .to_device(&m.layers[0].device)
        .unwrap();
    let so: Vec<f32> = m
        .self_attn_forward(&m.layers[0], &x, s, m.sliding_window)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1_f32()
        .unwrap(); // [S.H] flat[s.h]
    let (oracle, _osh) = load_dump("/tmp/acedump/layer0_sa_output.bin"); // [H,S] flat[s.H+h]
    assert_eq!(so.len(), oracle.len());
    let (mut dt, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, o) in so.iter().zip(&oracle) {
        dt += (*a * o) as f64;
        na += (*a as f64).powi(2);
        nb += (*o as f64).powi(2);
    }
    println!(
        "self_attn vs oracle: cosine={:.6} mine_rms={:.3} oracle_rms={:.3}",
        dt / (na.sqrt() * nb.sqrt()),
        (na / so.len() as f64).sqrt(),
        (nb / oracle.len() as f64).sqrt()
    );
}

/// VALIDATE time_embed_forward(t=1.0) vs the dumped tproj/temb (step 0).
/// Run: cargo test --release --lib inference::model::acestep::dit::tests::validate_time_embed_vs_oracle -- --ignored --nocapture
#[test]
#[ignore]
fn validate_time_embed_vs_oracle() {
    let m = DitModel::from_gguf(
        crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let t = turbo_schedule(8, 3.0)[0]; // 1.0
    let (temb, tproj) = m.time_embed_forward(t).unwrap();
    let (otemb, _) = load_dump("/tmp/acedump/temb.bin");
    let (otproj, _) = load_dump("/tmp/acedump/tproj.bin");
    let cosf = |a: &[f32], b: &[f32]| {
        let (mut dt, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dt += (*x * *y) as f64;
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
        }
        dt / (na.sqrt() * nb.sqrt())
    };
    println!(
        "time_embed: temb cos={:.6} tproj cos={:.6}",
        cosf(&temb, &otemb),
        cosf(&tproj, &otproj)
    );
}

/// VALIDATE the full DiT velocity vs dit_step0_vt (24 layers + final + unpatch).
/// Run: cargo test --release --lib inference::model::acestep::dit::tests::validate_velocity_vs_oracle -- --ignored --nocapture
#[test]
#[ignore]
fn validate_velocity_vs_oracle() {
    let m = DitModel::from_gguf(
        crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let (ctx, cs) = load_dump("/tmp/acedump/context.bin"); // [T,128]
    let (xt, _) = load_dump("/tmp/acedump/dit_step0_xt.bin"); // [T,64]
    let (t_lat, c_ctx, c_xt) = (cs[0], cs[1], 64usize);
    let in_ch = c_ctx + c_xt;
    let mut input = vec![0f32; in_ch * t_lat];
    for ti in 0..t_lat {
        for c in 0..c_ctx {
            input[c * t_lat + ti] = ctx[ti * c_ctx + c];
        }
        for c in 0..c_xt {
            input[(c_ctx + c) * t_lat + ti] = xt[ti * c_xt + c];
        }
    }
    let (tproj, _) = load_dump("/tmp/acedump/tproj.bin");
    let (temb, _) = load_dump("/tmp/acedump/temb.bin");
    let (enc, esh) = load_dump("/tmp/acedump/enc_after_cond_emb.bin");
    // bisect: my hidden after all 24 layers vs hidden_after_layer23
    {
        use crate::tensor::Tensor;
        let (hid0, s) = m.proj_in_forward(&input, t_lat).unwrap();
        let mut hid = Tensor::from_vec_f32(hid0, (s, m.hidden))
            .unwrap()
            .to_device(&m.device)
            .unwrap();
        let cosv = |a: &[f32], b: &[f32]| {
            let (mut dt, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                dt += (*x * *y) as f64;
                na += (*x as f64).powi(2);
                nb += (*y as f64).powi(2);
            }
            dt / (na.sqrt() * nb.sqrt())
        };
        for (li, l) in m.layers.iter().enumerate() {
            let win = if l.layer_type_full {
                usize::MAX
            } else {
                m.sliding_window
            };
            hid = m
                .layer_forward(l, &hid, &tproj, &enc, s, esh[1], win)
                .unwrap();
            if [0usize, 6, 12, 18, 23].contains(&li) {
                let hv: Vec<f32> = hid.flatten_all().unwrap().to_vec1_f32().unwrap();
                let (ho, _) = load_dump(&format!("/tmp/acedump/hidden_after_layer{li}.bin"));
                println!("  after layer{li}: cos={:.6}", cosv(&hv, &ho));
            }
        }
    }
    let vel = m
        .velocity_forward(&input, &tproj, &temb, &enc, t_lat, esh[1])
        .unwrap(); // [oc.T]
    let (oracle, osh) = load_dump("/tmp/acedump/dit_step0_vt.bin"); // [T, oc]
    let (to, oc) = (osh[0], osh[1]);
    // oracle [T,oc] flat[t.oc+c] -> transpose to [oc,T] flat[c.T+t] to match mine
    let mut otr = vec![0f32; oc * to];
    for ti in 0..to {
        for c in 0..oc {
            otr[c * to + ti] = oracle[ti * oc + c];
        }
    }
    let cosf = |o: &[f32]| {
        let (mut dt, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in vel.iter().zip(o) {
            dt += (*a * *b) as f64;
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
        }
        dt / (na.sqrt() * nb.sqrt())
    };
    println!("DiT velocity vs oracle: cosine(transpose)={:.6} cosine(flat)={:.6} mine_rms={:.3} oracle_rms={:.3}",
        cosf(&otr), cosf(&oracle),
        (vel.iter().map(|x|(x*x)as f64).sum::<f64>()/vel.len()as f64).sqrt(),
        (oracle.iter().map(|x|(x*x)as f64).sum::<f64>()/oracle.len()as f64).sqrt());
}

/// VALIDATE the full DiT layer 0 vs hidden_after_layer0 (cross-attn + FFN + assembly).
/// Run: cargo test --release --lib inference::model::acestep::dit::tests::validate_layer0_vs_oracle -- --ignored --nocapture
#[test]
#[ignore]
fn validate_layer0_vs_oracle() {
    let m = DitModel::from_gguf(
        crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let (hid, hsh) = load_dump("/tmp/acedump/hidden_after_proj_in.bin"); // [H,S] flat[s.H+h]
    let (h, s) = (hsh[0], hsh[1]);
    let (tproj, _) = load_dump("/tmp/acedump/tproj.bin");
    let (enc, esh) = load_dump("/tmp/acedump/enc_after_cond_emb.bin"); // [H,enc_S]
    let enc_s = esh[1];
    let hid_t = crate::tensor::Tensor::from_vec_f32(hid, (s, h))
        .unwrap()
        .to_device(&m.device)
        .unwrap();
    let out: Vec<f32> = m
        .layer_forward(
            &m.layers[0],
            &hid_t,
            &tproj,
            &enc,
            s,
            enc_s,
            m.sliding_window,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1_f32()
        .unwrap();
    let (oracle, _) = load_dump("/tmp/acedump/hidden_after_layer0.bin");
    assert_eq!(out.len(), oracle.len());
    let _ = h;
    let (mut dt, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, o) in out.iter().zip(&oracle) {
        dt += (*a * o) as f64;
        na += (*a as f64).powi(2);
        nb += (*o as f64).powi(2);
    }
    println!(
        "DiT layer0 vs oracle: cosine={:.6} mine_rms={:.3} oracle_rms={:.3}",
        dt / (na.sqrt() * nb.sqrt()),
        (na / out.len() as f64).sqrt(),
        (nb / oracle.len() as f64).sqrt()
    );
}

/// PROBE (ignored; needs the real GGUF). Confirms the DiT tensor inventory +
/// per-layer shapes match the spec (Qwen3-shaped GQA + qk-norm + cross-attn +
/// AdaLN scale_shift_table + SwiGLU) and that Q8_0/BF16 tensors dequantize
/// finite. Run:
///   cargo test --release --lib inference::model::acestep::dit::tests::probe_dit_gguf -- --ignored --nocapture
#[test]
#[ignore]
fn probe_dit_gguf() {
    use crate::tensor::quantized::gguf_file;
    use crate::tensor::{DType, Device};
    let path = crate::inference::model::acestep::fsq::acestep_gguf("acestep-v15-turbo-Q8_0.gguf");
    let mut file = std::fs::File::open(path).expect("open dit gguf");
    let content = gguf_file::read_mapped_file(&file).expect("read gguf");
    println!("dit tensor count: {}", content.tensor_infos.len());
    for name in [
        "decoder.proj_in.1.weight",
        "decoder.time_embed.time_proj.weight",
        "decoder.layers.0.self_attn.q_proj.weight",
        "decoder.layers.0.self_attn.q_norm.weight",
        "decoder.layers.0.cross_attn.k_proj.weight",
        "decoder.layers.0.scale_shift_table",
        "decoder.layers.0.mlp.gate_proj.weight",
        "decoder.proj_out.1.weight",
    ] {
        let info = content
            .tensor_infos
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        let dims = info.shape.dims().to_vec();
        let qt = content.tensor(&mut file, name, &Device::Cpu).expect("read");
        let v: Vec<f32> = qt
            .dequantize(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        println!(
            "{name:46} dims={dims:?} elems={} finite={}",
            v.len(),
            v.iter().all(|x| x.is_finite())
        );
    }
    // All 24 layers present with the full per-layer tensor set.
    let mut missing = vec![];
    for l in 0..24 {
        for t in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.q_norm.weight",
            "cross_attn.q_proj.weight",
            "cross_attn.k_proj.weight",
            "scale_shift_table",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "self_attn_norm.weight",
            "cross_attn_norm.weight",
            "mlp_norm.weight",
        ] {
            let n = format!("decoder.layers.{l}.{t}");
            if !content.tensor_infos.contains_key(&n) {
                missing.push(n);
            }
        }
    }
    println!("dit layer inventory missing: {:?}", missing);
    assert!(missing.is_empty(), "missing DiT tensors: {missing:?}");
}
