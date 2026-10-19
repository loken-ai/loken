//! The host decode step over several rows at once, for a draft verify.
//!
//! A verified row stands for the decode step it replaces, so it must carry exactly that
//! step's arithmetic: the same activation quantisation, the same dot kernels, the same cache
//! append and attention at its own position. Each projection here streams its weight once for
//! all the rows, which is what makes the verify cheaper than the steps.

#[allow(unused_imports)]
use super::*;

impl GenericTransformerLayer {
    /// [`Self::decode_layer_cpu`] over the rows of `xs` (`[rows, hidden]`) at the positions
    /// from `index_pos`: each row comes out bit-identical to the decode step at its position,
    /// and the caches gain the rows as those steps append them. `Ok(false)`, with nothing
    /// changed, for a layer outside the plain dense subset this covers.
    pub(super) fn decode_rows_cpu(
        &mut self,
        xs: &mut [f32],
        hidden: usize,
        index_pos: usize,
    ) -> Result<bool> {
        let (nh, hd) = (self.n_head, self.head_dim);
        if hidden == 0 || xs.is_empty() || !xs.len().is_multiple_of(hidden) {
            return Ok(false);
        }
        let raw = |w: Option<&super::projection::QMatMul>| w.is_some_and(|w| w.cpu_raw().is_some());
        let projections_ok = match self.attn_qkv.as_ref() {
            Some(qkv) => qkv.cpu_raw().is_some(),
            None => {
                raw(self.attn_q.as_ref()) && raw(self.attn_k.as_ref()) && raw(self.attn_v.as_ref())
            }
        } && self.attn_output.cpu_raw().is_some()
            && (self.ffn_gate.is_none() || raw(self.ffn_gate.as_ref()))
            && self.ffn_up.cpu_raw().is_some()
            && self.ffn_down.cpu_raw().is_some();
        if !projections_ok
            || !self.host_decode_arch_ok(false)
            || self.attn_v_norm_ones.is_some()
            || self.attn_sinks.is_some()
            || self.moe.is_some()
            || self.ple_inp_gate.is_some()
            || self.populate_dual_kv
            || self.post_attn_norm.is_some()
            || self.post_ffn_norm.is_some()
            || self.flags.intermediate_size == 0
            || self.ffn_norm.is_none()
            || hd == 0
        {
            return Ok(false);
        }
        self.ensure_cpu_norm_cache()?;
        if self.cpu_q8_kv.is_none() {
            self.cpu_q8_kv = Some(crate::inference::cache::cpu_q8_kv::CpuQ8Kv::new(
                nh,
                self.n_kv_head,
                hd,
            ));
        }
        let nc = self.cpu_norm_cache.take().unwrap();
        let done = self.decode_rows_with(&nc, xs, hidden, index_pos);
        self.cpu_norm_cache = Some(nc);
        done.map(|()| true)
    }

    /// The body of [`Self::decode_rows_cpu`] once the layer is known to be covered, in the
    /// order of [`Self::decode_layer_cpu`]'s steps.
    fn decode_rows_with(
        &mut self,
        nc: &crate::inference::kernel::cpu_decode_exec::DecodeNormCache,
        xs: &mut [f32],
        hidden: usize,
        index_pos: usize,
    ) -> Result<()> {
        use crate::inference::kernel::cpu_decode_exec as ex;
        let (nh, nkv, hd) = (self.n_head, self.n_kv_head, self.head_dim);
        let rows = xs.len() / hidden;
        let (qd, kd) = (nh * hd, nkv * hd);
        let need = |served: bool| -> Result<()> {
            if served {
                Ok(())
            } else {
                crate::tensor::bail!("a host projection declined the rows it was checked for")
            }
        };

        // -- Attention block ----------------------------------------------
        let mut norm = vec![0f32; rows * hidden];
        for (x, n) in xs.chunks(hidden).zip(norm.chunks_mut(hidden)) {
            ex::rms_norm_slice(x, &nc.attn_norm_w, nc.attn_eps, n);
        }
        let mut q = vec![0f32; rows * qd];
        let mut k = vec![0f32; rows * kd];
        let mut v = vec![0f32; rows * kd];
        if let Some(qkv) = self.attn_qkv.as_ref() {
            let w = qd + 2 * kd;
            let mut full = vec![0f32; rows * w];
            need(qkv.forward_rows_cpu(&norm, &mut full)?)?;
            for r in 0..rows {
                let f = &full[r * w..(r + 1) * w];
                q[r * qd..(r + 1) * qd].copy_from_slice(&f[..qd]);
                k[r * kd..(r + 1) * kd].copy_from_slice(&f[qd..qd + kd]);
                v[r * kd..(r + 1) * kd].copy_from_slice(&f[qd + kd..]);
            }
        } else {
            need(
                self.attn_q
                    .as_ref()
                    .unwrap()
                    .forward_rows_cpu(&norm, &mut q)?,
            )?;
            need(
                self.attn_k
                    .as_ref()
                    .unwrap()
                    .forward_rows_cpu(&norm, &mut k)?,
            )?;
            need(
                self.attn_v
                    .as_ref()
                    .unwrap()
                    .forward_rows_cpu(&norm, &mut v)?,
            )?;
        }
        let scale = self
            .attention_scale
            .unwrap_or_else(|| 1.0 / (hd as f64).sqrt()) as f32;
        let mut attn = vec![0f32; rows * qd];
        let mut tmp = vec![0f32; hd];
        for r in 0..rows {
            let qr = &mut q[r * qd..(r + 1) * qd];
            let kr = &mut k[r * kd..(r + 1) * kd];
            let vr = &mut v[r * kd..(r + 1) * kd];
            if self.flags.has_qkv_bias {
                if let Some(b) = &nc.q_bias {
                    ex::residual_add(qr, b);
                }
                if let Some(b) = &nc.k_bias {
                    ex::residual_add(kr, b);
                }
                if let Some(b) = &nc.v_bias {
                    ex::residual_add(vr, b);
                }
            }
            if let (Some((qw, qe)), Some((kw, ke))) = (&nc.q_norm_w, &nc.k_norm_w) {
                for h in qr.chunks_mut(hd) {
                    ex::rms_norm_slice(h, qw, *qe, &mut tmp);
                    h.copy_from_slice(&tmp);
                }
                for h in kr.chunks_mut(hd) {
                    ex::rms_norm_slice(h, kw, *ke, &mut tmp);
                    h.copy_from_slice(&tmp);
                }
            }
            let pos = index_pos + r;
            let cos = self
                .cos
                .narrow(0, pos, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let sin = self
                .sin
                .narrow(0, pos, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            ex::rope_neox_slice(qr, &cos, &sin, nh, hd);
            ex::rope_neox_slice(kr, &cos, &sin, nkv, hd);
            let out = &mut attn[r * qd..(r + 1) * qd];
            let cache = self.cpu_q8_kv.as_mut().unwrap();
            cache.append(kr, vr)?;
            if nh > nkv {
                cache.attention_grouped(qr, scale, out)?;
            } else {
                cache.attention(qr, scale, out)?;
            }
            self.kv_cache.advance_seq_len(1);
        }
        let mut proj = vec![0f32; rows * hidden];
        need(self.attn_output.forward_rows_cpu(&attn, &mut proj)?)?;
        let rs = self.residual_scale.map(|s| s as f32);
        for (x, p) in xs.chunks_mut(hidden).zip(proj.chunks_mut(hidden)) {
            if let Some(b) = &nc.o_bias {
                ex::residual_add(p, b);
            }
            if let Some(s) = rs {
                for v in p.iter_mut() {
                    *v *= s;
                }
            }
            ex::residual_add(x, p);
        }

        // -- FFN block ------------------------------------------------------
        let inter = self.flags.intermediate_size;
        let (nw, neps) = (&nc.ffn_norm_w, nc.ffn_eps);
        let use_gelu = self.flags.use_gelu;
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::dispatch_avx2() && !use_gelu {
            if let (Some(u), Some(d)) = (self.ffn_up.cpu_raw(), self.ffn_down.cpu_raw()) {
                let g = self.ffn_gate.as_ref().and_then(|g| g.cpu_raw());
                let norm_fn = |x: &[f32], out: &mut [f32]| ex::rms_norm_slice(x, nw, neps, out);
                let act_fn = |g: &[f32], u: &[f32], out: &mut [f32]| ex::silu_mul_slice(g, u, out);
                if crate::tensor::quant_cpu::ffn_swiglu_coop_rows(
                    xs, hidden, inter, g, u, d, rs, &norm_fn, &act_fn,
                )? {
                    return Ok(());
                }
            }
        }
        for (x, n) in xs.chunks(hidden).zip(norm.chunks_mut(hidden)) {
            ex::rms_norm_slice(x, nw, neps, n);
        }
        let mut gateup = vec![0f32; rows * 2 * inter];
        if let Some(gate_w) = self.ffn_gate.as_ref() {
            let mut g = vec![0f32; rows * inter];
            let mut u = vec![0f32; rows * inter];
            need(gate_w.forward_rows_cpu(&norm, &mut g)?)?;
            need(self.ffn_up.forward_rows_cpu(&norm, &mut u)?)?;
            for r in 0..rows {
                let row = &mut gateup[r * 2 * inter..(r + 1) * 2 * inter];
                row[..inter].copy_from_slice(&g[r * inter..(r + 1) * inter]);
                row[inter..].copy_from_slice(&u[r * inter..(r + 1) * inter]);
            }
        } else {
            need(self.ffn_up.forward_rows_cpu(&norm, &mut gateup)?)?;
        }
        let mut act = vec![0f32; rows * inter];
        for (gu, a) in gateup.chunks(2 * inter).zip(act.chunks_mut(inter)) {
            if use_gelu {
                ex::gelu_mul_slice(&gu[..inter], &gu[inter..], a);
            } else {
                ex::silu_mul_slice(&gu[..inter], &gu[inter..], a);
            }
        }
        need(self.ffn_down.forward_rows_cpu(&act, &mut proj)?)?;
        for (x, p) in xs.chunks_mut(hidden).zip(proj.chunks_mut(hidden)) {
            if let Some(s) = rs {
                for v in p.iter_mut() {
                    *v *= s;
                }
            }
            ex::residual_add(x, p);
        }
        Ok(())
    }

    /// Entry of the rows executor for a `[1, rows, hidden]` F32 activation on the host:
    /// the updated residual, or `None` for the Tensor forward to run instead.
    pub(super) fn try_decode_rows_cpu(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<Option<Tensor>> {
        let dims = x.dims();
        if !matches!(x.device(), Device::Cpu)
            || dims.len() != 3
            || dims[0] != 1
            || dims[1] < 2
            || x.dtype() != DType::F32
        {
            return Ok(None);
        }
        let mut xs = x.flatten_all()?.to_vec1::<f32>()?;
        if !self.decode_rows_cpu(&mut xs, dims[2], index_pos)? {
            return Ok(None);
        }
        Ok(Some(Tensor::from_vec(xs, dims, &Device::Cpu)?))
    }
}
