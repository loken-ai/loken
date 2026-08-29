
// A block that has to agree on one number per row - a sum of squares, a running maximum, the
// best logit - parks one candidate per thread in shared memory and then folds the upper half
// onto the lower, halving what is live each round until slot zero holds the answer.
//
// The fold is guarded because only the lower half has a partner to take from; the barrier that
// follows it is not, because every thread has to see a round finish before the next one reads.
// That asymmetry is the whole of what these are, and it is stated here rather than at each of
// the places that needs it.

/// Fold two same-length vectors of per-thread partials down to slot zero of each.
///
/// They travel together because their rows were read together - a variance needs the sum and
/// the sum of squares, a two-input norm needs one square per input - and folding them in one
/// pass costs one set of barriers instead of two.
static __device__ __forceinline__ void fold_sum_pair(float* u, float* v, int tid) {
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            u[tid] += u[tid + s];
            v[tid] += v[tid + s];
        }
        __syncthreads();
    }
}

/// The best (value, index) a thread finds walking the logits from `first` in `stride` steps,
/// with the repetition penalty applied to the tokens the caller listed.
///
/// The penalty moves a logit towards zero from whichever side it is on, so it divides a
/// positive one and multiplies a negative one. The list is a handful of recent tokens, short
/// enough that scanning it per logit beats building anything to look them up in.
///
/// A strict `>` keeps the FIRST of equal values, and the walk visits indices in ascending
/// order, so the lowest token id wins a tie no matter how the vocabulary was divided up.
static __device__ __forceinline__ void best_penalised_logit(
    const float* __restrict__ logits,
    const int* __restrict__ penalty_ids,
    int n_penalty,
    float penalty,
    int vocab_size,
    int first,
    int stride,
    float& best_val,
    int& best_idx) {
    for (int i = first; i < vocab_size; i += stride) {
        float val = logits[i];
        for (int p = 0; p < n_penalty; p++) {
            if (penalty_ids[p] == i) {
                if (val >= 0.0f) { val /= penalty; }
                else { val *= penalty; }
                break;
            }
        }
        if (val > best_val) { best_val = val; best_idx = i; }
    }
}

/// Fold per-thread (value, index) candidates down to the block's best at slot zero.
///
/// `pairs` holds two floats per thread: the value, and the index carried alongside it in the
/// bits of a float, which keeps the pair in one array and one fold. The larger value wins.
///
/// `TIE_BY_INDEX` says what a draw means. Where the candidates were gathered in ascending
/// index order the left-hand slot already holds the lower index and keeping it is enough;
/// where they arrive from elsewhere - a first stage's per-block winners, say - the index has
/// to be compared, or which of two equal values is returned depends on how the work was split.
template <bool TIE_BY_INDEX>
static __device__ __forceinline__ void fold_best_pair(float* pairs, int tid) {
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float here = pairs[tid * 2];
            const float there = pairs[(tid + s) * 2];
            bool take = there > here;
            if (TIE_BY_INDEX && !take && there == here) {
                take = __float_as_int(pairs[(tid + s) * 2 + 1])
                     < __float_as_int(pairs[tid * 2 + 1]);
            }
            if (take) {
                pairs[tid * 2] = there;
                pairs[tid * 2 + 1] = pairs[(tid + s) * 2 + 1];
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void fused_silu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float s = g / (1.0f + expf(-g));
        out[idx] = s * up[idx];
    }
}

// Standalone SiLU / Sigmoid (F32), elementwise. Native replacements for
// {silu,sigmoid} on the F32 CUDA path.
extern "C" __global__ void silu_f32(
    const float* __restrict__ x, float* __restrict__ out, const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float v = x[idx]; out[idx] = v / (1.0f + expf(-v)); }
}
extern "C" __global__ void sigmoid_f32(
    const float* __restrict__ x, float* __restrict__ out, const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = 1.0f / (1.0f + expf(-x[idx])); }
}

// RoPE (F32). x is [b, h, s, d] contiguous (row = b*h*s flat, d=rope dim);
// cos/sin are [s, d/2]. One block per row; seq index = row % seq (the [b,h,s,d]
// layout makes consecutive rows within a (b,h) increment the seq position).
// Native replacement for {rope,rope_i},
// matching the reference exact formulas.
//   NeoX (non-interleaved): pairs (j, j+d/2)
extern "C" __global__ void rope_neox_f32(
    const float* __restrict__ x, const float* __restrict__ cos, const float* __restrict__ sin,
    float* __restrict__ out, const int n_rows, const int d, const int seq
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int half = d / 2;
    int sidx = row % seq;
    int xoff = row * d;
    int coff = sidx * half;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float c = cos[coff + j], s = sin[coff + j];
        float x1 = x[xoff + j], x2 = x[xoff + j + half];
        out[xoff + j]        = x1 * c - x2 * s;
        out[xoff + j + half] = x2 * c + x1 * s;
    }
}
//   Interleaved (GPT-J / llama-ggml): pairs (2j, 2j+1)
extern "C" __global__ void rope_interleaved_f32(
    const float* __restrict__ x, const float* __restrict__ cos, const float* __restrict__ sin,
    float* __restrict__ out, const int n_rows, const int d, const int seq
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int half = d / 2;
    int sidx = row % seq;
    int xoff = row * d;
    int coff = sidx * half;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float c = cos[coff + j], s = sin[coff + j];
        float x1 = x[xoff + 2 * j], x2 = x[xoff + 2 * j + 1];
        out[xoff + 2 * j]     = x1 * c - x2 * s;
        out[xoff + 2 * j + 1] = x1 * s + x2 * c;
    }
}

// On-device additive attention mask (gpt-oss). Writes the [seq, kv] causal +
// optional sliding-window mask (0 visible, -inf masked) directly on the GPU,
// replacing a host-built Vec + H2D upload. The upload was the sole CUDA-graph
// replay blocker (a captured H2D memcpy from a host buffer freed after capture ->
// wild pointer); generating on-device removes it. `input_pos` is a kernel arg
// (host scalar, baked into the graph node) - correct for normal decode; the
// device-pos rewrite will source it from a device buffer for correct replay.
extern "C" __global__ void fused_gptoss_mask_f32(
    float* __restrict__ out,   // [seq * kv]
    const int seq,
    const int kv,
    const int input_pos,
    const int window           // 0 = full causal; >0 = sliding window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq * kv) return;
    int i = idx / kv;
    int j = idx % kv;
    int qpos = input_pos + i;
    bool masked = (j > qpos) || (window > 0 && (qpos - j) >= window);
    out[idx] = masked ? __int_as_float(0xff800000) : 0.0f; // -inf bit pattern
}

// Fused OAI clamped-SwiGLU (gpt-oss): one elementwise launch in place of the
// ~7 ops in swiglu_oai_combine (2 clamps, affine, sigmoid, 2 muls, affine).
//   x   = min(gate, limit)
//   g   = clamp(up, -limit, limit)
//   out = (x * sigmoid(alpha * x)) * (1 + g)
extern "C" __global__ void fused_swiglu_oai_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int n,
    const float alpha,
    const float limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float x = fminf(gate[idx], limit);
    const float g = fminf(fmaxf(up[idx], -limit), limit);
    const float sig = 1.0f / (1.0f + expf(-alpha * x));
    out[idx] = (x * sig) * (1.0f + g);
}

// Mamba2 post-SSM: D-skip + SiLU(z) gate + group-wise gated RMSNorm, fused into
// one launch (was ~12). One BLOCK per (batch, group) reduces over the group of
// size gs = d_inner/ngroups:
//   yv[idx] = (y_ssm[idx] + d[head]*x_c[idx]) * silu(z[idx])    head = idx/hd
//   var = mean_{idx in group}(yv^2);  out[idx] = yv*rsqrt(var+eps)*norm_w[g,local]
extern "C" __global__ void fused_mamba2_gate_gnorm_f32(
    const float* __restrict__ y_ssm,  // [b, d_inner]
    const float* __restrict__ x_c,    // [b, d_inner]
    const float* __restrict__ dvec,   // [nh]
    const float* __restrict__ z,      // [b, d_inner]
    const float* __restrict__ norm_w, // [ngroups, gs]
    float* __restrict__ out,          // [b, d_inner]
    const int b, const int d_inner, const int hd, const int ngroups, const float eps
) {
    const int bg = blockIdx.x;           // = bi*ngroups + g
    if (bg >= b * ngroups) return;
    const int bi = bg / ngroups;
    const int g  = bg % ngroups;
    const int gs = d_inner / ngroups;
    const int gbase = g * gs;            // group start within d_inner
    extern __shared__ float sdata[];
    float local = 0.0f;
    for (int j = threadIdx.x; j < gs; j += blockDim.x) {
        const int idx = gbase + j;        // within d_inner
        const int head = idx / hd;
        const float ys = y_ssm[bi * d_inner + idx];
        const float xc = x_c[bi * d_inner + idx];
        const float zv = z[bi * d_inner + idx];
        const float silu_z = zv / (1.0f + expf(-zv));
        const float yv = (ys + dvec[head] * xc) * silu_z;
        out[bi * d_inner + idx] = yv;     // stash; normalized in 2nd pass
        local += yv * yv;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sdata[0] / (float)gs + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < gs; j += blockDim.x) {
        const int idx = gbase + j;
        out[bi * d_inner + idx] = out[bi * d_inner + idx] * inv * norm_w[g * gs + j];
    }
}

// Mamba2 causal depthwise conv1d + bias + SiLU + state-shift, fused per
// (batch, channel) into one launch (was ~15: state narrow, cat, d_conv-step
// conv loop, state store, silu). state holds the last L=d_conv inputs.
//   window = [state_in[1..L], x];  acc = bias[d] + Σ window.conv_w[d];
//   y = silu(acc);  state_out = window
extern "C" __global__ void fused_causal_conv1d_silu_f32(
    const float* __restrict__ x,         // [b, D]
    const float* __restrict__ state_in,  // [b, D, L]
    const float* __restrict__ conv_w,    // [D, L]
    const float* __restrict__ conv_b,    // [D]
    float* __restrict__ y_out,           // [b, D]
    float* __restrict__ state_out,       // [b, D, L]
    const int b, const int D, const int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // = bi*D + d
    if (idx >= b * D) return;
    const int d = idx % D;
    const float* st = state_in + (size_t)idx * L;
    const float* cw = conv_w + (size_t)d * L;
    const float xv = x[idx];
    float acc = conv_b[d];
    #pragma unroll 1
    for (int k = 0; k < L - 1; ++k) acc += st[k + 1] * cw[k];
    acc += xv * cw[L - 1];
    y_out[idx] = acc / (1.0f + expf(-acc)); // SiLU
    float* so = state_out + (size_t)idx * L;
    #pragma unroll 1
    for (int k = 0; k < L - 1; ++k) so[k] = st[k + 1];
    so[L - 1] = xv;
}

// Mamba2 single-token SSM recurrence, fused per (batch, head, headdim) into one
// launch in place of ~25 (group broadcasts, decay=exp(dt*a), x⊗B outer, state
// update, C contraction). For each (b,head,p), loop over d_state:
//   decay = exp(dt[b,head]*a[head]);  x = x_c[b,head,p]
//   h[s]  = h_in[s]*decay + dt*x*B[group,s];   y += h[s]*C[group,s]
// B/C are passed raw [b, ngroups*ds] with the group = head/(nh/ngroups).
extern "C" __global__ void fused_mamba2_ssm_step_f32(
    const float* __restrict__ x_c,   // [b, nh*hd]
    const float* __restrict__ b_,    // [b, ngroups*ds]
    const float* __restrict__ c_,    // [b, ngroups*ds]
    const float* __restrict__ dt,    // [b, nh]
    const float* __restrict__ a,     // [nh]
    const float* __restrict__ h_in,  // [b, nh, hd, ds]
    float* __restrict__ y_out,       // [b, nh, hd]
    float* __restrict__ h_out,       // [b, nh, hd, ds]
    const int b, const int nh, const int hd, const int ds, const int ngroups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // = (bi*nh+head)*hd + p
    if (idx >= b * nh * hd) return;
    const int p    = idx % hd;
    const int head = (idx / hd) % nh;
    const int bi   = idx / (nh * hd);
    const int hpg  = nh / ngroups;
    const int group = head / hpg;
    const float dt_v  = dt[bi * nh + head];
    const float decay = expf(dt_v * a[head]);
    const float x_v   = x_c[bi * nh * hd + head * hd + p];
    const float* b_row = b_ + (size_t)bi * ngroups * ds + (size_t)group * ds;
    const float* c_row = c_ + (size_t)bi * ngroups * ds + (size_t)group * ds;
    const float* hin  = h_in + (size_t)idx * ds;
    float* hout = h_out + (size_t)idx * ds;
    float y = 0.0f;
    for (int s = 0; s < ds; ++s) {
        const float hn = hin[s] * decay + dt_v * x_v * b_row[s];
        hout[s] = hn;
        y += hn * c_row[s];
    }
    y_out[idx] = y;
}

// LFM2 gated short-conv inner, fused per (batch, channel) into one launch in
// place of ~20 (3 narrows, B*X gate, cat, the l_cache-step conv loop, state
// shift, C gate). bcx = in_proj(x) split as [B | C | X] each d_model wide.
//   bx        = B[d] * X[d]
//   acc       = Σ_{k<L-1} state_in[b,d,k]*conv_w[d,k] + bx*conv_w[d,L-1]
//   y[b,d]    = C[d] * acc
//   state_out = shift_left(state_in) with bx appended    (length L-1)
extern "C" __global__ void fused_lfm2_shortconv_f32(
    const float* __restrict__ bcx,        // [b, 3*D]
    const float* __restrict__ state_in,   // [b, D, L-1]
    const float* __restrict__ conv_w,     // [D, L]
    float* __restrict__ y_out,            // [b, D]
    float* __restrict__ state_out,        // [b, D, L-1]
    const int b, const int D, const int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * D) return;
    const int bi = idx / D;
    const int d  = idx % D;
    const float bg = bcx[bi * 3 * D + d];
    const float cg = bcx[bi * 3 * D + D + d];
    const float xg = bcx[bi * 3 * D + 2 * D + d];
    const float bx = bg * xg;
    const int Lm1 = L - 1;
    const float* st = state_in + (size_t)(bi * D + d) * Lm1;
    const float* cw = conv_w + (size_t)d * L;
    float acc = 0.0f;
    #pragma unroll 1
    for (int k = 0; k < Lm1; ++k) acc += st[k] * cw[k];
    acc += bx * cw[Lm1];
    y_out[bi * D + d] = cg * acc;
    float* so = state_out + (size_t)(bi * D + d) * Lm1;
    #pragma unroll 1
    for (int k = 0; k < Lm1 - 1; ++k) so[k] = st[k + 1];
    so[Lm1 - 1] = bx;
}

// Same as fused_swiglu_oai_f32 but adds per-row gate/up bias (already gathered
// to [M,N], matching gate/up) before the GLU - folds 2 broadcast_adds into the
// epilogue. gpt-oss has both biases.
extern "C" __global__ void fused_swiglu_oai_bias_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ gbias,
    const float* __restrict__ ubias,
    float* __restrict__ out,
    const int n,
    const float alpha,
    const float limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float x = fminf(gate[idx] + gbias[idx], limit);
    const float g = fminf(fmaxf(up[idx] + ubias[idx], -limit), limit);
    const float sig = 1.0f / (1.0f + expf(-alpha * x));
    out[idx] = (x * sig) * (1.0f + g);
}

// Fused GELU(tanh) * mul: out[i] = gelu_tanh(gate[i]) * up[i].
// Mirrors fused_silu_mul_f32 but with the gelu_pytorch_tanh activation.
// Replaces gate.gelu() + .mul(up) (2 launches) with 1 elementwise launch.
extern "C" __global__ void fused_gelu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = gate[idx];
    const float u = up[idx];
    const float k0 = 0.7978845608028654f;       // sqrt(2/pi)
    const float k1 = 0.044715f;
    const float gelu_g = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
    out[idx] = gelu_g * u;
}

// Fused split + GELU(tanh-approx) + mul on a packed [M, 2N] tensor.
//   For row r in [0, M), col c in [0, N):
//     g = gu[r, c]              // gate half (first N cols)
//     u = gu[r, N + c]           // up half  (second N cols)
//     out[r, c] = gelu_tanh(g) * u
// Replaces narrow + contiguous + narrow + contiguous + gelu + mul
// (≈6 kernel launches) with a single elementwise launch. Used by the
// gemma4-MoE forward path on the moe_gemm_gguf gate||up output.
//
// gelu_tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__ void fused_split_gelu_mul_f32(
    const float* __restrict__ gu,    // [M, 2N], contiguous
    float* __restrict__ out,         // [M, N], contiguous
    const int M,
    const int N
) {
    const int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int r = idx / N;
    const int c = idx - r * N;
    const int two_n = N << 1;
    const int gu_row = r * two_n;
    const float g = gu[gu_row + c];
    const float u = gu[gu_row + N + c];
    // gelu(tanh-approx)
    const float k0 = 0.7978845608028654f;       // sqrt(2/pi)
    const float k1 = 0.044715f;
    const float gelu_g = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
    out[idx] = gelu_g * u;
}

// Fused split + SiLU + mul on a packed [M, 2N] tensor - SiLU sister of
// fused_split_gelu_mul_f32. SiLU(x) = x * sigmoid(x). Used by SiLU-FFN
// models (llama, qwen2/qwen3 dense, deepcoder, devstral) when ffn_up
// holds [gate || up] concatenated.
extern "C" __global__ void fused_split_silu_mul_f32(
    const float* __restrict__ gu,    // [M, 2N], contiguous
    float* __restrict__ out,         // [M, N], contiguous
    const int M,
    const int N
) {
    const int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int r = idx / N;
    const int c = idx - r * N;
    const int two_n = N << 1;
    const int gu_row = r * two_n;
    const float g = gu[gu_row + c];
    const float u = gu[gu_row + N + c];
    const float silu_g = g / (1.0f + expf(-g));
    out[idx] = silu_g * u;
}

// Fused (x + residual) + RmsNorm with dual output:
//   sum_out[i] = x[i] + residual[i]           (for next residual connection)
//   norm_out[i] = rmsnorm(sum, weight, eps)    (for next sublayer input)
// Replaces: add kernel + rmsnorm kernel (2->1, saves 28 launches/token)
// Gemma4 fused post-attn-norm + residual + ffn-norm in one kernel:
//   y       = rmsnorm(attn_out, w_post)
//   x       = y + residual                       (saved -> next-layer residual)
//   x_norm  = rmsnorm(x, w_ffn)                   (fed to FFN)
// Replaces 3 launches (post_norm + add + ffn_norm) with 1 launch.
// Per project_gemma4_fused_post_norm_add_norm_2026_05_23 memory:
// saves 2 launches/layer x 30 = 60 launches/token ≈ +3 tok/s,
// exact flip margin for gemma4:latest medium cell.
extern "C" __global__ void fused_gemma4_post_add_norm_f32(
    const float* __restrict__ attn_out,    // [rows, cols]
    const float* __restrict__ residual,    // [rows, cols]
    const float* __restrict__ w_post,      // [cols] - post_attn_norm weight
    const float* __restrict__ w_ffn,       // [cols] - ffn_norm weight
    float* __restrict__ x_out,             // [rows, cols] - post_norm + residual
    float* __restrict__ x_norm_out,        // [rows, cols] - rmsnorm(x_out, w_ffn)
    const float eps,
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];
    int offset = row * cols;

    // Reduction 1: sum(attn_out²) -> rms_post
    float sum_a2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = attn_out[offset + i];
        sum_a2 += v * v;
    }
    sdata[threadIdx.x] = sum_a2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms_post = rsqrtf(sdata[0] / (float)cols + eps);
    __syncthreads();

    // Apply post_norm + add residual, accumulate sum for ffn_norm
    float sum_x2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float y = attn_out[offset + i] * rms_post * w_post[i];
        float x = y + residual[offset + i];
        x_out[offset + i] = x;
        sum_x2 += x * x;
    }
    sdata[threadIdx.x] = sum_x2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms_ffn = rsqrtf(sdata[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        x_norm_out[offset + i] = x_out[offset + i] * rms_ffn * w_ffn[i];
    }
}

extern "C" __global__ void fused_add_rmsnorm_dual_f32(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ sum_out,
    float* __restrict__ norm_out,
    const float eps,
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];
    int offset = row * cols;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[offset + i] + residual[offset + i];
        sum_out[offset + i] = val;
        thread_sum += val * val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        norm_out[offset + i] = sum_out[offset + i] * rms * weight[i];
    }
}

// Fused dual RmsNorm + add residual:
//   out[r,c] = rmsnorm(a[r,:], w1)[c] + rmsnorm(b[r,:], w2)[c] + c_in[r,c]
// Replaces 4 launches for the gemma4-MoE post-FFN combine:
//   cur_mlp = post_ffw_norm_1(cur_mlp)        // 1 launch
//   cur_moe = post_ffw_norm_2(cur_moe)        // 1 launch
//   combined = cur_mlp + cur_moe              // 1 launch
//   x = combined + residual_ffn               // 1 launch
// with one kernel that reads each row of a, b, c_in once, applies their
// norms (a,b only), then sums all three.
extern "C" __global__ void fused_dual_rmsnorm_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c_in,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    float* __restrict__ out,
    const float eps,
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];           // 2 x blockDim.x: [a^2 | b^2]
    float* s_a = sdata;
    float* s_b = sdata + blockDim.x;
    int offset = row * cols;

    float sum_a2 = 0.0f;
    float sum_b2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float av = a[offset + i];
        const float bv = b[offset + i];
        sum_a2 += av * av;
        sum_b2 += bv * bv;
    }
    s_a[threadIdx.x] = sum_a2;
    s_b[threadIdx.x] = sum_b2;
    __syncthreads();
    fold_sum_pair(s_a, s_b, threadIdx.x);
    const float rms_a = rsqrtf(s_a[0] / (float)cols + eps);
    const float rms_b = rsqrtf(s_b[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[offset + i] =
            a[offset + i] * rms_a * w1[i] +
            b[offset + i] * rms_b * w2[i] +
            c_in[offset + i];
    }
}

// Fused rmsnorm(x, w) + add(residual): out[r,c] = rmsnorm(x[r,:], w)[c] + residual[r,c]
// Replaces the 2-launch chain `post_norm.forward(x) + residual` used at
// the tail of every gemma4 transformer layer (post_attn_norm/post_ffn_norm
// pattern). Each layer saves 1 launch - over 30 layers = ~30 launches
// per token. Per project_gemma4_latest_medium_profile_2026_05_22 memory:
// 5 µs/layer saved x 30 layers ≈ the +3 tok/s margin needed to flip the
// gemma4:latest medium cell.
//
// Unlike fused_gemma4_post_add_norm_f32 (which tried to absorb the next
// layer's ffn_norm too - 3 ops -> 1 - and regressed because the kernel
// re-stored its first output for a second reduction, doubling write
// traffic), this kernel does a single reduction and writes once.
extern "C" __global__ void fused_rmsnorm_then_add_f32(
    const float* __restrict__ x,           // [rows, cols] - pre-norm input
    const float* __restrict__ weight,      // [cols]
    const float* __restrict__ residual,    // [rows, cols] - added AFTER norm
    float* __restrict__ out,               // [rows, cols]
    const float eps,
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata_rta[];
    int offset = row * cols;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[offset + i];
        thread_sum += v * v;
    }
    sdata_rta[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_rta[threadIdx.x] += sdata_rta[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata_rta[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[offset + i] = x[offset + i] * rms * weight[i] + residual[offset + i];
    }
}

// Standalone RMSNorm (F32), no residual: out = x / sqrt(mean(x^2)+eps) * weight.
// Drop-in native replacement for crate::inference::native_ops::rms_norm on the F32 CUDA path
// Reduction in F32 - matches the reference
// internal F32 accumulation for F16/BF16 too (callers promote first).
extern "C" __global__ void fused_rmsnorm_f32(
    const float* __restrict__ x,           // [rows, cols]
    const float* __restrict__ weight,      // [cols]
    float* __restrict__ out,               // [rows, cols]
    const float eps,
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata_rn[];
    int offset = row * cols;
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[offset + i];
        thread_sum += v * v;
    }
    sdata_rn[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_rn[threadIdx.x] += sdata_rn[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata_rn[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[offset + i] = x[offset + i] * rms * weight[i];
    }
}

// Wide-block variant of fused_rmsnorm_f32 for the launch-starved decode case
// (few rows -> few blocks -> a 256-thread block leaves the SM mostly idle and
// the row is read from global memory twice). The row is staged once into
// shared memory by the FULL block, the squared-sum keeps the EXACT 256-lane
// strided accumulation + 256-entry tree of the narrow kernel (bit-identical
// output - the parity snapshots hash the downstream KV cache), and the
// normalize/scale pass again uses the full block reading the smem copy.
// Caller guarantees: cols >= 256 and cols*4 fits the dynamic-smem budget.
extern "C" __global__ void fused_rmsnorm_wide_f32(
    const float* __restrict__ x,           // [rows, cols]
    const float* __restrict__ weight,      // [cols]
    float* __restrict__ out,               // [rows, cols]
    const float eps,
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float srow_rnw[];    // [cols] staged row
    __shared__ float stree_rnw[256];
    int offset = row * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        srow_rnw[i] = x[offset + i];
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        // Same per-lane sequential order as the narrow kernel (stride 256).
        float thread_sum = 0.0f;
        for (int i = threadIdx.x; i < cols; i += 256) {
            float v = srow_rnw[i];
            thread_sum += v * v;
        }
        stree_rnw[threadIdx.x] = thread_sum;
    }
    __syncthreads();
    // Same binary-tree combine as the narrow kernel (blockDim.x there = 256).
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) stree_rnw[threadIdx.x] += stree_rnw[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(stree_rnw[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[offset + i] = srow_rnw[i] * rms * weight[i];
    }
}

// Fused LayerNorm (F32): out = (x - mean) / sqrt(var + eps) * w (+ b).
// One block per row; one shared-mem pass reducing sum and sum-of-squares
// together. The composed-op LayerNorm costs ~8-10 launches per call - on
// launch-bound fast models (moondream phi2: ~50 norms/token at 434 tok/s)
// that alone roughly halved decode (measured).
extern "C" __global__ void fused_layernorm_f32(
    const float* __restrict__ x,           // [rows, cols]
    const float* __restrict__ weight,      // [cols]
    const float* __restrict__ bias,        // [cols] (ignored if has_bias=0)
    float* __restrict__ out,               // [rows, cols]
    const float eps,
    const int cols,
    const int has_bias
) {
    int row = blockIdx.x;
    extern __shared__ float sdata_ln[];
    float* ssum = sdata_ln;
    float* ssq = sdata_ln + blockDim.x;
    int offset = row * cols;
    float tsum = 0.0f, tsq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[offset + i];
        tsum += v;
        tsq += v * v;
    }
    ssum[threadIdx.x] = tsum;
    ssq[threadIdx.x] = tsq;
    __syncthreads();
    fold_sum_pair(ssum, ssq, threadIdx.x);
    float mean = ssum[0] / (float)cols;
    float var = ssq[0] / (float)cols - mean * mean;
    float inv = rsqrtf(fmaxf(var, 0.0f) + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float r = (x[offset + i] - mean) * inv * weight[i];
        out[offset + i] = has_bias ? r + bias[i] : r;
    }
}

// Last-dim softmax (F32): out = exp(x - rowmax) / sum(exp(x - rowmax)).
// Native replacement for crate::inference::native_ops::softmax_last_dim on the F32 CUDA
// path. One block per row (last-dim
// vector); two block reductions (max, then exp-sum). Handles -inf mask
// entries (exp(-inf)=0) like the tensor-level path.
// Softmax over the last dim with BF16 storage and F32 math.
//
// The attention score tile is the biggest buffer in a DiT forward, and it used to
// cross HBM four times around this operation: the BF16 GEMM wrote it, a cast read it
// and wrote it as F32, the F32 softmax read and wrote it, and a second cast read it
// back down to BF16 for the value GEMM. Reading and writing it AS BF16, with the
// arithmetic still in F32, removes both casts and halves the bytes of what remains.
//
// BIT-IDENTICAL to that chain by construction. Widening bf16->f32 is exact, the
// reductions and the exp run in f32 exactly as before, and the result is rounded to
// bf16 exactly once - which is what the final cast did. The exponential is RECOMPUTED
// in the last pass rather than stashed: storing it would round it to bf16 and then
// round again after scaling, and two roundings are not one.
//
// NVRTC compiles this without headers, so bf16 is carried as unsigned short and the
// rounding is spelled out to match the host side's round-to-nearest-even exactly.
__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    return __int_as_float(((int)h) << 16);
}
__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
    unsigned int x = __float_as_uint(f);
    // NaN: keep it a NaN with the quiet bit set, as the host conversion does.
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short)((x >> 16) | 0x0040u);
    }
    const unsigned int round_bit = 0x00008000u;
    if ((x & round_bit) != 0u && (x & (3u * round_bit - 1u)) != 0u) {
        return (unsigned short)((x >> 16) + 1u);
    }
    return (unsigned short)(x >> 16);
}

extern "C" __global__ void fused_softmax_lastdim_bf16(
    const unsigned short* __restrict__ x,  // [rows, cols]
    unsigned short* __restrict__ out,      // [rows, cols]
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata_bf[];
    int offset = row * cols;
    float tmax = __int_as_float(0xff800000); // -inf
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        tmax = fmaxf(tmax, bf16_to_f32(x[offset + i]));
    }
    sdata_bf[threadIdx.x] = tmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_bf[threadIdx.x] = fmaxf(sdata_bf[threadIdx.x], sdata_bf[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata_bf[0];
    __syncthreads();
    float tsum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        tsum += expf(bf16_to_f32(x[offset + i]) - row_max);
    }
    sdata_bf[threadIdx.x] = tsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_bf[threadIdx.x] += sdata_bf[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata_bf[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(bf16_to_f32(x[offset + i]) - row_max);
        out[offset + i] = f32_to_bf16(e * inv);
    }
}

extern "C" __global__ void fused_softmax_lastdim_f32(
    const float* __restrict__ x,           // [rows, cols]
    float* __restrict__ out,               // [rows, cols]
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata_sm[];
    int offset = row * cols;
    float tmax = __int_as_float(0xff800000); // -inf (NVRTC has no <math.h> INFINITY)
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        tmax = fmaxf(tmax, x[offset + i]);
    }
    sdata_sm[threadIdx.x] = tmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_sm[threadIdx.x] = fmaxf(sdata_sm[threadIdx.x], sdata_sm[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata_sm[0];
    __syncthreads();
    float tsum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        tsum += expf(x[offset + i] - row_max);
    }
    sdata_sm[threadIdx.x] = tsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_sm[threadIdx.x] += sdata_sm[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata_sm[0];
    // RECOMPUTE the exponential rather than parking it in `out` and reading it back.
    // The old shape cost a write and a read of the full row in exchange for saving one
    // expf, and on the buffers this runs on - attention scores are the widest tensor in
    // a DiT forward - that trade is backwards: the row does not fit in cache, so the
    // round trip is HBM traffic while the expf is free arithmetic on a bandwidth-bound
    // kernel. Five passes over the row become four. An f32 store and load is exact, so
    // the recomputed value is the same bits the reload would have produced.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x[offset + i] - row_max);
        out[offset + i] = e * inv;
    }
}

// Variant of fused_rmsnorm_then_add that also applies a column-broadcast
// scale to every row of the output. Targets gemma4's PLE block tail:
//   x = rmsnorm(ple_state) + x;
//   x = x * layer_output_scale       // [hidden_dim] broadcast over (B, T)
// Folds three launches (rmsnorm, add, broadcast_mul) into one.
extern "C" __global__ void fused_rmsnorm_add_scale_f32(
    const float* __restrict__ x,           // [rows, cols] - pre-norm input
    const float* __restrict__ weight,      // [cols]
    const float* __restrict__ residual,    // [rows, cols]
    const float* __restrict__ scale,       // [cols] - column-broadcast
    float* __restrict__ out,               // [rows, cols]
    const float eps,
    const int cols
) {
    int row = blockIdx.x;
    extern __shared__ float sdata_ras[];
    int offset = row * cols;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[offset + i];
        thread_sum += v * v;
    }
    sdata_ras[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_ras[threadIdx.x] += sdata_ras[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata_ras[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[offset + i] * rms * weight[i] + residual[offset + i];
        out[offset + i] = v * scale[i];
    }
}

// Fused repeat penalty + argmax: apply penalty to specific indices, then find argmax.
// Replaces: index_select + ge + affine + mul + scatter + argmax (6->1 kernel)
// Returns the argmax token index via out_token[0].
extern "C" __global__ void fused_penalty_argmax_f32(
    const float* __restrict__ logits,     // [vocab_size]
    const int* __restrict__ penalty_ids,  // [n_penalty] - token indices to penalize
    const int n_penalty,
    const float penalty,                  // repeat penalty value (e.g., 1.1)
    const int vocab_size,
    int* __restrict__ out_token           // [1] - output argmax token
) {
    extern __shared__ float sdata[];       // [blockDim.x * 2] - val + idx pairs

    int tid = threadIdx.x;
    float best_val = -1e30f;
    int best_idx = 0;

    // One block covers the whole vocabulary, so a thread strides by the block.
    best_penalised_logit(logits, penalty_ids, n_penalty, penalty, vocab_size,
                         tid, blockDim.x, best_val, best_idx);

    // Store in shared memory for reduction
    sdata[tid * 2] = best_val;
    sdata[tid * 2 + 1] = __int_as_float(best_idx);
    __syncthreads();

    // Candidates were gathered in ascending token order, so the left slot breaks a tie.
    fold_best_pair</*TIE_BY_INDEX=*/false>(sdata, tid);

    if (tid == 0) {
        out_token[0] = __float_as_int(sdata[1]);
    }
}

// ------------------------------------------------------------
// Multi-block argmax - 2-kernel chain replaces the single-block scan.
//
// nsys profile of gemma4:latest decode: the single-block
// fused_penalty_argmax_f32 averages 417 µs/call across 320 calls = 5.6 %
// of decode time (133 ms / 2.4 s). vocab_size=262144 for gemma4. With
// gridDim=(1,1,1) only 1 of the GPU's ~84 SMs is active, so the kernel
// runs ~80x slower than achievable. Multi-block fix:
//   Pass 1 (block_argmax):  N blocks each find local argmax over
//                           vocab_size/N elements, write (val,idx) into
//                           global[N] pair buffer.
//   Pass 2 (final_reduce):  1 block reduces the N candidates.
//
// Tie-breaking: lowest index wins (matches the original > predicate's
// implicit behavior since the lower-index thread loops first).
// ------------------------------------------------------------

extern "C" __global__ void fused_penalty_argmax_block_f32(
    const float* __restrict__ logits,
    const int* __restrict__ penalty_ids,
    const int n_penalty,
    const float penalty,
    const int vocab_size,
    float* __restrict__ block_vals,
    int* __restrict__ block_idxs
) {
    extern __shared__ float sdata_mb[];        // [blockDim.x * 2]
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gridSize = gridDim.x * blockDim.x;
    int g0 = bid * blockDim.x + tid;
    float best_val = -1e30f;
    int   best_idx = 0;
    // The vocabulary is spread over the whole grid here, so a thread strides by the grid.
    best_penalised_logit(logits, penalty_ids, n_penalty, penalty, vocab_size,
                         g0, gridSize, best_val, best_idx);
    sdata_mb[tid * 2]     = best_val;
    sdata_mb[tid * 2 + 1] = __int_as_float(best_idx);
    __syncthreads();
    // Ascending token order within the block, so the left slot breaks a tie here too.
    fold_best_pair</*TIE_BY_INDEX=*/false>(sdata_mb, tid);
    if (tid == 0) {
        block_vals[bid] = sdata_mb[0];
        block_idxs[bid] = __float_as_int(sdata_mb[1]);
    }
}

extern "C" __global__ void fused_penalty_argmax_final_f32(
    const float* __restrict__ block_vals,
    const int*   __restrict__ block_idxs,
    const int    n_blocks,
    int*   __restrict__ out_token
) {
    extern __shared__ float sdata_fr[];        // [blockDim.x * 2]
    int tid = threadIdx.x;
    float best_val = -1e30f;
    int   best_idx = 0;
    for (int i = tid; i < n_blocks; i += blockDim.x) {
        float v = block_vals[i];
        int   idx = block_idxs[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }
    sdata_fr[tid * 2]     = best_val;
    sdata_fr[tid * 2 + 1] = __int_as_float(best_idx);
    __syncthreads();
    // These candidates are the first stage's per-block winners, whose indices are in no
    // particular order, so a tie has to be settled on the index itself.
    fold_best_pair</*TIE_BY_INDEX=*/true>(sdata_fr, tid);
    if (tid == 0) {
        out_token[0] = __float_as_int(sdata_fr[1]);
    }
}

// u32-out variant of the final reduce, for Path B (Tensor::DType::U32).
extern "C" __global__ void fused_penalty_argmax_final_f32_u32_out(
    const float* __restrict__ block_vals,
    const int*   __restrict__ block_idxs,
    const int    n_blocks,
    unsigned int* __restrict__ out_token
) {
    extern __shared__ float sdata_fru[];
    int tid = threadIdx.x;
    float best_val = -1e30f;
    int   best_idx = 0;
    for (int i = tid; i < n_blocks; i += blockDim.x) {
        float v = block_vals[i];
        int   idx = block_idxs[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }
    sdata_fru[tid * 2]     = best_val;
    sdata_fru[tid * 2 + 1] = __int_as_float(best_idx);
    __syncthreads();
    // Per-block winners again: settle a tie on the index.
    fold_best_pair</*TIE_BY_INDEX=*/true>(sdata_fru, tid);
    if (tid == 0) {
        out_token[0] = (unsigned int)__float_as_int(sdata_fru[1]);
    }
}

// Same as fused_penalty_argmax_f32 but writes the argmax index as unsigned
// int.
extern "C" __global__ void fused_penalty_argmax_f32_u32_out(
    const float* __restrict__ logits,
    const int* __restrict__ penalty_ids,
    const int n_penalty,
    const float penalty,
    const int vocab_size,
    unsigned int* __restrict__ out_token
) {
    extern __shared__ float sdata2[];
    int tid = threadIdx.x;
    float best_val = -1e30f;
    int best_idx = 0;
    best_penalised_logit(logits, penalty_ids, n_penalty, penalty, vocab_size,
                         tid, blockDim.x, best_val, best_idx);
    sdata2[tid * 2] = best_val;
    sdata2[tid * 2 + 1] = __int_as_float(best_idx);
    __syncthreads();
    // Ascending token order, so the left slot breaks a tie.
    fold_best_pair</*TIE_BY_INDEX=*/false>(sdata2, tid);
    if (tid == 0) {
        out_token[0] = (unsigned int)__float_as_int(sdata2[1]);
    }
}

// Three-way elementwise add: out[i] = a[i] + b[i] + c[i].
// Replaces `(a + b)? + c` (2 launches + 1 intermediate alloc) with one
// kernel + one output alloc. Used by the parallel-attn (phi2 / gpt-neox)
// forward path where the residual + attn_out + ffn_out merge happens
// once per layer x token.
extern "C" __global__ void fused_add_three_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx] + c[idx];
    }
}

// Phi2 / GPT-NeoX parallel-attention residual merge with both biases
// folded in:
//   out[i] = residual[i] + (attn_out[i] + attn_bias[i % d])
//                        + (ffn_out[i] + ffn_bias[i % d])
// Replaces a 4-launch chain (attn_output_bias broadcast_add +
// ffn_down_bias broadcast_add + fused_add_three) with a single fused
// elementwise launch - 3 element launches -> 1 per layer x token.
// Used only on the phi2 simple-FFN parallel-attn fast path.
extern "C" __global__ void fused_phi2_residual_merge_f32(
    const float* __restrict__ residual,
    const float* __restrict__ attn_out,
    const float* __restrict__ attn_bias,
    const float* __restrict__ ffn_out,
    const float* __restrict__ ffn_bias,
    float* __restrict__ out,
    const int n,
    const int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int bi = idx % d;
    out[idx] = residual[idx] + attn_out[idx] + attn_bias[bi]
             + ffn_out[idx] + ffn_bias[bi];
}

// Fused (broadcast-add bias) + GELU(tanh-approx): out[i] = gelu_new(x[i] + bias[i % d]).
// Replaces `x.broadcast_add(bias)? .gelu()?` (2 launches + 1 intermediate
// alloc) with a single elementwise launch. Used by the phi2 simple FFN
// path on the ffn_up output prior to the down projection. Bias is the
// trailing dim of x; broadcast is implicit via modulo.
//
// gelu_new (== HF gelu_pytorch_tanh, == phi2 reference activation):
//   0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
extern "C" __global__ void fused_bias_gelu_new_f32(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int n,
    const int bias_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int b_idx = idx % bias_dim;
    const float v = x[idx] + bias[b_idx];
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    out[idx] = 0.5f * v * (1.0f + tanhf(k0 * (v + k1 * v * v * v)));
}

// ------------------------------------------------------------
// F32 single-query attention for HD=512 graph mode bypass (Path C).
//
// Replaces the cuBLAS-backed Q@K^T + softmax + P@V chain in
// `padded_standard_attention` that crashes on graph replay due to
// cuBLAS workspace state not being tracked by the reference Tensor lifetime
// (see project_gemma4_graph_ROOT_CAUSE_2026_05_28.md + the FA HD=512
// sm_120 SASS miscompile in project_fa_hd512_broken_sm120_2026_05_28).
//
// Designed for gemma4 Global layers: Q[1,8,1,512] x K[1,2,max_kv,512]
// with n_q_per_kv=4. Single-token decode (seq_q=1). Padded K/V layout
// matches the engine's existing stable graph_kv_buffer convention.
//
// Grid: (n_q_heads,) - one block per query head
// Block: 256 threads (8 warps x 32 lanes)
// Shared mem: (HD + TILE_KV) * 4 bytes = constant 3 KB (independent of max_kv_padded)
//
// Online softmax (FlashAttention-style) processes K/V in tiles of TILE_KV
// positions, maintaining (m_running, l_running, o_running) per thread.
// This makes smem footprint O(1) in max_kv_padded - sm_120 dynamic smem
// limit (≈100 KB) is no longer a problem for large contexts (32 K+).
//
// Output written to out[h_q * HD + d] for d ∈ [0, HD).
extern "C" __global__ void fused_attn_decode_f32_hd512(
    const float* __restrict__ Q,
    const float* __restrict__ K_full,
    const float* __restrict__ V_full,
    const float* __restrict__ mask,
    float* __restrict__ out,
    const int* __restrict__ seq_kv_dev,
    const int max_kv_padded,
    const int n_kv_heads,
    const int n_q_per_kv,
    const float scale
) {
    // NVRTC doesn't include <cmath> or <cstdint>: use bit-cast for -INF
    // and `int` (which is 32-bit on CUDA target ABI) instead of int32_t.
    const float NEG_INF_F = __int_as_float(0xff800000);
    constexpr int HD = 512;
    constexpr int THREADS = 256;
    constexpr int TILE_KV = 256;            // K/V positions per tile
    constexpr int WARP = 32;
    constexpr int NWARPS = THREADS / WARP;
    constexpr int OUT_PER_THREAD = HD / THREADS;  // = 2

    const int h_q = blockIdx.x;
    const int kv  = h_q / n_q_per_kv;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp_id = tid / WARP;
    const int seq_kv = seq_kv_dev[0] + 1;

    extern __shared__ float smem[];
    float* s_q      = smem;                  // [HD]
    float* s_scores = smem + HD;             // [TILE_KV]

    // Load Q[h_q] into shared memory.
    for (int d = tid; d < HD; d += THREADS) {
        s_q[d] = Q[h_q * HD + d];
    }
    __syncthreads();

    // Per-thread output accumulators for OUT_PER_THREAD d-positions.
    // d_base = tid * OUT_PER_THREAD covers all HD positions across THREADS
    // (THREADS * OUT_PER_THREAD == HD).
    const int d_base = tid * OUT_PER_THREAD;
    float o_local[OUT_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OUT_PER_THREAD; ++i) o_local[i] = 0.0f;

    // Online softmax running state. Use -1e30 (not -INF) so the first
    // exp() in alpha doesn't trap or produce NaN.
    float m_running = -1.0e30f;
    float l_running = 0.0f;

    __shared__ float warp_red[NWARPS];

    const int num_tiles = (seq_kv + TILE_KV - 1) / TILE_KV;
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = tile * TILE_KV;
        const int t_global = tile_start + tid;

        // Phase 1: compute score for this thread's t position.
        float s_val;
        if (t_global < seq_kv) {
            const float* k = K_full + (size_t)kv * max_kv_padded * HD + (size_t)t_global * HD;
            float sum = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HD; ++d) sum += s_q[d] * k[d];
            s_val = sum * scale + mask[t_global];
        } else {
            s_val = NEG_INF_F;
        }

        // Block-wide max of tile.
        float m_tile = s_val;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, off));
        }
        if (lane == 0) warp_red[warp_id] = m_tile;
        __syncthreads();
        if (warp_id == 0) {
            float m = (lane < NWARPS) ? warp_red[lane] : NEG_INF_F;
            #pragma unroll
            for (int off = NWARPS / 2; off > 0; off >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
            }
            if (lane == 0) warp_red[0] = m;
        }
        __syncthreads();
        const float m_tile_max = warp_red[0];

        // Online merge: m_new = max(m_running, m_tile_max).
        const float m_new = fmaxf(m_running, m_tile_max);
        const float alpha = __expf(m_running - m_new);
        const float p_local = (t_global < seq_kv) ? __expf(s_val - m_new) : 0.0f;
        s_scores[tid] = p_local;

        // Block-wide sum of tile's probs.
        float l_tile = p_local;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            l_tile += __shfl_xor_sync(0xFFFFFFFFu, l_tile, off);
        }
        if (lane == 0) warp_red[warp_id] = l_tile;
        __syncthreads();
        if (warp_id == 0) {
            float s = (lane < NWARPS) ? warp_red[lane] : 0.0f;
            #pragma unroll
            for (int off = NWARPS / 2; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
            }
            if (lane == 0) warp_red[0] = s;
        }
        __syncthreads();
        const float l_tile_sum = warp_red[0];

        // Rescale running state.
        l_running = l_running * alpha + l_tile_sum;
        #pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) {
            o_local[i] *= alpha;
        }

        // Update o_local: o += Σ_t in tile p[t] * V[t,kv,d]
        const int tile_count = min(TILE_KV, seq_kv - tile_start);
        for (int t_local = 0; t_local < tile_count; ++t_local) {
            const int t_global2 = tile_start + t_local;
            const float p = s_scores[t_local];
            const float* v = V_full + (size_t)kv * max_kv_padded * HD + (size_t)t_global2 * HD;
            #pragma unroll
            for (int i = 0; i < OUT_PER_THREAD; ++i) {
                o_local[i] += p * v[d_base + i];
            }
        }

        m_running = m_new;
        __syncthreads();
    }

    // Final normalize and write.
    const float inv_l = 1.0f / l_running;
    #pragma unroll
    for (int i = 0; i < OUT_PER_THREAD; ++i) {
        out[h_q * HD + d_base + i] = o_local[i] * inv_l;
    }
}

// SPLIT-K HD512 F32 flash-decode. The single-block-per-head kernel above
// launches only n_q_heads (=8 for gemma4 global) blocks, so at long KV those
// 8 blocks serially scan thousands of positions and underfill the GPU, losing
// to cuBLAS. This partial pass splits the KV range across `nsplit` blocks per
// head: grid=(n_q_heads, nsplit) -> 8xnsplit blocks. Each (h_q, split) block
// online-softmaxes its KV chunk [split*chunk, (split+1)*chunk) and writes an
// UN-normalized partial (o_local exp-weighted by its local max m, plus m and
// l). The combine kernel below flash-merges the nsplit partials per head.
// partials layout: [(h_q*nsplit + split) * (HD+2)] = { o[0..HD], m, l }.
extern "C" __global__ void fused_attn_decode_f32_hd512_splitk_partial(
    const float* __restrict__ Q,
    const float* __restrict__ K_full,
    const float* __restrict__ V_full,
    const float* __restrict__ mask,
    float* __restrict__ partials,
    const int* __restrict__ seq_kv_dev,
    const int max_kv_padded,
    const int n_kv_heads,
    const int n_q_per_kv,
    const int nsplit,
    const float scale
) {
    const float NEG_INF_F = __int_as_float(0xff800000);
    constexpr int HD = 512;
    constexpr int THREADS = 256;
    constexpr int TILE_KV = 256;
    constexpr int WARP = 32;
    constexpr int NWARPS = THREADS / WARP;
    constexpr int OUT_PER_THREAD = HD / THREADS; // 2

    const int h_q   = blockIdx.x;
    const int split = blockIdx.y;
    const int kv    = h_q / n_q_per_kv;
    const int tid   = threadIdx.x;
    const int lane  = tid & (WARP - 1);
    const int warp_id = tid / WARP;
    const int seq_kv = seq_kv_dev[0] + 1;

    const int chunk = (seq_kv + nsplit - 1) / nsplit;
    const int r0 = split * chunk;
    const int r1 = min(r0 + chunk, seq_kv);

    const int d_base = tid * OUT_PER_THREAD;
    float* p = partials + ((size_t)h_q * nsplit + split) * (HD + 2);

    // Empty split -> neutral partial.
    if (r0 >= r1) {
        #pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) p[d_base + i] = 0.0f;
        if (tid == 0) { p[HD] = NEG_INF_F; p[HD + 1] = 0.0f; }
        return;
    }

    extern __shared__ float smem[];
    float* s_q      = smem;          // [HD]
    float* s_scores = smem + HD;     // [TILE_KV]
    for (int d = tid; d < HD; d += THREADS) s_q[d] = Q[h_q * HD + d];
    __syncthreads();

    float o_local[OUT_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OUT_PER_THREAD; ++i) o_local[i] = 0.0f;
    float m_running = -1.0e30f;
    float l_running = 0.0f;
    __shared__ float warp_red[NWARPS];

    const int num_tiles = (r1 - r0 + TILE_KV - 1) / TILE_KV;
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = r0 + tile * TILE_KV;
        const int t_global = tile_start + tid;

        float s_val;
        if (t_global < r1) {
            const float* k = K_full + (size_t)kv * max_kv_padded * HD + (size_t)t_global * HD;
            float sum = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HD; ++d) sum += s_q[d] * k[d];
            s_val = sum * scale + mask[t_global];
        } else {
            s_val = NEG_INF_F;
        }

        float m_tile = s_val;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, off));
        if (lane == 0) warp_red[warp_id] = m_tile;
        __syncthreads();
        if (warp_id == 0) {
            float m = (lane < NWARPS) ? warp_red[lane] : NEG_INF_F;
            #pragma unroll
            for (int off = NWARPS / 2; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
            if (lane == 0) warp_red[0] = m;
        }
        __syncthreads();
        const float m_tile_max = warp_red[0];

        const float m_new = fmaxf(m_running, m_tile_max);
        const float alpha = __expf(m_running - m_new);
        const float p_local = (t_global < r1) ? __expf(s_val - m_new) : 0.0f;
        s_scores[tid] = p_local;

        float l_tile = p_local;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) l_tile += __shfl_xor_sync(0xFFFFFFFFu, l_tile, off);
        if (lane == 0) warp_red[warp_id] = l_tile;
        __syncthreads();
        if (warp_id == 0) {
            float s = (lane < NWARPS) ? warp_red[lane] : 0.0f;
            #pragma unroll
            for (int off = NWARPS / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
            if (lane == 0) warp_red[0] = s;
        }
        __syncthreads();
        const float l_tile_sum = warp_red[0];

        l_running = l_running * alpha + l_tile_sum;
        #pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) o_local[i] *= alpha;

        const int tile_count = min(TILE_KV, r1 - tile_start);
        for (int t_local = 0; t_local < tile_count; ++t_local) {
            const int t_global2 = tile_start + t_local;
            const float pw = s_scores[t_local];
            const float* v = V_full + (size_t)kv * max_kv_padded * HD + (size_t)t_global2 * HD;
            #pragma unroll
            for (int i = 0; i < OUT_PER_THREAD; ++i) o_local[i] += pw * v[d_base + i];
        }
        m_running = m_new;
        __syncthreads();
    }

    // Write UN-normalized partial (o exp-weighted by m_running) + m + l.
    #pragma unroll
    for (int i = 0; i < OUT_PER_THREAD; ++i) p[d_base + i] = o_local[i];
    if (tid == 0) { p[HD] = m_running; p[HD + 1] = l_running; }
}

// Combine: per head, flash-merge the nsplit partials. grid=(n_q_heads), block=256.
extern "C" __global__ void fused_attn_decode_f32_hd512_splitk_combine(
    const float* __restrict__ partials,
    float* __restrict__ out,
    const int nsplit
) {
    constexpr int HD = 512;
    constexpr int THREADS = 256;
    constexpr int OUT_PER_THREAD = HD / THREADS; // 2
    const int h_q = blockIdx.x;
    const int tid = threadIdx.x;
    const int d_base = tid * OUT_PER_THREAD;
    const float* base = partials + (size_t)h_q * nsplit * (HD + 2);

    float gm = __int_as_float(0xff800000);
    for (int sp = 0; sp < nsplit; ++sp) gm = fmaxf(gm, base[(size_t)sp * (HD + 2) + HD]);

    float gl = 0.0f;
    float acc[OUT_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OUT_PER_THREAD; ++i) acc[i] = 0.0f;
    for (int sp = 0; sp < nsplit; ++sp) {
        const float* p = base + (size_t)sp * (HD + 2);
        const float l_sp = p[HD + 1];
        if (l_sp <= 0.0f) continue;
        const float w = __expf(p[HD] - gm);
        gl += l_sp * w;
        #pragma unroll
        for (int i = 0; i < OUT_PER_THREAD; ++i) acc[i] += p[d_base + i] * w;
    }
    const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    #pragma unroll
    for (int i = 0; i < OUT_PER_THREAD; ++i) out[(size_t)h_q * HD + d_base + i] = acc[i] * inv;
}

// Fused last-dim softmax with a per-head attention sink (gpt-oss).
// Replaces the ~9-op composition (max_keepdim + broadcast_maximum + sub + exp +
// sub + exp + sum_keepdim + add + div) with one block-per-row kernel: block-
// reduce the row max, fold the sink into the stabilizer, block-reduce Σexp, fold
// exp(sink-m) into the denominator, then write the normalized weights.
// Rows are the leading dims of the [b, n_head, q, kv] scores (row = b*nh*q index);
// the per-row head is (row / q) % n_head. Masked entries arrive as -inf and
// exp() them to 0 naturally. Uses expf (not __expf) to match the reference
// precision so greedy argmax is unchanged.
extern "C" __global__ void fused_softmax_sinks_f32(
    const float* __restrict__ scores,   // [rows, kv] (raw Q.Kᵀ, optionally +mask)
    const float* __restrict__ sinks,    // [n_head]
    float* __restrict__ out,            // [rows, kv]
    const int kv,
    const int q,
    const int n_head,
    const float scale                   // applied to scores (not the sink)
) {
    int row = blockIdx.x;
    extern __shared__ float sdata_sm[];
    int offset = row * kv;
    int head = (row / q) % n_head;
    float sink = sinks[head];

    // The scale folds in here (scale.scores vs the raw learned sink). A masked
    // entry is -inf, and scale.(-inf) = -inf, so the mask survives the fold.
    // 1) row max (-FLT_MAX init; every query row has >=1 visible key, and NVRTC
    // has no INFINITY macro without headers)
    float tmax = -3.402823466e+38f;
    for (int i = threadIdx.x; i < kv; i += blockDim.x) {
        tmax = fmaxf(tmax, scale * scores[offset + i]);
    }
    sdata_sm[threadIdx.x] = tmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_sm[threadIdx.x] = fmaxf(sdata_sm[threadIdx.x], sdata_sm[threadIdx.x + s]);
        __syncthreads();
    }
    float m = fmaxf(sdata_sm[0], sink);
    __syncthreads();

    // 2) Σ exp(scale.scores - m)
    float tsum = 0.0f;
    for (int i = threadIdx.x; i < kv; i += blockDim.x) {
        tsum += expf(scale * scores[offset + i] - m);
    }
    sdata_sm[threadIdx.x] = tsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_sm[threadIdx.x] += sdata_sm[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / (sdata_sm[0] + expf(sink - m));

    // 3) normalize
    for (int i = threadIdx.x; i < kv; i += blockDim.x) {
        out[offset + i] = expf(scale * scores[offset + i] - m) * inv;
    }
}
