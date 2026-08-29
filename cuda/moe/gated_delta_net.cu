// The fused gated DeltaNet recurrence: ONE launch carries a layer's whole token sequence,
// where the tensor-level form spends ~40 small ops per token in a host-side loop.
//
// Per head h, per token t (g is the PRE-exp decay, one scalar per head):
//   g_val      = exp(g[t,h])
//   kv[col]    = sum_i S[i][col] * k[i]
//   delta[col] = (v[col] - g_val * kv[col]) * beta[t,h]
//   S[i][col]  = g_val * S[i][col] + k[i] * delta[col]
//   o[t,h,col] = (sum_i S[i][col] * q[i]) * scale
//
// What the kernel assumes of its caller: one sequence, GQA already tiled so q, k and v all
// carry H heads, and no state snapshots - the final state is written once. The state is the
// contiguous `[H, vd(col), kd(i)]` layout, so S[i][col] lives at `h*S_v*S_v + col*S_v + i`;
// q and k are [n_tokens, H, S_v] with i contiguous, v the same, g and beta [n_tokens, H].
//
// One warp owns one column of the state: its `rows_per_lane` rows per lane stay in registers
// for the whole sequence, and the two per-token sums close over the warp with `warp_sum`.

#include "gguf.cuh"  // warp_sum(float) - the 32-wide shuffle reduction

template <int S_v>
__global__ void loken_gated_delta_net_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ state_in,
        float * __restrict__ o_out,
        float * __restrict__ state_out,
        int   H,
        int   n_tokens,
        long  sq1, long sq2,   // q/k strides (floats): head, token
        long  sv1, long sv2,   // v strides:            head, token
        long  sg1, long sg2,   // g/beta strides:       head, token
        float scale) {
    const int h    = blockIdx.x;                              // head
    const int col  = blockIdx.z * blockDim.y + threadIdx.y;   // v-dim column (one warp owns it)
    const int lane = threadIdx.x;
    constexpr int warp_size     = WARP_SIZE;                  // the shared statement of it
    constexpr int rows_per_lane = S_v / warp_size;            // 4 for S_v=128

    // S[i][col] for this head/col is contiguous over i at col*S_v.
    const float * cs = state_in + (long) h * S_v * S_v + (long) col * S_v;
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = cs[r * warp_size + lane];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + (long) t * sq2 + (long) h * sq1;
        const float * k_t = k + (long) t * sq2 + (long) h * sq1;
        const float * v_t = v + (long) t * sv2 + (long) h * sv1;
        const long    gbo = (long) t * sg2 + (long) h * sg1;
        const float   gv  = expf(g[gbo]);
        const float   bv  = beta[gbo];

        // The lane's rows of k and q, and its part of kv[col] = sum_i S[i][col] * k[i] - the
        // state row is already in a register, so the sum rides along with the load.
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
            kv_shard += s_shard[r] * k_reg[r];
        }
        const float kv_col = warp_sum(kv_shard);

        const float delta_col = (v_t[col] - gv * kv_col) * bv;

        // S[i][col] = g*S[i][col] + k[i]*delta ; attn[col] = sum_i S[i][col]*q[i]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]    = gv * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = warp_sum(attn_partial);

        if (lane == 0) {
            o_out[(long) t * H * S_v + (long) h * S_v + col] = attn_col * scale;
        }
    }

    // write the final state back
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        state_out[(long) h * S_v * S_v + (long) col * S_v + r * warp_size + lane] = s_shard[r];
    }
}

extern "C" void loken_gated_delta_net(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state_in,
        float * o_out, float * state_out,
        int H, int n_tokens, int s_v,
        long sq1, long sq2, long sv1, long sv2, long sg1, long sg2,
        float scale, cudaStream_t stream) {
    const int num_warps = 4;
    dim3 grid(H, 1, ceil_div(s_v, num_warps));
    dim3 block(WARP_SIZE, num_warps, 1);
    // The head dim is a template parameter - it fixes how many state rows a lane holds - so
    // the launch is written once and the switch names the shapes that instantiate it.
#define LOKEN_GDN_LAUNCH(s)                                                     \
    case s:                                                                     \
        loken_gated_delta_net_kernel<s><<<grid, block, 0, stream>>>(            \
            q, k, v, g, beta, state_in, o_out, state_out, H, n_tokens,          \
            sq1, sq2, sv1, sv2, sg1, sg2, scale);                               \
        break;
    switch (s_v) {
        LOKEN_GDN_LAUNCH(64)
        LOKEN_GDN_LAUNCH(128)
        LOKEN_GDN_LAUNCH(256)
        default:
            break; // unsupported head dim - the caller falls back to the tensor-level loop
    }
#undef LOKEN_GDN_LAUNCH
}
