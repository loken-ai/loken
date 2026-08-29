// Staging a KV page: f32 or f16 in, q8_0 or q4_0 blocks out, laid out the way the attention
// kernels below read them back.

// Paired K+V quantize at REPLAY-time slot (graph-capture twin of
// `quantize_q8_0_kv_paired`). gridDim.z selects the side (0 = K, 1 = V);
// the destination slot is read from `slot_dev` when the captured graph
// replays, exactly like `quantize_q8_0_dev_slot`. One launch per layer
// instead of two - on a ~17-kernel/layer decode graph the saved node and
// its inter-node gap are a real slice of the per-token budget.
extern "C" __global__ void quantize_q8_0_kv_paired_dev_slot(
    const float * __restrict__ xk,
    const float * __restrict__ xv,
    void  * __restrict__ vyk,
    void  * __restrict__ vyv,
    const int32_t * __restrict__ slot_dev,
    const int kx,
    const int kx_padded,
    const int num_blocks_per_token
) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) return;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;

    const float * __restrict__ x = (blockIdx.z == 0) ? xk : xv;
    void * __restrict__ vy       = (blockIdx.z == 0) ? vyk : vyv;

    const int slot = slot_dev[0];
    block_q8_0 * y = (block_q8_0 *) vy + (size_t)slot * (size_t)num_blocks_per_token;

    const int ib  = i_padded / QK8_0;
    const int iqs = i_padded % QK8_0;

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);
    amax = warp_max(amax);

    const float d  = amax / 127;
    const float id = amax == 0.0f ? 0.0f : 1.0f / d;
    const int8_t q = roundf(xi * id);

    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].d = d;
    }
}

// Paired K+V quantize at host-given byte offset. Equivalent to two back-to-back
// `quantize_q8_0_f32_into_offset` calls but launched as one kernel - saves 1
// launch per layer.
//
// gridDim.z selects which side runs (0 = K, 1 = V); within a side the kernel
// is identical to the single-side `quantize_q8_0` above. Source layout is one
// row of `kx` F32 elements (single-token decode); destination is the same
// `dst_byte_offset` in both K and V buffers (callers must guarantee both
// buffers are equally sized).
extern "C" __global__ void quantize_q8_0_kv_paired(
    const float * __restrict__ xk,
    const float * __restrict__ xv,
    void  * __restrict__ vyk,
    void  * __restrict__ vyv,
    const int kx,
    const int kx_padded,
    const int dst_byte_offset
) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) return;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;

    const float * x   = (blockIdx.z == 0) ? xk  : xv;
    char        * vy0 = (blockIdx.z == 0) ? (char*)vyk : (char*)vyv;
    block_q8_0  * y   = (block_q8_0 *)(vy0 + dst_byte_offset);

    const int ib  = i_padded / QK8_0;
    const int iqs = i_padded % QK8_0;

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);
    amax = warp_max(amax);

    const float d  = amax / 127;
    const float id = amax == 0.0f ? 0.0f : 1.0f / d;
    const int8_t q = roundf(xi * id);

    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].d = d;
    }
}

// GPU-native Q4_0 quantizer. Block layout: 1 half scale + 16 bytes qs (two
// 4-bit nibbles per byte), 32 input elements per block. Follows llama.cpp's
// quantize_row_q4_0_ref: the scale is signed ("max / -8") so the dequantize
// step `(nibble - 8) * d` recovers the original sign; the +8 bias lets the
// stored value fit in an unsigned nibble.
extern "C" __global__ void quantize_q4_0(const float * __restrict__ x, void * __restrict__ vy,
                                         const int kx, const int kx_padded) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) return;

    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;

    block_q4_0 * y = (block_q4_0 *) vy;
    const int ib = i_padded / QK4_0;
    const int iqs = i_padded % QK4_0;

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;

    // Warp reduction: find the signed value whose absolute magnitude is
    // largest across the 32 lanes of this block. We can't just take the
    // signed max - the scale must key off the most-extreme magnitude.
    float amax = fabsf(xi);
    float mval = xi;
    for (int off = 16; off > 0; off /= 2) {
        float a2 = __shfl_xor_sync(0xffffffff, amax, off);
        float v2 = __shfl_xor_sync(0xffffffff, mval, off);
        if (a2 > amax) { amax = a2; mval = v2; }
    }

    const float d  = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    // +8 bias -> unsigned nibble in [0, 15]. Clamp matches llama.cpp's ref.
    int q_int = (int)(xi * id + 8.5f);
    q_int = q_int < 0 ? 0 : (q_int > 15 ? 15 : q_int);

    // Pack: lane j in [0, 16) stores qs[j] with low nibble from x[j] and
    // high nibble from x[j + 16]; we pull the high value via warp shuffle.
    //
    // All 32 lanes must participate in __shfl_down_sync with the same mask
    // - conditioning the shuffle on `iqs < 16` is undefined behaviour and
    // empirically zeroed every high nibble (positions 16..31 of each block
    // all dequantised to the block's max). Do the shuffle unconditionally,
    // then guard only the write.
    int q_high = __shfl_down_sync(0xffffffff, q_int, 16);
    if (iqs < 16) {
        y[ib].qs[iqs] = (uint8_t)((q_int & 0xF) | ((q_high & 0xF) << 4));
    }

    if (iqs == 0) {
        y[ib].d = __float2half(d);
    }
}

extern "C" __global__ void quantize_q4_0_f16(const half * __restrict__ x, void * __restrict__ vy,
                                              const int kx, const int kx_padded) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) return;

    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;

    block_q4_0 * y = (block_q4_0 *) vy;
    const int ib = i_padded / QK4_0;
    const int iqs = i_padded % QK4_0;

    const float xi = ix < kx ? __half2float(x[iy * kx + ix]) : 0.0f;

    float amax = fabsf(xi);
    float mval = xi;
    for (int off = 16; off > 0; off /= 2) {
        float a2 = __shfl_xor_sync(0xffffffff, amax, off);
        float v2 = __shfl_xor_sync(0xffffffff, mval, off);
        if (a2 > amax) { amax = a2; mval = v2; }
    }

    const float d  = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q_int = (int)(xi * id + 8.5f);
    q_int = q_int < 0 ? 0 : (q_int > 15 ? 15 : q_int);

    // See f32 variant for why the shuffle must be outside the write guard.
    int q_high = __shfl_down_sync(0xffffffff, q_int, 16);
    if (iqs < 16) {
        y[ib].qs[iqs] = (uint8_t)((q_int & 0xF) | ((q_high & 0xF) << 4));
    }

    if (iqs == 0) {
        y[ib].d = __float2half(d);
    }
}

// Small-k attention-score gemv: Q (q8_1) x K^T (q8_0) where K is per-token
// row-major so its "n" axis is the sequence length (potentially huge) and
// its "k" axis is head_dim (small, 64/128/256). The existing
// mul_mat_vec_q*_q8_1 kernels are tuned for large-k LLM weights and
// underperform on this shape by 5-7x at long contexts. This kernel
// parallelises across the large n axis and uses all 32 warp lanes for the
// small dot product.
//
// Launch geometry: blockDim = (32, WARPS_PER_BLOCK), one warp per output
// row. gridDim = (ceil(n_kv / WARPS_PER_BLOCK), b_size, 1).
//
// Layout assumptions:
//   K: block_q8_0[n_kv][HEAD_DIM/32]   (per-token row of quantized blocks)
//   Q: block_q8_1[b_size][HEAD_DIM/32] (same)
//   dst: float[b_size][n_kv]
template <int HEAD_DIM>
static __device__ __forceinline__ float attn_score_row_q8_0_q8_1(
    const block_q8_0 * __restrict__ K_row,
    const block_q8_1 * __restrict__ Q_row,
    int lane
) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be multiple of 32");
    constexpr int HD_BLOCKS = HEAD_DIM / 32;

    if constexpr (HD_BLOCKS == 4) {
        // head_dim=128: 4 blocks x 8 int32 = 32 lanes exactly, no waste.
        const int bi  = lane >> 3;   // 0..3
        const int iqs = lane & 7;    // 0..7
        int kv = four_bytes_unaligned(K_row[bi].qs, iqs);
        int qv = four_bytes(Q_row[bi].qs, iqs);
        int sumi = __dp4a(kv, qv, 0);
        // reduce within block (8 lanes)
        sumi += __shfl_xor_sync(0xffffffff, sumi, 4);
        sumi += __shfl_xor_sync(0xffffffff, sumi, 2);
        sumi += __shfl_xor_sync(0xffffffff, sumi, 1);
        // scale at block head (iqs == 0)
        float contrib = 0.0f;
        if (iqs == 0) {
            float d_k = __half2float(K_row[bi].d);
            float d_q = __half2float(__low2half(Q_row[bi].ds));
            contrib = sumi * d_k * d_q;
        }
        // reduce across the 4 block heads (lanes 0, 8, 16, 24)
        contrib += __shfl_xor_sync(0xffffffff, contrib, 16);
        contrib += __shfl_xor_sync(0xffffffff, contrib, 8);
        return contrib;
    } else if constexpr (HD_BLOCKS == 2) {
        // head_dim=64: 2 blocks x 8 int32 = 16 lanes active, upper half idle.
        const int bi  = lane >> 3;
        const int iqs = lane & 7;
        int sumi = 0;
        if (lane < 16) {
            int kv = four_bytes_unaligned(K_row[bi].qs, iqs);
            int qv = four_bytes(Q_row[bi].qs, iqs);
            sumi = __dp4a(kv, qv, 0);
        }
        sumi += __shfl_xor_sync(0xffffffff, sumi, 4);
        sumi += __shfl_xor_sync(0xffffffff, sumi, 2);
        sumi += __shfl_xor_sync(0xffffffff, sumi, 1);
        float contrib = 0.0f;
        if (lane < 16 && iqs == 0) {
            float d_k = __half2float(K_row[bi].d);
            float d_q = __half2float(__low2half(Q_row[bi].ds));
            contrib = sumi * d_k * d_q;
        }
        contrib += __shfl_xor_sync(0xffffffff, contrib, 8);
        return contrib;
    } else {
        // Generic (head_dim=256, 512, ...): iterate blocks, 8 lanes per block.
        float total = 0.0f;
        #pragma unroll
        for (int bi = 0; bi < HD_BLOCKS; ++bi) {
            int sumi = 0;
            if (lane < 8) {
                int kv = four_bytes_unaligned(K_row[bi].qs, lane);
                int qv = four_bytes(Q_row[bi].qs, lane);
                sumi = __dp4a(kv, qv, 0);
            }
            sumi += __shfl_xor_sync(0xffffffff, sumi, 4);
            sumi += __shfl_xor_sync(0xffffffff, sumi, 2);
            sumi += __shfl_xor_sync(0xffffffff, sumi, 1);
            if (lane == 0) {
                float d_k = __half2float(K_row[bi].d);
                float d_q = __half2float(__low2half(Q_row[bi].ds));
                total += sumi * d_k * d_q;
            }
        }
        return total;
    }
}
