// BOUNDED PROBE (task: WMMA flash-decode go/no-go). A tiled/vectorized SCALAR
// flash-decode variant to test option (b): "if llama.cpp's FA decode is a better-
// TILED scalar scan (not tensor-core), the lever is a better scalar kernel."
//
// The production split-K kernel (flash_decode_f16.cu) uses ONE warp per
// (head,split) and a full warp-shuffle reduction of the QK dot PER kv token
// (5 __shfl / token) with scalar __half loads. This probe uses the llama.cpp
// "vec" layout instead: each LANE owns a DISTINCT kv token, computes the whole
// head_dim dot in registers with VECTORIZED half2 loads (no per-token cross-lane
// reduction), runs online softmax across its own tokens, then the 32 lanes'
// partial (m,l,acc) are flash-merged once at the end. This trades the per-token
// shuffle for a per-lane serial hd loop + one final 32-way reduction.
//
// Split over KV in the grid-z (same as production) so a batch=1 decode fills the
// GPU. head_dim templated (64/128/256). F16 KV, contiguous innermost.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TP_MAXHD 256

// Combine: one warp per head merges its nsplit partials (lane d owns dims d,d+32,..).
template <int HD>
__global__ void tp_flash_decode_combine_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, __half* __restrict__ out,
    int n_head, int nsplit)
{
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int dpl = HD >> 5;
    const long base = (long)head * nsplit;
    float gm = -INFINITY;
    for (int s = 0; s < nsplit; ++s) gm = fmaxf(gm, part_m[base + s]);
    float l = 0.0f;
    float acc[TP_MAXHD/32];
    #pragma unroll
    for (int t = 0; t < dpl; ++t) acc[t] = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float pm = part_m[base + s];
        if (pm == -INFINITY) continue;
        const float w = __expf(pm - gm);
        l += part_l[base + s] * w;
        const float* pa = part_acc + (base + s) * HD;
        #pragma unroll
        for (int t = 0; t < dpl; ++t) acc[t] += pa[lane + (t << 5)] * w;
    }
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    __half* o = out + (long)head * HD;
    #pragma unroll
    for (int t = 0; t < dpl; ++t) o[lane + (t << 5)] = __float2half(acc[t] * inv);
}

// -- Multi-warp variant: W warps cooperate on ONE (head,split), so a MODERATE
// nsplit still floods the block with threads (higher occupancy for the starved
// low-kv-head shapes) AND the W warp-partials merge in SHARED memory instead of
// the F32-partials HBM round-trip. Each thread = token-per-lane (vectorized).
template <int HD, int W>
__global__ void tp_mw_flash_decode_split_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    int n_head, int n_kv, int kv_len, int nsplit, float scale,
    long k_head_stride, long k_pos_stride, long v_head_stride, long v_pos_stride)
{
    const int head  = blockIdx.x;
    const int split = blockIdx.z;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int tid   = threadIdx.x;         // 0..W*32-1
    const int T     = W * 32;
    const int groups = n_head / n_kv;
    const int kvh    = head / groups;
    const int split_len = (kv_len + nsplit - 1) / nsplit;
    const int j0 = split * split_len;
    const int j1 = min(j0 + split_len, kv_len);
    const long pidx = ((long)head) * nsplit + split;

    const __half* q = Q + (long)head * HD;
    const half2* q2 = reinterpret_cast<const half2*>(q);
    float2 qr[HD/2];
    #pragma unroll
    for (int i = 0; i < HD/2; ++i) qr[i] = __half22float2(q2[i]);
    const __half* kh = K + (long)kvh * k_head_stride;
    const __half* vh = V + (long)kvh * v_head_stride;

    float m = -INFINITY, l = 0.0f;
    float acc[HD];
    #pragma unroll
    for (int d = 0; d < HD; ++d) acc[d] = 0.0f;

    for (int j = j0 + tid; j < j1; j += T) {
        const half2* k2 = reinterpret_cast<const half2*>(kh + (long)j * k_pos_stride);
        float p = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD/2; ++i) { float2 kf = __half22float2(k2[i]); p += qr[i].x*kf.x + qr[i].y*kf.y; }
        float s = scale * p;
        float m_new = fmaxf(m, s);
        float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        float pj = __expf(s - m_new);
        l = l * alpha + pj;
        const half2* v2 = reinterpret_cast<const half2*>(vh + (long)j * v_pos_stride);
        #pragma unroll
        for (int i = 0; i < HD/2; ++i) { float2 vf = __half22float2(v2[i]); acc[2*i]=acc[2*i]*alpha+pj*vf.x; acc[2*i+1]=acc[2*i+1]*alpha+pj*vf.y; }
        m = m_new;
    }
    // Intra-warp flash reduce -> lane 0 of each warp.
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, o);
        float ol = __shfl_down_sync(0xffffffff, l, o);
        float m_new = fmaxf(m, om);
        float a_s = (m==-INFINITY)?0.0f:__expf(m-m_new);
        float a_o = (om==-INFINITY)?0.0f:__expf(om-m_new);
        l = l*a_s + ol*a_o;
        #pragma unroll
        for (int d = 0; d < HD; ++d) { float od = __shfl_down_sync(0xffffffff, acc[d], o); acc[d]=acc[d]*a_s+od*a_o; }
        m = m_new;
    }
    __shared__ float sm_m[W], sm_l[W], sm_acc[W*HD];
    if (lane == 0) { sm_m[warp]=m; sm_l[warp]=l; for (int d=0; d<HD; ++d) sm_acc[warp*HD+d]=acc[d]; }
    __syncthreads();
    if (warp == 0) {
        // warp 0 merges the W partials (serial, W small) with lane d owning dims d,d+32,..
        const int dpl = HD >> 5;
        float gm = -INFINITY;
        #pragma unroll
        for (int w = 0; w < W; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f; float ga[HD/32];
        #pragma unroll
        for (int t=0;t<dpl;++t) ga[t]=0.0f;
        #pragma unroll
        for (int w = 0; w < W; ++w) {
            if (sm_m[w]==-INFINITY) continue;
            float wt = __expf(sm_m[w]-gm);
            gl += sm_l[w]*wt;
            #pragma unroll
            for (int t=0;t<dpl;++t) ga[t] += sm_acc[w*HD + lane + (t<<5)]*wt;
        }
        if (lane == 0) { part_m[pidx]=gm; part_l[pidx]=gl; }
        // store UN-normalized acc (combine divides by global l): write acc*? no  - 
        // partials hold acc weighted to gm; combine re-merges across splits.
        #pragma unroll
        for (int t=0;t<dpl;++t) part_acc[pidx*HD + lane + (t<<5)] = ga[t];
    }
}

extern "C" void tp_mw_flash_decode_split(
    const void* Q, const void* K, const void* V,
    float* part_m, float* part_l, float* part_acc, void* out,
    int n_head, int n_kv, int kv_len, int nsplit, int head_dim, float scale,
    long k_head_stride, long k_pos_stride, long v_head_stride, long v_pos_stride,
    long long stream_i64)
{
    cudaStream_t stream = (cudaStream_t)stream_i64;
    constexpr int W = 4;
    dim3 grid(n_head, 1, nsplit), block(32*W, 1, 1), cgrid(n_head, 1, 1), cb(32,1,1);
    if (head_dim == 64) {
        tp_mw_flash_decode_split_kernel<64,W><<<grid,block,0,stream>>>((const __half*)Q,(const __half*)K,(const __half*)V,part_m,part_l,part_acc,n_head,n_kv,kv_len,nsplit,scale,k_head_stride,k_pos_stride,v_head_stride,v_pos_stride);
        tp_flash_decode_combine_kernel<64><<<cgrid,cb,0,stream>>>(part_m,part_l,part_acc,(__half*)out,n_head,nsplit);
    } else if (head_dim == 128) {
        tp_mw_flash_decode_split_kernel<128,W><<<grid,block,0,stream>>>((const __half*)Q,(const __half*)K,(const __half*)V,part_m,part_l,part_acc,n_head,n_kv,kv_len,nsplit,scale,k_head_stride,k_pos_stride,v_head_stride,v_pos_stride);
        tp_flash_decode_combine_kernel<128><<<cgrid,cb,0,stream>>>(part_m,part_l,part_acc,(__half*)out,n_head,nsplit);
    } else if (head_dim == 256) {
        tp_mw_flash_decode_split_kernel<256,W><<<grid,block,0,stream>>>((const __half*)Q,(const __half*)K,(const __half*)V,part_m,part_l,part_acc,n_head,n_kv,kv_len,nsplit,scale,k_head_stride,k_pos_stride,v_head_stride,v_pos_stride);
        tp_flash_decode_combine_kernel<256><<<cgrid,cb,0,stream>>>(part_m,part_l,part_acc,(__half*)out,n_head,nsplit);
    }
}

