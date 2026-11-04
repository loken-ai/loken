// F16-stream RMSNorm with optional residual add, compiled by nvcc (the NVRTC
// FUSED_CUDA_SRC path is F32-only - cuda_fp16.h breaks its compilation). One
// block per row reduces over `cols`.
//   res != null: sum = f16(f32(x)+f32(res)) (stored = kept residual); norm in = f32(sum)
//   res == null: norm in = f32(x)
//   norm = f16( (in / sqrt(mean(in^2)+eps)) * weight )   (norm in F32, round at end)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float ll_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

extern "C" __global__ void loken_fused_rmsnorm_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ res,      // or nullptr
    const float*  __restrict__ weight,
    __half* __restrict__ norm_out,
    __half* __restrict__ sum_out,        // or nullptr
    int rows, int cols, float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const __half* xr = x + (size_t)row * cols;
    const __half* rr = res ? res + (size_t)row * cols : nullptr;
    __half* nor = norm_out + (size_t)row * cols;
    __half* sor = sum_out ? sum_out + (size_t)row * cols : nullptr;
    // Single global read pass: cache the F32 input in shared mem so the second
    // (normalize) pass reads from shared, not a re-read + re-convert from global.
    // The old kernel was latency-bound reading `cols` F16 twice. cache = cols
    // floats (<= ~12KB for these models; caller guards).
    extern __shared__ float cache[];
    float local = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float sf;
        if (rr) {
            const float s = __half2float(xr[j]) + __half2float(rr[j]);
            const __half sh = __float2half(s);
            sor[j] = sh;
            sf = __half2float(sh);
        } else {
            sf = __half2float(xr[j]);
        }
        cache[j] = sf;
        local += sf * sf;
    }
    // Two-level warp-shuffle reduction (2 __syncthreads vs the old 8-step tree).
    local = ll_warp_sum(local);
    __shared__ float wsum[32];
    const int warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
    if (lane == 0) wsum[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarps) ? wsum[lane] : 0.0f;
        v = ll_warp_sum(v);
        if (lane == 0) wsum[0] = v;
    }
    __syncthreads();
    const float denom = sqrtf(wsum[0] / (float)cols + eps);
    for (int j = tid; j < cols; j += blockDim.x) {
        nor[j] = __float2half(cache[j] / denom * weight[j]);
    }
}

// The same normalisation 8 columns per thread, for rows whose pointers and width allow
// 16-byte accesses. The sum is rounded to f16 and the division is kept, as in the scalar kernel.
extern "C" __global__ void loken_fused_rmsnorm_f16_vec_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ res,
    const float*  __restrict__ weight,
    __half* __restrict__ norm_out,
    __half* __restrict__ sum_out,
    int rows, int cols, float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int nvec = cols >> 3;
    const uint4* xv = (const uint4*)(x + (size_t)row * cols);
    const uint4* rv = res ? (const uint4*)(res + (size_t)row * cols) : nullptr;
    uint4* nv = (uint4*)(norm_out + (size_t)row * cols);
    uint4* sv = sum_out ? (uint4*)(sum_out + (size_t)row * cols) : nullptr;
    const float4* wv = (const float4*)weight;
    extern __shared__ float4 cache4[];
    float local = 0.0f;
    for (int v = tid; v < nvec; v += blockDim.x) {
        uint4 xu = xv[v];
        uint4 ru;
        if (rv) ru = rv[v];
        const __half2* xh = (const __half2*)&xu;
        const __half2* rh = (const __half2*)&ru;
        uint4 su;
        __half2* sh = (__half2*)&su;
        float f[8];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 xf = __half22float2(xh[k]);
            if (rv) {
                const float2 rf = __half22float2(rh[k]);
                sh[k] = __floats2half2_rn(xf.x + rf.x, xf.y + rf.y);
                xf = __half22float2(sh[k]);
            }
            f[2 * k] = xf.x;
            f[2 * k + 1] = xf.y;
            local += xf.x * xf.x + xf.y * xf.y;
        }
        if (rv) sv[v] = su;
        cache4[2 * v] = make_float4(f[0], f[1], f[2], f[3]);
        cache4[2 * v + 1] = make_float4(f[4], f[5], f[6], f[7]);
    }
    local = ll_warp_sum(local);
    __shared__ float wsum[32];
    const int warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
    if (lane == 0) wsum[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? wsum[lane] : 0.0f;
        t = ll_warp_sum(t);
        if (lane == 0) wsum[0] = t;
    }
    __syncthreads();
    const float denom = sqrtf(wsum[0] / (float)cols + eps);
    for (int v = tid; v < nvec; v += blockDim.x) {
        const float4 a = cache4[2 * v], b = cache4[2 * v + 1];
        const float4 wa = wv[2 * v], wb = wv[2 * v + 1];
        uint4 ou;
        __half2* oh = (__half2*)&ou;
        oh[0] = __floats2half2_rn(a.x / denom * wa.x, a.y / denom * wa.y);
        oh[1] = __floats2half2_rn(a.z / denom * wa.z, a.w / denom * wa.w);
        oh[2] = __floats2half2_rn(b.x / denom * wb.x, b.y / denom * wb.y);
        oh[3] = __floats2half2_rn(b.z / denom * wb.z, b.w / denom * wb.w);
        nv[v] = ou;
    }
}

// The same normalisation with one 16-byte vector per thread, held in registers.
// A block has one thread per vector of the row, so a row is read once and written once and
// no row is cached in shared memory. The sum of squares reduces by warp shuffles.
extern "C" __global__ void loken_fused_rmsnorm_f16_regs_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ res,
    const float*  __restrict__ weight,
    __half* __restrict__ norm_out,
    __half* __restrict__ sum_out,
    int rows, int cols, float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int v = threadIdx.x;
    const uint4* xv = (const uint4*)(x + (size_t)row * cols);
    const uint4* rv = res ? (const uint4*)(res + (size_t)row * cols) : nullptr;
    const float4* wv = (const float4*)weight;
    const uint4 xu = xv[v];
    uint4 ru;
    if (rv) ru = rv[v];
    const float4 wa = wv[2 * v], wb = wv[2 * v + 1];
    const __half2* xh = (const __half2*)&xu;
    const __half2* rh = (const __half2*)&ru;
    uint4 su;
    __half2* sh = (__half2*)&su;
    float f[8];
    float local = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 xf = __half22float2(xh[k]);
        if (rv) {
            const float2 rf = __half22float2(rh[k]);
            sh[k] = __floats2half2_rn(xf.x + rf.x, xf.y + rf.y);
            xf = __half22float2(sh[k]);
        }
        f[2 * k] = xf.x;
        f[2 * k + 1] = xf.y;
        local += xf.x * xf.x + xf.y * xf.y;
    }
    if (rv) ((uint4*)(sum_out + (size_t)row * cols))[v] = su;
    local = ll_warp_sum(local);
    __shared__ float wsum[32];
    const int warp = v >> 5, lane = v & 31, nwarps = blockDim.x >> 5;
    if (lane == 0) wsum[warp] = local;
    __syncthreads();
    float t = (lane < nwarps) ? wsum[lane] : 0.0f;
    t = ll_warp_sum(t);
    t = __shfl_sync(0xffffffffu, t, 0);
    const float denom = sqrtf(t / (float)cols + eps);
    uint4 ou;
    __half2* oh = (__half2*)&ou;
    oh[0] = __floats2half2_rn(f[0] / denom * wa.x, f[1] / denom * wa.y);
    oh[1] = __floats2half2_rn(f[2] / denom * wa.z, f[3] / denom * wa.w);
    oh[2] = __floats2half2_rn(f[4] / denom * wb.x, f[5] / denom * wb.y);
    oh[3] = __floats2half2_rn(f[6] / denom * wb.z, f[7] / denom * wb.w);
    ((uint4*)(norm_out + (size_t)row * cols))[v] = ou;
}

// Launches the register kernel when the row is one vector per thread of a whole number of warps
// and the pointers allow 16-byte accesses. Returns 0 when launched, 1 when it declined.
extern "C" int loken_fused_rmsnorm_f16_regs(
    const void* x, const void* res, const float* weight,
    void* norm_out, void* sum_out, int rows, int cols, float eps, cudaStream_t stream
) {
    const int nvec = cols >> 3;
    const bool aligned = (cols & 7) == 0 &&
        (((size_t)x | (size_t)res | (size_t)weight | (size_t)norm_out | (size_t)sum_out) & 15) == 0;
    if (!aligned || nvec % 32 != 0 || nvec > 1024) return 1;
    loken_fused_rmsnorm_f16_regs_kernel<<<rows, nvec, 0, stream>>>(
        (const __half*)x, (const __half*)res, weight,
        (__half*)norm_out, (__half*)sum_out, rows, cols, eps);
    return 0;
}

extern "C" void loken_fused_rmsnorm_f16(
    const void* x, const void* res, const float* weight,
    void* norm_out, void* sum_out, int rows, int cols, float eps, cudaStream_t stream
) {
    const int block = 256; // 8 warps; dynamic shmem caches the F32 input row
    const int shmem = cols * (int)sizeof(float); // <= ~12KB for these hidden sizes
    const bool aligned = (cols & 7) == 0 &&
        (((size_t)x | (size_t)res | (size_t)weight | (size_t)norm_out | (size_t)sum_out) & 15) == 0;
    if (aligned) {
        loken_fused_rmsnorm_f16_vec_kernel<<<rows, block, shmem, stream>>>(
            (const __half*)x, (const __half*)res, weight,
            (__half*)norm_out, (__half*)sum_out, rows, cols, eps);
        return;
    }
    loken_fused_rmsnorm_f16_kernel<<<rows, block, shmem, stream>>>(
        (const __half*)x, (const __half*)res, weight,
        (__half*)norm_out, (__half*)sum_out, rows, cols, eps);
}

// out = silu(f32(g)) * f32(u), F16 in/out, F32 internal. Collapses the
// chain silu(g.to_f32).u.to_f32 -> f16 (cast+silu+mul+cast = 4 dispatches) into
// one FFI launch. F32-internal silu so it is bit-identical to the F32 reference
// (matters for greedy-argmax stability over many layers). n = total elements.
__global__ void loken_silu_mul_f16_kernel(
    const __half* __restrict__ g, const __half* __restrict__ u,
    __half* __restrict__ out, long n
) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gf = __half2float(g[i]);
    const float uf = __half2float(u[i]);
    const float y = (gf / (1.0f + expf(-gf))) * uf;   // silu(gf) * uf in F32
    out[i] = __float2half(y);
}

extern "C" void loken_silu_mul_f16(
    const void* g, const void* u, void* out, long n, cudaStream_t stream
) {
    const int block = 256;
    const long grid = (n + block - 1) / block;
    loken_silu_mul_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)g, (const __half*)u, (__half*)out, n);
}

// out[r, c] = silu(f32(gu[r, c])) * f32(gu[r, f + c]) for a fused gate+up buffer gu of
// rows x 2f F16, out rows x f. Same F32 arithmetic and single rounding as
// loken_silu_mul_f16_kernel. f is even: a thread takes 2 adjacent columns by one half2 access,
// and gu and out are 4-byte aligned. n2 = rows * f / 2 pairs.
__global__ void loken_silu_mul_gu_f16_kernel(
    const __half* __restrict__ gu, __half* __restrict__ out, long f, long n2
) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    const long half_f = f >> 1;
    const long row = i / half_f;
    const long c2 = i - row * half_f;
    const __half2* gp = (const __half2*)(gu + row * 2 * f) + c2;
    const __half2* up = (const __half2*)(gu + row * 2 * f + f) + c2;
    const float2 gf = __half22float2(*gp);
    const float2 uf = __half22float2(*up);
    const float y0 = (gf.x / (1.0f + expf(-gf.x))) * uf.x;
    const float y1 = (gf.y / (1.0f + expf(-gf.y))) * uf.y;
    ((__half2*)out)[i] = __halves2half2(__float2half(y0), __float2half(y1));
}

extern "C" void loken_silu_mul_gu_f16(
    const void* gu, void* out, long f, long rows, cudaStream_t stream
) {
    const long n2 = rows * (f >> 1);
    if (n2 <= 0) return;
    const int block = 256;
    const long grid = (n2 + block - 1) / block;
    loken_silu_mul_gu_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)gu, (__half*)out, f, n2);
}

// The same activation 8 columns per thread, by 16-byte accesses, with the exponential and the
// division of the fast math units. The result differs from the kernel above in the last bit of
// some values.
__global__ void loken_silu_mul_gu_vec_f16_kernel(
    const __half* __restrict__ gu, __half* __restrict__ out, long f, long n8
) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const long per_row = f >> 3;
    const long row = i / per_row;
    const long c8 = i - row * per_row;
    const uint4 gv = ((const uint4*)(gu + row * 2 * f))[c8];
    const uint4 uv = ((const uint4*)(gu + row * 2 * f + f))[c8];
    const __half2* gh = (const __half2*)&gv;
    const __half2* uh = (const __half2*)&uv;
    uint4 ov;
    __half2* oh = (__half2*)&ov;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 g = __half22float2(gh[k]);
        const float2 u = __half22float2(uh[k]);
        const float y0 = __fdividef(g.x, 1.0f + __expf(-g.x)) * u.x;
        const float y1 = __fdividef(g.y, 1.0f + __expf(-g.y)) * u.y;
        oh[k] = __floats2half2_rn(y0, y1);
    }
    ((uint4*)out)[i] = ov;
}

// Returns 0 when launched, 1 when f is not a multiple of 8 or a pointer is not 16-byte aligned.
extern "C" int loken_silu_mul_gu_vec_f16(
    const void* gu, void* out, long f, long rows, cudaStream_t stream
) {
    if (rows <= 0 || (f & 7) != 0 || (((size_t)gu | (size_t)out) & 15) != 0) return 1;
    const long n8 = rows * (f >> 3);
    const int block = 256;
    const long grid = (n8 + block - 1) / block;
    loken_silu_mul_gu_vec_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)gu, (__half*)out, f, n8);
    return 0;
}

// out = f32(y) . sigmoid(f32(g)) in one F16 launch: an output gate over a head's rows, the
// sigmoid in F32 as the tensor ops compute it. n = total elements.
__global__ void loken_sigmoid_mul_f16_kernel(
    const __half* __restrict__ g, const __half* __restrict__ y,
    __half* __restrict__ out, long n
) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gf = __half2float(g[i]);
    const float yf = __half2float(y[i]);
    out[i] = __float2half(yf * (1.0f / (1.0f + expf(-gf))));
}

extern "C" void loken_sigmoid_mul_f16(
    const void* g, const void* y, void* out, long n, cudaStream_t stream
) {
    const int block = 256;
    const long grid = (n + block - 1) / block;
    loken_sigmoid_mul_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)g, (const __half*)y, (__half*)out, n);
}

// MoE shared-expert output epilogue: out = routed + f32(down).sigmoid(gate_logit),
// gate_logit broadcast per token over `hidden`. Collapses sigmoid + f16->f32 cast +
// broadcast_mul + add (4 tensor-level dispatches) into one launch. All F32 math ->
// bit-identical to the reference. routed/out F32 [n_tokens,hidden]; down F16
// [n_tokens,hidden]; gate_logit F32 [n_tokens] (pre-sigmoid scalar gate).
__global__ void loken_fused_shexp_out_kernel(
    const float* __restrict__ routed, const __half* __restrict__ down,
    const float* __restrict__ gate_logit, float* __restrict__ out,
    int n_tokens, int hidden
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)n_tokens * hidden;
    if (idx >= total) return;
    const int i = (int)(idx / hidden);                 // token row
    const float sg = 1.0f / (1.0f + expf(-gate_logit[i]));
    out[idx] = routed[idx] + __half2float(down[idx]) * sg;
}

extern "C" void loken_fused_shexp_out(
    const float* routed, const void* down, const float* gate_logit,
    float* out, int n_tokens, int hidden, cudaStream_t stream
) {
    const int block = 256;
    const long total = (long)n_tokens * hidden;
    const long grid = (total + block - 1) / block;
    loken_fused_shexp_out_kernel<<<grid, block, 0, stream>>>(
        routed, (const __half*)down, gate_logit, out, n_tokens, hidden);
}

// out = relu(f32(x))^2, F16 in/out, F32 internal -> bit-identical to the
// x.to_f32().relu().sqr().to_f16() chain (cast+relu+sqr+cast = 4 dispatches -> 1).
// nemotron's squared-ReLU FFN activation.
__global__ void loken_relu2_f16_kernel(
    const __half* __restrict__ x, __half* __restrict__ out, long n
) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    v = v > 0.0f ? v : 0.0f;
    out[i] = __float2half(v * v);
}

extern "C" void loken_relu2_f16(const void* x, void* out, long n, cudaStream_t stream) {
    const int block = 256;
    const long grid = (n + block - 1) / block;
    loken_relu2_f16_kernel<<<grid, block, 0, stream>>>((const __half*)x, (__half*)out, n);
}

// out[i,j] = softplus(dt[i,j] + bias[j]) = log(exp(dt+bias) + 1), all F32.
// Bit-identical to ((dt.broadcast_add(bias)).exp() + 1).log() at tensor level (same
// formula, same overflow behaviour). Collapses broadcast_add+exp+add+log -> 1.
// dt/out: [rows, cols] F32; bias: [cols] F32. nemotron Mamba2 dt = softplus(dt+b).
__global__ void loken_softplus_bias_kernel(
    const float* __restrict__ dt, const float* __restrict__ bias,
    float* __restrict__ out, int rows, int cols
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)rows * cols;
    if (idx >= total) return;
    const int j = (int)(idx % cols);
    out[idx] = logf(expf(dt[idx] + bias[j]) + 1.0f);
}

extern "C" void loken_softplus_bias(
    const float* dt, const float* bias, float* out, int rows, int cols, cudaStream_t stream
) {
    const int block = 256;
    const long total = (long)rows * cols;
    const long grid = (total + block - 1) / block;
    loken_softplus_bias_kernel<<<grid, block, 0, stream>>>(dt, bias, out, rows, cols);
}

// Partial NEOX RoPE in one launch (was narrow+contiguous+rope+cat+contiguous, x2
// for q & k). Rotates the first `rope_dim` dims (NEOX: split into two halves of
// rope_dim/2; pair (i, i+rope_dim/2)); dims [rope_dim, hd) pass through. x is
// [outer, hd] contiguous (outer = b*n_head*seq, seq innermost); cos/sin are
// [seq, rope_dim/2]. Each op uses an explicit __float2half so it is BIT-IDENTICAL
// to a tensor-level f16 rope (the `half` crate does each f16 op via f32-convert), which
// keeps greedy argmax unchanged across layers.
__global__ void loken_neox_rope_f16_kernel(
    const __half* __restrict__ x, const __half* __restrict__ cos, const __half* __restrict__ sin,
    __half* __restrict__ out, int outer, int seq, int hd, int rope_dim
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)outer * hd;
    if (idx >= total) return;
    const int d = (int)(idx % hd);
    const long row = idx / hd;            // [b*n_head*seq] flattened, seq innermost
    const int s = (int)(row % seq);
    if (d >= rope_dim) { out[idx] = x[idx]; return; }   // passthrough
    const int half = rope_dim >> 1;
    const int i = (d < half) ? d : (d - half);          // rotary pair index
    const long base = row * hd;
    const __half* cr = cos + (long)s * half;
    const __half* sr = sin + (long)s * half;
    const float x0 = __half2float(x[base + i]);          // first-half element
    const float x1 = __half2float(x[base + i + half]);   // second-half element
    const float c = __half2float(cr[i]);
    const float sn = __half2float(sr[i]);
    if (d < half) {
        // y0 = x0*cos - x1*sin  (per-op f16 rounding to match the tensor-level path)
        const float a = __half2float(__float2half(x0 * c));
        const float b = __half2float(__float2half(x1 * sn));
        out[idx] = __float2half(a - b);
    } else {
        // y1 = x0*sin + x1*cos
        const float a = __half2float(__float2half(x0 * sn));
        const float b = __half2float(__float2half(x1 * c));
        out[idx] = __float2half(a + b);
    }
}

extern "C" void loken_neox_rope_f16(
    const void* x, const void* cos, const void* sin, void* out,
    int outer, int seq, int hd, int rope_dim, cudaStream_t stream
) {
    const int block = 256;
    const long total = (long)outer * hd;
    const long grid = (total + block - 1) / block;
    loken_neox_rope_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)x, (const __half*)cos, (const __half*)sin, (__half*)out, outer, seq, hd, rope_dim);
}

// Paged-decode RoPE (F16): x is [B, heads, hd] (one position per sequence-row),
// cos/sin are [B, half] (per-BATCH, not per-seq - each sequence is at its own
// position). Replaces the ~8-op tensor rope_apply in the CB decode with one
// kernel. `interleaved`: GPT-J pairs (2i,2i+1) for Llama/Mistral, else NeoX pairs
// (i,i+half). Per-op f16 rounding matches paged_attention::rope_apply bit-exact.
extern "C" __global__ void loken_paged_rope_f16_kernel(
    const __half* __restrict__ x, const __half* __restrict__ cos, const __half* __restrict__ sin,
    __half* __restrict__ out, int outer, int heads, int hd, int rope_dim, int interleaved
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)outer * hd;
    if (idx >= total) return;
    const int d = (int)(idx % hd);
    const long row = idx / hd;                 // 0..outer, outer = B*heads
    const int b = (int)(row / heads);          // batch -> cos/sin row
    if (d >= rope_dim) { out[idx] = x[idx]; return; }
    const int half = rope_dim >> 1;
    const long base = row * hd;
    const __half* cr = cos + (long)b * half;
    const __half* sr = sin + (long)b * half;
    if (interleaved) {
        const int i = d >> 1;                  // pair index; d = 2i (even) or 2i+1 (odd)
        const float x0 = __half2float(x[base + 2 * i]);
        const float x1 = __half2float(x[base + 2 * i + 1]);
        const float c = __half2float(cr[i]);
        const float sn = __half2float(sr[i]);
        if ((d & 1) == 0) {                    // out[2i] = x0*cos - x1*sin
            const float a = __half2float(__float2half(x0 * c));
            const float e = __half2float(__float2half(x1 * sn));
            out[idx] = __float2half(a - e);
        } else {                               // out[2i+1] = x0*sin + x1*cos
            const float a = __half2float(__float2half(x0 * sn));
            const float e = __half2float(__float2half(x1 * c));
            out[idx] = __float2half(a + e);
        }
    } else {
        const int i = (d < half) ? d : (d - half);
        const float x0 = __half2float(x[base + i]);
        const float x1 = __half2float(x[base + i + half]);
        const float c = __half2float(cr[i]);
        const float sn = __half2float(sr[i]);
        if (d < half) {                        // out[i] = x0*cos - x1*sin
            const float a = __half2float(__float2half(x0 * c));
            const float e = __half2float(__float2half(x1 * sn));
            out[idx] = __float2half(a - e);
        } else {                               // out[i+half] = x0*sin + x1*cos
            const float a = __half2float(__float2half(x0 * sn));
            const float e = __half2float(__float2half(x1 * c));
            out[idx] = __float2half(a + e);
        }
    }
}

extern "C" void loken_paged_rope_f16(
    const void* x, const void* cos, const void* sin, void* out,
    int outer, int heads, int hd, int rope_dim, int interleaved, cudaStream_t stream
) {
    const int block = 256;
    const long total = (long)outer * hd;
    const long grid = (total + block - 1) / block;
    loken_paged_rope_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)x, (const __half*)cos, (const __half*)sin, (__half*)out,
        outer, heads, hd, rope_dim, interleaved);
}

// Input stage of a prefill attention layer in one launch, one block per (row, head):
//   q head: rope(rmsnorm(q) * q_weight) into q_out
//   k head: rope(rmsnorm(k) * k_weight) into k_out and into the paged store Kp at the row's slot
//   v head: v into v_out and into the paged store Vp at the row's slot
// A null weight skips the norm of that group.
// norm_mode 0: the norm is bit-identical to a cast to f32, fused_rmsnorm_f32, and a cast back
// to f16. The block holds min(256, hd) rounded up to a power of two threads.
// norm_mode 1: the norm is bit-identical to loken_fused_rmsnorm_f16 without a residual on the
// same head: 256 threads, a warp-shuffle sum, then x / sqrtf(mean + eps) * w rounded once to f16.
// norm_vec holds bit 0 for the q norm and bit 1 for the k norm. A set bit sums 8 columns per
// step, as loken_fused_rmsnorm_f16_vec_kernel does. A clear bit sums one column per step.
// The norm output is rounded to f16 before the rotation, and the rotation rounds each operation
// to f16, as the separate launches do.
// The rotation covers the whole head: pair (i, i + hd/2), or (2i, 2i+1) when interleaved.
// q, k and v point at the first element of their group in row 0 and are `*_stride` elements
// apart from row to row. cos and sin are [rows, hd/2]. slot_dev holds one slot per row.
// A null q_head_major writes the rotated q to q_out as f16 [rows, n_head, hd].
// A non-null q_head_major writes it there as f32 [n_head, rows, hd] and leaves q_out unread.
// The f32 values are the f16 results widened, so no rounding differs between the 2 layouts.
extern "C" __global__ void loken_prefill_qk_rope_kv_f16_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    long q_stride, long k_stride, long v_stride,
    const float* __restrict__ q_weight, const float* __restrict__ k_weight,
    float q_eps, float k_eps,
    const __half* __restrict__ cos, const __half* __restrict__ sin,
    __half* __restrict__ q_out, float* __restrict__ q_head_major,
    __half* __restrict__ k_out, __half* __restrict__ v_out,
    __half* __restrict__ Kp, __half* __restrict__ Vp, const int* __restrict__ slot_dev,
    int n_head, int n_kv, int hd, int interleaved, int norm_mode, int norm_vec
) {
    extern __shared__ float row[];
    const long tok = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const __half* src;
    const float* w = nullptr;
    float eps = 0.0f;
    bool vec = false;
    __half* dst = nullptr;
    float* dst_f32 = nullptr;
    __half* store = nullptr;
    bool rotate = true;
    if (h < n_head) {
        src = q + tok * q_stride + (long)h * hd;
        w = q_weight;
        eps = q_eps;
        vec = (norm_vec & 1) != 0;
        if (q_head_major != nullptr) {
            dst_f32 = q_head_major + ((long)h * gridDim.x + tok) * hd;
        } else {
            dst = q_out + (tok * n_head + h) * hd;
        }
    } else if (h < n_head + n_kv) {
        const int head = h - n_head;
        src = k + tok * k_stride + (long)head * hd;
        w = k_weight;
        eps = k_eps;
        vec = (norm_vec & 2) != 0;
        dst = k_out + (tok * n_kv + head) * hd;
        store = Kp + (long)slot_dev[tok] * n_kv * hd + (long)head * hd;
    } else {
        const int head = h - n_head - n_kv;
        src = v + tok * v_stride + (long)head * hd;
        dst = v_out + (tok * n_kv + head) * hd;
        store = Vp + (long)slot_dev[tok] * n_kv * hd + (long)head * hd;
        rotate = false;
    }
    float local = 0.0f;
    for (int d = tid; d < hd; d += blockDim.x) {
        const float xv = __half2float(src[d]);
        row[d] = xv;
        local += xv * xv;
    }
    if (w != nullptr && norm_mode == 1) {
        // The row is complete in shared memory before any thread sums a column it did not load.
        __syncthreads();
        local = 0.0f;
        if (vec) {
            const int nvec = hd >> 3;
            for (int u = tid; u < nvec; u += blockDim.x) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float2 xf;
                    xf.x = row[8 * u + 2 * c];
                    xf.y = row[8 * u + 2 * c + 1];
                    local += xf.x * xf.x + xf.y * xf.y;
                }
            }
        } else {
            for (int j = tid; j < hd; j += blockDim.x) {
                const float sf = row[j];
                local += sf * sf;
            }
        }
        local = ll_warp_sum(local);
        __shared__ float wsum[32];
        const int warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
        if (lane == 0) wsum[warp] = local;
        __syncthreads();
        if (warp == 0) {
            float t = (lane < nwarps) ? wsum[lane] : 0.0f;
            t = ll_warp_sum(t);
            if (lane == 0) wsum[0] = t;
        }
        __syncthreads();
        const float denom = sqrtf(wsum[0] / (float)hd + eps);
        for (int d = tid; d < hd; d += blockDim.x) {
            row[d] = __half2float(__float2half(row[d] / denom * w[d]));
        }
    } else if (w != nullptr) {
        // Same arithmetic as fused_rmsnorm_f32 on the values widened to f32:
        // per-thread strided sums, a halving tree over blockDim.x entries, rsqrtf, x * rms * w.
        float* tree = row + hd;
        tree[tid] = local;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) tree[tid] += tree[tid + s];
            __syncthreads();
        }
        const float rms = rsqrtf(tree[0] / (float)hd + eps);
        for (int d = tid; d < hd; d += blockDim.x) {
            row[d] = __half2float(__float2half(row[d] * rms * w[d]));
        }
    }
    __syncthreads();
    const int half = hd >> 1;
    const __half* cr = cos + tok * half;
    const __half* sr = sin + tok * half;
    for (int d = tid; d < hd; d += blockDim.x) {
        __half o;
        if (!rotate) {
            o = __float2half(row[d]);
        } else {
            // The pair index of lane d, and whether d is the first lane of its pair.
            const int i = interleaved ? (d >> 1) : ((d < half) ? d : (d - half));
            const bool first = interleaved ? ((d & 1) == 0) : (d < half);
            const float x0 = interleaved ? row[2 * i] : row[i];
            const float x1 = interleaved ? row[2 * i + 1] : row[i + half];
            const float c = __half2float(cr[i]);
            const float sn = __half2float(sr[i]);
            if (first) {
                const float a = __half2float(__float2half(x0 * c));
                const float e = __half2float(__float2half(x1 * sn));
                o = __float2half(a - e);
            } else {
                const float a = __half2float(__float2half(x0 * sn));
                const float e = __half2float(__float2half(x1 * c));
                o = __float2half(a + e);
            }
        }
        if (dst_f32 != nullptr) {
            dst_f32[d] = __half2float(o);
        } else {
            dst[d] = o;
        }
        if (store != nullptr) store[d] = o;
    }
}

extern "C" void loken_prefill_qk_rope_kv_f16(
    const void* q, const void* k, const void* v,
    long q_stride, long k_stride, long v_stride,
    const float* q_weight, const float* k_weight, float q_eps, float k_eps,
    const void* cos, const void* sin,
    void* q_out, float* q_head_major, void* k_out, void* v_out, void* Kp, void* Vp,
    const int* slot_dev,
    int rows, int n_head, int n_kv, int hd, int interleaved, int norm_mode, cudaStream_t stream
) {
    // The block size of the row norm kernel: min(256, hd) rounded up to a power of two.
    // Shared memory holds the head as f32 and one partial sum per thread.
    int block = 1;
    while (block < hd && block < 256) block <<= 1;
    // Mode 1 runs the thread count of loken_fused_rmsnorm_f16, whatever the head width.
    if (norm_mode == 1 && (q_weight != nullptr || k_weight != nullptr)) block = 256;
    // A norm sums 8 columns per step where loken_fused_rmsnorm_f16 takes its vector kernel:
    // the width is a multiple of 8 and the weight is 16-byte aligned.
    const int vec_width = (hd & 7) == 0;
    const int norm_vec = (vec_width && (((size_t)q_weight) & 15) == 0 ? 1 : 0) |
                         (vec_width && (((size_t)k_weight) & 15) == 0 ? 2 : 0);
    const int shmem = (hd + block) * (int)sizeof(float);
    dim3 grid(rows, n_head + 2 * n_kv, 1);
    loken_prefill_qk_rope_kv_f16_kernel<<<grid, block, shmem, stream>>>(
        (const __half*)q, (const __half*)k, (const __half*)v, q_stride, k_stride, v_stride,
        q_weight, k_weight, q_eps, k_eps, (const __half*)cos, (const __half*)sin,
        (__half*)q_out, q_head_major, (__half*)k_out, (__half*)v_out, (__half*)Kp, (__half*)Vp,
        slot_dev, n_head, n_kv, hd, interleaved, norm_mode, norm_vec);
}

// The decode input stage with one warp for each head of a row.
//   q head: rope(rmsnorm(q) * q_weight) into q_out
//   k head: rope(rmsnorm(k) * k_weight) into the paged store Kp at the row's slot
//   v head: v into the paged store Vp at the row's slot
// A null weight skips the norm of that group. Each lane holds E consecutive values of the head,
// so hd is 32 * E. The sum of squares reduces by a butterfly of shuffles, and the partner of a
// value in the rotation is read from the lane hd / 2 values away, or from the same lane when the
// pairs are interleaved. The norm output is rounded to f16 before the rotation, and the rotation
// rounds each operation to f16.
template <int E> struct LaneVec;
template <> struct LaneVec<2> { typedef unsigned int type; };
template <> struct LaneVec<4> { typedef uint2 type; };
template <> struct LaneVec<8> { typedef uint4 type; };

template <int E>
__device__ __forceinline__ void ll_load_lane(const __half* p, float (&f)[E]) {
    typename LaneVec<E>::type u = *(const typename LaneVec<E>::type*)p;
    const __half2* h = (const __half2*)&u;
    #pragma unroll
    for (int k = 0; k < E / 2; ++k) {
        const float2 t = __half22float2(h[k]);
        f[2 * k] = t.x;
        f[2 * k + 1] = t.y;
    }
}

template <int E>
__device__ __forceinline__ void ll_store_lane(__half* p, const float (&f)[E]) {
    typename LaneVec<E>::type u;
    __half2* h = (__half2*)&u;
    #pragma unroll
    for (int k = 0; k < E / 2; ++k) h[k] = __floats2half2_rn(f[2 * k], f[2 * k + 1]);
    *(typename LaneVec<E>::type*)p = u;
}

template <int E>
__global__ void loken_decode_qkv_warp_f16_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    long q_stride, long k_stride, long v_stride,
    const float* __restrict__ q_weight, const float* __restrict__ k_weight,
    float q_eps, float k_eps,
    const __half* __restrict__ cos, const __half* __restrict__ sin,
    __half* __restrict__ q_out, __half* __restrict__ Kp, __half* __restrict__ Vp,
    const int* __restrict__ slot_dev, int rows, int n_head, int n_kv, int interleaved
) {
    constexpr int HD = 32 * E;
    const int lane = threadIdx.x & 31;
    const long gw = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int per_row = n_head + 2 * n_kv;
    if (gw >= (long)rows * per_row) return;
    const long tok = gw / per_row;
    const int h = (int)(gw - tok * per_row);
    const int d0 = E * lane;
    const __half* src;
    const float* w = nullptr;
    float eps = 0.0f;
    __half* dst;
    bool rotate = true;
    if (h < n_head) {
        src = q + tok * q_stride + (long)h * HD;
        w = q_weight;
        eps = q_eps;
        dst = q_out + (tok * n_head + h) * HD;
    } else if (h < n_head + n_kv) {
        const int head = h - n_head;
        src = k + tok * k_stride + (long)head * HD;
        w = k_weight;
        eps = k_eps;
        dst = Kp + (long)slot_dev[tok] * n_kv * HD + (long)head * HD;
    } else {
        const int head = h - n_head - n_kv;
        src = v + tok * v_stride + (long)head * HD;
        dst = Vp + (long)slot_dev[tok] * n_kv * HD + (long)head * HD;
        rotate = false;
    }
    float x[E];
    ll_load_lane<E>(src + d0, x);
    if (!rotate) {
        ll_store_lane<E>(dst + d0, x);
        return;
    }
    if (w != nullptr) {
        float local = 0.0f;
        #pragma unroll
        for (int j = 0; j < E; ++j) local += x[j] * x[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) local += __shfl_xor_sync(0xffffffffu, local, o);
        const float denom = sqrtf(local / (float)HD + eps);
        #pragma unroll
        for (int j = 0; j < E; ++j)
            x[j] = __half2float(__float2half(x[j] / denom * w[d0 + j]));
    }
    constexpr int HALF_HD = HD / 2;
    float out[E];
    if (interleaved) {
        float c[E / 2], sn[E / 2];
        const __half* cr = cos + tok * HALF_HD + (d0 >> 1);
        const __half* sr = sin + tok * HALF_HD + (d0 >> 1);
        #pragma unroll
        for (int j = 0; j < E / 2; ++j) {
            c[j] = __half2float(cr[j]);
            sn[j] = __half2float(sr[j]);
            const float x0 = x[2 * j], x1 = x[2 * j + 1];
            out[2 * j] = __half2float(__float2half(x0 * c[j])) -
                         __half2float(__float2half(x1 * sn[j]));
            out[2 * j + 1] = __half2float(__float2half(x0 * sn[j])) +
                             __half2float(__float2half(x1 * c[j]));
        }
    } else {
        float c[E], sn[E], other[E];
        const int i0 = d0 & (HALF_HD - 1);
        ll_load_lane<E>(cos + tok * HALF_HD + i0, c);
        ll_load_lane<E>(sin + tok * HALF_HD + i0, sn);
        #pragma unroll
        for (int j = 0; j < E; ++j) other[j] = __shfl_xor_sync(0xffffffffu, x[j], 16);
        const bool first = d0 < HALF_HD;
        #pragma unroll
        for (int j = 0; j < E; ++j) {
            if (first) {
                out[j] = __half2float(__float2half(x[j] * c[j])) -
                         __half2float(__float2half(other[j] * sn[j]));
            } else {
                out[j] = __half2float(__float2half(other[j] * sn[j])) +
                         __half2float(__float2half(x[j] * c[j]));
            }
        }
    }
    ll_store_lane<E>(dst + d0, out);
}

// Launches the warp kernel when the head is 64, 128 or 256 values and every pointer and stride
// allows the lane's vector access. Returns 0 when launched, 1 when it declined.
extern "C" int loken_decode_qkv_warp_f16(
    const void* q, const void* k, const void* v,
    long q_stride, long k_stride, long v_stride,
    const float* q_weight, const float* k_weight, float q_eps, float k_eps,
    const void* cos, const void* sin,
    void* q_out, void* Kp, void* Vp, const int* slot_dev,
    int rows, int n_head, int n_kv, int hd, int interleaved, cudaStream_t stream
) {
    if (hd != 64 && hd != 128 && hd != 256) return 1;
    const int e = hd / 32;
    const size_t mask = (size_t)(2 * e) - 1;
    if ((((size_t)q | (size_t)k | (size_t)v | (size_t)cos | (size_t)sin | (size_t)q_out |
          (size_t)Kp | (size_t)Vp | (size_t)(q_stride * 2) | (size_t)(k_stride * 2) |
          (size_t)(v_stride * 2)) & mask) != 0) return 1;
    const long warps = (long)rows * (n_head + 2 * n_kv);
    if (warps <= 0) return 1;
    const int block = 128;
    const long grid = (warps * 32 + block - 1) / block;
#define LOKEN_QKV_WARP(E) \
    loken_decode_qkv_warp_f16_kernel<E><<<grid, block, 0, stream>>>( \
        (const __half*)q, (const __half*)k, (const __half*)v, q_stride, k_stride, v_stride, \
        q_weight, k_weight, q_eps, k_eps, (const __half*)cos, (const __half*)sin, \
        (__half*)q_out, (__half*)Kp, (__half*)Vp, slot_dev, rows, n_head, n_kv, interleaved)
    if (e == 2) { LOKEN_QKV_WARP(2); }
    else if (e == 4) { LOKEN_QKV_WARP(4); }
    else { LOKEN_QKV_WARP(8); }
#undef LOKEN_QKV_WARP
    return 0;
}
