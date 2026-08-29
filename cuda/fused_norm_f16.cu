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

extern "C" void loken_fused_rmsnorm_f16(
    const void* x, const void* res, const float* weight,
    void* norm_out, void* sum_out, int rows, int cols, float eps, cudaStream_t stream
) {
    const int block = 256; // 8 warps; dynamic shmem caches the F32 input row
    const int shmem = cols * (int)sizeof(float); // <= ~12KB for these hidden sizes
    loken_fused_rmsnorm_f16_kernel<<<rows, block, shmem, stream>>>(
        (const __half*)x, (const __half*)res, weight,
        (__half*)norm_out, (__half*)sum_out, rows, cols, eps);
}

// F32-out RMSNorm variant: same math as loken_fused_rmsnorm_f16 (no
// residual/sum path) but stores f32(f16(v)) - BIT-IDENTICAL to the F16 kernel's
// output followed by a to_dtype(F32) cast launch, which this replaces. For
// consumers that need the norm in F32 (lfm2 MoE FFN input: router GEMV +
// expert q8_1 quantize) - was 38 cast launches/token on lfm2 decode.
extern "C" __global__ void loken_fused_rmsnorm_f16_out_f32_kernel(
    const __half* __restrict__ x,
    const float*  __restrict__ weight,
    float* __restrict__ norm_out,
    int rows, int cols, float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const __half* xr = x + (size_t)row * cols;
    float* nor = norm_out + (size_t)row * cols;
    extern __shared__ float cache[];
    float local = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        const float sf = __half2float(xr[j]);
        cache[j] = sf;
        local += sf * sf;
    }
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
        // round through F16 so downstream sees exactly what the F16 kernel +
        // cast produced (greedy-argmax stability across 38 layers).
        nor[j] = __half2float(__float2half(cache[j] / denom * weight[j]));
    }
}

extern "C" void loken_fused_rmsnorm_f16_out_f32(
    const void* x, const float* weight, void* norm_out,
    int rows, int cols, float eps, cudaStream_t stream
) {
    const int block = 256;
    const int shmem = cols * (int)sizeof(float);
    loken_fused_rmsnorm_f16_out_f32_kernel<<<rows, block, shmem, stream>>>(
        (const __half*)x, weight, (float*)norm_out, rows, cols, eps);
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

// out_f16 = f16(a_f32 + f32(b_f16)). Collapses the a.broadcast_add(b.to_f32())
// .to_f16() chain (cast+add+cast = 3 dispatches -> 1). All F32 math -> bit-identical.
// nemotron MoE shared-expert merge into the routed (F32) output, written F16.
__global__ void loken_add_to_f16_kernel(
    const float* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, long n
) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(a[i] + __half2float(b[i]));
}

extern "C" void loken_add_to_f16(const float* a, const void* b, void* out, long n, cudaStream_t stream) {
    const int block = 256;
    const long grid = (n + block - 1) / block;
    loken_add_to_f16_kernel<<<grid, block, 0, stream>>>(a, (const __half*)b, (__half*)out, n);
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

// Device-position variant: cos_full/sin_full are the FULL [max_seq, rope_dim/2]
// tables; the position is read from *pos_dev so a captured graph picks the right
// rope angle on replay (instead of a host-narrowed slice that freezes). Position
// for query s is (*pos_dev + s). Otherwise identical/bit-exact to the above.
extern "C" __global__ void loken_neox_rope_devpos_f16_kernel(
    const __half* __restrict__ x, const __half* __restrict__ cos_full, const __half* __restrict__ sin_full,
    const int* __restrict__ pos_dev, __half* __restrict__ out, int outer, int seq, int hd, int rope_dim
) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)outer * hd;
    if (idx >= total) return;
    const int d = (int)(idx % hd);
    const long row = idx / hd;
    const int s = (int)(row % seq);
    if (d >= rope_dim) { out[idx] = x[idx]; return; }
    const int half = rope_dim >> 1;
    const int i = (d < half) ? d : (d - half);
    const long base = row * hd;
    const int gpos = *pos_dev + s;                      // device-resident position
    const __half* cr = cos_full + (long)gpos * half;
    const __half* sr = sin_full + (long)gpos * half;
    const float x0 = __half2float(x[base + i]);
    const float x1 = __half2float(x[base + i + half]);
    const float c = __half2float(cr[i]);
    const float sn = __half2float(sr[i]);
    if (d < half) {
        const float a = __half2float(__float2half(x0 * c));
        const float b = __half2float(__float2half(x1 * sn));
        out[idx] = __float2half(a - b);
    } else {
        const float a = __half2float(__float2half(x0 * sn));
        const float b = __half2float(__float2half(x1 * c));
        out[idx] = __float2half(a + b);
    }
}

extern "C" void loken_neox_rope_devpos_f16(
    const void* x, const void* cos_full, const void* sin_full, const int* pos_dev, void* out,
    int outer, int seq, int hd, int rope_dim, cudaStream_t stream
) {
    const int block = 256;
    const long total = (long)outer * hd;
    const long grid = (total + block - 1) / block;
    loken_neox_rope_devpos_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)x, (const __half*)cos_full, (const __half*)sin_full, pos_dev, (__half*)out,
        outer, seq, hd, rope_dim);
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
