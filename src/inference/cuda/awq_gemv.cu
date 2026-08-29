// AWQ (uniform 4-bit, GEMM format) decode GEMV kernel. Correct-first f32
// activation; mirrors the CPU reference `awq::AwqTensor::gemv` exactly (index
// mapping validated bit-exact vs vLLM `awq_dequantize`).
//
//   y[out] = sum_k x[k] * W[out,k],  W[out,k] = (wq - zq) * scale[g,out]
//
// Layout (AutoAWQ GEMM, Linear(in=K, out=N), pcols = N/8):
//   qweight : [K, pcols]      i32  - 8 output nibbles per i32 (AWQ order)
//   qzeros  : [K/gs, pcols]   i32  - 8 zero nibbles per i32, per group
//   scales  : [K/gs, N]       f16  - per (group, output channel)
//   x       : [K]             f32
//   out     : [N]             f32  (must be PRE-ZEROED - kernel atomicAdds)
//
// Output channel n (0..7 within a packed i32) lives at nibble ORDER[n].
//
// ## Parallelisation: split-K with one thread per packed column
// qweight is K-major, so consecutive threads (consecutive pc) read consecutive
// i32 at a fixed k -> each warp's load is one coalesced 128-byte transaction.
// A single-thread-per-column GEMV exposes only N/8 threads (≈5 blocks -> most of
// the GPU idle, ~16 GB/s). So `blockIdx.y` partitions K into chunks of `KCHUNK`
// (a multiple of the group size): each (pc, k-chunk) block accumulates a partial
// over its K-slice and `atomicAdd`s the 8 outputs. That multiplies live blocks by
// the chunk count while preserving the coalesced pc access. x[k] stays hot in L2
// (tiny vs the weight volume that dominates bandwidth). dp4a / q8_1-quantised
// activation is a later increment.

#include "cuda_fp16.h"

extern "C" __global__ void awq_gemv_f32(
    const int *__restrict__ qweight,   // [K, pcols]
    const int *__restrict__ qzeros,    // [ngroups, pcols]
    const __half *__restrict__ scales, // [ngroups, N]
    const float *__restrict__ x,       // [K]
    float *__restrict__ out,           // [N], pre-zeroed
    const int K, const int N, const int group_size, const int kchunk) {
  const int pcols = N >> 3; // N / 8
  const int pc = blockIdx.x * blockDim.x + threadIdx.x;
  if (pc >= pcols) return;

  const int k0 = blockIdx.y * kchunk;
  if (k0 >= K) return;
  int k1 = k0 + kchunk;
  if (k1 > K) k1 = K;

  // AWQ nibble order: output channel n is at nibble ORDER[n] = {0,4,1,5,2,6,3,7}.
  const int sh0 = 0, sh1 = 16, sh2 = 4, sh3 = 20, sh4 = 8, sh5 = 24, sh6 = 12, sh7 = 28;

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

  int cur_g = -1;
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0, z6 = 0, z7 = 0;

  const int out_base = pc << 3;
#define AWQ_RELOAD(g) do {                                                    \
    cur_g = (g);                                                              \
    const int zp = qzeros[(g) * pcols + pc];                                  \
    z0=(zp>>sh0)&0xF; z1=(zp>>sh1)&0xF; z2=(zp>>sh2)&0xF; z3=(zp>>sh3)&0xF;    \
    z4=(zp>>sh4)&0xF; z5=(zp>>sh5)&0xF; z6=(zp>>sh6)&0xF; z7=(zp>>sh7)&0xF;    \
    const __half *sp = scales + (size_t)(g) * N + out_base;                   \
    s0=__half2float(sp[0]); s1=__half2float(sp[1]); s2=__half2float(sp[2]);   \
    s3=__half2float(sp[3]); s4=__half2float(sp[4]); s5=__half2float(sp[5]);   \
    s6=__half2float(sp[6]); s7=__half2float(sp[7]);                           \
  } while (0)
#define AWQ_ACC(w, xk) do {                                                   \
    acc0 += (float)(((w)>>sh0&0xF)-z0)*s0*(xk); acc1 += (float)(((w)>>sh1&0xF)-z1)*s1*(xk); \
    acc2 += (float)(((w)>>sh2&0xF)-z2)*s2*(xk); acc3 += (float)(((w)>>sh3&0xF)-z3)*s3*(xk); \
    acc4 += (float)(((w)>>sh4&0xF)-z4)*s4*(xk); acc5 += (float)(((w)>>sh5&0xF)-z5)*s5*(xk); \
    acc6 += (float)(((w)>>sh6&0xF)-z6)*s6*(xk); acc7 += (float)(((w)>>sh7&0xF)-z7)*s7*(xk); \
  } while (0)
  int k = k0;
  // 8x unroll: issue 8 independent weight loads before consuming -> 8 outstanding
  // memory requests per thread -> higher memory-level parallelism (latency hiding)
  // for this bandwidth-bound kernel, without changing the coalesced access.
  for (; k + 7 < k1; k += 8) {
    const int w0 = qweight[(size_t)(k + 0) * pcols + pc];
    const int w1 = qweight[(size_t)(k + 1) * pcols + pc];
    const int w2 = qweight[(size_t)(k + 2) * pcols + pc];
    const int w3 = qweight[(size_t)(k + 3) * pcols + pc];
    const int w4 = qweight[(size_t)(k + 4) * pcols + pc];
    const int w5 = qweight[(size_t)(k + 5) * pcols + pc];
    const int w6 = qweight[(size_t)(k + 6) * pcols + pc];
    const int w7 = qweight[(size_t)(k + 7) * pcols + pc];
    const float x0=x[k], x1=x[k+1], x2=x[k+2], x3=x[k+3], x4=x[k+4], x5=x[k+5], x6=x[k+6], x7=x[k+7];
    int g = (k + 0) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w0, x0);
    g = (k + 1) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w1, x1);
    g = (k + 2) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w2, x2);
    g = (k + 3) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w3, x3);
    g = (k + 4) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w4, x4);
    g = (k + 5) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w5, x5);
    g = (k + 6) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w6, x6);
    g = (k + 7) / group_size; if (g != cur_g) AWQ_RELOAD(g); AWQ_ACC(w7, x7);
  }
  for (; k < k1; ++k) {
    const int g = k / group_size; if (g != cur_g) AWQ_RELOAD(g);
    AWQ_ACC(qweight[(size_t)k * pcols + pc], x[k]);
  }
#undef AWQ_RELOAD
#undef AWQ_ACC
  // Single block per column (kchunk >= K) -> plain store; else atomic accumulate.
  if (k0 == 0 && k1 >= K) {
    out[out_base + 0] = acc0; out[out_base + 1] = acc1;
    out[out_base + 2] = acc2; out[out_base + 3] = acc3;
    out[out_base + 4] = acc4; out[out_base + 5] = acc5;
    out[out_base + 6] = acc6; out[out_base + 7] = acc7;
  } else {
    atomicAdd(&out[out_base + 0], acc0); atomicAdd(&out[out_base + 1], acc1);
    atomicAdd(&out[out_base + 2], acc2); atomicAdd(&out[out_base + 3], acc3);
    atomicAdd(&out[out_base + 4], acc4); atomicAdd(&out[out_base + 5], acc5);
    atomicAdd(&out[out_base + 6], acc6); atomicAdd(&out[out_base + 7], acc7);
  }
}

// Multi-row repacked AWQ GEMV (mmvq-style). Each warp computes ROWS=4 consecutive
// output rows: the per-iteration x-loads are shared across the 4 rows and the 4
// independent accumulators expose instruction-level parallelism for latency
// hiding (the 1-row-per-warp version above stalls on the serial reduce + single
// acc chain -> only 386 GB/s). Output-major layout -> NO atomics, NO pre-zeroed
// output (direct store), NO split-K - the per-call memset + atomic-finalize that
// throttle the K-major split-K kernel in the real forward are gone.
//   blockDim=(32, WPB); grid.x = ceil(N / (WPB*4)).
extern "C" __global__ void awq_gemv_repacked_mr4_f32(
    const int *__restrict__ qw_t,           // [N, K/8]  i32 (k-order, 8 w/i32)
    const __half *__restrict__ scales_t,    // [N, K/gs] f16
    const unsigned char *__restrict__ zeros_t, // [N, K/gs] u8 (unpacked 4-bit zero)
    const float *__restrict__ x,            // [K] f32
    float *__restrict__ out,                // [N] f32
    const int N, const int K, const int group_size) {
  const int lane = threadIdx.x;                            // 0..31
  const int warp_global = blockIdx.x * blockDim.y + threadIdx.y;
  const int n0 = warp_global * 4;                          // first of 4 rows
  if (n0 >= N) return;
  const int kdiv8 = K >> 3;
  const int ng = K / group_size;
  const int nrows = (N - n0) < 4 ? (N - n0) : 4;
  const int *qw0 = qw_t + (size_t)n0 * kdiv8;              // row n0 base
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (int i = lane; i < kdiv8; i += 32) {
    const int k0 = i << 3;
    const int g = k0 / group_size;
    const float xv0=x[k0],   xv1=x[k0+1], xv2=x[k0+2], xv3=x[k0+3];
    const float xv4=x[k0+4], xv5=x[k0+5], xv6=x[k0+6], xv7=x[k0+7];
#define MR_ROW(r, accr) do {                                                     \
    if ((r) < nrows) {                                                           \
      const int w = qw0[(size_t)(r) * kdiv8 + i];                               \
      const float sc = __half2float(scales_t[(size_t)(n0 + (r)) * ng + g]);     \
      const float zq = (float)(int)zeros_t[(size_t)(n0 + (r)) * ng + g];        \
      float a = ((float)((w)&0xF)-zq)*xv0 + ((float)(((w)>>4)&0xF)-zq)*xv1      \
              + ((float)(((w)>>8)&0xF)-zq)*xv2 + ((float)(((w)>>12)&0xF)-zq)*xv3\
              + ((float)(((w)>>16)&0xF)-zq)*xv4 + ((float)(((w)>>20)&0xF)-zq)*xv5\
              + ((float)(((w)>>24)&0xF)-zq)*xv6 + ((float)(((w)>>28)&0xF)-zq)*xv7;\
      accr += a * sc;                                                            \
    } } while (0)
    MR_ROW(0, acc0); MR_ROW(1, acc1); MR_ROW(2, acc2); MR_ROW(3, acc3);
#undef MR_ROW
  }
#define MR_RED(r, accr) do {                                                     \
    float a = (accr);                                                            \
    a += __shfl_down_sync(0xffffffffu, a, 16); a += __shfl_down_sync(0xffffffffu, a, 8); \
    a += __shfl_down_sync(0xffffffffu, a, 4);  a += __shfl_down_sync(0xffffffffu, a, 2); \
    a += __shfl_down_sync(0xffffffffu, a, 1);                                    \
    if (lane == 0 && (r) < nrows) out[n0 + (r)] = a; } while (0)
  MR_RED(0, acc0); MR_RED(1, acc1); MR_RED(2, acc2); MR_RED(3, acc3);
#undef MR_RED
}

// 8-nibble dot for one packed i32 `w` against 8 consecutive activations `xp`,
// with a per-group zero `zq` (scale applied by the caller). Shared by the
// repacked GEMV kernels.
__device__ __forceinline__ float awq_dot8(int w, const float *xp, float zq) {
  return ((float)((w) & 0xF) - zq) * xp[0] + ((float)(((w) >> 4) & 0xF) - zq) * xp[1]
       + ((float)(((w) >> 8) & 0xF) - zq) * xp[2] + ((float)(((w) >> 12) & 0xF) - zq) * xp[3]
       + ((float)(((w) >> 16) & 0xF) - zq) * xp[4] + ((float)(((w) >> 20) & 0xF) - zq) * xp[5]
       + ((float)(((w) >> 24) & 0xF) - zq) * xp[6] + ((float)(((w) >> 28) & 0xF) - zq) * xp[7];
}

// q8_1 activation block (matches llama.cpp's `block_q8_1`): a 32-element
// group quantized to int8 with `ds.x` = scale (amax/127) and `ds.y` = the f32
// SUM of the block's original activations (used for the asymmetric zero term).
typedef struct { __half2 ds; signed char qs[32]; } awq_block_q8_1;

// dp4a AWQ decode GEMV - the mmvq-class kernel: output-major repacked
// weights (k-contiguous int4) dotted against a PER-32-BLOCK q8_1-quantised
// activation via __dp4a (int4-weight x int8-act). Per AWQ group (128 k = 4 q8_1
// blocks) one scale `sc` and zero `zq`; per block: contribution =
//   sc * (d * dp4a(q4, q8) - zq * sum(x))   with d=ds.x, sum(x)=ds.y.
// One warp per output row; lanes stride the 32-wide blocks; warp-reduce. No
// atomics, no memset, no f32 dequant-per-weight (the split-K/mr4 bottleneck).
//   qw_t : [N, K/8]   i32  (k-order, 8 int4 / i32)
//   yq   : [K/32]     q8_1 blocks (the quantised activation for this token)
extern "C" __global__ void awq_gemv_dp4a_f32(
    const int *__restrict__ qw_t,
    const __half *__restrict__ scales_t,        // [N, K/gs] f16
    const unsigned char *__restrict__ zeros_t,  // [N, K/gs] u8
    const awq_block_q8_1 *__restrict__ yq,
    float *__restrict__ out,                     // [N] f32
    const int N, const int K, const int group_size) {
  const int lane = threadIdx.x;                  // 0..31
  const int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) return;
  const int nblk = K >> 5;                       // K / 32 q8_1 blocks
  const int kdiv8 = K >> 3;                       // i32 per row
  const int ng = K / group_size;
  const int *qw_row = qw_t + (size_t)n * kdiv8;
  const __half *sc_row = scales_t + (size_t)n * ng;
  const unsigned char *zr_row = zeros_t + (size_t)n * ng;

  float acc = 0.f;
  for (int b = lane; b < nblk; b += 32) {
    const int i4 = b << 2;                        // first of 4 i32 for this block
    const int g = (b << 5) / group_size;          // group index (128-wide)
    const float sc = __half2float(sc_row[g]);
    const float zq = (float)(int)zr_row[g];
    const awq_block_q8_1 blk = yq[b];
    const float2 ds = __half22float2(blk.ds);     // ds.x=d, ds.y=sum(x)
    const int *qs = (const int *)blk.qs;          // 32 int8 = 8 int32
    int sumi = 0;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int w = qw_row[i4 + jj];
      const int wlo = (w & 0xF) | (((w >> 4) & 0xF) << 8)
                    | (((w >> 8) & 0xF) << 16) | (((w >> 12) & 0xF) << 24);
      const int whi = ((w >> 16) & 0xF) | (((w >> 20) & 0xF) << 8)
                    | (((w >> 24) & 0xF) << 16) | (((w >> 28) & 0xF) << 24);
      sumi = __dp4a(wlo, qs[2 * jj + 0], sumi);
      sumi = __dp4a(whi, qs[2 * jj + 1], sumi);
    }
    acc += sc * (ds.x * (float)sumi - zq * ds.y);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) out[n] = acc;
}

// Read-ahead-unrolled dp4a GEMV: identical math to awq_gemv_dp4a_f32 but
// issues FOUR int4 (16B) weight loads before consuming -> 4 outstanding DRAM
// requests/thread -> latency hiding for short-row projections (q/o,qkv) that drag
// the in-forward aggregate below big-shape peak (673 vs 833 GB/s). qw rows 16B-aligned.
extern "C" __global__ void awq_gemv_dp4a_u4_f32(
    const int *__restrict__ qw_t,
    const __half *__restrict__ scales_t,
    const unsigned char *__restrict__ zeros_t,
    const awq_block_q8_1 *__restrict__ yq,
    float *__restrict__ out,
    const int N, const int K, const int group_size) {
  const int lane = threadIdx.x;
  const int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) return;
  const int nblk = K >> 5;
  const int kdiv8 = K >> 3;
  const int ng = K / group_size;
  const int *qw_row = qw_t + (size_t)n * kdiv8;
  const __half *sc_row = scales_t + (size_t)n * ng;
  const unsigned char *zr_row = zeros_t + (size_t)n * ng;
  const int gs = group_size;
  float acc = 0.f;
#define U4_BLK(bb, wv) do {                                                      \
    const int g = ((bb) << 5) / gs;                                              \
    const float sc = __half2float(sc_row[g]);                                    \
    const float zq = (float)(int)zr_row[g];                                      \
    const awq_block_q8_1 blk = yq[bb];                                           \
    const float2 ds = __half22float2(blk.ds);                                    \
    const int *qs = (const int *)blk.qs;                                         \
    const int ww[4] = {(wv).x, (wv).y, (wv).z, (wv).w};                          \
    int sumi = 0;                                                                \
    _Pragma("unroll")                                                            \
    for (int jj = 0; jj < 4; ++jj) {                                             \
      const int w = ww[jj];                                                      \
      const int wlo = (w & 0xF) | (((w >> 4) & 0xF) << 8)                        \
                    | (((w >> 8) & 0xF) << 16) | (((w >> 12) & 0xF) << 24);      \
      const int whi = ((w >> 16) & 0xF) | (((w >> 20) & 0xF) << 8)               \
                    | (((w >> 24) & 0xF) << 16) | (((w >> 28) & 0xF) << 24);     \
      sumi = __dp4a(wlo, qs[2 * jj + 0], sumi);                                  \
      sumi = __dp4a(whi, qs[2 * jj + 1], sumi);                                  \
    }                                                                            \
    acc += sc * (ds.x * (float)sumi - zq * ds.y);                               \
  } while (0)
  int b = lane;
  for (; b + 96 < nblk; b += 128) {
    const int4 w0 = *(const int4 *)(qw_row + (b << 2));
    const int4 w1 = *(const int4 *)(qw_row + ((b + 32) << 2));
    const int4 w2 = *(const int4 *)(qw_row + ((b + 64) << 2));
    const int4 w3 = *(const int4 *)(qw_row + ((b + 96) << 2));
    U4_BLK(b, w0); U4_BLK(b + 32, w1); U4_BLK(b + 64, w2); U4_BLK(b + 96, w3);
  }
  for (; b < nblk; b += 32) {
    const int4 wv = *(const int4 *)(qw_row + (b << 2));
    U4_BLK(b, wv);
  }
#undef U4_BLK
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) out[n] = acc;
}

// Prefill dequant from the REPACKED (output-major) layout -> dense f16 `[K, N]`
// (row-major, out[k*N + n] = W[n,k]). Lets the AWQ weight keep ONLY the repacked
// tensors (decode uses dp4a over them) and drop the K-major copy -> no 2x VRAM.
// One thread per OUTPUT element (k, n) with CONSECUTIVE THREADS = CONSECUTIVE n
// -> the f16 writes to out[k*N + n] are fully COALESCED (the dominant traffic:
// K*N f16). The qw_t read is N-major so strided across n, but it's 8x smaller
// (K*N/8 i32) and the 8 k-values sharing an i32 hit L2. This is ~mmvq-fast vs
// the old thread-per-(n,i32) version's strided 8-f16 writes (2.7ms/call ->).
// This path is hot: PLD `forward_all` calls it per speculative draft.
extern "C" __global__ void awq_dequant_f16_repacked(
    const int *__restrict__ qw_t,               // [N, K/8] i32 (k-order)
    const __half *__restrict__ scales_t,        // [N, K/gs] f16
    const unsigned char *__restrict__ zeros_t,  // [N, K/gs] u8
    __half *__restrict__ out,                    // [K, N] f16
    const int N, const int K, const int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;  // consecutive threads -> consecutive n
  const int k = blockIdx.y;
  if (n >= N || k >= K) return;
  const int kdiv8 = K >> 3;
  const int ng = K / group_size;
  const int w = qw_t[(size_t)n * kdiv8 + (k >> 3)];
  const int sh = (k & 7) << 2;
  const int g = k / group_size;
  const float sc = __half2float(scales_t[(size_t)n * ng + g]);
  const float zq = (float)(int)zeros_t[(size_t)n * ng + g];
  out[(size_t)k * N + n] = __float2half((float)(((w >> sh) & 0xF) - zq) * sc);
}

// AWQ dequantize -> dense f16 weight `W^T` shaped [K, N] (row-major), i.e.
// out[k*N + out] = W[out,k]. This is exactly the layout a tensor matmul wants
// for `y[.,N] = x[.,K] @ out[K,N]`, so prefill (seq>1) = this dequant + a cuBLAS
// f16 GEMM (the GEMV kernel only handles seq=1). One thread per (k, packed col):
// reads qweight[k,pc] (coalesced across pc), writes 8 consecutive f16 (coalesced).
extern "C" __global__ void awq_dequant_f16(
    const int *__restrict__ qweight,   // [K, pcols]
    const int *__restrict__ qzeros,    // [ngroups, pcols]
    const __half *__restrict__ scales, // [ngroups, N]
    __half *__restrict__ out,          // [K, N]
    const int K, const int N, const int group_size) {
  const int pcols = N >> 3;
  const int pc = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y;
  if (pc >= pcols || k >= K) return;

  const int sh[8] = {0, 16, 4, 20, 8, 24, 12, 28}; // 4*ORDER[n]
  const int g = k / group_size;
  const int w = qweight[(size_t)k * pcols + pc];
  const int zp = qzeros[g * pcols + pc];
  const int out_base = pc << 3;
  const __half *sp = scales + (size_t)g * N + out_base;
  const size_t obase = (size_t)k * N + out_base;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int wq = (w >> sh[n]) & 0xF;
    const int zq = (zp >> sh[n]) & 0xF;
    out[obase + n] = __float2half((float)(wq - zq) * __half2float(sp[n]));
  }
}
