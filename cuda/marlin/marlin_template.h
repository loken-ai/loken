/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME loken_w4a16
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "dequant.h"
#include "marlin_mma.h"
#include "scalar_type.hpp"

namespace MARLIN_NAMESPACE_NAME {

// ---------------------------------------------------------------------------------------
// What the tensor cores and the lock need, below the kernel.

/// Load `count` 8x8 fragments of 16-bit values from shared memory, already in the register
/// order `mma.sync` wants.
///
/// `ldmatrix` exists precisely so this is one instruction: the lanes of a warp each hold two
/// elements of an 8x8 tile, in an order that a plain load would have to gather for. The
/// pointer must be shared and 16-byte aligned; the instruction has no unaligned form.
///
/// The three widths are three instructions and not a loop, because the register list is part
/// of the encoding - so the width, the destination list and the operand number of the address
/// are what the three calls below differ in, and the emission itself is written once.
template <int count, loken::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  static_assert(count == 1 || count == 2 || count == 4, "ldmatrix takes 1, 2 or 4 fragments");

#define LOKEN_LDMATRIX(width, dsts, addr, ...)                                             \
  asm volatile("ldmatrix.sync.aligned.m8n8." width ".shared.b16 {" dsts "}, [" addr "];\n" \
               : __VA_ARGS__                                                               \
               : "r"(smem))

  if constexpr (count == 4) {
    LOKEN_LDMATRIX("x4", "%0,%1,%2,%3", "%4", "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]));
  } else if constexpr (count == 2) {
    LOKEN_LDMATRIX("x2", "%0,%1", "%2", "=r"(a[0]), "=r"(a[1]));
  } else {
    LOKEN_LDMATRIX("x1", "%0", "%1", "=r"(a[0]));
  }
#undef LOKEN_LDMATRIX
}

/// Broadcast one of a pair of 16-bit values across a weight fragment, combining it with both
/// halves at once.
///
/// A fragment carries two values per register, so a single `half2` operation covers both, and
/// `i` picks which of the pair - which of the group's two scales, or which half of the zero
/// point - this half of the fragment belongs to. The scale and the zero point are the same
/// broadcast; only the operation differs, so only the operation is written twice.
template <loken::ScalarTypeId type_id, class Op>
__device__ inline void broadcast_into(typename MarlinScalarType<type_id>::FragB& frag_b,
                                      const void* pair, int i, Op combine) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  const auto v =
      MarlinScalarType<type_id>::num2num2(reinterpret_cast<const scalar_t*>(pair)[i]);
#pragma unroll
  for (int h = 0; h < 2; h++) {
    frag_b[h] = combine(frag_b[h], v);
  }
}

/// Multiply a fragment by its group's scale.
template <loken::ScalarTypeId type_id>
__device__ inline void scale(typename MarlinScalarType<type_id>::FragB& frag_b,
                             typename MarlinScalarType<type_id>::FragS& frag_s, int i) {
  broadcast_into<type_id>(frag_b, &frag_s, i,
                          [](half2 x, half2 y) { return __hmul2(x, y); });
}

/// Take the zero point off a fragment.
///
/// It comes off before the scale multiplies: it is an integer in the weight's own units, which
/// is exactly what the dequantised value still is at this point.
template <loken::ScalarTypeId type_id>
__device__ inline void sub_zp(typename MarlinScalarType<type_id>::FragB& frag_b,
                              typename MarlinScalarType<type_id>::scalar_t2& frag_zp, int i) {
  broadcast_into<type_id>(frag_b, &frag_zp, i,
                          [](half2 x, half2 y) { return __hsub2(x, y); });
}

// ---------------------------------------------------------------------------------------
// The per-tile lock.
//
// Several threadblocks compute pieces of the same output tile, and they add into it one at a
// time in slice order. The lock counts how many have finished: a block waits for the count to
// reach its own slice index, adds, then hands it on. Both sides are explicit about visibility  - 
// an acquire on the read so the adds of earlier blocks are visible here, a release fence on
// the write so this block's adds are visible to the next.

/// Spin until the lock's count reaches `count`.
///
/// Each read acquires, so whatever the block before this one stored is visible along with the
/// count that announces it.
__device__ inline void spin_until(const int* lock, int count) {
  int state = -1;
  while (state != count) {
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
  }
}

/// Wait until the lock reaches `count`, then hold it for this threadblock.
///
/// One thread spins and the whole block waits, which is what the barrier at the end is for: it
/// is also where the other threads' reads of the tile are ordered after the acquire above.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    spin_until(lock, count);
  }
  __syncthreads();
}

/// Publish this block's stores and let one more slice through.
__device__ inline void lock_advance(int* lock) {
  const int one = 1;
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(one));
}

/// Release the lock. `reset` puts it back to zero instead, which is what the last slice does so
/// the next launch finds it clean.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x != 0) return;
  if (reset) {
    lock[0] = 0;
    return;
  }
  lock_advance(lock);
}

// The cell this kernel is built for: f16 activations, u4 weights with an integer zero point,
// f16 scales, a four-stage pipeline, and a scale every 128 weights.
constexpr loken::ScalarTypeId a_type_id = loken::kFloat16.id();
constexpr loken::ScalarTypeId b_type_id = loken::kU4.id();
constexpr loken::ScalarTypeId c_type_id = loken::kFloat16.id();
constexpr loken::ScalarTypeId s_type_id = loken::kFloat16.id();
constexpr int stages = 4;
constexpr int group_blocks = 8;

// The five parameters a launch varies: how the output tile is cut, and how wide it is.
// Everything else this kernel took as a parameter is fixed by what the build instantiates  - 
// f16 activations, u4 weights with an integer zero point, f16 scales, a four-stage pipeline
// and a scale every 128 weights - and is a constant in the body, so the signature says which
// kernel this is.
template <const int threads,          // threads in a threadblock
          const int thread_m_blocks,  // 16x16 blocks along m (batch)
          const int thread_n_blocks,  // ... along n (output)
          const int thread_k_blocks,  // ... along k (reduction)
          const bool m_block_size_8   // an 8-row tile, only when thread_m_blocks == 1
          >
/// A four-bit weight against a sixteen-bit activation, `[m, k] x [k, n]`.
///
/// Everything is `int4` - sixteen bytes, one `cp.async` - because that is the unit every load
/// moves: eight activations, or thirty-two weights.
///
/// `C_tmp` is where a block leaves its share of a tile it does not finish, for whichever block
/// writes that tile to add in; `locks` is one word per shared tile, which is how they take
/// turns. Both are only touched by the second of the two parts the work is cut into - see the
/// note below.
__global__ void Marlin(
    const int4* __restrict__ A0,
    const int4* __restrict__ B,
    int4* __restrict__ C0,
    /// Partial sums, in f32, for the tiles that are shared between blocks.
    int4* __restrict__ C_tmp,
    /// One scale per group of weights along k: `[k / group_size, n]`.
    const int4* __restrict__ scales_ptr,
    /// One zero-point per group, packed as many to a word as the weights are: `[k /
    /// group_size, n / pack_factor]`.
    const int4* __restrict__ zp_ptr,
    int prob_m,
    int prob_n,
    int prob_k,
    /// `A`'s row stride, which is `prob_k` when it is contiguous and more when it is a view.
    int lda,
    /// One word per tile that blocks share, for them to take turns through.
    int* locks,
    int max_shared_mem) {
  // ---------------------------------------------------------------------------------------
  // Which piece of the problem this threadblock owns.
  //
  // The output is `parallel * n_tiles` tiles. If there are at least as many tiles as blocks,
  // each block could take whole tiles and never share one - but the tail rarely divides, and a
  // block left holding one tile while its neighbours hold two is a block that finishes early.
  // So the work is cut in two parts:
  //
  //   part 1  whole tiles, `own_tiles` rounds of one per block
  //   part 2  the tiles left over, cut ALONG K into one stripe per block
  //
  // Part 2 is where the cross-block reduction lives, which is why it is kept small: the split
  // takes the leftover modulo the grid, and grows it by a whole grid when that leftover is so
  // thin that the reduction would cost more than the balance it buys.
  //
  // Within part 2 a block's stripe is rounded UP to a whole number of scale groups so that no
  // stripe begins in the middle of one - a group straddling a stripe boundary would have to be
  // read by two blocks.

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  // Turing accumulates in f16: its f32 tensor-core rate for this shape is half its f16 one,
  // and a 128-weight group keeps the partial sums inside f16's range. Later architectures
  // accumulate in f32 and lose nothing for it.
  constexpr bool use_fp16_accum = true;
  #else
  constexpr bool use_fp16_accum = false;
  #endif

  // The activation's carrier and the output's, and what each of them makes a fragment of. They
  // are the same type here - the static assertions below say so - but they are named apart
  // because the scales follow the OUTPUT and the fragments follow the ACTIVATION.
  using Adtype = MarlinScalarType<a_type_id>;
  using Cdtype = MarlinScalarType<c_type_id>;
  using scalar_t = typename Adtype::scalar_t;
  using scalar_t2 = typename Adtype::scalar_t2;
  using scalar_32bit_t = typename Adtype::scalar_32bit_t;
  using c_scalar_t = typename Cdtype::scalar_t;
  using c_scalar_t2 = typename Cdtype::scalar_t2;
  using FragA = typename Adtype::FragA;
  using FragB = typename Adtype::FragB;
  using FragC = typename Adtype::FragC;
  using FragS = typename Cdtype::FragS;
  using FragZP = typename Cdtype::FragZP;

  static constexpr auto a_type = loken::ScalarType::from_id(a_type_id);
  static constexpr auto b_type = loken::ScalarType::from_id(b_type_id);
  static constexpr auto c_type = loken::ScalarType::from_id(c_type_id);
  static constexpr auto s_type = loken::ScalarType::from_id(s_type_id);
  static_assert(std::is_same<scalar_t, half>::value && s_type == loken::kFloat16);
  static_assert(b_type == loken::kU4, "the dequantiser below reads unsigned 4-bit");
  static_assert(std::is_same<scalar_t, c_scalar_t>::value, "the epilogue stores what it accumulates");
  static_assert(thread_m_blocks == 1 || !m_block_size_8, "an 8-row tile is one m block");

  constexpr int m_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);
  constexpr int pack_factor = 32 / b_type.size_bits();
  // How far apart two problems of a tall batch sit, in int4, per unit of row stride.
  constexpr int problem_rows = 16 * thread_m_blocks / 8;

  extern __shared__ int4 sh[];

  // A batch taller than one tile becomes several independent problems of one tile each, which
  // is fewer blocks contending for any single output tile.
  const bool tall = prob_m > m_block_size;
  const int parallel = tall ? prob_m / m_block_size : 1;
  if (tall) {
    prob_m = m_block_size;
  }

  const int k_tiles = prob_k / 16 / thread_k_blocks;
  const int n_tiles = prob_n / 16 / thread_n_blocks;
  const int global_mn_tiles = parallel * n_tiles;
  const int grid = static_cast<int>(gridDim.x);

  // How many tiles go into part 2. With no more tiles than blocks every tile is shared; with
  // more, part 2 takes what part 1 cannot hand out in whole rounds, grown by a further grid
  // whenever that leftover is thin enough to be worth less than the reduction it would cost.
  int shared_tiles = global_mn_tiles;
  if (global_mn_tiles > grid) {
    shared_tiles = global_mn_tiles % grid;
    if (shared_tiles * 3 <= grid) {
      shared_tiles += grid;
    }
  }
  // What is left divides exactly into rounds of one tile per block, and is zero by construction
  // whenever there were no more tiles than blocks.
  int own_tiles = (global_mn_tiles - shared_tiles) / grid;

  // Part 2's work, flattened: `shared_tiles` tiles of `k_tiles` blocks each, cut into one
  // stripe per block. A stripe is rounded up to a whole number of scale groups.
  constexpr int tiles_per_group = group_blocks / thread_k_blocks;
  const int stripe =
      tiles_per_group * div_ceil(div_ceil(k_tiles * shared_tiles, grid), tiles_per_group);

  bool in_part2 = false;           // whether the whole tiles are behind this block
  int slice_row = 0;               // where in the tile's k row this block starts
  int tile_id = static_cast<int>(blockIdx.x);  // the tile, counted across the parallel problems
  int slice_col;                   // the tile's column within one problem
  int slice_iters = k_tiles;       // k-blocks left in this block's piece
  int slice_count = 1;             // how many blocks share this tile
  int slice_idx = 0;               // where this block sits in their order, last first
  int par_id = 0;

  // Which lock guards this tile. With at least a grid's worth of tiles in part 2, no more than
  // `grid` tiles are ever shared at once, so the block index is a sufficient name.
  int locks_off =
      shared_tiles >= grid ? static_cast<int>(blockIdx.x) : stripe * static_cast<int>(blockIdx.x) / k_tiles - 1;

  const int4* A;
  int4* C;
  /// Point A and C at problem `par_id`. A tall batch is several one-tile problems, laid out one
  /// after another in both.
  auto seek_to_problem = [&]() {
    A = A0 + problem_rows * par_id * lda;
    C = C0 + problem_rows * par_id * prob_n;
  };
  seek_to_problem();

  /// A part-2 slice: a run of k-blocks inside one tile, shared with other blocks.
  auto init_part2_slice = [&]() {
    // What is left of this block's stripe from where it now stands, and no further than the end
    // of the tile it stands in.
    const int here = k_tiles * tile_id + slice_row;
    const int left = stripe * static_cast<int>(blockIdx.x + 1) - here;
    const int room = k_tiles - slice_row;
    slice_iters = (tile_id < shared_tiles && left > 0) ? (left < room ? left : room) : 0;
    if (slice_iters == 0) {
      return;
    }

    // How many stripes this tile is cut between, and where this block sits among them. Stripe
    // boundaries fall on multiples of `stripe`, so the tile holds a head - its share of the
    // stripe that started before it - and then whole stripes.
    //
    // The order is counted BACKWARDS, the last slice being index 0, because the last one is the
    // block that ends up holding the total and writes it.
    const int tile_start = k_tiles * tile_id;
    const int first_cut = stripe * div_ceil(tile_start, stripe);
    if (first_cut > tile_start + k_tiles) {
      // No stripe boundary falls inside this tile, so one block holds the whole of it.
      slice_count = 1;
      slice_idx = 0;
    } else {
      const int head = first_cut - tile_start;
      slice_count = div_ceil(k_tiles - head, stripe) + (head > 0 ? 1 : 0);
      const int past_cut = stripe * static_cast<int>(blockIdx.x) - first_cut;
      const int forward = past_cut < 0 ? 0 : past_cut / stripe + (head > 0 ? 1 : 0);
      slice_idx = slice_count - 1 - forward;
    }

    // Past the last column of one problem is the first column of the next.
    if (slice_col == n_tiles) {
      par_id++;
      slice_col = 0;
      seek_to_problem();
    }

    // Which lock guards this tile. Below a grid's worth of shared tiles they come one after
    // another and so do their locks; at or above it the block index already names a tile, so
    // only the block that opens a fresh one moves the name on.
    if (shared_tiles < grid || (slice_count > 1 && slice_idx == slice_count - 1)) {
      locks_off++;
    }
  };

  /// Take the next piece of work, whichever part it comes from.
  auto init_slice = [&]() {
    // The crossing into part 2 happens once: this block's whole tiles are done, and what is
    // left is a stripe inside a tile it shares. Part 2's tiles are the LAST ones of the
    // problem, so the flat index restarts at the head of that run.
    if (!in_part2 && !own_tiles) {
      in_part2 = true;
      const int flat = stripe * static_cast<int>(blockIdx.x);
      tile_id = flat / k_tiles;
      slice_row = flat % k_tiles;
      const int global_tile = tile_id + global_mn_tiles - shared_tiles;
      slice_col = global_tile % n_tiles;
      par_id = global_tile / n_tiles;
      seek_to_problem();
    }
    if (in_part2) {
      init_part2_slice();
      return;
    }
    // A whole tile, this block's alone, the whole k row. Reached only while `own_tiles` is
    // positive, which is what the crossing above tests.
    own_tiles--;
    par_id = tile_id / n_tiles;
    slice_col = tile_id % n_tiles;
    slice_iters = k_tiles;
    seek_to_problem();
  };

  /// Move on once a piece is finished: the next tile in part 1, the next column of the shared
  /// run in part 2. Either way the new piece starts at the top of its tile's k.
  auto next_slice = [&]() {
    slice_row = 0;
    if (in_part2) {
      tile_id++;
      slice_col++;
    } else {
      tile_id += grid;
    }
    init_slice();
  };

  init_slice();

  // ---------------------------------------------------------------------------------------
  // Strides.
  //
  // Everything is counted in `int4` - 16 bytes, the width of one `cp.async` - because that is
  // the unit every load below moves. Eight f16 activations, or thirty-two 4-bit weights, fit
  // in one.
  //
  // Names: `gl` is global memory, `sh` is shared, `rd`/`wr` are the read and write index a
  // thread holds, `_o` steps between tiles and `_i` within one, `_stage` is a whole tile.

  // The activation: `lda / 8` int4 per row, and a tile is `thread_k_blocks` of 16 columns.
  const int a_gl_stride = lda / 8;
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  const int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * m_block_size;
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // The weights: `pack_factor` of them to an int, four ints to an int4. Four bits fit one
  // int4 per thread per pass; a wider weight needs two.
  const int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  constexpr int b_thread_vecs = b_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;
  const int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // The scales: one f16 per output column per group, eight to an int4. A group spans at least
  // as much k as a tile does, so a tile never straddles two and one stage carries one group.
  const int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;

  // The zero points: one per output column per group, packed like the weights.
  const int zp_gl_stride = (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = ((16 * thread_n_blocks) / pack_factor) / 4;

  // The tile, in elements, and the warps across its n dimension.
  constexpr int tb_m = thread_m_blocks * 16;
  constexpr int tb_n = thread_n_blocks * 16;
  constexpr int tb_k = thread_k_blocks * 16;
  constexpr int tb_n_warps = thread_n_blocks / 4;

  // A thread holds four column pairs by two column halves of accumulator. An 8-row tile fills
  // only the first half of each pair, so everything that walks them steps by two.
  constexpr int acc_per_thread = 4 * 2;
  constexpr int acc_step = m_block_size_8 ? 2 : 1;

  // ---------------------------------------------------------------------------------------
  // Where each thread reads and writes.
  //
  // Almost every index below is a lane within a warp and a warp within the block, so both are
  // taken apart once here.
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;

  // The activation is read row-major from global and written the same way to shared.
  const int a_sh_wr =
      a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);

  // Read back in the order `ldmatrix` wants: a lane's row within the fragment, then which int4
  // of that row, then which of the tile's k rows this warp is on. An 8-row tile packs two
  // fragments into a 16-row block, so its lanes wrap at eight rows instead of sixteen.
  constexpr int frag_rows = m_block_size_8 ? 8 : 16;
  const int a_sh_rd = a_sh_stride * (lane % frag_rows) + lane / frag_rows +
                      2 * (warp / tb_n_warps) * b_sh_wr_iters;

  // The weights are already in fragment order from the repack, so the read is contiguous  - 
  // one row of the tile per pass when the block is wide enough to cover it.
  const int b_sh_rd =
      threadIdx.x * b_thread_vecs +
      (threadIdx.x * b_thread_vecs / b_sh_stride) * b_sh_stride * (b_sh_wr_iters - 1);

  // One value per output column per group, so the low threads write and the rest sit out.
  const int group_sh_wr = threadIdx.x;
  const bool s_sh_wr_pred = group_sh_wr < s_sh_stride;
  const bool zp_sh_wr_pred = group_sh_wr < zp_sh_stride;

  /// Point the four global reads at the start of this block's piece: row `slice_row` of the k
  /// dimension, column `slice_col` of the output.
  ///
  /// The prologue and every move to the next piece are the same seek, so it is written once. A
  /// group covers at least a tile of k, so the row divides through and no thread straddles two
  /// groups.
  int a_gl_rd, b_gl_rd, s_gl_rd, zp_gl_rd;
  auto seek_to_slice = [&]() {
    a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o) +
              a_gl_rd_delta_o * slice_row;
    b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride) +
              b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + group_sh_wr;
    zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
               zp_sh_stride * slice_col + group_sh_wr;
  };
  seek_to_slice();

  // A grouped quantisation scales a `half2` tile in column-major order, which is why the read
  // index walks the lane's quarter of a warp rather than its column. The zero points are read
  // the same way - the same eight columns per warp, the same quarter-of-a-warp row - only
  // packed several to an int.
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  const int s_sh_rd = num_col_threads * (warp % tb_n_warps) + lane / num_row_threads;
  const int zp_sh_rd = num_ints_per_thread * s_sh_rd;

  // A thread reads nothing when its row is past the batch, which happens whenever the tile is
  // wider than the problem or the batch is not a whole number of 16-row blocks. Where it puts
  // what it does read is the shuffle below.
  //
  // The activation tile is written row-major and read in fragment order, and those two orders
  // land on the same shared-memory banks unless the rows are shuffled against each other. XOR
  // the row into the offset WITHIN the row and they do not: eight consecutive threads then
  // touch eight different banks on both sides, and each warp still writes one contiguous run.
  //
  // Every loop that indexes shared memory below is fully unrolled, so the shuffle is a
  // compile-time constant at each use - which is why it is tabulated here rather than redone.
  auto swizzle = [&](int i) {
    const int row = i / a_gl_rd_delta_o;
    const int in_row = i % a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (in_row ^ (row % 8));
  };
  bool a_sh_wr_pred[a_sh_wr_iters];
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++) {
    const int at = a_sh_wr_delta * i + a_sh_wr;
    a_sh_wr_pred[i] = at < a_sh_stride * prob_m;
    a_sh_wr_trans[i] = swizzle(at);
  }
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++) {
      a_sh_rd_trans[i][j] = swizzle(2 * i + a_sh_rd_delta_i * j + a_sh_rd);
    }
  }

  // ---------------------------------------------------------------------------------------
  // Shared memory, in one run.
  //
  // The weight tile and the reduction buffer never live at once - the reduction happens after
  // the last tile is consumed - so they share a region sized by whichever is larger.
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  constexpr int sh_size_b_red_max = sh_red_size > sh_b_size ? sh_red_size : sh_b_size;
  constexpr int sh_s_size = stages * s_sh_stride;

  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_zp = sh + sh_size_b_red_max;
  int4* sh_s = sh_zp + (stages * zp_sh_stride);
  int4* sh_a = sh_s + sh_s_size;

  // Registers, double-buffered: the fragments for step k+1 are read while step k multiplies.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];
  int frag_qzp[2][num_ints_per_thread];   // the zero points, as read
  FragZP frag_zp;                         // the zero points, dequantised

  auto zero_accums = [&]() {
    float* acc = reinterpret_cast<float*>(frag_c);
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * acc_per_thread * 4; i++) {
      acc[i] = 0;
    }
  };

  // ---------------------------------------------------------------------------------------
  // The pipeline: one tile arriving while the one before it is read.
  /// Start one stage's worth of loads: the activation tile, the weight tile, and the scales
  /// and zero points if this tile opens a new group.
  ///
  /// Nothing is waited on here. `cp.async` copies global to shared without going through
  /// registers and without blocking, and the fence at the end names this batch so that
  /// `cp_async_wait` can later count how many batches are still outstanding - which is what
  /// keeps a tile arriving while the one before it is being read. The fence is issued even
  /// when `pred` is false, on the wind-down, so that the counting stays right to the end.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        const int from = a_gl_rd + a_gl_rd_delta_i * i + a_gl_rd_delta_o * a_off;
        cp_async4_pred(&a_stage[a_sh_wr_trans[i]], &A[from], a_sh_wr_pred[i]);
      }

      // The weights need no predicate: the repack padded the tensor to whole tiles. One pass of
      // the block covers `per_pass` int4 of a tile row, so a row wider than that takes several,
      // and a row narrower means one pass spans `rows_per_pass` of them.
      constexpr int per_pass = div_ceil(b_sh_stride, threads);
      constexpr int rows_per_pass = div_ceil(threads, b_sh_stride);
      int4* b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters * b_thread_vecs; i++) {
        const int from =
            b_gl_rd + threads * (i % per_pass) + b_gl_stride * rows_per_pass * (i / per_pass);
        cp_async4(&b_stage[threads * i + threadIdx.x], &B[from]);
      }
      b_gl_rd += b_gl_rd_delta_o;

      // A group spans several tiles of k, so its scales and zero points are read once, by the
      // tile that opens it, and the tiles after it reuse what is already in shared memory.
      if (pipe % tiles_per_group == 0) {
        if (s_sh_wr_pred) {
          cp_async4(&sh_s[s_sh_stride * pipe + group_sh_wr], &scales_ptr[s_gl_rd]);
        }
        s_gl_rd += s_gl_stride;

        if (zp_sh_wr_pred) {
          cp_async4(&sh_zp[zp_sh_stride * pipe + group_sh_wr], &zp_ptr[zp_gl_rd]);
        }
        zp_gl_rd += zp_gl_stride;
      }
    }
    cp_async_fence();
  };

  /// How many batches of copies are allowed to still be in flight when a stage is read.
  ///
  /// Two, not zero: the registers are double-buffered, and the next fetch may not be issued
  /// until the shared memory it will overwrite is finished with. Waiting for every batch would
  /// drain the pipe on every step.
  constexpr int in_flight = stages - 2;

  /// Wait until the stage about to be read has landed.
  auto wait_for_stage = [&]() {
    cp_async_wait<in_flight>();
    __syncthreads();
  };

  /// Read step `k`'s fragments out of shared memory into the register buffer it will use.
  ///
  /// The activation goes through `ldmatrix`, which is what the shuffle above was for; the
  /// weights are already in fragment order from the repack, so they are a plain load.
  auto fetch_to_registers = [&](int k, int pipe) {
    const int buf = k % 2;                 // which of the two register buffers this step fills
    const int step = k % b_sh_wr_iters;    // which 16-wide slice of the tile it reads

    const int4* a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      ldsm<m_block_size_8 ? 2 : 4, a_type_id>(frag_a[buf][i], &a_stage[a_sh_rd_trans[step][i]]);
    }

    const int4* b_stage = sh_b + b_sh_stage * pipe + b_sh_stride * step + b_sh_rd;
  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[buf][i] = *reinterpret_cast<const I4*>(&b_stage[i]);
    }
  };

  // One scale group spans `tiles_per_group` tiles of k, so a group is read once every that many
  // pipeline stages, and the second half of a k row reuses what the first half read.
  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    const int pipe = full_pipe % stages;
    if (pipe % tiles_per_group != 0) {
      return;
    }
    if (k % b_sh_wr_iters == 0) {
      const int4* head = sh_s + s_sh_stride * (tiles_per_group * (pipe / tiles_per_group));
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = head[s_sh_rd];
    } else {
      reinterpret_cast<int4*>(&frag_s[1])[0] = reinterpret_cast<int4*>(&frag_s[0])[0];
    }
  };

  // The zero points, on the same schedule as the scales: one read at the head of each group
  // serves every k of it.
  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    const int pipe = full_pipe % stages;
    if (pipe % tiles_per_group == 0 && k % b_sh_wr_iters == 0) {
      const int4* head = sh_zp + zp_sh_stride * (tiles_per_group * (pipe / tiles_per_group));
      const int* words = reinterpret_cast<const int*>(head);
  #pragma unroll
      for (int i = 0; i < num_ints_per_thread; i++) {
        frag_qzp[k % 2][i] = words[zp_sh_rd + i];
      }
    }
  };

  /// Everything step `k` of stage `pipe` needs, in the order the loads want to be issued.
  auto fetch_step_to_registers = [&](int k, int pipe) {
    fetch_to_registers(k, pipe % stages);
    fetch_scales_to_registers(k, pipe);
    fetch_zp_to_registers(k, pipe);
  };

  /// One step of k: unpack, subtract the zero point, scale, multiply.
  ///
  /// The zero point is dequantised only when the group changes - it is the same value for
  /// every k of a group, and unpacking it per step would be work for nothing. The m loop is
  /// innermost so that the dequantisation of the NEXT column pair overlaps this one's
  /// multiplies.
  auto matmul = [&](int k, int pipe) {
    const int buf = k % 2;
    const bool is_new_zp =
        (group_blocks < b_sh_wr_iters || k == 0) && (pipe % tiles_per_group == 0);

    if (is_new_zp) {
      // Four bits each, so both halves of the group's zero points come out of one int.
      const int zp_quant = frag_qzp[buf][0];
      dequant_u4_biased(zp_quant, reinterpret_cast<scalar_32bit_t*>(&frag_zp));
      dequant_u4_biased(zp_quant >> 8, reinterpret_cast<scalar_32bit_t*>(&frag_zp) + 2);
    }

  #pragma unroll
    for (int j = 0; j < 4; j++) {
      // The two column halves this thread holds of the pair: nibbles 0 and 4 of the word, then
      // nibbles 1 and 5, which is the order the repack stored the eight weights in.
      FragB frag_b[2];
      const int b_quant = frag_b_quant[buf][0][j];
      dequant_u4_biased(b_quant, reinterpret_cast<scalar_32bit_t*>(&frag_b[0]));
      dequant_u4_biased(b_quant >> 8, reinterpret_cast<scalar_32bit_t*>(&frag_b[1]));

  #pragma unroll
      for (int h = 0; h < 2; h++) {
        sub_zp<a_type_id>(frag_b[h], frag_zp[j], h);
        scale<a_type_id>(frag_b[h], frag_s[buf][j], h);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          // At eight rows the second accumulator would be half empty, so one transposed
          // instruction takes both column halves instead of two.
          mma_trans<a_type_id, use_fp16_accum>(frag_a[buf][i], frag_b[0], frag_b[1],
                                               frag_c[i][j][0]);
        } else {
  #pragma unroll
          for (int h = 0; h < 2; h++) {
            mma<a_type_id, use_fp16_accum>(frag_a[buf][i], frag_b[h], frag_c[i][j][h]);
          }
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------
  // Folding the partial sums together.
  //
  // A tile is cut across k so that more warps can work on it without making its n dimension
  // unreasonably wide. Several warps therefore hold partial sums of the SAME output element,
  // and those have to meet before anything is written.

  /// Fold the warps of this threadblock together, through shared memory.
  ///
  /// A logarithmic reduction: half the warps write, the other half read and add, and the
  /// halving repeats. The bounds are chosen so no warp writes a value nobody reads - with two
  /// warps, warp 1 writes once and warp 0 reads once, and that is all.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if constexpr (red_off < 1) {
      return;
    }

    constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
    constexpr int red_sh_delta = b_sh_stride_threads;
    const int red_idx = threadIdx.x / b_sh_stride_threads;
    const int red_lane = threadIdx.x % b_sh_stride_threads;
    const int red_sh_rd = red_sh_stride * red_idx + red_lane;
    int4* acc = reinterpret_cast<int4*>(&frag_c);

  #pragma unroll
    for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
      for (int live = red_off; live > 0; live /= 2) {
        if (red_idx >= live && red_idx < 2 * live) {
  #pragma unroll
          for (int j = 0; j < acc_per_thread; j += acc_step) {
            const int mine = red_sh_delta * j + red_sh_rd;
            const int theirs = mine - red_sh_stride * live;
            // The first round has nothing to read back yet: it only writes.
            if (live < red_off) {
              float* here = reinterpret_cast<float*>(&sh_red[mine]);
              float* below = reinterpret_cast<float*>(&sh_red[theirs]);
              float* into = reinterpret_cast<float*>(&acc[acc_per_thread * m_block + j]);
  #pragma unroll
              for (int f = 0; f < 4; f++) {
                into[f] += here[f] + below[f];
              }
            }
            sh_red[theirs] = acc[acc_per_thread * m_block + j];
          }
        }
        __syncthreads();
      }

      // The last round leaves the total in shared memory for warp 0 to take up.
      if (red_idx == 0) {
  #pragma unroll
        for (int j = 0; j < acc_per_thread; j += acc_step) {
          float* here = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
          float* into = reinterpret_cast<float*>(&acc[acc_per_thread * m_block + j]);
  #pragma unroll
          for (int f = 0; f < 4; f++) {
            into[f] += here[f];
          }
        }
      }
      __syncthreads();
    }
  };

  /// Fold this threadblock's tile into the partials of the blocks that share it.
  ///
  /// Through an f32 scratch buffer rather than through `C`, which is f16: the partial sums of
  /// a tile can be much larger than any one of them, and rounding each addition to f16 would
  /// cost accuracy the final result does not have to lose. The blocks take turns under the
  /// lock, so `first` reads nothing and `last` writes nothing back - it holds the total.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int active_threads = 32 * tb_n_warps;
    constexpr int th_size = thread_m_blocks * acc_per_thread * sizeof(float4) / 16;

    // Only the warps that hold accumulators take part; the rest have nothing to add.
    if (threadIdx.x >= active_threads) {
      return;
    }
    int4* mine = C_tmp + locks_off * c_size + threadIdx.x;
    int4* acc = reinterpret_cast<int4*>(&frag_c);

  #pragma unroll
    for (int k = 0; k < th_size; k += acc_step) {
      // Through shared memory, so the global side of it is one int4 per thread either way.
      if (!first) {
        sh_red[threadIdx.x] = mine[active_threads * k];
        float* theirs = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
        float* into = reinterpret_cast<float*>(&acc[k]);
  #pragma unroll
        for (int f = 0; f < 4; f++) {
          into[f] += theirs[f];
        }
      }
      if (!last) {
        mine[active_threads * k] = acc[k];
      }
    }
  };

  /// Write the finished tile out.
  ///
  /// The accumulators are in the instruction's element order, which is not the order `C` wants
  /// its rows in. Going through shared memory turns a scattered global write into a contiguous
  /// one: every thread writes its fragment where the row layout wants it, and then the whole
  /// block reads back rows and stores them. The reduction above ran in fragment order
  /// precisely so that this is the only place the reshuffle happens.
  auto write_result = [&]() {
    // One output row of the tile is this many int4, and one pass over the block covers this
    // many rows of it.
    constexpr int c_row_int4 = 2 * thread_n_blocks;
    constexpr int c_rows_per_pass = threads / c_row_int4;
    constexpr int c_sh_stride = c_row_int4 + 1;  // +1: one spare, against bank conflicts
    constexpr int c_sh_rd_delta = c_sh_stride * c_rows_per_pass;

    const int c_gl_stride = prob_n / 8;
    const int c_gl_wr_delta = c_gl_stride * c_rows_per_pass;
    const int c_gl_wr_end = c_gl_stride * prob_m;
    const int c_gl_wr = c_gl_stride * (threadIdx.x / c_row_int4) + (threadIdx.x % c_row_int4) +
                        c_row_int4 * slice_col;
    const int c_sh_rd = c_sh_stride * (threadIdx.x / c_row_int4) + (threadIdx.x % c_row_int4);

    // Where a lane's fragment lands in the row layout. The 8-row tile holds two column halves
    // per accumulator, so its lanes are spread twice as far apart.
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * (lane % 4 * 2) + lane / 4 + 64 * warp;
    } else {
      c_sh_wr = (4 * c_sh_stride) * (lane / 4) + lane % 4 + 32 * warp;
    }

    /// One accumulator pair into the row layout, at `idx`.
    ///
    /// A 16-row tile puts the pair side by side in one row, which is a single 32-bit store; an
    /// 8-row tile holds two COLUMN halves in the one accumulator, so its two values are eight
    /// rows apart and go down separately.
    auto write = [&](int idx, float c0, float c1) {
      const c_scalar_t2 pair = Cdtype::nums2num2(Cdtype::float2num(c0), Cdtype::float2num(c1));
      if constexpr (m_block_size_8) {
        c_scalar_t* out = reinterpret_cast<c_scalar_t*>(sh_red);
        out[idx] = pair.x;
        out[idx + 8 * c_sh_stride] = pair.y;
      } else {
        reinterpret_cast<c_scalar_t2*>(sh_red)[idx] = pair;
      }
    };

    if (warp < tb_n_warps) {
  #pragma unroll
      for (int mb = 0; mb < thread_m_blocks; mb++) {
  #pragma unroll
        for (int col = 0; col < 4; col++) {
          if constexpr (m_block_size_8) {
            const int wr = c_sh_wr + 16 * col;
            write(wr, frag_c[mb][col][0][0], frag_c[mb][col][0][1]);
            write(wr + 8, frag_c[mb][col][0][2], frag_c[mb][col][0][3]);
          } else {
            // The accumulator holds a 16x8 tile as two halves of four values, and each half
            // covers two row groups eight rows apart. So `half` picks the column pair and
            // `group` the rows, which is the instruction's layout written as the loop it is.
            const int wr = c_sh_wr + 8 * col;
  #pragma unroll
            for (int half = 0; half < 2; half++) {
  #pragma unroll
              for (int group = 0; group < 2; group++) {
                write(wr + (4 * c_sh_stride) * (8 * group) + 4 * half,
                      frag_c[mb][col][half][2 * group],
                      frag_c[mb][col][half][2 * group + 1]);
              }
            }
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    // Read the rows back and store them. A thread's row moves down by a whole pass each time,
    // so once it is past the batch it stays past it.
  #pragma unroll
    for (int pass = 0; pass < div_ceil(tb_m, c_rows_per_pass); pass++) {
      const int at = c_gl_wr + c_gl_wr_delta * pass;
      if (at < c_gl_wr_end) {
        C[at] = sh_red[c_sh_rd + c_sh_rd_delta * pass];
      }
    }
    __syncthreads();
  };

  // ---------------------------------------------------------------------------------------
  // The loop.

  /// How many stages go into flight before the first multiply. Not `stages`: the last one is
  /// what the first pass round the loop below overwrites, so it is fetched there instead.
  constexpr int prefill = stages - 1;

  /// Fill the pipeline, and take the first step's operands out of it.
  auto start_pipes = [&]() {
    zero_accums();
  #pragma unroll
    for (int i = 0; i < prefill; i++) {
      // `i < slice_iters` is the wind-up on a slice shorter than the pipeline. The fence still
      // fires on the empty ones, so the count of outstanding batches stays right.
      fetch_to_shared(i, i, i < slice_iters);
    }
    a_gl_rd += a_gl_rd_delta_o * prefill;
    wait_for_stage();
    fetch_step_to_registers(0, 0);
  };

  while (slice_iters) {
    start_pipes();

    // One round of the pipeline is `stages` tiles of k. A slice longer than that goes round
    // again with the pipeline still full - only the seek advances.
    do {
      // Both loops are fully unrolled so that every shared-memory index is a compile-time
      // constant, and both have even length so the next round starts back at zero.
  #pragma unroll
      for (int stage = 0; stage < stages; stage++) {
  #pragma unroll
        for (int k = 0; k < b_sh_wr_iters; k++) {
          // Read the NEXT step's operands before multiplying this one, so the shared-memory
          // read overlaps the arithmetic instead of preceding it. The last step of a tile reads
          // the first step of the tile after it, which lives one stage on.
          fetch_step_to_registers(k + 1, k == b_sh_wr_iters - 1 ? stage + 1 : stage);

          // Two steps from the end, start the tile that will land `stages - 1` from here and
          // let the pipeline move on by one, so its loads are already in flight when this
          // tile's last step multiplies.
          if (k == b_sh_wr_iters - 2) {
            fetch_to_shared((stage + stages - 1) % stages, stage, slice_iters >= stages);
            wait_for_stage();
          }

          matmul(k, stage);
        }
        if (--slice_iters == 0) {
          break;
        }
      }
      a_gl_rd += a_gl_rd_delta_o * stages;
    } while (slice_iters);

    // This block's piece of the tile is finished.
    if constexpr (use_fp16_accum) {
      // The accumulator was f16 for the multiplies; widen it in place before anything adds to
      // it, so the reduction and the write see f32. Backwards, because the f16 values sit in
      // the low half of the f32 slots they are being written into.
  #pragma unroll
      for (int i = 0; i < thread_m_blocks * acc_per_thread; i++) {
        float* frag_c_part_float = reinterpret_cast<float*>(frag_c) + i * 4;
        scalar_t* frag_c_part_half = reinterpret_cast<scalar_t*>(frag_c_part_float);
  #pragma unroll
        for (int j = 3; j >= 0; j--) {
          frag_c_part_float[j] = Cdtype::num2float(frag_c_part_half[j]);
        }
      }
    }

    cp_async_wait<0>();
    const bool last = slice_idx == slice_count - 1;
    thread_block_reduce();

    if (slice_count > 1) {
      // The blocks sharing this tile add into it one at a time, in slice order: wait for the
      // count to reach this block's index, add, and hand it on.
      int* const lock = &locks[locks_off];
      barrier_acquire(lock, slice_idx);
      global_reduce_fp32(slice_idx == 0, last);
      barrier_release(lock, last);
    }

    // The last one holds the total, so it is the one that writes.
    if (last) {
      write_result();
    }
    next_slice();
    if (slice_iters) {
      seek_to_slice();
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME
