# W4A16: a 4-bit weight against a 16-bit activation, on the tensor cores

What this kernel computes: `C = A . Bᵀ`, with `A` f16 of shape `m x k`, `B` four-bit unsigned
integers of shape `n x k`, and every 128 consecutive weights of a row sharing one f16 scale and
one 4-bit zero point. The output is f16.

The difficulty is not the arithmetic. It is that a 4-bit weight is a quarter the width of the
activation it meets, so a kernel that reads weights at the rate the tensor cores consume them
is bound by everything *except* the multiply: by how the weights are laid out, by whether the
loads reach shared memory before the previous tile is done with it, and by how partial results
find each other afterwards. This file states the four answers, so the code below can be read
against a design rather than reverse-engineered from itself.

Names: `m` is the batch, `n` the output width, `k` the reduction. A *tile* is 16x16. A
threadblock owns `thread_m_blocks x thread_n_blocks` output tiles and walks `thread_k_blocks`
at a time along k.

---

## 1. The weights arrive pre-permuted

The kernel never permutes anything. `src/tensor/marlin.rs` repacks the checkpoint once at load
time so that **every load in the inner loop is a contiguous 16-byte read that lands exactly
where the tensor-core fragment wants it**. Three arrays come out of that repack:

| array | shape | layout |
|---|---|---|
| `b_q` | `[k/16, n.16/8]` u32 | 16x64 tile-permuted, lane-interleaved |
| `scales` | `[k/128, n]` f16 | 64-column lane permutation |
| `zp` | `[k/128, n/8]` u32 | the same permutation, plus the dequant interleave |

Two permutations are folded together in `b_q`:

- **The tile permutation.** `mma.sync.m16n8k16` wants each lane to hold specific elements of
  the 16x16 fragment. Storing `B` in its natural order would make every fragment load a
  gather. Storing it in fragment order makes it one `ldmatrix`-shaped read.
- **The dequant interleave.** Unpacking four bits into an f16 is done by the bit trick in
  `dequant.h`: mask the nibble into the mantissa of a constant exponent, then subtract that
  constant. It produces the four values of a 32-bit word in the order `0 2 4 6 1 3 5 7`, not
  `0 1 2 3 4 5 6 7`. Rather than shuffle after every unpack, the repack stores the nibbles
  pre-shuffled, so the trick's output order *is* the fragment order.

The scales and zero points carry the same 64-column lane permutation, so a lane that owns
output columns `c` finds the scale for `c` at its own index with no cross-lane traffic.

**Consequence for anyone editing this kernel:** an index here is meaningless without its
counterpart in `marlin.rs`. `repack_matches_reference_dequant` is the test that ties them.

---

## 2. Four stages of `cp.async`, one barrier apart

Global memory is far enough away that a tile must be in flight several iterations before it is
needed. The kernel keeps `stages = 4` tiles of `A` and `B` in shared memory at once:

```
        fetch(i)      for i in 0..stages-1        <- start_pipes
   +--> wait_for_stage()                          <- cp_async_wait<stages-2>
   |        for k in 0..b_sh_wr_iters:
   |            fetch_to_registers(k+1)           <- next k's fragments, from shared
   |            if k == b_sh_wr_iters-1:
   |                fetch(pipe)                   <- next tile, from global
   |            matmul(k)                         <- this k's mma
   +----   advance pipe
```

`cp_async_wait<stages-2>` is the whole trick: it waits until all but the last two groups have
landed, which means the tile being read is complete while the tile after it is still arriving.
Fetching the *next* tile at the *last* k of the current one is what keeps the pipe full without
a second barrier.

Registers are double-buffered on the same principle: `fetch_to_registers(k+1)` runs before
`matmul(k)`, so the shared-memory read for the next step overlaps this step's arithmetic.

Scales and zero points ride the same pipeline but on a slower schedule: a 128-weight group
spans `g = group_blocks / thread_k_blocks` stages, so they are read once per `g` stages and
held.

---

## 3. The inner loop: unpack, subtract, scale, multiply

Per k step, per output column pair `j`:

1. **Unpack.** `dequant_data` turns one 32-bit word into eight f16 weights via the mantissa
   trick - no integer-to-float conversion instruction, no lookup.
2. **Subtract the zero point.** The zero point is an integer in the weight's own scale, so it
   comes off the dequantised value directly: `sub_zp`.
3. **Scale.** One f16 multiply per pair against the group scale: `scale`.
4. **Multiply.** `mma.sync.aligned.m16n8k16` per `thread_m_blocks`, accumulating into `frag_c`.

Steps 1-3 are why the weights were permuted in §1: each is a plain elementwise operation on a
register the load already put in the right place.

The `m_block_size_8` variant exists for decode, where `m <= 8`: it runs `mma_trans` on a
transposed fragment pair rather than two `mma`s, because at eight rows the second accumulator
would be half empty.

On Turing the accumulator is f16 (`use_fp16_accum`): its f32 tensor-core rate for this shape is
half its f16 one, and a 128-weight group keeps the partial sums inside f16's range. Later
architectures accumulate in f32 and lose nothing for it.

---

## 4. Partial tiles, and how they find each other

A threadblock's slice of the k dimension usually does not cover a whole output tile, so the
same tile is computed in pieces by several blocks. Three mechanisms close that gap:

- **Within a threadblock** - `thread_block_reduce` folds the warps' accumulators together
  through shared memory. The region it uses is the same one the weight tile occupied; they
  never live at once, which is why shared memory is sized by whichever is larger.
- **Across threadblocks, in order** - `barrier_acquire` / `barrier_release` on a per-tile lock
  let blocks add into `C` one at a time, in slice order. `global_reduce_fp32` accumulates
  through an f32 scratch buffer (`C_tmp`); `global_reduce_fp16` adds in f16 directly, which is
  cheaper and less accurate.
- **Across threadblocks, out of order** - with `use_atomic_add`, blocks add into `C` with
  atomics and no lock at all. Only correct when the accumulation is associative enough for the
  caller, which is why it is a flag rather than the default.

The `parallel` split above these: when `m` exceeds one tile of rows, the problem is cut into
independent batch-size-`m_block_size` problems, which reduces how many blocks contend for any
one tile.

---

## What this kernel deliberately does not do

Its upstream carries nvfp4, mxfp8, fp8, signed 4- and 8-bit, bf16, 8-bit activations, float
zero points, activation reordering, a bias, and any stage count. This build instantiates one
cell - f16 x u4-with-integer-zero-point -> f16, four stages, a scale every 128 weights - and
the kernel is written for that cell alone. The five template parameters that remain are the
ones a launch varies: `threads`, and how the output tile is cut.

Adding a format back means adding it as a policy on the pieces above, not as a flag threaded
through all four of them.

---

## Judging a change

`the_int4_tensor_core_gemm_agrees_with_the_host_on_the_same_weights` runs this kernel and a
host dequantise-then-multiply against the **same repacked weights**, so the quantisation error
cancels and the comparison is of the two paths, not of one path against a reference. It asserts
first that the Marlin layer was actually built - a test that silently covers a fallback is
worse than no test.

`repack_matches_reference_dequant` covers §1 on the host side. Between them, an index changed
here without its counterpart there fails one or the other.
