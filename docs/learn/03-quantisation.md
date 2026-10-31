# Fewer bits per weight, and what it costs

Decode is bound by bytes, so the bytes a weight occupies are the rate. Quantisation trades
precision for bytes. This lesson is how the trade is made, how it is judged, and where it went
wrong.

## The idea

- **Block quantisation**: a trained weight is a float. Stored in 16 bits it costs 2
  bytes. A 70B model then costs 140 GB, which no pair of consumer cards holds. Block
  quantisation stores a group of consecutive weights as small integers plus one scale: the
  block's values are divided by the scale, rounded, and packed. Q4_0 packs 32 values at 4
  bits each behind one 16-bit scale, 18 bytes for 32 weights, 4.5 bits a weight. Q8_0 spends
  8 bits, 8.5 a weight.
- **k-quants** nest the idea: Q4_K holds 256 values in 8 sub-blocks, each with its own
  6-bit scale and minimum under one 16-bit super-scale, 144 bytes for 256, 4.5 bits a weight
  with a better fit. Q6_K sits at 6.56. Below 3 bits a plain grid loses too much, and
  IQ2_XXS stores an index into a fixed codebook of 8-value patterns instead, 2.06 bits a
  weight. Checkpoints trained in fp8 or fp4 ship their own block scales, one per 32 by 32
  tile, which a reader applies rather than recomputes.
- **The scale**: the decision inside every format. The obvious choice, the block's extreme
  value over the largest level, is right only when the extreme is worth as much as the rest,
  and it rarely is: every quantiser here starts from that answer and searches for one that
  loses less. **Calibration** asks a corpus which weights matter, so the search can weight its
  error by an importance value per column. **Error compensation** carries the rounding error
  of one block into the columns quantised after it, the way GPTQ does.
- **The matmul**: 2 ways to use a block. Dequantise to float and multiply, which is simple
  and moves twice the bytes. Or multiply in the quantised domain: the activation is quantised
  to 8 bits per block too, the products are integer dot products (a SIMD multiply-add of
  unsigned by signed bytes on AVX2, an integer tensor-core instruction on a card), and the 2
  scales are applied once at the end. The second is what makes a 4-bit model fast.
- **Judging** is 2 questions: how well the blocks reconstruct the original matrix, and how
  far the model's output moves (KL divergence and top-1 agreement against the unquantised
  model on held-out text). The second is the one that counts.

## What loken does

- **Formats**: reads the legacy, k-quant and IQ families, the fp8, fp4 and bf16 checkpoints with
  their own block scales, and a second 4-bit format. Every format is checked against the
  reference implementation's output.
- **Matmul**: each call takes the fastest path that accepts it, a mat-vec for decode, a tiled
  one for prefill, a repacked one when the format has it. Falling back to dequantise and
  multiply always works.
- **Re-quantising**: `/api/create` with `quantize` re-quantises per tensor, with calibration
  and error compensation. `"keep"` with `tensor_types` rules leaves the other tensors alone.

## What was measured

**Re-quantising on the card (2026-09-13, desktop node).** The `/api/create` route that
re-quantises a model was rebuilt as a detached job that streams its progress and survives a
client disconnect. The speed came from parallelism, not from the card: the per-tensor work
had been single-threaded. Card kernels for q8_0, q4_K, q3_K and q2_K exist since, and the
contract with them is equivalent quality and a reproducible result, not a bit-identical one:
the f32 reduction order and the reciprocals on the card let the iterated fit pick a different
candidate scale on a near-tie, so a card-made file has a stable digest distinct from the
host-made one. A dead end from the same day: a uniform base type re-quantised the attention
matrices downward and lost the precision the original mix had kept there. `quantize: "keep"`
with per-tensor rules is the answer.

**A better file than the published ones, at the same size (2026-09-16, both cards).** For
the 552B mixture, experts at IQ2_XXS for gate and up and Q2_K for down, calibrated on a
corpus and gated on 512 held-out positions against the official checkpoint through an
identical forward path: KL 0.139 against 0.169 for the best published file of the same
format and size, top-1 agreement 0.717 against 0.674. The lever was error compensation:
gate and up went from 0.267 to 0.177 relative error, a third. Rejected on the same gate:
per-channel scaling, 1 024-column compensation blocks, and an importance shared across a
layer's experts, which was worse out of sample than the published file.

**2 things to check before the code.** A published file whose attention matrices had
been converted from fp8 without their block scales (root-mean-square 1e-10, cosine 0.39
against the official weights): no engine could make that file speak, and the port was
right. And a converter that wrote expert codes column-wise where the format is row-major,
producing experts worse than a zero matrix, while the verification compared the file to
the cache that already carried the defect. Verify against the source, not against a
derivative.

## Try it

Check the arithmetic of 2 formats: Q4_0 is 32 values at 4 bits plus a 16-bit scale, 18 bytes,
4.5 bits a weight. Q8_0 is 32 values at 8 bits plus the scale, 34 bytes, 8.5 bits a weight.

A re-quantisation of the whole model at 3 bits:

```sh
curl -s -N localhost:11435/api/create -d '{"name":"qwen3-0.6b-q3","from":"qwen3:0.6b",
  "quantize":"q3_K"}'
```

The reply streams the job's progress as NDJSON and ends with `success`. Then `/api/tags`
lists the new tag beside the source (observed: 329 MB against 523 MB). A uniform base type
flattens the matrices the original mix kept at higher precision. With `"quantize":"keep"` and a
`tensor_types` map (`{"ffn_gate":"q3_K","ffn_up":"q3_K","ffn_down":"q3_K"}`), only the
named tensors are re-quantised. `/api/delete` with `{"model":"qwen3-0.6b-q3"}` removes the result.
