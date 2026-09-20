# Fewer bits per weight, and what it costs

Decode is bound by bytes, so the bytes a weight occupies are the rate. Quantisation is the
trade of precision for bytes, and this lesson is how the trade is made, how it is judged,
and where it went wrong.

## The idea

A trained weight is a float. Stored in sixteen bits it costs two bytes; a 70B model then
costs 140 GB, which no pair of consumer cards holds. **Block quantisation** stores a group
of consecutive weights as small integers plus one scale: the block's values are divided by
the scale, rounded, and packed. Q4_0 packs 32 values at four bits each behind one 16-bit
scale, 18 bytes for 32 weights, 4.5 bits a weight. Q8_0 spends eight bits, 8.5 a weight.
The **k-quants** nest the idea: Q4_K holds 256 values in eight sub-blocks, each with its own
6-bit scale and minimum under one 16-bit super-scale, 144 bytes for 256, 4.5 bits a weight
with a better fit; Q6_K sits at 6.56. Below three bits a plain grid loses too much, and
IQ2_XXS stores an index into a fixed codebook of eight-value patterns instead, 2.06 bits a
weight. Checkpoints trained in fp8 or fp4 ship their own block scales, one per 32 by 32
tile, which a reader applies rather than recomputes.

The decision inside every format is the scale. The obvious choice, the block's extreme
value over the largest level, is right only when the extreme is worth as much as the rest,
and it rarely is: every quantiser here starts from that answer and searches for one that
loses less. **Calibration** goes further and asks a corpus which weights matter, so the
search can weight its error by an importance value per column; **error compensation**
carries the rounding error of one block into the columns quantised after it, the way
GPTQ does.

The matmul then has two ways to use such a block. Dequantise to float and multiply, which
is simple and moves twice the bytes; or multiply in the quantised domain: the activation is
quantised to eight bits per block too, the products are integer dot products (a SIMD
multiply-add of unsigned by signed bytes on AVX2, an integer tensor-core instruction on a
card), and the two scales are applied once at the end. The second is what makes a 4-bit
model fast.

Judging a quantisation is two questions: how well the blocks reconstruct the original
matrix, and how far the model's output moves (KL divergence and top-1 agreement against the
unquantised model on held-out text). The second is the one that counts.

## In loken

| File | What it decides |
|---|---|
| `src/tensor/quant_cpu/format/q4_0.rs`, `q4_k.rs`, `iq2_xxs.rs` | One file per format, holding layout, dequantiser and quantiser together, because a sub-block count changed in one and not the others is a silent defect. Start with `q4_0.rs`: the bias that makes an unsigned nibble signed, and why the two nibbles of a byte are not neighbours. |
| `src/tensor/quant_cpu/scale.rs` | The scale searches: signed, with offset, and the second stage that quantises the scales themselves. |
| `src/tensor/quant_cpu/avx/mod.rs` | The AVX2 dot products: the four-step shape every kernel follows and the unsigned-by-signed constraint of the multiply-add. |
| `src/tensor/quantized/kernel_matmul.rs` | Which matmul serves a call: mat-vec for decode, tiled for prefill, a repack when the format has one, each able to decline, and the dequantised matmul at the end that never does. |
| `src/tensor/oracle_parity.rs` | Every dequantiser pinned against ggml's scalar output, since a format compared only with itself proves nothing. |
| `src/inference/load/requantize.rs`, `calibration.rs`, `compensate.rs` | Re-quantising per tensor rather than per model, the importance record, the error carried into later columns. |
| `src/tensor/blockscaled/` | fp8, fp4 and bf16 with the checkpoint's block scales, read in place by any model that ships them. |
| `cuda/marlin/DESIGN.md`, `cuda/README.md` | The 4-bit tensor-core GEMM explained as a design, and which kernels are built by nvcc against which are compiled at run time. |
| `src/inference/load/awq.rs` | Why a second 4-bit format is worth carrying, in bytes a weight. |

## What was measured

**Re-quantising on the card (2026-09-13, desktop node).** The `/api/create` route that
re-quantises a model was rebuilt as a detached job that streams its progress and survives a
client disconnect. The speed came from parallelism, not from the card: the per-tensor work
had been single-threaded. Card kernels for q8_0, q4_K, q3_K and q2_K exist since, and the
contract with them is equivalent quality and a reproducible result, not a bit-identical one:
the f32 reduction order and the reciprocals on the card let the iterated fit pick a different
candidate scale on a near-tie, so a card-made file has a stable digest distinct from the
host-made one. A dead end from the same day: a uniform base type re-quantised the attention
matrices downward and lost the precision the original mix had kept there; `quantize: "keep"`
with per-tensor rules is the answer.

**A better file than the published ones, at the same size (2026-09-16, both cards).** For
the 552B mixture, experts at IQ2_XXS for gate and up and Q2_K for down, calibrated on a
corpus and gated on 512 held-out positions against the official checkpoint through an
identical forward path: KL 0.139 against 0.169 for the best published file of the same
format and size, top-1 agreement 0.717 against 0.674. The lever was error compensation:
gate and up went from 0.267 to 0.177 relative error, a third. Rejected on the same gate:
per-channel scaling, 1 024-column compensation blocks, and an importance shared across a
layer's experts, which was worse out of sample than the published file.

**Two things to check before the code.** A published file whose attention matrices had
been converted from fp8 without their block scales (root-mean-square 1e-10, cosine 0.39
against the official weights): no engine could make that file speak, and the port was
right. And a converter that wrote expert codes column-wise where the format is row-major,
producing experts worse than a zero matrix, while the verification compared the file to
the cache that already carried the defect. Verify against the source, not against a
derivative.

## Try it

Every format against ggml:

```sh
cargo test --lib oracle_parity
```

Expected: fourteen tests, one per format plus `every_format_reconstructs_about_as_well_as_ggml`
and the importance-matrix one, all passing. Then open `src/tensor/quant_cpu/format/q4_0.rs`
and check the arithmetic: 32 values, a 16-bit scale, 4 bits each, 18 bytes, 4.5 bits a
weight; `q8_0.rs` gives 34 bytes for 32, 8.5.

A re-quantisation of the whole model at three bits:

```sh
curl -s -N localhost:11435/api/create -d '{"name":"qwen3-0.6b-q3","from":"qwen3:0.6b",
  "quantize":"q3_K"}'
```

The reply streams the job's progress as NDJSON and ends with `success`; `/api/tags` then
lists the new tag beside the source (observed: 329 MB against 523 MB). A uniform base type
flattens the matrices the original mix kept at higher precision; `"quantize":"keep"` with a
`tensor_types` map (`{"ffn_gate":"q3_K","ffn_up":"q3_K","ffn_down":"q3_K"}`) re-quantises
only the named tensors. `/api/delete` with `{"model":"qwen3-0.6b-q3"}` removes the result.
