# Fewer bits per weight, and what it costs

Decode is bound by bytes, so the bytes a weight occupies set the rate.
Quantisation trades precision for bytes.

## The idea

- **Block quantisation**: a trained weight is a float, 2 bytes in 16 bits. A 70B model then costs 140 GB, more than a pair of consumer cards holds. Block quantisation stores a group of consecutive weights as small integers plus 1 scale. The block's values are divided by the scale, rounded and packed.
- **k-quants** nest the idea. Q4_K holds 256 values in 8 sub-blocks, each with its own 6-bit scale and minimum under 1 16-bit super-scale. That is 144 bytes for 256 values, 4.5 bits a weight, with a better fit than Q4_0.
- **Codebooks**: below 3 bits a plain grid loses too much. IQ2_XXS stores an index into a fixed codebook of 8-value patterns instead, at 2.06 bits a weight.
- **Native scales**: checkpoints trained in fp8 or fp4 ship their own block scales, 1 per 32 by 32 tile. A reader applies them and does not recompute them.
- **The scale**: the decision inside every format. The obvious choice is the block's extreme value over the largest level. It is right only when the extreme is worth as much as the rest, which is rare. Every quantiser here starts from that answer and searches for one that loses less.
- **Calibration** asks a corpus which weights matter, so the search weights its error by an importance value per column.
- **Error compensation** carries the rounding error of 1 block into the columns quantised after it, as GPTQ does.
- **The matmul** uses a block in 1 of 2 ways. Dequantising to float and multiplying is simple and moves twice the bytes. Multiplying in the quantised domain quantises the activation to 8 bits per block too, takes integer dot products and applies the 2 scales once at the end. The integer dot product is a SIMD multiply-add of unsigned by signed bytes on AVX2, and an integer tensor-core instruction on a card. The second way makes a 4-bit model fast.
- **Judging** asks 2 questions. How well do the blocks reconstruct the original matrix? How far does the model's output move? The second counts. It is measured as KL divergence and top-1 agreement against the unquantised model on held-out text.

| Format | Layout | Bytes per block | Bits per weight |
|---|---|---|---|
| Q4_0 | 32 values at 4 bits plus a 16-bit scale | 18 | 4.5 |
| Q8_0 | 32 values at 8 bits plus a 16-bit scale | 34 | 8.5 |
| Q4_K | 256 values, 8 sub-blocks | 144 | 4.5 |
| Q6_K | 256 values, 16 sub-blocks | 210 | 6.56 |
| IQ2_XXS | 256 values as indexes into a codebook of 8-value patterns | 66 | 2.06 |

## What loken does

| Part | Behaviour |
|---|---|
| Formats | Reads the legacy, k-quant and IQ families, the fp8, fp4 and bf16 checkpoints with their own block scales, and a second 4-bit format. Every format is checked against the reference implementation's output. |
| Matmul | Each call takes the fastest path that accepts it: a mat-vec for decode, a tiled path for prefill, a repacked path when the format has one. Dequantise and multiply always works as a fallback. |
| Re-quantising | `/api/create` with `quantize` re-quantises per tensor, with calibration and error compensation. `"keep"` with `tensor_types` rules leaves the other tensors alone. |

## What was measured

**Re-quantising on the card (2026-09-13, desktop node).**
The `/api/create` route was rebuilt as a detached job that streams its progress and survives a client disconnect.
The speed came from parallelism. The per-tensor work had been single-threaded.

Card kernels exist for q8_0, q4_K, q3_K and q2_K.
Their contract is equivalent quality and a reproducible result, not a bit-identical one.
The f32 reduction order and the reciprocals on the card let the iterated fit pick a different candidate scale on a near-tie.
A card-made file therefore has a stable digest that differs from the host-made one.

A uniform base type re-quantised the attention matrices downward and lost the precision the original mix kept there.
`quantize: "keep"` with per-tensor rules fixes that.

**A better file than the published ones, at the same size (2026-09-16, both cards).**
The model is the 552B mixture, with experts at IQ2_XXS for gate and up and Q2_K for down.
The file was calibrated on a corpus and gated on 512 held-out positions against the official checkpoint through an identical forward path.

<!-- site: widget chart type=bar id=quantisation-gate -->
| Metric | Best published file | loken file |
|---|---|---|
| KL (lower is better) | 0.169 | 0.139 |
| Top-1 agreement (higher is better) | 0.674 | 0.717 |
<!-- site: end -->

Error compensation was the lever: relative error on gate and up went from 0.267 to 0.177, a third lower.
The same gate rejected 3 changes:

- per-channel scaling
- 1 024-column compensation blocks
- an importance shared across a layer's experts, which was worse out of sample than the published file

**2 failures to check before the code.**

| Failure | Evidence |
|---|---|
| A published file had its attention matrices converted from fp8 without their block scales. | Root-mean-square 1e-10, cosine 0.39 against the official weights. No engine could make the file speak, and the port was right. |
| A converter wrote expert codes column-wise where the format is row-major. | The experts were worse than a zero matrix. The verification compared the file to the cache that already carried the defect. |

Verify against the source, not against a derivative.

## Try it

<!-- site: widget steps id=quantisation-steps -->
1. Check the arithmetic of 2 formats. Q4_0 is 32 values at 4 bits plus a 16-bit scale: 18 bytes, 4.5 bits a weight. Q8_0 is 32 values at 8 bits plus the scale: 34 bytes, 8.5 bits a weight.
2. Re-quantise the whole model at 3 bits.

   ```sh
   curl -s -N localhost:11435/api/create -d '{"name":"qwen3-0.6b-q3","from":"qwen3:0.6b",
     "quantize":"q3_K"}'
   ```

3. Check that the reply streams the job's progress as NDJSON and ends with `success`.
4. Check that `/api/tags` lists the new tag beside the source. Observed: 329 MB against 523 MB.
5. Re-quantise only some tensors with `"quantize":"keep"` and a `tensor_types` map: `{"ffn_gate":"q3_K","ffn_up":"q3_K","ffn_down":"q3_K"}`. A uniform type would flatten the matrices the original mix kept at higher precision.
6. Remove the result with `/api/delete` and `{"model":"qwen3-0.6b-q3"}`.
<!-- site: end -->

## Quiz

<!-- site: widget quiz answer=1 id=quantisation-quiz-1 -->
Which question about a quantised model counts most?

1. How far the model's output moves.
2. How well the blocks reconstruct the matrix.
3. How small the file is.

Judging uses KL divergence and top-1 agreement against the unquantised model on held-out text.
<!-- site: end -->

<!-- site: widget quiz answer=3 id=quantisation-quiz-2 -->
What makes a 4-bit model fast in the matmul?

1. Dequantising to float first.
2. Storing a larger scale.
3. Multiplying in the quantised domain with integer dot products.

Dequantising moves twice the bytes. The quantised-domain product applies the 2 scales once at the end.
<!-- site: end -->

<!-- site: widget quiz answer=2 id=quantisation-quiz-3 -->
How do you re-quantise only the feed-forward tensors?

1. `"quantize":"q3_K"`
2. `"quantize":"keep"` with a `tensor_types` map
3. `/api/delete`

`"keep"` leaves the tensors without a rule alone.
<!-- site: end -->

## Key terms

<!-- site: widget flashcards id=quantisation-terms -->
| Term | Meaning |
|---|---|
| Block quantisation | A group of consecutive weights stored as small integers plus 1 scale. |
| k-quant | A format that nests sub-blocks, each with its own scale, under 1 super-scale. |
| Calibration | Asking a corpus which weights matter, to weight the quantiser's error. |
| Error compensation | Carrying the rounding error of 1 block into the columns quantised after it. |
| KL divergence | The distance between the quantised and the unquantised model's output distributions. |
<!-- site: end -->
