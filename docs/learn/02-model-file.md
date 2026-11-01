# What a model file holds

A checkpoint is hundreds of tensors and hundreds of metadata entries in a container.
The container lets a server read the entries without reading the tensors, so the catalogue opens only a few kilobytes of a file.

## The idea

- **GGUF**: a magic, a version and 2 counts (metadata pairs, tensors). Then the metadata, key by key: architecture name, layer count, trained context length, head counts, rotary base and the whole tokenizer (vocabulary, merges, scores, token types, chat template). Then 1 descriptor per tensor: name, dimensions, element type, offset. Dimensions are stored innermost first, so a reader reverses them. Then the payload, 1 tensor after another, aligned to `general.alignment` (32 by default).
- **Element type**: how many bytes a weight takes. F16 takes 2 bytes. The block formats of [lesson 3](03-quantisation.md) take between 2 and 8 bits. `general.file_type` names the mix a file was packed with. Q4_K_M, for instance, packs most matrices at 4.5 bits and a few at 6.5.
- **safetensors**: the same idea with a JSON header of names, dtypes, shapes and offsets, then the bytes. Hugging Face checkpoints ship in this format, as 1 file or many with an index.
- **Memory-mapped**: nothing is copied at open, and a tensor is a view at an offset. The operating system pages bytes in on first touch and keeps them in its page cache while memory allows. A model loaded on a card is read from the mapping once and uploaded. A layer left on the host is read from the mapping on every token.

qwen3:0.6b in numbers:

| Property | Value |
|---|---|
| Parameters | 751 632 384 |
| Layers | 28 |
| Embedding width | 1024 |
| Trained context | 40 960 tokens |
| Resident at Q4_K_M | 522 MB |
| Bits per parameter, embedding and output matrix counted | 5.6 |

## What loken does

| Part | Behaviour |
|---|---|
| Formats | Reads GGUF (a checkpoint split in parts counts as 1) and safetensors (1 file, or many with an index). |
| Memory mapping | Reads a tensor in place from the mapping. A layer served from the host needs no copy. |
| Catalogue | `/api/tags` and `/v1/models` list from the header facts, read up to the first `tokenizer.*` key. `/api/show` reads the whole header on demand. |
| Model swaps | Keeps the parsed checkpoint on the host, so switching back does not re-read gigabytes. |

## What was measured

**The catalogue read the vocabularies (2026-09-08, desktop node).**
44 GGUF files sat in the Ollama store.
Their headers total 326 MB, of which 197 MB is vocabulary arrays.
Listing them read all of it and took 11 s at every catalogue request.

A GGUF puts the general keys first and the tokenizer arrays after them.
A read that stops at the first `tokenizer.*` key gets the architecture, the file type and the parameter count in under 8 KB a file.
That read takes 2 ms for the 44 files.

The facts are cached by path, size and modification time, and the daemon warms the cache before it binds its port.
`/api/show` still reads the whole header, once, on demand.

**Load time is the disk.**
A 24 GB checkpoint over a USB disk took 52 s reading and 6 s uploading to the cards.
The loader still uploads some tensors twice.

| Catalogue read | Time (ms) |
|---|---|
| Whole headers, 44 files | 11000 |
| Up to the first tokenizer key, 44 files | 2 |

## Try it

<!-- site: widget steps id=model-file-steps -->
1. Read the model's metadata.

   ```sh
   curl -s localhost:11435/api/show -d '{"model":"qwen3:0.6b"}' | python3 -c "
   import sys, json
   d = json.load(sys.stdin)
   print(d['details'])
   print({k: v for k, v in d['model_info'].items() if not k.startswith('tokenizer')})"
   ```

2. Check `details` on qwen3:0.6b: `format: gguf`, `family: qwen3`, `parameter_size: 751.63M`, `quantization_level: Q4_K_M`.
3. Check the metadata: `general.file_type: 15` (the id of that mix), `qwen3.block_count: 28`, `qwen3.context_length: 40960`, `qwen3.embedding_length: 1024`.
4. Check `qwen3.attention.head_count: 16` against `head_count_kv: 8`: 2 query heads share each key head ([lesson 4](04-forward-pass.md)).
5. Compare with what is resident.

   ```sh
   curl -s localhost:11435/api/models/loaded
   ```

6. Read `size_bytes`, the resident weight bytes. `context_length` is the window the model was loaded with, not the trained one.
<!-- site: end -->

## Quiz

<!-- site: widget quiz answer=2 id=model-file-quiz-1 -->
Why does the catalogue read under 8 KB of each file?

1. It samples random tensors.
2. The general keys come first, so it stops at the first `tokenizer.*` key.
3. It reads only the file name.

The tokenizer arrays, 197 MB of the 326 MB of headers, come after the keys the catalogue needs.
<!-- site: end -->

<!-- site: widget quiz answer=1 id=model-file-quiz-2 -->
What does a layer left on the host cost?

1. A read from the memory mapping on every token.
2. A copy to the card at load.
3. Nothing.

A tensor is a view at an offset, and the operating system pages it in from the mapping.
<!-- site: end -->

<!-- site: widget quiz answer=3 id=model-file-quiz-3 -->
Where is the time of a 24 GB load spent?

1. Uploading to the cards.
2. Parsing the header.
3. Reading the disk.

Reading took 52 s and uploading 6 s.
<!-- site: end -->

## Key terms

<!-- site: widget flashcards id=model-file-terms -->
| Term | Meaning |
|---|---|
| GGUF | A container of metadata, tensor descriptors and an aligned payload. |
| Element type | The format a tensor's weights are stored in, and so their bytes. |
| safetensors | A container with a JSON header of names, dtypes, shapes and offsets. |
| Memory-mapped | Read in place at an offset, paged in by the operating system on first touch. |
<!-- site: end -->
