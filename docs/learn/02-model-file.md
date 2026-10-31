# What a model file holds

A checkpoint is hundreds of tensors and hundreds of metadata entries, in a container that
lets a server read the entries without reading the tensors. That is why the catalogue never
opens more than a few kilobytes of it.

## The idea

- **GGUF**: a magic, a version, 2 counts (metadata pairs, tensors). Then metadata key by
  key: architecture name, layer count, trained context length, head counts, rotary base, and
  the whole tokenizer (vocabulary, merges, scores, token types, chat template). Then one
  descriptor per tensor: name, dimensions (stored innermost first, so a reader reverses them),
  element type, offset. Then the payload, aligned to `general.alignment` (32 by default), one
  tensor after another.
- **Element type**: where the bytes go. F16 costs 2 bytes a weight. Lesson 3's block
  formats cost between 2 and 8 bits. `general.file_type` names the mix a file was packed
  with (Q4_K_M: most matrices at 4.5 bits, a few at 6.5).
- **safetensors**: the same idea with a JSON header: names, dtypes, shapes, offsets, then the
  bytes. The format Hugging Face checkpoints ship in, one or many files with an index.
- **Memory-mapped**: nothing is copied at open. A tensor is a view at an offset. The operating
  system pages bytes in when they are first touched and keeps them in its page cache while
  memory allows. A model "loaded" on a card was read from that mapping and uploaded. A layer
  left on the host is read from it on every token.

Orders of magnitude on qwen3:0.6b: 751 632 384 parameters, 28 layers, an embedding width of
1024, a trained context of 40 960 tokens, and 522 MB resident at Q4_K_M, which is 5.6 bits
a parameter once the embedding and the output matrix are counted.

## What loken does

- **Formats**: reads GGUF (a checkpoint split in parts counts as one) and safetensors
  (one file or many, with an index).
- **Memory mapping**: a tensor is read in place from the mapping. A layer served from the host
  needs no copy.
- **Catalogue**: `/api/tags` and `/v1/models` list from the header facts, read up to the first
  `tokenizer.*` key. `/api/show` reads the whole header on demand.
- **Model swaps**: the parsed checkpoint stays on the host between swaps, so switching back
  does not re-read gigabytes.

## What was measured

**The catalogue read the vocabularies (2026-09-08, desktop node).** 44 GGUF files in
the Ollama store. Their headers total 326 MB, of which 197 MB is vocabulary arrays. Reading
them all took 11 s at every catalogue request. The general keys come first in a GGUF and the
tokenizer arrays after them, so a read that stops at the first `tokenizer.*` key gets the
architecture, the file type and the parameter count in under 8 KB a file: 2 ms for the
44. The facts are cached by path, size and modification time, and the daemon warms
that cache before it binds its port. `/api/show` still reads the whole header, once, on
demand.

**Load time is the disk.** A 24 GB checkpoint over a USB disk: 52 s reading, 6 s uploading to
the cards. The loader still uploads some tensors twice.

## Try it

```sh
curl -s localhost:11435/api/show -d '{"model":"qwen3:0.6b"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['details'])
print({k: v for k, v in d['model_info'].items() if not k.startswith('tokenizer')})"
```

Expected on qwen3:0.6b: `details` says `format: gguf`, `family: qwen3`,
`parameter_size: 751.63M` and `quantization_level: Q4_K_M`. The metadata says
`general.file_type: 15` (the id of that mix), `qwen3.block_count: 28`, `qwen3.context_length: 40960`,
`qwen3.embedding_length: 1024`, `qwen3.attention.head_count: 16` against `head_count_kv: 8`
(lesson 4: 8 query heads share a key head). Then compare with what is resident:

```sh
curl -s localhost:11435/api/models/loaded
```

`size_bytes` is the resident weight bytes. `context_length` is the window the model was
loaded with, not the trained one above.
