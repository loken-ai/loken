# Tokens

A model reads a row of ints and writes one int at a time. Text becomes ints going in, ints
become text coming out. Both translations fail silently: no crash, no slowdown, just plausible
nonsense.

That is the whole risk. The rest of this page is where each half lives in loken and the five
times it actually broke.

## The idea

- **Tokenizer**: a fixed vocabulary (~150k strings; qwen3 has 151 936) and a rule for cutting
  text into it. Byte-pair merges up from bytes; a SentencePiece unigram picks the
  highest-scoring split. Both ship in the checkpoint. Wrong tokenizer, wrong output, every
  kernel innocent.
- **Special tokens**: entries that mark a turn or a reasoning block and stand for no text.
  Shown or hidden is a setting, not a property of the model.
- **Chat template**: a Jinja program in the checkpoint that folds messages into the one
  string shape the model trained on. Render it slightly off and answers trail away.
- **The return split**: ids back to text, then split into answer, reasoning (`<think>`, a
  Harmony `analysis` channel), and any tool call. Not arithmetic; fails with no kernel
  complaining.

Rough sizes: an English word is ~1.3 tokens, source code ~2.

## In loken

| File | What it decides |
|---|---|
| `src/inference/engine/llm_engine/gguf.rs` | Builds the tokenizer from GGUF metadata (`tokenizer.ggml.model`, `.tokens`, `.merges`, `.token_type`, `.pre`). Synthesises merges for a SentencePiece vocabulary that ships none. |
| `src/inference/token/sentencepiece.rs` | The unigram tokenizer from scratch: `ModelProto` wire format, Viterbi segmentation, the space meta-symbol, byte fallback. |
| `src/api/handlers/prompt_format.rs` | Renders the checkpoint's `tokenizer.chat_template` through a Jinja engine. |
| `src/api/thinking.rs` | Splits reasoning from the answer (`<think>`, Harmony channels, the `thought` variant), surviving a chunk boundary mid-stream. `SPLIT_MARKERS` are registered as ordinary text so a decoder that drops special tokens keeps them. |
| `src/api/tool_calls.rs` | Reads a tool call out of the raw token stream, on every family's spelling. |

## What was measured

Four defects, none in the arithmetic, on months-old models the same week (2026-09-12). Each
sat between the model's tokens and the client's JSON, and none moved a token rate, so no speed
or coherence check could see it.

| Defect | Reach | Cause |
|---|---|---|
| a bare `thought` at the head of every answer | gemma4 31b and 26b, both surfaces | the channel tags `<\|channel>...<channel\|>` were dropped as special at decode; the channel name between them survived |
| reasoning delivered as the answer, `thinking` empty | 19 of the 44 models scanned | drop-special-tokens keyed on one spelling, gpt-oss's `<\|channel\|>`; `<think>` is a control token and fell with the rest |
| the template failed to render, a fallback rendered instead | falcon3, granite3-moe, granite3.1-dense, llama3.2 | `tojson(indent=4)` on a filter that took no named argument |
| a tool call lost | qwen3 (emitted inside the reasoning block), granite (an array where an object was expected) | `enable_thinking` never set when rendering; a valid JSON array short-circuited to nothing |

A fifth surfaced when a guard added since (2026-09-13) began rendering every stored template
with and without a tool: smollm3's `{% generation %}` tags, unknown to the Jinja engine, are
now stripped before parsing.

Dead end: patching gemma4 through the prompt fixes 31b and 26b and breaks `gemma4:latest`,
which then reasons without stopping. Four tags, three architectures; test all four.

## Try it

A reasoning model splits its answer in two:

```sh
curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
  "messages":[{"role":"user","content":"Say hello in six words."}]}'
```

`message.thinking` holds the reasoning, `message.content` the answer. Streamed, the split
happens per chunk, not once at the end.

Add a tool and `content` goes empty:

```sh
curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather",
    "description":"Current weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                  "required":["city"]}}}]}'
```

Expect `message.tool_calls[0].function` naming `get_weather` with `{"city":"Paris"}`, and
`prompt_eval_count` up by the rendered tool definition (160 tokens against 14). A `failed to
render` line in the log means the model got something other than the format it declares.
