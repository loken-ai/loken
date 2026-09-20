# Text becomes ids, ids become text

A model reads a row of integers and emits one integer at a time. This lesson covers the two
conversions around that row: the JSON a client sends into ids, and the ids that come back into
the JSON it reads.

## The idea

A **tokenizer** is a fixed vocabulary of about a hundred thousand strings and a rule for
cutting text into them. Two rules are in use. Byte-pair encoding starts from bytes and applies
a learned list of merges in order. A unigram model, the SentencePiece kind, picks the
segmentation of highest total score. Both are deterministic and both ship in the checkpoint.
Serving a model with the wrong tokenizer produces wrong output even though every kernel is
correct.

Some entries are **special tokens** that mark the start of a turn, the end of a message, or the
start of a reasoning block. They have ids like any other but no text. Whether a decoder shows
them or drops them is a configured decision, not a property of the model.

A **chat template** turns a list of messages into one string. It is a Jinja program that ships
in the checkpoint and knows the model's markup: which token opens a system turn, where a tool
definition goes, what the assistant turn starts with. The model was trained on that exact
shape, and an approximate rendering makes answers stop after a few tokens.

The return path runs the same layer in reverse. Ids become text, and the text is split into the
answer, the reasoning (a `<think>` block, or a Harmony `analysis` channel), and any tool call.
These splits are not arithmetic and can fail without any kernel reporting an error.

Orders of magnitude: a vocabulary is around 150 000 entries (qwen3: 151 936); an English word
is about 1.3 tokens; source code runs nearer two tokens a word.

## In loken

| File | What it decides |
|---|---|
| `src/inference/engine/llm_engine/gguf.rs` | Builds the tokenizer from the GGUF metadata: `tokenizer.ggml.model`, `.tokens`, `.merges`, `.token_type`, and `.pre`, the pre-tokenizer family (the llama.cpp names are the authority). Synthesises merges for a SentencePiece vocabulary that ships none. |
| `src/inference/token/sentencepiece.rs` | The unigram tokenizer from scratch: the `ModelProto` wire format, the Viterbi segmentation, the meta-symbol for a space, the byte fallback. |
| `src/api/handlers/prompt_format.rs` | Renders the checkpoint's own `tokenizer.chat_template` through a Jinja engine, with the Python string and dict methods templates call. |
| `src/api/thinking.rs` | Splits reasoning from the answer: `<think>`, the Harmony channels, the `thought` variant, and a splitter that survives a chunk boundary in a stream. The markers are listed in `SPLIT_MARKERS` and registered as ordinary text at load, so a decoder that drops special tokens keeps them. |
| `src/api/tool_calls.rs` | Reads a tool call out of a raw token stream: the tags, the JSON, the argument object, on every family's spelling. |

## What was measured

**Four defects, none in the arithmetic (2026-09-12, desktop node).** Models benchmarked for
months broke for a user the same week. Every defect sat between the model's tokens and the
client's JSON:

| Defect | Reach | Cause |
|---|---|---|
| a bare `thought` at the head of every answer | gemma4 31b and 26b, both surfaces | the channel tags `<\|channel>...<channel\|>` were dropped as special at decode; the channel name between them survived |
| reasoning delivered as the answer, `thinking` empty | 19 of the 44 models scanned | whether to drop special tokens was decided by one spelling, gpt-oss's `<\|channel\|>`; `<think>` is a control token and fell with the rest |
| the template failed to render, and a fallback rendered instead | falcon3, granite3-moe, granite3.1-dense, llama3.2 | `tojson(indent=4)` on a filter that took no named argument |
| a tool call lost | qwen3 (emitted inside the reasoning block), granite (an array where an object was expected) | `enable_thinking` never set when rendering; a valid JSON array short-circuited to nothing |

None of them moved a token rate, so no speed or coherence measurement caught them: reasoning
delivered as prose is coherent text at a normal rate, and a tool call in the wrong field
changes neither `eval_count` nor throughput. Each failed render logged a warning, but a warning
is not a failing test.

**The guard added since (2026-09-13).** A test reads every chat template the store holds, the
way the server reads it, and renders it without a tool and with one, on a bare question and on a
system prompt plus a question. Its first run found a fifth defect: smollm3 carries
`{% generation %}` tags, a training-time extension the Jinja engine does not know, so the
template did not parse. The tags are now stripped before parsing.

**A dead end.** Fixing gemma4 by adding the channel opener to the prompt repairs 31b and 26b
and breaks `gemma4:latest`, which then reasons without stopping. Four tags, three
architectures; test all four.

## Try it

One question, whole, on a reasoning model:

```sh
curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
  "messages":[{"role":"user","content":"Say hello in six words."}]}'
```

The answer carries two fields: `message.thinking` holds the reasoning, `message.content` the
answer, and the closing object counts them apart (`eval_count` against `thinking_duration`).
Streamed (drop `"stream":false`), the first chunks carry `"content":""` and a piece of
`thinking` each; the split happens per chunk, not once at the end.

The same question with a tool:

```sh
curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather",
    "description":"Current weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                  "required":["city"]}}}]}'
```

Expected: `content` empty, `message.tool_calls[0].function` naming `get_weather` with
`{"city":"Paris"}`, `prompt_eval_count` grown by the rendered tool definition (160 tokens
against 14 without it). A `failed to render` line in the daemon's log means the model received
something other than the format it declares.
