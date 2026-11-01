# Tokens

A model reads a row of ints and writes 1 int at a time.  
Text becomes ints going in, and ints become text coming out.
Both translations fail silently, with plausible nonsense and no crash or slowdown.

## The idea

- **Tokenizer**: a fixed vocabulary of about 150k strings (151 936 for qwen3) and a rule for cutting text into it.
  Byte-pair encoding merges up from bytes.  
  A SentencePiece unigram picks the highest-scoring split.  
  Both ship in the checkpoint.
  A wrong tokenizer gives wrong output with every kernel correct.
- **Special tokens**: entries that mark a turn or a reasoning block and stand for no text.
  Whether they are shown or hidden is a setting, not a property of the model.
- **Chat template**: a Jinja program in the checkpoint that folds the messages into the 1 string shape the model trained on.
  A template rendered slightly off makes answers trail away.
- **Return split**: the step that turns ids back into text, then splits it into the answer, the reasoning (`<think>`, a Harmony `analysis` channel) and any tool call.
  It is not arithmetic, so no kernel reports its failures.

An English word is about 1.3 tokens.
Source code is about 2.

## What loken does

| Part | Behaviour |
|---|---|
| Tokenizer | Built from the checkpoint's own metadata, for byte-pair and SentencePiece vocabularies. A SentencePiece vocabulary with no merges gets them synthesised. |
| Chat template | The stored template is rendered as written, with the tools and the reasoning switch of the request. |
| Return split | The output is split into `thinking`, `content` and `tool_calls` on every family's spelling of those markers. The split holds when a marker straddles 2 streamed chunks, and when a decoder drops special tokens. |

## What was measured

4 defects found the same week (2026-09-12), on models months old.  
None was in the arithmetic.
Each sat between the model's tokens and the client's JSON and moved no token rate, so no speed or coherence check saw it.

| Defect | Reach | Cause |
|---|---|---|
| A bare `thought` at the head of every answer | gemma4 31b and 26b, both surfaces | Decoding dropped the channel tags `<\|channel>...<channel\|>` as special and kept the channel name between them. |
| Reasoning delivered as the answer, `thinking` empty | 19 of the 44 models scanned | The drop of special tokens was keyed on 1 spelling, gpt-oss's `<\|channel\|>`. `<think>` is a control token and was dropped with the rest. |
| The template failed to render and a fallback rendered instead | falcon3, granite3-moe, granite3.1-dense, llama3.2 | `tojson(indent=4)` called a filter that took no named argument. |
| A tool call lost | qwen3, granite | qwen3 emitted the call inside the reasoning block because `enable_thinking` was never set when rendering. granite returned an array where an object was expected, and a valid JSON array short-circuited to nothing. |

A fifth surfaced on 2026-09-13, when a guard began rendering every stored template with and without a tool. smollm3's `{% generation %}` tags are unknown to the Jinja engine, so they are now stripped before parsing.

Patching gemma4 through the prompt fixes 31b and 26b and breaks `gemma4:latest`, which then reasons without stopping.  
The 4 tags span 3 architectures: test all 4.

## Try it

<!-- site: widget steps id=tokens-steps -->
1. Ask a reasoning model a question.  
   `message.thinking` holds the reasoning and `message.content` the answer.
   When streamed, the split happens per chunk.

   ```sh
   curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
     "messages":[{"role":"user","content":"Say hello in six words."}]}'
   ```

2. Add a tool.
   `content` goes empty.

   ```sh
   curl -s localhost:11435/api/chat -d '{"model":"qwen3:0.6b","stream":false,
     "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
     "tools":[{"type":"function","function":{"name":"get_weather",
       "description":"Current weather for a city",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                     "required":["city"]}}}]}'
   ```

3. Check that `message.tool_calls[0].function` names `get_weather` with `{"city":"Paris"}`.
4. Check that `prompt_eval_count` grew by the rendered tool definition: 160 tokens against 14.
5. If the log shows `failed to render`, the model received something other than the format it declares.
<!-- site: end -->

## Quiz

<!-- site: widget quiz answer=2 id=tokens-quiz-1 -->
Why did no speed or coherence check see the 4 defects?

1. They only appeared on large models.
2. They sat between the model's tokens and the client's JSON, and moved no token rate.
3. They crashed the server before any check ran.

The defects were in the translation of text and ids, not in the arithmetic.
<!-- site: end -->

<!-- site: widget quiz answer=3 id=tokens-quiz-2 -->
Who decides whether special tokens are shown?

1. The model architecture.
2. The tokenizer vocabulary.
3. A setting.

Shown or hidden is a setting, not a property of the model.
<!-- site: end -->

<!-- site: widget quiz answer=1 id=tokens-quiz-3 -->
Where does a request with a tool put the call in the response?

1. In `message.tool_calls`, with `content` empty.
2. In `message.content`.
3. In `message.thinking`.

The return split separates `thinking`, `content` and `tool_calls`.
<!-- site: end -->

## Key terms

<!-- site: widget flashcards id=tokens-terms -->
| Term | Meaning |
|---|---|
| Tokenizer | A fixed vocabulary and a rule for cutting text into it. |
| Special token | An entry that marks a turn or a reasoning block and stands for no text. |
| Chat template | A Jinja program in the checkpoint that folds messages into the string shape the model trained on. |
| Return split | The step that turns ids into text and separates answer, reasoning and tool call. |
<!-- site: end -->
