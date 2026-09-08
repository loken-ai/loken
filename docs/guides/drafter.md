# Speed up a model that spills to the host with a drafter

A model too large for the cards keeps part of its layers on the host, and every token reads
those layers over the bus. A small drafter proposes several tokens at once, the large model
verifies them in one pass, and the bus is crossed once for several tokens. It pays only there:
on a model that fits, verification costs about what drafting saves.

## Configuration

```toml
[inference]
model_id = "deepseek-r1:70b"
draft_model = "llama3.2:1b"
draft_device_index = 0
```

The drafter and the target must share their ordinary vocabulary: same pieces at the same ids,
the added control tokens aside. The pairing above is the one measured in
[`../STATUS.md`](../STATUS.md#placement-measured-separately). The drafter loads whole on the
card `draft_device_index` names, and takes the streamed generate path.

## Without a restart

`POST /api/draft/attach` attaches a drafter to the resident model for the session, after the
same vocabulary check; `/api/draft/detach` removes it and `/api/draft/status` says what is
attached.

## What to expect

The answer is the target's answer token for token, since the target verifies every draft. The
gain depends on how many drafts the target accepts; a drafter of the same family accepts more.
The Ollama surface reports `eval_count` and `eval_duration` on the drafted stream as on the
plain one.
