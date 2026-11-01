# Speed up a model that spills to the host with a drafter

A small drafter model proposes several tokens, and the large model verifies them in one pass, so a model that spills to the host crosses the bus once for several tokens.

## Before you start

<!-- site: widget checklist id=drafter-prereq -->
- [ ] The target model is too large for the cards, so part of its layers sit on the host.
- [ ] A smaller model with the same ordinary vocabulary is in the store.
- [ ] For the manual route, the target is loaded.
<!-- site: end -->

A model that fits the cards gains nothing: verification costs about what drafting saves.

## Smallest configuration

None. loken attaches a drafter by itself when a model is predicted to spill.

## Attach a drafter

<!-- site: widget steps id=drafter-attach -->
1. Load the target model.
2. Attach the drafter.

   ```sh
   curl -s localhost:11435/api/draft/attach -H 'Content-Type: application/json' -d '{"target": "TARGET", "draft": "DRAFT"}'
   ```

3. List the pairs.

   ```sh
   curl -s localhost:11435/api/draft/status
   ```

4. Detach it when you no longer need it.

   ```sh
   curl -s localhost:11435/api/draft/detach -H 'Content-Type: application/json' -d '{"target": "TARGET"}'
   ```
<!-- site: end -->

| Field | Route | Effect |
|---|---|---|
| `target` | attach, detach | The loaded model to speed up. |
| `draft` | attach | The drafter model. |
| `device_index` | attach | The card for the drafter. Default: the fastest card. |

These routes are loken's own.
The Ollama API has none.

## Expected result

```json
{"target": "TARGET", "draft": "DRAFT", "target_vocab": N, "draft_vocab": N, "tokenizer_fp_match": true, "status": "attached"}
```

`/api/draft/status` answers `{"pairs": [{"target": ..., "draft": ..., "draft_length": ...}]}`.

The answers keep the target's quality, because the target verifies every draft with the request's own sampling.
The gain grows with the share of drafts the target accepts, and a drafter of the same family accepts more.  
The measured gains are in [Status](../STATUS.md).

## Automatic attach

When a model is predicted to spill, loken picks the smallest model of the store with the same ordinary vocabulary.
It loads that model on the fastest card before the target and places the target around it.

loken releases the drafter when the target ends up on the cards, or when the vocabularies differ.

loken learns the number of tokens drafted per cycle.
It measures how often each draft position is accepted and what a cycle of each length costs, then drafts the length that commits the most tokens per second.

Plain decoding is one of the lengths, so a drafter that stops paying is dropped.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 404 `Target model 'x' not loaded` | The target is not resident. | Load it with a request first. |
| 400 `Tokenizer fingerprint mismatch` | The drafter has another ordinary vocabulary, or more logit rows than the target. | Choose a drafter of the same family. |
| 400 `No card has room for the draft` | The target is held on the cards and the drafter would run on the host. | Free a card, or skip the drafter. |
| 500 `Draft load failed` | The drafter did not load. | Read the message that follows. |

## Next

- [Measured gains](../STATUS.md)
- [API routes](../API.md)
