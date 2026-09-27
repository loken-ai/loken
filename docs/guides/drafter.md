# Speed up a model that spills to the host with a drafter

A model too large for the cards keeps part of its layers on the host, and every token reads
those layers over the bus. A small drafter proposes several tokens at once, the large model
verifies them in one pass, and the bus is crossed once for several tokens. It pays only there:
on a model that fits, verification costs about what drafting saves.

## Automatic

When a model is predicted to spill, loken looks in the store for the smallest model with the
same ordinary vocabulary (same pieces at the same ids, the added control tokens aside), loads
it on the fastest card before the target, and places the target around it. If the target ends
up on the cards after all, or the vocabularies differ, the drafter is released.

The number of tokens drafted per cycle is learned: loken measures how often each draft position
is accepted and what a cycle of each length costs, and drafts the length that commits the most
tokens per second. Plain decoding is one of the lengths: when drafting stops paying, as with a
drafter attached to a model that fits the cards, it is dropped.

## By hand

`POST /api/draft/attach` with `target` and `draft` attaches a drafter to a loaded model after
the same vocabulary check, on the fastest card unless the request's `device_index` names one;
`/api/draft/detach` removes it and `/api/draft/status` lists the pairs and their current draft
length. These endpoints are loken's own; the Ollama API has none.

## What to expect

The answer is the target's answer, since the target verifies every draft with the request's
own sampling. A verify runs several rows at once, and on a card its kernels can settle a
near-tie between two tokens the other way than a one-row step does. The gain depends on how many drafts the target accepts; a drafter
of the same family accepts more. Measured in [`../STATUS.md`](../STATUS.md).
