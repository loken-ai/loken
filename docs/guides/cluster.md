# Two machines

Every node runs the same daemon and any node is an entry point: a request arrives anywhere, the
node forwards it whole to the best holder and relays the stream back. This holds for every
surface that names a model: Ollama generate and chat, OpenAI chat and completions, and the
Messages API. A request that only loads a model goes to a node that has it. Every other
route that loads a model consults the cluster too: image generation, edits and variations,
video, sound and music, speech, transcription and translation, embeddings, reranking, and
each turn of the conversation route. A node without the weights, or without a card that
holds an image family whole, hands the request to the least loaded peer that has them, and
an upload travels as it arrived. Each node needs a `[cluster]` block; the rest is discovered.

```toml
[cluster]
name = "home"
advertise = "http://192.0.2.10:11435"
```

`advertise` is the address peers reach this node at, and the one field that cannot be derived.
Nodes on one network hear each other by multicast; across subnets, list the others in `join`.
Turn `require_auth` on before a node listens beyond localhost.

## Check it

```sh
curl -s localhost:11435/api/cluster/peers
```

lists who this node sees, with liveness and the measured round trip; `/api/cluster/state` is
what it publishes about itself. `/api/tags` on any node lists the whole cluster's models, the
ones held elsewhere under the node that holds them, so a client sees one catalogue whichever
node it talks to.

## What runs and what does not

Replicated serving runs. Sharding one model across hosts is written and reached by nothing:
[`../CLUSTER.md`](../CLUSTER.md) describes both halves, and
[`../STATUS.md`](../STATUS.md#the-cluster) says what has been measured.
