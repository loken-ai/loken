# Two machines

Every node runs the same daemon and any node is an entry point: a request arrives anywhere, the
node forwards it whole to the best holder and relays the stream back. Each node needs a
`[cluster]` block; the rest is discovered.

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
what it publishes about itself.

## What runs and what does not

Replicated serving runs. Sharding one model across hosts is written and reached by nothing:
[`../CLUSTER.md`](../CLUSTER.md) describes both halves, and
[`../STATUS.md`](../STATUS.md#the-cluster) says what has been measured.
