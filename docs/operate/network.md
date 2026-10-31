# Expose loken on a network

This page turns on authentication and a rate limit, and picks the address the daemon listens on.

## Before you start

The default is safe for one person on one machine: `host` is `127.0.0.1`, no key is demanded and no rate limit applies.
The API can pull models, copy them and delete them from disk.
Whoever reaches the port can do that.

<!-- site: widget checklist id=operate-network-before -->
- [ ] `lokend` runs and answers on `localhost`.
- [ ] You know which addresses of this machine other machines may use.
- [ ] You can generate a secret key and keep it out of the repository.
<!-- site: end -->

## Smallest configuration

```toml
[server]
host = "0.0.0.0"
require_auth = true
api_keys = ["REPLACE-WITH-A-LONG-RANDOM-SECRET"]
rate_limit_per_minute = 120
```

| Key | Default | Effect |
|---|---|---|
| `host` | `"127.0.0.1"` | The listening address. `0.0.0.0` listens on every interface. An address of 1 interface listens there only. |
| `require_auth` | `false` | Demands a key on every request but the health routes. |
| `api_keys` | `[]` | The accepted keys. Empty with `require_auth = true` refuses every request. |
| `allowed_origins` | `[]` | The browser origins that may call the API once `require_auth` is on. Empty allows none. |
| `rate_limit_per_minute` | `0` | Requests per minute per client. `0` is unlimited. |
| `rate_limit_burst` | `10` | Requests that may arrive back to back before the per-minute rate applies. |

Every key is in [Configuration](../CONFIGURATION.md#server).

## Run it

Send the key as a bearer token or as `X-API-Key`:

```sh
curl -s http://192.0.2.10:11435/api/tags -H 'Authorization: Bearer REPLACE-WITH-A-LONG-RANDOM-SECRET'
```

## Expected result

The daemon logs the policy at start:

```text
API authentication ON (1 key(s), 0 allowed browser origin(s))
Rate limit: 120 requests/minute per client, burst 10
```

A request without a valid key gets `401`, in the error format of the surface it called. On `/v1`:

```json
{"error": {"message": "missing or invalid API key", "type": "invalid_request_error", "code": "unauthorized"}}
```

On `/api` the body is `{"error": "missing or invalid API key"}`.

A client over its rate gets `429` with `rate limit exceeded` and a `Retry-After` header in seconds.

## How the checks behave

| Subject | Behaviour |
|---|---|
| Health routes | `/health`, `/healthz`, `/api/health` and `/ping` need no key and are never rate limited. |
| Key lookup | Keys are compared by SHA-256 digest in constant time. |
| Rate limit client | The API key identifies the client. Without `require_auth`, all callers share 1 bucket. |
| Rate limit shape | A bucket holds `rate_limit_burst` requests and refills at `rate_limit_per_minute`. |
| Browsers | Without `require_auth`, any origin may call. With it, only `allowed_origins`. |
| Typo in the keys | `require_auth = true` with no key refuses everything and logs a warning at start. |

## What the API can do

A caller with a valid key, or any caller when `require_auth` is off, can:

| Capability | Routes |
|---|---|
| Pull a model from a registry | `POST /api/pull` |
| Delete a model and its blobs from disk | `DELETE /api/delete` |
| Copy a model, create one, upload blobs | `POST /api/copy`, `POST /api/create`, `POST /api/blobs/{digest}` |
| Swap the resident model, and read the resident list | `POST /api/swap`, `GET /api/ps` |
| Generate text, images, audio and video, and use up every card | the `/v1` and `/api` generation routes |
| Upload files, up to the body limits | `/v1/files`, `/v1/batches` |
| Fetch images by URL from the hosts in `image_fetch_hosts` | the Messages API and `/v1/embeddings` |

A key grants all of it. There is no read-only key.
Share a key only with callers you would trust to delete models.

## Check before you expose it

<!-- site: widget checklist id=operate-network-checklist -->
- [ ] `require_auth = true` and `api_keys` holds at least 1 long random key.
- [ ] `host` is the interface of the network that needs access, not `0.0.0.0` when 1 address is enough.
- [ ] `rate_limit_per_minute` is set to a value that stops a loop and lets a UI work.
- [ ] `allowed_origins` lists the browser origins that call the API, or stays empty.
- [ ] The daemon logs `API authentication ON` and `Rate limit: ...` at start.
- [ ] A request without a key returns `401` from another machine.
- [ ] The firewall allows the API port from the networks that need it and from no other.
- [ ] A reverse proxy terminates TLS when the traffic crosses a network you do not control. The daemon speaks plain HTTP.
- [ ] `image_fetch_hosts` stays empty or names only hosts you trust.
- [ ] The key is stored in a secret store, and each client reads it from there.
<!-- site: end -->

## Cluster nodes

Cluster nodes poll each other on `/api/cluster/state` without a key. A node with `require_auth` on refuses the poll with `401`, so peers cannot join it.
Restrict the port with a firewall to the addresses of the cluster nodes instead.

A node that takes part in a pipeline split also opens a binary port on every interface, at a port number the system picks. Allow all traffic between the cluster nodes.

Discovery uses multicast announcements on UDP port 41999. Any host on the segment can send one, and a node that hears it contacts the endpoint it names. Set `join` and block the port when the segment is not yours.
[Run 2 machines](cluster.md) covers the cluster itself.

## Troubleshooting

Every row comes from the status codes and log lines of the daemon.

| Symptom | Cause | Fix |
|---|---|---|
| `401` with `missing or invalid API key` | The key is absent or not in `api_keys`. | Send `Authorization: Bearer <key>` or `X-API-Key: <key>`. |
| `401` on every request after you turned `require_auth` on | `api_keys` is empty. The log says `[server] require_auth is on with no api_keys: every request will be refused`. | List a key and restart. |
| `429` with `rate limit exceeded` | The client used its burst and its per-minute rate. | Wait the seconds in `Retry-After`, or raise `rate_limit_per_minute`. |
| `429` for callers that share an address | Without `require_auth`, all callers share 1 bucket. | Turn on `require_auth`, so each key has its own bucket. |
| The log says `The API is listening on ... and it is NOT authenticated` | `host` is not loopback. The daemon logs it whether `require_auth` is on or off. | Check that `require_auth` is on, or set `host` to `127.0.0.1`. |
| A browser call fails on CORS after `require_auth` | The origin is not in `allowed_origins`. | Add the origin. |

## Next

- [Run 2 machines](cluster.md)
- [Troubleshoot](troubleshooting.md)
- [API reference](../API.md)
