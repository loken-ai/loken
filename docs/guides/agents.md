# Run a coding agent on loken

`loken launch` starts Claude Code or Cline with the daemon as the model provider, as `ollama launch` does. Nothing leaves the machine.

## Before you start

<!-- site: widget checklist id=agents-prereq -->
- [ ] `lokend` is running.
- [ ] The agent is installed: `claude` or `cline` on the `PATH`.
- [ ] The model follows tool calls and declares a context of at least 32k tokens.
- [ ] `LOKEN_API_KEY` is set when the daemon requires a key.
<!-- site: end -->

## Smallest configuration

None. `loken launch` writes what the agent reads.
Without `--model`, the agent gets the one model loaded, or the one model held.

## Run it

<!-- site: widget tabs group=agent id=agents-run -->
### Claude Code

```sh
loken pull qwen3:8b --source ollama
loken launch claude --model qwen3:8b
```

### Cline

```sh
loken pull qwen3:8b --source ollama
loken launch cline --model qwen3:8b
```
<!-- site: end -->

Arguments after `--` go to the agent:

```sh
loken launch claude -- -p "explain this repository"
```

## Expected result

`loken launch` prints one line, then hands the terminal to the agent:

```
claude on qwen3:8b at http://localhost:11435, window 40960
```

The window is the one the checkpoint declares. Cline also prints the files it wrote.

## What each agent gets

| Agent | Where it reads | What is set |
|---|---|---|
| Claude Code | the environment of the process | `ANTHROPIC_BASE_URL` on the daemon, `ANTHROPIC_AUTH_TOKEN`, an empty `ANTHROPIC_API_KEY`, the model behind every tier (opus, sonnet, haiku and subagents), and the window the checkpoint declares in `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, so the session compacts within it |
| Cline | `~/.cline/data/settings/providers.json` and `~/.cline/data/globalState.json` | the Ollama provider on the daemon for both modes, the model, and the welcome screen skipped. The previous file stays next to it as `.bak` |

`LOKEN_API_KEY` becomes the token Claude Code presents and the key in Cline's provider file.

## Options

| Option | Effect |
|---|---|
| `--model`, `-m` | The model the agent talks to. |
| `--config` | Writes the configuration and stops. Claude Code gets `export` lines on the terminal, Cline gets its files. Start the agent by hand or from an editor. |
| `-- ARGS` | Arguments handed to the agent. |

## Choosing a model

The floor is a model that follows tool calls and a context of at least 32k tokens.
The model loads with the window its checkpoint declares, and `loken launch` bounds the session by it.
Claude Code speaks the Messages API and Cline the Ollama API.
Any chat model the daemon serves works with both. On a cluster, the request goes to the node that holds it.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pass --model; the daemon has ...` | No `--model`, and the daemon has 0 or several models loaded and held. | Pass `--model` with a name from `loken list`. |
| `cannot ask the daemon what is loaded: ...` | The daemon is unreachable. | Start `lokend`. |
| `claude is not installed. Install it with: ...` | The agent binary is not on the `PATH`. | Run the install command the message prints. Cline is `npm install -g cline`. |
| 401 `missing or invalid API key` | The daemon requires a key. | Export a key from `api_keys` as `LOKEN_API_KEY`, then launch again. |

## Next

- [Messages API and Ollama API routes](../API.md)
- [Environment variables, `LOKEN_API_KEY`](../CONFIGURATION.md)
