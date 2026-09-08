# Run a coding agent on loken

`loken launch` starts Claude Code or Cline with the daemon as their model provider, the way
`ollama launch` does. Nothing leaves the machine.

```sh
loken pull qwen3:8b --source ollama
loken launch claude --model qwen3:8b
loken launch cline --model qwen3:8b
```

Without `--model`, the agent gets the one model loaded, or the one model held. Arguments after
`--` go to the agent:

```sh
loken launch claude -- -p "explain this repository"
```

## What each agent gets

| Agent | Where it reads | What is set |
|---|---|---|
| Claude Code | the environment of the process | `ANTHROPIC_BASE_URL` on the daemon, `ANTHROPIC_AUTH_TOKEN`, an empty `ANTHROPIC_API_KEY`, the model behind every tier: opus, sonnet, haiku and subagents, and the window the model is loaded with, so the session compacts within it |
| Cline | `~/.cline/data/settings/providers.json` and `~/.cline/data/globalState.json` | the Ollama provider on the daemon for both modes, the model, the welcome screen behind. The previous file stays next to it as `.bak` |

`--config` writes or prints the configuration and stops, so the agent can be started by hand or
from an editor.

When the daemon asks for a key, set `LOKEN_API_KEY` before launching: it becomes the token
Claude Code presents and the key in Cline's provider file.

## Choosing a model

A coding agent sends long prompts with tools. A model that follows tool calls and a context of
at least 32k tokens are the floor; `context_length` in the daemon's configuration sets the
window a model is loaded with.
Claude Code speaks the Messages API and Cline the Ollama API, so any chat model the daemon
serves works for both, and on a cluster the request goes to the node that holds it.
