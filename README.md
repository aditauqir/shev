# shev

**A simple Jev-like decision making tool, without the hype.**

You hand it a whole message — an email, a ticket, a review, a document, a paste
from a support queue — and you tell it what you want to do with that message. It
gives you back **one** decision, as JSON:

```json
{
  "actionable": "Reply to Sam and cc finance about the overdue March invoice.",
  "take_action": true,
  "confidence": 95,
  "generated_response": "Hi Sam, ... (the reply, written out)"
}
```

That is the whole idea. No agents, no orchestration, no prompt engineering, no
five-hundred-token system preamble you have to memorise. One message in, one
decision out, saved to a file you can read.

- **The whole message is judged at once.** shev never splits it into paragraphs
  and never decides line by line. One message, one decision.
- **You say what you want.** Before it answers, shev asks what you want to do
  with the message. That intent drives the written response.
- **The decision has a shape.** `actionable` (a string), `take_action` (a
  boolean), and `confidence` (an integer, always inside `[34, 100]`). The model
  cannot hand you a tuple you have to guess at.
- **It is local-friendly.** Talk to DeepSeek over the network, or run a small
  ternary MLX model entirely on your Mac. Your call, switchable at runtime.
- **It records everything.** Decisions accumulate in `responses.json` as
  `response1`, `response2`, … and can be exposed to other models over MCP.

## Install

One line, copy-pasteable:

```sh
curl -fsSL https://raw.githubusercontent.com/aditauqir/shev/main/install.sh | bash
```

If you already have Rust, this is the same thing and just as short:

```sh
cargo install --git https://github.com/aditauqir/shev
```

`shev` needs Apple Silicon or any machine with a terminal. Network features (the
DeepSeek model) need an internet connection; the local model needs an Apple
Silicon Mac.

## Quick start

```sh
shev
```

1. On first run, paste a DeepSeek API key when asked. It is stored at
   `~/.config/shev/config.json` with `0600` permissions and reused after that.
2. Type (or paste) the whole message. Press **enter** to send, **ctrl+j** for a
   new line inside a multi-line message.
3. Say what you want to do with it — "draft a short reply", "tell me if this
   needs me today", "summarise the risk".
4. Read the decision. Press **enter** to start the next one, or click the
   `▸ open responses.json in Finder` button to jump to the file.

Everything after the first run is just: run `shev`, type, done.

## The TUI

`shev` opens a terminal UI: a centred logo on its own solid panel, a prompt box
that starts at two lines and grows with your text, a command palette, and a
token/usage readout in the corner. The blocks sit on solid boxes over a dim
background so the text always stays readable.

Type `/` for the command list:

| Command | What it does |
| --- | --- |
| `/file <path>` | Load a message from a file (drag-and-drop works too) |
| `/copy [json]` | Copy the last response (or its JSON) to the clipboard |
| `/key` | Enter or replace the DeepSeek API key |
| `/cancel` | Abort the request in flight |
| `/clear` | Clear the box, or the transcript |
| `/help` | Show the keybindings |
| `/model` | Pick the model and manage it (keys, install, uninstall) |
| `/home` | Back to the home screen (the transcript is kept) |
| `/mcp` | Install, remove, or report the local MCP bridge |
| `/quit` | Leave |

## Models

`/model` switches between two backends and manages each one.

**DeepSeek** — the default. Remote, fast, needs an API key. Options: enter a new
key, remove the stored key, or cancel.

**Bonsai** — `prism-ml/Ternary-Bonsai-2-2B-mlx-2bit`, a small ternary model that
runs on Apple Silicon through MLX. Options: install it from the internet
(about 1.2 GB, into `~/.local/share/shev/models`) or uninstall it.

The Bonsai pack ships its own loader, because its weights cannot be loaded by
stock `mlx-lm`/`mlx-vlm` (they report `Unsupported model type:
prism_hadamard_qwen35`). shev therefore runs the pack's bundled runtime in a
one-shot subprocess per request, using `uv` with the pack's pinned requirements.
Nothing listens on a port, and no server is left running. The first local
request installs the MLX stack into a `uv` environment, which can take a few
minutes.

## MCP

`/mcp install` registers a local, read-only MCP server that exposes your
recorded decisions to other models. Tools: `list_decisions`, `latest_decision`,
`get_decision`, `search_decisions`; resources: `deepslop://decisions` and
`deepslop://decisions/latest`. The same server runs standalone:

```sh
shev --mcp
```

`/mcp uninstall` reverses the registration cleanly.

## CLI

The TUI is the friendly front door. The line-based CLI is `shev-cli` and is meant
for pipes:

```sh
echo "Your invoice is 30 days overdue." | shev-cli -i "draft a polite reminder"
shev-cli "Should I ship this today?" -i "answer yes or no and why"
shev-cli --mcp                       # serve decisions over MCP on stdio
```

| Option | Meaning |
| --- | --- |
| `-c`, `--once <message>` | Judge one message and exit |
| `-i`, `--intent <text>` | What you want to do with it |
| `-f`, `--file <path>` | Where decisions are appended (default `responses.json`) |
| `--mcp` | Serve the recorded decisions over MCP (stdio) |

Useful environment variables:

| Variable | Meaning |
| --- | --- |
| `DEEPSEEK_API_KEY` | Use this key instead of the stored one |
| `SHEV_RESPONSES_FILE` | Where decisions are appended |
| `SHEV_INTENT` | Default intent for a non-interactive run |
| `SHEV_MODELS_DIR` | Where local models live |
| `SHEV_BONSAI_MAX_TOKENS` | Token cap for the local model (default `2048`) |

## Where things live

| Path | What it is |
| --- | --- |
| `~/.config/shev/config.json` | API key and the selected model (`0600`) |
| `responses.json` | The decisions, appended as `response1`, `response2`, … |
| `~/.local/share/shev/models/` | Downloaded local models |

## License

Apache-2.0. See `LICENSE`.
