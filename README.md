# shev

**A simple Jev-like decision maker. Without the hype.**

Hand it one whole message — an email, a ticket, a review, a doc, whatever you
pasted out of the support queue — and tell it what you want done with it. It
hands back **one** decision, as JSON:

```json
{
  "actionable": "Reply to Sam and cc finance about the overdue March invoice.",
  "take_action": true,
  "confidence": 95,
  "generated_response": "Hi Sam, ... (the reply, written out)"
}
```

That's the whole thing. No agents. No orchestration. No prompt engineering. No
five-hundred-token preamble to memorise. One message in, one decision out, in a
file you can just read.

- **Whole message, judged once.** It doesn't chop the thing into paragraphs and
  it doesn't go line by line. One message, one decision.
- **You say what you want.** It asks before it answers, and what you want done
  is what drives the reply it writes.
- **The answer has a shape.** `actionable` (a string), `take_action` (a
  boolean), `confidence` (an integer, always in `[34, 100]`). No tuple you have
  to guess at.
- **Runs on your Mac if you want.** DeepSeek over the network, or a small
  ternary MLX model sitting right there on the machine. Switch whenever.
- **It writes it all down.** Decisions pile up in `responses.json` as
  `response1`, `response2`, … and other models can read them over MCP.

## Install

One line, copy-pasteable:

```sh
curl -fsSL https://raw.githubusercontent.com/aditauqir/shev/main/install.sh | bash
```

Got Rust already? Same thing, just as short:

```sh
cargo install --git https://github.com/aditauqir/shev
```

## What you need

- **Best for local use:** Apple Silicon — **M1 or newer, 16 GB of RAM and up**.
- **8 GB** will fit the local model too, but it's tight.
- **Everything else:** runs anywhere with a DeepSeek API key. No Mac required.

<sub>Only the local model cares about the Mac. It runs on MLX, and MLX is Apple
Silicon only, so "run it on your machine" means "run it on a Mac". Point shev at
a DeepSeek key instead and the work goes over the network — then it builds and
runs on any platform Rust does.</sub>

## Quick start

```sh
shev
```

1. First run, paste a DeepSeek API key when it asks. It's stored at
   `~/.config/shev/config.json` as `0600` and reused after that.
2. Type or paste the whole message. **enter** sends it, **ctrl+j** drops a new
   line in the middle of a multi-line one.
3. Say what you want done with it — "draft a short reply", "does this need me
   today", "where's the risk here".
4. Read the decision. **enter** starts the next one, or click
   `▸ open responses.json in Finder` and go look at the file.

After the first run it's just: run `shev`, type, done.

## The TUI

`shev` opens a terminal UI: a centred wordmark on its own solid panel, a prompt
box that starts at two lines and grows with your text, a command palette, and a
token readout in the corner. The blocks sit on solid boxes over a dim background
so the text stays readable.

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

The Bonsai pack ships its own loader, because its weights can't be loaded by
stock `mlx-lm`/`mlx-vlm` (they report `Unsupported model type:
prism_hadamard_qwen35`). So shev runs the pack's bundled runtime in a one-shot
subprocess per request, using `uv` with the pack's pinned requirements. Nothing
listens on a port, and no server is left running. The first local request
installs the MLX stack into a `uv` environment, which can take a few minutes.

## MCP

`/mcp install` registers a local, read-only MCP server that hands your recorded
decisions to other models. Tools: `list_decisions`, `latest_decision`,
`get_decision`, `search_decisions`; resources: `deepslop://decisions` and
`deepslop://decisions/latest`. The same server runs standalone:

```sh
shev --mcp
```

`/mcp uninstall` reverses the registration cleanly.

## CLI

The TUI is the friendly front door. The line CLI is `shev-cli` and it's meant
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
