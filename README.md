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

1. Bonsai is the default. Install it through `/model` if needed; no API key is
   required. DeepSeek is optional through `/model`; its key is stored at
   `~/.config/shev/config.json` as `0600`.
2. Type or paste the whole message. **enter** sends it, **ctrl+j** drops a new
   line in the middle of a multi-line one.
3. Say what you want done with it — "draft a short reply", "does this need me
   today", "where's the risk here".
4. Read the decision. **enter** starts the next one, or click
   `▸ open responses.json in Finder` and go look at the file.

After the first run it's just: run `shev`, type, done.

## The TUI

The home/start screen has a white ASCII wordmark directly over the wallpaper,
without a background box or JSON pane. Once a chat starts, two windows appear:

- **Left, the model output.** Every decision as a clean field list. The prompt box sits flush at the
  foot, with its mode line ("Build · bonsai (local) · …") on the last row.
- **Right, this turn's JSON only.** The raw record for the decision you are on,
  so you can read the file without leaving the screen. Drag across JSON text to
  select it; releasing the mouse copies exactly the selection and shows
  **✓ copied to clipboard** at the top center. The mouse wheel scrolls the JSON
  independently. `/copy json` copies the complete record, including off-screen text.

Text panels use **`#0F0E08`** over the dimmed photo wallpaper. Type `/` for commands:

| Command | What it does |
| --- | --- |
| `/file <path>` | Load a message from a file (drag-and-drop works too) |
| `/copy [json]` | Copy selected JSON or the last response; `json` copies the complete record |
| `/key` | Enter or replace the DeepSeek API key |
| `/cancel` | Abort the request in flight |
| `/clear` | Clear the visible chat and input; keep saved history |
| `/reset` | Confirm clearing the chat, all saved history, loaded context and counters; keep keys/models |
| `/help` | Show the keybindings |
| `/model` | Pick the model and manage it (keys, install, uninstall) |
| `/home` | Back to the home screen (the transcript is kept) |
| `/history` | Jump through the decisions so far |
| `/mcp` | Install, remove, or report the local MCP bridge |
| `/skills` | Load existing skill guidance; `/skills off` clears supporting context |
| `/tools [server]` | Connect to a registered MCP server and inspect read-only tools |
| `/tool server/tool <JSON>` | Call a read-only tool and attach its result to the next decision |
| `/plugins` | Show host-plugin compatibility limits |
| `/quit` | Leave |

Handy keys:

| Key | What it does |
| --- | --- |
| `enter` | Send the message, then the intent |
| `ctrl+j` | Newline inside the box |
| `ctrl+t` | Switch the model (DeepSeek ↔ Bonsai) |
| `ctrl+h` | Open the history (same as `/history`) |
| `ctrl+d` in history | Confirm deletion of only the selected saved entry |
| `ctrl+y` | Copy selected JSON, or the last generated response |
| `ctrl+c` | Quit |

`/clear` starts a fresh chat view; `/history` can reopen retained records.
`/reset` and history deletion default to **cancel** in their confirmation menus.
Confirmed changes update the responses file, so MCP clients see them too.

## Models

`/model` switches between two backends and manages each one.

**DeepSeek** — the secondary, explicitly selected remote option; needs an API key. Options: enter a new
key, remove the stored key, or cancel.

**Bonsai (default)** — `prism-ml/Ternary-Bonsai-2-2B-mlx-2bit`, a small ternary model that
runs on Apple Silicon through MLX. Options: install it from the internet
(about 1.2 GB, into `~/.local/share/shev/models`) or uninstall it.

The Bonsai pack ships its own loader, because its weights can't be loaded by
stock `mlx-lm`/`mlx-vlm` (they report `Unsupported model type:
prism_hadamard_qwen35`). So shev runs the pack's bundled runtime in a resident
worker, using `uv` with the pack's pinned requirements. Repeated requests in the
same session reuse the loaded model; a one-shot loader remains as fallback.
Nothing listens on a port. The first local request
installs the MLX stack into a `uv` environment, which can take a few minutes.

### Output-head experiments

This pack ties its output to `language_model.model.embed_tokens.as_linear`;
there is no separate `language_model.lm_head`. Its native 2-bit operation
requires the pack's Hadamard transform. Changing a `bits` parameter alone is
not a valid conversion.

`tools/bench-bonsai-head.py` compares native 2-bit with a correctly repacked
4-bit output head **in memory**, leaving input embeddings and installed files
untouched. Run with the pack's `runtime/requirements.txt`, after a local shev
request has created `~/.local/share/shev/models/shev_bonsai_worker.py`:

```sh
uv run --python 3.12 --with-requirements \
  ~/.local/share/shev/models/Ternary-Bonsai-2-2B-mlx-2bit/runtime/requirements.txt \
  python tools/bench-bonsai-head.py
```

With the current deterministic prompt, one local run measured 1.55 ms/head and
669 ms/decision for native 2-bit, versus 2.29 ms/head and 734 ms/decision for
repacked 4-bit. Both scored 6/6 action labels on a tiny smoke test; an earlier
prompt/sampled baseline scored 3/6. This is not evidence of general accuracy,
and the improvement is from the prompt/sampling changes, not from repacking.
Repacking cannot recover precision already lost in the 2-bit checkpoint.
Native 2-bit therefore remains the default. Hardware, prompts and output length
affect these timings; existing explicit provider selections are preserved.

## MCP

Shev's MCP server lets **another assistant read your saved decisions**. It does
not run Bonsai/DeepSeek or edit your files. No API key, OAuth login, terminal,
network port, or separately started TUI is required: the MCP host starts shev
as a subprocess and talks JSON-RPC over stdin/stdout.

### 1. Choose the history file

Save at least one decision using an explicit file path, for example:

```sh
shev --file "$HOME/shev-responses.json"
```

Inside that TUI run **`/mcp install`** (or `/mcp` → install). This writes a
descriptor under `~/.config/shev/mcp/` and registers the executable and absolute
history path in an existing global OpenCode config, with a private backup.
`/mcp status` reports the registration; `/mcp uninstall` removes shev's own
unchanged registration. It does not delete your decisions.

### 2. Connect OpenCode or Codex

For OpenCode, reconnect **shev** from `/mcps` or restart the client after
installation. Alternatively, register it explicitly from a shell (replace
the executable path if you built from source):

```sh
opencode mcp add shev --global -- "$HOME/.local/bin/shev" --mcp --file "$HOME/shev-responses.json"
opencode mcp list
```

The equivalent OpenCode V2 configuration is:

```json
{
  "mcp": {
    "servers": {
      "shev": {
        "type": "local",
        "command": ["/absolute/path/to/shev", "--mcp", "--file", "/absolute/path/to/responses.json"],
        "enabled": true
      }
    }
  }
}
```

For Codex, add this to `~/.codex/config.toml` (or your `CODEX_HOME` config),
replacing the paths, then restart Codex and check `codex mcp list`:

```toml
[mcp_servers.shev]
command = "/absolute/path/to/shev"
args = ["--mcp", "--file", "/absolute/path/to/responses.json"]
```

Both `shev --mcp` and `shev-cli --mcp` work. For a source build, use
`/absolute/path/to/repository/target/release/shev` after `cargo build --release`.
Do not put shell expressions such as `$(pwd)` inside a JSON/TOML command array.

### 3. Ask the host assistant to use shev

Example prompts in OpenCode/Codex:

- “Use shev's `latest_decision` tool and summarize my most recent decision.”
- “Use shev's `search_decisions` with query `invoice` and limit `10`.”
- “Read `response2` using shev's `get_decision` tool.”

| Tool | Example arguments | Result |
| --- | --- | --- |
| `list_decisions` | `{"limit": 20}` | Most recent records, oldest first |
| `latest_decision` | `{}` | Latest saved record |
| `get_decision` | `{"key": "response2"}` or `{"index": 2}` | One saved record |
| `search_decisions` | `{"query": "invoice", "limit": 10}` | Matching saved decisions |

Resources are `shev://decisions` (all records) and
`shev://decisions/latest` (latest record). Reads use the file afresh, so decisions
saved by an open TUI become available without restarting the MCP process.

**Troubleshooting:** `unknown option --mcp` means the host is launching an old
TUI binary. Update/rebuild the **exact executable in the MCP command** and
reconnect shev. Current releases dispatch `--mcp` before terminal setup. If
connection succeeds but no decisions appear, check that the host and TUI use
the same absolute history path. Starting `shev --mcp` manually will wait quietly
for protocol input; this is normal, not a frozen chat window.

### Reuse existing skills and tools

`/skills` reads existing `SKILL.md` files from Codex, Claude, shared
`~/.agents/skills`, `~/skills`, OpenCode's XDG config directory, and ancestor
project `.opencode`, `.agents`, and `.codex` skill folders, plus local skill
directories configured in OpenCode. It loads guidance
into memory; it does not install, edit, or execute skill scripts.

`/tools` reads MCP registrations from Codex's `config.toml` and OpenCode's
global/project JSON or JSONC, including V2 `mcp.servers`. `CODEX_HOME` and
`XDG_CONFIG_HOME` are respected. Selecting a server connects and lists tools;
selecting a tool shows its argument schema and prepares `/tool server/tool {}`.
Edit the arguments, then enter to call it. Successful results become reference
context for the next decision; `/skills off` unloads them. Supporting context
does not replace the user's original intent in history.

Only tools with `readOnlyHint: true` and without `destructiveHint: true` can
be called. Unknown/mutating tools are blocked, not silently approved. **MCP
annotations are declarations from your trusted server, not a security sandbox**;
only use servers you trust. Starting a configured server can run its startup
code. Calls have time/output limits and cannot request sampling or approvals.
Stdio and streamable HTTP (JSON/SSE) are supported. Configured headers and
environment-based tokens are reused without logging them; host-managed OAuth
credentials are not copied. An OAuth-only server reports an authentication
requirement rather than claiming success.

The top of the right pane names the loaded skill or successfully called tool
and its source. Integration identity is also stored with its decision. Merely
discovering/selecting a server is not reported as a successful tool call.

OpenCode's configured JavaScript plugins are listed with compatibility notices.
**Host-specific plugins cannot run natively in shev**; they require their host
runtime. Their MCP interfaces, where registered, can be used through `/tools`.
shev never rewrites host configuration during discovery or tool use. Explicit
`/mcp install` is separate and backs up configuration before registration.

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
| `SHEV_BONSAI_MAX_TOKENS` | Local worker token cap (default `256`; cold fallback `2048`) |
| `SHEV_BONSAI_STREAM` | `1` enables worker token-delta protocol messages |
| `SHEV_BONSAI_TEMPERATURE` | Local sampling temperature (default `0`, deterministic) |

The footer shows token usage and the last decision's elapsed milliseconds.
The resident worker reports real local token counts; the cold fallback has no
usage metadata and reports zero rather than guessing. The concise summary and
generated response remain alongside the JSON. A measured warm run took about
750 ms versus 2500 ms including initial model loading—not a claim of Jev's
sub-100-ms latency.
Malformed local decisions receive at most two format-correction retries; usage
includes those attempts. Retries can add latency and still fail rather than
silently recording invalid JSON.

## Where things live

| Path | What it is |
| --- | --- |
| `~/.config/shev/config.json` | API key and the selected model (`0600`) |
| `responses.json` | The decisions, appended as `response1`, `response2`, … |
| `~/.local/share/shev/models/` | Downloaded local models |

## License

Apache-2.0. See `LICENSE`.
