//! shev — hand a blob of text to DeepSeek and record the resulting decision
//! as JSON.
//!
//! Flow:
//!   1. onboarding: read `DEEPSEEK_API_KEY`, or read the saved key from
//!      `~/.config/shev/config.json`, or prompt for it once and save it.
//!   2. verification: hit the DeepSeek API with the key and refuse to continue
//!      if the key is not accepted.
//!   3. loop: read one whole message (blank lines included), ask the user what
//!      they want to do with it, hand both to the model, validate the decision
//!      and append it to `responses.json` as `response1`, `response2`, ...
//!
//! The message is judged as a whole — one decision for the entire message, never
//! per sentence or per paragraph.
//!
//! Every decision has these fields:
//!   actionable         : string  — the concrete action, or "" when there is none
//!   take_action        : boolean — whether the action should be taken
//!   confidence         : integer — inside the closed range [34, 100]
//!   generated_response : string  — the reply or next step, written out, driven
//!                                  by what the user said they want to do

use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

pub mod integrations;

const API_BASE: &str = "https://api.deepseek.com";
pub const MODEL: &str = "deepseek-chat";
pub const MIN_CONFIDENCE: i64 = 34;
pub const MAX_CONFIDENCE: i64 = 100;
pub const DEFAULT_RESPONSES_FILE: &str = "responses.json";
pub const DEFAULT_INTENT: &str = "Decide whether this message needs action.";
/// How many earlier decisions the TUI loads into its transcript.
pub const MAX_HISTORY: usize = 20;
/// The summary the model returns is capped at three sentences and 120 words.
pub const MAX_SUMMARY_WORDS: usize = 120;
pub const MAX_SUMMARY_SENTENCES: usize = 3;

const SYSTEM_PROMPT: &str = r#"You are a decision engine. The user hands you one message plus a statement of
what the user wants to do with that message.

Judge the message AS A WHOLE. There is exactly one decision for the entire message.
Never judge it sentence by sentence, line by line or paragraph by paragraph, and never
return a list of decisions.

Reply with a single JSON object and nothing else. It must use exactly these keys:
{
  "actionable": "<one short sentence naming the concrete action the message calls for, or \"\" if there is none>",
  "take_action": <true if an action should be taken, otherwise false>,
  "confidence": <integer between 34 and 100 inclusive: how sure you are of this decision>,
  "summary": "<at most 3 sentences and at most 120 words saying what the user should actually do>",
  "generated_response": "<the reply, or the concrete next step, written out in full so the user can use it as-is>"
}

Rules:
- The whole message is one unit: one actionable, one take_action, one confidence.
- "take_action" is true only when a concrete, useful action follows from the message
  given what the user says they want to do.
- "summary" is the short read: at most 3 sentences, at most 120 words, plain prose,
  telling the user what they should actually do next. No bullet lists, no headings.
- "generated_response" is the deliverable: write the actual reply text, or the actual
  concrete steps to take, taking the user's stated intent as the goal. Write it as
  finished prose the user could send or follow without editing.
- If the user's intent is to do nothing, or the message needs nothing, return "" for
  "summary" and "generated_response", and false for "take_action".
- "confidence" must be an integer inside the closed range [34, 100]. Never below 34,
  never above 100.
- No extra keys, no prose outside the JSON, no markdown fences."#;

/// The lean prompt for the local model. The remote prompt above is long — some
/// four hundred words — and on a small on-device model almost all of the latency
/// is the prompt prefill, so the faster path wants the same rules in far fewer
/// tokens. Same five fields, same one-object contract.
const LOCAL_SYSTEM_PROMPT: &str = r#"You are a decision engine. The user gives one message and what they want to do with it.
Judge the message as a whole, once. Reply with one JSON object and nothing else:
{"actionable":"","take_action":false,"confidence":70,"summary":"","generated_response":""}
Fill in those five fields. take_action is true only for an outstanding action relevant to the user's intent; false when none remains. Do not invent obligations. confidence is an integer 34..100. summary is at most 3 sentences and 120 words. generated_response is the concise reply or next step, ready to use.
No prose outside the JSON, no markdown fences."#;

/// The instruction that wraps the user's message and their stated intent.
const USER_TEMPLATE: &str = "WHAT THE USER WANTS TO DO WITH THIS MESSAGE:\n%INTENT%\n\nTHE MESSAGE, AS A WHOLE:\n<message>\n%MESSAGE%\n</message>";

/// The line-based CLI: read a message, ask for the intent interactively, print
/// the decision and append it to the responses file.
pub fn cli_run() -> Result<(), String> {
    let options = match parse_args()? {
        Some(options) => options,
        None => return Ok(()), // --help was printed
    };

    // Serving MCP replaces the whole CLI: stdin/stdout are a JSON-RPC channel.
    if options.mcp {
        return mcp_serve(&options.responses);
    }

    let interactive = io::stdin().is_terminal();
    let client = http_client()?;

    // Local by default. Only the explicitly selected remote provider needs a key.
    let target = match load_provider() {
        Provider::Bonsai => {
            ensure_bonsai_ready()?;
            Target::bonsai()
        }
        Provider::DeepSeek => Target::deepseek(&obtain_api_key(&client, interactive)?),
    };

    // 3: one message, or a loop over messages.
    if let Some(message) = options.once {
        let intent = options
            .intent
            .clone()
            .unwrap_or_else(|| DEFAULT_INTENT.to_string());
        let decision = request_decision_target(&client, &target, &message, &intent)?;
        store_decision(&options.responses, &decision, &message, &intent)?;
        return Ok(());
    }

    if !interactive {
        // Piped in: the whole of stdin is one message.
        let mut message = String::new();
        io::stdin()
            .read_to_string(&mut message)
            .map_err(|e| format!("could not read stdin: {e}"))?;
        let message = message.trim().to_string();
        if message.is_empty() {
            return Err("no message was piped in on stdin".to_string());
        }
        let intent = options
            .intent
            .clone()
            .unwrap_or_else(|| DEFAULT_INTENT.to_string());
        let decision = request_decision_target(&client, &target, &message, &intent)?;
        store_decision(&options.responses, &decision, &message, &intent)?;
        return Ok(());
    }

    println!(
        "\nshev is ready.\n\
         1. Paste the whole message, then a line containing only EOF (Ctrl-D ends it too).\n\
         \x20  Blank lines inside the message are kept, so paragraphs survive.\n\
         2. You are then asked what you want to do with it.\n\
         The model judges the message as a whole and writes the response for you.\n\
         Decisions are appended to {}.\n\
         Type 'exit' as the first line to quit.\n",
        options.responses.display()
    );

    let mut stdin = io::stdin().lock();
    while let Some(message) = read_message(&mut stdin)? {
        let Some(intent) = read_intent(&mut stdin)? else {
            break;
        };
        match request_decision_target(&client, &target, &message, &intent) {
            Ok(decision) => store_decision(&options.responses, &decision, &message, &intent)?,
            Err(err) => eprintln!("request failed: {err}\n(the message was not recorded)"),
        }
    }

    println!("bye.");
    Ok(())
}

// ---------------------------------------------------------------------------
// onboarding: api key > verification
// ---------------------------------------------------------------------------

/// Resolve a usable API key: env var, then saved config, then an interactive
/// prompt. Every candidate is verified against the API before it is accepted.
fn obtain_api_key(client: &reqwest::blocking::Client, interactive: bool) -> Result<String, String> {
    if let Ok(key) = env::var("DEEPSEEK_API_KEY") {
        let key = key.trim().to_string();
        if !key.is_empty() {
            println!("using DEEPSEEK_API_KEY from the environment");
            match verify_key(client, &key) {
                Ok(()) => {
                    println!("API key verified.");
                    return Ok(key);
                }
                Err(err) => {
                    if !interactive {
                        return Err(format!("DEEPSEEK_API_KEY was rejected: {err}"));
                    }
                    eprintln!("DEEPSEEK_API_KEY was rejected: {err}");
                }
            }
        }
    }

    if let Some(key) = load_saved_key()? {
        println!("found a saved key in {}", config_path().display());
        match verify_key(client, &key) {
            Ok(()) => {
                println!("API key verified.");
                return Ok(key);
            }
            Err(err) => {
                if !interactive {
                    return Err(format!("the saved API key was rejected: {err}"));
                }
                eprintln!("the saved API key was rejected: {err}");
            }
        }
    }

    if !interactive {
        return Err(
            "no API key available. Set DEEPSEEK_API_KEY, or run shev interactively once to save one."
                .to_string(),
        );
    }

    for attempt in 1..=3 {
        let key = read_secret("Paste your DeepSeek API key (input is hidden): ")?;
        if key.is_empty() {
            println!("nothing entered, try again.");
            continue;
        }
        match verify_key(client, &key) {
            Ok(()) => {
                save_key(&key)?;
                println!("API key verified and saved to {}", config_path().display());
                return Ok(key);
            }
            Err(err) => eprintln!("verification failed (attempt {attempt}/3): {err}"),
        }
    }

    Err("could not verify an API key after 3 attempts".to_string())
}

/// Verification: ask the API who we are. A 200 means the key works.
pub fn verify_key(client: &reqwest::blocking::Client, key: &str) -> Result<(), String> {
    let response = client
        .get(format!("{API_BASE}/models"))
        .bearer_auth(key)
        .send()
        .map_err(|e| format!("could not reach {API_BASE}: {e}"))?;

    let status = response.status();
    let body = response.text().unwrap_or_default();
    if status.is_success() {
        return Ok(());
    }
    Err(format!("HTTP {} {}", status.as_u16(), snippet(&body)))
}

pub fn config_path() -> PathBuf {
    let base = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    base.join("shev").join("config.json")
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Where the config lived before the rename to `shev`.
fn legacy_config_path() -> PathBuf {
    let base = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    base.join("deepslop").join("config.json")
}

/// What to read: the new path, or the pre-rename one when the new is absent.
/// Writes always go to [`config_path`], so the first save migrates the file.
fn read_config_path() -> PathBuf {
    let path = config_path();
    if path.exists() {
        return path;
    }
    let legacy = legacy_config_path();
    if legacy.exists() { legacy } else { path }
}

pub fn load_saved_key() -> Result<Option<String>, String> {
    let path = read_config_path();
    if !path.exists() {
        return Ok(None);
    }
    let raw =
        fs::read_to_string(&path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let parsed: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
    let key = parsed
        .get("api_key")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    Ok((!key.is_empty()).then_some(key))
}

pub fn save_key(key: &str) -> Result<(), String> {
    let mut map = load_config_map();
    map.insert("api_key".to_string(), Value::String(key.to_string()));
    write_config_map(&map)
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) {}

/// Read a line from the terminal without echoing it back.
fn read_secret(prompt: &str) -> Result<String, String> {
    print!("{prompt}");
    io::stdout().flush().map_err(|e| e.to_string())?;

    let _ = std::process::Command::new("stty").arg("-echo").status();
    let mut line = String::new();
    let read = io::stdin().read_line(&mut line);
    let _ = std::process::Command::new("stty").arg("echo").status();
    println!();

    read.map_err(|e| format!("could not read the key: {e}"))?;
    Ok(line.trim().to_string())
}

// ---------------------------------------------------------------------------
// the model call
// ---------------------------------------------------------------------------

pub fn http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .user_agent(concat!("shev/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("could not build the HTTP client: {e}"))
}

/// Where a request goes: which endpoint, which model, which key.
pub struct Target {
    pub provider: Provider,
    pub chat_url: String,
    pub models_url: String,
    pub model: String,
    pub key: String,
}

impl Target {
    pub fn deepseek(key: &str) -> Self {
        Self {
            provider: Provider::DeepSeek,
            chat_url: format!("{API_BASE}/chat/completions"),
            models_url: format!("{API_BASE}/models"),
            model: DEEPSEEK_MODEL.to_string(),
            key: key.to_string(),
        }
    }

    pub fn bonsai() -> Self {
        Self {
            provider: Provider::Bonsai,
            // no network server: the local pack runs in a resident worker
            chat_url: String::new(),
            models_url: String::new(),
            model: bonsai_dir().display().to_string(),
            key: "none".to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        self.provider == Provider::Bonsai
    }
}

/// Send one whole message plus the user's stated intent to DeepSeek and return
/// the validated decision object.
pub fn request_decision(
    client: &reqwest::blocking::Client,
    key: &str,
    message: &str,
    intent: &str,
) -> Result<Value, String> {
    request_decision_target(client, &Target::deepseek(key), message, intent)
}

/// The same, against whichever endpoint the target names.
pub fn request_decision_target(
    client: &reqwest::blocking::Client,
    target: &Target,
    message: &str,
    intent: &str,
) -> Result<Value, String> {
    let user_content = USER_TEMPLATE
        .replace("%INTENT%", intent)
        .replace("%MESSAGE%", message);

    // the local pack has no server to post to: run its own loader for this one
    // request and read the decision back out of the subprocess
    if target.is_local() {
        return request_decision_bonsai(&user_content);
    }

    let payload = json!({
        "model": target.model,
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_content },
        ],
        "stream": false,
        "response_format": { "type": "json_object" },
        "temperature": 0.2,
        "max_tokens": 2048,
    });

    let mut last_error = String::from("no attempt was made");

    for attempt in 1..=3u32 {
        let sent = client
            .post(&target.chat_url)
            .bearer_auth(&target.key)
            .json(&payload)
            .send();

        match sent {
            Ok(response) => {
                let status = response.status();
                let body = response.text().unwrap_or_default();

                if status.is_success() {
                    let parsed: Value = serde_json::from_str(&body).map_err(|e| {
                        format!("{} returned unparseable JSON: {e}", target.provider.slug())
                    })?;
                    let content = parsed
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            format!(
                                "{} response had no message content: {}",
                                target.provider.slug(),
                                snippet(&body)
                            )
                        })?;
                    let mut decision = parse_decision(content)?;
                    // carry the token accounting alongside the decision so the
                    // TUI can show a running total without a second request
                    if let Some(object) = decision.as_object_mut() {
                        object.insert("usage".to_string(), normalize_usage(parsed.get("usage")));
                    }
                    return Ok(decision);
                }

                if status.as_u16() == 401 || status.as_u16() == 403 {
                    return Err(format!(
                        "the API key was rejected (HTTP {}): {}",
                        status.as_u16(),
                        snippet(&body)
                    ));
                }

                last_error = format!("HTTP {} {}", status.as_u16(), snippet(&body));
                if status.is_server_error() || status.as_u16() == 429 {
                    sleep(Duration::from_secs(u64::from(attempt) * 2));
                    continue;
                }
                return Err(last_error);
            }
            Err(err) => {
                last_error = format!("request failed: {err}");
                sleep(Duration::from_secs(u64::from(attempt) * 2));
            }
        }
    }

    Err(format!(
        "giving up after 3 attempts, last error: {last_error}"
    ))
}

/// Answer one decision with the local Bonsai pack. Prefers the warm engine: the
/// model is loaded once and kept resident, so every decision after the first
/// skips the reload that dominates the one-shot path. If the engine cannot
/// start, or dies mid-request, this falls back to the cold one-shot loader.
fn request_decision_bonsai(user_content: &str) -> Result<Value, String> {
    local_decision_with_retry(user_content, |prompt| match warm_engine_decide(prompt) {
        Ok(reply) => Ok(reply),
        Err(warm) => Ok(LocalReply {
            text: bonsai_oneshot(prompt).map_err(|cold| {
                format!("local worker failed ({warm}); fallback failed ({cold})")
            })?,
            usage: json!({"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}),
        }),
    })
}

fn local_decision_with_retry(
    user_content: &str,
    mut run: impl FnMut(&str) -> Result<LocalReply, String>,
) -> Result<Value, String> {
    let mut prompt = format!("{LOCAL_SYSTEM_PROMPT}\n\n{user_content}");
    let mut usage = json!({"prompt_tokens":0,"completion_tokens":0,"total_tokens":0});
    let mut last_error = String::new();
    for attempt in 0..3 {
        let reply = run(&prompt)?;
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            usage[field] = json!(
                usage[field]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_add(reply.usage[field].as_u64().unwrap_or(0))
            );
        }
        match parse_decision(&reply.text) {
            Ok(mut decision) => {
                decision["model"] = json!(BONSAI_REPO);
                decision["usage"] = usage;
                return Ok(decision);
            }
            Err(error) => last_error = error,
        }
        prompt.push_str(&format!("\nFormat correction {}: return strictly valid JSON. Include commas between all five fields and use double-quoted strings. Keep every string short so the complete object fits in the token budget.", attempt + 1));
    }
    Err(format!(
        "local model returned an invalid decision after 3 attempts: {last_error}"
    ))
}

/// The original cold path: run the pack's bundled loader in a subprocess for a
/// single request. Kept as the fallback for a machine where the resident engine
/// cannot start.
fn bonsai_oneshot(prompt: &str) -> Result<String, String> {
    let argv = bonsai_run_command()?;
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| "empty command".to_string())?;

    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start the bonsai runner ({program}): {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .map_err(|e| format!("could not send the prompt to the bonsai runner: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("the bonsai runner failed: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "the bonsai runner exited ({}) — {}",
            output.status,
            snippet(&String::from_utf8_lossy(&output.stderr))
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// ---------------------------------------------------------------------------
// the warm local engine: one resident process, many decisions
// ---------------------------------------------------------------------------

/// A resident local engine: one Python process that holds the model in memory
/// and answers one JSON request per line. The point is the reload it does *not*
/// pay — the model load is the single biggest cost of a local decision.
struct LocalEngine {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: io::BufReader<std::process::ChildStdout>,
}

struct LocalReply {
    text: String,
    usage: Value,
}

impl LocalEngine {
    fn start() -> Result<Self, String> {
        let argv = bonsai_worker_command()?;
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| "empty command".to_string())?;

        let mut child = std::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("could not start the local engine ({program}): {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "the local engine had no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "the local engine had no stdout".to_string())?;

        // the worker announces itself once the model is loaded
        let mut stdout = io::BufReader::new(stdout);
        let mut ready = String::new();
        BufRead::read_line(&mut stdout, &mut ready)
            .map_err(|e| format!("the local engine did not start: {e}"))?;
        if !ready.contains("ready") {
            let _ = child.kill();
            return Err(format!(
                "the local engine did not report ready: {}",
                snippet(&ready)
            ));
        }

        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    fn decide(&mut self, prompt: &str) -> Result<LocalReply, String> {
        let request = json!({ "prompt": prompt });
        self.stdin
            .write_all(request.to_string().as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| format!("could not reach the local engine: {e}"))?;

        loop {
            let mut line = String::new();
            let read = BufRead::read_line(&mut self.stdout, &mut line)
                .map_err(|e| format!("could not read from the local engine: {e}"))?;
            if read == 0 {
                return Err("the local engine closed its output".to_string());
            }
            let line = line.trim();
            if line.is_empty() {
                continue; // a streaming delta with no payload
            }
            let value: Value = serde_json::from_str(line)
                .map_err(|e| format!("the local engine sent bad JSON: {e}"))?;
            if let Some(text) = value.get("text").and_then(Value::as_str) {
                return Ok(LocalReply {
                    text: text.to_string(),
                    usage: value.get("usage").cloned().unwrap_or_else(
                        || json!({"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}),
                    ),
                });
            }
            if let Some(err) = value.get("error").and_then(Value::as_str) {
                return Err(format!("the local engine failed: {err}"));
            }
            // otherwise it is a `delta` line: keep reading until `text`
        }
    }
}

impl Drop for LocalEngine {
    fn drop(&mut self) {
        // closing stdin ends the worker's loop; kill is the belt to that braces
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The one resident engine, behind a lock. `None` until the first local request.
fn local_engine_slot() -> &'static Mutex<Option<LocalEngine>> {
    static SLOT: OnceLock<Mutex<Option<LocalEngine>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Answer one prompt from the resident engine, starting it on first use and
/// restarting it once if it died.
fn warm_engine_decide(prompt: &str) -> Result<LocalReply, String> {
    let slot = local_engine_slot();
    let mut guard = slot
        .lock()
        .map_err(|_| "the local engine lock was poisoned".to_string())?;

    if guard.is_none() {
        *guard = Some(LocalEngine::start()?);
    }

    // drop the whole engine on failure so the retry gets a clean process
    let first = guard
        .as_mut()
        .ok_or_else(|| "the local engine vanished".to_string())?
        .decide(prompt);
    match first {
        Ok(text) => Ok(text),
        Err(first_error) => {
            *guard = None;
            let mut fresh = LocalEngine::start()?;
            let text = fresh.decide(prompt).map_err(|second| {
                format!("the local engine failed twice ({first_error}; {second})")
            })?;
            *guard = Some(fresh);
            Ok(text)
        }
    }
}

/// Normalise whatever the model said into the guaranteed fields.
pub fn parse_decision(raw: &str) -> Result<Value, String> {
    let cleaned = strip_code_fences(raw);
    let parsed: Value =
        serde_json::from_str(&cleaned).map_err(|e| format!("model reply was not JSON: {e}"))?;

    if !parsed.is_object() {
        return Err("model reply was not a JSON object".to_string());
    }

    let actionable = match lookup(&parsed, "actionable") {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };

    let take_action = match lookup(&parsed, "take_action") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "y" | "1" => true,
            "false" | "no" | "n" | "0" | "" => false,
            other => return Err(format!("take_action was not a boolean: {other:?}")),
        },
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(other) => return Err(format!("take_action had an unsupported type: {other}")),
        None => return Err("the model reply is missing the key \"take_action\"".to_string()),
    };

    let raw_confidence = match lookup(&parsed, "confidence") {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        Some(other) => return Err(format!("confidence had an unsupported type: {other}")),
        None => return Err("the model reply is missing the key \"confidence\"".to_string()),
    }
    .ok_or_else(|| "confidence was not a number".to_string())?;

    // 34 < x < 100 "including both" -> a closed range of [34, 100].
    let rounded = raw_confidence.round() as i64;
    let confidence = rounded.clamp(MIN_CONFIDENCE, MAX_CONFIDENCE);
    if confidence != rounded {
        eprintln!(
            "note: model returned confidence {raw_confidence}, clamped into [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}]"
        );
    }

    // The deliverable. Tolerate the model answering as a list of steps.
    let generated_response =
        match lookup(&parsed, "generated_response").or_else(|| lookup(&parsed, "response")) {
            Some(Value::String(s)) => s.trim().to_string(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| match item {
                    Value::String(s) => s.trim().to_string(),
                    other => other.to_string(),
                })
                .filter(|item| !item.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        };

    let summary = match lookup(&parsed, "summary") {
        Some(Value::String(s)) => clamp_summary(s),
        Some(Value::Array(items)) => clamp_summary(
            &items
                .iter()
                .map(|item| match item {
                    Value::String(s) => s.trim().to_string(),
                    other => other.to_string(),
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        Some(Value::Null) | None => String::new(),
        Some(other) => clamp_summary(&other.to_string()),
    };

    Ok(json!({
        "actionable": actionable,
        "take_action": take_action,
        "confidence": confidence,
        "summary": summary,
        "generated_response": generated_response,
    }))
}

/// Keep a summary inside three sentences and 120 words, whatever the model sent.
pub fn clamp_summary(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }

    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
            if sentences.len() == MAX_SUMMARY_SENTENCES {
                break;
            }
        }
    }
    if sentences.len() < MAX_SUMMARY_SENTENCES {
        let tail = current.trim();
        if !tail.is_empty() {
            sentences.push(tail.to_string());
        }
    }

    let mut out = sentences.join(" ");
    let words: Vec<&str> = out.split_whitespace().collect();
    if words.len() > MAX_SUMMARY_WORDS {
        out = words[..MAX_SUMMARY_WORDS].join(" ");
        out.push('…');
    }
    out
}

/// Reduce whatever `usage` the API sent into the three integers we care about.
fn normalize_usage(usage: Option<&Value>) -> Value {
    let number = |key: &str| -> u64 {
        usage
            .and_then(|u| lookup(u, key))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let prompt = number("prompt_tokens");
    let completion = number("completion_tokens");
    let total = match usage
        .and_then(|u| lookup(u, "total_tokens"))
        .and_then(Value::as_u64)
    {
        Some(total) if total > 0 => total,
        _ => prompt + completion,
    };
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    })
}

/// Total tokens a stored record accounts for, for the running counter.
pub fn record_tokens(record: &Value) -> u64 {
    let usage = record.get("usage");
    let number = |key: &str| {
        usage
            .and_then(|u| u.get(key))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let total = number("total_tokens");
    if total > 0 {
        total
    } else {
        number("prompt_tokens") + number("completion_tokens")
    }
}

/// `100000` reads as `100k`, `1500` as `1.5k`, `2_400_000` as `2.4M`.
pub fn format_tokens(tokens: u64) -> String {
    if tokens < 1_000 {
        return tokens.to_string();
    }
    if tokens < 1_000_000 {
        let thousands = tokens as f64 / 1_000.0;
        if thousands >= 100.0 {
            format!("{}k", thousands.round() as u64)
        } else if thousands >= 10.0 {
            format!("{thousands:.0}k")
        } else {
            format!("{thousands:.1}k")
        }
    } else {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    }
}

fn lookup<'a>(object: &'a Value, key: &str) -> Option<&'a Value> {
    let map = object.as_object()?;
    if let Some(value) = map.get(key) {
        return Some(value);
    }
    map.iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v)
}

fn strip_code_fences(raw: &str) -> String {
    let trimmed = raw.trim();
    let without_open = match trimmed.strip_prefix("```") {
        Some(rest) => match rest.find('\n') {
            Some(newline) => &rest[newline + 1..],
            None => rest,
        },
        None => trimmed,
    };
    let without_close = without_open
        .strip_suffix("```")
        .unwrap_or(without_open)
        .trim();
    without_close.to_string()
}

fn snippet(body: &str) -> String {
    let one_line = body.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out: String = one_line.chars().take(300).collect();
    if one_line.chars().count() > 300 {
        out.push('…');
    }
    if out.is_empty() {
        out.push_str("(empty response)");
    }
    out
}

// ---------------------------------------------------------------------------
// recording decisions: {"response1": {..}, "response2": {..}}
// ---------------------------------------------------------------------------

/// The record that gets written for one decision, minus its `responseN` key.
pub fn build_record(decision: &Value, message: &str, intent: &str) -> Value {
    json!({
        "actionable": decision.get("actionable").cloned().unwrap_or(Value::String(String::new())),
        "take_action": decision.get("take_action").cloned().unwrap_or(Value::Bool(false)),
        "confidence": decision.get("confidence").cloned().unwrap_or(json!(MIN_CONFIDENCE)),
        "summary": decision.get("summary").cloned().unwrap_or(Value::String(String::new())),
        "generated_response": decision.get("generated_response").cloned().unwrap_or(Value::String(String::new())),
        "usage": decision.get("usage").cloned().unwrap_or(Value::Null),
        "user_intent": intent,
        "input": message,
        "model": decision.get("model").and_then(Value::as_str).unwrap_or(MODEL),
        "created_at": unix_timestamp(),
    })
}

/// Append one record to the responses file under the next free `responseN` key.
/// Returns the key it was filed under. Prints nothing, so a TUI can call it.
pub fn append_record(path: &Path, record: &Value) -> Result<String, String> {
    let mut root: Map<String, Value> = if path.exists() {
        let raw = fs::read_to_string(path)
            .map_err(|e| format!("could not read {}: {e}", path.display()))?;
        if raw.trim().is_empty() {
            Map::new()
        } else {
            match serde_json::from_str::<Value>(&raw) {
                Ok(Value::Object(map)) => map,
                Ok(_) => {
                    return Err(format!(
                        "{} does not contain a JSON object, refusing to overwrite it",
                        path.display()
                    ));
                }
                Err(e) => {
                    return Err(format!(
                        "{} is not valid JSON ({e}), refusing to overwrite it",
                        path.display()
                    ));
                }
            }
        }
    } else {
        Map::new()
    };

    let next_index = root
        .keys()
        .filter_map(|key| key.strip_prefix("response"))
        .filter_map(|index| index.parse::<u64>().ok())
        .max()
        .unwrap_or(0)
        + 1;
    let key = format!("response{next_index}");

    root.insert(key.clone(), record.clone());

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| format!("could not encode the responses file: {e}"))?;
    fs::write(path, format!("{body}\n"))
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;

    Ok(key)
}

/// Delete only the chosen saved history record. Never renumbers other records.
pub fn delete_record(path: &Path, key: &str) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    let mut records = history_map(path)?;
    if records.remove(key).is_none() {
        return Ok(false);
    }
    write_history(path, &records)?;
    Ok(true)
}

/// Clear saved chat data, not configuration, credentials, or downloaded models.
pub fn clear_records(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let mut records = history_map(path)?; // Refuse malformed files; preserve non-chat metadata.
    records.retain(|key, _| {
        key.strip_prefix("response")
            .and_then(|n| n.parse::<u64>().ok())
            .is_none()
    });
    write_history(path, &records)
}

fn history_map(path: &Path) -> Result<Map<String, Value>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("could not read history: {e}"))?;
    if raw.trim().is_empty() {
        return Ok(Map::new());
    }
    serde_json::from_str::<Value>(&raw)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .ok_or_else(|| "history is not a valid JSON object; left unchanged".into())
}

fn write_history(path: &Path, records: &Map<String, Value>) -> Result<(), String> {
    let resolved =
        fs::canonicalize(path).map_err(|e| format!("could not resolve history file: {e}"))?;
    let path = resolved.as_path(); // Update the target, not an existing user symlink.
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = parent.join(format!(".shev-history-{}-{stamp}.tmp", std::process::id()));
    let mut created = false;
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| format!("could not prepare history update: {e}"))?;
        created = true;
        if let Ok(metadata) = fs::metadata(path) {
            fs::set_permissions(&temporary, metadata.permissions()).map_err(|e| e.to_string())?;
        }
        let body = serde_json::to_string_pretty(records).map_err(|e| e.to_string())?;
        writeln!(file, "{body}")
            .and_then(|()| file.sync_all())
            .map_err(|e| format!("could not write history: {e}"))?;
        fs::rename(&temporary, path).map_err(|e| format!("could not replace history: {e}"))
    })();
    if result.is_err() && created {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn store_decision(
    path: &Path,
    decision: &Value,
    message: &str,
    intent: &str,
) -> Result<(), String> {
    let record = build_record(decision, message, intent);
    let key = append_record(path, &record)?;

    println!("\n--- {key} ---");
    let summary = record["summary"].as_str().unwrap_or_default();
    if !summary.is_empty() {
        println!("summary: {summary}");
    }
    let generated = record["generated_response"].as_str().unwrap_or_default();
    if generated.is_empty() {
        println!("generated_response: (empty — nothing to send)");
    } else {
        println!("generated_response:\n{generated}");
    }
    let tokens = record_tokens(&record);
    if tokens > 0 {
        println!("\ntokens: {} ({tokens})", format_tokens(tokens));
    }
    println!(
        "\njson:\n{}",
        serde_json::to_string_pretty(&record).unwrap()
    );
    println!("appended to {}\n", path.display());
    Ok(())
}

/// Put text on the system clipboard. Falls back to the OSC 52 escape sequence,
/// which works in terminals that allow it even without a clipboard utility.
pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    use std::process::{Command, Stdio};

    let candidates: [&[&str]; 4] = [
        &["pbcopy"],
        &["wl-copy"],
        &["xclip", "-selection", "clipboard"],
        &["termux-clipboard-set"],
    ];

    let mut attempted = Vec::new();
    for command in candidates {
        let spawned = Command::new(command[0])
            .args(&command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        let Ok(mut child) = spawned else {
            attempted.push(command[0]);
            continue;
        };
        let wrote = child
            .stdin
            .as_mut()
            .map(|stdin| stdin.write_all(text.as_bytes()).is_ok())
            .unwrap_or(false);
        if wrote && child.wait().map(|status| status.success()).unwrap_or(false) {
            return Ok(());
        }
        attempted.push(command[0]);
    }

    // Last resort: OSC 52. Terminals that support it put the text on the
    // clipboard; the ones that do not simply ignore the sequence.
    let encoded = base64_encode(text.as_bytes());
    let sequence = format!("\x1b]52;c;{encoded}\x07");
    let mut stdout = io::stdout();
    stdout
        .write_all(sequence.as_bytes())
        .and_then(|_| stdout.flush())
        .map_err(|e| format!("could not reach a clipboard (tried {attempted:?}): {e}"))
}

fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let first = chunk[0] as u32;
        let second = *chunk.get(1).unwrap_or(&0) as u32;
        let third = *chunk.get(2).unwrap_or(&0) as u32;
        let triple = (first << 16) | (second << 8) | third;
        out.push(TABLE[(triple >> 18) as usize & 0x3f] as char);
        out.push(TABLE[(triple >> 12) as usize & 0x3f] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(triple >> 6) as usize & 0x3f] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[triple as usize & 0x3f] as char
        } else {
            '='
        });
    }
    out
}

/// The current git branch of `directory`, if it is inside a repository.
pub fn git_branch(directory: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args([
            "-C",
            &directory.to_string_lossy(),
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!branch.is_empty()).then_some(branch)
}

pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------

/// Read one whole message. Everything is kept verbatim, including blank lines
/// between paragraphs. A line containing only `EOF` ends it, as does Ctrl-D, so
/// the message is judged as one unit rather than one paragraph at a time.
fn read_message(stdin: &mut impl BufRead) -> Result<Option<String>, String> {
    println!("Paste the message (a line with only EOF ends it, Ctrl-D works too):");
    let mut buffer = String::new();

    loop {
        let mut line = String::new();
        let read = stdin
            .read_line(&mut line)
            .map_err(|e| format!("could not read input: {e}"))?;
        if read == 0 {
            break; // EOF from the terminal
        }
        let line = line.trim_end_matches(['\n', '\r']);

        if buffer.is_empty() && matches!(line.trim(), "exit" | "quit" | ":q") {
            return Ok(None);
        }
        if line.trim().eq_ignore_ascii_case("eof") {
            break;
        }
        buffer.push_str(line);
        buffer.push('\n');
    }

    let trimmed = buffer.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed.to_string()))
}

/// Ask the user what they want to do with the message. This is what the model
/// writes its response against.
fn read_intent(stdin: &mut impl BufRead) -> Result<Option<String>, String> {
    println!("What do you want to do with it? (press Enter for the default)");
    let mut line = String::new();
    let read = stdin
        .read_line(&mut line)
        .map_err(|e| format!("could not read input: {e}"))?;
    if read == 0 {
        return Ok(None); // EOF
    }
    let intent = line.trim();
    let intent = if intent.is_empty() {
        DEFAULT_INTENT.to_string()
    } else {
        intent.to_string()
    };
    println!("intent: {intent}\n");
    Ok(Some(intent))
}

// ---------------------------------------------------------------------------
// cli
// ---------------------------------------------------------------------------

struct Options {
    responses: PathBuf,
    once: Option<String>,
    intent: Option<String>,
    mcp: bool,
}

fn parse_args() -> Result<Option<Options>, String> {
    let mut responses = env::var_os("SHEV_RESPONSES_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_RESPONSES_FILE));
    let mut once: Option<String> = None;
    let mut intent: Option<String> = env::var("SHEV_INTENT").ok();
    let mut mcp = false;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            "--mcp" => mcp = true,
            "-f" | "--file" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} needs a path argument"))?;
                responses = PathBuf::from(value);
            }
            "-c" | "--once" | "--message" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} needs a message argument"))?;
                once = Some(value);
            }
            "-i" | "--intent" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} needs an intent argument"))?;
                intent = Some(value);
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option {other} (try --help)"));
            }
            other => once = Some(other.to_string()),
        }
    }

    Ok(Some(Options {
        responses,
        once,
        intent,
        mcp,
    }))
}

fn print_help() {
    println!(
        "shev — hand a whole message and your intent to Bonsai (default) or DeepSeek, get back a decision and a response\n\n\
         USAGE:\n\
         \x20   shev                          interactive loop\n\
         \x20   shev \"some message\"            judge one message and exit\n\
         \x20   echo \"some message\" | shev    judge one piped message and exit\n\n\
         \x20   shev --mcp                     serve the decisions over MCP (stdio)\n\n\
         OPTIONS:\n\
         \x20   -f, --file <path>      where decisions are appended (default {DEFAULT_RESPONSES_FILE})\n\
         \x20   -c, --once <message>   judge one message and exit\n\
         \x20   -i, --intent <text>    what you want to do with it\n\
         \x20   --mcp                  speak MCP on stdin/stdout instead of the CLI\n\
         \x20   -h, --help             show this help\n\n\
         ENVIRONMENT:\n\
         \x20   DEEPSEEK_API_KEY         api key, used before the saved config\n\
         \x20   SHEV_RESPONSES_FILE  same as --file\n\
         \x20   SHEV_INTENT          same as --intent\n\n\
         Decisions look like:\n\
         \x20   {{ \"actionable\": \"...\", \"take_action\": true, \"confidence\": 87,\n\
         \x20     \"generated_response\": \"...\" }}"
    );
}

// ---------------------------------------------------------------------------
// skills, plugins and tools: reuse what Codex and OpenCode already have
// ---------------------------------------------------------------------------

/// One thing shev can load as extra guidance: a skill, a plugin, or an MCP tool
/// that is already installed for Codex or OpenCode. shev only reads these; it
/// never edits them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    /// "skill", "plugin" or "tool".
    pub kind: String,
    pub name: String,
    pub description: String,
    /// The file holding the guidance (a SKILL.md), when there is one.
    pub path: Option<PathBuf>,
    /// Where it came from, for the label.
    pub source: String,
}

/// The folders Codex and OpenCode keep their skills in. Only the ones that
/// exist are returned, so a machine with none simply discovers nothing.
pub fn capability_roots() -> Vec<(String, PathBuf)> {
    let home = home_dir();
    let config = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".config"));
    let codex = env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".codex"));
    let mut candidates = vec![
        ("opencode", config.join("opencode/skills")),
        ("opencode", config.join("opencode/skill")),
        ("claude", home.join(".claude/skills")),
        ("codex", codex.join("skills")),
        ("shared", home.join(".agents/skills")),
        ("shared", home.join("skills")),
    ];
    if let Ok(cwd) = env::current_dir() {
        for ancestor in cwd.ancestors() {
            for relative in [
                ".opencode/skills",
                ".opencode/skill",
                ".agents/skills",
                ".codex/skills",
            ] {
                candidates.push(("project", ancestor.join(relative)));
            }
        }
    }
    for config_file in opencode_configs()
        .into_iter()
        .chain(opencode_project_configs())
    {
        let Some(value) = fs::read_to_string(&config_file)
            .ok()
            .and_then(|text| parse_jsonc(&text))
        else {
            continue;
        };
        let Some(skills) = value.get("skills") else {
            continue;
        };
        if let Some(paths) = skills
            .as_array()
            .or_else(|| skills.get("paths").and_then(Value::as_array))
        {
            for path in paths.iter().filter_map(Value::as_str) {
                if path.contains("://") {
                    continue;
                } // Never download instructions during discovery.
                let path = if let Some(relative) = path.strip_prefix("~/") {
                    home.join(relative)
                } else {
                    let path = PathBuf::from(path);
                    if path.is_absolute() {
                        path
                    } else {
                        config_file.parent().unwrap_or(Path::new(".")).join(path)
                    }
                };
                candidates.push(("configured", path));
            }
        }
    }
    candidates
        .into_iter()
        .filter(|(_, path)| path.is_dir())
        .map(|(source, path)| (source.to_string(), path))
        .collect()
}

/// Read `name:` and `description:` out of a SKILL.md front matter block.
fn parse_front_matter(text: &str) -> (Option<String>, Option<String>) {
    let mut name = None;
    let mut description = None;
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (None, None);
    }
    for line in lines {
        let line = line.trim();
        if line == "---" {
            break;
        }
        let unquote = |value: &str| {
            value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string()
        };
        if let Some(value) = line.strip_prefix("name:") {
            name = Some(unquote(value));
        } else if let Some(value) = line.strip_prefix("description:") {
            description = Some(unquote(value));
        }
    }
    (name, description)
}

/// Strip a leading `---` front matter block, leaving the body.
pub fn strip_front_matter(text: &str) -> String {
    let mut in_body = false;
    let mut started = false;
    let mut body = String::new();
    for (index, line) in text.lines().enumerate() {
        if !started {
            if index == 0 && line.trim() == "---" {
                started = true;
                continue;
            }
            break;
        }
        if !in_body {
            if line.trim() == "---" {
                in_body = true;
            }
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    if in_body {
        body.trim_start_matches('\n').to_string()
    } else {
        text.to_string()
    }
}

/// Best-effort parse of JSON with `//` and `/* */` comments and trailing commas,
/// which is how OpenCode writes its config. Strict JSON is a subset, so this
/// reads both. Read-only and tolerant: returns `None` rather than failing a scan.
fn parse_jsonc(text: &str) -> Option<Value> {
    let chars: Vec<char> = text.chars().collect();
    let mut stripped = String::with_capacity(text.len());
    let mut i = 0;
    let mut in_string = false;
    let mut escaped = false;
    while i < chars.len() {
        let ch = chars[i];
        if in_string {
            stripped.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if ch == '"' {
            in_string = true;
            stripped.push(ch);
            i += 1;
            continue;
        }
        if ch == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if ch == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }
        stripped.push(ch);
        i += 1;
    }

    // drop the commas that the removed comments left dangling before a closer
    let mut cleaned = String::with_capacity(stripped.len());
    let chars: Vec<char> = stripped.chars().collect();
    let mut i = 0;
    let mut in_string = false;
    let mut escaped = false;
    while i < chars.len() {
        let ch = chars[i];
        if in_string {
            cleaned.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if ch == '"' {
            in_string = true;
            cleaned.push(ch);
            i += 1;
            continue;
        }
        if ch == ',' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if matches!(chars.get(j), Some(&'}') | Some(&']')) {
                i += 1; // the trailing comma; leave it out
                continue;
            }
        }
        cleaned.push(ch);
        i += 1;
    }

    serde_json::from_str(&cleaned).ok()
}

/// Everything loadable: the skills under the Codex/OpenCode roots, plus the MCP
/// servers registered for OpenCode (those are the "tools").
pub fn discover_capabilities() -> Vec<Capability> {
    let mut out: Vec<Capability> = Vec::new();

    for (source, root) in capability_roots() {
        let Ok(entries) = fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let file = dir.join("SKILL.md");
            if !file.is_file() {
                continue;
            }
            let Ok(text) = fs::read_to_string(&file) else {
                continue;
            };
            let (name, description) = parse_front_matter(&text);
            let name = name
                .or_else(|| dir.file_name().map(|n| n.to_string_lossy().to_string()))
                .unwrap_or_default();
            out.push(Capability {
                kind: "skill".to_string(),
                name,
                description: description.unwrap_or_default(),
                path: Some(file),
                source: source.clone(),
            });
        }
    }

    for server in integrations::servers() {
        out.push(Capability {
            kind: "tool".to_string(),
            name: server.name,
            description: "MCP server · enter to inspect read-only tools".to_string(),
            path: None,
            source: server.source,
        });
    }
    for path in opencode_configs()
        .into_iter()
        .chain(opencode_project_configs())
    {
        let Some(value) = fs::read_to_string(&path)
            .ok()
            .and_then(|text| parse_jsonc(&text))
        else {
            continue;
        };
        if let Some(plugins) = value
            .get("plugins")
            .or_else(|| value.get("plugin"))
            .and_then(Value::as_array)
        {
            for plugin in plugins {
                if let Some(name) = plugin
                    .as_str()
                    .or_else(|| plugin.get("package").and_then(Value::as_str))
                {
                    out.push(Capability {
                        kind: "plugin".into(),
                        name: name.to_string(),
                        description: "Requires OpenCode host runtime · not a native shev plugin"
                            .into(),
                        path: None,
                        source: "opencode".into(),
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| (&a.kind, &a.name).cmp(&(&b.kind, &b.name)));
    out.dedup_by(|a, b| a.kind == b.kind && a.name == b.name);
    out
}

/// The guidance text of a capability: the SKILL.md body, front matter stripped.
/// Read-only — shev never writes back to the source.
pub fn capability_guidance(capability: &Capability) -> Result<String, String> {
    let Some(path) = &capability.path else {
        return Ok(String::new());
    };
    let text =
        fs::read_to_string(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    Ok(strip_front_matter(&text).trim().to_string())
}

// ---------------------------------------------------------------------------
// models: a remote DeepSeek endpoint, or a local MLX model
// ---------------------------------------------------------------------------

/// The DeepSeek model.
pub const DEEPSEEK_MODEL: &str = MODEL;
/// The local MLX model pack, run on Apple Silicon through its bundled loader.
/// Stock mlx-lm/mlx-vlm cannot load these weights, so there is no server to hit:
/// the pack's own runtime stays resident between requests.
pub const BONSAI_REPO: &str = "prism-ml/Ternary-Bonsai-2-2B-mlx-2bit";
pub const BONSAI_DIR_NAME: &str = "Ternary-Bonsai-2-2B-mlx-2bit";
/// How many tokens the local model may generate for one decision.
pub const BONSAI_MAX_TOKENS: u32 = 256;

/// Which model backs a request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    /// Remote DeepSeek, needs an API key.
    DeepSeek,
    /// The local ternary Bonsai model, served over MLX on this machine.
    Bonsai,
}

impl Provider {
    pub fn slug(self) -> &'static str {
        match self {
            Provider::DeepSeek => "deepseek",
            Provider::Bonsai => "bonsai",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Provider::DeepSeek => "deepseek (api.deepseek.com)",
            Provider::Bonsai => "bonsai (local MLX pack)",
        }
    }

    pub fn from_slug(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "deepseek" | "ds" | "remote" => Some(Provider::DeepSeek),
            "bonsai" | "local" | "mlx" | "prism" => Some(Provider::Bonsai),
            _ => None,
        }
    }
}

fn load_config_map() -> Map<String, Value> {
    fs::read_to_string(read_config_path())
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

fn write_config_map(map: &Map<String, Value>) -> Result<(), String> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    let body =
        serde_json::to_string_pretty(map).map_err(|e| format!("could not encode config: {e}"))?;
    fs::write(&path, format!("{body}\n"))
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    restrict_permissions(&path);
    Ok(())
}

/// Which model the user last selected. Bonsai unless told otherwise.
pub fn load_provider() -> Provider {
    load_config_map()
        .get("provider")
        .and_then(Value::as_str)
        .and_then(Provider::from_slug)
        .unwrap_or(DEFAULT_PROVIDER)
}

/// Local inference is primary; remote inference must be explicitly selected.
pub const DEFAULT_PROVIDER: Provider = Provider::Bonsai;

pub fn save_provider(provider: Provider) -> Result<(), String> {
    let mut map = load_config_map();
    map.insert(
        "provider".to_string(),
        Value::String(provider.slug().to_string()),
    );
    write_config_map(&map)
}

/// Whether an API key is on disk (without reading its value back out).
pub fn saved_key_present() -> bool {
    load_config_map()
        .get("api_key")
        .and_then(Value::as_str)
        .map(|key| !key.trim().is_empty())
        .unwrap_or(false)
}

/// Forget the stored API key, leaving the rest of the config alone.
pub fn remove_saved_key() -> Result<(), String> {
    let mut map = load_config_map();
    map.remove("api_key");
    write_config_map(&map)
}

/// Where the local models live.
pub fn models_dir() -> PathBuf {
    env::var_os("SHEV_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".local/share/shev/models"))
}

pub fn bonsai_dir() -> PathBuf {
    models_dir().join(BONSAI_DIR_NAME)
}

pub fn bonsai_installed() -> bool {
    let dir = bonsai_dir();
    dir.join("model.safetensors").exists() || dir.join("config.json").exists()
}

pub fn bonsai_size() -> u64 {
    fn dir_size(path: &Path) -> u64 {
        let mut total = 0;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let Ok(meta) = entry.metadata() else { continue };
                if meta.is_dir() {
                    total += dir_size(&entry.path());
                } else {
                    total += meta.len();
                }
            }
        }
        total
    }
    let dir = bonsai_dir();
    if dir.exists() { dir_size(&dir) } else { 0 }
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn program_on_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// The command that downloads the Bonsai model, preferring the tools on PATH.
pub fn bonsai_download_command() -> Result<Vec<String>, String> {
    let dir = bonsai_dir().display().to_string();
    let tail = vec!["download", BONSAI_REPO, "--local-dir", dir.as_str()]
        .into_iter()
        .map(|part| part.to_string())
        .collect::<Vec<_>>();

    let lead: Vec<String> = if program_on_path("hf").is_some() {
        vec!["hf".to_string()]
    } else if program_on_path("huggingface-cli").is_some() {
        vec!["huggingface-cli".to_string()]
    } else if program_on_path("uvx").is_some() {
        vec![
            "uvx".to_string(),
            "--from".to_string(),
            "huggingface_hub".to_string(),
            "hf".to_string(),
        ]
    } else if program_on_path("uv").is_some() {
        vec![
            "uv".to_string(),
            "tool".to_string(),
            "run".to_string(),
            "--from".to_string(),
            "huggingface_hub".to_string(),
            "hf".to_string(),
        ]
    } else {
        return Err(
            "no Hugging Face tooling on PATH — install `uv` (or huggingface_hub) so `hf download` can run"
                .to_string(),
        );
    };

    Ok(lead.into_iter().chain(tail).collect())
}

/// The one-shot runner written into the models dir. The pack ships this loader
/// because stock MLX loaders cannot load its weights, so we drive `runtime/`
/// directly. The prompt arrives on stdin and only the generated text is printed,
/// which keeps the Rust side free of shell quoting.
const BONSAI_RUNNER: &str = r####""""shev's one-shot runner for the Ternary Bonsai MLX pack.

    python shev_bonsai_runner.py <pack-dir>    # prompt on stdin, text on stdout
"""
import json
import os
import sys
from pathlib import Path

pack = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(pack / "runtime"))
max_tokens = int(os.environ.get("SHEV_BONSAI_MAX_TOKENS", "2048"))


def read_json(name, default):
    path = pack / name
    return json.loads(path.read_text()) if path.is_file() else default


def sampler_settings():
    gen = read_json("generation_config.json", {})
    out = {
        "temperature": float(os.environ.get("SHEV_BONSAI_TEMPERATURE", "0")),
        "top_p": gen.get("top_p", 0.95),
        "top_k": gen.get("top_k", 20),
    }
    if gen.get("do_sample") is False:
        out["temperature"] = 0.0
    return out


def render(prompt):
    template = pack / "chat_template.jinja"
    if not template.is_file():
        return prompt
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template.read_text()).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_vision(prompt, settings):
    from mlx_vlm import generate
    from vision_artifact import load_vl_model

    model, processor, _ = load_vl_model(pack)
    result = generate(model, processor, render(prompt), max_tokens=max_tokens, **settings)
    return result if isinstance(result, str) else result.text


def generate_text(prompt, settings):
    import mlx.core as mx
    from artifact import load_model
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from tokenizers import Tokenizer

    model, _ = load_model(pack)
    tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
    eos = read_json("generation_config.json", {}).get("eos_token_id")
    stop = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    stop |= {
        i
        for i in (
            tokenizer.token_to_id("<|im_end|>"),
            tokenizer.token_to_id("<|endoftext|>"),
        )
        if i is not None
    }
    sample = make_sampler(
        temp=settings["temperature"],
        top_p=settings["top_p"],
        top_k=settings["top_k"],
        min_p=settings.get("min_p", 0.0),
    )
    processors = make_logits_processors(repetition_penalty=settings.get("repetition_penalty"))
    ids = tokenizer.encode(render(prompt), add_special_tokens=False).ids
    x = mx.array([ids])
    cache = model.make_cache()
    generated = []
    for _ in range(max_tokens):
        logits = model.lm_head(model.model(x, cache=cache)[:, -1:, :])[:, -1, :]
        for processor in processors:
            logits = processor(mx.array(generated), logits)
        x = sample(logits)[:, None]
        mx.eval(x)
        token = int(x.item())
        if token in stop:
            break
        generated.append(token)
    return tokenizer.decode(generated)


def main():
    prompt = sys.stdin.read()
    settings = sampler_settings()
    if read_json("config.json", {}).get("components", {}).get("vision"):
        print(generate_vision(prompt, settings))
    else:
        print(generate_text(prompt, settings))


if __name__ == "__main__":
    main()
"####;

/// The warm worker: the same loader as `BONSAI_RUNNER`, but it loads the model
/// once and then answers one JSON request per line, so the reload that dominates
/// a one-shot decision is paid once and not again. Generation stops the moment
/// the reply closes its top-level JSON object, which keeps the answer short and
/// guarantees a single object.
const BONSAI_WORKER: &str = r####""""shev's warm worker for the Ternary Bonsai MLX pack.

    python shev_bonsai_worker.py <pack-dir>

Loads the pack's model once, then serves one request per line on stdin:

    {"prompt": "<the whole prompt>"}

and answers one result per line on stdout:

    {"text": "<the decision, a single JSON object>"}
"""
import json
import os
import sys
from pathlib import Path

pack = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(pack / "runtime"))
max_tokens = int(os.environ.get("SHEV_BONSAI_MAX_TOKENS", "256"))
stream = os.environ.get("SHEV_BONSAI_STREAM") == "1"


def read_json(name, default):
    path = pack / name
    return json.loads(path.read_text()) if path.is_file() else default


def sampler_settings():
    gen = read_json("generation_config.json", {})
    out = {
        "temperature": float(os.environ.get("SHEV_BONSAI_TEMPERATURE", "0")),
        "top_p": gen.get("top_p", 0.95),
        "top_k": gen.get("top_k", 20),
    }
    if gen.get("do_sample") is False:
        out["temperature"] = 0.0
    return out


def render(prompt):
    template = pack / "chat_template.jinja"
    if not template.is_file():
        return prompt
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template.read_text()).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


class JsonWatch:
    "Track whether a stream of pieces has closed one top-level {...} object."

    def __init__(self):
        self.depth = 0
        self.seen = False
        self.in_string = False
        self.escaped = False
        self.offset = 0
        self.end = None

    def feed(self, piece):
        for ch in piece:
            self.offset += 1
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
                continue
            if ch == '"':
                self.in_string = True
            elif ch == "{":
                self.depth += 1
                self.seen = True
            elif ch == "}":
                self.depth -= 1
                if self.seen and self.depth == 0:
                    self.end = self.offset
                    return True
        return False


class Engine:
    def __init__(self):
        self.vision = bool(read_json("config.json", {}).get("components", {}).get("vision"))
        if self.vision:
            from vision_artifact import load_vl_model

            self.model, self.processor, _ = load_vl_model(pack)
        else:
            import mlx.core as mx
            from artifact import load_model
            from tokenizers import Tokenizer

            self.mx = mx
            self.model, _ = load_model(pack)
            self.tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
            eos = read_json("generation_config.json", {}).get("eos_token_id")
            self.stop = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
            self.stop |= {
                i
                for i in (
                    self.tokenizer.token_to_id("<|im_end|>"),
                    self.tokenizer.token_to_id("<|endoftext|>"),
                )
                if i is not None
            }

    def generate(self, prompt, on_token=None):
        settings = sampler_settings()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if self.vision:
            from mlx_vlm import stream_generate

            watch = JsonWatch()
            pieces = []
            generation = stream_generate(
                self.model, self.processor, render(prompt), max_tokens=max_tokens, **settings
            )
            try:
                for result in generation:
                    piece = result if isinstance(result, str) else result.text
                    if not isinstance(result, str):
                        self.usage = {"prompt_tokens": result.prompt_tokens,
                                      "completion_tokens": result.generation_tokens,
                                      "total_tokens": result.total_tokens}
                    pieces.append(piece)
                    if on_token:
                        on_token(piece)
                    if watch.feed(piece):
                        break
            finally:
                generation.close()
            text = "".join(pieces)
            return text[:watch.end] if watch.end is not None else text

        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sample = make_sampler(
            temp=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            min_p=settings.get("min_p", 0.0),
        )
        processors = make_logits_processors(repetition_penalty=settings.get("repetition_penalty"))
        ids = self.tokenizer.encode(render(prompt), add_special_tokens=False).ids
        x = self.mx.array([ids])
        cache = self.model.make_cache()
        watch = JsonWatch()
        pieces = []
        for _ in range(max_tokens):
            logits = self.model.lm_head(self.model.model(x, cache=cache)[:, -1:, :])[:, -1, :]
            for processor in processors:
                logits = processor(self.mx.array(ids + pieces), logits)
            x = sample(logits)[:, None]
            self.mx.async_eval(x)  # queue the next step while Python runs
            token = int(x.item())
            if token in self.stop:
                break
            piece = self.tokenizer.decode([token])
            pieces.append(token)
            if on_token:
                on_token(piece)
            if watch.feed(piece):
                break
        self.usage = {"prompt_tokens": len(ids), "completion_tokens": len(pieces),
                      "total_tokens": len(ids) + len(pieces)}
        return self.tokenizer.decode(pieces)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    engine = Engine()
    emit({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            prompt = json.loads(line).get("prompt", "")
            if stream:
                text = engine.generate(
                    prompt,
                    on_token=lambda piece: emit({"delta": piece}),
                )
            else:
                text = engine.generate(prompt)
            emit({"text": text, "usage": engine.usage})
        except Exception as exc:  # keep serving; report the failure
            emit({"error": str(exc)})


if __name__ == "__main__":
    main()
"####;

/// Where the generated runner lives.
pub fn bonsai_runner_path() -> PathBuf {
    models_dir().join("shev_bonsai_runner.py")
}

/// Write the runner next to the model, refreshing it on every call.
pub fn write_bonsai_runner() -> Result<PathBuf, String> {
    let dir = models_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let path = bonsai_runner_path();
    fs::write(&path, BONSAI_RUNNER)
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(path)
}

/// Where the generated warm worker lives.
pub fn bonsai_worker_path() -> PathBuf {
    models_dir().join("shev_bonsai_worker.py")
}

/// Write the warm worker next to the model, refreshing it on every call.
pub fn write_bonsai_worker() -> Result<PathBuf, String> {
    let dir = models_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let path = bonsai_worker_path();
    fs::write(&path, BONSAI_WORKER)
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(path)
}

/// The command that answers one decision locally. Prefers `uv` so the pinned
/// MLX stack from the pack's `runtime/requirements.txt` is used in an ephemeral
/// environment; on a machine that already has the stack, plain `python3` works.
pub fn bonsai_run_command() -> Result<Vec<String>, String> {
    if !bonsai_installed() {
        return Err(format!(
            "the bonsai model is not installed yet — install it into {}",
            bonsai_dir().display()
        ));
    }
    let runner = write_bonsai_runner()?.display().to_string();
    let requirements = bonsai_dir()
        .join("runtime")
        .join("requirements.txt")
        .display()
        .to_string();

    let mut argv: Vec<String> = if program_on_path("uv").is_some() {
        vec![
            "uv".to_string(),
            "run".to_string(),
            "--quiet".to_string(),
            "--python".to_string(),
            "3.12".to_string(),
            "--with-requirements".to_string(),
            requirements,
            "python".to_string(),
            runner,
        ]
    } else {
        vec!["python3".to_string(), runner]
    };
    argv.push(bonsai_dir().display().to_string());
    Ok(argv)
}

/// The command that starts the warm worker: same environment as the one-shot
/// runner, but a different script, with the model held in memory between
/// requests. The pack dir is the trailing argument, as for the runner.
pub fn bonsai_worker_command() -> Result<Vec<String>, String> {
    if !bonsai_installed() {
        return Err(format!(
            "the bonsai model is not installed yet — install it into {}",
            bonsai_dir().display()
        ));
    }
    let worker = write_bonsai_worker()?.display().to_string();
    let requirements = bonsai_dir()
        .join("runtime")
        .join("requirements.txt")
        .display()
        .to_string();

    let mut argv: Vec<String> = if program_on_path("uv").is_some() {
        vec![
            "uv".to_string(),
            "run".to_string(),
            "--quiet".to_string(),
            "--python".to_string(),
            "3.12".to_string(),
            "--with-requirements".to_string(),
            requirements,
            "python".to_string(),
            worker,
        ]
    } else {
        vec!["python3".to_string(), worker]
    };
    argv.push(bonsai_dir().display().to_string());
    Ok(argv)
}

/// Everything a local request needs: the pack on disk and a fresh runner.
pub fn ensure_bonsai_ready() -> Result<(), String> {
    if !bonsai_installed() {
        return Err(format!(
            "the bonsai model is not installed yet — install it from the /model menu (into {})",
            bonsai_dir().display()
        ));
    }
    write_bonsai_runner()?;
    write_bonsai_worker()?;
    Ok(())
}

/// Is the local model downloaded and ready to run?
pub fn bonsai_ready() -> bool {
    bonsai_installed()
}

pub fn run_command(argv: &[String]) -> Result<std::process::Output, String> {
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| "empty command".to_string())?;
    std::process::Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("could not run {program}: {e}"))
}

/// Download the model from the internet.
pub fn install_bonsai() -> Result<String, String> {
    let argv = bonsai_download_command()?;
    fs::create_dir_all(models_dir())
        .map_err(|e| format!("could not create {}: {e}", models_dir().display()))?;
    let output = run_command(&argv)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = if stderr.trim().is_empty() {
            stdout
        } else {
            stderr
        };
        return Err(format!("download failed: {}", snippet(&detail)));
    }
    if !bonsai_installed() {
        return Err(format!(
            "the download finished but {} still looks empty",
            bonsai_dir().display()
        ));
    }
    Ok(format!(
        "installed {BONSAI_REPO} into {} ({})",
        bonsai_dir().display(),
        human_bytes(bonsai_size())
    ))
}

/// Delete the downloaded model.
pub fn uninstall_bonsai() -> Result<String, String> {
    let dir = bonsai_dir();
    if !dir.exists() {
        return Ok(format!("{} was not installed", dir.display()));
    }
    let size = bonsai_size();
    fs::remove_dir_all(&dir).map_err(|e| format!("could not remove {}: {e}", dir.display()))?;
    let _ = fs::remove_file(bonsai_runner_path());
    Ok(format!(
        "removed {} (freed {})",
        dir.display(),
        human_bytes(size)
    ))
}

// ---------------------------------------------------------------------------
// mcp: hand the recorded JSON decisions to other models
// ---------------------------------------------------------------------------

/// Every decision in the responses file, oldest first.
pub fn load_records(path: &Path) -> Result<Vec<(String, Value)>, String> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    let parsed: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
    let map = parsed
        .as_object()
        .ok_or_else(|| format!("{} is not a JSON object", path.display()))?;
    let mut indexed: Vec<(u64, String, Value)> = map
        .iter()
        .filter_map(|(key, record)| {
            key.strip_prefix("response")
                .and_then(|index| index.parse::<u64>().ok())
                .map(|index| (index, key.clone(), record.clone()))
        })
        .collect();
    indexed.sort_by_key(|(index, _, _)| *index);
    Ok(indexed
        .into_iter()
        .map(|(_, key, record)| (key, record))
        .collect())
}

/// Serve the recorded decisions over MCP on stdin/stdout. Read-only: other
/// models pull the JSON that shev already produced.
pub fn mcp_serve(responses: &Path) -> Result<(), String> {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| format!("could not read stdin: {e}"))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(e) => {
                write_json(
                    &mut out,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": { "code": -32700, "message": format!("parse error: {e}") }
                    }),
                )?;
                continue;
            }
        };

        // notifications have no id and get no reply
        let Some(id) = message.get("id").cloned() else {
            continue;
        };
        let method = message.get("method").and_then(Value::as_str).unwrap_or("");

        let reply = match method {
            "initialize" => Ok(mcp_initialize()),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": mcp_tools() })),
            "tools/call" => mcp_call(responses, message.get("params")),
            "resources/list" => Ok(json!({ "resources": mcp_resources(responses) })),
            "resources/read" => mcp_read(responses, message.get("params")),
            other => {
                Err(json!({ "code": -32601, "message": format!("method not found: {other}") }))
            }
        };

        let envelope = match reply {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err(error) => json!({ "jsonrpc": "2.0", "id": id, "error": error }),
        };
        write_json(&mut out, &envelope)?;
    }
    Ok(())
}

fn write_json(out: &mut impl Write, value: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *out, value).map_err(|e| format!("could not write: {e}"))?;
    out.write_all(b"\n")
        .map_err(|e| format!("could not write: {e}"))?;
    out.flush().map_err(|e| format!("could not flush: {e}"))
}

fn mcp_initialize() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": { "tools": {}, "resources": {} },
        "serverInfo": { "name": "shev", "version": env!("CARGO_PKG_VERSION") },
        "instructions": "Read-only access to the JSON decisions shev recorded. \
                         Use list_decisions or latest_decision to pull them into context."
    })
}

fn tool(name: &str, description: &str, schema: Value) -> Value {
    json!({ "name": name, "description": description, "inputSchema": schema,
        "annotations": {"readOnlyHint": true, "destructiveHint": false, "idempotentHint": true} })
}

fn mcp_tools() -> Value {
    json!([
        tool(
            "list_decisions",
            "List the JSON decisions shev recorded, oldest first.",
            json!({
                "type": "object",
                "properties": { "limit": { "type": "integer", "description": "how many of the most recent decisions (default 20)" } },
                "additionalProperties": false
            })
        ),
        tool(
            "latest_decision",
            "The most recent decision shev recorded.",
            json!({ "type": "object", "properties": {}, "additionalProperties": false })
        ),
        tool(
            "get_decision",
            "One decision, by its responseN key or by 1-based index (defaults to the latest).",
            json!({
                "type": "object",
                "properties": { "key": { "type": "string" }, "index": { "type": "integer" } },
                "additionalProperties": false
            })
        ),
        tool(
            "search_decisions",
            "Decisions whose message, action or response contains a phrase.",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" }, "limit": { "type": "integer" } },
                "required": ["query"],
                "additionalProperties": false
            })
        ),
    ])
}

fn mcp_call(responses: &Path, params: Option<&Value>) -> Result<Value, Value> {
    let params = params.cloned().unwrap_or(Value::Null);
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;

    let records = load_records(responses).map_err(|e| json!({ "code": -32603, "message": e }))?;
    let as_entry = |(key, record): &(String, Value)| json!({ "key": key, "decision": record });

    let payload = match name {
        "list_decisions" => {
            let start = records.len().saturating_sub(limit);
            json!(records[start..].iter().map(as_entry).collect::<Vec<_>>())
        }
        "latest_decision" => records.last().map(as_entry).unwrap_or(Value::Null),
        "get_decision" => {
            let found = match (
                args.get("key").and_then(Value::as_str),
                args.get("index").and_then(Value::as_u64),
            ) {
                (Some(key), _) => records.iter().find(|(k, _)| k == key),
                (_, Some(index)) if index > 0 => records.get((index - 1) as usize),
                _ => records.last(),
            };
            found.map(as_entry).unwrap_or(Value::Null)
        }
        "search_decisions" => {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_ascii_lowercase();
            let hits: Vec<Value> = records
                .iter()
                .filter(|(_, record)| {
                    let haystack = format!(
                        "{} {} {}",
                        record.get("input").and_then(Value::as_str).unwrap_or(""),
                        record
                            .get("actionable")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                        record
                            .get("generated_response")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                    )
                    .to_ascii_lowercase();
                    haystack.contains(&query)
                })
                .map(as_entry)
                .collect();
            let start = hits.len().saturating_sub(limit);
            json!(hits[start..].to_vec())
        }
        other => {
            return Err(json!({ "code": -32602, "message": format!("unknown tool {other}") }));
        }
    };

    let text = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string());
    Ok(json!({
        "content": [{ "type": "text", "text": text }],
        "isError": false
    }))
}

fn mcp_resources(responses: &Path) -> Value {
    json!([
        {
            "uri": "shev://decisions",
            "name": "decisions",
            "description": format!("Every decision shev has recorded in {}", responses.display()),
            "mimeType": "application/json"
        },
        {
            "uri": "shev://decisions/latest",
            "name": "latest decision",
            "description": "The most recent decision, on its own",
            "mimeType": "application/json"
        }
    ])
}

fn mcp_read(responses: &Path, params: Option<&Value>) -> Result<Value, Value> {
    let uri = params
        .and_then(|params| params.get("uri"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let records = load_records(responses).map_err(|e| json!({ "code": -32603, "message": e }))?;
    let text = match uri {
        "shev://decisions" => {
            let map: Map<String, Value> = records.into_iter().collect();
            serde_json::to_string_pretty(&Value::Object(map)).unwrap_or_default()
        }
        "shev://decisions/latest" => match records.last() {
            Some((_, record)) => serde_json::to_string_pretty(record).unwrap_or_default(),
            None => "null".to_string(),
        },
        other => {
            return Err(json!({ "code": -32602, "message": format!("unknown resource {other}") }));
        }
    };
    Ok(json!({
        "contents": [{ "uri": uri, "mimeType": "application/json", "text": text }]
    }))
}

/// Where the generated MCP config and manifest live.
pub fn mcp_dir() -> PathBuf {
    config_path()
        .parent()
        .map(|parent| parent.join("mcp"))
        .unwrap_or_else(|| PathBuf::from(".shev-mcp"))
}

/// The paste-anywhere descriptor (`mcpServers`, as Claude Desktop and Cursor use).
fn mcp_descriptor(exe: &Path, responses: &Path) -> Value {
    let exe = exe.display().to_string();
    let file = responses.display().to_string();
    json!({
        "mcpServers": {
            "shev": {
                "command": exe,
                "args": ["--mcp"],
                "env": { "SHEV_RESPONSES_FILE": file }
            }
        }
    })
}

/// An OpenCode local-server entry (under `mcp.servers.<name>` in V2).
fn mcp_opencode_entry(exe: &Path, responses: &Path) -> Value {
    json!({
        "type": "local",
        "command": [exe.display().to_string(), "--mcp"],
        "environment": { "SHEV_RESPONSES_FILE": responses.display().to_string() },
        "enabled": true
    })
}

/// Where OpenCode keeps its config: the shared XDG config dir, under `opencode`.
/// Both the modern `.json` and the comment-friendly `.jsonc`. (This used to point
/// at a file inside shev's own config dir, so registrations and the tools shev
/// borrowed from it never actually reached OpenCode.)
fn opencode_configs() -> Vec<PathBuf> {
    let base = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    let dir = base.join("opencode");
    vec![dir.join("opencode.json"), dir.join("opencode.jsonc")]
}

/// OpenCode config in the current project. Discovery only — shev never writes
/// to these, so a borrowed tool is never registered behind your back.
fn opencode_project_configs() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        let ancestors: Vec<_> = cwd.ancestors().collect();
        for relative in ["", ".opencode"] {
            for ancestor in ancestors.iter().rev() {
                for name in ["opencode.json", "opencode.jsonc"] {
                    paths.push(ancestor.join(relative).join(name));
                }
            }
        }
    }
    paths
}

fn write_pretty(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(value).map_err(|e| format!("could not encode: {e}"))?;
    fs::write(path, format!("{body}\n"))
        .map_err(|e| format!("could not write {}: {e}", path.display()))
}

/// Install the local MCP server: write its descriptor and, when opencode is
/// present, register it there too. Reversible with [`mcp_uninstall`].
pub fn mcp_install(responses: &Path) -> Result<String, String> {
    let exe = env::current_exe().map_err(|e| format!("could not find my own path: {e}"))?;
    let responses = if responses.is_absolute() {
        responses.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|e| format!("could not resolve the history path: {e}"))?
            .join(responses)
    };
    let responses = fs::canonicalize(&responses).unwrap_or(responses);
    let dir = mcp_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;

    let descriptor = mcp_descriptor(&exe, &responses);
    let entry = mcp_opencode_entry(&exe, &responses);
    write_pretty(&dir.join("server.json"), &descriptor)?;
    write_pretty(
        &dir.join("opencode.json"),
        &json!({ "mcp": { "servers": { "shev": entry.clone() } } }),
    )?;

    let mut report = vec![format!("wrote {}", dir.join("server.json").display())];
    let mut registrations = Map::new();
    for candidate in opencode_configs() {
        if !candidate.exists() {
            continue;
        }
        match register_opencode(&candidate, &entry) {
            Ok(mode) => {
                report.push(format!(
                    "registered with opencode in {} ({mode})",
                    candidate.display()
                ));
                registrations.insert(
                    candidate.display().to_string(),
                    json!({ "mode": mode, "entry": entry }),
                );
            }
            Err(e) => report.push(format!("could not update {}: {e}", candidate.display())),
        }
    }
    if registrations.is_empty() {
        report.push(format!(
            "no opencode config found — paste this into your MCP client:\n{}",
            serde_json::to_string_pretty(&descriptor).unwrap_or_default()
        ));
    }

    write_pretty(
        &dir.join("installed.json"),
        &json!({
            "installed_at": unix_timestamp(),
            "exe": exe.display().to_string(),
            "responses": responses.display().to_string(),
            "registrations": Value::Object(registrations),
        }),
    )?;
    Ok(report.join("\n"))
}

fn register_opencode(path: &Path, entry: &Value) -> Result<&'static str, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;

    let backup = PathBuf::from(format!("{}.shev.bak", path.display()));
    if !backup.exists() {
        fs::write(&backup, &text).map_err(|_| "could not back up OpenCode config".to_string())?;
        restrict_permissions(&backup);
    }

    if let Ok(mut parsed) = serde_json::from_str::<Value>(&text) {
        let root = parsed
            .as_object_mut()
            .ok_or_else(|| "the opencode config is not a JSON object".to_string())?;
        let mcp = root.entry("mcp".to_string()).or_insert_with(|| json!({}));
        if !mcp.is_object() {
            return Err("the existing MCP configuration is not an object; left unchanged".into());
        }
        let native = mcp.get("servers").is_some() || mcp.as_object().is_some_and(Map::is_empty);
        let servers = if native {
            mcp.as_object_mut()
                .unwrap()
                .entry("servers")
                .or_insert_with(|| json!({}))
        } else {
            mcp
        };
        let servers = servers
            .as_object_mut()
            .ok_or("the existing MCP servers configuration is not an object")?;
        if servers.get("shev") == Some(entry) {
            return Ok("json");
        }
        if servers.contains_key("shev") {
            return Err("a shev registration already exists; left unchanged".into());
        }
        servers.insert("shev".to_string(), entry.clone());
        write_pretty(path, &parsed)?;
        return Ok("json");
    }

    // JSONC: preserve comments and unrelated fields, and reject duplicates.
    let parsed = parse_jsonc(&text).ok_or("the OpenCode config is not valid JSONC")?;
    let servers = parsed.pointer("/mcp/servers").or_else(|| parsed.get("mcp"));
    if servers.and_then(|m| m.get("shev")) == Some(entry) {
        return Ok("jsonc");
    }
    if servers.and_then(|m| m.get("shev")).is_some() {
        return Err("a shev registration already exists; left unchanged".into());
    }
    let Some(anchor) = text.find("\"mcp\"") else {
        return Err("there is no \"mcp\" section to merge into".to_string());
    };
    let anchor = if parsed.pointer("/mcp/servers").is_some() {
        anchor
            + text[anchor..]
                .find("\"servers\"")
                .ok_or("the MCP servers section could not be located")?
    } else {
        anchor
    };
    let Some(brace) = text[anchor..].find('{') else {
        return Err("the \"mcp\" section could not be parsed".to_string());
    };
    let insert_at = anchor + brace + 1;
    let serialized = serde_json::to_string(entry).map_err(|e| format!("could not encode: {e}"))?;
    let insert = format!("\n\t\"shev\": {serialized},\n");
    let mut merged = String::with_capacity(text.len() + insert.len());
    merged.push_str(&text[..insert_at]);
    merged.push_str(&insert);
    merged.push_str(&text[insert_at..]);
    let mut verified = parse_jsonc(&merged)
        .ok_or("MCP registration would produce invalid JSONC; left unchanged")?;
    let location = if parsed.pointer("/mcp/servers").is_some() {
        "/mcp/servers"
    } else {
        "/mcp"
    };
    let removed = verified
        .pointer_mut(location)
        .and_then(Value::as_object_mut)
        .and_then(|servers| servers.remove("shev"));
    if removed.as_ref() != Some(entry) || verified != parsed {
        return Err("could not safely locate the MCP registration; left unchanged".into());
    }
    fs::write(path, &merged).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok("jsonc")
}

fn unregister_opencode(path: &Path, mode: &str) -> Result<bool, String> {
    let Ok(text) = fs::read_to_string(path) else {
        return Ok(false);
    };

    if mode == "json"
        && let Ok(mut parsed) = serde_json::from_str::<Value>(&text)
    {
        let removed = parsed
            .get_mut("mcp")
            .map(|mcp| {
                if mcp.get("servers").is_some() {
                    mcp.get_mut("servers").unwrap()
                } else {
                    mcp
                }
            })
            .and_then(Value::as_object_mut)
            .map(|mcp| mcp.remove("shev").is_some())
            .unwrap_or(false);
        if removed {
            write_pretty(path, &parsed)?;
        }
        return Ok(removed);
    }

    // jsonc: drop the exact line we inserted
    let Some(anchor) = text.find("\"shev\":") else {
        return Ok(false);
    };
    let line_start = text[..anchor].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line = text[line_start
        ..text[anchor..]
            .find('\n')
            .map(|i| anchor + i)
            .unwrap_or(text.len())]
        .trim();
    if !line.starts_with("\"shev\":") {
        return Ok(false);
    }
    let line_end = text[anchor..]
        .find('\n')
        .map(|i| anchor + i + 1)
        .unwrap_or(text.len());
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..line_start]);
    out.push_str(&text[line_end..]);
    let mut expected = parse_jsonc(&text).ok_or("could not safely parse the existing config")?;
    let location = if expected.pointer("/mcp/servers").is_some() {
        "/mcp/servers"
    } else {
        "/mcp"
    };
    expected
        .pointer_mut(location)
        .and_then(Value::as_object_mut)
        .and_then(|servers| servers.remove("shev"));
    if parse_jsonc(&out).as_ref() != Some(&expected) {
        return Err("could not safely remove the registration; left unchanged".into());
    }
    fs::write(path, out).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(true)
}

/// Remove the MCP config and any registration it made.
pub fn mcp_uninstall() -> Result<String, String> {
    let dir = mcp_dir();
    let manifest: Value = fs::read_to_string(dir.join("installed.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or(Value::Null);

    let mut report = Vec::new();
    if let Some(registrations) = manifest.get("registrations").and_then(Value::as_object) {
        for (path, info) in registrations {
            let path = PathBuf::from(path);
            let mode = info.get("mode").and_then(Value::as_str).unwrap_or("json");
            if let Some(expected) = info.get("entry") {
                let current = fs::read_to_string(&path)
                    .ok()
                    .and_then(|text| parse_jsonc(&text));
                let actual = current.as_ref().and_then(|value| {
                    value
                        .pointer("/mcp/servers/shev")
                        .or_else(|| value.pointer("/mcp/shev"))
                });
                if actual != Some(expected) {
                    report.push(format!(
                        "left changed registration untouched in {}",
                        path.display()
                    ));
                    continue;
                }
            }
            match unregister_opencode(&path, mode) {
                Ok(true) => report.push(format!("unregistered from {}", path.display())),
                Ok(false) => report.push(format!("nothing to undo in {}", path.display())),
                Err(e) => report.push(format!("could not update {}: {e}", path.display())),
            }
        }
    }

    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| format!("could not remove {}: {e}", dir.display()))?;
        report.push(format!("removed {}", dir.display()));
    }
    if report.is_empty() {
        report.push("nothing was installed".to_string());
    }
    Ok(report.join("\n"))
}

/// One line describing whether the MCP server is installed.
pub fn mcp_status() -> String {
    let dir = mcp_dir();
    if !dir.exists() {
        return "off — /mcp install adds the local shev MCP".to_string();
    }
    let mut parts = vec![format!("on — {}", dir.display())];
    if let Ok(raw) = fs::read_to_string(dir.join("installed.json"))
        && let Ok(manifest) = serde_json::from_str::<Value>(&raw)
        && let Some(registrations) = manifest.get("registrations").and_then(Value::as_object)
    {
        for path in registrations.keys() {
            parts.push(format!("registered in {path}"));
        }
    }
    parts.push(
        "tools: list_decisions · latest_decision · get_decision · search_decisions".to_string(),
    );
    parts.join(" · ")
}

/// Show a file in the desktop file manager (Finder on macOS).
pub fn reveal_path(path: &Path) -> Result<(), String> {
    let mut command;
    if cfg!(target_os = "macos") {
        command = std::process::Command::new("open");
        command.arg("-R").arg(path);
    } else if cfg!(target_os = "windows") {
        command = std::process::Command::new("explorer");
        command.arg(format!("/select,{}", path.display()));
    } else {
        command = std::process::Command::new("xdg-open");
        command.arg(path.parent().unwrap_or(path));
    }

    match command.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("could not open the file manager: {status}")),
        Err(e) => Err(format!("could not open the file manager: {e}")),
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonsai_is_primary_and_deepseek_remains_selectable() {
        assert_eq!(DEFAULT_PROVIDER, Provider::Bonsai);
        assert_eq!(Provider::from_slug("deepseek"), Some(Provider::DeepSeek));
        assert!(Target::bonsai().is_local());
        let decision = json!({"model": BONSAI_REPO, "actionable": "Reply",
                              "confidence": 80, "take_action": true});
        assert_eq!(build_record(&decision, "m", "i")["model"], BONSAI_REPO);
    }

    #[test]
    fn invalid_local_json_is_retried_and_all_usage_is_counted() {
        let mut calls = 0;
        let result = local_decision_with_retry("original intent", |prompt| {
            calls += 1;
            assert!(prompt.contains("original intent"));
            if calls > 1 { assert!(prompt.contains("Format correction")); }
            Ok(LocalReply {
                text: if calls == 1 { "not JSON".into() } else {
                    r#"{"actionable":"Reply","take_action":true,"confidence":80,"summary":"Reply.","generated_response":"OK."}"#.into()
                },
                usage: json!({"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}),
            })
        }).unwrap();
        assert_eq!(calls, 2);
        assert_eq!(result["usage"]["total_tokens"], 30);
        let mut failures = 0;
        assert!(
            local_decision_with_retry("intent", |_| {
                failures += 1;
                Ok(LocalReply {
                    text: "bad".into(),
                    usage: json!({}),
                })
            })
            .is_err()
        );
        assert_eq!(failures, 3);
    }

    #[test]
    fn history_deletion_refuses_malformed_files_and_preserves_other_data() {
        let path = env::temp_dir().join(format!("shev-history-delete-{}.json", std::process::id()));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        write!(file, "not JSON").unwrap();
        drop(file);
        assert!(delete_record(&path, "response1").is_err());
        assert!(clear_records(&path).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "not JSON");
        fs::write(
            &path,
            r#"{"response1":{"text":"one"},"response2":{"text":"two"},"metadata":"keep"}"#,
        )
        .unwrap();
        assert!(delete_record(&path, "response1").unwrap());
        assert!(!delete_record(&path, "missing").unwrap());
        let saved: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert!(saved.get("response2").is_some());
        clear_records(&path).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&fs::read_to_string(&path).unwrap()).unwrap(),
            json!({"metadata":"keep"})
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn clearing_history_keeps_a_user_symlink_intact() {
        let target =
            env::temp_dir().join(format!("shev-history-target-{}.json", std::process::id()));
        let link = target.with_extension("link");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .unwrap();
        write!(file, "{}", json!({"response1":{"text":"one"}})).unwrap();
        drop(file);
        std::os::unix::fs::symlink(&target, &link).unwrap();
        clear_records(&link).unwrap();
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            serde_json::from_str::<Value>(&fs::read_to_string(&target).unwrap()).unwrap(),
            json!({})
        );
        fs::remove_file(link).unwrap();
        fs::remove_file(target).unwrap();
    }

    #[test]
    fn parses_a_healthy_decision() {
        let decision = parse_decision(
            r#"{"actionable": "Reply to Sam", "take_action": true, "confidence": 87, "generated_response": "Hi Sam, ..."}"#,
        )
        .unwrap();
        assert_eq!(decision["actionable"], json!("Reply to Sam"));
        assert_eq!(decision["take_action"], json!(true));
        assert_eq!(decision["confidence"], json!(87));
        assert_eq!(decision["generated_response"], json!("Hi Sam, ..."));
    }

    #[test]
    fn accepts_a_generated_response_written_as_steps() {
        let decision = parse_decision(
            r#"{"actionable": "Prepare for spring", "take_action": true, "confidence": 88, "generated_response": ["Open PAWS.", "Check holds.", "Register on October 28."]}"#,
        )
        .unwrap();
        assert_eq!(
            decision["generated_response"],
            json!("Open PAWS.\nCheck holds.\nRegister on October 28.")
        );
    }

    #[test]
    fn missing_generated_response_becomes_an_empty_string() {
        let decision =
            parse_decision(r#"{"actionable": "", "take_action": false, "confidence": 40}"#)
                .unwrap();
        assert_eq!(decision["generated_response"], json!(""));
    }

    #[test]
    fn clamps_confidence_into_range() {
        let low =
            parse_decision(r#"{"actionable": "", "take_action": false, "confidence": 3}"#).unwrap();
        assert_eq!(low["confidence"], json!(MIN_CONFIDENCE));

        let high = parse_decision(r#"{"actionable": "", "take_action": false, "confidence": 140}"#)
            .unwrap();
        assert_eq!(high["confidence"], json!(MAX_CONFIDENCE));

        let edge =
            parse_decision(r#"{"actionable": "", "take_action": true, "confidence": 34}"#).unwrap();
        assert_eq!(edge["confidence"], json!(34));

        let top = parse_decision(r#"{"actionable": "", "take_action": true, "confidence": 100}"#)
            .unwrap();
        assert_eq!(top["confidence"], json!(100));
    }

    #[test]
    fn tolerates_fences_and_loose_types() {
        let decision = parse_decision(
            "```json\n{\"Actionable\": \"fix the build\", \"Take_Action\": \"yes\", \"Confidence\": \"72.6\"}\n```",
        )
        .unwrap();
        assert_eq!(decision["actionable"], json!("fix the build"));
        assert_eq!(decision["take_action"], json!(true));
        assert_eq!(decision["confidence"], json!(73));
    }

    #[test]
    fn rejects_replies_without_the_decision_keys() {
        assert!(parse_decision(r#"{"confidence": 50}"#).is_err());
        assert!(parse_decision("not json at all").is_err());
    }

    #[test]
    fn appends_responses_with_incrementing_keys() {
        let path = env::temp_dir().join(format!("shev-test-{}.json", std::process::id()));
        let _ = fs::remove_file(&path);

        let decision = json!({ "actionable": "a", "take_action": true, "confidence": 50 });
        store_decision(&path, &decision, "first", "reply").unwrap();
        store_decision(&path, &decision, "second", "archive").unwrap();

        let raw = fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["response1"]["input"], json!("first"));
        assert_eq!(parsed["response2"]["input"], json!("second"));
        assert_eq!(parsed["response1"]["user_intent"], json!("reply"));
        assert_eq!(parsed["response2"]["user_intent"], json!("archive"));
        assert_eq!(parsed["response1"]["confidence"], json!(50));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn reads_the_whole_message_including_blank_lines() {
        let input = "Hello Adi,\n\nThe schedule is in PAWS.\n\nSincerely,\nAdi\nEOF\n";
        let mut cursor = io::Cursor::new(input);
        let message = read_message(&mut cursor).unwrap().unwrap();
        assert_eq!(
            message,
            "Hello Adi,\n\nThe schedule is in PAWS.\n\nSincerely,\nAdi"
        );
    }

    #[test]
    fn quits_on_exit_and_returns_none_on_eof() {
        let mut cursor = io::Cursor::new("exit\n");
        assert!(read_message(&mut cursor).unwrap().is_none());

        let mut cursor = io::Cursor::new("");
        assert!(read_message(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn empty_intent_falls_back_to_the_default() {
        let mut cursor = io::Cursor::new("\n");
        assert_eq!(read_intent(&mut cursor).unwrap().unwrap(), DEFAULT_INTENT);

        let mut cursor = io::Cursor::new("draft a polite reply\n");
        assert_eq!(
            read_intent(&mut cursor).unwrap().unwrap(),
            "draft a polite reply"
        );
    }

    #[test]
    fn the_prompt_carries_both_the_intent_and_the_message() {
        let content = USER_TEMPLATE
            .replace("%INTENT%", "draft a reply")
            .replace("%MESSAGE%", "Hello Adi,\n\nSincerely, Adi");
        assert!(content.contains("draft a reply"));
        assert!(content.contains("Hello Adi,\n\nSincerely, Adi"));
        assert!(!content.contains("%INTENT%"));
        assert!(!content.contains("%MESSAGE%"));
    }

    #[test]
    fn keeps_a_summary_to_three_sentences() {
        let summary = clamp_summary("Do this. Then that. Then more. And a fourth that must go.");
        assert_eq!(summary, "Do this. Then that. Then more.");
    }

    #[test]
    fn keeps_a_summary_to_120_words() {
        let long = (1..=200)
            .map(|n| format!("word{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        let summary = clamp_summary(&long);
        assert_eq!(summary.split_whitespace().count(), MAX_SUMMARY_WORDS);
        assert!(summary.ends_with('…'));
    }

    #[test]
    fn parses_the_summary_out_of_a_decision() {
        let decision = parse_decision(
            r#"{"actionable": "pay it", "take_action": true, "confidence": 80, "summary": "Pay the invoice today. Then reply to Sam.", "generated_response": "ok"}"#,
        )
        .unwrap();
        assert_eq!(
            decision["summary"],
            json!("Pay the invoice today. Then reply to Sam.")
        );
    }

    #[test]
    fn formats_token_counts_compactly() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(950), "950");
        assert_eq!(format_tokens(1_500), "1.5k");
        assert_eq!(format_tokens(12_400), "12k");
        assert_eq!(format_tokens(100_000), "100k");
        assert_eq!(format_tokens(999_000), "999k");
        assert_eq!(format_tokens(2_400_000), "2.4M");
    }

    #[test]
    fn provider_slugs_round_trip() {
        assert_eq!(Provider::from_slug("deepseek"), Some(Provider::DeepSeek));
        assert_eq!(Provider::from_slug("BONSAI"), Some(Provider::Bonsai));
        assert_eq!(Provider::from_slug("mlx"), Some(Provider::Bonsai));
        assert_eq!(Provider::from_slug("nope"), None);
        assert_eq!(Provider::DeepSeek.slug(), "deepseek");
        assert_eq!(Provider::Bonsai.slug(), "bonsai");
    }

    #[test]
    fn reads_byte_sizes_compactly() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.0 KB");
        assert_eq!(human_bytes(1_200_000_000), "1.1 GB");
    }

    #[test]
    fn the_targets_point_at_the_right_endpoints() {
        let bonsai = Target::bonsai();
        assert!(bonsai.is_local());
        // the local pack runs in a subprocess, not over HTTP
        assert!(bonsai.chat_url.is_empty());
        assert!(bonsai.models_url.is_empty());

        let deepseek = Target::deepseek("sk-test");
        assert!(!deepseek.is_local());
        assert_eq!(deepseek.model, DEEPSEEK_MODEL);
        assert!(
            deepseek.chat_url.ends_with("/chat/completions"),
            "{}",
            deepseek.chat_url
        );
    }

    #[test]
    fn the_bonsai_runner_is_a_self_contained_loader() {
        // the prompt arrives on stdin and the pack's own loader is used, because
        // stock MLX loaders cannot load these weights
        assert!(BONSAI_RUNNER.contains("sys.stdin.read()"));
        assert!(BONSAI_RUNNER.contains("load_vl_model"));
        assert!(BONSAI_RUNNER.contains("artifact"));
        assert!(BONSAI_RUNNER.contains("generation_config.json"));
        assert!(BONSAI_RUNNER.contains("sampler_settings"));
    }

    #[test]
    fn the_warm_worker_loads_once_and_stops_at_the_json_object() {
        // it announces readiness, then loops over requests without reloading
        assert!(BONSAI_WORKER.contains("emit({\"ready\": True})"));
        assert!(BONSAI_WORKER.contains("for line in sys.stdin"));
        assert!(BONSAI_WORKER.contains("class JsonWatch"));
        assert!(BONSAI_WORKER.contains("artifact"));
        assert!(BONSAI_WORKER.contains("generation_config.json"));
        // the model is loaded in Engine.__init__, i.e. once, not per request
        assert!(BONSAI_WORKER.contains("class Engine"));
    }

    #[test]
    fn the_vision_worker_really_stops_at_json_and_reports_usage() {
        let script = r#"
import ast, sys, types
source = ast.parse(sys.stdin.read())
classes = [node for node in source.body if isinstance(node, ast.ClassDef)]
env = {'sampler_settings':lambda: {}, 'render':lambda s:s, 'max_tokens':256}
exec(compile(ast.Module(body=classes,type_ignores=[]), '<worker>', 'exec'), env)
watch = env['JsonWatch']()
assert not watch.feed('{"a":"brace } inside string",')
assert watch.feed('"b":{"c":1}} trailing')
closed = []
def stream(*args, **kwargs):
    try:
        yield types.SimpleNamespace(text='{"a":',prompt_tokens=9,generation_tokens=1,total_tokens=10)
        yield types.SimpleNamespace(text='true} trailing',prompt_tokens=9,generation_tokens=2,total_tokens=11)
        raise AssertionError('generated beyond closed JSON')
    finally:
        closed.append(True)
sys.modules['mlx_vlm'] = types.SimpleNamespace(stream_generate=stream)
engine = env['Engine'].__new__(env['Engine'])
engine.vision, engine.model, engine.processor = True, None, None
assert engine.generate('prompt') == '{"a":true}'
assert closed == [True]
assert engine.usage['total_tokens'] == 11
"#;
        let Ok(mut child) = std::process::Command::new("python3")
            .args(["-c", script])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        else {
            return;
        };
        child
            .stdin
            .take()
            .unwrap()
            .write_all(BONSAI_WORKER.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn the_local_prompt_keeps_the_same_five_fields_in_fewer_tokens() {
        for key in [
            "actionable",
            "take_action",
            "confidence",
            "summary",
            "generated_response",
        ] {
            assert!(
                LOCAL_SYSTEM_PROMPT.contains(key),
                "the lean prompt lost {key}"
            );
        }
        // the whole point is that the fast prompt is much smaller than the remote one
        assert!(LOCAL_SYSTEM_PROMPT.len() * 2 < SYSTEM_PROMPT.len());
    }

    /// End-to-end timing for the resident local engine. Needs the Bonsai pack
    /// installed, so it is ignored by default:
    ///   cargo test --lib the_warm_local -- --ignored --nocapture
    #[test]
    #[ignore = "needs the local Bonsai pack installed"]
    fn the_warm_local_engine_serves_a_second_decision_faster() {
        if !bonsai_installed() {
            eprintln!("bonsai is not installed; skipping the warm-engine timing");
            return;
        }
        let client = http_client().unwrap();
        let target = Target::bonsai();
        let message = "the electricity bill is due friday, pay it tonight";
        let intent = "decide whether this needs action and say so";

        let first = std::time::Instant::now();
        let a = request_decision_target(&client, &target, message, intent).unwrap();
        let first_ms = first.elapsed().as_millis();

        let second = std::time::Instant::now();
        let b = request_decision_target(&client, &target, message, intent).unwrap();
        let second_ms = second.elapsed().as_millis();

        eprintln!("first (load + decide): {first_ms} ms");
        eprintln!("second (warm decide):  {second_ms} ms");
        assert!(a.get("take_action").is_some());
        assert!(b.get("take_action").is_some());
        assert!(b["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0);
        assert!(
            second_ms < first_ms,
            "the warm decision should beat the cold one ({second_ms} ms vs {first_ms} ms)"
        );
    }

    #[test]
    fn reads_a_skill_front_matter_without_touching_the_body() {
        let text = "---\nname: tldraw-offline\ndescription: \"Drive the offline canvas.\"\n---\n\n# Steps\n\nDo the thing.\n";
        let (name, description) = parse_front_matter(text);
        assert_eq!(name.as_deref(), Some("tldraw-offline"));
        assert_eq!(description.as_deref(), Some("Drive the offline canvas."));

        let body = strip_front_matter(text);
        assert!(body.starts_with("# Steps"), "body was {body:?}");
        assert!(body.contains("Do the thing."));
        assert!(
            !body.contains("description:"),
            "front matter leaked through"
        );
    }

    #[test]
    fn a_skill_without_front_matter_is_left_whole() {
        let text = "# Just a body\n\nNo front matter here.\n";
        assert_eq!(parse_front_matter(text), (None, None));
        assert_eq!(strip_front_matter(text), text);
    }

    #[test]
    fn reads_a_jsonc_config_with_comments_and_trailing_commas() {
        let text = r#"{
  // the servers shev can borrow
  "mcp": {
    "github": { "type": "remote", "url": "https://example/" }, /* inline */
    "shadcn": { "command": ["npx", "shadcn@latest", "mcp"], },
  },
}"#;
        let value = parse_jsonc(text).expect("jsonc should parse");
        let mcp = value.get("mcp").unwrap().as_object().unwrap();
        assert!(mcp.contains_key("github"), "the github tool must survive");
        assert_eq!(mcp["shadcn"]["command"].as_array().unwrap().len(), 3);

        // a `//` inside a string is not a comment
        let value = parse_jsonc(r#"{"url": "https://a//b"}"#).unwrap();
        assert_eq!(value["url"], "https://a//b");
    }

    #[test]
    fn carries_usage_into_the_record_and_counts_it() {
        let decision = json!({
            "actionable": "a",
            "take_action": true,
            "confidence": 50,
            "summary": "do it.",
            "generated_response": "done",
            "usage": { "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140 }
        });
        let record = build_record(&decision, "message", "intent");
        assert_eq!(record["summary"], json!("do it."));
        assert_eq!(record_tokens(&record), 140);

        // totals fall back to prompt + completion when the API omits the total
        let record = build_record(
            &json!({ "usage": { "prompt_tokens": 10, "completion_tokens": 5 } }),
            "m",
            "i",
        );
        assert_eq!(record_tokens(&record), 15);
    }

    #[test]
    fn the_mcp_server_lists_and_reads_decisions() {
        let path = env::temp_dir().join(format!("shev-mcp-test-{}.json", std::process::id()));
        let _ = fs::remove_file(&path);
        let decision = json!({
            "actionable": "Pay the invoice",
            "take_action": true,
            "confidence": 90,
            "summary": "Pay it today.",
            "generated_response": "Paid."
        });
        append_record(
            &path,
            &build_record(&decision, "the invoice is overdue", "reply"),
        )
        .unwrap();
        append_record(
            &path,
            &build_record(
                &json!({ "actionable": "", "take_action": false, "confidence": 40 }),
                "just a notice",
                "archive",
            ),
        )
        .unwrap();

        // tools/list
        let tools = mcp_tools();
        let names: Vec<&str> = tools
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"list_decisions"));
        assert!(names.contains(&"search_decisions"));

        // tools/call list_decisions
        let listed = mcp_call(&path, Some(&json!({ "name": "list_decisions" }))).unwrap();
        let text = listed["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("response1"), "{text}");
        assert!(text.contains("response2"), "{text}");

        // tools/call latest_decision
        let latest = mcp_call(&path, Some(&json!({ "name": "latest_decision" }))).unwrap();
        let text = latest["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("response2"), "{text}");

        // tools/call search_decisions
        let found = mcp_call(
            &path,
            Some(&json!({ "name": "search_decisions", "arguments": { "query": "overdue" } })),
        )
        .unwrap();
        let text = found["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("response1"), "{text}");
        assert!(!text.contains("response2"), "{text}");

        // resources/read
        let resource = mcp_read(&path, Some(&json!({ "uri": "shev://decisions" }))).unwrap();
        let text = resource["contents"][0]["text"].as_str().unwrap();
        assert!(text.contains("Paid."), "{text}");
        assert!(text.contains("just a notice"), "{text}");

        let latest = mcp_read(&path, Some(&json!({ "uri": "shev://decisions/latest" }))).unwrap();
        let text = latest["contents"][0]["text"].as_str().unwrap();
        assert!(text.contains("just a notice"), "{text}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn reads_the_mcp_opencode_registration_and_removes_it() {
        let path = env::temp_dir().join(format!("shev-oc-test-{}.json", std::process::id()));
        let _ = fs::remove_file(&path);
        // JSONC with a trailing comma and a comment, as opencode writes it
        fs::write(
            &path,
            "{\n  // mine\n  \"mcp\": {\n    \"other\": { \"type\": \"local\", \"command\": [\"x\"] },\n  },\n}\n",
        )
        .unwrap();

        let entry = json!({ "type": "local", "command": ["/bin/shev", "--mcp"], "enabled": true });
        assert_eq!(register_opencode(&path, &entry).unwrap(), "jsonc");
        let merged = fs::read_to_string(&path).unwrap();
        assert!(merged.contains("\"shev\""));
        assert!(merged.contains("\"other\""));

        assert!(unregister_opencode(&path, "jsonc").unwrap());
        let cleaned = fs::read_to_string(&path).unwrap();
        assert!(!cleaned.contains("shev"));
        assert!(cleaned.contains("\"other\""));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(format!("{}.shev.bak", path.display()));
    }

    #[test]
    fn native_mcp_registration_preserves_other_servers_and_is_idempotent() {
        let entry = json!({"type":"local","command":["shev","--mcp"]});
        for (index, original) in [
            "{\"mcp\":{\"servers\":{\"other\":{\"type\":\"remote\",\"url\":\"https://example.test\"}}}}",
            "{\n// keep this comment\n\"mcp\": {\"servers\": {\"other\": {\"type\":\"remote\",\"url\":\"https://example.test\"},},},\n}",
        ].into_iter().enumerate() {
            let path = env::temp_dir().join(format!("shev-native-{}-{index}.jsonc", std::process::id()));
            fs::write(&path, original).unwrap();
            let mode = register_opencode(&path, &entry).unwrap();
            let registered = fs::read_to_string(&path).unwrap();
            assert_eq!(parse_jsonc(&registered).unwrap()["mcp"]["servers"]["shev"], entry);
            register_opencode(&path, &entry).unwrap();
            assert_eq!(fs::read_to_string(&path).unwrap(), registered);
            assert!(register_opencode(&path, &json!({"command":["different"]})).is_err());
            assert_eq!(fs::read_to_string(&path).unwrap(), registered);
            assert!(unregister_opencode(&path, mode).unwrap());
            assert_eq!(parse_jsonc(&fs::read_to_string(&path).unwrap()), parse_jsonc(original));
            fs::remove_file(&path).unwrap();
            fs::remove_file(format!("{}.shev.bak", path.display())).unwrap();
        }
    }
}
