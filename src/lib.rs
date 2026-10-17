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
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

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

    // 1 + 2: onboarding and verification, in that order.
    let api_key = obtain_api_key(&client, interactive)?;

    // 3: one message, or a loop over messages.
    if let Some(message) = options.once {
        let intent = options
            .intent
            .clone()
            .unwrap_or_else(|| DEFAULT_INTENT.to_string());
        let decision = request_decision(&client, &api_key, &message, &intent)?;
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
        let decision = request_decision(&client, &api_key, &message, &intent)?;
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
        match request_decision(&client, &api_key, &message, &intent) {
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
            // no server: the local pack runs in a one-shot subprocess
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

/// Answer one decision with the local Bonsai pack. There is no server, so this
/// runs the pack's bundled loader in a subprocess, feeds the whole prompt on
/// stdin (system prompt and user template together) and parses the decision out
/// of its stdout.
fn request_decision_bonsai(user_content: &str) -> Result<Value, String> {
    let argv = bonsai_run_command()?;
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| "empty command".to_string())?;
    let prompt = format!("{SYSTEM_PROMPT}\n\n{user_content}");

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

    let text = String::from_utf8_lossy(&output.stdout);
    let mut decision = parse_decision(&text)?;
    // the local pack does not report token usage
    if let Some(object) = decision.as_object_mut() {
        object.insert(
            "usage".to_string(),
            json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        );
    }
    Ok(decision)
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
        "model": MODEL,
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
        "shev — hand a whole message and your intent to DeepSeek, get back a decision and a response\n\n\
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
// models: a remote DeepSeek endpoint, or a local MLX model
// ---------------------------------------------------------------------------

/// The DeepSeek model.
pub const DEEPSEEK_MODEL: &str = MODEL;
/// The local MLX model pack, run on Apple Silicon through its bundled loader.
/// Stock mlx-lm/mlx-vlm cannot load these weights, so there is no server to hit:
/// each request runs the pack's own runtime in a one-shot subprocess.
pub const BONSAI_REPO: &str = "prism-ml/Ternary-Bonsai-2-2B-mlx-2bit";
pub const BONSAI_DIR_NAME: &str = "Ternary-Bonsai-2-2B-mlx-2bit";
/// How many tokens the local model may generate for one decision.
pub const BONSAI_MAX_TOKENS: u32 = 2048;

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

/// Which model the user last selected. DeepSeek unless told otherwise.
pub fn load_provider() -> Provider {
    load_config_map()
        .get("provider")
        .and_then(Value::as_str)
        .and_then(Provider::from_slug)
        .unwrap_or(Provider::DeepSeek)
}

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
        "temperature": gen.get("temperature", 1.0),
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

/// Everything a local request needs: the pack on disk and a fresh runner.
pub fn ensure_bonsai_ready() -> Result<(), String> {
    if !bonsai_installed() {
        return Err(format!(
            "the bonsai model is not installed yet — install it from the /model menu (into {})",
            bonsai_dir().display()
        ));
    }
    write_bonsai_runner()?;
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
    json!({ "name": name, "description": description, "inputSchema": schema })
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

/// The opencode-flavoured entry (`mcp.<name>` with `type: local`).
fn mcp_opencode_entry(exe: &Path, responses: &Path) -> Value {
    json!({
        "type": "local",
        "command": [exe.display().to_string(), "--mcp"],
        "environment": { "SHEV_RESPONSES_FILE": responses.display().to_string() },
        "enabled": true
    })
}

fn opencode_configs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(config_root) = config_path().parent() {
        out.push(config_root.join("opencode").join("opencode.json"));
        out.push(config_root.join("opencode").join("opencode.jsonc"));
    }
    out
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
    let responses = fs::canonicalize(responses).unwrap_or_else(|_| responses.to_path_buf());
    let dir = mcp_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;

    let descriptor = mcp_descriptor(&exe, &responses);
    let entry = mcp_opencode_entry(&exe, &responses);
    write_pretty(&dir.join("server.json"), &descriptor)?;
    write_pretty(
        &dir.join("opencode.json"),
        &json!({ "mcp": { "shev": entry.clone() } }),
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
                registrations.insert(candidate.display().to_string(), json!({ "mode": mode }));
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
        let _ = fs::write(&backup, &text);
    }

    if let Ok(mut parsed) = serde_json::from_str::<Value>(&text) {
        let root = parsed
            .as_object_mut()
            .ok_or_else(|| "the opencode config is not a JSON object".to_string())?;
        let mcp = root.entry("mcp".to_string()).or_insert_with(|| json!({}));
        if !mcp.is_object() {
            *mcp = json!({});
        }
        mcp.as_object_mut()
            .expect("just made it an object")
            .insert("shev".to_string(), entry.clone());
        write_pretty(path, &parsed)?;
        return Ok("json");
    }

    // JSONC (comments, trailing commas): insert straight after the mcp brace.
    let Some(anchor) = text.find("\"mcp\"") else {
        return Err("there is no \"mcp\" section to merge into".to_string());
    };
    let Some(brace) = text[anchor..].find('{') else {
        return Err("the \"mcp\" section could not be parsed".to_string());
    };
    let insert_at = anchor + brace + 1;
    let serialized = serde_json::to_string(entry).map_err(|e| format!("could not encode: {e}"))?;
    let insert = format!("\n\t\"shev\": {serialized},");
    let mut merged = String::with_capacity(text.len() + insert.len());
    merged.push_str(&text[..insert_at]);
    merged.push_str(&insert);
    merged.push_str(&text[insert_at..]);
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
}
