//! Read-only reuse of existing MCP registrations. Configuration is never edited.
use crate::{home_dir, opencode_configs, opencode_project_configs, parse_jsonc};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(30);
const LIMIT: usize = 256 * 1024;

#[derive(Clone)]
pub struct Server {
    pub name: String,
    pub source: String,
    config: Value,
}

pub fn servers() -> Vec<Server> {
    let codex = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".codex"));
    let mut found = BTreeMap::new();
    if let Ok(text) = std::fs::read_to_string(codex.join("config.toml"))
        && let Ok(value) = text.parse::<toml::Value>()
        && let Some(entries) = value.get("mcp_servers").and_then(toml::Value::as_table)
    {
        for (name, cfg) in entries {
            if let Ok(config) = serde_json::to_value(cfg) {
                found.insert(
                    name.clone(),
                    Server {
                        name: name.clone(),
                        source: "codex".into(),
                        config,
                    },
                );
            }
        }
    }
    for path in opencode_configs()
        .into_iter()
        .chain(opencode_project_configs())
    {
        let Some(value) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| parse_jsonc(&t))
        else {
            continue;
        };
        let mcp = value.get("mcp").unwrap_or(&Value::Null);
        let Some(entries) = mcp.get("servers").unwrap_or(mcp).as_object() else {
            continue;
        };
        for (name, config) in entries {
            if config.get("enabled") == Some(&json!(false)) {
                found.remove(name);
                continue;
            }
            if !config.is_object() {
                continue;
            }
            found.insert(
                name.clone(),
                Server {
                    name: name.clone(),
                    source: "opencode".into(),
                    config: config.clone(),
                },
            );
        }
    }
    found
        .into_values()
        .filter(|s| s.config.get("enabled") != Some(&json!(false)))
        .collect()
}

/// Only tools explicitly declared read-only and non-destructive are offered.
/// These declarations come from a trusted configured server, not from the model.
pub fn read_only(tool: &Value) -> bool {
    tool.pointer("/annotations/readOnlyHint")
        .and_then(Value::as_bool)
        == Some(true)
        && tool
            .pointer("/annotations/destructiveHint")
            .and_then(Value::as_bool)
            != Some(true)
}

fn expand(value: &str) -> Result<String, String> {
    let mut text = value.to_string();
    for _ in 0..32 {
        let Some(start) = text.find("{env:") else {
            return Ok(text);
        };
        let end = text[start..]
            .find('}')
            .ok_or("invalid environment reference")?
            + start;
        let name = &text[start + 5..end];
        let replacement =
            std::env::var(name).map_err(|_| "an MCP environment variable is missing")?;
        text.replace_range(start..=end, &replacement);
    }
    Err("recursive environment reference in MCP configuration".into())
}

struct Local {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<Result<String, String>>,
}

impl Drop for Local {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

enum Transport {
    Local(Local),
    Remote {
        client: reqwest::blocking::Client,
        url: String,
        session: Option<String>,
    },
}

impl Transport {
    fn open(server: &Server) -> Result<Self, String> {
        let cfg = &server.config;
        if let Some(url) = cfg.get("url").and_then(Value::as_str) {
            let mut headers = reqwest::header::HeaderMap::new();
            if let Some(values) = cfg
                .get("headers")
                .or_else(|| cfg.get("http_headers"))
                .and_then(Value::as_object)
            {
                for (name, value) in values {
                    let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                        .map_err(|_| "invalid MCP header name")?;
                    let value = expand(value.as_str().ok_or("invalid MCP header")?)?;
                    headers.insert(name, value.parse().map_err(|_| "invalid MCP header value")?);
                }
            }
            if let Some(name) = cfg.get("bearer_token_env_var").and_then(Value::as_str) {
                let token = std::env::var(name)
                    .map_err(|_| "MCP bearer-token environment variable is missing")?;
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {token}")
                        .parse()
                        .map_err(|_| "invalid MCP bearer token")?,
                );
            }
            if let Some(values) = cfg.get("env_http_headers").and_then(Value::as_object) {
                for (name, variable) in values {
                    let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                        .map_err(|_| "invalid MCP header name")?;
                    let value = std::env::var(
                        variable
                            .as_str()
                            .ok_or("invalid MCP header environment reference")?,
                    )
                    .map_err(|_| "MCP header environment variable is missing")?;
                    headers.insert(name, value.parse().map_err(|_| "invalid MCP header value")?);
                }
            }
            let client = reqwest::blocking::Client::builder()
                .default_headers(headers)
                .timeout(TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|_| "could not build MCP client")?;
            return Ok(Self::Remote {
                client,
                url: expand(url)?,
                session: None,
            });
        }
        let command = cfg
            .get("command")
            .ok_or("MCP server has no command or URL")?;
        let mut args: Vec<String> = if let Some(parts) = command.as_array() {
            parts
                .iter()
                .map(|s| expand(s.as_str().ok_or("invalid MCP command")?))
                .collect::<Result<_, _>>()?
        } else {
            vec![expand(command.as_str().ok_or("invalid MCP command")?)?]
        };
        if let Some(extra) = cfg.get("args").and_then(Value::as_array) {
            for arg in extra {
                args.push(expand(arg.as_str().ok_or("invalid MCP argument")?)?);
            }
        }
        let (program, args) = args.split_first().ok_or("empty MCP command")?;
        let mut command = Command::new(program);
        command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(cwd) = cfg.get("cwd").and_then(Value::as_str) {
            command.current_dir(expand(cwd)?);
        }
        if let Some(env) = cfg
            .get("environment")
            .or_else(|| cfg.get("env"))
            .and_then(Value::as_object)
        {
            for (key, value) in env {
                command.env(
                    key,
                    expand(value.as_str().ok_or("invalid MCP environment")?)?,
                );
            }
        }
        let mut child = command
            .spawn()
            .map_err(|_| "could not launch configured MCP server")?;
        let stdin = child.stdin.take().ok_or("MCP stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("MCP stdout unavailable")?;
        let (tx, lines) = mpsc::sync_channel(16);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                // Bound each protocol message; never collect unlimited server output.
                let result =
                    std::io::Read::take(&mut reader, LIMIT as u64 + 1).read_line(&mut line);
                match result {
                    Ok(0) | Err(_) => break,
                    Ok(_) if line.len() > LIMIT => {
                        let _ = tx.send(Err("MCP message is too large".into()));
                        break;
                    }
                    Ok(_) => {
                        if tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Ok(Self::Local(Local {
            child,
            stdin,
            lines,
        }))
    }

    fn send(&mut self, request: Value) -> Result<Value, String> {
        let id = request.get("id").cloned();
        match self {
            Self::Local(local) => {
                writeln!(local.stdin, "{request}")
                    .and_then(|()| local.stdin.flush())
                    .map_err(|_| "MCP write failed")?;
                let Some(id) = id else { return Ok(Value::Null) };
                let start = Instant::now();
                loop {
                    let line = local
                        .lines
                        .recv_timeout(TIMEOUT.saturating_sub(start.elapsed()))
                        .map_err(|_| "MCP response timed out or server disconnected")??;
                    let value: Value =
                        serde_json::from_str(&line).map_err(|_| "MCP server sent invalid JSON")?;
                    if value.get("id") == Some(&id) {
                        return response(value);
                    }
                    if value.get("id").is_some() && value.get("method").is_some() {
                        writeln!(local.stdin, "{}", json!({"jsonrpc":"2.0","id":value["id"],"error":{"code":-32601,"message":"Client requests not supported"}})).map_err(|_| "MCP write failed")?;
                    }
                }
            }
            Self::Remote {
                client,
                url,
                session,
            } => {
                let mut req = client
                    .post(url.as_str())
                    .header("Accept", "application/json, text/event-stream")
                    .json(&request);
                if let Some(value) = session.as_ref() {
                    req = req
                        .header("Mcp-Session-Id", value)
                        .header("MCP-Protocol-Version", "2025-03-26");
                }
                let reply = req.send().map_err(|_| "MCP connection failed")?;
                let status = reply.status();
                if status.as_u16() == 401 || status.as_u16() == 403 {
                    return Err("MCP authentication required; authenticate in its host, or configure an environment-based token (OAuth credentials are not copied)".into());
                }
                if !status.is_success() {
                    return Err(format!("MCP returned HTTP {}", status.as_u16()));
                }
                if let Some(value) = reply
                    .headers()
                    .get("Mcp-Session-Id")
                    .and_then(|h| h.to_str().ok())
                {
                    *session = Some(value.to_string());
                }
                if id.is_none() {
                    return Ok(Value::Null);
                }
                let sse = reply
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|h| h.to_str().ok())
                    .is_some_and(|v| v.contains("text/event-stream"));
                let mut reader = BufReader::new(std::io::Read::take(reply, LIMIT as u64 + 1));
                let mut text = String::new();
                let mut total = 0;
                loop {
                    let mut line = String::new();
                    let n = reader
                        .read_line(&mut line)
                        .map_err(|_| "MCP response read failed")?;
                    total += n;
                    if total > LIMIT {
                        return Err("MCP response is too large".into());
                    }
                    if !sse {
                        text.push_str(&line);
                    } else if let Some(data) = line.strip_prefix("data:") {
                        text.push_str(data.trim_start_matches(' ').trim_end_matches(['\r', '\n']));
                        text.push('\n');
                    } else if line.trim().is_empty() && !text.is_empty() {
                        if let Ok(value) = serde_json::from_str::<Value>(&text)
                            && value.get("id") == id.as_ref()
                        {
                            return response(value);
                        }
                        text.clear();
                    }
                    if n == 0 {
                        break;
                    }
                }
                let value: Value =
                    serde_json::from_str(&text).map_err(|_| "unsupported MCP response format")?;
                if value.get("id") != id.as_ref() {
                    return Err("MCP response ID did not match".into());
                }
                response(value)
            }
        }
    }

    fn initialize(&mut self) -> Result<(), String> {
        self.send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"shev","version":env!("CARGO_PKG_VERSION")}}}))?;
        self.send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))?;
        Ok(())
    }

    fn tools(&mut self) -> Result<Vec<Value>, String> {
        let mut tools = Vec::new();
        let mut cursor = Value::Null;
        for id in 2..=101 {
            let params = if cursor.is_null() {
                json!({})
            } else {
                json!({"cursor":cursor})
            };
            let result =
                self.send(json!({"jsonrpc":"2.0","id":id,"method":"tools/list","params":params}))?;
            tools.extend(
                result
                    .get("tools")
                    .and_then(Value::as_array)
                    .cloned()
                    .ok_or("MCP server returned no tool list")?,
            );
            cursor = result.get("nextCursor").cloned().unwrap_or(Value::Null);
            if cursor.is_null() {
                return Ok(tools);
            }
        }
        Err("MCP tool pagination exceeded its limit".into())
    }
}

fn response(value: Value) -> Result<Value, String> {
    // Never echo raw server errors: they can include credentials or request data.
    if value.get("error").is_some() {
        return Err("MCP server rejected the request".into());
    }
    value
        .get("result")
        .cloned()
        .ok_or_else(|| "MCP response has no result".into())
}

pub fn list(server: &Server) -> Result<Vec<Value>, String> {
    let mut transport = Transport::open(server)?;
    transport.initialize()?;
    transport.tools()
}

pub fn call(server: &Server, name: &str, arguments: Value) -> Result<Value, String> {
    if !arguments.is_object() {
        return Err("tool arguments must be a JSON object".into());
    }
    let mut transport = Transport::open(server)?;
    transport.initialize()?;
    let tools = transport.tools()?;
    let tool = tools
        .iter()
        .find(|t| t.get("name").and_then(Value::as_str) == Some(name))
        .ok_or("tool not found")?;
    if !read_only(tool) {
        return Err(
            "blocked: the server does not declare this tool read-only and non-destructive".into(),
        );
    }
    let result = transport.send(json!({"jsonrpc":"2.0","id":1000,"method":"tools/call","params":{"name":name,"arguments":arguments}}))?;
    if result.get("isError") == Some(&json!(true)) {
        return Err("MCP tool reported a failure".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_and_mutating_tools_are_blocked() {
        assert!(!read_only(&json!({"name":"read"})));
        assert!(!read_only(
            &json!({"annotations":{"readOnlyHint":true,"destructiveHint":true}})
        ));
        assert!(read_only(
            &json!({"annotations":{"readOnlyHint":true,"destructiveHint":false}})
        ));
    }
    #[test]
    fn protocol_errors_do_not_leak_server_messages() {
        assert!(
            !response(json!({"error":{"message":"secret"}}))
                .unwrap_err()
                .contains("secret")
        );
    }

    #[test]
    fn stdio_tools_are_really_called_and_mutations_never_reach_the_server() {
        if Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let script = r#"
import sys, json
for line in sys.stdin:
    q = json.loads(line)
    if 'id' not in q: continue
    method = q['method']
    if method == 'initialize': result = {'protocolVersion':'2025-03-26','capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}
    elif method == 'tools/list':
        if q.get('params',{}).get('cursor') == 'page2':
            result = {'tools':[{'name':'erase','inputSchema':{'type':'object'},'annotations':{'readOnlyHint':False,'destructiveHint':True}}]}
        else:
            result = {'tools':[{'name':'lookup','inputSchema':{'type':'object'},'annotations':{'readOnlyHint':True,'destructiveHint':False}}], 'nextCursor':'page2'}
    elif method == 'tools/call':
        assert q['params']['name'] == 'lookup', 'MUTATING TOOL WAS CALLED'
        result = {'content':[{'type':'text','text':q['params']['arguments']['query']}], 'isError':False}
    else: raise AssertionError(method)
    print(json.dumps({'jsonrpc':'2.0','id':q['id'],'result':result}), flush=True)
"#;
        let server = Server {
            name: "fixture".into(),
            source: "test".into(),
            config: json!({"command":["python3","-u","-c",script]}),
        };
        assert_eq!(list(&server).unwrap().len(), 2);
        let output = call(&server, "lookup", json!({"query":"real result"})).unwrap();
        assert_eq!(output["content"][0]["text"], "real result");
        let error = call(&server, "erase", json!({})).unwrap_err();
        assert!(error.starts_with("blocked:"), "{error}");
        assert!(
            call(&server, "missing", json!({}))
                .unwrap_err()
                .contains("not found")
        );
    }

    #[test]
    fn http_streamable_mcp_reuses_session_and_reads_sse() {
        use std::io::Read;
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let thread = std::thread::spawn(move || {
            for request_index in 0..4 {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut length = 0;
                let mut session = false;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let lower = line.to_lowercase();
                    if let Some(n) = lower.strip_prefix("content-length:") {
                        length = n.trim().parse().unwrap();
                    }
                    if lower.starts_with("mcp-session-id: fixture") {
                        session = true;
                    }
                    if line == "\r\n" {
                        break;
                    }
                }
                if request_index > 0 {
                    assert!(session);
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let q: Value = serde_json::from_slice(&body).unwrap();
                let result = match q["method"].as_str().unwrap() {
                    "initialize" => {
                        json!({"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1"}})
                    }
                    "notifications/initialized" => {
                        write!(socket, "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                        continue;
                    }
                    "tools/list" => {
                        json!({"tools":[{"name":"lookup","inputSchema":{"type":"object"},"annotations":{"readOnlyHint":true}}]})
                    }
                    "tools/call" => json!({"content":[{"type":"text","text":"HTTP result"}]}),
                    _ => panic!("unexpected MCP method"),
                };
                let event = format!(
                    "data: {}\n\n",
                    json!({"jsonrpc":"2.0","id":q["id"],"result":result})
                );
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nMcp-Session-Id: fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", event.len(), event).unwrap();
            }
        });
        let server = Server {
            name: "fixture".into(),
            source: "test".into(),
            config: json!({"url":format!("http://{address}")}),
        };
        let result = call(&server, "lookup", json!({})).unwrap();
        assert_eq!(result["content"][0]["text"], "HTTP result");
        thread.join().unwrap();
    }
}
