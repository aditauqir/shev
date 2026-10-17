use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn both_binaries_serve_mcp_without_a_terminal_or_api_key() {
    let path = std::env::temp_dir().join(format!("shev-mcp-stdio-{}.json", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    write!(
        file,
        "{}",
        json!({"response1":{"actionable":"Test","take_action":true,"confidence":80}})
    )
    .unwrap();
    drop(file);
    for binary in [env!("CARGO_BIN_EXE_shev"), env!("CARGO_BIN_EXE_shev-cli")] {
        let mut child = Command::new(binary)
            .args(["--mcp", "--file"])
            .arg(&path)
            .env_remove("DEEPSEEK_API_KEY")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        for request in [
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"latest_decision","arguments":{}}}),
        ] {
            writeln!(stdin, "{request}").unwrap();
        }
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let replies: Vec<Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            replies.len(),
            3,
            "stdout must contain only protocol responses"
        );
        assert!(replies[0]["result"]["capabilities"]["tools"].is_object());
        assert!(
            replies[1]["result"]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "latest_decision")
        );
        assert!(
            replies[2]["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("response1")
        );
    }
    std::fs::remove_file(path).unwrap();
}
