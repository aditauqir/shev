use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU32, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

/// Tests run in parallel and the wall clock can be coarse, so combine the time
/// with a per-process counter to keep sandbox names distinct.
static SANDBOX_COUNTER: AtomicU32 = AtomicU32::new(0);

fn sandbox() -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "shev-uninstall-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        SANDBOX_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    root
}

fn command(binary: &Path, home: &Path) -> Command {
    let mut command = Command::new(binary);
    command
        .env_clear()
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .stdin(Stdio::null());
    command
}

#[test]
fn uninstall_is_confirmed_scoped_and_preserves_modified_or_unrelated_files() {
    let home = sandbox();
    let prefix = home.join(".local");
    let bin = prefix.join("bin");
    fs::create_dir_all(&bin).unwrap();
    let binary = bin.join("shev");
    let cli = bin.join("shev-cli");
    fs::copy(env!("CARGO_BIN_EXE_shev"), &binary).unwrap();
    fs::copy(env!("CARGO_BIN_EXE_shev-cli"), &cli).unwrap();
    let profile = home.join(".zshrc");
    let block = "# >>> shev PATH >>>\nexport PATH=example:$PATH\n# <<< shev PATH <<<\n";
    fs::write(&profile, format!("# user's settings\n{block}")).unwrap();
    assert!(
        command(&binary, &home)
            .args(["--register-install"])
            .arg(&bin)
            .env("SHEV_PATH_PROFILE", &profile)
            .env("SHEV_PATH_BLOCK", block)
            .output()
            .unwrap()
            .status
            .success()
    );
    let config = home.join(".config/shev");
    fs::write(
        config.join("config.json"),
        r#"{"api_key":"fixture-only-not-a-real-key"}"#,
    )
    .unwrap();
    let history = home.join("export.json");
    fs::write(
        &history,
        r#"{"response1":{"text":"private test chat"},"metadata":"keep"}"#,
    )
    .unwrap();
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(config.join("install.json")).unwrap()).unwrap();
    manifest["histories"] = json!([history]);
    fs::write(config.join("install.json"), manifest.to_string()).unwrap();
    let models = prefix.join("share/shev/models");
    fs::create_dir_all(&models).unwrap();
    fs::write(models.join("fixture-model"), "owned").unwrap();
    let host = home.join(".config/opencode/opencode.json");
    fs::create_dir_all(host.parent().unwrap()).unwrap();
    let entry = json!({"type":"local","command":[binary,"--mcp"]});
    fs::write(&host,json!({"mcp":{"servers":{"shev":entry,"other":{"type":"remote","url":"https://example.test"}}},"user_setting":true}).to_string()).unwrap();
    fs::create_dir_all(config.join("mcp")).unwrap();
    fs::write(
        config.join("mcp/installed.json"),
        json!({"registrations":{host.to_str().unwrap():{"mode":"json","entry":entry}}}).to_string(),
    )
    .unwrap();
    fs::write(prefix.join(".crates.toml"),"[v1]\n\"shev 0.2.0 (test)\" = [\"shev\",\"shev-cli\"]\n\"other 1.0 (test)\" = [\"other\"]\n").unwrap();
    fs::write(&cli, "user-replaced binary; do not delete").unwrap();
    let preview = command(&binary, &home)
        .args(["--uninstall", "--dry-run"])
        .output()
        .unwrap();
    assert!(preview.status.success());
    assert!(!String::from_utf8_lossy(&preview.stdout).contains("fixture-only-not-a-real-key"));
    assert!(models.exists() && config.exists() && binary.exists());
    assert!(
        !command(&binary, &home)
            .arg("--uninstall")
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(config.exists());
    let result = command(&binary, &home)
        .args(["--uninstall", "--yes"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!binary.exists() && !models.exists() && !config.exists());
    assert_eq!(
        fs::read_to_string(&cli).unwrap(),
        "user-replaced binary; do not delete"
    );
    assert_eq!(fs::read_to_string(&profile).unwrap(), "# user's settings\n");
    assert_eq!(
        serde_json::from_str::<Value>(&fs::read_to_string(&history).unwrap()).unwrap(),
        json!({"metadata":"keep"})
    );
    let host: Value = serde_json::from_str(&fs::read_to_string(&host).unwrap()).unwrap();
    assert!(host.pointer("/mcp/servers/shev").is_none());
    assert!(host.pointer("/mcp/servers/other").is_some());
    assert_eq!(host["user_setting"], true);
    let metadata = fs::read_to_string(prefix.join(".crates.toml")).unwrap();
    assert!(metadata.contains("other 1.0") && !metadata.contains("shev 0.2.0"));
    fs::remove_dir_all(home).unwrap();
}

#[test]
fn npm_state_requires_an_ownership_marker_before_recursive_cleanup() {
    let home = sandbox();
    let state = home.join("state");
    fs::create_dir(&state).unwrap();
    fs::write(state.join("user-file"), "keep").unwrap();
    let result = command(Path::new(env!("CARGO_BIN_EXE_shev")), &home)
        .env("SHEV_HOME", &state)
        .args(["--uninstall", "--yes"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(state.join("user-file").exists());
    fs::write(state.join(".shev-owned"), "shev managed state\n").unwrap();
    let result = command(Path::new(env!("CARGO_BIN_EXE_shev")), &home)
        .env("SHEV_HOME", &state)
        .args(["--uninstall", "--yes"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!state.exists());
    assert!(
        Path::new(env!("CARGO_BIN_EXE_shev")).exists(),
        "source build must never be removed"
    );
    fs::remove_dir_all(home).unwrap();
}
