//! Installation ownership and explicit, scoped removal. No shared toolchains are removed.
use crate::{config_path, home_dir, managed_home, restrict_permissions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Default, Serialize, Deserialize)]
struct Manifest {
    #[serde(default)]
    installation_id: Option<String>,
    #[serde(default)]
    binaries: Vec<OwnedFile>,
    #[serde(default)]
    histories: Vec<PathBuf>,
    #[serde(default)]
    profiles: Vec<ProfileEdit>,
}

#[derive(Serialize, Deserialize)]
struct OwnedFile {
    path: PathBuf,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
struct ProfileEdit {
    path: PathBuf,
    block: String,
    #[serde(default)]
    created: bool,
}

fn manifest_path() -> PathBuf {
    config_path().with_file_name("install.json")
}

fn load_manifest() -> Result<Manifest, String> {
    let path = manifest_path();
    if !path.exists() {
        return Ok(Manifest::default());
    }
    serde_json::from_str(&fs::read_to_string(path).map_err(|e| e.to_string())?)
        .map_err(|_| "shev install manifest is invalid; refusing automatic cleanup".into())
}

fn save_manifest(manifest: &Manifest) -> Result<(), String> {
    let path = manifest_path();
    fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    fs::write(
        &path,
        serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    restrict_permissions(&path);
    Ok(())
}

fn checksum(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn remember_history(path: &Path) -> Result<(), String> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let mut manifest = load_manifest()?;
    if !manifest.histories.contains(&path) {
        manifest.histories.push(path);
        save_manifest(&manifest)?;
    }
    Ok(())
}

fn register(directory: &Path) -> Result<(), String> {
    let exe = fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let directory = fs::canonicalize(directory).map_err(|e| e.to_string())?;
    if exe.parent() != Some(directory.as_path())
        || directory.file_name().and_then(|s| s.to_str()) != Some("bin")
    {
        return Err("installation registration must name this executable's bin directory".into());
    }
    let mut manifest = load_manifest()?;
    manifest.installation_id = std::env::var("SHEV_INSTALL_ID").ok();
    manifest
        .binaries
        .retain(|entry| entry.path.parent() != Some(directory.as_path()));
    for name in ["shev", "shev-cli"] {
        let path = directory.join(name);
        if path.is_file() {
            manifest.binaries.push(OwnedFile {
                sha256: checksum(&path)?,
                path,
            });
        }
    }
    if let (Ok(path), Ok(block)) = (
        std::env::var("SHEV_PATH_PROFILE"),
        std::env::var("SHEV_PATH_BLOCK"),
    ) {
        if !block.starts_with("# >>> shev PATH >>>\n") || !block.ends_with("# <<< shev PATH <<<\n")
        {
            return Err("invalid installer PATH block".into());
        }
        let path = PathBuf::from(path);
        if !manifest
            .profiles
            .iter()
            .any(|edit| edit.path == path && edit.block == block)
        {
            manifest.profiles.push(ProfileEdit {
                path,
                block,
                created: std::env::var("SHEV_PATH_PROFILE_CREATED").as_deref() == Ok("1"),
            });
        }
    }
    save_manifest(&manifest)
}

fn owned_roots() -> Result<Vec<PathBuf>, String> {
    if let Some(home) = managed_home() {
        if home.file_name().and_then(|s| s.to_str()) != Some("state")
            || fs::read_to_string(home.join(".shev-owned")).ok().as_deref()
                != Some("shev managed state\n")
        {
            return Err(
                "SHEV_HOME is not marked as shev-owned state; refusing recursive removal".into(),
            );
        }
        return Ok(vec![home]);
    }
    let config = config_path().parent().unwrap().to_path_buf();
    let data = home_dir().join(".local/share/shev");
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".cache"))
        .join("shev");
    Ok(vec![config, data, cache])
}

fn remove_owned_root(path: &Path) -> Result<(), String> {
    if path == home_dir() || path.parent().is_none() || path.components().count() < 3 {
        return Err("refusing to remove an unsafe installation root".into());
    }
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() || meta.is_file() => fs::remove_file(path),
        Ok(_) => fs::remove_dir_all(path),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.to_string()),
    }
    .map_err(|e| format!("could not remove {}: {e}", path.display()))
}

fn remove_cargo_entry(bin_dir: &Path) -> Result<(), String> {
    let Some(root) = bin_dir.parent() else {
        return Ok(());
    };
    let toml_file = root.join(".crates.toml");
    if let Ok(text) = fs::read_to_string(&toml_file)
        && let Ok(mut value) = text.parse::<toml::Value>()
        && let Some(installs) = value.get_mut("v1").and_then(toml::Value::as_table_mut)
    {
        installs.retain(|name, _| !name.starts_with("shev "));
        fs::write(
            toml_file,
            toml::to_string(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    let json_file = root.join(".crates2.json");
    if let Ok(text) = fs::read_to_string(&json_file)
        && let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(installs) = value
            .get_mut("installs")
            .and_then(serde_json::Value::as_object_mut)
    {
        installs.retain(|name, _| !name.starts_with("shev "));
        fs::write(
            json_file,
            serde_json::to_string(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn uninstall(yes: bool, dry_run: bool) -> Result<ExitCode, String> {
    let mut manifest = load_manifest()?;
    let roots = owned_roots()?;
    // A directly cargo-installed binary may predate the ownership manifest.
    if manifest.binaries.is_empty()
        && let Ok(exe) = std::env::current_exe()
        && exe
            .parent()
            .and_then(Path::file_name)
            .and_then(|s| s.to_str())
            == Some("bin")
        && matches!(
            exe.file_name().and_then(|s| s.to_str()),
            Some("shev" | "shev-cli")
        )
    {
        manifest.binaries.push(OwnedFile {
            sha256: checksum(&exe)?,
            path: exe,
        });
    }
    println!(
        "Remove shev, saved keys, models, recorded chat data, and unchanged shev MCP registrations:"
    );
    for path in &roots {
        println!("  {}", path.display());
    }
    for binary in &manifest.binaries {
        println!("  {} (only if unchanged)", binary.path.display());
    }
    for path in &manifest.histories {
        println!("  {} (shev history entries only)", path.display());
    }
    for profile in &manifest.profiles {
        println!("  shev PATH block in {}", profile.path.display());
    }
    if managed_home().is_none() && crate::legacy_config_path().exists() {
        println!(
            "  {} (legacy shev key)",
            crate::legacy_config_path().display()
        );
    }
    let models = crate::models_dir();
    if !roots.iter().any(|root| models.starts_with(root)) {
        println!(
            "  marked shev model and unchanged runners under {} (unmarked/shared weights are kept)",
            models.display()
        );
    }
    println!("Shared runtimes, Codex/OpenCode, unrelated config, and source checkouts are kept.");
    if dry_run {
        return Ok(ExitCode::SUCCESS);
    }
    if !yes {
        if !io::stdin().is_terminal() {
            return Err("uninstall needs confirmation; use --yes for non-interactive removal or --dry-run to preview".into());
        }
        print!("Uninstall and delete this data? [y/N] ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        let mut reply = String::new();
        io::stdin()
            .read_line(&mut reply)
            .map_err(|e| e.to_string())?;
        if !matches!(reply.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            println!("Cancelled.");
            return Ok(ExitCode::from(2));
        }
    }
    // Preserve the manifest if cleanup fails, so a retry can finish safely.
    println!("{}", crate::mcp_uninstall()?);
    for path in &manifest.histories {
        crate::clear_records(path)?;
        if fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            == Some(serde_json::json!({}))
            && !fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
        {
            fs::remove_file(path).map_err(|e| e.to_string())?;
        }
    }
    for profile in &manifest.profiles {
        if let Ok(text) = fs::read_to_string(&profile.path) {
            // Never delete a modified profile or infer ownership from a marker alone.
            if text.matches(&profile.block).count() == 1 {
                let restored = text.replacen(&profile.block, "", 1);
                if profile.created && restored.trim().is_empty() {
                    fs::remove_file(&profile.path).map_err(|e| e.to_string())?;
                } else {
                    fs::write(&profile.path, restored).map_err(|e| e.to_string())?;
                }
            } else if text.contains("# >>> shev PATH >>>") {
                println!("Kept modified PATH block in {}", profile.path.display());
            }
        }
    }
    let mut bin_dirs = Vec::new();
    for binary in &manifest.binaries {
        if !binary.path.exists() {
            continue;
        }
        if checksum(&binary.path)? != binary.sha256 {
            println!("Kept modified binary: {}", binary.path.display());
            continue;
        }
        if let Some(parent) = binary.path.parent() {
            bin_dirs.push(parent.to_path_buf());
        }
        fs::remove_file(&binary.path).map_err(|e| e.to_string())?;
    }
    bin_dirs.sort();
    bin_dirs.dedup();
    for directory in bin_dirs {
        remove_cargo_entry(&directory)?;
    }
    // Custom model roots can be shared. Never infer ownership from a model name.
    if !roots.iter().any(|root| models.starts_with(root)) {
        let pack = models.join(crate::BONSAI_DIR_NAME);
        if fs::read_to_string(pack.join(".shev-owned")).ok().as_deref() == Some("shev model\n") {
            remove_owned_root(&pack)?;
        } else if pack.exists() {
            println!("Kept externally managed model: {}", pack.display());
        }
        for (name, source) in [
            ("shev_bonsai_runner.py", crate::BONSAI_RUNNER),
            ("shev_bonsai_worker.py", crate::BONSAI_WORKER),
        ] {
            let path = models.join(name);
            if fs::read_to_string(&path).ok().as_deref() == Some(source) {
                fs::remove_file(path).map_err(|e| e.to_string())?;
            }
        }
    }
    if managed_home().is_none() {
        let legacy = crate::legacy_config_path();
        if legacy.exists() {
            fs::remove_file(legacy).map_err(|e| e.to_string())?;
        }
    }
    for path in roots.into_iter().rev() {
        remove_owned_root(&path)?;
    }
    if let Some(parent) = managed_home().and_then(|path| path.parent().map(Path::to_path_buf))
        && parent.file_name().and_then(|name| name.to_str()) == Some("shev-npm")
    {
        let _ = fs::remove_dir(parent); // Only succeeds when the app's container is empty.
    }
    println!("shev uninstalled.");
    Ok(ExitCode::SUCCESS)
}

/// Runs before terminal setup in both binaries. Nothing is deleted without consent.
pub fn maintenance() -> Option<ExitCode> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let first = args.first()?.as_str();
    let result = match first {
        "--version" | "-V" => {
            println!("shev {}", env!("CARGO_PKG_VERSION"));
            Ok(ExitCode::SUCCESS)
        }
        "--register-install" if args.len() == 2 => {
            register(Path::new(&args[1])).map(|()| ExitCode::SUCCESS)
        }
        "--uninstall" => {
            if args[1..]
                .iter()
                .any(|arg| !matches!(arg.as_str(), "--yes" | "--dry-run"))
            {
                Err("usage: shev --uninstall [--dry-run] [--yes]".into())
            } else {
                uninstall(
                    args.iter().any(|a| a == "--yes"),
                    args.iter().any(|a| a == "--dry-run"),
                )
            }
        }
        _ => return None,
    };
    Some(match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    })
}
