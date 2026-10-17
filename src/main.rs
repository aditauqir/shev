//! `shev` — the line-based CLI.
//!
//! The TUI lives in the `shev` binary: `cargo run --release --bin shev`.

use std::process::ExitCode;

fn main() -> ExitCode {
    if let Some(outcome) = shev::installation::maintenance() {
        return outcome;
    }
    match shev::cli_run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("\nerror: {err}");
            ExitCode::FAILURE
        }
    }
}
