#!/usr/bin/env bash
# shev installer — https://github.com/aditauqir/shev
#
#   curl -fsSL https://raw.githubusercontent.com/aditauqir/shev/main/install.sh | bash
#
# Installs the `shev` (TUI) and `shev-cli` (pipe) binaries. Prefers a prebuilt
# release for this platform; otherwise builds from source with cargo.
set -euo pipefail

REPO="aditauqir/shev"
PREFIX="${SHEV_PREFIX:-$HOME/.local}"
DEST="$PREFIX/bin"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1; }

finish() {
  say ""
  say "shev is installed in $DEST"
  case ":$PATH:" in
    *":$DEST:"*) : ;;
    *)
      say ""
      say "Add it to your PATH, then open a new terminal:"
      say "  export PATH=\"$DEST:\$PATH\""
      ;;
  esac
  say ""
  say "Run it with:"
  say "  shev"
}

os="$(uname -s)"
arch="$(uname -m)"
case "$os-$arch" in
  Darwin-arm64) target="aarch64-apple-darwin" ;;
  Darwin-x86_64) target="x86_64-apple-darwin" ;;
  Linux-x86_64) target="x86_64-unknown-linux-gnu" ;;
  Linux-aarch64) target="aarch64-unknown-linux-gnu" ;;
  *) target="" ;;
esac

mkdir -p "$DEST"

# --- 1. prebuilt release, if there is one for this platform ----------------
if [ -n "$target" ] && need curl && need tar; then
  url="https://github.com/$REPO/releases/latest/download/shev-$target.tar.gz"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  if curl -fsSL "$url" -o "$tmp/shev.tar.gz" 2>/dev/null; then
    tar -xzf "$tmp/shev.tar.gz" -C "$tmp"
    [ -f "$tmp/shev" ] || die "the downloaded archive had no shev binary"
    install -m 0755 "$tmp/shev" "$DEST/shev"
    if [ -f "$tmp/shev-cli" ]; then
      install -m 0755 "$tmp/shev-cli" "$DEST/shev-cli"
    fi
    finish
    exit 0
  fi
  rm -rf "$tmp"
  trap - EXIT
fi

# --- 2. build from source ---------------------------------------------------
if ! need cargo; then
  need curl || die "no cargo and no curl — install Rust from https://rustup.rs first"
  say "No cargo found. Installing Rust (rustup, minimal profile)…"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

say "Building shev from source with cargo (this takes a minute)…"
cargo install --git "https://github.com/$REPO" --locked --force --root "$PREFIX"

[ -x "$DEST/shev" ] || die "cargo finished but $DEST/shev is missing"
finish
