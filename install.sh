#!/usr/bin/env bash
# shev installer — https://github.com/aditauqir/shev
#
#   curl -fsSL https://raw.githubusercontent.com/aditauqir/shev/main/install.sh | bash
#
# Installs the `shev` (TUI) and `shev-cli` (pipe) binaries. Prefers a prebuilt
# release for this platform; otherwise builds from source with cargo.
set -euo pipefail

REPO="aditauqir/shev"
VERSION="0.2.1"
if [ -n "${SHEV_PREFIX:-}" ]; then
  PREFIX="$SHEV_PREFIX"
elif [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
  PREFIX="$HOME/.local"
elif [[ ":$PATH:" == *":$HOME/.cargo/bin:"* ]]; then
  PREFIX="$HOME/.cargo"
else
  PREFIX="$HOME/.local"
fi
DEST="$PREFIX/bin"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1; }

finish() {
  "$DEST/shev" --register-install "$DEST"
  say ""
  say "shev is installed in $DEST"
  if [ "${SHEV_NO_PATH_SETUP:-0}" = 1 ]; then return; fi
  case ":$PATH:" in
    *":$DEST:"*) : ;;
    *)
      local profiles=() quoted block profile created existing login_profile shell_name="${SHELL:-sh}"
      printf -v quoted '%q' "$DEST"
      printf -v block '# >>> shev PATH >>>\nexport PATH=%s:$PATH\n# <<< shev PATH <<<\n' "$quoted"
      case "${shell_name##*/}" in
        zsh) profiles=("${ZDOTDIR:-$HOME}/.zshrc") ;;
        bash)
          login_profile="$HOME/.bash_profile"
          if [ ! -f "$login_profile" ]; then
            if [ -f "$HOME/.bash_login" ]; then login_profile="$HOME/.bash_login"
            elif [ -f "$HOME/.profile" ]; then login_profile="$HOME/.profile"; fi
          fi
          profiles=("$HOME/.bashrc" "$login_profile")
          ;;
        fish)
          profiles=("${XDG_CONFIG_HOME:-$HOME/.config}/fish/conf.d/shev.fish")
          printf -v block '# >>> shev PATH >>>\nfish_add_path --prepend %s\n# <<< shev PATH <<<\n' "$quoted"
          ;;
        *) profiles=("$HOME/.profile") ;;
      esac
      for profile in "${profiles[@]}"; do
        mkdir -p "$(dirname "$profile")"
        existing="$(cat "$profile" 2>/dev/null || true)"
        if [[ "$existing" != *"${block%$'\n'}"* ]]; then
          created=0
          [ -e "$profile" ] || created=1
          printf '\n%s' "$block" >> "$profile"
          SHEV_PATH_PROFILE="$profile" SHEV_PATH_BLOCK="$block" SHEV_PATH_PROFILE_CREATED="$created" "$DEST/shev" --register-install "$DEST"
        fi
      done
      say "PATH is configured. Open a new terminal, then type shev."
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
  url="https://github.com/$REPO/releases/download/v$VERSION/shev-$target.tar.gz"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  if curl -fsSL "$url" -o "$tmp/shev.tar.gz" 2>/dev/null && tar -xzf "$tmp/shev.tar.gz" -C "$tmp"; then
    [ -f "$tmp/shev" ] || die "the downloaded archive had no shev binary"
    if [ "$("$tmp/shev" --version 2>/dev/null)" = "shev $VERSION" ]; then
    install -m 0755 "$tmp/shev" "$DEST/shev"
    if [ -f "$tmp/shev-cli" ]; then
      install -m 0755 "$tmp/shev-cli" "$DEST/shev-cli"
    fi
    finish
    exit 0
    fi
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
if [ -n "${SHEV_SOURCE_DIR:-}" ]; then
  cargo install --path "$SHEV_SOURCE_DIR" --locked --force --root "$PREFIX"
else
  cargo install --git "https://github.com/$REPO" --locked --force --root "$PREFIX"
fi

[ -x "$DEST/shev" ] || die "cargo finished but $DEST/shev is missing"
finish
