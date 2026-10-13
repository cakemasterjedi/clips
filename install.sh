#!/usr/bin/env bash
# One-command setup for clips on Linux.
#   ./install.sh                 core: ffmpeg check, .venv, requirements, `clips` command
#   ./install.sh --vision        + local subject detection (--subject, ONNX Runtime, CPU)
set -euo pipefail

cd "$(dirname "$0")"
REPO=$(pwd)
VISION=0
for a in "$@"; do
  case "$a" in
    --vision) VISION=1 ;;
    -h|--help) sed -n 2,5p "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO=sudo

pkg_install() {  # best effort: apt, dnf, pacman, zypper
  if command -v apt-get >/dev/null; then $SUDO apt-get update -qq && $SUDO apt-get install -y "$@"
  elif command -v dnf >/dev/null; then $SUDO dnf install -y "$@"
  elif command -v pacman >/dev/null; then $SUDO pacman -S --needed --noconfirm "$@"
  elif command -v zypper >/dev/null; then $SUDO zypper install -y "$@"
  else return 1; fi
}

# 1. Python 3.10+
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || { echo "python3 not found; install Python 3.10 or newer" >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "Python 3.10+ needed, found $("$PY" -V)" >&2; exit 1; }

# 2. ffmpeg (optional: a bundled static build is used when there is none)
say "Checking ffmpeg"
if command -v ffmpeg >/dev/null; then
  echo "found $(command -v ffmpeg)"
else
  echo "ffmpeg not on PATH, trying the package manager (may ask for your password)"
  pkg_install ffmpeg || echo "could not install ffmpeg; the bundled imageio-ffmpeg build will be used"
fi

# 3. venv + requirements + `clips` command
say "Creating .venv"
if [ ! -x .venv/bin/python ]; then
  if ! "$PY" -m venv .venv 2>/dev/null; then
    rm -rf .venv
    echo "python venv module missing, installing it"
    pkg_install python3-venv || true
    "$PY" -m venv .venv
  fi
fi
say "Installing requirements"
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -m pip install -q --no-deps -e .

if [ "$VISION" = 1 ]; then
  say "Installing vision extras (CPU ONNX Runtime)"
  .venv/bin/python -m pip install -q -r requirements-vision.txt
fi

# 4. Put `clips` on PATH
mkdir -p "$HOME/.local/bin"
ln -sf "$REPO/.venv/bin/clips" "$HOME/.local/bin/clips"

say "Checking the install"
.venv/bin/clips --help >/dev/null
.venv/bin/python -c 'from clips.ffmpeg import ffmpeg_exe; print("ffmpeg:", ffmpeg_exe())'

say "Done"
echo "Run: clips --help"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *)
  echo "(~/.local/bin is not on your PATH yet; open a new terminal or use $REPO/.venv/bin/clips)" ;; esac
