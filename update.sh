#!/usr/bin/env bash
# Update clips: pull the latest code from GitHub and refresh the install.
#   ./update.sh
set -euo pipefail

# Wrapped in a function so bash reads the whole script before `git pull` can change it.
main() {
  cd "$(dirname "$0")"
  say() { printf '\n==> %s\n' "$*"; }

  [ -x .venv/bin/python ] || { echo "no .venv here; run ./install.sh first" >&2; exit 1; }

  say "Pulling the latest version"
  OLD=$(git rev-parse HEAD)
  if ! git pull --ff-only; then
    echo "update stopped: you have local changes or a different branch." >&2
    echo "See them with 'git status'; 'git stash' sets them aside, then run ./update.sh again." >&2
    exit 1
  fi
  NEW=$(git rev-parse HEAD)

  if [ "$OLD" = "$NEW" ]; then
    echo "already up to date ($(git log -1 --format='%h %s'))"
  else
    echo "updated $(git rev-parse --short "$OLD") -> $(git rev-parse --short "$NEW"):"
    git log --format='  - %s' --no-merges "$OLD..$NEW"
  fi

  say "Refreshing requirements"
  .venv/bin/python -m pip install -q -r requirements.txt
  .venv/bin/python -m pip install -q --no-deps -e .
  # Vision extras only when the CPU build was installed by ./install.sh --vision;
  # a GPU build (directml, migraphx, gpu) you swapped in yourself is left alone.
  if .venv/bin/python -m pip show -q onnxruntime >/dev/null 2>&1; then
    .venv/bin/python -m pip install -q -r requirements-vision.txt
  fi

  .venv/bin/clips --help >/dev/null
  say "Done"
}
main "$@"; exit
