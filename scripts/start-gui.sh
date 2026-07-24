#!/usr/bin/env bash
# usage: ./scripts/start-gui.sh [--skip-build] [-- --host 127.0.0.1 --port 8765]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_BUILD=0
GUI_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    --)
      shift
      GUI_ARGS+=("$@")
      break
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d frontend/dist && "$SKIP_BUILD" -eq 0 ]]; then
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "pnpm not found. Install pnpm, or run: cd frontend && npm install -g pnpm" >&2
    exit 1
  fi
  (cd frontend && pnpm install && pnpm build)
fi

if [[ -x .venv/bin/cv-preprocess ]]; then
  exec .venv/bin/cv-preprocess gui "${GUI_ARGS[@]}"
fi
exec uv run cv-preprocess gui "${GUI_ARGS[@]}"
