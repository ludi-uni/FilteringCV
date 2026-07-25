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

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  NEED_BUILD=0
  if [[ ! -d frontend/dist ]]; then
    NEED_BUILD=1
  else
    # Rebuild when source is newer than the last dist build (mtime of index.html).
    NEWEST_SRC="$(find frontend/src frontend/index.html frontend/package.json frontend/vite.config.ts \
      -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 || true)"
    DIST_MTIME="$(stat -c '%Y' frontend/dist/index.html 2>/dev/null || echo 0)"
    if [[ -n "${NEWEST_SRC:-}" ]]; then
      NEWEST_INT="${NEWEST_SRC%%.*}"
      if (( NEWEST_INT > DIST_MTIME )); then
        NEED_BUILD=1
      fi
    fi
  fi
  if [[ "$NEED_BUILD" -eq 1 ]]; then
    if ! command -v pnpm >/dev/null 2>&1; then
      echo "pnpm not found. Install pnpm, or run: cd frontend && npm install -g pnpm" >&2
      exit 1
    fi
    echo "Building frontend (source newer than dist or dist missing)…"
    (cd frontend && pnpm install && pnpm build)
  fi
fi

if [[ -x .venv/bin/cv-preprocess ]]; then
  exec .venv/bin/cv-preprocess gui "${GUI_ARGS[@]}"
fi
exec uv run cv-preprocess gui "${GUI_ARGS[@]}"
