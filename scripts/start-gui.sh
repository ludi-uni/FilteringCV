#!/usr/bin/env bash
# usage: ./scripts/start-gui.sh [--skip-build] [-- --host 0.0.0.0 --port 8765]
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

# Inside Docker/containers, 127.0.0.1 is unreachable from the host even with
# published ports. Default to all interfaces unless the caller set --host.
has_host=0
for arg in "${GUI_ARGS[@]+"${GUI_ARGS[@]}"}"; do
  if [[ "$arg" == "--host" || "$arg" == --host=* ]]; then
    has_host=1
    break
  fi
done
if [[ "$has_host" -eq 0 && -f /.dockerenv ]]; then
  GUI_ARGS=(--host 0.0.0.0 "${GUI_ARGS[@]+"${GUI_ARGS[@]}"}")
  echo "Detected Docker: binding GUI to 0.0.0.0 (override with -- --host …)"
fi

ensure_pnpm() {
  if command -v pnpm >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v node >/dev/null 2>&1; then
    return 1
  fi
  if command -v corepack >/dev/null 2>&1; then
    corepack enable >/dev/null 2>&1 || true
    if corepack prepare pnpm@9.15.4 --activate >/dev/null 2>&1; then
      command -v pnpm >/dev/null 2>&1 && return 0
    fi
  fi
  if command -v npm >/dev/null 2>&1; then
    npm install -g pnpm@9.15.4 >/dev/null 2>&1 || true
    command -v pnpm >/dev/null 2>&1 && return 0
  fi
  return 1
}

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
    if ! ensure_pnpm; then
      if [[ -d frontend/dist ]]; then
        echo "Warning: pnpm/node not available; using existing frontend/dist (may be stale)." >&2
        echo "  Install with: apt-get install -y nodejs npm && corepack enable && corepack prepare pnpm@9.15.4 --activate" >&2
        echo "  Or skip the check: ./scripts/start-gui.sh --skip-build" >&2
      else
        echo "pnpm/node not found and frontend/dist is missing." >&2
        echo "  Install with: apt-get install -y nodejs npm && corepack enable && corepack prepare pnpm@9.15.4 --activate" >&2
        echo "  Then: cd frontend && pnpm install && pnpm build" >&2
        exit 1
      fi
    else
      echo "Building frontend (source newer than dist or dist missing)…"
      (cd frontend && pnpm install && pnpm build)
    fi
  fi
fi

if [[ -x .venv/bin/cv-preprocess ]]; then
  exec .venv/bin/cv-preprocess gui "${GUI_ARGS[@]}"
fi
exec uv run cv-preprocess gui "${GUI_ARGS[@]}"
