#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo "Missing .venv/bin/python; create the project virtual environment first." >&2
  exit 1
fi
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/tools/start_server.py" "$@"
