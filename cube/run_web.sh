#!/usr/bin/env bash
# Live viewer on all IPv4 interfaces. Use --host 127.0.0.1 for localhost only.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
exec "${PY:-$root/.venv/bin/python}" "$root/cube/web/server.py" "$@"
