#!/usr/bin/env bash
# Use the isolated Isaac Sim 5.1 / Isaac Lab 2.3.2 runtime prepared on Thor.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
runtime=${PEN_RUNTIME:-"$root/runs/pen-runtime"}
export ISAAC_PY=${ISAAC_PY:-"$runtime/venv/bin/python"}
export PY=${PY:-"$runtime/render-venv/bin/python"}
export SHARPA_ROOT=${SHARPA_ROOT:-"$runtime/sharpa"}
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
for executable in "$ISAAC_PY" "$PY"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing pen runtime: $executable. See pen/THOR.md." >&2
        exit 1
    fi
done
exec bash "$root/pen/run_policy.sh" "${1:-$root/runs/pen-policy-thor-$(date +%Y%m%d-%H%M%S)}" "${2:-0}"
