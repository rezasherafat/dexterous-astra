#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
runtime=${PEN_RUNTIME:-"$root/runs/pen-runtime"}
export SHARPA_ROOT=${SHARPA_ROOT:-"$runtime/sharpa"}
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd "$root"
exec "$runtime/venv/bin/python" pen/training/train.py "$@"
