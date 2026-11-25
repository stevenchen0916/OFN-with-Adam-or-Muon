#!/usr/bin/env bash
set -euo pipefail
BURGERS_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
BURGERS_PY="${BURGERS_PY:-$HOME/HeatProject/.venv/bin/python}"
export DDE_BACKEND=pytorch CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$BURGERS_ROOT/.mplcache"
exec "$BURGERS_PY" -u "$BURGERS_ROOT/burgers_formal16/runner.py" --out "$BURGERS_ROOT/results16" "$@"
