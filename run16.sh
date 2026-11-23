#!/usr/bin/env bash
set -euo pipefail
HEAT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
HEAT_PY="$HEAT_ROOT/.venv/bin/python"
HEAT_RESULTS_DIR="${HEAT_RESULTS_DIR:-$HEAT_ROOT/results16}"
[[ -x "$HEAT_PY" ]] || { echo 'Existing HeatProject/.venv is required; do not reinstall data.' >&2; exit 1; }
[[ $# -gt 0 ]] || { echo 'Usage: bash run16.sh command [args]'; exit 2; }
case "$1" in
  verify|smoke|sweep|final|evaluate|throughput)
    [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-0}" -ge 16 ]] || {
      echo 'Submit a Slurm job with --ntasks=1 --cpus-per-task=16. Do not train on login nodes.' >&2
      exit 1
    }
    ;;
esac
cd "$HEAT_ROOT"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 DDE_BACKEND=pytorch
export CUDA_VISIBLE_DEVICES=''
export MPLCONFIGDIR="$HEAT_ROOT/.mplcache"
exec "$HEAT_PY" -u "$HEAT_ROOT/launch16.py" --out "$HEAT_RESULTS_DIR" "$@"
