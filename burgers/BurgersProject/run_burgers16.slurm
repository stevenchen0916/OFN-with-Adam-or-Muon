#!/usr/bin/env bash
#SBATCH --job-name=burgers-formal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=burgers-%j.out
set -euo pipefail
BURGERS_ROOT="${BURGERS_ROOT:-$HOME/BurgersProject}"
BURGERS_PY="${BURGERS_PY:-$HOME/HeatProject/.venv/bin/python}"
BURGERS_STAGE="${BURGERS_STAGE:-all}"
[[ -x "$BURGERS_PY" ]] || { echo "Set BURGERS_PY to the existing PyTorch/DeepXDE environment." >&2; exit 1; }
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 16 ]] || { echo 'Allocate 16 CPUs.' >&2; exit 1; }
cd "$BURGERS_ROOT"
export DDE_BACKEND=pytorch CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$BURGERS_ROOT/.mplcache"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
"$BURGERS_PY" -u run_campaign.py --stage "$BURGERS_STAGE" &
burgers_pid=$!
trap 'kill -USR1 "$burgers_pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$burgers_pid" 2>/dev/null || true' TERM INT
set +e
wait "$burgers_pid"
burgers_code=$?
if kill -0 "$burgers_pid" 2>/dev/null; then
  wait "$burgers_pid"
  burgers_code=$?
fi
exit "$burgers_code"
