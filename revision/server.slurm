#!/usr/bin/env bash
#SBATCH --partition=T1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=revision-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the unzipped project root}"
py="${REV_PY:-$HOME/HeatProject/.venv/bin/python}"
[[ -x "$py" ]] || { echo "Missing Python environment: $py" >&2; exit 1; }
export DDE_BACKEND=pytorch CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/.mplcache"
case "$1" in
  smoke)
    "$py" revision/campaign.py heat smoke
    "$py" revision/campaign.py burgers smoke
    ;;
  tune|formal)
    pids=()
    for lane in 0 1 2 3; do
      "$py" revision/campaign.py "$2" "$1" --batch "$SLURM_ARRAY_TASK_ID" --lane "$lane" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    exit "$failed"
    ;;
  freeze) "$py" revision/campaign.py "$2" freeze ;;
  evaluate)
    if [[ "$2" == heat ]]; then "$py" revision/fresh_heat.py original; fi
    "$py" revision/campaign.py "$2" evaluate
    ;;
  original) "$py" revision/fresh_heat.py original ;;
  *) echo "Unknown stage $1" >&2; exit 2 ;;
esac
