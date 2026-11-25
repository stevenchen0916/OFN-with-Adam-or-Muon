#!/bin/bash
#SBATCH --job-name=heat16-formal
#SBATCH --partition=T1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=heat16-formal-%j.out
#SBATCH --signal=B:USR1@300
set -euo pipefail
cd "${HEAT_PROJECT_DIR:-$HOME/HeatProject}"
export HEAT_RESULTS_DIR="${HEAT_RESULTS_DIR:-$PWD/results16}"
HEAT_CHILD=''
heat_stop() {
  trap '' USR1 TERM INT
  if [[ -n "$HEAT_CHILD" ]]; then kill -INT "$HEAT_CHILD" 2>/dev/null || true; wait "$HEAT_CHILD" || true; fi
  echo 'Stopped with complete checkpoints; resubmit formal16.slurm to resume.'
  exit 130
}
trap heat_stop USR1 TERM INT
heat_stage() {
  bash run16.sh "$@" & HEAT_CHILD=$!
  wait "$HEAT_CHILD"
  HEAT_CHILD=''
}
echo "job=$SLURM_JOB_ID host=$(hostname) CPUs=$SLURM_CPUS_PER_TASK; 4 workers x 4 threads"
.venv/bin/python -c 'import json,os; from pathlib import Path; r=json.loads((Path(os.environ["HEAT_RESULTS_DIR"])/"resources.json").read_text()); assert (r["workers"],r["threads_per_worker"])==(4,4)'
heat_stage approve --note 'User authorized 16 CPU; reviewed 4x4 smoke checks and measured budget'
heat_stage sweep
heat_stage final
heat_stage evaluate
echo 'Formal workflow complete. Analysis: results16/analysis'
