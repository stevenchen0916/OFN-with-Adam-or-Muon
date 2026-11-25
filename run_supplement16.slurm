#!/bin/bash
#SBATCH --job-name=heat-extra
#SBATCH --partition=T1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=heat-extra-%j.out
#SBATCH --signal=B:USR1@300
set -euo pipefail
cd "${HEAT_PROJECT_DIR:-$HOME/HeatProject}"
export DDE_BACKEND=pytorch CUDA_VISIBLE_DEVICES=''
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export MPLCONFIGDIR="$PWD/results_supplement_20261118/.mplcache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
HEAT_EXTRA_CHILD=''
heat_extra_stop() {
  trap '' USR1 TERM INT
  if [[ -n "$HEAT_EXTRA_CHILD" ]]; then
    kill -INT "$HEAT_EXTRA_CHILD" 2>/dev/null || true
    wait "$HEAT_EXTRA_CHILD" || true
  fi
  echo 'Stopped; resubmit run_supplement16.slurm to resume separate continuation checkpoints.'
  exit 130
}
trap heat_extra_stop USR1 TERM INT
heat_extra_stage() {
  .venv/bin/python -u supplement/experiment.py "$@" & HEAT_EXTRA_CHILD=$!
  wait "$HEAT_EXTRA_CHILD"
  HEAT_EXTRA_CHILD=''
}
echo "Supplement job=$SLURM_JOB_ID; CPUs=$SLURM_CPUS_PER_TASK; 4 workers x 4 threads"
heat_extra_stage smoke --workers 1 --threads 4
heat_extra_stage diagnose --workers 4 --threads 4
heat_extra_stage continue --workers 4 --threads 4
.venv/bin/python supplement/plot.py results_supplement_20261118
heat_extra_stage export
echo 'Finished. Download results_supplement_20261118/补充实验_日志与诊断.zip'
