#!/bin/bash
#SBATCH --job-name=heat-fresh
#SBATCH --partition=T1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=heat-fresh-%j.out
set -euo pipefail
cd "${HEAT_PROJECT_DIR:-$HOME/HeatProject}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export DDE_BACKEND=pytorch CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/results_fresh_test_20261118/.mplcache"
mkdir -p "$MPLCONFIGDIR"
echo "Fresh-test replication: job=$SLURM_JOB_ID; 16 CPUs; 4 workers x 4 threads; no training"
exec .venv/bin/python -u fresh_test/replicate.py run --project "$PWD"
