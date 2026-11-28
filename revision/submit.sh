#!/usr/bin/env bash
# Queue serial 16-core allocations. Each array element handles up to eight runs.
set -euo pipefail
cd "$(dirname "$0")/.."
dependency=()
enqueue() {
  local id
  id=$(sbatch --parsable "${dependency[@]}" "$@")
  id=${id%%;*}
  echo "Submitted job $id ($*)"
  dependency=("--dependency=afterok:$id")
}
enqueue --job-name=ofn-smoke revision/server.slurm smoke
for pde in heat burgers; do
  enqueue --job-name="$pde-tune" --array=0-48%1 revision/server.slurm tune "$pde"
  enqueue --job-name="$pde-freeze" revision/server.slurm freeze "$pde"
  enqueue --job-name="$pde-formal" --array=0-27%1 revision/server.slurm formal "$pde"
  enqueue --job-name="$pde-eval" revision/server.slurm evaluate "$pde"
done
