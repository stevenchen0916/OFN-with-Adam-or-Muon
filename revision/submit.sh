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
mode="${1:-core}"
case "$mode" in
  core) last_batch=8 ;;
  extend)
    for pde in heat burgers; do
      [[ -s "revision_runs/$pde/evaluation_core.json" ]] || {
        echo "Complete core for $pde before extending" >&2; exit 2;
      }
    done
    last_batch=21 ;;
  *) echo "Usage: bash revision/submit.sh [core|extend]" >&2; exit 2 ;;
esac
scope=core
[[ "$mode" == core ]] || scope=all
enqueue --job-name=ofn-smoke revision/server.slurm smoke both "$scope"
for pde in heat burgers; do
  if [[ "$mode" == core ]]; then
    enqueue --job-name="$pde-tune" --array=0-4%1 revision/server.slurm tune "$pde"
    enqueue --job-name="$pde-freeze" revision/server.slurm freeze "$pde"
  fi
  enqueue --job-name="$pde-formal" --array="0-${last_batch}%1" revision/server.slurm formal "$pde" "$scope"
  enqueue --job-name="$pde-eval" revision/server.slurm evaluate "$pde" "$scope"
done
