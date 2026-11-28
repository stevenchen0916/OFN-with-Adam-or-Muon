# Orthogonal Flow Networks for the heat and Burgers equations

This repository provides the heat-equation experiment code and the datasets and numerical results for two operator-learning tasks. Both experiments compare Adam with a Muon–Adam hybrid across nine configurations.

## Check the heat-equation example and replay the results

Create a Python environment and install the dependencies. The archived experiment used Python 3.12.16 and PyTorch 2.13.0 on CPU.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux.txt
python prepare_data.py --check-example
python verify_results.py --out replay
```

- `examples/test_case_000.npz` contains the first formal test case illustrated in the manuscript. It includes one diffusivity value (`kappa`), 100 initial-field observations (`u0`), and 8,000 reference temperatures (`solution`) aligned with the rows of `queries`. Each query row is ordered as `x, y, t`. The verification command recomputes the reference solution.
- `results/test_runs.csv` contains run-level test errors and timing metrics for the nine configurations and ten formal seeds. The replay command uses these measurements to recompute the main summary and paired contrasts.
- `results/validation_lr.csv` records the validation search scores. `results/selected_learning_rates.csv` records the learning rates used in the formal runs. `results/manifest.json` records the original environment and source hashes.

The replay yields hybrid-to-Adam geometric mean test-error ratios of 0.608 for Full OFN and 0.707 for No-Flow OFN. The ratio of the two effects is 0.860 (95% CI [0.808, 0.915]).

## Formal heat-equation dataset

The full dataset is included in [`heat_fair_v2/results/data/`](heat_fair_v2/results/data/). The directory contains `train.npz` (1,800 cases), `val.npz` (200 cases), and `test.npz` (200 cases). Each case has 8,000 reference values. The split archives contain diffusivities (`x1`), PCA features (`x2`), query coordinates (`xt`), reference temperatures (`y`), and sample indices (`original_ids`). The `preprocessing.npz` archive contains the PCA parameters and normalization bounds fitted on the training set. The `manifest.json` file records the generation seeds, split counts, and SHA-256 checksums.

To regenerate the dataset in a separate directory, run `python prepare_data.py --out runs/regenerated-data`. This command samples the Gaussian initial fields and diffusivities, solves the PDE, and applies the fixed split.

## Run the heat-equation experiment

The runner reads the included dataset from its default location.

```sh
python heat_formal_v3_16/runner.py --out runs/formal check
python heat_formal_v3_16/runner.py --out runs/formal verify
python heat_formal_v3_16/runner.py --out runs/formal resources --workers 4 --threads 4
python heat_formal_v3_16/runner.py --out runs/formal smoke
python heat_formal_v3_16/runner.py --out runs/formal approve --note "Reviewed local smoke results"
python heat_formal_v3_16/runner.py --out runs/formal sweep
python heat_formal_v3_16/runner.py --out runs/formal select
python heat_formal_v3_16/runner.py --out runs/formal final
python heat_formal_v3_16/runner.py --out runs/formal evaluate
```

Use the resource setting appropriate to your CPU before running the smoke test. The recorded experiment used four CPU workers with four threads each. The full run performs the learning-rate search and 90 formal training runs of 20,000 updates. Runtime and floating-point results can vary with the environment.

The optional `run16.sh` launcher is included for Slurm jobs allocated at least 16 CPUs. The direct Python commands above also work for local runs.

The archived protocol is in `heat_formal_v3_16/protocol.json`. The distributed data preparation follows that protocol and writes a fresh manifest with checksums for the regenerated arrays. The published runner recognizes that manifest and records its identity in new runs.

## Formal Burgers experiment

The [Burgers dataset](burgers/data/) contains 1,800 training cases, 200 validation cases, and 200 independently generated test cases for the forced periodic viscous Burgers equation. Each case includes 13 initial-condition values (`x1`), 13 forcing values (`x2`), and 2,500 reference solution values (`y`) indexed by the query features in `xt`. The directory also includes the training-derived normalization bounds, the split indices, and the checksummed dataset manifest.

See the [formal protocol](burgers/protocol.json) for the dataset definition and evaluation design. The [Burgers results](burgers/results/) contain the 90 run-level test results, paired contrasts, validation curves, and learning-rate search and selection records.

The [Burgers training entry point](burgers/BurgersProject/run_campaign.py), [paired analysis](burgers/BurgersProject/burgers_formal16/analysis.py), and [report generator](burgers/BurgersProject/report.py) accompany the published results.

## Additional OFN controls

The [revision instructions](revision/README.md) describe grouped Adam, joint and common learning-rate settings, normalization and fixed-gate controls, and the longer cosine schedule. From a clone, submit from the repository root with `bash revision/submit.sh`. The [leave-one-seed-out table](revision/burgers_leave_one_out.csv) uses the original Burgers results. The fresh heat evaluator uses the [90 original checkpoints and their checksums](revision/original_heat/index.json). New training and fresh-test scores are generated on the compute node.
