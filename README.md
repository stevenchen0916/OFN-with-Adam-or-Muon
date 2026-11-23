# Orthogonal Flow Networks for the heat equation

Code and numerical results for *Architecture–Optimizer Interactions in Orthogonal Flow Networks for Heat Equation Operator Learning*. The experiment compares Adam with a Muon–Adam hybrid across nine configurations on a source-free two-dimensional heat-equation task.

## Check the example and replay the results

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

## Regenerate the complete dataset and run the experiment

`python prepare_data.py` samples the Gaussian initial fields and diffusivities, solves the PDE, and applies the fixed split of 1,800 training, 200 validation, and 200 test cases. PCA and normalization bounds are computed using the training split. The generated arrays and their checksums are written to `heat_fair_v2/results/data/`.

```sh
python prepare_data.py
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

The archived protocol is in `heat_formal_v3_16/protocol.json`. The distributed data preparation follows that protocol and writes a fresh manifest with checksums for the regenerated arrays. The published runner recognizes that manifest and records its identity in new runs.
