"""Generate the heat-equation inputs and reference solutions from fixed seeds."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from heat_fair_v2.generator.data_heat2d_gaussian import (
    build_gaussian_initial_inputs,
    sample_kappa,
    solve_heat_2d_kappa,
)


ROOT = Path(__file__).resolve().parent
SENSOR_GRID = np.linspace(0, 1, 10, dtype=np.float32)
QUERY_GRID = np.linspace(0, 1, 20, dtype=np.float32)
QUERIES = np.stack(
    np.meshgrid(QUERY_GRID, QUERY_GRID, QUERY_GRID, indexing="xy"), axis=-1,
).reshape(-1, 3)
DATA = ROOT / "heat_fair_v2/results/data"


def sample_inputs(count, offset):
    kappa = sample_kappa(count, 0.0005, 0.05, seed=1234 + offset)
    sensors, _ = build_gaussian_initial_inputs(
        SENSOR_GRID, SENSOR_GRID, count, shape_mode="elliptic",
        amp_range=(0.8, 1.2), x0_range=(0.3, 0.7), y0_range=(0.3, 0.7),
        sigma_x_range=(0.15, 0.30), sigma_y_range=(0.15, 0.30),
        seed=5678 + offset,
    )
    return kappa, sensors


def solve_case(kappa, sensors):
    fields = solve_heat_2d_kappa(
        float(kappa), sensors, SENSOR_GRID, SENSOR_GRID, QUERY_GRID, QUERY_GRID,
        t_eval=QUERY_GRID,
    )
    return fields.transpose(1, 2, 0).reshape(-1)


def verify_example():
    with np.load(ROOT / "examples/test_case_000.npz") as example:
        kappa, sensors = sample_inputs(200, 100000)
        np.testing.assert_array_equal(kappa[0], example["kappa"])
        np.testing.assert_array_equal(sensors[0], example["u0"])
        np.testing.assert_array_equal(QUERIES, example["queries"])
        np.testing.assert_allclose(
            solve_case(kappa[0, 0], sensors[0]), example["solution"],
            rtol=1e-6, atol=1e-7,
        )
    print(f"Verified test case 0 at {len(QUERIES):,} query points.")


def solve_pool(kappa, sensors, name):
    fields = np.empty((len(kappa), len(QUERIES)), dtype=np.float32)
    for index in tqdm(range(len(kappa)), desc=f"Solving {name}"):
        fields[index] = solve_case(kappa[index, 0], sensors[index])
    return fields


def prepare_data(data_dir=DATA):
    if data_dir.exists():
        raise FileExistsError(f"Prepared data already exists at {data_dir}.")
    protocol = json.loads((ROOT / "heat_formal_v3_16/protocol.json").read_text())
    kappa, sensors = sample_inputs(2000, 0)
    test_kappa, test_sensors = sample_inputs(200, 100000)
    order = np.random.default_rng(2026).permutation(len(kappa))
    train_ids, val_ids = np.sort(order[:1800]), np.sort(order[1800:])
    pca = PCA(n_components=10, svd_solver="full").fit(sensors[train_ids])
    features = pca.transform(sensors).astype(np.float32)
    test_features = pca.transform(test_sensors).astype(np.float32)
    fields = solve_pool(kappa, sensors, "training pool")
    test_fields = solve_pool(test_kappa, test_sensors, "test pool")

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=data_dir.parent) as temporary:
        output = Path(temporary) / "data"
        output.mkdir()
        for split, ids in (("train", train_ids), ("val", val_ids)):
            np.savez_compressed(
                output / f"{split}.npz", x1=kappa[ids], x2=features[ids],
                xt=QUERIES, y=fields[ids], original_ids=ids,
            )
        np.savez_compressed(
            output / "test.npz", x1=test_kappa, x2=test_features, xt=QUERIES,
            y=test_fields, original_ids=np.arange(len(test_kappa)),
        )
        bounds = {}
        for index, values in enumerate((kappa[train_ids], features[train_ids], QUERIES)):
            bounds[f"low{index}"] = values.min(axis=0)
            bounds[f"high{index}"] = values.max(axis=0)
        np.savez_compressed(
            output / "preprocessing.npz", pca_mean=pca.mean_,
            pca_components=pca.components_, **bounds,
        )
        manifest = {
            "source_dataset_id": protocol["data"]["dataset_id"],
            "function_seeds": {"train": [1234, 5678], "test": [101234, 105678]},
            "split_seed": 2026,
            "counts": {"train": 1800, "val": 200, "test": 200},
            "target_order": "y,x,t",
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.glob("*.npz"))
            },
        }
        manifest["dataset_id"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode(),
        ).hexdigest()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(data_dir)
    print(f"Prepared 1,800 training, 200 validation, and 200 test cases in {data_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-example", action="store_true")
    parser.add_argument("--out", type=Path, default=DATA, help="Directory for regenerated data.")
    args = parser.parse_args()
    verify_example()
    if not args.check_example:
        prepare_data(args.out)
