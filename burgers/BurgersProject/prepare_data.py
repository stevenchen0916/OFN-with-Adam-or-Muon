"""Prepare a fixed train/validation split and a fresh sealed test set; never score a model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))
import numpy as np
import generator as g

HERE = Path(__file__).resolve().parent
SOURCE_SHA = "a0fdf11ea1d356e369e82db8854f64249016ecf8ccfc7f182229800c882424cb"
SPLIT_SEED = 2026111801
GRF_SEED = 2026111802
AMPLITUDE_SEED = 2026111803


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def solve(args):
    u0, forcing, nx_solve = args
    sensors = np.arange(13, dtype=np.float32) / 13
    coords = np.arange(50, dtype=np.float32) / 50
    return g.solve_burgers_forced_periodic(.1, u0, forcing, sensors,
        x_out=coords, T=1., nx_solve=nx_solve, nt_solve=2000, cfl=.4, Nt=50).reshape(-1)


def prepare(source, workers):
    data = HERE / "support/results/data"
    if data.exists():
        manifest = json.loads((data / "manifest.json").read_text())
        if object_sha({k: v for k, v in manifest.items() if k != "dataset_id"}) != manifest["dataset_id"]:
            raise RuntimeError("Prepared manifest changed.")
        for name, expected in manifest["files"].items():
            if sha(data / name) != expected:
                raise RuntimeError("Prepared data changed: " + name)
        print("Existing prepared data verified; no regeneration.", flush=True)
        return manifest
    if sha(source) != SOURCE_SHA:
        raise RuntimeError("Unexpected legacy training source. Do not replace data under a frozen protocol.")
    # Object arrays are restricted to this explicitly identified, local legacy archive.
    with np.load(source, allow_pickle=True) as f:
        x1, x2, xt = [np.asarray(a, dtype=np.float32) for a in f["X_train"]]
        y = f["y_train"].astype(np.float32)
        nu = float(f["nu"])
    assert (x1.shape, x2.shape, xt.shape, y.shape) == ((2000, 13), (2000, 13), (2500, 5), (2000, 2500))
    assert abs(nu - .1) < 1e-7
    assert all(np.isfinite(a).all() for a in [x1, x2, xt, y])
    assert np.all(np.linalg.norm(y, axis=1) > 1e-12)
    spatial = np.arange(50, dtype=np.float32) / 50
    times = np.linspace(0, 1, 50, dtype=np.float32)
    expected = g.trunk_periodic_features(np.tile(spatial, 50), np.repeat(times, 50))
    np.testing.assert_allclose(xt, expected, atol=1e-6, rtol=1e-6)
    periodic0 = g.trunk_periodic_features(np.array([0., 1.]), np.array([.5, .5]))
    np.testing.assert_allclose(periodic0[0], periodic0[1], atol=1e-6)
    audit = []
    sensors = np.arange(13, dtype=np.float32) / 13
    for i in [0, 999, 1999]:
        initial = g.interp_periodic(x1[i], sensors, spatial)
        np.testing.assert_allclose(y[i, :50], initial, atol=1e-6)
        original, refined = solve((x1[i], x2[i], 256)), solve((x1[i], x2[i], 512))
        rel = float(np.linalg.norm(original - y[i]) / np.linalg.norm(y[i]))
        if rel > 1e-5:
            raise RuntimeError(f"Generator failed to reproduce target {i}: {rel}")
        audit.append(dict(index=i, generator_relative_l2=rel,
            refined_relative_l2=float(np.linalg.norm(refined - original) / np.linalg.norm(refined)),
            initial_max_difference=float(np.max(np.abs(y[i, :50] - initial)))))
        print("Audited training sample", audit[-1], flush=True)

    order = np.random.default_rng(SPLIT_SEED).permutation(2000)
    train_ids, val_ids = order[:1800], order[1800:]
    pending = data.with_name("data.pending")
    if pending.exists():
        raise RuntimeError("Incomplete data.pending exists; inspect before rerunning preparation.")
    pending.mkdir(parents=True)
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        np.savez_compressed(pending / f"{split}.npz", x1=x1[ids], x2=x2[ids], xt=xt, y=y[ids])
    bounds = {}
    for i, a in enumerate([x1[train_ids], x2[train_ids], xt]):
        bounds[f"low{i}"] = a.min(0)
        bounds[f"high{i}"] = a.max(0)
    np.savez_compressed(pending / "preprocessing.npz", **bounds)
    np.savez_compressed(pending / "split_indices.npz", train=train_ids, val=val_ids)
    # Same stored recipe as the original generator, with both RNG streams explicit.
    import deepxde as dde
    np.random.seed(GRF_SEED)
    space = dde.data.GRF(T=1., kernel="ExpSineSquared", length_scale=1.5, N=1000, interp="cubic")
    u0, forcing = g.build_inputs_from_grf(space, sensors, 200,
        u0_amp_range=(.05, .3), f_amp_range=(.05, .2), seed=AMPLITUDE_SEED)
    hashes = {hashlib.sha256(np.concatenate([a, b]).tobytes()).hexdigest() for a, b in zip(x1, x2)}
    fresh_hashes = [hashlib.sha256(np.concatenate([a, b]).tobytes()).hexdigest() for a, b in zip(u0, forcing)]
    if len(set(fresh_hashes)) != 200 or hashes.intersection(fresh_hashes):
        raise RuntimeError("Duplicate input functions in the new test set.")
    targets = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, target in enumerate(pool.map(solve, [(a, b, 256) for a, b in zip(u0, forcing)]), 1):
            targets.append(target)
            if i % 20 == 0:
                print(f"Generated sealed test targets {i}/200 ({time.perf_counter()-started:.1f}s)", flush=True)
    np.savez_compressed(pending / "test.npz", x1=u0, x2=forcing, xt=xt, y=np.stack(targets))
    manifest = dict(version="burgers-fresh-test-v1", source_training_sha=SOURCE_SHA,
        generator_sha=sha(HERE / "generator.py"), original_generator_sha=sha(HERE / "provenance/original_data_BU.py"),
        counts=dict(train=1800, val=200, test=200), split_seed=SPLIT_SEED,
        branch_dimensions=[13, 13], trunk_dimension=5, layout="t,x; x varies fastest",
        viscosity=.1, query_points=2500, preprocessing="train-only per-feature min/max; no PCA; raw dense baselines",
        fresh_test=dict(grf_seed=GRF_SEED, amplitude_seed=AMPLITUDE_SEED, n=200,
            recipe="ExpSineSquared GRF length_scale=1.5 N=1000 cubic; centered and RMS-normalized; u0 amplitude [.05,.3], f amplitude [.05,.2]",
            solver="original Rusanov + diffusion, nx=256 nt_nominal=2000 cfl=.4; first solver step reaching label time",
            legacy_test_opened=False, test_model_evaluations=0,
            numpy=str(np.__version__), deepxde=str(dde.__version__)),
        files={p.name: sha(p) for p in sorted(pending.glob("*.npz"))})
    manifest["dataset_id"] = object_sha(manifest)
    write(pending / "manifest.json", manifest)
    pending.rename(data)
    write(HERE / "provenance/DATA_AUDIT.json", dict(training_checks=audit,
        targets_reproduced=True, periodic_features_checked=True, initial_condition_checked=True,
        source_training_sha=SOURCE_SHA, dataset_id=manifest["dataset_id"],
        test_model_evaluations=0, legacy_test_opened=False,
        reference_limit="Three training examples only; refinement measures sensitivity of the specified discrete solver, not a certified continuum error bound."))
    print("Prepared dataset:", manifest["dataset_id"], flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.source, args.workers)
