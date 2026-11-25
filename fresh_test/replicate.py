"""Independent test-input replication. This program contains no training step."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
WORKER = {}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def csv_write(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def locked(out):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another fresh-test process is using this results directory.") from e
        yield


def event(out, name, **fields):
    with (out / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time_unix=time.time(), event=name, **fields), ensure_ascii=False) + "\n")


def imports(project, threads=4):
    project = Path(project)
    runner = project / "heat_formal_v3_16"
    if not runner.is_dir():
        raise RuntimeError(f"Existing 16-CPU runner is missing: {runner}")
    sys.path.insert(0, str(runner))
    import model_config as m
    import analysis as a
    import control as k
    import torch
    torch.set_default_device("cpu")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return m, a, k, torch


def freeze(project, original, out):
    """Verify all 90 final checkpoints before any new samples or scores exist."""
    m, _, k, torch = imports(project)
    p = read(HERE / "protocol.json")
    training = k.protocol()
    if training["protocol_id"] != p["frozen_training_protocol_id"]:
        raise RuntimeError("The training protocol differs from the planned replication.")
    if training["formal"]["seeds"] != p["training_seeds"]:
        raise RuntimeError("Expected the original ten formal seeds, 292--301.")
    selected = read(original / "selection.json")
    old_manifest = read(original / "manifest.json")
    if old_manifest["protocol_id"] != training["protocol_id"]:
        raise RuntimeError("Original result manifest has a different protocol.")
    for name, expected in old_manifest["code"]["local"].items():
        if digest(project / "heat_formal_v3_16" / name) != expected:
            raise RuntimeError(f"Original runner source changed: {name}")
    if m.c.code_hash() != old_manifest["code"]["old_helper_sha"]:
        raise RuntimeError("Original model/data helper sources changed since training.")
    current_software = dict(torch=str(torch.__version__), deepxde=str(m.c.dde.__version__),
                            numpy=str(m.c.np.__version__), python=__import__("platform").python_version())
    if current_software != old_manifest["software"]:
        raise RuntimeError("Use the original Python/PyTorch/DeepXDE/NumPy environment.")
    if set(selected["configs"]) != set(training["groups"]):
        raise RuntimeError("The frozen selection must contain all nine configurations.")
    if selected["context"]["manifest_sha"] != k.object_sha(old_manifest):
        raise RuntimeError("Frozen selection does not belong to the original manifest.")
    evidence = dict(selected.get("source_hashes", {}))
    for config in selected["configs"].values():
        evidence.update(config.get("source_hashes", {}))
    for name, expected in evidence.items():
        if digest(original / name) != expected:
            raise RuntimeError(f"Frozen learning-rate selection evidence changed: {name}")
    data = project / "heat_fair_v2/results/data"
    data_manifest = read(data / "manifest.json")
    if m.c.object_hash({k: v for k, v in data_manifest.items() if k != "dataset_id"}) != data_manifest["dataset_id"]:
        raise RuntimeError("Original data manifest body changed.")
    if data_manifest["dataset_id"] != training["data"]["dataset_id"]:
        raise RuntimeError("Original preprocessing/data identity differs from training.")
    if digest(data / "preprocessing.npz") != data_manifest["files"]["preprocessing.npz"]:
        raise RuntimeError("Original training-only preprocessing changed.")
    old_seeds = sum(data_manifest["function_seeds"].values(), [])
    if p["kappa_seed"] in old_seeds or p["initial_field_seed"] in old_seeds:
        raise RuntimeError("Fresh function seeds must differ from both original data pools.")
    jobs = []
    for seed in p["training_seeds"]:
        for group in training["groups"]:
            lr = selected["configs"][group]
            path = k.run_path(original, "formal", group, lr["adam_lr"], lr["muon_lr"], seed)
            result = read(path / "result.json")
            identity = result["identity"]
            required = dict(stage="formal", group=group, seed=seed, adam_lr=lr["adam_lr"],
                            muon_lr=lr["muon_lr"], total_steps=20000, horizon=20000)
            if any(identity.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"Incorrect formal checkpoint identity: {path}")
            if identity["context"] != selected["context"]:
                raise RuntimeError(f"Checkpoint execution context differs: {path}")
            if result["status"] != "completed" or result["actual_steps"] != 20000:
                raise RuntimeError(f"Expected all 90 completed 20,000-step runs: {path}")
            checkpoint = path / "final.pt"
            if digest(checkpoint) != result["checkpoint_sha"]:
                raise RuntimeError(f"Final checkpoint changed: {checkpoint}")
            jobs.append(dict(group=group, seed=seed, checkpoint=str(checkpoint),
                             checkpoint_sha=result["checkpoint_sha"], result_sha=digest(path / "result.json"),
                             parameters=result["parameters"]))
    source_hashes = {str(f.relative_to(project)): digest(f)
                     for base in [project / "heat_formal_v3_16", project / "heat_fair_v2"]
                     for f in sorted(base.rglob("*.py"))}
    value = dict(protocol=p, protocol_sha=digest(HERE / "protocol.json"),
                 replication_code_sha=digest(Path(__file__)), original_results=str(original),
                 selection_sha=digest(original / "selection.json"), original_manifest_sha=digest(original / "manifest.json"),
                 original_dataset_id=data_manifest["dataset_id"],
                 preprocessing_sha=digest(data / "preprocessing.npz"),
                 source_hashes=source_hashes, software=old_manifest["software"], jobs=jobs)
    marker = out / "frozen_inputs.json"
    if marker.exists():
        if read(marker) != value:
            raise RuntimeError("Frozen replication inputs changed; refusing to mix outputs.")
    else:
        write(marker, value)
        event(out, "inputs_frozen", models=len(jobs), n_test=p["n_test"])
    print(f"Verified and frozen: {len(jobs)} checkpoints; original files are read-only inputs.", flush=True)
    return value


def generator_import(project):
    sys.path.insert(0, str(Path(project) / "heat_fair_v2"))
    from generator import data_heat2d_gaussian as g
    return g


def raw_inputs(project, n, kappa_seed, field_seed):
    import numpy as np
    g = generator_import(project)
    sensor = np.linspace(0, 1, 10, dtype=np.float32)
    grid = np.linspace(0, 1, 20, dtype=np.float32)
    times = np.linspace(0, 1, 20, dtype=np.float32)
    kappa = g.sample_kappa(n, .0005, .05, seed=kappa_seed)
    initial, parameters = g.build_gaussian_initial_inputs(
        sensor, sensor, n, shape_mode="elliptic", amp_range=(.8, 1.2),
        x0_range=(.3, .7), y0_range=(.3, .7), sigma_x_range=(.15, .30),
        sigma_y_range=(.15, .30), seed=field_seed)
    queries = np.stack(np.meshgrid(grid, grid, times, indexing="xy"), axis=-1).reshape(-1, 3)
    return sensor, grid, times, kappa, initial, parameters, queries


def solve_one(task):
    project, index, kappa, initial = task
    import numpy as np
    g = generator_import(project)
    sensor = np.linspace(0, 1, 10, dtype=np.float32)
    grid = np.linspace(0, 1, 20, dtype=np.float32)
    times = np.linspace(0, 1, 20, dtype=np.float32)
    snapshots = g.solve_heat_2d_kappa(kappa, initial, sensor, sensor, grid, grid,
                                   T=1, nx_solve=64, ny_solve=64, t_eval=times)
    return index, snapshots.transpose(1, 2, 0).reshape(-1)


def validate_arrays(queries, targets):
    import numpy as np
    if queries.shape != (8000, 3) or targets.ndim != 2 or targets.shape[1] != 8000:
        raise RuntimeError("Unexpected query/target shapes.")
    if not np.isfinite(targets).all() or not np.isfinite(queries).all():
        raise RuntimeError("Nonfinite numerical target.")
    boundary = (queries[:, 0] == 0) | (queries[:, 0] == 1) | (queries[:, 1] == 0) | (queries[:, 1] == 1)
    if np.max(np.abs(targets[:, boundary])) > 1e-7:
        raise RuntimeError("Boundary check failed: inspect physical coordinate/flatten ordering.")
    if np.any(np.linalg.norm(targets.astype(np.float64), axis=1) <= 0):
        raise RuntimeError("Zero reference solution norm.")


def generate(project, out, frozen):
    import numpy as np
    p = frozen["protocol"]
    path, marker = out / "data/test.npz", out / "data/manifest.json"
    key = object_hash(frozen)
    if marker.exists():
        v = read(marker)
        if v["frozen_inputs_sha"] != key or digest(path) != v["test_sha"]:
            raise RuntimeError("Fresh test cache changed.")
        print("Reusing the same frozen fresh test set.", flush=True)
        return v
    start = time.perf_counter()
    _, _, _, kappa, initial, params, queries = raw_inputs(
        project, p["n_test"], p["kappa_seed"], p["initial_field_seed"])
    pending = out / "data/solutions"
    pending.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(p["n_test"]):
        if not (pending / f"{i:03d}.npy").exists():
            tasks.append((str(project), i, float(kappa[i, 0]), initial[i]))
    with ProcessPoolExecutor(max_workers=p["workers"], mp_context=mp.get_context("spawn")) as pool:
        for done, (i, values) in enumerate(pool.map(solve_one, tasks), 1):
            dest = pending / f"{i:03d}.npy"
            with dest.with_suffix(".tmp").open("wb") as f:
                np.save(f, values)
            dest.with_suffix(".tmp").replace(dest)
            if done % 20 == 0 or done == len(tasks):
                print(f"DATA: {done}/{len(tasks)} pending solutions completed", flush=True)
    targets = np.stack([np.load(pending / f"{i:03d}.npy") for i in range(p["n_test"])])
    validate_arrays(queries, targets)
    with np.load(project / "heat_fair_v2/results/data/preprocessing.npz") as fitted:
        coefficients = ((initial - fitted["pca_mean"]) @ fitted["pca_components"].T).astype(np.float32)
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, x1=kappa, x2=coefficients, xt=queries, y=targets, initial_observations=initial)
    path.with_suffix(".tmp").replace(path)
    write(out / "data/initial_parameters.json", params)
    value = dict(frozen_inputs_sha=key, test_sha=digest(path), n_test=p["n_test"],
                 kappa_seed=p["kappa_seed"], initial_field_seed=p["initial_field_seed"],
                 target_order="y,x,t; snapshots.transpose(1,2,0).reshape(-1)",
                 original_preprocessing_sha=frozen["preprocessing_sha"], generation_seconds=time.perf_counter()-start)
    write(marker, value)
    print(f"Fresh test data ready: {p['n_test']} input pairs, {value['generation_seconds']:.1f} seconds.", flush=True)
    return value


def init_evaluator(project, out, frozen_key, test_sha):
    import numpy as np
    m, _, _, torch = imports(project)
    data = Path(out) / "data/test.npz"
    if digest(data) != test_sha:
        raise RuntimeError("Fresh test data changed before inference.")
    with np.load(data) as f:
        raw = [f[k] for k in ("x1", "x2", "xt")]
        target = torch.as_tensor(f["y"])
    with np.load(Path(project) / "heat_fair_v2/results/data/preprocessing.npz") as f:
        bounds = [(f[f"low{i}"], f[f"high{i}"]) for i in range(3)]
    raw_tensors = tuple(torch.as_tensor(x) for x in raw)
    qofn_tensors = tuple(torch.as_tensor(m.c.safe_minmax(x, *b)) for x, b in zip(raw, bounds))
    WORKER.update(m=m, torch=torch, x=raw_tensors, xq=qofn_tensors, target=target,
                  out=Path(out), frozen_key=frozen_key, test_sha=test_sha)


def score_one(job):
    w, start = WORKER, time.perf_counter()
    m, torch = w["m"], w["torch"]
    if digest(job["checkpoint"]) != job["checkpoint_sha"]:
        raise RuntimeError("Frozen checkpoint changed during evaluation.")
    m.c.seed_all(job["seed"])
    net = m.build_model(job["group"])
    net.load_state_dict(torch.load(job["checkpoint"], map_location="cpu", weights_only=True))
    net.eval()
    x = w["xq"] if m.c.is_qofn(job["group"]) else w["x"]
    with torch.no_grad():
        prediction = net(m.c.inputs_for(job["group"], x))
        values = (prediction - w["target"]).double().norm(dim=1) / w["target"].double().norm(dim=1).clamp_min(1e-12)
    if not bool(torch.isfinite(values).all()) or float(values.mean()) <= 0:
        raise RuntimeError(f"Nonfinite values or nonpositive mean error for log analysis: {job['group']}, {job['seed']}")
    row = dict(group=job["group"], seed=job["seed"], checkpoint_sha=job["checkpoint_sha"],
               frozen_inputs_sha=w["frozen_key"], fresh_test_sha=w["test_sha"],
               test_relative_l2=float(values.mean()), per_solution_relative_l2=values.tolist(),
               evaluation_seconds=time.perf_counter()-start)
    dest = w["out"] / "scores" / f"{job['group']}_seed_{job['seed']}.json"
    write(dest, row)
    return row


def evaluate(project, out, frozen, data):
    key = object_hash(frozen)
    rows, tasks = [], []
    for job in frozen["jobs"]:
        path = out / "scores" / f"{job['group']}_seed_{job['seed']}.json"
        if path.exists():
            row = read(path)
            if (row["frozen_inputs_sha"], row["fresh_test_sha"], row["checkpoint_sha"]) != (
                    key, data["test_sha"], job["checkpoint_sha"]):
                raise RuntimeError(f"Model score cache identity changed: {path}")
            if row["group"] != job["group"] or row["seed"] != job["seed"]:
                raise RuntimeError(f"Model score cache group/seed changed: {path}")
            rows.append(row)
        else:
            tasks.append(job)
    event(out, "evaluation_start_or_resume", cached=len(rows), pending=len(tasks))
    if tasks:
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
                                 initializer=init_evaluator, initargs=(str(project), str(out), key, data["test_sha"])) as pool:
            futures = [pool.submit(score_one, job) for job in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"TEST {len(rows)}/90: {row['group']} seed={row['seed']} error={100*row['test_relative_l2']:.4f}%", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["group"]))
    if len(rows) != 90:
        raise RuntimeError("Exactly 90 model evaluations are required; no seed replacement or omission.")
    return rows


def summarize(project, out, frozen, rows):
    import statistics
    _, a, _, _ = imports(project)
    groups = sorted({r["group"] for r in rows})
    by = {(r["group"], r["seed"]): r["test_relative_l2"] for r in rows}
    csv_write(out / "analysis/test_runs.csv", [{k: v for k, v in r.items() if k != "per_solution_relative_l2"} for r in rows])
    table = []
    for group in groups:
        values = [by[group, s] for s in frozen["protocol"]["training_seeds"]]
        table.append(dict(group=group, n_training_seeds=len(values), mean_percent=100*statistics.mean(values),
                          sample_sd_percent=100*statistics.stdev(values)))
    csv_write(out / "analysis/main_table.csv", table)
    effects = []
    raw = {}
    for name in ["deeponet", "mionet", "full", "noflow", "deeponet_output"]:
        adam = "deeponet_adam" if name == "deeponet_output" else name + "_adam"
        muon = name + "_muon_adam"
        values = [math.log(by[muon, s]/by[adam, s]) for s in frozen["protocol"]["training_seeds"]]
        raw[name] = values
        stats = a.contrast_summary(values)
        stats["confirmatory_t_p"] = None
        effects.append(dict(contrast=name, role="secondary_unadjusted", **stats,
                            geometric_error_ratio=math.exp(stats["mean"])))
    interactions = [f-n for f, n in zip(raw["full"], raw["noflow"])]
    stats = a.contrast_summary(interactions)
    primary = dict(contrast=frozen["protocol"]["primary_contrast"], role="primary_new_test_input_replication",
                   interpretation=frozen["protocol"]["interpretation"], **stats,
                   ratio_of_optimizer_ratios=math.exp(stats["mean"]),
                   ratio_ci_low=math.exp(stats["t_ci_low"]) if stats["t_ci_low"] is not None else None,
                   ratio_ci_high=math.exp(stats["t_ci_high"]) if stats["t_ci_high"] is not None else None)
    write(out / "analysis/primary_I.json", primary)
    write(out / "analysis/paired_effects.json", effects)
    csv_write(out / "analysis/seed_interactions.csv", [dict(seed=s, log_full=f, log_noflow=n, I_log=f-n)
              for s, f, n in zip(frozen["protocol"]["training_seeds"], raw["full"], raw["noflow"])])
    lines = ["独立测试输入复核结果", "", "200 个新测试输入；9 配置 × 10 个原有训练种子；只推理，不重新训练/调参。", "",
             "误差单位：百分比；SD 为训练种子间样本标准差。"]
    lines += [f"{r['group']}: {r['mean_percent']:.3f} ± {r['sample_sd_percent']:.3f}" for r in table]
    lines += ["", f"主要平均 log 交互 = {primary['mean']:.6f}",
              f"95% CI = [{primary['t_ci_low']}, {primary['t_ci_high']}]",
              f"交互比 = {primary['ratio_of_optimizer_ratios']:.6f}",
              f"配对 t p = {primary['paired_t_p']}; Wilcoxon p = {primary['wilcoxon']['pvalue']}", "",
              "区间以原有 10 个训练种子为统计单位，条件于这一个新测试集；不包含数据划分或问题分布间不确定性。",
              "本次是新测试输入上的复核，并未重新独立训练。无论方向和显著性如何，完整报告结果；不据此继续挑选测试集、种子或模型。"]
    (out / "复核结果.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def export(out):
    dest = out / "独立测试复核_供核对.zip"
    files = [p for p in sorted(out.rglob("*")) if p.is_file() and p != dest and p.name != ".lock"
             and "solutions" not in p.parts and ".mplcache" not in p.parts and not p.name.endswith(".tmp")]
    with ZipFile(dest.with_suffix(".tmp"), "w", ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, str(Path(out.name) / path.relative_to(out)))
    dest.with_suffix(".tmp").replace(dest)
    with ZipFile(dest) as z:
        if z.testzip() is not None:
            raise RuntimeError("Result archive checksum verification failed.")
    print(f"Result archive: {dest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--project", type=Path, default=HERE.parent)
    args = parser.parse_args()
    project = args.project.resolve()
    out = project / "results_fresh_test_20261118"
    if args.command == "status":
        marker = out / "status.json"
        print(json.dumps(read(marker) if marker.exists() else dict(stage="not_started"), ensure_ascii=False, indent=2))
        print("Completed score files:", len(list((out / "scores").glob("*.json"))))
        return
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 16:
        raise RuntimeError("Submit run_fresh_test16.slurm; numeric work must run on an allocated 16-CPU compute node.")
    started = time.perf_counter()
    with locked(out):
        try:
            event(out, "invocation_start", job_id=os.environ["SLURM_JOB_ID"],
                  host=__import__("socket").gethostname(), cpus=os.environ["SLURM_CPUS_PER_TASK"])
            write(out / "status.json", dict(stage="verifying_original_models", job_id=os.environ["SLURM_JOB_ID"]))
            frozen = freeze(project, project / "results16", out)
            m, _, _, torch = imports(project)
            write(out / "evaluation_hardware.json", m.c.environment(torch.device("cpu")))
            write(out / "status.json", dict(stage="generating_fresh_test", models=90, n_test=200))
            data = generate(project, out, frozen)
            write(out / "status.json", dict(stage="evaluating", models=90, n_test=200))
            rows = evaluate(project, out, frozen, data)
            summarize(project, out, frozen, rows)
            # Recheck all sources/models after scoring, including cached runs.
            freeze(project, project / "results16", out)
            write(out / "status.json", dict(stage="completed", n_models=90, n_test=200,
                                            invocation_seconds=time.perf_counter()-started))
            export(out)
        except BaseException as error:
            write(out / "status.json", dict(stage="stopped", error_type=type(error).__name__, message=str(error)))
            event(out, "stopped", error_type=type(error).__name__, message=str(error))
            raise


if __name__ == "__main__":
    main()
