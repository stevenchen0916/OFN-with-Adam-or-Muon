"""Read-only source campaign, fixed diagnostic probes, separate resumable outputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
STOP = None
CACHE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def locked(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another process owns {path}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_modules(project, threads):
    project = Path(project).resolve()
    if "modules" not in CACHE:
        os.environ.setdefault("DDE_BACKEND", "pytorch")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        sys.path.insert(0, str(project / "heat_formal_v3_16"))
        import engine as e
        import model_config as m
        import torch
        torch.set_default_device("cpu")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        CACHE["modules"] = (m, e, torch)
    return CACHE["modules"]


def init_worker(project, threads, stop):
    global STOP
    STOP = stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    load_modules(project, threads)


def dataset(project):
    if "data" not in CACHE:
        m, _, torch = CACHE["modules"]
        CACHE["data"] = m.c.Dataset(Path(project) / "heat_fair_v2/results", torch.device("cpu"))
    return CACHE["data"]


def source_jobs(project):
    """Verify all sources before making any scientific outputs."""
    project = Path(project).resolve()
    p = read(HERE / "protocol.json")
    root = project / "results16"
    manifest = read(root / "manifest.json")
    source_code = project / "heat_formal_v3_16"
    for name, expected in manifest["code"]["local"].items():
        if sha(source_code / name) != expected:
            raise RuntimeError(f"Original code changed: {name}")
    helper = project / "heat_fair_v2"
    helper_hashes = {str(f.relative_to(helper)): sha(f) for f in sorted(helper.rglob("*.py"))
                     if "." not in f.relative_to(helper).parts[0]}
    helper_hash = hashlib.sha256(json.dumps(helper_hashes, sort_keys=True).encode()).hexdigest()
    if helper_hash != manifest["code"]["old_helper_sha"]:
        raise RuntimeError("Original helper/model code differs from the frozen campaign.")
    selection = read(root / "selection.json")
    data_manifest = read(helper / "results/data/manifest.json")
    if data_manifest["dataset_id"] != manifest["dataset_id"]:
        raise RuntimeError("Source data identity mismatch.")
    for name in ["preprocessing.npz", "train.npz", "val.npz"]:
        if sha(helper / "results/data" / name) != data_manifest["files"][name]:
            raise RuntimeError(f"Changed original data: {name}")
    jobs = []
    for group in p["groups"]:
        for seed in p["seeds"]:
            matches = list((root / "formal" / group).glob(f"*/seed_{seed}/result.json"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one completed source for {group}, seed={seed}")
            directory = matches[0].parent
            result = read(matches[0])
            ident = result["identity"]
            selected = selection["configs"][group]
            if (result["status"] != "completed" or result["actual_steps"] != 20000
                or ident["group"] != group or ident["seed"] != seed
                or ident["context"]["dataset_id"] != manifest["dataset_id"]
                or any(ident[k] != selected[k] for k in ["adam_lr", "muon_lr"])):
                raise RuntimeError(f"Invalid source result: {directory}")
            if sha(directory / "final.pt") != result["checkpoint_sha"]:
                raise RuntimeError(f"Invalid final model hash: {directory}")
            jobs.append(dict(group=group, seed=seed, source=str(directory.relative_to(project)),
                             final_sha=sha(directory / "final.pt"), latest_sha=sha(directory / "latest.pt"),
                             result_sha=sha(matches[0]), identity=ident))
    frozen = dict(protocol=p, protocol_sha=sha(HERE / "protocol.json"),
                  supplement_sources={f.name: sha(f) for f in sorted(HERE.glob("*.py"))},
                  execution_software={name: importlib.metadata.version(name) for name in ["torch", "numpy", "deepxde"]},
                  execution_python=sys.version,
                  dataset_id=manifest["dataset_id"], original_manifest_sha=sha(root / "manifest.json"),
                  selection_sha=sha(root / "selection.json"), jobs=jobs)
    return frozen


def freeze(project, out):
    value = source_jobs(project)
    path = Path(out) / "frozen_inputs.json"
    if path.exists() and read(path) != value:
        raise RuntimeError("Sources or supplement code changed; use a new output directory.")
    write(path, value)
    return value


def indices(n, count, torch):
    return torch.linspace(0, n - 1, min(n, count)).long()


def fixed_probe(data, split, functions, coordinates, torch):
    x, y = data.load(split, True)
    fi, qi = indices(len(y), functions, torch), indices(y.shape[1], coordinates, torch)
    return (x[0][fi], x[1][fi], x[2][qi]), y[fi][:, qi], fi.tolist(), qi.tolist()


def layer_jacobians(encoder, index, values, torch):
    """Exact dense local Jacobian of the actual normalized affine/Flow/ReLU map."""
    layer = encoder.hidden_layers[index]
    n = layer.in_features
    eye = torch.eye(n, dtype=values.dtype)
    matrix = layer(eye) - layer(torch.zeros(1, n))
    radius = values.norm(dim=1)
    safe_radius = radius.clamp_min(1e-12)
    dn = eye.unsqueeze(0) / safe_radius[:, None, None]
    dn = dn - (values[:, :, None] * values[:, None, :]
               / safe_radius[:, None, None].pow(3)) * (radius >= 1e-12)[:, None, None]
    pre_j = matrix.T.unsqueeze(0) @ dn
    active = bool(encoder.flow and layer.in_features == layer.out_features)
    if active:
        gate = torch.sigmoid(encoder.h_params[index])
        pre_j = (1 - gate) * eye.unsqueeze(0) + gate * pre_j
    pre = encoder.hidden_preactivation(values, index)
    post_j = (pre > 0).to(values.dtype)[:, :, None] * pre_j
    return pre, pre_j, post_j, active


def diagnostics(net, opts, data, protocol, torch):
    """No optimizer steps; fixed small training gradient probe and validation layer probes."""
    x, target, fi, qi = fixed_probe(data, "train", protocol["gradient_probe_functions"],
                                  protocol["gradient_probe_coordinates"], torch)
    net.eval()
    net.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(net(x), target)
    loss.backward()
    parameter_rows = []
    opt_state = {p: opt.state.get(p, {}) for opt in opts for pg in opt.param_groups for p in pg["params"]}
    for name, param in net.named_parameters():
        if param.grad is None:
            parameter_rows.append(dict(name=name, gradient_norm=None, gradient_absent=True))
            continue
        row = dict(name=name, gradient_norm=float(param.grad.norm()), gradient_absent=False)
        if name.endswith("output_layer.weight"):
            row["gradient_singular_values"] = torch.linalg.svdvals(param.grad).tolist()
            momentum = opt_state.get(param, {}).get("momentum_buffer")
            row["muon_momentum_singular_values"] = (torch.linalg.svdvals(momentum).tolist()
                                                    if momentum is not None else None)
        parameter_rows.append(row)
    xv, _ = data.load("val", True)
    layers = []
    with torch.no_grad():
        for name, inputs in zip(["branch1", "branch2", "trunk"], xv):
            encoder = getattr(net, name)
            ids = indices(len(inputs), protocol["diagnostic_samples_per_encoder"], torch)
            values = inputs[ids]
            for i, layer in enumerate(encoder.hidden_layers):
                pre, pre_j, post_j, active = layer_jacobians(encoder, i, values, torch)
                post = encoder.activation(pre)
                direction = values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)
                radial = (post_j @ direction[:, :, None]).squeeze(-1).norm(dim=1)
                layers.append(dict(encoder=name, layer=i, gate_active=active,
                    alpha=float(torch.sigmoid(encoder.h_params[i])) if encoder.flow else None,
                    input_norm=values.norm(dim=1).tolist(), output_norm=post.norm(dim=1).tolist(),
                    pre_singular_values=torch.linalg.svdvals(pre_j).tolist(),
                    post_singular_values=torch.linalg.svdvals(post_j).tolist(),
                    post_radial_gain=radial.tolist(), input_ids=ids.tolist()))
                values = post
    net.zero_grad(set_to_none=True)
    return dict(training_probe_mse=float(loss.detach()), training_function_ids=fi,
                training_coordinate_ids=qi, parameters=parameter_rows, layers=layers)


def restore_source(job, project):
    m, e, torch = CACHE["modules"]
    source = Path(project) / job["source"]
    for name, expected in [("latest.pt", job["latest_sha"]), ("final.pt", job["final_sha"])]:
        if sha(source / name) != expected:
            raise RuntimeError(f"Source changed after freezing: {source / name}")
    saved = torch.load(source / "latest.pt", map_location="cpu", weights_only=True)
    if saved["step"] != 20000 or saved["identity"] != job["identity"]:
        raise RuntimeError("Continuation requires full step-20000 optimizer/RNG checkpoint.")
    net = m.build_model(job["group"])
    ident = job["identity"]
    opts, _ = e.make_optimizers(net, job["group"], ident["adam_lr"], ident["muon_lr"])
    scheds = e.schedulers(opts, 20000, .01)
    e.load_pack(saved, net, opts, scheds)
    final = torch.load(source / "final.pt", map_location="cpu", weights_only=True)
    for name, value in net.state_dict().items():
        if not torch.equal(value, final[name]):
            raise RuntimeError("Final model and full optimizer checkpoint disagree.")
    e.lr_check(opts, scheds, 20000, 20000, .01)
    return net, opts, scheds, saved


def diagnose_job(job, project, out):
    if STOP is not None and STOP.is_set():
        return "paused"
    _, _, torch = CACHE["modules"]
    net, opts, _, _ = restore_source(job, project)
    path = Path(out) / "diagnostics" / job["group"] / f"seed_{job['seed']}.json"
    if not path.exists():
        result = diagnostics(net, opts, dataset(project), read(HERE / "protocol.json"), torch)
        write(path, dict(group=job["group"], seed=job["seed"], step=20000, **result))
    print(f"DIAG {job['group']} seed={job['seed']}", flush=True)
    return "completed"


def continue_job(job, project, out, max_extra=None):
    if STOP is not None and STOP.is_set():
        return "paused"
    m, e, torch = CACHE["modules"]
    p = read(HERE / "protocol.json")
    path = Path(out) / "continuation" / job["group"] / f"seed_{job['seed']}"
    path.mkdir(parents=True, exist_ok=True)
    with locked(path / ".lock"):
        if (path / "result.json").exists():
            return read(path / "result.json")["status"]
        net, opts, scheds, source_saved = restore_source(job, project)
        fixed_lrs = [o.param_groups[0]["lr"] for o in opts]
        step, train_s, wall_s, records, diag_rows = 20000, 0., 0., [], []
        if (path / "latest.pt").exists():
            saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
            if saved["source_job"] != job:
                raise RuntimeError("Continuation source identity changed.")
            e.load_pack(saved, net, opts, scheds)
            step, train_s, wall_s = saved["step"], saved["extra_training_seconds"], saved["extra_wall_seconds"]
            records, diag_rows = saved["records"], saved["diagnostics"]
        data = dataset(project)
        x, y = data.load("train", True)
        xv, yv = data.load("val", True)
        started = time.perf_counter()
        base_wall_s = wall_s
        def wall():
            return base_wall_s + time.perf_counter() - started
        def observe(loss):
            value = e.relative_error(net, job["group"], xv, yv)
            records.append(dict(group=job["group"], seed=job["seed"], step=step,
                train_mse=loss, validation_relative_l2=value,
                extra_training_seconds=train_s, extra_wall_seconds=wall(),
                total_training_seconds=source_saved["training_seconds"] + train_s,
                total_active_wall_seconds=source_saved["wall_seconds"] + wall(),
                learning_rates=fixed_lrs))
            rows_csv(path / "validation.csv", records)
            write(path / "progress.json", dict(status="running", **records[-1]))
            print(f"EXTEND {job['group']} seed={job['seed']} {step}/40000 val={value:.8g}", flush=True)
        def checkpoint():
            e.atomic_torch(path / "latest.pt", dict(**e.pack(net, opts, scheds), source_job=job,
                step=step, extra_training_seconds=train_s, extra_wall_seconds=wall(),
                records=records, diagnostics=diag_rows))
        if not records:
            observe(None)
            diag_rows.append(dict(step=step, **diagnostics(net, opts, data, p, torch)))
            checkpoint()
        target = min(p["total_steps"], p["original_steps"] + max_extra) if max_extra is not None else p["total_steps"]
        try:
            while step < target:
                if STOP is not None and STOP.is_set():
                    checkpoint()
                    write(path / "progress.json", dict(status="paused", step=step))
                    return "paused"
                tick = time.perf_counter()
                for o in opts:
                    o.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(net(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite continuation loss")
                loss.backward()
                if not e.finite_parameters(net.parameters(), gradients=True):
                    raise FloatingPointError("Nonfinite continuation gradient")
                for o in opts:
                    o.step()
                if not e.finite_parameters(net.parameters()):
                    raise FloatingPointError("Nonfinite continuation parameters")
                train_s += time.perf_counter() - tick
                step += 1
                if [o.param_groups[0]["lr"] for o in opts] != fixed_lrs:
                    raise RuntimeError("Fixed continuation learning rate changed")
                if step % p["validation_every"] == 0 or step == target:
                    observe(float(loss.detach()))
                if step % p["diagnostic_every"] == 0 or step == target:
                    diag_rows.append(dict(step=step, **diagnostics(net, opts, data, p, torch)))
                if step % p["checkpoint_every"] == 0 or step == target:
                    checkpoint()
        except FloatingPointError as error:
            write(path / "result.json", dict(status="diverged", group=job["group"], seed=job["seed"],
                  step=step, reason=str(error), records=records))
            return "diverged"
        if step == p["total_steps"]:
            e.atomic_torch(path / "final.pt", net.state_dict())
            write(path / "result.json", dict(status="completed", group=job["group"], seed=job["seed"],
                step=step, records=records, diagnostics=diag_rows,
                extra_training_seconds=train_s, extra_wall_seconds=wall()))
            write(path / "progress.json", dict(status="completed", step=step))
            return "completed"
        return "smoke_completed"


def collect_existing(project, out):
    """Export original time curves and an explicitly exploratory common-rate contrast."""
    root = Path(project) / "results16"
    traces = []
    for f in sorted((root / "formal").rglob("result.json")):
        r = read(f)
        for row in r["records"]:
            traces.append(dict(group=r["identity"]["group"], seed=r["identity"]["seed"], **row))
    rows_csv(Path(out) / "original_time_curves.csv", traces)
    selected = read(root / "selection.json")["configs"]
    result = []
    for rate in [.0003, .001, .003, .01, .03]:
        for i, seed in enumerate([289, 290, 291]):
            values = {}
            for group in ["full_adam", "noflow_adam", "full_muon_adam", "noflow_muon_adam"]:
                table = selected[group]["table"]
                entry = next(t for t in table if t["rate"] == (rate if "muon" in group else .001))
                values[group] = entry["seed_medians"][i]
            contrast = (math.log(values["full_muon_adam"] / values["full_adam"])
                        - math.log(values["noflow_muon_adam"] / values["noflow_adam"]))
            result.append(dict(muon_lr=rate, adam_lr=.001, seed=seed, log_interaction=contrast,
                               source="tuning validation terminal medians; exploratory", **values))
    rows_csv(Path(out) / "common_rate_tuning_interaction.csv", result)


def export(out):
    out = Path(out)
    rows = []
    for f in sorted((out / "continuation").rglob("result.json")):
        r = read(f)
        row = dict(group=r["group"], seed=r["seed"], status=r["status"], step=r["step"])
        if r["status"] == "completed":
            a, b = r["records"][0], r["records"][-1]
            row.update(val_20000=a["validation_relative_l2"], val_40000=b["validation_relative_l2"],
                       relative_validation_change=b["validation_relative_l2"] / a["validation_relative_l2"] - 1)
            tail = r["records"][-5:]
            row["last_2000_step_validation_change"] = (tail[-1]["validation_relative_l2"] /
                                                       tail[0]["validation_relative_l2"] - 1)
        rows.append(row)
    rows_csv(out / "continuation_summary.csv", rows)
    with ZipFile(out / "补充实验_日志与诊断.zip", "w", ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.suffix in [".json", ".csv", ".jsonl", ".png", ".pdf", ".md", ".log", ".py"]:
                z.write(f, f.relative_to(out))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["diagnose", "continue", "smoke", "status", "export"])
    parser.add_argument("--project", type=Path, default=HERE.parent)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    project = args.project.resolve()
    out = (args.out or project / "results_supplement_20261118").resolve()
    if out == project or out.is_relative_to(project / "results16") or out.is_relative_to(project / "heat_fair_v2"):
        raise RuntimeError("Output must be separate from original code/data/results.")
    if args.command == "status":
        ds = list((out / "diagnostics").rglob("seed_*.json"))
        rs = [read(f) for f in (out / "continuation").rglob("result.json")]
        print(json.dumps(dict(diagnostics=len(ds), continued_completed=sum(r["status"] == "completed" for r in rs),
                              continued_diverged=sum(r["status"] == "diverged" for r in rs), expected=40), indent=2))
        return
    if args.command == "smoke":
        out = out / "smoke_only"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))
    with locked(out / ".controller.lock"):
        if args.command == "export":
            export(out)
            return
        if args.workers < 1 or args.threads < 1 or args.workers * args.threads > 16:
            raise ValueError("Use at most 16 CPU threads.")
        frozen = freeze(project, out)
        collect_existing(project, out)
        jobs = frozen["jobs"]
        if args.command == "smoke":
            jobs = [jobs[10]]  # Full/Muon-Adam seed 292, chosen without looking at outcomes.
        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1]:
            signal.signal(sig, lambda *_: stop.set())
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                 initializer=init_worker, initargs=(project, args.threads, stop)) as pool:
            task = diagnose_job if args.command == "diagnose" else continue_job
            futures = [pool.submit(task, j, project, out, *([3] if args.command == "smoke" else [])) for j in jobs]
            for f in as_completed(futures):
                print("JOB", f.result(), flush=True)
        export(out)
        if stop.is_set():
            raise SystemExit(130)


if __name__ == "__main__":
    main()
