"""Explicit commands only; heat validation experiment, version 2.

init -> benchmark -> pilot -> pilot-report -> diagnose -> lock-t
     -> configure-search -> sweep -> select -> final -> evaluate
"""
import argparse
import copy
import csv
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import time

import common as c
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PURE = ["deeponet_adam", "mionet_adam", "full_adam", "noflow_adam"]
PILOTS = ["deeponet_adam", "mionet_adam", "full_adam"]


def write_csv(path, rows):
    c.write_csv(path, rows)


def settings(args):
    return c.read_json(args.out / "settings.json")


def validate_settings(s):
    if not s["groups"] or len(set(s["groups"])) != len(s["groups"]) or not set(s["groups"]) <= set(c.GROUPS):
        raise ValueError("Use distinct, supported model groups.")
    if not set(PURE) <= set(s["groups"]):
        raise ValueError("The four Adam reference groups must be retained for staged LR selection.")
    tuning, final = s["tuning_seeds"], s["final_seeds"]
    if len(tuning) != 3 or len(final) != 10 or len(set(tuning + final)) != 13:
        raise ValueError("Use three tuning seeds and ten distinct new final seeds.")
    if s["validation_every"] <= 0 or s["target_validation_error"] <= 0:
        raise ValueError("Validation interval and target error must be positive.")
    if s["muon_weight_decay"] != 0 or s["adam_weight_decay"] != 0:
        raise ValueError("This comparison requires both weight decays to equal zero.")
    if s["muon_adjust_lr_fn"] != "match_rms_adamw":
        raise ValueError("This protocol uses match_rms_adamw; do not add manual shape scaling.")
    p = s["pilot"]
    if not 0 < p["window_fraction"] < 1 or not 0 < p["minimum_relative_improvement"] < 1:
        raise ValueError("Invalid plateau fractions.")
    for field in ["reference_steps", "qofn_max_steps", "minimum_steps"]:
        if p[field] < s["validation_every"] or p[field] % s["validation_every"]:
            raise ValueError(f"{field} must be a positive multiple of validation_every.")
    if p["minimum_steps"] > min(p["reference_steps"], p["qofn_max_steps"]):
        raise ValueError("Minimum steps exceed a pilot budget.")
    if p["smoothing_checkpoints"] < 1 or p["patience_checkpoints"] < 1 or p["T_multiplier"] < 1:
        raise ValueError("Invalid smoothing, patience or T multiplier.")


def verify_data_manifest(source):
    manifest = c.read_json(source / "manifest.json")
    if c.object_hash({k: v for k, v in manifest.items() if k != "dataset_id"}) != manifest["dataset_id"]:
        raise RuntimeError("Source data manifest changed.")
    if manifest["counts"] != dict(train=1800, val=200, test=200):
        raise RuntimeError("Expected the prepared 1800/200/200 split.")
    for name, sha in manifest["files"].items():
        if c.digest(source / name) != sha:
            raise RuntimeError(f"Prepared data changed: {source / name}")
    return manifest


def initialize(args):
    proposed = c.read_json(args.settings)
    validate_settings(proposed)
    target = args.out / "settings.json"
    if target.exists() and c.read_json(target) != proposed:
        raise RuntimeError("Settings already frozen; use a new output directory to change them.")
    source = args.data_source.resolve()
    verify_data_manifest(source)
    data = args.out / "data"
    if data.exists():
        if verify_data_manifest(data)["dataset_id"] != c.read_json(source / "manifest.json")["dataset_id"]:
            raise RuntimeError("Output contains a different dataset.")
    else:
        pending = args.out / "data.pending"
        if pending.exists():
            raise RuntimeError("An incomplete data.pending exists; inspect it before retrying init.")
        shutil.copytree(source, pending)
        verify_data_manifest(pending)
        pending.rename(data)
    c.write_json(target, proposed)
    print(f"Data verified and copied; no PDE regeneration needed: {data}")
    show_plan(args)


def show_plan(args):
    s = settings(args)
    print("Groups:", ", ".join(s["groups"]))
    print("Tuning seeds:", s["tuning_seeds"], "| independent final seeds:", s["final_seeds"])
    print("Pilots: DeepONet/Adam and MIONet/Adam,", s["pilot"]["reference_steps"], "steps each;")
    print("Full QOFN/Adam, at most", s["pilot"]["qofn_max_steps"], "steps, with plateau stopping.")
    print("Plateau settings:", json.dumps(s["pilot"]))
    print("Target is a validation diagnostic:", s["target_validation_error"])
    print("Search requires an explicit configure-search command. Five candidates per group x three seeds.")
    print("Sweep runs:", len(s["groups"]) * 5 * 3, "| final runs:", len(s["groups"]) * 10)
    print("No command automatically starts the next experiment stage.")


def estimate_cost(args):
    s = settings(args)
    benchmark = c.read_json(args.out / "benchmark.json")
    if benchmark["code_sha"] != c.code_hash() or benchmark["hardware"]["device"] != str(args.device) or benchmark["hardware"]["threads"] != args.threads:
        raise RuntimeError("Benchmark code/device/thread count differs; rerun the short benchmark.")
    steps = args.steps if args.steps is not None else frozen_T(args)
    if steps <= 0:
        raise ValueError("steps must be positive")
    rows = []
    for group in s["groups"]:
        seconds = next(r["median_step_seconds"] for r in benchmark["timings"]
                       if r["group"] == group and r["implementation"] == "matrix-composition")
        rows.append(dict(group=group, steps=steps, seconds_per_step=seconds,
                         sweep_runs=15, final_runs=10,
                         sweep_training_hours=15 * steps * seconds / 3600,
                         final_training_hours=10 * steps * seconds / 3600))
    write_csv(args.out / "cost_estimate.csv", rows)
    print("Estimated pure training hours at T=", steps, ":",
          sum(r["sweep_training_hours"] + r["final_training_hours"] for r in rows))
    print("Short-benchmark projection; excludes validation, diagnostics, saving and sustained thermal throttling.")
    print("Saved cost_estimate.csv; no training was started.")


def protocol(args, data):
    return dict(version="heat-validation-v2", code_sha=c.code_hash(),
                settings_sha=c.digest(args.out / "settings.json"), dataset_id=data.manifest["dataset_id"],
                torch=str(torch.__version__), deepxde=str(c.dde.__version__), numpy=np.__version__,
                device=str(args.device), threads=args.threads, deterministic=True,
                batch="full:1800x8000", evaluation="mean per-function relative L2 at final fixed T",
                normalization="Full/No-Flow share min-max augmentation and norm clamp=1e-12")


def check_protocol(args, data):
    current = protocol(args, data)
    path = args.out / "protocol.json"
    if path.exists() and c.read_json(path) != current:
        raise RuntimeError("Code/settings/environment changed. Use a new output directory.")
    if not path.exists():
        c.write_json(path, current)


def make_optimizers(net, group, adam_lr, muon_lr=None):
    named = dict(net.named_parameters())
    if group.endswith("_muon_adam"):
        if muon_lr is None or not math.isfinite(muon_lr) or muon_lr <= 0:
            raise ValueError("A hybrid requires a separate positive muon_lr.")
        if c.is_qofn(group):
            chosen = [getattr(net, name).output_layer.weight for name in ("branch1", "branch2", "trunk")]
        elif group == "deeponet_output_muon_adam":
            chosen = [net.branch.linears[-1].weight, net.trunk.linears[-1].weight]
        else:
            chosen = [p for p in named.values() if p.ndim == 2]
        ids = {id(p) for p in chosen}
        other = [p for p in named.values() if id(p) not in ids]
        opts = [torch.optim.Muon(chosen, lr=muon_lr, weight_decay=0, adjust_lr_fn="match_rms_adamw"),
                torch.optim.Adam(other, lr=adam_lr, weight_decay=0)]
    else:
        if muon_lr is not None:
            raise ValueError("Pure Adam groups must not receive a Muon LR.")
        ids = set()
        opts = [torch.optim.Adam(net.parameters(), lr=adam_lr, weight_decay=0)]
    return opts, dict(adam_lr=adam_lr, muon_lr=muon_lr,
                      muon=[n for n, p in named.items() if id(p) in ids],
                      adam=[n for n, p in named.items() if id(p) not in ids],
                      defaults=[o.defaults for o in opts])


def smoothed_values(records, count):
    values = [r["validation_relative_l2"] for r in records]
    return [float(np.median(values[max(0, i - count + 1):i + 1])) for i in range(len(values))]


def plateau(records, s):
    """Latest sustained plateau, using only observations available at each step.

    Compare median(last 3 observations) at step s to the same statistic at
    the most recent checkpoint <= 0.8*s. Stop after three qualifying checks.
    Later improvement resets a plateau in a completed fixed-budget curve.
    """
    p = s["pilot"]
    smooth = smoothed_values(records, p["smoothing_checkpoints"])
    streak = []
    improvement = None
    for i, row in enumerate(records):
        step = row["step"]
        if step < p["minimum_steps"]:
            continue
        baseline = (1 - p["window_fraction"]) * step
        earlier = [j for j in range(i) if records[j]["step"] <= baseline]
        if not earlier:
            continue
        j = earlier[-1]
        improvement = (smooth[j] - smooth[i]) / max(smooth[j], 1e-12)
        if improvement < p["minimum_relative_improvement"]:
            streak.append(step)
        else:
            streak = []
    found = len(streak) >= p["patience_checkpoints"]
    return dict(reached=found, step=streak[0] if found else None,
                latest_window_relative_improvement=improvement,
                consecutive_checks=len(streak))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def relative_error(net, group, x, y):
    net.eval()
    prediction = net(c.inputs_for(group, x))
    numerator = (prediction - y).double().square().sum(1).sqrt()
    denominator = y.double().square().sum(1).sqrt().clamp_min(1e-12)
    value = float((numerator / denominator).mean())
    if not math.isfinite(value):
        raise FloatingPointError("Relative L2 is not finite.")
    return value


@torch.no_grad()
def diagnostics(net, x):
    rows = []
    for name, values in zip(("branch1", "branch2", "trunk"), x):
        encoder = getattr(net, name)
        ids = torch.linspace(0, len(values) - 1, min(32, len(values)), device=values.device).long()
        h = values[ids]
        def stats(z):
            n = z.norm(dim=1)
            return dict(mean=float(n.mean()), rms=float(n.square().mean().sqrt()), maximum=float(n.max()))
        for i, layer in enumerate(encoder.hidden_layers):
            pre = encoder.hidden_preactivation(h, i)
            post = encoder.activation(pre)
            rows.append(dict(encoder=name, layer=i,
                             alpha=float(torch.sigmoid(encoder.h_params[i])) if encoder.flow else None,
                             gate_active=bool(encoder.flow and layer.in_features == layer.out_features),
                             input_norm=stats(h), preactivation_norm=stats(pre), feature_norm=stats(post)))
            h = post
        rows.append(dict(encoder=name, layer="output", feature_norm=stats(encoder.output_layer(h))))
    return rows


def run_dir(root, stage, group, adam_lr, muon_lr, seed):
    candidate = f"adam_{adam_lr:.12g}_muon_{muon_lr:.12g}" if muon_lr is not None else f"adam_{adam_lr:.12g}"
    return root / stage / group / candidate / f"seed_{seed}"


def read_lines(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def first_target(records, target):
    return next((dict(step=r["step"], training_seconds=r["training_seconds"], wall_seconds=r["wall_seconds"])
                 for r in records if r["validation_relative_l2"] <= target), None)


def train_one(args, data, stage, group, adam_lr, muon_lr, seed, steps, early_stop=False):
    s = settings(args)
    path = run_dir(args.out, stage, group, adam_lr, muon_lr, seed)
    path.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol=protocol(args, data), stage=stage, group=group, adam_lr=adam_lr,
                    muon_lr=muon_lr, seed=seed, maximum_steps=steps, early_stop=early_stop)
    identity_path = path / "identity.json"
    if identity_path.exists() and c.read_json(identity_path) != identity:
        raise RuntimeError(f"Cached run has different settings: {path}")
    c.write_json(identity_path, identity)
    if (path / "result.json").exists():
        result = c.read_json(path / "result.json")
        if result["status"] == "complete" and c.digest(path / "final.pt") != result["checkpoint_sha"]:
            raise RuntimeError(f"Checkpoint changed: {path}")
        print(f"Reusing {stage} | {group} | seed={seed}", flush=True)
        return result
    session_start = time.perf_counter()
    c.seed_all(seed)
    net = c.build_model(group).to(args.device)
    opts, routing = make_optimizers(net, group, adam_lr, muon_lr)
    x, y = data.load("train", c.is_qofn(group))
    xv, yv = data.load("val", c.is_qofn(group))
    start, training_seconds, wall_offset = 0, 0.0, 0.0
    records = []
    resumed_from_checkpoint = (path / "latest.pt").exists()
    if resumed_from_checkpoint:
        saved = torch.load(path / "latest.pt", map_location=args.device, weights_only=True)
        net.load_state_dict(saved["model"])
        for opt, state in zip(opts, saved["optimizers"]):
            opt.load_state_dict(state)
        start, training_seconds = saved["step"], saved["training_seconds"]
        records = saved["records"]
        wall_offset = saved["wall_seconds"]
        if (path / "latest_timing.json").exists():
            timing = c.read_json(path / "latest_timing.json")
            if timing["step"] == start:
                wall_offset = timing["wall_seconds"]
        for filename in ["validation.jsonl", "layers.jsonl"]:
            rows = [r for r in read_lines(path / filename) if r["step"] <= start]
            (path / filename).write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    else:
        for filename in ["validation.jsonl", "layers.jsonl"]:
            (path / filename).write_text("")
    c.write_json(path / "hardware.json", c.environment(args.device))
    c.write_json(path / "optimizer.json", routing)
    def wall():
        sync(args.device)
        return wall_offset + time.perf_counter() - session_start
    def observe(step, loss):
        value = relative_error(net, group, xv, yv)
        if c.is_qofn(group):
            with (path / "layers.jsonl").open("a") as f:
                for row in diagnostics(net, x):
                    f.write(json.dumps(dict(step=step, **row), allow_nan=False) + "\n")
        row = dict(step=step, train_mse=loss, validation_relative_l2=value,
                   training_seconds=training_seconds, wall_seconds=wall(),
                   mean_training_seconds_per_step=training_seconds / step if step else None)
        records.append(row)
        with (path / "validation.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"  step={step} val={value:.7g} train_s={training_seconds:.1f} wall_s={row['wall_seconds']:.1f}", flush=True)
    def checkpoint(step):
        saved = dict(model=net.state_dict(), optimizers=[o.state_dict() for o in opts],
                     step=step, training_seconds=training_seconds, wall_seconds=wall(), records=records)
        torch.save(saved, path / "latest.tmp")
        (path / "latest.tmp").replace(path / "latest.pt")
        c.write_json(path / "latest_timing.json", dict(step=step, wall_seconds=wall()))
    print(f"{stage} | {group} | Adam LR={adam_lr:g} | Muon LR={muon_lr} | seed={seed} | {start}/{steps}", flush=True)
    stop_reason, current_step = "budget", start
    try:
        if not resumed_from_checkpoint:
            observe(0, None)
            checkpoint(0)
        already_stopped = early_stop and plateau(records, s)["reached"]
        if already_stopped:
            stop_reason = "plateau"
        else:
            for step in range(start + 1, steps + 1):
                sync(args.device)
                tick = time.perf_counter()
                net.train()
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(net(c.inputs_for(group, x)), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}.")
                loss.backward()
                for opt in opts:
                    opt.step()
                sync(args.device)
                training_seconds += time.perf_counter() - tick
                current_step = step
                if step % s["validation_every"] == 0 or step == steps:
                    observe(step, float(loss.detach()))
                    checkpoint(step)
                    if early_stop and plateau(records, s)["reached"]:
                        stop_reason = "plateau"
                        break
        torch.save(net.state_dict(), path / "final.tmp")
        (path / "final.tmp").replace(path / "final.pt")
        reached = first_target(records, s["target_validation_error"])
        result = dict(**identity, status="complete", actual_steps=current_step, stop_reason=stop_reason,
                      final_validation_relative_l2=records[-1]["validation_relative_l2"],
                      training_seconds=training_seconds, wall_seconds=wall(),
                      target_validation_error=s["target_validation_error"], first_target=reached,
                      target_not_observed=reached is None,
                      parameters=sum(p.numel() for p in net.parameters()),
                      checkpoint_sha=c.digest(path / "final.pt"))
    except FloatingPointError as error:
        if stage != "sweep":
            raise
        result = dict(**identity, status="diverged", actual_steps=current_step,
                      final_validation_relative_l2=None, reason=str(error),
                      training_seconds=training_seconds, wall_seconds=wall())
        print("DIVERGED:", error, flush=True)
    c.write_json(path / "result.json", result)
    return result


def pilot_paths(args):
    s = settings(args)
    p = s["pilot"]
    return {g: run_dir(args.out, "pilot", g, p["adam_lr"], None, p["seed"]) for g in PILOTS}


def run_pilots(args, data):
    p = settings(args)["pilot"]
    for group in PILOTS:
        train_one(args, data, "pilot", group, p["adam_lr"], None, p["seed"],
                  p["qofn_max_steps"] if c.is_qofn(group) else p["reference_steps"],
                  early_stop=c.is_qofn(group))
    print("Three pilots completed. Run pilot-report; no sweep has started.")


def pilot_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = settings(args)
    rows, hashes = [], {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, path in pilot_paths(args).items():
        r = c.read_json(path / "result.json")
        if r["status"] != "complete":
            raise RuntimeError(f"Incomplete pilot: {group}")
        records = read_lines(path / "validation.jsonl")
        status = plateau(records, s)
        rows.append(dict(group=group, actual_steps=r["actual_steps"],
                         final_validation=r["final_validation_relative_l2"],
                         plateau_step=status["step"], plateau_observed=status["reached"],
                         latest_window_relative_improvement=status["latest_window_relative_improvement"],
                         target_observed=r["first_target"] is not None,
                         mean_training_seconds_per_step=r["training_seconds"] / max(r["actual_steps"], 1)))
        hashes[str((path / "result.json").relative_to(args.out))] = c.digest(path / "result.json")
        ax.semilogy([v["step"] for v in records], [v["validation_relative_l2"] for v in records], label=group)
    ax.axhline(s["target_validation_error"], color="gray", linestyle="--", label="validation target")
    ax.set(xlabel="Optimizer steps", ylabel="Validation mean relative L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "pilot_validation.png", dpi=180)
    plt.close(fig)
    all_plateau = all(r["plateau_observed"] for r in rows)
    suggestion = (math.ceil(max(r["plateau_step"] for r in rows) * s["pilot"]["T_multiplier"] / s["validation_every"])
                  * s["validation_every"]) if all_plateau else None
    deeponet = next(r for r in rows if r["group"] == "deeponet_adam")
    report = dict(rows=rows, all_pilots_have_observed_plateau=all_plateau,
                  suggested_T=suggestion, deeponet_target_observed=deeponet["target_observed"],
                  pilot_result_hashes=hashes, protocol_sha=c.digest(args.out / "protocol.json"))
    c.write_json(args.out / "pilot_report.json", report)
    write_csv(args.out / "pilot_report.csv", rows)
    print(json.dumps(report, indent=2))
    if not all_plateau:
        print("No common plateau was established. Do not relabel a budget endpoint as convergence.")
    if not deeponet["target_observed"]:
        print("DeepONet did not reach the validation target. Run diagnose before choosing T.")


def diagnose(args, data):
    """Only training/validation inputs and labels; no test prediction or test fitting."""
    s = settings(args)
    rows = []
    for group, path in pilot_paths(args).items():
        r = c.read_json(path / "result.json")
        c.seed_all(s["pilot"]["seed"])
        net = c.build_model(group).to(args.device)
        net.load_state_dict(torch.load(path / "final.pt", map_location=args.device, weights_only=True))
        row = dict(group=group, parameters=sum(p.numel() for p in net.parameters()),
                   actual_steps=r["actual_steps"])
        for split in ["train", "val"]:
            x, y = data.load(split, c.is_qofn(group))
            row[f"{split}_relative_l2"] = relative_error(net, group, x, y)
            with np.load(args.out / "data" / f"{split}.npz") as f:
                boundary = np.any(np.isclose(f["xt"][:, :2], 0) | np.isclose(f["xt"][:, :2], 1), axis=1)
                row[f"{split}_maximum_boundary_label"] = float(np.abs(f["y"][:, boundary]).max())
        rows.append(row)
    report = dict(pilot_report_sha=c.digest(args.out / "pilot_report.json"), rows=rows,
                  interpretation="Train/validation errors and boundary checks are diagnostics, not proof of a cause.")
    # PCA reconstruction is measured on the original training pool only.
    manifest = data.manifest
    raw_source = Path(manifest["original_sources"]["train"]["path"])
    if raw_source.is_file() and c.digest(raw_source) == manifest["original_sources"]["train"]["sha256"]:
        with np.load(raw_source, allow_pickle=True) as f, np.load(args.out / "data/preprocessing.npz") as p:
            u = np.asarray(f["X_train"][1], dtype=np.float32)
            z = (u - p["pca_mean"]) @ p["pca_components"].T
            reconstructed = z @ p["pca_components"] + p["pca_mean"]
            error = np.linalg.norm(reconstructed - u, axis=1) / np.maximum(np.linalg.norm(u, axis=1), 1e-12)
            report["pca_mean_input_reconstruction_error"] = {
                split: float(error[manifest[f"{split}_ids"]].mean()) for split in ["train", "val"]}
            report["pca_note"] = "Input reconstruction error is not a lower bound on solution-operator prediction error."
    else:
        report["pca_note"] = "Original training pool unavailable or changed; PCA reconstruction diagnostic skipped."
    c.write_json(args.out / "diagnostics.json", report)
    print(json.dumps(report, indent=2))


def lock_t(args):
    if (args.out / "sweep").exists():
        raise RuntimeError("The sweep has started; T cannot be changed.")
    s = settings(args)
    report = c.read_json(args.out / "pilot_report.json")
    if report["protocol_sha"] != c.digest(args.out / "protocol.json"):
        raise RuntimeError("Pilot protocol changed.")
    for relative, sha in report["pilot_result_hashes"].items():
        if c.digest(args.out / relative) != sha:
            raise RuntimeError("Pilot results changed since the report.")
    if not report["all_pilots_have_observed_plateau"] and not args.allow_fixed_budget:
        raise RuntimeError("No common plateau. Review the curves; only --allow-fixed-budget can record an explicit budget rather than convergence.")
    if not report["deeponet_target_observed"]:
        if not (args.out / "diagnostics.json").exists():
            raise RuntimeError("DeepONet target was not reached. Run diagnose and review its report before lock-t.")
        diagnostic = c.read_json(args.out / "diagnostics.json")
        if diagnostic["pilot_report_sha"] != c.digest(args.out / "pilot_report.json"):
            raise RuntimeError("Diagnostics do not correspond to the current pilots.")
    steps = args.steps if args.steps is not None else report["suggested_T"]
    if steps is None or steps < s["validation_every"] or steps % s["validation_every"]:
        raise ValueError("Provide a positive T that is a multiple of the validation interval.")
    record = dict(steps=steps, reason=args.reason, fixed_budget_override=args.allow_fixed_budget,
                  suggested_T=report["suggested_T"], pilot_report_sha=c.digest(args.out / "pilot_report.json"),
                  diagnostics_sha=c.digest(args.out / "diagnostics.json") if (args.out / "diagnostics.json").exists() else None,
                  protocol_sha=c.digest(args.out / "protocol.json"))
    target = args.out / "T.json"
    if target.exists() and c.read_json(target) != record:
        raise RuntimeError("T was already frozen; use a new output directory to change it.")
    c.write_json(target, record)
    print("Locked common budget T=", steps)


def frozen_T(args):
    r = c.read_json(args.out / "T.json")
    if r["protocol_sha"] != c.digest(args.out / "protocol.json"):
        raise RuntimeError("Frozen T protocol changed.")
    return r["steps"]


def configure_search(args):
    frozen_T(args)
    if (args.out / "sweep").exists():
        raise RuntimeError("Cannot change the search after it has started.")
    for rates in [args.adam_lrs, args.muon_lrs]:
        if len(rates) != 5 or len(set(rates)) != 5 or any(not math.isfinite(v) or v <= 0 for v in rates):
            raise ValueError("Provide exactly five distinct positive learning rates for each optimizer.")
    value = dict(adam_lrs=sorted(args.adam_lrs), muon_lrs=sorted(args.muon_lrs),
                 auxiliary_adam_policy="freeze each architecture's selected pure-Adam LR, then tune Muon LR",
                 rule="minimum three-seed mean validation relative L2 at final T; ties choose the smaller LR",
                 T_sha=c.digest(args.out / "T.json"))
    path = args.out / "search.json"
    if path.exists() and c.read_json(path) != value:
        raise RuntimeError("Search already frozen; do not silently replace its grids.")
    c.write_json(path, value)
    print(json.dumps(value, indent=2))


def search_settings(args):
    frozen_T(args)
    search = c.read_json(args.out / "search.json")
    if search["T_sha"] != c.digest(args.out / "T.json"):
        raise RuntimeError("T changed after search configuration.")
    return search


def baseline_for(group):
    return group.split("_")[0] + "_adam"


def mean_error(results):
    return float(np.mean([r["final_validation_relative_l2"] for r in results])) if all(r["status"] == "complete" for r in results) else math.inf


def candidates(args, group, rates, fixed_adam=None):
    s = settings(args)
    table, result_hashes = [], {}
    for rate in rates:
        adam, muon = (rate, None) if fixed_adam is None else (fixed_adam, rate)
        observations = []
        for seed in s["tuning_seeds"]:
            path = run_dir(args.out, "sweep", group, adam, muon, seed) / "result.json"
            r = c.read_json(path)
            if r["actual_steps"] != frozen_T(args) and r["status"] == "complete":
                raise RuntimeError(f"Sweep did not reach T: {path}")
            if r["protocol"] != c.read_json(args.out / "protocol.json"):
                raise RuntimeError("Sweep protocol changed.")
            observations.append(r)
            result_hashes[str(path.relative_to(args.out))] = c.digest(path)
        value = mean_error(observations)
        finite = [r["final_validation_relative_l2"] for r in observations if r["status"] == "complete"]
        table.append(dict(group=group, adam_lr=adam, muon_lr=muon,
                          mean_validation=value if math.isfinite(value) else None,
                          sample_sd=float(np.std(finite, ddof=1)) if len(finite) == 3 else None,
                          n_complete=len(finite)))
    valid = [r for r in table if r["mean_validation"] is not None]
    if not valid:
        raise RuntimeError(f"All five candidates diverged for {group}; revise the protocol in a new output directory.")
    best = min(valid, key=lambda r: (r["mean_validation"], r["muon_lr"] if r["muon_lr"] is not None else r["adam_lr"]))
    best = dict(best, optimum_at_grid_boundary=(best["muon_lr"] if fixed_adam is not None else best["adam_lr"]) in [min(rates), max(rates)])
    return table, best, result_hashes


def sweep(args, data):
    if (args.out / "selection.json").exists():
        raise RuntimeError("Search was already selected and frozen.")
    s, search = settings(args), search_settings(args)
    steps = frozen_T(args)
    adam_best = {}
    for group in PURE:
        for lr in search["adam_lrs"]:
            for seed in s["tuning_seeds"]:
                train_one(args, data, "sweep", group, lr, None, seed, steps)
        _, best, _ = candidates(args, group, search["adam_lrs"])
        adam_best[group] = best["adam_lr"]
    for group in s["groups"]:
        if group in PURE:
            continue
        auxiliary_lr = adam_best[baseline_for(group)]
        for muon_lr in search["muon_lrs"]:
            for seed in s["tuning_seeds"]:
                train_one(args, data, "sweep", group, auxiliary_lr, muon_lr, seed, steps)
    print("Search finished. Run select; no final-seed training has started.")


def select(args):
    s, search = settings(args), search_settings(args)
    rows, chosen, hashes = [], {}, {}
    for group in PURE:
        table, best, hs = candidates(args, group, search["adam_lrs"])
        rows += table
        chosen[group] = best
        hashes.update(hs)
    for group in s["groups"]:
        if group in PURE:
            continue
        table, best, hs = candidates(args, group, search["muon_lrs"], chosen[baseline_for(group)]["adam_lr"])
        rows += table
        chosen[group] = best
        hashes.update(hs)
    record = dict(configs=chosen, steps=frozen_T(args), tuning_seeds=s["tuning_seeds"],
                  final_seeds=s["final_seeds"], result_hashes=hashes,
                  search_sha=c.digest(args.out / "search.json"), T_sha=c.digest(args.out / "T.json"))
    path = args.out / "selection.json"
    if path.exists() and c.read_json(path) != record:
        raise RuntimeError("Selection changed after freezing.")
    c.write_json(path, record)
    write_csv(args.out / "validation_lr.csv", rows)
    plot_rates(args, rows)
    print(json.dumps(chosen, indent=2))
    if any(r["optimum_at_grid_boundary"] for r in chosen.values()):
        print("Some optima lie on a grid boundary. This is recorded; no extra trials are started automatically.")


def plot_rates(args, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = settings(args)["groups"]
    fig, axes = plt.subplots(math.ceil(len(groups) / 3), 3, figsize=(12, 9), squeeze=False, constrained_layout=True)
    for ax, group in zip(axes.flat, groups):
        rs = [r for r in rows if r["group"] == group and r["mean_validation"] is not None]
        hybrid = group not in PURE
        ax.errorbar([r["muon_lr"] if hybrid else r["adam_lr"] for r in rs],
                    [r["mean_validation"] for r in rs], yerr=[r["sample_sd"] for r in rs], marker="o", capsize=3)
        ax.set(xscale="log", yscale="log", title=group,
               xlabel="Muon LR (auxiliary Adam frozen)" if hybrid else "Adam LR", ylabel="Validation relative L2")
        ax.title.set_fontsize(9)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.savefig(args.out / "validation_lr.png", dpi=180)
    fig.savefig(args.out / "validation_lr.pdf")
    plt.close(fig)


def selection(args):
    r = c.read_json(args.out / "selection.json")
    if r["steps"] != frozen_T(args) or r["T_sha"] != c.digest(args.out / "T.json") or r["search_sha"] != c.digest(args.out / "search.json"):
        raise RuntimeError("Selection settings changed.")
    for relative, sha in r["result_hashes"].items():
        if c.digest(args.out / relative) != sha:
            raise RuntimeError("Validation results changed after selection.")
    return r


def final_runs(args, data):
    chosen = selection(args)
    for group in settings(args)["groups"]:
        lr = chosen["configs"][group]
        for seed in chosen["final_seeds"]:
            train_one(args, data, "final", group, lr["adam_lr"], lr["muon_lr"], seed, chosen["steps"])
    print("Ten independent seeds per group completed. Run evaluate to open the test set.")


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(2026)
    ids = rng.integers(0, len(values), (20000, len(values)))
    low, high = np.quantile(values[ids].mean(1), [.025, .975])
    return dict(n=len(values), mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                bootstrap_p025=float(low), bootstrap_p975=float(high))


def summarize(args, rows):
    from scipy.stats import ttest_1samp, wilcoxon
    s = settings(args)
    arrays = {g: np.array([next(r["test_relative_l2"] for r in rows if r["group"] == g and r["seed"] == seed)
                           for seed in s["final_seeds"]]) for g in s["groups"]}
    write_csv(args.out / "test_runs.csv", rows)
    summaries = []
    for g, values in arrays.items():
        rs = [r for r in rows if r["group"] == g]
        summaries.append(dict(group=g, **interval(values), parameters=rs[0]["parameters"],
                              mean_training_seconds=float(np.mean([r["training_seconds"] for r in rs])),
                              mean_wall_seconds=float(np.mean([r["wall_seconds"] for r in rs])),
                              target_observed_count=sum(r["first_target_step"] is not None for r in rs)))
    write_csv(args.out / "main_table.csv", summaries)
    effects = {}
    for prefix in ["deeponet", "mionet", "full", "noflow"]:
        a, m = prefix + "_adam", prefix + "_muon_adam"
        if a in arrays and m in arrays:
            effects[f"{prefix}: Adam minus Muon-Adam"] = arrays[a] - arrays[m]
    if "deeponet_output_muon_adam" in arrays:
        effects["DeepONet: Adam minus output-only Muon-Adam"] = arrays["deeponet_adam"] - arrays["deeponet_output_muon_adam"]
    if "full: Adam minus Muon-Adam" in effects and "noflow: Adam minus Muon-Adam" in effects:
        effects["I: tuned delta_full minus tuned delta_noflow"] = effects["full: Adam minus Muon-Adam"] - effects["noflow: Adam minus Muon-Adam"]
    table = []
    for name, values in effects.items():
        t_p = float(ttest_1samp(values, 0).pvalue) if np.std(values) > 0 else (1.0 if np.all(values == 0) else None)
        w_p = float(wilcoxon(values).pvalue) if np.any(values) else 1.0
        table.append(dict(contrast=name, **interval(values), paired_t_p=t_p, wilcoxon_p=w_p))
    write_csv(args.out / "paired_effects.csv", table)
    write_csv(args.out / "seed_effects.csv", [dict(seed=seed, **{k: float(v[i]) for k, v in effects.items()})
                                            for i, seed in enumerate(s["final_seeds"])])


def evaluate(args, data):
    s, chosen = settings(args), selection(args)
    runs = []
    for group in s["groups"]:
        lr = chosen["configs"][group]
        for seed in chosen["final_seeds"]:
            path = run_dir(args.out, "final", group, lr["adam_lr"], lr["muon_lr"], seed)
            r = c.read_json(path / "result.json")
            if r["status"] != "complete" or r["actual_steps"] != chosen["steps"] or r["protocol"] != protocol(args, data):
                raise RuntimeError(f"Incomplete or incompatible final model: {path}")
            if c.digest(path / "final.pt") != r["checkpoint_sha"]:
                raise RuntimeError(f"Checkpoint changed: {path}")
            runs.append((group, seed, lr, path, r))
    evaluation_id = c.object_hash(dict(selection=c.digest(args.out / "selection.json"),
                                       checkpoints=[r[4]["checkpoint_sha"] for r in runs]))
    frozen = dict(evaluation_id=evaluation_id, test_sha=data.manifest["files"]["test.npz"])
    freeze = args.out / "test_evaluation.json"
    if freeze.exists() and c.read_json(freeze) != frozen:
        raise RuntimeError("The test set was already opened for different final checkpoints.")
    c.write_json(freeze, frozen)
    scores = args.out / "test_scores"
    scores.mkdir(exist_ok=True)
    rows = []
    for group, seed, lr, path, r in runs:
        target = scores / f"{group}_seed_{seed}.json"
        if target.exists():
            row = c.read_json(target)
            if row["evaluation_id"] != evaluation_id:
                raise RuntimeError("Test cache mismatch.")
        else:
            c.seed_all(seed)
            net = c.build_model(group).to(args.device)
            net.load_state_dict(torch.load(path / "final.pt", map_location=args.device, weights_only=True))
            x, y = data.load("test", c.is_qofn(group))
            value = relative_error(net, group, x, y)
            reached = r["first_target"]
            row = dict(group=group, seed=seed, adam_lr=lr["adam_lr"], muon_lr=lr["muon_lr"],
                       steps=r["actual_steps"], test_relative_l2=value, parameters=r["parameters"],
                       training_seconds=r["training_seconds"], wall_seconds=r["wall_seconds"],
                       first_target_step=reached["step"] if reached else None,
                       first_target_training_seconds=reached["training_seconds"] if reached else None,
                       first_target_wall_seconds=reached["wall_seconds"] if reached else None,
                       target_not_observed=r["target_not_observed"], evaluation_id=evaluation_id)
            c.write_json(target, row)
            print(f"Test | {group} | seed={seed} | error={value:.7g}", flush=True)
        rows.append(row)
    summarize(args, rows)
    print("Final tables saved. Completed test scores are cached and not recomputed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--settings", type=Path, default=HERE / "settings.json")
    init.add_argument("--data-source", type=Path, default=HERE.parent / "heat_fair_v1/results/data")
    for name in ["plan", "check", "pilot", "pilot-report", "diagnose", "sweep", "select", "final", "evaluate"]:
        sub.add_parser(name)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--repeats", type=int, default=5)
    estimate = sub.add_parser("estimate")
    estimate.add_argument("--steps", type=int)
    lock = sub.add_parser("lock-t")
    lock.add_argument("--steps", type=int)
    lock.add_argument("--reason", required=True)
    lock.add_argument("--allow-fixed-budget", action="store_true")
    search = sub.add_parser("configure-search")
    search.add_argument("--adam-lrs", nargs=5, type=float, required=True)
    search.add_argument("--muon-lrs", nargs=5, type=float, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.device = torch.device(args.device)
    if args.device.type not in ["cpu", "cuda"]:
        parser.error("This protocol supports CPU or CUDA. Use --device cpu on this Mac.")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with c.exclusive(args.out):
        if args.command == "init":
            initialize(args)
            return
        s = settings(args)
        validate_settings(s)
        if args.command == "plan":
            show_plan(args)
            return
        data = c.Dataset(args.out, args.device)
        check_protocol(args, data)
        if args.command == "check":
            c.require_muon(args.device)
            print(json.dumps(c.environment(args.device), indent=2))
        elif args.command == "benchmark":
            from timing import benchmark
            benchmark(args, data)
        elif args.command == "estimate":
            estimate_cost(args)
        elif args.command == "pilot":
            run_pilots(args, data)
        elif args.command == "pilot-report":
            pilot_report(args)
        elif args.command == "diagnose":
            diagnose(args, data)
        elif args.command == "lock-t":
            lock_t(args)
        elif args.command == "configure-search":
            configure_search(args)
        elif args.command == "sweep":
            c.require_muon(args.device)
            sweep(args, data)
        elif args.command == "select":
            select(args)
        elif args.command == "final":
            final_runs(args, data)
        elif args.command == "evaluate":
            evaluate(args, data)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Stopped: {error}")
