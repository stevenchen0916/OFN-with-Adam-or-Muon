"""Frozen-protocol validation and data-free experiment control."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
FROZEN_ID = "24b6affe1d3341cb3df7343969b146bd004fbc523e29c950f08a66d534a920aa"
TERMINAL = {"completed", "diverged"}
PRIORITY = ["deeponet_adam", "mionet_adam", "deeponet_muon_adam",
            "deeponet_output_muon_adam", "mionet_muon_adam", "full_adam",
            "full_muon_adam", "noflow_adam", "noflow_muon_adam"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def csv_write(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        if fields:
            w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()})
    tmp.replace(path)


def protocol(path=HERE / "protocol.json"):
    p = read(path)
    body = dict(p)
    identifier = body.pop("protocol_id")
    if identifier != FROZEN_ID or object_sha(body) != identifier:
        raise RuntimeError("Frozen protocol/ID changed. This runner will not execute a revised protocol.")
    return p


def baseline(group):
    return group.split("_")[0] + "_adam"


def hybrid(group):
    return group.endswith("_muon_adam")


def run_path(root, stage, group, adam_lr, muon_lr, seed):
    key = object_sha(dict(adam_lr=adam_lr, muon_lr=muon_lr))[:12]
    label = f"adam_{adam_lr:.7g}_muon_{muon_lr:.7g}" if muon_lr is not None else f"adam_{adam_lr:.7g}"
    return Path(root) / stage / group / f"{label}_{key}" / f"seed_{seed}"


def final_order(p):
    result = []
    groups = p["groups"]
    for i, seed in enumerate(p["formal"]["seeds"]):
        shift = i % len(groups)
        result += [(seed, g) for g in groups[shift:] + groups[:shift]]
    return result


def terminal_median(records, steps=(19000, 19500, 20000)):
    values = {r["step"]: r["validation_relative_l2"] for r in records}
    if any(s not in values for s in steps):
        raise RuntimeError("Missing terminal validation checkpoints; not a divergence or a valid score.")
    if any(not math.isfinite(values[s]) for s in steps):
        raise RuntimeError("Nonfinite completed validation record.")
    return float(statistics.median(values[s] for s in steps))


def candidate_score(results, expected_seeds=(289, 290, 291)):
    if sorted(r["identity"]["seed"] for r in results) != sorted(expected_seeds):
        raise RuntimeError("Need exactly the original three tuning seeds; no omission/replacement.")
    if any(r["status"] not in TERMINAL for r in results):
        raise RuntimeError("Incomplete candidate is pending, not an infinite score.")
    if any(r["status"] == "diverged" for r in results):
        return dict(score=None, score_status="positive_infinity", seed_medians=None,
                    n_completed=sum(r["status"] == "completed" for r in results))
    medians = [terminal_median(r["records"]) for r in results]
    return dict(score=float(statistics.mean(medians)), score_status="finite",
                seed_medians=medians, sample_sd=statistics.stdev(medians), n_completed=3)


def choose_candidate(table):
    valid = [r for r in table if r["score_status"] == "finite"]
    if not valid:
        return None, None
    best = min(valid, key=lambda r: (r["score"], r["rate"]))
    rates = [r["rate"] for r in table]
    extension = best["rate"] / 3 if best["rate"] == min(rates) else (
        best["rate"] * 3 if best["rate"] == max(rates) else None)
    if extension is not None and (not math.isfinite(extension) or extension <= 0 or extension in rates):
        raise RuntimeError("Grid extension reached floating-point limit; no invented optimum.")
    return best, extension


def target_events(result, thresholds=(.06, .045)):
    events = []
    for threshold in thresholds:
        hit = next((r for r in result["records"] if r["validation_relative_l2"] <= threshold), None)
        events.append(dict(threshold=threshold, event=hit is not None,
                           first_step=hit["step"] if hit else None,
                           wall_seconds=hit["wall_seconds"] if hit else None,
                           training_seconds=hit["training_seconds"] if hit else None,
                           run_status=result["status"],
                           censor_wall_seconds=result["wall_seconds"] if not hit and result["status"] == "completed" else None,
                           failure_wall_seconds=result["wall_seconds"] if result["status"] == "diverged" else None))
    return events


def complete_seed_blocks(p, selection, get_result, partial=False):
    blocks, missing = [], []
    for seed in p["formal"]["seeds"]:
        rows = [get_result(seed, g, selection["configs"][g]) for g in p["groups"]]
        if all(r is not None and r["status"] in TERMINAL for r in rows):
            blocks.append(seed)
        else:
            missing.append(seed)
    if missing and not partial:
        raise RuntimeError(f"Test set stays closed: incomplete seed blocks {missing}.")
    if not blocks:
        raise RuntimeError("Test set stays closed: no complete whole-seed block.")
    return blocks


def simulate_budget(durations, p, workers):
    """Dependency-aware base sweep and seed-barrier final schedule; no extensions."""
    remaining = {g: 15 for g in p["groups"]}
    done = {g: 0 for g in p["groups"]}
    active, now, counts = [], 0., 0
    while counts < 135:
        while len(active) < workers:
            ready = [g for g in PRIORITY if remaining[g] and
                     (not hybrid(g) or done[baseline(g)] == 15)]
            if not ready:
                break
            g = ready[0]
            remaining[g] -= 1
            active.append((now + durations[g], g))
        if not active:
            raise RuntimeError("Unresolved dependency in budget simulation.")
        now = min(t for t, _ in active)
        finished = [(t, g) for t, g in active if t == now]
        active = [(t, g) for t, g in active if t != now]
        for _, g in finished:
            done[g] += 1
            counts += 1
    sweep = now
    formal = 0.
    order = final_order(p)
    for seed in p["formal"]["seeds"]:
        slots = [0.] * workers
        for s, g in order:
            if s == seed:
                i = min(range(workers), key=slots.__getitem__)
                slots[i] += durations[g]
        formal += max(slots)
    return dict(sweep_seconds=sweep, formal_seconds=formal, total_seconds=sweep + formal,
                total_days=(sweep + formal) / 86400, baseline_runs=225, extensions_included=False)
