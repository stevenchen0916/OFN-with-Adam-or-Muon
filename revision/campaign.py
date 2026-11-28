"""Validation selection, formal retraining and paired evaluation of OFN controls."""
import argparse
import csv
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("full", "noflow", "full_raw", "noflow_raw", "full_fixed")
METHODS = ("adam", "split", "hybrid")


def archive_inputs(pde):
    directory = ROOT / ("results" if pde == "heat" else "burgers/results")

    def read_table(name):
        with (directory / name).open(newline="") as f:
            return list(csv.DictReader(f))

    def config(row):
        group = row["group"]
        model = group.split("_")[0]
        method = "hybrid" if group.endswith("_muon_adam") else "adam"
        proj = float(row["muon_lr"]) if row["muon_lr"] else None
        return model, method, float(row["adam_lr"]), proj, 20000

    original = {config(row): row["group"] for row in read_table("selected_learning_rates.csv")
                if row["group"].startswith(("full_", "noflow_"))}
    if len(original) != 4:
        raise ValueError("Expected four published OFN settings")
    cached = {}
    for row in read_table("validation_lr.csv"):
        if row["group"] not in original.values():
            continue
        if row["score_status"] == "finite":
            if int(row["n_completed"]) != 3 or len(json.loads(row["seed_medians"])) != 3:
                raise ValueError("Incomplete archived validation score")
            cached[config(row)] = float(row["score"])
        elif row["score_status"] == "positive_infinity":
            cached[config(row)] = math.inf
        else:
            raise ValueError("Unknown validation status")
    return original, cached


def candidates(original):
    # Both published architectures select the same auxiliary Adam rate.
    bases = {cfg[2] for cfg in original}
    if bases != {.001}:
        raise ValueError("Archived base rates differ from the local search design")
    projections = sorted({cfg[3] for cfg in original if cfg[1] == "hybrid"})
    if len(projections) != 2:
        raise ValueError("Expected two distinct published projection rates")
    for model, method, base, proj in itertools.product(
            ("full", "noflow"), ("split", "hybrid"), (.0003, .001), projections):
        yield model, method, base, proj, 20000


def select(scores, original):
    chosen = {}
    common_proj = next(cfg[3] for cfg in original if cfg[:2] == ("noflow", "hybrid"))
    for cfg in original:
        model, method = cfg[:2]
        chosen[f"published/{model}/{method}"] = cfg
        if method == "adam":
            chosen[f"individual/{model}/adam"] = cfg
            chosen[f"common/{model}/adam"] = cfg
        else:
            chosen[f"common/{model}/hybrid"] = (model, "hybrid", .001, common_proj, 20000)
            chosen[f"common/{model}/split"] = (model, "split", .001, common_proj, 20000)
    for model, method in itertools.product(("full","noflow"), ("split","hybrid")):
        eligible = [k for k in scores if k[:2] == (model, method) and math.isfinite(scores[k])]
        if not eligible:
            raise ValueError(f"No finite local candidate for {model}/{method}")
        chosen[f"individual/{model}/{method}"] = min(eligible, key=lambda k: (scores[k], k))
    for model, method in itertools.product(MODELS[2:], ("adam","hybrid")):
        reference = chosen[f"published/{model.split('_')[0]}/{method}"]
        chosen[f"individual/{model}/{method}"] = (model, *reference[1:])
    for cfg in original:
        chosen[f"long/{cfg[0]}/{cfg[1]}"] = (*cfg[:-1], 40000)
    return chosen


def model_for(spec, group):
    import model_config
    from normalization_controls import unnormalized_preactivation
    net = model_config.build_model(group)
    for encoder in (net.branch1, net.branch2, net.trunk):
        if spec.endswith("_raw"):
            encoder.hidden_preactivation = MethodType(unnormalized_preactivation, encoder)
        if spec.endswith("_fixed"):
            for gate in encoder.h_params:
                gate.data.zero_()
                gate.requires_grad_(False)
    return net


def split_adam(net, group, base, projection):
    import torch
    weights = [encoder.output_layer.weight for encoder in (net.branch1, net.branch2, net.trunk)]
    ids = {id(p) for p in weights}
    named = dict(net.named_parameters())
    rest = [p for p in named.values() if id(p) not in ids]
    opts = [torch.optim.Adam(weights, lr=projection), torch.optim.Adam(rest, lr=base)]
    return opts, dict(projection_adam=[n for n,p in named.items() if id(p) in ids],
                      other_adam=[n for n,p in named.items() if id(p) not in ids],
                      projection_lr=projection, base_lr=base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pde", choices=["heat", "burgers"])
    parser.add_argument("stage", choices=["plan", "smoke", "tune", "freeze", "formal", "evaluate"])
    parser.add_argument("--scope", choices=["core","all"], default="core")
    parser.add_argument("--index", type=int, help="zero-based task, omit to process sequentially")
    parser.add_argument("--batch", type=int, help="zero-based array batch of eight runs")
    parser.add_argument("--lane", type=int, choices=range(4), help="worker within a batch")
    args = parser.parse_args()
    if (args.batch is None) != (args.lane is None):
        parser.error("--batch and --lane must be supplied together")
    if args.batch is not None and (args.batch < 0 or args.index is not None):
        parser.error("invalid batch or conflicting --index")
    if (args.batch is not None or args.index is not None) and args.stage not in ("tune", "formal"):
        parser.error("task indices apply only to training stages")
    module = (ROOT / "heat_formal_v3_16" if args.pde == "heat" else
              ROOT / "burgers/BurgersProject/burgers_formal16")
    protocol = json.loads((module / "protocol.json").read_text())
    published, archived_scores = archive_inputs(args.pde)
    grid = list(candidates(published))
    tuning = list(itertools.product([cfg for cfg in grid if cfg not in archived_scores], protocol["tuning"]["seeds"]))
    seeds = protocol["formal"]["seeds"]
    if args.stage == "plan":
        print(f"{len(tuning)} tuning runs; at most {(7 if args.scope == "core" else 17)*len(seeds)} new formal runs (duplicates reused)")
        return
    sys.path[:0] = [str(module), str(ROOT / "heat_formal_v3_16")]
    import engine
    import runtime as rt
    rt.configure(4)
    out = ROOT / "revision_runs" / args.pde
    # Slurm starts four processes together. Serialize first-time manifest creation.
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".initialize.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        first_launch = not (out / "manifest.json").exists()
        rt.initialize(out)
        if first_launch:
            rt.k.write(out / "resources.json", dict(workers=4, threads_per_worker=4, device="cpu"))
    ctx = dict(rt.context(out), revision=rt.k.sha(Path(__file__)),
               controls=rt.k.sha(ROOT / "heat_formal_v3_16/normalization_controls.py"))
    archive = ROOT / ("results" if args.pde == "heat" else "burgers/results")
    ctx["archive_hashes"] = {name: rt.k.sha(archive/name) for name in ("selected_learning_rates.csv", "validation_lr.csv")}
    original = engine.make_optimizers

    def location(cfg, seed, stage):
        model, method, base, proj, steps = cfg
        group = model.split("_")[0] + ("_adam" if method == "adam" else "_muon_adam")
        ident = rt.run_identity(dict(ctx, model=model, method=method), stage, group, seed, base, proj, steps)
        ident["horizon"] = steps
        return out / stage / rt.k.object_sha(ident), ident

    def run(jobs, stage):
        if args.index is not None:
            if not 0 <= args.index < len(jobs):
                parser.error(f"index outside 0..{len(jobs)-1}")
            jobs = [jobs[args.index]]
        if args.batch is not None:
            first = args.batch * 8 + args.lane
            jobs = jobs[first:(args.batch + 1)*8:4]
        for cfg, seed in jobs:
            path, ident = location(cfg, seed, stage)
            engine.make_optimizers = split_adam if cfg[1] == "split" else original
            result = engine.train(path, ident, factory=lambda group: model_for(cfg[0], group),
                                  routing_check=(stage == "smoke"))
            if stage == "smoke" and result["status"] != "completed":
                raise RuntimeError(f"Smoke check failed for {cfg}")

    if args.stage == "smoke":
        smoke_jobs = []
        for model in (MODELS if args.scope == "all" else MODELS[:2]):
            for method in (METHODS if model in ("full","noflow") else ("adam","hybrid")):
                proj = None if method == "adam" else .001
                smoke_jobs.append(((model, method, .0003, proj, 2), 289))
        run(smoke_jobs, "smoke")
        return

    if args.stage == "tune":
        run(tuning, "tune")
        return
    frozen = out / "selection.json"
    if args.stage == "freeze":
        scores = {}
        for cfg in grid:
            if cfg in archived_scores:
                scores[cfg] = archived_scores[cfg]
                continue
            rows = []
            for seed in protocol["tuning"]["seeds"]:
                path, ident = location(cfg, seed, "tune")
                result = rt.load_result(path, ident)
                if result is None:
                    raise RuntimeError(f"Missing tuning run {cfg}, seed {seed}")
                rows.append(result)
            scores[cfg] = (st.mean(rt.k.terminal_median(r["records"]) for r in rows)
                           if all(r["status"] == "completed" for r in rows) else math.inf)
        table = [dict(model=k[0], method=k[1], base_lr=k[2], projection_lr=k[3],
                      source="published_validation" if k in archived_scores else "new_validation",
                      score=v if math.isfinite(v) else None) for k,v in scores.items()]
        payload = dict(context=ctx, chosen=select(scores, published), validation_scores=table)
        with rt.lock(out / "freeze.lock"):
            if frozen.exists() and rt.k.read(frozen) != json.loads(json.dumps(payload)):
                raise RuntimeError("Existing selection differs")
            rt.k.write(frozen, payload)
        return
    selection = rt.k.read(frozen)
    if selection["context"] != ctx:
        raise RuntimeError("Frozen context differs")
    chosen = {name: tuple(cfg) for name,cfg in selection["chosen"].items()}
    if args.scope == "core":
        chosen = {k:v for k,v in chosen.items() if v[0] in ("full","noflow") and v[-1] == 20000}
    jobs = list(itertools.product(sorted(set(chosen.values()) - set(published)), seeds))
    if args.stage == "formal":
        print(f"{len(jobs)} formal runs")
        run(jobs, "formal")
        return
    import torch
    completed = {}
    for cfg, seed in jobs:
        path, ident = location(cfg, seed, "formal")
        result = rt.load_result(path, ident)
        if result is None:
            raise RuntimeError(f"Missing formal run {cfg}, seed {seed}")
        if result["status"] == "completed" and result["checkpoint_sha"] != rt.k.sha(path / "final.pt"):
            raise RuntimeError(f"Changed checkpoint {path}")
        completed[cfg,seed] = result
    if args.pde == "heat":
        import replicate
        fresh = ROOT / "results_fresh_test_revision"
        manifest = replicate.read(fresh / "data/manifest.json")
        if manifest["n_test"] != 1000:
            raise ValueError("Expected frozen 1000-input heat test")
        replicate.init_evaluator(ROOT, fresh, manifest["frozen_inputs_sha"], manifest["test_sha"])
        x, y = replicate.WORKER["xq"], replicate.WORKER["target"]
    else:
        data = rt.GuardedData(rt.TestPermit(rt.k.sha(frozen), seeds))
        x, y = data.load("test", True)
    errors = {}
    if args.pde == "burgers":
        with (archive / "test_runs.csv").open(newline="") as f:
            old = {(r["group"],int(r["seed"])):float(r["test_relative_l2"])
                   for r in csv.DictReader(f) if r["status"]=="completed"}
    for cfg, group in published.items():
        for seed in seeds:
            if args.pde == "heat":
                row = replicate.read(fresh / "scores" / f"{group}_seed_{seed}.json")
                if (row["group"],row["seed"],row["fresh_test_sha"]) != (group,seed,manifest["test_sha"]):
                    raise RuntimeError("Fresh heat baseline and control test inputs differ")
                errors[cfg,seed] = row["test_relative_l2"]
            else:
                errors[cfg,seed] = old[group,seed]
    for cfg, seed in jobs:
        path, ident = location(cfg, seed, "formal")
        score = None
        if completed[cfg,seed]["status"] == "completed":
            net = model_for(cfg[0], ident["group"])
            net.load_state_dict(torch.load(path / "final.pt", map_location="cpu", weights_only=True))
            score = engine.relative_error(net, ident["group"], x, y)
        errors[cfg,seed] = score
    rows = {name: [errors[cfg,s] for s in seeds] for name,cfg in chosen.items()}
    effects = {}
    import analysis
    for section, full, noflow in [("published","full","noflow"), ("individual","full","noflow"), ("common","full","noflow"),
                                  ("long","full","noflow"), ("individual","full_raw","noflow_raw"),
                                  ("individual","full_fixed","noflow")]:
        if f"{section}/{full}/hybrid" not in rows:
            continue
        for base in (["adam","split"] if section in ("individual","common") and full == "full" else ["adam"]):
            series = [rows[f"{section}/{model}/{method}"] for model in (full,noflow)
                      for method in ("hybrid",base)]
            key = f"{section}/{full}-{noflow}/hybrid-{base}"
            effects[key] = None
            if all(v is not None and v > 0 for row in series for v in row):
                contrasts = [math.log(a/b)-math.log(c/d) for a,b,c,d in zip(*series)]
                effects[key] = dict(by_seed=contrasts, ratio=math.exp(st.mean(contrasts)),
                                    summary=analysis.contrast_summary(contrasts))
    per_model = {}
    for section, model in itertools.product(("common", "individual"), ("full", "noflow")):
        hybrid = rows[f"{section}/{model}/hybrid"]
        split = rows[f"{section}/{model}/split"]
        key = f"{section}/{model}/hybrid-split"
        per_model[key] = None
        if all(v is not None and v > 0 for v in hybrid + split):
            delta = [math.log(h/s) for h,s in zip(hybrid, split)]
            per_model[key] = dict(by_seed=delta, ratio=math.exp(st.mean(delta)),
                                  summary=analysis.contrast_summary(delta))
    # Paired changes answer whether the interaction itself moves with a control.
    changes = {}
    reference = effects["published/full-noflow/hybrid-adam"]
    for label, comparison in (("double_budget", "long/full-noflow/hybrid-adam"),
                              ("remove_normalization", "individual/full_raw-noflow_raw/hybrid-adam"),
                              ("fix_gate", "individual/full_fixed-noflow/hybrid-adam")):
        if comparison not in effects:
            continue
        controlled = effects[comparison]
        changes[label] = None
        if reference and controlled:
            delta = [a-b for a,b in zip(controlled["by_seed"], reference["by_seed"])]
            changes[label] = dict(by_seed=delta, summary=analysis.contrast_summary(delta))
    rt.k.write(out / f"evaluation_{args.scope}.json", dict(seeds=seeds, errors=rows, interactions=effects,
               interaction_changes=changes, per_model=per_model,
               selection_sha=rt.k.sha(frozen), test_id=manifest["test_sha"] if args.pde=="heat" else ctx["dataset_id"]))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "fresh_test"))
    main()
