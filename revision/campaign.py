"""Validation selection, formal retraining and paired evaluation of OFN controls."""
import argparse
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


def candidates(protocol):
    tuning = protocol["tuning"]
    base = tuning["adam_initial_lr_grid"]
    output = sorted(set(base + tuning["muon_initial_lr_grid"]))
    for model, method, rate in itertools.product(("full","noflow"), METHODS, base):
        for proj in ([None] if method == "adam" else output):
            yield model, method, rate, proj, 20000


def select(scores):
    """Keys are (model, method, base rate, projection rate, steps)."""
    chosen = {}
    for model, method in itertools.product(("full","noflow"), METHODS):
        eligible = [k for k in scores if k[:2] == (model, method) and math.isfinite(scores[k])]
        if not eligible:
            raise ValueError(f"No finite candidate for {model}/{method}")
        chosen[f"individual/{model}/{method}"] = min(eligible, key=lambda k: (scores[k], k))
    # Equal-weight validation score for both architectures and all three methods.
    pairs = []
    for model, method, base, proj, steps in scores:
        if (model, method) != ("full", "split"):
            continue
        keys = [(arch, opt, base, None if opt == "adam" else proj, steps)
                for arch, opt in itertools.product(("full","noflow"), METHODS)]
        value = st.mean(scores[k] for k in keys)
        if math.isfinite(value):
            pairs.append((value, base, proj, keys))
    if not pairs:
        raise ValueError("No jointly finite common rate pair")
    for k in min(pairs)[-1]:
        chosen[f"common/{k[0]}/{k[1]}"] = k
    # Hold rates fixed within each structural comparison.
    for model, method in itertools.product(MODELS[2:], ("adam","hybrid")):
        reference = chosen[f"individual/{model.split('_')[0]}/{method}"]
        chosen[f"individual/{model}/{method}"] = (model, *reference[1:])
    # Same initial rates and seeds, different complete cosine horizons.
    for model, method in itertools.product(("full", "noflow"), ("adam", "hybrid")):
        k = chosen[f"individual/{model}/{method}"]
        chosen[f"long/{model}/{method}"] = (*k[:-1], 40000)
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
    grid = list(candidates(protocol))
    tuning = list(itertools.product(grid, protocol["tuning"]["seeds"]))
    seeds = protocol["formal"]["seeds"]
    if args.stage == "plan":
        print(f"{len(tuning)} tuning runs; at most {22*len(seeds)} formal runs (duplicates reused)")
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
        for model in MODELS:
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
                      score=v if math.isfinite(v) else None) for k,v in scores.items()]
        payload = dict(context=ctx, chosen=select(scores), validation_scores=table)
        with rt.lock(out / "freeze.lock"):
            if frozen.exists() and rt.k.read(frozen) != json.loads(json.dumps(payload)):
                raise RuntimeError("Existing selection differs")
            rt.k.write(frozen, payload)
        return
    selection = rt.k.read(frozen)
    if selection["context"] != ctx:
        raise RuntimeError("Frozen context differs")
    chosen = {name: tuple(cfg) for name,cfg in selection["chosen"].items()}
    jobs = list(itertools.product(sorted(set(chosen.values())), seeds))
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
    for section, full, noflow in [("individual","full","noflow"), ("common","full","noflow"),
                                  ("long","full","noflow"), ("individual","full_raw","noflow_raw"),
                                  ("individual","full_fixed","noflow")]:
        for base in (["adam","split"] if section != "long" and full == "full" else ["adam"]):
            series = [rows[f"{section}/{model}/{method}"] for model in (full,noflow)
                      for method in ("hybrid",base)]
            key = f"{section}/{full}-{noflow}/hybrid-{base}"
            effects[key] = None
            if all(v is not None and v > 0 for row in series for v in row):
                contrasts = [math.log(a/b)-math.log(c/d) for a,b,c,d in zip(*series)]
                effects[key] = dict(by_seed=contrasts, ratio=math.exp(st.mean(contrasts)),
                                    summary=analysis.contrast_summary(contrasts))
    # Paired changes answer whether the interaction itself moves with a control.
    changes = {}
    reference = effects["individual/full-noflow/hybrid-adam"]
    for label, comparison in (("double_budget", "long/full-noflow/hybrid-adam"),
                              ("remove_normalization", "individual/full_raw-noflow_raw/hybrid-adam"),
                              ("fix_gate", "individual/full_fixed-noflow/hybrid-adam")):
        controlled = effects[comparison]
        changes[label] = None
        if reference and controlled:
            delta = [a-b for a,b in zip(controlled["by_seed"], reference["by_seed"])]
            changes[label] = dict(by_seed=delta, summary=analysis.contrast_summary(delta))
    rt.k.write(out / "evaluation.json", dict(seeds=seeds, errors=rows, interactions=effects,
               interaction_changes=changes,
               selection_sha=rt.k.sha(frozen), test_id=manifest["test_sha"] if args.pde=="heat" else ctx["dataset_id"]))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "fresh_test"))
    main()
