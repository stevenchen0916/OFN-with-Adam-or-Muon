"""Explicit staged execution of the immutable Burgers formal protocol."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import zipfile

PROCESS_STARTED=time.perf_counter()

import control as k
import model_config as m
import runtime as rt
import engine
import torch

HERE = Path(__file__).resolve().parent
_RESULT_CACHE = {}


def result(path, expected=None):
    path = Path(path)
    file = path / "result.json"
    if not file.exists():
        return None
    sig = (file.stat().st_mtime_ns, file.stat().st_size)
    final = path / "final.pt"
    if final.exists():
        sig += (final.stat().st_mtime_ns, final.stat().st_size)
    cached = _RESULT_CACHE.get(str(path))
    if cached is None or cached[0] != sig:
        cached = (sig, rt.load_result(path, expected))
        _RESULT_CACHE[str(path)] = cached
    r = cached[1]
    if expected is not None and r["identity"] != expected:
        raise RuntimeError("Mismatched cached run identity.")
    return r


def spec(out, stage, group, seed, adam, muon, steps=20000):
    identity = rt.run_identity(rt.context(out), stage, group, seed, adam, muon, steps)
    path = k.run_path(out, stage, group, adam, muon, seed)
    return dict(path=str(path), identity=identity)


def dispatch(out, provider, *, dry=False, max_new_runs=None):
    """Bounded worker pool; provider rechecks architecture-local dependencies."""
    resources = rt.resources(out)
    active, started, stop = {}, 0, [False]
    old = {}
    def interrupt(signum, frame):
        stop[0] = True
        print("Stopping after current updates; saving complete checkpoints…", flush=True)
        for proc in active.values():
            if proc.poll() is None:
                rt.interrupt_process(proc)
    for sig in ((signal.SIGINT, signal.SIGTERM, signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else (signal.SIGINT, signal.SIGTERM)):
        old[sig] = signal.signal(sig, interrupt)
    error = None
    try:
        while True:
            jobs = provider()
            for job in jobs:
                path = job["path"]
                if stop[0] or path in active or len(active) >= resources["workers"]:
                    continue
                if max_new_runs is not None and started >= max_new_runs:
                    break
                p = Path(path)
                p.mkdir(parents=True, exist_ok=True)
                jobfile = p / "job.json"
                k.write(jobfile, job)
                phase = "train"
                if dry:
                    phase = "dry-second" if (p / "resume_reference.pt").exists() else "dry-first"
                cmd = [sys.executable, "-u", str(HERE / "runner.py"), "--out", str(out),
                       "_worker", "--spec", str(jobfile), "--phase", phase]
                active[path] = rt.spawn(cmd)
                started += 1
                rt.emit_event(out, "worker_started", path=path, phase=phase, pid=active[path].pid)
                print(f"START {job['identity']['group']} seed={job['identity']['seed']} phase={phase}", flush=True)
            if not active:
                break
            for path, proc in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[path]
                rt.emit_event(out, "worker_exited", path=path, exit_code=code)
                if code == 130:
                    stop[0] = True
                elif code != 0:
                    error = RuntimeError(f"Worker software/infrastructure error ({code}): {path}/run.log")
                    stop[0] = True
                if stop[0]:
                    for other in active.values():
                        if other.poll() is None:
                            rt.interrupt_process(other)
            if not stop[0]:
                time.sleep(.25)
            elif active:
                time.sleep(.25)
        if error:
            raise error
        if stop[0]:
            raise engine.Interrupted("Campaign interrupted; repeat the same command to resume.")
    except BaseException:
        for proc in active.values():
            if proc.poll() is None:
                rt.interrupt_process(proc)
        for proc in active.values():
            proc.wait()
        raise
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def worker(out, args):
    job = k.read(args.spec)
    identity = job["identity"]
    current = rt.context(out)
    expected = rt.run_identity(current, identity["stage"], identity["group"], identity["seed"],
                              identity["adam_lr"], identity["muon_lr"],
                              200 if identity["stage"].startswith("dry_run/") else 20000)
    if identity != expected:
        raise RuntimeError("Worker spec differs from current frozen execution.")
    if Path(job["path"]).resolve() != k.run_path(out, identity["stage"], identity["group"],
            identity["adam_lr"], identity["muon_lr"], identity["seed"]).resolve():
        raise RuntimeError("Worker output path mismatch.")
    rt.configure(current["resources"]["threads_per_worker"])
    kwargs = {}
    if args.phase != "train":
        if not identity["stage"].startswith("dry_run/"):
            raise RuntimeError("Resume checks are isolated to dry runs.")
        kwargs = dict(routing_check=True, stop_at=100 if args.phase == "dry-first" else None,
                      prepare_resume_reference=args.phase == "dry-first", verify_resume=args.phase == "dry-second")
    engine.train(job["path"], identity, session_started=PROCESS_STARTED, **kwargs)


def dry_jobs(out):
    p = k.protocol()
    stage = "dry_run/" + rt.resource_id(out)
    jobs = [spec(out, stage, g, 289, .001, .001 if k.hybrid(g) else None, 200) for g in p["groups"]]
    return [j for j in jobs if result(j["path"], j["identity"]) is None]


def smoke_report(out):
    p = k.protocol()
    ctx = rt.context(out)
    stage = "dry_run/" + rt.resource_id(out)
    rows, durations, failures = [], {}, []
    for group in p["groups"]:
        job = spec(out, stage, group, 289, .001, .001 if k.hybrid(group) else None, 200)
        r = result(job["path"], job["identity"])
        if r is None or r["status"] != "completed" or not r["resume_verified"] or not r["routing_verified"]:
            failures.append(group)
            continue
        mean = r["mean_step_seconds"]
        estimate = mean * 20000 + 41 * r["mean_observation_seconds"] + 11 * r["mean_checkpoint_seconds"] + r["setup_seconds"]
        durations[group] = estimate
        rows.append(dict(group=group, parameters=r["parameters"], mean_step_seconds=mean,
                         median_step_seconds=r["median_step_seconds"], last_100_mean_seconds=r["last_100_mean_seconds"],
                         estimated_20000_step_seconds=estimate, resume_verified=True, isolated_routing_verified=True,
                         both_schedulers_verified=True, warmup_excluded=20, main_updates=200))
    root = Path(out) / stage
    value = dict(status="passed" if not failures else "incomplete_or_failed", context=ctx,
                 failures=failures, timings=rows,
                 budget=k.simulate_budget(durations, p, ctx["resources"]["workers"]) if not failures else None,
                 budget_limitations="200-step estimates, not all-day throughput; excludes LR extensions; includes sampled observer/checkpoint overhead",
                 validation_used_to_select_hyperparameters=False, test_evaluations=0)
    k.write(root / "report.json", value)
    k.csv_write(root / "timings.csv", rows)
    lines = ["九配置试运行报告", "状态：" + value["status"], "", "配置 | 参数量 | 平均秒/步 | 中位秒/步 | 断点/路由/调度"]
    lines += [f"{r['group']} | {r['parameters']} | {r['mean_step_seconds']:.6f} | {r['median_step_seconds']:.6f} | 通过" for r in rows]
    if value["budget"]:
        b = value["budget"]
        lines += ["", f"基础225次预计活动墙钟合计：{b['total_seconds']/3600:.1f}小时，{b['total_days']:.2f}天。", value["budget_limitations"]]
    lines += ["", "试运行没有用于调参；测试集未打开。扫描尚未自动启动。", "原始结果、硬件、swap和断点位于本报告同目录各配置子目录。"]
    (root / "给Claude_试运行结果.txt").write_text("\n".join(lines) + "\n")
    # Convenient stable filenames outside the resource-specific directory.
    import shutil
    shutil.copy2(root/"给Claude_试运行结果.txt",Path(out)/"给Claude_试运行结果.txt")
    shutil.copy2(root/"report.json",Path(out)/"dry_run_report.json")
    shutil.copy2(root/"timings.csv",Path(out)/"dry_run_timings.csv")
    if not failures:
        bundle=Path(out)/"试运行结果_供核对.zip"
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.rglob("*")):
                if file.is_file() and not file.name.startswith('.') and file.suffix!='.tmp':
                    z.write(file,str(Path("dry_run")/file.relative_to(root)))
            for name in ["protocol.json","manifest.json","unit_verification.json"]:
                if (Path(out)/name).exists():
                    z.write(Path(out)/name,name)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("Nine-configuration smoke gate is not yet passed.")
    return value


def require_unit_verification(out):
    file=Path(out)/"unit_verification.json"
    manifest=k.read(Path(out)/"manifest.json")
    if not file.exists():
        raise RuntimeError("Run verify before the real-data smoke trial.")
    r=k.read(file)
    if not r["passed"] or r["code"]!=manifest["code"] or r["protocol_id"]!=k.FROZEN_ID:
        raise RuntimeError("Synthetic verification does not match the current code; repeat verify.")
    current_tests={name:k.sha(HERE/name) for name in ["verify.py","verify_runner.py","verify_platform.py","verify_parallel16.py","verify_burgers_data.py"]}
    if r.get("test_source_sha")!=current_tests:
        raise RuntimeError("Verification test sources changed; repeat verify.")


def verify_all(out):
    before=rt.code_identity();protocol_sha=k.sha(HERE/"protocol.json")
    suites=[]
    for filename in ["verify.py","verify_runner.py","verify_platform.py","verify_parallel16.py","verify_burgers_data.py"]:
        target=Path(out)/(filename.removesuffix(".py")+".log")
        print(f"Running {filename}; log: {target}",flush=True)
        with target.open("w") as log:
            process=subprocess.run([sys.executable,"-u",str(HERE/filename)],stdout=log,stderr=subprocess.STDOUT)
        lines=target.read_text().splitlines()
        print("\n".join(lines[-6:]),flush=True)
        if process.returncode:
            raise RuntimeError(f"Verification failed; inspect {target}")
        import re
        count=re.search(r"Ran (\d+) tests?",target.read_text())
        if count is None:
            raise RuntimeError("No unittest execution count in verification log.")
        suites.append(dict(file=filename,test_count=int(count.group(1))))
    if before!=rt.code_identity() or protocol_sha!=k.sha(HERE/"protocol.json"):
        raise RuntimeError("Source/protocol changed during verification.")
    value=dict(passed=True,protocol_id=k.FROZEN_ID,protocol_file_sha=protocol_sha,code=before,
               tests=suites,
               test_source_sha={name:k.sha(HERE/name) for name in ["verify.py","verify_runner.py","verify_platform.py","verify_parallel16.py","verify_burgers_data.py"]},
               real_training_started=False,test_split_opened=False)
    k.write(Path(out)/"unit_verification.json",value)
    print("Verification passed. Real-data smoke trial has not started.")


def search_choice(out, group):
    file = Path(out) / "search" / (group + ".selected.json")
    return k.read(file) if file.exists() else None


def refresh_search(out, families=None):
    p = k.protocol()
    ctx = rt.context(out)
    jobs, unresolved = [], []
    for group in k.PRIORITY:
        if families and group.split("_")[0] not in families:
            continue
        selected_file = Path(out) / "search" / (group + ".selected.json")
        if selected_file.exists():
            continue
        fixed = None
        if k.hybrid(group):
            pure = search_choice(out, k.baseline(group))
            if pure is None:
                continue
            fixed = pure["adam_lr"]
        statefile = Path(out) / "search" / (group + ".json")
        if statefile.exists():
            state = k.read(statefile)
            if state["context"] != ctx or state["fixed_adam_lr"] != fixed:
                raise RuntimeError("Search state changed execution or its auxiliary Adam LR.")
        else:
            state = dict(context=ctx, group=group, fixed_adam_lr=fixed,
                         rates=p["tuning"]["muon_initial_lr_grid" if k.hybrid(group) else "adam_initial_lr_grid"],
                         status="searching")
            k.write(statefile, state)
        if state["status"] == "unresolved":
            unresolved.append(group)
            continue
        table, source_hashes, pending = [], {}, []
        for rate in state["rates"]:
            adam, muon = (fixed, rate) if k.hybrid(group) else (rate, None)
            results = []
            for seed in p["tuning"]["seeds"]:
                job = spec(out, "sweep", group, seed, adam, muon)
                r = result(job["path"], job["identity"])
                if r is None:
                    pending.append(job)
                else:
                    results.append(r)
                    relative = str((Path(job["path"]) / "result.json").relative_to(out))
                    source_hashes[relative] = k.sha(Path(job["path"]) / "result.json")
            if len(results) == 3:
                table.append(dict(rate=rate, adam_lr=adam, muon_lr=muon, **k.candidate_score(results)))
        if pending:
            jobs += pending
            continue
        best, extension = k.choose_candidate(table)
        state["table"] = table
        if best is None:
            state["status"] = "unresolved"
            state["reason"] = "No learning rate has three finite completed tuning seeds."
            unresolved.append(group)
        elif extension is not None:
            state["rates"] = sorted(state["rates"] + [extension])
            state["extension_count"] = state.get("extension_count", 0) + 1
            print(f"EXTEND {group}: add initial LR {extension:.9g}; original three seeds retained.", flush=True)
        else:
            state["status"] = "selected"
            selected = dict(group=group, adam_lr=best["adam_lr"], muon_lr=best["muon_lr"],
                            score=best["score"], rates=state["rates"], context=ctx,
                            table=table, source_hashes=source_hashes)
            k.write(selected_file, selected)
            print(f"SELECTED {group}: Adam={best['adam_lr']:.7g}, Muon={best['muon_lr']}, score={best['score']:.8g}", flush=True)
        k.write(statefile, state)
    choices = {g: search_choice(out, g) for g in p["groups"]}
    if all(choices.values()):
        selected = dict(context=ctx, protocol_id=p["protocol_id"], steps=20000,
                        tuning_seeds=p["tuning"]["seeds"], final_seeds=p["formal"]["seeds"],
                        configs=choices, source_hashes={str((Path("search") / (g + ".selected.json"))):
                                                    k.sha(Path(out) / "search" / (g + ".selected.json")) for g in choices})
        target = Path(out) / "selection.json"
        if target.exists() and k.read(target) != selected:
            raise RuntimeError("Frozen LR selection changed.")
        k.write(target, selected)
    # Extensions/dependencies created in this pass become ready on the next pass.
    if not jobs and not all(choices.values()) and not unresolved:
        if any((not families or g.split("_")[0] in families) and (Path(out)/"search"/(g+".json")).exists() and
               k.read(Path(out)/"search"/(g+".json"))["status"] == "searching" for g in choices):
            return refresh_search(out, families)
    return jobs, unresolved


def selection(out):
    file = Path(out) / "selection.json"
    if not file.exists():
        raise RuntimeError("All 9 LR searches must finish before formal seed training.")
    selected = k.read(file)
    if selected["context"] != rt.context(out) or set(selected["configs"]) != set(k.protocol()["groups"]):
        raise RuntimeError("Frozen LR selection context mismatch.")
    for group, c in selected["configs"].items():
        for rel, digest in c["source_hashes"].items():
            if k.sha(Path(out) / rel) != digest:
                raise RuntimeError("Tuning evidence changed after LR selection.")
    for rel, digest in selected["source_hashes"].items():
        if k.sha(Path(out)/rel) != digest:
            raise RuntimeError("Selected LR file changed.")
    return selected


def formal_jobs(out, selected):
    p = k.protocol()
    for seed in p["formal"]["seeds"]:
        jobs = []
        for s, group in k.final_order(p):
            if s != seed:
                continue
            lr = selected["configs"][group]
            job = spec(out, "formal", group, seed, lr["adam_lr"], lr["muon_lr"])
            if result(job["path"], job["identity"]) is None:
                jobs.append(job)
        if jobs:
            return jobs
    return []


def evaluate(out, partial=False, reason=None, *, provider=None, factory=None):
    p, selected = k.protocol(), selection(out)
    if partial and not reason:
        raise RuntimeError("Partial emergency export requires the infrastructure interruption reason.")
    def get(seed, group, lr):
        job = spec(out, "formal", group, seed, lr["adam_lr"], lr["muon_lr"])
        return result(job["path"], job["identity"])
    seeds = k.complete_seed_blocks(p, selected, get, partial)
    selected_sha = k.sha(Path(out)/"selection.json")
    invariant = dict(protocol_id=p["protocol_id"], context=rt.context(out), selection_sha=selected_sha,
                     test_sha=k.read(m.V2/"results/data/manifest.json")["files"]["test.npz"])
    marker = Path(out)/"test_evaluation.json"
    if marker.exists() and k.read(marker) != invariant:
        raise RuntimeError("Test cache already belongs to different frozen settings.")
    k.write(marker, invariant)
    permit = rt.TestPermit(selected_sha, seeds)
    data = rt.GuardedData(permit=permit, provider=provider)
    todo, rows = [], []
    for seed in seeds:
        for group in p["groups"]:
            lr = selected["configs"][group]
            job = spec(out, "formal", group, seed, lr["adam_lr"], lr["muon_lr"])
            r = get(seed, group, lr)
            key = k.object_sha(dict(invariant=invariant, group=group, seed=seed,
                                    result_sha=k.sha(Path(job["path"])/"result.json")))
            cache = Path(out)/"test_scores"/f"{group}_seed_{seed}.json"
            pending = cache.with_suffix(".pending.json")
            if cache.exists():
                row = k.read(cache)
                if row["evaluation_key"] != key:
                    raise RuntimeError("Test cache checkpoint/settings mismatch.")
                rows.append(row)
            else:
                if pending.exists():
                    raise RuntimeError(f"Uncommitted test attempt found: {pending}. No automatic repeat scoring.")
                todo.append((seed, group, r, job, key, cache, pending))
    # Every checkpoint and cache has been checked before the first test data load.
    for seed, group, r, job, key, cache, pending in todo:
        row = dict(group=group, seed=seed, status=r["status"], parameters=r["parameters"],
                   training_seconds=r["training_seconds"], wall_seconds=r["wall_seconds"],
                   targets=r["targets"], evaluation_key=key, test_relative_l2=None,
                   failure_reason=r.get("reason"))
        if r["status"] == "completed":
            k.write(pending, dict(evaluation_key=key, checkpoint_sha=r["checkpoint_sha"],
                                 message="Transaction started; interrupted scoring is never silently repeated."))
            m.c.seed_all(seed)
            net = (factory or m.build_model)(group)
            net.load_state_dict(torch.load(Path(job["path"])/"final.pt",map_location="cpu",weights_only=True))
            x, y = data.load("test", m.c.is_qofn(group))
            try:
                row["test_relative_l2"] = engine.relative_error(net, group, x, y)
                print(f"TEST {group} seed={seed}: {row['test_relative_l2']:.8g}", flush=True)
            except FloatingPointError as error:
                row["status"]="test_nonfinite"
                row["failure_reason"]=str(error)
        k.write(cache, row)
        if pending.exists():
            pending.unlink()
        rows.append(row)
    rows.sort(key=lambda r:(r["seed"], p["groups"].index(r["group"])))
    from analysis import analyze
    target = Path(out)/("analysis_partial" if partial else "analysis")
    value = analyze(target, p, rows)
    rt.emit_event(out,"test_export",partial=partial,seeds=seeds,reason=reason)
    print(f"Analysis saved: {target}; primary I n={value['n']}, status={value['analysis_status']}",flush=True)
    return rows


def search_report(out):
    p = k.protocol()
    rows = []
    for group in p["groups"]:
        file = Path(out)/"search"/(group+".json")
        if file.exists():
            rows += [dict(group=group, **r) for r in k.read(file).get("table",[])]
    k.csv_write(Path(out)/"validation_lr.csv",rows)
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for ax,g in zip(axes.flat,p["groups"]):
        rs=sorted((r for r in rows if r["group"]==g and r["score_status"]=="finite"),key=lambda r:r["rate"])
        ax.errorbar([r["rate"] for r in rs],[r["score"] for r in rs],
                    yerr=[r.get("sample_sd",0) for r in rs],marker="o",capsize=3)
        ax.set(xscale="log",title=g,xlabel="Initial Muon LR" if k.hybrid(g) else "Initial Adam LR",ylabel="Validation tail-median, 3-seed mean")
        ax.title.set_fontsize(9)
        ax.grid(alpha=.2)
    fig.savefig(Path(out)/"validation_lr.png",dpi=180)
    fig.savefig(Path(out)/"validation_lr.pdf")
    plt.close(fig)


def status_report(out):
    rows=[]
    for path in sorted(Path(out).rglob("progress.json")):
        r=k.read(path)
        rows.append(dict(path=str(path.parent.relative_to(out)),**r))
    k.csv_write(Path(out)/"status.csv",rows)
    counts={s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})}
    value=dict(protocol_id=k.FROZEN_ID,resources=rt.resources(out),counts=counts,
               lr_selection_frozen=(Path(out)/"selection.json").exists(),
               machine=rt.machine_usage(),
               timing_drift=[r for r in rows if r.get("timing_drift_ratio",0) and r["timing_drift_ratio"]>1.2])
    k.write(Path(out)/"status.json",value)
    print(json.dumps(value,ensure_ascii=False,indent=2))


def backup(out,destination,include_data=False):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    destination=Path(destination).expanduser().resolve()
    if destination.is_relative_to(HERE.resolve()) or destination.is_relative_to(Path(out).resolve()):
        raise RuntimeError("Backup destination must be outside the code/results tree.")
    destination.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    target=destination/f"burgers_backup_{stamp}.zip"
    files=[]
    for file in HERE.iterdir():
        if file.is_file() and file.suffix in [".py",".json",".md",".sh",".txt"]:
            files.append((file,Path("burgers_formal16")/file.name))
    for file in Path(out).rglob("*"):
        if file.is_file() and not file.name.startswith(".") and file.suffix!=".tmp":
            files.append((file,Path("results16")/file.relative_to(out)))
    for name in ["run16.sh", "run_campaign.py", "run_burgers16.slurm", "check_package.py", "prepare_data.py", "generator.py", "report.py", "PACKAGE_MANIFEST.json", "运行说明.md"]:
        source=HERE.parent/name
        if source.exists():
            files.append((source,Path(name)))
    for file in (HERE.parent/"provenance").rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts:
            files.append((file,Path("provenance")/file.relative_to(HERE.parent/"provenance")))
    for file in m.V2.rglob("*.py"):
        if "__pycache__" not in file.parts and not any(x.startswith(".") for x in file.relative_to(m.V2).parts):
            files.append((file,Path("support")/file.relative_to(m.V2)))
    data=m.V2/"results/data"
    files.append((data/"manifest.json",Path("support/results/data/manifest.json")))
    if include_data:
        files += [(file,Path("support/results/data")/file.name) for file in data.glob("*.npz")]
    with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as z:
        for file,name in files:
            z.write(file,str(name))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise RuntimeError("Backup integrity check failed.")
    print(f"Backup verified: {target} ({target.stat().st_size/2**20:.1f} MiB)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"results")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["init","check","plan","verify","smoke","smoke-report","select","final","status"]:
        sub.add_parser(name)
    a=sub.add_parser("approve");a.add_argument("--note",required=True)
    a=sub.add_parser("resources");a.add_argument("--workers",type=int,choices=[1,2,4],required=True);a.add_argument("--threads",type=int,choices=[2,4],required=True)
    a=sub.add_parser("sweep");a.add_argument("--families",nargs="+",choices=["deeponet","mionet","full","noflow"]);a.add_argument("--max-new-runs",type=int)
    a=sub.add_parser("evaluate");a.add_argument("--partial",action="store_true");a.add_argument("--reason")
    a=sub.add_parser("backup");a.add_argument("--dest",type=Path,required=True);a.add_argument("--include-data",action="store_true")
    a=sub.add_parser("throughput");a.add_argument("--seconds",type=int,default=600)
    a=sub.add_parser("_worker");a.add_argument("--spec",type=Path,required=True);a.add_argument("--phase",choices=["train","dry-first","dry-second"],required=True)
    args=parser.parse_args();args.out=args.out.expanduser().resolve()
    if args.command=="_worker":
        worker(args.out,args);return
    with rt.lock(args.out/(".status.lock" if args.command in ["status","plan"] else ".controller.lock")):
        p,manifest=rt.initialize(args.out)
        resources=rt.resources(args.out)
        rt.configure(resources["threads_per_worker"])
        if args.command in ["init","check"]:
            if args.command=="check":
                m.c.require_muon(torch.device("cpu"))
            print(json.dumps(dict(protocol_id=p["protocol_id"],resources=resources,
                                  parameters=m.parameter_table(),data_id=manifest["dataset_id"]),indent=2))
        elif args.command=="plan":
            order=k.final_order(p)
            print(json.dumps(dict(baseline_sweep_runs=135,formal_runs=90,total=225,
                tuning_seeds=p["tuning"]["seeds"],formal_seeds=p["formal"]["seeds"],
                sweep_priority=k.PRIORITY,first_formal_seed_order=[g for s,g in order if s==292]),indent=2))
        elif args.command=="verify":
            verify_all(args.out)
        elif args.command=="resources":
            if (args.workers,args.threads) not in [(1,4),(2,2),(4,4)]:
                raise RuntimeError("Use 1x4, 2x2 or the user-authorized 4x4 execution.")
            if (args.out/"execution_approval.json").exists() or (args.out/"sweep").exists():
                raise RuntimeError("Resources are frozen after approval/search start.")
            k.write(args.out/"resources.json",dict(workers=args.workers,threads_per_worker=args.threads,device="cpu"))
            print("Resources selected. Run smoke for this setup before approval.")
        elif args.command=="smoke":
            require_unit_verification(args.out)
            dispatch(args.out,lambda:dry_jobs(args.out),dry=True)
            smoke_report(args.out)
            print("Dry run ends here. Inspect/send its report; no scan has started.")
        elif args.command=="smoke-report":
            smoke_report(args.out)
        elif args.command=="approve":
            require_unit_verification(args.out)
            value=smoke_report(args.out)
            report=args.out/"dry_run"/rt.resource_id(args.out)/"report.json"
            approval=dict(context=rt.context(args.out),smoke_report_sha=k.sha(report),note=args.note)
            target=args.out/"execution_approval.json"
            if target.exists() and k.read(target)!=approval:
                raise RuntimeError("Execution approval already frozen.")
            k.write(target,approval)
            print("Execution checks and budget review recorded. No scan started by this command.")
        elif args.command=="sweep":
            rt.require_approval(args.out)
            dispatch(args.out,lambda:refresh_search(args.out,args.families)[0],max_new_runs=args.max_new_runs)
            _,unresolved=refresh_search(args.out,args.families)
            search_report(args.out)
            if unresolved:
                raise RuntimeError("No valid LR for: "+", ".join(unresolved))
            print("Scan command finished/paused at its run limit. No formal-seed training started.")
        elif args.command=="select":
            rt.require_approval(args.out)
            refresh_search(args.out)
            print(json.dumps(selection(args.out)["configs"],indent=2))
            search_report(args.out)
        elif args.command=="final":
            rt.require_approval(args.out)
            chosen=selection(args.out)
            dispatch(args.out,lambda:formal_jobs(args.out,chosen))
            print("Formal seed blocks reached terminal status. Test set remains closed; use evaluate.")
        elif args.command=="evaluate":
            rt.require_approval(args.out)
            evaluate(args.out,args.partial,args.reason)
        elif args.command=="status":
            status_report(args.out)
        elif args.command=="backup":
            backup(args.out,args.dest,args.include_data)
        elif args.command=="throughput":
            from throughput import compare
            compare(args.out,args.seconds)


if __name__=="__main__":
    try:
        main()
    except (KeyboardInterrupt,engine.Interrupted) as e:
        print(str(e) or "Interrupted. Repeat the same command to resume.",file=sys.stderr)
        sys.exit(130)
