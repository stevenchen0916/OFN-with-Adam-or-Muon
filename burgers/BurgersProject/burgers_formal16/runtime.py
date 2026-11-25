"""Execution manifests, guarded datasets, locking and machine observations."""
from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import control as k
import model_config as m
import torch

IS_WINDOWS = sys.platform == "win32"
if not IS_WINDOWS:
    import fcntl
    import resource

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ["control.py", "runtime.py", "engine.py", "runner.py", "analysis.py",
                "throughput.py", "model_config.py"]
_CONTEXT_CACHE = {}


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if IS_WINDOWS:
            import msvcrt
            f.seek(0, 2)
            if f.tell() == 0:
                f.write(b"0"); f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise RuntimeError(f"Another process owns {path}") from e
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another process owns {path}")
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def spawn(command, **kwargs):
    if IS_WINDOWS:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(command, **kwargs)


def interrupt_process(process):
    if process.poll() is None:
        import signal
        process.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGINT)


def code_identity():
    return dict(local={name: k.sha(HERE / name) for name in SOURCE_FILES},
                old_helper_sha=m.c.code_hash(), execution_amendment_sha=k.sha(HERE/"execution_16cpu.json"),
                campaign_source={name:k.sha(HERE.parent/name) for name in
                    ["generator.py", "prepare_data.py", "run_campaign.py", "report.py", "check_package.py", "run_burgers16.slurm"]})


def configure(threads):
    torch.set_default_device("cpu")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    torch.use_deterministic_algorithms(True)


def machine_usage():
    if IS_WINDOWS:
        import psutil
        memory = psutil.Process().memory_info()
        swap = psutil.swap_memory()
        virtual = psutil.virtual_memory()
        return dict(peak_rss_bytes=int(getattr(memory, "peak_wset", memory.rss)),
                    rss_bytes=memory.rss, system_available_memory_bytes=virtual.available,
                    system_memory_percent=virtual.percent,
                    swap=dict(total_bytes=swap.total,used_bytes=swap.used,percent=swap.percent),
                    swap_counters_note="Windows psutil does not provide meaningful sin/sout paging rates")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    row = dict(peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)))
    if sys.platform.startswith("linux"):
        import psutil
        memory=psutil.virtual_memory();swap=psutil.swap_memory()
        row.update(rss_bytes=psutil.Process().memory_info().rss,
                   system_available_memory_bytes=memory.available,
                   system_memory_percent=memory.percent,
                   swap=dict(total_bytes=swap.total,used_bytes=swap.used,
                             sin_bytes=swap.sin,sout_bytes=swap.sout))
    if sys.platform == "darwin":
        for key, cmd in [("swap", ["sysctl", "-n", "vm.swapusage"]),
                         ("power", ["pmset", "-g", "batt"])]:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                row[key] = r.stdout.strip() if r.returncode == 0 else "unavailable: " + r.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as e:
                row[key] = "unavailable: " + str(e)
    return row


def initialize(out):
    out = Path(out)
    p = k.protocol()
    manifest = k.read(m.V2 / "results/data/manifest.json")
    if manifest["dataset_id"] != p["data"]["dataset_id"]:
        raise RuntimeError("Data identity differs from frozen protocol.")
    current = dict(protocol_id=p["protocol_id"], protocol_file_sha=k.sha(HERE / "protocol.json"),
                   code=code_identity(), dataset_id=manifest["dataset_id"],
                   software=dict(torch=str(torch.__version__), deepxde=str(m.c.dde.__version__),
                                 numpy=str(m.c.np.__version__), python=platform.python_version()),
                   device="cpu", interoperability_threads=1)
    file = out / "manifest.json"
    if file.exists():
        if k.read(file) != current:
            raise RuntimeError("Code/protocol/software changed. Refusing to mix versions in this output directory.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        k.write(file, current)
        (out / "protocol.json").write_bytes((HERE / "protocol.json").read_bytes())
        k.write(out / "resources.json", dict(workers=1, threads_per_worker=4, device="cpu"))
        for name in SOURCE_FILES:
            dest=out/"source_snapshot"/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(HERE/name,dest)
    if k.sha(out / "protocol.json") != current["protocol_file_sha"]:
        raise RuntimeError("Copied protocol changed.")
    return p, current


def resources(out):
    r = k.read(Path(out) / "resources.json")
    if (r["workers"], r["threads_per_worker"]) not in [(1, 4), (2, 2), (4, 4)] or r["device"] != "cpu":
        raise RuntimeError("Execution resources must be the prespecified CPU 1x4, 2x2 or user-authorized 4x4 setup.")
    return r


def resource_id(out):
    return k.object_sha(resources(out))[:12]


def context(out):
    out=Path(out)
    watch=[HERE/n for n in SOURCE_FILES]+[HERE/"protocol.json",out/"resources.json",out/"manifest.json"]
    watch += sorted(m.V2.rglob("*.py")) + list(HERE.parent.glob("*.py")) + [HERE/"execution_16cpu.json"]
    signature=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in watch if p.exists())
    cached=_CONTEXT_CACHE.get(str(out))
    if cached is not None and cached[0]==signature:
        return cached[1]
    p, manifest = initialize(out)
    r = resources(out)
    value=dict(protocol_id=p["protocol_id"], manifest_sha=k.object_sha(manifest),
                resources=r, resource_id=resource_id(out), dataset_id=manifest["dataset_id"])
    _CONTEXT_CACHE[str(out)]=(signature,value)
    return value


def require_approval(out):
    path = Path(out) / "execution_approval.json"
    if not path.exists():
        raise RuntimeError("Run smoke, inspect its report/budget, then approve before starting the scan.")
    value = k.read(path)
    if value["context"] != context(out):
        raise RuntimeError("Approved code/resources differ from the current execution.")
    smoke = Path(out) / "dry_run" / resource_id(out) / "report.json"
    if not smoke.exists() or k.sha(smoke) != value["smoke_report_sha"]:
        raise RuntimeError("Approved dry-run report changed.")
    return value


class TestPermit:
    """Created by the evaluation controller only after all eligibility checks."""
    def __init__(self, selection_sha, seeds):
        self.selection_sha, self.seeds = selection_sha, seeds


class GuardedData:
    def __init__(self, permit=None, provider=None):
        self.permit = permit
        self.source = provider if provider is not None else m.c.Dataset(m.V2 / "results", torch.device("cpu"))
        self.manifest = self.source.manifest

    def load(self, split, qofn):
        if split == "test" and not isinstance(self.permit, TestPermit):
            raise RuntimeError("Test split is closed during training, tuning and dry runs.")
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        return self.source.load(split, qofn)


def run_identity(ctx, stage, group, seed, adam_lr, muon_lr, total_steps=None):
    if group not in k.protocol()["groups"]:
        raise ValueError(group)
    return dict(context=ctx, stage=stage, group=group, seed=seed, adam_lr=adam_lr,
                muon_lr=muon_lr, total_steps=total_steps or 20000,
                horizon=20000, minimum_lr_ratio=.01,
                validation_every=50 if stage.startswith("dry_run") else 500,
                checkpoint_every=100 if stage.startswith("dry_run") else 2000)


def load_result(path, expected=None):
    file = Path(path) / "result.json"
    if not file.exists():
        return None
    r = k.read(file)
    if expected is not None and r["identity"] != expected:
        raise RuntimeError(f"Run settings mismatch: {path}")
    if r["status"] not in k.TERMINAL:
        raise RuntimeError(f"Invalid terminal result: {file}")
    if r["status"] == "completed":
        if r["actual_steps"] != r["identity"]["total_steps"]:
            raise RuntimeError("Completed result did not reach its fixed budget.")
        if k.sha(Path(path) / "final.pt") != r["checkpoint_sha"]:
            raise RuntimeError(f"Final model changed: {path}")
    return r


def emit_event(out, event, **fields):
    path = Path(out) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        import json
        f.write(json.dumps(dict(time_unix=time.time(), event=event, **fields), ensure_ascii=False) + "\n")
