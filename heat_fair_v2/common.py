"""Shared data, model, normalization, device and hardware helpers for v2.

No experiment is started by this module.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import deepxde as dde

# DeepXDE selects MPS/CUDA globally while importing its PyTorch backend.
# Keep initialization and float64 correctness checks on CPU; training inputs
# and models are moved explicitly to the CLI-selected device below.
torch.set_default_device("cpu")

from deepxde.nn.pytorch.deeponet import DeepONetCartesianProd
from deepxde.nn.pytorch.mionet import MIONetCartesianProd
from models.qofn import QOFN

HERE = Path(__file__).resolve().parent
GROUPS = ["deeponet_adam", "deeponet_muon_adam", "mionet_adam",
          "mionet_muon_adam", "full_adam", "full_muon_adam",
          "noflow_adam", "noflow_muon_adam", "deeponet_output_muon_adam"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_hash():
    return object_hash({str(p.relative_to(HERE)): digest(p)
                        for p in sorted(HERE.rglob("*.py"))
                        if "." not in p.relative_to(HERE).parts[0]})


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def exclusive(root):
    from runtime import lock
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".lock"):
        yield


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment(device):
    hardware = {}
    if sys.platform == "darwin":
        for field in ("machdep.cpu.brand_string", "hw.model", "hw.memsize"):
            result = subprocess.run(["sysctl", "-n", field], capture_output=True, text=True)
            if result.returncode == 0:
                hardware[field] = result.stdout.strip()
            else:
                hardware[field] = {"unavailable": result.stderr.strip()}
    elif sys.platform == "win32":
        import psutil
        hardware = dict(total_memory_bytes=psutil.virtual_memory().total,
                        physical_cpus=psutil.cpu_count(logical=False),
                        logical_cpus=psutil.cpu_count(logical=True))
    elif sys.platform.startswith("linux"):
        for field, path in (("cpuinfo", "/proc/cpuinfo"), ("meminfo", "/proc/meminfo")):
            if Path(path).is_file():
                hardware[field] = Path(path).read_text()[:6000]
    return dict(python=sys.version, executable=sys.executable, hardware=hardware,
                torch=str(torch.__version__), deepxde=str(dde.__version__),
                numpy=np.__version__, platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), threads=torch.get_num_threads(),
                device=str(device), cuda=torch.version.cuda,
                gpu=(torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                gpu_memory_bytes=(torch.cuda.get_device_properties(device).total_memory
                                  if device.type == "cuda" else None),
                mps_available=torch.backends.mps.is_available(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                torch_build=torch.__config__.show())


def require_muon(device):
    if not hasattr(torch.optim, "Muon") or "adjust_lr_fn" not in inspect.signature(torch.optim.Muon).parameters:
        raise RuntimeError("This PyTorch lacks Muon(adjust_lr_fn=...). Use the original project's environment.")
    # An actual update checks the option and the selected device, not only the signature.
    p = torch.nn.Parameter(torch.eye(2, device=device))
    opt = torch.optim.Muon([p], lr=1e-3, weight_decay=0,
                           adjust_lr_fn="match_rms_adamw")
    p.grad = torch.ones_like(p)
    opt.step()
    if not torch.isfinite(p).all():
        raise RuntimeError("Muon capability check returned nonfinite parameters.")








def repair_targets(y, nt=20, nx=20, ny=20):
    # Legacy target: (t,x,y), corrected target: (y,x,t).
    return y.reshape(-1, nt, nx, ny).transpose(0, 3, 2, 1).reshape(y.shape).copy()


def safe_minmax(x, low, high):
    scale = np.where(np.abs(high - low) < 1e-12, 1, high - low)
    z = (2 * (x - low) / scale - 1) / np.sqrt(x.shape[1])
    return np.concatenate((z, np.sqrt(np.clip(1 - (z * z).sum(1, keepdims=True), 0, None))), 1).astype(np.float32)




class Dataset:
    """Training constructor deliberately opens no test file."""
    def __init__(self, root, device):
        self.root, self.device = root, device
        self.manifest = read_json(root / "data/manifest.json")
        body = {k: v for k, v in self.manifest.items() if k != "dataset_id"}
        if object_hash(body) != self.manifest["dataset_id"]:
            raise RuntimeError("Dataset manifest changed.")
        if digest(root / "data/preprocessing.npz") != self.manifest["files"]["preprocessing.npz"]:
            raise RuntimeError("Preprocessing changed.")
        with np.load(root / "data/preprocessing.npz") as f:
            self.bounds = [(f[f"low{i}"], f[f"high{i}"]) for i in range(3)]
        self.cache = {}

    def load(self, split, qofn):
        key = (split, qofn)
        if key not in self.cache:
            name = f"{split}.npz"
            if digest(self.root / "data" / name) != self.manifest["files"][name]:
                raise RuntimeError(f"Dataset changed: {name}")
            with np.load(self.root / "data" / name) as f:
                x = [f[name] for name in ("x1", "x2", "xt")]
                y = f["y"]
            if qofn:
                x = [safe_minmax(a, *b) for a, b in zip(x, self.bounds)]
            self.cache[key] = (tuple(torch.as_tensor(a, device=self.device) for a in x),
                               torch.as_tensor(y, device=self.device))
        return self.cache[key]


def is_qofn(group):
    return group.startswith(("full_", "noflow_"))


def build_model(group):
    # Exact dimensions from the original heat PCA Config; activations unchanged.
    if group.startswith("deeponet_"):
        return DeepONetCartesianProd([11] + [25] * 7, [3] + [25] * 7,
                                     "relu", "Glorot normal")
    if group.startswith("mionet_"):
        return MIONetCartesianProd([1, 8, 8, 25], [10] + [25] * 7,
                                   [3] + [25] * 7, "relu", "Glorot normal")
    return QOFN([2, 8, 8, 25], [11] + [25] * 12, [4] + [25] * 12,
                "relu", flow=group.startswith("full_"))


def inputs_for(group, x):
    return (torch.cat(x[:2], dim=1), x[2]) if group.startswith("deeponet_") else x

