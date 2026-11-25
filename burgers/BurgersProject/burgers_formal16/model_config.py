"""Burgers architectures fixed by input dimensions and parameter counts, before training."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "support"
sys.path.insert(0, str(V2))
import common as c
from deepxde.nn.pytorch.deeponet import DeepONetCartesianProd
from deepxde.nn.pytorch.mionet import MIONetCartesianProd
from models.qofn import QOFN

LAYERS = {
    "deeponet": {"branch": [26] + [62] * 6 + [25], "trunk": [5] + [61] * 6 + [25]},
    "mionet": {"branch1": [13] + [50] * 6 + [25], "branch2": [13] + [50] * 6 + [25], "trunk": [5] + [50] * 6 + [25]},
    "qofn": {"branch1": [14] + [50] * 11 + [25], "branch2": [14] + [50] * 11 + [25], "trunk": [6] + [50] * 11 + [25]},
}


def architecture(group):
    if group not in c.GROUPS:
        raise ValueError(group)
    return group.split("_")[0]


def _construct(name):
    spec = LAYERS["qofn" if name in ["full", "noflow"] else name]
    if name == "deeponet":
        return DeepONetCartesianProd(spec["branch"], spec["trunk"], "relu", "Glorot normal")
    if name == "mionet":
        return MIONetCartesianProd(spec["branch1"], spec["branch2"], spec["trunk"], "relu", "Glorot normal")
    return QOFN(spec["branch1"], spec["branch2"], spec["trunk"], "relu", flow=name == "full")


def _dense_count(dims):
    return sum(a*b+b for a,b in zip(dims[:-1], dims[1:]))


# Dense counts follow their layer lists; orthogonal counts are asserted explicitly.
EXPECTED_PARAMETERS = {
    "deeponet": 1 + sum(_dense_count(v) for v in LAYERS["deeponet"].values()),
    "mionet": 1 + sum(_dense_count(v) for v in LAYERS["mionet"].values()),
    "full": 43728,
    "noflow": 43695,
}


def build_model(group):
    name = architecture(group)
    net = _construct(name)
    count = sum(p.numel() for p in net.parameters())
    if count != EXPECTED_PARAMETERS[name]:
        raise RuntimeError(f"Unexpected parameter count for {name}: {count}")
    return net


def parameter_table():
    baseline = EXPECTED_PARAMETERS["deeponet"]
    return [dict(architecture=name, parameters=count,
        relative_difference_from_deeponet=(count-baseline)/baseline)
        for name,count in EXPECTED_PARAMETERS.items()]
