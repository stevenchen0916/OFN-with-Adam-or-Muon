"""Budget-matched wide models; import starts no experiment.

The small kappa branches retain width 8; the main encoders become width 50.
All final branch/trunk representations have dimension p=25. Original depths,
activations, Full/No-Flow normalization, and optimizer routing are preserved.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "heat_fair_v2"
sys.path.insert(0, str(V2))
import common as c
from deepxde.nn.pytorch.deeponet import DeepONetCartesianProd
from deepxde.nn.pytorch.mionet import MIONetCartesianProd
from models.qofn import QOFN

LAYERS = {
    "deeponet": {"branch": [11] + [50] * 6 + [25], "trunk": [3] + [50] * 6 + [25]},
    "mionet": {"branch1": [1, 8, 8, 25], "branch2": [10] + [50] * 6 + [25], "trunk": [3] + [50] * 6 + [25]},
    "qofn": {"branch1": [2, 8, 8, 25], "branch2": [11] + [50] * 11 + [25], "trunk": [4] + [50] * 11 + [25]},
}
EXPECTED_PARAMETERS = {"deeponet": 28851, "mionet": 29114, "full": 29131, "noflow": 29107}


def architecture(group):
    if group not in c.GROUPS:
        raise ValueError(group)
    return group.split("_")[0]


def build_model(group):
    name = architecture(group)
    spec = LAYERS["qofn" if name in ["full", "noflow"] else name]
    if name == "deeponet":
        net = DeepONetCartesianProd(spec["branch"], spec["trunk"], "relu", "Glorot normal")
    elif name == "mionet":
        net = MIONetCartesianProd(spec["branch1"], spec["branch2"], spec["trunk"], "relu", "Glorot normal")
    else:
        net = QOFN(spec["branch1"], spec["branch2"], spec["trunk"], "relu", flow=name == "full")
    count = sum(p.numel() for p in net.parameters())
    if count != EXPECTED_PARAMETERS[name]:
        raise RuntimeError(f"Unexpected parameter count for {name}: {count}")
    return net


def parameter_table():
    baseline = EXPECTED_PARAMETERS["deeponet"]
    return [dict(architecture=name, parameters=count,
                 relative_difference_from_deeponet=(count-baseline)/baseline)
            for name, count in EXPECTED_PARAMETERS.items()]
