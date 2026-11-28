"""Run a heat normalization control at an explicitly chosen rate and seed."""
import argparse
from functools import partial
import math
from pathlib import Path
from types import MethodType


def unnormalized_preactivation(encoder, x, index):
    """Apply the original rotation and optional gate to the raw hidden vector."""
    import torch

    x = x.float()
    z = encoder.hidden_layers[index](x)
    if encoder.flow and x.shape[1] == z.shape[1]:
        h = torch.sigmoid(encoder.h_params[index])
        return (1.0 - h) * x + h * z
    return z


def build_control(group, *, normalized):
    import model_config

    net = model_config.build_model(group)
    if not normalized:
        for encoder in (net.branch1, net.branch2, net.trunk):
            encoder.hidden_preactivation = MethodType(unnormalized_preactivation, encoder)
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", choices=("full", "noflow"), required=True)
    parser.add_argument("--normalization", choices=("on", "off"), required=True)
    parser.add_argument("--optimizer", choices=("adam", "muon_adam"), default="adam")
    parser.add_argument("--adam-lr", type=float, required=True)
    parser.add_argument("--muon-lr", type=float)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if (args.optimizer == "muon_adam") != (args.muon_lr is not None):
        parser.error("--muon-lr is required exclusively for muon_adam")
    if any(not math.isfinite(rate) or rate <= 0 for rate in
           (args.adam_lr, args.muon_lr) if rate is not None):
        parser.error("learning rates must be finite and positive")

    import engine
    import runtime as rt

    root = Path(__file__).resolve().parent.parent / "runs" / "normalization_controls"
    rt.configure(4)
    with rt.lock(root.parent / ".normalization_controls_init.lock"):
        rt.initialize(root)
    ctx = dict(rt.context(root), normalization=args.normalization,
               control_code_sha=rt.k.sha(Path(__file__)))
    group = f"{args.flow}_{args.optimizer}"
    identity = rt.run_identity(ctx, "normalization_control", group, args.seed,
                               args.adam_lr, args.muon_lr)
    path = root / args.normalization / group / f"seed_{args.seed}" / rt.k.object_sha(identity)
    result = engine.train(path, identity, factory=partial(
        build_control, normalized=args.normalization == "on"))
    print(path, result["status"])
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
