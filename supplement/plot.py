"""Compact baseline mechanism panels; medians/IQR across ten training seeds."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(p.read_text()) for p in sorted((out / "diagnostics").rglob("seed_*.json"))]
    groups = ["full_adam", "full_muon_adam", "noflow_adam", "noflow_muon_adam"]
    if len(rows) != 40 or any(sum(r["group"] == g for r in rows) != 10 for g in groups):
        raise RuntimeError("Plot requires all 40 prespecified source models.")
    labels = ["Full / Adam", "Full / Muon–Adam", "No-Flow / Adam", "No-Flow / Muon–Adam"]
    colors = ["#3974a8", "#d46b37", "#3974a8", "#d46b37"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), layout="constrained")
    def band(ax, x, values, color, label, ls="-"):
        lo, mid, hi = np.quantile(values, [.25, .5, .75], axis=0)
        ax.plot(x, mid, color=color, label=label, ls=ls, lw=1.5)
        ax.fill_between(x, lo, hi, color=color, alpha=.13)
    for g, label, color in zip(groups, labels, colors):
        seeds = [r for r in rows if r["group"] == g]
        layers = [[v for v in r["layers"] if v["encoder"] == "branch2"] for r in seeds]
        depth = np.arange(1, len(layers[0]) + 1)
        ls = "--" if g.startswith("noflow") else "-"
        if g.startswith("full"):
            active = [i for i, v in enumerate(layers[0]) if v["gate_active"]]
            band(axes[0, 0], depth[active], np.array([[s[i]["alpha"] for i in active] for s in layers]), color, label)
        radial = [[np.median(v["post_radial_gain"]) for v in s] for s in layers]
        band(axes[0, 1], depth, np.maximum(radial, 1e-10), color, label, ls)
        maximum = [[np.median(np.asarray(v["post_singular_values"])[:, 0]) for v in s] for s in layers]
        band(axes[1, 0], depth, maximum, color, label, ls)
        values = []
        for r in seeds:
            params = {v["name"]: v for v in r["parameters"]}
            values.append([params[f"{name}.output_layer.weight"]["gradient_norm"]
                           for name in ["branch1", "branch2", "trunk"]])
        band(axes[1, 1], np.arange(3), values, color, label, ls)
    titles = ["(a) Active Flow gates", "(b) Post-ReLU radial response",
              "(c) Largest complete-layer singular value", "(d) Output projection gradient norm"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(alpha=.15)
    axes[0, 0].set(xlabel="Initial-field branch hidden layer", ylabel="Gate h")
    axes[0, 1].set(xlabel="Initial-field branch hidden layer", ylabel="‖J x / ‖x‖‖", yscale="log")
    axes[1, 0].set(xlabel="Initial-field branch hidden layer", ylabel="Largest singular value")
    axes[1, 1].set(ylabel="Frobenius norm", yscale="log", xticks=np.arange(3),
                   xticklabels=["Diffusivity", "Initial field", "Trunk"])
    axes[1, 0].legend(fontsize=7, loc="best")
    dest = out / "figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / "mechanism_snapshot.png", dpi=240)
    fig.savefig(dest / "mechanism_snapshot.pdf")
    plt.close(fig)
    (dest / "caption.txt").write_text(
        "Step-20000 descriptive diagnostics of all 40 Full/No-Flow formal models. "
        "Lines: median over 10 training seeds; bands: interquartile ranges, not confidence intervals. "
        "Layer probes use four fixed validation inputs; within-input summaries use medians. "
        "Gradients are recomputed on the same fixed 16 training functions and 128 coordinates; "
        "they are snapshots, not historical full-batch gradient curves. Jacobians include normalization, "
        "Flow mixing and ReLU. Near-zero No-Flow radial responses reflect floating-point residuals. "
        "Inactive width-changing gates are omitted. These measurements do not identify a causal training mechanism.\n")
    print(dest / "mechanism_snapshot.png")


if __name__ == "__main__":
    main()
