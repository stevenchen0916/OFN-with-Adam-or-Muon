"""Export final errors, logged training dynamics and the prespecified interaction."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
sys.path.insert(0, str(HERE / "burgers_formal16"))
import control as k
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"full_adam": "Full / Adam", "full_muon_adam": "Full / Hybrid",
    "noflow_adam": "No-Flow / Adam", "noflow_muon_adam": "No-Flow / Hybrid",
    "deeponet_adam": "DeepONet / Adam", "deeponet_muon_adam": "DeepONet / Hybrid",
    "deeponet_output_muon_adam": "DeepONet / output-only Hybrid",
    "mionet_adam": "MIONet / Adam", "mionet_muon_adam": "MIONet / Hybrid"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, root, name):
    fig.savefig(root / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(root / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def report(out):
    protocol = k.protocol()
    selected = k.read(out / "selection.json")
    analysis = out / "analysis"
    table = read_csv(analysis / "main_table.csv")
    primary = k.read(analysis / "primary_I.json")
    root = out / "paper_evidence"
    root.mkdir(exist_ok=True)
    curves, runs = [], {}
    for group in protocol["groups"]:
        runs[group] = []
        for seed in protocol["formal"]["seeds"]:
            lr = selected["configs"][group]
            path = k.run_path(out, "formal", group, lr["adam_lr"], lr["muon_lr"], seed)
            r = k.read(path / "result.json")
            runs[group].append(r)
            curves += [dict(group=group, seed=seed, run_status=r["status"], **v) for v in r["records"]]
    k.csv_write(root / "training_curves.csv", curves)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    core = ["full_adam", "full_muon_adam", "noflow_adam", "noflow_muon_adam"]
    for row, groups in enumerate([core, [g for g in protocol["groups"] if g not in core]]):
        for index, group in enumerate(groups):
            local = runs[group]
            color = plt.cm.tab10(index)
            valid = [r for r in local if r["status"] == "completed"]
            # Avoid conditioning curves on successful runs: plot each failed trajectory separately.
            for failed in [r for r in local if r["status"] != "completed"]:
                rec = failed["records"]
                for col, (xkey, ykey) in enumerate([("step", "train_mse"), ("step", "validation_relative_l2"), ("wall_seconds", "validation_relative_l2")]):
                    usable = [v for v in rec if v[ykey] is not None]
                    axes[row, col].plot([v[xkey] for v in usable], [v[ykey] for v in usable], color=color, ls=":", alpha=.6)
            if not valid:
                continue
            steps = sorted(set.intersection(*[{v["step"] for v in r["records"]} for r in valid]))
            lookup = [{v["step"]: v for v in r["records"]} for r in valid]
            for col, (xkey, ykey) in enumerate([("step", "train_mse"), ("step", "validation_relative_l2"), ("wall_seconds", "validation_relative_l2")]):
                usable_steps = [s for s in steps if all(d[s][ykey] is not None for d in lookup)]
                if not usable_steps:
                    continue
                x = np.array([[d[s][xkey] for s in usable_steps] for d in lookup]).mean(0)
                values = np.array([[d[s][ykey] for s in usable_steps] for d in lookup])
                mean = values.mean(0)
                label = LABELS[group] + (f" (successes n={len(valid)}/10)" if len(valid) != 10 else "")
                axes[row, col].plot(x, mean, label=label, color=color)
                if len(valid) > 1:
                    sd = values.std(0, ddof=1)
                    axes[row, col].fill_between(x, np.maximum(mean-sd, 1e-12), mean+sd, color=color, alpha=.12)
        for col, ax in enumerate(axes[row]):
            ax.set(yscale="log", xlabel="Mean active wall time (s) at common steps" if col == 2 else "Update",
                ylabel="Training MSE (before logged update)" if col == 0 else "Validation mean per-function relative L2")
            ax.grid(alpha=.2)
        axes[row, 0].legend(fontsize=8)
    save(fig, root, "burgers_training_dynamics")

    valid = [r for r in table if r["mean_test_error"]]
    if valid:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.barh(np.arange(len(valid)), [100*float(r["mean_test_error"]) for r in valid],
            xerr=[100*float(r["sample_sd"]) for r in valid], capsize=3, color="#4679a9")
        ax.set(yticks=np.arange(len(valid)), yticklabels=[LABELS[r["group"]] for r in valid],
            xlabel="Final test mean per-function relative L2 (%)", title="Burgers: mean ± sample SD across 10 seeds")
        ax.invert_yaxis()
        save(fig, root, "burgers_final_errors")

    effects = read_csv(analysis / "seed_effects.csv")
    irows = [r for r in effects if r.get("I_log")]
    if irows:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax.scatter([int(r["seed"]) for r in irows], [float(r["I_log"]) for r in irows], color="#336b87", label="Paired seed interaction")
        ax.axhline(0, c=".5", lw=1)
        ax.axhline(primary["mean"], color="#c45538", label="Mean interaction")
        if primary["t_ci_low"] is not None:
            ax.axhspan(primary["t_ci_low"], primary["t_ci_high"], color="#c45538", alpha=.15, label="95% paired-t CI for the mean")
        ax.set(xlabel="Training seed", ylabel="I = log(E Full,H / E Full,A) − log(E No-Flow,H / E No-Flow,A)",
            title="Burgers: " + primary["analysis_status"].replace("_", " "))
        ax.legend(fontsize=9)
        save(fig, root, "burgers_primary_interaction")

    paired = [dict(PDE="Burgers", **primary)]
    heat = HERE / "provenance/heat_primary_I.json"
    if heat.exists():
        paired.insert(0, dict(PDE="Heat", **k.read(heat)))
    k.csv_write(root / "two_pde_interactions.csv", paired)
    k.write(root / "evidence_status.json", dict(Burgers=primary["analysis_status"],
        formal_seeds=protocol["formal"]["seeds"], protocol_id=protocol["protocol_id"],
        interpretation="Separate PDE-specific tuned interactions. No pooled claim, no requirement that signs agree.",
        heat_reference_present=heat.exists()))
    (root / "图注与口径.md").write_text(
        "# Burgers 正式图表\n\n"
        "误差图：固定第 20,000 步，逐函数计算时空相对 L2 后对 200 个新测试函数取算术平均；柱为十训练种子均值，误差条为种子样本标准差。失败配置不填完整十种子均值，见 main_table.csv。\n\n"
        "训练动态图：500 步一次，训练 MSE 为该次更新前的损失，验证误差为更新后的完整验证集结果；阴影为成功种子之间的样本标准差。若有失败，失败轨迹另以虚线呈现，成功均值标出分母。时间轴取共同步数处的种子平均活动墙钟时间，包含观测和检查点开销，不含暂停与排队；阴影不是时间不确定区间。固定预算不构成收敛证明。\n\n"
        "交互图：I=ln(E_Full,Hybrid/E_Full,Adam)−ln(E_NoFlow,Hybrid/E_NoFlow,Adam)。负值表示 Full 的比例误差下降更强；点为训练种子，阴影为均值的 95% 配对 t 区间。学习率在本 PDE 验证集上各自选择，因此交互条件于本次分阶段调参流程。完整十种子缺失时推断仅为探索性。\n\n"
        "双 PDE 表分别报告两道方程的 I 及区间；不把不同 PDE 的误差或种子合并，不以符号一致作为实验成功条件。\n", encoding="utf-8")
    bundle = out / "Burgers_正式结果_日志与图表.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for directory in [analysis, root]:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    z.write(file, str(file.relative_to(out)))
        for name in ["protocol.json", "manifest.json", "selection.json", "dry_run_report.json", "validation_lr.csv", "status.json"]:
            if (out / name).exists():
                z.write(out / name, name)
        for group, local in runs.items():
            for r in local:
                ident = r["identity"]
                path = k.run_path(out, "formal", group, ident["adam_lr"], ident["muon_lr"], ident["seed"])
                for name in ["result.json", "validation.csv", "optimizer.json", "hardware.json", "run.log"]:
                    if (path / name).exists():
                        z.write(path / name, str((path / name).relative_to(out)))
    print("Paper evidence:", root, "\nDownload:", bundle, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results16")
    args = parser.parse_args()
    report(args.out.expanduser().resolve())
