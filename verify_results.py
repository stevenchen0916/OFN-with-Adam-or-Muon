"""Recompute the formal result tables from the archived run-level metrics."""

import argparse
import csv
import json
import math
from pathlib import Path

from heat_formal_v3_16 import control


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=ROOT / "replay", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Choose a new output directory. {args.out} already exists.")

    with (ROOT / "results/test_runs.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for key in ("seed", "parameters"):
            row[key] = int(row[key])
        for key in ("training_seconds", "wall_seconds", "test_relative_l2"):
            row[key] = float(row[key]) if row[key] else None
        row["targets"] = json.loads(row["targets"])

    protocol = control.protocol()
    expected = {(seed, group) for seed in protocol["formal"]["seeds"] for group in protocol["groups"]}
    actual = {(row["seed"], row["group"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("The archived results do not match the planned seed blocks.")

    # The archived analyzer imports its sibling control module as a top-level module.
    import sys
    sys.path.insert(0, str(ROOT / "heat_formal_v3_16"))
    from analysis import analyze

    primary = analyze(args.out, protocol, rows)
    archived = json.loads((ROOT / "results/primary_I.json").read_text())
    for key in ("mean", "t_ci_low", "t_ci_high", "paired_t_p"):
        if not math.isclose(primary[key], archived[key], rel_tol=1e-10):
            raise ValueError(f"Recomputed primary statistic differs for {key}.")
    for row in json.loads((args.out / "paired_effects.json").read_text()):
        if row["contrast"] in ("full", "noflow") and row["scale"] == "natural_log_ratio":
            print(f"{row['contrast']} hybrid-to-Adam ratio: {row['geometric_error_ratio']:.3f}")
    print(
        f"Ratio of ratios: {math.exp(primary['mean']):.3f} "
        f"(95% CI [{math.exp(primary['t_ci_low']):.3f}, {math.exp(primary['t_ci_high']):.3f}])"
    )
    print(f"Replayed {len(rows)} runs. Tables written to {args.out}.")


if __name__ == "__main__":
    main()
