"""Generate the fixed 1000-input heat split and optionally score the original models."""
import argparse
import itertools
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fresh_test"))
import replicate as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["data", "original"])
    args = parser.parse_args()
    project = ROOT
    out = project / "results_fresh_test_revision"
    with r.locked(out):
        data_dir = project / "heat_fair_v2/results/data"
        old = r.read(data_dir / "manifest.json")
        protocol = dict(n_test=1000, kappa_seed=2026112801, initial_field_seed=2026112802, workers=4)
        seeds = set(sum(old["function_seeds"].values(), [])) | {2026111801, 2026111802}
        if seeds.intersection([protocol["kappa_seed"], protocol["initial_field_seed"]]):
            raise RuntimeError("Fresh seed collision")
        pre_sha = r.digest(data_dir / "preprocessing.npz")
        if pre_sha != old["files"]["preprocessing.npz"]:
            raise RuntimeError("Training preprocessing changed")
        frozen_data = dict(protocol=protocol, preprocessing_sha=pre_sha,
                           generator_sha=r.digest(project / "heat_fair_v2/generator/data_heat2d_gaussian.py"),
                           evaluator_sha=r.digest(Path(r.__file__)))
        marker = out / "split.json"
        if marker.exists() and r.read(marker) != frozen_data:
            raise RuntimeError("Frozen split changed")
        r.write(marker, frozen_data)
        if args.action == "original":
            weights = ROOT / "revision/original_heat"
            index = r.read(weights / "index.json")
            groups = r.read(ROOT / "heat_formal_v3_16/protocol.json")["groups"]
            expected = set(itertools.product(groups, range(292, 302)))
            jobs = index["jobs"]
            if len(jobs) != 90 or {(j["group"], j["seed"]) for j in jobs} != expected:
                raise RuntimeError("Incomplete original model archive")
            if index["preprocessing_sha"] != pre_sha or r.digest(ROOT / "results/test_runs.csv") != index["published_scores_sha"]:
                raise RuntimeError("Original model evidence changed")
            for job in jobs:
                path = weights / job["checkpoint"]
                if r.digest(path) != job["checkpoint_sha"]:
                    raise RuntimeError(f"Changed original checkpoint: {path}")
                job["checkpoint"] = str(path)
            frozen = dict(jobs=jobs, protocol=protocol, split=r.object_hash(frozen_data),
                          index_sha=r.digest(weights / "index.json"))
            r.write(out / "evaluation_inputs.json", frozen)
        data = r.generate(project, out, frozen_data)
        if args.action == "data":
            return
        rows = r.evaluate(project, out, frozen, data)
        stats = r.imports(project)[1]
        groups = sorted({row["group"] for row in rows})
        by_group = {}
        errors = {}
        table = []
        for group in groups:
            subset = sorted((row for row in rows if row["group"] == group), key=lambda row: row["seed"])
            values = [row["test_relative_l2"] for row in subset]
            errors[group] = values
            table.append(dict(group=group, mean=statistics.mean(values), sd=statistics.stdev(values)))
            vectors = [row["per_solution_relative_l2"] for row in subset]
            by_group[group] = list(map(statistics.mean, zip(*vectors)))
        r.csv_write(out / "analysis/main_table.csv", table)
        contrasts = [math.log(a/b)-math.log(c/d) for a,b,c,d in zip(
            errors["full_muon_adam"], errors["full_adam"], errors["noflow_muon_adam"], errors["noflow_adam"])]
        r.write(out / "analysis/primary_I.json", stats.contrast_summary(contrasts))
        from scipy.stats import t
        pairs = []
        comparisons = math.comb(len(groups), 2)
        for a,b in itertools.combinations(groups, 2):
            d = [x-y for x,y in zip(by_group[a],by_group[b])]
            mean = statistics.mean(d)
            hw = t.ppf(1-.05/(2*comparisons), len(d)-1)*statistics.stdev(d)/math.sqrt(len(d))
            pairs.append(dict(first=a, second=b, mean_difference=mean,
                              simultaneous_low=mean-hw, simultaneous_high=mean+hw))
        r.write(out / "analysis/input_paired_ranking.json", pairs)
        r.export(out)


if __name__ == "__main__":
    main()
