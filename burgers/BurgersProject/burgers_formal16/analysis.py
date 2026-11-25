"""Frozen log-ratio analysis, failures and validation-only target-time summaries."""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import rankdata, t as student_t
import control as k


def signed_rank_exact(values):
    d = np.asarray(values, dtype=float)
    d = d[d != 0]
    if not len(d):
        return dict(statistic=0., pvalue=1., nonzero_n=0, method="enumerated signs; all-zero convention")
    if len(d) > 10 or not np.isfinite(d).all():
        raise ValueError("This prespecified exact procedure expects <=10 finite seed contrasts.")
    ranks = rankdata(np.abs(d), method="average")
    positive = float(ranks[d > 0].sum())
    total = float(ranks.sum())
    observed = min(positive, total - positive)
    extreme = 0
    for signs in itertools.product([0, 1], repeat=len(d)):
        pos = float(np.dot(signs, ranks))
        extreme += min(pos, total - pos) <= observed + 1e-12
    return dict(statistic=observed, pvalue=extreme / 2 ** len(d), nonzero_n=len(d),
                method="all sign assignments; absolute midranks, exact zeros removed")


def contrast_summary(values, planned_n=10):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("No silent nonfinite omission or artificial log clipping.")
    n = len(a)
    mean = float(a.mean()) if n else None
    sd = float(a.std(ddof=1)) if n > 1 else None
    pvalue, low, high, stat = None, None, None, None
    if n > 1 and sd > 0:
        se = sd / math.sqrt(n)
        stat = mean / se
        pvalue = float(2 * student_t.sf(abs(stat), n - 1))
        half = float(student_t.ppf(.975, n - 1)) * se
        low, high = mean - half, mean + half
    complete = n == planned_n
    return dict(n=n, planned_n=planned_n, mean=mean, sample_sd=sd,
                analysis_status="planned_complete" if complete else "exploratory_available_seed_subset",
                t_statistic=stat, paired_t_p=pvalue, t_ci_low=low, t_ci_high=high,
                t_status="zero_variance_or_insufficient_n" if pvalue is None else "computed",
                wilcoxon=signed_rank_exact(a) if n else None,
                confirmatory_t_p=pvalue if complete else None)


def analyze(out, p, rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_key = {(r["seed"], r["group"]): r for r in rows}
    seeds = p["formal"]["seeds"]
    tables, targets = [], []
    for group in p["groups"]:
        local = [by_key[s, group] for s in seeds if (s, group) in by_key]
        valid = [r for r in local if r["status"] == "completed" and r.get("test_relative_l2") is not None]
        errors = [r["test_relative_l2"] for r in valid]
        all10 = len(errors) == 10
        tables.append(dict(group=group, planned_n=10, n_success=len(valid),
            n_diverged=sum(r["status"] == "diverged" for r in local),
            n_invalid_test=sum(r["status"] == "test_nonfinite" for r in local), n_not_evaluated=10 - len(local),
            mean_test_error=statistics.mean(errors) if all10 else None,
            sample_sd=statistics.stdev(errors) if all10 else None,
            successful_only_mean=statistics.mean(errors) if errors else None,
            successful_only_sd=statistics.stdev(errors) if len(errors) > 1 else None,
            full10_summary_status="available" if all10 else "unavailable; successful-only summary is conditional",
            parameters=local[0]["parameters"] if local else None,
            mean_training_seconds=statistics.mean(r["training_seconds"] for r in local) if local else None,
            mean_active_wall_seconds=statistics.mean(r["wall_seconds"] for r in local) if local else None))
        for r in local:
            targets += [dict(group=group, seed=r["seed"], **e) for e in r["targets"]]
    k.csv_write(out / "main_table.csv", tables)
    k.csv_write(out / "test_runs.csv", rows)
    k.csv_write(out / "target_time_runs.csv", targets)
    time_summary = []
    for g in p["groups"]:
        for threshold in p["time_to_target"]["thresholds"]:
            rs = [r for r in targets if r["group"] == g and r["threshold"] == threshold]
            reached = [r for r in rs if r["event"]]
            time_summary.append(dict(group=g, threshold=threshold, planned_n=10,
                n_reached=len(reached), n_diverged=sum(r["run_status"] == "diverged" for r in rs),
                n_success_not_reached=sum(not r["event"] and r["run_status"] == "completed" for r in rs),
                n_not_evaluated=10-len(rs),
                conditional_median_wall_seconds=statistics.median(r["wall_seconds"] for r in reached) if reached else None,
                warning="Conditional on crossing; inspect censor/failure counts before comparing times."))
    k.csv_write(out / "target_time_summary.csv", time_summary)

    def error(seed, group):
        r = by_key.get((seed, group))
        e = r.get("test_relative_l2") if r and r["status"] == "completed" else None
        return e if e is not None and math.isfinite(e) and e > 0 else None

    contrast_defs = [(a, a + "_adam", a + "_muon_adam") for a in ["deeponet", "mionet", "full", "noflow"]]
    contrast_defs += [("deeponet_output_only", "deeponet_adam", "deeponet_output_muon_adam")]
    effects, raw, per_seed = [], {}, {s: dict(seed=s) for s in seeds}
    for name, adam, muon in contrast_defs:
        log_values, absolute, available = [], [], []
        raw[name] = {}
        for seed in seeds:
            a, b = error(seed, adam), error(seed, muon)
            if a is None or b is None:
                per_seed[seed]["log_delta_" + name] = None
                per_seed[seed]["absolute_benefit_" + name] = None
                continue
            delta, benefit = math.log(b / a), a - b
            raw[name][seed] = (delta, benefit)
            available.append(seed)
            log_values.append(delta)
            absolute.append(benefit)
            per_seed[seed]["log_delta_" + name] = delta
            per_seed[seed]["absolute_benefit_" + name] = benefit
        stats = contrast_summary(log_values)
        stats["confirmatory_t_p"]=None  # Only the prespecified I test is primary.
        effects.append(dict(contrast=name, scale="natural_log_ratio", role="secondary", available_seeds=available,
                            **stats, geometric_error_ratio=math.exp(stats["mean"]) if stats["mean"] is not None else None,
                            percent_reduction=100*(1-math.exp(stats["mean"])) if stats["mean"] is not None else None))
        abs_stats=contrast_summary(absolute);abs_stats["confirmatory_t_p"]=None
        effects.append(dict(contrast=name, scale="Adam_minus_Muon_absolute", role="secondary",
                            available_seeds=available, **abs_stats))
    i_values, absolute_i, i_seeds = [], [], []
    for seed in seeds:
        if seed in raw["full"] and seed in raw["noflow"]:
            value = raw["full"][seed][0] - raw["noflow"][seed][0]
            benefit = raw["full"][seed][1] - raw["noflow"][seed][1]
            i_values.append(value)
            absolute_i.append(benefit)
            i_seeds.append(seed)
            per_seed[seed].update(I_log=value, I_absolute=benefit)
        else:
            per_seed[seed].update(I_log=None, I_absolute=None)
    primary = dict(contrast="I", scale="natural_log_ratio", role="unique_primary", available_seeds=i_seeds,
                   missing_seeds=[s for s in seeds if s not in i_seeds], **contrast_summary(i_values),
                   definition="ln(E_full_M/E_full_A) - ln(E_noflow_M/E_noflow_A)",
                   direction="negative: greater proportional Muon benefit with Full flow",
                   secondary_test_role="Wilcoxon sensitivity; do not select the smaller p")
    abs_stats=contrast_summary(absolute_i);abs_stats["confirmatory_t_p"]=None
    effects += [primary, dict(contrast="I", scale="Adam_minus_Muon_absolute", role="secondary",
                             available_seeds=i_seeds, **abs_stats)]
    k.write(out / "primary_I.json", primary)
    k.write(out / "paired_effects.json", effects)
    k.csv_write(out / "paired_effects.csv", effects)
    k.csv_write(out / "seed_effects.csv", list(per_seed.values()))
    k.write(out / "analysis_status.json", dict(protocol_id=p["protocol_id"],
             primary_I_status=primary["analysis_status"], primary_I_n=len(i_seeds), planned_n=10,
             complete_seed_blocks=sorted({r["seed"] for r in rows}),
             historical_test_exposure_disclosed=True))
    return primary
