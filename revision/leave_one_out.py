"""Leave one training seed out of the published Burgers log interaction."""
import csv
import math
from pathlib import Path
import statistics
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "burgers/results/seed_effects.csv").open() as f:
        rows = [(int(r["seed"]),float(r["I_log"])) for r in csv.DictReader(f)]
    if sorted(seed for seed,_ in rows) != list(range(292,302)):
        raise ValueError("Expected ten distinct formal training seeds")
    with (ROOT / "revision/burgers_leave_one_out.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["omitted_seed","n","mean_log","ratio","ratio_low_95","ratio_high_95"])
        for omitted,_ in rows:
            values = [v for s,v in rows if s != omitted]
            mean = statistics.mean(values)
            half = t.ppf(.975,8)*statistics.stdev(values)/3
            writer.writerow([omitted,9,mean,math.exp(mean),math.exp(mean-half),math.exp(mean+half)])


if __name__ == "__main__":
    main()
