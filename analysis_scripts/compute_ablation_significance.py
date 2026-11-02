"""Paired t-test (matched by seed) between the GEM-active and GEM-inactive
arms of the component ablation (Section 5.6, Table 7 of the manuscript).
Reuses gem_twp_component_ablation_results.json's existing 'GEM only' and
'Neither' entries -- no retraining needed. Both arms' per-seed value lists
are in the same fixed seed order ([1, 2, 3, 4, 5], see SEEDS in
compute_gem_twp_component_ablation.py), so index-wise pairing is valid.
"""
import json
from pathlib import Path

from scipy import stats

RESULTS_PATH = Path(__file__).parent / "gem_twp_component_ablation_results.json"


def main():
    d = json.loads(RESULTS_PATH.read_text())
    active = d["GEM only"]
    inactive = d["Neither"]
    for metric in ["ACC", "BWT", "F1", "Cold-start F1"]:
        a = active[metric]["values"]
        b = inactive[metric]["values"]
        t, p = stats.ttest_rel(a, b)
        mean_diff = sum(x - y for x, y in zip(a, b)) / len(a)
        print(f"{metric}: t({len(a) - 1}) = {t:.2f}, p = {p:.6f}, mean diff = {mean_diff:+.4f}")


if __name__ == "__main__":
    main()
