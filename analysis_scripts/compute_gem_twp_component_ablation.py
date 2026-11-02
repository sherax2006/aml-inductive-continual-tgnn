"""RTM REQ-5.39: GEM-vs-TWP component ablation table -- isolates how much of
"gem_twp"'s result comes from GEM's gradient projection versus TWP's regularization
penalty, alongside (not replacing) the existing backbone-component ablation (Table 7).

Four arms (user-requested framing):
  Full      -- both GEM and TWP active (the unmodified "gem_twp" strategy; reuses the
               already-trained my_gem_twp_cs_seed{k} checkpoints, no retraining needed).
  GEM only  -- --no-twp   (cfg.disable_twp=True, cfg.disable_gem=False)
  TWP only  -- --no-gem   (cfg.disable_gem=True, cfg.disable_twp=False)
  Neither   -- --no-gem --no-twp (plain per-task fine-tuning, no continual mechanism)

Same methodology as Table 7 (analysis_scripts/compute_ablation_table.py): reuses
compute_task_matrix/compute_summary_metrics directly, not a new formula. "ACC"/"BWT" are
the accuracy-matrix ACC/BWT formulas; "F1"/"Cold-start F1" reuse the same ACC-formula
(mean final-checkpoint value) applied to the F1 matrix and the cold-start F1 matrix,
since cold_start_ratio=0.15 is enabled on every run below.

Run from the project root:  python3 analysis_scripts/compute_gem_twp_component_ablation.py
Writes analysis_scripts/gem_twp_component_ablation_results.json.

Needs, per seed in SEEDS: my_gem_twp_cs_seed{k} (already exists), and
my_gem_twp_component_{gemonly,twponly,neither}_seed{k} (train these first --
see the companion run commands this script's own docstring / the chat that produced it).
"""
import json
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from aml_tgnn.data.elliptic_dataset import EllipticTemporalDataset
from aml_tgnn.task_matrix import compute_summary_metrics, compute_task_matrix

ROOT = ROOT_DIR
CK = ROOT / "Checkpoints" / "Elliptic"
OUT = Path(__file__).resolve().parent / "gem_twp_component_ablation_results.json"

ARMS = {
    "Full (GEM+TWP)": "my_gem_twp_cs_seed{seed}",
    "GEM only": "my_gem_twp_component_gemonly_seed{seed}",
    "TWP only": "my_gem_twp_component_twponly_seed{seed}",
    "Neither": "my_gem_twp_component_neither_seed{seed}",
}
SEEDS = [1, 2, 3, 4, 5]  # extended to n=5 to match the rest of the paper's convention

first_cfg = torch.load(
    CK / "my_gem_twp_cs_seed1" / "task_43_complete.pt", map_location="cpu", weights_only=False,
)["cfg"]
ds = EllipticTemporalDataset(
    str(ROOT / "elliptic_bitcoin_dataset"), edge_in_dim=first_cfg.backbone.edge_in_dim,
    time_steps_per_task=first_cfg.elliptic_time_steps_per_task,
)
print("dataset loaded", flush=True)

results = {}
for arm_name, pattern in ARMS.items():
    per_seed = {"ACC": [], "BWT": [], "F1": [], "Cold-start F1": []}
    for seed in SEEDS:
        run_dir = CK / pattern.format(seed=seed)
        if not run_dir.exists():
            print(f"skipping [{arm_name}] seed {seed}: {run_dir} not found yet", flush=True)
            continue
        matrices, task_order = compute_task_matrix(ds, run_dir, device="cpu", metric_names=("accuracy", "f1"))
        acc_summary = compute_summary_metrics(matrices["disjoint_val_accuracy"], task_order)
        f1_summary = compute_summary_metrics(matrices["disjoint_val_f1"], task_order)
        coldf1_summary = compute_summary_metrics(matrices["cold_f1"], task_order)
        per_seed["ACC"].append(acc_summary["ACC"])
        per_seed["BWT"].append(acc_summary["BWT"])
        per_seed["F1"].append(f1_summary["ACC"])
        per_seed["Cold-start F1"].append(coldf1_summary["ACC"])
        print(f"[{arm_name}] seed {seed}: ACC={per_seed['ACC'][-1]:.4f} BWT={per_seed['BWT'][-1]:+.4f} "
              f"F1={per_seed['F1'][-1]:.4f} ColdF1={per_seed['Cold-start F1'][-1]:.4f}", flush=True)
    if not per_seed["ACC"]:
        print(f"--- {arm_name}: no completed runs found, skipped ---")
        continue
    results[arm_name] = {
        k: {"mean": statistics.mean(v), "std": statistics.pstdev(v), "n": len(v), "values": v}
        for k, v in per_seed.items()
    }
    r = results[arm_name]
    print(f"--- {arm_name} (n={r['ACC']['n']}): "
          f"ACC={r['ACC']['mean']:.4f}+/-{r['ACC']['std']:.4f}  "
          f"BWT={r['BWT']['mean']:+.4f}+/-{r['BWT']['std']:.4f}  "
          f"F1={r['F1']['mean']:.4f}+/-{r['F1']['std']:.4f}  "
          f"ColdF1={r['Cold-start F1']['mean']:.4f}+/-{r['Cold-start F1']['std']:.4f}", flush=True)

OUT.write_text(json.dumps(results, indent=2))
print(f"\nwritten -> {OUT}")
