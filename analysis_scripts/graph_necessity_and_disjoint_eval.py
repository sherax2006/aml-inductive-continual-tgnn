"""RTM REQ-5.35: (a) proper cold-start-vs-validation comparison on DISJOINT node sets, and
(b) graph-necessity checks. No training -- evaluates existing checkpoints.

Why (a): task_matrix.py's `_build_val_splits` scores a two-way 20% validation set, but with
cold-start enabled training uses a three-way (train / val / cold) split; the cold-start set
turns out to lie entirely inside that two-way set (75% of it), so "normal vs cold-start" from
the matrices compares overlapping nodes. Here "val" = the training-time three-way val split
(`stratified_train_val_coldstart_split`), disjoint from cold-start by construction.

Why (b): (i) score each checkpoint with ALL edges removed at test time; (ii) a graph-free random
forest per task on the identical train/val/cold splits.

Run from the project root:  python3 analysis_scripts/graph_necessity_and_disjoint_eval.py
Results are cached incrementally in analysis_scripts/graph_necessity_results.json (safe to re-run).
"""
import json, statistics, sys, time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from sklearn.ensemble import RandomForestClassifier

from aml_tgnn.data.elliptic_dataset import EllipticTemporalDataset
from aml_tgnn.data.split import stratified_train_val_coldstart_split
from aml_tgnn.infer import load_model_from_checkpoint
from aml_tgnn.task_matrix import compute_summary_metrics, discover_task_checkpoints
from aml_tgnn.utils.metrics import binary_classification_metrics, run_eval

CK = ROOT_DIR / "Checkpoints" / "Elliptic"
OUT = Path(__file__).resolve().parent / "graph_necessity_results.json"
STRATS = ["ewc_replay", "gem_twp", "agem_twp", "mas_twp"]
RUNS = [f"my_{s}_cs_seed{k}" for s in STRATS for k in range(1, 6)]

t0 = time.time()
first = discover_task_checkpoints(CK / RUNS[0])
order = sorted(first)
T = len(order)
cfg0 = torch.load(first[order[0]], map_location="cpu", weights_only=False)["cfg"]
ds = EllipticTemporalDataset(str(ROOT_DIR / "elliptic_bitcoin_dataset"), edge_in_dim=cfg0.backbone.edge_in_dim,
                             time_steps_per_task=cfg0.elliptic_time_steps_per_task)
splits = {}
for tid in order:
    d = ds.get_task(tid)
    tr, va, co = stratified_train_val_coldstart_split(
        ds.labeled_seed_nodes(tid), d.y, cfg0.train.val_ratio, cfg0.train.cold_start_ratio, cfg0.train.split_seed, tid)
    splits[tid] = {"train": tr, "val": va, "cold": co}
print(f"dataset loaded, order={order} ({time.time()-t0:.0f}s)", flush=True)

results = json.loads(OUT.read_text()) if OUT.exists() else {"runs": {}, "rf": None}


def strip_edges(data):
    e = data.clone()
    e.edge_index = torch.empty(2, 0, dtype=torch.long)
    e.edge_attr = torch.zeros(0, data.edge_attr.size(1))
    e.edge_sign = torch.zeros(0, dtype=torch.long)
    e.edge_time = torch.zeros(0)
    return e


# ---------- graph-free random forest, identical train/val/cold splits ----------
if results.get("rf") is None:
    rf = {"val_acc": [], "val_f1": [], "cold_acc": [], "cold_f1": []}
    for tid in order:
        d = ds.get_task(tid)
        X, y = d.x.numpy(), d.y.numpy()
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=8, random_state=0)
        clf.fit(X[splits[tid]["train"].numpy()], y[splits[tid]["train"].numpy()])
        for name in ("val", "cold"):
            idx = splits[tid][name]
            m = binary_classification_metrics(d.y[idx], torch.tensor(clf.predict(X[idx.numpy()])))
            rf[f"{name}_acc"].append(m["accuracy"])
            rf[f"{name}_f1"].append(m["f1"])
    results["rf"] = rf
    OUT.write_text(json.dumps(results, indent=2))
    print("RF per-task val F1:", [round(x, 3) for x in rf["val_f1"]], "| mean val F1", round(statistics.mean(rf["val_f1"]), 4),
          "| mean cold F1", round(statistics.mean(rf["cold_f1"]), 4), flush=True)


def mean(xs):
    return sum(xs) / len(xs)


for run in RUNS:
    if run in results["runs"]:
        continue
    cks = discover_task_checkpoints(CK / run)
    assert sorted(cks) == order
    M = {name: {m: {i: {} for i in order} for m in ("accuracy", "f1")} for name in ("val", "cold")}
    E = {name: {m: {} for m in ("accuracy", "f1")} for name in ("val", "cold")}   # edges removed: diag
    EF = {name: {m: {} for m in ("accuracy", "f1")} for name in ("val", "cold")}  # edges removed: final checkpoint row
    for pos, i in enumerate(order):
        model, cfg = load_model_from_checkpoint(str(cks[i]), device="cpu")
        for j in order:
            d = ds.get_task(j)
            for name in ("val", "cold"):
                r = run_eval(model, d, splits[j][name], cfg, "cpu")
                for m in ("accuracy", "f1"):
                    M[name][m][i][j] = r[m]
            if j == i or i == order[-1]:
                dn = strip_edges(d)
                for name in ("val", "cold"):
                    r = run_eval(model, dn, splits[j][name], cfg, "cpu")
                    for m in ("accuracy", "f1"):
                        if j == i:
                            E[name][m][j] = r[m]
                        if i == order[-1]:
                            EF[name][m][j] = r[m]
    rec = {}
    for name in ("val", "cold"):
        acc_s = compute_summary_metrics(M[name]["accuracy"], order)
        f1_s = compute_summary_metrics(M[name]["f1"], order)
        rec[name] = {
            "ACC": acc_s["ACC"], "BWT": acc_s["BWT"], "F1": f1_s["ACC"], "F1_BWT": f1_s["BWT"],
            "diag_acc": mean([M[name]["accuracy"][i][i] for i in order]),
            "diag_f1": mean([M[name]["f1"][i][i] for i in order]),
            "diag_f1_per_task": [M[name]["f1"][i][i] for i in order],
            "noedge_diag_acc": mean(list(E[name]["accuracy"].values())),
            "noedge_diag_f1": mean(list(E[name]["f1"].values())),
            "noedge_diag_f1_per_task": [E[name]["f1"][i] for i in order],
            "final_row_f1": mean([M[name]["f1"][order[-1]][j] for j in order]),
            "noedge_final_row_acc": mean(list(EF[name]["accuracy"].values())),
            "noedge_final_row_f1": mean(list(EF[name]["f1"].values())),
            "final_row_acc": mean([M[name]["accuracy"][order[-1]][j] for j in order]),
        }
    results["runs"][run] = rec
    OUT.write_text(json.dumps(results, indent=2))
    print(f"{run}: val diag F1 {rec['val']['diag_f1']:.4f} -> no-edge {rec['val']['noedge_diag_f1']:.4f} | "
          f"cold diag F1 {rec['cold']['diag_f1']:.4f} -> no-edge {rec['cold']['noedge_diag_f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)

print("ALL DONE", flush=True)
