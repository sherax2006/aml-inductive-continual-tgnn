"""RTM REQ-5.33: forward-transfer variants from existing checkpoints (no training).

Run from the project root:  python3 analysis_scripts/compute_fwt_variants.py
Writes analysis_scripts/fwt_variants_results.json (the numbers behind manuscript Table 3's last four rows
and Section 5.5). Needs the 20-run campaign (my_<strategy>_cs_seed1-5) and 9 ablation runs on disk.
"""
import json, statistics, sys, time
ROOT_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pathlib import Path
import torch
from aml_tgnn.data.elliptic_dataset import EllipticTemporalDataset
from aml_tgnn.infer import load_model_from_checkpoint
from aml_tgnn.models.continual_wrapper import ContinualTGNN
from aml_tgnn.task_matrix import _build_val_splits, discover_task_checkpoints
from aml_tgnn.utils.metrics import run_eval
from aml_tgnn.utils.seeding import set_seed

ROOT = ROOT_DIR
CK = ROOT / "Checkpoints" / "Elliptic"
OUT = Path(__file__).resolve().parent / "fwt_variants_results.json"
K_BASELINE = 8

# group -> (architecture key for the random-init baseline, run dirs)
GROUPS = {}
for s in ["ewc_replay", "gem_twp", "agem_twp", "mas_twp"]:
    GROUPS[s] = ("full", [f"my_{s}_cs_seed{k}" for k in range(1, 6)])
GROUPS["abl_full(3 seeds)"] = ("full", [f"my_gem_twp_cs_seed{k}" for k in range(1, 4)])
for v in ["nobidir", "nolstm", "noattn"]:
    GROUPS[f"abl_{v}"] = (v, [f"my_gem_twp_ablation_{v}_seed{k}" for k in range(1, 4)])
ARCH_REP_RUN = {"full": "my_gem_twp_cs_seed1", "nobidir": "my_gem_twp_ablation_nobidir_seed1",
                "nolstm": "my_gem_twp_ablation_nolstm_seed1", "noattn": "my_gem_twp_ablation_noattn_seed1"}

t0 = time.time()
first_ck = discover_task_checkpoints(CK / "my_gem_twp_cs_seed1")
order = sorted(first_ck)
T = len(order)
first_cfg = torch.load(first_ck[order[0]], map_location="cpu", weights_only=False)["cfg"]
ds = EllipticTemporalDataset(str(ROOT / "elliptic_bitcoin_dataset"), edge_in_dim=first_cfg.backbone.edge_in_dim,
                             time_steps_per_task=first_cfg.elliptic_time_steps_per_task)
val_splits = _build_val_splits(ds, order, first_cfg.train.val_ratio, first_cfg.train.split_seed)
print(f"dataset loaded, task order {order}, T={T}", flush=True)


def ev(model, cfg, j):
    r = run_eval(model, ds.get_task(order[j]), val_splits[order[j]], cfg, "cpu")
    return r


# ---- random-init baselines b_j, per architecture, averaged over K random initializations ----
baselines = {}
for arch, rep in ARCH_REP_RUN.items():
    ck = discover_task_checkpoints(CK / rep)
    cfg = torch.load(ck[order[0]], map_location="cpu", weights_only=False)["cfg"]
    accs = [[] for _ in range(T)]
    f1s = [[] for _ in range(T)]
    for k in range(K_BASELINE):
        set_seed(9000 + k)
        m = ContinualTGNN(cfg)
        m.eval()
        for j in range(T):
            r = ev(m, cfg, j)
            accs[j].append(r["accuracy"])
            f1s[j].append(r["f1"])
    baselines[arch] = {"acc": [statistics.mean(a) for a in accs], "f1": [statistics.mean(f) for f in f1s],
                       "acc_sd_over_inits": [statistics.pstdev(a) for a in accs],
                       "f1_sd_over_inits": [statistics.pstdev(f) for f in f1s]}
    print(f"baseline[{arch}] acc per task={[round(x,3) for x in baselines[arch]['acc']]} "
          f"f1 per task={[round(x,3) for x in baselines[arch]['f1']]}  ({time.time()-t0:.0f}s)", flush=True)

# ---- per-run zero-shot / transfer quantities ----
results = {}
for gname, (arch, runs) in GROUPS.items():
    results[gname] = {}
    for run in runs:
        cks = discover_task_checkpoints(CK / run)
        assert sorted(cks) == order, (run, sorted(cks))
        R = {}
        for i in range(T):
            model, cfg = load_model_from_checkpoint(str(cks[order[i]]), device="cpu")
            js = set()
            if i == 0:
                js |= set(range(T))
            else:
                js.add(i)
                if i + 1 < T:
                    js.add(i + 1)
            for j in js:
                R[(i, j)] = ev(model, cfg, j)
        b = baselines[arch]
        pairs = range(1, T)
        one_acc = [R[(i - 1, i)]["accuracy"] for i in pairs]
        one_f1 = [R[(i - 1, i)]["f1"] for i in pairs]
        rec = {
            "onestep_acc": statistics.mean(one_acc),
            "onestep_f1": statistics.mean(one_f1),
            "onestep_f1_per_pair": one_f1,
            "std_fwt_acc": statistics.mean(a - b["acc"][i] for a, i in zip(one_acc, pairs)),
            "std_fwt_f1": statistics.mean(f - b["f1"][i] for f, i in zip(one_f1, pairs)),
            "racla_fwt_acc": statistics.mean(R[(i, i)]["accuracy"] - R[(0, i)]["accuracy"] for i in pairs),
            "racla_fwt_f1": statistics.mean(R[(i, i)]["f1"] - R[(0, i)]["f1"] for i in pairs),
            "A_1i_acc": statistics.mean(R[(0, i)]["accuracy"] for i in pairs),
            "A_ii_acc": statistics.mean(R[(i, i)]["accuracy"] for i in pairs),
            "alllicit_acc": statistics.mean(R[(0, i)]["support_neg"] / R[(0, i)]["n"] for i in pairs),
            "baseline_acc": statistics.mean(b["acc"][i] for i in pairs),
            "baseline_f1": statistics.mean(b["f1"][i] for i in pairs),
            "indiag_f1": statistics.mean(R[(i, i)]["f1"] for i in pairs),
        }
        results[gname][run] = rec
        print(f"[{gname}] {run}: onestep acc={rec['onestep_acc']:.4f} f1={rec['onestep_f1']:.4f} | "
              f"stdFWT acc={rec['std_fwt_acc']:+.4f} f1={rec['std_fwt_f1']:+.4f} | RACLA-FWT acc={rec['racla_fwt_acc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

OUT.write_text(json.dumps({"baselines": baselines, "results": results}, indent=2))

print("\n\n=== SUMMARY (mean +/- pstdev across seeds) ===", flush=True)
keys = ["onestep_acc", "onestep_f1", "std_fwt_acc", "std_fwt_f1", "racla_fwt_acc", "racla_fwt_f1",
        "A_1i_acc", "A_ii_acc", "alllicit_acc", "baseline_acc", "baseline_f1", "indiag_f1"]
for gname in GROUPS:
    print(f"\n--- {gname} (n={len(results[gname])}) ---")
    for k in keys:
        vals = [results[gname][r][k] for r in results[gname]]
        print(f"  {k}: {statistics.mean(vals):.4f} +/- {statistics.pstdev(vals):.4f}")
    pp = [results[gname][r]["onestep_f1_per_pair"] for r in results[gname]]
    print("  onestep_f1 per pair (task2..7):", [round(statistics.mean(p[i] for p in pp), 3) for i in range(T - 1)])
print("\nALL DONE", flush=True)
