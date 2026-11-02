"""RTM REQ-5.36: one-shot preprocessing of the real IBM AML CSV into the
per-task cache + manifest IBMAMLTemporalDataset then loads directly.

Run from the project root:  python3 analysis_scripts/preprocess_ibm_aml.py [csv_name] [n_tasks]
"""
import sys, time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from aml_tgnn.data.ibm_aml_dataset import IBMAMLTemporalDataset

csv_name = sys.argv[1] if len(sys.argv) > 1 else "HI-Small_Trans.csv"
n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 7

t0 = time.time()
ds = IBMAMLTemporalDataset(str(ROOT_DIR / csv_name), edge_in_dim=16, node_in_dim=8, n_tasks=n_tasks)
print(f"\npreprocessing took {time.time()-t0:.1f}s", flush=True)

print(f"\n{len(ds.task_ids)} tasks: {ds.task_ids}\n")
for t in ds.task_ids:
    d = ds.get_task(t)
    n_illicit = int((d.y == 1).sum())
    print(f"task {t}: nodes={d.num_nodes:>8,} edges={d.edge_index.size(1):>9,} "
          f"illicit_nodes={n_illicit:>5,} ({100*n_illicit/d.num_nodes:.3f}%)")

print("\nALL DONE", flush=True)
