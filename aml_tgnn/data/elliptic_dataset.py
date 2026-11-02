"""Elliptic Bitcoin dataset adapter (RTM REQ-3.1) -- node-feature-heavy case.

Expected raw files (standard Kaggle/Elliptic release layout), all under `root/`:
  - elliptic_txs_features.csv : no header; col0=txId, col1=time_step (1..49),
                                 col2..167=166 node features (93 local + 72 aggregated).
  - elliptic_txs_classes.csv  : header txId,class ; class in {'1' illicit, '2' licit, 'unknown'}.
  - elliptic_txs_edgelist.csv : header txId1,txId2 (directed, txId1 -> txId2).

Elliptic has NO edge features and NO per-transaction timestamp -- only a per-node
`time_step`. See REQ-D5 / ARCHITECTURE.md for how the backbone is adapted to that:
edge_attr is replaced by a small learned "edge-type" embedding shared across all
edges (there is nothing else to put there), and edge_time is proxied by the
target node's time_step, since Elliptic gives no finer-grained ordering.

`time_steps_per_task` (RTM REQ-5.9) controls task granularity: the default of 1
matches the original 49-task setup (one continual-learning task per raw time
step). Setting it to e.g. 7 groups consecutive time steps into one task (49 / 7
= exactly 7 equal-sized tasks) -- coarser tasks pool more labeled nodes per task,
which matters because Elliptic's illicit class is thin enough per time step that
a 49-task run's per-task validation metrics can be dominated by sampling noise
(some individual time steps have fewer than 5 illicit nodes total). A task's ID
is the *starting* time step of its group, so `time_steps_per_task=1` reproduces
the exact old task numbering (1..49) rather than switching to 0-indexed groups.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch_geometric.data import Data


class EllipticTemporalDataset:
    def __init__(self, root: str, edge_in_dim: int = 8, time_steps_per_task: int = 1):
        if time_steps_per_task < 1:
            raise ValueError(f"time_steps_per_task must be >= 1, got {time_steps_per_task}")
        self.root = Path(root)
        self.edge_in_dim = edge_in_dim
        self.time_steps_per_task = time_steps_per_task
        self._tasks: Dict[int, Data] = {}
        self._load()

    def _load(self) -> None:
        feat_path = self.root / "elliptic_txs_features.csv"
        class_path = self.root / "elliptic_txs_classes.csv"
        edge_path = self.root / "elliptic_txs_edgelist.csv"

        features = pd.read_csv(feat_path, header=None)
        features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(features.shape[1] - 2)]

        classes = pd.read_csv(class_path)
        classes["label"] = classes["class"].map({"1": 1, "2": 0, "unknown": -1})

        edges = pd.read_csv(edge_path)

        merged = features.merge(classes[["txId", "label"]], on="txId", how="left")
        merged["label"] = merged["label"].fillna(-1).astype(int)

        tx_id_to_idx = {tx: i for i, tx in enumerate(merged["txId"].values)}
        node_feat_cols = [c for c in merged.columns if c.startswith("f")]

        x_all = torch.tensor(merged[node_feat_cols].values, dtype=torch.float)
        y_all = torch.tensor(merged["label"].values, dtype=torch.long)
        time_step_all = torch.tensor(merged["time_step"].values, dtype=torch.long)

        # Task id = the group's starting time step, so time_steps_per_task=1
        # reduces to task_id == time_step exactly (old numbering, unchanged).
        group_start_all = ((time_step_all - 1) // self.time_steps_per_task) * self.time_steps_per_task + 1

        src = edges["txId1"].map(tx_id_to_idx)
        dst = edges["txId2"].map(tx_id_to_idx)
        valid = src.notna() & dst.notna()
        src = torch.tensor(src[valid].values, dtype=torch.long)
        dst = torch.tensor(dst[valid].values, dtype=torch.long)
        edge_time_all = time_step_all[dst]  # proxy: target node's time_step

        for g in sorted(group_start_all.unique().tolist()):
            node_mask = group_start_all == g
            node_idx = node_mask.nonzero(as_tuple=False).view(-1)
            local_map = {int(gi.item()): i for i, gi in enumerate(node_idx)}

            # both endpoints must fall in the same task group, not just the same raw time step
            edge_mask = (group_start_all[src] == g) & (group_start_all[dst] == g)
            e_src = src[edge_mask]
            e_dst = dst[edge_mask]
            local_src = torch.tensor([local_map[int(s.item())] for s in e_src], dtype=torch.long)
            local_dst = torch.tensor([local_map[int(d.item())] for d in e_dst], dtype=torch.long)
            e_time = edge_time_all[edge_mask].float()

            n = node_idx.size(0)
            e = local_src.size(0)
            data = Data(
                x=x_all[node_idx],
                edge_index=torch.stack([local_src, local_dst], dim=0),
                edge_attr=torch.zeros(e, self.edge_in_dim),  # no native edge features; see module docstring
                edge_sign=torch.zeros(e, dtype=torch.long),  # not applicable for Elliptic; constant class 0
                edge_time=e_time,
                y=y_all[node_idx],
                num_nodes=n,
            )
            self._tasks[int(g)] = data

    @property
    def task_ids(self) -> List[int]:
        return sorted(self._tasks.keys())

    @property
    def node_in_dim(self) -> int:
        return next(iter(self._tasks.values())).x.size(1)

    def get_task(self, task_id: int) -> Data:
        return self._tasks[task_id]

    def labeled_seed_nodes(self, task_id: int) -> torch.Tensor:
        data = self._tasks[task_id]
        return (data.y != -1).nonzero(as_tuple=False).view(-1)
