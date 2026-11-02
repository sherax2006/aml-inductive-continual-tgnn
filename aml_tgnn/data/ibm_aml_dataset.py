"""IBM AML (fiat banking simulator) dataset adapter (RTM REQ-3.2 / RTM REQ-5.36)
-- edge-feature-heavy case.

Expected raw CSV (standard IBM/Kaggle AML release schema, e.g. `HI-Small_Trans.csv`,
~5.08M rows / 475MB for the "HI-Small" split), columns: Timestamp, From Bank,
Account, To Bank, Account.1 (pandas' own dedup-rename of the raw file's second
"Account" column), Amount Received, Receiving Currency, Amount Paid, Payment
Currency, Payment Format, Is Laundering. `Account` is the payer (edge source),
`Account.1` is the payee (edge target).

Unlike Elliptic, this dataset is edge-feature-heavy: transactions carry rich
tabular attributes (amounts, currencies, payment rail) while accounts carry
almost no native attributes -- indeed the reference Multi-GNN implementation
(github.com/IBM/Multi-GNN) uses an all-ones placeholder node feature and relies
entirely on edge-feature message passing; see REQ-D5 / ARCHITECTURE.md for why
this adapter instead derives cheap structural/behavioral node aggregates
(in/out-degree, mean amount sent/received) rather than a dense native vector or
a constant placeholder, since this project's backbone consumes per-node `x`
directly in `message()`.

Node identity, verified against the reference implementation's own convention
(github.com/IBM/Multi-GNN, `format_kaggle_files.py`: `fromAccIdStr = raw["From
Bank"] + raw["Account"]`) rather than assumed: an account number is NOT
globally unique across banks on this file (8 of 515,080 account strings occur
under more than one bank, confirmed directly against the real HI-Small file) --
so a node's key is the `(bank, account)` pair, not the account string alone.

Task framing (RTM REQ-5.37): `Is Laundering` is natively an EDGE (per-
transaction) label -- this dataset's own entities are accounts (nodes) and
transactions (edges), unlike Elliptic where each node *is* the labeled
transaction. The reference Multi-GNN implementation trains GIN/GAT/PNA/RGCN
as edge classifiers accordingly. `y` below is a node-level rollup (an
account is "illicit" for a task if it touches >=1 laundering-flagged
transaction) kept only for backward compatibility / a node-classification
fallback; `edge_y` is the raw, unrolled per-transaction label and is what
this project's own edge-classification mode (`ExperimentConfig.task_type =
"edge"`, `EdgeClassificationHead`, `build_temporal_link_loader`) actually
trains and evaluates against for IBM AML. Elliptic stays node classification
unchanged -- this dual-mode split is a property of the two datasets' own
native entity structure, not an inconsistency to reconcile.

Continual tasks are contiguous groups of calendar days, chosen to balance
transaction *volume* rather than day *count* (RTM REQ-5.36): per-day row
counts on HI-Small span three orders of magnitude (1,114,921 on day 1 down to
11 on day 18), so naive one-task-per-day windowing (this adapter's original
behavior) produces several near-empty tail tasks unusable for continual
training -- fewer than 400 transactions on 8 of 18 days. A greedy cumulative-
volume partition into `n_tasks` contiguous day-groups (default 7, matching
this project's own Elliptic T=7 standard for cross-dataset comparability)
avoids this while still being a genuine, non-overlapping, chronologically-
ordered temporal split -- task boundaries are real timestamps throughout, only
the *grouping* of days into a task is volume-aware rather than fixed-width.

Preprocessing a 5-million-row file is expensive enough to be worth caching
(RTM REQ-5.36): the original implementation's per-task node-id remapping used
a pure-Python list comprehension calling `.item()` on every edge endpoint --
correct, but O(n_edges) in the Python interpreter, which is minutes of wall
time on a task with over a million edges, repeated on *every* construction
since nothing was ever persisted to disk. This version (a) vectorizes that
remapping via a single numpy fancy-indexing lookup per task instead of a
Python-level loop, and (b) caches the fully-preprocessed per-task graphs to a
`.pt` file (plus a small human-readable `.manifest.json` summary) next to the
source CSV, keyed by every parameter that affects the result -- so a second
construction with the same `(csv_path, edge_in_dim, node_in_dim, n_tasks)`
loads directly from disk rather than re-parsing the raw file.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

CACHE_VERSION = 2  # v2 (RTM REQ-5.37): added edge_y (raw per-transaction label) to each task's Data


class IBMAMLTemporalDataset:
    def __init__(
        self,
        csv_path: str,
        edge_in_dim: int = 16,
        node_in_dim: int = 8,
        n_tasks: int = 7,
        cache_dir: Optional[str] = None,
        force_reprocess: bool = False,
    ):
        if n_tasks < 1:
            raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
        self.csv_path = Path(csv_path)
        self.edge_in_dim = edge_in_dim
        self.node_in_dim = node_in_dim
        self.n_tasks = n_tasks
        self.cache_dir = Path(cache_dir) if cache_dir else self.csv_path.parent
        self._tasks: Dict[int, Data] = {}

        cache_path, manifest_path = self._cache_paths()
        if not force_reprocess and cache_path.exists():
            self._load_from_cache(cache_path)
        else:
            self._load_from_raw_csv()
            self._save_to_cache(cache_path, manifest_path)

    # ------------------------------------------------------------------ cache

    def _cache_paths(self) -> "tuple[Path, Path]":
        """RTM REQ-5.36: cache key covers every parameter that changes the
        preprocessed result, so a stale cache from a different configuration
        is never silently reused -- a differently-parametrized run gets its
        own cache file rather than colliding with (or being masked by) one
        computed under different settings.
        """
        key = f"{self.csv_path.resolve()}|edge{self.edge_in_dim}|node{self.node_in_dim}|tasks{self.n_tasks}|v{CACHE_VERSION}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        stem = self.csv_path.stem
        # NOTE: Path.with_suffix() replaces whatever follows the LAST "." in the
        # name -- since `digest` is a bare hex string with no dot of its own, an
        # earlier version of this used base.with_suffix(".pt") and silently ate
        # the digest itself (treating it as an existing "extension" to replace),
        # caught by the synthetic adapter test's cache-file-name assertion.
        base_name = f"{stem}.ibmaml_cache.T{self.n_tasks}.{digest}"
        return self.cache_dir / f"{base_name}.pt", self.cache_dir / f"{base_name}.manifest.json"

    def _save_to_cache(self, cache_path: Path, manifest_path: Path) -> None:
        payload = {
            "cache_version": CACHE_VERSION,
            "csv_path": str(self.csv_path),
            "edge_in_dim": self.edge_in_dim,
            "node_in_dim": self.node_in_dim,
            "n_tasks": self.n_tasks,
            "tasks": self._tasks,
        }
        torch.save(payload, cache_path)

        manifest = {
            "source_csv": str(self.csv_path),
            "n_tasks": self.n_tasks,
            "task_ids": self.task_ids,
            "per_task": [
                {
                    "task_id": t,
                    "n_nodes": int(self._tasks[t].num_nodes),
                    "n_edges": int(self._tasks[t].edge_index.size(1)),
                    "n_labeled": int(self._tasks[t].y.numel()),
                    "n_illicit": int((self._tasks[t].y == 1).sum()),
                    "n_illicit_edges": int((self._tasks[t].edge_y == 1).sum()),
                    "start_timestamp": self._task_time_ranges[t][0],
                    "end_timestamp": self._task_time_ranges[t][1],
                }
                for t in self.task_ids
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"IBMAMLTemporalDataset: cached preprocessed data -> {cache_path} ({manifest_path.name} for a quick summary)")

    def _load_from_cache(self, cache_path: Path) -> None:
        payload = torch.load(cache_path, weights_only=False)
        if payload.get("cache_version") != CACHE_VERSION:
            raise RuntimeError(
                f"{cache_path} was built with cache_version={payload.get('cache_version')}, "
                f"this code expects {CACHE_VERSION} -- delete the cache file (or pass force_reprocess=True) to rebuild it."
            )
        self._tasks = payload["tasks"]
        print(f"IBMAMLTemporalDataset: loaded preprocessed data from cache -> {cache_path}")

    # -------------------------------------------------------------- raw load

    def _load_from_raw_csv(self) -> None:
        df = pd.read_csv(
            self.csv_path,
            dtype={
                "From Bank": str, "Account": str, "To Bank": str, "Account.1": str,
                "Receiving Currency": "category", "Payment Currency": "category", "Payment Format": "category",
                "Amount Received": np.float64, "Amount Paid": np.float64, "Is Laundering": np.int8,
            },
            parse_dates=["Timestamp"],
        )

        # Node identity is the (bank, account) pair, not the account string alone
        # (see module docstring -- 8/515,080 account strings collide across banks
        # on the real HI-Small file, verified directly, not assumed).
        src_key = df["From Bank"].astype(str) + "\x1f" + df["Account"].astype(str)
        dst_key = df["To Bank"].astype(str) + "\x1f" + df["Account.1"].astype(str)
        all_accounts = pd.unique(pd.concat([src_key, dst_key], ignore_index=True))
        acct_to_idx = pd.Series(np.arange(len(all_accounts), dtype=np.int64), index=all_accounts)
        global_src = acct_to_idx.reindex(src_key.values).to_numpy()
        global_dst = acct_to_idx.reindex(dst_key.values).to_numpy()
        n_accounts_global = len(all_accounts)

        fmt_codes = df["Payment Format"].cat.codes.to_numpy().astype(np.float32)
        pay_cur_codes = df["Payment Currency"].cat.codes.to_numpy().astype(np.float32)
        recv_cur_codes = df["Receiving Currency"].cat.codes.to_numpy().astype(np.float32)
        amount_paid_log = np.log1p(df["Amount Paid"].to_numpy(dtype=np.float64)).astype(np.float32)
        amount_recv_log = np.log1p(df["Amount Received"].to_numpy(dtype=np.float64)).astype(np.float32)
        edge_sign = (df["Payment Currency"].to_numpy() != df["Receiving Currency"].to_numpy()).astype(np.int64)
        edge_time = (df["Timestamp"].astype("int64").to_numpy() // 10**9).astype(np.float32)
        is_laundering = df["Is Laundering"].to_numpy(dtype=bool)

        task_id_per_row = self._assign_task_ids(df["Timestamp"])
        self._task_time_ranges: Dict[int, "tuple[str, str]"] = {}

        for t in np.unique(task_id_per_row):
            mask = task_id_per_row == t
            n_edges = int(mask.sum())

            g_src, g_dst = global_src[mask], global_dst[mask]
            nodes_in_task = np.unique(np.concatenate([g_src, g_dst]))
            n = len(nodes_in_task)

            # RTM REQ-5.36: vectorized global->local remap via a single numpy
            # fancy-indexing lookup, replacing the original's O(n_edges)
            # Python-level list comprehension (minutes per million-edge task).
            lookup = np.full(n_accounts_global, -1, dtype=np.int64)
            lookup[nodes_in_task] = np.arange(n, dtype=np.int64)
            local_src = torch.from_numpy(lookup[g_src])
            local_dst = torch.from_numpy(lookup[g_dst])
            assert (local_src >= 0).all() and (local_dst >= 0).all(), "unmapped node id -- lookup table bug"

            amount_paid = torch.from_numpy(amount_paid_log[mask])
            amount_recv = torch.from_numpy(amount_recv_log[mask])
            edge_feats = torch.stack([
                amount_paid, amount_recv,
                torch.from_numpy(fmt_codes[mask]),
                torch.from_numpy(pay_cur_codes[mask]),
                torch.from_numpy(recv_cur_codes[mask]),
            ], dim=1)
            edge_attr = self._pad_or_trim(edge_feats, self.edge_in_dim)

            out_deg = torch.zeros(n)
            in_deg = torch.zeros(n)
            out_deg.scatter_add_(0, local_src, torch.ones(n_edges))
            in_deg.scatter_add_(0, local_dst, torch.ones(n_edges))
            mean_sent = torch.zeros(n).scatter_add_(0, local_src, amount_paid) / out_deg.clamp(min=1)
            mean_recv = torch.zeros(n).scatter_add_(0, local_dst, amount_recv) / in_deg.clamp(min=1)
            node_feats = torch.stack([out_deg, in_deg, mean_sent, mean_recv], dim=1)
            x = self._pad_or_trim(node_feats, self.node_in_dim)

            y = torch.zeros(n, dtype=torch.long)
            flagged_mask = is_laundering[mask]
            if flagged_mask.any():
                flagged_local = torch.unique(torch.cat([
                    local_src[torch.from_numpy(flagged_mask)], local_dst[torch.from_numpy(flagged_mask)],
                ]))
                y[flagged_local] = 1

            ts_slice = df["Timestamp"].to_numpy()[mask]
            self._task_time_ranges[int(t)] = (str(ts_slice.min()), str(ts_slice.max()))

            # RTM REQ-5.37: the dataset's NATIVE label is per-transaction (edge),
            # not per-account (node) -- `y` above is a node-level rollup (an
            # account is "illicit" if it touches >=1 flagged transaction) kept
            # for backward compatibility with node-classification callers;
            # `edge_y` is the raw, unrolled label for edge-classification mode
            # (see module docstring's "Task framing" section), aligned 1:1 with
            # this task's own edge_index/edge_attr/edge_sign/edge_time.
            edge_y = torch.from_numpy(flagged_mask.astype("int64"))

            self._tasks[int(t)] = Data(
                x=x,
                edge_index=torch.stack([local_src, local_dst], dim=0),
                edge_attr=edge_attr,
                edge_sign=torch.from_numpy(edge_sign[mask]),
                edge_time=torch.from_numpy(edge_time[mask]),
                y=y,
                edge_y=edge_y,
                num_nodes=n,
            )

    def _assign_task_ids(self, timestamps: pd.Series) -> np.ndarray:
        """RTM REQ-5.36: greedy cumulative-volume partition of calendar days
        into `self.n_tasks` contiguous groups (see module docstring for why
        volume-balanced rather than fixed-width). A task's id is the group's
        starting day index (0-indexed from the file's first calendar day).
        """
        day_index = (timestamps.dt.floor("1D") - timestamps.dt.floor("1D").min()).dt.days.to_numpy()
        unique_days = np.unique(day_index)
        counts = pd.Series(day_index).value_counts().reindex(unique_days).to_numpy()
        total = counts.sum()
        target_per_task = total / self.n_tasks

        day_to_task = {}
        group_start_day = unique_days[0]
        running = 0
        remaining_tasks = self.n_tasks
        for i, day in enumerate(unique_days):
            day_to_task[day] = group_start_day
            running += counts[i]
            days_left = len(unique_days) - i - 1
            if running >= target_per_task and remaining_tasks > 1 and days_left > 0:
                remaining_tasks -= 1
                total -= running
                target_per_task = total / remaining_tasks
                running = 0
                group_start_day = unique_days[i + 1]

        return np.array([day_to_task[d] for d in day_index])

    @staticmethod
    def _pad_or_trim(feats: torch.Tensor, target_dim: int) -> torch.Tensor:
        n, d = feats.shape
        if d < target_dim:
            return torch.cat([feats, torch.zeros(n, target_dim - d)], dim=1)
        return feats[:, :target_dim]

    # --------------------------------------------------------------- public

    @property
    def task_ids(self) -> List[int]:
        return sorted(self._tasks.keys())

    def get_task(self, task_id: int) -> Data:
        return self._tasks[task_id]

    def labeled_seed_nodes(self, task_id: int) -> torch.Tensor:
        data = self._tasks[task_id]
        return torch.arange(data.num_nodes, dtype=torch.long)
