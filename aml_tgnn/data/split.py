"""Stratified per-task train/val split (RTM REQ-5.6).

Splits each task's *labeled* seed nodes into a training set (used for gradient
updates, replay-buffer admission, and EWC's Fisher estimate) and a held-out
validation set (used only for the metrics in `train.py::_end_of_task` --
never trained on). Stratified by label so that the illicit class -- always
the rare minority in AML data -- doesn't disappear from a small task's
validation split by chance, which would silently make its recall undefined.

Deterministic and resume-safe: the split for a given `(task_id, base_seed)`
is always identical, so re-deriving it after a crash/resume reproduces
exactly the same train/val partition the interrupted run was using -- no
need to persist the split itself in a checkpoint.
"""
from typing import Tuple

import torch


def stratified_train_val_split(
    seed_nodes: torch.Tensor,
    labels: torch.Tensor,
    val_ratio: float,
    base_seed: int,
    task_id,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`labels` is the task's full per-node label tensor (e.g. `data.y`),
    indexed by `seed_nodes`. Returns (train_nodes, val_nodes), both subsets
    of `seed_nodes`. A class with only one labeled example keeps it in
    train rather than orphaning it in an empty-context validation split.
    """
    node_labels = labels[seed_nodes]
    generator = torch.Generator().manual_seed(base_seed + int(task_id))

    train_parts, val_parts = [], []
    for cls in torch.unique(node_labels).tolist():
        cls_nodes = seed_nodes[node_labels == cls]
        perm = cls_nodes[torch.randperm(cls_nodes.numel(), generator=generator)]

        if perm.numel() <= 1:
            n_val = 0
        else:
            n_val = min(max(int(round(perm.numel() * val_ratio)), 0), perm.numel() - 1)

        val_parts.append(perm[:n_val])
        train_parts.append(perm[n_val:])

    train_nodes = torch.cat(train_parts) if train_parts else seed_nodes[:0]
    val_nodes = torch.cat(val_parts) if val_parts else seed_nodes[:0]
    return train_nodes, val_nodes


def stratified_train_val_coldstart_split(
    seed_nodes: torch.Tensor,
    labels: torch.Tensor,
    val_ratio: float,
    cold_start_ratio: float,
    base_seed: int,
    task_id,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RTM REQ-5.30: three-way stratified split -- train / val / cold-start --
    for the genuine held-out-node inductive generalization experiment
    (RACLA's own fixed-node-set framing, Definition 2 in BasePaper1_RACLA.md,
    has no analogue for this at all).

    `cold_start_ratio` is carved out FIRST, per class (same stratification
    `stratified_train_val_split` already uses, for the same reason -- the
    illicit class is always the rare minority and must not vanish from a
    small split by chance), then `val_ratio` splits the *remainder* --  so
    `val_ratio`'s meaning is unchanged whether or not cold-start is enabled
    (always "this fraction of what's left for training/validation", not "this
    fraction of everything including the cold-start reserve"). Uses the exact
    same `(base_seed, task_id)`-derived generator as `stratified_train_val_split`,
    so with `cold_start_ratio=0.0` this reduces to the identical train/val
    partition that function would produce on its own.

    The returned `cold_nodes` are only meaningful for evaluation against the
    *unpruned* graph (`data/coldstart.py::prune_edges_for_nodes` removes their
    edges from what training actually sees) -- this function only decides
    *which* nodes are held out, not how the graph is pruned around them.
    """
    node_labels = labels[seed_nodes]
    generator = torch.Generator().manual_seed(base_seed + int(task_id))

    train_parts, val_parts, cold_parts = [], [], []
    for cls in torch.unique(node_labels).tolist():
        cls_nodes = seed_nodes[node_labels == cls]
        perm = cls_nodes[torch.randperm(cls_nodes.numel(), generator=generator)]

        if perm.numel() <= 1:
            n_cold = 0
        else:
            n_cold = min(max(int(round(perm.numel() * cold_start_ratio)), 0), perm.numel() - 1)
        cold_parts.append(perm[:n_cold])
        remainder = perm[n_cold:]

        if remainder.numel() <= 1:
            n_val = 0
        else:
            n_val = min(max(int(round(remainder.numel() * val_ratio)), 0), remainder.numel() - 1)
        val_parts.append(remainder[:n_val])
        train_parts.append(remainder[n_val:])

    train_nodes = torch.cat(train_parts) if train_parts else seed_nodes[:0]
    val_nodes = torch.cat(val_parts) if val_parts else seed_nodes[:0]
    cold_nodes = torch.cat(cold_parts) if cold_parts else seed_nodes[:0]
    return train_nodes, val_nodes, cold_nodes


def stratified_train_val_coldstart_split_edges(
    num_edges: int,
    edge_labels: torch.Tensor,
    val_ratio: float,
    cold_start_ratio: float,
    base_seed: int,
    task_id,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RTM REQ-5.37: the edge-classification analogue of
    `stratified_train_val_coldstart_split` -- same three-way stratified,
    deterministic, cold-start-carved-out-first construction, but the unit
    being split is EDGES (transactions), not nodes (accounts), and held out
    directly regardless of whether their endpoint accounts appear elsewhere
    in the task (user's explicit design choice: "held-out edges directly",
    not "held-out accounts, score their edges" -- the latter would reuse
    `stratified_train_val_coldstart_split` + `prune_edges_for_nodes`
    unchanged; this one needs its own held-out mechanism because a held-out
    edge's own endpoints stay fully active via their OTHER edges).

    Returns (train_edge_idx, val_edge_idx, cold_edge_idx): local indices into
    the task's own `edge_index`/`edge_attr`/edge-label arrays (0..num_edges-1),
    not node ids. `cold_start_ratio=0.0` reduces to an ordinary two-way
    train/val split over edges, identical in structure to
    `stratified_train_val_split` but edge-indexed.
    """
    edge_idx = torch.arange(num_edges, dtype=torch.long)
    generator = torch.Generator().manual_seed(base_seed + int(task_id))

    train_parts, val_parts, cold_parts = [], [], []
    for cls in torch.unique(edge_labels).tolist():
        cls_edges = edge_idx[edge_labels == cls]
        perm = cls_edges[torch.randperm(cls_edges.numel(), generator=generator)]

        if perm.numel() <= 1:
            n_cold = 0
        else:
            n_cold = min(max(int(round(perm.numel() * cold_start_ratio)), 0), perm.numel() - 1)
        cold_parts.append(perm[:n_cold])
        remainder = perm[n_cold:]

        if remainder.numel() <= 1:
            n_val = 0
        else:
            n_val = min(max(int(round(remainder.numel() * val_ratio)), 0), remainder.numel() - 1)
        val_parts.append(remainder[:n_val])
        train_parts.append(remainder[n_val:])

    train_edges = torch.cat(train_parts) if train_parts else edge_idx[:0]
    val_edges = torch.cat(val_parts) if val_parts else edge_idx[:0]
    cold_edges = torch.cat(cold_parts) if cold_parts else edge_idx[:0]
    return train_edges, val_edges, cold_edges


def compute_class_weights(labels: torch.Tensor, num_classes: int, power: float = 0.5) -> torch.Tensor:
    """RTM REQ-5.20: dampened inverse-frequency class weights for
    `F.cross_entropy(..., weight=...)`, computed fresh from a task's own
    *train* split every task -- this project's continual-learning setup
    means the illicit-class ratio can genuinely shift from task to task, so
    a single fixed global weighting wouldn't track that drift.

    Without any weighting, plain cross-entropy on Elliptic's/IBM AML's
    heavily imbalanced illicit class has a stable, trivial optimum: predict
    the majority (licit) class for every node -- confirmed on real data, see
    RTM.md REQ-5.20. The straightforward fix, full linear inverse-frequency
    weighting (`weight_c = N / (num_classes * count_c)`, `power=1.0`), was
    tried first and *also* confirmed on real data (same row) to overcorrect
    at severe ratios (~49x on a ~2%-illicit task): with each rare-class
    example weighted 49x, a batch containing even a handful of them can have
    their aggregate weighted loss dominate that batch's gradient, making
    "predict the rare class whenever unsure" a cheap way to satisfy the
    reweighted objective -- the mirror-image collapse (100% recall, ~2%
    precision) of the original all-majority one.

    `power` dampens the ratio: `weight_c = (N / (num_classes * count_c)) **
    power`. `power=0.5` (the default, RTM REQ-5.21) takes the square root of
    the full inverse-frequency ratio -- a standard, well-established
    technique for exactly this failure mode, turning e.g. a 49x ratio into
    ~7x. `power=1.0` recovers the original (REQ-5.20) full linear weighting;
    `power=0.0` recovers unweighted cross-entropy. Classes absent from this
    split get weight 0 regardless of `power` (no examples of that class need
    reweighting; `F.cross_entropy` tolerates a zero weight).
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = torch.zeros(num_classes)
    present = counts > 0
    weights[present] = (total / (num_classes * counts[present])) ** power
    return weights
