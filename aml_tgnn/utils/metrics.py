"""Binary classification metrics for the illicit/positive class (RTM REQ-5.6).

Implemented directly on `torch.Tensor` rather than via scikit-learn: this runs
inside the training loop after every single task, so it shouldn't depend on
an optional package being present. Reports precision/recall/F1 on the
positive (illicit) class specifically rather than plain accuracy -- AML
datasets are heavily imbalanced (illicit is the rare class), so accuracy
alone would stay high even for a model that never predicts illicit at all.
"""
import csv
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..data.temporal_loader import build_temporal_link_loader, build_temporal_neighbor_loader


def binary_classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, positive_class: int = 1) -> Dict[str, Any]:
    y_true = y_true.long()
    y_pred = y_pred.long()

    tp = int(((y_pred == positive_class) & (y_true == positive_class)).sum())
    fp = int(((y_pred == positive_class) & (y_true != positive_class)).sum())
    fn = int(((y_pred != positive_class) & (y_true == positive_class)).sum())
    tn = int(((y_pred != positive_class) & (y_true != positive_class)).sum())
    n = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    # RTM REQ-5.14: micro-F1, for matching a paper's evaluation protocol exactly.
    # For single-label classification (argmax picks exactly one predicted class per
    # sample, true for every eval path in this codebase), pooling TP/FP/FN across
    # *all* classes before computing precision/recall makes micro-precision ==
    # micro-recall == micro-F1 == accuracy -- an algebraic identity, not an
    # approximation. Kept as its own explicit field (rather than just reusing
    # `accuracy`) so paper-comparison code reads as "micro-F1" at the call site.
    micro_f1 = accuracy

    return {
        "n": n,
        "support_pos": tp + fn,
        "support_neg": tn + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "micro_f1": micro_f1,
    }


@torch.no_grad()
def run_eval(model, data, nodes: torch.Tensor, cfg, device) -> Optional[Dict[str, Any]]:
    """Shared eval core: samples `nodes`' neighborhoods from `data`, forwards
    through `model` (the one shared classification head, RTM REQ-5.15 --
    no task_id routing any more), and returns `binary_classification_metrics`.
    Used by `train.py` (post-training val metrics, forward-transfer probes,
    the forgetting check) and `task_matrix.py` (the full R matrix, RTM
    REQ-5.12) so all of them score exactly the same way.

    RTM REQ-5.37: `nodes` is actually "seed units" -- node ids (`cfg.task_type
    == "node"`, Elliptic, unchanged) or edge indices into `data.edge_index`
    (`cfg.task_type == "edge"`, IBM AML's native per-transaction label). Every
    existing call site's signature is unchanged; only what the argument means
    differs, dispatched here rather than by adding a second near-duplicate
    function, so `train.py`/`task_matrix.py`/`infer.py` don't need their own
    task_type branch just to call this.
    """
    if nodes.numel() == 0:
        return None
    model.eval()
    if cfg.task_type == "edge":
        loader = build_temporal_link_loader(data, nodes, cfg.loader.num_neighbors, cfg.loader.batch_size, shuffle=False)
    else:
        loader = build_temporal_neighbor_loader(data, nodes, cfg.loader.num_neighbors, cfg.loader.batch_size, shuffle=False)

    y_true_all, y_pred_all = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model.forward(batch)
        y_pred_all.append(logits.argmax(dim=-1).cpu())
        y_true_all.append(model.batch_targets(batch).cpu())

    return binary_classification_metrics(torch.cat(y_true_all), torch.cat(y_pred_all))


def append_metrics_csv(path: Path, row: Dict[str, Any]) -> None:
    """Appends one row to a per-run metrics history file, writing the header
    on first use. Field order is fixed by the first call in a run so the CSV
    stays consistent across resumed sessions appending to the same file.
    """
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
