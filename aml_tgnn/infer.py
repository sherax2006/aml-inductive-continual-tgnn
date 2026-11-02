"""Standalone inference over a trained checkpoint (RTM REQ-5.7).

Loads any saved checkpoint (`latest.pt` or a `task_<id>_complete.pt`
milestone) and runs the model on one task's nodes -- either to get
predictions for nodes you don't have labels for, or to recompute the same
precision/recall/F1 metrics `train.py` logs during training, on demand,
without retraining anything.

    python -m aml_tgnn.infer --dataset elliptic --path elliptic_bitcoin_dataset \\
        --checkpoint Checkpoints/Elliptic/run_.../task_5_complete.pt --task-id 5

`--task-id` only selects *which task's graph/seed units* to score (via
`dataset.get_task`/`labeled_seed_nodes`, or every edge for IBM AML's
edge-classification mode -- RTM REQ-5.38) -- it no longer selects a
classification head. Parameter isolation was removed (RTM REQ-5.15): every
checkpoint has exactly one shared head, so any task_id can be scored by any
checkpoint's model with no routing decision to get wrong.

`load_model_from_checkpoint` restores the exact `ExperimentConfig` a checkpoint
was trained with (backbone dims, `elliptic_time_steps_per_task`) rather than
trusting the caller to re-supply matching flags by hand -- get a dimension
wrong and `load_state_dict` fails with a loud shape mismatch, rather than the
silent garbage-predictions failure mode this guarded against back when head
routing was mode-dependent. Checkpoints saved before config persistence existed
have no `cfg` in their payload; those fall back to `ExperimentConfig()`
defaults, with a warning printed so it's never a silent assumption.
"""
import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import torch

from .config import ExperimentConfig
from .data.temporal_loader import build_temporal_link_loader, build_temporal_neighbor_loader
from .models.continual_wrapper import ContinualTGNN
from .utils.metrics import binary_classification_metrics


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> Tuple[ContinualTGNN, ExperimentConfig]:
    # weights_only=False: our own checkpoints, not untrusted downloads -- the
    # payload includes the plain-Python ExperimentConfig dataclass, which
    # torch's newer default weights_only=True unpickling would refuse to load.
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = payload.get("cfg")
    if cfg is None:
        cfg = ExperimentConfig()
        print(
            f"WARNING: {checkpoint_path} predates config persistence -- assuming default "
            f"ExperimentConfig() (elliptic_time_steps_per_task=1). "
            "If this checkpoint was trained with non-default settings, predictions will be wrong."
        )

    model = ContinualTGNN(cfg)
    # RTM REQ-5.32: checkpoints saved before the ablation switches existed have
    # no `pool_proj` weights at all (that module didn't exist yet) -- strict
    # loading would reject them outright even though nothing else changed.
    # `pool_proj` is always constructed but only ever *used* when
    # `cfg.backbone.use_lstm=False` (never true for these older checkpoints,
    # which predate the flag entirely), so leaving it at fresh random init is
    # exactly correct: it's dead code for this model, not a silently-wrong
    # restoration. Verified explicitly, not just assumed permissive: any
    # missing/unexpected key *other* than `pool_proj.{weight,bias}` per
    # backbone layer still raises, so a genuine shape/name mismatch elsewhere
    # is not masked by this.
    result = model.load_state_dict(payload["model_state"], strict=False)
    allowed_missing = {f"backbone.layers.{i}.pool_proj.{p}" for i in range(cfg.backbone.num_hops) for p in ("weight", "bias")}
    unexpected_missing = set(result.missing_keys) - allowed_missing
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"load_state_dict for {checkpoint_path}: unexpected mismatch -- "
            f"missing={unexpected_missing or None}, unexpected={result.unexpected_keys or None}"
        )
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_task(
    model: ContinualTGNN,
    dataset,
    task_id,
    cfg: ExperimentConfig,
    device: str = "cpu",
    node_ids: Optional[torch.Tensor] = None,
) -> dict:
    """Runs inference over `node_ids` (default: every labeled seed node in the
    task). Ground truth (`y_true`) is included whenever the dataset has it --
    both adapters always attach `y`, so metrics come back automatically
    whenever the requested nodes are labeled; unlabeled nodes (`y == -1`,
    Elliptic-style) just make `metrics` meaningless, not absent.

    RTM REQ-5.38: `node_ids` is actually "seed units" -- node ids
    (`cfg.task_type == "node"`, Elliptic, unchanged) or edge indices into
    `data.edge_index` (`cfg.task_type == "edge"`, IBM AML's native
    per-transaction label), same dispatch convention `run_eval` (RTM
    REQ-5.37) and `train.py`'s per-task loop already use. The parameter
    keeps its original name rather than being renamed to "unit_ids" to avoid
    an unrelated diff across every existing node-mode call site; only its
    meaning is broadened. Edge mode has no `-1`-unlabeled sentinel to filter
    (every IBM AML transaction is labeled), so `node_ids=None` defaults to
    every edge in the task, mirroring how `labeled_seed_nodes` already
    returns every node unconditionally for node mode.
    """
    data = dataset.get_task(task_id)
    if cfg.task_type == "edge":
        seed_units = node_ids if node_ids is not None else torch.arange(data.edge_index.size(1), dtype=torch.long)
        loader = build_temporal_link_loader(
            data, seed_units, cfg.loader.num_neighbors, cfg.loader.batch_size, shuffle=False
        )
    else:
        seed_units = node_ids if node_ids is not None else dataset.labeled_seed_nodes(task_id)
        loader = build_temporal_neighbor_loader(
            data, seed_units, cfg.loader.num_neighbors, cfg.loader.batch_size, shuffle=False
        )

    unit_ids_out, probs_out, preds_out, y_true_out = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model.forward(batch)
        probs = torch.softmax(logits, dim=-1)

        if cfg.task_type == "edge":
            # RTM REQ-5.38: the target edges' GLOBAL (source, destination) node
            # ids -- `edge_label_index` holds only LOCAL (per-batch) endpoint
            # indices (RTM REQ-5.37's `LinkNeighborLoader` verification), so
            # `n_id` must be used to recover the globally-meaningful pair, the
            # same lookup `train.py`/replay-buffer admission already rely on.
            src_local, dst_local = batch.edge_label_index[0], batch.edge_label_index[1]
            unit_ids_out.append(torch.stack([batch.n_id[src_local], batch.n_id[dst_local]], dim=1).cpu())
        else:
            unit_ids_out.append(batch.n_id[: batch.batch_size].cpu())
        probs_out.append(probs.cpu())
        preds_out.append(probs.argmax(dim=-1).cpu())
        y_true_out.append(model.batch_targets(batch).cpu())

    result = {
        "node_ids": torch.cat(unit_ids_out),
        "probs": torch.cat(probs_out),
        "y_pred": torch.cat(preds_out),
        "y_true": torch.cat(y_true_out),
    }
    labeled_mask = result["y_true"] != -1
    if labeled_mask.any():
        result["metrics"] = binary_classification_metrics(result["y_true"][labeled_mask], result["y_pred"][labeled_mask])
    else:
        result["metrics"] = None
    return result


def _write_predictions_csv(path: Path, result: dict, task_type: str = "node") -> None:
    """RTM REQ-5.38: edge mode's `result["node_ids"]` is `[n, 2]` (src, dst
    global node ids per target edge, see `predict_task` above), not a flat
    per-node id -- written as two columns instead of one so a transaction's
    prediction stays traceable to the accounts it's between.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if task_type == "edge":
            writer.writerow(["src_node_id", "dst_node_id", "y_true", "y_pred", "prob_licit", "prob_illicit"])
            for (src, dst), yt, yp, p in zip(result["node_ids"].tolist(), result["y_true"].tolist(),
                                              result["y_pred"].tolist(), result["probs"].tolist()):
                writer.writerow([src, dst, yt, yp, f"{p[0]:.6f}", f"{p[1]:.6f}"])
        else:
            writer.writerow(["node_id", "y_true", "y_pred", "prob_licit", "prob_illicit"])
            for nid, yt, yp, p in zip(result["node_ids"].tolist(), result["y_true"].tolist(),
                                       result["y_pred"].tolist(), result["probs"].tolist()):
                writer.writerow([nid, yt, yp, f"{p[0]:.6f}", f"{p[1]:.6f}"])


if __name__ == "__main__":
    from .data.elliptic_dataset import EllipticTemporalDataset
    from .data.ibm_aml_dataset import IBMAMLTemporalDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["elliptic", "ibm_aml"], required=True)
    parser.add_argument("--path", required=True, help="dataset root (elliptic) or CSV path (ibm_aml)")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint (latest.pt or task_<id>_complete.pt)")
    parser.add_argument(
        "--task-id", required=True, type=int,
        help="which task's seed units (nodes, or edges for IBM AML's edge-classification mode, "
             "RTM REQ-5.38) to run inference on",
    )
    parser.add_argument("--out-csv", default=None, help="optional path to write per-unit predictions")
    parser.add_argument(
        "--model-type", choices=["continual_tgnn", "racla"], default="continual_tgnn",
        help="RTM REQ-5.25: which model class --checkpoint holds -- this project's own "
             "ContinualTGNN (default), or the RACLA baseline reimplementation (aml_tgnn/racla/). "
             "predict_task() only ever calls model.forward(batch), so only the checkpoint loader differs.",
    )
    args = parser.parse_args()

    if args.model_type == "racla":
        from .racla.checkpoint import load_racla_model_from_checkpoint as model_loader
    else:
        model_loader = load_model_from_checkpoint

    device = ExperimentConfig().train.device  # runtime device choice, independent of what training used
    model, cfg = model_loader(args.checkpoint, device=device)

    if args.dataset == "elliptic":
        ds = EllipticTemporalDataset(
            args.path, edge_in_dim=cfg.backbone.edge_in_dim,
            time_steps_per_task=cfg.elliptic_time_steps_per_task,
        )
    else:
        ds = IBMAMLTemporalDataset(
            args.path, edge_in_dim=cfg.backbone.edge_in_dim, node_in_dim=cfg.backbone.node_in_dim,
            n_tasks=getattr(cfg, "ibm_aml_n_tasks", 7),
        )

    result = predict_task(model, ds, args.task_id, cfg, device=device)

    # RTM REQ-5.38: edge mode's `node_ids` is `[n, 2]` (src, dst pairs) --
    # `len(...)` (size of dim 0) gives the unit count correctly either way;
    # `.numel()` would double-count edge mode's 2 columns per row.
    unit_label = "edges" if cfg.task_type == "edge" else "nodes"
    print(f"Task {args.task_id}: {len(result['node_ids'])} {unit_label} scored")
    if result["metrics"] is not None:
        m = result["metrics"]
        print(f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} "
              f"accuracy={m['accuracy']:.4f} (n={m['n']}, pos={m['support_pos']}, neg={m['support_neg']})")
    else:
        print(f"no labeled {unit_label} in this task -- predictions only, no metrics")

    if args.out_csv:
        _write_predictions_csv(Path(args.out_csv), result, task_type=cfg.task_type)
        print(f"per-{unit_label[:-1]} predictions written to {args.out_csv}")
