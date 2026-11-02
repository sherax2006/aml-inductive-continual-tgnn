"""RACLA training orchestration (RTM REQ-5.25).

Mirrors `aml_tgnn/train.py`'s task-stream structure (logging, checkpointing,
resume, per-task val metrics, zero-shot forward transfer, forgetting check,
paper-style eval) by directly reusing the same shared infrastructure --
`RunManager`, `setup_logging`, `run_eval`, `stratified_train_val_split`,
`compute_class_weights`, `discover_task_checkpoints`,
`compute_paper_performance_matrix`, `compute_paper_summary` -- but implements
RACLA's own per-task mechanics, which are genuinely different from this
project's own two continual-learning strategies:

  - **Drift-gated conditional replay** (Eq. 15-19): unlike `ewc_replay`/
    `gem_twp`, where replay/regularization is active on every step once any
    prior task exists, RACLA only replays during a task if the drift
    detector fired comparing this task's node features against the previous
    task's (Fig. 1: "Drift Detector -> if Drift: Replay -> AML Expert Layer").
  - **MoE load-balancing loss** (Eq. 7-9), added to every step's loss.
  - **An auxiliary edge-prediction loss** (Eq. 13/14), trained alongside the
    classifier every step on the current batch's own subgraph.
  - **End-of-task buffer update** via routing-based coverage selection
    (Eq. 10-12), using the just-trained model's own routing vectors.

    python -m aml_tgnn.racla.train --dataset elliptic --path elliptic_bitcoin_dataset \\
        [--resume [--run-id <id>]] [--epochs-per-task N] [--time-steps-per-task N]
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..data.split import compute_class_weights, stratified_train_val_split
from ..data.temporal_loader import build_temporal_neighbor_loader
from ..task_matrix import compute_paper_performance_matrix, compute_paper_summary, discover_task_checkpoints
from ..utils.checkpoint import load_checkpoint, read_progress
from ..utils.metrics import append_metrics_csv, run_eval
from ..utils.run_manager import RunManager, setup_logging
from .checkpoint import load_racla_model_from_checkpoint, save_racla_checkpoint
from .drift import DriftDetector
from .model import RACLAModel
from .replay import EdgePredictor, RACLAReplayBuffer


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log_architecture_summary(logger, model: RACLAModel, edge_predictor, cfg: ExperimentConfig, dataset_label: str, dataset) -> None:
    n_model = sum(p.numel() for p in model.parameters())
    n_edge = sum(p.numel() for p in edge_predictor.parameters())
    task_ids = list(dataset.task_ids)
    preview = task_ids if len(task_ids) <= 8 else f"{task_ids[:4]} ... {task_ids[-4:]}"

    logger.info("=" * 78)
    logger.info(f"RACLA baseline reimplementation (RTM REQ-5.25) | dataset: {dataset_label} | device: {cfg.train.device}")
    logger.info("=" * 78)
    logger.info(f"Task stream: {len(task_ids)} tasks -> {preview}")
    logger.info("--- Model architecture ---")
    for line in str(model).splitlines():
        logger.info(line)
    logger.info("--- Parameter counts ---")
    logger.info(f"model={n_model:,} | edge_predictor={n_edge:,} | total={n_model + n_edge:,}")
    logger.info("--- Config ---")
    for name, section in (
        ("backbone", cfg.backbone), ("racla", cfg.racla), ("task_head", cfg.task_head),
        ("loader", cfg.loader), ("train", cfg.train),
    ):
        logger.info(f"[{name}] {section}")
    logger.info("=" * 78)


def _training_step(
    model: RACLAModel, edge_predictor: EdgePredictor, batch, optimizer, cfg: ExperimentConfig,
    class_weights, replay_active: bool, replay_buffer: RACLAReplayBuffer, device,
) -> float:
    optimizer.zero_grad()
    logits, z_all = model.forward(batch, return_embeddings=True)
    y = batch.y[: batch.batch_size]
    loss = F.cross_entropy(logits, y, weight=class_weights)
    loss = loss + model.load_balance_loss()  # Eq. 7-9
    loss = loss + edge_predictor.edge_loss(z_all, batch.edge_index, cfg.racla.edge_predictor_neg_ratio)  # Eq. 13/14

    # Eq. 15-19: replay only mixed in when the drift detector fired for this
    # task -- reuses cfg.replay.replay_batch_size (this project's existing
    # replay-batching knob) rather than adding a redundant RACLA-specific one.
    if replay_active and len(replay_buffer) > 0:
        replay_batches = replay_buffer.sample_replay_batch(cfg.replay.replay_batch_size)
        replay_loss = torch.zeros((), device=device)
        for b in replay_batches:
            b = b.to(device)
            r_logits = model.forward(b)
            r_y = b.y[: b.batch_size]
            replay_loss = replay_loss + F.cross_entropy(r_logits, r_y, weight=class_weights)
        loss = loss + replay_loss / len(replay_batches)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _collect_routing_vectors(model: RACLAModel, data, node_ids: torch.Tensor, cfg: ExperimentConfig, device):
    """Eq. 10-12 need a routing vector g_i for every candidate node in a
    task, gathered in eval mode (Top-K gating injects noise only in
    training, so eval-mode gates are deterministic and stable to select a
    buffer from).
    """
    model.eval()
    loader = build_temporal_neighbor_loader(data, node_ids, cfg.loader.num_neighbors, cfg.loader.batch_size, shuffle=False)
    all_vectors, all_ids = [], []
    for batch in loader:
        batch = batch.to(device)
        model.forward(batch)  # populates every layer's last_gates
        all_vectors.append(model.routing_vectors(batch).cpu())
        all_ids.append(batch.n_id[: batch.batch_size].cpu())
    model.train()
    if not all_vectors:
        return torch.zeros(0, cfg.racla.num_experts), torch.zeros(0, dtype=torch.long)
    return torch.cat(all_vectors), torch.cat(all_ids)


def run_racla_training(
    model: RACLAModel, edge_predictor: EdgePredictor, dataset, cfg: ExperimentConfig, dataset_key: str,
    resume: bool = False, run_id: Optional[str] = None,
) -> RACLAModel:
    device = torch.device(cfg.train.device)

    run_manager = RunManager(dataset_key, logs_root=cfg.train.logs_root, checkpoints_root=cfg.train.checkpoints_root)
    run_id, log_dir, ckpt_dir = run_manager.prepare_run(resume=resume, run_id=run_id)
    logger = setup_logging(log_dir, run_id)
    logger.info(f"Run '{run_id}' | logs: {log_dir} | checkpoints: {ckpt_dir}")

    replay_buffer = RACLAReplayBuffer(cfg.racla)
    drift_detector = DriftDetector(cfg.racla)

    existing_progress = read_progress(ckpt_dir)
    ckpt = None
    if existing_progress is not None and not existing_progress.get("is_fully_completed", False):
        ckpt = load_checkpoint(ckpt_dir)
        progress = ckpt["progress"]

        if progress["task_ids_order"] != list(dataset.task_ids):
            raise ValueError(
                f"Refusing to resume run '{run_id}': the dataset's task stream doesn't match what this "
                "run was trained on (different --time-steps-per-task, or a different dataset/path). "
                "Pass the same dataset construction args this run started with."
            )

        model.load_state_dict(ckpt["model_state"])
        edge_predictor.load_state_dict(ckpt["edge_predictor_state"])
        replay_buffer.load_state_dict(ckpt["replay_state"])
        drift_detector.load_state_dict(ckpt["drift_state"])
        logger.info(
            f"Resumed: completed_tasks={progress['completed_task_ids']} | "
            f"current_task={progress['current_task_id']} | next_epoch={progress['next_epoch']} | "
            f"global_step={progress['global_step']}"
        )
    else:
        progress = {
            "dataset_name": run_manager.dataset_label,
            "run_id": run_id,
            "task_ids_order": list(dataset.task_ids),
            "completed_task_ids": [],
            "current_task_id": None,
            "next_epoch": 0,
            "global_step": 0,
            "is_fully_completed": False,
            "last_updated": _now_iso(),
        }

    model.to(device)
    edge_predictor.to(device)
    _log_architecture_summary(logger, model, edge_predictor, cfg, run_manager.dataset_label, dataset)

    completed_task_ids = set(progress["completed_task_ids"])
    global_step = progress["global_step"]
    task_ids = list(dataset.task_ids)

    for idx, task_id in enumerate(task_ids):
        if task_id in completed_task_ids:
            continue

        data = dataset.get_task(task_id)
        all_seed_nodes = dataset.labeled_seed_nodes(task_id)
        if all_seed_nodes.numel() == 0:
            logger.info(f"Task {task_id} ({idx + 1}/{len(task_ids)}): no labeled seed nodes, skipping.")
            continue

        train_nodes, val_nodes = stratified_train_val_split(
            all_seed_nodes, data.y, cfg.train.val_ratio, cfg.train.split_seed, task_id
        )
        if train_nodes.numel() == 0:
            logger.info(f"Task {task_id} ({idx + 1}/{len(task_ids)}): train split empty after val hold-out, skipping.")
            continue

        class_weights = None
        if cfg.train.use_class_weights:
            class_weights = compute_class_weights(data.y[train_nodes], cfg.task_head.num_classes).to(device)
            logger.info(f"task {task_id} | class weights (licit, illicit) = {class_weights.tolist()}")

        # Eq. 15-19: drift check against the previous task's features, BEFORE
        # this task's training starts -- gates whether replay is active for
        # the whole of this task's training (Fig. 1's own sequencing).
        replay_active, drift_proportion = drift_detector.check_and_update(data.x[train_nodes])
        logger.info(
            f"task {task_id} | drift check | proportion_significant="
            f"{'n/a (first task)' if drift_proportion is None else f'{drift_proportion:.4f}'} | "
            f"replay_active={replay_active}"
        )

        params = list(model.parameters()) + list(edge_predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        is_resume_point = ckpt is not None and progress["current_task_id"] == task_id
        if is_resume_point:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = progress["next_epoch"]
        else:
            start_epoch = 0
        ckpt = None

        zero_shot_metrics = None
        if start_epoch == 0:
            zero_shot_metrics = run_eval(model, data, val_nodes, cfg, device)
            if zero_shot_metrics is not None:
                logger.info(
                    f"task {task_id} | ZERO-SHOT forward-transfer (current model, before any training on "
                    f"task {task_id}) | precision={zero_shot_metrics['precision']:.4f} "
                    f"recall={zero_shot_metrics['recall']:.4f} f1={zero_shot_metrics['f1']:.4f}"
                )

        train_loader = build_temporal_neighbor_loader(
            data, train_nodes, cfg.loader.num_neighbors, cfg.loader.batch_size, cfg.loader.shuffle
        )
        num_batches = len(train_loader)

        logger.info(
            f"=== Task {task_id} ({idx + 1}/{len(task_ids)}) | nodes={data.num_nodes} "
            f"edges={data.edge_index.size(1)} labeled={all_seed_nodes.numel()} "
            f"train={train_nodes.numel()} val={val_nodes.numel()} "
            f"| starting at epoch {start_epoch + 1}/{cfg.train.epochs_per_task} ==="
        )

        model.train()
        edge_predictor.train()
        for epoch in range(start_epoch, cfg.train.epochs_per_task):
            epoch_start = time.perf_counter()
            batch_times = []
            epoch_loss = 0.0
            n_steps = 0

            for batch_idx, batch in enumerate(train_loader):
                batch_start = time.perf_counter()
                batch = batch.to(device)
                loss_value = _training_step(
                    model, edge_predictor, batch, optimizer, cfg, class_weights, replay_active, replay_buffer, device
                )

                batch_dt = time.perf_counter() - batch_start
                batch_times.append(batch_dt)
                epoch_loss += loss_value
                n_steps += 1
                global_step += 1

                if (batch_idx + 1) % cfg.train.log_every_n_steps == 0 or (batch_idx + 1) == num_batches:
                    logger.info(
                        f"task {task_id} | epoch {epoch + 1}/{cfg.train.epochs_per_task} | "
                        f"batch {batch_idx + 1}/{num_batches} | loss={loss_value:.4f} | "
                        f"batch_time={batch_dt:.3f}s | step={global_step}"
                    )

            epoch_dt = time.perf_counter() - epoch_start
            avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0.0
            remaining_epochs = cfg.train.epochs_per_task - (epoch + 1)
            eta = remaining_epochs * epoch_dt
            logger.info(
                f"--- task {task_id} | epoch {epoch + 1}/{cfg.train.epochs_per_task} complete | "
                f"avg_loss={epoch_loss / max(n_steps, 1):.4f} | epoch_time={epoch_dt:.2f}s | "
                f"avg_batch_time={avg_batch:.3f}s | eta_this_task={eta:.1f}s ---"
            )

            progress.update(
                current_task_id=task_id, next_epoch=epoch + 1, global_step=global_step,
                is_fully_completed=False, last_updated=_now_iso(),
            )
            save_racla_checkpoint(ckpt_dir, model, edge_predictor, optimizer, replay_buffer, drift_detector, progress, cfg)

        # End of task: evaluate, then update the buffer (Eq. 10-12) using
        # this task's own just-trained routing vectors.
        model.eval()
        m = run_eval(model, data, val_nodes, cfg, device)
        if m is not None:
            logger.info(
                f"task {task_id} | VAL metrics (illicit class) | n={m['n']} (pos={m['support_pos']}, neg={m['support_neg']}) | "
                f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} accuracy={m['accuracy']:.4f} | "
                f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}"
            )
            row = {"task_id": task_id, "timestamp": _now_iso(), **m}
            if zero_shot_metrics is not None:
                row.update(
                    zero_shot_precision=zero_shot_metrics["precision"], zero_shot_recall=zero_shot_metrics["recall"],
                    zero_shot_f1=zero_shot_metrics["f1"], zero_shot_accuracy=zero_shot_metrics["accuracy"],
                )
            else:
                row.update(zero_shot_precision=None, zero_shot_recall=None, zero_shot_f1=None, zero_shot_accuracy=None)
            append_metrics_csv(log_dir / "task_metrics.csv", row)
        else:
            logger.info(f"task {task_id} | val split empty, skipping metrics")

        routing_vectors, node_ids = _collect_routing_vectors(model, data, train_nodes, cfg, device)
        n_admitted = replay_buffer.add_task_buffer(data, node_ids, routing_vectors, cfg.racla.khop_num_neighbors) \
            if routing_vectors.numel() > 0 else 0
        logger.info(f"task {task_id} | replay buffer: admitted {n_admitted} nodes (buffer size now {len(replay_buffer)})")
        model.train()

        completed_task_ids.add(task_id)
        progress.update(
            completed_task_ids=sorted(completed_task_ids, key=lambda t: task_ids.index(t)),
            current_task_id=None, next_epoch=0, global_step=global_step,
            is_fully_completed=False, last_updated=_now_iso(),
        )
        save_racla_checkpoint(
            ckpt_dir, model, edge_predictor, None, replay_buffer, drift_detector, progress, cfg,
            milestone_name=f"task_{task_id}_complete.pt",
        )
        logger.info(f"Task {task_id} fully consolidated. Checkpoint saved -> {ckpt_dir / f'task_{task_id}_complete.pt'}")

    if cfg.train.paper_style_eval:
        n = len(discover_task_checkpoints(ckpt_dir))
        logger.info(f"=== Paper-style evaluation: computing M ({n}x{n}, i<=j only -> {n * (n + 1) // 2} evaluation passes) ===")
        M, m_task_order = compute_paper_performance_matrix(dataset, ckpt_dir, device=str(device), model_loader=load_racla_model_from_checkpoint)
        logger.info("Performance Matrix M (row = test task i, col = trained-through task j; 0.0000 where i > j)")
        header = "i\\j".rjust(8) + "".join(f"{j:>10}" for j in m_task_order)
        logger.info(header)
        for i in m_task_order:
            cells = "".join(f"{M[i][j]:>10.4f}" for j in m_task_order)
            logger.info(f"{i:>8}{cells}")
        paper_summary = compute_paper_summary(M, m_task_order)
        logger.info(
            f"Average Performance = {paper_summary['average_performance']:.4f}   "
            f"Average Forgetting = {paper_summary['average_forgetting']:+.4f}"
        )

    progress.update(is_fully_completed=True, current_task_id=None, last_updated=_now_iso())
    save_racla_checkpoint(ckpt_dir, model, edge_predictor, None, replay_buffer, drift_detector, progress, cfg)
    logger.info(f"=== Training complete: all {len(task_ids)} tasks done ({run_id}) ===")

    return model


if __name__ == "__main__":
    import argparse

    from ..data.elliptic_dataset import EllipticTemporalDataset
    from ..data.ibm_aml_dataset import IBMAMLTemporalDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["elliptic", "ibm_aml"], required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs-per-task", type=int, default=None)
    parser.add_argument("--time-steps-per-task", type=int, default=1)
    args = parser.parse_args()

    if args.dataset == "ibm_aml" and args.time_steps_per_task != 1:
        parser.error("--time-steps-per-task is Elliptic-only for now")

    cfg = ExperimentConfig()
    if args.epochs_per_task is not None:
        cfg.train.epochs_per_task = args.epochs_per_task
    cfg.elliptic_time_steps_per_task = args.time_steps_per_task

    if args.dataset == "elliptic":
        ds = EllipticTemporalDataset(
            args.path, edge_in_dim=cfg.backbone.edge_in_dim, time_steps_per_task=cfg.elliptic_time_steps_per_task,
        )
        cfg.backbone.node_in_dim = ds.node_in_dim
    else:
        ds = IBMAMLTemporalDataset(args.path, edge_in_dim=cfg.backbone.edge_in_dim, node_in_dim=cfg.backbone.node_in_dim)

    model = RACLAModel(cfg)
    edge_predictor = EdgePredictor(cfg.racla.expert_hidden_dim, cfg.racla.edge_predictor_hidden_dim)
    run_racla_training(model, edge_predictor, ds, cfg, dataset_key=args.dataset, resume=args.resume, run_id=args.run_id)
