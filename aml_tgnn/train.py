"""Continual task-stream orchestration (RTM REQ-3.3), with logging, console
progress reporting, checkpointing, and crash-safe resume (RTM Part 5).

Per task:
  0. split the task's labeled seed nodes into a stratified train/val partition
     (RTM REQ-5.6) -- validation nodes are never trained on, never fed to the
     replay buffer, and never used for EWC's Fisher estimate,
  1. train for `cfg.train.epochs_per_task` epochs on the *train* split, each step
     adding the current task's cross-entropy + the running EWC penalty (REQ-2.2)
     + a freshly resampled replay loss (REQ-2.4) into one backward() call,
  2. at task end, sweep the *train* split once more to harvest high-confidence
     illicit nodes into the replay buffer (REQ-2.3) and consolidate EWC's Fisher
     estimate (REQ-2.1) so the *next* task's training is penalized for drifting
     the backbone away from what this task needed, then evaluate on the *val*
     split (precision/recall/F1 on the illicit class) and append the result to
     this run's `task_metrics.csv`.

Two more evaluation signals run around that per-task loop (RTM REQ-5.8, added
after a design discussion on 2026-08-20 about what "holistic evaluation"
should mean for a continually-trained model rather than a one-shot classifier):

  - **Forward transfer ("test-then-train"):** immediately before a task is
    ever trained on (i.e. only when starting fresh, not when resuming mid-task),
    it is scored zero-shot on its own val split using the model exactly as it
    stands at that moment -- a real "has this model, as it stood a moment ago,
    generalized to a time period it has never touched" number, logged alongside
    the normal post-training val metrics in the same `task_metrics.csv` row.
    For the very first task of a run this reflects a freshly-initialized
    model (a random-baseline reference point, not "missing data") -- computed
    unconditionally, no special-casing needed. Gives a full forward-transfer
    curve across every task, rather than a single number from an arbitrary
    train/val/test task split.
  - **Backward transfer / forgetting check:** once the entire task stream
    finishes, a handful of "anchor" tasks (first 3 trained, by default --
    `cfg.train.forgetting_anchor_task_ids` to override) are re-scored on
    their original val split using the model as it now stands, and compared
    against the F1 they got when first trained. Written to `forgetting_check.csv`.
    This is a cheap proxy for full backward-transfer (a handful of extra eval
    passes, not an O(T^2) re-evaluation of every prior task after every new
    one), which was explicitly scoped out for cost reasons.

Once the whole task stream finishes, a third evaluation stage runs (RTM
REQ-5.14, `cfg.train.paper_style_eval`, default on): the M performance matrix
and Average Performance / Average Forgetting scalars exactly as defined by a
specific published paper this project's evaluation was asked to match --
`M[i,j]` = micro-F1 on test task i using the model trained through task j,
for i<=j only, 0.0 otherwise (see `task_matrix.py`'s module docstring for how
this deliberately differs from this project's own richer R matrix). This is
the O(T(T+1)/2) full-lower-triangle re-evaluation the forgetting-check proxy
above was designed to avoid paying every task -- paid once, at the very end,
instead.

Works against either `EllipticTemporalDataset` or `IBMAMLTemporalDataset` --
both expose the same `task_ids` / `get_task` / `labeled_seed_nodes` interface.

Resume is epoch-granular: a checkpoint is only ever written at the end of a
fully-completed epoch or fully-completed task, never mid-epoch. If the process
is killed mid-epoch (power loss, network drop, laptop sleep), the next run
with `--resume` restarts that epoch from its beginning -- nothing partial is
ever trusted. See `aml_tgnn/utils/checkpoint.py` for the on-disk layout.

RTM REQ-5.15 (2026-08-21): parameter isolation (per-task dynamic heads, and
the earlier task_il/domain_il mode switch, REQ-5.9) was removed at the user's
explicit request. There is exactly one shared classification head, always --
no more task_id argument anywhere in the model's forward pass, no more
per-task head provisioning, no more task_head_mode resume guard. This
simplified a fair amount of what's below relative to earlier versions of this
file: `model.parameters()` is now static across an entire run (backbone + the
one fixed head), so there's no more optimizer-parameter-ordering concern on
resume either.
"""
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .continual.twp import TWP
from .data.coldstart import prune_edges_for_nodes, prune_specific_edges
from .data.split import (
    compute_class_weights,
    stratified_train_val_coldstart_split,
    stratified_train_val_coldstart_split_edges,
    stratified_train_val_split,
)
from .data.temporal_loader import build_temporal_link_loader, build_temporal_neighbor_loader
from .models.continual_wrapper import ContinualTGNN
from .task_matrix import discover_task_checkpoints, compute_paper_performance_matrix, compute_paper_summary
from .utils.checkpoint import load_checkpoint, read_progress, save_checkpoint
from .utils.metrics import append_metrics_csv, run_eval
from .utils.run_manager import RunManager, setup_logging


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _build_sweep_loader(data, units: torch.Tensor, cfg: ExperimentConfig, shuffle: bool):
    """RTM REQ-5.37: dispatches to the node- or edge-classification loader
    depending on `cfg.task_type` -- every internal sweep in this file (the
    training loader, the replay/GEM/A-GEM/MAS/TWP consolidation sweeps) goes
    through this one function so the two modes can't drift apart on which
    loader they use. `units` is node ids (`task_type == "node"`) or edge
    indices into `data.edge_index` (`task_type == "edge"`).
    """
    if cfg.task_type == "edge":
        return build_temporal_link_loader(data, units, cfg.loader.num_neighbors, cfg.loader.batch_size, shuffle=shuffle)
    return build_temporal_neighbor_loader(data, units, cfg.loader.num_neighbors, cfg.loader.batch_size, shuffle=shuffle)


def _log_architecture_summary(logger, model: ContinualTGNN, cfg: ExperimentConfig, dataset_label: str, dataset) -> None:
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    # RTM REQ-5.37: exactly one of task_head/edge_head is constructed, per cfg.task_type.
    head = model.task_head if cfg.task_type == "node" else model.edge_head
    n_head = sum(p.numel() for p in head.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    task_ids = list(dataset.task_ids)
    preview = task_ids if len(task_ids) <= 8 else f"{task_ids[:4]} ... {task_ids[-4:]}"

    logger.info("=" * 78)
    logger.info(f"Inductive Hybrid Continual TGNN | dataset: {dataset_label} | device: {cfg.train.device}")
    logger.info("=" * 78)
    logger.info(f"Task stream: {len(task_ids)} tasks -> {preview}")
    logger.info("--- Model architecture ---")
    for line in str(model).splitlines():
        logger.info(line)
    logger.info("--- Parameter counts ---")
    logger.info(
        f"backbone={n_backbone:,} | task_head={n_head:,} (one shared head, fixed size) "
        f"| total={n_total:,} | trainable={n_trainable:,}"
    )
    logger.info("--- Config ---")
    logger.info(f"[task_type] {cfg.task_type!r}")
    logger.info(f"[continual_strategy] {cfg.continual_strategy!r}")
    if cfg.continual_strategy == "gem_twp" and (cfg.disable_gem or cfg.disable_twp):
        logger.info(f"[gem_twp component ablation, RTM REQ-5.39] disable_gem={cfg.disable_gem} disable_twp={cfg.disable_twp}")
    if cfg.continual_strategy == "ewc_replay":
        strategy_sections = (("ewc", cfg.ewc), ("replay", cfg.replay))
    elif cfg.continual_strategy == "gem_twp":
        strategy_sections = (("gem", cfg.gem), ("twp", cfg.twp))
    elif cfg.continual_strategy == "agem_twp":
        strategy_sections = (("agem", cfg.agem), ("twp", cfg.twp))
    else:  # mas_twp
        strategy_sections = (("mas", cfg.mas), ("twp", cfg.twp))
    for name, section in (("backbone", cfg.backbone),) + strategy_sections + (
        ("task_head", cfg.task_head),
        ("loader", cfg.loader),
        ("train", cfg.train),
    ):
        logger.info(f"[{name}] {section}")
    logger.info("=" * 78)


def run_continual_training(
    model: ContinualTGNN,
    dataset,
    cfg: ExperimentConfig,
    dataset_key: str,
    resume: bool = False,
    run_id: Optional[str] = None,
) -> ContinualTGNN:
    device = torch.device(cfg.train.device)

    run_manager = RunManager(dataset_key, logs_root=cfg.train.logs_root, checkpoints_root=cfg.train.checkpoints_root)
    run_id, log_dir, ckpt_dir = run_manager.prepare_run(resume=resume, run_id=run_id)
    logger = setup_logging(log_dir, run_id)
    logger.info(f"Run '{run_id}' | logs: {log_dir} | checkpoints: {ckpt_dir}")

    existing_progress = read_progress(ckpt_dir)
    ckpt = None
    if existing_progress is not None and not existing_progress.get("is_fully_completed", False):
        ckpt = load_checkpoint(ckpt_dir)
        progress = ckpt["progress"]

        # Guard against resuming against a different task stream than the run
        # started with (e.g. forgetting --time-steps-per-task on a resumed
        # command) -- silently proceeding would retrain everything under a task
        # numbering that doesn't match what was already trained.
        if progress["task_ids_order"] != list(dataset.task_ids):
            raise ValueError(
                f"Refusing to resume run '{run_id}': the dataset's task stream doesn't match what this "
                "run was trained on (different --time-steps-per-task, or a different dataset/path). "
                "Pass the same dataset construction args this run started with."
            )

        # RTM REQ-5.18: a resumed run's checkpoint only contains the state for
        # whichever continual-learning strategy it was originally trained
        # under -- proceeding under the other one would either KeyError on a
        # missing state dict or silently start a fresh EWC/GEM+TWP state
        # instead of resuming it.
        original_strategy = ckpt["cfg"].continual_strategy
        if original_strategy != cfg.continual_strategy:
            raise ValueError(
                f"Refusing to resume run '{run_id}': it was trained under continual_strategy="
                f"{original_strategy!r}, but this call requested {cfg.continual_strategy!r}. "
                "Pass --continual-strategy matching the original run."
            )

        # RTM REQ-5.32: a run started before the ablation switches existed has
        # no `pool_proj` weights saved (see infer.py::load_model_from_checkpoint
        # for the full reasoning) -- tolerate exactly that, nothing else.
        _load_result = model.load_state_dict(ckpt["model_state"], strict=False)
        _allowed_missing = {
            f"backbone.layers.{i}.pool_proj.{p}"
            for i in range(cfg.backbone.num_hops) for p in ("weight", "bias")
        }
        _unexpected_missing = set(_load_result.missing_keys) - _allowed_missing
        if _unexpected_missing or _load_result.unexpected_keys:
            raise RuntimeError(
                f"Refusing to resume run '{run_id}': model_state mismatch -- "
                f"missing={_unexpected_missing or None}, unexpected={_load_result.unexpected_keys or None}"
            )
        if cfg.continual_strategy == "ewc_replay":
            model.ewc.load_state_dict(ckpt["ewc_state"])
            model.replay_buffer.load_state_dict(ckpt["replay_state"])
        elif cfg.continual_strategy == "gem_twp":
            model.twp.load_state_dict(ckpt["twp_state"])
            model.gem.load_state_dict(ckpt["gem_state"])
        elif cfg.continual_strategy == "agem_twp":
            model.twp.load_state_dict(ckpt["twp_state"])
            model.agem.load_state_dict(ckpt["agem_state"])
        else:  # mas_twp
            model.twp.load_state_dict(ckpt["twp_state"])
            model.mas.load_state_dict(ckpt["mas_state"])
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
    _log_architecture_summary(logger, model, cfg, run_manager.dataset_label, dataset)

    completed_task_ids = set(progress["completed_task_ids"])
    global_step = progress["global_step"]
    task_ids = list(dataset.task_ids)

    for idx, task_id in enumerate(task_ids):
        if task_id in completed_task_ids:
            continue

        full_data = dataset.get_task(task_id)
        # RTM REQ-5.37: "seed units" are node ids (task_type=="node", Elliptic,
        # unchanged) or edge indices into full_data.edge_index (task_type==
        # "edge", IBM AML's native per-transaction label -- every edge is a
        # "labeled seed", mirroring how labeled_seed_nodes already returns
        # every node for this project's own two datasets). `train_nodes`/
        # `val_nodes`/`cold_nodes` below hold whichever unit type is active;
        # names kept as-is (not renamed to "_units") to minimize the diff
        # against the rest of this already-large function -- every value they
        # hold flows straight into task_type-dispatched helpers
        # (`_build_sweep_loader`, `run_eval`, `prune_edges_for_nodes`/
        # `prune_specific_edges`) that already know which unit type to expect.
        if cfg.task_type == "edge":
            all_seed_nodes = torch.arange(full_data.edge_index.size(1), dtype=torch.long)
            unit_labels = full_data.edge_y
        else:
            all_seed_nodes = dataset.labeled_seed_nodes(task_id)
            unit_labels = full_data.y
        if all_seed_nodes.numel() == 0:
            logger.info(f"Task {task_id} ({idx + 1}/{len(task_ids)}): no labeled seed units, skipping.")
            continue

        # RTM REQ-5.30/5.37: cold-start units must be invisible for *everything*
        # that happens during this task's training (train_loader, val eval,
        # replay/GEM/A-GEM/MAS sweeps inside `_end_of_task`) -- `data` is
        # reassigned to the pruned graph for the rest of this task's block;
        # `full_data` (edges intact) is kept aside solely for the cold-start
        # evaluation itself, done once training has finished.
        if cfg.train.cold_start_ratio > 0:
            if cfg.task_type == "edge":
                train_nodes, val_nodes, cold_nodes = stratified_train_val_coldstart_split_edges(
                    all_seed_nodes.numel(), unit_labels, cfg.train.val_ratio, cfg.train.cold_start_ratio,
                    cfg.train.split_seed, task_id,
                )
                # RTM REQ-5.37: unlike node pruning, edge pruning compacts the
                # array (see prune_specific_edges's own docstring for the bug
                # this remap prevents) -- train_nodes/val_nodes, computed
                # against the ORIGINAL edge ordering above, must be remapped
                # to their new positions in the pruned edge_index before use.
                data, _edge_remap = prune_specific_edges(full_data, cold_nodes)
                train_nodes = _edge_remap[train_nodes]
                val_nodes = _edge_remap[val_nodes]
                assert (train_nodes >= 0).all() and (val_nodes >= 0).all(), \
                    "train/val edges must never overlap the cold-start reserve"
            else:
                train_nodes, val_nodes, cold_nodes = stratified_train_val_coldstart_split(
                    all_seed_nodes, unit_labels, cfg.train.val_ratio, cfg.train.cold_start_ratio,
                    cfg.train.split_seed, task_id,
                )
                data = prune_edges_for_nodes(full_data, cold_nodes)
        else:
            if cfg.task_type == "edge":
                train_nodes, val_nodes, _ = stratified_train_val_coldstart_split_edges(
                    all_seed_nodes.numel(), unit_labels, cfg.train.val_ratio, 0.0, cfg.train.split_seed, task_id,
                )
            else:
                train_nodes, val_nodes = stratified_train_val_split(
                    all_seed_nodes, unit_labels, cfg.train.val_ratio, cfg.train.split_seed, task_id
                )
            cold_nodes = all_seed_nodes[:0]
            data = full_data

        if train_nodes.numel() == 0:
            logger.info(f"Task {task_id} ({idx + 1}/{len(task_ids)}): train split empty after val hold-out, skipping.")
            continue

        # RTM REQ-5.20: recomputed fresh every task from that task's own train
        # split -- the illicit-class ratio can genuinely shift task to task.
        class_weights = None
        if cfg.train.use_class_weights:
            train_labels = data.edge_y[train_nodes] if cfg.task_type == "edge" else data.y[train_nodes]
            class_weights = compute_class_weights(train_labels, cfg.task_head.num_classes).to(device)
            logger.info(f"task {task_id} | class weights (licit, illicit) = {class_weights.tolist()}")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        is_resume_point = ckpt is not None and progress["current_task_id"] == task_id
        if is_resume_point:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = progress["next_epoch"]
        else:
            start_epoch = 0
        ckpt = None  # restoration is a one-shot event; every subsequent task starts clean

        zero_shot_metrics = None
        if start_epoch == 0:  # only ever probe a task the *first* time it's entered, never on a mid-task resume
            zero_shot_metrics = run_eval(model, data, val_nodes, cfg, device)
            if zero_shot_metrics is not None:
                logger.info(
                    f"task {task_id} | ZERO-SHOT forward-transfer (current model, before any training on "
                    f"task {task_id}) | precision={zero_shot_metrics['precision']:.4f} "
                    f"recall={zero_shot_metrics['recall']:.4f} f1={zero_shot_metrics['f1']:.4f}"
                )

        train_loader = _build_sweep_loader(data, train_nodes, cfg, shuffle=cfg.loader.shuffle)
        num_batches = len(train_loader)

        logger.info(
            f"=== Task {task_id} ({idx + 1}/{len(task_ids)}) | nodes={data.num_nodes} "
            f"edges={data.edge_index.size(1)} labeled={all_seed_nodes.numel()} "
            f"train={train_nodes.numel()} val={val_nodes.numel()} cold_start={cold_nodes.numel()} "
            f"| starting at epoch {start_epoch + 1}/{cfg.train.epochs_per_task} ==="
        )

        model.train()
        for epoch in range(start_epoch, cfg.train.epochs_per_task):
            epoch_start = time.perf_counter()
            batch_times = []
            epoch_loss = 0.0
            n_steps = 0

            for batch_idx, batch in enumerate(train_loader):
                batch_start = time.perf_counter()
                batch = batch.to(device)
                loss_value = _training_step(model, batch, optimizer, cfg, device, class_weights)

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
                current_task_id=task_id,
                next_epoch=epoch + 1,
                global_step=global_step,
                is_fully_completed=False,
                last_updated=_now_iso(),
            )
            save_checkpoint(ckpt_dir, model, optimizer, progress, cfg)

        _end_of_task(
            model, data, train_nodes, val_nodes, task_id, cfg, device, logger, log_dir, zero_shot_metrics, class_weights,
        )

        if cold_nodes.numel() > 0:
            # RTM REQ-5.30: the actual cold-start test -- `full_data` here, not
            # `data` (which stayed pruned for everything above) -- these nodes'
            # real edges are revealed to the model for the first time right now.
            model.eval()
            cold_metrics = run_eval(model, full_data, cold_nodes, cfg, device)
            if cold_metrics is not None:
                logger.info(
                    f"task {task_id} | COLD-START (n={cold_metrics['n']} nodes never seen during training, "
                    f"edges revealed for the first time) | precision={cold_metrics['precision']:.4f} "
                    f"recall={cold_metrics['recall']:.4f} f1={cold_metrics['f1']:.4f} "
                    f"accuracy={cold_metrics['accuracy']:.4f}"
                )
                append_metrics_csv(
                    log_dir / "cold_start_metrics.csv",
                    {"task_id": task_id, "timestamp": _now_iso(), **cold_metrics},
                )
            model.train()

        completed_task_ids.add(task_id)
        progress.update(
            completed_task_ids=sorted(completed_task_ids, key=lambda t: task_ids.index(t)),
            current_task_id=None,
            next_epoch=0,
            global_step=global_step,
            is_fully_completed=False,
            last_updated=_now_iso(),
        )
        save_checkpoint(ckpt_dir, model, None, progress, cfg, milestone_name=f"task_{task_id}_complete.pt")
        logger.info(f"Task {task_id} fully consolidated. Checkpoint saved -> {ckpt_dir / f'task_{task_id}_complete.pt'}")

    _check_forgetting(model, dataset, cfg, device, logger, log_dir, progress["completed_task_ids"])

    # RTM REQ-5.38: compute_paper_performance_matrix is Elliptic/RACLA-comparison-
    # specific (node classification only, see task_matrix.py) and now raises
    # NotImplementedError for edge-classification (IBM AML) checkpoints rather
    # than silently misreading data.y's node-level rollup as the real target --
    # skip the call entirely for cfg.task_type == "edge" instead of crashing
    # every IBM AML run at the very end of training.
    if cfg.train.paper_style_eval and cfg.task_type == "node":
        _log_paper_style_evaluation(dataset, ckpt_dir, device, logger)

    progress.update(is_fully_completed=True, current_task_id=None, last_updated=_now_iso())
    save_checkpoint(ckpt_dir, model, None, progress, cfg)
    logger.info(f"=== Training complete: all {len(task_ids)} tasks done ({run_id}) ===")

    return model


def _training_step(
    model: ContinualTGNN, batch, optimizer: torch.optim.Optimizer, cfg: ExperimentConfig, device,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    """RTM REQ-5.18: one optimizer step, strategy-dependent.

    `"ewc_replay"`: task loss + running EWC penalty + a freshly resampled
    replay loss, all folded into one scalar before a single backward() --
    unchanged from before this row.

    `"gem_twp"`: task loss + TWP's Eq. 8/Eq. 9 penalty terms, backward(), then
    `GEMMemory.project_gradient` overwrites every parameter's `.grad` in place
    (Eq. 7-11) *before* `optimizer.step()` -- GEM constrains the gradient
    direction itself, it does not add anything to the loss. RTM REQ-5.39:
    `cfg.disable_gem` skips the projection call (GEM's memory stays empty,
    see `_end_of_task`, so this is a genuine "no GEM" arm, not a stub);
    `cfg.disable_twp` (consumed in `_end_of_task`, not here -- see there)
    keeps the penalty terms in the loss expression below structurally, but
    they stay exactly 0.0 for the whole run since no TWP snapshot is ever
    stored.

    `"agem_twp"` (RTM REQ-5.26): identical to `"gem_twp"` except
    `AGEMMemory.project_gradient` does the projection -- a single closed-form
    step against one reference gradient from a unified memory pool, instead
    of GEM's per-completed-task constraint set.

    `"mas_twp"` (RTM REQ-5.27): no post-backward projection step at all --
    MAS's penalty is already folded into `task_loss()` alongside TWP's, so
    this branch is just a plain `backward()`, structurally closer to
    `"ewc_replay"`'s own step (minus the replay-loss addition) than to
    `"gem_twp"`/`"agem_twp"`.

    `class_weights` (RTM REQ-5.20): this task's own inverse-frequency class
    weights, applied to every cross-entropy computed in this step.
    """
    optimizer.zero_grad()
    if cfg.continual_strategy == "ewc_replay":
        loss, _ = model.task_loss(batch, include_penalty=True, class_weights=class_weights)
        loss = loss + model.replay_loss(class_weights=class_weights)
        loss.backward()
    elif cfg.continual_strategy == "gem_twp":
        loss, _ = model.task_loss(batch, include_penalty=True, class_weights=class_weights)
        loss.backward()
        if not cfg.disable_gem:  # RTM REQ-5.39: GEM-disabled arm skips the projection entirely
            model.gem.project_gradient(model, device)
    elif cfg.continual_strategy == "agem_twp":
        loss, _ = model.task_loss(batch, include_penalty=True, class_weights=class_weights)
        loss.backward()
        model.agem.project_gradient(model, device)
    else:  # mas_twp
        loss, _ = model.task_loss(batch, include_penalty=True, class_weights=class_weights)
        loss.backward()
    optimizer.step()
    return loss.item()


def _end_of_task(
    model, data, train_nodes, val_nodes, task_id, cfg, device, logger, log_dir: Path,
    zero_shot_metrics: Optional[dict] = None, class_weights: Optional[torch.Tensor] = None,
) -> None:
    model.eval()

    if cfg.continual_strategy == "ewc_replay":
        sweep_loader = _build_sweep_loader(data, train_nodes, cfg, shuffle=False)

        global_ids: list = []
        confs: list = []
        for batch in sweep_loader:
            batch = batch.to(device)
            if cfg.task_type == "edge":
                # RTM REQ-5.37: `batch.input_id` gives each confident edge's
                # position WITHIN `train_nodes` (the array handed to
                # `_build_sweep_loader`, per `LinkNeighborLoader`'s own
                # semantics -- confirmed directly, not assumed, before relying
                # on it), so `train_nodes[batch.input_id[local_pos]]` recovers
                # the GLOBAL (into this task's own edge_index) edge index,
                # exactly mirroring node mode's `batch.n_id[local_ids]` below.
                local_pos, batch_confs = model.confident_illicit_edges(batch)
                if local_pos.numel() > 0:
                    global_ids.append(train_nodes[batch.input_id[local_pos]].cpu())
                    confs.append(batch_confs.cpu())
            else:
                local_ids, batch_confs = model.confident_illicit_nodes(batch)
                if local_ids.numel() > 0:
                    global_ids.append(batch.n_id[local_ids].cpu())
                    confs.append(batch_confs.cpu())

        if global_ids:
            ids_cat = torch.cat(global_ids)
            if cfg.task_type == "edge":
                model.replay_buffer.add_candidate_edges(data, task_id, ids_cat, torch.cat(confs))
            else:
                model.replay_buffer.add_candidates(data, task_id, ids_cat, torch.cat(confs))
            logger.info(f"task {task_id} | replay buffer: admitted {ids_cat.numel()} candidates "
                        f"(buffer size now {len(model.replay_buffer)})")

        def nll_losses():
            fisher_loader = _build_sweep_loader(data, train_nodes, cfg, shuffle=True)
            for batch in fisher_loader:
                batch = batch.to(device)
                logits = model.forward(batch)
                y = model.batch_targets(batch)
                yield F.cross_entropy(logits, y, weight=class_weights)

        model.train()
        model.ewc.consolidate(nll_losses())
        logger.info(f"task {task_id} | EWC consolidated (task_count={model.ewc.task_count})")

    elif cfg.continual_strategy in ("gem_twp", "agem_twp"):  # share identical TWP consolidation, differ only in memory
        if cfg.continual_strategy == "gem_twp":
            if cfg.disable_gem:
                # RTM REQ-5.39: GEM-disabled arm -- never store a memory slot, so
                # GEMMemory stays permanently empty and `project_gradient`'s skip
                # (RTM REQ-5.39, `_train_step`) is never mismatched with leftover
                # unused memory from an earlier task.
                logger.info(f"task {task_id} | GEM disabled (--no-gem) -- skipping memory storage")
            else:
                # RTM REQ-5.18: GEM's memory is filled with a fixed per-task budget of
                # this task's own train-split units, no confidence filtering at all --
                # deliberately dumber than REQ-2.3's replay buffer, matching the paper.
                # RTM REQ-5.20: this task's own class weights are stored alongside its
                # memory so a later task's memory-loss constraint scores it correctly.
                if cfg.task_type == "edge":
                    model.gem.add_task_examples_edges(data, task_id, train_nodes, class_weights=class_weights)
                else:
                    model.gem.add_task_examples(data, task_id, train_nodes, class_weights=class_weights)
                logger.info(f"task {task_id} | GEM memory: stored this task's slot "
                            f"(total buffered examples across all tasks now {len(model.gem)})")
        else:  # agem_twp
            # RTM REQ-5.26: A-GEM's single unified pool admits this task's own
            # train-split units via reservoir sampling, competing on equal
            # footing with every example already in the pool from any prior task.
            if cfg.task_type == "edge":
                model.agem.add_task_examples_edges(data, task_id, train_nodes, class_weights=class_weights)
            else:
                model.agem.add_task_examples(data, task_id, train_nodes, class_weights=class_weights)
            logger.info(f"task {task_id} | A-GEM memory: reservoir-sampled this task's examples "
                        f"(pool size now {len(model.agem)})")

        def nll_and_topo_losses():
            importance_loader = _build_sweep_loader(data, train_nodes, cfg, shuffle=True)
            for batch in importance_loader:
                batch = batch.to(device)
                logits = model.forward(batch)
                y = model.batch_targets(batch)
                nll = F.cross_entropy(logits, y, weight=class_weights)
                topo = TWP.topology_loss(model.backbone.attention_coefficients())
                yield nll, topo

        model.train()
        if cfg.continual_strategy == "gem_twp" and cfg.disable_twp:
            # RTM REQ-5.39: TWP-disabled arm -- never store a task snapshot, so
            # `TWP.penalty()` (continual/twp.py) keeps returning exactly 0.0 for
            # the rest of the run (it already short-circuits to zero when its
            # snapshot list is empty -- no change needed there, only here).
            # Deliberately NOT applied when continual_strategy == "agem_twp":
            # these flags isolate "gem_twp"'s own two mechanisms specifically,
            # not a general TWP kill switch for every strategy that uses it.
            logger.info(f"task {task_id} | TWP disabled (--no-twp) -- skipping consolidation")
        else:
            model.twp.consolidate(nll_and_topo_losses())
            logger.info(f"task {task_id} | TWP consolidated (task_count={model.twp.task_count})")

    else:  # mas_twp -- RTM REQ-5.27: no memory/replay mechanism at all, just two
           # regularizers to consolidate; costs one extra sweep over this task's
           # train split versus gem_twp/agem_twp (one for MAS's importance, one
           # for TWP's), since the two need different scalars from the forward pass.
        def output_norms():
            importance_loader = _build_sweep_loader(data, train_nodes, cfg, shuffle=True)
            for batch in importance_loader:
                batch = batch.to(device)
                logits = model.forward(batch)
                yield logits.pow(2).sum(dim=-1).mean()

        def nll_and_topo_losses():
            importance_loader = _build_sweep_loader(data, train_nodes, cfg, shuffle=True)
            for batch in importance_loader:
                batch = batch.to(device)
                logits = model.forward(batch)
                y = model.batch_targets(batch)
                nll = F.cross_entropy(logits, y, weight=class_weights)
                topo = TWP.topology_loss(model.backbone.attention_coefficients())
                yield nll, topo

        model.train()
        model.mas.consolidate(output_norms())
        logger.info(f"task {task_id} | MAS consolidated (task_count={model.mas.task_count})")
        model.twp.consolidate(nll_and_topo_losses())
        logger.info(f"task {task_id} | TWP consolidated (task_count={model.twp.task_count})")

    _evaluate_task(model, data, val_nodes, task_id, cfg, device, logger, log_dir, zero_shot_metrics)


def _evaluate_task(
    model, data, val_nodes, task_id, cfg, device, logger, log_dir: Path,
    zero_shot_metrics: Optional[dict] = None,
) -> None:
    """RTM REQ-5.6/5.8: precision/recall/F1 on the illicit class over the task's
    held-out validation split -- never seen during this task's training,
    replay-buffer admission, or EWC Fisher estimate -- plus, when available,
    the zero-shot forward-transfer score computed on the *same* val split
    before this task was trained on at all. Appends one row to this run's
    `task_metrics.csv` (created on first use).
    """
    m = run_eval(model, data, val_nodes, cfg, device)
    if m is None:
        logger.info(f"task {task_id} | val split empty, skipping metrics")
        model.train()
        return

    logger.info(
        f"task {task_id} | VAL metrics (illicit class) | n={m['n']} (pos={m['support_pos']}, neg={m['support_neg']}) | "
        f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} accuracy={m['accuracy']:.4f} | "
        f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}"
    )

    row = {"task_id": task_id, "timestamp": _now_iso(), **m}
    if zero_shot_metrics is not None:
        row.update(
            zero_shot_precision=zero_shot_metrics["precision"],
            zero_shot_recall=zero_shot_metrics["recall"],
            zero_shot_f1=zero_shot_metrics["f1"],
            zero_shot_accuracy=zero_shot_metrics["accuracy"],
        )
    else:
        row.update(zero_shot_precision=None, zero_shot_recall=None, zero_shot_f1=None, zero_shot_accuracy=None)

    append_metrics_csv(log_dir / "task_metrics.csv", row)
    model.train()


def _check_forgetting(model, dataset, cfg, device, logger, log_dir: Path, completed_order: list) -> None:
    """RTM REQ-5.8: cheap backward-transfer proxy. Re-scores a handful of
    early "anchor" tasks -- on their own original val split, using the model
    exactly as it now stands after the entire task stream -- and compares
    against the F1 each anchor got when it was first trained (read back from
    this run's `task_metrics.csv`).
    """
    anchor_ids = cfg.train.forgetting_anchor_task_ids
    if anchor_ids is None:
        anchor_ids = completed_order[: min(3, len(completed_order))]
    if not anchor_ids:
        return

    original_f1 = {}
    metrics_csv = log_dir / "task_metrics.csv"
    if metrics_csv.exists():
        with open(metrics_csv, newline="") as f:
            for row in csv.DictReader(f):
                original_f1[row["task_id"]] = row.get("f1")

    logger.info(f"=== Forgetting check: re-evaluating anchor tasks {anchor_ids} after the full task stream ===")
    for anchor_id in anchor_ids:
        data = dataset.get_task(anchor_id)
        # RTM REQ-5.37: reproduces the EXACT same val split the main per-task
        # loop used for this task (same deterministic (split_seed, task_id)
        # generator either way) -- required for the "compare against F1 when
        # first trained" delta above to compare like with like.
        if cfg.task_type == "edge":
            all_units = torch.arange(data.edge_index.size(1), dtype=torch.long)
            _, val_nodes, _ = stratified_train_val_coldstart_split_edges(
                all_units.numel(), data.edge_y, cfg.train.val_ratio, 0.0, cfg.train.split_seed, anchor_id,
            )
        else:
            all_units = dataset.labeled_seed_nodes(anchor_id)
            _, val_nodes = stratified_train_val_split(all_units, data.y, cfg.train.val_ratio, cfg.train.split_seed, anchor_id)

        m = run_eval(model, data, val_nodes, cfg, device)
        if m is None:
            continue

        orig_raw = original_f1.get(str(anchor_id))
        orig_f1 = float(orig_raw) if orig_raw not in (None, "") else None
        delta = m["f1"] - orig_f1 if orig_f1 is not None else None

        logger.info(
            f"task {anchor_id} | forgetting check | f1_after_full_stream={m['f1']:.4f} | "
            f"f1_when_first_trained={'n/a' if orig_f1 is None else f'{orig_f1:.4f}'} | "
            f"delta={'n/a' if delta is None else f'{delta:+.4f}'}"
        )
        append_metrics_csv(log_dir / "forgetting_check.csv", {
            "task_id": anchor_id,
            "timestamp": _now_iso(),
            "f1_when_first_trained": orig_f1,
            "f1_after_full_stream": m["f1"],
            "delta_f1": delta,
            "precision_after": m["precision"],
            "recall_after": m["recall"],
        })
    model.train()


def _log_paper_style_evaluation(dataset, ckpt_dir: Path, device, logger) -> None:
    """RTM REQ-5.14: the full paper-matching M matrix + Average Performance /
    Average Forgetting, computed from this run's own just-saved
    `task_<id>_complete.pt` milestones (every task has one by the time this
    runs) and printed to the console/log -- see `task_matrix.py`'s module
    docstring for exactly how M differs from this project's own R matrix.
    """
    n = len(discover_task_checkpoints(ckpt_dir))
    logger.info(
        f"=== Paper-style evaluation: computing M ({n}x{n}, i<=j only -> "
        f"{n * (n + 1) // 2} evaluation passes) ==="
    )

    M, task_order = compute_paper_performance_matrix(dataset, ckpt_dir, device=str(device))

    logger.info("Performance Matrix M (row = test task i, col = trained-through task j; "
                "0.0000 where i > j, per the paper's own definition)")
    header = "i\\j".rjust(8) + "".join(f"{j:>10}" for j in task_order)
    logger.info(header)
    for i in task_order:
        cells = "".join(f"{M[i][j]:>10.4f}" for j in task_order)
        logger.info(f"{i:>8}{cells}")

    summary = compute_paper_summary(M, task_order)
    logger.info(
        f"Average Performance = {summary['average_performance']:.4f}   "
        f"Average Forgetting = {summary['average_forgetting']:+.4f}"
    )


if __name__ == "__main__":
    import argparse

    from .config import ExperimentConfig
    from .data.elliptic_dataset import EllipticTemporalDataset
    from .data.ibm_aml_dataset import IBMAMLTemporalDataset
    from .utils.seeding import set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["elliptic", "ibm_aml"], required=True)
    parser.add_argument("--path", required=True, help="dataset root (elliptic) or CSV path (ibm_aml)")
    parser.add_argument("--resume", action="store_true", help="resume the most recent incomplete run for this dataset")
    parser.add_argument("--run-id", default=None, help="resume/continue a specific run id instead of the latest")
    parser.add_argument(
        "--epochs-per-task", type=int, default=None,
        help=f"epochs to train on each temporal task before moving to the next (default: {ExperimentConfig().train.epochs_per_task})",
    )
    parser.add_argument(
        "--time-steps-per-task", type=int, default=1,
        help="Elliptic only: number of raw time steps grouped into one continual-learning task "
             "(default 1 = the original 49-task setup; e.g. 7 groups Elliptic's 49 time steps into 7 tasks).",
    )
    parser.add_argument(
        "--ibm-aml-n-tasks", type=int, default=None,
        help=f"IBM AML only (RTM REQ-5.36): number of continual-learning tasks the raw file's calendar "
             f"days are grouped into by cumulative-volume (not fixed-width), avoiding the near-empty "
             f"tail-day tasks a naive one-task-per-day split produces on this dataset's real timestamp "
             f"skew (default: {ExperimentConfig().ibm_aml_n_tasks}, matching this project's Elliptic T=7 "
             f"standard for cross-dataset comparability).",
    )
    parser.add_argument(
        "--ibm-aml-force-reprocess", action="store_true",
        help="IBM AML only: ignore any existing preprocessed-data cache and rebuild it from the raw CSV "
             "(RTM REQ-5.36) -- use after changing the adapter's own preprocessing logic, not routinely.",
    )
    parser.add_argument(
        "--continual-strategy", choices=["ewc_replay", "gem_twp", "agem_twp", "mas_twp"], default="ewc_replay",
        help="RTM REQ-5.18/5.26/5.27: which continual-learning strategy to train under -- 'ewc_replay' "
             "(default, Elastic Weight Consolidation + inductive episodic replay), 'gem_twp' (Gradient "
             "Episodic Memory + Topology-aware Weight Preserving), 'agem_twp' (Averaged GEM + the same TWP "
             "-- a cheaper O(1)-per-step alternative to gem_twp's O(completed tasks)-per-step cost), or "
             "'mas_twp' (Memory Aware Synapses + the same TWP -- a second additive regularization penalty "
             "in place of GEM/A-GEM's replay/projection mechanism; no buffer, no gradient projection). A "
             "resumed run must use the same strategy it started with -- its checkpoint only contains that "
             "strategy's state.",
    )
    parser.add_argument(
        "--cold-start-ratio", type=float, default=0.0,
        help="RTM REQ-5.30: fraction of each task's own labeled nodes (default 0.0 = disabled, identical "
             "to prior behavior) to reserve as 'cold-start' nodes -- their edges are pruned from the graph "
             "used for this task's training entirely, then they're scored at end-of-task against the real, "
             "unpruned graph -- a genuine 'never seen until this exact moment' inductive generalization "
             "test, logged to this run's cold_start_metrics.csv. Carved out before cfg.train.val_ratio's own "
             "split (no CLI override exists for that one either), so its meaning is unchanged either way.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RTM REQ-5.29: seed for model init/dropout/NeighborLoader sampling/replay-GEM-A-GEM sampling "
             "(default: unseeded, this project's original behavior). Does NOT affect the train/val split, "
             "which stays fixed at cfg.train.split_seed=42 (no CLI override exists for it) regardless of "
             "this flag -- so every --seed run trains on the identical per-task data split, varying only "
             "the stochastic training dynamics this flag controls.",
    )
    parser.add_argument(
        "--no-bidirectional-signed", action="store_true",
        help="RTM REQ-5.32 (ablation): disable bidirectional signed message passing (REQ-5.17) -- revert "
             "to plain single-direction (destination-only) message passing, no +-1 direction scaling.",
    )
    parser.add_argument(
        "--no-lstm", action="store_true",
        help="RTM REQ-5.32 (ablation): replace each relation's temporal LSTM summary with an order-blind "
             "mean-pool + linear projection -- tests whether message *order* within a relation matters.",
    )
    parser.add_argument(
        "--no-attention", action="store_true",
        help="RTM REQ-5.32 (ablation): replace GATv2-style softmax attention over relation summaries with "
             "uniform averaging -- tests whether the learned, query-dependent weighting beats a plain mean.",
    )
    parser.add_argument(
        "--no-gem", action="store_true",
        help="RTM REQ-5.39 (ablation, --continual-strategy gem_twp only): disable GEM's gradient projection "
             "and memory entirely -- isolates how much of gem_twp's result comes from GEM versus TWP. "
             "Combine with --no-twp for a 'neither' arm (plain per-task fine-tuning); combine with neither "
             "for the unmodified strategy ('TWP only' is --no-gem alone).",
    )
    parser.add_argument(
        "--no-twp", action="store_true",
        help="RTM REQ-5.39 (ablation, --continual-strategy gem_twp only): disable TWP's regularization "
             "penalty entirely (no task snapshot is ever consolidated) -- isolates how much of gem_twp's "
             "result comes from TWP versus GEM. 'GEM only' is --no-twp alone.",
    )
    args = parser.parse_args()

    if args.dataset == "ibm_aml" and args.time_steps_per_task != 1:
        parser.error("--time-steps-per-task is Elliptic-only for now (use --ibm-aml-n-tasks for IBM AML)")
    if args.dataset == "elliptic" and args.ibm_aml_n_tasks is not None:
        parser.error("--ibm-aml-n-tasks is IBM AML-only")
    if (args.no_gem or args.no_twp) and args.continual_strategy != "gem_twp":
        parser.error("--no-gem/--no-twp (RTM REQ-5.39) only apply to --continual-strategy gem_twp")

    cfg = ExperimentConfig()
    if args.epochs_per_task is not None:
        cfg.train.epochs_per_task = args.epochs_per_task
    cfg.elliptic_time_steps_per_task = args.time_steps_per_task
    if args.ibm_aml_n_tasks is not None:
        cfg.ibm_aml_n_tasks = args.ibm_aml_n_tasks
    cfg.continual_strategy = args.continual_strategy
    cfg.train.seed = args.seed
    if cfg.train.seed is not None:
        set_seed(cfg.train.seed)
    cfg.train.cold_start_ratio = args.cold_start_ratio
    cfg.backbone.bidirectional_signed = not args.no_bidirectional_signed
    cfg.backbone.use_lstm = not args.no_lstm
    cfg.backbone.use_attention = not args.no_attention
    cfg.disable_gem = args.no_gem
    cfg.disable_twp = args.no_twp

    # RTM REQ-5.37: task_type is derived from --dataset, not independently
    # settable -- each dataset has exactly one native entity to classify
    # (Elliptic's nodes already *are* the labeled transactions; IBM AML's
    # native label lives on the transaction/edge). See data/ibm_aml_dataset
    # .py's module docstring for the full reasoning.
    if args.dataset == "elliptic":
        cfg.task_type = "node"
        ds = EllipticTemporalDataset(
            args.path, edge_in_dim=cfg.backbone.edge_in_dim,
            time_steps_per_task=cfg.elliptic_time_steps_per_task,
        )
        cfg.backbone.node_in_dim = ds.node_in_dim
    else:
        cfg.task_type = "edge"
        ds = IBMAMLTemporalDataset(
            args.path, edge_in_dim=cfg.backbone.edge_in_dim, node_in_dim=cfg.backbone.node_in_dim,
            n_tasks=cfg.ibm_aml_n_tasks, force_reprocess=args.ibm_aml_force_reprocess,
        )

    model = ContinualTGNN(cfg)
    run_continual_training(model, ds, cfg, dataset_key=args.dataset, resume=args.resume, run_id=args.run_id)
