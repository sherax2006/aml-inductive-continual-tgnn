"""Full Task Performance Matrix R (RTM REQ-5.12).

R[i, j] = the model's performance on task j's held-out val split, evaluated
using the model exactly as it stood right after finishing training on task i
-- for *every* (i, j) pair. This is the standard continual-learning "task
performance matrix" (Lopez-Paz & Ranzato, GEM, 2017, and the many papers that
followed it) that formal ACC/BWT/FWT scalars are derived from -- broader than
what `train.py` computes inline during training: the diagonal R[i,i] (per-task
val metrics), the super-diagonal R[i,i+1] (one-step-ahead zero-shot forward
transfer), and R[T, anchor] for a handful of anchor tasks (the cheap
forgetting-check proxy). This module fills in the rest -- the full upper
triangle (forward transfer to *every* future task, not just the next one) and
full lower triangle (forgetting on *every* past task, not just 3 anchors).

Built entirely post-hoc from a completed run's per-task milestone checkpoints
(`task_<id>_complete.pt`) -- no retraining needed, since one is already saved
at the end of every task (see `aml_tgnn/utils/checkpoint.py`). Cost: T
checkpoint loads + T^2 evaluation (forward-only, no backward/optimizer) passes.
This is exactly the O(T^2) cost `train.py`'s design explicitly avoids paying
*during* training (re-evaluating every prior task after every new one would
slow every single task down); computing it once, after training, from
checkpoints that already exist, is a different cost profile and fine to run
on demand -- though for Elliptic's default 49-task granularity that's still
2,401 evaluation passes, worth expecting to take a while (grouping tasks via
`--time-steps-per-task`, RTM REQ-5.9, cuts this down quadratically).

Every cell just evaluates the checkpoint-i model directly (RTM REQ-5.15:
parameter isolation was removed, so there is no per-task head to route to --
one shared classification head, always. R[i, j] for j > i is therefore a
*genuine* zero-shot evaluation of the actual model state, not an approximation
via a "most recently trained task's head" stand-in the way it was before that
removal.).

    python -m aml_tgnn.task_matrix --dataset elliptic --path elliptic_bitcoin_dataset \\
        --run-dir Checkpoints/Elliptic/run_.../ --out-prefix task_matrix

This module also provides `compute_paper_performance_matrix` (RTM REQ-5.14) --
a *separate*, deliberately more restrictive matrix matching a specific
published paper's own evaluation protocol exactly, rather than this project's
richer `compute_task_matrix`/R above. Differences, both intentional:
  - Axis convention is the paper's own: M[i, j] = performance on test task i
    using the model trained through task j (row = test task, column = train
    checkpoint) -- the *opposite* of R[i, j]'s (row = train checkpoint,
    column = test task). Kept as its own function with its own orientation,
    not a transposed view of R, specifically so the code maps directly onto
    the paper's M_{i,j} formula for citation purposes.
  - Only defined for i <= j (test task already trained on by checkpoint j);
    M[i, j] = 0.0 for i > j *by the paper's own definition* -- not computed
    via zero-shot the way R's upper triangle is, just zero. This means it
    carries strictly less information than R (no forward-transfer signal at
    all), and costs less to compute (T(T+1)/2 passes, not T^2) as a result.
  - Metric is micro-F1, not per-class F1/recall -- see the paper-comparison
    discussion in ARCHITECTURE.md for why this is *not* actually more
    imbalance-robust than accuracy (they're mathematically identical for
    single-label classification), despite that being the paper's stated
    motivation for choosing it. Implemented anyway, faithfully, because the
    point here is matching their protocol exactly for a citable comparison,
    not arguing they should have done it differently.
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .data.split import (
    stratified_train_val_coldstart_split,
    stratified_train_val_coldstart_split_edges,
    stratified_train_val_split,
)
from .infer import load_model_from_checkpoint
from .utils.metrics import run_eval


def discover_task_checkpoints(run_dir: Path) -> Dict[int, Path]:
    """Maps task_id -> its `task_<id>_complete.pt` milestone path. A run
    interrupted partway through simply yields a matrix over whichever tasks
    it actually finished -- nothing here assumes the run is complete.
    """
    out = {}
    for p in run_dir.glob("task_*_complete.pt"):
        m = re.match(r"task_(-?\d+)_complete\.pt$", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _build_val_splits(dataset, task_order: List[int], val_ratio: float, split_seed: int) -> Dict[int, torch.Tensor]:
    """Each task's val split, fixed once and reused as that task's "test set"
    everywhere it's evaluated -- identical to what train.py scored during
    training, since it's the same deterministic (task_id, split_seed)-keyed split.
    """
    val_splits: Dict[int, torch.Tensor] = {}
    for j in task_order:
        data = dataset.get_task(j)
        all_nodes = dataset.labeled_seed_nodes(j)
        _, val_nodes = stratified_train_val_split(all_nodes, data.y, val_ratio, split_seed, j)
        val_splits[j] = val_nodes
    return val_splits


def _build_disjoint_val_splits(
    dataset, task_order: List[int], val_ratio: float, cold_start_ratio: float, split_seed: int
) -> Dict[int, torch.Tensor]:
    """RTM REQ-5.35: each task's val split, but from the same three-way
    (train/val/cold) call `train.py` itself used during training -- disjoint
    from `_build_coldstart_splits`'s cold-start nodes by construction, unlike
    `_build_val_splits`'s plain two-way split (which is carved from the SAME
    labeled-node pool `stratified_train_val_coldstart_split` starts from,
    before cold-start nodes are set aside first per Section 4.3 -- so its
    "val" ends up containing every cold-start node too, about 75% of it in
    practice on Elliptic T=7). A "normal vs. cold-start" comparison (Table 4)
    needs these two populations to be genuinely disjoint; a plain "F1 on
    held-out validation data" claim (Table 3, unqualified) does not, so this
    is intentionally a separate, additional matrix -- not a change to what
    `f"{metric}"` already means for existing callers/tables.
    """
    val_splits: Dict[int, torch.Tensor] = {}
    for j in task_order:
        data = dataset.get_task(j)
        all_nodes = dataset.labeled_seed_nodes(j)
        _, val_nodes, _ = stratified_train_val_coldstart_split(
            all_nodes, data.y, val_ratio, cold_start_ratio, split_seed, j
        )
        val_splits[j] = val_nodes
    return val_splits


def _build_coldstart_splits(
    dataset, task_order: List[int], val_ratio: float, cold_start_ratio: float, split_seed: int
) -> Dict[int, torch.Tensor]:
    """RTM REQ-5.30: each task's cold-start reserve, re-derived the exact same
    deterministic way `train.py`'s own per-task loop did during training
    (same `(task_id, split_seed)`-keyed generator) -- these are the nodes
    whose edges were pruned from the graph while every checkpoint below was
    being trained, so evaluating against them (using `dataset.get_task(j)`'s
    normal, *unpruned* graph -- pruning was only ever transient inside
    `train.py`'s own loop, never persisted) is the genuine held-out-node test,
    for every (i, j) pair, not just the single i==j point `train.py` itself
    logs per task.
    """
    cold_splits: Dict[int, torch.Tensor] = {}
    for j in task_order:
        data = dataset.get_task(j)
        all_nodes = dataset.labeled_seed_nodes(j)
        _, _, cold_nodes = stratified_train_val_coldstart_split(
            all_nodes, data.y, val_ratio, cold_start_ratio, split_seed, j
        )
        cold_splits[j] = cold_nodes
    return cold_splits


def _build_val_splits_edges(dataset, task_order: List[int], val_ratio: float, split_seed: int) -> Dict[int, torch.Tensor]:
    """RTM REQ-5.38: edge-classification analogue of `_build_val_splits` --
    every "seed unit" is an edge (0..num_edges-1 into `data.edge_index`), not
    a node id, mirroring exactly how `train.py`'s own per-task loop builds
    `all_seed_nodes`/`unit_labels` for `cfg.task_type == "edge"` (every
    transaction is a labeled seed; there is no `labeled_seed_nodes`-style
    filtering step needed since IBM AML's `edge_y` has no "unlabeled"
    sentinel). `cold_start_ratio=0.0` here reduces
    `stratified_train_val_coldstart_split_edges` to a plain two-way split --
    the exact same call `train.py` itself makes when `cold_start_ratio == 0`
    (see its per-task loop, the `else` branch under `cfg.task_type == "edge"`).
    """
    val_splits: Dict[int, torch.Tensor] = {}
    for j in task_order:
        data = dataset.get_task(j)
        num_edges = data.edge_index.size(1)
        _, val_edges, _ = stratified_train_val_coldstart_split_edges(
            num_edges, data.edge_y, val_ratio, 0.0, split_seed, j
        )
        val_splits[j] = val_edges
    return val_splits


def _build_disjoint_val_splits_edges(
    dataset, task_order: List[int], val_ratio: float, cold_start_ratio: float, split_seed: int
) -> Dict[int, torch.Tensor]:
    """RTM REQ-5.38: edge-classification analogue of `_build_disjoint_val_splits`
    -- same rationale (the plain two-way `_build_val_splits_edges` call above
    is carved from the full edge pool *before* cold-start edges are set aside,
    so its "val" would contain most cold-start edges too; this three-way call
    carves cold-start out first, giving a val set genuinely disjoint from
    `_build_coldstart_splits_edges`'s cold edges).
    """
    val_splits: Dict[int, torch.Tensor] = {}
    for j in task_order:
        data = dataset.get_task(j)
        num_edges = data.edge_index.size(1)
        _, val_edges, _ = stratified_train_val_coldstart_split_edges(
            num_edges, data.edge_y, val_ratio, cold_start_ratio, split_seed, j
        )
        val_splits[j] = val_edges
    return val_splits


def _build_coldstart_splits_edges(
    dataset, task_order: List[int], val_ratio: float, cold_start_ratio: float, split_seed: int
) -> Dict[int, torch.Tensor]:
    """RTM REQ-5.38: edge-classification analogue of `_build_coldstart_splits`.
    `dataset.get_task(j)` always returns the natural, full graph -- edges are
    never removed at the dataset level, only ever transiently pruned inside
    `train.py`'s own per-task loop -- so evaluating checkpoint i against
    task j's cold edges here needs no `prune_specific_edges`/remap step at
    all: these edge indices are computed directly against the SAME (already
    unpruned) `edge_index` ordering `dataset.get_task(j)` returns, exactly
    mirroring how node-mode cold-start evaluation already works.
    """
    cold_splits: Dict[int, torch.Tensor] = {}
    for j in task_order:
        data = dataset.get_task(j)
        num_edges = data.edge_index.size(1)
        _, _, cold_edges = stratified_train_val_coldstart_split_edges(
            num_edges, data.edge_y, val_ratio, cold_start_ratio, split_seed, j
        )
        cold_splits[j] = cold_edges
    return cold_splits


def compute_task_matrix(
    dataset,
    run_dir: Path,
    device: str = "cpu",
    metric_names: Tuple[str, ...] = ("f1", "recall"),
    model_loader=load_model_from_checkpoint,
) -> Tuple[Dict[str, Dict[int, Dict[int, Optional[float]]]], List[int]]:
    """Returns ({metric_name: {i: {j: value}}}, task_order).

    `model_loader` (RTM REQ-5.25): a `(checkpoint_path, device) -> (model, cfg)`
    callable, defaulting to `ContinualTGNN`'s own loader. `run_eval` below only
    ever calls `model.forward(batch)` -- already model-agnostic -- so passing
    `aml_tgnn.racla.checkpoint.load_racla_model_from_checkpoint` here is enough
    to build the exact same kind of matrix for a `RACLAModel` run, with zero
    other changes: the point is a genuinely controlled, identical-protocol
    comparison between architectures, not a parallel/duplicated evaluation path.

    RTM REQ-5.30: if the run being loaded was trained with `--cold-start-ratio
    > 0`, this automatically ALSO computes a full cold-start matrix -- every
    checkpoint i evaluated against every task j's held-out cold-start nodes,
    not just the single i==j point `train.py` itself logs per task -- and adds
    it to the *same* returned dict under `f"cold_{metric_name}"` keys, so every
    existing caller (printing, CSV export, ACC/BWT/FWT summaries) picks it up
    for free with no changes of their own. Runs saved before this row existed
    have no `cold_start_ratio` field at all on their persisted cfg (`getattr`
    with a `0.0` default handles that -- accessing the plain attribute would
    raise `AttributeError` on those older checkpoints, since dataclass field
    defaults only apply at construction time, not to an object unpickled
    straight from an older class version's saved `__dict__`).

    RTM REQ-5.38: `first_cfg.task_type` (also `getattr`-guarded, `"node"`
    default, for the same reason -- checkpoints saved before REQ-5.37 added
    the field have no `task_type` attribute at all) picks node-indexed splits
    (`_build_val_splits`/`_build_coldstart_splits`/`_build_disjoint_val_splits`,
    Elliptic, unchanged) or edge-indexed ones
    (`_build_val_splits_edges`/`_build_coldstart_splits_edges`/
    `_build_disjoint_val_splits_edges`, IBM AML). Nothing else below this
    branch needs to know which kind of split it's holding: `run_eval` already
    dispatches on `cfg.task_type` itself (RTM REQ-5.37) for both loader choice
    and target extraction, so the i,j evaluation loop is unchanged either way.
    """
    ckpts = discover_task_checkpoints(run_dir)
    task_order = sorted(ckpts.keys())
    if not task_order:
        raise FileNotFoundError(f"no task_<id>_complete.pt milestones found under {run_dir}")

    first_cfg = torch.load(ckpts[task_order[0]], map_location="cpu", weights_only=False)["cfg"]
    task_type = getattr(first_cfg, "task_type", "node")
    cold_start_ratio = getattr(first_cfg.train, "cold_start_ratio", 0.0)

    if task_type == "edge":
        val_splits = _build_val_splits_edges(dataset, task_order, first_cfg.train.val_ratio, first_cfg.train.split_seed)
        cold_splits = disjoint_val_splits = None
        if cold_start_ratio > 0:
            cold_splits = _build_coldstart_splits_edges(
                dataset, task_order, first_cfg.train.val_ratio, cold_start_ratio, first_cfg.train.split_seed
            )
            disjoint_val_splits = _build_disjoint_val_splits_edges(
                dataset, task_order, first_cfg.train.val_ratio, cold_start_ratio, first_cfg.train.split_seed
            )
    else:
        val_splits = _build_val_splits(dataset, task_order, first_cfg.train.val_ratio, first_cfg.train.split_seed)
        cold_splits = disjoint_val_splits = None
        if cold_start_ratio > 0:
            cold_splits = _build_coldstart_splits(
                dataset, task_order, first_cfg.train.val_ratio, cold_start_ratio, first_cfg.train.split_seed
            )
            disjoint_val_splits = _build_disjoint_val_splits(
                dataset, task_order, first_cfg.train.val_ratio, cold_start_ratio, first_cfg.train.split_seed
            )

    all_metric_names = list(metric_names)
    if cold_splits is not None:
        all_metric_names += [f"cold_{m}" for m in metric_names] + [f"disjoint_val_{m}" for m in metric_names]
    matrix = {m: {i: {} for i in task_order} for m in all_metric_names}

    for i in task_order:
        model, cfg = model_loader(str(ckpts[i]), device=device)
        for j in task_order:
            data = dataset.get_task(j)
            result = run_eval(model, data, val_splits[j], cfg, device)
            for m in metric_names:
                matrix[m][i][j] = result[m] if result is not None else None

            if cold_splits is not None:
                cold_result = run_eval(model, data, cold_splits[j], cfg, device)
                disjoint_val_result = run_eval(model, data, disjoint_val_splits[j], cfg, device)
                for m in metric_names:
                    matrix[f"cold_{m}"][i][j] = cold_result[m] if cold_result is not None else None
                    matrix[f"disjoint_val_{m}"][i][j] = disjoint_val_result[m] if disjoint_val_result is not None else None

    return matrix, task_order


def compute_paper_performance_matrix(
    dataset, run_dir: Path, device: str = "cpu", model_loader=load_model_from_checkpoint
) -> Tuple[Dict[int, Dict[int, float]], List[int]]:
    """M[i, j] = micro-F1 on test task i's val split, using the model as it
    stood right after training through task j -- for i <= j (both meant
    positionally in the training stream: `task_order`'s index, since grouped
    Elliptic task ids like 1/8/15/... aren't numerically contiguous). M[i, j]
    = 0.0 for i > j, exactly per the paper's own definition (not zero-shot
    evaluated, just zero) -- see this module's docstring for how this
    deliberately differs from `compute_task_matrix`'s R.

    Cost: T(T+1)/2 eval passes (only the defined half), vs. R's T^2 -- still
    real for large T, since it's still one full checkpoint-load + eval loop.

    `model_loader`: see `compute_task_matrix`'s own docstring (RTM REQ-5.25) --
    same pluggable-loader mechanism, so this also works unchanged for a
    `RACLAModel` run's checkpoints.

    RTM REQ-5.38: deliberately NOT extended to edge-classification (IBM AML)
    checkpoints -- this matrix exists specifically to match a published
    paper's own node-classification protocol for an Elliptic/RACLA
    comparison (see this module's docstring), not as part of IBM AML's own
    campaign needs, unlike `compute_task_matrix`'s R (which RTM REQ-5.38 does
    extend, since every table this project's own campaign reports is built
    from R). Raises clearly rather than silently misreading `data.y`'s
    node-level rollup as if it were the real per-transaction target.
    """
    ckpts = discover_task_checkpoints(run_dir)
    task_order = sorted(ckpts.keys())
    if not task_order:
        raise FileNotFoundError(f"no task_<id>_complete.pt milestones found under {run_dir}")

    first_cfg = torch.load(ckpts[task_order[0]], map_location="cpu", weights_only=False)["cfg"]
    if getattr(first_cfg, "task_type", "node") == "edge":
        raise NotImplementedError(
            "compute_paper_performance_matrix is Elliptic/RACLA-comparison-specific "
            "(node classification only) and has not been extended for edge-classification "
            "(IBM AML) checkpoints -- use compute_task_matrix instead, which RTM REQ-5.38 "
            "extended for both task types."
        )
    val_splits = _build_val_splits(dataset, task_order, first_cfg.train.val_ratio, first_cfg.train.split_seed)

    M: Dict[int, Dict[int, float]] = {i: {} for i in task_order}
    for pos_j, j in enumerate(task_order):  # j = training checkpoint (column)
        model, cfg = model_loader(str(ckpts[j]), device=device)
        for pos_i, i in enumerate(task_order):  # i = test task (row)
            if pos_i > pos_j:
                M[i][j] = 0.0
                continue
            data = dataset.get_task(i)
            result = run_eval(model, data, val_splits[i], cfg, device)
            M[i][j] = result["micro_f1"] if result is not None else 0.0

    return M, task_order


def compute_paper_summary(M: Dict[int, Dict[int, float]], task_order: List[int]) -> Dict[str, float]:
    """Average Performance and Average Forgetting, exactly as derived from the
    paper's own M (row = test task i, column = train checkpoint j):
      Average Performance = mean_i M[i, k]                (k = last-trained task)
      Average Forgetting  = mean_{i<k} (M[i,i] - M[i,k])   (positive = forgetting
                                                             occurred; negative =
                                                             later tasks helped)
    Note the sign convention is the *opposite* of this module's own BWT (which
    is final-minus-initial) -- "forgetting" here is initial-minus-final, so a
    positive number reads naturally as "how much was forgotten."
    """
    k = task_order[-1]
    avg_performance = sum(M[i][k] for i in task_order) / len(task_order)
    forgetting_terms = [M[i][i] - M[i][k] for i in task_order[:-1]]
    avg_forgetting = sum(forgetting_terms) / len(forgetting_terms) if forgetting_terms else 0.0
    return {"average_performance": avg_performance, "average_forgetting": avg_forgetting}


def _print_matrix(metric_name: str, matrix: Dict[int, Dict[int, Optional[float]]], task_order: List[int]) -> None:
    print(f"\n=== Task Performance Matrix R -- {metric_name.upper()} ===")
    print("(row = trained through task i, col = tested on task j)")
    print("i\\j".rjust(8) + "".join(f"{j:>10}" for j in task_order))
    for i in task_order:
        cells = "".join(
            f"{matrix[i][j]:>10.4f}" if matrix[i][j] is not None else f"{'n/a':>10}" for j in task_order
        )
        print(f"{i:>8}{cells}")


def _print_paper_matrix(M: Dict[int, Dict[int, float]], task_order: List[int]) -> None:
    print("\n=== Paper-style Performance Matrix M (micro-F1) ===")
    print("(row = test task i, col = trained-through task j; 0.0000 where i > j, per the paper's own definition)")
    print("i\\j".rjust(8) + "".join(f"{j:>10}" for j in task_order))
    for i in task_order:
        cells = "".join(f"{M[i][j]:>10.4f}" for j in task_order)
        print(f"{i:>8}{cells}")


def compute_summary_metrics(
    matrix: Dict[int, Dict[int, Optional[float]]], task_order: List[int]
) -> Dict[str, Optional[float]]:
    """Standard GEM-style aggregate scalars derived from R (Lopez-Paz &
    Ranzato, 2017) -- see the earlier discussion comparing this project's
    numbers against a published paper's ACC/BWT/FWT table:

      ACC = mean_j R[T, j]                 -- final model, scored on every task
      BWT = mean_{i<T} (R[T, i] - R[i, i])  -- forgetting; negative = forgotten,
                                                positive = later tasks helped
      FWT = mean over consecutive (i, next(i)) in task_order of R[i, next(i)]
                                             -- one-step-ahead zero-shot transfer
      MEAN = mean over every cell in R      -- simple grand mean, all i, j

    `T` is `task_order[-1]` (the last-trained task); "consecutive" for FWT
    means positionally adjacent in `task_order`, not numerically adjacent
    task ids -- grouped Elliptic tasks (e.g. ids 1, 8, 15, ...) aren't
    contiguous integers. `None` cells (empty val split) are dropped from
    whichever average they'd fall into; an average with no valid cells at
    all comes back `None` rather than raising.
    """
    def _mean(vals) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    final_task = task_order[-1]

    acc = _mean(matrix[final_task][j] for j in task_order)

    bwt = _mean(
        matrix[final_task][i] - matrix[i][i]
        for i in task_order[:-1]
        if matrix[final_task][i] is not None and matrix[i][i] is not None
    )

    fwt = _mean(
        matrix[task_order[k]][task_order[k + 1]]
        for k in range(len(task_order) - 1)
    )

    grand_mean = _mean(matrix[i][j] for i in task_order for j in task_order)

    return {"ACC": acc, "BWT": bwt, "FWT": fwt, "MEAN": grand_mean}


def compute_racla_summary_metrics(
    matrix: Dict[int, Dict[int, Optional[float]]], task_order: List[int]
) -> Dict[str, Optional[float]]:
    """RTM REQ-5.24: ACC/BWT/FWT computed via RACLA's own equations (Zhang, Liu,
    Lan, Chen, Chen & Zhang, "RACLA: Role-aware continual learning for robust
    AML detection", Expert Systems With Applications 299 (2026) 130136, Sec.
    5.1 -- see BasePaper1_RACLA.md for the source formulas), as an explicitly
    configurable *alternative* to this project's own `compute_summary_metrics`,
    not a replacement -- both are always available side by side.

      ACC = (1/T) * sum_i A_{T,i}                          -- IDENTICAL to this
                                                                project's own ACC.
      BWT = (1/(T-1)) * sum_{i=1}^{T-1} (A_{T,i} - A_{i,i}) -- IDENTICAL to this
                                                                project's own BWT.
      FWT = (1/(T-1)) * sum_{i=2}^{T} (A_{i,i} - A_{1,i})   -- DIFFERENT: RACLA
                                                                compares every
                                                                task's own
                                                                post-training
                                                                accuracy against
                                                                *task 1's*
                                                                accuracy on that
                                                                same task (a fixed
                                                                baseline). This
                                                                project's own FWT
                                                                (compute_summary_
                                                                metrics) is a raw
                                                                one-step-ahead
                                                                zero-shot value
                                                                with no baseline
                                                                subtracted at all
                                                                -- a third,
                                                                simpler variant,
                                                                not RACLA's and
                                                                not the classic
                                                                Lopez-Paz &
                                                                Ranzato (2017)
                                                                random-init-
                                                                baseline version
                                                                either.

    A_{i,i} = R[i][i] (diagonal: task i's accuracy right after training it).
    A_{1,i} = R[task_order[0]][i] (the row for the *first*-trained task, at
    column i -- the model as it stood after only task 1, zero-shot-evaluated
    on every later task). A_{T,i} = R[task_order[-1]][i] (row for the final/
    last-trained task). "T" and "1" here are *positions* in `task_order`, not
    raw task ids -- matches how `compute_summary_metrics` already handles
    grouped, non-contiguous Elliptic task ids (e.g. 1, 8, 15, ...).

    Computed entirely from the *existing* R matrix -- no extra evaluation
    passes needed, since R already contains every (i, j) cell RACLA's formula
    reads from.
    """
    def _mean(vals) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    first_task = task_order[0]
    final_task = task_order[-1]

    acc = _mean(matrix[final_task][j] for j in task_order)

    bwt = _mean(
        matrix[final_task][i] - matrix[i][i]
        for i in task_order[:-1]
        if matrix[final_task][i] is not None and matrix[i][i] is not None
    )

    fwt = _mean(
        matrix[i][i] - matrix[first_task][i]
        for i in task_order[1:]
        if matrix[i][i] is not None and matrix[first_task][i] is not None
    )

    return {"ACC": acc, "BWT": bwt, "FWT": fwt}


def _print_racla_summary(metric_name: str, summary: Dict[str, Optional[float]]) -> None:
    def _fmt(v: Optional[float], signed: bool = False) -> str:
        if v is None:
            return "n/a"
        return f"{v:+.4f}" if signed else f"{v:.4f}"

    print(f"--- RACLA-formula Summary ({metric_name.upper()}) --- "
          f"ACC={_fmt(summary['ACC'])}  BWT={_fmt(summary['BWT'], signed=True)}  "
          f"FWT={_fmt(summary['FWT'], signed=True)}  "
          f"(ACC/BWT identical to the default formula; FWT differs -- see "
          f"compute_racla_summary_metrics docstring / BasePaper1_RACLA.md)")


def _print_summary(metric_name: str, summary: Dict[str, Optional[float]]) -> None:
    def _fmt(v: Optional[float], signed: bool = False) -> str:
        if v is None:
            return "n/a"
        return f"{v:+.4f}" if signed else f"{v:.4f}"

    print(f"--- Summary ({metric_name.upper()}) --- "
          f"ACC={_fmt(summary['ACC'])}  BWT={_fmt(summary['BWT'], signed=True)}  "
          f"FWT={_fmt(summary['FWT'])}  overall mean (all cells)={_fmt(summary['MEAN'])}")


def _write_matrix_csv(path: Path, matrix: Dict[int, Dict[int, Optional[float]]], task_order: List[int]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trained_through_task"] + [f"test_task_{j}" for j in task_order])
        for i in task_order:
            writer.writerow([i] + [matrix[i][j] for j in task_order])


if __name__ == "__main__":
    from .data.elliptic_dataset import EllipticTemporalDataset
    from .data.ibm_aml_dataset import IBMAMLTemporalDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["elliptic", "ibm_aml"], required=True)
    parser.add_argument("--path", required=True, help="dataset root (elliptic) or CSV path (ibm_aml)")
    parser.add_argument("--run-dir", required=True, help="a run's checkpoint directory, e.g. Checkpoints/Elliptic/run_.../")
    parser.add_argument("--out-prefix", default=None, help="if set, writes <prefix>_f1.csv and <prefix>_recall.csv")
    parser.add_argument(
        "--paper-style", action="store_true",
        help="also compute and print the paper-matching M matrix (micro-F1, i<=j only) "
             "plus Average Performance / Average Forgetting -- see task_matrix.py's module docstring",
    )
    parser.add_argument(
        "--racla-metrics", action="store_true",
        help="RTM REQ-5.24: also print ACC/BWT/FWT computed using RACLA's own equations "
             "(Zhang et al. 2026, BasePaper1_RACLA.md) alongside -- not instead of -- this "
             "project's default formulas (still always printed). ACC/BWT are numerically "
             "identical to the default; FWT differs (task-1-baseline vs. one-step-ahead "
             "zero-shot with no baseline).",
    )
    parser.add_argument(
        "--model-type", choices=["continual_tgnn", "racla"], default="continual_tgnn",
        help="RTM REQ-5.25: which model class the checkpoints in --run-dir hold -- this "
             "project's own ContinualTGNN (default), or the RACLA baseline reimplementation "
             "(aml_tgnn/racla/). Both use the identical matrix/summary machinery below; only "
             "which checkpoint loader runs differs.",
    )
    args = parser.parse_args()

    if args.model_type == "racla":
        from .racla.checkpoint import load_racla_model_from_checkpoint as model_loader
    else:
        model_loader = load_model_from_checkpoint

    run_dir = Path(args.run_dir)
    ckpts = discover_task_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"no task_<id>_complete.pt milestones found under {run_dir}")
    task_order = sorted(ckpts.keys())
    first_cfg = torch.load(ckpts[task_order[0]], map_location="cpu", weights_only=False)["cfg"]

    if args.dataset == "elliptic":
        ds = EllipticTemporalDataset(
            args.path, edge_in_dim=first_cfg.backbone.edge_in_dim,
            time_steps_per_task=first_cfg.elliptic_time_steps_per_task,
        )
    else:
        ds = IBMAMLTemporalDataset(
            args.path, edge_in_dim=first_cfg.backbone.edge_in_dim, node_in_dim=first_cfg.backbone.node_in_dim,
            n_tasks=getattr(first_cfg, "ibm_aml_n_tasks", 7),
        )

    has_cold_start = getattr(first_cfg.train, "cold_start_ratio", 0.0) > 0
    n_passes = len(task_order) ** 2 * (3 if has_cold_start else 1)
    print(f"Found {len(task_order)} task checkpoints -> computing a {len(task_order)}x{len(task_order)} "
          f"matrix ({n_passes} evaluation passes"
          f"{', includes cold-start and disjoint-val matrices (RTM REQ-5.30/5.35)' if has_cold_start else ''})...")
    matrices, task_order = compute_task_matrix(ds, run_dir, device=first_cfg.train.device, model_loader=model_loader)

    for metric_name, matrix in matrices.items():
        _print_matrix(metric_name, matrix, task_order)
        _print_summary(metric_name, compute_summary_metrics(matrix, task_order))
        if args.racla_metrics:
            _print_racla_summary(metric_name, compute_racla_summary_metrics(matrix, task_order))
        if args.out_prefix:
            out_path = Path(f"{args.out_prefix}_{metric_name}.csv")
            _write_matrix_csv(out_path, matrix, task_order)
            print(f"written -> {out_path}")

    if args.paper_style:
        print(f"\nComputing paper-style matrix ({len(task_order) * (len(task_order) + 1) // 2} evaluation passes)...")
        M, task_order = compute_paper_performance_matrix(ds, run_dir, device=first_cfg.train.device, model_loader=model_loader)
        _print_paper_matrix(M, task_order)
        paper_summary = compute_paper_summary(M, task_order)
        print(f"--- Average Performance = {paper_summary['average_performance']:.4f}   "
              f"Average Forgetting = {paper_summary['average_forgetting']:+.4f} ---")
        if args.out_prefix:
            out_path = Path(f"{args.out_prefix}_paper_micro_f1.csv")
            _write_matrix_csv(out_path, M, task_order)
            print(f"written -> {out_path}")
