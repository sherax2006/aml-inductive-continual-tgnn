"""Gradient Episodic Memory (Lopez-Paz & Ranzato, NeurIPS 2017) -- RTM REQ-5.18.

Verified against the actual paper (not recollection) before implementing.

Memory (Sec. 3): one slot per task, populated with a fixed per-task budget
`m = memory_budget // num_tasks_so_far` -- "the last m examples from each
task" in the paper's streaming setting; the closest analogue in this project's
per-task (not per-example) training loop is an even stride across the task's
labeled nodes, taken once at end-of-task. Deliberately **not** curated by model
confidence, unlike this project's own replay buffer (REQ-2.3) -- the paper's
own description, and the one the user asked to match: "the memory is filled
without any smart allocation of replay."

Projection (Eq. 6-11): at every training step, the current gradient `g` must
satisfy `<g, g_k> >= 0` for every previously-completed task `k` (don't let this
update increase that task's memory loss). If violated, project `g` onto the
closest feasible gradient by solving the dual QP (Eq. 11) -- a QP over
`t-1` variables (completed task count), not the parameter count, so it costs
essentially nothing to solve even though computing each `g_k` (one extra
backward pass per completed task, every step) is the real, and real-real,
cost: it grows linearly with how far into the task stream training is.
Implemented via projected gradient descent rather than a `quadprog`/`cvxopt`
dependency (this project already hit an unavailable-wheel wall with
`pyg-lib`, see memory `env_pyg_lib_incompatible.md` -- avoided here since the
QP is tiny and PGD with an eigenvalue-safe step size is exact enough).
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data

from ..config import GEMConfig
from ..data.temporal_loader import build_temporal_link_loader, build_temporal_neighbor_loader


def project_gradient_qp(g: torch.Tensor, G: torch.Tensor, iters: int, step_fraction: float, margin: float) -> torch.Tensor:
    """Solve Eq. 11's dual QP (minimize_v 1/2 v^T G G^T v + g^T G^T v, v>=0) via
    projected gradient descent, then recover the projected primal gradient
    g~ = G^T v* + g (Sec. 3, after Eq. 11).

    g: [p] current gradient (flattened over all optimized parameters).
    G: [t-1, p] one row per previously-completed task's memory-loss gradient.
    Returns: [p] projected gradient.
    """
    if G.size(0) == 0:
        return g

    P = G @ G.T                      # [t-1, t-1]
    q = G @ g                        # [t-1]
    lipschitz = torch.linalg.eigvalsh(P)[-1].clamp(min=1e-8)
    step = step_fraction / lipschitz

    v = torch.zeros(G.size(0), device=g.device, dtype=g.dtype)
    for _ in range(iters):
        v = (v - step * (P @ v + q)).clamp(min=0.0)

    # Paper: "adding a small constant gamma >= 0 to v* biased the gradient
    # projection to updates that favoured beneficial backwards transfer."
    v = v + margin
    return G.T @ v + g


def _flatten_grads(params: List[nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in params
    ])


def _assign_grads(params: List[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset:offset + n].view_as(p).clone()
        offset += n


class GEMMemory:
    def __init__(self, cfg: GEMConfig):
        self.cfg = cfg
        self.task_memories: Dict[object, List[Data]] = {}
        self.task_class_weights: Dict[object, torch.Tensor] = {}
        self._task_order: List[object] = []

    def add_task_examples(
        self, data: Data, task_id, node_local_ids: torch.Tensor, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        """Call once, right after finishing a task: populates this task's fixed
        memory slot and shrinks every earlier task's slot to match the new,
        smaller per-task budget (Sec. 3: "if the number of total tasks T is
        unknown, we can gradually reduce the value of m as we observe new
        tasks"). No confidence filtering -- unlike REQ-2.3's replay buffer.

        `class_weights` (RTM REQ-5.20): this task's own inverse-frequency
        class weights, stored alongside its memory so a *later* task's
        `project_gradient()` call scores this memory with the weighting it
        was originally trained under, not whatever task is currently active.
        """
        self._add_task_examples(data, task_id, node_local_ids, build_temporal_neighbor_loader, class_weights)

    def add_task_examples_edges(
        self, data: Data, task_id, edge_idx: torch.Tensor, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        """RTM REQ-5.37: the edge-classification analogue of
        `add_task_examples` above -- same even-stride-across-the-task's-train-
        split sampling (no confidence filtering, matching GEM's own paper
        design) and the same fixed-per-task-budget shrinking (factored into
        `_add_task_examples` so the two modes can't silently drift apart on
        it), but over EDGE indices (transactions) via `build_temporal_link_
        loader` instead of node ids. The frozen subgraphs this produces
        already carry their own `edge_label_index`/`edge_label`, so
        `task_loss()` below needs no changes to replay them -- it already
        reads the target via `model.batch_targets(g)`, which branches on the
        model's own `cfg.task_type`, not on which admission method was used.
        """
        self._add_task_examples(data, task_id, edge_idx, build_temporal_link_loader, class_weights)

    def _add_task_examples(self, data, task_id, ids: torch.Tensor, loader_fn, class_weights) -> None:
        if task_id not in self._task_order:
            self._task_order.append(task_id)
        m = max(1, self.cfg.memory_budget // len(self._task_order))

        if ids.numel() > m:
            pick = torch.linspace(0, ids.numel() - 1, m).round().long().unique()
            ids = ids[pick]

        loader = loader_fn(data, ids, self.cfg.khop_num_neighbors, batch_size=1, shuffle=False)
        self.task_memories[task_id] = [g.cpu() for g in loader]
        if class_weights is not None:
            self.task_class_weights[task_id] = class_weights.cpu()

        for tid, subgraphs in self.task_memories.items():
            if tid != task_id and len(subgraphs) > m:
                keep = torch.linspace(0, len(subgraphs) - 1, m).round().long().unique().tolist()
                self.task_memories[tid] = [subgraphs[i] for i in keep]

    def task_loss(self, model: nn.Module, task_id, device) -> torch.Tensor:
        """Eq. 5: average cross-entropy over a stored task's frozen memory
        subgraphs, using the *whole* model (backbone + head) -- GEM's own
        formulation makes no backbone/head distinction, unlike this project's
        EWC (which deliberately scopes to the backbone only, REQ-2.1).
        """
        subgraphs = self.task_memories[task_id]
        class_weights = self.task_class_weights.get(task_id)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        total = torch.zeros((), device=device)
        for g in subgraphs:
            g = g.to(device)
            logits = model.forward(g)
            y = model.batch_targets(g)
            total = total + F.cross_entropy(logits, y, weight=class_weights)
        return total / len(subgraphs)

    def project_gradient(self, model: nn.Module, device) -> None:
        """Eq. 7-11: reads the gradient currently sitting in every optimized
        parameter's `.grad` (populated by the caller's own `total_loss.backward()`
        beforehand), computes one memory-loss gradient per completed task,
        projects, and overwrites `.grad` in place with the projected gradient
        -- the caller then just calls `optimizer.step()` as usual. A no-op
        (leaves `.grad` untouched) if no task has consolidated memory yet.
        """
        if not self.task_memories:
            return

        params = [p for p in model.parameters() if p.requires_grad]
        g = _flatten_grads(params)

        task_grads = []
        for task_id in self._task_order:
            model.zero_grad(set_to_none=True)
            self.task_loss(model, task_id, device).backward()
            task_grads.append(_flatten_grads(params))
        model.zero_grad(set_to_none=True)

        G = torch.stack(task_grads)
        g_proj = project_gradient_qp(g, G, self.cfg.qp_iters, self.cfg.qp_lr, self.cfg.qp_margin)
        _assign_grads(params, g_proj)

    def __len__(self) -> int:
        return sum(len(v) for v in self.task_memories.values())

    def state_dict(self) -> Dict[str, object]:
        return {
            "task_memories": {tid: subgraphs for tid, subgraphs in self.task_memories.items()},
            "task_class_weights": {tid: w for tid, w in self.task_class_weights.items()},
            "task_order": list(self._task_order),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.task_memories = dict(state["task_memories"])
        # .get(): checkpoints saved before RTM REQ-5.20 have no per-task class
        # weights at all -- task_loss() already tolerates a missing entry
        # (falls back to unweighted cross-entropy for that task's memory).
        self.task_class_weights = dict(state.get("task_class_weights", {}))
        self._task_order = list(state["task_order"])
