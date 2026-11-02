"""Averaged Gradient Episodic Memory (Chaudhry, Ranzato, Rohrbach & Elhoseiny,
ICLR 2019, "Efficient Lifelong Learning with A-GEM") -- RTM REQ-5.26.

Added specifically to fix vanilla GEM's real, accepted-at-the-time cost
(RTM REQ-5.18): one extra backward pass PER COMPLETED TASK, every training
step, because GEM keeps a separate memory per task and constrains the
gradient against every one of them individually (Eq. 6-11 in `gem.py`). That
cost is O(completed tasks) and was observed to genuinely bite late in
Elliptic's 49-task stream (task 44 took ~200s/epoch vs ~40-60s early on).

A-GEM's fix: ONE unified memory pool across every task (not one per task),
filled via reservoir sampling as the stream progresses, and ONE constraint
per step -- draw a single random reference batch from the whole pool, compute
ONE reference gradient, and project the current gradient against it with a
closed-form formula (no QP, no per-task loop). Cost becomes O(1) per step,
independent of how many tasks have completed.

Paired with the same `TWP` regularizer GEM uses (`continual_strategy ==
"agem_twp"`) -- this module only replaces GEM's replay/projection half, not
TWP's regularization half, so it can be compared head-to-head against
`gem_twp` on cost and performance under otherwise identical conditions.
"""
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from ..config import AGEMConfig
from ..data.temporal_loader import build_temporal_link_loader, build_temporal_neighbor_loader
from .gem import _assign_grads, _flatten_grads


class AGEMMemory:
    def __init__(self, cfg: AGEMConfig):
        self.cfg = cfg
        self.memory: List[Data] = []
        self.memory_weights: List[Optional[torch.Tensor]] = []
        self._n_seen = 0  # total examples ever offered, for reservoir sampling

    def add_task_examples(
        self, data: Data, task_id, node_local_ids: torch.Tensor, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        """Call once, at end of task: extracts each admitted node's frozen
        k-hop subgraph (same convention as GEM's own memory, REQ-5.11) and
        offers them one at a time to the single combined pool via classic
        reservoir sampling (Algorithm R) -- every example seen so far, from
        every task, has equal probability of being in the final pool,
        regardless of which task it came from or how many tasks have
        followed it. Unlike GEM's per-task shrinking slots, earlier tasks'
        examples are never deliberately evicted in favor of newer ones; they
        just compete on equal footing with everything else.

        `class_weights` stored alongside each individual admitted example
        (not per-task, since the pool itself is no longer task-partitioned)
        so the reference-gradient loss (`project_gradient`) scores each
        sampled example under the class balance it was actually captured
        under.
        """
        self._add_to_pool(build_temporal_neighbor_loader(data, node_local_ids, self.cfg.khop_num_neighbors, batch_size=1, shuffle=False), class_weights)

    def add_task_examples_edges(
        self, data: Data, task_id, edge_idx: torch.Tensor, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        """RTM REQ-5.37: the edge-classification analogue of
        `add_task_examples` above -- identical reservoir-sampling admission
        into the single unified pool, but over EDGE indices (transactions)
        via `build_temporal_link_loader` instead of node ids. `task_id` is
        accepted for call-site symmetry with the node-mode method and GEM's
        own signature, but (like `add_task_examples` above) is not itself
        stored -- the pool is not task-partitioned.
        """
        self._add_to_pool(build_temporal_link_loader(data, edge_idx, self.cfg.khop_num_neighbors, batch_size=1, shuffle=False), class_weights)

    def _add_to_pool(self, loader, class_weights: Optional[torch.Tensor]) -> None:
        w = class_weights.cpu() if class_weights is not None else None
        for g in loader:
            g = g.cpu()
            self._n_seen += 1
            if len(self.memory) < self.cfg.memory_budget:
                self.memory.append(g)
                self.memory_weights.append(w)
            else:
                j = random.randint(0, self._n_seen - 1)
                if j < self.cfg.memory_budget:
                    self.memory[j] = g
                    self.memory_weights[j] = w

    def project_gradient(self, model: nn.Module, device) -> None:
        """Eq. 2-3: reads the gradient currently sitting in every optimized
        parameter's `.grad` (populated by the caller's own `total_loss.backward()`
        beforehand), computes ONE reference gradient from a fresh random batch
        drawn from the whole pool, and -- only if the two gradients actually
        conflict (negative dot product) -- projects the current gradient onto
        the closest one that doesn't violate the reference:

            g~ = g - (g . g_ref / g_ref . g_ref) * g_ref   if g . g_ref < 0
            g~ = g                                          otherwise

        Closed-form, no QP -- this is the entire cost saving over GEM's own
        `project_gradient`. A no-op (leaves `.grad` untouched) if the pool is
        still empty.
        """
        if not self.memory:
            return

        params = [p for p in model.parameters() if p.requires_grad]
        g = _flatten_grads(params)

        k = min(self.cfg.sample_batch_size, len(self.memory))
        idx = random.sample(range(len(self.memory)), k)

        model.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for i in idx:
            subgraph = self.memory[i].to(device)
            w = self.memory_weights[i]
            w = w.to(device) if w is not None else None
            logits = model.forward(subgraph)
            y = model.batch_targets(subgraph)
            total = total + F.cross_entropy(logits, y, weight=w)
        (total / k).backward()
        g_ref = _flatten_grads(params)
        model.zero_grad(set_to_none=True)

        dot = torch.dot(g, g_ref)
        if dot < 0:
            g_proj = g - (dot / torch.dot(g_ref, g_ref).clamp(min=1e-12)) * g_ref
        else:
            g_proj = g
        _assign_grads(params, g_proj)

    def __len__(self) -> int:
        return len(self.memory)

    def state_dict(self) -> Dict[str, object]:
        return {
            "memory": list(self.memory),
            "memory_weights": list(self.memory_weights),
            "n_seen": self._n_seen,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.memory = list(state["memory"])
        self.memory_weights = list(state["memory_weights"])
        self._n_seen = state["n_seen"]
