"""Continual Learning wrapper (CLAUDE.md Part 2 / RTM REQ-D1, REQ-D2).

Wires the pipeline:

    sampled subgraph -> StackedSignedTemporalBackbone (LSTM -> GAT, x`cfg.backbone.num_hops`)
    -> seed-node slice -> ClassificationHead -> logits

and owns the continual-learning cores as sub-objects rather than mixins:
  - `self.backbone` is the shared aggregator regularized by whichever strategy
    is active (`cfg.continual_strategy`).
  - `self.task_head` is one small MLP, shared across every task, always
    (RTM REQ-5.15: parameter isolation -- CLAUDE.md's original per-task
    `nn.ModuleDict` design, line 22 -- was removed at the user's explicit
    request; there is no more task-id-based head routing anywhere in this
    class, `forward()` takes only a batch).

RTM REQ-5.18/5.26: `cfg.continual_strategy` selects exactly one of three
continual-learning cores, the other two pairs stay `None`:
  - `"ewc_replay"` (default): `self.ewc` (Elastic Weight Consolidation,
    REQ-2.1/2.2) + `self.replay_buffer` (inductive episodic replay,
    REQ-2.3/2.4) -- this project's original strategy, unchanged.
  - `"gem_twp"`: `self.gem` (Gradient Episodic Memory, Lopez-Paz & Ranzato
    2017) + `self.twp` (Topology-aware Weight Preserving, Liu, Yang & Wang
    2021) -- added as a second, toggleable strategy specifically so the two
    can be compared head-to-head with this project's own Task Performance
    Matrix / paper-style evaluation harness, not as a replacement.
  - `"agem_twp"` (RTM REQ-5.26): `self.agem` (Averaged GEM, Chaudhry et al.
    2019) + the same `self.twp` -- added because vanilla GEM's own cost
    (one extra backward pass per completed task, every step) was observed
    to genuinely bite late in a long task stream; A-GEM's single unified
    memory + closed-form projection is O(1) per step instead. A third,
    coexisting strategy, not a replacement for `"gem_twp"`.
  - `"mas_twp"` (RTM REQ-5.27): `self.mas` (Memory Aware Synapses, Aljundi
    et al. 2018) + the same `self.twp` -- MAS takes over GEM/A-GEM's
    structural slot but via a SECOND additive regularization penalty
    (stacked with TWP's own in `task_loss()`), not a memory/replay
    mechanism -- no buffer, no gradient projection anywhere in this
    strategy. A fourth, coexisting strategy.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from ..config import ExperimentConfig
from ..continual.agem import AGEMMemory
from ..continual.ewc import EWC
from ..continual.gem import GEMMemory
from ..continual.mas import MAS
from ..continual.replay_buffer import GraphReplayBuffer
from ..continual.twp import TWP
from ..data.temporal_loader import seed_batch_targets
from .classification_head import ClassificationHead, EdgeClassificationHead
from .message_backbone import StackedSignedTemporalBackbone

import torch.nn as nn


class ContinualTGNN(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = StackedSignedTemporalBackbone(cfg.backbone)
        # RTM REQ-5.37: exactly one of the two heads below is actually used,
        # selected by `cfg.task_type` -- "node" (Elliptic) or "edge" (IBM AML,
        # whose native label lives on the transaction, not the account; see
        # data/ibm_aml_dataset.py's module docstring). Unlike the ablation
        # study's backbone switches (RTM REQ-5.32), which always construct
        # both variants so one checkpoint format covers every ablation arm,
        # `task_type` is fixed per dataset for the life of a run -- a node-
        # classification checkpoint is never loaded into an edge-classification
        # model or vice versa, so there is no cross-mode state_dict compatibility
        # to preserve, and constructing only the head actually used keeps a
        # model's own parameter count honest for whichever task it runs.
        if cfg.task_type == "edge":
            self.task_head = None
            self.edge_head = EdgeClassificationHead(cfg.backbone.node_out_dim, cfg.task_head)
        elif cfg.task_type == "node":
            self.task_head = ClassificationHead(cfg.backbone.node_out_dim, cfg.task_head)
            self.edge_head = None
        else:
            raise ValueError(f"Unknown task_type {cfg.task_type!r}; expected 'node' or 'edge'.")

        if cfg.continual_strategy == "ewc_replay":
            self.ewc = EWC(self.backbone, cfg.ewc)
            self.replay_buffer = GraphReplayBuffer(cfg.replay)
            self.gem = None
            self.agem = None
            self.mas = None
            self.twp = None
        elif cfg.continual_strategy == "gem_twp":
            self.ewc = None
            self.replay_buffer = None
            self.gem = GEMMemory(cfg.gem)
            self.agem = None
            self.mas = None
            self.twp = TWP(self.backbone, cfg.twp)
        elif cfg.continual_strategy == "agem_twp":
            self.ewc = None
            self.replay_buffer = None
            self.gem = None
            self.agem = AGEMMemory(cfg.agem)
            self.mas = None
            self.twp = TWP(self.backbone, cfg.twp)
        elif cfg.continual_strategy == "mas_twp":
            self.ewc = None
            self.replay_buffer = None
            self.gem = None
            self.agem = None
            self.mas = MAS(self.backbone, cfg.mas)
            self.twp = TWP(self.backbone, cfg.twp)
        else:
            raise ValueError(
                f"Unknown continual_strategy {cfg.continual_strategy!r}; expected "
                "'ewc_replay', 'gem_twp', 'agem_twp', or 'mas_twp'."
            )

    def embed_seed_nodes(self, batch: Data) -> Tensor:
        """Runs the backbone over the sampled subgraph and slices down to the
        seed nodes this batch was built to classify (REQ-D2's "subgraph -> LSTM
        -> GAT" half of the pipeline). Node-classification mode only -- edge
        mode's `forward()` below reads two endpoint embeddings directly rather
        than a single seed-node slice, so it does not go through this method.
        """
        z = self.backbone(batch.x, batch.edge_index, batch.edge_attr, batch.edge_sign, batch.edge_time)
        seed_idx = seed_batch_targets(batch)
        return z[seed_idx]

    def forward(self, batch: Data) -> Tensor:
        """Full pipeline: subgraph -> backbone -> classification head -> logits.

        RTM REQ-5.37: branches on `cfg.task_type`. Node mode (Elliptic)
        unchanged from before this row -- one seed-node embedding per shared
        `ClassificationHead`. Edge mode (IBM AML) instead reads BOTH endpoint
        embeddings for each target edge (`batch.edge_label_index`, supplied by
        `LinkNeighborLoader` -- see `data/temporal_loader.py::build_temporal_
        link_loader`) through `EdgeClassificationHead`; the backbone call
        itself is identical in both modes, since it always just aggregates
        over whatever subgraph it's handed.
        """
        if self.cfg.task_type == "edge":
            z = self.backbone(batch.x, batch.edge_index, batch.edge_attr, batch.edge_sign, batch.edge_time)
            z_src = z[batch.edge_label_index[0]]
            z_dst = z[batch.edge_label_index[1]]
            return self.edge_head(z_src, z_dst)
        z_seed = self.embed_seed_nodes(batch)
        return self.task_head(z_seed)

    def batch_targets(self, batch: Data) -> Tensor:
        """RTM REQ-5.37: the ground-truth label for whatever `forward(batch)`
        just classified -- `batch.edge_label` (edge mode) or `batch.y` sliced
        to this batch's seed nodes (node mode, unchanged).
        """
        if self.cfg.task_type == "edge":
            return batch.edge_label
        return batch.y[: batch.batch_size]

    def task_loss(
        self, batch: Data, include_penalty: bool = True, class_weights: Optional[Tensor] = None
    ) -> "tuple[Tensor, Tensor]":
        """Cross-entropy on the current task's mini-batch, optionally plus the
        active strategy's regularization penalty: the running EWC penalty
        (REQ-2.2) under `"ewc_replay"`, or TWP's Eq. 8 (summed per-completed-task
        importance penalty) + Eq. 9 (live L1 on this batch's own importance
        estimate) under `"gem_twp"`/`"agem_twp"`/`"mas_twp"` -- all three share
        the identical TWP regularization core; `"gem_twp"`/`"agem_twp"` differ
        only in which replay/projection mechanism the training loop pairs it
        with outside this method, while `"mas_twp"` additionally stacks MAS's
        own penalty directly into this same loss (RTM REQ-5.27 -- MAS has no
        post-backward projection step at all, unlike GEM/A-GEM). Returns
        (total_loss, logits).

        `class_weights` (RTM REQ-5.20): optional `[num_classes]` inverse-
        frequency weights (`data/split.py::compute_class_weights`), recomputed
        by the caller fresh every task from that task's own train split.
        Without this, plain cross-entropy on Elliptic's/IBM AML's heavily
        imbalanced illicit class has a stable trivial optimum -- confirmed on
        real data, see RTM.md REQ-5.20.
        """
        logits = self.forward(batch)
        y = self.batch_targets(batch)
        nll = F.cross_entropy(logits, y, weight=class_weights)
        loss = nll
        if include_penalty:
            if self.cfg.continual_strategy == "ewc_replay":
                loss = loss + self.ewc.penalty()
            else:  # gem_twp, agem_twp, or mas_twp -- identical TWP penalty in all three
                topo_loss = TWP.topology_loss(self.backbone.attention_coefficients())
                loss = loss + self.twp.penalty() + self.twp.live_importance_l1(nll, topo_loss)
                if self.cfg.continual_strategy == "mas_twp":
                    # RTM REQ-5.27: MAS's own penalty stacks on top of TWP's --
                    # this is the strategy's *entire* mechanism, there is no
                    # replay buffer or gradient projection anywhere for it.
                    loss = loss + self.mas.penalty()
        return loss, logits

    def replay_loss(self, class_weights: Optional[Tensor] = None) -> Tensor:
        """REQ-2.4: fresh k-hop subgraphs around buffered historical nodes,
        summed into one scalar the training loop adds to the current step's
        loss before backward(). Only meaningful under `"ewc_replay"` -- the
        `"gem_twp"`/`"agem_twp"` strategies have no replay-loss term; they
        constrain the *gradient direction* instead (`GEMMemory.project_gradient`
        or `AGEMMemory.project_gradient`), applied by the training loop after
        backward(), not folded into the loss. `"mas_twp"` has no replay-loss
        term either, and no gradient projection -- its entire mechanism is
        `task_loss()`'s own stacked MAS+TWP penalty.

        `class_weights` (RTM REQ-5.20): the *current* task's own weights,
        applied uniformly here even though a replayed node originally came
        from a different (past) task -- a deliberate simplification over
        tracking each buffered node's originating task's own class balance.
        """
        device = next(self.parameters()).device
        batches = self.replay_buffer.sample_replay_batch()
        if not batches:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)
        for b in batches:
            b = b.to(device)  # freshly sampled by NeighborLoader on CPU regardless of model device
            logits = self.forward(b)
            y = self.batch_targets(b)
            total = total + F.cross_entropy(logits, y, weight=class_weights)
        return total / len(batches)

    @torch.no_grad()
    def confident_illicit_nodes(self, batch: Data, illicit_class: int = 1):
        """Helper for REQ-2.3 admission: returns (local_node_ids, confidences)
        for seed nodes the current model predicts as `illicit_class` with
        probability >= replay.confidence_threshold, for the caller to hand to
        `replay_buffer.add_candidates` after a task finishes training. Node-
        classification mode only -- edge mode uses `confident_illicit_edges`
        below, since there is no single "seed node" to report a confidence for.
        """
        logits = self.forward(batch)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        mask = (pred == illicit_class) & (conf >= self.cfg.replay.confidence_threshold)
        seed_idx = seed_batch_targets(batch)
        local_ids = seed_idx[mask]
        return local_ids, conf[mask]

    @torch.no_grad()
    def confident_illicit_edges(self, batch: Data, illicit_class: int = 1):
        """RTM REQ-5.37: the edge-classification analogue of
        `confident_illicit_nodes` above -- returns (local_batch_positions,
        confidences) for target edges (from a `build_temporal_link_loader`
        batch) the current model predicts as `illicit_class` with probability
        >= `replay.confidence_threshold`. `local_batch_positions` indexes into
        THIS batch's own `edge_label_index`/`edge_label`/`input_id` columns --
        the caller recovers the GLOBAL (into the task's own edge_index) edge
        index via `batch.input_id[local_batch_positions]`, exactly mirroring
        how node mode's caller recovers a global node id via `batch.n_id[
        local_ids]` (verified directly against `LinkNeighborLoader`'s actual
        `input_id` semantics before relying on it, not assumed from the docs).
        """
        logits = self.forward(batch)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        mask = (pred == illicit_class) & (conf >= self.cfg.replay.confidence_threshold)
        local_positions = torch.arange(logits.size(0), device=logits.device)[mask]
        return local_positions, conf[mask]
