"""Topology-aware Weight Preserving (Liu, Yang & Wang, AAAI 2021,
"Overcoming Catastrophic Forgetting in Graph Neural Networks") -- RTM REQ-5.18.

Verified against the actual paper (not recollection) before implementing. Two
importance sub-modules, both from gradient *magnitude* (not squared, unlike
this project's EWC / Fisher):

  Minimized Loss Preserving (Eq. 3): I_k^loss = |d(task loss)/dW|, evaluated on
  task k's own training data right after finishing it.

  Topological Structure Preserving (Eq. 4-6): the model's post-softmax GATv2
  attention coefficients ARE this backbone's "e_ij" (the paper's own attention
  coefficient definition -- Eq. 2 -- already includes the softmax normalization,
  so no extra step is needed to match it). Build one scalar per batch -- the
  summed squared L2 norm of every node's attention-coefficient vector, which
  is algebraically just the sum of every coefficient squared, regardless of how
  they're partitioned per node -- and take I_k^ts = |d(that scalar)/dW|. The
  paper picks one arbitrary "middle layer" for this in their experiments (Sec.
  Topology-aware Weight Preserving, after Eq. 6); this project generalizes that
  to *every* stacked hop (see message_backbone.py::StackedSignedTemporalBackbone
  .attention_coefficients()), since with only `num_hops=2` layers there is no
  unambiguous middle one.

  Combined (Eq. 7): I_k = lambda_l * I_k^loss + lambda_t * I_k^ts.

Unlike this project's EWC (a single running/online Fisher estimate, Schwarz et
al. 2018), TWP's own Eq. 8 sums a *separate* penalty per completed task against
that task's own snapshot of optimal weights -- so this class stores one
(importance, anchor) pair per task, growing with the task stream, not one
running estimate. Eq. 9 (promoting minimized importance scores) adds a live,
differentiable L1 term computed from the *in-progress* task's own current
batch -- a genuine second-order (double-backward) computation, since I_{k+1}
is itself defined via a gradient and Eq. 9 asks the optimizer to shrink it.
"""
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from ..config import TWPConfig


class TWP:
    def __init__(self, backbone: nn.Module, cfg: TWPConfig):
        self.backbone = backbone
        self.cfg = cfg
        self._task_snapshots: List[Dict[str, Dict[str, torch.Tensor]]] = []
        self.task_count = 0

    def _backbone_params(self):
        return [(n, p) for n, p in self.backbone.named_parameters() if p.requires_grad]

    def compute_importance(
        self, nll_and_topo_losses: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Eq. 3 + Eq. 6 + Eq. 7. `nll_and_topo_losses` yields (task_loss,
        topo_loss) pairs from real forward passes on the task that just
        finished -- topo_loss is the sum of squared attention coefficients
        from that same forward pass (see `topology_loss()` below), capped at
        `cfg.importance_num_batches` pairs.
        """
        params = self._backbone_params()
        loss_importance = {n: torch.zeros_like(p) for n, p in params}
        topo_importance = {n: torch.zeros_like(p) for n, p in params}

        n_batches = 0
        for nll_loss, topo_loss in nll_and_topo_losses:
            self.backbone.zero_grad(set_to_none=True)
            nll_loss.backward(retain_graph=True)
            for n, p in params:
                if p.grad is not None:
                    loss_importance[n] += p.grad.detach().abs()

            # A batch whose sampled subgraph has no edges at all (an isolated
            # seed node with nothing in its k-hop window -- a real, occurring
            # case, not hypothetical) makes `topology_loss()` fall back to a
            # bare `torch.zeros(())` with no grad_fn (module docstring / Eq. 6):
            # there are no attention coefficients to differentiate. Treat that
            # batch's topology contribution as zero rather than crashing --
            # this is a genuine "no signal" case, not an error.
            if topo_loss.requires_grad:
                self.backbone.zero_grad(set_to_none=True)
                topo_loss.backward(retain_graph=False)
                for n, p in params:
                    if p.grad is not None:
                        topo_importance[n] += p.grad.detach().abs()

            n_batches += 1
            if n_batches >= self.cfg.importance_num_batches:
                break

        if n_batches > 0:
            for n in loss_importance:
                loss_importance[n] /= n_batches
                topo_importance[n] /= n_batches

        self.backbone.zero_grad(set_to_none=True)
        return {
            n: self.cfg.lambda_l * loss_importance[n] + self.cfg.lambda_t * topo_importance[n]
            for n, _ in params
        }

    def consolidate(self, nll_and_topo_losses: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Call once, right after finishing a task: stores this task's own
        (importance, anchor-weights) snapshot. Eq. 8 sums over every stored
        snapshot -- nothing here is merged into a running estimate.
        """
        importance = self.compute_importance(nll_and_topo_losses)
        anchor = {n: p.detach().clone() for n, p in self._backbone_params()}
        self._task_snapshots.append({"importance": importance, "anchor": anchor})
        self.task_count += 1
        self.backbone.zero_grad(set_to_none=True)

    def penalty(self) -> torch.Tensor:
        """Eq. 8: sum_{n=1}^{k} I_n (.) (W - W_n*)^2 over every completed task's
        own snapshot. lambda_l/lambda_t are already folded into each snapshot's
        `importance` (computed at that task's own end, Eq. 7), so no further
        global weight is applied here.
        """
        device = next(self.backbone.parameters()).device
        if not self._task_snapshots:
            return torch.zeros((), device=device)

        loss = torch.zeros((), device=device)
        for snap in self._task_snapshots:
            importance, anchor = snap["importance"], snap["anchor"]
            for n, p in self._backbone_params():
                if n in importance:
                    loss = loss + (importance[n] * (p - anchor[n]).pow(2)).sum()
        return loss

    def live_importance_l1(self, nll_loss: torch.Tensor, topo_loss: torch.Tensor) -> torch.Tensor:
        """Eq. 9: beta * ||I_{k+1}||_1, with I_{k+1} computed *live* from the
        in-progress task's own current batch (not a completed task's snapshot).
        Genuinely second-order: `nll_loss`/`topo_loss` come from the same
        forward pass as the caller's main task loss, and this method's own
        `torch.autograd.grad(..., create_graph=True)` calls make the resulting
        scalar differentiable, so the caller's later `total_loss.backward()`
        can propagate through *this* method's internal gradients too --
        exactly what "minimize the importance scores" requires, since the
        scores are themselves defined via a gradient.

        RTM REQ-5.22/5.23: `cfg.beta=0.0` is now the default -- diagnosed as
        actively harmful on severely class-imbalanced data (it minimizes
        gradient magnitude everywhere, which suppresses exactly the gradient
        spikes a rare-class detector needs). Short-circuits before the
        expensive double-backward `torch.autograd.grad(..., create_graph=True)`
        calls below when disabled, since the result would be exactly 0.0 either
        way -- no point paying for a term that's structurally zero.
        """
        if self.cfg.beta == 0.0:
            return torch.zeros((), device=nll_loss.device)

        params = [p for _, p in self._backbone_params()]
        if not params:
            return torch.zeros((), device=nll_loss.device)

        loss_grads = torch.autograd.grad(
            nll_loss, params, retain_graph=True, create_graph=True, allow_unused=True
        )
        # Same edgeless-batch case as `compute_importance()` above: a bare,
        # graph-disconnected `topo_loss` (no attention coefficients this batch)
        # can't be differentiated -- treat its contribution as all-None
        # (skipped below) rather than crashing.
        if topo_loss.requires_grad:
            topo_grads = torch.autograd.grad(
                topo_loss, params, retain_graph=True, create_graph=True, allow_unused=True
            )
        else:
            topo_grads = [None] * len(params)

        l1 = torch.zeros((), device=nll_loss.device)
        for lg, tg in zip(loss_grads, topo_grads):
            if lg is not None:
                l1 = l1 + self.cfg.lambda_l * lg.abs().sum()
            if tg is not None:
                l1 = l1 + self.cfg.lambda_t * tg.abs().sum()
        return self.cfg.beta * l1

    @staticmethod
    def topology_loss(attention_coefficients: List[torch.Tensor]) -> torch.Tensor:
        """Eq. 6's scalar: summed squared L2 norm of every node's attention-
        coefficient vector, generalized across every stacked hop (module
        docstring). Algebraically just the sum of every coefficient squared.
        """
        device = attention_coefficients[0].device if attention_coefficients else "cpu"
        total = torch.zeros((), device=device)
        for alpha in attention_coefficients:
            if alpha.numel() > 0:
                total = total + (alpha ** 2).sum()
        return total

    def state_dict(self) -> Dict[str, object]:
        return {
            "task_snapshots": [
                {
                    "importance": {n: t.detach().cpu() for n, t in snap["importance"].items()},
                    "anchor": {n: t.detach().cpu() for n, t in snap["anchor"].items()},
                }
                for snap in self._task_snapshots
            ],
            "task_count": self.task_count,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        device = next(self.backbone.parameters()).device
        self._task_snapshots = [
            {
                "importance": {n: t.to(device) for n, t in snap["importance"].items()},
                "anchor": {n: t.to(device) for n, t in snap["anchor"].items()},
            }
            for snap in state["task_snapshots"]
        ]
        self.task_count = state["task_count"]
