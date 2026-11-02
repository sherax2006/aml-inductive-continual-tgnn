"""Elastic Weight Consolidation over the shared backbone only (CLAUDE.md line 20 /
RTM REQ-2.1, REQ-2.2, REQ-D4).

Deliberately scoped to `backbone.named_parameters()` and nothing else: the
dynamic per-task heads (REQ-2.5) must stay free to specialize, so only the
shared aggregator is regularized against drift. Uses the standard "online EWC"
formulation (Schwarz et al., 2018) -- a single running Fisher estimate updated
after every task with decay `gamma`, plus one running anchor-parameter snapshot
-- rather than keeping one Fisher matrix per task, since Elliptic alone has 49
temporal tasks and per-task storage would be wasteful. `gamma=1.0` (default)
reduces exactly to naive Fisher accumulation across tasks.

Fisher is the *empirical* Fisher (squared gradient of the true-label NLL),
which is the standard practical approximation used by nearly all EWC
implementations, not the exact Fisher (which would require sampling labels
from the model's own predictive distribution).
"""
from typing import Dict, Iterable

import torch
import torch.nn as nn

from ..config import EWCConfig


class EWC:
    def __init__(self, backbone: nn.Module, cfg: EWCConfig):
        self.backbone = backbone
        self.cfg = cfg
        self._fisher: Dict[str, torch.Tensor] = {}
        self._anchor_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0

    def compute_fisher(self, nll_losses: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """REQ-2.1: estimate the diagonal Fisher Information Matrix over backbone
        weights. `nll_losses` yields scalar cross-entropy losses from real
        forward passes (backbone + whatever head was active) on data from the
        task that just finished; capped at `cfg.fisher_num_batches` batches.
        """
        fisher = {
            n: torch.zeros_like(p) for n, p in self.backbone.named_parameters() if p.requires_grad
        }
        n_batches = 0
        for nll_loss in nll_losses:
            self.backbone.zero_grad(set_to_none=True)
            nll_loss.backward(retain_graph=False)
            for n, p in self.backbone.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1
            if n_batches >= self.cfg.fisher_num_batches:
                break

        if n_batches > 0:
            for n in fisher:
                fisher[n] /= n_batches
        return fisher

    def consolidate(self, nll_losses: Iterable[torch.Tensor]) -> None:
        """Call once, right after finishing a task's training: folds a fresh
        Fisher estimate into the running one and snapshots current backbone
        weights as the new penalty anchor for every subsequent task.
        """
        new_fisher = self.compute_fisher(nll_losses)
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            prev = self._fisher.get(n, torch.zeros_like(p))
            self._fisher[n] = self.cfg.gamma * prev + new_fisher.get(n, torch.zeros_like(p))
        self._anchor_params = {
            n: p.detach().clone() for n, p in self.backbone.named_parameters() if p.requires_grad
        }
        self.task_count += 1
        self.backbone.zero_grad(set_to_none=True)

    def penalty(self) -> torch.Tensor:
        """REQ-2.2: quadratic EWC penalty, zero before any task has consolidated."""
        device = next(self.backbone.parameters()).device
        if self.task_count == 0:
            return torch.zeros((), device=device)

        loss = torch.zeros((), device=device)
        for n, p in self.backbone.named_parameters():
            if n in self._fisher:
                loss = loss + (self._fisher[n] * (p - self._anchor_params[n]).pow(2)).sum()
        return self.cfg.lambda_ewc * loss

    def state_dict(self) -> Dict[str, object]:
        """RTM REQ-5.3/5.4: everything needed to resume EWC exactly -- the
        running Fisher, the anchor snapshot, and how many tasks have
        consolidated so far (`penalty()` short-circuits to zero at 0).
        """
        return {
            "fisher": {n: t.detach().cpu() for n, t in self._fisher.items()},
            "anchor_params": {n: t.detach().cpu() for n, t in self._anchor_params.items()},
            "task_count": self.task_count,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        device = next(self.backbone.parameters()).device
        self._fisher = {n: t.to(device) for n, t in state["fisher"].items()}
        self._anchor_params = {n: t.to(device) for n, t in state["anchor_params"].items()}
        self.task_count = state["task_count"]
