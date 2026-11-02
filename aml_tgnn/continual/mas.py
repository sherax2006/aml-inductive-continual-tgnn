"""Memory Aware Synapses (Aljundi, Babiloni, Elhoseiny, Rohrbach & Tuytelaars,
ECCV 2018) -- RTM REQ-5.27.

An alternative regularization core, added specifically to compare against
GEM/A-GEM's replay/projection approach under otherwise-identical conditions.
Per an explicit user design choice (asked via AskUserQuestion): MAS takes
over GEM/A-GEM's *structural slot* in the strategy (`continual_strategy ==
"mas_twp"`, paired with the same `TWP` regularizer `gem_twp`/`agem_twp`
already use), but does so via a SECOND additive regularization penalty
stacked with TWP's own -- not a memory/replay mechanism. There is no buffer,
no memory pool, and no gradient projection anywhere in this strategy; unlike
`gem_twp`/`agem_twp`, `_training_step` for `mas_twp` is just
`loss.backward()` -- MAS's penalty is already folded into `task_loss()`.

Mirrors this project's own `EWC` almost exactly -- same online running-
estimate-with-decay accumulation (`gamma`), same backbone-only scoping
(leaving the shared classification head free to specialize, matching EWC's
own established convention) -- differing only in what "importance" measures:

    Omega_ij = (1/N) * sum_k | d(||F(x_k;theta)||^2_2) / d(theta_ij) |

the gradient MAGNITUDE (not squared, unlike EWC's empirical Fisher) of the
model's own output function's squared L2 norm -- entirely label-free by
construction (the paper's own headline property: importance can be
accumulated from unlabeled data too). This implementation still draws
`output_norms` from the same task/batch stream `EWC`/`TWP` consolidate from,
for cadence consistency with the rest of this project -- MAS's own
unlabeled-data selling point isn't separately exploited here.

`F(x)` is taken as the shared classification head's own logits (not just the
backbone's raw embedding) -- the paper's own definition uses "the learned
function", typically the actual output layer for a classifier -- while the
resulting penalty itself is still scoped to `backbone.named_parameters()`
only (gradients still flow back through the head via the chain rule; only
which parameters get penalized is restricted), matching EWC's own choice to
leave the task head free to specialize.

Same per-batch approximation EWC already makes: each batch's own
`||logits||^2` (meaned over that batch's seed nodes, not summed, so
`sample_batch_size` doesn't rescale the resulting gradient magnitudes) is
treated as one importance sample, rather than computing a true per-example
gradient (which would need one backward pass per example).
"""
from typing import Dict, Iterable

import torch
import torch.nn as nn

from ..config import MASConfig


class MAS:
    def __init__(self, backbone: nn.Module, cfg: MASConfig):
        self.backbone = backbone
        self.cfg = cfg
        self._importance: Dict[str, torch.Tensor] = {}
        self._anchor_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0

    def compute_importance(self, output_norms: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """`output_norms` yields scalar ||F(x)||^2_2 values (meaned over a
        batch's seed nodes) from real forward passes on data from the task
        that just finished; capped at `cfg.importance_num_batches` batches.
        Entirely label-free -- these are squared-output-norm scalars, never
        losses, so no `y`/class weights are involved anywhere in this method.
        """
        importance = {
            n: torch.zeros_like(p) for n, p in self.backbone.named_parameters() if p.requires_grad
        }
        n_batches = 0
        for output_norm in output_norms:
            self.backbone.zero_grad(set_to_none=True)
            output_norm.backward(retain_graph=False)
            for n, p in self.backbone.named_parameters():
                if p.requires_grad and p.grad is not None:
                    importance[n] += p.grad.detach().abs()
            n_batches += 1
            if n_batches >= self.cfg.importance_num_batches:
                break

        if n_batches > 0:
            for n in importance:
                importance[n] /= n_batches
        return importance

    def consolidate(self, output_norms: Iterable[torch.Tensor]) -> None:
        """Call once, right after finishing a task's training: folds a fresh
        importance estimate into the running one (same online decay-and-
        accumulate style as this project's own `EWC.consolidate`) and
        snapshots current backbone weights as the new penalty anchor for
        every subsequent task.
        """
        new_importance = self.compute_importance(output_norms)
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            prev = self._importance.get(n, torch.zeros_like(p))
            self._importance[n] = self.cfg.gamma * prev + new_importance.get(n, torch.zeros_like(p))
        self._anchor_params = {
            n: p.detach().clone() for n, p in self.backbone.named_parameters() if p.requires_grad
        }
        self.task_count += 1
        self.backbone.zero_grad(set_to_none=True)

    def penalty(self) -> torch.Tensor:
        """Quadratic MAS penalty, zero before any task has consolidated --
        identical form to `EWC.penalty()`, just reading MAS's own importance
        and anchor snapshots instead of EWC's Fisher/anchor pair."""
        device = next(self.backbone.parameters()).device
        if self.task_count == 0:
            return torch.zeros((), device=device)

        loss = torch.zeros((), device=device)
        for n, p in self.backbone.named_parameters():
            if n in self._importance:
                loss = loss + (self._importance[n] * (p - self._anchor_params[n]).pow(2)).sum()
        return self.cfg.lambda_mas * loss

    def state_dict(self) -> Dict[str, object]:
        return {
            "importance": {n: t.detach().cpu() for n, t in self._importance.items()},
            "anchor_params": {n: t.detach().cpu() for n, t in self._anchor_params.items()},
            "task_count": self.task_count,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        device = next(self.backbone.parameters()).device
        self._importance = {n: t.to(device) for n, t in state["importance"].items()}
        self._anchor_params = {n: t.to(device) for n, t in state["anchor_params"].items()}
        self.task_count = state["task_count"]
