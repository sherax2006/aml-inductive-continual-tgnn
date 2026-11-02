"""Single shared classification head (RTM REQ-5.15).

CLAUDE.md's original spec (line 22) called for parameter isolation -- a
dynamically-provisioned, isolated MLP head per temporal task. That mechanism
was removed from this project's continual-learning strategy at the user's
explicit request on 2026-08-21 (see RTM.md and ARCHITECTURE.md for the full
history: it started as a `task_il`/`domain_il` mode switch, REQ-5.9, and was
then removed outright rather than kept as a configurable option).

What's here now is just a small MLP applied identically to every task's
seed-node embeddings, always -- no per-task state, no task ID needed to
select anything. The two remaining continual-learning defenses (EWC
regularization on the shared backbone, and episodic replay) are unaffected;
this head was never protected by them (it didn't need to be, being isolated),
and now that it's shared, it's exposed to the same drift risk EWC already
guards the backbone against -- EWC is not extended to cover it, since the
user's request was to remove isolation, not to add a new regularizer in its
place.
"""
import torch
import torch.nn as nn

from ..config import TaskHeadConfig


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, cfg: TaskHeadConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EdgeClassificationHead(nn.Module):
    """RTM REQ-5.37: classifies an EDGE (transaction) rather than a node
    (account) -- IBM AML's native task (see `data/ibm_aml_dataset.py`'s module
    docstring for why this differs from Elliptic, which stays node
    classification since its nodes already *are* the labeled transactions).

    Takes the backbone's two endpoint node embeddings for a target edge and
    concatenates them (source, then target) before the same small MLP shape
    `ClassificationHead` uses -- the backbone itself is completely unchanged;
    only this readout differs. Concatenation (not sum/mean) deliberately
    preserves which endpoint sent and which received, since "who paid whom"
    is exactly the asymmetry the backbone's own signed, bidirectional message
    passing (REQ-5.17) already tracks -- collapsing it here would throw that
    signal away right before the final decision.
    """

    def __init__(self, in_dim: int, cfg: TaskHeadConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_src, z_dst], dim=-1))
