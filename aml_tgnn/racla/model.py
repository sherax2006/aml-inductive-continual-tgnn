"""RACLA's AML Expert Layer -- Mixture-of-Experts over GAT-based experts
(RTM REQ-5.25), reimplemented from Zhang, Liu, Lan, Chen, Chen & Zhang,
"RACLA: Role-aware continual learning for robust AML detection", Expert
Systems With Applications 299 (2026) 130136, Sec. 4.2 (Eq. 1-9). See
BasePaper1_RACLA.md for the exact source formulas this maps onto.

This is a genuinely separate model from `ContinualTGNN` -- RACLA's MoE-of-
GAT-experts architecture is a different backbone entirely, not a different
continual-learning strategy layered on this project's own signed-temporal
backbone. It's built specifically as a controlled comparison baseline: same
`NeighborLoader`-based inductive sampling, same dataset adapters, same
checkpoint/logging/evaluation infrastructure as the rest of this project, so
a run of this model produces results directly comparable (same protocol) to
`ContinualTGNN`'s, not just to RACLA's own reported numbers.

Reproducibility note (documented once here, applies throughout `racla/`):
several implementation details are not stated in the paper's main text (no
public code was found, see RTM.md REQ-5.25). Every such gap is filled with a
reasonable, explicitly documented choice rather than left ambiguous -- this
makes the reimplementation faithful to RACLA's *described* method, not a
guarantee of reproducing their *exact* reported numbers.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from ..config import ExperimentConfig
from ..data.temporal_loader import seed_batch_targets


class NoisyTopKGating(nn.Module):
    """Eq. 4, 5, 6: Top-K noisy gating.

    `Q(h_i) = h_i . W_clean + StandardNormal() . Softplus(h_i . W_noise)`
    (Eq. 6) -- implemented as two `nn.Linear` layers (bias-free, matching the
    paper's pure matrix-product notation) rather than literally `W_clean in
    R^{F x m}` / `W_noise in R^{m x F}` as printed: those two shapes are not
    both dimensionally consistent with `h_i . W` for a row vector `h_i` (one
    of them is very likely a transpose typo in the paper itself, the same
    category of minor notational looseness flagged elsewhere in
    BasePaper1_RACLA.md) -- both matrices are implemented here as `[F, m]`,
    the only shape that makes Eq. 6 consistent.

    Noise is injected only in `self.training` mode: the paper motivates it as
    encouraging exploration/diversity during learning, and eval-time
    determinism is needed for reproducible reported metrics -- this project's
    own choice, not stated either way in the paper.
    """

    def __init__(self, in_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.w_clean = nn.Linear(in_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        clean = self.w_clean(h)
        if self.training:
            noise_std = F.softplus(self.w_noise(h))
            q = clean + torch.randn_like(clean) * noise_std
        else:
            q = clean

        # Eq. 5 (KeepTopK): mask every logit outside the top-k to -inf.
        top_vals, top_idx = q.topk(self.top_k, dim=-1)
        masked = torch.full_like(q, float("-inf"))
        masked.scatter_(-1, top_idx, top_vals)

        # Eq. 4: softmax over the masked logits -- zero weight on non-top-k experts.
        return F.softmax(masked, dim=-1)  # [N, num_experts], rows sum to 1


class AMLExpertLayer(nn.Module):
    """Eq. 3: h_i' = sigma( sum_o g_i(h_i)_o . E_o(h_j, e_ij, W) ).

    Each expert `E_o` is a `torch_geometric.nn.GATConv` -- RACLA's own related
    work cites the *original* GAT (Velickovic et al. 2018) as the expert
    backbone, and PyG's `GATConv` (not `GATv2Conv`, which this project's own
    `InductiveSignedTemporalBackbone` uses instead, RTM REQ-5.16) is exactly
    that formulation. `edge_dim` incorporates edge features directly into
    the attention computation, matching "each expert GAT incorporates both
    node and edge information." All experts consume the same input
    features/edges but hold independent learned parameters (the paper's own
    "every expert shares the same input features but leverages distinct
    learned parameters").

    `sigma` (the paper's own notation, Eq. 3) is implemented as ReLU -- a
    standard, reasonable choice; the paper does not specify which
    nonlinearity.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, num_experts: int, top_k: int, heads: int = 1):
        super().__init__()
        self.gating = NoisyTopKGating(in_dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            GATConv(in_dim, out_dim, heads=heads, concat=False, edge_dim=edge_dim, add_self_loops=True)
            for _ in range(num_experts)
        )
        # RTM REQ-5.25: last forward pass's gate weights g_i(h_i), stashed here
        # (not a buffer/parameter) for the load-balancing loss (Eq. 7/8) and
        # routing-based replay selection (Eq. 10-12, which needs g_i as a
        # per-node embedding) to read without recomputation.
        self.last_gates: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        gates = self.gating(x)  # [N, m]
        self.last_gates = gates
        expert_outs = torch.stack(
            [expert(x, edge_index, edge_attr=edge_attr) for expert in self.experts], dim=1
        )  # [N, m, out_dim]
        combined = (gates.unsqueeze(-1) * expert_outs).sum(dim=1)  # [N, out_dim]
        return F.relu(combined)


class RACLAModel(nn.Module):
    """Full RACLA classifier: `cfg.backbone.num_hops` stacked `AMLExpertLayer`s
    (matching this project's own `StackedSignedTemporalBackbone`'s hop depth,
    for a fair, identical-receptive-field comparison -- the paper's own Fig. 1
    draws a single expert-layer block per timestamp, with no stated depth) +
    a linear classifier head on the final layer's seed-node output.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.racla
        dims: List[int] = [cfg.backbone.node_in_dim] + [r.expert_hidden_dim] * cfg.backbone.num_hops
        self.layers = nn.ModuleList(
            AMLExpertLayer(dims[i], dims[i + 1], cfg.backbone.edge_in_dim, r.num_experts, r.top_k, r.expert_heads)
            for i in range(cfg.backbone.num_hops)
        )
        self.classifier = nn.Linear(r.expert_hidden_dim, cfg.task_head.num_classes)

    def embed_all_nodes(self, batch: Data) -> torch.Tensor:
        x = batch.x
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr)
        return x

    def embed_seed_nodes(self, batch: Data) -> torch.Tensor:
        z = self.embed_all_nodes(batch)
        return z[seed_batch_targets(batch)]

    def forward(self, batch: Data, return_embeddings: bool = False):
        """Full pipeline: subgraph -> stacked AML expert layers -> seed-node
        slice -> classifier -> logits. Default signature (`return_embeddings
        =False`) matches `ContinualTGNN.forward()` exactly, so `utils/metrics
        .py::run_eval` and `task_matrix.py` work against this model unchanged.

        `return_embeddings=True` additionally returns the *all-nodes*
        embedding tensor (pre-classifier) from this same forward pass, so the
        edge predictor's loss (Eq. 13/14, `replay.py::EdgePredictor
        .edge_loss`) can be computed from it directly -- calling
        `embed_all_nodes()` a second time would not just waste compute, it
        would draw a *different* Top-K gating noise sample (Eq. 6 injects
        noise only in training mode), making the logits and embeddings
        inconsistent with each other.
        """
        z_all = self.embed_all_nodes(batch)
        logits = self.classifier(z_all[seed_batch_targets(batch)])
        if return_embeddings:
            return logits, z_all
        return logits

    def batch_targets(self, batch: Data) -> torch.Tensor:
        """RTM REQ-5.37: RACLA is always node classification (it has no IBM
        AML edge-classification mode), so this always reads `batch.y` --
        added only so `utils/metrics.py::run_eval` (now shared with
        `ContinualTGNN`'s node/edge dispatch) stays genuinely model-agnostic,
        matching this class's own `forward()` docstring's claim above.
        """
        return batch.y[: batch.batch_size]

    def routing_vectors(self, batch: Data) -> torch.Tensor:
        """g_i for this batch's seed nodes, from the *last* expert layer --
        used by `RACLAReplayBuffer` (Eq. 10-12) as the per-node embedding for
        coverage-based diverse buffer selection. Requires a forward pass to
        have just populated `self.layers[-1].last_gates`.
        """
        gates = self.layers[-1].last_gates
        if gates is None:
            raise RuntimeError("routing_vectors() called before any forward() pass")
        return gates[seed_batch_targets(batch)]

    def load_balance_loss(self) -> torch.Tensor:
        """Eq. 7 + 8 + 9 combined, summed over every stacked layer's most
        recent forward pass: `Importance(o) = mean_i g_{i,o}` over this
        batch's nodes (Eq. 7), `L_balance = w_importance * CV(Importance)^2`
        (Eq. 8) where CV is computed over the length-m Importance *vector*
        (the paper's own `L_balance(o)` notation implies one value per
        expert, but a coefficient of variation needs a distribution to take
        dispersion over -- the sensible reading, matching the paper's own
        prose "operating on the vector of expert importances", is one scalar
        CV over all m experts' importance values, not one per expert; this is
        the same category of minor notational looseness already flagged for
        Eq. 6). `w_importance` (Eq. 8) and `lambda` (Eq. 9) are combined into
        one `cfg.racla.load_balance_lambda` (see RACLAConfig's own docstring).
        """
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        eps = 1e-8
        for layer in self.layers:
            if layer.last_gates is None:
                continue
            importance = layer.last_gates.mean(dim=0)  # [m], Eq. 7
            cv = importance.std(unbiased=False) / (importance.mean() + eps)
            total = total + cv.pow(2)
        return self.cfg.racla.load_balance_lambda * total
