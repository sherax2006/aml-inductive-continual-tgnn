"""Routing-based replay selection (RRS) + structure learning for the buffer
(RTM REQ-5.25), from BasePaper1_RACLA.md Sec. 4.3-4.4 (Eq. 10-14).

Frozen-at-admission design, matching this project's own established replay
buffer convention (RTM REQ-5.11 -- `GraphReplayBuffer`/`GEMMemory` both
already freeze a node's k-hop subgraph at the moment it's selected, not
resampled later): each buffer node's ego-subgraph is extracted once, at
selection time, via the same `build_temporal_neighbor_loader` primitive.
This is a consistent reuse of an already-justified design decision, not a
new one made for RACLA specifically.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from ..config import RACLAConfig
from ..data.temporal_loader import build_temporal_neighbor_loader


def _greedy_max_coverage(coverage: torch.Tensor, ell: int) -> List[int]:
    """Greedy approximation to Eq. 12's `argmax |Cover({w_b1,...,w_bl})|` --
    the paper doesn't specify a solving algorithm; max-coverage is NP-hard in
    general, and greedy (repeatedly pick the candidate adding the most
    not-yet-covered nodes) is the standard, well-known (1 - 1/e)-approximation
    for it, so it's used here as the defensible default choice.

    `coverage`: `[n, n]` bool, `coverage[i, j]` True iff node j is in C(v_i)
    (Eq. 10, so a node always covers itself: dist(g_i,g_i)=0, sim(g_i,g_i)=1).
    Always returns exactly `min(ell, n)` node indices -- once every node is
    already covered, remaining picks just fill out the requested buffer quota
    (e.g. the paper's own "1% of nodes" budget) rather than stopping early and
    under-filling it.
    """
    n = coverage.size(0)
    covered = torch.zeros(n, dtype=torch.bool, device=coverage.device)
    selected_mask = torch.zeros(n, dtype=torch.bool, device=coverage.device)
    selected: List[int] = []
    for _ in range(min(ell, n)):
        marginal = (coverage & ~covered.unsqueeze(0)).sum(dim=1).float()
        marginal[selected_mask] = -1.0
        best = int(marginal.argmax())
        selected.append(best)
        selected_mask[best] = True
        covered |= coverage[best]
    return selected


def select_buffer_node_positions(routing_vectors: torch.Tensor, cfg: RACLAConfig, ell: int) -> List[int]:
    """Eq. 10-12: which of `routing_vectors`' rows (one per candidate node,
    already restricted by the caller to a task's labeled/train nodes) to
    admit into the buffer. Returns positions into `routing_vectors`, not
    global node ids -- the caller maps back to real node ids.
    """
    dists = torch.cdist(routing_vectors, routing_vectors)          # Eq. 10's dist(g_i, g_j)
    d_bar = dists.mean()                                            # Eq. 11's D-bar
    beta = cfg.replay_coverage_r * d_bar                            # Eq. 11's beta = r * D-bar

    normed = F.normalize(routing_vectors, dim=-1, eps=1e-8)
    sims = normed @ normed.T                                        # Eq. 10's sim(g_i, g_j), cosine

    coverage = (dists < beta) & (sims > cfg.replay_similarity_threshold)  # Eq. 10's C(v_i)
    return _greedy_max_coverage(coverage, ell)


class EdgePredictor(nn.Module):
    """Eq. 14: S_ij = Sigmoid(MLP([h_i || h_j]))."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))

    def edge_loss(self, h: torch.Tensor, edge_index: torch.Tensor, neg_ratio: float) -> torch.Tensor:
        """Eq. 13: binary cross-entropy over positive (real) edges in this
        subgraph plus an equal-ish number (`neg_ratio`) of sampled negative
        (non-existing) node pairs -- D_t^edge, reconstructed fresh from
        whatever subgraph `h`/`edge_index` describe. `h` is the model's own
        node embeddings (post expert-layer, pre-classifier) for this
        subgraph -- the paper's own "typically derived by concatenating the
        feature representations of node i and j (e.g., h_i and h_j)".
        """
        n = h.size(0)
        pos_i, pos_j = edge_index[0], edge_index[1]
        if pos_i.numel() == 0:
            return torch.zeros((), device=h.device)

        num_neg = max(1, int(pos_i.numel() * neg_ratio))
        neg_i = torch.randint(0, n, (num_neg,), device=h.device)
        neg_j = torch.randint(0, n, (num_neg,), device=h.device)

        pos_scores = self.forward(h[pos_i], h[pos_j])
        neg_scores = self.forward(h[neg_i], h[neg_j])
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        return F.binary_cross_entropy(scores, labels)


class RACLAReplayBuffer:
    def __init__(self, cfg: RACLAConfig):
        self.cfg = cfg
        self.entries: List[Data] = []  # frozen k-hop ego-subgraphs, one per admitted buffer node

    def add_task_buffer(
        self, data: Data, task_node_ids: torch.Tensor, routing_vectors: torch.Tensor, khop_num_neighbors: List[int]
    ) -> int:
        """Call once, at end of task: selects this task's buffer quota (Eq.
        10-12, `cfg.replay_buffer_ratio` of this task's labeled nodes) and
        freezes each admitted node's k-hop subgraph. Appended to the
        buffer's running entries -- unlike this project's own GEM (RTM
        REQ-5.18, one fixed-and-shrinking budget slot per task), RACLA's
        buffer just accumulates every task's own selection, matching the
        paper's description of a persistent replay buffer rather than a
        per-task memory partition (the paper gives no shrink-over-time rule).
        Returns the number of nodes admitted.
        """
        ell = max(1, round(self.cfg.replay_buffer_ratio * task_node_ids.numel()))
        positions = select_buffer_node_positions(routing_vectors, self.cfg, ell)
        admitted_ids = task_node_ids[positions]

        loader = build_temporal_neighbor_loader(data, admitted_ids, khop_num_neighbors, batch_size=1, shuffle=False)
        self.entries.extend(g.cpu() for g in loader)
        return len(positions)

    def __len__(self) -> int:
        return len(self.entries)

    def sample_replay_batch(self, replay_batch_size: int) -> List[Data]:
        import random
        if not self.entries:
            return []
        k = min(replay_batch_size, len(self.entries))
        return random.sample(self.entries, k)

    def state_dict(self):
        return {"entries": list(self.entries)}

    def load_state_dict(self, state) -> None:
        self.entries = list(state["entries"])
