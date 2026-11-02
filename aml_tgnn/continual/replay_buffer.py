"""Inductive graph episodic replay buffer (CLAUDE.md line 21 / RTM REQ-2.3, REQ-2.4).

**Frozen-at-admission design** (RTM REQ-5.10, decided 2026-08-20): the moment a
node clears the confidence threshold, its k-hop subgraph is extracted right
then via `NeighborLoader` and stored as-is -- a self-contained `Data` object,
not a `(task, node_id)` pointer. Every later replay of that node reuses the
exact same frozen subgraph. This is a deliberate reversal of this project's
earlier "resample fresh every time" design (CLAUDE.md's original spec: "sample
entirely new k-hop subgraphs... at training time"): the user explicitly asked
for the topology/context to be captured and locked in at the moment the node
was flagged, rather than redrawn (and therefore potentially different, since
`NeighborLoader` sampling is itself stochastic when a node has more neighbors
than `num_neighbors` allows) on every subsequent replay.

Admission uses reservoir sampling (Vitter, 1985) rather than plain FIFO or
top-confidence eviction: FIFO would let old tasks fall out entirely once the
buffer fills, defeating the point of protecting against forgetting the *far*
past; keeping only the highest-confidence nodes ever seen would let a handful
of easy, high-confidence tasks crowd out everything else. Reservoir sampling
keeps each admitted (past-confidence-threshold) node roughly uniformly likely
to survive regardless of when it arrived.

`sample_replay_batch()` returns a *list* of single-node mini-batches (one per
sampled buffer entry) rather than one merged `Data` object -- each was frozen
independently, at a different point in training, from a potentially different
task's graph, so there's no single common node-index space to merge them into.
The training loop runs the model on each returned mini-batch and sums the
resulting losses into the current step's total loss -- equivalent to "mixing
into the current batch" for gradient purposes.

`ReplayEntry.task_id` is kept as provenance metadata (which task originally
flagged this node) even though nothing routes on it any more -- parameter
isolation (per-task heads) was removed (RTM REQ-5.15), so there's no more
"replay through that task's own head" concept; every replayed node just goes
through the one shared `ContinualTGNN.forward(batch)`, no task_id needed.
"""
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch_geometric.data import Data

from ..config import ReplayConfig
from ..data.temporal_loader import build_temporal_link_loader, build_temporal_neighbor_loader


@dataclass
class ReplayEntry:
    task_id: object
    local_node_id: int
    confidence: float
    subgraph: Data  # frozen k-hop ego-subgraph, captured at admission time; batch_size == 1


class GraphReplayBuffer:
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.entries: List[ReplayEntry] = []
        self._n_seen = 0

    def add_candidates(
        self,
        data: Data,
        task_id,
        node_local_ids: torch.Tensor,
        confidences: torch.Tensor,
    ) -> None:
        """REQ-2.3/REQ-5.10: admit nodes whose predicted-class confidence on the
        task they were labeled illicit in clears `cfg.confidence_threshold`,
        extracting and freezing each admitted node's `cfg.khop_num_neighbors`
        subgraph from `data` right now -- not a pointer resolved later. Call
        this once per task, after training on it, with the confidences the
        just-trained model assigns to its own predicted-illicit nodes.
        """
        keep_mask = confidences >= self.cfg.confidence_threshold
        if not keep_mask.any():
            return
        admitted_ids = node_local_ids[keep_mask]
        admitted_confs = confidences[keep_mask].tolist()

        # batch_size=1, shuffle=False: NeighborLoader yields one single-node
        # subgraph per input node, in input order (verified empirically) --
        # safe to zip directly against admitted_confs below.
        loader = build_temporal_neighbor_loader(
            data, admitted_ids, self.cfg.khop_num_neighbors, batch_size=1, shuffle=False
        )
        entries = [
            ReplayEntry(task_id=task_id, local_node_id=int(local_id), confidence=float(conf), subgraph=subgraph.cpu())
            for local_id, conf, subgraph in zip(admitted_ids.tolist(), admitted_confs, loader)
        ]
        self._admit(entries)

    def add_candidate_edges(
        self,
        data: Data,
        task_id,
        edge_idx: torch.Tensor,
        confidences: torch.Tensor,
    ) -> None:
        """RTM REQ-5.37: the edge-classification analogue of `add_candidates`
        above -- admits confident illicit EDGES (transactions) instead of
        nodes, freezing a k-hop subgraph around BOTH endpoints of each via
        `build_temporal_link_loader` (`LinkNeighborLoader` under the hood).
        The frozen subgraph already carries its own `edge_label_index`/
        `edge_label` (set by the loader at construction time), so replaying
        it later is just `ContinualTGNN.forward(frozen_subgraph)` -- identical
        in shape to the node-mode replay path, no separate edge-replay logic
        needed anywhere else.

        `edge_idx` are GLOBAL indices into `data.edge_index` (the caller's
        responsibility to have already converted from a `build_temporal_link_
        loader` batch's own `input_id`-relative positions, mirroring how node
        mode's caller converts a batch-local id via `batch.n_id` before
        calling `add_candidates`).
        """
        keep_mask = confidences >= self.cfg.confidence_threshold
        if not keep_mask.any():
            return
        admitted_idx = edge_idx[keep_mask]
        admitted_confs = confidences[keep_mask].tolist()

        loader = build_temporal_link_loader(
            data, admitted_idx, self.cfg.khop_num_neighbors, batch_size=1, shuffle=False
        )
        entries = [
            ReplayEntry(task_id=task_id, local_node_id=-1, confidence=float(conf), subgraph=subgraph.cpu())
            for conf, subgraph in zip(admitted_confs, loader)
        ]
        self._admit(entries)

    def _admit(self, entries: List["ReplayEntry"]) -> None:
        """RTM REQ-5.37: reservoir-sampling admission (Vitter, 1985), shared by
        both `add_candidates` (node mode) and `add_candidate_edges` (edge
        mode) -- factored out so the two modes can't silently drift apart on
        this shared mechanism.
        """
        for entry in entries:
            self._n_seen += 1
            if len(self.entries) < self.cfg.buffer_capacity:
                self.entries.append(entry)
            else:
                j = random.randint(0, self._n_seen - 1)
                if j < self.cfg.buffer_capacity:
                    self.entries[j] = entry

    def __len__(self) -> int:
        return len(self.entries)

    def state_dict(self) -> Dict[str, object]:
        """RTM REQ-5.3/5.4/5.10: each entry's frozen `subgraph` is a small
        self-contained k-hop ego-net (bounded by `khop_num_neighbors`'s
        fan-out), so persisting it directly is cheap -- unlike the old
        pointer-based design, there's no task-level graph to rehydrate from
        any more; the frozen subgraph *is* the state.
        """
        return {
            "entries": [
                {"task_id": e.task_id, "local_node_id": e.local_node_id, "confidence": e.confidence, "subgraph": e.subgraph}
                for e in self.entries
            ],
            "n_seen": self._n_seen,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.entries = [
            ReplayEntry(
                task_id=rec["task_id"],
                local_node_id=rec["local_node_id"],
                confidence=rec["confidence"],
                subgraph=rec["subgraph"],
            )
            for rec in state["entries"]
        ]
        self._n_seen = state["n_seen"]

    def sample_replay_batch(self) -> List[Data]:
        """REQ-2.4/REQ-5.10: draw up to `cfg.replay_batch_size` buffered entries
        and return their frozen subgraphs as-is -- no resampling.

        Each returned `Data` already has the usual `batch_size`-seed-nodes-first
        convention (`batch_size == 1`, this is a single-node ego-net) -- ready
        to feed straight into `ContinualTGNN.forward(batch)`.
        """
        if not self.entries:
            return []

        k = min(self.cfg.replay_batch_size, len(self.entries))
        sampled = random.sample(self.entries, k)
        return [e.subgraph for e in sampled]
