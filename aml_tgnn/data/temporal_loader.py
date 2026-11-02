"""Inductive subgraph mini-batching (CLAUDE.md line 12 / RTM REQ-1.1).

Wraps PyG's `NeighborLoader` so the model only ever sees localized k-hop
subgraphs sampled around a set of seed/target nodes -- never the global
adjacency matrix. This is what makes the model inductive: at inference time
an unseen account is just a new seed node with its own freshly-sampled
neighborhood, not a row that had to exist in some fixed global matrix during
training.

Both `EllipticTemporalDataset` and `IBMAMLTemporalDataset` (REQ-3.1/3.2) expose
one `torch_geometric.data.Data` snapshot per temporal task with `edge_time`
attached to `edge_attr`-adjacent storage, so a single loader factory works for
either dataset; the difference between datasets lives entirely in the adapters,
not here.
"""
from typing import List, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader


def build_temporal_neighbor_loader(
    data: Data,
    seed_nodes: Sequence[int],
    num_neighbors: List[int],
    batch_size: int,
    shuffle: bool = True,
) -> NeighborLoader:
    """Build a `NeighborLoader` restricted to a temporal task's snapshot.

    `data` must carry `x`, `edge_index`, `edge_attr`, `edge_sign`, `edge_time`,
    and `y` (node labels; unlabeled nodes should use -1 and be filtered out of
    `seed_nodes` by the caller). `NeighborLoader` samples a fresh, independent
    k-hop neighborhood around each seed node per epoch -- exactly the inductive
    subgraph sampling REQ-1.1 requires, and the same primitive the replay
    buffer (REQ-2.4) reuses to resample around historical nodes.
    """
    input_nodes = torch.as_tensor(list(seed_nodes), dtype=torch.long)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=shuffle,
        replace=False,
        directed=True,
    )


def build_temporal_link_loader(
    data: Data,
    edge_idx: torch.Tensor,
    num_neighbors: List[int],
    batch_size: int,
    shuffle: bool = True,
) -> LinkNeighborLoader:
    """RTM REQ-5.37: the edge-classification analogue of
    `build_temporal_neighbor_loader` above -- samples a k-hop neighborhood
    around BOTH endpoints of each target edge in `edge_idx` (local indices
    into `data.edge_index`, e.g. from `stratified_train_val_coldstart_split_
    edges`), for IBM AML's native edge classification (see
    `data/ibm_aml_dataset.py`'s module docstring). Elliptic's own node-
    classification path is untouched and keeps using
    `build_temporal_neighbor_loader`.

    Confirmed directly (not assumed from PyG's docs) before relying on this:
    the target edge itself remains part of the returned batch's message-
    passing `edge_index` -- unlike link-*prediction* tasks, which typically
    mask the target edge out to avoid it trivially informing its own
    prediction, edge *property* classification needs the edge's own features
    (amount, currency, format) to reach both endpoints via ordinary message
    passing; there is nothing to leak, since the label itself is never part
    of `edge_attr`. `batch.edge_label_index` gives the LOCAL (batch-remapped)
    endpoint indices for each target edge, and `batch.edge_label` its label,
    both supplied by PyG directly -- no custom bookkeeping needed, unlike
    node classification's own `seed_batch_targets` helper below.
    """
    edge_label_index = data.edge_index[:, edge_idx]
    edge_label = data.edge_y[edge_idx]
    return LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
        replace=False,
        directed=True,
    )


def seed_batch_targets(batch: Data) -> torch.Tensor:
    """Return the indices, within `batch.x`, of the seed nodes for this
    mini-batch. PyG's NeighborLoader always places seed nodes first and
    records how many in `batch.batch_size` -- callers use this to slice the
    backbone's per-node output down to just the nodes being classified/replayed.
    """
    return torch.arange(batch.batch_size, dtype=torch.long)
