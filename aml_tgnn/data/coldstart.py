"""Genuine held-out-node inductive generalization test (RTM REQ-5.30).

RACLA's own problem formulation (Definition 2, BasePaper1_RACLA.md) fixes
the node set `V` for the life of the graph -- it has no way to even pose the
question this module answers: can the model classify a node it has never
seen at all, using nothing but that node's local neighborhood, revealed for
the first time at test time? This is the standard way inductive-GNN papers
(e.g. GraphSAGE) demonstrate inductive capability, and the one this project's
own NeighborLoader-based sampling was built around from the start (CLAUDE.md
line 12) but had never actually been put to this specific test.

Two pieces, used together every task when `--cold-start-ratio > 0`:
  1. `data/split.py::stratified_train_val_coldstart_split` decides *which*
     nodes are held out (a third split, alongside train/val).
  2. `prune_edges_for_nodes` (below) removes every edge touching those nodes
     from the graph *training actually sees* -- not just excluding them as
     loss targets, which would still let their edges passively shape other
     nodes' neighborhoods during training (information leakage a truly
     "never seen" claim can't tolerate). Node rows themselves (`x`/`y`) are
     left in place, just edge-isolated, so no other node's index shifts.
"""
import torch
from torch_geometric.data import Data


def prune_edges_for_nodes(data: Data, node_ids: torch.Tensor) -> Data:
    """Returns a new `Data` sharing `x`/`y`/`num_nodes` with `data` but with
    every edge where the source or destination is in `node_ids` removed --
    the pruned nodes become edge-isolated, contributing nothing to any other
    node's sampled neighborhood, and nothing can be sampled *from* them
    either (an isolated node's own NeighborLoader draw just yields an empty
    neighborhood at every hop). Node indices are untouched (no row is
    deleted), so every other node's id, and every existing checkpoint's
    assumptions about node ordering, stay valid.
    """
    if node_ids.numel() == 0:
        return data

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[node_ids] = True

    src, dst = data.edge_index[0], data.edge_index[1]
    keep = ~(mask[src] | mask[dst])

    pruned = Data(
        x=data.x,
        y=data.y,
        edge_index=data.edge_index[:, keep],
        edge_attr=data.edge_attr[keep],
        edge_sign=data.edge_sign[keep],
        edge_time=data.edge_time[keep],
        num_nodes=data.num_nodes,
    )
    return pruned


def prune_specific_edges(data: Data, edge_idx: torch.Tensor) -> "tuple[Data, torch.Tensor]":
    """RTM REQ-5.37: the edge-classification cold-start analogue of
    `prune_edges_for_nodes` above -- removes exactly the given EDGES (by
    their own local index into `data.edge_index`, from
    `stratified_train_val_coldstart_split_edges`'s cold-start return) from
    the graph training actually sees, leaving every node (including their
    endpoint accounts) and every other edge completely untouched. Unlike
    node-holdout cold-start, a held-out edge's endpoints stay fully active
    in training via whatever other transactions touch them -- only this
    specific transaction is "never seen" during training, restored (added
    back to the graph, alongside `edge_y`) only at evaluation.

    Unlike `prune_edges_for_nodes` -- which explicitly preserves every
    surviving NODE's index, since removing edges doesn't shift node rows --
    removing specific EDGES necessarily shifts every subsequent edge's
    position in the compacted array. So this returns `(pruned_data, remap)`,
    not just the pruned `Data`: `remap` is a `[data.edge_index.size(1)]`
    tensor mapping each ORIGINAL edge index to its position in
    `pruned_data.edge_index` (`-1` for a pruned-out edge). The caller must
    apply `remap` to any train/val edge indices computed against the
    ORIGINAL (unpruned) edge ordering before using them against
    `pruned_data` -- train.py does this immediately after calling this
    function, exactly the bug this two-value return exists to prevent
    (caught directly: a first version returning only `pruned_data` let
    `train_nodes`/`val_nodes` silently index the wrong edges once any
    cold-start edges preceded them in the original ordering).

    `edge_y` (RTM REQ-5.37): the edge classification adapter's per-edge label
    tensor, pruned identically to `edge_attr`/`edge_sign`/`edge_time` so it
    stays index-aligned with the surviving edges. Optional (defaults to
    `None`) since node-classification `Data` objects have no such attribute.
    """
    n_edges = data.edge_index.size(1)
    if edge_idx.numel() == 0:
        return data, torch.arange(n_edges, dtype=torch.long)

    keep = torch.ones(n_edges, dtype=torch.bool)
    keep[edge_idx] = False

    remap = torch.full((n_edges,), -1, dtype=torch.long)
    remap[keep] = torch.arange(int(keep.sum()), dtype=torch.long)

    edge_y = getattr(data, "edge_y", None)
    pruned = Data(
        x=data.x,
        y=data.y,
        edge_index=data.edge_index[:, keep],
        edge_attr=data.edge_attr[keep],
        edge_sign=data.edge_sign[keep],
        edge_time=data.edge_time[keep],
        edge_y=edge_y[keep] if edge_y is not None else None,
        num_nodes=data.num_nodes,
    )
    return pruned, remap
