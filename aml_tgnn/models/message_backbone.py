"""Inductive, signed, temporal message-passing backbone.

Implements CLAUDE.md Part 1 (lines 11-17) / RTM rows REQ-1.1..REQ-1.6, REQ-D2, REQ-D3:

  sampled subgraph -> signed per-edge message -> group by (target, source neighbor)
  -> chronological sort -> LSTM summary per relation -> GATv2-style attention over
  relations -> softmax-weighted aggregation -> updated target node representation.

Built as a single `torch_geometric.nn.MessagePassing` subclass so the exact
message()/aggregate()/update() hooks required by the deliverable are visible and
testable in isolation. `aggregate()` is overridden (aggr=None) because the required
behaviour -- per-relation temporal grouping + LSTM + attention -- is not a plain
scatter-reduce and cannot be expressed via the built-in `aggr` string.

REQ-5.17 (LAS-GNN-style bidirectional signed message passing, user request
2026-08-21): every edge is aggregated at BOTH endpoints, not just its destination
-- a +1-signed message at the receiving node ("money came in") and a -1-signed
message at the sending node ("money went out"), scaling the same message vector
rather than routing through a separate embedding. This is a genuinely new signal,
independent of REQ-1.2's `edge_sign` field (credit/debit per CLAUDE.md L13; in
practice a same/cross-currency flag for IBM AML, a constant no-op for Elliptic --
see `data/ibm_aml_dataset.py`/`data/elliptic_dataset.py`). See `forward()`/`message()`
below and ARCHITECTURE.md's "Bidirectional signed message passing" section.

Batching (REQ-1.1) lives in `aml_tgnn/data/temporal_loader.py`; this module only
consumes the per-batch `Data` object a `NeighborLoader` yields.
"""
import dataclasses
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax as pyg_softmax

from ..config import BackboneConfig


class RelationalLSTMEncoder(nn.Module):
    """LSTM over one temporally-sorted message sequence per (source, target) relation.

    REQ-1.4: extracts the LSTM's final hidden state as the summary vector for that
    edge relationship. REQ-D3: sequences have different lengths (a source neighbor
    may have sent 1 or 50 transactions to the target), so they are padded with
    `pad_sequence` and packed with `pack_padded_sequence` before the LSTM call --
    this both skips compute on padding and keeps cuDNN's fused LSTM kernel usable.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, sequences: List[Tensor]) -> Tensor:
        """sequences[r]: [len_r, input_dim], already sorted ascending by time.

        Returns [R, hidden_dim], one summary vector per relation, in input order.
        """
        if len(sequences) == 0:
            return torch.zeros(0, self.hidden_dim, device=self.lstm.weight_ih_l0.device)

        lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True)  # [R, max_len, input_dim]
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]  # final layer's final hidden state, [R, hidden_dim]


class InductiveSignedTemporalBackbone(MessagePassing):
    """The shared TGNN aggregator (CLAUDE.md steps 2-6). This is the module EWC
    protects (REQ-2.1/2.2) -- its parameters are what accumulate a Fisher penalty
    across tasks, as opposed to the per-task dynamic heads which are deliberately
    left unregularized (parameter isolation, REQ-2.5).
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__(aggr=None, flow="source_to_target", node_dim=0)
        self.cfg = cfg

        # RTM REQ-5.32: resolved once here via getattr, not read off `self.cfg`
        # directly at call time in forward()/aggregate()/update() -- a `cfg`
        # unpickled from a checkpoint saved *before* this row existed has no
        # `bidirectional_signed`/`use_lstm`/`use_attention` attributes at all
        # (dataclass field defaults only apply at construction time, not to an
        # object unpickled straight from an older class version's saved
        # `__dict__` -- the same backward-compatibility gap `cold_start_ratio`
        # already had, REQ-5.31's `getattr(..., 0.0)` fix). Defaulting `True`
        # here reproduces exactly what those older checkpoints were actually
        # trained with (the full, pre-ablation model).
        self.bidirectional_signed = getattr(cfg, "bidirectional_signed", True)
        self.use_lstm = getattr(cfg, "use_lstm", True)
        self.use_attention = getattr(cfg, "use_attention", True)

        # REQ-1.2: signed message construction (source feats + edge feats + sign)
        self.sign_embedding = nn.Embedding(2, cfg.sign_dim)
        msg_in_dim = cfg.node_in_dim + cfg.edge_in_dim + cfg.sign_dim
        self.message_proj = nn.Linear(msg_in_dim, cfg.message_dim)

        # REQ-1.4 (+ REQ-D3): per-relation temporal LSTM
        self.rel_encoder = RelationalLSTMEncoder(
            cfg.message_dim, cfg.lstm_hidden_dim, cfg.lstm_layers, cfg.dropout
        )
        # RTM REQ-5.32 (ablation, cfg.use_lstm=False): order-blind substitute for the
        # LSTM above -- mean-pools a relation's message sequence (no temporal signal
        # survives) then projects message_dim -> lstm_hidden_dim so aggregate()'s
        # output shape is identical either way and every downstream consumer
        # (target_proj/rel_key_proj/rel_out_proj, all lstm_hidden_dim-sized) needs
        # no branching of its own. Always constructed (even when cfg.use_lstm=True)
        # so a single saved checkpoint's state_dict shape doesn't depend on which
        # ablation arm produced it; simply unused (no gradient) when not selected.
        self.pool_proj = nn.Linear(cfg.message_dim, cfg.lstm_hidden_dim)

        # REQ-1.5/1.6: GATv2-style attention (Brody et al. 2021) of the target
        # node's current state over its relation summaries. Verified against
        # torch_geometric.nn.conv.GATv2Conv's actual message() implementation
        # (x = x_i + x_j; x = leaky_relu(x); alpha = (x * att).sum(-1)): two
        # SEPARATE linear projections -- one for the query/target side, one for
        # the key/relation side -- are ADDED (not concatenated) and passed
        # through LeakyReLU BEFORE the dot product with the attention vector.
        # This ordering (nonlinearity after combining query+key, not before) is
        # what fixes original GAT's "static attention" limitation: the ranking
        # of scores can now actually depend on the query, not just the key.
        self.target_proj = nn.Linear(cfg.node_in_dim, cfg.lstm_hidden_dim)       # query (target) projection
        self.rel_key_proj = nn.Linear(cfg.lstm_hidden_dim, cfg.lstm_hidden_dim)  # key (relation summary) projection
        self.attn_vector = nn.Parameter(torch.empty(cfg.lstm_hidden_dim))
        nn.init.xavier_uniform_(self.attn_vector.view(1, -1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        self.rel_out_proj = nn.Linear(cfg.lstm_hidden_dim, cfg.node_out_dim)
        self.root_lin = nn.Linear(cfg.node_in_dim, cfg.node_out_dim)  # self-loop / residual term
        self.norm = nn.LayerNorm(cfg.node_out_dim)
        self.out_dropout = nn.Dropout(cfg.dropout)

        # RTM REQ-5.18 (TWP): last forward pass's post-softmax attention coefficients,
        # set inside update(). Not a buffer/parameter -- a plain attribute refreshed
        # every forward() call, read by TWP's topology-importance computation only.
        self.last_alpha: Tensor = torch.zeros(0)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_sign: Tensor,
        edge_time: Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """One aggregator layer over a sampled inductive subgraph.

        x: [N, node_in_dim] features for every node in the sampled k-hop batch
           (seed/target nodes are the first `batch_size` rows, per PyG's
           NeighborLoader convention -- callers slice that off, this layer does not
           assume it).
        edge_index: [2, E] source_to_target directed edges within the subgraph.
        edge_attr: [E, edge_in_dim] transaction/edge features.
        edge_sign: [E] long tensor in {0, 1} (e.g. debit/credit, or a dataset-
           specific flag -- see REQ-1.2's row in RTM.md for what this actually
           holds per dataset). Independent of the direction sign below.
        edge_time: [E] numeric timestamp per edge, any monotonic unit.

        Returns [N, node_out_dim]: updated representation for every node that had
        at least one edge (incoming OR outgoing, REQ-5.17) touching it in this
        subgraph; nodes with neither get a pure-residual (root-transform only)
        representation.
        """
        # REQ-5.17: expand every edge into two message-passing edges so both
        # endpoints aggregate it -- the original direction (receiver's view,
        # sign +1) and the reverse (sender's view, sign -1). `flow="source_to_
        # target"` + a single propagate() over the unmodified edge_index would
        # only ever let the *destination* node see this edge; the flipped half
        # is what lets the *source* node see it too, from `edge_index.flip(0)`
        # putting the original destination in the "x_j" (source) slot and the
        # original source in the "index"/target slot for that half.
        #
        # RTM REQ-5.32 (ablation, cfg.bidirectional_signed=False): skip the
        # expansion entirely -- propagate over the original edge_index once,
        # direction fixed at +1 (message() scaling becomes a no-op), reverting
        # to plain single-direction (destination-only) message passing.
        if self.bidirectional_signed:
            num_edges = edge_index.size(1)
            edge_index_use = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr_use = torch.cat([edge_attr, edge_attr], dim=0)
            edge_sign_use = torch.cat([edge_sign, edge_sign], dim=0)
            edge_time_use = torch.cat([edge_time, edge_time], dim=0)
            edge_direction = torch.cat([x.new_ones(num_edges), x.new_full((num_edges,), -1.0)], dim=0)
        else:
            edge_index_use = edge_index
            edge_attr_use = edge_attr
            edge_sign_use = edge_sign
            edge_time_use = edge_time
            edge_direction = x.new_ones(edge_index.size(1))
        return self.propagate(
            edge_index_use,
            x=x,
            edge_attr=edge_attr_use,
            edge_sign=edge_sign_use,
            edge_direction=edge_direction,
            edge_time=edge_time_use,
            size=size,
        )

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_sign: Tensor, edge_direction: Tensor) -> Tensor:
        """REQ-1.2 + REQ-5.17: unified message vector per directed edge, scaled
        by its direction (+1 incoming / -1 outgoing, LAS-GNN-style signed
        message passing). The scaling happens on the *same* projected message
        (not a separate learned embedding) so aggregate()'s per-relation LSTM
        sees one chronologically-merged, sign-tagged stream per neighbor pair,
        covering both directions of traffic with that neighbor.
        """
        sign_emb = self.sign_embedding(edge_sign)
        raw = torch.cat([x_j, edge_attr, sign_emb], dim=-1)
        msg = self.message_proj(raw)  # [E_bidir, message_dim]
        return msg * edge_direction.unsqueeze(-1)

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        edge_index_j: Tensor,
        edge_time: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """REQ-1.3 + REQ-1.4: group per-edge messages by (target, source-neighbor)
        relation, sort each group ascending by time, run each sorted sequence
        through the LSTM. `index` (== edge_index_i) is the target id per edge;
        `edge_index_j` is the source-neighbor id per edge -- both requested by
        name and auto-supplied by MessagePassing's collector. Since REQ-5.17,
        `inputs` (== `message()`'s output) already carries the bidirectional
        edges and their +-1 sign scaling -- this method needs no changes for
        that: a relation group naturally merges a pair's incoming and outgoing
        history together, sign already baked into each entry's values.

        Returns (rel_summary [R, lstm_hidden_dim], rel_target [R]) -- deferred to
        update() rather than reduced here, because turning R relation summaries
        into N node representations requires the attention step in REQ-1.5/1.6,
        which needs each target's own current state (only available via `x`, and
        `update()` is the hook that receives `x` alongside the aggregate output).
        """
        device = inputs.device
        if inputs.size(0) == 0:
            return (
                torch.zeros(0, self.cfg.lstm_hidden_dim, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )
        if not self.use_lstm:
            # RTM REQ-5.32 (ablation): still group by relation and keep the return
            # shape identical, but skip time-sorting entirely -- order-blindness is
            # the whole point of this arm, so the sort below is unnecessary work.
            pairs = torch.stack([index, edge_index_j], dim=1)
            uniq_pairs, rel_id = torch.unique(pairs, dim=0, return_inverse=True)
            rel_id = rel_id.view(-1)
            rel_target = uniq_pairs[:, 0]
            pooled = scatter(inputs, rel_id, dim=0, dim_size=uniq_pairs.size(0), reduce="mean")
            rel_summary = self.pool_proj(pooled)
            return rel_summary, rel_target

        pairs = torch.stack([index, edge_index_j], dim=1)  # [E, 2]
        uniq_pairs, rel_id = torch.unique(pairs, dim=0, return_inverse=True)
        rel_id = rel_id.view(-1)
        R = uniq_pairs.size(0)
        rel_target = uniq_pairs[:, 0]  # [R]

        # Multi-key stable sort: sort by time, then stably sort that order by
        # rel_id, so each relation's slice is internally time-ascending.
        time_order = torch.argsort(edge_time, stable=True)
        rel_sort_order = torch.argsort(rel_id[time_order], stable=True)
        final_order = time_order[rel_sort_order]

        sorted_inputs = inputs[final_order]
        sorted_rel_id = rel_id[final_order]
        counts = torch.bincount(sorted_rel_id, minlength=R)

        sequences = list(torch.split(sorted_inputs, counts.tolist()))
        rel_summary = self.rel_encoder(sequences)  # [R, lstm_hidden_dim]

        return rel_summary, rel_target

    def update(self, aggr_out: Tuple[Tensor, Tensor], x: Tensor, dim_size: Optional[int] = None) -> Tensor:
        """REQ-1.5/1.6: attention between each target's current state and its
        relation summaries, softmax-normalized per target, weighted sum -> updated
        node representation. Combined with a root/residual transform of the node's
        own prior features (standard GraphSAGE/GAT-style self term) so nodes with
        no incoming edges in this subgraph still get a defined output.
        """
        rel_summary, rel_target = aggr_out
        N = dim_size if dim_size is not None else x.size(0)

        if rel_summary.size(0) == 0:
            node_msg = torch.zeros(N, self.cfg.node_out_dim, device=x.device)
            self.last_alpha = torch.zeros(0, device=x.device)
        else:
            if self.use_attention:
                target_q = self.target_proj(x[rel_target])     # [R, lstm_hidden_dim], query side
                rel_k = self.rel_key_proj(rel_summary)          # [R, lstm_hidden_dim], key side
                # GATv2 scoring: a^T . LeakyReLU(W_query . target + W_key . relation)
                # -- addition + LeakyReLU BEFORE the attention-vector dot product,
                # not concatenation + LeakyReLU after (that would be original GAT).
                e = (self.leaky_relu(target_q + rel_k) * self.attn_vector).sum(dim=-1)  # [R]
                alpha = pyg_softmax(e, rel_target, num_nodes=N)  # softmax per target group
                # RTM REQ-5.18 (TWP): stash the post-softmax, pre-dropout coefficients --
                # these are this layer's analogue of the GAT attention coefficient e_ij
                # in Liu, Yang & Wang 2021 Eq. 4-6, kept differentiable (no .detach())
                # so TWP's topological-importance gradient (Eq. 6) can flow through it.
                # Dropout is a training-only regularization detail, not part of "the
                # attention/aggregation strategy" the topology term is meant to protect,
                # so this is captured *before* attn_dropout is applied below.
                self.last_alpha = alpha
                alpha = self.attn_dropout(alpha)
            else:
                # RTM REQ-5.32 (ablation, cfg.use_attention=False): uniform weights per
                # target -- 1/(number of relations feeding that target) -- i.e. a plain
                # mean over relation summaries instead of a learned, query-dependent
                # weighting. Built purely from index counts (no learnable tensor
                # participates), so this carries no gradient; TWP's importance
                # computation already guards on `topo_loss.requires_grad` (RTM
                # REQ-5.28) for exactly this "nothing to differentiate" case, so it
                # degrades to zero topology-importance here rather than crashing --
                # correct, since there is no learned attention mechanism to protect.
                counts = scatter(torch.ones_like(rel_target, dtype=rel_summary.dtype), rel_target, dim=0, dim_size=N, reduce="sum")
                alpha = 1.0 / counts[rel_target].clamp(min=1.0)
                self.last_alpha = alpha
            weighted = alpha.unsqueeze(-1) * self.rel_out_proj(rel_summary)  # [R, node_out_dim]
            node_msg = scatter(weighted, rel_target, dim=0, dim_size=N, reduce="sum")

        root = self.root_lin(x)
        out = self.norm(node_msg + root)
        return self.out_dropout(F.relu(out))


class StackedSignedTemporalBackbone(nn.Module):
    """Stacks `cfg.num_hops` independent `InductiveSignedTemporalBackbone` layers
    so the model actually performs multi-hop reasoning over the sampled
    subgraph, matching `NeighborLoader`'s sampled hop depth.

    A single `InductiveSignedTemporalBackbone` layer only aggregates *direct*
    (1-hop) edges: `message()` builds each edge's message from the source
    node's *raw* input features (`x_j`), and `forward()` was previously called
    exactly once. Even though `NeighborLoader` samples e.g. a 2-hop
    neighborhood (`num_neighbors=[15, 10]`), a single-call backbone never lets
    a 2nd-hop node's features reach a seed node's representation -- that
    signal only flows through *if* the 1-hop node's own representation is
    first updated using its 2-hop neighbors, then the seed node is updated
    using that already-updated 1-hop representation. This class does exactly
    that: `layer[0]` consumes the raw node features over the sampled edges;
    `layer[k>0]` consumes `layer[k-1]`'s *output* as its node features, over
    the *same* edges (same static sampled subgraph throughout) -- the
    standard multi-layer MessagePassing GNN pattern (e.g. stacking `GCNConv`
    layers), applied here with each layer independently parameterized rather
    than weight-tied.

    `cfg.num_hops` should match `len(LoaderConfig.num_neighbors)` (and
    `ReplayConfig.khop_num_neighbors`, for replayed subgraphs) -- a deeper
    stack than the sampled hop depth has nothing further to aggregate at its
    outer layers, and a shallower stack (the old default behavior at
    `num_hops=1`) leaves sampled outer-hop nodes structurally present but
    functionally unused, as described above.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        if cfg.num_hops < 1:
            raise ValueError(f"BackboneConfig.num_hops must be >= 1, got {cfg.num_hops}")
        self.cfg = cfg

        layer_cfgs = [cfg] + [
            dataclasses.replace(cfg, node_in_dim=cfg.node_out_dim) for _ in range(cfg.num_hops - 1)
        ]
        self.layers = nn.ModuleList(InductiveSignedTemporalBackbone(c) for c in layer_cfgs)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_sign: Tensor,
        edge_time: Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_sign, edge_time, size=size)
        return x

    def attention_coefficients(self) -> List[Tensor]:
        """RTM REQ-5.18 (TWP): the last forward pass's post-softmax attention
        coefficients from every stacked hop, generalizing Liu, Yang & Wang 2021's
        "middle layer" choice (Sec. Topology-aware Weight Preserving, after Eq. 6)
        to "every hop" -- with only `num_hops` (default 2) layers there is no
        unambiguous middle one, and summing every layer's topology-preserving
        signal is more complete than an arbitrary pick between the first and last.
        """
        return [layer.last_alpha for layer in self.layers]
