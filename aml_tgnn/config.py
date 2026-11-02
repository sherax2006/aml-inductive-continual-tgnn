"""Central hyperparameter configuration for the Inductive Hybrid Continual TGNN.

See RTM.md REQ-0.3. One dataclass covers backbone, continual-learning, and
data-loading knobs so a training script only needs to construct/override one
object instead of threading loose kwargs through every module.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BackboneConfig:
    node_in_dim: int = 64          # dim of raw node features after per-dataset adapter
    edge_in_dim: int = 16          # dim of raw edge features after per-dataset adapter
    sign_dim: int = 2              # one-hot credit/debit (or learned scalar embedding)
    message_dim: int = 128         # dim of constructed per-edge message vector
    lstm_hidden_dim: int = 128     # LSTM hidden size = per-relationship summary dim
    lstm_layers: int = 1
    attn_heads: int = 4
    node_out_dim: int = 128        # updated target-node representation dim
    dropout: float = 0.1
    num_hops: int = 2              # stacked aggregation rounds; must match (<=) the NeighborLoader's
                                    # sampled hop depth (len(LoaderConfig.num_neighbors)) or deeper hops
                                    # go unused -- see message_backbone.py::StackedSignedTemporalBackbone

    # Ablation switches (RTM REQ-5.32): each defaults to the full model's behavior;
    # flipping one to False swaps that component for the simplest architecturally-
    # sound substitute, holding everything else fixed, to isolate its contribution.
    bidirectional_signed: bool = True  # False: single-direction (target-only) message
                                        # passing, no +-1 direction scaling (REQ-5.17 off)
    use_lstm: bool = True              # False: mean-pool each relation's message sequence
                                        # instead of an LSTM final hidden state (order-blind)
    use_attention: bool = True         # False: uniform average over relation summaries
                                        # instead of GATv2-style softmax attention


@dataclass
class EWCConfig:
    lambda_ewc: float = 1000.0     # penalty weight
    fisher_num_batches: int = 50   # batches sampled from prior task to estimate Fisher
    gamma: float = 1.0             # online-EWC decay on the running Fisher (1.0 = plain accumulation)


@dataclass
class ReplayConfig:
    buffer_capacity: int = 2000        # max historical illicit nodes retained
    confidence_threshold: float = 0.85 # min softmax prob to admit a node into the buffer
    replay_batch_size: int = 32
    khop_num_neighbors: list = field(default_factory=lambda: [15, 10])


@dataclass
class GEMConfig:
    """RTM REQ-5.18: Gradient Episodic Memory (Lopez-Paz & Ranzato, NeurIPS 2017).
    Only used when `ExperimentConfig.continual_strategy == "gem_twp"`.
    """
    memory_budget: int = 2000          # total M across all tasks (m = M / num_tasks_so_far per task)
    khop_num_neighbors: list = field(default_factory=lambda: [15, 10])  # subgraph depth for stored memory examples
    qp_margin: float = 0.5             # paper's small constant added to v* -- biases projection toward
                                        # updates that favour beneficial backward transfer (Sec. 3, after Eq. 11)
    qp_iters: int = 200                # projected-gradient iterations for the dual QP solve (Eq. 11)
    qp_lr: float = 0.1                 # step size for the projected-gradient QP solve


@dataclass
class AGEMConfig:
    """RTM REQ-5.26: Averaged GEM (Chaudhry, Ranzato, Rohrbach & Elhoseiny,
    ICLR 2019, "Efficient Lifelong Learning with A-GEM") -- a cheaper
    drop-in replacement for vanilla GEM's per-task constraint set, added
    specifically because GEM's own real cost (one extra backward pass PER
    COMPLETED TASK, every step) becomes impractical late in a long task
    stream (RTM REQ-5.18's own accepted tradeoff at the time -- revisited
    once "later" arrived: task 44 of Elliptic's 49-task stream took
    ~200s/epoch under vanilla GEM, up from ~40-60s early on). A-GEM
    replaces GEM's per-task memory + (t-1)-variable QP with ONE unified,
    reservoir-sampled memory pool and a single closed-form gradient
    projection -- cost becomes O(1) per step instead of O(completed tasks).
    Only used when `ExperimentConfig.continual_strategy == "agem_twp"`.
    """
    memory_budget: int = 2000        # total examples in the single combined pool (reservoir-sampled
                                      # across the whole stream, not partitioned per task like GEMConfig's)
    khop_num_neighbors: list = field(default_factory=lambda: [15, 10])  # subgraph depth for stored examples
    sample_batch_size: int = 32      # examples drawn from the pool each step to compute the one reference
                                      # gradient (A-GEM's entire extra cost per step, regardless of stream length)


@dataclass
class TWPConfig:
    """RTM REQ-5.18: Topology-aware Weight Preserving (Liu, Yang & Wang, AAAI 2021).
    Only used when `ExperimentConfig.continual_strategy == "gem_twp"`.
    """
    # RTM REQ-5.19 (2026-08-23): retuned down from the paper's own reported value
    # (10000.0 for both) after a real Elliptic run showed it collapsing the model
    # to predicting the majority class every time -- Eq. 8/9's importance terms
    # are a *raw, unnormalized sum* over every backbone parameter (~450K here),
    # so the same constant that worked for the paper's much smaller GCN/GAT/GIN
    # architectures was 2-4 orders of magnitude too large at this parameter count,
    # completely dominating the actual cross-entropy loss. See RTM.md for the
    # real numbers this was derived from.
    lambda_l: float = 1.0        # task-related importance weight (Eq. 7) -- feeds Eq. 8's cross-task penalty
    lambda_t: float = 1.0        # topology-related importance weight (Eq. 7) -- feeds Eq. 8's cross-task penalty

    # RTM REQ-5.22/5.23 (2026-08-23): Eq. 9 disabled by default (beta=0.0), not just
    # re-tuned. Diagnosed precisely: on a task with no prior completed tasks, Eq. 8's
    # penalty is exactly zero and GEM's projection is a no-op, so Eq. 9 was the
    # *entire* difference between gem_twp and ewc_replay -- and a same-task,
    # same-class-weights, same-split comparison showed ewc_replay training a
    # genuinely discriminative classifier (F1=0.647) while gem_twp collapsed to
    # predicting the majority class every time, reproduced identically across three
    # independent (differently-initialized) runs. Root cause: Eq. 9 minimizes the
    # summed gradient magnitude across every parameter every batch -- a classifier
    # that predicts the majority class everywhere has small gradients almost
    # everywhere except on the rare batches with minority-class errors, which
    # produce exactly the larger gradient spikes a rare-class detector needs to
    # learn from. Minimizing that sum structurally suppresses those spikes,
    # independent of beta's scale being "reasonable" -- REQ-5.19's magnitude fix
    # didn't touch this because it isn't a magnitude problem.
    # `beta=0.0` disables Eq. 9's contribution completely (TWP.live_importance_l1
    # short-circuits to avoid the wasted double-backward compute) while leaving
    # Eq. 8 -- the actual cross-task forgetting-prevention penalty -- fully intact
    # at its REQ-5.19 calibration. This is a real, deliberate reduction in fidelity
    # to the paper's complete method (Eq. 9 is part of what they propose) in
    # exchange for a working classifier on this project's severely imbalanced AML
    # data; the mechanism itself is still fully implemented and can be re-enabled
    # by setting beta back to a nonzero value.
    beta: float = 0.0            # L1 sparsity weight on the live importance estimate (Eq. 9)
    importance_num_batches: int = 50  # batches used to estimate I_k at task end (Eq. 3/6/7) -- mirrors
                                       # EWCConfig.fisher_num_batches so the two strategies cost about the same


@dataclass
class MASConfig:
    """RTM REQ-5.27: Memory Aware Synapses (Aljundi, Babiloni, Elhoseiny,
    Rohrbach & Tuytelaars, ECCV 2018) -- an alternative regularization core
    to GEM/A-GEM's replay/projection role, paired with the same TWP
    regularizer `gem_twp`/`agem_twp` already use (`continual_strategy ==
    "mas_twp"`). Per an explicit user design choice, MAS takes over GEM/
    A-GEM's structural slot but does so via a SECOND additive regularization
    penalty stacked with TWP's own -- there is no replay buffer or memory
    pool in this strategy at all, unlike `gem_twp`/`agem_twp`.

    Importance measure (unlike EWC's squared-gradient empirical Fisher):
    `Omega_ij = (1/N) * sum_k |d(||F(x_k;theta)||^2_2) / d(theta_ij)|` --
    gradient MAGNITUDE (not squared) of the model's own output function's
    squared L2 norm, entirely label-free by construction (though this
    implementation draws it from the same task/batch stream as EWC/TWP's own
    cadence in this project, for consistency -- MAS's own headline selling
    point of using unlabeled data isn't separately exploited here). Same
    "online" running-sum-with-decay accumulation style as this project's own
    `EWCConfig.gamma`, and the same backbone-only penalty scoping (leaves the
    shared classification head free to specialize, matching EWC's own
    established convention).
    """
    lambda_mas: float = 1.0            # penalty weight, mirrors EWCConfig.lambda_ewc's role
    importance_num_batches: int = 50   # batches sampled from the just-finished task to estimate importance,
                                        # mirrors EWCConfig.fisher_num_batches / TWPConfig.importance_num_batches
    gamma: float = 1.0                 # online decay on the running importance estimate (1.0 = plain
                                        # accumulation), mirrors EWCConfig.gamma


@dataclass
class TaskHeadConfig:
    # RTM REQ-5.15 (2026-08-21): parameter isolation was removed from this project's
    # continual-learning strategy at the user's explicit request, deviating from
    # CLAUDE.md's original "Tri-Core" spec (line 22) -- there is now exactly one
    # classification head, shared across every task, always. No mode switch.
    hidden_dim: int = 64
    num_classes: int = 2   # licit / illicit


@dataclass
class LoaderConfig:
    num_neighbors: list = field(default_factory=lambda: [15, 10])
    batch_size: int = 256
    shuffle: bool = True


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs_per_task: int = 10
    device: str = field(default_factory=_default_device)  # auto-picks "cuda" when available, else "cpu"

    # RTM REQ-5.29: the torch/random seed controlling this run's model init, dropout,
    # NeighborLoader sampling, and replay/GEM/A-GEM sampling -- None (default) means
    # unseeded (this project's original behavior, unchanged for anyone not passing
    # --seed). Deliberately separate from `split_seed` below, which stays fixed
    # across a multi-seed comparison so every seed sees the identical train/val split.
    seed: Optional[int] = None

    # RTM REQ-5.30: genuine held-out-node inductive generalization test. 0.0 (default)
    # disables it entirely -- identical behavior to before this row existed. When > 0,
    # each task additionally reserves this fraction of its own labeled nodes (per class,
    # same stratification as val_ratio) as "cold-start" nodes: their edges are pruned
    # from the graph training actually sees (data/coldstart.py::prune_edges_for_nodes),
    # then they're scored at end-of-task against the *unpruned* graph -- the genuine
    # "never seen until this exact moment" test RACLA's fixed-node-set framing has no
    # analogue for at all.
    cold_start_ratio: float = 0.0

    # logging / checkpointing / resume (RTM Part 5)
    logs_root: str = "Logs"
    checkpoints_root: str = "Checkpoints"
    log_every_n_steps: int = 10        # console/log batch-progress line frequency

    # per-task evaluation (RTM Part 5, REQ-5.6)
    val_ratio: float = 0.2             # fraction of each task's labeled nodes held out for metrics
    split_seed: int = 42               # base seed for the per-task stratified train/val split

    # forward-transfer / forgetting evaluation (RTM Part 5, REQ-5.8)
    forgetting_anchor_task_ids: Optional[List] = None  # None = auto-pick the first 3 trained tasks

    # paper-matching evaluation (RTM REQ-5.14): the M matrix (i<=j only, micro-F1) +
    # Average Performance / Average Forgetting, printed once at the end of a run.
    # Costs T(T+1)/2 extra eval passes over the run's own just-saved checkpoints --
    # real for large T (e.g. Elliptic's default 49 tasks); set False to skip it.
    paper_style_eval: bool = True

    # RTM REQ-5.20: inverse-frequency class weighting on the training
    # cross-entropy loss (data/split.py::compute_class_weights), recomputed
    # fresh every task from that task's own train split. Default True --
    # confirmed on real data that plain unweighted cross-entropy on Elliptic's
    # heavily imbalanced illicit class has a stable trivial optimum (predict
    # the majority class always; high accuracy, exactly zero precision/recall
    # on the class that matters). Kept as a toggle (not made unconditional)
    # so an unweighted run stays reproducible if ever wanted for comparison.
    use_class_weights: bool = True


@dataclass
class RACLAConfig:
    """RTM REQ-5.25: RACLA (Zhang, Liu, Lan, Chen, Chen & Zhang, "RACLA:
    Role-aware continual learning for robust AML detection", Expert Systems
    With Applications 299 (2026) 130136) -- reimplemented as a genuinely
    separate baseline model (not a `continual_strategy` variant of
    `ContinualTGNN`, since its MoE-of-GAT-experts architecture is a different
    model entirely, not a different regularizer on the same backbone), for a
    controlled, same-evaluation-harness comparison against this project's own
    architecture. See BasePaper1_RACLA.md for the source formulas (Eq. 1-19)
    and `aml_tgnn/racla/` for the implementation.

    Several hyperparameters below are NOT stated in the paper's main text and
    are this project's own reasonable, documented choices -- flagged
    individually. This is standard, expected practice when reproducing a
    baseline without the original authors' code (see RTM.md REQ-5.25's row
    for the full reproducibility discussion); it means this reimplementation
    is faithful to RACLA's *described* method, not guaranteed to reproduce
    their *exact* reported numbers.
    """
    # --- AML Expert Layer (Eq. 1-9) ---
    num_experts: int = 3          # m. NOT stated numerically in the paper's main text; Fig. 1's own
                                   # "AML Expert Layer" inset literally draws exactly 3 experts
                                   # (Expert1/Expert2/Expert3), so 3 is used as a paper-grounded
                                   # default rather than an arbitrary guess.
    top_k: int = 2                # k, Eq. 4/5. Not stated; chosen < num_experts so Top-K sparsity
                                   # (the paper's own stated mechanism) is genuinely exercised.
    expert_hidden_dim: int = 128  # per-expert GAT output width. Not stated; matches this project's
                                   # own node_out_dim/lstm_hidden_dim convention for comparability.
    expert_heads: int = 1         # attention heads per expert GAT layer. Not stated.
    load_balance_lambda: float = 0.01  # combines the paper's two separate scaling constants, its own
                                        # w_importance (Eq. 8) and lambda (Eq. 9), into one knob --
                                        # both are pure multiplicative scalars on the same term before
                                        # it's added to the task loss, so keeping them as two separate,
                                        # redundant constants would add a parameter with no distinct
                                        # effect. Neither is given a numeric value in the paper.

    # --- Routing-based replay selection (Eq. 10-12) ---
    replay_coverage_r: float = 0.5        # r, Eq. 11 (adaptive distance threshold beta = r * mean pairwise dist). Not stated.
    replay_similarity_threshold: float = 0.5  # the paper's own "lambda" in Eq. 10 (cosine similarity
                                               # threshold) -- renamed here to avoid colliding with
                                               # load_balance_lambda, since the paper reuses the symbol
                                               # lambda for two unrelated quantities (see Table 1's own
                                               # note in BasePaper1_RACLA.md). Not given a numeric value.
    replay_buffer_ratio: float = 0.01     # ell as a fraction of a task's labeled nodes -- matches the
                                           # paper's own stated "1% of nodes" buffer budget (Sec. 5.3),
                                           # the one number in this config that IS directly from the text.
    khop_num_neighbors: list = field(default_factory=lambda: [15, 10])  # subgraph depth for frozen buffer
                                           # entries -- mirrors GEMConfig's own identically-named field
                                           # (RTM REQ-5.18): each buffer-holding strategy keeps its own copy
                                           # of this knob rather than sharing LoaderConfig's, since a buffer
                                           # entry's subgraph is frozen once at admission time and need not
                                           # track the main training loader's depth if that's ever retuned.

    # --- Structure learning for buffer (Eq. 13/14) ---
    edge_predictor_hidden_dim: int = 64   # MLP hidden width for S_ij = Sigmoid(MLP([h_i || h_j])). Not stated.
    edge_predictor_neg_ratio: float = 1.0  # negative:positive edge sample ratio for training D_t^edge. Not stated.

    # --- Drift detector (Eq. 15-19) ---
    drift_alpha: float = 0.05     # alpha, the pre-Bonferroni per-dimension KS significance level --
                                   # the standard statistical convention, not paper-specific.
    drift_tau: float = 0.5        # tau, the proportion-threshold (Eq. 19) -- this is the exact knob
                                   # RACLA's own Table 4 sweeps; 0.5 matches their Elliptic default row.


@dataclass
class ExperimentConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    gem: GEMConfig = field(default_factory=GEMConfig)
    agem: AGEMConfig = field(default_factory=AGEMConfig)
    mas: MASConfig = field(default_factory=MASConfig)
    twp: TWPConfig = field(default_factory=TWPConfig)
    task_head: TaskHeadConfig = field(default_factory=TaskHeadConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    racla: RACLAConfig = field(default_factory=RACLAConfig)

    # RTM REQ-5.18/5.26/5.27: which continual-learning strategy actually runs.
    # "ewc_replay" (default) is the existing, unchanged two-core strategy (EWC +
    # episodic replay). "gem_twp" swaps in Gradient Episodic Memory + Topology-aware
    # Weight Preserving instead. "agem_twp" swaps GEM specifically for Averaged GEM
    # (RTM REQ-5.26) -- same TWP regularization core, but a single O(1)-per-step
    # unified-memory projection instead of GEM's O(completed tasks)-per-step
    # per-task constraint set, added because that cost was observed to genuinely
    # bite late in a long task stream. "mas_twp" (RTM REQ-5.27) swaps GEM/A-GEM's
    # replay/projection role for Memory Aware Synapses instead -- a SECOND additive
    # regularization penalty stacked with TWP's own, not a memory mechanism at all
    # (no buffer, no projection step). All four are toggleable, coexisting
    # strategies, not replacements of one another -- specifically so they can be
    # compared head-to-head with the existing Task Performance Matrix / paper-style
    # evaluation harness. The `ewc`/`replay`, `gem`/`twp`, `agem`/`twp`, and
    # `mas`/`twp` fields above are always present on every config; only the fields
    # matching the active strategy are actually used.
    continual_strategy: str = "ewc_replay"

    # RTM REQ-5.39: component ablation WITHIN "gem_twp" specifically -- isolates
    # how much of that strategy's result comes from GEM's gradient projection
    # versus TWP's regularization penalty, orthogonal to the three backbone
    # switches above (REQ-5.32) and to continual_strategy itself. Only read
    # inside train.py's "gem_twp"-specific branches (never "agem_twp", which
    # also uses `self.twp` but is a different strategy -- these flags do not
    # touch it): disable_gem=True skips `GEMMemory.project_gradient` and
    # `GEMMemory.add_task_examples` (GEM's memory stays permanently empty, so
    # it contributes nothing), disable_twp=True skips `TWP.consolidate` (no
    # task snapshot is ever stored, so `TWP.penalty()` -- which already
    # returns exactly 0.0 with no stored snapshots, see continual/twp.py --
    # stays 0.0 for the rest of the run with no other code changes needed).
    # Both False (default) reproduces the unmodified "gem_twp" strategy;
    # both True is a lower-bound arm with neither mechanism active, i.e. plain
    # per-task fine-tuning. Both only meaningful when continual_strategy ==
    # "gem_twp"; harmless no-ops otherwise since nothing reads them.
    disable_gem: bool = False
    disable_twp: bool = False

    # RTM REQ-5.37: which entity gets classified -- "node" (Elliptic: each node
    # already *is* the labeled transaction) or "edge" (IBM AML: transactions
    # are edges, accounts are nodes, and the dataset's own native label lives
    # on the edge; see data/ibm_aml_dataset.py's module docstring). Derived
    # automatically from `--dataset` in train.py/task_matrix.py/infer.py, not
    # meant to be set independently of which dataset a run actually uses.
    task_type: str = "node"

    # RTM REQ-5.9, Elliptic-only for now: task granularity (raw time steps grouped
    # per continual-learning task; 1 = the original 49-task setup). Lives at the top
    # level, not nested under a dataset-specific config, so the whole ExperimentConfig
    # stays a single self-contained object that gets persisted into every checkpoint
    # (see checkpoint.py) -- infer.py restores it automatically instead of requiring
    # the caller to re-supply matching dataset-construction flags by hand.
    elliptic_time_steps_per_task: int = 1

    # RTM REQ-5.36: IBM AML task granularity -- number of continual-learning
    # tasks the raw file's calendar days are grouped into (see
    # aml_tgnn/data/ibm_aml_dataset.py's module docstring for why this is a
    # volume-balanced day-grouping, not a fixed-width time window). Default 7
    # matches this project's own Elliptic T=7 standard for cross-dataset
    # comparability, not anything intrinsic to IBM AML's own time span.
    ibm_aml_n_tasks: int = 7
