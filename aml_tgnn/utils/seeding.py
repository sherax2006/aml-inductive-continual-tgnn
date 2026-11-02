"""Global RNG seeding for reproducible training runs (RTM REQ-5.29).

Deliberately separate from `TrainConfig.split_seed`, which controls only the
per-task stratified train/val partition and must stay fixed across a
multi-seed comparison -- varying both at once would confound "does this
strategy's outcome vary with model init/training stochasticity" with "does
it vary because each seed also saw a different train/val split."
"""
import random

import torch


def set_seed(seed: int) -> None:
    """Seeds every RNG source this codebase actually draws from: Python's
    own `random` (GEM/A-GEM/GraphReplayBuffer's reservoir sampling and
    replay-batch sampling all use it directly, not torch's RNG) and torch's
    CPU/CUDA generators (model initialization, dropout, `NeighborLoader`'s
    neighbor sampling, batch shuffling).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on a CPU-only machine, correct if ever run on GPU
