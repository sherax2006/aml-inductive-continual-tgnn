"""Checkpoint persistence for resumable training (RTM REQ-5.3, REQ-5.4).

Two files per save, both in the run's `Checkpoints/<Dataset>/<run_id>/` folder:
  - `latest.pt`             : full binary state (model/optimizer tensors, plus
                              whichever continual-learning strategy's own state --
                              EWC+replay, or GEM+TWP, RTM REQ-5.18 -- is active)
                              needed to actually resume training.
  - `training_state.json`   : the small human-readable progress summary --
                              which tasks are done, which epoch is next -- also
                              what `RunManager.find_latest_incomplete_run`
                              reads to detect a resumable run without touching
                              the (potentially large) .pt file.

Resume is epoch-granular by design: a checkpoint is only ever written at the
end of a fully-completed epoch (see `train.py`), never mid-epoch. If training
is interrupted mid-epoch, that epoch is simply re-run in full from its start
on resume -- no partial-batch state is ever saved or restored.

A permanent milestone copy is also written as `task_<task_id>_complete.pt`
the moment a task fully finishes (post EWC-consolidation, post replay-buffer
update) -- `latest.pt` keeps getting overwritten every epoch, so these are the
only checkpoints guaranteed to survive past their own task.

The full `ExperimentConfig` used for training (backbone dims,
elliptic_time_steps_per_task, ...) is saved into every checkpoint too (RTM
REQ-5.9). Without this, `infer.py` would need the caller to correctly re-supply
every architecture flag a checkpoint was trained with by hand -- get a
dimension wrong and `load_state_dict` fails with a shape mismatch rather than
just working automatically.

There is no more `provisioned_tasks` entry (RTM REQ-5.15) -- parameter
isolation was removed, so there is exactly one classification head, always,
already part of a freshly-constructed `ContinualTGNN(cfg)`; nothing needs to
be provisioned before `model.load_state_dict(...)` runs.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..config import ExperimentConfig
from ..models.continual_wrapper import ContinualTGNN


def save_checkpoint(
    ckpt_dir: Path,
    model: ContinualTGNN,
    optimizer: Optional[torch.optim.Optimizer],
    progress: Dict[str, Any],
    cfg: ExperimentConfig,
    milestone_name: Optional[str] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "progress": progress,
        "cfg": cfg,
    }
    # RTM REQ-5.18/5.26/5.27: persist only the active strategy's state -- the
    # other three pairs are None on `model` and have nothing to save.
    if cfg.continual_strategy == "ewc_replay":
        payload["ewc_state"] = model.ewc.state_dict()
        payload["replay_state"] = model.replay_buffer.state_dict()
    elif cfg.continual_strategy == "gem_twp":
        payload["twp_state"] = model.twp.state_dict()
        payload["gem_state"] = model.gem.state_dict()
    elif cfg.continual_strategy == "agem_twp":
        payload["twp_state"] = model.twp.state_dict()
        payload["agem_state"] = model.agem.state_dict()
    else:  # mas_twp
        payload["twp_state"] = model.twp.state_dict()
        payload["mas_state"] = model.mas.state_dict()
    torch.save(payload, ckpt_dir / "latest.pt")
    (ckpt_dir / "training_state.json").write_text(json.dumps(progress, indent=2, default=str))

    if milestone_name is not None:
        torch.save(payload, ckpt_dir / milestone_name)


def load_checkpoint(ckpt_dir: Path, filename: str = "latest.pt", map_location: str = "cpu") -> Dict[str, Any]:
    # weights_only=False: this checkpoint is one we produced ourselves (not an
    # untrusted download) and its payload includes plain-Python objects (the
    # ExperimentConfig dataclass, the progress dict) that torch's newer default
    # weights_only=True unpickling would refuse to load.
    return torch.load(ckpt_dir / filename, map_location=map_location, weights_only=False)


def read_progress(ckpt_dir: Path) -> Optional[Dict[str, Any]]:
    path = ckpt_dir / "training_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
