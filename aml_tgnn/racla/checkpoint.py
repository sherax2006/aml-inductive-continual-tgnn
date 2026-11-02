"""Checkpoint save/load for `RACLAModel` (RTM REQ-5.25).

Reuses `utils/checkpoint.py`'s `load_checkpoint`/`read_progress` directly --
both are already model-agnostic (`load_checkpoint` is a plain `torch.load`,
`read_progress` a plain JSON read, neither references `ContinualTGNN`
specifically). Only *saving* needs a RACLA-specific function, since
`utils/checkpoint.py::save_checkpoint` reads `model.ewc`/`model.replay_buffer`
/`model.gem`/`model.twp`, none of which exist on `RACLAModel`.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ..config import ExperimentConfig
from .drift import DriftDetector
from .model import RACLAModel
from .replay import RACLAReplayBuffer


def save_racla_checkpoint(
    ckpt_dir: Path,
    model: RACLAModel,
    edge_predictor: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    replay_buffer: RACLAReplayBuffer,
    drift_detector: DriftDetector,
    progress: Dict[str, Any],
    cfg: ExperimentConfig,
    milestone_name: Optional[str] = None,
) -> None:
    payload = {
        "model_type": "racla",
        "model_state": model.state_dict(),
        "edge_predictor_state": edge_predictor.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "replay_state": replay_buffer.state_dict(),
        "drift_state": drift_detector.state_dict(),
        "progress": progress,
        "cfg": cfg,
    }
    torch.save(payload, ckpt_dir / "latest.pt")
    (ckpt_dir / "training_state.json").write_text(json.dumps(progress, indent=2, default=str))

    if milestone_name is not None:
        torch.save(payload, ckpt_dir / milestone_name)


def load_racla_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> Tuple[RACLAModel, ExperimentConfig]:
    """Same shape of contract as `aml_tgnn.infer.load_model_from_checkpoint`
    (a `(model, cfg)` pair, model in `.eval()` mode) so it can be passed as a
    `model_loader` into `task_matrix.py`'s matrix-building functions
    unchanged -- see `task_matrix.py`'s `model_loader` parameter, RTM REQ-5.25.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    model = RACLAModel(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, cfg
