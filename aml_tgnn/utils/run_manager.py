"""Per-dataset run directories + console/file logging (RTM REQ-5.1, REQ-5.2).

Every training invocation belongs to a "run": a timestamp-stamped id
(`run_YYYYMMDD_HHMMSS`) that gets its own subfolder under both
`Logs/<Dataset>/` and `Checkpoints/<Dataset>/` -- `<Dataset>` is `Elliptic` or
`IBM_AML`, so the two datasets' logs/checkpoints never mix. Resuming a run
reuses its original run_id (and therefore its original folders) instead of
minting a new one, so a run's full history stays in one place.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

DATASET_LABELS = {"elliptic": "Elliptic", "ibm_aml": "IBM_AML"}


class RunManager:
    def __init__(self, dataset_key: str, logs_root: str = "Logs", checkpoints_root: str = "Checkpoints"):
        if dataset_key not in DATASET_LABELS:
            raise ValueError(f"unknown dataset_key {dataset_key!r}; expected one of {list(DATASET_LABELS)}")
        self.dataset_label = DATASET_LABELS[dataset_key]
        self.logs_root = Path(logs_root)
        self.checkpoints_root = Path(checkpoints_root)

    @property
    def dataset_log_dir(self) -> Path:
        d = self.logs_root / self.dataset_label
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def dataset_ckpt_dir(self) -> Path:
        d = self.checkpoints_root / self.dataset_label
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _new_run_id() -> str:
        return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def find_latest_incomplete_run(self) -> Optional[str]:
        """Scan this dataset's checkpoint folder for runs whose
        `training_state.json` exists and isn't marked fully completed. Run ids
        are timestamp strings, so lexicographic max == most recent.
        """
        candidates = []
        for d in self.dataset_ckpt_dir.glob("run_*"):
            state_path = d / "training_state.json"
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not state.get("is_fully_completed", False):
                candidates.append(d.name)
        return max(candidates) if candidates else None

    def prepare_run(self, resume: bool, run_id: Optional[str] = None) -> Tuple[str, Path, Path]:
        """Returns (run_id, log_dir, ckpt_dir), creating both dirs. If
        `resume` is set and no explicit `run_id` was given, picks up the most
        recently interrupted run for this dataset; raises if none exists.
        """
        if run_id is None and resume:
            run_id = self.find_latest_incomplete_run()
            if run_id is None:
                raise FileNotFoundError(
                    f"--resume was set but no incomplete run was found under {self.dataset_ckpt_dir}"
                )
        if run_id is None:
            run_id = self._new_run_id()

        log_dir = self.dataset_log_dir / run_id
        ckpt_dir = self.dataset_ckpt_dir / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return run_id, log_dir, ckpt_dir


def setup_logging(log_dir: Path, run_id: str) -> logging.Logger:
    """One logger per run_id, writing to both `<log_dir>/train.log` and
    stdout with the same formatter, so anything printed via the logger is
    simultaneously visible on the console and persisted to disk. Safe to call
    more than once for the same run_id (e.g. repeated test invocations in one
    process) -- handlers are only attached the first time.
    """
    logger = logging.getLogger(f"aml_tgnn.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
