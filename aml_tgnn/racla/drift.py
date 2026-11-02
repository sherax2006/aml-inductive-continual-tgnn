"""KS-test + Bonferroni-corrected drift detector (RTM REQ-5.25), from
BasePaper1_RACLA.md Sec. 4.5 (Eq. 15-19).

Sliding-window design: the paper describes "a sliding window-based drift
detector that maintains a reference window of historical features and
compares against incoming data at each time step." In a per-example data
stream that's a literal sliding window; in this project's per-*task*
training loop (the same unit every other strategy in this codebase
consolidates on), the natural analogue is: reference window = the most
recently completed task's own node features, current window = the new
task's. This is this project's own reasonable reading of "sliding window"
translated to a task-stream setting, not a literal quote from the paper.
"""
from typing import Optional, Tuple

import torch
from scipy.stats import ks_2samp

from ..config import RACLAConfig


class DriftDetector:
    def __init__(self, cfg: RACLAConfig):
        self.cfg = cfg
        self._reference: Optional[torch.Tensor] = None  # [n_prev, d] previous task's node features

    def check_and_update(self, current_features: torch.Tensor) -> Tuple[bool, Optional[float]]:
        """Eq. 15-19: compares `current_features` (this task's own node
        feature matrix, `[n, d]`) against the stored reference window (the
        previous task's), then replaces the reference with `current_features`
        for the next call. Returns `(drift_detected, proportion_significant)`
        -- `proportion_significant` is `None` on the very first call (no
        reference exists yet, so no comparison is made -- no drift by
        definition, matching this project's own established pattern of not
        treating "the very first task" as a special-cased false condition
        elsewhere, e.g. REQ-5.15's zero-shot-on-task-1 change).
        """
        if self._reference is None:
            self._reference = current_features.detach().cpu()
            return False, None

        ref = self._reference.numpy()
        cur = current_features.detach().cpu().numpy()
        d = ref.shape[1]

        alpha_adjusted = self.cfg.drift_alpha / d  # Eq. 17

        significant = 0
        for dim in range(d):
            # Eq. 15: D = max|F_his(x) - F_cur(x)|, via scipy's two-sample KS test,
            # which returns both the D statistic and its p-value directly.
            _, p_value = ks_2samp(ref[:, dim], cur[:, dim])
            if p_value <= alpha_adjusted:  # Eq. 18
                significant += 1

        proportion = significant / d
        drift = proportion > self.cfg.drift_tau  # Eq. 19

        self._reference = current_features.detach().cpu()
        return drift, proportion

    def state_dict(self):
        return {"reference": self._reference}

    def load_state_dict(self, state) -> None:
        self._reference = state["reference"]
