from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TemporalSegment:
    """
    One temporal segment (cycle / phrase / gait / work-stroke).

    Attributes
    ----------
    onset_times :
        1D array of onset times, in beats or seconds (monotonic).
    accents :
        Optional 1D array of same length as onset_times, with binary or
        graded accent values. None if not available.
    cycle_duration :
        Duration of the cycle in the same units as onset_times.
    metadata :
        Arbitrary dictionary of identifiers and context (tradition_id,
        modality, functional_context, etc.).
    """
    onset_times: np.ndarray
    accents: Optional[np.ndarray]
    cycle_duration: float
    metadata: Dict[str, Any]
