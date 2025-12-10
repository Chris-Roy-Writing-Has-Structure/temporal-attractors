from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .temporal_types import TemporalSegment

# Order matters; this is the canonical TFS-1.0 feature vector.
TEMPORAL_FEATURE_NAMES: List[str] = [
    "events_per_cycle",                 # 1. normalized event density
    "ioi_regularization",               # 2. 1 - CV(IOI)
    "ioi_quantization_concentration",   # 3. mass in top IOI categories
    "ioi_entropy_H0_norm",              # 4. normalized unigram entropy
    "ioi_entropy_H2_norm",              # 5. normalized bigram entropy
    "temporal_rate_H2_per_cycle_norm",  # 6. H2 per unit time
    "strong_position_coverage",         # 7. events near strong beats
    "accent_on_strong_positions",       # 8. accent mass on strong beats
    "syncopation_index",                # 9. accent off strong beats
    "hierarchical_level_count_norm",    # 10. effective # of IOI scales
    "hierarchical_ratio_stability",     # 11. stability of level ratios
    "grid_quantization_jitter_norm",    # 12. 1 - RMS jitter vs grid
]


class TemporalFeatureExtractor:
    """
    TFS-1.0 temporal feature extractor.

    The 12D vector mirrors your ledger 12D vector, but for rhythm.
    Definitions are fixed conceptually; implementation can drift
    across TFS versions (1.0, 1.1, 2.0, ...) and is tracked in
    docs/drift_log.md and config/tfs-*.yaml.
    """

    def __init__(
        self,
        ioi_bins: int = 5,
        strong_positions: Optional[np.ndarray] = None,
        strong_tol: float = 0.05,
        max_levels: int = 4,
        rate_min: float = 0.0,
        rate_max: float = 10.0,
        jitter_max: float = 0.1,
    ) -> None:
        """
        Parameters mirror the design notes; see config_schema.yaml
        for how they are serialized and versioned.
        """
        self.ioi_bins = ioi_bins
        self.strong_positions = strong_positions  # if None, infer later
        self.strong_tol = strong_tol
        self.max_levels = max_levels
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.jitter_max = jitter_max

    def extract(self, seg: TemporalSegment) -> np.ndarray:
        """
        Main entry point: returns a (12,) float32 vector for a single segment.

        Implementation is intentionally left as TODO for later-you:
        wire in the actual formulas from the design text once you're
        rested and have real data to test against.
        """
        features = np.zeros(len(TEMPORAL_FEATURE_NAMES), dtype=np.float32)

        # TODO: implement the actual feature calculations:
        # - compute IOIs and IOI categories
        # - density, CV, concentration, entropies
        # - temporal information rate
        # - strong-beat coverage, accents, syncopation
        # - hierarchical levels and ratio stability
        # - grid quantization and jitter
        return features

    def extract_with_meta(self, seg: TemporalSegment) -> Dict[str, float]:
        """
        Convenience wrapper: maps feature names to values.
        """
        vec = self.extract(seg)
        return dict(zip(TEMPORAL_FEATURE_NAMES, vec))
