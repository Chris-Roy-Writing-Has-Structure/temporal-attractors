"""
Temporal Attractor: feature extraction and pipelines for rhythmic cycles.
"""

from .temporal_types import TemporalSegment
from .temporal_features import (
    TemporalFeatureExtractor,
    TEMPORAL_FEATURE_NAMES,
)
from .temporal_pipeline import main as run_pipeline

__version__ = "0.1.0"
__all__ = [
    "TemporalSegment",
    "TemporalFeatureExtractor",
    "TEMPORAL_FEATURE_NAMES",
    "run_pipeline",
]
