from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .temporal_types import TemporalSegment

# Order matters; this is the canonical TFS-1.1 feature vector.
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


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits for a probability vector."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _normalize_entropy(H: float, n_bins: int) -> float:
    if n_bins <= 1:
        return 0.0
    H_max = np.log2(n_bins)
    return _safe_div(H, H_max, default=0.0)


def _nearest_grid_jitter(onsets: np.ndarray, cycle_duration: float, n_grid: int = 8) -> float:
    """
    RMS distance of onsets to nearest point on an n_grid subdivision of the cycle.
    """
    if onsets.size == 0 or cycle_duration <= 0:
        return 0.0

    grid = np.linspace(0.0, cycle_duration, n_grid, endpoint=False)
    # For each onset, find distance to nearest grid point
    dists = []
    for t in onsets:
        d = np.min(np.abs(grid - t))
        dists.append(d)
    dists = np.array(dists, dtype=float)
    return float(np.sqrt(np.mean(dists**2)))


def _infer_strong_positions(cycle_duration: float, n_positions: int = 4) -> np.ndarray:
    """
    Very simple 'metrical grid': divide the cycle into equal strong positions.
    """
    if cycle_duration <= 0:
        cycle_duration = 1.0
    return np.linspace(0.0, cycle_duration, n_positions, endpoint=False)


def _strong_beat_stats(
    onsets: np.ndarray,
    accents: Optional[np.ndarray],
    cycle_duration: float,
    strong_tol: float,
) -> Dict[str, float]:
    """
    Compute:
      - strong_position_coverage
      - accent_on_strong_positions
      - syncopation_index
    """
    if onsets.size == 0:
        return {
            "strong_position_coverage": 0.0,
            "accent_on_strong_positions": 0.0,
            "syncopation_index": 0.0,
        }

    strong_positions = _infer_strong_positions(cycle_duration, n_positions=4)
    n_events = onsets.size

    # Coverage: how many events are "close" to some strong position
    near_strong_mask = np.zeros_like(onsets, dtype=bool)
    for i, t in enumerate(onsets):
        if np.min(np.abs(strong_positions - t)) <= strong_tol * max(cycle_duration, 1e-6):
            near_strong_mask[i] = True

    strong_coverage = float(np.mean(near_strong_mask))

    # If no accents, approximate them as uniform 1.0
    if accents is None or accents.size != onsets.size:
        accents = np.ones_like(onsets, dtype=float)
    else:
        accents = accents.astype(float)

    total_accent = float(np.sum(accents)) if np.sum(accents) > 0 else 1.0
    accent_on_strong = float(np.sum(accents[near_strong_mask]) / total_accent)
    accent_off_strong = float(np.sum(accents[~near_strong_mask]) / total_accent)

    syncopation_index = accent_off_strong  # "accent where it doesn't belong"

    return {
        "strong_position_coverage": strong_coverage,
        "accent_on_strong_positions": accent_on_strong,
        "syncopation_index": syncopation_index,
    }


def _hierarchy_metrics(iois: np.ndarray, max_levels: int = 4) -> Dict[str, float]:
    """
    Very simple hierarchical structure:
      - cluster IOIs by their log2 relative to median IOI
      - count distinct "levels"
      - check how stable ratios are around simple factors (1, 2, 1/2, 3/2)
    """
    if iois.size == 0:
        return {
            "hierarchical_level_count_norm": 0.0,
            "hierarchical_ratio_stability": 0.0,
        }

    median_ioi = float(np.median(iois))
    if median_ioi <= 0:
        median_ioi = float(np.mean(iois[iois > 0])) if np.any(iois > 0) else 1.0

    # map IOIs to log2 scale relative to median
    log_rel = np.log2(iois / median_ioi)

    # quantize log_rel to nearest integer in [-max_levels, max_levels]
    levels = np.clip(np.round(log_rel), -max_levels, max_levels).astype(int)
    unique_levels = np.unique(levels)
    n_levels = unique_levels.size

    level_count_norm = _safe_div(float(n_levels), float(2 * max_levels + 1), default=0.0)

    # ratio stability: how close are IOI ratios to simple {0.5, 1, 1.5, 2}?
    simple_ratios = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)
    ratios = []
    for i in range(len(iois) - 1):
        if iois[i] > 0 and iois[i + 1] > 0:
            ratios.append(iois[i + 1] / iois[i])
    if len(ratios) == 0:
        ratio_stability = 0.0
    else:
        ratios = np.array(ratios, dtype=float)
        # For each ratio, distance to nearest simple ratio
        dists = []
        for r in ratios:
            d = np.min(np.abs(simple_ratios - r))
            dists.append(d)
        dists = np.array(dists, dtype=float)
        # Convert to stability score in [0,1]: smaller distance -> higher stability
        # Assume distances up to 0.5 are "ok"; clip larger
        norm_dists = np.clip(dists / 0.5, 0.0, 1.0)
        ratio_stability = float(1.0 - np.mean(norm_dists))

    return {
        "hierarchical_level_count_norm": level_count_norm,
        "hierarchical_ratio_stability": ratio_stability,
    }


class TemporalFeatureExtractor:
    """
    TFS-1.1 temporal feature extractor.

    The 12D vector mirrors your ledger 12D vector, but for rhythm.
    This implementation is deliberately simple but real: it computes
    IOIs, entropies, coverage, hierarchy, and grid jitter using only
    the starter onsets+accents we have now.
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
        self.strong_positions = strong_positions  # reserved; we infer grid inside
        self.strong_tol = strong_tol
        self.max_levels = max_levels
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.jitter_max = jitter_max

    def extract(self, seg: TemporalSegment) -> np.ndarray:
        """
        Main entry point: returns a (12,) float32 vector for a single segment.
        """
        onsets = np.asarray(seg.onset_times, dtype=float)
        onsets = np.sort(onsets)  # just in case

        cycle_duration = float(seg.cycle_duration) if seg.cycle_duration is not None else 1.0
        if cycle_duration <= 0:
            cycle_duration = 1.0

        n_events = onsets.size
        iois = np.diff(onsets) if n_events >= 2 else np.array([], dtype=float)

        # 1. events_per_cycle
        events_per_cycle = _safe_div(float(n_events), cycle_duration, default=0.0)

        # 2. ioi_regularization (1 - CV)
        if iois.size >= 2 and np.mean(iois) > 0:
            cv = float(np.std(iois) / np.mean(iois))
            ioi_regularization = float(1.0 - np.clip(cv, 0.0, 1.0))
        else:
            ioi_regularization = 0.0

        # 3–6: IOI histogram, entropies, and rate
        if iois.size == 0:
            ioi_quant_conc = 0.0
            H0_norm = 0.0
            H2_norm = 0.0
            rate_H2_norm = 0.0
        else:
            # Build simple IOI bins between min and max
            ioi_min = float(np.min(iois))
            ioi_max = float(np.max(iois))
            if ioi_max == ioi_min:
                ioi_max = ioi_min + 1e-6

            bins = np.linspace(ioi_min, ioi_max, self.ioi_bins + 1)
            counts, _ = np.histogram(iois, bins=bins)
            total = counts.sum()
            if total == 0:
                p = np.ones(self.ioi_bins, dtype=float) / self.ioi_bins
            else:
                p = counts.astype(float) / float(total)

            # 3. concentration: mass in top-2 probability bins
            sorted_p = np.sort(p)[::-1]
            k = min(2, sorted_p.size)
            ioi_quant_conc = float(np.sum(sorted_p[:k]))

            # 4. H0 normalized
            H0 = _entropy(p)
            H0_norm = _normalize_entropy(H0, self.ioi_bins)

            # 5. Bigram entropy H2 over IOI categories
            cat = np.digitize(iois, bins=bins[:-1], right=True)
            cat = np.clip(cat, 0, self.ioi_bins - 1)
            if cat.size >= 2:
                pair_counts = np.zeros((self.ioi_bins, self.ioi_bins), dtype=float)
                for i in range(len(cat) - 1):
                    pair_counts[cat[i], cat[i + 1]] += 1.0
                total_pairs = pair_counts.sum()
                if total_pairs == 0:
                    P2 = np.ones_like(pair_counts) / pair_counts.size
                else:
                    P2 = pair_counts / total_pairs
                H2 = _entropy(P2.ravel())
                H2_norm = _normalize_entropy(H2, self.ioi_bins * self.ioi_bins)
            else:
                H2_norm = 0.0

            # 6. temporal_rate_H2_per_cycle_norm
            # First, H2 per second (or unit time), then clamp into [rate_min, rate_max]
            rate_raw = _safe_div(H2_norm, cycle_duration, default=0.0)
            rate_clamped = np.clip(rate_raw, self.rate_min, self.rate_max)
            # Normalize back to [0,1] by dividing by max rate window if >0
            rate_H2_norm = _safe_div(rate_clamped, self.rate_max if self.rate_max > 0 else 1.0)

        # 7–9: strong beat / accent / syncopation stats
        strong_stats = _strong_beat_stats(
            onsets=onsets,
            accents=seg.accents,
            cycle_duration=cycle_duration,
            strong_tol=self.strong_tol,
        )

        # 10–11: hierarchical metrics
        hierarchy = _hierarchy_metrics(iois=iois, max_levels=self.max_levels)

        # 12: grid jitter (1 - normalized RMS jitter)
        jitter = _nearest_grid_jitter(onsets, cycle_duration, n_grid=8)
        # Treat jitter_max as the "bad" RMS; anything >= jitter_max maps to 0
        jitter_norm = 1.0 - np.clip(jitter / max(self.jitter_max, 1e-6), 0.0, 1.0)

        features = np.array(
            [
                events_per_cycle,                          # 1
                ioi_regularization,                        # 2
                ioi_quant_conc,                            # 3
                H0_norm,                                   # 4
                H2_norm,                                   # 5
                rate_H2_norm,                              # 6
                strong_stats["strong_position_coverage"],  # 7
                strong_stats["accent_on_strong_positions"],  # 8
                strong_stats["syncopation_index"],         # 9
                hierarchy["hierarchical_level_count_norm"],   # 10
                hierarchy["hierarchical_ratio_stability"],    # 11
                jitter_norm,                               # 12
            ],
            dtype=np.float32,
        )

        return features

    def extract_with_meta(self, seg: TemporalSegment) -> Dict[str, float]:
        """
        Convenience wrapper: maps feature names to values.
        """
        vec = self.extract(seg)
        return dict(zip(TEMPORAL_FEATURE_NAMES, vec))

