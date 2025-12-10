import argparse
import json
from typing import List

import numpy as np
import pandas as pd
import yaml

from .temporal_types import TemporalSegment
from .temporal_features import (
    TemporalFeatureExtractor,
    TEMPORAL_FEATURE_NAMES,
)


def load_segments_from_csv(path: str) -> List[TemporalSegment]:
    df = pd.read_csv(path)
    segments: List[TemporalSegment] = []

    for _, row in df.iterrows():
        onsets = np.array(json.loads(row["onsets"]), dtype=float)

        accents = None
        accents_val = row.get("accents", None)
        if isinstance(accents_val, str) and accents_val.strip().startswith("["):
            accents = np.array(json.loads(accents_val), dtype=float)

        meta = {
            "segment_id": row["segment_id"],
            "tradition_id": row["tradition_id"],
            "modality": row["modality"],
            "functional_context": row.get("functional_context", None),
            "source_type": row.get("source_type", None),
            "tempo_bpm": row.get("tempo_bpm", None),
            "language": row.get("language", None),
            "participant_id": row.get("participant_id", None),
        }

        seg = TemporalSegment(
            onset_times=onsets,
            accents=accents,
            cycle_duration=float(row.get("cycle_duration", 1.0)),
            metadata=meta,
        )
        segments.append(seg)

    return segments


def build_extractor_from_config(config_path: str) -> TemporalFeatureExtractor:
    """
    Minimal config hook: we read feature-extractor parameters from YAML.

    See config/config_schema.yaml and config/tfs-1.0.yaml for structure.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    fe_cfg = cfg.get("feature_extractor", {})
    return TemporalFeatureExtractor(
        ioi_bins=fe_cfg.get("ioi_bins", 5),
        strong_positions=None,  # TODO: implement from config if needed
        strong_tol=fe_cfg.get("strong_tol", 0.05),
        max_levels=fe_cfg.get("max_levels", 4),
        rate_min=fe_cfg.get("rate_min", 0.0),
        rate_max=fe_cfg.get("rate_max", 10.0),
        jitter_max=fe_cfg.get("jitter_max", 0.1),
    )


def compute_feature_table(
    segments: List[TemporalSegment],
    extractor: TemporalFeatureExtractor,
) -> pd.DataFrame:
    rows = []

    for seg in segments:
        vec = extractor.extract(seg)
        feat_dict = dict(zip(TEMPORAL_FEATURE_NAMES, vec))

        row = {
            **seg.metadata,
            "cycle_duration": seg.cycle_duration,
            **feat_dict,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute temporal features for rhythmic segments."
    )
    parser.add_argument("--input", required=True, help="Input CSV of segments")
    parser.add_argument("--output", required=True, help="Output CSV of features")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file (e.g. config/tfs-1.0.yaml)",
    )

    args = parser.parse_args()

    segments = load_segments_from_csv(args.input)
    extractor = build_extractor_from_config(args.config)
    df_out = compute_feature_table(segments, extractor)
    df_out.to_csv(args.output, index=False)
    print(f"Extracted features for {len(df_out)} segments to {args.output}")


if __name__ == "__main__":
    main()
