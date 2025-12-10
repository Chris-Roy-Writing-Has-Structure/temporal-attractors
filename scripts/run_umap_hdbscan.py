"""
Run UMAP + HDBSCAN on the temporal features table.

Usage (from repo root):
    python -m scripts.run_umap_hdbscan \
      --input data/temporal_features_tfs1.0.csv \
      --config config/tfs-1.0.yaml \
      --output_prefix results/tfs1.0

This will create:
    results/tfs1.0_clusters.csv   (features + cluster + UMAP coords)
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
import umap
import hdbscan

from temporal_patterns.temporal_features import TEMPORAL_FEATURE_NAMES


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def standardize_features(X: np.ndarray) -> np.ndarray:
    """
    Simple z-score (mean 0, std 1); avoids one feature dominating.
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def run_umap_and_hdbscan(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    # Extract feature matrix
    missing = [name for name in TEMPORAL_FEATURE_NAMES if name not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input CSV: {missing}")

    X = df[TEMPORAL_FEATURE_NAMES].to_numpy(dtype=float)
    X_std = standardize_features(X)

    # --- UMAP ---
    umap_cfg = cfg.get("umap", {})
    n_neighbors = int(umap_cfg.get("n_neighbors", 30))
    min_dist = float(umap_cfg.get("min_dist", 0.1))
    n_components = int(umap_cfg.get("n_components", 2))

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    )

    embedding = umap_model.fit_transform(X_std)
    # embedding shape: (n_samples, n_components)

    # --- HDBSCAN ---
    clust_cfg = cfg.get("clustering", {})
    algorithm = clust_cfg.get("algorithm", "hdbscan")

    if algorithm.lower() != "hdbscan":
        raise ValueError(
            f"config.clustering.algorithm must be 'hdbscan', got {algorithm!r}"
        )

    min_cluster_size = int(clust_cfg.get("hdbscan_min_cluster_size", 30))
    min_samples = int(clust_cfg.get("hdbscan_min_samples", 10))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    cluster_labels = clusterer.fit_predict(embedding)
    # HDBSCAN probabilities_ is "membership strength" in assigned cluster
    cluster_probs = getattr(clusterer, "probabilities_", None)
    if cluster_probs is None:
        cluster_probs = np.ones_like(cluster_labels, dtype=float)

    # Attach results to DataFrame
    df_out = df.copy()
    df_out["cluster_label"] = cluster_labels
    df_out["cluster_probability"] = cluster_probs

    # UMAP coordinates
    if n_components >= 1:
        df_out["umap_x"] = embedding[:, 0]
    if n_components >= 2:
        df_out["umap_y"] = embedding[:, 1]
    # If more components, we could add umap_z, etc.; not needed for now.

    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UMAP + HDBSCAN on temporal features."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV of temporal features (output of temporal_pipeline.py)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file (e.g. config/tfs-1.0.yaml)",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix for output files, e.g. results/tfs1.0",
    )

    args = parser.parse_args()

    # Load data and config
    df = pd.read_csv(args.input)
    cfg = load_config(args.config)

    # Run UMAP + HDBSCAN
    df_out = run_umap_and_hdbscan(df, cfg)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    clusters_path = f"{args.output_prefix}_clusters.csv"
    df_out.to_csv(clusters_path, index=False)

    # Basic summary
    n_segments = len(df_out)
    labels = df_out["cluster_label"].to_numpy()
    unique_labels = sorted(set(labels))
    n_clusters = len([lab for lab in unique_labels if lab != -1])

    print(f"Processed {n_segments} segments.")
    print(f"Found {n_clusters} clusters (HDBSCAN; -1 = noise).")
    print(f"Results saved to: {clusters_path}")


if __name__ == "__main__":
    main()
