# Temporal Attractor – Cross-Modal Rhythm Structure

This repo is the temporal companion to [language-patterns](https://github.com/Chris-Roy-Writing-Has-Structure/language-patterns) (ledger-grammar project).

- Treat each rhythmic cycle (bar / phrase / gait / work-stroke) as a "line".
- Extract a 12D temporal feature vector per cycle (mirroring the ledger 12D vector).
- Reuse the ledger analysis stack (L₁ distances, UMAP, clustering, nulls) to detect a "temporal attractor".

**Status**
- Feature set: `TFS-1.0` (design fixed, math not yet implemented).
- Corpus: `v0.1` (starter synthetic corpus in `data/`).
- Code: pipeline stubs exist; `TemporalFeatureExtractor.extract()` is intentionally TODO.

**Quickstart**
```bash
pip install -r requirements.txt

python -m temporal_patterns.temporal_pipeline \
  --input data/temporal_starter_corpus.csv \
  --output data/temporal_features_tfs1.0.csv \
  --config config/tfs-1.0.yaml
