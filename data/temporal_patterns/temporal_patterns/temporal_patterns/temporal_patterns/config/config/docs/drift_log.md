# Drift Log – Temporal Attractor Project

This log records **conceptual changes** to the temporal feature set,
clustering pipeline, and analysis assumptions.

A drift entry is created only when:

- a feature's mathematical definition changes,
- a clustering or alignment model changes its conceptual basis,
- a parameter becomes a learned quantity rather than fixed, or
- empirical data invalidates an assumption and forces revision.

Each entry includes:

- Date
- Feature-set version (TFS-x.y)
- Nature of the change
- Motivation (what broke / what drifted)
- Impact (empirical + theoretical)
- Required actions (re-run corpus, re-run attractor tests, etc.)

---

## [2025-12-10] — **TFS-1.0** (initial definition)

### Change

Initial registration of the TFS-1.0 feature set:
12D temporal vector mirroring the ledger 12D vector, with placeholder
implementations in `TemporalFeatureExtractor.extract()`.

### Motivation

Create a stable conceptual target before seeing large-scale data and
before committing to any particular IOI coding or syncopation metric.

### Impact

- No empirical impact yet (features return zeros).
- Safe to build pipelines, configs, and analysis code using the names.

### Required actions

- Implement a minimal version of `TemporalFeatureExtractor.extract()`
  for the starter corpus.
- Run clustering once and archive results as the first empirical baseline.

---

## [2026-01-10] — **TFS-1.1** (example future change)

### Change

**Feature 3 – IOI concentration** changed from:

- mass in top 2 histogram bins → **Gini concentration index**
  over a fixed log-IOI codebook shared across traditions.

### Motivation

Asymmetric meters and dense interlocks showed strong sensitivity of
top-2-bin mass to bin choices; the feature was not robust to essential
timing variability.

### Impact

- Attractor cluster remains; within-cluster L₁ tightened slightly.
- Some edge traditions move closer to the main attractor.

### Required actions

- Re-run core corpora under TFS-1.1.
- Update any published rate-band estimates to reference TFS-1.1.
