# Changelog

All notable changes to this repository will be documented in this file.

This repository contains the code, experimental outputs, reproducibility documentation, and archival metadata associated with:

**Reliability-Aware Multi-View Forensic Evidence Synthesis for Network Intrusion Investigation**

The repository follows a versioned release model so that changes to the implementation, experimental outputs, documentation, or archival metadata remain traceable.

The format of this changelog is based on the principles of [Keep a Changelog](https://keepachangelog.com/), and repository releases are intended to follow semantic versioning where practical.

---

## [Unreleased]

### Added

- Initial repository structure for publication and reproducibility.
- `README.md` for the main branch.
- `CITATION.cff` for repository citation metadata.
- `requirements.txt` for Python dependencies.
- `environment.yml` for Conda-based environment creation.
- `Dataset_sources.md` documenting the official dataset sources and preparation requirements.
- `REPRODUCIBILITY.md` describing the full eight-stage experimental workflow.
- `RESULTS_INDEX.md` providing navigation between experiment-specific results branches.
- Planned `RELEASE_MANIFEST.md` for recording exact branch commits associated with the archival release.
- Planned `.zenodo.json` metadata for DOI archiving.
- Planned DOI integration after creation of the archival release.

### Repository Organization

The repository is organized using separate Git branches for implementation and experimental evidence:

- `main`
- `code`
- `results`
- `results/01-dataset-preparation`
- `results/02-baseline-models`
- `results/03-framework-validation`
- `results/04-ablation-and-degradation`
- `results/05-cross-dataset-validation`
- `results/06-computational-efficiency`
- `results/07-statistical-analysis`
- `results/08-final-results-consolidation`

### Notes

- Original third-party datasets are not redistributed through the repository.
- Dataset source information is maintained centrally in `Dataset_sources.md`.
- Experiment 8 is defined as a final results-consolidation stage and does not introduce new model training or new experimental observations.
- Publication figures and tables should be generated from archived numerical outputs rather than manually reconstructed values.
- The repository DOI will be added only after the archival release is created.

---

## [1.0.0] - To be released

### Manuscript Reproducibility Release

This release will correspond to the computational materials associated with the manuscript:

**Reliability-Aware Multi-View Forensic Evidence Synthesis for Network Intrusion Investigation**

### Added

#### Main Branch

- Publication-oriented repository landing page.
- Citation metadata.
- Software environment specifications.
- Dataset-source documentation.
- Reproducibility instructions.
- Results index.
- Release manifest.
- Archival metadata.
- Version history.

#### Code Branch

- Dataset preparation implementation.
- Semantic multi-view evidence representation.
- Reliability estimation modules.
- Global evidence-view weighting.
- Unweighted multi-view fusion.
- Forensic Evidence Reliability Fusion (FERF).
- Forensic reasoning procedures.
- Investigation-confidence estimation.
- Evaluation utilities.
- Experiment-specific scripts.
- Final results-consolidation scripts.
- Figure-generation scripts where applicable.

#### Results Branches

Experiment-specific outputs will be preserved in dedicated result branches.

##### Experiment 1 — Dataset Preparation and Integrity

Branch:

```text
results/01-dataset-preparation
