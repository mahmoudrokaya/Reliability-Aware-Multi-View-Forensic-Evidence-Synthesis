# Reliability-Aware Multi-View Forensic Evidence Synthesis

[![Code](https://img.shields.io/badge/Branch-Code-blue)](../../tree/code)
[![Results](https://img.shields.io/badge/Branch-Results-green)](../../tree/results)
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey)](LICENSE)

## Overview

This repository provides the source code and experimental results associated with the study:

**Reliability-Aware Multi-View Forensic Evidence Synthesis for Network Intrusion Investigation**

The study presents a computational framework for transforming network-flow observations into structured, reliability-aware, and traceable forensic evidence representations.

Rather than treating network-flow attributes as a single undifferentiated feature space, the framework organizes them into six complementary semantic evidence views:

- Volume
- Temporal
- Transport
- Flags and Errors
- Directional
- General

The proposed **Forensic Evidence Reliability Fusion (FERF)** mechanism combines validation-selected global evidence-view importance with record-specific reliability information derived from:

- acquisition-level integrity,
- information quality, and
- temporal validity.

The framework further distinguishes the fused attack probability from investigation confidence and preserves evidence provenance, view-level analytical contributions, reliability information, and computational decisions within a structured forensic record.

The framework is intended as computational support for network forensic investigation. It does **not** determine legal admissibility, evidentiary weight, judicial certainty, or replace human forensic judgment.

---

## Repository Organization

To maintain a clear separation between implementation and experimental evidence, this repository uses dedicated Git branches.

### Main Branch

The `main` branch contains the documentation, citation metadata, reproducibility information, dataset-source information, and files required for archival release and DOI registration.

### Code Branch

The [`code`](../../tree/code) branch contains the complete implementation used for the experimental evaluation, including data preparation, semantic evidence representation, reliability estimation, FERF, forensic reasoning, confidence estimation, and evaluation procedures.

### Results Branch

The [`results`](../../tree/results) branch provides an index to the experimental outputs.

The complete experimental results are organized into dedicated branches:

| Experiment | Description | Branch |
|---|---|---|
| 1 | Dataset preparation and integrity verification | `results/01-dataset-preparation` |
| 2 | Conventional baseline models | `results/02-baseline-models` |
| 3 | Framework and multi-view validation | `results/03-framework-validation` |
| 4 | Component ablation and evidence degradation | `results/04-ablation-and-degradation` |
| 5 | Cross-dataset validation | `results/05-cross-dataset-validation` |
| 6 | Computational efficiency and scalability | `results/06-computational-efficiency` |
| 7 | Repeated-CV statistical analysis | `results/07-statistical-analysis` |
| 8 | Final results consolidation | `results/08-final-results-consolidation` |

Experiment 8 is a **results-consolidation stage**. It does not perform additional model training or introduce new experimental observations. It consolidates validated outputs from the preceding experiments for final analysis and reporting.

See [`RESULTS_INDEX.md`](RESULTS_INDEX.md) for a detailed mapping between experiments, result branches, and reported outputs.

---

## Experimental Evaluation

The repository supports the experimental workflow used to evaluate the proposed framework from complementary predictive, forensic, statistical, and computational perspectives.

The evaluation includes:

1. dataset preparation and integrity verification;
2. conventional machine-learning baseline evaluation;
3. semantic multi-view evidence analysis and FERF validation;
4. component-wise ablation analysis;
5. controlled evidence-degradation experiments;
6. zero-shot cross-dataset validation;
7. repeated stratified cross-validation and paired statistical testing;
8. computational-efficiency and scalability assessment; and
9. final consolidation of validated experimental outputs.

The conventional baseline models include:

- Decision Tree
- Random Forest
- LightGBM
- CatBoost
- XGBoost

The statistical evaluation uses repeated stratified cross-validation with **10 repetitions × 5 folds**, producing 50 paired evaluations per dataset for the principal comparative configurations.

---

## Datasets

The experiments use the following publicly available network-intrusion datasets:

### CICIDS2017

CICIDS2017 contains benign and malicious network traffic representing multiple contemporary attack scenarios and associated network-flow characteristics.

### CSE-CIC-IDS2018

CSE-CIC-IDS2018 provides a larger and heterogeneous collection of benign and attack traffic for intrusion-detection evaluation.

### Data Availability

**The datasets are not redistributed through this repository.**

Users should obtain the original datasets directly from their official providers. This avoids unnecessary redistribution of large external datasets and preserves the relationship between the experiments and the authoritative dataset sources.

Official source links, required files, preparation information, and dataset-specific notes are provided in:

**[`Dataset_sources.md`](Dataset_sources.md)**

Users wishing to reproduce the experiments should download the datasets from the listed official sources and follow the preparation instructions before executing the experimental code.

---

## Reproducibility

The repository is structured so that the relationship between the implementation and reported experimental evidence can be traced as:

**Dataset → Preparation → Experiment Code → Experimental Output → Consolidated Result**

The exact scripts used for the experiments are retained in the `code` branch, while generated outputs are preserved in their corresponding experiment-specific result branches.

Detailed reproduction instructions are provided in:

**[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)**

The software environment is documented through:

- `requirements.txt`
- `environment.yml`

Where applicable, fixed random seeds, validation-controlled parameter selection, common dataset partitions, and leakage-safe preprocessing are used to support reproducibility.

The held-out test partitions are not used to fit predictive parameters, reliability coefficients, evidence-view weights, decision thresholds, or confidence parameters.

---

## Reliability-Aware Evidence Fusion

The framework combines two complementary forms of evidence weighting.

**Global evidence importance** represents the discriminative contribution of each semantic evidence view and is selected using validation data.

**Record-specific reliability** represents the condition of a particular evidence view for an individual network record.

The record-specific reliability formulation considers three components:

1. **Integrity reliability** — acquisition-level source-file integrity context propagated to associated records.
2. **Information-quality reliability** — completeness, numerical plausibility, and categorical validity of the available evidence.
3. **Temporal reliability** — availability and validity of timestamp information.

These quantities are combined by FERF to produce a reliability-aware fused analytical probability.

The integrity component should not be interpreted as independent cryptographic authentication of every individual network-flow record. It represents verified acquisition-level source-file integrity information propagated through the analytical workflow.

---

## Investigation Confidence

Investigation confidence is maintained separately from the FERF attack probability.

The confidence formulation considers complementary characteristics of the analytical evidence, including:

- fused-output strength,
- agreement among semantic evidence views,
- mean evidence reliability, and
- dispersion among view-specific probabilities.

This separation prevents predictive probability from being treated automatically as forensic certainty.

Investigation confidence is a **computational analytical measure** and should not be interpreted as legal certainty, evidentiary admissibility, or human expert confidence.

---

## Scope and Limitations

The repository provides a reproducible implementation of the computational methodology evaluated in the associated study.

The current quantitative validation is concentrated on CICIDS2017 and CSE-CIC-IDS2018. Cross-dataset experiments are intentionally performed without target-domain adaptation to assess zero-shot transfer under domain shift.

The temporal-reliability component evaluates timestamp availability and validity; it does not represent event recency or complete forensic timeline consistency.

Computational-efficiency measurements characterize the implemented analytical inference path and should not be interpreted as the wall-clock cost of a complete acquisition-to-report forensic investigation.

The framework provides computational support for evidence synthesis and investigator review. It does not automate legal interpretation or determine whether evidence is admissible in a particular jurisdiction.

---

## Results

Experimental outputs are available through the `results` branch and its experiment-specific branches.

The final consolidated outputs include:

- baseline model results;
- clean-data ablation comparisons;
- evidence-degradation comparisons;
- cross-dataset results;
- computational-efficiency measurements;
- scalability results;
- repeated cross-validation scores;
- statistical performance summaries;
- paired statistical tests;
- descriptive statistics;
- dataset assessments; and
- a claims-to-evidence matrix.

See:

**[`RESULTS_INDEX.md`](RESULTS_INDEX.md)**

for the complete mapping between experimental stages and result branches.

---

## Citation

If you use the code, experimental results, or framework provided in this repository, please cite the associated paper and archived software release.

Citation metadata is provided in:

**[`CITATION.cff`](CITATION.cff)**

### Software Archive

A versioned archival release of this repository will be deposited in Zenodo.

**DOI:** To be added after the archival release is generated.

The DOI in this README and `CITATION.cff` will be updated when the publication release is archived.

---

## Release and Archival Information

The publication-associated repository version will be released as:

**v1.0.0 — Manuscript Reproducibility Release**

The release will preserve the code and experimental results associated with the manuscript.

Because code and experimental outputs are maintained across separate branches, the exact branch versions associated with the archival release are documented in:

**[`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md)**

The manifest records the corresponding commit identifiers for the code and experiment-specific result branches, allowing the archived study version to be reconstructed unambiguously.

---

## Repository Contents on `main`

The main branch contains the following publication and reproducibility resources:

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── Dataset_sources.md
├── REPRODUCIBILITY.md
├── RESULTS_INDEX.md
├── RELEASE_MANIFEST.md
├── CHANGELOG.md
├── requirements.txt
├── environment.yml
└── .gitignore
