# Dataset Sources and Preparation

## Overview

The experimental evaluation of **Reliability-Aware Multi-View Forensic Evidence Synthesis for Network Intrusion Investigation** uses two publicly available network-intrusion datasets:

1. **CICIDS2017**
2. **CSE-CIC-IDS2018**

The datasets are **not redistributed through this repository**.

Users wishing to reproduce the experiments must obtain the datasets directly from their official providers and comply with the corresponding terms of use and distribution conditions.

This document identifies the dataset sources and describes the general preparation requirements used by the experimental workflow.

---

# 1. CICIDS2017

## 1.1 Dataset Description

The **CICIDS2017** dataset was developed by the Canadian Institute for Cybersecurity (CIC), University of New Brunswick.

It contains benign network activity and multiple categories of malicious traffic generated in a controlled network environment. The dataset was designed to provide realistic background traffic together with contemporary intrusion scenarios.

The experimental workflow in this repository uses the network-flow representation of CICIDS2017.

### Reference

I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani,  
“Toward generating a new intrusion detection dataset and intrusion traffic characterization,”  
in *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116.  
DOI: `10.5220/0006639801080116`

## 1.2 Official Source

**Provider:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick

**Dataset:** CICIDS2017

Official dataset page:

https://www.unb.ca/cic/datasets/ids-2017.html

Users should obtain the dataset from the official provider rather than from copies redistributed by third parties.

## 1.3 Data Required for Reproduction

The experiments use the flow-based CSV representation of the network traffic.

After downloading CICIDS2017, the required flow files should be made available to the dataset-preparation scripts before executing the subsequent experiments.

The original source files should remain unchanged. Any cleaning, harmonization, semantic-view construction, or feature transformation required by the framework is performed as part of the experimental preparation workflow.

## 1.4 Integrity Handling

The framework maintains acquisition-level integrity information for the source files used in the analysis.

SHA-256 information is associated with the acquired source-file context and propagated to records derived from those files during the analytical workflow.

This mechanism should **not** be interpreted as independent cryptographic authentication of every individual network-flow record. It preserves the integrity context of the acquired source file from which the analytical records originate.

---

# 2. CSE-CIC-IDS2018

## 2.1 Dataset Description

The **CSE-CIC-IDS2018** dataset was produced through collaboration involving the Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC).

It contains benign and malicious network traffic covering multiple attack scenarios and provides a large-scale environment for intrusion-detection and network-security research.

The experimental workflow in this repository uses the flow-based representation of CSE-CIC-IDS2018.

## 2.2 Official Source

**Provider:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick

**Dataset:** CSE-CIC-IDS2018

Official dataset page:

https://www.unb.ca/cic/datasets/ids-2018.html

Users should download the dataset from the official provider and follow any access or usage requirements specified by the provider.

## 2.3 Data Required for Reproduction

The experimental workflow requires the flow-based CSV data used by the dataset-preparation stage.

Because the original dataset is large, it is intentionally excluded from this GitHub repository and from the archived software release.

Users should preserve the downloaded source files in their original form and perform preprocessing using the scripts supplied in the `code` branch.

## 2.4 Integrity Handling

As with CICIDS2017, source-file integrity information is maintained at the acquisition level.

SHA-256 information associated with the source files is propagated through the analytical workflow so that derived records retain their source-file integrity context.

This does not constitute independent per-flow cryptographic verification.

---

# 3. Dataset Storage

The original datasets should **not** be committed to this repository.

A local directory outside Git version control should be used for downloaded datasets.

A suggested local organization is:

```text
datasets/
├── CICIDS2017/
│   └── [original downloaded flow files]
│
└── CSE-CIC-IDS2018/
    └── [original downloaded flow files]
