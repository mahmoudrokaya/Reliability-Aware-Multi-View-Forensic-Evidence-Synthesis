# Experiment 8 — Final Results

## Purpose

Experiment 8 consolidates the validated outputs from the preceding experimental stages. No model was trained, refitted, reoptimized, or evaluated on new observations during this stage.

## Final Dataset-Level Assessment

| Dataset | Held-out BA | Held-out F1 | Held-out MCC | Held-out ROC-AUC | Repeated-CV BA Mean | Repeated-CV BA SD | Latency ms/record | Throughput records/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CICIDS2017 | 0.999102 | 0.996805 | 0.996160 | 0.999919 | 0.998706 | 0.000284 | 0.004068 | 245812.965778 |
| CSE-CIC-IDS2018 | 0.970763 | 0.957066 | 0.949745 | 0.995354 | 0.969660 | 0.001394 | 0.003011 | 332100.140972 |

## Clean-Data Ablation Assessment

| Dataset | Configuration | BA | F1 | MCC | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| CICIDS2017 | A0_Original_Feature_XGBoost | 0.999189 | 0.997056 | 0.996461 | 0.999975 |
| CICIDS2017 | A1_Best_Single_View | 0.999105 | 0.996760 | 0.996107 | 0.999975 |
| CICIDS2017 | A2_Unweighted_Multiview_Mean | 0.998799 | 0.996243 | 0.995484 | 0.999914 |
| CICIDS2017 | A3_Global_View_Weights_Only | 0.999102 | 0.996805 | 0.996160 | 0.999923 |
| CICIDS2017 | A4_FERF_Without_Integrity | 0.999102 | 0.996805 | 0.996160 | 0.999918 |
| CICIDS2017 | A5_FERF_Without_Quality | 0.999099 | 0.996790 | 0.996142 | 0.999924 |
| CICIDS2017 | A6_FERF_Without_Temporal | 0.999102 | 0.996805 | 0.996160 | 0.999922 |
| CICIDS2017 | A7_Full_FERF | 0.999102 | 0.996805 | 0.996160 | 0.999919 |
| CSE-CIC-IDS2018 | A0_Original_Feature_XGBoost | 0.971482 | 0.955614 | 0.947961 | 0.996014 |
| CSE-CIC-IDS2018 | A1_Best_Single_View | 0.970763 | 0.957066 | 0.949745 | 0.995354 |
| CSE-CIC-IDS2018 | A2_Unweighted_Multiview_Mean | 0.966262 | 0.938393 | 0.927598 | 0.980633 |
| CSE-CIC-IDS2018 | A3_Global_View_Weights_Only | 0.970763 | 0.957066 | 0.949745 | 0.995354 |
| CSE-CIC-IDS2018 | A4_FERF_Without_Integrity | 0.970763 | 0.957066 | 0.949745 | 0.995354 |
| CSE-CIC-IDS2018 | A5_FERF_Without_Quality | 0.970763 | 0.957066 | 0.949745 | 0.995354 |
| CSE-CIC-IDS2018 | A6_FERF_Without_Temporal | 0.970763 | 0.957066 | 0.949745 | 0.995354 |
| CSE-CIC-IDS2018 | A7_Full_FERF | 0.970763 | 0.957066 | 0.949745 | 0.995354 |

## Cross-Dataset Assessment

| Source | Target | Source-test BA | Target BA | Target F1 | Target MCC | Target ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| CICIDS2017 | CSE-CIC-IDS2018 | 0.997679 | 0.499946 | 0.009954 | -0.000531 | 0.640776 |
| CSE-CIC-IDS2018 | CICIDS2017 | 0.969981 | 0.554272 | 0.197730 | 0.284462 | 0.629397 |

## Computational Efficiency

| Dataset | Historical Training s | Model Size MB | Analytical Inference s | Latency ms/record | Throughput records/s | FERF Overhead % | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CICIDS2017 | 29.506388 | 7.908527 | 0.813631 | 0.004068 | 245812.965778 | 4.081663 | 2109.761719 |
| CSE-CIC-IDS2018 | 21.178723 | 9.368330 | 0.602231 | 0.003011 | 332100.140972 | 6.529239 | 2193.132812 |

## Scalability

| Dataset | Slope s/record | Intercept s | R² | Minimum Rows | Maximum Rows |
| --- | --- | --- | --- | --- | --- |
| CICIDS2017 | 0.000003990 | 0.002510737 | 0.999115357 | 1000.000000000 | 200001.000000000 |
| CSE-CIC-IDS2018 | 0.000002969 | 0.003732352 | 0.998768534 | 1000.000000000 | 200001.000000000 |

## Repeated-CV Performance

| Dataset | Configuration | N | Macro-F1 | Weighted-F1 | BA | ROC-AUC | PR-AUC | Kappa | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CICIDS2017 | C0_Original_XGBoost | 50.000000 | 0.997896 | 0.998812 | 0.998728 | 0.999957 | 0.999794 | 0.995792 | 0.001967 |
| CICIDS2017 | C1_Unweighted_Multiview | 50.000000 | 0.997103 | 0.998363 | 0.998358 | 0.999885 | 0.999522 | 0.994207 | 0.018496 |
| CICIDS2017 | C2_Global_Weighted_Multiview | 50.000000 | 0.997802 | 0.998759 | 0.998705 | 0.999911 | 0.999607 | 0.995605 | 0.008207 |
| CICIDS2017 | C3_Full_FERF | 50.000000 | 0.997809 | 0.998762 | 0.998706 | 0.999913 | 0.999618 | 0.995617 | 0.007733 |
| CSE-CIC-IDS2018 | C0_Original_XGBoost | 50.000000 | 0.973038 | 0.986388 | 0.969935 | 0.995168 | 0.983783 | 0.946076 | 0.018053 |
| CSE-CIC-IDS2018 | C1_Unweighted_Multiview | 50.000000 | 0.962700 | 0.981058 | 0.965064 | 0.978952 | 0.965004 | 0.925401 | 0.076012 |
| CSE-CIC-IDS2018 | C2_Global_Weighted_Multiview | 50.000000 | 0.974032 | 0.986908 | 0.969637 | 0.993012 | 0.980607 | 0.948064 | 0.024263 |
| CSE-CIC-IDS2018 | C3_Full_FERF | 50.000000 | 0.973925 | 0.986852 | 0.969660 | 0.992922 | 0.980438 | 0.947852 | 0.024753 |

## Claim-to-Evidence Matrix

| Claim | Dataset/Transfer | Status | Statement | Evidence |
| --- | --- | --- | --- | --- |
| C1 | CICIDS2017 | SUPPORTED | Full FERF outperforms naive unweighted multi-view fusion on clean repeated-CV data. | ΔBA=0.000349; Holm-adjusted paired t-test p=2.48119e-13 |
| C1 | CSE-CIC-IDS2018 | SUPPORTED | Full FERF outperforms naive unweighted multi-view fusion on clean repeated-CV data. | ΔBA=0.004597; Holm-adjusted paired t-test p=1.0227e-41 |
| C2 | CICIDS2017 | NOT_SUPPORTED | Record-specific reliability provides a significant clean-data advantage over optimized global weighting. | ΔBA=0.000001; Holm-adjusted paired t-test p=1 |
| C2 | CSE-CIC-IDS2018 | NOT_SUPPORTED | Record-specific reliability provides a significant clean-data advantage over optimized global weighting. | ΔBA=0.000023; Holm-adjusted paired t-test p=0.982991 |
| C3 | CICIDS2017 | NOT_SUPPORTED | Full FERF universally outperforms original-feature XGBoost on clean data. | ΔBA=-0.000022; Holm-adjusted paired t-test p=0.546866 |
| C3 | CSE-CIC-IDS2018 | NOT_SUPPORTED | Full FERF universally outperforms original-feature XGBoost on clean data. | ΔBA=-0.000275; Holm-adjusted paired t-test p=0.000168619 |
| C4 | CICIDS2017 | SUPPORTED | Full FERF improves robustness relative to global weighting under severe record-view evidence degradation. | 50% degradation: FERF BA=0.961730, global BA=0.749142, Δ=0.212588 |
| C4 | CSE-CIC-IDS2018 | NOT_SUPPORTED | Full FERF improves robustness relative to global weighting under severe record-view evidence degradation. | 50% degradation: FERF BA=0.735554, global BA=0.735554, Δ=0.000000 |
| C5 | CICIDS2017->CSE-CIC-IDS2018 | NOT_SUPPORTED | The framework demonstrates strong zero-shot cross-dataset classification. | target BA=0.499946; target ROC-AUC=0.640776 |
| C5 | CSE-CIC-IDS2018->CICIDS2017 | NOT_SUPPORTED | The framework demonstrates strong zero-shot cross-dataset classification. | target BA=0.554272; target ROC-AUC=0.629397 |
| C6 | CICIDS2017->CSE-CIC-IDS2018 | SUPPORTED | The harmonized semantic representation retains above-chance ranking information under cross-dataset transfer. | target ROC-AUC=0.640776 |
| C6 | CSE-CIC-IDS2018->CICIDS2017 | SUPPORTED | The harmonized semantic representation retains above-chance ranking information under cross-dataset transfer. | target ROC-AUC=0.629397 |
| C7 | CICIDS2017 | SUPPORTED | FERF introduces limited computational overhead relative to view-level inference. | FERF overhead=4.082% |
| C7 | CSE-CIC-IDS2018 | SUPPORTED | FERF introduces limited computational overhead relative to view-level inference. | FERF overhead=6.529% |
| C8 | CICIDS2017 | SUPPORTED | Analytical inference exhibits near-linear scaling with record count. | linear-fit R²=0.999115 |
| C8 | CSE-CIC-IDS2018 | SUPPORTED | Analytical inference exhibits near-linear scaling with record count. | linear-fit R²=0.998769 |

## Integrated Interpretation

The final assessment should distinguish intrusion-classification performance from the broader forensic evidence-synthesis contribution. Clean-data results indicate that the Full FERF configuration should not be presented as universally superior to the strongest original-feature classifier or to optimized global weighting. The strongest clean-data statistical evidence supports structured multi-view weighting relative to naive equal averaging.

Record-specific reliability should be interpreted primarily in relation to evidence condition and robustness rather than as a mechanism that must always increase clean-data discrimination. This interpretation is consistent with the degradation experiment, where the benefit is dataset dependent.

Cross-dataset results should be reported as evidence of substantial domain sensitivity. Above-chance ROC-AUC may indicate retained ranking information, but the observed threshold-dependent degradation does not support a claim of strong zero-shot cross-dataset classification.

Computational measurements support efficient analytical inference and near-linear scaling over the tested record-count range. Efficiency claims should refer to the analytical inference path rather than the entire digital-forensic acquisition and preprocessing workflow.

## Evidence Status

- Supported claim rows: 9
- Not-supported claim rows: 7
- Insufficient-evidence rows: 0

Experiment 8 does not create additional experimental observations; it only consolidates and interprets the validated outputs produced by Experiments 1–7.