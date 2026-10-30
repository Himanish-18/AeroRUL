# AeroRUL — Phase 5.2 Documentation Change Log

This document provides a complete, forensic audit trail of all documentation, reporting, and presentation corrections applied during **Phase 5.2: Documentation Corrections & Final Report Reconciliation**.

## 1. Material Changes Summary

| Location | Issue | Change Made | Evidence / Source Artifact | Status |
|---|---|---|---|:---:|
| `reports/phase5_1/phase5_1_final_audit_report.md` (Section G) | Copy-pasted feature-window rationale across FD001, FD003, FD004 referencing FD002's W75 result. | Replaced copy-pasted rationale with dataset-specific explanations grounded in empirical window ablation results. | `reports/phase5/window_ablation_metrics.csv` | **Corrected** |
| `reports/phase5_1/phase5_1_final_audit_report.md` (Section C) | Missing per-metric original vs recalculated values and differences for all claimed metrics. | Added comprehensive 20-row metric-by-metric comparison table showing MAE, RMSE, R², Mean Signed Error, and NASA Score Sum with exact differences and precision policy. | `reports/phase5/final_test_metrics.csv`, `reports/phase5_1/independent_test_metric_recalculation.csv` | **Corrected** |
| `reports/phase5_1/phase5_1_final_audit_report.md` (Section B) | Table column labelled `Features` contained configuration strings (`ALL_WINDOWS`, `W75`) rather than numeric counts. | Split into two explicit columns: `Feature Configuration` and `Number of Features`, populating exact numeric counts (670, 231, 681, 768). | `models/final/*/feature_names.json`, `reports/phase5/final_model_manifest.json` | **Corrected** |
| `reports/phase5_1/phase5_1_final_audit_report.md` (Section H) | Unreconciled Inner-CV MAE values between feature window section and target cap section (14.26 vs 14.08 for FD001; 16.89 vs 16.41 for FD002). | Added Section H.1 with a dedicated cross-experiment reconciliation table explaining the sequential development protocol (uncapped window ablation vs capped sensitivity). | `reports/phase5/window_ablation_metrics.csv`, `reports/phase5/target_cap_sensitivity.csv` | **Clarified** |
| `reports/phase5_1/phase5_1_final_audit_report.md` (Section K) | Unsupported superlative claiming "state-of-the-art classical performance" without literature baseline comparisons. | Replaced with precise, factual statement describing measured official test endpoint MAEs (11.43, 13.78, 14.68, 22.81 cycles) and RUL-range performance. | Official test predictions and evaluation records (`reports/phase5/final_test_metrics.csv`) | **Corrected** |
| `reports/phase5_1/window_selection_reconciliation.csv` | Selection rationale column contained copy-pasted FD002 text across all rows. | Updated generation script (`scripts/run_phase5_1_audit.py`) to generate subset-specific rationales and audit findings. | `reports/phase5/window_ablation_metrics.csv` | **Corrected** |
| `reports/phase5_1/final_manifest_audit.csv` | Manifest audit lacked explicit numeric feature count column. | Added `n_features` column to the output CSV and audit pipeline. | `models/final/*/feature_names.json` | **Clarified** |

---

## 2. Detailed Original vs. Corrected Text Comparisons

### Correction A: Feature-Window Selection Rationale (Section G)

#### Original Text (Phase 5.1):
```markdown
| Dataset | Best Inner-CV Config | Inner-CV MAE | Best Dev-Val Config | Dev-Val MAE | Selected Final Config | Selection Rationale |
|---|---|---:|---|---:|---|---|
| **FD001** | ALL_WINDOWS | 14.26 | W75 | 11.48 | **ALL_WINDOWS** | Inner-CV Engine-Macro MAE was declared primary criterion: ALL_WINDOWS achieved 14.26. In FD002, W75 won on Inner-CV (16.89 vs 16.90) with only 231 features vs 776. |
| **FD002** | W75 | 16.89 | ALL_WINDOWS | 16.65 | **W75** | Inner-CV Engine-Macro MAE was declared primary criterion: W75 achieved 16.89. In FD002, W75 won on Inner-CV (16.89 vs 16.90) with only 231 features vs 776. |
| **FD003** | ALL_WINDOWS | 25.98 | W50 | 20.33 | **ALL_WINDOWS** | Inner-CV Engine-Macro MAE was declared primary criterion: ALL_WINDOWS achieved 25.98. In FD002, W75 won on Inner-CV (16.89 vs 16.90) with only 231 features vs 776. |
| **FD004** | ALL_WINDOWS | 25.60 | ALL_WINDOWS | 27.63 | **ALL_WINDOWS** | Inner-CV Engine-Macro MAE was declared primary criterion: ALL_WINDOWS achieved 25.60. In FD002, W75 won on Inner-CV (16.89 vs 16.90) with only 231 features vs 776. |
```

#### Corrected Text (Phase 5.2):
```markdown
| Dataset | Best Inner-CV Config | Inner-CV MAE | Best Dev-Val Config | Dev-Val MAE | Selected Final Config | Selection Rationale |
|---|---|---:|---|---:|---|---|
| **FD001** | ALL_WINDOWS | 14.26 | W75 | 11.48 | **ALL_WINDOWS** | Inner-CV Engine-Macro MAE was designated primary criterion: ALL_WINDOWS achieved 14.26 cycles (670 features), outperforming all individual windows on Inner-CV (W75: 14.57 cycles). Despite W75 achieving lower Dev-Val MAE (11.48 vs 11.72 cycles), ALL_WINDOWS was selected to capture multi-timescale degradation dynamics. |
| **FD002** | W75 | 16.89 | ALL_WINDOWS | 16.65 | **W75** | Inner-CV Engine-Macro MAE was designated primary criterion: W75 achieved 16.89 cycles (231 features), outperforming ALL_WINDOWS (16.90 cycles, 776 features) and all smaller windows on Inner-CV. Despite ALL_WINDOWS achieving slightly lower Dev-Val MAE (16.65 vs 16.95 cycles), W75 was selected for optimal Inner-CV performance and a 3.4x more parsimonious feature space. |
| **FD003** | ALL_WINDOWS | 25.98 | W50 | 20.33 | **ALL_WINDOWS** | Inner-CV Engine-Macro MAE was designated primary criterion: ALL_WINDOWS achieved 25.98 cycles (681 features), significantly outperforming all individual windows on Inner-CV (W75: 27.07, W50: 29.99 cycles). Despite W50 achieving lower Dev-Val MAE (20.33 vs 21.60 cycles), ALL_WINDOWS was selected to represent complex multi-fault modes across multiple timescales. |
| **FD004** | ALL_WINDOWS | 25.60 | ALL_WINDOWS | 27.63 | **ALL_WINDOWS** | Inner-CV Engine-Macro MAE was designated primary criterion: ALL_WINDOWS achieved 25.60 cycles (768 features), outperforming all individual windows on Inner-CV (W75: 25.85 cycles). ALL_WINDOWS also achieved the best Dev-Val MAE (27.63 cycles vs W75's 28.31 cycles), winning unambiguously across both evaluation splits. |
```

---

### Correction B: Complete Metric-Verification Evidence (Section C)

#### Original Text (Phase 5.1):
Contained only a single 4-row summary table showing MAE, RMSE, R², and NASA score sum, without explicit display of Original vs Recalculated values for R² or Mean Signed Error, and without raw absolute differences.

#### Corrected Text (Phase 5.2):
Added an explicit 20-row table covering all 4 datasets across all 5 metrics with exact differences and documentation of precision and conventions:
- **Signed Error Convention**: $e_i = \hat{y}_i - y_i$ (>0 overprediction, <0 underprediction).
- **Precision Policy**: Original values from `final_test_metrics.csv`; recalculations from unit-level test predictions and `RUL_FD*.txt`. Differences below $10^{-8}$ arise strictly from IEEE 754 floating-point operations.
- Full visibility of MAE, RMSE, R², Mean Signed Error, and NASA Score Sum.

---

### Correction C: Frozen Configuration Table (Section B)

#### Original Text (Phase 5.1):
```markdown
| Dataset | Manifest Model | Saved Model Artifact | Target Cap | Feature Config | Features | Preprocessing Match | Parameter Match | Audit Status |
|---|---|---|---:|---|---:|---|---|---|
| FD001 | LightGBM | `LGBMRegressor` | 150 | ALL_WINDOWS | ALL_WINDOWS | ✓ | ✓ | **VERIFIED** |
| FD002 | LightGBM | `LGBMRegressor` | 150 | W75 | W75 | ✓ | ✓ | **VERIFIED** |
| FD003 | LightGBM | `LGBMRegressor` | None | ALL_WINDOWS | ALL_WINDOWS | ✓ | ✓ | **VERIFIED** |
| FD004 | LightGBM | `LGBMRegressor` | None | ALL_WINDOWS | ALL_WINDOWS | ✓ | ✓ | **VERIFIED** |
```

#### Corrected Text (Phase 5.2):
```markdown
| Dataset | Manifest Model | Saved Model Artifact | Target Cap | Feature Configuration | Number of Features | Preprocessing Match | Parameter Match | Audit Status |
|---|---|---|---:|---|---:|---|---|---|
| FD001 | LightGBM | `LGBMRegressor` | 150 | ALL_WINDOWS | 670 | ✓ | ✓ | **VERIFIED** |
| FD002 | LightGBM | `LGBMRegressor` | 150 | W75 | 231 | ✓ | ✓ | **VERIFIED** |
| FD003 | LightGBM | `LGBMRegressor` | None | ALL_WINDOWS | 681 | ✓ | ✓ | **VERIFIED** |
| FD004 | LightGBM | `LGBMRegressor` | None | ALL_WINDOWS | 768 | ✓ | ✓ | **VERIFIED** |
```

---

### Correction D: Inner-CV MAE Cross-Experiment Reconciliation (Section H.1)

#### Original Text (Phase 5.1):
The report presented 14.26 (FD001) and 16.89 (FD002) in Section G, but 14.08 (FD001) and 16.41 (FD002) in Section H, with no explanation of why they differed.

#### Corrected Text (Phase 5.2):
Added Section H.1 documenting:
1. **Window Ablation Protocol**: Evaluated feature window sets under uncapped baseline targets (`target_cap = None`), yielding 14.26 cycles (FD001 ALL_WINDOWS) and 16.89 cycles (FD002 W75).
2. **Target Cap Sensitivity Protocol**: Varied training target caps on the selected feature configuration; capping at 150 cycles suppressed early-life variance, improving Inner-CV MAE to 14.08 cycles (FD001) and 16.41 cycles (FD002).
3. **Multi-Fault Subsets (FD003/FD004)**: Capping degraded MAE, so `Cap=None` remained optimal, yielding identical 25.98 and 25.60 cycles across both tables.
Both sets of values are valid outputs of different stages of the pre-registered model development protocol.

---

### Correction E: Removal of Unsupported Performance Claim (Section K)

#### Original Text (Phase 5.1):
`- The LightGBM ensemble models demonstrate state-of-the-art classical performance on C-MAPSS, with endpoint MAEs of **11.43 (FD001)**, **13.78 (FD002)**, **14.68 (FD003)**, and **22.81 (FD004)** cycles.`

#### Corrected Text (Phase 5.2):
`- The frozen LightGBM models achieved official C-MAPSS test endpoint MAEs of **11.43 (FD001)**, **13.78 (FD002)**, **14.68 (FD003)**, and **22.81 (FD004)** cycles. Performance varied by subset, with higher errors on FD004 and lower errors near the end of life in the reported RUL-bin analysis.`
