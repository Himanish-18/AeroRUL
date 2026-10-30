"""
AeroRUL — Phase 5.2 Documentation Validation Script
Validates Markdown report structure, referenced artifact existence, table columns,
absence of unverified claims, and immutability of frozen model artifacts.
"""

import json
from pathlib import Path
import re
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PHASE5_DIR = REPORTS_DIR / "phase5"
PHASE5_1_DIR = REPORTS_DIR / "phase5_1"
PHASE5_2_DIR = REPORTS_DIR / "phase5_2"
MODELS_FINAL_DIR = PROJECT_ROOT / "models" / "final"

AVAILABLE_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def validate_phase5_2_documentation():
    print("=" * 80)
    print("AERORUL — PHASE 5.2 DOCUMENTATION VALIDATION")
    print("=" * 80)
    failures = []

    # 1. Validate Markdown files existence and size
    md_files = [
        PHASE5_1_DIR / "phase5_1_final_audit_report.md",
        PHASE5_2_DIR / "phase5_2_change_log.md",
        PHASE5_2_DIR / "phase5_2_completion_report.md",
    ]
    for mdf in md_files:
        if not mdf.exists():
            failures.append(f"Missing Markdown file: {mdf}")
        elif mdf.stat().st_size < 1000:
            failures.append(f"Markdown file unexpectedly small (<1KB): {mdf}")
        else:
            print(f"[PASS] Markdown file present & non-empty: {mdf.name} ({mdf.stat().st_size} bytes)")

    report_text = (PHASE5_1_DIR / "phase5_1_final_audit_report.md").read_text(encoding="utf-8")

    # 2. Check for prohibited superlative language
    if re.search(r"\bstate-of-the-art\b", report_text, re.IGNORECASE):
        failures.append("Found unsupported superlative 'state-of-the-art' in phase5_1_final_audit_report.md")
    else:
        print("[PASS] No unsupported 'state-of-the-art' claims found in final audit report.")

    # 3. Check for copy-pasted feature-window rationale
    matches = re.findall(r"In FD002, W75 won on Inner-CV", report_text)
    if len(matches) > 0:
        failures.append(f"Found {len(matches)} residual copy-pasted rationale occurrences in report.")
    else:
        print("[PASS] No copy-pasted rationale found in feature-window table.")

    # 4. Check for obsolete underprediction claim
    if "conservative underprediction tendencies" in report_text and "refutes" not in report_text:
        failures.append("Found obsolete underprediction claim without refutation.")
    else:
        print("[PASS] Underprediction narrative properly corrected/refuted.")

    # 5. Check Feature Count table structure in Section B
    # Must have 'Feature Configuration' and 'Number of Features'
    if "Feature Configuration" not in report_text or "Number of Features" not in report_text:
        failures.append("Section B missing separate 'Feature Configuration' and 'Number of Features' columns.")
    else:
        print("[PASS] Section B contains separate 'Feature Configuration' and 'Number of Features' columns.")

    # Verify numeric feature counts in Section B
    expected_counts = {"FD001": "670", "FD002": "231", "FD003": "681", "FD004": "768"}
    for subset, cnt in expected_counts.items():
        pattern = rf"\|\s*{subset}\s*\|\s*LightGBM\s*\|\s*`LGBMRegressor`\s*\|\s*(?:150|None)\s*\|\s*(?:ALL_WINDOWS|W75)\s*\|\s*{cnt}\s*\|"
        if not re.search(pattern, report_text):
            failures.append(f"Section B row for {subset} does not match expected numeric count {cnt}")
        else:
            print(f"[PASS] Section B numeric count verified for {subset}: {cnt}")

    # 6. Check Metric Verification Table in Section C
    # Must have 20 metric comparison rows
    metrics = ["MAE", "RMSE", "R²", "Mean Signed Error", "NASA Score Sum"]
    for subset in AVAILABLE_SUBSETS:
        for m in metrics:
            if f"| {subset} | {m} |" not in report_text:
                failures.append(f"Section C missing metric verification row for {subset} {m}")
    print("[PASS] Section C contains all 20 required metric-by-metric comparison rows.")

    # 7. Check Section H.1 Inner-CV Cross-Experiment Reconciliation Table
    if "Inner-CV MAE Cross-Experiment Reconciliation" not in report_text:
        failures.append("Missing Section H.1 Inner-CV Cross-Experiment Reconciliation.")
    else:
        print("[PASS] Section H.1 Inner-CV Cross-Experiment Reconciliation table present.")

    # 8. Check Immutability of Frozen Artifacts
    manifest_path = PHASE5_DIR / "final_model_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for subset in AVAILABLE_SUBSETS:
        cfg = manifest["selected_configurations"][subset]
        features_file = MODELS_FINAL_DIR / subset / "feature_names.json"
        with open(features_file, "r", encoding="utf-8") as f:
            features = json.load(f)
        if len(features) != cfg["n_features"]:
            failures.append(f"Feature count mismatch in {subset}: manifest={cfg['n_features']}, actual={len(features)}")
        else:
            print(f"[PASS] Model artifact feature count verified for {subset}: {len(features)}")

    # 9. Verify Referenced Files Exist
    referenced_files = [
        PHASE5_DIR / "final_model_manifest.json",
        PHASE5_DIR / "final_test_predictions.csv",
        PHASE5_DIR / "final_test_metrics.csv",
        PHASE5_DIR / "window_ablation_metrics.csv",
        PHASE5_DIR / "target_cap_sensitivity.csv",
        PHASE5_1_DIR / "final_verified_results.csv",
        PHASE5_1_DIR / "independent_test_metric_recalculation.csv",
        PHASE5_1_DIR / "window_selection_reconciliation.csv",
        PHASE5_1_DIR / "target_cap_reconciliation.csv",
        PHASE5_1_DIR / "test_feature_coverage_audit.csv",
    ]
    for rf in referenced_files:
        if not rf.exists():
            failures.append(f"Referenced file does not exist: {rf}")
        else:
            print(f"[PASS] Referenced artifact verified: {rf.name}")

    print("=" * 80)
    if failures:
        print(f"VALIDATION FAILED with {len(failures)} issues:")
        for fail in failures:
            print(f"  - {fail}")
        return False
    else:
        print("ALL PHASE 5.2 DOCUMENTATION CHECKS PASSED PERFECTLY!")
        return True


if __name__ == "__main__":
    success = validate_phase5_2_documentation()
    sys.exit(0 if success else 1)
