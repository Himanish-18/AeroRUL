"""
AeroRUL — Phase 5.2 Documentation & Report Reconciliation Automated Tests
Verifies that all Phase 5.2 documentation corrections, table structural fixes,
metric comparisons, and evidence-backed rationales are intact and accurate.
"""

import json
from pathlib import Path
import re
import pandas as pd
import pytest

from src.config import AVAILABLE_SUBSETS, REPORTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE5_DIR = REPORTS_DIR / "phase5"
PHASE5_1_DIR = REPORTS_DIR / "phase5_1"
PHASE5_2_DIR = REPORTS_DIR / "phase5_2"
MODELS_FINAL_DIR = PROJECT_ROOT / "models" / "final"


@pytest.fixture(scope="session")
def report_text():
    report_path = PHASE5_1_DIR / "phase5_1_final_audit_report.md"
    assert report_path.exists(), f"Missing audit report: {report_path}"
    return report_path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def manifest():
    manifest_path = PHASE5_DIR / "final_model_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TEST 1: All Markdown Deliverables Exist and Non-Empty
# ---------------------------------------------------------------------------
def test_markdown_deliverables_exist_and_non_empty():
    """Verify that all Phase 5.1 and 5.2 report files exist and have substantial content."""
    deliverables = [
        PHASE5_1_DIR / "phase5_1_final_audit_report.md",
        PHASE5_2_DIR / "phase5_2_change_log.md",
        PHASE5_2_DIR / "phase5_2_completion_report.md",
    ]
    for doc in deliverables:
        assert doc.exists(), f"Missing deliverable: {doc}"
        assert doc.stat().st_size > 1000, f"Deliverable too small: {doc}"


# ---------------------------------------------------------------------------
# TEST 2: Feature-Window Rationale is Dataset-Specific (No Copy-Paste)
# ---------------------------------------------------------------------------
def test_feature_window_rationale_dataset_specific(report_text):
    """Verify that feature window rationales in Section G are unique and dataset-specific."""
    assert "In FD002, W75 won on Inner-CV (16.89 vs 16.90)" not in report_text, (
        "Residual copy-pasted FD002 rationale found in report text"
    )

    # Check window reconciliation CSV
    win_recon_path = PHASE5_1_DIR / "window_selection_reconciliation.csv"
    assert win_recon_path.exists(), f"Missing window reconciliation CSV: {win_recon_path}"
    df_win = pd.read_csv(win_recon_path)

    rationales = df_win["selection_rationale"].tolist()
    assert len(rationales) == 4
    # All 4 rationales must be distinct
    assert len(set(rationales)) == 4, "Selection rationales across subsets are not distinct"

    # Specific keywords per dataset
    fd001_rat = df_win.loc[df_win["dataset"] == "FD001", "selection_rationale"].values[0]
    fd002_rat = df_win.loc[df_win["dataset"] == "FD002", "selection_rationale"].values[0]
    fd003_rat = df_win.loc[df_win["dataset"] == "FD003", "selection_rationale"].values[0]
    fd004_rat = df_win.loc[df_win["dataset"] == "FD004", "selection_rationale"].values[0]

    assert "670 features" in fd001_rat
    assert "231 features" in fd002_rat and "parsimonious" in fd002_rat
    assert "681 features" in fd003_rat and "multi-fault" in fd003_rat
    assert "768 features" in fd004_rat and "unambiguously" in fd004_rat


# ---------------------------------------------------------------------------
# TEST 3: Section C Metric-Verification Table Completeness
# ---------------------------------------------------------------------------
def test_metric_verification_table_completeness(report_text):
    """Verify that Section C contains the full 20-row comparison table."""
    metrics = ["MAE", "RMSE", "R²", "Mean Signed Error", "NASA Score Sum"]
    for subset in AVAILABLE_SUBSETS:
        for m in metrics:
            row_prefix = f"| {subset} | {m} |"
            assert row_prefix in report_text, f"Missing metric comparison row: {row_prefix}"

    assert "EXACT_MATCH" in report_text
    assert "Metric Conventions and Rounding Policy" in report_text


# ---------------------------------------------------------------------------
# TEST 4: Section B Feature Count Columns and Numeric Values
# ---------------------------------------------------------------------------
def test_feature_count_table_columns_and_values(report_text):
    """Verify that Section B has separate Feature Configuration and Number of Features columns."""
    assert "| Feature Configuration | Number of Features |" in report_text

    expected_counts = {"FD001": 670, "FD002": 231, "FD003": 681, "FD004": 768}
    for subset, count in expected_counts.items():
        pattern = rf"\|\s*{subset}\s*\|\s*LightGBM\s*\|\s*`LGBMRegressor`\s*\|\s*(?:150|None)\s*\|\s*(?:ALL_WINDOWS|W75)\s*\|\s*{count}\s*\|"
        assert re.search(pattern, report_text), f"Row for {subset} does not match expected count {count}"


# ---------------------------------------------------------------------------
# TEST 5: Inner-CV Discrepancy Cross-Experiment Reconciliation
# ---------------------------------------------------------------------------
def test_inner_cv_discrepancy_reconciled(report_text):
    """Verify that Section H.1 formally reconciles Inner-CV MAE values across experiment stages."""
    assert "Inner-CV MAE Cross-Experiment Reconciliation" in report_text
    assert "14.26" in report_text and "14.08" in report_text
    assert "16.89" in report_text and "16.41" in report_text
    assert "RECONCILED" in report_text
    assert "sequential stages" in report_text


# ---------------------------------------------------------------------------
# TEST 6: Removal of Unsupported Superlative Language
# ---------------------------------------------------------------------------
def test_no_unsupported_state_of_the_art_language(report_text):
    """Verify that unverified superlative 'state-of-the-art' is absent from the audit report."""
    assert not re.search(r"\bstate-of-the-art\b", report_text, re.IGNORECASE), (
        "Found unsupported superlative 'state-of-the-art' in report"
    )
    # Check that the qualified replacement sentence is present
    assert "The frozen LightGBM models achieved official C-MAPSS test endpoint MAEs" in report_text


# ---------------------------------------------------------------------------
# TEST 7: Frozen Artifacts and Evaluation Metrics Immutability
# ---------------------------------------------------------------------------
def test_frozen_artifacts_and_metrics_immutable(manifest):
    """Verify that model manifest, saved models, and test metrics remain unchanged."""
    expected_configs = {
        "FD001": {"model": "LightGBM", "cap": "150", "features": 670, "win": "ALL_WINDOWS"},
        "FD002": {"model": "LightGBM", "cap": "150", "features": 231, "win": "W75"},
        "FD003": {"model": "LightGBM", "cap": "None", "features": 681, "win": "ALL_WINDOWS"},
        "FD004": {"model": "LightGBM", "cap": "None", "features": 768, "win": "ALL_WINDOWS"},
    }

    for subset, exp in expected_configs.items():
        cfg = manifest["selected_configurations"][subset]
        assert cfg["model_family"] == exp["model"]
        assert str(cfg["target_cap"]) == exp["cap"]
        assert cfg["n_features"] == exp["features"]
        assert cfg["feature_configuration"] == exp["win"]

        # Check model features file
        feat_file = MODELS_FINAL_DIR / subset / "feature_names.json"
        with open(feat_file, "r", encoding="utf-8") as f:
            features = json.load(f)
        assert len(features) == exp["features"]

    # Verify official metrics CSV exists and matches expected values
    metrics_path = PHASE5_DIR / "final_test_metrics.csv"
    assert metrics_path.exists()
    df_metrics = pd.read_csv(metrics_path)
    assert len(df_metrics) == 4

    expected_maes = {"FD001": 11.4348, "FD002": 13.7838, "FD003": 14.6807, "FD004": 22.8125}
    for _, r in df_metrics.iterrows():
        sub = r["dataset"]
        assert pytest.approx(r["endpoint_mae"], abs=1e-4) == expected_maes[sub]
