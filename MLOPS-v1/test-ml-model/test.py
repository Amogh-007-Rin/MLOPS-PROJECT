"""
ML model tests — feature engineering, artifact loading, and prediction shape.

Run with:
    pytest test-ml-model/test.py -v
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Allow importing from server/ for the shared feature-engineering logic
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "model", "artifacts")
sys.path.insert(0, os.path.join(REPO_ROOT, "server"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def feature_engineer(
    est_diameter_min: float,
    est_diameter_max: float,
    relative_velocity: float,
    absolute_magnitude: float,
    miss_distance: float,
) -> dict:
    """Mirrors the feature engineering in Pre-processing.ipynb and server/app.py."""
    diameter_avg          = (est_diameter_min + est_diameter_max) / 2
    diameter_ratio        = est_diameter_max / est_diameter_min
    log_diameter_avg      = np.log1p(diameter_avg)
    log_diameter_ratio    = np.log1p(diameter_ratio)
    log_relative_velocity = np.log1p(relative_velocity)
    log_miss_distance     = np.log1p(miss_distance)
    return {
        "diameter_avg":          diameter_avg,
        "diameter_ratio":        diameter_ratio,
        "log_diameter_avg":      log_diameter_avg,
        "log_diameter_ratio":    log_diameter_ratio,
        "log_relative_velocity": log_relative_velocity,
        "log_miss_distance":     log_miss_distance,
        "absolute_magnitude":    absolute_magnitude,
    }


ARTIFACTS_AVAILABLE = all(
    os.path.exists(os.path.join(ARTIFACT_DIR, f))
    for f in [
        "classifier.joblib",
        "regressor.joblib",
        "scaler_clf.joblib",
        "scaler_reg.joblib",
        "feature_names.json",
    ]
)
requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE,
    reason="model/artifacts/ not found — run the training notebooks first",
)

# ── Feature engineering tests ─────────────────────────────────────────────────

def test_feature_engineering_basic():
    feats = feature_engineer(
        est_diameter_min=0.12,
        est_diameter_max=0.27,
        relative_velocity=48000.0,
        absolute_magnitude=22.1,
        miss_distance=14_500_000.0,
    )
    assert feats["diameter_avg"] == pytest.approx(0.195, abs=1e-6)
    assert feats["diameter_ratio"] == pytest.approx(0.27 / 0.12, rel=1e-6)
    # All log values must be finite and non-negative (log1p of positive inputs)
    for key in ("log_diameter_avg", "log_diameter_ratio", "log_relative_velocity", "log_miss_distance"):
        assert np.isfinite(feats[key])
        assert feats[key] > 0


def test_feature_engineering_no_inf_or_nan():
    feats = feature_engineer(0.001, 0.002, 1.0, 25.0, 1.0)
    for v in feats.values():
        assert np.isfinite(v), f"Got non-finite value: {v}"


def test_log_miss_distance_inverse():
    miss_distance = 14_500_000.0
    feats = feature_engineer(0.12, 0.27, 48000.0, 22.1, miss_distance)
    recovered = np.expm1(feats["log_miss_distance"])
    assert recovered == pytest.approx(miss_distance, rel=1e-9)


def test_diameter_ratio_always_gte_one():
    # max >= min so ratio should be >= 1
    feats = feature_engineer(0.05, 0.20, 30000.0, 21.0, 5_000_000.0)
    assert feats["diameter_ratio"] >= 1.0


# ── Artifact and model tests (skipped if artifacts absent) ────────────────────

@requires_artifacts
def test_feature_names_json_structure():
    with open(os.path.join(ARTIFACT_DIR, "feature_names.json")) as f:
        names = json.load(f)
    assert "clf_features" in names
    assert "reg_features" in names
    assert len(names["clf_features"]) == 7
    assert len(names["reg_features"]) == 6
    # regression features must not include log_miss_distance (it's the target)
    assert "log_miss_distance" not in names["reg_features"]


@requires_artifacts
def test_classifier_predicts_binary():
    import joblib
    clf        = joblib.load(os.path.join(ARTIFACT_DIR, "classifier.joblib"))
    scaler_clf = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_clf.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_names.json")) as f:
        names = json.load(f)

    feats = feature_engineer(0.12, 0.27, 48000.0, 22.1, 14_500_000.0)
    clf_input = pd.DataFrame([[
        feats["log_diameter_avg"], feats["log_diameter_ratio"],
        feats["log_relative_velocity"], feats["log_miss_distance"],
        feats["absolute_magnitude"], feats["diameter_avg"], feats["diameter_ratio"],
    ]], columns=names["clf_features"])

    scaled = scaler_clf.transform(clf_input)
    prediction = clf.predict(scaled)
    proba      = clf.predict_proba(scaled)

    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


@requires_artifacts
def test_regressor_predicts_positive_distance():
    import joblib
    reg        = joblib.load(os.path.join(ARTIFACT_DIR, "regressor.joblib"))
    scaler_reg = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_reg.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_names.json")) as f:
        names = json.load(f)

    feats = feature_engineer(0.12, 0.27, 48000.0, 22.1, 14_500_000.0)
    reg_input = pd.DataFrame([[
        feats["log_diameter_avg"], feats["log_diameter_ratio"],
        feats["log_relative_velocity"], feats["absolute_magnitude"],
        feats["diameter_avg"], feats["diameter_ratio"],
    ]], columns=names["reg_features"])

    scaled           = scaler_reg.transform(reg_input)
    log_pred         = reg.predict(scaled)
    miss_distance_km = float(np.expm1(log_pred[0]))

    assert log_pred.shape == (1,)
    assert miss_distance_km > 0


@requires_artifacts
def test_model_metadata_keys():
    with open(os.path.join(ARTIFACT_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    for key in ("trained_at", "clf_params", "reg_params"):
        assert key in meta, f"Missing key in model_metadata.json: {key}"
