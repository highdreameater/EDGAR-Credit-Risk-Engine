"""
Unit tests for Probability of Default (PD) Modeling, Platt Calibration, and RMG Validation.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import (
    compute_rmg_validation_metrics,
    calculate_psi,
    generate_roc_curve_data,
    generate_ks_curve_data
)


@pytest.fixture
def synthetic_training_data():
    np.random.seed(42)
    n = 200
    altman_z = np.random.normal(2.5, 1.2, n)
    icr = np.random.exponential(5.0, n)
    debt_to_ebitda = np.random.gamma(2.0, 2.0, n)
    current_ratio = np.random.normal(1.8, 0.6, n)
    ocf_to_debt = np.random.normal(0.25, 0.15, n)
    operating_margin = np.random.normal(0.14, 0.08, n)

    # Risk score: higher debt/low z increases default probability
    latent_score = -0.8 * altman_z + 0.3 * debt_to_ebitda - 0.2 * icr
    prob = 1.0 / (1.0 + np.exp(-latent_score))
    target = (np.random.rand(n) < prob).astype(int)

    # Ensure both classes exist
    if target.sum() == 0:
        target[0:10] = 1

    df = pd.DataFrame({
        "cik": [f"0000{i:06d}" for i in range(n)],
        "altman_z": altman_z,
        "icr": icr,
        "debt_to_ebitda": debt_to_ebitda,
        "current_ratio": current_ratio,
        "ocf_to_debt": ocf_to_debt,
        "operating_margin": operating_margin,
        "target": target
    })
    return df


def test_pd_training_and_calibration_output(synthetic_training_data):
    df_scored, model = train_and_calibrate_pd_model(synthetic_training_data)
    
    assert "pd_calibrated" in df_scored.columns
    assert "pd_uncalibrated" in df_scored.columns
    assert len(df_scored) == len(synthetic_training_data)
    
    # Check bounds
    assert (df_scored['pd_calibrated'] >= 0.0).all()
    assert (df_scored['pd_calibrated'] <= 1.0).all()
    assert (df_scored['pd_uncalibrated'] >= 0.0).all()
    assert (df_scored['pd_uncalibrated'] <= 1.0).all()


def test_rmg_validation_metrics(synthetic_training_data):
    df_scored, _ = train_and_calibrate_pd_model(synthetic_training_data)
    metrics = compute_rmg_validation_metrics(df_scored['target'], df_scored['pd_calibrated'])
    
    assert "roc_auc" in metrics
    assert "gini" in metrics
    assert "ks_stat_pct" in metrics
    assert "brier_score" in metrics
    
    # Mathematical sanity bounds
    assert 0.0 <= metrics['roc_auc'] <= 1.0
    assert -1.0 <= metrics['gini'] <= 1.0
    assert np.isclose(metrics['gini'], 2.0 * metrics['roc_auc'] - 1.0, atol=1e-5)
    assert 0.0 <= metrics['ks_stat_pct'] <= 100.0
    assert 0.0 <= metrics['brier_score'] <= 1.0


def test_psi_calculation():
    np.random.seed(42)
    baseline_p = np.random.beta(2, 8, 1000)
    # Similar distribution -> low PSI
    actual_p_similar = np.random.beta(2, 8, 1000)
    psi_low = calculate_psi(baseline_p, actual_p_similar, num_bins=10)
    assert psi_low < 0.10  # Stable

    # Shifted distribution -> high PSI
    actual_p_shifted = np.random.beta(5, 5, 1000)
    psi_high = calculate_psi(baseline_p, actual_p_shifted, num_bins=10)
    assert psi_high > psi_low


def test_scorecard_waterfall_explainability(synthetic_training_data):
    _, model = train_and_calibrate_pd_model(synthetic_training_data)
    sample = synthetic_training_data.iloc[0]
    
    explanation = model.explain_counterparty_score(sample)
    
    assert "intercept" in explanation
    assert "feature_contributions" in explanation
    assert "calibrated_pd" in explanation
    assert len(explanation["feature_contributions"]) == len(model.feature_names)
    
    for item in explanation["feature_contributions"]:
        assert "feature" in item
        assert "contribution" in item
        assert "risk_impact" in item


def test_model_save_and_load(tmp_path, synthetic_training_data):
    _, model = train_and_calibrate_pd_model(synthetic_training_data)
    model_path = str(tmp_path / "model.joblib")
    
    model.save(model_path)
    loaded_model = model.load(model_path)
    
    sample = synthetic_training_data.iloc[0:5]
    orig_preds = model.predict_calibrated(sample)
    loaded_preds = loaded_model.predict_calibrated(sample)
    
    assert np.allclose(orig_preds, loaded_preds)

