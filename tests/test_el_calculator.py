"""
Unit tests for Basel III Capital Requirement (K), RWA, and Expected/Unexpected Loss Engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.risk.el_calculator import (
    compute_basel_correlation,
    compute_basel_maturity_adjustment,
    compute_basel_capital_requirement,
    calculate_expected_loss,
    aggregate_portfolio_metrics
)


def test_basel_correlation_bounds():
    # Basel correlation R must be between 0.12 and 0.24
    low_pd_r = compute_basel_correlation(0.0005)
    high_pd_r = compute_basel_correlation(0.20)
    
    assert 0.12 <= low_pd_r <= 0.24
    assert 0.12 <= high_pd_r <= 0.24
    # Lower PD should have higher asset correlation under Basel formula
    assert low_pd_r > high_pd_r


def test_basel_maturity_adjustment():
    # Longer maturity must increase maturity adjustment
    pd_val = 0.01
    adj_1yr = compute_basel_maturity_adjustment(pd_val, maturity=1.0)
    adj_25yr = compute_basel_maturity_adjustment(pd_val, maturity=2.5)
    adj_5yr = compute_basel_maturity_adjustment(pd_val, maturity=5.0)
    
    assert adj_1yr < adj_25yr < adj_5yr
    assert adj_1yr > 0


def test_basel_capital_requirement_and_rwa():
    pd_val = 0.01   # 1% PD
    lgd_val = 0.45  # 45% LGD
    k = compute_basel_capital_requirement(pd_val, lgd_val, maturity=2.5)
    
    # K should be a reasonable percentage (typically 5% - 10% for 1% PD)
    assert 0.05 < k < 0.12
    # RWA = K * 12.5 should be between 60% and 150% of EAD
    rwa_weight = k * 12.5
    assert 0.60 < rwa_weight < 1.50


def test_calculate_expected_and_unexpected_loss():
    df = pd.DataFrame([
        {"cik": "0000000001", "pd_calibrated": 0.01, "lgd": 0.45, "ead": 100.0, "credit_limit": 120.0},
        {"cik": "0000000002", "pd_calibrated": 0.05, "lgd": 0.45, "ead": 50.0, "credit_limit": 60.0},
    ])
    
    result = calculate_expected_loss(df, exposures_path=None)
    
    # Check expected loss
    assert np.isclose(result.loc[0, 'expected_loss'], 0.01 * 0.45 * 100.0)
    assert np.isclose(result.loc[1, 'expected_loss'], 0.05 * 0.45 * 50.0)
    
    # Check Basel fields
    for col in ['capital_requirement_k', 'unexpected_loss', 'rwa', 'regulatory_capital', 'total_loss_999']:
        assert col in result.columns
        assert (result[col] >= 0).all()
        
    # Check RWA = K * 12.5 * EAD
    expected_rwa_0 = result.loc[0, 'capital_requirement_k'] * 12.5 * 100.0
    assert np.isclose(result.loc[0, 'rwa'], expected_rwa_0)
    
    # Check Regulatory Capital (8%) = Unexpected Loss
    assert np.isclose(result.loc[0, 'regulatory_capital'], result.loc[0, 'unexpected_loss'])
    assert np.isclose(result.loc[0, 'total_loss_999'], result.loc[0, 'expected_loss'] + result.loc[0, 'unexpected_loss'])


def test_aggregate_portfolio_metrics_with_basel():
    df = pd.DataFrame([
        {"cik": "0000000001", "pd_calibrated": 0.01, "lgd": 0.45, "ead": 100.0, "credit_limit": 120.0},
        {"cik": "0000000002", "pd_calibrated": 0.02, "lgd": 0.40, "ead": 100.0, "credit_limit": 120.0},
    ])
    df_calc = calculate_expected_loss(df, exposures_path=None)
    summary = aggregate_portfolio_metrics(df_calc)
    
    assert summary['total_ead'] == 200.0
    assert summary['total_expected_loss'] > 0
    assert summary['total_unexpected_loss'] > 0
    assert summary['total_rwa'] > 0
    assert summary['capital_density_pct'] > 0
