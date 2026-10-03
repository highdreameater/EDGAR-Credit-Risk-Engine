"""
Unit tests for fundamental financial ratios and Altman Z-score engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.features.financial_ratios import compute_financial_ratios


@pytest.fixture
def sample_financials():
    return pd.DataFrame([{
        "cik": "0000320193",
        "company_name": "Test Tech Corp",
        "sector": "Technology",
        "fiscal_year": 2023,
        "total_assets": 10000.0,
        "total_liabilities": 6000.0,
        "current_assets": 4000.0,
        "current_liab": 2000.0,
        "total_debt": 3000.0,
        "revenue": 12000.0,
        "ebit": 2400.0,
        "ebitda": 3000.0,
        "interest_exp": 120.0,
        "operating_cf": 2800.0,
        "retained_earning": 3500.0,
        "market_cap": 25000.0
    }])


def test_coverage_and_leverage_ratios(sample_financials):
    res = compute_financial_ratios(sample_financials)
    
    # ICR = 2400 / 120 = 20.0
    assert np.isclose(res['icr'].iloc[0], 20.0, atol=1e-3)
    
    # Debt-to-EBITDA = 3000 / 3000 = 1.0
    assert np.isclose(res['debt_to_ebitda'].iloc[0], 1.0, atol=1e-3)
    
    # Current Ratio = 4000 / 2000 = 2.0
    assert np.isclose(res['current_ratio'].iloc[0], 2.0, atol=1e-3)
    
    # OCF-to-Debt = 2800 / 3000 = 0.9333
    assert np.isclose(res['ocf_to_debt'].iloc[0], 2800.0 / 3000.0, atol=1e-3)
    
    # Operating Margin = 2400 / 12000 = 0.20
    assert np.isclose(res['operating_margin'].iloc[0], 0.20, atol=1e-3)


def test_altman_z_score(sample_financials):
    res = compute_financial_ratios(sample_financials)
    row = sample_financials.iloc[0]
    
    x1 = (row['current_assets'] - row['current_liab']) / row['total_assets'] # (4000 - 2000) / 10000 = 0.2
    x2 = row['retained_earning'] / row['total_assets']                         # 3500 / 10000 = 0.35
    x3 = row['ebit'] / row['total_assets']                                     # 2400 / 10000 = 0.24
    x4 = row['market_cap'] / row['total_liabilities']                          # 25000 / 6000 = 4.1667
    x5 = row['revenue'] / row['total_assets']                                  # 12000 / 10000 = 1.2
    
    expected_z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    assert np.isclose(res['altman_z'].iloc[0], expected_z, atol=1e-3)


def test_zero_division_guard():
    zero_df = pd.DataFrame([{
        "cik": "0000000001",
        "company_name": "Zero Metrics LLC",
        "sector": "Industrials",
        "fiscal_year": 2023,
        "total_assets": 0.0,
        "total_liabilities": 0.0,
        "current_assets": 0.0,
        "current_liab": 0.0,
        "total_debt": 0.0,
        "revenue": 0.0,
        "ebit": 0.0,
        "ebitda": 0.0,
        "interest_exp": 0.0,
        "operating_cf": 0.0,
        "retained_earning": 0.0,
        "market_cap": 0.0
    }])
    
    res = compute_financial_ratios(zero_df)
    # Should not raise ZeroDivisionError and should not produce inf/-inf
    assert not np.isinf(res['icr'].iloc[0])
    assert not np.isinf(res['debt_to_ebitda'].iloc[0])
    assert not np.isinf(res['current_ratio'].iloc[0])
    assert not np.isinf(res['altman_z'].iloc[0])
