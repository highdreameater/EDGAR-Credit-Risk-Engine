"""
Unit tests for Macro Stress-Testing & Rating Migration Engine (RMG Mandate).
"""

import pytest
import numpy as np
import pandas as pd
from src.risk.stress_tester import (
    run_macro_stress_test,
    apply_scenario_to_portfolio,
    compute_rating_migration_matrix
)


@pytest.fixture
def baseline_risk_portfolio():
    return pd.DataFrame([
        {
            "cik": "0000000001",
            "sector": "Technology",
            "rating": "A",
            "pd_calibrated": 0.0030,
            "lgd": 0.40,
            "ead": 100.0,
            "expected_loss": 0.0030 * 0.40 * 100.0
        },
        {
            "cik": "0000000002",
            "sector": "Industrials",
            "rating": "BBB",
            "pd_calibrated": 0.0150,
            "lgd": 0.45,
            "ead": 200.0,
            "expected_loss": 0.0150 * 0.45 * 200.0
        },
        {
            "cik": "0000000003",
            "sector": "Energy",
            "rating": "BB",
            "pd_calibrated": 0.0400,
            "lgd": 0.50,
            "ead": 150.0,
            "expected_loss": 0.0400 * 0.50 * 150.0
        }
    ])


def test_macro_stress_test_deltas(baseline_risk_portfolio):
    stress_results = run_macro_stress_test(
        baseline_risk_portfolio, 
        scenario_config_path="config/stress_scenarios.yaml"
    )
    
    assert len(stress_results) >= 2  # Baseline + at least 1 scenario
    assert stress_results.iloc[0]['scenario'] == 'Baseline'
    assert stress_results.iloc[0]['delta_el'] == 0.0
    
    # Stressed scenarios must have positive delta EL
    for idx in range(1, len(stress_results)):
        assert stress_results.iloc[idx]['portfolio_el'] > stress_results.iloc[0]['portfolio_el']
        assert stress_results.iloc[idx]['delta_el'] > 0.0
        assert stress_results.iloc[idx]['pct_increase'] > 0.0


def test_apply_scenario_counterparty_level(baseline_risk_portfolio):
    scenario_spec = {
        "name": "Test Shock",
        "pd_multipliers": {"Technology": 1.5, "Industrials": 2.0, "Energy": 2.5},
        "lgd_shift": 0.10
    }
    
    stressed = apply_scenario_to_portfolio(baseline_risk_portfolio, scenario_spec)
    
    assert "pd_stressed" in stressed.columns
    assert "lgd_stressed" in stressed.columns
    assert "el_stressed" in stressed.columns
    assert "stressed_rating" in stressed.columns
    assert "downgrade_notches" in stressed.columns
    
    # Technology firm: PD should be 0.0030 * 1.5 = 0.0045, LGD = 0.50
    tech_row = stressed[stressed['sector'] == 'Technology'].iloc[0]
    assert np.isclose(tech_row['pd_stressed'], 0.0045)
    assert np.isclose(tech_row['lgd_stressed'], 0.50)
    assert tech_row['el_stressed'] > tech_row['expected_loss']


def test_migration_matrix():
    baseline_ratings = pd.Series(['AAA', 'AA', 'A', 'BBB', 'BB'])
    stressed_ratings = pd.Series(['AA', 'A', 'BBB', 'BB', 'B'])  # each downgraded 1 notch
    
    matrix = compute_rating_migration_matrix(baseline_ratings, stressed_ratings)
    assert isinstance(matrix, pd.DataFrame)
    # Total row percentages should sum to 100 or 0
    for idx, row in matrix.iterrows():
        assert row.sum() in [0.0, 100.0]
