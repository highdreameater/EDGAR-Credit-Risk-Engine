"""
Unit tests for Credit Limit Monitoring & Breach Detection Engine (CEM Mandate).
"""

import pytest
import numpy as np
import pandas as pd
from src.risk.limit_monitor import evaluate_limit_utilization, get_breach_summary


@pytest.fixture
def exposure_data():
    return pd.DataFrame([
        {"cik": "0000000001", "ead": 70.0, "credit_limit": 100.0},   # 70% -> Normal
        {"cik": "0000000002", "ead": 90.0, "credit_limit": 100.0},   # 90% -> Warning
        {"cik": "0000000003", "ead": 125.0, "credit_limit": 100.0},  # 125% -> BREACH
        {"cik": "0000000004", "ead": 85.0, "credit_limit": 100.0},   # 85% -> Warning
        {"cik": "0000000005", "ead": 100.0, "credit_limit": 100.0},  # 100% -> Warning
    ])


def test_limit_utilization_classification(exposure_data):
    res = evaluate_limit_utilization(exposure_data)
    
    assert res.loc[0, 'status'] == 'Normal'
    assert res.loc[0, 'excess_exposure'] == 0.0
    
    assert res.loc[1, 'status'] == 'Warning'
    assert res.loc[1, 'excess_exposure'] == 0.0
    
    assert res.loc[2, 'status'] == 'BREACH'
    assert np.isclose(res.loc[2, 'excess_exposure'], 25.0)
    
    assert res.loc[3, 'status'] == 'Warning'
    assert res.loc[4, 'status'] == 'Warning'


def test_breach_summary_counts(exposure_data):
    res = evaluate_limit_utilization(exposure_data)
    summary = get_breach_summary(res)
    
    assert summary['total_counterparties'] == 5
    assert summary['breach_count'] == 1
    assert summary['warning_count'] == 3
    assert summary['normal_count'] == 1
    assert np.isclose(summary['total_excess_exposure'], 25.0)
