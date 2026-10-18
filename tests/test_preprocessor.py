"""
Unit tests for Financial Data Preprocessor & Winsorization Module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import preprocess_financial_data, winsorize_series


def test_winsorize_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, -500.0])
    w = winsorize_series(s, lower_quantile=0.10, upper_quantile=0.90)
    
    assert w.max() < 1000.0
    assert w.min() > -500.0


def test_preprocess_financial_data_imputation_and_infs():
    df = pd.DataFrame({
        "cik": ["1", "2", "3", "4"],
        "sector": ["Tech", "Tech", "Energy", "Energy"],
        "total_assets": [100.0, np.nan, 200.0, 300.0],
        "total_debt": [50.0, np.inf, -np.inf, 120.0],
        "revenue": [80.0, 120.0, np.nan, 250.0]
    })
    
    clean = preprocess_financial_data(df, winsorize=True)
    
    # Assert no NaNs or Infs remain
    for col in ["total_assets", "total_debt", "revenue"]:
        assert not clean[col].isna().any()
        assert not np.isinf(clean[col]).any()
        
    # Sector median imputation check: row 1 (Tech) should be filled with Tech's 100.0
    assert clean.loc[1, "total_assets"] == 100.0
