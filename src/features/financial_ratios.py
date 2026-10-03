"""
Fundamental Ratio & Composite Distress Signal Feature Engineering Engine for GCCR.
Computes core leverage, coverage, liquidity, cash flow solvency, and Altman Z-Score.
"""

import pandas as pd
import numpy as np


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes corporate leverage, coverage, liquidity ratios, and Altman Z-Score.
    
    Ratios:
    - Interest Coverage Ratio (ICR): ebit / |interest_exp|
    - Debt-to-EBITDA: total_debt / ebitda
    - Current Ratio: current_assets / current_liab
    - Operating Cash Flow to Total Debt (OCF-to-Debt): operating_cf / total_debt
    - Operating Margin: ebit / revenue
    - Altman Z-Score: 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
    """
    data = df.copy()
    
    # Coverage & Leverage Ratios
    denom_interest = np.maximum(data['interest_exp'].abs(), 1e-5)
    data['icr'] = data['ebit'] / denom_interest
    
    denom_ebitda = np.maximum(data['ebitda'], 1e-5)
    data['debt_to_ebitda'] = data['total_debt'] / denom_ebitda
    
    # Liquidity Ratios
    denom_curr_liab = np.maximum(data['current_liab'], 1e-5)
    data['current_ratio'] = data['current_assets'] / denom_curr_liab
    
    # Cash Flow Solvency
    denom_debt = np.maximum(data['total_debt'], 1e-5)
    data['ocf_to_debt'] = data['operating_cf'] / denom_debt
    
    # Profitability Ratios
    denom_revenue = np.maximum(data['revenue'], 1e-5)
    data['operating_margin'] = data['ebit'] / denom_revenue
    
    # Altman Z-Score Calculation
    denom_assets = np.maximum(data['total_assets'], 1e-5)
    denom_liab = np.maximum(data['total_liabilities'], 1e-5)
    
    x1 = (data['current_assets'] - data['current_liab']) / denom_assets
    x2 = data['retained_earning'] / denom_assets
    x3 = data['ebit'] / denom_assets
    x4 = data['market_cap'] / denom_liab
    x5 = data['revenue'] / denom_assets
    
    data['altman_z'] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    
    return data
