"""
Portfolio Expected Loss (EL) & Basel III Capital/RWA Calculator for GCCR.
Implements wholesale credit risk formulations:
1. Expected Loss: EL_i = PD_i * LGD_i * EAD_i
2. Basel III / IV Advanced Internal Ratings-Based (A-IRB) Capital Requirement (K):
   - Asset Correlation (R): Exponential interpolation [0.12, 0.24]
   - Maturity Adjustment b(PD) & Factor: (1 + (M - 2.5)*b) / (1 - 1.5*b)
   - Capital Requirement (K): [LGD * N((G(PD) + sqrt(R)*G(0.999))/sqrt(1-R)) - PD*LGD] * MatAdj
   - Unexpected Loss (UL) / Economic Capital: UL_i = K_i * EAD_i
   - Risk-Weighted Assets: RWA_i = K_i * 12.5 * EAD_i
   - Regulatory Minimum Capital (8%): Capital_i = 0.08 * RWA_i = UL_i
"""

import os
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


BASEL_PD_FLOOR = 0.0003  # 3 bps Basel III regulatory floor
BASEL_PD_CEILING = 0.9999
BASEL_DEFAULT_MATURITY = 2.5  # Standard wholesale effective maturity (years)
BASEL_CONFIDENCE_QUANTILE = 0.999  # 99.9% 1-year solvency confidence level


def compute_basel_correlation(pd_vals: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
    """
    Computes Basel III corporate asset correlation R:
    R = 0.12 * (1 - exp(-50 * PD)) / (1 - exp(-50)) + 0.24 * [1 - (1 - exp(-50 * PD)) / (1 - exp(-50))]
    
    R ranges between 0.12 for high PD to 0.24 for low PD.
    """
    pd_arr = np.asarray(pd_vals, dtype=float)
    pd_clamped = np.clip(pd_arr, BASEL_PD_FLOOR, BASEL_PD_CEILING)
    
    exp_factor = (1.0 - np.exp(-50.0 * pd_clamped)) / (1.0 - np.exp(-50.0))
    r = 0.12 * exp_factor + 0.24 * (1.0 - exp_factor)
    
    if np.isscalar(pd_vals):
        return float(r)
    return r


def compute_basel_maturity_adjustment(
    pd_vals: Union[float, np.ndarray, pd.Series], 
    maturity: float = BASEL_DEFAULT_MATURITY
) -> Union[float, np.ndarray]:
    """
    Computes Basel III maturity adjustment factor:
    b(PD) = (0.11852 - 0.05478 * ln(PD))^2
    Maturity Adjustment = (1 + (M - 2.5) * b) / (1 - 1.5 * b)
    """
    pd_arr = np.asarray(pd_vals, dtype=float)
    pd_clamped = np.clip(pd_arr, BASEL_PD_FLOOR, BASEL_PD_CEILING)
    
    b = (0.11852 - 0.05478 * np.log(pd_clamped)) ** 2
    denom = np.maximum(1.0 - 1.5 * b, 1e-4)
    mat_adj = (1.0 + (maturity - 2.5) * b) / denom
    
    if np.isscalar(pd_vals):
        return float(mat_adj)
    return mat_adj


def compute_basel_capital_requirement(
    pd_vals: Union[float, np.ndarray, pd.Series],
    lgd_vals: Union[float, np.ndarray, pd.Series],
    maturity: float = BASEL_DEFAULT_MATURITY
) -> Union[float, np.ndarray]:
    r"""
    Computes Basel III capital requirement K per unit of exposure ($K \in [0, 1]$):
    K = [LGD * N((N^-1(PD) + sqrt(R)*N^-1(0.999)) / sqrt(1 - R)) - (PD * LGD)] * MaturityAdjustment
    """
    pd_arr = np.asarray(pd_vals, dtype=float)
    lgd_arr = np.asarray(lgd_vals, dtype=float)
    
    pd_clamped = np.clip(pd_arr, BASEL_PD_FLOOR, BASEL_PD_CEILING)
    lgd_clamped = np.clip(lgd_arr, 0.0, 1.0)
    
    r = compute_basel_correlation(pd_clamped)
    mat_adj = compute_basel_maturity_adjustment(pd_clamped, maturity=maturity)
    
    g_0999 = norm.ppf(BASEL_CONFIDENCE_QUANTILE)
    g_pd = norm.ppf(pd_clamped)
    
    conditional_pdr = norm.cdf((g_pd + np.sqrt(r) * g_0999) / np.sqrt(1.0 - r))
    k = (lgd_clamped * conditional_pdr - (pd_clamped * lgd_clamped)) * mat_adj
    k_clamped = np.maximum(0.0, k)
    
    if np.isscalar(pd_vals) and np.isscalar(lgd_vals):
        return float(k_clamped)
    return k_clamped


def calculate_expected_loss(
    df: pd.DataFrame, 
    exposures_path: Optional[str] = "data/portfolio/exposures_limits.csv",
    pd_col: str = "pd_calibrated",
    maturity: float = BASEL_DEFAULT_MATURITY
) -> pd.DataFrame:
    """
    Computes wholesale credit risk parameters:
    1. Counterparty Expected Loss (EL):
       EL_i = PD_i * LGD_i * EAD_i
    2. Basel III Capital Requirement (K)
    3. Unexpected Loss (UL) / Economic Capital:
       UL_i = K_i * EAD_i
    4. Risk-Weighted Assets (RWA):
       RWA_i = K_i * 12.5 * EAD_i
    5. Regulatory Minimum Capital (8% of RWA = UL):
       Regulatory Capital_i = 0.08 * RWA_i
    6. Total Loss at 99.9% VaR:
       Total Loss = EL + UL
       
    If exposures_path is provided and columns 'ead', 'lgd', 'credit_limit'
    are not already present, loads and merges exposure parameters by CIK.
    """
    data = df.copy()
    if 'cik' in data.columns:
        data['cik'] = data['cik'].astype(str).str.zfill(10)

    # Check if exposure parameters need to be merged
    needs_merge = any(col not in data.columns for col in ['ead', 'lgd', 'credit_limit'])
    
    if needs_merge and exposures_path and os.path.exists(exposures_path):
        exp_df = pd.read_csv(exposures_path, dtype={'cik': str})
        exp_df['cik'] = exp_df['cik'].astype(str).str.zfill(10)
        
        # Merge on CIK, dropping duplicates in exposure table if any
        exp_df = exp_df.drop_duplicates(subset=['cik'])
        
        # Keep track of existing columns to avoid suffix collisions
        cols_to_use = ['cik'] + [c for c in ['ead', 'lgd', 'credit_limit'] if c in exp_df.columns]
        data = pd.merge(data, exp_df[cols_to_use], on='cik', how='left')

    # Fill defaults for any missing exposure fields
    if 'ead' not in data.columns:
        data['ead'] = 50.0
    else:
        data['ead'] = data['ead'].fillna(50.0)

    if 'lgd' not in data.columns:
        data['lgd'] = 0.45
    else:
        data['lgd'] = data['lgd'].fillna(0.45)

    if 'credit_limit' not in data.columns:
        data['credit_limit'] = data['ead'] * 1.25
    else:
        data['credit_limit'] = data['credit_limit'].fillna(data['ead'] * 1.25)

    # Expected Loss calculation: EL = PD * LGD * EAD
    data['expected_loss'] = data[pd_col] * data['lgd'] * data['ead']

    # Basel III A-IRB Capital Calculations
    k_vals = compute_basel_capital_requirement(data[pd_col].values, data['lgd'].values, maturity=maturity)
    data['capital_requirement_k'] = k_vals
    data['unexpected_loss'] = k_vals * data['ead']
    data['rwa'] = k_vals * 12.5 * data['ead']
    data['regulatory_capital'] = 0.08 * data['rwa']
    data['total_loss_999'] = data['expected_loss'] + data['unexpected_loss']

    return data


def aggregate_portfolio_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregates top-level portfolio credit statistics:
    Total EAD, Total Expected Loss, Unexpected Loss (Capital), Total RWA,
    Capital Density (RWA/EAD %), Weighted Average PD, Weighted Average LGD.
    """
    total_ead = float(df['ead'].sum())
    total_el = float(df['expected_loss'].sum())
    
    total_ul = float(df['unexpected_loss'].sum()) if 'unexpected_loss' in df.columns else 0.0
    total_rwa = float(df['rwa'].sum()) if 'rwa' in df.columns else 0.0
    total_reg_cap = float(df['regulatory_capital'].sum()) if 'regulatory_capital' in df.columns else 0.0
    
    capital_density = (total_rwa / total_ead * 100.0) if total_ead > 0 else 0.0
    
    if total_ead > 0:
        weighted_pd = float((df['pd_calibrated'] * df['ead']).sum() / total_ead)
        weighted_lgd = float((df['lgd'] * df['ead']).sum() / total_ead)
    else:
        weighted_pd = float(df['pd_calibrated'].mean())
        weighted_lgd = float(df['lgd'].mean())

    return {
        "total_ead": total_ead,
        "total_expected_loss": total_el,
        "total_unexpected_loss": total_ul,
        "total_rwa": total_rwa,
        "total_regulatory_capital": total_reg_cap,
        "capital_density_pct": capital_density,
        "weighted_avg_pd": weighted_pd,
        "weighted_avg_lgd": weighted_lgd,
        "counterparty_count": len(df)
    }
