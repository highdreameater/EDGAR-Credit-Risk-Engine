"""
Portfolio Expected Loss (EL) Calculator for GCCR.
Implements wholesale credit risk formulations:
EL_i = PD_i * LGD_i * EAD_i
Merges counterparty facilities from portfolio exposures with calibrated model PDs.
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


def calculate_expected_loss(
    df: pd.DataFrame, 
    exposures_path: Optional[str] = "data/portfolio/exposures_limits.csv",
    pd_col: str = "pd_calibrated"
) -> pd.DataFrame:
    """
    Computes Counterparty Expected Loss (EL):
    EL_i = PD_i * LGD_i * EAD_i
    
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

    # Expected Loss calculation
    data['expected_loss'] = data[pd_col] * data['lgd'] * data['ead']

    return data


def aggregate_portfolio_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregates top-level portfolio credit statistics:
    Total EAD, Total Expected Loss, Weighted Average PD, Weighted Average LGD.
    """
    total_ead = float(df['ead'].sum())
    total_el = float(df['expected_loss'].sum())
    
    if total_ead > 0:
        weighted_pd = float((df['pd_calibrated'] * df['ead']).sum() / total_ead)
        weighted_lgd = float((df['lgd'] * df['ead']).sum() / total_ead)
    else:
        weighted_pd = float(df['pd_calibrated'].mean())
        weighted_lgd = float(df['lgd'].mean())

    return {
        "total_ead": total_ead,
        "total_expected_loss": total_el,
        "weighted_avg_pd": weighted_pd,
        "weighted_avg_lgd": weighted_lgd,
        "counterparty_count": len(df)
    }
