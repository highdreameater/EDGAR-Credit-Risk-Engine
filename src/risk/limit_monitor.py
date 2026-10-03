"""
Credit Limit Monitoring & Breach Detection Engine for GCCR (CEM Mandate).
Computes real-time facility limit utilization percentages, classifies exposure states
(Normal, Warning, BREACH), and monitors excess exposure and sector concentrations.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def evaluate_limit_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes credit limit utilization percentage and tags exposure breach warnings.
    
    Status Flags:
    - Normal: Utilization < 85%
    - Warning: 85% <= Utilization <= 100%
    - BREACH: Utilization > 100%
    """
    data = df.copy()
    
    # Calculate Utilization %
    denom_limit = np.maximum(data['credit_limit'], 1e-5)
    data['utilization_pct'] = (data['ead'] / denom_limit) * 100.0
    
    # Assign Status Classification
    conditions = [
        data['utilization_pct'] < 85.0,
        (data['utilization_pct'] >= 85.0) & (data['utilization_pct'] <= 100.0),
        data['utilization_pct'] > 100.0
    ]
    choices = ['Normal', 'Warning', 'BREACH']
    
    data['status'] = np.select(conditions, choices, default='Unknown')
    data['excess_exposure'] = np.maximum(0.0, data['ead'] - data['credit_limit'])
    
    return data


def get_breach_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns summary metrics of limit utilization and breach flags across the portfolio.
    """
    breaches = df[df['status'] == 'BREACH']
    warnings = df[df['status'] == 'Warning']
    normal = df[df['status'] == 'Normal']

    return {
        "total_counterparties": len(df),
        "breach_count": len(breaches),
        "warning_count": len(warnings),
        "normal_count": len(normal),
        "total_excess_exposure": float(breaches['excess_exposure'].sum()),
        "max_utilization_pct": float(df['utilization_pct'].max()) if len(df) > 0 else 0.0,
        "avg_utilization_pct": float(df['utilization_pct'].mean()) if len(df) > 0 else 0.0
    }


def get_sector_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates sector-level exposure concentration, average utilization, and breach counts.
    """
    grouped = df.groupby('sector').agg(
        counterparty_count=('cik', 'count'),
        total_ead=('ead', 'sum'),
        total_credit_limit=('credit_limit', 'sum'),
        total_expected_loss=('expected_loss', 'sum'),
        breach_count=('status', lambda s: (s == 'BREACH').sum()),
        warning_count=('status', lambda s: (s == 'Warning').sum())
    ).reset_index()

    grouped['ead_share_pct'] = (grouped['total_ead'] / grouped['total_ead'].sum()) * 100.0
    grouped['sector_utilization_pct'] = (grouped['total_ead'] / grouped['total_credit_limit']) * 100.0

    return grouped.sort_values(by='total_ead', ascending=False)
