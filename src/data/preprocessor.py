"""
Financial Data Preprocessing & Data Quality Module for GCCR.
Executes sector-specific median imputation for missing items and
winsorizes extreme outlier distributions at the 1st and 99th percentiles.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def winsorize_series(
    series: pd.Series, 
    lower_quantile: float = 0.01, 
    upper_quantile: float = 0.99
) -> pd.Series:
    """
    Clips extreme values outside [lower_quantile, upper_quantile] bounds.
    """
    valid = series.dropna()
    if len(valid) == 0:
        return series
    
    q_low = float(valid.quantile(lower_quantile))
    q_high = float(valid.quantile(upper_quantile))
    return series.clip(lower=q_low, upper=q_high)


def preprocess_financial_data(
    df: pd.DataFrame, 
    numeric_cols: Optional[List[str]] = None,
    winsorize: bool = True,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> pd.DataFrame:
    """
    Preprocesses financial statements:
    1. Replaces inf/-inf with NaN.
    2. Imputes missing numeric values using sector-specific median (falling back to global median).
    3. Winsorizes outliers across numerical lines to control leverage/ratio spikes.
    """
    clean_df = df.copy()

    if numeric_cols is None:
        numeric_cols = [
            'total_assets', 'total_liabilities', 'current_assets', 'current_liab',
            'total_debt', 'revenue', 'ebit', 'ebitda', 'interest_exp',
            'operating_cf', 'retained_earning', 'market_cap'
        ]

    # Filter to existing columns
    present_numeric = [col for col in numeric_cols if col in clean_df.columns]

    # Handle infs
    clean_df[present_numeric] = clean_df[present_numeric].replace([np.inf, -np.inf], np.nan)

    # Sector-specific median imputation
    if 'sector' in clean_df.columns:
        for col in present_numeric:
            sector_medians = clean_df.groupby('sector')[col].transform('median')
            clean_df[col] = clean_df[col].fillna(sector_medians)
    
    # Fallback to global median for any remaining NaNs
    for col in present_numeric:
        col_median = clean_df[col].median()
        if pd.isna(col_median):
            col_median = 0.0
        clean_df[col] = clean_df[col].fillna(col_median)

    # Winsorization
    if winsorize:
        for col in present_numeric:
            clean_df[col] = winsorize_series(
                clean_df[col], 
                lower_quantile=lower_quantile, 
                upper_quantile=upper_quantile
            )

    return clean_df
