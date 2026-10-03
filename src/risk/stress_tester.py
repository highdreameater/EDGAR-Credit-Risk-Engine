"""
Portfolio Macroeconomic Stress-Testing Engine for GCCR (RMG Mandate).
Applies macro shock scenario multipliers (e.g. Severe Global Recession, Rate Hike Cycle)
to baseline PD and LGD figures across corporate sectors, computing stressed portfolio loss deltas
and post-stress rating migration downgrade matrices.
"""

import os
from typing import Dict, Any, List, Optional
import yaml
import numpy as np
import pandas as pd
from .rating_engine import assign_master_scale_ratings, RATING_ORDER


def run_macro_stress_test(
    risk_df: pd.DataFrame, 
    scenario_config_path: str = "config/stress_scenarios.yaml"
) -> pd.DataFrame:
    """
    Applies macro shock multipliers to portfolio PD and LGD by corporate sector.
    Returns portfolio-level Expected Loss comparison table.
    """
    if not os.path.exists(scenario_config_path):
        raise FileNotFoundError(f"Scenario config not found at: {scenario_config_path}")

    with open(scenario_config_path, 'r') as f:
        scenarios = yaml.safe_load(f)['scenarios']
        
    results = []
    base_el = (risk_df['pd_calibrated'] * risk_df['lgd'] * risk_df['ead']).sum()
    
    results.append({
        'scenario': 'Baseline',
        'portfolio_el': float(base_el),
        'delta_el': 0.0,
        'pct_increase': 0.0
    })
    
    for key, spec in scenarios.items():
        stressed_df = risk_df.copy()
        
        # Apply Sector PD Multiplier
        multipliers = stressed_df['sector'].map(spec.get('pd_multipliers', {})).fillna(1.0)
        stressed_df['pd_stressed'] = np.minimum(1.0, stressed_df['pd_calibrated'] * multipliers)
        stressed_df['lgd_stressed'] = np.minimum(1.0, stressed_df['lgd'] + spec.get('lgd_shift', 0.0))
        
        # Stressed Expected Loss
        stressed_el = (stressed_df['pd_stressed'] * stressed_df['lgd_stressed'] * stressed_df['ead']).sum()
        delta = stressed_el - base_el
        pct_increase = (delta / base_el) * 100.0 if base_el > 0 else 0.0
        
        results.append({
            'scenario': spec.get('name', key),
            'portfolio_el': float(stressed_el),
            'delta_el': float(delta),
            'pct_increase': float(pct_increase)
        })
        
    return pd.DataFrame(results)


def apply_scenario_to_portfolio(
    risk_df: pd.DataFrame, 
    scenario_spec: Dict[str, Any],
    rating_config_path: str = "config/rating_bands.yaml"
) -> pd.DataFrame:
    """
    Applies a specific macro stress scenario at counterparty facility level.
    Computes counterparty-level stressed PD, LGD, EL, and post-stress rating migrations.
    """
    stressed_df = risk_df.copy()
    
    # Sector PD Multiplier
    multipliers = stressed_df['sector'].map(scenario_spec.get('pd_multipliers', {})).fillna(1.0)
    stressed_df['pd_stressed'] = np.minimum(0.9999, stressed_df['pd_calibrated'] * multipliers)
    stressed_df['lgd_stressed'] = np.minimum(1.0, stressed_df['lgd'] + scenario_spec.get('lgd_shift', 0.0))
    
    # Calculate Stressed EL
    stressed_df['el_stressed'] = stressed_df['pd_stressed'] * stressed_df['lgd_stressed'] * stressed_df['ead']
    stressed_df['delta_el'] = stressed_df['el_stressed'] - stressed_df['expected_loss']

    # Assign Stressed Rating
    temp_rated = assign_master_scale_ratings(
        stressed_df[['pd_stressed']].rename(columns={'pd_stressed': 'pd_calibrated'}),
        pd_col='pd_calibrated',
        config_path=rating_config_path
    )
    stressed_df['stressed_rating'] = temp_rated['rating']
    stressed_df['stressed_grade_type'] = temp_rated['grade_type']

    # Calculate Downgrade Notches
    order_map = {rating: i for i, rating in enumerate(RATING_ORDER)}
    base_idx = stressed_df['rating'].map(order_map).fillna(0)
    stress_idx = stressed_df['stressed_rating'].map(order_map).fillna(0)
    stressed_df['downgrade_notches'] = (stress_idx - base_idx).clip(lower=0)

    return stressed_df


def compute_rating_migration_matrix(
    baseline_ratings: pd.Series, 
    stressed_ratings: pd.Series
) -> pd.DataFrame:
    """
    Constructs a rating transition/migration matrix from baseline to stressed states.
    Rows: Baseline Rating. Columns: Stressed Rating.
    Values: Percentage of counterparties transitioning.
    """
    matrix = pd.crosstab(
        baseline_ratings, 
        stressed_ratings, 
        rownames=['Baseline Rating'], 
        colnames=['Stressed Rating']
    )
    
    # Reindex to standard master scale
    all_ratings = [r for r in RATING_ORDER if r in matrix.index or r in matrix.columns]
    matrix = matrix.reindex(index=RATING_ORDER, columns=RATING_ORDER, fill_value=0)
    
    # Normalize row-wise to percentage
    row_sums = matrix.sum(axis=1)
    matrix_pct = matrix.div(row_sums.replace(0, 1), axis=0) * 100.0
    return matrix_pct.round(1)
