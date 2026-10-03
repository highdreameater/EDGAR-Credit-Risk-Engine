"""
Internal Master Scale Rating Engine for GCCR.
Maps continuous default probabilities (PD) into 8 standardized master scale rating bands
ranging from Minimal Risk (AAA) to Vulnerable/Default (CCC), categorizing into IG vs HY tiers.
"""

import os
from typing import Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd


DEFAULT_RATING_BANDS = {
    'AAA': {'min_pd': 0.0000, 'max_pd': 0.0005, 'risk_category': 'Minimal Risk', 'grade_type': 'Investment Grade'},
    'AA':  {'min_pd': 0.0005, 'max_pd': 0.0015, 'risk_category': 'Low Risk', 'grade_type': 'Investment Grade'},
    'A':   {'min_pd': 0.0015, 'max_pd': 0.0050, 'risk_category': 'Low-Moderate Risk', 'grade_type': 'Investment Grade'},
    'BBB': {'min_pd': 0.0050, 'max_pd': 0.0200, 'risk_category': 'Moderate Risk', 'grade_type': 'Investment Grade'},
    'BB':  {'min_pd': 0.0200, 'max_pd': 0.0600, 'risk_category': 'Sub-Investment Grade', 'grade_type': 'High Yield'},
    'B':   {'min_pd': 0.0600, 'max_pd': 0.1500, 'risk_category': 'High Risk', 'grade_type': 'High Yield'},
    'CCC': {'min_pd': 0.1500, 'max_pd': 1.0000, 'risk_category': 'Vulnerable / Default Imminent', 'grade_type': 'High Yield'}
}

RATING_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']


def load_rating_bands(config_path: str = "config/rating_bands.yaml") -> Dict[str, Any]:
    """Loads rating scale definitions from YAML configuration."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'rating_scale' in cfg:
                return cfg['rating_scale']
    return DEFAULT_RATING_BANDS


def pd_to_rating(pd_val: float, rating_scale: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Maps a single PD scalar to its internal rating band and classification."""
    if rating_scale is None:
        rating_scale = DEFAULT_RATING_BANDS

    val = float(np.clip(pd_val, 0.0, 1.0))
    for rating, info in rating_scale.items():
        min_p = info.get('min_pd', 0.0)
        max_p = info.get('max_pd', 1.0)
        # Check boundary
        if rating == 'CCC':
            if val >= min_p:
                return {
                    'rating': rating,
                    'risk_category': info.get('risk_category', 'High Risk'),
                    'grade_type': info.get('grade_type', 'High Yield')
                }
        else:
            if min_p <= val < max_p:
                return {
                    'rating': rating,
                    'risk_category': info.get('risk_category', 'Standard'),
                    'grade_type': info.get('grade_type', 'Investment Grade' if min_p < 0.02 else 'High Yield')
                }

    return {
        'rating': 'CCC',
        'risk_category': 'Vulnerable / Default Imminent',
        'grade_type': 'High Yield'
    }


def assign_master_scale_ratings(
    df: pd.DataFrame, 
    pd_col: str = "pd_calibrated", 
    config_path: str = "config/rating_bands.yaml"
) -> pd.DataFrame:
    """
    Maps calibrated PD values in the DataFrame to master scale rating bands,
    risk categories, and grade types (Investment Grade vs High Yield).
    """
    data = df.copy()
    rating_scale = load_rating_bands(config_path)

    ratings = []
    risk_cats = []
    grade_types = []

    for pd_val in data[pd_col]:
        res = pd_to_rating(pd_val, rating_scale)
        ratings.append(res['rating'])
        risk_cats.append(res['risk_category'])
        grade_types.append(res['grade_type'])

    data['rating'] = ratings
    data['risk_category'] = risk_cats
    data['grade_type'] = grade_types

    return data
