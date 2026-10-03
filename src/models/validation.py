"""
RMG Quantitative Model Validation Engine for GCCR.
Implements discrimination power (ROC-AUC, Gini, KS),
calibration reliability (Brier Score, Calibration Curves),
and population stability (Population Stability Index - PSI).
"""

import os
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve


def compute_rmg_validation_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes standard RMG validation metrics: ROC-AUC, Gini, KS-Stat, and Brier Score.
    
    Formulations:
    - Gini = 2 * AUC - 1
    - KS = max_t |F_Default(t) - F_NonDefault(t)| * 100%
    - Brier = (1 / N) * sum((y_prob - y_true)^2)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    
    # Handle single class edge case
    if len(np.unique(y_true_arr)) < 2:
        return {
            "roc_auc": 0.5,
            "gini": 0.0,
            "ks_stat_pct": 0.0,
            "brier_score": float(np.mean((y_prob_arr - y_true_arr) ** 2))
        }

    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    gini = 2.0 * auc - 1.0
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    
    # Calculate KS Statistic
    df_eval = pd.DataFrame({'target': y_true_arr, 'prob': y_prob_arr}).sort_values(by='prob', ascending=False)
    df_eval['default'] = df_eval['target']
    df_eval['non_default'] = 1 - df_eval['target']
    
    total_defaults = df_eval['default'].sum()
    total_non_defaults = df_eval['non_default'].sum()
    
    cum_default = df_eval['default'].cumsum() / max(total_defaults, 1)
    cum_non_default = df_eval['non_default'].cumsum() / max(total_non_defaults, 1)
    
    ks_stat = float(np.max(np.abs(cum_default - cum_non_default)) * 100.0)
    
    return {
        "roc_auc": auc,
        "gini": gini,
        "ks_stat_pct": ks_stat,
        "brier_score": brier
    }


def calculate_psi(
    expected_probs: np.ndarray, 
    actual_probs: np.ndarray, 
    num_bins: int = 10
) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and monitoring distributions.
    PSI = sum((Actual_b - Expected_b) * ln(Actual_b / Expected_b))
    """
    exp_arr = np.asarray(expected_probs, dtype=float)
    act_arr = np.asarray(actual_probs, dtype=float)
    
    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0
    
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(exp_arr, quantiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    
    bins[0] = 0.0
    bins[-1] = 1.0 + 1e-7
    
    expected_counts, _ = np.histogram(exp_arr, bins=bins)
    actual_counts, _ = np.histogram(act_arr, bins=bins)
    
    expected_pct = np.maximum(expected_counts / len(exp_arr), 1e-4)
    actual_pct = np.maximum(actual_counts / len(act_arr), 1e-4)
    
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def generate_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Generates FPR, TPR, and threshold coordinates for ROC plotting.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist()
    }


def generate_ks_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> Dict[str, Any]:
    """
    Generates cumulative default vs non-default separation curves for KS evaluation.
    """
    df_eval = pd.DataFrame({'target': y_true, 'prob': y_prob}).sort_values(by='prob', ascending=True)
    df_eval['cum_total'] = np.linspace(0, 1, len(df_eval))
    
    tot_def = max(df_eval['target'].sum(), 1)
    tot_non_def = max((1 - df_eval['target']).sum(), 1)
    
    df_eval['cum_def'] = df_eval['target'].cumsum() / tot_def
    df_eval['cum_non_def'] = (1 - df_eval['target']).cumsum() / tot_non_def
    df_eval['ks_diff'] = np.abs(df_eval['cum_def'] - df_eval['cum_non_def'])
    
    ks_max_idx = df_eval['ks_diff'].idxmax()
    ks_max_row = df_eval.loc[ks_max_idx]
    
    return {
        "prob": df_eval['prob'].tolist(),
        "cum_def": df_eval['cum_def'].tolist(),
        "cum_non_def": df_eval['cum_non_def'].tolist(),
        "ks_stat_pct": float(ks_max_row['ks_diff'] * 100.0),
        "cutoff_prob": float(ks_max_row['prob'])
    }


def generate_calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Generates fractional positives and mean predicted probabilities for calibration reliability curve.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }


def evaluate_rmg_kpis(metrics: Dict[str, float], config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """
    Checks validation metrics against RMG operational targets and recalibration trigger bounds.
    """
    thresholds = {
        "roc_auc_target": 0.80, "roc_auc_trigger": 0.72,
        "gini_target": 0.60, "gini_trigger": 0.44,
        "ks_stat_pct_target": 45.0, "ks_stat_pct_trigger": 35.0,
        "brier_score_target": 0.08, "brier_score_trigger": 0.15,
        "psi_stable": 0.10, "psi_action": 0.25
    }
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'validation' in cfg and 'thresholds' in cfg['validation']:
                thresholds.update(cfg['validation']['thresholds'])

    eval_status = {}
    
    # ROC-AUC
    auc = metrics.get("roc_auc", 0.0)
    if auc >= thresholds["roc_auc_target"]:
        eval_status["roc_auc"] = ("PASS", "Exceeds RMG operational target (>= 0.80)")
    elif auc >= thresholds["roc_auc_trigger"]:
        eval_status["roc_auc"] = ("MONITOR", "Acceptable, monitor closely (0.72 - 0.80)")
    else:
        eval_status["roc_auc"] = ("RECALIBRATE", "Breached lower bound (< 0.72)")

    # Gini
    gini = metrics.get("gini", 0.0)
    if gini >= thresholds["gini_target"]:
        eval_status["gini"] = ("PASS", "High discrimination power (>= 0.60)")
    elif gini >= thresholds["gini_trigger"]:
        eval_status["gini"] = ("MONITOR", "Adequate (0.44 - 0.60)")
    else:
        eval_status["gini"] = ("RECALIBRATE", "Sub-standard discrimination (< 0.44)")

    # KS Stat
    ks = metrics.get("ks_stat_pct", 0.0)
    if ks >= thresholds["ks_stat_pct_target"]:
        eval_status["ks_stat"] = ("PASS", "Excellent default/non-default separation (>= 45%)")
    elif ks >= thresholds["ks_stat_pct_trigger"]:
        eval_status["ks_stat"] = ("MONITOR", "Moderate separation (35% - 45%)")
    else:
        eval_status["ks_stat"] = ("RECALIBRATE", "Poor score separation (< 35%)")

    # Brier
    brier = metrics.get("brier_score", 1.0)
    if brier <= thresholds["brier_score_target"]:
        eval_status["brier_score"] = ("PASS", "High calibration probability accuracy (<= 0.08)")
    elif brier <= thresholds["brier_score_trigger"]:
        eval_status["brier_score"] = ("MONITOR", "Moderate probability accuracy (0.08 - 0.15)")
    else:
        eval_status["brier_score"] = ("RECALIBRATE", "Inaccurate probability calibration (> 0.15)")

    return eval_status
