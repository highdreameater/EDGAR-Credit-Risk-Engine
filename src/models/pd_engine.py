"""
Probability of Default (PD) Modeling & Scorecard Calibration Engine for GCCR.
Supports Logistic Regression Scorecards and XGBoost Classifiers with
Platt Scaling (Logistic Sigmoidal Calibration) or Isotonic Calibration.
"""

import os
from typing import Tuple, Optional, Dict, Any, List, Union
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


DEFAULT_FEATURES = [
    'icr',
    'debt_to_ebitda',
    'current_ratio',
    'ocf_to_debt',
    'operating_margin',
    'altman_z'
]


class CreditScorecardModel:
    """
    Wrapper for calibrated credit default probability model.
    Provides feature coefficients/importance, uncalibrated score, calibrated 1-year PD,
    scorecard waterfall explainability, and joblib serialization.
    """
    def __init__(
        self,
        pipeline: Pipeline,
        calibrated_clf: CalibratedClassifierCV,
        feature_names: List[str],
        algorithm: str = "logistic_regression"
    ):
        self.pipeline = pipeline
        self.calibrated_clf = calibrated_clf
        self.feature_names = feature_names
        self.algorithm = algorithm

    def predict_uncalibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Outputs raw baseline model default probabilities."""
        return self.pipeline.predict_proba(X[self.feature_names])[:, 1]

    def predict_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Outputs Platt-calibrated 1-year default probabilities (PD in [0, 1])."""
        return self.calibrated_clf.predict_proba(X[self.feature_names])[:, 1]

    def get_feature_weights(self) -> Dict[str, float]:
        """Returns scorecard coefficients or tree feature importances."""
        model_step = self.pipeline.named_steps.get('model')
        if hasattr(model_step, 'coef_'):
            coefs = model_step.coef_[0]
            return dict(zip(self.feature_names, [float(c) for c in coefs]))
        elif hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            return dict(zip(self.feature_names, [float(i) for i in importances]))
        return {feat: 0.0 for feat in self.feature_names}

    def explain_counterparty_score(self, sample_row: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Decomposes the model score into transparent feature contributions.
        For Logistic Regression: logit contributions Z_j * beta_j relative to base intercept.
        For XGBoost: normalized tree attribution.
        Returns waterfall components suitable for regulatory audit and visual plotting.
        """
        if isinstance(sample_row, pd.Series):
            df_row = sample_row.to_frame().T
        elif isinstance(sample_row, pd.DataFrame):
            df_row = sample_row.iloc[0:1]
        else:
            df_row = pd.DataFrame([sample_row])

        X = df_row[self.feature_names]
        calibrated_pd = float(self.predict_calibrated(X)[0])
        uncalibrated_pd = float(self.predict_uncalibrated(X)[0])

        model_step = self.pipeline.named_steps.get('model')
        scaler = self.pipeline.named_steps.get('scaler')

        contributions = []
        intercept = 0.0

        if hasattr(model_step, 'coef_') and scaler is not None:
            intercept = float(model_step.intercept_[0]) if hasattr(model_step, 'intercept_') else 0.0
            coefs = model_step.coef_[0]
            scaled_vals = scaler.transform(X)[0]
            raw_vals = X.iloc[0].values

            for feat, raw_v, scaled_v, coef in zip(self.feature_names, raw_vals, scaled_vals, coefs):
                contrib = float(scaled_v * coef)
                contributions.append({
                    "feature": feat,
                    "value": float(raw_v),
                    "scaled_z": float(scaled_v),
                    "coefficient": float(coef),
                    "contribution": contrib,
                    "risk_impact": "Increases Risk" if contrib > 0 else "Decreases Risk"
                })
        elif hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            raw_vals = X.iloc[0].values
            for feat, raw_v, imp in zip(self.feature_names, raw_vals, importances):
                contrib = float(imp * (raw_v / (abs(raw_v) + 1.0)))
                contributions.append({
                    "feature": feat,
                    "value": float(raw_v),
                    "scaled_z": 0.0,
                    "coefficient": float(imp),
                    "contribution": contrib,
                    "risk_impact": "Increases Risk" if contrib > 0 else "Decreases Risk"
                })

        return {
            "intercept": intercept,
            "feature_contributions": contributions,
            "uncalibrated_pd": uncalibrated_pd,
            "calibrated_pd": calibrated_pd
        }

    def save(self, filepath: str) -> None:
        """Serializes trained scorecard model to disk via joblib."""
        import joblib
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CreditScorecardModel':
        """Loads serialized scorecard model from disk."""
        import joblib
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at: {filepath}")
        return joblib.load(filepath)


def train_and_calibrate_pd_model(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "target",
    algorithm: str = "logistic_regression",
    calibration_method: str = "sigmoid",
    config_path: str = "config/model_config.yaml"
) -> Tuple[pd.DataFrame, CreditScorecardModel]:
    """
    Trains and calibrates corporate default probability scorecard.
    
    Returns:
    - df_scored: DataFrame containing all initial fields plus 'target', 'pd_uncalibrated', and 'pd_calibrated'
    - pd_model: Fitted CreditScorecardModel wrapper instance
    """
    data = df.copy()

    # Load configuration if present
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if features is None and 'features' in cfg:
                features = cfg['features']
            if 'model' in cfg:
                if 'algorithm' in cfg['model']:
                    algorithm = cfg['model']['algorithm']
                if 'calibration' in cfg['model'] and 'method' in cfg['model']['calibration']:
                    calibration_method = cfg['model']['calibration']['method']

    if features is None:
        features = DEFAULT_FEATURES

    # Ensure target column exists
    if target_col not in data.columns:
        if 'default_flag' in data.columns:
            data[target_col] = data['default_flag'].astype(int)
        elif os.path.exists("data/raw/corporate_defaults.csv"):
            defaults_df = pd.read_csv("data/raw/corporate_defaults.csv", dtype={'cik': str})
            defaults_df['cik'] = defaults_df['cik'].astype(str).str.zfill(10)
            if 'fiscal_year' in data.columns and 'fiscal_year' in defaults_df.columns:
                data = pd.merge(data, defaults_df[['cik', 'fiscal_year', 'default_flag']], on=['cik', 'fiscal_year'], how='left')
            else:
                data = pd.merge(data, defaults_df[['cik', 'default_flag']].drop_duplicates('cik'), on='cik', how='left')
            data[target_col] = data['default_flag'].fillna(0).astype(int)
        else:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Clean missing values in feature set
    X = data[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = data[target_col].astype(int)

    # Initialize Base Estimator
    if algorithm == "xgboost":
        base_estimator = xgb.XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        base_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_estimator)
        ])
    else:
        # Standard Institutional Logistic Regression Scorecard
        base_estimator = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        base_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_estimator)
        ])

    # Fit base pipeline
    base_pipe.fit(X, y)

    # Execute Platt Scaling (sigmoid) or Isotonic calibration via cross-validation
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_pipe,
        method=calibration_method,
        cv=3
    )
    calibrated_clf.fit(X, y)

    scorecard_model = CreditScorecardModel(
        pipeline=base_pipe,
        calibrated_clf=calibrated_clf,
        feature_names=features,
        algorithm=algorithm
    )

    # Predict baseline and calibrated PDs
    data['pd_uncalibrated'] = scorecard_model.predict_uncalibrated(X)
    data['pd_calibrated'] = scorecard_model.predict_calibrated(X)
    
    # Ensure PD is strictly in [0.0001, 0.9999] for numerical stability
    data['pd_calibrated'] = data['pd_calibrated'].clip(lower=0.0001, upper=0.9999)

    return data, scorecard_model
