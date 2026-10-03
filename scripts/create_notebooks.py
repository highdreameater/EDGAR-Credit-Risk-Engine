"""
Generates clean, executable Jupyter notebooks for the GCCR engine:
1. 01_sec_edgar_data_cleaning.ipynb
2. 02_pd_model_training_calibration.ipynb
3. 03_rmg_validation_and_stress_testing.ipynb
"""

import os
import nbformat as nbf

os.makedirs("notebooks", exist_ok=True)

# -------------------------------------------------------------
# Notebook 1: SEC EDGAR Data Cleaning & Fundamental Feature Engineering
# -------------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("""# Global Corporate Counterparty Rating (GCCR)
## Notebook 01: SEC EDGAR Ingestion, Preprocessing & Fundamental Feature Engineering

In this notebook, we demonstrate wholesale credit fundamentals ingestion, sector-specific missing value imputation, outlier winsorization, and institutional solvency/liquidity ratio engineering.
"""),
    nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src modules are resolvable
sys.path.append("..")

from src.data.edgar_loader import load_sec_financials, load_corporate_defaults, merge_financials_with_defaults
from src.data.preprocessor import preprocess_financial_data
from src.features.financial_ratios import compute_financial_ratios

print("Setup completed successfully.")
"""),
    nbf.v4.new_markdown_cell("""### 1. Ingest SEC EDGAR Filings & Merge Default Histories"""),
    nbf.v4.new_code_cell("""# Load official SEC 10-K reported financials
raw_financials = load_sec_financials("../data/raw/edgar_sec_financials.csv")
defaults_df = load_corporate_defaults("../data/raw/corporate_defaults.csv")

merged_df = merge_financials_with_defaults(raw_financials, defaults_df)
print(f"Total SEC Filings Ingested: {len(merged_df)}")
print(f"Unique Corporate Counterparties: {merged_df['cik'].nunique()}")
print(f"Historical Default Events: {merged_df['target'].sum()}")
merged_df.head(5)
"""),
    nbf.v4.new_markdown_cell("""### 2. Preprocessing: Sector Median Imputation & Outlier Winsorization"""),
    nbf.v4.new_code_cell("""# Execute FR-1.3: Sector-specific median imputation and 1st/99th percentile winsorization
clean_df = preprocess_financial_data(merged_df, winsorize=True)
clean_df.describe().round(2)
"""),
    nbf.v4.new_markdown_cell("""### 3. Feature Engineering: Solvency, Leverage & Altman Z-Score"""),
    nbf.v4.new_code_cell("""# Compute fundamental ratios (FR-2)
features_df = compute_financial_ratios(clean_df)

ratio_cols = ['icr', 'debt_to_ebitda', 'current_ratio', 'ocf_to_debt', 'operating_margin', 'altman_z']
features_df[['company_name', 'sector', 'fiscal_year'] + ratio_cols].head(10)
"""),
    nbf.v4.new_markdown_cell("""### 4. Sector-Level Ratio Distribution Analysis"""),
    nbf.v4.new_code_cell("""# Compare Debt-to-EBITDA and Altman Z by Sector
summary_by_sector = features_df.groupby('sector')[ratio_cols].median()
summary_by_sector.round(3)
""")
]

with open("notebooks/01_sec_edgar_data_cleaning.ipynb", "w") as f:
    nbf.write(nb1, f)

# -------------------------------------------------------------
# Notebook 2: PD Model Training & Platt Calibration
# -------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("""# Global Corporate Counterparty Rating (GCCR)
## Notebook 02: Probability of Default (PD) Modeling & Scorecard Calibration

In this notebook, we train institutional credit scorecards (Logistic Regression and XGBoost), execute Platt Scaling (Logistic Sigmoidal Calibration), and inspect scorecard feature weights.
"""),
    nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")

from src.data.edgar_loader import load_sec_financials
from src.features.financial_ratios import compute_financial_ratios
from src.models.pd_engine import train_and_calibrate_pd_model

print("Setup completed successfully.")
"""),
    nbf.v4.new_markdown_cell("""### 1. Ingest Features & Prepare Training Cohort"""),
    nbf.v4.new_code_cell("""df_raw = load_sec_financials("../data/raw/edgar_sec_financials.csv")
df_features = compute_financial_ratios(df_raw)
print(f"Features engineered: {len(df_features)} observations")
"""),
    nbf.v4.new_markdown_cell("""### 2. Train Scorecard with Platt Sigmoidal Calibration"""),
    nbf.v4.new_code_cell("""df_scored, pd_model = train_and_calibrate_pd_model(
    df_features, 
    algorithm="logistic_regression",
    calibration_method="sigmoid"
)

print(f"Calibrated 1-year PD summary:")
print(df_scored[['pd_uncalibrated', 'pd_calibrated']].describe().round(4))
"""),
    nbf.v4.new_markdown_cell("""### 3. Feature Coefficients & Scorecard Interpretability"""),
    nbf.v4.new_code_cell("""weights = pd_model.get_feature_weights()
weights_df = pd.DataFrame(list(weights.items()), columns=['Feature', 'Coefficient']).sort_values(by='Coefficient')

print("Scorecard Risk Drivers:")
print(weights_df.to_string(index=False))
""")
]

with open("notebooks/02_pd_model_training_calibration.ipynb", "w") as f:
    nbf.write(nb2, f)

# -------------------------------------------------------------
# Notebook 3: RMG Validation & Stress Testing
# -------------------------------------------------------------
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("""# Global Corporate Counterparty Rating (GCCR)
## Notebook 03: RMG Quantitative Validation & Macro Stress Testing

In this notebook, we calculate discrimination metrics (ROC-AUC, Gini, KS-Statistic), evaluate probability calibration (Brier Score), measure population stability (PSI), and simulate portfolio macroeconomic stress scenarios.
"""),
    nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.append("..")

from src.data.edgar_loader import load_sec_financials
from src.features.financial_ratios import compute_financial_ratios
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import compute_rmg_validation_metrics, calculate_psi, evaluate_rmg_kpis
from src.risk.rating_engine import assign_master_scale_ratings
from src.risk.el_calculator import calculate_expected_loss
from src.risk.limit_monitor import evaluate_limit_utilization
from src.risk.stress_tester import run_macro_stress_test, compute_rating_migration_matrix, apply_scenario_to_portfolio
import yaml

print("Setup completed successfully.")
"""),
    nbf.v4.new_markdown_cell("""### 1. Execute Scoring & Validation Suite"""),
    nbf.v4.new_code_cell("""df_raw = load_sec_financials("../data/raw/edgar_sec_financials.csv")
df_features = compute_financial_ratios(df_raw)
df_scored, _ = train_and_calibrate_pd_model(df_features)

metrics = compute_rmg_validation_metrics(df_scored['target'], df_scored['pd_calibrated'])
print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
print(f"Gini:          {metrics['gini']:.4f}")
print(f"KS-Statistic:  {metrics['ks_stat_pct']:.2f}%")
print(f"Brier Score:   {metrics['brier_score']:.4f}")

kpi_status = evaluate_rmg_kpis(metrics)
for metric, (status, reason) in kpi_status.items():
    print(f"[{status}] {metric}: {reason}")
"""),
    nbf.v4.new_markdown_cell("""### 2. Limit Monitoring & Facility Utilization (CEM Mandate)"""),
    nbf.v4.new_code_cell("""df_rated = assign_master_scale_ratings(df_scored, config_path="../config/rating_bands.yaml")
df_el = calculate_expected_loss(df_rated, exposures_path="../data/portfolio/exposures_limits.csv")
df_monitored = evaluate_limit_utilization(df_el)

print(df_monitored['status'].value_counts())
df_monitored[['company_name', 'sector', 'rating', 'ead', 'credit_limit', 'utilization_pct', 'status']].head(10)
"""),
    nbf.v4.new_markdown_cell("""### 3. Macro Stress Testing & Rating Migration"""),
    nbf.v4.new_code_cell("""stress_summary = run_macro_stress_test(df_monitored, scenario_config_path="../config/stress_scenarios.yaml")
print(stress_summary.to_string(index=False))

with open("../config/stress_scenarios.yaml") as f:
    scenarios = yaml.safe_load(f)['scenarios']
recession_spec = scenarios['global_recession']

stressed_portfolio = apply_scenario_to_portfolio(df_monitored, recession_spec, rating_config_path="../config/rating_bands.yaml")
migration_matrix = compute_rating_migration_matrix(stressed_portfolio['rating'], stressed_portfolio['stressed_rating'])

print("\\nRating Migration Matrix Under Severe Global Recession (% Transition):")
print(migration_matrix)
""")
]

with open("notebooks/03_rmg_validation_and_stress_testing.ipynb", "w") as f:
    nbf.write(nb3, f)

print("Generated 3 institutional Jupyter notebooks.")
