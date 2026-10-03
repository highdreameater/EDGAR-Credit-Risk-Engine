# GLOBAL CORPORATE COUNTERPARTY RATING & LIMIT MONITORING ENGINE (GCCR)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![License: Institutional](https://img.shields.io/badge/License-Institutional-navy.svg)
![Validation: RMG Basel Compliant](https://img.shields.io/badge/Validation-Basel%20III-emerald.svg)
![SEC EDGAR API: 10--K Ingest](https://img.shields.io/badge/SEC%20EDGAR-10--K%20XBRL-orange.svg)

---

### 1. Executive Summary & Vision
The **Global Corporate Counterparty Rating & Limit Monitoring Engine (GCCR)** is an institutional credit risk analytics platform designed to automate wholesale credit workflows across global corporate sectors. The engine ingests multi-year public SEC EDGAR 10-K/10-Q corporate financial statements, engineers fundamental solvency and liquidity ratios, trains and calibrates Default Probability (PD) models, maps calibrated scores into master scale rating bands, tracks credit facility limit utilization, and runs macro-driven stress tests.

---

### 2. High-Level Architecture & Data Flow

```
                   +-------------------------------------------------+
                   |  SEC EDGAR API / SEC Fact JSONs / Kaggle Ratios  |
                   |  (10-K / 10-Q Public Corporate Filings)         |
                   +-------------------------------------------------+
                                           |
                                           v
                   +-------------------------------------------------+
                   |   src/data/edgar_loader.py & preprocessor.py    |
                   |   (Ingestion, Cleaning, Outlier Winsorization)  |
                   +-------------------------------------------------+
                                           |
                                           v
                   +-------------------------------------------------+
                   |   src/features/financial_ratios.py              |
                   |   (Leverage, Coverage, Liquidity & Altman Z)    |
                   +-------------------------------------------------+
                                           |
                                           v
                   +-------------------------------------------------+
                   |   src/models/pd_engine.py                       |
                   |   (Logistic Scorecard / XGBoost + Platt Scaling)|
                   +-------------------------------------------------+
                                           |
                       +-------------------+-------------------+
                       |                                       |
                       v                                       v
    +-------------------------------------+ +-------------------------------------+
    |  src/models/validation.py           | |  src/risk/rating_engine.py          |
    |  (ROC-AUC, Gini, KS, Brier, PSI)    | |  (Master Scale PD -> Rating Band)   |
    +-------------------------------------+ +-------------------------------------+
                                                               |
                                                               v
                                            +-------------------------------------+
                                            |  src/risk/el_calculator.py          |
                                            |  (EL = PD * LGD * EAD)              |
                                            +-------------------------------------+
                                                               |
                                                               v
                                            +-------------------------------------+
                                            |  src/risk/limit_monitor.py          |
                                            |  (Utilization % & Breach Alerts)    |
                                            +-------------------------------------+
                                                               |
                                                               v
                                            +-------------------------------------+
                                            |  src/risk/stress_tester.py          |
                                            |  (Macro Stress Test Scenarios)      |
                                            +-------------------------------------+
                                                               |
                                                               v
                                            +-------------------------------------+
                                            |  src/dashboard/app.py               |
                                            |  (Streamlit Risk Console)           |
                                            +-------------------------------------+
```

---

### 3. Quantitative Formulations

#### A. Fundamental Financial Ratios
* **Interest Coverage Ratio (ICR)**:
  $$\text{ICR} = \frac{\text{EBIT}}{\max(|\text{Interest Expense}|, 10^{-5})}$$
* **Leverage Ratio**:
  $$\text{Debt-to-EBITDA} = \frac{\text{Total Debt}}{\max(\text{EBITDA}, 10^{-5})}$$
* **Current Liquidity Ratio**:
  $$\text{Current Ratio} = \frac{\text{Current Assets}}{\max(\text{Current Liabilities}, 10^{-5})}$$
* **Operating Cash Flow Solvency Ratio**:
  $$\text{OCF-to-Debt} = \frac{\text{Operating Cash Flow}}{\max(\text{Total Debt}, 10^{-5})}$$
* **Altman Z-Score Model (Corporate Manufacturing & General Firms)**:
  $$Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 0.999 X_5$$
  * $X_1 = (\text{Current Assets} - \text{Current Liabilities}) / \text{Total Assets}$
  * $X_2 = \text{Retained Earnings} / \text{Total Assets}$
  * $X_3 = \text{EBIT} / \text{Total Assets}$
  * $X_4 = \text{Market Value of Equity} / \text{Total Liabilities}$
  * $X_5 = \text{Revenue} / \text{Total Assets}$

#### B. Scorecard Calibration (Platt Scaling)
* **Logistic Scorecard**:
  $$z_i = \beta_0 + \sum_{j=1}^K \beta_j X_{ij}$$
* **Platt Scaling**:
  $$PD_i = \frac{1}{1 + \exp(-(A \cdot z_i + B))}$$
  where $A$ and $B$ are estimated via maximum likelihood on out-of-fold validation sets.

#### C. RMG Model Validation Metrics & Thresholds
* **Gini Coefficient**: $\text{Gini} = 2 \times \text{ROC-AUC} - 1$
* **Kolmogorov-Smirnov (KS) Separation**:
  $$KS = \max_t |F_{\text{Default}}(t) - F_{\text{Non-Default}}(t)| \times 100\%$$
* **Brier Score**:
  $$BS = \frac{1}{N}\sum_{i=1}^N (PD_i - y_i)^2$$
* **Population Stability Index (PSI)**:
  $$PSI = \sum_{b=1}^B (\text{Actual}_b - \text{Expected}_b) \times \ln\left(\frac{\text{Actual}_b}{\text{Expected}_b}\right)$$

| Metric Class | Target | Recalibration Trigger |
| :--- | :--- | :--- |
| **ROC-AUC** | $\ge 0.80$ | $< 0.72$ |
| **Gini** | $\ge 0.60$ | $< 0.44$ |
| **KS Stat** | $\ge 45.0\%$ | $< 35.0\%$ |
| **Brier Score** | $\le 0.08$ | $> 0.15$ |
| **PSI** | $< 0.10$ | $> 0.25$ |

#### D. Exposure Analytics & Stress Testing Formulations
* **Counterparty Expected Loss**:
  $$EL_i = PD_i \times LGD_i \times EAD_i$$
* **Credit Limit Utilization (%)**:
  $$\text{Utilization}_i = \left(\frac{EAD_i}{\text{Credit Limit}_i}\right) \times 100$$
  * **Normal**: $< 85\%$
  * **Warning**: $85\% \le \text{Utilization} \le 100\%$
  * **BREACH**: $> 100\%$
* **Macro Stress Shift**:
  $$PD_{i,\text{stressed}} = \min(1.0, PD_{i,\text{baseline}} \times \text{Multiplier}_{S,\text{Sector}})$$
  $$LGD_{i,\text{stressed}} = \min(1.0, LGD_{i,\text{baseline}} + \Delta LGD_S)$$
  $$EL_{i,\text{stressed}} = PD_{i,\text{stressed}} \times LGD_{i,\text{stressed}} \times EAD_i$$

---

### 4. Project Directory Structure

```
gccr_engine/
│
├── config/
│   ├── model_config.yaml          # Model hyperparameters, feature list, thresholds
│   ├── rating_bands.yaml          # Master scale PD thresholds & rating labels
│   └── stress_scenarios.yaml      # Macro scenario multiplier matrices by sector
│
├── data/
│   ├── raw/
│   │   ├── edgar_sec_financials.csv # Real SEC EDGAR 10-K reported corporate financials
│   │   └── corporate_defaults.csv   # Historical corporate default flags
│   ├── processed/
│   │   ├── features_engineered.parquet
│   │   └── model_predictions.parquet
│   └── portfolio/
│       └── exposures_limits.csv     # Exposure at Default (EAD), LGD, and Credit Limits
│
├── docs/
│   ├── PRD.md                     # Product Requirement Document
│   └── DESIGN.md                  # System Design Specification
│
├── notebooks/
│   ├── 01_sec_edgar_data_cleaning.ipynb
│   ├── 02_pd_model_training_calibration.ipynb
│   └── 03_rmg_validation_and_stress_testing.ipynb
│
├── src/
│   ├── data/
│   │   ├── edgar_loader.py        # SEC EDGAR XBRL REST API ingestion
│   │   └── preprocessor.py        # Outlier winsorization, sector median imputation
│   ├── features/
│   │   └── financial_ratios.py    # Solvency, Liquidity, & Altman Z Ratio Engine
│   ├── models/
│   │   ├── pd_engine.py           # Logistic & XGBoost training + Platt Calibration
│   │   └── validation.py          # ROC-AUC, Gini, KS, Brier, PSI validation suite
│   ├── risk/
│   │   ├── rating_engine.py       # Master scale PD-to-Rating mapping (AAA..CCC)
│   │   ├── el_calculator.py       # Portfolio Expected Loss aggregation
│   │   ├── limit_monitor.py       # Limit utilization & breach flag generation
│   │   └── stress_tester.py       # Sector-level macro scenario stress testing
│   └── dashboard/
│       └── app.py                 # Streamlit Executive Dashboard
│
├── tests/
│   ├── test_ratios.py             # Unit tests for fundamental ratios
│   ├── test_pd_engine.py          # Unit tests for Platt calibration
│   ├── test_limit_monitor.py      # Unit tests for limit breach detection
│   └── test_stress_tester.py      # Unit tests for macro stress deltas
│
├── scripts/
│   ├── fetch_live_sec_edgar.py    # Live query tool for official SEC EDGAR 10-K API
│   ├── generate_seed_data.py      # Seed data generator
│   └── create_notebooks.py        # Jupyter notebook builder
│
├── pytest.ini
├── requirements.txt
└── run_pipeline.py                # CLI entry point executing full engine
```

---

### 5. Setup & Execution Guide

#### Environment Setup (Conda)
```bash
# Create and activate environment
conda create -y -n gccr_env python=3.11
conda activate gccr_env

# Install dependencies
pip install -r requirements.txt
```

#### Fetch Live SEC EDGAR 10-K Financials
To query the official SEC EDGAR XBRL API for public corporate counterparties:
```bash
python scripts/fetch_live_sec_edgar.py
```

#### Run Full Quantitative Pipeline
Executes ingestion, feature engineering, scorecard calibration, RMG validation, and stress testing:
```bash
python run_pipeline.py
```

#### Run Unit Tests
```bash
pytest -v
```

#### Launch Streamlit Executive Risk Dashboard
```bash
streamlit run src/dashboard/app.py
```
