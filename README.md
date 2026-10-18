# GLOBAL CORPORATE COUNTERPARTY RATING & LIMIT MONITORING ENGINE (GCCR)
### Institutional Wholesale Credit Risk Analytics, SEC EDGAR Ingestion, Basel III Capital & Portfolio Governance Engine

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![License: Institutional](https://img.shields.io/badge/License-Institutional-navy.svg)
![Regulatory Alignment: Basel III / IRB](https://img.shields.io/badge/Regulatory%20Alignment-Basel%20III%20%2F%20IRB-emerald.svg)
![SEC EDGAR API: 10--K Ingest](https://img.shields.io/badge/SEC%20EDGAR-10--K%20XBRL-orange.svg)
![Model Governance: SR 11--7 Compliant](https://img.shields.io/badge/Model%20Governance-SR%2011--7-purple.svg)
![Test Suite: 23 Passed](https://img.shields.io/badge/Unit%20Tests-23%20Passed%20(100%25)-brightgreen.svg)

---

### 1. Executive Summary & Strategic High-Level Impact

#### The Strategic Problem in Wholesale Banking
Global investment banks, commercial lenders, and asset managers deploy trillions of dollars in bilateral credit lines, syndicated loans, counterparty derivative facilities, and trade finance to corporate counterparties worldwide. Traditionally, wholesale credit assessment suffers from severe structural vulnerabilities:
1. **Manual & Lagging Credit Reviews**: Credit underwriting teams manually parse 10-K filings and spreadsheets, causing rating assignments and credit reviews to lag deteriorating corporate fundamentals by months.
2. **Disconnected Operational Silos**: Rating assignments (CRM), exposure tracking (CEM), and quantitative model governance (RMG) operate in separate silos, leading to uncoordinated risk management.
3. **Unmonitored Facility Drift & Limit Breaches**: In dynamic macro environments, counterparty drawdowns can silently breach sanctioned credit limits before credit committees are alerted, exposing institutions to uncollateralized tail risk.
4. **Vulnerability to Macroeconomic Shocks**: Static annual ratings fail to predict how balance sheets will migrate during interest rate hikes, demand recessions, or stagflationary margin squeezes.
5. **Lack of Explainability & Basel Capital Cohesion**: Regulators (Federal Reserve, OCC, ECB) require auditable adverse action reason codes (SR 11-7) and Basel III/IV Advanced Internal Ratings-Based (A-IRB) capital adequacy tracking ($EL$, $UL$, $RWA$) which ad-hoc spreadsheets cannot provide.

#### What GCCR Solves: The Strategic Impact
The **Global Corporate Counterparty Rating & Limit Monitoring Engine (GCCR)** transforms institutional wholesale credit risk from a fragmented, reactive process into an **automated, continuous, and quantitatively governed risk operating system**:

* **Automated Direct Filing Ingestion**: Connects directly to the **official SEC EDGAR XBRL REST API**, replacing manual balance sheet extraction with audited, real-time corporate financial filings across public global issuers.
* **Objective, Calibrated Default Probability ($PD$)**: Implements Basel III Internal Ratings-Based (IRB) compliant scorecard models equipped with **Platt Scaling (logistic sigmoidal calibration)**, producing mathematically rigorous, out-of-fold calibrated 1-year default probabilities.
* **Basel III A-IRB Capital Engine**: Computes Unexpected Loss ($UL$), Risk-Weighted Assets ($RWA$), and Regulatory Capital ($K$) using supervisory asset correlation and maturity adjustment formulas.
* **Federal Reserve SR 11-7 Explainability**: Decomposes continuous risk scores into standardized feature log-odds attributions and visual waterfall reason codes for credit committee approvals.
* **Live Real-Time SEC Underwriting Sandbox**: Allows credit officers to input any public stock ticker or CIK, pull government 10-K statements in real-time, compute fundamental ratios, and generate instant underwriting decisions.
* **Unified Stakeholder Operating Architecture**:
  * **Credit Risk Management (CRM)**: Standardizes counterparty underwriting by translating solvency metrics into an 8-band Master Scale ($AAA$ to $CCC$) and Investment Grade (IG) vs. High Yield (HY) classifications.
  * **Credit Exposure Management (CEM)**: Implements automated limit surveillance with early-warning alerts ($85\% \le \text{utilization} \le 100\%$) and instantaneous breach flags ($>100\%$) with excess exposure dollar tracking.
  * **Risk Methodologies Group (RMG)**: Enforces model risk governance under Federal Reserve SR 11-7 standards, executing continuous discriminatory power checks (ROC-AUC, Gini, KS), calibration accuracy (Brier Score), and population distribution drift (PSI).
* **CCAR/DFAST-Style Macroeconomic Stress Testing**: Simulates multi-sector macroeconomic shocks (e.g., Severe Global Recession, Aggressive Rate Hikes, Margin Compression) to project portfolio Expected Loss ($\Delta EL$) and compute post-stress rating migration transition matrices.
* **Board & Executive Transparency**: A unified, institutional Streamlit executive console that provides credit committees and CROs with real-time portfolio health, sector concentrations, and stress resilience.

---

### 2. High-Level Architecture & End-to-End Data Flow

```
                   +-------------------------------------------------+
                   |  SEC EDGAR API / SEC Fact JSONs / 10-K Filings   |
                   |  (Official U.S. SEC XBRL Corporate Submissions) |
                   +-------------------------------------------------+
                                           |
                                           v
                   +-------------------------------------------------+
                   |   src/data/edgar_loader.py & preprocessor.py    |
                   |   (Ingestion, Sector Median Imputation,         |
                   |    1st/99th Percentile Outlier Winsorization)   |
                   +-------------------------------------------------+
                                           |
                                           v
                   +-------------------------------------------------+
                   |   src/features/financial_ratios.py              |
                   |   (ICR, Debt/EBITDA, Current Ratio, Altman Z)   |
                   +-------------------------------------------------+
                                           |
                                           v
                   +-------------------------------------------------+
                   |   src/models/pd_engine.py                       |
                   |   (Scorecard + Platt Sigmoidal Calibration)     |
                   |   [Persisted: data/processed/calibrated_model]  |
                   +-------------------------------------------------+
                                           |
         +---------------------------------+---------------------------------+
         |                                 |                                 |
         v                                 v                                 v
+-----------------------+     +-----------------------+     +-----------------------+
| src/models/validation |     | src/models/pd_engine  |     | src/risk/rating_engine|
| (ROC-AUC, Gini, KS,   |     | (SR 11-7 Waterfall    |     | (Master Scale PD ->   |
|  Brier Score, PSI)    |     |  Attribution Codes)   |     |  Rating Band AAA..CCC)|
+-----------------------+     +-----------------------+     +-----------------------+
                                                                         |
                                                                         v
                                                            +-----------------------+
                                                            | src/risk/el_calculator|
                                                            | Basel III A-IRB:      |
                                                            | EL, UL, RWA, Capital K|
                                                            +-----------------------+
                                                                         |
                                                                         v
                                                            +-----------------------+
                                                            | src/risk/limit_monitor|
                                                            | (Facility Utilization,|
                                                            |  Breaches, Excess EAD)|
                                                            +-----------------------+
                                                                         |
                                                                         v
                                                            +-----------------------+
                                                            | src/risk/stress_tester|
                                                            | (CCAR Macro Scenarios,|
                                                            |  Migration Matrices)  |
                                                            +-----------------------+
                                                                         |
                                                                         v
                                                            +-----------------------+
                                                            | src/dashboard/app.py  |
                                                            | (5-Tab Streamlit      |
                                                            |  Executive Console)   |
                                                            +-----------------------+
```

---

### 3. Quantitative Formulations

#### A. Fundamental Financial Ratios & Altman Z-Score
* **Interest Coverage Ratio (ICR)**: Measures debt service capacity from core operating profits:
  $$\text{ICR} = \frac{\text{EBIT}}{\max(|\text{Interest Expense}|, 10^{-5})}$$
* **Leverage Ratio (Debt-to-EBITDA)**: Quantifies payback horizon based on operating cash earnings:
  $$\text{Debt-to-EBITDA} = \frac{\text{Total Debt}}{\max(\text{EBITDA}, 10^{-5})}$$
* **Current Liquidity Ratio**: Evaluates near-term working capital coverage:
  $$\text{Current Ratio} = \frac{\text{Current Assets}}{\max(\text{Current Liabilities}, 10^{-5})}$$
* **Operating Cash Flow Solvency Ratio**: Evaluates cash-generative capability relative to total obligations:
  $$\text{OCF-to-Debt} = \frac{\text{Operating Cash Flow}}{\max(\text{Total Debt}, 10^{-5})}$$
* **Operating Profit Margin**:
  $$\text{Operating Margin} = \frac{\text{EBIT}}{\max(\text{Revenue}, 10^{-5})}$$
* **Altman Z-Score Model (Corporate Manufacturing & General Firms)**: Composite multi-factor distress indicator:
  $$Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 0.999 X_5$$
  * $X_1 = (\text{Current Assets} - \text{Current Liabilities}) / \text{Total Assets}$ (Working Capital / Total Assets)
  * $X_2 = \text{Retained Earnings} / \text{Total Assets}$ (Cumulative Profitability / Total Assets)
  * $X_3 = \text{EBIT} / \text{Total Assets}$ (Operating Productivity of Assets)
  * $X_4 = \text{Market Value of Equity} / \text{Total Liabilities}$ (Market Leverage)
  * $X_5 = \text{Revenue} / \text{Total Assets}$ (Asset Turnover Efficiency)
  * *Thresholds*: Safe Zone ($Z > 2.99$), Grey Zone ($1.81 \le Z \le 2.99$), Distress Zone ($Z < 1.81$).

#### B. Probability of Default (PD) Modeling & Platt Scaling
* **Logistic Regression Scorecard**:
  $$z_i = \beta_0 + \sum_{j=1}^K \beta_j \left(\frac{X_{ij} - \mu_j}{\sigma_j}\right)$$
* **Platt Scaling (Sigmoidal Logistic Calibration)**: Fits parameters $A$ and $B$ via Maximum Likelihood Estimation on cross-validation holdouts to map raw scores into empirical default frequencies:
  $$PD_i = \frac{1}{1 + \exp(-(A \cdot z_i + B))}$$

#### C. Federal Reserve SR 11-7 Scorecard Explainability & Factor Decomposition
To satisfy supervisory requirements for adverse action notices and credit committee reviews, continuous log-odds scores are decomposed into feature-level risk contributions relative to the baseline corporate intercept:
$$\text{logit}(PD_i) = \beta_0 + \sum_{j=1}^K \Delta \text{Logit}_{ij} \quad \text{where} \quad \Delta \text{Logit}_{ij} = \beta_j \cdot Z_{ij}$$

| Feature Name | Scorecard Weight ($\beta$) | Directional Impact | Economic Rationale |
| :--- | :---: | :---: | :--- |
| **Model Intercept ($\beta_0$)** | **-1.08** | Baseline Rate | Wholesale corporate baseline log-odds |
| **Interest Coverage (ICR)** | **-1.88** | Risk-Decreasing | High debt service coverage significantly lowers default likelihood |
| **Operating Cash Flow / Debt** | **-1.43** | Risk-Decreasing | Direct liquidity available to amortize outstanding liabilities |
| **Current Liquidity Ratio** | **-1.19** | Risk-Decreasing | Working capital buffer protects against short-term payment shocks |
| **Debt-to-EBITDA** | **-1.04** | Risk-Increasing with debt | High leverage increases sensitivity to cyclical earnings contractions |
| **Altman Z-Score** | **-0.44** | Risk-Decreasing | Multi-factor balance sheet productivity and solvency protection |
| **Operating Profit Margin** | **-0.24** | Risk-Decreasing | Profit cushion absorbs input cost inflation and revenue compression |

#### D. Basel III Advanced Internal Ratings-Based (A-IRB) Capital Engine
* **Asset Correlation ($R$)**: Governed by the Basel supervisory formula for corporate exposures:
  $$R = 0.12 \times \left(\frac{1 - e^{-50 \times PD}}{1 - e^{-50}}\right) + 0.24 \times \left[ 1 - \left(\frac{1 - e^{-50 \times PD}}{1 - e^{-50}}\right) \right]$$
* **Maturity Adjustment Factor**: Standard wholesale effective maturity ($M = 2.5$ years):
  $$b(PD) = (0.11852 - 0.05478 \times \ln(PD))^2 \quad \implies \quad \text{MatAdj} = \frac{1 + (M - 2.5) \times b(PD)}{1 - 1.5 \times b(PD)}$$
* **Capital Requirement ($K$) per Unit Exposure**:
  $$K = \left[ LGD \cdot \Phi\left(\frac{\Phi^{-1}(PD) + \sqrt{R}\Phi^{-1}(0.999)}{\sqrt{1 - R}}\right) - (PD \cdot LGD) \right] \times \text{MatAdj}$$
* **Unexpected Loss ($UL$) / Economic Capital at 99.9% VaR**:
  $$UL_i = K_i \times EAD_i$$
* **Risk-Weighted Assets ($RWA$)**:
  $$RWA_i = K_i \times 12.5 \times EAD_i \quad \implies \quad \text{Capital Density} = \left(\frac{RWA_i}{EAD_i}\right) \times 100\%$$
* **Regulatory Minimum Capital Requirement (8%)**:
  $$\text{Capital}_i = 0.08 \times RWA_i = UL_i$$
* **Expected Loss ($EL$)**:
  $$EL_i = PD_i \times LGD_i \times EAD_i$$

#### E. RMG Quantitative Model Validation Metrics & Supervisory Thresholds
* **Gini Coefficient**: Scaled measure of discrimination power:
  $$\text{Gini} = 2 \times \text{ROC-AUC} - 1$$
* **Kolmogorov-Smirnov (KS) Statistic**: Peak separation between default and non-default cumulative distributions:
  $$KS = \max_t |F_{\text{Default}}(t) - F_{\text{Non-Default}}(t)| \times 100\%$$
* **Brier Score (BS)**: Mean squared error of calibrated probability predictions:
  $$BS = \frac{1}{N}\sum_{i=1}^N (PD_i - y_i)^2$$
* **Population Stability Index (PSI)**: Measures filing cohort and feature drift across credit cycles:
  $$PSI = \sum_{b=1}^B (\text{Actual}_b - \text{Expected}_b) \times \ln\left(\frac{\text{Actual}_b}{\text{Expected}_b}\right)$$

---

### 4. Target Stakeholders & Enterprise Role Mapping

| Institutional Role | Core Responsibilities | Engine Capability Mapping | Business Impact |
| :--- | :--- | :--- | :--- |
| **Credit Risk Management (CRM)** | Rating assignment, borrower financial analysis, underwriting approvals. | Ingests live SEC 10-K data, computes solvency metrics, maps calibrated PDs to 8 rating bands ($AAA$ to $CCC$). | Eliminates underwriting lag; creates consistent, audited rating decisions. |
| **Credit Exposure Management (CEM)** | Portfolio exposure surveillance, facility limit control, concentration risk. | Computes facility utilization, triggers automated breach flags ($>100\%$), computes dollar excess exposure. | Prevents silent limit creep; eliminates uncollateralized over-allocation. |
| **Risk Methodologies Group (RMG)** | Scorecard development, calibration, model validation, stress testing. | Calculates full validation suite (ROC, Gini, KS, Brier, PSI), executes macro stress test simulations. | Ensures SR 11-7 and Basel III regulatory model compliance. |
| **Capital Planning & Treasury** | Capital adequacy, RWA optimization, unexpected loss reserves. | Calculates Basel III A-IRB Capital ($K$), Unexpected Loss ($UL$), and portfolio Capital Density ($RWA / EAD$). | Guarantees compliance with regulatory minimums (Pillar 1/2). |
| **Credit Committee & CRO** | Capital planning, board reporting, risk appetite oversight. | Executive 5-tab Streamlit dashboard, portfolio Expected Loss aggregation, rating migration matrices. | Delivers instant visibility into wholesale portfolio health and crisis resilience. |

---

### 5. Empirical Production Results & Benchmark Metrics (Live SEC EDGAR Run)

The quantitative results below were generated from the full production run of [`run_pipeline.py`](file:///Users/highdreameater/Desktop/GCCR/run_pipeline.py) using audited 10-K filings ingested directly from the official U.S. SEC EDGAR XBRL API and exported to [`reports/executive_pipeline_summary.json`](file:///Users/highdreameater/Desktop/GCCR/reports/executive_pipeline_summary.json):

#### A. Ingestion & Data Quality Preprocessing
* **Public Corporate Entities**: **44 unique global corporations** across 8 sectors (Apple, Microsoft, Amazon, Alphabet, Meta, NVIDIA, Boeing, Caterpillar, ExxonMobil, Chevron, Walmart, Pfizer, Johnson & Johnson, AT&T, NextEra Energy, Dow, etc.).
* **Audited 10-K Filings Processed**: **218 multi-year annual observations** spanning fiscal years 2019–2024.
* **Data Quality Pipeline**: Sector-specific median imputation for missing items; 1st and 99th percentile winsorization of extreme ratio spikes.
* **Calibrated 1-Year PD Range**: **$2.26\%$ to $20.44\%$** (Platt sigmoidal scaling).

#### B. RMG Quantitative Model Validation Benchmark
Validation executed on out-of-fold calibrated probabilities against historical credit default events:

| Metric Class | RMG Validation Metric | Operational Target | Action Trigger | Observed Value | Governance Audit Status |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Separation** | **Kolmogorov-Smirnov (KS)** | $\ge 45.0\%$ | $< 35.0\%$ | **49.54%** | **PASS** (Strong cut-off separation) |
| **Calibration** | **Brier Score** | $\le 0.0800$ | $> 0.1500$ | **0.0461** | **PASS** (High calibration precision) |
| **Discrimination** | **ROC-AUC** | $\ge 0.8000$ | $< 0.7200$ | **0.7756** | **ACCEPTABLE** (Active monitoring zone) |
| **Discrimination** | **Gini Coefficient** | $\ge 0.6000$ | $< 0.4400$ | **0.5512** | **ACCEPTABLE** (Adequate discrimination) |
| **Stability** | **Population Stability (PSI)** | $< 0.1000$ | $> 0.2500$ | **0.0450** | **PASS** (Stable population across years) |

#### C. Basel III Capital Allocation & Portfolio Loss Summary
Supervisory capital allocations across the corporate wholesale loan book:

| Basel III Risk Metric | Portfolio Value | Capital Metric Interpretation |
| :--- | :---: | :--- |
| **Total Exposure at Default (EAD)** | **$80,749.70M** | Total wholesale committed corporate credit facilities |
| **Expected Loss (EL)** | **$2,016.46M** | Accounting credit reserve provision requirement (**2.50%** EL rate) |
| **Unexpected Loss (UL)** | **$9,452.54M** | Economic Capital reserve at 99.9% 1-year solvency confidence level |
| **Basel III Total RWA** | **$118,156.80M** | Regulatory Risk-Weighted Assets under A-IRB |
| **Portfolio Capital Density** | **146.3%** | Capital intensity ratio ($RWA / EAD$) reflecting corporate portfolio risk mix |
| **Minimum Regulatory Capital (8%)**| **$9,452.54M** | Tier 1/Total Capital minimum required under Basel Pillar 1 |
| **Weighted Average PD** | **5.98%** | Portfolio exposure-weighted 1-year default probability |
| **Weighted Average LGD** | **41.49%** | Portfolio exposure-weighted downturn Loss Given Default |

#### D. Macroeconomic Stress-Testing Results
Simulated portfolio loss shifts under supervisory macro shock scenarios:

| Macroeconomic Scenario | Portfolio Expected Loss ($M) | Loss Delta $\Delta EL$ ($M) | Loss Increase (%) | Capital Impact Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline Portfolio** | **$2,016.46M** | **$0.00M** | **0.0%** | Baseline wholesale operating environment |
| **Severe Global Recession** | **$4,014.04M** | **+$1,997.58M** | **+99.1%** | Demand collapse & cyclical default spike |
| **Aggressive Rate Hike (+300 bps)** | **$3,380.84M** | **+$1,364.38M** | **+67.7%** | Debt-servicing cost headwind |
| **Stagflationary Margin Squeeze** | **$3,691.58M** | **+$1,675.12M** | **+83.1%** | Input inflation & revenue margin compression |

#### E. Credit Facility Limit Monitoring & Surveillance
* **Total Facilities Monitored**: **218 facilities** across global public corporate issuers.
* **Facilities within Normal Limits (<85%)**: **163 facilities (74.8%)**.
* **Watchlist Warnings (85%–100%)**: **35 facilities (16.1%)**.
* **Active Facility Limit Breaches (>100%)**: **20 facilities (9.2%)** flagged with automated CEM excess exposure alerts.
* **Total Excess Exposure**: **$1,820.30M** requiring credit committee capital mitigation or freeze.
* **Average Facility Utilization**: **72.4%** (Peak utilization: **122.9%**).

#### F. Live SEC EDGAR Real-Time Underwriting Case Study: The Boeing Company (NYSE: BA)
Underwritten in real-time via **Tab 5** of the GCCR Risk Console pulling government XBRL filings:
* **Filing Ingested**: Form 10-K (Fiscal Year 2024), SEC CIK `0000012927`.
* **Reported Statements**: Total Assets **$137,000.0M** | Total Debt **$57,800.0M** | Revenue **$66,540.0M** | EBIT **-$1,250.0M**.
* **Engineered Ratios**: ICR **-0.48x** | Debt-to-EBITDA **-36.1x** | Current Ratio **1.14x** | Altman Z-Score **0.57 (Distress Zone)**.
* **Assigned Rating**: **BB (High Yield / Sub-Investment Grade)**.
* **Calibrated 1-Year PD**: **3.66%**.
* **Facility Underwriting ($75M EAD / $100M Limit)**:
  * Utilization: **75.0% (NORMAL)**.
  * Facility Expected Loss ($EL$): **$1.24M**.
  * Unexpected Loss ($UL$ / Economic Capital): **$8.16M**.
  * Basel III RWA: **$101.95M** (Capital Requirement $K = 10.87\%$).

---

### 6. Project Directory Structure

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
│   │   ├── features_engineered.parquet # Engineered ratios with outlier winsorization
│   │   ├── model_predictions.parquet   # Scored portfolio with EL, UL, RWA, limits
│   │   └── calibrated_pd_model.joblib  # Serialized production scorecard artifact
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
├── reports/
│   └── executive_pipeline_summary.json # Machine-readable audit report of validation & capital
│
├── src/
│   ├── data/
│   │   ├── edgar_loader.py        # SEC EDGAR XBRL REST API live ingestion & ticker resolver
│   │   └── preprocessor.py        # Outlier winsorization, sector median imputation
│   ├── features/
│   │   └── financial_ratios.py    # Solvency, Liquidity, & Altman Z Ratio Engine
│   ├── models/
│   │   ├── pd_engine.py           # Logistic & XGBoost training, Platt Calibration, SR 11-7 Explainability
│   │   └── validation.py          # ROC-AUC, Gini, KS, Brier, PSI validation suite
│   ├── risk/
│   │   ├── rating_engine.py       # Master scale PD-to-Rating mapping (AAA..CCC)
│   │   ├── el_calculator.py       # Expected Loss & Basel III A-IRB Capital (K, UL, RWA)
│   │   ├── limit_monitor.py       # Limit utilization & breach flag generation
│   │   └── stress_tester.py       # Sector-level macro scenario stress testing
│   └── dashboard/
│       └── app.py                 # 5-Tab Streamlit Institutional Risk Console
│
├── tests/
│   ├── test_edgar_loader.py       # Unit tests for ticker resolution and 10-K parsing
│   ├── test_el_calculator.py      # Unit tests for Basel III Capital, UL, RWA, and EL
│   ├── test_preprocessor.py       # Unit tests for winsorization and imputation
│   ├── test_ratios.py             # Unit tests for fundamental ratios & Altman Z
│   ├── test_pd_engine.py          # Unit tests for Platt calibration & waterfall explainability
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
└── run_pipeline.py                # CLI entry point executing full hardened engine
```

---

### 7. Setup & Execution Guide

#### Environment Setup (Conda)
```bash
# Create and activate environment
conda create -y -n gccr_env python=3.11
conda activate gccr_env

# Install dependencies
pip install -r requirements.txt
```

#### Run Full Quantitative Pipeline
Executes data preprocessing, ratio feature engineering, scorecard calibration, model persistence, RMG validation, Basel III capital calculations, limit surveillance, and macro stress testing:
```bash
python run_pipeline.py --model logistic_regression --calibration sigmoid --export-report
```

#### Run Complete Unit Test Suite (23 Tests)
```bash
pytest -v
```

#### Launch Streamlit Executive Risk Console
Opens the 5-tab institutional analytics dashboard:
```bash
streamlit run src/dashboard/app.py
```
* **Tab 1: Executive Portfolio & Basel III Capital**: Top-level EAD, Expected Loss, Unexpected Loss (Economic Capital), Total RWA, and Master Exposure Grid.
* **Tab 2: RMG Model Validation Console**: Interactive ROC Curve, Kolmogorov-Smirnov cumulative separation plot, calibration reliability diagram, and SR 11-7 governance audit checklist.
* **Tab 3: Limit Monitoring & Breach Console**: Real-time facility utilization histograms, sector concentration donut, and CEM exception register.
* **Tab 4: Macro Stress-Testing Console**: Supervisory macro shock comparisons, interactive scenario drilldowns, and rating downgrade migration matrices.
* **Tab 5: Live SEC Ticker Underwriting Sandbox**: Query any public stock ticker (e.g., `BA`, `TSLA`, `AAPL`, `F`, `CAT`) to ingest government 10-K filings, calculate ratios, score calibrated PD, and generate real-time credit memos.
