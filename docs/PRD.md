# GLOBAL CORPORATE COUNTERPARTY RATING & LIMIT MONITORING ENGINE (GCCR)
## Product Requirement Document (PRD)

### 1. Executive Summary & Vision
The Global Corporate Counterparty Rating & Limit Monitoring Engine (GCCR) is an institutional credit risk analytics platform designed to automate wholesale credit workflows across corporate sectors using real-world public financial filings (SEC EDGAR 10-K/10-Q submissions and historical financial ratio datasets).

GCCR ingests multi-year fundamentals (P&L, Balance Sheet, Cash Flow), engineers standardized solvency and liquidity ratios, trains and calibrates default probability models, maps default probabilities into internal rating scorecards, monitors credit limit utilization, and runs macro-driven stress tests.

---

### 2. Target Stakeholders & Role Mapping

| Risk Function | Primary Job Responsibilities | Engine Capability Mapping |
| :--- | :--- | :--- |
| **Credit Risk Management (CRM)** | Rating assignment, fundamental financial analysis, multi-period balance sheet review. | Ingests SEC EDGAR XBRL data, computes solvency metrics, maps calibrated PDs to 8 rating bands. |
| **Credit Exposure Management (CEM)** | Portfolio exposure monitoring, counterparty credit limits, tracking excess leverage. | Computes limit utilization, triggers breach warning flags, monitors sector concentration. |
| **Risk Methodologies Group (RMG)** | Model development, scorecard calibration, validation (ROC, Gini, KS, PSI, Brier), stress testing. | Computes model validation suite (ROC-AUC, Gini, KS, Brier, PSI), simulates macro shocks. |

---

### 3. Functional Requirements (FR)

#### FR-1: Real SEC EDGAR & Fundamental Financial Ingestion
- **FR-1.1**: Query and ingest SEC EDGAR API 10-K/10-Q company facts (or financial ratio datasets) for global public corporate counterparties.
- **FR-1.2**: Extract primary accounting line items: Revenue, Operating Income (EBIT), EBITDA, Net Income, Total Assets, Total Liabilities, Short-Term Debt, Long-Term Debt, Cash & Equivalents, Operating Cash Flow, and Free Cash Flow.
- **FR-1.3**: Handle missing financial fields using sector-specific median imputation and winsorize extreme outliers at the 1st and 99th percentiles.

#### FR-2: Fundamental Ratio Feature Engineering
- **FR-2.1**: Compute core leverage, coverage, liquidity, and efficiency ratios: Debt-to-EBITDA, Interest Coverage Ratio (ICR), Current Ratio, Quick Ratio, Operating Cash Flow-to-Total Debt, and Net Profit Margin.
- **FR-2.2**: Calculate composite distress signals including Altman Z-Score equivalents:
  $$Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 0.999 X_5$$

#### FR-3: Probability of Default (PD) Modeling & Scorecard Calibration
- **FR-3.1**: Train Logistic Regression Scorecards and XGBoost Classifiers on multi-year corporate financial ratios and default/distress labels.
- **FR-3.2**: Execute Platt Scaling (Logistic Sigmoidal Calibration) or Isotonic Regression to output calibrated 1-year Probabilities of Default ($PD \in [0, 1]$).

#### FR-4: RMG Quantitative Model Validation Engine
- **FR-4.1**: Compute discrimination power metrics: ROC-AUC, Gini Coefficient ($Gini = 2 \times AUC - 1$), and Kolmogorov-Smirnov (KS) Statistic.
- **FR-4.2**: Measure score accuracy using Brier Score ($BS$) and produce predicted vs. observed calibration curves.
- **FR-4.3**: Quantify population drift across filing years using Population Stability Index (PSI).

#### FR-5: Internal Rating Assignment Engine
- **FR-5.1**: Map calibrated PDs into 8 internal master scale rating bands (AAA to CCC/C).
- **FR-5.2**: Categorize counterparties into Investment Grade (IG) vs. High Yield (HY) tiers.

#### FR-6: Portfolio Exposure & Credit Limit Monitoring (CEM Mandate)
- **FR-6.1**: Calculate Counterparty Expected Loss (EL):
  $$EL_i = PD_i \times LGD_i \times EAD_i$$
- **FR-6.2**: Calculate Credit Limit Utilization:
  $$\text{Utilization}_i = \left(\frac{EAD_i}{\text{Credit Limit}_i}\right) \times 100$$
- **FR-6.3**: Assign automated status flags:
  - **Normal**: $\text{Utilization} < 85\%$
  - **Warning**: $85\% \le \text{Utilization} \le 100\%$
  - **Breach**: $\text{Utilization} > 100\%$

#### FR-7: Portfolio Macroeconomic Stress-Testing Engine (RMG Mandate)
- **FR-7.1**: Apply macroeconomic shocks (Recession, Rate Hike Cycle, Margin Squeeze) to baseline PD and LGD figures across corporate sectors.
- **FR-7.2**: Compute stressed portfolio loss deltas ($\Delta EL = EL_{\text{Stressed}} - EL_{\text{Baseline}}$) and output post-stress rating downgrade matrices.

#### FR-8: Executive Dashboard
- **FR-8.1**: Interactive Streamlit risk console displaying SEC fundamental analysis, internal rating distribution, breach monitoring tables, model validation performance curves, and macro stress test results.

---

### 4. Key Performance Indicators (KPIs) & Validation Thresholds

| Metric Class | Metric Name | Operational Target | Recalibration Trigger |
| :--- | :--- | :--- | :--- |
| **Discrimination** | ROC-AUC | $\ge 0.80$ | $< 0.72$ |
| **Discrimination** | Gini Coefficient | $\ge 0.60$ | $< 0.44$ |
| **Separation** | KS Statistic | $\ge 45.0\%$ | $< 35.0\%$ |
| **Calibration** | Brier Score | $\le 0.08$ | $> 0.15$ |
| **Stability** | Population Stability (PSI) | $< 0.10$ (Stable) | $> 0.25$ (Action Req.) |
