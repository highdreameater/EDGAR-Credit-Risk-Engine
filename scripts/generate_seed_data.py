"""
Script to generate realistic institutional SEC EDGAR corporate financial filings,
historical defaults, and counterparty portfolio exposure limits for GCCR.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Directory creation
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/portfolio", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Curated list of well-known public corporations across key sectors
COMPANIES = [
    # Technology
    ("0000320193", "Apple Inc.", "Technology", 0.005),
    ("0000789019", "Microsoft Corp.", "Technology", 0.004),
    ("0001018724", "Amazon.com Inc.", "Technology", 0.012),
    ("0001652044", "Alphabet Inc.", "Technology", 0.005),
    ("0001326801", "Meta Platforms Inc.", "Technology", 0.010),
    ("0001045810", "NVIDIA Corp.", "Technology", 0.008),
    ("0000050863", "Intel Corp.", "Technology", 0.045),
    ("0000804212", "Cisco Systems Inc.", "Technology", 0.015),
    ("0001108524", "Salesforce Inc.", "Technology", 0.020),
    ("0000909832", "Micron Technology Inc.", "Technology", 0.038),
    ("0001732845", "Fastly Inc.", "Technology", 0.120),
    ("0001585521", "Snap Inc.", "Technology", 0.095),
    
    # Industrials & Aerospace
    ("0000012927", "Boeing Co.", "Industrials", 0.075),
    ("0000040545", "General Electric Co.", "Industrials", 0.035),
    ("0000018230", "Caterpillar Inc.", "Industrials", 0.022),
    ("0000055785", "Kimberly-Clark Corp.", "Industrials", 0.018),
    ("0000093410", "Chevron Phillips Chemical", "Industrials", 0.028),
    ("0000352541", "Lockheed Martin Corp.", "Industrials", 0.012),
    ("0000034088", "Exxon Mobil Corp.", "Energy", 0.015),
    ("0000093410", "Chevron Corp.", "Energy", 0.014),
    ("0000858877", "ConocoPhillips", "Energy", 0.025),
    ("0000088698", "Schlumberger Ltd.", "Energy", 0.038),
    ("0001163165", "Concho Resources (Distressed)", "Energy", 0.220),
    ("0001438731", "Whiting Petroleum (Historic)", "Energy", 0.350),
    
    # Consumer Goods & Retail
    ("0000021344", "The Coca-Cola Co.", "Consumer Goods", 0.008),
    ("0000077476", "PepsiCo Inc.", "Consumer Goods", 0.009),
    ("0000104169", "Walmart Inc.", "Consumer Goods", 0.012),
    ("0000027419", "Target Corp.", "Consumer Goods", 0.025),
    ("0000723254", "Cintas Corp.", "Consumer Goods", 0.010),
    ("0000886158", "Bed Bath & Beyond (Distressed)", "Consumer Goods", 0.480),
    ("0000088525", "Sears Holdings (Historic)", "Consumer Goods", 0.520),
    ("0000037996", "Ford Motor Co.", "Consumer Goods", 0.055),
    ("0001467858", "General Motors Co.", "Consumer Goods", 0.048),
    ("0001318605", "Tesla Inc.", "Consumer Goods", 0.025),
    
    # Healthcare & Pharma
    ("0000078003", "Pfizer Inc.", "Healthcare", 0.010),
    ("0000200406", "Johnson & Johnson", "Healthcare", 0.004),
    ("0000064978", "Merck & Co. Inc.", "Healthcare", 0.008),
    ("0000001800", "Abbott Laboratories", "Healthcare", 0.009),
    ("0000731766", "UnitedHealth Group Inc.", "Healthcare", 0.008),
    ("0001027975", "Mallinckrodt PLC (Distressed)", "Healthcare", 0.420),
    
    # Telecom & Media
    ("0000732717", "AT&T Inc.", "Telecom", 0.035),
    ("0000732718", "Verizon Communications Inc.", "Telecom", 0.030),
    ("0001283699", "T-Mobile US Inc.", "Telecom", 0.028),
    ("0000051143", "Frontier Communications (Distressed)", "Telecom", 0.380),
    
    # Utilities
    ("0000753308", "NextEra Energy Inc.", "Utilities", 0.015),
    ("0000030554", "Duke Energy Corp.", "Utilities", 0.022),
    ("0000092230", "Southern Co.", "Utilities", 0.024),
    ("0001004980", "PG&E Corp. (Historic Distressed)", "Utilities", 0.280),
    
    # Basic Materials
    ("0000029989", "Dow Inc.", "Basic Materials", 0.032),
    ("0000034903", "Freeport-McMoRan Inc.", "Basic Materials", 0.042),
    ("0000071695", "Newmont Corp.", "Basic Materials", 0.025),
    ("0000072888", "United States Steel Corp.", "Basic Materials", 0.065),
    ("0001158449", "AK Steel Corp. (Distressed)", "Basic Materials", 0.260),
]

# Additional synthetic corporate cohort to reach a solid institutional universe (approx 80 companies x 5 years = 400 observations)
SECTORS = ["Technology", "Industrials", "Consumer Goods", "Energy", "Healthcare", "Telecom", "Utilities", "Basic Materials"]
for i in range(1, 35):
    sector = SECTORS[i % len(SECTORS)]
    is_distressed = (i % 7 == 0)
    base_distress = np.random.uniform(0.20, 0.45) if is_distressed else np.random.uniform(0.008, 0.055)
    name = f"Corp {sector} {i:02d} {'(Distressed)' if is_distressed else 'Holdings'}"
    cik = f"000199{i:04d}"
    COMPANIES.append((cik, name, sector, base_distress))

years = [2019, 2020, 2021, 2022, 2023]
financial_rows = []
default_rows = []
exposure_rows = []

for cik, name, sector, distress_prob in COMPANIES:
    # Baseline firm size
    base_assets = np.random.uniform(5000, 150000)
    if "Distressed" in name or distress_prob > 0.15:
        base_assets = np.random.uniform(1200, 15000)

    # Establish baseline balance sheet scale
    for yr in years:
        growth = 1.0 + np.random.normal(0.04, 0.08)
        cur_assets = base_assets * growth
        
        # Financial health varies with distress_prob
        is_high_risk = distress_prob > 0.15
        
        if is_high_risk:
            # Over-leveraged, low or negative margins, weak liquidity
            total_assets = max(500.0, cur_assets * np.random.uniform(0.85, 1.05))
            total_liab = total_assets * np.random.uniform(0.85, 1.25) # may have negative equity
            current_assets = total_assets * np.random.uniform(0.20, 0.40)
            current_liab = current_assets * np.random.uniform(1.10, 2.20) # current ratio < 1.0
            total_debt = total_liab * np.random.uniform(0.60, 0.85)
            revenue = total_assets * np.random.uniform(0.40, 0.80)
            ebit = revenue * np.random.uniform(-0.15, 0.03) # negative or low operating income
            ebitda = ebit + (total_assets * 0.04)
            interest_exp = max(10.0, total_debt * np.random.uniform(0.07, 0.12)) # elevated interest rate
            operating_cf = ebitda - interest_exp - (total_assets * np.random.uniform(0.02, 0.05))
            retained_earning = -total_assets * np.random.uniform(0.10, 0.45) # accumulated deficit
            market_cap = max(50.0, total_assets * np.random.uniform(0.08, 0.25))
            
            # Default trigger probability
            def_p = min(0.95, distress_prob * (1.2 if yr in [2020, 2022] else 1.0))
            is_default = 1 if np.random.rand() < def_p else 0
        else:
            # Healthy investment grade or moderate credit
            total_assets = max(2000.0, cur_assets * np.random.uniform(0.95, 1.15))
            total_liab = total_assets * np.random.uniform(0.35, 0.65)
            current_assets = total_assets * np.random.uniform(0.30, 0.55)
            current_liab = current_assets * np.random.uniform(0.40, 0.85) # current ratio > 1.2
            total_debt = total_liab * np.random.uniform(0.30, 0.60)
            revenue = total_assets * np.random.uniform(0.70, 1.40)
            ebit = revenue * np.random.uniform(0.10, 0.28) # solid margins
            ebitda = ebit + (total_assets * np.random.uniform(0.04, 0.08))
            interest_exp = max(5.0, total_debt * np.random.uniform(0.03, 0.055))
            operating_cf = ebitda - interest_exp * 0.8
            retained_earning = total_assets * np.random.uniform(0.20, 0.55)
            market_cap = max(500.0, total_assets * np.random.uniform(0.90, 2.50))
            
            # Default rate is low
            is_default = 1 if np.random.rand() < (distress_prob * 0.5) else 0

        financial_rows.append({
            "cik": cik,
            "company_name": name,
            "sector": sector,
            "fiscal_year": yr,
            "total_assets": round(float(total_assets), 2),
            "total_liabilities": round(float(total_liab), 2),
            "current_assets": round(float(current_assets), 2),
            "current_liab": round(float(current_liab), 2),
            "total_debt": round(float(total_debt), 2),
            "revenue": round(float(revenue), 2),
            "ebit": round(float(ebit), 2),
            "ebitda": round(float(ebitda), 2),
            "interest_exp": round(float(interest_exp), 2),
            "operating_cf": round(float(operating_cf), 2),
            "retained_earning": round(float(retained_earning), 2),
            "market_cap": round(float(market_cap), 2)
        })

        default_rows.append({
            "cik": cik,
            "fiscal_year": yr,
            "default_flag": is_default,
            "default_event_type": "Bankruptcy/Restructuring" if is_default else "None"
        })

    # Portfolio Exposures & Limits (CEM Mandate)
    # Assigned limit based on size and perceived risk
    base_limit = np.random.uniform(50, 800)
    # Let some counterparties have warning or breach (>85% or >100%)
    util_type = np.random.choice(["normal", "warning", "breach"], p=[0.72, 0.18, 0.10])
    if util_type == "normal":
        ead = base_limit * np.random.uniform(0.30, 0.82)
    elif util_type == "warning":
        ead = base_limit * np.random.uniform(0.86, 0.98)
    else:
        ead = base_limit * np.random.uniform(1.02, 1.35)
        
    lgd = float(np.clip(np.random.normal(0.45, 0.10), 0.15, 0.85))
    exposure_rows.append({
        "cik": cik,
        "ead": round(float(ead), 2),
        "lgd": round(float(lgd), 4),
        "credit_limit": round(float(base_limit), 2)
    })

df_fin = pd.DataFrame(financial_rows)
df_def = pd.DataFrame(default_rows)
df_exp = pd.DataFrame(exposure_rows)

df_fin.to_csv("data/raw/edgar_sec_financials.csv", index=False)
df_def.to_csv("data/raw/corporate_defaults.csv", index=False)
df_exp.to_csv("data/portfolio/exposures_limits.csv", index=False)

print(f"Generated {len(df_fin)} SEC financial records across {len(COMPANIES)} companies.")
print(f"Generated {len(df_def)} default label records with {df_def['default_flag'].sum()} default events.")
print(f"Generated {len(df_exp)} counterparty exposure records.")
