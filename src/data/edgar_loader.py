"""
SEC EDGAR API & Kaggle Dataset Ingestion Module for GCCR.
Supports live querying of SEC EDGAR company facts via XBRL REST API
as well as ingestion and joining of historical corporate fundamentals and defaults.
"""

import os
import time
import requests
import pandas as pd
from typing import Optional, Dict, Any


SEC_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_USER_AGENT = "InstitutionalCreditAnalyticsRiskLab/1.0 (contact: risk-analytics@institutional-gccr.org)"


def load_sec_financials(filepath: str = "data/raw/edgar_sec_financials.csv") -> pd.DataFrame:
    """
    Loads raw SEC EDGAR financial statement data from CSV.
    Ensures CIK formatting is standardized.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SEC Financials file not found at: {filepath}")
    
    df = pd.read_csv(filepath, dtype={"cik": str})
    # Standardize CIK to 10-digit zero-padded string
    df['cik'] = df['cik'].astype(str).str.zfill(10)
    return df


def load_corporate_defaults(filepath: str = "data/raw/corporate_defaults.csv") -> pd.DataFrame:
    """
    Loads corporate default histories and distress events from CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Corporate defaults file not found at: {filepath}")
    
    df = pd.read_csv(filepath, dtype={"cik": str})
    df['cik'] = df['cik'].astype(str).str.zfill(10)
    return df


def merge_financials_with_defaults(
    financials_df: pd.DataFrame, 
    defaults_df: Optional[pd.DataFrame] = None,
    defaults_filepath: str = "data/raw/corporate_defaults.csv"
) -> pd.DataFrame:
    """
    Merges financial statement line items with 1-year forward default/distress labels.
    If defaults_df is not provided, attempts to load from defaults_filepath.
    """
    if defaults_df is None:
        if os.path.exists(defaults_filepath):
            defaults_df = load_corporate_defaults(defaults_filepath)
        else:
            defaults_df = pd.DataFrame(columns=['cik', 'fiscal_year', 'default_flag'])

    merged = pd.merge(
        financials_df,
        defaults_df[['cik', 'fiscal_year', 'default_flag', 'default_event_type']],
        on=['cik', 'fiscal_year'],
        how='left'
    )
    
    # Fill missing default flags with 0 (assumed performing if no default record)
    merged['default_flag'] = merged['default_flag'].fillna(0).astype(int)
    if 'default_event_type' in merged.columns:
        merged['default_event_type'] = merged['default_event_type'].fillna('None')
        
    # Standardize target column name
    merged['target'] = merged['default_flag']
    return merged


def fetch_edgar_company_facts(
    cik: str, 
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Queries SEC EDGAR XBRL company facts API for a given corporate CIK.
    SEC guidelines require a declared User-Agent with contact email.
    """
    cik_padded = str(cik).strip().zfill(10)
    url = SEC_FACTS_URL_TEMPLATE.format(cik=cik_padded)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"error": f"CIK {cik_padded} not found on SEC EDGAR", "status_code": 404}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}", "status_code": response.status_code}
    except Exception as e:
        return {"error": str(e), "status_code": 500}


def parse_sec_facts_to_series(facts_json: Dict[str, Any], fiscal_year: int) -> Dict[str, float]:
    """
    Parses key US-GAAP accounting tags from SEC EDGAR facts payload for a given fiscal year.
    Extracts Balance Sheet, Income Statement, and Cash Flow items.
    """
    if "facts" not in facts_json or "us-gaap" not in facts_json["facts"]:
        return {}
    
    us_gaap = facts_json["facts"]["us-gaap"]
    
    def extract_val(tag_candidates):
        for tag in tag_candidates:
            if tag in us_gaap and "units" in us_gaap[tag]:
                units = us_gaap[tag]["units"]
                # Typically USD
                for unit_key in ["USD", "pure"]:
                    if unit_key in units:
                        entries = units[unit_key]
                        # Filter by fy and 10-K form
                        for entry in reversed(entries):
                            if entry.get("fy") == fiscal_year and entry.get("form") in ["10-K", "10-K/A"]:
                                val = entry.get("val")
                                if val is not None:
                                    return float(val) / 1e6  # Return in $M
        return None

    total_assets = extract_val(["Assets", "AssetsCurrentAndNoncurrent"])
    total_liab = extract_val(["Liabilities", "LiabilitiesCurrentAndNoncurrent"])
    revenues = extract_val(["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"])
    operating_income = extract_val(["OperatingIncomeLoss"])
    operating_cf = extract_val(["NetCashProvidedByUsedInOperatingActivities"])
    current_assets = extract_val(["AssetsCurrent"])
    current_liab = extract_val(["LiabilitiesCurrent"])
    retained_earnings = extract_val(["RetainedEarningsAccumulatedDeficit"])

    return {
        "total_assets": total_assets,
        "total_liabilities": total_liab,
        "revenue": revenues,
        "ebit": operating_income,
        "operating_cf": operating_cf,
        "current_assets": current_assets,
        "current_liab": current_liab,
        "retained_earning": retained_earnings
    }
