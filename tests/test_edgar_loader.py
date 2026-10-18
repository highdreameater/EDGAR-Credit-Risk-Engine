"""
Unit tests for SEC EDGAR Ingestion, Ticker Resolution, and 10-K Parsing Engine.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from src.data.edgar_loader import (
    resolve_ticker_or_cik,
    extract_concept_value,
    KNOWN_TICKERS,
    fetch_and_parse_live_filing
)


def test_resolve_ticker_or_cik():
    # Test known ticker
    cik, ticker, name, sector = resolve_ticker_or_cik("AAPL")
    assert cik == "0000320193"
    assert ticker == "AAPL"
    assert "Apple" in name
    assert sector == "Technology"

    # Test lowercase
    cik_ba, tick_ba, name_ba, sec_ba = resolve_ticker_or_cik("ba")
    assert cik_ba == "0000012927"
    assert tick_ba == "BA"
    assert sec_ba == "Industrials"

    # Test resolution by CIK string
    cik_f, tick_f, name_f, sec_f = resolve_ticker_or_cik("0000037996")
    assert cik_f == "0000037996"
    assert tick_f == "F"

    # Test fallback for unknown ticker
    cik_unk, tick_unk, name_unk, sec_unk = resolve_ticker_or_cik("XYZCORP")
    assert tick_unk == "XYZCORP"
    assert sec_unk == "Wholesale Corporate"


def test_extract_concept_value_from_mock():
    mock_gaap = {
        "Assets": {
            "units": {
                "USD": [
                    {"fy": 2022, "form": "10-K", "val": 350000000000},  # $350,000M
                    {"fy": 2023, "form": "10-K", "val": 380000000000}   # $380,000M
                ]
            }
        }
    }
    
    val_2023 = extract_concept_value(mock_gaap, ["Assets"], 2023)
    val_2022 = extract_concept_value(mock_gaap, ["Assets"], 2022)
    val_missing = extract_concept_value(mock_gaap, ["Assets"], 2018)
    
    assert val_2023 == 380000.0
    assert val_2022 == 350000.0
    assert val_missing is None


def test_fetch_and_parse_live_filing_mocked():
    mock_response = {
        "entityName": "Mock Aerospace Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {"USD": [{"fy": 2023, "form": "10-K", "val": 100000000000}]}
                },
                "Liabilities": {
                    "units": {"USD": [{"fy": 2023, "form": "10-K", "val": 60000000000}]}
                },
                "Revenues": {
                    "units": {"USD": [{"fy": 2023, "form": "10-K", "val": 80000000000}]}
                },
                "OperatingIncomeLoss": {
                    "units": {"USD": [{"fy": 2023, "form": "10-K", "val": 12000000000}]}
                }
            }
        }
    }
    
    with patch("src.data.edgar_loader.fetch_edgar_company_facts", return_value=mock_response):
        profile = fetch_and_parse_live_filing("BA", requested_year=2023)
        assert profile["company_name"] == "Mock Aerospace Inc."
        assert profile["total_assets"] == 100000.0
        assert profile["revenue"] == 80000.0
        assert profile["ebit"] == 12000.0
        assert profile["fiscal_year"] == 2023
