import os
import sys
import pandas as pd
from src.data.edgar_loader import load_sec_financials
from src.features.financial_ratios import compute_financial_ratios
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import compute_rmg_validation_metrics
from src.risk.rating_engine import assign_master_scale_ratings
from src.risk.el_calculator import calculate_expected_loss
from src.risk.limit_monitor import evaluate_limit_utilization
from src.risk.stress_tester import run_macro_stress_test

def main():
    os.makedirs("data/processed", exist_ok=True)

    print("=== [1/6] Ingesting SEC EDGAR Corporate Financial Filings ===")
    df_raw = load_sec_financials("data/raw/edgar_sec_financials.csv")
    print(f"Loaded {len(df_raw)} filings across {df_raw['cik'].nunique()} unique corporate entities.")
    
    print("=== [2/6] Engineering Fundamental Leverage & Solvency Ratios ===")
    df_features = compute_financial_ratios(df_raw)
    df_features.to_parquet("data/processed/features_engineered.parquet")
    print("Saved engineered features to data/processed/features_engineered.parquet")
    
    print("=== [3/6] Scoring & Calibrating Probability of Default (PD) ===")
    df_scored, pd_model = train_and_calibrate_pd_model(df_features)
    print(f"Model Training Complete ({pd_model.algorithm.upper()}). Calibrated PD range: [{df_scored['pd_calibrated'].min():.4f}, {df_scored['pd_calibrated'].max():.4f}]")
    
    print("=== [4/6] Executing RMG Quantitative Model Validation Suite ===")
    metrics = compute_rmg_validation_metrics(df_scored['target'], df_scored['pd_calibrated'])
    print(f"Validation -> ROC-AUC: {metrics['roc_auc']:.4f} | Gini: {metrics['gini']:.4f} | KS: {metrics['ks_stat_pct']:.2f}% | Brier: {metrics['brier_score']:.4f}")
    
    print("=== [5/6] Rating Assignment, EL & Basel III Capital/RWA Calculation ===")
    df_rated = assign_master_scale_ratings(df_scored, config_path="config/rating_bands.yaml")
    df_el = calculate_expected_loss(df_rated, exposures_path="data/portfolio/exposures_limits.csv")
    df_final = evaluate_limit_utilization(df_el)
    
    from src.risk.el_calculator import aggregate_portfolio_metrics
    summary = aggregate_portfolio_metrics(df_final)
    print(f"Basel III A-IRB -> Portfolio EAD: ${summary['total_ead']:,.1f}M | Expected Loss (EL): ${summary['total_expected_loss']:,.2f}M | Unexpected Loss (UL): ${summary['total_unexpected_loss']:,.2f}M | Total RWA: ${summary['total_rwa']:,.1f}M (Density: {summary['capital_density_pct']:.1f}%)")
    
    # Save Pipeline Outputs
    df_final.to_parquet("data/processed/model_predictions.parquet")
    print("Saved pipeline outputs to data/processed/model_predictions.parquet")
    
    print("=== [6/6] Executing Macro Stress-Testing Engine ===")
    stress_results = run_macro_stress_test(df_final, scenario_config_path="config/stress_scenarios.yaml")
    print(stress_results.to_string(index=False))

if __name__ == "__main__":
    main()
