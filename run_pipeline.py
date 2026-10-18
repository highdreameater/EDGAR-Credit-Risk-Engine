import os
import sys
import json
import argparse
import pandas as pd
from src.data.edgar_loader import load_sec_financials
from src.data.preprocessor import preprocess_financial_data
from src.features.financial_ratios import compute_financial_ratios
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import compute_rmg_validation_metrics
from src.risk.rating_engine import assign_master_scale_ratings
from src.risk.el_calculator import calculate_expected_loss, aggregate_portfolio_metrics
from src.risk.limit_monitor import evaluate_limit_utilization, get_breach_summary
from src.risk.stress_tester import run_macro_stress_test


def parse_args():
    parser = argparse.ArgumentParser(description="GCCR Institutional Wholesale Credit Risk Analytics Engine")
    parser.add_argument("--data", default="data/raw/edgar_sec_financials.csv", help="Path to raw SEC financials CSV")
    parser.add_argument("--model", default="logistic_regression", choices=["logistic_regression", "xgboost"], help="Scorecard algorithm")
    parser.add_argument("--calibration", default="sigmoid", choices=["sigmoid", "isotonic"], help="PD calibration technique")
    parser.add_argument("--export-report", action="store_true", default=True, help="Export executive audit report JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=== [1/6] Ingesting & Preprocessing SEC EDGAR Corporate Filings ===")
    df_raw = load_sec_financials(args.data)
    print(f"Loaded {len(df_raw)} filings across {df_raw['cik'].nunique()} unique corporate entities.")
    df_clean = preprocess_financial_data(df_raw, winsorize=True)
    print(f"Preprocessed financials: handled NaNs via sector median imputation & winsorized 1st/99th percentiles.")
    
    print("=== [2/6] Engineering Fundamental Leverage & Solvency Ratios ===")
    df_features = compute_financial_ratios(df_clean)
    df_features.to_parquet("data/processed/features_engineered.parquet")
    print("Saved engineered features to data/processed/features_engineered.parquet")
    
    print("=== [3/6] Scoring & Calibrating Probability of Default (PD) ===")
    df_scored, pd_model = train_and_calibrate_pd_model(
        df_features, 
        algorithm=args.model, 
        calibration_method=args.calibration
    )
    print(f"Model Training Complete ({pd_model.algorithm.upper()}). Calibrated PD range: [{df_scored['pd_calibrated'].min():.4f}, {df_scored['pd_calibrated'].max():.4f}]")
    
    # Persist serialized model artifact for zero-latency inference
    model_artifact_path = "data/processed/calibrated_pd_model.joblib"
    pd_model.save(model_artifact_path)
    print(f"Persisted calibrated model artifact to {model_artifact_path}")

    print("=== [4/6] Executing RMG Quantitative Model Validation Suite ===")
    metrics = compute_rmg_validation_metrics(df_scored['target'], df_scored['pd_calibrated'])
    print(f"Validation -> ROC-AUC: {metrics['roc_auc']:.4f} | Gini: {metrics['gini']:.4f} | KS: {metrics['ks_stat_pct']:.2f}% | Brier: {metrics['brier_score']:.4f}")
    
    print("=== [5/6] Rating Assignment, EL & Basel III Capital/RWA Calculation ===")
    df_rated = assign_master_scale_ratings(df_scored, config_path="config/rating_bands.yaml")
    df_el = calculate_expected_loss(df_rated, exposures_path="data/portfolio/exposures_limits.csv")
    df_final = evaluate_limit_utilization(df_el)
    
    summary = aggregate_portfolio_metrics(df_final)
    breach_sum = get_breach_summary(df_final)
    print(f"Basel III A-IRB -> Portfolio EAD: ${summary['total_ead']:,.1f}M | Expected Loss (EL): ${summary['total_expected_loss']:,.2f}M | Unexpected Loss (UL): ${summary['total_unexpected_loss']:,.2f}M | Total RWA: ${summary['total_rwa']:,.1f}M (Density: {summary['capital_density_pct']:.1f}%)")
    print(f"Limit Surveillance -> Breaches: {breach_sum['breach_count']} | Warnings: {breach_sum['warning_count']} | Total Excess Exposure: ${breach_sum['total_excess_exposure']:,.2f}M")

    # Save Pipeline Outputs
    df_final.to_parquet("data/processed/model_predictions.parquet")
    print("Saved pipeline outputs to data/processed/model_predictions.parquet")
    
    print("=== [6/6] Executing Macro Stress-Testing Engine ===")
    stress_results = run_macro_stress_test(df_final, scenario_config_path="config/stress_scenarios.yaml")
    print(stress_results.to_string(index=False))

    if args.export_report:
        report_data = {
            "model_metadata": {
                "algorithm": args.model,
                "calibration_method": args.calibration,
                "feature_count": len(pd_model.feature_names),
                "features": pd_model.feature_names
            },
            "validation_metrics": metrics,
            "portfolio_summary": summary,
            "breach_summary": breach_sum,
            "stress_testing": stress_results.to_dict(orient="records")
        }
        with open("reports/executive_pipeline_summary.json", "w") as f:
            json.dump(report_data, f, indent=2)
        print("Generated executive audit report at reports/executive_pipeline_summary.json")


if __name__ == "__main__":
    main()
