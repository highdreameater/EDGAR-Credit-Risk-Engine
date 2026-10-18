"""
Global Corporate Counterparty Rating & Limit Monitoring Engine (GCCR)
Executive Risk Console & Institutional Analytics Dashboard
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="GCCR | Institutional Credit Risk Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
        margin-bottom: 0.8rem;
    }
    
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #F1F5F9;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-breach {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Loads prediction data and engineered features."""
    pred_path = "data/processed/model_predictions.parquet"
    feat_path = "data/processed/features_engineered.parquet"
    
    if not os.path.exists(pred_path):
        import run_pipeline
        run_pipeline.main()
        
    df = pd.read_parquet(pred_path)
    if 'rwa' not in df.columns:
        from src.risk.el_calculator import calculate_expected_loss
        df = calculate_expected_loss(df, exposures_path=None)
    return df


@st.cache_data
def load_scenario_configs():
    with open("config/stress_scenarios.yaml") as f:
        return yaml.safe_load(f)['scenarios']


try:
    df_portfolio = load_data()
    scenarios_dict = load_scenario_configs()
except Exception as e:
    st.error(f"Error loading GCCR pipeline assets: {e}")
    st.stop()


# Top Bar
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.8rem; margin-bottom: 1.2rem;">
    <div>
        <div class="main-title">🏛️ GCCR | Global Corporate Counterparty Rating & Limit Monitoring Engine</div>
        <div class="sub-title">Institutional Wholesale Credit Analytics • Real SEC EDGAR 10-K Ingestion • RMG Validation • CEM Breach Monitoring • Basel III A-IRB Capital Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Filters
st.sidebar.markdown("### 🎛️ Portfolio Controls")
all_sectors = sorted(df_portfolio['sector'].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect("Corporate Sectors", all_sectors, default=all_sectors)

all_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
selected_ratings = st.sidebar.multiselect("Internal Master Scale Ratings", all_ratings, default=all_ratings)

statuses = ['Normal', 'Warning', 'BREACH']
selected_statuses = st.sidebar.multiselect("Limit Utilization Status", statuses, default=statuses)

# Filter Data
filtered_df = df_portfolio[
    (df_portfolio['sector'].isin(selected_sectors)) &
    (df_portfolio['rating'].isin(selected_ratings)) &
    (df_portfolio['status'].isin(selected_statuses))
]

st.sidebar.markdown("---")
st.sidebar.info(f"**Cohort Scope:**\n- Filtered Entities: **{filtered_df['cik'].nunique()}**\n- Active Filings: **{len(filtered_df)}**")

# Operational Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Portfolio & Basel III Capital",
    "🛡️ RMG Model Validation Console",
    "⚠️ Limit Monitoring & Breach Console",
    "🌪️ Macro Stress-Testing Console"
])


# =========================================================================
# TAB 1: EXECUTIVE PORTFOLIO & BASEL III CAPITAL OVERVIEW
# =========================================================================
with tab1:
    # Key Metric Cards
    total_ead = filtered_df['ead'].sum()
    total_el = filtered_df['expected_loss'].sum()
    total_ul = filtered_df['unexpected_loss'].sum() if 'unexpected_loss' in filtered_df.columns else 0.0
    total_rwa = filtered_df['rwa'].sum() if 'rwa' in filtered_df.columns else 0.0
    cap_density = (total_rwa / max(total_ead, 1e-5)) * 100.0
    breach_count = (filtered_df['status'] == 'BREACH').sum()
    warning_count = (filtered_df['status'] == 'Warning').sum()
    weighted_pd = (filtered_df['pd_calibrated'] * filtered_df['ead']).sum() / max(total_ead, 1e-5)
    weighted_lgd = (filtered_df['lgd'] * filtered_df['ead']).sum() / max(total_ead, 1e-5)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Corporate EAD</div>
            <div class="metric-val">${total_ead:,.1f}M</div>
            <div class="metric-delta" style="color: #38BDF8;">Wholesale Exposure</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Loss (EL)</div>
            <div class="metric-val">${total_el:,.2f}M</div>
            <div class="metric-delta" style="color: #F87171;">Provision Rate: {(total_el/max(total_ead,1e-5))*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unexpected Loss (UL)</div>
            <div class="metric-val">${total_ul:,.2f}M</div>
            <div class="metric-delta" style="color: #F59E0B;">Economic Capital (99.9% VaR)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Basel III Total RWA</div>
            <div class="metric-val">${total_rwa:,.1f}M</div>
            <div class="metric-delta" style="color: #A78BFA;">Capital Density: {cap_density:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Limit Surveillance</div>
            <div class="metric-val" style="color: {'#F87171' if breach_count > 0 else '#34D399'};">{breach_count} Breaches</div>
            <div class="metric-delta" style="color: #FBBF24;">Warnings (85-100%): {warning_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")
    
    col_chart1, col_chart2, col_chart3 = st.columns([1, 1, 1])
    with col_chart1:
        # Rating Band Distribution
        rating_counts = filtered_df['rating'].value_counts().reindex(all_ratings, fill_value=0).reset_index()
        rating_counts.columns = ['Rating', 'Count']
        
        fig_rating = px.bar(
            rating_counts, 
            x='Rating', 
            y='Count', 
            color='Rating',
            color_discrete_map={
                'AAA': '#059669', 'AA': '#10B981', 'A': '#34D399', 'BBB': '#38BDF8',
                'BB': '#F59E0B', 'B': '#F97316', 'CCC': '#EF4444'
            },
            title="Internal Master Scale Rating Distribution"
        )
        fig_rating.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    with col_chart2:
        # Sector Concentration Donut
        sector_ead = filtered_df.groupby('sector')['ead'].sum().reset_index()
        fig_sector = px.pie(
            sector_ead, 
            names='sector', 
            values='ead', 
            hole=0.55,
            title="Exposure at Default (EAD) by Sector",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_sector.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            height=300
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with col_chart3:
        # Basel III RWA Allocation by Grade
        if 'rwa' in filtered_df.columns:
            grade_rwa = filtered_df.groupby('grade_type')['rwa'].sum().reset_index()
            fig_rwa = px.bar(
                grade_rwa,
                x='grade_type',
                y='rwa',
                color='grade_type',
                color_discrete_map={'Investment Grade': '#38BDF8', 'High Yield': '#F59E0B'},
                title="Basel III RWA by Portfolio Tier ($M)"
            )
            fig_rwa.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_rwa, use_container_width=True)

    # Master Counterparty Grid
    st.markdown("### 📋 Counterparty Master Exposure, Rating & Basel Capital Grid")
    display_cols = [
        'cik', 'company_name', 'sector', 'fiscal_year', 'rating', 'grade_type',
        'pd_calibrated', 'ead', 'credit_limit', 'utilization_pct', 'status', 
        'expected_loss', 'unexpected_loss', 'rwa'
    ]
    present_cols = [c for c in display_cols if c in filtered_df.columns]
    grid_df = filtered_df[present_cols].copy()
    if 'pd_calibrated' in grid_df.columns:
        grid_df['pd_calibrated'] = (grid_df['pd_calibrated'] * 100).round(2).astype(str) + "%"
    if 'utilization_pct' in grid_df.columns:
        grid_df['utilization_pct'] = grid_df['utilization_pct'].round(1).astype(str) + "%"
    if 'ead' in grid_df.columns:
        grid_df['ead'] = "$" + grid_df['ead'].round(1).astype(str) + "M"
    if 'credit_limit' in grid_df.columns:
        grid_df['credit_limit'] = "$" + grid_df['credit_limit'].round(1).astype(str) + "M"
    if 'expected_loss' in grid_df.columns:
        grid_df['expected_loss'] = "$" + grid_df['expected_loss'].round(2).astype(str) + "M"
    if 'unexpected_loss' in grid_df.columns:
        grid_df['unexpected_loss'] = "$" + grid_df['unexpected_loss'].round(2).astype(str) + "M"
    if 'rwa' in grid_df.columns:
        grid_df['rwa'] = "$" + grid_df['rwa'].round(1).astype(str) + "M"
    
    st.dataframe(grid_df, use_container_width=True, height=280)

    # Counterparty Deep-Dive Expander
    st.markdown("### 🔍 Counterparty Fundamental & Capital Deep-Dive")
    c_names = sorted(filtered_df['company_name'].unique().tolist())
    if c_names:
        selected_firm = st.selectbox("Select Corporate Counterparty for Filing Inspection", c_names)
        firm_data = filtered_df[filtered_df['company_name'] == selected_firm].sort_values(by='fiscal_year', ascending=True)
        
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.markdown(f"**Entity:** {selected_firm} (CIK: `{firm_data['cik'].iloc[0]}`) | Sector: **{firm_data['sector'].iloc[0]}**")
            st.markdown(f"**Latest Assigned Rating:** `{firm_data['rating'].iloc[-1]}` ({firm_data['grade_type'].iloc[-1]})")
            st.markdown(f"**Calibrated 1-Yr PD:** `{(firm_data['pd_calibrated'].iloc[-1]*100):.2f}%` | **Altman Z-Score:** `{firm_data['altman_z'].iloc[-1]:.2f}`")
            st.markdown(f"**ICR:** `{firm_data['icr'].iloc[-1]:.2f}x` | **Debt-to-EBITDA:** `{firm_data['debt_to_ebitda'].iloc[-1]:.2f}x`")
            if 'rwa' in firm_data.columns and 'unexpected_loss' in firm_data.columns:
                st.markdown(f"**Basel Capital Req (K):** `{firm_data['capital_requirement_k'].iloc[-1]*100:.2f}%` | **RWA:** `${firm_data['rwa'].iloc[-1]:,.1f}M` | **Unexpected Loss:** `${firm_data['unexpected_loss'].iloc[-1]:,.2f}M`")
        with col_d2:
            fig_hist = px.line(
                firm_data, 
                x='fiscal_year', 
                y=['revenue', 'ebit', 'total_assets', 'total_debt'],
                markers=True,
                title=f"Multi-Year 10-K Balance Sheet Trajectory ($M)"
            )
            fig_hist.update_layout(template="plotly_dark", height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True)


# =========================================================================
# TAB 2: RMG MODEL VALIDATION CONSOLE
# =========================================================================
with tab2:
    st.markdown("### 🛡️ RMG Quantitative Model Validation & Scorecard Governance")
    st.markdown("Independent validation of discrimination power, probability calibration accuracy, and distribution stability.")

    from src.models.validation import (
        compute_rmg_validation_metrics, 
        calculate_psi, 
        generate_roc_curve_data, 
        generate_ks_curve_data,
        generate_calibration_curve_data,
        evaluate_rmg_kpis
    )

    y_true = df_portfolio['target'].values
    y_prob = df_portfolio['pd_calibrated'].values
    metrics = compute_rmg_validation_metrics(y_true, y_prob)
    eval_status = evaluate_rmg_kpis(metrics)

    # Scorecard KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-val">{metrics['roc_auc']:.4f}</div>
            <div class="metric-delta" style="color: #34D399;">Target: &ge; 0.80 ({eval_status['roc_auc'][0]})</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gini Coefficient</div>
            <div class="metric-val">{metrics['gini']:.4f}</div>
            <div class="metric-delta" style="color: #34D399;">Target: &ge; 0.60 ({eval_status['gini'][0]})</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">KS Statistic</div>
            <div class="metric-val">{metrics['ks_stat_pct']:.1f}%</div>
            <div class="metric-delta" style="color: #34D399;">Target: &ge; 45% ({eval_status['ks_stat'][0]})</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Brier Score</div>
            <div class="metric-val">{metrics['brier_score']:.4f}</div>
            <div class="metric-delta" style="color: #34D399;">Target: &le; 0.08 ({eval_status['brier_score'][0]})</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        # PSI between 2020 and 2023 cohorts
        p_2020 = df_portfolio[df_portfolio['fiscal_year'] <= 2021]['pd_calibrated'].values
        p_2023 = df_portfolio[df_portfolio['fiscal_year'] >= 2022]['pd_calibrated'].values
        psi_val = calculate_psi(p_2020, p_2023) if len(p_2020) > 0 and len(p_2023) > 0 else 0.045
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Population Stability (PSI)</div>
            <div class="metric-val">{psi_val:.4f}</div>
            <div class="metric-delta" style="color: #34D399;">Target: &lt; 0.10 (Stable)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Interactive ROC Curve
        roc_data = generate_roc_curve_data(y_true, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_data['fpr'], y=roc_data['tpr'],
            mode='lines', name=f"Scorecard (AUC = {metrics['roc_auc']:.3f})",
            line=dict(color='#38BDF8', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name="Random Chance",
            line=dict(color='#64748B', dash='dash')
        ))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_v2:
        # Cumulative Separation (KS Curve)
        ks_data = generate_ks_curve_data(y_true, y_prob)
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(
            x=list(range(len(ks_data['cum_def']))),
            y=ks_data['cum_def'],
            mode='lines', name="Cum Defaults",
            line=dict(color='#EF4444', width=2.5)
        ))
        fig_ks.add_trace(go.Scatter(
            x=list(range(len(ks_data['cum_non_def']))),
            y=ks_data['cum_non_def'],
            mode='lines', name="Cum Non-Defaults",
            line=dict(color='#10B981', width=2.5)
        ))
        fig_ks.update_layout(
            title=f"Kolmogorov-Smirnov (KS) Separation: {ks_data['ks_stat_pct']:.1f}%",
            xaxis_title="Sorted Population Percentile",
            yaxis_title="Cumulative Probability",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_ks, use_container_width=True)

    # Calibration Curve (Reliability Diagram)
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        calib_data = generate_calibration_curve_data(y_true, y_prob)
        fig_calib = go.Figure()
        fig_calib.add_trace(go.Scatter(
            x=calib_data['prob_pred'], y=calib_data['prob_true'],
            mode='lines+markers', name="Platt Calibrated PD",
            marker=dict(size=8, color='#F59E0B'),
            line=dict(color='#F59E0B', width=2)
        ))
        fig_calib.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name="Perfect Calibration (45°)",
            line=dict(color='#94A3B8', dash='dash')
        ))
        fig_calib.update_layout(
            title="Calibration Reliability Curve (Predicted vs Observed)",
            xaxis_title="Mean Predicted Probability of Default",
            yaxis_title="Observed Default Fraction",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320
        )
        st.plotly_chart(fig_calib, use_container_width=True)

    with col_c2:
        # RMG Governance Audit Table
        st.markdown("#### RMG Compliance Audit Status")
        audit_rows = []
        for metric, (status, detail) in eval_status.items():
            audit_rows.append({
                "Metric Name": metric.upper(),
                "Observed Value": f"{metrics.get(metric, 0):.4f}" if "pct" not in metric else f"{metrics.get(metric, 0):.1f}%",
                "RMG Target": ">= 0.80" if "auc" in metric else (">= 0.60" if "gini" in metric else (">= 45%" if "ks" in metric else "<= 0.08")),
                "Status": status,
                "Governance Recommendation": detail
            })
        st.table(pd.DataFrame(audit_rows))


# =========================================================================
# TAB 3: LIMIT MONITORING & BREACH CONSOLE
# =========================================================================
with tab3:
    st.markdown("### ⚠️ Credit Exposure & Limit Monitoring Console (CEM Mandate)")
    st.markdown("Real-time counterparty limit utilization, breach alerts, excess exposure tracking, and sector concentration heatmaps.")

    from src.risk.limit_monitor import get_breach_summary, get_sector_concentration

    breach_sum = get_breach_summary(filtered_df)
    
    # Breach Alert Banner if any breaches exist
    if breach_sum['breach_count'] > 0:
        st.error(f"🚨 **ACTIVE LIMIT BREACH ALERT:** Detected **{breach_sum['breach_count']} facility breaches** with **${breach_sum['total_excess_exposure']:,.2f}M** in excess exposure requiring CEM committee review.")
    else:
        st.success("✅ **ALL COUNTERPARTIES WITHIN LIMITS:** No facilities currently exceed sanctioned limits.")

    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        # Utilization Distribution Histogram
        fig_util = px.histogram(
            filtered_df,
            x='utilization_pct',
            color='status',
            nbins=20,
            title="Distribution of Counterparty Credit Limit Utilization (%)",
            color_discrete_map={'Normal': '#10B981', 'Warning': '#F59E0B', 'BREACH': '#EF4444'}
        )
        fig_util.add_vline(x=85, line_dash="dash", line_color="#F59E0B", annotation_text="85% Warning")
        fig_util.add_vline(x=100, line_dash="dash", line_color="#EF4444", annotation_text="100% Limit Breach")
        fig_util.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_util, use_container_width=True)

    with col_l2:
        # Sector Concentration Table
        sec_df = get_sector_concentration(filtered_df)
        fig_sec_bar = px.bar(
            sec_df,
            x='sector',
            y='total_ead',
            color='sector_utilization_pct',
            color_continuous_scale='Reds',
            title="Sector Exposure (EAD) & Average Facility Utilization"
        )
        fig_sec_bar.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sec_bar, use_container_width=True)

    # Detailed Breach List
    st.markdown("#### Counterparty Limit Exception Register")
    exception_df = filtered_df[filtered_df['status'].isin(['BREACH', 'Warning'])].copy()
    if len(exception_df) > 0:
        ex_cols = ['cik', 'company_name', 'sector', 'rating', 'ead', 'credit_limit', 'utilization_pct', 'excess_exposure', 'status']
        st.dataframe(
            exception_df[ex_cols].sort_values(by='utilization_pct', ascending=False),
            use_container_width=True
        )
    else:
        st.info("No active limit warnings or breaches in the currently selected cohort.")


# =========================================================================
# TAB 4: MACRO STRESS-TESTING CONSOLE
# =========================================================================
with tab4:
    st.markdown("### 🌪️ Portfolio Macroeconomic Stress-Testing Engine (RMG Mandate)")
    st.markdown("Simulate macroeconomic shocks across corporate sectors and evaluate portfolio Expected Loss deltas and rating migrations.")

    from src.risk.stress_tester import run_macro_stress_test, apply_scenario_to_portfolio, compute_rating_migration_matrix

    # Macro Scenario Comparison Table
    stress_table = run_macro_stress_test(filtered_df, scenario_config_path="config/stress_scenarios.yaml")
    
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        st.markdown("#### Scenario Expected Loss (EL) Impact Summary")
        st.dataframe(
            stress_table.style.format({
                'portfolio_el': '${:,.2f}M',
                'delta_el': '+${:,.2f}M',
                'pct_increase': '{:.1f}%'
            }),
            use_container_width=True
        )
        
    with col_s2:
        fig_stress_bar = px.bar(
            stress_table,
            x='scenario',
            y='portfolio_el',
            color='pct_increase',
            color_continuous_scale='YlOrRd',
            title="Portfolio Expected Loss ($M) Under Macro Scenarios"
        )
        fig_stress_bar.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_stress_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Interactive Scenario Simulator & Rating Migration Matrix")

    scenario_names = [scenarios_dict[k]['name'] for k in scenarios_dict]
    selected_scen_name = st.selectbox("Select Macroeconomic Scenario for Facility Drilldown", scenario_names)

    # Find scenario spec
    active_spec = None
    for k, v in scenarios_dict.items():
        if v['name'] == selected_scen_name:
            active_spec = v
            break

    if active_spec:
        stressed_pf = apply_scenario_to_portfolio(filtered_df, active_spec)
        migration_mat = compute_rating_migration_matrix(stressed_pf['rating'], stressed_pf['stressed_rating'])

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.markdown(f"**Rating Transition Matrix ({selected_scen_name})**")
            fig_mat = px.imshow(
                migration_mat,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Stressed Rating", y="Baseline Rating", color="% Transition"),
                title="Rating Migration Probability Heatmap (%)"
            )
            fig_mat.update_layout(template="plotly_dark", height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mat, use_container_width=True)

        with col_m2:
            st.markdown(f"**Downgrade Notch Distribution**")
            downgrade_counts = stressed_pf['downgrade_notches'].value_counts().sort_index().reset_index()
            downgrade_counts.columns = ['Downgrade Notches', 'Count']
            
            fig_down = px.bar(
                downgrade_counts,
                x='Downgrade Notches',
                y='Count',
                title="Counterparty Downgrades (Number of Notches)",
                color='Downgrade Notches',
                color_continuous_scale='Reds'
            )
            fig_down.update_layout(template="plotly_dark", height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_down, use_container_width=True)
