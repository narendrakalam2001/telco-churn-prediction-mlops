# ============================================================
# MONITORING DASHBOARD — Telco Churn ML System
# ============================================================
# Sections:
#   Sidebar  : Live prediction form → POST /predict
#   Section 1: Real-Time Monitoring Alerts
#   Section 2: Champion vs Challenger — 4-gate visual metrics
#   Section 3: KPIs + Charts (action distribution, risk bands)
#   Section 4: Revenue Impact Panel (churn-specific)
#   Section 5: Feature Drift PSI bar chart
#   Section 6: Recent Predictions log
#
# UPGRADE: Section 2 now shows each gate as a st.metric block
# with ✅/❌ status — same style as Credit Risk dashboard upgrade.
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(
    page_title = "Churn Prediction Dashboard",
    layout     = "wide",
    page_icon  = "📡"
)
st.title("📡 Telco Churn Prediction — Monitoring Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "CHURN_API_URL",
    "http://127.0.0.1:8000"
) + "/predict"

# ── Alert thresholds ──────────────────────────────────────────
PSI_MODERATE      = 0.10
PSI_HIGH          = 0.20
CHURN_RATE_ALERT  = 0.35
URGENT_RATE_ALERT = 0.20

# ── Gate thresholds (mirrors config.py / model_loader.py) ────
MIN_F1_IMPROVEMENT     = 0.005
MIN_ROCAUC_THRESHOLD   = 0.80
MIN_RECALL_THRESHOLD   = 0.70
MAX_GENERALIZATION_GAP = 0.10

# ── Path resolution — Streamlit Cloud safe ────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "churn_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs",         "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "churn_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "churn_models", "challenger_log.json")
CARD_DIR        = os.path.join(BASE_DIR, "churn_models")


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔮 Predict Customer Churn")

tenure         = st.sidebar.number_input("Tenure (months)",     value=12,   min_value=0,   max_value=72)
monthly_charge = st.sidebar.number_input("Monthly Charges (₹)", value=65.0, min_value=0.0)
total_charges  = st.sidebar.number_input("Total Charges (₹)",   value=780.0,min_value=0.0)
contract       = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])
internet       = st.sidebar.selectbox(
    "Internet Service", ["Fiber optic", "DSL", "No"])
online_sec     = st.sidebar.selectbox(
    "Online Security", ["No", "Yes", "No internet service"])
tech_support   = st.sidebar.selectbox(
    "Tech Support",    ["No", "Yes", "No internet service"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior         = st.sidebar.selectbox("Senior Citizen",    [0, 1])

if st.sidebar.button("🔍 Predict Churn"):
    payload = {
        "tenure": tenure, "MonthlyCharges": monthly_charge,
        "TotalCharges": total_charges, "Contract": contract,
        "PaymentMethod": payment_method, "InternetService": internet,
        "OnlineSecurity": online_sec, "TechSupport": tech_support,
        "PaperlessBilling": paperless, "SeniorCitizen": senior,
        "gender": "Male", "Partner": "No", "Dependents": "No",
        "OnlineBackup": "No", "DeviceProtection": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "PhoneService": "Yes", "MultipleLines": "No",
    }
    with st.sidebar:
        with st.spinner("Calling API... (first request ~60s cold start)"):
            try:
                resp = requests.post(API_URL, json=payload, timeout=90)
                if resp.status_code == 200:
                    result = resp.json()
                    action = result["action"]
                    color  = {"RETAIN": "green", "OUTREACH": "orange", "URGENT": "red"}.get(action, "gray")
                    st.success("Prediction received!")
                    st.markdown(f"**Churn Probability:** `{result['churn_probability']}`")
                    st.markdown(f"**Risk Band:** `{result['churn_risk_band']}`")
                    st.markdown(
                        f"<h3 style='color:{color}'>Action: {action}</h3>",
                        unsafe_allow_html=True
                    )
                    if result.get("rule_triggered"):
                        st.warning(f"Rule: {result['rule_triggered']}")
                    desc = {
                        "RETAIN":   "✅ Low churn risk — standard engagement",
                        "OUTREACH": "⚠️ Medium risk — send retention offer",
                        "URGENT":   "🚨 High risk — immediate intervention",
                    }.get(action, "")
                    if desc:
                        st.info(desc)
                else:
                    st.error(f"API error: HTTP {resp.status_code}")
            except requests.exceptions.Timeout:
                st.warning("Timeout — Render cold start. Retry in 30s.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_mon = pd.read_csv(MONITOR_PATH)

    if "churn_probability" in df_mon.columns:
        avg_prob = df_mon["churn_probability"].mean()
        if avg_prob > CHURN_RATE_ALERT:
            st.error(
                f"🔴 HIGH CHURN RISK: avg prob={avg_prob:.4f} "
                f"(threshold {CHURN_RATE_ALERT}) — investigate cohort"
            )
            alerts_found = True

    if "action" in df_mon.columns:
        urgent_rate   = df_mon["action"].str.contains("URGENT").mean()
        outreach_rate = df_mon["action"].str.contains("OUTREACH").mean()
        if urgent_rate > URGENT_RATE_ALERT:
            st.error(
                f"🔴 HIGH URGENT RATE: {urgent_rate:.1%} > "
                f"{URGENT_RATE_ALERT:.0%} — escalate to retention team"
            )
            alerts_found = True
        if outreach_rate > 0.40:
            st.warning(
                f"🟡 HIGH OUTREACH QUEUE: {outreach_rate:.1%} — scale campaigns"
            )
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi_a = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi_a.columns:
        max_psi     = df_psi_a["drift_score"].max()
        top_feature = df_psi_a.iloc[0]["feature"] if "feature" in df_psi_a.columns else "unknown"
        if max_psi >= PSI_HIGH:
            st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}' — retrain now")
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}' — monitor")
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        chall_log = json.load(f)

    if chall_log:
        latest = chall_log[-1]
        d_col  = "green" if latest["decision"] == "PROMOTED" else "red"
        icon   = "✅" if latest["decision"] == "PROMOTED" else "❌"

        # ── Summary metrics row ───────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Challenger",        latest.get("challenger_name", "—"))
        with col2: st.metric("Challenger F1",     latest.get("challenger_f1",   "—"))
        with col3: st.metric("Challenger Recall", latest.get("challenger_recall","—"))
        with col4: st.metric("Champion F1",       latest.get("champion_f1",  "First Run"))

        st.markdown(
            f"<h4 style='color:{d_col}'>{icon} {latest['decision']} — {latest.get('reason','')}</h4>",
            unsafe_allow_html=True
        )

        # ── UPGRADE: 4-gate visual metrics ────────────────────
        # Each gate shown as its own st.metric block with ✅/❌
        # so interviewers immediately see gate-level pass/fail.
        gates = latest.get("gates", {})
        if gates:
            st.markdown("**Promotion Gates**")
            g1, g2, g3, g4 = st.columns(4)

            def _gate_icon(passed: bool) -> str:
                return "✅ PASS" if passed else "❌ FAIL"

            with g1:
                st.metric(
                    label = f"F1 Gate  (≥ +{MIN_F1_IMPROVEMENT:.1%})",
                    value = _gate_icon(gates.get("f1_improvement_passed", False)),
                )
            with g2:
                st.metric(
                    label = f"ROC-AUC Gate  (≥ {MIN_ROCAUC_THRESHOLD})",
                    value = _gate_icon(gates.get("roc_auc_passed", False)),
                )
            with g3:
                st.metric(
                    label = f"Recall Gate  (≥ {MIN_RECALL_THRESHOLD})",
                    value = _gate_icon(gates.get("recall_passed", False)),
                )
            with g4:
                st.metric(
                    label = f"Gap Gate  (≤ {MAX_GENERALIZATION_GAP:.0%})",
                    value = _gate_icon(gates.get("gap_passed", False)),
                )

        # ── Full challenger history table ─────────────────────
        if len(chall_log) > 1:
            with st.expander("📋 Full challenger history"):
                hdf  = pd.DataFrame(chall_log)
                cols = [c for c in ["evaluated_at", "challenger_name", "challenger_f1",
                        "challenger_recall", "champion_f1", "decision", "reason"]
                        if c in hdf.columns]
                st.dataframe(hdf[cols])
else:
    st.info("No challenger log — run train_model.py first")


# ============================================================
# SECTION 3 — KPI METRICS + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Model Performance KPIs")

if os.path.exists(MONITOR_PATH):
    df = pd.read_csv(MONITOR_PATH)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if "label" in df.columns:
            st.metric("Actual Churn Rate", f"{df['label'].mean():.1%}")
    with c2:
        if "action" in df.columns:
            st.metric("RETAIN Rate", f"{(df['action']=='RETAIN').mean():.1%}")
    with c3:
        if "action" in df.columns:
            st.metric("OUTREACH Queue", f"{df['action'].str.contains('OUTREACH').mean():.1%}")
    with c4:
        if "action" in df.columns:
            st.metric("URGENT Rate", f"{df['action'].str.contains('URGENT').mean():.1%}")

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Churn Probability Distribution")
        if "churn_probability" in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df["churn_probability"], bins=50, alpha=0.7, color="steelblue")
            ax.axvline(0.30, color="orange", linestyle="--", label="LOW/MED")
            ax.axvline(0.60, color="red",    linestyle="--", label="MED/HIGH")
            ax.legend(fontsize=8)
            st.pyplot(fig); plt.close(fig)
    with cb:
        st.subheader("Action Distribution")
        if "action" in df.columns:
            st.bar_chart(df["action"].value_counts())

    if "churn_risk_band" in df.columns:
        st.subheader("Risk Band Breakdown")
        st.bar_chart(df["churn_risk_band"].value_counts())

    st.subheader("Churn Probability Statistics")
    if "churn_probability" in df.columns:
        st.write(df["churn_probability"].describe().to_frame().T.round(4))
else:
    st.warning("No monitor scores — run train_model.py first")


# ============================================================
# SECTION 4 — REVENUE IMPACT PANEL  (Churn-specific)
# ============================================================

st.markdown("---")
st.subheader("💰 Revenue Impact Panel")

card_loaded = False
if os.path.exists(CARD_DIR):
    json_files = sorted([f for f in os.listdir(CARD_DIR) if f.startswith("model_card_")])
    if json_files:
        try:
            with open(os.path.join(CARD_DIR, json_files[-1])) as f:
                card = json.load(f)
            rev = card.get("revenue_impact", {})
            if rev:
                card_loaded = True
                r1, r2, r3, r4 = st.columns(4)
                with r1: st.metric("Missed Revenue (FN)", f"₹{rev.get('missed_revenue_loss',0):,.0f}")
                with r2: st.metric("Revenue Saved (TP)",  f"₹{rev.get('revenue_saved_by_model',0):,.0f}")
                with r3: st.metric("Retention ROI",       f"{rev.get('retention_roi',0):.2f}x")
                with r4: st.metric("Campaign Waste (FP)", f"₹{rev.get('wasted_campaign_cost',0):,.0f}")
                st.caption(
                    f"FN={rev.get('false_negative_count',0)}  |  "
                    f"FP={rev.get('false_positive_count',0)}  |  "
                    f"TP={rev.get('true_positive_count',0)}"
                )
        except Exception as e:
            st.warning(f"Could not load model card: {e}")

if not card_loaded:
    if os.path.exists(MONITOR_PATH):
        df_r = pd.read_csv(MONITOR_PATH)
        if "churn_probability" in df_r.columns and "MonthlyCharges" in df_r.columns:
            high_risk   = df_r[df_r["churn_probability"] > 0.60]
            rev_at_risk = (high_risk["MonthlyCharges"] * 24).sum()
            urg_count   = df_r["action"].str.contains("URGENT").sum() if "action" in df_r.columns else 0
            r1, r2 = st.columns(2)
            with r1: st.metric("Estimated Revenue at Risk", f"₹{rev_at_risk:,.0f}")
            with r2: st.metric("Customers Needing Urgent Action", f"{urg_count}")
    else:
        st.info("Train model to see revenue impact")


# ============================================================
# SECTION 5 — FEATURE DRIFT (PSI)
# ============================================================

st.markdown("---")
st.subheader("📉 Feature Drift Report (PSI)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns:
        def _flag(v):
            if v >= PSI_HIGH:       return "🔴 CRITICAL"
            elif v >= PSI_MODERATE: return "🟡 MODERATE"
            return "🟢 OK"
        df_psi["status"] = df_psi["drift_score"].apply(_flag)
        st.dataframe(df_psi.head(15), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = [
            "#e74c3c" if v >= PSI_HIGH else "#f39c12" if v >= PSI_MODERATE else "#2ecc71"
            for v in df_psi["drift_score"].head(15)
        ]
        ax.barh(df_psi["feature"].head(15)[::-1], df_psi["drift_score"].head(15)[::-1],
                color=bar_colors[::-1])
        ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label="Moderate")
        ax.axvline(PSI_HIGH,     color="red",    linestyle="--", label="Critical")
        ax.set_xlabel("PSI Score")
        ax.set_title("Feature Drift — Top 15 Features")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.warning("'drift_score' column missing in PSI file")
else:
    st.warning("Feature drift report not found")
    st.caption(f"Expected: `{PSI_PATH}`")
    if os.path.exists(CARD_DIR):
        st.caption(f"churn_models/ contents: {os.listdir(CARD_DIR)}")


# ============================================================
# SECTION 6 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Recent Predictions")

if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH).tail(20), use_container_width=True)
else:
    st.info(
        "Prediction logs are written by the Render API — "
        "not available on Streamlit Cloud. Run locally to see logs."
    )