# ============================================================
# PREDICTION SERVICE — Telco Churn ML System
# ============================================================
# Same architecture as Credit Risk prediction_service.py.
#
# Churn-specific vs Credit Risk:
#   - prepare_features() → adds all 16 churn engineered features
#   - predict_customer() → returns churn_probability + action
#   - score_customer() from churn_engine (vs score_applicant)
#   - Output keys: churn_probability, churn_risk_band, action
# ============================================================

import pandas as pd
import numpy as np
import logging

from src.churn_engine import score_customer
from src.data_loader  import add_engineered_features

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(input_data: dict) -> pd.DataFrame:
    """
    Takes raw API input dict → returns engineered feature DataFrame.
    Mirrors the training pipeline feature engineering exactly.

    Raw API input fields:
      tenure, MonthlyCharges, TotalCharges, Contract,
      PaymentMethod, InternetService, PhoneService,
      MultipleLines, OnlineSecurity, OnlineBackup,
      DeviceProtection, TechSupport, StreamingTV,
      StreamingMovies, gender, SeniorCitizen,
      Partner, Dependents, PaperlessBilling

    Returns:
        DataFrame with all engineered features (same columns as training)
    """
    df = pd.DataFrame([input_data])

    # ── TotalCharges coerce (API may send string) ─────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ── Fill TotalCharges for new customers (tenure=0) ────────
    if df["TotalCharges"].isnull().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"] * df["tenure"].clip(lower=1)
        )

    df = add_engineered_features(df)

    return df


# ============================================================
# PREDICT — single customer
# ============================================================

def predict_customer(model, input_data: dict, threshold: float) -> dict:
    """
    Full prediction flow for one customer:
      1. Feature engineering
      2. Model churn probability
      3. Churn engine scoring (rules + ML)

    Args:
        model      : loaded sklearn pipeline
        input_data : raw API input dict
        threshold  : calibrated decision threshold

    Returns:
        structured churn output dict:
        {
          churn_probability, churn_risk_band,
          action, rule_triggered
        }
    """
    df = prepare_features(input_data)

    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        logger.error("predict_proba failed: %s", e)
        prob = 0.5

    # ── Pass engineered row to churn engine ───────────────────
    row    = df.iloc[0].to_dict()
    result = score_customer(row, prob, threshold)

    logger.info(
        "Prediction | prob=%.4f  band=%s  action=%s",
        result["churn_probability"],
        result["churn_risk_band"],
        result["action"]
    )

    return result
