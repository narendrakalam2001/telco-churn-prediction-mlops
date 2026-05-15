# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Telco Churn ML System
# ============================================================
# Telco dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
# 7,043 customers | 26 columns | ~26.5% churn rate
#
# Feature engineering philosophy:
#   Every feature maps to a real telecom business signal.
#   Banks use credit ratios → Telcos use tenure/charge/service signals.
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Required columns in raw dataset ──────────────────────────
REQUIRED_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Contract", "PaymentMethod", "Churn"
]

# ── Service columns used for counting ────────────────────────
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates raw Telco Churn dataset.
    Handles: column names, types, nulls, duplicates, target encoding.
    """

    # ── Normalize column names ────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Drop customer ID (non-predictive) ─────────────────────
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        logger.info("Dropped column: customerID")

    # ── Check required columns ────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── TotalCharges: coerce spaces → NaN → fill median ───────
    # Raw data has " " (space strings) for new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_null = df["TotalCharges"].isnull().sum()
    if n_null > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_val, inplace=True)
        logger.info("TotalCharges: filled %d nulls with median=%.2f", n_null, median_val)

    # ── Target encoding: Yes/No → 1/0 ────────────────────────
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        if df["Churn"].isnull().any():
            raise ValueError("Churn column has unexpected values (expected Yes/No)")
    df["Churn"] = df["Churn"].astype(int)

    # ── SeniorCitizen: already 0/1 in raw data ────────────────
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # ── Null check after fixes ────────────────────────────────
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning("Dataset has %d remaining nulls — handled in preprocessing", nulls)

    # ── Minimum size check ────────────────────────────────────
    if df.shape[0] < 100:
        raise ValueError("Dataset too small for training (< 100 rows)")

    # ── Deduplication ─────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    logger.info(
        "Data validation passed | shape=%s | churn_rate=%.3f",
        df.shape, df["Churn"].mean()
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Telecom-grade feature engineering for churn prediction.

    Signal categories:
      1. Financial signals      — charge ratios, avg spend
      2. Tenure / loyalty       — cohorts, new customer flags
      3. Service adoption       — breadth of services used
      4. Contract / payment     — stickiness signals
      5. Interaction features   — combined risk signals
      6. Behavioral flags       — binary risk indicators

    Every feature has a documented business rationale.
    """
    df = df.copy()

    # ── 1. FINANCIAL SIGNALS ──────────────────────────────────

    # Avg monthly spend = total / tenure (catches pricing changes over time)
    df["AvgMonthlyCharges"] = (
        df["TotalCharges"] / df["tenure"].replace(0, 1)
    )

    # Charge deviation = how much current month differs from avg
    # High positive deviation → recent price hike → churn risk
    df["ChargeDeviation"] = df["MonthlyCharges"] - df["AvgMonthlyCharges"]

    # Revenue at risk = monthly charges × estimated remaining lifetime
    # Proxy for LTV impact if customer churns
    df["RevenueAtRisk"] = df["MonthlyCharges"] * (72 - df["tenure"]).clip(lower=0)

    # ── 2. TENURE / LOYALTY SIGNALS ───────────────────────────

    # New customer flag — first 3 months = highest churn risk window
    df["IsNewCustomer"] = (df["tenure"] <= 3).astype(int)

    # Loyal customer flag — 2+ years = stickiest segment
    df["IsLoyalCustomer"] = (df["tenure"] >= 24).astype(int)

    # Tenure group — cohort-level segmentation (same bins as telecom SLA periods)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 3, 12, 24, 48, 72],
        labels=[0, 1, 2, 3, 4],   # numeric labels → ordinal safe
        include_lowest=True
    ).astype(float)

    # ── 3. SERVICE ADOPTION SIGNALS ───────────────────────────

    # Count of "Yes" services — more services = higher switching cost
    present_cols = [c for c in SERVICE_COLS if c in df.columns]
    df["ServiceCount"] = (df[present_cols] == "Yes").sum(axis=1)

    # Has internet service (core stickiness driver)
    if "InternetService" in df.columns:
        df["HasInternet"] = (df["InternetService"] != "No").astype(int)

    # Add-on adoption rate — % of available services subscribed
    n_services = len(present_cols)
    df["ServiceAdoptionRate"] = df["ServiceCount"] / n_services if n_services > 0 else 0

    # ── 4. CONTRACT / PAYMENT STICKINESS ──────────────────────

    # Long-term contract = strong retention signal
    if "Contract" in df.columns:
        df["LongTermContract"] = df["Contract"].isin(
            ["One year", "Two year"]
        ).astype(int)

    # Auto-payment = friction to cancel (reduces churn)
    if "PaymentMethod" in df.columns:
        df["AutoPays"] = df["PaymentMethod"].str.contains(
            "automatic", case=False, na=False
        ).astype(int)

    # Paperless billing — digital-only customers churn more (less friction)
    if "PaperlessBilling" in df.columns:
        df["PaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)

    # ── 5. INTERACTION FEATURES ───────────────────────────────

    # Charges per service — high = overpaying → churn risk
    df["ChargesPerService"] = (
        df["MonthlyCharges"] / (df["ServiceCount"] + 1)
    )

    # Tenure × monthly charges — high tenure + high spend = high LTV customer
    df["TenureValueScore"] = df["tenure"] * df["MonthlyCharges"]

    # Month-to-month + high charges = maximum churn risk combination
    if "Contract" in df.columns:
        df["M2M_HighCharge"] = (
            (df["Contract"] == "Month-to-month") &
            (df["MonthlyCharges"] > df["MonthlyCharges"].median())
        ).astype(int)

    # No contract + no autopay = easiest to leave (double friction-free)
    if "Contract" in df.columns and "PaymentMethod" in df.columns:
        df["NoContractNoAutopay"] = (
            (df["Contract"] == "Month-to-month") &
            (df["AutoPays"] == 0)
        ).astype(int)

    # ── 6. BEHAVIORAL FLAGS ───────────────────────────────────

    # High monthly charge flag (above 75th percentile)
    high_charge_thr = df["MonthlyCharges"].quantile(0.75)
    df["HighChargeFlag"] = (df["MonthlyCharges"] > high_charge_thr).astype(int)

    # Zero total charges = brand new customer (special edge case)
    df["ZeroTotalCharges"] = (df["TotalCharges"] == 0).astype(int)

    logger.info(
        "Feature engineering done | total columns=%d | engineered=%d",
        df.shape[1],
        df.shape[1] - 26    # approx new features added
    )

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 15):
    """
    Auto-detect: ordinal / continuous / binary columns.
    Excludes target column 'Churn'.

    Churn dataset has many Yes/No object columns → treated as ordinal.
    """
    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    for col in df.columns:
        if col == "Churn":
            continue

        dtype_name = str(df[col].dtype)
        n_unique   = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category"):
            ordinal_cols.append(col)

        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info(
        "Feature types | ordinal=%d  continuous=%d  binary=%d",
        len(ordinal_cols), len(continuous_cols), len(binary_cols)
    )

    return ordinal_cols, continuous_cols, binary_cols
