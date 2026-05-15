# ============================================================
# CHURN ENGINE — Telco Churn ML System
# ============================================================
# Telecom-grade 3-tier intervention system:
#
#   RETAIN   → low churn risk, standard engagement
#   OUTREACH → medium risk, proactive retention offer
#   URGENT   → high risk, immediate intervention required
#
# Decision hierarchy (mirrors Credit Risk risk_engine.py):
#   1. Hard business rules (override ML score)
#   2. ML model probability + churn risk bands
#
# Why rules before ML?
#   Some churn patterns are deterministic from business data
#   (e.g., month-to-month + no contract + first month = almost certain churn)
#   Rules catch these before expensive ML inference.
#
# Churn action vs Credit Risk decision mapping:
#   RETAIN   ↔ APPROVE
#   OUTREACH ↔ REVIEW
#   URGENT   ↔ DECLINE
# ============================================================

import pandas as pd
import logging

from src.config import (
    CHURN_RISK_BANDS,
    MIN_TENURE_RULE,
    MAX_MONTHLY_CHARGES_RULE,
    MONTH_TO_MONTH_HIGH_PROB,
)

logger = logging.getLogger(__name__)


# ============================================================
# CHURN RISK BAND — probability → LOW / MEDIUM / HIGH
# ============================================================

def get_churn_risk_band(prob: float) -> str:
    """
    Maps churn probability to risk band.

    LOW    (0.00 – 0.30) → safe customer
    MEDIUM (0.30 – 0.60) → at-risk customer
    HIGH   (0.60 – 1.00) → likely churner
    """
    for band, (low, high) in CHURN_RISK_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


# ============================================================
# CHURN ENGINE — batch decisions for evaluation
# ============================================================

def churn_engine(
    customer_df: pd.DataFrame,
    probs,
    threshold: float
) -> list:
    """
    For each customer row → returns intervention action string.

    Rule priority:
      1. IsNewCustomer + Month-to-month + no AutoPays
             → URGENT (triple churn signal)
      2. MonthlyCharges > MAX_MONTHLY_CHARGES_RULE
         + no LongTermContract
             → OUTREACH (pricing pressure signal)
      3. tenure < MIN_TENURE_RULE (new customer window)
             → OUTREACH (early churn window flag)
      4. prob >= threshold              → URGENT   (ML)
      5. prob >= threshold × 0.55      → OUTREACH (ML borderline)
      6. else                          → RETAIN

    Args:
        customer_df : engineered feature DataFrame
        probs       : array of churn probabilities
        threshold   : decision threshold (calibrated)

    Returns:
        list of action strings per row
    """
    actions = []

    for idx, (_, row) in enumerate(customer_df.iterrows()):
        p = probs[idx]

        # ── Hard rule 1: triple churn signal ─────────────────
        # New customer + month-to-month + no autopay
        # = highest churn risk combination in telecom
        is_new    = row.get("IsNewCustomer",    0)
        long_term = row.get("LongTermContract", 1)   # default 1 = safe
        auto_pays = row.get("AutoPays",         1)   # default 1 = safe

        if is_new == 1 and long_term == 0 and auto_pays == 0:
            actions.append("URGENT_TRIPLE_RISK")
            continue

        # ── Hard rule 2: high charges + no contract ───────────
        monthly = row.get("MonthlyCharges", 0)
        if monthly > MAX_MONTHLY_CHARGES_RULE and long_term == 0:
            actions.append("OUTREACH_HIGH_CHARGE")
            continue

        # ── Soft rule 3: new customer tenure window ───────────
        tenure = row.get("tenure", 99)
        if tenure < MIN_TENURE_RULE:
            actions.append("OUTREACH_NEW_CUSTOMER")
            continue

        # ── ML model decisions ────────────────────────────────
        if p >= threshold:
            actions.append("URGENT_MODEL")

        elif p >= threshold * 0.55:
            actions.append("OUTREACH_MODEL")

        else:
            actions.append("RETAIN")

    return actions


# ============================================================
# CHURN SCORING — single customer (for API)
# ============================================================

def score_customer(row: dict, prob: float, threshold: float) -> dict:
    """
    Returns structured churn intervention output for one customer.
    Used by FastAPI prediction endpoint.

    Maps directly to Credit Risk score_applicant() pattern.

    Args:
        row       : dict of engineered feature values
        prob      : churn probability from model
        threshold : calibrated decision threshold

    Returns:
        dict with churn_probability, risk_band, action, rule_triggered
    """
    risk_band     = get_churn_risk_band(prob)
    rule_triggered = None

    # ── Derived fields ────────────────────────────────────────
    is_new    = int(row.get("IsNewCustomer",    0))
    long_term = int(row.get("LongTermContract", 1))
    auto_pays = int(row.get("AutoPays",         1))
    tenure    = float(row.get("tenure",         99))
    monthly   = float(row.get("MonthlyCharges", 0))

    # ── Rule-based overrides ──────────────────────────────────

    # Triple risk: new + no contract + no autopay
    if is_new == 1 and long_term == 0 and auto_pays == 0:
        action         = "URGENT"
        rule_triggered = "NEW_CUSTOMER_NO_CONTRACT_NO_AUTOPAY"

    # High charges + no contract
    elif monthly > MAX_MONTHLY_CHARGES_RULE and long_term == 0:
        action         = "OUTREACH"
        rule_triggered = "HIGH_CHARGES_NO_CONTRACT"

    # New customer tenure window
    elif tenure < MIN_TENURE_RULE:
        action         = "OUTREACH"
        rule_triggered = "NEW_CUSTOMER_TENURE_RISK"

    # ── ML model decisions ────────────────────────────────────
    elif prob >= threshold:
        action = "URGENT"

    elif prob >= threshold * 0.55:
        action = "OUTREACH"

    else:
        action = "RETAIN"

    return {
        "churn_probability": round(float(prob), 4),
        "churn_risk_band":   risk_band,
        "action":            action,
        "rule_triggered":    rule_triggered,
    }


# ============================================================
# ACTION DESCRIPTION — human-readable output
# ============================================================

ACTION_DESCRIPTIONS = {
    "RETAIN": (
        "Low churn risk. Standard engagement — "
        "no immediate intervention needed."
    ),
    "OUTREACH": (
        "Medium churn risk. Proactive retention recommended — "
        "send personalized offer or discount."
    ),
    "URGENT": (
        "High churn risk. Immediate intervention required — "
        "escalate to retention team with priority offer."
    ),
}


def get_action_description(action: str) -> str:
    """Returns human-readable description for a churn action."""
    # Normalize URGENT_TRIPLE_RISK → URGENT etc.
    base = action.split("_")[0]
    return ACTION_DESCRIPTIONS.get(base, ACTION_DESCRIPTIONS.get(action, "Unknown action"))
