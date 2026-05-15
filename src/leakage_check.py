# ============================================================
# LEAKAGE CHECK — Telco Churn ML System
# ============================================================
# Detects potential data leakage before model training.
#
# Churn-specific leakage risks (different from Credit Risk):
#
#   1. TotalCharges leakage:
#      TotalCharges = MonthlyCharges × tenure (near-perfect)
#      If tenure is a feature AND TotalCharges is a feature,
#      there is multicollinearity but NOT leakage.
#      Real leakage would be: a "churn_date" column slipping in.
#
#   2. customerID leakage:
#      If customerID is not dropped and correlates with
#      churn label (e.g., sequential IDs for recent customers
#      who churned more) → spurious correlation.
#
#   3. Target-derived features:
#      Any feature computed AFTER knowing churn outcome
#      (e.g., "days_since_last_contact" filled only for churners)
#
#   4. Near-perfect correlation:
#      Same threshold=0.99 as Credit Risk — catches obvious cases.
#
# Two checks (same architecture as Credit Risk):
#   1. Exact match  — feature column identical to target
#   2. Near-perfect correlation — |corr| >= threshold with target
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Churn-specific columns to always warn about ──────────────
# These are known leakage risks in telecom datasets
KNOWN_LEAKAGE_RISK_COLS = {
    "churn_date":        "Direct leakage — churn date implies churn=1",
    "days_to_churn":     "Direct leakage — only filled for churners",
    "cancel_reason":     "Direct leakage — only exists post-churn",
    "exit_survey":       "Direct leakage — only collected from churners",
    "retention_contact": "Post-churn label — remove from features",
}


def detect_leakage(
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    threshold_corr: float = 0.99,
) -> list:
    """
    Runs leakage heuristics on training set.

    Args:
        X_train        : feature DataFrame (after split, before preprocessing)
        y_train        : target Series (Churn 0/1)
        threshold_corr : correlation threshold above which feature is flagged

    Returns:
        List of warning strings. Empty list = no leakage detected.
    """
    warnings = []

    for col in X_train.columns:

        # ── Churn-specific: known dangerous column names ──────
        col_lower = col.lower()
        for risk_col, reason in KNOWN_LEAKAGE_RISK_COLS.items():
            if risk_col in col_lower:
                warnings.append(
                    f"[LEAKAGE] '{col}' matches known leakage pattern "
                    f"'{risk_col}' → {reason}"
                )
                break

        try:
            # ── Check 1: exact match with target ──────────────
            if X_train[col].equals(y_train.astype(X_train[col].dtype)):
                warnings.append(
                    f"[LEAKAGE] '{col}' is identical to target → remove this feature"
                )
                continue

            # ── Check 2: near-perfect correlation ─────────────
            if np.issubdtype(X_train[col].dtype, np.number):
                corr = abs(
                    np.corrcoef(
                        X_train[col].fillna(0),
                        y_train.fillna(0)
                    )[0, 1]
                )
                if corr >= threshold_corr:
                    warnings.append(
                        f"[LEAKAGE] '{col}' has corr={corr:.4f} with target "
                        f"(>= {threshold_corr}) → possible leakage"
                    )

        except Exception as e:
            logger.warning("Leakage check failed for column '%s': %s", col, e)

    # ── Log results ───────────────────────────────────────────
    if warnings:
        logger.warning("Leakage check found %d warning(s):", len(warnings))
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Leakage check passed — no obvious leakage detected")

    return warnings
