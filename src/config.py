# ============================================================
# CONFIGURATION — Telco Churn Prediction ML System
# ============================================================
# FIX: Champion vs Challenger gate constants moved here from
# model_loader.py so they are importable from a single source
# of truth (config.py) rather than hardcoded in model_loader.
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE  = 42
N_JOBS        = 1          # Windows joblib fix

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS             = 5
RANDOM_SEARCH_ITERS  = 20

# ── Feature selection ────────────────────────────────────────
SELECT_K = 15

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5

# ── Feature type detection ───────────────────────────────────
ORDINAL_UNIQUE_THRESHOLD = 15

# ── Churn risk bands (probability → tier) ───────────────────
CHURN_RISK_BANDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

# ── Business rule thresholds ─────────────────────────────────
MIN_TENURE_RULE            = 3
MAX_MONTHLY_CHARGES_RULE   = 100.0
MONTH_TO_MONTH_HIGH_PROB   = 0.45

# ── Churn intervention tiers ─────────────────────────────────
CHURN_ACTIONS = {
    "RETAIN":    "Low churn risk — standard engagement",
    "OUTREACH":  "Medium churn risk — proactive retention offer",
    "URGENT":    "High churn risk — immediate intervention required",
}

# ── Revenue impact assumptions (₹ / month) ───────────────────
AVG_CUSTOMER_LTV_MONTHS     = 24
RETENTION_COST_PER_CUSTOMER = 500
REVENUE_LOSS_PER_CHURNER    = 1200

# ── PSI thresholds ────────────────────────────────────────────
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

# ── Champion vs Challenger gate constants ─────────────────────
# Moved from model_loader.py — single source of truth.
# Import these in model_loader.py and any test that validates gates.
MIN_F1_IMPROVEMENT     = 0.005   # challenger must beat champion F1 by >= 0.5%
MIN_ROCAUC_THRESHOLD   = 0.80    # minimum ROC-AUC for promotion
MIN_RECALL_THRESHOLD   = 0.70    # churn-specific: recall guarantee gate
MAX_GENERALIZATION_GAP = 0.10    # train-test gap <= 10%

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR    = "churn_models"
LOG_DIR      = "logs"
METRICS_LOG  = "churn_models/metrics_log.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)