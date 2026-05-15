# ============================================================
# METRICS — Telco Churn ML System
# ============================================================
# Churn-specific metrics vs Credit Risk:
#
#   Credit Risk focus:  Precision (avoid false approvals = money lost)
#   Churn focus:        Recall    (catch churners before they leave)
#
#   Why recall matters more in churn:
#     Missing a churner (FN) = lost revenue forever
#     False alarm (FP)       = small retention cost only
#
#   Hence tune_threshold targets recall >= 0.75 by default
#   (vs F1-maximize in credit risk)
#
# Additional churn metrics:
#   - Revenue impact evaluation (ECL-equivalent for telecom)
#   - Retention ROI calculation
#   - Recall@K / Lift@K (top-K campaign targeting)
#   - PSI for score drift monitoring
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(
    y_true:           np.ndarray,
    y_prob:           np.ndarray,
    target_recall:    float = 0.75,
    target_precision: float = None,
) -> float:
    """
    Churn default: maximize F1 subject to recall >= 0.75.

    Rationale:
      In telecom, missing a churner costs full LTV.
      So we prioritize catching churners (recall)
      while maximizing precision within that constraint.

    Args:
        y_true           : true binary labels
        y_prob           : predicted probabilities
        target_recall    : minimum recall constraint (default 0.75)
        target_precision : optional precision constraint

    Returns:
        best threshold (float)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # ── Default: maximize F1 ─────────────────────────────────
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx  = np.nanargmax(f1_scores)
    best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    # ── Constrain on recall ───────────────────────────────────
    if target_recall is not None:
        idxs = np.where(recall >= target_recall)[0]
        if idxs.size > 0:
            # Among thresholds that meet recall target,
            # pick the one with highest precision
            chosen  = idxs[np.argmax(precision[idxs])]
            best_thr = float(thresholds[chosen]) if chosen < len(thresholds) else 0.5

    # ── Constrain on precision ────────────────────────────────
    if target_precision is not None:
        idxs = np.where(precision >= target_precision)[0]
        if idxs.size > 0:
            chosen  = idxs[np.argmax(recall[idxs])]
            best_thr = float(thresholds[chosen]) if chosen < len(thresholds) else 0.5

    logger.info(
        "Best threshold: %.4f  |  precision=%.4f  recall=%.4f  f1=%.4f",
        best_thr,
        precision[best_idx],
        recall[best_idx],
        f1_scores[best_idx]
    )

    return best_thr


# ============================================================
# PSI — Population Stability Index
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → stable (no significant shift)
    PSI 0.1–0.2 → moderate shift (monitor closely)
    PSI > 0.2   → major shift (retrain recommended)

    Correct approach:
      1. Compute quantile bin EDGES from `expected` (reference)
      2. Bin BOTH distributions using SAME edges
      3. Compare bin proportions

    Bug to avoid:
      Ranking both independently → PSI always ~0 (wrong!)
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        # ── Bin edges from reference distribution ─────────────
        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)
        bin_edges  = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        # Extend edges to cover full range of actual
        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        # ── Bin BOTH using SAME edges ─────────────────────────
        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        # ── Proportions — avoid div-by-zero ───────────────────
        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        psi_value = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))
        return psi_value

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# KS STATISTIC
# ============================================================

def ks_statistic(y_true, y_prob) -> float:
    """
    KS = max separation between churner and non-churner
    score distributions. Higher = better discrimination.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ============================================================
# RECALL @ K  (campaign targeting metric)
# ============================================================

def recall_at_k(y_true, y_prob, k: float = 0.20) -> float:
    """
    What % of actual churners do we catch if we
    target the top-K% highest-risk customers?

    Telecom use case:
      k=0.20 → send retention offer to top 20% risky customers
      recall@20% = % of all churners captured in that campaign
    """
    df = (
        pd.DataFrame({"y": y_true, "p": y_prob})
        .sort_values("p", ascending=False)
    )
    top_n = int(len(df) * k)
    return float(df.iloc[:top_n]["y"].sum() / (df["y"].sum() + 1e-9))


# ============================================================
# LIFT @ K
# ============================================================

def lift_at_k(y_true, y_prob, k: float = 0.20) -> float:
    """
    How many times better than random targeting?

    lift@20% = 3.0 means:
      Targeting top 20% risky customers catches 3x more
      churners than random selection of 20% customers.
    """
    base = np.mean(y_true)
    if base == 0:
        return 0.0
    return float(recall_at_k(y_true, y_prob, k) / (base + 1e-9))


# ============================================================
# REVENUE IMPACT EVALUATION  (Churn-specific, ECL equivalent)
# ============================================================

def revenue_impact_evaluation(
    X_test,
    y_true,
    y_pred,
    monthly_charges_col:      str   = "MonthlyCharges",
    avg_lifetime_months:      int   = 24,
    retention_cost_per_case:  float = 500.0,    # ₹ per retention campaign
    false_alarm_review_cost:  float = 100.0,    # ₹ analyst cost for FP
) -> dict:
    """
    Telecom Revenue Impact Analysis — equivalent to Credit Risk ECL.

    Cost model:
      FN (missed churner) = lost LTV = MonthlyCharges × avg_lifetime_months
      FP (false alarm)    = wasted retention campaign cost

    This maps directly to:
      Credit Risk FN → missed bad loan → capital loss
      Credit Risk FP → unnecessary review → analyst cost

    Returns:
        dict with revenue loss, retention cost, total impact
    """
    df = X_test.copy()
    df["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df["y_pred"] = y_pred

    # ── Missed churners (FN) → lost revenue ──────────────────
    fn_mask = (df["y_true"] == 1) & (df["y_pred"] == 0)

    if monthly_charges_col in df.columns:
        fn_revenue_loss = float(
            df.loc[fn_mask, monthly_charges_col].sum() * avg_lifetime_months
        )
    else:
        # Fallback: count × fixed avg revenue
        fn_revenue_loss = float(fn_mask.sum() * 1200 * avg_lifetime_months)

    # ── False alarms (FP) → wasted campaign cost ─────────────
    fp_mask          = (df["y_true"] == 0) & (df["y_pred"] == 1)
    fp_campaign_cost = float(fp_mask.sum() * retention_cost_per_case)

    total_impact = fn_revenue_loss + fp_campaign_cost

    # ── Retention ROI: what if we acted on all predicted churners ──
    # TP = churners we caught and can retain
    tp_mask          = (df["y_true"] == 1) & (df["y_pred"] == 1)
    tp_count         = int(tp_mask.sum())

    if monthly_charges_col in df.columns:
        tp_revenue_saved = float(
            df.loc[tp_mask, monthly_charges_col].sum() * avg_lifetime_months
        )
    else:
        tp_revenue_saved = float(tp_count * 1200 * avg_lifetime_months)

    campaign_cost_total = float((tp_count + int(fp_mask.sum())) * retention_cost_per_case)
    retention_roi       = round(
        (tp_revenue_saved - campaign_cost_total) / (campaign_cost_total + 1e-9), 2
    )

    result = {
        "false_negative_count":    int(fn_mask.sum()),
        "false_positive_count":    int(fp_mask.sum()),
        "true_positive_count":     tp_count,
        "missed_revenue_loss":     round(fn_revenue_loss,   2),
        "wasted_campaign_cost":    round(fp_campaign_cost,  2),
        "total_cost_impact":       round(total_impact,      2),
        "revenue_saved_by_model":  round(tp_revenue_saved,  2),
        "retention_roi":           retention_roi,
    }

    logger.info(
        "Revenue impact | FN=%d FP=%d TP=%d | "
        "missed_loss=%.0f  campaign_waste=%.0f  total=%.0f  ROI=%.2f",
        result["false_negative_count"],
        result["false_positive_count"],
        result["true_positive_count"],
        result["missed_revenue_loss"],
        result["wasted_campaign_cost"],
        result["total_cost_impact"],
        result["retention_roi"],
    )

    return result


# ============================================================
# DRIFT REPORT — feature mean shift
# ============================================================

def simple_drift_report(
    X_ref: "pd.DataFrame",
    X_new: "pd.DataFrame",
    top_n: int = 10
) -> "pd.Series":
    """
    Computes normalized mean shift per feature.
    Used for feature-level PSI monitoring dashboard.

    Bug fixed: X_ref and X_new contain raw (unencoded) columns like
    'gender', 'Contract', 'InternetService' etc. which are object/string
    dtype. Calling .mean() on object columns raises TypeError.
    Fix: filter to numeric columns only before computing drift.
    """
    # ── Only numeric columns can have drift computed ────────────
    num_cols = X_ref.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        import pandas as _pd
        return _pd.Series(dtype=float)

    X_ref_num = X_ref[num_cols]
    X_new_num = X_new[num_cols]

    diffs = (X_ref_num.mean() - X_new_num.mean()).abs()
    rel   = (diffs / (X_ref_num.std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)