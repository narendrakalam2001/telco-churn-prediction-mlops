# ============================================================
# TRAINING PIPELINE — Telco Churn ML System
# ============================================================
# Full 24-step orchestration.
#
# FIX: card_path is now captured from save_model_card() and
# passed to both save_model_and_registry() and
# run_challenger_comparison() so latest_model.json stores the
# model_card_path directly — no brittle string reconstruction.
# ============================================================

import os
import time
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_DIR, RANDOM_STATE, SELECT_K,
    PSI_MODERATE, PSI_HIGH
)
from src.data_loader    import validate_input_data, add_engineered_features, detect_feature_types
from src.leakage_check  import detect_leakage
from src.preprocessing  import build_preprocessors
from src.model_tuning   import scaled_models, unscaled_models, tune_models, train_mlp_pipeline
from src.metrics        import tune_threshold, psi, recall_at_k, revenue_impact_evaluation
from src.churn_engine   import churn_engine, score_customer
from src.evaluation     import (
    evaluate_all_models,
    apply_generalization_filter,
    calibrate_model,
    evaluate_best_model,
    plot_evaluation_charts,
    compute_shap,
    log_to_mlflow,
    save_model_and_registry,
    get_feature_importances,
)
from src.model_card     import build_model_card, save_model_card
from src.model_loader   import run_challenger_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training(
    data_path: str  = None,
    version:   str  = "v1",
) -> dict:
    """
    Runs the full 24-step Telco Churn training pipeline.

    Args:
        data_path : path to WA_Fn-UseC_-Telco-Customer-Churn.csv
        version   : model version string (default 'v1')

    Returns:
        result dict with best_name, metrics, threshold, model_path
    """

    start_time = time.time()

    logger.info("=" * 65)
    logger.info("  TELCO CHURN PREDICTION — TRAINING PIPELINE START")
    logger.info("=" * 65)

    if data_path is None:
        data_path = os.getenv(
            "CHURN_DATA_PATH",
            r"D:\Data Science Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

    # ── STEP 1 — LOAD + VALIDATE DATA ─────────────────────────
    logger.info("STEP 1 — Loading and validating data")

    df_raw = pd.read_csv(data_path)
    df     = validate_input_data(df_raw.copy())

    logger.info(
        "Dataset loaded | shape=%s | churn_rate=%.3f",
        df.shape, df["Churn"].mean()
    )

    # ── STEP 2 — FEATURE ENGINEERING ──────────────────────────
    logger.info("STEP 2 — Feature engineering")

    df = add_engineered_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    logger.info("Features after engineering: %d", X.shape[1])

    # ── STEP 3 — TRAIN / CALIBRATION / TEST SPLIT ─────────────
    logger.info("STEP 3 — Train / calibration / test split")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20, stratify=y_train_full, random_state=RANDOM_STATE
    )

    logger.info(
        "Split sizes | train_fit=%d  cal=%d  test=%d",
        len(X_train_fit), len(X_cal), len(X_test)
    )

    # ── STEP 4 — LEAKAGE DETECTION ────────────────────────────
    logger.info("STEP 4 — Leakage detection")

    leakage_warnings = detect_leakage(X_train_fit, y_train_fit, threshold_corr=0.99)

    if leakage_warnings:
        for w in leakage_warnings:
            logger.warning("  %s", w)
    else:
        logger.info("No leakage detected ✓")

    # ── STEP 5 — FEATURE TYPE DETECTION ───────────────────────
    logger.info("STEP 5 — Feature type detection")

    ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=15)

    # ── STEP 6 — BUILD DUAL PREPROCESSORS ─────────────────────
    logger.info("STEP 6 — Building dual preprocessors")

    pre_scaled, pre_unscaled, cat_indices, feature_order = build_preprocessors(
        ord_cols, cont_cols, bin_cols, X_train_fit
    )

    # ── STEP 7 — TUNE SCALED MODELS ───────────────────────────
    logger.info("STEP 7 — Tuning scaled models")

    scaled_pipelines = tune_models(
        scaled_models, pre_scaled, cat_indices,
        X_train_fit, y_train_fit,
        use_smote=True, selector_k=SELECT_K
    )

    # ── STEP 8 — TUNE UNSCALED MODELS ─────────────────────────
    logger.info("STEP 8 — Tuning unscaled models")

    unscaled_pipelines = tune_models(
        unscaled_models, pre_unscaled, cat_indices,
        X_train_fit, y_train_fit,
        use_smote=True, selector_k=SELECT_K
    )

    # ── STEP 9 — TRAIN MLP ────────────────────────────────────
    logger.info("STEP 9 — Training Neural Network (MLP)")

    mlp_pipe = train_mlp_pipeline(
        X_train_fit, y_train_fit, pre_scaled, cat_indices
    )

    # ── STEP 10 — MERGE ALL PIPELINES ─────────────────────────
    logger.info("STEP 10 — Merging all model pipelines")

    all_pipelines = {**scaled_pipelines, **unscaled_pipelines, "NeuralNet": mlp_pipe}
    logger.info("Total models trained: %d", len(all_pipelines))

    # ── STEP 11 — EVALUATE ALL MODELS ─────────────────────────
    logger.info("STEP 11 — Evaluating all models")

    summary = evaluate_all_models(
        all_pipelines, X_train_fit, y_train_fit, X_test, y_test
    )

    # ── STEP 12 — GENERALIZATION FILTER ───────────────────────
    logger.info("STEP 12 — Applying generalization filter (gap <= 10%%)")

    filtered_summary, filtered_pipelines = apply_generalization_filter(
        summary, all_pipelines, max_gap=0.10
    )

    # ── STEP 13 — SELECT BEST MODEL ───────────────────────────
    logger.info("STEP 13 — Selecting best model")

    best_name = filtered_summary.loc[0, "model"]
    best_pipe = filtered_pipelines[best_name]

    logger.info(
        "Best model: %s  |  cv_f1=%.4f  recall=%.4f  gap=%.4f",
        best_name,
        filtered_summary.loc[0, "cv_mean_f1"],
        filtered_summary.loc[0, "recall"],
        filtered_summary.loc[0, "gap"],
    )

    from src.metrics import tune_threshold as _tune_thr
    y_prob_uncal     = best_pipe.predict_proba(X_test)[:, 1]
    thr_uncalibrated = _tune_thr(y_test.values, y_prob_uncal, target_recall=0.75)

    # ── STEP 14 — CALIBRATE PROBABILITIES ─────────────────────
    logger.info("STEP 14 — Calibrating probabilities (holdout isotonic)")

    best_pipe, calibrated = calibrate_model(best_pipe, X_cal, y_cal)

    # ── STEP 15 — TUNE THRESHOLD ───────────────────────────────
    logger.info("STEP 15 — Tuning decision threshold (target_recall >= 0.75)")

    y_prob_cal    = best_pipe.predict_proba(X_cal)[:, 1]
    thr_calibrated = tune_threshold(
        y_cal.values, y_prob_cal, target_recall=0.75
    )

    active_threshold = thr_calibrated if calibrated else thr_uncalibrated

    logger.info(
        "Threshold | uncalibrated=%.4f  calibrated=%.4f  active=%.4f",
        thr_uncalibrated, thr_calibrated, active_threshold
    )

    # ── STEP 16 — FINAL METRICS EVALUATION ────────────────────
    logger.info("STEP 16 — Final metrics at tuned threshold")

    metrics = evaluate_best_model(
        best_name, best_pipe,
        X_train_fit, y_train_fit,
        X_test, y_test,
        threshold=active_threshold
    )

    # ── STEP 17 — PLOTS ───────────────────────────────────────
    logger.info("STEP 17 — Generating evaluation plots")

    plot_evaluation_charts(
        best_name, best_pipe, X_test, y_test,
        threshold=active_threshold,
        save_dir="docs/plots"
    )

    # ── STEP 18 — REVENUE IMPACT EVALUATION ───────────────────
    logger.info("STEP 18 — Revenue impact evaluation")

    y_prob_test = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= active_threshold).astype(int)

    revenue_result = revenue_impact_evaluation(
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        pd.Series(y_pred_test),
        monthly_charges_col     = "MonthlyCharges",
        avg_lifetime_months     = 24,
        retention_cost_per_case = 500.0,
    )

    logger.info(
        "Revenue impact | FN=%d  FP=%d  missed_loss=%.0f  ROI=%.2f",
        revenue_result["false_negative_count"],
        revenue_result["false_positive_count"],
        revenue_result["missed_revenue_loss"],
        revenue_result["retention_roi"],
    )

    # ── STEP 19 — ACTION DISTRIBUTION ─────────────────────────
    logger.info("STEP 19 — Computing action distribution")

    actions = churn_engine(
        X_test.reset_index(drop=True),
        y_prob_test,
        active_threshold
    )

    from collections import Counter
    action_distribution = dict(Counter(actions))
    logger.info("Action distribution: %s", action_distribution)

    # ── STEP 20 — SHAP EXPLAINABILITY ─────────────────────────
    logger.info("STEP 20 — Computing SHAP values")

    shap_dict = compute_shap(
        best_name, best_pipe,
        X_train_fit, X_test,
        save_dir="docs/plots"
    )

    # ── STEP 21 — FEATURE IMPORTANCES ─────────────────────────
    logger.info("STEP 21 — Extracting feature importances")

    fi_dict = get_feature_importances(best_pipe, top_n=15)

    if fi_dict:
        logger.info("Top feature: %s = %.6f", list(fi_dict.keys())[0], list(fi_dict.values())[0])

    # ── STEP 22 — BUILD + SAVE MODEL CARD ─────────────────────
    logger.info("STEP 22 — Building and saving model card")

    card = build_model_card(
        selected_name       = best_name,
        train_fit_size      = len(X_train_fit),
        cal_size            = len(X_cal),
        test_size           = len(X_test),
        churn_rate_train    = float(y_train_fit.mean()),
        metrics             = metrics,
        thr_uncalibrated    = thr_uncalibrated,
        thr_calibrated      = thr_calibrated if calibrated else None,
        revenue_result      = revenue_result,
        action_distribution = action_distribution,
        feature_order       = feature_order,
        cat_indices         = cat_indices,
        selector_k          = SELECT_K,
        version             = version,
        fi_dict             = fi_dict   if fi_dict   else None,
        shap_dict           = shap_dict if shap_dict else None,
    )

    # FIX: capture card_path so it can be stored in the registry
    card_path = save_model_card(card, MODEL_DIR, best_name, version)

    # ── STEP 23 — SAVE MODEL + REGISTRY ───────────────────────
    # CRITICAL: snapshot champion BEFORE saving new model.
    # If we load after save_model_and_registry(), latest_model.json
    # already points to the new model → champion == challenger → F1 diff = 0.
    logger.info("STEP 23 — Saving model and updating registry")

    from src.model_loader import _load_champion_metrics as _snap_champion
    _old_champion_snapshot = _snap_champion()

    # FIX: pass card_path so it's stored directly in latest_model.json
    model_path = save_model_and_registry(
        best_pipe, best_name, active_threshold, version,
        model_card_path=card_path
    )

    # ── STEP 24 — CHAMPION vs CHALLENGER ──────────────────────
    logger.info("STEP 24 — Champion vs Challenger comparison")

    challenger_result = run_challenger_comparison(
        challenger_name          = best_name,
        challenger_f1            = metrics["test_f1"],
        challenger_roc_auc       = metrics["roc_auc"] or 0.0,
        challenger_recall        = metrics["recall"],
        challenger_gap           = metrics["gap"],
        challenger_model_path    = model_path,
        challenger_threshold     = active_threshold,
        challenger_card_path     = card_path,          # FIX: pass card path
        _pre_loaded_champion     = _old_champion_snapshot,
    )

    logger.info(
        "Challenger decision: %s — %s",
        challenger_result["decision"],
        challenger_result["reason"]
    )

    # ── PSI check ─────────────────────────────────────────────
    train_scores = best_pipe.predict_proba(X_train_fit)[:, 1]
    test_scores  = y_prob_test
    psi_score    = psi(train_scores, test_scores)

    logger.info("PSI (train vs test score drift): %.4f", psi_score)
    if psi_score >= PSI_HIGH:
        logger.warning("HIGH DRIFT — PSI=%.4f — consider retraining", psi_score)
    elif psi_score >= PSI_MODERATE:
        logger.warning("MODERATE DRIFT — PSI=%.4f — monitor closely", psi_score)

    # ── Save experiment results CSV ───────────────────────────
    exp_path = os.path.join(MODEL_DIR, "model_experiment_results.csv")
    summary.to_csv(exp_path, index=False)
    logger.info("Experiment results saved → %s", exp_path)

    # ── Save monitor scores CSV ───────────────────────────────
    monitor_df = X_test.copy().reset_index(drop=True)
    monitor_df["churn_probability"] = y_prob_test
    monitor_df["label"]             = y_test.reset_index(drop=True)
    monitor_df["action"]            = actions

    monitor_path = os.path.join(MODEL_DIR, "monitor_scores.csv")
    monitor_df.to_csv(monitor_path, index=False)
    logger.info("Monitor scores saved → %s", monitor_path)

    # ── Save feature drift report ─────────────────────────────
    from src.metrics import simple_drift_report
    drift_series = simple_drift_report(X_train_fit, X_test, top_n=15)
    drift_df     = drift_series.reset_index()
    drift_df.columns = ["feature", "drift_score"]
    drift_path   = os.path.join(MODEL_DIR, "feature_drift_report.csv")
    drift_df.to_csv(drift_path, index=False)
    logger.info("Feature drift report saved → %s", drift_path)

    # ── Log to MLflow ─────────────────────────────────────────
    log_to_mlflow(best_name, metrics, active_threshold, best_pipe)

    # ── PIPELINE COMPLETE ──────────────────────────────────────
    elapsed = time.time() - start_time

    logger.info("=" * 65)
    logger.info("  TRAINING PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("  Best model    : %s", best_name)
    logger.info("  F1 Score      : %.4f", metrics["test_f1"])
    logger.info("  ROC-AUC       : %.4f", metrics.get("roc_auc") or 0)
    logger.info("  Recall        : %.4f", metrics["recall"])
    logger.info("  Precision     : %.4f", metrics["precision"])
    logger.info("  Threshold     : %.4f", active_threshold)
    logger.info("  Revenue ROI   : %.2f", revenue_result["retention_roi"])
    logger.info("=" * 65)

    return {
        "best_name":      best_name,
        "metrics":        metrics,
        "threshold":      active_threshold,
        "model_path":     model_path,
        "revenue_impact": revenue_result,
        "challenger":     challenger_result,
        "psi_score":      psi_score,
        "summary":        summary,
    }