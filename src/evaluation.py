# ============================================================
# EVALUATION — Telco Churn ML System
# ============================================================
# Full evaluation pipeline after model training.
#
# BUG FIX: save_model_and_registry() now accepts model_card_path
# and stores it directly in latest_model.json. This eliminates
# the brittle string-parsing pattern in model_loader.py that
# reconstructed the card path from the model filename.
# ============================================================

import os
import json
import logging
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix,
    classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.config       import MODEL_DIR, RANDOM_STATE, CV_FOLDS
from src.metrics      import (
    tune_threshold, ks_statistic, psi,
    recall_at_k, lift_at_k,
    revenue_impact_evaluation
)
from src.churn_engine import churn_engine
from src.model_card   import build_model_card, save_model_card

logger = logging.getLogger(__name__)

MAX_GAP = 0.10


# ============================================================
# HELPER — safe predict_proba
# ============================================================

def _safe_proba(pipe, X) -> np.ndarray:
    try:
        return pipe.predict_proba(X)[:, 1]
    except Exception:
        return None


# ============================================================
# STEP 1 — EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    all_pipelines: dict,
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
) -> pd.DataFrame:
    """
    Evaluates all trained pipelines on test set.

    Metrics computed:
      F1, precision, recall, ROC-AUC, PR-AUC, KS, Brier,
      recall@10%, recall@20%, lift@20%,
      CV mean/std F1, train-test gap

    Returns:
        summary DataFrame sorted by cv_mean_f1 desc,
        cv_std_f1 asc, recall desc
    """
    rows = []
    cv   = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, pipe in all_pipelines.items():
        try:
            y_prob       = _safe_proba(pipe, X_test)
            y_pred       = pipe.predict(X_test)

            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                scoring="f1", cv=cv, n_jobs=1, error_score=0.0
            )

            train_f1 = f1_score(y_train, pipe.predict(X_train), zero_division=0)
            test_f1  = f1_score(y_test,  y_pred,                zero_division=0)
            gap      = round(abs(train_f1 - test_f1), 4)

            row = {
                "model":      name,
                "cv_mean_f1": round(cv_scores.mean(), 4),
                "cv_std_f1":  round(cv_scores.std(),  4),
                "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1":         round(test_f1, 4),
                "train_f1":   round(train_f1, 4),
                "gap":        gap,
                "roc_auc":    round(roc_auc_score(y_test, y_prob), 4)            if y_prob is not None else None,
                "pr_auc":     round(average_precision_score(y_test, y_prob), 4)  if y_prob is not None else None,
                "ks":         round(ks_statistic(y_test, y_prob), 4)             if y_prob is not None else None,
                "brier":      round(brier_score_loss(y_test, y_prob), 4)         if y_prob is not None else None,
                "recall@10%": round(recall_at_k(y_test, y_prob, 0.10), 4)       if y_prob is not None else None,
                "recall@20%": round(recall_at_k(y_test, y_prob, 0.20), 4)       if y_prob is not None else None,
                "lift@20%":   round(lift_at_k(y_test,   y_prob, 0.20), 4)       if y_prob is not None else None,
            }
            rows.append(row)

            logger.info(
                "%-20s  cv_f1=%.4f±%.4f  test_f1=%.4f  recall=%.4f  gap=%.4f",
                name, cv_scores.mean(), cv_scores.std(), test_f1,
                row["recall"], gap
            )

        except Exception as e:
            logger.warning("Evaluation failed for %s: %s", name, e)

    summary = (
        pd.DataFrame(rows)
        .sort_values(
            ["cv_mean_f1", "cv_std_f1", "recall"],
            ascending=[False, True, False]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "\n%s",
        summary[["model","cv_mean_f1","f1","recall","roc_auc","gap"]].to_string()
    )

    return summary


# ============================================================
# STEP 2 — GENERALIZATION FILTER
# ============================================================

def apply_generalization_filter(
    summary:       pd.DataFrame,
    all_pipelines: dict,
    max_gap:       float = MAX_GAP,
) -> tuple:
    """
    Removes models with train-test gap > max_gap.
    Falls back to full list if all fail the filter.

    Returns:
        (filtered_summary, filtered_pipelines)
    """
    filtered = summary[summary["gap"] <= max_gap].reset_index(drop=True)

    if filtered.empty:
        logger.warning(
            "All models exceeded gap=%.2f — using full list as fallback", max_gap
        )
        filtered = summary.copy()

    removed = set(summary["model"]) - set(filtered["model"])
    if removed:
        logger.info("Generalization filter removed: %s", removed)

    filtered_pipelines = {
        name: pipe
        for name, pipe in all_pipelines.items()
        if name in set(filtered["model"])
    }

    return filtered, filtered_pipelines


# ============================================================
# STEP 3 — PROBABILITY CALIBRATION
# ============================================================

def calibrate_model(
    best_pipe,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> tuple:
    """
    Holdout isotonic regression calibration.

    Returns:
        (calibrated_or_original_pipeline, calibration_succeeded_bool)
    """
    try:
        cal_model = CalibratedClassifierCV(
            best_pipe, method="isotonic", cv="prefit"
        )
        cal_model.fit(X_cal, y_cal)
        logger.info("Probability calibration successful (isotonic)")
        return cal_model, True

    except Exception as e:
        logger.warning("Calibration failed: %s — using uncalibrated model", e)
        return best_pipe, False


# ============================================================
# STEP 4 — FULL BEST-MODEL EVALUATION
# ============================================================

def evaluate_best_model(
    best_name: str,
    best_pipe,
    X_train:   pd.DataFrame,
    y_train:   pd.Series,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    threshold: float,
) -> dict:
    """
    Full metrics for selected best model at tuned threshold.

    Returns:
        metrics dict → used by model_card and MLflow
    """
    y_prob   = _safe_proba(best_pipe, X_test)
    y_pred   = (y_prob >= threshold).astype(int) if y_prob is not None else best_pipe.predict(X_test)
    train_f1 = f1_score(y_train, best_pipe.predict(X_train), zero_division=0)
    test_f1  = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "test_f1":      round(test_f1, 4),
        "precision":    round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":       round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":      round(roc_auc_score(y_test, y_prob), 4)           if y_prob is not None else None,
        "pr_auc":       round(average_precision_score(y_test, y_prob), 4) if y_prob is not None else None,
        "ks":           round(ks_statistic(y_test, y_prob), 4)            if y_prob is not None else None,
        "brier":        round(brier_score_loss(y_test, y_prob), 4)        if y_prob is not None else None,
        "recall_at_10": round(recall_at_k(y_test, y_prob, 0.10), 4)      if y_prob is not None else None,
        "recall_at_20": round(recall_at_k(y_test, y_prob, 0.20), 4)      if y_prob is not None else None,
        "lift_at_20":   round(lift_at_k(y_test, y_prob, 0.20), 4)        if y_prob is not None else None,
        "train_f1":     round(train_f1, 4),
        "gap":          round(abs(train_f1 - test_f1), 4),
    }

    logger.info("=" * 55)
    logger.info("BEST MODEL FINAL METRICS — %s  (threshold=%.4f)", best_name, threshold)
    for k, v in metrics.items():
        logger.info("  %-20s = %s", k, v)
    logger.info("=" * 55)

    logger.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=["Stay", "Churn"])
    )

    return metrics


# ============================================================
# STEP 5 — PLOTS
# ============================================================

def plot_evaluation_charts(
    best_name: str,
    best_pipe,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    threshold: float,
    save_dir:  str = "docs/plots",
) -> None:
    """Saves confusion matrix + ROC/PR curves."""
    os.makedirs(save_dir, exist_ok=True)

    y_prob = _safe_proba(best_pipe, X_test)
    y_pred = (y_prob >= threshold).astype(int) if y_prob is not None else best_pipe.predict(X_test)

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax
    )
    ax.set_title(f"{best_name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── ROC + PR curves ───────────────────────────────────────
    if y_prob is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name=best_name)
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=best_name)
        axes[0].set_title(f"{best_name} — ROC Curve")
        axes[1].set_title(f"{best_name} — Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Evaluation charts saved → %s", save_dir)


# ============================================================
# STEP 6 — SHAP EXPLAINABILITY
# ============================================================

def compute_shap(
    best_name:   str,
    best_pipe,
    X_train:     pd.DataFrame,
    X_test:      pd.DataFrame,
    save_dir:    str = "docs/plots",
    max_display: int = 15,
) -> dict:
    """
    TreeExplainer for tree models, KernelExplainer fallback for others.
    Returns dict of {feature: mean_abs_shap}.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available — skipping")
        return {}

    os.makedirs(save_dir, exist_ok=True)

    try:
        from sklearn.calibration import CalibratedClassifierCV as _CalCV
        if isinstance(best_pipe, _CalCV):
            base_pipe = best_pipe.estimator
        else:
            base_pipe = best_pipe

        preprocessed_train = base_pipe[:-1].transform(X_train)
        preprocessed_test  = base_pipe[:-1].transform(X_test)
        clf                = base_pipe.named_steps["classifier"]

        tree_keywords = ["Forest", "Tree", "Boost", "LGBM", "Cat", "XGB", "Ada", "Extra"]
        model_type    = type(clf).__name__

        if any(k in model_type for k in tree_keywords):
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(preprocessed_test)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            sample      = shap.sample(preprocessed_train, 100)
            explainer   = shap.KernelExplainer(clf.predict_proba, sample)
            shap_values = explainer.shap_values(preprocessed_test[:100])
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        try:
            feat_names = base_pipe[:-1].get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(sv.shape[1])]

        mean_shap = np.abs(sv).mean(axis=0)
        shap_df   = (
            pd.Series(mean_shap, index=feat_names)
            .sort_values(ascending=False)
            .head(max_display)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        shap_df[::-1].plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"{best_name} — SHAP Feature Importance (Top {max_display})")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("SHAP computed — top feature: %s", shap_df.index[0])
        return shap_df.to_dict()

    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return {}


# ============================================================
# STEP 7 — MLFLOW LOGGING
# ============================================================

def log_to_mlflow(
    best_name: str,
    metrics:   dict,
    threshold: float,
    best_pipe,
    params:    dict = None,
) -> None:
    """Logs experiment to MLflow."""
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_experiment("telco_churn_prediction")

        with mlflow.start_run(run_name=f"churn_{best_name}_{int(time.time())}"):
            mlflow.log_param("model_name", best_name)
            mlflow.log_param("threshold",  round(threshold, 4))

            if params:
                for k, v in params.items():
                    mlflow.log_param(k, v)

            for k, v in metrics.items():
                if v is not None:
                    mlflow.log_metric(k, float(v))

            mlflow.sklearn.log_model(
                sk_model         = best_pipe,
                name             = f"churn_model_{best_name}",
                pip_requirements = ["scikit-learn", "lightgbm", "xgboost"],
            )

        logger.info("MLflow run logged for %s", best_name)

    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


# ============================================================
# STEP 8 — SAVE MODEL + REGISTRY
# ============================================================

def save_model_and_registry(
    best_pipe,
    best_name:       str,
    threshold:       float,
    version:         str  = "v1",
    model_card_path: str  = None,
) -> str:
    """
    Saves model .joblib + updates latest_model.json.

    FIX: model_card_path is now stored directly in latest_model.json.
    This eliminates the brittle string-parsing in model_loader.py
    that reconstructed the card path from the model filename.

    Returns model_path.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_filename = f"churn_model_{best_name}_{version}.joblib"
    model_path     = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(best_pipe, model_path)

    registry = {
        "model_name":       model_filename,
        "threshold":        round(float(threshold), 4),
        "model_card_path":  model_card_path or "",
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Model saved → %s  |  threshold=%.4f", model_path, threshold)

    return model_path


# ============================================================
# STEP 9 — FEATURE IMPORTANCES
# ============================================================

def get_feature_importances(best_pipe, top_n: int = 15) -> dict:
    """
    Extracts feature importances from classifier.
    Works for tree models (feature_importances_) and
    linear models (coef_).

    Handles CalibratedClassifierCV wrapper — unwraps to base pipeline.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV as _CalCV
        if isinstance(best_pipe, _CalCV):
            base_pipe = best_pipe.estimator
        else:
            base_pipe = best_pipe

        clf = base_pipe.named_steps["classifier"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return {}

        try:
            feat_names = base_pipe[:-1].get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(len(importances))]

        fi_series = (
            pd.Series(importances, index=feat_names)
            .sort_values(ascending=False)
            .head(top_n)
        )
        return fi_series.round(6).to_dict()

    except Exception as e:
        logger.warning("Feature importance extraction failed: %s", e)
        return {}