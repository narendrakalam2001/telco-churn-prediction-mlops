# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — Telco Churn ML System
# ============================================================
# Champion vs Challenger with 4 promotion gates.
#
# BUG FIX: _load_champion_metrics() now reads model_card_path
# directly from latest_model.json instead of reconstructing
# the card path via brittle string parsing.
# Previous brittle pattern:
#   parts = model_name.replace("churn_model_","").replace(".joblib","")
#   card_path = f"model_card_{parts}.json"
# This breaks if naming conventions change. Direct path is robust.
#
# Promotion gates (churn-specific):
#   Gate 1: F1 improvement  >= 0.5%   (MIN_F1_IMPROVEMENT)
#   Gate 2: ROC-AUC         >= 0.80   (MIN_ROCAUC_THRESHOLD)
#   Gate 3: Recall          >= 0.70   (MIN_RECALL_THRESHOLD — churn-specific)
#   Gate 4: train-test gap  <= 10%    (MAX_GENERALIZATION_GAP)
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import (
    MODEL_DIR,
    MIN_F1_IMPROVEMENT,
    MIN_ROCAUC_THRESHOLD,
    MIN_RECALL_THRESHOLD,
    MAX_GENERALIZATION_GAP,
)

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """
    Reads churn_models/latest_model.json → loads .joblib + threshold.

    Returns:
        (model_pipeline, threshold)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "Run train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model     = joblib.load(model_path)
    threshold = float(registry.get("threshold", 0.5))

    logger.info(
        "Champion model loaded: %s  |  threshold=%.4f",
        model_path, threshold
    )

    return model, threshold


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    """
    Reads the current champion model card to get its metrics.
    Returns empty dict if no champion exists yet.

    FIX: reads model_card_path directly from latest_model.json
    instead of reconstructing it via string manipulation.
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    # FIX: use stored card path directly — no string parsing
    card_path = registry.get("model_card_path", "")

    # Fallback: legacy registries without model_card_path field
    if not card_path or not os.path.exists(card_path):
        model_name = registry.get("model_name", "")
        parts      = model_name.replace("churn_model_", "").replace(".joblib", "")
        card_path  = os.path.join(MODEL_DIR, f"model_card_{parts}.json")

    if not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", card)

    return {
        "model_name": card.get("model_name", "unknown"),
        "f1":         float(metrics.get("test_f1",     0)),
        "roc_auc":    float(metrics.get("roc_auc",     0)),
        "recall":     float(metrics.get("test_recall", 0)),
        "threshold":  float(card.get("thresholds", {}).get("active", 0.5)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    challenger_name:       str,
    challenger_f1:         float,
    challenger_roc_auc:    float,
    challenger_recall:     float,
    challenger_gap:        float,
    challenger_model_path: str,
    challenger_threshold:  float,
    challenger_card_path:  str  = None,
    _pre_loaded_champion:  dict = None,
) -> dict:
    """
    Compares challenger vs current champion on 4 gates.

    Args:
        challenger_name       : model name e.g. 'CatBoost'
        challenger_f1         : challenger test F1 score
        challenger_roc_auc    : challenger ROC-AUC
        challenger_recall     : challenger test recall
        challenger_gap        : challenger train-test accuracy gap
        challenger_model_path : path to challenger .joblib
        challenger_threshold  : challenger decision threshold
        challenger_card_path  : path to challenger model card JSON (FIX)
        _pre_loaded_champion  : optional pre-loaded champion dict.
                                MUST be passed from training_pipeline to avoid
                                the bug where save_model_and_registry() is called
                                first → latest_model.json already points to new
                                model → _load_champion_metrics() returns new
                                model itself → F1 improvement = 0.0 → REJECTED.

    Returns:
        result dict with decision: 'PROMOTED' or 'REJECTED'
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _pre_loaded_champion if _pre_loaded_champion is not None \
               else _load_champion_metrics()

    # ── No champion yet → auto-promote first model ────────────
    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_threshold, challenger_card_path)

        result = {
            "decision":              "PROMOTED",
            "reason":                "No existing champion — first model auto-promoted",
            "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
            "challenger_name":       challenger_name,
            "challenger_f1":         round(challenger_f1,      4),
            "challenger_roc_auc":    round(challenger_roc_auc, 4),
            "challenger_recall":     round(challenger_recall,  4),
            "challenger_gap":        round(challenger_gap,     4),
            "champion_name":         None,
            "champion_f1":           None,
            "champion_roc_auc":      None,
            "champion_recall":       None,
            "gates": {
                "f1_improvement_passed": True,
                "roc_auc_passed":        True,
                "recall_passed":         True,
                "gap_passed":            True,
            }
        }
        _save_challenger_log(result)
        return result

    champion_f1     = champion.get("f1",       0.0)
    champion_roc    = champion.get("roc_auc",  0.0)
    champion_recall = champion.get("recall",   0.0)
    champion_name   = champion.get("model_name", "unknown")

    logger.info("=" * 60)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info(
        "  Champion  : %-20s  F1=%.4f  ROC=%.4f  Recall=%.4f",
        champion_name, champion_f1, champion_roc, champion_recall
    )
    logger.info(
        "  Challenger: %-20s  F1=%.4f  ROC=%.4f  Recall=%.4f",
        challenger_name, challenger_f1, challenger_roc_auc, challenger_recall
    )
    logger.info("=" * 60)

    # ── 4 Promotion gates ─────────────────────────────────────
    gate1_f1_improvement = (challenger_f1 - champion_f1) >= MIN_F1_IMPROVEMENT
    gate2_roc_auc        = challenger_roc_auc >= MIN_ROCAUC_THRESHOLD
    gate3_recall         = challenger_recall  >= MIN_RECALL_THRESHOLD
    gate4_gap            = challenger_gap     <= MAX_GENERALIZATION_GAP

    gates_passed = gate1_f1_improvement and gate2_roc_auc and gate3_recall and gate4_gap

    if gates_passed:
        decision = "PROMOTED"
        reason   = (
            f"Challenger beats champion: "
            f"F1 {champion_f1:.4f} → {challenger_f1:.4f} "
            f"(+{challenger_f1 - champion_f1:.4f})"
        )
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_threshold, challenger_card_path)

    else:
        decision = "REJECTED"
        failed   = []

        if not gate1_f1_improvement:
            failed.append(
                f"F1 improvement {challenger_f1 - champion_f1:+.4f} < {MIN_F1_IMPROVEMENT}"
            )
        if not gate2_roc_auc:
            failed.append(f"ROC-AUC {challenger_roc_auc:.4f} < {MIN_ROCAUC_THRESHOLD}")
        if not gate3_recall:
            failed.append(f"Recall {challenger_recall:.4f} < {MIN_RECALL_THRESHOLD}")
        if not gate4_gap:
            failed.append(f"train-test gap {challenger_gap:.4f} > {MAX_GENERALIZATION_GAP}")

        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":              decision,
        "reason":                reason,
        "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":       challenger_name,
        "challenger_f1":         round(challenger_f1,      4),
        "challenger_roc_auc":    round(challenger_roc_auc, 4),
        "challenger_recall":     round(challenger_recall,  4),
        "challenger_gap":        round(challenger_gap,     4),
        "champion_name":         champion_name,
        "champion_f1":           round(champion_f1,        4),
        "champion_roc_auc":      round(champion_roc,       4),
        "champion_recall":       round(champion_recall,    4),
        "gates": {
            "f1_improvement_passed": gate1_f1_improvement,
            "roc_auc_passed":        gate2_roc_auc,
            "recall_passed":         gate3_recall,
            "gap_passed":            gate4_gap,
        }
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, threshold: float, model_card_path: str = None):
    """
    Updates latest_model.json with new champion.
    FIX: stores model_card_path directly — no string reconstruction needed.
    """
    registry = {
        "model_name":       os.path.basename(model_path),
        "threshold":        round(threshold, 4),
        "model_card_path":  model_card_path or "",
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s", registry["model_name"])


def _to_serializable(obj):
    """
    Recursively convert numpy types → native Python types for JSON.
    Fixes: numpy bool_ → bool, numpy int64 → int, numpy float64 → float.
    """
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(i) for i in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _save_challenger_log(result: dict):
    """
    Appends challenger comparison result to challenger_log.json.
    Converts numpy types to native Python before json.dump().
    """
    history = []

    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    result = _to_serializable(result)
    history.append(result)

    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved → %s", CHALLENGER_LOG)