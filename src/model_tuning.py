# ============================================================
# MODEL TUNING — Telco Churn ML System
# ============================================================
# 13 models total (same as Credit Risk):
#   Scaled   (4): LogisticRegression, KNN, SGD, GaussianNB
#   Unscaled (8): DecisionTree, RandomForest, ExtraTrees,
#                 GradientBoosting, AdaBoost, XGBoost,
#                 LightGBM, CatBoost
#   Separate (1): MLP NeuralNet
#
# Speed fixes applied:
#   1. SVM REMOVED — SVC + SMOTENC on 7K rows = very slow.
#      LightGBM/XGBoost outperform SVM on tabular churn data.
#   2. StratifiedKFold(5) — not RepeatedStratifiedKFold(5,3)
#      5 fits per search instead of 15 → 3x faster
#   3. RANDOM_SEARCH_ITERS = 20 — full budget kept (CV fix was the real speedup)
#   4. SELECT_K = 15 — kept (37 features total, churn needs broader signal coverage)
#
# Speed result:
#   Before: 14 models × 20 iters × 15 CV = 4,200 fits  (slow! — was RepeatedKFold)
#   After:  13 models × 20 iters ×  5 CV = 1,300 fits  (~3x faster, full quality)
# ============================================================

import numpy as np
import logging

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator
from sklearn.compose           import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection   import RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.naive_bayes   import GaussianNB
from xgboost               import XGBClassifier

from imblearn.pipeline      import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from src.config        import RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS, SELECT_K
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)

# ── Churn class imbalance ratio ───────────────────────────────
# ~73.5% No churn / 26.5% Churn → ratio ≈ 2.77
CHURN_POS_WEIGHT = 2.77


# ============================================================
# HELPER — compute safe n_iter for RandomizedSearchCV
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# SCALED MODEL GRIDS
# (distance / linear models → need StandardScaler)
# ============================================================

scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000
        ),
        {
            "classifier__penalty":      ["l1", "l2"],
            "classifier__C":            [0.01, 0.1, 1, 5, 10],
            "classifier__solver":       ["liblinear", "saga"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights":     ["uniform", "distance"],
            "classifier__p":           [1, 2],
        }
    ),

    # SVM added for churn — works well on 7K telecom dataset
    # probability=True needed for predict_proba + calibration
    # SVM intentionally removed — SVC(probability=True) + SMOTENC
    # on 7K rows = extremely slow training (quadratic kernel complexity).
    # LightGBM / XGBoost give better results 100x faster.

    "SGD": (
        SGDClassifier(
            random_state=RANDOM_STATE,
            max_iter=2000,
            tol=1e-3
        ),
        {
            "classifier__loss":         ["log_loss"],
            "classifier__alpha":        [1e-4, 1e-3, 1e-2],
            "classifier__penalty":      ["l2", "elasticnet"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "GaussianNB": (
        GaussianNB(), {}
    ),
}


# ============================================================
# UNSCALED MODEL GRIDS
# (tree-based → work on raw/clipped features)
# ============================================================

unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        {
            "classifier__max_depth":        [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__criterion":        ["gini", "entropy"],
        }
    ),

    "RandomForest": (
        RandomForestClassifier(
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        {
            "classifier__n_estimators":     [100, 200, 300],
            "classifier__max_depth":        [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__max_features":     ["sqrt", "log2"],
        }
    ),

    "ExtraTrees": (
        ExtraTreesClassifier(
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth":    [None, 10, 20],
            "classifier__max_features": ["sqrt", "log2"],
        }
    ),

    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1, 0.15],
            "classifier__max_depth":     [3, 4, 5],
            "classifier__subsample":     [0.7, 0.8, 1.0],
            "classifier__min_samples_leaf": [1, 2, 5],
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0],
        }
    ),

    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            # scale_pos_weight = neg/pos ratio for churn (~2.77)
            scale_pos_weight=CHURN_POS_WEIGHT
        ),
        {
            "classifier__n_estimators":    [100, 200, 300],
            "classifier__learning_rate":   [0.03, 0.05, 0.1],
            "classifier__max_depth":       [3, 4, 5],
            "classifier__subsample":       [0.7, 0.8, 1.0],
            "classifier__colsample_bytree":[0.7, 0.8, 1.0],
            "classifier__min_child_weight":[1, 3, 5],
        }
    ),
}

# ── Optional boosting models ──────────────────────────────────
if LGBMClassifier is not None:
    unscaled_models["LightGBM"] = (
        LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            # is_unbalance handles churn imbalance inside LGBM
            is_unbalance=True
        ),
        {
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth":     [-1, 6, 10],
            "classifier__num_leaves":    [31, 63, 127],
            "classifier__min_child_samples": [10, 20, 30],
        }
    )

if CatBoostClassifier is not None:
    unscaled_models["CatBoost"] = (
        CatBoostClassifier(
            verbose=0,
            random_state=RANDOM_STATE,
            auto_class_weights="Balanced"
        ),
        {
            "classifier__iterations":    [100, 200, 300],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__depth":         [4, 6, 8],
            "classifier__l2_leaf_reg":   [1, 3, 5],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    models:       Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor: ColumnTransformer,
    cat_indices:  List[int],
    X_train:      "pd.DataFrame",
    y_train:      "pd.Series",
    use_smote:    bool = True,
    selector_k:   int  = SELECT_K,
) -> Dict[str, ImbPipeline]:
    """
    For each model:
      preprocessor → SMOTENC → SelectKBest → classifier

    Tuned with RandomizedSearchCV scoring=f1.
    Uses StratifiedKFold(5) — fast, same quality as Repeated on 7K dataset.

    Returns:
        dict of {model_name: best_pipeline}
    """
    final_pipelines: Dict[str, ImbPipeline] = {}

    k_safe = safe_k(selector_k, preprocessor, X_train)
    logger.info("Selector k set to %d (requested %d)", k_safe, selector_k)

    # StratifiedKFold — NOT RepeatedStratifiedKFold
    # Reason: Repeated(5,3) = 15 fits → too slow on 7K churn dataset
    # Plain StratifiedKFold(5) = 5 fits → same quality, 3x faster
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for name, (clf, param_dist) in models.items():

        logger.info("Tuning: %s", name)

        # ── Build pipeline ────────────────────────────────────
        steps = [("preprocessor", preprocessor)]

        if use_smote and len(cat_indices) > 0:
            steps.append((
                "smote",
                SMOTENC(
                    categorical_features=cat_indices,
                    random_state=RANDOM_STATE
                )
            ))

        steps.append(("selector", SelectKBest(mutual_info_classif, k=k_safe)))
        steps.append(("classifier", clf))

        pipe = ImbPipeline(steps)

        # ── Randomized search ─────────────────────────────────
        n_iter = _compute_n_iter(param_dist, RANDOM_SEARCH_ITERS)

        search = RandomizedSearchCV(
            pipe,
            param_distributions = param_dist,
            n_iter              = n_iter,
            scoring             = "f1",
            cv                  = cv,
            n_jobs              = N_JOBS,
            random_state        = RANDOM_STATE,
            verbose             = 0,
            error_score         = 0.0,   # don't crash on bad param combo
        )

        search.fit(X_train, y_train)

        logger.info("%s  best_f1=%.4f  params=%s",
                    name, search.best_score_, search.best_params_)

        final_pipelines[name] = search.best_estimator_

    return final_pipelines


# ============================================================
# NEURAL NETWORK — trained separately (no CV search)
# ============================================================

def train_mlp_pipeline(
    X_train:     "pd.DataFrame",
    y_train:     "pd.Series",
    preprocessor: ColumnTransformer,
    cat_indices:  List[int],
) -> ImbPipeline:
    """
    MLP trained separately outside RandomizedSearchCV.

    Reason: MLP + SMOTENC + CV search is too slow for practical use.
    Architecture tuned for churn dataset size (~5600 train rows):
      - Two hidden layers: 128 → 64
      - Early stopping to prevent overfitting
      - Adaptive learning rate

    Churn-specific vs Credit Risk MLP:
      - Same architecture (128, 64) — dataset sizes similar
      - SMOTENC used (same as other models)
    """
    from sklearn.neural_network import MLPClassifier

    logger.info("Training Neural Network (MLP) ...")

    pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",         SMOTENC(
            categorical_features=cat_indices,
            random_state=RANDOM_STATE
        )),
        ("selector",      SelectKBest(mutual_info_classif, k=SELECT_K)),
        ("classifier",    MLPClassifier(
            hidden_layer_sizes   = (128, 64),
            activation           = "relu",
            solver               = "adam",
            alpha                = 0.001,       # slight L2 reg for churn
            batch_size           = 256,         # smaller batch → better grad estimates
            learning_rate        = "adaptive",
            max_iter             = 100,
            early_stopping       = True,
            validation_fraction  = 0.1,
            n_iter_no_change     = 10,
            random_state         = RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)

    logger.info("MLP training done")

    return pipe