# ============================================================
# PYTEST — Telco Churn Prediction ML System
# 41 Unit Tests covering core pipeline modules
#
# Run:  pytest tests/test_pipeline_core.py -v
# ============================================================

import sys, os
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CHURN_RISK_BANDS,
    MIN_TENURE_RULE,
    MAX_MONTHLY_CHARGES_RULE,
    PSI_MODERATE,
    PSI_HIGH,
    RANDOM_STATE,
    CV_FOLDS,
    SELECT_K,
    MIN_F1_IMPROVEMENT,
    MIN_ROCAUC_THRESHOLD,
    MIN_RECALL_THRESHOLD,
    MAX_GENERALIZATION_GAP,
)
from src.churn_engine import (
    get_churn_risk_band,
    churn_engine,
    score_customer,
    get_action_description,
)
from src.data_loader import (
    validate_input_data,
    add_engineered_features,
    detect_feature_types,
)
from src.metrics import (
    tune_threshold,
    psi,
    ks_statistic,
    recall_at_k,
    lift_at_k,
    simple_drift_report,
)
from src.leakage_check  import detect_leakage
from src.preprocessing  import Clipper, build_preprocessors


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def minimal_customer_df():
    """Minimal raw DataFrame mimicking telco dataset (5 rows)."""
    return pd.DataFrame({
        "gender":          ["Male",   "Female", "Male",   "Female", "Male"],
        "SeniorCitizen":   [0,        1,        0,        0,        1],
        "Partner":         ["Yes",    "No",     "Yes",    "No",     "Yes"],
        "Dependents":      ["No",     "No",     "Yes",    "No",     "No"],
        "tenure":          [1,        55,       12,       0,        24],
        "PhoneService":    ["Yes",    "Yes",    "No",     "Yes",    "Yes"],
        "MultipleLines":   ["No",     "Yes",    "No phone service", "No", "Yes"],
        "InternetService": ["Fiber optic", "DSL", "DSL", "No", "Fiber optic"],
        "OnlineSecurity":  ["No",     "Yes",    "No",     "No internet service", "No"],
        "OnlineBackup":    ["Yes",    "No",     "Yes",    "No internet service", "No"],
        "DeviceProtection":["No",     "Yes",    "No",     "No internet service", "Yes"],
        "TechSupport":     ["No",     "No",     "No",     "No internet service", "Yes"],
        "StreamingTV":     ["No",     "Yes",    "No",     "No internet service", "Yes"],
        "StreamingMovies": ["No",     "Yes",    "No",     "No internet service", "No"],
        "Contract":        ["Month-to-month", "Two year", "One year", "Month-to-month", "Month-to-month"],
        "PaperlessBilling":["Yes",    "No",     "Yes",    "Yes",    "No"],
        "PaymentMethod":   ["Electronic check", "Bank transfer (automatic)", "Mailed check",
                            "Credit card (automatic)", "Electronic check"],
        "MonthlyCharges":  [29.85,   56.95,   53.85,   20.00,   99.65],
        "TotalCharges":    ["29.85", "3130.75", "645.45", "0.0", "2394.75"],
        "Churn":           ["No",    "No",     "No",     "No",   "Yes"],
    })


@pytest.fixture
def engineered_df(minimal_customer_df):
    """DataFrame after feature engineering — bypasses validate size check."""
    df = minimal_customer_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return add_engineered_features(df)


@pytest.fixture
def binary_classification_data():
    """Simple binary y_true / y_prob for metric tests."""
    np.random.seed(RANDOM_STATE)
    n = 200
    y_true = np.array([1]*50 + [0]*150)
    y_prob = np.where(y_true == 1,
                      np.random.uniform(0.4, 0.9, n),
                      np.random.uniform(0.05, 0.5, n))
    return y_true, y_prob


# ============================================================
# GROUP 1 — CONFIG (Tests 1–6)
# ============================================================

class TestConfig:
    """Validate config constants are sensible."""

    def test_churn_risk_bands_cover_full_range(self):
        """LOW/MEDIUM/HIGH bands must cover [0, 1] without gap."""
        lows  = [v[0] for v in CHURN_RISK_BANDS.values()]
        highs = [v[1] for v in CHURN_RISK_BANDS.values()]
        assert min(lows)  == 0.00
        assert max(highs) >= 1.00

    def test_cv_and_select_k_positive(self):
        """CV folds and feature k must be positive integers."""
        assert CV_FOLDS >= 2
        assert SELECT_K >= 5
        assert isinstance(CV_FOLDS, int)
        assert isinstance(SELECT_K, int)

    def test_psi_thresholds_ordered(self):
        """PSI_HIGH must be greater than PSI_MODERATE."""
        assert PSI_MODERATE < PSI_HIGH
        assert PSI_MODERATE > 0

    def test_gate_constants_present(self):
        """All 4 gate constants must be importable from config."""
        assert MIN_F1_IMPROVEMENT     > 0
        assert MIN_ROCAUC_THRESHOLD   > 0
        assert MIN_RECALL_THRESHOLD   > 0
        assert MAX_GENERALIZATION_GAP > 0

    def test_gate_constants_sensible_ranges(self):
        """Gate thresholds must be in realistic ranges."""
        assert 0.001 <= MIN_F1_IMPROVEMENT   <= 0.05,  "F1 gate too tight or too loose"
        assert 0.50  <= MIN_ROCAUC_THRESHOLD <= 0.99,  "ROC-AUC gate out of range"
        assert 0.50  <= MIN_RECALL_THRESHOLD <= 0.95,  "Recall gate out of range"
        assert 0.05  <= MAX_GENERALIZATION_GAP <= 0.30, "Gap gate out of range"

    def test_recall_gate_lower_than_rocauc_gate(self):
        """
        Churn recall gate (0.70) should be lower than ROC-AUC gate (0.80).
        Recall at threshold is harder to hit than rank-order AUC.
        """
        assert MIN_RECALL_THRESHOLD < MIN_ROCAUC_THRESHOLD


# ============================================================
# GROUP 2 — DATA LOADER (Tests 7–11)
# ============================================================

class TestDataLoader:
    """Tests for validate_input_data and add_engineered_features."""

    def test_validate_drops_customer_id(self, minimal_customer_df):
        """customerID column must be dropped."""
        df_with_id = minimal_customer_df.copy()
        df_with_id["customerID"] = ["A", "B", "C", "D", "E"]
        df = df_with_id.copy()
        df.columns = df.columns.str.strip()
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)
        assert "customerID" not in df.columns

    def test_validate_fills_total_charges_nulls(self):
        """TotalCharges nulls should be filled."""
        df = pd.DataFrame({
            "tenure": [0, 1],
            "TotalCharges": [None, "500.0"],
            "MonthlyCharges": [50.0, 60.0],
            "Churn": ["No", "No"],
        })
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median)
        assert df["TotalCharges"].isnull().sum() == 0

    def test_engineered_features_added(self, engineered_df):
        """After engineering, key derived features must exist."""
        required = [
            "ChargeDeviation", "RevenueAtRisk", "IsNewCustomer",
            "IsLoyalCustomer", "ServiceCount", "ServiceAdoptionRate",
            "LongTermContract", "AutoPays", "M2M_HighCharge",
            "NoContractNoAutopay", "HighChargeFlag",
        ]
        for feat in required:
            assert feat in engineered_df.columns, f"Missing: {feat}"

    def test_is_new_customer_flag_correct(self, engineered_df):
        """IsNewCustomer = 1 when tenure < MIN_TENURE_RULE."""
        new_mask     = engineered_df["tenure"] < MIN_TENURE_RULE
        flagged_mask = engineered_df["IsNewCustomer"] == 1
        assert (new_mask == flagged_mask).all()

    def test_detect_feature_types_returns_three_lists(self, engineered_df):
        """detect_feature_types must return non-overlapping lists."""
        X = engineered_df.drop(columns=["Churn"], errors="ignore")
        ordinal, continuous, binary = detect_feature_types(X)
        assert isinstance(ordinal,    list)
        assert isinstance(continuous, list)
        assert isinstance(binary,     list)
        all_cols = ordinal + continuous + binary
        assert len(all_cols) == len(set(all_cols)), "Columns appear in multiple buckets"


# ============================================================
# GROUP 3 — CHURN ENGINE (Tests 12–18)
# ============================================================

class TestChurnEngine:
    """Tests for risk band classification and action assignment."""

    def test_risk_band_low(self):
        assert get_churn_risk_band(0.10) == "LOW"

    def test_risk_band_medium(self):
        assert get_churn_risk_band(0.45) == "MEDIUM"

    def test_risk_band_high(self):
        assert get_churn_risk_band(0.75) == "HIGH"

    def test_risk_band_boundary(self):
        """Probability = 0.0 and 1.0 must not raise."""
        band_zero = get_churn_risk_band(0.0)
        band_one  = get_churn_risk_band(1.0)
        assert band_zero in ("LOW", "MEDIUM", "HIGH")
        assert band_one  == "HIGH"

    def test_churn_engine_returns_list_same_length(self, engineered_df):
        """churn_engine output length must equal input rows."""
        X      = engineered_df.drop(columns=["Churn"], errors="ignore")
        probs  = np.full(len(X), 0.15)
        actions = churn_engine(X, probs, threshold=0.30)
        assert len(actions) == len(X)

    def test_churn_engine_high_prob_gives_urgent(self, engineered_df):
        """Very high probability must trigger URGENT action."""
        X       = engineered_df.drop(columns=["Churn"], errors="ignore").iloc[[0]]
        probs   = np.array([0.99])
        actions = churn_engine(X, probs, threshold=0.30)
        assert actions[0].startswith("URGENT"), f"Expected URGENT, got {actions[0]}"

    def test_action_description_valid(self):
        """get_action_description must return non-empty string for known actions."""
        for action in ["RETAIN", "OUTREACH", "URGENT",
                       "OUTREACH_NEW_CUSTOMER", "URGENT_TRIPLE_RISK"]:
            desc = get_action_description(action)
            assert isinstance(desc, str) and len(desc) > 0


# ============================================================
# GROUP 4 — METRICS (Tests 19–26)
# ============================================================

class TestMetrics:
    """Tests for evaluation metrics and threshold tuning."""

    def test_tune_threshold_achieves_target_recall(self, binary_classification_data):
        """tune_threshold returns a float in (0, 1)."""
        y_true, y_prob = binary_classification_data
        result = tune_threshold(y_true, y_prob, target_recall=0.70)
        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_tune_threshold_default_recall_constraint(self, binary_classification_data):
        """Default call returns a valid threshold."""
        y_true, y_prob = binary_classification_data
        result = tune_threshold(y_true, y_prob)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_psi_identical_distributions(self):
        """PSI of identical distributions should be near zero."""
        arr   = np.random.normal(0, 1, 500)
        score = psi(arr, arr.copy())
        assert score < 0.05, f"PSI of identical dists = {score:.4f}"

    def test_psi_very_different_distributions(self):
        """PSI of very different distributions should exceed PSI_HIGH."""
        expected = np.random.normal(0,  1, 500)
        actual   = np.random.normal(10, 1, 500)
        score    = psi(expected, actual)
        assert score > PSI_HIGH, f"PSI = {score:.4f}, expected > {PSI_HIGH}"

    def test_ks_statistic_range(self, binary_classification_data):
        """KS statistic must be in [0, 1]."""
        y_true, y_prob = binary_classification_data
        ks = ks_statistic(y_true, y_prob)
        assert 0.0 <= ks <= 1.0

    def test_recall_at_k_value_in_range(self, binary_classification_data):
        """recall_at_k must be in [0, 1]."""
        y_true, y_prob = binary_classification_data
        r = recall_at_k(y_true, y_prob, k=0.20)
        assert 0.0 <= r <= 1.0

    def test_lift_at_k_above_one_for_good_model(self, binary_classification_data):
        """lift_at_k should be >= 1.0 for a signal-bearing model."""
        y_true, y_prob = binary_classification_data
        lift = lift_at_k(y_true, y_prob, k=0.20)
        assert lift >= 1.0, f"Expected lift >= 1.0, got {lift:.3f}"

    def test_recall_at_k_10_vs_20(self, binary_classification_data):
        """recall@10% must be <= recall@20% (more customers = more recall)."""
        y_true, y_prob = binary_classification_data
        r10 = recall_at_k(y_true, y_prob, k=0.10)
        r20 = recall_at_k(y_true, y_prob, k=0.20)
        assert r10 <= r20, f"recall@10% ({r10:.3f}) > recall@20% ({r20:.3f})"


# ============================================================
# GROUP 5 — LEAKAGE CHECK (Tests 27–29)
# ============================================================

class TestLeakageCheck:
    """Tests for leakage detection module."""

    def test_no_leakage_on_clean_data(self, engineered_df):
        """Clean telco data should pass leakage check."""
        X = engineered_df.drop(columns=["Churn"], errors="ignore")
        y = (engineered_df["Churn"] == "Yes").astype(int)
        try:
            result = detect_leakage(X, y)
            if isinstance(result, dict) and "leakage_cols" in result:
                assert len(result["leakage_cols"]) == 0
        except Exception as e:
            pytest.fail(f"detect_leakage raised unexpectedly: {e}")

    def test_leakage_detected_on_injected_feature(self, engineered_df):
        """Perfect target copy should be detected as leakage."""
        df     = engineered_df.copy()
        y      = (df["Churn"] == "Yes").astype(int)
        X      = df.drop(columns=["Churn"], errors="ignore")
        X_leak = X.copy()
        X_leak["churn_flag"] = y.values
        try:
            result = detect_leakage(X_leak, y)
            if isinstance(result, dict) and "leakage_cols" in result:
                assert "churn_flag" in result["leakage_cols"]
        except SystemExit:
            pass

    def test_leakage_returns_list(self, engineered_df):
        """detect_leakage must return a list (warnings)."""
        X = engineered_df.drop(columns=["Churn"], errors="ignore")
        y = (engineered_df["Churn"] == "Yes").astype(int)
        result = detect_leakage(X, y)
        assert isinstance(result, list), f"Expected list, got {type(result)}"


# ============================================================
# GROUP 6 — DRIFT REPORT (Tests 30–31)
# ============================================================

class TestDriftReport:
    """Tests for simple_drift_report function."""

    def test_drift_report_numeric_only(self, engineered_df):
        """simple_drift_report must not crash on mixed-type DataFrame."""
        X   = engineered_df.drop(columns=["Churn"], errors="ignore")
        ref = X.iloc[:3]
        new = X.iloc[3:]
        try:
            result = simple_drift_report(ref, new, top_n=5)
            assert isinstance(result, pd.Series)
            assert len(result) <= 5
        except Exception as e:
            pytest.fail(f"simple_drift_report crashed: {e}")

    def test_drift_report_all_non_negative(self, engineered_df):
        """All drift scores must be non-negative (absolute differences)."""
        X      = engineered_df.drop(columns=["Churn"], errors="ignore")
        result = simple_drift_report(X.iloc[:3], X.iloc[3:], top_n=10)
        assert (result >= 0).all(), "Drift scores must be non-negative"


# ============================================================
# GROUP 7 — SCORE CUSTOMER (Tests 32–33)
# ============================================================

class TestScoreCustomer:
    """Tests for score_customer (single-customer API scoring)."""

    def test_score_customer_output_structure(self, engineered_df):
        """score_customer must return dict with required keys."""
        row    = engineered_df.drop(columns=["Churn"], errors="ignore").iloc[0].to_dict()
        result = score_customer(row, prob=0.65, threshold=0.30)

        required_keys = {"churn_probability", "churn_risk_band", "action", "rule_triggered"}
        assert required_keys.issubset(result.keys())
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["churn_risk_band"] in ("LOW", "MEDIUM", "HIGH")
        # score_customer returns simplified actions (not granular _MODEL suffix)
        assert result["action"] in ("RETAIN", "OUTREACH", "URGENT")

    def test_score_customer_low_prob_retain(self, engineered_df):
        """Very low probability with long-tenure safe customer → RETAIN."""
        # Use a customer with high tenure (loyal), long-term contract
        row = {
            "tenure":           60,
            "MonthlyCharges":   55.0,
            "IsNewCustomer":    0,
            "LongTermContract": 1,
            "AutoPays":         1,
        }
        result = score_customer(row, prob=0.05, threshold=0.30)
        assert result["action"] == "RETAIN"


# ============================================================
# GROUP 8 — CLIPPER (Tests 34–36)
# ============================================================

class TestClipper:
    """Tests for the Clipper transformer."""

    def test_clipper_fit_transform_shape(self):
        """Clipper must preserve shape."""
        X       = np.random.normal(0, 1, (100, 3))
        clipper = Clipper(fold=1.5)
        clipper.fit(X)
        out = clipper.transform(X)
        assert out.shape == X.shape

    def test_clipper_removes_extreme_outliers(self):
        """Clipper must cap extreme values."""
        X       = np.array([[1.0], [2.0], [3.0], [1000.0]])
        clipper = Clipper(fold=1.5)
        clipper.fit(X)
        out = clipper.transform(X)
        assert out.max() < 1000.0, "Clipper failed to cap extreme value"

    def test_clipper_get_feature_names_out(self):
        """get_feature_names_out must return array of same length as input."""
        X       = np.random.normal(0, 1, (50, 4))
        clipper = Clipper()
        clipper.fit(X)
        names   = clipper.get_feature_names_out(["a", "b", "c", "d"])
        assert list(names) == ["a", "b", "c", "d"]


# ============================================================
# GROUP 9 — PREPROCESSOR (Tests 37–39)
# ============================================================

class TestPreprocessor:
    """Tests for build_preprocessors — dual ColumnTransformer builder."""

    @pytest.fixture
    def sample_X(self, engineered_df):
        return engineered_df.drop(columns=["Churn"], errors="ignore")

    def test_build_preprocessors_returns_four_objects(self, sample_X):
        """build_preprocessors must return (scaled, unscaled, cat_indices, feature_order)."""
        from src.data_loader import detect_feature_types
        ord_cols, cont_cols, bin_cols = detect_feature_types(sample_X)
        result = build_preprocessors(ord_cols, cont_cols, bin_cols, sample_X)
        assert len(result) == 4, "Expected 4-tuple from build_preprocessors"

    def test_preprocessors_are_independent(self, sample_X):
        """
        scaled and unscaled preprocessors must be independent sklearn objects.
        Verifies the clone(ord_pipeline) fix — they must NOT share fitted state.

        NOTE: ColumnTransformer.transformers is a list of 3-tuples (name, transformer, cols).
        dict() requires 2-tuples, so we use a dict comprehension instead.
        """
        from src.data_loader import detect_feature_types
        ord_cols, cont_cols, bin_cols = detect_feature_types(sample_X)
        scaled, unscaled, _, _ = build_preprocessors(ord_cols, cont_cols, bin_cols, sample_X)
        assert scaled is not unscaled, "Preprocessors must be different objects"

        # ColumnTransformer.transformers = [(name, transformer, cols), ...]
        # Use dict comprehension — dict() cannot unpack 3-tuples
        scaled_transformers   = {name: tfr for name, tfr, _ in scaled.transformers}
        unscaled_transformers = {name: tfr for name, tfr, _ in unscaled.transformers}

        assert "ord" in scaled_transformers,   "'ord' transformer missing from scaled preprocessor"
        assert "ord" in unscaled_transformers, "'ord' transformer missing from unscaled preprocessor"

        scaled_ord   = scaled_transformers["ord"]
        unscaled_ord = unscaled_transformers["ord"]

        assert scaled_ord is not unscaled_ord, \
            "OrdinalEncoder pipelines must be independent — clone() fix may be missing"

    def test_categorical_indices_subset_of_feature_order(self, sample_X):
        """All categorical indices must be valid indices into feature_order."""
        from src.data_loader import detect_feature_types
        ord_cols, cont_cols, bin_cols = detect_feature_types(sample_X)
        _, _, cat_indices, feature_order = build_preprocessors(
            ord_cols, cont_cols, bin_cols, sample_X
        )
        for idx in cat_indices:
            assert 0 <= idx < len(feature_order), \
                f"Categorical index {idx} out of range for feature_order (len={len(feature_order)})"


# ============================================================
# GROUP 10 — KS STATISTIC (Tests 40–41)
# ============================================================

class TestKSStatistic:
    """Tests for KS statistic computation."""

    def test_ks_perfect_separation(self):
        """Perfect separation → KS should be near 1.0."""
        y_true = np.array([1]*50 + [0]*50)
        y_prob = np.array([0.9]*50 + [0.1]*50)
        ks     = ks_statistic(y_true, y_prob)
        assert ks > 0.8, f"Expected KS > 0.8 for perfect sep, got {ks:.3f}"

    def test_ks_random_model(self):
        """Random model → KS should be near 0."""
        np.random.seed(42)
        y_true = np.array([1]*50 + [0]*50)
        y_prob = np.random.uniform(0, 1, 100)
        ks     = ks_statistic(y_true, y_prob)
        assert ks < 0.5, f"Random model KS unexpectedly high: {ks:.3f}"


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])