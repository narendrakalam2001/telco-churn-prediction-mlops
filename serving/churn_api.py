# ============================================================
# CHURN API — FastAPI Serving
# ============================================================
# Same architecture as Credit Risk credit_risk_api.py.
#
# Churn-specific additions:
#   - /predict returns churn_probability + action + rule_triggered
#   - /model_info returns model card from churn_models/
#   - /health includes threshold info
#   - Logs: churn_probability, action columns
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import logging
import time
import os
import json

from src.model_loader            import load_latest_model
from services.prediction_service import predict_customer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "Telco Churn Prediction API",
    description = "Production-grade churn prediction with 3-tier intervention engine",
    version     = "1.0.0"
)

# ── Load model on startup ─────────────────────────────────────
try:
    model, threshold = load_latest_model()
    logger.info("Churn model loaded | threshold=%.4f", threshold)
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model     = None
    threshold = 0.5


# ============================================================
# INPUT SCHEMA
# ============================================================

class CustomerInput(BaseModel):
    # ── Core numeric ──────────────────────────────────────────
    tenure:         float = Field(..., ge=0,  description="Months with company")
    MonthlyCharges: float = Field(..., ge=0,  description="Monthly bill amount")
    TotalCharges:   float = Field(0.0, ge=0,  description="Total billed (0 for new customers)")

    # ── Demographics ──────────────────────────────────────────
    gender:        str = Field("Male", description="Male / Female")
    SeniorCitizen: int = Field(0,      description="1 = senior citizen")
    Partner:       str = Field("No",   description="Yes / No")
    Dependents:    str = Field("No",   description="Yes / No")

    # ── Contract + billing ────────────────────────────────────
    Contract:         str = Field("Month-to-month",   description="Month-to-month / One year / Two year")
    PaymentMethod:    str = Field("Electronic check", description="Payment method string")
    PaperlessBilling: str = Field("Yes",              description="Yes / No")

    # ── Internet services ─────────────────────────────────────
    InternetService:  str = Field("Fiber optic", description="DSL / Fiber optic / No")
    OnlineSecurity:   str = Field("No",          description="Yes / No / No internet service")
    OnlineBackup:     str = Field("No",          description="Yes / No / No internet service")
    DeviceProtection: str = Field("No",          description="Yes / No / No internet service")
    TechSupport:      str = Field("No",          description="Yes / No / No internet service")
    StreamingTV:      str = Field("No",          description="Yes / No / No internet service")
    StreamingMovies:  str = Field("No",          description="Yes / No / No internet service")

    # ── Phone services ────────────────────────────────────────
    PhoneService:  str = Field("Yes", description="Yes / No")
    MultipleLines: str = Field("No",  description="Yes / No / No phone service")


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Telco Churn Prediction API is live 🚀",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": model is not None,
        "threshold":    threshold,
    }


@app.get("/model_info")
def model_info():
    registry_path = os.path.join("churn_models", "latest_model.json")
    if not os.path.exists(registry_path):
        return {"error": "Model registry not found"}

    with open(registry_path) as f:
        registry = json.load(f)

    model_name = registry.get("model_name", "")
    parts      = model_name.replace("churn_model_", "").replace(".joblib", "")
    card_path  = os.path.join("churn_models", f"model_card_{parts}.json")

    if os.path.exists(card_path):
        with open(card_path) as f:
            return json.load(f)

    return registry


@app.post("/predict")
def predict(customer: CustomerInput):
    """
    Returns churn probability + 3-tier intervention action.

    Response:
        churn_probability : float 0-1
        churn_risk_band   : LOW | MEDIUM | HIGH
        action            : RETAIN | OUTREACH | URGENT
        rule_triggered    : str | null
        latency_seconds   : float
    """
    start      = time.time()
    input_data = customer.dict()

    if model is None:
        return {
            "error":             "Model not loaded — run train_model.py first",
            "churn_probability": None,
            "churn_risk_band":   None,
            "action":            "OUTREACH",
            "rule_triggered":    "MODEL_UNAVAILABLE",
        }

    result = predict_customer(model, input_data, threshold)

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":         time.time(),
        "tenure":            input_data["tenure"],
        "MonthlyCharges":    input_data["MonthlyCharges"],
        "Contract":          input_data["Contract"],
        "churn_probability": result["churn_probability"],
        "churn_risk_band":   result["churn_risk_band"],
        "action":            result["action"],
        "rule_triggered":    result.get("rule_triggered"),
    }

    log_path = os.path.join("logs", "prediction_logs.csv")
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)

    return result
