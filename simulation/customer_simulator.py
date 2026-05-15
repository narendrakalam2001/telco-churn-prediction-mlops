# ============================================================
# CUSTOMER SIMULATOR — Telco Churn ML System
# ============================================================
# Same architecture as Credit Risk applicant_simulator.py.
#
# Churn-specific scenarios vs Credit Risk:
#   Credit Risk scenarios: risky / safe / random
#   Churn scenarios:
#     high_churn   — new customer, month-to-month, high charges
#     low_churn    — long-tenure, annual contract, autopay
#     new_customer — tenure=0, no services, first month
#     senior       — senior citizen, fiber optic, no tech support
#     random       — mixed realistic profiles
#
# Output: churn_probability, churn_risk_band, action
# ============================================================

import requests
import random
import time
import os

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "CHURN_API_URL",
    "http://127.0.0.1:8000"
) + "/predict"

# ── Realistic value pools ─────────────────────────────────────
CONTRACTS        = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS  = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
]
INTERNET_TYPES   = ["Fiber optic", "DSL", "No"]
YES_NO           = ["Yes", "No"]
YES_NO_NOINT     = ["Yes", "No", "No internet service"]
YES_NO_NOPHONE   = ["Yes", "No", "No phone service"]


# ============================================================
# GENERATE SYNTHETIC CUSTOMER
# ============================================================

def generate_customer(scenario: str = "random") -> dict:
    """
    Generates a synthetic Telco customer profile.

    Scenarios:
        high_churn   — maximum churn risk profile
        low_churn    — minimum churn risk profile
        new_customer — brand new customer (tenure=0)
        senior       — senior citizen profile
        random       — mixed realistic profiles

    Returns:
        dict matching CustomerInput schema
    """

    if scenario == "high_churn":
        # ── High churn profile ────────────────────────────────
        # New customer + month-to-month + fiber + high charges
        # = classic churner pattern in telecom
        tenure         = random.randint(0, 6)
        monthly        = random.uniform(70, 110)
        return {
            "tenure":            tenure,
            "MonthlyCharges":    round(monthly, 2),
            "TotalCharges":      round(monthly * max(tenure, 1), 2),
            "gender":            random.choice(["Male", "Female"]),
            "SeniorCitizen":     random.choice([0, 1]),
            "Partner":           "No",
            "Dependents":        "No",
            "Contract":          "Month-to-month",
            "PaymentMethod":     "Electronic check",
            "PaperlessBilling":  "Yes",
            "InternetService":   "Fiber optic",
            "OnlineSecurity":    "No",
            "OnlineBackup":      "No",
            "DeviceProtection":  "No",
            "TechSupport":       "No",
            "StreamingTV":       random.choice(["Yes", "No"]),
            "StreamingMovies":   random.choice(["Yes", "No"]),
            "PhoneService":      "Yes",
            "MultipleLines":     "No",
        }

    elif scenario == "low_churn":
        # ── Low churn profile ─────────────────────────────────
        # Long tenure + 2-year contract + autopay + all services
        # = stickiest customer segment
        tenure  = random.randint(36, 72)
        monthly = random.uniform(55, 90)
        return {
            "tenure":            tenure,
            "MonthlyCharges":    round(monthly, 2),
            "TotalCharges":      round(monthly * tenure, 2),
            "gender":            random.choice(["Male", "Female"]),
            "SeniorCitizen":     0,
            "Partner":           "Yes",
            "Dependents":        random.choice(["Yes", "No"]),
            "Contract":          "Two year",
            "PaymentMethod":     random.choice([
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]),
            "PaperlessBilling":  random.choice(["Yes", "No"]),
            "InternetService":   "DSL",
            "OnlineSecurity":    "Yes",
            "OnlineBackup":      "Yes",
            "DeviceProtection":  "Yes",
            "TechSupport":       "Yes",
            "StreamingTV":       "Yes",
            "StreamingMovies":   "Yes",
            "PhoneService":      "Yes",
            "MultipleLines":     random.choice(["Yes", "No"]),
        }

    elif scenario == "new_customer":
        # ── New customer profile ──────────────────────────────
        # tenure=0 — just signed up, highest early churn window
        monthly = random.uniform(45, 95)
        return {
            "tenure":            0,
            "MonthlyCharges":    round(monthly, 2),
            "TotalCharges":      0.0,
            "gender":            random.choice(["Male", "Female"]),
            "SeniorCitizen":     0,
            "Partner":           random.choice(["Yes", "No"]),
            "Dependents":        "No",
            "Contract":          "Month-to-month",
            "PaymentMethod":     "Electronic check",
            "PaperlessBilling":  "Yes",
            "InternetService":   random.choice(["Fiber optic", "DSL"]),
            "OnlineSecurity":    "No",
            "OnlineBackup":      "No",
            "DeviceProtection":  "No",
            "TechSupport":       "No",
            "StreamingTV":       "No",
            "StreamingMovies":   "No",
            "PhoneService":      "Yes",
            "MultipleLines":     "No",
        }

    elif scenario == "senior":
        # ── Senior citizen profile ────────────────────────────
        # Higher churn risk due to service complexity + higher charges
        tenure  = random.randint(1, 48)
        monthly = random.uniform(65, 100)
        return {
            "tenure":            tenure,
            "MonthlyCharges":    round(monthly, 2),
            "TotalCharges":      round(monthly * tenure, 2),
            "gender":            random.choice(["Male", "Female"]),
            "SeniorCitizen":     1,
            "Partner":           random.choice(["Yes", "No"]),
            "Dependents":        "No",
            "Contract":          random.choice(["Month-to-month", "One year"]),
            "PaymentMethod":     random.choice(PAYMENT_METHODS),
            "PaperlessBilling":  random.choice(["Yes", "No"]),
            "InternetService":   "Fiber optic",
            "OnlineSecurity":    random.choice(["Yes", "No"]),
            "OnlineBackup":      random.choice(["Yes", "No"]),
            "DeviceProtection":  random.choice(["Yes", "No"]),
            "TechSupport":       random.choice(["Yes", "No"]),
            "StreamingTV":       random.choice(["Yes", "No"]),
            "StreamingMovies":   random.choice(["Yes", "No"]),
            "PhoneService":      "Yes",
            "MultipleLines":     random.choice(["Yes", "No"]),
        }

    else:
        # ── Random mixed profile ──────────────────────────────
        tenure  = random.randint(0, 72)
        monthly = random.uniform(18, 110)
        return {
            "tenure":            tenure,
            "MonthlyCharges":    round(monthly, 2),
            "TotalCharges":      round(monthly * max(tenure, 1), 2),
            "gender":            random.choice(["Male", "Female"]),
            "SeniorCitizen":     random.choice([0, 0, 0, 1]),  # ~25% senior
            "Partner":           random.choice(YES_NO),
            "Dependents":        random.choice(YES_NO),
            "Contract":          random.choice(CONTRACTS),
            "PaymentMethod":     random.choice(PAYMENT_METHODS),
            "PaperlessBilling":  random.choice(YES_NO),
            "InternetService":   random.choice(INTERNET_TYPES),
            "OnlineSecurity":    random.choice(YES_NO_NOINT),
            "OnlineBackup":      random.choice(YES_NO_NOINT),
            "DeviceProtection":  random.choice(YES_NO_NOINT),
            "TechSupport":       random.choice(YES_NO_NOINT),
            "StreamingTV":       random.choice(YES_NO_NOINT),
            "StreamingMovies":   random.choice(YES_NO_NOINT),
            "PhoneService":      random.choice(YES_NO),
            "MultipleLines":     random.choice(YES_NO_NOPHONE),
        }


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_customer(customer: dict, idx: int, scenario: str):
    """Sends one customer to API and prints result."""
    try:
        response = requests.post(API_URL, json=customer, timeout=15)

        if response.status_code == 200:
            result = response.json()
            action = result.get("action", "—")
            rule   = result.get("rule_triggered", "")
            rule_str = f"  rule={rule}" if rule else ""

            # ── Color-coded console output ─────────────────────
            action_icon = {
                "RETAIN":   "✅",
                "OUTREACH": "⚠️",
                "URGENT":   "🚨"
            }.get(action.split("_")[0] if "_" in action else action, "❓")

            print(
                f"[{idx+1:>2}] {action_icon}  "
                f"tenure={customer['tenure']:>2}mo  "
                f"charges=₹{customer['MonthlyCharges']:>6.1f}  "
                f"contract={customer['Contract']:<20}  "
                f"→  prob={result['churn_probability']:.4f}  "
                f"band={result['churn_risk_band']:<6}  "
                f"action={action}{rule_str}"
            )
        else:
            print(f"[{idx+1:>2}] API error: HTTP {response.status_code}")

    except requests.exceptions.Timeout:
        print(f"[{idx+1:>2}] Timeout — API cold start, retry in 30s")
    except Exception as e:
        print(f"[{idx+1:>2}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_customers(n: int = 20, scenario: str = "random"):
    """
    Simulates n customer predictions against the churn API.

    Args:
        n        : number of customers to simulate
        scenario : high_churn / low_churn / new_customer / senior / random
    """
    print(
        f"\n{'='*65}\n"
        f"  Simulating {n} customers  |  scenario={scenario}\n"
        f"{'='*65}"
    )

    action_counts = {"RETAIN": 0, "OUTREACH": 0, "URGENT": 0}

    for i in range(n):
        customer = generate_customer(scenario)
        send_customer(customer, i, scenario)

        # ── Rough tally (from printed output) ─────────────────
        time.sleep(0.3)

    print(f"{'='*65}")
    print(f"  Simulation complete — {n} customers processed")
    print(f"{'='*65}\n")


# ============================================================
# MULTI-SCENARIO REPORT
# ============================================================

def run_scenario_comparison(n_each: int = 10):
    """
    Runs all 4 scenarios and compares churn action distributions.
    Useful for validating model behavior across customer segments.
    """
    scenarios = ["high_churn", "low_churn", "new_customer", "senior"]

    print(
        f"\n{'='*65}\n"
        f"  SCENARIO COMPARISON — {n_each} customers per scenario\n"
        f"{'='*65}"
    )

    for scenario in scenarios:
        print(f"\n--- {scenario.upper()} ---")
        simulate_customers(n=n_each, scenario=scenario)
        time.sleep(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    simulate_customers(n=20, scenario="random")