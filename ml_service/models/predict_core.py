"""
models/predict_core.py
----------------------
This is the UPGRADED wrapper around the ORIGINAL predict.py logic.
It loads the existing pipeline.pkl (trained Random Forest + SMOTE) and
exposes the prediction functions for use by FastAPI routers.

Original logic is PRESERVED. New features added:
  - SHAP value computation
  - Confidence interval simulation
  - Structured response model (Pydantic)
  - Feature importance extraction from pipeline
"""
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Module-level cached objects (loaded once on first call) ──────────────────
_pipeline = None
_features: list[str] = []
_feature_names_encoded: list[str] = []  # after OHE


# ── Pydantic Response Models ──────────────────────────────────────────────────
class PredictionResult(BaseModel):
    prediction: str            # "yes" | "no"
    probability: float         # 0–100
    priority: str              # "High" | "Medium" | "Low"
    strategy: str              # Human-readable strategy
    model: str                 # model name
    shap_values: Optional[dict] = None   # feature -> shap contribution
    top_positive_factors: Optional[list] = None
    top_negative_factors: Optional[list] = None
    confidence_band: Optional[dict] = None   # {"low": x, "high": y}


class BulkPredictionItem(BaseModel):
    row_index: int
    prediction: str
    probability: float
    priority: str
    strategy: str


# ── Loader ────────────────────────────────────────────────────────────────────
def _load_pipeline():
    """Lazy-loads the pipeline.pkl once and caches it."""
    global _pipeline, _features, _feature_names_encoded
    if _pipeline is not None:
        return

    pipeline_path = os.path.join(_MODELS_DIR, "pipeline.pkl")
    features_path = os.path.join(_MODELS_DIR, "features.json")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"pipeline.pkl not found at {pipeline_path}. "
            "Please copy your trained model file into ml_service/models/"
        )

    _pipeline = joblib.load(pipeline_path)

    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            _features = json.load(f)
    else:
        # Fallback: use the standard UCI Bank Marketing features
        _features = [
            "age", "job", "marital", "education", "default",
            "balance", "housing", "loan", "contact", "day",
            "month", "campaign", "pdays", "previous", "poutcome"
        ]

    # Try to extract encoded feature names from the fitted OHE inside the pipeline
    try:
        preprocessor = _pipeline.named_steps["preprocessor"]
        # Numerical columns (passthrough) come first
        num_transformer = preprocessor.transformers_[0]
        cat_transformer = preprocessor.transformers_[1]
        num_cols = list(num_transformer[2])
        ohe_cols = cat_transformer[1].get_feature_names_out(cat_transformer[2])
        _feature_names_encoded = num_cols + list(ohe_cols)
    except Exception:
        _feature_names_encoded = _features  # graceful fallback


# ── Strategy Generator (original logic preserved) ─────────────────────────────
def _get_strategy(prob: float, data: dict) -> str:
    """
    ORIGINAL logic from Bank/models/predict.py — PRESERVED.
    Converts probability into a dynamic banking strategy string.
    """
    if prob > 75:
        balance = data.get("balance", 0)
        if balance > 5000:
            return "High-Value Lead. Offer 'Premium Gold' Term Deposit with 0.5% bonus rate."
        return "Hot Lead. Immediate closing recommended via digital signature link."
    elif prob > 40:
        if data.get("housing") == "yes":
            return "Homeowner segment. Pitch 'Equity-Linked' savings for better conversion."
        return "Medium Potential. Schedule follow-up. Focus on tax-saving benefits."
    else:
        if data.get("previous", 0) > 2:
            return "High churn risk. Do not call. Move to automated email nurturing."
        return "Low potential. Low-cost automated marketing only."


# ── SHAP Computation ──────────────────────────────────────────────────────────
def _compute_shap(df_input: pd.DataFrame) -> dict:
    """
    Computes SHAP values for a single prediction.
    Returns a dict of {feature_name: shap_contribution}.
    Falls back gracefully if SHAP is unavailable.
    """
    try:
        import shap

        # Transform input through the preprocessing step only
        preprocessor = _pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(df_input)

        classifier = _pipeline.named_steps["classifier"]
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_transformed)

        # shap_vals is shape (n_samples, n_features, n_classes) for multi-output
        # or (n_samples, n_features) for binary with some versions
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]  # class=1 (subscription=yes)
        else:
            vals = shap_vals[0]

        features = _feature_names_encoded if _feature_names_encoded else [f"f{i}" for i in range(len(vals))]

        shap_dict = {
            feat: round(float(v), 4)
            for feat, v in zip(features, vals)
        }
        return shap_dict

    except Exception:
        # SHAP not available or error — return empty dict
        return {}


def _top_factors(shap_dict: dict, n: int = 3):
    """Returns top positive and negative SHAP contributors."""
    if not shap_dict:
        return [], []
    sorted_items = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    positives = [
        {"feature": k, "impact": v, "direction": "increases_probability"}
        for k, v in sorted_items if v > 0
    ][:n]
    negatives = [
        {"feature": k, "impact": v, "direction": "decreases_probability"}
        for k, v in sorted(shap_dict.items(), key=lambda x: x[1])
        if v < 0
    ][:n]
    return positives, negatives


# ── Core Prediction Functions ─────────────────────────────────────────────────
def predict_single(customer_data: dict, include_shap: bool = True) -> PredictionResult:
    """
    UPGRADED version of the original predict_subscription().
    Runs the pipeline, generates SHAP values, and returns a structured result.
    """
    _load_pipeline()

    # Build DataFrame — drop target/duration if accidentally included
    df = pd.DataFrame([customer_data])
    df = df.drop(columns=[c for c in ["y", "duration"] if c in df.columns])

    prob_val = float(_pipeline.predict_proba(df)[0][1])
    prob_pct = round(prob_val * 100, 2)

    priority = "High" if prob_val > 0.70 else ("Medium" if prob_val > 0.40 else "Low")
    strategy = _get_strategy(prob_pct, customer_data)

    # SHAP (optional, adds ~200ms)
    shap_dict, top_pos, top_neg = {}, [], []
    if include_shap:
        shap_dict = _compute_shap(df)
        top_pos, top_neg = _top_factors(shap_dict, n=3)

    # Pseudo confidence band (±5% uncertainty for display purposes)
    conf_band = {
        "low": round(max(0, prob_pct - 5), 1),
        "high": round(min(100, prob_pct + 5), 1),
    }

    return PredictionResult(
        prediction="yes" if prob_val >= 0.5 else "no",
        probability=prob_pct,
        priority=priority,
        strategy=strategy,
        model="Random Forest + SMOTE Pipeline (v1.0)",
        shap_values=shap_dict if shap_dict else None,
        top_positive_factors=top_pos if top_pos else None,
        top_negative_factors=top_neg if top_neg else None,
        confidence_band=conf_band,
    )


def predict_bulk(df: pd.DataFrame) -> list[BulkPredictionItem]:
    """
    UPGRADED version of the original predict_subscription_bulk().
    Processes a whole DataFrame at once and returns structured results.
    """
    _load_pipeline()

    safe_df = df.copy().drop(columns=[c for c in ["y", "duration"] if c in df.columns])
    probs = _pipeline.predict_proba(safe_df)[:, 1]

    results = []
    for i, p in enumerate(probs):
        row_data = df.iloc[i].to_dict()
        prob_pct = round(float(p) * 100, 2)
        results.append(
            BulkPredictionItem(
                row_index=i,
                prediction="yes" if p >= 0.5 else "no",
                probability=prob_pct,
                priority="High" if p > 0.70 else ("Medium" if p > 0.40 else "Low"),
                strategy=_get_strategy(prob_pct, row_data),
            )
        )
    return results


def get_feature_importance() -> dict:
    """
    Extracts feature importance from the trained Random Forest classifier.
    Returns top-15 features sorted by importance (descending).
    """
    _load_pipeline()
    try:
        classifier = _pipeline.named_steps["classifier"]
        importances = classifier.feature_importances_
        features = _feature_names_encoded if _feature_names_encoded else [f"f{i}" for i in range(len(importances))]

        importance_dict = {
            feat: round(float(imp), 4)
            for feat, imp in zip(features, importances)
        }
        # Sort descending, keep top 15
        return dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        )
    except Exception as e:
        return {"error": str(e)}


def get_model_info() -> dict:
    """Returns metadata about the loaded pipeline."""
    _load_pipeline()
    return {
        "model_name": "Random Forest + SMOTE Pipeline",
        "version": "1.0.0",
        "features": _features,
        "feature_count": len(_features),
        "pipeline_steps": list(_pipeline.named_steps.keys()),
        "status": "loaded",
    }
