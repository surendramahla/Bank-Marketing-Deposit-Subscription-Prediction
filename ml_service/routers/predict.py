"""
routers/predict.py
------------------
FastAPI router exposing the ML prediction endpoints.

Endpoints:
  POST /predict/single   → Single customer prediction with SHAP
  POST /predict/bulk     → Bulk prediction from uploaded CSV
  GET  /predict/features → Returns the list of model input features
  GET  /predict/info     → Returns model metadata / status
  GET  /predict/feature-importance → Returns top feature importances

All endpoints are protected with API key or open for internal service use.
For production, add authentication middleware.
"""
import io
import os
import json
import pandas as pd
from typing import Optional, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models.predict_core import (
    predict_single,
    predict_bulk,
    get_feature_importance,
    get_model_info,
    PredictionResult,
    BulkPredictionItem,
)
from core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/predict", tags=["Predictions"])


# ── Request / Response Schemas ────────────────────────────────────────────────
class SinglePredictRequest(BaseModel):
    """
    Input schema for a single customer prediction.
    Mirrors the UCI Bank Marketing dataset fields.
    """
    age: int = Field(..., ge=18, le=100, example=42)
    job: str = Field(..., example="management")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="tertiary")
    default: str = Field(..., example="no")
    balance: int = Field(..., example=2500)
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    day: int = Field(..., ge=1, le=31, example=15)
    month: str = Field(..., example="may")
    campaign: int = Field(..., ge=1, example=2)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., ge=0, example=0)
    poutcome: str = Field(..., example="unknown")
    include_shap: bool = Field(default=True, description="Whether to compute SHAP values (adds ~200ms)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 42,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 2500,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
                "include_shap": True,
            }
        }


class BulkPredictResponse(BaseModel):
    total_records: int
    predictions: list[BulkPredictionItem]
    summary: dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/single",
    response_model=PredictionResult,
    summary="Predict subscription for a single customer",
    description="""
Run the Random Forest + SMOTE pipeline on a single customer's data.

Returns:
- **probability**: Likelihood of term deposit subscription (0–100%)
- **prediction**: "yes" or "no"
- **priority**: "High" / "Medium" / "Low"
- **strategy**: Recommended marketing action
- **shap_values**: Feature contributions (if include_shap=True)
- **top_positive_factors**: Features pushing prediction UP
- **top_negative_factors**: Features pushing prediction DOWN
- **confidence_band**: Uncertainty range (±5%)
    """,
)
async def predict_single_customer(request: SinglePredictRequest):
    """Single customer prediction endpoint."""
    try:
        # Convert Pydantic model to dict (excluding include_shap field)
        customer_data = request.model_dump(exclude={"include_shap"})
        result = predict_single(customer_data, include_shap=request.include_shap)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/bulk",
    response_model=BulkPredictResponse,
    summary="Bulk prediction from CSV upload",
    description="""
Upload a CSV file containing multiple customers' data.
Returns predictions for all rows with download option.

CSV must have columns matching the features list (from GET /predict/features).
Use GET /predict/bulk/template to download a sample CSV.
    """,
)
async def predict_bulk_customers(
    file: UploadFile = File(..., description="CSV file with customer data"),
):
    """Bulk prediction from uploaded CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported. Please upload a .csv file.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    # Validate minimum required columns
    features_path = os.path.join(settings.MODEL_DIR, "features.json")
    if os.path.exists(features_path):
        with open(features_path) as f:
            required_features = json.load(f)
        missing = [col for col in required_features if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"CSV is missing columns: {missing}. "
                       f"Download the template at GET /predict/bulk/template",
            )

    try:
        results = predict_bulk(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk prediction failed: {str(e)}")

    # Compute summary statistics
    total = len(results)
    yes_count = sum(1 for r in results if r.prediction == "yes")
    high_count = sum(1 for r in results if r.priority == "High")
    med_count = sum(1 for r in results if r.priority == "Medium")
    low_count = sum(1 for r in results if r.priority == "Low")
    avg_prob = round(sum(r.probability for r in results) / total, 2) if total > 0 else 0

    summary = {
        "total_records": total,
        "predicted_yes": yes_count,
        "predicted_no": total - yes_count,
        "conversion_rate_pct": round(yes_count / total * 100, 1) if total > 0 else 0,
        "avg_probability_pct": avg_prob,
        "high_priority_leads": high_count,
        "medium_priority_leads": med_count,
        "low_priority_leads": low_count,
    }

    return BulkPredictResponse(
        total_records=total,
        predictions=results,
        summary=summary,
    )


@router.get(
    "/bulk/template",
    summary="Download CSV template for bulk prediction",
    description="Returns a sample CSV with the correct column headers and one example row.",
)
async def download_bulk_template():
    """Returns a downloadable CSV template."""
    features = [
        "age", "job", "marital", "education", "default",
        "balance", "housing", "loan", "contact", "day",
        "month", "campaign", "pdays", "previous", "poutcome",
    ]
    sample_row = ["35", "management", "married", "tertiary", "no",
                  "1500", "yes", "no", "cellular", "15", "may",
                  "1", "-1", "0", "unknown"]

    output = io.StringIO()
    output.write(",".join(features) + "\n")
    output.write(",".join(sample_row) + "\n")
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bulk_upload_template.csv"},
    )


@router.get(
    "/features",
    summary="Get model input features",
    description="Returns the list of feature names the model expects as input.",
)
async def get_features():
    """Returns model input features."""
    features_path = os.path.join(settings.MODEL_DIR, "features.json")
    if os.path.exists(features_path):
        with open(features_path) as f:
            features = json.load(f)
    else:
        features = [
            "age", "job", "marital", "education", "default",
            "balance", "housing", "loan", "contact", "day",
            "month", "campaign", "pdays", "previous", "poutcome",
        ]

    return {
        "features": features,
        "count": len(features),
        "description": "UCI Bank Marketing dataset features",
    }


@router.get(
    "/feature-importance",
    summary="Get model feature importance",
    description="Returns the top-15 most important features from the Random Forest pipeline.",
)
async def feature_importance():
    """Returns feature importance from the trained pipeline."""
    try:
        importance = get_feature_importance()
        return {
            "feature_importance": importance,
            "model": "Random Forest + SMOTE Pipeline",
            "note": "Higher values indicate more influence on predictions",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/info",
    summary="Model metadata and status",
    description="Returns information about the loaded model pipeline.",
)
async def model_info():
    """Returns model metadata."""
    try:
        return get_model_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
