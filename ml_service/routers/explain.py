"""
routers/explain.py
------------------
FastAPI router for SHAP-based model explainability endpoints.

Endpoints:
  POST /explain/shap      → SHAP values for a single prediction
  POST /explain/summary   → Human-readable explanation via LangChain
  GET  /explain/global    → Global feature importance (model-level)

These endpoints power the "Why did the model decide this?" UI panel
in the React frontend.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from models.predict_core import predict_single, get_feature_importance
from chains.prediction_explainer import explain_prediction

router = APIRouter(prefix="/explain", tags=["Explainability"])


# ── Request Schemas ───────────────────────────────────────────────────────────
class ExplainRequest(BaseModel):
    """
    Full customer + prediction data needed for explanation.
    Accepts raw customer data (model runs internally if no prediction provided).
    """
    # Customer features
    age: int = Field(..., example=42)
    job: str = Field(..., example="management")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="tertiary")
    default: str = Field(..., example="no")
    balance: int = Field(..., example=2500)
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    day: int = Field(..., example=15)
    month: str = Field(..., example="may")
    campaign: int = Field(..., example=2)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., example=0)
    poutcome: str = Field(..., example="unknown")

    # Optional employee question
    question: str = Field(
        default="Why did the model give this prediction?",
        example="Why is this customer predicted to subscribe?",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/shap",
    summary="Get SHAP values for a customer prediction",
    description="""
Runs the prediction pipeline AND computes SHAP values.
Returns both the numeric prediction result and feature-level SHAP contributions.

SHAP (SHapley Additive exPlanations) shows HOW MUCH each feature
contributed to pushing the prediction up or down.
    """,
)
async def get_shap_values(request: ExplainRequest):
    """Returns full prediction result including SHAP values."""
    try:
        customer_data = request.model_dump(exclude={"question"})
        result = predict_single(customer_data, include_shap=True)
        return {
            "prediction": result.prediction,
            "probability": result.probability,
            "priority": result.priority,
            "shap_values": result.shap_values,
            "top_positive_factors": result.top_positive_factors,
            "top_negative_factors": result.top_negative_factors,
            "confidence_band": result.confidence_band,
            "model": result.model,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {str(e)}")


@router.post(
    "/summary",
    summary="Get AI-generated plain English explanation",
    description="""
Runs the prediction, computes SHAP values, then uses LangChain + Gemini/OpenAI
to generate a plain-English business explanation for the bank employee.

This powers the "Why?" button in the AI Copilot chat interface.

Requires GOOGLE_API_KEY or OPENAI_API_KEY to be configured.
    """,
)
async def get_ai_explanation(request: ExplainRequest):
    """Returns LangChain-generated plain English explanation."""
    try:
        customer_data = request.model_dump(exclude={"question"})

        # Run prediction with SHAP
        result = predict_single(customer_data, include_shap=True)
        prediction_dict = {
            "prediction": result.prediction,
            "probability": result.probability,
            "priority": result.priority,
            "strategy": result.strategy,
            "top_positive_factors": result.top_positive_factors,
            "top_negative_factors": result.top_negative_factors,
        }

        # Get LangChain explanation
        explanation = await explain_prediction(
            customer_data=customer_data,
            prediction_result=prediction_dict,
            question=request.question,
        )

        return {
            "prediction_result": prediction_dict,
            "ai_explanation": explanation,
            "question_asked": request.question,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get(
    "/global",
    summary="Global model feature importance",
    description="""
Returns the top 15 most important features across ALL predictions
(not customer-specific). This is the model-level feature importance
from the Random Forest's Gini impurity calculation.
    """,
)
async def global_feature_importance():
    """Returns global feature importance from the trained model."""
    try:
        importance = get_feature_importance()

        if "error" in importance:
            raise HTTPException(status_code=500, detail=importance["error"])

        # Format for charting in the frontend
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return {
            "feature_importance": importance,
            "chart_data": {
                "labels": [f[0] for f in sorted_features],
                "values": [f[1] for f in sorted_features],
            },
            "top_feature": sorted_features[0][0] if sorted_features else None,
            "model": "Random Forest + SMOTE Pipeline",
            "interpretation": "Higher values = more influence on subscription prediction",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
