"""
routers/ai_chat.py
------------------
FastAPI router for all AI Copilot / chat endpoints.

This is the core of the LangChain integration. Each endpoint
triggers a different LangChain chain to generate AI responses.

Endpoints:
  POST /chat/ask             → General Q&A (RAG + LLM)
  POST /chat/explain         → Explain a specific prediction
  POST /chat/strategy        → Recommend marketing strategy
  POST /chat/generate-script → Generate call script
  POST /chat/generate-email  → Generate email
  POST /chat/quick-action    → Unified endpoint for all quick actions

The /chat/quick-action endpoint is what the React frontend chat UI calls.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal

from chains.prediction_explainer import explain_prediction
from chains.strategy_advisor import get_strategy_recommendation
from chains.script_generator import generate_content
from chains.rag_chain import answer_question
from models.predict_core import predict_single

router = APIRouter(prefix="/chat", tags=["AI Copilot"])


# ── Request / Response Schemas ────────────────────────────────────────────────
class ChatAskRequest(BaseModel):
    """For general Q&A questions (RAG-backed)."""
    question: str = Field(..., example="What is a term deposit?")
    context: Optional[str] = Field(
        default=None,
        description="Optional additional context to include in the query",
    )


class CustomerContext(BaseModel):
    """Embedded customer data for prediction-aware chat actions."""
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


class ChatWithCustomerRequest(BaseModel):
    """For actions that require customer context."""
    customer: CustomerContext
    question: Optional[str] = Field(
        default="Why did the model give this prediction?",
    )


class QuickActionRequest(BaseModel):
    """
    Unified request for all quick-action buttons in the chat UI.

    action values:
      - "explain"         → Explain the prediction probability
      - "strategy"        → Recommend marketing strategy
      - "call_script"     → Generate phone call script
      - "email"           → Generate marketing email
      - "general_ask"     → General banking Q&A (RAG)
    """
    action: Literal["explain", "strategy", "call_script", "email", "general_ask"]
    customer: Optional[CustomerContext] = None
    question: Optional[str] = Field(
        default=None,
        description="Custom question for 'explain' and 'general_ask' actions",
    )
    message: Optional[str] = Field(
        default=None,
        description="Free-form message from the employee (used in general_ask)",
    )


class ChatResponse(BaseModel):
    """Standardized response for all chat endpoints."""
    action: str
    response: str
    metadata: Optional[dict] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="General banking Q&A (RAG-powered)",
    description="""
Ask any banking-related question. The AI searches the banking document
knowledge base (RAG) before answering, ensuring grounded responses.

Example questions:
- "What are the benefits of a term deposit?"
- "How do I improve conversion rates for homeowners?"
- "What is the typical term deposit interest rate?"
    """,
)
async def ask_general_question(request: ChatAskRequest):
    """General Q&A using RAG over banking documents."""
    question = request.question
    if request.context:
        question = f"Context: {request.context}\n\nQuestion: {question}"

    answer = await answer_question(question)
    return ChatResponse(
        action="general_ask",
        response=answer,
        metadata={"rag_enabled": True},
    )


@router.post(
    "/explain",
    response_model=ChatResponse,
    summary="Explain a customer's prediction",
    description="""
Given customer data, runs the ML model and asks the LLM to explain
the result in plain English.

Powers the "Why is this prediction X%?" button in the UI.
    """,
)
async def explain_customer_prediction(request: ChatWithCustomerRequest):
    """Explains a specific customer's prediction."""
    try:
        customer_data = request.customer.model_dump()
        result = predict_single(customer_data, include_shap=True)

        prediction_dict = {
            "prediction": result.prediction,
            "probability": result.probability,
            "priority": result.priority,
            "strategy": result.strategy,
            "top_positive_factors": result.top_positive_factors,
            "top_negative_factors": result.top_negative_factors,
        }

        explanation = await explain_prediction(
            customer_data=customer_data,
            prediction_result=prediction_dict,
            question=request.question or "Why did the model give this prediction?",
        )

        return ChatResponse(
            action="explain",
            response=explanation,
            metadata={
                "probability": result.probability,
                "prediction": result.prediction,
                "priority": result.priority,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/strategy",
    response_model=ChatResponse,
    summary="Get marketing strategy recommendation",
    description="""
Generates ranked, actionable marketing recommendations for a specific
customer based on their profile and ML prediction score.

Powers the "What should the bank employee do?" button.
    """,
)
async def get_marketing_strategy(request: ChatWithCustomerRequest):
    """Generates marketing strategy recommendations."""
    try:
        customer_data = request.customer.model_dump()
        result = predict_single(customer_data, include_shap=False)

        prediction_dict = {
            "probability": result.probability,
            "priority": result.priority,
            "strategy": result.strategy,
        }

        strategy = await get_strategy_recommendation(
            customer_data=customer_data,
            prediction_result=prediction_dict,
        )

        return ChatResponse(
            action="strategy",
            response=strategy,
            metadata={
                "probability": result.probability,
                "priority": result.priority,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generate-script",
    response_model=ChatResponse,
    summary="Generate personalized call script",
    description="""
Generates a complete, personalized phone call script for the bank employee
to use when contacting this specific customer.

The script is tailored to the customer's profile and prediction score.
Powers the "Generate a call script" button.
    """,
)
async def generate_call_script(request: ChatWithCustomerRequest):
    """Generates a personalized call script."""
    try:
        customer_data = request.customer.model_dump()
        result = predict_single(customer_data, include_shap=False)
        prediction_dict = result.model_dump()

        script = await generate_content(
            customer_data=customer_data,
            prediction_result=prediction_dict,
            content_type="call_script",
        )

        return ChatResponse(
            action="call_script",
            response=script,
            metadata={"content_type": "call_script"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generate-email",
    response_model=ChatResponse,
    summary="Generate personalized marketing email",
    description="""
Generates a complete marketing email (subject + body) for the customer.
Tailored to their profile, balance tier, and subscription probability.

Powers the "Generate an email" button.
    """,
)
async def generate_marketing_email(request: ChatWithCustomerRequest):
    """Generates a personalized marketing email."""
    try:
        customer_data = request.customer.model_dump()
        result = predict_single(customer_data, include_shap=False)
        prediction_dict = result.model_dump()

        email = await generate_content(
            customer_data=customer_data,
            prediction_result=prediction_dict,
            content_type="email",
        )

        return ChatResponse(
            action="email",
            response=email,
            metadata={"content_type": "email"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/quick-action",
    response_model=ChatResponse,
    summary="Unified quick action endpoint (used by React chat UI)",
    description="""
Single endpoint that handles ALL quick-action buttons in the chat UI.
The React frontend calls this with different 'action' values.

Actions:
- explain      → Explain prediction in business language
- strategy     → Marketing strategy recommendations
- call_script  → Generate phone call script
- email        → Generate marketing email
- general_ask  → General banking Q&A (RAG)
    """,
)
async def handle_quick_action(request: QuickActionRequest):
    """Routes to the appropriate chain based on the action type."""
    try:
        if request.action == "general_ask":
            question = request.message or request.question or "Help me understand this prediction."
            answer = await answer_question(question)
            return ChatResponse(action="general_ask", response=answer)

        # All other actions require customer context
        if not request.customer:
            raise HTTPException(
                status_code=400,
                detail=f"Customer data is required for action '{request.action}'",
            )

        customer_data = request.customer.model_dump()

        if request.action == "explain":
            result = predict_single(customer_data, include_shap=True)
            prediction_dict = result.model_dump()
            response = await explain_prediction(
                customer_data=customer_data,
                prediction_result=prediction_dict,
                question=request.question or "Why did the model give this prediction?",
            )
            return ChatResponse(
                action="explain",
                response=response,
                metadata={"probability": result.probability, "prediction": result.prediction},
            )

        elif request.action == "strategy":
            result = predict_single(customer_data, include_shap=False)
            prediction_dict = result.model_dump()
            response = await get_strategy_recommendation(customer_data, prediction_dict)
            return ChatResponse(
                action="strategy",
                response=response,
                metadata={"priority": result.priority},
            )

        elif request.action in ("call_script", "email"):
            result = predict_single(customer_data, include_shap=False)
            prediction_dict = result.model_dump()
            response = await generate_content(customer_data, prediction_dict, content_type=request.action)
            return ChatResponse(
                action=request.action,
                response=response,
                metadata={"content_type": request.action},
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action '{request.action}' failed: {str(e)}")
