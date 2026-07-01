"""
chains/prediction_explainer.py
-------------------------------
LangChain chain that explains WHY the ML model assigned a specific
probability to a customer.

Flow:
  customer_data + prediction_result
        ↓
  PromptTemplate (explainer.txt system prompt)
        ↓
  ChatGemini / ChatOpenAI
        ↓
  StrOutputParser → plain English explanation string
"""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.config import get_settings

settings = get_settings()

# ── Load system prompt from file ──────────────────────────────────────────────
_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts", "explainer.txt"
)

with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
    EXPLAINER_SYSTEM_PROMPT = f.read()


# ── LLM Factory ───────────────────────────────────────────────────────────────
def _get_llm():
    """Returns the configured LLM (Gemini or OpenAI) based on settings."""
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True,
        )


# ── Prompt Template ───────────────────────────────────────────────────────────
_EXPLAINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXPLAINER_SYSTEM_PROMPT),
    ("human", """
Please explain this prediction to the bank employee:

CUSTOMER PROFILE:
- Age: {age}
- Job: {job}
- Marital Status: {marital}
- Education: {education}
- Account Balance: €{balance}
- Housing Loan: {housing}
- Personal Loan: {loan}
- Previous Campaigns Contacted: {campaign}
- Days Since Last Contact: {pdays}
- Previous Outcome: {poutcome}

PREDICTION RESULT:
- Subscription Probability: {probability}%
- Decision: Customer will {prediction} subscribe
- Priority Tier: {priority}
- Top Positive Factors: {top_positive_factors}
- Top Negative Factors: {top_negative_factors}

Question from employee: {question}

Please provide a clear, business-language explanation.
"""),
])


# ── Chain Builder ─────────────────────────────────────────────────────────────
def build_explainer_chain():
    """
    Returns a LangChain LCEL (LangChain Expression Language) chain.

    Input dict keys:
      - age, job, marital, education, balance, housing, loan,
        campaign, pdays, poutcome  (customer fields)
      - probability, prediction, priority (ML result fields)
      - top_positive_factors, top_negative_factors (SHAP summary)
      - question (employee's natural language question)

    Returns: str (the explanation)
    """
    llm = _get_llm()
    chain = _EXPLAINER_PROMPT | llm | StrOutputParser()
    return chain


# ── Convenience Function ──────────────────────────────────────────────────────
async def explain_prediction(
    customer_data: dict,
    prediction_result: dict,
    question: str = "Why did the model give this prediction?",
) -> str:
    """
    High-level async function called by the /explain router.

    Args:
        customer_data: Raw customer feature dict
        prediction_result: Output from predict_core.predict_single()
        question: Optional employee question

    Returns:
        Plain-English explanation string from the LLM
    """
    chain = build_explainer_chain()

    # Merge customer data with prediction result into a flat dict for the prompt
    input_dict = {
        # Customer features (with defaults for missing fields)
        "age": customer_data.get("age", "N/A"),
        "job": customer_data.get("job", "N/A"),
        "marital": customer_data.get("marital", "N/A"),
        "education": customer_data.get("education", "N/A"),
        "balance": customer_data.get("balance", "N/A"),
        "housing": customer_data.get("housing", "N/A"),
        "loan": customer_data.get("loan", "N/A"),
        "campaign": customer_data.get("campaign", "N/A"),
        "pdays": customer_data.get("pdays", "N/A"),
        "poutcome": customer_data.get("poutcome", "N/A"),
        # Prediction result
        "probability": prediction_result.get("probability", "N/A"),
        "prediction": prediction_result.get("prediction", "N/A"),
        "priority": prediction_result.get("priority", "N/A"),
        "top_positive_factors": str(prediction_result.get("top_positive_factors", "Not computed")),
        "top_negative_factors": str(prediction_result.get("top_negative_factors", "Not computed")),
        # Employee question
        "question": question,
    }

    try:
        return await chain.ainvoke(input_dict)
    except Exception as e:
        return f"[AI Explanation unavailable: {str(e)}. Please check your API key configuration.]"
