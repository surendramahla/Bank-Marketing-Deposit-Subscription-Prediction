"""
chains/strategy_advisor.py
---------------------------
LangChain chain that generates ranked, actionable marketing strategy
recommendations for a specific customer based on their profile and
prediction probability.

Flow:
  customer_data + prediction_result
        ↓
  PromptTemplate (strategy.txt system prompt)
        ↓
  ChatGemini / ChatOpenAI
        ↓
  StrOutputParser → structured strategy markdown string
"""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_settings

settings = get_settings()

# ── Load system prompt ────────────────────────────────────────────────────────
_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts", "strategy.txt"
)

with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
    STRATEGY_SYSTEM_PROMPT = f.read()


# ── LLM Factory ───────────────────────────────────────────────────────────────
def _get_llm():
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.4,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True,
        )


# ── Prompt Template ───────────────────────────────────────────────────────────
_STRATEGY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STRATEGY_SYSTEM_PROMPT),
    ("human", """
Develop a marketing strategy for the following customer:

CUSTOMER PROFILE:
- Age: {age} years old
- Occupation: {job}
- Marital Status: {marital}
- Education Level: {education}
- Account Balance: €{balance}
- Has Housing Loan: {housing}
- Has Personal Loan: {loan}
- Contact Method Used: {contact}
- Month of Last Contact: {month}
- Number of Campaign Contacts: {campaign}
- Days Since Previous Campaign: {pdays}
- Previous Campaign Outcome: {poutcome}
- Times Contacted Previously: {previous}

AI PREDICTION RESULTS:
- Subscription Probability: {probability}%
- Priority Tier: {priority}
- Current ML Strategy: {ml_strategy}

Based on this data, provide your top marketing strategy recommendations.
"""),
])


# ── Chain Builder ─────────────────────────────────────────────────────────────
def build_strategy_chain():
    """
    Returns the strategy advisor LCEL chain.

    Input dict keys: all customer fields + probability, priority, ml_strategy
    Returns: str (structured markdown with ranked recommendations)
    """
    llm = _get_llm()
    return _STRATEGY_PROMPT | llm | StrOutputParser()


# ── Convenience Function ──────────────────────────────────────────────────────
async def get_strategy_recommendation(
    customer_data: dict,
    prediction_result: dict,
) -> str:
    """
    High-level async function called by the /chat/strategy router.

    Args:
        customer_data: Customer feature dict
        prediction_result: Dict from predict_core.predict_single()

    Returns:
        Markdown-formatted strategy recommendation string
    """
    chain = build_strategy_chain()

    input_dict = {
        "age": customer_data.get("age", "N/A"),
        "job": customer_data.get("job", "N/A"),
        "marital": customer_data.get("marital", "N/A"),
        "education": customer_data.get("education", "N/A"),
        "balance": customer_data.get("balance", 0),
        "housing": customer_data.get("housing", "N/A"),
        "loan": customer_data.get("loan", "N/A"),
        "contact": customer_data.get("contact", "N/A"),
        "month": customer_data.get("month", "N/A"),
        "campaign": customer_data.get("campaign", "N/A"),
        "pdays": customer_data.get("pdays", -1),
        "poutcome": customer_data.get("poutcome", "unknown"),
        "previous": customer_data.get("previous", 0),
        # Prediction fields
        "probability": prediction_result.get("probability", "N/A"),
        "priority": prediction_result.get("priority", "N/A"),
        "ml_strategy": prediction_result.get("strategy", "N/A"),
    }

    try:
        return await chain.ainvoke(input_dict)
    except Exception as e:
        return f"[Strategy unavailable: {str(e)}. Please check your API key configuration.]"
