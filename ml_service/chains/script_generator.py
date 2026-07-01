"""
chains/script_generator.py
---------------------------
LangChain chain that generates personalized call scripts and email templates
for bank employees based on the customer profile and prediction data.

Supported content types:
  - "call_script"  → Phone call opening script with objection handling
  - "email"        → Personalized marketing email with subject line

Flow:
  customer_data + prediction_result + content_type
        ↓
  PromptTemplate (script.txt system prompt)
        ↓
  ChatGemini / ChatOpenAI (higher temperature for creativity)
        ↓
  StrOutputParser → script/email string
"""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_settings

settings = get_settings()

# ── Load system prompt ────────────────────────────────────────────────────────
_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts", "script.txt"
)

with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
    SCRIPT_SYSTEM_PROMPT = f.read()


# ── LLM Factory ───────────────────────────────────────────────────────────────
def _get_llm():
    """Higher temperature for more creative marketing copy."""
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.7,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True,
        )


# ── Prompt Templates ──────────────────────────────────────────────────────────
_CALL_SCRIPT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SCRIPT_SYSTEM_PROMPT),
    ("human", """
Generate a personalized PHONE CALL SCRIPT for a bank employee to use.

CUSTOMER PROFILE:
- Age: {age}
- Job/Occupation: {job}
- Marital Status: {marital}
- Education: {education}
- Account Balance: €{balance}
- Has Housing Loan: {housing}
- Has Personal Loan: {loan}
- Previous Campaign Outcome: {poutcome}

AI PREDICTION:
- Subscription Probability: {probability}%
- Priority: {priority}
- Recommended Offer: {strategy}

Please generate a complete, natural-sounding phone script that a bank employee
can use verbatim. Include the compliance opening and a clear close.
Remember to tailor the tone to the customer's profile (age={age}, job={job}).
"""),
])

_EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SCRIPT_SYSTEM_PROMPT),
    ("human", """
Generate a personalized MARKETING EMAIL for the following customer.

CUSTOMER PROFILE:
- Age: {age}
- Job/Occupation: {job}
- Marital Status: {marital}
- Education: {education}
- Account Balance: €{balance}
- Has Housing Loan: {housing}
- Has Personal Loan: {loan}
- Previous Campaign Outcome: {poutcome}

AI PREDICTION:
- Subscription Probability: {probability}%
- Priority: {priority}
- Recommended Offer: {strategy}

Please generate a complete email with:
1. A personalized subject line
2. A warm, relevant opening
3. Clear value proposition (2-3 sentences)
4. A strong CTA
5. Professional sign-off

Format it clearly with "SUBJECT:" on the first line, then "BODY:" for the email body.
"""),
])


# ── Chain Builders ────────────────────────────────────────────────────────────
def build_call_script_chain():
    """Returns the call script generation chain."""
    return _CALL_SCRIPT_PROMPT | _get_llm() | StrOutputParser()


def build_email_chain():
    """Returns the email generation chain."""
    return _EMAIL_PROMPT | _get_llm() | StrOutputParser()


# ── Convenience Function ──────────────────────────────────────────────────────
async def generate_content(
    customer_data: dict,
    prediction_result: dict,
    content_type: str = "call_script",
) -> str:
    """
    High-level async function called by the /chat/generate-script router.

    Args:
        customer_data: Customer feature dict
        prediction_result: Dict from predict_core
        content_type: "call_script" | "email"

    Returns:
        Generated script or email as a string
    """
    input_dict = {
        "age": customer_data.get("age", "N/A"),
        "job": customer_data.get("job", "N/A"),
        "marital": customer_data.get("marital", "N/A"),
        "education": customer_data.get("education", "N/A"),
        "balance": customer_data.get("balance", 0),
        "housing": customer_data.get("housing", "N/A"),
        "loan": customer_data.get("loan", "N/A"),
        "poutcome": customer_data.get("poutcome", "unknown"),
        "probability": prediction_result.get("probability", "N/A"),
        "priority": prediction_result.get("priority", "N/A"),
        "strategy": prediction_result.get("strategy", "Term Deposit subscription"),
    }

    try:
        if content_type == "email":
            chain = build_email_chain()
        else:
            chain = build_call_script_chain()
        return await chain.ainvoke(input_dict)
    except Exception as e:
        return f"[Content generation unavailable: {str(e)}. Please check your API key configuration.]"
