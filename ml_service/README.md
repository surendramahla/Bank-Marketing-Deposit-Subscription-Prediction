# BankAI Pro — Phase 1: Python FastAPI ML Service

> **Status**: ✅ Phase 1 Complete  
> **Port**: `8000`  
> **Docs**: http://localhost:8000/docs

---

## What This Is

This is the **Python ML Service** for the BankAI Pro portfolio project. It wraps the original Bank Marketing ML model (Random Forest + SMOTE pipeline trained on the UCI Bank Marketing dataset) and extends it with:

- ⚡ **FastAPI REST endpoints** — Production-grade, auto-documented API
- 🔍 **SHAP Explainability** — Understand WHY the model made each prediction
- 🤖 **LangChain AI Copilot** — LLM-powered explanations, strategies, and scripts
- 📚 **RAG Q&A** — Answers banking questions from internal documents
- 🧪 **Test Suite** — pytest-based tests for all endpoints

---

## Folder Structure

```
ml_service/
├── main.py                    ← FastAPI app entry point
├── requirements.txt           ← Python dependencies
├── .env.example               ← Environment variable template
│
├── core/
│   └── config.py              ← Pydantic settings (reads from .env)
│
├── models/
│   ├── pipeline.pkl           ← Original trained ML model (copied)
│   ├── features.json          ← Feature list for the model
│   └── predict_core.py        ← Extended prediction logic + SHAP
│
├── routers/
│   ├── predict.py             ← POST /predict/single, /predict/bulk
│   ├── explain.py             ← POST /explain/shap, /explain/summary
│   └── ai_chat.py             ← POST /chat/*, /chat/quick-action
│
├── chains/
│   ├── prediction_explainer.py ← LangChain: Explain predictions
│   ├── strategy_advisor.py     ← LangChain: Marketing strategy
│   ├── script_generator.py     ← LangChain: Call scripts & emails
│   └── rag_chain.py           ← LangChain: RAG over banking docs
│
├── prompts/
│   ├── explainer.txt          ← System prompt for explanation chain
│   ├── strategy.txt           ← System prompt for strategy chain
│   └── script.txt             ← System prompt for script/email chain
│
├── rag/
│   └── documents/
│       ├── banking_faq.txt    ← Banking FAQ for RAG
│       └── marketing_guidelines.txt ← Marketing strategy guide
│
└── tests/
    └── test_predict.py        ← pytest test suite
```

---

## Quick Start

### 1. Setup Environment

```bash
# Navigate to ml_service
cd bank2/ml_service

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy template
cp .env.example .env

# Edit .env and add your Gemini API key
# Get free key at: https://aistudio.google.com
GOOGLE_API_KEY=your_key_here
```

### 4. Start the Server

```bash
uvicorn main:app --reload --port 8000
```

### 5. Open API Docs

Visit http://localhost:8000/docs for interactive Swagger UI

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/predict/single` | Single customer prediction |
| `POST` | `/predict/bulk` | Bulk CSV prediction |
| `GET` | `/predict/bulk/template` | Download CSV template |
| `GET` | `/predict/features` | Get feature list |
| `GET` | `/predict/feature-importance` | Global feature importance |
| `GET` | `/predict/info` | Model metadata |
| `POST` | `/explain/shap` | SHAP values for prediction |
| `POST` | `/explain/summary` | AI explanation (LangChain) |
| `GET` | `/explain/global` | Global feature importance chart |
| `POST` | `/chat/ask` | General Q&A (RAG) |
| `POST` | `/chat/explain` | Explain a prediction |
| `POST` | `/chat/strategy` | Marketing strategy |
| `POST` | `/chat/generate-script` | Phone call script |
| `POST` | `/chat/generate-email` | Marketing email |
| `POST` | `/chat/quick-action` | **Unified chat endpoint** (used by UI) |

---

## Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
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
    "include_shap": true
  }'
```

**Response:**
```json
{
  "prediction": "yes",
  "probability": 72.5,
  "priority": "High",
  "strategy": "Hot Lead. Immediate closing recommended via digital signature link.",
  "model": "Random Forest + SMOTE Pipeline (v1.0)",
  "shap_values": {"age": 0.023, "balance": 0.087, ...},
  "top_positive_factors": [{"feature": "balance", "impact": 0.087, "direction": "increases_probability"}],
  "top_negative_factors": [...],
  "confidence_band": {"low": 67.5, "high": 77.5}
}
```

---

## Example: AI Copilot (LangChain)

```bash
curl -X POST http://localhost:8000/chat/quick-action \
  -H "Content-Type: application/json" \
  -d '{
    "action": "explain",
    "customer": {
      "age": 42, "job": "management", "marital": "married",
      "education": "tertiary", "default": "no", "balance": 2500,
      "housing": "yes", "loan": "no", "contact": "cellular",
      "day": 15, "month": "may", "campaign": 2,
      "pdays": -1, "previous": 0, "poutcome": "unknown"
    },
    "question": "Why is this prediction 72%?"
  }'
```

**Response:**
```json
{
  "action": "explain",
  "response": "This customer shows strong subscription potential primarily due to their substantial account balance of €2,500 and professional management role...",
  "metadata": {"probability": 72.5, "prediction": "yes"}
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## LLM Configuration

| Setting | Default | Options |
|---------|---------|---------|
| `LLM_PROVIDER` | `gemini` | `gemini`, `openai` |
| `GEMINI_MODEL` | `gemini-1.5-flash` | `gemini-1.5-pro`, etc. |
| `OPENAI_MODEL` | `gpt-4o` | `gpt-4o-mini`, etc. |

**Gemini is the default** because it has a generous free tier (no credit card required). Switch to OpenAI by setting `LLM_PROVIDER=openai` in your `.env`.
