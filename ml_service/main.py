"""
main.py
-------
FastAPI Application Entry Point — BankAI Pro ML Service

This service:
1. Exposes the original Bank Marketing ML model (pipeline.pkl) via REST API
2. Provides SHAP-based explainability
3. Integrates LangChain + Gemini/OpenAI for AI copilot features
4. Includes RAG pipeline for banking document Q&A

Architecture:
  React Frontend
      ↓ REST
  Node.js Backend (API Gateway)
      ↓ Internal HTTP
  THIS SERVICE (Python FastAPI) ← You are here
      ↓
  pipeline.pkl (Random Forest + SMOTE)
      ↓
  LangChain + Gemini/OpenAI + ChromaDB

Run locally:
  uvicorn main:app --reload --port 8000

API Docs (auto-generated):
  http://localhost:8000/docs     ← Swagger UI
  http://localhost:8000/redoc   ← ReDoc
"""
import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from routers import predict, explain, ai_chat

# ── Configuration ─────────────────────────────────────────────────────────────
settings = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bankai.ml_service")


# ── Lifespan (Startup / Shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    - Preloads the ML pipeline into memory (fast first request)
    - Initializes the RAG vector store
    - Creates upload directory
    """
    # ── STARTUP ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  BankAI Pro ML Service - Starting Up")
    logger.info("=" * 60)

    # 1. Create required directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")

    # 2. Preload ML Pipeline (avoid cold-start delay on first request)
    try:
        from models.predict_core import _load_pipeline, get_model_info
        _load_pipeline()
        info = get_model_info()
        logger.info(f"✅ ML Pipeline loaded: {info['model_name']}")
        logger.info(f"   Features: {info['feature_count']} | Steps: {info['pipeline_steps']}")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  ML Pipeline not found: {e}")
        logger.warning("   Prediction endpoints will return 503 until pipeline.pkl is available.")
    except Exception as e:
        logger.error(f"❌ ML Pipeline load error: {e}")

    # 3. Initialize RAG Vector Store (async-compatible)
    if settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY:
        try:
            from chains.rag_chain import initialize_vector_store
            initialize_vector_store()
            logger.info("✅ RAG vector store initialized")
        except Exception as e:
            logger.warning(f"⚠️  RAG initialization failed: {e} (RAG features will be disabled)")
    else:
        logger.warning("⚠️  No LLM API key configured. AI chat features will be unavailable.")
        logger.warning("   Set GOOGLE_API_KEY or OPENAI_API_KEY in .env to enable AI features.")

    logger.info(f"✅ Server ready — LLM Provider: {settings.LLM_PROVIDER.upper()}")
    logger.info("=" * 60)

    yield  # App runs here

    # ── SHUTDOWN ───────────────────────────────────────────────────
    logger.info("ML Service shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="BankAI Pro - ML Service",
    description="""
## Bank Marketing Deposit Prediction API

This is the AI/ML backend for **BankAI Pro** — a production-ready AI-powered
bank marketing intelligence platform.

### What This Service Provides

- 🤖 **ML Predictions** — Random Forest + SMOTE pipeline for subscription probability
- 🔍 **SHAP Explainability** — Feature-level explanation of every prediction
- 💬 **AI Copilot** — LangChain + Gemini/GPT powered chat assistant
- 📚 **RAG Q&A** — Banking document knowledge retrieval
- 📊 **Analytics** — Feature importance, model performance metrics

### Original Model
This service wraps the original [Bank Marketing Prediction model](https://github.com/surendramahla/Bank-Marketing-Deposit-Subscription-Prediction)
trained on the UCI Bank Marketing dataset (45,211 records, 16 features).

### Quick Start
1. Set your API keys in `.env`
2. Run: `uvicorn main:app --reload --port 8000`
3. Open: http://localhost:8000/docs
    """,
    version="1.0.0",
    contact={
        "name": "BankAI Pro",
        "url": "https://github.com/surendramahla/Bank-Marketing-Deposit-Subscription-Prediction",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware ─────────────────────────────────────────────────────────────────
# CORS — allows React frontend and Node.js backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware (logs slow requests for performance monitoring)
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)

    log_level = logging.WARNING if duration_ms > 3000 else logging.INFO
    logger.log(
        log_level,
        f"{request.method} {request.url.path} → {response.status_code} ({duration_ms}ms)",
    )
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


# ── Register Routers ──────────────────────────────────────────────────────────
# All prediction endpoints: /predict/single, /predict/bulk, etc.
app.include_router(predict.router)

# All explainability endpoints: /explain/shap, /explain/summary, /explain/global
app.include_router(explain.router)

# All AI chat endpoints: /chat/ask, /chat/quick-action, /chat/explain, etc.
app.include_router(ai_chat.router)


# ── Root & Health Endpoints ───────────────────────────────────────────────────
@app.get("/", tags=["Health"], summary="Service information")
async def root():
    """Returns basic service information."""
    return {
        "service": "BankAI Pro - ML Service",
        "version": "1.0.0",
        "status": "running",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_configured": bool(settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY),
        "docs": "/docs",
        "endpoints": {
            "predict_single": "POST /predict/single",
            "predict_bulk": "POST /predict/bulk",
            "explain_shap": "POST /explain/shap",
            "ai_explain": "POST /explain/summary",
            "ai_chat": "POST /chat/quick-action",
            "rag_qa": "POST /chat/ask",
            "feature_importance": "GET /predict/feature-importance",
            "model_info": "GET /predict/info",
        },
    }


@app.get("/health", tags=["Health"], summary="Health check endpoint")
async def health_check():
    """
    Health check for container orchestration (Docker, Kubernetes).
    Returns 200 if service is healthy, 503 if model is not loaded.
    """
    try:
        from models.predict_core import get_model_info
        model_info = get_model_info()
        return {
            "status": "healthy",
            "model_status": model_info["status"],
            "model_name": model_info["model_name"],
            "llm_provider": settings.LLM_PROVIDER,
            "llm_configured": bool(settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY),
        }
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "model_status": "not_loaded",
                "message": "pipeline.pkl not found. ML predictions unavailable.",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


# ── Development Runner ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # Auto-reload on file changes
        reload_dirs=["./"],   # Watch this directory
        log_level="info",
    )
