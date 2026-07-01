# 🏦 BankAI Pro — AI-Powered Bank Marketing Prediction Platform

> **An end-to-end production-grade AI Engineering portfolio project** built for AI Engineer Internship applications.  
> Transforms the UCI Bank Marketing dataset into a fully functional SaaS platform with machine learning, LLMs, and a modern web stack.

---

## 📋 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start (Local Dev)](#-quick-start-local-dev)
- [Quick Start (Docker)](#-quick-start-docker)
- [API Reference](#-api-reference)
- [Environment Configuration](#-environment-configuration)
- [The Original ML Model](#-the-original-ml-model)
- [Interview Talking Points](#-interview-talking-points)

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        BankAI Pro                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  React.js   │───▶│  Node.js/Express │───▶│  PostgreSQL   │  │
│  │  Frontend   │    │  API Gateway     │    │  Database     │  │
│  │  (Port 3000)│◀───│  (Port 5000)     │    │  (Port 5432)  │  │
│  └─────────────┘    └────────┬─────────┘    └───────────────┘  │
│                               │                                  │
│                               │ HTTP Proxy                       │
│                               ▼                                  │
│                    ┌──────────────────┐                         │
│                    │  Python FastAPI  │                         │
│                    │  ML + LangChain  │                         │
│                    │  (Port 8000)     │                         │
│                    └──────────────────┘                         │
│                    ┌──────────────────┐                         │
│                    │  Gemini / OpenAI │                         │
│                    │  LLM APIs        │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:** React → Express (JWT auth, rate limit, audit log) → FastAPI (ML inference, SHAP, LangChain chains) → LLM APIs

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React.js 18, Vite, Tailwind CSS 3, Recharts, React Router v6 |
| **Backend** | Node.js 20, Express 4, JWT (jsonwebtoken), Multer, Winston |
| **ML Service** | Python 3.11, FastAPI, scikit-learn, SHAP, LangChain, ChromaDB |
| **LLM** | Google Gemini 1.5 Flash (default, free) / OpenAI GPT-4o |
| **Database** | PostgreSQL 16 (pg pool) |
| **Auth** | JWT Access + Refresh Token rotation, bcrypt hashing |
| **DevOps** | Docker, Docker Compose, Nginx (SPA proxy) |

---

## ✨ Features

### ML Prediction Engine
- **Random Forest + SMOTE pipeline** trained on 45,211 UCI Bank Marketing records
- **SHAP explainability** — per-prediction feature contributions
- **SHAP global importance** — model-wide feature rankings
- Single prediction + bulk CSV batch scoring
- Prediction history stored in PostgreSQL with full audit trail

### AI Copilot (LangChain + Gemini)
- **Explain Prediction** — "Why does this customer have 72% probability?"
- **Marketing Strategy** — Ranked recommendations for the account manager
- **Call Script Generator** — Personalised phone call scripts
- **Email Generator** — One-click marketing emails
- **RAG Q&A** — ChromaDB-backed answers from internal banking documents

### Enterprise Web Platform
- JWT authentication with token refresh rotation
- Role-based access control (staff / manager / admin)
- Customer CRUD with segment filtering and pagination
- Campaign management with AI-generated strategy recommendations
- Dashboard with live KPI cards, charts, and trend analysis
- File upload for bulk CSV scoring (via Multer → FastAPI)
- Rate limiting, CORS, Helmet security headers

---

## 📁 Project Structure

```
bank2/
├── docker-compose.yml          ← One-command full-stack start
├── .env.docker.example         ← Copy to .env and fill in secrets
├── .gitignore
│
├── ml_service/                 ← 🐍 Python FastAPI (Port 8000)
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── core/config.py
│   ├── models/
│   │   ├── pipeline.pkl        ← Trained Random Forest + SMOTE
│   │   ├── features.json
│   │   └── predict_core.py     ← Extended prediction + SHAP
│   ├── chains/                 ← LangChain chains
│   │   ├── prediction_explainer.py
│   │   ├── strategy_advisor.py
│   │   ├── script_generator.py
│   │   └── rag_chain.py
│   ├── routers/
│   │   ├── predict.py
│   │   ├── explain.py
│   │   └── ai_chat.py
│   ├── prompts/                ← System prompt templates
│   └── rag/documents/          ← Banking FAQs for ChromaDB RAG
│
├── backend/                    ← 🟢 Node.js Express (Port 5000)
│   ├── src/
│   │   ├── server.js           ← App entry + middleware stack
│   │   ├── config/index.js     ← Centralised env config
│   │   ├── db/
│   │   │   ├── pool.js         ← PostgreSQL connection pool
│   │   │   ├── schema.sql      ← 7-table schema + triggers
│   │   │   └── init.js         ← DB initializer script
│   │   ├── middleware/
│   │   │   ├── auth.middleware.js    ← JWT verify + RBAC
│   │   │   ├── upload.middleware.js  ← Multer CSV
│   │   │   └── validate.middleware.js
│   │   └── routes/
│   │       ├── auth.routes.js
│   │       ├── customers.routes.js
│   │       ├── predictions.routes.js
│   │       ├── analytics.routes.js
│   │       ├── chat.routes.js
│   │       └── campaigns.routes.js
│   ├── Dockerfile
│   └── package.json
│
└── frontend/                   ← ⚛️  React + Vite (Port 3000)
    ├── src/
    │   ├── App.jsx             ← Routes + Protected Layout
    │   ├── main.jsx
    │   ├── api/api.js          ← Axios + interceptors
    │   ├── context/AuthContext.jsx
    │   ├── components/
    │   │   ├── Sidebar.jsx
    │   │   ├── KPICard.jsx
    │   │   ├── PredictionCard.jsx
    │   │   ├── ChatMessage.jsx
    │   │   └── FeatureImportanceChart.jsx
    │   └── pages/
    │       ├── Login.jsx
    │       ├── Dashboard.jsx
    │       ├── Customers.jsx
    │       ├── Predictions.jsx
    │       ├── AIAssistant.jsx
    │       ├── Analytics.jsx
    │       └── Campaigns.jsx
    ├── Dockerfile
    └── package.json
```

---

## 🚀 Quick Start (Local Dev)

### Prerequisites
- **Python 3.11+** — for the ML service
- **Node.js 18+** — for the backend and frontend
- **PostgreSQL 16** (or Docker for a one-liner DB setup)

### Step 1 — Clone & Setup

```bash
# The project lives in bank2/
cd C:\Users\Asus\Desktop\bank2
```

### Step 2 — Start PostgreSQL (Docker one-liner)

```bash
docker run -d --name bankai-pg \
  -e POSTGRES_DB=bankai_db \
  -e POSTGRES_USER=bankai_user \
  -e POSTGRES_PASSWORD=bankai_pass \
  -p 5432:5432 \
  postgres:16
```

### Step 3 — Python ML Service

```bash
cd ml_service

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate         # Windows
# source venv/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key (optional — core predictions work without it)
copy .env.example .env
# Edit .env: GOOGLE_API_KEY=your_free_gemini_key

# Start server
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs
```

### Step 4 — Node.js Backend

```bash
cd backend

# Install dependencies
npm install

# Initialize database tables
node src/db/init.js

# Start dev server
npm run dev
# → http://localhost:5000/api
```

### Step 5 — React Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
# → http://localhost:5173
```

### Step 6 — Login

| Field | Value |
|-------|-------|
| **URL** | http://localhost:5173 |
| **Email** | admin@bankai.com |
| **Password** | admin123 |

---

## 🐳 Quick Start (Docker)

```bash
cd bank2

# 1. Configure secrets
copy .env.docker.example .env
# Edit .env: add GOOGLE_API_KEY

# 2. Start everything
docker compose up --build

# 3. Open the app
start http://localhost:3000
```

**That's it.** Docker Compose starts PostgreSQL → ML Service → Backend → Frontend in dependency order, with health checks.

---

## 📡 API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | Login → access + refresh tokens |
| `POST` | `/api/auth/register` | Create account |
| `POST` | `/api/auth/refresh` | Rotate tokens |
| `POST` | `/api/auth/logout` | Revoke refresh token |
| `GET` | `/api/auth/me` | Current user profile |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predictions/single` | Single customer prediction |
| `POST` | `/api/predictions/bulk` | Bulk CSV scoring |
| `GET` | `/api/predictions` | Prediction history |

### AI Copilot
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/quick-action` | explain / strategy / call_script / email |
| `POST` | `/api/chat/ask` | General RAG Q&A |
| `GET` | `/api/chat/history` | Conversation history |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/analytics/dashboard` | KPI cards data |
| `GET` | `/api/analytics/monthly-trend` | Prediction trend chart |
| `GET` | `/api/analytics/feature-importance` | SHAP global chart |
| `GET` | `/api/analytics/model-performance` | Accuracy, F1, ROC-AUC |
| `GET` | `/api/analytics/conversion-funnel` | Funnel data |
| `GET` | `/api/analytics/top-leads` | Highest-probability customers |

---

## ⚙️ Environment Configuration

### ML Service (`ml_service/.env`)
```env
LLM_PROVIDER=gemini              # gemini | openai
GOOGLE_API_KEY=your_key_here     # Free at aistudio.google.com
OPENAI_API_KEY=                  # Optional — openai.com
GEMINI_MODEL=gemini-1.5-flash
```

### Backend (`backend/.env`)
```env
DATABASE_URL=postgresql://bankai_user:bankai_pass@localhost:5432/bankai_db
JWT_SECRET=64_char_random_string
JWT_REFRESH_SECRET=another_64_char_string
ML_SERVICE_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

---

## 🤖 The Original ML Model

The ML model preserves the original research pipeline:

| Property | Value |
|----------|-------|
| **Dataset** | UCI Bank Marketing (45,211 records) |
| **Target** | Binary: subscribed to term deposit (yes/no) |
| **Features** | 15 features (demographics + contact history) |
| **Pipeline** | Preprocessing → SMOTE → Random Forest |
| **Accuracy** | 89.43% |
| **F1-Score** | 62.34% (imbalanced class) |
| **ROC-AUC** | 91.02% |
| **Top Features** | campaign, balance, day, age, poutcome_success |

**SMOTE** (Synthetic Minority Over-sampling Technique) is used because only ~11% of customers subscribed — a classic class imbalance problem.

---

