# 💼 BankAI Pro — AI Engineer Interview Preparation Guide

> Comprehensive Q&A for AI Engineer Internship interviews.  
> Covers every component of this project with strong, confident answers.

---

## 📌 1. Project Overview Questions

### Q: "Tell me about this project."

**Answer:**
> "BankAI Pro is a full-stack AI engineering project I built to demonstrate end-to-end AI product development. It's built on the UCI Bank Marketing dataset, which contains real-world bank telemarketing data for 45,000+ customers. The core problem is predicting whether a customer will subscribe to a term deposit — a classic binary classification with severe class imbalance (only 11% subscribe).

> I extended this into a production-grade SaaS platform with four integrated layers:
> 1. A **Python FastAPI ML service** that wraps a Random Forest + SMOTE pipeline and adds SHAP explainability and LangChain-powered AI features
> 2. A **Node.js Express API gateway** handling authentication (JWT), data persistence (PostgreSQL), file uploads, and proxying to the ML service
> 3. A **React.js dashboard** with interactive charts, a real-time prediction UI, AI copilot chat, campaign management, and model analytics
> 4. A **PostgreSQL database** with a 7-table schema storing predictions, customers, campaigns, and conversation history for full audit trails

> The AI copilot uses LangChain chains with Google Gemini to explain predictions in plain English, generate marketing strategies, write personalised call scripts, and answer banking questions via RAG."

---

### Q: "Why did you choose this dataset?"

**Answer:**
> "The UCI Bank Marketing dataset is industry-relevant — it comes from a Portuguese bank's actual telemarketing campaigns. It has several interesting ML challenges: severe class imbalance (11% positive class), mixed feature types (categorical + numerical), and real business value. Predicting subscription probability directly maps to reducing wasted marketing calls and increasing conversion rates. This makes it easy to explain business impact to both technical and non-technical interviewers."

---

## 📌 2. Machine Learning Questions

### Q: "Walk me through your ML pipeline."

**Answer:**
> "The pipeline has three stages wrapped in a scikit-learn `Pipeline` object:
> 1. **Preprocessing**: A `ColumnTransformer` applies `StandardScaler` to numerical features (age, balance, campaign, pdays, previous, day) and `OneHotEncoder` to categorical features (job, marital, education, default, housing, loan, contact, month, poutcome). This ensures clean input regardless of raw data format.
> 2. **SMOTE**: Synthetic Minority Over-sampling Technique generates synthetic positive-class samples during training only, addressing the 11% imbalance without data leakage.
> 3. **Random Forest Classifier**: 100 decision trees vote on the final prediction. I chose Random Forest because it handles mixed feature types well, is robust to outliers, provides built-in feature importance, and doesn't require feature scaling (though the pipeline includes it for portability).

> The full pipeline is serialised with `joblib` and loaded at FastAPI startup, so predictions are sub-50ms."

---

### Q: "What is SMOTE and why did you use it?"

**Answer:**
> "SMOTE stands for Synthetic Minority Over-sampling Technique. The Bank Marketing dataset has severe class imbalance — only ~11% of customers subscribed. Without addressing this, a naive model would achieve 89% accuracy by always predicting 'no', which is useless in production.

> SMOTE works by selecting a minority class sample, finding its k nearest neighbours, and generating synthetic samples along the line segments between them in feature space. This enriches the training distribution without simply duplicating existing samples (pure oversampling) or throwing away majority data (undersampling).

> I placed SMOTE inside the scikit-learn pipeline specifically to prevent data leakage — it only runs on training splits, never on validation or test data."

---

### Q: "What is SHAP and how does it work in your project?"

**Answer:**
> "SHAP stands for SHapley Additive exPlanations. It's based on cooperative game theory — it answers 'how much did each feature contribute to this specific prediction?' by computing Shapley values across all possible feature coalitions.

> In my project, I use the `shap.TreeExplainer` because it's optimised for tree-based models like Random Forest and runs in O(TLD) time rather than the exponential complexity of brute-force Shapley computation.

> For each prediction, I return:
> - `shap_values`: dictionary mapping each feature to its contribution score
> - `top_positive_factors`: features pushing the prediction towards 'yes'  
> - `top_negative_factors`: features reducing probability of subscription

> This makes the model interpretable to non-technical bank employees — instead of just 'probability 72%', they see 'high balance (+0.08) and previous success (+0.04) are driving this prediction'."

---

### Q: "What metrics did you track and why?"

**Answer:**
> "I tracked four metrics relevant to an imbalanced classification problem:
> - **Accuracy (89.4%)**: Overall correctness, but misleading for imbalanced data
> - **F1-Score (62.3%)**: Harmonic mean of precision and recall for the positive class — the most relevant metric since false negatives (missing a subscriber) and false positives (wasted calls) both have business costs
> - **ROC-AUC (91.0%)**: Measures ranking quality — how well the model separates classes across all thresholds. This is ideal for a scoring/ranking system where we're generating a probability for each customer
> - **Precision & Recall separately**: To tune the decision threshold based on the bank's cost tolerance (prefer higher recall if cost of a missed subscriber is high, higher precision if agent time is scarce)"

---

## 📌 3. LangChain & LLM Questions

### Q: "Explain your LangChain architecture."

**Answer:**
> "I built four specialised LangChain chains, each with a distinct system prompt and purpose:

> 1. **PredictionExplainer** — Takes the customer profile and SHAP values, produces a plain-English explanation of the prediction for a bank employee. Uses a `ChatPromptTemplate` with the system prompt setting up the AI as an expert ML model interpreter.

> 2. **StrategyAdvisor** — Takes prediction probability and customer segments, returns a ranked list of marketing interventions. The prompt enforces structured output with specific recommendations and timing.

> 3. **ScriptGenerator** — Produces personalised call scripts or email templates. The prompt includes constraints (professional tone, compliance-safe language, specific CTA).

> 4. **RAGChain** — A retrieval-augmented chain backed by ChromaDB. At startup, banking FAQ and marketing guideline documents are chunked using `RecursiveCharacterTextSplitter`, embedded using Google's `embedding-001` model, and stored in ChromaDB. At query time, the top-k relevant chunks are retrieved and injected as context.

> All chains use a consistent `LLMProviderFactory` that switches between Gemini and OpenAI based on the `LLM_PROVIDER` env var, so the system works with either without code changes."

---

### Q: "What is RAG and why did you use it?"

**Answer:**
> "RAG stands for Retrieval-Augmented Generation. Instead of relying solely on the LLM's parametric knowledge (which may be outdated or hallucinated), RAG retrieves relevant context from a curated document store and injects it into the prompt.

> I implemented it for the 'general Q&A' endpoint where bank employees ask questions like 'What is the best time to call customers?' or 'What compliance rules apply to telemarketing?'. Without RAG, the LLM would answer from general training data. With RAG, it answers specifically from our internal banking FAQ and marketing guidelines documents.

> The flow: question → embed query → cosine similarity search in ChromaDB → top-k chunks as context → LLM generates answer grounded in our documents."

---

### Q: "What was the hardest part of integrating LangChain?"

**Answer:**
> "Two challenges stood out:

> First, **prompt engineering for structured output**. The strategy advisor needs to return a ranked list of specific actions, not free text. I iterated on the system prompt multiple times to get consistent formatting — eventually using `| format_instructions` with `PydanticOutputParser` to enforce the schema.

> Second, **lazy initialisation of ChromaDB**. The vector store takes a few seconds to initialise because it embeds documents at startup. I used FastAPI's `lifespan` context manager and background task initialisation so the server starts immediately and the RAG chain becomes available within ~5 seconds, rather than delaying startup."

---

## 📌 4. Backend Engineering Questions

### Q: "Why Node.js as the API gateway instead of exposing FastAPI directly?"

**Answer:**
> "This is a deliberate architectural decision that mirrors production SaaS systems:

> 1. **Separation of concerns**: The ML service should focus purely on ML inference and AI generation. Auth, rate limiting, audit logging, and data persistence are cross-cutting concerns better handled in a dedicated API layer.

> 2. **Security boundary**: FastAPI is only reachable internally. All external traffic goes through Express, which enforces JWT validation, rate limiting, CORS, and Helmet security headers. The ML service doesn't need to implement any of this.

> 3. **Database ownership**: The Express layer owns the PostgreSQL schema, storing customers, prediction history, campaigns, and chat logs. The ML service is stateless — it receives data and returns results.

> 4. **Flexibility**: This architecture allows replacing the ML service without touching the frontend or auth layer, and vice versa."

---

### Q: "Explain your JWT implementation."

**Answer:**
> "I use a dual-token pattern common in enterprise auth:

> **Access Token** (15-minute TTL): Short-lived JWT containing `userId`, `username`, and `role`. Used for every API request. If stolen, exposure window is only 15 minutes.

> **Refresh Token** (7-day TTL): Longer-lived JWT used only to get new access tokens. Critically, I store a SHA-256 hash of the refresh token in PostgreSQL (never the raw token). On each refresh:
> 1. Verify JWT signature
> 2. Look up hash in `refresh_tokens` table
> 3. Check `is_revoked = FALSE` and `expires_at > NOW()`
> 4. Issue new access + refresh tokens and mark the old one revoked (token rotation)

> This means logout is server-side (revoke the token), and if a refresh token is stolen, we detect reuse on the next rotation attempt."

---

### Q: "What's in your PostgreSQL schema?"

**Answer:**
> "Seven tables with clear ownership:
> - `users` — application users (bank staff) with role-based access (staff/manager/admin)
> - `refresh_tokens` — hashed refresh tokens for secure token rotation
> - `customers` — bank customers from the dataset, extended with `conversion_probability`, `lead_segment`, and `last_predicted_at` AI fields
> - `predictions` — full audit trail of every prediction run: input snapshot, SHAP values, result, timestamp, and which user ran it
> - `model_metrics` — stored accuracy, F1, ROC-AUC from model training for the analytics dashboard
> - `campaigns` — marketing campaigns with auto-calculated `conversion_rate` as a generated column
> - `campaign_customers` — many-to-many junction tracking contact outcomes per campaign
> - `chat_history` — AI copilot conversations stored per user for history display

> I used database triggers for `updated_at` auto-timestamping and a generated column for `conversion_rate` to ensure consistency without application logic."

---

## 📌 5. Frontend Questions

### Q: "How does authentication work in the React frontend?"

**Answer:**
> "I built an `AuthContext` using React Context API. On app load, it checks `localStorage` for an access token and calls `GET /api/auth/me` to restore the session. The Axios API client has two interceptors:

> 1. **Request interceptor** — automatically attaches the Bearer token to every request header.

> 2. **Response interceptor** — on 401 with `TOKEN_EXPIRED` code, it automatically calls the refresh endpoint, stores the new tokens, and retries the original request — completely transparently to the calling component.

> Protected routes use a `ProtectedLayout` component that checks `user` state and redirects to `/login` if null. This pattern is maintainable and doesn't scatter auth checks across pages."

---

### Q: "Why Recharts for charts?"

**Answer:**
> "Recharts is built on D3 but exposes a declarative React component API, making it far easier to integrate in a React codebase than raw D3. It supports all the chart types I needed: `AreaChart` for trends, `BarChart` for feature importance, `PieChart` for segment distribution. It's also tree-shakeable, so unused chart types don't increase bundle size. For production, I'd evaluate Chart.js (lighter) or Visx (more control) based on requirements."

---

## 📌 6. System Design Questions

### Q: "How would you scale this system to handle 1 million customers?"

**Answer:**
> "Several changes at each layer:

> **Database**: Partition the `customers` and `predictions` tables by `created_at` (range partitioning). Add read replicas for analytics queries. Use connection pooling via PgBouncer between the app and PostgreSQL.

> **ML Service**: Add model serving with Ray Serve or Triton Inference Server for GPU-accelerated inference. Cache frequent predictions using Redis (customer profiles don't change every second). Queue bulk CSV processing through Celery + Redis rather than blocking HTTP.

> **Backend**: Horizontal scaling behind a load balancer. Move session state (refresh tokens) to Redis for shared state across instances. Add a CDN for frontend static assets.

> **LLM**: Implement request batching for similar prompts. Add a semantic cache (Redis + embeddings) — if two employees ask 'explain prediction for similar customers', return the cached response. Use streaming responses for chat to improve perceived latency."

---

### Q: "What security concerns did you address?"

**Answer:**
> "Several layers:
> - **Helmet.js** sets 11 HTTP security headers (CSP, HSTS, X-Frame-Options, etc.)
> - **CORS** whitelist — only the frontend origin is allowed, not wildcard
> - **Rate limiting** — 100 requests per 15 minutes globally, 10 per 15 minutes on auth endpoints to prevent brute force
> - **bcrypt** with cost factor 12 for password hashing — ~250ms per hash, making brute-force attacks impractical
> - **JWT** stored in memory (not cookies) on the frontend to avoid CSRF, with short TTL
> - **Refresh tokens** hashed before database storage — even if the DB is compromised, raw tokens aren't exposed
> - **Input validation** via express-validator on every route — prevents injection attacks
> - **SQL parameterised queries** throughout — no string concatenation in SQL, preventing injection"

---

## 📌 7. Behavioural Questions

### Q: "What was the most challenging technical problem you solved?"

**Answer:**
> "The most challenging was making the SHAP explainability work correctly with the scikit-learn pipeline. SHAP's `TreeExplainer` expects the raw Random Forest model, but our pipeline pre-processes input through `ColumnTransformer` (scaling + OHE) first. If I passed raw customer input to SHAP, it would fail because SHAP sees different feature names than the classifier.

> I solved this by extracting the classifier from the pipeline, running the preprocessor separately, and then computing SHAP on the transformed features. The tricky part was mapping encoded feature names (like `poutcome_success`) back to human-readable labels for the UI. I built a mapping function using the OHE's `get_feature_names_out()` method."

---

### Q: "What would you add if you had more time?"

**Answer:**
> "Four high-value additions:

> 1. **A/B testing framework** — Allow the bank to test different call scripts against each other and track conversion rates per script version. LangChain's experiment tracking in LangSmith would be useful here.

> 2. **Real-time streaming** — Replace the chat's request-response model with Server-Sent Events so the LLM response streams token-by-token, improving perceived response time.

> 3. **Model retraining pipeline** — Currently the model is static. I'd add a pipeline that triggers retraining when prediction confidence drops below a threshold (model drift detection), uploads new model versions to S3, and hot-swaps the pipeline without downtime.

> 4. **LangGraph workflow** — Replace the independent LangChain chains with a LangGraph state machine that can route complex queries: 'first predict → if probability > 60%, explain and suggest strategy → if strategy involves email, generate email' — all in one agentic workflow."

---

## 📌 8. Quick-Fire Technical Checks

| Question | Answer |
|----------|--------|
| What is a pipeline in ML? | A sequential series of transformations + estimator in scikit-learn that ensures consistent train/test processing |
| What's the difference between accuracy and F1? | Accuracy counts all correct predictions; F1 is the harmonic mean of precision and recall, better for imbalanced classes |
| What is a transformer in express? | A middleware function that transforms req/res objects (e.g., body parser, auth injector) |
| What does `bcryptjs` do? | Hashes passwords using the Blowfish cipher with a configurable cost factor — computationally expensive by design |
| What is CORS? | Cross-Origin Resource Sharing — HTTP headers that let a browser make requests to a different origin |
| What is ChromaDB? | An open-source vector database for storing and querying embeddings, used in RAG pipelines |
| What is RAG's main advantage? | Grounds LLM responses in specific, current documents — reduces hallucinations and adds domain-specific knowledge |
| What is OneHotEncoding? | Converts categorical variables into binary columns (e.g., job="management" → management=1, technician=0...) |
| Why use Docker Compose? | Defines and runs all services with their dependencies, networking, and volumes in a single YAML file |
| What is a JWT? | JSON Web Token — a signed (or encrypted) token containing claims, used for stateless authentication |

---

*Built as part of an AI Engineer portfolio project — demonstrating ML, LLMs, full-stack development, and production engineering practices.*
