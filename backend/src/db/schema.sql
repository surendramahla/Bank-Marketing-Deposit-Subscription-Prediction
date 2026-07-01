-- ============================================================
-- BankAI Pro - PostgreSQL Database Schema
-- ============================================================
-- Run: psql -U postgres -d bankai_db -f schema.sql
-- Or:  node src/db/init.js
-- ============================================================

-- ── Extensions ───────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For fuzzy text search on customer names

-- ── Drop existing tables (for clean reinstall) ───────────────
-- Comment these out in production!
DROP TABLE IF EXISTS chat_history CASCADE;
DROP TABLE IF EXISTS campaign_customers CASCADE;
DROP TABLE IF EXISTS campaigns CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS model_metrics CASCADE;
DROP TABLE IF EXISTS refresh_tokens CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- ═══════════════════════════════════════════════════════════════
-- TABLE: users
-- Application users (bank employees, managers, admins)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username    VARCHAR(80) UNIQUE NOT NULL,
    email       VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role        VARCHAR(20) NOT NULL DEFAULT 'staff'
                CHECK (role IN ('staff', 'manager', 'admin')),
    first_name  VARCHAR(50),
    last_name   VARCHAR(50),
    is_active   BOOLEAN DEFAULT TRUE,
    last_login  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- ═══════════════════════════════════════════════════════════════
-- TABLE: refresh_tokens
-- JWT refresh token storage (for secure token rotation)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE refresh_tokens (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  VARCHAR(255) NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    is_revoked  BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_hash ON refresh_tokens(token_hash);

-- ═══════════════════════════════════════════════════════════════
-- TABLE: customers
-- Bank customers from the UCI Bank Marketing dataset
-- Extended with AI prediction fields
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE customers (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- ── UCI Dataset Fields (original model features) ──────────
    age             INTEGER CHECK (age BETWEEN 18 AND 100),
    job             VARCHAR(50),
    marital         VARCHAR(20),
    education       VARCHAR(50),
    default_credit  VARCHAR(5),          -- 'yes' | 'no'
    balance         INTEGER,             -- avg yearly balance in euros
    housing         VARCHAR(5),          -- housing loan: 'yes' | 'no'
    loan            VARCHAR(5),          -- personal loan: 'yes' | 'no'
    contact         VARCHAR(20),         -- contact type: cellular | telephone | unknown
    day             INTEGER CHECK (day BETWEEN 1 AND 31),
    month           VARCHAR(10),
    duration        INTEGER,             -- last contact duration (seconds)
    campaign        INTEGER,             -- contacts during this campaign
    pdays           INTEGER,             -- days since last contact (-1 = never)
    previous        INTEGER,             -- contacts before this campaign
    poutcome        VARCHAR(20),         -- previous campaign outcome
    subscribed      VARCHAR(5),          -- actual outcome: 'yes' | 'no' (ground truth)

    -- ── AI Prediction Fields (extended) ───────────────────────
    conversion_probability  FLOAT DEFAULT 0.0,
    lead_segment    VARCHAR(20)          -- 'Hot' | 'Warm' | 'Cold' | NULL
                    CHECK (lead_segment IN ('Hot', 'Warm', 'Cold') OR lead_segment IS NULL),
    last_predicted_at       TIMESTAMPTZ,

    -- ── App Metadata ───────────────────────────────────────────
    created_by      UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_customers_lead_segment ON customers(lead_segment);
CREATE INDEX idx_customers_conversion_prob ON customers(conversion_probability DESC);
CREATE INDEX idx_customers_subscribed ON customers(subscribed);
CREATE INDEX idx_customers_job ON customers(job);
CREATE INDEX idx_customers_age ON customers(age);

-- ═══════════════════════════════════════════════════════════════
-- TABLE: predictions
-- Every ML prediction result stored here for audit trail
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE predictions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id     UUID REFERENCES customers(id) ON DELETE SET NULL,

    -- ── Prediction Result ─────────────────────────────────────
    prediction      VARCHAR(5) NOT NULL CHECK (prediction IN ('yes', 'no')),
    probability     FLOAT NOT NULL CHECK (probability BETWEEN 0 AND 100),
    priority        VARCHAR(10) CHECK (priority IN ('High', 'Medium', 'Low')),
    strategy        TEXT,                -- ML-generated strategy text
    model_version   VARCHAR(50) DEFAULT 'Random Forest + SMOTE v1.0',

    -- ── SHAP / Explainability ─────────────────────────────────
    shap_values     JSONB,               -- {"feature": shap_value, ...}
    top_positive    JSONB,               -- top positive contributing features
    top_negative    JSONB,               -- top negative contributing features
    confidence_band JSONB,               -- {"low": x, "high": y}

    -- ── Input Snapshot (what data was used) ───────────────────
    input_snapshot  JSONB,               -- Full customer data at time of prediction

    -- ── App Metadata ───────────────────────────────────────────
    predicted_by    UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_customer_id ON predictions(customer_id);
CREATE INDEX idx_predictions_probability ON predictions(probability DESC);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_priority ON predictions(priority);

-- ═══════════════════════════════════════════════════════════════
-- TABLE: model_metrics
-- Stores model performance metrics for analytics dashboard
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE model_metrics (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name  VARCHAR(100) NOT NULL,
    accuracy    FLOAT,
    f1_score    FLOAT,
    roc_auc     FLOAT,
    precision   FLOAT,
    recall      FLOAT,
    metrics_json JSONB,           -- Full metrics including confusion matrix, feature importance
    is_active   BOOLEAN DEFAULT FALSE,
    trained_at  TIMESTAMPTZ DEFAULT NOW(),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════
-- TABLE: campaigns
-- Marketing campaign management
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE campaigns (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(200) NOT NULL,
    description     TEXT,
    status          VARCHAR(20) DEFAULT 'draft'
                    CHECK (status IN ('draft', 'active', 'paused', 'completed')),
    target_segment  VARCHAR(20)  -- 'Hot' | 'Warm' | 'Cold' | 'All'
                    CHECK (target_segment IN ('Hot', 'Warm', 'Cold', 'All') OR target_segment IS NULL),
    channel         VARCHAR(30)  -- 'phone' | 'email' | 'sms' | 'all'
                    CHECK (channel IN ('phone', 'email', 'sms', 'all') OR channel IS NULL),
    start_date      DATE,
    end_date        DATE,
    total_contacted INTEGER DEFAULT 0,
    total_converted INTEGER DEFAULT 0,
    conversion_rate FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_contacted > 0
        THEN ROUND((total_converted::FLOAT / total_contacted * 100)::NUMERIC, 2)::FLOAT
        ELSE 0 END
    ) STORED,
    ai_recommendations JSONB,    -- AI-generated strategy recommendations
    created_by      UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaigns_created_at ON campaigns(created_at DESC);

-- ═══════════════════════════════════════════════════════════════
-- TABLE: campaign_customers (junction)
-- Which customers are targeted by which campaigns
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE campaign_customers (
    campaign_id     UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    customer_id     UUID REFERENCES customers(id) ON DELETE CASCADE,
    status          VARCHAR(20) DEFAULT 'pending'
                    CHECK (status IN ('pending', 'contacted', 'converted', 'rejected')),
    contacted_at    TIMESTAMPTZ,
    outcome         VARCHAR(20),
    notes           TEXT,
    PRIMARY KEY (campaign_id, customer_id)
);

-- ═══════════════════════════════════════════════════════════════
-- TABLE: chat_history
-- AI Copilot conversation history per user
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE chat_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    customer_id     UUID REFERENCES customers(id) ON DELETE SET NULL,
    action          VARCHAR(30),         -- 'explain' | 'strategy' | 'call_script' | 'email' | 'general_ask'
    question        TEXT,                -- employee's question
    response        TEXT,                -- AI response
    metadata        JSONB,               -- prediction result, probability, etc.
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_chat_history_created_at ON chat_history(created_at DESC);

-- ═══════════════════════════════════════════════════════════════
-- FUNCTIONS & TRIGGERS
-- ═══════════════════════════════════════════════════════════════

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_campaigns_updated_at
    BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ═══════════════════════════════════════════════════════════════
-- SEED: Default admin user
-- Password: admin123 (bcrypt hash — CHANGE IN PRODUCTION)
-- ═══════════════════════════════════════════════════════════════
INSERT INTO users (username, email, password_hash, role, first_name, last_name)
VALUES (
    'admin',
    'admin@bankai.com',
    '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.s5uKaa',  -- admin123
    'admin',
    'Bank',
    'Admin'
) ON CONFLICT (username) DO NOTHING;

-- Seed initial model metrics (from the original trained model)
INSERT INTO model_metrics (model_name, accuracy, f1_score, roc_auc, precision, recall, is_active, metrics_json)
VALUES (
    'Random Forest + SMOTE Pipeline',
    0.8943,
    0.6234,
    0.9102,
    0.6891,
    0.5712,
    TRUE,
    '{"confusion_matrix": [[7421, 412], [438, 729]], "feature_importance": {"campaign": 0.1308, "balance": 0.1010, "day": 0.0925, "age": 0.0867, "poutcome_success": 0.0434}}'
) ON CONFLICT DO NOTHING;
