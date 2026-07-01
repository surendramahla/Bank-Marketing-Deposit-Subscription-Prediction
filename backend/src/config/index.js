/**
 * src/config/index.js
 * -------------------
 * Centralised configuration loaded from environment variables.
 * All modules import config from here — no direct process.env access.
 */
require('dotenv').config();

const config = {
  // ── App ──────────────────────────────────────────────────
  env: process.env.NODE_ENV || 'development',
  port: parseInt(process.env.PORT, 10) || 5000,
  isDev: (process.env.NODE_ENV || 'development') === 'development',

  // ── Database (PostgreSQL) ─────────────────────────────────
  db: {
    url: process.env.DATABASE_URL || null,
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT, 10) || 5432,
    name: process.env.DB_NAME || 'bankai_db',
    user: process.env.DB_USER || 'bankai_user',
    password: process.env.DB_PASSWORD || 'bankai_pass',
    // Connection pool settings
    max: 10,
    idleTimeoutMs: 30000,
    connectionTimeoutMs: 2000,
  },

  // ── JWT ──────────────────────────────────────────────────
  jwt: {
    secret: process.env.JWT_SECRET || 'dev-secret-change-in-production',
    refreshSecret: process.env.JWT_REFRESH_SECRET || 'dev-refresh-secret-change-in-production',
    expiresIn: process.env.JWT_EXPIRES_IN || '15m',
    refreshExpiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d',
  },

  // ── ML Service (Python FastAPI - Phase 1) ─────────────────
  mlService: {
    url: process.env.ML_SERVICE_URL || 'http://localhost:8000',
    timeout: 30000, // 30s timeout for ML predictions
  },

  // ── File Uploads ─────────────────────────────────────────
  upload: {
    maxSizeMb: parseInt(process.env.UPLOAD_MAX_SIZE_MB, 10) || 32,
    dir: process.env.UPLOAD_DIR || './uploads',
  },

  // ── CORS ─────────────────────────────────────────────────
  frontendUrl: process.env.FRONTEND_URL || 'http://localhost:3000',

  // ── Rate Limiting ─────────────────────────────────────────
  rateLimit: {
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS, 10) || 15 * 60 * 1000, // 15 min
    max: parseInt(process.env.RATE_LIMIT_MAX, 10) || 100,
  },

  // ── Logging ──────────────────────────────────────────────
  logLevel: process.env.LOG_LEVEL || 'info',
};

module.exports = config;
