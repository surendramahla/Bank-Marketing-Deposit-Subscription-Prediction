/**
 * src/server.js
 * -------------
 * BankAI Pro — Node.js + Express API Gateway
 *
 * This server acts as the API Gateway between:
 *   - React Frontend  (port 3000)
 *   - PostgreSQL Database
 *   - Python ML Service (port 8000)
 *
 * Architecture:
 *   React (3000) → Node.js (5000) → PostgreSQL + ML Service (8000)
 *
 * Features:
 *   ✓ JWT Authentication
 *   ✓ Rate Limiting
 *   ✓ CORS
 *   ✓ Helmet (security headers)
 *   ✓ Request Compression
 *   ✓ Structured Logging (Winston + Morgan)
 *   ✓ File Upload (Multer)
 *   ✓ Input Validation (express-validator)
 *   ✓ ML Service Proxy (axios)
 *   ✓ PostgreSQL (pg pool)
 *
 * Run:
 *   npm run dev      → Development with nodemon auto-reload
 *   npm start        → Production
 */
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const cookieParser = require('cookie-parser');
const rateLimit = require('express-rate-limit');
const path = require('path');
const fs = require('fs');

const config = require('./config');
const logger = require('./utils/logger');
const { pool } = require('./db/pool');

// ── Route imports ─────────────────────────────────────────────────────────────
const authRoutes = require('./routes/auth.routes');
const customerRoutes = require('./routes/customers.routes');
const predictionRoutes = require('./routes/predictions.routes');
const analyticsRoutes = require('./routes/analytics.routes');
const chatRoutes = require('./routes/chat.routes');
const campaignRoutes = require('./routes/campaigns.routes');

// ── App initialization ────────────────────────────────────────────────────────
const app = express();

// Ensure upload directory exists
const uploadDir = config.upload.dir;
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// ══════════════════════════════════════════════════════════════
// MIDDLEWARE STACK
// ══════════════════════════════════════════════════════════════

// 1. Security headers (Helmet)
app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' },
}));

// 2. CORS — allow React frontend and other allowed origins
app.use(cors({
  origin: (origin, callback) => {
    const allowed = [config.frontendUrl, 'http://localhost:3000', 'http://localhost:5173', 'http://localhost:5174'];
    if (!origin || allowed.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error(`CORS blocked: ${origin}`));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
}));

// 3. Compression (gzip responses)
app.use(compression());

// 4. Body parsers
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cookieParser());

// 5. HTTP request logging (Morgan → Winston)
app.use(
  morgan(config.isDev ? 'dev' : 'combined', {
    stream: { write: (msg) => logger.http(msg.trim()) },
    skip: (req) => req.url === '/health',  // Don't log health checks
  })
);

// 6. Rate limiting — protect all API endpoints
const limiter = rateLimit({
  windowMs: config.rateLimit.windowMs,
  max: config.rateLimit.max,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: 'Too many requests',
    detail: `Please try again after ${config.rateLimit.windowMs / 60000} minutes`,
  },
});
app.use('/api/', limiter);

// Stricter rate limit for auth endpoints (prevent brute force)
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,   // 15 minutes
  max: 10,                      // 10 login attempts
  message: { success: false, error: 'Too many auth attempts. Try again in 15 minutes.' },
});
app.use('/api/auth/login', authLimiter);
app.use('/api/auth/register', authLimiter);

// ══════════════════════════════════════════════════════════════
// ROUTES
// ══════════════════════════════════════════════════════════════
app.use('/api/auth', authRoutes);
app.use('/api/customers', customerRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/campaigns', campaignRoutes);

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', async (req, res) => {
  try {
    // Quick DB connectivity check
    await pool.query('SELECT 1');
    res.json({
      status: 'healthy',
      service: 'BankAI Pro Backend',
      version: '1.0.0',
      database: 'connected',
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    res.status(503).json({
      status: 'degraded',
      service: 'BankAI Pro Backend',
      database: 'disconnected',
      error: err.message,
    });
  }
});

// ── API root info ─────────────────────────────────────────────────────────────
app.get('/api', (req, res) => {
  res.json({
    service: 'BankAI Pro API',
    version: '1.0.0',
    description: 'Bank Marketing AI Prediction Platform',
    endpoints: {
      auth: '/api/auth',
      customers: '/api/customers',
      predictions: '/api/predictions',
      analytics: '/api/analytics',
      chat: '/api/chat',
      campaigns: '/api/campaigns',
    },
    docs: 'See /health for system status',
  });
});

// ── 404 handler ───────────────────────────────────────────────────────────────
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Route not found',
    path: req.originalUrl,
    available_routes: ['/api/auth', '/api/customers', '/api/predictions', '/api/analytics', '/api/chat', '/api/campaigns'],
  });
});

// ── Global error handler ──────────────────────────────────────────────────────
app.use((err, req, res, next) => {
  logger.error(`Unhandled error on ${req.method} ${req.path}:`, err.message, err.stack);

  // Handle CORS errors
  if (err.message && err.message.startsWith('CORS blocked')) {
    return res.status(403).json({ success: false, error: err.message });
  }

  res.status(err.status || 500).json({
    success: false,
    error: config.isDev ? err.message : 'Internal server error',
    ...(config.isDev && { stack: err.stack }),
  });
});

// ══════════════════════════════════════════════════════════════
// STARTUP
// ══════════════════════════════════════════════════════════════
async function startServer() {
  logger.info('');
  logger.info('============================================================');
  logger.info('  BankAI Pro — Node.js Backend Starting');
  logger.info('============================================================');

  // Test DB connection
  try {
    await pool.query('SELECT NOW() AS server_time');
    logger.info('✅ PostgreSQL connected');
  } catch (err) {
    logger.warn(`⚠️  PostgreSQL not connected: ${err.message}`);
    logger.warn('   Run: node src/db/init.js to initialize the database');
    logger.warn('   Or start PostgreSQL: docker run -e POSTGRES_PASSWORD=bankai_pass -p 5432:5432 -d postgres:16');
    if (config.env === 'production') {
      logger.error('Database connection is required in production. Exiting.');
      process.exit(1);
    }
    logger.warn('   Continuing in development mode without DB...');
  }

  app.listen(config.port, () => {
    logger.info(`✅ Server running on http://localhost:${config.port}`);
    logger.info(`   Environment: ${config.env}`);
    logger.info(`   ML Service:  ${config.mlService.url}`);
    logger.info(`   Frontend:    ${config.frontendUrl}`);
    logger.info('');
    logger.info('Available routes:');
    logger.info(`  POST http://localhost:${config.port}/api/auth/login`);
    logger.info(`  GET  http://localhost:${config.port}/api/customers`);
    logger.info(`  POST http://localhost:${config.port}/api/predictions/single`);
    logger.info(`  POST http://localhost:${config.port}/api/chat/quick-action`);
    logger.info(`  GET  http://localhost:${config.port}/api/analytics/dashboard`);
    logger.info('============================================================');
  });
}

// ── Graceful shutdown ─────────────────────────────────────────────────────────
const shutdown = async (signal) => {
  logger.info(`\n${signal} received. Shutting down gracefully...`);
  await pool.end();
  logger.info('Database pool closed');
  process.exit(0);
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

startServer();

module.exports = app; // Export for testing
