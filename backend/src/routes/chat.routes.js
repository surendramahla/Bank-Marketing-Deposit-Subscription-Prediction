/**
 * src/routes/chat.routes.js
 * --------------------------
 * AI Copilot chat routes — proxies to the Python ML service
 * and persists conversation history in PostgreSQL.
 *
 * Endpoints:
 *   POST /api/chat/ask           → General banking Q&A (RAG)
 *   POST /api/chat/quick-action  → AI actions (explain/strategy/script/email)
 *   GET  /api/chat/history       → User's chat history
 *   DELETE /api/chat/history     → Clear chat history
 */
const express = require('express');
const axios = require('axios');
const { body } = require('express-validator');

const { query } = require('../db/pool');
const config = require('../config');
const { authenticate } = require('../middleware/auth.middleware');
const { validate } = require('../middleware/validate.middleware');
const logger = require('../utils/logger');

const router = express.Router();
router.use(authenticate);

const mlClient = axios.create({
  baseURL: config.mlService.url,
  timeout: 60000, // 60s for LLM calls
});

// ── Helper: save chat to history ──────────────────────────────────────────────
async function saveChatHistory(userId, action, question, response, metadata, customerId = null) {
  try {
    await query(
      `INSERT INTO chat_history (user_id, customer_id, action, question, response, metadata)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [userId, customerId, action, question, response, metadata ? JSON.stringify(metadata) : null]
    );
  } catch (err) {
    logger.warn('Failed to save chat history:', err.message);
    // Non-critical — don't fail the request
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/chat/ask
// General Q&A (RAG-backed) — no customer context needed
// ─────────────────────────────────────────────────────────────────────────────
router.post(
  '/ask',
  [body('question').notEmpty().withMessage('Question is required'), validate],
  async (req, res) => {
    const { question, context } = req.body;
    try {
      const mlResponse = await mlClient.post('/chat/ask', { question, context });
      const { response } = mlResponse.data;

      await saveChatHistory(req.user.id, 'general_ask', question, response, null);

      res.json({ success: true, data: { response, action: 'general_ask' } });
    } catch (err) {
      if (err.code === 'ECONNREFUSED') {
        return res.status(503).json({ success: false, error: 'AI service unavailable. Ensure ML service is running.' });
      }
      if (err.response?.status === 500) {
        // ML service likely missing API key
        return res.status(503).json({
          success: false,
          error: 'AI features require a GOOGLE_API_KEY or OPENAI_API_KEY in ml_service/.env',
        });
      }
      logger.error('Chat ask error:', err.message);
      res.status(500).json({ success: false, error: 'Chat request failed' });
    }
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/chat/quick-action
// Unified action endpoint — called by the React chat UI buttons
// ─────────────────────────────────────────────────────────────────────────────
router.post(
  '/quick-action',
  [
    body('action')
      .isIn(['explain', 'strategy', 'call_script', 'email', 'general_ask'])
      .withMessage('Invalid action'),
    validate,
  ],
  async (req, res) => {
    const { action, customer, question, message, customer_id } = req.body;

    try {
      // Forward to Python ML service
      const mlResponse = await mlClient.post('/chat/quick-action', {
        action,
        customer,
        question,
        message,
      });

      const result = mlResponse.data;

      // Save to chat history
      await saveChatHistory(
        req.user.id,
        action,
        question || message || action,
        result.response,
        result.metadata,
        customer_id || null
      );

      res.json({ success: true, data: result });
    } catch (err) {
      if (err.code === 'ECONNREFUSED') {
        return res.status(503).json({
          success: false,
          error: 'AI service unavailable',
          detail: 'Start the ML service: uvicorn main:app --port 8000',
        });
      }
      if (err.response?.status === 422) {
        return res.status(422).json({
          success: false,
          error: 'Invalid request',
          details: err.response.data,
        });
      }
      if (err.response?.status === 500 && err.response?.data?.detail?.includes('API key')) {
        return res.status(503).json({
          success: false,
          error: 'AI features require GOOGLE_API_KEY or OPENAI_API_KEY in ml_service/.env',
          hint: 'Get a free Gemini key at https://aistudio.google.com',
        });
      }
      logger.error(`Quick action '${action}' error:`, err.message);
      res.status(500).json({ success: false, error: `Action '${action}' failed` });
    }
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/chat/history
// User's conversation history (paginated)
// ─────────────────────────────────────────────────────────────────────────────
router.get('/history', async (req, res) => {
  try {
    const page = parseInt(req.query.page, 10) || 1;
    const limit = parseInt(req.query.limit, 10) || 20;
    const offset = (page - 1) * limit;

    const countRes = await query(
      'SELECT COUNT(*) FROM chat_history WHERE user_id = $1',
      [req.user.id]
    );
    const total = parseInt(countRes.rows[0].count, 10);

    const { rows } = await query(
      `SELECT ch.id, ch.action, ch.question, ch.response, ch.metadata, ch.created_at,
              c.job, c.age
       FROM chat_history ch
       LEFT JOIN customers c ON ch.customer_id = c.id
       WHERE ch.user_id = $1
       ORDER BY ch.created_at DESC
       LIMIT $2 OFFSET $3`,
      [req.user.id, limit, offset]
    );

    res.json({
      success: true,
      data: rows,
      pagination: { total, page, limit, totalPages: Math.ceil(total / limit) },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch chat history' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// DELETE /api/chat/history
// Clear current user's chat history
// ─────────────────────────────────────────────────────────────────────────────
router.delete('/history', async (req, res) => {
  try {
    await query('DELETE FROM chat_history WHERE user_id = $1', [req.user.id]);
    res.json({ success: true, message: 'Chat history cleared' });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to clear history' });
  }
});

module.exports = router;
