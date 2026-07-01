/**
 * src/routes/predictions.routes.js
 * ----------------------------------
 * Prediction management routes — single, bulk CSV, history.
 *
 * Endpoints:
 *   POST /api/predictions/single        → Run single prediction (proxy → ML service)
 *   POST /api/predictions/bulk          → Bulk CSV upload (proxy → ML service)
 *   GET  /api/predictions/bulk/template → Download CSV template
 *   GET  /api/predictions               → Prediction history (paginated)
 *   GET  /api/predictions/:id           → Single prediction detail
 *   GET  /api/predictions/customer/:cid → Predictions for a customer
 */
const express = require('express');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const { query } = require('../db/pool');
const config = require('../config');
const { authenticate } = require('../middleware/auth.middleware');
const { uploadCSV, handleUploadError } = require('../middleware/upload.middleware');
const logger = require('../utils/logger');

const router = express.Router();
router.use(authenticate);

// ── ML service axios instance ─────────────────────────────────────────────────
const mlClient = axios.create({
  baseURL: config.mlService.url,
  timeout: config.mlService.timeout,
});

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/predictions/single
// ─────────────────────────────────────────────────────────────────────────────
router.post('/single', async (req, res) => {
  try {
    const payload = { ...req.body, include_shap: req.body.include_shap ?? true };

    // Forward to Python ML service
    const mlResponse = await mlClient.post('/predict/single', payload);
    const mlResult = mlResponse.data;

    // Optionally store in DB if customer_id is provided
    if (req.body.customer_id) {
      try {
        await query(
          `INSERT INTO predictions
             (customer_id, prediction, probability, priority, strategy, model_version,
              shap_values, top_positive, top_negative, confidence_band, input_snapshot, predicted_by)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`,
          [
            req.body.customer_id, mlResult.prediction, mlResult.probability,
            mlResult.priority, mlResult.strategy, mlResult.model,
            mlResult.shap_values ? JSON.stringify(mlResult.shap_values) : null,
            mlResult.top_positive_factors ? JSON.stringify(mlResult.top_positive_factors) : null,
            mlResult.top_negative_factors ? JSON.stringify(mlResult.top_negative_factors) : null,
            mlResult.confidence_band ? JSON.stringify(mlResult.confidence_band) : null,
            JSON.stringify(req.body), req.user.id,
          ]
        );
      } catch (dbErr) {
        if (config.isDev || process.env.NODE_ENV === 'development') {
          logger.warn('DB unavailable — prediction will not be saved');
        } else {
          throw dbErr;
        }
      }
    }

    res.json({ success: true, data: mlResult });
  } catch (err) {
    if (err.code === 'ECONNREFUSED') {
      return res.status(503).json({
        success: false,
        error: 'ML service unavailable',
        detail: `Start the ML service: uvicorn main:app --port 8000 (in bank2/ml_service)`,
      });
    }
    // Forward ML service validation errors
    if (err.response?.status === 422) {
      return res.status(422).json({ success: false, error: 'Invalid input', details: err.response.data });
    }
    logger.error('Single prediction error:', err.message);
    res.status(500).json({ success: false, error: 'Prediction failed' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/predictions/bulk
// ─────────────────────────────────────────────────────────────────────────────
router.post('/bulk', uploadCSV.single('file'), handleUploadError, async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No CSV file uploaded' });
  }

  try {
    // Forward file to ML service
    const formData = new FormData();
    formData.append('file', fs.createReadStream(req.file.path), {
      filename: req.file.originalname || 'upload.csv',
      contentType: 'text/csv',
    });

    const mlResponse = await mlClient.post('/predict/bulk', formData, {
      headers: formData.getHeaders(),
      timeout: 60000, // 60s for large files
    });

    // Clean up uploaded file
    fs.unlink(req.file.path, () => {});

    const mlResult = mlResponse.data;

    res.json({
      success: true,
      data: {
        total_records: mlResult.total_records,
        predictions: mlResult.predictions,
        summary: mlResult.summary,
      },
    });
  } catch (err) {
    // Clean up on error
    if (req.file) fs.unlink(req.file.path, () => {});

    if (err.code === 'ECONNREFUSED') {
      return res.status(503).json({ success: false, error: 'ML service unavailable' });
    }
    logger.error('Bulk prediction error:', err.message);
    res.status(500).json({ success: false, error: 'Bulk prediction failed' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/predictions/bulk/template
// ─────────────────────────────────────────────────────────────────────────────
router.get('/bulk/template', async (req, res) => {
  try {
    const mlResponse = await mlClient.get('/predict/bulk/template', { responseType: 'stream' });
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="bulk_upload_template.csv"');
    mlResponse.data.pipe(res);
  } catch (err) {
    res.status(503).json({ success: false, error: 'ML service unavailable' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/predictions  — Prediction history
// ─────────────────────────────────────────────────────────────────────────────
router.get('/', async (req, res) => {
  try {
    const page = parseInt(req.query.page, 10) || 1;
    const limit = parseInt(req.query.limit, 10) || 20;
    const offset = (page - 1) * limit;
    const priority = req.query.priority;

    let where = 'WHERE 1=1';
    const params = [];
    let pidx = 1;

    if (priority) {
      where += ` AND p.priority = $${pidx++}`;
      params.push(priority);
    }

    const countResult = await query(`SELECT COUNT(*) FROM predictions p ${where}`, params);
    const total = parseInt(countResult.rows[0].count, 10);

    const { rows } = await query(
      `SELECT p.id, p.prediction, p.probability, p.priority, p.strategy,
              p.model_version, p.confidence_band, p.created_at,
              c.age, c.job, c.lead_segment
       FROM predictions p
       LEFT JOIN customers c ON p.customer_id = c.id
       ${where}
       ORDER BY p.created_at DESC
       LIMIT $${pidx} OFFSET $${pidx + 1}`,
      [...params, limit, offset]
    );

    res.json({
      success: true,
      data: rows,
      pagination: { total, page, limit, totalPages: Math.ceil(total / limit) },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch predictions' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/predictions/customer/:customerId
// ─────────────────────────────────────────────────────────────────────────────
router.get('/customer/:customerId', async (req, res) => {
  try {
    const { rows } = await query(
      `SELECT id, prediction, probability, priority, strategy, model_version,
              top_positive, top_negative, confidence_band, created_at
       FROM predictions WHERE customer_id = $1
       ORDER BY created_at DESC`,
      [req.params.customerId]
    );
    res.json({ success: true, data: rows });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch predictions' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/predictions/:id
// ─────────────────────────────────────────────────────────────────────────────
router.get('/:id', async (req, res) => {
  try {
    const { rows } = await query(
      `SELECT p.*, c.age, c.job, c.marital, c.education, c.balance
       FROM predictions p LEFT JOIN customers c ON p.customer_id = c.id
       WHERE p.id = $1`,
      [req.params.id]
    );
    if (!rows.length) {
      return res.status(404).json({ success: false, error: 'Prediction not found' });
    }
    res.json({ success: true, data: rows[0] });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch prediction' });
  }
});

module.exports = router;
