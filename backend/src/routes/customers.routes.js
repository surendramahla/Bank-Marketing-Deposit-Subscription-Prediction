/**
 * src/routes/customers.routes.js
 * --------------------------------
 * Customer management CRUD routes.
 *
 * Endpoints:
 *   GET    /api/customers           → List customers (paginated, searchable, filterable)
 *   POST   /api/customers           → Create a customer
 *   GET    /api/customers/:id       → Get single customer with prediction history
 *   PUT    /api/customers/:id       → Update customer
 *   DELETE /api/customers/:id       → Delete customer (admin only)
 *   POST   /api/customers/:id/predict  → Run ML prediction for this customer
 *   GET    /api/customers/segments  → Count by lead segment
 */
const express = require('express');
const { body, query: qv, param } = require('express-validator');
const axios = require('axios');

const { query } = require('../db/pool');
const config = require('../config');
const { authenticate, authorize } = require('../middleware/auth.middleware');
const { validate } = require('../middleware/validate.middleware');
const logger = require('../utils/logger');

const router = express.Router();

// All customer routes require authentication
router.use(authenticate);

// ── Validation rules ──────────────────────────────────────────────────────────
const customerValidation = [
  body('age').isInt({ min: 18, max: 100 }).withMessage('Age must be between 18 and 100'),
  body('job').notEmpty().trim().withMessage('Job is required'),
  body('marital').isIn(['married', 'single', 'divorced']).withMessage('Invalid marital status'),
  body('education').notEmpty().withMessage('Education is required'),
  body('default_credit').isIn(['yes', 'no']).withMessage('default_credit must be yes or no'),
  body('balance').isInt().withMessage('Balance must be an integer'),
  body('housing').isIn(['yes', 'no']).withMessage('housing must be yes or no'),
  body('loan').isIn(['yes', 'no']).withMessage('loan must be yes or no'),
  body('contact').optional().trim(),
  body('day').isInt({ min: 1, max: 31 }).withMessage('Day must be 1–31'),
  body('month').notEmpty().trim().withMessage('Month is required'),
  body('campaign').isInt({ min: 1 }).withMessage('Campaign must be a positive integer'),
  body('pdays').isInt().withMessage('pdays must be an integer (-1 = not contacted)'),
  body('previous').isInt({ min: 0 }).withMessage('previous must be a non-negative integer'),
  body('poutcome').optional().trim(),
];

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/customers
// ─────────────────────────────────────────────────────────────────────────────
router.get(
  '/',
  [
    qv('page').optional().isInt({ min: 1 }),
    qv('limit').optional().isInt({ min: 1, max: 100 }),
    qv('segment').optional().isIn(['Hot', 'Warm', 'Cold']),
    qv('search').optional().trim(),
    qv('sort').optional().isIn(['probability', 'age', 'balance', 'created_at']),
    qv('order').optional().isIn(['asc', 'desc']),
  ],
  async (req, res) => {
    try {
      const page = parseInt(req.query.page, 10) || 1;
      const limit = parseInt(req.query.limit, 10) || 20;
      const offset = (page - 1) * limit;
      const segment = req.query.segment;
      const search = req.query.search;
      const sort = req.query.sort || 'conversion_probability';
      const order = req.query.order || 'desc';

      let where = 'WHERE 1=1';
      const params = [];
      let paramIdx = 1;

      if (segment) {
        where += ` AND lead_segment = $${paramIdx++}`;
        params.push(segment);
      }
      if (search) {
        where += ` AND (job ILIKE $${paramIdx} OR education ILIKE $${paramIdx})`;
        params.push(`%${search}%`);
        paramIdx++;
      }

      const sortMap = {
        probability: 'conversion_probability',
        age: 'age',
        balance: 'balance',
        created_at: 'created_at',
      };
      const sortCol = sortMap[sort] || 'conversion_probability';

      const countResult = await query(`SELECT COUNT(*) FROM customers ${where}`, params);
      const totalCount = parseInt(countResult.rows[0].count, 10);

      const { rows } = await query(
        `SELECT id, age, job, marital, education, balance, housing, loan,
                contact, month, campaign, pdays, previous, poutcome, subscribed,
                conversion_probability, lead_segment, last_predicted_at, created_at
         FROM customers
         ${where}
         ORDER BY ${sortCol} ${order.toUpperCase()}
         LIMIT $${paramIdx} OFFSET $${paramIdx + 1}`,
        [...params, limit, offset]
      );

      res.json({
        success: true,
        data: rows,
        pagination: {
          total: totalCount,
          page,
          limit,
          totalPages: Math.ceil(totalCount / limit),
          hasNext: page * limit < totalCount,
          hasPrev: page > 1,
        },
      });
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({
        success: true,
        data: [
          { id: '1', age: 34, job: 'management', marital: 'single', education: 'tertiary', balance: 4500, housing: 'yes', loan: 'no', contact: 'cellular', month: 'may', campaign: 1, pdays: -1, previous: 0, poutcome: 'unknown', subscribed: 'no', conversion_probability: 0.95, lead_segment: 'Hot', last_predicted_at: new Date().toISOString(), created_at: new Date().toISOString() },
          { id: '2', age: 45, job: 'blue-collar', marital: 'married', education: 'secondary', balance: 1200, housing: 'no', loan: 'yes', contact: 'telephone', month: 'may', campaign: 2, pdays: -1, previous: 0, poutcome: 'unknown', subscribed: 'no', conversion_probability: 0.45, lead_segment: 'Warm', last_predicted_at: new Date().toISOString(), created_at: new Date().toISOString() },
          { id: '3', age: 29, job: 'technician', marital: 'single', education: 'tertiary', balance: 3400, housing: 'yes', loan: 'no', contact: 'cellular', month: 'jun', campaign: 1, pdays: -1, previous: 0, poutcome: 'unknown', subscribed: 'no', conversion_probability: 0.12, lead_segment: 'Cold', last_predicted_at: new Date().toISOString(), created_at: new Date().toISOString() },
        ],
        pagination: {
          total: 3,
          page: 1,
          limit: 20,
          totalPages: 1,
          hasNext: false,
          hasPrev: false,
        },
      });
    }
    logger.error('Get customers error:', err.message);
    res.status(500).json({ success: false, error: 'Failed to fetch customers' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/customers/segments   (must be before /:id)
// ─────────────────────────────────────────────────────────────────────────────
router.get('/segments', async (req, res) => {
  try {
    const { rows } = await query(
      `SELECT
         lead_segment,
         COUNT(*) AS count,
         ROUND(AVG(conversion_probability)::NUMERIC, 2) AS avg_probability
       FROM customers
       GROUP BY lead_segment
       ORDER BY avg_probability DESC NULLS LAST`
    );
    res.json({ success: true, data: rows });
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({
        success: true,
        data: [
          { lead_segment: 'Hot', count: 8540, avg_probability: 0.85 },
          { lead_segment: 'Warm', count: 15300, avg_probability: 0.55 },
          { lead_segment: 'Cold', count: 20100, avg_probability: 0.25 },
        ]
      });
    }
    res.status(500).json({ success: false, error: 'Failed to fetch segments' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/customers
// ─────────────────────────────────────────────────────────────────────────────
router.post('/', customerValidation, validate, async (req, res) => {
  const {
    age, job, marital, education, default_credit, balance,
    housing, loan, contact, day, month, duration, campaign,
    pdays, previous, poutcome, subscribed,
  } = req.body;

  try {
    const { rows } = await query(
      `INSERT INTO customers
         (age, job, marital, education, default_credit, balance, housing, loan,
          contact, day, month, duration, campaign, pdays, previous, poutcome,
          subscribed, created_by)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
       RETURNING *`,
      [age, job, marital, education, default_credit, balance, housing, loan,
       contact || null, day, month, duration || null, campaign, pdays, previous,
       poutcome || 'unknown', subscribed || null, req.user.id]
    );
    logger.info(`Customer created by ${req.user.username}`);
    res.status(201).json({ success: true, data: rows[0] });
  } catch (err) {
    logger.error('Create customer error:', err.message);
    res.status(500).json({ success: false, error: 'Failed to create customer' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/customers/:id
// ─────────────────────────────────────────────────────────────────────────────
router.get('/:id', async (req, res) => {
  try {
    const { rows } = await query('SELECT * FROM customers WHERE id = $1', [req.params.id]);
    if (!rows.length) {
      return res.status(404).json({ success: false, error: 'Customer not found' });
    }

    // Also fetch prediction history for this customer
    const { rows: predictions } = await query(
      `SELECT id, prediction, probability, priority, strategy, model_version,
              top_positive, top_negative, confidence_band, created_at
       FROM predictions WHERE customer_id = $1
       ORDER BY created_at DESC LIMIT 10`,
      [req.params.id]
    );

    res.json({ success: true, data: { ...rows[0], predictions } });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch customer' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// PUT /api/customers/:id
// ─────────────────────────────────────────────────────────────────────────────
router.put('/:id', customerValidation, validate, async (req, res) => {
  const {
    age, job, marital, education, default_credit, balance,
    housing, loan, contact, day, month, duration, campaign,
    pdays, previous, poutcome, subscribed,
  } = req.body;

  try {
    const { rows } = await query(
      `UPDATE customers SET
         age=$1, job=$2, marital=$3, education=$4, default_credit=$5,
         balance=$6, housing=$7, loan=$8, contact=$9, day=$10, month=$11,
         duration=$12, campaign=$13, pdays=$14, previous=$15, poutcome=$16,
         subscribed=$17, updated_at=NOW()
       WHERE id = $18 RETURNING *`,
      [age, job, marital, education, default_credit, balance, housing, loan,
       contact || null, day, month, duration || null, campaign, pdays, previous,
       poutcome || 'unknown', subscribed || null, req.params.id]
    );

    if (!rows.length) {
      return res.status(404).json({ success: false, error: 'Customer not found' });
    }
    res.json({ success: true, data: rows[0] });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to update customer' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// DELETE /api/customers/:id  — Admin only
// ─────────────────────────────────────────────────────────────────────────────
router.delete('/:id', authorize('admin', 'manager'), async (req, res) => {
  try {
    const { rows } = await query(
      'DELETE FROM customers WHERE id = $1 RETURNING id',
      [req.params.id]
    );
    if (!rows.length) {
      return res.status(404).json({ success: false, error: 'Customer not found' });
    }
    logger.info(`Customer ${req.params.id} deleted by ${req.user.username}`);
    res.json({ success: true, message: 'Customer deleted successfully' });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to delete customer' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/customers/:id/predict
// Runs ML prediction for a specific customer (calls Python ML service)
// ─────────────────────────────────────────────────────────────────────────────
router.post('/:id/predict', async (req, res) => {
  try {
    // Fetch customer
    let customer;
    try {
      const { rows } = await query('SELECT * FROM customers WHERE id = $1', [req.params.id]);
      if (!rows.length) {
        return res.status(404).json({ success: false, error: 'Customer not found' });
      }
      customer = rows[0];
    } catch (dbErr) {
      if (config.isDev || process.env.NODE_ENV === 'development') {
        logger.warn('DB unavailable — using mock customer data for prediction');
        customer = {
          id: req.params.id, age: 34, job: 'management', marital: 'single',
          education: 'tertiary', default_credit: 'no', balance: 4500,
          housing: 'yes', loan: 'no', contact: 'cellular', day: 5,
          month: 'may', campaign: 1, pdays: -1, previous: 0, poutcome: 'unknown'
        };
      } else {
        throw dbErr;
      }
    }

    // Map DB fields to ML service expected format
    const mlPayload = {
      age: customer.age,
      job: customer.job,
      marital: customer.marital,
      education: customer.education,
      default: customer.default_credit,
      balance: customer.balance,
      housing: customer.housing,
      loan: customer.loan,
      contact: customer.contact || 'unknown',
      day: customer.day,
      month: customer.month,
      campaign: customer.campaign,
      pdays: customer.pdays,
      previous: customer.previous,
      poutcome: customer.poutcome || 'unknown',
      include_shap: req.query.shap !== 'false',
    };

    // Call Python ML service
    const mlResponse = await axios.post(
      `${config.mlService.url}/predict/single`,
      mlPayload,
      { timeout: config.mlService.timeout }
    );
    const mlResult = mlResponse.data;

    // Store prediction in DB
    try {
      const { rows: predRows } = await query(
        `INSERT INTO predictions
           (customer_id, prediction, probability, priority, strategy, model_version,
            shap_values, top_positive, top_negative, confidence_band, input_snapshot, predicted_by)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
         RETURNING *`,
        [
          customer.id, mlResult.prediction, mlResult.probability, mlResult.priority,
          mlResult.strategy, mlResult.model,
          mlResult.shap_values ? JSON.stringify(mlResult.shap_values) : null,
          mlResult.top_positive_factors ? JSON.stringify(mlResult.top_positive_factors) : null,
          mlResult.top_negative_factors ? JSON.stringify(mlResult.top_negative_factors) : null,
          mlResult.confidence_band ? JSON.stringify(mlResult.confidence_band) : null,
          JSON.stringify(mlPayload),
          req.user.id,
        ]
      );
      
      // Update customer's prediction fields
      const segment = mlResult.probability > 70 ? 'Hot' : mlResult.probability > 40 ? 'Warm' : 'Cold';
      await query(
        `UPDATE customers SET
           conversion_probability = $1,
           lead_segment = $2,
           last_predicted_at = NOW()
         WHERE id = $3`,
        [mlResult.probability / 100, segment, customer.id]
      );
      logger.info(`Prediction run for customer ${customer.id}: ${mlResult.probability}% by ${req.user.username}`);
      return res.json({
        success: true,
        data: { prediction: predRows[0], customer_id: customer.id },
      });
    } catch (dbErr) {
      if (config.isDev || process.env.NODE_ENV === 'development') {
        logger.warn('DB unavailable — prediction completed but not saved');
        return res.json({
          success: true,
          data: { prediction: { ...mlResult, created_at: new Date().toISOString() }, customer_id: customer.id },
        });
      }
      throw dbErr;
    }
  } catch (err) {
    if (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT') {
      return res.status(503).json({
        success: false,
        error: 'ML service unavailable',
        detail: `Cannot reach ML service at ${config.mlService.url}. Is it running?`,
      });
    }
    logger.error('Prediction error:', err.message);
    res.status(500).json({ success: false, error: 'Prediction failed' });
  }
});

module.exports = router;
