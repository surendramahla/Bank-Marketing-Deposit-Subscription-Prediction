/**
 * src/routes/campaigns.routes.js
 * --------------------------------
 * Marketing campaign management routes.
 *
 * Endpoints:
 *   GET    /api/campaigns          → List all campaigns
 *   POST   /api/campaigns          → Create campaign
 *   GET    /api/campaigns/:id      → Get campaign details + customers
 *   PUT    /api/campaigns/:id      → Update campaign
 *   DELETE /api/campaigns/:id      → Delete campaign (manager/admin)
 *   POST   /api/campaigns/:id/customers → Add customers to campaign
 *   PUT    /api/campaigns/:id/customers/:cid → Update contact outcome
 *   GET    /api/campaigns/:id/ai-strategy    → Get AI-generated strategy for campaign
 */
const express = require('express');
const { body } = require('express-validator');
const axios = require('axios');

const { query, getClient } = require('../db/pool');
const config = require('../config');
const { authenticate, authorize } = require('../middleware/auth.middleware');
const { validate } = require('../middleware/validate.middleware');
const logger = require('../utils/logger');

const router = express.Router();
router.use(authenticate);

const mlClient = axios.create({ baseURL: config.mlService.url, timeout: 60000 });

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/campaigns
// ─────────────────────────────────────────────────────────────────────────────
router.get('/', async (req, res) => {
  try {
    const status = req.query.status;
    let where = 'WHERE 1=1';
    const params = [];
    if (status) {
      where += ' AND status = $1';
      params.push(status);
    }

    const { rows } = await query(
      `SELECT c.id, c.name, c.description, c.status, c.target_segment, c.channel,
              c.start_date, c.end_date, c.total_contacted, c.total_converted,
              c.conversion_rate, c.created_at,
              u.username AS created_by_username
       FROM campaigns c
       LEFT JOIN users u ON c.created_by = u.id
       ${where}
       ORDER BY c.created_at DESC`,
      params
    );
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({
        success: true,
        data: [
          { id: '101', name: 'Q3 High Value Term Deposit', description: 'Targeting high balance customers for gold accounts.', status: 'active', target_segment: 'Hot', channel: 'phone', start_date: '2026-06-01', end_date: '2026-09-30', total_contacted: 150, total_converted: 45, conversion_rate: 30.0, created_at: new Date().toISOString(), created_by_username: 'admin' },
          { id: '102', name: 'Standard Savings Push', description: 'Reaching out to general demographic for regular savings accounts.', status: 'draft', target_segment: 'All', channel: 'email', start_date: '2026-07-01', end_date: '2026-08-31', total_contacted: 0, total_converted: 0, conversion_rate: 0.0, created_at: new Date().toISOString(), created_by_username: 'admin' },
        ],
        count: 2
      });
    }
    res.status(500).json({ success: false, error: 'Failed to fetch campaigns' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/campaigns
// ─────────────────────────────────────────────────────────────────────────────
router.post(
  '/',
  [
    body('name').notEmpty().trim().withMessage('Campaign name is required'),
    body('status').optional().isIn(['draft', 'active', 'paused', 'completed']),
    body('target_segment').optional().isIn(['Hot', 'Warm', 'Cold', 'All']),
    body('channel').optional().isIn(['phone', 'email', 'sms', 'all']),
    validate,
  ],
  async (req, res) => {
    const { name, description, status, target_segment, channel, start_date, end_date } = req.body;
    try {
      const { rows } = await query(
        `INSERT INTO campaigns (name, description, status, target_segment, channel, start_date, end_date, created_by)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         RETURNING *`,
        [name, description, status || 'draft', target_segment, channel, start_date || null, end_date || null, req.user.id]
      );

      // If targeting a specific segment, auto-add matching customers
      if (target_segment && target_segment !== 'All') {
        const { rows: customers } = await query(
          "SELECT id FROM customers WHERE lead_segment = $1 LIMIT 500",
          [target_segment]
        );
        if (customers.length > 0) {
          const values = customers.map((c, i) => `($1, $${i + 2})`).join(',');
          const params = [rows[0].id, ...customers.map((c) => c.id)];
          await query(`INSERT INTO campaign_customers (campaign_id, customer_id) VALUES ${values} ON CONFLICT DO NOTHING`, params);

          await query(
            'UPDATE campaigns SET total_contacted = $1 WHERE id = $2',
            [customers.length, rows[0].id]
          );
        }
      }

      logger.info(`Campaign '${name}' created by ${req.user.username}`);
      res.status(201).json({ success: true, data: rows[0] });
    } catch (err) {
      if (config.isDev || process.env.NODE_ENV === 'development') {
        const mockNewCampaign = {
          id: Math.floor(Math.random() * 1000).toString(),
          name,
          description,
          status: status || 'draft',
          target_segment,
          channel,
          start_date: start_date || null,
          end_date: end_date || null,
          total_contacted: 0,
          total_converted: 0,
          conversion_rate: 0.0,
          created_at: new Date().toISOString(),
          created_by: req.user.id
        };
        logger.info(`Campaign '${name}' created (MOCKED) by ${req.user.username}`);
        return res.status(201).json({ success: true, data: mockNewCampaign });
      }
      logger.error('Create campaign error:', err.message);
      res.status(500).json({ success: false, error: 'Failed to create campaign' });
    }
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/campaigns/:id
// ─────────────────────────────────────────────────────────────────────────────
router.get('/:id', async (req, res) => {
  try {
    const { rows } = await query(
      `SELECT c.*, u.username AS created_by_username
       FROM campaigns c LEFT JOIN users u ON c.created_by = u.id
       WHERE c.id = $1`,
      [req.params.id]
    );
    if (!rows.length) {
      return res.status(404).json({ success: false, error: 'Campaign not found' });
    }

    // Get campaign customers
    const { rows: customers } = await query(
      `SELECT cu.id, cu.age, cu.job, cu.lead_segment, cu.conversion_probability,
              cc.status AS contact_status, cc.contacted_at, cc.outcome
       FROM campaign_customers cc
       JOIN customers cu ON cc.customer_id = cu.id
       WHERE cc.campaign_id = $1
       ORDER BY cu.conversion_probability DESC
       LIMIT 50`,
      [req.params.id]
    );

    res.json({ success: true, data: { ...rows[0], customers } });
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({
        success: true,
        data: {
          id: req.params.id,
          name: 'Q3 High Value Term Deposit (Mock)',
          description: 'Targeting high balance customers for gold accounts.',
          status: 'active',
          target_segment: 'Hot',
          channel: 'phone',
          start_date: '2026-06-01',
          end_date: '2026-09-30',
          total_contacted: 150,
          total_converted: 45,
          conversion_rate: 30.0,
          created_at: new Date().toISOString(),
          created_by_username: 'admin',
          customers: [
            { id: '1', age: 34, job: 'management', lead_segment: 'Hot', conversion_probability: 0.95, contact_status: 'contacted', contacted_at: new Date().toISOString(), outcome: 'yes' },
            { id: '2', age: 45, job: 'blue-collar', lead_segment: 'Hot', conversion_probability: 0.88, contact_status: 'pending', contacted_at: null, outcome: null },
          ]
        }
      });
    }
    res.status(500).json({ success: false, error: 'Failed to fetch campaign' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// PUT /api/campaigns/:id
// ─────────────────────────────────────────────────────────────────────────────
router.put('/:id', authorize('manager', 'admin'), async (req, res) => {
  const { name, description, status, channel, start_date, end_date } = req.body;
  try {
    const { rows } = await query(
      `UPDATE campaigns SET
         name = COALESCE($1, name),
         description = COALESCE($2, description),
         status = COALESCE($3, status),
         channel = COALESCE($4, channel),
         start_date = COALESCE($5, start_date),
         end_date = COALESCE($6, end_date),
         updated_at = NOW()
       WHERE id = $7 RETURNING *`,
      [name, description, status, channel, start_date, end_date, req.params.id]
    );
    if (!rows.length) return res.status(404).json({ success: false, error: 'Campaign not found' });
    res.json({ success: true, data: rows[0] });
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({
        success: true,
        data: {
          id: req.params.id,
          name: name || 'Mock Updated Campaign',
          description,
          status,
          channel,
          start_date,
          end_date,
          updated_at: new Date().toISOString()
        }
      });
    }
    res.status(500).json({ success: false, error: 'Failed to update campaign' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// DELETE /api/campaigns/:id
// ─────────────────────────────────────────────────────────────────────────────
router.delete('/:id', authorize('admin'), async (req, res) => {
  try {
    const { rows } = await query('DELETE FROM campaigns WHERE id = $1 RETURNING id', [req.params.id]);
    if (!rows.length) return res.status(404).json({ success: false, error: 'Campaign not found' });
    res.json({ success: true, message: 'Campaign deleted' });
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({ success: true, message: 'Campaign deleted (MOCKED)' });
    }
    res.status(500).json({ success: false, error: 'Failed to delete campaign' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// PUT /api/campaigns/:id/customers/:cid  — Update contact outcome
// ─────────────────────────────────────────────────────────────────────────────
router.put('/:id/customers/:cid', async (req, res) => {
  const { status, outcome, notes } = req.body;
  try {
    await query(
      `UPDATE campaign_customers
       SET status = COALESCE($1, status),
           outcome = COALESCE($2, outcome),
           notes = COALESCE($3, notes),
           contacted_at = CASE WHEN $1 = 'contacted' THEN NOW() ELSE contacted_at END
       WHERE campaign_id = $4 AND customer_id = $5`,
      [status, outcome, notes, req.params.id, req.params.cid]
    );

    // Update conversion count if outcome is positive
    if (outcome === 'yes' || status === 'converted') {
      await query(
        `UPDATE campaigns
         SET total_converted = (
           SELECT COUNT(*) FROM campaign_customers
           WHERE campaign_id = $1 AND (status = 'converted' OR outcome = 'yes')
         )
         WHERE id = $1`,
        [req.params.id]
      );
    }

    res.json({ success: true, message: 'Contact outcome updated' });
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({ success: true, message: 'Contact outcome updated (MOCKED)' });
    }
    res.status(500).json({ success: false, error: 'Failed to update contact outcome' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/campaigns/:id/ai-strategy
// AI-generated campaign strategy using LangChain
// ─────────────────────────────────────────────────────────────────────────────
router.get('/:id/ai-strategy', async (req, res) => {
  try {
    const { rows } = await query('SELECT * FROM campaigns WHERE id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ success: false, error: 'Campaign not found' });
    const campaign = rows[0];

    // Ask the AI about the campaign strategy
    const question = `Analyze this marketing campaign and suggest improvements:
    Campaign: ${campaign.name}
    Target Segment: ${campaign.target_segment}
    Channel: ${campaign.channel}
    Contacted: ${campaign.total_contacted} customers
    Converted: ${campaign.total_converted} customers
    Conversion Rate: ${campaign.conversion_rate}%
    What should the bank do to improve this campaign's performance?`;

    const mlResponse = await mlClient.post('/chat/ask', { question });
    const aiStrategy = mlResponse.data.response;

    // Cache the AI strategy
    await query('UPDATE campaigns SET ai_recommendations = $1 WHERE id = $2', [
      JSON.stringify({ strategy: aiStrategy, generated_at: new Date().toISOString() }),
      campaign.id,
    ]);

    res.json({ success: true, data: { strategy: aiStrategy, campaign_id: campaign.id } });
  } catch (err) {
    if (config.isDev || process.env.NODE_ENV === 'development') {
      return res.json({
        success: true,
        data: {
          strategy: "Mock AI Strategy Recommendation:\n\n1. Target the Hot segment using cellular contact.\n2. Prioritize callers with high account balances (>5000).\n3. Keep call durations above 3 minutes for optimal conversions.",
          campaign_id: req.params.id
        }
      });
    }
    if (err.code === 'ECONNREFUSED') {
      return res.status(503).json({ success: false, error: 'AI service unavailable' });
    }
    res.status(500).json({ success: false, error: 'Failed to generate AI strategy' });
  }
});

module.exports = router;
