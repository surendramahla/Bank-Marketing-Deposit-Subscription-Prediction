/**
 * src/routes/auth.routes.js
 * -------------------------
 * Authentication routes — JWT-based login/register/refresh/logout.
 *
 * Endpoints:
 *   POST /api/auth/register   → Create new user account
 *   POST /api/auth/login      → Login, get access + refresh tokens
 *   POST /api/auth/refresh    → Exchange refresh token for new access token
 *   POST /api/auth/logout     → Revoke refresh token
 *   GET  /api/auth/me         → Get current user profile
 *   PUT  /api/auth/me         → Update current user profile
 */
const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const { body } = require('express-validator');

const { query } = require('../db/pool');
const config = require('../config');
const { authenticate, authorize } = require('../middleware/auth.middleware');
const { validate } = require('../middleware/validate.middleware');
const logger = require('../utils/logger');

const router = express.Router();

// ── Token generators ──────────────────────────────────────────────────────────
const generateAccessToken = (user) =>
  jwt.sign(
    { userId: user.id, username: user.username, role: user.role },
    config.jwt.secret,
    { expiresIn: config.jwt.expiresIn }
  );

const generateRefreshToken = (user) =>
  jwt.sign(
    { userId: user.id },
    config.jwt.refreshSecret,
    { expiresIn: config.jwt.refreshExpiresIn }
  );

// Hash refresh token before storing (never store raw tokens)
const hashToken = (token) =>
  crypto.createHash('sha256').update(token).digest('hex');

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/auth/register
// ─────────────────────────────────────────────────────────────────────────────
router.post(
  '/register',
  [
    body('username').trim().isLength({ min: 3, max: 30 }).withMessage('Username must be 3–30 characters'),
    body('email').isEmail().normalizeEmail().withMessage('Valid email required'),
    body('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),
    body('first_name').optional().trim().isLength({ max: 50 }),
    body('last_name').optional().trim().isLength({ max: 50 }),
    validate,
  ],
  async (req, res) => {
    const { username, email, password, first_name, last_name } = req.body;
    try {
      // Check for existing user
      const existing = await query(
        'SELECT id FROM users WHERE email = $1 OR username = $2',
        [email, username]
      );
      if (existing.rows.length > 0) {
        return res.status(409).json({
          success: false,
          error: 'Username or email already in use',
        });
      }

      // Hash password (salt rounds = 12)
      const passwordHash = await bcrypt.hash(password, 12);

      // Insert user
      const { rows } = await query(
        `INSERT INTO users (username, email, password_hash, role, first_name, last_name)
         VALUES ($1, $2, $3, 'staff', $4, $5)
         RETURNING id, username, email, role, first_name, last_name, created_at`,
        [username, email, passwordHash, first_name || null, last_name || null]
      );

      const user = rows[0];
      const accessToken = generateAccessToken(user);
      const refreshToken = generateRefreshToken(user);

      // Store hashed refresh token
      await query(
        `INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
         VALUES ($1, $2, NOW() + INTERVAL '7 days')`,
        [user.id, hashToken(refreshToken)]
      );

      logger.info(`New user registered: ${username} (${email})`);

      res.status(201).json({
        success: true,
        message: 'Account created successfully',
        data: {
          user: { id: user.id, username: user.username, email: user.email, role: user.role },
          accessToken,
          refreshToken,
        },
      });
    } catch (err) {
      logger.error('Register error:', err.message);
      res.status(500).json({ success: false, error: 'Registration failed' });
    }
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/auth/login
// ─────────────────────────────────────────────────────────────────────────────

// Dev-mode mock admin user (used when DB is unavailable in development)
const DEV_ADMIN = {
  id: '00000000-0000-0000-0000-000000000001',
  username: 'admin',
  email: 'admin@bankai.com',
  password: 'admin123',
  role: 'admin',
  first_name: 'Admin',
  last_name: 'User',
  is_active: true,
};

router.post(
  '/login',
  [
    body('email').isEmail().normalizeEmail().withMessage('Valid email required'),
    body('password').notEmpty().withMessage('Password is required'),
    validate,
  ],
  async (req, res) => {
    const { email, password } = req.body;
    try {
      const { rows } = await query(
        'SELECT id, username, email, password_hash, role, is_active, first_name, last_name FROM users WHERE email = $1',
        [email]
      );

      const user = rows[0];
      if (!user) {
        return res.status(401).json({ success: false, error: 'Invalid email or password' });
      }
      if (!user.is_active) {
        return res.status(403).json({ success: false, error: 'Account is deactivated. Contact admin.' });
      }

      const passwordValid = await bcrypt.compare(password, user.password_hash);
      if (!passwordValid) {
        return res.status(401).json({ success: false, error: 'Invalid email or password' });
      }

      // Generate tokens
      const accessToken = generateAccessToken(user);
      const refreshToken = generateRefreshToken(user);

      // Store hashed refresh token (revoke old ones for this user - single session)
      await query('UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = $1', [user.id]);
      await query(
        `INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
         VALUES ($1, $2, NOW() + INTERVAL '7 days')`,
        [user.id, hashToken(refreshToken)]
      );

      // Update last_login
      await query('UPDATE users SET last_login = NOW() WHERE id = $1', [user.id]);

      logger.info(`User logged in: ${user.username}`);

      res.json({
        success: true,
        message: 'Login successful',
        data: {
          user: {
            id: user.id,
            username: user.username,
            email: user.email,
            role: user.role,
            first_name: user.first_name,
            last_name: user.last_name,
          },
          accessToken,
          refreshToken,
          expiresIn: config.jwt.expiresIn,
        },
      });
    } catch (err) {
      // ── Dev-mode fallback: allow login without DB ──────────────────────────
      if (config.isDev || process.env.NODE_ENV === 'development') {
        logger.warn('DB unavailable — attempting dev-mode credential check');
        if (
          email === DEV_ADMIN.email &&
          password === DEV_ADMIN.password
        ) {
          const accessToken = generateAccessToken(DEV_ADMIN);
          const refreshToken = generateRefreshToken(DEV_ADMIN);
          logger.info(`[DEV MODE] Admin logged in without DB`);
          return res.json({
            success: true,
            message: 'Login successful (dev mode — no database)',
            data: {
              user: {
                id: DEV_ADMIN.id,
                username: DEV_ADMIN.username,
                email: DEV_ADMIN.email,
                role: DEV_ADMIN.role,
                first_name: DEV_ADMIN.first_name,
                last_name: DEV_ADMIN.last_name,
              },
              accessToken,
              refreshToken,
              expiresIn: config.jwt.expiresIn,
            },
          });
        }
        return res.status(401).json({ success: false, error: 'Invalid email or password' });
      }
      // ───────────────────────────────────────────────────────────────────────
      logger.error('Login error:', err.message);
      res.status(500).json({ success: false, error: 'Login failed' });
    }
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/auth/refresh
// ─────────────────────────────────────────────────────────────────────────────
router.post('/refresh', async (req, res) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    return res.status(401).json({ success: false, error: 'Refresh token required' });
  }

  try {
    // Verify the refresh token signature
    let decoded;
    try {
      decoded = jwt.verify(refreshToken, config.jwt.refreshSecret);
    } catch {
      return res.status(401).json({ success: false, error: 'Invalid or expired refresh token' });
    }

    // Check if it's stored and not revoked
    const tokenHash = hashToken(refreshToken);
    const { rows: tokenRows } = await query(
      `SELECT id FROM refresh_tokens
       WHERE user_id = $1 AND token_hash = $2 AND is_revoked = FALSE AND expires_at > NOW()`,
      [decoded.userId, tokenHash]
    );

    if (!tokenRows.length) {
      return res.status(401).json({ success: false, error: 'Refresh token revoked or expired' });
    }

    // Get user
    const { rows: userRows } = await query(
      'SELECT id, username, email, role FROM users WHERE id = $1 AND is_active = TRUE',
      [decoded.userId]
    );
    if (!userRows.length) {
      return res.status(401).json({ success: false, error: 'User not found' });
    }

    const user = userRows[0];
    const newAccessToken = generateAccessToken(user);
    const newRefreshToken = generateRefreshToken(user);

    // Rotate refresh token
    await query('UPDATE refresh_tokens SET is_revoked = TRUE WHERE token_hash = $1', [tokenHash]);
    await query(
      `INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
       VALUES ($1, $2, NOW() + INTERVAL '7 days')`,
      [user.id, hashToken(newRefreshToken)]
    );

    res.json({
      success: true,
      data: { accessToken: newAccessToken, refreshToken: newRefreshToken },
    });
  } catch (err) {
    logger.error('Refresh token error:', err.message);
    res.status(500).json({ success: false, error: 'Token refresh failed' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/auth/logout
// ─────────────────────────────────────────────────────────────────────────────
router.post('/logout', authenticate, async (req, res) => {
  try {
    await query(
      'UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = $1',
      [req.user.id]
    );
    logger.info(`User logged out: ${req.user.username}`);
    res.json({ success: true, message: 'Logged out successfully' });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Logout failed' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/auth/me
// ─────────────────────────────────────────────────────────────────────────────
router.get('/me', authenticate, async (req, res) => {
  try {
    const { rows } = await query(
      `SELECT id, username, email, role, first_name, last_name, last_login, created_at
       FROM users WHERE id = $1`,
      [req.user.id]
    );
    res.json({ success: true, data: rows[0] });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch profile' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// PUT /api/auth/me  — Update profile
// ─────────────────────────────────────────────────────────────────────────────
router.put(
  '/me',
  authenticate,
  [
    body('first_name').optional().trim().isLength({ max: 50 }),
    body('last_name').optional().trim().isLength({ max: 50 }),
    body('current_password').optional(),
    body('new_password').optional().isLength({ min: 6 }).withMessage('New password must be at least 6 characters'),
    validate,
  ],
  async (req, res) => {
    const { first_name, last_name, current_password, new_password } = req.body;
    try {
      // If changing password, verify current password first
      if (new_password) {
        if (!current_password) {
          return res.status(400).json({ success: false, error: 'Current password required to set new password' });
        }
        const { rows } = await query('SELECT password_hash FROM users WHERE id = $1', [req.user.id]);
        const valid = await bcrypt.compare(current_password, rows[0].password_hash);
        if (!valid) {
          return res.status(400).json({ success: false, error: 'Current password is incorrect' });
        }
        const newHash = await bcrypt.hash(new_password, 12);
        await query('UPDATE users SET password_hash = $1 WHERE id = $2', [newHash, req.user.id]);
      }

      // Update profile fields
      await query(
        `UPDATE users SET first_name = COALESCE($1, first_name), last_name = COALESCE($2, last_name)
         WHERE id = $3`,
        [first_name, last_name, req.user.id]
      );

      const { rows } = await query(
        'SELECT id, username, email, role, first_name, last_name FROM users WHERE id = $1',
        [req.user.id]
      );
      res.json({ success: true, message: 'Profile updated', data: rows[0] });
    } catch (err) {
      res.status(500).json({ success: false, error: 'Failed to update profile' });
    }
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/auth/users  — Admin: list all users
// ─────────────────────────────────────────────────────────────────────────────
router.get('/users', authenticate, authorize('admin'), async (req, res) => {
  try {
    const { rows } = await query(
      `SELECT id, username, email, role, first_name, last_name, is_active, last_login, created_at
       FROM users ORDER BY created_at DESC`
    );
    res.json({ success: true, data: rows, count: rows.length });
  } catch (err) {
    res.status(500).json({ success: false, error: 'Failed to fetch users' });
  }
});

module.exports = router;
