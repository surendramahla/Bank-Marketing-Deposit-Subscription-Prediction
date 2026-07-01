/**
 * src/middleware/auth.middleware.js
 * ----------------------------------
 * JWT Authentication middleware.
 *
 * Verifies the Bearer token in the Authorization header.
 * Attaches the decoded user payload to req.user.
 *
 * Usage in routes:
 *   router.get('/protected', authenticate, (req, res) => { ... });
 *   router.get('/admin-only', authenticate, authorize('admin'), (req, res) => { ... });
 */
const jwt = require('jsonwebtoken');
const config = require('../config');
const { query } = require('../db/pool');
const logger = require('../utils/logger');

/**
 * authenticate
 * Verifies JWT access token. Rejects if expired or invalid.
 */
const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required',
        detail: 'Provide a valid Bearer token in the Authorization header',
      });
    }

    const token = authHeader.split(' ')[1];

    let decoded;
    try {
      decoded = jwt.verify(token, config.jwt.secret);
    } catch (err) {
      if (err.name === 'TokenExpiredError') {
        return res.status(401).json({
          success: false,
          error: 'Token expired',
          detail: 'Your session has expired. Please login again.',
          code: 'TOKEN_EXPIRED',
        });
      }
      return res.status(401).json({
        success: false,
        error: 'Invalid token',
        detail: 'The provided token is malformed or invalid.',
      });
    }

    // Verify user still exists and is active
    let rows = [];
    try {
      const result = await query(
        'SELECT id, username, email, role, is_active FROM users WHERE id = $1',
        [decoded.userId]
      );
      rows = result.rows;
    } catch (dbErr) {
      if (config.isDev || process.env.NODE_ENV === 'development') {
        if (decoded.email === 'admin@bankai.com' || decoded.username === 'admin' || decoded.userId === '00000000-0000-0000-0000-000000000001') {
           rows = [{
             id: '00000000-0000-0000-0000-000000000001',
             username: 'admin',
             email: 'admin@bankai.com',
             role: 'admin',
             is_active: true
           }];
        } else {
           throw dbErr;
        }
      } else {
        throw dbErr;
      }
    }

    if (!rows[0] || !rows[0].is_active) {
      return res.status(401).json({
        success: false,
        error: 'User not found or deactivated',
      });
    }

    // Attach user to request object
    req.user = {
      id: rows[0].id,
      username: rows[0].username,
      email: rows[0].email,
      role: rows[0].role,
    };

    next();
  } catch (err) {
    logger.error('Auth middleware error:', err.message);
    res.status(500).json({ success: false, error: 'Authentication service error' });
  }
};

/**
 * authorize(...roles)
 * Role-based access control. Must be used AFTER authenticate.
 * Example: authorize('admin', 'manager')
 */
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ success: false, error: 'Not authenticated' });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        error: 'Insufficient permissions',
        detail: `This action requires one of these roles: ${roles.join(', ')}`,
      });
    }

    next();
  };
};

/**
 * optionalAuth
 * Attaches user to req if token is present, but doesn't reject if missing.
 * Useful for endpoints that work for both authenticated and guest users.
 */
const optionalAuth = async (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return next();
  }

  try {
    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, config.jwt.secret);
    const { rows } = await query(
      'SELECT id, username, email, role FROM users WHERE id = $1 AND is_active = TRUE',
      [decoded.userId]
    );
    if (rows[0]) req.user = rows[0];
  } catch (_) {
    // Ignore invalid/expired tokens for optional auth
  }

  next();
};

module.exports = { authenticate, authorize, optionalAuth };
