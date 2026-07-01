/**
 * src/db/pool.js
 * --------------
 * PostgreSQL connection pool using the `pg` library.
 *
 * Uses a singleton pool so all routes share the same set of
 * database connections. The pool is configured from config/index.js
 * which reads environment variables.
 *
 * Usage:
 *   const pool = require('../db/pool');
 *   const { rows } = await pool.query('SELECT * FROM users WHERE id = $1', [id]);
 */
const { Pool } = require('pg');
const config = require('../config');
const logger = require('../utils/logger');

// ── Build pool config from environment ────────────────────────────────────────
const poolConfig = config.db.url
  ? {
      connectionString: config.db.url,
      ssl: config.env === 'production' ? { rejectUnauthorized: false } : false,
    }
  : {
      host: config.db.host,
      port: config.db.port,
      database: config.db.name,
      user: config.db.user,
      password: config.db.password,
    };

// ── Shared pool instance ──────────────────────────────────────────────────────
const pool = new Pool({
  ...poolConfig,
  max: config.db.max,
  idleTimeoutMillis: config.db.idleTimeoutMs,
  connectionTimeoutMillis: config.db.connectionTimeoutMs,
});

// Log pool errors (network issues, DB restarts, etc.)
pool.on('error', (err) => {
  logger.error('PostgreSQL pool unexpected error:', err.message);
});

pool.on('connect', () => {
  logger.info('New database connection established');
});

/**
 * Executes a query with automatic error logging.
 * Wraps pool.query to add context to errors.
 */
const query = async (text, params) => {
  const start = Date.now();
  try {
    const result = await pool.query(text, params);
    const duration = Date.now() - start;
    logger.debug(`Query executed in ${duration}ms — rows: ${result.rowCount}`);
    return result;
  } catch (err) {
    logger.error(`Query failed: ${text.substring(0, 100)}... Error: ${err.message}`);
    throw err;
  }
};

/**
 * Gets a dedicated client from the pool for transactions.
 *
 * Usage:
 *   const client = await pool.getClient();
 *   try {
 *     await client.query('BEGIN');
 *     await client.query(...);
 *     await client.query('COMMIT');
 *   } catch (e) {
 *     await client.query('ROLLBACK');
 *   } finally {
 *     client.release();
 *   }
 */
const getClient = () => pool.connect();

module.exports = { pool, query, getClient };
