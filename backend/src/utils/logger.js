/**
 * src/utils/logger.js
 * --------------------
 * Winston-based structured logger.
 * Logs to console (dev) and files (production).
 */
const { createLogger, format, transports } = require('winston');
const path = require('path');
const config = require('../config');

const { combine, timestamp, colorize, printf, json, errors } = format;

// ── Console format (development) ─────────────────────────────────────────────
const devFormat = combine(
  colorize({ all: true }),
  timestamp({ format: 'HH:mm:ss' }),
  errors({ stack: true }),
  printf(({ level, message, timestamp, stack }) => {
    return `${timestamp} ${level}: ${stack || message}`;
  })
);

// ── JSON format (production) ──────────────────────────────────────────────────
const prodFormat = combine(
  timestamp(),
  errors({ stack: true }),
  json()
);

const logger = createLogger({
  level: config.logLevel,
  format: config.isDev ? devFormat : prodFormat,
  transports: [
    new transports.Console(),
    // Uncomment to log to files in production:
    // new transports.File({ filename: 'logs/error.log', level: 'error' }),
    // new transports.File({ filename: 'logs/combined.log' }),
  ],
});

module.exports = logger;
