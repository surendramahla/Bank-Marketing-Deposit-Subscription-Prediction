/**
 * src/middleware/upload.middleware.js
 * ------------------------------------
 * Multer-based file upload middleware for CSV bulk prediction uploads.
 *
 * Validates:
 * - File type (CSV only)
 * - File size (max 32MB by default)
 * - Filename sanitization
 *
 * Usage:
 *   router.post('/bulk', uploadCSV.single('file'), handler);
 */
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const config = require('../config');

// ── Ensure upload directory exists ────────────────────────────────────────────
const uploadDir = config.upload.dir;
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// ── Storage engine ────────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Sanitize: uuid prefix + original extension to avoid collisions
    const ext = path.extname(file.originalname).toLowerCase();
    const safeName = `${uuidv4()}${ext}`;
    req.uploadedFilename = safeName; // Store for use in route handler
    cb(null, safeName);
  },
});

// ── File type filter ──────────────────────────────────────────────────────────
const csvFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase();
  const mime = file.mimetype;

  const allowedExts = ['.csv'];
  const allowedMimes = ['text/csv', 'application/csv', 'text/plain', 'application/vnd.ms-excel'];

  if (allowedExts.includes(ext) || allowedMimes.includes(mime)) {
    cb(null, true);
  } else {
    cb(
      new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'Only CSV files are allowed'),
      false
    );
  }
};

// ── Multer instance ───────────────────────────────────────────────────────────
const uploadCSV = multer({
  storage,
  fileFilter: csvFilter,
  limits: {
    fileSize: config.upload.maxSizeMb * 1024 * 1024, // Convert MB to bytes
    files: 1, // Only one file at a time
  },
});

// ── Error handler middleware ──────────────────────────────────────────────────
/**
 * Wraps multer errors into a consistent JSON response.
 * Use after uploadCSV middleware:
 *   router.post('/upload', uploadCSV.single('file'), handleUploadError, handler);
 */
const handleUploadError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({
        success: false,
        error: `File too large. Maximum size is ${config.upload.maxSizeMb}MB`,
      });
    }
    return res.status(400).json({
      success: false,
      error: err.message || 'File upload error',
    });
  }

  if (err) {
    return res.status(400).json({
      success: false,
      error: err.message || 'Unknown upload error',
    });
  }

  next();
};

module.exports = { uploadCSV, handleUploadError };
