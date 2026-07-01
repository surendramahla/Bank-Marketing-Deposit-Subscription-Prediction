/**
 * src/middleware/validate.middleware.js
 * --------------------------------------
 * express-validator helper for route validation.
 * Extracts validation errors and returns a consistent 422 response.
 *
 * Usage:
 *   const { body } = require('express-validator');
 *   router.post('/login',
 *     body('email').isEmail(),
 *     body('password').isLength({ min: 6 }),
 *     validate,
 *     loginHandler
 *   );
 */
const { validationResult } = require('express-validator');

const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(422).json({
      success: false,
      error: 'Validation failed',
      details: errors.array().map((e) => ({
        field: e.path,
        message: e.msg,
        value: e.value,
      })),
    });
  }
  next();
};

module.exports = { validate };
