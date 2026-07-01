/**
 * tests/server.test.js
 * --------------------
 * Integration tests for the BankAI Pro Express backend.
 *
 * Run: npm test
 *
 * These tests use supertest to make real HTTP requests.
 * They test public endpoints and protected endpoints with JWT.
 */
const request = require('supertest');
const app = require('../src/server');

// ── Health & Root ─────────────────────────────────────────────────────────────
describe('Health & Root', () => {
  it('GET /health returns 200 or 503', async () => {
    const res = await request(app).get('/health');
    expect([200, 503]).toContain(res.status);
    expect(res.body).toHaveProperty('service', 'BankAI Pro Backend');
  });

  it('GET /api returns service info', async () => {
    const res = await request(app).get('/api');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('service');
    expect(res.body).toHaveProperty('endpoints');
  });

  it('GET /unknown returns 404', async () => {
    const res = await request(app).get('/unknown-route-xyz');
    expect(res.status).toBe(404);
    expect(res.body).toHaveProperty('success', false);
  });
});

// ── Auth — Validation ─────────────────────────────────────────────────────────
describe('Auth — Input Validation', () => {
  it('POST /api/auth/login with missing email returns 422', async () => {
    const res = await request(app)
      .post('/api/auth/login')
      .send({ password: 'test123' });
    expect(res.status).toBe(422);
    expect(res.body.success).toBe(false);
    expect(res.body.details).toBeDefined();
  });

  it('POST /api/auth/login with invalid email returns 422', async () => {
    const res = await request(app)
      .post('/api/auth/login')
      .send({ email: 'not-an-email', password: 'test123' });
    expect(res.status).toBe(422);
  });

  it('POST /api/auth/register with short password returns 422', async () => {
    const res = await request(app)
      .post('/api/auth/register')
      .send({ username: 'test', email: 'test@test.com', password: '123' });
    expect(res.status).toBe(422);
  });
});

// ── Protected Routes — Unauthenticated ────────────────────────────────────────
describe('Protected Routes — No Auth', () => {
  const protectedRoutes = [
    { method: 'get', path: '/api/customers' },
    { method: 'get', path: '/api/predictions' },
    { method: 'get', path: '/api/analytics/dashboard' },
    { method: 'get', path: '/api/chat/history' },
    { method: 'get', path: '/api/campaigns' },
  ];

  protectedRoutes.forEach(({ method, path }) => {
    it(`${method.toUpperCase()} ${path} returns 401 without token`, async () => {
      const res = await request(app)[method](path);
      expect(res.status).toBe(401);
      expect(res.body.success).toBe(false);
    });
  });
});

// ── Protected Routes — Invalid Token ─────────────────────────────────────────
describe('Protected Routes — Invalid Token', () => {
  it('Returns 401 for malformed JWT', async () => {
    const res = await request(app)
      .get('/api/customers')
      .set('Authorization', 'Bearer invalid.token.here');
    expect(res.status).toBe(401);
  });
});

// ── Prediction Validation ─────────────────────────────────────────────────────
describe('Prediction — Validation (no auth check)', () => {
  it('POST /api/predictions/single without auth returns 401', async () => {
    const res = await request(app)
      .post('/api/predictions/single')
      .send({ age: 42, job: 'management' });
    expect(res.status).toBe(401);
  });
});

// Close any open handles after tests
afterAll(async () => {
  // Give time for any pending operations to complete
  await new Promise((resolve) => setTimeout(resolve, 500));
});
