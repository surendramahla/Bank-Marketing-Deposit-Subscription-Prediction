/**
 * src/db/init.js
 * --------------
 * Initializes the PostgreSQL database by running schema.sql.
 * Run once: node src/db/init.js
 *
 * Reads DATABASE_URL or individual DB_* env vars.
 * Safe to re-run — uses IF NOT EXISTS and ON CONFLICT.
 */
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
require('dotenv').config({ path: path.join(__dirname, '../../.env') });

const config = require('../config');

async function initDatabase() {
  const poolConfig = config.db.url
    ? { connectionString: config.db.url }
    : {
        host: config.db.host,
        port: config.db.port,
        database: config.db.name,
        user: config.db.user,
        password: config.db.password,
      };

  const pool = new Pool(poolConfig);

  console.log('\n🔵 BankAI Pro — Database Initialization');
  console.log('=========================================');
  console.log(`Host: ${config.db.host}:${config.db.port}`);
  console.log(`Database: ${config.db.name}`);

  try {
    const schemaPath = path.join(__dirname, 'schema.sql');
    const schema = fs.readFileSync(schemaPath, 'utf8');

    console.log('\n⏳ Running schema.sql...');
    await pool.query(schema);

    console.log('✅ Database initialized successfully!');
    console.log('\nTables created:');
    const tables = await pool.query(
      "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    );
    tables.rows.forEach((r) => console.log(`  ✓ ${r.tablename}`));

    console.log('\nDefault credentials:');
    console.log('  Username: admin');
    console.log('  Password: admin123');
    console.log('  ⚠️  Change these in production!\n');

  } catch (err) {
    console.error('\n❌ Database initialization failed:', err.message);
    if (err.code === 'ECONNREFUSED') {
      console.error('\nIs PostgreSQL running? Start it with:');
      console.error('  Windows: Start PostgreSQL service in Services panel');
      console.error('  Or use Docker: docker run -e POSTGRES_PASSWORD=bankai_pass -p 5432:5432 postgres:16');
    }
    process.exit(1);
  } finally {
    await pool.end();
  }
}

initDatabase();
