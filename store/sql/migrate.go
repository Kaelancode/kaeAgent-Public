package sql

import "strings"

const currentSchemaVersion = 1

func migratePostgreSQL(db queryExecutor) error {
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS schema_migrations (
		version    INTEGER PRIMARY KEY,
		applied_at TIMESTAMP NOT NULL DEFAULT NOW()
	)`); err != nil {
		return err
	}

	stmts := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id         TEXT PRIMARY KEY,
			created_at TIMESTAMP NOT NULL DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS agents (
			id            TEXT PRIMARY KEY,
			user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			name          TEXT NOT NULL DEFAULT '',
			model         TEXT NOT NULL DEFAULT '',
			system_prompt  TEXT NOT NULL DEFAULT '',
			max_tokens    INTEGER NOT NULL DEFAULT 4096,
			temperature   REAL NOT NULL DEFAULT 0.7,
			created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
			updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS idx_agents_user ON agents(user_id)`,
		`CREATE TABLE IF NOT EXISTS sessions (
			id            TEXT PRIMARY KEY,
			agent_id      TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
			user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			trim_strategy TEXT NOT NULL DEFAULT 'sliding_window',
			max_history   INTEGER NOT NULL DEFAULT 50,
			token_budget  INTEGER NOT NULL DEFAULT 128000,
			budget        JSONB NOT NULL DEFAULT '{}',
			metadata      JSONB NOT NULL DEFAULT '{}',
			created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
			updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)`,
		`CREATE TABLE IF NOT EXISTS messages (
			id           TEXT PRIMARY KEY,
			session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			seq          INTEGER NOT NULL,
			role         TEXT NOT NULL,
			content      TEXT NOT NULL DEFAULT '',
			name         TEXT NOT NULL DEFAULT '',
			tool_call_id TEXT NOT NULL DEFAULT '',
			tool_calls   JSONB NOT NULL DEFAULT '[]',
			created_at   TIMESTAMP NOT NULL DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_session_seq_unique ON messages(session_id, seq)`,
		`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS config JSONB NOT NULL DEFAULT '{}'`,
	}
	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return err
		}
	}
	if _, err := db.Exec(`INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT (version) DO NOTHING`, currentSchemaVersion); err != nil {
		return err
	}
	return nil
}

func migrateSQLite(db queryExecutor) error {
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS schema_migrations (
		version    INTEGER PRIMARY KEY,
		applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	)`); err != nil {
		return err
	}

	stmts := []string{
		`PRAGMA foreign_keys = ON`,
		`CREATE TABLE IF NOT EXISTS users (
			id         TEXT PRIMARY KEY,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS agents (
			id             TEXT PRIMARY KEY,
			user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			name           TEXT NOT NULL DEFAULT '',
			model          TEXT NOT NULL DEFAULT '',
			system_prompt  TEXT NOT NULL DEFAULT '',
			max_tokens     INTEGER NOT NULL DEFAULT 4096,
			temperature    REAL NOT NULL DEFAULT 0.7,
			created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_agents_user ON agents(user_id)`,
		`CREATE TABLE IF NOT EXISTS sessions (
			id            TEXT PRIMARY KEY,
			agent_id      TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
			user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			trim_strategy TEXT NOT NULL DEFAULT 'sliding_window',
			max_history   INTEGER NOT NULL DEFAULT 50,
			token_budget  INTEGER NOT NULL DEFAULT 128000,
			budget        TEXT NOT NULL DEFAULT '{}',
			metadata      TEXT NOT NULL DEFAULT '{}',
			created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)`,
		`CREATE TABLE IF NOT EXISTS messages (
			id           TEXT PRIMARY KEY,
			session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			seq          INTEGER NOT NULL,
			role         TEXT NOT NULL,
			content      TEXT NOT NULL DEFAULT '',
			name         TEXT NOT NULL DEFAULT '',
			tool_call_id TEXT NOT NULL DEFAULT '',
			tool_calls   TEXT NOT NULL DEFAULT '[]',
			created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_session_seq_unique ON messages(session_id, seq)`,
	}
	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return err
		}
	}
	if _, err := db.Exec(`ALTER TABLE sessions ADD COLUMN config TEXT NOT NULL DEFAULT '{}'`); err != nil {
		if !strings.Contains(strings.ToLower(err.Error()), "duplicate column name") {
			return err
		}
	}
	if _, err := db.Exec(`INSERT OR IGNORE INTO schema_migrations (version) VALUES (?)`, currentSchemaVersion); err != nil {
		return err
	}
	return nil
}
