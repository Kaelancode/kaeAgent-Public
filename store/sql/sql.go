package sql

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/store"
	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	_ "modernc.org/sqlite"
)

type queryExecutor interface {
	Exec(query string, args ...any) (sql.Result, error)
	Query(query string, args ...any) (*sql.Rows, error)
	QueryRow(query string, args ...any) *sql.Row
}

type contextExecutor interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

type Config struct {
	Driver      string
	DSN         string
	TablePrefix string
}

type SQLStore struct {
	Messages *SQLConversationStore
	Session  *SQLSessionStore
	db       *sql.DB
	dialect  string
	prefix   string
	locksMu  sync.Mutex
	locks    map[string]*sync.Mutex
}

const maxMessageInsertRows = 100

var (
	_ store.ConversationStore = (*SQLConversationStore)(nil)
	_ store.SessionStore      = (*SQLSessionStore)(nil)
)

func NewStore(cfg Config) (*SQLStore, error) {
	driverName := cfg.Driver
	dsn := cfg.DSN
	dialect := driverName
	if dialect == "pgx" {
		dialect = "postgres"
	}
	if dialect == "sqlite" {
		dsn = withSQLitePragmas(dsn, []string{
			"busy_timeout(5000)",
			"foreign_keys(1)",
		})
	}

	db, err := sql.Open(driverName, dsn)
	if err != nil {
		return nil, fmt.Errorf("sql: open: %w", err)
	}

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("sql: ping: %w", err)
	}

	s := &SQLStore{
		db:      db,
		dialect: dialect,
		prefix:  cfg.TablePrefix,
		locks:   make(map[string]*sync.Mutex),
	}

	messageStore := &SQLConversationStore{store: s}
	sessionStore := &SQLSessionStore{store: s}

	s.Messages = messageStore
	s.Session = sessionStore

	var migrateFn func(queryExecutor) error
	switch dialect {
	case "postgres":
		migrateFn = migratePostgreSQL
	case "sqlite":
		migrateFn = migrateSQLite
	default:
		db.Close()
		return nil, fmt.Errorf("sql: unsupported dialect %q (use \"postgres\" or \"sqlite\")", dialect)
	}

	if err := migrateFn(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("sql: migrate: %w", err)
	}

	return s, nil
}

func (s *SQLStore) Close() error {
	return s.db.Close()
}

func withSQLitePragmas(dsn string, pragmas []string) string {
	if len(pragmas) == 0 {
		return dsn
	}

	base := dsn
	var rawQuery string
	if idx := strings.IndexRune(dsn, '?'); idx >= 0 {
		base = dsn[:idx]
		rawQuery = dsn[idx+1:]
	}

	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		values = url.Values{}
	}

	existing := make(map[string]struct{}, len(values["_pragma"]))
	for _, v := range values["_pragma"] {
		existing[v] = struct{}{}
	}
	for _, pragma := range pragmas {
		if _, ok := existing[pragma]; ok {
			continue
		}
		values.Add("_pragma", pragma)
	}

	encoded := values.Encode()
	if encoded == "" {
		return base
	}
	return base + "?" + encoded
}

func (s *SQLStore) placeholder(n int) string {
	if s.dialect == "postgres" {
		return fmt.Sprintf("$%d", n)
	}
	return "?"
}

func (s *SQLStore) nowExpr() string {
	if s.dialect == "postgres" {
		return "NOW()"
	}
	return "CURRENT_TIMESTAMP"
}

func (s *SQLStore) tableName(name string) string {
	if s.prefix != "" {
		return s.prefix + name
	}
	return name
}

func (s *SQLStore) ensureSessionStub(ctx context.Context, sessionID string) error {
	return s.ensureSessionStubExec(ctx, s.db, sessionID)
}

func autoStubUserID(sessionID string) string {
	return "auto-user:" + sessionID
}

func autoStubAgentID(sessionID string) string {
	return "auto-agent:" + sessionID
}

func (s *SQLStore) ensureSessionStubExec(ctx context.Context, exec contextExecutor, sessionID string) error {
	userID := autoStubUserID(sessionID)
	agentID := autoStubAgentID(sessionID)

	userEnsure := fmt.Sprintf(
		"INSERT INTO %s (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
		s.tableName("users"), s.placeholder(1), s.nowExpr(),
	)
	if _, err := exec.ExecContext(ctx, userEnsure, userID); err != nil {
		return fmt.Errorf("sql: ensure user stub: %w", err)
	}

	agentEnsure := fmt.Sprintf(
		"INSERT INTO %s (id, user_id, model, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
		s.tableName("agents"),
		s.placeholder(1), s.placeholder(2), s.placeholder(3),
		s.nowExpr(), s.nowExpr(),
	)
	if _, err := exec.ExecContext(ctx, agentEnsure, agentID, userID, ""); err != nil {
		return fmt.Errorf("sql: ensure agent stub: %w", err)
	}

	sessionEnsure := fmt.Sprintf(
		"INSERT INTO %s (id, agent_id, user_id, trim_strategy, max_history, token_budget, config, budget, metadata, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
		s.tableName("sessions"),
		s.placeholder(1), s.placeholder(2), s.placeholder(3),
		s.placeholder(4), s.placeholder(5), s.placeholder(6), s.placeholder(7),
		s.placeholder(8), s.placeholder(9),
		s.nowExpr(), s.nowExpr(),
	)
	if _, err := exec.ExecContext(ctx, sessionEnsure,
		sessionID, agentID, userID,
		"sliding_window", 50, 128000,
		"{}", "{}", "{}",
	); err != nil {
		return fmt.Errorf("sql: ensure session stub: %w", err)
	}

	return nil
}

type SQLConversationStore struct {
	store *SQLStore
}

func (c *SQLConversationStore) Save(ctx context.Context, convID string, messages []llm.Message) error {
	unlock := c.store.lockConversation(convID)
	defer unlock()

	return c.withBusyRetry(ctx, func() error {
		return c.saveTx(ctx, convID, messages)
	})
}

func (c *SQLConversationStore) saveTx(ctx context.Context, convID string, messages []llm.Message) error {
	tx, err := c.store.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sql: begin tx: %w", err)
	}
	defer tx.Rollback()

	if err := c.ensureLockedSession(ctx, tx, convID); err != nil {
		return err
	}

	_, err = tx.ExecContext(ctx,
		fmt.Sprintf("DELETE FROM %s WHERE session_id = %s",
			c.store.tableName("messages"), c.store.placeholder(1)),
		convID,
	)
	if err != nil {
		return fmt.Errorf("sql: delete existing messages: %w", err)
	}
	if err := c.insertMessagesTx(ctx, tx, convID, 0, messages); err != nil {
		return err
	}

	return tx.Commit()
}

func (c *SQLConversationStore) Load(ctx context.Context, convID string) ([]llm.Message, error) {
	query := fmt.Sprintf(
		"SELECT role, content, name, tool_call_id, tool_calls FROM %s WHERE session_id = %s ORDER BY seq",
		c.store.tableName("messages"), c.store.placeholder(1),
	)
	rows, err := c.store.db.QueryContext(ctx, query, convID)
	if err != nil {
		return nil, fmt.Errorf("sql: load messages: %w", err)
	}
	defer rows.Close()

	var messages []llm.Message
	for rows.Next() {
		var msg llm.Message
		var toolCallsJSON string
		if err := rows.Scan(&msg.Role, &msg.Content, &msg.Name, &msg.ToolCallID, &toolCallsJSON); err != nil {
			return nil, fmt.Errorf("sql: scan message: %w", err)
		}
		if toolCallsJSON != "" && toolCallsJSON != "[]" && toolCallsJSON != "null" {
			if err := json.Unmarshal([]byte(toolCallsJSON), &msg.ToolCalls); err != nil {
				return nil, fmt.Errorf("sql: unmarshal tool_calls: %w", err)
			}
		}
		messages = append(messages, msg)
	}
	return messages, rows.Err()
}

func (c *SQLConversationStore) Append(ctx context.Context, convID string, messages []llm.Message) error {
	unlock := c.store.lockConversation(convID)
	defer unlock()

	return c.withBusyRetry(ctx, func() error {
		return c.appendTx(ctx, convID, messages)
	})
}

func (c *SQLConversationStore) appendTx(ctx context.Context, convID string, messages []llm.Message) error {
	tx, err := c.store.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sql: begin tx: %w", err)
	}
	defer tx.Rollback()

	if err := c.ensureLockedSession(ctx, tx, convID); err != nil {
		return err
	}

	var maxSeq int
	row := tx.QueryRowContext(ctx,
		fmt.Sprintf("SELECT COALESCE(MAX(seq), -1) FROM %s WHERE session_id = %s",
			c.store.tableName("messages"), c.store.placeholder(1)),
		convID,
	)
	if err := row.Scan(&maxSeq); err != nil {
		return fmt.Errorf("sql: get max seq: %w", err)
	}
	nextSeq := maxSeq + 1

	if err := c.insertMessagesTx(ctx, tx, convID, nextSeq, messages); err != nil {
		return err
	}

	return tx.Commit()
}

func (c *SQLConversationStore) Delete(ctx context.Context, convID string) error {
	unlock := c.store.lockConversation(convID)
	defer unlock()

	_, err := c.store.db.ExecContext(ctx,
		fmt.Sprintf("DELETE FROM %s WHERE session_id = %s",
			c.store.tableName("messages"), c.store.placeholder(1)),
		convID,
	)
	if err != nil {
		return fmt.Errorf("sql: delete messages: %w", err)
	}
	return nil
}

func (c *SQLConversationStore) insertMessages(ctx context.Context, convID string, startSeq int, messages []llm.Message) error {
	tx, err := c.store.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sql: begin tx: %w", err)
	}
	defer tx.Rollback()

	if err := c.insertMessagesTx(ctx, tx, convID, startSeq, messages); err != nil {
		return err
	}

	return tx.Commit()
}

func (c *SQLConversationStore) insertMessagesTx(ctx context.Context, tx *sql.Tx, convID string, startSeq int, messages []llm.Message) error {
	if len(messages) == 0 {
		return nil
	}

	for start := 0; start < len(messages); start += maxMessageInsertRows {
		end := start + maxMessageInsertRows
		if end > len(messages) {
			end = len(messages)
		}

		stmt, args, err := c.buildInsertMessagesStatement(convID, startSeq+start, messages[start:end], start)
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, stmt, args...); err != nil {
			return fmt.Errorf("sql: insert messages %d-%d: %w", start, end-1, err)
		}
	}

	return nil
}

func (c *SQLConversationStore) buildInsertMessagesStatement(convID string, startSeq int, messages []llm.Message, offset int) (string, []any, error) {
	args := make([]any, 0, len(messages)*8)
	values := make([]string, 0, len(messages))

	for i, msg := range messages {
		var toolCallsJSON []byte
		if len(msg.ToolCalls) > 0 {
			var err error
			toolCallsJSON, err = json.Marshal(msg.ToolCalls)
			if err != nil {
				return "", nil, fmt.Errorf("sql: marshal tool_calls for message %d: %w", offset+i, err)
			}
		} else {
			toolCallsJSON = []byte("[]")
		}

		base := len(args) + 1
		values = append(values, fmt.Sprintf("(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
			c.store.placeholder(base),
			c.store.placeholder(base+1),
			c.store.placeholder(base+2),
			c.store.placeholder(base+3),
			c.store.placeholder(base+4),
			c.store.placeholder(base+5),
			c.store.placeholder(base+6),
			c.store.placeholder(base+7),
			c.store.nowExpr(),
		))
		args = append(args,
			uuid.New().String(), convID, startSeq+i,
			msg.Role, msg.Content, msg.Name, msg.ToolCallID,
			string(toolCallsJSON),
		)
	}

	stmt := fmt.Sprintf(
		"INSERT INTO %s (id, session_id, seq, role, content, name, tool_call_id, tool_calls, created_at) VALUES %s",
		c.store.tableName("messages"),
		strings.Join(values, ", "),
	)
	return stmt, args, nil
}

func (c *SQLConversationStore) lockSession(ctx context.Context, tx *sql.Tx, convID string) error {
	lockSQL := fmt.Sprintf(
		"UPDATE %s SET updated_at = updated_at WHERE id = %s",
		c.store.tableName("sessions"), c.store.placeholder(1),
	)
	if _, err := tx.ExecContext(ctx, lockSQL, convID); err != nil {
		return fmt.Errorf("sql: lock session %q: %w", convID, err)
	}
	return nil
}

func (c *SQLConversationStore) ensureLockedSession(ctx context.Context, tx *sql.Tx, convID string) error {
	exists, err := c.sessionExists(ctx, tx, convID)
	if err != nil {
		return err
	}
	if !exists {
		if err := c.store.ensureSessionStubExec(ctx, tx, convID); err != nil {
			return err
		}
	}
	if err := c.lockSession(ctx, tx, convID); err != nil {
		return err
	}
	return nil
}

func (c *SQLConversationStore) sessionExists(ctx context.Context, tx *sql.Tx, convID string) (bool, error) {
	query := fmt.Sprintf(
		"SELECT 1 FROM %s WHERE id = %s",
		c.store.tableName("sessions"), c.store.placeholder(1),
	)
	var exists int
	if err := tx.QueryRowContext(ctx, query, convID).Scan(&exists); err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, fmt.Errorf("sql: check session %q: %w", convID, err)
	}
	return true, nil
}

func (c *SQLConversationStore) withBusyRetry(ctx context.Context, fn func() error) error {
	if c.store.dialect != "sqlite" {
		return fn()
	}

	var lastErr error
	for attempt := 0; attempt < 20; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return fmt.Errorf("sql: retry cancelled: %w", ctx.Err())
			case <-time.After(time.Duration(attempt) * 50 * time.Millisecond):
			}
		}

		err := fn()
		if err == nil {
			return nil
		}
		if !isSQLiteBusyError(err) {
			return err
		}
		lastErr = err
	}

	return lastErr
}

func isSQLiteBusyError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "sqlite_busy") || strings.Contains(msg, "database is locked")
}

func (s *SQLStore) lockConversation(convID string) func() {
	if s.dialect != "sqlite" {
		return func() {}
	}

	s.locksMu.Lock()
	mu, ok := s.locks[convID]
	if !ok {
		mu = &sync.Mutex{}
		s.locks[convID] = mu
	}
	s.locksMu.Unlock()

	mu.Lock()
	return mu.Unlock
}

type SQLSessionStore struct {
	store *SQLStore
}

func (s *SQLSessionStore) SaveSession(ctx context.Context, data *store.SessionData) error {
	tx, err := s.store.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sql: begin tx: %w", err)
	}
	defer tx.Rollback()

	userEnsure := fmt.Sprintf(
		"INSERT INTO %s (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
		s.store.tableName("users"), s.store.placeholder(1), s.store.nowExpr(),
	)
	if _, err := tx.ExecContext(ctx, userEnsure, data.UserID); err != nil {
		return fmt.Errorf("sql: upsert user: %w", err)
	}

	agentEnsure := fmt.Sprintf(
		"INSERT INTO %s (id, user_id, model, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
		s.store.tableName("agents"),
		s.store.placeholder(1), s.store.placeholder(2), s.store.placeholder(3),
		s.store.nowExpr(), s.store.nowExpr(),
	)
	if _, err := tx.ExecContext(ctx, agentEnsure, data.AgentID, data.UserID, ""); err != nil {
		return fmt.Errorf("sql: upsert agent: %w", err)
	}

	metadataJSON, err := json.Marshal(data.Metadata)
	if err != nil {
		return fmt.Errorf("sql: marshal metadata: %w", err)
	}
	if metadataJSON == nil {
		metadataJSON = []byte("{}")
	}
	budgetJSON := data.Budget
	if budgetJSON == nil {
		budgetJSON = []byte("{}")
	}
	configJSON := data.Config
	if len(configJSON) == 0 {
		configJSON = []byte("{}")
	}

	var configMap map[string]any
	if err := json.Unmarshal(configJSON, &configMap); err != nil {
		return fmt.Errorf("sql: unmarshal config: %w", err)
	}

	trimStrategy := ""
	maxHistory := 50
	tokenBudget := 128000
	if configMap != nil {
		if ts, ok := configMap["trim_strategy"].(string); ok {
			trimStrategy = ts
		}
		if mh, ok := configMap["max_history"].(float64); ok {
			maxHistory = int(mh)
		}
		if tb, ok := configMap["token_budget"].(float64); ok {
			tokenBudget = int(tb)
		}
	}

	upsertSQL := fmt.Sprintf(
		`INSERT INTO %s (id, agent_id, user_id, trim_strategy, max_history, token_budget, config, budget, metadata, created_at, updated_at)
		 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
		 ON CONFLICT (id) DO UPDATE SET
				agent_id = EXCLUDED.agent_id,
				user_id = EXCLUDED.user_id,
				trim_strategy = EXCLUDED.trim_strategy,
				max_history = EXCLUDED.max_history,
				token_budget = EXCLUDED.token_budget,
				config = EXCLUDED.config,
				budget = EXCLUDED.budget,
				metadata = EXCLUDED.metadata,
				updated_at = EXCLUDED.updated_at`,
		s.store.tableName("sessions"),
		s.store.placeholder(1), s.store.placeholder(2), s.store.placeholder(3),
		s.store.placeholder(4), s.store.placeholder(5), s.store.placeholder(6),
		s.store.placeholder(7), s.store.placeholder(8), s.store.placeholder(9),
		s.store.nowExpr(), s.store.nowExpr(),
	)

	if _, err := tx.ExecContext(ctx, upsertSQL,
		data.ID, data.AgentID, data.UserID,
		trimStrategy, maxHistory, tokenBudget,
		string(configJSON), string(budgetJSON), string(metadataJSON),
	); err != nil {
		return fmt.Errorf("sql: upsert session: %w", err)
	}

	return tx.Commit()
}

func (s *SQLSessionStore) LoadSession(ctx context.Context, sessionID string) (*store.SessionData, error) {
	query := fmt.Sprintf(
		"SELECT id, agent_id, user_id, trim_strategy, max_history, token_budget, config, budget, metadata, created_at, updated_at FROM %s WHERE id = %s",
		s.store.tableName("sessions"), s.store.placeholder(1),
	)

	var data store.SessionData
	var configTrimStrategy string
	var configMaxHistory int
	var configTokenBudget int
	var configJSON, budgetJSON, metadataJSON string
	var createdAt, updatedAt time.Time

	if err := s.store.db.QueryRowContext(ctx, query, sessionID).Scan(
		&data.ID, &data.AgentID, &data.UserID,
		&configTrimStrategy, &configMaxHistory, &configTokenBudget,
		&configJSON, &budgetJSON, &metadataJSON,
		&createdAt, &updatedAt,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("sql: load session: %w", err)
	}

	data.Config = json.RawMessage(configJSON)
	if len(data.Config) == 0 {
		config := map[string]any{
			"trim_strategy": configTrimStrategy,
			"max_history":   configMaxHistory,
			"token_budget":  configTokenBudget,
		}
		var err error
		data.Config, err = json.Marshal(config)
		if err != nil {
			return nil, fmt.Errorf("sql: marshal default config: %w", err)
		}
	}
	data.Budget = json.RawMessage(budgetJSON)
	if err := json.Unmarshal([]byte(metadataJSON), &data.Metadata); err != nil {
		data.Metadata = make(map[string]string)
	}

	return &data, nil
}

func (s *SQLSessionStore) DeleteSession(ctx context.Context, sessionID string) error {
	_, err := s.store.db.ExecContext(ctx,
		fmt.Sprintf("DELETE FROM %s WHERE id = %s",
			s.store.tableName("sessions"), s.store.placeholder(1)),
		sessionID,
	)
	if err != nil {
		return fmt.Errorf("sql: delete session: %w", err)
	}
	return nil
}

func (s *SQLSessionStore) ListSessions(ctx context.Context, userID, agentID string) ([]store.SessionEntry, error) {
	var query string
	var args []any

	if userID != "" && agentID != "" {
		query = fmt.Sprintf(
			"SELECT s.id, s.user_id, s.agent_id, s.config, s.created_at, s.updated_at FROM %s s WHERE s.user_id = %s AND s.agent_id = %s ORDER BY s.updated_at DESC",
			s.store.tableName("sessions"),
			s.store.placeholder(1), s.store.placeholder(2),
		)
		args = []any{userID, agentID}
	} else if userID != "" {
		query = fmt.Sprintf(
			"SELECT s.id, s.user_id, s.agent_id, s.config, s.created_at, s.updated_at FROM %s s WHERE s.user_id = %s ORDER BY s.updated_at DESC",
			s.store.tableName("sessions"),
			s.store.placeholder(1),
		)
		args = []any{userID}
	} else if agentID != "" {
		query = fmt.Sprintf(
			"SELECT s.id, s.user_id, s.agent_id, s.config, s.created_at, s.updated_at FROM %s s WHERE s.agent_id = %s ORDER BY s.updated_at DESC",
			s.store.tableName("sessions"),
			s.store.placeholder(1),
		)
		args = []any{agentID}
	} else {
		query = fmt.Sprintf(
			"SELECT s.id, s.user_id, s.agent_id, s.config, s.created_at, s.updated_at FROM %s s ORDER BY s.updated_at DESC",
			s.store.tableName("sessions"),
		)
	}

	rows, err := s.store.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sql: list sessions: %w", err)
	}
	defer rows.Close()

	var entries []store.SessionEntry
	for rows.Next() {
		var e store.SessionEntry
		var configJSON string
		if err := rows.Scan(&e.ID, &e.UserID, &e.AgentID, &configJSON, &e.CreatedAt, &e.UpdatedAt); err != nil {
			return nil, fmt.Errorf("sql: scan session entry: %w", err)
		}
		if configJSON != "" && configJSON != "{}" {
			var config map[string]any
			if err := json.Unmarshal([]byte(configJSON), &config); err == nil {
				if model, ok := config["model"].(string); ok {
					e.Model = model
				}
			}
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}
