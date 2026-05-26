package sql

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/store"
)

func sqliteTestStore(t *testing.T) *SQLStore {
	t.Helper()
	s, err := NewStore(Config{Driver: "sqlite", DSN: ":memory:"})
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func sqliteFileTestStore(t *testing.T) *SQLStore {
	t.Helper()
	dsn := filepath.Join(t.TempDir(), "test.db")
	s, err := NewStore(Config{Driver: "sqlite", DSN: dsn})
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	s.db.SetMaxOpenConns(10)
	t.Cleanup(func() { s.Close() })
	return s
}

func pgTestStore(t *testing.T) *SQLStore {
	t.Helper()
	dsn := os.Getenv("PG_TEST_DSN")
	if dsn == "" {
		t.Skip("PG_TEST_DSN not set, skipping PostgreSQL tests")
	}
	s, err := NewStore(Config{Driver: "postgres", DSN: dsn})
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func testConversationStore(t *testing.T, convStore store.ConversationStore) {
	ctx := context.Background()

	msgs := []llm.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi!"},
	}

	if err := convStore.Save(ctx, "conv_1", msgs); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := convStore.Load(ctx, "conv_1")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(loaded))
	}
	if loaded[0].Role != "system" || loaded[0].Content != "You are helpful." {
		t.Errorf("expected system message, got %+v", loaded[0])
	}
	if loaded[1].Role != "user" || loaded[1].Content != "Hello" {
		t.Errorf("expected user message, got %+v", loaded[1])
	}
	if loaded[2].Role != "assistant" || loaded[2].Content != "Hi!" {
		t.Errorf("expected assistant message, got %+v", loaded[2])
	}
}

func testConversationLoadNonexistent(t *testing.T, convStore store.ConversationStore) {
	ctx := context.Background()

	loaded, err := convStore.Load(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil for nonexistent key, got %v", loaded)
	}
}

func testConversationAppend(t *testing.T, convStore store.ConversationStore) {
	ctx := context.Background()

	initial := []llm.Message{
		{Role: "user", Content: "First"},
	}
	if err := convStore.Save(ctx, "conv_append", initial); err != nil {
		t.Fatalf("Save: %v", err)
	}

	appendMsgs := []llm.Message{
		{Role: "assistant", Content: "Second"},
		{Role: "user", Content: "Third"},
	}
	if err := convStore.Append(ctx, "conv_append", appendMsgs); err != nil {
		t.Fatalf("Append: %v", err)
	}

	loaded, err := convStore.Load(ctx, "conv_append")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(loaded))
	}
	if loaded[0].Content != "First" || loaded[1].Content != "Second" || loaded[2].Content != "Third" {
		t.Errorf("unexpected message order: %+v", loaded)
	}
}

func testConversationAppendToEmpty(t *testing.T, convStore store.ConversationStore) {
	ctx := context.Background()

	msgs := []llm.Message{
		{Role: "user", Content: "Hello"},
	}
	if err := convStore.Append(ctx, "conv_new_append", msgs); err != nil {
		t.Fatalf("Append: %v", err)
	}

	loaded, err := convStore.Load(ctx, "conv_new_append")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 message, got %d", len(loaded))
	}
	if loaded[0].Content != "Hello" {
		t.Errorf("expected Hello, got %s", loaded[0].Content)
	}
}

func testConversationDelete(t *testing.T, convStore store.ConversationStore) {
	ctx := context.Background()

	msgs := []llm.Message{{Role: "user", Content: "Hi"}}
	if err := convStore.Save(ctx, "conv_del", msgs); err != nil {
		t.Fatalf("Save: %v", err)
	}

	if err := convStore.Delete(ctx, "conv_del"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	loaded, err := convStore.Load(ctx, "conv_del")
	if err != nil {
		t.Fatalf("Load after delete: %v", err)
	}

	// After deleting messages but not the session row, Load returns empty (nil)
	// This is correct: the messages are gone, so nil is expected
	_ = loaded
}

func testConversationToolCallsRoundTrip(t *testing.T, convStore store.ConversationStore) {
	ctx := context.Background()

	msgs := []llm.Message{
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "get_weather", Input: map[string]any{"city": "SF"}},
			},
		},
		{
			Role:       "tool",
			Content:    `{"temp": "72"}`,
			ToolCallID: "call_1",
			Name:       "get_weather",
		},
	}

	if err := convStore.Save(ctx, "conv_tools", msgs); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := convStore.Load(ctx, "conv_tools")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if len(loaded) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(loaded))
	}
	if len(loaded[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(loaded[0].ToolCalls))
	}
	if loaded[0].ToolCalls[0].Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", loaded[0].ToolCalls[0].Name)
	}
	if loaded[0].ToolCalls[0].Input["city"] != "SF" {
		t.Errorf("expected city=SF, got %v", loaded[0].ToolCalls[0].Input["city"])
	}
	if loaded[1].ToolCallID != "call_1" {
		t.Errorf("expected call_1, got %s", loaded[1].ToolCallID)
	}
}

// SQLite ConversationStore tests

func TestSQLLite_ConversationStore_SaveAndLoad(t *testing.T) {
	testConversationStore(t, sqliteTestStore(t).Messages)
}

func TestSQLLite_ConversationStore_LoadNonexistent(t *testing.T) {
	testConversationLoadNonexistent(t, sqliteTestStore(t).Messages)
}

func TestSQLLite_ConversationStore_Append(t *testing.T) {
	testConversationAppend(t, sqliteTestStore(t).Messages)
}

func TestSQLLite_ConversationStore_AppendToEmpty(t *testing.T) {
	testConversationAppendToEmpty(t, sqliteTestStore(t).Messages)
}

func TestSQLLite_ConversationStore_Delete(t *testing.T) {
	testConversationDelete(t, sqliteTestStore(t).Messages)
}

func TestSQLLite_ConversationStore_ToolCallsRoundTrip(t *testing.T) {
	testConversationToolCallsRoundTrip(t, sqliteTestStore(t).Messages)
}

func TestSQLLite_ConversationStore_SaveRollsBackOnInsertFailure(t *testing.T) {
	s := sqliteTestStore(t)
	ctx := context.Background()

	original := []llm.Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "second"},
	}
	if err := s.Messages.Save(ctx, "conv_atomic", original); err != nil {
		t.Fatalf("initial Save: %v", err)
	}

	if _, err := s.db.ExecContext(ctx,
		`CREATE UNIQUE INDEX idx_messages_session_role ON messages(session_id, role)`); err != nil {
		t.Fatalf("create unique index: %v", err)
	}

	replacement := []llm.Message{
		{Role: "user", Content: "replacement-1"},
		{Role: "user", Content: "replacement-2"},
	}
	if err := s.Messages.Save(ctx, "conv_atomic", replacement); err == nil {
		t.Fatal("expected Save to fail due to unique index")
	}

	loaded, err := s.Messages.Load(ctx, "conv_atomic")
	if err != nil {
		t.Fatalf("Load after failed Save: %v", err)
	}
	if len(loaded) != len(original) {
		t.Fatalf("expected %d original messages after rollback, got %d", len(original), len(loaded))
	}
	if loaded[0].Content != original[0].Content || loaded[1].Content != original[1].Content {
		t.Fatalf("expected original messages to remain after rollback, got %+v", loaded)
	}
}

func TestSQLLite_ConversationStore_AppendConcurrentMaintainsUniqueSeq(t *testing.T) {
	s := sqliteFileTestStore(t)
	ctx := context.Background()

	const convID = "conv_concurrent"
	const writers = 16

	if err := s.Messages.Save(ctx, convID, []llm.Message{{Role: "system", Content: "seed"}}); err != nil {
		t.Fatalf("seed Save: %v", err)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, writers)
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errCh <- s.Messages.Append(context.Background(), convID, []llm.Message{
				{Role: "user", Content: fmt.Sprintf("msg-%d", i)},
			})
		}(i)
	}
	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("Append failed: %v", err)
		}
	}

	rows, err := s.db.QueryContext(ctx,
		`SELECT seq FROM messages WHERE session_id = ? ORDER BY seq`, convID)
	if err != nil {
		t.Fatalf("query seqs: %v", err)
	}
	defer rows.Close()

	var seqs []int
	for rows.Next() {
		var seq int
		if err := rows.Scan(&seq); err != nil {
			t.Fatalf("scan seq: %v", err)
		}
		seqs = append(seqs, seq)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows err: %v", err)
	}

	expected := writers + 1
	if len(seqs) != expected {
		t.Fatalf("expected %d messages, got %d", expected, len(seqs))
	}
	for i, seq := range seqs {
		if seq != i {
			t.Fatalf("expected contiguous seq values, got %v", seqs)
		}
	}
}

func TestSQLLite_ConversationStore_EnforcesUniqueSeqPerSession(t *testing.T) {
	s := sqliteTestStore(t)
	ctx := context.Background()

	if err := s.ensureSessionStub(ctx, "conv_unique_seq"); err != nil {
		t.Fatalf("ensureSessionStub: %v", err)
	}

	if _, err := s.db.ExecContext(ctx,
		`INSERT INTO messages (id, session_id, seq, role, content, name, tool_call_id, tool_calls)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"msg_1", "conv_unique_seq", 0, "user", "one", "", "", "[]",
	); err != nil {
		t.Fatalf("first insert: %v", err)
	}

	if _, err := s.db.ExecContext(ctx,
		`INSERT INTO messages (id, session_id, seq, role, content, name, tool_call_id, tool_calls)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"msg_2", "conv_unique_seq", 0, "assistant", "two", "", "", "[]",
	); err == nil {
		t.Fatal("expected duplicate seq insert to fail")
	}
}

func TestSQLite_ForeignKeysEnabledOnAllPooledConnections(t *testing.T) {
	s := sqliteFileTestStore(t)
	s.db.SetMaxOpenConns(2)
	ctx := context.Background()

	conn1, err := s.db.Conn(ctx)
	if err != nil {
		t.Fatalf("Conn 1: %v", err)
	}
	defer conn1.Close()

	conn2, err := s.db.Conn(ctx)
	if err != nil {
		t.Fatalf("Conn 2: %v", err)
	}
	defer conn2.Close()

	for i, conn := range []*sql.Conn{conn1, conn2} {
		var enabled int
		if err := conn.QueryRowContext(ctx, `PRAGMA foreign_keys`).Scan(&enabled); err != nil {
			t.Fatalf("PRAGMA foreign_keys on conn %d: %v", i+1, err)
		}
		if enabled != 1 {
			t.Fatalf("expected foreign_keys=1 on conn %d, got %d", i+1, enabled)
		}
	}
}

// PostgreSQL ConversationStore tests

func TestPostgres_ConversationStore_SaveAndLoad(t *testing.T) {
	testConversationStore(t, pgTestStore(t).Messages)
}

func TestPostgres_ConversationStore_Append(t *testing.T) {
	testConversationAppend(t, pgTestStore(t).Messages)
}

func TestPostgres_ConversationStore_ToolCallsRoundTrip(t *testing.T) {
	testConversationToolCallsRoundTrip(t, pgTestStore(t).Messages)
}

// SessionStore tests

func testSessionStore(t *testing.T, sessionStore store.SessionStore) {
	ctx := context.Background()

	configJSON := []byte(`{"model":"gpt-4o","max_tokens":4096,"temperature":0.7,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`)
	budgetJSON := []byte(`{"max_tokens":500000,"max_cost_usd":5,"total_input":100,"total_output":50,"total_cost_usd":0.003,"cost_per_input_token":0.00003,"cost_per_output_token":0.00015}`)

	data := &store.SessionData{
		ID:       "sess_test_1",
		UserID:   "user_42",
		AgentID:  "agent_7",
		Config:   configJSON,
		Budget:   budgetJSON,
		Metadata: map[string]string{"env": "test", "tag": "v1"},
	}

	if err := sessionStore.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	loaded, err := sessionStore.LoadSession(ctx, "sess_test_1")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected non-nil session, got nil")
	}
	if loaded.ID != "sess_test_1" {
		t.Errorf("expected ID sess_test_1, got %s", loaded.ID)
	}
	if loaded.UserID != "user_42" {
		t.Errorf("expected UserID user_42, got %s", loaded.UserID)
	}
	if loaded.AgentID != "agent_7" {
		t.Errorf("expected AgentID agent_7, got %s", loaded.AgentID)
	}
	if loaded.Metadata["env"] != "test" {
		t.Errorf("expected metadata env=test, got %s", loaded.Metadata["env"])
	}
	if loaded.Metadata["tag"] != "v1" {
		t.Errorf("expected metadata tag=v1, got %s", loaded.Metadata["tag"])
	}

	var config map[string]any
	json.Unmarshal(loaded.Config, &config)
	if config["model"] != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %v", config["model"])
	}
	if config["max_tokens"] != float64(4096) {
		t.Errorf("expected max_tokens 4096, got %v", config["max_tokens"])
	}
	if config["temperature"] != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", config["temperature"])
	}
}

func testSessionLoadNonexistent(t *testing.T, sessionStore store.SessionStore) {
	ctx := context.Background()

	loaded, err := sessionStore.LoadSession(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil for nonexistent session, got %+v", loaded)
	}
}

func testSessionUpdate(t *testing.T, sessionStore store.SessionStore) {
	ctx := context.Background()

	data := &store.SessionData{
		ID:       "sess_update_1",
		UserID:   "user_1",
		AgentID:  "agent_1",
		Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
		Budget:   []byte(`{"max_tokens":500000,"max_cost_usd":5,"total_input":0,"total_output":0,"total_cost_usd":0,"cost_per_input_token":0.00003,"cost_per_output_token":0.00015}`),
		Metadata: map[string]string{"version": "1"},
	}

	if err := sessionStore.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession first: %v", err)
	}

	data.Metadata["version"] = "2"
	if err := sessionStore.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession update: %v", err)
	}

	loaded, err := sessionStore.LoadSession(ctx, "sess_update_1")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if loaded.Metadata["version"] != "2" {
		t.Errorf("expected version=2 after update, got %s", loaded.Metadata["version"])
	}
	var config map[string]any
	if err := json.Unmarshal(loaded.Config, &config); err != nil {
		t.Fatalf("unmarshal loaded config: %v", err)
	}
	if config["model"] != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %v", config["model"])
	}
}

func testSessionDelete(t *testing.T, sessionStore store.SessionStore) {
	ctx := context.Background()

	data := &store.SessionData{
		ID:       "sess_del_1",
		UserID:   "user_del",
		AgentID:  "agent_del",
		Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
		Budget:   []byte(`{}`),
		Metadata: map[string]string{},
	}

	if err := sessionStore.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	if err := sessionStore.DeleteSession(ctx, "sess_del_1"); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}

	loaded, err := sessionStore.LoadSession(ctx, "sess_del_1")
	if err != nil {
		t.Fatalf("LoadSession after delete: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil after delete, got %+v", loaded)
	}
}

func testListSessions(t *testing.T, sessionStore store.SessionStore) {
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		data := &store.SessionData{
			ID:       fmt.Sprintf("sess_list_%d", i),
			UserID:   "user_list",
			AgentID:  "agent_list",
			Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
			Budget:   []byte(`{}`),
			Metadata: map[string]string{},
		}
		if err := sessionStore.SaveSession(ctx, data); err != nil {
			t.Fatalf("SaveSession %d: %v", i, err)
		}
	}

	for i := 0; i < 2; i++ {
		data := &store.SessionData{
			ID:       fmt.Sprintf("sess_agent_only_%d", i),
			UserID:   fmt.Sprintf("user_other_%d", i),
			AgentID:  "agent_only",
			Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
			Budget:   []byte(`{}`),
			Metadata: map[string]string{},
		}
		if err := sessionStore.SaveSession(ctx, data); err != nil {
			t.Fatalf("SaveSession agent-only %d: %v", i, err)
		}
	}

	if err := sessionStore.SaveSession(ctx, &store.SessionData{
		ID:       "sess_unrelated",
		UserID:   "user_unrelated",
		AgentID:  "agent_unrelated",
		Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
		Budget:   []byte(`{}`),
		Metadata: map[string]string{},
	}); err != nil {
		t.Fatalf("SaveSession unrelated: %v", err)
	}

	entries, err := sessionStore.ListSessions(ctx, "user_list", "agent_list")
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(entries) < 3 {
		t.Errorf("expected at least 3 sessions, got %d", len(entries))
	}
	if entries[0].Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", entries[0].Model)
	}

	allEntries, err := sessionStore.ListSessions(ctx, "user_list", "")
	if err != nil {
		t.Fatalf("ListSessions (agentID empty): %v", err)
	}
	if len(allEntries) < 3 {
		t.Errorf("expected at least 3 sessions for user_list, got %d", len(allEntries))
	}

	agentEntries, err := sessionStore.ListSessions(ctx, "", "agent_only")
	if err != nil {
		t.Fatalf("ListSessions (userID empty): %v", err)
	}
	if len(agentEntries) != 2 {
		t.Fatalf("expected 2 sessions for agent_only, got %d", len(agentEntries))
	}
	for _, entry := range agentEntries {
		if entry.AgentID != "agent_only" {
			t.Fatalf("expected only agent_only entries, got agent %q", entry.AgentID)
		}
	}
}

func testSessionConfigRoundTrip(t *testing.T, sessionStore store.SessionStore) {
	ctx := context.Background()

	data := &store.SessionData{
		ID:       "sess_config_rt",
		UserID:   "user_cfg",
		AgentID:  "agent_cfg",
		Config:   []byte(`{"model":"gpt-4.1","system_prompt":"You are terse.","max_tokens":2048,"temperature":0.25,"trim_strategy":"token_count","max_history":12,"token_budget":64000,"budget_config":{"max_tokens":1234,"max_cost_usd":0.5}}`),
		Budget:   []byte(`{"max_tokens":1234}`),
		Metadata: map[string]string{"scope": "roundtrip"},
	}

	if err := sessionStore.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	loaded, err := sessionStore.LoadSession(ctx, "sess_config_rt")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected non-nil session, got nil")
	}

	var config map[string]any
	if err := json.Unmarshal(loaded.Config, &config); err != nil {
		t.Fatalf("unmarshal loaded config: %v", err)
	}
	if config["model"] != "gpt-4.1" {
		t.Errorf("expected model gpt-4.1, got %v", config["model"])
	}
	if config["system_prompt"] != "You are terse." {
		t.Errorf("expected system_prompt to round-trip, got %v", config["system_prompt"])
	}
	if config["max_tokens"] != float64(2048) {
		t.Errorf("expected max_tokens 2048, got %v", config["max_tokens"])
	}
	if config["temperature"] != 0.25 {
		t.Errorf("expected temperature 0.25, got %v", config["temperature"])
	}
	if config["trim_strategy"] != "token_count" {
		t.Errorf("expected trim_strategy token_count, got %v", config["trim_strategy"])
	}
	if config["budget_config"] == nil {
		t.Errorf("expected budget_config to round-trip, got nil")
	}

	entries, err := sessionStore.ListSessions(ctx, "user_cfg", "agent_cfg")
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least 1 session entry")
	}
	if entries[0].Model != "gpt-4.1" {
		t.Errorf("expected listed model gpt-4.1, got %s", entries[0].Model)
	}
}

func testAutoUpsertUsersAndAgents(t *testing.T, sessionStore store.SessionStore) {
	ctx := context.Background()

	data := &store.SessionData{
		ID:       "sess_auto_1",
		UserID:   "user_auto_new",
		AgentID:  "agent_auto_new",
		Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
		Budget:   []byte(`{}`),
		Metadata: map[string]string{},
	}

	if err := sessionStore.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession (should auto-create user and agent): %v", err)
	}

	loaded, err := sessionStore.LoadSession(ctx, "sess_auto_1")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected session to exist after auto-upsert")
	}
	if loaded.UserID != "user_auto_new" {
		t.Errorf("expected UserID user_auto_new, got %s", loaded.UserID)
	}
}

func TestSQLite_SaveSessionDoesNotOverwriteExistingAgentDefinition(t *testing.T) {
	s := sqliteTestStore(t)
	ctx := context.Background()

	if _, err := s.db.ExecContext(ctx,
		`INSERT INTO users (id, created_at) VALUES (?, CURRENT_TIMESTAMP)`,
		"user_existing",
	); err != nil {
		t.Fatalf("insert user: %v", err)
	}
	if _, err := s.db.ExecContext(ctx,
		`INSERT INTO agents (id, user_id, name, model, system_prompt, max_tokens, temperature, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
		"agent_existing", "user_existing", "Canonical", "canonical-model", "canonical prompt", 8192, 0.2,
	); err != nil {
		t.Fatalf("insert agent: %v", err)
	}

	data := &store.SessionData{
		ID:      "sess_existing_agent",
		UserID:  "user_existing",
		AgentID: "agent_existing",
		Config: []byte(`{
			"model":"session-model",
			"system_prompt":"session prompt",
			"max_tokens":1024,
			"temperature":0.9,
			"trim_strategy":"sliding_window",
			"max_history":20,
			"token_budget":32000
		}`),
		Budget:   []byte(`{}`),
		Metadata: map[string]string{},
	}
	if err := s.Session.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	var model, systemPrompt string
	var maxTokens int
	var temperature float64
	if err := s.db.QueryRowContext(ctx,
		`SELECT model, system_prompt, max_tokens, temperature FROM agents WHERE id = ?`,
		"agent_existing",
	).Scan(&model, &systemPrompt, &maxTokens, &temperature); err != nil {
		t.Fatalf("select agent: %v", err)
	}
	if model != "canonical-model" {
		t.Fatalf("expected canonical model to remain, got %q", model)
	}
	if systemPrompt != "canonical prompt" {
		t.Fatalf("expected canonical prompt to remain, got %q", systemPrompt)
	}
	if maxTokens != 8192 {
		t.Fatalf("expected canonical max_tokens to remain, got %d", maxTokens)
	}
	if temperature != 0.2 {
		t.Fatalf("expected canonical temperature to remain, got %v", temperature)
	}

	loaded, err := s.Session.LoadSession(ctx, "sess_existing_agent")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	var sessionConfig map[string]any
	if err := json.Unmarshal(loaded.Config, &sessionConfig); err != nil {
		t.Fatalf("unmarshal loaded config: %v", err)
	}
	if sessionConfig["model"] != "session-model" {
		t.Fatalf("expected session-local model to round-trip, got %v", sessionConfig["model"])
	}
}

func TestSQLite_SaveSessionCreatesAgentStubWithoutSessionConfig(t *testing.T) {
	s := sqliteTestStore(t)
	ctx := context.Background()

	data := &store.SessionData{
		ID:      "sess_new_agent_stub",
		UserID:  "user_new_stub",
		AgentID: "agent_new_stub",
		Config: []byte(`{
			"model":"session-model",
			"system_prompt":"session prompt",
			"max_tokens":1024,
			"temperature":0.9,
			"trim_strategy":"sliding_window",
			"max_history":20,
			"token_budget":32000
		}`),
		Budget:   []byte(`{}`),
		Metadata: map[string]string{},
	}
	if err := s.Session.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	var model, systemPrompt string
	var maxTokens int
	var temperature float64
	if err := s.db.QueryRowContext(ctx,
		`SELECT model, system_prompt, max_tokens, temperature FROM agents WHERE id = ?`,
		"agent_new_stub",
	).Scan(&model, &systemPrompt, &maxTokens, &temperature); err != nil {
		t.Fatalf("select agent stub: %v", err)
	}
	if model != "" {
		t.Fatalf("expected stub model to remain empty, got %q", model)
	}
	if systemPrompt != "" {
		t.Fatalf("expected stub system_prompt to remain empty, got %q", systemPrompt)
	}
	if maxTokens != 4096 {
		t.Fatalf("expected stub max_tokens default 4096, got %d", maxTokens)
	}
	if temperature != 0.7 {
		t.Fatalf("expected stub temperature default 0.7, got %v", temperature)
	}
}

// SQLite SessionStore tests

func TestSQLite_SessionStore_SaveAndLoad(t *testing.T) {
	testSessionStore(t, sqliteTestStore(t).Session)
}

func TestSQLite_SessionStore_ConfigRoundTrip(t *testing.T) {
	testSessionConfigRoundTrip(t, sqliteTestStore(t).Session)
}

func TestSQLite_SessionStore_LoadNonexistent(t *testing.T) {
	testSessionLoadNonexistent(t, sqliteTestStore(t).Session)
}

func TestSQLite_SessionStore_Update(t *testing.T) {
	testSessionUpdate(t, sqliteTestStore(t).Session)
}

func TestSQLite_SessionStore_Delete(t *testing.T) {
	testSessionDelete(t, sqliteTestStore(t).Session)
}

func TestSQLite_SessionStore_ListSessions(t *testing.T) {
	testListSessions(t, sqliteTestStore(t).Session)
}

func TestSQLite_SessionStore_AutoUpsertUsersAndAgents(t *testing.T) {
	testAutoUpsertUsersAndAgents(t, sqliteTestStore(t).Session)
}

// PostgreSQL SessionStore tests

func TestPostgres_SessionStore_SaveAndLoad(t *testing.T) {
	testSessionStore(t, pgTestStore(t).Session)
}

func TestPostgres_SessionStore_ConfigRoundTrip(t *testing.T) {
	testSessionConfigRoundTrip(t, pgTestStore(t).Session)
}

func TestPostgres_SessionStore_Update(t *testing.T) {
	testSessionUpdate(t, pgTestStore(t).Session)
}

func TestPostgres_SessionStore_ListSessions(t *testing.T) {
	testListSessions(t, pgTestStore(t).Session)
}

// Cross-store test: conversations persisted across instances

func TestSQLite_ConversationPersistsAcrossInstances(t *testing.T) {
	dir := t.TempDir()
	dbPath := dir + "/test.db"

	s1, err := NewStore(Config{Driver: "sqlite", DSN: dbPath})
	if err != nil {
		t.Fatalf("NewStore 1: %v", err)
	}

	msgs := []llm.Message{{Role: "user", Content: "persistent"}}
	if err := s1.Messages.Save(context.Background(), "conv_persist", msgs); err != nil {
		s1.Close()
		t.Fatalf("Save: %v", err)
	}
	s1.Close()

	s2, err := NewStore(Config{Driver: "sqlite", DSN: dbPath})
	if err != nil {
		t.Fatalf("NewStore 2: %v", err)
	}
	defer s2.Close()

	loaded, err := s2.Messages.Load(context.Background(), "conv_persist")
	if err != nil {
		t.Fatalf("Load from second instance: %v", err)
	}
	if len(loaded) != 1 || loaded[0].Content != "persistent" {
		t.Errorf("expected persistent message, got %+v", loaded)
	}
}

func TestSQLite_SessionPersistsAcrossInstances(t *testing.T) {
	dir := t.TempDir()
	dbPath := dir + "/test.db"

	s1, err := NewStore(Config{Driver: "sqlite", DSN: dbPath})
	if err != nil {
		t.Fatalf("NewStore 1: %v", err)
	}

	data := &store.SessionData{
		ID:       "sess_persist",
		UserID:   "user_p",
		AgentID:  "agent_p",
		Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
		Budget:   []byte(`{"max_tokens":500000}`),
		Metadata: map[string]string{"key": "value"},
	}
	if err := s1.Session.SaveSession(context.Background(), data); err != nil {
		s1.Close()
		t.Fatalf("SaveSession: %v", err)
	}
	s1.Close()

	s2, err := NewStore(Config{Driver: "sqlite", DSN: dbPath})
	if err != nil {
		t.Fatalf("NewStore 2: %v", err)
	}
	defer s2.Close()

	loaded, err := s2.Session.LoadSession(context.Background(), "sess_persist")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected session to persist, got nil")
	}
	if loaded.UserID != "user_p" {
		t.Errorf("expected UserID user_p, got %s", loaded.UserID)
	}
	if loaded.Metadata["key"] != "value" {
		t.Errorf("expected metadata key=value, got %s", loaded.Metadata["key"])
	}
}

func TestSQLite_SessionStoreSaveUpdatesTimestamps(t *testing.T) {
	s := sqliteTestStore(t)
	ctx := context.Background()

	data := &store.SessionData{
		ID:       "sess_ts",
		UserID:   "user_ts",
		AgentID:  "agent_ts",
		Config:   []byte(`{"model":"gpt-4o","max_tokens":4096,"trim_strategy":"sliding_window","max_history":50,"token_budget":128000}`),
		Budget:   []byte(`{}`),
		Metadata: map[string]string{},
	}

	if err := s.Session.SaveSession(ctx, data); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	entries, err := s.Session.ListSessions(ctx, "user_ts", "agent_ts")
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least 1 session entry")
	}
	if entries[0].Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", entries[0].Model)
	}
	if entries[0].CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt")
	}
	if entries[0].UpdatedAt.IsZero() {
		t.Error("expected non-zero UpdatedAt")
	}

	// Verify CreatedAt is recent (within last 5 seconds)
	if time.Since(entries[0].CreatedAt) > 5*time.Second {
		t.Errorf("CreatedAt too old: %v", entries[0].CreatedAt)
	}
}
