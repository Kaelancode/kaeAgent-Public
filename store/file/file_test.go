package file

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

func testStore(t *testing.T) *ConversationStore {
	dir := t.TempDir()
	s, err := NewConversationStore(Config{Dir: dir})
	if err != nil {
		t.Fatalf("NewConversationStore: %v", err)
	}
	return s
}

func TestConversationStore_SaveAndLoad(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	msgs := []llm.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi!"},
	}

	if err := s.Save(ctx, "conv_1", msgs); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_1")
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

func TestConversationStore_LoadNonexistent(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	loaded, err := s.Load(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil for nonexistent key, got %v", loaded)
	}
}

func TestConversationStore_Append(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	initial := []llm.Message{
		{Role: "user", Content: "First"},
	}
	if err := s.Save(ctx, "conv_1", initial); err != nil {
		t.Fatalf("Save: %v", err)
	}

	appendMsgs := []llm.Message{
		{Role: "assistant", Content: "Second"},
		{Role: "user", Content: "Third"},
	}
	if err := s.Append(ctx, "conv_1", appendMsgs); err != nil {
		t.Fatalf("Append: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_1")
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

func TestConversationStore_AppendToEmpty(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	msgs := []llm.Message{
		{Role: "user", Content: "Hello"},
	}
	if err := s.Append(ctx, "conv_new", msgs); err != nil {
		t.Fatalf("Append: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_new")
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

func TestConversationStore_Delete(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	msgs := []llm.Message{{Role: "user", Content: "Hi"}}
	if err := s.Save(ctx, "conv_1", msgs); err != nil {
		t.Fatalf("Save: %v", err)
	}

	if err := s.Delete(ctx, "conv_1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_1")
	if err != nil {
		t.Fatalf("Load after delete: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil after delete, got %v", loaded)
	}
}

func TestConversationStore_RespectsCanceledContext(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	canceled, cancel := context.WithCancel(ctx)
	cancel()

	msgs := []llm.Message{{Role: "user", Content: "blocked"}}
	if err := s.Save(canceled, "conv_cancel", msgs); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled Save error, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(s.dir, "conv_cancel.json")); !os.IsNotExist(err) {
		t.Fatalf("expected canceled Save to skip file creation, stat err=%v", err)
	}

	if err := s.Save(ctx, "conv_existing", []llm.Message{{Role: "user", Content: "initial"}}); err != nil {
		t.Fatalf("Save existing failed: %v", err)
	}
	if _, err := s.Load(canceled, "conv_existing"); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled Load error, got %v", err)
	}
	if err := s.Append(canceled, "conv_existing", []llm.Message{{Role: "assistant", Content: "blocked"}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled Append error, got %v", err)
	}
	loaded, err := s.Load(ctx, "conv_existing")
	if err != nil {
		t.Fatalf("Load after canceled Append failed: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected canceled Append to skip mutation, got %d messages", len(loaded))
	}
	if err := s.Delete(canceled, "conv_existing"); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled Delete error, got %v", err)
	}
	loaded, err = s.Load(ctx, "conv_existing")
	if err != nil {
		t.Fatalf("Load after canceled Delete failed: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected canceled Delete to preserve data, got %#v", loaded)
	}
}

func TestConversationStore_PersistsAcrossInstances(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	s1, err := NewConversationStore(Config{Dir: dir})
	if err != nil {
		t.Fatalf("NewConversationStore 1: %v", err)
	}

	msgs := []llm.Message{
		{Role: "user", Content: "persistent"},
	}
	if err := s1.Save(ctx, "conv_persist", msgs); err != nil {
		t.Fatalf("Save: %v", err)
	}

	s2, err := NewConversationStore(Config{Dir: dir})
	if err != nil {
		t.Fatalf("NewConversationStore 2: %v", err)
	}

	loaded, err := s2.Load(ctx, "conv_persist")
	if err != nil {
		t.Fatalf("Load from second instance: %v", err)
	}
	if len(loaded) != 1 || loaded[0].Content != "persistent" {
		t.Errorf("expected persistent message, got %+v", loaded)
	}
}

func TestConversationStore_ToolCallsRoundTrip(t *testing.T) {
	s := testStore(t)
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

	if err := s.Save(ctx, "conv_tools", msgs); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_tools")
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

	_ = os.RemoveAll(filepath.Dir(s.dir))
}

func TestConversationStore_RejectsPathTraversalConvID(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	err := s.Save(ctx, "../../escape", []llm.Message{{Role: "user", Content: "blocked"}})
	if err == nil {
		t.Fatal("expected path traversal convID to be rejected")
	}
	if !strings.Contains(err.Error(), "invalid conversation id") {
		t.Fatalf("expected invalid conversation id error, got %v", err)
	}
}

func TestAtomicWriteRemovesTempFileOnRenameFailure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "target")
	if err := os.Mkdir(path, 0755); err != nil {
		t.Fatalf("mkdir target: %v", err)
	}

	err := atomicWrite(path, []byte("data"))
	if err == nil {
		t.Fatal("expected rename failure")
	}
	if !strings.Contains(err.Error(), "rename temp file") {
		t.Fatalf("expected rename temp file error, got %v", err)
	}
	if _, statErr := os.Stat(path + ".tmp"); !os.IsNotExist(statErr) {
		t.Fatalf("expected temp file to be removed, stat err=%v", statErr)
	}
}
