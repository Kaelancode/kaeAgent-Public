package file

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yourorg/agent-sdk/llm"
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
