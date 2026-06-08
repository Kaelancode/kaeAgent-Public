package inmem

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

func TestConversationStore_SaveAndLoad(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	msgs := []llm.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
	}

	if err := s.Save(ctx, "conv_1", msgs); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_1")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if len(loaded) != len(msgs) {
		t.Fatalf("expected %d messages, got %d", len(msgs), len(loaded))
	}
	for i := range msgs {
		if loaded[i].Role != msgs[i].Role || loaded[i].Content != msgs[i].Content || loaded[i].Name != msgs[i].Name || loaded[i].ToolCallID != msgs[i].ToolCallID {
			t.Errorf("message %d mismatch: got %+v want %+v", i, loaded[i], msgs[i])
		}
	}
}

func TestConversationStore_LoadNonexistent(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	loaded, err := s.Load(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil for nonexistent conversation, got %v", loaded)
	}
}

func TestConversationStore_Append(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	initial := []llm.Message{
		{Role: "user", Content: "Hello"},
	}
	appendMsgs := []llm.Message{
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "How are you?"},
	}

	if err := s.Save(ctx, "conv_1", initial); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	if err := s.Append(ctx, "conv_1", appendMsgs); err != nil {
		t.Fatalf("Append failed: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_1")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	expectedLen := len(initial) + len(appendMsgs)
	if len(loaded) != expectedLen {
		t.Fatalf("expected %d messages, got %d", expectedLen, len(loaded))
	}
}

func TestConversationStore_AppendToEmpty(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	msgs := []llm.Message{
		{Role: "user", Content: "First message"},
	}

	if err := s.Append(ctx, "conv_new", msgs); err != nil {
		t.Fatalf("Append failed: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_new")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if len(loaded) != len(msgs) {
		t.Fatalf("expected %d messages, got %d", len(msgs), len(loaded))
	}
}

func TestConversationStore_Delete(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	msgs := []llm.Message{
		{Role: "user", Content: "To be deleted"},
	}

	if err := s.Save(ctx, "conv_1", msgs); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	if err := s.Delete(ctx, "conv_1"); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	loaded, err := s.Load(ctx, "conv_1")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if loaded != nil {
		t.Errorf("expected nil after delete, got %v", loaded)
	}
}

func TestConversationStore_RespectsCanceledContext(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()
	canceled, cancel := context.WithCancel(ctx)
	cancel()

	msgs := []llm.Message{{Role: "user", Content: "blocked"}}
	if err := s.Save(canceled, "conv_cancel", msgs); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled Save error, got %v", err)
	}
	loaded, err := s.Load(ctx, "conv_cancel")
	if err != nil {
		t.Fatalf("Load after canceled Save failed: %v", err)
	}
	if loaded != nil {
		t.Fatalf("expected canceled Save to skip mutation, got %#v", loaded)
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
	loaded, err = s.Load(ctx, "conv_existing")
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

func TestConversationStore_Isolation(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	msgs1 := []llm.Message{{Role: "user", Content: "conv1"}}
	msgs2 := []llm.Message{{Role: "user", Content: "conv2"}}

	s.Save(ctx, "conv_1", msgs1)
	s.Save(ctx, "conv_2", msgs2)

	loaded1, _ := s.Load(ctx, "conv_1")
	loaded2, _ := s.Load(ctx, "conv_2")

	if loaded1[0].Content != "conv1" {
		t.Errorf("expected conv1, got %s", loaded1[0].Content)
	}
	if loaded2[0].Content != "conv2" {
		t.Errorf("expected conv2, got %s", loaded2[0].Content)
	}
}

func TestConversationStore_ConcurrentAccess(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(3)
		id := "conv_" + string(rune('0'+i))
		go func() {
			defer wg.Done()
			s.Save(ctx, id, []llm.Message{{Role: "user", Content: id}})
		}()
		go func() {
			defer wg.Done()
			s.Load(ctx, id)
		}()
		go func() {
			defer wg.Done()
			s.Append(ctx, id, []llm.Message{{Role: "assistant", Content: id}})
		}()
	}
	wg.Wait()
}

func TestConversationStore_DeepCopiesToolInputs(t *testing.T) {
	s := NewConversationStore()
	ctx := context.Background()

	msgs := []llm.Message{
		{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{
				{
					ID:   "call_1",
					Name: "lookup",
					Input: map[string]any{
						"query": "weather",
						"nested": map[string]any{
							"unit": "celsius",
						},
					},
				},
			},
		},
	}

	if err := s.Save(ctx, "conv_tools", msgs); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	msgs[0].ToolCalls[0].Input["query"] = "mutated"
	msgs[0].ToolCalls[0].Input["nested"].(map[string]any)["unit"] = "fahrenheit"

	loaded, err := s.Load(ctx, "conv_tools")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if got := loaded[0].ToolCalls[0].Input["query"]; got != "weather" {
		t.Fatalf("expected stored query weather, got %#v", got)
	}
	if got := loaded[0].ToolCalls[0].Input["nested"].(map[string]any)["unit"]; got != "celsius" {
		t.Fatalf("expected stored nested unit celsius, got %#v", got)
	}

	loaded[0].ToolCalls[0].Input["query"] = "loaded-mutated"
	loaded[0].ToolCalls[0].Input["nested"].(map[string]any)["unit"] = "kelvin"

	reloaded, err := s.Load(ctx, "conv_tools")
	if err != nil {
		t.Fatalf("Reload failed: %v", err)
	}
	if got := reloaded[0].ToolCalls[0].Input["query"]; got != "weather" {
		t.Fatalf("expected reloaded query weather, got %#v", got)
	}
	if got := reloaded[0].ToolCalls[0].Input["nested"].(map[string]any)["unit"]; got != "celsius" {
		t.Fatalf("expected reloaded nested unit celsius, got %#v", got)
	}
}
