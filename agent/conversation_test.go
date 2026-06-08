package agent

import (
	"encoding/json"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

func TestConversation_AppendAndMessages(t *testing.T) {
	c := NewConversation(TrimSlidingWindow, 100, 128000)
	c.Append(llm.Message{Role: "user", Content: "hello"})
	c.Append(llm.Message{Role: "assistant", Content: "hi"})

	msgs := c.Messages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "user" || msgs[1].Role != "assistant" {
		t.Errorf("unexpected roles: %s, %s", msgs[0].Role, msgs[1].Role)
	}
}

func TestConversation_SetSystem(t *testing.T) {
	c := NewConversation(TrimSlidingWindow, 100, 128000)
	c.SetSystem("you are helpful")
	c.Append(llm.Message{Role: "user", Content: "hello"})

	msgs := c.Messages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Errorf("expected system message first, got %s", msgs[0].Role)
	}

	c.SetSystem("updated system")
	msgs = c.Messages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages after system update, got %d", len(msgs))
	}
	if msgs[0].Content != "updated system" {
		t.Errorf("expected updated system content, got %s", msgs[0].Content)
	}
}

func TestConversation_PreservesFullHistoryWithoutInlineTrimming(t *testing.T) {
	c := NewConversation(TrimSlidingWindow, 3, 128000)
	c.SetSystem("system prompt")

	for i := 0; i < 5; i++ {
		c.Append(llm.Message{Role: "user", Content: "msg"})
	}

	msgs := c.Messages()
	if len(msgs) != 6 {
		t.Errorf("expected full message history to be preserved, got %d messages", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Errorf("system message should be preserved, got role=%s", msgs[0].Role)
	}
}

func TestConversation_Slice(t *testing.T) {
	c := NewConversation(TrimSlidingWindow, 100, 128000)
	c.Append(llm.Message{Role: "user", Content: "a"})
	c.Append(llm.Message{Role: "assistant", Content: "b"})
	c.Append(llm.Message{Role: "user", Content: "c"})

	slice := c.Slice(1, 3)
	if len(slice) != 2 {
		t.Fatalf("expected 2 messages in slice, got %d", len(slice))
	}
	if slice[0].Content != "b" {
		t.Errorf("expected 'b', got %s", slice[0].Content)
	}
}

func TestConversation_Clear(t *testing.T) {
	c := NewConversation(TrimSlidingWindow, 100, 128000)
	c.Append(llm.Message{Role: "user", Content: "hello"})
	c.Clear()
	if c.Len() != 0 {
		t.Errorf("expected 0 messages after clear, got %d", c.Len())
	}
}

func TestConversation_MessagesOwnedDeepCopiesNestedData(t *testing.T) {
	c := NewConversation(TrimSlidingWindow, 100, 128000)
	c.Append(llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{{
			ID:   "call_1",
			Name: "lookup",
			Input: map[string]any{
				"query": "original",
				"nested": map[string]any{
					"key": "value",
				},
			},
		}},
	})

	msgs := c.messagesOwned()
	msgs[0].ToolCalls[0].Input["query"] = "mutated"
	msgs[0].ToolCalls[0].Input["nested"].(map[string]any)["key"] = "changed"

	snap := c.Snapshot()
	got := snap.Messages[0].ToolCalls[0].Input
	if got["query"] != "original" {
		t.Fatalf("expected original query to remain unchanged, got %v", got["query"])
	}
	if got["nested"].(map[string]any)["key"] != "value" {
		t.Fatalf("expected nested map to remain unchanged, got %v", got["nested"].(map[string]any)["key"])
	}
}

func TestConversation_UnmarshalJSON_IgnoresLegacyID(t *testing.T) {
	data := []byte(`{"id":"conv_legacy","messages":[{"role":"user","content":"hello"}],"strategy":"sliding_window","max_msgs":5,"max_chars":40}`)

	var c Conversation
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	msgs := c.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Content != "hello" {
		t.Fatalf("expected restored content 'hello', got %q", msgs[0].Content)
	}

	snap := c.Snapshot()
	if snap.Strategy != TrimSlidingWindow {
		t.Fatalf("expected strategy %q, got %q", TrimSlidingWindow, snap.Strategy)
	}
	if snap.MaxMsgs != 5 {
		t.Fatalf("expected max msgs 5, got %d", snap.MaxMsgs)
	}
	if snap.MaxChars != 40 {
		t.Fatalf("expected max chars 40, got %d", snap.MaxChars)
	}
}
