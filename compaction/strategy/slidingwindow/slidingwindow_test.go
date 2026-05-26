package slidingwindow

import (
	"context"
	"strings"
	"testing"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/llm"
)

func TestCompactErrorsWhenSystemMessagesExceedMaxMessages(t *testing.T) {
	strategy := New(2)

	_, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys-1"},
			{Role: "system", Content: "sys-2"},
			{Role: "system", Content: "sys-3"},
			{Role: "user", Content: "hello"},
		},
	})
	if err == nil {
		t.Fatal("expected compaction error when system messages exceed MaxMessages")
	}
	if !strings.Contains(err.Error(), "system message count") {
		t.Fatalf("expected system message count error, got %v", err)
	}
}

func TestCompactPreservesSystemMessagesWithinLimit(t *testing.T) {
	strategy := New(3)

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys-1"},
			{Role: "system", Content: "sys-2"},
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "b"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if len(out.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "system" || out.Messages[1].Role != "system" {
		t.Fatalf("expected system messages preserved first, got %#v", out.Messages)
	}
	if out.Messages[2].Content != "b" {
		t.Fatalf("expected newest non-system message kept, got %#v", out.Messages[2])
	}
}
