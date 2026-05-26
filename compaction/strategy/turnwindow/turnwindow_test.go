package turnwindow

import (
	"context"
	"testing"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/llm"
)

func TestStrategy_KeepsLatestTurns(t *testing.T) {
	strategy := New(2)

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "system"},
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "one"},
			{Role: "user", Content: "b"},
			{Role: "assistant", Content: "two"},
			{Role: "user", Content: "c"},
			{Role: "assistant", Content: "three"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	if !out.Compacted {
		t.Fatalf("expected compacted output")
	}
	if len(out.Messages) != 5 {
		t.Fatalf("expected system plus 2 retained turns, got %d messages", len(out.Messages))
	}
	if out.Messages[1].Content != "b" || out.Messages[2].Content != "two" || out.Messages[3].Content != "c" || out.Messages[4].Content != "three" {
		t.Fatalf("unexpected retained messages: %+v", out.Messages)
	}
}

func TestStrategy_KeepsToolBackedTurnIntact(t *testing.T) {
	strategy := New(1)

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "system"},
			{Role: "user", Content: "first"},
			{Role: "assistant", Content: "alpha"},
			{Role: "user", Content: "question"},
			{Role: "assistant", Content: ""},
			{Role: "tool", Content: "tool result"},
			{Role: "assistant", Content: "final"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	if len(out.Messages) != 5 {
		t.Fatalf("expected system plus one intact turn, got %d messages", len(out.Messages))
	}
	if out.Messages[1].Content != "question" || out.Messages[2].Role != "assistant" || out.Messages[3].Role != "tool" || out.Messages[4].Content != "final" {
		t.Fatalf("expected latest tool-backed turn to remain intact, got %+v", out.Messages)
	}
}
