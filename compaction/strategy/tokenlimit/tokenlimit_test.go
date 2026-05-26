package tokenlimit

import (
	"context"
	"testing"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/llm"
)

func TestStrategy_DropsWholeTurns(t *testing.T) {
	strategy := New(40, fixedEstimator{
		values: map[string]int{
			"system":      1,
			"user:a":      5,
			"assistant:b": 5,
			"user:c":      2,
			"assistant:d": 2,
		},
	})

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "system"},
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "b"},
			{Role: "user", Content: "c"},
			{Role: "assistant", Content: "d"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	if !out.Compacted {
		t.Fatalf("expected compacted output")
	}
	if len(out.Messages) != 3 {
		t.Fatalf("expected 3 messages after dropping the oldest full turn, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "system" || out.Messages[1].Content != "c" || out.Messages[2].Content != "d" {
		t.Fatalf("unexpected retained messages: %+v", out.Messages)
	}
}

func TestStrategy_DropsWholeTurnWithToolMessages(t *testing.T) {
	strategy := New(28, fixedEstimator{
		values: map[string]int{
			"system":           1,
			"user:question":    3,
			"assistant:":       1,
			"tool:tool result": 5,
			"user:followup":    2,
			"assistant:answer": 2,
		},
	})

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "system"},
			{Role: "user", Content: "question"},
			{Role: "assistant", Content: ""},
			{Role: "tool", Content: "tool result"},
			{Role: "user", Content: "followup"},
			{Role: "assistant", Content: "answer"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	if len(out.Messages) != 3 {
		t.Fatalf("expected system plus the newest full turn, got %d messages", len(out.Messages))
	}
	if out.Messages[1].Content != "followup" || out.Messages[2].Content != "answer" {
		t.Fatalf("expected the tool-backed turn to be dropped as a unit, got %+v", out.Messages)
	}
}

type fixedEstimator struct {
	values map[string]int
}

func (e fixedEstimator) Estimate(messages []llm.Message) int {
	total := 0
	for _, m := range messages {
		total += e.values[m.Role+":"+m.Content]
	}
	return total
}
