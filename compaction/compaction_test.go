package compaction_test

import (
	"context"
	"testing"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/compaction/strategy/slidingwindow"
	"github.com/yourorg/agent-sdk/llm"
)

func TestEngine_NoTriggerLeavesMessagesUntouched(t *testing.T) {
	engine := compaction.NewEngine(nil, nil, nil)
	input := compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "hello"},
		},
	}

	out, err := engine.Compact(context.Background(), input)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if len(out.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out.Messages))
	}
	if out.Compacted {
		t.Fatalf("expected non-compacted output")
	}
}

func TestEngine_AppliesStrategyWhenTriggered(t *testing.T) {
	engine := compaction.NewEngine(
		compaction.MaxMessagesTrigger{MaxMessages: 2},
		slidingwindow.New(2),
		nil,
	)
	input := compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "b"},
		},
	}

	out, err := engine.Compact(context.Background(), input)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !out.Compacted {
		t.Fatalf("expected compacted output")
	}
	if len(out.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out.Messages))
	}
}

func TestMaxTurnsTrigger_CountsUserTurns(t *testing.T) {
	trigger := compaction.MaxTurnsTrigger{MaxTurns: 2}
	input := compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "b"},
			{Role: "tool", Content: "tool result"},
			{Role: "user", Content: "c"},
			{Role: "assistant", Content: "d"},
			{Role: "user", Content: "e"},
		},
	}

	ok, reason, err := trigger.ShouldCompact(context.Background(), input)
	if err != nil {
		t.Fatalf("ShouldCompact: %v", err)
	}
	if !ok {
		t.Fatalf("expected turn-based trigger to compact")
	}
	if reason != "max turns exceeded" {
		t.Fatalf("unexpected reason: %q", reason)
	}
}

func TestMaxTurnsTrigger_IgnoresNonUserMessages(t *testing.T) {
	trigger := compaction.MaxTurnsTrigger{MaxTurns: 2}
	input := compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "b"},
			{Role: "tool", Content: "tool result"},
		},
	}

	ok, _, err := trigger.ShouldCompact(context.Background(), input)
	if err != nil {
		t.Fatalf("ShouldCompact: %v", err)
	}
	if ok {
		t.Fatalf("expected turn-based trigger not to compact")
	}
}
