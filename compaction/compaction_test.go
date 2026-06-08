package compaction_test

import (
	"context"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/compaction/strategy/slidingwindow"
	"github.com/Kaelancode/kaeAgent-Public/llm"
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

func TestEngine_ForceCompactSetsDefaultReason(t *testing.T) {
	engine := compaction.NewEngine(nil, reasonStrategy{}, nil)

	out, err := engine.ForceCompact(context.Background(), compaction.Input{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("ForceCompact: %v", err)
	}
	if out.Reason != "forced compaction" {
		t.Fatalf("expected forced compaction reason, got %q", out.Reason)
	}
}

func TestEngine_ForceCompactPreservesStrategyReason(t *testing.T) {
	engine := compaction.NewEngine(nil, reasonStrategy{reason: "strategy reason"}, nil)

	out, err := engine.ForceCompact(context.Background(), compaction.Input{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("ForceCompact: %v", err)
	}
	if out.Reason != "strategy reason" {
		t.Fatalf("expected strategy reason, got %q", out.Reason)
	}
}

func TestCloneMessages_DeepCopiesToolCalls(t *testing.T) {
	original := []llm.Message{
		{
			Role:    "assistant",
			Content: "calling tool",
			ToolCalls: []llm.ToolCall{
				{
					ID:   "call_1",
					Name: "lookup",
					Input: map[string]any{
						"query": "weather",
						"nested": map[string]any{
							"unit": "celsius",
						},
						"items": []any{"a", map[string]any{"key": "value"}},
					},
				},
			},
		},
	}

	cloned := compaction.CloneMessages(original)
	cloned[0].ToolCalls[0].Input["query"] = "mutated"
	cloned[0].ToolCalls[0].Input["nested"].(map[string]any)["unit"] = "fahrenheit"
	cloned[0].ToolCalls[0].Input["items"].([]any)[1].(map[string]any)["key"] = "changed"

	input := original[0].ToolCalls[0].Input
	if got := input["query"]; got != "weather" {
		t.Fatalf("expected original query to be unchanged, got %v", got)
	}
	if got := input["nested"].(map[string]any)["unit"]; got != "celsius" {
		t.Fatalf("expected original nested unit to be unchanged, got %v", got)
	}
	if got := input["items"].([]any)[1].(map[string]any)["key"]; got != "value" {
		t.Fatalf("expected original nested slice map to be unchanged, got %v", got)
	}
}

type reasonStrategy struct {
	reason string
}

func (s reasonStrategy) Name() string {
	return "reason"
}

func (s reasonStrategy) Compact(_ context.Context, input compaction.Input) (compaction.Output, error) {
	return compaction.Output{
		Messages: compaction.CloneMessages(input.Messages),
		Reason:   s.reason,
	}, nil
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
