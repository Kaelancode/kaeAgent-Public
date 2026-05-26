package compaction

import (
	"context"

	"github.com/yourorg/agent-sdk/llm"
)

type MaxTurnsTrigger struct {
	MaxTurns int
}

func (t MaxTurnsTrigger) Name() string {
	return "max_turns"
}

func (t MaxTurnsTrigger) ShouldCompact(_ context.Context, input Input) (bool, string, error) {
	if t.MaxTurns <= 0 {
		return false, "", nil
	}
	if countTurns(input.Messages) > t.MaxTurns {
		return true, "max turns exceeded", nil
	}
	return false, "", nil
}

type MaxMessagesTrigger struct {
	MaxMessages int
}

func (t MaxMessagesTrigger) Name() string {
	return "max_messages"
}

func (t MaxMessagesTrigger) ShouldCompact(_ context.Context, input Input) (bool, string, error) {
	if t.MaxMessages <= 0 {
		return false, "", nil
	}
	if len(input.Messages) > t.MaxMessages {
		return true, "max messages exceeded", nil
	}
	return false, "", nil
}

type MaxTokensTrigger struct {
	MaxTokens int
	Estimator TokenEstimator
}

func (t MaxTokensTrigger) Name() string {
	return "max_tokens"
}

func (t MaxTokensTrigger) ShouldCompact(_ context.Context, input Input) (bool, string, error) {
	if t.MaxTokens <= 0 {
		return false, "", nil
	}
	estimator := t.Estimator
	if EstimatePromptTokens(input.Messages, input.Tools, estimator) > t.MaxTokens {
		return true, "max tokens exceeded", nil
	}
	return false, "", nil
}

type AnyTrigger []Trigger

func (t AnyTrigger) Name() string {
	return "any"
}

func (t AnyTrigger) ShouldCompact(ctx context.Context, input Input) (bool, string, error) {
	for _, trigger := range t {
		if trigger == nil {
			continue
		}
		ok, reason, err := trigger.ShouldCompact(ctx, input)
		if err != nil {
			return false, "", err
		}
		if ok {
			return true, reason, nil
		}
	}
	return false, "", nil
}

func countTurns(messages []llm.Message) int {
	turns := 0
	for _, m := range messages {
		if m.Role == "user" {
			turns++
		}
	}
	return turns
}
