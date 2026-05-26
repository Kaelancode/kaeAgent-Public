package turnwindow

import (
	"context"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/llm"
)

type Strategy struct {
	MaxTurns int
}

func New(maxTurns int) *Strategy {
	return &Strategy{MaxTurns: maxTurns}
}

func (s *Strategy) Name() string {
	return "turn_window"
}

func (s *Strategy) Compact(_ context.Context, input compaction.Input) (compaction.Output, error) {
	if s.MaxTurns <= 0 {
		return compaction.Output{Messages: cloneMessages(input.Messages)}, nil
	}

	systemMsgs, turns := splitTurns(input.Messages)
	if len(turns) <= s.MaxTurns {
		return compaction.Output{Messages: cloneMessages(input.Messages)}, nil
	}

	turns = turns[len(turns)-s.MaxTurns:]
	out := cloneMessages(systemMsgs)
	for _, turn := range turns {
		out = append(out, turn...)
	}

	return compaction.Output{
		Messages:  out,
		Compacted: len(out) != len(input.Messages),
	}, nil
}

func Factory(config map[string]any) (compaction.Strategy, error) {
	maxTurns := intValue(config, "max_turns", 0)
	if maxTurns == 0 {
		maxTurns = intValue(config, "max_history", 0)
	}
	return New(maxTurns), nil
}

func splitTurns(messages []llm.Message) ([]llm.Message, [][]llm.Message) {
	systemMsgs := make([]llm.Message, 0, len(messages))
	turns := make([][]llm.Message, 0)
	current := make([]llm.Message, 0)

	for _, m := range messages {
		if m.Role == "system" {
			systemMsgs = append(systemMsgs, m)
			continue
		}
		if m.Role == "user" {
			if len(current) > 0 {
				turns = append(turns, current)
			}
			current = []llm.Message{m}
			continue
		}
		if len(current) == 0 {
			current = []llm.Message{m}
			continue
		}
		current = append(current, m)
	}

	if len(current) > 0 {
		turns = append(turns, current)
	}

	return systemMsgs, turns
}

func cloneMessages(messages []llm.Message) []llm.Message {
	out := make([]llm.Message, len(messages))
	copy(out, messages)
	return out
}

func intValue(config map[string]any, key string, fallback int) int {
	if config == nil {
		return fallback
	}
	switch v := config[key].(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return fallback
	}
}
