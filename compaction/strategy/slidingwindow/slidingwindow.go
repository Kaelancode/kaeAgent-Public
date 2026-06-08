package slidingwindow

import (
	"context"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/llm"
)

type Strategy struct {
	MaxMessages int
}

func New(maxMessages int) *Strategy {
	return &Strategy{MaxMessages: maxMessages}
}

func (s *Strategy) Name() string {
	return "sliding_window"
}

func (s *Strategy) Compact(_ context.Context, input compaction.Input) (compaction.Output, error) {
	if s.MaxMessages <= 0 || len(input.Messages) <= s.MaxMessages {
		return compaction.Output{
			Messages: compaction.CloneMessages(input.Messages),
		}, nil
	}

	var systemMsgs []llm.Message
	var otherMsgs []llm.Message
	for _, m := range input.Messages {
		if m.Role == "system" {
			systemMsgs = append(systemMsgs, m)
			continue
		}
		otherMsgs = append(otherMsgs, m)
	}

	if len(systemMsgs) > s.MaxMessages {
		return compaction.Output{}, fmt.Errorf(
			"slidingwindow: MaxMessages=%d is smaller than system message count %d",
			s.MaxMessages,
			len(systemMsgs),
		)
	}

	keep := s.MaxMessages - len(systemMsgs)
	if keep < 0 {
		keep = 0
	}
	if len(otherMsgs) > keep {
		otherMsgs = otherMsgs[len(otherMsgs)-keep:]
	}

	out := append(compaction.CloneMessages(systemMsgs), compaction.CloneMessages(otherMsgs)...)
	return compaction.Output{
		Messages:  out,
		Compacted: len(out) != len(input.Messages),
	}, nil
}

func Factory(config map[string]any) (compaction.Strategy, error) {
	maxMessages := intValue(config, "max_messages", 0)
	return New(maxMessages), nil
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
