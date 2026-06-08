package tokenlimit

import (
	"context"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/llm"
)

type Strategy struct {
	MaxTokens int
	Estimator compaction.TokenEstimator
}

func New(maxTokens int, estimator compaction.TokenEstimator) *Strategy {
	if estimator == nil {
		estimator = compaction.NewApproxTokenEstimator(compaction.DefaultCharsPerToken)
	}
	return &Strategy{
		MaxTokens: maxTokens,
		Estimator: estimator,
	}
}

func (s *Strategy) Name() string {
	return "token_limit"
}

func (s *Strategy) Compact(_ context.Context, input compaction.Input) (compaction.Output, error) {
	msgs := cloneMessages(input.Messages)
	if s.MaxTokens <= 0 {
		return compaction.Output{Messages: msgs}, nil
	}

	tokens := compaction.EstimatePromptTokens(msgs, input.Tools, s.Estimator)
	for tokens > s.MaxTokens && hasDroppableTurns(msgs) {
		msgs = dropOldestTurn(msgs)
		tokens = compaction.EstimatePromptTokens(msgs, input.Tools, s.Estimator)
	}

	if tokens > s.MaxTokens {
		return compaction.Output{}, fmt.Errorf("tokenlimit: compacted messages exceed max tokens: %d > %d", tokens, s.MaxTokens)
	}

	return compaction.Output{
		Messages:  msgs,
		Compacted: len(msgs) != len(input.Messages),
	}, nil
}

func Factory(config map[string]any) (compaction.Strategy, error) {
	maxTokens := intValue(config, "max_tokens", 0)
	return New(maxTokens, nil), nil
}

func cloneMessages(messages []llm.Message) []llm.Message {
	return compaction.CloneMessages(messages)
}

func hasDroppableTurns(messages []llm.Message) bool {
	return firstUserIndex(messages) >= 0
}

func dropOldestTurn(messages []llm.Message) []llm.Message {
	start := firstUserIndex(messages)
	if start < 0 {
		return messages
	}

	end := len(messages)
	for i := start + 1; i < len(messages); i++ {
		if messages[i].Role == "user" {
			end = i
			break
		}
	}

	out := make([]llm.Message, 0, len(messages)-(end-start))
	out = append(out, messages[:start]...)
	out = append(out, messages[end:]...)
	return out
}

func firstUserIndex(messages []llm.Message) int {
	for i, m := range messages {
		if m.Role == "user" {
			return i
		}
	}
	return -1
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
