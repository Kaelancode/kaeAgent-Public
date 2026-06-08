package compaction

import (
	"context"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

type Input struct {
	SessionID string
	Messages  []llm.Message
	Tools     []llm.ToolDef
	Metadata  map[string]string
}

type Output struct {
	Messages     []llm.Message
	Compacted    bool
	Reason       string
	Metadata     map[string]string
	TokensBefore int
	TokensAfter  int
}

const forcedCompactionReason = "forced compaction"

type Compactor interface {
	Compact(ctx context.Context, input Input) (Output, error)
	ForceCompact(ctx context.Context, input Input) (Output, error)
}

type Strategy interface {
	Name() string
	Compact(ctx context.Context, input Input) (Output, error)
}

type Trigger interface {
	Name() string
	ShouldCompact(ctx context.Context, input Input) (bool, string, error)
}

type Engine struct {
	trigger   Trigger
	strategy  Strategy
	estimator TokenEstimator
}

func NewEngine(trigger Trigger, strategy Strategy, estimator TokenEstimator) *Engine {
	if estimator == nil {
		estimator = NewApproxTokenEstimator(DefaultCharsPerToken)
	}
	return &Engine{
		trigger:   trigger,
		strategy:  strategy,
		estimator: estimator,
	}
}

func (e *Engine) Compact(ctx context.Context, input Input) (Output, error) {
	before := CloneMessages(input.Messages)
	estimatedBefore := EstimatePromptTokens(before, input.Tools, e.estimator)

	if e.trigger == nil || e.strategy == nil {
		return Output{
			Messages:     before,
			TokensBefore: estimatedBefore,
			TokensAfter:  estimatedBefore,
		}, nil
	}

	shouldCompact, reason, err := e.trigger.ShouldCompact(ctx, input)
	if err != nil {
		return Output{}, err
	}
	if !shouldCompact {
		return Output{
			Messages:     before,
			TokensBefore: estimatedBefore,
			TokensAfter:  estimatedBefore,
		}, nil
	}

	out, err := e.strategy.Compact(ctx, input)
	if err != nil {
		return Output{}, err
	}
	if out.Messages == nil {
		out.Messages = before
	}
	if out.Reason == "" {
		out.Reason = reason
	}
	out.TokensBefore = estimatedBefore
	out.TokensAfter = EstimatePromptTokens(out.Messages, input.Tools, e.estimator)
	return out, nil
}

func (e *Engine) ForceCompact(ctx context.Context, input Input) (Output, error) {
	before := CloneMessages(input.Messages)
	estimatedBefore := EstimatePromptTokens(before, input.Tools, e.estimator)

	if e.strategy == nil {
		return Output{
			Messages:     before,
			TokensBefore: estimatedBefore,
			TokensAfter:  estimatedBefore,
		}, nil
	}

	out, err := e.strategy.Compact(ctx, input)
	if err != nil {
		return Output{}, err
	}
	if out.Messages == nil {
		out.Messages = before
	}
	if out.Reason == "" {
		out.Reason = forcedCompactionReason
	}
	out.TokensBefore = estimatedBefore
	out.TokensAfter = EstimatePromptTokens(out.Messages, input.Tools, e.estimator)
	return out, nil
}

func CloneMessages(messages []llm.Message) []llm.Message {
	if messages == nil {
		return nil
	}

	out := make([]llm.Message, len(messages))
	for i, msg := range messages {
		out[i] = CloneMessage(msg)
	}
	return out
}

func CloneMessage(msg llm.Message) llm.Message {
	out := msg
	if len(msg.ToolCalls) > 0 {
		out.ToolCalls = make([]llm.ToolCall, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			out.ToolCalls[i] = llm.ToolCall{
				ID:    tc.ID,
				Name:  tc.Name,
				Input: cloneMap(tc.Input),
			}
		}
	}
	return out
}

func cloneMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}

	out := make(map[string]any, len(input))
	for k, v := range input {
		out[k] = cloneValue(v)
	}
	return out
}

func cloneSlice(input []any) []any {
	if input == nil {
		return nil
	}

	out := make([]any, len(input))
	for i, v := range input {
		out[i] = cloneValue(v)
	}
	return out
}

func cloneValue(v any) any {
	switch typed := v.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		return cloneSlice(typed)
	default:
		return typed
	}
}
