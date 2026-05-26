package compaction

import (
	"context"

	"github.com/yourorg/agent-sdk/llm"
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
	before := cloneMessages(input.Messages)
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
	before := cloneMessages(input.Messages)
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
	out.TokensBefore = estimatedBefore
	out.TokensAfter = EstimatePromptTokens(out.Messages, input.Tools, e.estimator)
	return out, nil
}

func cloneMessages(messages []llm.Message) []llm.Message {
	out := make([]llm.Message, len(messages))
	copy(out, messages)
	return out
}
