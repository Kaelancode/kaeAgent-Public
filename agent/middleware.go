package agent

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

type Step struct {
	SessionID    string
	RunID        string
	StepIndex    int
	Messages     []llm.Message
	AvailTools   []tools.ToolDef
	ProviderName string
	UserID       string
	AgentID      string
	AgentName    string
}

type StepResult struct {
	Response   *llm.Response
	ToolCalls  []tools.ToolCall
	Transfer   *TransferStep
	TokensUsed llm.Usage
}

type Handler func(ctx context.Context, step *Step) (*StepResult, error)

type Middleware func(Handler) Handler

func Chain(h Handler, mw ...Middleware) Handler {
	for i := len(mw) - 1; i >= 0; i-- {
		h = mw[i](h)
	}
	return h
}

func RetryMiddleware(maxAttempts int, backoff time.Duration) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, step *Step) (*StepResult, error) {
			var lastErr error
			for attempt := 0; attempt < maxAttempts; attempt++ {
				if attempt > 0 {
					wait := time.Duration(float64(backoff) * math.Pow(2, float64(attempt-1)))
					select {
					case <-ctx.Done():
						return nil, fmt.Errorf("middleware: retry cancelled: %w", ctx.Err())
					case <-time.After(wait):
					}
				}
				result, err := next(ctx, step)
				if err == nil {
					return result, nil
				}
				lastErr = err
			}
			return nil, fmt.Errorf("middleware: retry exhausted after %d attempts: %w", maxAttempts, lastErr)
		}
	}
}

func CostGuardMiddleware(budget *streaming.Budget) Middleware {
	return func(next Handler) Handler {
		return func(ctx context.Context, step *Step) (*StepResult, error) {
			if err := budget.Check(); err != nil {
				return nil, fmt.Errorf("middleware: cost guard pre-check: %w", err)
			}

			result, err := next(ctx, step)
			if err != nil {
				return nil, err
			}

			budget.Add(result.TokensUsed.InputTokens, result.TokensUsed.OutputTokens)
			if err := budget.Check(); err != nil {
				return result, fmt.Errorf("middleware: cost guard post-check: %w", err)
			}
			return result, nil
		}
	}
}

type StreamingStep struct {
	SessionID    string
	RunID        string
	StepIndex    int
	Messages     []llm.Message
	AvailTools   []tools.ToolDef
	ProviderName string
	UserID       string
	AgentID      string
	AgentName    string
}

type StreamingStepResult struct {
	Response     *llm.Response
	ToolCalls    []tools.ToolCall
	Transfer     *TransferStep
	TokensUsed   llm.Usage
	StreamedText string
}

type StreamingHandler func(ctx context.Context, step *StreamingStep, out chan<- streaming.Event) (*StreamingStepResult, error)

type StreamingMiddleware func(StreamingHandler) StreamingHandler

func ChainStreaming(h StreamingHandler, mw ...StreamingMiddleware) StreamingHandler {
	for i := len(mw) - 1; i >= 0; i-- {
		h = mw[i](h)
	}
	return h
}

func CostGuardStreamingMiddleware(budget *streaming.Budget) StreamingMiddleware {
	return func(next StreamingHandler) StreamingHandler {
		return func(ctx context.Context, step *StreamingStep, out chan<- streaming.Event) (*StreamingStepResult, error) {
			if err := budget.Check(); err != nil {
				return nil, fmt.Errorf("middleware: cost guard pre-check: %w", err)
			}

			result, err := next(ctx, step, out)
			if err != nil {
				return nil, err
			}

			budget.Add(result.TokensUsed.InputTokens, result.TokensUsed.OutputTokens)
			if err := budget.Check(); err != nil {
				return result, fmt.Errorf("middleware: cost guard post-check: %w", err)
			}
			return result, nil
		}
	}
}

func RetryStreamingMiddleware(maxAttempts int, backoff time.Duration) StreamingMiddleware {
	return func(next StreamingHandler) StreamingHandler {
		return func(ctx context.Context, step *StreamingStep, out chan<- streaming.Event) (*StreamingStepResult, error) {
			var lastErr error
			for attempt := 0; attempt < maxAttempts; attempt++ {
				if attempt > 0 {
					wait := time.Duration(float64(backoff) * math.Pow(2, float64(attempt-1)))
					select {
					case <-ctx.Done():
						return nil, fmt.Errorf("middleware: retry cancelled: %w", ctx.Err())
					case <-time.After(wait):
					}
				}
				result, err := next(ctx, step, out)
				if err == nil {
					return result, nil
				}
				lastErr = err
			}
			return nil, fmt.Errorf("middleware: retry exhausted after %d attempts: %w", maxAttempts, lastErr)
		}
	}
}
