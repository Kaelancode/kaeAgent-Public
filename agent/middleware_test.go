package agent

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/streaming"
)

func TestChain(t *testing.T) {
	var order []string

	base := func(_ context.Context, _ *Step) (*StepResult, error) {
		order = append(order, "handler")
		return &StepResult{TokensUsed: llm.Usage{}}, nil
	}

	mwA := func(next Handler) Handler {
		return func(ctx context.Context, step *Step) (*StepResult, error) {
			order = append(order, "A-before")
			result, err := next(ctx, step)
			order = append(order, "A-after")
			return result, err
		}
	}

	mwB := func(next Handler) Handler {
		return func(ctx context.Context, step *Step) (*StepResult, error) {
			order = append(order, "B-before")
			result, err := next(ctx, step)
			order = append(order, "B-after")
			return result, err
		}
	}

	chained := Chain(base, mwA, mwB)
	_, err := chained(context.Background(), &Step{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"A-before", "B-before", "handler", "B-after", "A-after"}
	if len(order) != len(expected) {
		t.Fatalf("expected %d calls, got %d: %v", len(expected), len(order), order)
	}
	for i, v := range expected {
		if order[i] != v {
			t.Errorf("position %d: expected %s, got %s", i, v, order[i])
		}
	}
}

func TestRetryMiddleware(t *testing.T) {
	attempts := 0
	base := func(_ context.Context, _ *Step) (*StepResult, error) {
		attempts++
		if attempts < 3 {
			return nil, fmt.Errorf("transient error")
		}
		return &StepResult{TokensUsed: llm.Usage{}}, nil
	}

	handler := Chain(base, RetryMiddleware(3, 1*time.Millisecond))
	_, err := handler(context.Background(), &Step{})
	if err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if attempts != 3 {
		t.Errorf("expected 3 attempts, got %d", attempts)
	}
}

func TestRetryMiddleware_Exhausted(t *testing.T) {
	base := func(_ context.Context, _ *Step) (*StepResult, error) {
		return nil, fmt.Errorf("always fails")
	}

	handler := Chain(base, RetryMiddleware(2, 1*time.Millisecond))
	_, err := handler(context.Background(), &Step{})
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
}

func TestCostGuardMiddleware(t *testing.T) {
	budget := streaming.NewBudget(streaming.BudgetConfig{MaxTokens: 100})
	budget.Add(90, 0)

	base := func(_ context.Context, _ *Step) (*StepResult, error) {
		return &StepResult{TokensUsed: llm.Usage{InputTokens: 20, OutputTokens: 0, TotalTokens: 20}}, nil
	}

	handler := Chain(base, CostGuardMiddleware(budget))
	_, err := handler(context.Background(), &Step{})
	if err == nil {
		t.Fatal("expected budget exceeded error")
	}
}
