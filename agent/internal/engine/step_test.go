package engine

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

func TestExecuteStepReturnsTextResponseAndUsage(t *testing.T) {
	out, err := ExecuteStep(context.Background(), StepInput{
		SessionID: "session",
		RunID:     "run",
		Messages:  []llm.Message{{Role: "user", Content: "hello"}},
	}, Config{Model: "model", MaxTokens: 100}, Hooks{
		Complete: func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			if req.Model != "model" || req.Execution.SessionID != "session" {
				t.Fatalf("unexpected request: %+v", req)
			}
			return &llm.Response{
				Content: []llm.ContentBlock{{Type: "text", Text: "ok"}},
				Usage:   llm.Usage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("ExecuteStep: %v", err)
	}
	if out.Text != "ok" || len(out.ToolCalls) != 0 {
		t.Fatalf("unexpected step output: %+v", out)
	}
	if out.Usage.InputTokens != 2 || out.Usage.OutputTokens != 3 {
		t.Fatalf("unexpected usage: %+v", out.Usage)
	}
	if out.Request == nil || out.Response == nil {
		t.Fatalf("expected request and response on output: %+v", out)
	}
}

func TestExecuteStepReturnsToolCalls(t *testing.T) {
	out, err := ExecuteStep(context.Background(), StepInput{}, Config{}, Hooks{
		Complete: func(context.Context, *llm.Request) (*llm.Response, error) {
			return &llm.Response{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "call_1",
							Name:  "lookup",
							Input: map[string]any{"nested": map[string]any{"key": "value"}},
						},
					},
				},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("ExecuteStep: %v", err)
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].Name != "lookup" {
		t.Fatalf("unexpected tool calls: %+v", out.ToolCalls)
	}
	out.ToolCalls[0].Input["nested"].(map[string]any)["key"] = "mutated"
	got := out.Response.Content[0].ToolCall.Input["nested"].(map[string]any)["key"]
	if got != "value" {
		t.Fatalf("expected tool calls not to alias response input, got %v", got)
	}
}

func TestExecuteStepPropagatesCompleteError(t *testing.T) {
	want := errors.New("provider down")
	_, err := ExecuteStep(context.Background(), StepInput{}, Config{}, Hooks{
		Complete: func(context.Context, *llm.Request) (*llm.Response, error) {
			return nil, want
		},
	})
	if !errors.Is(err, want) {
		t.Fatalf("expected wrapped provider error, got %v", err)
	}
}

func TestExecuteStepRejectsMissingCompleteHook(t *testing.T) {
	_, err := ExecuteStep(context.Background(), StepInput{}, Config{}, Hooks{})
	if err == nil || !strings.Contains(err.Error(), "complete hook is nil") {
		t.Fatalf("expected nil complete hook error, got %v", err)
	}
}

func TestExecuteStepRejectsNilResponse(t *testing.T) {
	_, err := ExecuteStep(context.Background(), StepInput{}, Config{}, Hooks{
		Complete: func(context.Context, *llm.Request) (*llm.Response, error) {
			return nil, nil
		},
	})
	if err == nil || !strings.Contains(err.Error(), "nil response") {
		t.Fatalf("expected nil response error, got %v", err)
	}
}

func TestExtractResponseTextReturnsFirstTextBlock(t *testing.T) {
	text := ExtractResponseText(&llm.Response{
		Content: []llm.ContentBlock{
			{Type: "tool_call", ToolCall: &llm.ToolCall{Name: "lookup"}},
			{Type: "text", Text: "first"},
			{Type: "text", Text: "second"},
		},
	})
	if text != "first" {
		t.Fatalf("expected first text block, got %q", text)
	}
}
