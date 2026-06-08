package agent

import (
	"context"
	"testing"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestGolden_ParallelToolResultsPreserveModelOrder(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "call_slow",
							Name:  "slow_lookup",
							Input: map[string]any{"key": "first"},
						},
					},
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "call_fast",
							Name:  "fast_lookup",
							Input: map[string]any{"key": "second"},
						},
					},
				},
				FinishReason: "tool_calls",
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "done"}},
				FinishReason: "stop",
			},
		},
	}

	registry := tools.NewRegistry()
	registry.Register(testToolWithHandler("slow_lookup", func(context.Context, map[string]any) (any, error) {
		time.Sleep(25 * time.Millisecond)
		return "slow result", nil
	}))
	registry.Register(testToolWithHandler("fast_lookup", func(context.Context, map[string]any) (any, error) {
		return "fast result", nil
	}))

	rt := NewRuntime(RuntimeConfig{
		Provider:           provider,
		Session:            NewSession(SessionConfig{Model: "fake-model"}),
		Tools:              registry,
		Dispatcher:         tools.NewDispatcher(registry),
		MaxToolConcurrency: 2,
	})

	out, err := rt.Run(context.Background(), "run tools")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "done" {
		t.Fatalf("expected done, got %q", out)
	}

	if len(provider.requests) != 2 {
		t.Fatalf("expected two provider requests, got %d", len(provider.requests))
	}

	secondReq := provider.requests[1]
	if len(secondReq.Messages) < 4 {
		t.Fatalf("expected continuation request to include tool results, got %+v", secondReq.Messages)
	}

	assertGoldenToolResultOrder(t, secondReq.Messages[len(secondReq.Messages)-2:])

	msgs := rt.ConversationMessages()
	if len(msgs) < 4 {
		t.Fatalf("expected runtime conversation to include tool results, got %+v", msgs)
	}
	assertGoldenToolResultOrder(t, msgs[len(msgs)-3:len(msgs)-1])
}

func assertGoldenToolResultOrder(t *testing.T, msgs []llm.Message) {
	t.Helper()

	if len(msgs) != 2 {
		t.Fatalf("expected two tool messages, got %+v", msgs)
	}
	if msgs[0].Role != "tool" || msgs[1].Role != "tool" {
		t.Fatalf("expected tool messages, got %+v", msgs)
	}
	if msgs[0].ToolCallID != "call_slow" || msgs[0].Name != "slow_lookup" || msgs[0].Content != "slow result" {
		t.Fatalf("expected first tool response to stay first, got %+v", msgs[0])
	}
	if msgs[1].ToolCallID != "call_fast" || msgs[1].Name != "fast_lookup" || msgs[1].Content != "fast result" {
		t.Fatalf("expected second tool response to stay second, got %+v", msgs[1])
	}
}
