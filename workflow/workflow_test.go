package workflow

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/schema"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

type fakeProvider struct {
	responses []*llm.Response
	callIdx   int
}

func (f *fakeProvider) Complete(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	if f.callIdx >= len(f.responses) {
		return &llm.Response{
			Content:      []llm.ContentBlock{{Type: "text", Text: "done"}},
			FinishReason: "stop",
		}, nil
	}
	resp := f.responses[f.callIdx]
	f.callIdx++
	return resp, nil
}

func (f *fakeProvider) Stream(_ context.Context, _ *llm.Request) (<-chan llm.Event, error) {
	ch := make(chan llm.Event)
	close(ch)
	return ch, nil
}

func (f *fakeProvider) Models(_ context.Context) ([]llm.ModelInfo, error) { return nil, nil }
func (f *fakeProvider) Name() string                                      { return "fake" }

func TestWorkflowAgentToolUsesAgentOwnedTools(t *testing.T) {
	child := agent.NewAgent(agent.AgentConfig{
		Name:         "worker",
		Model:        "child-model",
		SystemPrompt: "use tools when needed",
	})
	called := false
	child.RegisterTool(tools.ToolDef{
		Name:   "lookup",
		Schema: &schema.Schema{Type: "object"},
		Handler: func(_ context.Context, _ map[string]any) (any, error) {
			called = true
			return map[string]any{"ok": true}, nil
		},
	})

	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{{
					Type: "tool_call",
					ToolCall: &llm.ToolCall{
						ID:    "call_1",
						Name:  "lookup",
						Input: map[string]any{},
					},
				}},
				FinishReason: "tool_calls",
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "done"}},
				FinishReason: "stop",
			},
		},
	}

	tool := WorkflowAgentTool(AgentConfig{
		Agent:       child,
		Name:        "worker",
		Description: "worker agent",
		Tags:        []string{"delegate"},
	}, provider)

	result, err := tool.Handler(context.Background(), map[string]any{"message": "help"})
	if err != nil {
		t.Fatalf("WorkflowAgentTool handler: %v", err)
	}
	if !called {
		t.Fatal("expected agent-owned tool to be executed")
	}
	if result != "done" {
		t.Fatalf("expected done, got %v", result)
	}
}

func TestJoinAllDetailedCancelsSiblingsAndReturnsPartialResults(t *testing.T) {
	started := make(chan struct{}, 1)
	block := make(chan struct{})

	results, err := JoinAllDetailed(context.Background(), map[string]func(context.Context) (string, error){
		"fast_fail": func(context.Context) (string, error) {
			return "", errors.New("boom")
		},
		"blocked": func(ctx context.Context) (string, error) {
			started <- struct{}{}
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-block:
				return "unexpected", nil
			}
		},
	})
	if err == nil || !strings.Contains(err.Error(), "fast_fail") {
		t.Fatalf("expected first failure to mention fast_fail, got %v", err)
	}
	<-started
	blocked, ok := results["blocked"]
	if !ok {
		t.Fatalf("expected blocked result to be present")
	}
	if blocked.Err == nil {
		t.Fatalf("expected blocked sibling to be cancelled")
	}
}
