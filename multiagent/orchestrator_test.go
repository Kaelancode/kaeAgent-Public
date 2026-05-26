package multiagent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/yourorg/agent-sdk/agent"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
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

type fakeStreamingProvider struct {
	text string
}

func (f *fakeStreamingProvider) Complete(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return &llm.Response{
		Content:      []llm.ContentBlock{{Type: "text", Text: f.text}},
		FinishReason: "stop",
	}, nil
}

func (f *fakeStreamingProvider) Stream(_ context.Context, _ *llm.Request) (<-chan llm.Event, error) {
	ch := make(chan llm.Event, 2)
	ch <- llm.Event{Kind: llm.EventText, Text: &llm.TextDelta{Content: f.text}}
	ch <- llm.Event{Kind: llm.EventDone}
	close(ch)
	return ch, nil
}

func (f *fakeStreamingProvider) Models(_ context.Context) ([]llm.ModelInfo, error) { return nil, nil }
func (f *fakeStreamingProvider) Name() string                                      { return "fake-stream" }

type failingStreamingProvider struct{}

func (f *failingStreamingProvider) Complete(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return &llm.Response{
		Content:      []llm.ContentBlock{{Type: "text", Text: "unexpected"}},
		FinishReason: "stop",
	}, nil
}

func (f *failingStreamingProvider) Stream(_ context.Context, _ *llm.Request) (<-chan llm.Event, error) {
	ch := make(chan llm.Event, 1)
	ch <- llm.Event{Kind: llm.EventError, Err: errors.New("stream failed")}
	close(ch)
	return ch, nil
}

func (f *failingStreamingProvider) Models(_ context.Context) ([]llm.ModelInfo, error) {
	return nil, nil
}
func (f *failingStreamingProvider) Name() string { return "failing-stream" }

func TestOrchestratorRunAgentUsesAgentDefinition(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "writer",
			Model:        "child-model",
			SystemPrompt: "write carefully",
		}),
		Name:        "writer",
		Description: "writer agent",
		Tags:        []string{"write"},
	})

	orch := NewOrchestrator(&fakeProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "child result"}},
				FinishReason: "stop",
			},
		},
	}, router)

	out, err := orch.RunAgent(context.Background(), "writer", "draft")
	if err != nil {
		t.Fatalf("RunAgent: %v", err)
	}
	if out != "child result" {
		t.Fatalf("expected child result, got %q", out)
	}
}

func TestOrchestratorConsultUsesExplicitRequest(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "researcher",
			Model:        "child-model",
			SystemPrompt: "research carefully",
		}),
		Name: "researcher",
	})

	orch := NewOrchestrator(&fakeProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "consulted"}},
				FinishReason: "stop",
			},
		},
	}, router)

	out, err := orch.Consult(context.Background(), ConsultRequest{
		SessionID: "sess_parent",
		AgentName: "researcher",
		Input:     "find sources",
		Context: []llm.Message{
			{Role: "user", Content: "prior relevant context"},
		},
		Metadata: map[string]string{"active_agent": "root"},
	})
	if err != nil {
		t.Fatalf("Consult: %v", err)
	}
	if out != "consulted" {
		t.Fatalf("expected consulted, got %q", out)
	}
}

func TestOrchestratorTransferSetsActiveAgentMetadata(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "billing",
			Model:        "child-model",
			SystemPrompt: "billing specialist",
		}),
		Name: "billing",
	})

	parent := agent.NewRuntime(agent.RuntimeConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "root",
			Model:        "root-model",
			SystemPrompt: "root agent",
			Subagents:    []string{"billing"},
		}),
	})

	orch := NewOrchestrator(&fakeProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "billing reply"}},
				FinishReason: "stop",
			},
		},
	}, router)

	out, err := orch.Transfer(context.Background(), TransferRequest{
		Runtime:   parent,
		AgentName: "billing",
		Input:     "handle refund",
		Context: []llm.Message{
			{Role: "user", Content: "customer asks for refund"},
		},
	})
	if err != nil {
		t.Fatalf("Transfer: %v", err)
	}
	if out != "billing reply" {
		t.Fatalf("expected billing reply, got %q", out)
	}

	snap := parent.SessionSnapshot()
	if snap.Metadata[ActiveAgentMetadataKey] != "billing" {
		t.Fatalf("expected active agent metadata to be billing, got %q", snap.Metadata[ActiveAgentMetadataKey])
	}
}

func TestOrchestratorActiveAgentFallsBackToRoot(t *testing.T) {
	rootAgent := "root"
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{Name: "billing"}),
		Name:  "billing",
	})

	orch := NewOrchestrator(&fakeProvider{}, router)
	rt := agent.NewRuntime(agent.RuntimeConfig{
		Agent: agent.NewAgent(agent.AgentConfig{Name: rootAgent}),
	})

	if got := orch.ActiveAgent(rt, rootAgent); got != rootAgent {
		t.Fatalf("expected root fallback on missing metadata, got %q", got)
	}

	rt.SetSessionMetadata(ActiveAgentMetadataKey, "unknown")
	if got := orch.ActiveAgent(rt, rootAgent); got != rootAgent {
		t.Fatalf("expected root fallback on unknown metadata, got %q", got)
	}

	rt.SetSessionMetadata(ActiveAgentMetadataKey, "billing")
	if got := orch.ActiveAgent(rt, rootAgent); got != "billing" {
		t.Fatalf("expected active agent billing, got %q", got)
	}
}

func TestOrchestratorConsultStream(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "researcher",
			Model:        "child-model",
			SystemPrompt: "research carefully",
		}),
		Name: "researcher",
	})

	orch := NewOrchestrator(&fakeStreamingProvider{text: "streamed consult"}, router)
	ch, err := orch.ConsultStream(context.Background(), ConsultRequest{
		SessionID: "sess_parent",
		AgentName: "researcher",
		Input:     "find sources",
	})
	if err != nil {
		t.Fatalf("ConsultStream: %v", err)
	}

	events, streamErr := streaming.Collect(context.Background(), ch)
	if streamErr != nil {
		t.Fatalf("Collect: %v", streamErr)
	}

	var final string
	for _, e := range events {
		if e.Kind == streaming.EventFinalText {
			final = e.Final.Content
		}
	}
	if final != "streamed consult" {
		t.Fatalf("expected final streamed consult text, got %q", final)
	}
}

func TestOrchestratorTransferStreamSetsActiveAgentMetadata(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "billing",
			Model:        "child-model",
			SystemPrompt: "billing specialist",
		}),
		Name: "billing",
	})

	parent := agent.NewRuntime(agent.RuntimeConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "root",
			Model:        "root-model",
			SystemPrompt: "root agent",
			Subagents:    []string{"billing"},
		}),
	})

	orch := NewOrchestrator(&fakeStreamingProvider{text: "billing stream"}, router)
	ch, err := orch.TransferStream(context.Background(), TransferRequest{
		Runtime:   parent,
		AgentName: "billing",
		Input:     "handle refund",
		Context: []llm.Message{
			{Role: "user", Content: "customer asks for refund"},
		},
	})
	if err != nil {
		t.Fatalf("TransferStream: %v", err)
	}
	if got := parent.SessionSnapshot().Metadata[ActiveAgentMetadataKey]; got != "" {
		t.Fatalf("did not expect active agent metadata before stream completes, got %q", got)
	}

	events, streamErr := streaming.Collect(context.Background(), ch)
	if streamErr != nil {
		t.Fatalf("Collect: %v", streamErr)
	}

	snap := parent.SessionSnapshot()
	if snap.Metadata[ActiveAgentMetadataKey] != "billing" {
		t.Fatalf("expected active agent metadata to be billing, got %q", snap.Metadata[ActiveAgentMetadataKey])
	}

	var final string
	for _, e := range events {
		if e.Kind == streaming.EventFinalText {
			final = e.Final.Content
		}
	}
	if final != "billing stream" {
		t.Fatalf("expected final billing stream text, got %q", final)
	}
}

func TestOrchestratorTransferStreamDoesNotSetActiveAgentMetadataOnError(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "billing",
			Model:        "child-model",
			SystemPrompt: "billing specialist",
		}),
		Name: "billing",
	})

	parent := agent.NewRuntime(agent.RuntimeConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "root",
			Model:        "root-model",
			SystemPrompt: "root agent",
			Subagents:    []string{"billing"},
		}),
	})

	orch := NewOrchestrator(&failingStreamingProvider{}, router)
	ch, err := orch.TransferStream(context.Background(), TransferRequest{
		Runtime:   parent,
		AgentName: "billing",
		Input:     "handle refund",
		Context: []llm.Message{
			{Role: "user", Content: "customer asks for refund"},
		},
	})
	if err != nil {
		t.Fatalf("TransferStream: %v", err)
	}

	_, streamErr := streaming.Collect(context.Background(), ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), "stream failed") {
		t.Fatalf("expected stream failure, got %v", streamErr)
	}

	snap := parent.SessionSnapshot()
	if snap.Metadata[ActiveAgentMetadataKey] != "" {
		t.Fatalf("did not expect active agent metadata after failed stream, got %q", snap.Metadata[ActiveAgentMetadataKey])
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

func TestWorkflowAgentToolUsesAgentOwnedTools(t *testing.T) {
	child := agent.NewAgent(agent.AgentConfig{
		Name:         "worker",
		Model:        "child-model",
		SystemPrompt: "use tools when needed",
	})
	called := false
	child.RegisterTool(tools.ToolDef{
		Name: "lookup",
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

func TestAgentToolCompatibilityWrapper(t *testing.T) {
	child := agent.NewAgent(agent.AgentConfig{
		Name:         "worker",
		Model:        "child-model",
		SystemPrompt: "answer directly",
	})
	provider := &fakeProvider{
		responses: []*llm.Response{{
			Content:      []llm.ContentBlock{{Type: "text", Text: "done"}},
			FinishReason: "stop",
		}},
	}

	tool := AgentTool(AgentConfig{
		Agent:       child,
		Name:        "worker",
		Description: "worker agent",
	}, provider)

	if tool.Name != "agent_worker" {
		t.Fatalf("expected compatibility tool name agent_worker, got %q", tool.Name)
	}
	result, err := tool.Handler(context.Background(), map[string]any{"message": "help"})
	if err != nil {
		t.Fatalf("AgentTool compatibility handler: %v", err)
	}
	if result != "done" {
		t.Fatalf("expected done, got %v", result)
	}
}

func TestRegisterWorkflowAgentToolsRegistersWorkflowTools(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "outline",
			Model:        "child-model",
			SystemPrompt: "outline",
		}),
		Name:        "outline",
		Description: "outline agent",
		Tags:        []string{"workflow"},
	})

	orch := NewOrchestrator(&fakeProvider{}, router)
	orch.RegisterWorkflowAgentTools()

	tool, ok := orch.ToolRegistry().Get("agent_outline")
	if !ok {
		t.Fatal("expected workflow tool agent_outline to be registered")
	}
	if tool.Description != "outline agent" {
		t.Fatalf("expected workflow tool description, got %q", tool.Description)
	}
}

func TestRegisterAgentToolsCompatibilityAlias(t *testing.T) {
	router := NewRouter()
	router.Register(AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "outline",
			Model:        "child-model",
			SystemPrompt: "outline",
		}),
		Name: "outline",
	})

	orch := NewOrchestrator(&fakeProvider{}, router)
	orch.RegisterAgentTools()

	if _, ok := orch.ToolRegistry().Get("agent_outline"); !ok {
		t.Fatal("expected compatibility alias to register workflow tool")
	}
}
