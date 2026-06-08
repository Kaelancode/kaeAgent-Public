package agent

import (
	"context"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestEngineTurnInputCapturesRunState(t *testing.T) {
	temp := float32(0.4)
	root := NewAgent(AgentConfig{
		Name:         "coordinator",
		Model:        "agent-model",
		SystemPrompt: "coordinate",
		MaxTokens:    900,
		Temperature:  &temp,
		Subagents:    []string{"billing"},
		MaxSteps:     5,
	})
	root.RegisterTool(tools.ToolDef{
		Name:    "agent_lookup",
		Schema:  testTool("agent_lookup").Schema,
		Tags:    []string{"agent"},
		Handler: func(context.Context, map[string]any) (any, error) { return "ok", nil },
	})

	explicitTools := tools.NewRegistry()
	explicitTools.Register(tools.ToolDef{
		Name:    "runtime_lookup",
		Schema:  testTool("runtime_lookup").Schema,
		Tags:    []string{"runtime"},
		Handler: func(context.Context, map[string]any) (any, error) { return "ok", nil },
	})

	session := NewSession(SessionConfig{
		Model:        "session-model",
		SystemPrompt: "session prompt",
		MaxTokens:    700,
		Temperature:  &temp,
		BudgetConfig: &streaming.BudgetConfig{
			MaxTokens:          1000,
			MaxCostUSD:         2.5,
			CostPerInputToken:  0.01,
			CostPerOutputToken: 0.02,
		},
	})
	session.SetMeta("active_agent", "coordinator")
	session.Budget.Add(10, 5)

	rt := NewRuntime(RuntimeConfig{
		Provider:           &fakeProvider{},
		Agent:              root,
		Session:            session,
		Tools:              explicitTools,
		MaxSteps:           8,
		MaxToolConcurrency: 3,
		ModelContextLimit:  4096,
		OutputTokenReserve: 512,
		UserID:             "user-1",
		AgentID:            "coordinator",
	})
	rt.AppendConversationMessage(llm.Message{
		Role:    "assistant",
		Content: "using a tool",
		ToolCalls: []llm.ToolCall{
			{
				ID:    "call_1",
				Name:  "agent_lookup",
				Input: map[string]any{"nested": map[string]any{"key": "value"}},
			},
		},
	})

	input := rt.newRunExecutor().engineTurnInput("hello")

	if input.UserMessage != "hello" {
		t.Fatalf("expected user message to be retained, got %q", input.UserMessage)
	}
	if input.State.SessionID != session.ID || input.State.UserID != "user-1" || input.State.AgentID != "coordinator" {
		t.Fatalf("unexpected engine state identity: %+v", input.State)
	}
	if input.State.ActiveAgent.Name != "coordinator" || input.State.ActiveAgent.MaxSteps != 5 {
		t.Fatalf("unexpected active agent view: %+v", input.State.ActiveAgent)
	}
	if input.State.RootAgent.Name != "coordinator" {
		t.Fatalf("unexpected root agent view: %+v", input.State.RootAgent)
	}
	if input.Config.Model != "session-model" || input.Config.MaxTokens != 700 || input.Config.MaxSteps != 8 {
		t.Fatalf("unexpected engine config: %+v", input.Config)
	}
	if input.Config.Temperature == nil || *input.Config.Temperature != temp {
		t.Fatalf("expected temperature %v, got %#v", temp, input.Config.Temperature)
	}
	if input.Config.MaxToolConcurrency != 3 || input.Config.ModelContextLimit != 4096 || input.Config.OutputTokenReserve != 512 {
		t.Fatalf("unexpected execution limits: %+v", input.Config)
	}
	if input.State.Budget.MaxTokens != 1000 || input.State.Budget.TotalInput != 10 || input.State.Budget.TotalOutput != 5 {
		t.Fatalf("unexpected budget snapshot: %+v", input.State.Budget)
	}
	if got := engineToolNames(input.State.ActiveAgent.Tools); !sameStrings(got, []string{"agent_lookup", "runtime_lookup"}) {
		t.Fatalf("expected run-local tools, got %v", got)
	}
}

func TestEngineStateDoesNotAliasRuntimeState(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	rt.SetSessionMetadata("scope", "original")
	rt.AppendConversationMessage(llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{
			{
				ID:    "call_1",
				Name:  "lookup",
				Input: map[string]any{"nested": map[string]any{"key": "original"}},
			},
		},
	})

	state := engineStateFromRunState(rt.captureRunState())
	state.Metadata["scope"] = "mutated"
	state.Messages[0].ToolCalls[0].Input["nested"].(map[string]any)["key"] = "mutated"

	if got := rt.SessionSnapshot().Metadata["scope"]; got != "original" {
		t.Fatalf("expected runtime metadata not to alias engine state, got %q", got)
	}
	msgs := rt.ConversationMessages()
	got := msgs[0].ToolCalls[0].Input["nested"].(map[string]any)["key"]
	if got != "original" {
		t.Fatalf("expected runtime messages not to alias engine state, got %v", got)
	}
}

func engineToolNames(defs []tools.ToolDef) []string {
	out := make([]string, len(defs))
	for i, def := range defs {
		out[i] = def.Name
	}
	return out
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
