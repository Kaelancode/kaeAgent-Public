package agent

import (
	"context"
	"testing"

	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

func TestAgentSessionConfigAndTools(t *testing.T) {
	temp := float32(0.25)
	budget := &streaming.BudgetConfig{MaxTokens: 1234}
	a := NewAgent(AgentConfig{
		Name:         "triage",
		Model:        "gpt-test",
		SystemPrompt: "route and respond",
		MaxTokens:    512,
		Temperature:  &temp,
		TrimStrategy: TrimSlidingWindow,
		MaxHistory:   12,
		TokenBudget:  8192,
		BudgetConfig: budget,
		Subagents:    []string{"billing", "refund"},
		MaxSteps:     7,
	})

	a.RegisterTool(tools.ToolDef{Name: "lookup"})
	a.AddSubagent("support")

	cfg := a.SessionConfig()
	if cfg.Model != "gpt-test" || cfg.SystemPrompt != "route and respond" {
		t.Fatalf("unexpected session config: %+v", cfg)
	}
	if cfg.Temperature == nil || *cfg.Temperature != temp {
		t.Fatalf("expected temperature %v, got %#v", temp, cfg.Temperature)
	}
	if a.Name() != "triage" {
		t.Fatalf("expected agent name triage, got %q", a.Name())
	}
	if a.MaxSteps() != 7 {
		t.Fatalf("expected max steps 7, got %d", a.MaxSteps())
	}
	if names := a.ToolRegistry().Names(); len(names) != 1 || names[0] != "lookup" {
		t.Fatalf("expected tool registry to contain lookup, got %v", names)
	}
	subagents := a.Subagents()
	if len(subagents) != 3 {
		t.Fatalf("expected 3 subagents, got %v", subagents)
	}
}

func TestRuntimeUsesAgentDefaultsAndTools(t *testing.T) {
	agentDef := NewAgent(AgentConfig{
		Name:         "assistant",
		Model:        "agent-model",
		SystemPrompt: "agent instructions",
		MaxTokens:    256,
	})
	agentDef.RegisterTool(tools.ToolDef{
		Name: "from_agent",
		Handler: func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		},
	})

	session := NewSession(SessionConfig{})
	explicitTools := tools.NewRegistry()
	explicitTools.Register(tools.ToolDef{
		Name: "from_runtime",
		Handler: func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		},
	})

	rt := NewRuntime(RuntimeConfig{
		Agent:   agentDef,
		Session: session,
		Tools:   explicitTools,
	})

	snap := rt.SessionSnapshot()
	if snap.Config.Model != "agent-model" {
		t.Fatalf("expected model inherited from agent, got %q", snap.Config.Model)
	}
	if snap.Config.SystemPrompt != "agent instructions" {
		t.Fatalf("expected system prompt inherited from agent, got %q", snap.Config.SystemPrompt)
	}
	names := rt.tools.Names()
	if len(names) != 2 {
		t.Fatalf("expected merged agent/runtime tools, got %v", names)
	}
	if _, ok := rt.tools.Get("from_agent"); !ok {
		t.Fatal("expected agent-owned tool in runtime registry")
	}
	if _, ok := rt.tools.Get("from_runtime"); !ok {
		t.Fatal("expected runtime-owned tool in runtime registry")
	}
}
