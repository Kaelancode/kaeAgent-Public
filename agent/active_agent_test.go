package agent

import (
	"context"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/schema"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestResolveSessionAgentFallsBackToRoot(t *testing.T) {
	root := NewAgent(AgentConfig{Name: "root", Model: "root-model"})
	reg := NewRegistry()
	reg.Register(NewAgent(AgentConfig{Name: "billing", Model: "billing-model"}))

	missing := SessionSnapshot{Metadata: map[string]string{}}
	if got := ResolveSessionAgent(root, missing, reg); got != root {
		t.Fatalf("expected root on missing metadata, got %v", got)
	}

	unknown := SessionSnapshot{Metadata: map[string]string{ActiveAgentMetadataKey: "unknown"}}
	if got := ResolveSessionAgent(root, unknown, reg); got != root {
		t.Fatalf("expected root on unknown metadata, got %v", got)
	}

	known := SessionSnapshot{Metadata: map[string]string{ActiveAgentMetadataKey: "billing"}}
	got := ResolveSessionAgent(root, known, reg)
	if got == nil || got.Name() != "billing" {
		t.Fatalf("expected billing on known metadata, got %v", got)
	}
}

func TestNewRuntimeResolvesRestoredActiveAgent(t *testing.T) {
	temp := float32(0.2)
	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
	})
	root.RegisterTool(tools.ToolDef{
		Name: "root_tool",
		Schema: &schema.Schema{
			Type: "object",
		},
		Handler: func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		},
	})

	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
		MaxTokens:    256,
		Temperature:  float32Ptr(0.8),
		TrimStrategy: TrimSlidingWindow,
		MaxHistory:   20,
		TokenBudget:  4096,
		BudgetConfig: &streaming.BudgetConfig{MaxTokens: 4096},
	})
	billing.RegisterTool(tools.ToolDef{
		Name: "billing_tool",
		Schema: &schema.Schema{
			Type: "object",
		},
		Handler: func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		},
	})

	reg := NewRegistry()
	reg.Register(root)
	reg.Register(billing)

	session := NewSessionFromSnapshot(SessionSnapshot{
		ID: "sess_restore",
		Config: SessionConfig{
			Model:        "stale-root-model",
			SystemPrompt: "stale root system",
			MaxTokens:    64,
			Temperature:  &temp,
			TrimStrategy: TrimSlidingWindow,
			MaxHistory:   5,
			TokenBudget:  1024,
			BudgetConfig: &streaming.BudgetConfig{MaxTokens: 1024},
		},
		Metadata: map[string]string{
			ActiveAgentMetadataKey: "billing",
		},
	})

	rt := NewRuntime(RuntimeConfig{
		Agent:            root,
		SubagentResolver: reg,
		Session:          session,
	})

	if rt.agent == nil || rt.agent.Name() != "billing" {
		t.Fatalf("expected runtime to bind to billing agent, got %#v", rt.agent)
	}

	snap := rt.SessionSnapshot()
	if snap.Config.Model != "billing-model" {
		t.Fatalf("expected session model rebound to billing-model, got %q", snap.Config.Model)
	}
	if snap.Config.SystemPrompt != "billing system" {
		t.Fatalf("expected session system prompt rebound to billing system, got %q", snap.Config.SystemPrompt)
	}
	if snap.Config.MaxTokens != 64 {
		t.Fatalf("expected session max tokens override to be preserved, got %d", snap.Config.MaxTokens)
	}
	if snap.Config.Temperature == nil || *snap.Config.Temperature != temp {
		t.Fatalf("expected session temperature override to be preserved, got %v", snap.Config.Temperature)
	}
	if snap.Config.MaxHistory != 5 {
		t.Fatalf("expected session max history override to be preserved, got %d", snap.Config.MaxHistory)
	}
	if snap.Config.TokenBudget != 1024 {
		t.Fatalf("expected session token budget override to be preserved, got %d", snap.Config.TokenBudget)
	}
	if snap.Config.BudgetConfig == nil || snap.Config.BudgetConfig.MaxTokens != 1024 {
		t.Fatalf("expected session budget override to be preserved, got %#v", snap.Config.BudgetConfig)
	}
	if _, ok := rt.tools.Get("billing_tool"); !ok {
		t.Fatal("expected billing tool in runtime registry")
	}
	if _, ok := rt.tools.Get("root_tool"); ok {
		t.Fatal("did not expect root tool in active billing runtime registry")
	}
}

func TestNewRuntimeKeepsRootAgentWhenActiveMetadataMissing(t *testing.T) {
	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
	})
	reg := NewRegistry()
	reg.Register(root)

	session := NewSession(SessionConfig{})
	rt := NewRuntime(RuntimeConfig{
		Agent:            root,
		SubagentResolver: reg,
		Session:          session,
	})

	if rt.agent == nil || rt.agent.Name() != "root" {
		t.Fatalf("expected runtime to remain bound to root, got %#v", rt.agent)
	}
}
