package multiagent

import (
	"testing"
)

func TestRouter_RegisterAndRoute(t *testing.T) {
	r := NewRouter()
	r.Register(AgentConfig{
		Name:        "researcher",
		Description: "Research agent",
		Tags:        []string{"research", "web"},
	})
	r.Register(AgentConfig{
		Name:        "coder",
		Description: "Coding agent",
		Tags:        []string{"code", "web"},
	})

	cfg, err := r.Route("research")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Name != "researcher" {
		t.Errorf("expected researcher, got %s", cfg.Name)
	}
}

func TestRouter_RouteNotFound(t *testing.T) {
	r := NewRouter()
	_, err := r.Route("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent tag")
	}
}

func TestRouter_RouteAll(t *testing.T) {
	r := NewRouter()
	r.Register(AgentConfig{Name: "a", Tags: []string{"web"}})
	r.Register(AgentConfig{Name: "b", Tags: []string{"web"}})
	r.Register(AgentConfig{Name: "c", Tags: []string{"code"}})

	web := r.RouteAll("web")
	if len(web) != 2 {
		t.Errorf("expected 2 web agents, got %d", len(web))
	}
}

func TestRouter_Get(t *testing.T) {
	r := NewRouter()
	r.Register(AgentConfig{Name: "test", Description: "test agent"})

	cfg, ok := r.Get("test")
	if !ok {
		t.Fatal("expected to find agent")
	}
	if cfg.Description != "test agent" {
		t.Errorf("expected 'test agent', got %q", cfg.Description)
	}

	_, ok = r.Get("missing")
	if ok {
		t.Error("expected not to find missing agent")
	}
}

func TestRouter_RegisterMaterializesConfigOnlyAgentOnce(t *testing.T) {
	r := NewRouter()
	r.Register(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
		MaxSteps:     4,
		Subagents:    []string{"refunds", "invoices"},
	})

	cfg1, ok := r.Get("billing")
	if !ok {
		t.Fatal("expected billing agent")
	}
	cfg2, ok := r.Get("billing")
	if !ok {
		t.Fatal("expected billing agent on second lookup")
	}

	def1 := cfg1.Definition()
	def2 := cfg2.Definition()
	if def1 == nil {
		t.Fatal("expected config-only registration to materialize an agent")
	}
	if def1 != def2 {
		t.Fatal("expected router to return the same materialized agent instance")
	}
	if def1.Name() != "billing" {
		t.Fatalf("expected materialized agent name billing, got %q", def1.Name())
	}
	if def1.SessionConfig().Model != "billing-model" {
		t.Fatalf("expected materialized agent model billing-model, got %q", def1.SessionConfig().Model)
	}
	if def1.MaxSteps() != 4 {
		t.Fatalf("expected materialized agent max steps 4, got %d", def1.MaxSteps())
	}
	subagents := def1.Subagents()
	if len(subagents) != 2 || subagents[0] != "refunds" || subagents[1] != "invoices" {
		t.Fatalf("expected materialized subagents to be preserved, got %v", subagents)
	}
}

func TestRouter_List(t *testing.T) {
	r := NewRouter()
	r.Register(AgentConfig{Name: "a"})
	r.Register(AgentConfig{Name: "b"})

	names := r.List()
	if len(names) != 2 {
		t.Errorf("expected 2 names, got %d", len(names))
	}
}

func TestRouter_RegisterReplacesStaleTagsAndAvoidsDuplicates(t *testing.T) {
	r := NewRouter()
	r.Register(AgentConfig{Name: "writer", Tags: []string{"draft", "longform"}})
	r.Register(AgentConfig{Name: "writer", Tags: []string{"rewrite"}})

	if got := r.RouteAll("draft"); len(got) != 0 {
		t.Fatalf("expected stale tag to be removed, got %v", got)
	}
	rewrite := r.RouteAll("rewrite")
	if len(rewrite) != 1 || rewrite[0].Name != "writer" {
		t.Fatalf("expected rewrite tag to contain one writer entry, got %v", rewrite)
	}

	r.Register(AgentConfig{Name: "writer", Tags: []string{"rewrite"}})
	rewrite = r.RouteAll("rewrite")
	if len(rewrite) != 1 {
		t.Fatalf("expected no duplicate rewrite entries, got %v", rewrite)
	}
}

func TestRemoveAgentNameDoesNotMutateAliasedSlices(t *testing.T) {
	shared := []string{"writer", "coder"}
	tagA := shared[:]
	tagB := shared[:]

	tagA = removeAgentName(tagA, "writer")

	if len(tagA) != 1 || tagA[0] != "coder" {
		t.Fatalf("expected writer removed from tagA, got %v", tagA)
	}
	if len(tagB) != 2 || tagB[0] != "writer" || tagB[1] != "coder" {
		t.Fatalf("expected aliased tagB to remain unchanged, got %v", tagB)
	}
}
