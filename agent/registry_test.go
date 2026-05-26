package agent

import "testing"

func TestRegistryRegisterGetAndNames(t *testing.T) {
	reg := NewRegistry()
	if reg.Count() != 0 {
		t.Fatalf("expected empty registry, got %d", reg.Count())
	}

	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	refund := NewAgent(AgentConfig{Name: "refund", Model: "refund-model"})

	reg.Register(billing)
	reg.Register(refund)

	if reg.Count() != 2 {
		t.Fatalf("expected 2 agents, got %d", reg.Count())
	}
	if got, ok := reg.Get("billing"); !ok || got != billing {
		t.Fatalf("expected billing agent lookup to succeed, got %v %v", got, ok)
	}

	names := reg.Names()
	if len(names) != 2 {
		t.Fatalf("expected 2 names, got %v", names)
	}

	listed := reg.List()
	if len(listed) != 2 {
		t.Fatalf("expected 2 listed agents, got %d", len(listed))
	}
}

func TestRegistryRegisterOverwritesByName(t *testing.T) {
	reg := NewRegistry()
	first := NewAgent(AgentConfig{Name: "writer", Model: "v1"})
	second := NewAgent(AgentConfig{Name: "writer", Model: "v2"})

	reg.Register(first)
	reg.Register(second)

	got, ok := reg.Get("writer")
	if !ok {
		t.Fatal("expected writer to exist")
	}
	if got != second {
		t.Fatalf("expected second writer definition to overwrite first")
	}
	if reg.Count() != 1 {
		t.Fatalf("expected count 1 after overwrite, got %d", reg.Count())
	}
}

func TestRegistryIgnoresNilAndUnnamedAgents(t *testing.T) {
	reg := NewRegistry()
	reg.Register(nil)
	reg.Register(NewAgent(AgentConfig{}))

	if reg.Count() != 0 {
		t.Fatalf("expected empty registry after ignored registrations, got %d", reg.Count())
	}
}
