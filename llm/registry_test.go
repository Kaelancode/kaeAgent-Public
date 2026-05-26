package llm

import (
	"context"
	"testing"
)

type mockProvider struct {
	name string
}

func (m *mockProvider) Complete(_ context.Context, _ *Request) (*Response, error) {
	return &Response{}, nil
}
func (m *mockProvider) Stream(_ context.Context, _ *Request) (<-chan Event, error) {
	return nil, nil
}
func (m *mockProvider) Models(_ context.Context) ([]ModelInfo, error) {
	return nil, nil
}
func (m *mockProvider) Name() string { return m.name }

func TestProviderRegistry_RegisterAndResolve(t *testing.T) {
	r := NewProviderRegistry()
	openai := &mockProvider{name: "openai"}
	claude := &mockProvider{name: "claude"}

	r.Register("gpt-", openai)
	r.Register("claude-", claude)

	p, err := r.Resolve("gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Name() != "openai" {
		t.Errorf("expected openai, got %s", p.Name())
	}

	p, err = r.Resolve("claude-3-opus")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Name() != "claude" {
		t.Errorf("expected claude, got %s", p.Name())
	}
}

func TestProviderRegistry_ResolveLongestPrefix(t *testing.T) {
	r := NewProviderRegistry()
	general := &mockProvider{name: "general"}
	specific := &mockProvider{name: "specific"}

	r.Register("gpt-", general)
	r.Register("gpt-4", specific)

	p, err := r.Resolve("gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Name() != "specific" {
		t.Errorf("expected specific (longest prefix), got %s", p.Name())
	}
}

func TestProviderRegistry_ResolveNotFound(t *testing.T) {
	r := NewProviderRegistry()
	_, err := r.Resolve("unknown-model")
	if err == nil {
		t.Fatal("expected error for unknown model")
	}
}

func TestProviderRegistry_List(t *testing.T) {
	r := NewProviderRegistry()
	r.Register("gpt-", &mockProvider{name: "openai"})
	r.Register("claude-", &mockProvider{name: "claude"})

	list := r.List()
	if len(list) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(list))
	}
	if list["gpt-"] != "openai" {
		t.Errorf("expected openai for gpt-, got %s", list["gpt-"])
	}
}
